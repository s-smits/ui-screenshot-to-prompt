import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from openai import OpenAI
from typing import List, Tuple
from dotenv import load_dotenv
import logging
import base64
from io import BytesIO
from config import build_super_prompt, SYSTEM_PROMPT, VISION_ANALYSIS_PROMPT
import pytesseract
from dataclasses import dataclass
from typing import Optional
import json
import gradio as gr
from concurrent.futures import ThreadPoolExecutor

# Log Point 1: Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('image_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
logger.info("Environment variables loaded") # Log Point 2

# Initialize the OpenAI client with OpenRouter configuration
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    default_headers={
        "HTTP-Referer": os.getenv("YOUR_SITE_URL", "https://your-app-url.com"),  # Replace with your actual URL
        "X-Title": os.getenv("YOUR_SITE_NAME", "Your App Name"),  # Replace with your actual app name
    }
)

@dataclass
class UIElement:
    type: str  # 'button', 'text_field', 'checkbox', 'text'
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    text: Optional[str] = None
    confidence: float = 0.0
    description: Optional[str] = None  # New field for analysis

class SmartImageSplitter:
    def __init__(self, image_path):
        logger.info(f"Initializing SmartImageSplitter with image: {image_path}") 
        self.image_path = image_path
        
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        self.image = cv2.imread(image_path)
        if self.image is None:
            logger.error(f"Failed to load image: {image_path}")
            raise ValueError(f"Failed to load image: {image_path}")
            
        self.output_dir = "split_components"
        self.min_confidence = 0.6
        self.padding = 10
        
        if not os.path.exists(self.output_dir):
            logger.info(f"Creating output directory: {self.output_dir}")
            os.makedirs(self.output_dir)

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better element detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray)
        # Adaptive threshold
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        return thresh

    def detect_components(self, min_area=500, max_components=20) -> List[UIElement]:
        """Detect UI components in the image"""
        logger.info("Detecting components")
        
        # Get text regions using OCR
        text_elements = self.detect_text_regions(self.image)
        
        # Get UI elements using contour detection
        ui_elements = self.detect_ui_elements(self.image)
        
        # Combine all elements
        all_elements = text_elements + ui_elements
        
        # Filter by area
        all_elements = [elem for elem in all_elements if elem.bbox[2] * elem.bbox[3] >= min_area]
        
        # Sort by importance and limit number
        all_elements.sort(key=self._get_component_importance, reverse=True)
        all_elements = all_elements[:max_components]
        
        # Merge overlapping components
        merged_elements = self._merge_overlapping_components_ui_elements(all_elements)
        
        logger.info(f"Final component count: {len(merged_elements)}")
        return merged_elements

    def detect_text_regions(self, image: np.ndarray) -> List[UIElement]:
        """Detect text regions using Tesseract OCR"""
        text_elements = []
        
        # Get OCR data including bounding boxes
        ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        
        n_boxes = len(ocr_data['text'])
        for i in range(n_boxes):
            # Filter empty results and low confidence
            if float(ocr_data['conf'][i]) < self.min_confidence:
                continue
            if not ocr_data['text'][i].strip():
                continue
                
            x = ocr_data['left'][i]
            y = ocr_data['top'][i]
            w = ocr_data['width'][i]
            h = ocr_data['height'][i]
            
            text_elements.append(UIElement(
                type='text',
                bbox=(x, y, w, h),
                text=ocr_data['text'][i],
                confidence=float(ocr_data['conf'][i])
            ))
            
        return text_elements

    def detect_ui_elements(self, image: np.ndarray) -> List[UIElement]:
        """Detect UI elements using contour detection"""
        ui_elements = []
        
        # Preprocess
        processed = self.preprocess_image(image)
        
        # Find contours
        contours, _ = cv2.findContours(
            processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        for contour in contours:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Skip very small elements
            if w < 20 or h < 20:
                continue
                
            # Analyze shape for classification
            aspect_ratio = float(w) / h
            area = cv2.contourArea(contour)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0
            
            # Classify based on shape characteristics
            if 0.9 <= aspect_ratio <= 1.1 and solidity > 0.9:
                element_type = 'checkbox'
            elif 2.5 <= aspect_ratio <= 5 and solidity > 0.8:
                element_type = 'text_field'
            elif 1.5 <= aspect_ratio <= 3 and solidity > 0.85:
                element_type = 'button'
            else:
                continue
                
            ui_elements.append(UIElement(
                type=element_type,
                bbox=(x, y, w, h),
                confidence=solidity
            ))
            
        return ui_elements

    def _get_component_importance(self, element: UIElement) -> float:
        """Calculate component importance based on position and content"""
        x, y, w, h = element.bbox
        
        # Factors that increase importance:
        # 1. Larger area
        area_score = w * h
        
        # 2. Central position (prefer components closer to center)
        center_x = x + w / 2
        center_y = y + h / 2
        img_center_x = self.image.shape[1] / 2
        img_center_y = self.image.shape[0] / 2
        
        distance_from_center = np.sqrt(
            (center_x - img_center_x) ** 2 + 
            (center_y - img_center_y) ** 2
        )
        position_score = 1 / (1 + distance_from_center)
        
        # 3. Aspect ratio closer to common UI element ratios
        aspect_ratio = w / h if h != 0 else 0
        ratio_score = min(aspect_ratio, 1 / aspect_ratio) if aspect_ratio > 0 else 0
        
        # Combine scores with weights
        total_score = (
            0.5 * area_score +
            0.3 * position_score +
            0.2 * ratio_score
        )
        
        return total_score

    def _merge_overlapping_components_ui_elements(self, elements: List[UIElement], overlap_threshold=0.3) -> List[UIElement]:
        """Merge UIElements that overlap significantly"""
        if not elements:
            logger.warning("No components to merge")
            return []
            
        merged_elements = []
        used = set()
        
        for i, elem1 in enumerate(elements):
            if i in used:
                continue
                
            current_bbox = list(elem1.bbox)
            current_type = elem1.type
            used.add(i)
            
            for j, elem2 in enumerate(elements[i+1:], i+1):
                if j in used:
                    continue
                    
                # Calculate overlap
                x1, y1, w1, h1 = current_bbox
                x2, y2, w2, h2 = elem2.bbox
                
                # Find intersection
                x_left = max(x1, x2)
                y_top = max(y1, y2)
                x_right = min(x1 + w1, x2 + w2)
                y_bottom = min(y1 + h1, y2 + h2)
                
                if x_right > x_left and y_bottom > y_top:
                    intersection = (x_right - x_left) * (y_bottom - y_top)
                    area1 = w1 * h1
                    area2 = w2 * h2
                    overlap_ratio = intersection / min(area1, area2)
                    
                    if overlap_ratio > overlap_threshold:
                        # Merge components
                        current_bbox[0] = min(x1, x2)
                        current_bbox[1] = min(y1, y2)
                        current_bbox[2] = max(x1 + w1, x2 + w2) - current_bbox[0]
                        current_bbox[3] = max(y1 + h1, y2 + h2) - current_bbox[1]
                        used.add(j)
        
            merged_elements.append(UIElement(
                type=current_type,
                bbox=tuple(current_bbox),
                confidence=min(elem1.confidence, elem2.confidence) if 'elem2' in locals() else elem1.confidence
            ))
        
        logger.info(f"Merged into {len(merged_elements)} components")
        return merged_elements

    def save_components(self, components: List[UIElement]):
        """Save detected components as separate images with proper cropping"""
        logger.info(f"Saving {len(components)} components...")
        
        # Create subdirectories for different component types
        for component_type in ['text', 'button', 'text_field', 'checkbox']:
            type_dir = os.path.join(self.output_dir, component_type)
            os.makedirs(type_dir, exist_ok=True)
        
        saved_components = []
        for i, component in enumerate(tqdm(components)):
            try:
                x, y, w, h = component.bbox
                
                # Add padding around the component (if specified)
                x1 = max(0, x - self.padding)
                y1 = max(0, y - self.padding)
                x2 = min(self.image.shape[1], x + w + self.padding)
                y2 = min(self.image.shape[0], y + h + self.padding)
                
                # Crop the component
                component_img = self.image[y1:y2, x1:x2]
                
                if component_img.size == 0:
                    logger.warning(f"Skipping empty component {i}")
                    continue
                
                # Create filename with component info
                filename = f"component_{i:03d}"
                if component.text:
                    # Clean text for filename
                    clean_text = "".join(c for c in component.text if c.isalnum() or c in (' ', '-', '_'))
                    clean_text = clean_text[:30]  # Limit length
                    filename += f"_{clean_text}"
                
                # Save to type-specific subdirectory
                output_path = os.path.join(self.output_dir, component.type, f"{filename}.png")
                
                # Save the image
                cv2.imwrite(output_path, component_img)
                
                # Store component info
                saved_components.append({
                    'type': component.type,
                    'bbox': (x1, y1, x2-x1, y2-y1),
                    'text': component.text,
                    'confidence': component.confidence,
                    'file': output_path
                })
                
                logger.debug(f"Saved {component.type} component {i} to {output_path}")
                
            except Exception as e:
                logger.error(f"Error saving component {i}: {str(e)}")
                continue
        
        # Save component metadata
        metadata_path = os.path.join(self.output_dir, 'components_metadata.json')
        try:
            with open(metadata_path, 'w') as f:
                json.dump(saved_components, f, indent=2)
            logger.info(f"Saved component metadata to {metadata_path}")
        except Exception as e:
            logger.error(f"Error saving metadata: {str(e)}")
        
        return saved_components

    def visualize_components(self, components):
        """Visualize detected components on the original image"""
        logger.info("Visualizing components")
        viz_image = self.image.copy()
        logger.info("Drawing component boundaries...")
        
        # Different colors for different types
        colors = {
            'text': (0, 255, 0),      # Green
            'button': (255, 0, 0),    # Blue
            'text_field': (0, 0, 255),# Red
            'checkbox': (255, 255, 0)  # Cyan
        }
        
        for component in tqdm(components):
            # Handle both UIElement objects and tuples
            if isinstance(component, UIElement):
                x, y, w, h = component.bbox
                component_type = component.type
            else:
                # If it's a tuple, unpack directly
                x, y, w, h = component
                component_type = 'unknown'  # Default type for raw bounding boxes
                
            color = colors.get(component_type, (0, 255, 0))
            cv2.rectangle(viz_image, (x, y), (x+w, y+h), color, 2)
            
            # Add label if type is known
            if component_type != 'unknown':
                cv2.putText(viz_image, component_type, (x, y-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        output_path = "detected_components.png"
        cv2.imwrite(output_path, viz_image)
        logger.info(f"Saved visualization to {output_path}")

    def analyze_components(self, components: List[UIElement]) -> List[str]:
        """Analyze detected UI components"""
        analyses = []
        for idx, component in enumerate(components):
            x, y, w, h = component.bbox
            
            analysis = (
                f"Component {idx}: {component.type}"
                f" at position ({x}, {y})"
                f" with size {w}x{h}"
            )
            if component.text:
                analysis += f" containing text: {component.text}"
                
            analyses.append(analysis)
            logger.info(analysis)
        
        return analyses

def analyze_main_design_choices(image: Image.Image, temp: float = 0.1) -> str:
    """Analyze the main flow/purpose of the entire image"""
    logger.info("Analyzing main design choices")
    
    try:
        # Convert PIL Image to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Normally gpt-4o but for testing
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analyze the main design patterns and purpose of this interface, focus on every component of the UI and what you see, which is needed later for an AI coder to reproduce the design."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_str}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000,
            temperature=temp
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"OpenAI API error: {str(e)}")
        return f"Error analyzing design: {str(e)}"

def analyze_component(args):
    """Analyze individual component of the image"""
    component_image, main_design_choices, component_index = args
    logger.info(f"Analyzing component {component_index}")
    
    try:
        # Convert PIL Image to base64
        buffered = BytesIO()
        component_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Updated model name
            messages=[
                {"role": "system", "content": VISION_ANALYSIS_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Context: {main_design_choices}\n\nAnalyze component {component_index}:"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_str}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000,
            temperature=0.1
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        logger.error(f"API error analyzing component {component_index}: {str(e)}")
        return f"Error analyzing component {component_index}: {str(e)}"

def process_image(image_path: str, min_area: int = 1000, max_components: int = 10):
    """Main function to process and analyze an image"""
    logger.info(f"Starting image processing for {image_path}")
    try:
        # Initialize splitter
        splitter = SmartImageSplitter(image_path)
        
        # Detect components
        components = splitter.detect_components(
            min_area=min_area,
            max_components=max_components
        )
        
        # Analyze main image first
        main_image = Image.open(image_path)
        main_design_choices = analyze_main_design_choices(main_image)
        
        # Get activity description
        activity_description = describe_activity(main_image)
        
        # Prepare component analysis arguments
        analysis_args = []
        for i, component in enumerate(components):
            x, y, w, h = component.bbox
            component_img = splitter.image[y:y+h, x:x+w]
            component_rgb = cv2.cvtColor(component_img, cv2.COLOR_BGR2RGB)
            component_pil = Image.fromarray(component_rgb)
            analysis_args.append((component_pil, main_design_choices, i))
            
            # Save the component
            output_path = os.path.join(splitter.output_dir, f"component_{i}.png")
            cv2.imwrite(output_path, component_img)
        
        # Process components in parallel
        with ThreadPoolExecutor(max_workers=5) as executor:
            component_analyses = list(executor.map(analyze_component, analysis_args))
        
        # Link analyses to components
        for component, analysis in zip(components, component_analyses):
            component.description = analysis  # Storing the analysis in the UIElement
        
        # Extract descriptions for the super prompt
        descriptions = [component.description for component in components]
        
        # Build and call super prompt
        final_analysis = call_super_prompt(
            main_design_choices,
            descriptions,
            activity_description
        )
        
        logger.info("Final analysis: %s", final_analysis)
        # Visualize all components
        splitter.visualize_components(components)
        
        logger.info("Image processing completed successfully")
        return main_design_choices, descriptions, final_analysis
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}", exc_info=True)
        return "Error processing image", [], "Error in final analysis"

def gradio_process_image(image):
    """Process image uploaded through Gradio interface"""
    logger.info("Processing image uploaded through Gradio")
    
    # Save the uploaded image temporarily
    temp_image_path = "temp_uploaded_image.png"
    image.save(temp_image_path)
    
    # Process the image
    main_design_choices, analyses, final_analysis = process_image(temp_image_path)
    
    # Remove temporary image
    os.remove(temp_image_path)
    
    # Prepare output
    output = f"Main Design Choices:\n{main_design_choices}\n\n"
    output += "Component Analyses:\n"
    for i, analysis in enumerate(analyses):
        output += f"Component {i}: {analysis}\n"
    output += f"\nFinal Analysis:\n{final_analysis}"
    
    return output

def launch_gradio_interface():
    """Launch Gradio interface"""
    iface = gr.Interface(
        fn=gradio_process_image,
        inputs=gr.Image(type="pil"),
        outputs="text",
        title="Bolt.new Prompt Generator",
        description="Upload an image of a UI to generate a prompt for an AI coder to reproduce the design."
    )
    iface.launch()

def main():
    # Process single image
    logger.info("Starting image processing")
    image_path = os.path.join("image", "image.png")
    logger.info(f"\nProcessing image...")
    main_design_choices, analyses, final_analysis = process_image(image_path)
    
    if analyses:
        logger.info(f"Main design choices: {main_design_choices}")
        logger.info("\nComponent analyses:")
        for i, analysis in enumerate(analyses):
            logger.info(f"Component {i}: {analysis}")
        logger.info("\nFinal Analysis:")
        logger.info(final_analysis)
    else:
        logger.error("Failed to process image")

def describe_activity(image: Image.Image):
    logger.info("Describing activity in image")
    
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Updated model name
            messages=[
                {"role": "system", "content": "Describe the activity in a few sentences."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What activity is shown in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_str}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            temperature=0.1,
            max_tokens=50
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"OpenAI API error: {str(e)}")
        return "Error describing activity"

def call_super_prompt(main_image_caption: str, component_captions: List[str], activity_description: str) -> str:
    """Build and send the super prompt integrating all analyses"""
    super_prompt = build_super_prompt(main_image_caption, component_captions, activity_description)
    
    MODEL_ID = "anthropic/claude-3.5-sonnet"
    
    logger.info(f"Sending request to model: {MODEL_ID}")
    
    try:
        completion = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {
                    "role": "user",
                    "content": super_prompt
                }
            ]
        )
        
        logger.info("Super prompt successfully processed.")
        return completion.choices[0].message.content
    
    except client.APIError as e:
        logger.error(f"OpenAI API error: {str(e)}")
        raise Exception(f"OpenAI API error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise Exception(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    launch_gradio_interface()
