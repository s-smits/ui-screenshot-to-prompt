from PIL import Image
import cv2
import numpy as np
import os
import json
from typing import List, Tuple, Union, Dict
from dataclasses import dataclass
from logging import getLogger
from tqdm import tqdm
from config import MIN_COMPONENT_WIDTH_SIMPLE, MIN_COMPONENT_HEIGHT_SIMPLE, MIN_COMPONENT_WIDTH_ADVANCED, MIN_COMPONENT_HEIGHT_ADVANCED

logger = getLogger(__name__)

@dataclass
class UIComponent:  # Renamed from UIElement for clarity and consistency
    """Data class representing a detected UI component with its properties"""
    type: str
    bbox: Tuple[int, int, int, int]
    confidence: float = 1.0
    text: str = ""

class ComponentDetectorBase:  # Renamed from BaseComponentHandler
    """Base class implementing core UI component detection functionality"""
    
    @staticmethod
    def create_component(bbox: Tuple[int, int, int, int], 
                        component_type: str = "unknown",
                        location: str = "") -> UIComponent:
        """Creates a new UI component instance with specified parameters"""
        return UIComponent(
            type=component_type,
            bbox=bbox,
            confidence=1.0,
            text=location
        )

    @staticmethod
    def visualize_components(image: Union[np.ndarray, Image.Image], 
                           components: List[UIComponent],
                           output_path: str,
                           pre_merge: bool = False):
        """Unified component visualization"""
        try:
            # Convert image to OpenCV format if needed
            if isinstance(image, Image.Image):
                image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            elif len(image.shape) == 2:  # Grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
            viz_image = image.copy()
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Draw components
            for component in components:
                x, y, w, h = component.bbox
                
                # Use different colors based on component type
                color = {
                    'checkbox': (0, 255, 0),
                    'text_field': (255, 0, 0),
                    'button': (0, 0, 255),
                    'grid_section': (0, 255, 0),
                    'unknown': (255, 255, 0)
                }.get(component.type, (0, 255, 0))
                
                # Draw rectangle
                cv2.rectangle(viz_image, (x, y), (x + w, y + h), color, 2)
                
                # Add label
                label = component.text if component.text else component.type
                cv2.putText(viz_image, label[:30], (x, max(y - 5, 10)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Save visualization
            cv2.imwrite(output_path, viz_image)
            logger.info(f"Successfully saved visualization to {output_path}")
            
        except Exception as e:
            logger.error(f"Error in visualization: {str(e)}", exc_info=True)
            raise

    @staticmethod
    def analyze_components(components: List[UIComponent]) -> List[str]:
        """Unified component analysis"""
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

class BasicComponentDetector(ComponentDetectorBase):
    """Implements basic grid-based component detection strategy"""
    def __init__(self, image_path: str):
        logger.info(f"Initializing BasicComponentDetector with image: {image_path}")
        self.image_path = image_path
        
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        self.image = Image.open(image_path)
        self.width, self.height = self.image.size
        self.output_dir = "split_components"
        
        if not os.path.exists(self.output_dir):
            logger.info(f"Creating output directory: {self.output_dir}")
            os.makedirs(self.output_dir)

    def determine_grid_size(self) -> Tuple[int, int]:
        """Determine optimal grid size based on image dimensions"""
        grid_size, _ = self.get_grid_pattern()
        return grid_size

    def get_grid_pattern(self) -> Tuple[Tuple[int, int], List[str]]:
        """
        Determine grid pattern and return grid size and location names
        Returns tuple of ((rows, cols), location_names)
        """
        aspect_ratio = self.width / self.height if self.height != 0 else 1

        # Each pattern now returns a tuple of grid size and explicit location names
        if self.width < MIN_COMPONENT_WIDTH_SIMPLE and self.height < MIN_COMPONENT_HEIGHT_SIMPLE:
            return (1, 1), ['full']
        elif aspect_ratio >= 3:
            return (1, 3), ['left side', 'center portion', 'right side']  # More descriptive locations
        elif aspect_ratio <= 0.33:
            return (3, 1), ['upper portion', 'middle portion', 'lower portion']
        elif aspect_ratio >= 2:
            return (2, 3), [
                'upper left section', 'upper center section', 'upper right section',
                'lower left section', 'lower center section', 'lower right section'
            ]
        elif aspect_ratio <= 0.5:
            return (3, 2), [
                'top left area', 'top right area',
                'middle left area', 'middle right area',
                'bottom left area', 'bottom right area'
            ]
        elif self.width >= 600 and self.height >= 600:
            return (3, 3), [
                'top left region', 'top center region', 'top right region',
                'middle left region', 'center region', 'middle right region',
                'bottom left region', 'bottom center region', 'bottom right region'
            ]
        elif self.width >= 400 and self.height >= 400:
            return (2, 2), [
                'top left quadrant', 'top right quadrant',
                'bottom left quadrant', 'bottom right quadrant'
            ]
        else:
            if self.width > self.height:
                return (1, 2), ['left half', 'right half']
            else:
                return (2, 1), ['upper half', 'lower half']

    def get_grid_components(self) -> List[UIComponent]:
        """Get components using NumPy for efficient grid division"""
        grid_size, location_names = self.get_grid_pattern()
        rows, cols = grid_size
        
        h_splits = np.array_split(range(self.height), rows)
        w_splits = np.array_split(range(self.width), cols)
        
        components = []
        location_idx = 0

        for i, h_split in enumerate(h_splits):
            for j, w_split in enumerate(w_splits):
                location = location_names[location_idx] if location_idx < len(location_names) else f"section_{location_idx}"
                location_idx += 1
                
                x = w_split[0]
                y = h_split[0]
                w = w_split[-1] - w_split[0] + 1
                h = h_split[-1] - h_split[0] + 1
                
                component = self.create_component(
                    bbox=(x, y, w, h),
                    component_type="grid_section",  # Generic type
                    location=location  # Pass location as separate parameter
                )
                components.append(component)
                logger.info(f"Created component for {location} at ({x}, {y}) with size {w}x{h}")

        return components

    def get_composition_info(self) -> Dict:
        """Return detailed composition information"""
        grid_size, _ = self.get_grid_pattern()
        components = self.get_grid_components()
        
        return {
            "resolution": f"{self.width}x{self.height}",
            "grid_size": f"{grid_size[0]}x{grid_size[1]}",
            "aspect_ratio": round(self.width / self.height, 2),
            "components": [
                {
                    "type": component.type,
                    "location": component.text,  # Include location information
                    "position": f"({component.bbox[0]}, {component.bbox[1]})",
                    "size": f"{component.bbox[2]}x{component.bbox[3]}"
                }
                for component in components
            ]
        }

    def get_components(self) -> List[UIComponent]:
        """Convert split regions to UIElements for compatibility"""
        return self.get_grid_components()

class AdvancedComponentDetector(ComponentDetectorBase):
    """Implements advanced contour-based component detection with morphological operations"""
    def __init__(self, image_path: str, min_width: int = MIN_COMPONENT_WIDTH_ADVANCED, min_height: int = MIN_COMPONENT_HEIGHT_ADVANCED):
        logger.info(f"Initializing AdvancedComponentDetector with image: {image_path}")
        self.image_path = image_path
        
        # Component size thresholds
        self.min_width = min_width
        self.min_height = min_height
        self.MIN_COMPONENT_AREA = self.min_width * self.min_height
        
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        self.image = cv2.imread(image_path)
        if self.image is None:
            logger.error(f"Failed to load image: {image_path}")
            raise ValueError(f"Failed to load image: {image_path}")
            
        self.output_dir = "split_components"
        self.padding = 5
        
        if not os.path.exists(self.output_dir):
            logger.info(f"Creating output directory: {self.output_dir}")
            os.makedirs(self.output_dir)

        # Confidence threshold
        self.confidence_threshold = 0.3  # Initial confidence threshold 

    def preprocess_image_with_kernel(self, image: np.ndarray, kernel_size: Tuple[int, int]) -> np.ndarray:
        """Preprocess image with a specific kernel size for morphological operations"""
        logger.info(f"Preprocessing image with kernel size: {kernel_size}")
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply Gaussian Blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        # Morphological operations to close gaps with specified kernel size
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        return closed

    def preprocess_image_advanced(self, image: np.ndarray) -> np.ndarray:
        """Advanced preprocessing with multiple morphological operations"""
        logger.info("Advanced preprocessing of the image")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply median blur
        blurred = cv2.medianBlur(gray, 5)
        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
            cv2.THRESH_BINARY_INV, 15, 4
        )
        # Dilate to merge adjacent elements
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        dilated = cv2.dilate(thresh, kernel, iterations=1)
        return dilated

    def detect_components(self, max_components: int = 40) -> List[UIComponent]:
        """Detect UI components in the image"""
        logger.info("Detecting components")
        
        attempts = 0
        max_attempts = 5
        kernel_sizes = [(5,5), (7,7), (3,3), (9,9), (11,11)]  # Different kernel sizes for retries

        while attempts < max_attempts:
            logger.info(f"Attempt {attempts + 1}")

            # Preprocess image with current kernel size
            kernel_size = kernel_sizes[attempts % len(kernel_sizes)]
            preprocessed = self.preprocess_image_with_kernel(self.image, kernel_size)

            # Detect UI elements using contour detection
            ui_elements = self.detect_ui_elements(preprocessed)

            logger.info(f"Detected {len(ui_elements)} UI components before filtering")

            # Filter out components smaller than MIN_COMPONENT_AREA
            filtered_elements = [
                elem for elem in ui_elements 
                if (elem.bbox[2] * elem.bbox[3] >= self.MIN_COMPONENT_AREA) and (elem.confidence >= self.confidence_threshold)
            ]
            logger.info(f"{len(filtered_elements)} UI components after area and confidence filtering")

            num_components = len(filtered_elements)

            if num_components > 10:
                logger.info("Number of components > 10, proceeding to merge overlapping components.")
                merged_elements = self._merge_overlapping_components_ui_elements(filtered_elements, overlap_threshold=0.2)
                logger.info(f"Merged into {len(merged_elements)} UI components")
                return merged_elements
            elif 3 <= num_components <= 10:
                logger.info(f"Detected {num_components} components which is within the acceptable range.")
                return filtered_elements
            else:
                logger.warning(f"Detected only {num_components} components, which is below the threshold. Retrying detection with adjusted parameters.")
                # Adjust parameters for the next attempt
                self.confidence_threshold *= 0.9  # Decrease confidence threshold by 10%
                self.confidence_threshold = max(self.confidence_threshold, 0.1)  # Set a lower bound for confidence
                attempts += 1

        # After max attempts, return the current filtered elements if within 3-10, else log error
        if 3 <= num_components <= 10:
            logger.info(f"Final number of components after {attempts} attempts: {num_components}")
            return filtered_elements
        else:
            logger.error(f"Unable to detect a sufficient number of components ({num_components}) after {max_attempts} attempts.")
            return []

    def detect_ui_elements(self, preprocessed_image: np.ndarray) -> List[UIComponent]:
        """Detect UI elements using contour detection"""
        logger.info("Detecting UI elements using contours")
        ui_elements = []
        
        # Find contours with hierarchy
        contours, hierarchy = cv2.findContours(
            preprocessed_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if hierarchy is None:
            logger.warning("No contours found")
            return ui_elements

        for idx, contour in enumerate(contours):
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Skip components smaller than MIN_COMPONENT_AREA
            if w * h < self.MIN_COMPONENT_AREA:
                continue
                
            # Analyze shape for classification
            aspect_ratio = float(w) / h if h != 0 else 1
            area = cv2.contourArea(contour)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0
            
            # Improved classification based on aspect ratios and solidity
            if 0.8 <= aspect_ratio <= 1.2 and solidity > 0.9:
                element_type = 'checkbox'
            elif 2.5 <= aspect_ratio <= 6 and solidity > 0.8:
                element_type = 'text_field'
            elif 1.0 <= aspect_ratio <= 4.0 and solidity > 0.85:
                element_type = 'button'
            else:
                element_type = 'unknown'  # Catch-all for other UI elements
                
            ui_elements.append(UIComponent(
                type=element_type,
                bbox=(x, y, w, h),
                confidence=solidity
            ))
        
        logger.info(f"Detected {len(ui_elements)} UI elements via contours")
        return ui_elements

    def _merge_overlapping_components_ui_elements(self, elements: List[UIComponent], overlap_threshold: float = 0.2) -> List[UIComponent]:
        """Merge overlapping UI components based on the overlap threshold"""
        logger.info(f"Merging overlapping components with overlap_threshold={overlap_threshold}")
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
                overlap_ratio = self._calculate_overlap(current_bbox, elem2.bbox)
                
                if overlap_ratio > overlap_threshold and elem1.type == elem2.type:
                    # Merge components
                    current_bbox[0] = min(current_bbox[0], elem2.bbox[0])
                    current_bbox[1] = min(current_bbox[1], elem2.bbox[1])
                    current_bbox[2] = max(current_bbox[0] + current_bbox[2], elem2.bbox[0] + elem2.bbox[2]) - current_bbox[0]
                    current_bbox[3] = max(current_bbox[1] + current_bbox[3], elem2.bbox[1] + elem2.bbox[3]) - current_bbox[1]
                    used.add(j)
            
            merged_elements.append(UIComponent(
                type=current_type,
                bbox=tuple(current_bbox),
                confidence=elem1.confidence
            ))
        
        logger.info(f"Merged into {len(merged_elements)} UI components")
        return merged_elements

    def _calculate_overlap(self, bbox1: List[int], bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate the overlap ratio between two bounding boxes"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Determine the coordinates of the intersection rectangle
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0  # No overlap
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        min_area = min(w1 * h1, w2 * h2)
        
        return intersection_area / min_area if min_area > 0 else 0.0

    def save_components(self, components: List[UIComponent]):
        """Save detected components as separate images with proper cropping"""
        logger.info(f"Saving {len(components)} components...")
        
        # Create subdirectories for different component types
        for component_type in ['button', 'text_field', 'checkbox', 'unknown']:
            type_dir = os.path.join(self.output_dir, component_type)
            os.makedirs(type_dir, exist_ok=True)
        
        saved_components = []
        for i, component in enumerate(tqdm(components)):
            try:
                x, y, w, h = component.bbox
                
                # Skip components smaller than MIN_COMPONENT_AREA
                if w * h < self.MIN_COMPONENT_AREA:
                    continue
                
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
                    'bbox': (x1, y1, x2 - x1, y2 - y1),
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

    def get_components(self) -> List[UIComponent]:
        """Get components using smart detection"""
        logger.info("Getting components using smart detection")
        return self.detect_components()

    def update_min_dimensions(self, min_width: int, min_height: int):
        """Update minimum component dimensions"""
        self.min_width = min_width
        self.min_height = min_height
        self.MIN_COMPONENT_AREA = self.min_width * self.min_height
        logger.info(f"Updated minimum dimensions to: width={min_width}, height={min_height}")

def create_component_detector(detection_mode: str, image_path: str) -> ComponentDetectorBase:
    """
    Factory function to create appropriate component detector instance
    
    Args:
        detection_mode: Strategy to use ('basic' or 'advanced')
        image_path: Path to input image
        
    Returns:
        ComponentDetectorBase: Configured detector instance
        
    Raises:
        ValueError: If detection_mode is invalid
    """
    if detection_mode == 'advanced':
        return AdvancedComponentDetector(image_path)
    elif detection_mode == 'basic':  # Changed from 'easy' to 'basic'
        return BasicComponentDetector(image_path)
    else:
        raise ValueError(f"Invalid detection mode: {detection_mode}")
