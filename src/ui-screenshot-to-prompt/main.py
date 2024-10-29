import os
import cv2
import numpy as np
from PIL import Image
from typing import List, Optional
import logging
import base64
from io import BytesIO
import gradio as gr
from concurrent.futures import ThreadPoolExecutor


# Configuration imports consolidated
from config import (
    build_super_prompt,
    VISION_ANALYSIS_PROMPT,
    MAIN_DESIGN_ANALYSIS_PROMPT,
    load_and_initialize_clients,
    set_prompt_choice,
    MIN_COMPONENT_WIDTH_ADVANCED,
    MIN_COMPONENT_HEIGHT_ADVANCED,
    MAX_UI_COMPONENTS,
)

from detect_components import create_detector  # Only import what we use

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set to INFO for general logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('image_processing.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Disable debug logging for openai to reduce noise
logging.getLogger("openai").setLevel(logging.WARNING)

# Initialize OpenAI clients
openai_client, super_prompt_function = load_and_initialize_clients()

# Global variables for UI-selected settings
DETECTION_METHOD = None
DETECTION_TERM = None

def set_detection_method(mode: str):
    """Configure detection method and terminology"""
    global DETECTION_METHOD, DETECTION_TERM
    DETECTION_METHOD = mode.lower()
    DETECTION_TERM = "region" if DETECTION_METHOD == "basic" else "component"
    logger.info(f"Detection method set to: {DETECTION_METHOD} ({DETECTION_TERM}s)")

def get_detection_method() -> str:
    """Get current detection method"""
    return DETECTION_METHOD

def encode_image_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string"""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def call_vision_api(model: str, image: Image.Image, system_prompt: str, user_prompt: str, 
                   temperature: float = 0.1, json_response: bool = True) -> str:
    """Unified function for calling OpenAI Vision API"""
    try:
        img_str = encode_image_base64(image)
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_str}",
                            "detail": "high"
                        }
                    }
                ]
            }
        ]
        
        response = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=1024,
            temperature=temperature,
            response_format={"type": "json_object"} if json_response else None
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        logger.error(f"OpenAI API error: {str(e)}")
        raise

def analyze_main_design_choices(image: Image.Image, temp: float = 0.1) -> str:
    """Analyze the main flow/purpose of the entire image, returns main_image_caption"""
    logger.info("Analyzing main design choices")
    
    try:
        return call_vision_api(
            model="gpt-4o",
            image=image, 
            system_prompt=MAIN_DESIGN_ANALYSIS_PROMPT, 
            user_prompt="Analyze this interface's complete design system and structure.", 
            temperature=temp,
            json_response=False
        )
    except Exception as e:
        logger.error(f"Error analyzing main design: {str(e)}")
        return "Error analyzing main design structure"

def describe_activity(image: Image.Image) -> str:
    """Describe the activity shown in the image"""
    logger.info("Describing activity in image")
    print(" Used image: ", image.filename if hasattr(image, 'filename') else 'Image without filename')
    
    return call_vision_api(
        model="gpt-4o",
        image=image,
        system_prompt="Describe the activity of this webpage in a few sentences.",
        user_prompt="What activity is shown in this image?",
        temperature=0.1,
        json_response=False
    )

def analyze_detection(args):
    """Analyze individual detection (region/component) of the image"""
    detection_image, main_design_choices, index, location = args
    logger.info(f"Analyzing {DETECTION_TERM} {index} in {location}")
    
    prompt = f"""Analyze this UI {DETECTION_TERM}:
    - Located in: {location}
    - {DETECTION_TERM.title()} number: {index}
    
    Provide structured analysis following the JSON schema in the system prompt.
    Focus on implementation-relevant details."""
    
    analysis = call_vision_api(model="gpt-4o-mini", image=detection_image, system_prompt=VISION_ANALYSIS_PROMPT, user_prompt=prompt)
    return f"[Location: {location}]\n{analysis}"

def process_image(image_path: str, min_area: Optional[float] = None, max_detections: int = MAX_UI_COMPONENTS):
    """Main function to process and analyze an image"""
    try:
        output_dir = "split_detections"
        os.makedirs(output_dir, exist_ok=True)
        
        # Pass max_detections to create_detector
        detector = create_detector(
            get_detection_method(), 
            image_path, 
            max_components=max_detections
        )
        
        # Get detections
        detections = detector.get_components()[:max_detections]  # Limit detections here too
        
        if not detections:
            logger.error(f"No {DETECTION_TERM}s detected. Exiting processing.")
            return f"No {DETECTION_TERM}s detected.", [], "Error in final analysis"
        
        # Analyze main image first
        main_image = Image.open(image_path)
        main_design_choices = analyze_main_design_choices(main_image)
        activity_description = describe_activity(main_image)
        
        # Prepare detection analysis arguments
        analysis_args = []
        for i, detection in enumerate(detections):
            x, y, w, h = detection.bbox
            # Convert cropped component back to RGB for VLM
            detection_img = cv2.cvtColor(np.array(detector.image), cv2.COLOR_RGB2BGR)[y:y+h, x:x+w]
            detection_rgb = cv2.cvtColor(detection_img, cv2.COLOR_BGR2RGB)
            detection_pil = Image.fromarray(detection_rgb)
            
            analysis_args.append((detection_pil, main_design_choices, i, detection.text))
            
            # Save the detection
            output_path = os.path.join(output_dir, f"{DETECTION_TERM}_{i}.png")
            cv2.imwrite(output_path, detection_img)
        
        # Process detections in parallel
        with ThreadPoolExecutor(max_workers=5) as executor:
            detection_analyses = list(executor.map(analyze_detection, analysis_args))
        
        # Link analyses to detections
        descriptions = []
        for detection, analysis in zip(detections, detection_analyses):
            location_info = f"[{detection.text}] "
            full_analysis = location_info + analysis
            detection.text = full_analysis
            descriptions.append(full_analysis)
        
        # Build and call super prompt
        final_analysis = call_super_prompt(
            main_design_choices,
            descriptions,
            activity_description
        )
        
        # Visualize all detections
        detector.visualize_detections(
            cv2.cvtColor(np.array(detector.image), cv2.COLOR_RGB2BGR),
            detections,
            os.path.join(output_dir, "visualization.png")
        )
        
        logger.info("Image processing completed successfully")
        return main_design_choices, descriptions, final_analysis
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}", exc_info=True)
        return "Error processing image", [], "Error in final analysis"

def gradio_process_image(image, splitting_mode, max_components=MAX_UI_COMPONENTS):
    """Process image uploaded through Gradio interface"""
    logger.info(f"Processing image uploaded through Gradio with splitting mode: {splitting_mode}")
    
    # Save the uploaded image temporarily
    temp_image_path = "temp_uploaded_image.png"
    image.save(temp_image_path)
    
    # Set detection method from UI selection
    set_detection_method(splitting_mode)
    
    # Process the image with max_components
    main_design_choices, analyses, final_analysis = process_image(
        temp_image_path,
        max_detections=max_components
    )
    
    # Get visualization image if in advanced mode
    visualization_image = None
    if DETECTION_METHOD == "advanced":
        viz_path = os.path.join("split_components", "visualization.png")
        if os.path.exists(viz_path):
            visualization_image = Image.open(viz_path)
    
    # Prepare output with correct terminology
    output = f"**Main Design Choices:**\n{main_design_choices}\n\n"
    output += f"**{DETECTION_TERM.title()} Analyses:**\n"
    for i, analysis in enumerate(analyses):
        output += f"**{DETECTION_TERM.title()} {i}:** {analysis}\n"
    output += f"\n**Final Analysis:**\n{final_analysis}"
    
    return final_analysis, output, visualization_image

def launch_gradio_interface():
    """Launch Gradio interface"""
    with gr.Blocks(css="""
        button { margin: 0.5em; }
        .container { margin: 0 auto; max-width: 1200px; }
        .image-container { flex: 0 0 70%; }
        .controls-container { flex: 0 0 30%; }
    """) as iface:
        gr.Markdown("# UI Screenshot to Prompt Generator")
        gr.Markdown("Upload an image of a UI to generate a prompt for an AI coder to reproduce the design.")
        
        # Main container with 70-30 split
        with gr.Row(equal_height=True):
            # Left side - Image containers (70%)
            with gr.Column(scale=7):
                input_image = gr.Image(type="pil", label="Upload UI Image")
                visualization_image = gr.Image(
                    label="Component Detection Visualization",
                    visible=False
                )
            
            # Right side - Controls (30%)
            with gr.Column(scale=3):
                # Add prompt choice selector at the top of control
                prompt_choice = gr.Radio(
                    choices=["Concise", "Extensive"],
                    value="Extensive",
                    label="Prompt Detail Level",
                    info="Choose between concise or extensive prompt generation"
                )
                
                detection_method = gr.Radio(
                    choices=["Basic", "Advanced"],
                    value="Basic",
                    label="Detection Method",
                    info="Basic: Grid-based splitting, Advanced: Smart component detection"
                )
                
                # Advanced settings (individual components, not in a Column)
                component_width = gr.Number(
                    label="Minimum Component Width",
                    value=MIN_COMPONENT_WIDTH_ADVANCED,
                    step=5,
                    minimum=20,
                    visible=False
                )
                component_height = gr.Number(
                    label="Minimum Component Height",
                    value=MIN_COMPONENT_HEIGHT_ADVANCED,
                    step=5,
                    minimum=20,
                    visible=False
                )
                max_components = gr.Number(
                    label="Maximum Components",
                    value=MAX_UI_COMPONENTS,
                    step=1,
                    minimum=1,
                    visible=False
                )

        # Output section
        with gr.Row():
            output_text = gr.Textbox(
                label="Caption Analyses", 
                lines=15,
                show_copy_button=True
            )
        
        with gr.Row():
            final_analysis_text = gr.Textbox(
                label="Final Analysis", 
                lines=5,
                show_copy_button=True,
                visible=False
            )
        
        with gr.Row():
            process_btn = gr.Button("Generate Prompt")
            copy_btn = gr.Button("Copy Final Analysis")
            
        notification = gr.Textbox(label="Status", interactive=False)
        
        def update_detection_method(mode):
            """Update visibility of advanced settings based on mode"""
            set_detection_method(mode)
            is_advanced = mode.lower() == "advanced"
            return (
                gr.update(visible=is_advanced),
                gr.update(visible=is_advanced),
                gr.update(visible=is_advanced)
            )
        
        # Update detection_method change handler
        detection_method.change(
            fn=update_detection_method,
            inputs=[detection_method],
            outputs=[component_width, component_height, max_components]
        )
        
        def update_prompt_choice(choice):
            """Update prompt choice configuration"""
            try:
                set_prompt_choice(choice)
                return "Prompt style updated successfully"
            except ValueError as e:
                return f"Error: {str(e)}"

        # Add prompt choice change handler
        prompt_choice.change(
            fn=update_prompt_choice,
            inputs=[prompt_choice],
            outputs=[notification]
        )
        
        def process_with_settings(image, mode, width, height, max_components, prompt_style):
            """Process image with current settings"""
            if image is None:
                return "", "Please upload an image first.", None, "Please upload an image"
            
            try:
                logger.info(f"Processing with width={width}, height={height}, max_components={max_components}, prompt_style={prompt_style}")
                final_analysis, full_output, viz_image = gradio_process_image(
                    image=image,
                    splitting_mode=mode,
                    max_components=max_components
                )
                
                viz_visible = mode.lower() == "advanced"
                return (
                    final_analysis,
                    full_output,
                    gr.update(value=viz_image, visible=viz_visible) if viz_image else gr.update(visible=False),
                    "Analysis generated successfully"
                )
                
            except Exception as e:
                logger.error(f"Error processing image: {str(e)}")
                return "", f"Error processing image: {str(e)}", gr.update(visible=False), "Error occurred"

        # Update process button to include prompt_choice
        process_btn.click(
            fn=process_with_settings,
            inputs=[
                input_image,
                detection_method,
                component_width,
                component_height,
                max_components,
                prompt_choice
            ],
            outputs=[
                final_analysis_text,
                output_text,
                visualization_image,
                notification
            ]
        )
        
        def copy_final_analysis(final_analysis):
            """Copy final analysis to clipboard with robust error handling and validation"""
            try:
                # Validate input
                if not isinstance(final_analysis, str):
                    logger.warning("Invalid final analysis type")
                    return (
                        gr.update(value="", visible=False),
                        "⚠️ Invalid analysis format. Please regenerate the analysis."
                    )
                
                if not final_analysis or final_analysis.strip() == "":
                    logger.info("Empty analysis detected")
                    return (
                        gr.update(value="", visible=False),
                        "⚠️ No analysis available to copy. Please generate an analysis first."
                    )
                
                # Clean and validate the analysis text
                cleaned_analysis = final_analysis.strip()
                if len(cleaned_analysis) < 10:  # Arbitrary minimum length
                    logger.warning("Analysis too short")
                    return (
                        gr.update(value=cleaned_analysis, visible=True),
                        "⚠️ Analysis seems incomplete. Consider regenerating."
                    )
                
                # Make the text visible and return success message
                logger.info("Successfully copied analysis")
                return (
                    gr.update(value=cleaned_analysis, visible=True),
                    "✅ Analysis ready to copy!"
                )
                
            except Exception as e:
                logger.error(f"Error in copy operation: {str(e)}", exc_info=True)
                return (
                    gr.update(value="", visible=False),
                    "❌ Error processing analysis. Please try again."
                )

        # Update the copy button click handler with improved error handling
        copy_btn.click(
            fn=copy_final_analysis,
            inputs=final_analysis_text,
            outputs=[final_analysis_text, notification],
            show_progress=True,
            api_name="copy_analysis"
        )
    
    iface.launch()

def main():
    logger.info("Starting image processing")
    image_path = os.path.join("images", "image.png")
    logger.info("Processing image...")
    main_design_choices, analyses, final_analysis = process_image(image_path)
    
    if analyses:
        logger.info(f"Main design choices: {main_design_choices}")
        logger.info(f"\n{DETECTION_TERM.title()} analyses:")
        for i, analysis in enumerate(analyses):
            logger.info(f"{DETECTION_TERM.title()} {i}: {analysis}")
        logger.info("\nFinal Analysis:")
        logger.info(final_analysis)
    else:
        logger.error("Failed to process image")

def call_super_prompt(main_image_caption: str, component_captions: List[str], activity_description: str) -> str:
    """Build and send the super prompt integrating all analyses"""
    try:
        # First build the base super prompt
        super_prompt = build_super_prompt(main_image_caption, component_captions, activity_description)
        
        # Clean up the prompt by removing first/last lines and extra whitespace
        cleaned_prompt = "\n".join(
            line for line in super_prompt.split("\n")[1:-1] 
            if line.strip()
        ).strip()
        
        # Add the "Build this app:" prefix
        final_prompt = f"Build this app: {cleaned_prompt}"
        
        logger.info("Generated super prompt: %s", final_prompt)
        
        if not super_prompt_function:
            raise ValueError("No API client available for super prompt generation")
            
        return super_prompt_function(final_prompt)
        
    except Exception as e:
        logger.error("Error in super prompt generation: %s", str(e))
        raise

if __name__ == "__main__":
    launch_gradio_interface()
