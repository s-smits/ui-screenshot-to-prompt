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
    SPLITTING,
    set_splitting_mode,
    MIN_COMPONENT_WIDTH_SIMPLE,
    MIN_COMPONENT_HEIGHT_SIMPLE,
    MIN_COMPONENT_WIDTH_ADVANCED,
    MIN_COMPONENT_HEIGHT_ADVANCED
)

from image_splitter import get_image_splitter

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

def encode_image_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string"""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def call_vision_api(image: Image.Image, system_prompt: str, user_prompt: str, 
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
            model="gpt-4o-mini",
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
    """Analyze the main flow/purpose of the entire image"""
    logger.info("Analyzing main design choices")
    prompt = """Analyze this interface's complete design system and structure.
    Provide a detailed analysis following the JSON schema in the system prompt.
    Focus on implementation-relevant details and consistent patterns."""
    
    return call_vision_api(image, MAIN_DESIGN_ANALYSIS_PROMPT, prompt, temp)

def analyze_component(args):
    """Analyze individual component of the image"""
    component_image, main_design_choices, component_index, location = args
    logger.info(f"Analyzing component {component_index} in {location}")
    
    prompt = f"""Analyze this UI component:
    - Located in: {location}
    - Component number: {component_index}
    
    Provide structured analysis following the JSON schema in the system prompt.
    Focus on implementation-relevant details."""
    
    return call_vision_api(component_image, VISION_ANALYSIS_PROMPT, prompt)

def process_image(image_path: str, min_area: Optional[float] = None, max_components: int = 10):
    """Main function to process and analyze an image"""
    logger.info(f"Starting image processing for {image_path}")
    try:
        # Initialize splitter based on SPLITTING mode
        splitter = get_image_splitter(SPLITTING, image_path)
        
        # Get components (now includes location information)
        components = splitter.get_components()
        
        if not components:
            logger.error("No components detected. Exiting processing.")
            return "No components detected.", [], "Error in final analysis"
        
        # Analyze main image first
        main_image = Image.open(image_path)
        main_design_choices = analyze_main_design_choices(main_image)
        
        # Get activity description
        activity_description = describe_activity(main_image)
        
        # Prepare component analysis arguments
        analysis_args = []
        for i, component in enumerate(components):
            x, y, w, h = component.bbox
            component_img = cv2.cvtColor(np.array(splitter.image), cv2.COLOR_RGB2BGR)[y:y+h, x:x+w]
            component_rgb = cv2.cvtColor(component_img, cv2.COLOR_BGR2RGB)
            component_pil = Image.fromarray(component_rgb)
            
            # Include location information from component.text
            analysis_args.append((component_pil, main_design_choices, i, component.text))
            
            # Save the component
            output_path = os.path.join(splitter.output_dir, f"component_{i}.png")
            cv2.imwrite(output_path, component_img)
        
        # Process components in parallel
        with ThreadPoolExecutor(max_workers=5) as executor:
            component_analyses = list(executor.map(analyze_component, analysis_args))
        
        # Link analyses to components and include location information
        descriptions = []
        for component, analysis in zip(components, component_analyses):
            location_info = f"[{component.text}] "  # Add location prefix
            full_analysis = location_info + analysis
            component.text = full_analysis  # Store the full analysis
            descriptions.append(full_analysis)
        
        # Build and call super prompt
        final_analysis = call_super_prompt(
            main_design_choices,
            descriptions,
            activity_description
        )
        
        # Visualize all components
        splitter.visualize_components(
            cv2.cvtColor(np.array(splitter.image), cv2.COLOR_RGB2BGR),
            components,
            os.path.join(splitter.output_dir, "visualization.png")
        )
        
        logger.info("Image processing completed successfully")
        return main_design_choices, descriptions, final_analysis
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}", exc_info=True)
        return "Error processing image", [], "Error in final analysis"

def gradio_process_image(image, splitting_mode):
    """Process image uploaded through Gradio interface"""
    logger.info(f"Processing image uploaded through Gradio with splitting mode: {splitting_mode}")
    
    # Save the uploaded image temporarily
    temp_image_path = "temp_uploaded_image.png"
    image.save(temp_image_path)
    
    # Update config with selected splitting mode
    global SPLITTING
    SPLITTING = splitting_mode.lower()
    
    # Process the image
    main_design_choices, analyses, final_analysis = process_image(temp_image_path)
    
    # Get visualization image if in advanced mode
    visualization_image = None
    if splitting_mode.lower() == "advanced":
        viz_path = os.path.join("split_components", "visualization.png")
        if os.path.exists(viz_path):
            visualization_image = Image.open(viz_path)
    
    # Remove temporary image
    os.remove(temp_image_path)
    
    # Prepare output with Markdown formatting
    output = f"**Main Design Choices:**\n{main_design_choices}\n\n"
    output += "**Component Analyses:**\n"
    for i, analysis in enumerate(analyses):
        output += f"**Component {i}:** {analysis}\n"
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
        gr.Markdown("# Bolt.new Prompt Generator")
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
                splitting_mode = gr.Radio(
                    choices=["Easy", "Advanced"],
                    value="Easy",
                    label="Splitting Mode",
                    info="Easy: Grid-based splitting, Advanced: Smart component detection"
                )
                
                # Advanced settings (individual components, not in a Column)
                component_width = gr.Number(
                    label="Minimum Component Width",
                    value=MIN_COMPONENT_WIDTH_ADVANCED,
                    step=1,
                    minimum=1,
                    visible=False
                )
                component_height = gr.Number(
                    label="Minimum Component Height",
                    value=MIN_COMPONENT_HEIGHT_ADVANCED,
                    step=1,
                    minimum=1,
                    visible=False
                )

        # Output section
        with gr.Row():
            output_text = gr.Textbox(
                label="Caption Analyses", 
                lines=15,
                show_copy_button=False
            )
        
        with gr.Row():
            final_analysis_text = gr.Textbox(
                label="Final Analysis", 
                lines=5,
                visible=False
            )
        
        with gr.Row():
            process_btn = gr.Button("Generate Prompt")
            copy_btn = gr.Button("Copy Final Analysis")
            
        notification = gr.Textbox(label="Status", interactive=False)
        
        def update_mode(mode):
            """Update visibility of advanced settings based on mode"""
            set_splitting_mode(mode)
            is_advanced = mode.lower() == "advanced"
            return gr.update(visible=is_advanced), gr.update(visible=is_advanced)
        
        # Update splitting_mode change handler
        splitting_mode.change(
            fn=update_mode,
            inputs=[splitting_mode],
            outputs=[component_width, component_height]
        )
        
        # Rest of the event handlers...
        def process_with_settings(image, mode, width, height):
            """Process image with current settings"""
            if image is None:
                return "", "Please upload an image first.", None, "Please upload an image"
            
            try:
                logger.info(f"Processing with width={width}, height={height}")
                final_analysis, full_output, viz_image = gradio_process_image(
                    image=image,
                    splitting_mode=mode
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

        # Connect the process button
        process_btn.click(
            fn=process_with_settings,
            inputs=[
                input_image,
                splitting_mode,
                component_width,
                component_height
            ],
            outputs=[
                final_analysis_text,
                output_text,
                visualization_image,
                notification
            ]
        )
        
        def copy_final_analysis(final_analysis):
            """Copy final analysis to clipboard with error handling and validation"""
            try:
                if not final_analysis or final_analysis.strip() == "":
                    return (
                        gr.update(value="", visible=False),
                        "⚠️ No analysis available to copy. Please generate an analysis first."
                    )
                
                # Clean up the analysis text
                cleaned_analysis = final_analysis.strip()
                
                # Make the text visible and return success message
                return (
                    gr.update(value=cleaned_analysis, visible=True),
                    "✅ Copied final analysis!"
                )
                
            except Exception as e:
                logger.error(f"Error copying final analysis: {str(e)}")
                return (
                    gr.update(value="", visible=False),
                    "❌ Error copying analysis. Please try again."
                )

        # Update the copy button click handler
        copy_btn.click(
            fn=copy_final_analysis,
            inputs=final_analysis_text,
            outputs=[final_analysis_text, notification],
            show_progress=True
        )
    
    iface.launch()

def main():
    # Process single image (if needed)
    logger.info("Starting image processing")
    image_path = os.path.join("images", "image.png")
    logger.info("Processing image...")
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

def describe_activity(image: Image.Image) -> str:
    """Describe the activity shown in the image"""
    logger.info("Describing activity in image")
    prompt = "What activity is shown in this image?"
    
    return call_vision_api(
        image, 
        "Describe the activity in a few sentences.", 
        prompt,
        temperature=0.1,
        json_response=False
    )

def call_super_prompt(main_image_caption: str, component_captions: List[str], activity_description: str) -> str:
    """Build and send the super prompt integrating all analyses"""
    super_prompt = build_super_prompt(main_image_caption, component_captions, activity_description)
    
    # Add "Build this app:" to the beginning of the prompt
    super_prompt = f"Build this app: {super_prompt}"
    
    # Remove last line from super prompt
    super_prompt = super_prompt.rstrip()
    
    # Print the super prompt
    print(f"Super prompt: {super_prompt}")
    
    if not super_prompt_function:
        raise Exception("No API client available for super prompt generation")

    try:
        logger.info("Calling super prompt function")
        return super_prompt_function(super_prompt)
    except Exception as e:
        logger.error("Error in super prompt generation: %s", str(e))
        raise Exception(f"Error in super prompt generation: {str(e)}")

if __name__ == "__main__":
    launch_gradio_interface()
