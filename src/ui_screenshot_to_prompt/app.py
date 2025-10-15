"""Gradio interface setup for the UI screenshot to prompt workflow."""

from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

import gradio as gr
from PIL import Image

from .config import (
    MAX_UI_COMPONENTS,
    MIN_COMPONENT_HEIGHT_ADVANCED,
    MIN_COMPONENT_WIDTH_ADVANCED,
    get_detection_method,
    get_detection_term,
    set_detection_method,
    set_prompt_choice,
)
from .pipeline import process_image

logger = logging.getLogger(__name__)


def gradio_process_image(
    image: Image.Image,
    splitting_mode: str,
    max_components: int = MAX_UI_COMPONENTS,
) -> Tuple[str, str, Optional[Image.Image]]:
    """Wrapper for processing images uploaded via Gradio."""
    logger.info("Processing image uploaded through Gradio with splitting mode: %s", splitting_mode)

    temp_image_path = "temp_uploaded_image.png"
    image.save(temp_image_path)

    set_detection_method(splitting_mode)

    main_design_choices, analyses, final_analysis = process_image(
        temp_image_path,
        max_detections=max_components,
    )

    visualization_image: Optional[Image.Image] = None
    if get_detection_method() == "advanced":
        viz_path = os.path.join("split_detections", "visualization.png")
        if os.path.exists(viz_path):
            visualization_image = Image.open(viz_path)

    detection_term = get_detection_term()
    output = f"**Main Design Choices:**\n{main_design_choices}\n\n"
    output += f"**{detection_term.title()} Analyses:**\n"
    for index, analysis in enumerate(analyses):
        output += f"**{detection_term.title()} {index}:** {analysis}\n"
    output += f"\n**Final Analysis:**\n{final_analysis}"

    return final_analysis, output, visualization_image


def launch_gradio_interface():
    """Launch the Gradio interface for the application."""
    with gr.Blocks(css="""
        button { margin: 0.5em; }
        .container { margin: 0 auto; max-width: 1200px; }
        .image-container { flex: 0 0 70%; }
        .controls-container { flex: 0 0 30%; }
    """) as iface:
        gr.Markdown("# UI Screenshot to Prompt Generator")
        gr.Markdown("Upload an image of a UI to generate a prompt for an AI coder to reproduce the design.")

        with gr.Row(equal_height=True):
            with gr.Column(scale=7):
                input_image = gr.Image(type="pil", label="Upload UI Image")
                visualization_image = gr.Image(
                    label="Component Detection Visualization",
                    visible=False,
                )

            with gr.Column(scale=3):
                prompt_choice = gr.Radio(
                    choices=["Concise", "Extensive"],
                    value="Extensive",
                    label="Prompt Detail Level",
                    info="Choose between concise or extensive prompt generation",
                )

                detection_method = gr.Radio(
                    choices=["Basic", "Advanced"],
                    value="Basic",
                    label="Detection Method",
                    info="Basic: Grid-based splitting, Advanced: Smart component detection",
                )

                component_width = gr.Number(
                    label="Minimum Component Width",
                    value=MIN_COMPONENT_WIDTH_ADVANCED,
                    step=5,
                    minimum=20,
                    visible=False,
                )
                component_height = gr.Number(
                    label="Minimum Component Height",
                    value=MIN_COMPONENT_HEIGHT_ADVANCED,
                    step=5,
                    minimum=20,
                    visible=False,
                )
                max_components = gr.Number(
                    label="Maximum Components",
                    value=MAX_UI_COMPONENTS,
                    step=1,
                    minimum=1,
                    visible=False,
                )

        with gr.Row():
            output_text = gr.Textbox(
                label="Caption Analyses",
                lines=15,
                show_copy_button=True,
            )

        with gr.Row():
            final_analysis_text = gr.Textbox(
                label="Final Analysis",
                lines=5,
                show_copy_button=True,
                visible=False,
            )

        with gr.Row():
            process_btn = gr.Button("Generate Prompt")
            copy_btn = gr.Button("Copy Final Analysis")

        notification = gr.Textbox(label="Status", interactive=False)

        def update_detection_method(mode: str):
            set_detection_method(mode)
            is_advanced = mode.lower() == "advanced"
            return (
                gr.update(visible=is_advanced),
                gr.update(visible=is_advanced),
                gr.update(visible=is_advanced),
            )

        detection_method.change(
            fn=update_detection_method,
            inputs=[detection_method],
            outputs=[component_width, component_height, max_components],
        )

        def update_prompt_choice(choice: str):
            try:
                set_prompt_choice(choice)
                return "Prompt style updated successfully"
            except ValueError as exc:
                return f"Error: {exc}"

        prompt_choice.change(
            fn=update_prompt_choice,
            inputs=[prompt_choice],
            outputs=[notification],
        )

        def process_with_settings(image, mode, width, height, max_components_value, prompt_style):
            if image is None:
                return "", "Please upload an image first.", None, "Please upload an image"

            try:
                set_prompt_choice(prompt_style)
                logger.info(
                    "Processing with width=%s, height=%s, max_components=%s, prompt_style=%s",
                    width,
                    height,
                    max_components_value,
                    prompt_style,
                )

                final_analysis, full_output, viz_image = gradio_process_image(
                    image=image,
                    splitting_mode=mode,
                    max_components=int(max_components_value),
                )

                viz_visible = mode.lower() == "advanced"
                viz_update = (
                    gr.update(value=viz_image, visible=viz_visible)
                    if viz_image
                    else gr.update(visible=False)
                )

                return (
                    final_analysis,
                    full_output,
                    viz_update,
                    "Analysis generated successfully",
                )

            except Exception as exc:
                logger.error("Error processing image: %s", exc, exc_info=True)
                return "", f"Error processing image: {exc}", gr.update(visible=False), "Error occurred"

        process_btn.click(
            fn=process_with_settings,
            inputs=[
                input_image,
                detection_method,
                component_width,
                component_height,
                max_components,
                prompt_choice,
            ],
            outputs=[
                final_analysis_text,
                output_text,
                visualization_image,
                notification,
            ],
        )

        def copy_final_analysis(final_analysis: str):
            try:
                if not isinstance(final_analysis, str):
                    logger.warning("Invalid final analysis type")
                    return (
                        gr.update(value="", visible=False),
                        "⚠️ Invalid analysis format. Please regenerate the analysis.",
                    )

                if not final_analysis or final_analysis.strip() == "":
                    logger.info("Empty analysis detected")
                    return (
                        gr.update(value="", visible=False),
                        "⚠️ No analysis available to copy. Please generate an analysis first.",
                    )

                cleaned_analysis = final_analysis.strip()
                if len(cleaned_analysis) < 10:
                    logger.warning("Analysis too short")
                    return (
                        gr.update(value=cleaned_analysis, visible=True),
                        "⚠️ Analysis seems incomplete. Consider regenerating.",
                    )

                logger.info("Successfully copied analysis")
                return (
                    gr.update(value=cleaned_analysis, visible=True),
                    "✅ Analysis ready to copy!",
                )

            except Exception as exc:
                logger.error("Error in copy operation: %s", exc, exc_info=True)
                return (
                    gr.update(value="", visible=False),
                    "❌ Error processing analysis. Please try again.",
                )

        copy_btn.click(
            fn=copy_final_analysis,
            inputs=final_analysis_text,
            outputs=[final_analysis_text, notification],
            show_progress=True,
            api_name="copy_analysis",
        )

    iface.launch()
