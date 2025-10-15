"""Legacy entry points preserved for backward compatibility."""

from __future__ import annotations

import logging

from .app import launch_gradio_interface, gradio_process_image
from .pipeline import call_super_prompt, process_image
from .vision import (
    analyze_main_design_choices,
    call_vision_api,
    describe_activity,
    encode_image_base64,
)

logger = logging.getLogger(__name__)


def main() -> None:
    """CLI entry point that launches the Gradio interface."""
    logger.info("Launching Gradio interface")
    launch_gradio_interface()


__all__ = [
    "process_image",
    "call_super_prompt",
    "gradio_process_image",
    "launch_gradio_interface",
    "encode_image_base64",
    "call_vision_api",
    "analyze_main_design_choices",
    "describe_activity",
    "main",
]
