"""Core image processing pipeline for generating UI prompts."""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

import cv2
from PIL import Image

from .clients import ensure_super_prompt_function
from .config import (
    VISION_ANALYSIS_PROMPT,
    build_super_prompt,
    get_detection_method,
    get_detection_term,
    MAX_UI_COMPONENTS,
)
from .detection import create_detector
from .vision import analyze_main_design_choices, describe_activity, call_vision_api

logger = logging.getLogger(__name__)


def analyze_detection(detection_image: Image.Image, index: int, location: str) -> str:
    """Analyze a single detection using the vision model."""
    detection_term = get_detection_term()
    logger.info("Analyzing %s %d in %s", detection_term, index, location)

    prompt = f"""Analyze this UI {detection_term}:
    - Located in: {location}
    - {detection_term.title()} number: {index}

    Provide structured analysis following the JSON schema in the system prompt.
    Focus on implementation-relevant details."""

    return call_vision_api(
        model="gpt-4o-mini",
        image=detection_image,
        system_prompt=VISION_ANALYSIS_PROMPT,
        user_prompt=prompt,
    )


def process_image(
    image_path: str,
    min_area: Optional[float] = None,
    max_detections: int = MAX_UI_COMPONENTS,
) -> Tuple[str, List[str], str]:
    """Process an image and return main design, region analyses, and final prompt."""
    del min_area  # Maintained for backward compatibility with previous signatures.

    try:
        output_dir = "split_detections"
        os.makedirs(output_dir, exist_ok=True)

        detection_method = get_detection_method() or "basic"
        detector = create_detector(
            detection_method,
            image_path,
            max_components=max_detections,
        )

        detections = detector.get_components()[:max_detections]
        detection_term = get_detection_term()

        if not detections:
            logger.error("No %ss detected. Exiting processing.", detection_term)
            return f"No {detection_term}s detected.", [], "Error in final analysis"

        main_image = Image.open(image_path)
        main_design_choices = analyze_main_design_choices(main_image)
        activity_description = describe_activity(main_image)

        analysis_args: List[Tuple[Image.Image, int, str]] = []
        locations: List[str] = []

        for index, detection in enumerate(detections):
            x, y, w, h = detection.bbox
            detection_img = detector.image[y:y + h, x:x + w]
            detection_rgb = cv2.cvtColor(detection_img, cv2.COLOR_BGR2RGB)
            detection_pil = Image.fromarray(detection_rgb)
            location_label = detection.text or f"{detection_term.title()} {index}"

            locations.append(location_label)
            analysis_args.append((detection_pil, index, location_label))

            output_path = os.path.join(output_dir, f"{detection_term}_{index}.png")
            cv2.imwrite(output_path, detection_img)

        with ThreadPoolExecutor(max_workers=5) as executor:
            detection_analyses = list(
                executor.map(lambda args: analyze_detection(*args), analysis_args)
            )

        descriptions: List[str] = []
        for detection, location_label, analysis in zip(detections, locations, detection_analyses):
            full_analysis = f"[{location_label}] {analysis}"
            detection.text = full_analysis
            descriptions.append(full_analysis)

        final_analysis = call_super_prompt(
            main_design_choices,
            descriptions,
            activity_description,
        )

        detector.visualize_detections(
            detector.image,
            detections,
            os.path.join(output_dir, "visualization.png"),
        )

        logger.info("Image processing completed successfully")
        return main_design_choices, descriptions, final_analysis

    except Exception as exc:  # pragma: no cover - relies on external services
        logger.error("Error processing image: %s", exc, exc_info=True)
        return "Error processing image", [], "Error in final analysis"


def call_super_prompt(
    main_image_caption: str,
    component_captions: List[str],
    activity_description: str,
) -> str:
    """Build and execute the super prompt for final analysis."""
    super_prompt = build_super_prompt(main_image_caption, component_captions, activity_description)
    cleaned_prompt = "\n".join(
        line for line in super_prompt.split("\n")[1:-1]
        if line.strip()
    ).strip()
    final_prompt = f"Build this app: {cleaned_prompt}"

    logger.info("Generated super prompt: %s", final_prompt)

    prompt_callable = ensure_super_prompt_function()
    return prompt_callable(final_prompt)
