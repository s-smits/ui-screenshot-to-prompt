"""Vision related helpers and OpenAI interaction utilities."""

from __future__ import annotations

import base64
from io import BytesIO
import logging
from PIL import Image

from .config import MAIN_DESIGN_ANALYSIS_PROMPT
from .clients import ensure_openai_client

logger = logging.getLogger(__name__)


def encode_image_base64(image: Image.Image) -> str:
    """Convert a PIL image into a base64 encoded PNG string."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def call_vision_api(
    model: str,
    image: Image.Image,
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float = 0.1,
    json_response: bool = True,
) -> str:
    """Call the OpenAI vision API and return the response text."""
    client = ensure_openai_client()

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
                            "detail": "high",
                        },
                    },
                ],
            },
        ]

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=1024,
            temperature=temperature,
            response_format={"type": "json_object"} if json_response else None,
        )
        return response.choices[0].message.content.strip()

    except Exception as exc:  # pragma: no cover - relies on external service
        logger.error("OpenAI API error: %s", exc, exc_info=True)
        raise


def analyze_main_design_choices(image: Image.Image, temp: float = 0.1) -> str:
    """Analyze the entire UI design structure."""
    logger.info("Analyzing main design choices")
    try:
        return call_vision_api(
            model="gpt-4o",
            image=image,
            system_prompt=MAIN_DESIGN_ANALYSIS_PROMPT,
            user_prompt="Analyze this interface's complete design system and structure.",
            temperature=temp,
            json_response=False,
        )
    except Exception as exc:  # pragma: no cover - relies on external service
        logger.error("Error analyzing main design: %s", exc)
        return "Error analyzing main design structure"


def describe_activity(image: Image.Image) -> str:
    """Describe the primary activity shown in the image."""
    logger.info("Describing activity in image")
    return call_vision_api(
        model="gpt-4o",
        image=image,
        system_prompt="Describe the activity of this webpage in a few sentences.",
        user_prompt="What activity is shown in this image?",
        temperature=0.1,
        json_response=False,
    )
