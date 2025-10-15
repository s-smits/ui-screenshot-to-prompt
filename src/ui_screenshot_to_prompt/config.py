from typing import List, Callable, Tuple, Optional
from functools import lru_cache
from dotenv import load_dotenv
import os
import logging

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - handled gracefully at runtime
    OpenAI = None  # type: ignore

try:
    from anthropic import Anthropic
except ImportError:  # pragma: no cover - handled gracefully at runtime
    Anthropic = None  # type: ignore

# Initialize logger
logger = logging.getLogger(__name__)

# Global configuration variables
DETECTION_METHOD = 'basic'
DETECTION_TERM = 'region' if DETECTION_METHOD == 'basic' else 'component'
PROMPT_CHOICE = 'extensive'

# Default component dimensions
MIN_COMPONENT_WIDTH_ADVANCED = 50
MIN_COMPONENT_HEIGHT_ADVANCED = 50
MAX_UI_COMPONENTS = 6

MIN_COMPONENT_WIDTH = MIN_COMPONENT_WIDTH_ADVANCED
MIN_COMPONENT_HEIGHT = MIN_COMPONENT_HEIGHT_ADVANCED

MIN_REGION_WIDTH_SIMPLE = 200
MIN_REGION_HEIGHT_SIMPLE = 200

def set_detection_method(method: str):
    """Update the detection method configuration"""
    global DETECTION_METHOD, DETECTION_TERM
    if method.lower() not in ['basic', 'advanced']:
        raise ValueError("Invalid detection method. Must be 'basic' or 'advanced'")
    DETECTION_METHOD = method.lower()
    DETECTION_TERM = 'region' if DETECTION_METHOD == 'basic' else 'component'
    logger.info(f"Detection method set to: {DETECTION_METHOD} ({DETECTION_TERM}s)")

def get_detection_method() -> str:
    """Get current detection method"""
    return DETECTION_METHOD

def get_detection_term() -> str:
    """Get current detection terminology"""
    return DETECTION_TERM

def set_prompt_choice(choice: str):
    """Update the prompt choice configuration"""
    global PROMPT_CHOICE
    normalized = choice.lower()
    if normalized not in ['concise', 'extensive']:
        raise ValueError("Invalid prompt choice. Must be 'concise' or 'extensive'")
    PROMPT_CHOICE = normalized
    logger.info("Prompt choice set to: %s", PROMPT_CHOICE)


def get_prompt_choice() -> str:
    """Return the currently configured prompt choice"""
    return PROMPT_CHOICE


def set_splitting_mode(mode: str):
    """Alias for UI backwards compatibility"""
    set_detection_method(mode)


def get_splitting_mode() -> str:
    """Alias for UI backwards compatibility"""
    return get_detection_method()


@lru_cache(maxsize=1)
def load_and_initialize_clients() -> Tuple[OpenAI, Optional[Callable[[str], str]]]:
    # Load environment variables from the .env file
    load_dotenv()
    logger.info("Environment variables loaded")

    # Retrieve API keys from environment variables
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

    # Check if OpenAI API key is available
    if OpenAI is None or not openai_api_key:
        logger.error("OPENAI_API_KEY is not set or OpenAI client missing.")
        raise ValueError("Missing OpenAI API client configuration")

    # Initialize the OpenAI client
    openai_client = OpenAI(api_key=openai_api_key)
    logger.info("OpenAI client initialized")

    # Initialize super prompt function
    super_prompt_function: Optional[Callable[[str], str]] = None

    if anthropic_api_key and Anthropic is not None:
        anthropic_client = Anthropic(api_key=anthropic_api_key)
        
        def anthropic_super_prompt(prompt: str) -> str:
            response = anthropic_client.messages.create(
                model="claude-3-sonnet-latest",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4096 
            )
            return response.content[0].text

        super_prompt_function = anthropic_super_prompt
        logger.info("Anthropic client initialized and set as super prompt client")
    elif openrouter_api_key:
        openrouter_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_api_key,
        )
        
        def openrouter_super_prompt(prompt: str) -> str:
            response = openrouter_client.chat.completions.create(
                model="anthropic/claude-3-sonnet",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4096 
            )
            return response.choices[0].message.content

        super_prompt_function = openrouter_super_prompt
        logger.info("OpenRouter client initialized and set as super prompt client")
    else:
        logger.warning("No client available for super prompt generation")

    return openai_client, super_prompt_function

VISION_ANALYSIS_PROMPT = """You are an expert AI system analyzing UI components for development replication.

COMPONENT ANALYSIS REQUIREMENTS:
{
    "type": "Identify component type (button/input/card/etc)",
    "visual": {
        "colors": ["primary", "secondary", "text"],
        "dimensions": "size and spacing",
        "typography": "font styles and weights",
        "borders": "border styles and radius",
        "shadows": "elevation and depth"
    },
    "content": {
        "text": "content and labels",
        "icons": "icon types if present",
        "images": "image content if present"
    },
    "interaction": {
        "primary": "main interaction type",
        "states": ["hover", "active", "disabled"],
        "animations": "transitions and effects"
    },
    "location": {
        "position": "relative to parent/siblings",
        "alignment": "layout alignment",
        "spacing": "margins and padding"
    }
}

OUTPUT FORMAT:
{
    "component": "technical name (<5 words)",
    "specs": {
        // Fill above structure with detected values
    },
    "implementation": "key technical considerations (<15 words)"
}"""


MAIN_DESIGN_ANALYSIS_PROMPT = """You are an expert UI/UX analyzer creating structured design specifications.

ANALYZE AND OUTPUT THE FOLLOWING JSON STRUCTURE:
{
    "layout": {
        "pattern": "primary layout system (grid/flex/etc)",
        "structure": {
            "sections": ["header", "main", "footer", etc],
            "columns": {
                "count": "number of columns",
                "sizes": "column width distributions"
            },
            "elements": {
                "boxes": "count and arrangement",
                "circles": "diameter and placement"
            }
        },
        "spacing": {
            "between_sections": "major gaps",
            "between_elements": "element spacing"
        },
        "responsive_hints": "visible breakpoint considerations"
    },
    "design_system": {
        "colors": {
            "primary": "main color palette",
            "secondary": "supporting colors",
            "text": "text hierarchy colors",
            "background": "surface colors",
            "interactive": "button/link colors"
        },
        "typography": {
            "headings": "heading hierarchy",
            "body": "body text styles",
            "special": "distinctive text styles"
        },
        "components": {
            "shadows": "elevation levels",
            "borders": "border styles",
            "radius": "corner rounding"
        }
    },
    "interactions": {
        "buttons": {
            "types": "button variations",
            "states": "visible states (hover/disabled)"
        },
        "inputs": "form element patterns",
        "feedback": "visible status indicators"
    },
    "content": {
        "media": {
            "images": "image usage patterns",
            "aspect_ratios": "common ratios"
        },
        "text": {
            "lengths": "content constraints",
            "density": "text distribution"
        }
    },
    "visual_hierarchy": {
        "emphasis": "attention hierarchy",
        "flow": "visual reading order",
        "density": "content distribution"
    },
    "implementation_notes": "key technical considerations (<30 words)"
}"""

def build_super_prompt(
    main_image_caption: str, 
    region_descriptions: List[str],
    activity_description: str,
    prompt_size: Optional[str] = None
) -> str:
    """Build UI recreation prompt with configurable detail level
    
    Args:
        main_image_caption: Overall layout description
        region_descriptions: List of detected UI regions from image splitting
        activity_description: User interaction patterns
        prompt_size: Size of prompt - "concise" or "extensive" (default: current prompt setting)
    """

    # Get current detection terminology
    detection_term = get_detection_term()
    active_prompt = (prompt_size or get_prompt_choice()).lower()
    
    # Build the region specifications string with proper terminology
    region_specs = "\n".join(
        f"{detection_term.title()} {i + 1}: {desc}"
        for i, desc in enumerate(region_descriptions)
    ) or "No regional analysis available"

    # Format main caption if it's not empty
    layout_section = main_image_caption if main_image_caption else "No layout analysis available"

    if active_prompt == "concise":
        prompt = f"""This study presents a systematic analysis framework for precise UI replication, incorporating component specifications and visual hierarchy assessment. The framework examines:

        [{detection_term.title()} Analysis]
        {region_specs}

        [Layout Analysis]
        {layout_section}

        [Interactive Elements]
        {str(activity_description)}

        Technical Specifications for Implementation:

        1. Layout Architecture
        - Container dimensions and responsive breakpoints
        - Component positioning matrix including:
            • Primary sections (header, content, footer)
            • Grid system specifications
            • Spatial relationships and padding metrics

        2. Visual Parameters
        - Color schema (primary, secondary, accent)
        - Typography specifications
        - Elevation system (shadows, borders)

        3. Component Specifications
        - Interactive controls
        - Static elements
        - State representations

        4. Content Parameters
        - Text constraints and overflow behavior
        - Media dimensions and ratios
        - Component hierarchy

        This framework enables precise replication while maintaining structural integrity and interactive functionality across various viewport dimensions.
        """
    else:
        prompt = f"""You are an expert UI development agent tasked with providing exact technical specifications for recreating this interface. Analyze all details with high precision:

        [Components Specifications by Location]
        {region_specs}

        [Layout Structure]
        {layout_section}

        [Interaction Patterns]
        {str(activity_description)}

        Note: If a component has already been explained in detail above, only its name and location will be listed below to provide geographical context.

        Provide a complete technical specification for exact replication in text format:

        1. Layout Structure
        - Primary container dimensions
        - Component positioning map:
            • Header, main content, sidebars, footer
            • Layout elements:
                - Number and size of columns (e.g., 3 columns at 33% each)
                - Number and height of rows
                - Grid/box count and arrangement
                - Circular elements diameter and placement
            • Spacing and gaps:
                - Between major sections
                - Between grid items
                - Inner padding
        - Responsive behavior:
            • Breakpoint dimensions
            • Layout changes at each breakpoint
            • Element reflow rules
        
        2. Visual Style
        - Colors:
            • Primary, secondary, accent colors
            • Background colors
            • Text colors
            • Border colors
        - Typography:
            • Font sizes
            • Text weights
            • Text alignment
        - Depth and Emphasis:
            • Visible shadows
            • Border styles
            • Opacity levels
        
        3. Visible Elements
        - Controls:
            • Button appearances (if new, otherwise location only)
            • Form element styling (if new, otherwise location only)
            • Interactive element looks (if new, otherwise location only)
        - Static Elements:
            • Images and icons (if new, otherwise location only)
            • Text content (if new, otherwise location only)
            • Decorative elements (if new, otherwise location only)
        - Visual States:
            • Active/selected states
            • Disabled appearances
            • Current page indicators
        
        4. Content Presentation
        - Text:
            • Visible length limits
            • Current overflow handling
            • Text wrapping behavior
        - Media:
            • Image dimensions
            • Aspect ratios
            • Current placeholder states
        
        5. Visual Hierarchy
        - Element stacking
        - Content grouping
        - Visual emphasis
        - Spatial relationships between previously described components
        """

    return prompt
