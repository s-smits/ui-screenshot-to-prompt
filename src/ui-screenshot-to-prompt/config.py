from typing import List, Callable, Any, Tuple, Optional
from dotenv import load_dotenv
import os
import logging
from openai import OpenAI
from anthropic import Anthropic

# Initialize logger
logger = logging.getLogger(__name__)

# Global configuration variables
global SPLITTING
SPLITTING = 'easy'

# Default component dimensions
MIN_COMPONENT_WIDTH_ADVANCED = 50
MIN_COMPONENT_HEIGHT_ADVANCED = 50
MIN_COMPONENT_WIDTH_SIMPLE = 200
MIN_COMPONENT_HEIGHT_SIMPLE = 200

def set_splitting_mode(mode: str):
    """Update the splitting mode configuration"""
    global SPLITTING
    if mode.lower() not in ['easy', 'advanced']:
        raise ValueError("Invalid splitting mode. Must be 'easy' or 'advanced'")
    SPLITTING = mode.lower()
    logger.info(f"Splitting mode set to: {SPLITTING}")

def get_splitting_mode() -> str:
    """Get the current splitting mode"""
    return SPLITTING

def load_and_initialize_clients() -> Tuple[OpenAI, Optional[Callable[[str], str]]]:
    # Load environment variables from the .env file
    load_dotenv()
    logger.info("Environment variables loaded")

    # Retrieve API keys from environment variables
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

    # Check if OpenAI API key is available
    if not openai_api_key:
        logger.error("OPENAI_API_KEY is not set in the environment variables.")
        raise ValueError("Missing OpenAI API key")

    # Initialize the OpenAI client
    openai_client = OpenAI(api_key=openai_api_key)
    logger.info("OpenAI client initialized")

    # Initialize super prompt function
    super_prompt_function: Optional[Callable[[str], str]] = None

    if anthropic_api_key:
        anthropic_client = Anthropic(api_key=anthropic_api_key)
        
        def anthropic_super_prompt(prompt: str) -> str:
            response = anthropic_client.messages.create(
                model="claude-3-sonnet-latest",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048
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
                max_tokens=2048
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

def build_super_prompt(main_image_caption: str, component_captions: List[str], activity_description: str) -> str:
    """Build comprehensive prompt incorporating all image analyses for UI recreation"""
    # Parse raw text for components
    components = [caption.strip() for caption in component_captions]
    
    # Build structured component breakdown with clear numbering
    component_specs = "\n".join([
        f"Component {i + 1}:\n{comp}"
        for i, comp in enumerate(components)
    ])
    
    super_prompt = f"""You are an expert UI development agent tasked with providing exact technical specifications for recreating this interface. Analyze all details with high precision:

    [Detailed Analysis]
    {component_specs}

    [Layout Structure]
    {main_image_caption}

    [Interaction Patterns]
    {activity_description}

    Provide a complete technical specification for exact replication:

    1. Layout Structure
    - Primary container dimensions
    - Component positioning map:
        • Header, main content, sidebars, footer
        �� Layout elements:
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
        • Button appearances
        • Form element styling
        • Interactive element looks
    - Static Elements:
        • Images and icons
        • Text content
        • Decorative elements
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
    - Spatial relationships
    """
    
    return super_prompt
