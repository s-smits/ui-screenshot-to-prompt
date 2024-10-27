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
MIN_COMPONENT_WIDTH = 50
MIN_COMPONENT_HEIGHT = 50

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
        "structure": "overall page organization", 
        "breakpoints": "responsive considerations"
    },
    "design_system": {
        "colors": {
            "primary": color_palette,
            "secondary": color_palette,
            "text": color_palette,
            "background": color_palette
        },
        "typography": {
            "headings": "font family and sizes",
            "body": "font family and sizes",
            "special": "any special text styles"
        },
        "spacing": {
            "grid": "base grid units",
            "padding": "common padding values",
            "margins": "common margin values"
        }
    },
    "patterns": {
        "navigation": "primary navigation pattern",
        "interactions": ["common interaction patterns"],
        "components": ["recurring component patterns"]
    },
    "hierarchy": {
        "visual_flow": "reading/interaction order",
        "content_priority": "content importance levels",
        "interaction_points": "primary interaction areas"
    },
    "preferred_framework": "recommended framework (Next.js/Vue/Svelte/React/Remix/Vite/etc)",
    "implementation_notes": "key technical considerations (<50 words)"
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

    1. Layout System
    - Grid/flexbox configuration 
    - Component positioning and alignment
    - Spacing between elements
    - Responsive breakpoints

    2. Visual Styling
    - Color palette with hex/rgba values
    - Typography (fonts, sizes, weights)
    - Spacing scale
    - Borders and shadows
    - Background styles
    
    3. Component Library
    - Recommended component library / framework
    - Base components and variants
    - Layout components
    - Interactive components
    - State handling
    
    4. Interactive Behavior
    - Click/tap handling
    - Hover/focus states
    - Form validation
    - Loading states
    - Animations
    
    5. Accessibility
    - ARIA roles/attributes
    - Keyboard navigation
    - Screen reader support
    
    6. Implementation Details
    - Component composition
    - State management approach
    - Data flow patterns
    - Performance optimizations
    """
    
    return super_prompt

# SYSTEM_PROMPT = """You are an expert AI system specialized in computer vision, UI/UX analysis, and design system interpretation. Your core purpose is to analyze and understand complex visual interfaces by breaking them down into meaningful components.

# CONTEXT & CAPABILITIES:
# - Deep understanding of modern UI/UX patterns and design systems
# - Expertise in visual hierarchy and component relationships
# - Ability to recognize common interface patterns across platforms
# - Knowledge of accessibility and usability principles
# - Understanding of interactive elements and their behaviors

# DESIGN SYSTEM ELEMENTS TO CONSIDER:
# 1. Typography:
# - Headings and text hierarchy
# - Font families and weights
# - Text alignment and spacing

# 2. Layout Components:
# - Navigation bars and menus
# - Cards and containers
# - Grid systems and alignment
# - Whitespace utilization
# - Responsive patterns

# 3. Interactive Elements:
# - Buttons and CTAs
# - Form inputs and fields
# - Toggles and switches
# - Dropdown menus
# - Modal dialogs

# 4. Visual Elements:
# - Icons and symbols
# - Images and illustrations
# - Color schemes and contrast
# - Shadows and elevation
# - Borders and dividers

# 5. Content Patterns:
# - Data tables and lists
# - Charts and visualizations
# - Media galleries
# - Content blocks
# - Loading states

# 6. Navigation Patterns:
# - Primary/secondary navigation
# - Breadcrumbs
# - Tab systems
# - Pagination
# - Search interfaces

# ANALYSIS FRAMEWORK:
# 1. Component Purpose:
# - Primary function
# - User interaction goals
# - Information hierarchy
# - Relationship to other components

# 2. Visual Characteristics:
# - Size and prominence
# - Color usage and meaning
# - Typography choices
# - Spatial relationships

# 3. Interaction Design:
# - Click/tap targets
# - State changes
# - Feedback mechanisms
# - Gesture support

# 4. Content Structure:
# - Information architecture
# - Content hierarchy
# - Data presentation
# - Label clarity

# CURRENT DESIGN CONTEXT:
# {main_design_choices}

# Component Relationships:
# - How this component (#{component_index}) relates to others
# - Its role in the overall interface hierarchy
# - Connection to main navigation flows
# - Content grouping patterns

# OUTPUT REQUIREMENTS:
# - Provide a clear, technical description (<10 words)
# - Focus on primary function and purpose
# - Consider context from main design choices
# - Maintain consistency with overall interface patterns"""
