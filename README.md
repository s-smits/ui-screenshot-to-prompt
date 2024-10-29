# ui-screenshot-to-prompt

ui-screenshot-to-prompt is an AI-powered tool that analyzes UI images to generate detailed prompts for AI coders. It uses computer vision and natural language processing to break down UI components, analyze design patterns, and create comprehensive descriptions for reproducing the design. Very useful for Bolt.new, v0 and other upcoming SaaS.

## Demo
https://github.com/user-attachments/assets/79c2722e-942d-4f0c-84bd-11066b63f4c5

## Features
- Smart image splitting and component detection
- OCR for text extraction
- UI element classification (buttons, text fields, checkboxes, etc.)
- Individual component analysis
- Overall design pattern analysis
- Gradio web interface for easy usage

## Detailed Usage Guide

### Splitting Modes

The tool offers two splitting modes for analyzing UI images:

1. **Easy Mode**
- Grid-based splitting of the image
- Automatically determines optimal grid size based on image dimensions and aspect ratio (max 3x3)
- Creates detailed spatial annotations that describe component placement and hierarchy within the interface layout

2. **Advanced Mode** (Experimental)
- Smart component detection using computer vision techniques
- Identifies UI elements like buttons, text fields, and checkboxes
- Includes visualization of detected components
- Uses configurable minimum dimensions for component detection
- Note: This mode is still experimental and may need improvements for optimal results

### Component Analysis

Each detected component is analyzed for:
- Component type classification
- Position and dimensions
- Confidence score for detection
- Location

## Requirements

### API Requirements

The tool requires:

1. **OpenAI API**
- Used for vision analysis; general analysis through GPT-4o and individual components through GPT-4o-mini
- Required for component and design analysis

2. **Anthropic/Openrouter API**
- Used for creating detailed super prompts via Claude
- Recommended for most accurate results

### System Requirements

- Python 3.10+
- Rust (unfortunately for [tokenizers]([url](https://pypi.org/project/tokenizers/)) dependency)
- Poetry (for dependency management)

## Installation

1. Clone the repository:
```
git clone https://github.com/s-smits/ui-screenshot-to-prompt.git
cd ui-screenshot-to-prompt
```

2. Install required system dependencies:

For Unix-based systems (macOS/Linux):
```
# macOS (using Homebrew)
brew install rust

# Linux (Ubuntu/Debian)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

3. Install Poetry if you haven't already:
```
curl -sSL https://install.python-poetry.org | python3 -
```

4. Install dependencies:
```
poetry install
```

5. Set up environment variables:
- Rename the `.env.example` file to `.env`:
  ```
  mv .env.example .env
  ```
- Open the `.env` file and replace the placeholder values with your actual API keys and URL:
  ```
  OPENAI_API_KEY=your_openai_api_key
  
  ANTHROPIC_API_KEY=your_anthropic_api_key
  # OR
  OPENROUTER_API_KEY=your_openrouter_api_key
  ```

## Usage

1. Activate the Poetry environment:
```
poetry shell
```

2. Run the Gradio interface:
```
python src/ui-screenshot-to-prompt/main.py
```

3. Open the provided URL in your web browser to access the Gradio interface.

4. Upload an image of a UI design, and the tool will generate a detailed prompt for reproducing the design.

## Configuration

You can adjust various parameters in the `config.py` file, such as:

- System prompts
- Vision analysis prompts
- Super prompt template

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenAI for GPT models
- OpenRouter for API access
- Gradio for the web interface
- Tesseract OCR for text extraction

[![Star History Chart](https://api.star-history.com/svg?repos=s-smits/ui-screenshot-to-prompt&type=Date)](https://star-history.com/#s-smits/ui-screenshot-to-prompt&Date)
