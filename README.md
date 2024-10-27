# ui-screenshot-to-prompt
ui-screenshot-to-prompt is an AI-powered tool that analyzes UI images to generate detailed prompts for AI coders. It uses computer vision and natural language processing to break down UI components, analyze design patterns, and create comprehensive descriptions for reproducing the design. Very useful for Bolt.new and other upcoming SaaS

## Features

- Smart image splitting and component detection
- OCR for text extraction
- UI element classification (buttons, text fields, checkboxes, etc.)
- Individual component analysis
- Overall design pattern analysis
- Activity description generation
- Gradio web interface for easy usage

## Installation

This project uses Poetry for dependency management. Follow these steps to set up the project:

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
   Create a `.env` file in the project root and add the following:
   ```
   OPENAI_API_KEY=your_openai_api_key
   OPENROUTER_API_KEY=your_openrouter_api_key
   YOUR_SITE_URL=https://your-app-url.com
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

## Demo
https://github.com/user-attachments/assets/79c2722e-942d-4f0c-84bd-11066b63f4c5

