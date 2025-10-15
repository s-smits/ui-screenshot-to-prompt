# ui-screenshot-to-prompt

Turn a UI screenshot into an implementation brief. The pipeline detects layout regions, runs OCR where it matters, and assembles a prompt you can hand to an AI coding copilot.

## Demo
https://github.com/user-attachments/assets/79c2722e-942d-4f0c-84bd-11066b63f4c5

## What it does
- Basic mode: deterministic grid slicing for fast spatial summaries.
- Advanced mode: EasyOCR + OpenCV to pull out likely UI components (still experimental; expect false positives).
- Aggregated analysis: OpenAI vision models describe each region, then a Claude/OpenRouter pass crafts the final build instructions.
- Gradio front-end so designers can drag-and-drop without touching the CLI.

## Getting started
```bash
git clone https://github.com/s-smits/ui-screenshot-to-prompt.git
cd ui-screenshot-to-prompt
uv venv .venv
uv sync
cp .env.example .env  # fill in OPENAI / optional ANTHROPIC or OPENROUTER keys
```

Run locally:
```bash
uv run python -m ui_screenshot_to_prompt.main
```

## Configuration highlights
- `ui_screenshot_to_prompt/config.py` – detection mode defaults, prompt templates, component thresholds.
- `.env` – choose which LLM vendors to call; OpenAI Vision is required, Anthropic/OpenRouter is optional but improves the final prompt.

## Notes
- The project is CPU-friendly; EasyOCR will download weights on the first run.
- Advanced detection is under active tuning—log output will tell you what it caught.
- Pytest suites live under `tests/`; run them with `uv run pytest` if you touch the core pipeline.

## Contributing
Issues and PRs are welcome. Aim for pragmatic fixes and keep prompts reversible for future tuning.

## License
MIT

[![Star History Chart](https://api.star-history.com/svg?repos=s-smits/ui-screenshot-to-prompt&type=Date)](https://star-history.com/#s-smits/ui-screenshot-to-prompt&Date)
