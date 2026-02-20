# Next-Token Visualizer

## Project Overview

Interactive single-page web app that visualizes next-token predictions using a HuggingFace Transformers model loaded in-process. Users enter a prompt, see top-K candidate tokens as a probability bar chart, click to append a token, and repeat — building text one token at a time.

## Architecture

- **Backend**: `demo.py` — Flask app (port 5005)
- **Frontend**: `templates/index.html` — single HTML file with inline CSS/JS, no build step
- **Model loading**: uses `transformers` (`AutoModelForCausalLM` / `AutoTokenizer`) to load a HuggingFace model directly into the process — no external service required

### Key API flow

1. Frontend `POST /api/next-token` with `{text, top_k, temperature}`
2. Backend tokenizes input, runs a forward pass through the model, extracts last-position logits
3. Backend applies temperature scaling, computes softmax, takes top-K candidates
4. Frontend renders horizontal bar chart; click appends token and re-queries

### Model loading

The model is loaded at module level (guarded by `WERKZEUG_RUN_MAIN` to avoid double-loading with Flask's reloader). First run downloads the model from HuggingFace Hub (~720 MB for the default model). `use_reloader=False` is set in `app.run()` to prevent double loading.

## Tech Stack

- Python 3.11+, Flask, torch, transformers
- Package management: `uv`
- Run: `uv run demo.py`
- No frontend build tools — vanilla HTML/CSS/JS with Jinja2 templating (only `{{ default_model }}`)

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `NEXTTOKEN_MODEL` | `HuggingFaceTB/SmolLM2-360M-Instruct` | HuggingFace model ID |
| `NEXTTOKEN_DEVICE` | `auto` | Device for inference (`auto`, `cpu`, `cuda`, `mps`) |

## File Structure

```
demo.py              # Flask app — model loading, inference, routes, entry point
templates/index.html # Single-page UI (HTML + inline CSS + inline JS)
pyproject.toml       # uv/pip project config
README.md            # User-facing setup & troubleshooting
```

## Code Conventions

- Keep it single-file backend (`demo.py`) — don't split into modules unless it grows significantly
- Frontend stays as a single HTML file with inline styles and scripts — no npm, no bundler
- Comments use `# ── Section ──` banner style for visual separation
- Error responses from `/api/next-token` return `{"error": "..."}` with appropriate HTTP status codes (400, 500, 503)

## Common Tasks

- **Run locally**: `uv run demo.py` (opens on http://localhost:5005, first run downloads model)
- **Change default model**: `NEXTTOKEN_MODEL=HuggingFaceTB/SmolLM2-135M-Instruct uv run demo.py`
- **Force CPU**: `NEXTTOKEN_DEVICE=cpu uv run demo.py`
- **Test import**: `uv run python -c "import demo; print('OK')"`
