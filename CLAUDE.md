# Next-Token Visualizer

## Project Overview

Interactive single-page web app that visualizes next-token predictions from a local Ollama instance. Users enter a prompt, see top-K candidate tokens as a probability bar chart, click to append a token, and repeat — building text one token at a time.

## Architecture

- **Backend**: `demo.py` — Flask app (port 5005)
- **Frontend**: `templates/index.html` — single HTML file with inline CSS/JS, no build step
- **No `ollama` Python library** — uses `requests` to call Ollama's REST API directly at `/api/generate`, because the Python library does not reliably support `logprobs`/`top_logprobs`

### Key API flow

1. Frontend `POST /api/next-token` with `{text, model, top_k, temperature, raw}`
2. Backend forwards to Ollama `/api/generate` with `num_predict=1`, `stream=false`, `logprobs=true`, `top_logprobs=K`
3. Backend converts log-probabilities to probabilities via `math.exp(logprob)`, returns sorted candidates
4. Frontend renders horizontal bar chart; click appends token and re-queries

### Graceful degradation

If Ollama doesn't return `top_logprobs`, the backend falls back to showing only the sampled token with a UI warning. This path is tested by the `else` branches in the logprobs parsing section of `next_token()`.

## Tech Stack

- Python 3.11+, Flask, requests
- Package management: `uv`
- Run: `uv run demo.py`
- No frontend build tools — vanilla HTML/CSS/JS with Jinja2 templating (only `{{ default_model }}`)

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `NEXTTOKEN_MODEL` | `gemma3` | Default model name |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama API base URL |

## File Structure

```
demo.py              # Flask app — routes, Ollama API calls, entry point
templates/index.html # Single-page UI (HTML + inline CSS + inline JS)
pyproject.toml       # uv/pip project config
README.md            # User-facing setup & troubleshooting
```

## Code Conventions

- Keep it single-file backend (`demo.py`) — don't split into modules unless it grows significantly
- Frontend stays as a single HTML file with inline styles and scripts — no npm, no bundler
- Use `requests` (not the `ollama` library) for Ollama API calls
- Comments use `# ── Section ──` banner style for visual separation
- Error responses from `/api/next-token` return `{"error": "..."}` with appropriate HTTP status codes (502, 504)

## Common Tasks

- **Run locally**: `uv run demo.py` (opens on http://localhost:5005)
- **Change default model**: `NEXTTOKEN_MODEL=llama3.2 uv run demo.py`
- **Test import**: `uv run python -c "import demo; print('OK')"`
