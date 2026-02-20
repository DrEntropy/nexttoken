Build a local, self-contained interactive “next token” visualization demo using ONLY Ollama as the model runner.

GOAL
Create a web app that:
1) Accepts an initial prompt from the user.
2) Queries Ollama for the next-token candidates (top-K tokens + probabilities).
3) Displays candidates as probability bars.
4) Lets the user click a candidate token; when clicked:
   - append that token to the current text
   - re-query Ollama for the next-token distribution
   - update the bars
5) Repeat step-by-step.

HARD CONSTRAINTS
- Model runner: Ollama ONLY (no OpenAI API, no external services).
- Runs locally via uv run
- No cloud keys required.
- Minimal dependencies and clear setup.

TECH STACK
- Package management: uv
- Python 3.11 + Flask + ollama python library
- Ollama model:  gemma3, but make it configurable via environment variable

IMPORTANT NOTE ABOUT PROBABILITIES
Ollama can return per-token logprobs for generated tokens and (depending on model/runtime) can expose top candidates when enabled. Implement against Ollama’s /api/generate endpoint with:
- options.num_predict = 1 (one token at a time)
- stream = false
- raw = true (so prompt templating does not interfere; provide a UI toggle)
- logprobs = true
- top_logprobs = K

If top candidates are unavailable for a model, the app MUST degrade gracefully:
- Show only the sampled token as a single bar with probability 1.0
- Display a clear warning in the UI: “Top-K candidate probabilities not available for this model/runner. Showing only sampled token.”

FUNCTIONAL REQUIREMENTS

A) UI (single page)
- Input textarea: initial prompt
- Read-only textarea: “current text” (updates as tokens are appended)
- Controls:
  - model name (text input, default from env)
  - top_k (number input, default 10)
  - temperature (0.0–2.0, default 0.7)
  - raw mode toggle (default ON)
- Buttons:
  - Start (initializes state)
  - Greedy (optional): if top candidates exist, pick argmax automatically
  - Reset (clears state)
- Visualization:
  - Horizontal bar chart (preferred) with candidate tokens and probabilities
  - Token labels must make whitespace visible:
    - replace leading space with “␠” for display only (do not modify the actual token used)
  - Show probability numeric values (tooltip or inline)
 
B) README
Include:
- prerequisites (Docker) and ollama
- how to run: uv run demo.py
- open http://localhost:5005
- troubleshooting section:
  - model not found
  - top_k unavailable warning meaning
  - slow first token due to model load

 

QUALITY BAR
- Clean, readable code with comments.
- Good error handling and user-friendly messages.
- Avoid heavy frameworks; keep it simple and teachable.

Now implement the full project end-to-end with the above constraints and deliver all files.