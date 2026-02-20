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
- Python 3.11 + Flask + requests
  (call Ollama REST API directly at /api/generate — do NOT use the
   ollama Python library, it doesn't reliably support logprobs/top_logprobs)
- Single HTML file frontend (templates/index.html), no build tools
- Ollama model: gemma3, configurable via NEXTTOKEN_MODEL env var
- Ollama host: configurable via OLLAMA_HOST env var (default http://localhost:11434)

IMPORTANT NOTE ABOUT PROBABILITIES
Ollama can return per-token logprobs for generated tokens and (depending on model/runtime) can expose top candidates when enabled. Implement against Ollama’s /api/generate endpoint with:
- options.num_predict = 1 (one token at a time)
- stream = false
- raw = true (so prompt templating does not interfere; provide a UI toggle)
- logprobs = true
- top_logprobs = num_candidates (number of candidate tokens to return, separate from sampling top_k)

If top candidates are unavailable for a model, the app MUST degrade gracefully:
- Show only the sampled token as a single bar with probability 1.0
- Display a clear warning in the UI: “Top-K candidate probabilities not available for this model/runner. Showing only sampled token.”

FUNCTIONAL REQUIREMENTS

A) UI (single page)
- Input textarea: initial prompt
- Read-only textarea: “current text” (updates as tokens are appended)
- Controls:
  - model name (text input, default from env)
  - num_candidates (number input, default 10) — controls top_logprobs (how many candidate tokens to display); this is separate from Ollama's sampling top_k
  - temperature (0.0–2.0, default 0.7)
  - raw mode toggle (default ON)
- Buttons:
  - Start (initializes state)
  - Greedy (optional): if top candidates exist, pick argmax automatically
  - Reset (clears state)
- Visualization:
  - Horizontal bar chart (preferred) with candidate tokens and probabilities
  - Bars should be clickable (clicking appends the token). Use a clean, minimal color scheme.
  - Token labels must make whitespace visible:
    - replace leading space with “␠” for display only (do not modify the actual token used)
  - Show probability numeric values inline next to each bar
 
B) Backend API
- Include a GET /api/models endpoint that queries Ollama's /api/tags so the UI can offer a model dropdown.
- Serve the frontend from templates/index.html using Flask's render_template.

C) README
Include:
- prerequisites: Ollama installed and running
- how to run: uv run demo.py
- open http://localhost:5005
- troubleshooting section:
  - model not found
  - top_k unavailable warning meaning
  - slow first token due to model load

D) CLAUDE.md
Generate a CLAUDE.md file documenting: architecture, key API flow, tech stack, env vars, file structure, code conventions, and common tasks. This serves as context for future AI-assisted development.

 

QUALITY BAR
- Clean, readable code with comments.
- Good error handling and user-friendly messages.
- Avoid heavy frameworks; keep it simple and teachable.

Now implement the full project end-to-end with the above constraints and deliver all files.