Build a local, self-contained interactive “next token” visualization demo using HuggingFace Transformers for in-process inference.

GOAL
Create a web app that:
1) Accepts an initial prompt from the user.
2) Runs a forward pass through a HuggingFace model to get next-token candidates (top-K tokens + probabilities).
3) Displays candidates as probability bars.
4) Lets the user click a candidate token; when clicked:
   - append that token to the current text
   - re-run inference for the next-token distribution
   - update the bars
5) Repeat step-by-step.

HARD CONSTRAINTS
- No external services — model runs in-process via torch + transformers.
- Runs locally via uv run
- No cloud keys required.
- Minimal dependencies and clear setup.

TECH STACK
- Package management: uv
- Python 3.11 + Flask + torch + transformers
  (load model with AutoModelForCausalLM / AutoTokenizer, compute logits directly)
- Single HTML file frontend (templates/index.html), no build tools
- Default model: HuggingFaceTB/SmolLM2-360M-Instruct, configurable via NEXTTOKEN_MODEL env var
- Device: auto-detect CUDA > MPS > CPU, configurable via NEXTTOKEN_DEVICE env var (default “auto”)

INFERENCE APPROACH
- Tokenize the prompt, run a forward pass (torch.no_grad), extract last-position logits
- Apply temperature scaling (divide logits by temperature), then torch.softmax to get probabilities
- Use torch.topk to get top-K candidates; decode token IDs to strings via tokenizer.decode
- Sample from the full distribution via torch.multinomial for the “sampled” field
- Handle temperature=0 as greedy (argmax, no division)
- Load model at module level; use float16 on GPU/MPS, float32 on CPU; call model.eval()
- Set use_reloader=False in app.run() to prevent double model loading

FUNCTIONAL REQUIREMENTS

A) UI (single page)
- Input textarea: initial prompt
- Read-only textarea: “current text” (updates as tokens are appended)
- Controls:
  - model name (read-only text input showing loaded model, with tooltip explaining NEXTTOKEN_MODEL env var)
  - num_candidates (number input, default 10) — how many top-K candidates to display
  - temperature (0.0–2.0, default 0.7)
- Buttons:
  - Start (initializes state)
  - Greedy: if top candidates exist, pick argmax automatically
  - Reset (clears state)
- Visualization:
  - Horizontal bar chart (preferred) with candidate tokens and probabilities
  - Bars should be clickable (clicking appends the token). Use a clean, minimal color scheme.
  - Token labels must make whitespace visible:
    - replace leading space with “␠” for display only (do not modify the actual token used)
  - Show probability numeric values inline next to each bar

B) Backend API
- POST /api/next-token: accepts {text, top_k, temperature}, returns {candidates, sampled, warning}
- GET /api/models: returns the single loaded model name
- Serve the frontend from templates/index.html using Flask’s render_template.
- Error codes: 400 (empty prompt), 500 (inference error), 503 (model still loading)

C) README
Include:
- prerequisites: Python 3.11+, uv
- how to run: uv run demo.py (first run downloads model)
- open http://localhost:5005
- troubleshooting section:
  - model download slow
  - out of memory
  - slow first inference

D) CLAUDE.md
Generate a CLAUDE.md file documenting: architecture, key API flow, tech stack, env vars, file structure, code conventions, and common tasks. This serves as context for future AI-assisted development.

 

QUALITY BAR
- Clean, readable code with comments.
- Good error handling and user-friendly messages.
- Avoid heavy frameworks; keep it simple and teachable.

Now implement the full project end-to-end with the above constraints and deliver all files.