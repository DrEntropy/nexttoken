"""
Next-Token Visualization Demo
==============================
A Flask app that loads a HuggingFace model and lets the user explore
the probability distribution interactively, one token at a time.

Run:  uv run demo.py
Open: http://localhost:5005
"""

import os
import textwrap

import torch
from flask import Flask, jsonify, render_template_string, request
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_MODEL = os.environ.get("NEXTTOKEN_MODEL", "HuggingFaceTB/SmolLM2-360M-Instruct")
DEVICE_OVERRIDE = os.environ.get("NEXTTOKEN_DEVICE", "auto")

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------

def _resolve_device(override: str) -> str:
    if override != "auto":
        return override
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = _resolve_device(DEVICE_OVERRIDE)

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
_tokenizer = None
_model = None

def _load_model():
    global _tokenizer, _model
    dtype = torch.float32 if DEVICE == "cpu" else torch.float16
    print(f"Loading model {DEFAULT_MODEL} on {DEVICE} ({dtype})…")
    _tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    _model = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL, dtype=dtype)
    _model.to(DEVICE)
    _model.eval()
    print("Model loaded.")

_load_model()

# ---------------------------------------------------------------------------
# HTML template (served inline — single-file simplicity)
# ---------------------------------------------------------------------------
HTML_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "templates", "index.html")

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Serve the single-page UI."""
    with open(HTML_TEMPLATE_PATH) as f:
        html = f.read()
    return render_template_string(html, default_model=DEFAULT_MODEL)


@app.route("/api/next-token", methods=["POST"])
def next_token():
    """
    Compute next-token predictions using the loaded HuggingFace model.

    Expects JSON:
      { "text": "...", "top_k": 10, "temperature": 0.7 }

    Returns JSON:
      { "candidates": [ {"token": "...", "prob": 0.42}, ... ],
        "sampled": "...",
        "warning": null }
    """
    if _model is None or _tokenizer is None:
        return jsonify({"error": "Model is still loading. Please wait."}), 503

    body = request.get_json(force=True)
    prompt_text = body.get("text", "")
    top_k = int(body.get("top_k", 10))
    temperature = float(body.get("temperature", 0.7))

    if not prompt_text:
        return jsonify({"error": "Prompt text is empty."}), 400

    try:
        # ── Tokenize ──
        inputs = _tokenizer(prompt_text, return_tensors="pt").to(DEVICE)

        # ── Forward pass ──
        with torch.no_grad():
            outputs = _model(**inputs)

        # ── Extract last-position logits ──
        logits = outputs.logits[0, -1, :]  # shape: (vocab_size,)

        # ── Temperature scaling + softmax ──
        if temperature <= 0:
            # Greedy: argmax, uniform probs over top-k for display
            probs = torch.zeros_like(logits)
            probs[logits.argmax()] = 1.0
        else:
            scaled = logits / temperature
            probs = torch.softmax(scaled, dim=-1)

        # ── Top-K candidates ──
        top_probs, top_indices = torch.topk(probs, k=min(top_k, probs.size(0)))
        candidates = []
        for p, idx in zip(top_probs.tolist(), top_indices.tolist()):
            token_str = _tokenizer.decode([idx])
            candidates.append({"token": token_str, "prob": round(p, 6)})

        # ── Sample from full distribution ──
        if temperature <= 0:
            sampled_idx = logits.argmax().item()
        else:
            sampled_idx = torch.multinomial(probs, num_samples=1).item()
        sampled_token = _tokenizer.decode([sampled_idx])

    except Exception as exc:
        return jsonify({"error": f"Inference error: {exc}"}), 500

    return jsonify({
        "candidates": candidates,
        "sampled": sampled_token,
        "warning": None,
    })


@app.route("/api/models", methods=["GET"])
def list_models():
    """Return the loaded model name."""
    return jsonify({"models": [DEFAULT_MODEL]})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print(textwrap.dedent(f"""\
        ╔══════════════════════════════════════════╗
        ║   Next-Token Visualization Demo          ║
        ║   http://localhost:5005                   ║
        ║   Model: {DEFAULT_MODEL:<31s} ║
        ║   Device: {DEVICE:<30s} ║
        ╚══════════════════════════════════════════╝
    """))
    app.run(host="0.0.0.0", port=5005, debug=True, use_reloader=False)


if __name__ == "__main__":
    main()
