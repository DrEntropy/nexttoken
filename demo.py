"""
Next-Token Visualization Demo
==============================
A Flask app that queries Ollama one token at a time and lets the user
explore the probability distribution interactively.

Run:  uv run demo.py
Open: http://localhost:5005
"""

import math
import os
import textwrap

import requests
from flask import Flask, jsonify, render_template_string, request

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OLLAMA_BASE = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("NEXTTOKEN_MODEL", "gemma3")

app = Flask(__name__)

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
    Ask Ollama for the next token given the current text.

    Expects JSON:
      { "text": "...", "model": "gemma3", "top_k": 10,
        "temperature": 0.7, "raw": true }

    Returns JSON:
      { "candidates": [ {"token": "...", "prob": 0.42}, ... ],
        "sampled": "...",
        "warning": null | "..." }
    """
    body = request.get_json(force=True)
    prompt_text = body.get("text", "")
    model = body.get("model", DEFAULT_MODEL)
    top_k = int(body.get("top_k", 10))
    temperature = float(body.get("temperature", 0.7))
    raw = bool(body.get("raw", True))

    # Build request payload for Ollama /api/generate
    payload = {
        "model": model,
        "prompt": prompt_text,
        "stream": False,
        "raw": raw,
        "options": {
            "num_predict": 1,
            "temperature": temperature,
            "top_k": top_k,
        },
        "logprobs": True,
        "top_logprobs": top_k,
    }

    try:
        resp = requests.post(
            f"{OLLAMA_BASE}/api/generate",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
    except requests.ConnectionError:
        return jsonify({"error": "Cannot connect to Ollama. Is it running?"}), 502
    except requests.Timeout:
        return jsonify({"error": "Ollama request timed out."}), 504
    except requests.HTTPError as exc:
        return jsonify({"error": f"Ollama error: {exc.response.text}"}), 502

    data = resp.json()
    sampled_token = data.get("response", "")

    # Parse logprobs ----------------------------------------------------------
    warning = None
    candidates = []

    logprobs_list = data.get("logprobs")

    if logprobs_list and len(logprobs_list) > 0:
        entry = logprobs_list[0]  # we requested num_predict=1
        top = entry.get("top_logprobs")
        if top and len(top) > 0:
            # Convert log-probabilities to probabilities
            for item in top:
                token = item.get("token", "")
                logprob = item.get("logprob", 0.0)
                prob = math.exp(logprob)
                candidates.append({"token": token, "prob": round(prob, 6)})
            # Sort descending by probability
            candidates.sort(key=lambda c: c["prob"], reverse=True)
        else:
            # top_logprobs not available — fallback to sampled token
            logprob = entry.get("logprob", 0.0)
            prob = math.exp(logprob)
            candidates = [{"token": sampled_token, "prob": round(prob, 6)}]
            warning = (
                "Top-K candidate probabilities not available for this "
                "model/runner. Showing only sampled token."
            )
    else:
        # No logprobs at all — full fallback
        candidates = [{"token": sampled_token, "prob": 1.0}]
        warning = (
            "Top-K candidate probabilities not available for this "
            "model/runner. Showing only sampled token."
        )

    return jsonify({
        "candidates": candidates,
        "sampled": sampled_token,
        "warning": warning,
    })


@app.route("/api/models", methods=["GET"])
def list_models():
    """Return available Ollama models for the UI dropdown helper."""
    try:
        resp = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=10)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        return jsonify({"models": models})
    except Exception as exc:
        return jsonify({"models": [], "error": str(exc)})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print(textwrap.dedent(f"""\
        ╔══════════════════════════════════════════╗
        ║   Next-Token Visualization Demo          ║
        ║   http://localhost:5005                   ║
        ║   Model: {DEFAULT_MODEL:<31s} ║
        ║   Ollama: {OLLAMA_BASE:<30s} ║
        ╚══════════════════════════════════════════╝
    """))
    app.run(host="0.0.0.0", port=5005, debug=True)


if __name__ == "__main__":
    main()
