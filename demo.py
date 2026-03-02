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
import torch.nn as nn
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template_string, request
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_MODEL = os.environ.get("NEXTTOKEN_MODEL") or "HuggingFaceTB/SmolLM2-135M-Instruct"
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


def _abbreviate_tensor(tensor: torch.Tensor, head: int = 6, tail: int = 3, digits: int = 4):
    """
    Return an abbreviated list representation:
      [a, b, c, ..., y, z]
    """
    flat = tensor.detach().cpu().flatten()
    total = flat.numel()
    def _format_number(value: float):
        if digits == 0:
            return int(round(float(value)))
        return round(float(value), digits)

    if total <= head + tail:
        return [_format_number(v) for v in flat.tolist()]
    start = [_format_number(v) for v in flat[:head].tolist()]
    end = [_format_number(v) for v in flat[-tail:].tolist()]
    return start + ["..."] + end

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
# 2D logistic regression classifier
# ---------------------------------------------------------------------------

def _generate_blob_data(n_per_class=50, seed=42):
    """Generate synthetic 2-class data from two Gaussian clusters."""
    gen = torch.Generator().manual_seed(seed)
    x_a = torch.randn(n_per_class, 2, generator=gen) * 0.8 + torch.tensor([-1.0, -1.0])
    x_b = torch.randn(n_per_class, 2, generator=gen) * 0.8 + torch.tensor([1.0, 1.0])
    X = torch.cat([x_a, x_b], dim=0)
    y = torch.cat([torch.zeros(n_per_class), torch.ones(n_per_class)]).long()
    return X, y

_blob_X, _blob_y = _generate_blob_data()
_classifier = nn.Linear(2, 2)
_classifier_trained = False


def _decision_boundary_endpoints():
    """Compute two endpoints of the decision boundary line for plotting.

    The boundary is where w·x + b = 0 for the weight/bias *difference*
    between the two output classes.
    """
    w = _classifier.weight.detach()
    b = _classifier.bias.detach()
    dw = w[1] - w[0]  # (2,)
    db = b[1] - b[0]  # scalar
    # Line: dw[0]*x + dw[1]*y + db = 0  →  y = -(dw[0]*x + db) / dw[1]
    if abs(float(dw[1])) < 1e-8:
        # Nearly vertical boundary
        x_intercept = -float(db) / (float(dw[0]) + 1e-12)
        return [[x_intercept, -4], [x_intercept, 4]]
    x1, x2 = -4.0, 4.0
    y1 = -(float(dw[0]) * x1 + float(db)) / float(dw[1])
    y2 = -(float(dw[0]) * x2 + float(db)) / float(dw[1])
    return [[x1, y1], [x2, y2]]


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
        "inputs": _abbreviate_tensor(inputs["input_ids"][0], digits=0),
        "logits": _abbreviate_tensor(logits),
        "warning": None,
    })


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Generate a chat reply using the loaded model.

    Expects JSON:
      { "messages": [{"role": "user", "content": "..."}, ...],
        "max_new_tokens": 200, "temperature": 0.7 }

    Returns JSON:
      { "reply": "..." }
    """
    if _model is None or _tokenizer is None:
        return jsonify({"error": "Model is still loading. Please wait."}), 503

    body = request.get_json(force=True)
    messages = body.get("messages", [])
    max_new_tokens = min(int(body.get("max_new_tokens", 200)), 512)
    temperature = float(body.get("temperature", 0.7))

    if not messages:
        return jsonify({"error": "No messages provided."}), 400

    if not getattr(_tokenizer, "chat_template", None):
        return jsonify({
            "error": f"Model '{DEFAULT_MODEL}' does not have a chat template. "
                     "Chat requires an instruct-tuned model."
        }), 400

    try:
        # ── Format conversation using the model's chat template ──
        prompt_text = _tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = _tokenizer(prompt_text, return_tensors="pt").to(DEVICE)
        input_len = inputs["input_ids"].shape[1]

        # ── Generate ──
        with torch.no_grad():
            output_ids = _model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=0.9,
                repetition_penalty=1.1,
            )

        # ── Decode only newly generated tokens ──
        new_ids = output_ids[0][input_len:]
        reply = _tokenizer.decode(new_ids, skip_special_tokens=True)

    except Exception as exc:
        return jsonify({"error": f"Generation error: {exc}"}), 500

    return jsonify({"reply": reply})


@app.route("/api/models", methods=["GET"])
def list_models():
    """Return the loaded model name."""
    return jsonify({"models": [DEFAULT_MODEL]})


# ── 2D classifier endpoints ──────────────────────────────────────────────

@app.route("/api/classifier/data", methods=["GET"])
def classifier_data():
    """Return training points and decision boundary for the scatter plot."""
    points = []
    for i in range(_blob_X.size(0)):
        points.append({
            "x": round(float(_blob_X[i, 0]), 4),
            "y": round(float(_blob_X[i, 1]), 4),
            "label": int(_blob_y[i].item()),
        })
    boundary = _decision_boundary_endpoints()
    # Expose all 6 parameters for the neural-net diagram
    w = _classifier.weight.detach()
    b = _classifier.bias.detach()
    weights = {
        "w00": round(float(w[0, 0]), 4),  # x → Class A
        "w01": round(float(w[0, 1]), 4),  # y → Class A
        "w10": round(float(w[1, 0]), 4),  # x → Class B
        "w11": round(float(w[1, 1]), 4),  # y → Class B
        "b0":  round(float(b[0]), 4),      # Class A bias
        "b1":  round(float(b[1]), 4),      # Class B bias
    }
    return jsonify({
        "points": points,
        "boundary": boundary,
        "weights": weights,
        "trained": _classifier_trained,
    })


@app.route("/api/classify", methods=["POST"])
def classify_point():
    """
    Classify a 2D point.

    Expects JSON: { "x": 1.5, "y": -0.3, "temperature": 1.0 }
    Returns JSON: { "probabilities": [{label, prob}], "predicted": 0,
                     "logits": [...], "trained": bool }
    """
    body = request.get_json(force=True)
    x = float(body.get("x", 0))
    y = float(body.get("y", 0))
    temperature = float(body.get("temperature", 1.0))

    try:
        inp = torch.tensor([[x, y]], dtype=torch.float32)
        _classifier.eval()
        with torch.no_grad():
            logits = _classifier(inp)[0]  # shape: (2,)

        if temperature <= 0:
            probs = torch.zeros_like(logits)
            probs[logits.argmax()] = 1.0
        else:
            scaled = logits / temperature
            probs = torch.softmax(scaled, dim=-1)

        labels = ["Class A", "Class B"]
        probabilities = [
            {"label": labels[i], "prob": round(p, 6)}
            for i, p in enumerate(probs.tolist())
        ]
        predicted = int(probs.argmax().item())
        raw_logits = [round(float(v), 4) for v in logits.tolist()]

    except Exception as exc:
        return jsonify({"error": f"Classification error: {exc}"}), 500

    return jsonify({
        "probabilities": probabilities,
        "predicted": predicted,
        "logits": raw_logits,
        "trained": _classifier_trained,
    })


@app.route("/api/classifier/train", methods=["POST"])
def classifier_train():
    """Train the linear classifier with SGD — instant (~10ms)."""
    global _classifier_trained

    _classifier.train()
    optimizer = torch.optim.SGD(_classifier.parameters(), lr=0.5)
    criterion = nn.CrossEntropyLoss()

    steps = 200
    for _ in range(steps):
        optimizer.zero_grad()
        loss = criterion(_classifier(_blob_X), _blob_y)
        loss.backward()
        optimizer.step()

    # Accuracy
    _classifier.eval()
    with torch.no_grad():
        preds = _classifier(_blob_X).argmax(dim=1)
        accuracy = round((preds == _blob_y).float().mean().item(), 4)

    _classifier_trained = True
    return jsonify({"status": "ok", "accuracy": accuracy, "steps": steps})


@app.route("/api/classifier/reset", methods=["POST"])
def classifier_reset():
    """Re-initialize the classifier with random weights."""
    global _classifier_trained
    _classifier.reset_parameters()
    _classifier_trained = False
    return jsonify({"status": "ok"})


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
