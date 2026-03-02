"""
Next-Token Visualization Demo
==============================
A Flask app that loads a HuggingFace model and lets the user explore
the probability distribution interactively, one token at a time.

Run:  uv run demo.py
Open: http://localhost:5005
"""

import base64
import io
import os
import textwrap
import threading

import torch
import torch.nn as nn
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template_string, request
from PIL import Image
from torchvision import datasets, transforms
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_MODEL = os.environ.get("NEXTTOKEN_MODEL", "HuggingFaceTB/SmolLM2-135M-Instruct")
DEVICE_OVERRIDE = os.environ.get("NEXTTOKEN_DEVICE", "auto")
MNIST_DATA_DIR = os.environ.get("NEXTTOKEN_MNIST_DATA", "data")

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
# MNIST digit classifier
# ---------------------------------------------------------------------------

class MnistCNN(nn.Module):
    """Simple 2-layer CNN for MNIST (~60k params)."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))   # (B,16,14,14)
        x = self.pool(torch.relu(self.conv2(x)))    # (B,32,7,7)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

_mnist_model = MnistCNN().to(DEVICE)
_mnist_trained = False
_mnist_training = False
_mnist_lock = threading.Lock()


def _preprocess_canvas_image(base64_str):
    """Decode a base64 data-URL PNG from the canvas into a normalized 28×28 tensor."""
    if "," in base64_str:
        base64_str = base64_str.split(",", 1)[1]
    img_bytes = base64.b64decode(base64_str)
    img = Image.open(io.BytesIO(img_bytes)).convert("L")
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    return transform(img).unsqueeze(0)  # (1,1,28,28)


def _image_to_base64_png(tensor_1x1x28x28):
    """Convert a (1,1,28,28) tensor back to a base64 PNG for preview."""
    img = tensor_1x1x28x28.squeeze().detach().cpu()
    # Undo normalization for display
    img = img * 0.3081 + 0.1307
    img = (img.clamp(0, 1) * 255).byte().numpy()
    pil = Image.fromarray(img, mode="L")
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


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


# ── MNIST endpoints ──────────────────────────────────────────────────────

@app.route("/api/mnist/train", methods=["POST"])
def mnist_train():
    """Train the MNIST CNN for 2 epochs. Thread-locked to prevent concurrent runs."""
    global _mnist_trained, _mnist_training

    with _mnist_lock:
        if _mnist_training:
            return jsonify({"error": "Training is already in progress."}), 409
        _mnist_training = True

    try:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        print("Loading MNIST training data…")
        train_data = datasets.MNIST(MNIST_DATA_DIR, train=True, download=True, transform=transform)
        loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

        _mnist_model.train()
        optimizer = torch.optim.Adam(_mnist_model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        epochs = 2
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}...")
            for images, labels in loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(_mnist_model(images), labels)
                loss.backward()
                optimizer.step()

        # Quick accuracy check on test set
        _mnist_model.eval()
        test_data = datasets.MNIST(MNIST_DATA_DIR, train=False, download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=256)
        correct = total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                preds = _mnist_model(images).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        accuracy = round(correct / total, 4)
        _mnist_trained = True

        return jsonify({"status": "ok", "accuracy": accuracy, "epochs": epochs})
    except Exception as exc:
        return jsonify({"error": f"Training error: {exc}"}), 500
    finally:
        with _mnist_lock:
            _mnist_training = False


@app.route("/api/mnist/reset", methods=["POST"])
def mnist_reset():
    """Re-initialize the MNIST CNN with random weights."""
    global _mnist_trained
    with _mnist_lock:
        if _mnist_training:
            return jsonify({"error": "Cannot reset while training is in progress."}), 409
        _mnist_model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
        _mnist_trained = False
    return jsonify({"status": "ok"})


@app.route("/api/classify", methods=["POST"])
def classify_digit():
    """
    Classify a hand-drawn digit from a canvas image.

    Expects JSON: { "image": "<base64 data-URL PNG>", "temperature": 1.0 }
    Returns JSON: { "probabilities": [{digit, prob}×10], "predicted": 7,
                     "logits": [...], "preview": "<base64 PNG>", "trained": bool }
    """
    body = request.get_json(force=True)
    image_data = body.get("image", "")
    temperature = float(body.get("temperature", 1.0))

    if not image_data:
        return jsonify({"error": "No image data provided."}), 400

    try:
        tensor = _preprocess_canvas_image(image_data).to(DEVICE)
        preview = _image_to_base64_png(tensor)

        _mnist_model.eval()
        with torch.no_grad():
            logits = _mnist_model(tensor)[0]  # shape: (10,)

        # Temperature scaling + softmax (identical to /api/next-token)
        if temperature <= 0:
            probs = torch.zeros_like(logits)
            probs[logits.argmax()] = 1.0
        else:
            scaled = logits / temperature
            probs = torch.softmax(scaled, dim=-1)

        probabilities = [
            {"digit": i, "prob": round(p, 6)}
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
        "preview": preview,
        "trained": _mnist_trained,
    })


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
