# Next-Token Visualizer

Interactive demo that lets you explore how a language model generates text, powered by [HuggingFace Transformers](https://huggingface.co/docs/transformers).

The app has three tabs that build on each other. Start by **chatting** with the model to see what it can do. Then try the **Digit Classifier** to see how a simpler model turns an input into probabilities — with only 10 possible answers, you can see the full picture. Finally, open **Token Explorer** to watch the language model predict text one piece at a time, choosing from 50,000+ candidates at each step.

See also [MicroGpt demo](https://github.com/DrEntropy/microgpt) that goes into more detail on the structure of a transformer model (using character GPT).

## What is a token?

A **token** is the basic unit a language model works with. It's usually a word or part of a word — for example, the word "prediction" might be split into two tokens: "predict" and "ion". Common short words like "the" or "is" are typically one token each.

When you type a prompt, the model's **tokenizer** splits your text into tokens and converts each one to a number (called a **token ID**). The model processes these numbers and produces a probability for every possible next token. The bar chart shows the highest-probability candidates — clicking one appends it to your text and repeats the process.

## Prerequisites

| Requirement | Notes |
|---|---|
| **Python 3.11+** | |
| **uv** | Install from [docs.astral.sh/uv](https://docs.astral.sh/uv/) |

## Quick Start

```bash
uv run demo.py
```

The first run will download the default model (`HuggingFaceTB/SmolLM2-135M-Instruct`, ~270 MB). Subsequent runs use the cached model.

Open **http://localhost:5005** in your browser.

### Docker (optional)

If you prefer not to install Python dependencies locally, you can run with Docker:

```bash
docker compose up --build
```

The first run builds the image (~600 MB) and downloads the model (~270 MB). Model weights and MNIST data are stored in a Docker volume so subsequent starts are fast.

Open **http://localhost:5005** in your browser.

To use a different model:

```bash
NEXTTOKEN_MODEL=HuggingFaceTB/SmolLM2-360M-Instruct docker compose up --build
```

## Configuration

| Environment variable | Default | Description |
|---|---|---|
| `NEXTTOKEN_MODEL` | `HuggingFaceTB/SmolLM2-135M-Instruct` | HuggingFace model ID to load |
| `NEXTTOKEN_DEVICE` | `auto` | Inference device: `auto` (CUDA > MPS > CPU), `cpu`, `cuda`, `mps` |

To see which models you already have cached locally:

```bash
uv run list_models.py
```

This prints model IDs ready to copy-paste as the `NEXTTOKEN_MODEL` value.

Example with a different model:

```bash
NEXTTOKEN_MODEL=HuggingFaceTB/SmolLM2-360M-Instruct uv run demo.py
```

Example forcing CPU:

```bash
NEXTTOKEN_DEVICE=cpu uv run demo.py
```

## UI Controls

The three tabs are meant to be explored in order: **Chat → Digit Classifier → Token Explorer**.

### Chat (start here)

Chat with the model to see what it can do. Under the hood, the model generates its reply one token at a time — the other tabs let you see that process. **Requires an instruct-tuned model** (e.g. one with `-Instruct` in the name).

- **Max reply tokens** — Cap on generated tokens per reply (1–512)
- **Temperature** — Controls randomness (0 = predictable, higher = more creative)
- **Clear Chat** — Reset the conversation history

### Digit Classifier (peek behind the curtain)

Now see how a model makes decisions. Draw a digit, and a small neural network scores each possible answer (0–9) and shows the probabilities as a bar chart. With only 10 possible outputs, you can see the full distribution at a glance. The model starts with random weights — train it and watch predictions go from random to confident.

- **Train Model** — Train the CNN for 2 epochs on MNIST (~3–5s on CPU). Downloads the dataset (~11 MB) on first run.
- **Canvas** — Draw a digit (0–9) with your mouse or touchscreen
- **Classify** — Run the drawn digit through the CNN and display a probability bar chart
- **Clear** — Reset the canvas
- **Temperature** — Controls how spread out the probabilities are (same concept as Chat)

### Token Explorer (the full picture)

The same idea as the Digit Classifier, but now applied to text. Enter a prompt and see which tokens the model thinks should come next — except now there are 50,000+ candidates instead of 10. Click any bar to append that token and repeat.

- **Model** — Shows the loaded model (read-only; change via `NEXTTOKEN_MODEL` env var)
- **Top-K** — Number of candidate tokens to display (1–50)
- **Temperature** — Sampling temperature (0.0–2.0; 0 = greedy/deterministic)
- **Start** — Initialize with the current prompt and fetch first candidates
- **Greedy** — Auto-pick the highest-probability candidate
- **Reset** — Clear everything and start over

## Glossary

| Term | Definition |
|---|---|
| **Token** | A word or word-fragment — the basic unit a language model reads and generates |
| **Tokenizer** | Converts text into a list of token IDs (numbers) that the model can process |
| **Logits** | Raw scores the model outputs for each possible next token — higher means more likely |
| **Softmax** | A function that converts logits into probabilities that sum to 1 (0–100%) |
| **Temperature** | Controls randomness: 0 = always pick the most likely token, higher = more random/creative |
| **Top-K** | How many of the highest-probability candidates to show |
| **Forward pass** | One run of the model: feed in tokens, get out logits |
| **Greedy** | Picking the single most likely token at each step (temperature = 0 has the same effect) |
| **MNIST** | A classic dataset of 70,000 handwritten digit images (0–9), used to train and test classifiers |

## Troubleshooting

### Model download is slow

The first run downloads the model from HuggingFace Hub. For the default 135M model this is ~270 MB. You can use a larger model:

```bash
NEXTTOKEN_MODEL=HuggingFaceTB/SmolLM2-360M-Instruct uv run demo.py
```

### Out of memory

The default 135M model runs on CPU with float32 (~0.6 GB RAM). If you're low on memory, it's already the smallest available. On GPU/MPS the model uses float16 automatically.

### First query is slow

The first inference after startup may be slower as the model warms up. Subsequent queries are faster.

### Tokens look strange (special characters)

The visualizer replaces leading spaces with `␠` so whitespace is visible. The actual token used when you click is the original (with the real space). Newlines show as `↵` and tabs as `→`.

---

## How This Was Built

This project was vibe-coded using [Claude Code](https://claude.com/claude-code). The initial project was created from a single prompt similar to [`prompt.md`](prompt.md) but the original used Ollama instead of HuggingFace. I switched to HuggingFace for easier local testing and broader model compatibility, but the core idea is the same. The Chat tab and Digit Classifier tab were added later, also with Claude Code's help but are not reflected in this prompt.

The prompt itself was created by chatgpt (5.2) with a conversation on what i was trying to create. A simple prompt like this would work for a start I think:

```
create a prompt for a coding agent like Claude Code to create a demo using flask and huggingface with probability bars and the ability of the user to select a token on the graph and then the next token appears and so on. Default model SmolLM2-360M-Instruct. Use uv for package management, keep it to a single-file backend and single HTML template. Include a CLAUDE.md for AI-assisted development context.
```

Note added 2025-02-27: The above applies to the version tagged `v0.1.0`. The current `main` branch has diverged a bit since then.


