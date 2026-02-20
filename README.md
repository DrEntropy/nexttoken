# Next-Token Visualizer

Interactive demo that lets you explore language model predictions **one token at a time**, powered by [HuggingFace Transformers](https://huggingface.co/docs/transformers).

Enter a prompt, hit **Start**, and the app runs a forward pass through a local model to get the top-K next-token candidates with their probabilities. Click any candidate to append it and see the next distribution — repeat to build text step by step.

## Prerequisites

| Requirement | Notes |
|---|---|
| **Python 3.11+** | |
| **uv** | Install from [docs.astral.sh/uv](https://docs.astral.sh/uv/) |

## Quick Start

```bash
uv run demo.py
```

The first run will download the default model (`HuggingFaceTB/SmolLM2-360M-Instruct`, ~720 MB). Subsequent runs use the cached model.

Open **http://localhost:5005** in your browser.

## Configuration

| Environment variable | Default | Description |
|---|---|---|
| `NEXTTOKEN_MODEL` | `HuggingFaceTB/SmolLM2-360M-Instruct` | HuggingFace model ID to load |
| `NEXTTOKEN_DEVICE` | `auto` | Inference device: `auto` (CUDA > MPS > CPU), `cpu`, `cuda`, `mps` |

To see which models you already have cached locally:

```bash
uv run list_models.py
```

This prints model IDs ready to copy-paste as the `NEXTTOKEN_MODEL` value.

Example with a different model:

```bash
NEXTTOKEN_MODEL=HuggingFaceTB/SmolLM2-135M-Instruct uv run demo.py
```

Example forcing CPU:

```bash
NEXTTOKEN_DEVICE=cpu uv run demo.py
```

## UI Controls

The app has two tabs: **Token Explorer** and **Chat**.

### Token Explorer

- **Model** — Shows the loaded model (read-only; change via `NEXTTOKEN_MODEL` env var)
- **Top-K** — Number of candidate tokens to display (1–50)
- **Temperature** — Sampling temperature (0.0–2.0; 0 = greedy/deterministic)
- **Start** — Initialize with the current prompt and fetch first candidates
- **Greedy** — Auto-pick the highest-probability candidate
- **Reset** — Clear everything and start over

### Chat

A simple multi-turn chat interface that lets you converse with the loaded model. **Requires an instruct-tuned model** (e.g. one with `-Instruct` in the name). If you load a base model, the Chat tab will show an error explaining that a chat template is needed.

- **Max reply tokens** — Cap on generated tokens per reply (1–512)
- **Temperature** — Sampling temperature for chat responses
- **Clear Chat** — Reset the conversation history

## Troubleshooting

### Model download is slow

The first run downloads the model from HuggingFace Hub. For the default 360M model this is ~720 MB. You can use a smaller model:

```bash
NEXTTOKEN_MODEL=HuggingFaceTB/SmolLM2-135M-Instruct uv run demo.py
```

### Out of memory

The default model runs on CPU with float32 (~1.5 GB RAM). If you're low on memory, try a smaller model. On GPU/MPS the model uses float16 automatically.

### First query is slow

The first inference after startup may be slower as the model warms up. Subsequent queries are faster.

### Tokens look strange (special characters)

The visualizer replaces leading spaces with `␠` so whitespace is visible. The actual token used when you click is the original (with the real space). Newlines show as `↵` and tabs as `→`.

---

## How This Was Built

This project was vibe-coded using [Claude Code](https://claude.com/claude-code). The initial project was created from a single prompt similar to [`PROMPT.md`](PROMPT.md) but the original used Ollama instead of HuggingFace. I switched to HuggingFace for easier local testing and broader model compatibility, but the core idea is the same. The Chat tab was also added later, also with Claude Code's help but that is not reflected in this prompt.

The prompt itself was created by chatgpt (5.2) with a conversation on what i was trying to create. A simple prompt like this would work for a start I think:

```
create a prompt for a coding agent like Claude Code to create a demo using flask and huggingface with probability bars and the ability of the user to select a token on the graph and then the next token appears and so on. Default model SmolLM2-360M-Instruct. Use uv for package management, keep it to a single-file backend and single HTML template. Include a CLAUDE.md for AI-assisted development context.
```


