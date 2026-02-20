# Next-Token Visualizer

Interactive demo that lets you explore language model predictions **one token at a time**, powered by [Ollama](https://ollama.com).

Enter a prompt, hit **Start**, and the app queries Ollama for the top-K next-token candidates with their probabilities. Click any candidate to append it and see the next distribution — repeat to build text step by step.

## Prerequisites

| Requirement | Notes |
|---|---|
| **Ollama** | Install from [ollama.com](https://ollama.com). Must be running (`ollama serve`). |
| **A pulled model** | Default is `gemma3`. Pull it first: `ollama pull gemma3` |
| **Python 3.11+** | |
| **uv** | Install from [docs.astral.sh/uv](https://docs.astral.sh/uv/) |

## Quick Start

```bash
# 1. Make sure Ollama is running
ollama serve          # if not already running

# 2. Pull the model (one-time)
ollama pull gemma3

# 3. Run the demo
uv run demo.py
```

Open **http://localhost:5005** in your browser.

## Configuration

| Environment variable | Default | Description |
|---|---|---|
| `NEXTTOKEN_MODEL` | `gemma3` | Default model shown in the UI |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama API base URL |

Example with a different model:

```bash
NEXTTOKEN_MODEL=llama3.2 uv run demo.py
```

## UI Controls

- **Model** — Ollama model name (editable at runtime)
- **Top-K** — Number of candidate tokens to request (1–50)
- **Temperature** — Sampling temperature (0.0–2.0)
- **Raw mode** — When ON, sends the prompt without model-specific chat templating (`raw: true`). Recommended for this demo.
- **Start** — Initialize with the current prompt and fetch first candidates
- **Greedy** — Auto-pick the highest-probability candidate
- **Reset** — Clear everything and start over

## Troubleshooting

### "Cannot connect to Ollama. Is it running?"

Ollama must be running in the background. Start it with:

```bash
ollama serve
```

If Ollama is on a different host/port, set `OLLAMA_HOST`:

```bash
OLLAMA_HOST=http://192.168.1.10:11434 uv run demo.py
```

### "model 'xyz' not found"

You need to pull the model first:

```bash
ollama pull gemma3
```

### "Top-K candidate probabilities not available"

This warning means the model or your version of Ollama does not support the `logprobs` / `top_logprobs` API parameters. The app degrades gracefully and shows only the single sampled token.

**Fix:** Update Ollama to the latest version (`ollama update` or reinstall). The `logprobs` feature requires Ollama **v0.12.11+**.

### First query is very slow

The first request after starting Ollama (or switching models) triggers model loading into GPU/CPU memory. This is a one-time cost per model — subsequent queries are fast.

### Tokens look strange (special characters)

The visualizer replaces leading spaces with `␠` so whitespace is visible. The actual token used when you click is the original (with the real space). Newlines show as `↵` and tabs as `→`.

---

## How This Was Built

This project was vibe-coded using [Claude Code](https://claude.com/claude-code) from a single prompt. See [`PROMPT.md`](PROMPT.md) for the full prompt used.

The prompt itself was created by chatgpt (5.2) with a conversation on what i was trying to create. I did some light editing to align with what I really wanted.  A simple prompt like this would work for a start I think:

```
create a prompt for a coding agent like Claude Code to create a demo using flask and Ollama running locally with the probability bars and the ability of the user to select a token on the graph and then the next token appears and so on. Default model gemma3.
```

