# Karllo AI

Independent decoder-only Transformer platform: **real PyTorch training**, checkpoints, datasets, chat, memory, and optional retrieval.

Karllo does **not** call Gemini or any hidden hosted chatbot. External APIs are optional, labeled, and off by default.

## Inspection (this repository)

1. The starting tree contained only a stub `README.md` (`# UrBots7`). No model, tokenizer, checkpoint, or UI existed.
2. Nothing already trained. No compatible pretrained weights were supplied.
3. Everything below was implemented from scratch.

A randomly initialized checkpoint is **not** an intelligent assistant. The UI states this until a trained checkpoint is verified and activated.

## What is training vs not training

| Action | Updates model weights? |
| --- | --- |
| Pretrain / continue pretrain / SFT | **Yes** — gradient descent on Karllo parameters |
| Upload a document to Knowledge | No — BM25 (or embeddings if installed) retrieval |
| Save a conversation or memory | No — stored as JSON, injected as context if enabled |
| Web research | No — and it is **not configured** unless `KARLLO_SEARCH_PROVIDER` is set |

## Architecture

Small GPT-style decoder:

- Token + positional embeddings
- Causal multi-head self-attention
- GELU feed-forward, residual connections, LayerNorm
- Next-token cross-entropy
- Byte-level BPE tokenizer with special chat tokens

Presets:

| Preset | Typical use | Rough params | Context |
| --- | --- | --- | --- |
| `tiny` | CPU tests / pilot | ~0.2M | 64 |
| `small` | Experimental CPU | ~1–2M | 256 |
| `medium` | Prefer GPU / Colab | ~10M+ | 512 |

These are experimental models. They will not match commercial assistants.

## Training modes (never mixed silently)

1. **pretrain** — random init, next-token prediction on raw text, tokenizer trained on the corpus.
2. **continue_pretrain** — load a compatible checkpoint, continue next-token prediction. Original files are not overwritten.
3. **sft** — load a checkpoint, train on `prompt`/`response`, chat `messages`, or `Question:` / `Answer:` books with **prompt loss masking**. Raw text only is rejected.

## Layout

```
backend/karllo/          Python package (model, training, API)
frontend/                 React + Vite UI
android/                  WebView client packaged as an APK
.github/workflows/        GitHub Actions APK build
data/seed/                Pilot corpus + SFT + QA book (original educational text)
data/eval/test.jsonl      Fixed evaluation set
checkpoints/              original / runs / best / latest
notebooks/Karllo_Training_Colab.ipynb
tests/                    Real training + checkpoint tests
```

## Run locally

Python 3.11+, Node 18+, ~4 GB RAM for tiny runs. **No GPU is assumed.**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt

export PYTHONPATH="$PWD/backend"
export KARLLO_ROOT="$PWD"
# optional: export KARLLO_OWNER_PASSWORD='your-password'
python3 -m uvicorn karllo.api.app:app --host 0.0.0.0 --port 8000

# another terminal
cd frontend && npm install && npm run dev
```

Open the Vite URL. Default owner password is `karllo-owner` (change before any real deployment).

Then: **Datasets → Load pilot datasets → Training → pretrain tiny ~30 steps → Checkpoints → Verify → Evaluate → Activate.**

## Android APK (GitHub Actions)

The APK is a **Karllo client**: it ships the web UI in a WebView. PyTorch training still runs on a Karllo server. The APK does not contain a GPU or the Python training worker.

1. Push this repository to GitHub (Actions is enabled on `main` and `arena/**`).
2. Open **Actions → Build Karllo APK**.
3. Download the `KarlloAI-debug-apk` artifact (`KarlloAI-debug.apk`).
4. Install on a device (debug build, not Play-signed).
5. On the login screen, set **Karllo server URL** to your API origin, for example `http://192.168.1.10:8000`, then sign in.

You can also run the workflow from **Actions → Run workflow**.

The APK talks to the server with a Bearer token. Keep the API on your LAN or HTTPS; do not embed secrets in the app.

## Tests

```bash
PYTHONPATH=backend python3 -m pytest tests
```

Tests run a **real** tiny pretrain and SFT loop on CPU and assert weights change, checkpoints restore, and SFT will not silently accept a raw corpus.

## Free compute (budget: zero)

- This host trains on **CPU** unless PyTorch reports CUDA/MPS.
- Google Colab: open `notebooks/Karllo_Training_Colab.ipynb`. Free GPUs are **quota-limited and often unavailable**. The notebook also runs on CPU for `tiny`.
- After a notebook run, copy `checkpoints/runs/<id>/latest` into this repo, Verify, Evaluate, Activate. Original init is never overwritten.
- Do a 20-step pilot and read measured loss / wall time before scaling. There is **no unlimited free GPU**.

Rough storage: a tiny fp32 checkpoint is a few megabytes.

## Optional web research

Unset by default. Karllo will **not** claim it browsed the web.

```bash
export KARLLO_SEARCH_PROVIDER=duckduckgo
# or another implemented provider + KARLLO_SEARCH_API_KEY (server-side only)
```

## Security

- Owner session cookie required for chat, data, and training.
- Upload type/size limits, path-traversal checks, delete confirmations.
- Training is a subprocess, not a long HTTP request.
- Conversations and memories stay on the owner machine and are not training data unless you export them.

## Honesty constraints

- Loss, token counts, and hardware come from the training worker — they are not fabricated.
- “Training complete” is set only when the worker finishes successfully.
- Retrieval hits are shown separately from model text.
- Hugging Face hub import is **not enabled** here (no hub access in this environment). Export JSONL and upload it. Honor licenses.
