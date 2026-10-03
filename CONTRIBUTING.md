# Contributing

## Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -m spacy download en_core_web_sm
```

## Before opening a pull request

Run the standard verification gate:

```bash
make check
```

This is the lightweight CI gate. CI also runs `make full-check` on Python 3.12
with CPU-only PyTorch. On a machine with the full project dependencies and
spaCy English model installed, run it before a release or a model-backed change.
It checks the runtime dependencies and runs every test. The live service test
remains opt-in through `HALO_RAG_RUN_LIVE=1` and requires a local corpus.

Changes to experiment behavior should include a focused regression test and a note about
whether historical results need to be regenerated. Do not commit downloaded datasets,
model weights, secrets, local caches, or generated checkpoints.

Keep changes scoped, use type hints for new interfaces, and prefer structured logging in
long-running experiment code. Pull requests should explain the motivation, validation
performed, and any compute or data assumptions needed to reproduce the result.
