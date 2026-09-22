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

Changes to experiment behavior should include a focused regression test and a note about
whether historical results need to be regenerated. Do not commit downloaded datasets,
model weights, secrets, local caches, or generated checkpoints.

Keep changes scoped, use type hints for new interfaces, and prefer structured logging in
long-running experiment code. Pull requests should explain the motivation, validation
performed, and any compute or data assumptions needed to reproduce the result.
