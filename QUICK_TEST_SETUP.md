# Quick validation

Install the lightweight development dependencies and run the repository checks without
downloading any ML models:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
make check
```

For an end-to-end smoke run, install the runtime dependencies and spaCy model first, then
use a small sample:

```bash
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
python experiments/exp1_baseline.py --limit 10 --no-wandb
```

The first end-to-end run downloads model and dataset assets and therefore needs network
access and several gigabytes of free disk space.
