# Installation

## Requirements

- Python 3.10+
- Git
- Several gigabytes of free disk space for model and dataset caches
- A CUDA GPU for 4-bit QLoRA training; inference can use CUDA, Apple Silicon MPS, or CPU

## Standard environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
python scripts/check_setup.py
```

The Hugging Face datasets used by the experiments download on first use. To prefetch all
configured models, run `bash scripts/download_models.sh` after installing dependencies.

## CUDA and QLoRA

Install PyTorch using the command recommended for the host's CUDA version, then run:

```bash
python -m pip install -r requirements-gpu.txt
```

`bitsandbytes` is deliberately kept out of the base dependency set because it is not
portable to every platform. Experiment 6 exits with a clear error when its CUDA QLoRA
requirements are unavailable. Other experiments fall back to an unquantized generator.

## Development environment

```bash
python -m pip install -r requirements-dev.txt
make check
```

`make check` is the lightweight CI gate. Once the full project dependencies and
spaCy model are installed, use `make full-check` for the dependency check and all
tests. The optional live service test is skipped unless `HALO_RAG_RUN_LIVE=1` and
`HALO_RAG_CORPUS_PATH` are set.

Use `python scripts/check_setup.py --skip-dependencies` when validating only repository
structure and syntax, such as in a lightweight CI job.

## Common issues

- **Out of disk space:** remove unused Hugging Face cache entries or set
  `HF_HOME` to a volume with more capacity.
- **MPS operation unsupported:** set `experiments.device: "cpu"` in
  `config/config.yaml` for the affected run.
- **CUDA mismatch:** reinstall PyTorch for the installed CUDA runtime before installing
  `bitsandbytes`.
- **Missing spaCy model:** run `python -m spacy download en_core_web_sm` in the active
  virtual environment.
