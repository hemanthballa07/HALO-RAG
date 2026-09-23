# HALO-RAG

HALO-RAG is an experimental retrieval-augmented generation pipeline that verifies
generated claims against retrieved evidence and revises answers that are not well
supported. It combines hybrid retrieval, cross-encoder reranking, FLAN-T5 generation,
natural-language-inference verification, and adaptive revision strategies.

The project is research software. Verification can reduce unsupported claims, but it
does not guarantee that every answer is correct.

## Pipeline

1. Retrieve passages with MPNet embeddings (FAISS) and BM25.
2. Fuse dense and sparse scores, then rerank with an MS MARCO cross-encoder.
3. Generate an answer with FLAN-T5.
4. Extract claims and score their entailment against the retrieved evidence.
5. Re-retrieve, constrain generation, or revise claim-by-claim when support is weak.

The default model and experiment settings live in `config/config.yaml`. Runtime device
selection is automatic: CUDA is preferred, then Apple Silicon MPS, then CPU. QLoRA is
enabled only when CUDA and `bitsandbytes` are available.

## Setup

Python 3.10 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
python scripts/check_setup.py
```

For QLoRA training on a supported CUDA host, install the additional dependency:

```bash
python -m pip install -r requirements-gpu.txt
```

Model and dataset downloads can require several gigabytes. See [INSTALL.md](INSTALL.md)
for environment notes and troubleshooting.

## Run experiments

Run a small baseline first:

```bash
python experiments/exp1_baseline.py --limit 10 --no-wandb
```

For a paired baseline-versus-revision check with distractor passages and both
answerable and unanswerable questions, run:

```bash
python experiments/run_representative_benchmark.py \
  --questions 20 --corpus-size 500 --seed 42
```

This writes per-question answers and aggregate metrics to
`results/metrics/representative_benchmark.json`. The models must be downloaded on the
first run. The benchmark evaluates the same questions and corpus in both variants;
it does not train or tune the system.

To test the optional single-passage no-answer prompt against that saved sample:

```bash
python experiments/evaluate_focused_answers.py \
  --benchmark results/metrics/representative_benchmark.json
```

This is a generation-only comparison. It does not replace the full pipeline or
validate the verifier and revision path.
For an individual pipeline call, `generate(question, evidence_limit=1,
abstain_if_unanswered=True, do_sample=False)` enables the same experimental
prompt. The default behavior is unchanged.

Run the complete experiment sequence:

```bash
python experiments/run_final_experiments.py --dry-run
```

Individual experiments are documented in [experiments/README.md](experiments/README.md).
Outputs are written under `results/`; checkpoints and downloaded data are intentionally
excluded from version control.

## Development

```bash
python -m pip install -r requirements-dev.txt
make check
```

`make check` performs static checks, compiles every Python module, validates repository
structure, and runs the lightweight regression and benchmark tests. CI runs the
same checks on every push and pull request. With the full runtime installed, run
`python -m pytest -q` for the complete local test suite.

## Repository layout

```text
config/       experiment and model configuration
experiments/  reproducible experiment entry points
notebooks/    interactive pipeline walkthrough
scripts/      setup, validation, and results-lock utilities
src/          retrieval, generation, verification, revision, and evaluation code
tests/        regression and metric tests
results/      tracked result summaries and human-evaluation templates
```

## Evaluation

The suite reports retrieval metrics, answer overlap metrics, factual precision/recall,
coverage, hallucination rate, verified F1, FEVER-style scores, and abstention rate.
The paired benchmark additionally separates answerable and unanswerable exact match,
token F1, evidence hit rate, false acceptance, and abstention. Passage support alone
does not establish that an answer addresses the question.
Targets in `config/config.yaml` are evaluation goals, not guaranteed performance claims.
Historical outputs should be regenerated after changes to models, data processing, or
verification logic.

## License

See [LICENSE](LICENSE).
