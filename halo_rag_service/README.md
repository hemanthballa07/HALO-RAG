# HALO-RAG service

This optional API exposes the existing inference pipeline. It does not change the
experiment code or turn diagnostic results into release results. The service loads
one pipeline at startup and uses it for both endpoints. It does not load QLoRA
adapters or claim to serve a fine-tuned checkpoint.

The `fix/main-exp` branch contains a larger service, but its response mapping does
not match the current pipeline and its model loader duplicates model construction.
This smaller service keeps the health check and answer endpoint while using the
pipeline directly.

## Setup

Install the normal project dependencies and the optional service dependencies:

```bash
python -m pip install -r requirements-service.txt
python -m spacy download en_core_web_sm
```

Provide a UTF-8 text file with one evidence passage per line. The service will not
start without it, and it does not silently fall back to a demo corpus.
`halo_rag_service/example_corpus.txt` is a small local smoke-test corpus.

```bash
export HALO_RAG_CORPUS_PATH=/absolute/path/to/corpus.txt
python -m uvicorn halo_rag_service.app:app --host 127.0.0.1 --port 8000
```

`HALO_RAG_CONFIG_PATH` can override `config/config.yaml`. Model downloads and
startup may take several minutes. Set `HF_HOME` before starting if model caches
live on an external drive.

Check readiness with `GET /health`. Send a JSON request to `POST /generate`:

```json
{"query": "Where is Paris?", "top_k_retrieve": 20, "top_k_rerank": 5}
```

The response includes the answer, evidence passages, a verification summary, and
one of `verified`, `unverified`, or `abstained`. No public authentication, rate
limiting, or multi-user isolation is included, so bind to localhost only.

For single-passage questions, an optional no-answer mode limits generation to
the top passage and asks the model to abstain when that passage does not answer:

```json
{"query": "Where is Paris?", "evidence_limit": 1, "abstain_if_unanswered": true}
```

This can reduce unsupported answers, but it is not a guarantee. A `verified`
response means the current claim checker accepted the answer, not that the
question was answerable from the evidence. The checker can mistake nearby words
for support when a passage describes a different subject or action. Inspect the
returned source text before relying on an answer. Single-passage mode may also
omit evidence needed for multi-hop questions, so it is opt-in.

Install `requirements-dev.txt` to run the HTTP tests with
`python -m pytest -q tests/test_service.py`. They use a
fake pipeline and do not download models. This API adapter does not affect the
historical experiment outputs, so those results do not need regeneration.

For a model-backed smoke test, point `HALO_RAG_CORPUS_PATH` at the example corpus,
set `HALO_RAG_RUN_LIVE=1`, and run
`python -m pytest -q tests/test_service_live.py`. The live test uses CPU when no
supported accelerator is available and may take a few minutes.
