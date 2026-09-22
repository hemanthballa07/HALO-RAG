#!/usr/bin/env bash
set -euo pipefail

HALO_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HALO_PYTHON_BIN="${HALO_PYTHON_BIN:-python3}"

cd "$HALO_PROJECT_ROOT"

"$HALO_PYTHON_BIN" - <<'PY'
from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

models = {
    "retriever": "sentence-transformers/all-mpnet-base-v2",
    "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "generator": "google/flan-t5-large",
    "verifier": "cross-encoder/nli-deberta-v3-base",
}

print(f"Downloading {models['retriever']}")
SentenceTransformer(models["retriever"])

print(f"Downloading {models['reranker']}")
CrossEncoder(models["reranker"])

print(f"Downloading {models['generator']}")
AutoTokenizer.from_pretrained(models["generator"])
AutoModelForSeq2SeqLM.from_pretrained(models["generator"])

print(f"Downloading {models['verifier']}")
AutoTokenizer.from_pretrained(models["verifier"])
AutoModelForSequenceClassification.from_pretrained(models["verifier"])

print("All configured models are cached.")
PY
