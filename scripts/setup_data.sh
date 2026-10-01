#!/usr/bin/env bash
set -euo pipefail

HALO_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HALO_PYTHON_BIN="${HALO_PYTHON_BIN:-python3}"

cd "$HALO_PROJECT_ROOT"
mkdir -p data results/metrics results/figures checkpoints logs

"$HALO_PYTHON_BIN" -m spacy download en_core_web_sm

echo "Local directories and the spaCy model are ready."
echo "Configured Hugging Face datasets will download on the first experiment run."
