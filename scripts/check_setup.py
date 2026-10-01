#!/usr/bin/env python3
"""Validate repository structure, syntax, configuration, and dependencies."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_PATHS = (
    "README.md",
    "requirements.txt",
    "config/config.yaml",
    "src/data/loaders.py",
    "src/retrieval/hybrid_retrieval.py",
    "src/generator/flan_t5_generator.py",
    "src/verification/entailment_verifier.py",
    "src/pipeline/rag_pipeline.py",
    "experiments/run_final_experiments.py",
    "tests/test_regressions.py",
)

REQUIRED_MODULES = (
    "torch",
    "transformers",
    "sentence_transformers",
    "faiss",
    "rank_bm25",
    "datasets",
    "accelerate",
    "peft",
    "spacy",
    "sklearn",
    "numpy",
    "pandas",
    "matplotlib",
    "seaborn",
    "scipy",
    "tqdm",
    "wandb",
    "yaml",
    "rouge_score",
    "nltk",
    "statsmodels",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-dependencies",
        action="store_true",
        help="validate repository structure and syntax without checking the ML environment",
    )
    return parser.parse_args()


def check_structure(errors: list[str]) -> None:
    for relative_path in REQUIRED_PATHS:
        if not (PROJECT_ROOT / relative_path).exists():
            errors.append(f"missing required path: {relative_path}")


def check_syntax(errors: list[str]) -> None:
    source_roots = ("src", "experiments", "scripts", "tests")
    paths = [PROJECT_ROOT / "test_revision_strategies.py"]
    for source_root in source_roots:
        paths.extend((PROJECT_ROOT / source_root).rglob("*.py"))

    for path in sorted(set(paths)):
        try:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
        except (OSError, SyntaxError, UnicodeError) as exc:
            errors.append(f"invalid Python file {path.relative_to(PROJECT_ROOT)}: {exc}")


def check_dependencies(errors: list[str], warnings: list[str]) -> None:
    for module_name in REQUIRED_MODULES:
        if importlib.util.find_spec(module_name) is None:
            errors.append(f"missing Python dependency: {module_name}")

    if importlib.util.find_spec("bitsandbytes") is None:
        warnings.append("bitsandbytes is not installed; CUDA QLoRA training is unavailable")

    if importlib.util.find_spec("spacy") is not None:
        import spacy

        if not spacy.util.is_package("en_core_web_sm"):
            errors.append("missing spaCy model: en_core_web_sm")


def main() -> int:
    args = parse_args()
    errors: list[str] = []
    warnings: list[str] = []

    if sys.version_info < (3, 10):
        errors.append("Python 3.10 or newer is required")

    check_structure(errors)
    check_syntax(errors)
    if not args.skip_dependencies:
        check_dependencies(errors, warnings)

    for warning in warnings:
        print(f"warning: {warning}")
    for error in errors:
        print(f"error: {error}")

    if errors:
        print(f"setup check failed with {len(errors)} error(s)")
        return 1

    print("setup check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
