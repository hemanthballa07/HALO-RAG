"""Export nonexact benchmark answers with their evidence for manual review."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from src.data import load_dataset_from_config
from src.evaluation.benchmark import build_benchmark


COLUMNS = (
    "seed", "example_id", "source_label", "question", "reference_answers",
    "generated_answer", "abstained", "verified", "evidence_hit",
    "source_passage", "evidence_passage", "answers_question",
    "supported_by_evidence", "reviewer_notes",
)


def spreadsheet_safe(value):
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def review_rows(benchmark, source: dict, variant: str) -> list[dict]:
    metadata = source["metadata"]
    cases = source["cases"][variant]
    if [row["example_id"] for row in cases] != [case.example_id for case in benchmark.cases]:
        raise ValueError("saved case order does not match the dataset")

    selected = []
    for case, row in zip(benchmark.cases, cases):
        if row["exact_match"] == 1.0:
            continue
        evidence_ids = row["reranked_doc_ids"]
        if not evidence_ids or any(not 0 <= index < len(benchmark.corpus) for index in evidence_ids):
            raise ValueError(f"invalid evidence IDs for {case.example_id}")
        selected.append({
            "seed": metadata["seed"],
            "example_id": case.example_id,
            "source_label": "answerable" if case.answerable else "unanswerable",
            "question": case.question,
            "reference_answers": " | ".join(dict.fromkeys(case.references)),
            "generated_answer": row["generated"],
            "abstained": row["abstained"],
            "verified": row["verified"],
            "evidence_hit": row["final_evidence_hit"],
            "source_passage": case.context,
            "evidence_passage": "\n\n".join(benchmark.corpus[index] for index in evidence_ids),
            "answers_question": "",
            "supported_by_evidence": "",
            "reviewer_notes": "",
        })
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmarks", nargs="+", type=Path)
    parser.add_argument("--variant", default="focused")
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"review file already exists: {args.output}")

    config_bytes = args.config.read_bytes()
    config = yaml.safe_load(config_bytes)
    full_config = copy.deepcopy(config)
    full_config["datasets"]["sample_limit"] = None
    examples = load_dataset_from_config(full_config, split="validation")
    rows = []
    for path in args.benchmarks:
        source = json.loads(path.read_text(encoding="utf-8"))
        metadata = source["metadata"]
        if metadata["dataset"] != "squad_v2" or metadata["split"] != "validation":
            raise ValueError(f"unsupported dataset in {path}")
        if metadata["config_sha256"] != hashlib.sha256(config_bytes).hexdigest():
            raise ValueError(f"configuration does not match {path}")
        if args.variant not in source["cases"]:
            raise ValueError(f"variant {args.variant!r} is missing from {path}")
        benchmark = build_benchmark(
            examples, metadata["question_count"], metadata["corpus_size"], metadata["seed"]
        )
        if list(benchmark.document_hashes) != metadata["document_sha256"]:
            raise ValueError(f"passage hashes do not match {path}")
        if [case.example_id for case in benchmark.cases] != metadata["question_ids"]:
            raise ValueError(f"question IDs do not match {path}")
        rows.extend(review_rows(benchmark, source, args.variant))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(
            {column: spreadsheet_safe(row[column]) for column in COLUMNS}
            for row in rows
        )
    print(f"Saved {len(rows)} cases to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
