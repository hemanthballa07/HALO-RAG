"""Validate and summarize human judgments of nonexact benchmark answers."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


REQUIRED_COLUMNS = {
    "seed", "example_id", "source_label", "abstained", "verified",
    "answers_question", "supported_by_evidence",
}
RELEVANCE_LABELS = {"YES", "NO", "UNCLEAR"}
SUPPORT_LABELS = {"SUPPORTED", "CONTRADICTED", "NO EVIDENCE", "UNCLEAR"}
SOURCE_LABELS = {"answerable", "unanswerable"}


class ReviewIncomplete(ValueError):
    def __init__(self, completed: int, total: int, rows: list[int]):
        self.completed = completed
        self.total = total
        self.rows = rows
        super().__init__(
            f"Review incomplete: {completed}/{total} rows labeled. "
            f"Fill both judgment columns in CSV rows {rows[:5]}"
            + (" (and others)." if len(rows) > 5 else ".")
        )


def parse_bool(value: str, row_number: int, column: str) -> bool:
    normalized = value.strip().lower()
    if normalized not in {"true", "false"}:
        raise ValueError(f"CSV row {row_number}: {column} must be True or False")
    return normalized == "true"


def summarize_answered(rows: list[dict]) -> dict:
    relevance = Counter(row["answers_question"] for row in rows)
    support = Counter(row["supported_by_evidence"] for row in rows)
    return {
        "count": len(rows),
        "answers_question": {label: relevance[label] for label in sorted(RELEVANCE_LABELS)},
        "supported_by_evidence": {label: support[label] for label in sorted(SUPPORT_LABELS)},
        "valid_supported_answers": sum(
            row["answers_question"] == "YES" and row["supported_by_evidence"] == "SUPPORTED"
            for row in rows
        ),
        "unsupported_answers": support["CONTRADICTED"] + support["NO EVIDENCE"],
        "answers_not_addressing_question": relevance["NO"],
        "unclear_on_either_axis": sum(
            row["answers_question"] == "UNCLEAR"
            or row["supported_by_evidence"] == "UNCLEAR"
            for row in rows
        ),
    }


def score_review(path: Path) -> dict:
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Review CSV has no header")
        missing_columns = REQUIRED_COLUMNS - set(reader.fieldnames)
        if missing_columns:
            raise ValueError(f"Review CSV is missing columns: {', '.join(sorted(missing_columns))}")
        rows = list(reader)

    if not rows:
        raise ValueError("Review CSV has no cases")

    seen = set()
    incomplete = []
    completed = 0
    normalized_rows = []
    for row_number, row in enumerate(rows, start=2):
        if None in row or any(row[column] is None for column in REQUIRED_COLUMNS):
            raise ValueError(f"CSV row {row_number}: column count does not match header")
        key = (row["seed"].strip(), row["example_id"].strip())
        if not all(key):
            raise ValueError(f"CSV row {row_number}: seed and example_id are required")
        if key in seen:
            raise ValueError(f"CSV row {row_number}: duplicate seed/example_id {key}")
        seen.add(key)

        source_label = row["source_label"].strip().lower()
        if source_label not in SOURCE_LABELS:
            raise ValueError(f"CSV row {row_number}: invalid source_label {source_label!r}")
        abstained = parse_bool(row["abstained"], row_number, "abstained")
        verified = parse_bool(row["verified"], row_number, "verified")
        relevance = row["answers_question"].strip().upper()
        support = row["supported_by_evidence"].strip().upper()
        allowed_relevance = {"NOT APPLICABLE"} if abstained else RELEVANCE_LABELS
        allowed_support = {"NOT APPLICABLE"} if abstained else SUPPORT_LABELS
        if relevance and relevance not in allowed_relevance:
            raise ValueError(f"CSV row {row_number}: invalid answers_question {relevance!r}")
        if support and support not in allowed_support:
            raise ValueError(f"CSV row {row_number}: invalid supported_by_evidence {support!r}")
        if not relevance or not support:
            incomplete.append(row_number)
        else:
            completed += 1
        normalized_rows.append({
            "source_label": source_label,
            "abstained": abstained,
            "verified": verified,
            "answers_question": relevance,
            "supported_by_evidence": support,
        })

    if incomplete:
        raise ReviewIncomplete(completed, len(rows), incomplete)

    answered = [row for row in normalized_rows if not row["abstained"]]
    abstained = [row for row in normalized_rows if row["abstained"]]
    return {
        "scope": "nonexact benchmark cases selected for human review",
        "interpretation": "Targeted subset only; not a factuality rate for the full benchmark.",
        "source_csv": str(path),
        "source_sha256": source_hash,
        "reviewed_cases": len(rows),
        "abstained": {
            "count": len(abstained),
            "by_source_label": {
                label: sum(row["source_label"] == label for row in abstained)
                for label in sorted(SOURCE_LABELS)
            },
        },
        "answered": summarize_answered(answered),
        "answered_by_source_label": {
            label: summarize_answered([row for row in answered if row["source_label"] == label])
            for label in sorted(SOURCE_LABELS)
        },
        "answered_by_verification": {
            label: summarize_answered([row for row in answered if row["verified"] == verified])
            for label, verified in (("verified", True), ("not_verified", False))
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Write a new JSON report instead of stdout")
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error(f"report already exists: {args.output}")
    try:
        result = score_review(args.csv)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    report = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(report)
        print(f"Saved review report to {args.output}")
    else:
        sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
