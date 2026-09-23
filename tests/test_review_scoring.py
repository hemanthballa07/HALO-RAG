"""Checks for scoring the targeted human benchmark review."""

import csv
import subprocess
import sys

import pytest

from experiments.score_benchmark_review import ReviewIncomplete, score_review


COLUMNS = (
    "seed", "example_id", "source_label", "abstained", "verified",
    "answers_question", "supported_by_evidence",
)


def review_row(**changes):
    row = {
        "seed": "2026",
        "example_id": "example-1",
        "source_label": "answerable",
        "abstained": "False",
        "verified": "True",
        "answers_question": "YES",
        "supported_by_evidence": "SUPPORTED",
    }
    row.update(changes)
    return row


def write_review(path, rows, columns=COLUMNS):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def test_score_review_keeps_source_label_and_human_judgment_separate(tmp_path):
    path = tmp_path / "review.csv"
    write_review(path, [
        review_row(),
        review_row(example_id="example-2", source_label="unanswerable",
                   supported_by_evidence="NO EVIDENCE"),
        review_row(example_id="example-3", verified="False", answers_question="NO",
                   supported_by_evidence="UNCLEAR"),
        review_row(example_id="example-4", abstained="True", verified="False",
                   answers_question="NOT APPLICABLE", supported_by_evidence="NOT APPLICABLE"),
    ])

    result = score_review(path)

    assert result["reviewed_cases"] == 4
    assert result["abstained"]["count"] == 1
    assert result["answered"]["count"] == 3
    assert result["answered"]["valid_supported_answers"] == 1
    assert result["answered"]["unsupported_answers"] == 1
    assert result["answered"]["unclear_on_either_axis"] == 1
    assert result["answered_by_source_label"]["unanswerable"]["unsupported_answers"] == 1
    assert result["answered_by_verification"]["verified"]["unsupported_answers"] == 1
    assert result["answered_by_verification"]["not_verified"]["unsupported_answers"] == 0
    assert len(result["source_sha256"]) == 64


def test_incomplete_review_does_not_write_report(tmp_path):
    path = tmp_path / "review.csv"
    output = tmp_path / "report.json"
    write_review(path, [review_row(answers_question="", supported_by_evidence="")])

    with pytest.raises(ReviewIncomplete, match="0/1 rows labeled"):
        score_review(path)
    result = subprocess.run(
        [sys.executable, "experiments/score_benchmark_review.py", "--csv", str(path),
         "--output", str(output)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode != 0
    assert "Review incomplete" in result.stderr
    assert not output.exists()


@pytest.mark.parametrize("changes,error", [
    ({"abstained": "True", "answers_question": "YES"}, "invalid answers_question"),
    ({"abstained": "False", "supported_by_evidence": "NOT APPLICABLE"},
     "invalid supported_by_evidence"),
    ({"verified": "maybe"}, "verified must be True or False"),
    ({"source_label": "unknown"}, "invalid source_label"),
])
def test_invalid_review_values_are_rejected(tmp_path, changes, error):
    path = tmp_path / "review.csv"
    write_review(path, [review_row(**changes)])
    with pytest.raises(ValueError, match=error):
        score_review(path)


def test_duplicate_cases_and_missing_columns_are_rejected(tmp_path):
    path = tmp_path / "review.csv"
    write_review(path, [review_row(), review_row()])
    with pytest.raises(ValueError, match="duplicate seed/example_id"):
        score_review(path)

    write_review(path, [review_row()], columns=COLUMNS[:-1])
    with pytest.raises(ValueError, match="missing columns: supported_by_evidence"):
        score_review(path)
