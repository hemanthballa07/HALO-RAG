"""Checks for the 100-sample human evaluation scorer."""

import csv
import json
import subprocess
import sys

import pytest

from experiments.score_human_eval import compute_agreement_metrics


def sample(sample_id, auto_label, human_label):
    return {"id": sample_id, "auto_label": auto_label, "human_label": human_label}


def write_samples(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("id", "auto_label", "human_label"))
        writer.writeheader()
        writer.writerows(rows)


def test_agreement_uses_complete_normalized_labels():
    metrics = compute_agreement_metrics([
        sample("one", "supported", " SUPPORTED "),
        sample("two", "CONTRADICTED", "NO EVIDENCE"),
        sample("three", "NO EVIDENCE", "NO EVIDENCE"),
    ])

    assert metrics["total_samples"] == 3
    assert metrics["percent_match"] == pytest.approx(200 / 3)
    assert metrics["cohens_kappa"] == pytest.approx(0.5)
    assert metrics["confusion_matrix"]["CONTRADICTED"]["NO EVIDENCE"] == 1


def test_one_class_agreement_has_undefined_kappa():
    metrics = compute_agreement_metrics([sample("one", "SUPPORTED", "SUPPORTED")])

    assert metrics["percent_match"] == 100
    assert metrics["cohens_kappa"] is None


@pytest.mark.parametrize("rows,error", [
    ([sample("one", "SUPPORTED", "")], "Review incomplete: 0/1"),
    ([sample("one", "SUPPORTED", "YES")], "invalid human_label"),
    ([sample("one", "YES", "SUPPORTED")], "invalid auto_label"),
    ([sample("one", "SUPPORTED", "SUPPORTED"), sample("one", "SUPPORTED", "SUPPORTED")],
     "duplicate id"),
])
def test_malformed_or_partial_review_is_rejected(rows, error):
    with pytest.raises(ValueError, match=error):
        compute_agreement_metrics(rows)


def test_cli_does_not_save_partial_or_overwrite_existing_report(tmp_path):
    csv_path = tmp_path / "review.csv"
    output = tmp_path / "report.json"
    command = [
        sys.executable, "experiments/score_human_eval.py", "--csv", str(csv_path),
        "--output", str(output), "--no-wandb",
    ]
    write_samples(csv_path, [sample("one", "SUPPORTED", "")])
    incomplete = subprocess.run(command, capture_output=True, text=True, check=False)
    assert incomplete.returncode != 0
    assert not output.exists()

    write_samples(csv_path, [sample("one", "SUPPORTED", "SUPPORTED")])
    complete = subprocess.run(command, capture_output=True, text=True, check=False)
    assert complete.returncode == 0
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["total_samples"] == 1
    assert saved["cohens_kappa"] is None
    assert len(saved["metadata"]["csv_sha256"]) == 64

    repeated = subprocess.run(command, capture_output=True, text=True, check=False)
    assert repeated.returncode != 0
    assert json.loads(output.read_text(encoding="utf-8")) == saved
