"""Score agreement between verifier labels and a completed human review."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import get_commit_hash, get_timestamp, log_metrics, setup_wandb


LABELS = ("CONTRADICTED", "NO EVIDENCE", "SUPPORTED")
REQUIRED_COLUMNS = {"id", "auto_label", "human_label"}


def load_human_eval_samples(csv_path: str | Path) -> list[dict[str, str]]:
    with Path(csv_path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Human evaluation CSV has no header")
        missing = REQUIRED_COLUMNS - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Human evaluation CSV is missing columns: {', '.join(sorted(missing))}")
        return list(reader)


def compute_agreement_metrics(samples: list[dict[str, str]]) -> dict:
    if not samples:
        raise ValueError("Human evaluation CSV has no samples")

    auto_labels = []
    human_labels = []
    incomplete = []
    seen = set()
    for row_number, sample in enumerate(samples, start=2):
        if None in sample or any(sample.get(column) is None for column in REQUIRED_COLUMNS):
            raise ValueError(f"CSV row {row_number}: column count does not match header")
        sample_id = sample["id"].strip()
        if not sample_id:
            raise ValueError(f"CSV row {row_number}: id is required")
        if sample_id in seen:
            raise ValueError(f"CSV row {row_number}: duplicate id {sample_id!r}")
        seen.add(sample_id)

        auto_label = sample["auto_label"].strip().upper()
        human_label = sample["human_label"].strip().upper()
        if auto_label not in LABELS:
            raise ValueError(f"CSV row {row_number}: invalid auto_label {auto_label!r}")
        if human_label and human_label not in LABELS:
            raise ValueError(f"CSV row {row_number}: invalid human_label {human_label!r}")
        if not human_label:
            incomplete.append(row_number)
        auto_labels.append(auto_label)
        human_labels.append(human_label)

    if incomplete:
        raise ValueError(
            f"Review incomplete: {len(samples) - len(incomplete)}/{len(samples)} rows labeled. "
            f"Fill human_label in CSV rows {incomplete[:5]}"
            + (" (and others)." if len(incomplete) > 5 else ".")
        )

    count = len(samples)
    matches = sum(auto == human for auto, human in zip(auto_labels, human_labels))
    observed_agreement = matches / count
    auto_counts = Counter(auto_labels)
    human_counts = Counter(human_labels)
    expected_agreement = sum(
        auto_counts[label] * human_counts[label] / count**2 for label in LABELS
    )
    kappa = None if math.isclose(expected_agreement, 1.0) else (
        (observed_agreement - expected_agreement) / (1 - expected_agreement)
    )

    all_labels = sorted(set(auto_labels + human_labels))
    per_label_agreement = {}
    for label in all_labels:
        relevant = [
            (auto, human) for auto, human in zip(auto_labels, human_labels)
            if auto == label or human == label
        ]
        label_matches = sum(auto == human for auto, human in relevant)
        per_label_agreement[label] = {
            "matches": label_matches,
            "total": len(relevant),
            "agreement": label_matches / len(relevant) * 100,
        }

    confusion_matrix = {
        auto_label: {
            human_label: sum(
                auto == auto_label and human == human_label
                for auto, human in zip(auto_labels, human_labels)
            )
            for human_label in all_labels
        }
        for auto_label in all_labels
    }
    return {
        "total_samples": count,
        "percent_match": observed_agreement * 100,
        "cohens_kappa": kappa,
        "per_label_agreement": per_label_agreement,
        "confusion_matrix": confusion_matrix,
        "auto_label_distribution": dict(auto_counts),
        "human_label_distribution": dict(human_counts),
        "all_labels": all_labels,
    }


def save_agreement_metrics(metrics: dict, output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")


def print_agreement_summary(metrics: dict) -> None:
    print(f"Scored {metrics['total_samples']} reviewed samples")
    print(f"Verifier agreement: {metrics['percent_match']:.2f}%")
    kappa = metrics["cohens_kappa"]
    if kappa is None:
        print("Cohen's kappa: undefined (both label sets contain one class)")
    else:
        print(f"Cohen's kappa: {kappa:.4f}")
    for label, stats in metrics["per_label_agreement"].items():
        print(f"  {label}: {stats['matches']}/{stats['total']} matched")
    print(f"85% agreement target: {'met' if metrics['percent_match'] >= 85 else 'not met'}")
    kappa_target = kappa is not None and kappa >= 0.70
    print(f"0.70 kappa target: {'met' if kappa_target else 'not met or undefined'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("results/human_eval/human_eval_samples.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/metrics/human_eval_agreement.json"))
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    args = parser.parse_args()

    if args.output.exists():
        parser.error(f"report already exists: {args.output}")
    try:
        samples = load_human_eval_samples(args.csv)
        metrics = compute_agreement_metrics(samples)
        metrics["metadata"] = {
            "csv_path": str(args.csv),
            "csv_sha256": hashlib.sha256(args.csv.read_bytes()).hexdigest(),
            "total_samples": len(samples),
            "annotated_samples": len(samples),
            "commit_hash": get_commit_hash(),
            "timestamp": get_timestamp(),
        }
        save_agreement_metrics(metrics, args.output)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    print_agreement_summary(metrics)
    print(f"Saved agreement report to {args.output}")
    if not args.no_wandb:
        wandb_run = setup_wandb(
            project_name="SelfVerifyRAG",
            run_name="human_eval_agreement",
            config={"experiment": "human_eval_agreement", "csv_path": str(args.csv)},
            enabled=True,
        )
        if wandb_run:
            reported = {
                "percent_match": metrics["percent_match"],
                "total_samples": metrics["total_samples"],
            }
            if metrics["cohens_kappa"] is not None:
                reported["cohens_kappa"] = metrics["cohens_kappa"]
            log_metrics(reported, prefix="human_eval/")
            for label, stats in metrics["per_label_agreement"].items():
                log_metrics(
                    {f"agreement_{label.lower().replace(' ', '_')}": stats["agreement"]},
                    prefix="human_eval/",
                )
            try:
                import wandb

                wandb.finish()
            except Exception:
                pass


if __name__ == "__main__":
    main()
