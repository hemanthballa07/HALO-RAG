"""Read the metric summary selected from each experiment's JSON artifact."""

from __future__ import annotations

import json
import math
from pathlib import Path


RESULT_FILES = {
    "exp1_baseline": "exp1_baseline.json",
    "exp2_retrieval_comparison": "exp2_retrieval_comparison.json",
    "exp3_threshold_tuning": "exp3_threshold_tuning.json",
    "exp4_revision_strategies": "exp4_revision_strategies.json",
    "exp5_self_consistency": "exp5_self_consistency.json",
    "exp6_iterative_training": "exp6_iterative_training.json",
    "exp7_ablation_study": "exp7_ablation.json",
    "exp8_stress_test": "exp8_stress.json",
}


def extract_numeric_metrics(payload: dict) -> dict[str, float]:
    """Extract finite scalar values or metric means, without filling missing fields."""
    if not isinstance(payload, dict):
        raise ValueError("metric section must be an object")
    metrics = {}
    for name, value in payload.items():
        if isinstance(value, dict):
            value = value.get("mean")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if not math.isfinite(value):
                raise ValueError(f"non-finite metric {name}")
            metrics[name] = float(value)
    if not metrics:
        raise ValueError("metric section contains no numeric metrics")
    return metrics


def select_metrics(experiment_name: str, payload: dict, threshold: float) -> dict[str, float]:
    """Select the documented variant or iteration from a saved experiment."""
    if "processed_queries" in payload and "total_queries" in payload:
        if payload["processed_queries"] != payload["total_queries"] or not payload["total_queries"]:
            raise ValueError("experiment did not process every query")
    if experiment_name == "exp1_baseline":
        section = payload["aggregated_metrics"]
    elif experiment_name == "exp2_retrieval_comparison":
        section = payload["aggregated_metrics"]["hybrid_rerank"]
    elif experiment_name == "exp3_threshold_tuning":
        matches = [
            result for value, result in payload["threshold_results"].items()
            if math.isclose(float(value), threshold, abs_tol=1e-9)
        ]
        if len(matches) != 1:
            raise ValueError(f"expected exactly one result for threshold {threshold}")
        section = matches[0]["aggregated_metrics"]
    elif experiment_name == "exp4_revision_strategies":
        section = payload["revision_metrics"]
    elif experiment_name == "exp5_self_consistency":
        section = payload["aggregated_metrics"]["self_consistency"]
    elif experiment_name == "exp6_iterative_training":
        iterations = payload["iteration_results"]
        final_iteration = max(int(value) for value in iterations)
        if final_iteration != payload["total_iterations"]:
            raise ValueError("iterative training did not reach its final iteration")
        section = iterations[str(final_iteration)]["metrics"]
    elif experiment_name == "exp7_ablation_study":
        section = payload["aggregated"]["full"]
    elif experiment_name == "exp8_stress_test":
        section = payload["baseline"]
    else:
        raise ValueError(f"unknown experiment {experiment_name!r}")
    return extract_numeric_metrics(section)


def load_metrics(path: Path, experiment_name: str, threshold: float,
                 expected_seed: int | None = None, expected_split: str | None = None) -> dict[str, float]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    try:
        metadata = payload.get("metadata", {})
        if expected_seed is not None and "seed" in metadata:
            if metadata["seed"] != expected_seed:
                raise ValueError(f"artifact seed {metadata['seed']} does not match {expected_seed}")
        if expected_split is not None and "split" in metadata:
            if metadata["split"] != expected_split:
                raise ValueError(f"artifact split {metadata['split']} does not match {expected_split}")
        return select_metrics(experiment_name, payload, threshold)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid {experiment_name} result in {path}: {exc}") from exc
