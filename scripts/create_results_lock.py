"""Create a release record from a verified complete experiment run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.final_result_reader import RESULT_FILES, load_metrics


SUMMARY_METRICS = (
    "exact_match", "f1_score", "bleu4", "rouge_l", "factual_precision",
    "hallucination_rate", "verified_f1", "abstention_rate", "recall@20", "coverage",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repository_file(root: Path, reference: str) -> Path:
    relative = Path(reference)
    if relative.is_absolute():
        raise ValueError(f"path must be relative to the repository: {reference}")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root.resolve()) or not resolved.is_file():
        raise ValueError(f"missing or outside repository: {reference}")
    return resolved


def validate_summary(path: Path, aggregated: dict, experiments: set[str]) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["experiment", *SUMMARY_METRICS]:
            raise ValueError("final summary has an unexpected header")
        rows = list(reader)
    if len(rows) != len(experiments) or {row["experiment"] for row in rows} != experiments:
        raise ValueError("final summary does not contain each experiment exactly once")
    for row in rows:
        if None in row or any(value is None for value in row.values()):
            raise ValueError("final summary has malformed rows")
        metrics = aggregated[row["experiment"]]
        for name in SUMMARY_METRICS:
            stats = metrics.get(name)
            expected = f"{stats['mean']:.4f} ± {stats['std']:.4f}" if stats else ""
            if row[name] != expected:
                raise ValueError(f"final summary disagrees with {row['experiment']} {name}")


def validate_manifest(manifest_path: Path, root: Path) -> dict:
    root = root.resolve()
    manifest_path = manifest_path.resolve()
    if not manifest_path.is_relative_to(root) or not manifest_path.is_file():
        raise ValueError("manifest must be a file inside the repository")
    if manifest_path.parent.parent != root / "results/metrics/final_runs":
        raise ValueError("manifest must come from the final runner archive")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be a JSON object")

    try:
        if manifest["status"] != "complete" or manifest["failures"]:
            raise ValueError("manifest does not describe a complete run")
        if manifest["dry_run"] or manifest["limit"] is not None:
            raise ValueError("diagnostic or limited runs cannot be locked")
        experiments = manifest["experiments"]
        expected_experiments = set(RESULT_FILES)
        if len(experiments) != len(expected_experiments) or set(experiments) != expected_experiments:
            raise ValueError("manifest must include all eight experiments exactly once")
        seeds = manifest["seeds"]
        if (len(seeds) < 3 or len(set(seeds)) != len(seeds)
                or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)):
            raise ValueError("manifest must include at least three distinct integer seeds")

        config_path = repository_file(root, manifest["config_path"])
        if sha256(config_path) != manifest["config_sha256"]:
            raise ValueError("configuration changed after the run")
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if config["datasets"]["active"] != manifest["dataset"]:
            raise ValueError("manifest dataset disagrees with configuration")
        if not math.isclose(
            float(config["verification"]["threshold"]),
            float(manifest["selected_threshold"]), abs_tol=1e-12,
        ):
            raise ValueError("manifest threshold disagrees with configuration")
        expected_iterations = config.get("experiments", {}).get("exp6", {}).get("iterations", 3)
        if (isinstance(expected_iterations, bool)
                or not isinstance(expected_iterations, int)
                or expected_iterations <= 0):
            raise ValueError("release requires positive Experiment 6 training iterations")

        artifacts = manifest["artifacts"]
        aggregated = manifest["aggregated_results"]
        if set(artifacts) != expected_experiments or set(aggregated) != expected_experiments:
            raise ValueError("manifest artifact or aggregate set is incomplete")
        expected_seed_keys = {str(seed) for seed in seeds}
        for experiment in experiments:
            per_seed = artifacts[experiment]
            if set(per_seed) != expected_seed_keys:
                raise ValueError(f"{experiment} is missing a seed artifact")
            observed_metrics = []
            for seed in seeds:
                record = per_seed[str(seed)]
                artifact_path = repository_file(root, record["path"])
                if artifact_path.parent != manifest_path.parent:
                    raise ValueError(f"{experiment} seed {seed} is outside the run archive")
                if artifact_path.name != f"{experiment}_seed{seed}.json":
                    raise ValueError(f"{experiment} seed {seed} has an unexpected artifact name")
                if sha256(artifact_path) != record["sha256"]:
                    raise ValueError(f"{experiment} seed {seed} artifact changed")
                parsed_metrics = load_metrics(
                    artifact_path, experiment, float(manifest["selected_threshold"]),
                    expected_seed=seed, expected_split=manifest["split"],
                    expected_iterations=expected_iterations,
                    expected_commit=manifest["commit_hash"],
                    expected_dataset=manifest["dataset"],
                    expected_sample_limit=config["datasets"].get("sample_limit"),
                    check_sample_limit=True,
                )
                if parsed_metrics != record["metrics"]:
                    raise ValueError(f"{experiment} seed {seed} metrics disagree with archived JSON")
                observed_metrics.append(record["metrics"])
            metric_names = set(observed_metrics[0])
            if any(set(metrics) != metric_names for metrics in observed_metrics):
                raise ValueError(f"{experiment} metrics differ between seeds")
            if set(aggregated[experiment]) != metric_names:
                raise ValueError(f"{experiment} aggregate metrics are incomplete")
            for name in metric_names:
                values = [float(metrics[name]) for metrics in observed_metrics]
                stats = aggregated[experiment][name]
                if (not all(math.isfinite(value) for value in values)
                        or stats["n"] != len(seeds) or stats["values"] != values
                        or not math.isclose(stats["mean"], statistics.mean(values), abs_tol=1e-12)
                        or not math.isclose(stats["std"], statistics.pstdev(values), abs_tol=1e-12)):
                    raise ValueError(f"{experiment} {name} aggregate disagrees with seed artifacts")

        canonical = repository_file(root, "results/metrics/final_aggregated_results.json")
        if json.loads(canonical.read_text(encoding="utf-8")) != manifest:
            raise ValueError("published aggregate does not match the run manifest")
        summary = repository_file(root, "results/metrics/final_summary.csv")
        validate_summary(summary, aggregated, expected_experiments)
        if (not isinstance(manifest["commit_hash"], str)
                or manifest["commit_hash"] in {"", "unknown"}
                or not isinstance(manifest["split"], str) or not manifest["split"]):
            raise ValueError("manifest is missing the run commit or dataset split")
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError(f"malformed run manifest: {exc}") from exc

    return manifest


def create_results_lock(manifest_path: Path, output_path: Path,
                        root: Path = PROJECT_ROOT) -> Path:
    root = root.resolve()
    manifest_path = manifest_path if manifest_path.is_absolute() else root / manifest_path
    output_path = output_path if output_path.is_absolute() else root / output_path
    if output_path.exists():
        raise FileExistsError(f"results lock already exists: {output_path}")
    manifest = validate_manifest(manifest_path, root)
    summary = root / "results/metrics/final_summary.csv"
    config = yaml.safe_load(repository_file(root, manifest["config_path"]).read_text(encoding="utf-8"))
    manifest_reference = manifest_path.resolve().relative_to(root)
    artifact_rows = []
    for experiment in manifest["experiments"]:
        for seed in manifest["seeds"]:
            record = manifest["artifacts"][experiment][str(seed)]
            artifact_rows.append(
                f"| {experiment} | {seed} | `{record['path']}` | `{record['sha256']}` |"
            )
    content = "\n".join([
        "# HALO-RAG results lock",
        "",
        "Status: validated complete Exp1-8 run. Human evaluation is separate.",
        "",
        f"Generated (UTC): {datetime.now(timezone.utc).isoformat()}",
        f"Run commit: `{manifest['commit_hash']}`",
        f"Dataset and split: `{manifest['dataset']}` / `{manifest['split']}`",
        f"Seeds: {', '.join(str(seed) for seed in manifest['seeds'])}",
        f"Configured verification threshold: {manifest['selected_threshold']}",
        f"Configured dataset sample limit: {config['datasets'].get('sample_limit')}",
        "",
        "## Source files",
        "",
        f"- Manifest: `{manifest_reference}` (SHA-256 `{sha256(manifest_path)}`)",
        f"- Configuration: `{manifest['config_path']}` (SHA-256 `{manifest['config_sha256']}`)",
        f"- Final summary: `results/metrics/final_summary.csv` (SHA-256 `{sha256(summary)}`)",
        "- Published aggregate: `results/metrics/final_aggregated_results.json`",
        "",
        "## Per-seed artifacts",
        "",
        "| Experiment | Seed | Archived JSON | SHA-256 |",
        "| --- | ---: | --- | --- |",
        *artifact_rows,
        "",
        "This file records validated artifact consistency. It does not establish",
        "independent human agreement or guarantee factual correctness.",
        "",
    ])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        handle.write(content)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="Manifest from a completed final experiment run")
    parser.add_argument("--output", type=Path, default=Path("RESULTS_LOCK.md"))
    args = parser.parse_args()
    try:
        output = create_results_lock(args.manifest, args.output)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    print(f"Saved results lock to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
