"""Run Exp1-8 for each seed and aggregate fresh, archived result artifacts."""

import sys
import os
import argparse
from pathlib import Path
import json
import csv
import subprocess
from datetime import datetime
import hashlib
import numpy as np
from typing import Dict, List, Any
import shutil
from uuid import uuid4

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
from experiments.final_result_reader import RESULT_FILES, load_metrics
from src.utils import get_timestamp
from src.utils.device import qlora_supported, resolve_device

PLOT_FILES = {
    "exp2_retrieval_comparison": ("exp2_retrieval_bars.png", "retrieval_bars.png"),
    "exp3_threshold_tuning": ("exp3_verified_f1_vs_tau.png", "tau_sweep.png"),
    "exp5_self_consistency": ("exp5_decoding_comparison.png", "decoding_comparison.png"),
    "exp6_iterative_training": ("exp6_iteration_curves.png", "iteration_curves.png"),
    "exp7_ablation_study": ("exp7_ablation_bars.png", "ablation_bars.png"),
    "exp8_stress_test": ("exp8_pareto_frontier.png", "pareto_frontier.png"),
}


def load_config(config_path: str = "config/config.yaml"):
    """Load configuration."""
    path = Path(config_path)
    if not path.is_absolute():
        path = project_root / path
    with path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def training_iterations(config: dict) -> int:
    """Read the configured number of Exp6 fine-tuning iterations."""
    iterations = config.get("experiments", {}).get("exp6", {}).get("iterations", 3)
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 0:
        raise ValueError("Experiment 6 iterations must be a nonnegative integer")
    return iterations


def check_training_readiness(config: dict, experiments: list[str],
                             split: str, diagnostic: bool) -> None:
    """Reject unsupported Exp6 runs before starting other experiments."""
    if "exp6_iterative_training" not in experiments:
        return
    if split != "validation":
        raise ValueError("Experiment 6 evaluates only the validation split")
    iterations = training_iterations(config)
    if iterations == 0:
        if not diagnostic:
            raise ValueError("A full run requires at least one Experiment 6 training iteration")
        return
    device = resolve_device(config.get("experiments", {}).get("device", "auto"))
    if not qlora_supported(device):
        raise RuntimeError(
            "Experiment 6 needs CUDA and bitsandbytes for QLoRA training "
            f"(resolved device: {device}). Use a supported CUDA host or omit "
            "Experiment 6 from diagnostic runs."
        )


def repository_is_clean(root: Path) -> bool:
    """Check tracked and untracked source files, excluding ignored run outputs."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=root, capture_output=True, text=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("cannot verify repository worktree state") from exc
    return not result.stdout.strip()


def repository_commit(root: Path) -> str:
    """Read the revision of the repository being evaluated."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root, capture_output=True, text=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()[:8] or "unknown"


def extract_numeric_metrics(payload: Dict[str, Any]) -> Dict[str, float]:
    """Extract scalar metric means from an experiment result document."""
    metric_payload = payload.get("aggregated_metrics", payload)
    from experiments.final_result_reader import extract_numeric_metrics as extract

    return extract(metric_payload)


def artifact_signature(path: Path) -> tuple[int, int, str] | None:
    """Detect whether a run wrote its expected result, including identical content rewrites."""
    if not path.is_file():
        return None
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size, hashlib.sha256(path.read_bytes()).hexdigest()


def failure_reason(stderr: str | None) -> str:
    """Keep the useful final error line without storing a full traceback in the manifest."""
    lines = [line.strip() for line in (stderr or "").splitlines() if line.strip()]
    return lines[-1][:300] if lines else "no stderr output"


def run_experiment(experiment_name: str, seed: int, config_path: str = "config/config.yaml", 
                   split: str = "validation", limit: int = None, dry_run: bool = False) -> Dict[str, Any]:
    """
    Run a single experiment with a given seed.
    
    Args:
        experiment_name: Name of experiment (e.g., "exp1_baseline")
        seed: Random seed
        config_path: Path to config file
        split: Dataset split
        limit: Limit number of examples
        dry_run: Dry run mode
    
    Returns:
        Dictionary with experiment results
    """
    print(f"\n{'='*60}")
    print(f"Running {experiment_name} with seed {seed}")
    print(f"{'='*60}")
    
    # Build command
    cmd = [sys.executable, f"experiments/{experiment_name}.py",
           "--config", config_path,
           "--split", split,
           "--seed", str(seed)]
    
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    if dry_run:
        cmd.append("--dry-run")
    cmd.append("--no-wandb")  # Disable W&B for final runs
    
    # Run experiment
    try:
        result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True, check=True)
        print(f"✓ {experiment_name} completed with seed {seed}")
        return {"status": "success", "output": result.stdout, "error": result.stderr}
    except subprocess.CalledProcessError as e:
        print(f"✗ {experiment_name} failed with seed {seed}: {failure_reason(e.stderr)}")
        return {"status": "error", "output": e.stdout, "error": e.stderr}


def aggregate_results_across_seeds(experiments: List[str], seeds: List[int],
                                   config_path: str = "config/config.yaml",
                                   split: str = "validation", limit: int = None,
                                   dry_run: bool = False, threshold: float = 0.75,
                                   archive_dir: Path | None = None,
                                   expected_iterations: int | None = None,
                                   expected_commit: str | None = None,
                                   require_plots: bool = False) -> tuple[dict, list[str], dict]:
    """
    Run experiments with multiple seeds and aggregate results.
    
    Args:
        experiments: List of experiment names
        seeds: List of random seeds
        config_path: Path to config file
        split: Dataset split
        limit: Limit number of examples
        dry_run: Dry run mode
    
    Returns:
        Aggregated results, failures, and paths to per-seed source artifacts
    """
    if archive_dir is None:
        run_id = datetime.now().strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8]
        archive_dir = project_root / "results/metrics/final_runs" / run_id
    archive_dir.mkdir(parents=True, exist_ok=False)
    all_results = {}
    failures = []
    artifacts = {}
    
    for exp_name in experiments:
        print(f"\n{'='*60}")
        print(f"Running {exp_name} with seeds {seeds}")
        print(f"{'='*60}")
        
        seed_results = []
        artifacts[exp_name] = {}
        
        for seed in seeds:
            artifact_path = project_root / "results/metrics" / RESULT_FILES[exp_name]
            before = artifact_signature(artifact_path)
            plot_name = PLOT_FILES[exp_name][0] if require_plots and exp_name in PLOT_FILES else None
            plot_path = project_root / "results/figures" / plot_name if plot_name else None
            plot_before = artifact_signature(plot_path) if plot_path else None
            run_result = run_experiment(exp_name, seed, config_path, split, limit, dry_run)
            if run_result["status"] != "success":
                failures.append(
                    f"{exp_name} (seed {seed}): experiment failed: "
                    f"{failure_reason(run_result.get('error'))}"
                )
                continue
            after = artifact_signature(artifact_path)
            if after is None or after == before:
                failures.append(f"{exp_name} (seed {seed}): no fresh metrics artifact")
                continue
            if plot_path and (artifact_signature(plot_path) in (None, plot_before)):
                failures.append(f"{exp_name} (seed {seed}): no fresh plot {plot_name}")
                continue
            try:
                metrics = load_metrics(
                    artifact_path, exp_name, threshold,
                    expected_seed=seed, expected_split=split,
                    expected_iterations=expected_iterations,
                    expected_commit=expected_commit,
                )
                if seed_results and metrics.keys() != seed_results[0].keys():
                    raise ValueError("metric names differ from earlier seeds")
                archived_path = archive_dir / f"{exp_name}_seed{seed}.json"
                with artifact_path.open("rb") as source, archived_path.open("xb") as target:
                    shutil.copyfileobj(source, target)
                archived_hash = hashlib.sha256(archived_path.read_bytes()).hexdigest()
                if archived_hash != after[2]:
                    raise ValueError("archived artifact differs from source")
                if archived_hash in (item["sha256"] for item in artifacts[exp_name].values()):
                    raise ValueError("identical artifact was produced for another seed")
            except (OSError, ValueError) as exc:
                failures.append(f"{exp_name} (seed {seed}): {exc}")
                continue
            seed_results.append(metrics)
            artifacts[exp_name][str(seed)] = {
                "path": str(archived_path.relative_to(project_root)),
                "sha256": archived_hash,
                "metrics": metrics,
            }
        
        # Aggregate across seeds
        if seed_results:
            # Get all metric names
            all_metrics = set()
            for result in seed_results:
                all_metrics.update(result.keys())
            
            aggregated = {}
            for metric in all_metrics:
                values = [r.get(metric, 0.0) for r in seed_results if metric in r]
                if values:
                    aggregated[metric] = {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                        "values": values,
                        "n": len(values)
                    }
            
            all_results[exp_name] = aggregated
            
            print(f"\n{exp_name} aggregated results:")
            for metric, stats in aggregated.items():
                print(f"  {metric}: {stats['mean']:.4f} ± {stats['std']:.4f} (n={stats['n']})")
    
    return all_results, failures, artifacts


def create_final_summary_csv(aggregated_results: Dict[str, Dict[str, Any]], 
                            output_path: str = "results/metrics/final_summary.csv"):
    """
    Create final summary CSV with mean ± sd for all experiments.
    
    Args:
        aggregated_results: Dictionary with aggregated results per experiment
        output_path: Output CSV path
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Define key metrics to include
    key_metrics = [
        "exact_match", "f1_score", "bleu4", "rouge_l",
        "factual_precision", "hallucination_rate", "verified_f1",
        "abstention_rate", "recall@20", "coverage"
    ]
    
    # Create CSV
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        header = ["experiment"] + key_metrics
        writer.writerow(header)
        
        # Data rows
        for exp_name, results in aggregated_results.items():
            row = [exp_name]
            for metric in key_metrics:
                if metric in results:
                    mean = results[metric]["mean"]
                    std = results[metric]["std"]
                    row.append(f"{mean:.4f} ± {std:.4f}")
                else:
                    row.append("")
            writer.writerow(row)
    
    print(f"\n✓ Created final summary CSV: {output_path}")


def copy_key_plots_to_final(output_dir: str = "results/figures/final") -> List[str]:
    """
    Copy 6 key plots to final directory.
    
    Args:
        output_dir: Output directory for final plots

    Returns:
        Source paths for plots that were not produced
    """
    # Define key plots to copy
    key_plots = list(PLOT_FILES.values())
    
    figures_dir = project_root / "results/figures"
    target_dir = Path(output_dir)
    if not target_dir.is_absolute():
        target_dir = project_root / target_dir
    missing = [str(figures_dir / source) for source, _ in key_plots
               if not (figures_dir / source).is_file()]
    if missing:
        for path in missing:
            print(f"✗ Plot not found: {path}")
        return missing
    target_dir.mkdir(parents=True, exist_ok=True)
    
    copied = []
    for src_name, dst_name in key_plots:
        shutil.copy2(figures_dir / src_name, target_dir / dst_name)
        copied.append(dst_name)
        print(f"✓ Copied {src_name} -> {dst_name}")
    
    print(f"\n✓ Copied {len(copied)}/{len(key_plots)} plots to {target_dir}")
    return []


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run final experiments with multiple seeds")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                       help="Path to config file")
    parser.add_argument("--split", type=str, default="validation",
                       help="Dataset split")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of examples")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456],
                       help="Random seeds to use")
    parser.add_argument("--dry-run", action="store_true",
                       help="Dry run mode")
    parser.add_argument("--experiments", type=str, nargs="+",
                       default=["exp1_baseline", "exp2_retrieval_comparison", "exp3_threshold_tuning",
                               "exp4_revision_strategies",
                               "exp5_self_consistency", "exp6_iterative_training", "exp7_ablation_study",
                               "exp8_stress_test"],
                       help="Experiments to run")
    parser.add_argument("--skip-runs", action="store_true",
                       help="Deprecated: existing artifacts cannot establish per-seed provenance")
    parser.add_argument("--copy-plots", action="store_true",
                       help="Copy key plots to final directory")
    
    args = parser.parse_args()
    if args.skip_runs:
        parser.error("--skip-runs cannot establish per-seed provenance; rerun the experiments")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must be unique")
    if len(set(args.experiments)) != len(args.experiments):
        parser.error("--experiments must be unique")
    unknown = set(args.experiments) - set(RESULT_FILES)
    if unknown:
        parser.error(f"unknown experiments: {', '.join(sorted(unknown))}")
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    config = load_config(args.config)
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    try:
        config_reference = str(config_path.relative_to(project_root))
    except ValueError:
        config_reference = str(config_path)
    threshold = float(config["verification"]["threshold"])
    if not 0 <= threshold <= 1:
        parser.error("configured verification threshold must be between 0 and 1")
    diagnostic = (
        args.dry_run or args.limit is not None or len(args.seeds) < 3
        or set(args.experiments) != set(RESULT_FILES)
    )
    try:
        check_training_readiness(config, args.experiments, args.split, diagnostic)
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    run_commit = repository_commit(project_root)
    if not diagnostic:
        if run_commit in {"", "unknown"}:
            parser.error("a full run requires a Git commit")
        try:
            if not repository_is_clean(project_root):
                parser.error("a full run requires a clean Git worktree")
        except RuntimeError as exc:
            parser.error(str(exc))
    run_id = datetime.now().strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8]
    archive_dir = project_root / "results/metrics/final_runs" / run_id
    
    print("="*60)
    print("FINAL EXPERIMENT RUNNER")
    print("="*60)
    print(f"Experiments: {args.experiments}")
    print(f"Seeds: {args.seeds}")
    print(f"Split: {args.split}")
    print(f"Limit: {args.limit}")
    print(f"Dry run: {args.dry_run}")
    print("="*60)
    
    aggregated_results, failures, artifacts = aggregate_results_across_seeds(
        experiments=args.experiments,
        seeds=args.seeds,
        config_path=args.config,
        split=args.split,
        limit=args.limit,
        dry_run=args.dry_run,
        threshold=threshold,
        archive_dir=archive_dir,
        expected_iterations=(
            training_iterations(config) if "exp6_iterative_training" in args.experiments else None
        ),
        expected_commit=run_commit if not diagnostic else None,
        require_plots=args.copy_plots and not diagnostic,
    )
    if hashlib.sha256(config_path.read_bytes()).hexdigest() != config_hash:
        failures.append("configuration changed during the run")
    if not diagnostic:
        if repository_commit(project_root) != run_commit:
            failures.append("repository commit changed during the run")
        try:
            if not repository_is_clean(project_root):
                failures.append("repository worktree changed during the run")
        except RuntimeError as exc:
            failures.append(str(exc))
    if args.copy_plots and not failures and not diagnostic:
        missing_plots = copy_key_plots_to_final()
        failures.extend(f"missing plot: {path}" for path in missing_plots)
    if args.copy_plots and diagnostic:
        print("Skipping final plot publication for a diagnostic run")

    manifest = {
        "status": "incomplete" if failures else "diagnostic" if diagnostic else "complete",
        "aggregated_results": aggregated_results,
        "artifacts": artifacts,
        "failures": failures,
        "seeds": args.seeds,
        "experiments": args.experiments,
        "split": args.split,
        "limit": args.limit,
        "dry_run": args.dry_run,
        "selected_threshold": threshold,
        "config_path": config_reference,
        "config_sha256": config_hash,
        "dataset": config["datasets"]["active"],
        "timestamp": get_timestamp(),
        "commit_hash": run_commit,
    }
    manifest_path = archive_dir / "manifest.json"
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print(f"Saved per-seed results and manifest to {archive_dir}")
    if failures:
        print("\nIncomplete runs:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    if diagnostic:
        print("Diagnostic run complete. Final summary files were not published.")
        return 0

    create_final_summary_csv(
        aggregated_results, output_path=str(project_root / "results/metrics/final_summary.csv")
    )
    results_json_path = project_root / "results/metrics/final_aggregated_results.json"
    with results_json_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print(f"Saved final aggregated results to {results_json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
