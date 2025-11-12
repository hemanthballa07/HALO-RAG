"""
Final Experiment Runner
Re-runs Exp1-8 and Human Eval with seeds {42, 123, 456} and optimal τ from Exp8.
Aggregates results and creates final_summary.csv with mean ± sd.
"""

import sys
import os
import argparse
from pathlib import Path
import json
import csv
import subprocess
from datetime import datetime
import numpy as np
from typing import Dict, List, Any
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
from src.utils import get_commit_hash, get_timestamp


def load_config(config_path: str = "config/config.yaml"):
    """Load configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


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
    cmd = ["python", f"experiments/{experiment_name}.py", 
           "--config", config_path,
           "--split", split,
           "--seed", str(seed)]
    
    if limit:
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
        print(f"✗ {experiment_name} failed with seed {seed}: {e}")
        return {"status": "error", "output": e.stdout, "error": e.stderr}


def load_experiment_results(experiment_name: str, seed: int) -> Dict[str, Any]:
    """
    Load results from an experiment run.
    
    Args:
        experiment_name: Name of experiment
        seed: Random seed
    
    Returns:
        Dictionary with metrics
    """
    # Map experiment names to result files
    result_files = {
        "exp1_baseline": "results/metrics/exp1_baseline.json",
        "exp2_retrieval_comparison": "results/metrics/exp2_retrieval.csv",  # CSV format
        "exp3_threshold_tuning": "results/metrics/exp3_threshold_sweep.csv",  # CSV format
        "exp5_self_consistency": "results/metrics/exp5_self_consistency.json",
        "exp6_iterative_training": "results/metrics/exp6_iterative_training.csv",  # CSV format
        "exp7_ablation_study": "results/metrics/exp7_ablation.csv",  # CSV format
        "exp8_stress_test": "results/metrics/exp8_stress.json",
    }
    
    result_file = result_files.get(experiment_name)
    if not result_file or not os.path.exists(result_file):
        return {}
    
    try:
        if result_file.endswith(".json"):
            with open(result_file, 'r') as f:
                return json.load(f)
        elif result_file.endswith(".csv"):
            # Load CSV and convert to dict
            metrics = {}
            with open(result_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # For experiments with multiple rows, use the first/main row
                    if experiment_name == "exp2_retrieval_comparison":
                        # Use Hybrid+Rerank results
                        if row.get("config") == "hybrid_rerank" or "hybrid" in row.get("config", "").lower():
                            for key, value in row.items():
                                if key != "config" and value:
                                    try:
                                        metrics[key] = float(value)
                                    except ValueError:
                                        pass
                            break
                    elif experiment_name == "exp3_threshold_tuning":
                        # Use optimal threshold (0.75) results
                        if abs(float(row.get("threshold", 0)) - 0.75) < 0.01:
                            for key, value in row.items():
                                if key != "threshold" and value:
                                    try:
                                        metrics[key] = float(value)
                                    except ValueError:
                                        pass
                            break
                    elif experiment_name in ["exp6_iterative_training", "exp7_ablation_study"]:
                        # Use first row (full system) or specified variant
                        if experiment_name == "exp7_ablation_study" and row.get("variant") == "full":
                            for key, value in row.items():
                                if key != "variant" and value:
                                    try:
                                        metrics[key] = float(value)
                                    except ValueError:
                                        pass
                            break
                        elif experiment_name == "exp6_iterative_training":
                            # Use Iter3 results
                            if row.get("iteration") == "3" or "iter3" in row.get("iteration", "").lower():
                                for key, value in row.items():
                                    if key != "iteration" and value:
                                        try:
                                            metrics[key] = float(value)
                                        except ValueError:
                                            pass
                                break
            return metrics
    except Exception as e:
        print(f"Error loading results from {result_file}: {e}")
        return {}
    
    return {}


def aggregate_results_across_seeds(experiments: List[str], seeds: List[int], 
                                   config_path: str = "config/config.yaml",
                                   split: str = "validation", limit: int = None,
                                   dry_run: bool = False) -> Dict[str, Dict[str, Any]]:
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
        Dictionary with aggregated results per experiment
    """
    all_results = {}
    
    for exp_name in experiments:
        print(f"\n{'='*60}")
        print(f"Running {exp_name} with seeds {seeds}")
        print(f"{'='*60}")
        
        exp_results = {}
        seed_results = []
        
        for seed in seeds:
            # Run experiment
            run_result = run_experiment(exp_name, seed, config_path, split, limit, dry_run)
            
            if run_result["status"] == "success":
                # Load results
                metrics = load_experiment_results(exp_name, seed)
                if metrics:
                    seed_results.append(metrics)
        
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
            
            exp_results = aggregated
            all_results[exp_name] = exp_results
            
            print(f"\n{exp_name} aggregated results:")
            for metric, stats in aggregated.items():
                print(f"  {metric}: {stats['mean']:.4f} ± {stats['std']:.4f} (n={stats['n']})")
    
    return all_results


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


def copy_key_plots_to_final(output_dir: str = "results/figures/final"):
    """
    Copy 6 key plots to final directory.
    
    Args:
        output_dir: Output directory for final plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Define key plots to copy
    key_plots = [
        ("exp2_retrieval_bars.png", "retrieval_bars.png"),
        ("exp3_verified_f1_vs_tau.png", "tau_sweep.png"),
        ("exp5_decoding_comparison.png", "decoding_comparison.png"),
        ("exp6_iteration_curves.png", "iteration_curves.png"),
        ("exp8_pareto_frontier.png", "pareto_frontier.png"),
        ("exp7_ablation_bars.png", "ablation_bars.png"),
    ]
    
    figures_dir = "results/figures"
    
    copied = []
    for src_name, dst_name in key_plots:
        src_path = os.path.join(figures_dir, src_name)
        dst_path = os.path.join(output_dir, dst_name)
        
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            copied.append(dst_name)
            print(f"✓ Copied {src_name} -> {dst_name}")
        else:
            print(f"✗ Plot not found: {src_path}")
    
    print(f"\n✓ Copied {len(copied)}/{len(key_plots)} plots to {output_dir}")


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
                               "exp5_self_consistency", "exp6_iterative_training", "exp7_ablation_study",
                               "exp8_stress_test"],
                       help="Experiments to run")
    parser.add_argument("--skip-runs", action="store_true",
                       help="Skip running experiments, only aggregate existing results")
    parser.add_argument("--copy-plots", action="store_true",
                       help="Copy key plots to final directory")
    
    args = parser.parse_args()
    
    print("="*60)
    print("FINAL EXPERIMENT RUNNER")
    print("="*60)
    print(f"Experiments: {args.experiments}")
    print(f"Seeds: {args.seeds}")
    print(f"Split: {args.split}")
    print(f"Limit: {args.limit}")
    print(f"Dry run: {args.dry_run}")
    print("="*60)
    
    # Run experiments and aggregate results
    if not args.skip_runs:
        aggregated_results = aggregate_results_across_seeds(
            experiments=args.experiments,
            seeds=args.seeds,
            config_path=args.config,
            split=args.split,
            limit=args.limit,
            dry_run=args.dry_run
        )
    else:
        print("Skipping experiment runs, aggregating existing results...")
        aggregated_results = {}
        for exp_name in args.experiments:
            seed_results = []
            for seed in args.seeds:
                metrics = load_experiment_results(exp_name, seed)
                if metrics:
                    seed_results.append(metrics)
            
            if seed_results:
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
                aggregated_results[exp_name] = aggregated
    
    # Create final summary CSV
    if aggregated_results:
        create_final_summary_csv(aggregated_results)
    
    # Copy key plots
    if args.copy_plots:
        copy_key_plots_to_final()
    
    # Save aggregated results to JSON
    results_json_path = "results/metrics/final_aggregated_results.json"
    os.makedirs(os.path.dirname(results_json_path), exist_ok=True)
    with open(results_json_path, 'w') as f:
        json.dump({
            "aggregated_results": aggregated_results,
            "seeds": args.seeds,
            "experiments": args.experiments,
            "timestamp": get_timestamp(),
            "commit_hash": get_commit_hash()
        }, f, indent=2)
    
    print(f"\n✓ Saved aggregated results to {results_json_path}")
    print("\n" + "="*60)
    print("FINAL EXPERIMENT RUNNER COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()

