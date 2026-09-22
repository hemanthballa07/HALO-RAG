"""
Experiment 4: Revision Strategies
Effectiveness of adaptive revision strategies
"""

import sys
from pathlib import Path
import os
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
import torch
import numpy as np
from typing import List, Dict, Any
from tqdm import tqdm
import json
import random

from src.data import load_dataset_from_config, prepare_for_experiments
from src.pipeline import SelfVerificationRAGPipeline
from src.evaluation import StatisticalTester
from src.utils.cli import parse_experiment_args
from src.utils.device import resolve_device


def load_config(config_path: str = "config/config.yaml"):
    """Load configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_revision_strategies_experiment(
    queries: List[str],
    ground_truths: List[str],
    relevant_docs: List[List[int]],
    corpus: List[str],
    config: Dict[str, Any]
):
    """
    Compare revision strategies.
    
    Args:
        queries: List of queries
        ground_truths: List of ground truth answers
        relevant_docs: List of relevant document IDs for each query
        corpus: List of documents
        config: Configuration dictionary
    """
    device = resolve_device(config.get("experiments", {}).get("device", "auto"))
    
    # Initialize evaluator
    stats_tester = StatisticalTester(alpha=0.05)
    
    # Baseline (no revision)
    print("Running baseline (no revision)...")
    pipeline_baseline = SelfVerificationRAGPipeline(
        corpus=corpus,
        device=device,
        enable_revision=False,
        use_qlora=config["generation"]["qlora"]["training_enabled"]
    )
    
    baseline_results = []
    for query, gt, rel_docs in tqdm(zip(queries, ground_truths, relevant_docs),
                                   total=len(queries), desc="Baseline"):
        result = pipeline_baseline.evaluate(query, gt, rel_docs)
        baseline_results.append(result)
    
    # With revision
    print("Running with adaptive revision...")
    pipeline_revision = SelfVerificationRAGPipeline(
        corpus=corpus,
        device=device,
        enable_revision=True,
        max_revision_iterations=config["revision"]["max_iterations"],
        use_qlora=config["generation"]["qlora"]["training_enabled"]
    )
    
    revision_results = []
    for query, gt, rel_docs in tqdm(zip(queries, ground_truths, relevant_docs),
                                   total=len(queries), desc="With revision"):
        result = pipeline_revision.evaluate(query, gt, rel_docs)
        revision_results.append(result)
    
    # Aggregate metrics
    metric_names = [
        "factual_precision", "hallucination_rate", 
        "verified_f1", "f1_score"
    ]
    
    baseline_aggregated = {}
    revision_aggregated = {}
    
    for metric_name in metric_names:
        baseline_scores = [r["metrics"][metric_name] for r in baseline_results]
        revision_scores = [r["metrics"][metric_name] for r in revision_results]
        
        baseline_aggregated[metric_name] = {
            "mean": np.mean(baseline_scores),
            "std": np.std(baseline_scores),
            "scores": baseline_scores
        }
        
        revision_aggregated[metric_name] = {
            "mean": np.mean(revision_scores),
            "std": np.std(revision_scores),
            "scores": revision_scores
        }
    
    # Statistical comparison
    comparisons = {}
    for metric_name in metric_names:
        baseline_scores = baseline_aggregated[metric_name]["scores"]
        revision_scores = revision_aggregated[metric_name]["scores"]
        
        comparison = stats_tester.compare_metrics(
            baseline_scores, revision_scores, metric_name
        )
        comparisons[metric_name] = comparison
    
    # Revision statistics
    revision_stats = {
        "num_revisions": [r["revision_iterations"] for r in revision_results],
        "avg_revision_iterations": np.mean([r["revision_iterations"] for r in revision_results]),
        "fraction_revised": np.mean([r["revision_iterations"] > 0 for r in revision_results])
    }
    
    # Save results
    os.makedirs("results/metrics", exist_ok=True)
    with open("results/metrics/exp4_revision_strategies.json", "w") as f:
        json.dump({
            "baseline_metrics": baseline_aggregated,
            "revision_metrics": revision_aggregated,
            "statistical_comparisons": comparisons,
            "revision_statistics": revision_stats
        }, f, indent=2, default=lambda value: value.item())
    
    print("\n=== Experiment 4: Revision Strategies ===")
    print("\nBaseline (no revision):")
    for metric_name, stats in baseline_aggregated.items():
        print(f"  {metric_name}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    print("\nWith revision:")
    for metric_name, stats in revision_aggregated.items():
        print(f"  {metric_name}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    print("\nStatistical comparisons:")
    for metric_name, comp in comparisons.items():
        print(f"  {metric_name}: improvement={comp['improvement']:.4f} "
              f"({comp['improvement_pct']:.2f}%), p={comp['p_value']:.4f}, "
              f"significant={comp['is_significant']}")
    
    print(f"\nRevision statistics:")
    print(f"  Average revision iterations: {revision_stats['avg_revision_iterations']:.2f}")
    print(f"  Fraction of queries revised: {revision_stats['fraction_revised']:.2f}")
    
    return baseline_aggregated, revision_aggregated, comparisons


def main():
    """Load the configured dataset and compare revision against the baseline."""
    args = parse_experiment_args("Experiment 4: revision strategies")
    config = load_config(args.config)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    sample_limit = 30 if args.dry_run else args.limit
    if sample_limit is None:
        sample_limit = config.get("datasets", {}).get("sample_limit")

    examples = load_dataset_from_config(config, split=args.split)
    if sample_limit:
        examples = examples[:sample_limit]
    if not examples:
        raise RuntimeError("The configured dataset returned no examples.")

    queries, ground_truths, relevant_docs, corpus = prepare_for_experiments(examples)
    return run_revision_strategies_experiment(
        queries=queries,
        ground_truths=ground_truths,
        relevant_docs=relevant_docs,
        corpus=corpus,
        config=config,
    )


if __name__ == "__main__":
    main()
