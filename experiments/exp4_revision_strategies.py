"""
Experiment 4: Revision Strategies
Effectiveness of adaptive revision strategies
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch
import numpy as np
from typing import List, Dict, Any
from tqdm import tqdm
import json

from src.pipeline import SelfVerificationRAGPipeline
from src.evaluation import EvaluationMetrics, StatisticalTester


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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize evaluator
    evaluator = EvaluationMetrics()
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
    os.makedirs("results", exist_ok=True)
    with open("results/exp4_revision_strategies.json", "w") as f:
        json.dump({
            "baseline_metrics": baseline_aggregated,
            "revision_metrics": revision_aggregated,
            "statistical_comparisons": comparisons,
            "revision_statistics": revision_stats
        }, f, indent=2)
    
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


if __name__ == "__main__":
    config = load_config()
    print("Experiment 4: Revision Strategies")
    print("Note: Replace with actual dataset loading")

