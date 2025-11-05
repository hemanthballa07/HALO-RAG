"""
Experiment 1: Baseline Comparison
Standard RAG vs Self-Verification RAG
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


def run_baseline_experiment(
    queries: List[str],
    ground_truths: List[str],
    relevant_docs: List[List[int]],
    corpus: List[str],
    config: Dict[str, Any]
):
    """
    Run baseline comparison experiment.
    
    Args:
        queries: List of queries
        ground_truths: List of ground truth answers
        relevant_docs: List of relevant document IDs for each query
        corpus: List of documents
        config: Configuration dictionary
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize pipeline
    pipeline = SelfVerificationRAGPipeline(
        corpus=corpus,
        device=device,
        enable_revision=False,  # Disable revision for baseline
        use_qlora=config["generation"]["qlora"]["training_enabled"]
    )
    
    # Initialize evaluator
    evaluator = EvaluationMetrics()
    stats_tester = StatisticalTester(alpha=0.05)
    
    # Run experiments
    results = []
    for query, gt, rel_docs in tqdm(zip(queries, ground_truths, relevant_docs), 
                                     total=len(queries), desc="Running baseline"):
        result = pipeline.evaluate(query, gt, rel_docs)
        results.append(result)
    
    # Aggregate metrics
    metric_names = [
        "recall@20", "coverage", "factual_precision", 
        "hallucination_rate", "verified_f1", "f1_score"
    ]
    
    aggregated = {}
    for metric_name in metric_names:
        scores = [r["metrics"][metric_name] for r in results]
        aggregated[metric_name] = {
            "mean": np.mean(scores),
            "std": np.std(scores),
            "scores": scores
        }
    
    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/exp1_baseline.json", "w") as f:
        json.dump({
            "aggregated_metrics": aggregated,
            "individual_results": results
        }, f, indent=2)
    
    print("\n=== Experiment 1: Baseline Results ===")
    for metric_name, stats in aggregated.items():
        print(f"{metric_name}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    return aggregated, results


if __name__ == "__main__":
    # Load config
    config = load_config()
    
    # TODO: Load dataset
    # For now, using placeholder data
    print("Experiment 1: Baseline Comparison")
    print("Note: Replace with actual dataset loading")
    
    # Placeholder data structure
    queries = []
    ground_truths = []
    relevant_docs = []
    corpus = []
    
    # Run experiment
    # aggregated, results = run_baseline_experiment(
    #     queries, ground_truths, relevant_docs, corpus, config
    # )

