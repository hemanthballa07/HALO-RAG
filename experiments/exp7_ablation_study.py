"""
Experiment 7: Ablation Study
Component-wise contribution analysis
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

from src.pipeline import SelfVerificationRAGPipeline
from src.evaluation import EvaluationMetrics, StatisticalTester


def load_config(config_path: str = "config/config.yaml"):
    """Load configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_ablation_study(
    queries: List[str],
    ground_truths: List[str],
    relevant_docs: List[List[int]],
    corpus: List[str],
    config: Dict[str, Any]
):
    """
    Run ablation study on pipeline components.
    
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
    
    # Full pipeline (baseline)
    print("Running full pipeline...")
    pipeline_full = SelfVerificationRAGPipeline(
        corpus=corpus,
        device=device,
        enable_revision=True,
        use_qlora=config["generation"]["qlora"]["training_enabled"]
    )
    
    full_results = []
    for query, gt, rel_docs in tqdm(zip(queries, ground_truths, relevant_docs),
                                   total=len(queries), desc="Full pipeline"):
        result = pipeline_full.evaluate(query, gt, rel_docs)
        full_results.append(result)
    
    # Ablation variants
    variants = {
        "no_reranking": {"use_reranking": False},
        "no_verification": {"use_verification": False},
        "no_revision": {"enable_revision": False},
        "dense_only": {"dense_weight": 1.0, "sparse_weight": 0.0},
        "sparse_only": {"dense_weight": 0.0, "sparse_weight": 1.0}
    }
    
    all_results = {"full": full_results}
    
    for variant_name, variant_params in variants.items():
        print(f"Running {variant_name}...")
        
        # Create pipeline with variant
        variant_pipeline = SelfVerificationRAGPipeline(
            corpus=corpus,
            device=device,
            enable_revision=variant_params.get("enable_revision", True),
            use_qlora=config["generation"]["qlora"]["training_enabled"]
        )
        
        # Apply variant-specific modifications
        if "use_reranking" in variant_params and not variant_params["use_reranking"]:
            # Skip reranking (use retrieved docs directly)
            pass  # Would need to modify pipeline.generate() method
        
        if "use_verification" in variant_params and not variant_params["use_verification"]:
            # Skip verification (just generate)
            pass  # Would need to modify pipeline.generate() method
        
        if "dense_weight" in variant_params:
            variant_pipeline.retriever.dense_weight = variant_params["dense_weight"]
            variant_pipeline.retriever.sparse_weight = variant_params["sparse_weight"]
        
        results = []
        for query, gt, rel_docs in tqdm(zip(queries, ground_truths, relevant_docs),
                                       total=len(queries), desc=variant_name):
            result = variant_pipeline.evaluate(query, gt, rel_docs)
            results.append(result)
        
        all_results[variant_name] = results
    
    # Aggregate metrics
    metric_names = [
        "recall@20", "coverage", "factual_precision",
        "hallucination_rate", "verified_f1", "f1_score"
    ]
    
    aggregated = {}
    for variant_name, results in all_results.items():
        aggregated[variant_name] = {}
        for metric_name in metric_names:
            scores = [r["metrics"][metric_name] for r in results]
            aggregated[variant_name][metric_name] = {
                "mean": np.mean(scores),
                "std": np.std(scores),
                "scores": scores
            }
    
    # Statistical comparisons vs full pipeline
    comparisons = {}
    full_scores = {
        metric: aggregated["full"][metric]["scores"]
        for metric in metric_names
    }
    
    for variant_name in variants.keys():
        comparisons[variant_name] = {}
        for metric_name in metric_names:
            variant_scores = aggregated[variant_name][metric_name]["scores"]
            comparison = stats_tester.compare_metrics(
                full_scores[metric_name],
                variant_scores,
                f"{metric_name}_{variant_name}_vs_full"
            )
            comparisons[variant_name][metric_name] = comparison
    
    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/exp7_ablation_study.json", "w") as f:
        json.dump({
            "aggregated_metrics": aggregated,
            "statistical_comparisons": comparisons
        }, f, indent=2)
    
    print("\n=== Experiment 7: Ablation Study ===")
    for variant_name, metrics in aggregated.items():
        print(f"\n{variant_name.upper()}:")
        for metric_name, stats in metrics.items():
            print(f"  {metric_name}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    return aggregated, comparisons


if __name__ == "__main__":
    config = load_config()
    
    # Load your dataset here
    # queries, ground_truths, relevant_docs, corpus = load_dataset(...)
    
    # Run experiment
    # aggregated, comparisons = run_ablation_study(
    #     queries, ground_truths, relevant_docs, corpus, config
    # )

