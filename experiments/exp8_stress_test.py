"""
Experiment 8: Stress Test
Performance on adversarial queries
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
from src.evaluation import EvaluationMetrics


def load_config(config_path: str = "config/config.yaml"):
    """Load configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_stress_test(
    adversarial_queries: List[str],
    ground_truths: List[str],
    relevant_docs: List[List[int]],
    corpus: List[str],
    config: Dict[str, Any]
):
    """
    Run stress test on adversarial queries.
    
    Args:
        adversarial_queries: List of adversarial queries
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
        enable_revision=True,
        use_qlora=config["generation"]["qlora"]["training_enabled"]
    )
    
    # Initialize evaluator
    evaluator = EvaluationMetrics()
    
    # Run experiments
    results = []
    for query, gt, rel_docs in tqdm(zip(adversarial_queries, ground_truths, relevant_docs),
                                   total=len(adversarial_queries), desc="Stress test"):
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
            "min": np.min(scores),
            "max": np.max(scores),
            "scores": scores
        }
    
    # Error analysis
    error_analysis = {
        "high_hallucination": [],
        "low_verification": [],
        "poor_retrieval": []
    }
    
    for i, result in enumerate(results):
        if result["metrics"]["hallucination_rate"] > 0.15:
            error_analysis["high_hallucination"].append({
                "query": result["query"],
                "hallucination_rate": result["metrics"]["hallucination_rate"],
                "generated": result["generated_text"]
            })
        
        if result["metrics"]["verified_f1"] < 0.40:
            error_analysis["low_verification"].append({
                "query": result["query"],
                "verified_f1": result["metrics"]["verified_f1"],
                "generated": result["generated_text"]
            })
        
        if result["metrics"]["recall@20"] < 0.80:
            error_analysis["poor_retrieval"].append({
                "query": result["query"],
                "recall@20": result["metrics"]["recall@20"],
                "retrieved_docs": len(result["retrieved_docs"])
            })
    
    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/exp8_stress_test.json", "w") as f:
        json.dump({
            "aggregated_metrics": aggregated,
            "error_analysis": error_analysis,
            "individual_results": results
        }, f, indent=2)
    
    print("\n=== Experiment 8: Stress Test ===")
    print("Metrics on adversarial queries:")
    for metric_name, stats in aggregated.items():
        print(f"{metric_name}: {stats['mean']:.4f} ± {stats['std']:.4f} "
              f"(min={stats['min']:.4f}, max={stats['max']:.4f})")
    
    print(f"\nError analysis:")
    print(f"  High hallucination cases: {len(error_analysis['high_hallucination'])}")
    print(f"  Low verification cases: {len(error_analysis['low_verification'])}")
    print(f"  Poor retrieval cases: {len(error_analysis['poor_retrieval'])}")
    
    return aggregated, error_analysis


if __name__ == "__main__":
    config = load_config()
    
    # Load your adversarial dataset here
    # adversarial_queries, ground_truths, relevant_docs, corpus = load_adversarial_dataset(...)
    
    # Run experiment
    # aggregated, error_analysis = run_stress_test(
    #     adversarial_queries, ground_truths, relevant_docs, corpus, config
    # )

