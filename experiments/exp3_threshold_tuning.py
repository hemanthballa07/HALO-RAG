"""
Experiment 3: Threshold Tuning
Optimal τ (tau) for entailment verification
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
import matplotlib.pyplot as plt

from src.pipeline import SelfVerificationRAGPipeline
from src.evaluation import EvaluationMetrics


def load_config(config_path: str = "config/config.yaml"):
    """Load configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_threshold_tuning(
    queries: List[str],
    ground_truths: List[str],
    relevant_docs: List[List[int]],
    corpus: List[str],
    config: Dict[str, Any]
):
    """
    Find optimal entailment threshold τ.
    
    Args:
        queries: List of queries
        ground_truths: List of ground truth answers
        relevant_docs: List of relevant document IDs for each query
        corpus: List of documents
        config: Configuration dictionary
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Thresholds to test
    thresholds = config["verification"]["threshold_sweep"]
    # thresholds = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]
    
    # Initialize pipeline (will be updated with different thresholds)
    pipeline = SelfVerificationRAGPipeline(
        corpus=corpus,
        device=device,
        enable_revision=False,  # Disable revision for threshold tuning
        use_qlora=config["generation"]["qlora"]["training_enabled"]
    )
    
    # Initialize evaluator
    evaluator = EvaluationMetrics()
    
    # Results for each threshold
    threshold_results = {}
    
    for threshold in tqdm(thresholds, desc="Threshold tuning"):
        # Set threshold
        pipeline.set_entailment_threshold(threshold)
        
        # Run experiments
        results = []
        for query, gt, rel_docs in tqdm(zip(queries, ground_truths, relevant_docs),
                                       total=len(queries), desc=f"τ={threshold}"):
            result = pipeline.evaluate(query, gt, rel_docs)
            results.append(result)
        
        # Aggregate metrics
        metric_names = [
            "factual_precision", "hallucination_rate", 
            "verified_f1", "f1_score"
        ]
        
        aggregated = {}
        for metric_name in metric_names:
            scores = [r["metrics"][metric_name] for r in results]
            aggregated[metric_name] = {
                "mean": np.mean(scores),
                "std": np.std(scores),
                "scores": scores
            }
        
        threshold_results[threshold] = {
            "aggregated_metrics": aggregated,
            "individual_results": results
        }
    
    # Find optimal threshold (maximize Verified F1 while maintaining Factual Precision ≥ 0.90)
    optimal_threshold = None
    optimal_f1 = 0.0
    
    for threshold, results in threshold_results.items():
        factual_precision = results["aggregated_metrics"]["factual_precision"]["mean"]
        verified_f1 = results["aggregated_metrics"]["verified_f1"]["mean"]
        
        if factual_precision >= 0.90 and verified_f1 > optimal_f1:
            optimal_f1 = verified_f1
            optimal_threshold = threshold
    
    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/exp3_threshold_tuning.json", "w") as f:
        json.dump({
            "threshold_results": threshold_results,
            "optimal_threshold": optimal_threshold,
            "optimal_verified_f1": optimal_f1
        }, f, indent=2)
    
    # Plot threshold curves
    plot_threshold_curves(threshold_results)
    
    print("\n=== Experiment 3: Threshold Tuning ===")
    print(f"Optimal threshold (τ): {optimal_threshold}")
    print(f"Optimal Verified F1: {optimal_f1:.4f}")
    
    return threshold_results, optimal_threshold


def plot_threshold_curves(threshold_results: Dict[float, Dict]):
    """Plot threshold optimization curves."""
    thresholds = sorted(threshold_results.keys())
    
    factual_precision = [
        threshold_results[t]["aggregated_metrics"]["factual_precision"]["mean"]
        for t in thresholds
    ]
    hallucination_rate = [
        threshold_results[t]["aggregated_metrics"]["hallucination_rate"]["mean"]
        for t in thresholds
    ]
    verified_f1 = [
        threshold_results[t]["aggregated_metrics"]["verified_f1"]["mean"]
        for t in thresholds
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Factual Precision vs Threshold
    axes[0].plot(thresholds, factual_precision, marker='o', label='Factual Precision')
    axes[0].axhline(y=0.90, color='r', linestyle='--', label='Target (0.90)')
    axes[0].set_xlabel('Threshold (τ)')
    axes[0].set_ylabel('Factual Precision')
    axes[0].set_title('Factual Precision vs Threshold')
    axes[0].legend()
    axes[0].grid(True)
    
    # Hallucination Rate vs Threshold
    axes[1].plot(thresholds, hallucination_rate, marker='o', color='orange', label='Hallucination Rate')
    axes[1].axhline(y=0.10, color='r', linestyle='--', label='Target (≤0.10)')
    axes[1].set_xlabel('Threshold (τ)')
    axes[1].set_ylabel('Hallucination Rate')
    axes[1].set_title('Hallucination Rate vs Threshold')
    axes[1].legend()
    axes[1].grid(True)
    
    # Verified F1 vs Threshold
    axes[2].plot(thresholds, verified_f1, marker='o', color='green', label='Verified F1')
    axes[2].set_xlabel('Threshold (τ)')
    axes[2].set_ylabel('Verified F1')
    axes[2].set_title('Verified F1 vs Threshold')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    plt.savefig("results/figures/exp3_threshold_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Pareto frontier: Factual Precision vs Verified F1
    plt.figure(figsize=(8, 6))
    plt.scatter(factual_precision, verified_f1, s=100, alpha=0.6)
    for i, t in enumerate(thresholds):
        plt.annotate(f'τ={t}', (factual_precision[i], verified_f1[i]), 
                    fontsize=8, ha='right')
    plt.xlabel('Factual Precision')
    plt.ylabel('Verified F1')
    plt.title('Pareto Frontier: Factual Precision vs Verified F1')
    plt.grid(True)
    plt.axhline(y=0.52, color='r', linestyle='--', alpha=0.5, label='Target (0.52)')
    plt.axvline(x=0.90, color='r', linestyle='--', alpha=0.5, label='Target (0.90)')
    plt.legend()
    plt.savefig("results/figures/exp3_pareto_frontier.png", dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    config = load_config()
    
    # Load your dataset here
    # queries, ground_truths, relevant_docs, corpus = load_dataset(...)
    
    # Run experiment
    # threshold_results, optimal_threshold = run_threshold_tuning(
    #     queries, ground_truths, relevant_docs, corpus, config
    # )

