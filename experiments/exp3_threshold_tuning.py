"""
Experiment 3: Threshold Tuning
Optimal τ (tau) for entailment verification
"""

import sys
import os
import argparse
from pathlib import Path
import json
import csv
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
import torch
import numpy as np
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import seaborn as sns

from src.data import load_dataset_from_config, prepare_for_experiments
from src.pipeline import SelfVerificationRAGPipeline
from src.evaluation import EvaluationMetrics
from src.utils import setup_wandb, log_metrics, log_metadata, get_commit_hash, get_timestamp
from src.utils.cli import parse_experiment_args


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
    config: Dict[str, Any],
    thresholds: List[float],
    seed: int = 42,
    wandb_run: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Find optimal entailment threshold τ.
    
    Args:
        queries: List of queries
        ground_truths: List of ground truth answers
        relevant_docs: List of relevant document IDs for each query
        corpus: List of documents
        config: Configuration dictionary
        thresholds: List of thresholds to test
        seed: Random seed
        wandb_run: W&B run object (optional)
    
    Returns:
        Dictionary with results and metrics
    """
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    device = "cuda"
    print(f"Using device: {device}")
    
    # Initialize pipeline (will be updated with different thresholds)
    print("Initializing pipeline...")
    pipeline = SelfVerificationRAGPipeline(
        corpus=corpus,
        device=device,
        enable_revision=False,  # Disable revision for threshold tuning
        use_qlora=config.get("generation", {}).get("qlora", {}).get("training_enabled", False)
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
        all_metrics = []
        
        for query, gt, rel_docs in tqdm(
            zip(queries, ground_truths, relevant_docs),
            total=len(queries),
            desc=f"τ={threshold}",
            leave=False
        ):
            try:
                # Generate answer with verification
                result = pipeline.generate(query, top_k_retrieve=20, top_k_rerank=5)
                
                # Get retrieved texts for coverage calculation
                retrieved_texts = result.get("reranked_texts", result.get("retrieved_texts", []))
                
                # Compute metrics
                metrics = evaluator.compute_all_metrics(
                    retrieved_docs=result["retrieved_docs"],
                    relevant_docs=rel_docs,
                    verification_results=result["verification_results"]["verification_results"],
                    generated=result["generated_text"],
                    ground_truth=gt,
                    retrieved_texts=retrieved_texts
                )
                
                all_metrics.append(metrics)
                results.append({
                    "query": query,
                    "ground_truth": gt,
                    "generated": result["generated_text"],
                    "metrics": metrics
                })
            except Exception as e:
                print(f"Error processing query: {e}")
                continue
        
        # Aggregate metrics
        metric_names = [
            "factual_precision", "factual_recall", "hallucination_rate",
            "verified_f1", "f1_score", "exact_match", "abstention_rate"
        ]
        
        aggregated = {}
        for metric_name in metric_names:
            if all_metrics and metric_name in all_metrics[0]:
                scores = [m[metric_name] for m in all_metrics if metric_name in m]
                if scores:
            aggregated[metric_name] = {
                        "mean": float(np.mean(scores)),
                        "std": float(np.std(scores)),
                        "min": float(np.min(scores)),
                        "max": float(np.max(scores)),
                "scores": scores
            }
        
        threshold_results[threshold] = {
            "aggregated_metrics": aggregated,
            "individual_results": results
        }
        
        # Log to W&B
        if wandb_run:
            final_metrics = {k: v["mean"] for k, v in aggregated.items()}
            final_metrics["threshold"] = threshold
            log_metrics(final_metrics, prefix="threshold_tuning/")
    
    # Find optimal threshold (maximize Verified F1 while maintaining Factual Precision ≥ 0.90)
    optimal_threshold = None
    optimal_f1 = 0.0
    
    for threshold, results in threshold_results.items():
        aggregated = results["aggregated_metrics"]
        factual_precision = aggregated.get("factual_precision", {}).get("mean", 0)
        verified_f1 = aggregated.get("verified_f1", {}).get("mean", 0)
        
        if factual_precision >= 0.90 and verified_f1 > optimal_f1:
            optimal_f1 = verified_f1
            optimal_threshold = threshold
    
    # If no threshold meets Factual Precision ≥ 0.90, select threshold with highest Verified F1
    if optimal_threshold is None:
        for threshold, results in threshold_results.items():
            aggregated = results["aggregated_metrics"]
            verified_f1 = aggregated.get("verified_f1", {}).get("mean", 0)
            if verified_f1 > optimal_f1:
                optimal_f1 = verified_f1
                optimal_threshold = threshold
    
    return {
            "threshold_results": threshold_results,
            "optimal_threshold": optimal_threshold,
            "optimal_verified_f1": optimal_f1
    }


def plot_threshold_curves(results: Dict[str, Any], output_dir: str = "results/figures"):
    """Plot threshold optimization curves."""
    os.makedirs(output_dir, exist_ok=True)
    
    threshold_results = results["threshold_results"]
    thresholds = sorted(threshold_results.keys())
    
    # Extract metrics
    factual_precision = [
        threshold_results[t]["aggregated_metrics"].get("factual_precision", {}).get("mean", 0)
        for t in thresholds
    ]
    factual_recall = [
        threshold_results[t]["aggregated_metrics"].get("factual_recall", {}).get("mean", 0)
        for t in thresholds
    ]
    hallucination_rate = [
        threshold_results[t]["aggregated_metrics"].get("hallucination_rate", {}).get("mean", 0)
        for t in thresholds
    ]
    verified_f1 = [
        threshold_results[t]["aggregated_metrics"].get("verified_f1", {}).get("mean", 0)
        for t in thresholds
    ]
    f1_score = [
        threshold_results[t]["aggregated_metrics"].get("f1_score", {}).get("mean", 0)
        for t in thresholds
    ]
    abstention_rate = [
        threshold_results[t]["aggregated_metrics"].get("abstention_rate", {}).get("mean", 0)
        for t in thresholds
    ]
    
    # Plot 1: Verified F1 vs Threshold
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, verified_f1, marker='o', linewidth=2, markersize=8, label='Verified F1', color='green')
    plt.axhline(y=0.52, color='r', linestyle='--', alpha=0.5, label='Target (0.52)')
    plt.xlabel('Threshold (τ)', fontsize=12)
    plt.ylabel('Verified F1', fontsize=12)
    plt.title('Verified F1 vs Threshold', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    # Mark optimal threshold
    optimal_threshold = results.get("optimal_threshold")
    if optimal_threshold:
        optimal_idx = thresholds.index(optimal_threshold)
        plt.plot(optimal_threshold, verified_f1[optimal_idx], 'ro', markersize=12, label=f'Optimal (τ={optimal_threshold})')
        plt.legend(fontsize=11)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "exp3_verified_f1_vs_tau.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved plot to {output_path}")
    
    # Plot 2: Precision vs Recall (Answer Recall)
    plt.figure(figsize=(10, 6))
    plt.plot(factual_recall, factual_precision, marker='o', linewidth=2, markersize=8, label='Precision-Recall Curve', color='blue')
    plt.xlabel('Factual Recall (Answer Recall)', fontsize=12)
    plt.ylabel('Factual Precision', fontsize=12)
    plt.title('Precision vs Recall (Threshold Sweep)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0.90, color='r', linestyle='--', alpha=0.5, label='Target Precision (0.90)')
    plt.axvline(x=0.85, color='r', linestyle='--', alpha=0.5, label='Target Recall (0.85)')
    
    # Annotate thresholds
    for i, t in enumerate(thresholds):
        plt.annotate(f'τ={t}', (factual_recall[i], factual_precision[i]), 
                    fontsize=8, ha='left', alpha=0.7)
    
    plt.legend(fontsize=11)
    plt.tight_layout()
    output_path = os.path.join(output_dir, "exp3_precision_vs_recall.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved plot to {output_path}")
    
    # Plot 3: Additional comprehensive plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Factual Precision vs Threshold
    axes[0, 0].plot(thresholds, factual_precision, marker='o', label='Factual Precision', color='blue')
    axes[0, 0].axhline(y=0.90, color='r', linestyle='--', label='Target (0.90)')
    axes[0, 0].set_xlabel('Threshold (τ)')
    axes[0, 0].set_ylabel('Factual Precision')
    axes[0, 0].set_title('Factual Precision vs Threshold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Hallucination Rate vs Threshold
    axes[0, 1].plot(thresholds, hallucination_rate, marker='o', color='orange', label='Hallucination Rate')
    axes[0, 1].axhline(y=0.10, color='r', linestyle='--', label='Target (≤0.10)')
    axes[0, 1].set_xlabel('Threshold (τ)')
    axes[0, 1].set_ylabel('Hallucination Rate')
    axes[0, 1].set_title('Hallucination Rate vs Threshold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Verified F1 vs Threshold
    axes[1, 0].plot(thresholds, verified_f1, marker='o', color='green', label='Verified F1')
    axes[1, 0].axhline(y=0.52, color='r', linestyle='--', alpha=0.5, label='Target (0.52)')
    axes[1, 0].set_xlabel('Threshold (τ)')
    axes[1, 0].set_ylabel('Verified F1')
    axes[1, 0].set_title('Verified F1 vs Threshold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Abstention Rate vs Threshold
    axes[1, 1].plot(thresholds, abstention_rate, marker='o', color='purple', label='Abstention Rate')
    axes[1, 1].set_xlabel('Threshold (τ)')
    axes[1, 1].set_ylabel('Abstention Rate')
    axes[1, 1].set_title('Abstention Rate vs Threshold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "exp3_threshold_curves.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved comprehensive plot to {output_path}")


def save_results(results: Dict[str, Any], output_dir: str = "results/metrics"):
    """Save results to CSV."""
    os.makedirs(output_dir, exist_ok=True)
    
    threshold_results = results["threshold_results"]
    thresholds = sorted(threshold_results.keys())
    
    # Save CSV
    csv_path = os.path.join(output_dir, "exp3_threshold_sweep.csv")
    
    metric_names = [
        "factual_precision", "factual_recall", "hallucination_rate",
        "verified_f1", "f1_score", "exact_match", "abstention_rate"
    ]
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["threshold"] + metric_names)
        
        for threshold in thresholds:
            aggregated = threshold_results[threshold]["aggregated_metrics"]
            row = [threshold]
            for metric_name in metric_names:
                value = aggregated.get(metric_name, {}).get("mean", 0)
                row.append(f"{value:.4f}")
            writer.writerow(row)
    
    print(f"✓ Saved results to {csv_path}")
    
    # Save JSON
    json_path = os.path.join(output_dir, "exp3_threshold_tuning.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved full results to {json_path}")


def main():
    """Main experiment function."""
    # Parse arguments
    args = parse_experiment_args(description="Experiment 3: Threshold Tuning")
    
    # Load config
    config = load_config(args.config)
    
    # Set random seed
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Get thresholds from config
    thresholds = config.get("verification", {}).get("threshold_sweep", [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9])
    
    # Determine sample limit
    sample_limit = args.limit
    if args.dry_run:
        sample_limit = 30
        print("⚠ DRY RUN MODE: Using 30 samples")
    elif sample_limit is None:
        sample_limit = config.get("datasets", {}).get("sample_limit")
    
    # Load dataset
    print("Loading dataset...")
    examples = load_dataset_from_config(config, split=args.split)
    
    # Apply sample limit if specified
    if sample_limit:
        examples = examples[:sample_limit]
        print(f"Limited to {len(examples)} examples")
    
    # Prepare for experiments
    queries, ground_truths, relevant_docs, corpus = prepare_for_experiments(examples)
    
    print(f"Loaded {len(queries)} queries")
    print(f"Corpus size: {len(corpus)}")
    print(f"Testing {len(thresholds)} thresholds: {thresholds}")
    
    # Get metadata
    commit_hash = get_commit_hash()
    timestamp = get_timestamp()
    dataset_name = config.get("datasets", {}).get("active", "unknown")
    
    # Setup W&B
    wandb_run = None
    if not args.no_wandb:
        wandb_run = setup_wandb(
            project_name="SelfVerifyRAG",
            run_name="exp3_threshold_tuning",
            config={
                "experiment": "exp3_threshold_tuning",
                "dataset": dataset_name,
                "split": args.split,
                "sample_limit": sample_limit,
                "seed": seed,
                "thresholds": thresholds
            },
            enabled=True
        )
        
        # Log metadata
        log_metadata(
            dataset_name=dataset_name,
            split=args.split,
            sample_limit=sample_limit,
            commit_hash=commit_hash,
            timestamp=timestamp
        )
    
    # Run experiment
    results = run_threshold_tuning(
        queries=queries,
        ground_truths=ground_truths,
        relevant_docs=relevant_docs,
        corpus=corpus,
        config=config,
        thresholds=thresholds,
        seed=seed,
        wandb_run=wandb_run
    )
    
    # Add metadata to results
    results["metadata"] = {
        "dataset": dataset_name,
        "split": args.split,
        "sample_limit": sample_limit,
        "seed": seed,
        "commit_hash": commit_hash,
        "timestamp": timestamp,
        "thresholds": thresholds,
        "total_queries": len(queries)
    }
    
    # Save results
    save_results(results)
    
    # Generate plots
    plot_threshold_curves(results)
    
    # Print summary
    print("\n" + "=" * 70)
    print("Experiment 3: Threshold Tuning Results")
    print("=" * 70)
    
    optimal_threshold = results.get("optimal_threshold")
    optimal_f1 = results.get("optimal_verified_f1", 0)
    
    print(f"Optimal threshold (τ): {optimal_threshold}")
    print(f"Optimal Verified F1: {optimal_f1:.4f}")
    
    # Print metrics for each threshold
    threshold_results = results["threshold_results"]
    thresholds = sorted(threshold_results.keys())
    
    print("\nThreshold Results:")
    print(f"{'τ':<8} {'Factual Prec':<15} {'Verified F1':<15} {'Abstention':<15}")
    print("-" * 60)
    for threshold in thresholds:
        aggregated = threshold_results[threshold]["aggregated_metrics"]
        factual_precision = aggregated.get("factual_precision", {}).get("mean", 0)
        verified_f1 = aggregated.get("verified_f1", {}).get("mean", 0)
        abstention_rate = aggregated.get("abstention_rate", {}).get("mean", 0)
        
        marker = " *" if threshold == optimal_threshold else ""
        print(f"{threshold:<8} {factual_precision:<15.4f} {verified_f1:<15.4f} {abstention_rate:<15.4f}{marker}")
    
    print("\n" + "=" * 70)
    
    # Close W&B run
    if wandb_run:
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass
    
    return results


if __name__ == "__main__":
    results = main()
