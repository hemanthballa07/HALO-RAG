"""
Experiment 1: Baseline Comparison
Standard RAG (no verification) baseline
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

from src.data import load_dataset_from_config, prepare_for_experiments
from src.pipeline import SelfVerificationRAGPipeline
from src.evaluation import EvaluationMetrics, StatisticalTester
from src.utils import setup_wandb, log_metrics, log_metadata, get_commit_hash, get_timestamp
from src.utils.cli import parse_experiment_args


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
    config: Dict[str, Any],
    seed: int = 42,
    wandb_run: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Run baseline comparison experiment (no verification).
    
    Args:
        queries: List of queries
        ground_truths: List of ground truth answers
        relevant_docs: List of relevant document IDs for each query
        corpus: List of documents
        config: Configuration dictionary
        seed: Random seed
        wandb_run: W&B run object (optional)
    
    Returns:
        Dictionary with results and metrics
    """
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize pipeline (baseline: no verification, no revision)
    print("Initializing pipeline (baseline: no verification)...")
    pipeline = SelfVerificationRAGPipeline(
        corpus=corpus,
        device=device,
        enable_revision=False,  # No revision for baseline
        use_qlora=config.get("generation", {}).get("qlora", {}).get("training_enabled", False)
    )
    
    # Disable verification for true baseline
    # We'll still compute verification metrics but pipeline won't use them
    pipeline.verifier.threshold = 0.0  # Accept all claims for baseline
    
    # Initialize evaluator
    evaluator = EvaluationMetrics()
    
    # Run experiments
    print(f"Running baseline experiment on {len(queries)} queries...")
    results = []
    all_metrics = []
    
    for idx, (query, gt, rel_docs) in enumerate(tqdm(
        zip(queries, ground_truths, relevant_docs), 
        total=len(queries), 
        desc="Baseline"
    )):
        try:
            # Generate answer (no verification filtering)
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
            
            # Log to W&B periodically
            if wandb_run and (idx + 1) % 100 == 0:
                avg_metrics = {
                    k: np.mean([m[k] for m in all_metrics])
                    for k in all_metrics[0].keys()
                }
                log_metrics(avg_metrics, step=idx + 1, prefix="baseline/")
        
        except Exception as e:
            print(f"Error processing query {idx}: {e}")
            continue
    
    # Aggregate metrics
    print("\nAggregating metrics...")
    metric_names = [
        "recall@20", "coverage", "factual_precision", 
        "hallucination_rate", "verified_f1", "f1_score",
        "exact_match", "bleu4", "rouge_l", "abstention_rate",
        "fever_score"
    ]
    
    aggregated = {}
    for metric_name in metric_names:
        if metric_name in all_metrics[0]:
            scores = [m[metric_name] for m in all_metrics if metric_name in m]
            if scores:
                aggregated[metric_name] = {
                    "mean": float(np.mean(scores)),
                    "std": float(np.std(scores)),
                    "min": float(np.min(scores)),
                    "max": float(np.max(scores)),
                    "scores": scores
                }
    
    # Log final metrics to W&B
    if wandb_run:
        final_metrics = {k: v["mean"] for k, v in aggregated.items()}
        log_metrics(final_metrics, prefix="baseline/final/")
    
    return {
        "aggregated_metrics": aggregated,
        "individual_results": results,
        "total_queries": len(queries),
        "processed_queries": len(results)
    }


def save_results(results: Dict[str, Any], output_dir: str = "results/metrics"):
    """Save results to JSON and CSV."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save JSON
    json_path = os.path.join(output_dir, "exp1_baseline.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved results to {json_path}")
    
    # Save CSV
    csv_path = os.path.join(output_dir, "exp1_baseline.csv")
    aggregated = results["aggregated_metrics"]
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "mean", "std", "min", "max"])
        for metric_name, stats in aggregated.items():
            writer.writerow([
                metric_name,
                f"{stats['mean']:.4f}",
                f"{stats['std']:.4f}",
                f"{stats['min']:.4f}",
                f"{stats['max']:.4f}"
            ])
    print(f"✓ Saved metrics to {csv_path}")


def main():
    """Main experiment function."""
    # Parse arguments
    args = parse_experiment_args(description="Experiment 1: Baseline Comparison")
    
    # Load config
    config = load_config(args.config)
    
    # Set random seed
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Determine sample limit
    sample_limit = args.limit
    if args.dry_run:
        sample_limit = 30  # Dry run with 30 samples
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
    
    # Get metadata
    commit_hash = get_commit_hash()
    timestamp = get_timestamp()
    dataset_name = config.get("datasets", {}).get("active", "unknown")
    
    # Setup W&B
    wandb_run = None
    if not args.no_wandb:
        wandb_run = setup_wandb(
            project_name="SelfVerifyRAG",
            run_name="exp1_baseline",
            config={
                "experiment": "exp1_baseline",
                "dataset": dataset_name,
                "split": args.split,
                "sample_limit": sample_limit,
                "seed": seed
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
    results = run_baseline_experiment(
        queries=queries,
        ground_truths=ground_truths,
        relevant_docs=relevant_docs,
        corpus=corpus,
        config=config,
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
        "total_queries": len(queries),
        "processed_queries": results["processed_queries"]
    }
    
    # Save results
    save_results(results)
    
    # Print summary
    print("\n" + "=" * 70)
    print("Experiment 1: Baseline Results")
    print("=" * 70)
    aggregated = results["aggregated_metrics"]
    for metric_name, stats in aggregated.items():
        print(f"{metric_name}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    print(f"\nVerified F1: {aggregated.get('verified_f1', {}).get('mean', 0):.4f}")
    print(f"Factual Precision: {aggregated.get('factual_precision', {}).get('mean', 0):.4f}")
    print(f"Hallucination Rate: {aggregated.get('hallucination_rate', {}).get('mean', 0):.4f}")
    
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
