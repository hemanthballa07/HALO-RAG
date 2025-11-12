"""
Experiment 2: Retrieval Comparison
Dense vs Sparse vs Hybrid vs Hybrid+Rerank
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
from src.retrieval import HybridRetriever, CrossEncoderReranker
from src.evaluation import EvaluationMetrics, StatisticalTester
from src.utils import setup_wandb, log_metrics, log_metadata, get_commit_hash, get_timestamp
from src.utils.cli import parse_experiment_args


def load_config(config_path: str = "config/config.yaml"):
    """Load configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_retrieval_comparison(
    queries: List[str],
    ground_truths: List[str],
    relevant_docs: List[List[int]],
    corpus: List[str],
    config: Dict[str, Any],
    seed: int = 42,
    wandb_run: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Compare dense, sparse, hybrid, and hybrid+rerank retrieval methods.
    
    Args:
        queries: List of queries
        ground_truths: List of ground truth answers (for coverage calculation)
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
    
    # Initialize retriever
    print("Building retrieval index...")
    retriever = HybridRetriever(
        dense_weight=config.get("retrieval", {}).get("fusion", {}).get("dense_weight", 0.6),
        sparse_weight=config.get("retrieval", {}).get("fusion", {}).get("sparse_weight", 0.4),
        device=device
    )
    retriever.build_index(corpus)
    
    # Initialize reranker
    print("Initializing reranker...")
    reranker = CrossEncoderReranker(
        model_name=config.get("retrieval", {}).get("reranker", {}).get("model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        device=device
    )
    
    # Initialize evaluator
    evaluator = EvaluationMetrics()
    stats_tester = StatisticalTester(alpha=0.05)
    
    # Results for each method
    methods = {
        "dense": [],
        "sparse": [],
        "hybrid": [],
        "hybrid_rerank": []
    }
    
    print(f"Running retrieval comparison on {len(queries)} queries...")
    
    for idx, (query, gt, rel_docs) in enumerate(tqdm(
        zip(queries, ground_truths, relevant_docs),
        total=len(queries),
        desc="Retrieval comparison"
    )):
        # Dense retrieval
        dense_retrieved = retriever.retrieve_dense_only(query, top_k=20)
        dense_ids = [doc[0] for doc in dense_retrieved]
        dense_texts = [doc[1] for doc in dense_retrieved]
        
        dense_metrics = {
            "recall@5": evaluator.recall_at_k(dense_ids, rel_docs, 5),
            "recall@10": evaluator.recall_at_k(dense_ids, rel_docs, 10),
            "recall@20": evaluator.recall_at_k(dense_ids, rel_docs, 20),
            "mrr": evaluator.mrr(dense_ids, rel_docs),
            "ndcg@10": evaluator.ndcg_at_k(dense_ids, rel_docs, 10),
            "coverage": evaluator.coverage(gt, dense_texts) if gt else 0.0
        }
        methods["dense"].append(dense_metrics)
        
        # Sparse retrieval
        sparse_retrieved = retriever.retrieve_sparse_only(query, top_k=20)
        sparse_ids = [doc[0] for doc in sparse_retrieved]
        sparse_texts = [doc[1] for doc in sparse_retrieved]
        
        sparse_metrics = {
            "recall@5": evaluator.recall_at_k(sparse_ids, rel_docs, 5),
            "recall@10": evaluator.recall_at_k(sparse_ids, rel_docs, 10),
            "recall@20": evaluator.recall_at_k(sparse_ids, rel_docs, 20),
            "mrr": evaluator.mrr(sparse_ids, rel_docs),
            "ndcg@10": evaluator.ndcg_at_k(sparse_ids, rel_docs, 10),
            "coverage": evaluator.coverage(gt, sparse_texts) if gt else 0.0
        }
        methods["sparse"].append(sparse_metrics)
        
        # Hybrid retrieval
        hybrid_retrieved = retriever.retrieve(query, top_k=20)
        hybrid_ids = [doc[0] for doc in hybrid_retrieved]
        hybrid_texts = [doc[1] for doc in hybrid_retrieved]
        
        hybrid_metrics = {
            "recall@5": evaluator.recall_at_k(hybrid_ids, rel_docs, 5),
            "recall@10": evaluator.recall_at_k(hybrid_ids, rel_docs, 10),
            "recall@20": evaluator.recall_at_k(hybrid_ids, rel_docs, 20),
            "mrr": evaluator.mrr(hybrid_ids, rel_docs),
            "ndcg@10": evaluator.ndcg_at_k(hybrid_ids, rel_docs, 10),
            "coverage": evaluator.coverage(gt, hybrid_texts) if gt else 0.0
        }
        methods["hybrid"].append(hybrid_metrics)
        
        # Hybrid + Rerank
        # Rerank top 50, then select top 5
        hybrid_top50 = retriever.retrieve(query, top_k=50)
        hybrid_top50_texts = [doc[1] for doc in hybrid_top50]
        hybrid_top50_ids = [doc[0] for doc in hybrid_top50]
        
        # Rerank
        reranked = reranker.rerank_with_ids(
            query,
            hybrid_top50_ids,
            hybrid_top50_texts,
            top_k=20
        )
        rerank_ids = [doc[0] for doc in reranked]
        rerank_texts = [doc[1] for doc in reranked]
        
        hybrid_rerank_metrics = {
            "recall@5": evaluator.recall_at_k(rerank_ids, rel_docs, 5),
            "recall@10": evaluator.recall_at_k(rerank_ids, rel_docs, 10),
            "recall@20": evaluator.recall_at_k(rerank_ids, rel_docs, 20),
            "mrr": evaluator.mrr(rerank_ids, rel_docs),
            "ndcg@10": evaluator.ndcg_at_k(rerank_ids, rel_docs, 10),
            "coverage": evaluator.coverage(gt, rerank_texts) if gt else 0.0
        }
        methods["hybrid_rerank"].append(hybrid_rerank_metrics)
        
        # Log to W&B periodically
        if wandb_run and (idx + 1) % 100 == 0:
            for method_name, method_results in methods.items():
                if method_results:
                    avg_metrics = {
                        k: np.mean([m[k] for m in method_results])
                        for k in method_results[0].keys()
                    }
                    log_metrics(avg_metrics, step=idx + 1, prefix=f"retrieval/{method_name}/")
    
    # Aggregate metrics
    print("\nAggregating metrics...")
    aggregated = {}
    metric_names = ["recall@5", "recall@10", "recall@20", "mrr", "ndcg@10", "coverage"]
    
    for method_name, method_results in methods.items():
        aggregated[method_name] = {}
        for metric_name in metric_names:
            scores = [r[metric_name] for r in method_results]
            aggregated[method_name][metric_name] = {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "min": float(np.min(scores)),
                "max": float(np.max(scores)),
                "scores": scores
            }
    
    # Statistical comparison
    comparisons = {}
    for metric_name in ["recall@20", "mrr", "ndcg@10", "coverage"]:
        dense_scores = [r[metric_name] for r in methods["dense"]]
        sparse_scores = [r[metric_name] for r in methods["sparse"]]
        hybrid_scores = [r[metric_name] for r in methods["hybrid"]]
        hybrid_rerank_scores = [r[metric_name] for r in methods["hybrid_rerank"]]
        
        # Compare hybrid_rerank vs others
        comp_dense = stats_tester.compare_metrics(
            dense_scores, hybrid_rerank_scores, f"{metric_name}_hybrid_rerank_vs_dense"
        )
        comp_sparse = stats_tester.compare_metrics(
            sparse_scores, hybrid_rerank_scores, f"{metric_name}_hybrid_rerank_vs_sparse"
        )
        comp_hybrid = stats_tester.compare_metrics(
            hybrid_scores, hybrid_rerank_scores, f"{metric_name}_hybrid_rerank_vs_hybrid"
        )
        
        comparisons[metric_name] = {
            "hybrid_rerank_vs_dense": comp_dense,
            "hybrid_rerank_vs_sparse": comp_sparse,
            "hybrid_rerank_vs_hybrid": comp_hybrid
        }
    
    # Log final metrics to W&B
    if wandb_run:
        for method_name, metrics in aggregated.items():
            final_metrics = {k: v["mean"] for k, v in metrics.items()}
            log_metrics(final_metrics, prefix=f"retrieval/{method_name}/final/")
    
    return {
            "aggregated_metrics": aggregated,
        "statistical_comparisons": comparisons,
        "methods": methods,
        "total_queries": len(queries)
    }


def plot_retrieval_comparison(results: Dict[str, Any], output_dir: str = "results/figures"):
    """Plot retrieval comparison bar charts."""
    os.makedirs(output_dir, exist_ok=True)
    
    aggregated = results["aggregated_metrics"]
    methods = list(aggregated.keys())
    metric_names = ["recall@20", "mrr", "ndcg@10", "coverage"]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, metric_name in enumerate(metric_names):
        ax = axes[idx]
        
        # Get means and stds for each method
        means = [aggregated[method][metric_name]["mean"] for method in methods]
        stds = [aggregated[method][metric_name]["std"] for method in methods]
        
        # Create bar plot
        x_pos = np.arange(len(methods))
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.8, edgecolor='black')
        
        # Color bars
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # Add value labels on bars
        for i, (mean, std) in enumerate(zip(means, stds)):
            ax.text(i, mean + std + 0.01, f'{mean:.3f}', 
                   ha='center', va='bottom', fontsize=9)
        
        # Formatting
        ax.set_xlabel('Retrieval Method', fontsize=11)
        ax.set_ylabel(metric_name.replace('@', '@'), fontsize=11)
        ax.set_title(f'{metric_name.replace("@", "@").replace("_", " ").title()}', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in methods], rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, max(means) * 1.2 if max(means) > 0 else 1.0)
        
        # Add target line for recall@20 and coverage
        if metric_name == "recall@20":
            ax.axhline(y=0.95, color='r', linestyle='--', alpha=0.5, label='Target (0.95)')
            ax.legend()
        elif metric_name == "coverage":
            ax.axhline(y=0.90, color='r', linestyle='--', alpha=0.5, label='Target (0.90)')
            ax.legend()
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, "exp2_retrieval_bars.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved plot to {output_path}")


def save_results(results: Dict[str, Any], output_dir: str = "results/metrics"):
    """Save results to JSON and CSV."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save per-config JSONs
    aggregated = results["aggregated_metrics"]
    for method_name, metrics in aggregated.items():
        json_path = os.path.join(output_dir, f"exp2_retrieval_{method_name}.json")
        with open(json_path, 'w') as f:
            json.dump({
                "method": method_name,
                "metrics": metrics
            }, f, indent=2)
    
    # Save consolidated CSV
    csv_path = os.path.join(output_dir, "exp2_retrieval.csv")
    metric_names = ["recall@5", "recall@10", "recall@20", "mrr", "ndcg@10", "coverage"]
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["method", "metric", "mean", "std", "min", "max"])
        
        for method_name in aggregated.keys():
            for metric_name in metric_names:
                if metric_name in aggregated[method_name]:
                    stats = aggregated[method_name][metric_name]
                    writer.writerow([
                        method_name,
                        metric_name,
                        f"{stats['mean']:.4f}",
                        f"{stats['std']:.4f}",
                        f"{stats['min']:.4f}",
                        f"{stats['max']:.4f}"
                    ])
    
    print(f"✓ Saved results to {csv_path}")
    
    # Save full results JSON
    json_path = os.path.join(output_dir, "exp2_retrieval_comparison.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved full results to {json_path}")


def main():
    """Main experiment function."""
    # Parse arguments
    args = parse_experiment_args(description="Experiment 2: Retrieval Comparison")
    
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
    
    # Get metadata
    commit_hash = get_commit_hash()
    timestamp = get_timestamp()
    dataset_name = config.get("datasets", {}).get("active", "unknown")
    
    # Setup W&B
    wandb_run = None
    if not args.no_wandb:
        wandb_run = setup_wandb(
            project_name="SelfVerifyRAG",
            run_name="exp2_retrieval_comparison",
            config={
                "experiment": "exp2_retrieval_comparison",
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
    results = run_retrieval_comparison(
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
        "total_queries": len(queries)
    }
    
    # Save results
    save_results(results)
    
    # Generate plots
    plot_retrieval_comparison(results)
    
    # Print summary
    print("\n" + "=" * 70)
    print("Experiment 2: Retrieval Comparison Results")
    print("=" * 70)
    
    aggregated = results["aggregated_metrics"]
    for method_name, metrics in aggregated.items():
        print(f"\n{method_name.upper().replace('_', ' ')}:")
        print(f"  Recall@20: {metrics['recall@20']['mean']:.4f} ± {metrics['recall@20']['std']:.4f}")
        print(f"  Coverage: {metrics['coverage']['mean']:.4f} ± {metrics['coverage']['std']:.4f}")
        print(f"  MRR: {metrics['mrr']['mean']:.4f} ± {metrics['mrr']['std']:.4f}")
        print(f"  NDCG@10: {metrics['ndcg@10']['mean']:.4f} ± {metrics['ndcg@10']['std']:.4f}")
    
    # Check if targets are met
    hybrid_rerank = aggregated.get("hybrid_rerank", {})
    recall20 = hybrid_rerank.get("recall@20", {}).get("mean", 0)
    coverage = hybrid_rerank.get("coverage", {}).get("mean", 0)
    
    print("\n" + "=" * 70)
    print("Target Check (Hybrid+Rerank):")
    print(f"  Recall@20 ≥ 0.95: {'✓' if recall20 >= 0.95 else '✗'} ({recall20:.4f})")
    print(f"  Coverage ≥ 0.90: {'✓' if coverage >= 0.90 else '✗'} ({coverage:.4f})")
    print("=" * 70)
    
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
