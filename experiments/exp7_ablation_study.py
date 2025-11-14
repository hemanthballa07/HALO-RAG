"""
Experiment 7: Ablation Study
Component-wise contribution analysis
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

from src.data import load_dataset_from_config, prepare_for_experiments
from src.pipeline import SelfVerificationRAGPipeline
from src.verification import EntailmentVerifier, ClaimExtractor
from src.verification.lexical_verifier import LexicalOverlapVerifier
from src.evaluation import EvaluationMetrics
from src.utils import setup_wandb, log_metrics, log_metadata, get_commit_hash, get_timestamp


def load_config(config_path: str = "config/config.yaml"):
    """Load configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


class AblationPipeline(SelfVerificationRAGPipeline):
    """Pipeline variant for ablation study with component disabling."""
    
    def __init__(
        self,
        corpus: List[str],
        enable_reranking: bool = True,
        enable_verification: bool = True,
        enable_revision: bool = True,
        use_lexical_verifier: bool = False,
        **kwargs
    ):
        """
        Initialize ablation pipeline.
        
        Args:
            corpus: List of documents
            enable_reranking: Whether to use reranking
            enable_verification: Whether to use verification
            enable_revision: Whether to use revision
            use_lexical_verifier: Whether to use lexical verifier instead of NLI
            **kwargs: Other pipeline arguments
        """
        # Initialize base pipeline
        super().__init__(corpus=corpus, enable_revision=enable_revision, **kwargs)
        
        self.enable_reranking = enable_reranking
        self.enable_verification = enable_verification
        self.use_lexical_verifier = use_lexical_verifier
        
        # Replace verifier if needed
        if use_lexical_verifier:
            self.verifier = LexicalOverlapVerifier(
                threshold=self.verifier.threshold if hasattr(self.verifier, 'threshold') else 0.75
            )
        
        # Note: Verification is always run for metrics computation
        # But if enable_verification=False, we don't use it for filtering
    
    def generate(
        self,
        query: str,
        top_k_retrieve: int = 20,
        top_k_rerank: int = 5,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate answer with optional component disabling.
        
        Args:
            query: Query string
            top_k_retrieve: Number of documents to retrieve
            top_k_rerank: Number of documents to rerank
            **kwargs: Other generation arguments
        
        Returns:
            Dictionary with generation results
        """
        # Step 1: Hybrid retrieval
        retrieved_docs = self.retriever.retrieve(query, top_k=top_k_retrieve)
        retrieved_texts = [doc[1] for doc in retrieved_docs]
        retrieved_ids = [doc[0] for doc in retrieved_docs]
        
        # Step 2: Reranking (if enabled)
        if self.enable_reranking:
            reranked_docs = self.reranker.rerank(
                query,
                retrieved_texts,
                top_k=top_k_rerank
            )
            reranked_ids = [retrieved_ids[doc[0]] for doc in reranked_docs]
            reranked_texts = [doc[1] for doc in reranked_docs]
            context = " ".join(reranked_texts)
        else:
            # Skip reranking: use retrieved docs directly
            reranked_ids = retrieved_ids[:top_k_rerank]
            reranked_texts = retrieved_texts[:top_k_rerank]
            context = " ".join(reranked_texts)
        
        # Step 3: Generation
        generated_text = self.generator.generate(query, context, **kwargs)
        
        # Step 4: Claim extraction
        claims = self.claim_extractor.extract_claims(generated_text)
        
        # Step 5: Verification
        # Always run verification for metrics computation, but only use it for filtering if enabled
        verification_results = self.verifier.verify_generation(
            generated_text,
            reranked_texts,
            claims
        )
        
        # If verification is disabled, don't use it for filtering (but keep results for metrics)
        if not self.enable_verification:
            # Mark as verified so revision doesn't trigger
            verification_results["verified"] = True
        
        # Step 6: Revision (if enabled and verification failed)
        revision_iterations = 0
        if self.enable_revision and self.revision_strategy and self.enable_verification:
            if not verification_results.get("verified", False):
                max_revision_iterations = kwargs.get("max_revision_iterations", self.max_revision_iterations)
                for iteration in range(max_revision_iterations):
                    revised_text, new_verification = self.revision_strategy.revise(
                        query=query,
                        initial_generation=generated_text,
                        verification_results=verification_results,
                        retrieval_fn=lambda q, k: self.retriever.retrieve(q, top_k=k),
                        generation_fn=lambda q, ctx: self.generator.generate(q, ctx),
                        verification_fn=lambda gen, ctxs, clms: self.verifier.verify_generation(
                            gen, ctxs, clms
                        ),
                        iteration=iteration
                    )
                    
                    generated_text = revised_text
                    verification_results = new_verification
                    revision_iterations += 1
                    
                    if verification_results.get("verified", False):
                        break
        
        return {
            "query": query,
            "generated_text": generated_text,
            "retrieved_docs": retrieved_ids,
            "retrieved_texts": retrieved_texts,
            "reranked_docs": reranked_ids,
            "reranked_texts": reranked_texts,
            "context": context,
            "claims": claims,
            "verification_results": verification_results,
            "revision_iterations": revision_iterations,
            "verified": verification_results.get("verified", False)
        }


def run_ablation_study(
    queries: List[str],
    ground_truths: List[str],
    relevant_docs: List[List[int]],
    corpus: List[str],
    config: Dict[str, Any],
    seed: int = 42,
    limit: Optional[int] = None,
    wandb_run: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Run ablation study on pipeline components.
    
    Args:
        queries: List of queries
        ground_truths: List of ground truth answers
        relevant_docs: List of relevant document IDs for each query
        corpus: List of documents
        config: Configuration dictionary
        seed: Random seed
        limit: Limit number of examples (for testing)
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
    
    # Limit examples if specified
    if limit:
        queries = queries[:limit]
        ground_truths = ground_truths[:limit]
        relevant_docs = relevant_docs[:limit]
        print(f"Limited to {limit} examples for testing")
    
    # Initialize evaluator
    evaluator = EvaluationMetrics()
    
    # Define ablation variants
    variants = {
        "full": {
            "enable_reranking": True,
            "enable_verification": True,
            "enable_revision": True,
            "use_lexical_verifier": False
        },
        "no_reranking": {
            "enable_reranking": False,
            "enable_verification": True,
            "enable_revision": True,
            "use_lexical_verifier": False
        },
        "no_verification": {
            "enable_reranking": True,
            "enable_verification": False,
            "enable_revision": False,  # No revision if no verification
            "use_lexical_verifier": False
        },
        "no_revision": {
            "enable_reranking": True,
            "enable_verification": True,
            "enable_revision": False,
            "use_lexical_verifier": False
        },
        "simple_verifier": {
            "enable_reranking": True,
            "enable_verification": True,
            "enable_revision": True,
            "use_lexical_verifier": True
        }
    }
    
    # Run each variant
    all_results = {}
    all_metrics = {}
    
    for variant_name, variant_params in variants.items():
        print(f"\n{'='*60}")
        print(f"Running variant: {variant_name}")
        print(f"{'='*60}")
        
        # Initialize pipeline with variant
        pipeline = AblationPipeline(
            corpus=corpus,
            device=device,
            use_qlora=config.get("generation", {}).get("qlora", {}).get("training_enabled", False),
            **variant_params
        )
        
        # Run experiments
        results = []
        metrics_list = []
        
        for idx, (query, gt, rel_docs) in enumerate(tqdm(
            zip(queries, ground_truths, relevant_docs),
            total=len(queries),
            desc=f"{variant_name}"
        )):
            try:
                # Generate answer
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
                
                metrics_list.append(metrics)
                results.append({
                    "query": query,
                    "ground_truth": gt,
                    "generated": result["generated_text"],
                    "metrics": metrics
                })
                
            except Exception as e:
                print(f"Error processing query {idx} in {variant_name}: {e}")
                continue
        
        all_results[variant_name] = results
        all_metrics[variant_name] = metrics_list
        
        # Log to W&B
        if wandb_run and metrics_list:
            avg_metrics = {
                k: np.mean([m[k] for m in metrics_list if k in m])
                for k in metrics_list[0].keys()
            }
            log_metrics(avg_metrics, prefix=f"ablation/{variant_name}/")
    
    # Aggregate metrics
    print("\n" + "="*60)
    print("Aggregating metrics...")
    print("="*60)
    
    metric_names = [
        "verified_f1",
        "factual_precision",
        "hallucination_rate",
        "exact_match",
        "f1_score"
    ]
    
    aggregated = {}
    for variant_name, metrics_list in all_metrics.items():
        aggregated[variant_name] = {}
        for metric_name in metric_names:
            scores = [m[metric_name] for m in metrics_list if metric_name in m]
            if scores:
                aggregated[variant_name][metric_name] = {
                    "mean": float(np.mean(scores)),
                    "std": float(np.std(scores)),
                    "scores": scores
                }
    
    # Compute drops vs full system
    drops = {}
    if "full" in aggregated:
        full_metrics = aggregated["full"]
        for variant_name in variants.keys():
            if variant_name != "full":
                drops[variant_name] = {}
                for metric_name in metric_names:
                    if metric_name in full_metrics and metric_name in aggregated[variant_name]:
                        full_mean = full_metrics[metric_name]["mean"]
                        variant_mean = aggregated[variant_name][metric_name]["mean"]
                        # Compute drop (negative means improvement)
                        drop = full_mean - variant_mean
                        drop_pct = (drop / full_mean * 100) if full_mean > 0 else 0.0
                        drops[variant_name][metric_name] = {
                            "absolute": drop,
                            "percent": drop_pct
                        }
    
    # Print results
    print("\n" + "="*60)
    print("ABLATION STUDY RESULTS")
    print("="*60)
    for variant_name, metrics in aggregated.items():
        print(f"\n{variant_name.upper()}:")
        for metric_name, stats in metrics.items():
            print(f"  {metric_name}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    print("\n" + "="*60)
    print("DROPS VS FULL SYSTEM")
    print("="*60)
    for variant_name, drop_metrics in drops.items():
        print(f"\n{variant_name.upper()}:")
        for metric_name, drop_stats in drop_metrics.items():
            print(f"  {metric_name}: {drop_stats['absolute']:.4f} ({drop_stats['percent']:+.2f}%)")
    
    return {
        "aggregated": aggregated,
        "drops": drops,
        "all_results": all_results,
        "variants": list(variants.keys())
    }


def save_results(results: Dict[str, Any], output_dir: str = "results/metrics"):
    """Save results to CSV and JSON."""
    os.makedirs(output_dir, exist_ok=True)
    
    aggregated = results["aggregated"]
    drops = results["drops"]
    variants = results["variants"]
    
    # Save CSV
    csv_path = os.path.join(output_dir, "exp7_ablation.csv")
    metric_names = ["verified_f1", "factual_precision", "hallucination_rate", "exact_match", "f1_score"]
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header
        header = ["variant"] + metric_names
        writer.writerow(header)
        
        # Data rows
        for variant in variants:
            row = [variant]
            for metric in metric_names:
                if variant in aggregated and metric in aggregated[variant]:
                    row.append(f"{aggregated[variant][metric]['mean']:.4f}")
                else:
                    row.append("")
            writer.writerow(row)
    
    print(f"\nSaved metrics to {csv_path}")
    
    # Save JSON
    json_path = os.path.join(output_dir, "exp7_ablation.json")
    with open(json_path, 'w') as f:
        json.dump({
            "aggregated": aggregated,
            "drops": drops,
            "timestamp": get_timestamp(),
            "commit_hash": get_commit_hash()
        }, f, indent=2)
    
    print(f"Saved full results to {json_path}")


def plot_ablation_results(results: Dict[str, Any], output_dir: str = "results/figures"):
    """Plot ablation study results."""
    os.makedirs(output_dir, exist_ok=True)
    
    aggregated = results["aggregated"]
    drops = results["drops"]
    variants = results["variants"]
    
    # Remove "full" from variants for plotting drops
    ablation_variants = [v for v in variants if v != "full"]
    
    # Metrics to plot
    metric_names = ["verified_f1", "factual_precision", "hallucination_rate", "exact_match", "f1_score"]
    metric_labels = {
        "verified_f1": "Verified F1",
        "factual_precision": "Factual Precision",
        "hallucination_rate": "Hallucination Rate",
        "exact_match": "Exact Match",
        "f1_score": "F1 Score"
    }
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    # Plot 1: Absolute values for all variants
    ax = axes[0]
    variant_labels = [v.replace("_", " ").title() for v in variants]
    x = np.arange(len(variants))
    width = 0.15
    
    for i, metric in enumerate(metric_names):
        values = [aggregated[v][metric]["mean"] if v in aggregated and metric in aggregated[v] else 0.0
                 for v in variants]
        ax.bar(x + i * width, values, width, label=metric_labels[metric], alpha=0.8)
    
    ax.set_xlabel("Variant")
    ax.set_ylabel("Score")
    ax.set_title("Ablation Study: Absolute Metrics")
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(variant_labels, rotation=45, ha='right')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 2-6: Drops vs full system for each metric
    for idx, metric in enumerate(metric_names):
        ax = axes[idx + 1]
        
        variant_labels_ablation = [v.replace("_", " ").title() for v in ablation_variants]
        drops_values = [drops[v][metric]["percent"] if v in drops and metric in drops[v] else 0.0
                       for v in ablation_variants]
        
        # Use different colors for positive/negative drops
        colors = ['red' if d > 0 else 'green' for d in drops_values]
        bars = ax.barh(variant_labels_ablation, drops_values, color=colors, alpha=0.7)
        
        ax.set_xlabel("Drop (%)")
        ax.set_title(f"{metric_labels[metric]}: Drop vs Full System")
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, drops_values)):
            ax.text(val, i, f'{val:+.2f}%', va='center', 
                   ha='left' if val > 0 else 'right', fontsize=9)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "exp7_ablation_bars.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plot to {plot_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Experiment 7: Ablation Study")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                       help="Path to config file")
    parser.add_argument("--split", type=str, default="validation",
                       help="Dataset split to use")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of examples (for testing)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--dry-run", action="store_true",
                       help="Dry run mode (limit to 50 examples)")
    parser.add_argument("--no-wandb", action="store_true",
                       help="Disable W&B logging")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Set limit for dry-run
    if args.dry_run:
        args.limit = 50
        print("Dry-run mode: limiting to 50 examples")
    
    # Setup W&B
    wandb_run = None
    if not args.no_wandb:
        try:
            wandb_run = setup_wandb(
                project="SelfVerifyRAG",
                run_name="exp7_ablation",
                config=config
            )
            log_metadata({
                "experiment": "exp7_ablation",
                "split": args.split,
                "limit": args.limit,
                "seed": args.seed,
                "commit_hash": get_commit_hash(),
                "timestamp": get_timestamp()
            })
        except Exception as e:
            print(f"W&B setup failed: {e}")
    
    # Load dataset
    print("Loading dataset...")
    examples = load_dataset_from_config(config, split=args.split)
    
    # Prepare data for experiments
    queries, ground_truths, relevant_docs, corpus = prepare_for_experiments(examples)
    
    print(f"Loaded {len(queries)} examples")
    print(f"Corpus size: {len(corpus)} documents")
    
    # Run ablation study
    results = run_ablation_study(
        queries=queries,
        ground_truths=ground_truths,
        relevant_docs=relevant_docs,
        corpus=corpus,
        config=config,
        seed=args.seed,
        limit=args.limit,
        wandb_run=wandb_run
    )
    
    # Save results
    save_results(results)
    
    # Plot results
    plot_ablation_results(results)
    
    # Finish W&B run
    if wandb_run:
        wandb_run.finish()
    
    print("\n" + "="*60)
    print("Experiment 7: Ablation Study completed!")
    print("="*60)


if __name__ == "__main__":
    main()
