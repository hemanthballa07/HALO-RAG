"""
Experiment 9: Complete HALO-RAG Pipeline
Fine-tuned generator + adaptive revision strategy
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


def run_complete_pipeline_experiment(
    queries: List[str],
    ground_truths: List[str],
    relevant_docs: List[List[int]],
    corpus: List[str],
    config: Dict[str, Any],
    fine_tuned_checkpoint: Optional[str] = None,
    enable_revision: bool = True,
    max_revision_iterations: int = 3,
    seed: int = 42,
    top_k_retrieve: int = 20,
    top_k_rerank: int = 5,
    wandb_run: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Run complete HALO-RAG pipeline experiment (fine-tuned generator + revision).
    
    Args:
        queries: List of queries
        ground_truths: List of ground truth answers
        relevant_docs: List of relevant document IDs for each query
        corpus: List of documents
        config: Configuration dictionary
        fine_tuned_checkpoint: Path to fine-tuned generator checkpoint
        enable_revision: Whether to enable adaptive revision
        max_revision_iterations: Maximum revision attempts
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
    
    use_qlora = config.get("generation", {}).get("qlora", {}).get("training_enabled", False)
    
    # Initialize pipeline (with optional fine-tuned generator + revision)
    if fine_tuned_checkpoint:
        print("Initializing pipeline (fine-tuned generator + revision)...")
    else:
        print("Initializing pipeline (base generator + revision)...")
    # Get verifier model from config
    verifier_model = config.get("verification", {}).get("entailment_model", "cross-encoder/nli-deberta-v3-base")
    verifier_threshold = config.get("verification", {}).get("threshold", 0.75)
    
    # Build pipeline kwargs
    pipeline_kwargs = {
        "corpus": corpus,
        "device": device,
        "enable_revision": enable_revision,  # Enable revision strategy
        "max_revision_iterations": max_revision_iterations,
        "use_qlora": use_qlora,
        "verifier_model": verifier_model,
        "entailment_threshold": verifier_threshold
    }
    
    # Only add checkpoint if provided
    if fine_tuned_checkpoint:
        pipeline_kwargs["generator_lora_checkpoint"] = fine_tuned_checkpoint
    
    pipeline = SelfVerificationRAGPipeline(**pipeline_kwargs)
    
    if fine_tuned_checkpoint:
        print(f"✓ Pipeline initialized with fine-tuned checkpoint: {fine_tuned_checkpoint}")
    else:
        print(f"✓ Pipeline initialized with base (non-fine-tuned) generator")
    print(f"✓ Revision strategy: {'Enabled' if enable_revision else 'Disabled'}")
    
    # Initialize evaluator
    evaluator = EvaluationMetrics()
    
    # Run experiments
    print(f"Running complete pipeline experiment on {len(queries)} queries...")
    results = []
    all_metrics = []
    
    for idx, (query, gt, rel_docs) in enumerate(tqdm(
        zip(queries, ground_truths, relevant_docs), 
        total=len(queries), 
        desc="Complete Pipeline"
    )):
        try:
            # Generate answer (with verification + revision)
            result = pipeline.generate(query, top_k_retrieve=top_k_retrieve, top_k_rerank=top_k_rerank)
            
            # Get retrieved texts for coverage calculation
            retrieved_texts = result.get("reranked_texts", result.get("retrieved_texts", []))
            
            # Extract revision iterations (only in exp9)
            revision_iterations = result.get("revision_iterations", 0)
            
            # Extract claims from ground truth for factual recall calculation
            ground_truth_claims = pipeline.claim_extractor.extract_claims(gt)
            
            # Extract claims from generated text (already extracted in pipeline, but get them)
            generated_claims = result.get("claims", [])
            if not generated_claims:
                # Fallback: extract claims if not in result
                generated_claims = pipeline.claim_extractor.extract_claims(result["generated_text"])
            
            # Format claims for MNLI model (as they are passed to the entailment verifier)
            combined_context = result.get("context", "")
            formatted_generated_claims = []
            formatted_ground_truth_claims = []
            
            # Format generated claims
            for claim in generated_claims:
                formatted_claim = pipeline.verifier._format_claim_for_verification(claim, combined_context)
                formatted_generated_claims.append({
                    "original_claim": claim,
                    "formatted_claim": formatted_claim
                })
            
            # Format ground truth claims
            for claim in ground_truth_claims:
                formatted_claim = pipeline.verifier._format_claim_for_verification(claim, combined_context)
                formatted_ground_truth_claims.append({
                    "original_claim": claim,
                    "formatted_claim": formatted_claim
                })
            
            # Compute metrics
            # Get abstention flag from result
            abstained = result.get("abstained", False)
            
            metrics = evaluator.compute_all_metrics(
                retrieved_docs=result["retrieved_docs"],
                relevant_docs=rel_docs,
                verification_results=result["verification_results"]["verification_results"],
                generated=result["generated_text"],
                ground_truth=gt,
                retrieved_texts=retrieved_texts,
                ground_truth_claims=ground_truth_claims,
                verifier=pipeline.verifier,  # Pass verifier for factual recall calculation
                abstained=abstained  # Pass abstention flag to exclude from hallucination_rate
            )
            
            all_metrics.append(metrics)
            results.append({
                "query": query,
                "ground_truth": gt,
                "generated": result["generated_text"],
                "retrieved_texts": result.get("retrieved_texts", []),
                "reranked_texts": result.get("reranked_texts", []),
                "context": result.get("context", ""),
                "generated_claims": generated_claims,
                "ground_truth_claims": ground_truth_claims,
                "formatted_generated_claims": formatted_generated_claims,
                "formatted_ground_truth_claims": formatted_ground_truth_claims,
                "verification_results": result.get("verification_results", {}),
                "verified": result.get("verification_results", {}).get("verified", False),
                "revision_iterations": revision_iterations,  # Only in exp9
                "revision_history": result.get("revision_history", []),  # Include revision history for transparency
                "abstained": result.get("abstained", False),  # Include abstention flag
                "metrics": metrics
            })
            
            # Log to W&B periodically
            if wandb_run and (idx + 1) % 100 == 0:
                avg_metrics = {
                    k: np.mean([m[k] for m in all_metrics])
                    for k in all_metrics[0].keys()
                }
                log_metrics(avg_metrics, step=idx + 1, prefix="complete_pipeline/")
        
        except Exception as e:
            import traceback
            print(f"Error processing query {idx}: {e}")
            print(f"Query: {query[:100]}...")
            traceback.print_exc()
            continue
    
    # Aggregate metrics
    print("\nAggregating metrics...")
    metric_names = [
        "recall@20", "coverage", "factual_precision", "factual_recall",
        "hallucination_rate", "verified_f1", "f1_score",
        "exact_match", "bleu4", "rouge_l", "abstention_rate",
        "fever_score"
    ]
    
    aggregated = {}
    if not all_metrics:
        print("⚠ Warning: No metrics collected. Check query processing errors above.")
        return {
            "results": results,
            "aggregated_metrics": {},
            "num_processed": len(results),
            "num_errors": len(queries) - len(results)
        }
    
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
    
    # Log final metrics to W&B
    if wandb_run:
        final_metrics = {k: v["mean"] for k, v in aggregated.items()}
        log_metrics(final_metrics, prefix="complete_pipeline/final/")
    
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
    json_path = os.path.join(output_dir, "exp9_complete_pipeline.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved results to {json_path}")
    
    # Save CSV
    csv_path = os.path.join(output_dir, "exp9_complete_pipeline.csv")
    aggregated = results["aggregated_metrics"]
    metadata = results.get("metadata", {})
    
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
        # Add metadata rows
        writer.writerow([])  # Empty row
        writer.writerow(["metadata", "value", "", "", ""])
        writer.writerow(["corpus_size", metadata.get("corpus_size", "N/A"), "", "", ""])
        writer.writerow(["top_k_retrieve", metadata.get("top_k_retrieve", "N/A"), "", "", ""])
        writer.writerow(["top_k_rerank", metadata.get("top_k_rerank", "N/A"), "", "", ""])
        writer.writerow(["corpus_to_k_ratio", f"{metadata.get('corpus_to_k_ratio', 0):.2f}", "", "", ""])
    print(f"✓ Saved metrics to {csv_path}")


def main():
    """Main experiment function."""
    # Parse arguments (extend parse_experiment_args with checkpoint)
    parser = argparse.ArgumentParser(description="Experiment 9: Complete HALO-RAG Pipeline")
    
    # Add standard experiment args
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    parser.add_argument("--split", type=str, default="validation", choices=["train", "validation", "test"], help="Dataset split to use")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples (for testing). Overrides config.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dry-run", action="store_true", help="Dry run with 20-50 samples for quick testing")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    
    # Add exp9-specific args
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to fine-tuned generator checkpoint (e.g., checkpoints/exp6_iter3/). If not provided, uses base (non-fine-tuned) model.")
    parser.add_argument("--max-revision-iterations", type=int, default=3,
                       help="Maximum revision iterations")
    parser.add_argument("--disable-revision", action="store_true",
                       help="Disable revision strategy (for comparison)")
    
    # Add top-k arguments
    parser.add_argument("--top-k-retrieve", type=int, default=None, help="Number of documents to retrieve (default: from config)")
    parser.add_argument("--top-k-rerank", type=int, default=None, help="Number of documents to rerank (default: from config)")
    
    args = parser.parse_args()
    
    # Respect --no-wandb flag
    if args.no_wandb:
        os.environ["WANDB_DISABLED"] = "true"
    
    # Load config
    config = load_config(args.config)
    
    # Get top-k values (from args or config)
    top_k_retrieve = args.top_k_retrieve if args.top_k_retrieve is not None else config.get("retrieval", {}).get("fusion", {}).get("top_k", 20)
    top_k_rerank = args.top_k_rerank if args.top_k_rerank is not None else config.get("retrieval", {}).get("reranker", {}).get("top_k", 5)
    
    print(f"Retrieval settings: top_k_retrieve={top_k_retrieve}, top_k_rerank={top_k_rerank}")
    
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
    if args.split != "validation":
        print(f"⚠ WARNING: Using '{args.split}' split. For fair comparison with Exp1 and Exp6, use 'validation' split.")
        print("   Exp6 uses 'train' split for fine-tuning, so evaluation should use 'validation' to avoid data leakage.")
    else:
        print("✓ Using 'validation' split (correct for evaluation, avoids data leakage with Exp6 training data)")
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
            run_name="exp9_complete_pipeline",
            config={
                "experiment": "exp9_complete_pipeline",
                "dataset": dataset_name,
                "split": args.split,
                "sample_limit": sample_limit,
                "seed": seed,
                "checkpoint": args.checkpoint,
                "enable_revision": not args.disable_revision,
                "max_revision_iterations": args.max_revision_iterations
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
    results = run_complete_pipeline_experiment(
        queries=queries,
        ground_truths=ground_truths,
        relevant_docs=relevant_docs,
        corpus=corpus,
        config=config,
        fine_tuned_checkpoint=args.checkpoint,
        enable_revision=not args.disable_revision,
        max_revision_iterations=args.max_revision_iterations,
        seed=seed,
        top_k_retrieve=top_k_retrieve,
        top_k_rerank=top_k_rerank,
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
        "checkpoint": args.checkpoint or "base_model",
        "enable_revision": not args.disable_revision,
        "max_revision_iterations": args.max_revision_iterations,
        "top_k_retrieve": top_k_retrieve,
        "top_k_rerank": top_k_rerank,
        "corpus_size": len(corpus),
        "corpus_to_k_ratio": len(corpus) / top_k_retrieve if top_k_retrieve > 0 else 0,
        "total_queries": len(queries),
        "processed_queries": results["processed_queries"]
    }
    
    # Save results
    save_results(results)
    
    # Print summary
    print("\n" + "=" * 70)
    print("Experiment 9: Complete HALO-RAG Pipeline Results")
    print("=" * 70)
    aggregated = results["aggregated_metrics"]
    for metric_name, stats in aggregated.items():
        print(f"{metric_name}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    print(f"\nVerified F1: {aggregated.get('verified_f1', {}).get('mean', 0):.4f}")
    print(f"Factual Precision: {aggregated.get('factual_precision', {}).get('mean', 0):.4f}")
    print(f"Hallucination Rate: {aggregated.get('hallucination_rate', {}).get('mean', 0):.4f}")
    
    # Revision statistics (exp9 only)
    revision_counts = [r.get("revision_iterations", 0) for r in results["individual_results"]]
    verified_counts = sum(1 for r in results["individual_results"] if r.get("verified", False))
    if revision_counts:
        print(f"\nRevision Statistics:")
        print(f"  Average revision iterations: {np.mean(revision_counts):.2f}")
        print(f"  Verified answers: {verified_counts}/{len(results['individual_results'])} ({100*verified_counts/len(results['individual_results']):.1f}%)")
    
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
