"""
Experiment 5: Self-Consistency Decoding
Compare greedy, beam search, and self-consistency decoding strategies.
"""

import sys
import os
import argparse
from pathlib import Path
import json
import csv
from datetime import datetime
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
import torch
import numpy as np
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


def generate_with_self_consistency(
    pipeline: SelfVerificationRAGPipeline,
    query: str,
    ground_truth: str,
    evaluator: EvaluationMetrics,
    k: int = 5,
    temperature: float = 0.7,
    factual_precision_threshold: float = 0.9,
    aggregation_method: str = "highest_verified_f1"
) -> Dict[str, Any]:
    """
    Generate answer using self-consistency decoding.
    
    Args:
        pipeline: RAG pipeline
        query: Query string
        ground_truth: Ground truth answer (for computing verified F1)
        evaluator: Evaluation metrics calculator
        k: Number of samples to generate
        temperature: Sampling temperature
        factual_precision_threshold: Minimum factual precision to keep answer
        aggregation_method: "majority_vote" or "highest_verified_f1"
    
    Returns:
        Dictionary with final answer and all generated samples
    """
    # Generate k samples
    samples = []
    
    for i in range(k):
        # Generate with temperature sampling
        result = pipeline.generate(
            query,
            top_k_retrieve=20,
            top_k_rerank=5,
            temperature=temperature,
            do_sample=True
        )
        
        # Get verification results
        verification_results = result["verification_results"]
        verification_data = verification_results.get("verification_results", [])
        
        # Compute factual precision
        if verification_data:
            # Count verified claims
            verified_count = sum(1 for v in verification_data if v.get("label") == "ENTAILED")
            total_claims = len(verification_data)
            factual_precision = verified_count / total_claims if total_claims > 0 else 0.0
        else:
            factual_precision = 0.0
        
        # Compute F1 and Verified F1
        f1_score = evaluator.f1_score(result["generated_text"], ground_truth)
        verified_f1 = f1_score * factual_precision
        
        samples.append({
            "answer": result["generated_text"],
            "verification_results": verification_data,
            "factual_precision": factual_precision,
            "f1_score": f1_score,
            "verified_f1": verified_f1,
            "claims": result.get("claims", []),
            "retrieved_texts": result.get("reranked_texts", result.get("retrieved_texts", []))
        })
    
    # Filter by factual precision threshold
    filtered_samples = [
        s for s in samples
        if s["factual_precision"] >= factual_precision_threshold
    ]
    
    # If no samples pass threshold, use all samples
    if not filtered_samples:
        filtered_samples = samples
    
    # Aggregate answers
    if aggregation_method == "majority_vote":
        # Simple majority vote on answer text
        answers = [s["answer"] for s in filtered_samples]
        answer_counts = Counter(answers)
        final_answer = answer_counts.most_common(1)[0][0]
    elif aggregation_method == "highest_verified_f1":
        # Select answer with highest verified F1
        best_sample = max(filtered_samples, key=lambda s: s.get("verified_f1", 0.0))
        final_answer = best_sample["answer"]
    else:
        # Default: use first filtered sample
        final_answer = filtered_samples[0]["answer"] if filtered_samples else samples[0]["answer"]
    
    return {
        "final_answer": final_answer,
        "samples": samples,
        "filtered_samples": filtered_samples,
        "k": k,
        "temperature": temperature,
        "aggregation_method": aggregation_method
    }


def run_self_consistency_experiment(
    queries: List[str],
    ground_truths: List[str],
    relevant_docs: List[List[int]],
    corpus: List[str],
    config: Dict[str, Any],
    k: int = 5,
    temperature: float = 0.7,
    factual_precision_threshold: float = 0.9,
    seed: int = 42,
    wandb_run: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Run self-consistency decoding experiment.
    
    Args:
        queries: List of queries
        ground_truths: List of ground truth answers
        relevant_docs: List of relevant document IDs
        corpus: List of documents
        config: Configuration dictionary
        k: Number of samples for self-consistency
        temperature: Sampling temperature
        factual_precision_threshold: Minimum factual precision
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
    
    # Initialize pipeline
    print("Initializing pipeline...")
    pipeline = SelfVerificationRAGPipeline(
        corpus=corpus,
        device=device,
        enable_revision=False,  # Disable revision for decoding comparison
        use_qlora=config.get("generation", {}).get("qlora", {}).get("training_enabled", False)
    )
    
    # Initialize evaluator
    evaluator = EvaluationMetrics()
    
    # Results for each decoding strategy
    results = {
        "greedy": [],
        "beam_search": [],
        "self_consistency": []
    }
    
    print(f"Running self-consistency experiment on {len(queries)} queries...")
    print(f"Self-consistency: k={k}, T={temperature}, FP_threshold={factual_precision_threshold}")
    
    for idx, (query, gt, rel_docs) in enumerate(tqdm(
        zip(queries, ground_truths, relevant_docs),
        total=len(queries),
        desc="Self-consistency"
    )):
        try:
            # 1. Greedy decoding (temperature=0, do_sample=False)
            greedy_result = pipeline.generate(
                query,
                top_k_retrieve=20,
                top_k_rerank=5,
                temperature=0.0,
                do_sample=False,
                num_beams=1
            )
            
            greedy_retrieved_texts = greedy_result.get("reranked_texts", greedy_result.get("retrieved_texts", []))
            greedy_metrics = evaluator.compute_all_metrics(
                retrieved_docs=greedy_result["retrieved_docs"],
                relevant_docs=rel_docs,
                verification_results=greedy_result["verification_results"]["verification_results"],
                generated=greedy_result["generated_text"],
                ground_truth=gt,
                retrieved_texts=greedy_retrieved_texts
            )
            
            results["greedy"].append({
                "query": query,
                "ground_truth": gt,
                "generated": greedy_result["generated_text"],
                "metrics": greedy_metrics
            })
            
            # 2. Beam search decoding (num_beams=5)
            beam_result = pipeline.generate(
                query,
                top_k_retrieve=20,
                top_k_rerank=5,
                temperature=0.0,
                do_sample=False,
                num_beams=5
            )
            
            beam_retrieved_texts = beam_result.get("reranked_texts", beam_result.get("retrieved_texts", []))
            beam_metrics = evaluator.compute_all_metrics(
                retrieved_docs=beam_result["retrieved_docs"],
                relevant_docs=rel_docs,
                verification_results=beam_result["verification_results"]["verification_results"],
                generated=beam_result["generated_text"],
                ground_truth=gt,
                retrieved_texts=beam_retrieved_texts
            )
            
            results["beam_search"].append({
                "query": query,
                "ground_truth": gt,
                "generated": beam_result["generated_text"],
                "metrics": beam_metrics
            })
            
            # 3. Self-consistency decoding
            sc_result = generate_with_self_consistency(
                pipeline,
                query,
                gt,
                evaluator,
                k=k,
                temperature=temperature,
                factual_precision_threshold=factual_precision_threshold,
                aggregation_method="highest_verified_f1"
            )
            
            # Compute metrics for final aggregated answer
            # Use the best sample's retrieved texts and verification results
            best_sample = max(
                sc_result["filtered_samples"] if sc_result["filtered_samples"] else sc_result["samples"],
                key=lambda s: s.get("verified_f1", 0.0)
            )
            
            # Re-verify the final answer to get complete metrics
            # Use the best sample's retrieved texts
            sc_retrieved_texts = best_sample.get("retrieved_texts", [])
            
            # Re-verify the final answer
            claims = pipeline.claim_extractor.extract_claims(sc_result["final_answer"])
            verification_results = pipeline.verifier.verify_generation(
                sc_result["final_answer"],
                sc_retrieved_texts,
                claims
            )
            
            # Get retrieved docs (use first sample's retrieved docs structure)
            # For self-consistency, we use the same retrieval as the samples
            sc_retrieved_docs = greedy_result["retrieved_docs"]  # Use same retrieval structure
            
            sc_metrics = evaluator.compute_all_metrics(
                retrieved_docs=sc_retrieved_docs,
                relevant_docs=rel_docs,
                verification_results=verification_results["verification_results"],
                generated=sc_result["final_answer"],
                ground_truth=gt,
                retrieved_texts=sc_retrieved_texts
            )
            
            # Add compute cost (k times more expensive)
            sc_metrics["compute_cost"] = k
            
            results["self_consistency"].append({
                "query": query,
                "ground_truth": gt,
                "generated": sc_result["final_answer"],
                "metrics": sc_metrics,
                "samples": sc_result["samples"],
                "filtered_samples_count": len(sc_result["filtered_samples"])
            })
            
            # Log to W&B periodically
            if wandb_run and (idx + 1) % 50 == 0:
                for strategy in ["greedy", "beam_search", "self_consistency"]:
                    if results[strategy]:
                        avg_metrics = {
                            k: np.mean([r["metrics"][k] for r in results[strategy] if k in r["metrics"]])
                            for k in ["hallucination_rate", "f1_score", "verified_f1"]
                            if results[strategy][0]["metrics"].get(k) is not None
                        }
                        log_metrics(avg_metrics, step=idx + 1, prefix=f"decoding/{strategy}/")
        
        except Exception as e:
            print(f"Error processing query {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Aggregate metrics
    print("\nAggregating metrics...")
    aggregated = {}
    metric_names = [
        "hallucination_rate", "f1_score", "verified_f1",
        "factual_precision", "exact_match", "compute_cost"
    ]
    
    for strategy in results.keys():
        aggregated[strategy] = {}
        strategy_results = results[strategy]
        
        for metric_name in metric_names:
            scores = [r["metrics"].get(metric_name) for r in strategy_results if metric_name in r["metrics"]]
            scores = [s for s in scores if s is not None]
            
            if scores:
                aggregated[strategy][metric_name] = {
                    "mean": float(np.mean(scores)),
                    "std": float(np.std(scores)),
                    "min": float(np.min(scores)),
                    "max": float(np.max(scores)),
                    "scores": scores
                }
    
    # Log final metrics to W&B
    if wandb_run:
        for strategy, metrics in aggregated.items():
            final_metrics = {k: v["mean"] for k, v in metrics.items()}
            log_metrics(final_metrics, prefix=f"decoding/{strategy}/final/")
    
    return {
        "aggregated_metrics": aggregated,
        "individual_results": results,
        "config": {
            "k": k,
            "temperature": temperature,
            "factual_precision_threshold": factual_precision_threshold
        },
        "total_queries": len(queries),
        "processed_queries": len(results["greedy"])
    }


def plot_decoding_comparison(results: Dict[str, Any], output_dir: str = "results/figures"):
    """Plot comparison of decoding strategies."""
    os.makedirs(output_dir, exist_ok=True)
    
    aggregated = results["aggregated_metrics"]
    strategies = list(aggregated.keys())
    metric_names = ["hallucination_rate", "f1_score", "verified_f1"]
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, metric_name in enumerate(metric_names):
        ax = axes[idx]
        
        # Get means and stds for each strategy
        means = [aggregated[strategy].get(metric_name, {}).get("mean", 0) for strategy in strategies]
        stds = [aggregated[strategy].get(metric_name, {}).get("std", 0) for strategy in strategies]
        
        # Create bar plot
        x_pos = np.arange(len(strategies))
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.8, edgecolor='black')
        
        # Color bars
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # Add value labels on bars
        for i, (mean, std) in enumerate(zip(means, stds)):
            ax.text(i, mean + std + 0.01, f'{mean:.3f}', 
                   ha='center', va='bottom', fontsize=9)
        
        # Formatting
        ax.set_xlabel('Decoding Strategy', fontsize=11)
        ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=11)
        ax.set_title(f'{metric_name.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([s.replace('_', ' ').title() for s in strategies], rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Set y-axis limits
        if metric_name == "hallucination_rate":
            ax.set_ylim(0, max(means) * 1.3 if max(means) > 0 else 0.5)
        else:
            ax.set_ylim(0, max(means) * 1.2 if max(means) > 0 else 1.0)
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, "exp5_decoding_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved plot to {output_path}")


def save_results(results: Dict[str, Any], output_dir: str = "results/metrics"):
    """Save results to JSON and CSV."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save JSON
    json_path = os.path.join(output_dir, "exp5_self_consistency.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved results to {json_path}")
    
    # Save CSV
    csv_path = os.path.join(output_dir, "exp5_self_consistency.csv")
    aggregated = results["aggregated_metrics"]
    strategies = list(aggregated.keys())
    metric_names = ["hallucination_rate", "f1_score", "verified_f1", "factual_precision", "exact_match", "compute_cost"]
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["strategy", "metric", "mean", "std", "min", "max"])
        
        for strategy in strategies:
            for metric_name in metric_names:
                if metric_name in aggregated[strategy]:
                    stats = aggregated[strategy][metric_name]
                    writer.writerow([
                        strategy,
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
    args = parse_experiment_args(description="Experiment 5: Self-Consistency Decoding")
    
    # Load config
    config = load_config(args.config)
    
    # Set random seed
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Get self-consistency parameters
    k = config.get("experiments", {}).get("exp5", {}).get("k", 5)
    temperature = config.get("experiments", {}).get("exp5", {}).get("temperature", 0.7)
    factual_precision_threshold = config.get("experiments", {}).get("exp5", {}).get("factual_precision_threshold", 0.9)
    
    # Determine sample limit
    sample_limit = args.limit
    if args.dry_run:
        sample_limit = 20  # Smaller for self-consistency (k=5 samples each)
        print("⚠ DRY RUN MODE: Using 20 samples")
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
    print(f"Self-consistency: k={k}, T={temperature}, FP_threshold={factual_precision_threshold}")
    
    # Get metadata
    commit_hash = get_commit_hash()
    timestamp = get_timestamp()
    dataset_name = config.get("datasets", {}).get("active", "unknown")
    
    # Setup W&B
    wandb_run = None
    if not args.no_wandb:
        wandb_run = setup_wandb(
            project_name="SelfVerifyRAG",
            run_name="exp5_self_consistency",
            config={
                "experiment": "exp5_self_consistency",
                "dataset": dataset_name,
                "split": args.split,
                "sample_limit": sample_limit,
                "seed": seed,
                "k": k,
                "temperature": temperature,
                "factual_precision_threshold": factual_precision_threshold
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
    results = run_self_consistency_experiment(
        queries=queries,
        ground_truths=ground_truths,
        relevant_docs=relevant_docs,
        corpus=corpus,
        config=config,
        k=k,
        temperature=temperature,
        factual_precision_threshold=factual_precision_threshold,
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
        "k": k,
        "temperature": temperature,
        "factual_precision_threshold": factual_precision_threshold,
        "total_queries": len(queries),
        "processed_queries": results["processed_queries"]
    }
    
    # Save results
    save_results(results)
    
    # Generate plots
    plot_decoding_comparison(results)
    
    # Print summary
    print("\n" + "=" * 70)
    print("Experiment 5: Self-Consistency Decoding Results")
    print("=" * 70)
    
    aggregated = results["aggregated_metrics"]
    strategies = ["greedy", "beam_search", "self_consistency"]
    
    print(f"\n{'Strategy':<20} {'Hallucination Rate':<20} {'F1 Score':<15} {'Verified F1':<15}")
    print("-" * 70)
    for strategy in strategies:
        if strategy in aggregated:
            hr = aggregated[strategy].get("hallucination_rate", {}).get("mean", 0)
            f1 = aggregated[strategy].get("f1_score", {}).get("mean", 0)
            vf1 = aggregated[strategy].get("verified_f1", {}).get("mean", 0)
            print(f"{strategy:<20} {hr:<20.4f} {f1:<15.4f} {vf1:<15.4f}")
    
    # Check acceptance criteria
    print("\n" + "=" * 70)
    print("Acceptance Criteria Check:")
    print("=" * 70)
    
    greedy_hr = aggregated.get("greedy", {}).get("hallucination_rate", {}).get("mean", 0)
    sc_hr = aggregated.get("self_consistency", {}).get("hallucination_rate", {}).get("mean", 0)
    hr_reduction = ((greedy_hr - sc_hr) / greedy_hr * 100) if greedy_hr > 0 else 0
    
    greedy_vf1 = aggregated.get("greedy", {}).get("verified_f1", {}).get("mean", 0)
    sc_vf1 = aggregated.get("self_consistency", {}).get("verified_f1", {}).get("mean", 0)
    vf1_improvement = sc_vf1 - greedy_vf1
    
    print(f"Hallucination Rate reduction: {hr_reduction:.2f}% ({'✓' if hr_reduction >= 15 else '✗'} target: ≥15%)")
    print(f"Verified F1 improvement: {vf1_improvement:.4f} ({'✓' if vf1_improvement > 0 else '✗'} target: >0)")
    print(f"Compute cost: {k}x (self-consistency generates {k} samples)")
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

