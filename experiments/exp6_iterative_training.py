"""
Experiment 6: Iterative Fine-Tuning with Verified Data Collection
Collect verified data (FP ≥ 0.85) and fine-tune FLAN-T5 iteratively.
"""

import sys
import os
import argparse
from pathlib import Path
import json
import csv
from datetime import datetime
from typing import List, Dict, Any, Optional

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
from src.data.verified_collector import (
    collect_verified_data,
    save_verified_data,
    load_verified_data,
    compute_diversity_stats
)
from src.pipeline import SelfVerificationRAGPipeline
from src.generator import FLANT5Generator, QLoRATrainer
from src.evaluation import EvaluationMetrics
from src.utils import setup_wandb, log_metrics, log_metadata, get_commit_hash, get_timestamp
from src.utils.cli import parse_experiment_args
from datasets import Dataset


def load_config(config_path: str = "config/config.yaml"):
    """Load configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def collect_verified_training_data(
    pipeline: SelfVerificationRAGPipeline,
    queries: List[str],
    ground_truths: List[str],
    relevant_docs: List[List[int]],
    corpus: List[str],
    iteration: int,
    config: Dict[str, Any],
    output_dir: str = "data/verified"
) -> List[Dict[str, Any]]:
    """
    Collect verified training data for iteration.
    
    Args:
        pipeline: RAG pipeline
        queries: List of queries
        ground_truths: List of ground truth answers
        relevant_docs: List of relevant document IDs
        corpus: List of documents
        iteration: Iteration number
        config: Configuration dictionary
        output_dir: Output directory for verified data
    
    Returns:
        List of verified examples
    """
    factual_precision_threshold = config.get("verification", {}).get("accept_min", 0.85)
    top_k_passages = config.get("experiments", {}).get("exp6", {}).get("top_k_passages", 5)
    
    # Collect verified data
    verified_examples = collect_verified_data(
        pipeline=pipeline,
        queries=queries,
        ground_truths=ground_truths,
        relevant_docs=relevant_docs,
        corpus=corpus,
        factual_precision_threshold=factual_precision_threshold,
        top_k_passages=top_k_passages
    )
    
    # Save to JSONL
    output_path = os.path.join(output_dir, f"train_iter{iteration}.jsonl")
    save_verified_data(verified_examples, output_path, iteration)
    
    # Compute diversity stats
    diversity_stats = compute_diversity_stats(verified_examples)
    logger.info(f"Iteration {iteration} diversity stats: {diversity_stats}")
    
    return verified_examples


def fine_tune_iteration(
    verified_data_path: str,
    iteration: int,
    config: Dict[str, Any],
    previous_checkpoint: Optional[str] = None,
    checkpoint_dir: str = "checkpoints/exp6"
) -> str:
    """
    Fine-tune FLAN-T5 with QLoRA on verified data.
    
    Args:
        verified_data_path: Path to verified data JSONL file
        iteration: Iteration number
        config: Configuration dictionary
        previous_checkpoint: Path to previous iteration's checkpoint (for iterative training)
        checkpoint_dir: Checkpoint directory
    
    Returns:
        Path to saved checkpoint
    """
    logger.info(f"Fine-tuning iteration {iteration}...")
    
    # Load verified data
    verified_examples = load_verified_data(verified_data_path)
    
    if not verified_examples:
        raise ValueError(f"No verified data found in {verified_data_path}")
    
    # Prepare training data
    queries = [ex["question"] for ex in verified_examples]
    contexts = [ex["context"] for ex in verified_examples]
    answers = [ex["verified_answer"] for ex in verified_examples]
    
    # Initialize generator (for tokenizer and model setup)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load previous checkpoint if available (for iterative training)
    generator = FLANT5Generator(
        model_name=config.get("generation", {}).get("model_name", "google/flan-t5-large"),
        device=device,
        lora_checkpoint=previous_checkpoint,  # Load previous checkpoint if available
        use_qlora=True,
        r=config.get("generation", {}).get("qlora", {}).get("r", 16),
        lora_alpha=config.get("generation", {}).get("qlora", {}).get("lora_alpha", 32),
        lora_dropout=config.get("generation", {}).get("qlora", {}).get("lora_dropout", 0.1)
    )
    
    # If model already has PEFT adapters (from previous checkpoint), use it directly
    # Otherwise, initialize trainer to add adapters
    from peft import PeftModel
    if isinstance(generator.model, PeftModel):
        # Model already has PEFT adapters, use directly
        model = generator.model
        # Set to training mode
        model.train()
    else:
        # Initialize trainer to add adapters
        trainer = QLoRATrainer(
            model=generator.model,
            tokenizer=generator.tokenizer,
            r=config.get("generation", {}).get("qlora", {}).get("r", 16),
            lora_alpha=config.get("generation", {}).get("qlora", {}).get("lora_alpha", 32),
            lora_dropout=config.get("generation", {}).get("qlora", {}).get("lora_dropout", 0.1),
            target_modules=config.get("generation", {}).get("qlora", {}).get("target_modules", ["q", "v", "k", "o"])
        )
        model = trainer.model
        model.train()
    
    # Prepare dataset using tokenizer
    from datasets import Dataset
    from transformers import DataCollatorForSeq2Seq
    
    # Format inputs: "Question: {query} Context: {context} Answer:"
    inputs = [
        f"Question: {q} Context: {c} Answer:"
        for q, c in zip(queries, contexts)
    ]
    
    # Tokenize inputs
    model_inputs = generator.tokenizer(
        inputs,
        max_length=config.get("generation", {}).get("max_length", 512),
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    
    # Tokenize targets (answers)
    with generator.tokenizer.as_target_tokenizer():
        labels = generator.tokenizer(
            answers,
            max_length=config.get("generation", {}).get("max_length", 512),
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
    
    model_inputs["labels"] = labels["input_ids"]
    
    # Convert to dataset
    train_dataset = Dataset.from_dict({
        "input_ids": model_inputs["input_ids"],
        "attention_mask": model_inputs["attention_mask"],
        "labels": model_inputs["labels"]
    })
    
    # Training config
    training_config = config.get("training", {})
    
    # Save checkpoint path
    checkpoint_path = os.path.join(checkpoint_dir, f"iter{iteration}")
    os.makedirs(checkpoint_path, exist_ok=True)
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=generator.tokenizer,
        model=model,
        padding=True
    )
    
    # Training arguments
    from transformers import TrainingArguments, Trainer
    
    # Ensure learning_rate is a float (YAML may parse scientific notation as string)
    learning_rate = training_config.get("learning_rate", 2e-4)
    if isinstance(learning_rate, str):
        learning_rate = float(learning_rate)
    
    training_args = TrainingArguments(
        output_dir=checkpoint_path,
        num_train_epochs=int(training_config.get("num_epochs", 3)),
        per_device_train_batch_size=int(training_config.get("batch_size", 8)),
        gradient_accumulation_steps=int(training_config.get("gradient_accumulation_steps", 4)),
        learning_rate=learning_rate,
        warmup_steps=int(training_config.get("warmup_steps", 100)),
        logging_steps=int(training_config.get("logging_steps", 100)),
        save_strategy="epoch",
        save_total_limit=3,
        fp16=True,
        report_to="wandb" if os.getenv("WANDB_API_KEY") else None,
        run_name=f"exp6_iter{iteration}"
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=generator.tokenizer
    )
    
    # Train
    trainer.train()
    
    # Save final model
    model.save_pretrained(checkpoint_path)
    generator.tokenizer.save_pretrained(checkpoint_path)
    
    logger.info(f"Fine-tuning complete. Checkpoint saved to {checkpoint_path}")
    
    return checkpoint_path


def evaluate_iteration(
    pipeline: SelfVerificationRAGPipeline,
    queries: List[str],
    ground_truths: List[str],
    relevant_docs: List[List[int]],
    corpus: List[str],
    evaluator: EvaluationMetrics
) -> Dict[str, float]:
    """
    Evaluate pipeline on validation split.
    
    Args:
        pipeline: RAG pipeline
        queries: List of queries
        ground_truths: List of ground truth answers
        relevant_docs: List of relevant document IDs
        corpus: List of documents
        evaluator: Evaluation metrics calculator
    
    Returns:
        Dictionary with metrics
    """
    all_metrics = []
    
    logger.info(f"Evaluating on {len(queries)} queries...")
    
    for query, gt, rel_docs in tqdm(
        zip(queries, ground_truths, relevant_docs),
        total=len(queries),
        desc="Evaluating"
    ):
        try:
            result = pipeline.generate(query, top_k_retrieve=20, top_k_rerank=5)
            
            retrieved_texts = result.get("reranked_texts", result.get("retrieved_texts", []))
            
            metrics = evaluator.compute_all_metrics(
                retrieved_docs=result["retrieved_docs"],
                relevant_docs=rel_docs,
                verification_results=result["verification_results"]["verification_results"],
                generated=result["generated_text"],
                ground_truth=gt,
                retrieved_texts=retrieved_texts
            )
            
            all_metrics.append(metrics)
        
        except Exception as e:
            logger.warning(f"Error evaluating query: {e}")
            continue
    
    # Aggregate metrics
    aggregated = {}
    metric_names = [
        "hallucination_rate", "factual_precision", "f1_score",
        "exact_match", "verified_f1", "abstention_rate"
    ]
    
    for metric_name in metric_names:
        scores = [m.get(metric_name) for m in all_metrics if metric_name in m]
        scores = [s for s in scores if s is not None]
        
        if scores:
            aggregated[metric_name] = float(np.mean(scores))
        else:
            aggregated[metric_name] = 0.0
    
    return aggregated


def run_iterative_training(
    train_queries: List[str],
    train_ground_truths: List[str],
    train_relevant_docs: List[List[int]],
    val_queries: List[str],
    val_ground_truths: List[str],
    val_relevant_docs: List[List[int]],
    corpus: List[str],
    config: Dict[str, Any],
    iterations: int = 3,
    seed: int = 42,
    wandb_run: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Run iterative fine-tuning experiment.
    
    Args:
        train_queries: Training queries
        train_ground_truths: Training ground truths
        train_relevant_docs: Training relevant docs
        val_queries: Validation queries
        val_ground_truths: Validation ground truths
        val_relevant_docs: Validation relevant docs
        corpus: Corpus of documents
        config: Configuration dictionary
        iterations: Number of iterations
        seed: Random seed
        wandb_run: W&B run object (optional)
    
    Returns:
        Dictionary with results
    """
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize evaluator
    evaluator = EvaluationMetrics()
    
    # Results for each iteration
    iteration_results = {}
    
    # Iteration 0: Baseline (no fine-tuning)
    logger.info("=" * 70)
    logger.info("Iteration 0: Baseline (no fine-tuning)")
    logger.info("=" * 70)
    
    pipeline = SelfVerificationRAGPipeline(
        corpus=corpus,
        device=device,
        enable_revision=False,
        use_qlora=config.get("generation", {}).get("qlora", {}).get("training_enabled", False)
    )
    
    # Evaluate baseline
    baseline_metrics = evaluate_iteration(
        pipeline=pipeline,
        queries=val_queries,
        ground_truths=val_ground_truths,
        relevant_docs=val_relevant_docs,
        corpus=corpus,
        evaluator=evaluator
    )
    
    iteration_results[0] = {
        "metrics": baseline_metrics,
        "checkpoint": None,
        "verified_data_count": 0
    }
    
    logger.info(f"Baseline metrics: {baseline_metrics}")
    
    # Log baseline to W&B
    if wandb_run:
        baseline_metrics_with_iter = {**baseline_metrics, "iteration": 0}
        log_metrics(baseline_metrics_with_iter, prefix="iteration_0/")
    
    # Iterations 1 to N
    checkpoint_path = None
    
    for iteration in range(1, iterations + 1):
        logger.info("=" * 70)
        logger.info(f"Iteration {iteration}")
        logger.info("=" * 70)
        
        # Reinitialize pipeline with previous checkpoint if available
        if checkpoint_path and os.path.exists(checkpoint_path):
            pipeline = SelfVerificationRAGPipeline(
                corpus=corpus,
                device=device,
                enable_revision=False,
                use_qlora=True,
                generator_lora_checkpoint=checkpoint_path
            )
        else:
            # For iteration 1, use baseline (no checkpoint)
            # For later iterations, use previous checkpoint
            if iteration == 1:
                pipeline = SelfVerificationRAGPipeline(
                    corpus=corpus,
                    device=device,
                    enable_revision=False,
                    use_qlora=config.get("generation", {}).get("qlora", {}).get("training_enabled", False)
                )
            else:
                # Use previous iteration's checkpoint
                prev_checkpoint = os.path.join("checkpoints/exp6", f"iter{iteration-1}")
                if os.path.exists(prev_checkpoint):
                    pipeline = SelfVerificationRAGPipeline(
                        corpus=corpus,
                        device=device,
                        enable_revision=False,
                        use_qlora=True,
                        generator_lora_checkpoint=prev_checkpoint
                    )
                else:
                    pipeline = SelfVerificationRAGPipeline(
                        corpus=corpus,
                        device=device,
                        enable_revision=False,
                        use_qlora=config.get("generation", {}).get("qlora", {}).get("training_enabled", False)
                    )
        
        # Collect verified training data
        verified_examples = collect_verified_training_data(
            pipeline=pipeline,
            queries=train_queries,
            ground_truths=train_ground_truths,
            relevant_docs=train_relevant_docs,
            corpus=corpus,
            iteration=iteration,
            config=config
        )
        
        if not verified_examples:
            logger.warning(f"No verified data collected for iteration {iteration}. Skipping fine-tuning.")
            continue
        
        # Fine-tune
        verified_data_path = f"data/verified/train_iter{iteration}.jsonl"
        
        # Get previous checkpoint for iterative training
        previous_checkpoint = None
        if iteration > 1:
            prev_checkpoint_path = os.path.join("checkpoints/exp6", f"iter{iteration-1}")
            if os.path.exists(prev_checkpoint_path):
                previous_checkpoint = prev_checkpoint_path
        
        checkpoint_path = fine_tune_iteration(
            verified_data_path=verified_data_path,
            iteration=iteration,
            config=config,
            previous_checkpoint=previous_checkpoint
        )
        
        # Reinitialize pipeline with new checkpoint
        pipeline = SelfVerificationRAGPipeline(
            corpus=corpus,
            device=device,
            enable_revision=False,
            use_qlora=True,
            generator_lora_checkpoint=checkpoint_path
        )
        
        # Evaluate on validation split
        metrics = evaluate_iteration(
            pipeline=pipeline,
            queries=val_queries,
            ground_truths=val_ground_truths,
            relevant_docs=val_relevant_docs,
            corpus=corpus,
            evaluator=evaluator
        )
        
        iteration_results[iteration] = {
            "metrics": metrics,
            "checkpoint": checkpoint_path,
            "verified_data_count": len(verified_examples)
        }
        
        logger.info(f"Iteration {iteration} metrics: {metrics}")
        
        # Log to W&B
        if wandb_run:
            metrics_with_iter = {**metrics, "iteration": iteration}
            log_metrics(metrics_with_iter, prefix=f"iteration_{iteration}/")
    
    return {
        "iteration_results": iteration_results,
        "total_iterations": iterations
    }


def plot_iteration_curves(results: Dict[str, Any], output_dir: str = "results/figures"):
    """Plot iteration curves showing Hallucination Rate ↓ and Verified F1 ↑."""
    os.makedirs(output_dir, exist_ok=True)
    
    iteration_results = results["iteration_results"]
    iterations = sorted([k for k in iteration_results.keys()])
    
    # Extract metrics
    hallucination_rates = [
        iteration_results[i]["metrics"].get("hallucination_rate", 0)
        for i in iterations
    ]
    verified_f1s = [
        iteration_results[i]["metrics"].get("verified_f1", 0)
        for i in iterations
    ]
    factual_precisions = [
        iteration_results[i]["metrics"].get("factual_precision", 0)
        for i in iterations
    ]
    f1_scores = [
        iteration_results[i]["metrics"].get("f1_score", 0)
        for i in iterations
    ]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Hallucination Rate (should decrease)
    axes[0, 0].plot(iterations, hallucination_rates, marker='o', linewidth=2, markersize=8, color='red', label='Hallucination Rate')
    axes[0, 0].axhline(y=0.10, color='r', linestyle='--', alpha=0.5, label='Target (≤0.10)')
    axes[0, 0].set_xlabel('Iteration', fontsize=12)
    axes[0, 0].set_ylabel('Hallucination Rate', fontsize=12)
    axes[0, 0].set_title('Hallucination Rate vs Iteration', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Verified F1 (should increase)
    axes[0, 1].plot(iterations, verified_f1s, marker='o', linewidth=2, markersize=8, color='green', label='Verified F1')
    axes[0, 1].set_xlabel('Iteration', fontsize=12)
    axes[0, 1].set_ylabel('Verified F1', fontsize=12)
    axes[0, 1].set_title('Verified F1 vs Iteration', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Factual Precision
    axes[1, 0].plot(iterations, factual_precisions, marker='o', linewidth=2, markersize=8, color='blue', label='Factual Precision')
    axes[1, 0].axhline(y=0.85, color='r', linestyle='--', alpha=0.5, label='Threshold (0.85)')
    axes[1, 0].set_xlabel('Iteration', fontsize=12)
    axes[1, 0].set_ylabel('Factual Precision', fontsize=12)
    axes[1, 0].set_title('Factual Precision vs Iteration', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # F1 Score (should be stable or slightly increase)
    axes[1, 1].plot(iterations, f1_scores, marker='o', linewidth=2, markersize=8, color='purple', label='F1 Score')
    axes[1, 1].set_xlabel('Iteration', fontsize=12)
    axes[1, 1].set_ylabel('F1 Score', fontsize=12)
    axes[1, 1].set_title('F1 Score vs Iteration', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
            
    # Save figure
    output_path = os.path.join(output_dir, "exp6_iteration_curves.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved plot to {output_path}")


def save_results(results: Dict[str, Any], output_dir: str = "results/metrics"):
    """Save results to CSV."""
    os.makedirs(output_dir, exist_ok=True)
    
    iteration_results = results["iteration_results"]
    iterations = sorted([k for k in iteration_results.keys()])
    
    # Save CSV
    csv_path = os.path.join(output_dir, "exp6_iterative_training.csv")
    
    metric_names = [
        "hallucination_rate", "factual_precision", "f1_score",
        "exact_match", "verified_f1", "abstention_rate"
    ]
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["iteration"] + metric_names + ["verified_data_count", "checkpoint"])
        
        for iteration in iterations:
            result = iteration_results[iteration]
            metrics = result["metrics"]
            row = [iteration]
            
            for metric_name in metric_names:
                row.append(f"{metrics.get(metric_name, 0):.4f}")
            
            row.append(result.get("verified_data_count", 0))
            row.append(result.get("checkpoint", ""))
            
            writer.writerow(row)
    
    print(f"✓ Saved results to {csv_path}")
    
    # Save JSON
    json_path = os.path.join(output_dir, "exp6_iterative_training.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved full results to {json_path}")


def main():
    """Main experiment function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Experiment 6: Iterative Fine-Tuning")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Config file path")
    parser.add_argument("--iterations", type=int, default=None, help="Number of iterations")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dry-run", action="store_true", help="Dry run with ≤100 examples")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Set random seed
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Get iterations from config or args
    iterations = args.iterations
    if iterations is None:
        iterations = config.get("experiments", {}).get("exp6", {}).get("iterations", 3)
    
    # Determine sample limit
    sample_limit = args.limit
    if args.dry_run:
        sample_limit = 100
        print("⚠ DRY RUN MODE: Using 100 samples")
    elif sample_limit is None:
        sample_limit = config.get("experiments", {}).get("exp6", {}).get("train_limit")
    
    # Load datasets
    print("Loading datasets...")
    print("⚠ IMPORTANT: Exp6 uses 'train' split for fine-tuning and 'validation' split for evaluation")
    print("   This ensures no data leakage between training and evaluation.")
    train_examples = load_dataset_from_config(config, split="train")
    val_examples = load_dataset_from_config(config, split="validation")
    
    # Verify splits are different (sanity check)
    if len(train_examples) > 0 and len(val_examples) > 0:
        train_ids = {ex["id"] for ex in train_examples[:100]}  # Check first 100
        val_ids = {ex["id"] for ex in val_examples[:100]}
        overlap = train_ids.intersection(val_ids)
        if overlap:
            print(f"⚠ WARNING: Found {len(overlap)} overlapping IDs between train and validation splits!")
            print(f"   This indicates potential data leakage. Please check dataset loading.")
        else:
            print("✓ Verified: Train and validation splits have no overlapping examples (checked first 100)")
    
    # Apply sample limit if specified
    if sample_limit:
        train_examples = train_examples[:sample_limit]
        val_examples = val_examples[:min(sample_limit, len(val_examples))]
        print(f"Limited to {len(train_examples)} train and {len(val_examples)} val examples")
            
    # Prepare for experiments
    train_queries, train_ground_truths, train_relevant_docs, corpus = prepare_for_experiments(train_examples)
    val_queries, val_ground_truths, val_relevant_docs, _ = prepare_for_experiments(val_examples)
    
    print(f"Train: {len(train_queries)} queries")
    print(f"Validation: {len(val_queries)} queries")
    print(f"Corpus size: {len(corpus)}")
    print(f"Iterations: {iterations}")
    
    # Get metadata
    commit_hash = get_commit_hash()
    timestamp = get_timestamp()
    dataset_name = config.get("datasets", {}).get("active", "unknown")
    
    # Setup W&B
    wandb_run = None
    if not args.no_wandb:
        wandb_run = setup_wandb(
            project_name="SelfVerifyRAG",
            run_name="exp6_iterative_training",
            config={
                "experiment": "exp6_iterative_training",
                "dataset": dataset_name,
                "iterations": iterations,
                "sample_limit": sample_limit,
                "seed": seed
            },
            enabled=True
        )
        
        # Log metadata
        log_metadata(
            dataset_name=dataset_name,
            split="train+validation",
            sample_limit=sample_limit,
            commit_hash=commit_hash,
            timestamp=timestamp
            )
    
    # Run iterative training
    results = run_iterative_training(
        train_queries=train_queries,
        train_ground_truths=train_ground_truths,
        train_relevant_docs=train_relevant_docs,
        val_queries=val_queries,
        val_ground_truths=val_ground_truths,
        val_relevant_docs=val_relevant_docs,
        corpus=corpus,
        config=config,
        iterations=iterations,
        seed=seed,
        wandb_run=wandb_run
    )
    
    # Add metadata to results
    results["metadata"] = {
        "dataset": dataset_name,
        "iterations": iterations,
        "sample_limit": sample_limit,
        "seed": seed,
        "commit_hash": commit_hash,
        "timestamp": timestamp
    }
    
    # Save results
    save_results(results)
    
    # Generate plots
    plot_iteration_curves(results)
    
    # Print summary
    print("\n" + "=" * 70)
    print("Experiment 6: Iterative Fine-Training Results")
    print("=" * 70)
    
    iteration_results = results["iteration_results"]
    iterations = sorted([k for k in iteration_results.keys()])
    
    print(f"\n{'Iteration':<12} {'Hallucination Rate':<20} {'Verified F1':<15} {'F1 Score':<15} {'Factual Prec':<15}")
    print("-" * 80)
    for iteration in iterations:
        result = iteration_results[iteration]
        metrics = result["metrics"]
        hr = metrics.get("hallucination_rate", 0)
        vf1 = metrics.get("verified_f1", 0)
        f1 = metrics.get("f1_score", 0)
        fp = metrics.get("factual_precision", 0)
        print(f"{iteration:<12} {hr:<20.4f} {vf1:<15.4f} {f1:<15.4f} {fp:<15.4f}")
    
    # Check acceptance criteria
    print("\n" + "=" * 70)
    print("Acceptance Criteria Check:")
    print("=" * 70)
    
    baseline_hr = iteration_results[0]["metrics"].get("hallucination_rate", 0)
    final_hr = iteration_results[iterations[-1]]["metrics"].get("hallucination_rate", 0)
    hr_reduction = ((baseline_hr - final_hr) / baseline_hr * 100) if baseline_hr > 0 else 0
    
    baseline_vf1 = iteration_results[0]["metrics"].get("verified_f1", 0)
    final_vf1 = iteration_results[iterations[-1]]["metrics"].get("verified_f1", 0)
    vf1_improvement = final_vf1 - baseline_vf1
    
    print(f"Hallucination Rate reduction: {hr_reduction:.2f}% ({'✓' if hr_reduction >= 10 else '✗'} target: ≥10% per iteration)")
    print(f"Verified F1 improvement: {vf1_improvement:.4f} ({'✓' if vf1_improvement > 0 else '✗'} target: >0)")
    print(f"Final Hallucination Rate: {final_hr:.4f} ({'✓' if final_hr <= 0.10 else '✗'} target: ≤0.10)")
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
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    results = main()
