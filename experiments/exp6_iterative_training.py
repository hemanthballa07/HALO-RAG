"""
Experiment 6: Iterative Training
Self-improvement through fine-tuning loops
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch
import numpy as np
from typing import List, Dict, Any
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

from src.pipeline import SelfVerificationRAGPipeline
from src.generator import QLoRATrainer, FLANT5Generator
from src.evaluation import EvaluationMetrics


def load_config(config_path: str = "config/config.yaml"):
    """Load configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_iterative_training_experiment(
    train_queries: List[str],
    train_contexts: List[str],
    train_answers: List[str],
    eval_queries: List[str],
    eval_ground_truths: List[str],
    eval_relevant_docs: List[List[int]],
    corpus: List[str],
    config: Dict[str, Any],
    num_iterations: int = 3
):
    """
    Run iterative training experiment.
    
    Args:
        train_queries: Training queries
        train_contexts: Training contexts
        train_answers: Training answers
        eval_queries: Evaluation queries
        eval_ground_truths: Evaluation ground truth answers
        eval_relevant_docs: Evaluation relevant document IDs
        corpus: List of documents
        config: Configuration dictionary
        num_iterations: Number of training iterations
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize evaluator
    evaluator = EvaluationMetrics()
    
    # Initialize pipeline
    pipeline = SelfVerificationRAGPipeline(
        corpus=corpus,
        device=device,
        enable_revision=False,  # Disable revision for training experiment
        use_qlora=config["generation"]["qlora"]["training_enabled"]
    )
    
    # Training history
    training_history = []
    
    for iteration in range(num_iterations):
        print(f"\n=== Iteration {iteration + 1}/{num_iterations} ===")
        
        # Evaluate before training (iteration 0) or after training
        print("Evaluating current model...")
        eval_results = []
        for query, gt, rel_docs in tqdm(zip(eval_queries, eval_ground_truths, eval_relevant_docs),
                                       total=len(eval_queries), desc="Evaluation"):
            result = pipeline.evaluate(query, gt, rel_docs)
            eval_results.append(result)
        
        # Aggregate evaluation metrics
        metric_names = ["factual_precision", "hallucination_rate", "verified_f1", "f1_score"]
        eval_metrics = {}
        for metric_name in metric_names:
            scores = [r["metrics"][metric_name] for r in eval_results]
            eval_metrics[metric_name] = {
                "mean": np.mean(scores),
                "std": np.std(scores)
            }
        
        training_history.append({
            "iteration": iteration,
            "eval_metrics": eval_metrics
        })
        
        print(f"Verified F1: {eval_metrics['verified_f1']['mean']:.4f} ± {eval_metrics['verified_f1']['std']:.4f}")
        print(f"Factual Precision: {eval_metrics['factual_precision']['mean']:.4f} ± {eval_metrics['factual_precision']['std']:.4f}")
        
        # Train if not last iteration
        if iteration < num_iterations - 1:
            print("Training model...")
            
            # Prepare training dataset
            from datasets import Dataset
            trainer = QLoRATrainer(
                pipeline.generator.model,
                pipeline.generator.tokenizer,
                r=config["generation"]["qlora"]["r"],
                lora_alpha=32,
                lora_dropout=0.1
            )
            
            train_dataset = trainer.prepare_dataset(
                train_queries,
                train_contexts,
                train_answers
            )
            
            # Train
            output_dir = f"checkpoints/iteration_{iteration + 1}"
            os.makedirs(output_dir, exist_ok=True)
            
            trainer.train(
                train_dataset=train_dataset,
                output_dir=output_dir,
                num_epochs=config["training"]["num_epochs"],
                batch_size=config["training"]["batch_size"],
                gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
                learning_rate=config["training"]["learning_rate"]
            )
            
            # Load updated model
            pipeline.generator = FLANT5Generator(
                model_name=config["generation"]["model_name"],
                device=device,
                lora_checkpoint=output_dir,
                use_qlora=True
            )
    
    # Plot training curves
    plot_training_curves(training_history)
    
    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/exp6_iterative_training.json", "w") as f:
        json.dump({
            "training_history": training_history
        }, f, indent=2)
    
    print("\n=== Experiment 6: Iterative Training ===")
    print("Training history:")
    for history in training_history:
        print(f"  Iteration {history['iteration']}: "
              f"Verified F1 = {history['eval_metrics']['verified_f1']['mean']:.4f}")
    
    return training_history


def plot_training_curves(training_history: List[Dict]):
    """Plot training curves."""
    iterations = [h["iteration"] for h in training_history]
    
    verified_f1 = [h["eval_metrics"]["verified_f1"]["mean"] for h in training_history]
    factual_precision = [h["eval_metrics"]["factual_precision"]["mean"] for h in training_history]
    hallucination_rate = [h["eval_metrics"]["hallucination_rate"]["mean"] for h in training_history]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Verified F1
    axes[0].plot(iterations, verified_f1, marker='o', label='Verified F1')
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Verified F1')
    axes[0].set_title('Verified F1 vs Training Iteration')
    axes[0].legend()
    axes[0].grid(True)
    
    # Factual Precision
    axes[1].plot(iterations, factual_precision, marker='o', color='green', label='Factual Precision')
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Factual Precision')
    axes[1].set_title('Factual Precision vs Training Iteration')
    axes[1].legend()
    axes[1].grid(True)
    
    # Hallucination Rate
    axes[2].plot(iterations, hallucination_rate, marker='o', color='orange', label='Hallucination Rate')
    axes[2].set_xlabel('Iteration')
    axes[2].set_ylabel('Hallucination Rate')
    axes[2].set_title('Hallucination Rate vs Training Iteration')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    plt.savefig("results/figures/exp6_training_curves.png", dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    config = load_config()
    print("Experiment 6: Iterative Training")
    print("Note: Replace with actual dataset loading")

