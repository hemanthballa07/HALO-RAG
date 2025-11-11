"""
Generate Human Evaluation Samples
Creates a CSV of 100 samples for human annotation.
"""

import sys
import os
import argparse
from pathlib import Path
import csv
import json
import random
from typing import List, Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
import torch
import numpy as np
from tqdm import tqdm

from src.data import load_dataset_from_config, prepare_for_experiments
from src.pipeline import SelfVerificationRAGPipeline
from src.evaluation import EvaluationMetrics
from src.utils.cli import parse_experiment_args


def load_config(config_path: str = "config/config.yaml"):
    """Load configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def generate_human_eval_samples(
    queries: List[str],
    ground_truths: List[str],
    relevant_docs: List[List[int]],
    corpus: List[str],
    config: Dict[str, Any],
    num_samples: int = 100,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Generate samples for human evaluation.
    
    Args:
        queries: List of queries
        ground_truths: List of ground truth answers
        relevant_docs: List of relevant document IDs
        corpus: List of documents
        config: Configuration dictionary
        num_samples: Number of samples to generate
        seed: Random seed
    
    Returns:
        List of sample dictionaries
    """
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize pipeline
    print("Initializing pipeline...")
    pipeline = SelfVerificationRAGPipeline(
        corpus=corpus,
        device=device,
        enable_revision=False,  # No revision for human eval
        use_qlora=config.get("generation", {}).get("qlora", {}).get("training_enabled", False)
    )
    
    # Initialize evaluator
    evaluator = EvaluationMetrics()
    
    # Sample queries (stratified sampling if possible)
    if len(queries) > num_samples:
        # Random sampling
        indices = random.sample(range(len(queries)), num_samples)
        sampled_queries = [queries[i] for i in indices]
        sampled_ground_truths = [ground_truths[i] for i in indices]
        sampled_relevant_docs = [relevant_docs[i] for i in indices]
    else:
        sampled_queries = queries
        sampled_ground_truths = ground_truths
        sampled_relevant_docs = relevant_docs
    
    print(f"Generating {len(sampled_queries)} samples for human evaluation...")
    
    samples = []
    
    for idx, (query, gt, rel_docs) in enumerate(tqdm(
        zip(sampled_queries, sampled_ground_truths, sampled_relevant_docs),
        total=len(sampled_queries),
        desc="Generating samples"
    )):
        try:
            # Generate answer
            result = pipeline.generate(query, top_k_retrieve=20, top_k_rerank=5)
            
            # Get context (reranked texts)
            context_texts = result.get("reranked_texts", result.get("retrieved_texts", []))
            context = " ".join(context_texts[:5])  # Top 5 passages
            
            # Get verification results
            verification_results = result["verification_results"]
            verification_data = verification_results.get("verification_results", [])
            
            # Determine auto_label based on verification
            # SUPPORTED: All claims are ENTAILED
            # CONTRADICTED: Any claim is CONTRADICTED
            # NO EVIDENCE: All claims are NO_EVIDENCE or no claims
            if not verification_data:
                auto_label = "NO EVIDENCE"
            else:
                # Extract labels from verification results
                labels = [v.get("label", "NO_EVIDENCE") for v in verification_data]
                
                # Normalize labels (handle different formats)
                normalized_labels = []
                for l in labels:
                    if l == "ENTAILED" or l == "entailment":
                        normalized_labels.append("ENTAILED")
                    elif l == "CONTRADICTED" or l == "contradiction":
                        normalized_labels.append("CONTRADICTED")
                    else:
                        normalized_labels.append("NO_EVIDENCE")
                
                # Determine auto_label
                if all(l == "ENTAILED" for l in normalized_labels):
                    auto_label = "SUPPORTED"
                elif any(l == "CONTRADICTED" for l in normalized_labels):
                    auto_label = "CONTRADICTED"
                else:
                    auto_label = "NO EVIDENCE"
            
            # Create sample
            sample = {
                "id": f"sample_{idx+1:03d}",
                "question": query,
                "context": context,
                "generated_answer": result["generated_text"],
                "gold_answer": gt if gt else "",  # Optional
                "auto_label": auto_label,
                "human_label": "",  # To be filled by annotator
                "notes": ""  # Optional notes from annotator
            }
            
            samples.append(sample)
        
        except Exception as e:
            print(f"Error processing query {idx}: {e}")
            continue
    
    print(f"Generated {len(samples)} samples for human evaluation")
    
    return samples


def save_human_eval_samples(
    samples: List[Dict[str, Any]],
    output_path: str = "results/human_eval/human_eval_samples.csv"
) -> None:
    """
    Save human evaluation samples to CSV.
    
    Args:
        samples: List of sample dictionaries
        output_path: Output CSV file path
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Define columns
    columns = [
        "id", "question", "context", "generated_answer",
        "gold_answer", "auto_label", "human_label", "notes"
    ]
    
    # Write CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        
        for sample in samples:
            writer.writerow(sample)
    
    print(f"✓ Saved {len(samples)} samples to {output_path}")


def main():
    """Main function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Generate Human Evaluation Samples")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Config file path")
    parser.add_argument("--split", type=str, default="validation", choices=["train", "validation", "test"], help="Dataset split")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples from dataset")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Set random seed
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Determine sample limit
    sample_limit = args.limit
    if sample_limit is None:
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
    print(f"Generating {args.num_samples} samples for human evaluation")
    
    # Generate samples
    samples = generate_human_eval_samples(
        queries=queries,
        ground_truths=ground_truths,
        relevant_docs=relevant_docs,
        corpus=corpus,
        config=config,
        num_samples=args.num_samples,
        seed=seed
    )
    
    # Save samples
    output_path = "results/human_eval/human_eval_samples.csv"
    save_human_eval_samples(samples, output_path)
    
    # Print summary
    print("\n" + "=" * 70)
    print("Human Evaluation Samples Summary")
    print("=" * 70)
    print(f"Total samples: {len(samples)}")
    
    # Count by auto_label
    label_counts = {}
    for sample in samples:
        label = sample["auto_label"]
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print("\nAuto-label distribution:")
    for label, count in label_counts.items():
        print(f"  {label}: {count} ({count/len(samples)*100:.1f}%)")
    
    print(f"\n✓ Samples saved to {output_path}")
    print("✓ Annotators can fill in 'human_label' and 'notes' columns")
    print("=" * 70)
    
    return samples


if __name__ == "__main__":
    samples = main()

