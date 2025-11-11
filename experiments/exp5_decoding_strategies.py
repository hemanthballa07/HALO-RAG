"""
Experiment 5: Decoding Strategies
Greedy vs Beam vs Nucleus sampling
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

from src.pipeline import SelfVerificationRAGPipeline
from src.evaluation import EvaluationMetrics, StatisticalTester


def load_config(config_path: str = "config/config.yaml"):
    """Load configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_decoding_strategies_experiment(
    queries: List[str],
    ground_truths: List[str],
    relevant_docs: List[List[int]],
    corpus: List[str],
    config: Dict[str, Any]
):
    """
    Compare decoding strategies.
    
    Args:
        queries: List of queries
        ground_truths: List of ground truth answers
        relevant_docs: List of relevant document IDs for each query
        corpus: List of documents
        config: Configuration dictionary
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize evaluator
    evaluator = EvaluationMetrics()
    stats_tester = StatisticalTester(alpha=0.05)
    
    # Initialize pipeline
    pipeline = SelfVerificationRAGPipeline(
        corpus=corpus,
        device=device,
        enable_revision=False,  # Disable revision for decoding comparison
        use_qlora=config["generation"]["qlora"]["training_enabled"]
    )
    
    # Decoding strategies
    strategies = {
        "greedy": {"do_sample": False, "num_beams": 1},
        "beam_3": {"do_sample": False, "num_beams": 3},
        "beam_5": {"do_sample": False, "num_beams": 5},
        "nucleus": {"do_sample": True, "temperature": 0.7, "top_p": 0.95}
    }
    
    all_results = {}
    
    for strategy_name, strategy_params in strategies.items():
        print(f"Running {strategy_name} decoding...")
        
        results = []
        for query, gt, rel_docs in tqdm(zip(queries, ground_truths, relevant_docs),
                                       total=len(queries), desc=strategy_name):
            # Generate with specific decoding strategy
            generation_result = pipeline.generate(query)
            
            # Override generation with custom parameters
            context = generation_result["context"]
            generated = pipeline.generator.generate(
                query, context, **strategy_params
            )
            
            # Re-verify
            claims = pipeline.claim_extractor.extract_claims(generated)
            # Get reranked texts (not IDs) for verification
            reranked_texts = generation_result.get("reranked_texts", [])
            verification = pipeline.verifier.verify_generation(
                generated,
                reranked_texts,
                claims
            )
            
            # Evaluate
            # Get retrieved texts from generation result for coverage calculation
            retrieved_texts = generation_result.get("reranked_texts", generation_result.get("retrieved_texts", []))
            metrics = evaluator.compute_all_metrics(
                retrieved_docs=generation_result["retrieved_docs"],
                relevant_docs=rel_docs,
                verification_results=verification["verification_results"],
                generated=generated,
                ground_truth=gt,
                retrieved_texts=retrieved_texts
            )
            
            results.append({
                "query": query,
                "generated": generated,
                "metrics": metrics
            })
        
        all_results[strategy_name] = results
    
    # Aggregate metrics
    metric_names = [
        "factual_precision", "hallucination_rate",
        "verified_f1", "f1_score"
    ]
    
    aggregated = {}
    for strategy_name, results in all_results.items():
        aggregated[strategy_name] = {}
        for metric_name in metric_names:
            scores = [r["metrics"][metric_name] for r in results]
            aggregated[strategy_name][metric_name] = {
                "mean": np.mean(scores),
                "std": np.std(scores),
                "scores": scores
            }
    
    # Statistical comparisons
    comparisons = {}
    baseline_strategy = "greedy"
    baseline_scores = {
        metric: aggregated[baseline_strategy][metric]["scores"]
        for metric in metric_names
    }
    
    for strategy_name in strategies.keys():
        if strategy_name == baseline_strategy:
            continue
        
        comparisons[strategy_name] = {}
        for metric_name in metric_names:
            strategy_scores = aggregated[strategy_name][metric_name]["scores"]
            comparison = stats_tester.compare_metrics(
                baseline_scores[metric_name],
                strategy_scores,
                f"{metric_name}_{strategy_name}_vs_{baseline_strategy}"
            )
            comparisons[strategy_name][metric_name] = comparison
    
    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/exp5_decoding_strategies.json", "w") as f:
        json.dump({
            "aggregated_metrics": aggregated,
            "statistical_comparisons": comparisons
        }, f, indent=2)
    
    print("\n=== Experiment 5: Decoding Strategies ===")
    for strategy_name, metrics in aggregated.items():
        print(f"\n{strategy_name.upper()}:")
        for metric_name, stats in metrics.items():
            print(f"  {metric_name}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    return aggregated, comparisons


if __name__ == "__main__":
    config = load_config()
    
    # Load your dataset here
    # queries, ground_truths, relevant_docs, corpus = load_dataset(...)
    
    # Run experiment
    # aggregated, comparisons = run_decoding_strategies_experiment(
    #     queries, ground_truths, relevant_docs, corpus, config
    # )

