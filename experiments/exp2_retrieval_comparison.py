"""
Experiment 2: Retrieval Comparison
Dense vs Sparse vs Hybrid Retrieval
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

from src.retrieval import HybridRetriever
from src.evaluation import EvaluationMetrics, StatisticalTester


def load_config(config_path: str = "config/config.yaml"):
    """Load configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_retrieval_comparison(
    queries: List[str],
    relevant_docs: List[List[int]],
    corpus: List[str],
    config: Dict[str, Any]
):
    """
    Compare dense, sparse, and hybrid retrieval methods.
    
    Args:
        queries: List of queries
        relevant_docs: List of relevant document IDs for each query
        corpus: List of documents
        config: Configuration dictionary
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize retriever
    retriever = HybridRetriever(
        dense_weight=0.6,
        sparse_weight=0.4,
        device=device
    )
    retriever.build_index(corpus)
    
    # Initialize evaluator
    evaluator = EvaluationMetrics()
    stats_tester = StatisticalTester(alpha=0.05)
    
    # Results for each method
    dense_results = []
    sparse_results = []
    hybrid_results = []
    
    for query, rel_docs in tqdm(zip(queries, relevant_docs), 
                                total=len(queries), desc="Retrieval comparison"):
        # Dense retrieval
        dense_retrieved = retriever.retrieve_dense_only(query, top_k=20)
        dense_ids = [doc[0] for doc in dense_retrieved]
        dense_metrics = {
            "recall@5": evaluator.recall_at_k(dense_ids, rel_docs, 5),
            "recall@10": evaluator.recall_at_k(dense_ids, rel_docs, 10),
            "recall@20": evaluator.recall_at_k(dense_ids, rel_docs, 20),
            "mrr": evaluator.mrr(dense_ids, rel_docs),
            "ndcg@10": evaluator.ndcg_at_k(dense_ids, rel_docs, 10)
        }
        dense_results.append(dense_metrics)
        
        # Sparse retrieval
        sparse_retrieved = retriever.retrieve_sparse_only(query, top_k=20)
        sparse_ids = [doc[0] for doc in sparse_retrieved]
        sparse_metrics = {
            "recall@5": evaluator.recall_at_k(sparse_ids, rel_docs, 5),
            "recall@10": evaluator.recall_at_k(sparse_ids, rel_docs, 10),
            "recall@20": evaluator.recall_at_k(sparse_ids, rel_docs, 20),
            "mrr": evaluator.mrr(sparse_ids, rel_docs),
            "ndcg@10": evaluator.ndcg_at_k(sparse_ids, rel_docs, 10)
        }
        sparse_results.append(sparse_metrics)
        
        # Hybrid retrieval
        hybrid_retrieved = retriever.retrieve(query, top_k=20)
        hybrid_ids = [doc[0] for doc in hybrid_retrieved]
        hybrid_metrics = {
            "recall@5": evaluator.recall_at_k(hybrid_ids, rel_docs, 5),
            "recall@10": evaluator.recall_at_k(hybrid_ids, rel_docs, 10),
            "recall@20": evaluator.recall_at_k(hybrid_ids, rel_docs, 20),
            "mrr": evaluator.mrr(hybrid_ids, rel_docs),
            "ndcg@10": evaluator.ndcg_at_k(hybrid_ids, rel_docs, 10)
        }
        hybrid_results.append(hybrid_metrics)
    
    # Aggregate metrics
    methods = {
        "dense": dense_results,
        "sparse": sparse_results,
        "hybrid": hybrid_results
    }
    
    aggregated = {}
    for method_name, method_results in methods.items():
        aggregated[method_name] = {}
        for metric_name in ["recall@5", "recall@10", "recall@20", "mrr", "ndcg@10"]:
            scores = [r[metric_name] for r in method_results]
            aggregated[method_name][metric_name] = {
                "mean": np.mean(scores),
                "std": np.std(scores),
                "scores": scores
            }
    
    # Statistical comparison
    comparisons = {}
    for metric_name in ["recall@20", "mrr", "ndcg@10"]:
        dense_scores = [r[metric_name] for r in dense_results]
        sparse_scores = [r[metric_name] for r in sparse_results]
        hybrid_scores = [r[metric_name] for r in hybrid_results]
        
        # Compare hybrid vs dense
        comp_dense = stats_tester.compare_metrics(
            dense_scores, hybrid_scores, f"{metric_name}_hybrid_vs_dense"
        )
        
        # Compare hybrid vs sparse
        comp_sparse = stats_tester.compare_metrics(
            sparse_scores, hybrid_scores, f"{metric_name}_hybrid_vs_sparse"
        )
        
        comparisons[metric_name] = {
            "hybrid_vs_dense": comp_dense,
            "hybrid_vs_sparse": comp_sparse
        }
    
    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/exp2_retrieval_comparison.json", "w") as f:
        json.dump({
            "aggregated_metrics": aggregated,
            "statistical_comparisons": comparisons
        }, f, indent=2)
    
    print("\n=== Experiment 2: Retrieval Comparison ===")
    for method_name, metrics in aggregated.items():
        print(f"\n{method_name.upper()}:")
        for metric_name, stats in metrics.items():
            print(f"  {metric_name}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    return aggregated, comparisons


if __name__ == "__main__":
    config = load_config()
    
    # Load your dataset here
    # queries, relevant_docs, corpus = load_dataset(...)
    
    # Run experiment
    # aggregated, comparisons = run_retrieval_comparison(
    #     queries, relevant_docs, corpus, config
    # )

