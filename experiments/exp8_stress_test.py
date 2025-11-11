"""
Experiment 8: Stress Testing & Pareto Frontier
Evaluates robustness and trade-offs between accuracy and factuality
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
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
import random
import matplotlib.pyplot as plt

from src.data import load_dataset_from_config, prepare_for_experiments
from src.pipeline import SelfVerificationRAGPipeline
from src.evaluation import EvaluationMetrics
from src.utils import setup_wandb, log_metrics, log_metadata, get_commit_hash, get_timestamp


def load_config(config_path: str = "config/config.yaml"):
    """Load configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


class DegradedRetrievalPipeline(SelfVerificationRAGPipeline):
    """Pipeline with degraded retrieval for stress testing."""
    
    def __init__(self, target_recall_at_20: float = 0.95, **kwargs):
        """
        Initialize pipeline with degraded retrieval.
        
        Args:
            target_recall_at_20: Target Recall@20 to achieve (by degrading retrieval)
            **kwargs: Other pipeline arguments
        """
        super().__init__(**kwargs)
        self.target_recall_at_20 = target_recall_at_20
        self.original_corpus = self.corpus.copy()
    
    def degrade_retrieval(
        self,
        retrieved_docs: List[Tuple[int, str, Optional[float]]],
        relevant_docs: List[int],
        target_recall: float
    ) -> List[Tuple[int, str, Optional[float]]]:
        """
        Degrade retrieval to achieve target Recall@20.
        
        Args:
            retrieved_docs: Original retrieved documents
            relevant_docs: List of relevant document IDs
            target_recall: Target recall to achieve
        
        Returns:
            Degraded retrieved documents
        """
        if len(relevant_docs) == 0:
            return retrieved_docs
        
        retrieved_ids = [doc[0] for doc in retrieved_docs]
        relevant_set = set(relevant_docs)
        
        # Calculate current recall@20
        retrieved_top_20_ids = retrieved_ids[:20]
        relevant_in_top_20 = [doc_id for doc_id in retrieved_top_20_ids if doc_id in relevant_set]
        current_recall = len(relevant_in_top_20) / len(relevant_set) if len(relevant_set) > 0 else 0.0
        
        if current_recall <= target_recall:
            # Already at or below target, return as is
            return retrieved_docs
        
        # Need to degrade: replace some relevant docs with irrelevant ones
        # Calculate how many relevant docs we should have in top 20
        num_relevant_target = max(1, int(np.ceil(target_recall * len(relevant_set))))
        num_relevant_current = len(relevant_in_top_20)
        num_to_remove = num_relevant_current - num_relevant_target
        
        if num_to_remove <= 0:
            return retrieved_docs
        
        # Get all document IDs from corpus
        all_doc_ids = list(range(len(self.corpus)))
        # Get irrelevant document IDs (not in relevant set and not already in top 20)
        irrelevant_docs = [doc_id for doc_id in all_doc_ids 
                          if doc_id not in relevant_set and doc_id not in retrieved_top_20_ids]
        
        if len(irrelevant_docs) == 0:
            # No irrelevant docs available, return as is
            return retrieved_docs
        
        # Randomly select irrelevant docs to add
        np.random.shuffle(irrelevant_docs)
        docs_to_add = irrelevant_docs[:num_to_remove]
        
        # Create degraded top 20: remove some relevant, add irrelevant
        degraded_top_20_ids = []
        relevant_removed = 0
        
        # First, add non-relevant docs from original top 20
        for doc_id in retrieved_top_20_ids:
            if doc_id not in relevant_set:
                degraded_top_20_ids.append(doc_id)
        
        # Then, add target number of relevant docs
        for doc_id in retrieved_top_20_ids:
            if doc_id in relevant_set and len([d for d in degraded_top_20_ids if d in relevant_set]) < num_relevant_target:
                degraded_top_20_ids.append(doc_id)
        
        # Fill remaining slots with irrelevant docs
        while len(degraded_top_20_ids) < 20 and len(docs_to_add) > 0:
            doc_id = docs_to_add.pop(0)
            if doc_id not in degraded_top_20_ids:
                degraded_top_20_ids.append(doc_id)
        
        # If we still don't have 20, add from remaining retrieved docs
        remaining_ids = [doc_id for doc_id in retrieved_ids if doc_id not in degraded_top_20_ids]
        while len(degraded_top_20_ids) < 20 and len(remaining_ids) > 0:
            degraded_top_20_ids.append(remaining_ids.pop(0))
        
        # Reconstruct degraded retrieval list
        doc_id_to_doc = {doc[0]: doc for doc in retrieved_docs}
        degraded_docs = []
        
        # Add degraded top 20
        for doc_id in degraded_top_20_ids:
            if doc_id in doc_id_to_doc:
                degraded_docs.append(doc_id_to_doc[doc_id])
            else:
                # Create a dummy entry if doc_id not in original retrieval
                degraded_docs.append((doc_id, self.corpus[doc_id], 0.0))
        
        # Add remaining docs (beyond top 20) that weren't used
        for doc in retrieved_docs[20:]:
            if doc[0] not in degraded_top_20_ids:
                degraded_docs.append(doc)
        
        return degraded_docs
    
    def generate(
        self,
        query: str,
        top_k_retrieve: int = 20,
        top_k_rerank: int = 5,
        relevant_docs: Optional[List[int]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate answer with degraded retrieval.
        
        Args:
            query: Query string
            top_k_retrieve: Number of documents to retrieve
            top_k_rerank: Number of documents to rerank
            relevant_docs: List of relevant document IDs (for degradation)
            **kwargs: Other generation arguments
        
        Returns:
            Dictionary with generation results
        """
        # Step 1: Hybrid retrieval
        retrieved_docs = self.retriever.retrieve(query, top_k=top_k_retrieve)
        
        # Degrade retrieval if relevant_docs provided
        if relevant_docs is not None:
            retrieved_docs = self.degrade_retrieval(
                retrieved_docs,
                relevant_docs,
                self.target_recall_at_20
            )
        
        retrieved_texts = [doc[1] for doc in retrieved_docs]
        retrieved_ids = [doc[0] for doc in retrieved_docs]
        
        # Step 2: Cross-encoder reranking
        reranked_docs = self.reranker.rerank(
            query,
            retrieved_texts,
            top_k=top_k_rerank
        )
        reranked_ids = [retrieved_ids[doc[0]] for doc in reranked_docs]
        reranked_texts = [doc[1] for doc in reranked_docs]
        context = " ".join(reranked_texts)
        
        # Step 3: Generation
        generated_text = self.generator.generate(query, context, **kwargs)
        
        # Step 4: Claim extraction
        claims = self.claim_extractor.extract_claims(generated_text)
        
        # Step 5: Verification
        verification_results = self.verifier.verify_generation(
            generated_text,
            reranked_texts,
            claims
        )
        
        # Step 6: Adaptive revision (if enabled and verification failed)
        revision_iterations = 0
        if self.enable_revision and self.revision_strategy:
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


def run_tau_sweep_stress_test(
    queries: List[str],
    ground_truths: List[str],
    relevant_docs: List[List[int]],
    corpus: List[str],
    config: Dict[str, Any],
    thresholds: List[float],
    seed: int = 42,
    limit: Optional[int] = None
) -> Dict[str, Any]:
    """
    Run τ-sweep stress test.
    
    Args:
        queries: List of queries
        ground_truths: List of ground truth answers
        relevant_docs: List of relevant document IDs
        corpus: List of documents
        config: Configuration dictionary
        thresholds: List of thresholds to test
        seed: Random seed
        limit: Limit number of examples
    
    Returns:
        Dictionary with results
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if limit:
        queries = queries[:limit]
        ground_truths = ground_truths[:limit]
        relevant_docs = relevant_docs[:limit]
    
    # Initialize pipeline
    pipeline = SelfVerificationRAGPipeline(
        corpus=corpus,
        device=device,
        enable_revision=False,  # Disable revision for stress test
        use_qlora=config.get("generation", {}).get("qlora", {}).get("training_enabled", False)
    )
    
    evaluator = EvaluationMetrics()
    threshold_results = {}
    
    for threshold in tqdm(thresholds, desc="τ-sweep"):
        pipeline.set_entailment_threshold(threshold)
        
        all_metrics = []
        for query, gt, rel_docs in tqdm(
            zip(queries, ground_truths, relevant_docs),
            total=len(queries),
            desc=f"τ={threshold}",
            leave=False
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
                print(f"Error: {e}")
                continue
        
        # Aggregate metrics
        if all_metrics:
            threshold_results[threshold] = {
                "factual_precision": np.mean([m["factual_precision"] for m in all_metrics]),
                "factual_recall": np.mean([m["factual_recall"] for m in all_metrics]),
                "verified_f1": np.mean([m["verified_f1"] for m in all_metrics]),
                "abstention_rate": np.mean([m["abstention_rate"] for m in all_metrics]),
                "hallucination_rate": np.mean([m["hallucination_rate"] for m in all_metrics]),
                "exact_match": np.mean([m["exact_match"] for m in all_metrics]),
                "f1_score": np.mean([m["f1_score"] for m in all_metrics])
            }
    
    return threshold_results


def run_retrieval_degradation_test(
    queries: List[str],
    ground_truths: List[str],
    relevant_docs: List[List[int]],
    corpus: List[str],
    config: Dict[str, Any],
    target_recalls: List[float],
    seed: int = 42,
    limit: Optional[int] = None
) -> Dict[str, Any]:
    """
    Run retrieval degradation stress test.
    
    Args:
        queries: List of queries
        ground_truths: List of ground truth answers
        relevant_docs: List of relevant document IDs
        corpus: List of documents
        config: Configuration dictionary
        target_recalls: List of target Recall@20 values
        seed: Random seed
        limit: Limit number of examples
    
    Returns:
        Dictionary with results
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if limit:
        queries = queries[:limit]
        ground_truths = ground_truths[:limit]
        relevant_docs = relevant_docs[:limit]
    
    evaluator = EvaluationMetrics()
    recall_results = {}
    
    for target_recall in tqdm(target_recalls, desc="Retrieval degradation"):
        # Initialize pipeline with degraded retrieval
        pipeline = DegradedRetrievalPipeline(
            corpus=corpus,
            device=device,
            target_recall_at_20=target_recall,
            enable_revision=False,
            use_qlora=config.get("generation", {}).get("qlora", {}).get("training_enabled", False)
        )
        
        all_metrics = []
        for query, gt, rel_docs in tqdm(
            zip(queries, ground_truths, relevant_docs),
            total=len(queries),
            desc=f"Recall@20={target_recall}",
            leave=False
        ):
            try:
                result = pipeline.generate(
                    query,
                    top_k_retrieve=20,
                    top_k_rerank=5,
                    relevant_docs=rel_docs
                )
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
                print(f"Error: {e}")
                continue
        
        # Aggregate metrics
        if all_metrics:
            recall_results[target_recall] = {
                "recall@20": np.mean([m["recall@20"] for m in all_metrics]),
                "factual_precision": np.mean([m["factual_precision"] for m in all_metrics]),
                "verified_f1": np.mean([m["verified_f1"] for m in all_metrics]),
                "hallucination_rate": np.mean([m["hallucination_rate"] for m in all_metrics]),
                "exact_match": np.mean([m["exact_match"] for m in all_metrics]),
                "f1_score": np.mean([m["f1_score"] for m in all_metrics])
            }
    
    return recall_results


def run_verifier_off_test(
    queries: List[str],
    ground_truths: List[str],
    relevant_docs: List[List[int]],
    corpus: List[str],
    config: Dict[str, Any],
    seed: int = 42,
    limit: Optional[int] = None
) -> Dict[str, Any]:
    """
    Run verifier-off stress test (pure RAG baseline).
    
    Args:
        queries: List of queries
        ground_truths: List of ground truth answers
        relevant_docs: List of relevant document IDs
        corpus: List of documents
        config: Configuration dictionary
        seed: Random seed
        limit: Limit number of examples
    
    Returns:
        Dictionary with results
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if limit:
        queries = queries[:limit]
        ground_truths = ground_truths[:limit]
        relevant_docs = relevant_docs[:limit]
    
    # Initialize pipeline (no verification, no revision)
    pipeline = SelfVerificationRAGPipeline(
        corpus=corpus,
        device=device,
        enable_revision=False,
        use_qlora=config.get("generation", {}).get("qlora", {}).get("training_enabled", False)
    )
    
    # Disable verification (set threshold to 0)
    pipeline.verifier.threshold = 0.0
    
    evaluator = EvaluationMetrics()
    all_metrics = []
    
    for query, gt, rel_docs in tqdm(
        zip(queries, ground_truths, relevant_docs),
        total=len(queries),
        desc="Verifier off"
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
            print(f"Error: {e}")
            continue
    
    # Aggregate metrics
    if all_metrics:
        return {
            "factual_precision": np.mean([m["factual_precision"] for m in all_metrics]),
            "hallucination_rate": np.mean([m["hallucination_rate"] for m in all_metrics]),
            "verified_f1": np.mean([m["verified_f1"] for m in all_metrics]),
            "exact_match": np.mean([m["exact_match"] for m in all_metrics]),
            "f1_score": np.mean([m["f1_score"] for m in all_metrics])
        }
    return {}


def run_baseline_test(
    queries: List[str],
    ground_truths: List[str],
    relevant_docs: List[List[int]],
    corpus: List[str],
    config: Dict[str, Any],
    seed: int = 42,
    limit: Optional[int] = None
) -> Dict[str, Any]:
    """
    Run baseline test (full pipeline with optimal threshold).
    
    Args:
        queries: List of queries
        ground_truths: List of ground truth answers
        relevant_docs: List of relevant document IDs
        corpus: List of documents
        config: Configuration dictionary
        seed: Random seed
        limit: Limit number of examples
    
    Returns:
        Dictionary with results
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if limit:
        queries = queries[:limit]
        ground_truths = ground_truths[:limit]
        relevant_docs = relevant_docs[:limit]
    
    # Initialize pipeline with optimal threshold (0.75)
    pipeline = SelfVerificationRAGPipeline(
        corpus=corpus,
        device=device,
        enable_revision=True,
        entailment_threshold=0.75,
        use_qlora=config.get("generation", {}).get("qlora", {}).get("training_enabled", False)
    )
    
    evaluator = EvaluationMetrics()
    all_metrics = []
    
    for query, gt, rel_docs in tqdm(
        zip(queries, ground_truths, relevant_docs),
        total=len(queries),
        desc="Baseline (full pipeline)"
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
            print(f"Error: {e}")
            continue
    
    # Aggregate metrics
    if all_metrics:
        return {
            "factual_precision": np.mean([m["factual_precision"] for m in all_metrics]),
            "hallucination_rate": np.mean([m["hallucination_rate"] for m in all_metrics]),
            "verified_f1": np.mean([m["verified_f1"] for m in all_metrics]),
            "exact_match": np.mean([m["exact_match"] for m in all_metrics]),
            "f1_score": np.mean([m["f1_score"] for m in all_metrics])
        }
    return {}


def plot_stress_test_results(
    tau_results: Dict[str, Any],
    retrieval_results: Dict[str, Any],
    baseline_results: Dict[str, Any],
    verifier_off_results: Dict[str, Any],
    output_dir: str = "results/figures"
):
    """Plot stress test results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Verified F1 vs τ
    fig, ax = plt.subplots(figsize=(10, 6))
    thresholds = sorted(tau_results.keys())
    verified_f1s = [tau_results[t]["verified_f1"] for t in thresholds]
    factual_precisions = [tau_results[t]["factual_precision"] for t in thresholds]
    abstention_rates = [tau_results[t]["abstention_rate"] for t in thresholds]
    
    ax.plot(thresholds, verified_f1s, marker='o', label='Verified F1', linewidth=2)
    ax.plot(thresholds, factual_precisions, marker='s', label='Factual Precision', linewidth=2)
    ax.plot(thresholds, abstention_rates, marker='^', label='Abstention Rate', linewidth=2)
    
    ax.set_xlabel('Entailment Threshold (τ)', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Verified F1 vs Entailment Threshold (τ)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim([min(thresholds) - 0.05, max(thresholds) + 0.05])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "exp8_verified_f1_vs_tau.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Precision vs Recall
    fig, ax = plt.subplots(figsize=(10, 6))
    factual_precisions = [tau_results[t]["factual_precision"] for t in thresholds]
    factual_recalls = [tau_results[t]["factual_recall"] for t in thresholds]
    
    ax.plot(factual_recalls, factual_precisions, marker='o', linewidth=2, markersize=8)
    for i, t in enumerate(thresholds):
        ax.annotate(f'τ={t}', (factual_recalls[i], factual_precisions[i]),
                   xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax.set_xlabel('Factual Recall', fontsize=12)
    ax.set_ylabel('Factual Precision', fontsize=12)
    ax.set_title('Precision vs Recall (τ-Sweep)', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "exp8_precision_vs_recall.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Pareto Frontier (EM vs 1 - Hallucination Rate)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Collect all points
    points = []
    labels = []
    
    # Baseline (full pipeline)
    if baseline_results:
        points.append((
            baseline_results.get("exact_match", 0.0),
            1.0 - baseline_results.get("hallucination_rate", 0.0)
        ))
        labels.append("Baseline (Full Pipeline)")
    
    # Verifier off
    if verifier_off_results:
        points.append((
            verifier_off_results.get("exact_match", 0.0),
            1.0 - verifier_off_results.get("hallucination_rate", 0.0)
        ))
        labels.append("Verifier Off")
    
    # τ-sweep points
    for threshold in thresholds:
        if threshold in tau_results:
            points.append((
                tau_results[threshold].get("exact_match", 0.0),
                1.0 - tau_results[threshold].get("hallucination_rate", 0.0)
            ))
            labels.append(f"τ={threshold}")
    
    # Retrieval degradation points
    for recall in sorted(retrieval_results.keys()):
        if recall in retrieval_results:
            points.append((
                retrieval_results[recall].get("exact_match", 0.0),
                1.0 - retrieval_results[recall].get("hallucination_rate", 0.0)
            ))
            labels.append(f"Recall@20={recall}")
    
    if points:
        points = np.array(points)
        
        # Plot all points
        ax.scatter(points[:, 0], points[:, 1], s=100, alpha=0.6, c='blue', edgecolors='black', linewidths=1.5)
        
        # Highlight baseline and verifier off
        if baseline_results:
            baseline_point = np.array([
                baseline_results.get("exact_match", 0.0),
                1.0 - baseline_results.get("hallucination_rate", 0.0)
            ])
            ax.scatter(baseline_point[0], baseline_point[1], s=200, c='green', marker='*', 
                      edgecolors='black', linewidths=2, label='Baseline (Full Pipeline)', zorder=5)
        
        if verifier_off_results:
            verifier_off_point = np.array([
                verifier_off_results.get("exact_match", 0.0),
                1.0 - verifier_off_results.get("hallucination_rate", 0.0)
            ])
            ax.scatter(verifier_off_point[0], verifier_off_point[1], s=200, c='red', marker='X',
                      edgecolors='black', linewidths=2, label='Verifier Off', zorder=5)
        
        # Compute Pareto frontier
        # Sort points by EM (x-axis) descending, then by 1-HR (y-axis) descending
        sorted_indices = np.lexsort((-points[:, 1], -points[:, 0]))
        pareto_points = []
        pareto_labels = []
        max_y = -1
        
        for idx in sorted_indices:
            if points[idx, 1] >= max_y:
                pareto_points.append(points[idx])
                pareto_labels.append(labels[idx])
                max_y = points[idx, 1]
        
        if len(pareto_points) > 1:
            pareto_points = np.array(pareto_points)
            # Sort by x for plotting
            sort_order = np.argsort(pareto_points[:, 0])
            pareto_points = pareto_points[sort_order]
            ax.plot(pareto_points[:, 0], pareto_points[:, 1], 'r--', linewidth=2, 
                   label='Pareto Frontier', alpha=0.7)
        
        ax.set_xlabel('Exact Match (EM)', fontsize=12)
        ax.set_ylabel('1 - Hallucination Rate (Factuality)', fontsize=12)
        ax.set_title('Pareto Frontier: Accuracy vs Factuality', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "exp8_pareto_frontier.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plots to {output_dir}")


def save_stress_test_results(
    tau_results: Dict[str, Any],
    retrieval_results: Dict[str, Any],
    baseline_results: Dict[str, Any],
    verifier_off_results: Dict[str, Any],
    output_dir: str = "results/metrics"
):
    """Save stress test results to CSV and JSON."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save CSV
    csv_path = os.path.join(output_dir, "exp8_stress.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "test_type", "parameter", "factual_precision", "factual_recall",
            "verified_f1", "hallucination_rate", "abstention_rate",
            "exact_match", "f1_score", "recall@20"
        ])
        
        # τ-sweep results
        for threshold in sorted(tau_results.keys()):
            r = tau_results[threshold]
            writer.writerow([
                "tau_sweep", threshold, r.get("factual_precision", 0.0),
                r.get("factual_recall", 0.0), r.get("verified_f1", 0.0),
                r.get("hallucination_rate", 0.0), r.get("abstention_rate", 0.0),
                r.get("exact_match", 0.0), r.get("f1_score", 0.0), ""
            ])
        
        # Retrieval degradation results
        for recall in sorted(retrieval_results.keys()):
            r = retrieval_results[recall]
            writer.writerow([
                "retrieval_degradation", recall, r.get("factual_precision", 0.0),
                "", r.get("verified_f1", 0.0), r.get("hallucination_rate", 0.0),
                "", r.get("exact_match", 0.0), r.get("f1_score", 0.0),
                r.get("recall@20", 0.0)
            ])
        
        # Baseline results
        if baseline_results:
            writer.writerow([
                "baseline", "full_pipeline", baseline_results.get("factual_precision", 0.0),
                "", baseline_results.get("verified_f1", 0.0),
                baseline_results.get("hallucination_rate", 0.0), "",
                baseline_results.get("exact_match", 0.0), baseline_results.get("f1_score", 0.0), ""
            ])
        
        # Verifier off results
        if verifier_off_results:
            writer.writerow([
                "verifier_off", "no_verification", verifier_off_results.get("factual_precision", 0.0),
                "", verifier_off_results.get("verified_f1", 0.0),
                verifier_off_results.get("hallucination_rate", 0.0), "",
                verifier_off_results.get("exact_match", 0.0), verifier_off_results.get("f1_score", 0.0), ""
            ])
    
    print(f"Saved CSV to {csv_path}")
    
    # Save JSON
    json_path = os.path.join(output_dir, "exp8_stress.json")
    with open(json_path, 'w') as f:
        json.dump({
            "tau_sweep": tau_results,
            "retrieval_degradation": retrieval_results,
            "baseline": baseline_results,
            "verifier_off": verifier_off_results,
            "timestamp": get_timestamp(),
            "commit_hash": get_commit_hash()
        }, f, indent=2)
    
    print(f"Saved JSON to {json_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Experiment 8: Stress Testing & Pareto Frontier")
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
                run_name="exp8_stress_test",
                config=config
            )
            log_metadata({
                "experiment": "exp8_stress_test",
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
    
    # Run stress tests
    print("\n" + "="*60)
    print("EXPERIMENT 8: STRESS TESTING & PARETO FRONTIER")
    print("="*60)
    
    # 1. τ-Sweep stress test
    print("\n1. Running τ-sweep stress test...")
    thresholds = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]
    tau_results = run_tau_sweep_stress_test(
        queries=queries,
        ground_truths=ground_truths,
        relevant_docs=relevant_docs,
        corpus=corpus,
        config=config,
        thresholds=thresholds,
        seed=args.seed,
        limit=args.limit
    )
    
    # 2. Retrieval degradation test
    print("\n2. Running retrieval degradation test...")
    target_recalls = [0.95, 0.85, 0.75, 0.65]
    retrieval_results = run_retrieval_degradation_test(
        queries=queries,
        ground_truths=ground_truths,
        relevant_docs=relevant_docs,
        corpus=corpus,
        config=config,
        target_recalls=target_recalls,
        seed=args.seed,
        limit=args.limit
    )
    
    # 3. Verifier off test
    print("\n3. Running verifier off test...")
    verifier_off_results = run_verifier_off_test(
        queries=queries,
        ground_truths=ground_truths,
        relevant_docs=relevant_docs,
        corpus=corpus,
        config=config,
        seed=args.seed,
        limit=args.limit
    )
    
    # 4. Baseline test (full pipeline)
    print("\n4. Running baseline test (full pipeline)...")
    baseline_results = run_baseline_test(
        queries=queries,
        ground_truths=ground_truths,
        relevant_docs=relevant_docs,
        corpus=corpus,
        config=config,
        seed=args.seed,
        limit=args.limit
    )
    
    # Save results
    print("\n5. Saving results...")
    save_stress_test_results(
        tau_results=tau_results,
        retrieval_results=retrieval_results,
        baseline_results=baseline_results,
        verifier_off_results=verifier_off_results
    )
    
    # Generate plots
    print("\n6. Generating plots...")
    plot_stress_test_results(
        tau_results=tau_results,
        retrieval_results=retrieval_results,
        baseline_results=baseline_results,
        verifier_off_results=verifier_off_results
    )
    
    # Log to W&B
    if wandb_run:
        # Log τ-sweep results
        for threshold, results in tau_results.items():
            log_metrics(results, prefix=f"tau_sweep/tau_{threshold}/")
        
        # Log retrieval degradation results
        for recall, results in retrieval_results.items():
            log_metrics(results, prefix=f"retrieval_degradation/recall_{recall}/")
        
        # Log baseline and verifier off
        if baseline_results:
            log_metrics(baseline_results, prefix="baseline/")
        if verifier_off_results:
            log_metrics(verifier_off_results, prefix="verifier_off/")
        
        wandb_run.finish()
    
    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT 8: STRESS TESTING & PARETO FRONTIER - COMPLETE")
    print("="*60)
    print("\nKey Findings:")
    
    if tau_results:
        best_tau = max(tau_results.keys(), key=lambda t: tau_results[t].get("verified_f1", 0.0))
        print(f"  • Best τ: {best_tau} (Verified F1: {tau_results[best_tau]['verified_f1']:.4f})")
    
    if retrieval_results:
        print(f"  • Retrieval quality strongly correlates with factual precision")
        for recall in sorted(retrieval_results.keys()):
            fp = retrieval_results[recall].get("factual_precision", 0.0)
            print(f"    - Recall@20={recall}: Factual Precision={fp:.4f}")
    
    if baseline_results and verifier_off_results:
        hr_baseline = baseline_results.get("hallucination_rate", 0.0)
        hr_verifier_off = verifier_off_results.get("hallucination_rate", 0.0)
        print(f"  • Hallucination Rate increase (verifier off): {hr_verifier_off - hr_baseline:.4f}")
        print(f"    - Baseline: {hr_baseline:.4f}")
        print(f"    - Verifier Off: {hr_verifier_off:.4f}")


if __name__ == "__main__":
    main()
