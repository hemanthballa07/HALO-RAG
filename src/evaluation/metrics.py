"""
Evaluation Metrics Module
Implements Recall@K, MRR, NDCG, Coverage, Factual Precision, Verified F1, etc.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import re


class EvaluationMetrics:
    """
    Comprehensive evaluation metrics for RAG pipeline.
    """
    
    def __init__(self):
        """Initialize evaluation metrics."""
        pass
    
    def recall_at_k(
        self,
        retrieved_docs: List[int],
        relevant_docs: List[int],
        k: int
    ) -> float:
        """
        Compute Recall@K.
        
        Args:
            retrieved_docs: List of retrieved document IDs
            relevant_docs: List of relevant document IDs
            k: Number of top documents to consider
        
        Returns:
            Recall@K score
        """
        if len(relevant_docs) == 0:
            return 0.0
        
        retrieved_top_k = set(retrieved_docs[:k])
        relevant_set = set(relevant_docs)
        
        intersection = len(retrieved_top_k & relevant_set)
        recall = intersection / len(relevant_set)
        
        return recall
    
    def precision_at_k(
        self,
        retrieved_docs: List[int],
        relevant_docs: List[int],
        k: int
    ) -> float:
        """
        Compute Precision@K.
        
        Args:
            retrieved_docs: List of retrieved document IDs
            relevant_docs: List of relevant document IDs
            k: Number of top documents to consider
        
        Returns:
            Precision@K score
        """
        if k == 0:
            return 0.0
        
        retrieved_top_k = set(retrieved_docs[:k])
        relevant_set = set(relevant_docs)
        
        intersection = len(retrieved_top_k & relevant_set)
        precision = intersection / k
        
        return precision
    
    def mrr(
        self,
        retrieved_docs: List[int],
        relevant_docs: List[int]
    ) -> float:
        """
        Compute Mean Reciprocal Rank (MRR).
        
        Args:
            retrieved_docs: List of retrieved document IDs
            relevant_docs: List of relevant document IDs
        
        Returns:
            MRR score
        """
        relevant_set = set(relevant_docs)
        
        for rank, doc_id in enumerate(retrieved_docs, 1):
            if doc_id in relevant_set:
                return 1.0 / rank
        
        return 0.0
    
    def ndcg_at_k(
        self,
        retrieved_docs: List[int],
        relevant_docs: List[int],
        k: int,
        scores: Optional[List[float]] = None
    ) -> float:
        """
        Compute Normalized Discounted Cumulative Gain@K.
        
        Args:
            retrieved_docs: List of retrieved document IDs
            relevant_docs: List of relevant document IDs
            k: Number of top documents to consider
            scores: Optional relevance scores
        
        Returns:
            NDCG@K score
        """
        if k == 0:
            return 0.0
        
        relevant_set = set(relevant_docs)
        
        # DCG
        dcg = 0.0
        for rank, doc_id in enumerate(retrieved_docs[:k], 1):
            if doc_id in relevant_set:
                if scores:
                    relevance = scores[rank - 1]
                else:
                    relevance = 1.0
                dcg += relevance / np.log2(rank + 1)
        
        # IDCG (ideal DCG)
        idcg = 0.0
        num_relevant = min(len(relevant_set), k)
        for rank in range(1, num_relevant + 1):
            idcg += 1.0 / np.log2(rank + 1)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def coverage(
        self,
        retrieved_docs: List[int],
        corpus_size: int,
        k: int = 20
    ) -> float:
        """
        Compute Coverage: fraction of corpus covered by top-k retrievals.
        
        Args:
            retrieved_docs: List of retrieved document IDs
            corpus_size: Total corpus size
            k: Number of top documents to consider
        
        Returns:
            Coverage score
        """
        if corpus_size == 0:
            return 0.0
        
        unique_docs = len(set(retrieved_docs[:k]))
        coverage = unique_docs / corpus_size
        
        return coverage
    
    def factual_precision(
        self,
        verification_results: List[Dict[str, any]]
    ) -> float:
        """
        Compute Factual Precision: fraction of claims that are entailed.
        
        Args:
            verification_results: List of verification results
        
        Returns:
            Factual precision score
        """
        if len(verification_results) == 0:
            return 0.0
        
        num_entailed = sum(
            1 for r in verification_results
            if r.get("is_entailed", False)
        )
        
        precision = num_entailed / len(verification_results)
        
        return precision
    
    def factual_recall(
        self,
        verification_results: List[Dict[str, any]],
        ground_truth_claims: List[str]
    ) -> float:
        """
        Compute Factual Recall: fraction of ground truth claims that are entailed.
        
        Args:
            verification_results: List of verification results
            ground_truth_claims: List of ground truth claims
        
        Returns:
            Factual recall score
        """
        if len(ground_truth_claims) == 0:
            return 0.0
        
        # Extract verified claims
        verified_claims = {
            r["claim"] for r in verification_results
            if r.get("is_entailed", False)
        }
        
        ground_truth_set = set(ground_truth_claims)
        intersection = len(verified_claims & ground_truth_set)
        
        recall = intersection / len(ground_truth_set)
        
        return recall
    
    def hallucination_rate(
        self,
        verification_results: List[Dict[str, any]]
    ) -> float:
        """
        Compute Hallucination Rate: fraction of claims that are not entailed.
        
        Args:
            verification_results: List of verification results
        
        Returns:
            Hallucination rate
        """
        if len(verification_results) == 0:
            return 0.0
        
        num_unverified = sum(
            1 for r in verification_results
            if not r.get("is_entailed", False)
        )
        
        hallucination_rate = num_unverified / len(verification_results)
        
        return hallucination_rate
    
    def verified_f1(
        self,
        factual_precision: float,
        factual_recall: float
    ) -> float:
        """
        Compute Verified F1: harmonic mean of factual precision and recall.
        
        Args:
            factual_precision: Factual precision score
            factual_recall: Factual recall score
        
        Returns:
            Verified F1 score
        """
        if factual_precision + factual_recall == 0:
            return 0.0
        
        f1 = 2 * (factual_precision * factual_recall) / (factual_precision + factual_recall)
        
        return f1
    
    def exact_match(
        self,
        generated: str,
        ground_truth: str
    ) -> float:
        """
        Compute Exact Match: 1.0 if exact match, 0.0 otherwise.
        
        Args:
            generated: Generated text
            ground_truth: Ground truth text
        
        Returns:
            Exact match score (0.0 or 1.0)
        """
        generated_clean = generated.strip().lower()
        ground_truth_clean = ground_truth.strip().lower()
        
        return 1.0 if generated_clean == ground_truth_clean else 0.0
    
    def f1_score(
        self,
        generated: str,
        ground_truth: str
    ) -> float:
        """
        Compute F1 score based on token overlap.
        
        Args:
            generated: Generated text
            ground_truth: Ground truth text
        
        Returns:
            F1 score
        """
        # Tokenize
        gen_tokens = set(generated.lower().split())
        gt_tokens = set(ground_truth.lower().split())
        
        if len(gen_tokens) == 0 or len(gt_tokens) == 0:
            return 0.0
        
        intersection = len(gen_tokens & gt_tokens)
        
        precision = intersection / len(gen_tokens) if len(gen_tokens) > 0 else 0.0
        recall = intersection / len(gt_tokens) if len(gt_tokens) > 0 else 0.0
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        
        return f1
    
    def compute_all_metrics(
        self,
        retrieved_docs: List[int],
        relevant_docs: List[int],
        verification_results: List[Dict[str, any]],
        generated: str,
        ground_truth: str,
        corpus_size: int,
        ground_truth_claims: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Compute all metrics at once.
        
        Args:
            retrieved_docs: List of retrieved document IDs
            relevant_docs: List of relevant document IDs
            verification_results: List of verification results
            generated: Generated text
            ground_truth: Ground truth text
            corpus_size: Total corpus size
            ground_truth_claims: Optional list of ground truth claims
        
        Returns:
            Dictionary of all metric scores
        """
        metrics = {}
        
        # Retrieval metrics
        metrics["recall@5"] = self.recall_at_k(retrieved_docs, relevant_docs, 5)
        metrics["recall@10"] = self.recall_at_k(retrieved_docs, relevant_docs, 10)
        metrics["recall@20"] = self.recall_at_k(retrieved_docs, relevant_docs, 20)
        metrics["precision@5"] = self.precision_at_k(retrieved_docs, relevant_docs, 5)
        metrics["precision@10"] = self.precision_at_k(retrieved_docs, relevant_docs, 10)
        metrics["mrr"] = self.mrr(retrieved_docs, relevant_docs)
        metrics["ndcg@10"] = self.ndcg_at_k(retrieved_docs, relevant_docs, 10)
        metrics["coverage"] = self.coverage(retrieved_docs, corpus_size, k=20)
        
        # Verification metrics
        metrics["factual_precision"] = self.factual_precision(verification_results)
        metrics["factual_recall"] = self.factual_recall(
            verification_results,
            ground_truth_claims or []
        )
        metrics["hallucination_rate"] = self.hallucination_rate(verification_results)
        metrics["verified_f1"] = self.verified_f1(
            metrics["factual_precision"],
            metrics["factual_recall"]
        )
        
        # Generation metrics
        metrics["exact_match"] = self.exact_match(generated, ground_truth)
        metrics["f1_score"] = self.f1_score(generated, ground_truth)
        
        return metrics

