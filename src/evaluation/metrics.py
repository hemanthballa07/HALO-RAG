"""
Evaluation Metrics Module
Implements Recall@K, MRR, NDCG, Coverage, Factual Precision, Verified F1, etc.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import re

# For BLEU-4
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

# For ROUGE-L
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False


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
        answer_text: str,
        retrieved_texts: List[str]
    ) -> float:
        """
        Compute Coverage Index: fraction of answer tokens that appear in retrieved documents.
        
        Proposal definition: Coverage Index = (answer tokens in retrieved docs) / (total answer tokens)
        Target: Coverage ≥ 0.90
        
        Args:
            answer_text: Generated or ground truth answer text
            retrieved_texts: List of retrieved document texts
        
        Returns:
            Coverage Index score (0.0 to 1.0)
        """
        if not answer_text or not retrieved_texts:
            return 0.0
        
        # Tokenize answer (lowercase, split on whitespace)
        answer_tokens = set(answer_text.lower().split())
        
        if len(answer_tokens) == 0:
            return 0.0
        
        # Combine all retrieved texts and tokenize
        combined_retrieved_text = " ".join(retrieved_texts)
        retrieved_tokens = set(combined_retrieved_text.lower().split())
        
        # Find answer tokens that appear in retrieved documents
        answer_tokens_in_retrieved = answer_tokens & retrieved_tokens
        
        # Coverage Index = answer tokens in retrieved / total answer tokens
        coverage = len(answer_tokens_in_retrieved) / len(answer_tokens)
        
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
        ground_truth_claims: List[str],
        retrieved_texts: List[str] = None,
        verifier = None
    ) -> float:
        """
        Compute Factual Recall: fraction of ground truth claims that are entailed by the context.
        
        This verifies ground truth claims against the retrieved context to see if they're supported.
        
        Args:
            verification_results: List of verification results (for context, not used directly)
            ground_truth_claims: List of ground truth claims to verify
            retrieved_texts: List of retrieved context documents
            verifier: Verifier instance to verify ground truth claims (optional)
        
        Returns:
            Factual recall score
        """
        if len(ground_truth_claims) == 0:
            return 0.0
        
        # If we have a verifier and retrieved texts, verify ground truth claims against context
        if verifier is not None and retrieved_texts is not None and len(retrieved_texts) > 0:
            # Verify each ground truth claim against the retrieved context
            combined_context = " ".join(retrieved_texts[:3])  # Use top 3 contexts
            num_entailed = 0
            for gt_claim in ground_truth_claims:
                is_entailed, _ = verifier.is_entailed(gt_claim, combined_context)
                if is_entailed:
                    num_entailed += 1
            recall = num_entailed / len(ground_truth_claims)
            return recall
        
        # Fallback: Compare ground truth claims with verified generated claims
        # This is less accurate but works if verifier is not available
        verified_claims = {
            r["claim"] for r in verification_results
            if r.get("is_entailed", False)
        }
        
        ground_truth_set = set(ground_truth_claims)
        intersection = len(verified_claims & ground_truth_set)
        
        recall = intersection / len(ground_truth_set) if len(ground_truth_set) > 0 else 0.0
        
        return recall
    
    def hallucination_rate(
        self,
        verification_results: List[Dict[str, any]],
        abstained: bool = False
    ) -> float:
        """
        Compute Hallucination Rate: fraction of claims that are not entailed.
        
        When the system abstains (abstained=True), it means no confident claims were made,
        so hallucination_rate should be 0.0 (no hallucinations if no claims were made).
        
        Args:
            verification_results: List of verification results
            abstained: Whether the system abstained from making a claim
        
        Returns:
            Hallucination rate (0.0 if abstained, otherwise fraction of unentailed claims)
        """
        # If system abstained, no claims were made, so no hallucinations
        if abstained:
            return 0.0
        
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
        f1_score: float,
        factual_precision: float
    ) -> float:
        """
        Compute Verified F1: F1 score multiplied by Factual Precision.
        
        Proposal definition (Section 3.1 Stage 4): Verified F1 = F1 × Factual Precision
        This composite metric shows factuality AND quality together.
        
        Examples:
        - Baseline RAG: F1 = 0.60, Factual Precision = 0.70 → Verified F1 = 0.42
        - Verified RAG: F1 = 0.58, Factual Precision = 0.92 → Verified F1 = 0.53 (+26%)
        Target: Verified F1 ≥ 0.52
        
        Args:
            f1_score: F1 score (token overlap between generated and ground truth)
            factual_precision: Factual precision score (fraction of claims that are entailed)
        
        Returns:
            Verified F1 score (0.0 to 1.0)
        """
        return f1_score * factual_precision
    
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
    
    def bleu4(
        self,
        generated: str,
        ground_truth: str
    ) -> float:
        """
        Compute BLEU-4 score for multi-sentence answers.
        
        Proposal requirement: BLEU-4 for text similarity (use for HotpotQA).
        
        Args:
            generated: Generated text
            ground_truth: Ground truth text
        
        Returns:
            BLEU-4 score (0.0 to 1.0)
        """
        if not NLTK_AVAILABLE:
            # Fallback: return 0.0 if NLTK not available
            return 0.0
        
        # Tokenize
        gen_tokens = generated.lower().split()
        gt_tokens = ground_truth.lower().split()
        
        if len(gen_tokens) == 0 or len(gt_tokens) == 0:
            return 0.0
        
        # BLEU-4 with smoothing
        smoothing = SmoothingFunction().method1
        score = sentence_bleu(
            [gt_tokens],
            gen_tokens,
            smoothing_function=smoothing,
            weights=(0.25, 0.25, 0.25, 0.25)  # BLEU-4 weights
        )
        
        return float(score)
    
    def rouge_l(
        self,
        generated: str,
        ground_truth: str
    ) -> float:
        """
        Compute ROUGE-L score for multi-sentence answers.
        
        Proposal requirement: ROUGE-L for text similarity.
        ROUGE-L measures longest common subsequence (LCS) based F-score.
        
        Args:
            generated: Generated text
            ground_truth: Ground truth text
        
        Returns:
            ROUGE-L F1 score (0.0 to 1.0)
        """
        if not ROUGE_AVAILABLE:
            # Fallback: return 0.0 if rouge_score not available
            return 0.0
        
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = scorer.score(ground_truth, generated)
        
        # Return F1 score (harmonic mean of precision and recall)
        return float(scores['rougeL'].fmeasure)
    
    def fever_score(
        self,
        verification_results: List[Dict[str, any]],
        retrieved_texts: List[str],
        ground_truth: str
    ) -> float:
        """
        Compute FEVER Score: harmonic mean of label accuracy and evidence recall.
        
        Proposal requirement: FEVER Score = harmonic_mean(label_accuracy × evidence_recall)
        
        FEVER Score measures:
        - Label accuracy: Fraction of claims correctly labeled (SUPPORTED/REFUTED/NO_EVIDENCE)
        - Evidence recall: Fraction of ground truth tokens found in retrieved evidence
        
        Args:
            verification_results: List of verification results with labels
            retrieved_texts: List of retrieved document texts (evidence)
            ground_truth: Ground truth answer text
        
        Returns:
            FEVER Score (0.0 to 1.0)
        """
        if len(verification_results) == 0:
            return 0.0
        
        # Label accuracy: Fraction of claims that are entailed (SUPPORTED)
        # For FEVER, we consider a claim SUPPORTED if entailment_score > threshold
        num_supported = sum(
            1 for r in verification_results
            if r.get("is_entailed", False)
        )
        label_accuracy = num_supported / len(verification_results) if len(verification_results) > 0 else 0.0
        
        # Evidence recall: Fraction of ground truth tokens found in retrieved evidence
        if not ground_truth or not retrieved_texts:
            evidence_recall = 0.0
        else:
            gt_tokens = set(ground_truth.lower().split())
            combined_evidence = " ".join(retrieved_texts).lower()
            evidence_tokens = set(combined_evidence.split())
            
            if len(gt_tokens) == 0:
                evidence_recall = 0.0
            else:
                gt_tokens_in_evidence = gt_tokens & evidence_tokens
                evidence_recall = len(gt_tokens_in_evidence) / len(gt_tokens)
        
        # FEVER Score: Harmonic mean of label accuracy and evidence recall
        if label_accuracy + evidence_recall == 0:
            return 0.0
        
        fever_score = 2 * (label_accuracy * evidence_recall) / (label_accuracy + evidence_recall)
        
        return fever_score
    
    def abstention_rate(
        self,
        generated: str
    ) -> float:
        """
        Compute Abstention Rate: % of "insufficient evidence" responses.
        
        Proposal requirement: Abstention Rate should increase as verification strengthens.
        System should respond with "Cannot answer based on context" when insufficient evidence.
        
        Args:
            generated: Generated text
        
        Returns:
            Abstention rate (0.0 to 1.0), where 1.0 means all responses are abstentions
        """
        if not generated:
            return 0.0
        
        generated_lower = generated.lower().strip()
        
        # Check for common abstention phrases
        abstention_phrases = [
            "cannot answer",
            "cannot be answered",
            "insufficient evidence",
            "no information",
            "not enough information",
            "unable to answer",
            "cannot determine",
            "cannot provide",
            "no answer",
            "not available"
        ]
        
        # Check if generated text contains any abstention phrase
        is_abstention = any(phrase in generated_lower for phrase in abstention_phrases)
        
        # Also check if the text is very short (might indicate abstention)
        # or if it's exactly one of the abstention phrases
        if len(generated_lower.split()) <= 3:
            # Very short responses might be abstentions
            is_abstention = is_abstention or any(
                generated_lower == phrase or generated_lower.startswith(phrase)
                for phrase in abstention_phrases
            )
        
        return 1.0 if is_abstention else 0.0
    
    def compute_all_metrics(
        self,
        retrieved_docs: List[int],
        relevant_docs: List[int],
        verification_results: List[Dict[str, any]],
        generated: str,
        ground_truth: str,
        retrieved_texts: List[str],
        ground_truth_claims: Optional[List[str]] = None,
        verifier = None,
        abstained: bool = False
    ) -> Dict[str, float]:
        """
        Compute all metrics at once.
        
        Args:
            retrieved_docs: List of retrieved document IDs
            relevant_docs: List of relevant document IDs
            verification_results: List of verification results
            generated: Generated text
            ground_truth: Ground truth text
            retrieved_texts: List of retrieved document texts (for coverage calculation)
            ground_truth_claims: Optional list of ground truth claims
            verifier: Optional verifier instance for factual recall calculation
            abstained: Whether the system abstained from making a claim
        
        Returns:
            Dictionary of all metric scores including:
            - Retrieval: recall@k, precision@k, mrr, ndcg@10, coverage
            - Verification: factual_precision, factual_recall, hallucination_rate, fever_score
            - Generation: exact_match, f1_score, bleu4, rouge_l, abstention_rate
            - Composite: verified_f1
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
        
        # Coverage Index: answer tokens in retrieved docs / total answer tokens
        metrics["coverage"] = self.coverage(ground_truth, retrieved_texts)
        
        # Verification metrics
        metrics["factual_precision"] = self.factual_precision(verification_results)
        metrics["factual_recall"] = self.factual_recall(
            verification_results,
            ground_truth_claims or [],
            retrieved_texts=retrieved_texts,
            verifier=verifier
        )
        metrics["hallucination_rate"] = self.hallucination_rate(verification_results, abstained=abstained)
        
        # Generation metrics
        metrics["exact_match"] = self.exact_match(generated, ground_truth)
        metrics["f1_score"] = self.f1_score(generated, ground_truth)
        
        # Verified F1: F1 × Factual Precision (proposal definition)
        metrics["verified_f1"] = self.verified_f1(
            metrics["f1_score"],
            metrics["factual_precision"]
        )
        
        # BLEU-4: For multi-sentence answers (HotpotQA)
        metrics["bleu4"] = self.bleu4(generated, ground_truth)
        
        # ROUGE-L: For multi-sentence answers
        metrics["rouge_l"] = self.rouge_l(generated, ground_truth)
        
        # FEVER Score: Harmonic mean of label accuracy and evidence recall
        metrics["fever_score"] = self.fever_score(
            verification_results,
            retrieved_texts,
            ground_truth
        )
        
        # Abstention Rate: % of "insufficient evidence" responses
        metrics["abstention_rate"] = self.abstention_rate(generated)
        
        return metrics

