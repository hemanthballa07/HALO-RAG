"""
Verified Data Collector for Iterative Fine-Tuning.
Collects examples with Factual Precision ≥ threshold.
"""

import os
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
from collections import Counter

logger = logging.getLogger(__name__)


def compute_diversity_stats(examples: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Compute diversity statistics for collected examples.
    
    Args:
        examples: List of verified examples
    
    Returns:
        Dictionary with diversity statistics
    """
    if not examples:
        return {
            "type_token_ratio": 0.0,
            "avg_question_length": 0.0,
            "avg_answer_length": 0.0,
            "avg_context_length": 0.0,
            "total_examples": 0
        }
    
    # Collect all tokens
    all_tokens = []
    question_lengths = []
    answer_lengths = []
    context_lengths = []
    
    for example in examples:
        question = example.get("question", "")
        answer = example.get("verified_answer", "")
        context = example.get("context", "")
        
        # Tokenize (simple whitespace split)
        question_tokens = question.lower().split()
        answer_tokens = answer.lower().split()
        context_tokens = context.lower().split()
        
        all_tokens.extend(question_tokens)
        all_tokens.extend(answer_tokens)
        all_tokens.extend(context_tokens)
        
        question_lengths.append(len(question_tokens))
        answer_lengths.append(len(answer_tokens))
        context_lengths.append(len(context_tokens))
    
    # Type-token ratio (unique tokens / total tokens)
    unique_tokens = len(set(all_tokens))
    total_tokens = len(all_tokens)
    type_token_ratio = unique_tokens / total_tokens if total_tokens > 0 else 0.0
    
    return {
        "type_token_ratio": type_token_ratio,
        "avg_question_length": sum(question_lengths) / len(question_lengths) if question_lengths else 0.0,
        "avg_answer_length": sum(answer_lengths) / len(answer_lengths) if answer_lengths else 0.0,
        "avg_context_length": sum(context_lengths) / len(context_lengths) if context_lengths else 0.0,
        "total_examples": len(examples),
        "unique_tokens": unique_tokens,
        "total_tokens": total_tokens
    }


def collect_verified_data(
    pipeline,
    queries: List[str],
    ground_truths: List[str],
    relevant_docs: List[List[int]],
    corpus: List[str],
    factual_precision_threshold: float = 0.85,
    top_k_passages: int = 5
) -> List[Dict[str, Any]]:
    """
    Collect verified training data with Factual Precision ≥ threshold.
    
    Args:
        pipeline: RAG pipeline
        queries: List of queries
        ground_truths: List of ground truth answers
        relevant_docs: List of relevant document IDs
        corpus: List of documents
        factual_precision_threshold: Minimum factual precision (default: 0.85)
        top_k_passages: Number of top passages to include (default: 5)
    
    Returns:
        List of verified examples with Factual Precision ≥ threshold
    """
    verified_examples = []
    
    logger.info(f"Collecting verified data with FP ≥ {factual_precision_threshold}...")
    
    for idx, (query, gt, rel_docs) in enumerate(zip(queries, ground_truths, relevant_docs)):
        try:
            # Generate answer
            result = pipeline.generate(
                query,
                top_k_retrieve=20,
                top_k_rerank=top_k_passages
            )
            
            # Get verification results
            verification_results = result["verification_results"]
            verification_data = verification_results.get("verification_results", [])
            
            # Compute factual precision
            if verification_data:
                verified_count = sum(1 for v in verification_data if v.get("label") == "ENTAILED")
                total_claims = len(verification_data)
                factual_precision = verified_count / total_claims if total_claims > 0 else 0.0
            else:
                factual_precision = 0.0
            
            # Check if meets threshold
            if factual_precision >= factual_precision_threshold:
                # Get top-k passages (reranked texts)
                top_k_passages_texts = result.get("reranked_texts", result.get("retrieved_texts", []))[:top_k_passages]
                context = " ".join(top_k_passages_texts)
                
                # Create verified example
                verified_example = {
                    "question": query,
                    "context": context,
                    "top_k_passages": top_k_passages_texts,
                    "verified_answer": result["generated_text"],
                    "factual_precision": factual_precision,
                    "ground_truth": gt,
                    "verification_results": verification_data,
                    "claims": result.get("claims", [])
                }
                
                verified_examples.append(verified_example)
            
            if (idx + 1) % 100 == 0:
                logger.info(f"Processed {idx + 1}/{len(queries)} queries, collected {len(verified_examples)} verified examples")
        
        except Exception as e:
            logger.warning(f"Error processing query {idx}: {e}")
            continue
    
    logger.info(f"Collected {len(verified_examples)} verified examples (FP ≥ {factual_precision_threshold})")
    
    return verified_examples


def save_verified_data(
    examples: List[Dict[str, Any]],
    output_path: str,
    iteration: int
) -> None:
    """
    Save verified examples to JSONL file.
    
    Args:
        examples: List of verified examples
        output_path: Output file path
        iteration: Iteration number
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        for example in examples:
            # Create training triple: (question, top-k passages, verified_answer)
            training_triple = {
                "question": example["question"],
                "context": example["context"],
                "top_k_passages": example["top_k_passages"],
                "verified_answer": example["verified_answer"],
                "factual_precision": example["factual_precision"],
                "iteration": iteration
            }
            f.write(json.dumps(training_triple) + '\n')
    
    logger.info(f"Saved {len(examples)} verified examples to {output_path}")


def load_verified_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Load verified examples from JSONL file.
    
    Args:
        file_path: Path to JSONL file
    
    Returns:
        List of verified examples
    """
    examples = []
    
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                example = json.loads(line)
                examples.append(example)
    
    logger.info(f"Loaded {len(examples)} verified examples from {file_path}")
    return examples

