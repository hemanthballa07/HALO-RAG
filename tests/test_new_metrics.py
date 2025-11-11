#!/usr/bin/env python3
"""
Test new metrics: FEVER Score, BLEU-4, ROUGE-L, Abstention Rate
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.metrics import EvaluationMetrics

def test_bleu4():
    """Test BLEU-4 score calculation"""
    evaluator = EvaluationMetrics()
    
    # Test case: Exact match
    generated = "The capital of France is Paris"
    ground_truth = "The capital of France is Paris"
    bleu = evaluator.bleu4(generated, ground_truth)
    print(f"✓ BLEU-4 (exact match): {bleu:.4f}")
    assert bleu > 0.9, f"BLEU-4 should be high for exact match, got {bleu}"
    
    # Test case: Partial match
    generated = "Paris is the capital of France"
    ground_truth = "The capital of France is Paris"
    bleu = evaluator.bleu4(generated, ground_truth)
    print(f"✓ BLEU-4 (partial match): {bleu:.4f}")
    assert bleu > 0.5, f"BLEU-4 should be moderate for partial match, got {bleu}"
    
    # Test case: No match
    generated = "Berlin is the capital of Germany"
    ground_truth = "The capital of France is Paris"
    bleu = evaluator.bleu4(generated, ground_truth)
    print(f"✓ BLEU-4 (no match): {bleu:.4f}")
    assert bleu < 0.3, f"BLEU-4 should be low for no match, got {bleu}"


def test_rouge_l():
    """Test ROUGE-L score calculation"""
    evaluator = EvaluationMetrics()
    
    # Test case: Exact match
    generated = "The capital of France is Paris"
    ground_truth = "The capital of France is Paris"
    rouge = evaluator.rouge_l(generated, ground_truth)
    print(f"✓ ROUGE-L (exact match): {rouge:.4f}")
    assert rouge > 0.9, f"ROUGE-L should be high for exact match, got {rouge}"
    
    # Test case: Partial match
    generated = "Paris is the capital of France"
    ground_truth = "The capital of France is Paris"
    rouge = evaluator.rouge_l(generated, ground_truth)
    print(f"✓ ROUGE-L (partial match): {rouge:.4f}")
    assert rouge > 0.7, f"ROUGE-L should be high for partial match, got {rouge}"


def test_fever_score():
    """Test FEVER Score calculation"""
    evaluator = EvaluationMetrics()
    
    # Test case: All claims supported, all tokens in evidence
    verification_results = [
        {"claim": "Paris is capital", "is_entailed": True},
        {"claim": "France is country", "is_entailed": True}
    ]
    retrieved_texts = ["Paris is the capital of France. France is a country."]
    ground_truth = "Paris is the capital of France"
    
    fever = evaluator.fever_score(verification_results, retrieved_texts, ground_truth)
    print(f"✓ FEVER Score (high): {fever:.4f}")
    assert fever > 0.8, f"FEVER Score should be high, got {fever}"
    
    # Test case: Some claims not supported
    verification_results = [
        {"claim": "Paris is capital", "is_entailed": True},
        {"claim": "Berlin is capital", "is_entailed": False}
    ]
    retrieved_texts = ["Paris is the capital of France."]
    ground_truth = "Paris is the capital of France"
    
    fever = evaluator.fever_score(verification_results, retrieved_texts, ground_truth)
    print(f"✓ FEVER Score (moderate): {fever:.4f}")
    assert 0.4 < fever < 0.9, f"FEVER Score should be moderate, got {fever}"


def test_abstention_rate():
    """Test Abstention Rate calculation"""
    evaluator = EvaluationMetrics()
    
    # Test case: Abstention phrase
    generated = "Cannot answer based on context"
    abstention = evaluator.abstention_rate(generated)
    print(f"✓ Abstention Rate (abstention): {abstention:.4f}")
    assert abstention == 1.0, f"Abstention rate should be 1.0 for abstention, got {abstention}"
    
    # Test case: Normal answer
    generated = "Paris is the capital of France"
    abstention = evaluator.abstention_rate(generated)
    print(f"✓ Abstention Rate (normal): {abstention:.4f}")
    assert abstention == 0.0, f"Abstention rate should be 0.0 for normal answer, got {abstention}"
    
    # Test case: Insufficient evidence phrase
    generated = "Insufficient evidence to answer this question"
    abstention = evaluator.abstention_rate(generated)
    print(f"✓ Abstention Rate (insufficient evidence): {abstention:.4f}")
    assert abstention == 1.0, f"Abstention rate should be 1.0, got {abstention}"


if __name__ == "__main__":
    print("=" * 70)
    print("New Metrics Validation Test")
    print("=" * 70)
    
    try:
        test_bleu4()
        print()
        test_rouge_l()
        print()
        test_fever_score()
        print()
        test_abstention_rate()
        print()
        print("=" * 70)
        print("✓ All new metrics validation tests passed!")
        print("=" * 70)
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

