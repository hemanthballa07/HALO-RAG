#!/usr/bin/env python3
"""
Validation test for corrected metrics (Verified F1 and Coverage Index)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.metrics import EvaluationMetrics

def test_verified_f1():
    """Test Verified F1 calculation: F1 × Factual Precision"""
    evaluator = EvaluationMetrics()
    
    # Test case from proposal: F1 = 0.60, Factual Precision = 0.70 → Verified F1 = 0.42
    f1_score = 0.60
    factual_precision = 0.70
    verified_f1 = evaluator.verified_f1(f1_score, factual_precision)
    expected = 0.42
    
    assert abs(verified_f1 - expected) < 0.001, f"Expected {expected}, got {verified_f1}"
    print(f"✓ Verified F1 test passed: {verified_f1:.4f} = {f1_score:.2f} × {factual_precision:.2f}")
    
    # Test case from proposal: F1 = 0.58, Factual Precision = 0.92 → Verified F1 = 0.53
    f1_score = 0.58
    factual_precision = 0.92
    verified_f1 = evaluator.verified_f1(f1_score, factual_precision)
    expected = 0.5336  # 0.58 * 0.92
    
    assert abs(verified_f1 - expected) < 0.001, f"Expected {expected}, got {verified_f1}"
    print(f"✓ Verified F1 test passed: {verified_f1:.4f} = {f1_score:.2f} × {factual_precision:.2f}")
    
    # Edge cases
    assert evaluator.verified_f1(0.0, 0.5) == 0.0, "Zero F1 should give zero Verified F1"
    assert evaluator.verified_f1(0.5, 0.0) == 0.0, "Zero Factual Precision should give zero Verified F1"
    print("✓ Verified F1 edge cases passed")


def test_coverage_index():
    """Test Coverage Index: answer tokens in retrieved docs / total answer tokens"""
    evaluator = EvaluationMetrics()
    
    # Test case: all answer tokens appear in retrieved texts
    answer_text = "Paris is the capital of France"
    retrieved_texts = [
        "Paris is a city in France. The capital of France is Paris.",
        "France is a country in Europe."
    ]
    coverage = evaluator.coverage(answer_text, retrieved_texts)
    
    # Answer tokens: {"paris", "is", "the", "capital", "of", "france"} = 6 tokens
    # All tokens should appear in retrieved texts
    assert coverage > 0.9, f"Coverage should be high, got {coverage}"
    print(f"✓ Coverage Index test passed: {coverage:.4f} (high coverage)")
    
    # Test case: partial coverage
    answer_text = "Berlin is the capital of Germany"
    retrieved_texts = [
        "Berlin is a city in Germany.",
        "Germany is a country."
    ]
    coverage = evaluator.coverage(answer_text, retrieved_texts)
    
    # Answer tokens: {"berlin", "is", "the", "capital", "of", "germany"} = 6 tokens
    # Missing "the" and "capital" tokens, so coverage should be around 4/6 = 0.67
    assert 0.6 < coverage < 0.75, f"Coverage should be around 0.67, got {coverage}"
    print(f"✓ Coverage Index test passed: {coverage:.4f} (partial coverage)")
    
    # Edge cases
    assert evaluator.coverage("", ["some text"]) == 0.0, "Empty answer should give zero coverage"
    assert evaluator.coverage("some text", []) == 0.0, "Empty retrieved texts should give zero coverage"
    print("✓ Coverage Index edge cases passed")


if __name__ == "__main__":
    print("=" * 70)
    print("Metrics Validation Test")
    print("=" * 70)
    
    try:
        test_verified_f1()
        print()
        test_coverage_index()
        print()
        print("=" * 70)
        print("✓ All metrics validation tests passed!")
        print("=" * 70)
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

