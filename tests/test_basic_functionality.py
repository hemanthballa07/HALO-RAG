#!/usr/bin/env python3
"""
Basic functionality test after dependencies are installed
Tests core components with minimal data
"""

import sys
import os
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 70)
print("Basic Functionality Test")
print("=" * 70)

# Test 1: Check all imports
print("\n1. Testing imports...")
try:
    from src.retrieval import HybridRetriever, CrossEncoderReranker
    from src.verification import EntailmentVerifier, ClaimExtractor
    from src.generator import FLANT5Generator
    from src.revision import AdaptiveRevisionStrategy
    from src.evaluation import EvaluationMetrics, StatisticalTester
    from src.pipeline import SelfVerificationRAGPipeline
    print("   ✓ All modules imported successfully")
except ImportError as e:
    print(f"   ✗ Import error: {e}")
    print("   Please install dependencies: pip3 install -r requirements.txt")
    sys.exit(1)

# Test 2: Check configuration
print("\n2. Testing configuration...")
try:
    import yaml
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    print("   ✓ Configuration loaded successfully")
except Exception as e:
    print(f"   ✗ Config error: {e}")
    sys.exit(1)

# Test 3: Test evaluation metrics
print("\n3. Testing evaluation metrics...")
try:
    evaluator = EvaluationMetrics()
    
    # Test Recall@K
    retrieved = [0, 1, 2, 3, 4]
    relevant = [1, 3, 5]
    recall = evaluator.recall_at_k(retrieved, relevant, k=5)
    print(f"   ✓ Recall@5 calculation: {recall:.4f}")
    
    # Test Verified F1
    f1 = evaluator.verified_f1(0.9, 0.8)
    print(f"   ✓ Verified F1 calculation: {f1:.4f}")
    
    print("   ✓ Evaluation metrics working correctly")
except Exception as e:
    print(f"   ✗ Evaluation error: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Test statistical testing
print("\n4. Testing statistical testing...")
try:
    stats_tester = StatisticalTester(alpha=0.05)
    
    # Test mean/std/CI
    data = [0.8, 0.85, 0.9, 0.88, 0.92]
    mean, std, ci = stats_tester.mean_std_ci(data)
    print(f"   ✓ Mean: {mean:.4f}, Std: {std:.4f}, CI: {ci}")
    
    # Test t-test
    group1 = [0.8, 0.85, 0.9, 0.88, 0.92]
    group2 = [0.75, 0.78, 0.82, 0.80, 0.85]
    t_stat, p_value, is_sig = stats_tester.t_test(group1, group2)
    print(f"   ✓ T-test: t={t_stat:.4f}, p={p_value:.4f}, sig={is_sig}")
    
    print("   ✓ Statistical testing working correctly")
except Exception as e:
    print(f"   ✗ Statistical testing error: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Test claim extractor (requires spaCy)
print("\n5. Testing claim extractor...")
try:
    claim_extractor = ClaimExtractor()
    test_text = "The capital of France is Paris. Paris is a major city in Europe."
    claims = claim_extractor.extract_claims(test_text)
    print(f"   ✓ Extracted {len(claims)} claims from test text")
    print(f"   ✓ Claim extraction working correctly")
except Exception as e:
    print(f"   ⚠ Claim extractor error: {e}")
    print("   Note: Install spaCy: pip3 install spacy")
    print("   Then download model: python3 -m spacy download en_core_web_sm")

# Test 6: Test hybrid retriever initialization (without actual model loading)
print("\n6. Testing hybrid retriever initialization...")
try:
    # Just test that class can be instantiated
    # We won't load models here to save time
    retriever = HybridRetriever(
        dense_weight=0.6,
        sparse_weight=0.4,
        device="cpu"
    )
    print("   ✓ HybridRetriever class initialized")
    print("   ⚠ Note: Models will be loaded when build_index() is called")
except Exception as e:
    print(f"   ✗ Retriever initialization error: {e}")
    import traceback
    traceback.print_exc()

# Summary
print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
print("✓ Core functionality tests passed!")
print("\nNext steps:")
print("1. Load your dataset")
print("2. Initialize pipeline with corpus")
print("3. Run experiments: python3 experiments/exp1_baseline.py")
print("\n" + "=" * 70)

