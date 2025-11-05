#!/usr/bin/env python3
"""
Test script to check all imports and basic functionality
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("Testing Self-Verification RAG Pipeline Imports")
print("=" * 60)

# Test 1: Basic Python imports
print("\n1. Testing basic Python imports...")
try:
    import torch
    import numpy as np
    import yaml
    print("   ✓ torch, numpy, yaml imported successfully")
except ImportError as e:
    print(f"   ✗ Import error: {e}")
    sys.exit(1)

# Test 2: Configuration
print("\n2. Testing configuration loading...")
try:
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    print(f"   ✓ Config loaded: Dense={config['retrieval']['fusion']['dense_weight']}, "
          f"Sparse={config['retrieval']['fusion']['sparse_weight']}")
except Exception as e:
    print(f"   ✗ Config error: {e}")
    sys.exit(1)

# Test 3: Retrieval module
print("\n3. Testing retrieval module...")
try:
    from src.retrieval import HybridRetriever, CrossEncoderReranker
    print("   ✓ Retrieval module imported successfully")
except ImportError as e:
    print(f"   ✗ Retrieval import error: {e}")
    print("   Note: This is OK if dependencies are not installed yet")

# Test 4: Verification module
print("\n4. Testing verification module...")
try:
    from src.verification import EntailmentVerifier, ClaimExtractor
    print("   ✓ Verification module imported successfully")
except ImportError as e:
    print(f"   ✗ Verification import error: {e}")
    print("   Note: This is OK if dependencies are not installed yet")

# Test 5: Generator module
print("\n5. Testing generator module...")
try:
    from src.generator import FLANT5Generator, QLoRATrainer
    print("   ✓ Generator module imported successfully")
except ImportError as e:
    print(f"   ✗ Generator import error: {e}")
    print("   Note: This is OK if dependencies are not installed yet")

# Test 6: Revision module
print("\n6. Testing revision module...")
try:
    from src.revision import AdaptiveRevisionStrategy
    print("   ✓ Revision module imported successfully")
except ImportError as e:
    print(f"   ✗ Revision import error: {e}")
    print("   Note: This is OK if dependencies are not installed yet")

# Test 7: Evaluation module
print("\n7. Testing evaluation module...")
try:
    from src.evaluation import EvaluationMetrics, StatisticalTester
    print("   ✓ Evaluation module imported successfully")
except ImportError as e:
    print(f"   ✗ Evaluation import error: {e}")
    print("   Note: This is OK if dependencies are not installed yet")

# Test 8: Pipeline module
print("\n8. Testing pipeline module...")
try:
    from src.pipeline import SelfVerificationRAGPipeline
    print("   ✓ Pipeline module imported successfully")
except ImportError as e:
    print(f"   ✗ Pipeline import error: {e}")
    print("   Note: This is OK if dependencies are not installed yet")

# Test 9: Experiment scripts
print("\n9. Testing experiment scripts...")
try:
    import importlib.util
    exp_files = [
        'experiments/exp1_baseline.py',
        'experiments/exp2_retrieval_comparison.py',
        'experiments/exp3_threshold_tuning.py',
        'experiments/exp4_revision_strategies.py',
        'experiments/exp5_decoding_strategies.py',
        'experiments/exp6_iterative_training.py',
        'experiments/exp7_ablation_study.py',
        'experiments/exp8_stress_test.py',
    ]
    for exp_file in exp_files:
        if os.path.exists(exp_file):
            print(f"   ✓ {exp_file} exists")
        else:
            print(f"   ✗ {exp_file} not found")
except Exception as e:
    print(f"   ✗ Error checking experiment files: {e}")

# Test 10: File structure
print("\n10. Testing file structure...")
required_dirs = ['src', 'experiments', 'config', 'notebooks', 'scripts']
for dir_name in required_dirs:
    if os.path.exists(dir_name):
        print(f"   ✓ {dir_name}/ directory exists")
    else:
        print(f"   ✗ {dir_name}/ directory not found")

print("\n" + "=" * 60)
print("Import test complete!")
print("=" * 60)
print("\nNext steps:")
print("1. Install dependencies: pip install -r requirements.txt")
print("2. Download spaCy model: python -m spacy download en_core_web_sm")
print("3. Load your dataset and run experiments")

