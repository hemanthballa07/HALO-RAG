#!/usr/bin/env python3
"""
Example: How to use dataset loading in experiments
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
from src.data import load_dataset_from_config, prepare_for_experiments

# Load config
with open("config/config.yaml", 'r') as f:
    config = yaml.safe_load(f)

# Load dataset from config
examples = load_dataset_from_config(config, split="train")

# Prepare for experiments
queries, ground_truths, relevant_docs, corpus = prepare_for_experiments(examples)

print(f"Loaded {len(examples)} examples")
print(f"Queries: {len(queries)}")
print(f"Ground truths: {len(ground_truths)}")
print(f"Corpus size: {len(corpus)}")
print(f"\nFirst query: {queries[0]}")
print(f"First ground truth: {ground_truths[0]}")

