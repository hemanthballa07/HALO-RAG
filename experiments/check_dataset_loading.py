#!/usr/bin/env python3
"""
Dataset Loading Verification Script
Checks that datasets load correctly and match the unified schema.
"""

import sys
import os
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
from src.data import load_dataset, normalize_text, validate_example

def load_config(config_path: str = "config/config.yaml"):
    """Load configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def check_dataset_loading():
    """Check dataset loading and schema."""
    print("=" * 70)
    print("Dataset Loading Verification")
    print("=" * 70)
    
    # Load config
    config = load_config()
    
    # Get dataset configuration
    dataset_config = config.get("datasets", {})
    active_dataset = dataset_config.get("active", "squad_v2")
    sample_limit = dataset_config.get("sample_limit")
    
    # Get cache directory
    paths_config = config.get("paths", {})
    cache_dir = paths_config.get("cache_dir", None)
    if cache_dir and cache_dir.startswith("~"):
        cache_dir = os.path.expanduser(cache_dir)
    
    print(f"\nDataset: {active_dataset}")
    print(f"Sample limit: {sample_limit if sample_limit else 'None (load all)'}")
    print(f"Cache directory: {cache_dir if cache_dir else 'Default'}")
    
    # Load dataset
    try:
        print(f"\nLoading {active_dataset}...")
        examples = load_dataset(
            dataset_name=active_dataset,
            split="train",
            limit=sample_limit,
            cache_dir=cache_dir
        )
        print(f"✓ Successfully loaded {len(examples)} examples")
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Check schema
    print("\n" + "=" * 70)
    print("Schema Validation")
    print("=" * 70)
    
    required_keys = ["id", "question", "context", "answers"]
    schema_errors = []
    
    for i, example in enumerate(examples[:10]):  # Check first 10 examples
        for key in required_keys:
            if key not in example:
                schema_errors.append(f"Example {i}: Missing key '{key}'")
        
        # Check types
        if not isinstance(example["id"], str):
            schema_errors.append(f"Example {i}: 'id' should be str, got {type(example['id'])}")
        if not isinstance(example["question"], str):
            schema_errors.append(f"Example {i}: 'question' should be str, got {type(example['question'])}")
        if not isinstance(example["context"], str):
            schema_errors.append(f"Example {i}: 'context' should be str, got {type(example['context'])}")
        if not isinstance(example["answers"], list):
            schema_errors.append(f"Example {i}: 'answers' should be list, got {type(example['answers'])}")
    
    if schema_errors:
        print("✗ Schema validation failed:")
        for error in schema_errors:
            print(f"  - {error}")
        return False
    else:
        print("✓ Schema validation passed")
    
    # Display first 3 examples
    print("\n" + "=" * 70)
    print("Sample Examples (first 3)")
    print("=" * 70)
    
    for i, example in enumerate(examples[:3]):
        print(f"\nExample {i + 1}:")
        print(f"  ID: {example['id']}")
        print(f"  Question: {example['question'][:100]}..." if len(example['question']) > 100 else f"  Question: {example['question']}")
        print(f"  Context: {example['context'][:100]}..." if len(example['context']) > 100 else f"  Context: {example['context']}")
        print(f"  Answers: {example['answers']}")
        print(f"  Answerable: {'Yes' if example['answers'] else 'No'}")
    
    # Statistics
    print("\n" + "=" * 70)
    print("Dataset Statistics")
    print("=" * 70)
    
    total_examples = len(examples)
    answerable_count = sum(1 for ex in examples if ex["answers"])
    unanswerable_count = total_examples - answerable_count
    
    print(f"Total examples: {total_examples}")
    print(f"Answerable: {answerable_count} ({answerable_count / total_examples * 100:.2f}%)")
    print(f"Unanswerable: {unanswerable_count} ({unanswerable_count / total_examples * 100:.2f}%)")
    
    # Check non-empty ratio
    non_empty_ratio = answerable_count / total_examples if total_examples > 0 else 0
    print(f"Non-empty answer ratio: {non_empty_ratio:.2%}")
    
    if non_empty_ratio < 0.95 and active_dataset == "squad_v2":
        # SQuAD v2 has unanswerable questions, so this is expected
        print("  Note: SQuAD v2 includes unanswerable questions, so ratio < 95% is expected")
    elif non_empty_ratio < 0.95:
        print(f"  Warning: Non-empty answer ratio is below 95%")
    
    # Average lengths
    avg_question_len = sum(len(ex["question"].split()) for ex in examples) / total_examples
    avg_context_len = sum(len(ex["context"].split()) for ex in examples) / total_examples
    
    print(f"\nAverage question length: {avg_question_len:.1f} words")
    print(f"Average context length: {avg_context_len:.1f} words")
    
    # Save preview
    print("\n" + "=" * 70)
    print("Saving Preview")
    print("=" * 70)
    
    preview_data = {
        "dataset": active_dataset,
        "total_examples": total_examples,
        "answerable_count": answerable_count,
        "unanswerable_count": unanswerable_count,
        "non_empty_ratio": non_empty_ratio,
        "sample_examples": examples[:3]
    }
    
    preview_path = Path("data/sample_preview.json")
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(preview_path, 'w') as f:
        json.dump(preview_data, f, indent=2)
    
    print(f"✓ Saved preview to {preview_path}")
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("✓ Dataset loaded successfully")
    print("✓ Schema validation passed")
    print(f"✓ Loaded {total_examples} examples")
    print(f"✓ Non-empty answer ratio: {non_empty_ratio:.2%}")
    print("✓ Preview saved to data/sample_preview.json")
    print("\n" + "=" * 70)
    
    return True


if __name__ == "__main__":
    success = check_dataset_loading()
    sys.exit(0 if success else 1)

