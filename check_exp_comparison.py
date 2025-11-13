#!/usr/bin/env python3
"""
Check if exp1 and exp9 used the same data in the same order for one-to-one comparison.
"""

import json
import sys
from pathlib import Path

def load_json_results(filepath):
    """Load JSON results file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def compare_experiments(exp1_path, exp9_path):
    """Compare exp1 and exp9 results to check if they used the same data in the same order."""
    
    print("Loading results...")
    exp1_results = load_json_results(exp1_path)
    exp9_results = load_json_results(exp9_path)
    
    # Get individual results
    exp1_individual = exp1_results.get("individual_results", [])
    exp9_individual = exp9_results.get("individual_results", [])
    
    # Get metadata
    exp1_meta = exp1_results.get("metadata", {})
    exp9_meta = exp9_results.get("metadata", {})
    
    print("\n" + "="*70)
    print("METADATA COMPARISON")
    print("="*70)
    print(f"Exp1 - Dataset: {exp1_meta.get('dataset')}, Split: {exp1_meta.get('split')}, "
          f"Limit: {exp1_meta.get('sample_limit')}, Seed: {exp1_meta.get('seed')}")
    print(f"Exp9 - Dataset: {exp9_meta.get('dataset')}, Split: {exp9_meta.get('split')}, "
          f"Limit: {exp9_meta.get('sample_limit')}, Seed: {exp9_meta.get('seed')}")
    
    # Check if metadata matches
    metadata_match = (
        exp1_meta.get('dataset') == exp9_meta.get('dataset') and
        exp1_meta.get('split') == exp9_meta.get('split') and
        exp1_meta.get('sample_limit') == exp9_meta.get('sample_limit') and
        exp1_meta.get('seed') == exp9_meta.get('seed')
    )
    
    print(f"\nMetadata Match: {'✓ YES' if metadata_match else '✗ NO'}")
    
    # Compare number of results
    print(f"\nNumber of results: Exp1={len(exp1_individual)}, Exp9={len(exp9_individual)}")
    
    if len(exp1_individual) != len(exp9_individual):
        print("⚠ WARNING: Different number of results! Cannot do one-to-one comparison.")
        return False
    
    # Compare queries in order
    print("\n" + "="*70)
    print("QUERY ORDER COMPARISON")
    print("="*70)
    
    mismatches = []
    for idx, (exp1_item, exp9_item) in enumerate(zip(exp1_individual, exp9_individual)):
        exp1_query = exp1_item.get("query", "")
        exp9_query = exp9_item.get("query", "")
        
        if exp1_query != exp9_query:
            mismatches.append({
                "index": idx,
                "exp1_query": exp1_query[:100],
                "exp9_query": exp9_query[:100]
            })
    
    if mismatches:
        print(f"✗ Found {len(mismatches)} query mismatches:")
        for mm in mismatches[:10]:  # Show first 10
            print(f"  Index {mm['index']}:")
            print(f"    Exp1: {mm['exp1_query']}")
            print(f"    Exp9: {mm['exp9_query']}")
        if len(mismatches) > 10:
            print(f"  ... and {len(mismatches) - 10} more")
        return False
    else:
        print("✓ All queries match in order!")
    
    # Compare ground truths in order
    print("\n" + "="*70)
    print("GROUND TRUTH ORDER COMPARISON")
    print("="*70)
    
    gt_mismatches = []
    for idx, (exp1_item, exp9_item) in enumerate(zip(exp1_individual, exp9_individual)):
        exp1_gt = exp1_item.get("ground_truth", "")
        exp9_gt = exp9_item.get("ground_truth", "")
        
        if exp1_gt != exp9_gt:
            gt_mismatches.append({
                "index": idx,
                "query": exp1_item.get("query", "")[:50],
                "exp1_gt": exp1_gt[:100],
                "exp9_gt": exp9_gt[:100]
            })
    
    if gt_mismatches:
        print(f"✗ Found {len(gt_mismatches)} ground truth mismatches:")
        for mm in gt_mismatches[:10]:  # Show first 10
            print(f"  Index {mm['index']} (Query: {mm['query']}...):")
            print(f"    Exp1 GT: {mm['exp1_gt']}")
            print(f"    Exp9 GT: {mm['exp9_gt']}")
        if len(gt_mismatches) > 10:
            print(f"  ... and {len(gt_mismatches) - 10} more")
        return False
    else:
        print("✓ All ground truths match in order!")
    
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print("✓ Exp1 and Exp9 used the same data in the same order!")
    print("  You can safely do one-to-one comparison between results.")
    return True

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python check_exp_comparison.py <exp1_json_path> <exp9_json_path>")
        print("Example: python check_exp_comparison.py results/metrics/exp1_baseline.json results/metrics/exp9_complete_pipeline.json")
        sys.exit(1)
    
    exp1_path = sys.argv[1]
    exp9_path = sys.argv[2]
    
    if not Path(exp1_path).exists():
        print(f"Error: {exp1_path} not found")
        sys.exit(1)
    
    if not Path(exp9_path).exists():
        print(f"Error: {exp9_path} not found")
        sys.exit(1)
    
    success = compare_experiments(exp1_path, exp9_path)
    sys.exit(0 if success else 1)

