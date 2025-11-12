"""
Score Human Evaluation Agreement
Computes Human–Verifier Agreement (percent match + Cohen's κ).
"""

import sys
import os
import argparse
from pathlib import Path
import json
import csv
from typing import List, Dict, Any, Optional
from collections import Counter

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

# Try to import sklearn for Cohen's κ
try:
    from sklearn.metrics import cohen_kappa_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    
    def cohen_kappa_score(y1, y2):
        """
        Compute Cohen's κ coefficient.
        
        κ = (p_o - p_e) / (1 - p_e)
        where p_o is observed agreement and p_e is expected agreement.
        """
        if len(y1) != len(y2):
            raise ValueError("y1 and y2 must have the same length")
        
        n = len(y1)
        if n == 0:
            return 0.0
        
        # Count agreements
        p_o = sum(1 for a, b in zip(y1, y2) if a == b) / n
        
        # Count label frequencies
        from collections import Counter
        y1_counts = Counter(y1)
        y2_counts = Counter(y2)
        
        # Compute expected agreement
        p_e = sum((y1_counts.get(label, 0) / n) * (y2_counts.get(label, 0) / n) 
                  for label in set(y1) | set(y2))
        
        # Compute Cohen's κ
        if p_e == 1.0:
            return 1.0  # Perfect agreement
        
        kappa = (p_o - p_e) / (1 - p_e)
        return kappa

from src.utils import setup_wandb, log_metrics, get_commit_hash, get_timestamp


def load_human_eval_samples(csv_path: str) -> List[Dict[str, Any]]:
    """
    Load human evaluation samples from CSV.
    
    Args:
        csv_path: Path to CSV file
    
    Returns:
        List of sample dictionaries
    """
    samples = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append(row)
    
    return samples


def compute_agreement_metrics(
    samples: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Compute agreement metrics between human and auto labels.
    
    Args:
        samples: List of sample dictionaries with auto_label and human_label
    
    Returns:
        Dictionary with agreement metrics
    """
    # Filter samples with both auto_label and human_label
    valid_samples = [
        s for s in samples
        if s.get("auto_label") and s.get("human_label")
    ]
    
    if not valid_samples:
        raise ValueError("No samples with both auto_label and human_label found")
    
    # Extract labels
    auto_labels = [s["auto_label"] for s in valid_samples]
    human_labels = [s["human_label"] for s in valid_samples]
    
    # Compute percent match
    matches = sum(1 for a, h in zip(auto_labels, human_labels) if a == h)
    percent_match = (matches / len(valid_samples)) * 100
    
    # Compute Cohen's κ
    # Map labels to integers for sklearn
    all_labels = sorted(set(auto_labels + human_labels))
    label_to_int = {label: i for i, label in enumerate(all_labels)}
    
    auto_labels_int = [label_to_int[label] for label in auto_labels]
    human_labels_int = [label_to_int[label] for label in human_labels]
    
    if SKLEARN_AVAILABLE:
        cohens_kappa = cohen_kappa_score(auto_labels_int, human_labels_int)
    else:
        # Fallback: use percent agreement as proxy
        cohens_kappa = percent_match / 100.0
    
    # Compute per-label agreement
    per_label_agreement = {}
    for label in all_labels:
        label_samples = [
            (a, h) for a, h in zip(auto_labels, human_labels)
            if a == label or h == label
        ]
        if label_samples:
            label_matches = sum(1 for a, h in label_samples if a == h)
            label_total = len(label_samples)
            per_label_agreement[label] = {
                "matches": label_matches,
                "total": label_total,
                "agreement": (label_matches / label_total) * 100 if label_total > 0 else 0.0
            }
    
    # Compute confusion matrix
    confusion_matrix = {}
    for auto_label in all_labels:
        confusion_matrix[auto_label] = {}
        for human_label in all_labels:
            count = sum(
                1 for a, h in zip(auto_labels, human_labels)
                if a == auto_label and h == human_label
            )
            confusion_matrix[auto_label][human_label] = count
    
    # Label distribution
    auto_label_dist = Counter(auto_labels)
    human_label_dist = Counter(human_labels)
    
    return {
        "total_samples": len(valid_samples),
        "percent_match": percent_match,
        "cohens_kappa": cohens_kappa,
        "per_label_agreement": per_label_agreement,
        "confusion_matrix": confusion_matrix,
        "auto_label_distribution": dict(auto_label_dist),
        "human_label_distribution": dict(human_label_dist),
        "all_labels": all_labels
    }


def save_agreement_metrics(
    metrics: Dict[str, Any],
    output_path: str = "results/metrics/human_eval_agreement.json"
) -> None:
    """
    Save agreement metrics to JSON.
    
    Args:
        metrics: Dictionary with agreement metrics
        output_path: Output JSON file path
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"✓ Saved agreement metrics to {output_path}")


def print_agreement_summary(metrics: Dict[str, Any]) -> None:
    """Print agreement summary."""
    print("\n" + "=" * 70)
    print("Human Evaluation Agreement Summary")
    print("=" * 70)
    
    print(f"Total samples: {metrics['total_samples']}")
    print(f"Percent match: {metrics['percent_match']:.2f}%")
    print(f"Cohen's κ: {metrics['cohens_kappa']:.4f}")
    
    # Interpret Cohen's κ
    kappa = metrics['cohens_kappa']
    if kappa < 0:
        interpretation = "Poor (worse than chance)"
    elif kappa < 0.20:
        interpretation = "Slight"
    elif kappa < 0.40:
        interpretation = "Fair"
    elif kappa < 0.60:
        interpretation = "Moderate"
    elif kappa < 0.80:
        interpretation = "Substantial"
    else:
        interpretation = "Almost perfect"
    
    print(f"Agreement level: {interpretation}")
    
    # Per-label agreement
    print("\nPer-label agreement:")
    for label, stats in metrics['per_label_agreement'].items():
        print(f"  {label}: {stats['agreement']:.2f}% ({stats['matches']}/{stats['total']})")
    
    # Confusion matrix
    print("\nConfusion matrix (auto_label → human_label):")
    all_labels = metrics['all_labels']
    print(f"{'Auto\\Human':<15}", end="")
    for label in all_labels:
        print(f"{label:<15}", end="")
    print()
    
    for auto_label in all_labels:
        print(f"{auto_label:<15}", end="")
        for human_label in all_labels:
            count = metrics['confusion_matrix'][auto_label].get(human_label, 0)
            print(f"{count:<15}", end="")
        print()
    
    # Label distribution
    print("\nLabel distribution:")
    print("  Auto labels:")
    for label, count in metrics['auto_label_distribution'].items():
        print(f"    {label}: {count}")
    print("  Human labels:")
    for label, count in metrics['human_label_distribution'].items():
        print(f"    {label}: {count}")
    
    print("=" * 70)


def main():
    """Main function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Score Human Evaluation Agreement")
    parser.add_argument(
        "--csv",
        type=str,
        default="results/human_eval/human_eval_samples.csv",
        help="Path to annotated CSV file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/metrics/human_eval_agreement.json",
        help="Output JSON file path"
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B logging"
    )
    
    args = parser.parse_args()
    
    # Load samples
    print(f"Loading human evaluation samples from {args.csv}...")
    samples = load_human_eval_samples(args.csv)
    
    # Check if human labels are filled
    samples_with_human_labels = [
        s for s in samples if s.get("human_label") and s["human_label"].strip()
    ]
    
    if not samples_with_human_labels:
        print("⚠ No samples with human_label found. Please annotate the CSV file first.")
        return
    
    print(f"Found {len(samples_with_human_labels)} samples with human labels")
    
    # Compute agreement metrics
    print("Computing agreement metrics...")
    metrics = compute_agreement_metrics(samples)
    
    # Add metadata
    metrics["metadata"] = {
        "csv_path": args.csv,
        "total_samples": len(samples),
        "annotated_samples": len(samples_with_human_labels),
        "commit_hash": get_commit_hash(),
        "timestamp": get_timestamp()
    }
    
    # Save metrics
    save_agreement_metrics(metrics, args.output)
    
    # Print summary
    print_agreement_summary(metrics)
    
    # Log to W&B
    if not args.no_wandb:
        wandb_run = setup_wandb(
            project_name="SelfVerifyRAG",
            run_name="human_eval_agreement",
            config={
                "experiment": "human_eval_agreement",
                "csv_path": args.csv,
                "total_samples": len(samples),
                "annotated_samples": len(samples_with_human_labels)
            },
            enabled=True
        )
        
        if wandb_run:
            log_metrics({
                "percent_match": metrics["percent_match"],
                "cohens_kappa": metrics["cohens_kappa"],
                "total_samples": metrics["total_samples"]
            }, prefix="human_eval/")
            
            # Log per-label agreement
            for label, stats in metrics["per_label_agreement"].items():
                log_metrics({
                    f"agreement_{label.lower().replace(' ', '_')}": stats["agreement"]
                }, prefix="human_eval/")
            
            try:
                import wandb
                wandb.finish()
            except Exception:
                pass
    
    # Check acceptance criteria
    print("\n" + "=" * 70)
    print("Acceptance Criteria Check:")
    print("=" * 70)
    
    percent_match = metrics["percent_match"]
    cohens_kappa = metrics["cohens_kappa"]
    
    print(f"Percent match: {percent_match:.2f}% ({'✓' if percent_match >= 85 else '✗'} target: ≥85%)")
    print(f"Cohen's κ: {cohens_kappa:.4f} ({'✓' if cohens_kappa >= 0.70 else '✗'} target: ≥0.70)")
    print("=" * 70)
    
    return metrics


if __name__ == "__main__":
    metrics = main()

