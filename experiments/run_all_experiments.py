"""
Run all experiments in sequence.
"""

import sys
from pathlib import Path
import os
import argparse
import yaml
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from experiments.exp1_baseline import run_baseline_experiment
from experiments.exp2_retrieval_comparison import run_retrieval_comparison
from experiments.exp3_threshold_tuning import run_threshold_tuning
from experiments.exp4_revision_strategies import run_revision_strategies_experiment
from experiments.exp5_decoding_strategies import run_decoding_strategies_experiment
from experiments.exp6_iterative_training import run_iterative_training_experiment
from experiments.exp7_ablation_study import run_ablation_study
from experiments.exp8_stress_test import run_stress_test


def load_config(config_path: str = "config/config.yaml"):
    """Load configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="Run all experiments")
    parser.add_argument("--experiments", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 7, 8],
                       help="Experiment numbers to run (default: all)")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                       help="Path to config file")
    parser.add_argument("--data", type=str, required=True,
                       help="Path to dataset file")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Load dataset from args.data
    # queries, ground_truths, relevant_docs, corpus = load_dataset(args.data)
    
    print(f"Starting experiments at {datetime.now()}")
    print(f"Running experiments: {args.experiments}")
    
    results = {}
    
    # Experiment 1: Baseline
    if 1 in args.experiments:
        print("\n" + "="*50)
        print("EXPERIMENT 1: Baseline Comparison")
        print("="*50)
        # results["exp1"] = run_baseline_experiment(
        #     queries, ground_truths, relevant_docs, corpus, config
        # )
    
    # Experiment 2: Retrieval Comparison
    if 2 in args.experiments:
        print("\n" + "="*50)
        print("EXPERIMENT 2: Retrieval Comparison")
        print("="*50)
        # results["exp2"] = run_retrieval_comparison(
        #     queries, relevant_docs, corpus, config
        # )
    
    # Experiment 3: Threshold Tuning
    if 3 in args.experiments:
        print("\n" + "="*50)
        print("EXPERIMENT 3: Threshold Tuning")
        print("="*50)
        # results["exp3"] = run_threshold_tuning(
        #     queries, ground_truths, relevant_docs, corpus, config
        # )
    
    # Experiment 4: Revision Strategies
    if 4 in args.experiments:
        print("\n" + "="*50)
        print("EXPERIMENT 4: Revision Strategies")
        print("="*50)
        # results["exp4"] = run_revision_strategies_experiment(
        #     queries, ground_truths, relevant_docs, corpus, config
        # )
    
    # Experiment 5: Decoding Strategies
    if 5 in args.experiments:
        print("\n" + "="*50)
        print("EXPERIMENT 5: Decoding Strategies")
        print("="*50)
        # results["exp5"] = run_decoding_strategies_experiment(
        #     queries, ground_truths, relevant_docs, corpus, config
        # )
    
    # Experiment 6: Iterative Training
    if 6 in args.experiments:
        print("\n" + "="*50)
        print("EXPERIMENT 6: Iterative Training")
        print("="*50)
        # results["exp6"] = run_iterative_training_experiment(
        #     train_queries, train_contexts, train_answers,
        #     eval_queries, eval_ground_truths, eval_relevant_docs,
        #     corpus, config
        # )
    
    # Experiment 7: Ablation Study
    if 7 in args.experiments:
        print("\n" + "="*50)
        print("EXPERIMENT 7: Ablation Study")
        print("="*50)
        # results["exp7"] = run_ablation_study(
        #     queries, ground_truths, relevant_docs, corpus, config
        # )
    
    # Experiment 8: Stress Test
    if 8 in args.experiments:
        print("\n" + "="*50)
        print("EXPERIMENT 8: Stress Test")
        print("="*50)
        # results["exp8"] = run_stress_test(
        #     adversarial_queries, ground_truths, relevant_docs, corpus, config
        # )
    
    print(f"\n{'='*50}")
    print(f"All experiments completed at {datetime.now()}")
    print(f"{'='*50}")
    
    # Save summary
    os.makedirs("results", exist_ok=True)
    import json
    with open("results/all_experiments_summary.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()

