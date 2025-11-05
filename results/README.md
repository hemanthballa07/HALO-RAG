# Results Directory

This directory contains the results from all experiments.

## Experiment Results

- `exp1_baseline.json`: Baseline comparison results
- `exp2_retrieval_comparison.json`: Retrieval comparison results
- `exp3_threshold_tuning.json`: Threshold tuning results
- `exp4_revision_strategies.json`: Revision strategies results
- `exp5_decoding_strategies.json`: Decoding strategies results
- `exp6_iterative_training.json`: Iterative training results
- `exp7_ablation_study.json`: Ablation study results
- `exp8_stress_test.json`: Stress test results

## Figures

- `figures/exp3_threshold_curves.png`: Threshold optimization curves
- `figures/exp3_pareto_frontier.png`: Pareto frontier (Factual Precision vs Verified F1)
- `figures/exp6_training_curves.png`: Training curves over iterations

## Summary

- `all_experiments_summary.json`: Summary of all experiments

## Metrics Format

All metrics are reported as:
- Mean ± Standard Deviation
- With 95% confidence intervals
- Statistical significance (p < 0.05)

