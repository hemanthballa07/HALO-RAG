# Experiments

This directory contains experiment scripts for the HALO-RAG project.

## Experiments

### Experiment 1: Baseline Comparison
**File**: `exp1_baseline.py`

Runs baseline comparison (no verification) to establish baseline metrics.

**Metrics**:
- EM, F1, BLEU-4, ROUGE-L
- Factual Precision, Hallucination Rate
- Verified F1

**Output**:
- `results/metrics/exp1_baseline.json`
- `results/metrics/exp1_baseline.csv`

**Usage**:
```bash
# Full experiment
python experiments/exp1_baseline.py --split validation

# Dry run (30 samples)
python experiments/exp1_baseline.py --dry-run

# Custom limit
python experiments/exp1_baseline.py --limit 100 --split validation
```

### Experiment 2: Retrieval Comparison
**File**: `exp2_retrieval_comparison.py`

Compares different retrieval methods: Dense, Sparse, Hybrid, Hybrid+Rerank.

**Metrics**:
- Recall@5/10/20
- MRR, NDCG@10
- Coverage

**Output**:
- `results/metrics/exp2_retrieval.csv`
- `results/metrics/exp2_retrieval_*.json` (per config)
- `results/figures/exp2_retrieval_bars.png`

**Usage**:
```bash
# Full experiment
python experiments/exp2_retrieval_comparison.py --split validation

# Dry run
python experiments/exp2_retrieval_comparison.py --dry-run
```

### Experiment 3: Threshold Tuning
**File**: `exp3_threshold_tuning.py`

Sweeps entailment threshold τ to find optimal value.

**Metrics**:
- Factual Precision, Factual Recall
- Verified F1, Abstention Rate
- EM, F1

**Output**:
- `results/metrics/exp3_threshold_sweep.csv`
- `results/figures/exp3_verified_f1_vs_tau.png`
- `results/figures/exp3_precision_vs_recall.png`

**Usage**:
```bash
# Full experiment
python experiments/exp3_threshold_tuning.py --split validation

# Dry run
python experiments/exp3_threshold_tuning.py --dry-run
```

### Experiment 5: Self-Consistency Decoding
**File**: `exp5_self_consistency.py`

Compares greedy, beam search, and self-consistency decoding strategies.

**Features**:
- Generate k=5 samples at T=0.7
- Filter by Factual Precision ≥ 0.9
- Aggregate via highest Verified F1
- Compare with greedy and beam search baselines

**Metrics**:
- Hallucination Rate, F1, Verified F1
- Compute cost (×k for self-consistency)

**Output**:
- `results/metrics/exp5_self_consistency.json`
- `results/metrics/exp5_self_consistency.csv`
- `results/figures/exp5_decoding_comparison.png`

**Usage**:
```bash
# Full experiment
python experiments/exp5_self_consistency.py --split validation

# Dry run (20 samples, k=5 each = 100 generations)
python experiments/exp5_self_consistency.py --dry-run
```

**Acceptance Criteria**:
- Hallucination Rate drops ≥15% vs baseline
- Verified F1 increases vs baseline

### Experiment 6: Iterative Fine-Tuning
**File**: `exp6_iterative_training.py`

Collects verified data (FP ≥ 0.85) and fine-tunes FLAN-T5 iteratively.

**Features**:
- Collect verified training data with Factual Precision ≥ 0.85
- Create training triples: (question, top-k passages, verified_answer)
- Fine-tune FLAN-T5 with QLoRA on accept set
- Repeat for 3 iterations (Iter0 baseline → Iter1 → Iter2 → Iter3)
- Track metrics across iterations

**Metrics**:
- Hallucination Rate, Factual Precision, F1, EM, Verified F1, Abstention Rate
- Diversity stats (type-token ratio, avg length)

**Output**:
- `results/metrics/exp6_iterative_training.csv`
- `results/metrics/exp6_iterative_training.json`
- `results/figures/exp6_iteration_curves.png`
- `data/verified/train_iter{N}.jsonl` (verified training triples)
- `checkpoints/exp6_iter{N}/` (saved adapters)

**Usage**:
```bash
# Full experiment (3 iterations)
python experiments/exp6_iterative_training.py --iterations 3

# Dry run (≤100 examples)
python experiments/exp6_iterative_training.py --dry-run

# Custom limit
python experiments/exp6_iterative_training.py --limit 200 --iterations 3
```

**Acceptance Criteria**:
- Hallucination Rate drops toward ≤0.10 by Iter3
- Verified F1 increases each iteration
- Verified pool strictly respects Factual Precision ≥ 0.85
- F1/EM stable or slightly increases

### Human Evaluation
**Files**: `generate_human_eval_samples.py`, `score_human_eval.py`

Generate samples for human evaluation and compute agreement metrics.

**Features**:
- Generate 100 samples for annotation
- Create CSV with columns: id, question, context, generated_answer, gold_answer, auto_label, human_label, notes
- Compute Human–Verifier Agreement (percent match + Cohen's κ)
- Log metrics to JSON and W&B

**Output**:
- `results/human_eval/human_eval_samples.csv` (100 samples for annotation)
- `results/human_eval/README.md` (annotation instructions)
- `results/metrics/human_eval_agreement.json` (agreement metrics)

**Usage**:
```bash
# Generate samples
python experiments/generate_human_eval_samples.py --num-samples 100 --split validation

# Score agreement (after annotation)
python experiments/score_human_eval.py --csv results/human_eval/human_eval_samples.csv
```

**Acceptance Criteria**:
- 100 rows generated; annotators can fill in human_label
- Scorer runs end-to-end and reports agreement ≥ 0.85
- Cohen's κ ≥ 0.70 (substantial agreement)

## CLI Arguments

All experiments support the following CLI arguments:

- `--config`: Path to config file (default: `config/config.yaml`)
- `--split`: Dataset split (`train`, `validation`, `test`) (default: `train`)
- `--limit`: Limit number of examples (default: from config)
- `--seed`: Random seed (default: 42)
- `--dry-run`: Run with 30 samples for quick testing
- `--no-wandb`: Disable W&B logging

## W&B Logging

Experiments log metrics to W&B if available:
- Project: `SelfVerifyRAG`
- Run names: `exp1_baseline`, `exp2_retrieval_comparison`, `exp3_threshold_tuning`

To enable W&B:
1. Install: `pip install wandb`
2. Login: `wandb login`
3. Run experiments (W&B logging enabled by default)

To disable: use `--no-wandb` flag

## Output Structure

```
results/
├── metrics/
│   ├── exp1_baseline.json
│   ├── exp1_baseline.csv
│   ├── exp2_retrieval.csv
│   ├── exp2_retrieval_*.json
│   ├── exp3_threshold_sweep.csv
│   ├── exp3_threshold_tuning.json
│   ├── exp5_self_consistency.json
│   ├── exp5_self_consistency.csv
│   ├── exp6_iterative_training.json
│   └── exp6_iterative_training.csv
└── figures/
    ├── exp2_retrieval_bars.png
    ├── exp3_verified_f1_vs_tau.png
    ├── exp3_precision_vs_recall.png
    ├── exp5_decoding_comparison.png
    └── exp6_iteration_curves.png

data/
└── verified/
    ├── train_iter1.jsonl
    ├── train_iter2.jsonl
    └── train_iter3.jsonl

checkpoints/
└── exp6/
    ├── iter1/
    ├── iter2/
    └── iter3/
```

## Quick Start

1. **Setup**:
   ```bash
   # Install dependencies
   pip install -r requirements.txt
   
   # Download datasets (will be downloaded automatically on first run)
   ```

2. **Run Experiments**:
   ```bash
   # Dry run to test
   python experiments/exp1_baseline.py --dry-run
   python experiments/exp2_retrieval_comparison.py --dry-run
   python experiments/exp3_threshold_tuning.py --dry-run
   python experiments/exp5_self_consistency.py --dry-run
   python experiments/exp6_iterative_training.py --dry-run
   
   # Full experiments
   python experiments/exp1_baseline.py --split validation
   python experiments/exp2_retrieval_comparison.py --split validation
   python experiments/exp3_threshold_tuning.py --split validation
   python experiments/exp5_self_consistency.py --split validation
   python experiments/exp6_iterative_training.py --iterations 3
   ```

3. **Check Results**:
   ```bash
   # View metrics
   cat results/metrics/exp1_baseline.csv
   cat results/metrics/exp5_self_consistency.csv
   
   # View plots
   open results/figures/exp2_retrieval_bars.png
   open results/figures/exp5_decoding_comparison.png
   ```

## Notes

- Experiments use dataset loaders from `src/data/`
- All experiments log commit hash and timestamp
- Metrics are saved locally regardless of W&B availability
- Dry run uses 30 samples for quick testing

