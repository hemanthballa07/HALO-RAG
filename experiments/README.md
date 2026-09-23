# Experiments

This directory contains experiment scripts for the HALO-RAG project.

## Paired question-answering benchmark

`run_representative_benchmark.py` compares revision disabled and enabled on the
same seeded SQuAD v2 validation cases. It builds the retrieval corpus before
choosing questions, includes unrelated passages, and uses distinct source passages
for the selected questions. Half the questions are unanswerable. It saves the
selected question IDs, passage hashes, model names, dependency versions, individual
predictions, and split summary metrics.
It records both initial retrieval/reranking IDs and final evidence IDs after any
revision, so those stages can be evaluated separately.

```bash
python experiments/run_representative_benchmark.py \
  --questions 20 --corpus-size 500 --seed 42 \
  --top-k-retrieve 20 --top-k-rerank 5 --max-revisions 1
```

Output: `results/metrics/representative_benchmark.json`. A small run is a
diagnostic sample, not a release-grade performance estimate. Exact match is
strict; a nonexact answer may still be partially useful, so inspect F1 and
per-question predictions too. The `false_accept_rate` is reported only for
unanswerable questions and counts verified non-abstaining answers;
`unanswerable_answer_rate` counts every nonempty non-abstaining answer.

## Focused no-answer prompt

`evaluate_focused_answers.py` reads a saved paired benchmark, checks that its
question IDs and passage hashes still match the dataset, and runs the generator
against only the top reranked passage. The prompt permits an explicit
`UNANSWERABLE` response. It scores the same references as the paired benchmark.

```bash
python experiments/evaluate_focused_answers.py \
  --benchmark results/metrics/representative_benchmark.json
```

The output is saved beside the source file with `_top1_abstain` added to its name.
Use `--prompt standard` to measure passage selection without changing the
generator prompt.
This trial does not rerun retrieval, verification, or revision. It is an
experimental answer-generation setting, not a substitute for an end-to-end
pipeline comparison.

For an end-to-end comparison on the same sampled corpus and questions, run
`run_representative_benchmark.py` with `--include-focused`. That third variant
uses the top reranked passage and explicit no-answer prompt, verifies generated
answers, and leaves revision disabled. The result includes its per-question
latency and abstention alongside the baseline and revision variants.

To combine compatible runs, pass their result files to
`summarize_paired_benchmarks.py`:

```bash
python experiments/summarize_paired_benchmarks.py \
  results/metrics/benchmark_seed1.json \
  results/metrics/benchmark_seed2.json \
  --output results/metrics/paired_summary.json
```

The summary checks that the dataset, models, configuration, retrieval settings,
and variants match. It reports split scores, paired improvements and harms, and
median and 95th-percentile latency. It also lists repeated question IDs across
runs; repeated questions should not be treated as independent observations.

If label-based errors need review, export the nonexact cases without running
the models again:

```bash
python experiments/export_benchmark_review.py \
  results/metrics/benchmark_seed1.json \
  results/metrics/benchmark_seed2.json \
  --variant focused \
  --output results/human_eval/focused_review.csv
```

The CSV contains both the source passage and the passage used by the pipeline.
The human judgment columns are intentionally blank. The exporter refuses to
overwrite an existing review file.

Once an independent reviewer fills those columns, run
`score_benchmark_review.py` to validate the labels and count supported,
unsupported, irrelevant, and unclear answers. See
`results/human_eval/README.md` for the labels and command. This targeted
nonexact subset cannot estimate factuality across the full benchmark.

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

### Experiment 4: Revision Strategies
**File**: `exp4_revision_strategies.py`

Compares the verified pipeline with revision disabled and enabled, then reports paired
metric comparisons and revision frequency.

**Output**:
- `results/metrics/exp4_revision_strategies.json`

**Usage**:
```bash
python experiments/exp4_revision_strategies.py --split validation
python experiments/exp4_revision_strategies.py --dry-run --no-wandb
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

This experiment requires CUDA and `bitsandbytes`; it intentionally exits before model
loading when those requirements are unavailable.

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

### Experiment 7: Ablation Study
**File**: `exp7_ablation_study.py`

Performs component-wise ablation study to measure impact of each module.

**Ablation Variants**:
- Full system (baseline): Hybrid retrieval + Reranking + NLI verification + Revision
- No reranking: Removes cross-encoder reranking
- No verification: Pure RAG (no verification, no revision)
- No revision: Verification but no adaptive revision
- Simple verifier: Lexical overlap instead of NLI verification

**Metrics**:
- Verified F1, Factual Precision, Hallucination Rate
- EM, F1 Score

**Output**:
- `results/metrics/exp7_ablation.csv`
- `results/metrics/exp7_ablation.json`
- `results/figures/exp7_ablation_bars.png`

**Usage**:
```bash
# Full experiment
python experiments/exp7_ablation_study.py --split validation

# Dry run (50 examples)
python experiments/exp7_ablation_study.py --dry-run

# Custom limit
python experiments/exp7_ablation_study.py --limit 100 --split validation
```

**Acceptance Criteria**:
- Clear ranking of components by impact
- Verified F1 drops show verification > reranking > revision > simple verifier
- All metrics computed for all variants
- Artifacts generated (CSV, JSON, plots)

### Experiment 8: Stress Testing & Pareto Frontier
**File**: `exp8_stress_test.py`

Evaluates robustness and trade-offs between accuracy and factuality through comprehensive stress testing.

**Stress Tests**:
- τ-Sweep (Re-verification): τ ∈ {0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9}
- Retrieval Degradation: Recall@20 ∈ {0.95, 0.85, 0.75, 0.65}
- Verifier Off: Disable verification and measure hallucination rate increase

**Metrics**:
- Factual Precision, Factual Recall, Verified F1
- Hallucination Rate, Abstention Rate
- EM, F1 Score

**Output**:
- `results/metrics/exp8_stress.csv`
- `results/metrics/exp8_stress.json`
- `results/figures/exp8_verified_f1_vs_tau.png`
- `results/figures/exp8_precision_vs_recall.png`
- `results/figures/exp8_pareto_frontier.png`

**Usage**:
```bash
# Full experiment
python experiments/exp8_stress_test.py --split validation

# Dry run (50 examples)
python experiments/exp8_stress_test.py --dry-run

# Custom limit
python experiments/exp8_stress_test.py --limit 100 --split validation
```

**Acceptance Criteria**:
- Verified RAG dominates baseline on Pareto plot (higher EM & factuality)
- Select τ from the measured validation trade-off and report whether Verified F1 reaches 0.52
- Retrieval quality correlates strongly with factual precision
- Artifacts + plots saved and logged (W&B optional)

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

The generator requires the requested sample count and stops if a sample fails.
It will not replace an existing CSV; pass `--output` for a new sheet. The scorer
requires all human labels and will not replace an existing JSON report. Pass
`--output` there too when scoring a new review. If both sides assign one label
to every sample, Cohen's κ is undefined and the report records `null`.

**Acceptance Criteria**:
- 100 rows generated and independently annotated
- Scorer runs end-to-end and reports observed agreement
- Cohen's κ is reported when defined by the observed labels

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
│   ├── exp6_iterative_training.csv
│   ├── exp7_ablation.json
│   ├── exp7_ablation.csv
│   ├── exp8_stress.json
│   ├── exp8_stress.csv
│   └── human_eval_agreement.json
└── figures/
    ├── exp2_retrieval_bars.png
    ├── exp3_verified_f1_vs_tau.png
    ├── exp3_precision_vs_recall.png
    ├── exp5_decoding_comparison.png
    ├── exp6_iteration_curves.png
    ├── exp7_ablation_bars.png
    ├── exp8_verified_f1_vs_tau.png
    ├── exp8_precision_vs_recall.png
    └── exp8_pareto_frontier.png

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
