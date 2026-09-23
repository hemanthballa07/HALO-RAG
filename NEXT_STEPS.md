# Release checklist

## Before the full experiment run

1. Review the 24 nonexact focused answers in
   `results/human_eval/focused_seed2026_2027_review.csv`. Judge whether each
   answer addresses the question and is supported by its retrieved passage.
   Keep those judgments separate from the SQuAD v2 answerability label. Once
   every row is labeled, run `experiments/score_benchmark_review.py` as shown in
   `results/human_eval/README.md`. Do not treat its targeted counts as rates
   across the full benchmark.
2. Use confirmed unsupported answers to improve verification beyond lexical
   co-occurrence. The two larger runs still show 25% label-based false
   acceptance for focused mode. Add regression cases from adjudicated errors,
   then evaluate on new seeds and another dataset without tuning to test cases.
3. Repeat the experiment matrix on the target CUDA host after choosing a
   setting. The CPU runs are useful checks, not release results.

## 1. Validate the environment

```bash
python scripts/check_setup.py
make check
```

For QLoRA training, install `requirements-gpu.txt` on a compatible CUDA host and confirm
that `bitsandbytes` imports successfully.

## 2. Run a smoke experiment

```bash
python experiments/exp1_baseline.py --limit 10 --no-wandb
```

Confirm that the model and dataset caches have sufficient free space before starting the
full matrix.

## 3. Run the experiment matrix

```bash
python experiments/run_final_experiments.py \
  --seeds 42 123 456 \
  --split validation \
  --copy-plots
```

The runner now returns a nonzero status if an experiment fails or does not produce a
metrics artifact. Do not publish a partial summary as a complete run.

## 4. Complete human evaluation

```bash
python experiments/generate_human_eval_samples.py --num-samples 100 --split validation
python experiments/score_human_eval.py --csv results/human_eval/human_eval_samples.csv
```

The scoring command should be run only after annotators fill the human-label column.

## 5. Record reproducibility information

```bash
python scripts/create_results_lock.py \
  --tau 0.75 \
  --seeds 42 123 456 \
  --dataset squad_v2 \
  --split validation
```

Review the resulting lock document and ensure it contains the actual commit, configuration,
dataset split, seed list, and timestamps from the completed run.

## 6. Release review

- Inspect every generated plot and summary table.
- Confirm that no credentials, model caches, datasets, or checkpoints are staged.
- Re-run `make check` from a clean checkout.
- Document missed metric targets as results, not as implementation failures.
- Tag a release only after the experiment and human-evaluation artifacts are complete.
