# Release checklist

## Before the full experiment run

1. Independently review the nine focused false-accept flags from the three-seed
   CPU diagnostic in `results/human_eval/focused_false_accepts_seed2028_2030.csv`.
   Judge whether each answer addresses the question and is supported by its
   retrieved passage. Keep those judgments separate from the SQuAD v2
   answerability label. The four cases in
   `results/human_eval/focused_false_accepts_current_protocol.csv` and the older
   24-case sheet in `results/human_eval/focused_seed2026_2027_review.csv` are
   separate reviews. Do not combine scores across different protocols. Once
   each sheet is fully labeled, score it as shown in
   `results/human_eval/README.md`.
2. Use confirmed errors to improve verification beyond lexical co-occurrence.
   The three-seed diagnostic flagged nine verified answers on 30
   source-unanswerable focused questions; the flags are not yet confirmed
   evidence failures. These questions have been inspected, so do not treat them
   as fresh holdout data after a fix. Add regression cases from adjudicated
   errors, then validate on new seeds and another dataset.
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

The runner returns a nonzero status if an experiment fails, skips queries, or does
not produce a fresh metrics artifact. It archives each seed's JSON and writes a
manifest. Diagnostic and incomplete runs do not publish final summary files.
Check the manifest before reporting results.

## 4. Complete human evaluation

```bash
python experiments/generate_human_eval_samples.py --num-samples 100 --split validation
python experiments/score_human_eval.py --csv results/human_eval/human_eval_samples.csv
```

The scoring command should be run only after annotators fill the human-label column.

## 5. Record reproducibility information

```bash
python scripts/create_results_lock.py \
  --manifest results/metrics/final_runs/RUN_ID/manifest.json
```

Use the path printed by a completed full runner invocation. The lock command
checks all eight experiments, three or more distinct seeds, each archived JSON,
the configuration, and the published aggregate and summary. It refuses an
incomplete run or an existing `RESULTS_LOCK.md`. The lock covers experiment
artifacts only; human evaluation remains a separate release requirement.

## 6. Release review

- Inspect every generated plot and summary table.
- Confirm that no credentials, model caches, datasets, or checkpoints are staged.
- Re-run `make check` from a clean checkout.
- Document missed metric targets as results, not as implementation failures.
- Tag a release only after the experiment and human-evaluation artifacts are complete.
