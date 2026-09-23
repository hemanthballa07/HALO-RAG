# Release checklist

## Before the full experiment run

1. Make verification check whether the evidence answers the question, not just
   whether the answer text appears somewhere in a passage. Add negative cases for
   a matching name or location in an unrelated sentence, then rerun the paired
   benchmark to measure false acceptance and lost correct answers.
2. Compare the optional top-passage no-answer setting through the full pipeline
   on an untouched sample. Report answerable and unanswerable results separately,
   along with latency and abstention. Keep the three development seeds out of
   the held-out comparison.
3. After selecting a setting, run the release checks below on the target CUDA
   environment. Do not treat the current CPU sample as a release result.

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
