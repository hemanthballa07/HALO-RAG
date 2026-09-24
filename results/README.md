# Results

Experiment scripts write scalar metrics to `results/metrics/`, figures to
`results/figures/`, and annotation files to `results/human_eval/`.

Tracked metric artifacts were removed after corrections to dataset parsing, retrieval
score normalization, and NLI label mapping. Those changes affect reported factuality and
retrieval values, so previous outputs are not comparable to current runs.
Claim extraction now verifies each declarative sentence separately, drops questions,
and deduplicates repeated claims. Any results generated before this change also need
to be rerun before comparison with current verification metrics.

Generate a fresh smoke result with:

```bash
python experiments/exp1_baseline.py --limit 10 --no-wandb
```

For a paired evaluation with a seeded distractor corpus and a balanced
answerable/unanswerable sample:

```bash
python experiments/run_representative_benchmark.py \
  --questions 20 --corpus-size 500 --seed 42
```

The JSON output includes individual cases, source passages, final evidence,
claim-level verification scores, and corpus fingerprints so failures can be
audited. Small samples are useful for finding failures but should not be
presented as validated system performance.

Short-answer verification still has an answerability limitation. Its lexical
shortcut can accept an answer that appears near question terms even when the
passage describes a different subject or action. For example, a passage saying
the British captured a fort does not support a question asking which fort they
surrendered. Treat a high verified rate as factuality evidence, not proof that
unanswerable questions are handled correctly; inspect the unanswerable false
accept rate and saved evidence separately.

Generate the multi-seed summary with:

```bash
python experiments/run_final_experiments.py --seeds 42 123 456 --split validation
```

The runner creates `results/metrics/final_summary.csv` and
`results/metrics/final_aggregated_results.json`. A successful multi-seed summary should
report `n=3` per metric. Use `--copy-plots` to publish the six key figures only when
each seed has produced fresh metrics and its associated plot. Existing figures from
earlier runs do not satisfy this check. `--skip-runs` is no longer supported because
single experiment files cannot establish per-seed provenance.
Archived metrics must also record the configured dataset and the effective sample
limit for that run. A mismatch leaves the run incomplete.

The root `final_run_results.zip` is a pre-correction historical archive. It remains in the
repository for provenance, but its metrics should not be cited as current HALO-RAG results.
