# Results

Experiment scripts write scalar metrics to `results/metrics/`, figures to
`results/figures/`, and annotation files to `results/human_eval/`.

Tracked metric artifacts were removed after corrections to dataset parsing, retrieval
score normalization, and NLI label mapping. Those changes affect reported factuality and
retrieval values, so previous outputs are not comparable to current runs.
Claim extraction now verifies each declarative sentence separately, drops questions,
and deduplicates repeated claims. Any results generated before this change also need
to be rerun before comparison with current verification metrics.

SVO claims now retain auxiliary verbs and negation, which also changes verification
inputs. Regenerate historical verification metrics before comparing them with new runs.

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
Each run also records a fingerprint of the benchmark runner and `src/` code.
The paired summary refuses to combine runs with missing or different source
fingerprints. Regenerate older benchmark files before combining them with new
runs, even when their configuration and model names match.
It also counts how many claims each verification method scored and accepted.
For unanswerable cases, `false_accept_claim_method_counts` attributes accepted
claims in falsely verified answers to their scoring method. These are claim
counts, not independent question counts.

## Diagnostic snapshot, 2026-09-30

Three SQuAD v2 validation runs used seeds 42, 123, and 456, with 20 distinct
questions and a 500-passage corpus per seed. All runs used source fingerprint
`9d716906aa3ff3039874b6298cb99507c562360c25df4aaf49c7da20c1a4a17d`.
The focused variant uses one reranked passage and an explicit no-answer prompt.

| Variant | Overall exact match | Answerable exact match | Verified answers on source-unanswerable questions |
| --- | ---: | ---: | ---: |
| Baseline | 19/60 | 19/30 | 14/30 |
| Revision | 32/60 | 20/30 | 18/30 |
| Focused | 41/60 | 24/30 | 4/30 |

The last column is a diagnostic false-accept count based on the source dataset
label. It is not an independent judgment of support in the retrieved passage.
All four focused false accepts used `question_sentence_match`; NLI accepted no
focused claims in this sample. The sample is too small for a production accuracy
claim. The four cases need independent evidence review before further tuning.

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
