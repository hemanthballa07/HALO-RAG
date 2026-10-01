# Experiment 7: Ablation study

Experiment 7 measures the contribution of reranking, verification, revision, and the NLI
verifier. The current repository does not include a post-correction result artifact, so
this document describes the protocol rather than reporting findings.

## Variants

- `full`: hybrid retrieval, reranking, NLI verification, and revision
- `no_reranking`: hybrid retrieval, NLI verification, and revision
- `no_verification`: retrieval and generation without verification or revision
- `no_revision`: verification enabled with revision disabled
- `simple_verifier`: lexical overlap in place of the NLI verifier

## Metrics

- Exact Match and token F1
- Factual precision
- Hallucination rate
- Verified F1 (`answer F1 × factual precision`)

## Run

```bash
python experiments/exp7_ablation_study.py --split validation --no-wandb
```

Use `--dry-run` or `--limit 50` for a smoke run. The script writes JSON and CSV metrics
under `results/metrics/` and a comparison plot under `results/figures/`.

## Interpretation

Compare each variant with `full` on the same examples and seed. Report absolute and
relative changes with paired significance tests. Treat expected component rankings as
hypotheses; do not label them as findings until the generated artifacts have been
reviewed.
