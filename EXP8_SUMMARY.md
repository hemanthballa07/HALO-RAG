# Experiment 8: Stress testing and Pareto analysis

Experiment 8 evaluates how verification thresholds, retrieval degradation, and disabling
the verifier affect answer quality and factual support. The current repository does not
include a post-correction result artifact, so no numerical conclusions are asserted here.

## Stress tests

1. Sweep the entailment threshold from 0.50 to 0.90.
2. Degrade the retrieval set and measure downstream factuality.
3. Disable verification to establish the pure-RAG comparison.

## Metrics

- Exact Match and token F1
- Factual precision and factual recall
- Hallucination and abstention rates
- Verified F1 (`answer F1 × factual precision`)
- Retrieval recall and coverage

## Run

```bash
python experiments/exp8_stress_test.py --split validation --no-wandb
```

Use `--dry-run` or `--limit 50` for a smoke run. Expected outputs include
`results/metrics/exp8_stress.json`, the stress-test CSV, threshold plots, and a Pareto
frontier under `results/figures/`.

## Interpretation

Choose an operating threshold from measured trade-offs rather than assuming that the
configured value is optimal. A point is Pareto-efficient only if no observed alternative
improves one objective without worsening another. Retrieval/factuality correlations and
verifier effects should be reported with their sample sizes and uncertainty.
