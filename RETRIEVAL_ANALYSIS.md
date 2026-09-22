# Retrieval analysis protocol

The pipeline first retrieves candidates with fused dense and sparse scores, then applies a
cross-encoder and passes only the highest-ranked evidence to the generator.

## Questions to measure

- How does retrieval depth affect Recall@K and answer-token coverage?
- Does reranking improve the rank of the gold context?
- How much latency does each additional candidate add?
- When revision re-retrieves evidence, does the final evidence set improve support for the
  generated claims?

## Controlled sweep

Evaluate the same query set and random seeds across this grid:

```text
top_k_retrieve: 5, 10, 20
top_k_rerank:   3, 5
```

Reject combinations where reranking depth exceeds retrieval depth. Report retrieval
recall, coverage, factual precision, hallucination rate, answer F1, and elapsed time. Use
paired tests because every configuration is evaluated on the same queries.

The configured default is `20 → 5`. It is a starting point, not a claimed optimum. Select
the operating point on validation data and reserve test data for the final comparison.
