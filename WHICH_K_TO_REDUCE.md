# Choosing retrieval and reranking depth

`top_k_retrieve` controls how many documents enter the reranker.
`top_k_rerank` controls how many reranked documents are supplied to generation.

Always keep `top_k_rerank <= top_k_retrieve`. Select both values on a validation split,
then report the chosen values unchanged on the test split. Do not lower retrieval depth
solely to make a revision strategy appear stronger.

Recommended evaluation grid:

```text
top_k_retrieve: 5, 10, 20
top_k_rerank:   3, 5
```

For every valid pair, compare retrieval recall, coverage, latency, generation quality,
and factuality. A smaller retrieval pool is cheaper but can exclude the gold passage; a
larger pool improves candidate recall but increases encoding and reranking cost.

Both Experiment 1 and Experiment 9 expose the settings directly:

```bash
python experiments/exp1_baseline.py --top-k-retrieve 10 --top-k-rerank 5
python experiments/exp9_complete_pipeline.py --top-k-retrieve 10 --top-k-rerank 5
```
