# Which K to Reduce for Re-Retrieval Evidence?

## Answer: Reduce `top_k_retrieve` (Not `top_k_rerank`)

### Why `top_k_retrieve` is More Important

**`top_k_retrieve`** controls the **initial retrieval from corpus**:
- This is what the re-retrieval strategy can improve upon
- Reducing it makes initial retrieval worse (misses relevant docs)
- Re-retrieval with expanded query can then find different/better documents
- **This directly demonstrates re-retrieval effectiveness**

**`top_k_rerank`** controls **how many documents to keep after reranking**:
- This happens AFTER retrieval (both initial and re-retrieval)
- Reducing it doesn't affect what documents are retrieved
- It just filters the reranked results
- **Less impactful for showing re-retrieval benefit**

## Recommended Settings

### For Demonstrating Re-Retrieval Effectiveness:

```bash
# Reduce top_k_retrieve (the key parameter)
python experiments/exp1_baseline.py --top-k-retrieve 10 --top-k-rerank 5
python experiments/exp9_complete_pipeline.py --top-k-retrieve 10 --top-k-rerank 5
```

**Why this works:**
- Initial retrieval gets only 10 documents (instead of 20)
- More likely to miss relevant documents
- Re-retrieval with expanded query retrieves NEW 10 documents
- Reranker then selects best 5 from the new retrieval
- Clear improvement shows re-retrieval is working

### Even More Aggressive (Stronger Evidence):

```bash
# Very reduced retrieval
python experiments/exp1_baseline.py --top-k-retrieve 5 --top-k-rerank 3
python experiments/exp9_complete_pipeline.py --top-k-retrieve 5 --top-k-rerank 3
```

**Why this works even better:**
- Initial retrieval gets only 5 documents (very constrained)
- High chance of missing relevant documents
- Re-retrieval with expanded query can find completely different documents
- Very clear improvement demonstrates re-retrieval effectiveness

## What Happens in Each Case

### Case 1: Reduce `top_k_retrieve` (Recommended)
```
Baseline:
- Retrieve 10 docs → Rerank → Keep top 5
- Might miss relevant docs (smaller pool)

Re-Retrieval:
- Retrieve 10 NEW docs with expanded query → Rerank → Keep top 5
- Can find different/better docs that were missed initially
- ✅ Clear improvement shows re-retrieval works
```

### Case 2: Reduce `top_k_rerank` Only (Not Recommended)
```
Baseline:
- Retrieve 20 docs → Rerank → Keep top 3
- Still has good pool of 20 docs to choose from

Re-Retrieval:
- Retrieve 20 NEW docs → Rerank → Keep top 3
- Both have same retrieval pool size (20)
- Less clear improvement (reranking already filtered well)
- ❌ Doesn't clearly demonstrate re-retrieval benefit
```

## Summary

**To demonstrate re-retrieval effectiveness:**
1. ✅ **Reduce `top_k_retrieve`** (e.g., 20 → 10 or 5)
2. ✅ Keep `top_k_rerank` reasonable (e.g., 5 or 3)
3. ✅ This makes initial retrieval worse, so re-retrieval improvement is clear

**Example commands:**
```bash
# Moderate reduction (balanced)
--top-k-retrieve 10 --top-k-rerank 5

# Aggressive reduction (strong evidence)
--top-k-retrieve 5 --top-k-rerank 3
```

The `corpus_to_k_ratio` in CSV will show the difficulty:
- k=20: ratio = 250 (easier)
- k=10: ratio = 500 (harder) ← Good for demonstration
- k=5: ratio = 1000 (very hard) ← Strong evidence

