# Retrieval Analysis: Top-K Impact on Hallucination and Re-Retrieval Effectiveness

## Current Settings

**Baseline (Exp1 & Exp9):**
- `top_k_retrieve`: **20** documents
- `top_k_rerank`: **5** documents (used for generation)
- Corpus size: **~5000 samples** (limited dataset)

**Re-Retrieval Strategy:**
- Uses `top_k=20` for re-retrieval (same as baseline)
- Expands query with failed claims

## Problem Analysis

### Why Hallucination Rate is Low in Baseline

With a **limited corpus (~5000 samples)** and `top_k_retrieve=20`:
1. **High recall**: Retrieving 20 docs from 5000 means you're likely getting most/all relevant documents
2. **Good context coverage**: Even if initial retrieval isn't perfect, reranking to top 5 often includes the answer
3. **Low hallucination**: Generator has sufficient context, so claims are more likely to be entailed

### Why Most Strategies are Re-Retrieval

1. **Low entailment rate triggers re-retrieval**: When `entailment_rate < 0.5`, re-retrieval is selected
2. **Limited corpus impact**: With only 5000 docs, re-retrieval with expanded query might retrieve similar docs
3. **Strategy selection logic**:
   - `entailment_rate < 0.5` → Re-retrieval
   - `entailment_rate < 0.8` → Constrained generation
   - Otherwise → Claim-by-claim

## Hypothesis: Reducing Top-K

**Proposed Experiment:**
- Reduce `top_k_retrieve` from **20 → 5 or 10**
- Keep `top_k_rerank` at **5** (or reduce to 3)

**Expected Effects:**

### Baseline (Exp1) - Higher Hallucination
- **Lower recall**: Retrieving only 5-10 docs from 5000 means missing relevant documents
- **Poorer context**: Generator has less information, more likely to hallucinate
- **Higher hallucination rate**: Claims not supported by limited context

### Re-Retrieval Strategy (Exp9) - More Effective
- **Room for improvement**: With initial k=5, re-retrieval with expanded query can find different/better docs
- **Better query expansion**: Failed claims in expanded query help retrieve missed relevant docs
- **Demonstrates effectiveness**: Clear improvement over baseline shows re-retrieval works

## Recommended Experiment Settings

### Option 1: Aggressive Reduction (Stress Test)
```yaml
top_k_retrieve: 5   # Down from 20
top_k_rerank: 3     # Down from 5
```
- **Pros**: Maximum stress test, clear difference
- **Cons**: Might be too aggressive, very low baseline performance

### Option 2: Moderate Reduction (Balanced)
```yaml
top_k_retrieve: 10  # Down from 20
top_k_rerank: 5     # Keep at 5
```
- **Pros**: More realistic, still shows re-retrieval benefit
- **Cons**: Less dramatic difference

### Option 3: Gradual Reduction (Sweep)
Test multiple values: `top_k_retrieve = [5, 10, 15, 20]`
- **Pros**: Shows relationship between k and hallucination
- **Cons**: More experiments to run

## Implementation

To test this, you can:

1. **Modify config.yaml**:
```yaml
retrieval:
  fusion:
    top_k: 10  # Reduced from 20
  reranker:
    top_k: 5   # Keep at 5 or reduce to 3
```

2. **Or pass as arguments** (if supported):
```bash
python experiments/exp1_baseline.py --top-k-retrieve 10 --top-k-rerank 5
python experiments/exp9_complete_pipeline.py --top-k-retrieve 10 --top-k-rerank 5
```

3. **Update re-retrieval strategy** to use same reduced k:
   - Currently: `retrieval_fn(expanded_query, top_k=20)`
   - Should match baseline: `retrieval_fn(expanded_query, top_k=10)`

## Expected Metrics

### Baseline (k=10 vs k=20)
- **Hallucination rate**: ↑ (higher, worse)
- **Factual precision**: ↓ (lower, worse)
- **Recall@10**: Similar (same k)
- **Coverage**: ↓ (less context)

### Re-Retrieval (k=10)
- **Hallucination rate**: ↓ (lower than baseline k=10, better)
- **Factual precision**: ↑ (higher than baseline k=10)
- **Revision success rate**: ↑ (more effective)
- **Strategy distribution**: More re-retrieval successes

## Conclusion

**Yes, reducing top-k is a good idea** to:
1. ✅ Create a more challenging baseline (higher hallucination)
2. ✅ Demonstrate re-retrieval effectiveness more clearly
3. ✅ Show the value of adaptive revision strategies
4. ✅ Better reflect real-world scenarios where retrieval isn't perfect

**Recommended**: Start with **Option 2 (k=10)** as a balanced approach, then test Option 1 if needed.

