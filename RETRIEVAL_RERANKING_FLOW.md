# Retrieval and Reranking Flow

## Two-Stage Process

The pipeline uses a **two-stage retrieval process**:

### Stage 1: Retrieval (top_k_retrieve)
- **Purpose**: Retrieve documents from the entire corpus
- **Method**: Hybrid retrieval (FAISS dense + BM25 sparse)
- **Parameter**: `top_k_retrieve` (default: 20)
- **Output**: Top k documents from corpus

### Stage 2: Reranking (top_k_rerank)
- **Purpose**: Reorder and filter the retrieved documents
- **Method**: Cross-encoder reranker (DeBERTa-v3-base on MS MARCO)
- **Parameter**: `top_k_rerank` (default: 5)
- **Input**: The documents from Stage 1 (top_k_retrieve documents)
- **Output**: Top k documents from the reranked results

## Flow Diagram

```
Corpus (5000 documents)
    ↓
[Stage 1: Hybrid Retrieval]
    ↓ top_k_retrieve = 20
20 retrieved documents
    ↓
[Stage 2: Cross-Encoder Reranking]
    ↓ top_k_rerank = 5
5 reranked documents (best from the 20)
    ↓
[Generation]
Uses these 5 documents as context
```

## Code Flow

```python
# Step 1: Retrieve top_k_retrieve documents from corpus
retrieved_docs = self.retriever.retrieve(query, top_k=top_k_retrieve)  # e.g., 20 docs

# Step 2: Rerank the retrieved documents and take top_k_rerank
reranked_docs = self.reranker.rerank(
    query,
    retrieved_texts,  # The 20 documents from step 1
    top_k=top_k_rerank  # e.g., take top 5 from reranked 20
)

# Step 3: Use reranked documents for generation
context = " ".join(reranked_texts)  # Top 5 documents
generated_text = self.generator.generate(query, context)
```

## Important Points

1. **Reranker does NOT retrieve from corpus**: It only reranks what was already retrieved
2. **top_k_rerank ≤ top_k_retrieve**: You can't rerank more documents than you retrieved
3. **Reducing top_k_retrieve**: 
   - Reduces the pool of documents the reranker can choose from
   - Example: If top_k_retrieve=10, reranker can only choose from 10 documents (not 20)
   - Then takes top_k_rerank=5 from those 10

## Example Scenarios

### Scenario 1: Default (k_retrieve=20, k_rerank=5)
- Retrieve 20 documents from corpus
- Rerank those 20 documents
- Take top 5 for generation
- **Reranker has 20 candidates to choose from**

### Scenario 2: Reduced (k_retrieve=10, k_rerank=5)
- Retrieve 10 documents from corpus
- Rerank those 10 documents
- Take top 5 for generation
- **Reranker has only 10 candidates to choose from** (more constrained)

### Scenario 3: Very Reduced (k_retrieve=5, k_rerank=3)
- Retrieve 5 documents from corpus
- Rerank those 5 documents
- Take top 3 for generation
- **Reranker has only 5 candidates to choose from** (very constrained)

## Impact on Re-Retrieval Strategy

When re-retrieval strategy runs:
- It uses `top_k_retrieve` to retrieve new documents with expanded query
- The reranker then reranks those documents and takes `top_k_rerank`
- So reducing `top_k_retrieve` makes re-retrieval more important because:
  - Initial retrieval might miss relevant docs (smaller k)
  - Re-retrieval with expanded query can find different/better docs
  - Reranker can then select better documents from the new retrieval

## Current Defaults

- `top_k_retrieve`: 20 (from config: `retrieval.fusion.top_k`)
- `top_k_rerank`: 5 (from config: `retrieval.reranker.top_k`)

Both are now configurable via CLI arguments:
- `--top-k-retrieve`: Override retrieval k
- `--top-k-rerank`: Override reranking k

