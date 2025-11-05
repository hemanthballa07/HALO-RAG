# Methods Overview

## Self-Verification Chains for Hallucination-Free RAG

### 1. Architecture Overview

Our Self-Verification RAG pipeline consists of five main components:

1. **Hybrid Retrieval**: Combines dense (FAISS) and sparse (BM25) retrieval
2. **Cross-Encoder Reranking**: Reorders retrieved documents using DeBERTa-v3-base
3. **FLAN-T5 Generation**: Generates answers with QLoRA fine-tuning
4. **Entailment-Based Verification**: Verifies claims using DeBERTa-v3-large
5. **Adaptive Revision**: Applies revision strategies when verification fails

### 2. Hybrid Retrieval

**Dense Retrieval (FAISS)**:
- Model: `sentence-transformers/all-mpnet-base-v2`
- Embedding dimension: 768
- Similarity metric: Cosine similarity
- Weight: 0.6

**Sparse Retrieval (BM25)**:
- Model: Rank-BM25
- Tokenization: Lowercase word splitting
- Weight: 0.4

**Fusion**:
- Score normalization to [0, 1]
- Weighted combination: `score_fusion = 0.6 × score_dense + 0.4 × score_sparse`
- Top-k retrieval: k = 20

**Target Metrics**:
- Recall@20 ≥ 0.95
- Coverage ≥ 0.90

### 3. Cross-Encoder Reranking

**Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Fine-tuned on MS MARCO dataset
- Input: Query-document pairs
- Output: Relevance scores
- Top-k after reranking: k = 5

### 4. FLAN-T5 Generation

**Base Model**: `google/flan-t5-large`
- Parameters: 780M
- Architecture: Encoder-decoder

**QLoRA Fine-Tuning**:
- Quantization: 4-bit NF4
- LoRA rank (r): 16
- LoRA alpha: 32
- LoRA dropout: 0.1
- Target modules: ["q", "v", "k", "o"]
- Training: 3 epochs, batch size 8, gradient accumulation 4

**Generation Parameters**:
- Max new tokens: 256
- Temperature: 0.7
- Top-p: 0.95
- Top-k: 50
- Sampling: Nucleus sampling

### 5. Entailment-Based Verification

**Model**: `microsoft/deberta-v3-large`
- Fine-tuned on MNLI + FEVER datasets
- Input: Premise (context) + Hypothesis (claim)
- Output: Entailment probability

**Claim Extraction**:
- Method: spaCy SVO (Subject-Verb-Object) extraction
- Model: `en_core_web_sm`
- Extracts factual claims from generated text

**Verification Process**:
1. Extract claims from generated text
2. For each claim, check entailment against retrieved contexts
3. Compute entailment score: P(entailment | context, claim)
4. Threshold decision: τ = 0.75 (tuned via Experiment 3)

**Target Metrics**:
- Factual Precision ≥ 0.90
- Hallucination Rate ≤ 0.10

### 6. Adaptive Revision Strategies

When verification fails (entailment rate < 0.90), apply one of three strategies:

**Strategy 1: Re-Retrieval**
- Expand query with failed claims
- Re-retrieve with expanded query (k = 20)
- Re-generate with new contexts

**Strategy 2: Constrained Generation**
- Use verified claims as constraints
- Generate with verified claims as hints
- Re-verify new generation

**Strategy 3: Claim-by-Claim Regeneration**
- Regenerate only unverified claims
- Preserve verified claims
- Reconstruct generation with verified + revised claims

**Selection Logic**:
- Entailment rate < 0.5: Re-retrieval
- Entailment rate < 0.8: Constrained generation
- Otherwise: Claim-by-claim

**Max Iterations**: 3

### 7. Evaluation Metrics

#### Retrieval Metrics
- **Recall@K**: Fraction of relevant docs in top-k
- **MRR**: Mean Reciprocal Rank
- **NDCG@K**: Normalized Discounted Cumulative Gain@K
- **Coverage**: Fraction of corpus covered by top-k retrievals

#### Verification Metrics
- **Factual Precision**: Fraction of claims that are entailed
- **Factual Recall**: Fraction of ground truth claims that are entailed
- **Hallucination Rate**: Fraction of claims that are not entailed
- **Verified F1**: Harmonic mean of factual precision and recall

#### Generation Metrics
- **Exact Match**: Binary match (0 or 1)
- **F1 Score**: Token overlap F1

### 8. Composite Metric: Verified F1

**Definition**:
```
Verified F1 = 2 × (Factual Precision × Factual Recall) / (Factual Precision + Factual Recall)
```

**Target**: Verified F1 ≥ 0.52

### 9. Statistical Testing

**Methods**:
- Independent samples t-test (α = 0.05)
- Paired t-test for matched samples
- Bootstrap confidence intervals (1000 iterations)

**Significance Criteria**: p < 0.05

### 10. Experiments

**Experiment 1**: Baseline Comparison
- Standard RAG vs Self-Verification RAG
- Metrics: All metrics

**Experiment 2**: Retrieval Comparison
- Dense vs Sparse vs Hybrid
- Metrics: Recall@K, MRR, NDCG@10

**Experiment 3**: Threshold Tuning
- τ ∈ {0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9}
- Find optimal τ maximizing Verified F1 while maintaining Factual Precision ≥ 0.90

**Experiment 4**: Revision Strategies
- No revision vs Adaptive revision
- Metrics: Factual Precision, Hallucination Rate, Verified F1

**Experiment 5**: Decoding Strategies
- Greedy vs Beam (3, 5) vs Nucleus sampling
- Metrics: Factual Precision, Verified F1

**Experiment 6**: Iterative Training
- Self-improvement through fine-tuning loops
- Metrics: Verified F1 over iterations

**Experiment 7**: Ablation Study
- Component-wise contribution analysis
- Variants: No reranking, No verification, No revision, Dense-only, Sparse-only

**Experiment 8**: Stress Test
- Performance on adversarial queries
- Error analysis: High hallucination, Low verification, Poor retrieval

### 11. Key Novel Contributions

1. **Modular Metrics Framework**: Comprehensive evaluation across retrieval, verification, and generation
2. **Verified F1 Composite Metric**: Unified metric combining factual precision and recall
3. **Iterative Self-Improvement**: Fine-tuning loops for continuous improvement
4. **Threshold Optimization Framework**: Systematic τ-tuning for optimal verification
5. **Evidence-Based Verification Loop**: End-to-end pipeline with adaptive revision

### 12. Expected Results

**Baseline (Standard RAG)**:
- Recall@20: ~0.85
- Factual Precision: ~0.75
- Verified F1: ~0.42

**Proposed (Self-Verification RAG)**:
- Recall@20: ≥0.95 (+12%)
- Coverage: ≥0.90
- Factual Precision: ≥0.90 (+20%)
- Hallucination Rate: ≤0.10
- Verified F1: ≥0.52 (+26%)

**Statistical Significance**: All improvements with p < 0.05

### 13. Implementation Details

**Hardware**: HiperGator (V100/A100 GPUs)

**Software**:
- PyTorch 2.0+
- Transformers 4.35+
- Sentence-Transformers 2.2+
- FAISS (CPU or GPU)
- spaCy 3.7+

**Reproducibility**:
- Random seed: 42
- Model checkpoints saved
- Configuration files versioned
- Dataset versioning

### 14. References

- Lewis et al. (2020): RAG: Retrieval-Augmented Generation
- Manakul et al. (2023): SelfCheckGPT
- Gao et al. (2023): Entailment-based Factual Verification
- Hu et al. (2021): LoRA: Low-Rank Adaptation
- Dettmers et al. (2023): QLoRA: Efficient Finetuning

