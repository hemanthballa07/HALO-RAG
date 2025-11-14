# Metrics Classification: Evaluation vs Production

This document classifies all metrics computed in exp9 based on whether they require ground truth (evaluation-only) or can be computed in production.

## Evaluation-Only Metrics (Require Ground Truth)

These metrics are **only for evaluation** and cannot be computed in production where ground truth is unavailable:

### Retrieval Metrics (require `relevant_docs`)
- **recall@5, recall@10, recall@20**: Fraction of relevant docs in top-k retrieved
- **precision@5, precision@10**: Fraction of top-k retrieved that are relevant
- **mrr** (Mean Reciprocal Rank): Reciprocal rank of first relevant document
- **ndcg@10**: Normalized Discounted Cumulative Gain@10

### Coverage Metrics
- **coverage**: Fraction of ground truth answer tokens that appear in retrieved documents
  - Uses `ground_truth` to compute which tokens should be in retrieved docs

### Verification Metrics
- **factual_recall**: Fraction of ground truth claims that are entailed by the context
  - Requires `ground_truth_claims` to verify against retrieved contexts

### Generation Quality Metrics (require `ground_truth` text)
- **exact_match**: Binary match (1.0 if exact match, 0.0 otherwise)
- **f1_score**: Token overlap F1 between generated and ground truth
- **bleu4**: BLEU-4 score comparing generated vs ground truth
- **rouge_l**: ROUGE-L F1 score comparing generated vs ground truth

### Composite Metrics
- **verified_f1**: F1 score × Factual Precision
  - Depends on `f1_score` which requires ground truth
- **fever_score**: Harmonic mean of label accuracy and evidence recall
  - Uses `ground_truth` for evidence recall calculation

---

## Production-Ready Metrics (No Ground Truth Required)

These metrics can be computed in production and are available in the deployed system:

### Verification Metrics
- **factual_precision**: Fraction of generated claims that are entailed by retrieved contexts
  - Only requires verification results (entailment checking)
- **hallucination_rate**: Fraction of generated claims that are NOT entailed
  - Only requires verification results (entailment checking)
  - Note: Returns 0.0 when system abstains (no claims made = no hallucinations)

### System Behavior Metrics
- **abstention_rate**: Whether the system abstained from answering
  - Detected from generated text patterns (e.g., "I cannot provide a confident answer...")
  - No ground truth required

---

## Summary

**Total metrics computed in exp9**: 12 metrics

**Evaluation-only (9 metrics)** - require ground truth:
1. recall@20 (also computes recall@5, recall@10, precision@5, precision@10, mrr, ndcg@10 internally)
2. coverage
3. factual_recall
4. exact_match
5. f1_score
6. bleu4
7. rouge_l
8. verified_f1
9. fever_score

**Production-ready (3 metrics)** - no ground truth required:
1. factual_precision
2. hallucination_rate
3. abstention_rate

Note: While `compute_all_metrics()` computes additional retrieval metrics (recall@5, recall@10, precision@5, precision@10, mrr, ndcg@10), exp9 only aggregates and reports `recall@20` in the summary. All retrieval metrics require `relevant_docs` (ground truth).

---

## Notes

- The revision process does **NOT** use ground truth - it's production-ready
- All production-ready metrics are based on self-verification (entailment checking against retrieved contexts)
- In production, you can monitor: `factual_precision`, `hallucination_rate`, and `abstention_rate` to track system performance

