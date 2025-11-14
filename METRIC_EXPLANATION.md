# Explanation of Verification Metrics

## How Metrics Are Calculated

### 1. **Hallucination Rate** (1.0 in your example)
- **Definition**: Fraction of claims from the **generated text** that are **NOT entailed** by the **retrieved/reranked context**
- **Formula**: `num_unverified_claims / total_claims_from_generated_text`
- **Reference**: It's calculated with respect to the **retrieved context**, NOT the ground truth
- **Code**: `src/evaluation/metrics.py` lines 273-296

### 2. **Factual Precision** (0.0 in your example)
- **Definition**: Fraction of claims from the **generated text** that ARE entailed by the **retrieved/reranked context**
- **Formula**: `num_entailed_claims / total_claims_from_generated_text`
- **Reference**: It's calculated with respect to the **retrieved context**, NOT the ground truth
- **Code**: `src/evaluation/metrics.py` lines 198-221
- **Note**: `factual_precision = 1 - hallucination_rate` (they're complementary)

### 3. **Factual Recall** (1.0 in your example)
- **Definition**: Fraction of **ground truth claims** that are entailed by the **retrieved/reranked context**
- **Formula**: `num_entailed_gt_claims / total_gt_claims`
- **Reference**: It's calculated with respect to the **retrieved context**, checking if ground truth claims are supported
- **Code**: `src/evaluation/metrics.py` lines 223-271

## Your Specific Example

**Query**: "Which album was darker in tone from her previous work?"
- **Ground truth**: "Beyoncé"
- **Generated**: "Beyoncé (2013)"
- **Context**: Contains "Beyoncé (2013)" explicitly: "Her critically acclaimed fifth studio album, Beyoncé (2013), was distinguished..."

### Why the Metrics Are:
1. **factual_precision = 0.0**: The claim extractor extracted "Beyoncé (2013)" as a claim from the generated text. The verifier checked if this claim is entailed by the context. Even though "Beyoncé (2013)" appears in the context, the NLI model (when checking if "The answer is Beyoncé (2013)" is entailed) may have determined that the specific claim format is not fully entailed. This could be due to:
   - The NLI model being strict about claim formatting
   - The claim "Beyoncé (2013)" being checked as a complete sentence hypothesis, which may not match the context's phrasing exactly
   - Potential encoding/character differences (accented vs non-accented characters)

2. **factual_recall = 1.0**: The ground truth claim "Beyoncé" was extracted and verified against the context. The verifier found that "Beyoncé" is entailed by the context (which it is - the context mentions "Beyoncé" multiple times).

3. **hallucination_rate = 1.0**: Since factual_precision = 0.0, this means 100% of claims from the generated text are not entailed, hence hallucination_rate = 1.0.

## Key Insight

**Hallucination Rate and Factual Precision are calculated with respect to the RETRIEVED/RERANKED CONTEXT, not the ground truth.**

This is by design - these metrics measure whether the generated answer is supported by the evidence that was retrieved, which is what matters for RAG systems. The system should only generate answers that can be verified against the retrieved context.

## Summary

- **Hallucination Rate** = % of generated claims NOT supported by retrieved context
- **Factual Precision** = % of generated claims supported by retrieved context  
- **Factual Recall** = % of ground truth claims supported by retrieved context

All three metrics use the **retrieved/reranked context** as the reference, not the ground truth answer.



## How Metrics Are Calculated

### 1. **Hallucination Rate** (1.0 in your example)
- **Definition**: Fraction of claims from the **generated text** that are **NOT entailed** by the **retrieved/reranked context**
- **Formula**: `num_unverified_claims / total_claims_from_generated_text`
- **Reference**: It's calculated with respect to the **retrieved context**, NOT the ground truth
- **Code**: `src/evaluation/metrics.py` lines 273-296

### 2. **Factual Precision** (0.0 in your example)
- **Definition**: Fraction of claims from the **generated text** that ARE entailed by the **retrieved/reranked context**
- **Formula**: `num_entailed_claims / total_claims_from_generated_text`
- **Reference**: It's calculated with respect to the **retrieved context**, NOT the ground truth
- **Code**: `src/evaluation/metrics.py` lines 198-221
- **Note**: `factual_precision = 1 - hallucination_rate` (they're complementary)

### 3. **Factual Recall** (1.0 in your example)
- **Definition**: Fraction of **ground truth claims** that are entailed by the **retrieved/reranked context**
- **Formula**: `num_entailed_gt_claims / total_gt_claims`
- **Reference**: It's calculated with respect to the **retrieved context**, checking if ground truth claims are supported
- **Code**: `src/evaluation/metrics.py` lines 223-271

## Your Specific Example

**Query**: "Which album was darker in tone from her previous work?"
- **Ground truth**: "Beyoncé"
- **Generated**: "Beyoncé (2013)"
- **Context**: Contains "Beyoncé (2013)" explicitly: "Her critically acclaimed fifth studio album, Beyoncé (2013), was distinguished..."

### Why the Metrics Are:
1. **factual_precision = 0.0**: The claim extractor extracted "Beyoncé (2013)" as a claim from the generated text. The verifier checked if this claim is entailed by the context. Even though "Beyoncé (2013)" appears in the context, the NLI model (when checking if "The answer is Beyoncé (2013)" is entailed) may have determined that the specific claim format is not fully entailed. This could be due to:
   - The NLI model being strict about claim formatting
   - The claim "Beyoncé (2013)" being checked as a complete sentence hypothesis, which may not match the context's phrasing exactly
   - Potential encoding/character differences (accented vs non-accented characters)

2. **factual_recall = 1.0**: The ground truth claim "Beyoncé" was extracted and verified against the context. The verifier found that "Beyoncé" is entailed by the context (which it is - the context mentions "Beyoncé" multiple times).

3. **hallucination_rate = 1.0**: Since factual_precision = 0.0, this means 100% of claims from the generated text are not entailed, hence hallucination_rate = 1.0.

## Key Insight

**Hallucination Rate and Factual Precision are calculated with respect to the RETRIEVED/RERANKED CONTEXT, not the ground truth.**

This is by design - these metrics measure whether the generated answer is supported by the evidence that was retrieved, which is what matters for RAG systems. The system should only generate answers that can be verified against the retrieved context.

## Summary

- **Hallucination Rate** = % of generated claims NOT supported by retrieved context
- **Factual Precision** = % of generated claims supported by retrieved context  
- **Factual Recall** = % of ground truth claims supported by retrieved context

All three metrics use the **retrieved/reranked context** as the reference, not the ground truth answer.

