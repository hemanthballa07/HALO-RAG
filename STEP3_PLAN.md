# Step 3: DeBERTa Entailment Verification - Implementation Plan

## Objective
Implement real entailment verification using DeBERTa-v3-base NLI model to replace the UNVERIFIED stub.

---

## What Step 3 Will Do

Given a **Claim** and **Evidence passages**, determine:
- **ENTAILMENT**: Evidence supports the claim (confidence ≥ threshold)
- **NEUTRAL**: Evidence doesn't confirm or deny (confidence < threshold)
- **CONTRADICTION**: Evidence contradicts the claim

---

## Implementation Steps

### 1. Load DeBERTa NLI Model
**Model:** `cross-encoder/nli-deberta-v3-base`
**Task:** Natural Language Inference (NLI)
**Labels:** 0=contradiction, 1=neutral, 2=entailment

### 2. Implement `_verify_claim()` Method
Replace stub in `src/verification_controller/controller.py`:

```python
def _verify_claim(self, claim: Claim, evidence_passages: List[str]) -> Claim:
    """
    Verify a single claim against evidence using DeBERTa NLI.
    
    Strategy:
    1. For each evidence passage, compute entailment score
    2. Take maximum score across all passages
    3. Assign label based on threshold
    """
    if not evidence_passages:
        claim.entailment = "NEUTRAL"
        claim.confidence = 0.0
        return claim
    
    # Compute entailment scores for each passage
    scores = []
    for passage in evidence_passages:
        score = self.entailment_model.predict(claim.text, passage)
        scores.append(score)
    
    # Take max score (best evidence)
    max_score = max(scores)
    claim.confidence = max_score
    
    # Assign label
    if max_score >= self.entailment_threshold:
        claim.entailment = "ENTAILMENT"
    else:
        claim.entailment = "NEUTRAL"
    
    return claim
```

### 3. Initialize Model in Controller

```python
class VerificationController:
    def __init__(self, ...):
        # ... existing code ...
        
        # Initialize DeBERTa entailment model (NEW - Step 3)
        logger.info("Loading DeBERTa NLI model...")
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        
        model_name = "cross-encoder/nli-deberta-v3-base"
        self.entailment_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.entailment_model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.entailment_model.eval()
        
        # Move to device if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.entailment_model.to(self.device)
```

### 4. Update `verify()` Method

```python
def verify(self, generated_answer, retrieved_docs, query):
    # Step 1: Extract claims (DONE - Step 2)
    claims = self._extract_claims(generated_answer)
    
    # Step 2: Verify claims (NEW - Step 3)
    evidence_passages = retrieved_docs.get("passages", [])
    
    verified_claims = []
    for claim in claims:
        verified_claim = self._verify_claim(claim, evidence_passages)
        verified_claims.append(verified_claim)
    
    # Step 3: Determine overall verification status
    num_verified = sum(1 for c in verified_claims if c.entailment == "ENTAILMENT")
    num_total = len(verified_claims)
    all_verified = num_verified == num_total and num_total > 0
    
    avg_confidence = sum(c.confidence for c in verified_claims) / max(num_total, 1)
    
    return VerificationResult(
        claims=verified_claims,
        verified=all_verified,
        revision_cycles=0,
        reason=f"{num_verified}/{num_total} claims verified" if all_verified else f"Only {num_verified}/{num_total} claims verified",
        total_claims=num_total,
        verified_claims=num_verified,
        hallucinated_claims=num_total - num_verified,
        avg_confidence=avg_confidence
    )
```

---

## Expected Behavior

### Before (Step 2)
```
Claim: "Paris be the capital"
Evidence: ["Paris is the capital of France."]
Result: UNVERIFIED (confidence: 0.0)
```

### After (Step 3)
```
Claim: "Paris be the capital"
Evidence: ["Paris is the capital of France."]
Result: ENTAILMENT (confidence: 0.95)
```

---

## Testing Strategy

### Unit Tests
```python
def test_verify_claim_entailment():
    controller = VerificationController()
    claim = Claim(claim_id="C1", text="Paris is the capital")
    evidence = ["Paris is the capital of France."]
    
    verified = controller._verify_claim(claim, evidence)
    
    assert verified.entailment == "ENTAILMENT"
    assert verified.confidence >= 0.75

def test_verify_claim_neutral():
    controller = VerificationController()
    claim = Claim(claim_id="C1", text="Paris has 10 million people")
    evidence = ["Paris is the capital of France."]
    
    verified = controller._verify_claim(claim, evidence)
    
    assert verified.entailment == "NEUTRAL"
    assert verified.confidence < 0.75
```

---

## Definition of DONE

✅ DeBERTa model loads successfully  
✅ `_verify_claim()` returns ENTAILMENT/NEUTRAL labels  
✅ Confidence scores are computed correctly  
✅ `verify()` method aggregates results  
✅ Tests pass  
✅ Demo shows verified claims (not UNVERIFIED)  

---

## Estimated Time
**45-60 minutes**

---

## Next Action
Ready to implement Step 3?
