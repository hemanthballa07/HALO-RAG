# Revision Strategies Explained

## 1. CONSTRAINED_GENERATION Strategy

### Purpose
**NOT for improving retrieval quality** - it uses the **SAME contexts** as before.  
**Purpose**: Improve **generation quality** by guiding the generator to include verified facts.

### How it works:

1. **Uses same retrieval** (line 219):
   ```python
   contexts = retrieval_fn(query, top_k=10)  # Same query, same contexts
   ```

2. **Gets verified claims** (lines 213-216):
   - Extracts claims that passed verification (entailment_score >= threshold)
   - These are facts we KNOW are correct

3. **Adds verified claims to prompt** (lines 224-228, 138-140):
   ```python
   # The prompt becomes:
   "Question: {query} Context: {context} Verified facts that must be included: {claim1} | {claim2} Answer:"
   ```

4. **Why this helps**:
   - The generator sees verified facts explicitly in the prompt
   - It's more likely to include these facts in its output
   - It can use these facts to generate a more complete, accurate answer
   - Example: If "UF colors are orange and blue" is verified, the generator is told to include this

### Example:
```
Original generation: "UF is a university in Florida."
Verified claims: ["UF colors are orange and blue", "UF mascot is Albert the Alligator"]

Constrained prompt: 
"Question: What are UF's colors and mascot? 
 Context: [retrieved context] 
 Verified facts that must be included: UF colors are orange and blue | UF mascot is Albert the Alligator 
 Answer:"

New generation: "UF's colors are orange and blue and the mascot is Albert the Alligator."
```

---

## 2. CLAIM_BY_CLAIM Strategy

### Purpose
Regenerate **only the unverified claims** while **preserving verified claims**.

### How it works:

1. **Separates claims** (lines 273-285):
   - `unverified_claims`: Claims that failed verification (hallucinations)
   - `verified_claims`: Claims that passed verification (keep these!)

2. **For each unverified claim** (lines 296-307):
   - Creates a **focused query**: `"{query} Specifically about: {claim}"`
   - Generates a **replacement** for that specific claim
   - Example: 
     - Unverified claim: "UF was founded in 1900"
     - Focused query: "When was UF founded? Specifically about: UF was founded in 1900"
     - Generates replacement: "UF was founded in 1853"

3. **Reconstructs answer** (line 314):
   ```python
   revised_generation = " ".join(verified_claims + revised_claims)
   ```
   - **Keeps verified claims AS-IS** (don't regenerate these)
   - **Replaces unverified claims** with newly generated ones
   - **Concatenates** them together

4. **Re-verifies** (lines 320-324):
   - Extracts claims from the reconstructed generation
   - Verifies the entire new answer

### Example Flow:

```
Initial generation: "UF was founded in 1900. UF colors are orange and blue. UF has 100,000 students."

Verification results:
- "UF was founded in 1900" → NOT ENTAILED (unverified)
- "UF colors are orange and blue" → ENTAILED (verified) ✓
- "UF has 100,000 students" → NOT ENTAILED (unverified)

CLAIM_BY_CLAIM Strategy:

1. Verified claims (keep): ["UF colors are orange and blue"]
2. Unverified claims (regenerate):
   - "UF was founded in 1900" → Generate replacement → "UF was founded in 1853"
   - "UF has 100,000 students" → Generate replacement → "UF has over 50,000 students"

3. Reconstruct:
   revised_generation = "UF colors are orange and blue" + " " + "UF was founded in 1853" + " " + "UF has over 50,000 students"
   
   Result: "UF colors are orange and blue UF was founded in 1853 UF has over 50,000 students"

4. Re-verify the entire reconstructed answer
```

### Key Points:
- ✅ **We DO regenerate unverified claims** (not omit them)
- ✅ **We DO keep verified claims** (preserve what's correct)
- ✅ **We DO concatenate** verified + revised claims
- ✅ **We DO re-verify** the final reconstructed answer
- ❌ **We DON'T give back original answer** - we reconstruct with verified + revised

---

## Comparison Table

| Strategy | Retrieval | Generation | What Changes |
|----------|-----------|------------|--------------|
| **RE_RETRIEVAL** | ✅ New (expanded query) | ✅ New (with new contexts) | Both retrieval and generation |
| **CONSTRAINED_GENERATION** | ❌ Same | ✅ New (with verified facts hint) | Only generation (guided by verified claims) |
| **CLAIM_BY_CLAIM** | ❌ Same | ✅ Partial (only unverified claims) | Only unverified claims regenerated |

---

## Why Each Strategy?

1. **RE_RETRIEVAL** (low entailment < 0.5):
   - Problem: Most claims are wrong → need better context
   - Solution: Expand query, get better documents, regenerate everything

2. **CONSTRAINED_GENERATION** (medium entailment 0.5-0.8):
   - Problem: Some claims are right, some are wrong → context is OK, generation needs guidance
   - Solution: Tell generator to include verified facts, regenerate with hints

3. **CLAIM_BY_CLAIM** (high entailment ≥ 0.8):
   - Problem: Most claims are right, only a few wrong → don't waste time regenerating everything
   - Solution: Only regenerate the few wrong claims, keep the rest

