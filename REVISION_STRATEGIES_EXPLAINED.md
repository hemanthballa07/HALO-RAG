
#  **Revision Strategies in HALO‑RAG**

HALO‑RAG uses an **adaptive revision loop** to eliminate hallucinations.  
After the generator produces an answer, we:

1. Extract atomic claims  
2. Verify each claim with NLI  
3. Choose a revision strategy based on entailment rate  
4. Rewrite the answer  
5. Re‑verify  
6. Repeat up to 3 iterations  

This section explains the **three revision strategies** with **examples**.

---

# 1. **RE_RETRIEVAL Strategy**

### **Purpose**
Fix **retrieval failures**, not generation failures.

- Used when **entailment_rate < 0.5**  
- Most claims are unsupported  
- The generator likely lacked the right evidence  
- Solution: expand the query and retrieve again

---

## **How it works**

### 1. Identify failed claims  
These often contain missing entities or relations.

### 2. Expand the query  
```python
expanded_query = f"{query} {' '.join(failed_claims[:2])}"
```

### 3. Re‑retrieve with expanded query  
```python
new_contexts = retrieval_fn(expanded_query, top_k=20)
```

### 4. Regenerate using new contexts  
This time the generator has better evidence.

### 5. Re‑verify  
If still unsupported → loop continues.

---

## **Example**

### **Question:**  
“When was the iPhone 5 released?”

### **Original retrieval:**  
Passages about iPhone 6, iPhone 5S, etc.

### **Original generation:**  
“The iPhone 5 was released in 2013.”

### **Extracted claim:**  
- “iPhone 5 was released in 2013.” →  unsupported

### **Failed claims:**  
- “iPhone 5 was released in 2013.”

### **Expanded query:**  
“iPhone 5 release date 2013”

### **New retrieval:**  
Now retrieves passages stating:  
- “The iPhone 5 was released in 2012.”

### **New generation:**  
“The iPhone 5 was released in 2012.”

### **Verification:**  
Now supported → revision loop ends.

---

# 2.  **CONSTRAINED_GENERATION Strategy**

### **Purpose**
Improve **generation quality**, not retrieval quality.

- Retrieval stays the same  
- Context stays the same  
- We simply guide the generator to include **verified facts**  
- Used when **0.5 ≤ entailment_rate < 0.8** (partially correct answer)

---

## **How it works**

### 1. Use the same retrieved contexts
```python
contexts = retrieval_fn(query, top_k=10)
```

### 2. Collect verified claims  
These are claims from the answer that NLI confirmed as supported.

### 3. Add verified claims to the prompt  
```text
Question: {query}
Context: {context}
Verified facts that must be included: {claim1} | {claim2}
Answer:
```

### 4. Generate a new answer  
The generator is now “anchored” by verified facts and less likely to drift.

---

## ** Example**

### **Question:**  
“What is the capital of France and what river runs through it?”

### **Retrieved context:**  
- “Paris is the capital of France.”  
- “The Seine River flows through Paris.”

### **Original generation:**  
“Paris is the capital of France.”

### **Extracted claims:**  
1. Paris is the capital of France.

### **Verified claims:**  
- “Paris is the capital of France.”

### **Constrained prompt:**  
```
Question: What is the capital of France and what river runs through it?
Context: [passages about Paris + Seine]
Verified facts that must be included: Paris is the capital of France
Answer:
```

### **New generation:**  
“Paris is the capital of France, and the Seine River runs through the city.”

Now the answer is complete and grounded.

---

# 3.  **CLAIM_BY_CLAIM Strategy**

### **Purpose**
Regenerate **only the unverified claims**, while preserving the verified ones.

- Used when **entailment_rate ≥ 0.8**  
- Most of the answer is correct  
- Only a few claims need fixing  
- No need to regenerate the whole answer

---

## **How it works**

### 1. Separate claims  
- `verified_claims`: keep these  
- `unverified_claims`: regenerate these

### 2. For each unverified claim  
Create a focused query:

```text
"{query} Specifically about: {claim}"
```

Generate a replacement claim.

### 3. Reconstruct the answer  
```python
revised_generation = " ".join(verified_claims + revised_claims)
```

### 4. Re‑verify the reconstructed answer

---

## **Example**

### **Initial generation:**  
“UF was founded in 1900. UF colors are orange and blue. UF has 100,000 students.”

### **Verification results:**  
- “UF was founded in 1900” →  not supported  
- “UF colors are orange and blue” → supported  
- “UF has 100,000 students” → not supported  

### **CLAIM_BY_CLAIM Strategy**

1. **Verified claims:**  
   - “UF colors are orange and blue”

2. **Unverified claims → regenerate:**  
   - “UF was founded in 1900” → “UF was founded in 1853”  
   - “UF has 100,000 students” → “UF has over 50,000 students”

3. **Reconstructed answer:**  
```
UF colors are orange and blue.
UF was founded in 1853.
UF has over 50,000 students.
```

4. **Re‑verify** → now all claims are supported.

---

#  **Strategy Comparison Table**

| Strategy | Retrieval | Generation | What Changes | When Used |
|----------|-----------|------------|--------------|-----------|
| **RE_RETRIEVAL** |  New retrieval |  New generation | Both retrieval + generation | Low entailment (<0.5) |
| **CONSTRAINED_GENERATION** | Same retrieval | New generation | Guided by verified claims | Medium entailment (0.5–0.8) |
| **CLAIM_BY_CLAIM** | Same retrieval | Partial regeneration | Only unverified claims | High entailment (≥0.8) |

---

#  **Why Three Strategies?**

### **1. RE_RETRIEVAL**  
When the model is wrong because the **evidence was wrong**.
   - Problem: Most claims are wrong → need better context
   - Solution: Expand query, get better documents, regenerate everything

### **2. CONSTRAINED_GENERATION**  
When the model is partially right but generation needs **guidance**.
   - Problem: Some claims are right, some are wrong → context is OK, generation needs guidance
   - Solution: Tell generator to include verified facts, regenerate with hints

### **3. CLAIM_BY_CLAIM**  
When the model is mostly right and needs **surgical repair**
   - Problem: Most claims are right, only a few wrong → don't waste time regenerating everything
   - Solution: Only regenerate the few wrong claims, keep the rest