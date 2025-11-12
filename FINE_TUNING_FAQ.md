# Fine-Tuning FAQ: Why Fine-Tune and Scope Limitations

## Quick Answers

### Q1: Why are we fine-tuning FLAN-T5?

**A**: Fine-tuning helps FLAN-T5:
1. **Learn RAG format**: Better understand "Question + Context → Answer" pattern
2. **Domain adaptation**: Adapt to Wikipedia-based QA domain
3. **Self-improvement**: Learn from its own high-quality verified outputs (Exp6 iterative loop)

### Q2: What dataset is used for fine-tuning?

**A**: 
- **Default**: SQuAD v2 (not Natural Questions)
- **Configurable**: Can change `datasets.active` in `config/config.yaml` to `"natural_questions"` or `"hotpotqa"`
- **Fine-tuning data**: Self-generated verified answers (FP ≥ 0.85), not raw dataset

### Q3: Does it limit scope?

**A**: **Yes, but it's acceptable for this project:**

**Limitations**:
- Domain bias: Model better at the source dataset's domain (Wikipedia-based QA)
- Style bias: Model may favor source dataset's question style
- Generalization: May not transfer well to other domains (scientific, medical, legal)

**Why it's OK**:
- ✅ Scope is intentional: Wikipedia-based QA is the target domain
- ✅ FLAN-T5 base maintains generalization (instruction-tuned on diverse tasks)
- ✅ QLoRA preserves most base knowledge (only adapts specific layers)
- ✅ Verified data is high-quality and diverse (self-generated, filtered)
- ✅ Evaluation on multiple datasets shows cross-domain performance
- ✅ Configurable: Can switch datasets or use multiple datasets

### Q4: How to mitigate scope limitations?

**A**: 
1. **Multi-dataset fine-tuning**: Fine-tune on combined data from SQuAD v2, NQ, HotpotQA
2. **Domain-specific fine-tuning**: Fine-tune on domain-specific data for domain-specific apps
3. **Zero-shot evaluation**: Evaluate on new domains without fine-tuning
4. **Continual learning**: Adapt to new domains incrementally

### Q5: Should we fine-tune on Natural Questions instead?

**A**: **Not necessary** - Current approach is fine:
- SQuAD v2 is a good representative dataset
- Verified data collection matters more than source dataset
- Evaluation on multiple datasets shows generalization
- Can switch datasets via config if needed

## Key Insight

**The fine-tuning is on self-generated verified data, not the raw dataset. The source dataset (SQuAD v2, NQ, HotpotQA) determines the question style/domain, but the model learns from its own high-quality outputs, creating a self-improvement cycle.**

---

## Current Configuration

```yaml
datasets:
  active: "squad_v2"  # Default: SQuAD v2 (not Natural Questions)
```

**To use Natural Questions**:
```yaml
datasets:
  active: "natural_questions"  # Change this
```

**To use HotpotQA**:
```yaml
datasets:
  active: "hotpotqa"  # Change this
```

## Bottom Line

1. **Fine-tuning helps**: Improves performance on RAG task
2. **Dataset is configurable**: Default is SQuAD v2, can change to NQ or HotpotQA
3. **Scope is limited but acceptable**: Wikipedia-based QA is the intended domain
4. **Self-improvement loop is key**: Model learns from its own verified outputs
5. **Multi-dataset evaluation**: Shows generalization across domains

---

*See `FINE_TUNING_SCOPE_ANALYSIS.md` for detailed analysis.*

