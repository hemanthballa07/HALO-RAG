# Experiment Configuration Guide

## Overview

You can now configure revision strategies and top-k retrieval settings via `config/config.yaml` or CLI arguments. This allows you to run different experiments and compare strategy effectiveness.

## Configuration Options

### 1. Top-K Retrieval Settings

In `config/config.yaml`:
```yaml
retrieval:
  top_k_retrieve: 20  # Number of documents to retrieve from corpus
  top_k_rerank: 5     # Number of documents to keep after reranking
```

**Priority order:**
1. CLI arguments (`--top-k-retrieve`, `--top-k-rerank`) - highest priority
2. `config.retrieval.top_k_retrieve/rerank` - medium priority
3. `config.retrieval.fusion/reranker.top_k` - fallback

### 2. Revision Strategy Selection

In `config/config.yaml`:
```yaml
revision:
  strategy_selection_mode: "dynamic"  # or "fixed"
  fixed_strategy: "re_retrieval"      # Only used if mode is "fixed"
```

**Modes:**
- **`"dynamic"`**: Automatically selects strategy based on entailment rate
  - `entailment_rate < 0.5` → re_retrieval
  - `entailment_rate < 0.8` → constrained_generation
  - Otherwise → claim_by_claim
- **`"fixed"`**: Always uses the specified `fixed_strategy`
  - Options: `"re_retrieval"`, `"constrained_generation"`, `"claim_by_claim"`

## Running Different Experiments

### Experiment 1: Test Re-Retrieval Strategy Only

**config.yaml:**
```yaml
revision:
  strategy_selection_mode: "fixed"
  fixed_strategy: "re_retrieval"
  
retrieval:
  top_k_retrieve: 10  # Reduced for better demonstration
  top_k_rerank: 5
```

**Run:**
```bash
python experiments/exp9_complete_pipeline.py
```

### Experiment 2: Test Constrained Generation Strategy Only

**config.yaml:**
```yaml
revision:
  strategy_selection_mode: "fixed"
  fixed_strategy: "constrained_generation"
  
retrieval:
  top_k_retrieve: 10
  top_k_rerank: 5
```

**Run:**
```bash
python experiments/exp9_complete_pipeline.py
```

### Experiment 3: Test Claim-by-Claim Strategy Only

**config.yaml:**
```yaml
revision:
  strategy_selection_mode: "fixed"
  fixed_strategy: "claim_by_claim"
  
retrieval:
  top_k_retrieve: 10
  top_k_rerank: 5
```

**Run:**
```bash
python experiments/exp9_complete_pipeline.py
```

### Experiment 4: Dynamic Strategy Selection (Default)

**config.yaml:**
```yaml
revision:
  strategy_selection_mode: "dynamic"
  
retrieval:
  top_k_retrieve: 10
  top_k_rerank: 5
```

**Run:**
```bash
python experiments/exp9_complete_pipeline.py
```

### Experiment 5: Override via CLI

You can also override config values via CLI:

```bash
# Use fixed strategy with CLI override
python experiments/exp9_complete_pipeline.py --top-k-retrieve 5 --top-k-rerank 3

# The strategy mode still comes from config.yaml
```

## Comparing Strategies

### Recommended Workflow

1. **Create separate config files** for each strategy:
   - `config_exp_re_retrieval.yaml`
   - `config_exp_constrained.yaml`
   - `config_exp_claim_by_claim.yaml`
   - `config_exp_dynamic.yaml`

2. **Run each experiment:**
   ```bash
   python experiments/exp9_complete_pipeline.py --config config_exp_re_retrieval.yaml
   python experiments/exp9_complete_pipeline.py --config config_exp_constrained.yaml
   python experiments/exp9_complete_pipeline.py --config config_exp_claim_by_claim.yaml
   python experiments/exp9_complete_pipeline.py --config config_exp_dynamic.yaml
   ```

3. **Compare results** in `results/metrics/exp9_complete_pipeline.json`:
   - Check `strategy_selection_mode` and `fixed_strategy` in metadata
   - Compare metrics: `hallucination_rate`, `factual_precision`, `verified_f1`
   - Check revision history to see which strategy was used

## Example Config Files

### config_exp_re_retrieval.yaml
```yaml
# Copy base config and modify:
revision:
  strategy_selection_mode: "fixed"
  fixed_strategy: "re_retrieval"
  
retrieval:
  top_k_retrieve: 10
  top_k_rerank: 5
```

### config_exp_constrained.yaml
```yaml
revision:
  strategy_selection_mode: "fixed"
  fixed_strategy: "constrained_generation"
  
retrieval:
  top_k_retrieve: 10
  top_k_rerank: 5
```

### config_exp_claim_by_claim.yaml
```yaml
revision:
  strategy_selection_mode: "fixed"
  fixed_strategy: "claim_by_claim"
  
retrieval:
  top_k_retrieve: 10
  top_k_rerank: 5
```

## CSV Output

The CSV file will include:
- All metrics (hallucination_rate, factual_precision, etc.)
- Metadata section with:
  - `corpus_size`
  - `top_k_retrieve`
  - `top_k_rerank`
  - `corpus_to_k_ratio`
  - `strategy_selection_mode`
  - `fixed_strategy`

This makes it easy to compare different experiments!

## Notes

- When `strategy_selection_mode: "fixed"`, the system will use the same strategy for ALL revisions
- This is useful for controlled experiments comparing strategy effectiveness
- The `fixed_strategy` must be one of: `"re_retrieval"`, `"constrained_generation"`, `"claim_by_claim"`
- If `strategy_selection_mode: "dynamic"`, the `fixed_strategy` is ignored

