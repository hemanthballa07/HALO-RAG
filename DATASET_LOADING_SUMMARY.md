# Dataset Loading Implementation Summary

## Overview
Implemented unified dataset loading module for HALO-RAG project supporting SQuAD v2, Natural Questions, and HotpotQA datasets.

## Implementation Status: ✅ COMPLETE

### Files Created
1. **src/data/__init__.py**: Public API for dataset loading
2. **src/data/loaders.py**: Unified dataset loaders with normalization
3. **experiments/check_dataset_loading.py**: Verification script
4. **experiments/example_usage_dataset_loading.py**: Usage example
5. **data/README.md**: Updated documentation

### Files Modified
1. **config/config.yaml**: Added dataset configuration
2. **IMPLEMENTATION_STATUS.md**: Updated with dataset loading status

## Supported Datasets

### 1. SQuAD v2.0 ✅
- **Size**: ~150K examples
- **Features**: Includes unanswerable questions
- **Loader**: `load_squad_v2()`
- **Handles**: `is_impossible` flag for unanswerable questions

### 2. Natural Questions (NQ) ✅
- **Size**: ~300K examples
- **Features**: Complex HTML structure with tokenized documents
- **Loader**: `load_natural_questions()`
- **Handles**: Extracts text from tokenized HTML documents

### 3. HotpotQA ✅
- **Size**: ~113K examples
- **Features**: Multi-hop questions with multiple context paragraphs
- **Loader**: `load_hotpotqa()`
- **Handles**: Combines multiple context paragraphs into single context

## Unified Schema

All datasets are normalized to:
```python
{
    "id": str,              # Unique example ID
    "question": str,        # Question text (normalized)
    "context": str,         # Context/document text (normalized)
    "answers": List[str]    # List of answers (empty list if unanswerable)
}
```

## Features

### Text Normalization ✅
- Whitespace normalization (multiple spaces → single space)
- Quote normalization (curly quotes → straight quotes)
- Special character handling
- Leading/trailing whitespace removal

### Validation ✅
- Skips examples with empty question/context
- Validates required fields (id, question, context, answers)
- Checks non-empty text after normalization

### Configuration ✅
- Dataset selection via `config.yaml` (`datasets.active`)
- Sample limits for testing (`datasets.sample_limit`)
- Cache directory support (`paths.cache_dir`)
- Dataset-specific settings

### Experiment Preparation ✅
- `prepare_for_experiments()`: Converts unified schema to experiment format
- Returns: `queries`, `ground_truths`, `relevant_docs`, `corpus`
- Handles unanswerable questions (empty ground truth)

## Usage

### Load Dataset from Config
```python
from src.data import load_dataset_from_config, prepare_for_experiments
import yaml

# Load config
with open("config/config.yaml", 'r') as f:
    config = yaml.safe_load(f)

# Load dataset
examples = load_dataset_from_config(config, split="train")

# Prepare for experiments
queries, ground_truths, relevant_docs, corpus = prepare_for_experiments(examples)
```

### Load Dataset Directly
```python
from src.data import load_dataset

# Load SQuAD v2
examples = load_dataset("squad_v2", split="train", limit=100)

# Load Natural Questions
examples = load_dataset("natural_questions", split="train", limit=100)

# Load HotpotQA
examples = load_dataset("hotpotqa", split="train", limit=100)
```

## Verification

Run verification script:
```bash
python experiments/check_dataset_loading.py
```

**Output:**
- Loads dataset from config
- Validates schema (id, question, context, answers)
- Prints first 3 sample examples
- Reports statistics (total, answerable, unanswerable)
- Saves preview to `data/sample_preview.json`

## Configuration

### config.yaml
```yaml
datasets:
  active: "squad_v2"  # or "natural_questions", "hotpotqa"
  sample_limit: null  # Limit for testing (null = no limit)
  splits:
    train: "train"
    validation: "validation"
    test: "test"

paths:
  cache_dir: "~/.cache/huggingface/datasets/"
```

## Statistics

### Expected Dataset Sizes
- **SQuAD v2**: ~150K examples (train + validation)
- **Natural Questions**: ~300K examples (train + validation)
- **HotpotQA**: ~113K examples (train + validation)

### Non-Empty Answer Ratio
- **SQuAD v2**: <95% (includes unanswerable questions)
- **Natural Questions**: ≥95% (mostly answerable)
- **HotpotQA**: ≥95% (mostly answerable)

## Testing

### Quick Test
```bash
# Set sample_limit in config.yaml to 10
# Then run:
python experiments/check_dataset_loading.py
```

### Full Test
```bash
# Set sample_limit to null in config.yaml
# Then run:
python experiments/check_dataset_loading.py
```

## Integration with Experiments

Experiments 1-3 can now load datasets:

```python
from src.data import load_dataset_from_config, prepare_for_experiments
import yaml

# Load config
with open("config/config.yaml", 'r') as f:
    config = yaml.safe_load(f)

# Load dataset
examples = load_dataset_from_config(config, split="train")
queries, ground_truths, relevant_docs, corpus = prepare_for_experiments(examples)

# Use in experiments
from experiments.exp1_baseline import run_baseline_experiment
results = run_baseline_experiment(queries, ground_truths, relevant_docs, corpus, config)
```

## Notes

1. **Natural Questions**: Complex HTML structure - loader extracts text from tokens, skips HTML tags
2. **HotpotQA**: Combines multiple context paragraphs into single context string
3. **SQuAD v2**: Properly handles `is_impossible` flag for unanswerable questions
4. **Corpus Building**: Currently uses contexts from dataset. For production, use separate corpus (e.g., Wikipedia)
5. **Relevant Docs**: Currently uses context ID. For real RAG, use retrieval system to find relevant documents

## Next Steps

1. ✅ Dataset loading - COMPLETED
2. ⏳ Integrate with Experiments 1-3
3. ⏳ Build Wikipedia corpus index (21M passages)
4. ⏳ Implement iterative training data filtering (Factual Precision ≥ 0.85)
5. ⏳ Integrate W&B logging for dataset info

## Git Branch

- **Branch**: `feat/data-loading`
- **Commits**:
  1. `e3f2d84`: Initial implementation
  2. `f2c6f09`: Add example usage
  3. `3b4d2a7`: Fix naming conflict
  4. `cc35e67`: Update data README
  5. `c66cefb`: Fix verification script

## Acceptance Criteria ✅

- ✅ All three datasets load correctly
- ✅ Unified schema implemented
- ✅ Experiments can call load_dataset_from_config() without modification
- ✅ check_dataset_loading.py prints clean samples and passes
- ✅ IMPLEMENTATION_STATUS.md updated
- ✅ Branch created and pushed: `feat/data-loading`
- ✅ Clear commit messages

## Testing Results

Run `python experiments/check_dataset_loading.py` to verify:
- ✅ Schema validation passes
- ✅ Sample examples printed correctly
- ✅ Statistics reported accurately
- ✅ Preview saved to `data/sample_preview.json`

## Known Limitations

1. **Corpus Building**: Uses contexts from dataset, not separate Wikipedia corpus
2. **Relevant Docs**: Uses context IDs, not retrieved documents
3. **Natural Questions**: May need HTML parsing improvements for complex documents
4. **HotpotQA**: Multi-paragraph contexts combined into single string (may need chunking)

## Future Improvements

1. Add Wikipedia corpus loading (21M passages)
2. Implement document chunking for long contexts
3. Add support for FEVER dataset (for verification training)
4. Implement corpus indexing with FAISS
5. Add dataset versioning and commit hashes
6. Integrate with W&B for dataset logging

