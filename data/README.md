# Data Directory

This directory contains the datasets used for the Self-Verification RAG project.

## Expected Datasets

- **Natural Questions**: Question-answering dataset
- **SQuAD**: Stanford Question Answering Dataset
- **TriviaQA**: Trivia question-answering dataset

## Dataset Format

Each dataset should contain:
- `queries`: List of query strings
- `ground_truths`: List of ground truth answers
- `contexts`: List of document contexts
- `relevant_docs`: List of relevant document IDs for each query

## Loading Datasets

Use the HuggingFace `datasets` library:
```python
from datasets import load_dataset
dataset = load_dataset("natural_questions", split="train")
```

## Data Preprocessing

1. Tokenize queries and contexts
2. Extract relevant document IDs
3. Create train/val/test splits
4. Build corpus index

