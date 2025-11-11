"""
Dataset loaders for SQuAD v2, Natural Questions, and HotpotQA.
Normalizes all datasets to a unified schema.
"""

import os
import re
from typing import List, Dict, Any, Optional, Tuple
from datasets import load_dataset as hf_load_dataset, Dataset
import logging

logger = logging.getLogger(__name__)

# Unified schema for all datasets
UNIFIED_SCHEMA = {
    "id": str,
    "question": str,
    "context": str,
    "answers": List[str]  # Empty list if unanswerable
}


def normalize_text(text: str) -> str:
    """
    Clean and normalize text.
    
    Args:
        text: Input text
    
    Returns:
        Normalized text
    """
    if not text:
        return ""
    
    # Strip whitespace
    text = text.strip()
    
    # Normalize quotes (convert curly quotes to straight quotes)
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    
    # Normalize whitespace (multiple spaces to single space)
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace again
    text = text.strip()
    
    return text


def validate_example(example: Dict[str, Any]) -> bool:
    """
    Validate that an example has required fields.
    
    Args:
        example: Example dictionary
    
    Returns:
        True if valid, False otherwise
    """
    # Check for required fields
    if "id" not in example:
        return False
    
    if "question" not in example or not example["question"]:
        return False
    
    if "context" not in example or not example["context"]:
        return False
    
    # Answers can be empty list (for unanswerable questions)
    if "answers" not in example:
        return False
    
    # Normalize and check if question/context are non-empty after normalization
    question = normalize_text(example["question"])
    context = normalize_text(example["context"])
    
    if not question or not context:
        return False
    
    return True


def load_squad_v2(
    split: str = "train",
    limit: Optional[int] = None,
    cache_dir: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Load SQuAD v2.0 dataset.
    
    Args:
        split: Dataset split ("train", "validation", "test")
        limit: Limit number of examples (for testing)
        cache_dir: Cache directory for datasets
    
    Returns:
        List of normalized examples
    """
    logger.info(f"Loading SQuAD v2.0 dataset (split: {split})...")
    
    try:
        dataset = hf_load_dataset("squad_v2", cache_dir=cache_dir, split=split)
    except Exception as e:
        logger.error(f"Failed to load SQuAD v2: {e}")
        raise
    
    normalized_examples = []
    
    for item in dataset:
        # Extract fields
        example_id = item.get("id", "")
        question = normalize_text(item.get("question", ""))
        context = normalize_text(item.get("context", ""))
        
        # Extract answers
        answers = []
        if "answers" in item and item["answers"]:
            answer_texts = item["answers"].get("text", [])
            answers = [normalize_text(ans) for ans in answer_texts if ans]
        
        # Check if question is answerable (SQuAD v2 has unanswerable questions)
        # If answers is empty, question is unanswerable
        is_impossible = item.get("is_impossible", False)
        if is_impossible:
            answers = []
        
        # Create normalized example
        example = {
            "id": example_id,
            "question": question,
            "context": context,
            "answers": answers
        }
        
        # Validate and add
        if validate_example(example):
            normalized_examples.append(example)
            
            # Apply limit if specified
            if limit and len(normalized_examples) >= limit:
                break
    
    logger.info(f"Loaded {len(normalized_examples)} examples from SQuAD v2.0 ({split})")
    return normalized_examples


def load_natural_questions(
    split: str = "train",
    limit: Optional[int] = None,
    cache_dir: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Load Natural Questions dataset.
    
    Note: Natural Questions has a complex structure with HTML documents.
    This loader extracts text from the document tokens.
    
    Args:
        split: Dataset split ("train", "validation")
        limit: Limit number of examples (for testing)
        cache_dir: Cache directory for datasets
    
    Returns:
        List of normalized examples
    """
    logger.info(f"Loading Natural Questions dataset (split: {split})...")
    
    try:
        # Natural Questions structure is complex - load the full dataset
        dataset = hf_load_dataset("natural_questions", cache_dir=cache_dir, split=split)
    except Exception as e:
        logger.error(f"Failed to load Natural Questions: {e}")
        raise
    
    normalized_examples = []
    
    for idx, item in enumerate(dataset):
        # Natural Questions structure:
        # - question: dict with "text" field (list of tokens)
        # - document: dict with "html", "title", "url", "tokens" (list of token dicts)
        # - annotations: list with "short_answers", "long_answer"
        
        # Extract question text from tokens
        question_text = ""
        if "question" in item:
            question_obj = item["question"]
            if isinstance(question_obj, dict):
                # Question is a dict with "text" field containing list of tokens
                question_tokens = question_obj.get("tokens", [])
                if question_tokens:
                    # Extract text from tokens
                    question_words = []
                    for token in question_tokens:
                        if isinstance(token, dict):
                            question_words.append(token.get("token", ""))
                        elif isinstance(token, str):
                            question_words.append(token)
                    question_text = " ".join(question_words)
                else:
                    # Fallback: try "text" field
                    question_text = question_obj.get("text", "")
            elif isinstance(question_obj, str):
                question_text = question_obj
        
        question = normalize_text(question_text)
        
        # Extract context from document tokens
        context_text = ""
        if "document" in item:
            doc = item["document"]
            
            # Extract text from document tokens
            doc_tokens = doc.get("tokens", [])
            if doc_tokens:
                doc_words = []
                for token in doc_tokens:
                    if isinstance(token, dict):
                        token_text = token.get("token", "")
                        # Skip HTML tags and special tokens
                        if token_text and not token_text.startswith("<") and not token_text.startswith("&"):
                            doc_words.append(token_text)
                    elif isinstance(token, str):
                        if not token.startswith("<") and not token.startswith("&"):
                            doc_words.append(token)
                context_text = " ".join(doc_words)
            
            # Fallback: try title
            if not context_text:
                title = doc.get("title", "")
                if title:
                    context_text = title
        
        context = normalize_text(context_text)
        
        # Extract answers (short answers)
        answers = []
        if "annotations" in item and item["annotations"]:
            annotations = item["annotations"]
            if isinstance(annotations, list) and len(annotations) > 0:
                # Get first annotation (typically there's one per question)
                ann = annotations[0]
                if "short_answers" in ann and ann["short_answers"]:
                    short_answers = ann["short_answers"]
                    if isinstance(short_answers, list) and len(short_answers) > 0:
                        # Extract text from short answer tokens
                        for sa in short_answers:
                            if isinstance(sa, dict):
                                # Short answer has start_token and end_token indices
                                # We need to extract the actual text from document tokens
                                start_token = sa.get("start_token", -1)
                                end_token = sa.get("end_token", -1)
                                
                                if start_token >= 0 and end_token >= 0 and "document" in item:
                                    doc_tokens = item["document"].get("tokens", [])
                                    if doc_tokens and end_token < len(doc_tokens):
                                        answer_words = []
                                        for i in range(start_token, end_token + 1):
                                            token = doc_tokens[i]
                                            if isinstance(token, dict):
                                                answer_words.append(token.get("token", ""))
                                            elif isinstance(token, str):
                                                answer_words.append(token)
                                        answer_text = " ".join(answer_words)
                                        if answer_text:
                                            answers.append(normalize_text(answer_text))
        
        # Create example ID
        example_id = item.get("id", f"nq_{split}_{idx}")
        if not example_id:
            example_id = f"nq_{split}_{idx}"
        
        # Create normalized example
        example = {
            "id": str(example_id),
            "question": question,
            "context": context,
            "answers": answers
        }
        
        # Validate and add
        if validate_example(example):
            normalized_examples.append(example)
            
            # Apply limit if specified
            if limit and len(normalized_examples) >= limit:
                break
    
    logger.info(f"Loaded {len(normalized_examples)} examples from Natural Questions ({split})")
    return normalized_examples


def load_hotpotqa(
    split: str = "train",
    limit: Optional[int] = None,
    cache_dir: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Load HotpotQA dataset.
    
    Args:
        split: Dataset split ("train", "validation", "test")
        limit: Limit number of examples (for testing)
        cache_dir: Cache directory for datasets
    
    Returns:
        List of normalized examples
    """
    logger.info(f"Loading HotpotQA dataset (split: {split})...")
    
    try:
        dataset = hf_load_dataset("hotpot_qa", "fullwiki", cache_dir=cache_dir, split=split)
    except Exception as e:
        logger.error(f"Failed to load HotpotQA: {e}")
        # Try alternative: "distractor" split
        try:
            logger.info("Trying HotpotQA distractor split...")
            dataset = hf_load_dataset("hotpot_qa", "distractor", cache_dir=cache_dir, split=split)
        except Exception as e2:
            logger.error(f"Failed to load HotpotQA distractor: {e2}")
            raise
    
    normalized_examples = []
    
    for item in dataset:
        # HotpotQA structure:
        # - question: str
        # - context: list of dicts with "sentences", "title"
        # - answer: str or list[str]
        
        # Extract question
        question = normalize_text(item.get("question", ""))
        
        # Extract context (combine all context paragraphs)
        context_parts = []
        if "context" in item:
            context_list = item["context"]
            if isinstance(context_list, list):
                for ctx in context_list:
                    if isinstance(ctx, dict):
                        title = ctx.get("title", "")
                        sentences = ctx.get("sentences", [])
                        if sentences:
                            paragraph = " ".join(sentences)
                            if title:
                                paragraph = f"{title}: {paragraph}"
                            context_parts.append(paragraph)
        
        context = normalize_text(" ".join(context_parts))
        
        # Extract answers
        answers = []
        if "answer" in item:
            answer = item["answer"]
            if isinstance(answer, str):
                answers = [normalize_text(answer)]
            elif isinstance(answer, list):
                answers = [normalize_text(ans) for ans in answer if ans]
        
        # Create example ID
        example_id = item.get("id", item.get("_id", ""))
        
        # Create normalized example
        example = {
            "id": example_id,
            "question": question,
            "context": context,
            "answers": answers
        }
        
        # Validate and add
        if validate_example(example):
            normalized_examples.append(example)
            
            # Apply limit if specified
            if limit and len(normalized_examples) >= limit:
                break
    
    logger.info(f"Loaded {len(normalized_examples)} examples from HotpotQA ({split})")
    return normalized_examples


def load_dataset(
    dataset_name: str,
    split: str = "train",
    limit: Optional[int] = None,
    cache_dir: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Load dataset by name with unified schema.
    
    Args:
        dataset_name: Dataset name ("squad_v2", "natural_questions", "hotpotqa")
        split: Dataset split ("train", "validation", "test")
        limit: Limit number of examples (for testing)
        cache_dir: Cache directory for datasets
    
    Returns:
        List of normalized examples with unified schema
    """
    dataset_name_lower = dataset_name.lower()
    
    if dataset_name_lower in ["squad_v2", "squad-v2", "squad"]:
        return load_squad_v2(split=split, limit=limit, cache_dir=cache_dir)
    elif dataset_name_lower in ["natural_questions", "natural-questions", "nq"]:
        return load_natural_questions(split=split, limit=limit, cache_dir=cache_dir)
    elif dataset_name_lower in ["hotpotqa", "hotpot_qa", "hotpot"]:
        return load_hotpotqa(split=split, limit=limit, cache_dir=cache_dir)
    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Supported datasets: squad_v2, natural_questions, hotpotqa"
        )


def prepare_for_experiments(
    examples: List[Dict[str, Any]]
) -> Tuple[List[str], List[str], List[List[int]], List[str]]:
    """
    Prepare normalized examples for experiments.
    
    Converts unified schema to format expected by experiments:
    - queries: List[str]
    - ground_truths: List[str] (first answer or empty string)
    - relevant_docs: List[List[int]] (placeholder, will be filled by retrieval)
    - corpus: List[str] (contexts)
    
    Args:
        examples: List of normalized examples
    
    Returns:
        Tuple of (queries, ground_truths, relevant_docs, corpus)
    """
    queries = []
    ground_truths = []
    relevant_docs = []
    corpus = []
    
    # Build corpus from contexts (deduplicate)
    context_to_id = {}
    corpus_ids = []
    
    for example in examples:
        # Extract query
        queries.append(example["question"])
        
        # Extract ground truth (first answer or empty string)
        if example["answers"] and len(example["answers"]) > 0:
            ground_truths.append(example["answers"][0])
        else:
            ground_truths.append("")  # Unanswerable question
        
        # Extract context and add to corpus
        context = example["context"]
        if context not in context_to_id:
            context_to_id[context] = len(corpus)
            corpus.append(context)
            corpus_ids.append([len(corpus) - 1])
        else:
            doc_id = context_to_id[context]
            corpus_ids.append([doc_id])
        
        # For now, relevant_docs is just the context ID
        # In real RAG, this would be multiple relevant document IDs
        relevant_docs.append(corpus_ids[-1])
    
    return queries, ground_truths, relevant_docs, corpus


def load_dataset_from_config(config: Dict[str, Any], split: str = "train") -> List[Dict[str, Any]]:
    """
    Load dataset from configuration.
    
    Args:
        config: Configuration dictionary
        split: Dataset split ("train", "validation", "test")
    
    Returns:
        List of normalized examples
    """
    # Get dataset configuration
    dataset_config = config.get("datasets", {})
    active_dataset = dataset_config.get("active", "squad_v2")
    sample_limit = dataset_config.get("sample_limit")
    
    # Get cache directory
    paths_config = config.get("paths", {})
    cache_dir = paths_config.get("cache_dir", None)
    if cache_dir and cache_dir.startswith("~"):
        import os
        cache_dir = os.path.expanduser(cache_dir)
    
    # Get split name from config
    splits_config = dataset_config.get("splits", {})
    split_name = splits_config.get(split, split)
    
    # Load dataset
    logger.info(f"Loading dataset '{active_dataset}' from config (split: {split_name})...")
    examples = load_dataset(
        dataset_name=active_dataset,
        split=split_name,
        limit=sample_limit,
        cache_dir=cache_dir
    )
    
    logger.info(f"Loaded {len(examples)} examples from {active_dataset} ({split_name})")
    return examples

