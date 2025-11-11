"""
Data loading module for HALO-RAG project.
Provides unified loaders for SQuAD v2, Natural Questions, and HotpotQA.
"""

from .loaders import (
    load_squad_v2,
    load_natural_questions,
    load_hotpotqa,
    load_dataset,
    load_dataset_from_config,
    prepare_for_experiments,
    normalize_text,
    validate_example
)

from .verified_collector import (
    collect_verified_data,
    save_verified_data,
    load_verified_data,
    compute_diversity_stats
)

__all__ = [
    "load_squad_v2",
    "load_natural_questions",
    "load_hotpotqa",
    "load_dataset",
    "load_dataset_from_config",
    "prepare_for_experiments",
    "normalize_text",
    "validate_example",
    "collect_verified_data",
    "save_verified_data",
    "load_verified_data",
    "compute_diversity_stats"
]

