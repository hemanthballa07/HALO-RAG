"""
CLI argument parsing for experiments.
"""

import argparse
from typing import Dict, Any, Optional
from pathlib import Path


def parse_experiment_args(
    description: str = "Experiment",
    default_config: str = "config/config.yaml"
) -> argparse.Namespace:
    """
    Parse common CLI arguments for experiments.
    
    Args:
        description: Experiment description
        default_config: Default config file path
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description=description)
    
    parser.add_argument(
        "--config",
        type=str,
        default=default_config,
        help="Path to config file"
    )
    
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "validation", "test"],
        help="Dataset split to use"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples (for testing). Overrides config."
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run with 20-50 samples for quick testing"
    )
    
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B logging"
    )
    
    return parser.parse_args()

