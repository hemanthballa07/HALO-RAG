"""
Logging utilities for experiments with W&B support.
"""

import os
import subprocess
import json
from datetime import datetime
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# W&B support (graceful degradation if not available)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("wandb not available. Logging to W&B will be disabled.")


def get_commit_hash() -> str:
    """
    Get current git commit hash.
    
    Returns:
        Commit hash string, or "unknown" if not available
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()[:8]  # First 8 characters
    except Exception:
        return "unknown"


def setup_wandb(
    project_name: str = "SelfVerifyRAG",
    run_name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    enabled: bool = True
) -> Optional[Any]:
    """
    Setup W&B logging.
    
    Args:
        project_name: W&B project name
        run_name: Run name
        config: Configuration dictionary to log
        enabled: Whether to enable W&B (default: True, but will disable if wandb not available)
    
    Returns:
        W&B run object, or None if not available/disabled
    """
    if not enabled or not WANDB_AVAILABLE:
        return None
    
    try:
        import wandb
        # Check if W&B is configured (optional - wandb can work without API key in offline mode)
        # api_key = os.getenv("WANDB_API_KEY")
        # if not api_key:
        #     logger.warning("WANDB_API_KEY not set. W&B logging disabled.")
        #     return None
        
        # Initialize W&B run
        run = wandb.init(
            project=project_name,
            name=run_name,
            config=config,
            reinit=True
        )
        
        logger.info(f"W&B logging initialized: {project_name}/{run_name}")
        return run
    except Exception as e:
        logger.warning(f"Failed to initialize W&B: {e}. Continuing without W&B.")
        return None


def log_metrics(
    metrics: Dict[str, float],
    step: Optional[int] = None,
    prefix: str = ""
) -> None:
    """
    Log metrics to W&B.
    
    Args:
        metrics: Dictionary of metrics to log
        step: Step number (optional)
        prefix: Prefix for metric names (optional)
    """
    if not WANDB_AVAILABLE:
        return
    
    try:
        import wandb
        wandb_run = wandb.run
        if wandb_run is None:
            return
        
        # Add prefix to metric names
        prefixed_metrics = {
            f"{prefix}{k}" if prefix else k: v
            for k, v in metrics.items()
        }
        
        if step is not None:
            wandb.log(prefixed_metrics, step=step)
        else:
            wandb.log(prefixed_metrics)
    except Exception as e:
        logger.warning(f"Failed to log metrics to W&B: {e}")


def log_metadata(
    dataset_name: str,
    split: str,
    sample_limit: Optional[int],
    commit_hash: str,
    timestamp: str
) -> None:
    """
    Log experiment metadata to W&B.
    
    Args:
        dataset_name: Dataset name
        split: Dataset split
        sample_limit: Sample limit (if any)
        commit_hash: Git commit hash
        timestamp: Timestamp
    """
    metadata = {
        "dataset_name": dataset_name,
        "split": split,
        "sample_limit": sample_limit,
        "commit_hash": commit_hash,
        "timestamp": timestamp
    }
    
    if WANDB_AVAILABLE:
        try:
            import wandb
            wandb_run = wandb.run
            if wandb_run is not None:
                wandb.config.update(metadata)
        except Exception as e:
            logger.warning(f"Failed to log metadata to W&B: {e}")
    
    # Also log to file
    logger.info(f"Experiment metadata: {json.dumps(metadata, indent=2)}")


def get_timestamp() -> str:
    """
    Get current timestamp as ISO format string.
    
    Returns:
        Timestamp string
    """
    return datetime.now().isoformat()

