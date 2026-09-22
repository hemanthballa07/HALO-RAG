"""
Create RESULTS_LOCK.md with reproducibility information.
"""

import sys
import os
from pathlib import Path
import json
import yaml
from datetime import datetime
import subprocess
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import get_commit_hash, get_timestamp


def get_git_info():
    """Get git repository information."""
    try:
        commit_hash = get_commit_hash()
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], 
                                        cwd=project_root, text=True).strip()
        remote_url = subprocess.check_output(["git", "config", "--get", "remote.origin.url"],
                                            cwd=project_root, text=True).strip()
        return {
            "commit_hash": commit_hash,
            "branch": branch,
            "remote_url": remote_url
        }
    except Exception as e:
        return {"error": str(e)}


def get_config_info(config_path: str = "config/config.yaml"):
    """Get configuration information."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        return {"error": str(e)}


def get_faiss_index_metadata(index_path: str = None):
    """Get FAISS index metadata if available."""
    # Check for FAISS index files
    index_metadata = {
        "index_path": index_path or "Not specified",
        "exists": False,
        "size_bytes": 0,
        "dimension": None,
        "num_vectors": None
    }
    
    if index_path and os.path.exists(index_path):
        try:
            index_metadata["exists"] = True
            index_metadata["size_bytes"] = os.path.getsize(index_path)
            # Try to load index and get metadata
            try:
                import faiss
                index = faiss.read_index(index_path)
                index_metadata["dimension"] = index.d
                index_metadata["num_vectors"] = index.ntotal
            except Exception:
                pass
        except Exception as e:
            index_metadata["error"] = str(e)
    
    return index_metadata


def get_verified_data_snapshots():
    """Get verified data snapshot paths."""
    verified_dir = "data/verified"
    snapshots = []
    
    if os.path.exists(verified_dir):
        for filename in os.listdir(verified_dir):
            if filename.endswith(".jsonl"):
                filepath = os.path.join(verified_dir, filename)
                snapshots.append({
                    "filename": filename,
                    "path": filepath,
                    "size_bytes": os.path.getsize(filepath) if os.path.exists(filepath) else 0,
                    "modified": datetime.fromtimestamp(os.path.getmtime(filepath)).isoformat() 
                               if os.path.exists(filepath) else None
                })
    
    return snapshots


def create_results_lock(output_path: str = "RESULTS_LOCK.md", 
                       config_path: str = "config/config.yaml",
                       optimal_tau: float = 0.75,
                       seeds: Optional[list[int]] = None,
                       dataset: str = "squad_v2",
                       split: str = "validation"):
    """
    Create RESULTS_LOCK.md with reproducibility information.
    
    Args:
        output_path: Output file path
        config_path: Config file path
        optimal_tau: Optimal threshold from Exp8
        seeds: Random seeds used
        dataset: Dataset name
        split: Dataset split
    """
    if seeds is None:
        seeds = [42, 123, 456]

    git_info = get_git_info()
    config = get_config_info(config_path)
    verified_snapshots = get_verified_data_snapshots()
    
    # Get FAISS index metadata (if available)
    faiss_index_path = config.get("paths", {}).get("faiss_index") or "data/faiss_index.bin"
    faiss_metadata = get_faiss_index_metadata(faiss_index_path)
    
    # Create RESULTS_LOCK.md content
    content = f"""# Results Lock Document

This document contains all information necessary to reproduce the final HALO-RAG experimental results.

## Timestamp

Generated: {get_timestamp()}

## Git Repository Information

- **Commit Hash**: {git_info.get('commit_hash', 'N/A')}
- **Branch**: {git_info.get('branch', 'N/A')}
- **Remote URL**: {git_info.get('remote_url', 'N/A')}

## Dataset Configuration

- **Dataset**: {dataset}
- **Split**: {split}
- **Sample Limit**: {config.get('datasets', {}).get('sample_limit', 'None (full dataset)')}

## Experimental Configuration

### Optimal Threshold (τ)

- **Value**: {optimal_tau}
- **Source**: Experiment 8 (Stress Testing & Pareto Frontier)
- **Rationale**: Optimal balance between Factual Precision and Verified F1

### Random Seeds

- **Seeds Used**: {seeds}
- **Purpose**: Statistical robustness across multiple runs
- **Experiments**: Exp1-8, Human Eval

### Verification Configuration

- **Entailment Model**: {config.get('verification', {}).get('entailment_model', 'N/A')}
- **Threshold (τ)**: {optimal_tau}
- **Accept Min**: {config.get('verification', {}).get('accept_min', 'N/A')}

### Retrieval Configuration

- **Dense Model**: {config.get('retrieval', {}).get('dense', {}).get('model_name', 'N/A')}
- **Reranker Model**: {config.get('retrieval', {}).get('reranker', {}).get('model_name', 'N/A')}
- **Dense Weight**: {config.get('retrieval', {}).get('fusion', {}).get('dense_weight', 'N/A')}
- **Sparse Weight**: {config.get('retrieval', {}).get('fusion', {}).get('sparse_weight', 'N/A')}
- **Top K Retrieve**: {config.get('retrieval', {}).get('fusion', {}).get('top_k', 'N/A')}
- **Top K Rerank**: {config.get('retrieval', {}).get('reranker', {}).get('top_k', 'N/A')}

### Generation Configuration

- **Model**: {config.get('generation', {}).get('model_name', 'N/A')}
- **Max New Tokens**: {config.get('generation', {}).get('max_new_tokens', 'N/A')}
- **Temperature**: {config.get('generation', {}).get('temperature', 'N/A')}
- **QLoRA Enabled**: {config.get('generation', {}).get('qlora', {}).get('training_enabled', 'N/A')}

### Revision Configuration

- **Max Iterations**: {config.get('revision', {}).get('max_iterations', 'N/A')}
- **Strategies**: {', '.join(config.get('revision', {}).get('strategies', []))}

## FAISS Index Metadata

- **Index Path**: {faiss_metadata.get('index_path', 'N/A')}
- **Exists**: {faiss_metadata.get('exists', False)}
- **Size (bytes)**: {faiss_metadata.get('size_bytes', 0):,}
- **Dimension**: {faiss_metadata.get('dimension', 'N/A')}
- **Number of Vectors**: {faiss_metadata.get('num_vectors', 'N/A')}

## Verified Data Snapshots

The following verified data snapshots were used for iterative training (Exp6):

"""
    
    if verified_snapshots:
        for snapshot in verified_snapshots:
            content += f"""### {snapshot['filename']}

- **Path**: {snapshot['path']}
- **Size**: {snapshot['size_bytes']:,} bytes
- **Modified**: {snapshot['modified']}

"""
    else:
        content += "No verified data snapshots found.\n\n"
    
    artifacts = {
        "Experiment 1: Baseline Comparison": ["results/metrics/exp1_baseline.json"],
        "Experiment 2: Retrieval Comparison": ["results/metrics/exp2_retrieval.csv"],
        "Experiment 3: Threshold Tuning": ["results/metrics/exp3_threshold_sweep.csv"],
        "Experiment 4: Revision Strategies": [
            "results/metrics/exp4_revision_strategies.json"
        ],
        "Experiment 5: Self-Consistency Decoding": [
            "results/metrics/exp5_self_consistency.json"
        ],
        "Experiment 6: Iterative Fine-Tuning": [
            "results/metrics/exp6_iterative_training.csv"
        ],
        "Experiment 7: Ablation Study": ["results/metrics/exp7_ablation.csv"],
        "Experiment 8: Stress Testing": ["results/metrics/exp8_stress.json"],
        "Human Evaluation": [
            "results/metrics/human_eval_agreement.json",
            "results/human_eval/human_eval_samples.csv",
        ],
    }

    content += "## Result Artifacts\n\n"
    for label, paths in artifacts.items():
        missing_paths = [path for path in paths if not os.path.exists(path)]
        status = "Complete" if not missing_paths else "Incomplete"
        content += f"### {label}\n\n- **Status**: {status}\n"
        content += "- **Expected output**: " + ", ".join(f"`{path}`" for path in paths) + "\n"
        if missing_paths:
            content += "- **Missing**: " + ", ".join(f"`{path}`" for path in missing_paths) + "\n"
        content += "\n"

    summary_path = "results/metrics/final_summary.csv"
    content += f"""## Final Summary Metrics

- **Output**: `{summary_path}`
- **Status**: {'Present' if os.path.exists(summary_path) else 'Missing'}
- **Requested seeds**: {seeds}

## Reproducibility Instructions

To reproduce these results:

1. **Checkout the commit**:
   ```bash
   git checkout {git_info.get('commit_hash', 'N/A')}
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Validate the environment**:
   ```bash
   python scripts/check_setup.py
   ```

4. **Run experiments**:
   ```bash
   python experiments/run_final_experiments.py --seeds 42 123 456 --split validation
   ```

5. **Verify results**:
   - Check `results/metrics/final_summary.csv` for aggregated metrics
   - Check `results/figures/final/` for key plots
   - Compare with metrics in this document

## Notes

- This file records requested settings and observed artifacts; it does not infer that a
  run completed when its files are absent.
- Confirm the seed count in `final_aggregated_results.json` before reporting multi-seed
  statistics.
- Experiment 6 requires CUDA and `bitsandbytes`.

## Contact

For questions about reproducibility, please refer to the main README.md or open an issue on GitHub.

---
*This document was automatically generated on {get_timestamp()}*
"""
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write(content)
    
    print(f"✓ Created RESULTS_LOCK.md: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create RESULTS_LOCK.md")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                       help="Path to config file")
    parser.add_argument("--output", type=str, default="RESULTS_LOCK.md",
                       help="Output file path")
    parser.add_argument("--tau", type=float, default=0.75,
                       help="Optimal threshold")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456],
                       help="Random seeds")
    parser.add_argument("--dataset", type=str, default="squad_v2",
                       help="Dataset name")
    parser.add_argument("--split", type=str, default="validation",
                       help="Dataset split")
    
    args = parser.parse_args()
    
    create_results_lock(
        output_path=args.output,
        config_path=args.config,
        optimal_tau=args.tau,
        seeds=args.seeds,
        dataset=args.dataset,
        split=args.split
    )
