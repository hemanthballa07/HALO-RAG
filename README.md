# Self-Verification Chains for Hallucination-Free RAG

**CIS 6930: Special Topics in Large Language Models (Fall 2025)**  
**University of Florida**

## Overview

This repository implements a Self-Verification RAG pipeline that combines hybrid retrieval, cross-encoder reranking, fine-tuned generation, and entailment-based factual verification to achieve hallucination-free retrieval-augmented generation.

## Architecture

1. **Hybrid Retrieval**: FAISS (dense) + BM25 (sparse) fusion (0.6/0.4)
2. **Reranking**: DeBERTa-v3-base cross-encoder on MS MARCO
3. **Generation**: FLAN-T5-Large fine-tuned with QLoRA (r=16, 4-bit NF4)
4. **Verification**: DeBERTa-v3-large entailment model (MNLI + FEVER) with spaCy SVO extraction
5. **Revision**: Adaptive strategies (re-retrieval, constrained generation, claim-by-claim)

## Key Metrics

- **Recall@20** ≥ 0.95
- **Coverage** ≥ 0.90
- **Factual Precision** ≥ 0.90
- **Verified F1** ≥ 0.52
- **Hallucination Rate** ≤ 0.10

## Project Structure

```
HALO-RAG/
├── src/
│   ├── retrieval/
│   │   ├── hybrid_retrieval.py
│   │   ├── reranker.py
│   │   └── __init__.py
│   ├── verification/
│   │   ├── entailment_verifier.py
│   │   ├── claim_extractor.py
│   │   └── __init__.py
│   ├── generator/
│   │   ├── flan_t5_generator.py
│   │   ├── qlora_trainer.py
│   │   └── __init__.py
│   ├── revision/
│   │   ├── adaptive_strategies.py
│   │   └── __init__.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── statistical_testing.py
│   │   └── __init__.py
│   └── pipeline/
│       ├── rag_pipeline.py
│       └── __init__.py
├── experiments/
│   ├── exp1_baseline.py
│   ├── exp2_retrieval_comparison.py
│   ├── exp3_threshold_tuning.py
│   ├── exp4_revision_strategies.py
│   ├── exp5_decoding_strategies.py
│   ├── exp6_iterative_training.py
│   ├── exp7_ablation_study.py
│   ├── exp8_stress_test.py
│   └── run_all_experiments.py
├── notebooks/
│   ├── main_experiment_notebook.ipynb
│   └── analysis_visualization.ipynb
├── config/
│   ├── config.yaml
│   └── model_configs.yaml
├── data/
│   └── README.md
├── results/
│   └── README.md
├── scripts/
│   ├── setup_data.sh
│   └── download_models.sh
└── requirements.txt
```

## Quick Start

1. **Install dependencies**:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

2. **Download models and data**:
```bash
bash scripts/download_models.sh
bash scripts/setup_data.sh
```

3. **Run experiments**:
```bash
python experiments/run_all_experiments.py
```

## Experiments

The project includes 8 comprehensive experiments:
1. **Baseline Comparison**: Standard RAG vs Self-Verification RAG
2. **Retrieval Comparison**: Dense vs Sparse vs Hybrid
3. **Threshold Tuning**: Optimal τ for entailment verification
4. **Revision Strategies**: Effectiveness of adaptive revision
5. **Decoding Strategies**: Greedy vs Beam vs Nucleus sampling
6. **Iterative Training**: Self-improvement through fine-tuning loops
7. **Ablation Study**: Component-wise contribution analysis
8. **Stress Test**: Performance on adversarial queries

## Citation

If you use this code, please cite:
```
@article{halo_rag_2025,
  title={Self-Verification Chains for Hallucination-Free Retrieval-Augmented Generation},
  author={[Your Team]},
  journal={NeurIPS 2025},
  year={2025}
}
```

## License

Academic use only - University of Florida CIS 6930

