#!/bin/bash
# Script to create commits in logical sequence

set -e

echo "Starting Git commit sequence..."
echo ""

# Check if git is initialized
if [ ! -d .git ]; then
    echo "Initializing git repository..."
    git init
fi

# Commit 1: Project structure and configuration
echo "Commit 1: Project structure and configuration"
git add .gitignore .gitattributes LICENSE README.md
git commit -m "chore: Initial project structure and configuration

- Add .gitignore with comprehensive ignore rules
- Add .gitattributes for consistent line endings
- Add LICENSE (Academic use)
- Add README.md with project overview"

# Commit 2: Retrieval module
echo ""
echo "Commit 2: Retrieval module"
git add src/retrieval/
git commit -m "feat: Add hybrid retrieval module (FAISS + BM25)

- Implement HybridRetriever with dense (FAISS) and sparse (BM25) retrieval
- Add fusion mechanism with configurable weights (0.6/0.4)
- Implement CrossEncoderReranker for document reranking
- Support for dense-only and sparse-only retrieval modes"

# Commit 3: Verification module
echo ""
echo "Commit 3: Verification module"
git add src/verification/
git commit -m "feat: Add entailment-based verification module

- Implement EntailmentVerifier using DeBERTa-v3-large
- Add ClaimExtractor with spaCy SVO extraction
- Support MNLI + FEVER fine-tuned models
- Configurable entailment threshold (τ)"

# Commit 4: Generator module
echo ""
echo "Commit 4: Generator module"
git add src/generator/
git commit -m "feat: Add FLAN-T5 generator with QLoRA support

- Implement FLANT5Generator for answer generation
- Add QLoRATrainer for efficient fine-tuning (r=16, 4-bit NF4)
- Support for various decoding strategies
- Integration with verification hints"

# Commit 5: Revision module
echo ""
echo "Commit 5: Revision module"
git add src/revision/
git commit -m "feat: Add adaptive revision strategies

- Implement AdaptiveRevisionStrategy
- Support re-retrieval with query expansion
- Add constrained generation with verified claims
- Implement claim-by-claim regeneration
- Adaptive strategy selection based on verification results"

# Commit 6: Evaluation module
echo ""
echo "Commit 6: Evaluation module"
git add src/evaluation/
git commit -m "feat: Add comprehensive evaluation framework

- Implement EvaluationMetrics with 12+ metrics
- Add StatisticalTester for significance testing
- Support for Recall@K, MRR, NDCG, Coverage
- Factual Precision, Hallucination Rate, Verified F1
- Bootstrap confidence intervals and t-tests"

# Commit 7: Pipeline module
echo ""
echo "Commit 7: Pipeline module"
git add src/pipeline/ src/__init__.py
git commit -m "feat: Add end-to-end Self-Verification RAG pipeline

- Implement SelfVerificationRAGPipeline
- Integrate all components (retrieval, reranking, generation, verification)
- Support adaptive revision strategies
- Comprehensive evaluation and metrics computation"

# Commit 8: Configuration
echo ""
echo "Commit 8: Configuration"
git add config/
git commit -m "feat: Add configuration files

- Add config.yaml with all pipeline parameters
- Configurable retrieval weights, thresholds, QLoRA params
- Experiment and evaluation settings"

# Commit 9: Experiments 1-4
echo ""
echo "Commit 9: Experiments 1-4"
git add experiments/exp1_baseline.py experiments/exp2_retrieval_comparison.py experiments/exp3_threshold_tuning.py experiments/exp4_revision_strategies.py
git commit -m "feat: Add experiments 1-4 (baseline, retrieval, threshold, revision)

- Experiment 1: Baseline comparison
- Experiment 2: Retrieval comparison (Dense vs Sparse vs Hybrid)
- Experiment 3: Threshold tuning with visualization
- Experiment 4: Revision strategies effectiveness"

# Commit 10: Experiments 5-8
echo ""
echo "Commit 10: Experiments 5-8"
git add experiments/exp5_decoding_strategies.py experiments/exp6_iterative_training.py experiments/exp7_ablation_study.py experiments/exp8_stress_test.py experiments/run_all_experiments.py
git commit -m "feat: Add experiments 5-8 (decoding, training, ablation, stress)

- Experiment 5: Decoding strategies comparison
- Experiment 6: Iterative training with self-improvement
- Experiment 7: Ablation study
- Experiment 8: Stress test on adversarial queries
- Add run_all_experiments.py script"

# Commit 11: Scripts and utilities
echo ""
echo "Commit 11: Scripts and utilities"
git add scripts/ prepare_git.sh
git commit -m "feat: Add setup scripts and utilities

- Add download_models.sh for model caching
- Add setup_data.sh for data preparation
- Add prepare_git.sh for Git setup"

# Commit 12: Documentation
echo ""
echo "Commit 12: Documentation"
git add METHODS.md PROJECT_SUMMARY.md INSTALL.md CONTRIBUTING.md
git commit -m "docs: Add comprehensive documentation

- Add METHODS.md with detailed methodology
- Add PROJECT_SUMMARY.md with project overview
- Add INSTALL.md with installation instructions
- Add CONTRIBUTING.md with contribution guidelines"

# Commit 13: Testing and notebooks
echo ""
echo "Commit 13: Testing and notebooks"
git add tests/ notebooks/
git commit -m "feat: Add test suite and Jupyter notebook

- Add test suite (imports, basic functionality, setup check)
- Add main_experiment_notebook.ipynb for interactive analysis
- Comprehensive test coverage"

# Commit 14: Data and results directories
echo ""
echo "Commit 14: Data and results directories"
git add data/README.md results/README.md
git commit -m "docs: Add data and results directory documentation

- Add README for data directory structure
- Add README for results directory format"

# Commit 15: Requirements and final setup
echo ""
echo "Commit 15: Requirements and final setup"
git add requirements.txt GIT_COMMIT_PLAN.md
git commit -m "chore: Add requirements and Git commit plan

- Add requirements.txt with all dependencies
- Add GIT_COMMIT_PLAN.md with commit strategy"

echo ""
echo "=========================================="
echo "All commits completed successfully!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Create a GitHub repository"
echo "2. Add remote: git remote add origin <your-repo-url>"
echo "3. Push: git push -u origin main"
echo ""
echo "To view commit history:"
echo "  git log --oneline"

