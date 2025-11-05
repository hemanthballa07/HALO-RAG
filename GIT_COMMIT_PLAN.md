# Git Commit Strategy

This document outlines the recommended commit strategy for creating activity in GitHub.

## Commit Strategy

We'll make logical, incremental commits that show the development progress:

### Phase 1: Project Structure (Commit 1)
- Initial project structure
- Core directories and files
- README.md
- .gitignore

### Phase 2: Core Modules (Commits 2-7)
- Retrieval module (Commit 2)
- Verification module (Commit 3)
- Generator module (Commit 4)
- Revision module (Commit 5)
- Evaluation module (Commit 6)
- Pipeline module (Commit 7)

### Phase 3: Experiments (Commits 8-15)
- Experiment 1: Baseline (Commit 8)
- Experiment 2: Retrieval Comparison (Commit 9)
- Experiment 3: Threshold Tuning (Commit 10)
- Experiment 4: Revision Strategies (Commit 11)
- Experiment 5: Decoding Strategies (Commit 12)
- Experiment 6: Iterative Training (Commit 13)
- Experiment 7: Ablation Study (Commit 14)
- Experiment 8: Stress Test (Commit 15)
- Run all experiments script (Commit 16)

### Phase 4: Configuration & Scripts (Commit 17)
- Configuration files
- Setup scripts
- Documentation

### Phase 5: Documentation (Commit 18)
- Methods documentation
- Installation guide
- Project summary

### Phase 6: Testing & Notebooks (Commit 19)
- Test suite
- Jupyter notebook
- Final documentation

## Commands to Run

```bash
# Initialize (already done)
git init

# Add files in logical groups (see commits above)
git add <files>
git commit -m "<message>"

# Set up remote (when ready)
git remote add origin <your-repo-url>
git branch -M main
git push -u origin main
```

## Commit Messages Format

```
feat: Add retrieval module with hybrid FAISS+BM25

- Implement HybridRetriever class
- Add FAISS dense retrieval
- Add BM25 sparse retrieval
- Implement fusion with configurable weights
```

## Example Commit Sequence

```bash
# Commit 1: Project structure
git add .gitignore README.md LICENSE
git commit -m "chore: Initial project structure and configuration"

# Commit 2: Retrieval module
git add src/retrieval/
git commit -m "feat: Add hybrid retrieval module (FAISS + BM25)"

# Commit 3: Verification module
git add src/verification/
git commit -m "feat: Add entailment-based verification module"

# Continue with other commits...
```

