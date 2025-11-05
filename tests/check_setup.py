#!/usr/bin/env python3
"""
Comprehensive setup check script
Tests project structure, file integrity, and dependencies
"""

import sys
import os
import importlib.util

print("=" * 70)
print("Self-Verification RAG Pipeline - Setup Check")
print("=" * 70)

# Track issues
issues = []
warnings = []

# Test 1: Check Python version
print("\n1. Python Version Check...")
python_version = sys.version_info
print(f"   Python {python_version.major}.{python_version.minor}.{python_version.micro}")
if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
    issues.append("Python 3.8+ required")
    print("   ✗ Python version too old")
else:
    print("   ✓ Python version OK")

# Test 2: Check project structure
print("\n2. Project Structure Check...")
required_dirs = {
    'src': 'Source code directory',
    'src/retrieval': 'Retrieval module',
    'src/verification': 'Verification module',
    'src/generator': 'Generator module',
    'src/revision': 'Revision module',
    'src/evaluation': 'Evaluation module',
    'src/pipeline': 'Pipeline module',
    'experiments': 'Experiment scripts',
    'config': 'Configuration files',
    'notebooks': 'Jupyter notebooks',
    'scripts': 'Setup scripts',
}

for dir_name, description in required_dirs.items():
    if os.path.exists(dir_name):
        print(f"   ✓ {dir_name}/ - {description}")
    else:
        issues.append(f"Missing directory: {dir_name}")
        print(f"   ✗ {dir_name}/ - NOT FOUND")

# Test 3: Check key files
print("\n3. Key Files Check...")
required_files = {
    'requirements.txt': 'Dependencies',
    'config/config.yaml': 'Configuration',
    'README.md': 'Documentation',
    'METHODS.md': 'Methods documentation',
    'QUICKSTART.md': 'Quick start guide',
    'src/__init__.py': 'Source package init',
    'src/retrieval/hybrid_retrieval.py': 'Hybrid retrieval',
    'src/verification/entailment_verifier.py': 'Entailment verifier',
    'src/generator/flan_t5_generator.py': 'FLAN-T5 generator',
    'src/pipeline/rag_pipeline.py': 'Main pipeline',
}

for file_name, description in required_files.items():
    if os.path.exists(file_name):
        print(f"   ✓ {file_name} - {description}")
    else:
        issues.append(f"Missing file: {file_name}")
        print(f"   ✗ {file_name} - NOT FOUND")

# Test 4: Check experiment scripts
print("\n4. Experiment Scripts Check...")
exp_files = [
    'exp1_baseline.py',
    'exp2_retrieval_comparison.py',
    'exp3_threshold_tuning.py',
    'exp4_revision_strategies.py',
    'exp5_decoding_strategies.py',
    'exp6_iterative_training.py',
    'exp7_ablation_study.py',
    'exp8_stress_test.py',
    'run_all_experiments.py',
]

for exp_file in exp_files:
    exp_path = os.path.join('experiments', exp_file)
    if os.path.exists(exp_path):
        print(f"   ✓ experiments/{exp_file}")
    else:
        issues.append(f"Missing experiment: {exp_file}")
        print(f"   ✗ experiments/{exp_file} - NOT FOUND")

# Test 5: Check dependencies
print("\n5. Dependencies Check...")
required_packages = {
    'torch': 'PyTorch',
    'transformers': 'Transformers',
    'sentence_transformers': 'Sentence Transformers',
    'numpy': 'NumPy',
    'pandas': 'Pandas',
    'yaml': 'PyYAML',
    'scipy': 'SciPy',
    'matplotlib': 'Matplotlib',
    'seaborn': 'Seaborn',
    'tqdm': 'tqdm',
}

missing_packages = []
for package, description in required_packages.items():
    try:
        # Handle different import names
        import_name = package
        if package == 'yaml':
            import_name = 'yaml'
        elif package == 'sentence_transformers':
            import_name = 'sentence_transformers'
        
        __import__(import_name)
        print(f"   ✓ {package} - {description}")
    except ImportError:
        missing_packages.append(package)
        warnings.append(f"Missing package: {package}")
        print(f"   ✗ {package} - NOT INSTALLED ({description})")

# Test 6: Check optional dependencies
print("\n6. Optional Dependencies Check...")
optional_packages = {
    'faiss': 'FAISS (for dense retrieval)',
    'rank_bm25': 'BM25 (for sparse retrieval)',
    'spacy': 'spaCy (for claim extraction)',
    'peft': 'PEFT (for QLoRA)',
    'bitsandbytes': 'BitsAndBytes (for 4-bit quantization)',
    'wandb': 'Weights & Biases (for experiment tracking)',
}

missing_optional = []
for package, description in optional_packages.items():
    try:
        import_name = package
        if package == 'rank_bm25':
            import_name = 'rank_bm25'
        elif package == 'bitsandbytes':
            import_name = 'bitsandbytes'
        __import__(import_name)
        print(f"   ✓ {package} - {description}")
    except ImportError:
        missing_optional.append(package)
        print(f"   ⚠ {package} - NOT INSTALLED ({description})")

# Test 7: Check configuration file
print("\n7. Configuration Check...")
try:
    import yaml
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    print("   ✓ config.yaml loads successfully")
    print(f"   ✓ Retrieval weights: {config['retrieval']['fusion']['dense_weight']}/{config['retrieval']['fusion']['sparse_weight']}")
    print(f"   ✓ Verification threshold (τ): {config['verification']['threshold']}")
except Exception as e:
    if 'yaml' in str(e) or 'No module named' in str(e):
        warnings.append("Cannot check config (PyYAML not installed)")
        print("   ⚠ Cannot check config (PyYAML not installed)")
    else:
        issues.append(f"Config error: {e}")
        print(f"   ✗ Config error: {e}")

# Test 8: Check code syntax (without importing)
print("\n8. Code Syntax Check...")
python_files = []
for root, dirs, files in os.walk('src'):
    for file in files:
        if file.endswith('.py'):
            python_files.append(os.path.join(root, file))

syntax_errors = []
for py_file in python_files[:10]:  # Check first 10 files
    try:
        with open(py_file, 'r') as f:
            compile(f.read(), py_file, 'exec')
        print(f"   ✓ {py_file} - syntax OK")
    except SyntaxError as e:
        syntax_errors.append(f"{py_file}: {e}")
        print(f"   ✗ {py_file} - syntax error: {e}")

if syntax_errors:
    issues.extend(syntax_errors)

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

if issues:
    print(f"\n✗ {len(issues)} ISSUES FOUND:")
    for issue in issues:
        print(f"   - {issue}")
else:
    print("\n✓ No critical issues found!")

if warnings:
    print(f"\n⚠ {len(warnings)} WARNINGS:")
    for warning in warnings:
        print(f"   - {warning}")

if missing_packages:
    print(f"\n📦 Missing {len(missing_packages)} required packages")
    print("   Install with: pip install -r requirements.txt")

if missing_optional:
    print(f"\n📦 Missing {len(missing_optional)} optional packages")
    print("   These are needed for full functionality")

print("\n" + "=" * 70)
print("Next Steps:")
print("=" * 70)
if missing_packages:
    print("1. Install dependencies:")
    print("   pip3 install -r requirements.txt")
    print("   # or")
    print("   python3 -m pip install -r requirements.txt")
    print()
print("2. Download spaCy model:")
print("   python3 -m spacy download en_core_web_sm")
print()
print("3. Load your dataset and run experiments:")
print("   python3 experiments/exp1_baseline.py")
print()
print("4. For interactive exploration:")
print("   jupyter notebook notebooks/main_experiment_notebook.ipynb")
print()

if not issues:
    print("✓ Project structure is complete and ready for use!")
else:
    print("⚠ Fix issues above before running experiments")

