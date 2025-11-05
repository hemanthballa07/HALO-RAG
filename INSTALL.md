# Installation Guide

## Quick Installation

### Step 1: Install Python Dependencies

```bash
# Install all required packages
pip3 install -r requirements.txt

# Or if pip3 is not available
python3 -m pip install -r requirements.txt
```

### Step 2: Download spaCy Model

```bash
python3 -m spacy download en_core_web_sm
```

### Step 3: Verify Installation

```bash
python3 check_setup.py
```

## Troubleshooting

### If you get permission errors:

```bash
# Use --user flag
pip3 install --user -r requirements.txt

# Or create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
pip install -r requirements.txt
```

### If you're on a GPU cluster (HiperGator):

```bash
# For FAISS GPU support
pip3 install faiss-gpu

# For CUDA-enabled PyTorch
# Check PyTorch installation matches your CUDA version
# https://pytorch.org/get-started/locally/
```

### If you have issues with bitsandbytes:

```bash
# bitsandbytes may require specific CUDA versions
# If installation fails, you can still use the pipeline
# without QLoRA quantization (set use_qlora=False in config)
```

## After Installation

Run the setup check:

```bash
python3 check_setup.py
```

All items should show ✓ (checkmarks) when installation is complete.

