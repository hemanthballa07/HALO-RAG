#!/bin/bash

# Script to download and cache models

echo "Downloading and caching models..."

# Create models directory
mkdir -p models

# Download models (will be cached by transformers)
python << EOF
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
import spacy

print("Downloading sentence transformer...")
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

print("Downloading cross-encoder...")
from sentence_transformers import CrossEncoder
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

print("Downloading FLAN-T5...")
tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-large')
model = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-large')

print("Downloading DeBERTa...")
tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-large')
model = AutoModelForSequenceClassification.from_pretrained('microsoft/deberta-v3-large')

print("Models downloaded successfully!")
EOF

echo "Model download complete!"

