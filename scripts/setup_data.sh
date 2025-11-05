#!/bin/bash

# Setup script for downloading and preparing data

echo "Setting up data for Self-Verification RAG project..."

# Create data directory
mkdir -p data

# Download spaCy model
echo "Downloading spaCy model..."
python -m spacy download en_core_web_sm

# TODO: Add dataset download commands
# For example:
# wget https://example.com/dataset.zip -O data/dataset.zip
# unzip data/dataset.zip -d data/

echo "Data setup complete!"

