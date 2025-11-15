#!/bin/bash
# Quick start script for Wikipedia training

set -e

echo "Wikipedia Training Quick Start"
echo "=============================="
echo ""

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Step 1: Download
echo "Step 1: Downloading Bangla Wikipedia..."
python scripts/download_wiki.py --lang bn --output datasets/wikipedia/raw

# Step 2: Preprocess
echo ""
echo "Step 2: Preprocessing..."
python scripts/preprocess_wiki.py \
    --input datasets/wikipedia/raw/bnwiki-latest-pages-articles.xml.bz2 \
    --output datasets/wikipedia/processed \
    --lang bn

# Step 3: Analyze
echo ""
echo "Step 3: Analyzing dataset..."
python scripts/analyze_wiki_dataset.py \
    --input datasets/wikipedia/processed/train/bn_train.txt \
    --output datasets/wikipedia/analysis

# Step 4: Train (test with 1 epoch)
echo ""
echo "Step 4: Training model (1 epoch test)..."
python scripts/train_wiki_lm.py \
    --data datasets/wikipedia/processed \
    --model ai4bharat/indic-bert \
    --output models/wikipedia/test \
    --epochs 1 \
    --batch-size 8

# Step 5: Evaluate
echo ""
echo "Step 5: Evaluating model..."
python scripts/evaluate_wiki_lm.py \
    --model models/wikipedia/test \
    --data datasets/wikipedia/processed/test/bn_test.txt \
    --output results/wikipedia_eval.json

echo ""
echo "Quick start complete!"
echo "For full training, see: docs/WIKIPEDIA_TRAINING_ROADMAP.md"
