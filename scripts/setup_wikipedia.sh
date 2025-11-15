#!/bin/bash
# Setup script for Wikipedia Language Model Training
# This script sets up the environment and directory structure

set -e  # Exit on error

echo "================================================"
echo "Wikipedia Language Model Training Setup"
echo "================================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" = "$REQUIRED_VERSION" ]; then 
    print_success "Python $PYTHON_VERSION (>= $REQUIRED_VERSION required)"
else
    print_error "Python $PYTHON_VERSION is too old. Please upgrade to Python >= $REQUIRED_VERSION"
    exit 1
fi

# Create directory structure
echo ""
echo "Creating directory structure..."

mkdir -p datasets/wikipedia/{raw,processed/{train,val,test},bilingual,analysis,checkpoints}
mkdir -p models/wikipedia/{base,finetuned_literary,xlm-bilingual}
mkdir -p results

print_success "Directory structure created"

# Create .gitkeep files
touch datasets/wikipedia/raw/.gitkeep
touch datasets/wikipedia/processed/train/.gitkeep
touch datasets/wikipedia/processed/val/.gitkeep
touch datasets/wikipedia/processed/test/.gitkeep
touch datasets/wikipedia/bilingual/.gitkeep
touch datasets/wikipedia/analysis/.gitkeep
touch models/wikipedia/.gitkeep

print_success "Placeholder files created"

# Check if virtual environment exists
echo ""
if [ -d "venv" ] || [ -d ".venv" ]; then
    print_warning "Virtual environment already exists. Skipping creation."
else
    echo "Creating virtual environment..."
    python3 -m venv venv
    print_success "Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "To activate the virtual environment, run:"
echo "  source venv/bin/activate"
echo ""

# Check if dependencies are installed
echo "Checking dependencies..."
if python3 -c "import transformers" 2>/dev/null; then
    print_success "Dependencies appear to be installed"
else
    print_warning "Dependencies not installed. Installing..."
    
    # Activate venv if it exists
    if [ -d "venv" ]; then
        source venv/bin/activate
    fi
    
    pip install --upgrade pip
    pip install -r requirements.txt
    
    print_success "Dependencies installed"
fi

# Check GPU availability
echo ""
echo "Checking GPU availability..."
if python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
    print_success "GPU detected: $GPU_NAME"
else
    print_warning "No GPU detected. Training will use CPU (slower)"
fi

# Check disk space
echo ""
echo "Checking disk space..."
AVAILABLE_SPACE=$(df -h . | awk 'NR==2 {print $4}')
print_success "Available disk space: $AVAILABLE_SPACE"

if [ $(df . | awk 'NR==2 {print $4}') -lt 52428800 ]; then  # 50GB in KB
    print_warning "Less than 50GB available. You may need more space for full Wikipedia dumps."
fi

# Create quick start script
echo ""
echo "Creating quick start script..."

cat > scripts/quickstart_wikipedia.sh << 'EOF'
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
EOF

chmod +x scripts/quickstart_wikipedia.sh
print_success "Quick start script created: scripts/quickstart_wikipedia.sh"

# Create example usage script
echo ""
echo "Creating example usage script..."

cat > examples/wikipedia_lm_example.py << 'EOF'
#!/usr/bin/env python3
"""
Example usage of Wikipedia Language Model
"""

from bilingual.modules.wikipedia_lm import load_model

def main():
    print("Loading Wikipedia Language Model...")
    model = load_model("models/wikipedia/base")
    
    print("\n" + "="*60)
    print("Example 1: Fill Masked Text")
    print("="*60)
    
    text = "আমি [MASK] খাই"
    print(f"Input: {text}")
    print("\nPredictions:")
    
    results = model.fill_mask(text, top_k=5)
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result['sequence']} (score: {result['score']:.4f})")
    
    print("\n" + "="*60)
    print("Example 2: Semantic Similarity")
    print("="*60)
    
    text1 = "আমি ভাত খাই"
    text2 = "আমি খাবার খাই"
    
    similarity = model.compute_similarity(text1, text2)
    print(f"Text 1: {text1}")
    print(f"Text 2: {text2}")
    print(f"Similarity: {similarity:.4f}")
    
    print("\n" + "="*60)
    print("Example 3: Get Embeddings")
    print("="*60)
    
    text = "আমি বাংলায় কথা বলি"
    embedding = model.get_sentence_embedding(text)
    print(f"Text: {text}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding (first 10 dims): {embedding[:10].tolist()}")
    
    print("\n" + "="*60)
    print("Example 4: Predict Next Word")
    print("="*60)
    
    text = "আমি ভাত"
    print(f"Input: {text}")
    print("\nPredictions:")
    
    predictions = model.predict_next_word(text, top_k=5)
    for i, pred in enumerate(predictions, 1):
        print(f"  {i}. {pred['word']} (score: {pred['score']:.4f})")

if __name__ == '__main__':
    main()
EOF

chmod +x examples/wikipedia_lm_example.py
print_success "Example script created: examples/wikipedia_lm_example.py"

# Print summary
echo ""
echo "================================================"
echo "Setup Complete!"
echo "================================================"
echo ""
echo "Directory structure:"
echo "  datasets/wikipedia/    - Wikipedia data"
echo "  models/wikipedia/      - Trained models"
echo "  scripts/               - Training scripts"
echo "  docs/                  - Documentation"
echo ""
echo "Next steps:"
echo ""
echo "1. Activate virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Run quick start (downloads, trains, evaluates):"
echo "   bash scripts/quickstart_wikipedia.sh"
echo ""
echo "3. Or follow manual steps:"
echo "   make -f Makefile.wiki help"
echo ""
echo "4. Read full documentation:"
echo "   docs/WIKIPEDIA_TRAINING_ROADMAP.md"
echo ""
echo "For help: make -f Makefile.wiki help"
echo ""
