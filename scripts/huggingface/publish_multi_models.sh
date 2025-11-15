#!/bin/bash
# Multi-Model Publishing Pipeline for KothaGPT Bilingual
# Publishes all models to Hugging Face Hub
# Usage: bash scripts/huggingface/publish_multi_models.sh [--dry-run] [--username YOUR_USERNAME]

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

print_success() { echo -e "${GREEN}✓ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠ $1${NC}"; }
print_error() { echo -e "${RED}✗ $1${NC}"; }
print_info() { echo -e "${BLUE}ℹ $1${NC}"; }

# Parse arguments
DRY_RUN=false
HF_USERNAME="KothaGPT"

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --username)
            HF_USERNAME="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "================================================"
echo "KothaGPT Multi-Model Publishing Pipeline"
echo "================================================"
echo ""
echo "Configuration:"
echo "  HF Username: $HF_USERNAME"
echo "  Dry Run: $DRY_RUN"
echo ""

# Define models to publish
declare -A MODELS=(
    ["bilingual-lm"]="models/bilingual-lm:bilingual-language-model:causal-lm"
    ["literary-lm"]="models/literary-lm:literary-language-model:causal-lm"
    ["readability-classifier"]="models/readability-classifier:readability-classifier:text-classification"
    ["poetic-meter-detector"]="models/poetic-meter-detector:poetic-meter-detector:text-classification"
    ["metaphor-simile-detector"]="models/metaphor-simile-detector:metaphor-simile-detector:text-classification"
    ["style-transfer-gpt"]="models/style-transfer-gpt:style-transfer-gpt:text-generation"
    ["sentiment-tone-classifier"]="models/sentiment-tone-classifier:sentiment-tone-classifier:text-classification"
    ["cross-lingual-embed"]="models/cross-lingual-embed:cross-lingual-embeddings:feature-extraction"
    ["named-entity-recognizer"]="models/named-entity-recognizer:named-entity-recognizer:token-classification"
)

# Statistics
TOTAL_MODELS=${#MODELS[@]}
SUCCESS_COUNT=0
FAILED_COUNT=0
SKIPPED_COUNT=0

declare -a FAILED_MODELS
declare -a SUCCESS_MODELS

# Function to publish a single model
publish_model() {
    local model_key=$1
    local model_info=${MODELS[$model_key]}
    
    IFS=':' read -r model_path repo_name model_type <<< "$model_info"
    local repo_id="${HF_USERNAME}/${repo_name}"
    local output_dir="models/huggingface_ready/${repo_name}"
    
    echo ""
    echo "================================================"
    echo "Publishing: $repo_name"
    echo "================================================"
    echo "  Source: $model_path"
    echo "  Repo: $repo_id"
    echo "  Type: $model_type"
    echo ""
    
    # Check if model exists
    if [ ! -d "$model_path" ]; then
        print_warning "Model directory not found: $model_path"
        print_info "Skipping $repo_name"
        ((SKIPPED_COUNT++))
        return 1
    fi
    
    # Check if model has required files
    if [ ! -f "$model_path/config.json" ] && [ ! -f "$model_path/training_args.json" ]; then
        print_warning "No config files found in $model_path"
        print_info "Skipping $repo_name"
        ((SKIPPED_COUNT++))
        return 1
    fi
    
    if [ "$DRY_RUN" = true ]; then
        print_info "DRY RUN: Would publish $repo_name"
        ((SUCCESS_COUNT++))
        SUCCESS_MODELS+=("$repo_name")
        return 0
    fi
    
    # Step 1: Prepare model
    print_info "Step 1/5: Preparing model..."
    if python3 scripts/huggingface/prepare_model.py \
        --model "$model_path" \
        --output "$output_dir" \
        --force; then
        print_success "Model prepared"
    else
        print_error "Model preparation failed"
        ((FAILED_COUNT++))
        FAILED_MODELS+=("$repo_name")
        return 1
    fi
    
    # Step 2: Generate model card
    print_info "Step 2/5: Generating model card..."
    if python3 scripts/huggingface/generate_model_card.py \
        --model "$output_dir" \
        --type "$model_type" \
        --repo "$repo_id"; then
        print_success "Model card generated"
    else
        print_error "Model card generation failed"
        ((FAILED_COUNT++))
        FAILED_MODELS+=("$repo_name")
        return 1
    fi
    
    # Step 3: Test locally
    print_info "Step 3/5: Testing model locally..."
    if python3 scripts/huggingface/test_hf_model.py \
        --model "$output_dir" \
        --local; then
        print_success "Local tests passed"
    else
        print_warning "Local tests failed (continuing anyway)"
    fi
    
    # Step 4: Upload to Hub
    print_info "Step 4/5: Uploading to Hugging Face Hub..."
    if python3 scripts/huggingface/upload_model.py \
        --model "$output_dir" \
        --repo "$repo_id" \
        --message "Publish ${repo_name} from KothaGPT Bilingual project"; then
        print_success "Model uploaded"
    else
        print_error "Upload failed"
        ((FAILED_COUNT++))
        FAILED_MODELS+=("$repo_name")
        return 1
    fi
    
    # Step 5: Test from Hub
    print_info "Step 5/5: Testing from Hub..."
    if python3 scripts/huggingface/test_hf_model.py \
        --repo "$repo_id"; then
        print_success "Hub tests passed"
    else
        print_warning "Hub tests failed (model still published)"
    fi
    
    print_success "Successfully published: $repo_name"
    ((SUCCESS_COUNT++))
    SUCCESS_MODELS+=("$repo_name")
    return 0
}

# Main publishing loop
echo "Starting publication of $TOTAL_MODELS models..."
echo ""

for model_key in "${!MODELS[@]}"; do
    publish_model "$model_key" || true
done

# Summary
echo ""
echo "================================================"
echo "Publishing Summary"
echo "================================================"
echo ""
echo "Total Models: $TOTAL_MODELS"
echo "  ${GREEN}✓ Success: $SUCCESS_COUNT${NC}"
echo "  ${RED}✗ Failed: $FAILED_COUNT${NC}"
echo "  ${YELLOW}⊘ Skipped: $SKIPPED_COUNT${NC}"
echo ""

if [ ${#SUCCESS_MODELS[@]} -gt 0 ]; then
    echo "Successfully Published:"
    for model in "${SUCCESS_MODELS[@]}"; do
        echo "  ✓ https://huggingface.co/${HF_USERNAME}/${model}"
    done
    echo ""
fi

if [ ${#FAILED_MODELS[@]} -gt 0 ]; then
    echo "Failed Models:"
    for model in "${FAILED_MODELS[@]}"; do
        echo "  ✗ $model"
    done
    echo ""
fi

echo "================================================"
echo ""

if [ "$DRY_RUN" = true ]; then
    print_info "This was a dry run. No models were actually published."
    echo "Run without --dry-run to publish for real."
fi

# Exit with error if any models failed
if [ $FAILED_COUNT -gt 0 ]; then
    exit 1
fi

exit 0
