#!/usr/bin/env bash
# ============================================================
# 🤗 publish_all.sh — Automated Hugging Face Publisher
# Author: KothaGPT Contributors
# License: MIT
# ============================================================

set -euo pipefail

DRY_RUN=${1:-"false"}
ORG_NAME="KothaGPT"

echo "============================================================"
echo " 🚀 Starting Hugging Face Publish (dry-run=$DRY_RUN)"
echo "============================================================"

if [ -z "${HF_TOKEN:-}" ]; then
  echo "❌ Error: HF_TOKEN not set. Exiting..."
  exit 1
fi

# Create log directory
mkdir -p logs

# Define models and datasets to publish
MODELS=(
  "bilingual-lm"
  "readability-classifier"
  "tokenizer"
  "bilingual-tokenizer"
)

DATASETS=(
  "bilingual_wikipedia"
  "bengali_corpus"
  "parallel_bn_en"
)

publish_model() {
  local MODEL_DIR=$1
  local MODEL_NAME=$2
  local HF_REPO="$ORG_NAME/$MODEL_NAME"

  echo "------------------------------------------------------------"
  echo "📦 Publishing model: $HF_REPO"
  echo "------------------------------------------------------------"

  if [ "$DRY_RUN" = "true" ]; then
    echo "🧪 Dry run mode — skipping upload"
    ls -R "$MODEL_DIR" | head -n 10
    echo ""
    return
  fi

  huggingface-cli upload "$HF_REPO" "$MODEL_DIR" \
    --token "$HF_TOKEN" \
    --repo-type model \
    --revision main \
    --ignore-pattern "*.tmp" \
    --yes || echo "⚠️ Warning: Upload failed for $HF_REPO"

  echo "✅ Uploaded model: $HF_REPO"
}

publish_dataset() {
  local DATASET_DIR=$1
  local DATASET_NAME=$2
  local HF_REPO="$ORG_NAME/$DATASET_NAME"

  echo "------------------------------------------------------------"
  echo "📂 Publishing dataset: $HF_REPO"
  echo "------------------------------------------------------------"

  if [ "$DRY_RUN" = "true" ]; then
    echo "🧪 Dry run mode — skipping upload"
    ls -R "$DATASET_DIR" | head -n 10
    echo ""
    return
  fi

  huggingface-cli upload "$HF_REPO" "$DATASET_DIR" \
    --token "$HF_TOKEN" \
    --repo-type dataset \
    --revision main \
    --ignore-pattern "*.tmp" \
    --yes || echo "⚠️ Warning: Upload failed for $HF_REPO"

  echo "✅ Uploaded dataset: $HF_REPO"
}

# Upload models
for MODEL in "${MODELS[@]}"; do
  MODEL_PATH="models/$MODEL"
  if [ -d "$MODEL_PATH" ]; then
    publish_model "$MODEL_PATH" "$MODEL"
  else
    echo "⚠️ Skipping missing model: $MODEL_PATH"
  fi
done

# Upload datasets
for DATASET in "${DATASETS[@]}"; do
  DATASET_PATH="datasets/$DATASET"
  if [ -d "$DATASET_PATH" ]; then
    publish_dataset "$DATASET_PATH" "$DATASET"
  else
    echo "⚠️ Skipping missing dataset: $DATASET_PATH"
  fi
done

echo "============================================================"
echo "🎉 All models and datasets processed successfully!"
echo "============================================================"

