# 🚀 Quick Start: Publish to Hugging Face

**Goal:** Publish all KothaGPT models to Hugging Face Hub in under 30 minutes.

---

## ⚡ Prerequisites (5 minutes)

### 1. Hugging Face Account
```bash
# Create account at https://huggingface.co/join
# Create organization: https://huggingface.co/organizations/new
# Recommended name: KothaGPT
```

### 2. Install Dependencies
```bash
pip install huggingface_hub transformers datasets torch
```

### 3. Login
```bash
huggingface-cli login
# Enter your token from https://huggingface.co/settings/tokens
```

---

## 🎯 One-Command Publication (10 minutes)

### Option A: Publish Everything

```bash
# Validate → Generate Cards → Publish All Models + Dataset
bash scripts/huggingface/publish_multi_models.sh --username KothaGPT
```

### Option B: Step-by-Step

```bash
# 1. Validate models (2 min)
python scripts/huggingface/validate_publish_readiness.py

# 2. Generate model cards (1 min)
python scripts/huggingface/generate_all_model_cards.py

# 3. Dry run (test without uploading) (2 min)
bash scripts/huggingface/publish_multi_models.sh --dry-run --username KothaGPT

# 4. Publish models (5 min)
bash scripts/huggingface/publish_multi_models.sh --username KothaGPT

# 5. Publish dataset (2 min)
python scripts/prepare_hf_dataset.py \
  --input datasets/processed/final \
  --push-to-hub \
  --repo KothaGPT/bilingual-corpus
```

---

## 📊 What Gets Published

### Models (9)
1. `KothaGPT/bilingual-language-model` - Causal LM
2. `KothaGPT/literary-language-model` - Literary LM
3. `KothaGPT/readability-classifier` - Text classification
4. `KothaGPT/poetic-meter-detector` - Poetry analysis
5. `KothaGPT/metaphor-simile-detector` - Figurative language
6. `KothaGPT/style-transfer-gpt` - Style transfer
7. `KothaGPT/sentiment-tone-classifier` - Sentiment analysis
8. `KothaGPT/cross-lingual-embeddings` - Embeddings
9. `KothaGPT/named-entity-recognizer` - NER

### Dataset (1)
10. `KothaGPT/bilingual-corpus` - Training data

---

## ✅ Verify Publication (5 minutes)

```bash
# Test loading from Hub
python scripts/huggingface/test_hf_model.py \
  --repo KothaGPT/bilingual-language-model

# Or test in Python
python -c "
from transformers import AutoTokenizer, AutoModel
model = AutoModel.from_pretrained('KothaGPT/bilingual-language-model')
print('✅ Model loaded successfully!')
"
```

---

## 🎨 Add Badges to README (2 minutes)

```bash
python scripts/huggingface/add_hf_badges.py --username KothaGPT
```

This adds badges to `README.md` and `README_HUGGINGFACE.md`:

[![Bilingual LM](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Bilingual%20LM-blue)](https://huggingface.co/KothaGPT/bilingual-language-model)

---

## 🤖 Using GitHub Actions (Automated)

### Setup (One-time)

1. **Add Secret to GitHub:**
   - Go to: `Settings` → `Secrets and variables` → `Actions`
   - Click: `New repository secret`
   - Name: `HF_TOKEN`
   - Value: Your Hugging Face token

2. **Trigger Workflow:**
   - Go to: `Actions` tab
   - Select: `🤗 Publish Models to Hugging Face`
   - Click: `Run workflow`
   - Choose: `all` models, dry_run: `false`

### Automatic Publishing

Push a version tag to trigger automatic publication:

```bash
git tag v1.0.0
git push origin v1.0.0
```

---

## 🔍 Troubleshooting

### "Model directory not found"
```bash
# Check which models exist
ls -la models/

# Update model paths in publish_multi_models.sh if needed
```

### "HF_TOKEN not found"
```bash
# Login again
huggingface-cli login

# Or set environment variable
export HF_TOKEN="your_token_here"
```

### "Upload failed: Large file"
```bash
# Install Git LFS
git lfs install

# Or use smaller model checkpoints
```

### "Model card validation failed"
```bash
# Regenerate model cards
python scripts/huggingface/generate_all_model_cards.py

# Check YAML syntax in cards/*.md files
```

---

## 📱 Quick Commands Reference

```bash
# Validate everything
python scripts/huggingface/validate_publish_readiness.py

# Publish one model
python scripts/huggingface/upload_model.py \
  --model models/huggingface_ready/MODEL_NAME \
  --repo KothaGPT/MODEL_NAME

# Test from Hub
python scripts/huggingface/test_hf_model.py \
  --repo KothaGPT/MODEL_NAME

# Create demo
python scripts/huggingface/create_demo.py \
  --repo KothaGPT/MODEL_NAME \
  --output spaces/MODEL_NAME-demo
```

---

## 🎉 Success!

After publication, your models will be available at:
- **Organization:** https://huggingface.co/KothaGPT
- **Models:** https://huggingface.co/KothaGPT/MODEL_NAME
- **Dataset:** https://huggingface.co/datasets/KothaGPT/bilingual-corpus

### Next Steps
1. ✅ Share on social media
2. ✅ Create demo spaces
3. ✅ Write blog post
4. ✅ Monitor downloads and feedback
5. ✅ Iterate and improve

---

## 📚 More Information

- **Complete Guide:** `docs/HUGGINGFACE_PUBLISHING_GUIDE.md`
- **Implementation Details:** `HUGGINGFACE_IMPLEMENTATION_COMPLETE.md`
- **Original TODO:** `HUGGINGFACE_TODO.md`

---

**Time to Publish:** ~30 minutes  
**Difficulty:** Easy  
**Result:** 9 models + 1 dataset on Hugging Face Hub 🎉
