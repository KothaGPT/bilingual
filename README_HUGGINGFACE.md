# 🚀 Hugging Face Model Publishing - Quick Reference

Publish your trained Bangla Wikipedia models to Hugging Face Hub in minutes.

---

## ⚡ Quick Start

### One-Command Publishing

```bash
bash scripts/huggingface/publish_all.sh \
  models/wikipedia/base \
  bn-wikipedia-lm \
  your-username \
  base
```

This runs the complete pipeline:
1. ✓ Prepares model
2. ✓ Generates model card
3. ✓ Tests locally
4. ✓ Uploads to Hub
5. ✓ Tests from Hub
6. ✓ Creates demo app

---

## 📋 Manual Steps

### 1. Setup

```bash
# Install dependencies
pip install huggingface_hub gradio

# Login to Hugging Face
huggingface-cli login
```

### 2. Prepare Model

```bash
python scripts/huggingface/prepare_model.py \
  --model models/wikipedia/base \
  --output models/huggingface_ready/bn-wikipedia-lm
```

### 3. Generate Model Card

```bash
python scripts/huggingface/generate_model_card.py \
  --model models/huggingface_ready/bn-wikipedia-lm \
  --type base \
  --repo your-username/bn-wikipedia-lm
```

### 4. Test Locally

```bash
python scripts/huggingface/test_hf_model.py \
  --model models/huggingface_ready/bn-wikipedia-lm \
  --local
```

### 5. Upload

```bash
python scripts/huggingface/upload_model.py \
  --model models/huggingface_ready/bn-wikipedia-lm \
  --repo your-username/bn-wikipedia-lm
```

### 6. Create Demo

```bash
python scripts/huggingface/create_demo.py \
  --repo your-username/bn-wikipedia-lm \
  --output spaces/bn-wikipedia-demo
```

---

## 🛠️ Available Scripts

| Script | Purpose | Example |
|--------|---------|---------|
| `prepare_model.py` | Clean and organize model | `python scripts/huggingface/prepare_model.py --model models/wikipedia/base --output models/huggingface_ready/bn-wikipedia-lm` |
| `upload_model.py` | Upload to Hugging Face Hub | `python scripts/huggingface/upload_model.py --model models/huggingface_ready/bn-wikipedia-lm --repo your-username/bn-wikipedia-lm` |
| `generate_model_card.py` | Create README.md | `python scripts/huggingface/generate_model_card.py --model models/huggingface_ready/bn-wikipedia-lm --type base --repo your-username/bn-wikipedia-lm` |
| `test_hf_model.py` | Test model inference | `python scripts/huggingface/test_hf_model.py --repo your-username/bn-wikipedia-lm` |
| `create_demo.py` | Create Gradio demo | `python scripts/huggingface/create_demo.py --repo your-username/bn-wikipedia-lm --output spaces/demo` |
| `publish_all.sh` | Complete pipeline | `bash scripts/huggingface/publish_all.sh models/wikipedia/base bn-wikipedia-lm your-username base` |

---

## 📚 Documentation

- **[Complete Guide](docs/HUGGINGFACE_PUBLISHING_GUIDE.md)** - Detailed 8-step process
- **[TODO Checklist](HUGGINGFACE_TODO.md)** - Step-by-step checklist
- **[Implementation Summary](HUGGINGFACE_IMPLEMENTATION_SUMMARY.md)** - What's been built

---

## 🎯 Models to Publish

### Base Wikipedia LM
```bash
bash scripts/huggingface/publish_all.sh \
  models/wikipedia/base \
  bn-wikipedia-lm \
  your-username \
  base
```

### Literary Fine-tuned LM
```bash
bash scripts/huggingface/publish_all.sh \
  models/wikipedia/finetuned_literary \
  bn-wikipedia-literary-lm \
  your-username \
  literary
```

### Bilingual XLM
```bash
bash scripts/huggingface/publish_all.sh \
  models/wikipedia/xlm-bilingual \
  bn-en-wikipedia-xlm \
  your-username \
  base
```

---

## 🧪 Testing

### Test Local Model
```bash
python scripts/huggingface/test_hf_model.py \
  --model models/huggingface_ready/bn-wikipedia-lm \
  --local
```

### Test Published Model
```bash
python scripts/huggingface/test_hf_model.py \
  --repo your-username/bn-wikipedia-lm
```

### Test Demo Locally
```bash
cd spaces/bn-wikipedia-demo
pip install -r requirements.txt
python app.py
```

---

## 🌐 Demo Features

The generated Gradio demo includes:

1. **Fill-Mask Tab**
   - Predict masked words
   - Top-k predictions
   - Example inputs

2. **Embeddings Tab**
   - Get sentence embeddings
   - View dimensions

3. **Similarity Tab**
   - Compute semantic similarity
   - Compare texts

---

## 🔄 CI/CD

GitHub Actions workflow automatically tests your published model:

- **Location:** `.github/workflows/test_hf_model.yml`
- **Triggers:** Weekly + Manual
- **Tests:** Model loading, inference, multiple Python versions

**Run manually:**
1. Go to GitHub Actions
2. Select "Test Hugging Face Model"
3. Click "Run workflow"
4. Enter your repo ID

---

## 📊 Checklist

- [ ] Model trained and evaluated
- [ ] Hugging Face account created
- [ ] Dependencies installed (`pip install huggingface_hub gradio`)
- [ ] Logged in (`huggingface-cli login`)
- [ ] Model prepared
- [ ] Model card generated and customized
- [ ] Local tests passed
- [ ] Model uploaded
- [ ] Hub tests passed
- [ ] Demo created (optional)
- [ ] CI/CD configured

---

## 🚨 Troubleshooting

### Upload Fails
```bash
# Check Git LFS
git lfs version

# Try direct git upload
git clone https://huggingface.co/your-username/bn-wikipedia-lm
cd bn-wikipedia-lm
cp -r ../models/huggingface_ready/bn-wikipedia-lm/* .
git add . && git commit -m "Upload model" && git push
```

### Model Card Not Rendering
- Check YAML frontmatter syntax
- Validate markdown formatting
- Preview locally before uploading

### Demo Crashes
- Test locally first
- Check requirements.txt versions
- Review Space logs on Hugging Face

---

## 🎓 Example Usage

### After Publishing

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM

# Load your published model
tokenizer = AutoTokenizer.from_pretrained("your-username/bn-wikipedia-lm")
model = AutoModelForMaskedLM.from_pretrained("your-username/bn-wikipedia-lm")

# Fill mask
from transformers import pipeline
fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)
results = fill_mask("বাংলাদেশের রাজধানী [MASK]")

for result in results:
    print(f"{result['sequence']} (score: {result['score']:.4f})")
```

---

## 📈 Next Steps

After publishing:

1. **Share** - Post on social media, Bangla NLP groups
2. **Monitor** - Track downloads, issues, discussions
3. **Iterate** - Gather feedback, improve model
4. **Build** - Create applications using your model

---

## 🔗 Links

- **Hugging Face Hub:** https://huggingface.co
- **Model Hub Docs:** https://huggingface.co/docs/hub
- **Transformers Docs:** https://huggingface.co/docs/transformers
- **Gradio Docs:** https://gradio.app/docs
- **GitHub:** https://github.com/KothaGPT/bilingual

---

## 💡 Tips

- **Start with base model** - Publish Wikipedia LM first
- **Test thoroughly** - Run all tests before uploading
- **Customize model card** - Add actual metrics and examples
- **Create demo** - Makes model more accessible
- **Version properly** - Use semantic versioning (v0.1.0, v0.2.0)
- **Monitor usage** - Check downloads and feedback

---

**Ready to publish?** Run the automated pipeline or follow the [complete guide](docs/HUGGINGFACE_PUBLISHING_GUIDE.md)!
