# 🌐 Wikipedia Language Model Training

Quick start guide for training language models on Wikipedia data.

---

## 🚀 Quick Start

### 1. Download Wikipedia Dump

```bash
python scripts/download_wiki.py --lang bn --output datasets/wikipedia/raw
```

### 2. Preprocess Data

```bash
python scripts/preprocess_wiki.py \
  --input datasets/wikipedia/raw/bnwiki-latest-pages-articles.xml.bz2 \
  --output datasets/wikipedia/processed \
  --lang bn
```

### 3. Train Model

```bash
python scripts/train_wiki_lm.py \
  --data datasets/wikipedia/processed \
  --model ai4bharat/indic-bert \
  --output models/wikipedia/base \
  --epochs 3
```

### 4. Evaluate

```bash
python scripts/evaluate_wiki_lm.py \
  --model models/wikipedia/base \
  --data datasets/wikipedia/processed/test/bn_test.txt
```

### 5. Use the Model

```python
from bilingual.modules.wikipedia_lm import load_model

model = load_model("models/wikipedia/base")
results = model.fill_mask("আমি [MASK] খাই", top_k=5)

for result in results:
    print(f"{result['sequence']} (score: {result['score']:.4f})")
```

---

## 📚 Available Scripts

| Script | Purpose | Example |
|--------|---------|---------|
| `download_wiki.py` | Download Wikipedia dumps | `python scripts/download_wiki.py --lang bn` |
| `preprocess_wiki.py` | Clean and tokenize text | `python scripts/preprocess_wiki.py --input raw/ --output processed/` |
| `analyze_wiki_dataset.py` | Analyze dataset quality | `python scripts/analyze_wiki_dataset.py --input processed/` |
| `train_wiki_lm.py` | Train language model | `python scripts/train_wiki_lm.py --data processed/ --model ai4bharat/indic-bert` |
| `evaluate_wiki_lm.py` | Evaluate trained model | `python scripts/evaluate_wiki_lm.py --model models/wikipedia/base` |
| `align_bilingual_wiki.py` | Align Bangla-English articles | `python scripts/align_bilingual_wiki.py --bn raw/bn --en raw/en` |

---

## 🎯 Use Cases

### Masked Language Modeling

```python
model = load_model("models/wikipedia/base")

# Fill masked tokens
results = model.fill_mask("বাংলাদেশের রাজধানী [MASK]", top_k=3)
# Output: ঢাকা, হচ্ছে, ছিল

# Predict next word
predictions = model.predict_next_word("আমি ভাত", top_k=5)
# Output: খাই, খেয়েছি, খাচ্ছি
```

### Text Generation

```python
model = load_model("models/wikipedia/gpt2-bn", model_type='clm')

texts = model.generate_text("আমি বাংলায়", max_length=100)
# Output: আমি বাংলায় কথা বলি এবং লিখি...
```

### Semantic Similarity

```python
similarity = model.compute_similarity(
    "আমি ভাত খাই",
    "আমি খাবার খাই"
)
# Output: 0.8532
```

### Embeddings

```python
embedding = model.get_sentence_embedding("আমি বাংলায় কথা বলি")
# Output: torch.Size([768])
```

---

## 🏗️ Project Structure

```
datasets/wikipedia/
├── raw/                          # Downloaded dumps
│   ├── bnwiki-latest-pages-articles.xml.bz2
│   └── enwiki-latest-pages-articles.xml.bz2
├── processed/                    # Processed text
│   ├── train/bn_train.txt
│   ├── val/bn_val.txt
│   └── test/bn_test.txt
├── bilingual/                    # Aligned Bangla-English
│   ├── aligned_articles.json
│   ├── bangla.txt
│   └── english.txt
└── analysis/                     # Dataset statistics

models/wikipedia/
├── base/                         # Base Wikipedia LM
│   ├── config.json
│   ├── pytorch_model.bin
│   └── tokenizer_config.json
└── finetuned_literary/          # Fine-tuned on literary data

scripts/
├── download_wiki.py             # Download Wikipedia dumps
├── preprocess_wiki.py           # Preprocess and clean text
├── analyze_wiki_dataset.py      # Analyze dataset quality
├── train_wiki_lm.py             # Train language model
├── evaluate_wiki_lm.py          # Evaluate model
└── align_bilingual_wiki.py      # Align bilingual articles

src/bilingual/modules/
└── wikipedia_lm.py              # Wikipedia LM module
```

---

## 📖 Documentation

- **[Full Training Roadmap](docs/WIKIPEDIA_TRAINING_ROADMAP.md)** - Complete guide with all phases
- **[Usage Examples](docs/examples/wikipedia_lm_usage.md)** - Code examples and recipes
- **[API Documentation](docs/api/index.md)** - API reference

---

## 🔧 Requirements

### Minimum

- Python 3.8+
- 8GB GPU (RTX 2070, T4)
- 16GB RAM
- 50GB disk space

### Recommended

- Python 3.9+
- 16GB+ GPU (V100, A100)
- 32GB+ RAM
- 100GB+ SSD

### Dependencies

```bash
pip install transformers datasets torch accelerate tensorboard
pip install wikiextractor indic-nlp-library
pip install matplotlib numpy scikit-learn
```

Or install all at once:

```bash
pip install -r requirements.txt
```

---

## ⚡ Performance Tips

1. **Use GPU:** Training is 10-50x faster on GPU
2. **Batch Processing:** Process multiple texts at once
3. **FP16 Training:** Enabled by default for 2x speedup
4. **Gradient Accumulation:** Simulate larger batch sizes
5. **Checkpoint Saving:** Save every 1000 steps

---

## 🎓 Training Tips

### Start Small

```bash
# Test with 1 epoch first
python scripts/train_wiki_lm.py \
  --data datasets/wikipedia/processed \
  --model ai4bharat/indic-bert \
  --output models/wikipedia/test \
  --epochs 1 \
  --batch-size 8
```

### Production Training

```bash
# Full training with optimizations
python scripts/train_wiki_lm.py \
  --data datasets/wikipedia/processed \
  --model ai4bharat/indic-bert \
  --output models/wikipedia/base \
  --epochs 5 \
  --batch-size 16 \
  --learning-rate 3e-5 \
  --gradient-accumulation-steps 4 \
  --save-steps 1000
```

### Monitor Training

```bash
# Launch TensorBoard
tensorboard --logdir models/wikipedia/base/logs
```

---

## 🌍 Bilingual Training

### Download Both Languages

```bash
python scripts/download_wiki.py --bilingual --output datasets/wikipedia/raw
```

### Align Articles

```bash
python scripts/align_bilingual_wiki.py \
  --bn datasets/wikipedia/raw/bn_extracted \
  --en datasets/wikipedia/raw/en_extracted \
  --output datasets/wikipedia/bilingual \
  --use-extracted
```

### Train Cross-lingual Model

```bash
python scripts/train_wiki_lm.py \
  --data datasets/wikipedia/bilingual \
  --model xlm-roberta-base \
  --output models/wikipedia/xlm-bilingual \
  --epochs 3
```

---

## 🐛 Troubleshooting

### Out of Memory

```bash
# Reduce batch size and increase gradient accumulation
python scripts/train_wiki_lm.py \
  --data datasets/wikipedia/processed \
  --model ai4bharat/indic-bert \
  --output models/wikipedia/base \
  --batch-size 4 \
  --gradient-accumulation-steps 8
```

### Slow Download

```bash
# Use wget with resume capability
wget -c https://dumps.wikimedia.org/bnwiki/latest/bnwiki-latest-pages-articles.xml.bz2
```

### Training Crashes

```bash
# Resume from checkpoint
python scripts/train_wiki_lm.py \
  --data datasets/wikipedia/processed \
  --model models/wikipedia/base/checkpoint-1000 \
  --output models/wikipedia/base \
  --epochs 3
```

---

## 📊 Expected Results

| Metric | Value | Notes |
|--------|-------|-------|
| **Perplexity** | 20-40 | On Bangla Wikipedia test set |
| **Training Time** | 1-3 days | Single V100 GPU |
| **Model Size** | ~400MB | BERT-base |
| **Vocabulary** | 50K-100K | Bangla tokens |

---

## 🤝 Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📝 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- **AI4Bharat** for Indic BERT models
- **Wikimedia Foundation** for Wikipedia dumps
- **HuggingFace** for Transformers library
- **Indic NLP Library** for text processing

---

## 📬 Contact

For questions or issues, please open an issue on GitHub or contact the maintainers.

---

## 🔗 Links

- [GitHub Repository](https://github.com/KothaGPT/bilingual)
- [Documentation](docs/WIKIPEDIA_TRAINING_ROADMAP.md)
- [Examples](docs/examples/wikipedia_lm_usage.md)
- [API Reference](docs/api/index.md)
