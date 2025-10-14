# Getting Started with Bilingual

Welcome! This guide will help you get up and running with the **bilingual** package in just a few minutes.

## 🎯 What is Bilingual?

**bilingual** is a Python package for Bangla and English natural language processing. It provides:

- ✅ **Text normalization** - Clean and standardize text
- ✅ **Tokenization** - Break text into tokens
- ✅ **Language models** - Generate and understand text
- ✅ **Translation** - Translate between Bangla and English
- ✅ **Classification** - Assess readability and safety
- ✅ **Data tools** - Manage datasets efficiently

## 📋 Prerequisites

- Python 3.8 or higher
- pip package manager

## ⚡ Quick Install

```bash
# Clone the repository
git clone https://github.com/YOUR_ORG/bilingual.git
cd bilingual

# Install the package
pip install -e .
```

## 🚀 Your First 5 Minutes

### 1. Test the Installation

```bash
python3 -c "import bilingual; print(f'✓ Bilingual v{bilingual.__version__}')"
```

### 2. Try Text Normalization

Create a file `test_normalize.py`:

```python
from bilingual import bilingual_api as bb

# Normalize Bangla text
text = "আমি   স্কুলে যাচ্ছি।"
normalized = bb.normalize_text(text, lang="bn")
print(f"Original:   {text}")
print(f"Normalized: {normalized}")
```

Run it:
```bash
python test_normalize.py
```

### 3. Check Readability

```python
from bilingual import bilingual_api as bb

text = "আমি স্কুলে যাই।"
result = bb.readability_check(text, lang="bn")

print(f"Text: {text}")
print(f"Reading Level: {result['level']}")
print(f"Age Range: {result['age_range']}")
print(f"Score: {result['score']}")
```

### 4. Work with Datasets

```python
from bilingual.data_utils import BilingualDataset

# Create a dataset
data = [
    {"text": "আমি স্কুলে যাই।", "lang": "bn"},
    {"text": "I go to school.", "lang": "en"},
]

dataset = BilingualDataset(data=data)
print(f"Dataset size: {len(dataset)}")

# Filter by language
bn_only = dataset.filter(lambda x: x["lang"] == "bn")
print(f"Bangla samples: {len(bn_only)}")
```

### 5. Use the CLI

```bash
# Normalize text
bilingual normalize --text "আমি স্কুলে যাই।" --lang bn

# Check readability
bilingual readability --text "আমি স্কুলে যাই।" --lang bn

# Check safety
bilingual safety --text "This is a nice story."
```

## 📚 Next Steps

### For Users

1. **Read the Documentation**
   - [English docs](docs/en/README.md)
   - [Bangla docs](docs/bn/README.md)

2. **Run Examples**
   ```bash
   python examples/basic_usage.py
   ```

3. **Explore the API**
   - Check out [docs/en/quickstart.md](docs/en/quickstart.md)

### For Contributors

1. **Install Dev Dependencies**
   ```bash
   pip install -e ".[dev]"
   ```

2. **Collect Sample Data**
   ```bash
   python scripts/collect_data.py --source sample --output data/raw/
   ```

3. **Prepare Data**
   ```bash
   python scripts/prepare_data.py \
       --input data/raw/ \
       --output datasets/processed/
   ```

4. **Run Tests**
   ```bash
   pytest tests/ -v
   ```

5. **Read Contributing Guide**
   - [CONTRIBUTING.md](CONTRIBUTING.md)

### For Researchers

1. **Train a Tokenizer**
   ```bash
   python scripts/train_tokenizer.py \
       --input data/raw/sample_bn.txt data/raw/sample_en.txt \
       --model-prefix bilingual_sp \
       --vocab-size 32000 \
       --output-dir models/tokenizer/
   ```

2. **Prepare Your Own Dataset**
   - See [docs/en/data.md](docs/en/data.md) (coming soon)

3. **Fine-tune Models**
   - See [docs/en/training.md](docs/en/training.md) (coming soon)

## 🎓 Learning Path

### Beginner
1. ✅ Install package
2. ✅ Try text normalization
3. ✅ Use readability checker
4. ✅ Work with datasets

### Intermediate
1. Collect and prepare data
2. Train a tokenizer
3. Fine-tune a small model
4. Run evaluations

### Advanced
1. Train production models
2. Deploy inference servers
3. Contribute to the project
4. Create custom pipelines

## 💡 Common Use Cases

### Use Case 1: Clean Text Data

```python
from bilingual import bilingual_api as bb

texts = [
    "আমি   স্কুলে যাই।",
    "আমি বই পড়ি।",
    "আমি খেলতে ভালোবাসি।"
]

cleaned = [bb.normalize_text(t, lang="bn") for t in texts]
for original, clean in zip(texts, cleaned):
    print(f"{original} → {clean}")
```

### Use Case 2: Filter Child-Appropriate Content

```python
from bilingual import bilingual_api as bb

texts = [
    "This is a nice story about rabbits.",
    "Once upon a time...",
    "A brave little girl...",
]

for text in texts:
    result = bb.safety_check(text)
    if result['is_safe']:
        print(f"✓ Safe: {text}")
    else:
        print(f"✗ Review needed: {text}")
```

### Use Case 3: Assess Reading Levels

```python
from bilingual import bilingual_api as bb

stories = {
    "elementary": "আমি স্কুলে যাই।",
    "intermediate": "আমি প্রতিদিন সকালে স্কুলে যাই এবং বন্ধুদের সাথে খেলি।",
    "advanced": "আমি বিশ্ববিদ্যালয়ে উচ্চশিক্ষা গ্রহণ করছি এবং গবেষণা কাজে নিয়োজিত আছি।",
}

for expected_level, text in stories.items():
    result = bb.readability_check(text, lang="bn")
    print(f"Expected: {expected_level}, Got: {result['level']}")
```

## 🆘 Getting Help

### Documentation
- 📖 [Full Documentation](docs/en/README.md)
- 🚀 [Quick Start](docs/en/quickstart.md)
- 🔧 [Setup Guide](SETUP.md)

### Community
- 💬 [GitHub Discussions](https://github.com/YOUR_ORG/bilingual/discussions)
- 🐛 [Report Issues](https://github.com/YOUR_ORG/bilingual/issues)
- 📧 Email: info@khulnasoft.com

### Resources
- 🗺️ [Project Roadmap](ROADMAP.md)
- 📊 [Project Status](PROJECT_STATUS.md)
- 🤝 [Contributing Guide](CONTRIBUTING.md)

## ❓ FAQ

**Q: Do I need trained models to use the package?**
A: Basic features (normalization, data utilities) work without models. For generation and translation, you'll need trained models.

**Q: How do I train my own models?**
A: See the training scripts in `scripts/` and documentation in `docs/en/training.md` (coming soon).

**Q: Can I use this for commercial projects?**
A: Yes! The package is licensed under Apache 2.0, which allows commercial use.

**Q: How can I contribute?**
A: See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines. We welcome all contributions!

**Q: Is there support for other languages?**
A: Currently focused on Bangla and English, but the architecture is extensible.

## 🎉 Success!

You're now ready to use **bilingual**! Start with the examples and explore the documentation.

**Happy coding! / শুভ কোডিং!**

---

For more information, visit the [full documentation](docs/en/README.md) or join our [community discussions](https://github.com/YOUR_ORG/bilingual/discussions).
