# Quick Start Guide

This guide will help you get started with the **bilingual** package in minutes.

## Installation

### From PyPI

```bash
pip install bilingual
```

### From Source

```bash
git clone https://github.com/YOUR_ORG/bilingual.git
cd bilingual
pip install -e .
```

### Optional Dependencies

For PyTorch support:
```bash
pip install bilingual[torch]
```

For TensorFlow support:
```bash
pip install bilingual[tensorflow]
```

For development:
```bash
pip install bilingual[dev]
```

## Basic Usage

### Text Normalization

```python
from bilingual import bilingual_api as bb

# Normalize Bangla text
text_bn = "আমি   স্কুলে যাচ্ছি।"
normalized = bb.normalize_text(text_bn, lang="bn")
print(normalized)  # "আমি স্কুলে যাচ্ছি."

# Normalize English text
text_en = "I am   going to school."
normalized = bb.normalize_text(text_en, lang="en")
print(normalized)  # "I am going to school."

# Auto-detect language
text = "আমি school যাই।"
normalized = bb.normalize_text(text)  # Auto-detects mixed language
```

### Tokenization

```python
from bilingual import bilingual_api as bb

# Tokenize text (requires trained tokenizer)
text = "আমি বই পড়ি।"
tokens = bb.tokenize(text)
print(tokens)

# Get token IDs
token_ids = bb.tokenize(text, return_ids=True)
print(token_ids)
```

### Text Generation

```python
from bilingual import bilingual_api as bb

# Generate text from prompt
prompt = "Once upon a time, there was a brave rabbit"
generated = bb.generate(
    prompt,
    model_name="bilingual-small-lm",
    max_tokens=100,
    temperature=0.7
)
print(generated)
```

### Translation

```python
from bilingual import bilingual_api as bb

# Translate Bangla to English
text_bn = "আমি বই পড়তে ভালোবাসি।"
translation = bb.translate(text_bn, src="bn", tgt="en")
print(translation)  # "I love to read books."

# Translate English to Bangla
text_en = "I go to school every day."
translation = bb.translate(text_en, src="en", tgt="bn")
print(translation)
```

### Readability Check

```python
from bilingual import bilingual_api as bb

# Check readability level
text = "আমি স্কুলে যাই।"
result = bb.readability_check(text, lang="bn")

print(f"Level: {result['level']}")        # elementary/intermediate/advanced
print(f"Age Range: {result['age_range']}")  # e.g., "6-8"
print(f"Score: {result['score']}")          # numerical score
```

### Safety Check

```python
from bilingual import bilingual_api as bb

# Check if content is safe for children
text = "This is a nice story about animals."
result = bb.safety_check(text)

print(f"Safe: {result['is_safe']}")
print(f"Confidence: {result['confidence']}")
print(f"Recommendation: {result['recommendation']}")
```

## Command-Line Interface

### Normalize Text

```bash
bilingual normalize --text "আমি স্কুলে যাই।" --lang bn
```

### Generate Text

```bash
bilingual generate --prompt "Once upon a time..." --max-tokens 100
```

### Translate

```bash
bilingual translate --text "আমি বই পড়ি।" --src bn --tgt en
```

### Check Readability

```bash
bilingual readability --text "আমি স্কুলে যাই।" --lang bn
```

## Working with Datasets

```python
from bilingual.data_utils import BilingualDataset

# Load dataset
dataset = BilingualDataset(file_path="data/train.jsonl")

# Iterate through samples
for sample in dataset:
    print(sample["text"])

# Split dataset
train, val, test = dataset.split(
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1
)

# Filter dataset
filtered = dataset.filter(lambda x: x["lang"] == "bn")

# Save dataset
dataset.save("output.jsonl", format="jsonl")
```

## Next Steps

- Read the [API Reference](api.md) for detailed documentation
- Learn about [Data Preparation](data.md)
- Explore [Model Training](training.md)
- Check out [Deployment Options](deployment.md)

## Getting Help

- 📖 [Full Documentation](README.md)
- 💬 [GitHub Discussions](https://github.com/YOUR_ORG/bilingual/discussions)
- 🐛 [Report Issues](https://github.com/YOUR_ORG/bilingual/issues)
