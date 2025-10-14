# Getting Started

# 🚀 **Getting Started with Bilingual**

This guide will help you get up and running with the Bilingual NLP Toolkit in minutes.

## Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-compatible GPU for faster model inference

## Installation

Install the package using pip:

```bash
pip install bilingual
```

For development installation with all optional dependencies:

```bash
pip install bilingual[dev]
```

Or install from source:

```bash
git clone https://github.com/bilingual-nlp/bilingual.git
cd bilingual
pip install -e .
```

## Quick Start

### 1. Basic Text Processing

```python
import bilingual as bb

# Detect language
text = "আমি স্কুলে যাই এবং বই পড়তে ভালোবাসি।"
result = bb.detect_language(text)
print(f"Language: {result['language']}")  # Output: bengali

# Process text with multiple tasks
processed = bb.process(text, tasks=["normalize", "tokenize"])
print(processed)
```

### 2. Translation

```python
# Translate between languages
english_text = "I love reading books"
bengali_translation = bb.translate_text(
    "t5-small", english_text, "en", "bn"
)

print(f"English: {english_text}")
print(f"Bengali: {bengali_translation}")
```

### 3. Text Generation

```python
# Generate text using transformer models
prompt = "Write a short story about a brave child"
story = bb.generate_text("t5-small", prompt, max_length=100)

print(f"Generated story: {story}")
```

### 4. Evaluation

```python
# Evaluate translation quality
references = ["I love reading books"]
candidates = ["I adore reading books"]

results = bb.evaluate_translation(references, candidates)
print(f"BLEU Score: {results['bleu']:.4f}")
print(f"METEOR Score: {results['meteor']:.4f}")
```

## Command Line Interface

The toolkit also provides a powerful CLI:

```bash
# Show information
bilingual info

# Process text
bilingual process "Hello world" --tasks detect normalize tokenize

# Translate text
bilingual translate "Hello world" --from en --to bn

# Evaluate translations
bilingual evaluate translation --reference refs.txt --candidate candidates.txt

# Generate text
bilingual generate "Write a story" --model t5-small
```

## Configuration

Customize the toolkit behavior using configuration:

```python
from bilingual.config import get_settings

# Get current settings
settings = get_settings()

# Customize settings
settings.model.default_model = "t5-base"
settings.evaluation.bleu_ngram_order = 4
```

Environment variables can also be used:

```bash
export BILINGUAL_MODEL_DEFAULT_MODEL="t5-base"
export BILINGUAL_EVAL_BLEU_NGRAM_ORDER=4
```

## Next Steps

- Explore the [API Reference](api/index.md) for detailed function documentation
- Check out [Examples](examples/index.md) for more advanced use cases
- Learn about [Contributing](contributing/index.md) to help improve the toolkit

## Troubleshooting

### Common Issues

**Import Error**: Make sure all dependencies are installed:
```bash
pip install transformers torch sentencepiece
```

**CUDA Issues**: For GPU support, ensure PyTorch is installed with CUDA:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Memory Issues**: For large models, use smaller variants:
```python
bb.load_model("t5-small")  # Instead of t5-base
```

### Getting Help

- Check the [GitHub Issues](https://github.com/bilingual-nlp/bilingual/issues) for known problems
- Join our [GitHub Discussions](https://github.com/bilingual-nlp/bilingual/discussions) for community support
- File a bug report or feature request on GitHub

---

*Ready to explore more? Continue to the [API Reference](api/index.md) or check out some [Examples](examples/index.md)!* 🚀
