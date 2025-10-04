# Bilingual Documentation

Welcome to the **bilingual** package documentation!

## Contents

- [Quick Start](quickstart.md)
- [API Reference](api.md)
- [Data Guide](data.md)
- [Model Training](training.md)
- [Deployment](deployment.md)
- [Contributing](../../CONTRIBUTING.md)

## Overview

**bilingual** is a production-ready Python package for Bangla and English natural language processing. It provides:

- **Text Normalization**: Clean and standardize text in both languages
- **Tokenization**: Efficient SentencePiece-based tokenization
- **Language Models**: Pretrained and fine-tuned models for generation
- **Translation**: Bangla ↔ English translation support
- **Classification**: Readability, safety, and content classification
- **Data Utilities**: Tools for dataset management and preprocessing

## Quick Links

- [GitHub Repository](https://github.com/YOUR_ORG/bilingual)
- [PyPI Package](https://pypi.org/project/bilingual/)
- [Issue Tracker](https://github.com/YOUR_ORG/bilingual/issues)
- [Bangla Documentation](../bn/README.md)

## Installation

```bash
pip install bilingual
```

For development:

```bash
git clone https://github.com/YOUR_ORG/bilingual.git
cd bilingual
pip install -e ".[dev]"
```

## Basic Usage

```python
from bilingual import bilingual_api as bb

# Normalize text
text = bb.normalize_text("আমি স্কুলে যাই।", lang="bn")

# Tokenize
tokens = bb.tokenize(text)

# Generate text
story = bb.generate("Once upon a time...", max_tokens=100)

# Translate
translation = bb.translate("আমি বই পড়ি।", src="bn", tgt="en")
```

## Support

- 📧 Email: bilingual@example.com
- 💬 Discussions: [GitHub Discussions](https://github.com/YOUR_ORG/bilingual/discussions)
- 🐛 Bug Reports: [Issue Tracker](https://github.com/YOUR_ORG/bilingual/issues)

## License

This project is licensed under the Apache License 2.0. See [LICENSE](../../LICENSE) for details.
