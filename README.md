# Bilingual | দ্বিভাষিক

<div align="center">

**High-quality Bangla + English NLP toolkit for production use**

**প্রোডাকশন ব্যবহারের জন্য উচ্চমানের বাংলা + ইংরেজি NLP টুলকিট**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[English](#english) | [বাংলা](#বাংলা)

</div>

---

## English

### Overview

**bilingual** is a Python package providing production-ready tools for Bangla and English natural language processing. It focuses on:

- 🌍 **Bilingual Support**: Equal treatment for Bangla and English
- 👶 **Child-Friendly Content**: Special focus on educational and age-appropriate material
- 🚀 **Production Ready**: Easy installation, comprehensive docs, robust testing
- 🔧 **Flexible**: From tokenization to translation, generation to classification
- 📚 **Well-Documented**: Full documentation in both English and Bangla

### Features

- **Text Normalization**: Unicode normalization, punctuation handling, script cleaning
- **Tokenization**: Shared SentencePiece tokenizer optimized for Bangla + English
- **Language Models**: Bilingual pretrained and fine-tuned models for generation
- **Translation**: Bangla ↔ English translation assistance
- **Classification**: Readability scoring, age-level detection, safety filtering
- **Utilities**: Dataset tools, evaluation metrics, preprocessing pipelines

### Quick Start

#### Installation

```bash
pip install bilingual
```

For development:

```bash
git clone https://github.com/YOUR_ORG/bilingual.git
cd bilingual
pip install -e ".[dev]"
```

#### Basic Usage

```python
from bilingual import bilingual_api as bb

# Load tokenizer
tokenizer = bb.load_tokenizer("bilingual-tokenizer")

# Normalize text
text_bn = bb.normalize_text("আমি স্কুলে যাচ্ছি।", lang="bn")
text_en = bb.normalize_text("I am going to school.", lang="en")

# Generate text
prompt = "A short story about a brave rabbit / সাহসী খরগোশের একটি ছোট গল্প"
story = bb.generate(prompt, model_name="bilingual-small-lm", max_tokens=150)

# Translate
translation = bb.translate("আমি বই পড়তে ভালোবাসি।", src="bn", tgt="en")
print(translation)  # "I love to read books."

# Check readability
level = bb.readability_check(text_bn, lang="bn")
print(f"Reading level: {level}")
```

#### CLI Usage

```bash
# Tokenize text
bilingual tokenize --lang bn --text "আমি ভাত খাই।"

# Generate text
bilingual generate --model bilingual-small-lm --prompt "Once upon a time..." --max-tokens 100

# Translate
bilingual translate --src bn --tgt en --text "আমি তোমাকে ভালোবাসি।"

# Evaluate model
bilingual evaluate --dataset data/test.jsonl --model bilingual-small-lm
```

### Project Structure

```
bilingual/
├── bilingual/              # Main package
│   ├── __init__.py
│   ├── api.py             # High-level API
│   ├── tokenizer.py       # Tokenization utilities
│   ├── normalize.py       # Text normalization
│   ├── models/            # Model implementations
│   │   ├── loader.py
│   │   ├── lm.py
│   │   └── translate.py
│   ├── evaluation.py      # Evaluation metrics
│   ├── data_utils.py      # Dataset utilities
│   └── cli.py             # Command-line interface
├── scripts/               # Training and data scripts
├── tests/                 # Test suite
├── docs/                  # Documentation
│   ├── en/               # English docs
│   └── bn/               # Bangla docs
├── datasets/              # Dataset storage
└── models/                # Model storage
```

### Documentation

- 📖 [Full Documentation](docs/en/README.md)
- 🚀 [Quick Start Guide](docs/en/quickstart.md)
- 🔧 [API Reference](docs/en/api.md)
- 🤝 [Contributing Guide](CONTRIBUTING.md)
- 🗺️ [Roadmap](ROADMAP.md)

### Development

```bash
# Run tests
pytest tests/

# Format code
black bilingual/ tests/

# Type checking
mypy bilingual/

# Lint
flake8 bilingual/
```

### Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

Areas where we need help:
- 📊 Dataset collection and curation
- 🤖 Model training and fine-tuning
- 📝 Documentation and translation
- 🧪 Testing and quality assurance
- 🐛 Bug fixes and improvements

### License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

### Citation

If you use this package in your research, please cite:

```bibtex
@software{bilingual2025,
  title = {Bilingual: High-quality Bangla and English NLP Toolkit},
  author = {Bilingual Project Contributors},
  year = {2025},
  url = {https://github.com/YOUR_ORG/bilingual}
}
```

### Acknowledgments

This project is built with support from the open-source community and aims to advance Bangla language technology for everyone.

---

## বাংলা

### সংক্ষিপ্ত বিবরণ

**bilingual** হল একটি Python প্যাকেজ যা বাংলা এবং ইংরেজি প্রাকৃতিক ভাষা প্রক্রিয়াকরণের জন্য প্রোডাকশন-রেডি টুল প্রদান করে। এটি ফোকাস করে:

- 🌍 **দ্বিভাষিক সমর্থন**: বাংলা এবং ইংরেজির জন্য সমান আচরণ
- 👶 **শিশু-বান্ধব কন্টেন্ট**: শিক্ষামূলক এবং বয়স-উপযুক্ত উপাদানের উপর বিশেষ ফোকাস
- 🚀 **প্রোডাকশন রেডি**: সহজ ইনস্টলেশন, ব্যাপক ডক্স, শক্তিশালী টেস্টিং
- 🔧 **নমনীয়**: টোকেনাইজেশন থেকে অনুবাদ, জেনারেশন থেকে শ্রেণীবিভাগ
- 📚 **ভালভাবে ডকুমেন্টেড**: ইংরেজি এবং বাংলা উভয় ভাষায় সম্পূর্ণ ডকুমেন্টেশন

### বৈশিষ্ট্য

- **টেক্সট নরমালাইজেশন**: ইউনিকোড নরমালাইজেশন, বিরামচিহ্ন হ্যান্ডলিং, স্ক্রিপ্ট পরিষ্কার করা
- **টোকেনাইজেশন**: বাংলা + ইংরেজির জন্য অপ্টিমাইজড শেয়ারড SentencePiece টোকেনাইজার
- **ভাষা মডেল**: জেনারেশনের জন্য দ্বিভাষিক প্রিট্রেইনড এবং ফাইন-টিউনড মডেল
- **অনুবাদ**: বাংলা ↔ ইংরেজি অনুবাদ সহায়তা
- **শ্রেণীবিভাগ**: পঠনযোগ্যতা স্কোরিং, বয়স-স্তর সনাক্তকরণ, নিরাপত্তা ফিল্টারিং
- **ইউটিলিটি**: ডেটাসেট টুল, মূল্যায়ন মেট্রিক্স, প্রিপ্রসেসিং পাইপলাইন

### দ্রুত শুরু

#### ইনস্টলেশন

```bash
pip install bilingual
```

ডেভেলপমেন্টের জন্য:

```bash
git clone https://github.com/YOUR_ORG/bilingual.git
cd bilingual
pip install -e ".[dev]"
```

#### মৌলিক ব্যবহার

```python
from bilingual import bilingual_api as bb

# টোকেনাইজার লোড করুন
tokenizer = bb.load_tokenizer("bilingual-tokenizer")

# টেক্সট নরমালাইজ করুন
text_bn = bb.normalize_text("আমি স্কুলে যাচ্ছি।", lang="bn")
text_en = bb.normalize_text("I am going to school.", lang="en")

# টেক্সট জেনারেট করুন
prompt = "A short story about a brave rabbit / সাহসী খরগোশের একটি ছোট গল্প"
story = bb.generate(prompt, model_name="bilingual-small-lm", max_tokens=150)

# অনুবাদ করুন
translation = bb.translate("আমি বই পড়তে ভালোবাসি।", src="bn", tgt="en")
print(translation)  # "I love to read books."

# পঠনযোগ্যতা চেক করুন
level = bb.readability_check(text_bn, lang="bn")
print(f"Reading level: {level}")
```

#### CLI ব্যবহার

```bash
# টেক্সট টোকেনাইজ করুন
bilingual tokenize --lang bn --text "আমি ভাত খাই।"

# টেক্সট জেনারেট করুন
bilingual generate --model bilingual-small-lm --prompt "Once upon a time..." --max-tokens 100

# অনুবাদ করুন
bilingual translate --src bn --tgt en --text "আমি তোমাকে ভালোবাসি।"

# মডেল মূল্যায়ন করুন
bilingual evaluate --dataset data/test.jsonl --model bilingual-small-lm
```

### ডকুমেন্টেশন

- 📖 [সম্পূর্ণ ডকুমেন্টেশন](docs/bn/README.md)
- 🚀 [দ্রুত শুরু গাইড](docs/bn/quickstart.md)
- 🔧 [API রেফারেন্স](docs/bn/api.md)
- 🤝 [অবদান গাইড](CONTRIBUTING.md)
- 🗺️ [রোডম্যাপ](ROADMAP.md)

### অবদান রাখা

আমরা অবদান স্বাগত জানাই! বিস্তারিত জানার জন্য অনুগ্রহ করে আমাদের [অবদান গাইড](CONTRIBUTING.md) দেখুন।

যেসব ক্ষেত্রে আমাদের সাহায্য প্রয়োজন:
- 📊 ডেটাসেট সংগ্রহ এবং কিউরেশন
- 🤖 মডেল ট্রেনিং এবং ফাইন-টিউনিং
- 📝 ডকুমেন্টেশন এবং অনুবাদ
- 🧪 টেস্টিং এবং কোয়ালিটি অ্যাসিউরেন্স
- 🐛 বাগ ফিক্স এবং উন্নতি

### লাইসেন্স

এই প্রকল্পটি Apache License 2.0 এর অধীনে লাইসেন্সপ্রাপ্ত - বিস্তারিত জানার জন্য [LICENSE](LICENSE) ফাইল দেখুন।

### স্বীকৃতি

এই প্রকল্পটি ওপেন-সোর্স কমিউনিটির সমর্থনে তৈরি এবং সবার জন্য বাংলা ভাষা প্রযুক্তি এগিয়ে নিয়ে যাওয়ার লক্ষ্যে কাজ করে।
