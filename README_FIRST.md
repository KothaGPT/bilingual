# 👋 START HERE - Bilingual Project

**Welcome to the bilingual package!** This file will guide you through the project.

---

## 🎯 What is This?

**bilingual** is a production-ready Python package for Bangla and English NLP with:
- Text normalization and cleaning
- Dataset management tools
- Readability and safety assessment
- API ready for tokenization, generation, and translation (needs trained models)

---

## ⚡ Quick Start (5 Minutes)

### 1. Install
```bash
cd /Users/kothagpt/bilingual
pip install -e .
```

### 2. Try It
```python
from bilingual import bilingual_api as bb

# Normalize Bangla text
text = bb.normalize_text("আমি   স্কুলে যাচ্ছি।", lang="bn")
print(text)  # "আমি স্কুলে যাচ্ছি."

# Check readability
result = bb.readability_check("আমি স্কুলে যাই।", lang="bn")
print(result)  # {'level': 'intermediate', 'age_range': '9-12', ...}
```

### 3. Run Examples
```bash
PYTHONPATH=/Users/kothagpt/bilingual python examples/basic_usage.py
```

---

## 📚 Documentation Guide

### For Users (Want to Use It)
1. **[GETTING_STARTED.md](GETTING_STARTED.md)** ⭐ Start here!
2. **[docs/en/quickstart.md](docs/en/quickstart.md)** - Detailed guide
3. **[examples/basic_usage.py](examples/basic_usage.py)** - Working examples

### For Contributors (Want to Help)
1. **[CONTRIBUTING.md](CONTRIBUTING.md)** - How to contribute
2. **[PROJECT_STATUS.md](PROJECT_STATUS.md)** - What's needed
3. **[ROADMAP.md](ROADMAP.md)** - Future plans

### For Developers (Want to Understand Code)
1. **[PROJECT_MAP.md](PROJECT_MAP.md)** - Navigation guide
2. **[bilingual/api.py](bilingual/api.py)** - Main API
3. **[SUMMARY.md](SUMMARY.md)** - Complete overview

### বাংলা ডকুমেন্টেশন
1. **[docs/bn/README.md](docs/bn/README.md)** - বাংলা ডক্স
2. **[docs/bn/quickstart.md](docs/bn/quickstart.md)** - দ্রুত শুরু

---

## 🗂️ Project Structure

```
bilingual/
├── README.md              ⭐ Main overview
├── README_FIRST.md        📍 This file
├── GETTING_STARTED.md     🚀 Quick start
├── bilingual/             📦 Core package
├── scripts/               🛠️ Utility scripts
├── tests/                 🧪 Test suite
├── docs/                  📖 Documentation
└── examples/              💡 Usage examples
```

---

## ✅ What's Working Now

- ✅ **Text normalization** - Clean Bangla & English text
- ✅ **Language detection** - Auto-detect bn/en/mixed
- ✅ **Dataset tools** - Load, save, split, filter datasets
- ✅ **Readability check** - Estimate reading level
- ✅ **Safety check** - Content appropriateness
- ✅ **CLI interface** - Command-line tools

---

## 🚧 What Needs Models

These features have complete APIs but need trained models:
- Tokenization
- Text generation
- Translation
- Advanced classification

---

## 🎓 Common Tasks

### Run Tests
```bash
pytest tests/ -v
```

### Collect Sample Data
```bash
python scripts/collect_data.py --source sample --output data/raw/
```

### Prepare Data
```bash
python scripts/prepare_data.py --input data/raw/ --output datasets/processed/
```

### Use CLI
```bash
bilingual normalize --text "আমি স্কুলে যাই।" --lang bn
bilingual readability --text "আমি স্কুলে যাই।" --lang bn
```

---

## 📊 Project Status

**Current Phase**: Foundation Complete ✅  
**Next Phase**: Data Collection & Model Training

### Completion Status
- ✅ Phase 0: Project Setup - 100%
- ✅ Phase 3: Package Engineering - 100%
- ✅ Phase 4: Documentation - 90%
- 🚧 Phase 1: Data Strategy - 70%
- 🚧 Phase 2: Modeling - 50%

See **[PROJECT_STATUS.md](PROJECT_STATUS.md)** for details.

---

## 🤝 Contributing

We welcome contributions! Areas needing help:
- 📊 Data collection (Bangla/English corpora)
- 🤖 Model training
- 📝 Documentation (especially Bangla)
- 🧪 Testing
- 🐛 Bug fixes

See **[CONTRIBUTING.md](CONTRIBUTING.md)** for guidelines.

---

## 📞 Need Help?

- 📖 **Documentation**: [docs/en/README.md](docs/en/README.md)
- 🗺️ **Navigation**: [PROJECT_MAP.md](PROJECT_MAP.md)
- 📋 **Summary**: [SUMMARY.md](SUMMARY.md)
- 📊 **Status**: [PROJECT_STATUS.md](PROJECT_STATUS.md)

---

## 🎉 Next Steps

1. **Read** [GETTING_STARTED.md](GETTING_STARTED.md)
2. **Try** [examples/basic_usage.py](examples/basic_usage.py)
3. **Explore** [docs/en/quickstart.md](docs/en/quickstart.md)
4. **Contribute** [CONTRIBUTING.md](CONTRIBUTING.md)

---

**Built with ❤️ for the Bangla & English NLP community**

*Last updated: 2025-10-04*
