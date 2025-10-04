# Bilingual Project Map 🗺️

A visual guide to navigating the bilingual package.

---

## 📁 Project Structure at a Glance

```
bilingual/
│
├── 📚 DOCUMENTATION (Start Here!)
│   ├── README.md              ⭐ Main entry point (EN/BN)
│   ├── GETTING_STARTED.md     🚀 Quick start guide
│   ├── SETUP.md               🔧 Installation & setup
│   ├── ROADMAP.md             🗺️ Project roadmap
│   ├── PROJECT_STATUS.md      📊 Current status
│   ├── SUMMARY.md             📋 Complete summary
│   └── PROJECT_MAP.md         📍 This file
│
├── 🤝 GOVERNANCE
│   ├── LICENSE                ⚖️ Apache 2.0
│   ├── CODE_OF_CONDUCT.md     👥 Community guidelines
│   └── CONTRIBUTING.md        🤲 How to contribute
│
├── 📦 CORE PACKAGE (bilingual/)
│   ├── __init__.py            Package entry point
│   ├── api.py                 ⭐ High-level API (START HERE)
│   ├── normalize.py           🧹 Text normalization
│   ├── tokenizer.py           ✂️ Tokenization
│   ├── data_utils.py          📊 Dataset management
│   ├── evaluation.py          📈 Evaluation metrics
│   ├── cli.py                 💻 Command-line interface
│   └── models/                🤖 Model implementations
│       ├── loader.py          Model loading
│       ├── lm.py              Language models
│       └── translate.py       Translation
│
├── 🛠️ SCRIPTS
│   ├── collect_data.py        📥 Data collection
│   ├── prepare_data.py        🔄 Data preprocessing
│   └── train_tokenizer.py     🎓 Tokenizer training
│
├── 🧪 TESTS
│   ├── test_normalize.py      Test normalization
│   ├── test_tokenizer.py      Test tokenization
│   ├── test_data_utils.py     Test datasets
│   └── test_api.py            Test API
│
├── 📖 DOCS
│   ├── en/                    🇬🇧 English docs
│   │   ├── README.md
│   │   └── quickstart.md
│   └── bn/                    🇧🇩 Bangla docs
│       ├── README.md
│       └── quickstart.md
│
├── 💡 EXAMPLES
│   └── basic_usage.py         ⭐ Usage examples
│
├── 📂 DATA DIRECTORIES
│   ├── data/raw/              Raw data storage
│   ├── data/processed/        Processed data
│   ├── datasets/              Dataset storage
│   └── models/                Model storage
│
├── ⚙️ CONFIGURATION
│   ├── pyproject.toml         📦 Package config
│   ├── Makefile               🔨 Build automation
│   └── .gitignore             🚫 Git ignore
│
└── 🔄 CI/CD (.github/)
    ├── workflows/ci.yml       Automated testing
    ├── ISSUE_TEMPLATE/        Bug & feature templates
    └── pull_request_template.md
```

---

## 🎯 Quick Navigation Guide

### 👋 New to the Project?
**Start here:**
1. 📖 [README.md](README.md) - Overview
2. 🚀 [GETTING_STARTED.md](GETTING_STARTED.md) - Quick start
3. 💡 [examples/basic_usage.py](examples/basic_usage.py) - See it in action

### 💻 Want to Use the Package?
**Go to:**
1. 📚 [docs/en/quickstart.md](docs/en/quickstart.md) - English guide
2. 📚 [docs/bn/quickstart.md](docs/bn/quickstart.md) - Bangla guide
3. 🔧 [SETUP.md](SETUP.md) - Installation

### 🤝 Want to Contribute?
**Read:**
1. 🤲 [CONTRIBUTING.md](CONTRIBUTING.md) - Guidelines
2. 📊 [PROJECT_STATUS.md](PROJECT_STATUS.md) - What's needed
3. 🗺️ [ROADMAP.md](ROADMAP.md) - Future plans

### 🔬 Want to Train Models?
**Check:**
1. 🛠️ [scripts/collect_data.py](scripts/collect_data.py) - Get data
2. 🛠️ [scripts/prepare_data.py](scripts/prepare_data.py) - Process data
3. 🛠️ [scripts/train_tokenizer.py](scripts/train_tokenizer.py) - Train tokenizer

### 🧪 Want to Run Tests?
**Use:**
```bash
pytest tests/              # Run all tests
pytest tests/test_api.py   # Run specific test
make test                  # Using Makefile
```

### 📝 Want to Read Code?
**Start with:**
1. ⭐ [bilingual/api.py](bilingual/api.py) - High-level API
2. 🧹 [bilingual/normalize.py](bilingual/normalize.py) - Text processing
3. 📊 [bilingual/data_utils.py](bilingual/data_utils.py) - Dataset tools

---

## 🎓 Learning Paths

### Path 1: User (Just Want to Use It)
```
README.md
    ↓
GETTING_STARTED.md
    ↓
docs/en/quickstart.md
    ↓
examples/basic_usage.py
    ↓
Start using the package!
```

### Path 2: Contributor (Want to Help)
```
README.md
    ↓
CONTRIBUTING.md
    ↓
PROJECT_STATUS.md
    ↓
Pick an issue
    ↓
Submit a PR!
```

### Path 3: Researcher (Want to Train Models)
```
README.md
    ↓
SETUP.md
    ↓
scripts/collect_data.py
    ↓
scripts/prepare_data.py
    ↓
scripts/train_tokenizer.py
    ↓
Train your models!
```

### Path 4: Developer (Want to Understand the Code)
```
README.md
    ↓
bilingual/__init__.py
    ↓
bilingual/api.py
    ↓
bilingual/normalize.py
    ↓
bilingual/data_utils.py
    ↓
Explore other modules!
```

---

## 🔑 Key Files Explained

### 📦 Package Files

| File | Purpose | When to Use |
|------|---------|-------------|
| `bilingual/api.py` | High-level API | Using the package |
| `bilingual/normalize.py` | Text cleaning | Processing text |
| `bilingual/tokenizer.py` | Tokenization | Breaking text into tokens |
| `bilingual/data_utils.py` | Dataset tools | Managing datasets |
| `bilingual/cli.py` | CLI commands | Command-line usage |

### 🛠️ Script Files

| File | Purpose | When to Use |
|------|---------|-------------|
| `scripts/collect_data.py` | Get data | Starting a project |
| `scripts/prepare_data.py` | Process data | Before training |
| `scripts/train_tokenizer.py` | Train tokenizer | Creating tokenizer |

### 📚 Documentation Files

| File | Purpose | Audience |
|------|---------|----------|
| `README.md` | Overview | Everyone |
| `GETTING_STARTED.md` | Quick start | New users |
| `SETUP.md` | Installation | Developers |
| `ROADMAP.md` | Future plans | Contributors |
| `PROJECT_STATUS.md` | Current state | Contributors |
| `CONTRIBUTING.md` | How to help | Contributors |

---

## 🎨 Code Organization

### Module Dependencies
```
bilingual/
    ├── normalize.py (standalone)
    ├── tokenizer.py (standalone)
    ├── data_utils.py (standalone)
    ├── evaluation.py (uses data_utils)
    ├── models/
    │   ├── loader.py (standalone)
    │   ├── lm.py (uses loader)
    │   └── translate.py (uses loader)
    ├── api.py (uses all above)
    └── cli.py (uses api)
```

### Import Patterns
```python
# High-level usage (recommended)
from bilingual import bilingual_api as bb
bb.normalize_text(...)
bb.translate(...)

# Direct module usage (advanced)
from bilingual.normalize import normalize_text
from bilingual.data_utils import BilingualDataset

# Model usage
from bilingual.models.loader import load_model_from_name
```

---

## 📊 Statistics

### Code Metrics
- **Total Python files**: 17
- **Total lines of code**: ~2,641
- **Test files**: 4
- **Documentation files**: 12+
- **Languages**: Python, Markdown, YAML

### Package Features
- ✅ Text normalization
- ✅ Dataset management
- ✅ Readability checking
- ✅ Safety checking
- ✅ CLI interface
- 🚧 Tokenization (needs model)
- 🚧 Generation (needs model)
- 🚧 Translation (needs model)

### Documentation Coverage
- ✅ English documentation
- ✅ Bangla documentation
- ✅ Code examples
- ✅ API reference structure
- ✅ Contributing guide
- ✅ Setup guide

---

## 🚀 Common Commands

### Installation
```bash
pip install -e .                # Basic install
pip install -e ".[dev]"         # With dev tools
pip install -e ".[torch]"       # With PyTorch
```

### Data Workflow
```bash
make collect-data               # Collect sample data
make prepare-data               # Process data
python scripts/train_tokenizer.py  # Train tokenizer
```

### Testing
```bash
make test                       # Run tests
make test-cov                   # With coverage
pytest tests/test_api.py -v     # Specific test
```

### Code Quality
```bash
make format                     # Format code
make lint                       # Lint code
black bilingual/                # Format specific dir
```

### Usage
```bash
make example                    # Run examples
bilingual normalize --text "..." --lang bn
bilingual readability --text "..." --lang bn
```

---

## 🎯 Finding What You Need

### "I want to..."

| Goal | Go to |
|------|-------|
| **Install the package** | [SETUP.md](SETUP.md) |
| **Learn how to use it** | [GETTING_STARTED.md](GETTING_STARTED.md) |
| **See examples** | [examples/basic_usage.py](examples/basic_usage.py) |
| **Understand the API** | [bilingual/api.py](bilingual/api.py) |
| **Contribute code** | [CONTRIBUTING.md](CONTRIBUTING.md) |
| **Report a bug** | [.github/ISSUE_TEMPLATE/](/.github/ISSUE_TEMPLATE/) |
| **Train models** | [scripts/](scripts/) |
| **Run tests** | [tests/](tests/) |
| **Read docs (EN)** | [docs/en/](docs/en/) |
| **Read docs (BN)** | [docs/bn/](docs/bn/) |
| **Check status** | [PROJECT_STATUS.md](PROJECT_STATUS.md) |
| **See roadmap** | [ROADMAP.md](ROADMAP.md) |

---

## 💡 Tips for Navigation

1. **Start with README.md** - Always begin here
2. **Use the examples** - Learn by doing
3. **Check PROJECT_STATUS.md** - See what's ready
4. **Read the code** - It's well-documented
5. **Run the tests** - Understand behavior
6. **Ask questions** - Use GitHub Discussions

---

## 🌟 Quick Links

- 🏠 [Home](README.md)
- 🚀 [Get Started](GETTING_STARTED.md)
- 📖 [Docs (EN)](docs/en/README.md)
- 📖 [Docs (BN)](docs/bn/README.md)
- 🤝 [Contribute](CONTRIBUTING.md)
- 📊 [Status](PROJECT_STATUS.md)
- 🗺️ [Roadmap](ROADMAP.md)

---

**Happy exploring! / শুভ অন্বেষণ!**

*Last updated: 2025-10-04*
