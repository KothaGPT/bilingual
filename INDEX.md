# 📑 Bilingual Project - Complete Index

**Quick reference to all project files and resources**

---

## 🌟 START HERE

| File | Description | Audience |
|------|-------------|----------|
| **[README_FIRST.md](README_FIRST.md)** | 👋 **START HERE** - Quick orientation | Everyone |
| **[README.md](README.md)** | Main project overview (EN/BN) | Everyone |
| **[GETTING_STARTED.md](GETTING_STARTED.md)** | Quick start guide | New users |

---

## 📚 Documentation Files

### User Documentation
| File | Description |
|------|-------------|
| [GETTING_STARTED.md](GETTING_STARTED.md) | Quick start guide (5 minutes) |
| [SETUP.md](SETUP.md) | Installation and setup instructions |
| [docs/en/README.md](docs/en/README.md) | English documentation home |
| [docs/en/quickstart.md](docs/en/quickstart.md) | English quick start tutorial |
| [docs/bn/README.md](docs/bn/README.md) | বাংলা ডকুমেন্টেশন হোম |
| [docs/bn/quickstart.md](docs/bn/quickstart.md) | বাংলা দ্রুত শুরু টিউটোরিয়াল |

### Project Documentation
| File | Description |
|------|-------------|
| [ROADMAP.md](ROADMAP.md) | Project roadmap and phases (EN/BN) |
| [PROJECT_STATUS.md](PROJECT_STATUS.md) | Current status and progress |
| [PROJECT_MAP.md](PROJECT_MAP.md) | Navigation guide |
| [SUMMARY.md](SUMMARY.md) | Complete project summary |
| [COMPLETION_REPORT.md](COMPLETION_REPORT.md) | Detailed completion report |
| [INDEX.md](INDEX.md) | This file - complete index |

### Governance Documentation
| File | Description |
|------|-------------|
| [LICENSE](LICENSE) | Apache 2.0 License |
| [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) | Community guidelines (EN/BN) |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Contribution guide (EN/BN) |

---

## 📦 Core Package Files

### Main Package (`bilingual/`)
| File | Description | Lines |
|------|-------------|-------|
| [bilingual/__init__.py](bilingual/__init__.py) | Package initialization | ~20 |
| [bilingual/api.py](bilingual/api.py) | High-level API | ~300 |
| [bilingual/normalize.py](bilingual/normalize.py) | Text normalization | ~350 |
| [bilingual/tokenizer.py](bilingual/tokenizer.py) | Tokenization utilities | ~300 |
| [bilingual/data_utils.py](bilingual/data_utils.py) | Dataset management | ~400 |
| [bilingual/evaluation.py](bilingual/evaluation.py) | Evaluation metrics | ~200 |
| [bilingual/cli.py](bilingual/cli.py) | Command-line interface | ~200 |

### Model Modules (`bilingual/models/`)
| File | Description | Lines |
|------|-------------|-------|
| [bilingual/models/__init__.py](bilingual/models/__init__.py) | Models package init | ~10 |
| [bilingual/models/loader.py](bilingual/models/loader.py) | Model loading | ~150 |
| [bilingual/models/lm.py](bilingual/models/lm.py) | Language models | ~100 |
| [bilingual/models/translate.py](bilingual/models/translate.py) | Translation models | ~100 |

---

## 🛠️ Scripts

| File | Description | Purpose |
|------|-------------|---------|
| [scripts/collect_data.py](scripts/collect_data.py) | Data collection | Get data from various sources |
| [scripts/prepare_data.py](scripts/prepare_data.py) | Data preprocessing | Clean and split datasets |
| [scripts/train_tokenizer.py](scripts/train_tokenizer.py) | Tokenizer training | Train SentencePiece model |

---

## 🧪 Tests

| File | Description | Tests |
|------|-------------|-------|
| [tests/__init__.py](tests/__init__.py) | Test package init | - |
| [tests/test_normalize.py](tests/test_normalize.py) | Normalization tests | 15+ |
| [tests/test_tokenizer.py](tests/test_tokenizer.py) | Tokenization tests | 5+ |
| [tests/test_data_utils.py](tests/test_data_utils.py) | Dataset tests | 15+ |
| [tests/test_api.py](tests/test_api.py) | API tests | 10+ |

---

## 💡 Examples

| File | Description |
|------|-------------|
| [examples/basic_usage.py](examples/basic_usage.py) | Comprehensive usage examples |

---

## ⚙️ Configuration Files

| File | Description |
|------|-------------|
| [pyproject.toml](pyproject.toml) | Package configuration and dependencies |
| [Makefile](Makefile) | Build automation commands |
| [.gitignore](.gitignore) | Git ignore rules |
| [bilingual/py.typed](bilingual/py.typed) | Type hints marker (PEP 561) |

---

## 🔄 CI/CD Files

| File | Description |
|------|-------------|
| [.github/workflows/ci.yml](.github/workflows/ci.yml) | GitHub Actions CI/CD pipeline |
| [.github/ISSUE_TEMPLATE/bug_report.md](.github/ISSUE_TEMPLATE/bug_report.md) | Bug report template (EN/BN) |
| [.github/ISSUE_TEMPLATE/feature_request.md](.github/ISSUE_TEMPLATE/feature_request.md) | Feature request template (EN/BN) |
| [.github/pull_request_template.md](.github/pull_request_template.md) | PR template (EN/BN) |

---

## 📂 Directories

| Directory | Description | Contents |
|-----------|-------------|----------|
| `bilingual/` | Core package | 7 modules + models/ |
| `bilingual/models/` | Model implementations | 3 modules |
| `scripts/` | Utility scripts | 3 scripts |
| `tests/` | Test suite | 4 test files |
| `docs/` | Documentation | en/ and bn/ |
| `docs/en/` | English docs | 2+ files |
| `docs/bn/` | Bangla docs | 2+ files |
| `examples/` | Example scripts | 1 file |
| `data/` | Data storage | raw/ and processed/ |
| `data/raw/` | Raw data | Empty (ready) |
| `data/processed/` | Processed data | Empty (ready) |
| `datasets/` | Dataset storage | Empty (ready) |
| `models/` | Model storage | Empty (ready) |
| `.github/` | GitHub config | workflows/ and templates |

---

## 🎯 Quick Reference by Task

### "I want to..."

#### Learn About the Project
- Start: [README_FIRST.md](README_FIRST.md)
- Overview: [README.md](README.md)
- Details: [SUMMARY.md](SUMMARY.md)

#### Install and Use
- Install: [SETUP.md](SETUP.md)
- Quick start: [GETTING_STARTED.md](GETTING_STARTED.md)
- Tutorial: [docs/en/quickstart.md](docs/en/quickstart.md)
- Examples: [examples/basic_usage.py](examples/basic_usage.py)

#### Understand the Code
- API: [bilingual/api.py](bilingual/api.py)
- Normalization: [bilingual/normalize.py](bilingual/normalize.py)
- Datasets: [bilingual/data_utils.py](bilingual/data_utils.py)
- Navigation: [PROJECT_MAP.md](PROJECT_MAP.md)

#### Contribute
- Guidelines: [CONTRIBUTING.md](CONTRIBUTING.md)
- Status: [PROJECT_STATUS.md](PROJECT_STATUS.md)
- Roadmap: [ROADMAP.md](ROADMAP.md)

#### Train Models
- Collect data: [scripts/collect_data.py](scripts/collect_data.py)
- Prepare data: [scripts/prepare_data.py](scripts/prepare_data.py)
- Train tokenizer: [scripts/train_tokenizer.py](scripts/train_tokenizer.py)

#### Test
- Run tests: `pytest tests/`
- Test files: [tests/](tests/)

#### Report Issues
- Bug report: [.github/ISSUE_TEMPLATE/bug_report.md](.github/ISSUE_TEMPLATE/bug_report.md)
- Feature request: [.github/ISSUE_TEMPLATE/feature_request.md](.github/ISSUE_TEMPLATE/feature_request.md)

---

## 📊 File Statistics

### By Type
- **Python files**: 17 (2,641+ lines)
- **Markdown files**: 20+
- **Config files**: 5
- **Total files**: 40+

### By Category
- **Documentation**: 20+ files
- **Source code**: 10 files
- **Tests**: 4 files
- **Scripts**: 3 files
- **Configuration**: 5 files
- **Templates**: 3 files

---

## 🔍 File Search Guide

### By Extension
- **`.py`** - Python source code
- **`.md`** - Documentation (Markdown)
- **`.toml`** - Configuration (TOML)
- **`.yml`** - CI/CD workflows (YAML)

### By Prefix
- **`README`** - Overview and introduction
- **`PROJECT_`** - Project-level documentation
- **`test_`** - Test files
- **`CONTRIBUTING`** - Contribution guidelines
- **`CODE_OF_CONDUCT`** - Community guidelines

### By Location
- **Root directory** - Main documentation
- **`bilingual/`** - Core package code
- **`scripts/`** - Utility scripts
- **`tests/`** - Test suite
- **`docs/`** - Detailed documentation
- **`.github/`** - GitHub configuration

---

## 🌐 Language Support

### English Files
- All root documentation
- `docs/en/` directory
- Code comments and docstrings

### Bangla Files
- `docs/bn/` directory
- Sections in CODE_OF_CONDUCT.md
- Sections in CONTRIBUTING.md
- Sections in ROADMAP.md
- Sections in README.md

### Bilingual Files
- CODE_OF_CONDUCT.md
- CONTRIBUTING.md
- ROADMAP.md
- README.md
- Issue/PR templates

---

## 📈 Completion Status

### ✅ Complete (100%)
- Core package structure
- API implementation
- Text normalization
- Dataset utilities
- CLI interface
- Test suite
- CI/CD pipeline
- English documentation
- Bangla documentation
- Governance files

### 🚧 Partial (50-90%)
- Model infrastructure (50%)
- Data collection (70%)
- Documentation (90%)
- Testing (80%)

### ⏳ Pending (0-50%)
- Trained models (0%)
- Large-scale corpus (0%)
- Production deployment (0%)

---

## 🎓 Recommended Reading Order

### For New Users
1. [README_FIRST.md](README_FIRST.md)
2. [GETTING_STARTED.md](GETTING_STARTED.md)
3. [docs/en/quickstart.md](docs/en/quickstart.md)
4. [examples/basic_usage.py](examples/basic_usage.py)

### For Contributors
1. [README.md](README.md)
2. [CONTRIBUTING.md](CONTRIBUTING.md)
3. [PROJECT_STATUS.md](PROJECT_STATUS.md)
4. [ROADMAP.md](ROADMAP.md)

### For Developers
1. [PROJECT_MAP.md](PROJECT_MAP.md)
2. [bilingual/api.py](bilingual/api.py)
3. [bilingual/normalize.py](bilingual/normalize.py)
4. [SUMMARY.md](SUMMARY.md)

### For Researchers
1. [SETUP.md](SETUP.md)
2. [scripts/](scripts/)
3. [ROADMAP.md](ROADMAP.md)
4. [bilingual/models/](bilingual/models/)

---

## 🔗 External Links

- **GitHub**: (To be added)
- **PyPI**: (To be added)
- **Documentation**: (To be added)
- **Issues**: (To be added)

---

## 📝 Notes

- All paths are relative to project root
- File sizes and line counts are approximate
- Status indicators: ✅ Complete, 🚧 Partial, ⏳ Pending
- Last updated: 2025-10-04

---

**This index is your complete reference to the bilingual project.**

*For navigation help, see [PROJECT_MAP.md](PROJECT_MAP.md)*
