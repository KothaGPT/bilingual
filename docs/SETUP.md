# Setup Guide | সেটআপ গাইড

## English Version

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) Git for version control

### Installation Steps

#### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_ORG/bilingual.git
cd bilingual
```

#### 2. Create a Virtual Environment (Recommended)

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n bilingual python=3.10
conda activate bilingual
```

#### 3. Install the Package

For basic usage:
```bash
pip install -e .
```

For development (includes testing and linting tools):
```bash
pip install -e ".[dev]"
```

For PyTorch support:
```bash
pip install -e ".[torch]"
```

For all dependencies:
```bash
pip install -e ".[all]"
```

#### 4. Verify Installation

```bash
python -c "import bilingual; print(bilingual.__version__)"
bilingual --version
```

### Quick Start Workflow

#### Step 1: Collect Sample Data

```bash
python scripts/collect_data.py --source sample --output data/raw/
```

This creates sample bilingual data in `data/raw/`:
- `sample_bn.txt` - Bangla sentences
- `sample_en.txt` - English sentences
- `parallel_corpus.jsonl` - Parallel translations

#### Step 2: Prepare Data

```bash
python scripts/prepare_data.py \
    --input data/raw/ \
    --output datasets/processed/ \
    --split 0.8 0.1 0.1
```

This processes and splits the data into:
- `train.jsonl` (80%)
- `validation.jsonl` (10%)
- `test.jsonl` (10%)

#### Step 3: Train Tokenizer

```bash
python scripts/train_tokenizer.py \
    --input data/raw/sample_bn.txt data/raw/sample_en.txt \
    --model-prefix bilingual_sp \
    --vocab-size 32000 \
    --output-dir models/tokenizer/
```

This creates:
- `models/tokenizer/bilingual_sp.model` - SentencePiece model
- `models/tokenizer/bilingual_sp.vocab` - Vocabulary file

#### Step 4: Use the Package

```python
from bilingual import bilingual_api as bb

# Normalize text
text = bb.normalize_text("আমি স্কুলে যাই।", lang="bn")
print(text)

# Check readability
result = bb.readability_check(text, lang="bn")
print(f"Reading level: {result['level']}")
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=bilingual --cov-report=html

# Run specific test file
pytest tests/test_normalize.py -v
```

### Code Quality Checks

```bash
# Format code
black bilingual/ tests/ scripts/

# Sort imports
isort bilingual/ tests/ scripts/

# Lint code
flake8 bilingual/ tests/ scripts/

# Type check
mypy bilingual/
```

### Building Documentation

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build docs (if using Sphinx)
cd docs/
make html
```

### Common Issues

#### Issue: `sentencepiece` not found

**Solution**: Install sentencepiece
```bash
pip install sentencepiece
```

#### Issue: Import errors

**Solution**: Make sure you installed the package in editable mode
```bash
pip install -e .
```

#### Issue: Tests failing

**Solution**: Install dev dependencies
```bash
pip install -e ".[dev]"
```

### Next Steps

1. Read the [Quick Start Guide](docs/en/quickstart.md)
2. Explore the [API Reference](docs/en/api.md)
3. Check the [Roadmap](ROADMAP.md) for upcoming features
4. Join the community and [contribute](CONTRIBUTING.md)

---

## বাংলা সংস্করণ

### পূর্বশর্ত

- Python 3.8 বা উচ্চতর
- pip প্যাকেজ ম্যানেজার
- (ঐচ্ছিক) সংস্করণ নিয়ন্ত্রণের জন্য Git

### ইনস্টলেশন পদক্ষেপ

#### 1. রিপোজিটরি ক্লোন করুন

```bash
git clone https://github.com/YOUR_ORG/bilingual.git
cd bilingual
```

#### 2. একটি ভার্চুয়াল এনভায়রনমেন্ট তৈরি করুন (প্রস্তাবিত)

```bash
# venv ব্যবহার করে
python -m venv venv
source venv/bin/activate  # Windows-এ: venv\Scripts\activate

# অথবা conda ব্যবহার করে
conda create -n bilingual python=3.10
conda activate bilingual
```

#### 3. প্যাকেজ ইনস্টল করুন

মৌলিক ব্যবহারের জন্য:
```bash
pip install -e .
```

ডেভেলপমেন্টের জন্য (টেস্টিং এবং লিন্টিং টুল অন্তর্ভুক্ত):
```bash
pip install -e ".[dev]"
```

PyTorch সমর্থনের জন্য:
```bash
pip install -e ".[torch]"
```

সমস্ত ডিপেন্ডেন্সির জন্য:
```bash
pip install -e ".[all]"
```

#### 4. ইনস্টলেশন যাচাই করুন

```bash
python -c "import bilingual; print(bilingual.__version__)"
bilingual --version
```

### দ্রুত শুরু ওয়ার্কফ্লো

#### ধাপ 1: নমুনা ডেটা সংগ্রহ করুন

```bash
python scripts/collect_data.py --source sample --output data/raw/
```

এটি `data/raw/`-এ নমুনা দ্বিভাষিক ডেটা তৈরি করে:
- `sample_bn.txt` - বাংলা বাক্য
- `sample_en.txt` - ইংরেজি বাক্য
- `parallel_corpus.jsonl` - সমান্তরাল অনুবাদ

#### ধাপ 2: ডেটা প্রস্তুত করুন

```bash
python scripts/prepare_data.py \
    --input data/raw/ \
    --output datasets/processed/ \
    --split 0.8 0.1 0.1
```

এটি ডেটা প্রক্রিয়া করে এবং বিভক্ত করে:
- `train.jsonl` (80%)
- `validation.jsonl` (10%)
- `test.jsonl` (10%)

#### ধাপ 3: টোকেনাইজার ট্রেন করুন

```bash
python scripts/train_tokenizer.py \
    --input data/raw/sample_bn.txt data/raw/sample_en.txt \
    --model-prefix bilingual_sp \
    --vocab-size 32000 \
    --output-dir models/tokenizer/
```

এটি তৈরি করে:
- `models/tokenizer/bilingual_sp.model` - SentencePiece মডেল
- `models/tokenizer/bilingual_sp.vocab` - শব্দভান্ডার ফাইল

#### ধাপ 4: প্যাকেজ ব্যবহার করুন

```python
from bilingual import bilingual_api as bb

# টেক্সট নরমালাইজ করুন
text = bb.normalize_text("আমি স্কুলে যাই।", lang="bn")
print(text)

# পঠনযোগ্যতা চেক করুন
result = bb.readability_check(text, lang="bn")
print(f"Reading level: {result['level']}")
```

### টেস্ট চালানো

```bash
# সমস্ত টেস্ট চালান
pytest tests/

# কভারেজ সহ চালান
pytest tests/ --cov=bilingual --cov-report=html

# নির্দিষ্ট টেস্ট ফাইল চালান
pytest tests/test_normalize.py -v
```

### পরবর্তী পদক্ষেপ

1. [দ্রুত শুরু গাইড](docs/bn/quickstart.md) পড়ুন
2. [API রেফারেন্স](docs/bn/api.md) অন্বেষণ করুন
3. আসন্ন ফিচারের জন্য [রোডম্যাপ](ROADMAP.md) দেখুন
4. সম্প্রদায়ে যোগ দিন এবং [অবদান রাখুন](CONTRIBUTING.md)
