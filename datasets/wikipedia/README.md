# Wikipedia Dataset for Bilingual Language Model Training

This directory contains Wikipedia data for training Bangla language models.

## 📁 Directory Structure

> Note: Some subdirectories shown below (for example `processed/bn_extracted/`, the split
> folders under `processed/`, `hf_dataset/`, and `checkpoints/`) are created automatically
> by the setup and preprocessing scripts. They may not exist immediately after cloning the
> repository and will appear after you run the workflow (see the Quick Start section).

```
datasets/wikipedia/
├── raw/                           # Raw Wikipedia dumps
│   └── bnwiki-latest-pages-articles.xml.bz2  (451.79 MB)
├── processed/                     # Processed & cleaned data
│   ├── bn_extracted/             # Extracted articles (WikiExtractor output)
│   ├── train/                    # Training split (80%)
│   │   └── bn_train.txt
│   ├── val/                      # Validation split (10%)
│   │   └── bn_val.txt
│   └── test/                     # Test split (10%)
│       └── bn_test.txt
├── hf_dataset/                   # HuggingFace Dataset format
│   ├── train/
│   ├── validation/
│   ├── test/
│   └── dataset_info.json
├── bilingual/                    # Bilingual aligned data (optional)
├── checkpoints/                  # Processing checkpoints
├── PROGRESS.md                   # Current progress tracker
├── CHECKLIST.md                  # Step-by-step checklist
└── README.md                     # This file
```

## 🚀 Quick Start

### 1. Download Wikipedia Dump
```bash
make -f Makefile.wiki download-bn
```

### 2. Preprocess Data
```bash
make -f Makefile.wiki preprocess
```

### 3. Monitor Progress
```bash
make -f Makefile.wiki monitor-watch
```

### 4. Validate Quality
```bash
make -f Makefile.wiki validate
```

### 5. Prepare for Training
```bash
make -f Makefile.wiki prepare-hf
```

## 📊 Current Status

**Extraction Progress:** 300,000+ pages preprocessed (IN PROGRESS)

**Estimated Completion:** 20-40 minutes

**Next Step:** Validation after extraction completes

See `PROGRESS.md` for detailed status and `CHECKLIST.md` for complete workflow.

## 🛠 Available Tools

### Monitoring
- **`monitor_wiki_extraction.py`** - Track extraction progress
  ```bash
  python3 scripts/monitor_wiki_extraction.py
  python3 scripts/monitor_wiki_extraction.py --watch
  ```

### Validation
- **`validate_wiki_dataset.py`** - Validate dataset quality
  ```bash
  python3 scripts/validate_wiki_dataset.py --data datasets/wikipedia/processed
  ```

### Dataset Preparation
- **`prepare_hf_dataset.py`** - Create HuggingFace Dataset
  ```bash
  python3 scripts/prepare_hf_dataset.py --input datasets/wikipedia/processed
  ```

## 📈 Expected Dataset Statistics

| Metric | Estimated Value |
|--------|----------------|
| Total Articles | 150,000 - 200,000 |
| Total Sentences | 1,000,000 - 2,000,000 |
| Train Sentences | 800,000 - 1,600,000 |
| Val Sentences | 100,000 - 200,000 |
| Test Sentences | 100,000 - 200,000 |
| Avg Sentence Length | 40-50 characters |
| Total Size | 500 MB - 1 GB |

## 🔍 Data Quality

### Preprocessing Steps
1. ✅ **Extraction** - WikiExtractor removes markup
2. ✅ **Cleaning** - Remove citations, templates, links
3. ✅ **Normalization** - Unicode NFC normalization
4. ✅ **Tokenization** - Sentence-level splitting
5. ✅ **Filtering** - Length-based quality filtering
6. ✅ **Splitting** - 80/10/10 train/val/test split

### Quality Filters
- **Min sentence length:** 10 characters
- **Max sentence length:** 500 characters
- **Encoding:** UTF-8
- **Normalization:** Unicode NFC

## 📚 Documentation

- **[WORKFLOW.md](../../docs/WIKIPEDIA_WORKFLOW.md)** - Complete workflow guide
- **[PROGRESS.md](PROGRESS.md)** - Current progress tracker
- **[CHECKLIST.md](CHECKLIST.md)** - Step-by-step checklist
- **[TRAINING_ROADMAP.md](../../docs/WIKIPEDIA_TRAINING_ROADMAP.md)** - Training roadmap

## 🎯 Usage Examples

### Load Processed Data (Python)
```python
# Load sentences from text file
with open('datasets/wikipedia/processed/train/bn_train.txt', 'r', encoding='utf-8') as f:
    sentences = [line.strip() for line in f if line.strip()]

print(f"Loaded {len(sentences):,} sentences")
```

### Load HuggingFace Dataset
```python
from datasets import load_from_disk

# Load dataset
dataset = load_from_disk('datasets/wikipedia/hf_dataset')

# Access splits
train_data = dataset['train']
val_data = dataset['validation']
test_data = dataset['test']

# Iterate
for example in train_data:
    print(example['text'])
```

### Use with Transformers
```python
from transformers import AutoTokenizer, AutoModelForMaskedLM
from datasets import load_from_disk

# Load dataset
dataset = load_from_disk('datasets/wikipedia/hf_dataset')

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('ai4bharat/indic-bert')

# Tokenize
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True)
```

## 🔧 Troubleshooting

### Extraction Taking Too Long?
This is normal. Expected times:
- Template collection: 5-10 minutes
- Article extraction: 20-60 minutes

Monitor progress:
```bash
make -f Makefile.wiki monitor-watch
```

### Validation Fails?
Check specific issues:
```bash
make -f Makefile.wiki validate
```

### Need to Resume?
If extraction was interrupted, you may need to restart:
```bash
# Clean and restart
make -f Makefile.wiki clean-extracted
make -f Makefile.wiki preprocess
```

## 📝 Notes

### Indic NLP Library
The warning about indic-nlp-library is expected:
```
WARNING:root:indic-nlp-library not available. Using basic tokenization.
```

Basic tokenization is sufficient for initial training. The library is installed but may need additional configuration for advanced features.

### Disk Space
Ensure you have sufficient disk space:
- Raw dump: ~450 MB
- Extracted articles: ~300-500 MB
- Processed data: ~200-400 MB
- HF dataset: ~200-400 MB
- **Total required:** ~1.5-2 GB

### Processing Time
Total processing time: 30-90 minutes
- Download: 5-15 minutes (depends on connection)
- Extraction: 30-60 minutes (depends on dump size)
- Processing: 5-15 minutes

## 🎓 Next Steps

After dataset preparation:

1. **Validate** - Ensure data quality
2. **Train** - Train language model
3. **Evaluate** - Test model performance
4. **Deploy** - Deploy to production

See the complete workflow in `docs/WIKIPEDIA_WORKFLOW.md`.

## 📞 Support

For issues or questions:
- Check `CHECKLIST.md` for common issues
- Review `docs/WIKIPEDIA_WORKFLOW.md` for detailed guide
- See troubleshooting section above

## 📄 License

Wikipedia content is licensed under CC BY-SA 3.0.
See: https://dumps.wikimedia.org/legal.html

---

**Last Updated:** 2025-10-23 16:30:00  
**Status:** Extraction in progress (300k+ pages)  
**Next Milestone:** Validation after extraction completes
