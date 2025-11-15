# Wikipedia Training Progress

**Last Updated:** 2025-10-23 16:27:00

## 📊 Current Status

### ✅ Completed Steps

1. **Dependencies Installed**
   - ✅ wikiextractor (3.0.6)
   - ✅ indic-nlp-library (0.92)
   - ✅ All required packages

2. **Data Download**
   - ✅ Bangla Wikipedia dump downloaded
   - ✅ File: `bnwiki-latest-pages-articles.xml.bz2` (451.79 MB)
   - ✅ Location: `datasets/wikipedia/raw/`

### 🔄 In Progress

3. **Preprocessing & Extraction**
   - 🔄 WikiExtractor running
   - 🔄 Template collection phase
   - 🔄 200,000+ pages preprocessed
   - ⏳ Estimated completion: 30-60 minutes

### ⏳ Pending Steps

4. **Post-Processing**
   - ⏳ Sentence tokenization
   - ⏳ Text cleaning & normalization
   - ⏳ Train/val/test split (80/10/10)

5. **Validation**
   - ⏳ Quality checks
   - ⏳ Sample verification
   - ⏳ Statistics generation

6. **Dataset Preparation**
   - ⏳ HuggingFace Dataset creation
   - ⏳ Dataset card generation

7. **Model Training**
   - ⏳ Base model selection
   - ⏳ Training configuration
   - ⏳ Model training
   - ⏳ Evaluation

## 🎯 Next Actions

### Immediate (While Extraction Runs)

1. **Monitor Progress**
   ```bash
   make -f Makefile.wiki monitor-watch
   ```

2. **Prepare Training Scripts**
   - Review `scripts/train_wiki_lm.py`
   - Configure training parameters
   - Set up checkpointing

### After Extraction Completes

1. **Validate Dataset**
   ```bash
   make -f Makefile.wiki validate
   ```

2. **Prepare HF Dataset**
   ```bash
   make -f Makefile.wiki prepare-hf
   ```

3. **Test Training**
   ```bash
   make -f Makefile.wiki train-test
   ```

4. **Full Training**
   ```bash
   make -f Makefile.wiki train
   ```

## 📈 Expected Metrics

### Dataset Size (Estimated)

- **Total Articles:** ~150,000 - 200,000
- **Total Sentences:** ~1,000,000 - 2,000,000
- **Train Split:** ~800,000 - 1,600,000 sentences
- **Val Split:** ~100,000 - 200,000 sentences
- **Test Split:** ~100,000 - 200,000 sentences

### Training Time (Estimated)

- **Test Training (1 epoch):** 1-2 hours (GPU) / 8-12 hours (CPU)
- **Full Training (3 epochs):** 3-6 hours (GPU) / 24-36 hours (CPU)
- **Production Training (5 epochs):** 5-10 hours (GPU) / 40-60 hours (CPU)

## 🛠 Tools & Scripts

### Monitoring
- `scripts/monitor_wiki_extraction.py` - Track extraction progress
- `make -f Makefile.wiki monitor` - Quick status check
- `make -f Makefile.wiki monitor-watch` - Real-time monitoring

### Validation
- `scripts/validate_wiki_dataset.py` - Quality validation
- `make -f Makefile.wiki validate` - Run validation

### Dataset Preparation
- `scripts/prepare_hf_dataset.py` - Create HF dataset
- `make -f Makefile.wiki prepare-hf` - Prepare dataset

### Training
- `scripts/train_wiki_lm.py` - Train language model
- `make -f Makefile.wiki train-test` - Test training
- `make -f Makefile.wiki train` - Full training

## 📝 Notes

### Indic NLP Warning
The warning about indic-nlp-library is expected and safe:
```
WARNING:root:indic-nlp-library not available. Using basic tokenization.
```

The library is installed but may need additional configuration. Basic tokenization is sufficient for initial training.

### Extraction Progress
WikiExtractor processes in two phases:
1. **Template Collection** (5-10 min) - Analyzes Wikipedia templates
2. **Article Extraction** (20-60 min) - Extracts and cleans articles

Current phase: Template collection → Article extraction

### Quality Considerations
- Minimum sentence length: 10 characters
- Maximum sentence length: 500 characters
- Unicode normalization: NFC
- Indic normalization: Available but using basic fallback

## 🔍 Monitoring Commands

```bash
# Check current status
make -f Makefile.wiki monitor

# Watch in real-time
make -f Makefile.wiki monitor-watch

# Check extraction directory
ls -lh datasets/wikipedia/processed/bn_extracted/

# Count extracted files
find datasets/wikipedia/processed/bn_extracted -name "wiki_*" | wc -l

# Check processed sentences (after completion)
wc -l datasets/wikipedia/processed/train/bn_train.txt
```

## 📚 Documentation

- **Workflow Guide:** `docs/WIKIPEDIA_WORKFLOW.md`
- **Training Roadmap:** `docs/WIKIPEDIA_TRAINING_ROADMAP.md`
- **API Docs:** `docs/api/`
- **Examples:** `examples/`

## ⚠️ Important Notes

1. **Do not interrupt** the extraction process - it cannot resume from checkpoint
2. **Monitor disk space** - processed data will be ~500 MB - 1 GB
3. **GPU recommended** for training - CPU training is very slow
4. **Save checkpoints** frequently during training
5. **Validate before training** to catch data quality issues early

## 🎓 Learning Resources

- WikiExtractor: https://github.com/attardi/wikiextractor
- Indic NLP: https://github.com/anoopkunchukuttan/indic_nlp_library
- HuggingFace Datasets: https://huggingface.co/docs/datasets
- Transformers Training: https://huggingface.co/docs/transformers/training

---

**Status Legend:**
- ✅ Completed
- 🔄 In Progress
- ⏳ Pending
- ⚠️ Issue/Warning
- ❌ Failed
