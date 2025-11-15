# Wikipedia Training Checklist

Quick reference checklist for Wikipedia language model training pipeline.

## ✅ Phase 1: Setup & Download

- [x] Install dependencies (`wikiextractor`, `indic-nlp-library`)
- [x] Download Bangla Wikipedia dump (451.79 MB)
- [x] Verify dump file integrity
- [x] Create directory structure

**Commands:**
```bash
pip install wikiextractor indic-nlp-library
make -f Makefile.wiki download-bn
```

## 🔄 Phase 2: Extraction & Preprocessing (IN PROGRESS)

- [x] Start WikiExtractor
- [x] Template collection phase (200k+ pages processed)
- [ ] Article extraction phase
- [ ] Text cleaning & normalization
- [ ] Sentence tokenization
- [ ] Quality filtering
- [ ] Train/val/test split (80/10/10)

**Commands:**
```bash
# Start preprocessing
make -f Makefile.wiki preprocess

# Monitor progress (in another terminal)
make -f Makefile.wiki monitor-watch
```

**Current Status:** 300,000+ pages preprocessed

## ⏳ Phase 3: Validation (PENDING)

- [ ] Run dataset validation
- [ ] Check split ratios
- [ ] Verify sentence lengths
- [ ] Sample quality review
- [ ] Check for encoding issues
- [ ] Verify no empty files
- [ ] Generate validation report

**Commands:**
```bash
# After preprocessing completes
make -f Makefile.wiki validate

# Get JSON report
make -f Makefile.wiki validate-json > validation.json
```

## ⏳ Phase 4: Dataset Preparation (PENDING)

- [ ] Create HuggingFace Dataset
- [ ] Verify dataset structure
- [ ] Test with small sample
- [ ] Generate dataset card
- [ ] (Optional) Push to HuggingFace Hub

**Commands:**
```bash
# Test with 1000 samples first
make -f Makefile.wiki prepare-hf-test

# Full dataset preparation
make -f Makefile.wiki prepare-hf

# (Optional) Push to Hub
python3 scripts/prepare_hf_dataset.py \
    --input datasets/wikipedia/processed \
    --push-to-hub \
    --repo KothaGPT/bn-wikipedia
```

## ⏳ Phase 5: Model Training (PENDING)

### Pre-Training Checks

- [ ] Verify GPU availability
- [ ] Check disk space for checkpoints
- [ ] Review training parameters
- [ ] Set up TensorBoard
- [ ] Configure checkpoint saving

### Test Training

- [ ] Run 1-epoch test training
- [ ] Verify training loop works
- [ ] Check memory usage
- [ ] Validate checkpoint saving
- [ ] Review initial metrics

**Commands:**
```bash
# Quick test (1 epoch)
make -f Makefile.wiki train-test

# Monitor with TensorBoard
make -f Makefile.wiki tensorboard
```

### Full Training

- [ ] Start full training (3-5 epochs)
- [ ] Monitor training metrics
- [ ] Save checkpoints regularly
- [ ] Track validation loss
- [ ] Handle any errors/interruptions

**Commands:**
```bash
# Standard training (3 epochs)
make -f Makefile.wiki train

# Production training (5 epochs, optimized)
make -f Makefile.wiki train-production
```

## ⏳ Phase 6: Evaluation (PENDING)

- [ ] Evaluate on test set
- [ ] Calculate perplexity
- [ ] Test on sample sentences
- [ ] Compare with baseline
- [ ] Generate evaluation report
- [ ] Interactive testing

**Commands:**
```bash
# Automated evaluation
make -f Makefile.wiki evaluate

# Interactive testing
make -f Makefile.wiki interactive
```

## ⏳ Phase 7: Deployment (PENDING)

- [ ] Create model card
- [ ] Document training details
- [ ] Package model artifacts
- [ ] (Optional) Push to HuggingFace Hub
- [ ] Set up inference API
- [ ] Create usage examples

**Commands:**
```bash
# Generate model card
python3 scripts/huggingface/generate_model_card.py

# Prepare for deployment
python3 scripts/huggingface/prepare_model.py
```

## 📊 Expected Outputs

### After Preprocessing
```
datasets/wikipedia/processed/
├── bn_extracted/          # ~150-200k articles
├── train/
│   └── bn_train.txt      # ~800k-1.6M sentences
├── val/
│   └── bn_val.txt        # ~100k-200k sentences
└── test/
    └── bn_test.txt       # ~100k-200k sentences
```

### After HF Dataset Preparation
```
datasets/wikipedia/hf_dataset/
├── train/
├── validation/
├── test/
└── dataset_info.json
```

### After Training
```
models/wikipedia/base/
├── config.json
├── pytorch_model.bin
├── tokenizer_config.json
├── vocab.txt
├── training_args.bin
└── logs/
```

## 🔍 Monitoring Commands

### Check Extraction Progress
```bash
# Quick status
make -f Makefile.wiki monitor

# Real-time monitoring
make -f Makefile.wiki monitor-watch

# Manual checks
find datasets/wikipedia/processed/bn_extracted -name "wiki_*" | wc -l
ls -lh datasets/wikipedia/processed/
```

### Check Processing Status
```bash
# Count sentences in each split
wc -l datasets/wikipedia/processed/train/bn_train.txt
wc -l datasets/wikipedia/processed/val/bn_val.txt
wc -l datasets/wikipedia/processed/test/bn_test.txt

# Check file sizes
du -sh datasets/wikipedia/processed/*

# Sample random sentences
shuf -n 5 datasets/wikipedia/processed/train/bn_train.txt
```

### Monitor Training
```bash
# TensorBoard
make -f Makefile.wiki tensorboard

# Check GPU usage
nvidia-smi

# Monitor disk space
df -h
```

## ⚠️ Common Issues & Solutions

### Issue: Extraction is slow
- **Solution:** Normal for large dumps. Expected: 30-90 minutes
- **Monitor:** `make -f Makefile.wiki monitor-watch`

### Issue: Out of memory during training
- **Solution:** Reduce batch size or use gradient accumulation
```bash
python3 scripts/train_wiki_lm.py --batch-size 4 --gradient-accumulation-steps 2
```

### Issue: Validation fails
- **Solution:** Check validation report for specific issues
```bash
make -f Makefile.wiki validate
```

### Issue: Training interrupted
- **Solution:** Resume from last checkpoint
```bash
python3 scripts/train_wiki_lm.py --resume-from-checkpoint models/wikipedia/base/checkpoint-1000
```

## 📝 Quick Reference

### Essential Commands
```bash
# Download
make -f Makefile.wiki download-bn

# Preprocess
make -f Makefile.wiki preprocess

# Monitor
make -f Makefile.wiki monitor-watch

# Validate
make -f Makefile.wiki validate

# Prepare HF dataset
make -f Makefile.wiki prepare-hf

# Train
make -f Makefile.wiki train

# Evaluate
make -f Makefile.wiki evaluate
```

### Status Checks
```bash
# Overall status
make -f Makefile.wiki status

# Help
make -f Makefile.wiki help

# Info
make -f Makefile.wiki info
```

## 📚 Documentation

- **Workflow Guide:** `docs/WIKIPEDIA_WORKFLOW.md`
- **Progress Tracker:** `datasets/wikipedia/PROGRESS.md`
- **Training Roadmap:** `docs/WIKIPEDIA_TRAINING_ROADMAP.md`

## 🎯 Current Priority

**RIGHT NOW:** Extraction is running (300k+ pages processed)

**NEXT STEP:** Wait for extraction to complete, then validate

**ESTIMATED TIME:** 20-40 minutes until extraction completes

**RECOMMENDED ACTION:** Monitor progress in a separate terminal:
```bash
make -f Makefile.wiki monitor-watch
```

---

**Last Updated:** 2025-10-23 16:30:00  
**Current Phase:** Phase 2 - Extraction & Preprocessing (IN PROGRESS)  
**Next Phase:** Phase 3 - Validation (after extraction completes)
