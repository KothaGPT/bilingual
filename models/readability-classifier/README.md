# Readability Classifier

## Task: Age-appropriate readability classification

### Labels
- 6-8
- 9-10
- 11-12
- general

### Training
To train this model, install transformers and run:

```bash
pip install transformers datasets
python scripts/train_classifier.py --task readability --data datasets/processed/
```

### Usage
```python
from bilingual import api as bb

# Use the classifier
result = bb.readability_check("Your text here")
print(result)
```

---
language:
- bn
- en
license: apache-2.0
tags:
- bangla
- bengali
- english
- readability
- classifier
- text-quality
- nlp
- transformers
datasets:
- wikipedia
- custom
metrics:
- accuracy
- f1
- precision
- recall
---

# Bangla–English Readability Classifier

This model classifies Bangla and English text into readability levels — *simple*, *medium*, or *complex*.  
It is part of the **KothaGPT Bilingual NLP suite**, trained on parallel corpora combining **Bangla Wikipedia**, **news articles**, and **simplified text datasets**.

---

## 🧠 Model Description

- **Model Type:** Text classifier (sequence classification)
- **Base Architecture:** BERT (Multilingual / IndicBERT variant)
- **Languages:** Bangla (bn), English (en)
- **Task:** Readability prediction (3-way classification)
- **License:** Apache 2.0
- **Framework:** PyTorch + Hugging Face Transformers

---

## 🧩 Intended Use

- Educational content simplification
- Readability filtering in datasets
- Adaptive text generation evaluation
- Research in Bangla and bilingual readability modeling

---

## 🧾 Training Data

| Source | Description | Size |
|--------|--------------|------|
| Bangla Wikipedia | Encyclopedic formal text | 800K sentences |
| News Articles | Mixed domain readability | 200K sentences |
| Simplified Text Corpora | Easy Bangla + English parallel samples | 100K sentences |

**Labels:**
- `0`: Simple
- `1`: Medium
- `2`: Complex

---

## ⚙️ Training Procedure

**Preprocessing:**
- Unicode normalization
- Sentence length filtering (5–200 tokens)
- Bilingual tokenization using SentencePiece
- Balanced sampling across readability levels

**Hyperparameters:**
- Epochs: 4  
- Batch size: 16  
- Learning rate: 3e-5  
- Optimizer: AdamW  
- Sequence length: 256  
- Dropout: 0.1  
- Mixed precision: FP16  

---

## 🧪 Evaluation

| Metric | Dev | Test |
|--------|-----|------|
| Accuracy | 0.88 | 0.86 |
| F1 (macro) | 0.87 | 0.85 |
| Precision | 0.88 | 0.86 |
| Recall | 0.87 | 0.84 |

**Confusion matrix trends:**
- Some overlap between *medium* and *complex* categories.
- Simpler texts (Wikipedia Simple or translated corpora) perform best.

---

## 🚀 Usage Example

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_id = "KothaGPT/bn-en-readability-classifier"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)

text = "বাংলাদেশের রাজধানী ঢাকা শহরটি দেশের অর্থনৈতিক কেন্দ্র।"

inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
with torch.no_grad():
    logits = model(**inputs).logits
    pred = torch.argmax(logits, dim=-1).item()

labels = ["simple", "medium", "complex"]
print(f"Predicted readability: {labels[pred]}")
