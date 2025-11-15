#!/usr/bin/env python3
"""
Generate model card (README.md) for Hugging Face Hub.

Usage:
    python scripts/huggingface/generate_model_card.py --model models/huggingface_ready/bn-wikipedia-lm --type base
    python scripts/huggingface/generate_model_card.py --model models/huggingface_ready/bn-wikipedia-literary --type literary
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


MODEL_CARD_TEMPLATES = {
    "base": """---
language:
- bn
license: apache-2.0
tags:
- bangla
- bengali
- language-model
- masked-lm
- wikipedia
datasets:
- wikipedia
metrics:
- perplexity
widget:
- text: "বাংলাদেশের রাজধানী [MASK]"
---

# Bangla Wikipedia Language Model

This is a {model_type} model trained on Bangla Wikipedia for masked language modeling tasks.

## Model Description

- **Model Type:** {model_type}
- **Language:** Bangla (Bengali)
- **Training Data:** Bangla Wikipedia
- **Parameters:** {num_parameters:,}
- **Model Size:** {model_size_mb} MB

## Intended Use

This model is intended for:
- Fill-mask tasks in Bangla
- Bangla text embeddings
- Fine-tuning on downstream Bangla NLP tasks
- Research in Bangla language understanding

## Training Data

The model was trained on Bangla Wikipedia articles, which were:
1. Extracted using WikiExtractor
2. Cleaned and normalized
3. Tokenized using Indic NLP library
4. Split into train/validation/test sets (80/10/10)

**Dataset Statistics:**
- Total sentences: ~1-2M
- Vocabulary size: {vocab_size:,}
- Average sentence length: 50-150 characters

## Training Procedure

### Preprocessing

- HTML/markup removal
- Unicode normalization (NFC)
- Indic script normalization
- Sentence tokenization
- Filtering (10-500 characters)

### Training Hyperparameters

- **Model:** {architecture}
- **Epochs:** 3-5
- **Batch Size:** 16
- **Learning Rate:** 3e-5
- **Max Sequence Length:** 512
- **Optimizer:** AdamW
- **Mixed Precision:** FP16

### Training Infrastructure

- **GPU:** V100 / A100
- **Training Time:** 1-3 days
- **Framework:** PyTorch + Transformers

## Evaluation

**Perplexity on test set:** 20-40

## Usage

### Fill-Mask

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

# Load model
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")
model = AutoModelForMaskedLM.from_pretrained("{repo_id}")

# Fill mask
text = "বাংলাদেশের রাজধানী [MASK]"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)

# Get predictions
mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
mask_token_logits = outputs.logits[0, mask_token_index, :]
top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

for token in top_5_tokens:
    print(f"{{text.replace('[MASK]', tokenizer.decode([token]))}}")
```

### Get Embeddings

```python
from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("{repo_id}")
model = AutoModel.from_pretrained("{repo_id}")

text = "আমি বাংলায় কথা বলি"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state

# Mean pooling
sentence_embedding = embeddings.mean(dim=1)
print(f"Embedding shape: {{sentence_embedding.shape}}")
```

### Fine-tuning

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# Load model for classification
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")
model = AutoModelForSequenceClassification.from_pretrained("{repo_id}", num_labels=2)

# Your training code here
```

## Limitations

- Trained only on Wikipedia text (formal, encyclopedic style)
- May not perform well on informal or colloquial Bangla
- Limited to text available in Bangla Wikipedia
- May contain biases present in Wikipedia

## Ethical Considerations

- The model may reflect biases present in Wikipedia
- Should not be used for generating harmful or misleading content
- Users should validate outputs for their specific use case

## Citation

If you use this model, please cite:

```bibtex
@misc{{bn-wikipedia-lm,
  author = {{KothaGPT}},
  title = {{Bangla Wikipedia Language Model}},
  year = {{2025}},
  publisher = {{Hugging Face}},
  url = {{https://huggingface.co/{repo_id}}}
}}
```

## License

Apache 2.0

## Contact

For questions or issues, please open an issue on [GitHub](https://github.com/KothaGPT/bilingual).

## Acknowledgments

- **Wikimedia Foundation** for Wikipedia data
- **AI4Bharat** for Indic NLP tools
- **Hugging Face** for the Transformers library
""",
    "literary": """---
language:
- bn
license: apache-2.0
tags:
- bangla
- bengali
- language-model
- masked-lm
- wikipedia
- literary
- poetry
datasets:
- wikipedia
- custom
metrics:
- perplexity
widget:
- text: "কবিতার [MASK] সুন্দর"
---

# Bangla Wikipedia + Literary Language Model

This is a {model_type} model fine-tuned on Bangla literary texts after pre-training on Wikipedia.

## Model Description

- **Model Type:** {model_type}
- **Language:** Bangla (Bengali)
- **Training Data:** Bangla Wikipedia + Literary Corpus
- **Parameters:** {num_parameters:,}
- **Model Size:** {model_size_mb} MB

## Intended Use

This model is intended for:
- Literary text analysis
- Poetry and prose understanding
- Style-aware text generation
- Literary text embeddings
- Fine-tuning on literary NLP tasks

## Training Data

### Pre-training (Wikipedia)
- Bangla Wikipedia articles
- ~1-2M sentences
- Formal, encyclopedic text

### Fine-tuning (Literary Corpus)
- Bangla poetry collections
- Classical and modern literature
- Curated literary datasets

## Training Procedure

### Stage 1: Wikipedia Pre-training
- Masked language modeling on Wikipedia
- 3-5 epochs
- Learning rate: 3e-5

### Stage 2: Literary Fine-tuning
- Fine-tuning on literary corpus
- 2-3 epochs
- Learning rate: 2e-5
- Preserves general language understanding while adapting to literary style

## Evaluation

**Perplexity:**
- Wikipedia test set: 20-40
- Literary test set: 15-30 (better on literary texts)

## Usage

### Fill-Mask (Literary Context)

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("{repo_id}")
model = AutoModelForMaskedLM.from_pretrained("{repo_id}")

# Literary text
text = "কবিতার [MASK] সুন্দর"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
```

### Literary Text Analysis

```python
from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("{repo_id}")
model = AutoModel.from_pretrained("{repo_id}")

# Analyze literary text
poem = "আমার সোনার বাংলা, আমি তোমায় ভালোবাসি"
inputs = tokenizer(poem, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state
```

## Comparison with Base Model

| Metric | Base (Wikipedia) | Literary (Fine-tuned) |
|--------|------------------|----------------------|
| Wikipedia Perplexity | 25 | 28 |
| Literary Perplexity | 35 | 22 |
| Style Adaptation | Low | High |

The literary model performs better on poetic and literary texts while maintaining reasonable performance on general text.

## Limitations

- Fine-tuned on specific literary corpus (may not generalize to all literary styles)
- May be less suitable for technical or formal writing
- Limited by the diversity of the literary training data

## Citation

```bibtex
@misc{{bn-wikipedia-literary-lm,
  author = {{KothaGPT}},
  title = {{Bangla Wikipedia + Literary Language Model}},
  year = {{2025}},
  publisher = {{Hugging Face}},
  url = {{https://huggingface.co/{repo_id}}}
}}
```

## License

Apache 2.0

## Acknowledgments

- **Wikimedia Foundation** for Wikipedia data
- **AI4Bharat** for Indic NLP tools
- **Hugging Face** for the Transformers library
- Literary corpus contributors
""",
}


def generate_model_card(
    model_path: Path,
    model_type: str = "base",
    repo_id: str = "your-username/model-name",
) -> str:
    """Generate model card content."""

    # Load metadata
    metadata_path = model_path / "model_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    else:
        metadata = {
            "model_type": "bert",
            "architecture": "BertForMaskedLM",
            "num_parameters": 110000000,
            "model_size_mb": 400,
            "vocab_size": 50000,
        }

    # Select template
    template = MODEL_CARD_TEMPLATES.get(model_type, MODEL_CARD_TEMPLATES["base"])

    # Fill template
    model_card = template.format(
        model_type=metadata.get("model_type", "bert"),
        architecture=metadata.get("architecture", "BertForMaskedLM"),
        num_parameters=metadata.get("num_parameters", 110000000),
        model_size_mb=metadata.get("model_size_mb", 400),
        vocab_size=metadata.get("vocab_size", 50000),
        repo_id=repo_id,
    )

    return model_card


def main():
    parser = argparse.ArgumentParser(description="Generate model card for Hugging Face Hub")
    parser.add_argument("--model", type=str, required=True, help="Path to prepared model directory")
    parser.add_argument(
        "--type",
        type=str,
        choices=["base", "literary"],
        default="base",
        help="Model type (base or literary)",
    )
    parser.add_argument(
        "--repo", type=str, default="your-username/model-name", help="Hugging Face repository ID"
    )
    parser.add_argument("--output", type=str, help="Output file (default: model/README.md)")

    args = parser.parse_args()

    model_path = Path(args.model)

    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        sys.exit(1)

    # Generate model card
    logger.info(f"Generating {args.type} model card")
    model_card = generate_model_card(model_path, args.type, args.repo)

    # Save
    output_path = Path(args.output) if args.output else model_path / "README.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(model_card)

    logger.info(f"✓ Model card saved to: {output_path}")

    print("\n" + "=" * 60)
    print("Model card generated!")
    print("=" * 60)
    print(f"Location: {output_path}")
    print("\nNext steps:")
    print("1. Review and customize the model card")
    print("2. Add specific examples for your use case")
    print("3. Update metrics with actual evaluation results")
    print("4. Upload model:")
    print(f"   python scripts/huggingface/upload_model.py --model {model_path} --repo {args.repo}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
