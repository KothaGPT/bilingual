#!/usr/bin/env python3
"""
Generate model cards for all KothaGPT models.

Usage:
    python scripts/huggingface/generate_all_model_cards.py
    python scripts/huggingface/generate_all_model_cards.py --output cards/
"""

import argparse
from pathlib import Path
from typing import Any, Dict

# Model card templates
MODEL_CARDS = {
    "literary-lm": {
        "title": "Literary Language Model (Bangla)",
        "tags": ["bengali", "bangla", "literary", "poetry", "causal-lm"],
        "task": "text-generation",
        "description": "A language model fine-tuned on Bengali literature and poetry for generating literary text.",
        "widget": [{"text": "কবিতার প্রথম লাইন"}, {"text": "একটি গল্পের শুরু"}],
    },
    "poetic-meter-detector": {
        "title": "Poetic Meter Detector (Bangla)",
        "tags": ["bengali", "bangla", "poetry", "meter", "text-classification"],
        "task": "text-classification",
        "description": "Classifies Bengali poetry by meter type (e.g., Panchali, Payar, Tripadi).",
        "widget": [
            {"text": "তোমারে দেখিয়া কে কয় মধুর হাসি"},
            {"text": "আমার সোনার বাংলা আমি তোমায় ভালোবাসি"},
        ],
    },
    "metaphor-simile-detector": {
        "title": "Metaphor and Simile Detector (Bangla-English)",
        "tags": [
            "bengali",
            "bangla",
            "metaphor",
            "simile",
            "figurative-language",
            "text-classification",
        ],
        "task": "text-classification",
        "description": "Detects metaphors and similes in Bangla and English text.",
        "widget": [{"text": "তার হাসি ফুলের মতো সুন্দর"}, {"text": "Time is money"}],
    },
    "style-transfer-gpt": {
        "title": "Style Transfer GPT (Bangla-English)",
        "tags": ["bengali", "bangla", "style-transfer", "text-generation"],
        "task": "text2text-generation",
        "description": "Transfers text style between formal/informal, modern/classical in Bangla and English.",
        "widget": [
            {"text": "আপনি কেমন আছেন? [INFORMAL]"},
            {"text": "Make this formal: Hey, what's up?"},
        ],
    },
    "sentiment-tone-classifier": {
        "title": "Sentiment and Tone Classifier (Bangla-English)",
        "tags": [
            "bengali",
            "bangla",
            "sentiment-analysis",
            "tone-detection",
            "text-classification",
        ],
        "task": "text-classification",
        "description": "Classifies sentiment (positive/negative/neutral) and tone (formal/informal/sarcastic) in text.",
        "widget": [{"text": "এটি অসাধারণ!"}, {"text": "This is terrible."}],
    },
    "cross-lingual-embed": {
        "title": "Cross-lingual Embeddings (Bangla-English)",
        "tags": ["bengali", "bangla", "embeddings", "cross-lingual", "feature-extraction"],
        "task": "feature-extraction",
        "description": "Generates aligned embeddings for Bangla and English text in shared semantic space.",
        "widget": [
            {"text": "বাংলাদেশ একটি সুন্দর দেশ"},
            {"text": "Bangladesh is a beautiful country"},
        ],
    },
    "named-entity-recognizer": {
        "title": "Named Entity Recognizer (Bangla-English)",
        "tags": ["bengali", "bangla", "ner", "named-entity-recognition", "token-classification"],
        "task": "token-classification",
        "description": "Identifies named entities (person, location, organization) in Bangla and English text.",
        "widget": [
            {"text": "রবীন্দ্রনাথ ঠাকুর কলকাতায় জন্মগ্রহণ করেন"},
            {"text": "Barack Obama was born in Hawaii"},
        ],
    },
}


def generate_model_card(model_name: str, info: Dict[str, Any]) -> str:
    """Generate a model card for a specific model."""

    # YAML frontmatter
    frontmatter = f"""---
language:
- bn
- en
license: apache-2.0
tags:
{chr(10).join(f'- {tag}' for tag in info['tags'])}
pipeline_tag: {info['task']}
widget:
{chr(10).join(f'- text: "{w["text"]}"' for w in info['widget'])}
---

# {info['title']}

## Model Description

{info['description']}

**Model Type:** {info['task'].replace('-', ' ').title()}  
**Languages:** Bangla (bn), English (en)  
**License:** Apache 2.0

## Intended Uses

### Primary Use Cases
- Text analysis and understanding
- NLP research and development
- Educational applications
- Content analysis tools

## How to Use

### Installation

```bash
pip install transformers torch
```

### Basic Usage

```python
from transformers import AutoTokenizer, AutoModel

# Load model and tokenizer
model_name = "KothaGPT/{model_name}"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Use the model
text = "আপনার টেক্সট এখানে"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
```

## Training Details

### Training Data
- Wikipedia articles (Bangla and English)
- Domain-specific corpus
- Curated and quality-checked data

### Training Procedure
- Fine-tuned from multilingual base model
- Optimized for Bangla-English bilingual tasks
- Trained on GPU infrastructure

## Evaluation

Performance metrics available in the [GitHub repository](https://github.com/KothaGPT/bilingual).

## Limitations

- Optimized for standard Bangla and English
- May not handle all dialects or code-switching
- Performance varies by domain and text type

## Ethical Considerations

- Model may reflect biases in training data
- Review outputs for appropriateness
- Not suitable for generating harmful content
- Consider cultural context in applications

## Citation

```bibtex
@misc{{kothagpt-{model_name},
  title={{{info['title']}}},
  author={{KothaGPT Team}},
  year={{2024}},
  publisher={{Hugging Face}},
  howpublished={{\\url{{https://huggingface.co/KothaGPT/{model_name}}}}}
}}
```

## Model Card Authors

KothaGPT Team

## Additional Resources

- **GitHub**: https://github.com/KothaGPT/bilingual
- **Documentation**: https://github.com/KothaGPT/bilingual/tree/main/docs
- **Dataset**: https://huggingface.co/datasets/KothaGPT/bilingual-corpus
"""

    return frontmatter


def main():
    parser = argparse.ArgumentParser(description="Generate all model cards")
    parser.add_argument(
        "--output", type=str, default="cards/", help="Output directory for model cards"
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating model cards...")
    print("=" * 60)

    for model_name, info in MODEL_CARDS.items():
        output_file = output_dir / f"{model_name}_card.md"
        card_content = generate_model_card(model_name, info)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(card_content)

        print(f"✓ Generated: {output_file}")

    print("=" * 60)
    print(f"Generated {len(MODEL_CARDS)} model cards in {output_dir}")


if __name__ == "__main__":
    main()
