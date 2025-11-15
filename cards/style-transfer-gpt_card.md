---
language:
- bn
- en
license: apache-2.0
tags:
- bengali
- bangla
- style-transfer
- text-generation
pipeline_tag: text2text-generation
widget:
- text: "আপনি কেমন আছেন? [INFORMAL]"
- text: "Make this formal: Hey, what's up?"
---

# Style Transfer GPT (Bangla-English)

## Model Description

Transfers text style between formal/informal, modern/classical in Bangla and English.

**Model Type:** Text2Text Generation  
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
model_name = "KothaGPT/style-transfer-gpt"
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
@misc{kothagpt-style-transfer-gpt,
  title={Style Transfer GPT (Bangla-English)},
  author={KothaGPT Team},
  year={2024},
  publisher={Hugging Face},
  howpublished={\url{https://huggingface.co/KothaGPT/style-transfer-gpt}}
}
```

## Model Card Authors

KothaGPT Team

## Additional Resources

- **GitHub**: https://github.com/KothaGPT/bilingual
- **Documentation**: https://github.com/KothaGPT/bilingual/tree/main/docs
- **Dataset**: https://huggingface.co/datasets/KothaGPT/bilingual-corpus
