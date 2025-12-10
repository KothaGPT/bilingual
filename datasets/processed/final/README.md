---
language:
- bn
- en
license: apache-2.0
tags:
- bilingual
- bengali
- bangla
- wikipedia
- education
- parallel-corpus
task_categories:
- text-generation
- translation
- fill-mask
size_categories:
- 1K<n<10K
---

# Bilingual Corpus (Bengali-English)

## Dataset Description

### Dataset Summary
This dataset contains parallel Bengali-English text data for training and evaluating bilingual language models. The corpus includes diverse text sources like Wikipedia articles, educational content, and literary texts.

### Supported Tasks
- **Machine Translation**: Bengali ↔ English translation
- **Text Generation**: Bilingual text generation
- **Cross-lingual Understanding**: Training models to understand both languages

### Languages
- Bengali (bn)
- English (en)

## Dataset Structure

### Data Instances
Each instance contains parallel text in both Bengali and English:

```json
{
  "text": "বাংলাদেশ দক্ষিণ এশিয়ার একটি দেশ।",
  "translation": "Bangladesh is a country in South Asia.",
  "source": "wikipedia",
  "domain": "geography"
}
```

### Data Fields
- `text`: The text content in the source language
- `translation`: The translated text in the target language
- `source`: Source of the text (wikipedia, educational, literary)
- `domain`: Content domain (geography, history, science, etc.)

### Data Splits
| Split | Examples | Size (MB) |
|-------|----------|-----------|
| Train | 10,000   | 12.5      |
| Validation | 1,000 | 1.2       |
| Test  | 1,000    | 1.3       |

## Usage

### Loading the Dataset
```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("KothaGPT/bilingual-corpus")

# Access the splits
train_data = dataset["train"]
val_data = dataset["validation"]
test_data = dataset["test"]

# Example: Print first training example
print(train_data[0])
```

### Training a Translation Model
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("facebook/m2m100_418M")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/m2m100_418M")

# Example translation
inputs = tokenizer("বাংলাদেশ একটি সুন্দর দেশ", return_tensors="pt")
translated_tokens = model.generate(
    **inputs, 
    forced_bos_token_id=tokenizer.get_lang_id("en")
)
print(tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0])
```

## Dataset Creation

### Source Data
- **Wikipedia**: Articles from Bengali and English Wikipedia
- **Educational Content**: Textbooks and learning materials
- **Literary Works**: Translated literary pieces

### Data Collection and Processing
1. **Collection**: Gathered from various open-source bilingual resources
2. **Cleaning**: Removed duplicates, special characters, and malformed text
3. **Alignment**: Paired Bengali and English sentences
4. **Splitting**: Divided into train/validation/test sets (80/10/10)

### Licensing Information
- **License**: Apache 2.0
- **Copyright**: 2025 KothaGPT

### Citation Information
```bibtex
@misc{bilingual-corpus-2025,
  author = {KothaGPT Team},
  title = {Bilingual Bengali-English Corpus},
  year = {2025},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/datasets/KothaGPT/bilingual-corpus}}
}
```

## Additional Information

### Dataset Curators
KothaGPT Team

### Contact
For questions or feedback, please open an issue on our [GitHub repository](https://github.com/KothaGPT/bilingual).

### Updates
- **2025-12-10**: Initial release of the dataset
