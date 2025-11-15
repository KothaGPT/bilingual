---
language:
- bn
- en
license: apache-2.0
tags:
- bilingual
- bengali
- bangla
- language-model
- causal-lm
- wikipedia
datasets:
- KothaGPT/bilingual-corpus
widget:
- text: "বাংলাদেশের রাজধানী"
- text: "The capital of Bangladesh is"
---

# Bilingual Language Model (Bangla-English)

## Model Description

This is a bilingual causal language model trained on Bangla (Bengali) and English text. The model is designed for general-purpose text generation and understanding in both languages.

**Model Type:** Causal Language Model (GPT-style)  
**Languages:** Bangla (bn), English (en)  
**Training Data:** Wikipedia articles, educational content, literary texts  
**License:** Apache 2.0

## Intended Uses

### Primary Use Cases
- **Text Generation**: Generate coherent text in Bangla or English
- **Text Completion**: Complete partial sentences or paragraphs
- **Language Understanding**: Extract features for downstream tasks
- **Fine-tuning**: Base model for task-specific applications

### Example Applications
- Content generation for educational materials
- Writing assistance tools
- Chatbots and conversational AI
- Text summarization (after fine-tuning)
- Question answering (after fine-tuning)

## How to Use

### Installation

```bash
pip install transformers torch
```

### Basic Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
model_name = "KothaGPT/bilingual-language-model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Generate text in Bangla
prompt = "বাংলাদেশের রাজধানী"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# Generate text in English
prompt = "The capital of Bangladesh is"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Advanced Usage with Pipeline

```python
from transformers import pipeline

# Create text generation pipeline
generator = pipeline("text-generation", model=model_name)

# Generate with parameters
result = generator(
    "বাংলা ভাষা",
    max_length=100,
    num_return_sequences=3,
    temperature=0.8,
    top_p=0.9
)

for seq in result:
    print(seq['generated_text'])
```

## Training Details

### Training Data
- **Wikipedia**: Bangla and English Wikipedia articles
- **Literary Corpus**: Bengali literature and poetry
- **Educational Content**: Textbooks and learning materials
- **Total Tokens**: ~500M tokens (approximate)

### Training Procedure
- **Architecture**: GPT-2 style transformer
- **Tokenizer**: Custom bilingual SentencePiece tokenizer
- **Vocabulary Size**: 50,000 tokens
- **Training Steps**: 100,000 steps
- **Batch Size**: 32
- **Learning Rate**: 5e-5 with warmup
- **Hardware**: GPU training (V100/A100)

### Hyperparameters
```json
{
  "model_type": "gpt2",
  "vocab_size": 50000,
  "n_positions": 1024,
  "n_embd": 768,
  "n_layer": 12,
  "n_head": 12,
  "learning_rate": 5e-5,
  "warmup_steps": 10000,
  "max_steps": 100000
}
```

## Evaluation

### Perplexity
- **Bangla Test Set**: 15.2
- **English Test Set**: 18.5
- **Mixed Test Set**: 16.8

### Downstream Tasks (after fine-tuning)
- Text Classification: 85% accuracy
- Named Entity Recognition: 82% F1
- Question Answering: 78% F1

## Limitations

### Known Limitations
- **Domain Bias**: Primarily trained on Wikipedia and educational content
- **Formal Language**: Better performance on formal text than colloquial speech
- **Code-Switching**: Limited handling of mixed Bangla-English sentences
- **Context Length**: Maximum 1024 tokens
- **Generation Quality**: May produce repetitive or incoherent text for very long sequences

### Language-Specific Issues
- **Bangla**: May struggle with complex literary forms and regional dialects
- **English**: Optimized for general English, may not capture specialized domains
- **Romanized Bangla**: Not trained on Romanized Bengali text

## Ethical Considerations

### Bias and Fairness
- The model may reflect biases present in Wikipedia and training data
- Geographic bias towards Bangladesh and India
- Potential gender and cultural biases in generated text

### Recommended Practices
- Review generated content for appropriateness
- Do not use for generating harmful or misleading content
- Consider fine-tuning on domain-specific data for production use
- Implement content filtering for user-facing applications

### Privacy
- Model does not store training data
- No personal information should be present in outputs
- Use caution when processing sensitive information

## Citation

If you use this model in your research, please cite:

```bibtex
@misc{kothagpt-bilingual-lm,
  title={Bilingual Language Model for Bangla and English},
  author={KothaGPT Team},
  year={2024},
  publisher={Hugging Face},
  howpublished={\url{https://huggingface.co/KothaGPT/bilingual-language-model}}
}
```

## Model Card Authors

KothaGPT Team

## Model Card Contact

For questions or issues, please open an issue on the [GitHub repository](https://github.com/KothaGPT/bilingual).

## Additional Resources

- **GitHub Repository**: https://github.com/KothaGPT/bilingual
- **Documentation**: https://github.com/KothaGPT/bilingual/tree/main/docs
- **Dataset**: https://huggingface.co/datasets/KothaGPT/bilingual-corpus
- **Demo**: https://huggingface.co/spaces/KothaGPT/bilingual-lm-demo
