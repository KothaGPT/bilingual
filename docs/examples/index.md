# Examples

# 💡 **Examples & Tutorials**

Learn how to use the Bilingual NLP Toolkit with hands-on examples and tutorials.

## Basic Usage

### Language Detection

```python
import bilingual as bb

# Simple language detection
text = "আমি স্কুলে যাই এবং বই পড়তে ভালোবাসি।"
result = bb.detect_language(text)
print(f"Language: {result['language']}")  # bengali

# Batch processing
texts = [
    "Hello world!",
    "আমি বাংলাদেশে থাকি।",
    "I love programming in Python."
]

for text in texts:
    result = bb.detect_language(text)
    print(f"{text[:30]}... -> {result['language']}")
```

### Text Processing Pipeline

```python
# Process mixed-language text
mixed_text = "Hello আমি John বলে ডাকি।"

# Multi-task processing
result = bb.process(mixed_text, tasks=["detect", "segment", "normalize"])
print(f"Language segments: {len(result.get('segments', []))}")

# Language-specific processing
segments = bb.detect_language_segments(mixed_text)
for segment in segments:
    print(f"{segment['text']} -> {segment['language']}")
```

## Advanced Usage

### Model Integration

```python
# Load and use transformer models
import bilingual as bb

# Load a model
bb.load_model("t5-small", "t5")

# Generate text
prompt = "Translate to Bengali: I love reading books"
translation = bb.generate_text("t5-small", prompt)
print(translation)

# Multilingual generation
story_prompt = "Write a short story about friendship"
story = bb.multilingual_generate("t5-small", story_prompt, "bengali")
print(story)
```

### Evaluation

```python
# Comprehensive evaluation
references = [
    "The weather is beautiful today",
    "I love spending time with friends"
]

candidates = [
    "Today the weather is very nice",
    "I enjoy being with my friends"
]

# Translation evaluation
trans_results = bb.evaluate_translation(references, candidates)
print(f"BLEU: {trans_results['bleu']:.4f}")
print(f"METEOR: {trans_results['meteor']:.4f}")

# Generation evaluation
gen_results = bb.evaluate_generation(references, candidates)
print(f"ROUGE-L: {gen_results['rouge_l']:.4f}")
```

### Data Augmentation

```python
# Generate diverse training data
original_text = "I love reading books and learning new things."

# Synonym replacement
synonyms = bb.augment_text(original_text, method="synonym", n=3)
print("Synonyms:", synonyms)

# Noise injection
noisy = bb.augment_text(original_text, method="noise", intensity=0.1)
print("Noisy:", noisy[0])

# Paraphrasing
paraphrases = bb.augment_text(original_text, method="paraphrase", n=2)
print("Paraphrases:", paraphrases)
```

## Production Deployment

### ONNX Conversion

```python
# Convert models for production
onnx_path = bb.convert_to_onnx(
    "my-model",
    "./models/pytorch/",
    "./models/onnx/"
)

# Create optimized session
session = bb.create_onnx_session("my-model")
print(f"Model loaded in ONNX format: {onnx_path}")
```

### FastAPI Server

```python
# Production API server
from fastapi import FastAPI
import bilingual as bb

app = FastAPI(title="Bilingual API", version="1.0.0")

@app.post("/translate")
async def translate(text: str, from_lang: str = "en", to_lang: str = "bn"):
    bb.load_model("t5-small")
    result = bb.translate_text("t5-small", text, from_lang, to_lang)
    return {"translation": result}
```

## Human-in-the-Loop Evaluation

```python
# Content safety evaluation
content = "একটি সুন্দর গল্প যা শিশুদের জন্য উপযুক্ত।"

# Submit for evaluation
eval_id = bb.submit_evaluation(
    content_id="story_001",
    content_text=content,
    evaluator_id="teacher_001",
    overall_rating="very_appropriate",
    safety_flags=[],
    age_appropriateness={"6-8": True, "9-12": True},
    educational_value=5,
    engagement_score=4
)

# Get safety score
safety = bb.calculate_content_safety_score("story_001")
print(f"Safety Score: {safety['safety_score']}")
```

## Configuration Management

```python
# Custom configuration
from bilingual.config import get_settings

settings = get_settings()

# Customize settings
settings.model.default_model = "t5-base"
settings.evaluation.bleu_ngram_order = 4
settings.api.host = "0.0.0.0"
settings.api.port = 8080

# Save configuration
settings.save_to_file(".bilingual_config.json")
```

## Error Handling

```python
# Robust error handling
import bilingual as bb

try:
    # Try to load model
    bb.load_model("t5-small")
    result = bb.generate_text("t5-small", "Hello world")
    print(result)

except Exception as e:
    print(f"Error: {e}")
    # Fallback to basic processing
    result = bb.detect_language("Hello world")
    print(f"Fallback: {result}")
```

---

*These examples demonstrate the versatility and power of the Bilingual NLP Toolkit. For more advanced use cases, check the [API Reference](api/index.md)!*
