# Quickstart Guide | কুইকস্টার্ট গাইড

## English Version

### Installation

```bash
pip install bilingual
```

### Basic Usage

```python
from bilingual import bilingual_api as bb

# Text Normalization
normalized = bb.normalize_text("আমি   স্কুলে যাচ্ছি।", lang="bn")
print(normalized)  # Output: আমি স্কুলে যাচ্ছি।

# Tokenization
tokens = bb.tokenize("Hello world!")
print(tokens)  # Output: ['▁Hello', '▁world', '!']

# Text Generation
story = bb.generate("Once upon a time", max_tokens=50)
print(story)

# Translation
translated = bb.translate("আমি বই পড়তে ভালোবাসি।", src="bn", tgt="en")
print(translated)  # Output: I love to read books.

# Safety Check
safe = bb.safety_check("This is a nice story about animals.")
print(safe)  # {'is_safe': True, 'confidence': 0.9, ...}

# Readability Assessment
reading_level = bb.readability_check("The quick brown fox jumps over the lazy dog.", lang="en")
print(reading_level)  # {'level': 'intermediate', 'age_range': '9-12', ...}
```

### Command Line Interface

```bash
# Tokenize text
bilingual tokenize --text "Hello world" --lang en

# Normalize text
bilingual normalize --text "আমি   স্কুলে যাই।" --lang bn

# Generate text
bilingual generate --prompt "Once upon a time" --max-tokens 50

# Translate text
bilingual translate --text "আমি স্কুলে যাই।" --src bn --tgt en

# Check readability
bilingual readability --text "The cat sat on the mat." --lang en

# Check safety
bilingual safety --text "This is a nice story." --lang en
```

## বাংলা সংস্করণ

### ইনস্টলেশন

```bash
pip install bilingual
```

### মৌলিক ব্যবহার

```python
from bilingual import bilingual_api as bb

# টেক্সট নরমালাইজেশন
স্বাভাবিক = bb.normalize_text("আমি   স্কুলে যাচ্ছি।", lang="bn")
print(স্বাভাবিক)  # আউটপুট: আমি স্কুলে যাচ্ছি।

# টোকেনাইজেশন
টোকেনসমূহ = bb.tokenize("হ্যালো ওয়ার্ল্ড!")
print(টোকেনসমূহ)  # আউটপুট: ['▁হ্যালো', '▁ওয়ার্ল্ড', '!']

# টেক্সট জেনারেশন
গল্প = bb.generate("একদা একটি", max_tokens=50)
print(গল্প)

# অনুবাদ
অনুবাদিত = bb.translate("আমি বই পড়তে ভালোবাসি।", src="bn", tgt="en")
print(অনুবাদিত)  # আউটপুট: I love to read books.

# নিরাপত্তা যাচাই
নিরাপদ = bb.safety_check("এটি প্রাণীদের নিয়ে একটি সুন্দর গল্প।")
print(নিরাপদ)  # {'is_safe': True, 'confidence': 0.9, ...}

# পাঠযোগ্যতা মূল্যায়ন
পাঠস্তর = bb.readability_check("দ্রুত বাদামী শিয়াল অলস কুকুরের উপর লাফ দেয়।", lang="bn")
print(পাঠস্তর)  # {'level': 'intermediate', 'age_range': '9-12', ...}
```

### কমান্ড লাইন ইন্টারফেস

```bash
# টেক্সট টোকেনাইজ করুন
bilingual tokenize --text "হ্যালো ওয়ার্ল্ড" --lang bn

# টেক্সট নরমালাইজ করুন
bilingual normalize --text "আমি   স্কুলে যাই।" --lang bn

# টেক্সট জেনারেট করুন
bilingual generate --prompt "একদা একটি" --max-tokens 50

# টেক্সট অনুবাদ করুন
bilingual translate --text "আমি স্কুলে যাই।" --src bn --tgt en

# পাঠযোগ্যতা যাচাই করুন
bilingual readability --text "বিড়ালটি মাদুরের উপর বসেছিল।" --lang bn

# নিরাপত্তা যাচাই করুন
bilingual safety --text "এটি একটি সুন্দর গল্প।" --lang bn
```

## Next Steps

- Explore the full API documentation
- Train custom models on your data
- Contribute to the project
- Join the community discussions
