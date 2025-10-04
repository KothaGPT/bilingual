# দ্রুত শুরু গাইড

এই গাইড আপনাকে কয়েক মিনিটের মধ্যে **bilingual** প্যাকেজ দিয়ে শুরু করতে সাহায্য করবে।

## ইনস্টলেশন

### PyPI থেকে

```bash
pip install bilingual
```

### সোর্স থেকে

```bash
git clone https://github.com/YOUR_ORG/bilingual.git
cd bilingual
pip install -e .
```

### ঐচ্ছিক ডিপেন্ডেন্সি

PyTorch সমর্থনের জন্য:
```bash
pip install bilingual[torch]
```

TensorFlow সমর্থনের জন্য:
```bash
pip install bilingual[tensorflow]
```

ডেভেলপমেন্টের জন্য:
```bash
pip install bilingual[dev]
```

## মৌলিক ব্যবহার

### টেক্সট নরমালাইজেশন

```python
from bilingual import bilingual_api as bb

# বাংলা টেক্সট নরমালাইজ করুন
text_bn = "আমি   স্কুলে যাচ্ছি।"
normalized = bb.normalize_text(text_bn, lang="bn")
print(normalized)  # "আমি স্কুলে যাচ্ছি."

# ইংরেজি টেক্সট নরমালাইজ করুন
text_en = "I am   going to school."
normalized = bb.normalize_text(text_en, lang="en")
print(normalized)  # "I am going to school."

# স্বয়ংক্রিয়ভাবে ভাষা সনাক্ত করুন
text = "আমি school যাই।"
normalized = bb.normalize_text(text)  # মিশ্র ভাষা স্বয়ংক্রিয়ভাবে সনাক্ত করে
```

### টোকেনাইজেশন

```python
from bilingual import bilingual_api as bb

# টেক্সট টোকেনাইজ করুন (প্রশিক্ষিত টোকেনাইজার প্রয়োজন)
text = "আমি বই পড়ি।"
tokens = bb.tokenize(text)
print(tokens)

# টোকেন ID পান
token_ids = bb.tokenize(text, return_ids=True)
print(token_ids)
```

### টেক্সট জেনারেশন

```python
from bilingual import bilingual_api as bb

# প্রম্পট থেকে টেক্সট জেনারেট করুন
prompt = "Once upon a time, there was a brave rabbit"
generated = bb.generate(
    prompt,
    model_name="bilingual-small-lm",
    max_tokens=100,
    temperature=0.7
)
print(generated)
```

### অনুবাদ

```python
from bilingual import bilingual_api as bb

# বাংলা থেকে ইংরেজিতে অনুবাদ করুন
text_bn = "আমি বই পড়তে ভালোবাসি।"
translation = bb.translate(text_bn, src="bn", tgt="en")
print(translation)  # "I love to read books."

# ইংরেজি থেকে বাংলায় অনুবাদ করুন
text_en = "I go to school every day."
translation = bb.translate(text_en, src="en", tgt="bn")
print(translation)
```

### পঠনযোগ্যতা চেক

```python
from bilingual import bilingual_api as bb

# পঠনযোগ্যতা স্তর চেক করুন
text = "আমি স্কুলে যাই।"
result = bb.readability_check(text, lang="bn")

print(f"স্তর: {result['level']}")           # elementary/intermediate/advanced
print(f"বয়স পরিসীমা: {result['age_range']}")  # যেমন, "6-8"
print(f"স্কোর: {result['score']}")            # সংখ্যাসূচক স্কোর
```

### নিরাপত্তা চেক

```python
from bilingual import bilingual_api as bb

# শিশুদের জন্য কন্টেন্ট নিরাপদ কিনা চেক করুন
text = "This is a nice story about animals."
result = bb.safety_check(text)

print(f"নিরাপদ: {result['is_safe']}")
print(f"আত্মবিশ্বাস: {result['confidence']}")
print(f"সুপারিশ: {result['recommendation']}")
```

## কমান্ড-লাইন ইন্টারফেস

### টেক্সট নরমালাইজ করুন

```bash
bilingual normalize --text "আমি স্কুলে যাই।" --lang bn
```

### টেক্সট জেনারেট করুন

```bash
bilingual generate --prompt "Once upon a time..." --max-tokens 100
```

### অনুবাদ করুন

```bash
bilingual translate --text "আমি বই পড়ি।" --src bn --tgt en
```

### পঠনযোগ্যতা চেক করুন

```bash
bilingual readability --text "আমি স্কুলে যাই।" --lang bn
```

## ডেটাসেটের সাথে কাজ করা

```python
from bilingual.data_utils import BilingualDataset

# ডেটাসেট লোড করুন
dataset = BilingualDataset(file_path="data/train.jsonl")

# নমুনাগুলির মাধ্যমে পুনরাবৃত্তি করুন
for sample in dataset:
    print(sample["text"])

# ডেটাসেট বিভক্ত করুন
train, val, test = dataset.split(
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1
)

# ডেটাসেট ফিল্টার করুন
filtered = dataset.filter(lambda x: x["lang"] == "bn")

# ডেটাসেট সংরক্ষণ করুন
dataset.save("output.jsonl", format="jsonl")
```

## পরবর্তী পদক্ষেপ

- বিস্তারিত ডকুমেন্টেশনের জন্য [API রেফারেন্স](api.md) পড়ুন
- [ডেটা প্রস্তুতি](data.md) সম্পর্কে জানুন
- [মডেল ট্রেনিং](training.md) অন্বেষণ করুন
- [ডিপ্লয়মেন্ট অপশন](deployment.md) দেখুন

## সাহায্য পাওয়া

- 📖 [সম্পূর্ণ ডকুমেন্টেশন](README.md)
- 💬 [GitHub Discussions](https://github.com/YOUR_ORG/bilingual/discussions)
- 🐛 [ইস্যু রিপোর্ট করুন](https://github.com/YOUR_ORG/bilingual/issues)
