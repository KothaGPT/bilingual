# উদাহরণসমূহ

# 💡 **উদাহরণসমূহ এবং টিউটোরিয়ালসমূহ**

হ্যান্ডস-অন উদাহরণসমূহ এবং টিউটোরিয়ালসমূহের মাধ্যমে বাইলিঙ্গুয়াল এনএলপি টুলকিট ব্যবহার করতে শিখুন।

## বেসিক ব্যবহার

### ভাষা শনাক্তকরণ

```python
import bilingual as bb

# সিম্পল ভাষা শনাক্তকরণ
text = "আমি স্কুলে যাই এবং বই পড়তে ভালোবাসি।"
result = bb.detect_language(text)
print(f"ভাষা: {result['language']}")  # bengali

# ব্যাচ প্রসেসিং
texts = [
    "Hello world!",
    "আমি বাংলাদেশে থাকি।",
    "I love programming in Python."
]

for text in texts:
    result = bb.detect_language(text)
    print(f"{text[:30]}... -> {result['language']}")
```

### টেক্সট প্রসেসিং পাইপলাইন

```python
# মিশ্র-ভাষা টেক্সট প্রসেস করুন
mixed_text = "Hello আমি John বলে ডাকি।"

# মাল্টি-টাস্ক প্রসেসিং
result = bb.process(mixed_text, tasks=["detect", "segment", "normalize"])
print(f"ভাষা সেগমেন্টসমূহ: {len(result.get('segments', []))}")

# ভাষা-নির্দিষ্ট প্রসেসিং
segments = bb.detect_language_segments(mixed_text)
for segment in segments:
    print(f"{segment['text']} -> {segment['language']}")
```

## উন্নত ব্যবহার

### মডেল ইন্টিগ্রেশন

```python
# লোড করুন এবং ট্রান্সফর্মার মডেলসমূহ ব্যবহার করুন
import bilingual as bb

# একটি মডেল লোড করুন
bb.load_model("t5-small", "t5")

# টেক্সট জেনারেট করুন
prompt = "Translate to Bengali: I love reading books"
translation = bb.generate_text("t5-small", prompt)
print(translation)

# বহু-ভাষা জেনারেশন
story_prompt = "Write a short story about friendship"
story = bb.multilingual_generate("t5-small", story_prompt, "bengali")
print(story)
```

### মূল্যায়ন

```python
# বিস্তৃত মূল্যায়ন
references = [
    "The weather is beautiful today",
    "I love spending time with friends"
]

candidates = [
    "Today the weather is very nice",
    "I enjoy being with my friends"
]

# অনুবাদ মূল্যায়ন
trans_results = bb.evaluate_translation(references, candidates)
print(f"BLEU: {trans_results['bleu']:.4f}")
print(f"METEOR: {trans_results['meteor']:.4f}")

# জেনারেশন মূল্যায়ন
gen_results = bb.evaluate_generation(references, candidates)
print(f"ROUGE-L: {gen_results['rouge_l']:.4f}")
```

### ডেটা অগমেন্টেশন

```python
# বৈচিত্র্যময় প্রশিক্ষণ ডেটা তৈরি করুন
original_text = "I love reading books and learning new things."

# সিনোনিম রিপ্লেসমেন্ট
synonyms = bb.augment_text(original_text, method="synonym", n=3)
print("সিনোনিমসমূহ:", synonyms)

# নয়েজ ইনজেকশন
noisy = bb.augment_text(original_text, method="noise", intensity=0.1)
print("নয়েজি:", noisy[0])

# প্যারাফ্রেজিং
paraphrases = bb.augment_text(original_text, method="paraphrase", n=2)
print("প্যারাফ্রেজসমূহ:", paraphrases)
```

## প্রোডাকশন ডেপ্লয়মেন্ট

### ONNX কনভার্শন

```python
# প্রোডাকশনের জন্য মডেলসমূহ কনভার্ট করুন
onnx_path = bb.convert_to_onnx(
    "my-model",
    "./models/pytorch/",
    "./models/onnx/"
)

# অপ্টিমাইজড সেশন তৈরি করুন
session = bb.create_onnx_session("my-model")
print(f"ONNX ফরম্যাটে মডেল লোড করা হয়েছে: {onnx_path}")
```

### FastAPI সার্ভার

```python
# প্রোডাকশন API সার্ভার
from fastapi import FastAPI
import bilingual as bb

app = FastAPI(title="Bilingual API", version="1.0.0")

@app.post("/translate")
async def translate(text: str, from_lang: str = "en", to_lang: str = "bn"):
    bb.load_model("t5-small")
    result = bb.translate_text("t5-small", text, from_lang, to_lang)
    return {"translation": result}
```

## হিউম্যান-ইন-দ্য-লুপ মূল্যায়ন

```python
# কনটেন্ট সেফটি মূল্যায়ন
content = "একটি সুন্দর গল্প যা শিশুদের জন্য উপযুক্ত।"

# মূল্যায়নের জন্য সাবমিট করুন
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

# সেফটি স্কোর পান
safety = bb.calculate_content_safety_score("story_001")
print(f"সেফটি স্কোর: {safety['safety_score']}")
```

## কনফিগারেশন ম্যানেজমেন্ট

```python
# কাস্টম কনফিগারেশন
from bilingual.config import get_settings

settings = get_settings()

# সেটিংসমূহ কাস্টমাইজ করুন
settings.model.default_model = "t5-base"
settings.evaluation.bleu_ngram_order = 4
settings.api.host = "0.0.0.0"
settings.api.port = 8080

# কনফিগারেশন সেভ করুন
settings.save_to_file(".bilingual_config.json")
```

## এরর হ্যান্ডলিং

```python
# রোবাস্ট এরর হ্যান্ডলিং
import bilingual as bb

try:
    # মডেল লোড করার চেষ্টা করুন
    bb.load_model("t5-small")
    result = bb.generate_text("t5-small", "Hello world")
    print(result)

except Exception as e:
    print(f"এরর: {e}")
    # ফলব্যাক টু বেসিক প্রসেসিং
    result = bb.detect_language("Hello world")
    print(f"ফলব্যাক: {result}")
```

---

*এই উদাহরণসমূহ বাইলিঙ্গুয়াল এনএলপি টুলকিটের বহুমুখিতা এবং শক্তি প্রদর্শন করে। আরও উন্নত ব্যবহারের ক্ষেত্রে, [API রেফারেন্স](api/index.md) দেখুন!* 🚀
