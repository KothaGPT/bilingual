# Bilingual ডকুমেন্টেশন

**bilingual** প্যাকেজ ডকুমেন্টেশনে স্বাগতম!

## বিষয়বস্তু

- [দ্রুত শুরু](quickstart.md)
- [API রেফারেন্স](api.md)
- [ডেটা গাইড](data.md)
- [মডেল ট্রেনিং](training.md)
- [ডিপ্লয়মেন্ট](deployment.md)
- [অবদান](../../CONTRIBUTING.md)

## সংক্ষিপ্ত বিবরণ

**bilingual** হল বাংলা এবং ইংরেজি প্রাকৃতিক ভাষা প্রক্রিয়াকরণের জন্য একটি প্রোডাকশন-রেডি Python প্যাকেজ। এটি প্রদান করে:

- **টেক্সট নরমালাইজেশন**: উভয় ভাষায় টেক্সট পরিষ্কার এবং মানসম্মত করুন
- **টোকেনাইজেশন**: দক্ষ SentencePiece-ভিত্তিক টোকেনাইজেশন
- **ভাষা মডেল**: জেনারেশনের জন্য প্রিট্রেইনড এবং ফাইন-টিউনড মডেল
- **অনুবাদ**: বাংলা ↔ ইংরেজি অনুবাদ সমর্থন
- **শ্রেণীবিভাগ**: পঠনযোগ্যতা, নিরাপত্তা এবং কন্টেন্ট শ্রেণীবিভাগ
- **ডেটা ইউটিলিটি**: ডেটাসেট ম্যানেজমেন্ট এবং প্রিপ্রসেসিংয়ের জন্য টুল

## দ্রুত লিঙ্ক

- [GitHub রিপোজিটরি](https://github.com/YOUR_ORG/bilingual)
- [PyPI প্যাকেজ](https://pypi.org/project/bilingual/)
- [ইস্যু ট্র্যাকার](https://github.com/YOUR_ORG/bilingual/issues)
- [ইংরেজি ডকুমেন্টেশন](../en/README.md)

## ইনস্টলেশন

```bash
pip install bilingual
```

ডেভেলপমেন্টের জন্য:

```bash
git clone https://github.com/YOUR_ORG/bilingual.git
cd bilingual
pip install -e ".[dev]"
```

## মৌলিক ব্যবহার

```python
from bilingual import bilingual_api as bb

# টেক্সট নরমালাইজ করুন
text = bb.normalize_text("আমি স্কুলে যাই।", lang="bn")

# টোকেনাইজ করুন
tokens = bb.tokenize(text)

# টেক্সট জেনারেট করুন
story = bb.generate("Once upon a time...", max_tokens=100)

# অনুবাদ করুন
translation = bb.translate("আমি বই পড়ি।", src="bn", tgt="en")
```

## সহায়তা

- 📧 ইমেল: bilingual@example.com
- 💬 আলোচনা: [GitHub Discussions](https://github.com/YOUR_ORG/bilingual/discussions)
- 🐛 বাগ রিপোর্ট: [ইস্যু ট্র্যাকার](https://github.com/YOUR_ORG/bilingual/issues)

## লাইসেন্স

এই প্রকল্পটি Apache License 2.0 এর অধীনে লাইসেন্সপ্রাপ্ত। বিস্তারিত জানার জন্য [LICENSE](../../LICENSE) দেখুন।
