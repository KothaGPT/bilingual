---
original: /docs/CONTRIBUTING.md
translated: 2025-10-24
---

> **বিঃদ্রঃ** এটি একটি স্বয়ংক্রিয়ভাবে অনুবাদকৃত নথি। মূল ইংরেজি সংস্করণের জন্য [এখানে ক্লিক করুন](/{rel_path}) করুন।

---

# Contributing to Bilingual | Bilingual-এ অবদান রাখা

## English Version

Thank you for your interest in contributing to **bilingual**! This project aims to provide high-quality Bangla and English language support for NLP tasks, with a focus on child-friendly content and bilingual applications.

### How to Contribute

#### Reporting Issues

- Use the issue templates provided
- Include clear descriptions and reproduction steps
- Tag issues appropriately (bug, feature, documentation, etc.)

#### Code Contributions

1. **Fork the repository** and create a new branch for your feature or bugfix
2. **Follow the code style**: We use `black` for Python formatting and `mypy` for type checking
3. **Write tests**: All new features should include unit tests
4. **Update documentation**: Add docstrings and update relevant docs
5. **Submit a pull request**: Use the PR template and reference related issues

#### Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/bilingual.git
cd bilingual

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black bilingual/ tests/

# Type check
mypy bilingual/
```

#### Contribution Areas

- **Data Collection**: Help gather and curate Bangla/English datasets
- **Model Training**: Contribute training scripts and model improvements
- **Documentation**: Translate docs, write tutorials, improve examples
- **Testing**: Add test cases, improve coverage
- **Bug Fixes**: Fix reported issues
- **Features**: Implement new functionality from the roadmap

### Code Review Process

1. All PRs require at least one review from a maintainer
2. CI tests must pass
3. Documentation must be updated
4. Code coverage should not decrease

### Community Guidelines

- Be respectful and inclusive
- Follow our [Code of Conduct](CODE_OF_CONDUCT.md)
- Help others learn and grow
- Focus on constructive feedback

### Questions?

- Open a discussion on GitHub
- Check existing issues and documentation
- Reach out to maintainers

---

## বাংলা সংস্করণ

**bilingual** প্রকল্পে অবদান রাখতে আপনার আগ্রহের জন্য ধন্যবাদ! এই প্রকল্পের লক্ষ্য হল NLP কাজের জন্য উচ্চমানের বাংলা এবং ইংরেজি ভাষা সমর্থন প্রদান করা, শিশু-বান্ধব বিষয়বস্তু এবং দ্বিভাষিক অ্যাপ্লিকেশনের উপর ফোকাস করে।

### কীভাবে অবদান রাখবেন

#### সমস্যা রিপোর্ট করা

- প্রদত্ত ইস্যু টেমপ্লেট ব্যবহার করুন
- স্পষ্ট বর্ণনা এবং পুনরুৎপাদন পদক্ষেপ অন্তর্ভুক্ত করুন
- ইস্যুগুলি যথাযথভাবে ট্যাগ করুন (বাগ, ফিচার, ডকুমেন্টেশন, ইত্যাদি)

#### কোড অবদান

1. **রিপোজিটরি ফর্ক করুন** এবং আপনার ফিচার বা বাগফিক্সের জন্য একটি নতুন ব্রাঞ্চ তৈরি করুন
2. **কোড স্টাইল অনুসরণ করুন**: আমরা Python ফরম্যাটিংয়ের জন্য `black` এবং টাইপ চেকিংয়ের জন্য `mypy` ব্যবহার করি
3. **টেস্ট লিখুন**: সমস্ত নতুন ফিচারে ইউনিট টেস্ট অন্তর্ভুক্ত থাকা উচিত
4. **ডকুমেন্টেশন আপডেট করুন**: ডকস্ট্রিং যোগ করুন এবং প্রাসঙ্গিক ডক্স আপডেট করুন
5. **পুল রিকোয়েস্ট জমা দিন**: PR টেমপ্লেট ব্যবহার করুন এবং সম্পর্কিত ইস্যু রেফারেন্স করুন

#### ডেভেলপমেন্ট সেটআপ

```bash
# আপনার ফর্ক ক্লোন করুন
git clone https://github.com/YOUR_USERNAME/bilingual.git
cd bilingual

# ডেভেলপমেন্ট ডিপেন্ডেন্সি ইনস্টল করুন
pip install -e ".[dev]"

# টেস্ট চালান
pytest tests/

# কোড ফরম্যাট করুন
black bilingual/ tests/

# টাইপ চেক করুন
mypy bilingual/
```

#### অবদানের ক্ষেত্র

- **ডেটা সংগ্রহ**: বাংলা/ইংরেজি ডেটাসেট সংগ্রহ এবং কিউরেট করতে সাহায্য করুন
- **মডেল ট্রেনিং**: ট্রেনিং স্ক্রিপ্ট এবং মডেল উন্নতিতে অবদান রাখুন
- **ডকুমেন্টেশন**: ডক্স অনুবাদ করুন, টিউটোরিয়াল লিখুন, উদাহরণ উন্নত করুন
- **টেস্টিং**: টেস্ট কেস যোগ করুন, কভারেজ উন্নত করুন
- **বাগ ফিক্স**: রিপোর্ট করা সমস্যা সমাধান করুন
- **ফিচার**: রোডম্যাপ থেকে নতুন কার্যকারিতা বাস্তবায়ন করুন

### কোড রিভিউ প্রক্রিয়া

1. সমস্ত PR-এর জন্য একজন মেইনটেইনারের কমপক্ষে একটি রিভিউ প্রয়োজন
2. CI টেস্ট পাস করতে হবে
3. ডকুমেন্টেশন আপডেট করতে হবে
4. কোড কভারেজ হ্রাস পাওয়া উচিত নয়

### সম্প্রদায়ের নির্দেশিকা

- সম্মানজনক এবং অন্তর্ভুক্তিমূলক হন
- আমাদের [আচরণবিধি](CODE_OF_CONDUCT.md) অনুসরণ করুন
- অন্যদের শিখতে এবং বৃদ্ধি পেতে সাহায্য করুন
- গঠনমূলক প্রতিক্রিয়ার উপর ফোকাস করুন

### প্রশ্ন আছে?

- GitHub-এ একটি আলোচনা খুলুন
- বিদ্যমান ইস্যু এবং ডকুমেন্টেশন চেক করুন
- মেইনটেইনারদের সাথে যোগাযোগ করুন
