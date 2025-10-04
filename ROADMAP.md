# Bilingual Project Roadmap | Bilingual প্রকল্প রোডম্যাপ

## English Version

### High-Level Product Goal

Build **bilingual** — a polished Python package and ecosystem that:

- Provides high-quality Bangla + English support (tokenization, normalization, bilingual pretrained/fine-tuned models, utilities)
- Works equally well for English and Bangla content (creation, classification, translation-assist, reading material for children)
- Is production-ready, easy-to-install (pip), well-documented (both languages), follows open-source best practices
- Includes dataset/model cards and responsible-use guidance

### Roadmap Phases

#### Phase 0 — Project Setup & Governance ✅

**Status**: In Progress

**Deliverables**:
- ✅ Repository skeleton
- ✅ Code of Conduct (Bangla + English)
- ✅ CONTRIBUTING.md (Bangla + English)
- ✅ License (Apache-2.0)
- ⏳ Project board / issue templates / PR templates (bilingual)
- ⏳ High-level roadmap added to repo

#### Phase 1 — Data Strategy & Dataset Creation

**Status**: Pending

**Goal**: Create a bilingual dataset suite focused on child-friendly content, conversational text, and educational documents.

**Deliverables**:
- Curated corpora for Bangla and English (child stories, dialogues, picture descriptions, nursery rhymes)
- Bilingual parallel corpora and code-switched corpora
- Annotation guidelines (BN & EN) and data cards
- Scripts to ingest, normalize, split, and release datasets
- Trained SentencePiece tokenizer with multilingual vocabulary

**Key Tasks**:
- Data source identification and collection
- Unicode normalization and cleaning pipelines
- Tokenizer training (SentencePiece BPE/Unigram)
- PII removal and privacy protection
- Dataset splits (train/validation/test)

#### Phase 2 — Modeling: Selection & Training

**Status**: Pending

**Goal**: Build/assemble models best-suited to bilingual usage.

**Deliverables**:
- Tokenizer distribution (shared SentencePiece model)
- Baseline bilingual language models (small & medium)
- Translation/specialized fine-tunes
- Classification models for safety/readability
- Evaluation suite & benchmarks (BN/EN parity metrics)

**Model Types**:
- Lightweight encoder for classification & NER
- Bilingual LM for generation (story generation, prompts)
- Translation assist (Bangla ↔ English)
- Conversational model for chat/suggestions
- Readability & age-level classifier

#### Phase 3 — Package Engineering & API Design

**Status**: Pending

**Goal**: Make bilingual easy to install and integrate into Python apps.

**Deliverables**:
- `bilingual` PyPI package with comprehensive API
- CLI tool (`bilingual-cli`) for common tasks
- Minimal runtime dependencies with optional extras
- Docker images for deployment

**API Surface**:
```python
from bilingual import bilingual_api as bb

# Load tokenizer & models
tok = bb.load_tokenizer("bilingual-tokenizer.model")
model = bb.load_model("bilingual-small-lm")

# Normalize text
norm_bn = bb.normalize_text("আমি কেমন আছি?", lang="bn")

# Generate
out = bb.generate(prompt, model_name="bilingual-small-lm", max_tokens=120)

# Translate
en = bb.translate("আমি স্কুলে যাচ্ছি।", src="bn", tgt="en")
```

#### Phase 4 — Documentation, Localization & UX

**Status**: Pending

**Goal**: Full bilingual docs, examples, tutorials, and policy pages.

**Deliverables**:
- Docs site (Sphinx/MkDocs) with English and Bangla versions
- Quickstart tutorials
- API reference (auto-generated)
- Security & safety pages (bilingual)
- Code examples in both languages

#### Phase 5 — Testing, QA & Evaluation

**Status**: Pending

**Goal**: Robust test coverage and evaluation metrics checking parity across languages.

**Deliverables**:
- Unit tests (tokenizer, normalization)
- Integration tests (model inference)
- End-to-end CLI tests
- Benchmark tests for BN vs EN performance
- Human evaluation protocol

#### Phase 6 — Production Deployment & Serving

**Status**: Pending

**Goal**: Provide production inference service and packaged models for offline/local use.

**Deliverables**:
- FastAPI inference server (Dockerized)
- Lightweight on-device models (quantized, ONNX)
- Deployment templates (k8s, Helm, Docker Compose)
- Telemetry & monitoring

#### Phase 7 — Publication, Model Cards, Dataset Cards & Legal

**Status**: Pending

**Goal**: Transparently publish models, datasets, and usage guidelines.

**Deliverables**:
- Dataset cards and model cards (EN & BN)
- License & acceptable-use policy
- Ethical statement and child-safety mitigation plan
- Release notes & migration guides

#### Phase 8 — Community, Contributors & Sustainability

**Status**: Pending

**Goal**: Build community adoption and sustainable maintenance.

**Deliverables**:
- Contributor guide and issue templates
- Community-sourced dataset annotation sprints
- Governance structure
- Funding options

### Minimal MVP Deliverables

For rapid initial release:

1. ✅ Repository structure and governance files
2. ⏳ Tokenizer + small bilingual LM adapter + generation API
3. ⏳ Dataset: cleaned small bilingual corpus + parallel test set
4. ⏳ Python package with core API and CLI
5. ⏳ Docs: Quickstart in EN + BN
6. ⏳ Model card & dataset card
7. ⏳ GitHub repo with CI & tests

### Evaluation & Parity Targets

- **Perplexity**: BN vs EN on held-out corpora
- **Translation**: BLEU / chrF / COMET on validation
- **Generation**: Human rating for fluency & child-appropriateness
- **Code-switch**: Language ID accuracy and semantic preservation
- **Readability**: Correlation with human labels
- **Safety**: Child-safety filters and content validation

### Safety, Ethics & Child-Safety

- **Child-safety policy**: Conservative filters for child-targeted content
- **Human-in-the-loop**: Generated stories require review until validated
- **PII removal**: Strict PII detection and redaction
- **Bias & cultural respect**: Cultural reviewers for validation
- **Reporting**: Clear process for harmful output reporting

---

## বাংলা সংস্করণ

### উচ্চ-স্তরের পণ্য লক্ষ্য

**bilingual** তৈরি করুন — একটি পালিশড Python প্যাকেজ এবং ইকোসিস্টেম যা:

- উচ্চমানের বাংলা + ইংরেজি সমর্থন প্রদান করে (টোকেনাইজেশন, নরমালাইজেশন, দ্বিভাষিক প্রিট্রেইনড/ফাইন-টিউনড মডেল, ইউটিলিটি)
- ইংরেজি এবং বাংলা উভয় কন্টেন্টের জন্য সমানভাবে ভাল কাজ করে (সৃষ্টি, শ্রেণীবিভাগ, অনুবাদ-সহায়তা, শিশুদের জন্য পাঠ্য উপাদান)
- প্রোডাকশন-রেডি, সহজে ইনস্টল করা যায় (pip), ভালভাবে ডকুমেন্টেড (উভয় ভাষায়), ওপেন-সোর্স সেরা অনুশীলন অনুসরণ করে
- ডেটাসেট/মডেল কার্ড এবং দায়িত্বশীল-ব্যবহার নির্দেশিকা অন্তর্ভুক্ত করে

### রোডম্যাপ পর্যায়সমূহ

#### পর্যায় 0 — প্রকল্প সেটআপ এবং গভর্নেন্স ✅

**স্ট্যাটাস**: চলমান

**ডেলিভারেবল**:
- ✅ রিপোজিটরি কাঠামো
- ✅ আচরণবিধি (বাংলা + ইংরেজি)
- ✅ CONTRIBUTING.md (বাংলা + ইংরেজি)
- ✅ লাইসেন্স (Apache-2.0)
- ⏳ প্রজেক্ট বোর্ড / ইস্যু টেমপ্লেট / PR টেমপ্লেট (দ্বিভাষিক)
- ⏳ উচ্চ-স্তরের রোডম্যাপ রেপোতে যোগ করা হয়েছে

#### পর্যায় 1 — ডেটা কৌশল এবং ডেটাসেট তৈরি

**স্ট্যাটাস**: মুলতুবি

**লক্ষ্য**: শিশু-বান্ধব কন্টেন্ট, কথোপকথন টেক্সট এবং শিক্ষামূলক নথিতে ফোকাস করে একটি দ্বিভাষিক ডেটাসেট স্যুট তৈরি করা।

[বাকি পর্যায়গুলি উপরের ইংরেজি সংস্করণের অনুরূপ...]

### ন্যূনতম MVP ডেলিভারেবল

দ্রুত প্রাথমিক রিলিজের জন্য:

1. ✅ রিপোজিটরি কাঠামো এবং গভর্নেন্স ফাইল
2. ⏳ টোকেনাইজার + ছোট দ্বিভাষিক LM অ্যাডাপ্টার + জেনারেশন API
3. ⏳ ডেটাসেট: পরিষ্কার ছোট দ্বিভাষিক কর্পাস + সমান্তরাল টেস্ট সেট
4. ⏳ কোর API এবং CLI সহ Python প্যাকেজ
5. ⏳ ডক্স: EN + BN-এ কুইকস্টার্ট
6. ⏳ মডেল কার্ড এবং ডেটাসেট কার্ড
7. ⏳ CI এবং টেস্ট সহ GitHub রেপো
