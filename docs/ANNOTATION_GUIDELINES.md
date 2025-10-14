# Annotation Guidelines | টীকা নির্দেশিকা

**Version**: 1.0.0  
**Last Updated**: October 2025  
**Languages**: English, বাংলা

---

## English Version

### Purpose

These guidelines provide instructions for annotating bilingual (Bangla-English) data for the **bilingual** project. The goal is to create high-quality, child-safe, and culturally appropriate datasets for training language models.

### Target Audience

- Data annotators
- Volunteer contributors
- Research partners
- Community members

### Core Principles

1. **Quality First**: Accuracy over quantity
2. **Child Safety**: All content must be appropriate for children ages 6-12
3. **Cultural Sensitivity**: Respect both Bangla and English cultural contexts
4. **Linguistic Accuracy**: Maintain proper grammar and natural language
5. **Privacy Protection**: Remove all personally identifiable information (PII)

---

## 1. Text Collection Guidelines

### 1.1 Acceptable Sources

✅ **Approved Sources**:
- Public domain books and stories
- Wikipedia articles (with attribution)
- Government educational materials
- Creative Commons licensed content
- Original content created for this project
- Traditional folk tales and rhymes

❌ **Prohibited Sources**:
- Copyrighted commercial content
- Social media posts (privacy concerns)
- News articles behind paywalls
- Content with unclear licensing
- Machine-translated content without review

### 1.2 Content Types

**Priority Content Types**:
1. **Children's Stories**: Age-appropriate narratives (6-12 years)
2. **Educational Content**: Science, math, history explanations
3. **Dialogues**: Natural conversations
4. **Descriptions**: Picture/scene descriptions
5. **Instructions**: How-to guides and recipes
6. **Rhymes and Poems**: Traditional and educational

**Quality Requirements**:
- Minimum length: 50 characters
- Maximum length: 5000 characters per document
- Clear, grammatically correct sentences
- Natural language (not formal/academic unless appropriate)

---

## 2. Language-Specific Guidelines

### 2.1 Bangla Text (বাংলা)

#### Character Set
- Use standard Unicode Bangla characters (U+0980 to U+09FF)
- Avoid deprecated or rarely-used characters
- Use proper vowel marks (কার) and conjuncts

#### Numerals
- Prefer Bangla numerals: ০১২৩৪৫৬৭৮৯
- Arabic numerals acceptable in technical contexts

#### Punctuation
- Use Bangla danda (।) for full stops
- Use standard commas, quotation marks as needed
- Maintain consistency within documents

#### Common Corrections
- Fix mixed scripts (e.g., avoid Bangla + Devanagari mix)
- Normalize zero-width characters
- Remove unnecessary variation selectors

### 2.2 English Text

#### Spelling
- Use standard spelling (prefer American English for consistency)
- Avoid slang unless contextually appropriate
- Maintain consistent capitalization

#### Grammar
- Follow standard English grammar rules
- Use simple, clear sentence structures for children's content
- Avoid complex subordinate clauses in educational material

---

## 3. Parallel Text Annotation

### 3.1 Translation Pairs

When creating Bangla ↔ English parallel data:

**Quality Criteria**:
- ✅ Accurate meaning preservation
- ✅ Natural phrasing in both languages
- ✅ Appropriate register/formality level
- ✅ Cultural adaptation where necessary
- ❌ Word-for-word literal translation
- ❌ Machine translation without review

**Format**:
```json
{
  "en_text": "The rabbit hopped through the garden.",
  "bn_text": "খরগোশটি বাগানের মধ্য দিয়ে লাফিয়ে গেল।",
  "domain": "children_story",
  "age_range": "6-8"
}
```

### 3.2 Code-Switching Data

For mixed language samples:

**Natural Code-Switching Examples**:
- "আমি school-এ যাচ্ছি" (I'm going to school)
- "Let's play cricket খেলি" (Let's play cricket)

**Mark Language Boundaries**:
```json
{
  "text": "আমি school-এ যাচ্ছি",
  "segments": [
    {"text": "আমি", "lang": "bn"},
    {"text": "school", "lang": "en"},
    {"text": "-এ যাচ্ছি", "lang": "bn"}
  ]
}
```

---

## 4. Content Safety & Appropriateness

### 4.1 Child-Safe Content

**Age-Appropriate Guidelines**:

**Ages 6-8 (Early Elementary)**:
- Simple vocabulary and sentence structures
- Positive themes: friendship, family, nature
- Clear moral lessons
- No scary or violent content

**Ages 9-10 (Middle Elementary)**:
- More complex vocabulary
- Educational science/history topics
- Age-appropriate problem-solving scenarios
- Mild adventure themes acceptable

**Ages 11-12 (Late Elementary)**:
- Academic content appropriate for grade level
- More sophisticated narratives
- Real-world topics presented sensitively

### 4.2 Content Exclusions

**Strictly Prohibited**:
- ❌ Violence, weapons, or harm to people/animals
- ❌ Frightening or horror content
- ❌ Inappropriate language or slang
- ❌ Religious or political controversy
- ❌ Commercial advertising
- ❌ Stereotypes or discriminatory content
- ❌ Adult themes or innuendo

**Questionable Content** (requires review):
- Death or loss (must be age-appropriate)
- Conflict resolution (must show positive outcomes)
- Historical events (present factually and sensitively)

### 4.3 Cultural Sensitivity

**Respect Both Cultures**:
- Avoid cultural stereotypes
- Present diverse perspectives
- Use culturally appropriate examples
- Respect religious and cultural holidays
- Show positive representation of both communities

---

## 5. Data Quality Checks

### 5.1 Automatic Quality Filters

The following are checked automatically:

- **Length**: 50-5000 characters
- **Language Detection**: Correct language identification
- **Character Set**: Valid Unicode characters
- **Encoding**: UTF-8 encoding
- **Duplicates**: No exact duplicates

### 5.2 Manual Quality Checks

Annotators should verify:

1. **Linguistic Quality**:
   - ✓ Grammatically correct
   - ✓ Natural phrasing
   - ✓ Proper punctuation
   - ✓ Consistent spelling

2. **Content Quality**:
   - ✓ Factually accurate
   - ✓ Age-appropriate
   - ✓ Culturally sensitive
   - ✓ Educational value

3. **Technical Quality**:
   - ✓ Proper formatting
   - ✓ No PII present
   - ✓ Correct metadata
   - ✓ Valid JSON structure

---

## 6. Privacy & PII Protection

### 6.1 Personal Information

**Must Be Removed**:
- ❌ Full names of real individuals
- ❌ Email addresses
- ❌ Phone numbers
- ❌ Physical addresses
- ❌ National ID numbers
- ❌ Financial information
- ❌ Medical information

**Acceptable**:
- ✅ Common first names in stories (e.g., "Rina", "Tom")
- ✅ Generic locations (e.g., "a village", "the city")
- ✅ Fictional character names

### 6.2 PII Redaction

**Replacement Strategy**:
- Names → [NAME] or generic name
- Locations → [LOCATION] or generic location
- Dates → [DATE] or generic time reference
- Numbers → [NUMBER] or rounded numbers

**Example**:
```
Original: "জন স্মিথ ঢাকায় থাকেন, তার ফোন +880-1234567890"
Redacted: "জন ঢাকায় থাকেন" (keep common first name, remove last name and phone)
```

---

## 7. Annotation Process

### 7.1 Workflow

1. **Select Source**: Choose from approved sources
2. **Extract Text**: Copy or scrape content with proper attribution
3. **Clean Text**: Remove formatting, ads, navigation elements
4. **Normalize**: Apply standard normalization (Unicode, whitespace)
5. **Review Content**: Check for safety and appropriateness
6. **Remove PII**: Redact personal information
7. **Add Metadata**: Tag with language, domain, age range
8. **Quality Check**: Verify against checklist
9. **Submit**: Save to appropriate format

### 7.2 Metadata Requirements

Each text sample must include:

```json
{
  "text": "...",
  "language": "bn|en|mixed",
  "domain": "story|education|dialogue|description|instruction",
  "age_range": "6-8|9-10|11-12|general",
  "source": "source_description",
  "license": "CC0|CC-BY|Public Domain|Original",
  "collected_date": "YYYY-MM-DD",
  "quality_score": 1.0
}
```

### 7.3 Quality Scoring

**Quality Score Scale** (0.0 to 1.0):

- **1.0**: Excellent - Natural, error-free, highly appropriate
- **0.9**: Very Good - Minor imperfections, fully usable
- **0.8**: Good - Some issues but acceptable
- **0.7**: Fair - Multiple issues, needs review
- **<0.7**: Poor - Should be rejected or heavily edited

---

## 8. Common Mistakes to Avoid

### 8.1 Linguistic Errors

❌ **Avoid**:
- Literal word-for-word translations
- Mixed scripts (e.g., বাংলা mixed with देवनागरी)
- Inconsistent spelling or romanization
- Broken Unicode characters
- Unnecessary English words in Bangla text (unless code-switching)

### 8.2 Content Errors

❌ **Avoid**:
- Unverified facts or claims
- Biased or one-sided perspectives
- Age-inappropriate content
- Copyrighted material without permission
- Machine-generated content without review

### 8.3 Technical Errors

❌ **Avoid**:
- Invalid JSON format
- Missing required metadata fields
- Incorrect language tags
- Duplicate entries
- Unredacted PII

---

## 9. Examples

### 9.1 Good Example - Children's Story

```json
{
  "text": "খরগোশ এবং কচ্ছপের দৌড় প্রতিযোগিতা একটি বিখ্যাত গল্প। খরগোশ খুব দ্রুত দৌড়াতে পারত, কিন্তু সে অহংকারী ছিল। কচ্ছপ ধীর গতিতে চলত, কিন্তু সে কখনো হাল ছাড়ত না।",
  "language": "bn",
  "domain": "story",
  "age_range": "6-8",
  "source": "Traditional Bangla Folk Tale (Public Domain)",
  "license": "Public Domain",
  "collected_date": "2025-10-13",
  "quality_score": 1.0
}
```

### 9.2 Good Example - Parallel Text

```json
{
  "en_text": "The sun rises in the east and sets in the west.",
  "bn_text": "সূর্য পূর্ব দিকে উঠে এবং পশ্চিম দিকে অস্ত যায়।",
  "domain": "education",
  "age_range": "9-10",
  "topic": "science",
  "quality_score": 1.0
}
```

### 9.3 Bad Example - Inappropriate Content

❌ **Rejected**:
```json
{
  "text": "The monster scared the children with its sharp teeth.",
  "reason": "Too scary for target age range"
}
```

---

## 10. Tools & Resources

### 10.1 Recommended Tools

- **Text Editors**: VS Code, Sublime Text (with UTF-8 support)
- **Bangla Input**: Avro Keyboard, Google Input Tools
- **Quality Checks**: Use provided validation scripts
- **JSON Validation**: JSONLint.com

### 10.2 Reference Materials

- Bangla Unicode Chart: unicode.org
- Bangla Grammar References: [specify]
- English Grammar: Strunk & White, AP Stylebook
- Child Development Guidelines: Age-appropriate content guides

---

## বাংলা সংস্করণ

### উদ্দেশ্য

এই নির্দেশিকাগুলি **bilingual** প্রকল্পের জন্য দ্বিভাষিক (বাংলা-ইংরেজি) ডেটা টীকা করার নির্দেশনা প্রদান করে। লক্ষ্য হল ভাষা মডেল প্রশিক্ষণের জন্য উচ্চমানের, শিশু-নিরাপদ এবং সাংস্কৃতিকভাবে উপযুক্ত ডেটাসেট তৈরি করা।

### মূল নীতিমালা

1. **গুণমান প্রথম**: পরিমাণের চেয়ে নির্ভুলতা
2. **শিশু নিরাপত্তা**: সমস্ত বিষয়বস্তু ৬-১২ বছর বয়সী শিশুদের জন্য উপযুক্ত হতে হবে
3. **সাংস্কৃতিক সংবেদনশীলতা**: বাংলা এবং ইংরেজি উভয় সাংস্কৃতিক প্রেক্ষাপটকে সম্মান করুন
4. **ভাষাগত নির্ভুলতা**: সঠিক ব্যাকরণ এবং স্বাভাবিক ভাষা বজায় রাখুন
5. **গোপনীয়তা সুরক্ষা**: সমস্ত ব্যক্তিগত শনাক্তযোগ্য তথ্য (PII) সরান

### সংগ্রহযোগ্য উৎস

✅ **অনুমোদিত উৎস**:
- পাবলিক ডোমেইন বই এবং গল্প
- উইকিপিডিয়া নিবন্ধ (অ্যাট্রিবিউশন সহ)
- সরকারি শিক্ষামূলক উপকরণ
- ক্রিয়েটিভ কমন্স লাইসেন্সযুক্ত বিষয়বস্তু
- এই প্রকল্পের জন্য তৈরি মৌলিক সামগ্রী
- ঐতিহ্যবাহী লোককাহিনী এবং ছড়া

### নিষিদ্ধ বিষয়বস্তু

**কঠোরভাবে নিষিদ্ধ**:
- ❌ সহিংসতা, অস্ত্র বা মানুষ/প্রাণীদের ক্ষতি
- ❌ ভীতিকর বা ভয়াবহ বিষয়বস্তু
- ❌ অনুপযুক্ত ভাষা বা অশালীন শব্দ
- ❌ ধর্মীয় বা রাজনৈতিক বিতর্ক
- ❌ বাণিজ্যিক বিজ্ঞাপন
- ❌ স্টেরিওটাইপ বা বৈষম্যমূলক বিষয়বস্তু
- ❌ প্রাপ্তবয়স্কদের থিম

### মেটাডেটা প্রয়োজনীয়তা

প্রতিটি পাঠ্য নমুনায় অন্তর্ভুক্ত থাকতে হবে:

```json
{
  "text": "...",
  "language": "bn|en|mixed",
  "domain": "story|education|dialogue|description|instruction",
  "age_range": "6-8|9-10|11-12|general",
  "source": "উৎসের বিবরণ",
  "license": "CC0|CC-BY|Public Domain|Original",
  "collected_date": "YYYY-MM-DD",
  "quality_score": 1.0
}
```

---

## Contact & Support

- **Questions**: Open an issue on GitHub
- **Contributions**: See CONTRIBUTING.md
- **Email**: info@khulnasoft.com

---

**Document Version**: 1.0.0  
**License**: CC-BY-4.0  
**Maintained by**: KhulnaSoft Ltd
