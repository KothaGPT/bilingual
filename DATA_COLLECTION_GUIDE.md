# 📊 Data Collection Guide

**Goal**: Collect high-quality bilingual data for training NLP models

---

## 🎯 Data Requirements

### Quantity Targets

| Data Type | Minimum | Target | Stretch Goal |
|-----------|---------|--------|--------------|
| Bangla Text | 100k tokens | 1M tokens | 10M tokens |
| English Text | 100k tokens | 1M tokens | 10M tokens |
| Parallel Pairs | 10k pairs | 100k pairs | 1M pairs |

### Quality Requirements

**All Data Must Be:**
- ✅ Age-appropriate (6-14 years)
- ✅ Grammatically correct
- ✅ Culturally sensitive
- ✅ Free of PII (Personal Identifiable Information)
- ✅ Properly licensed/public domain
- ✅ UTF-8 encoded
- ✅ Clean (no HTML, special characters)

---

## 📚 Bangla Data Sources

### 1. Wikipedia Bangla

**URL**: https://dumps.wikimedia.org/bnwiki/

**How to Download:**
```bash
# Download latest dump
wget https://dumps.wikimedia.org/bnwiki/latest/bnwiki-latest-pages-articles.xml.bz2

# Extract text (requires wikiextractor)
pip install wikiextractor
python -m wikiextractor.WikiExtractor bnwiki-latest-pages-articles.xml.bz2 \
    --output data/raw/wikipedia_bn/ \
    --bytes 100M \
    --compress
```

**Pros**: Large, high-quality, public domain  
**Cons**: May be too formal for children

### 2. Project Gutenberg

**URL**: https://www.gutenberg.org/

**Search for**: Bangla children's books, folk tales

**How to Download:**
```bash
# Example: Download a specific book
wget https://www.gutenberg.org/cache/epub/[ID]/pg[ID].txt \
    -O data/raw/gutenberg_bn/book_[ID].txt
```

### 3. Common Crawl

**URL**: https://commoncrawl.org/

**Filter for**: Bangla educational websites

**Tools**: Use `cc-pyspark` or `cdx-toolkit` to extract Bangla content

### 4. Bangla News (Public Domain)

**Sources**:
- Prothom Alo (check licensing)
- BBC Bangla (check licensing)
- VOA Bangla (public domain)

**Note**: Always verify licensing before use

### 5. Social Media (With Consent)

**Platforms**: Twitter, Facebook (public posts only)

**Tools**: Use official APIs with proper authentication

**Important**: 
- Only collect public posts
- Anonymize user information
- Follow platform ToS
- Get explicit consent if needed

---

## 📖 English Data Sources

### 1. Children's Books (Public Domain)

**Project Gutenberg**:
```bash
# Search for children's literature
# Filter by: Subject = "Children's literature"
```

**Internet Archive**:
- https://archive.org/details/texts
- Filter: Children's books, Public domain

### 2. Educational Resources

**OpenStax**: https://openstax.org/  
**Khan Academy**: https://www.khanacademy.org/ (check licensing)  
**Wikipedia Simple English**: https://simple.wikipedia.org/

### 3. News for Kids

**Newsela**: https://newsela.com/ (check licensing)  
**Time for Kids**: https://www.timeforkids.com/ (check licensing)

### 4. Story Collections

**Storynory**: https://www.storynory.com/ (check licensing)  
**International Children's Digital Library**: http://childrenslibrary.org/

---

## 🔄 Parallel Corpus Sources

### 1. OPUS Corpus

**URL**: http://opus.nlpl.eu/

**Available Datasets**:
- OpenSubtitles (movie subtitles)
- TED Talks
- Tatoeba
- Wikipedia aligned articles

**How to Download:**
```bash
# Install opus-tools
pip install opustools-pkg

# Download Bangla-English parallel corpus
opus_read -d OpenSubtitles -s bn -t en \
    -w data/raw/parallel/opensub_bn_en.txt \
    -wm moses
```

### 2. Tatoeba

**URL**: https://tatoeba.org/

**Download**:
```bash
# Download sentence pairs
wget https://downloads.tatoeba.org/exports/sentences.tar.bz2
wget https://downloads.tatoeba.org/exports/links.tar.bz2

# Extract Bangla-English pairs
# (requires processing script)
```

### 3. Manual Translation

**Crowdsourcing Platforms**:
- Amazon Mechanical Turk
- Appen
- Lionbridge

**Community Translation**:
- Create translation tasks for volunteers
- Use translation memory tools
- Quality control through multiple reviewers

---

## 🛠️ Data Collection Tools

### Web Scraping

```python
# Example scraper (respect robots.txt)
import requests
from bs4 import BeautifulSoup

def scrape_article(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Extract text
    article = soup.find('article')
    text = article.get_text(strip=True)
    
    return text
```

### PDF Extraction

```bash
# Install pdftotext
sudo apt-get install poppler-utils

# Extract text from PDF
pdftotext input.pdf output.txt
```

### OCR for Scanned Documents

```python
# Using Tesseract OCR
import pytesseract
from PIL import Image

# For Bangla
text = pytesseract.image_to_string(
    Image.open('image.png'),
    lang='ben'
)
```

---

## 🧹 Data Cleaning Pipeline

### Step 1: Initial Cleaning

```python
import re

def clean_text(text):
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters (keep Bangla and English)
    text = re.sub(r'[^\u0980-\u09FF\u0020-\u007E\s]', '', text)
    
    return text.strip()
```

### Step 2: Language Detection

```python
from bilingual.normalize import detect_language

def filter_by_language(text, target_lang='bn'):
    lang = detect_language(text)
    return lang == target_lang
```

### Step 3: Quality Filtering

```python
def quality_filter(text, min_length=50, max_length=1000):
    # Length check
    if len(text) < min_length or len(text) > max_length:
        return False
    
    # Check for proper sentences
    if not text.endswith(('.', '।', '?', '!')):
        return False
    
    # Check for minimum word count
    words = text.split()
    if len(words) < 10:
        return False
    
    return True
```

### Step 4: Deduplication

```python
def deduplicate(texts):
    seen = set()
    unique_texts = []
    
    for text in texts:
        # Use hash for efficient comparison
        text_hash = hash(text)
        if text_hash not in seen:
            seen.add(text_hash)
            unique_texts.append(text)
    
    return unique_texts
```

---

## 📋 Data Format

### Monolingual Data (JSONL)

```jsonl
{"text": "আমি স্কুলে যাই।", "lang": "bn", "source": "wikipedia", "id": "bn_001"}
{"text": "I go to school.", "lang": "en", "source": "gutenberg", "id": "en_001"}
```

### Parallel Data (JSONL)

```jsonl
{"src": "আমি বই পড়ি।", "tgt": "I read books.", "src_lang": "bn", "tgt_lang": "en", "source": "tatoeba", "id": "pair_001"}
```

---

## ✅ Data Collection Checklist

### Before Collection
- [ ] Identify data sources
- [ ] Verify licensing/permissions
- [ ] Set up collection tools
- [ ] Create storage structure
- [ ] Define quality criteria

### During Collection
- [ ] Track data sources
- [ ] Monitor data quality
- [ ] Remove duplicates
- [ ] Check for PII
- [ ] Validate encoding

### After Collection
- [ ] Clean and normalize
- [ ] Create metadata
- [ ] Split into train/val/test
- [ ] Document sources
- [ ] Create dataset card

---

## 📊 Tracking Progress

Create a `data_stats.json` file:

```json
{
  "last_updated": "2025-10-04",
  "bangla": {
    "total_tokens": 0,
    "total_sentences": 0,
    "sources": {
      "wikipedia": 0,
      "gutenberg": 0,
      "other": 0
    }
  },
  "english": {
    "total_tokens": 0,
    "total_sentences": 0,
    "sources": {
      "wikipedia": 0,
      "gutenberg": 0,
      "other": 0
    }
  },
  "parallel": {
    "total_pairs": 0,
    "sources": {
      "opus": 0,
      "tatoeba": 0,
      "manual": 0
    }
  }
}
```

---

## ⚖️ Legal & Ethical Considerations

### Copyright
- ✅ Use public domain content
- ✅ Verify CC licenses
- ✅ Get explicit permission when needed
- ❌ Don't scrape copyrighted content

### Privacy
- ✅ Remove PII (names, addresses, phone numbers)
- ✅ Anonymize user data
- ✅ Follow GDPR/privacy laws
- ❌ Don't collect private information

### Cultural Sensitivity
- ✅ Include diverse perspectives
- ✅ Avoid stereotypes
- ✅ Respect cultural norms
- ❌ Don't include offensive content

---

## 🚀 Quick Start

```bash
# 1. Create directories
mkdir -p data/raw/{bangla,english,parallel}

# 2. Run sample data collector
python scripts/collect_data.py --source sample --output data/raw/

# 3. Check what was created
ls -lh data/raw/

# 4. Prepare data
python scripts/prepare_data.py \
    --input data/raw/ \
    --output datasets/processed/

# 5. Check statistics
python -c "
from bilingual.data_utils import BilingualDataset
ds = BilingualDataset(file_path='datasets/processed/train.jsonl')
print(f'Total samples: {len(ds)}')
"
```

---

## 📞 Need Help?

- **Questions**: Open a GitHub Discussion
- **Data Contributions**: Create a PR with your dataset
- **Licensing Issues**: Contact the maintainers

---

*Happy data collecting! 📊*
