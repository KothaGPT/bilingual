# Wikipedia Language Model Usage Examples

Quick reference for using the Wikipedia-trained language models.

---

## Installation

```bash
pip install transformers torch
```

---

## Basic Usage

### 1. Load Model

```python
from bilingual.modules.wikipedia_lm import load_model

# Load trained model
model = load_model("models/wikipedia/base")
```

---

## Masked Language Modeling (MLM)

### Fill Masked Text

```python
# Fill single mask
results = model.fill_mask("আমি [MASK] খাই", top_k=5)

for result in results:
    print(f"{result['sequence']} (score: {result['score']:.4f})")

# Output:
# আমি ভাত খাই (score: 0.8532)
# আমি রুটি খাই (score: 0.0821)
# আমি খাবার খাই (score: 0.0432)
```

### Multiple Masks

```python
results = model.fill_mask("আমি [MASK] এবং [MASK] খাই", top_k=3)

for result in results:
    print(result['sequence'])
```

### Predict Next Word

```python
predictions = model.predict_next_word("আমি ভাত", top_k=5)

for pred in predictions:
    print(f"{pred['word']} (score: {pred['score']:.4f})")

# Output:
# খাই (score: 0.8532)
# খেয়েছি (score: 0.0821)
# খাচ্ছি (score: 0.0432)
```

---

## Text Generation (CLM)

### Generate Text

```python
from bilingual.modules.wikipedia_lm import load_model

# Load causal language model
model = load_model("models/wikipedia/gpt2-bn", model_type='clm')

# Generate text
texts = model.generate_text(
    "আমি বাংলায়",
    max_length=100,
    num_return_sequences=3,
    temperature=0.8
)

for i, text in enumerate(texts, 1):
    print(f"\n{i}. {text}")
```

### Controlled Generation

```python
# More deterministic (lower temperature)
texts = model.generate_text(
    "বাংলাদেশের রাজধানী",
    max_length=50,
    temperature=0.3,
    top_k=10
)

# More creative (higher temperature)
texts = model.generate_text(
    "একদিন এক",
    max_length=100,
    temperature=1.2,
    top_p=0.95
)
```

---

## Embeddings

### Get Sentence Embeddings

```python
# Single sentence
embedding = model.get_sentence_embedding("আমি বাংলায় কথা বলি")
print(f"Embedding shape: {embedding.shape}")  # torch.Size([768])

# Multiple sentences
embeddings = model.get_embeddings([
    "আমি বাংলায় কথা বলি",
    "আমি ইংরেজিতে লিখি",
    "আমি দুটি ভাষা জানি"
])
print(f"Embeddings shape: {embeddings.shape}")  # torch.Size([3, max_len, 768])
```

### Different Pooling Strategies

```python
# Mean pooling (default)
emb_mean = model.get_sentence_embedding("আমি বাংলায় কথা বলি", pooling='mean')

# Max pooling
emb_max = model.get_sentence_embedding("আমি বাংলায় কথা বলি", pooling='max')

# CLS token
emb_cls = model.get_sentence_embedding("আমি বাংলায় কথা বলি", pooling='cls')
```

---

## Semantic Similarity

### Compute Similarity

```python
# Similar sentences
similarity = model.compute_similarity(
    "আমি ভাত খাই",
    "আমি খাবার খাই"
)
print(f"Similarity: {similarity:.4f}")  # 0.8532

# Different sentences
similarity = model.compute_similarity(
    "আমি ভাত খাই",
    "সে বই পড়ে"
)
print(f"Similarity: {similarity:.4f}")  # 0.2341
```

### Find Most Similar

```python
query = "আমি বাংলা ভাষা ভালোবাসি"

candidates = [
    "আমি বাংলায় কথা বলি",
    "বাংলা আমার মাতৃভাষা",
    "আমি ইংরেজি শিখছি",
    "সে গান গায়"
]

similarities = [
    (candidate, model.compute_similarity(query, candidate))
    for candidate in candidates
]

# Sort by similarity
similarities.sort(key=lambda x: x[1], reverse=True)

print("Most similar:")
for text, score in similarities:
    print(f"  {text}: {score:.4f}")
```

---

## Cross-lingual Tasks

### Bilingual Similarity

```python
from bilingual.modules.wikipedia_lm import load_model

# Load cross-lingual model
model = load_model("models/wikipedia/xlm-bilingual")

# Compute cross-lingual similarity
similarity = model.compute_similarity(
    "আমি বাংলায় কথা বলি",
    "I speak in Bengali"
)
print(f"Cross-lingual similarity: {similarity:.4f}")
```

### Multilingual Embeddings

```python
# Get embeddings for multiple languages
bn_emb = model.get_sentence_embedding("আমি বাংলায় কথা বলি")
en_emb = model.get_sentence_embedding("I speak in Bengali")

# Embeddings are in same space
print(f"Bangla embedding: {bn_emb.shape}")
print(f"English embedding: {en_emb.shape}")

# Compute similarity
import torch
similarity = torch.nn.functional.cosine_similarity(
    bn_emb.unsqueeze(0),
    en_emb.unsqueeze(0)
).item()
print(f"Similarity: {similarity:.4f}")
```

---

## Batch Processing

### Process Multiple Texts

```python
texts = [
    "আমি বাংলায় কথা বলি",
    "তুমি কী করছ",
    "সে বই পড়ছে",
    "আমরা খেলছি"
]

# Get all embeddings at once
embeddings = model.get_embeddings(texts)
print(f"Batch embeddings shape: {embeddings.shape}")

# Compute pairwise similarities
import torch

similarities = torch.nn.functional.cosine_similarity(
    embeddings.unsqueeze(1),
    embeddings.unsqueeze(0),
    dim=2
)

print("Pairwise similarities:")
print(similarities)
```

---

## Advanced Usage

### Custom Model Configuration

```python
from bilingual.modules.wikipedia_lm import WikipediaLanguageModel

# Initialize with custom settings
model = WikipediaLanguageModel(
    model_path="models/wikipedia/base",
    device="cuda",  # Force GPU
    model_type="mlm"  # Specify model type
)
```

### Low-level Access

```python
# Access tokenizer
tokenizer = model.tokenizer
tokens = tokenizer.tokenize("আমি বাংলায় কথা বলি")
print(f"Tokens: {tokens}")

# Access model directly
raw_model = model.model
print(f"Model: {raw_model}")

# Get raw outputs
inputs = tokenizer("আমি বাংলায় কথা বলি", return_tensors="pt")
outputs = raw_model(**inputs, output_hidden_states=True)
print(f"Hidden states: {len(outputs.hidden_states)}")
```

---

## Integration Examples

### Text Classification

```python
from bilingual.modules.wikipedia_lm import load_model
from sklearn.linear_model import LogisticRegression
import numpy as np

# Load model
model = load_model("models/wikipedia/base")

# Get embeddings for training data
train_texts = ["আমি খুশি", "আমি দুঃখিত", "আমি রাগান্বিত"]
train_labels = ["positive", "negative", "negative"]

train_embeddings = [
    model.get_sentence_embedding(text).cpu().numpy()
    for text in train_texts
]

# Train classifier
clf = LogisticRegression()
clf.fit(train_embeddings, train_labels)

# Predict
test_text = "আমি আনন্দিত"
test_embedding = model.get_sentence_embedding(test_text).cpu().numpy()
prediction = clf.predict([test_embedding])
print(f"Prediction: {prediction[0]}")
```

### Semantic Search

```python
from bilingual.modules.wikipedia_lm import load_model
import numpy as np

# Load model
model = load_model("models/wikipedia/base")

# Index documents
documents = [
    "বাংলাদেশের রাজধানী ঢাকা",
    "ভারতের রাজধানী দিল্লি",
    "পাকিস্তানের রাজধানী ইসলামাবাদ",
    "আমি ভাত খাই",
    "সে বই পড়ে"
]

# Get embeddings
doc_embeddings = [
    model.get_sentence_embedding(doc).cpu().numpy()
    for doc in documents
]

# Search
query = "বাংলাদেশের রাজধানী কী"
query_embedding = model.get_sentence_embedding(query).cpu().numpy()

# Compute similarities
similarities = [
    np.dot(query_embedding, doc_emb) / (
        np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb)
    )
    for doc_emb in doc_embeddings
]

# Rank results
results = sorted(
    zip(documents, similarities),
    key=lambda x: x[1],
    reverse=True
)

print("Search results:")
for doc, score in results[:3]:
    print(f"  {doc}: {score:.4f}")
```

### Text Clustering

```python
from bilingual.modules.wikipedia_lm import load_model
from sklearn.cluster import KMeans
import numpy as np

# Load model
model = load_model("models/wikipedia/base")

# Get embeddings
texts = [
    "আমি ভাত খাই",
    "আমি রুটি খাই",
    "সে বই পড়ে",
    "সে গল্প পড়ে",
    "আমরা খেলি",
    "তারা খেলে"
]

embeddings = np.array([
    model.get_sentence_embedding(text).cpu().numpy()
    for text in texts
])

# Cluster
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(embeddings)

# Print clusters
for i in range(3):
    print(f"\nCluster {i}:")
    for text, cluster in zip(texts, clusters):
        if cluster == i:
            print(f"  - {text}")
```

---

## Performance Tips

### GPU Acceleration

```python
# Use GPU if available
model = load_model("models/wikipedia/base", device="cuda")

# Check device
print(f"Model device: {model.device}")
```

### Batch Processing for Speed

```python
# Process many texts at once
texts = ["text1", "text2", ..., "text1000"]

# Batch size of 32
batch_size = 32
all_embeddings = []

for i in range(0, len(texts), batch_size):
    batch = texts[i:i+batch_size]
    embeddings = model.get_embeddings(batch)
    all_embeddings.append(embeddings)

# Concatenate
import torch
all_embeddings = torch.cat(all_embeddings, dim=0)
```

### Caching Embeddings

```python
import pickle

# Compute embeddings once
embeddings = model.get_embeddings(texts)

# Save to disk
with open('embeddings.pkl', 'wb') as f:
    pickle.dump(embeddings, f)

# Load later
with open('embeddings.pkl', 'rb') as f:
    embeddings = pickle.load(f)
```

---

## Troubleshooting

### Out of Memory

```python
# Use CPU instead of GPU
model = load_model("models/wikipedia/base", device="cpu")

# Process smaller batches
batch_size = 8  # Reduce from 32
```

### Model Not Found

```python
# Check model path
from pathlib import Path
model_path = Path("models/wikipedia/base")
print(f"Model exists: {model_path.exists()}")

# List files
if model_path.exists():
    print(f"Files: {list(model_path.iterdir())}")
```

---

## See Also

- [Wikipedia Training Roadmap](../WIKIPEDIA_TRAINING_ROADMAP.md)
- [API Documentation](../api/index.md)
- [Training Guide](../training.md)
