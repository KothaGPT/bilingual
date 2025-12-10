# Readability Classifier

## Task: Age-appropriate readability classification

### Labels
- 6-8
- 9-10
- 11-12
- general

### Training
To train this model, install transformers and run:

```bash
pip install transformers datasets
python scripts/train_classifier.py --task readability --data datasets/processed/
```

### Usage
```python
from bilingual import bilingual_api as bb

# Use the classifier
result = bb.readability_check("Your text here")
print(result)
```
