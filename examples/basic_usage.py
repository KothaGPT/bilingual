#!/usr/bin/env python3
"""
Basic usage examples for the bilingual package.

This script demonstrates common use cases and API patterns.
"""

from bilingual import bilingual_api as bb
from bilingual.normalize import normalize_text, detect_language
from bilingual.data_utils import BilingualDataset


def example_normalization():
    """Example: Text normalization."""
    print("=" * 60)
    print("TEXT NORMALIZATION EXAMPLES")
    print("=" * 60)
    
    # Bangla text normalization
    text_bn = "আমি   স্কুলে যাচ্ছি।"
    normalized_bn = bb.normalize_text(text_bn, lang="bn")
    print(f"Original (BN):   '{text_bn}'")
    print(f"Normalized (BN): '{normalized_bn}'")
    print()
    
    # English text normalization
    text_en = "I am   going to school."
    normalized_en = bb.normalize_text(text_en, lang="en")
    print(f"Original (EN):   '{text_en}'")
    print(f"Normalized (EN): '{normalized_en}'")
    print()
    
    # Auto-detect language
    text_mixed = "আমি school যাই।"
    lang = detect_language(text_mixed)
    normalized_mixed = bb.normalize_text(text_mixed)
    print(f"Original (Mixed): '{text_mixed}'")
    print(f"Detected language: {lang}")
    print(f"Normalized: '{normalized_mixed}'")
    print()


def example_readability():
    """Example: Readability checking."""
    print("=" * 60)
    print("READABILITY CHECKING EXAMPLES")
    print("=" * 60)
    
    texts = [
        ("আমি স্কুলে যাই।", "bn"),
        ("আমি বিশ্ববিদ্যালয়ে পড়াশোনা করি এবং গবেষণা কাজে নিয়োজিত আছি।", "bn"),
        ("I go to school.", "en"),
        ("The implementation of sophisticated algorithms requires comprehensive understanding.", "en"),
    ]
    
    for text, lang in texts:
        result = bb.readability_check(text, lang=lang)
        print(f"Text: {text[:50]}...")
        print(f"  Language: {lang}")
        print(f"  Level: {result['level']}")
        print(f"  Age Range: {result['age_range']}")
        print(f"  Score: {result['score']:.2f}")
        print()


def example_safety():
    """Example: Safety checking."""
    print("=" * 60)
    print("SAFETY CHECKING EXAMPLES")
    print("=" * 60)
    
    texts = [
        "This is a nice story about rabbits.",
        "Once upon a time, there was a brave little girl.",
        "আমি একটি সুন্দর গল্প শুনেছি।",
    ]
    
    for text in texts:
        result = bb.safety_check(text)
        print(f"Text: {text}")
        print(f"  Safe: {result['is_safe']}")
        print(f"  Confidence: {result['confidence']:.2f}")
        print(f"  Recommendation: {result['recommendation']}")
        print()


def example_dataset():
    """Example: Working with datasets."""
    print("=" * 60)
    print("DATASET EXAMPLES")
    print("=" * 60)
    
    # Create a sample dataset
    data = [
        {"text": "আমি স্কুলে যাই।", "lang": "bn"},
        {"text": "I go to school.", "lang": "en"},
        {"text": "আমি বই পড়ি।", "lang": "bn"},
        {"text": "I read books.", "lang": "en"},
        {"text": "আমি খেলতে ভালোবাসি।", "lang": "bn"},
        {"text": "I love to play.", "lang": "en"},
    ]
    
    dataset = BilingualDataset(data=data)
    print(f"Total samples: {len(dataset)}")
    print()
    
    # Filter by language
    bn_dataset = dataset.filter(lambda x: x["lang"] == "bn")
    en_dataset = dataset.filter(lambda x: x["lang"] == "en")
    print(f"Bangla samples: {len(bn_dataset)}")
    print(f"English samples: {len(en_dataset)}")
    print()
    
    # Transform dataset
    normalized_dataset = dataset.map(
        lambda x: {
            **x,
            "normalized": normalize_text(x["text"], lang=x["lang"])
        }
    )
    
    print("Normalized samples:")
    for i, sample in enumerate(normalized_dataset):
        if i >= 3:  # Show first 3
            break
        print(f"  {sample['text']} -> {sample['normalized']}")
    print()


def example_classification():
    """Example: Text classification."""
    print("=" * 60)
    print("CLASSIFICATION EXAMPLES")
    print("=" * 60)
    
    texts = [
        "Once upon a time, there was a brave rabbit.",
        "Breaking news: Major event happened today.",
        "Hello, how are you? I'm fine, thank you.",
    ]
    
    labels = ["story", "news", "dialogue"]
    
    for text in texts:
        result = bb.classify(text, labels=labels)
        print(f"Text: {text}")
        print(f"  Predictions:")
        for label, score in sorted(result.items(), key=lambda x: x[1], reverse=True):
            print(f"    {label}: {score:.3f}")
        print()


def main():
    """Run all examples."""
    print("\n")
    print("*" * 60)
    print("BILINGUAL PACKAGE - BASIC USAGE EXAMPLES")
    print("*" * 60)
    print("\n")
    
    try:
        example_normalization()
        example_readability()
        example_safety()
        example_dataset()
        example_classification()
        
        print("=" * 60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print()
        print("Next steps:")
        print("  - Read the documentation: docs/en/README.md")
        print("  - Train a tokenizer: python scripts/train_tokenizer.py")
        print("  - Prepare your own data: python scripts/prepare_data.py")
        print()
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        print("Make sure you have installed the package:")
        print("  pip install -e .")


if __name__ == "__main__":
    main()
