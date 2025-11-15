#!/usr/bin/env python3
"""
Example usage of Wikipedia Language Model
"""

from bilingual.modules.wikipedia_lm import load_model

def main():
    print("Loading Wikipedia Language Model...")
    model = load_model("models/wikipedia/base")
    
    print("\n" + "="*60)
    print("Example 1: Fill Masked Text")
    print("="*60)
    
    text = "আমি [MASK] খাই"
    print(f"Input: {text}")
    print("\nPredictions:")
    
    results = model.fill_mask(text, top_k=5)
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result['sequence']} (score: {result['score']:.4f})")
    
    print("\n" + "="*60)
    print("Example 2: Semantic Similarity")
    print("="*60)
    
    text1 = "আমি ভাত খাই"
    text2 = "আমি খাবার খাই"
    
    similarity = model.compute_similarity(text1, text2)
    print(f"Text 1: {text1}")
    print(f"Text 2: {text2}")
    print(f"Similarity: {similarity:.4f}")
    
    print("\n" + "="*60)
    print("Example 3: Get Embeddings")
    print("="*60)
    
    text = "আমি বাংলায় কথা বলি"
    embedding = model.get_sentence_embedding(text)
    print(f"Text: {text}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding (first 10 dims): {embedding[:10].tolist()}")
    
    print("\n" + "="*60)
    print("Example 4: Predict Next Word")
    print("="*60)
    
    text = "আমি ভাত"
    print(f"Input: {text}")
    print("\nPredictions:")
    
    predictions = model.predict_next_word(text, top_k=5)
    for i, pred in enumerate(predictions, 1):
        print(f"  {i}. {pred['word']} (score: {pred['score']:.4f})")

if __name__ == '__main__':
    main()
