import json
import random
from pathlib import Path
from typing import List, Dict, Any
import random
import string

# Paths
DATASET_DIR = Path("../datasets/processed/final")
OUTPUT_DIR = Path("../datasets/enhanced")

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """Load a JSONL file into a list of dictionaries."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def generate_code_switched_sentence(en_text: str, bn_text: str) -> str:
    """Generate a code-switched sentence by combining English and Bangla."""
    en_words = en_text.strip().split()
    bn_words = bn_text.strip().split()
    
    # Simple code-switching: alternate between languages
    mixed_words = []
    max_len = max(len(en_words), len(bn_words))
    
    for i in range(max_len):
        if i < len(en_words) and i % 2 == 0:
            mixed_words.append(en_words[i])
        if i < len(bn_words) and i % 2 == 1:
            mixed_words.append(bn_words[i])
    
    return ' '.join(mixed_words)

def generate_conversation() -> List[Dict[str, str]]:
    """Generate a simple bilingual conversation."""
    greetings = [
        ("Hi, how are you?", "আপনি কেমন আছেন?"),
        ("Hello! What's up?", "হ্যালো! কি অবস্থা?"),
        ("Good morning!", "সুপ্রভাত!"),
        ("How's it going?", "কেমন যাচ্ছে?")
    ]
    
    responses = [
        ("I'm doing well, thanks!", "আমি ভালো আছি, ধন্যবাদ!"),
        ("Not bad, how about you?", "খারাপ না, আপনি কেমন আছেন?"),
        ("I'm great! Just working on some code.", "আমি খুব ভালো! কিছু কোড লিখছি।"),
        ("Could be better. It's been a long day.", "আরও ভালো হতে পারত। অনেক দীর্ঘ দিন ছিল।")
    ]
    
    conversation = []
    
    # Add greeting
    en_greeting, bn_greeting = random.choice(greetings)
    conversation.append({"role": "user", "content": en_greeting, "lang": "en"})
    conversation.append({"role": "assistant", "content": bn_greeting, "lang": "bn"})
    
    # Add response
    en_response, bn_response = random.choice(responses)
    conversation.append({"role": "user", "content": bn_response, "lang": "bn"})
    conversation.append({"role": "assistant", "content": en_response, "lang": "en"})
    
    return conversation

def enhance_dataset():
    """Enhance the dataset with code-switched and conversational data."""
    print("Loading existing dataset...")
    train_data = load_jsonl(DATASET_DIR / "train.jsonl")
    
    # Filter English and Bangla texts
    en_texts = [item["text"] for item in train_data if item["lang"] == "en"]
    bn_texts = [item["text"] for item in train_data if item["lang"] == "bn"]
    
    enhanced_data = []
    
    # 1. Add code-switched examples
    print("Generating code-switched examples...")
    min_len = min(len(en_texts), len(bn_texts))
    for i in range(min(1000, min_len)):  # Add up to 1000 code-switched examples
        code_switched = generate_code_switched_sentence(en_texts[i], bn_texts[i])
        enhanced_data.append({
            "text": code_switched,
            "lang": "code-mixed",
            "source": "synthetic",
            "type": "code-switched"
        })
    
    # 2. Add conversational data
    print("Generating conversational examples...")
    for _ in range(500):  # Add 500 conversation examples
        conversation = generate_conversation()
        for turn in conversation:
            enhanced_data.append({
                "text": turn["content"],
                "lang": turn["lang"],
                "role": turn["role"],
                "source": "synthetic",
                "type": "conversation"
            })
    
    # Save enhanced dataset
    output_file = OUTPUT_DIR / "enhanced_data.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in enhanced_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Enhanced dataset saved to {output_file}")
    print(f"Added {len(enhanced_data)} new examples")

if __name__ == "__main__":
    enhance_dataset()
