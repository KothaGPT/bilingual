import json
from pathlib import Path
from typing import Dict, List, Optional
import requests
from bs4 import BeautifulSoup

# Paths
DATASET_DIR = Path("../datasets/processed/final")
OUTPUT_DIR = Path("../datasets/regional")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Regional Bangla dialects with example phrases
REGIONAL_DIALECTS = {
    "sylheti": {
        "name": "Sylheti",
        "region": "Sylhet, Bangladesh and surrounding areas",
        "examples": [
            ("আমি যাইতাছি", "আমি যাইতেছি"),  # I am going
            ("কি খাইছিলা?", "কি খেয়েছ?"),  # What did you eat?
            ("ভালা আছনে?", "আপনি ভালো আছেন?"),  # How are you?
        ]
    },
    "chittagonian": {
        "name": "Chittagonian",
        "region": "Chittagong, Bangladesh",
        "examples": [
            ("আমি যাইতাছি", "আমি যাচ্ছি"),
            ("তুমি কেমন আছো?", "তুমি কেমন আছো?"),
            ("খাইছস?", "খেয়েছ?"),
        ]
    },
    "dhakaiya": {
        "name": "Dhakaiya",
        "region": "Dhaka, Bangladesh",
        "examples": [
            ("কি খবর?", "কি খবর?"),
            ("কেমন আছেন?", "কেমন আছেন?"),
            ("কোথায় যাও?", "কোথায় যাচ্ছ?"),
        ]
    },
    "rangpuri": {
        "name": "Rangpuri",
        "region": "Rangpur, Bangladesh",
        "examples": [
            ("কি করতাছো?", "কি করছো?"),
            ("আমার খুব ভালো লাগছে", "আমার খুব ভালো লাগছে"),
            ("কি খাবা?", "কি খাবে?"),
        ]
    },
}

def collect_wikipedia_dialect_data() -> List[Dict]:
    """Collect dialect information from Wikipedia."""
    # This is a simplified example - in practice, you'd want to scrape or use an API
    dialect_data = []
    
    for dialect_id, dialect_info in REGIONAL_DIALECTS.items():
        for dialect_phrase, std_bangla in dialect_info["examples"]:
            dialect_data.append({
                "text": dialect_phrase,
                "standard_bangla": std_bangla,
                "dialect": dialect_id,
                "region": dialect_info["region"],
                "source": "synthetic",
                "lang": "bn",
                "type": "dialect"
            })
    
    return dialect_data

def save_dialect_data(dialect_data: List[Dict], output_file: Path):
    """Save dialect data to a JSONL file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in dialect_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Saved {len(dialect_data)} dialect examples to {output_file}")

def main():
    # Collect dialect data
    print("Collecting regional dialect data...")
    dialect_data = collect_wikipedia_dialect_data()
    
    # Save to file
    output_file = OUTPUT_DIR / "bangla_dialects.jsonl"
    save_dialect_data(dialect_data, output_file)
    
    print(f"\nCollected data for {len(REGIONAL_DIALECTS)} Bangla dialects:")
    for dialect_id, info in REGIONAL_DIALECTS.items():
        print(f"- {info['name']} ({dialect_id}): {info['region']}")

if __name__ == "__main__":
    main()
