#!/usr/bin/env python3
"""
Preprocessing pipeline for bilingual datasets.
"""
import json

# Add the project root to the Python path
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from bilingual.preprocessing.text_processor import TextProcessor


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def process_dataset(
    input_files: List[Path], output_dir: Path, processor: TextProcessor, dataset_name: str
) -> Dict[str, Any]:
    """Process a dataset and return statistics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    all_stats = {}

    # Process each split
    for split in ["train", "validation", "test"]:
        split_files = [f for f in input_files if split in f.name]
        if not split_files:
            continue

        print(f"Processing {split} split...")
        split_stats = []

        for file in split_files:
            output_file = output_dir / f"{dataset_name}_{split}.parquet"
            stats = processor.process_dataset(file, output_file)
            split_stats.append(stats)

        # Aggregate stats
        if split_stats:
            all_stats[split] = {
                "processed_examples": sum(s["processed_examples"] for s in split_stats),
                "skipped_examples": sum(s["skipped_examples"] for s in split_stats),
                "avg_en_tokens": (
                    sum(s["avg_en_tokens"] * s["processed_examples"] for s in split_stats)
                    / sum(s["processed_examples"] for s in split_stats)
                    if any(s["processed_examples"] > 0 for s in split_stats)
                    else 0
                ),
                "avg_bn_tokens": (
                    sum(s["avg_bn_tokens"] * s["processed_examples"] for s in split_stats)
                    / sum(s["processed_examples"] for s in split_stats)
                    if any(s["processed_examples"] > 0 for s in split_stats)
                    else 0
                ),
            }

    return all_stats


def main():
    """Main preprocessing pipeline."""
    # Load configuration
    config = load_config("config/preprocessing.yaml")
    prep_config = config.get("preprocessing", {})

    # Set up directories
    output_dir = Path(prep_config.get("output_dir", "data/processed"))
    vocab_dir = Path(prep_config.get("vocab_dir", "data/vocab"))
    output_dir.mkdir(parents=True, exist_ok=True)
    vocab_dir.mkdir(parents=True, exist_ok=True)

    # Initialize text processor
    processor = TextProcessor(prep_config)

    # Find all dataset files
    data_dir = Path("data/raw")
    dataset_dirs = [d for d in data_dir.iterdir() if d.is_dir()]

    # Process each dataset
    all_stats = {}

    for dataset_dir in dataset_dirs:
        dataset_name = dataset_dir.name
        print(f"\nProcessing dataset: {dataset_name}")

        # Find all parquet files in the dataset directory
        parquet_files = list(dataset_dir.glob("*.parquet"))
        if not parquet_files:
            print(f"No parquet files found in {dataset_dir}")
            continue

        # Create vocabulary from training data if it doesn't exist
        vocab_file = vocab_dir / f"{processor.sp_model_prefix}.model"
        if not vocab_file.exists():
            print("Creating vocabulary...")
            train_files = [f for f in parquet_files if "train" in f.name]
            processor.create_vocab(train_files, vocab_dir)

        # Process the dataset
        stats = process_dataset(parquet_files, output_dir / dataset_name, processor, dataset_name)

        all_stats[dataset_name] = stats

    # Save statistics
    with open(output_dir / "preprocessing_stats.json", "w") as f:
        json.dump(all_stats, f, indent=2)

    print("\nPreprocessing complete!")
    print(f"Processed datasets: {list(all_stats.keys())}")


if __name__ == "__main__":
    main()
