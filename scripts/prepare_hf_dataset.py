#!/usr/bin/env python3
"""
Prepare Wikipedia dataset for Hugging Face Transformers training.

Usage:
    python scripts/prepare_hf_dataset.py --input datasets/wikipedia/processed --output datasets/wikipedia/hf_dataset
    python scripts/prepare_hf_dataset.py --input datasets/wikipedia/processed --push-to-hub --repo KothaGPT/bn-wikipedia
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

try:
    from datasets import Dataset, DatasetDict, load_dataset

    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    logging.warning("datasets library not available. Install: pip install datasets")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_text_file(file_path: Path) -> List[str]:
    """Load sentences from text file."""
    logger.info(f"Loading {file_path}")

    sentences = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                sentences.append(line)

    logger.info(f"Loaded {len(sentences):,} sentences")
    return sentences


def create_hf_dataset(
    data_dir: Path, language: str = "bn", max_samples: Optional[int] = None
) -> DatasetDict:
    """
    Create Hugging Face DatasetDict from processed Wikipedia data.

    Args:
        data_dir: Directory containing train/val/test splits
        language: Language code
        max_samples: Maximum samples per split (for testing)

    Returns:
        DatasetDict with train/validation/test splits
    """
    if not DATASETS_AVAILABLE:
        raise ImportError("datasets library required. Install: pip install datasets")

    logger.info(f"Creating HF dataset from {data_dir}")

    datasets = {}

    for split_name, hf_split in [("train", "train"), ("val", "validation"), ("test", "test")]:
        split_dir = data_dir / split_name

        if not split_dir.exists():
            logger.warning(f"Split '{split_name}' not found, skipping")
            continue

        # Load all sentences from split
        all_sentences = []
        for txt_file in split_dir.glob(f"{language}_*.txt"):
            sentences = load_text_file(txt_file)
            all_sentences.extend(sentences)

        # Limit samples if specified
        if max_samples and len(all_sentences) > max_samples:
            logger.info(f"Limiting {split_name} to {max_samples} samples")
            all_sentences = all_sentences[:max_samples]

        # Create dataset
        logger.info(f"Creating {hf_split} dataset with {len(all_sentences):,} samples")
        datasets[hf_split] = Dataset.from_dict(
            {"text": all_sentences, "language": [language] * len(all_sentences)}
        )

    dataset_dict = DatasetDict(datasets)

    logger.info(f"Created DatasetDict: {dataset_dict}")
    return dataset_dict


def save_dataset_info(dataset_dict: DatasetDict, output_dir: Path):
    """Save dataset information and statistics."""
    info = {
        "splits": {},
        "total_samples": 0,
        "features": list(dataset_dict["train"].features.keys()) if "train" in dataset_dict else [],
    }

    for split_name, dataset in dataset_dict.items():
        num_samples = len(dataset)
        info["splits"][split_name] = {
            "num_samples": num_samples,
            "features": list(dataset.features.keys()),
        }
        info["total_samples"] += num_samples

    info_path = output_dir / "dataset_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    logger.info(f"Saved dataset info to {info_path}")
    return info


def print_dataset_summary(dataset_dict: DatasetDict):
    """Print dataset summary."""
    print("\n" + "=" * 60)
    print("Hugging Face Dataset Summary")
    print("=" * 60)

    total_samples = sum(len(ds) for ds in dataset_dict.values())
    print(f"\nTotal Samples: {total_samples:,}")

    for split_name, dataset in dataset_dict.items():
        print(f"\n{split_name.upper()}:")
        print(f"  Samples: {len(dataset):,}")
        print(f"  Features: {list(dataset.features.keys())}")

        # Show sample
        if len(dataset) > 0:
            sample = dataset[0]
            text_preview = (
                sample["text"][:100] + "..." if len(sample["text"]) > 100 else sample["text"]
            )
            print(f"  Sample: {text_preview}")

    print("\n" + "=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Wikipedia dataset for Hugging Face training"
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Input directory with processed Wikipedia data"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="datasets/wikipedia/hf_dataset",
        help="Output directory for HF dataset",
    )
    parser.add_argument("--lang", type=str, default="bn", help="Language code")
    parser.add_argument("--max-samples", type=int, help="Maximum samples per split (for testing)")
    parser.add_argument(
        "--push-to-hub", action="store_true", help="Push dataset to Hugging Face Hub"
    )
    parser.add_argument(
        "--repo", type=str, help="Hub repository name (e.g., username/dataset-name)"
    )
    parser.add_argument("--private", action="store_true", help="Make Hub repository private")

    args = parser.parse_args()

    if not DATASETS_AVAILABLE:
        logger.error("datasets library not installed. Run: pip install datasets")
        return

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return

    try:
        # Create HF dataset
        dataset_dict = create_hf_dataset(input_dir, args.lang, args.max_samples)

        # Print summary
        print_dataset_summary(dataset_dict)

        # Save to disk
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving dataset to {output_dir}")
        dataset_dict.save_to_disk(str(output_dir))

        # Save dataset info
        save_dataset_info(dataset_dict, output_dir)

        logger.info(f"Dataset saved successfully to {output_dir}")

        # Push to Hub if requested
        if args.push_to_hub:
            if not args.repo:
                logger.error("--repo required when using --push-to-hub")
                return

            logger.info(f"Pushing dataset to Hub: {args.repo}")
            dataset_dict.push_to_hub(args.repo, private=args.private)
            logger.info(f"Dataset pushed to https://huggingface.co/datasets/{args.repo}")

        logger.info("Dataset preparation complete!")

    except Exception as e:
        logger.error(f"Dataset preparation failed: {e}", exc_info=True)
        exit(1)


if __name__ == "__main__":
    main()
