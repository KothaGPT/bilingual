#!/usr/bin/env python3
"""
Publish the bilingual dataset to Hugging Face Hub.

Usage:
    python scripts/huggingface/publish_dataset.py \
        --dataset-dir datasets/processed/final \
        --repo-id KothaGPT/bilingual-corpus \
        --commit-message "Initial dataset release"
"""

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, HfFolder, create_repo

from datasets import Dataset, DatasetDict, load_dataset


def load_dataset_from_dir(dataset_dir: str) -> DatasetDict:
    """Load dataset from directory with train/val/test splits."""
    data_files = {
        "train": str(Path(dataset_dir) / "train.jsonl"),
        "validation": str(Path(dataset_dir) / "val.jsonl"),
        "test": str(Path(dataset_dir) / "test.jsonl"),
    }

    # Verify files exist
    for split, filepath in data_files.items():
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"{split} file not found at {filepath}")

    # Load dataset
    dataset = load_dataset("json", data_files=data_files)
    return dataset


def update_dataset_card(dataset_dir: str, repo_id: str):
    """Update dataset card with repository information."""
    card_path = Path(dataset_dir) / "DATASET_CARD.md"
    if not card_path.exists():
        print(f"Warning: Dataset card not found at {card_path}")
        return

    with open(card_path, "r+", encoding="utf-8") as f:
        content = f.read()

        # Add dataset card header if not present
        if "---" not in content.split("\n")[0]:
            header = f"""---
language:
- bn
- en
license: apache-2.0
tags:
- bilingual
- bengali
- bangla
- wikipedia
- education
- parallel-corpus
---

"""
            content = header + content
            f.seek(0)
            f.write(content)
            f.truncate()


def publish_dataset(
    dataset_dir: str,
    repo_id: str,
    commit_message: str = "Update dataset",
    private: bool = False,
):
    """Publish dataset to Hugging Face Hub."""
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset_from_dir(dataset_dir)

    # Update dataset card
    print("Updating dataset card...")
    update_dataset_card(dataset_dir, repo_id)

    # Login to Hugging Face Hub
    api = HfApi()
    token = HfFolder.get_token()
    if token is None:
        raise ValueError(
            "Not logged in to Hugging Face Hub. Please run `huggingface-cli login` first."
        )

    # Create repository if it doesn't exist
    try:
        create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    except Exception as e:
        print(f"Warning: Could not create repository: {e}")
        print("Trying to push to existing repository...")

    # Upload dataset
    print("Uploading dataset to Hugging Face Hub...")
    dataset.push_to_hub(repo_id, commit_message=commit_message)

    # Upload dataset card
    print("Uploading dataset card...")
    api.upload_file(
        path_or_fileobj=str(Path(dataset_dir) / "DATASET_CARD.md"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=commit_message,
    )

    print(f"\n🎉 Dataset published successfully!")
    print(f"View at: https://huggingface.co/datasets/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Publish dataset to Hugging Face Hub")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="datasets/processed/final",
        help="Path to dataset directory containing train/val/test splits",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="KothaGPT/bilingual-corpus",
        help="Hugging Face repository ID (e.g., username/dataset-name)",
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default="Update dataset",
        help="Commit message for the upload",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the dataset private",
    )

    args = parser.parse_args()

    # Convert to absolute path
    dataset_dir = os.path.abspath(args.dataset_dir)

    publish_dataset(
        dataset_dir=dataset_dir,
        repo_id=args.repo_id,
        commit_message=args.commit_message,
        private=args.private,
    )


if __name__ == "__main__":
    main()
