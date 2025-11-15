"""
Dataset Manager for KothaGPT

This script handles downloading, preprocessing, and managing various Bangla and bilingual datasets.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import requests
from tqdm import tqdm

import datasets

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DatasetManager:
    def __init__(self, config_path: str = None):
        if config_path is None:
            # Use absolute path to the config file
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "data",
                "dataset_config.json",
            )
        """Initialize the dataset manager with configuration."""
        self.config = self._load_config(config_path)
        self.data_dir = Path(os.path.dirname(config_path))
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self._setup_directories()

    def _load_config(self, config_path: str) -> Dict:
        """Load the dataset configuration."""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise

    def _setup_directories(self):
        """Create necessary directories if they don't exist."""
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(exist_ok=True)

        # Create dataset-specific directories
        for dataset_id in self.config.get("datasets", {}):
            (self.raw_dir / self.config["datasets"][dataset_id]["path"]).mkdir(
                parents=True, exist_ok=True
            )

    def get_dataset_info(self, dataset_id: str) -> Dict:
        """Get information about a specific dataset."""
        return self.config.get("datasets", {}).get(dataset_id, {})

    def list_datasets(self) -> List[str]:
        """List all available datasets in the configuration."""
        return list(self.config.get("datasets", {}).keys())

    def download_dataset(self, dataset_id: str, **kwargs):
        """Download a dataset based on its ID."""
        dataset_info = self.get_dataset_info(dataset_id)
        if not dataset_info:
            raise ValueError(f"Unknown dataset: {dataset_id}")

        logger.info(f"Downloading dataset: {dataset_info['name']}")

        # TODO: Implement dataset-specific downloaders
        if dataset_id == "samanantar":
            self._download_samanantar(dataset_info, **kwargs)
        elif dataset_id == "flores200":
            self._download_flores200(dataset_info, **kwargs)
        else:
            logger.warning(f"Automatic download not implemented for {dataset_id}")
            logger.info(f"Please download the dataset manually to: {dataset_info['path']}")

    def _download_file(self, url: str, filepath: Path, chunk_size: int = 8192):
        """Download a file with progress bar."""
        import requests
        from tqdm import tqdm

        # Stream the download to show progress
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Get total size if available
        total_size = int(response.headers.get("content-length", 0))

        # Download with progress bar
        with open(filepath, "wb") as f, tqdm(
            desc=filepath.name,
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=chunk_size):
                size = f.write(data)
                bar.update(size)

    def _download_samanantar(self, dataset_info: Dict, **kwargs):
        """Download the English-Bangla parallel dataset using OPUS-100."""
        import os

        import pandas as pd

        from datasets import load_dataset

        target_dir = self.raw_dir / dataset_info["path"]
        target_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading English-Bangla parallel dataset to {target_dir}")

        try:
            # Load the OPUS-100 dataset which is larger and more reliable
            logger.info("Loading OPUS-100 English-Bangla dataset...")
            opus_dataset = load_dataset("opus100", "bn-en")

            # Save the dataset splits
            for split in ["train", "validation", "test"]:
                if split in opus_dataset:
                    # Convert to pandas DataFrame for easier handling
                    df = pd.DataFrame(opus_dataset[split])

                    # Save in a structured format
                    filepath = target_dir / f"opus100_en_bn_{split}.parquet"
                    df.to_parquet(filepath)
                    logger.info(f"Saved {split} split to {filepath} ({len(df)} rows)")

            logger.info("English-Bangla parallel dataset download complete.")

        except Exception as e:
            logger.error(f"Failed to download OPUS-100 dataset: {e}")
            logger.info("\nManual download instructions:")
            logger.info("1. Install the datasets library if not already installed:")
            logger.info("   pip install datasets pyarrow")
            logger.info("2. Use the Hugging Face datasets library to load the dataset:")
            logger.info("   from datasets import load_dataset")
            logger.info("   dataset = load_dataset('opus100', 'en-bn')")
            logger.info("3. Save the dataset to disk:")
            logger.info("   dataset.save_to_disk('path/to/save/directory')")
            logger.info("   # Or export to pandas DataFrame:")
            logger.info("   df = dataset['train'].to_pandas()")

    def _download_flores200(self, dataset_info: Dict, **kwargs):
        """Download the FLORES-200 dataset."""
        # TODO: Implement actual download from Meta
        pass

    def preprocess_dataset(self, dataset_id: str, **kwargs):
        """Preprocess a dataset."""
        dataset_info = self.get_dataset_info(dataset_id)
        if not dataset_info:
            raise ValueError(f"Unknown dataset: {dataset_id}")

        logger.info(f"Preprocessing dataset: {dataset_info['name']}")

        # TODO: Implement dataset-specific preprocessing
        # This would include:
        # 1. Loading the raw data
        # 2. Applying language-specific normalization
        # 3. Splitting into train/val/test
        # 4. Saving in a standardized format

    def load_processed_dataset(self, dataset_id: str, split: str = "train"):
        """Load a processed dataset split."""
        # TODO: Implement loading from processed directory
        pass


def main():
    """Command-line interface for dataset management."""
    import argparse

    parser = argparse.ArgumentParser(description="Manage datasets for KothaGPT")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # List datasets
    list_parser = subparsers.add_parser("list", help="List available datasets")

    # Download dataset
    download_parser = subparsers.add_parser("download", help="Download a dataset")
    download_parser.add_argument("dataset_id", help="ID of the dataset to download")

    # Preprocess dataset
    preprocess_parser = subparsers.add_parser("preprocess", help="Preprocess a dataset")
    preprocess_parser.add_argument("dataset_id", help="ID of the dataset to preprocess")

    args = parser.parse_args()

    dm = DatasetManager()

    if args.command == "list":
        print("Available datasets:")
        for i, dataset_id in enumerate(dm.list_datasets(), 1):
            info = dm.get_dataset_info(dataset_id)
            print(f"{i}. {info['name']} ({dataset_id})")
            print(f"   {info['description']}")
            print(f"   Use cases: {', '.join(info['use_case'])}")
            print()

    elif args.command == "download":
        dm.download_dataset(args.dataset_id)

    elif args.command == "preprocess":
        dm.preprocess_dataset(args.dataset_id)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
