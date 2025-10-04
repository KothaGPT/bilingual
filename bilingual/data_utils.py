"""
Dataset utilities for loading, processing, and managing bilingual data.

Handles various dataset formats and provides preprocessing pipelines.
"""

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple


class BilingualDataset:
    """
    Dataset class for bilingual text data.

    Supports loading from various formats (JSONL, TSV, TXT).
    """

    def __init__(
        self,
        data: Optional[List[Dict[str, Any]]] = None,
        file_path: Optional[str] = None,
    ):
        """
        Initialize dataset.

        Args:
            data: List of data samples
            file_path: Path to load data from
        """
        self.data = data or []

        if file_path:
            self.load(file_path)

    def load(self, file_path: str) -> None:
        """
        Load data from file.

        Args:
            file_path: Path to data file (.jsonl, .json, .tsv, .txt)
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if file_path.suffix == ".jsonl":
            self.data = self._load_jsonl(file_path)
        elif file_path.suffix == ".json":
            self.data = self._load_json(file_path)
        elif file_path.suffix == ".tsv":
            self.data = self._load_tsv(file_path)
        elif file_path.suffix == ".txt":
            self.data = self._load_txt(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

    def _load_jsonl(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load JSONL file."""
        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data

    def _load_json(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle both list and dict formats
        if isinstance(data, dict):
            data = [data]

        return data

    def _load_tsv(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load TSV file."""
        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            header = f.readline().strip().split("\t")
            for line in f:
                values = line.strip().split("\t")
                if len(values) == len(header):
                    data.append(dict(zip(header, values)))
        return data

    def _load_txt(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load plain text file (one sample per line)."""
        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append({"text": line})
        return data

    def save(self, file_path: str, format: str = "jsonl") -> None:
        """
        Save dataset to file.

        Args:
            file_path: Output file path
            format: Output format ('jsonl', 'json', 'tsv')
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "jsonl":
            with open(file_path, "w", encoding="utf-8") as f:
                for item in self.data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

        elif format == "json":
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)

        elif format == "tsv":
            if not self.data:
                return

            keys = list(self.data[0].keys())
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("\t".join(keys) + "\n")
                for item in self.data:
                    values = [str(item.get(k, "")) for k in keys]
                    f.write("\t".join(values) + "\n")

        else:
            raise ValueError(f"Unsupported format: {format}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return iter(self.data)

    def shuffle(self, seed: Optional[int] = None) -> None:
        """Shuffle the dataset in place."""
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.data)

    def split(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: Optional[int] = None,
    ) -> Tuple["BilingualDataset", "BilingualDataset", "BilingualDataset"]:
        """
        Split dataset into train/val/test sets.

        Args:
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            test_ratio: Proportion for test set
            seed: Random seed for reproducibility

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

        if seed is not None:
            random.seed(seed)

        # Shuffle data
        data_copy = self.data.copy()
        random.shuffle(data_copy)

        # Calculate split indices
        n = len(data_copy)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        # Create splits
        train_data = BilingualDataset(data=data_copy[:train_end])
        val_data = BilingualDataset(data=data_copy[train_end:val_end])
        test_data = BilingualDataset(data=data_copy[val_end:])

        return train_data, val_data, test_data

    def filter(self, condition) -> "BilingualDataset":
        """
        Filter dataset based on a condition.

        Args:
            condition: Function that takes a sample and returns bool

        Returns:
            New filtered dataset
        """
        filtered_data = [item for item in self.data if condition(item)]
        return BilingualDataset(data=filtered_data)

    def map(self, transform) -> "BilingualDataset":
        """
        Apply a transformation to all samples.

        Args:
            transform: Function that takes a sample and returns transformed sample

        Returns:
            New transformed dataset
        """
        transformed_data = [transform(item) for item in self.data]
        return BilingualDataset(data=transformed_data)


def load_parallel_corpus(
    src_file: str,
    tgt_file: str,
    src_lang: str = "bn",
    tgt_lang: str = "en",
) -> BilingualDataset:
    """
    Load parallel corpus from separate source and target files.

    Args:
        src_file: Path to source language file
        tgt_file: Path to target language file
        src_lang: Source language code
        tgt_lang: Target language code

    Returns:
        BilingualDataset with parallel sentences
    """
    with open(src_file, "r", encoding="utf-8") as f_src, open(
        tgt_file, "r", encoding="utf-8"
    ) as f_tgt:
        src_lines = [line.strip() for line in f_src if line.strip()]
        tgt_lines = [line.strip() for line in f_tgt if line.strip()]

    if len(src_lines) != len(tgt_lines):
        raise ValueError(
            f"Source and target files have different number of lines: "
            f"{len(src_lines)} vs {len(tgt_lines)}"
        )

    data = [
        {
            "src": src,
            "tgt": tgt,
            "src_lang": src_lang,
            "tgt_lang": tgt_lang,
        }
        for src, tgt in zip(src_lines, tgt_lines)
    ]

    return BilingualDataset(data=data)


def combine_corpora(*datasets: BilingualDataset) -> BilingualDataset:
    """
    Combine multiple datasets into one.

    Args:
        *datasets: Variable number of BilingualDataset instances

    Returns:
        Combined dataset
    """
    combined_data = []
    for dataset in datasets:
        combined_data.extend(dataset.data)

    return BilingualDataset(data=combined_data)
