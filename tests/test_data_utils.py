"""Tests for data utilities."""

import tempfile
from pathlib import Path

import pytest

from bilingual.data_utils import BilingualDataset, combine_corpora, load_parallel_corpus


class TestBilingualDataset:
    def test_empty_dataset(self):
        dataset = BilingualDataset()
        assert len(dataset) == 0

    def test_dataset_with_data(self):
        data = [
            {"text": "আমি স্কুলে যাই।", "lang": "bn"},
            {"text": "I go to school.", "lang": "en"},
        ]
        dataset = BilingualDataset(data=data)
        assert len(dataset) == 2
        assert dataset[0]["text"] == "আমি স্কুলে যাই।"

    def test_dataset_iteration(self):
        data = [{"text": f"sentence {i}"} for i in range(5)]
        dataset = BilingualDataset(data=data)

        count = 0
        for item in dataset:
            assert "text" in item
            count += 1
        assert count == 5

    def test_dataset_shuffle(self):
        data = [{"text": f"sentence {i}", "id": i} for i in range(10)]
        dataset = BilingualDataset(data=data)

        original_order = [item["id"] for item in dataset]
        dataset.shuffle(seed=42)
        shuffled_order = [item["id"] for item in dataset]

        assert original_order != shuffled_order
        assert sorted(original_order) == sorted(shuffled_order)

    def test_dataset_split(self):
        data = [{"text": f"sentence {i}"} for i in range(100)]
        dataset = BilingualDataset(data=data)

        train, val, test = dataset.split(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42)

        assert len(train) == 80
        assert len(val) == 10
        assert len(test) == 10

    def test_dataset_filter(self):
        data = [
            {"text": "short", "length": 5},
            {"text": "medium text", "length": 11},
            {"text": "this is a longer text", "length": 21},
        ]
        dataset = BilingualDataset(data=data)

        filtered = dataset.filter(lambda x: x["length"] > 10)
        assert len(filtered) == 2

    def test_dataset_map(self):
        data = [{"text": "hello"}, {"text": "world"}]
        dataset = BilingualDataset(data=data)

        transformed = dataset.map(lambda x: {"text": x["text"].upper()})
        assert transformed[0]["text"] == "HELLO"
        assert transformed[1]["text"] == "WORLD"

    def test_save_and_load_jsonl(self):
        data = [
            {"text": "আমি স্কুলে যাই।", "lang": "bn"},
            {"text": "I go to school.", "lang": "en"},
        ]
        dataset = BilingualDataset(data=data)

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.jsonl"
            dataset.save(str(file_path), format="jsonl")

            loaded = BilingualDataset(file_path=str(file_path))
            assert len(loaded) == len(dataset)
            assert loaded[0]["text"] == dataset[0]["text"]

    def test_save_and_load_json(self):
        data = [{"text": "test"}]
        dataset = BilingualDataset(data=data)

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.json"
            dataset.save(str(file_path), format="json")

            loaded = BilingualDataset(file_path=str(file_path))
            assert len(loaded) == 1


class TestParallelCorpus:
    def test_load_parallel_corpus(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src_file = Path(tmpdir) / "src.txt"
            tgt_file = Path(tmpdir) / "tgt.txt"

            with open(src_file, "w", encoding="utf-8") as f:
                f.write("আমি স্কুলে যাই।\n")
                f.write("আমি বই পড়ি।\n")

            with open(tgt_file, "w", encoding="utf-8") as f:
                f.write("I go to school.\n")
                f.write("I read books.\n")

            dataset = load_parallel_corpus(
                str(src_file), str(tgt_file), src_lang="bn", tgt_lang="en"
            )

            assert len(dataset) == 2
            assert dataset[0]["src"] == "আমি স্কুলে যাই।"
            assert dataset[0]["tgt"] == "I go to school."
            assert dataset[0]["src_lang"] == "bn"
            assert dataset[0]["tgt_lang"] == "en"

    def test_mismatched_lengths_raises_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src_file = Path(tmpdir) / "src.txt"
            tgt_file = Path(tmpdir) / "tgt.txt"

            with open(src_file, "w", encoding="utf-8") as f:
                f.write("line 1\n")
                f.write("line 2\n")

            with open(tgt_file, "w", encoding="utf-8") as f:
                f.write("line 1\n")

            with pytest.raises(ValueError):
                load_parallel_corpus(str(src_file), str(tgt_file))


class TestCombineCorpora:
    def test_combine_corpora(self):
        data1 = [{"text": "sentence 1"}]
        data2 = [{"text": "sentence 2"}]
        data3 = [{"text": "sentence 3"}]

        dataset1 = BilingualDataset(data=data1)
        dataset2 = BilingualDataset(data=data2)
        dataset3 = BilingualDataset(data=data3)

        combined = combine_corpora(dataset1, dataset2, dataset3)

        assert len(combined) == 3
        assert combined[0]["text"] == "sentence 1"
        assert combined[1]["text"] == "sentence 2"
        assert combined[2]["text"] == "sentence 3"
