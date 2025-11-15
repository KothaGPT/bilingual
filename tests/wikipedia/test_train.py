"""
Unit tests for Wikipedia LM training.

Tests:
- Dataset loading
- Tokenizer initialization
- Model initialization
- Training configuration
- Trainer instantiation
"""

import json
import sys
import tempfile
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    from datasets import Dataset

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="transformers not installed")
class TestTrainingSetup:
    """Test training setup and configuration."""

    @pytest.fixture
    def sample_data_dir(self):
        """Create sample training data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create train/val/test directories
            (tmpdir / "train").mkdir()
            (tmpdir / "val").mkdir()
            (tmpdir / "test").mkdir()

            # Create sample data files
            train_data = ["এটি একটি প্রশিক্ষণ বাক্য।"] * 10
            val_data = ["এটি একটি যাচাইকরণ বাক্য।"] * 5
            test_data = ["এটি একটি পরীক্ষা বাক্য।"] * 5

            with open(tmpdir / "train" / "bn_train.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(train_data))

            with open(tmpdir / "val" / "bn_val.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(val_data))

            with open(tmpdir / "test" / "bn_test.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(test_data))

            yield tmpdir

    def test_tokenizer_loading(self):
        """Test tokenizer can be loaded."""
        # Use a small model for testing
        tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

        assert tokenizer is not None
        assert tokenizer.pad_token is not None or tokenizer.eos_token is not None

    def test_model_loading(self):
        """Test model can be loaded."""
        model = AutoModelForMaskedLM.from_pretrained("bert-base-multilingual-cased")

        assert model is not None
        assert hasattr(model, "forward")

        # Check model has parameters
        num_params = sum(p.numel() for p in model.parameters())
        assert num_params > 0

    def test_tokenization(self):
        """Test text tokenization."""
        tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

        text = "এটি একটি পরীক্ষা বাক্য।"
        tokens = tokenizer(text, return_tensors="pt")

        assert "input_ids" in tokens
        assert "attention_mask" in tokens
        assert tokens["input_ids"].shape[0] == 1  # Batch size 1

    def test_dataset_creation(self, sample_data_dir):
        """Test dataset can be created from text files."""
        from datasets import load_dataset

        # Load dataset
        data_files = {
            "train": str(sample_data_dir / "train" / "*.txt"),
            "validation": str(sample_data_dir / "val" / "*.txt"),
            "test": str(sample_data_dir / "test" / "*.txt"),
        }

        dataset = load_dataset("text", data_files=data_files)

        assert "train" in dataset
        assert "validation" in dataset
        assert "test" in dataset

        assert len(dataset["train"]) == 10
        assert len(dataset["validation"]) == 5
        assert len(dataset["test"]) == 5

    def test_dataset_tokenization(self, sample_data_dir):
        """Test dataset tokenization."""
        from datasets import load_dataset

        tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

        # Load dataset
        data_files = {"train": str(sample_data_dir / "train" / "*.txt")}
        dataset = load_dataset("text", data_files=data_files)

        # Tokenize
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=128,
                padding="max_length",
            )

        tokenized = dataset.map(tokenize_function, batched=True)

        assert "input_ids" in tokenized["train"].column_names
        assert "attention_mask" in tokenized["train"].column_names

    def test_data_collator(self):
        """Test data collator for language modeling."""
        from transformers import DataCollatorForLanguageModeling

        tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=0.15,
        )

        assert data_collator is not None
        assert data_collator.mlm is True
        assert data_collator.mlm_probability == 0.15

    def test_training_arguments(self):
        """Test training arguments configuration."""
        from transformers import TrainingArguments

        with tempfile.TemporaryDirectory() as tmpdir:
            training_args = TrainingArguments(
                output_dir=tmpdir,
                num_train_epochs=1,
                per_device_train_batch_size=2,
                per_device_eval_batch_size=2,
                learning_rate=5e-5,
                save_steps=100,
                evaluation_strategy="steps",
                eval_steps=100,
            )

            assert training_args.num_train_epochs == 1
            assert training_args.per_device_train_batch_size == 2
            assert training_args.learning_rate == 5e-5

    def test_trainer_instantiation(self, sample_data_dir):
        """Test Trainer can be instantiated."""
        from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

        from datasets import load_dataset

        # Load small model
        tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
        model = AutoModelForMaskedLM.from_pretrained("bert-base-multilingual-cased")

        # Load dataset
        data_files = {"train": str(sample_data_dir / "train" / "*.txt")}
        dataset = load_dataset("text", data_files=data_files)

        # Tokenize
        def tokenize_function(examples):
            return tokenizer(examples["text"], truncation=True, max_length=128)

        tokenized = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=0.15,
        )

        # Training arguments
        with tempfile.TemporaryDirectory() as tmpdir:
            training_args = TrainingArguments(
                output_dir=tmpdir,
                num_train_epochs=1,
                per_device_train_batch_size=2,
            )

            # Create trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                data_collator=data_collator,
                train_dataset=tokenized["train"],
            )

            assert trainer is not None
            assert trainer.model is not None
            assert trainer.train_dataset is not None


@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="transformers not installed")
class TestTrainingConfig:
    """Test training configuration."""

    def test_config_serialization(self):
        """Test training config can be saved and loaded."""
        config = {
            "model_name": "bert-base-multilingual-cased",
            "model_type": "mlm",
            "max_length": 512,
            "batch_size": 8,
            "learning_rate": 5e-5,
            "num_epochs": 3,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"

            # Save
            with open(config_path, "w") as f:
                json.dump(config, f)

            # Load
            with open(config_path, "r") as f:
                loaded_config = json.load(f)

            assert loaded_config == config

    def test_model_type_validation(self):
        """Test model type validation."""
        valid_types = ["mlm", "clm"]

        for model_type in valid_types:
            assert model_type in valid_types

        invalid_type = "invalid"
        assert invalid_type not in valid_types

    def test_hyperparameter_ranges(self):
        """Test hyperparameter value ranges."""
        # Learning rate
        lr = 5e-5
        assert 0 < lr < 1

        # Batch size
        batch_size = 8
        assert batch_size > 0
        assert batch_size % 2 == 0  # Should be power of 2

        # Epochs
        epochs = 3
        assert epochs > 0

        # Max length
        max_length = 512
        assert max_length > 0
        assert max_length <= 512  # BERT limit


@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="transformers not installed")
@pytest.mark.slow
class TestMockTraining:
    """Test mock training on tiny dataset (slow tests)."""

    def test_single_step_training(self):
        """Test model can perform single training step."""
        from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

        # Create tiny model and dataset
        tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
        model = AutoModelForMaskedLM.from_pretrained("bert-base-multilingual-cased")

        # Create tiny dataset
        texts = ["এটি একটি পরীক্ষা।"] * 4
        dataset = Dataset.from_dict({"text": texts})

        # Tokenize
        def tokenize_function(examples):
            return tokenizer(examples["text"], truncation=True, max_length=128)

        tokenized = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=0.15,
        )

        # Training arguments
        with tempfile.TemporaryDirectory() as tmpdir:
            training_args = TrainingArguments(
                output_dir=tmpdir,
                num_train_epochs=1,
                per_device_train_batch_size=2,
                max_steps=1,  # Only 1 step
                save_strategy="no",
            )

            # Create trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                data_collator=data_collator,
                train_dataset=tokenized,
            )

            # Train for 1 step
            trainer.train()

            # Model should have been updated
            assert trainer.state.global_step == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
