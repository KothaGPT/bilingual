#!/usr/bin/env python3
"""
Train language model on Wikipedia dataset.

Usage:
    python scripts/train_wiki_lm.py --data datasets/wikipedia/processed --model ai4bharat/indic-bert --output models/wikipedia/base
    python scripts/train_wiki_lm.py --data datasets/wikipedia/processed --model xlm-roberta-base --bilingual --output models/wikipedia/bilingual
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class WikipediaLMTrainer:
    """Train language model on Wikipedia data."""

    def __init__(
        self,
        model_name: str,
        data_dir: Path,
        output_dir: Path,
        model_type: str = "mlm",  # 'mlm' or 'clm'
        max_length: int = 512,
        batch_size: int = 8,
        learning_rate: float = 5e-5,
        num_epochs: int = 3,
        gradient_accumulation_steps: int = 4,
        fp16: bool = True,
        save_steps: int = 1000,
    ):
        self.model_name = model_name
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.model_type = model_type
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.fp16 = fp16
        self.save_steps = save_steps

        self.tokenizer = None
        self.model = None
        self.dataset = None

    def load_tokenizer(self) -> None:
        """Load tokenizer."""
        logger.info(f"Loading tokenizer: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Add special tokens if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def load_model(self) -> None:
        """Load model."""
        logger.info(f"Loading model: {self.model_name}")

        if self.model_type == "mlm":
            self.model = AutoModelForMaskedLM.from_pretrained(self.model_name)
        elif self.model_type == "clm":
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        else:
            raise ValueError(f"Invalid model_type: {self.model_type}")

        logger.info(f"Model loaded: {self.model.__class__.__name__}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def load_dataset(self) -> None:
        """Load and prepare dataset."""
        logger.info(f"Loading dataset from {self.data_dir}")

        # Load text files
        data_files = {
            "train": str(self.data_dir / "train" / "*.txt"),
            "validation": str(self.data_dir / "val" / "*.txt"),
            "test": str(self.data_dir / "test" / "*.txt"),
        }

        self.dataset = load_dataset("text", data_files=data_files)

        logger.info(f"Dataset loaded:")
        logger.info(f"  Train: {len(self.dataset['train'])} examples")
        logger.info(f"  Validation: {len(self.dataset['validation'])} examples")
        logger.info(f"  Test: {len(self.dataset['test'])} examples")

    def tokenize_function(self, examples):
        """Tokenize examples."""
        return self.tokenizer(
            examples["text"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_special_tokens_mask=True,
        )

    def prepare_dataset(self) -> None:
        """Tokenize and prepare dataset for training."""
        logger.info("Tokenizing dataset...")

        self.dataset = self.dataset.map(
            self.tokenize_function,
            batched=True,
            num_proc=4,
            remove_columns=["text"],
        )

        logger.info("Dataset tokenization complete")

    def train(self) -> None:
        """Train the model."""
        logger.info("Starting training...")

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=(self.model_type == "mlm"),
            mlm_probability=0.15 if self.model_type == "mlm" else None,
        )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            overwrite_output_dir=True,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            weight_decay=0.01,
            warmup_steps=500,
            logging_dir=str(self.output_dir / "logs"),
            logging_steps=100,
            save_steps=self.save_steps,
            save_total_limit=3,
            evaluation_strategy="steps",
            eval_steps=self.save_steps,
            fp16=self.fp16 and torch.cuda.is_available(),
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=["tensorboard"],
        )

        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["validation"],
        )

        # Train
        logger.info("Training started...")
        train_result = trainer.train()

        # Save model
        logger.info(f"Saving model to {self.output_dir}")
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)

        # Save training metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

        logger.info("Training complete!")
        logger.info(f"Final train loss: {metrics['train_loss']:.4f}")

    def evaluate(self) -> Dict:
        """Evaluate the model."""
        logger.info("Evaluating model...")

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=(self.model_type == "mlm"),
            mlm_probability=0.15 if self.model_type == "mlm" else None,
        )

        # Training arguments (for evaluation)
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            per_device_eval_batch_size=self.batch_size,
            fp16=self.fp16 and torch.cuda.is_available(),
        )

        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            eval_dataset=self.dataset["test"],
        )

        # Evaluate
        metrics = trainer.evaluate()

        logger.info("Evaluation complete!")
        logger.info(f"Test loss: {metrics['eval_loss']:.4f}")
        logger.info(f"Test perplexity: {torch.exp(torch.tensor(metrics['eval_loss'])):.4f}")

        # Save evaluation metrics
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

        return metrics

    def save_training_config(self) -> None:
        """Save training configuration."""
        config = {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "num_epochs": self.num_epochs,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "fp16": self.fp16,
        }

        config_path = self.output_dir / "training_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Saved training config: {config_path}")


def main():
    parser = argparse.ArgumentParser(description="Train language model on Wikipedia data")
    parser.add_argument(
        "--data", type=str, required=True, help="Path to processed Wikipedia dataset"
    )
    parser.add_argument(
        "--model", type=str, default="ai4bharat/indic-bert", help="Pretrained model name or path"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/wikipedia/base",
        help="Output directory for trained model",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["mlm", "clm"],
        default="mlm",
        help="Model type: mlm (masked LM) or clm (causal LM)",
    )
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per device")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument(
        "--gradient-accumulation-steps", type=int, default=4, help="Gradient accumulation steps"
    )
    parser.add_argument("--no-fp16", action="store_true", help="Disable mixed precision training")
    parser.add_argument(
        "--save-steps", type=int, default=1000, help="Save checkpoint every N steps"
    )
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate, do not train")

    args = parser.parse_args()

    data_dir = Path(args.data)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Initialize trainer
        trainer = WikipediaLMTrainer(
            model_name=args.model,
            data_dir=data_dir,
            output_dir=output_dir,
            model_type=args.model_type,
            max_length=args.max_length,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_epochs=args.epochs,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            fp16=not args.no_fp16,
            save_steps=args.save_steps,
        )

        # Load components
        trainer.load_tokenizer()
        trainer.load_model()
        trainer.load_dataset()
        trainer.prepare_dataset()

        # Train or evaluate
        if args.eval_only:
            trainer.evaluate()
        else:
            trainer.save_training_config()
            trainer.train()
            trainer.evaluate()

        logger.info("Process complete!")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
