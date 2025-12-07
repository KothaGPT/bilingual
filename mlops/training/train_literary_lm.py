#!/usr/bin/env python3
"""
Training script for Literary Language Model.

This script handles the training of the literary LM on Bangla literary datasets
using the Hugging Face Transformers library.

Usage:
    python scripts/train_literary_lm.py \
        --dataset_path datasets/literary/corpus.txt \
        --model_path models/literary-lm \
        --base_model gpt2 \
        --epochs 3 \
        --batch_size 8
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)

from datasets import Dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Literary Language Model")
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the literary dataset (text file, one example per line)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/literary-lm",
        help="Path to save the trained model",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="gpt2",
        help="Base model to fine-tune (e.g., gpt2, facebook/mbart-large-cc25)",
    )
    parser.add_argument(
        "--tokenizer_path", type=str, default=None, help="Path to custom tokenizer (optional)"
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument(
        "--validation_split", type=float, default=0.1, help="Fraction of data to use for validation"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision training")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps",
    )
    return parser.parse_args()


def load_dataset(dataset_path: str, validation_split: float = 0.1) -> tuple:
    """
    Load and prepare the literary dataset.

    Args:
        dataset_path: Path to the dataset file
        validation_split: Fraction of data for validation

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    logger.info(f"Loading dataset from {dataset_path}")

    # Read text file
    with open(dataset_path, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]

    logger.info(f"Loaded {len(texts)} examples")

    # Split into train and validation
    split_idx = int(len(texts) * (1 - validation_split))
    train_texts = texts[:split_idx]
    val_texts = texts[split_idx:]

    logger.info(f"Train: {len(train_texts)}, Validation: {len(val_texts)}")

    # Create datasets
    train_dataset = Dataset.from_dict({"text": train_texts})
    val_dataset = Dataset.from_dict({"text": val_texts})

    return train_dataset, val_dataset


def tokenize_function(examples: Dict, tokenizer, max_length: int):
    """Tokenize the examples."""
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_special_tokens_mask=True,
    )


def train_literary_lm(args):
    """
    Train the literary language model.

    Args:
        args: Command line arguments
    """
    # Set seed for reproducibility
    set_seed(args.seed)

    logger.info("=" * 60)
    logger.info("Literary Language Model Training")
    logger.info("=" * 60)
    logger.info(f"Dataset: {args.dataset_path}")
    logger.info(f"Base model: {args.base_model}")
    logger.info(f"Output: {args.model_path}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info("=" * 60)

    # Load dataset
    train_dataset, val_dataset = load_dataset(args.dataset_path, args.validation_split)

    # Load tokenizer
    tokenizer_path = args.tokenizer_path or args.base_model
    logger.info(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize datasets
    logger.info("Tokenizing datasets...")
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer, args.max_length),
        batched=True,
        remove_columns=["text"],
    )
    val_dataset = val_dataset.map(
        lambda x: tokenize_function(x, tokenizer, args.max_length),
        batched=True,
        remove_columns=["text"],
    )

    # Load model
    logger.info(f"Loading model from {args.base_model}")
    config = AutoConfig.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, config=config)

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )

    # Training arguments
    output_dir = Path(args.model_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=str(output_dir / "logs"),
        logging_steps=100,
        eval_steps=500,
        save_steps=1000,
        evaluation_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        fp16=args.fp16,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        report_to=["tensorboard"],
        seed=args.seed,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    # Train
    logger.info("Starting training...")
    train_result = trainer.train()

    # Save model
    logger.info(f"Saving model to {args.model_path}")
    trainer.save_model(args.model_path)
    tokenizer.save_pretrained(args.model_path)

    # Save training metrics
    metrics = train_result.metrics
    metrics_path = output_dir / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("=" * 60)
    logger.info("Training completed successfully!")
    logger.info(f"Model saved to: {args.model_path}")
    logger.info(f"Training loss: {metrics.get('train_loss', 'N/A')}")
    logger.info("=" * 60)


def main():
    """Main entry point."""
    args = parse_args()
    train_literary_lm(args)


if __name__ == "__main__":
    main()
