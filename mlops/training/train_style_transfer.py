#!/usr/bin/env python3
"""
Training script for Style Transfer model.

This script trains the seq2seq style transfer model for converting
text between different styles (formal, informal, poetic, etc.).

Usage:
    python scripts/train_style_transfer.py \
        --dataset_path datasets/literary/style_pairs.json \
        --model_path models/style-transfer \
        --base_model google/mt5-small \
        --epochs 5
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

from datasets import Dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Style Transfer Model")
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the style transfer dataset (JSON with source/target pairs)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/style-transfer",
        help="Path to save the trained model",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="google/mt5-small",
        help="Base seq2seq model (e.g., google/mt5-small, facebook/mbart-large-cc25)",
    )
    parser.add_argument(
        "--source_style",
        type=str,
        default=None,
        help="Source style (optional, for filtering dataset)",
    )
    parser.add_argument(
        "--target_style",
        type=str,
        default=None,
        help="Target style (optional, for filtering dataset)",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum sequence length")
    parser.add_argument(
        "--validation_split", type=float, default=0.1, help="Fraction of data for validation"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision training")
    return parser.parse_args()


def load_style_dataset(
    dataset_path: str,
    source_style: str = None,
    target_style: str = None,
    validation_split: float = 0.1,
) -> tuple:
    """
    Load style transfer dataset.

    Expected format (JSON):
    [
        {"source": "text1", "target": "text1_styled", "source_style": "formal", "target_style": "informal"},
        ...
    ]

    Or simple format:
    [
        {"source": "text1", "target": "text1_styled"},
        ...
    ]
    """
    logger.info(f"Loading dataset from {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Filter by style if specified
    if source_style or target_style:
        filtered_data = []
        for item in data:
            if source_style and item.get("source_style") != source_style:
                continue
            if target_style and item.get("target_style") != target_style:
                continue
            filtered_data.append(item)
        data = filtered_data
        logger.info(f"Filtered to {len(data)} examples")

    logger.info(f"Loaded {len(data)} style pairs")

    # Split into train and validation
    split_idx = int(len(data) * (1 - validation_split))
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    logger.info(f"Train: {len(train_data)}, Validation: {len(val_data)}")

    # Create datasets
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)

    return train_dataset, val_dataset


def preprocess_function(examples: Dict, tokenizer, max_length: int, target_style: str = None):
    """Preprocess examples for seq2seq training."""
    # Add style markers to inputs
    if target_style:
        inputs = [f"convert to {target_style}: {text}" for text in examples["source"]]
    else:
        # Use target_style from data if available
        if "target_style" in examples:
            inputs = [
                f"convert to {style}: {text}"
                for text, style in zip(examples["source"], examples["target_style"])
            ]
        else:
            inputs = examples["source"]

    # Tokenize inputs
    model_inputs = tokenizer(
        inputs,
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )

    # Tokenize targets
    labels = tokenizer(
        examples["target"],
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def train_style_transfer(args):
    """
    Train the style transfer model.

    Args:
        args: Command line arguments
    """
    # Set seed
    set_seed(args.seed)

    logger.info("=" * 60)
    logger.info("Style Transfer Model Training")
    logger.info("=" * 60)
    logger.info(f"Dataset: {args.dataset_path}")
    logger.info(f"Base model: {args.base_model}")
    logger.info(f"Output: {args.model_path}")
    if args.source_style:
        logger.info(f"Source style: {args.source_style}")
    if args.target_style:
        logger.info(f"Target style: {args.target_style}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info("=" * 60)

    # Load dataset
    train_dataset, val_dataset = load_style_dataset(
        args.dataset_path, args.source_style, args.target_style, args.validation_split
    )

    # Load tokenizer and model
    logger.info(f"Loading tokenizer from {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    logger.info(f"Loading model from {args.base_model}")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)

    # Preprocess datasets
    logger.info("Preprocessing datasets...")
    train_dataset = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer, args.max_length, args.target_style),
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    val_dataset = val_dataset.map(
        lambda x: preprocess_function(x, tokenizer, args.max_length, args.target_style),
        batched=True,
        remove_columns=val_dataset.column_names,
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
    )

    # Training arguments
    output_dir = Path(args.model_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
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
        predict_with_generate=True,
        generation_max_length=args.max_length,
        report_to=["tensorboard"],
        seed=args.seed,
    )

    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Train
    logger.info("Starting training...")
    train_result = trainer.train()

    # Save model
    logger.info(f"Saving model to {args.model_path}")
    trainer.save_model(args.model_path)
    tokenizer.save_pretrained(args.model_path)

    # Save metrics
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
    train_style_transfer(args)


if __name__ == "__main__":
    main()
