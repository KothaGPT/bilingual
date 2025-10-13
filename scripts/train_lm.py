#!/usr/bin/env python3
"""
Fine-tune a bilingual language model.

This script fine-tunes a pre-trained language model (e.g., mT5, mBART, or mGPT)
for Bangla and English text generation using the Hugging Face Transformers library.

Usage:
    python scripts/train_lm.py \
        --train_data data/processed/train.txt \
        --val_data data/processed/val.txt \
        --model_name_or_path "facebook/mbart-large-cc25" \
        --output_dir models/language_model
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

from datasets import Dataset, load_dataset

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_data(
    train_file: str, val_file: str, max_samples: Optional[int] = None
) -> Tuple[Dataset, Dataset]:
    """
    Load and prepare the dataset for training.

    Args:
        train_file: Path to training data file (text file with one example per line)
        val_file: Path to validation data file (text file with one example per line)
        max_samples: Maximum number of samples to use (for testing)

    Returns:
        Tuple of (train_dataset, val_dataset)
    """

    # Load data from text files
    def read_lines(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    train_texts = read_lines(train_file)
    val_texts = read_lines(val_file)

    if max_samples is not None:
        train_texts = train_texts[:max_samples]
        val_texts = val_texts[: min(max_samples, len(val_texts))]

    train_dataset = Dataset.from_dict({"text": train_texts})
    val_dataset = Dataset.from_dict({"text": val_texts})

    return train_dataset, val_dataset


def tokenize_function(examples: Dict[str, List[str]], tokenizer, max_length: int = 128) -> Dict:
    """
    Tokenize the examples.

    Args:
        examples: Dictionary containing text examples
        tokenizer: Tokenizer for the model
        max_length: Maximum length of the tokenized sequences

    Returns:
        Dictionary containing tokenized inputs
    """
    return tokenizer(
        examples["text"],
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_special_tokens_mask=True,
    )


def train(
    train_data: str,
    val_data: str,
    model_name_or_path: str,
    output_dir: str,
    max_length: int = 128,
    batch_size: int = 8,
    num_epochs: int = 3,
    learning_rate: float = 5e-5,
    weight_decay: float = 0.01,
    warmup_steps: int = 500,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
    seed: int = 42,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    fp16: bool = torch.cuda.is_available(),
):
    """
    Train a language model.

    Args:
        train_data: Path to training data file (text file with one example per line)
        val_data: Path to validation data file (text file with one example per line)
        model_name_or_path: Model identifier from huggingface.co/models or a local path
        output_dir: Directory to save the model and checkpoints
        max_length: Maximum length of input sequences
        batch_size: Batch size for training and evaluation
        num_epochs: Number of training epochs
        learning_rate: Learning rate for the optimizer
        weight_decay: Weight decay for the optimizer
        warmup_steps: Number of warmup steps for learning rate scheduler
        gradient_accumulation_steps: Number of updates steps to accumulate before performing a backward/update pass
        max_grad_norm: Maximum gradient norm (for gradient clipping)
        seed: Random seed for reproducibility
        max_train_samples: Maximum number of training samples to use (for testing)
        max_val_samples: Maximum number of validation samples to use (for testing)
        fp16: Whether to use mixed precision training
    """
    # Set seed for reproducibility
    set_seed(seed)

    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer and model
    logger.info(f"Loading tokenizer and model from {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    config = AutoConfig.from_pretrained(model_name_or_path)

    # For causal language modeling
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        config=config,
    )

    # Add special tokens if they don't exist
    special_tokens = ["<|bn|>", "<|en|>"]
    tokenizer.add_tokens(special_tokens, special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    # Load and preprocess data
    logger.info("Loading and preprocessing data")
    train_dataset, val_dataset = load_data(train_data, val_data, max_samples=max_train_samples)

    # Tokenize the data
    tokenized_train = train_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, max_length=max_length),
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing training data",
    )

    tokenized_val = val_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, max_length=max_length),
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing validation data",
    )

    # Set up data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False  # Causal language modeling
    )

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        evaluation_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=weight_decay,
        save_total_limit=3,
        num_train_epochs=num_epochs,
        fp16=fp16,
        warmup_steps=warmup_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_grad_norm=max_grad_norm,
        logging_dir=str(output_dir / "logs"),
        logging_steps=100,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        report_to=["tensorboard"],
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Train the model
    logger.info("Starting training")
    train_result = trainer.train()

    # Save the model and tokenizer
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    # Log training summary
    logger.info("Training completed!")
    logger.info(f"Training metrics: {train_result.metrics}")

    # Evaluate the model
    logger.info("Evaluating on validation set")
    eval_metrics = trainer.evaluate()
    logger.info(f"Validation metrics: {eval_metrics}")

    # Save metrics
    with open(output_dir / "training_metrics.json", "w") as f:
        json.dump(train_result.metrics, f, indent=2)
    with open(output_dir / "eval_metrics.json", "w") as f:
        json.dump(eval_metrics, f, indent=2)

    logger.info(f"Model and tokenizer saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a language model")

    # Required parameters
    parser.add_argument(
        "--train_data",
        type=str,
        required=True,
        help="Path to training data file (text file with one example per line)",
    )
    parser.add_argument(
        "--val_data",
        type=str,
        required=True,
        help="Path to validation data file (text file with one example per line)",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="facebook/mbart-large-cc25",
        help="Model identifier from huggingface.co/models or a local path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/language_model",
        help="Directory to save the model and checkpoints",
    )

    # Training parameters
    parser.add_argument("--max_length", type=int, default=128, help="Max sequence length")
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for training and evaluation"
    )
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Number of warmup steps")
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps"
    )
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--max_train_samples", type=int, default=None, help="Max training samples (for testing)"
    )
    parser.add_argument(
        "--max_val_samples", type=int, default=None, help="Max validation samples (for testing)"
    )
    parser.add_argument(
        "--no_fp16", action="store_false", dest="fp16", help="Disable mixed precision training"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save command line arguments
    with open(output_dir / "training_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Train the model
    train(
        train_data=args.train_data,
        val_data=args.val_data,
        model_name_or_path=args.model_name_or_path,
        output_dir=args.output_dir,
        max_length=args.max_length,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        fp16=args.fp16,
    )


if __name__ == "__main__":
    main()
