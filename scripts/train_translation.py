#!/usr/bin/env python3
"""
Train a bilingual translation model.

This script fine-tunes a pre-trained model (e.g., mT5, mBART, or MarianMT)
for Bangla-English translation using the Hugging Face Transformers library.
It uses a custom, pre-trained SentencePiece tokenizer.

Usage:
    python scripts/train_translation.py \
        --train_data data/processed/train.jsonl \
        --val_data data/processed/val.jsonl \
        --model_name_or_path "Helsinki-NLP/opus-mt-en-mul" \
        --tokenizer_path "models/tokenizer/" \
        --output_dir models/translation
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from peft import get_peft_model, LoraConfig

from datasets import Dataset, load_dataset

# Add parent directory to path to allow importing from `bilingual`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


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
        train_file: Path to training data file (JSONL format)
        val_file: Path to validation data file (JSONL format)
        max_samples: Maximum number of samples to use (for testing)

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    # Load data from JSONL files
    data_files = {"train": train_file, "validation": val_file}
    raw_datasets = load_dataset("json", data_files=data_files)

    if max_samples is not None:
        for split in ["train", "validation"]:
            if len(raw_datasets[split]) > max_samples:
                raw_datasets[split] = raw_datasets[split].select(range(max_samples))

    return raw_datasets["train"], raw_datasets["validation"]


def preprocess_function(
    examples: Dict[str, List[str]],
    tokenizer,
    max_source_length: int = 128,
    max_target_length: int = 128,
    source_lang: str = "en",
    target_lang: str = "bn",
) -> Dict:
    """
    Preprocess the data for the model.

    Args:
        examples: Dictionary containing source and target texts
        tokenizer: Tokenizer for the model
        max_source_length: Maximum length of source sequence
        max_target_length: Maximum length of target sequence
        source_lang: Source language code
        target_lang: Target language code

    Returns:
        Dictionary containing model inputs and labels
    """
    inputs = [f"{ex[source_lang + '_text']}" for ex in examples["translation"]]
    targets = [f"{ex[target_lang + '_text']}" for ex in examples["translation"]]

    model_inputs = tokenizer(
        inputs, max_length=max_source_length, padding="max_length", truncation=True
    )

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets, max_length=max_target_length, padding="max_length", truncation=True
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def train(
    train_data: str,
    val_data: str,
    model_name_or_path: str,
    tokenizer_path: str,
    output_dir: str,
    use_qlora: bool = False,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    source_lang: str = "en",
    target_lang: str = "bn",
    max_source_length: int = 128,
    max_target_length: int = 128,
    batch_size: int = 16,
    num_epochs: int = 3,
    learning_rate: float = 2e-4,  # Higher LR for LoRA
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
    Train a translation model.

    Args:
        train_data: Path to training data file (JSONL format)
        val_data: Path to validation data file (JSONL format)
        model_name_or_path: Model identifier from huggingface.co/models or a local path
        tokenizer_path: Path to the directory containing the custom tokenizer
        output_dir: Directory to save the model and checkpoints
        use_qlora: Whether to use QLoRA for fine-tuning
        lora_r: LoRA attention dimension
        lora_alpha: LoRA alpha scaling parameter
        lora_dropout: Dropout probability for LoRA layers
        source_lang: Source language code
        target_lang: Target language code
        ... (and other training arguments)
    """
    # Set seed for reproducibility
    set_seed(seed)

    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Quantization and model loading
    quantization_config = None
    if use_qlora:
        logger.info("Using QLoRA. Preparing 4-bit quantization.")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    logger.info(f"Loading model from {model_name_or_path}")
    config = AutoConfig.from_pretrained(model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name_or_path,
        config=config,
        quantization_config=quantization_config,
        device_map="auto",
    )

    # Resize token embeddings to match the custom tokenizer
    model.resize_token_embeddings(len(tokenizer))

    # Add special tokens for language direction if they don't exist
    special_tokens = [f"<{source_lang}>", f"<{target_lang}>"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    model.resize_token_embeddings(len(tokenizer))

    if use_qlora:
        logger.info("Applying LoRA adapter to the model.")
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="SEQ_2_SEQ_LM",
            # Target attention layers for Seq2Seq models
            target_modules=["q_proj", "v_proj"],
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()


    # Load and preprocess data
    logger.info("Loading and preprocessing data")
    train_dataset, val_dataset = load_data(
        train_data, val_data, max_samples=max_train_samples
    )

    # Preprocess the data
    def preprocess_fn(examples):
        # This function needs to handle the structure of the JSONL file.
        # Assuming the JSONL has a "translation" field like: {"translation": {"en_text": "...", "bn_text": "..."}}
        inputs = [ex[source_lang + "_text"] for ex in examples["translation"]]
        targets = [ex[target_lang + "_text"] for ex in examples["translation"]]

        # Prepend language tokens
        inputs = [f"<{source_lang}> {text}" for text in inputs]
        targets = [f"<{target_lang}> {text}" for text in targets]

        model_inputs = tokenizer(
            inputs, max_length=max_source_length, padding="max_length", truncation=True
        )

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets, max_length=max_target_length, padding="max_length", truncation=True
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_dataset = train_dataset.map(
        preprocess_fn,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Processing train data",
    )

    val_dataset = val_dataset.map(
        preprocess_fn,
        batched=True,
        remove_columns=val_dataset.column_names,
        desc="Processing validation data",
    )

    # Set up data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, label_pad_token_id=tokenizer.pad_token_id
    )

    # Set up training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        evaluation_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=weight_decay,
        save_total_limit=3,
        num_train_epochs=num_epochs,
        predict_with_generate=True,
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
        seed=seed,
    )

    # Initialize Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Train the model
    logger.info("Starting training")
    train_result = trainer.train()

    # Save the model and tokenizer
    trainer.save_model()
    tokenizer.save_pretrained(str(output_dir))

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
    parser = argparse.ArgumentParser(description="Train a translation model")

    # Required parameters
    parser.add_argument(
        "--train_data",
        type=str,
        required=True,
        help="Path to training data file (JSONL format)",
    )
    parser.add_argument(
        "--val_data",
        type=str,
        required=True,
        help="Path to validation data file (JSONL format)",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="Helsinki-NLP/opus-mt-en-mul",
        help="Model identifier from huggingface.co/models or a local path",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="models/tokenizer/",
        help="Path to the directory containing the custom tokenizer",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/translation",
        help="Directory to save the model and checkpoints",
    )

    # QLoRA parameters
    parser.add_argument("--use_qlora", action="store_true", help="Enable QLoRA fine-tuning")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA r parameter")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")

    # Language parameters
    parser.add_argument("--source_lang", type=str, default="en", help="Source language code")
    parser.add_argument("--target_lang", type=str, default="bn", help="Target language code")

    # Training parameters
    parser.add_argument(
        "--max_source_length", type=int, default=128, help="Max source sequence length"
    )
    parser.add_argument(
        "--max_target_length", type=int, default=128, help="Max target sequence length"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for training and evaluation"
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
        tokenizer_path=args.tokenizer_path,
        output_dir=args.output_dir,
        use_qlora=args.use_qlora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
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
