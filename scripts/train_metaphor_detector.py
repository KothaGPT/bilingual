#!/usr/bin/env python3
"""
Metaphor and Simile Detection Training Script.

This script trains a token classification model to detect metaphors and similes in text.

Usage:
    python scripts/train_metaphor_detector.py \
        --config models/metaphor-simile-detector/task_config.json \
        --model_name_or_path bert-base-multilingual-cased \
        --train_data data/processed/metaphor/train.jsonl \
        --val_data data/processed/metaphor/val.jsonl \
        --output_dir models/metaphor-simile-detector/checkpoints
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from datasets import Dataset, load_metric
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

# Set up logging
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """Load and validate configuration."""
    with open(config_path) as f:
        config = json.load(f)
    
    # Validate required fields
    required_fields = ["task_name", "model_type", "num_labels", "label_names", "training"]
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required config field: {field}")
    
    return config


def load_data(file_path: str) -> Dataset:
    """Load dataset from JSONL file."""
    return Dataset.from_json(file_path)


def tokenize_and_align_labels(
    examples: Dict,
    tokenizer,
    label_to_id: Dict[str, int],
    max_length: int,
    label_all_tokens: bool = False,
) -> Dict:
    """Tokenize and align labels with tokens."""
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=max_length,
        padding="max_length",
    )

    labels = []
    for i, label in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        
        for word_idx in word_ids:
            if word_idx is None:
                # Special tokens get -100
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                # Only label the first token of each word
                label_ids.append(label_to_id.get(label[word_idx], -100))
            else:
                # For other tokens, use -100 or the current label
                label_ids.append(label_to_id.get(label[word_idx], -100) if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def compute_metrics(p):
    """Compute precision, recall, F1 score for each class."""
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [p for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [l for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    # Flatten predictions and labels
    flat_predictions = [p for sublist in true_predictions for p in sublist]
    flat_labels = [l for sublist in true_labels for l in sublist]

    # Calculate metrics
    precision_metric = load_metric("precision")
    recall_metric = load_metric("recall")
    f1_metric = load_metric("f1")
    
    results = {}
    
    # Overall metrics
    results["overall_precision"] = precision_metric.compute(
        predictions=flat_predictions, references=flat_labels, average="weighted"
    )["precision"]
    results["overall_recall"] = recall_metric.compute(
        predictions=flat_predictions, references=flat_labels, average="weighted"
    )["recall"]
    results["overall_f1"] = f1_metric.compute(
        predictions=flat_predictions, references=flat_labels, average="weighted"
    )["f1"]
    
    # Per-class metrics
    precision = precision_metric.compute(
        predictions=flat_predictions, references=flat_labels, average=None
    )["precision"]
    recall = recall_metric.compute(
        predictions=flat_predictions, references=flat_labels, average=None
    )["recall"]
    f1 = f1_metric.compute(
        predictions=flat_predictions, references=flat_labels, average=None
    )["f1"]
    
    # Add per-class metrics
    for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
        results[f"class_{i}_precision"] = p
        results[f"class_{i}_recall"] = r
        results[f"class_{i}_f1"] = f
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Train a metaphor and simile detection model.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration JSON file",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--train_data",
        type=str,
        required=True,
        help="Path to the training data file (JSONL format)",
    )
    parser.add_argument(
        "--val_data",
        type=str,
        required=True,
        help="Path to the validation data file (JSONL format)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory where the model checkpoints and predictions will be written.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--do_train",
        action="store_true",
        help="Whether to run training.",
    )
    parser.add_argument(
        "--do_eval",
        action="store_true",
        help="Whether to run eval on the dev set.",
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    training_args = config["training"]
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    
    # Set the verbosity to info of the Transformers logger
    logger.setLevel(logging.INFO)
    
    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save the configuration
    with open(os.path.join(args.output_dir, "training_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    # Load datasets
    logger.info("Loading datasets")
    train_dataset = load_data(args.train_data)
    val_dataset = load_data(args.val_data)
    
    # Prepare label mappings
    label_list = config["label_names"]
    label_to_id = {label: i for i, label in enumerate(label_list)}
    id_to_label = {i: label for i, label in enumerate(label_list)}
    
    # Save label mappings
    with open(os.path.join(args.output_dir, "label_mapping.json"), "w") as f:
        json.dump({"label2id": label_to_id, "id2label": id_to_label}, f, indent=2)
    
    # Load tokenizer and model
    logger.info("Loading tokenizer and model")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    # Tokenize datasets
    logger.info("Tokenizing datasets")
    train_dataset = train_dataset.map(
        lambda examples: tokenize_and_align_labels(
            examples, tokenizer, label_to_id, config["max_length"]
        ),
        batched=True,
    )
    
    val_dataset = val_dataset.map(
        lambda examples: tokenize_and_align_labels(
            examples, tokenizer, label_to_id, config["max_length"]
        ),
        batched=True,
    )
    
    # Load model
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=len(label_list),
        id2label=id_to_label,
        label2id=label_to_id,
    )
    
    # Set up data collator
    data_collator = DataCollatorForTokenClassification(tokenizer)
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=training_args["num_epochs"],
        per_device_train_batch_size=training_args["batch_size"],
        per_device_eval_batch_size=training_args["batch_size"],
        gradient_accumulation_steps=training_args.get("gradient_accumulation_steps", 1),
        learning_rate=training_args["learning_rate"],
        lr_scheduler_type=training_args["lr_scheduler_type"],
        warmup_ratio=training_args["warmup_ratio"],
        weight_decay=training_args["weight_decay"],
        fp16=training_args.get("fp16", False),
        evaluation_strategy=training_args["evaluation_strategy"],
        save_strategy=training_args["save_strategy"],
        load_best_model_at_end=training_args["load_best_model_at_end"],
        metric_for_best_model=training_args["metric_for_best_model"],
        greater_is_better=training_args["greater_is_better"],
        logging_steps=training_args.get("logging_steps", 100),
        save_total_limit=training_args.get("save_total_limit", 3),
        report_to=["tensorboard"],
        push_to_hub=False,
    )
    
    # Set up early stopping
    callbacks = []
    if "optimization" in config and "early_stopping_patience" in config["optimization"]:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=config["optimization"]["early_stopping_patience"],
                early_stopping_threshold=config["optimization"].get("early_stopping_threshold", 0.0),
            )
        )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if args.do_train else None,
        eval_dataset=val_dataset if args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )
    
    # Training
    if args.do_train:
        logger.info("*** Train ***")
        checkpoint = None
        if args.resume_from_checkpoint is not None:
            checkpoint = args.resume_from_checkpoint
        elif os.path.isdir(args.model_name_or_path):
            checkpoint = args.model_name_or_path
        
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    
    # Evaluation
    if args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        
        metrics["eval_samples"] = len(val_dataset)
        
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
