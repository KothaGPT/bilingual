#!/usr/bin/env python3
"""
Example script for fine-tuning a bilingual language model.

This script demonstrates how to:
1. Load and prepare bilingual data
2. Fine-tune a pretrained multilingual model
3. Evaluate the model
4. Save the trained model

Usage:
    python examples/train_language_model.py \
        --data datasets/processed/ \
        --model bert-base-multilingual-cased \
        --output models/bilingual-lm/
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bilingual.data_utils import BilingualDataset  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Train a bilingual language model")

    parser.add_argument(
        "--data", required=True, help="Path to processed dataset directory"
    )
    parser.add_argument(
        "--model",
        default="bert-base-multilingual-cased",
        help="Base model to fine-tune",
    )
    parser.add_argument(
        "--output", default="models/bilingual-lm/", help="Output directory for model"
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument(
        "--learning-rate", type=float, default=2e-5, help="Learning rate"
    )
    parser.add_argument("--max-length", type=int, default=512, help="Max sequence length")

    args = parser.parse_args()

    print("=" * 60)
    print("BILINGUAL LANGUAGE MODEL TRAINING")
    print("=" * 60)
    print()

    # Check if transformers is available
    try:
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )
    except ImportError:
        print("Error: transformers library is required for model training")
        print("Install it with: pip install transformers")
        sys.exit(1)

    # Load datasets
    print("Loading datasets...")
    data_dir = Path(args.data)

    train_data = BilingualDataset(file_path=str(data_dir / "train.jsonl"))
    val_data = BilingualDataset(file_path=str(data_dir / "validation.jsonl"))

    print(f"  Train samples: {len(train_data)}")
    print(f"  Validation samples: {len(val_data)}")
    print()

    # Load model and tokenizer
    print(f"Loading base model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print()

    # Tokenize datasets
    print("Tokenizing datasets...")

    def tokenize_function(examples):
        return tokenizer(
            [ex["text"] for ex in examples],
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
        )

    # Note: This is a simplified example
    # In practice, you'd want to use datasets library for better performance
    print("  (Using simplified tokenization for demo)")
    print()

    # Training arguments
    output_dir = Path(args.output)
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
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=1000,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    print("Training configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Output: {output_dir}")
    print()

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False  # Causal LM (not masked)
    )

    # Note: This is a placeholder - actual implementation would need proper dataset formatting
    print("Note: This is a demo script showing the training structure.")
    print("For actual training, you need to:")
    print("  1. Properly format your datasets")
    print("  2. Use the datasets library for efficient loading")
    print("  3. Configure appropriate hyperparameters")
    print("  4. Set up proper evaluation metrics")
    print()

    print("Example training command structure:")
    print(
        """
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
    )
    
    trainer.train()
    trainer.save_model(output_dir)
    """
    )

    print()
    print("For a complete training setup, refer to:")
    print("  - Hugging Face documentation: https://huggingface.co/docs/transformers/")
    print("  - Training examples: https://github.com/huggingface/transformers/tree/main/examples")


if __name__ == "__main__":
    main()
