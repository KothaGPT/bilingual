#!/usr/bin/env python3
"""
Training script for the Bangla-English translation model.
"""
import argparse
import json
import os

# Add the project root to the Python path
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter

sys.path.append(str(Path(__file__).parent.parent))

from bilingual.models.transformer import TransformerModel
from bilingual.preprocessing.text_processor import TextProcessor


class TranslationDataset(Dataset):
    """Dataset for translation tasks with support for multiple data formats."""

    def __init__(self, data_sources: list, max_length: int = 128, is_training: bool = True):
        """Initialize the dataset.

        Args:
            data_sources: List of dicts containing 'path', 'type', and 'weight' for each data source
            max_length: Maximum sequence length
            is_training: Whether this is for training (enables data augmentation)
        """
        self.data = []
        self.max_length = max_length
        self.is_training = is_training

        for source in data_sources:
            path = Path(source["path"])
            if not path.exists():
                print(f"Warning: Data file not found: {path}")
                continue

            if source["type"] == "parquet":
                df = pd.read_parquet(path)
                source_data = [
                    {"source": row["en"], "target": row["bn"], "type": "parallel"}
                    for _, row in df.iterrows()
                ]
            elif source["type"] == "jsonl":
                source_data = []
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            item = json.loads(line)
                            if "conversation" in item:  # Handle conversational data
                                for turn in item["conversation"]:
                                    if turn["role"] == "user":
                                        source_data.append(
                                            {
                                                "source": turn["content"],
                                                "target": "",
                                                "type": "conversation",
                                                "lang": turn.get("lang", "en"),
                                            }
                                        )
                            elif "text" in item:  # Handle single text examples
                                source_data.append(
                                    {
                                        "source": item["text"],
                                        "target": "",
                                        "type": item.get("type", "text"),
                                        "lang": item.get("lang", "en"),
                                    }
                                )
                        except json.JSONDecodeError:
                            continue

            # Apply sampling weight
            if "weight" in source and source["weight"] < 1.0 and self.is_training:
                sample_size = int(len(source_data) * source["weight"])
                source_data = random.sample(source_data, min(sample_size, len(source_data)))

            self.data.extend(source_data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single example from the dataset with support for multiple data types."""
        item = self.data[idx]

        # Handle different data types
        if item["type"] == "parallel":
            # Standard parallel data
            src_text = item["source"]
            tgt_text = item["target"]

        elif item["type"] == "conversation":
            # For conversation data, use the user's message as source
            # and the assistant's response as target (if available)
            src_text = item["source"]
            tgt_text = item.get("target", "")  # May be empty for single-turn

        elif item["type"] == "code-switched":
            # For code-switched data, we might want to translate to one language
            src_text = item["source"]
            tgt_text = item.get("target", "")  # Could be empty for inference

            # Data augmentation: randomly translate to one language
            if self.is_training and random.random() > 0.5:
                if item.get("lang") == "en":
                    tgt_text = src_text  # Self-reconstruction for monolingual data
                # Add more augmentation strategies here

        elif item["type"] == "dialect":
            # For dialect data, we might want to normalize to standard Bangla
            src_text = item["source"]
            tgt_text = item.get("standard_bangla", item["source"])

        # Tokenize the source and target texts
        src_tokens = self.tokenizer.encode(
            src_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        tgt_tokens = self.tokenizer.encode(
            tgt_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": src_tokens.squeeze(0),
            "labels": tgt_tokens.squeeze(0),
            "attention_mask": (src_tokens != self.tokenizer.pad_token_id).long().squeeze(0),
            "data_type": item.get("type", "parallel"),
        }
        tgt = torch.tensor(item["bn_ids"], dtype=torch.long)

        # Add BOS and EOS tokens
        bos_token = 2  # Assuming BOS token ID is 2
        eos_token = 3  # Assuming EOS token ID is 3

        # Prepare source (encoder input)
        src_input = src[: self.max_length - 1]  # Leave room for EOS
        src_input = torch.cat([src_input, torch.tensor([eos_token])])  # Add EOS

        # Prepare target (decoder input and output)
        tgt_input = tgt[: self.max_length - 1]  # Leave room for EOS
        tgt_output = torch.cat(
            [tgt_input[1:], torch.tensor([eos_token])]
        )  # Shift right for teacher forcing
        tgt_input = torch.cat([torch.tensor([bos_token]), tgt_input])  # Add BOS

        return {
            "src": src_input,
            "tgt_input": tgt_input,
            "tgt_output": tgt_output,
            "src_text": item.get("en_text", ""),
            "tgt_text": item.get("bn_text", ""),
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function for DataLoader."""
    # Find the maximum length in the batch
    src_max_len = max(item["src"].size(0) for item in batch)
    tgt_max_len = max(item["tgt_input"].size(0) for item in batch)

    # Initialize tensors with padding
    batch_size = len(batch)
    src_padded = torch.zeros(batch_size, src_max_len, dtype=torch.long)
    tgt_input_padded = torch.zeros(batch_size, tgt_max_len, dtype=torch.long)
    tgt_output_padded = torch.zeros(batch_size, tgt_max_len, dtype=torch.long)

    # Create padding masks
    src_padding_mask = torch.ones(batch_size, src_max_len, dtype=torch.bool)
    tgt_padding_mask = torch.ones(batch_size, tgt_max_len, dtype=torch.bool)

    # Fill tensors with data
    for i, item in enumerate(batch):
        src_len = item["src"].size(0)
        tgt_len = item["tgt_input"].size(0)

        src_padded[i, :src_len] = item["src"]
        tgt_input_padded[i, :tgt_len] = item["tgt_input"]
        tgt_output_padded[i, :tgt_len] = item["tgt_output"]

        # Update padding masks (False means not padding)
        src_padding_mask[i, :src_len] = False
        tgt_padding_mask[i, :tgt_len] = False

    return {
        "src": src_padded,
        "tgt_input": tgt_input_padded,
        "tgt_output": tgt_output_padded,
        "src_padding_mask": src_padding_mask,
        "tgt_padding_mask": tgt_padding_mask,
        "src_texts": [item["src_text"] for item in batch],
        "tgt_texts": [item["tgt_text"] for item in batch],
    }


class Trainer:
    """Trainer class for the translation model with enhanced dataset support."""

    def __init__(self, config: dict, tokenizer=None):
        """Initialize the trainer.

        Args:
            config: Configuration dictionary
            tokenizer: Pre-initialized tokenizer (optional)
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = tokenizer

        # Set random seeds for reproducibility
        seed = config.get("seed", 42)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Initialize model, optimizer, and loss function
        self._init_model()

        # Get training parameters from config with defaults
        training_config = config.get("training", {})

        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=training_config.get("learning_rate", 2e-5),
            betas=training_config.get("betas", (0.9, 0.999)),
            eps=training_config.get("eps", 1e-8),
            weight_decay=training_config.get("weight_decay", 0.01),
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config["training"].get("lr_step_size", 1),
            gamma=config["training"].get("lr_gamma", 0.95),
        )

        # Loss function (ignoring padding index)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

        # Create output directory
        self.output_dir = Path(config["training"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=self.output_dir / "logs")

        # Training state
        self.best_val_loss = float("inf")
        self.epoch = 0
        self.step = 0

    def _init_model(self):
        """Initialize the model with support for enhanced configurations."""
        model_config = self.config["model"]

        # Initialize model with configurable architecture
        model_args = {
            "src_vocab_size": model_config.get("src_vocab_size", 50000),
            "tgt_vocab_size": model_config.get("tgt_vocab_size", 50000),
            "d_model": model_config.get("d_model", 512),
            "nhead": model_config.get("nhead", 8),
            "num_encoder_layers": model_config.get("num_encoder_layers", 6),
            "num_decoder_layers": model_config.get("num_decoder_layers", 6),
            "dim_feedforward": model_config.get("dim_feedforward", 2048),
            "dropout": model_config.get("dropout", 0.1),
            "max_seq_length": model_config.get("max_seq_length", 128),
            "pad_token_id": self.tokenizer.pad_token_id if self.tokenizer else 0,
            "bos_token_id": self.tokenizer.bos_token_id if self.tokenizer else 1,
            "eos_token_id": self.tokenizer.eos_token_id if self.tokenizer else 2,
        }

        # Initialize the model
        self.model = TransformerModel(**model_args).to(self.device)

        # Resize token embeddings if using a pretrained tokenizer
        if self.tokenizer and hasattr(self.model, "resize_token_embeddings"):
            self.model.resize_token_embeddings(len(self.tokenizer))

        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model initialized with {total_params:,} trainable parameters")
        print(f"Model architecture: {self.model.__class__.__name__}")
        print(f"Using device: {self.device}")

        # Enable gradient checkpointing if specified
        if self.config["training"].get("gradient_checkpointing", False):
            self.model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled")

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        start_time = time.time()

        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            src = batch["src"].to(self.device)
            tgt_input = batch["tgt_input"].to(self.device)
            tgt_output = batch["tgt_output"].to(self.device)
            src_padding_mask = batch["src_padding_mask"].to(self.device)
            tgt_padding_mask = batch["tgt_padding_mask"].to(self.device)

            # Create target mask for decoder
            tgt_mask = self.model.generate_square_subsequent_mask(tgt_input.size(1)).to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(
                src=src,
                tgt=tgt_input,
                tgt_mask=tgt_mask,
                src_key_padding_mask=src_padding_mask,
                tgt_key_padding_mask=tgt_padding_mask,
            )

            # Calculate loss
            # Resh output to (batch_size * seq_len, vocab_size)
            # and target to (batch_size * seq_len)
            output = output.view(-1, output.size(-1))
            tgt_output = tgt_output.view(-1)
            loss = self.criterion(output, tgt_output)

            # Backward pass and optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Update learning rate
            self.scheduler.step()

            # Update statistics
            total_loss += loss.item()
            self.step += 1

            # Log training progress
            if batch_idx % self.config["training"].get("log_interval", 100) == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                ms_per_batch = (time.time() - start_time) * 1000 / (batch_idx + 1)
                print(
                    f"Epoch: {self.epoch} | Batch: {batch_idx} | LR: {lr:.6f} | "
                    f"Loss: {loss.item():.4f} | ms/batch: {ms_per_batch:.1f}"
                )

                # Log to TensorBoard
                self.writer.add_scalar("train/loss", loss.item(), self.step)
                self.writer.add_scalar("train/lr", lr, self.step)

        # Calculate average loss for the epoch
        avg_loss = total_loss / len(train_loader)
        return avg_loss

    def validate(self, val_loader: DataLoader) -> float:
        """Validate the model on the validation set."""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                src = batch["src"].to(self.device)
                tgt_input = batch["tgt_input"].to(self.device)
                tgt_output = batch["tgt_output"].to(self.device)
                src_padding_mask = batch["src_padding_mask"].to(self.device)
                tgt_padding_mask = batch["tgt_padding_mask"].to(self.device)

                # Create target mask for decoder
                tgt_mask = self.model.generate_square_subsequent_mask(tgt_input.size(1)).to(
                    self.device
                )

                # Forward pass
                output = self.model(
                    src=src,
                    tgt=tgt_input,
                    tgt_mask=tgt_mask,
                    src_key_padding_mask=src_padding_mask,
                    tgt_key_padding_mask=tgt_padding_mask,
                )

                # Calculate loss
                output = output.view(-1, output.size(-1))
                tgt_output = tgt_output.view(-1)
                loss = self.criterion(output, tgt_output)

                total_loss += loss.item()

        # Calculate average loss
        avg_loss = total_loss / len(val_loader)
        return avg_loss

    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.epoch,
            "step": self.step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }

        # Save latest checkpoint
        checkpoint_path = self.output_dir / "checkpoint_latest.pt"
        torch.save(checkpoint, checkpoint_path)

        # Save best checkpoint
        if is_best:
            best_path = self.output_dir / "checkpoint_best.pt"
            torch.save(checkpoint, best_path)

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"No checkpoint found at '{checkpoint_path}'")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model state
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Load training state
        self.epoch = checkpoint["epoch"]
        self.step = checkpoint["step"]
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        print(f"Loaded checkpoint from '{checkpoint_path}' (epoch {self.epoch}, step {self.step})")

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Train the model."""
        # Load checkpoint if resuming training
        if self.config["training"].get("resume"):
            self.load_checkpoint(self.config["training"]["resume"])

        # Training loop
        for epoch in range(self.epoch, self.config["training"]["num_epochs"]):
            self.epoch = epoch

            # Train for one epoch
            print(f"\nEpoch {epoch + 1}/{self.config['training']['num_epochs']}")
            train_loss = self.train_epoch(train_loader)

            # Validate
            val_loss = self.validate(val_loader)
            print(f"Epoch: {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            # Log validation loss to TensorBoard
            self.writer.add_scalar("val/loss", val_loss, self.epoch)

            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                print(f"New best validation loss: {val_loss:.4f}")

            self.save_checkpoint(is_best=is_best)

            # Save model every few epochs
            if (epoch + 1) % self.config["training"].get("save_interval", 1) == 0:
                self.save_checkpoint()

        # Close TensorBoard writer
        self.writer.close()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    """Main training function with support for enhanced datasets."""
    parser = argparse.ArgumentParser(description="Train Bangla-English translation model")
    parser.add_argument(
        "--config", type=str, default="config/train.yaml", help="Path to configuration file"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume training from"
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize tokenizer
    print("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"].get("tokenizer_name", "sagorsarker/bangla-bert-base"),
        use_fast=True,
        add_special_tokens=True,
        max_length=config["model"].get("max_seq_length", 128),
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    # Add special tokens for code-switching if not already present
    special_tokens_dict = {"additional_special_tokens": ["<bn>", "</bn>", "<en>", "</en>"]}
    tokenizer.add_special_tokens(special_tokens_dict)

    # Create datasets
    data_config = config["data"]
    max_length = config["model"].get("max_seq_length", 128)

    # Training dataset with enhanced data sources
    train_sources = []
    if "enhanced_data" in data_config:
        train_sources.extend(data_config["enhanced_data"])
    if "train_data" in data_config:  # For backward compatibility
        train_sources.append({"path": data_config["train_data"], "type": "parquet", "weight": 1.0})

    print("Loading training dataset...")
    train_dataset = TranslationDataset(
        data_sources=train_sources, max_length=max_length, is_training=True
    )

    # Validation dataset (use standard validation data)
    val_sources = [
        {
            "path": data_config.get("val_data", data_config["train_data"]),
            "type": "parquet",
            "weight": 1.0,
        }
    ]

    print("Loading validation dataset...")
    val_dataset = TranslationDataset(
        data_sources=val_sources,
        max_length=max_length,
        is_training=False,  # No augmentation for validation
    )

    # Set tokenizer for datasets
    train_dataset.tokenizer = tokenizer
    val_dataset.tokenizer = tokenizer

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config["training"].get("num_workers", 4),
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"].get("val_batch_size", config["training"]["batch_size"]),
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config["training"].get("num_workers", 4),
        pin_memory=True,
    )

    # Initialize trainer with tokenizer
    trainer = Trainer(config, tokenizer=tokenizer)

    # Log dataset statistics
    print("\n=== Dataset Statistics ===")
    print(f"Training samples: {len(train_dataset):,}")
    print(f"Validation samples: {len(val_dataset):,}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Number of batches per epoch: {len(train_loader)}")
    print(f"Number of epochs: {config['training'].get('num_epochs', 10)}")
    print("=" * 25 + "\n")

    # Start training
    print("Starting training...")
    try:
        trainer.train(train_loader, val_loader)
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving model...")
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise
    finally:
        # Always save the final model
        print("Saving final model...")
        trainer.save_checkpoint()

        # Close the TensorBoard writer
        if hasattr(trainer, "writer"):
            trainer.writer.close()


if __name__ == "__main__":
    main()
