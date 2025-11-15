#!/usr/bin/env python3
"""
Test script for the evaluation functionality.

This script creates a small test dataset, trains a tiny model,
and runs evaluation to verify the evaluation script works correctly.
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from bilingual.models.transformer import TransformerModel

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class TestDataset(Dataset):
    """A tiny test dataset for verification."""

    def __init__(self, num_examples=100, max_length=10):
        self.num_examples = num_examples
        self.max_length = max_length
        self.vocab_size = 50
        self.pad_idx = 0
        self.bos_idx = 2
        self.eos_idx = 3

        # Generate some simple parallel data
        self.data = []
        for _ in range(num_examples):
            # Generate random sequences
            src_len = np.random.randint(3, max_length - 2)
            tgt_len = min(src_len + np.random.randint(-1, 2), max_length - 2)

            src = (
                [self.bos_idx]
                + [np.random.randint(4, self.vocab_size - 1) for _ in range(src_len)]
                + [self.eos_idx]
            )
            tgt = (
                [self.bos_idx]
                + [np.random.randint(4, self.vocab_size - 1) for _ in range(tgt_len)]
                + [self.eos_idx]
            )

            self.data.append(
                {
                    "src": torch.tensor(src, dtype=torch.long),
                    "tgt": torch.tensor(tgt, dtype=torch.long),
                }
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "src": item["src"],
            "tgt_input": item["tgt"][:-1],
            "tgt_output": item["tgt"][1:],
            "src_text": " ".join(map(str, item["src"].tolist())),
            "tgt_text": " ".join(map(str, item["tgt"].tolist())),
        }


def collate_fn(batch):
    """Collate function for the test dataset."""
    src = [item["src"] for item in batch]
    tgt_input = [item["tgt_input"] for item in batch]
    tgt_output = [item["tgt_output"] for item in batch]
    src_text = [item["src_text"] for item in batch]
    tgt_text = [item["tgt_text"] for item in batch]

    # Pad sequences
    src = torch.nn.utils.rnn.pad_sequence(src, batch_first=True, padding_value=0)
    tgt_input = torch.nn.utils.rnn.pad_sequence(tgt_input, batch_first=True, padding_value=0)
    tgt_output = torch.nn.utils.rnn.pad_sequence(tgt_output, batch_first=True, padding_value=0)

    # Create padding masks
    src_padding_mask = src == 0
    tgt_padding_mask = tgt_input == 0

    return {
        "src": src,
        "tgt_input": tgt_input,
        "tgt_output": tgt_output,
        "src_padding_mask": src_padding_mask,
        "tgt_padding_mask": tgt_padding_mask,
        "src_texts": src_text,
        "tgt_texts": tgt_text,
    }


def train_tiny_model():
    """Train a tiny model for testing purposes."""
    # Create test dataset and dataloader
    train_dataset = TestDataset(num_examples=1000, max_length=10)
    test_dataset = TestDataset(num_examples=100, max_length=10)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn)

    # Create a tiny model with smaller dimensions for testing
    model = TransformerModel(
        src_vocab_size=50,
        tgt_vocab_size=50,
        d_model=32,  # Smaller model for faster testing
        nhead=2,  # Reduce number of attention heads
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=64,  # Smaller feedforward dimension
        dropout=0.0,  # Disable dropout for more stable training in tests
        max_seq_length=32,  # Increased to handle test cases
        pad_idx=0,
    )

    # Set up optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # Train for a few steps
    print("Training tiny model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()

    # Move test loader to device
    def move_to_device(batch, device):
        return {
            "src": batch["src"].to(device),
            "tgt_input": batch["tgt_input"].to(device),
            "tgt_output": batch["tgt_output"].to(device),
            "src_padding_mask": batch["src_padding_mask"].to(device),
            "tgt_padding_mask": batch["tgt_padding_mask"].to(device),
            "src_texts": batch["src_texts"],
            "tgt_texts": batch["tgt_texts"],
        }

    for epoch in range(3):
        total_loss = 0
        for batch in train_loader:
            # Move batch to device
            batch = move_to_device(batch, device)
            src = batch["src"]
            tgt_input = batch["tgt_input"]
            tgt_output = batch["tgt_output"]
            src_padding_mask = batch["src_padding_mask"]
            tgt_padding_mask = batch["tgt_padding_mask"]

            # Forward pass
            optimizer.zero_grad()
            output = model(
                src=src,
                tgt=tgt_input,
                src_key_padding_mask=src_padding_mask,
                tgt_key_padding_mask=tgt_padding_mask,
            )

            # Calculate loss
            loss = criterion(output.view(-1, output.size(-1)), tgt_output.view(-1))

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

    # Save the model
    os.makedirs("test_models", exist_ok=True)
    model_path = "test_models/tiny_transformer.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "src_vocab_size": 50,
                "tgt_vocab_size": 50,
                "d_model": 64,
                "nhead": 4,
                "num_encoder_layers": 2,
                "num_decoder_layers": 2,
                "dim_feedforward": 128,
                "dropout": 0.1,
                "max_seq_length": 20,
            },
        },
        model_path,
    )

    print(f"Model saved to {model_path}")
    return model_path, test_loader


def test_evaluation():
    """Test the evaluation script with the tiny model."""
    # First, train a tiny model
    model_path, test_loader = train_tiny_model()

    # Now test the evaluation script
    print("\nTesting evaluation script...")
    from evaluate import Translator

    # Create a temporary config file matching our test model
    config = {
        "model": {
            "src_vocab_size": 50,
            "tgt_vocab_size": 50,
            "d_model": 32,
            "nhead": 2,
            "num_encoder_layers": 2,
            "num_decoder_layers": 2,
            "dim_feedforward": 64,
            "dropout": 0.0,  # Match training config
            "max_seq_length": 32,  # Match max_seq_length from model
        }
    }

    import yaml

    config_path = "test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    try:
        # Initialize translator
        translator = Translator(config_path, model_path)

        # Test single translation
        print("\nTesting single translation:")
        test_sentence = "2 4 6 8 10 3"  # [BOS] 4 6 8 [EOS]
        translation = translator.translate(test_sentence)
        print(f"Input: {test_sentence}")
        print(f"Output: {translation}")

        # Test batch evaluation
        print("\nRunning batch evaluation...")
        metrics = translator.evaluate_on_test_set(
            test_loader,
            output_file="test_results.json",
            max_examples=20,  # Only evaluate on 20 examples for testing
            beam_size=3,
        )

        print("\nEvaluation metrics:")
        print(f"  Loss: {metrics['loss']:.4f}")
        print(f"  BLEU: {metrics['bleu']:.2f}")
        print(f"  ROUGE-1 F1: {metrics['rouge']['rouge-1']['f']:.4f}")
        print(f"  ROUGE-2 F1: {metrics['rouge']['rouge-2']['f']:.4f}")
        print(f"  ROUGE-L F1: {metrics['rouge']['rouge-l']['f']:.4f}")

        print("\nEvaluation test completed successfully!")

    finally:
        # Clean up
        if os.path.exists(config_path):
            os.remove(config_path)


if __name__ == "__main__":
    test_evaluation()
