#!/usr/bin/env python3
"""
Prepare trained model for Hugging Face Hub upload.

This script:
1. Copies model files to a clean directory
2. Removes training artifacts
3. Validates model structure
4. Generates metadata

Usage:
    python scripts/huggingface/prepare_model.py --model models/wikipedia/base --output models/huggingface_ready/bn-wikipedia-lm
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ModelPreparer:
    """Prepare model for Hugging Face Hub."""

    REQUIRED_FILES = [
        "config.json",
        "pytorch_model.bin",  # or model.safetensors
    ]

    TOKENIZER_FILES = [
        "tokenizer_config.json",
        "vocab.txt",
        "special_tokens_map.json",
    ]

    OPTIONAL_FILES = [
        "merges.txt",  # For BPE tokenizers
        "tokenizer.json",  # Fast tokenizer
        "model.safetensors",  # SafeTensors format
    ]

    EXCLUDE_PATTERNS = [
        "checkpoint-*",
        "optimizer.pt",
        "scheduler.pt",
        "trainer_state.json",
        "training_args.bin",
        "runs/",
        "logs/",
        "*.log",
    ]

    def __init__(self, model_path: Path, output_path: Path):
        self.model_path = model_path
        self.output_path = output_path
        self.metadata = {}

    def validate_model(self) -> bool:
        """Validate that model has required files."""
        logger.info(f"Validating model at {self.model_path}")

        missing_files = []

        # Check for model file (either .bin or .safetensors)
        has_model = (self.model_path / "pytorch_model.bin").exists() or (
            self.model_path / "model.safetensors"
        ).exists()

        if not has_model:
            missing_files.append("pytorch_model.bin or model.safetensors")

        # Check config
        if not (self.model_path / "config.json").exists():
            missing_files.append("config.json")

        # Check tokenizer
        has_tokenizer = (self.model_path / "tokenizer_config.json").exists()
        if not has_tokenizer:
            logger.warning("tokenizer_config.json not found")

        if missing_files:
            logger.error(f"Missing required files: {missing_files}")
            return False

        logger.info("✓ Model validation passed")
        return True

    def copy_model_files(self) -> None:
        """Copy model files to output directory."""
        logger.info(f"Copying model files to {self.output_path}")

        self.output_path.mkdir(parents=True, exist_ok=True)

        # Copy all files except excluded patterns
        for file_path in self.model_path.rglob("*"):
            if file_path.is_file():
                # Check if file should be excluded
                relative_path = file_path.relative_to(self.model_path)

                should_exclude = False
                for pattern in self.EXCLUDE_PATTERNS:
                    if file_path.match(pattern) or relative_path.match(pattern):
                        should_exclude = True
                        break

                if should_exclude:
                    logger.debug(f"Excluding: {relative_path}")
                    continue

                # Copy file
                dest_path = self.output_path / relative_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file_path, dest_path)
                logger.debug(f"Copied: {relative_path}")

        logger.info("✓ Model files copied")

    def generate_metadata(self) -> Dict:
        """Generate metadata about the model."""
        logger.info("Generating metadata")

        # Load config
        config_path = self.output_path / "config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                config = json.load(f)
        else:
            config = {}

        # Get model size
        model_file = self.output_path / "pytorch_model.bin"
        if not model_file.exists():
            model_file = self.output_path / "model.safetensors"

        model_size_mb = model_file.stat().st_size / (1024 * 1024) if model_file.exists() else 0

        # Count parameters
        num_parameters = config.get("num_parameters", "unknown")
        if num_parameters == "unknown":
            # Estimate from config
            hidden_size = config.get("hidden_size", 768)
            num_layers = config.get("num_hidden_layers", 12)
            vocab_size = config.get("vocab_size", 30000)

            # Rough estimate for BERT-like models
            num_parameters = (
                vocab_size * hidden_size  # Embeddings
                + num_layers * (12 * hidden_size * hidden_size)  # Transformer layers
                + hidden_size * vocab_size  # Output layer
            )

        metadata = {
            "model_type": config.get("model_type", "unknown"),
            "architecture": (
                config.get("architectures", ["unknown"])[0]
                if config.get("architectures")
                else "unknown"
            ),
            "hidden_size": config.get("hidden_size", "unknown"),
            "num_layers": config.get("num_hidden_layers", "unknown"),
            "vocab_size": config.get("vocab_size", "unknown"),
            "num_parameters": num_parameters,
            "model_size_mb": round(model_size_mb, 2),
        }

        self.metadata = metadata

        # Save metadata
        metadata_path = self.output_path / "model_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"✓ Metadata generated: {metadata}")
        return metadata

    def create_gitattributes(self) -> None:
        """Create .gitattributes for LFS."""
        logger.info("Creating .gitattributes for Git LFS")

        gitattributes_content = """# Git LFS configuration for model files
*.bin filter=lfs diff=lfs merge=lfs -text
*.safetensors filter=lfs diff=lfs merge=lfs -text
*.h5 filter=lfs diff=lfs merge=lfs -text
*.ot filter=lfs diff=lfs merge=lfs -text
*.msgpack filter=lfs diff=lfs merge=lfs -text
*.arrow filter=lfs diff=lfs merge=lfs -text
"""

        gitattributes_path = self.output_path / ".gitattributes"
        with open(gitattributes_path, "w") as f:
            f.write(gitattributes_content)

        logger.info("✓ .gitattributes created")

    def prepare(self) -> bool:
        """Run full preparation pipeline."""
        logger.info("=" * 60)
        logger.info("Starting model preparation for Hugging Face Hub")
        logger.info("=" * 60)

        # Validate
        if not self.validate_model():
            logger.error("Model validation failed")
            return False

        # Copy files
        self.copy_model_files()

        # Generate metadata
        self.generate_metadata()

        # Create .gitattributes
        self.create_gitattributes()

        logger.info("=" * 60)
        logger.info("✓ Model preparation complete!")
        logger.info(f"Output: {self.output_path}")
        logger.info("=" * 60)

        return True


def main():
    parser = argparse.ArgumentParser(description="Prepare model for Hugging Face Hub")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model directory")
    parser.add_argument(
        "--output", type=str, required=True, help="Output directory for prepared model"
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite output directory if it exists"
    )

    args = parser.parse_args()

    model_path = Path(args.model)
    output_path = Path(args.output)

    # Check if model exists
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        sys.exit(1)

    # Check if output exists
    if output_path.exists() and not args.force:
        logger.error(f"Output directory already exists: {output_path}")
        logger.error("Use --force to overwrite")
        sys.exit(1)

    # Prepare model
    preparer = ModelPreparer(model_path, output_path)
    success = preparer.prepare()

    if not success:
        sys.exit(1)

    # Print next steps
    print("\n" + "=" * 60)
    print("Next steps:")
    print("=" * 60)
    print(f"1. Review prepared model: {output_path}")
    print(f"2. Create model card: {output_path}/README.md")
    print(f"3. Upload to Hugging Face:")
    print(
        f"   python scripts/huggingface/upload_model.py --model {output_path} --repo your-username/model-name"
    )
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
