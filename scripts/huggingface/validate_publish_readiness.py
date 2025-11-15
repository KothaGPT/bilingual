#!/usr/bin/env python3
"""
Validate that models are ready for Hugging Face publication.

Usage:
    python scripts/huggingface/validate_publish_readiness.py
    python scripts/huggingface/validate_publish_readiness.py --model models/bilingual-lm
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Required files for different model types
REQUIRED_FILES = {
    "causal-lm": ["config.json", "training_args.json"],
    "text-classification": ["config.json", "task_config.json"],
    "token-classification": ["config.json", "task_config.json"],
    "feature-extraction": ["config.json"],
}

# Model definitions
MODELS_TO_CHECK = {
    "bilingual-lm": {"type": "causal-lm", "path": "models/bilingual-lm"},
    "literary-lm": {"type": "causal-lm", "path": "models/literary-lm"},
    "readability-classifier": {
        "type": "text-classification",
        "path": "models/readability-classifier",
    },
    "poetic-meter-detector": {
        "type": "text-classification",
        "path": "models/poetic-meter-detector",
    },
    "metaphor-simile-detector": {
        "type": "text-classification",
        "path": "models/metaphor-simile-detector",
    },
    "style-transfer-gpt": {"type": "causal-lm", "path": "models/style-transfer-gpt"},
    "sentiment-tone-classifier": {
        "type": "text-classification",
        "path": "models/sentiment-tone-classifier",
    },
    "cross-lingual-embed": {"type": "feature-extraction", "path": "models/cross-lingual-embed"},
    "named-entity-recognizer": {
        "type": "token-classification",
        "path": "models/named-entity-recognizer",
    },
}


class ModelValidator:
    """Validate model readiness for Hugging Face."""

    def __init__(self, model_path: Path, model_type: str):
        self.model_path = model_path
        self.model_type = model_type
        self.issues: List[str] = []
        self.warnings: List[str] = []
        self.info: List[str] = []

    def validate(self) -> bool:
        """Run all validation checks."""
        print(f"\n{'='*60}")
        print(f"Validating: {self.model_path.name}")
        print(f"Type: {self.model_type}")
        print(f"{'='*60}\n")

        # Check if directory exists
        if not self.model_path.exists():
            self.issues.append(f"Model directory not found: {self.model_path}")
            return False

        if not self.model_path.is_dir():
            self.issues.append(f"Path is not a directory: {self.model_path}")
            return False

        # Run validation checks
        self._check_required_files()
        self._check_config_files()
        self._check_model_files()
        self._check_tokenizer_files()
        self._check_readme()
        self._check_file_sizes()

        # Print results
        self._print_results()

        return len(self.issues) == 0

    def _check_required_files(self):
        """Check for required configuration files."""
        required = REQUIRED_FILES.get(self.model_type, ["config.json"])

        for filename in required:
            filepath = self.model_path / filename
            if filepath.exists():
                self.info.append(f"✓ Found {filename}")
            else:
                self.issues.append(f"✗ Missing required file: {filename}")

    def _check_config_files(self):
        """Validate configuration files."""
        config_file = self.model_path / "config.json"

        if config_file.exists():
            try:
                with open(config_file, "r") as f:
                    config = json.load(f)

                # Check for essential config keys
                essential_keys = ["model_type"]
                for key in essential_keys:
                    if key in config:
                        self.info.append(f"✓ Config has '{key}'")
                    else:
                        self.warnings.append(f"⚠ Config missing '{key}'")

            except json.JSONDecodeError as e:
                self.issues.append(f"✗ Invalid JSON in config.json: {e}")
            except Exception as e:
                self.warnings.append(f"⚠ Could not read config.json: {e}")

    def _check_model_files(self):
        """Check for model weight files."""
        model_files = [
            "pytorch_model.bin",
            "model.safetensors",
            "tf_model.h5",
            "flax_model.msgpack",
        ]

        found = False
        for filename in model_files:
            if (self.model_path / filename).exists():
                self.info.append(f"✓ Found model weights: {filename}")
                found = True
                break

        if not found:
            self.warnings.append("⚠ No model weight files found (may be added during preparation)")

    def _check_tokenizer_files(self):
        """Check for tokenizer files."""
        tokenizer_files = [
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.txt",
            "vocab.json",
            "merges.txt",
            "special_tokens_map.json",
        ]

        found_files = []
        for filename in tokenizer_files:
            if (self.model_path / filename).exists():
                found_files.append(filename)

        if found_files:
            self.info.append(f"✓ Found tokenizer files: {', '.join(found_files)}")
        else:
            self.warnings.append("⚠ No tokenizer files found (may use external tokenizer)")

    def _check_readme(self):
        """Check for README/model card."""
        readme_file = self.model_path / "README.md"

        if readme_file.exists():
            self.info.append("✓ Found README.md")

            # Check README content
            try:
                with open(readme_file, "r", encoding="utf-8") as f:
                    content = f.read()

                if len(content) < 100:
                    self.warnings.append("⚠ README.md is very short")

                if "---" in content[:50]:
                    self.info.append("✓ README has YAML frontmatter")
                else:
                    self.warnings.append("⚠ README missing YAML frontmatter")

            except Exception as e:
                self.warnings.append(f"⚠ Could not read README.md: {e}")
        else:
            self.warnings.append("⚠ No README.md (will be generated)")

    def _check_file_sizes(self):
        """Check for large files that might cause upload issues."""
        large_files = []

        for file in self.model_path.rglob("*"):
            if file.is_file():
                size_mb = file.stat().st_size / (1024 * 1024)
                if size_mb > 100:
                    large_files.append((file.name, size_mb))

        if large_files:
            self.info.append(f"ℹ Found {len(large_files)} large files (>100MB)")
            for name, size in large_files[:3]:
                self.info.append(f"  - {name}: {size:.1f} MB")

    def _print_results(self):
        """Print validation results."""
        print("\n📋 Validation Results:\n")

        if self.info:
            print("ℹ Information:")
            for msg in self.info:
                print(f"  {msg}")
            print()

        if self.warnings:
            print("⚠ Warnings:")
            for msg in self.warnings:
                print(f"  {msg}")
            print()

        if self.issues:
            print("✗ Issues (must fix):")
            for msg in self.issues:
                print(f"  {msg}")
            print()

        if not self.issues:
            print("✅ Model is ready for publication!")
        else:
            print("❌ Model has issues that must be resolved")

        print()


def validate_all_models() -> Tuple[int, int, int]:
    """Validate all models."""
    ready = 0
    not_ready = 0
    missing = 0

    print("\n" + "=" * 60)
    print("KothaGPT Model Publication Readiness Check")
    print("=" * 60)

    for model_name, model_info in MODELS_TO_CHECK.items():
        model_path = Path(model_info["path"])
        model_type = model_info["type"]

        if not model_path.exists():
            print(f"\n⊘ {model_name}: Directory not found")
            missing += 1
            continue

        validator = ModelValidator(model_path, model_type)
        is_ready = validator.validate()

        if is_ready:
            ready += 1
        else:
            not_ready += 1

    return ready, not_ready, missing


def main():
    parser = argparse.ArgumentParser(description="Validate models for Hugging Face publication")
    parser.add_argument("--model", type=str, help="Specific model directory to validate")
    parser.add_argument(
        "--type",
        type=str,
        default="causal-lm",
        help="Model type (causal-lm, text-classification, etc.)",
    )

    args = parser.parse_args()

    if args.model:
        # Validate single model
        model_path = Path(args.model)
        validator = ModelValidator(model_path, args.type)
        is_ready = validator.validate()
        sys.exit(0 if is_ready else 1)
    else:
        # Validate all models
        ready, not_ready, missing = validate_all_models()

        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        print(f"✅ Ready for publication: {ready}")
        print(f"❌ Not ready: {not_ready}")
        print(f"⊘ Missing: {missing}")
        print(f"📊 Total: {ready + not_ready + missing}")
        print("=" * 60 + "\n")

        if not_ready > 0:
            print("⚠ Some models need attention before publication")
            sys.exit(1)
        elif ready == 0:
            print("⚠ No models are ready for publication")
            sys.exit(1)
        else:
            print("✅ All available models are ready!")
            sys.exit(0)


if __name__ == "__main__":
    main()
