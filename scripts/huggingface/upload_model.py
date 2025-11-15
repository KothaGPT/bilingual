#!/usr/bin/env python3
"""
Upload model to Hugging Face Hub.

Usage:
    python scripts/huggingface/upload_model.py --model models/huggingface_ready/bn-wikipedia-lm --repo your-username/bn-wikipedia-lm
    python scripts/huggingface/upload_model.py --model models/huggingface_ready/bn-wikipedia-lm --repo your-username/bn-wikipedia-lm --private
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

try:
    from huggingface_hub import HfApi, create_repo, upload_folder
    from huggingface_hub.utils import RepositoryNotFoundError

    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    logging.error("huggingface_hub not installed. Install with: pip install huggingface_hub")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ModelUploader:
    """Upload model to Hugging Face Hub."""

    def __init__(
        self,
        model_path: Path,
        repo_id: str,
        private: bool = False,
        token: Optional[str] = None,
    ):
        self.model_path = model_path
        self.repo_id = repo_id
        self.private = private
        self.token = token
        self.api = HfApi(token=token)

    def validate_model(self) -> bool:
        """Validate model directory."""
        logger.info(f"Validating model at {self.model_path}")

        # Check required files
        required_files = ["config.json"]
        model_file = (self.model_path / "pytorch_model.bin").exists() or (
            self.model_path / "model.safetensors"
        ).exists()

        if not model_file:
            logger.error("No model file found (pytorch_model.bin or model.safetensors)")
            return False

        for file_name in required_files:
            if not (self.model_path / file_name).exists():
                logger.error(f"Required file not found: {file_name}")
                return False

        # Check for README
        if not (self.model_path / "README.md").exists():
            logger.warning("README.md not found. Consider adding a model card.")

        logger.info("✓ Model validation passed")
        return True

    def create_repository(self) -> bool:
        """Create repository on Hugging Face Hub."""
        logger.info(f"Creating repository: {self.repo_id}")

        try:
            # Check if repo exists
            try:
                self.api.repo_info(repo_id=self.repo_id, repo_type="model")
                logger.info(f"Repository already exists: {self.repo_id}")
                return True
            except RepositoryNotFoundError:
                pass

            # Create repo
            create_repo(
                repo_id=self.repo_id,
                repo_type="model",
                private=self.private,
                token=self.token,
            )

            logger.info(f"✓ Repository created: {self.repo_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to create repository: {e}")
            return False

    def upload(self, commit_message: str = "Upload model") -> bool:
        """Upload model to Hugging Face Hub."""
        logger.info(f"Uploading model to {self.repo_id}")

        try:
            # Upload folder
            upload_folder(
                folder_path=str(self.model_path),
                repo_id=self.repo_id,
                repo_type="model",
                commit_message=commit_message,
                token=self.token,
            )

            logger.info("✓ Model uploaded successfully!")
            return True

        except Exception as e:
            logger.error(f"Upload failed: {e}")
            return False

    def run(self, commit_message: str = "Upload model") -> bool:
        """Run full upload pipeline."""
        logger.info("=" * 60)
        logger.info("Starting model upload to Hugging Face Hub")
        logger.info("=" * 60)

        # Validate
        if not self.validate_model():
            return False

        # Create repo
        if not self.create_repository():
            return False

        # Upload
        if not self.upload(commit_message):
            return False

        logger.info("=" * 60)
        logger.info("✓ Upload complete!")
        logger.info(f"Model URL: https://huggingface.co/{self.repo_id}")
        logger.info("=" * 60)

        return True


def main():
    if not HF_HUB_AVAILABLE:
        logger.error("Please install huggingface_hub: pip install huggingface_hub")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Upload model to Hugging Face Hub")
    parser.add_argument("--model", type=str, required=True, help="Path to prepared model directory")
    parser.add_argument(
        "--repo", type=str, required=True, help="Hugging Face repository ID (username/model-name)"
    )
    parser.add_argument("--private", action="store_true", help="Make repository private")
    parser.add_argument(
        "--token", type=str, help="Hugging Face API token (or use huggingface-cli login)"
    )
    parser.add_argument("--message", type=str, default="Upload model", help="Commit message")

    args = parser.parse_args()

    model_path = Path(args.model)

    # Check if model exists
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        sys.exit(1)

    # Upload
    uploader = ModelUploader(
        model_path=model_path,
        repo_id=args.repo,
        private=args.private,
        token=args.token,
    )

    success = uploader.run(commit_message=args.message)

    if not success:
        sys.exit(1)

    # Print next steps
    print("\n" + "=" * 60)
    print("Next steps:")
    print("=" * 60)
    print(f"1. View model: https://huggingface.co/{args.repo}")
    print(f"2. Test inference:")
    print(f"   python scripts/huggingface/test_hf_model.py --repo {args.repo}")
    print(f"3. Create demo:")
    print(f"   python scripts/huggingface/create_demo.py --repo {args.repo}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
