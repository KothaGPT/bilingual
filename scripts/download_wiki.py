#!/usr/bin/env python3
"""
Download Wikipedia dumps for Bangla and optionally English.

Usage:
    python scripts/download_wiki.py --lang bn --output datasets/wikipedia/raw
    python scripts/download_wiki.py --lang bn,en --bilingual --output datasets/wikipedia/raw
"""

import argparse
import hashlib
import logging
import os
import sys
import ssl
import urllib.request
from pathlib import Path
from typing import List

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


WIKIPEDIA_DUMPS = {
    "bn": "https://dumps.wikimedia.org/bnwiki/latest/bnwiki-latest-pages-articles.xml.bz2",
    "en": "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2",
}


def download_file(url: str, output_path: Path, chunk_size: int = 8192) -> None:
    """
    Download a file with progress reporting.

    Args:
        url: URL to download from
        output_path: Path to save the file
        chunk_size: Size of chunks to download
    """
    logger.info(f"Downloading {url}")
    logger.info(f"Saving to {output_path}")

    try:
        with urllib.request.urlopen(url, context=ssl._create_unverified_context()) as response:
            total_size = int(response.headers.get("Content-Length", 0))
            downloaded = 0

            with open(output_path, "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        logger.info(f"Progress: {progress:.2f}% ({downloaded}/{total_size} bytes)")

        logger.info(f"Download complete: {output_path}")

    except Exception as e:
        logger.error(f"Download failed: {e}")
        if output_path.exists():
            output_path.unlink()
        raise


def verify_checksum(file_path: Path, expected_checksum: str = None) -> str:
    """
    Compute and optionally verify MD5 checksum of downloaded file.

    Args:
        file_path: Path to file
        expected_checksum: Expected MD5 checksum (optional)

    Returns:
        Computed checksum
    """
    logger.info(f"Computing checksum for {file_path}")

    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)

    checksum = md5.hexdigest()
    logger.info(f"Checksum: {checksum}")

    if expected_checksum and checksum != expected_checksum:
        raise ValueError(f"Checksum mismatch! Expected {expected_checksum}, got {checksum}")

    return checksum


def download_wikipedia_dumps(languages: List[str], output_dir: Path, verify: bool = False) -> None:
    """
    Download Wikipedia dumps for specified languages.

    Args:
        languages: List of language codes (e.g., ['bn', 'en'])
        output_dir: Directory to save dumps
        verify: Whether to verify checksums
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for lang in languages:
        if lang not in WIKIPEDIA_DUMPS:
            logger.warning(f"Language '{lang}' not supported. Skipping.")
            continue

        url = WIKIPEDIA_DUMPS[lang]
        filename = f"{lang}wiki-latest-pages-articles.xml.bz2"
        output_path = output_dir / filename

        # Skip if already downloaded
        if output_path.exists():
            logger.info(f"File already exists: {output_path}")
            response = input(f"Re-download? (y/n): ").strip().lower()
            if response != "y":
                logger.info(f"Skipping {lang}")
                continue

        # Download
        download_file(url, output_path)

        # Verify checksum if requested
        if verify:
            verify_checksum(output_path)

        logger.info(f"Successfully downloaded {lang} Wikipedia dump")


def main():
    parser = argparse.ArgumentParser(description="Download Wikipedia dumps for training")
    parser.add_argument(
        "--lang", type=str, default="bn", help="Comma-separated language codes (e.g., bn,en)"
    )
    parser.add_argument(
        "--output", type=str, default="datasets/wikipedia/raw", help="Output directory for dumps"
    )
    parser.add_argument("--verify", action="store_true", help="Verify checksums after download")
    parser.add_argument(
        "--bilingual",
        action="store_true",
        help="Download both Bangla and English for bilingual training",
    )

    args = parser.parse_args()

    # Parse languages
    if args.bilingual:
        languages = ["bn", "en"]
    else:
        languages = [lang.strip() for lang in args.lang.split(",")]

    output_dir = Path(args.output)

    logger.info(f"Languages to download: {languages}")
    logger.info(f"Output directory: {output_dir}")

    try:
        download_wikipedia_dumps(languages, output_dir, args.verify)
        logger.info("All downloads completed successfully!")

    except Exception as e:
        logger.error(f"Download failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
