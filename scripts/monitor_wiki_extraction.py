#!/usr/bin/env python3
"""
Monitor Wikipedia extraction and preprocessing progress.

Usage:
    python scripts/monitor_wiki_extraction.py
    python scripts/monitor_wiki_extraction.py --watch
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def count_files_recursive(directory: Path) -> int:
    """Count all files in directory recursively."""
    if not directory.exists():
        return 0
    return sum(1 for _ in directory.rglob("*") if _.is_file())


def get_directory_size(directory: Path) -> int:
    """Get total size of directory in bytes."""
    if not directory.exists():
        return 0
    return sum(f.stat().st_size for f in directory.rglob("*") if f.is_file())


def count_extracted_articles(extract_dir: Path) -> int:
    """Count extracted articles from WikiExtractor output."""
    if not extract_dir.exists():
        return 0

    article_count = 0
    for file_path in extract_dir.rglob("wiki_*"):
        if not file_path.is_file():
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    json.loads(line)
                    article_count += 1
                except json.JSONDecodeError:
                    continue

    return article_count


def count_processed_sentences(processed_dir: Path) -> Dict[str, int]:
    """Count sentences in processed train/val/test splits."""
    counts = {}

    for split in ["train", "val", "test"]:
        split_dir = processed_dir / split
        if not split_dir.exists():
            counts[split] = 0
            continue

        total_lines = 0
        for txt_file in split_dir.glob("*.txt"):
            with open(txt_file, "r", encoding="utf-8") as f:
                total_lines += sum(1 for line in f if line.strip())

        counts[split] = total_lines

    return counts


def format_size(size_bytes: int) -> str:
    """Format bytes to human-readable size."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def check_extraction_status(base_dir: Path) -> Dict:
    """Check status of Wikipedia extraction and preprocessing."""
    status = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "raw": {},
        "extracted": {},
        "processed": {},
    }

    # Check raw dumps
    raw_dir = base_dir / "raw"
    if raw_dir.exists():
        for dump_file in raw_dir.glob("*.xml.bz2"):
            status["raw"][dump_file.name] = {
                "size": format_size(dump_file.stat().st_size),
                "exists": True,
            }

    # Check extracted files
    processed_dir = base_dir / "processed"
    for extract_dir in processed_dir.glob("*_extracted"):
        lang = extract_dir.name.replace("_extracted", "")

        file_count = count_files_recursive(extract_dir)
        dir_size = get_directory_size(extract_dir)
        article_count = count_extracted_articles(extract_dir)

        status["extracted"][lang] = {
            "files": file_count,
            "size": format_size(dir_size),
            "articles": article_count,
        }

    # Check processed files
    if processed_dir.exists():
        sentence_counts = count_processed_sentences(processed_dir)
        total_sentences = sum(sentence_counts.values())

        status["processed"] = {
            "splits": sentence_counts,
            "total_sentences": total_sentences,
            "size": format_size(get_directory_size(processed_dir)),
        }

    return status


def print_status(status: Dict):
    """Pretty print extraction status."""
    print("\n" + "=" * 60)
    print(f"Wikipedia Extraction Status - {status['timestamp']}")
    print("=" * 60)

    # Raw dumps
    print("\n📦 Raw Dumps:")
    if status["raw"]:
        for name, info in status["raw"].items():
            print(f"  ✓ {name}: {info['size']}")
    else:
        print("  ⚠ No raw dumps found")

    # Extracted articles
    print("\n📄 Extracted Articles:")
    if status["extracted"]:
        for lang, info in status["extracted"].items():
            print(f"  ✓ {lang.upper()}:")
            print(f"    - Files: {info['files']:,}")
            print(f"    - Articles: {info['articles']:,}")
            print(f"    - Size: {info['size']}")
    else:
        print("  ⏳ Extraction in progress or not started...")

    # Processed sentences
    print("\n✨ Processed Data:")
    if status["processed"].get("total_sentences", 0) > 0:
        splits = status["processed"]["splits"]
        print(f"  ✓ Total Sentences: {status['processed']['total_sentences']:,}")
        print(f"  ✓ Train: {splits.get('train', 0):,}")
        print(f"  ✓ Val: {splits.get('val', 0):,}")
        print(f"  ✓ Test: {splits.get('test', 0):,}")
        print(f"  ✓ Size: {status['processed']['size']}")
    else:
        print("  ⏳ Processing not complete yet...")

    print("\n" + "=" * 60 + "\n")


def watch_progress(base_dir: Path, interval: int = 30):
    """Watch extraction progress in real-time."""
    logger.info(f"Watching extraction progress (updating every {interval}s)")
    logger.info("Press Ctrl+C to stop")

    try:
        while True:
            status = check_extraction_status(base_dir)
            print_status(status)
            time.sleep(interval)
    except KeyboardInterrupt:
        logger.info("\nStopped monitoring")


def main():
    parser = argparse.ArgumentParser(description="Monitor Wikipedia extraction progress")
    parser.add_argument(
        "--base-dir",
        type=str,
        default="datasets/wikipedia",
        help="Base Wikipedia dataset directory",
    )
    parser.add_argument("--watch", action="store_true", help="Watch progress in real-time")
    parser.add_argument(
        "--interval", type=int, default=30, help="Update interval in seconds (for watch mode)"
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()
    base_dir = Path(args.base_dir)

    if args.watch:
        watch_progress(base_dir, args.interval)
    else:
        status = check_extraction_status(base_dir)

        if args.json:
            print(json.dumps(status, indent=2))
        else:
            print_status(status)


if __name__ == "__main__":
    main()
