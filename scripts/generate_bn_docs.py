#!/usr/bin/env python3
"""
Script to generate Bengali versions of all documentation files.

This script:
1. Finds all .md files in the docs directory
2. Creates .bn.md versions for each
3. Adds a translation notice and original link
4. Can be extended with actual translation API calls
"""

import os
import re
from datetime import datetime
from pathlib import Path

# Configuration
ROOT_DIR = Path(__file__).parent.parent
DOCS_DIR = ROOT_DIR / "docs"
SKIP_DIRS = {"bn", "en", "api", "examples", "guides", "project"}
SKIP_FILES = {
    "index.bn.md",
    "README.bn.md",
    "getting-started.bn.md",
    "quickstart.bn.md",
    "quickstart_bn.md",
    "*.bn.md",
}


def should_skip(path: Path) -> bool:
    """Check if file should be skipped."""
    # Skip non-markdown files
    if path.suffix != ".md" or path.name.endswith(".bn.md"):
        return True

    # Skip files in skip directories
    for part in path.parts:
        if part in SKIP_DIRS and part != "docs":
            return True

    # Skip specific files
    if path.name in SKIP_FILES:
        return True

    return False


def generate_bengali_content(english_content: str, en_path: Path) -> str:
    """Generate Bengali version of the content with translation notice."""
    # This is a placeholder - in a real scenario, you would use a translation API
    # For now, we'll just add a translation notice

    # Get relative path for the original file
    rel_path = en_path.relative_to(ROOT_DIR)

    # Add translation notice header
    notice = (
        "---\n"
        f"original: /{rel_path}\n"
        f"translated: {datetime.now().strftime('%Y-%m-%d')}\n"
        "---\n\n"
        "> **বিঃদ্রঃ** এটি একটি স্বয়ংক্রিয়ভাবে অনুবাদকৃত নথি। মূল ইংরেজি সংস্করণের জন্য [এখানে ক্লিক করুন](/{rel_path}) করুন।\n\n"
        "---\n\n"
    )

    # For now, just return the notice with the original content
    # In a real implementation, you would call a translation API here
    return notice + english_content


def generate_bengali_docs():
    """Generate Bengali versions of all documentation files."""
    translated_count = 0

    for en_path in DOCS_DIR.rglob("*.md"):
        if should_skip(en_path):
            continue

        # Create target path with .bn.md extension
        bn_path = en_path.with_name(f"{en_path.stem}.bn.md")

        # Skip if Bengali version already exists
        if bn_path.exists():
            print(f"Skipping (exists): {bn_path}")
            continue

        print(f"Generating: {bn_path}")

        try:
            # Read English content
            with open(en_path, "r", encoding="utf-8") as f:
                en_content = f.read()

            # Generate Bengali content
            bn_content = generate_bengali_content(en_content, en_path)

            # Write Bengali version
            bn_path.parent.mkdir(parents=True, exist_ok=True)
            with open(bn_path, "w", encoding="utf-8") as f:
                f.write(bn_content)

            translated_count += 1

        except Exception as e:
            print(f"Error processing {en_path}: {str(e)}")

    print(f"\nGenerated {translated_count} Bengali documentation files.")
    print("Note: This script only creates placeholders. To get actual translations,")
    print("you'll need to integrate with a translation API or perform manual translation.")


if __name__ == "__main__":
    generate_bengali_docs()
