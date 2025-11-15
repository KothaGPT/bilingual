#!/usr/bin/env python3
"""
Add Hugging Face badges to README files.

Usage:
    python scripts/huggingface/add_hf_badges.py
    python scripts/huggingface/add_hf_badges.py --username KothaGPT
"""

import argparse
from pathlib import Path
from typing import Dict, List

# Model definitions
MODELS = [
    {"name": "bilingual-language-model", "display": "Bilingual LM", "type": "model"},
    {"name": "literary-language-model", "display": "Literary LM", "type": "model"},
    {"name": "readability-classifier", "display": "Readability Classifier", "type": "model"},
    {"name": "poetic-meter-detector", "display": "Poetic Meter Detector", "type": "model"},
    {"name": "metaphor-simile-detector", "display": "Metaphor-Simile Detector", "type": "model"},
    {"name": "style-transfer-gpt", "display": "Style Transfer GPT", "type": "model"},
    {"name": "sentiment-tone-classifier", "display": "Sentiment-Tone Classifier", "type": "model"},
    {"name": "cross-lingual-embeddings", "display": "Cross-lingual Embeddings", "type": "model"},
    {"name": "named-entity-recognizer", "display": "Named Entity Recognizer", "type": "model"},
]

DATASET = {"name": "bilingual-corpus", "display": "Bilingual Corpus", "type": "dataset"}


def generate_badge(username: str, name: str, display: str, badge_type: str = "model") -> str:
    """Generate a Hugging Face badge markdown."""
    if badge_type == "model":
        url = f"https://huggingface.co/{username}/{name}"
        badge_url = f"https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-{display.replace(' ', '%20')}-blue"
    else:  # dataset
        url = f"https://huggingface.co/datasets/{username}/{name}"
        badge_url = f"https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-{display.replace(' ', '%20')}-yellow"

    return f"[![{display}]({badge_url})]({url})"


def generate_badges_section(username: str) -> str:
    """Generate complete badges section."""
    lines = ["## 🤗 Hugging Face Models", "", "### Language Models", ""]

    # Language models
    for model in MODELS[:2]:
        badge = generate_badge(username, model["name"], model["display"], "model")
        lines.append(f"- {badge}")

    lines.extend(["", "### Classification Models", ""])

    # Classification models
    for model in MODELS[2:7]:
        badge = generate_badge(username, model["name"], model["display"], "model")
        lines.append(f"- {badge}")

    lines.extend(["", "### Embedding & NER Models", ""])

    # Embedding and NER models
    for model in MODELS[7:]:
        badge = generate_badge(username, model["name"], model["display"], "model")
        lines.append(f"- {badge}")

    lines.extend(["", "### Dataset", ""])

    # Dataset
    dataset_badge = generate_badge(username, DATASET["name"], DATASET["display"], "dataset")
    lines.append(f"- {dataset_badge}")

    lines.extend(["", "---", ""])

    return "\n".join(lines)


def update_readme(readme_path: Path, username: str, dry_run: bool = False):
    """Update README with Hugging Face badges."""
    if not readme_path.exists():
        print(f"⚠ README not found: {readme_path}")
        return

    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()

    badges_section = generate_badges_section(username)

    # Check if badges section already exists
    if "## 🤗 Hugging Face Models" in content:
        print(f"ℹ Badges section already exists in {readme_path}")
        # Replace existing section
        start_marker = "## 🤗 Hugging Face Models"
        end_marker = "---"

        start_idx = content.find(start_marker)
        if start_idx != -1:
            # Find the end marker after the start
            end_idx = content.find(end_marker, start_idx + len(start_marker))
            if end_idx != -1:
                # Include the end marker and newlines
                end_idx = content.find("\n", end_idx) + 1
                new_content = content[:start_idx] + badges_section + content[end_idx:]
            else:
                print(f"⚠ Could not find end marker in {readme_path}")
                return
        else:
            print(f"⚠ Could not find start marker in {readme_path}")
            return
    else:
        # Add badges section after the main title
        lines = content.split("\n")
        insert_idx = 0

        # Find first heading
        for i, line in enumerate(lines):
            if line.startswith("# "):
                insert_idx = i + 1
                # Skip any existing badges or shields
                while insert_idx < len(lines) and (
                    lines[insert_idx].startswith("[![") or lines[insert_idx].strip() == ""
                ):
                    insert_idx += 1
                break

        lines.insert(insert_idx, "\n" + badges_section)
        new_content = "\n".join(lines)

    if dry_run:
        print(f"🔍 DRY RUN - Would update {readme_path}")
        print("\nBadges section:")
        print(badges_section)
    else:
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        print(f"✅ Updated {readme_path}")


def main():
    parser = argparse.ArgumentParser(description="Add Hugging Face badges to README files")
    parser.add_argument(
        "--username", type=str, default="KothaGPT", help="Hugging Face username/organization"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be done without making changes"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Adding Hugging Face Badges")
    print("=" * 60)
    print(f"Username: {args.username}")
    print(f"Dry Run: {args.dry_run}")
    print()

    # Update main README
    readme_files = [
        Path("README.md"),
        Path("README_HUGGINGFACE.md"),
    ]

    for readme_path in readme_files:
        update_readme(readme_path, args.username, args.dry_run)

    print()
    print("=" * 60)
    print("✅ Badge update complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
