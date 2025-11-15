#!/usr/bin/env python3
"""
Script to organize documentation files into the docs directory.

This script moves markdown files from the root directory to appropriate
subdirectories in the docs folder while preserving the directory structure.
"""

import os
import shutil
from pathlib import Path

# Define root and docs directories
ROOT_DIR = Path(__file__).parent.parent
DOCS_DIR = ROOT_DIR / "docs"

# Files to keep in the root directory
KEEP_IN_ROOT = {
    "README.md",
    "README_HUGGINGFACE.md",
    "README_WIKIPEDIA.md",
    "LICENSE",
    "pyproject.toml",
    "requirements.txt",
    "Dockerfile",
    "docker-compose.yml",
    "Makefile",
    "Makefile.wiki",
    "pytest.ini",
    "mkdocs.yml",
    "deploy.sh",
    "release.sh",
    "run_tests.sh",
    "verify_imports.py",
    "verify_pr1.py",
    "1.txt",  # Not sure what this is, keeping it for now
}


def move_documentation():
    """Move documentation files to appropriate directories."""
    # Create necessary subdirectories if they don't exist
    (DOCS_DIR / "project").mkdir(exist_ok=True)
    (DOCS_DIR / "guides").mkdir(exist_ok=True)

    # Move files based on their type
    for file_path in ROOT_DIR.glob("*.md"):
        file_name = file_path.name

        # Skip files that should stay in root
        if file_name in KEEP_IN_ROOT:
            continue

        # Determine target directory based on filename
        if file_name.startswith(("IMPLEMENTATION_", "SCAFFOLD_", "PHASE", "CURRENT_", "COMPLETE_")):
            target_dir = DOCS_DIR / "project"
        elif file_name.startswith(("QUICKSTART_", "TESTING_", "DEVELOPER_")):
            target_dir = DOCS_DIR / "guides"
        else:
            target_dir = DOCS_DIR

        # Handle potential name conflicts
        target_path = target_dir / file_name
        if target_path.exists():
            print(f"Warning: {target_path} already exists. Skipping.")
            continue

        # Move the file
        print(f"Moving {file_path} to {target_path}")
        shutil.move(str(file_path), str(target_path))

    print("\nDocumentation organization complete!")
    print("The following files were kept in the root directory:")
    for f in sorted(KEEP_IN_ROOT):
        print(f"- {f}")


if __name__ == "__main__":
    move_documentation()
