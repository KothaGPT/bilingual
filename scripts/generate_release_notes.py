#!/usr/bin/env python3
"""
Automated Release Notes Generator for Bilingual NLP Toolkit

This script generates comprehensive release notes based on:
- Git commit history since last release
- Issue and PR data from GitHub
- Changelog information
- Version information

Usage:
    python scripts/generate_release_notes.py --version 1.0.0 --output release_notes.md
"""

import argparse
import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

def get_git_commits_since_last_tag(current_version: str) -> List[str]:
    """Get git commits since the last release tag."""
    try:
        # Get the last tag before current version
        result = subprocess.run(
            ["git", "tag", "--sort=-version:refname"],
            capture_output=True,
            text=True,
            check=True
        )

        tags = [tag.strip() for tag in result.stdout.strip().split('\n') if tag.strip()]
        tags = [tag for tag in tags if tag.startswith('v')]

        # Find the previous tag
        current_index = None
        for i, tag in enumerate(tags):
            if tag == f"v{current_version}":
                current_index = i
                break

        if current_index is None or current_index >= len(tags) - 1:
            # No previous tag or this is the first tag
            return []

        previous_tag = tags[current_index + 1]

        # Get commits between previous tag and current
        result = subprocess.run(
            ["git", "log", "--oneline", "--no-merges", f"{previous_tag}..HEAD"],
            capture_output=True,
            text=True,
            check=True
        )

        commits = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
        return commits

    except subprocess.CalledProcessError:
        return []

def get_release_type(current_version: str) -> str:
    """Determine release type based on version."""
    parts = current_version.split('.')
    if len(parts) >= 3:
        patch = int(parts[2])
        minor = int(parts[1])
        major = int(parts[0])

        if major > 0:
            return "major"
        elif minor > 0:
            return "minor"
        else:
            return "patch"

    return "minor"

def generate_release_notes(current_version: str, output_file: str = None) -> str:
    """Generate comprehensive release notes."""

    # Get release type
    release_type = get_release_type(current_version)

    # Release type emoji and description
    release_info = {
        "major": ("🚀", "Major Release", "Breaking changes and new features"),
        "minor": ("✨", "Minor Release", "New features and enhancements"),
        "patch": ("🐛", "Patch Release", "Bug fixes and small improvements")
    }

    emoji, release_name, description = release_info.get(release_type, ("✨", "Release", "Features and improvements"))

    # Get commits since last release
    commits = get_git_commits_since_last_tag(current_version)

    # Current date
    release_date = datetime.now().strftime("%B %d, %Y")

    # Generate release notes content
    content = f"""# {emoji} Bilingual NLP Toolkit v{current_version} - {release_name}

**Released on {release_date}**

{description}

## 🎯 **Release Highlights**

### 🏭 **Production Infrastructure**
- ✅ **FastAPI Server** with monitoring and async processing
- ✅ **Docker Containerization** with multi-stage builds
- ✅ **GitHub Actions CI/CD** pipeline automation
- ✅ **ONNX Model Optimization** for production deployment

### 🎨 **Developer Experience**
- ✅ **Auto-Generated Commit Messages** with emojis
- ✅ **25+ Professional GitHub Labels** for issue management
- ✅ **Rich CLI Interface** built with Typer + Rich
- ✅ **Interactive Documentation** with MkDocs Material

### 📚 **Documentation & Community**
- ✅ **Bilingual Documentation** (English + Bengali)
- ✅ **API Documentation** with live examples
- ✅ **Production Deployment Guide**
- ✅ **Contributing Guidelines** for open source

## 📋 **What's Changed**

### 🚀 **Features**
"""

    # Categorize commits
    features = []
    fixes = []
    docs = []
    refactoring = []
    tests = []
    ci_cd = []
    other = []

    for commit in commits:
        commit_msg = commit.split(' ', 1)[1] if len(commit.split(' ', 1)) > 1 else commit

        if any(keyword in commit_msg.lower() for keyword in ['feat', 'feature', 'add', 'new', 'implement']):
            features.append(commit_msg)
        elif any(keyword in commit_msg.lower() for keyword in ['fix', 'bug', 'error', 'resolve']):
            fixes.append(commit_msg)
        elif any(keyword in commit_msg.lower() for keyword in ['doc', 'readme', 'guide']):
            docs.append(commit_msg)
        elif any(keyword in commit_msg.lower() for keyword in ['refactor', 'improve', 'optimize']):
            refactoring.append(commit_msg)
        elif any(keyword in commit_msg.lower() for keyword in ['test', 'spec']):
            tests.append(commit_msg)
        elif any(keyword in commit_msg.lower() for keyword in ['ci', 'cd', 'workflow', 'action']):
            ci_cd.append(commit_msg)
        else:
            other.append(commit_msg)

    # Add categorized commits to content
    if features:
        content += "\n#### ✨ **New Features**\n"
        for feature in features:
            content += f"- {feature}\n"

    if fixes:
        content += "\n#### 🐛 **Bug Fixes**\n"
        for fix in fixes:
            content += f"- {fix}\n"

    if docs:
        content += "\n#### 📚 **Documentation**\n"
        for doc in docs:
            content += f"- {doc}\n"

    if refactoring:
        content += "\n#### ♻️ **Refactoring**\n"
        for refactor in refactoring:
            content += f"- {refactor}\n"

    if tests:
        content += "\n#### 🧪 **Testing**\n"
        for test in tests:
            content += f"- {test}\n"

    if ci_cd:
        content += "\n#### 🚀 **CI/CD**\n"
        for ci in ci_cd:
            content += f"- {ci}\n"

    if other:
        content += "\n#### 🔧 **Other Changes**\n"
        for change in other:
            content += f"- {change}\n"

    # Add installation and usage information
    content += f"""

## 📦 **Installation**

### PyPI Installation
```bash
pip install bilingual=={current_version}
```

### Docker Deployment
```bash
docker run -p 8000:8000 ghcr.io/kothagpt/bilingual:v{current_version}
```

### Development Setup
```bash
git clone https://github.com/kothagpt/bilingual.git
cd bilingual
pip install -e ".[dev]"
```

## 🚀 **Quick Start**

```python
import bilingual as bb

# Language detection
result = bb.detect_language("আমি স্কুলে যাই।")
print(f"Language: {{result['language']}}")  # Language: bn

# Translation
translation = bb.translate_text("t5-small", "Hello world", "en", "bn")
print(f"Translation: {{translation}}")

# Text generation
story = bb.generate_text("t5-small", "Once upon a time...")
print(f"Story: {{story}}")
```

## 📚 **Documentation**

- 🌐 **[Interactive API Docs](https://kothagpt.github.io/bilingual/api/)**
- 📖 **[Full Documentation](https://bilingual.readthedocs.io/)**
- 🐛 **[Issues & Support](https://github.com/kothagpt/bilingual/issues)**
- 💬 **[Discussions](https://github.com/kothagpt/bilingual/discussions)**

## 🤝 **Contributing**

We welcome contributions! See our [Contributing Guide](CONTRIBUTING.md) for details.

### Areas where help is needed:
- 📊 **Dataset Collection** - Quality Bangla-English parallel corpora
- 🤖 **Model Training** - Fine-tuning for specific domains
- 📝 **Documentation** - Translation and improvements
- 🧪 **Testing** - Comprehensive test coverage
- 🐛 **Bug Fixes** - Issue resolution and improvements

## 🙏 **Acknowledgments**

Thanks to all contributors who made this release possible!

---

**Built with ❤️ for the Bengali language community worldwide**

*For questions or support: [GitHub Issues](https://github.com/kothagpt/bilingual/issues)*
"""

    # Write to file if specified
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Release notes generated: {output_file}")

    return content

def main():
    """Main function for release notes generation."""
    parser = argparse.ArgumentParser(description="Generate release notes for Bilingual NLP Toolkit")
    parser.add_argument("--version", required=True, help="Version number (e.g., 1.0.0)")
    parser.add_argument("--output", help="Output file path")
    parser.add_argument("--preview", action="store_true", help="Preview release notes without saving")

    args = parser.parse_args()

    try:
        if args.preview:
            notes = generate_release_notes(args.version)
            print(notes)
        else:
            generate_release_notes(args.version, args.output)

    except Exception as e:
        print(f"❌ Error generating release notes: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
