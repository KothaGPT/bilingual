#!/usr/bin/env python3
"""
Auto-generated commit messages with emojis for the Bilingual NLP Toolkit.

This script analyzes git changes and generates standardized commit messages
with appropriate emojis based on the type and scope of changes.
"""

import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Emoji mappings for different types of changes
EMOJI_MAPPINGS = {
    # Types
    "feat": "✨",  # New feature
    "fix": "🐛",  # Bug fix
    "docs": "📚",  # Documentation
    "style": "💅",  # Code style/formatting
    "refactor": "♻️",  # Code refactoring
    "test": "🧪",  # Tests
    "chore": "🔧",  # Maintenance tasks
    "perf": "⚡",  # Performance improvements
    "ci": "🚀",  # CI/CD changes
    "build": "🏗️",  # Build system changes
    "revert": "⏪",  # Reverting changes
    # Scopes
    "api": "🔌",  # API changes
    "cli": "💻",  # CLI changes
    "models": "🤖",  # Model changes
    "tokenizer": "🔤",  # Tokenization
    "evaluation": "📊",  # Evaluation/metrics
    "data": "📁",  # Data processing
    "docs": "📖",  # Documentation
    "deployment": "🚢",  # Deployment
    "config": "⚙️",  # Configuration
    "testing": "🧪",  # Testing framework
    # Special cases
    "security": "🔒",  # Security fixes
    "breaking": "💥",  # Breaking changes
    "experimental": "🧪",  # Experimental features
}

# Commit message patterns
COMMIT_PATTERNS = [
    # Feature patterns
    (r"add.*feature|implement.*feature|new.*feature", "feat"),
    (r"add.*functionality|implement.*functionality", "feat"),
    (r"introduce.*capability|add.*capability", "feat"),
    # Bug fix patterns
    (r"fix.*bug|resolve.*bug|fix.*issue", "fix"),
    (r"correct.*error|fix.*error", "fix"),
    (r"resolve.*problem|fix.*problem", "fix"),
    # Documentation patterns
    (r"update.*doc|add.*doc|improve.*doc", "docs"),
    (r"add.*example|update.*example", "docs"),
    (r"document.*feature|document.*function", "docs"),
    # Refactoring patterns
    (r"refactor.*code|improve.*code|optimize.*code", "refactor"),
    (r"clean.*code|restructure.*code", "refactor"),
    (r"simplify.*logic|improve.*logic", "refactor"),
    # Test patterns
    (r"add.*test|implement.*test|write.*test", "test"),
    (r"test.*functionality|test.*feature", "test"),
    (r"add.*assertion|add.*test.*case", "test"),
    # Performance patterns
    (r"improve.*performance|optimize.*performance", "perf"),
    (r"increase.*speed|reduce.*latency", "perf"),
    (r"enhance.*efficiency|optimize.*efficiency", "perf"),
    # CI/CD patterns
    (r"update.*ci|modify.*ci|improve.*ci", "ci"),
    (r"update.*workflow|modify.*workflow", "ci"),
    (r"add.*automation|improve.*automation", "ci"),
    # Build patterns
    (r"update.*build|modify.*build|fix.*build", "build"),
    (r"add.*dependency|update.*dependency", "build"),
    (r"configure.*build|setup.*build", "build"),
]


def get_git_changes() -> Tuple[List[str], List[str]]:
    """Get staged and unstaged changes from git."""
    try:
        # Get staged files
        staged = (
            subprocess.check_output(["git", "diff", "--cached", "--name-only"], encoding="utf-8")
            .strip()
            .split("\n")
        )

        # Get unstaged files
        unstaged = (
            subprocess.check_output(["git", "diff", "--name-only"], encoding="utf-8")
            .strip()
            .split("\n")
        )

        # Filter out empty strings
        staged = [f for f in staged if f]
        unstaged = [f for f in unstaged if f]

        return staged, unstaged

    except subprocess.CalledProcessError:
        return [], []


def analyze_changes(files: List[str]) -> Dict[str, Any]:
    """Analyze file changes to determine commit type and scope."""
    analysis = {"types": [], "scopes": [], "breaking": False, "files": files}

    for file_path in files:
        path = Path(file_path)

        # Determine scope based on file location
        if path.parts[0] == "bilingual":
            if "api" in str(path):
                analysis["scopes"].append("api")
            elif "server" in str(path):
                analysis["scopes"].append("api")
            elif "tokenizer" in str(path):
                analysis["scopes"].append("tokenizer")
            elif "evaluation" in str(path):
                analysis["scopes"].append("evaluation")
            elif "models" in str(path):
                analysis["scopes"].append("models")
            elif "data" in str(path) or "datasets" in str(path):
                analysis["scopes"].append("data")
            elif "config" in str(path):
                analysis["scopes"].append("config")
            elif "testing" in str(path) or "test" in str(path):
                analysis["scopes"].append("testing")
            elif "cli" in str(path):
                analysis["scopes"].append("cli")
        elif path.parts[0] == "scripts":
            analysis["scopes"].append("deployment")
        elif path.parts[0] == "docs":
            analysis["scopes"].append("docs")
        elif path.parts[0] == ".github":
            analysis["scopes"].append("ci")

        # Check for breaking changes in file content (simplified check)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                if "BREAKING CHANGE" in content.upper() or "breaking change" in content.lower():
                    analysis["breaking"] = True
        except:
            pass  # Skip files that can't be read

    # Determine primary type based on file patterns and content
    for pattern, commit_type in COMMIT_PATTERNS:
        for file_path in files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    if re.search(pattern, content, re.IGNORECASE):
                        if commit_type not in analysis["types"]:
                            analysis["types"].append(commit_type)
            except:
                pass  # Skip files that can't be read

    # Default type if none detected
    if not analysis["types"]:
        analysis["types"].append("chore")

    # Remove duplicates and sort
    analysis["types"] = sorted(list(set(analysis["types"])))
    analysis["scopes"] = sorted(list(set(analysis["scopes"])))

    return analysis


def generate_commit_message(analysis: Dict[str, Any], custom_message: Optional[str] = None) -> str:
    """Generate a commit message with emojis based on analysis."""
    if custom_message:
        return custom_message

    # Get primary type and scope
    primary_type = analysis["types"][0]
    primary_scope = analysis["scopes"][0] if analysis["scopes"] else None

    # Get emojis
    type_emoji = EMOJI_MAPPINGS.get(primary_type, "🔧")
    scope_emoji = EMOJI_MAPPINGS.get(primary_scope, "") if primary_scope else ""

    # Build commit message
    if primary_scope:
        prefix = f"{type_emoji}{scope_emoji} {primary_type}({primary_scope}):"
    else:
        prefix = f"{type_emoji} {primary_type}:"

    # Add breaking change indicator
    if analysis["breaking"]:
        prefix = f"{EMOJI_MAPPINGS['breaking']} {prefix}"

    # Generate description based on files changed
    file_count = len(analysis["files"])

    if file_count == 1:
        file_desc = f"update {Path(analysis['files'][0]).name}"
    elif file_count <= 3:
        files = [Path(f).name for f in analysis["files"]]
        file_desc = f"update {', '.join(files)}"
    else:
        file_desc = f"update {file_count} files"

    # Combine prefix and description
    commit_message = f"{prefix} {file_desc}"

    return commit_message


def suggest_commit_message():
    """Interactive commit message suggestion."""
    print("🔍 Analyzing git changes...")

    # Get changes
    staged, unstaged = get_git_changes()

    if not staged and not unstaged:
        print("❌ No changes detected. Stage some files first with: git add <files>")
        return

    print(f"📁 Staged files: {len(staged)}")
    print(f"📁 Unstaged files: {len(unstaged)}")

    if unstaged:
        print("\n⚠️  You have unstaged changes. Consider staging them with: git add .")

    if not staged:
        print("❌ No staged changes. Stage files first with: git add <files>")
        return

    # Analyze changes
    analysis = analyze_changes(staged)

    print("\n🔍 Analysis Results:")
    print(f"  Types: {', '.join(analysis['types'])}")
    print(f"  Scopes: {', '.join(analysis['scopes']) if analysis['scopes'] else 'None'}")
    print(f"  Breaking: {'Yes' if analysis['breaking'] else 'No'}")

    # Generate suggested message
    suggested_message = generate_commit_message(analysis)

    print(f"\n💡 Suggested commit message:")
    print(f"   {suggested_message}")

    # Ask for confirmation or custom message
    print("\n📝 Options:")
    print("1. Use suggested message")
    print("2. Enter custom message")
    print("3. Cancel")

    choice = input("\nChoose (1-3): ").strip()

    if choice == "1":
        final_message = suggested_message
    elif choice == "2":
        final_message = input("Enter custom commit message: ").strip()
        if not final_message:
            print("❌ Empty message. Cancelled.")
            return
    elif choice == "3" or choice == "":
        print("❌ Cancelled.")
        return
    else:
        print("❌ Invalid choice. Cancelled.")
        return

    # Commit with the message
    try:
        subprocess.run(["git", "commit", "-m", final_message], check=True)
        print(f"\n✅ Committed with message: {final_message}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Git commit failed: {e}")


def show_commit_examples():
    """Show examples of auto-generated commit messages."""
    print("\n📋 Commit Message Examples:")
    print("=" * 40)

    examples = [
        (
            "feat(api): add language detection endpoint",
            "✨🔌 feat(api): add language detection endpoint",
        ),
        (
            "fix(models): resolve tokenizer memory leak",
            "🐛🤖 fix(models): resolve tokenizer memory leak",
        ),
        ("docs: update API documentation", "📚📖 docs: update API documentation"),
        (
            "refactor(evaluation): optimize BLEU calculation",
            "♻️📊 refactor(evaluation): optimize BLEU calculation",
        ),
        (
            "test: add unit tests for data augmentation",
            "🧪📁 test: add unit tests for data augmentation",
        ),
        ("perf: improve model inference speed", "⚡🤖 perf: improve model inference speed"),
        ("ci: update GitHub Actions workflow", "🚀 ci: update GitHub Actions workflow"),
        ("chore: update dependencies", "🔧🏗️ chore: update dependencies"),
    ]

    for conventional, with_emoji in examples:
        print(f"Conventional: {conventional}")
        print(f"With emojis:  {with_emoji}")
        print()


def main():
    """Main function for commit message generation."""
    print("🚀 Bilingual NLP Toolkit - Auto Commit Messages")
    print("=" * 55)

    if len(sys.argv) > 1:
        if sys.argv[1] == "--examples":
            show_commit_examples()
            return
        elif sys.argv[1] == "--help":
            print("\nUsage:")
            print("  python scripts/commit_messages.py          # Interactive mode")
            print("  python scripts/commit_messages.py --examples # Show examples")
            print("  python scripts/commit_messages.py --help     # Show help")
            print("\nThis tool analyzes your git changes and suggests commit messages with emojis.")
            return

    print("\nThis tool will analyze your staged changes and suggest a commit message.")
    print("Make sure you have staged your changes with: git add <files>")

    # Check if we're in a git repository
    try:
        subprocess.check_output(["git", "rev-parse", "--git-dir"], stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        print("❌ Not in a git repository.")
        return

    # Run interactive suggestion
    suggest_commit_message()


if __name__ == "__main__":
    main()
