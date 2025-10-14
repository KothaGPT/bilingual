#!/usr/bin/env python3
"""
Pre-commit hook for auto-generating commit messages with emojis.

This hook analyzes staged changes and suggests commit messages with
appropriate emojis based on the conventional commit format.
"""

import subprocess
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def get_staged_files():
    """Get list of staged files."""
    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            capture_output=True,
            text=True,
            check=True
        )
        return [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
    except subprocess.CalledProcessError:
        return []

def analyze_changes(files):
    """Analyze files to determine commit type and scope."""
    types = []
    scopes = []

    for file_path in files:
        path = Path(file_path)

        # Determine scope
        if path.parts[0] == "bilingual":
            if "api" in str(path) or "server" in str(path):
                scopes.append("api")
            elif "tokenizer" in str(path):
                scopes.append("tokenizer")
            elif "evaluation" in str(path):
                scopes.append("evaluation")
            elif "models" in str(path):
                scopes.append("models")
            elif "data" in str(path) or "datasets" in str(path):
                scopes.append("data")
            elif "config" in str(path):
                scopes.append("config")
            elif "testing" in str(path) or "test" in str(path):
                scopes.append("testing")
            elif "cli" in str(path):
                scopes.append("cli")
        elif path.parts[0] == "scripts":
            scopes.append("deployment")
        elif path.parts[0] == "docs":
            scopes.append("docs")
        elif path.parts[0] == ".github":
            scopes.append("ci")

    # Determine type based on file patterns
    for file_path in files:
        path_str = str(file_path).lower()

        if any(keyword in path_str for keyword in ["bug", "fix", "error", "issue"]):
            types.append("fix")
        elif any(keyword in path_str for keyword in ["feature", "add", "new", "implement"]):
            types.append("feat")
        elif any(keyword in path_str for keyword in ["doc", "readme", "guide"]):
            types.append("docs")
        elif any(keyword in path_str for keyword in ["test", "spec"]):
            types.append("test")
        elif any(keyword in path_str for keyword in ["refactor", "improve", "optimize"]):
            types.append("refactor")
        elif any(keyword in path_str for keyword in ["ci", "workflow", "action"]):
            types.append("ci")
        elif any(keyword in path_str for keyword in ["perf", "performance", "speed"]):
            types.append("perf")

    return types, scopes

def generate_commit_message(types, scopes):
    """Generate a commit message with emojis."""
    # Emoji mappings
    emoji_map = {
        "feat": "✨",
        "fix": "🐛",
        "docs": "📚",
        "test": "🧪",
        "refactor": "♻️",
        "ci": "🚀",
        "perf": "⚡",
        "style": "💅",
        "chore": "🔧",
        "build": "🏗️",
    }

    scope_emoji_map = {
        "api": "🔌",
        "cli": "💻",
        "models": "🤖",
        "tokenizer": "🔤",
        "evaluation": "📊",
        "data": "📁",
        "docs": "📖",
        "deployment": "🚢",
        "config": "⚙️",
        "testing": "🧪",
        "ci": "🚀",
    }

    # Get primary type and scope
    primary_type = types[0] if types else "chore"
    primary_scope = scopes[0] if scopes else None

    # Get emojis
    type_emoji = emoji_map.get(primary_type, "🔧")
    scope_emoji = scope_emoji_map.get(primary_scope, "") if primary_scope else ""

    # Build commit message
    if primary_scope:
        prefix = f"{type_emoji}{scope_emoji} {primary_type}({primary_scope}):"
    else:
        prefix = f"{type_emoji} {primary_type}:"

    # Generate description
    file_count = len(get_staged_files())
    if file_count == 1:
        desc = "update file"
    elif file_count <= 3:
        desc = f"update {file_count} files"
    else:
        desc = f"update {file_count} files"

    return f"{prefix} {desc}"

def main():
    """Main pre-commit hook function."""
    # Check if we're in a git repository
    try:
        subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            capture_output=True,
            check=True
        )
    except subprocess.CalledProcessError:
        print("❌ Not in a git repository")
        sys.exit(1)

    # Get staged files
    staged_files = get_staged_files()

    if not staged_files:
        print("❌ No staged files found. Stage some files first.")
        sys.exit(1)

    # Analyze changes
    types, scopes = analyze_changes(staged_files)

    if not types:
        types = ["chore"]

    # Generate commit message
    commit_message = generate_commit_message(types, scopes)

    print("🔍 Analyzing staged changes...")
    print(f"📁 Files: {len(staged_files)}")
    print(f"🏷️  Types: {', '.join(types)}")
    print(f"📍 Scopes: {', '.join(scopes) if scopes else 'None'}")

    print(f"\n💡 Suggested commit message:")
    print(f"   {commit_message}")

    # Ask user if they want to use this message
    print("\n📝 Options:")
    print("1. Use suggested message")
    print("2. Enter custom message")
    print("3. Cancel")

    try:
        choice = input("\nChoose (1-3): ").strip()

        if choice == "1":
            final_message = commit_message
        elif choice == "2":
            final_message = input("Enter custom commit message: ").strip()
            if not final_message:
                print("❌ Empty message. Cancelled.")
                sys.exit(1)
        elif choice == "3" or choice == "":
            print("❌ Cancelled.")
            sys.exit(1)
        else:
            print("❌ Invalid choice. Cancelled.")
            sys.exit(1)

        # Set the commit message as an environment variable for git
        # This would be used by a wrapper script that calls git commit
        print(f"\n✅ Using commit message: {final_message}")
        print(f"💻 Run: git commit -m \"{final_message}\"")

    except KeyboardInterrupt:
        print("\n❌ Cancelled.")
        sys.exit(1)

if __name__ == "__main__":
    main()
