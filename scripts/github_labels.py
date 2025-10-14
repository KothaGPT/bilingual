#!/usr/bin/env python3
"""
GitHub Labels Configuration for Bilingual NLP Toolkit

This script creates standardized GitHub labels for issues and pull requests
to improve project organization and contributor experience.
"""

import json
import sys
from typing import Dict, List, Any

# Comprehensive label definitions for the bilingual project
GITHUB_LABELS = [
    # Priority Labels
    {
        "name": "priority:critical",
        "color": "b60205",
        "description": "Critical priority - blocking release or security issue"
    },
    {
        "name": "priority:high",
        "color": "d93f0b",
        "description": "High priority - should be addressed soon"
    },
    {
        "name": "priority:medium",
        "color": "fbca04",
        "description": "Medium priority - address in next sprint"
    },
    {
        "name": "priority:low",
        "color": "0e8a16",
        "description": "Low priority - can be addressed later"
    },

    # Type Labels
    {
        "name": "type:bug",
        "color": "d73a49",
        "description": "Bug reports and fixes"
    },
    {
        "name": "type:enhancement",
        "color": "a2eeef",
        "description": "Feature requests and enhancements"
    },
    {
        "name": "type:documentation",
        "color": "0075ca",
        "description": "Documentation improvements and updates"
    },
    {
        "name": "type:refactoring",
        "color": "fbca04",
        "description": "Code refactoring and improvements"
    },
    {
        "name": "type:testing",
        "color": "c2e0c6",
        "description": "Testing related changes"
    },
    {
        "name": "type:performance",
        "color": "fbca04",
        "description": "Performance optimizations"
    },
    {
        "name": "type:security",
        "color": "ee0701",
        "description": "Security related issues"
    },
    {
        "name": "type:dependencies",
        "color": "0366d6",
        "description": "Dependency updates and changes"
    },
    {
        "name": "type:ci-cd",
        "color": "1d76db",
        "description": "CI/CD pipeline changes"
    },

    # Component Labels
    {
        "name": "component:api",
        "color": "0052cc",
        "description": "API related changes"
    },
    {
        "name": "component:cli",
        "color": "0052cc",
        "description": "Command-line interface changes"
    },
    {
        "name": "component:models",
        "color": "0052cc",
        "description": "Model training and inference"
    },
    {
        "name": "component:tokenizer",
        "color": "0052cc",
        "description": "Tokenization related changes"
    },
    {
        "name": "component:evaluation",
        "color": "0052cc",
        "description": "Evaluation metrics and testing"
    },
    {
        "name": "component:data",
        "color": "0052cc",
        "description": "Data collection and processing"
    },
    {
        "name": "component:docs",
        "color": "0052cc",
        "description": "Documentation changes"
    },
    {
        "name": "component:deployment",
        "color": "0052cc",
        "description": "Deployment and DevOps"
    },

    # Language Labels
    {
        "name": "language:bangla",
        "color": "5319e7",
        "description": "Bangla/Bengali language specific"
    },
    {
        "name": "language:english",
        "color": "5319e7",
        "description": "English language specific"
    },
    {
        "name": "language:multilingual",
        "color": "5319e7",
        "description": "Multilingual/cross-language features"
    },

    # Status Labels
    {
        "name": "status:ready",
        "color": "0e8a16",
        "description": "Ready for review or merge"
    },
    {
        "name": "status:in-progress",
        "color": "fbca04",
        "description": "Currently being worked on"
    },
    {
        "name": "status:review-needed",
        "color": "fbca04",
        "description": "Needs review from maintainers"
    },
    {
        "name": "status:blocked",
        "color": "b60205",
        "description": "Blocked by external factors"
    },
    {
        "name": "status:duplicate",
        "color": "cccccc",
        "description": "Duplicate of another issue"
    },
    {
        "name": "status:wont-fix",
        "color": "ffffff",
        "description": "Will not be fixed"
    },

    # Size Labels
    {
        "name": "size:xs",
        "color": "0e8a16",
        "description": "Extra small change (< 10 lines)"
    },
    {
        "name": "size:s",
        "color": "0e8a16",
        "description": "Small change (10-50 lines)"
    },
    {
        "name": "size:m",
        "color": "fbca04",
        "description": "Medium change (50-200 lines)"
    },
    {
        "name": "size:l",
        "color": "d93f0b",
        "description": "Large change (200-500 lines)"
    },
    {
        "name": "size:xl",
        "color": "b60205",
        "description": "Extra large change (> 500 lines)"
    },

    # Good First Issue Labels
    {
        "name": "good first issue",
        "color": "7057ff",
        "description": "Good for newcomers to contribute"
    },
    {
        "name": "help wanted",
        "color": "008672",
        "description": "Extra attention needed"
    },
    {
        "name": "question",
        "color": "d876e3",
        "description": "Questions and discussions"
    }
]

def generate_labels_json():
    """Generate GitHub labels configuration as JSON."""
    return json.dumps(GITHUB_LABELS, indent=2, ensure_ascii=False)

def print_labels_table():
    """Print labels in a formatted table."""
    print("\n📋 GitHub Labels Configuration")
    print("=" * 50)

    # Group labels by category
    priority_labels = [label for label in GITHUB_LABELS if label["name"].startswith("priority:")]
    type_labels = [label for label in GITHUB_LABELS if label["name"].startswith("type:")]
    component_labels = [label for label in GITHUB_LABELS if label["name"].startswith("component:")]
    status_labels = [label for label in GITHUB_LABELS if label["name"].startswith("status:")]

    print("\n🎯 Priority Labels:")
    for label in priority_labels:
        print(f"  • {label['name']} ({label['color']}) - {label['description']}")

    print("\n🏷️  Type Labels:")
    for label in type_labels:
        print(f"  • {label['name']} ({label['color']}) - {label['description']}")

    print("\n🧩 Component Labels:")
    for label in component_labels:
        print(f"  • {label['name']} ({label['color']}) - {label['description']}")

    print("\n📊 Status Labels:")
    for label in status_labels:
        print(f"  • {label['name']} ({label['color']}) - {label['description']}")

def main():
    """Main function to display GitHub labels configuration."""
    print("🚀 Bilingual NLP Toolkit - GitHub Labels Configuration")
    print("=" * 60)

    print("\nThis configuration provides standardized labels for:")
    print("• Issue categorization and prioritization")
    print("• Pull request organization")
    print("• Contributor guidance")
    print("• Project management")

    print_labels_table()

    print("💡 Usage:")
    print("1. Copy the JSON configuration to .github/labels.json")
    print("2. Use GitHub CLI: gh label create --from-file .github/labels.json")
    print("3. Or create labels manually in GitHub repository settings")

    # Generate JSON file
    try:
        with open(".github/labels.json", "w", encoding="utf-8") as f:
            f.write(generate_labels_json())
        print("\n✅ Labels configuration saved to: .github/labels.json")
    except Exception as e:
        print(f"\n❌ Could not save labels file: {e}")

if __name__ == "__main__":
    main()
