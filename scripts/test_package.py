#!/usr/bin/env python3
"""
Test script to verify Bilingual package functionality.

This script tests all major components of the package to ensure
they work correctly before PyPI release.
"""

import sys
import traceback
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_imports():
    """Test basic imports."""
    print("🧪 Testing imports...")

    try:
        import bilingual

        print(f"  ✅ Package imported successfully")
        print(f"  📦 Version: {bilingual.__version__}")
        print(f"  📁 Location: {bilingual.__file__}")

        # Test version attributes
        assert hasattr(bilingual, "__version__")
        assert hasattr(bilingual, "version")
        assert hasattr(bilingual, "__version_tuple__")
        print("  ✅ Version attributes available")

        return True
    except Exception as e:
        print(f"  ❌ Import failed: {e}")
        traceback.print_exc()
        return False


def test_core_functions():
    """Test core functionality."""
    print("🧪 Testing core functions...")

    try:
        import bilingual as bb

        # Test language detection
        result = bb.detect_language("Hello world")
        print(f"  ✅ Language detection: {result}")

        # Test normalization
        normalized = bb.normalize_text("Hello world!")
        print(f"  ✅ Text normalization: '{normalized}'")

        # Test Bengali text
        result_bn = bb.detect_language("আমি বাংলায় কথা বলি")
        print(f"  ✅ Bengali detection: {result_bn}")

        normalized_bn = bb.normalize_text("আমি বাংলায় কথা বলি।")
        print(f"  ✅ Bengali normalization: '{normalized_bn}'")

        return True
    except Exception as e:
        print(f"  ❌ Core functions failed: {e}")
        traceback.print_exc()
        return False


def test_tokenizer():
    """Test tokenizer functionality."""
    print("🧪 Testing tokenizer...")

    try:
        import bilingual as bb

        # Check if tokenizer files exist
        tokenizer_path = Path("models/tokenizer/bilingual_sp.model")
        if tokenizer_path.exists():
            tokenizer = bb.load_tokenizer(str(tokenizer_path))
            tokens = tokenizer.encode("Hello world")
            print(f"  ✅ Tokenizer loaded: {len(tokens)} tokens")

            decoded = tokenizer.decode(tokens)
            print(f"  ✅ Tokenization roundtrip: '{decoded}'")
        else:
            print("  ⚠️  Tokenizer model not found (this is expected)")

        return True
    except Exception as e:
        print(f"  ❌ Tokenizer test failed: {e}")
        return False


def test_api():
    """Test API functionality."""
    print("🧪 Testing API...")

    try:
        import bilingual as bb

        # Test API access
        assert hasattr(bb, "bilingual_api")
        print("  ✅ API module available")

        # Test settings
        settings = bb.get_settings()
        print(f"  ✅ Settings loaded: {type(settings)}")

        return True
    except Exception as e:
        print(f"  ❌ API test failed: {e}")
        traceback.print_exc()
        return False


def test_cli():
    """Test CLI functionality."""
    print("🧪 Testing CLI...")

    try:
        # Test CLI import
        from scripts.cli import run_cli

        print("  ✅ CLI module imports successfully")

        # Note: We don't run the CLI here as it requires user interaction
        print("  ℹ️  CLI functionality available (requires interactive testing)")

        return True
    except Exception as e:
        print(f"  ❌ CLI test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("🚀 Bilingual Package Pre-Release Tests")
    print("=" * 60)

    tests = [
        ("Imports", test_imports),
        ("Core Functions", test_core_functions),
        ("Tokenizer", test_tokenizer),
        ("API", test_api),
        ("CLI", test_cli),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}:")
        success = test_func()
        results.append((test_name, success))

    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)

    all_passed = True
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name}: {status}")
        if not success:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All tests passed! Package is ready for PyPI release.")
        print("\nNext steps:")
        print("  1. make pypi-build")
        print("  2. make pypi-test")
        print("  3. make pypi-release")
    else:
        print("❌ Some tests failed. Please fix issues before releasing.")
        print("\nCheck:")
        print("  - Import errors")
        print("  - Missing dependencies")
        print("  - Function implementations")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
