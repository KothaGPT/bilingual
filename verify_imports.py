#!/usr/bin/env python3
"""
Verify that all modules can be imported correctly.
Tests both base functionality and optional ML features.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_base_imports():
    """Test that base modules import without torch/transformers."""
    print("Testing base module imports...")
    
    try:
        from bilingual.modules import (
            metaphor_detector,
            simile_detector,
            tone_classifier,
            detect_meter,
            StyleTransferModel,
        )
        print("  ✓ All base imports successful")
        return True
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False


def test_ml_imports():
    """Test that ML modules import if torch/transformers available."""
    print("\nTesting ML module imports...")
    
    try:
        import torch
        import transformers
        ml_available = True
        print("  ✓ torch and transformers available")
    except ImportError:
        ml_available = False
        print("  ⚠ torch/transformers not available (optional)")
        return True  # Not a failure
    
    if ml_available:
        try:
            from bilingual.modules import PoeticMeterDetector, StyleTransferGPT
            print("  ✓ ML modules imported successfully")
            return True
        except ImportError as e:
            print(f"  ✗ ML import failed: {e}")
            return False
    
    return True


def test_basic_functionality():
    """Test that basic functions work."""
    print("\nTesting basic functionality...")
    
    try:
        from bilingual.modules import metaphor_detector, tone_classifier
        
        # Test metaphor detection
        result = metaphor_detector("Life is a journey")
        assert isinstance(result, list), "metaphor_detector should return list"
        print("  ✓ metaphor_detector works")
        
        # Test tone classification
        result = tone_classifier("This is wonderful!")
        assert isinstance(result, dict), "tone_classifier should return dict"
        assert 'positive' in result, "tone_classifier should have 'positive' key"
        print("  ✓ tone_classifier works")
        
        return True
    except Exception as e:
        print(f"  ✗ Functionality test failed: {e}")
        return False


def main():
    """Run all verification tests."""
    print("=" * 60)
    print("Module Import Verification")
    print("=" * 60)
    print()
    
    results = []
    
    # Run tests
    results.append(("Base Imports", test_base_imports()))
    results.append(("ML Imports", test_ml_imports()))
    results.append(("Basic Functionality", test_basic_functionality()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print()
    print(f"Results: {passed}/{total} passed")
    print("=" * 60)
    
    if passed == total:
        print("\n✓ All verifications passed!")
        return 0
    else:
        print(f"\n✗ {total - passed} verification(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
