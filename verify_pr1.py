#!/usr/bin/env python3
"""
Quick verification script for PR-1 literary modules.
Tests that all imports work and basic functionality is operational.
"""

def verify_imports():
    """Verify all new modules can be imported."""
    print("✓ Verifying imports...")
    
    try:
        from bilingual.modules import (
            metaphor_detector,
            simile_detector,
            tone_classifier,
            detect_meter,
            StyleTransferModel,
        )
        print("  ✓ All imports successful")
        return True
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False


def verify_literary_analysis():
    """Verify literary analysis functions work."""
    print("\n✓ Verifying literary analysis...")
    
    from bilingual.modules import metaphor_detector, simile_detector, tone_classifier
    
    # Test metaphor detection
    metaphors = metaphor_detector("Life is a journey")
    assert len(metaphors) > 0, "Metaphor detection failed"
    print(f"  ✓ Metaphor detection: found {len(metaphors)} metaphor(s)")
    
    # Test simile detection
    similes = simile_detector("She runs like the wind")
    assert len(similes) > 0, "Simile detection failed"
    print(f"  ✓ Simile detection: found {len(similes)} simile(s)")
    
    # Test tone classification
    tone = tone_classifier("This is wonderful!")
    assert 'positive' in tone, "Tone classification failed"
    assert tone['positive'] > tone['negative'], "Positive tone not detected"
    print(f"  ✓ Tone classification: {tone}")
    
    return True


def verify_poetic_meter():
    """Verify poetic meter detection works."""
    print("\n✓ Verifying poetic meter...")
    
    from bilingual.modules import detect_meter
    
    # Test English meter
    result = detect_meter("Shall I compare thee to a summer's day?", language='english')
    assert result['language'] == 'english', "Language detection failed"
    assert len(result['lines']) == 1, "Line parsing failed"
    print(f"  ✓ English meter: {result['lines'][0]['total_syllables']} syllables")
    
    # Test Bengali meter
    result = detect_meter("আমার সোনার বাংলা", language='bengali')
    assert result['language'] == 'bengali', "Bengali detection failed"
    print(f"  ✓ Bengali meter: {result['lines'][0]['total_matra']} matra")
    
    return True


def verify_style_transfer():
    """Verify style transfer model works."""
    print("\n✓ Verifying style transfer...")
    
    from bilingual.modules import StyleTransferModel
    
    # Test model initialization
    model = StyleTransferModel()
    print(f"  ✓ Model initialized: {model}")
    
    # Test loading
    model.load()
    assert model.loaded, "Model loading failed"
    print("  ✓ Model loaded")
    
    # Test conversion
    result = model.convert("I can't do this", target_style='formal')
    assert "cannot" in result, "Formal conversion failed"
    print(f"  ✓ Style conversion: '{result}'")
    
    # Test batch conversion
    results = model.batch_convert(["I can't", "I won't"], target_style='formal')
    assert len(results) == 2, "Batch conversion failed"
    print(f"  ✓ Batch conversion: {len(results)} texts processed")
    
    # Test available styles
    styles = model.available_styles()
    assert 'formal' in styles, "Available styles failed"
    print(f"  ✓ Available styles: {styles}")
    
    return True


def main():
    """Run all verification checks."""
    print("=" * 60)
    print("PR-1 Literary Modules Verification")
    print("=" * 60)
    
    checks = [
        ("Imports", verify_imports),
        ("Literary Analysis", verify_literary_analysis),
        ("Poetic Meter", verify_poetic_meter),
        ("Style Transfer", verify_style_transfer),
    ]
    
    passed = 0
    failed = 0
    
    for name, check_fn in checks:
        try:
            if check_fn():
                passed += 1
            else:
                failed += 1
                print(f"\n✗ {name} check failed")
        except Exception as e:
            failed += 1
            print(f"\n✗ {name} check failed with error: {e}")
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("\n✓ All checks passed! PR-1 is ready.")
        return 0
    else:
        print(f"\n✗ {failed} check(s) failed. Please review.")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
