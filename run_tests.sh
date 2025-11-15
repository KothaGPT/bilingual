#!/bin/bash
# Simple test runner that handles missing pytest gracefully

set -e

echo "=========================================="
echo "Running Bilingual Package Tests"
echo "=========================================="
echo ""

# Check if pytest is available
if command -v pytest &> /dev/null; then
    echo "✓ pytest found, running full test suite..."
    pytest tests/ --maxfail=1 --disable-warnings -q
elif command -v python3 &> /dev/null; then
    echo "⚠ pytest not found, trying python3 -m pytest..."
    python3 -m pytest tests/ --maxfail=1 --disable-warnings -q 2>/dev/null || {
        echo "✗ pytest module not installed"
        echo ""
        echo "Please install test dependencies:"
        echo "  pip install -e \".[dev]\""
        echo ""
        exit 1
    }
else
    echo "✗ Python not found"
    exit 1
fi

echo ""
echo "=========================================="
echo "✓ All tests passed!"
echo "=========================================="
