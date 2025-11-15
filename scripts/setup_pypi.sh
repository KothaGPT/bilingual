#!/bin/bash
# Make PyPI release script executable
chmod +x scripts/pypi_release.sh
chmod +x scripts/pypi_release.py

echo "🚀 Bilingual PyPI Release Setup Complete!"
echo ""
echo "Available commands:"
echo "  make pypi-build     - Build package"
echo "  make pypi-test      - Upload to TestPyPI"
echo "  make pypi-release   - Release to PyPI"
echo "  make pypi-install   - Install from PyPI"
echo "  make check-install  - Verify installation"
echo "  make pypi-full      - Complete workflow"
echo ""
echo "Or use the Python script:"
echo "  python scripts/pypi_release.py --help"
echo ""
echo "Quick start:"
echo "  1. make pypi-build"
echo "  2. make pypi-test"
echo "  3. make check-install"
echo "  4. make pypi-release"
echo ""
echo "For detailed guide, see: docs/PYPI_RELEASE_GUIDE.md"
