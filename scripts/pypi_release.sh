#!/bin/bash
# PyPI Release Script for Bilingual Project
# Usage: bash scripts/pypi_release.sh [version]

set -e

echo "========================================"
echo "🚀 Bilingual PyPI Release Script"
echo "========================================"

# Configuration
PROJECT_NAME="bilingual"
VERSION=${1:-""}
CURRENT_DIR=$(pwd)

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    print_error "pyproject.toml not found. Are you in the project root?"
    exit 1
fi

# Check for required tools
command -v python3 >/dev/null 2>&1 || { print_error "python3 is required but not installed. Aborting."; exit 1; }
command -v pip >/dev/null 2>&1 || { print_error "pip is required but not installed. Aborting."; exit 1; }

# Install build tools
print_info "Installing build tools..."
pip install --upgrade pip
pip install build twine setuptools-scm

# Clean previous builds
print_info "Cleaning previous builds..."
rm -rf dist/ build/ *.egg-info/

# Get current version if not specified
if [ -z "$VERSION" ]; then
    VERSION=$(python3 -c "import setuptools_scm; print(setuptools_scm.get_version())" 2>/dev/null || echo "0.1.0")
    print_info "Using version: $VERSION"
fi

# Update version in _version.py if needed
if [ -f "src/bilingual/_version.py" ]; then
    print_info "Updating version in _version.py..."
    sed -i.bak "s/__version__ = version = .*/__version__ = version = '$VERSION'/" src/bilingual/_version.py
    sed -i.bak "s/__version_tuple__ = version_tuple = .*/__version_tuple__ = version_tuple = $(echo $VERSION | sed 's/\./, /g' | sed 's/[^0-9, ]//g' | sed 's/,/, /g')/" src/bilingual/_version.py
fi

# Run tests
print_info "Running tests..."
python3 -m pytest tests/ -v --tb=short || {
    print_warning "Some tests failed. Continue anyway? (y/N)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        print_error "Aborting due to test failures."
        exit 1
    fi
}

# Check code quality
print_info "Running code quality checks..."
python3 -m black --check src/bilingual/ scripts/ || {
    print_warning "Code formatting issues found. Continue anyway? (y/N)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        print_info "Formatting code..."
        python3 -m black src/bilingual/ scripts/
    fi
}

python3 -m flake8 --max-line-length=100 src/bilingual/ scripts/ || {
    print_warning "Linting issues found. Continue anyway? (y/N)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        print_error "Aborting due to linting issues."
        exit 1
    fi
}

# Build the package
print_info "Building package..."
python3 -m build --wheel --sdist

# Check the built package
print_info "Checking built package..."
twine check dist/*

# Upload to PyPI (test first)
print_info "Uploading to TestPyPI..."
twine upload --repository testpypi dist/* || {
    print_warning "TestPyPI upload failed. Continue to PyPI? (y/N)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        print_info "Upload cancelled."
        exit 0
    fi
}

# Upload to PyPI
print_warning "Ready to upload to PyPI. Continue? (y/N)"
read -r response
if [[ "$response" =~ ^[Yy]$ ]]; then
    print_info "Uploading to PyPI..."
    twine upload dist/* || {
        print_error "PyPI upload failed."
        exit 1
    }
    print_success "Package uploaded to PyPI successfully!"
else
    print_info "Upload cancelled. Package built successfully in dist/ directory."
    print_info "To upload manually: twine upload dist/*"
fi

# Tag the release
if git rev-parse --git-dir > /dev/null 2>&1; then
    print_info "Creating git tag..."
    git tag -a "v$VERSION" -m "Release version $VERSION"
    print_info "To push tag: git push origin v$VERSION"
else
    print_warning "Not in a git repository. Skipping tag creation."
fi

# Summary
echo ""
echo "========================================"
echo "🎉 Release Complete!"
echo "========================================"
echo ""
echo "Package: $PROJECT_NAME"
echo "Version: $VERSION"
echo "PyPI: https://pypi.org/project/$PROJECT_NAME/$VERSION/"
echo "TestPyPI: https://test.pypi.org/project/$PROJECT_NAME/$VERSION/"
echo ""
echo "Installation:"
echo "  pip install $PROJECT_NAME==$VERSION"
echo ""
echo "Next steps:"
echo "  1. Verify installation: pip install $PROJECT_NAME==$VERSION"
echo "  2. Test import: python3 -c 'import bilingual; print(bilingual.__version__)'"
echo "  3. Update documentation with new version"
echo "  4. Push git tag: git push origin v$VERSION"
echo ""
echo "========================================"
