#!/bin/bash
# Production Release and Distribution Script for Bilingual NLP Toolkit
#
# This script handles the complete release process:
# 1. Version bumping and tagging
# 2. Package building and testing
# 3. PyPI upload (Test and Production)
# 4. Docker image building and pushing
# 5. Documentation deployment
# 6. Release notes generation

set -e  # Exit on any error

# Configuration
PROJECT_NAME="bilingual"
GITHUB_REPO="kothagpt/bilingual"
DOCKER_REGISTRY="ghcr.io"
DOCKER_IMAGE="${DOCKER_REGISTRY}/${GITHUB_REPO}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

log_section() {
    echo -e "${PURPLE}[SECTION]${NC} $*"
}

# Check requirements
check_requirements() {
    log_section "Checking Release Requirements"

    # Check if we're in a git repository
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        log_error "Not in a git repository. Please run this from the project root."
        exit 1
    fi

    # Check if we're on the main branch
    if [[ "$(git branch --show-current)" != "main" ]]; then
        log_warning "Not on main branch. Consider switching to main for release."
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi

    # Check for uncommitted changes
    if [[ -n "$(git status --porcelain)" ]]; then
        log_error "There are uncommitted changes. Please commit or stash them first."
        exit 1
    fi

    # Check required tools
    local missing_tools=()
    command -v python3 &> /dev/null || missing_tools+=("python3")
    command -v git &> /dev/null || missing_tools+=("git")
    command -v docker &> /dev/null || missing_tools+=("docker")

    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        exit 1
    fi

    # Check Python packages
    if ! python3 -c "import build, twine" &> /dev/null; then
        log_error "Required Python packages not installed. Run: pip install build twine"
        exit 1
    fi

    log_success "All requirements satisfied"
}

# Version management
manage_version() {
    log_section "Version Management"

    # Get current version
    if [[ -f "bilingual/_version.py" ]]; then
        current_version=$(grep -o '__version__ = ".*"' bilingual/_version.py | grep -o '[0-9]\+\.[0-9]\+\.[0-9]\+')
    else
        current_version="0.1.0"
    fi

    log_info "Current version: $current_version"

    # Ask for new version
    read -p "Enter new version (or press Enter for patch increment): " new_version

    if [[ -z "$new_version" ]]; then
        # Auto-increment patch version
        IFS='.' read -r major minor patch <<< "$current_version"
        new_version="$major.$minor.$((patch + 1))"
        log_info "Auto-incrementing to version: $new_version"
    fi

    # Validate version format
    if ! [[ $new_version =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        log_error "Invalid version format. Use: X.Y.Z (e.g., 1.2.3)"
        exit 1
    fi

    # Update version in files
    log_info "Updating version to $new_version..."

    # Update pyproject.toml
    sed -i.bak "s/dynamic = \[\"version\"\]/version = \"$new_version\"/" pyproject.toml

    # Update __init__.py
    sed -i.bak "s/__version__ = \".*\"/__version__ = \"$new_version\"/" src/bilingual/__init__.py

    # Update server.py
    sed -i.bak "s/version=\"1\.0\.0\"/version=\"$new_version\"/" src/bilingual/server.py

    log_success "Version updated to $new_version"

    # Commit version changes
    git add pyproject.toml src/bilingual/__init__.py src/bilingual/server.py
    git commit -m "chore: bump version to $new_version"
    git tag -a "v$new_version" -m "Release version $new_version"

    log_success "Version changes committed and tagged"
}

# Build and test package
build_and_test() {
    log_section "Building and Testing Package"

    log_info "Building package..."
    python3 -m build

    log_info "Checking package..."
    twine check dist/*

    log_info "Installing package for testing..."
    pip install -e .

    log_info "Running tests..."
    python3 -m pytest tests/ -v --tb=short

    log_success "Package built and tested successfully"
}

# Upload to PyPI
upload_to_pypi() {
    log_section "PyPI Distribution"

    # Upload to Test PyPI first
    if [[ "$1" == "test" ]]; then
        log_info "Uploading to Test PyPI..."
        TWINE_USERNAME="__token__" TWINE_PASSWORD="$TEST_PYPI_TOKEN" twine upload --repository testpypi dist/*
        log_success "✅ Uploaded to Test PyPI"
        log_info "📦 Test package available at: https://test.pypi.org/project/bilingual/"
    else
        # Upload to Production PyPI
        log_info "Uploading to Production PyPI..."
        if [[ -z "$PYPI_TOKEN" ]]; then
            log_error "PYPI_TOKEN environment variable not set"
            log_info "Set PYPI_TOKEN to your PyPI API token"
            exit 1
        fi

        TWINE_USERNAME="__token__" TWINE_PASSWORD="$PYPI_TOKEN" twine upload dist/*
        log_success "✅ Uploaded to Production PyPI"
        log_info "📦 Package available at: https://pypi.org/project/bilingual/"
    fi
}

# Build and push Docker image
build_docker_image() {
    log_section "Docker Image Management"

    if [[ -z "$GITHUB_TOKEN" ]]; then
        log_error "GITHUB_TOKEN environment variable not set"
        log_info "Set GITHUB_TOKEN for Docker registry access"
        exit 1
    fi

    # Login to GitHub Container Registry
    echo "$GITHUB_TOKEN" | docker login ghcr.io -u "$GITHUB_USERNAME" --password-stdin

    # Build and tag image
    docker build -t "${DOCKER_IMAGE}:latest" .
    docker tag "${DOCKER_IMAGE}:latest" "${DOCKER_IMAGE}:v$new_version"

    # Push images
    log_info "Pushing Docker images..."
    docker push "${DOCKER_IMAGE}:latest"
    docker push "${DOCKER_IMAGE}:v$new_version"

    log_success "✅ Docker images pushed to GitHub Container Registry"
    log_info "🐳 Images available at: https://github.com/${GITHUB_REPO}/pkgs/container/bilingual"
}

# Deploy documentation
deploy_docs() {
    log_section "Documentation Deployment"

    if command -v mkdocs &> /dev/null; then
        log_info "Building and deploying documentation..."

        # Install docs dependencies if needed
        pip install mkdocs mkdocs-material mkdocs-i18n mkdocstrings

        # Build and deploy docs
        mkdocs build

        # Deploy to GitHub Pages (if configured)
        if [[ -d "site" ]]; then
            log_success "✅ Documentation built successfully"
            log_info "📚 Local docs available at: ./site/index.html"
        fi
    else
        log_warning "MkDocs not available - skipping documentation build"
    fi
}

# Generate release notes
generate_release_notes() {
    log_section "Release Notes Generation"

    local release_notes="RELEASE_NOTES.md"

    cat > "$release_notes" << EOF
# Release Notes - Bilingual v$new_version

## What's New

### 🚀 **Production Ready**
- ✅ FastAPI server for production deployment
- ✅ Docker containerization with multi-stage builds
- ✅ ONNX model optimization for lightweight inference
- ✅ Comprehensive monitoring and telemetry
- ✅ GitHub Actions CI/CD pipeline

### 📦 **Enhanced Distribution**
- ✅ PyPI packaging with semantic versioning
- ✅ GitHub Container Registry integration
- ✅ Automated release workflows
- ✅ Production deployment scripts

### 🛠️ **Developer Experience**
- ✅ Rich CLI interface with Typer + Rich
- ✅ Pydantic configuration management
- ✅ Comprehensive testing framework
- ✅ Interactive documentation (MkDocs Material)

## Installation

### PyPI (Recommended)
\`\`\`bash
pip install bilingual==$new_version
\`\`\`

### Docker
\`\`\`bash
docker run -p 8000:8000 ghcr.io/$GITHUB_REPO:v$new_version
\`\`\`

### Development
\`\`\`bash
git clone https://github.com/$GITHUB_REPO.git
cd bilingual
pip install -e .
\`\`\`

## Quick Start

\`\`\`python
import bilingual as bb

# Language detection
result = bb.detect_language("আমি স্কুলে যাই।")
print(f"Language: {result['language']}")

# Translation
translation = bb.translate_text("t5-small", "Hello world", "en", "bn")
print(f"Translation: {translation}")

# API Server
# python3 -m bilingual.server --host 0.0.0.0 --port 8000
\`\`\`

## API Documentation

- 🌐 **Interactive API Docs**: http://localhost:8000/docs
- 📚 **Full Documentation**: https://bilingual.readthedocs.io
- 🐳 **Docker Images**: https://github.com/$GITHUB_REPO/pkgs/container/bilingual

## Breaking Changes

None in this release.

## Contributors

Thanks to all contributors who made this release possible!

## Support

- 📖 **Documentation**: https://bilingual.readthedocs.io
- 🐛 **Issues**: https://github.com/$GITHUB_REPO/issues
- 💬 **Discussions**: https://github.com/$GITHUB_REPO/discussions

---

*Released on $(date +%Y-%m-%d)*
EOF

    log_success "Release notes generated: $release_notes"
}

# Push changes to repository
push_to_repository() {
    log_section "Repository Management"

    # Push commits and tags
    log_info "Pushing changes to repository..."
    git push origin main
    git push origin "v$new_version"

    log_success "✅ Changes pushed to repository"
}

# Main release function
main() {
    echo "🚀 Bilingual NLP Toolkit - Production Release"
    echo "=========================================="

    # Check if version should be auto-detected from environment
    if [[ -n "$RELEASE_VERSION" ]]; then
        new_version="$RELEASE_VERSION"
        log_info "Using version from environment: $new_version"
    else
        # Get version from user or auto-increment
        manage_version
    fi

    # Run release steps
    check_requirements
    build_and_test

    # Ask about PyPI upload
    read -p "Upload to PyPI? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        read -p "Upload to Test PyPI first? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            upload_to_pypi "test"
        fi

        read -p "Upload to Production PyPI? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            upload_to_pypi "production"
        fi
    fi

    # Docker image (optional)
    read -p "Build and push Docker image? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        build_docker_image
    fi

    # Documentation (optional)
    read -p "Deploy documentation? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        deploy_docs
    fi

    # Generate release notes
    generate_release_notes

    # Push to repository (if not already done)
    read -p "Push changes to repository? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        push_to_repository
    fi

    echo ""
    log_success "🎉 Release v$new_version completed successfully!"
    echo ""
    echo "📋 Release Summary:"
    echo "   Version: v$new_version"
    echo "   Package: https://pypi.org/project/bilingual/$new_version/"
    echo "   Docker: ghcr.io/$GITHUB_REPO:v$new_version"
    echo "   Docs: https://bilingual.readthedocs.io"
    echo "   Repository: https://github.com/$GITHUB_REPO"
    echo ""
    echo "🔧 Next steps:"
    echo "1. Monitor the release on GitHub"
    echo "2. Update any deployment environments"
    echo "3. Announce the release to users"
    echo "4. Monitor for any issues in production"
}

# Help function
show_help() {
    echo "Bilingual NLP Toolkit - Release Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --help, -h          Show this help message"
    echo "  --version VERSION   Specify version (auto-increment if not provided)"
    echo "  --test-only         Only run tests, don't release"
    echo "  --pypi-only         Only upload to PyPI"
    echo "  --docker-only       Only build Docker image"
    echo ""
    echo "Environment Variables:"
    echo "  PYPI_TOKEN          PyPI API token for production upload"
    echo "  TEST_PYPI_TOKEN     Test PyPI API token"
    echo "  GITHUB_TOKEN        GitHub token for Docker registry"
    echo "  GITHUB_USERNAME     GitHub username for Docker registry"
    echo "  RELEASE_VERSION     Override version detection"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Interactive release"
    echo "  $0 --version 1.2.3                   # Release specific version"
    echo "  PYPI_TOKEN=... $0                     # Release with PyPI upload"
    echo "  $0 --test-only                        # Only run tests"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            show_help
            exit 0
            ;;
        --version)
            RELEASE_VERSION="$2"
            shift 2
            ;;
        --test-only)
            TEST_ONLY=true
            shift
            ;;
        --pypi-only)
            PYPI_ONLY=true
            shift
            ;;
        --docker-only)
            DOCKER_ONLY=true
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Run appropriate mode
if [[ "$TEST_ONLY" == "true" ]]; then
    log_section "Test Mode"
    check_requirements
    build_and_test
    log_success "✅ Tests completed successfully"
elif [[ "$PYPI_ONLY" == "true" ]]; then
    log_section "PyPI Only Mode"
    check_requirements
    build_and_test
    upload_to_pypi "production"
    log_success "✅ PyPI upload completed"
elif [[ "$DOCKER_ONLY" == "true" ]]; then
    log_section "Docker Only Mode"
    build_docker_image
    log_success "✅ Docker image built and pushed"
else
    # Full release
    main
fi
