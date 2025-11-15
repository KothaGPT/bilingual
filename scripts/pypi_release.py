#!/usr/bin/env python3
"""
PyPI Release Script for Bilingual Package

This script handles the complete PyPI release process:
1. Builds the package
2. Uploads to TestPyPI
3. Tests installation
4. Uploads to PyPI
5. Verifies installation

Usage:
    python scripts/pypi_release.py --version 1.0.0
    python scripts/pypi_release.py --test-only
    python scripts/pypi_release.py --help
"""

import argparse
import subprocess
import sys
from pathlib import Path


class PyPIReleaser:
    """Handle PyPI release process."""

    def __init__(self, version: str = None, test_only: bool = False):
        self.version = version
        self.test_only = test_only
        self.package_name = "bilingual"

    def run_command(self, cmd: str, check: bool = True) -> subprocess.CompletedProcess:
        """Run shell command."""
        print(f"Running: {cmd}")
        return subprocess.run(cmd, shell=True, check=check, capture_output=True, text=True)

    def check_requirements(self) -> bool:
        """Check if all requirements are met."""
        print("Checking requirements...")

        # Check if we're in project root
        if not Path("pyproject.toml").exists():
            print("❌ Error: pyproject.toml not found. Are you in project root?")
            return False

        # Check Python version
        python_version = sys.version_info
        if python_version < (3, 8):
            print(
                f"❌ Error: Python {python_version.major}.{python_version.minor} not supported. Need 3.8+"
            )
            return False

        # Check git status
        try:
            result = self.run_command("git status --porcelain", check=False)
            if result.returncode != 0:
                print("❌ Error: Not in a git repository")
                return False

            if result.stdout.strip():
                print("⚠️  Warning: You have uncommitted changes")
                response = input("Continue anyway? (y/N): ")
                if response.lower() != "y":
                    return False
        except Exception:
            print("⚠️  Warning: Could not check git status")

        print("✅ Requirements check passed")
        return True

    def install_build_tools(self) -> bool:
        """Install build tools."""
        print("Installing build tools...")
        try:
            self.run_command("pip install --upgrade pip")
            self.run_command("pip install build twine setuptools-scm")
            print("✅ Build tools installed")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Error installing build tools: {e}")
            return False

    def run_tests(self) -> bool:
        """Run test suite."""
        print("Running tests...")
        try:
            self.run_command("python -m pytest tests/ -v --tb=short")
            print("✅ Tests passed")
            return True
        except subprocess.CalledProcessError:
            print("❌ Tests failed")
            response = input("Continue anyway? (y/N): ")
            return response.lower() == "y"

    def run_quality_checks(self) -> bool:
        """Run code quality checks."""
        print("Running code quality checks...")

        checks_passed = True

        # Black formatting
        try:
            self.run_command("python -m black --check src/bilingual/ scripts/")
        except subprocess.CalledProcessError:
            print("⚠️  Code formatting issues found")
            if input("Auto-format code? (y/N): ").lower() == "y":
                self.run_command("python -m black src/bilingual/ scripts/")
                print("✅ Code formatted")

        # Flake8 linting
        try:
            self.run_command("python -m flake8 --max-line-length=100 src/bilingual/ scripts/")
            print("✅ Linting passed")
        except subprocess.CalledProcessError:
            print("⚠️  Linting issues found")
            checks_passed = False

        # MyPy type checking
        try:
            self.run_command("python -m mypy src/bilingual/")
            print("✅ Type checking passed")
        except subprocess.CalledProcessError:
            print("⚠️  Type checking issues found")
            checks_passed = False

        return checks_passed

    def build_package(self) -> bool:
        """Build the package."""
        print("Building package...")

        # Clean previous builds
        self.run_command("rm -rf dist/ build/ *.egg-info/", check=False)

        try:
            self.run_command("python -m build --wheel --sdist")
            print("✅ Package built successfully")

            # Show built files
            result = self.run_command("ls -lh dist/")
            print(f"Built files:\n{result.stdout}")

            # Check package
            self.run_command("twine check dist/*")
            print("✅ Package validation passed")

            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Package build failed: {e}")
            return False

    def upload_testpypi(self) -> bool:
        """Upload to TestPyPI."""
        print("Uploading to TestPyPI...")

        try:
            self.run_command("twine upload --repository testpypi dist/*")
            print("✅ Uploaded to TestPyPI")

            test_url = f"https://test.pypi.org/project/{self.package_name}/"
            install_cmd = (
                f"pip install --index-url https://test.pypi.org/simple/ {self.package_name}"
            )

            print(f"Test URL: {test_url}")
            print(f"Test installation: {install_cmd}")

            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ TestPyPI upload failed: {e}")
            return False

    def test_installation(self) -> bool:
        """Test package installation."""
        print("Testing installation...")

        # Create virtual environment
        venv_name = ".test_venv"
        self.run_command(f"python -m venv {venv_name}", check=False)

        try:
            # Install in venv
            if sys.platform == "win32":
                pip_cmd = f"{venv_name}/Scripts/pip"
                python_cmd = f"{venv_name}/Scripts/python"
            else:
                pip_cmd = f"{venv_name}/bin/pip"
                python_cmd = f"{venv_name}/bin/python"

            self.run_command(
                f"{pip_cmd} install --index-url https://test.pypi.org/simple/ {self.package_name}"
            )
            print("✅ Test installation successful")

            # Test import
            result = subprocess.run(
                f'{python_cmd} -c \'import bilingual; print(f"Version: {{bilingual.__version__}}"); print("Import successful!")\'',
                shell=True,
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                print(f"✅ Import test passed: {result.stdout.strip()}")
                return True
            else:
                print(f"❌ Import test failed: {result.stderr}")
                return False

        except subprocess.CalledProcessError as e:
            print(f"❌ Installation test failed: {e}")
            return False
        finally:
            # Clean up venv
            self.run_command(f"rm -rf {venv_name}", check=False)

    def upload_pypi(self) -> bool:
        """Upload to PyPI."""
        if self.test_only:
            print("⏭️  Skipping PyPI upload (test-only mode)")
            return True

        print("Uploading to PyPI...")
        response = input("Are you sure you want to upload to PyPI? (y/N): ")
        if response.lower() != "y":
            print("⏭️  Upload cancelled")
            return False

        try:
            self.run_command("twine upload dist/*")
            print("✅ Uploaded to PyPI")

            pypi_url = f"https://pypi.org/project/{self.package_name}/"
            install_cmd = f"pip install {self.package_name}"

            print(f"PyPI URL: {pypi_url}")
            print(f"Installation: {install_cmd}")

            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ PyPI upload failed: {e}")
            return False

    def create_git_tag(self) -> bool:
        """Create git tag."""
        if not self.version:
            print("⏭️  Skipping git tag (no version specified)")
            return True

        print(f"Creating git tag v{self.version}...")
        try:
            self.run_command(f'git tag -a "v{self.version}" -m "Release version {self.version}"')
            print("✅ Git tag created")
            print(f"To push tag: git push origin v{self.version}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Git tag creation failed: {e}")
            return False

    def run(self) -> bool:
        """Run complete release process."""
        print("=" * 60)
        print("🚀 Starting PyPI Release Process")
        print("=" * 60)

        steps = [
            ("Check requirements", self.check_requirements),
            ("Install build tools", self.install_build_tools),
            ("Run tests", self.run_tests),
            ("Run quality checks", self.run_quality_checks),
            ("Build package", self.build_package),
            ("Upload to TestPyPI", self.upload_testpypi),
            ("Test installation", self.test_installation),
            ("Upload to PyPI", self.upload_pypi),
            ("Create git tag", self.create_git_tag),
        ]

        for step_name, step_func in steps:
            print(f"\n📋 {step_name}...")
            if not step_func():
                print(f"❌ {step_name} failed")
                response = input("Continue anyway? (y/N): ")
                if response.lower() != "y":
                    return False

        print("\n" + "=" * 60)
        print("🎉 Release process completed successfully!")
        print("=" * 60)
        print(f"\nPackage: {self.package_name}")
        if self.version:
            print(f"Version: {self.version}")
        print(f"PyPI: https://pypi.org/project/{self.package_name}/")
        print(f"TestPyPI: https://test.pypi.org/project/{self.package_name}/")
        print(f"\nInstallation: pip install {self.package_name}")
        print("\nNext steps:")
        print("  1. Verify installation: pip install bilingual")
        print("  2. Test functionality: python -c 'import bilingual; print(bilingual.__version__)'")
        print("  3. Update documentation")
        print("  4. Push git tag: git push origin v{VERSION}")
        print("=" * 60)

        return True


def main():
    parser = argparse.ArgumentParser(description="PyPI Release Script for Bilingual Package")
    parser.add_argument("--version", type=str, help="Version to release (e.g., 1.0.0)")
    parser.add_argument(
        "--test-only", action="store_true", help="Only upload to TestPyPI, not production PyPI"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be done without actually doing it"
    )

    args = parser.parse_args()

    releaser = PyPIReleaser(args.version, args.test_only)

    if args.dry_run:
        print("🔍 Dry run mode - showing what would be done:")
        print("1. Check requirements")
        print("2. Install build tools")
        print("3. Run tests")
        print("4. Run quality checks")
        print("5. Build package")
        print("6. Upload to TestPyPI")
        print("7. Test installation")
        print("8. Upload to PyPI (if not test-only)")
        print("9. Create git tag")
        return

    success = releaser.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
