---
original: /docs/DEVELOPER_GUIDE.md
translated: 2025-10-24
---

> **বিঃদ্রঃ** এটি একটি স্বয়ংক্রিয়ভাবে অনুবাদকৃত নথি। মূল ইংরেজি সংস্করণের জন্য [এখানে ক্লিক করুন](/{rel_path}) করুন।

---

# Developer Guide

This guide provides detailed information for developers contributing to the Bilingual project.

## Table of Contents
1. [Getting Started](#getting-started)
2. [Contribution Guidelines](#contribution-guidelines)
3. [Code Review Process](#code-review-process)
4. [Testing Guide](#testing-guide)
5. [Debugging](#debugging)
6. [Release Process](#release-process)

## Getting Started

### Prerequisites
- Python 3.9+
- Git
- [Poetry](https://python-poetry.org/) (for dependency management)
- [pre-commit](https://pre-commit.com/)

### Setup Development Environment

1. Fork and clone the repository:
   ```bash
   git clone https://github.com/your-username/bilingual.git
   cd bilingual
   ```

2. Install development dependencies:
   ```bash
   poetry install --with dev
   pre-commit install
   ```

3. Run tests to verify setup:
   ```bash
   pytest tests/
   ```

## Contribution Guidelines

### Code Style
- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) for Python code
- Use type hints for all function signatures
- Include docstrings following [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- Keep lines under 100 characters

### Git Workflow

1. Create a new branch for your feature/fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit with a descriptive message:
   ```bash
   git commit -m "Add new feature for X"
   ```

3. Push to your fork and create a pull request

### Pull Request Guidelines
- Reference related issues in your PR description
- Ensure all tests pass
- Update documentation as needed
- Keep PRs focused and small when possible
- Request reviews from relevant team members

## Code Review Process

### Review Checklist
- [ ] Code follows style guidelines
- [ ] Tests are included and pass
- [ ] Documentation is updated
- [ ] Type hints are used correctly
- [ ] Error handling is appropriate
- [ ] Performance considerations are addressed

### Review Process
1. Create a draft PR for early feedback
2. Request reviews from at least 2 team members
3. Address all review comments
4. Update the PR with requested changes
5. Once approved, squash and merge

## Testing Guide

### Running Tests

#### Run all tests:
```bash
pytest
```

#### Run a specific test file:
```bash
pytest tests/test_module.py
```

#### Run with coverage:
```bash
pytest --cov=bilingual --cov-report=term-missing
```

### Writing Tests
- Place tests in the `tests/` directory
- Follow the `test_*.py` naming convention
- Use descriptive test names
- Test both success and error cases
- Use fixtures for common test data

## Debugging

### Common Issues

#### CUDA Out of Memory
```python
# Reduce batch size
config["training"]["batch_size"] = 16

# Enable gradient accumulation
trainer = Trainer(gradient_accumulation_steps=4)
```

#### Installation Issues
```bash
# Clear pip cache
pip cache purge

# Reinstall in development mode
pip install -e .
```

### Debugging Tools
- Use `pdb` for interactive debugging:
  ```python
  import pdb; pdb.set_trace()
  ```
- Use `logging` for runtime information:
  ```python
  import logging
  logging.basicConfig(level=logging.DEBUG)
  ```

## Release Process

### Versioning
We follow [Semantic Versioning](https://semver.org/).

### Steps for New Release
1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create a release tag:
   ```bash
   git tag -a v1.0.0 -m "v1.0.0: Major release"
   git push origin v1.0.0
   ```
4. Create a GitHub release with release notes
5. Publish to PyPI:
   ```bash
   rm -rf dist/*
   poetry build
   poetry publish
   ```

## License
By contributing to this project, you agree that your contributions will be licensed under its [MIT License](LICENSE).
