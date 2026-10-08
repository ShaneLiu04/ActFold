# Contributing to ActFold

Thank you for your interest in contributing to ActFold! This document provides guidelines for code contributions, reporting issues, and proposing changes.

## Getting Started

1. Fork the repository and clone your fork.
2. Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt
   pip install -e .
   ```
3. Run the test suite to ensure a clean baseline:
   ```bash
   python -m pytest tests/ -q
   ```

## Development Workflow

1. Create a new branch for your feature or bug fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes, following the coding conventions below.
3. Add or update tests for any new behavior.
4. Update documentation (`README.md`, `docs/`, `CHANGELOG.md`) as needed.
5. Run quality checks locally:
   ```bash
   python -m black actfold tests demo.py scripts
   python -m isort actfold tests demo.py scripts
   python -m pyflakes actfold tests demo.py scripts
   python -m mypy actfold --ignore-missing-imports
   python -m pytest tests/ -q
   python demo.py
   ```
6. Open a pull request with a clear description of the change and motivation.

## Coding Conventions

- Python 3.10+ with `from __future__ import annotations`.
- Use type hints for all public APIs.
- Follow PEP 8, with `black` formatting at 100 characters per line.
- Keep functions focused and modular.
- Write docstrings for classes and public methods.
- Avoid hard-coding magic numbers; use `ActFoldConfig` or module-level constants.

## Testing

- Add unit tests for new utility functions and classes.
- Add integration tests for new end-to-end workflows.
- Ensure all tests pass on both CPU and CUDA when possible.
- Use the `device` fixture for device-agnostic tests.

## Reporting Issues

When reporting bugs, please include:

- A minimal reproduction script.
- The output of `python -m pytest tests/ -q`.
- Your Python, PyTorch, and transformers versions.
- Whether you are running on CPU or CUDA.

## Code of Conduct

Be respectful, constructive, and inclusive. We welcome contributors from all backgrounds.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
