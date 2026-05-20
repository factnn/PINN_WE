.PHONY: help install install-dev test lint format clean docs

help:
	@echo "PINN_WE Development Commands"
	@echo "============================="
	@echo "install       - Install package in production mode"
	@echo "install-dev   - Install package with development dependencies"
	@echo "test          - Run tests with pytest"
	@echo "test-fast     - Run tests without slow tests"
	@echo "coverage      - Run tests with coverage report"
	@echo "lint          - Run linters (flake8, mypy)"
	@echo "format        - Format code with black and isort"
	@echo "format-check  - Check formatting without modifying files"
	@echo "clean         - Remove build artifacts and cache files"
	@echo "docs          - Build documentation"
	@echo "pre-commit    - Install pre-commit hooks"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

test:
	pytest tests/ -v

test-fast:
	pytest tests/ -v -m "not slow"

coverage:
	pytest tests/ --cov=PINNsrc --cov-report=html --cov-report=term

lint:
	flake8 PINNsrc/ tests/ --max-line-length=100 --extend-ignore=E203,W503
	mypy PINNsrc/ --ignore-missing-imports --allow-untyped-defs

format:
	black PINNsrc/ tests/ cases/ --line-length=100
	isort PINNsrc/ tests/ cases/

format-check:
	black PINNsrc/ tests/ cases/ --check --line-length=100
	isort PINNsrc/ tests/ cases/ --check-only

clean:
	rm -rf build/ dist/ *.egg-info
	rm -rf .pytest_cache/ .coverage htmlcov/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

docs:
	@echo "Documentation build not yet configured"
	@echo "Run: sphinx-quickstart docs/ to initialize"

pre-commit:
	pre-commit install
	pre-commit run --all-files
