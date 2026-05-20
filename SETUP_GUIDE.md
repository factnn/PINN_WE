# Quick Setup Guide

This guide will help you get started with the modernized PINN_WE project.

## Prerequisites

- Linux system (tested on Ubuntu)
- NVIDIA GPU with CUDA support (recommended)
- Conda or Python 3.10+

## Installation

### Option 1: Using Conda (Recommended)

```bash
# 1. Navigate to project directory
cd /share/project/zpy/PINN_WE

# 2. Activate the PINN_WE environment
source /share/project/zhaohuxing/anaconda3/bin/activate PINN_WE

# 3. Install the package in development mode
pip install -e ".[dev]"

# 4. Setup pre-commit hooks
pre-commit install
```

### Option 2: Using venv

```bash
# 1. Navigate to project directory
cd /share/project/zpy/PINN_WE

# 2. Create virtual environment
python3.10 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements-dev.txt

# 4. Install package in development mode
pip install -e .

# 5. Setup pre-commit hooks
pre-commit install
```

## Verify Installation

```bash
# Test imports
python -c "from PINNsrc import PINNs_WE_Euler_1D; print('Success!')"

# Run tests
pytest tests/ -v

# Check code quality
make lint
```

## Running Your First Example

```bash
# 1. Navigate to a test case
cd cases/1D/Blast

# 2. Run training script
python 1.py

# 3. Check results in output directory
ls -lh output/
```

## Development Workflow

### Making Changes

```bash
# 1. Create a feature branch
git checkout -b feature/my-improvement

# 2. Make your changes
vim PINNsrc/PINNs.py

# 3. Format code automatically
make format

# 4. Run tests
make test

# 5. Check code quality
make lint

# 6. Commit (pre-commit hooks will run automatically)
git add .
git commit -m "feat: add my improvement"
```

### Available Make Commands

```bash
make help          # Show all available commands
make install       # Install package
make install-dev   # Install with dev dependencies
make test          # Run all tests
make test-fast     # Run fast tests only
make coverage      # Run tests with coverage report
make lint          # Run linters (flake8, mypy)
make format        # Format code (black, isort)
make format-check  # Check formatting without changes
make clean         # Remove build artifacts
```

## Common Tasks

### Running Specific Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_models.py

# Run specific test function
pytest tests/test_models.py::TestPINNs1D::test_forward_pass_shape

# Run with coverage
pytest --cov=PINNsrc --cov-report=html

# Skip slow tests
pytest -m "not slow"

# Run only CUDA tests
pytest -m cuda
```

### Code Formatting

```bash
# Format all code
make format

# Check formatting without modifying
make format-check

# Format specific file
black PINNsrc/PINNs.py --line-length=100
```

### Linting

```bash
# Run all linters
make lint

# Run flake8 only
flake8 PINNsrc/ tests/ --max-line-length=100

# Run mypy only
mypy PINNsrc/ --ignore-missing-imports
```

## Project Structure

```
PINN_WE/
├── PINNsrc/              # Main package
│   ├── __init__.py       # Package initialization
│   ├── PINNs.py          # Model implementations
│   ├── utility.py        # Utility functions
│   └── ...
├── cases/                # Test cases
│   ├── 1D/               # 1D problems
│   └── 2D/               # 2D problems
├── tests/                # Unit tests
├── output/               # Training results (generated)
├── pyproject.toml        # Package configuration
├── Makefile              # Development commands
└── README.md             # Project documentation
```

## Troubleshooting

### Import Errors

If you get import errors:
```bash
# Ensure package is installed
pip install -e .

# Check Python path
python -c "import sys; print(sys.path)"
```

### GPU Issues

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# List available GPUs
python -c "import torch; print(torch.cuda.device_count())"
```

### Pre-commit Hook Issues

```bash
# Update pre-commit hooks
pre-commit autoupdate

# Run hooks manually
pre-commit run --all-files

# Skip hooks temporarily (not recommended)
git commit --no-verify
```

## Getting Help

- Read the [README.md](README.md) for project overview
- Check [DEVELOPMENT.md](DEVELOPMENT.md) for detailed developer guide
- See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines
- Open an issue on GitHub for bugs or questions

## Next Steps

1. **Explore Examples**: Check out the test cases in `cases/`
2. **Read Documentation**: Browse through the markdown files
3. **Run Tests**: Familiarize yourself with the test suite
4. **Start Contributing**: Pick an issue and submit a PR!

Happy coding! 🚀
