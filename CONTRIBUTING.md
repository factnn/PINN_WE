# Contributing to PINN_WE

Thank you for your interest in contributing to PINN_WE! This document provides guidelines and instructions for contributing.

## Development Setup

1. **Fork and clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/PINN_WE.git
cd PINN_WE
```

2. **Create a conda environment**
```bash
conda create -n PINN_WE python=3.10
conda activate PINN_WE
```

3. **Install in development mode**
```bash
make install-dev
# OR
pip install -e ".[dev]"
pre-commit install
```

## Code Style

We follow these conventions:

- **Python Style**: PEP 8 with 100 character line length
- **Formatting**: Black + isort
- **Linting**: flake8
- **Type Hints**: Gradual typing with mypy (not enforced but encouraged)

### Running Code Quality Tools

```bash
# Format code
make format

# Check formatting without changes
make format-check

# Run linters
make lint

# Run all pre-commit hooks
pre-commit run --all-files
```

## Testing

We use pytest for testing. All new features should include tests.

### Running Tests

```bash
# Run all tests
make test

# Run fast tests only (skip slow integration tests)
make test-fast

# Run with coverage report
make coverage
```

### Writing Tests

- Place tests in `tests/` directory
- Name test files `test_*.py`
- Use descriptive test function names: `test_model_forward_pass_shape()`
- Use fixtures for common setup (see `tests/conftest.py`)
- Mark slow tests with `@pytest.mark.slow`
- Mark CUDA-required tests with `@pytest.mark.cuda`

Example:
```python
import pytest

@pytest.mark.cuda
def test_model_on_gpu(model, device):
    """Test model runs on GPU."""
    model = model.to(device)
    x = torch.randn(10, 2).to(device)
    output = model(x)
    assert output.is_cuda
```

## Commit Messages

Follow conventional commit format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding/updating tests
- `chore`: Maintenance tasks

Example:
```
feat(models): add 2D Navier-Stokes PINN

Implement physics-informed neural network for 2D incompressible
Navier-Stokes equations with pressure Poisson formulation.

Closes #42
```

## Pull Request Process

1. **Create a feature branch**
```bash
git checkout -b feature/your-feature-name
```

2. **Make your changes**
   - Write code following our style guide
   - Add tests for new functionality
   - Update documentation as needed

3. **Run tests and checks**
```bash
make test
make lint
make format-check
```

4. **Commit your changes**
```bash
git add .
git commit -m "feat: your feature description"
```

5. **Push to your fork**
```bash
git push origin feature/your-feature-name
```

6. **Create a Pull Request**
   - Provide clear description of changes
   - Reference related issues
   - Ensure all CI checks pass

## Pull Request Review

Your PR will be reviewed for:
- Code quality and style
- Test coverage
- Documentation completeness
- Backward compatibility
- Performance implications

## Adding New Test Cases

When adding a new physics problem:

1. Create directory in `cases/1D/` or `cases/2D/`
2. Include numbered scripts (0.py, 1.py, etc.)
3. Add exact solution data if available
4. Update `CASES_README.md` with problem description
5. Add unit test in `tests/test_cases.py`

## Documentation

- Update README.md for user-facing changes
- Add docstrings to new functions/classes (Google style preferred)
- Update relevant .md files in docs/

Example docstring:
```python
def solve_riemann_problem(u_left, u_right, gamma=1.4):
    """
    Solve 1D Riemann problem for Euler equations.
    
    Args:
        u_left: Left state vector (rho, u, p)
        u_right: Right state vector (rho, u, p)
        gamma: Specific heat ratio (default: 1.4)
    
    Returns:
        Solution dictionary containing density, velocity, pressure
    
    Raises:
        ValueError: If states contain negative values
    """
```

## Reporting Bugs

Use GitHub Issues with the bug template:

- Clear, descriptive title
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, CUDA version, etc.)
- Minimal code example

## Requesting Features

Use GitHub Issues with feature request template:

- Problem description
- Proposed solution
- Alternative solutions considered
- Additional context

## Questions?

- Open a GitHub Discussion for general questions
- Check existing issues/PRs for similar topics
- Reach out to maintainers

## Code of Conduct

Be respectful, inclusive, and professional. We're all here to advance scientific computing!

Thank you for contributing! 🚀
