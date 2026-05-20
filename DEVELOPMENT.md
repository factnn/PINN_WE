# Development Guide

This guide provides detailed information for developers working on PINN_WE.

## Project Architecture

### Directory Structure

```
PINN_WE/
├── PINNsrc/              # Core library
│   ├── __init__.py
│   ├── PINNs.py          # Model implementations
│   ├── utility.py        # Helper functions
│   ├── riemann_solver.py # Triton kernels
│   ├── IC_1D.py          # 1D initial conditions
│   ├── IC_2D.py          # 2D initial conditions
│   ├── BC_1D.py          # 1D boundary conditions
│   └── BC_2D.py          # 2D boundary conditions
├── cases/                # Test problems
│   ├── 1D/
│   └── 2D/
├── tests/                # Unit tests
│   ├── __init__.py
│   ├── conftest.py       # Pytest configuration
│   ├── test_models.py
│   └── test_utility.py
├── output/               # Training results (gitignored)
├── docs/                 # Documentation
└── .github/              # GitHub workflows and templates
```

## Code Organization

### Model Classes

**PINNsrc/PINNs.py** contains:
- `PINNs_WE_Euler_1D`: Weakly-enforced 1D Euler equations
- `PINNs_Euler_1D`: Strongly-enforced 1D Euler equations
- `PINNs_WE_Euler_2D`: Weakly-enforced 2D Euler equations
- Helper functions: `gradients()`, initialization utilities

### Utility Functions

**PINNsrc/utility.py** provides:
- `select_gpu()`: Auto-select least-used GPU
- `save_results()`: Unified result saving
- Visualization functions
- Training callbacks

### Physics Modules

- **IC_1D.py / IC_2D.py**: Initial condition generators
- **BC_1D.py / BC_2D.py**: Boundary condition handlers
- **riemann_solver.py**: GPU-accelerated exact solvers

## Development Workflow

### 1. Environment Setup

```bash
# Create environment
conda create -n PINN_WE python=3.10
conda activate PINN_WE

# Install in editable mode
pip install -e ".[dev]"

# Setup pre-commit
pre-commit install
```

### 2. Making Changes

```bash
# Create feature branch
git checkout -b feature/my-feature

# Make changes
vim PINNsrc/PINNs.py

# Format code
make format

# Run tests
make test

# Check linting
make lint
```

### 3. Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_models.py -v

# Run specific test
pytest tests/test_models.py::TestPINNs1D::test_forward_pass_shape -v

# Run with coverage
pytest --cov=PINNsrc --cov-report=html

# Skip slow tests
pytest -m "not slow"

# Run only CUDA tests
pytest -m cuda
```

### 4. Code Quality

```bash
# Format with black
black PINNsrc/ tests/ --line-length=100

# Sort imports
isort PINNsrc/ tests/

# Lint with flake8
flake8 PINNsrc/ tests/ --max-line-length=100

# Type check with mypy
mypy PINNsrc/ --ignore-missing-imports

# Or use make
make format
make lint
```

### 5. Pre-commit Hooks

Hooks run automatically on `git commit`:
- Trailing whitespace removal
- End-of-file fixer
- YAML syntax check
- Large file detection
- Black formatting
- isort
- flake8
- mypy

Skip hooks (not recommended):
```bash
git commit --no-verify
```

## Adding New Features

### Adding a New Physics Problem

1. **Create case directory**
```bash
mkdir -p cases/1D/MyProblem
```

2. **Implement training script**
```python
# cases/1D/MyProblem/0.py
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from PINNsrc.PINNs import PINNs_WE_Euler_1D
from PINNsrc.utility import select_gpu, save_results

# Define initial conditions
def initial_condition(x):
    # Your IC here
    pass

# Setup model
model = PINNs_WE_Euler_1D(...)

# Train
# ...

# Save results
save_results(...)
```

3. **Add exact solution (if available)**
```bash
# Store in cases/1D/MyProblem/exact_solution.dat
```

4. **Document in CASES_README.md**

5. **Add unit test**
```python
# tests/test_cases.py
def test_my_problem():
    # Test your case
    pass
```

### Adding a New Model

1. **Implement model class**
```python
# PINNsrc/PINNs.py
class MyNewPINN(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # ...
    
    def forward(self, x):
        # ...
        return output
    
    def loss_pde(self, x):
        # Physics-informed loss
        # ...
        return loss
```

2. **Add tests**
```python
# tests/test_models.py
class TestMyNewPINN:
    def test_forward_pass(self):
        # ...
        pass
```

3. **Update documentation**

## Performance Optimization

### Profiling

```python
import torch.profiler as profiler

with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
    record_shapes=True
) as prof:
    # Training code
    pass

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

### Memory Optimization

```python
# Use gradient checkpointing for large networks
from torch.utils.checkpoint import checkpoint

# Mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    loss = model.loss_pde(x)
```

### Multi-GPU Training

```python
# Use DataParallel
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

# Or DistributedDataParallel (better performance)
from torch.nn.parallel import DistributedDataParallel as DDP
model = DDP(model, device_ids=[local_rank])
```

## Debugging Tips

### GPU Memory Issues

```python
# Monitor GPU memory
import torch
print(torch.cuda.memory_summary())

# Clear cache
torch.cuda.empty_cache()

# Enable memory debugging
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
```

### NaN/Inf Detection

```python
# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)

# Check for NaN
if torch.isnan(loss):
    print("NaN detected!")
    breakpoint()
```

### Visualization During Training

```python
# Use TensorBoard
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/experiment_1')
writer.add_scalar('Loss/train', loss, epoch)
writer.add_figure('Predictions', fig, epoch)
```

## Release Process

1. **Update version in pyproject.toml**
2. **Update CHANGELOG.md**
3. **Ensure all tests pass**
4. **Create git tag**
```bash
git tag -a v0.2.0 -m "Release version 0.2.0"
git push origin v0.2.0
```
5. **Create GitHub release**

## Useful Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Triton Language](https://triton-lang.org/)
- [Physics-Informed Neural Networks](https://maziarraissi.github.io/PINNs/)
- [SMT Documentation](https://smt.readthedocs.io/)

## Getting Help

- Check existing issues on GitHub
- Read the documentation
- Ask in GitHub Discussions
- Contact maintainers

Happy coding! 🚀
