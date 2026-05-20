# PINN_WE - Physics-Informed Neural Networks for Compressible Euler Equations

A PyTorch-based implementation of Physics-Informed Neural Networks (PINNs) for solving compressible Euler equations, with support for 1D and 2D shock problems.

## Features

- **Physics-Informed Training**: Enforces conservation laws and boundary conditions through custom loss functions
- **Adaptive Weighting**: Implements adaptive k-parameter for compression zone detection
- **GPU Acceleration**: Multi-GPU support with automatic GPU selection
- **Triton Kernels**: Optimized Riemann solvers using Triton for high performance
- **Comprehensive Test Cases**: 
  - 1D: Sod, Lax, Shu-Osher, Blast, Burgers
  - 2D: Vortex, Riemann2D, Shock-Circle interactions

## Installation

### Requirements
- Python >= 3.10
- CUDA-capable GPU (tested on NVIDIA A100)
- Anaconda/Miniconda (recommended)

### Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/factnn/PINN_WE.git
cd PINN_WE
```

2. **Create conda environment**
```bash
conda create -n PINN_WE python=3.10
conda activate PINN_WE
```

3. **Install dependencies**
```bash
pip install torch numpy scipy matplotlib smt triton
```

Or install from `pyproject.toml`:
```bash
pip install -e .
```

### Development Setup

Install with development dependencies:
```bash
pip install -e ".[dev]"
pre-commit install
```

## Usage

### Running Test Cases

**1D Sod Shock Tube:**
```bash
cd cases/1D/Blast
python 1.py
```

**2D Vortex Problem:**
```bash
cd cases/2D/Vortex
python 0.py
```

### Training Custom Models

```python
from PINNsrc.PINNs import PINNs_WE_Euler_1D
from PINNsrc.utility import select_gpu, save_results

# Auto-select GPU
gpu_id = select_gpu()
torch.cuda.set_device(gpu_id)

# Initialize model
model = PINNs_WE_Euler_1D(
    input_dim=2,      # (x, t)
    output_dim=3,     # (rho, u, p)
    hidden_layers=8,
    hidden_units=40
)

# Train with two-stage optimization
optimizer1 = torch.optim.Adam(model.parameters(), lr=0.001)
# ... training loop ...
optimizer2 = torch.optim.LBFGS(model.parameters(), lr=0.1)
# ... refinement loop ...

# Save results
save_results(
    case_name='MyCase',
    model=model,
    pred_dict={'x': x, 'rho': rho, 'p': p, 'u': u},
    save_model=True,
    save_plot=True
)
```

## Project Structure

```
PINN_WE/
├── PINNsrc/              # Core library
│   ├── PINNs.py          # Model implementations
│   ├── utility.py        # GPU selection, visualization, I/O
│   ├── riemann_solver.py # Triton-optimized solvers
│   ├── IC_1D.py / IC_2D.py   # Initial conditions
│   └── BC_1D.py / BC_2D.py   # Boundary conditions
├── cases/                # Test cases
│   ├── 1D/               # 1D problems
│   └── 2D/               # 2D problems
├── output/               # Training results (auto-generated)
└── docs/                 # Documentation
```

## Key Components

### Physics-Informed Loss

The model minimizes a composite loss function:
```
L = L_pde + L_ic + L_bc
```

- **L_pde**: PDE residuals (conservation laws)
- **L_ic**: Initial condition matching
- **L_bc**: Boundary condition enforcement

### Adaptive K-Parameter

Compression zones are detected using:
```python
d = 1 / (k * (abs(u_x) - u_x) + 1)
```

See `ADAPTIVE_K_ANALYSIS.md` for detailed analysis.

## Hardware Requirements

- **Minimum**: 1x NVIDIA GPU with 8GB VRAM
- **Recommended**: 1-8x NVIDIA A100 (40GB) for large-scale problems
- **CPU**: Multi-core processor for data preprocessing

## Documentation

- `COMMANDS.md` - Installation and usage commands
- `CASES_README.md` - Description of test cases
- `TRAINING_GUIDE.md` - Training best practices
- `DATA_FUSION_GUIDE.md` - Data-driven PINN approaches
- `ADAPTIVE_K_ANALYSIS.md` - Compression zone detection

## Testing

Run unit tests:
```bash
pytest
```

Run Burgers solver benchmark:
```bash
cd PINNsrc
python test_burgers.py
```

## Performance

Typical training times (on A100):
- 1D Sod problem: ~30-60 minutes (100k epochs)
- 2D Vortex: ~2-4 hours (50k epochs)

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Code Quality

This project uses:
- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking
- **pytest** for testing

Run all checks:
```bash
pre-commit run --all-files
```

## License

MIT License - see LICENSE file for details

## Citation

If you use this code in your research, please cite:

```bibtex
@software{pinn_we_2026,
  title={PINN_WE: Physics-Informed Neural Networks for Compressible Euler Equations},
  author={PINN_WE Contributors},
  year={2026},
  url={https://github.com/factnn/PINN_WE}
}
```

## Acknowledgments

- Built with PyTorch and Triton
- Surrogate Modeling Toolbox (SMT) for sampling
- Inspired by physics-informed machine learning research

## Contact

For questions and support, please open an issue on GitHub.
