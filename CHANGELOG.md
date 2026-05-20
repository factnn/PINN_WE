# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Modern Python packaging with pyproject.toml
- Comprehensive test suite with pytest
- Code quality tools (black, isort, flake8, mypy)
- Pre-commit hooks for automated code checking
- GitHub Actions CI/CD pipeline
- Contributing guidelines
- Issue and PR templates
- Comprehensive README
- MIT License

### Changed
- Restructured project for better maintainability

## [0.1.0] - 2026-01-16

### Added
- Initial implementation of 1D PINNs for Euler equations
- Initial implementation of 2D PINNs for Euler equations
- Triton-accelerated Riemann solvers
- Test cases: Sod, Lax, Shu-Osher, Blast, Burgers (1D)
- Test cases: Vortex, Riemann2D, Shock-Circle (2D)
- GPU auto-selection utility
- Result saving and visualization utilities
- Adaptive k-parameter for compression zones
- Two-stage optimization (Adam + LBFGS)

[Unreleased]: https://github.com/factnn/PINN_WE/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/factnn/PINN_WE/releases/tag/v0.1.0
