"""
PINNsrc - Physics-Informed Neural Networks for Compressible Euler Equations

This package provides implementations of PINNs for solving 1D and 2D
compressible Euler equations with various boundary and initial conditions.
"""

__version__ = "0.1.0"

from .PINNs import (
    PINNs_WE_Euler_1D,
    PINNs_Euler_1D,
    PINNs_WE_Euler_2D,
    gradients,
)

from .utility import (
    select_gpu,
    save_results,
)

__all__ = [
    "PINNs_WE_Euler_1D",
    "PINNs_Euler_1D",
    "PINNs_WE_Euler_2D",
    "gradients",
    "select_gpu",
    "save_results",
]
