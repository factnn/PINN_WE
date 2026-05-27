"""Plotting for poisson_3d (z-mid slice contourf)."""
from pathlib import Path
import sys; sys.path.insert(0, str(Path(__file__).parent.parent))
from plots.poisson_2d import plot_from_checkpoints  # same logic works for 3D z-mid

if __name__ == "__main__":
    import cases.poisson_3d.physics as physics
    plot_from_checkpoints(physics, "output/poisson_3d/figures")
