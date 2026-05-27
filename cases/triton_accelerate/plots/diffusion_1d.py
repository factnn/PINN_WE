"""Plotting for diffusion_1d (same structure as burgers_1d_unsteady)."""
from pathlib import Path
import sys; sys.path.insert(0, str(Path(__file__).parent.parent))
from plots.burgers_1d_unsteady import plot_from_checkpoints

if __name__ == "__main__":
    import cases.diffusion_1d.physics as physics
    plot_from_checkpoints(physics, "output/diffusion_1d/figures")
