"""Plotting for diffusion_2d (same structure as transport_2d)."""
from pathlib import Path
import sys; sys.path.insert(0, str(Path(__file__).parent.parent))
from plots.transport_2d import plot_from_checkpoints

if __name__ == "__main__":
    import cases.diffusion_2d.physics as physics
    plot_from_checkpoints(physics, "output/diffusion_2d/figures")
