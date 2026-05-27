"""Plotting for tgv_3d_smooth (same structure as tgv_3d)."""
from pathlib import Path
import sys; sys.path.insert(0, str(Path(__file__).parent.parent))
from plots.tgv_3d import plot_from_checkpoints

if __name__ == "__main__":
    import cases.tgv_3d_smooth.physics as physics
    plot_from_checkpoints(physics, "output/tgv_3d_smooth/figures")
