#!/usr/bin/env python3
"""Unified entry point for PINN Triton benchmark suite.

Usage:
    python run.py --case burgers_1d_steady --backend mlp_triton --track 2 --gpu 0
    python run.py --case ldc_2d --backend all --track 1 --gpu 0
    python run.py --case all --track all --gpu 0
"""
import sys, os, argparse, importlib
from pathlib import Path

# MUST set CUDA_VISIBLE_DEVICES before importing torch!
# Parse --gpu early from sys.argv
for i, arg in enumerate(sys.argv):
    if arg == '--gpu' and i + 1 < len(sys.argv):
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[i + 1]
        break

# Ensure triton_accelerate/ is on path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from engine.utils import BACKENDS

ALL_CASES = [
    "burgers_1d_steady", "burgers_1d_unsteady",
    "ldc_2d", "ldc_3d",
    "tgv_2d", "tgv_3d",
    "transport_2d", "sod_1d",
    "diffusion_1d", "diffusion_2d", "tgv_3d_smooth",
]


def load_physics(case_name):
    """Dynamically import cases.<case_name>.physics"""
    return importlib.import_module(f"cases.{case_name}.physics")


def parse_args():
    p = argparse.ArgumentParser(description="PINN Triton Benchmark Runner")
    p.add_argument("--case", type=str, default="all",
                   help=f"Case name or 'all'. Choices: {ALL_CASES}")
    p.add_argument("--backend", type=str, default="all",
                   help=f"Backend name or 'all'. Choices: {BACKENDS}")
    p.add_argument("--track", type=str, default="2", choices=["0", "1", "2", "all"],
                   help="Track: 0 (kernel), 1 (throughput), 2 (convergence), or all")
    p.add_argument("--gpu", type=int, default=0)
    # Track 2 params
    p.add_argument("--max-epochs", type=int, default=None,
                   help="Override max_epochs (default: 200000)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--threshold", type=float, default=None,
                   help="Override loss threshold (default: 1e-5)")
    p.add_argument("--runs", type=int, default=1)
    return p.parse_args()


def run_case(case_name, backends, tracks, args):
    """Run specified tracks for a single case."""
    print(f"\n{'='*60}")
    print(f"  Case: {case_name}")
    print(f"{'='*60}")

    physics = load_physics(case_name)
    device = "cuda"
    ctx = physics.make_context(device)

    max_epochs = args.max_epochs or 200000
    threshold = args.threshold  # None = auto from physics.py

    if "0" in tracks:
        from engine.track0 import run_track0
        run_track0(physics, ctx, device=device)

    if "1" in tracks:
        from engine.track1 import run_track1
        print(f"\n--- Track 1: Throughput ({case_name}) ---")
        run_track1(physics, ctx, backends=backends, device=device, gpu=args.gpu)

    if "2" in tracks:
        from engine.track2 import run_track2
        print(f"\n--- Track 2: Convergence ({case_name}) ---")
        for backend in backends:
            run_track2(physics, ctx, backend=backend, device=device,
                       max_epochs=max_epochs, lr=args.lr,
                       threshold=threshold, runs=args.runs)


def main():
    args = parse_args()

    # Resolve cases
    cases = ALL_CASES if args.case == "all" else [args.case]

    # Resolve backends
    if args.backend == "all":
        backends = BACKENDS
    else:
        backends = [b.strip() for b in args.backend.split(",")]

    # Resolve tracks
    if args.track == "all":
        tracks = ["0", "1", "2"]
    else:
        tracks = [args.track]

    for case_name in cases:
        run_case(case_name, backends, tracks, args)


if __name__ == "__main__":
    main()
