"""Scaling experiment: kernel speedup vs grid size.

For each case, tests PyTorch FD vs Triton at increasing grid sizes
until GPU OOM. Produces speedup-vs-size data for paper figures.

Usage:
    python run.py --case ldc_2d --track scaling --gpu 0
    python scaling.py --case burgers_1d_unsteady --gpu 0
"""
import sys, os, argparse, importlib, gc
from pathlib import Path

import torch
import triton
import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# Grid size sequences for each case type
SCALING_GRIDS = {
    "burgers_1d_steady": [256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304],
    "burgers_1d_unsteady": [
        (50, 512), (100, 1024), (200, 2048), (500, 2048),
        (1000, 2048), (1000, 4096), (2000, 4096), (4000, 4096),
    ],
    "ldc_2d": [32, 64, 128, 256, 512, 1024, 2048, 4096],
    "ldc_3d": [16, 32, 48, 64, 96, 128, 160, 192, 256],
    "tgv_2d": [
        (10, 32), (20, 64), (20, 128), (20, 256), (50, 256),
        (100, 256), (200, 256), (50, 512), (100, 512),
    ],
    "tgv_3d": [
        (5, 16), (10, 32), (10, 48), (10, 64), (10, 96),
        (20, 64), (20, 96), (20, 128), (40, 96),
    ],
    "transport_2d": [
        (10, 32), (20, 64), (20, 128), (50, 128), (50, 256),
        (100, 256), (200, 256), (100, 512), (200, 512),
    ],
    "sod_2d": [
        (50, 200), (100, 1000), (200, 2000), (500, 2000),
        (1000, 4000), (2000, 4000), (4000, 4000), (2000, 8000),
    ],
}


def make_random_fields(case_name, size, device="cuda", dtype=torch.float32):
    """Create random fields of given size for a case."""
    if case_name == "burgers_1d_steady":
        Nx = size
        return (torch.randn(Nx, device=device, dtype=dtype).requires_grad_(True),)
    elif case_name == "burgers_1d_unsteady":
        Nt, Nx = size
        return (torch.randn(Nt, Nx, device=device, dtype=dtype).requires_grad_(True),)
    elif case_name == "ldc_2d":
        N = size
        U = torch.randn(N, N, device=device, dtype=dtype).requires_grad_(True)
        V = torch.randn(N, N, device=device, dtype=dtype).requires_grad_(True)
        P = torch.randn(N, N, device=device, dtype=dtype).requires_grad_(True)
        return U, V, P
    elif case_name == "ldc_3d":
        N = size
        U = torch.randn(N, N, N, device=device, dtype=dtype).requires_grad_(True)
        V = torch.randn(N, N, N, device=device, dtype=dtype).requires_grad_(True)
        W = torch.randn(N, N, N, device=device, dtype=dtype).requires_grad_(True)
        P = torch.randn(N, N, N, device=device, dtype=dtype).requires_grad_(True)
        return U, V, W, P
    elif case_name in ("tgv_2d", "transport_2d"):
        Nt, N = size
        if case_name == "transport_2d":
            return (torch.randn(Nt, N, N, device=device, dtype=dtype).requires_grad_(True),)
        else:
            U = torch.randn(Nt, N, N, device=device, dtype=dtype).requires_grad_(True)
            V = torch.randn(Nt, N, N, device=device, dtype=dtype).requires_grad_(True)
            P = torch.randn(Nt, N, N, device=device, dtype=dtype).requires_grad_(True)
            return U, V, P
    elif case_name == "tgv_3d":
        Nt, N = size
        U = torch.randn(Nt, N, N, N, device=device, dtype=dtype).requires_grad_(True)
        V = torch.randn(Nt, N, N, N, device=device, dtype=dtype).requires_grad_(True)
        W = torch.randn(Nt, N, N, N, device=device, dtype=dtype).requires_grad_(True)
        P = torch.randn(Nt, N, N, N, device=device, dtype=dtype).requires_grad_(True)
        return U, V, W, P
    elif case_name == "sod_2d":
        Nt, Nx = size
        R = torch.randn(Nt, Nx, device=device, dtype=dtype).abs().requires_grad_(True)
        M = torch.randn(Nt, Nx, device=device, dtype=dtype).requires_grad_(True)
        E = torch.randn(Nt, Nx, device=device, dtype=dtype).abs().requires_grad_(True)
        return R, M, E


def get_kernel_fns(case_name):
    """Return (pytorch_fd_fn, triton_fn) callables that take fields + grid spacings."""
    if case_name == "burgers_1d_steady":
        from cases.burgers_1d_steady.physics import _pde_residual_fd, nu
        from kernels.stencil_1d_steady import burgers_steady_loss_triton
        def pt_fn(fields, dx):
            res = _pde_residual_fd(fields[0], dx)
            return (res**2).mean()
        def tr_fn(fields, dx):
            return burgers_steady_loss_triton(fields[0], dx, nu)
        return pt_fn, tr_fn, lambda size: 2.0 / (size - 1)  # dx_fn

    elif case_name == "burgers_1d_unsteady":
        from kernels.stencil_1d_unsteady import burgers_2d_loss_triton_autograd
        nu = 0.01 / np.pi
        def pt_fn(fields, spacings):
            U = fields[0]; dx, dt = spacings
            u_t = (U[2:, 1:-1] - U[:-2, 1:-1]) / (2*dt)
            u_x = (U[1:-1, 2:] - U[1:-1, :-2]) / (2*dx)
            u_xx = (U[1:-1, 2:] - 2*U[1:-1, 1:-1] + U[1:-1, :-2]) / dx**2
            uc = U[1:-1, 1:-1]
            res = u_t + uc * u_x - nu * u_xx
            return (res**2).mean()
        def tr_fn(fields, spacings):
            dx, dt = spacings
            return burgers_2d_loss_triton_autograd(fields[0], dx, dt, nu)
        def spacing_fn(size):
            Nt, Nx = size
            return (2.0 / (Nx - 1), 1.0 / (Nt - 1))
        return pt_fn, tr_fn, spacing_fn

    elif case_name == "ldc_2d":
        from kernels.stencil_2d_burgers_steady import ldc_residual_triton
        nu = 0.01
        def pt_fn(fields, spacings):
            U, V, P = fields; dx, dy = spacings
            from cases.ldc_2d.physics import _pde_residual_fd
            return _pde_residual_fd(U, V, P, dx, dy)
        def tr_fn(fields, spacings):
            U, V, P = fields; dx, dy = spacings
            return ldc_residual_triton(U, V, P, dx, dy, nu)
        def spacing_fn(size):
            return (1.0 / (size - 1), 1.0 / (size - 1))
        return pt_fn, tr_fn, spacing_fn

    elif case_name == "ldc_3d":
        from kernels.stencil_3d_ns_steady import ldc_residual_triton
        nu = 0.01
        def pt_fn(fields, spacings):
            U, V, W, P = fields; dx, dy, dz = spacings
            from cases.ldc_3d.physics import _pde_residual_fd
            return _pde_residual_fd(U, V, W, P, dx, dy, dz)
        def tr_fn(fields, spacings):
            U, V, W, P = fields; dx, dy, dz = spacings
            return ldc_residual_triton(U, V, W, P, dx, dy, dz, nu)
        def spacing_fn(size):
            d = 1.0 / (size - 1)
            return (d, d, d)
        return pt_fn, tr_fn, spacing_fn

    elif case_name == "tgv_2d":
        from kernels.stencil_2d_ns_unsteady import ns2d_residual_triton
        nu = 0.01
        def pt_fn(fields, spacings):
            U, V, P = fields; dx, dy, dt = spacings
            from cases.tgv_2d.physics import _pde_residual_fd
            return _pde_residual_fd(U, V, P, dx, dy, dt)
        def tr_fn(fields, spacings):
            U, V, P = fields; dx, dy, dt = spacings
            return ns2d_residual_triton(U, V, P, dx, dy, dt, nu)
        def spacing_fn(size):
            Nt, N = size
            dx = 2*np.pi / N; dy = dx; dt = 1.0 / (Nt - 1)
            return (dx, dy, dt)
        return pt_fn, tr_fn, spacing_fn

    elif case_name == "tgv_3d":
        from kernels.stencil_3d_ns_unsteady import ns3d_residual_triton
        nu = 0.01
        def pt_fn(fields, spacings):
            U, V, W, P = fields; dx, dy, dz, dt = spacings
            from cases.tgv_3d.physics import _pde_residual_fd
            return _pde_residual_fd(U, V, W, P, dx, dy, dz, dt)
        def tr_fn(fields, spacings):
            U, V, W, P = fields; dx, dy, dz, dt = spacings
            return ns3d_residual_triton(U, V, W, P, dx, dy, dz, dt, nu)
        def spacing_fn(size):
            Nt, N = size
            d = 2*np.pi / N; dt = 1.0 / (Nt - 1)
            return (d, d, d, dt)
        return pt_fn, tr_fn, spacing_fn

    elif case_name == "transport_2d":
        from kernels.stencil_2d_transport import advdiff_residual_triton
        nu = 0.01; u0 = 1.0; v0 = 1.0
        def pt_fn(fields, spacings):
            C = fields[0]; dx, dy, dt = spacings
            from cases.transport_2d.physics import _pde_residual_fd
            return _pde_residual_fd(C, dx, dy, dt)
        def tr_fn(fields, spacings):
            C = fields[0]; dx, dy, dt = spacings
            return advdiff_residual_triton(C, dx, dy, dt, nu, u0, v0)
        def spacing_fn(size):
            Nt, N = size
            dx = 2*np.pi / N; dy = dx; dt = 1.0 / (Nt - 1)
            return (dx, dy, dt)
        return pt_fn, tr_fn, spacing_fn

    elif case_name == "sod_2d":
        from kernels.stencil_2d_compressible import compressible_sod_residual_triton
        gamma = 1.4
        def pt_fn(fields, spacings):
            R, M, E = fields; dx, dt = spacings
            from cases.sod_2d.physics import _euler_residual_fd
            return _euler_residual_fd(R, M, E, dx, dt)
        def tr_fn(fields, spacings):
            R, M, E = fields; dx, dt = spacings
            return compressible_sod_residual_triton(R, M, E, dx, dt, gamma)
        def spacing_fn(size):
            Nt, Nx = size
            return (1.0 / (Nx - 1), 0.2 / (Nt - 1))
        return pt_fn, tr_fn, spacing_fn


def run_scaling(case_name, device="cuda"):
    """Run scaling experiment for one case."""
    print(f"\n{'='*70}")
    print(f"  SCALING EXPERIMENT: {case_name}")
    print(f"{'='*70}")

    grids = SCALING_GRIDS[case_name]
    pt_fn, tr_fn, spacing_fn = get_kernel_fns(case_name)

    results = []
    for size in grids:
        # Compute total points
        if isinstance(size, int):
            n_points = size if case_name == "burgers_1d_steady" else size**2
            if case_name == "ldc_3d":
                n_points = size**3
            label = str(size)
        else:
            if len(size) == 2:
                n_points = size[0] * size[1]
                if case_name in ("tgv_2d", "transport_2d"):
                    n_points = size[0] * size[1] * size[1]
                elif case_name == "tgv_3d":
                    n_points = size[0] * size[1]**3
            label = f"{size[0]}x{size[1]}"

        spacings = spacing_fn(size)

        try:
            # Clear cache
            torch.cuda.empty_cache()
            gc.collect()

            fields = make_random_fields(case_name, size, device)

            # Benchmark PyTorch FD: forward + backward
            def pt_fwd_bwd():
                fs = tuple(f.detach().requires_grad_(True) for f in fields)
                loss = pt_fn(fs, spacings)
                loss.backward()

            def tr_fwd_bwd():
                fs = tuple(f.detach().requires_grad_(True) for f in fields)
                loss = tr_fn(fs, spacings)
                loss.backward()

            pt_ms = triton.testing.do_bench(pt_fwd_bwd)
            tr_ms = triton.testing.do_bench(tr_fwd_bwd)
            speedup = pt_ms / tr_ms
            mem_gb = torch.cuda.max_memory_allocated() / 1e9

            results.append({
                "size": label, "n_points": n_points,
                "pt_ms": pt_ms, "tr_ms": tr_ms,
                "speedup": speedup, "mem_gb": mem_gb,
            })
            print(f"  {label:>12s} | {n_points:>10,} pts | "
                  f"PT={pt_ms:>8.3f}ms  TR={tr_ms:>8.3f}ms | "
                  f"speedup={speedup:.2f}x | mem={mem_gb:.2f}GB")

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  {label:>12s} | OOM — stopping.")
                torch.cuda.empty_cache()
                break
            else:
                print(f"  {label:>12s} | ERROR: {e}")
                break

    # Summary
    if results:
        print(f"\n  {'Size':>12s} {'Points':>10s} {'PT(ms)':>8s} {'TR(ms)':>8s} {'Speedup':>8s}")
        for r in results:
            print(f"  {r['size']:>12s} {r['n_points']:>10,} {r['pt_ms']:>8.3f} {r['tr_ms']:>8.3f} {r['speedup']:>8.2f}x")

    # Save
    out_dir = ROOT / "output" / "scaling"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"scaling_{case_name}.npy", results)
    print(f"\n  Saved: {out_dir}/scaling_{case_name}.npy")

    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Scaling experiment")
    p.add_argument("--case", type=str, required=True)
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    run_scaling(args.case)
