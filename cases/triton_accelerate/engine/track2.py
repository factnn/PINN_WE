"""Track 2: Convergence benchmark.

Trains until loss < threshold or max_epochs reached.
Metrics: T2S, total_epochs, avg_step_ms, peak_mem_GB, L2_error.
"""
import time
from pathlib import Path

import torch
import numpy as np
from .utils import make_model, get_loss_fn


def run_track2(physics, ctx, backend, device="cuda",
               max_epochs=200000, lr=1e-3, threshold=1e-5, runs=1,
               out_dir=None):
    """Run convergence training.

    Args:
        physics: physics module
        ctx: context dict
        backend: backend name string
        device: "cuda"
        max_epochs, lr, threshold, runs: training params
        out_dir: output directory for checkpoints/plots

    Returns:
        dict with aggregated results
    """
    if out_dir is None:
        out_dir = Path(__file__).parent.parent / "output" / physics.CASE_NAME
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loss_fn = get_loss_fn(physics, backend)
    all_elapsed = []
    all_t2s = []

    for run_i in range(runs):
        model = make_model(physics, backend, device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs, eta_min=1e-5)

        history = []
        t2s = None
        t2s_ep = None
        step_times = []
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        print(f"[{backend}] run {run_i+1}/{runs} | {physics.CASE_NAME} {physics.GRID_SHAPE}")

        for ep in range(1, max_epochs + 1):
            t_step = time.time()
            opt.zero_grad()
            loss = loss_fn(model, ctx)
            loss.backward()
            opt.step()
            sch.step()
            torch.cuda.synchronize()
            step_times.append(time.time() - t_step)

            wall = time.time() - t0
            lv = loss.item()
            history.append((wall, ep, lv))

            if t2s is None and lv < threshold:
                t2s = wall
                t2s_ep = ep
                print(f"  T2S={wall:.1f}s at ep {ep}")
                break
            if ep % 5000 == 0:
                print(f"  ep {ep:6d} loss={lv:.3e} t={wall:.1f}s")

        elapsed = time.time() - t0
        mem_gb = torch.cuda.max_memory_allocated() / 1e9
        avg_step_ms = float(np.mean(step_times) * 1000)
        all_elapsed.append(elapsed)
        all_t2s.append(t2s)
        print(f"[{backend}] run {run_i+1}: {elapsed:.1f}s  mem={mem_gb:.3f}GB  "
              f"avg_step={avg_step_ms:.2f}ms  epochs={t2s_ep or ep}")

    # Save model
    ckpt = out_dir / f"model_{backend}.pt"
    torch.save(getattr(model, '_orig_mod', model).state_dict(), ckpt)

    # Compute L2 error
    l2 = physics.compute_l2_error(model, ctx)

    # Final PDE loss (always available, unlike L2 which needs exact solution)
    final_loss = history[-1][2] if history else None

    # Plot
    physics.plot_solution(model, ctx, backend, out_dir)

    # Memory bandwidth estimate
    bps = physics.bytes_per_step()
    mem_bw_gbs = bps / (avg_step_ms * 1e-3) / 1e9

    # Save metrics
    metrics = dict(
        history=history, elapsed=elapsed, mem_gb=mem_gb,
        t2s=t2s, t2s_ep=t2s_ep, l2=l2, final_loss=final_loss,
        avg_step_ms=avg_step_ms, mem_bw_gbs=mem_bw_gbs,
        all_elapsed=all_elapsed, all_t2s=all_t2s,
    )
    np.save(out_dir / f"meta_{backend}.npy", metrics)

    # Summary
    print(f"\n[{backend}] === Summary ===")
    print(f"  T2S          : {t2s:.1f}s" if t2s else "  T2S          : N/A")
    print(f"  Total_Epochs : {t2s_ep or ep}")
    print(f"  Avg_Step_ms  : {avg_step_ms:.2f}")
    print(f"  Peak_Mem_GB  : {mem_gb:.3f}")
    print(f"  Final_Loss   : {final_loss:.4e}" if final_loss else "  Final_Loss   : N/A")
    print(f"  L2_Error     : {l2:.4e}" if l2 is not None else "  L2_Error     : N/A")

    return metrics
