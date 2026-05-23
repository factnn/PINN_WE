"""Track 1: Throughput benchmark.

Measures steady-state training speed after warmup.
Metrics: avg_ms, std_ms, tput(steps/s), mem_GB, bw_util%, speedup.
"""
import time
import torch
import numpy as np
from .utils import make_model, get_loss_fn, BACKENDS

WARMUP = 100
MEASURE = 2900
RUNS = 5
A100_BW_GBS = 1555.0  # A100 40GB peak memory bandwidth


def run_track1(physics, ctx, backends=None, device="cuda", gpu=0):
    """Run throughput benchmark for given backends.

    Args:
        physics: physics module with standard interface
        ctx: context dict from physics.make_context()
        backends: list of backend names, or None for all
        device: "cuda"
        gpu: GPU index (for reporting)

    Returns:
        dict[str, dict]: results per backend
    """
    if backends is None:
        backends = BACKENDS

    bps = physics.bytes_per_step()
    results = {}

    for backend in backends:
        print(f"\n=== {backend} ===")
        loss_fn = get_loss_fn(physics, backend)
        all_times = []
        all_mem = []

        for run_i in range(RUNS):
            model = make_model(physics, backend, device)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)

            def step():
                opt.zero_grad()
                loss = loss_fn(model, ctx)
                loss.backward()
                opt.step()

            # Warmup
            for _ in range(WARMUP):
                step()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

            # Measure
            t0 = time.time()
            for _ in range(MEASURE):
                step()
            torch.cuda.synchronize()
            elapsed = time.time() - t0
            mem_gb = torch.cuda.max_memory_allocated() / 1e9

            all_times.append(elapsed)
            all_mem.append(mem_gb)
            avg_ms = elapsed / MEASURE * 1000
            tput = MEASURE / elapsed
            print(f"  run {run_i+1}: {elapsed:.2f}s  avg={avg_ms:.2f}ms  "
                  f"tput={tput:.1f}steps/s  mem={mem_gb:.3f}GB")

        # Aggregate
        times_ms = [t / MEASURE * 1000 for t in all_times]
        median_ms = float(np.median(times_ms))
        avg_ms = float(np.mean(times_ms))
        std_ms = float(np.std(times_ms))
        tput = 1000.0 / median_ms
        mem_gb = float(np.median(all_mem))
        bw_gbs = bps / (median_ms * 1e-3) / 1e9
        bw_pct = bw_gbs / A100_BW_GBS * 100

        results[backend] = {
            "median_ms": median_ms,
            "avg_ms": avg_ms,
            "std_ms": std_ms,
            "tput": tput,
            "mem_gb": mem_gb,
            "bw_gbs": bw_gbs,
            "bw_util_pct": bw_pct,
        }
        print(f"  SUMMARY: median={median_ms:.2f}ms  std={std_ms:.3f}ms  "
              f"tput={tput:.1f}  mem={mem_gb:.3f}GB  bw={bw_pct:.1f}%")

    # Compute speedup relative to mlp_vanilla (or first)
    baseline_ms = results.get("mlp_vanilla", next(iter(results.values())))["median_ms"]
    for r in results.values():
        r["speedup"] = baseline_ms / r["median_ms"]

    # Print final table
    print(f"\n{'backend':<16} {'avg_ms':>8} {'std_ms':>8} {'tput':>8} "
          f"{'mem_GB':>8} {'bw%':>6} {'speedup':>8}")
    for b, r in results.items():
        print(f"{b:<16} {r['avg_ms']:>8.2f} {r['std_ms']:>8.3f} {r['tput']:>8.1f} "
              f"{r['mem_gb']:>8.3f} {r['bw_util_pct']:>6.1f} {r['speedup']:>8.2f}x")

    return results
