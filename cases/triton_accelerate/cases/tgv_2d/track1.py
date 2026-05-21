"""Track 1 throughput for 2D TGV: warmup 50, measure 2950 steps, 5 runs."""
import sys, time, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
import triton
import argparse
from cases.tgv_2d.common import *
from cases.tgv_2d.common import infer
from kernels.stencil_2d_unsteady import ns2d_residual_triton

# A100 40GB peak memory bandwidth
A100_BW_GBS = 1555.0

def make_loss_fn(backend):
    if backend == "mlp_vanilla":
        def fn(model, xyt, X, Y, T, U_e, V_e, P_e, dx, dy, dt):
            U, V, P = infer(model, xyt, X, Y, T)
            return pde_residual_pytorch(U, V, P, dx, dy, dt) + ic_bc_loss(U, V, P, U_e, V_e, P_e)
        return fn
    elif backend in ("mlp_canpinn", "mlp_compile", "cnn_canpinn", "cnn_compile"):
        return unified_loss_fn
    else:  # triton
        def fn(model, xyt, X, Y, T, U_e, V_e, P_e, dx, dy, dt):
            U, V, P = infer(model, xyt, X, Y, T)
            return ns2d_residual_triton(U.contiguous(), V.contiguous(), P.contiguous(), dx, dy, dt, nu) \
                   + ic_bc_loss(U, V, P, U_e, V_e, P_e)
        return fn

def make_model(backend):
    m = PhyCNN() if "cnn" in backend else MLP()
    if "compile" in backend:
        m = torch.compile(m)
    return m.to(device)

WARMUP = 50
MEASURE = 2950
RUNS = 5

# bytes read/written per step (U,V,P each [Nt,Nx,Ny] float32, stencil ~14 reads + 1 write per point)
N_POINTS = Nt * Nx * Ny
BYTES_PER_STEP = 3 * N_POINTS * 4 * (14 + 1)  # 3 fields, 4 bytes, 15 accesses

p = argparse.ArgumentParser()
p.add_argument('--backends', default='mlp_vanilla,mlp_canpinn,mlp_compile,mlp_triton,cnn_canpinn,cnn_compile,cnn_triton')
p.add_argument('--gpu', type=int, default=0)
args = p.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

X, Y, T, dx, dy, dt = make_grid()
U_e, V_e, P_e = exact_uvp(X, Y, T)
xyt = torch.stack([X.flatten(), Y.flatten(), T.flatten()], dim=1)

results = {}
for backend in args.backends.split(','):
    print(f"\n=== {backend} ===")
    loss_fn = make_loss_fn(backend)
    all_times = []; all_mem = []

    for run_i in range(RUNS):
        model = make_model(backend)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        def step():
            opt.zero_grad()
            loss = loss_fn(model, xyt, X, Y, T, U_e, V_e, P_e, dx, dy, dt)
            loss.backward(); opt.step()

        # warmup
        for _ in range(WARMUP):
            step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        # measure
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
        bw = BYTES_PER_STEP / (elapsed / MEASURE) / 1e9
        bw_pct = bw / A100_BW_GBS * 100
        print(f"  run {run_i+1}: {elapsed:.3f}s  avg={avg_ms:.2f}ms  tput={tput:.1f}steps/s  mem={mem_gb:.3f}GB  bw={bw:.1f}GB/s ({bw_pct:.1f}%)")

    med_t = np.median(all_times)
    med_mem = np.median(all_mem)
    avg_ms = med_t / MEASURE * 1000
    tput = MEASURE / med_t
    bw = BYTES_PER_STEP / (med_t / MEASURE) / 1e9
    bw_pct = bw / A100_BW_GBS * 100

    # kernel-only latency via do_bench
    model_bench = make_model(backend)
    opt_bench = torch.optim.Adam(model_bench.parameters(), lr=1e-3)
    def step_bench():
        opt_bench.zero_grad()
        loss = loss_fn(model_bench, xyt, X, Y, T, U_e, V_e, P_e, dx, dy, dt)
        loss.backward(); opt_bench.step()
    kernel_ms = triton.testing.do_bench(step_bench)

    results[backend] = dict(
        median_s=med_t, cv=np.std(all_times)/np.mean(all_times)*100,
        avg_step_ms=avg_ms, throughput_steps_per_sec=tput,
        peak_mem_gb=med_mem, bandwidth_gbs=bw, bandwidth_pct=bw_pct,
        kernel_do_bench_ms=kernel_ms,
    )
    print(f"  SUMMARY: median={med_t:.3f}s  avg_step={avg_ms:.2f}ms  tput={tput:.1f}steps/s  mem={med_mem:.3f}GB  bw={bw:.1f}GB/s({bw_pct:.1f}%)  do_bench={kernel_ms:.2f}ms")

print("\n\n=== FINAL TABLE ===")
print(f"{'backend':<16} {'median(s)':>10} {'avg_ms':>8} {'tput':>8} {'mem_GB':>8} {'bw_GBs':>8} {'bw_pct':>8} {'do_bench_ms':>12}")
for b, r in results.items():
    print(f"{b:<16} {r['median_s']:>10.3f} {r['avg_step_ms']:>8.2f} {r['throughput_steps_per_sec']:>8.1f} {r['peak_mem_gb']:>8.3f} {r['bandwidth_gbs']:>8.1f} {r['bandwidth_pct']:>8.1f} {r['kernel_do_bench_ms']:>12.2f}")
