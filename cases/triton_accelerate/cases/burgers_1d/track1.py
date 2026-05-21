"""Track 1 throughput for 1D Burgers: warmup 50, measure 2950 steps, 5 runs."""
import sys, time, os, shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Clear Triton cache
triton_cache = os.path.expanduser('~/.triton/cache')
if os.path.exists(triton_cache):
    shutil.rmtree(triton_cache)

import torch
import numpy as np
import triton
import argparse
from cases.burgers_1d.common import MLP, make_grid, ic_loss_from_U, bc_loss_from_U
from cases.burgers_1d.compare import loss_vanilla, loss_canpinn
from kernels.stencil_1d import burgers_2d_loss_triton_autograd

A100_BW_GBS = 1555.0
BYTES_PER_STEP = 1024 * 100 * 4 * 6  # Nt*Nx*float32*accesses

WARMUP, MEASURE, RUNS = 50, 2950, 5

def make_loss(backend):
    if backend == "vanilla":
        def fn(model, U, X, T, dx, dt):
            return loss_vanilla(model, X, T) + 10*ic_loss_from_U(U,X) + 10*bc_loss_from_U(U)
    elif backend in ("canpinn", "compile"):
        def fn(model, U, X, T, dx, dt):
            return loss_canpinn(model, X, T, dx, dt) + 10*ic_loss_from_U(U,X) + 10*bc_loss_from_U(U)
    else:
        nu_val = 0.01 / np.pi
        def fn(model, U, X, T, dx, dt):
            return burgers_2d_loss_triton_autograd(U, dx, dt, nu_val) + 10*ic_loss_from_U(U,X) + 10*bc_loss_from_U(U)
    return fn

def make_model(backend):
    m = MLP()
    if backend == "compile":
        m = torch.compile(m)
    return m.to("cuda")

p = argparse.ArgumentParser()
p.add_argument('--backends', default='vanilla,canpinn,compile,triton')
p.add_argument('--gpu', type=int, default=0)
args = p.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

X, T, dx, dt = make_grid()
xt_all = torch.stack([X.flatten(), T.flatten()], dim=1)

results = {}
for backend in args.backends.split(','):
    print(f"\n=== {backend} ===")
    loss_fn = make_loss(backend)
    all_times = []; all_mem = []

    for run_i in range(RUNS):
        model = make_model(backend)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        def step():
            opt.zero_grad()
            U = model(xt_all).reshape(*X.shape)
            loss = loss_fn(model, U, X, T, dx, dt)
            loss.backward(); opt.step()

        for _ in range(WARMUP): step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        t0 = time.time()
        for _ in range(MEASURE): step()
        torch.cuda.synchronize()
        elapsed = time.time()-t0
        mem_gb = torch.cuda.max_memory_allocated()/1e9

        all_times.append(elapsed); all_mem.append(mem_gb)
        avg_ms = elapsed/MEASURE*1000
        tput = MEASURE/elapsed
        bw = BYTES_PER_STEP/(elapsed/MEASURE)/1e9
        print(f"  run {run_i+1}: {elapsed:.3f}s  avg={avg_ms:.2f}ms  tput={tput:.1f}steps/s  mem={mem_gb:.3f}GB  bw={bw:.2f}GB/s ({bw/A100_BW_GBS*100:.3f}%)")

    med_t = np.median(all_times); med_mem = np.median(all_mem)
    avg_ms = med_t/MEASURE*1000; tput = MEASURE/med_t
    bw = BYTES_PER_STEP/(med_t/MEASURE)/1e9; bw_pct = bw/A100_BW_GBS*100

    model_b = make_model(backend)
    opt_b = torch.optim.Adam(model_b.parameters(), lr=1e-3)
    def step_b():
        opt_b.zero_grad()
        U = model_b(xt_all).reshape(*X.shape)
        loss_fn(model_b, U, X, T, dx, dt).backward(); opt_b.step()
    kernel_ms = triton.testing.do_bench(step_b)

    results[backend] = dict(median_s=med_t, avg_step_ms=avg_ms,
                            throughput=tput, peak_mem_gb=med_mem,
                            bw_gbs=bw, bw_pct=bw_pct, do_bench_ms=kernel_ms)
    print(f"  SUMMARY: median={med_t:.3f}s  avg={avg_ms:.2f}ms  tput={tput:.1f}  mem={med_mem:.3f}GB  bw={bw:.2f}GB/s({bw_pct:.3f}%)  do_bench={kernel_ms:.2f}ms")

print("\n\n=== FINAL TABLE ===")
print(f"{'backend':<12} {'median(s)':>10} {'avg_ms':>8} {'tput':>8} {'mem_GB':>8} {'bw_GBs':>8} {'bw_pct':>8} {'do_bench_ms':>12}")
for b, r in results.items():
    print(f"{b:<12} {r['median_s']:>10.3f} {r['avg_step_ms']:>8.2f} {r['throughput']:>8.1f} {r['peak_mem_gb']:>8.3f} {r['bw_gbs']:>8.3f} {r['bw_pct']:>8.3f} {r['do_bench_ms']:>12.2f}")
