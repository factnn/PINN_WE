"""
torch.compile baseline for 1D Burgers.
Compares compiled vs uncompiled CAN-PINN.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
from cases.burgers_1d_unsteady.compare import MLP, loss_canpinn, make_grid

device = "cuda"
WARMUP = 50
MEASURE = 2950
RUNS = 5

X, T, dx, dt = make_grid()
xt_all = torch.stack([X.flatten(), T.flatten()], dim=1)

def ic_loss(U, X):
    return (U[0] - (-torch.sin(np.pi * X[0]))).pow(2).mean()

def bc_loss(U):
    return U[:, 0].pow(2).mean() + U[:, -1].pow(2).mean()

def run(use_compile, n_runs=RUNS):
    times = []
    for run_i in range(n_runs):
        model = MLP(width=50, depth=4).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        if use_compile:
            model = torch.compile(model)

        def step():
            opt.zero_grad()
            U = model(xt_all).reshape(*X.shape)
            loss = loss_canpinn(model, X, T, dx, dt) + 10*ic_loss(U, X) + 10*bc_loss(U)
            loss.backward(); opt.step()

        for _ in range(WARMUP):
            step()
        torch.cuda.synchronize()

        t0 = time.time()
        for _ in range(MEASURE):
            step()
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        times.append(elapsed)
        print(f"  run {run_i+1}: {elapsed:.3f}s  ({elapsed/MEASURE*1000:.2f}ms/step)")

    print(f"  median={np.median(times):.3f}s  cv={np.std(times)/np.mean(times)*100:.1f}%")
    return np.median(times)

print("=== CAN-PINN (no compile) ===")
t_base = run(use_compile=False)

print("\n=== CAN-PINN (torch.compile) ===")
t_comp = run(use_compile=True)

print(f"\nSpeedup: {t_base/t_comp:.2f}x")
