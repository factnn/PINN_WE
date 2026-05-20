"""
Track 1 stability: 5 independent runs, measure steps 51-3000 (after warmup 50).
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from baseline.burgers_1d_compare import MLP, loss_canpinn, loss_triton, loss_vanilla, make_grid

device = "cuda"
WARMUP = 50
TOTAL = 3000

X, T, dx, dt = make_grid()
xt_all = torch.stack([X.flatten(), T.flatten()], dim=1)

def ic_loss(U, X):
    return (U[0] - (-torch.sin(3.14159265 * X[0]))).pow(2).mean()

def bc_loss(U):
    return U[:, 0].pow(2).mean() + U[:, -1].pow(2).mean()

for backend in ["vanilla", "canpinn", "compile", "triton"]:
    print(f"\n=== {backend} (5 runs, steps 51-{TOTAL}) ===")
    times = []
    for run in range(5):
        model = MLP(width=50, depth=4).to(device)
        if backend == "compile":
            model = torch.compile(model)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        def step_fn():
            opt.zero_grad()
            U = model(xt_all).reshape(*X.shape)
            if backend == "vanilla":
                loss = loss_vanilla(model, X, T) + 10*ic_loss(U, X) + 10*bc_loss(U)
            elif backend in ("canpinn", "compile"):
                loss = loss_canpinn(model, X, T, dx, dt) + 10*ic_loss(U, X) + 10*bc_loss(U)
            else:
                loss = loss_triton(U, dx, dt) + 10*ic_loss(U, X) + 10*bc_loss(U)
            loss.backward(); opt.step()

        for _ in range(WARMUP):
            step_fn()
        torch.cuda.synchronize()

        t0 = time.time()
        for _ in range(TOTAL - WARMUP):
            step_fn()
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        times.append(elapsed)
        print(f"  run {run+1}: {elapsed:.3f}s  ({elapsed/(TOTAL-WARMUP)*1000:.2f}ms/step)")

    print(f"  median={np.median(times):.3f}s  cv={np.std(times)/np.mean(times)*100:.1f}%")
