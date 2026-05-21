"""
Track 1 stability test.
Warmup 50 steps, then measure avg step time at intervals: 50, 450, 950, 1450, 1950 cumulative steps.
Each interval avg = (interval_time) / (interval_steps).
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from cases.burgers_1d_unsteady.compare import MLP, loss_canpinn, loss_triton, make_grid

device = "cuda"
WARMUP = 50
CHECKPOINTS = [50, 450, 950, 1450, 1950]  # cumulative steps after warmup

X, T, dx, dt = make_grid()
xt_all = torch.stack([X.flatten(), T.flatten()], dim=1)

for backend in ["canpinn", "triton"]:
    print(f"\n=== {backend} ===")
    model = MLP(width=50, depth=4).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    def step_fn():
        opt.zero_grad()
        U = model(xt_all).reshape(*X.shape)
        loss = loss_canpinn(model, X, T, dx, dt) if backend == "canpinn" else loss_triton(U, dx, dt)
        loss.backward(); opt.step()

    # warmup
    for _ in range(WARMUP):
        step_fn()
    torch.cuda.synchronize()

    # measure intervals
    prev = 0
    t_interval = time.time()
    for step in range(1, CHECKPOINTS[-1] + 1):
        step_fn()
        if step in CHECKPOINTS:
            torch.cuda.synchronize()
            elapsed = time.time() - t_interval
            n = step - prev
            print(f"  steps {prev+1:4d}-{step:4d} ({n:4d} steps): {elapsed/n*1000:.3f} ms/step")
            prev = step
            t_interval = time.time()
