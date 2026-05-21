"""
Compare three timing methods for Track 1:
1. time.time() wall clock
2. CUDA Event (GPU-only time)
3. triton.testing.do_bench (kernel-only)
Run 3 trials each to check stability.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import triton
from cases.burgers_1d_unsteady.compare import MLP, loss_canpinn, loss_triton, make_grid

device = "cuda"
WARMUP = 50
MEASURE = 450

X, T, dx, dt = make_grid()
xt_all = torch.stack([X.flatten(), T.flatten()], dim=1)

for backend in ["canpinn", "triton"]:
    print(f"\n{'='*50}\n{backend}")
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

    # Method 1: wall clock (3 trials)
    print("Method 1: wall clock (time.time)")
    for trial in range(3):
        t0 = time.time()
        for _ in range(MEASURE):
            step_fn()
        torch.cuda.synchronize()
        print(f"  trial {trial+1}: {time.time()-t0:.3f}s total, {(time.time()-t0)/MEASURE*1000:.2f}ms/step")

    # Method 2: CUDA Event
    print("Method 2: CUDA Event")
    for trial in range(3):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(MEASURE):
            step_fn()
        end.record()
        torch.cuda.synchronize()
        ms_total = start.elapsed_time(end)
        print(f"  trial {trial+1}: {ms_total/1000:.3f}s total, {ms_total/MEASURE:.2f}ms/step")

    # Method 3: do_bench (kernel only, single step)
    print("Method 3: triton.do_bench (single step, kernel only)")
    ms = triton.testing.do_bench(step_fn)
    print(f"  {ms:.2f}ms/step")
