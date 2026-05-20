"""
Profiling utility: wraps any callable with PyTorch Profiler.
Usage:
    from profiler import profile_fn
    profile_fn(lambda: pde_residual_pytorch(u, dx), name="burgers_1d_pytorch")
Outputs chrome trace to ./profiles/<name>.json (open in chrome://tracing)
"""
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from pathlib import Path

PROFILE_DIR = Path(__file__).parent / "profiles"
PROFILE_DIR.mkdir(exist_ok=True)


def profile_fn(fn, name: str, warmup: int = 3, steps: int = 10):
    """Profile fn() and save chrome trace + print summary."""
    # warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for _ in range(steps):
            with record_function(name):
                fn()
        torch.cuda.synchronize()

    # chrome trace
    trace_path = PROFILE_DIR / f"{name}.json"
    prof.export_chrome_trace(str(trace_path))

    # summary
    print(f"\n=== {name} ===")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    print(f"Chrome trace saved: {trace_path}")
    return prof
