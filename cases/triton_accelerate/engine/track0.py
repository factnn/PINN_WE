"""Track 0: Kernel-level benchmark.

Measures pure PDE residual computation speed (forward + backward),
without model inference or optimizer overhead.
Uses triton.testing.do_bench for accurate GPU timing.

Metrics: fwd_ms, bwd_ms, total_ms, speedup (pytorch_fd vs triton).
"""
import torch
import triton
import numpy as np
from pathlib import Path


def _make_fields(physics, ctx, device="cuda"):
    """Create random fields matching the case's output shape, with grad."""
    # Use infer to get the shape, then create random fields
    from .utils import make_model
    model = make_model(physics, "mlp_canpinn", device)
    with torch.no_grad():
        fields = physics.infer(model, ctx)
    del model

    if isinstance(fields, tuple):
        return tuple(torch.randn_like(f).requires_grad_(True) for f in fields)
    else:
        return torch.randn_like(fields).requires_grad_(True)


def _bench_forward_backward(loss_fn_call):
    """Benchmark forward + backward of a loss function call.
    Returns (fwd_ms, bwd_ms, total_ms).
    """
    # Forward only
    fwd_ms = triton.testing.do_bench(loss_fn_call)

    # Forward + backward
    def fwd_bwd():
        loss = loss_fn_call()
        loss.backward(retain_graph=False)

    total_ms = triton.testing.do_bench(fwd_bwd)
    bwd_ms = total_ms - fwd_ms

    return fwd_ms, bwd_ms, total_ms


def run_track0(physics, ctx, device="cuda"):
    """Run kernel-level benchmark for a single case.

    Compares pytorch FD (canpinn) vs Triton kernel, both on random fields.
    No model involved — pure PDE residual computation.

    Returns:
        dict with pytorch and triton timings + speedups
    """
    print(f"\n{'='*60}")
    print(f"  Track 0: Kernel Benchmark — {physics.CASE_NAME}")
    print(f"  Grid: {physics.GRID_SHAPE}")
    print(f"{'='*60}")

    fields = _make_fields(physics, ctx, device)

    # Build closures that compute PDE loss from fields directly
    # We need to bypass infer() and call the PDE residual directly
    # Strategy: override model with a dummy that returns our fixed fields

    class _DummyModel:
        """Dummy model that returns pre-set fields."""
        pass

    dummy = _DummyModel()

    # For pytorch FD (canpinn): we need fields with grad
    def make_fresh_fields():
        if isinstance(fields, tuple):
            return tuple(f.detach().requires_grad_(True) for f in fields)
        else:
            return fields.detach().requires_grad_(True)

    # Get the internal PDE residual functions
    # We'll call loss_canpinn and loss_triton but with a patched infer
    has_triton = True
    try:
        # Test if triton loss works
        fresh = make_fresh_fields()
        _patch_and_call(physics, physics.loss_triton, ctx, fresh)
    except (NotImplementedError, ImportError):
        has_triton = False
        print("  [SKIP] Triton kernel not implemented for this case")

    # --- Benchmark PyTorch FD ---
    print("\n  PyTorch FD (canpinn):")

    def pytorch_fwd():
        fresh = make_fresh_fields()
        return _patch_and_call(physics, physics.loss_canpinn, ctx, fresh)

    def pytorch_fwd_bwd():
        fresh = make_fresh_fields()
        loss = _patch_and_call(physics, physics.loss_canpinn, ctx, fresh)
        loss.backward()

    pt_fwd_ms = triton.testing.do_bench(pytorch_fwd)
    pt_total_ms = triton.testing.do_bench(pytorch_fwd_bwd)
    pt_bwd_ms = pt_total_ms - pt_fwd_ms
    print(f"    fwd: {pt_fwd_ms:.3f}ms  bwd: {pt_bwd_ms:.3f}ms  total: {pt_total_ms:.3f}ms")

    results = {
        "pytorch_fwd_ms": pt_fwd_ms,
        "pytorch_bwd_ms": pt_bwd_ms,
        "pytorch_total_ms": pt_total_ms,
    }

    # --- Benchmark Triton ---
    if has_triton:
        print("  Triton kernel:")

        def triton_fwd():
            fresh = make_fresh_fields()
            return _patch_and_call(physics, physics.loss_triton, ctx, fresh)

        def triton_fwd_bwd():
            fresh = make_fresh_fields()
            loss = _patch_and_call(physics, physics.loss_triton, ctx, fresh)
            loss.backward()

        tr_fwd_ms = triton.testing.do_bench(triton_fwd)
        tr_total_ms = triton.testing.do_bench(triton_fwd_bwd)
        tr_bwd_ms = tr_total_ms - tr_fwd_ms
        print(f"    fwd: {tr_fwd_ms:.3f}ms  bwd: {tr_bwd_ms:.3f}ms  total: {tr_total_ms:.3f}ms")

        fwd_speedup = pt_fwd_ms / tr_fwd_ms
        bwd_speedup = pt_bwd_ms / tr_bwd_ms if tr_bwd_ms > 0 else float('inf')
        total_speedup = pt_total_ms / tr_total_ms

        results.update({
            "triton_fwd_ms": tr_fwd_ms,
            "triton_bwd_ms": tr_bwd_ms,
            "triton_total_ms": tr_total_ms,
            "fwd_speedup": fwd_speedup,
            "bwd_speedup": bwd_speedup,
            "total_speedup": total_speedup,
        })

        print(f"\n  Speedup (PyTorch FD / Triton):")
        print(f"    forward:  {fwd_speedup:.2f}x")
        print(f"    backward: {bwd_speedup:.2f}x")
        print(f"    total:    {total_speedup:.2f}x")

    return results


def _patch_and_call(physics, loss_fn, ctx, fields):
    """Call a loss function with pre-computed fields instead of running the model.

    Temporarily patches physics.infer to return the given fields,
    then calls loss_fn with a dummy model.
    """
    original_infer = physics.infer

    def _patched_infer(model, ctx):
        return fields

    physics.infer = _patched_infer
    try:
        # Create a minimal dummy model that won't be called
        class _Dummy(torch.nn.Module):
            def forward(self, *args):
                raise RuntimeError("Should not be called")
        loss = loss_fn(_Dummy(), ctx)
    finally:
        physics.infer = original_infer

    return loss
