#!/usr/bin/env python3
"""Test script for stencil_2d.py — loads as proper module (not __main__)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Force clean Triton compilation
os.environ.setdefault('TRITON_CACHE_DIR', '/tmp/triton_stencil2d_test')

from kernels.stencil_2d import ns2d_residual_triton
from baseline.common_2d import pde_residual_pytorch
import torch, numpy as np

Nt, Nx, Ny = 10, 32, 32
nu_val = 0.01
device = "cuda"
dtype = torch.float64

x = torch.linspace(0, 2*np.pi, Nx, device=device, dtype=dtype)
y = torch.linspace(0, 2*np.pi, Ny, device=device, dtype=dtype)
t = torch.linspace(0, 1, Nt, device=device, dtype=dtype)
dx_v = float(x[1]-x[0]); dy_v = float(y[1]-y[0]); dt_v = float(t[1]-t[0])
T, X, Y = torch.meshgrid(t, x, y, indexing='ij')

def make_uvp(requires_grad=False):
    U = (torch.sin(X)*torch.cos(Y)).to(dtype).requires_grad_(requires_grad)
    V = (-torch.cos(X)*torch.sin(Y)).to(dtype).requires_grad_(requires_grad)
    P = (-0.25*(torch.cos(2*X)+torch.cos(2*Y))).to(dtype).requires_grad_(requires_grad)
    return U, V, P

# Forward check
U_pt, V_pt, P_pt = make_uvp(True)
U_tr, V_tr, P_tr = make_uvp(True)
loss_pt = pde_residual_pytorch(U_pt.float(), V_pt.float(), P_pt.float(), dx_v, dy_v, dt_v)
loss_tr = ns2d_residual_triton(U_tr.float(), V_tr.float(), P_tr.float(), dx_v, dy_v, dt_v, nu_val)
print(f"Forward rel_err: {abs(loss_pt.item()-loss_tr.item())/(abs(loss_pt.item())+1e-12):.2e}")

# Gradient check (float64)
U_pt2, V_pt2, P_pt2 = make_uvp(True)
U_tr2, V_tr2, P_tr2 = make_uvp(True)
loss_pt_d = pde_residual_pytorch(U_pt2, V_pt2, P_pt2, dx_v, dy_v, dt_v)
loss_tr_d = ns2d_residual_triton(U_tr2, V_tr2, P_tr2, dx_v, dy_v, dt_v, nu_val)
loss_pt_d.backward(); loss_tr_d.backward()
print(f"Grad U err: {(U_pt2.grad - U_tr2.grad).abs().max().item():.2e}")
print(f"Grad V err: {(V_pt2.grad - V_tr2.grad).abs().max().item():.2e}")
print(f"Grad P err: {(P_pt2.grad - P_tr2.grad).abs().max().item():.2e}")

# Benchmark
import triton
U_b, V_b, P_b = [v.float() for v in make_uvp()]
ms_tr = triton.testing.do_bench(lambda: ns2d_residual_triton(U_b, V_b, P_b, dx_v, dy_v, dt_v, nu_val))
print(f"Triton forward: {ms_tr:.3f}ms")
