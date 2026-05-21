"""TGV 2D - MLP + Triton fused kernel (with P, full Triton fwd+bwd)."""
import sys, os; sys.path.insert(0, __import__('pathlib').Path(__file__).parent.parent.parent.__str__())
import torch
from cases.tgv_2d.common import *
from kernels.stencil_2d_ns_unsteady import ns2d_residual_triton, ns2d_fwd_kernel, ns2d_bwd_kernel, _add_boundary_gradients

class _NSTriton(torch.autograd.Function):
    """Full Triton forward+backward (no PyTorch fallback)."""
    @staticmethod
    def forward(ctx, U, V, P, dx, dy, dt):
        U, V, P = U.contiguous(), V.contiguous(), P.contiguous()
        Nt, Nx, Ny = U.shape
        res_u = torch.empty((Nt-2, Nx-2, Ny-2), device=U.device, dtype=U.dtype)
        res_v = torch.empty_like(res_u)
        res_div = torch.empty_like(res_u)
        BLOCK_X, BLOCK_Y = 16, 16
        grid = (Nt-2, (Nx-2+BLOCK_X-1)//BLOCK_X, (Ny-2+BLOCK_Y-1)//BLOCK_Y)
        ns2d_fwd_kernel[grid](U, V, P, res_u, res_v, res_div,
                              Nt, Nx, Ny, dx, dy, dt, nu, BLOCK_X, BLOCK_Y)
        ctx.save_for_backward(U, V)
        ctx.res_u, ctx.res_v, ctx.res_div = res_u, res_v, res_div
        ctx.dx, ctx.dy, ctx.dt = dx, dy, dt
        return (res_u**2 + res_v**2 + res_div**2).mean()

    @staticmethod
    def backward(ctx, grad_out):
        U, V = ctx.saved_tensors
        res_u, res_v, res_div = ctx.res_u, ctx.res_v, ctx.res_div
        dx, dy, dt = ctx.dx, ctx.dy, ctx.dt
        Nt, Nx, Ny = U.shape
        N_total = res_u.numel()
        scale = (2.0 / N_total) * grad_out
        Gu = res_u * scale; Gv = res_v * scale; Gdiv = res_div * scale
        grad_u = torch.zeros_like(U); grad_v = torch.zeros_like(V); grad_p = torch.zeros_like(U)
        BLOCK_X, BLOCK_Y = 16, 16
        grid = (Nt-2, (Nx-2+BLOCK_X-1)//BLOCK_X, (Ny-2+BLOCK_Y-1)//BLOCK_Y)
        ns2d_bwd_kernel[grid](U, V, Gu, Gv, Gdiv, grad_u, grad_v, grad_p,
                              Nt, Nx, Ny, dx, dy, dt, nu, BLOCK_X, BLOCK_Y)
        _add_boundary_gradients(U, V, Gu, Gv, Gdiv, grad_u, grad_v, grad_p, dx, dy, dt, nu)
        return grad_u, grad_v, grad_p, None, None, None

def loss_fn(model, xyt, X, Y, T, U_exact, V_exact, P_exact, dx, dy, dt):
    u_f, v_f, p_f = model(xyt)
    U=u_f.reshape(Nt,Nx,Ny); V=v_f.reshape(Nt,Nx,Ny); P=p_f.reshape(Nt,Nx,Ny)
    return _NSTriton.apply(U,V,P,dx,dy,dt) + ic_bc_loss(U,V,P,U_exact,V_exact,P_exact)

if __name__ == "__main__":
    args = base_argparser("MLP + Triton").parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    train_and_save("mlp_triton", MLP, loss_fn, runs=args.runs,
                   max_epochs=args.max_epochs, lr=args.lr, loss_threshold=args.threshold)
