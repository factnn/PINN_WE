"""TGV 2D - Phy-CNN + Triton fused kernel (with P)."""
import sys, os; sys.path.insert(0, __import__('pathlib').Path(__file__).parent.parent.__str__())
import torch
from baseline.common_2d import *
from baseline.common_2d import infer
from kernels.stencil_2d import ns2d_residual_triton

class _NSTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, U, V, P, dx, dy, dt):
        ctx.save_for_backward(U, V, P)
        ctx.dx, ctx.dy, ctx.dt = dx, dy, dt
        return ns2d_residual_triton(U.contiguous(), V.contiguous(), P.contiguous(), dx, dy, dt, nu)

    @staticmethod
    def backward(ctx, grad):
        U, V, P = ctx.saved_tensors
        dx, dy, dt = ctx.dx, ctx.dy, ctx.dt
        Ua=U.detach().requires_grad_(True)
        Va=V.detach().requires_grad_(True)
        Pa=P.detach().requires_grad_(True)
        with torch.enable_grad():
            loss = pde_residual_pytorch(Ua, Va, Pa, dx, dy, dt)
        loss.backward(grad)
        return Ua.grad, Va.grad, Pa.grad, None, None, None

def loss_fn(model, xyt, X, Y, T, U_exact, V_exact, P_exact, dx, dy, dt):
    U, V, P = infer(model, xyt, X, Y, T)
    return _NSTriton.apply(U,V,P,dx,dy,dt) + ic_bc_loss(U,V,P,U_exact,V_exact,P_exact)

if __name__ == "__main__":
    args = base_argparser("Phy-CNN + Triton").parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    train_and_save("cnn_triton", PhyCNN, loss_fn, runs=args.runs,
                   max_epochs=args.max_epochs, lr=args.lr, loss_threshold=args.threshold)
