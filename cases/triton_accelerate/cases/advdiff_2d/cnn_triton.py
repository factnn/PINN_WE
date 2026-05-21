"""AdvDiff 2D - Phy-CNN + Triton fused kernel (full Triton fwd+bwd)."""
import sys, os; sys.path.insert(0, __import__('pathlib').Path(__file__).parent.parent.parent.__str__())
import torch
from cases.advdiff_2d.common import *
from kernels.stencil_advdiff import advdiff_residual_triton

def loss_fn(model, xyt, X, Y, T, C_exact, dx, dy, dt):
    C = infer(model, xyt, X, Y, T)
    return advdiff_residual_triton(C, dx, dy, dt, u0, v0, nu) + ic_bc_loss(C, C_exact)

if __name__ == "__main__":
    args = base_argparser("Phy-CNN + Triton").parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    train_and_save("cnn_triton", PhyCNN, loss_fn, runs=args.runs,
                   max_epochs=args.max_epochs, lr=args.lr, loss_threshold=args.threshold)
