"""LDC 2D - MLP + Triton fused kernel (steady-state, full Triton fwd+bwd)."""
import sys, os; sys.path.insert(0, __import__('pathlib').Path(__file__).parent.parent.parent.__str__())
import torch
from cases.ldc_2d.common import *
from kernels.stencil_2d_burgers_steady import ldc_residual_triton

def loss_fn(model, xy, X, Y, dx, dy):
    U, V, P = infer(model, xy, X, Y)
    return ldc_residual_triton(U, V, P, dx, dy, nu) + 10 * bc_loss(U, V)

if __name__ == "__main__":
    args = base_argparser("MLP + Triton").parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    train_and_save("mlp_triton", MLP, loss_fn, runs=args.runs,
                   max_epochs=args.max_epochs, lr=args.lr, loss_threshold=args.threshold)
