"""Steady Burgers 1D - MLP + Triton fused kernel (full Triton fwd+bwd)."""
import sys, os; sys.path.insert(0, __import__('pathlib').Path(__file__).parent.parent.parent.__str__())
import torch
from cases.burgers_1d_steady.common import *
from kernels.stencil_1d_steady import burgers_steady_loss_triton

def loss_fn(model, x_inp, X, dx):
    U = infer(model, x_inp, X)
    return burgers_steady_loss_triton(U, dx, nu) + 10 * bc_loss(U)

if __name__ == "__main__":
    args = base_argparser("MLP + Triton").parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    train_and_save("mlp_triton", MLP, loss_fn, runs=args.runs,
                   max_epochs=args.max_epochs, lr=args.lr, loss_threshold=args.threshold)
