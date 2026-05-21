"""Steady Burgers 1D - MLP Vanilla PINN (autograd)."""
import sys, os; sys.path.insert(0, __import__('pathlib').Path(__file__).parent.parent.parent.__str__())
import torch
from cases.burgers_1d_steady.common import *

def loss_fn(model, x_inp, X, dx):
    x_g = x_inp.detach().requires_grad_(True)
    u_g = model(x_g)
    u_x = torch.autograd.grad(u_g.sum(), x_g, create_graph=True)[0].squeeze(-1)
    u_xx = torch.autograd.grad(u_x.sum(), x_g, create_graph=True)[0].squeeze(-1)
    res = u_g * u_x - nu * u_xx
    return (res**2).mean() + 10 * bc_loss(u_g)

if __name__ == "__main__":
    args = base_argparser("MLP Vanilla PINN").parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    train_and_save("mlp_vanilla", MLP, loss_fn, runs=args.runs,
                   max_epochs=args.max_epochs, lr=args.lr, loss_threshold=args.threshold)
