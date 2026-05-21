"""1D Unsteady Burgers - MLP Vanilla PINN (autograd)."""
import sys, os; sys.path.insert(0, __import__('pathlib').Path(__file__).parent.parent.parent.__str__())
import torch
from cases.burgers_1d_unsteady.common import *

def loss_fn(model, xt, X, T, dx, dt):
    xt_g = xt.detach().requires_grad_(True)
    u_g = model(xt_g).reshape(Nt, Nx)
    gu = torch.autograd.grad(u_g.sum(), xt_g, create_graph=True)[0]
    u_x = gu[:,0].reshape(Nt, Nx); u_t = gu[:,1].reshape(Nt, Nx)
    u_xx = torch.autograd.grad(u_x.sum(), xt_g, create_graph=True)[0][:,0].reshape(Nt, Nx)
    res = u_t + u_g * u_x - nu * u_xx
    return (res**2).mean() + 10*ic_loss_from_U(u_g, X) + 10*bc_loss_from_U(u_g)

if __name__ == "__main__":
    args = base_argparser("MLP Vanilla PINN").parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    train_and_save("mlp_vanilla", MLP, loss_fn, runs=args.runs,
                   max_epochs=args.max_epochs, lr=args.lr, loss_threshold=args.threshold)
