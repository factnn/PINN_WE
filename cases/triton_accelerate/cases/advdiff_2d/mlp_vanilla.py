"""AdvDiff 2D - MLP Vanilla PINN (autograd)."""
import sys, os; sys.path.insert(0, __import__('pathlib').Path(__file__).parent.parent.parent.__str__())
import torch
from cases.advdiff_2d.common import *

def loss_fn(model, xyt, X, Y, T, C_exact, dx, dy, dt):
    xyt_g = xyt.detach().requires_grad_(True)
    c_g = model(xyt_g).reshape(Nt, Nx, Ny)
    gc = torch.autograd.grad(c_g.sum(), xyt_g, create_graph=True)[0]
    c_x = gc[:,0].reshape(Nt, Nx, Ny); c_y = gc[:,1].reshape(Nt, Nx, Ny)
    c_t = gc[:,2].reshape(Nt, Nx, Ny)
    c_xx = torch.autograd.grad(c_x.sum(), xyt_g, create_graph=True)[0][:,0].reshape(Nt, Nx, Ny)
    c_yy = torch.autograd.grad(c_y.sum(), xyt_g, create_graph=True)[0][:,1].reshape(Nt, Nx, Ny)
    res = c_t + u0*c_x + v0*c_y - nu*(c_xx + c_yy)
    return (res**2).mean() + ic_bc_loss(c_g, C_exact)

if __name__ == "__main__":
    args = base_argparser("MLP Vanilla PINN").parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    train_and_save("mlp_vanilla", MLP, loss_fn, runs=args.runs,
                   max_epochs=args.max_epochs, lr=args.lr, loss_threshold=args.threshold)
