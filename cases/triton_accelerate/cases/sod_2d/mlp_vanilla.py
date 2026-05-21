"""Euler 2D (Sod) - MLP Vanilla PINN (autograd)."""
import sys, os; sys.path.insert(0, __import__('pathlib').Path(__file__).parent.parent.parent.__str__())
import torch
from cases.sod_2d.common import *

def loss_fn(model, xt, X, T, dx, dt):
    xt_g = xt.detach().requires_grad_(True)
    rho, rhou, E = model(xt_g)
    rho = rho.reshape(Nt, Nx); rhou = rhou.reshape(Nt, Nx); E = E.reshape(Nt, Nx)
    u = rhou / (rho + 1e-10)
    p = (gamma - 1) * (E - 0.5 * rho * u**2)
    # autograd derivatives
    g_rho = torch.autograd.grad(rho.sum(), xt_g, create_graph=True)[0]
    rho_x = g_rho[:,0].reshape(Nt,Nx); rho_t = g_rho[:,1].reshape(Nt,Nx)
    g_rhou = torch.autograd.grad(rhou.sum(), xt_g, create_graph=True)[0]
    rhou_x = g_rhou[:,0].reshape(Nt,Nx); rhou_t = g_rhou[:,1].reshape(Nt,Nx)
    g_E = torch.autograd.grad(E.sum(), xt_g, create_graph=True)[0]
    E_x = g_E[:,0].reshape(Nt,Nx); E_t = g_E[:,1].reshape(Nt,Nx)
    g_p = torch.autograd.grad(p.sum(), xt_g, create_graph=True)[0]
    p_x = g_p[:,0].reshape(Nt,Nx)
    res1 = rho_t + rhou_x
    res2 = rhou_t + torch.autograd.grad((rhou*u).sum(), xt_g, create_graph=True)[0][:,0].reshape(Nt,Nx) + p_x
    res3 = E_t + torch.autograd.grad(((E+p)*u).sum(), xt_g, create_graph=True)[0][:,0].reshape(Nt,Nx)
    return (res1**2 + res2**2 + res3**2).mean() + \
           10 * ic_loss(rho, rhou, E, X) + bc_loss_euler(rho, rhou, E)

if __name__ == "__main__":
    args = base_argparser("MLP Vanilla PINN").parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    train_and_save("mlp_vanilla", MLP, loss_fn, runs=args.runs,
                   max_epochs=args.max_epochs, lr=args.lr, loss_threshold=args.threshold)
