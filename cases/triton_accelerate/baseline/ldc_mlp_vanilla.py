"""LDC 2D - MLP Vanilla PINN (autograd)."""
import sys, os; sys.path.insert(0, __import__('pathlib').Path(__file__).parent.parent.__str__())
import torch
from baseline.common_ldc import *

def loss_fn(model, xy, X, Y, dx, dy):
    xy_g = xy.detach().requires_grad_(True)
    u_g, v_g = model(xy_g)
    u_g = u_g.reshape(Nx, Ny); v_g = v_g.reshape(Nx, Ny)
    gu = torch.autograd.grad(u_g.sum(), xy_g, create_graph=True)[0]
    gv = torch.autograd.grad(v_g.sum(), xy_g, create_graph=True)[0]
    u_x = gu[:,0].reshape(Nx, Ny); u_y = gu[:,1].reshape(Nx, Ny)
    v_x = gv[:,0].reshape(Nx, Ny); v_y = gv[:,1].reshape(Nx, Ny)
    u_xx = torch.autograd.grad(u_x.sum(), xy_g, create_graph=True)[0][:,0].reshape(Nx, Ny)
    u_yy = torch.autograd.grad(u_y.sum(), xy_g, create_graph=True)[0][:,1].reshape(Nx, Ny)
    v_xx = torch.autograd.grad(v_x.sum(), xy_g, create_graph=True)[0][:,0].reshape(Nx, Ny)
    v_yy = torch.autograd.grad(v_y.sum(), xy_g, create_graph=True)[0][:,1].reshape(Nx, Ny)
    res_u = u_g*u_x + v_g*u_y - nu*(u_xx + u_yy)
    res_v = u_g*v_x + v_g*v_y - nu*(v_xx + v_yy)
    return (res_u**2 + res_v**2).mean() + 10*bc_loss(u_g, v_g)

if __name__ == "__main__":
    args = base_argparser("MLP Vanilla PINN").parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    train_and_save("mlp_vanilla", MLP, loss_fn, runs=args.runs,
                   max_epochs=args.max_epochs, lr=args.lr, loss_threshold=args.threshold)
