"""TGV 2D - MLP Vanilla PINN (autograd)."""
import sys, os; sys.path.insert(0, __import__('pathlib').Path(__file__).parent.parent.parent.__str__())
import torch
from cases.tgv_2d.common import *

def loss_fn(model, xyt, X, Y, T, U_exact, V_exact, P_exact, dx, dy, dt):
    xyt_g = xyt.detach().requires_grad_(True)
    u_g, v_g, p_g = model(xyt_g)
    u_g=u_g.reshape(Nt,Nx,Ny); v_g=v_g.reshape(Nt,Nx,Ny); p_g=p_g.reshape(Nt,Nx,Ny)
    gu=torch.autograd.grad(u_g.sum(),xyt_g,create_graph=True)[0]
    gv=torch.autograd.grad(v_g.sum(),xyt_g,create_graph=True)[0]
    u_x=gu[:,0].reshape(Nt,Nx,Ny); u_y=gu[:,1].reshape(Nt,Nx,Ny); u_t=gu[:,2].reshape(Nt,Nx,Ny)
    v_x=gv[:,0].reshape(Nt,Nx,Ny); v_y=gv[:,1].reshape(Nt,Nx,Ny); v_t=gv[:,2].reshape(Nt,Nx,Ny)
    u_xx=torch.autograd.grad(u_x.sum(),xyt_g,create_graph=True)[0][:,0].reshape(Nt,Nx,Ny)
    u_yy=torch.autograd.grad(u_y.sum(),xyt_g,create_graph=True)[0][:,1].reshape(Nt,Nx,Ny)
    v_xx=torch.autograd.grad(v_x.sum(),xyt_g,create_graph=True)[0][:,0].reshape(Nt,Nx,Ny)
    v_yy=torch.autograd.grad(v_y.sum(),xyt_g,create_graph=True)[0][:,1].reshape(Nt,Nx,Ny)
    gp=torch.autograd.grad(p_g.sum(),xyt_g,create_graph=True)[0]
    p_x=gp[:,0].reshape(Nt,Nx,Ny); p_y=gp[:,1].reshape(Nt,Nx,Ny)
    res_u=u_t+u_g*u_x+v_g*u_y+p_x-nu*(u_xx+u_yy)
    res_v=v_t+u_g*v_x+v_g*v_y+p_y-nu*(v_xx+v_yy)
    res_div=u_x+v_y
    return (res_u**2+res_v**2+res_div**2).mean() + ic_bc_loss(u_g,v_g,p_g,U_exact,V_exact,P_exact)

if __name__ == "__main__":
    args = base_argparser("MLP Vanilla PINN").parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    train_and_save("mlp_vanilla", MLP, loss_fn, runs=args.runs,
                   max_epochs=args.max_epochs, lr=args.lr, loss_threshold=args.threshold)
