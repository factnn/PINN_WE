"""TGV 3D - MLP Vanilla PINN (autograd)."""
import sys, os; sys.path.insert(0, __import__('pathlib').Path(__file__).parent.parent.parent.__str__())
import torch
from cases.tgv_3d.common import *

def loss_fn(model, xyzt, X, Y, Z, T, U_exact, V_exact, W_exact, P_exact, dx, dy, dz, dt):
    xyzt_g = xyzt.detach().requires_grad_(True)
    u_g, v_g, w_g, p_g = model(xyzt_g)
    u_g = u_g.reshape(Nt,Nx,Ny,Nz); v_g = v_g.reshape(Nt,Nx,Ny,Nz)
    w_g = w_g.reshape(Nt,Nx,Ny,Nz); p_g = p_g.reshape(Nt,Nx,Ny,Nz)
    gu = torch.autograd.grad(u_g.sum(), xyzt_g, create_graph=True)[0]
    gv = torch.autograd.grad(v_g.sum(), xyzt_g, create_graph=True)[0]
    gw = torch.autograd.grad(w_g.sum(), xyzt_g, create_graph=True)[0]
    gp = torch.autograd.grad(p_g.sum(), xyzt_g, create_graph=True)[0]
    u_x=gu[:,0].reshape(Nt,Nx,Ny,Nz); u_y=gu[:,1].reshape(Nt,Nx,Ny,Nz)
    u_z=gu[:,2].reshape(Nt,Nx,Ny,Nz); u_t=gu[:,3].reshape(Nt,Nx,Ny,Nz)
    v_x=gv[:,0].reshape(Nt,Nx,Ny,Nz); v_y=gv[:,1].reshape(Nt,Nx,Ny,Nz)
    v_z=gv[:,2].reshape(Nt,Nx,Ny,Nz); v_t=gv[:,3].reshape(Nt,Nx,Ny,Nz)
    w_x=gw[:,0].reshape(Nt,Nx,Ny,Nz); w_y=gw[:,1].reshape(Nt,Nx,Ny,Nz)
    w_z=gw[:,2].reshape(Nt,Nx,Ny,Nz); w_t=gw[:,3].reshape(Nt,Nx,Ny,Nz)
    p_x=gp[:,0].reshape(Nt,Nx,Ny,Nz); p_y=gp[:,1].reshape(Nt,Nx,Ny,Nz)
    p_z=gp[:,2].reshape(Nt,Nx,Ny,Nz)
    u_xx=torch.autograd.grad(u_x.sum(),xyzt_g,create_graph=True)[0][:,0].reshape(Nt,Nx,Ny,Nz)
    u_yy=torch.autograd.grad(u_y.sum(),xyzt_g,create_graph=True)[0][:,1].reshape(Nt,Nx,Ny,Nz)
    u_zz=torch.autograd.grad(u_z.sum(),xyzt_g,create_graph=True)[0][:,2].reshape(Nt,Nx,Ny,Nz)
    v_xx=torch.autograd.grad(v_x.sum(),xyzt_g,create_graph=True)[0][:,0].reshape(Nt,Nx,Ny,Nz)
    v_yy=torch.autograd.grad(v_y.sum(),xyzt_g,create_graph=True)[0][:,1].reshape(Nt,Nx,Ny,Nz)
    v_zz=torch.autograd.grad(v_z.sum(),xyzt_g,create_graph=True)[0][:,2].reshape(Nt,Nx,Ny,Nz)
    w_xx=torch.autograd.grad(w_x.sum(),xyzt_g,create_graph=True)[0][:,0].reshape(Nt,Nx,Ny,Nz)
    w_yy=torch.autograd.grad(w_y.sum(),xyzt_g,create_graph=True)[0][:,1].reshape(Nt,Nx,Ny,Nz)
    w_zz=torch.autograd.grad(w_z.sum(),xyzt_g,create_graph=True)[0][:,2].reshape(Nt,Nx,Ny,Nz)
    res_u = u_t + u_g*u_x + v_g*u_y + w_g*u_z + p_x - nu*(u_xx+u_yy+u_zz)
    res_v = v_t + u_g*v_x + v_g*v_y + w_g*v_z + p_y - nu*(v_xx+v_yy+v_zz)
    res_w = w_t + u_g*w_x + v_g*w_y + w_g*w_z + p_z - nu*(w_xx+w_yy+w_zz)
    res_div = u_x + v_y + w_z
    return (res_u**2+res_v**2+res_w**2+res_div**2).mean() + \
           ic_bc_loss(u_g, v_g, w_g, p_g, U_exact, V_exact, W_exact, P_exact)

if __name__ == "__main__":
    args = base_argparser("MLP Vanilla PINN").parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    train_and_save("mlp_vanilla", MLP, loss_fn, runs=args.runs,
                   max_epochs=args.max_epochs, lr=args.lr, loss_threshold=args.threshold)
