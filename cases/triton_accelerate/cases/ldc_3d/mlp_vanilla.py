"""LDC 3D - MLP Vanilla PINN (autograd)."""
import sys, os; sys.path.insert(0, __import__('pathlib').Path(__file__).parent.parent.parent.__str__())
import torch
from cases.ldc_3d.common import *

def loss_fn(model, xyz, X, Y, Z, dx, dy, dz):
    xyz_g = xyz.detach().requires_grad_(True)
    u_g, v_g, w_g, p_g = model(xyz_g)
    u_g = u_g.reshape(Nx,Ny,Nz); v_g = v_g.reshape(Nx,Ny,Nz); w_g = w_g.reshape(Nx,Ny,Nz); p_g = p_g.reshape(Nx,Ny,Nz)
    gu = torch.autograd.grad(u_g.sum(), xyz_g, create_graph=True)[0]
    gv = torch.autograd.grad(v_g.sum(), xyz_g, create_graph=True)[0]
    gw = torch.autograd.grad(w_g.sum(), xyz_g, create_graph=True)[0]
    gp = torch.autograd.grad(p_g.sum(), xyz_g, create_graph=True)[0]
    u_x=gu[:,0].reshape(Nx,Ny,Nz); u_y=gu[:,1].reshape(Nx,Ny,Nz); u_z=gu[:,2].reshape(Nx,Ny,Nz)
    v_x=gv[:,0].reshape(Nx,Ny,Nz); v_y=gv[:,1].reshape(Nx,Ny,Nz); v_z=gv[:,2].reshape(Nx,Ny,Nz)
    w_x=gw[:,0].reshape(Nx,Ny,Nz); w_y=gw[:,1].reshape(Nx,Ny,Nz); w_z=gw[:,2].reshape(Nx,Ny,Nz)
    p_x=gp[:,0].reshape(Nx,Ny,Nz); p_y=gp[:,1].reshape(Nx,Ny,Nz); p_z=gp[:,2].reshape(Nx,Ny,Nz)
    u_xx=torch.autograd.grad(u_x.sum(),xyz_g,create_graph=True)[0][:,0].reshape(Nx,Ny,Nz)
    u_yy=torch.autograd.grad(u_y.sum(),xyz_g,create_graph=True)[0][:,1].reshape(Nx,Ny,Nz)
    u_zz=torch.autograd.grad(u_z.sum(),xyz_g,create_graph=True)[0][:,2].reshape(Nx,Ny,Nz)
    v_xx=torch.autograd.grad(v_x.sum(),xyz_g,create_graph=True)[0][:,0].reshape(Nx,Ny,Nz)
    v_yy=torch.autograd.grad(v_y.sum(),xyz_g,create_graph=True)[0][:,1].reshape(Nx,Ny,Nz)
    v_zz=torch.autograd.grad(v_z.sum(),xyz_g,create_graph=True)[0][:,2].reshape(Nx,Ny,Nz)
    w_xx=torch.autograd.grad(w_x.sum(),xyz_g,create_graph=True)[0][:,0].reshape(Nx,Ny,Nz)
    w_yy=torch.autograd.grad(w_y.sum(),xyz_g,create_graph=True)[0][:,1].reshape(Nx,Ny,Nz)
    w_zz=torch.autograd.grad(w_z.sum(),xyz_g,create_graph=True)[0][:,2].reshape(Nx,Ny,Nz)
    res_u = u_g*u_x + v_g*u_y + w_g*u_z + p_x - nu*(u_xx+u_yy+u_zz)
    res_v = u_g*v_x + v_g*v_y + w_g*v_z + p_y - nu*(v_xx+v_yy+v_zz)
    res_w = u_g*w_x + v_g*w_y + w_g*w_z + p_z - nu*(w_xx+w_yy+w_zz)
    res_div = u_x + v_y + w_z
    return (res_u**2+res_v**2+res_w**2+res_div**2).mean() + 10*bc_loss(u_g, v_g, w_g)

if __name__ == "__main__":
    args = base_argparser("MLP Vanilla PINN").parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    train_and_save("mlp_vanilla", MLP, loss_fn, runs=args.runs,
                   max_epochs=args.max_epochs, lr=args.lr, loss_threshold=args.threshold)
