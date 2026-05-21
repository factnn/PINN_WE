"""3D steady-state NS Triton kernel for LDC 3D.
u*u_x + v*u_y + w*u_z = nu*(u_xx + u_yy + u_zz), same for v, w.
Data: U, V, W: [Nx, Ny, Nz]

TODO: implement ns3d_steady_fwd_kernel, ns3d_steady_bwd_kernel
"""
