"""1D compressible Euler Triton kernel for Sod shock tube.
ρ_t + (ρu)_x = 0
(ρu)_t + (ρu²+p)_x = 0
E_t + ((E+p)u)_x = 0
Data: ρ, ρu, E: [Nt, Nx]

TODO: implement euler1d_fwd_kernel, euler1d_bwd_kernel
"""
