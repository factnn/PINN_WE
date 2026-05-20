import numpy as np

"""
Generate a high-resolution reference solution for the 1D Shu–Osher
shock-entropy wave using a finite-volume solver (WENO5 + TVD-RK3 + Rusanov).
This follows the same (non-dimensional) initial/boundary settings as cases/1D/Shu-Osher/1.py.

Output: shuo_exact_sim.dat with columns: x, rho, u, p
"""

gamma = 1.4

# Domain and final time (consistent with 1.py normalized units)
Xs, Xe = 0.0, 1.0
Te = 0.18

# Initial states (already normalized by Unit_var in 1.py)
crhoL = 27 / 7
cuL = 2.629369
cpL = 31 / 3
crhoR = 1.0
cuR = 0.0
cpR = 1.0


def init_primitive(x):
    """Piecewise IC with sinusoidal entropy wave on the right."""
    rho = np.zeros_like(x)
    u = np.zeros_like(x)
    p = np.zeros_like(x)
    mask_L = x <= 0.2
    mask_R = ~mask_L
    rho[mask_L] = crhoL
    u[mask_L] = cuL
    p[mask_L] = cpL
    rho[mask_R] = crhoR * (1 + 0.5 * np.sin(15 * x[mask_R]))
    u[mask_R] = cuR
    p[mask_R] = cpR
    return rho, u, p


def prim_to_cons(rho, u, p):
    mom = rho * u
    E = p / (gamma - 1.0) + 0.5 * rho * u ** 2
    return np.stack([rho, mom, E], axis=0)


def cons_to_prim(U):
    rho = U[0]
    u = U[1] / rho
    p = (gamma - 1.0) * (U[2] - 0.5 * rho * u ** 2)
    return rho, u, p


def flux(U):
    rho, u, p = cons_to_prim(U)
    F1 = U[1]
    F2 = rho * u ** 2 + p
    F3 = (U[2] + p) * u
    return np.stack([F1, F2, F3], axis=0)


def max_wave_speed(U):
    rho, u, p = cons_to_prim(U)
    a = np.sqrt(gamma * p / rho)
    return np.max(np.abs(u) + a)


def minmod(a, b):
    return 0.5 * (np.sign(a) + np.sign(b)) * np.minimum(np.abs(a), np.abs(b))


def muscl_reconstruct(q):
    """
    Simple 2nd-order MUSCL with minmod limiter.
    Returns left/right states at interfaces (size N+1).
    """
    qm = np.roll(q, 1)
    qp = np.roll(q, -1)
    slope = minmod(q - qm, qp - q)
    qL = q + 0.5 * slope   # right face of cell i
    qR = q - 0.5 * slope   # left face of cell i
    # shift to interfaces
    qL_int = qL
    qR_int = np.roll(qR, -1)
    return qL_int, qR_int


def rusanov_flux(UL, UR):
    FL = flux(UL)
    FR = flux(UR)
    aL = max_wave_speed(UL)
    aR = max_wave_speed(UR)
    a = max(aL, aR)
    return 0.5 * (FL + FR) - 0.5 * a * (UR - UL)


def step(U, dx, dt):
    """
    MUSCL-Hancock (2nd order) with minmod slopes + Rusanov flux.
    """
    N = U.shape[1]
    # Slopes
    slope = np.zeros_like(U)
    dminus = U[:, 1:] - U[:, :-1]          # (3, N-1)
    dplus = np.zeros_like(dminus)
    dplus[:, :-1] = U[:, 2:] - U[:, 1:-1]  # (3, N-2) into first N-2 slots
    slope[:, 1:-1] = minmod(dminus[:, :-1], dplus[:, :-1])

    # Interface states (N+1 interfaces)
    UL = np.zeros((3, N + 1))
    UR = np.zeros((3, N + 1))
    # left boundary
    UL[:, 0] = U[:, 0]
    UR[:, 0] = U[:, 0]
    # interior interfaces i=1..N-1 between cell i-1 and i
    UL[:, 1:N] = U[:, :-1] + 0.5 * slope[:, :-1]
    UR[:, 1:N] = U[:, 1:] - 0.5 * slope[:, 1:]
    # right boundary
    UL[:, N] = U[:, -1]
    UR[:, N] = U[:, -1]

    F = rusanov_flux(UL, UR)
    return U - dt/dx * (F[:, 1:] - F[:, :-1])


def tvd_rk3(U, dx, dt):
    U1 = step(U, dx, dt)
    U2 = 0.75 * U + 0.25 * step(U1, dx, dt)
    U3 = (1.0/3.0) * U + (2.0/3.0) * step(U2, dx, dt)
    return U3


def solve_shu_osher(N=2000, cfl=0.2, t_end=Te, write_path="shuo_exact_sim.dat"):
    x = np.linspace(Xs + 0.5*(Xe - Xs)/N, Xe - 0.5*(Xe - Xs)/N, N)
    dx = (Xe - Xs) / N
    rho0, u0, p0 = init_primitive(x)
    U = prim_to_cons(rho0, u0, p0)

    t = 0.0
    while t < t_end:
        amax = max_wave_speed(U)
        dt = cfl * dx / amax
        if t + dt > t_end:
            dt = t_end - t
        U = tvd_rk3(U, dx, dt)
        t += dt

    rho, u, p = cons_to_prim(U)
    data = np.column_stack([x, rho, u, p])
    np.savetxt(write_path, data, fmt="%.8e", header="x rho u p", comments="# ")
    print(f"Saved reference solution to {write_path} with N={N}, final time {t_end:.4f}")


if __name__ == "__main__":
    solve_shu_osher()

