"""
2D Allen-Cahn PINN baseline (PyTorch native finite difference).
PDE: -eps^2 * (u_xx + u_yy) + u^3 - u = 0, x,y in [-1,1]
BC: u=tanh(x/sqrt(2)*eps) on boundary (exact solution for 1D profile)
eps=0.1
"""
import time
import torch
import torch.nn as nn
import numpy as np

eps = 0.1
N = 128  # grid points per dim
device = "cuda"


class MLP(nn.Module):
    def __init__(self, width=64, depth=5):
        super().__init__()
        layers = [nn.Linear(2, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, xy):
        return self.net(xy).squeeze(-1)


class PhyCNN2D(nn.Module):
    def __init__(self, channels=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, channels, 5, padding=2), nn.Tanh(),
            nn.Conv2d(channels, channels, 5, padding=2), nn.Tanh(),
            nn.Conv2d(channels, channels, 5, padding=2), nn.Tanh(),
            nn.Conv2d(channels, 1, 5, padding=2),
        )

    def forward(self, u_grid):
        # u_grid: (N,N) -> (1,1,N,N) -> (N,N)
        return self.net(u_grid.unsqueeze(0).unsqueeze(0)).squeeze()


def pde_residual_pytorch(u, dx):
    """Native PyTorch: -eps^2*(u_xx+u_yy) + u^3 - u, interior only"""
    u_xx = (u[2:, 1:-1] - 2*u[1:-1, 1:-1] + u[:-2, 1:-1]) / dx**2
    u_yy = (u[1:-1, 2:] - 2*u[1:-1, 1:-1] + u[1:-1, :-2]) / dx**2
    u_int = u[1:-1, 1:-1]
    return -eps**2 * (u_xx + u_yy) + u_int**3 - u_int


def bc_loss(u, x_grid, y_grid):
    u_exact_x0 = torch.tanh(x_grid[0, :] / (np.sqrt(2) * eps))
    u_exact_x1 = torch.tanh(x_grid[-1, :] / (np.sqrt(2) * eps))
    u_exact_y0 = torch.tanh(y_grid[:, 0] / (np.sqrt(2) * eps))
    u_exact_y1 = torch.tanh(y_grid[:, -1] / (np.sqrt(2) * eps))
    return ((u[0, :] - u_exact_x0)**2 + (u[-1, :] - u_exact_x1)**2 +
            (u[:, 0] - u_exact_y0)**2 + (u[:, -1] - u_exact_y1)**2).mean()


def train_mlp(epochs=3000, lr=1e-3):
    x = torch.linspace(-1, 1, N, device=device)
    y = torch.linspace(-1, 1, N, device=device)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    dx = float(x[1] - x[0])
    xy = torch.stack([X.flatten(), Y.flatten()], dim=1)

    model = MLP().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    t0 = time.time()
    for ep in range(1, epochs + 1):
        opt.zero_grad()
        u = model(xy).reshape(N, N)
        res = pde_residual_pytorch(u, dx)
        loss = res.pow(2).mean() + 100 * bc_loss(u, X, Y)
        loss.backward()
        opt.step()
        if ep % 500 == 0:
            print(f"ep {ep:4d}  loss={loss.item():.3e}  t={time.time()-t0:.1f}s")

    elapsed = time.time() - t0
    mem = torch.cuda.max_memory_allocated(device) / 1e9
    print(f"Total: {elapsed:.2f}s  Peak mem: {mem:.3f} GB")
    return elapsed, mem


def train_cnn(epochs=3000, lr=1e-3):
    x = torch.linspace(-1, 1, N, device=device)
    y = torch.linspace(-1, 1, N, device=device)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    dx = float(x[1] - x[0])

    model = PhyCNN2D().to(device)
    # CNN output is the u field directly on the grid
    u_param = nn.Parameter(torch.zeros(N, N, device=device))
    opt = torch.optim.Adam(list(model.parameters()) + [u_param], lr=lr)

    t0 = time.time()
    for ep in range(1, epochs + 1):
        opt.zero_grad()
        u = model(u_param)
        res = pde_residual_pytorch(u, dx)
        loss = res.pow(2).mean() + 100 * bc_loss(u, X, Y)
        loss.backward()
        opt.step()
        if ep % 500 == 0:
            print(f"ep {ep:4d}  loss={loss.item():.3e}  t={time.time()-t0:.1f}s")

    elapsed = time.time() - t0
    mem = torch.cuda.max_memory_allocated(device) / 1e9
    print(f"Total: {elapsed:.2f}s  Peak mem: {mem:.3f} GB")
    return elapsed, mem


if __name__ == "__main__":
    print("=== CAN-PINN (MLP) 2D Allen-Cahn ===")
    torch.cuda.reset_peak_memory_stats()
    train_mlp()

    print("\n=== Phy-CNN 2D Allen-Cahn ===")
    torch.cuda.reset_peak_memory_stats()
    train_cnn()
