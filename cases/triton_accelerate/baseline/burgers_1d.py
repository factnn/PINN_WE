"""
1D Burgers PINN baseline (PyTorch native finite difference).
PDE: u_t + u*u_x = nu*u_xx
Steady-state: u*u_x - nu*u_xx = 0, x in [-1,1], u(-1)=1, u(1)=-1
"""
import time
import torch
import torch.nn as nn
import numpy as np

nu = 0.01 / np.pi
N = 256  # grid points
device = "cuda"


class MLP(nn.Module):
    def __init__(self, width=64, depth=4):
        super().__init__()
        layers = [nn.Linear(1, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class PhyCNN1D(nn.Module):
    def __init__(self, channels=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, channels, 5, padding=2), nn.Tanh(),
            nn.Conv1d(channels, channels, 5, padding=2), nn.Tanh(),
            nn.Conv1d(channels, channels, 5, padding=2), nn.Tanh(),
            nn.Conv1d(channels, 1, 5, padding=2),
        )

    def forward(self, x):
        # x: (N,) -> (1,1,N) -> (N,)
        return self.net(x.unsqueeze(0).unsqueeze(0)).squeeze()


def pde_residual_pytorch(u, dx):
    """Native PyTorch finite difference: u*u_x - nu*u_xx"""
    u_x = (u[2:] - u[:-2]) / (2 * dx)          # central diff, interior
    u_xx = (u[2:] - 2 * u[1:-1] + u[:-2]) / dx**2
    return u[1:-1] * u_x - nu * u_xx


def bc_loss(u):
    return (u[0] - 1.0)**2 + (u[-1] + 1.0)**2


def train(model, epochs=5000, lr=1e-3, loss_threshold=1e-4):
    x = torch.linspace(-1, 1, N, device=device)
    dx = x[1] - x[0]
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    history = []          # (wall_time, epoch, loss)
    time_to_solution = None
    step_times = []

    t0 = time.time()
    for ep in range(1, epochs + 1):
        t_step = time.time()
        opt.zero_grad()
        if isinstance(model, MLP):
            u = model(x.unsqueeze(1)).squeeze()
        else:
            u = model(x)
        res = pde_residual_pytorch(u, dx)
        loss = res.pow(2).mean() + 100 * bc_loss(u)
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        step_times.append(time.time() - t_step)

        loss_val = loss.item()
        wall = time.time() - t0
        history.append((wall, ep, loss_val))

        if time_to_solution is None and loss_val < loss_threshold:
            time_to_solution = wall
            print(f"  >> Reached loss < {loss_threshold:.0e} at ep {ep}, t={wall:.2f}s")

        if ep % 1000 == 0:
            print(f"ep {ep:5d}  loss={loss_val:.3e}  t={wall:.1f}s")

    elapsed = time.time() - t0
    mem = torch.cuda.max_memory_allocated(device) / 1e9
    avg_step = np.mean(step_times) * 1000  # ms

    # kernel-level latency via CUDA events
    start_evt, end_evt = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    with torch.no_grad():
        u_test = model(x.unsqueeze(1)).squeeze() if isinstance(model, MLP) else model(x)
    start_evt.record(); pde_residual_pytorch(u_test, dx); end_evt.record()
    torch.cuda.synchronize()
    kernel_ms = start_evt.elapsed_time(end_evt)

    print(f"\n--- Metrics ---")
    print(f"Total wall time   : {elapsed:.2f} s")
    print(f"Avg step time     : {avg_step:.3f} ms")
    print(f"Kernel latency    : {kernel_ms:.4f} ms")
    print(f"Peak memory       : {mem:.4f} GB")
    print(f"Time-to-solution  : {time_to_solution:.2f} s" if time_to_solution else "Time-to-solution  : not reached")

    return dict(elapsed=elapsed, avg_step_ms=avg_step, kernel_ms=kernel_ms,
                peak_mem_gb=mem, time_to_solution=time_to_solution, history=history)


if __name__ == "__main__":
    print("=== CAN-PINN (MLP) ===")
    torch.cuda.reset_peak_memory_stats()
    mlp = MLP().to(device)
    train(mlp)

    print("\n=== Phy-CNN ===")
    torch.cuda.reset_peak_memory_stats()
    cnn = PhyCNN1D().to(device)
    train(cnn)
