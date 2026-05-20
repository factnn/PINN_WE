import torch
import time
import sys
import os

# 添加PINNsrc路径
current_file_dir = os.path.dirname(os.path.abspath(__file__))
pinn_src_path = os.path.join(current_file_dir, '..', '..', '..', 'PINNsrc')
pinn_src_path = os.path.abspath(pinn_src_path)
if pinn_src_path not in sys.path:
    sys.path.insert(0, pinn_src_path)

# 导入黎曼求解器
from riemann_solver import BurgersSolver

def test_correctness():
    """测试 1: 验证 Triton 算出来的结果和 PyTorch 是否一致"""
    print("\n=== Test 1: Correctness Check ===")
    torch.manual_seed(42)
    N = 10000
    
    # 生成随机测试数据 (包含正负值，模拟激波和稀疏波)
    ul = torch.randn(N, device='cuda')
    ur = torch.randn(N, device='cuda')
    
    # 分别运行
    flux_torch = BurgersSolver.godunov_flux(ul, ur, backend='torch')
    flux_triton = BurgersSolver.godunov_flux(ul, ur, backend='triton')
    
    # 比较差异
    if torch.allclose(flux_torch, flux_triton, atol=1e-6):
        print("✅ Pass! Triton result matches PyTorch result.")
    else:
        print("❌ Fail! Results diverge.")
        diff = (flux_torch - flux_triton).abs().max()
        print(f"Max difference: {diff.item()}")

def benchmark_speed():
    """测试 2: 测速 (Benchmark)"""
    print("\n=== Test 2: Speed Benchmark ===")
    # 数据量大一点，才能看出差距 (例如 1000万 个网格点)
    N = 10 * 1024 * 1024 
    ul = torch.randn(N, device='cuda')
    ur = torch.randn(N, device='cuda')
    
    # 预热 (Warmup) - 防止第一次启动编译时间干扰
    for _ in range(10):
        BurgersSolver.godunov_flux(ul, ur, backend='triton')
        BurgersSolver.godunov_flux(ul, ur, backend='torch')
    
    # 1. 测 PyTorch 时间
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        BurgersSolver.godunov_flux(ul, ur, backend='torch')
    torch.cuda.synchronize()
    torch_time = (time.time() - start) / 100
    print(f"PyTorch Average Time: {torch_time*1000:.3f} ms")

    # 2. 测 Triton 时间
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        BurgersSolver.godunov_flux(ul, ur, backend='triton')
    torch.cuda.synchronize()
    triton_time = (time.time() - start) / 100
    print(f"Triton  Average Time: {triton_time*1000:.3f} ms")
    
    print(f"🚀 Speedup: {torch_time / triton_time:.2f}x")

if __name__ == "__main__":
    if torch.cuda.is_available():
        test_correctness()
        benchmark_speed()
    else:
        print("Need GPU to run Triton!")