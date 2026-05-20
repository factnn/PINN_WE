import torch
import triton
import triton.language as tl

# ============================================================================
# Part A: Triton Kernel (GPU 高性能实现) - 你的论文核心贡献
# ============================================================================

@triton.jit
def _burgers_godunov_kernel(
    ul_ptr, ur_ptr, flux_ptr,  # 指针
    n_elements,                # 元素总数
    BLOCK_SIZE: tl.constexpr   # 块大小
):
    # 1. 确定当前线程处理的索引
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # 2. 加载数据 (Load)
    u_l = tl.load(ul_ptr + offsets, mask=mask)
    u_r = tl.load(ur_ptr + offsets, mask=mask)

    # 3. 物理公式：Burgers Flux f(u) = 0.5 * u^2
    f_l = 0.5 * u_l * u_l
    f_r = 0.5 * u_r * u_r

    # 4. Godunov 逻辑 (黎曼求解器核心)
    # 激波速度 s
    s = 0.5 * (u_l + u_r)
    
    # 逻辑分支 1: 激波 (Shock) -> u_l > u_r
    # 如果 s > 0 取左，否则取右
    flux_shock = tl.where(s > 0, f_l, f_r)

    # 逻辑分支 2: 稀疏波 (Rarefaction) -> u_l <= u_r
    # 稀疏波包含声速点(u=0)的处理
    # if u_l > 0 -> f_l
    # elif u_r < 0 -> f_r
    # else -> 0.0
    flux_rarefaction = tl.where(u_l > 0, f_l, 
                                tl.where(u_r < 0, f_r, 0.0))

    # 综合分支
    flux_final = tl.where(u_l > u_r, flux_shock, flux_rarefaction)

    # 5. 存储结果 (Store)
    tl.store(flux_ptr + offsets, flux_final, mask=mask)


# ============================================================================
# Part B: PyTorch Baseline (基准实现) - 用来做对标
# ============================================================================

def _burgers_torch_impl(u_left, u_right):
    """PyTorch 原生实现，用于验证 Triton 结果的正确性"""
    f_l = 0.5 * u_left**2
    f_r = 0.5 * u_right**2
    
    # Shock Case
    s = 0.5 * (u_left + u_right)
    flux_shock = torch.where(s > 0, f_l, f_r)
    
    # Rarefaction Case
    flux_rare = torch.where(u_left > 0, f_l, 
                            torch.where(u_right < 0, f_r, torch.zeros_like(f_l)))
    
    # Combine
    return torch.where(u_left > u_right, flux_shock, flux_rare)


# ============================================================================
# Part C: 对外统一接口 (Plugin Interface)
# ============================================================================

class BurgersSolver:
    @staticmethod
    def godunov_flux(u_left: torch.Tensor, u_right: torch.Tensor, backend='triton'):
        """
        计算 Burgers 方程的数值通量。
        Args:
            u_left, u_right: 输入 Tensor
            backend: 'triton' (加速版) 或 'torch' (验证版)
        """
        # 确保数据连续且在 GPU 上
        if not u_left.is_contiguous(): u_left = u_left.contiguous()
        if not u_right.is_contiguous(): u_right = u_right.contiguous()
        assert u_left.is_cuda, "Input must be on CUDA"

        if backend == 'torch':
            return _burgers_torch_impl(u_left, u_right)
        
        elif backend == 'triton':
            n_elements = u_left.numel()
            # 使用 zeros_like 并显式设置 requires_grad 以支持梯度
            flux = torch.zeros_like(u_left, requires_grad=u_left.requires_grad or u_right.requires_grad)

            # 设定 Block 大小，通常 128/256/512/1024
            BLOCK_SIZE = 1024
            # 计算需要启动多少个 Block
            grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )

            _burgers_godunov_kernel[grid](
                u_left, u_right, flux,
                n_elements,
                BLOCK_SIZE=BLOCK_SIZE
            )
            return flux
        else:
            raise ValueError(f"Unknown backend: {backend}")


# ============================================================================
# Part D: Euler 方程黎曼求解器 (1D 可压缩 Euler)
# ============================================================================

@triton.jit
def _euler_hllc_kernel(
    rho_l_ptr, u_l_ptr, p_l_ptr,      # 左状态
    rho_r_ptr, u_r_ptr, p_r_ptr,      # 右状态
    F1_ptr, F2_ptr, F3_ptr,           # 输出通量 (质量、动量、能量)
    n_elements,                        # 元素总数
    gamma: tl.constexpr,               # 比热比
    BLOCK_SIZE: tl.constexpr           # 块大小
):
    """
    Euler 方程的 HLLC 数值通量计算（完整版黎曼求解器）

    HLLC = Harten-Lax-van Leer-Contact
    - 能正确处理激波、稀疏波、接触间断
    - 满足 Rankine-Hugoniot 条件
    - 工业标准求解器

    守恒变量:
    - U1 = rho (密度)
    - U2 = rho*u (动量)
    - U3 = E (总能量) = p/(gamma-1) + 0.5*rho*u^2

    通量:
    - F1 = rho*u (质量通量)
    - F2 = rho*u^2 + p (动量通量)
    - F3 = u*(E + p) (能量通量)
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    eps = 1e-10

    # 1. 加载左右状态
    rho_l = tl.load(rho_l_ptr + offsets, mask=mask)
    u_l = tl.load(u_l_ptr + offsets, mask=mask)
    p_l = tl.load(p_l_ptr + offsets, mask=mask)

    rho_r = tl.load(rho_r_ptr + offsets, mask=mask)
    u_r = tl.load(u_r_ptr + offsets, mask=mask)
    p_r = tl.load(p_r_ptr + offsets, mask=mask)

    # 防止负值和除零
    rho_l = tl.maximum(rho_l, eps)
    rho_r = tl.maximum(rho_r, eps)
    p_l = tl.maximum(p_l, eps)
    p_r = tl.maximum(p_r, eps)

    # 2. 计算守恒变量和通量
    E_l = p_l / (gamma - 1.0) + 0.5 * rho_l * u_l * u_l
    E_r = p_r / (gamma - 1.0) + 0.5 * rho_r * u_r * u_r

    # 左状态通量
    F1_l = rho_l * u_l
    F2_l = rho_l * u_l * u_l + p_l
    F3_l = u_l * (E_l + p_l)

    # 右状态通量
    F1_r = rho_r * u_r
    F2_r = rho_r * u_r * u_r + p_r
    F3_r = u_r * (E_r + p_r)

    # 3. 估计波速（Davis 估计法）
    c_l = tl.sqrt(gamma * p_l / rho_l)
    c_r = tl.sqrt(gamma * p_r / rho_r)

    S_L = tl.minimum(u_l - c_l, u_r - c_r)
    S_R = tl.maximum(u_l + c_l, u_r + c_r)

    # 4. 计算接触波速度 S_star（HLLC 的核心）
    numerator = p_r - p_l + rho_l * u_l * (S_L - u_l) - rho_r * u_r * (S_R - u_r)
    denominator = rho_l * (S_L - u_l) - rho_r * (S_R - u_r)
    denominator = tl.where(tl.abs(denominator) < eps, eps, denominator)
    S_star = numerator / denominator

    # 5. 计算星区（Star Region）的守恒变量
    # 左星区
    factor_L = rho_l * (S_L - u_l) / (S_L - S_star + eps)
    U1_star_L = factor_L
    U2_star_L = factor_L * S_star
    U3_star_L = factor_L * (E_l / rho_l + (S_star - u_l) * (S_star + p_l / (rho_l * (S_L - u_l) + eps)))

    # 右星区
    factor_R = rho_r * (S_R - u_r) / (S_R - S_star + eps)
    U1_star_R = factor_R
    U2_star_R = factor_R * S_star
    U3_star_R = factor_R * (E_r / rho_r + (S_star - u_r) * (S_star + p_r / (rho_r * (S_R - u_r) + eps)))

    # 6. 计算星区通量
    # F_star_L = F_L + S_L * (U_star_L - U_L)
    F1_star_L = F1_l + S_L * (U1_star_L - rho_l)
    F2_star_L = F2_l + S_L * (U2_star_L - rho_l * u_l)
    F3_star_L = F3_l + S_L * (U3_star_L - E_l)

    # F_star_R = F_R + S_R * (U_star_R - U_R)
    F1_star_R = F1_r + S_R * (U1_star_R - rho_r)
    F2_star_R = F2_r + S_R * (U2_star_R - rho_r * u_r)
    F3_star_R = F3_r + S_R * (U3_star_R - E_r)

    # 7. 根据波速位置选择通量（4 个区域）
    # 区域 1: S_L > 0 --> F_L
    # 区域 2: S_L <= 0 < S_star --> F_star_L
    # 区域 3: S_star <= 0 < S_R --> F_star_R
    # 区域 4: S_R <= 0 --> F_R

    # 先处理右侧（S_R < 0）
    F1_temp = tl.where(S_R < 0, F1_r, F1_star_R)
    F2_temp = tl.where(S_R < 0, F2_r, F2_star_R)
    F3_temp = tl.where(S_R < 0, F3_r, F3_star_R)

    # 再处理中间（S_star < 0）
    F1_temp = tl.where(S_star < 0, F1_temp, F1_star_L)
    F2_temp = tl.where(S_star < 0, F2_temp, F2_star_L)
    F3_temp = tl.where(S_star < 0, F3_temp, F3_star_L)

    # 最后处理左侧（S_L > 0）
    F1_final = tl.where(S_L > 0, F1_l, F1_temp)
    F2_final = tl.where(S_L > 0, F2_l, F2_temp)
    F3_final = tl.where(S_L > 0, F3_l, F3_temp)

    # 8. 存储结果
    tl.store(F1_ptr + offsets, F1_final, mask=mask)
    tl.store(F2_ptr + offsets, F2_final, mask=mask)
    tl.store(F3_ptr + offsets, F3_final, mask=mask)


def _euler_torch_impl(rho_left, u_left, p_left, rho_right, u_right, p_right, gamma=1.4):
    """
    Euler 方程 HLLC 通量的 PyTorch 实现（完整版黎曼求解器）

    HLLC = Harten-Lax-van Leer-Contact
    - 能正确处理激波、稀疏波、接触间断
    - 满足 Rankine-Hugoniot 条件
    - 工业标准求解器

    参数:
        rho_left, u_left, p_left: 左状态 (密度、速度、压力)
        rho_right, u_right, p_right: 右状态
        gamma: 比热比 (默认 1.4 for 空气)

    返回:
        F1, F2, F3: 质量、动量、能量通量
    """
    eps = 1e-10

    # 防止负值和除零
    rho_left = torch.clamp(rho_left, min=eps)
    rho_right = torch.clamp(rho_right, min=eps)
    p_left = torch.clamp(p_left, min=eps)
    p_right = torch.clamp(p_right, min=eps)

    # 1. 计算守恒变量和通量
    E_left = p_left / (gamma - 1.0) + 0.5 * rho_left * u_left**2
    E_right = p_right / (gamma - 1.0) + 0.5 * rho_right * u_right**2

    # 左状态通量
    F1_left = rho_left * u_left
    F2_left = rho_left * u_left**2 + p_left
    F3_left = u_left * (E_left + p_left)

    # 右状态通量
    F1_right = rho_right * u_right
    F2_right = rho_right * u_right**2 + p_right
    F3_right = u_right * (E_right + p_right)

    # 2. 估计波速（Davis 估计法）
    c_left = torch.sqrt(gamma * p_left / rho_left)
    c_right = torch.sqrt(gamma * p_right / rho_right)

    S_L = torch.min(u_left - c_left, u_right - c_right)
    S_R = torch.max(u_left + c_left, u_right + c_right)

    # 3. 计算接触波速度 S_star（HLLC 的核心）
    numerator = p_right - p_left + rho_left * u_left * (S_L - u_left) - rho_right * u_right * (S_R - u_right)
    denominator = rho_left * (S_L - u_left) - rho_right * (S_R - u_right)
    denominator = torch.where(torch.abs(denominator) < eps, eps, denominator)
    S_star = numerator / denominator

    # 4. 计算星区（Star Region）的守恒变量
    # 左星区
    factor_L = rho_left * (S_L - u_left) / (S_L - S_star + eps)
    U1_star_L = factor_L
    U2_star_L = factor_L * S_star
    U3_star_L = factor_L * (E_left / rho_left + (S_star - u_left) * (S_star + p_left / (rho_left * (S_L - u_left) + eps)))

    # 右星区
    factor_R = rho_right * (S_R - u_right) / (S_R - S_star + eps)
    U1_star_R = factor_R
    U2_star_R = factor_R * S_star
    U3_star_R = factor_R * (E_right / rho_right + (S_star - u_right) * (S_star + p_right / (rho_right * (S_R - u_right) + eps)))

    # 5. 计算星区通量
    # F_star_L = F_L + S_L * (U_star_L - U_L)
    F1_star_L = F1_left + S_L * (U1_star_L - rho_left)
    F2_star_L = F2_left + S_L * (U2_star_L - rho_left * u_left)
    F3_star_L = F3_left + S_L * (U3_star_L - E_left)

    # F_star_R = F_R + S_R * (U_star_R - U_R)
    F1_star_R = F1_right + S_R * (U1_star_R - rho_right)
    F2_star_R = F2_right + S_R * (U2_star_R - rho_right * u_right)
    F3_star_R = F3_right + S_R * (U3_star_R - E_right)

    # 6. 根据波速位置选择通量（4 个区域）
    # 区域 1: S_L > 0 --> F_L
    # 区域 2: S_L <= 0 < S_star --> F_star_L
    # 区域 3: S_star <= 0 < S_R --> F_star_R
    # 区域 4: S_R <= 0 --> F_R

    # 先处理右侧（S_R < 0）
    F1_temp = torch.where(S_R < 0, F1_right, F1_star_R)
    F2_temp = torch.where(S_R < 0, F2_right, F2_star_R)
    F3_temp = torch.where(S_R < 0, F3_right, F3_star_R)

    # 再处理中间（S_star < 0）
    F1_temp = torch.where(S_star < 0, F1_temp, F1_star_L)
    F2_temp = torch.where(S_star < 0, F2_temp, F2_star_L)
    F3_temp = torch.where(S_star < 0, F3_temp, F3_star_L)

    # 最后处理左侧（S_L > 0）
    F1 = torch.where(S_L > 0, F1_left, F1_temp)
    F2 = torch.where(S_L > 0, F2_left, F2_temp)
    F3 = torch.where(S_L > 0, F3_left, F3_temp)

    return F1, F2, F3


class EulerSolver:
    """
    1D 可压缩 Euler 方程的黎曼求解器

    支持 torch 和 triton 两种后端
    使用 HLLC 近似黎曼求解器（工业标准）
    """

    @staticmethod
    def godunov_flux(
        rho_left: torch.Tensor,
        u_left: torch.Tensor,
        p_left: torch.Tensor,
        rho_right: torch.Tensor,
        u_right: torch.Tensor,
        p_right: torch.Tensor,
        gamma: float = 1.4,
        backend: str = 'triton'
    ):
        """
        计算 Euler 方程的 HLLC 数值通量（完整版 Godunov 格式）

        参数:
            rho_left, u_left, p_left: 左状态 (密度、速度、压力)
            rho_right, u_right, p_right: 右状态
            gamma: 比热比 (默认 1.4)
            backend: 'triton' (GPU加速) 或 'torch' (验证)

        返回:
            F1, F2, F3: 质量、动量、能量通量

        说明:
            HLLC 求解器能够:
            - 正确处理激波、稀疏波、接触间断
            - 满足 Rankine-Hugoniot 跳跃条件
            - 保证数值稳定性和物理一致性
        """
        # 确保数据连续且在 GPU 上
        if not rho_left.is_contiguous(): rho_left = rho_left.contiguous()
        if not u_left.is_contiguous(): u_left = u_left.contiguous()
        if not p_left.is_contiguous(): p_left = p_left.contiguous()
        if not rho_right.is_contiguous(): rho_right = rho_right.contiguous()
        if not u_right.is_contiguous(): u_right = u_right.contiguous()
        if not p_right.is_contiguous(): p_right = p_right.contiguous()

        assert rho_left.is_cuda, "Input must be on CUDA"

        if backend == 'torch':
            return _euler_torch_impl(rho_left, u_left, p_left,
                                    rho_right, u_right, p_right, gamma)

        elif backend == 'triton':
            n_elements = rho_left.numel()
            # 使用 zeros_like 并显式设置 requires_grad 以支持梯度
            requires_grad = any([
                rho_left.requires_grad, u_left.requires_grad, p_left.requires_grad,
                rho_right.requires_grad, u_right.requires_grad, p_right.requires_grad
            ])
            F1 = torch.zeros_like(rho_left, requires_grad=requires_grad)
            F2 = torch.zeros_like(rho_left, requires_grad=requires_grad)
            F3 = torch.zeros_like(rho_left, requires_grad=requires_grad)

            BLOCK_SIZE = 1024
            grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )

            _euler_hllc_kernel[grid](
                rho_left, u_left, p_left,
                rho_right, u_right, p_right,
                F1, F2, F3,
                n_elements,
                gamma=gamma,
                BLOCK_SIZE=BLOCK_SIZE
            )
            return F1, F2, F3

        else:
            raise ValueError(f"Unknown backend: {backend}")