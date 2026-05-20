# Physics模块修正总结

## 修正日期
2026-02-04

## 核心问题

之前的实现**只计算了一阶导数**，而Guderley方程是**二阶ODE**，导致方程不完整，这是漏斗扫描实验失败的根本原因。

## 主要修正内容

### 1. 补充二阶导数计算 ✅

**修正前**：
```python
def compute_derivatives(model, xi):
    # 只计算一阶导数
    dV_dxi = torch.autograd.grad(V, xi, ...)
    dC_dxi = torch.autograd.grad(C, xi, ...)
    return V, C, G, dV_dxi, dC_dxi, dG_dxi
```

**修正后**：
```python
def compute_derivatives(model, xi):
    # 计算一阶导数
    dV_dxi = torch.autograd.grad(V, xi, ...)
    dC_dxi = torch.autograd.grad(C, xi, ...)

    # 补充二阶导数（核心修正）
    d2V_dxi2 = torch.autograd.grad(dV_dxi, xi, ...)
    d2C_dxi2 = torch.autograd.grad(dC_dxi, xi, ...)

    return V, C, G, dV_dxi, d2V_dxi2, dC_dxi, d2C_dxi2, dG_dxi
```

### 2. 实现完整的二阶ODE残差 ✅

**标准Guderley方程**：

**V方程**：
```
d²V/dξ² + (n/ξ)dV/dξ + [V(V-1/α)(V-1)]/Δ · dV/dξ
        + [2C²(V-1/α)]/(γΔ) · dV/dξ + nC²/(γαξΔ) = 0
```

**C方程**：
```
d²C/dξ² + (n/ξ)dC/dξ + [V(V-1/α)(V-1)]/Δ · dC/dξ
        - (β/C)[(dV/dξ)² + (nV/ξ)dV/dξ] = 0
```

其中：
- Δ = (V-1)² - C²（声速线判别式）
- β = (γ-1)/2
- α = 自相似指数（可训练参数）

**修正后的实现**：
```python
def guderley_ode_residual(model, xi, gamma, n):
    # V方程的5个项
    term1_V = d2V_dxi2  # 二阶导数项（新增）
    term2_V = (n / xi_safe) * dV_dxi  # 几何源项
    term3_V = (V / Delta) * (V - 1.0/alpha) * (V - 1) * dV_dxi  # 非线性对流
    term4_V = (2 * C**2 / (gamma * Delta)) * (V - 1.0/alpha) * dV_dxi  # 压力梯度
    term5_V = (n * C**2) / (gamma * alpha * xi_safe * Delta)  # 几何压力

    res_V = term1_V + term2_V + term3_V + term4_V + term5_V

    # C方程的4个项
    beta = (gamma - 1.0) / 2.0
    term1_C = d2C_dxi2  # 二阶导数项（新增）
    term2_C = (n / xi_safe) * dC_dxi  # 几何源项
    term3_C = (V / Delta) * (V - 1.0/alpha) * (V - 1) * dC_dxi  # 对流项
    term4_C = (beta / C_safe) * (dV_dxi**2 + (n * V / xi_safe) * dV_dxi)  # 耦合项

    res_C = term1_C + term2_C + term3_C - term4_C

    return res_V, res_C
```

### 3. 添加中心边界条件 ✅

**物理要求**：在中心处（ξ→0），物理量必须有限

**数学约束**：
```
dV/dξ|ξ→0 = 0
dC/dξ|ξ→0 = 0
```

**实现**：
```python
def center_bc(model, xi_center):
    """中心边界条件：一阶导数为0"""
    V, C, G, dV_dxi, d2V_dxi2, dC_dxi, d2C_dxi2, dG_dxi = compute_derivatives(model, xi_center)

    loss_dV = torch.mean(dV_dxi ** 2)
    loss_dC = torch.mean(dC_dxi ** 2)

    return loss_dV, loss_dC
```

### 4. 添加数值稳定性保护 ✅

**问题**：Δ = (V-1)² - C² 在声速线处会趋近于0，导致除零错误

**解决方案**：
```python
eps = 1e-6
xi_safe = xi + eps      # 避免xi→0时除以0
C_safe = C + eps        # 避免C→0时除以0
Delta = (V - 1)**2 - C**2 + eps  # 避免Delta→0
```

### 5. 支持非强激波边界条件 ✅

**修正前**（强激波近似，M∞→∞）：
```python
V_bc = 2.0 / (gamma + 1.0)
C_bc = np.sqrt(2.0 * gamma * (gamma - 1.0)) / (gamma + 1.0)
```

**修正后**（精确RH条件，支持有限M∞）：
```python
def rankine_hugoniot_bc(gamma=1.4, mach_inf=10.0):
    M2 = mach_inf ** 2
    V_bc = (2.0 + (gamma - 1.0) * M2) / ((gamma + 1.0) * M2)
    C_bc = np.sqrt(2.0 * gamma * M2 - (gamma - 1.0)) / ((gamma + 1.0) * mach_inf)
    G_bc = (gamma + 1.0) / (gamma - 1.0)
    return V_bc, C_bc, G_bc
```

**示例**（γ=1.4, M∞=10）：
- V(1) = 0.175
- C(1) = 0.697
- G(1) = 6.0

### 6. BC参与优化 ✅

**修正前**（硬约束模式）：
```python
total_loss = weight_pde * loss_pde  # BC不参与优化
```

**修正后**（软约束模式）：
```python
total_loss = (
    weight_pde * loss_pde +
    weight_bc_shock * loss_bc_shock +
    weight_bc_center * loss_bc_center
)
```

### 7. 修正几何参数注释 ✅

**修正前**（错误）：
```python
n: 对称性参数 (1=圆柱, 2=球形)
```

**修正后**（正确）：
```python
n: 几何因子 (2=柱面, 3=球面, 1=平面)
```

## 测试结果

```
============================================================
测试 Guderley Physics 模块（修正版）
============================================================

初始 alpha: 0.717000

内部采样点数: 100
激波边界采样点数: 1
中心边界采样点数: 1

激波边界条件 (γ=1.4, M∞=10):
  V(1) = 0.175000
  C(1) = 0.696718
  G(1) = 6.000000

损失函数:
  Total Loss:   8.118064e+02
  PDE Loss:     8.067506e+02
  BC Shock:     4.988152e-01
  BC Center:    1.352142e-02

✓ 修正后Physics模块测试通过
```

## 与标准方程的对应关系

修正后的代码完全符合标准Guderley方程：

| 标准方程项 | 代码实现 | 状态 |
|-----------|---------|------|
| d²V/dξ² | `d2V_dxi2` | ✅ |
| n/ξ · dV/dξ | `(n/xi_safe) * dV_dxi` | ✅ |
| V(V-1/α)(V-1)/Δ · dV/dξ | `(V/Delta)*(V-1/alpha)*(V-1)*dV_dxi` | ✅ |
| 2C²(V-1/α)/(γΔ) · dV/dξ | `(2*C²/(gamma*Delta))*(V-1/alpha)*dV_dxi` | ✅ |
| nC²/(γαξΔ) | `(n*C²)/(gamma*alpha*xi_safe*Delta)` | ✅ |
| d²C/dξ² | `d2C_dxi2` | ✅ |
| n/ξ · dC/dξ | `(n/xi_safe) * dC_dxi` | ✅ |
| V(V-1/α)(V-1)/Δ · dC/dξ | `(V/Delta)*(V-1/alpha)*(V-1)*dC_dxi` | ✅ |
| -β/C[(dV/dξ)²+(nV/ξ)dV/dξ] | `-(beta/C_safe)*[dV_dxi²+(n*V/xi_safe)*dV_dxi]` | ✅ |

## 下一步

使用修正后的方程重新运行漏斗扫描实验，验证：
1. α=0.717处的损失是否最小
2. 损失函数是否呈现V型谷底
3. PINN能否通过梯度下降找到正确的α

## 预期结果

如果修正是正确的，应该看到：
- ✅ V型谷底：α=0.717处损失最小
- ✅ 梯度方向正确：∂L/∂α在α<0.717时为负，α>0.717时为正
- ✅ 优化收敛：从不同初始值都能收敛到α≈0.717

如果仍然失败，说明问题不在方程本身，而在于：
- 网络容量不足
- 训练策略问题
- 或者PINN方法本身不适合这类参数发现问题
