# PINN中实验数据融合指南

## 基本原理

PINN融合实验数据的核心思想：**在损失函数中添加数据拟合项**

总损失 = PDE损失 + 初始条件损失 + 边界条件损失 + **数据拟合损失**

## 数据融合的优势

1. **补充信息**：实验数据提供真实物理信息
2. **约束解**：数据点强制网络在特定位置匹配观测值
3. **提高精度**：特别是在数据点附近
4. **处理逆问题**：可以从数据反推参数

## 实现方法

### 方法1：点数据融合（最简单）

假设你有实验数据点：在特定位置 (t_i, x_i) 观测到的值 (ρ_i, u_i, p_i)

```python
def loss_data(self, x_data, rho_data, u_data, p_data):
    """
    数据拟合损失
    x_data: 实验数据点的坐标 [t, x]
    rho_data, u_data, p_data: 实验观测值
    """
    y_pred = self.net(x_data)
    rho_pred, p_pred, u_pred = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]
    
    loss_data = ((rho_pred - rho_data)**2).mean() + \
                ((u_pred - u_data)**2).mean() + \
                ((p_pred - p_data)**2).mean()
    
    return loss_data
```

### 方法2：部分物理量数据

如果只有部分物理量的数据（比如只有压力或只有速度）：

```python
def loss_data_partial(self, x_data, p_data):
    """只有压力数据"""
    y_pred = self.net(x_data)
    p_pred = y_pred[:, 1]
    loss_data = ((p_pred - p_data)**2).mean()
    return loss_data
```

### 方法3：带不确定性的数据融合

如果数据有测量误差，可以加权：

```python
def loss_data_weighted(self, x_data, rho_data, u_data, p_data, 
                       sigma_rho, sigma_u, sigma_p):
    """
    带不确定性的数据融合
    sigma: 测量误差的标准差
    """
    y_pred = self.net(x_data)
    rho_pred, p_pred, u_pred = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]
    
    loss_data = ((rho_pred - rho_data)**2 / (sigma_rho**2 + 1e-6)).mean() + \
                ((u_pred - u_data)**2 / (sigma_u**2 + 1e-6)).mean() + \
                ((p_pred - p_data)**2 / (sigma_p**2 + 1e-6)).mean()
    
    return loss_data
```

### 方法4：时间序列数据

如果有时间序列数据（多个时刻的观测）：

```python
def loss_data_timeseries(self, x_data_list, data_list):
    """
    x_data_list: 不同时刻的数据点坐标列表
    data_list: 对应时刻的观测值列表
    """
    total_loss = 0
    for x_data, (rho_data, u_data, p_data) in zip(x_data_list, data_list):
        y_pred = self.net(x_data)
        rho_pred, p_pred, u_pred = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]
        
        total_loss += ((rho_pred - rho_data)**2).mean() + \
                      ((u_pred - u_data)**2).mean() + \
                      ((p_pred - p_data)**2).mean()
    
    return total_loss / len(x_data_list)
```

## 完整实现示例

### 修改PINNs.py添加数据损失

在 `PINNs_WE_Euler_1D` 类中添加：

```python
def loss_data(self, x_data, rho_data, u_data, p_data):
    """实验数据拟合损失"""
    y_pred = self.net(x_data)
    rho_pred, p_pred, u_pred = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]
    
    loss_data = ((rho_pred - rho_data)**2).mean() + \
                ((u_pred - u_data)**2).mean() + \
                ((p_pred - p_data)**2).mean()
    
    return loss_data
```

### 修改训练函数

```python
def train(epoch):
    def closure():
        optimizer.zero_grad()
        
        # 原有损失
        loss_pde = model.loss_pde(x_int)
        loss_ic = model.loss_ic(x_ic, rho_ic, u_ic, p_ic)
        loss_rh = model.loss_rh(xrh, xrhL)
        loss_con = model.loss_con(x_en, x_ic, crhoL, cuL, cpL, crhoR, cuR, cpR, Te-Ts)
        
        # 新增：数据拟合损失
        loss_data = model.loss_data(x_exp, rho_exp, u_exp, p_exp)
        
        # 总损失（数据损失权重可以调整）
        loss = loss_pde + 10*loss_ic + 10*loss_rh + 10*loss_con + lambda_data*loss_data
        
        print(f'epoch {epoch} loss_pde:{loss_pde:.8f}, loss_data:{loss_data:.8f}, ...')
        loss.backward()
        return loss
    return optimizer.step(closure)
```

## 数据格式要求

### 实验数据格式

```python
# 假设你的实验数据格式
# 数据点：在时间t和位置x处的观测值

# 方式1：numpy数组
x_exp = np.array([
    [t1, x1],
    [t2, x2],
    [t3, x3],
    ...
])  # shape: (N_data, 2)

rho_exp = np.array([rho1, rho2, rho3, ...])  # shape: (N_data,)
u_exp = np.array([u1, u2, u3, ...])
p_exp = np.array([p1, p2, p3, ...])

# 转换为tensor
x_exp = torch.tensor(x_exp, dtype=dtype).to(cuda)
rho_exp = torch.tensor(rho_exp, dtype=dtype).to(cuda)
u_exp = torch.tensor(u_exp, dtype=dtype).to(cuda)
p_exp = torch.tensor(p_exp, dtype=dtype).to(cuda)
```

### 从文件读取数据

```python
# 假设数据文件格式：t, x, rho, u, p
data = np.loadtxt('experimental_data.txt')  # shape: (N, 5)

x_exp = data[:, [0, 1]]  # t, x
rho_exp = data[:, 2]
u_exp = data[:, 3]
p_exp = data[:, 4]
```

## 权重选择策略

数据损失的权重 `lambda_data` 很重要：

- **太小**：数据约束不够，网络可能忽略数据
- **太大**：可能过度拟合数据，违反PDE

### 自适应权重

```python
# 根据损失大小动态调整
lambda_data = loss_pde.item() / (loss_data.item() + 1e-6)
# 或者固定但可调
lambda_data = 100.0  # 根据实际情况调整
```

### 课程学习策略

```python
# 训练初期：更重视PDE
# 训练后期：更重视数据拟合
if epoch < 1000:
    lambda_data = 10.0
elif epoch < 10000:
    lambda_data = 50.0
else:
    lambda_data = 100.0
```

## 实际应用场景

### 场景1：稀疏观测数据
- 只有少数几个点的数据
- 用数据约束关键位置
- 其他区域由PDE约束

### 场景2：边界数据
- 有边界处的测量数据
- 替代或补充边界条件

### 场景3：时间序列数据
- 多个时刻的观测
- 可以追踪激波传播

### 场景4：逆问题
- 从数据反推初始条件或参数
- 数据损失 + PDE损失联合优化

## 注意事项

1. **数据质量**：确保数据准确，噪声大的数据需要加权
2. **数据分布**：尽量覆盖关键区域（如激波附近）
3. **单位一致性**：确保实验数据和PDE使用相同单位
4. **数据预处理**：可能需要归一化
5. **权重平衡**：PDE损失和数据损失需要平衡

## 优势总结

相比纯PDE求解：
- ✅ 利用真实物理数据
- ✅ 在数据点附近精度更高
- ✅ 可以处理逆问题
- ✅ 可以处理不完整的边界条件

相比纯数据驱动：
- ✅ 仍然满足物理方程
- ✅ 可以外推到数据未覆盖区域
- ✅ 物理一致性更好

