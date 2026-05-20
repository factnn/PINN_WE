# PINN_WE 训练和可视化指南

## 回答你的问题

### 1. 是否使用8卡GPU？

**答案：目前代码只使用单GPU**

- 代码中使用 `.to(cuda)` 将模型和数据放到GPU上
- **没有使用** `DataParallel` 或 `DistributedDataParallel` 进行多GPU并行
- 默认使用 `cuda:0`（第一块GPU）

**如果想使用多GPU，需要修改代码：**
```python
# 当前代码（单GPU）
model = PINNs_WE_Euler_1D(Nl=6,Nn=60).to(cuda)

# 改为多GPU（需要修改）
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(PINNs_WE_Euler_1D(Nl=6,Nn=60)).to(cuda)
```

### 2. 训练和求解的关系？

**答案：一步搞定！训练完成即求解完成**

PINN（Physics-Informed Neural Network）的工作方式：
1. **训练过程** = **求解过程**
   - 训练时，网络学习满足PDE方程的解
   - 训练完成后，网络本身就是方程的解
   - **不需要额外的求解步骤**

2. **流程**：
   ```
   初始化网络 → 训练（最小化PDE残差） → 训练完成 → 网络就是解
   ```

3. **使用训练好的网络**：
   ```python
   # 训练完成后，直接预测任意点的解
   x_test = torch.tensor([[t, x]], dtype=dtype).to(cuda)
   u_pred = model(x_test)  # 直接得到解 [rho, p, u]
   ```

### 3. 最后的结果有图吗？

**答案：有！但需要自己添加绘图代码**

#### 当前情况：
- **1.py** 等脚本：**没有绘图代码**，只训练
- **Jupyter notebooks**：有完整的绘图代码

#### 如何可视化结果：

**方法1：使用Jupyter notebook（推荐）**
```bash
cd /share/project/zpy/PINN_WE/cases/1D/Blast
source /share/project/zhaohuxing/anaconda3/bin/activate PINN_WE
export PYTHONPATH=/share/project/zpy/PINN_WE/PINNsrc:$PYTHONPATH
jupyter notebook SodNewCore.ipynb
```

**方法2：在Python脚本中添加绘图代码**

在训练完成后添加：
```python
import matplotlib.pyplot as plt
import numpy as np

# 创建测试网格
x = np.linspace(Xs, Xe, 200)
t = np.full_like(x, Te)  # 在最终时刻
x_test = np.hstack((t[:, None], x[:, None]))
x_test = torch.tensor(x_test, dtype=dtype).to(cuda)

# 预测
with torch.no_grad():
    u_pred = model(x_test)
    rho_pred = u_pred[:, 0].cpu().numpy()
    p_pred = u_pred[:, 1].cpu().numpy()
    u_pred = u_pred[:, 2].cpu().numpy()

# 绘图
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(x, rho_pred, 'b-', linewidth=2, label='PINN')
plt.xlabel('x')
plt.ylabel('Density')
plt.title('Density at t={}'.format(Te))
plt.grid(True)
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(x, p_pred, 'r-', linewidth=2, label='PINN')
plt.xlabel('x')
plt.ylabel('Pressure')
plt.title('Pressure at t={}'.format(Te))
plt.grid(True)
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(x, u_pred, 'g-', linewidth=2, label='PINN')
plt.xlabel('x')
plt.ylabel('Velocity')
plt.title('Velocity at t={}'.format(Te))
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('results.png', dpi=300)
plt.show()
```

## 完整运行流程

### 步骤1：训练（使用原始代码）

```bash
cd /share/project/zpy/PINN_WE/cases/1D/Blast
source /share/project/zhaohuxing/anaconda3/bin/activate PINN_WE
export PYTHONPATH=/share/project/zpy/PINN_WE/PINNsrc:$PYTHONPATH
python 1.py
```

**训练过程：**
- Adam优化器：100000个epoch（或直到loss < 0.05）
- LBFGS优化器：5000个epoch
- 训练时间：可能需要几小时到几天（取决于问题复杂度）

### 步骤2：可视化结果

**选项A：使用Jupyter notebook**
```bash
jupyter notebook SodNewCore.ipynb
# 在notebook中运行训练和可视化代码
```

**选项B：添加绘图代码到脚本**
在 `1.py` 末尾添加上述绘图代码

**选项C：保存模型后单独可视化**
```python
# 保存模型
torch.save(model.state_dict(), 'model.pth')

# 加载模型并可视化
model = PINNs_WE_Euler_1D(Nl=6,Nn=60).to(cuda).double()
model.load_state_dict(torch.load('model.pth'))
# 然后使用上面的绘图代码
```

## 现有结果文件

在 `cases/1D/Blast/` 目录下有很多 `.eps` 图片文件：
- `Sod_PINNS_WE.eps` - PINN结果
- `Sod_PINNS_WE_rh.eps` - 带RH条件的结果
- `Sod_PINNS_WE_weight.eps` - 加权损失的结果
- 等等...

这些是之前运行生成的结果图。

## 推荐工作流程

1. **快速测试**：使用 `test_simple.py`（减少epoch）
2. **完整训练**：使用 `1.py`（完整epoch）
3. **可视化**：使用Jupyter notebook或添加绘图代码
4. **保存结果**：保存模型和图片

## 多GPU使用（可选）

如果需要使用多GPU加速，可以修改代码：

```python
import torch.nn as nn

# 在模型创建后
if torch.cuda.device_count() > 1:
    print(f"使用 {torch.cuda.device_count()} 个GPU")
    model = nn.DataParallel(model)
```

但注意：PINN训练通常受内存限制，多GPU可能不会显著加速（除非数据并行化）。

