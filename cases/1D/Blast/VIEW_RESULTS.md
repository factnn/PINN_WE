# 快速查看结果 - 终端命令

## 方法1：训练时自动保存模型（推荐）

在训练代码中添加保存模型的代码，然后运行可视化：

```bash
cd /share/project/zpy/PINN_WE/cases/1D/Blast
source /share/project/zhaohuxing/anaconda3/bin/activate PINN_WE
export PYTHONPATH=/share/project/zpy/PINN_WE/PINNsrc:$PYTHONPATH

# 运行训练（会自动保存图片）
python 1.py
```

## 方法2：快速查看脚本

```bash
cd /share/project/zpy/PINN_WE/cases/1D/Blast
source /share/project/zhaohuxing/anaconda3/bin/activate PINN_WE
export PYTHONPATH=/share/project/zpy/PINN_WE/PINNsrc:$PYTHONPATH

# 快速查看（如果模型在内存中）
python quick_view.py
```

## 方法3：一行命令查看（如果刚训练完）

在训练完成后，模型还在内存中，可以直接运行：

```bash
cd /share/project/zpy/PINN_WE/cases/1D/Blast && source /share/project/zhaohuxing/anaconda3/bin/activate PINN_WE && export PYTHONPATH=/share/project/zpy/PINN_WE/PINNsrc:$PYTHONPATH && python -c "
import sys; sys.path.insert(0, '../../../PINNsrc')
from PINNs import *
import torch, numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
x = np.linspace(0, 1, 200)
t = np.full_like(x, 0.2)
x_test = torch.tensor(np.hstack((t[:, None], x[:, None])), dtype=torch.float64).to(cuda)
model = PINNs_WE_Euler_1D(Nl=6, Nn=60).to(cuda).double()
model.eval()
with torch.no_grad():
    u = model(x_test)
    rho, p, u_vel = u[:, 0].cpu().numpy(), u[:, 1].cpu().numpy(), u[:, 2].cpu().numpy()
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, data, label, color in zip(axes, [rho, p, u_vel], ['Density', 'Pressure', 'Velocity'], ['b', 'r', 'g']):
    ax.plot(x, data, color, linewidth=2)
    ax.set_xlabel('x')
    ax.set_ylabel(label)
    ax.set_title(f'{label} at t=0.2')
    ax.grid(True)
plt.tight_layout()
plt.savefig('quick_results.png', dpi=300)
print('结果已保存到 quick_results.png')
"
```

## 方法4：修改1.py添加模型保存

在1.py的训练完成后添加：

```python
# 保存模型
torch.save(model.state_dict(), 'model.pth')
print('模型已保存')

# 然后可以随时加载查看
# model.load_state_dict(torch.load('model.pth'))
```

然后运行可视化：

```bash
python quick_view.py
```

