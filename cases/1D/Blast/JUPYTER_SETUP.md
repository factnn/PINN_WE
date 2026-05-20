# Jupyter运行设置

## 快速启动（推荐）

### 方法1：在Jupyter第一个cell添加路径设置

打开 `SodNewCore.ipynb`，在**第一个cell的最前面**添加：

```python
# ========== 路径设置（必须放在最前面）==========
import sys
import os

# 自动检测并添加PINNsrc路径
current_dir = os.path.dirname(os.path.abspath(''))
# 从 cases/1D/Blast 向上三级到 PINN_WE，然后进入 PINNsrc
pinn_src_path = os.path.join(current_dir, '..', '..', '..', 'PINNsrc')
pinn_src_path = os.path.abspath(pinn_src_path)
sys.path.insert(0, pinn_src_path)

print(f'✓ 已添加路径: {pinn_src_path}')
print(f'✓ PINNs.py存在: {os.path.exists(os.path.join(pinn_src_path, "PINNs.py"))}')

# ========== 原有代码从这里开始 ==========
from PINNs import *
```

### 方法2：启动Jupyter时设置环境变量

```bash
cd /share/project/zpy/PINN_WE/cases/1D/Blast
source /share/project/zhaohuxing/anaconda3/bin/activate PINN_WE
export PYTHONPATH=/share/project/zpy/PINN_WE/PINNsrc:$PYTHONPATH
jupyter notebook
```

然后在notebook中直接使用 `from PINNs import *`（不需要修改notebook）

### 方法3：在Jupyter中运行Python脚本

在Jupyter中创建一个cell：

```python
%run 1_with_plot.py
```

这会运行脚本并显示所有输出（包括图片）。

## 测试导入

在Jupyter中运行这个cell测试是否设置成功：

```python
import sys
import os

# 设置路径
pinn_src_path = os.path.abspath('../../../PINNsrc')
if pinn_src_path not in sys.path:
    sys.path.insert(0, pinn_src_path)

# 测试导入
try:
    from PINNs import PINNs_WE_Euler_1D
    import torch
    print("✅ 导入成功！")
    print(f"✅ CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✅ GPU数量: {torch.cuda.device_count()}")
except Exception as e:
    print(f"❌ 导入失败: {e}")
```

## 推荐工作流程

1. **打开Jupyter**：
   ```bash
   cd /share/project/zpy/PINN_WE/cases/1D/Blast
   source /share/project/zhaohuxing/anaconda3/bin/activate PINN_WE
   export PYTHONPATH=/share/project/zpy/PINN_WE/PINNsrc:$PYTHONPATH
   jupyter notebook
   ```

2. **在notebook中**：
   - 第一个cell：测试导入（使用上面的测试代码）
   - 后续cells：运行原有的训练和可视化代码

3. **或者直接运行脚本**：
   ```python
   %run 1_with_plot.py
   ```

