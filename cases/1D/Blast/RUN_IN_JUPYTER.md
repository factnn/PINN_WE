# 在Jupyter中运行PINN_WE

## 方法1：在notebook第一个cell添加路径设置

在任何notebook的第一个cell添加：

```python
import sys
import os

# 设置PINNsrc路径（根据你的notebook位置调整）
# 如果notebook在 cases/1D/Blast/ 目录下
pinn_src_path = os.path.abspath('../../../PINNsrc')
sys.path.insert(0, pinn_src_path)

print(f'已添加路径: {pinn_src_path}')
print(f'PINNs.py存在: {os.path.exists(os.path.join(pinn_src_path, "PINNs.py"))}')

# 然后就可以正常导入了
from PINNs import *
```

## 方法2：启动Jupyter时设置环境变量

```bash
cd /share/project/zpy/PINN_WE
source /share/project/zhaohuxing/anaconda3/bin/activate PINN_WE
export PYTHONPATH=/share/project/zpy/PINN_WE/PINNsrc:$PYTHONPATH
jupyter notebook
```

然后在notebook中直接使用：
```python
from PINNs import *
```

## 方法3：修改现有notebook

在 `SodNewCore.ipynb` 的第一个cell最前面添加路径设置代码。

## 推荐：使用我创建的notebook模板

我已经创建了一个可以直接运行的notebook版本，包含：
- 自动路径设置
- 完整的训练代码
- 可视化代码
- GPU信息显示

文件：`SodNewCore_ready.ipynb`（如果存在）

## 快速测试

在Jupyter中运行这个cell测试：

```python
import sys
import os
sys.path.insert(0, os.path.abspath('../../../PINNsrc'))

from PINNs import PINNs_WE_Euler_1D
import torch

print("导入成功！")
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
```

