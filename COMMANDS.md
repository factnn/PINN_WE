# PINN_WE 使用命令参考

## 1. 激活环境

```bash
source /share/project/zhaohuxing/anaconda3/bin/activate PINN_WE
```

## 2. 安装依赖包（使用华为源）

```bash
source /share/project/zhaohuxing/anaconda3/bin/activate PINN_WE
pip install -i https://mirrors.huaweicloud.com/repository/pypi/simple torch numpy scipy matplotlib smt
```

## 3. 验证环境

### 3.1 验证包导入
```bash
source /share/project/zhaohuxing/anaconda3/bin/activate PINN_WE
python -c "import torch; import numpy as np; import scipy; import matplotlib; from smt.sampling_methods import LHS; print('所有包导入成功！')"
```

### 3.2 验证CUDA
```bash
source /share/project/zhaohuxing/anaconda3/bin/activate PINN_WE
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'GPU数量: {torch.cuda.device_count()}')"
```

### 3.3 验证PINNs模块导入
```bash
cd /share/project/zpy/PINN_WE
source /share/project/zhaohuxing/anaconda3/bin/activate PINN_WE
python -c "import sys; sys.path.insert(0, 'PINNsrc'); from PINNs import PINNs_WE_Euler_1D; print('导入成功！')"
```

## 4. 运行算例

### 4.1 运行一维欧拉方程算例（Sod激波管问题）

**方法1：使用原始脚本（推荐，使用PINNsrc的类）**
```bash
cd /share/project/zpy/PINN_WE/cases/1D/Blast
source /share/project/zhaohuxing/anaconda3/bin/activate PINN_WE
export PYTHONPATH=/share/project/zpy/PINN_WE/PINNsrc:$PYTHONPATH
python 1.py
```

**方法2：使用快速测试脚本（减少epoch，用于快速验证）**
```bash
cd /share/project/zpy/PINN_WE/cases/1D/Blast
source /share/project/zhaohuxing/anaconda3/bin/activate PINN_WE
python test_simple.py
```

**方法3：在Python中设置路径后运行**
```bash
cd /share/project/zpy/PINN_WE
source /share/project/zhaohuxing/anaconda3/bin/activate PINN_WE
python -c "
import sys
sys.path.insert(0, 'PINNsrc')
exec(open('cases/1D/Blast/1.py').read())
"
```

### 4.2 运行其他算例

**Lax问题（自包含，不需要设置路径）：**
```bash
cd /share/project/zpy/PINN_WE/cases/1D/Lax
source /share/project/zhaohuxing/anaconda3/bin/activate PINN_WE
python 1.py
```

**Shu-Osher问题（自包含，不需要设置路径）：**
```bash
cd /share/project/zpy/PINN_WE/cases/1D/Shu-Osher
source /share/project/zhaohuxing/anaconda3/bin/activate PINN_WE
python shuosh.py
```

## 5. 常用操作

### 5.1 查看GPU使用情况
```bash
nvidia-smi
```

### 5.2 后台运行训练（使用nohup）
```bash
cd /share/project/zpy/PINN_WE/cases/1D/Blast
source /share/project/zhaohuxing/anaconda3/bin/activate PINN_WE
nohup python test_simple.py > train.log 2>&1 &
```

### 5.3 查看训练日志
```bash
tail -f train.log
```

### 5.4 停止后台训练
```bash
# 查找进程
ps aux | grep python
# 杀死进程
kill <PID>
```

## 6. 环境信息

- **Python版本**: 3.10.12
- **PyTorch版本**: 2.7.1+cu126
- **CUDA版本**: 12.6
- **GPU**: 8x NVIDIA A100-SXM4-40GB
- **主要依赖**: torch, numpy, scipy, matplotlib, smt

## 7. 项目结构

```
PINN_WE/
├── PINNsrc/          # 核心代码
│   ├── PINNs.py      # PINN模型定义
│   ├── IC_1D.py      # 一维初始条件
│   ├── BC_1D.py      # 一维边界条件
│   └── utility.py    # 工具函数
└── cases/            # 算例
    ├── 1D/           # 一维算例
    │   ├── Blast/    # Sod激波管
    │   ├── Lax/      # Lax问题
    │   └── ...
    └── 2D/           # 二维算例
```

## 8. 重要说明

### 8.1 GPU使用
- **当前代码使用单GPU**（默认cuda:0）
- 代码中没有使用多GPU并行（DataParallel/DistributedDataParallel）
- 虽然你有8块A100，但当前代码只会用第一块

### 8.2 训练和求解
- **一步搞定！** PINN的训练过程就是求解过程
- 训练完成后，网络本身就是方程的解
- 不需要额外的求解步骤，直接用 `model(x_test)` 预测

### 8.3 可视化结果
- **原始脚本（1.py）**: 没有绘图代码，只训练
- **带可视化的脚本**: 使用 `1_with_plot.py`（我创建的）
- **Jupyter notebooks**: 有完整的可视化代码

### 运行带可视化的脚本：
```bash
cd /share/project/zpy/PINN_WE/cases/1D/Blast
source /share/project/zhaohuxing/anaconda3/bin/activate PINN_WE
export PYTHONPATH=/share/project/zpy/PINN_WE/PINNsrc:$PYTHONPATH
python 1_with_plot.py
```

## 9. 注意事项

1. **路径问题**: 原始脚本使用 `from PINNs import *`，需要确保PINNsrc在Python路径中
2. **设备设置**: 代码中硬编码使用 `cuda` 设备，确保有GPU可用
3. **训练时间**: 完整训练可能需要很长时间（原代码设置100000个epoch），建议先用少量epoch测试
4. **可视化**: 原始脚本不包含绘图，需要自己添加或使用notebook

