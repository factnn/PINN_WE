# PINN_WE 环境安装命令

## 完整安装步骤

### 1. 删除旧环境（如果存在）
```bash
conda deactivate
conda env remove -n PINN_WE -y
```

### 2. 创建新的Python 3.10环境
```bash
conda create -n PINN_WE python=3.10 -y
```

### 3. 激活环境
```bash
source /share/project/zhaohuxing/anaconda3/bin/activate PINN_WE
```

### 4. 安装所有必需的包（使用华为源）
```bash
pip install torch numpy scipy matplotlib smt -i https://repo.huaweicloud.com/repository/pypi/simple 
```

### 5. 验证安装
```bash
python -c "
import torch, numpy, scipy, matplotlib
from smt.sampling_methods import LHS
print('✅ 所有包安装成功')
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
"
```

## 一键安装脚本

```bash
# 删除旧环境
conda deactivate 2>/dev/null
conda env remove -n PINN_WE -y

# 创建新环境
conda create -n PINN_WE python=3.10 -y

# 激活并安装
source /share/project/zhaohuxing/anaconda3/bin/activate PINN_WE
pip install -i https://repo.huaweicloud.com/repository/pypi/simple torch numpy scipy matplotlib smt

# 验证
python -c "import torch, numpy, scipy, matplotlib; from smt.sampling_methods import LHS; print('✅ 安装成功')"
```

## 运行1.py的命令

```bash
cd /share/project/zpy/PINN_WE/cases/1D/Blast
source /share/project/zhaohuxing/anaconda3/bin/activate PINN_WE
export PYTHONPATH=/share/project/zpy/PINN_WE/PINNsrc:$PYTHONPATH
python 1.py
```

