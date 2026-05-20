# 实验数据格式说明

## 你的数据需要什么格式？

### 格式1：标准文本文件（推荐）

文件：`experimental_data.txt`
```
t    x    rho    u    p
0.05 0.1  0.95   0.02 0.98
0.05 0.3  0.85   0.05 0.92
0.05 0.5  0.50   0.10 0.60
0.10 0.1  0.90   0.03 0.95
...
```

**要求**：
- 每行一个数据点
- 5列：时间t, 位置x, 密度rho, 速度u, 压力p
- 用空格或制表符分隔
- 可以有表头（第一行会被跳过）

### 格式2：CSV文件

文件：`experimental_data.csv`
```csv
t,x,rho,u,p
0.05,0.1,0.95,0.02,0.98
0.05,0.3,0.85,0.05,0.92
...
```

### 格式3：只有部分物理量

如果你的数据只有压力或只有速度，也可以：

```python
# 只有压力数据
x_data = np.array([[t1, x1], [t2, x2], ...])
p_data = np.array([p1, p2, ...])

# 在损失函数中只使用压力
loss_data = ((p_pred - p_data)**2).mean()
```

## 数据要求

### 1. 坐标范围
- **时间t**: 应该在 [Ts, Te] 范围内（你的问题是[0, 0.2]）
- **位置x**: 应该在 [Xs, Xe] 范围内（你的问题是[0, 1]）

### 2. 单位一致性
- 确保实验数据和PDE使用**相同的单位系统**
- 如果不同，需要转换

### 3. 数据质量
- **噪声**：如果数据有噪声，可以使用加权损失
- **异常值**：建议先检查并处理异常值
- **稀疏性**：数据点不需要很密集，但最好覆盖关键区域

### 4. 关键区域采样
建议在以下区域有数据点：
- **激波附近**：激波位置随时间变化
- **初始位置**：x=0.5附近（Riemann问题的初始间断）
- **边界附近**：x=0和x=1
- **多个时刻**：不同时刻的观测更有价值

## 快速开始

### 步骤1：准备你的数据文件

假设你的数据在 `my_experimental_data.txt`：
```
0.05 0.1 0.95 0.02 0.98
0.05 0.3 0.85 0.05 0.92
0.10 0.2 0.80 0.08 0.85
...
```

### 步骤2：修改代码加载数据

在 `1_with_data_fusion.py` 中修改：

```python
# 替换这部分
def load_experimental_data(data_file=None):
    if data_file is None:
        # 生成模拟数据
        ...
    else:
        # 从文件读取
        data = np.loadtxt(data_file)
        x_exp = data[:, [0, 1]]  # t, x
        rho_exp = data[:, 2]
        u_exp = data[:, 3]
        p_exp = data[:, 4]
        return x_exp, rho_exp, u_exp, p_exp

# 使用你的数据
x_exp, rho_exp, u_exp, p_exp = load_experimental_data('my_experimental_data.txt')
```

### 步骤3：运行

```bash
cd /share/project/zpy/PINN_WE/cases/1D/Blast
source /share/project/zhaohuxing/anaconda3/bin/activate PINN_WE
export PYTHONPATH=/share/project/zpy/PINN_WE/PINNsrc:$PYTHONPATH
python 1_with_data_fusion.py
```

## 数据融合的效果

融合实验数据后：
- ✅ **在数据点附近精度更高**
- ✅ **网络被约束到真实物理值**
- ✅ **可以处理不完整的边界条件**
- ✅ **可以处理逆问题**（从数据反推参数）

## 常见问题

### Q: 数据点太少怎么办？
A: 即使只有几个点也有用，PINN会利用PDE约束其他区域

### Q: 数据有噪声怎么办？
A: 使用加权损失函数 `loss_data_weighted`，给噪声大的数据较小权重

### Q: 只有部分物理量的数据？
A: 可以只融合部分物理量，其他由PDE约束

### Q: 数据单位不一致？
A: 需要归一化或转换单位，确保和PDE一致

