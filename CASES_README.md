# PINN_WE 算例说明

## 算例分类

### 1. 一维欧拉方程算例（1D Euler）

#### **Blast/1.py** - Sod激波管问题（最简单）
- **使用**: `PINNs_WE_Euler_1D` 类（来自PINNsrc）
- **问题**: 经典的Sod激波管问题
- **特点**: 使用加权损失函数进行激波捕捉
- **运行方式**: 需要设置PYTHONPATH

#### **Lax/1.py** - Lax问题
- **使用**: 自己定义的 `DNN` 类（不依赖PINNsrc）
- **问题**: Lax激波管问题
- **特点**: 包含特征波速损失和熵条件
- **运行方式**: 可以直接运行（自包含）

#### **Shu-Osher/shuosh.py** - Shu-Osher问题
- **使用**: 自己定义的 `DNN` 类
- **问题**: 激波与密度波相互作用
- **特点**: 更复杂的初始条件
- **运行方式**: 可以直接运行（自包含）

#### **123problem/** - 123问题
- **使用**: Jupyter notebook
- **问题**: 特殊的Riemann问题

#### **2Blast/** - 双激波问题
- **使用**: Jupyter notebook
- **问题**: 两个激波相互作用

### 2. 一维标量方程

#### **Burgers/** - Burgers方程
- **问题**: 一维Burgers方程（比欧拉方程简单）
- **用途**: 测试基础方法

#### **linear/** - 线性方程
- **问题**: 线性对流方程
- **用途**: 最基础的测试

### 3. 二维算例（2D）

#### **Riemann2D_Case6/** - 二维Riemann问题
#### **shock_cicle/** - 激波绕圆柱
#### **Vortex/** - 涡流问题

## 为什么有这么多算例？

1. **不同复杂度**: 从简单（线性）到复杂（二维激波）
2. **不同方法**: 有些用PINNsrc的类，有些自己实现
3. **不同问题**: 测试不同物理现象（激波、涡流、密度波等）
4. **开发历史**: 可能是逐步开发的，不同时期用了不同方法

## 如何运行原始代码？

### 方法1：使用PINNsrc的算例（如Blast/1.py）

```bash
cd /share/project/zpy/PINN_WE/cases/1D/Blast
source /share/project/zhaohuxing/anaconda3/bin/activate PINN_WE
export PYTHONPATH=/share/project/zpy/PINN_WE/PINNsrc:$PYTHONPATH
python 1.py
```

### 方法2：自包含的算例（如Lax/1.py, Shu-Osher/shuosh.py）

```bash
cd /share/project/zpy/PINN_WE/cases/1D/Lax
source /share/project/zhaohuxing/anaconda3/bin/activate PINN_WE
python 1.py
```

### 方法3：Jupyter notebook算例

```bash
cd /share/project/zpy/PINN_WE/cases/1D/Blast
source /share/project/zhaohuxing/anaconda3/bin/activate PINN_WE
jupyter notebook SodNewCore.ipynb
```

## 我为什么创建了test_simple.py？

我创建 `test_simple.py` 是因为：
1. **快速测试**: 减少epoch数量（100 vs 100000），快速验证环境
2. **路径修复**: 自动设置PYTHONPATH，不需要手动export
3. **便于理解**: 添加了注释和打印信息

**但实际上，你可以直接使用原始代码！** 只需要设置PYTHONPATH即可。

## 推荐使用顺序

1. **Blast/1.py** - 最简单的Sod问题，使用PINNsrc的类
2. **Lax/1.py** - 自包含，可以直接运行
3. **Shu-Osher/shuosh.py** - 更复杂的问题
4. **Jupyter notebooks** - 包含可视化和分析

