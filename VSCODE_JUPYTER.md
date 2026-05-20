# 在VSCode中使用Jupyter运行PINN_WE

## VSCode的Jupyter支持

VSCode内置了Jupyter支持，可以直接运行 `.ipynb` 文件，非常方便！

## 设置步骤

### 1. 确保安装了Python扩展

VSCode需要安装：
- **Python扩展**（Microsoft）
- **Jupyter扩展**（通常随Python扩展一起安装）

### 2. 选择正确的Python解释器

1. 在VSCode中打开项目
2. 按 `Ctrl+Shift+P`（或 `Cmd+Shift+P` on Mac）
3. 输入 `Python: Select Interpreter`
4. 选择你的conda环境：`PINN_WE`

或者直接在VSCode底部状态栏点击Python版本，选择 `PINN_WE` 环境。

### 3. 设置环境变量（重要！）

在VSCode中，你需要设置 `PYTHONPATH`。有两种方法：

#### 方法A：在VSCode设置中配置（推荐）

1. 按 `Ctrl+,` 打开设置
2. 搜索 `python.envFile`
3. 创建或编辑 `.vscode/settings.json`：

```json
{
    "python.envFile": "${workspaceFolder}/.env",
    "python.terminal.activateEnvironment": true
}
```

4. 在项目根目录创建 `.env` 文件：

```bash
PYTHONPATH=/share/project/zpy/PINN_WE/PINNsrc:${PYTHONPATH}
```

#### 方法B：在每个notebook的第一个cell设置路径

在notebook的第一个cell添加：

```python
import sys
import os

# 设置PINNsrc路径
pinn_src_path = '/share/project/zpy/PINN_WE/PINNsrc'
if pinn_src_path not in sys.path:
    sys.path.insert(0, pinn_src_path)

print(f'✓ 路径已设置: {pinn_src_path}')
```

### 4. 打开notebook文件

直接在VSCode中打开 `.ipynb` 文件，VSCode会自动识别并显示notebook界面。

## 使用方式

### 方式1：直接运行notebook

1. 打开 `cases/1D/Blast/SodNewCore.ipynb`
2. 在第一个cell添加路径设置（见方法B）
3. 点击每个cell左侧的"运行"按钮，或按 `Shift+Enter`

### 方式2：在notebook中运行Python脚本

创建一个新的cell：

```python
# 运行带可视化的脚本
%run cases/1D/Blast/1_with_plot.py
```

### 方式3：创建新的notebook

1. 在VSCode中创建新文件 `test.ipynb`
2. VSCode会自动识别为Jupyter notebook
3. 添加代码并运行

## 测试设置

创建一个测试cell：

```python
# 测试cell
import sys
import os

# 设置路径
pinn_src_path = '/share/project/zpy/PINN_WE/PINNsrc'
sys.path.insert(0, pinn_src_path)

# 测试导入
try:
    from PINNs import PINNs_WE_Euler_1D
    import torch
    print("✅ 导入成功！")
    print(f"✅ CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"❌ 错误: {e}")
```

## VSCode Jupyter的优势

1. **集成调试**：可以直接在notebook中设置断点
2. **变量查看**：运行后可以在侧边栏查看变量值
3. **代码补全**：自动补全和智能提示
4. **Git集成**：notebook文件可以直接用Git管理
5. **多语言支持**：可以在同一个notebook中混合使用多种语言

## 常见问题

### Q: VSCode找不到Jupyter内核？

**A:** 确保：
1. 选择了正确的Python解释器（PINN_WE环境）
2. 安装了jupyter：`pip install jupyter`
3. 重启VSCode

### Q: 导入PINNs失败？

**A:** 确保：
1. 在第一个cell设置了路径
2. 或者配置了 `.env` 文件
3. 路径是正确的绝对路径

### Q: GPU不可用？

**A:** 在VSCode的终端中运行：
```bash
source /share/project/zhaohuxing/anaconda3/bin/activate PINN_WE
python -c "import torch; print(torch.cuda.is_available())"
```

## 推荐工作流程

1. **在VSCode中打开项目**：`/share/project/zpy/PINN_WE`
2. **打开notebook**：`cases/1D/Blast/SodNewCore.ipynb`
3. **第一个cell添加路径设置**（见方法B）
4. **运行cells**：逐步运行训练和可视化
5. **查看结果**：图片会直接显示在notebook中

## 快捷键

- `Shift+Enter`：运行当前cell并跳到下一个
- `Ctrl+Enter`：运行当前cell
- `A`：在上方插入cell
- `B`：在下方插入cell
- `DD`（按两次D）：删除cell

