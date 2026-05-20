# Legacy script index

这个文件汇总 `self-discovery` 下仍然保留、但不再作为默认入口的历史脚本。

## 当前默认入口

先看：

- [../canonical/README.md](../canonical/README.md)

默认运行：

- [../canonical/nonpinn_fit.py](../canonical/nonpinn_fit.py)
- [../canonical/pinn_inverse.py](../canonical/pinn_inverse.py)
- [../canonical/run_compare.py](../canonical/run_compare.py)

## 历史脚本分组

### 1. 旧扫描实验

- `scan_alpha.py`
- `scan_alpha_v2.py`
- `scan_alpha_v3.py`
- `scan_alpha_v4.py`
- `scan_alpha_v5.py`
- `scan_alpha_v6.py`
- `scan_alpha_v7.py`
- `scan_alpha_v8.py`
- `scan_alpha_v9.py`

用途：不同损失设计、边界条件、几何参数和训练策略的历史扫描。

### 2. 旧验证脚本

- `verify_forward.py`
- `verify_gradient.py`
- `verify_gradient_v2.py`

用途：定位旧版 PINN 原型中的正问题、梯度和训练行为问题。

### 3. 非 PINN 历史检查

- `numerical_quickcheck.py`
- `rigorous_shooting.py`

用途：在旧方程/旧约定上做的快速或较严格数值检查。

## 使用建议

- 想复现当前结论：不要从本文件列出的脚本开始
- 想追溯“为什么旧路线会漂”：按上面分组逐步阅读
- 想看当前统一结果：直接看 `canonical/output/`