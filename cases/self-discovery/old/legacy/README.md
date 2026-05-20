# Legacy experiment entrypoints

这个目录用于标记 `self-discovery` 中的历史入口和试验脚本。

## 目的

这些旧脚本不再作为当前主线的一部分，但它们代表了此前做过的多轮探索：

- 老的 PINN 训练入口
- 多轮 `scan_alpha_v*` 扫描
- 不同边界条件和损失权重试验
- 各种调参与问题定位脚本

## 当前原则

- 需要稳定复现当前结论时：使用 [../canonical/README.md](../canonical/README.md)
- 需要追查历史试验时：阅读旧脚本和旧输出目录

## 推荐主线

请优先运行：

- [../canonical/nonpinn_fit.py](../canonical/nonpinn_fit.py)
- [../canonical/pinn_inverse.py](../canonical/pinn_inverse.py)
- [../canonical/run_compare.py](../canonical/run_compare.py)

## 说明

当前仓库还没有把所有旧脚本物理移动到本目录下；
这个目录目前先承担“逻辑归档”和“文档标记”的作用。