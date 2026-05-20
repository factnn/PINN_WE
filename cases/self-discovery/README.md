# Guderley 自相似指数自发现实验

## 当前状态

这个目录里同时保留了两类内容：

1. **新的规范化主线**：见 [canonical/README.md](canonical/README.md)
2. **历史实验脚本**：大量 `scan_alpha_v*`、`run_improved.py`、`src/` 下旧实现等

当前默认应当把 `canonical/` 视为**唯一推荐入口**。

## 推荐入口

如果你的目标是验证当前采用的指数定义：

- 球面汇聚激波：`alpha = 0.717`
- 柱面汇聚激波：`alpha = 0.800`
- 相似律：`R(t) = A (t_c - t)^alpha`

请直接使用：

- [canonical/nonpinn_fit.py](canonical/nonpinn_fit.py)
- [canonical/pinn_inverse.py](canonical/pinn_inverse.py)
- [canonical/run_compare.py](canonical/run_compare.py)

对应说明文档见 [canonical/README.md](canonical/README.md)。

## 为什么现在默认切到 `canonical`

前面的历史代码存在明显混杂：

- 不同脚本混用了不同的 `alpha` 定义和几何记号
- 一部分脚本仍假设输出是 `(V, C, G)`，但模型实现已经只输出 `(V, C)`
- 一些扫描脚本得到的最优值会滑向 `0.8`、`0.9` 甚至 `1.0`
- 旧输出里还能看到明显偏离目标值的结果，例如 [output_improved/results.json](output_improved/results.json#L1-L8)

因此，这个目录现在的维护原则是：

- **验证当前约定**：走 `canonical/`
- **研究历史分叉**：再看旧脚本

## 现在应如何运行

在你的实验环境中：

```bash
source /share/project/zhaohuxing/anaconda3/bin/activate PINN_WE
python cases/self-discovery/canonical/run_compare.py --geometry spherical --pinn-epochs 3000
python cases/self-discovery/canonical/run_compare.py --geometry cylindrical --pinn-epochs 3000
```

## 当前目录说明

- [canonical](canonical)：新的干净主线，推荐使用
- [nonpinn_baseline](nonpinn_baseline)：较早建立的非 PINN 基线，可作为参考
- [src](src)：旧版 PINN 原型实现，**不再作为默认入口**
- `scan_alpha_v*` / `run_improved.py` / `two_stage_training.py`：历史试验脚本，结果仅供追溯
- `output_*`：历史试验输出，不能自动视为可信基准

## 建议阅读顺序

1. [canonical/README.md](canonical/README.md)
2. [nonpinn_baseline/README.md](nonpinn_baseline/README.md)
3. [LEGACY_NOTES.md](LEGACY_NOTES.md)

## 历史说明

如果你要追查“旧方程到底哪里开始漂了”，再去看这些文件：

- [PRINCIPLE.md](PRINCIPLE.md)
- [PHYSICS_CORRECTION_SUMMARY.md](PHYSICS_CORRECTION_SUMMARY.md)
- [FINAL_SUMMARY.md](FINAL_SUMMARY.md)

但它们不再构成当前主线的执行依据。
