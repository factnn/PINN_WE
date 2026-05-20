# Non-PINN Baseline for Similarity Exponent Recovery

> 说明：这个目录是比 `canonical/` 更早建立的非 PINN 基线。
> 现在它主要保留为参考；默认推荐入口已经切到 [../canonical/README.md](../canonical/README.md)。

这个目录提供一个完全不依赖 PINN 的基线：

- 直接采用用户给定的相似律定义
- 用经典非线性最小二乘反演自相似指数 `alpha`
- 先确认数值反演链路本身能够稳定回收到目标值

## 采用的约定

这里故意**不跟仓库旧代码的各种历史记号绑定**，只采用当前明确约定：

- 球面汇聚激波：`alpha = 0.717`
- 柱面汇聚激波：`alpha = 0.800`
- 激波半径演化：`R(t) = A (t_c - t)^alpha`

这意味着本目录验证的是：

> 在这个相似指数定义下，一个干净的非 PINN 数值反演器是否能稳定恢复 `alpha`。

它**不是**完整 Euler/Guderley 边值问题的最终证明版。

## 文件

- `fit_similarity_exponent.py`：主脚本
- `output/summary.json`：拟合结果汇总
- `output/*.png`：拟合曲线与残差图

## 运行

在你的实验环境中运行：

```bash
source /share/project/zhaohuxing/anaconda3/bin/activate PINN_WE
python cases/self-discovery/nonpinn_baseline/fit_similarity_exponent.py --geometry spherical
python cases/self-discovery/nonpinn_baseline/fit_similarity_exponent.py --geometry both
```

## 预期

球面情形应稳定回收到接近：

- `alpha ≈ 0.717`

柱面情形应稳定回收到接近：

- `alpha ≈ 0.800`

如果这一步都恢复不好，说明问题根本不在 PINN，而在更前面的参数定义或数值实现。