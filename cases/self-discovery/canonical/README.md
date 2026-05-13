# Canonical self-discovery mainline

这个目录是新的干净主线，和旧的 `self-discovery` 分叉脚本解耦。

## 目标

采用统一约定：

- 球面汇聚激波：`alpha = 0.717`
- 柱面汇聚激波：`alpha = 0.800`
- 激波半径相似律：`R(t) = A (t_c - t)^alpha`

先建立一个稳定、可复现的非 PINN 基线，再建立和它完全一致的 PINN 反演版本。

## 文件

- `common.py`：公共数据生成与相似律
- `nonpinn_fit.py`：经典最小二乘基线
- `pinn_inverse.py`：与基线一致的 `torch` 物理反演版（无旧仓库里的分叉方程）
- `run_compare.py`：对比两者恢复到的 `alpha`

## 运行

```bash
source /share/project/zhaohuxing/anaconda3/bin/activate PINN_WE
python cases/self-discovery/canonical/nonpinn_fit.py --geometry spherical
python cases/self-discovery/canonical/pinn_inverse.py --geometry spherical --epochs 3000
python cases/self-discovery/canonical/run_compare.py --geometry spherical --pinn-epochs 3000
```

## 解释

这条主线验证的是：

1. 在当前采用的 `alpha` 定义下，非 PINN 方法能回收 `0.717`
2. `torch` 物理反演版本在完全同一条相似律上也能回收同一个 `alpha`

这样可以把“指数定义是否自洽”和“PINN 训练是否稳定”分开处理。
