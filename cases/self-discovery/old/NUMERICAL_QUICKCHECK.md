# 非 PINN 快速验证

这个脚本用最传统的办法做一个快速 sanity check：

- 固定一个 `alpha`
- 从激波边界 `xi = 1` 往内积分一阶自相似 ODE
- 找到最接近声速线 `Delta = (V-1)^2 - C^2 = 0` 的位置
- 同时检查两个分子是否也接近 0

如果某个 `alpha` 真正对应平滑的 Guderley 解，那么它应该让：

- `|Delta|` 变小
- `|N_V| + |N_C|` 也同时变小

脚本输出一个 `regularity_score`：越小越好。

## 文件

- `numerical_quickcheck.py`

## 默认设定

- `gamma = 1.4`
- `geometry_source = 2`
- 默认使用强激波边界

这里的 `geometry_source` 指的是径向源项里的系数 `m`，也就是常见守恒律写法中的 `(m / xi)`：

- 平面：`m = 0`
- 柱面：`m = 1`
- 球面：`m = 2`

这和仓库里某些文件把球面写成 `n=3` 的“空间维数记号”不是一回事。

## 运行示例

```bash
python cases/self-discovery/numerical_quickcheck.py
python cases/self-discovery/numerical_quickcheck.py --alphas 0.68,0.70,0.717,0.73,0.75
python cases/self-discovery/numerical_quickcheck.py --geometry-source 2 --alpha-start 0.68 --alpha-stop 0.75 --num 15
python cases/self-discovery/numerical_quickcheck.py --finite-mach --mach-inf 10
```

## 解读建议

如果最佳 `alpha` 明显靠近 `0.717`，说明：

- 至少你现在写下的 ODE 形式和球面对称几何记号有一定一致性
- 问题更可能出在 PINN 训练策略，而不是方程完全写错

如果最佳 `alpha` 系统性远离 `0.717`，优先检查：

1. `geometry_source` 的定义是否用错
2. 强激波与有限马赫数边界是否混用了
3. 一阶 ODE 的分子公式是否与参考文献一致
4. 你想验证的是不是经典球形 Guderley 特征值问题

## 局限

这只是快速验证，不是严格文献复现：

- 起点用的是激波边界附近的一阶外推
- 没做完整 BVP / shooting 匹配
- 也没显式 enforce 更高阶的 sonic regularity 条件

但它足够用来回答一个很重要的问题：

> 按你当前仓库里写下的方程，经典数值积分会不会天然偏向 `alpha ≈ 0.717`？
