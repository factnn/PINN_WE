"""
改进的训练策略：多初始值搜索 + 两阶段训练

策略：
1. 阶段1：多初始值粗搜索
   - 从多个α候选值开始
   - 固定α，只训练网络权重
   - 记录每个候选值的最终损失

2. 阶段2：精细优化
   - 选择损失最小的top-K个候选值
   - 对每个候选值进行联合优化（α + 网络权重）
   - 选择最终损失最小的结果

3. 阶段3：验证
   - 检查最终α是否合理
   - 绘制收敛曲线和解分布
"""

import torch
import torch.optim as optim
import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

from model import GuderleyPINN
from physics import compute_total_loss, rankine_hugoniot_bc


class ImprovedTrainer:
    """改进的训练器"""

    def __init__(
        self,
        gamma: float = 1.4,
        n: int = 2,
        n_interior: int = 200,
        output_dir: str = "output_improved"
    ):
        self.gamma = gamma
        self.n = n
        self.n_interior = n_interior
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # 创建训练点
        self.xi_interior = torch.linspace(0.1, 1.0, n_interior).reshape(-1, 1)
        self.xi_boundary = torch.tensor([[1.0]])

        # 记录所有实验结果
        self.all_results = []

    def train_network_only(
        self,
        alpha_fixed: float,
        epochs: int = 2000,
        lr: float = 1e-3,
        print_every: int = 500
    ) -> Tuple[GuderleyPINN, float]:
        """
        阶段1：固定α，只训练网络权重

        Returns:
            model: 训练好的模型
            final_loss: 最终损失值
        """
        model = GuderleyPINN(alpha_init=alpha_fixed)
        model.train()

        # 冻结α参数
        model.raw_alpha.requires_grad = False

        # 优化器（只优化网络权重）
        optimizer = optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr
        )

        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()

            total_loss, loss_dict = compute_total_loss(
                model, self.xi_interior, self.xi_boundary,
                self.gamma, self.n
            )

            total_loss.backward()
            optimizer.step()

            if epoch % print_every == 0 or epoch == 1:
                print(f"  Epoch {epoch:5d} | Loss: {loss_dict['total']:.6e} | "
                      f"α: {alpha_fixed:.6f} (固定)")

        # 返回最终损失（不使用no_grad，因为compute_total_loss需要计算导数）
        final_loss, _ = compute_total_loss(
            model, self.xi_interior, self.xi_boundary,
            self.gamma, self.n
        )

        return model, final_loss.item()

    def joint_optimization(
        self,
        model: GuderleyPINN,
        epochs: int = 3000,
        lr_network: float = 1e-4,
        lr_alpha: float = 1e-2,
        print_every: int = 500
    ) -> Tuple[GuderleyPINN, float, List[float], List[float]]:
        """
        阶段2：联合优化α和网络权重

        Args:
            model: 预训练的模型
            epochs: 训练轮数
            lr_network: 网络权重学习率
            lr_alpha: α参数学习率

        Returns:
            model: 优化后的模型
            final_loss: 最终损失
            alpha_history: α的历史记录
            loss_history: 损失的历史记录
        """
        model.train()

        # 解冻α参数
        model.raw_alpha.requires_grad = True

        # 使用不同的学习率
        optimizer = optim.Adam([
            {'params': [p for n, p in model.named_parameters() if n != 'raw_alpha'],
             'lr': lr_network},
            {'params': [model.raw_alpha], 'lr': lr_alpha}
        ])

        alpha_history = []
        loss_history = []

        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()

            total_loss, loss_dict = compute_total_loss(
                model, self.xi_interior, self.xi_boundary,
                self.gamma, self.n
            )

            total_loss.backward()
            optimizer.step()

            alpha_current = model.get_alpha_value()
            alpha_history.append(alpha_current)
            loss_history.append(loss_dict['total'])

            if epoch % print_every == 0 or epoch == 1:
                grad_raw = model.raw_alpha.grad.item() if model.raw_alpha.grad is not None else 0.0
                print(f"  Epoch {epoch:5d} | Loss: {loss_dict['total']:.6e} | "
                      f"α: {alpha_current:.6f} | ∂L/∂α: {grad_raw:.6e}")

        return model, loss_history[-1], alpha_history, loss_history

    def run_multi_start_search(
        self,
        alpha_candidates: List[float],
        stage1_epochs: int = 2000,
        top_k: int = 3
    ) -> List[Tuple[float, float, GuderleyPINN]]:
        """
        运行多初始值搜索

        Args:
            alpha_candidates: α候选值列表
            stage1_epochs: 阶段1训练轮数
            top_k: 选择前k个最优候选值

        Returns:
            List of (alpha_init, loss, model)
        """
        print("\n" + "=" * 60)
        print("阶段1：多初始值粗搜索")
        print("=" * 60)
        print(f"候选值: {alpha_candidates}")
        print(f"每个候选值训练 {stage1_epochs} 轮")

        results = []

        for i, alpha_init in enumerate(alpha_candidates, 1):
            print(f"\n[{i}/{len(alpha_candidates)}] 测试 α = {alpha_init:.6f}")
            print("-" * 60)

            model, final_loss = self.train_network_only(
                alpha_fixed=alpha_init,
                epochs=stage1_epochs
            )

            results.append((alpha_init, final_loss, model))
            print(f"  最终损失: {final_loss:.6e}")

        # 按损失排序
        results.sort(key=lambda x: x[1])

        print("\n" + "=" * 60)
        print("阶段1结果汇总")
        print("=" * 60)
        for i, (alpha, loss, _) in enumerate(results, 1):
            marker = "★" if i <= top_k else " "
            print(f"{marker} {i}. α = {alpha:.6f}, Loss = {loss:.6e}")

        print(f"\n选择前 {top_k} 个候选值进入阶段2")

        return results[:top_k]

    def run_fine_optimization(
        self,
        top_candidates: List[Tuple[float, float, GuderleyPINN]],
        stage2_epochs: int = 3000
    ) -> Tuple[GuderleyPINN, float, float, List[float], List[float]]:
        """
        运行精细优化

        Args:
            top_candidates: 阶段1选出的top-k候选值
            stage2_epochs: 阶段2训练轮数

        Returns:
            best_model: 最优模型
            best_alpha: 最优α值
            best_loss: 最优损失
            alpha_history: α历史
            loss_history: 损失历史
        """
        print("\n" + "=" * 60)
        print("阶段2：精细优化（联合优化α和网络）")
        print("=" * 60)

        final_results = []

        for i, (alpha_init, stage1_loss, model) in enumerate(top_candidates, 1):
            print(f"\n[{i}/{len(top_candidates)}] 优化 α_init = {alpha_init:.6f}")
            print("-" * 60)

            model_opt, final_loss, alpha_hist, loss_hist = self.joint_optimization(
                model=model,
                epochs=stage2_epochs
            )

            alpha_final = model_opt.get_alpha_value()
            final_results.append((
                alpha_init, alpha_final, final_loss,
                model_opt, alpha_hist, loss_hist
            ))

            print(f"  α: {alpha_init:.6f} → {alpha_final:.6f}")
            print(f"  Loss: {stage1_loss:.6e} → {final_loss:.6e}")

        # 选择损失最小的
        best_result = min(final_results, key=lambda x: x[2])
        alpha_init, alpha_final, best_loss, best_model, alpha_hist, loss_hist = best_result

        print("\n" + "=" * 60)
        print("阶段2结果汇总")
        print("=" * 60)
        for i, (a_init, a_final, loss, _, _, _) in enumerate(final_results, 1):
            marker = "★" if loss == best_loss else " "
            print(f"{marker} {i}. α: {a_init:.6f} → {a_final:.6f}, Loss = {loss:.6e}")

        return best_model, alpha_final, best_loss, alpha_hist, loss_hist

    def save_results(
        self,
        model: GuderleyPINN,
        alpha_final: float,
        alpha_history: List[float],
        loss_history: List[float]
    ):
        """保存训练结果"""

        print("\n" + "=" * 60)
        print("保存结果")
        print("=" * 60)

        # 保存模型
        torch.save(model.state_dict(), self.output_dir / "model.pth")
        print(f"✓ 模型已保存: {self.output_dir / 'model.pth'}")

        # 保存结果JSON
        alpha_true = 0.717
        results = {
            "alpha_final": alpha_final,
            "alpha_true": alpha_true,
            "error_percent": abs(alpha_final - alpha_true) / alpha_true * 100,
            "final_loss": loss_history[-1],
            "gamma": self.gamma,
            "n": self.n
        }

        with open(self.output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"✓ 结果已保存: {self.output_dir / 'results.json'}")

        # 保存历史
        history = {
            "alpha": alpha_history,
            "loss": loss_history
        }

        with open(self.output_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)
        print(f"✓ 历史已保存: {self.output_dir / 'history.json'}")

    def plot_results(
        self,
        model: GuderleyPINN,
        alpha_history: List[float],
        loss_history: List[float]
    ):
        """绘制结果图表"""

        print("\n" + "=" * 60)
        print("绘制结果图表")
        print("=" * 60)

        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. α收敛曲线
        ax = axes[0, 0]
        ax.plot(alpha_history, 'b-', linewidth=2)
        ax.axhline(y=0.717, color='r', linestyle='--', label='真实值 α=0.717')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('α')
        ax.set_title('α 收敛曲线')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. 损失收敛曲线
        ax = axes[0, 1]
        ax.semilogy(loss_history, 'g-', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('损失收敛曲线')
        ax.grid(True, alpha=0.3)

        # 3. 解的分布 (V, C, G)
        ax = axes[1, 0]
        xi_plot = torch.linspace(0.1, 1.0, 200).reshape(-1, 1)
        with torch.no_grad():
            V, C, G = model(xi_plot)

        xi_np = xi_plot.numpy().flatten()
        ax.plot(xi_np, V.numpy().flatten(), 'b-', label='V (速度)', linewidth=2)
        ax.plot(xi_np, C.numpy().flatten(), 'r-', label='C (声速)', linewidth=2)
        ax.set_xlabel('ξ')
        ax.set_ylabel('V, C')
        ax.set_title('解的分布')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. α误差随epoch变化
        ax = axes[1, 1]
        alpha_true = 0.717
        errors = [abs(a - alpha_true) / alpha_true * 100 for a in alpha_history]
        ax.plot(errors, 'purple', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('相对误差 (%)')
        ax.set_title('α 相对误差')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "results.png", dpi=150, bbox_inches='tight')
        print(f"✓ 图表已保存: {self.output_dir / 'results.png'}")
        plt.close()

    def run(
        self,
        alpha_candidates: List[float] = None,
        stage1_epochs: int = 2000,
        stage2_epochs: int = 3000,
        top_k: int = 3
    ):
        """
        运行完整的训练流程

        Args:
            alpha_candidates: α候选值列表
            stage1_epochs: 阶段1训练轮数
            stage2_epochs: 阶段2训练轮数
            top_k: 选择前k个候选值进入阶段2
        """
        print("=" * 60)
        print("Guderley 自相似指数自发现 - 改进方案")
        print("=" * 60)
        print(f"策略: 多初始值搜索 + 两阶段训练")
        print(f"真实值: α = 0.717")

        # 默认候选值
        if alpha_candidates is None:
            alpha_candidates = [0.65, 0.68, 0.71, 0.74, 0.77, 0.80, 0.83]

        # 阶段1：多初始值搜索
        top_candidates = self.run_multi_start_search(
            alpha_candidates=alpha_candidates,
            stage1_epochs=stage1_epochs,
            top_k=top_k
        )

        # 阶段2：精细优化
        best_model, alpha_final, best_loss, alpha_hist, loss_hist = \
            self.run_fine_optimization(
                top_candidates=top_candidates,
                stage2_epochs=stage2_epochs
            )

        # 最终结果
        print("\n" + "=" * 60)
        print("最终结果")
        print("=" * 60)

        alpha_true = 0.717
        error_percent = abs(alpha_final - alpha_true) / alpha_true * 100

        print(f"\n发现的 α: {alpha_final:.6f}")
        print(f"真实的 α: {alpha_true:.6f}")
        print(f"相对误差: {error_percent:.2f}%")
        print(f"最终损失: {best_loss:.6e}")

        if error_percent < 5.0:
            print("\n✓ 成功！误差小于5%")
        elif error_percent < 10.0:
            print("\n✓ 较好！误差小于10%")
        else:
            print(f"\n✗ 误差较大（{error_percent:.2f}%）")

        # 保存结果
        self.save_results(best_model, alpha_final, alpha_hist, loss_hist)

        # 绘制图表
        self.plot_results(best_model, alpha_hist, loss_hist)

        print("\n" + "=" * 60)
        print("训练完成！")
        print("=" * 60)

        return best_model, alpha_final, error_percent


def main():
    """主函数"""
    trainer = ImprovedTrainer(
        gamma=1.4,
        n=2,
        n_interior=200,
        output_dir="output_improved"
    )

    # 运行训练
    model, alpha_final, error_percent = trainer.run(
        alpha_candidates=[0.65, 0.68, 0.71, 0.74, 0.77, 0.80, 0.83],
        stage1_epochs=2000,
        stage2_epochs=3000,
        top_k=3
    )

    return model, alpha_final, error_percent


if __name__ == "__main__":
    main()

