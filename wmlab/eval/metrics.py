"""评测指标。

本仓库要回答的核心问题是一个**单点数值**：
    世界模型预测到第几步开始不可靠？

因此指标定义必须写清楚、可复现、可对比 —— 否则面试官无法验证，结论也就不成立。
"""

from __future__ import annotations

from typing import Sequence

import torch


def mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """均方误差。"""
    return float(torch.mean((pred - target) ** 2).item())


def nmse(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    """归一化均方误差：MSE / 目标方差。

    除以目标方差后，不同量纲的状态维度之间可比，也便于跨任务比较。
    这是信道预测文献里最常用的口径，与本课题（信道/轨迹预测）保持一致。
    """
    var = float(torch.var(target, unbiased=False).item())
    return mse(pred, target) / max(var, eps)


def reliable_horizon(
    horizons: Sequence[int],
    errors: Sequence[float],
    threshold: float,
    mode: str = "rel",
) -> float | None:
    """可靠视界：误差**首次**超过阈值的那一步。

    Args:
        horizons: 递增的 rollout 步数
        errors:   对应的误差（mse 或 nmse）
        threshold: 阈值。mode="rel" 时按 NMSE 口径给定（如 0.05 表示误差达目标方差的 5%）
        mode:      "rel"（归一化）或 "abs"（绝对）

    Returns:
        首次穿越阈值的步数（线性插值，返回 float）；若全程未超阈返回 None
        （None 意味着在测得的视界范围内模型始终可靠 —— 应加大视界重测）。

    为什么返回插值而不是整数：整数步会让曲线上的穿越点跳变，
    多个种子一平均就丢掉了信息。插值后的值对种子更稳定，可作为论文里的一个连续指标。
    """
    hs = list(horizons)
    es = list(errors)
    if len(hs) != len(es):
        raise ValueError(f"horizons 与 errors 长度不一致：{len(hs)} vs {len(es)}")
    if len(hs) < 2:
        raise ValueError("至少需要两个视界点才能判定穿越")

    for i in range(len(es)):
        if es[i] <= threshold:
            continue
        if i == 0:
            # 第一步就已超阈 —— 模型基本不可用
            return float(hs[0])
        h0, h1 = hs[i - 1], hs[i]
        e0, e1 = es[i - 1], es[i]
        if e1 == e0:
            return float(h1)
        # 线性插值求穿越点
        frac = (threshold - e0) / (e1 - e0)
        return float(h0 + frac * (h1 - h0))
    return None


def coverage(pred_lo: torch.Tensor, pred_hi: torch.Tensor, target: torch.Tensor) -> float:
    """预测区间的经验覆盖率。

    用于 B1 的"不确定性校准"：名义 90% 的区间，实际是否覆盖了 90% 的真值。
    """
    inside = ((target >= pred_lo) & (target <= pred_hi)).float()
    return float(torch.mean(inside).item())


def ece(confidences: torch.Tensor, correct: torch.Tensor, n_bins: int = 10) -> float:
    """Expected Calibration Error（期望校准误差）。

    世界模型说"我有 80% 把握"时，实际正确率是否真的是 80%。
    这是 B1 的核心指标之一，先在这里定义好接口，等因子化潜空间实现后再接。
    """
    conf = confidences.reshape(-1).float()
    corr = correct.reshape(-1).float()
    edges = torch.linspace(0.0, 1.0, n_bins + 1)
    total = conf.numel()
    if total == 0:
        return 0.0
    e = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        n = int(mask.sum().item())
        if n == 0:
            continue
        avg_conf = float(conf[mask].mean().item())
        avg_acc = float(corr[mask].mean().item())
        e += (n / total) * abs(avg_conf - avg_acc)
    return e
