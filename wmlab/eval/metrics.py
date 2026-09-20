"""评测指标。

本仓库要回答的核心问题是一个**单点数值**：
    世界模型预测到第几步开始不可靠？

因此指标定义必须写清楚、可复现、可对比 —— 否则面试官无法验证，结论也就不成立。
"""

from __future__ import annotations

import warnings
from typing import Sequence

import torch


def mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """均方误差（累积口径：把 tensor 里所有元素一起平均）。"""
    return float(torch.mean((pred - target) ** 2).item())


def per_step_mse(pred: torch.Tensor, target: torch.Tensor) -> list[float]:
    """**逐点**均方误差：返回每一步自身的误差，不跨步平均。

    与 `mse()` 的关系：若 pred/target 形状为 (B, H, D)，则
        `mse(pred[:, :h], target[:, :h])`  = (1/h) * Σ_{k≤h} per_step_mse[k-1]
    也就是说累积口径是逐点口径的**运行平均**。两者回答不同的问题：

    | 口径 | 回答 |
    |---|---|
    | 累积 `mse` | "平均到第 h 步，误差水平是多少" |
    | 逐点 `per_step_mse` | "**第 h 步本身**有多不准" |

    `reliable_horizon` 问的是"误差**首次**超阈的那一步" ⇒ 必须用逐点。
    用累积口径会把前 h-1 步的小误差摊进均值，**系统性低估**长视界误差、**高估 H\\***。

    Args:
        pred, target: 形状 (B, H, ...) 或 (B, H) 或 (H,)
    Returns:
        长度 H 的 list，第 i 项 = 第 i+1 步的 MSE
    """
    err = (pred - target).float() ** 2
    if err.dim() <= 1:
        return [float(err.mean().item())]
    if err.dim() == 2:                      # (B, H)
        return [float(v) for v in err.mean(dim=0)]
    return [float(v) for v in err.flatten(2).mean(dim=(0, 2))]


def per_step_nmse(pred: torch.Tensor, target: torch.Tensor,
                  eps: float = 1e-8) -> list[float]:
    """**逐点**归一化均方误差：每一步的 MSE 除以该步目标方差。

    与 `nmse()` 同源（都除目标方差），差别只在"是否跨步平均"。
    注意：若下游需要用**全窗口 pooling** 的单一分母（为了与在线 tracking 严格可比），
    请用 `per_step_mse()` 自取分子再除以那个分母 —— 见 `wmlab/eval/tracking.py`。
    """
    ps = per_step_mse(pred, target)
    if target.dim() <= 1:
        vars_ = [float(torch.var(target.float(), unbiased=False).item())]
    else:
        flat = target.float().flatten(2)                      # (B, H, M)
        vars_ = [float(v) for v in flat.var(dim=(0, 2), unbiased=False)]
    return [m / max(v, eps) for m, v in zip(ps, vars_)]


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
    *,
    calibration: str = "pointwise",
) -> float | None:
    """可靠视界：误差**首次**超过阈值的那一步。

    Args:
        horizons: 递增的 rollout 步数
        errors:   对应的误差（mse 或 nmse）
        threshold: 阈值。mode="rel" 时按 NMSE 口径给定（如 0.05 表示误差达目标方差的 5%）
        mode:      "rel"（归一化）或 "abs"（绝对）
        calibration: 传进来的 errors 是哪种口径 —— "pointwise"（默认，**正确**）
                     或 "cumulative"（仅用于复现历史结论，会触发警告）。

    Returns:
        首次穿越阈值的步数（线性插值，返回 float）；若全程未超阈返回 None
        （None 意味着在测得的视界范围内模型始终可靠 —— 应加大视界重测）。

    为什么返回插值而不是整数：整数步会让曲线上的穿越点跳变，
    多个种子一平均就丢掉了信息。插值后的值对种子更稳定，可作为论文里的一个连续指标。

    ★★ 口径必须是逐点（2026-09-20 由 X26 对账发现）
    ------------------------------------------------
    "首次超阈" = 第 h 步**本身**的误差第一次越过阈值 ⇒ errors 必须是逐点口径
    （`per_step_mse` / `per_step_nmse`，或它们在 pooled 分母下的版本）。
    若误传累积口径（`mse(pred[:, :h], tgt[:, :h])`），前 h-1 步的误差会被摊成均值，
    穿越更晚 ⇒ H\\* 被**系统性抬高**。Pendulum 实测：开环 64.83→42.99（+50.8%）、
    闭环 51.96→39.01（+33.2%）；CartPole 开环累积口径直接测不出（`None`）而逐点给 12.95。

    ★ 另：`errors` 的**单位必须与 threshold 同源**。threshold 是归一化口径时，
    `errors` 必须是 NMSE（不是 MSE）—— 混用会让 H\\* 随环境的目标方差漂移。
    """
    if calibration not in ("pointwise", "cumulative"):
        raise ValueError(f"calibration 只能是 pointwise / cumulative，收到 {calibration!r}")
    if calibration == "cumulative":
        warnings.warn(
            "reliable_horizon 收到累积口径曲线：'首次超阈' 的定义要求逐点口径，"
            "累积口径会系统性抬高 H*（Pendulum 实测：开环 +50.8%、闭环 +33.2%；"
            "CartPole 开环甚至测不出）。仅在复现历史结论时才这样用。",
            stacklevel=2,
        )
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
