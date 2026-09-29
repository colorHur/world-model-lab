# -*- coding: utf-8 -*-
"""等预算（等口径）配对的公共逻辑 —— 从 `scripts/24_triggered_scheduling.py` 抽出。

★★ 为什么单独抽一个模块（2026-09-29 的教训）
   原先"两族曲线的重叠窗口"是在脚本里**内联**算的，守卫写成 `hi > lo * 1.0001`。
   结果：X40 首跑在 PER=0.7 档拿到窗口 `[0.1502, 0.1584]` —— **只有 5% 速率跨度** ——
   却照样插值出 9 个"网格点"，打印出「**9/9 更优**」。
   那 9 个点全挤在这 5% 里，**不是 9 个独立速率点** ⇒ 一个假头条。
   ⇒ 抽成有名字、有文档、**有回归测试**的函数，守卫参数显式化。

三条规则（缺一不可）
   1. 每族可比点数 ≥ `min_pts`；
   2. 窗口跨度 `hi / lo ≥ min_range`（默认 1.5×）；
   3. 调用方**必须**对被比较的每一族统一施加同一个过滤（如 `diverged`）——
      本模块只负责窗口，不负责过滤，但**过滤不一致时窗口本身就没有意义**。
"""

from __future__ import annotations

from collections.abc import Sequence


class WindowNotEvaluable(Exception):
    """该档位的公共窗口不足以做等预算比较（点数不足 或 跨度过窄）。"""


def overlap_window(curves: dict[str, Sequence[float]],
                   min_range: float = 1.5,
                   min_pts: int = 3) -> tuple[float, float]:
    """公共重叠窗口 `[max(各族最小), min(各族最大)]`，并施加两条守卫。

    Args:
        curves: {族名: 该族的 x 序列（此处为 tx_rate，须**升序**、已过滤发散点）}
        min_range: 窗口跨度下限 `hi / lo`；< 1.5 视为"同一段重复采样"
        min_pts: 每族至少几个可比点

    Returns:
        (lo, hi)

    Raises:
        WindowNotEvaluable: 附带**可读原因**（调用方应当把它打印出来，
            ★ 不许静默 `continue` —— 静默会被读成"这一档没问题"）。
        ValueError: 传进来的曲线为空或 x 未升序/含非正值。
    """
    if len(curves) < 2:
        raise ValueError("overlap_window 至少要两族才谈得上'等预算比较'")
    for name, xs in curves.items():
        xs = list(xs)
        if len(xs) == 0:
            raise ValueError(f"族 {name} 为空")
        if any(x <= 0 for x in xs):
            raise ValueError(f"族 {name} 含非正的 tx_rate：{xs[:5]}")
        if any(b < a for a, b in zip(xs, xs[1:])):
            raise ValueError(f"族 {name} 的 x 未升序")
    for name, xs in curves.items():
        if len(list(xs)) < min_pts:
            counts = " / ".join(f"{k} {len(list(v))}" for k, v in curves.items())
            raise WindowNotEvaluable(
                f"族内可比点数不足（{counts} < {min_pts}）")
    lo = max(min(list(xs)) for xs in curves.values())
    hi = min(max(list(xs)) for xs in curves.values())
    if not (hi >= lo * min_range):
        raise WindowNotEvaluable(
            f"窗口退化：[{lo:.4f},{hi:.4f}] = {hi / max(lo, 1e-12):.2f}× < {min_range}×"
            f" ⇒ 该档不可评估（9 个网格点会挤在同一段里）")
    return float(lo), float(hi)


def expo_grid(lo: float, hi: float, n: int = 9) -> list[float]:
    """对数等距网格（等预算比较默认 9 点）。"""
    import math
    if n < 2:
        raise ValueError("n 至少 2")
    a, b = math.log(lo), math.log(hi)
    return [math.exp(a + (b - a) * i / (n - 1)) for i in range(n)]


# ------------------------------------------------------------------ P4 聚合口径
DIST_METRIC = "mean_dist_tail"   # 任务指标：稳态跟踪距离（尾部均值）
NMSE_METRIC = "est_nmse"         # 代理指标：估计误差（X35 已证它**不能**当任务代理）


def aggregate_p4(rows) -> tuple:
    """把 `P4_matched_rate` 的 rows 聚成 `(距离比中位, NMSE 比中位, 参与档数)`。

    比值字段是 `utrigger_over_threshold_median` = 不确定性触发 ÷ 年龄阈值（**<1 ⇒ U 更优**）。

    ★★ 口径（回归测试 ㊶ 锁死）：**跨 PER 取中位**，不是取最大 / 最小 / 首个。
    为什么必须钉死：X40 的四个 PER 档比值是 `[1.147, 1.094, 1.023, 1.035]` ——
    取**最大**会报 **1.147**（"差 15%"），取**首个**也报 1.147，取**中位**报 **1.0648**（"差 6%"）。
    **报哪个直接决定结论的强弱**；"取最大"等于拿最差档当代表，是一种**选择性汇报**。
    """
    import statistics
    dists = [r["utrigger_over_threshold_median"] for r in rows
             if r.get("metric") == DIST_METRIC
             and r.get("utrigger_over_threshold_median") is not None]
    nmses = [r["utrigger_over_threshold_median"] for r in rows
             if r.get("metric") == NMSE_METRIC
             and r.get("utrigger_over_threshold_median") is not None]
    dist = float(statistics.median(dists)) if dists else None
    nmse = float(statistics.median(nmses)) if nmses else None
    return dist, nmse, len(dists)
