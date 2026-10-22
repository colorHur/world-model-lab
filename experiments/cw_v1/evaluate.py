# -*- coding: utf-8 -*-
"""`CW-EVAL-v1.1`：候选 / 基线比较的**唯一**评价器。

为什么要有这个文件（2026-10-22 验收 P0-A / P0-B）
--------------------------------------------------
旧入口 `r1_3_oracle.py` 有两处**判据级**缺陷，数字没错但结论方向错：

1. **比值方向反了**。旧代码写
       cost_ratio_median = median(baseline_cost) / median(oracle_cost)
   却用 `< 0.9` 判"oracle 更好"。该方向下 `<1` 的真实含义是**基线更便宜**。
   同一函数里逐 episode 胜率用 `baseline - oracle > 0`（oracle 更便宜）判
   ⇒ **两个指标方向相反**，所以"4/4 格比值达标 + 胜率全 0.25"这种自相矛盾的表能同时成立。

2. **4 格却要求 6 格**。`RHOS × RATIOS = 4`，`WIN_MIN_CELLS = 6` 写死
   ⇒ 判据**不可满足**：即使 4 格全过也必判失败。审计用合成四格（每格比值 0.5、胜率 1.0）
   复现了这个"永远 FAIL"。

本文件的纪律（写成代码，不靠自觉）
----------------------------------
· 比值**只有一个定义**：`ratio = candidate / baseline`，`< 1` 表示候选更便宜。
  想表达"基线/候选"的名字里必须带 `inverted`，且**不得**参与判词。
· 配对差 `delta = baseline - candidate`，`> 0` 表示候选更便宜。与上面同向。
· 三个统计量**分别命名、分别输出**，不得互换：
    - `median_pairwise_ratio`：逐 episode 求比值，再取中位（**配对**口径，与胜率同源）
    - `ratio_of_medians`：先各取中位，再相除（**聚合**口径，会被极端 episode 带偏）
    - `median_delta`：逐 episode 差值的中位（**绝对**口径，带成本单位）
  ⇒ 判词只用前两个里的 `median_pairwise_ratio` + 胜率 + `median_delta` 的符号一致性。
· 格数要求**从设计格数推导**：`required_cells = len(design_cells)`。
  旧代码的 `6` 来自一个**从未存在**的 8 格设计（代码里只有 4 格）。
  并且显式拒绝"要求数 > 可用格数"，把不可满足的判据变成**异常**而不是静默 FAIL。
· 缺格 / NaN / 零基线 ⇒ `status != "ok"` ⇒ 总判 `UNDEFINED`，**不能当成"败"**，
  更不能当成"胜"。旧代码把这些都折叠成布尔，于是"数据有问题"和"候选更差"长得一样。

★ 本评价器**不读**任何真值，只吃两份等长的逐 episode 成本数组。
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import numpy as np

EVAL_VERSION = "CW-EVAL-v1.1"

#: 判词词表。★ 已删除 `SPACE-VOID`：一个诊断臂没赢**不能**给整个决策空间下判词
#:   （2026-10-22 验收 P0-C）。新词表只描述"这个臂 vs 这些基线"。
VERDICT_BEATS = "CANDIDATE-BEATS-BASELINE"
VERDICT_NOT_BETTER = "CANDIDATE-NOT-BETTER"
VERDICT_UNDEFINED = "UNDEFINED"

INTERPRETATION = (
    "本判词只描述「被比较的这个候选臂 vs 这些具体基线」在给定格上的成本对比。"
    "候选臂若不是最优规划（例如真值辅助贪心），NOT-BETTER **不支持**"
    "「信息无价值 / 决策空间为空 / 方向不可做」这类推论。"
)


class UnreachableRequirement(ValueError):
    """判据要求的格数多于设计格数 ⇒ 判据不可满足。**必须**抛，不许静默 FAIL。"""


def _as_1d(x: Sequence[float], name: str) -> np.ndarray:
    a = np.asarray(x, dtype=float).reshape(-1)
    if a.size == 0:
        raise ValueError(f"{name} 为空：没有 episode 就没有统计量")
    return a


def paired_delta(baseline: Sequence[float], candidate: Sequence[float]) -> np.ndarray:
    """`delta = baseline − candidate`，逐 episode。`>0` ⇒ 候选更便宜。"""
    b = _as_1d(baseline, "baseline")
    c = _as_1d(candidate, "candidate")
    if b.shape != c.shape:
        raise ValueError(f"配对长度不一致：baseline {b.shape} vs candidate {c.shape}")
    return b - c


def per_episode_ratio(baseline: Sequence[float], candidate: Sequence[float]) -> np.ndarray:
    """`ratio = candidate / baseline`，逐 episode。`<1` ⇒ 候选更便宜。

    基线 ≤ 0 或非有限 ⇒ 该 episode 的比值是 `nan`（**不是** inf 后当赢）。
    """
    b = _as_1d(baseline, "baseline")
    c = _as_1d(candidate, "candidate")
    if b.shape != c.shape:
        raise ValueError(f"配对长度不一致：baseline {b.shape} vs candidate {c.shape}")
    out = np.full(b.shape, np.nan, dtype=float)
    ok = np.isfinite(b) & np.isfinite(c) & (b > 0.0)
    out[ok] = c[ok] / b[ok]
    return out


def bootstrap_median_ci(x: Sequence[float], *, n_boot: int = 4000,
                        alpha: float = 0.05, seed: int = 20261022) -> tuple[float, float]:
    """配对差中位数的 bootstrap 区间（固定种子 ⇒ 可复现）。"""
    a = _as_1d(x, "x")
    a = a[np.isfinite(a)]
    if a.size == 0:
        return (float("nan"), float("nan"))
    if a.size == 1:
        return (float(a[0]), float(a[0]))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(int(n_boot), a.size))
    meds = np.median(a[idx], axis=1)
    lo, hi = np.percentile(meds, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


@dataclass
class CellStats:
    """一个格（rho × 预算档 × 基线）上的配对统计。字段名即口径，改名前先读模块注释。"""

    cell: Any
    n_pairs: int
    baseline_median: float
    candidate_median: float
    median_delta: float
    mean_delta: float
    delta_ci: tuple[float, float]
    win_fraction: float
    n_wins: int
    n_ties: int
    median_pairwise_ratio: float
    ratio_of_medians: float
    status: str
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["cell"] = list(self.cell) if isinstance(self.cell, (tuple, list)) else self.cell
        d["delta_ci"] = [float(self.delta_ci[0]), float(self.delta_ci[1])]
        return d


def compare_cell(baseline: Sequence[float], candidate: Sequence[float], *,
                 cell: Any = None, n_boot: int = 4000) -> CellStats:
    """逐 episode 配对比较。**方向固定**：candidate 便宜 ⇒ `median_delta > 0`、`ratio < 1`。

    返回的 `status` 只有三种取值：
      · `"ok"` —— 全部 episode 都能算比值与差值；
      · `"undefined_nan"` —— 有非有限值；
      · `"undefined_nonpositive_baseline"` —— 基线成本 ≤ 0，比值无定义（差值仍可用）。
    后两种**不得**被判成"胜"或"败"。
    """
    b = _as_1d(baseline, "baseline")
    c = _as_1d(candidate, "candidate")
    if b.shape != c.shape:
        raise ValueError(f"配对长度不一致：baseline {b.shape} vs candidate {c.shape}")

    reasons: list[str] = []
    status = "ok"
    if not (np.all(np.isfinite(b)) and np.all(np.isfinite(c))):
        status = "undefined_nan"
        reasons.append("存在非有限成本（nan/inf）⇒ 比值与差值不可定义")
    elif not np.all(b > 0.0):
        status = "undefined_nonpositive_baseline"
        reasons.append("基线成本 ≤ 0 ⇒ 比值无定义；差值仍可算，但不得据此判胜")

    d = paired_delta(b, c)
    r = per_episode_ratio(b, c)
    finite_r = r[np.isfinite(r)]
    wins = int(np.sum(d > 0))
    ties = int(np.sum(d == 0))
    n = int(b.size)

    median_pairwise = float(np.median(finite_r)) if finite_r.size else float("nan")
    ratio_of_medians = (float(np.median(c) / np.median(b))
                        if np.isfinite(b).all() and float(np.median(b)) > 0 else float("nan"))

    # ★ 内部一致性：`ratio_i < 1` ⟺ `delta_i > 0`（正基线时逐 episode 恒成立），
    #   所以"比值中位数"与"胜率"必须指向同一方向。旧代码方向反了也不报错，这条就是补它。
    #
    # ★★ 但不能写成 `median<1 ⟺ 胜率>50%` —— 那是**错的**，会误杀合法结果：
    #      n 为偶数且恰好一半 episode 胜时，两个中间比值一个 <1、一个 >1，
    #      其中位数可以略小于 1（例 0.999 与 1.000 的平均），此时"胜率 > 50%"为假。
    #      2026-10-22 自查发现：这条写成严格充要会让 even-n 的格子随机被判"统计量混用"。
    #   正确形式（无并列时严格成立，含并列时按有效样本数放宽）：
    #      median<1 ⇒ 至少一半 episode 是胜（wins*2 ≥ n_eff）；
    #      median≥1 ⇒ 胜不可能超过一半（wins*2 ≤ n_eff）。
    if status == "ok" and finite_r.size == n:
        n_eff = n - ties
        left = bool(median_pairwise < 1.0)
        if n_eff <= 0:
            reasons.append("全部 episode 并列 ⇒ 比值与胜率都不含方向信息")
        elif left and wins * 2 < n_eff:
            status = "inconsistent"
            reasons.append(
                f"比值说候选更便宜（median_pairwise_ratio={median_pairwise:.6g}<1），"
                f"但赢的 episode 不到一半（{wins}/{n_eff}）⇒ 两个统计量方向相反，判词作废")
        elif (not left) and wins * 2 > n_eff:
            status = "inconsistent"
            reasons.append(
                f"比值说候选不便宜（median_pairwise_ratio={median_pairwise:.6g}≥1），"
                f"但赢的 episode 过半（{wins}/{n_eff}）⇒ 两个统计量方向相反，判词作废")

    return CellStats(
        cell=cell, n_pairs=n,
        baseline_median=float(np.median(b)), candidate_median=float(np.median(c)),
        median_delta=float(np.median(d)), mean_delta=float(np.mean(d)),
        delta_ci=bootstrap_median_ci(d, n_boot=n_boot),
        win_fraction=float(wins / n), n_wins=wins, n_ties=ties,
        median_pairwise_ratio=median_pairwise, ratio_of_medians=ratio_of_medians,
        status=status, reasons=reasons,
    )


def required_cells_from_design(design_cells: Sequence[Any]) -> int:
    """格数要求**从设计推导**：要"在每一格都成立" ⇒ 要求数 = 设计格数。

    ★ 旧代码的 `WIN_MIN_CELLS = 6` 在一个只有 4 格的实验里被写死 ⇒ 不可满足。
      这里不再接受任何"外部写死的格数"，只接受设计本身。
    """
    n = len(list(design_cells))
    if n <= 0:
        raise UnreachableRequirement("设计格数为 0：没有格就没有判据")
    return n


def decide(cells: Sequence[CellStats], *, required_cells: int,
           ratio_threshold: float = 0.9, win_threshold: float = 0.7) -> dict:
    """按**看结果前写定**的门槛给判词。`required_cells` 由 `required_cells_from_design` 给。"""
    cells = list(cells)
    n = len(cells)
    if required_cells > n:
        raise UnreachableRequirement(
            f"判据要求 {required_cells} 格，但设计只有 {n} 格 ⇒ 该判据不可满足"
            f"（旧代码 WIN_MIN_CELLS=6 vs 4 格正是这一类错误）"
        )

    per_cell = []
    for st in cells:
        # ★ 第二道一致性闸：即使调用方**手搓** CellStats，也要能被抓出来。
        #   `ratio<1` 与「过半 episode 更便宜」是同一件事的两种写法；两者不一致
        #   只可能来自"统计量被混用"（旧代码 P0-B 就是这种），此时该格作废。
        effective_status = st.status
        if effective_status == "ok" and st.n_pairs > 0:
            # ★ 与 `compare_cell` 用**同一条**规则（含 even-n 的放宽），
            #   否则两个入口会给出互相矛盾的判词 —— 那本身就是"同一事实两种口径"。
            n_eff = int(st.n_pairs) - int(st.n_ties)
            left = bool(st.median_pairwise_ratio < 1.0)
            if n_eff > 0 and left and st.n_wins * 2 < n_eff:
                effective_status = "inconsistent"
            elif n_eff > 0 and (not left) and st.n_wins * 2 > n_eff:
                effective_status = "inconsistent"
        ok_status = effective_status == "ok"
        pass_ratio = ok_status and math.isfinite(st.median_pairwise_ratio) \
            and st.median_pairwise_ratio <= ratio_threshold
        pass_win = ok_status and st.win_fraction >= win_threshold
        pass_delta = ok_status and st.median_delta > 0.0
        per_cell.append({
            "cell": list(st.cell) if isinstance(st.cell, (tuple, list)) else st.cell,
            "status": effective_status,
            "median_pairwise_ratio": st.median_pairwise_ratio,
            "ratio_of_medians": st.ratio_of_medians,
            "median_delta": st.median_delta,
            "delta_ci": [float(st.delta_ci[0]), float(st.delta_ci[1])],
            "win_fraction": st.win_fraction,
            "n_pairs": st.n_pairs,
            "pass_ratio": bool(pass_ratio),
            "pass_win": bool(pass_win),
            "pass_delta_sign": bool(pass_delta),
            "pass": bool(pass_ratio and pass_win and pass_delta),
            "reasons": st.reasons + (["手搓 CellStats 的比值与胜率不自洽 ⇒ 该格作废"]
                                     if effective_status == "inconsistent" and st.status == "ok"
                                     else []),
        })

    n_undefined = sum(1 for c in per_cell if c["status"] != "ok")
    n_pass = sum(1 for c in per_cell if c["pass"])

    if n_undefined:
        verdict = VERDICT_UNDEFINED
    elif n_pass >= required_cells:
        verdict = VERDICT_BEATS
    else:
        verdict = VERDICT_NOT_BETTER

    return {
        "eval_version": EVAL_VERSION,
        "verdict": verdict,
        "n_cells": n,
        "required_cells": int(required_cells),
        "n_pass": n_pass,
        "n_undefined": n_undefined,
        "thresholds": {"median_pairwise_ratio_le": ratio_threshold,
                       "win_fraction_ge": win_threshold,
                       "median_delta_gt": 0.0},
        "cells": per_cell,
        "failing_cells": [c["cell"] for c in per_cell if not c["pass"]],
        "interpretation": INTERPRETATION,
    }
