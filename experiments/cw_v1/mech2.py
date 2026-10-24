# -*- coding: utf-8 -*-
"""`CW-R2.2` 判据 **v2** 的唯一实现（阈值按 `n` 缩放 / 主判逐对 / 按地板分层）。

预注册：`docs/research/CW-R2.2-判据形态缩放-预注册-v0.1.md`（§2 定义、§3 分层、§5 否定条件）。

为什么单独成文件
----------------
R2.1 的判据写死在 `r2_1_sep.py:80`（`N_POS_SEP=18 / N_POS_WEAK_LO=15`）。**那是历史证据的一部分，不改**
（`R2.1` 的 `summary.json` 必须还能自证）。判据 v2 另立实现，两套并列，逐格对照 —— 这样
"旧判词 → 新判词" 的差别才是可核的，而不是把旧文件改写成一个看起来一直对的样子。

三条硬口径（本文件的全部纪律）
------------------------------
1. **阈值不写计数**：显著性只由精确二项 `p < ALPHA` 决定 ⇒ 天然随 `n` 缩放
   （`18/15` 在 `n=10` 上算术不可达，是 R2.1 的结构性缺陷）。
2. **主判是逐对**：`sign_u` 与 `beat_u = 1[|Δ_u| > floor_u]` 都按**单位配对**算，
   两个分布的**中位之比只作描述**（`ratio_desc`，不进判词）。
3. **不除零、不静默降级**：`floor_u = 0` 合法（记 `beat=1` 若 `|Δ|>0`，并报 `n_zero_floor`）；
   缺格 / 单位数不符 ⇒ `INCOMPLETE-PANEL`，**拒绝出判词**；任一分量非有限 ⇒ `UNDEFINED` 并点名。
"""

from __future__ import annotations

from math import comb

import numpy as np

#: 单侧显著性水平（预注册 §2，写死；**不随 n 变**，变的是 p 的算法本身）
ALPHA = 0.05

#: 判词（互斥、覆盖；预注册 §2 的同名表 + §2.1 反向两类）
V_SEP = "PANEL-SEPARATES"
V_SEP_REV = "PANEL-SEPARATES-REVERSED"
V_SIGN_ONLY = "SIGN-ONLY-NO-MAGNITUDE"
V_SIGN_ONLY_REV = "SIGN-ONLY-REVERSED"
V_MAG_ONLY = "MAGNITUDE-ONLY-NO-SIGN"
V_CANNOT = "CANNOT-SEPARATE"
V_UNDEF = "UNDEFINED"
V_INCOMPLETE = "INCOMPLETE-PANEL"
V_NO_CLAIM = "NO-CLAIM-HIGH-FLOOR"

#: 旧判据的复刻常量（**只用于活体对照**，绝不参与任何新判词；R2.1 `r2_1_sep.py:61` 同值）
LEGACY_N_POS_SEP = 18
LEGACY_N_POS_WEAK_LO = 15
LEGACY_ALPHA = 0.01


class IncompletePanel(ValueError):
    """缺格 / 单位数不符 ⇒ 必须拒绝出判词（预注册 §2 末行）。"""


# ==================================================================== 二项
def binom_p_ge(k: int, n: int) -> float:
    """`P(X ≥ k | n, 1/2)`，整数组合和精确算（不引 scipy；与 R2.1 同式）。"""
    k = int(k)
    n = int(n)
    if n <= 0:
        raise ValueError("n 必须 > 0")
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    return sum(comb(n, i) for i in range(k, n + 1)) / float(2 ** n)


# ==================================================================== 数据装配
def build_units(cells: list[dict], nulls: list[dict], plants: list[str],
                eps: list[int], model_strong: str = "belief_readout",
                model_weak: str = "prior_only") -> list[dict]:
    """把 40 格面板 + 20 格零控（R2.1 的落盘结构）装配成逐单位配对行。

    `Δ_u = cost(weak) − cost(strong)`（`>0` ⇒ 弱基线更贵 ⇒ 强基线更好）；
    `floor_u = |cost(strong, seed_panel) − cost(strong, seed_null)|`（与 R2.1 `null_pair` 同定义）。

    缺任何 `(plant, ep)` 组合 ⇒ 抛 `IncompletePanel`（**不许**用剩下的格凑一个数）。
    """
    rows: list[dict] = []
    missing: list[str] = []
    for plant in plants:
        for ep in eps:
            try:
                c_strong = next(c for c in cells
                                if c["plant"] == plant and c["model"] == model_strong
                                and int(c["ep"]) == int(ep))
                c_weak = next(c for c in cells
                              if c["plant"] == plant and c["model"] == model_weak
                              and int(c["ep"]) == int(ep))
                c_null = next(c for c in nulls
                              if c["plant"] == plant and int(c["ep"]) == int(ep))
            except StopIteration:
                missing.append(f"{plant}/ep{ep}")
                continue
            delta = float(c_weak["cost_true_per_step"]) - float(c_strong["cost_true_per_step"])
            floor = abs(float(c_strong["cost_true_per_step"]) - float(c_null["cost_true_per_step"]))
            rows.append({
                "plant": plant, "ep": int(ep),
                "cost_strong": float(c_strong["cost_true_per_step"]),
                "cost_weak": float(c_weak["cost_true_per_step"]),
                "delta": delta, "floor_pair": floor,
                "sign": (1 if delta > 0 else (0 if delta == 0 else -1)),
                "beat": (1 if abs(delta) > floor else 0),
                "ratio_desc": (abs(delta) / floor) if floor > 0 else None,
            })
    if missing:
        raise IncompletePanel(f"缺格 {len(missing)} 个：{missing[:8]}")
    return rows


# ==================================================================== 判词 v2
def _block(rows: list[dict], expect_n: int | None, alpha: float = ALPHA) -> dict:
    """单块（某个 plant 或 ALL）的判词 v2。"""
    n = len(rows)
    finite = [r for r in rows if np.isfinite(r["delta"]) and np.isfinite(r["floor_pair"])]
    bad = [f'{r["plant"]}/ep{r["ep"]}' for r in rows
           if not (np.isfinite(r["delta"]) and np.isfinite(r["floor_pair"]))]
    # 描述量（**不进判词**）
    d = np.array([r["delta"] for r in finite], dtype=float)
    f = np.array([r["floor_pair"] for r in finite], dtype=float)
    desc = {
        "n_units": n,
        "median_abs_delta": float(np.median(np.abs(d))) if finite else None,
        "median_floor_pair": float(np.median(f)) if finite else None,
        "delta_median": float(np.median(d)) if finite else None,
        "n_zero_floor": int((f == 0).sum()) if finite else None,
        "nonfinite_cells": bad,
    }
    if finite:
        med_abs = float(np.median(np.abs(d)))
        med_floor = float(np.median(f))
        desc["ratio_desc"] = (med_abs / med_floor) if med_floor > 0 else None
    # ---- 拒答优先：缺格 > 非有限
    if expect_n is not None and n != int(expect_n):
        return {"verdict": V_INCOMPLETE, "why": f"单位数 {n} ≠ 预期 {expect_n}", **desc}
    if bad:
        return {"verdict": V_UNDEF, "why": "存在非有限分量", **desc}
    # ---- 两个配对检验（都按 n 精确算，不写计数阈值）
    n_pos = int(sum(1 for r in finite if r["sign"] > 0))
    n_neg = int(sum(1 for r in finite if r["sign"] < 0))
    n_beat = int(sum(int(r["beat"]) for r in finite))
    # ★ 方向是预注册 §2 钉死的：只有 Δ>0 才叫"强基线更好" ⇒ 正向显著性只看 n_pos。
    #   反向另算 `n_neg`，且落到**显式的反向判词**里 —— 不许把"反向全成功"折进"分不开"
    #   （那会把"强基线更差"说成"看不出差别"，`C2` 就是查这个）。
    p_sign = binom_p_ge(n_pos, n)
    p_neg = binom_p_ge(n_neg, n)
    p_beat = binom_p_ge(n_beat, n)
    h_sign = bool(p_sign < alpha)
    h_neg = bool(p_neg < alpha)
    h_beat = bool(p_beat < alpha)
    if h_sign and h_beat:
        v = V_SEP
    elif h_neg and h_beat:
        v = V_SEP_REV
    elif h_sign:
        v = V_SIGN_ONLY
    elif h_neg:
        v = V_SIGN_ONLY_REV
    elif h_beat:
        v = V_MAG_ONLY
    else:
        v = V_CANNOT
    return {"verdict": v, "n_pos": n_pos, "n_neg": n_neg, "n_zero_delta": n - n_pos - n_neg,
            "n_beat": n_beat, "binom_p_sign": p_sign, "binom_p_neg": p_neg,
            "binom_p_beat": p_beat,
            "h_sign": h_sign, "h_neg_reversed": h_neg, "h_beat": h_beat, "alpha": alpha, **desc,
            "note": "Δ = cost(prior_only) − cost(belief_readout)；Δ>0 ⇒ 弱基线更贵 ⇒ 强基线更好"}


def verdicts_v2(rows: list[dict], plants: list[str], eps: list[int],
                alpha: float = ALPHA) -> dict:
    """`ALL` + 逐 plant 的判词 v2（单位数按 `plants × eps` 校验）。"""
    out = {"ALL": _block(rows, len(plants) * len(eps), alpha)}
    for p in plants:
        out[p] = _block([r for r in rows if r["plant"] == p], len(eps), alpha)
    return out


# ==================================================================== 分层
def stratify(rows: list[dict]) -> dict:
    """按 `median(floor_pair)`（**只用 floor 算，不看 Δ**）分 `LOW`/`HIGH`（预注册 §3）。

    `LOW` 判效应；`HIGH` 只报计数，判词 `NO-CLAIM-HIGH-FLOOR`（若给 v2 判词，仅放 `descriptive_v2`）。
    """
    f = np.array([r["floor_pair"] for r in rows], dtype=float)
    cut = float(np.median(f))
    low = [r for r in rows if r["floor_pair"] <= cut]
    high = [r for r in rows if r["floor_pair"] > cut]
    hi = _block(high, None)
    hi_desc = {"verdict": V_NO_CLAIM,
               "why": "预注册 §3：高地板层不出效应判词（地板乘性，跨格不可比）",
               "n_units": len(high),
               "descriptive_v2": hi,
               "n_pos": hi.get("n_pos"), "n_beat": hi.get("n_beat"),
               "median_floor_pair": hi.get("median_floor_pair"),
               "median_abs_delta": hi.get("median_abs_delta")}
    return {"cut_median_floor": cut, "LOW": _block(low, None), "HIGH": hi_desc,
            "n_low": len(low), "n_high": len(high),
            "low_keys": [f'{r["plant"]}/ep{r["ep"]}' for r in low],
            "high_keys": [f'{r["plant"]}/ep{r["ep"]}' for r in high]}


# ==================================================================== 旧判据（活体对照）
def legacy_block(rows: list[dict], n_units: int) -> dict:
    """**逐字复刻** R2.1 `r2_1_sep.py::verdict`（只用于新旧对照，不参与任何新结论）。"""
    d = np.array([r["delta"] for r in rows], dtype=float)
    f = np.array([r["floor_pair"] for r in rows], dtype=float)
    med_abs = float(np.median(np.abs(d)))
    floor = float(np.median(f))
    n_pos = int((d > 0).sum())
    p = binom_p_ge(n_pos, n_units)
    if not np.isfinite(med_abs) or not np.isfinite(floor):
        v = "UNDEFINED"
    elif med_abs <= floor:
        v = "CANNOT-SEPARATE-BASELINES"
    elif n_pos >= LEGACY_N_POS_SEP and p < LEGACY_ALPHA:
        v = "PANEL-SEPARATES"
    elif n_pos >= LEGACY_N_POS_WEAK_LO:
        v = "SEPARATES-MAGNITUDE-ONLY"
    else:
        v = "CANNOT-SEPARATE-BY-SIGN"
    return {"verdict": v, "median_abs_delta": med_abs, "floor_R2": floor,
            "ratio": (med_abs / floor) if floor else float("inf"),
            "n_pos": n_pos, "n_units": int(n_units), "binom_p": p}


# ==================================================================== 描述量
def spearman(x, y) -> float | None:
    """秩相关（无 scipy；平均秩处理并列）。任一非有限 ⇒ `None`。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 3 or not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        return None
    rx = _rank(x)
    ry = _rank(y)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def _rank(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(a.size, dtype=float)
    sa = a[order]
    i = 0
    while i < a.size:
        j = i
        while j + 1 < a.size and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def floor_cost_coupling(rows: list[dict]) -> dict:
    """★ 报 `floor_R2` 必须同报的量（STATE §7.5-4）：地板与成本水平的耦合。"""
    f = [r["floor_pair"] for r in rows]
    c = [r["cost_strong"] for r in rows]
    pos = [r["floor_pair"] for r in rows if r["floor_pair"] > 0]
    return {"spearman_floor_cost": spearman(f, c),
            "floor_min": float(np.min(f)) if f else None,
            "floor_max": float(np.max(f)) if f else None,
            "floor_span_ratio": (float(np.max(f)) / float(np.min(pos))) if pos else None,
            "n": len(rows)}
