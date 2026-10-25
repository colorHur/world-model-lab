# -*- coding: utf-8 -*-
"""`CW-R2.4` 判据 **v3**：把判据从"成本"推广到**任意逐单位标量**（本轮 = 任务陈旧度）。

预注册：`docs/research/CW-R2.4-AoI主指标-预注册-v0.1.md`（写于 2026-10-25 10:4x，`run_episode`=0）

为什么另立文件
--------------
`mech2.py` 是**成本量**的判据实现（`build_units` 直接读 `cost_true_per_step`），
R2.1/R2.2/R2.3 的 `summary.json` 必须还能自证 ⇒ **不改它**。
本文件只复用它的 `binom_p_ge`（同一个精确二项实现），三腿结构与 `ALPHA` 完全一致。

本文件新增的唯一实质内容
------------------------
1. `aoi_metrics()`：**非删除型**逐用户任务陈旧度（`A_t` 时间填补 / `A_T` 整期惩罚），
   并逐位记录 `k_i`（未送达时隙数）。存在的理由是 `aoi_tavg` 会**删掉首次送达前的窗口**
   （`runner.py:277` 的 `if a >= 0`）—— 见预注册 §2 的 D2/D3。
2. `block3()`：与 `mech2._block` 同构，但判词带 `AOI-` 前缀且**含显式反向**与
   `NO-RESOLUTION-AT-N30`，避免把"反向"折进"分不开"（R2.2 硬口径④）。
3. `null_diagnostics()`：★ 条件化零分布（预注册 §4）—— `N1` 地板符号率、`N2` 幅度经验分位。
   这是 R2.2 `D3` 的固化项：**写分层/条件化判据前先算条件化零分布**。

不做什么
--------
不碰 `mech2.ALPHA`；不引入任何计数阈值；缺格 / 非有限一律**拒答**而不是降级。
"""

from __future__ import annotations

from math import comb

import numpy as np

from .mech2 import ALPHA, IncompletePanel, binom_p_ge  # noqa: F401  (ALPHA 只借用，不改)

#: 判词（**带 `AOI-` 前缀**：与成本量的判词绝不混表）
V_SEP = "AOI-SEPARATES"
V_SEP_REV = "AOI-SEPARATES-REVERSED"
V_SIGN = "AOI-SIGN-ONLY-NO-MAGNITUDE"
V_SIGN_REV = "AOI-SIGN-ONLY-REVERSED"
V_MAG = "AOI-MAGNITUDE-ONLY-NO-SIGN"
V_NONE = "NO-RESOLUTION-AT-N30"
V_UNDEF = "UNDEFINED"
V_INCOMPLETE = "INCOMPLETE-PANEL"
V_CONFLICT = "AOI-VARIANT-CONFLICT"

#: 两个变体的名字（`A_t` 主判、`A_T` 必同报）
VAR_TIME = "A_t"
VAR_FULL = "A_T"


# ==================================================================== 逐用户陈旧度
def _user_aoi(k: int, m: float | None, T: int, variant: str) -> float:
    """单用户、单变体的全时隙均值。

    `k` = 从未送达的时隙数（**前缀**：交付后 `stamp>=0` 永久成立 ⇒ `-1` 只出现在开头）；
    `m` = `aoi_tavg`（只在已服务时隙上取均值；`k==T` 时是 `None`）；
    `T` = 该格实际时隙数。

    - `A_t`：未送达时隙按**槽序号 `t`** 填补（时间轴上的温和下界）
    - `A_T`：未送达时隙按 **`T`** 填补（整期惩罚）
    """
    k = int(k)
    T = int(T)
    if not (0 <= k <= T):
        raise ValueError(f"k={k} 超出 [0, T={T}]：前缀假设被破坏")
    tau = T - k
    if tau == 0:
        served_sum = 0.0
    else:
        if m is None:
            raise ValueError(f"k={k}<T 但 aoi_tavg=None：字段语义矛盾")
        served_sum = float(m) * tau
    if variant == VAR_TIME:
        fill = (k * (k - 1)) / 2.0          # Σ_{t=0}^{k-1} t
    elif variant == VAR_FULL:
        fill = float(k * T)
    else:
        raise ValueError(f"未知变体 {variant!r}")
    return (fill + served_sum) / T


def aoi_metrics(cell: dict, variants=(VAR_TIME, VAR_FULL)) -> dict:
    """从一个已落盘格子（必须含 `aoi_tavg_by_user` 与 `n_unserved_slots_by_user`）算全变体。

    ★ `n_unserved_slots_by_user` **不在 R2.3 的 `cells.json` 里**（`r2_1_sep.run_cell` 不写它），
      所以历史格子**算不出**这个量 —— 这正是本轮要重跑同一批格子的唯一原因。
    """
    T = int(cell["n_slots"])
    m = list(cell["aoi_tavg_by_user"])
    k = list(cell["n_unserved_slots_by_user"])
    if len(m) != len(k):
        raise ValueError(f"aoi_tavg({len(m)}) 与 n_unserved({len(k)}) 长度不符")
    out = {"plant": cell["plant"], "model": cell["model"], "ep": int(cell["ep"]),
           "T": T, "k_by_user": [int(x) for x in k], "aoi_tavg_by_user": m,
           "per_user": {}}
    for var in variants:
        vals = [_user_aoi(k[i], m[i], T, var) for i in range(len(m))]
        out["per_user"][var] = vals
        out[var] = float(np.mean(vals))
    out["k_total"] = int(sum(k))
    return out


# ==================================================================== 单位装配
def build_units(cells: list[dict], nulls: list[dict], plants: list[str], eps: list[int],
                variant: str = VAR_TIME, model_strong: str = "belief_readout",
                model_weak: str = "prior_only") -> list[dict]:
    """`Δ_u = A(weak) − A(strong)`；`floor_u = |A(strong,2001) − A(strong,3102)|`。

    缺任何 `(plant, ep)` 组合 ⇒ 抛 `IncompletePanel`（**不许**用剩下的格凑数）。
    """
    rows: list[dict] = []
    missing: list[str] = []
    for plant in plants:
        for ep in eps:
            try:
                cb = next(c for c in cells if c["plant"] == plant
                          and c["model"] == model_strong and int(c["ep"]) == int(ep))
                cw = next(c for c in cells if c["plant"] == plant
                          and c["model"] == model_weak and int(c["ep"]) == int(ep))
                cn = next(c for c in nulls if c["plant"] == plant and int(c["ep"]) == int(ep))
            except StopIteration:
                missing.append(f"{plant}/ep{ep}")
                continue
            mb, mw, mn = (aoi_metrics(cb), aoi_metrics(cw), aoi_metrics(cn))
            a_s, a_w, a_n = mb[variant], mw[variant], mn[variant]
            d = float(a_w) - float(a_s)
            f = float(abs(a_s - a_n))
            rows.append({"plant": plant, "ep": int(ep), "variant": variant,
                         "a_strong": float(a_s), "a_weak": float(a_w), "a_null": float(a_n),
                         "delta": d, "floor_pair": f,
                         "sign": (1 if d > 0 else (0 if d == 0 else -1)),
                         "beat": (1 if abs(d) > f else 0),
                         "ratio_desc": (abs(d) / f if f > 0 else None),
                         "k_strong": mb["k_total"], "k_weak": mw["k_total"], "k_null": mn["k_total"],
                         "per_user_delta": [float(w) - float(s) for s, w in
                                            zip(mb["per_user"][variant], mw["per_user"][variant])]})
    if missing:
        raise IncompletePanel(f"缺格 {len(missing)} 个：{missing[:8]}")
    return rows


# ==================================================================== 判词 v3
def block3(rows: list[dict], expect_n: int | None, alpha: float = ALPHA,
           label: str = "") -> dict:
    """单块三腿判词（结构同 `mech2._block`，判词表按预注册 §3.5）。"""
    n = len(rows)
    finite = [r for r in rows if np.isfinite(r["delta"]) and np.isfinite(r["floor_pair"])]
    bad = [f'{r["plant"]}/ep{r["ep"]}' for r in rows
           if not (np.isfinite(r["delta"]) and np.isfinite(r["floor_pair"]))]
    d = np.array([r["delta"] for r in finite], dtype=float)
    f = np.array([r["floor_pair"] for r in finite], dtype=float)
    desc = {"label": label or None, "n_units": n,
            "median_abs_delta": float(np.median(np.abs(d))) if finite else None,
            "median_floor_pair": float(np.median(f)) if finite else None,
            "delta_median": float(np.median(d)) if finite else None,
            "n_zero_floor": int((f == 0).sum()) if finite else None,
            "ratio_of_medians": None, "nonfinite_cells": bad}
    if finite and float(np.median(f)) > 0:
        desc["ratio_of_medians"] = float(np.median(np.abs(d)) / np.median(f))
    if expect_n is not None and n != int(expect_n):
        return {"verdict": V_INCOMPLETE, "why": f"单位数 {n} ≠ 预期 {expect_n}", **desc}
    if bad:
        return {"verdict": V_UNDEF, "why": "存在非有限分量", **desc}
    n_pos = int(sum(1 for r in finite if r["sign"] > 0))
    n_neg = int(sum(1 for r in finite if r["sign"] < 0))
    n_beat = int(sum(int(r["beat"]) for r in finite))
    p_sign = binom_p_ge(n_pos, n)
    p_neg = binom_p_ge(n_neg, n)
    p_beat = binom_p_ge(n_beat, n)
    h_sign = bool(p_sign < alpha)
    h_neg = bool(p_neg < alpha)
    h_beat = bool(p_beat < alpha)
    if h_sign and h_neg:
        v, why = V_UNDEF, "符号腿两向同时显著（不可能是 Bin(1/2) 的结果）"
    elif h_sign and h_beat:
        v, why = V_SEP, "方向 + 幅度同时过"
    elif h_neg and h_beat:
        v, why = V_SEP_REV, "反向 + 幅度同时过（显式反向判词）"
    elif h_sign:
        v, why = V_SIGN, "只有方向过，幅度压不过该单位自己的地板"
    elif h_neg:
        v, why = V_SIGN_REV, "只有反方向过"
    elif h_beat:
        v, why = V_MAG, "只有幅度过，方向不过"
    else:
        v, why = V_NONE, "三腿都不过 ⇒ 转做小地板，不再加 episode、不再换指标"
    return {"verdict": v, "why": why, "n_pos": n_pos, "n_neg": n_neg,
            "n_zero_delta": n - n_pos - n_neg, "n_beat": n_beat,
            "binom_p_sign": p_sign, "binom_p_neg": p_neg, "binom_p_beat": p_beat,
            "h_sign": h_sign, "h_neg_reversed": h_neg, "h_beat": h_beat, "alpha": alpha, **desc,
            "note": "Δ = A(prior_only) − A(belief_readout)；Δ>0 ⇒ 弱基线更陈旧 ⇒ 强基线更好"}


def verdicts3(rows: list[dict], plants: list[str], eps: list[int], alpha: float = ALPHA,
              variant: str = VAR_TIME) -> dict:
    out = {"ALL": block3(rows, len(plants) * len(eps), alpha, label=f"ALL/{variant}")}
    for p in plants:
        out[p] = block3([r for r in rows if r["plant"] == p], len(eps), alpha,
                        label=f"{p}/{variant}")
    return out


# ==================================================================== 条件化零分布
def binom_p_ge_p(k: int, n: int, p: float) -> float:
    """`P(X ≥ k | Bin(n, p))`，精确整数组合和（`p` 用有理数 1/2 时与 `binom_p_ge` 同值）。"""
    k = int(k)
    n = int(n)
    if n <= 0:
        raise ValueError("n 必须 > 0")
    p = float(p)
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    return sum(comb(n, i) * p ** i * (1.0 - p) ** (n - i) for i in range(k, n + 1))


def null_diagnostics(rows: list[dict], alpha: float = ALPHA) -> dict:
    """`N1` **有符号**零控符号率 + 条件化重算；`N2` 幅度经验分位。

    ★★ 2026-10-25 10:5x 修正（`CW-R2.4b`）：初版用 `floor_pair`（**绝对值**）判"正/负"，
      于是 `p0` 恒为 1.0 —— 那是个**恒真**的判据（`|x|>0` 与符号无关），
      既不能证明也不能否证任何东西。这里改用**有符号**的 `s_u = a_strong − a_null`。
      绝对值版保留在 `N1_abs_vacuous` 里**只作留档**，禁止用于任何决策。
    """
    n = len(rows)
    f = np.array([r["floor_pair"] for r in rows], dtype=float)
    s = np.array([float(r["a_strong"]) - float(r["a_null"]) for r in rows], dtype=float)
    k0 = int((s > 0).sum())
    n_zero_s = int((s == 0).sum())
    p0 = (k0 / float(n - n_zero_s)) if n - n_zero_s > 0 else float("nan")
    n_eff = n - n_zero_s
    p_up = binom_p_ge(k0, n_eff) if n_eff else float("nan")
    p_lo = binom_p_ge(n_eff - k0, n_eff) if n_eff else float("nan")
    p_dev = min(1.0, 2.0 * min(p_up, p_lo)) if n_eff else float("nan")
    n_pos = int(sum(1 for r in rows if r["sign"] > 0))
    n_neg = int(sum(1 for r in rows if r["sign"] < 0))
    if n_pos >= n_neg:
        side, k_obs = p0, n_pos
    else:
        side, k_obs = 1.0 - p0, n_neg
    p_cond = binom_p_ge_p(k_obs, n, side) if np.isfinite(side) else float("nan")
    med = float(np.median(np.abs(np.array([r["delta"] for r in rows], dtype=float))))
    emp = int((f >= med).sum()) / float(n)
    return {"N1": {"n": n, "n_signed_pos": k0, "n_signed_zero": n_zero_s,
                   "n_eff": n_eff, "p0_signed_sign_rate": p0,
                   "s_min": float(s.min()) if n else None,
                   "s_median": float(np.median(s)) if n else None,
                   "s_max": float(s.max()) if n else None,
                   "p_two_sided_vs_half": p_dev,
                   "deviates_at_alpha": bool(np.isfinite(p_dev) and p_dev < alpha),
                   "conditional_side_p": side, "n_pos": n_pos, "n_neg": n_neg,
                   "p_conditional": p_cond,
                   "usage": ("若 deviates ⇒ 判词取 max(p_naive, p_conditional)，两套都报；"
                             "若不 deviates ⇒ 朴素 1/2 零分布成立，不需条件化")},
            "N1_abs_vacuous": {"n_abs_pos": int((f > 0).sum()),
                               "note": ("★ 恒真：`floor` 是绝对值 ⇒ `>0` 与符号无关。"
                                        "初版误用它当符号率（得 30/30），已作废，只留档")},
            "N2": {"median_abs_delta": med, "n_floor_ge_median": int((f >= med).sum()),
                   "empirical_quantile_of_median_effect": emp,
                   "usage": "经验分位越低 ⇒ 中位效应越不像'同模型两次跑'能产生的量"},
            "N3_note": ("`beat` 腿用 Bin(n,1/2) 是**保守上界**：面板两臂共享 plan_seed ⇒ 严格 CRN，"
                        "H0 下 |Δ| 通常远小于换 seed 得到的 floor ⇒ P(beat|H0) < 1/2。"
                        "因此 beat 不显著**不能**说'没有幅度'。")}


# ==================================================================== 敏感性
def k_need_table(ns=(10, 30), alphas=(0.05, 0.025, 0.01)) -> dict:
    out: dict = {}
    for n in ns:
        row = {}
        for a in alphas:
            ks = [k for k in range(n + 1) if binom_p_ge(k, n) < a]
            row[str(a)] = {"k_need": (min(ks) if ks else None),
                           "p_at_k_need": (binom_p_ge(min(ks), n) if ks else None),
                           "p_at_k_need_minus_1": (binom_p_ge(min(ks) - 1, n)
                                                   if ks and min(ks) > 0 else None)}
        out[str(n)] = row
    return out


def variant_agreement(va: dict, vb: dict) -> dict:
    """`A_t` 与 `A_T` 判词是否一致（预注册 §7-6：不一致 ⇒ 拒答）。"""
    keys = [k for k in va if k in vb]
    diff = [k for k in keys if va[k].get("verdict") != vb[k].get("verdict")]
    return {"ok": not diff, "diff_blocks": diff,
            "a": {k: va[k].get("verdict") for k in keys},
            "b": {k: vb[k].get("verdict") for k in keys}}
