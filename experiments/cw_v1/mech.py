# -*- coding: utf-8 -*-
"""CW-POMDP-v1.1 的**机制量**（尺度不变形态）—— R1.17 起为判据与常驻闸的**唯一实现**。

抽出这个文件的原因：`gates.py` 与入口脚本必须用**同一份代码**算 `ku/kf`，
否则闸只是"另一个实现碰巧也过"（R1.15 的教训：口径一漂，闸就变装饰）。

定义（预注册 `docs/research/CW-R1.17-尺度不变机制判据-预注册-v0.1.md` §3）
--------------------------------------------------------------------------
给定 `ni_e/nd_e/ni_a/nd_a`（`n_users × n_bins` 计数矩阵），令 `N^E = Σni_e`：

    m_i   = Σ_b (ni_e[i,b] − ni_a[i,b])
    Mu    = Σ_i |m_i| / (2 N^E)                       # 用户级重分配的 TVD 比例
    Mf    = Σ_ib |ni_e[i,b] − ni_a[i,b]| / (2 N^E)    # 帧级 TVD 比例
    E_user = Σ_i m_i (ρ_i − ρ_a) / N^E                # 与 R1.15 `decompose` 逐位一致
    ku    = E_user / Mu = 2 Σ_i m_i (ρ_i − ρ_a) / Σ_i |m_i|

★ **有界性（本节是 R1.17 §8-5 更正后的正确形式）**
`ku = Σ_i (2 m_i / Σ|m|)(ρ_i − ρ_a)`，权重绝对值之和为 2（不是 1）
⇒ `|ku| ≤ 2 · max_i |ρ_i − ρ_a| ≤ 2`。
首版写成 `≤ max|ρ_i − ρ_a|` 漏了 TVD 定义里的 `1/2`，实测构造 `ku=−0.6` 而
`max|ρ−ρ_a|=0.4` 当场暴露（自伤 #4）。

★ **尺度不变性**
`E_user` 与 `Mu` 都只依赖 `ni_*` 与 `nd_a`，**不含 `nd_e`**
⇒ 只改送达结果（`nd_e`）时 `ku` 逐位不变，而 `Δr` 会变 ⇒ `share = E/Δr` 会变。
`S1`/`S6`/闸 `4aa` 就是这个构造的实测版。
"""

from __future__ import annotations

import math

import numpy as np


def binom_two_sided(k_neg: int, k_pos: int, p: float = 0.5) -> float:
    """双侧精确二项检验：`H0: P(负) = p`。

    双侧口径 = `2·min(P(X≤k_neg), P(X≥k_neg))`（`n = k_neg+k_pos`），截到 1。
    ★ 主判用保守端 `p=0.5`；另报 `p=1/3` 端（三项分解的恒等式使"指定项与 Δr 同号"
      的先验高于 1/2，用 1/3 是**更宽松**的一端，两个都报才算说清）。
    """
    n = k_neg + k_pos
    if n == 0:
        return float("nan")
    pk = [math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(n + 1)]
    return float(min(1.0, 2 * min(sum(pk[: k_neg + 1]), sum(pk[k_neg:]))))


def mech_quantities(ni_e, nd_e, ni_a, nd_a) -> dict:
    """预注册 §3 的 `Mu/Mf/ku/kf` + 与 `decompose` 同口径的三项。"""
    ni_e = np.asarray(ni_e, dtype=float)
    nd_e = np.asarray(nd_e, dtype=float)
    ni_a = np.asarray(ni_a, dtype=float)
    nd_a = np.asarray(nd_a, dtype=float)
    n_e, n_a = ni_e.sum(), ni_a.sum()
    if n_e <= 0 or n_a <= 0:
        return {"status": "NO-ATTEMPT"}
    r_e, r_a = nd_e.sum() / n_e, nd_a.sum() / n_a
    dr = r_e - r_a
    row_a = ni_a.sum(axis=1)
    rho_i = np.where(row_a > 0, nd_a.sum(axis=1) / np.maximum(row_a, 1), r_a)
    rho_ib = np.where(ni_a > 0, nd_a / np.maximum(ni_a, 1.0), rho_i[:, None])
    m_i = ni_e.sum(axis=1) - row_a
    m_ib = ni_e - ni_a

    e_user = float(np.sum(m_i * (rho_i - r_a)) / n_e)
    e_frame = float(np.sum(m_ib * (rho_ib - rho_i[:, None])) / n_e)
    e_within = float(np.sum((nd_e - nd_a) - rho_ib * m_ib) / n_e)

    tv_u = float(np.abs(m_i).sum())
    tv_f = float(np.abs(m_ib).sum())
    mu = tv_u / (2 * n_e)
    mf = tv_f / (2 * n_e)
    ku = e_user / mu if mu > 0 else float("nan")
    kf = e_frame / mf if mf > 0 else float("nan")

    return {
        "status": "OK", "n_attempt_e": n_e, "n_attempt_a": n_a,
        "n_deliver_e": float(nd_e.sum()), "n_deliver_a": float(nd_a.sum()),
        "rate_e": float(r_e), "rate_a": float(r_a), "dr": float(dr),
        "E_user": e_user, "E_frame": e_frame, "E_within": e_within,
        "additivity_residual": float(abs(e_user + e_frame + e_within - dr)),
        "Mu": mu, "Mf": mf, "ku": float(ku), "kf": float(kf),
        "tv_user": tv_u, "tv_frame": tv_f,
        "shares": {"user": e_user / dr if dr else float("nan"),
                   "frame": e_frame / dr if dr else float("nan"),
                   "within": e_within / dr if dr else float("nan")},
        # ★ 正确上界（含 TVD 的 1/2）：|ku| ≤ 2·max|ρ_i − ρ_a|
        "ku_bound": float(2.0 * np.max(np.abs(rho_i - r_a))),
    }


#: 符号计数门槛（9/10 ⇒ 双侧 p=0.0215；10/10 ⇒ 0.00195。见 R1.7.8 的教训：≥7/8 不是阈值）
SIGN_MIN = 9
#: `ku` 的量级门槛：1 个百分点（预注册 §4.1，量纲依据见 §8-2）
KAPPA_MIN = 0.01


def dir_verdict(kus, kappa_min: float = KAPPA_MIN, label: str = "user") -> dict:
    """预注册 §4.1：方向判词。`NaN`（分母为 0）**不进计数**，单独报 `n_nan`。"""
    vals = [float(v) for v in kus if v is not None and np.isfinite(v)]
    n_nan = len(kus) - len(vals)
    n_neu = sum(1 for v in vals if abs(v) < kappa_min)
    n_neg = sum(1 for v in vals if v <= -kappa_min)
    n_pos = sum(1 for v in vals if v >= kappa_min)
    n = len(vals)
    absmed = float(np.median([abs(v) for v in vals])) if vals else float("nan")
    p50 = binom_two_sided(n_neg, n_pos, 0.5)
    p33 = binom_two_sided(n_neg, n_pos, 1.0 / 3.0)
    if n_neg >= SIGN_MIN and absmed >= kappa_min:
        v = f"DIRECTION-CONFIRMED:{label.upper()}-NEGATIVE"
    elif n_pos >= SIGN_MIN and absmed >= kappa_min:
        v = f"DIRECTION-CONFIRMED:{label.upper()}-POSITIVE"
    else:
        v = "DIRECTION-UNRESOLVED"
    return {"verdict": v, "label": label, "n": n, "n_nan": int(n_nan),
            "n_neg": int(n_neg), "n_pos": int(n_pos), "n_neu": int(n_neu),
            "median_abs": absmed, "kappa_min": float(kappa_min),
            "p_binom_null50": p50, "p_binom_null33": p33}


def combine_dir(vu: dict, vf: dict) -> str:
    """两机制合并：两向都确认 ⇒ `BOTH`。"""
    cu = vu["verdict"].startswith("DIRECTION-CONFIRMED")
    cf = vf["verdict"].startswith("DIRECTION-CONFIRMED")
    if cu and cf:
        return "DIRECTION-CONFIRMED:BOTH"
    if cu:
        return vu["verdict"]
    if cf:
        return vf["verdict"]
    return "DIRECTION-UNRESOLVED"
