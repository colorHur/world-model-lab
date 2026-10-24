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


def _rates(ni_e, nd_e, ni_a, nd_a):
    """`ρ_i`/`ρ_ib`/`ρ_a` 的**唯一定义**（与 `mech_quantities` 逐字一致）。"""
    ni_e = np.asarray(ni_e, dtype=float)
    nd_e = np.asarray(nd_e, dtype=float)
    ni_a = np.asarray(ni_a, dtype=float)
    nd_a = np.asarray(nd_a, dtype=float)
    n_a = ni_a.sum()
    if n_a <= 0:
        return None
    r_a = nd_a.sum() / n_a
    row_a = ni_a.sum(axis=1)
    rho_i = np.where(row_a > 0, nd_a.sum(axis=1) / np.maximum(row_a, 1), r_a)
    rho_ib = np.where(ni_a > 0, nd_a / np.maximum(ni_a, 1.0), rho_i[:, None])
    return {"r_a": float(r_a), "rho_i": rho_i, "rho_ib": rho_ib}


def two_user_closed_form(ni_e, nd_e, ni_a, nd_a) -> dict:
    """R1.18 §5 的**2 用户闭式**（`n_users != 2` 时显式 `NOT-APPLICABLE`，不静默算）。

    推导（预注册 §5，纸上完成）：
        `m_0=(G+D)/2`、`m_1=(G−D)/2` ⇒ `|m_0|+|m_1| = max(|G|,|D|)`；
        `m_0 d_0 + m_1 d_1 = (G/2)(ρ_0+ρ_1−2ρ_a) + (D/2)(ρ_0−ρ_1)`
     ⇒ `ku = [G(ρ_0+ρ_1−2ρ_a) + D(ρ_0−ρ_1)] / max(|G|,|D|) = ku_net + ku_swap`。

    ★ 这是**代数恒等式**，不是经验规律（预注册 §8-1）。
    ★ 与 `mech_quantities` 的差别只在**算式路径**（用 `m`/`ρ` 直接算 vs `E_user/Mu`），
      所以它能抓"实现里分母/权重写错"，但抓不到"两者共用的 `ρ` 定义错了"。
    """
    ni_e = np.asarray(ni_e, dtype=float)
    nd_e = np.asarray(nd_e, dtype=float)
    ni_a = np.asarray(ni_a, dtype=float)
    nd_a = np.asarray(nd_a, dtype=float)
    n_users = ni_e.shape[0]
    if n_users != 2:
        return {"status": "NOT-APPLICABLE", "n_users": int(n_users)}
    rt = _rates(ni_e, nd_e, ni_a, nd_a)
    if rt is None:
        return {"status": "NO-ATTEMPT"}
    rho = rt["rho_i"]
    r_a = rt["r_a"]
    m = ni_e.sum(axis=1) - ni_a.sum(axis=1)
    G = float(m[0] + m[1])
    D = float(m[0] - m[1])
    den = max(abs(G), abs(D))
    if den <= 0:
        return {"status": "NO-REALLOCATION", "G": G, "D": D}
    ku_net = G * float(rho[0] + rho[1] - 2 * r_a) / den
    ku_swap = D * float(rho[0] - rho[1]) / den
    q = mech_quantities(ni_e, nd_e, ni_a, nd_a)
    gap = float(abs(rho[0] - rho[1]))
    bound2 = 2.0 * gap
    return {"status": "OK", "G": G, "D": D, "denom": den,
            "rho_0": float(rho[0]), "rho_1": float(rho[1]), "rho_a": r_a,
            "ku_net": float(ku_net), "ku_swap": float(ku_swap),
            "ku_closed_form": float(ku_net + ku_swap),
            "ku_from_mech": float(q["ku"]),
            "abs_residual": float(abs(ku_net + ku_swap - q["ku"])),
            "gap_abs": gap, "bound_2gap": bound2,
            "ratio_abs_ku_over_2gap": (abs(q["ku"]) / bound2) if bound2 > 0 else float("nan"),
            "bound_2gap_ok": bool(abs(q["ku"]) <= bound2 + 1e-12),
            "rho_a_is_midpoint": bool(abs(float(rho[0] + rho[1] - 2 * r_a)) < 1e-12)}


def kf_bound(ni_e, nd_e, ni_a, nd_a) -> dict:
    """`kf` 的加权平均上界：`kf = Σ w_ib(ρ_ib−ρ_i)`、`Σ|w| = 2` ⇒ `|kf| ≤ 2·max|ρ_ib−ρ_i|`。

    ★ 预注册 §5.2 更正：首版写 `≤ max` 漏了权重和的 2 倍（与 R1.17 自伤 #4 同族）。
    """
    ni_e = np.asarray(ni_e, dtype=float)
    nd_e = np.asarray(nd_e, dtype=float)
    ni_a = np.asarray(ni_a, dtype=float)
    nd_a = np.asarray(nd_a, dtype=float)
    rt = _rates(ni_e, nd_e, ni_a, nd_a)
    if rt is None:
        return {"status": "NO-ATTEMPT"}
    dev = np.abs(rt["rho_ib"] - rt["rho_i"][:, None])
    maxdev = float(dev.max()) if dev.size else float("nan")
    q = mech_quantities(ni_e, nd_e, ni_a, nd_a)
    b2 = 2.0 * maxdev
    return {"status": "OK", "max_abs_frame_dev": maxdev, "bound_2x": b2,
            "kf": float(q["kf"]),
            "bound_ok": bool(np.isfinite(q["kf"]) and abs(q["kf"]) <= b2 + 1e-12),
            "slack": float(b2 - abs(q["kf"])) if np.isfinite(q["kf"]) else float("nan")}


# ---- R1.18 §3/§4：逐 ep 标签与总判词（判据唯一实现）
def mech_state(k, kappa: float = KAPPA_MIN) -> str:
    """`NEG`/`POS`/`NEU`/`NA`。`NaN` ⇒ `NA`（**不是** `NEU`：机制未发生 ≠ 零效应）。"""
    if k is None or not np.isfinite(k):
        return "NA"
    if k <= -kappa:
        return "NEG"
    if k >= kappa:
        return "POS"
    return "NEU"


def ep_mech_label(ku, kf, kappa: float = KAPPA_MIN) -> dict:
    """预注册 §3：逐 ep 的 `(u_state, f_state)` + 四分类。"""
    us, fs = mech_state(ku, kappa), mech_state(kf, kappa)
    hu, hf = us == "NEG", fs == "NEG"
    if hu and hf:
        cls = "BOTH-HARMFUL"
    elif hu:
        cls = "USER-ONLY-HARMFUL"
    elif hf:
        cls = "FRAME-ONLY-HARMFUL"
    else:
        cls = "NEITHER-HARMFUL"
    return {"u_state": us, "f_state": fs, "harm_class": cls,
            "ku": None if ku is None else float(ku), "kf": None if kf is None else float(kf)}


def aggregate_mech_verdict(kus, kfs, kappa: float = KAPPA_MIN,
                           sign_min: int = SIGN_MIN) -> dict:
    """预注册 §4 的映射表（写死在跑之前）。"""
    def _state(vals):
        fin = [float(v) for v in vals if v is not None and np.isfinite(v)]
        n_neg = sum(1 for v in fin if v <= -kappa)
        n_pos = sum(1 for v in fin if v >= kappa)
        if n_neg >= sign_min:
            return "NEG", n_neg, n_pos
        if n_pos >= sign_min:
            return "POS", n_neg, n_pos
        return "UNRESOLVED", n_neg, n_pos

    U, un, up = _state(kus)
    F, fn, fp = _state(kfs)
    if U == "NEG" and F == "NEG":
        v = "MECH-BOTH-SYSTEMATIC-NEGATIVE"
    elif U == "NEG" and F == "UNRESOLVED":
        v = "MECH-USER-SYSTEMATIC-FRAME-UNRESOLVED"
    elif U == "POS" and F == "POS":
        v = "MECH-BOTH-SYSTEMATIC-POSITIVE"
    elif U == "POS" and F == "UNRESOLVED":
        v = "MECH-USER-SYSTEMATIC-POSITIVE-FRAME-UNRESOLVED"
    elif U == "UNRESOLVED":
        v = "MECH-UNRESOLVED"
    else:
        v = "MECH-SIGN-CONTRADICTORY"
    return {"verdict": v, "U": U, "F": F,
            "u_n_neg": int(un), "u_n_pos": int(up),
            "f_n_neg": int(fn), "f_n_pos": int(fp),
            "kappa": float(kappa), "sign_min": int(sign_min),
            "note": ("答'哪个通道在有害方向上系统性激活'，**不**答'谁解释缺口'；"
                     "与份额口径的 MIXED 并列，不互相替代。")}


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
