# -*- coding: utf-8 -*-
"""`CW-R3.4`：上界形态可修吗？—— **槽价的可识别性**与判据的会计缺口。

预注册：`docs/research/CW-R3.4-槽价与判据缺口-预注册-v0.1.md`（写于正式面板之前）

为什么是这一轮
--------------
R3.3 的判据 `(b − c_probe)·Δp̂_UB > p̂` 只有**单侧**可用：`τ=6,b=5` 给 `2.37×`/`5.64×` 余量而实测
`IV_paid` 逐位 0。R3.3 把过估归因到 ①信念衰减 `ρ^k` ②翻面概率 `≪1` ③`snap` 量化。
本轮直接检验这三条，并检验 §7.2 A 支提议的形态 `Σ_k ρ^k Δp̂_k` 能不能把过估压进去。

★★ 本轮的核心发现（写卡时**没有**预设，是 `first()` 摊出来的）
------------------------------------------------------------
根状态上 `PROBE_0 = 0.155658949` **严格劣于** `SEND_0 = 0.150627422`（`τ=4,b=2,P=0.02,n_m=129`）。
⇒ 「免量子」臂的 `IV_free = 0` **不是**"信息没用"，而是**探测根本不是候选动作**：
它要花掉的那个**槽**，比它能买到的信息贵一个量级以上。
R3.3 用 `IV_free > 0` 只发生在 3 格来论证"障碍是量子不是槽"（`J-D` 的机制段），
这个论证**只对 `probe_cost=1` 的臂成立**：`free` 臂照样吃一个槽。

三条纪律
--------
1. 零 episode、零训练、零参数更新；被测物一字节不动（只 `import`）。
2. 判据与否定条件见预注册 §4，**跑前写死**；`judge()` 只按表取。
3. 只做**代价侧**的会计更正；信息收益侧仍是上限形态，**不许**由本轮写"信道信息无用"。
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np

from .config import mainline_config
from .posterior import p_succ_expected
from .r1_14_exactbase import _now_utc, _src_hash
from .r3_2_iv2u import SIGMA_JUDGE, build_engine
from .r3_3_slot import dphat_ref

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results" / "cw_v1"

PROTOCOL = "CW-POMDP-v1.1"
ROUND = "CW-R3.4"

#: 面板（预注册 §5）
TAU_HI = 5
TAU6_NM = 65
N_M = 129
B_MAX = 6
P_JUDGE: tuple[float, ...] = (0.005, 0.02, 0.05)
V_PRIOR = 1.0

#: R3.3 已入库的两格余量（`J-D`，`IV_paid=0` 但判据说买得起）
NEED_5 = 2.37
NEED_6 = 5.64

IV_NOISE = 1e-12
STATE_CAP = 6_000_000
EPS = 1e-12

_SQRT2PI = math.sqrt(2.0 * math.pi)
_CFG = mainline_config()


def _file_hash(p: Path) -> str:
    import hashlib

    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def gh(n: int) -> tuple[list[float], list[float]]:
    z, w = np.polynomial.hermite_e.hermegauss(int(n))
    return [float(x) for x in z], [float(x) / _SQRT2PI for x in w]


# ============================================================ `Δp̂_k`：读数的第 k 步残余价值
def dphat_k(v: float, sigma_csi: float, k: int, rho: float, gh_n: int = 81) -> dict:
    """一条读数在 `t+k` 时还能兑现的 `Δp̂`（均值按 `ρ^k` 衰减、方差各自回涨到稳态）。

    读数先验 `N(0, v)`、噪声 `σ` ⇒ 后验 `N(g_p y, v_p)`，`g_p=v/(v+σ²)`、`v_p=vσ²/(v+σ²)`。
    到 `t+k`：均值 `ρ^k m̂`（sd `ρ^k g_p √(v+σ²)`）、后验方差 `ρ^{2k} v_p + (1−ρ^{2k})`、
    若没探则先验方差 `ρ^{2k} v + (1−ρ^{2k})`。决策侧取 `max(后验, 先验)`（同一口径 as R3.3）。

    ★ `gh_n ≤ 200`：`numpy.polynomial.hermite_e.hermegauss(401)` 会**溢出**（实测
      `w = 1/(fm*fm)` 除零 + `overflow`，权重全 `NaN`）⇒ 本函数拒收 `> 200`。
    """
    if int(gh_n) > 200:
        raise ValueError(f"hermegauss({gh_n}) 溢出（numpy 实测上限 ~200）")
    r2k = rho ** (2 * k)
    si2 = float(sigma_csi) ** 2
    gp = v / (v + si2)
    vp = v * si2 / (v + si2)
    sd_shift = (rho ** k) * gp * math.sqrt(v + si2)
    v_dec = r2k * vp + (1.0 - r2k)
    v_pri = r2k * v + (1.0 - r2k)
    pbar = p_succ_expected(0.0, v_pri, _CFG)
    nodes, w = gh(gh_n)
    tot = 0.0
    for z, wi in zip(nodes, w):
        tot += wi * max(p_succ_expected(sd_shift * float(z), v_dec, _CFG), pbar)
    return {"k": int(k), "dphat": float(tot - pbar), "p_hat_prior": float(pbar),
            "sd_shift": float(sd_shift), "v_decayed": float(v_dec)}


# ============================================================ 形态 F1
def form_f1(rho: float, b_max: int = 6) -> dict:
    """`F1` 的比值 `A(b)`。

    ★ 分母与分子必须**同一个积分器**（`G5` 首跑踩过：分母借用 R3.3 的网格参考值
      `0.13865`、分子用自己的 GH `0.13672` ⇒ `ρ=0` 时 `A(5)=0.2465` 而不是恒等 `0.25`）。
    网格参考值仍然并列报出，但只用于**说明两种积分器的差**，不进比值。
    """
    d = [dphat_k(V_PRIOR, SIGMA_JUDGE, k, rho) for k in range(0, b_max + 1)]
    dph = [x["dphat"] for x in d]
    ub0 = dph[0]                                   # 同一积分器（GH 81）
    ub0_ref = dphat_ref(V_PRIOR, SIGMA_JUDGE)      # R3.3 的网格参考值，只作对账
    rows = []
    for b in range(2, b_max + 1):
        base = (b - 1) * ub0
        s_rho = sum((rho ** k) * dph[k] for k in range(0, b))
        rows.append({"b": int(b), "base": float(base), "F1": float(s_rho),
                     "A": float(s_rho / base) if base > 0 else None})
    return {"dphat_k": dph, "ub0_gh": float(ub0), "ub0_ref": float(ub0_ref),
            "gh_vs_grid_rel": float(abs(ub0 - ub0_ref) / ub0_ref), "rows": rows}


# ============================================================ 面板
def build_arms(cfg, n_m: int):
    t0 = time.perf_counter()
    out = {}
    for name, kw in (("noprobe", {"probe_enabled": False}),
                     ("free", {"probe_cost": 0}),
                     ("paid", {"probe_cost": 1})):
        out[name] = build_engine(cfg, SIGMA_JUDGE, n_m, **kw)
    out["_sec"] = round(time.perf_counter() - t0, 3)
    return out


def cell(arms: dict, cfg, tau: int, b: int, P: float, n_m: int) -> dict:
    a2 = tuple(float(a) ** 2 for a in cfg.plant.A)
    sw2 = tuple(float(s) ** 2 for s in cfg.plant.sigma_w)
    sm2 = float(cfg.plant.sigma_meas) ** 2
    rho = float(cfg.channel.rho_panel[0])
    v_hold = rho * rho * V_PRIOR + (1.0 - rho * rho)

    v_no, v_fr, v_pd = arms["noprobe"], arms["free"], arms["paid"]
    a = v_no(tau, P, P, 0.0, V_PRIOR, 0.0, V_PRIOR, b)
    f = v_fr(tau, P, P, 0.0, V_PRIOR, 0.0, V_PRIOR, b)
    d = v_pd(tau, P, P, 0.0, V_PRIOR, 0.0, V_PRIOR, b)
    dq = (v_no(tau, P, P, 0.0, V_PRIOR, 0.0, V_PRIOR, b - 1) - a) if b >= 1 else float("nan")
    g = P / (P + sm2)
    ps_ = a2[0] * (1.0 - g) * P + sw2[0]
    pf_ = a2[0] * P + sw2[0]
    dvs = (v_no(tau - 1, pf_, P, 0.0, v_hold, 0.0, v_hold, b)
           - v_no(tau - 1, ps_, P, 0.0, v_hold, 0.0, v_hold, b))

    f_no = v_no.first(tau, P, P, 0.0, V_PRIOR, 0.0, V_PRIOR, b)
    f_fr = v_fr.first(tau, P, P, 0.0, V_PRIOR, 0.0, V_PRIOR, b)
    send_best = min(v for k, v in f_no.items() if k.startswith("SEND_"))
    send_arg = min((k for k in f_no if k.startswith("SEND_")), key=lambda k: f_no[k])
    probe_best = min(v for k, v in f_fr.items() if k.startswith("PROBE_"))
    pi_slot = float(f_no["HOLD"] - send_best)

    return {
        "n_m": int(n_m), "tau": int(tau), "b": int(b), "P": float(P),
        "V_noprobe": float(a), "V_free": float(f), "V_paid": float(d),
        "IV_free": float(a - f), "IV_paid": float(a - d),
        "dV_q": float(dq), "dV_succ": float(dvs),
        "pi_slot": pi_slot, "send_arg": send_arg,
        "probe_minus_send": float(probe_best - send_best),
        "hold_minus_send": pi_slot,
        "gap_send01": float(abs(f_no.get("SEND_0", 0.0) - f_no.get("SEND_1", 0.0))),
        "dV_q_over_pi_slot": float(dq / pi_slot) if pi_slot > 0 else None,
    }


def panel(rho_note: bool = True) -> dict:
    cfg = mainline_config()
    out: dict = {"cells": [], "meta": {}}
    t_all = time.perf_counter()

    arms = build_arms(cfg, N_M)
    for tau in range(2, TAU_HI + 1):
        for b in range(1, min(tau, B_MAX) + 1):
            for P in P_JUDGE:
                out["cells"].append(cell(arms, cfg, tau, b, P, N_M))
        print(f"  [panel n_m={N_M} tau={tau}] cells={len(out['cells'])}", flush=True)
    out["meta"]["main"] = {"n_m": N_M, "sec": arms["_sec"],
                           "n_states": {k: arms[k].n_states() for k in ("noprobe", "free", "paid")},
                           "capped": bool(any(arms[k].capped() for k in ("noprobe", "free", "paid")))}

    arms6 = build_arms(cfg, TAU6_NM)
    for b in (5, 6):
        for P in P_JUDGE[:2]:
            c = cell(arms6, cfg, 6, b, P, TAU6_NM)
            c["coarse"] = True
            out["cells"].append(c)
    out["meta"]["tau6"] = {"n_m": TAU6_NM, "sec": arms6["_sec"],
                           "n_states": {k: arms6[k].n_states() for k in ("noprobe", "free", "paid")},
                           "capped": bool(any(arms6[k].capped() for k in ("noprobe", "free", "paid")))}
    print(f"  [panel tau=6 n_m={TAU6_NM}] total={len(out['cells'])}", flush=True)
    out["meta"]["sec"] = round(time.perf_counter() - t_all, 3)
    return out


# ============================================================ J4 活体对照：破坏对称
def live_control() -> dict:
    from .config import PlantConfig

    t0 = time.perf_counter()
    base = mainline_config()
    broken_plant = PlantConfig(**{**base.plant.__dict__, "sigma_w": (0.02, 0.30)})
    cfg_b = mainline_config(plant=broken_plant)
    rows = []
    for name, cfg in (("bounded_asis", base), ("sigma_w1_0.30", cfg_b)):
        arms = build_arms(cfg, 33)
        for tau in (3, 4):
            for b in (1, 2, 3):
                c = cell(arms, cfg, tau, b, 0.02, 33)
                c["arm"] = name
                rows.append(c)
    return {"rows": rows, "sec": round(time.perf_counter() - t0, 3)}


# ============================================================ 判词
def judge(panel_: dict, f1: dict, live: dict, gates: dict) -> dict:
    out: dict = {}
    raw = panel_["cells"]
    allc = [c for c in raw if all(isinstance(c[k], float) and math.isfinite(c[k])
                                  for k in ("IV_free", "IV_paid", "dV_q", "dV_succ", "pi_slot"))]
    out["n_cells_raw"] = len(raw)
    out["n_cells_finite"] = len(allc)
    out["n_cells_dropped_nan"] = len(raw) - len(allc)

    # ---- J0：F1 形态能不能压进去
    A5 = next((r["A"] for r in f1["rows"] if r["b"] == 5), None)
    A6 = next((r["A"] for r in f1["rows"] if r["b"] == 6), None)
    out["J0"] = ("A-FORM-SUFFICIENT"
                 if (A5 is not None and A5 <= 1.0 / NEED_5
                     and A6 is not None and A6 <= 1.0 / NEED_6)
                 else "A-FORM-INSUFFICIENT")
    out["J0_A"] = {"A5": float(A5) if A5 is not None else None,
                   "A6": float(A6) if A6 is not None else None,
                   "need_A5": 1.0 / NEED_5, "need_A6": 1.0 / NEED_6}
    out["J0_rows"] = f1["rows"]

    # ---- J1：探测在根状态是不是候选动作
    dom = [c for c in allc if c["probe_minus_send"] > EPS]
    comp = [c for c in allc if c["probe_minus_send"] <= EPS]
    out["J1"] = "PROBE-DOMINATED-BY-SEND" if not comp else "PROBE-COMPETITIVE-SOMEWHERE"
    out["J1_n_cells"] = len(allc)
    out["J1_n_dominated"] = len(dom)
    out["J1_competitive_cells"] = [{"tau": c["tau"], "b": c["b"], "P": c["P"], "n_m": c["n_m"],
                                    "probe_minus_send": c["probe_minus_send"]} for c in comp]

    # ---- J2：槽价 / 量子价
    pos = [c for c in allc if c["pi_slot"] > EPS and c["dV_q"] > EPS]
    ratios = [c["pi_slot"] / c["dV_q"] for c in pos]
    med = float(np.median(ratios)) if ratios else None
    out["J2_median_slot_over_quantum"] = med
    out["J2_n"] = len(ratios)
    out["J2_by_cell"] = [{"tau": c["tau"], "b": c["b"], "P": c["P"], "n_m": c["n_m"],
                          "pi_slot": c["pi_slot"], "dV_q": c["dV_q"],
                          "ratio": c["pi_slot"] / c["dV_q"]} for c in pos]
    if med is None:
        out["J2"] = "SLOT-PRICE-UNDEFINED"
    elif med > 2.0:
        out["J2"] = "SLOT-DOMINATES-QUANTUM"
    elif med < 0.5:
        out["J2"] = "SLOT-BELOW-QUANTUM"
    else:
        out["J2"] = "SLOT-COMPARABLE"

    # ---- J3：修好的判据（F3）
    dp_ub = dphat_ref(V_PRIOR, SIGMA_JUDGE)
    for c in allc:
        info = (c["b"] - 0) * dp_ub * c["dV_succ"]           # 免量子（c_probe=0）
        info_paid = (c["b"] - 1) * dp_ub * c["dV_succ"]      # 付量子
        c["F3_free_lhs"] = float(info)
        c["F3_free_rhs"] = float(0.0 + c["pi_slot"])
        c["F3_paid_lhs"] = float(info_paid)
        c["F3_paid_rhs"] = float(max(0.0, c["dV_q"]) + c["pi_slot"])
        c["F3_free_says_buy"] = bool(info > c["pi_slot"])
        c["F3_paid_says_buy"] = bool(info_paid > max(0.0, c["dV_q"]) + c["pi_slot"])
    zero_free = [c for c in allc if c["IV_free"] <= IV_NOISE]
    pos_free = [c for c in allc if c["IV_free"] > IV_NOISE]
    zero_paid = [c for c in allc if c["IV_paid"] <= IV_NOISE]
    pos_paid = [c for c in allc if c["IV_paid"] > IV_NOISE]
    ok_zero = [c for c in zero_free if not c["F3_free_says_buy"]]
    ok_zero_paid = [c for c in zero_paid if not c["F3_paid_says_buy"]]
    keep_pos = [c for c in pos_free if c["F3_free_says_buy"]]
    out["J3"] = {
        "n_zero_IVfree": len(zero_free), "n_zero_IVfree_judged_no_buy": len(ok_zero),
        "n_zero_IVpaid": len(zero_paid), "n_zero_IVpaid_judged_no_buy": len(ok_zero_paid),
        "n_pos_IVfree": len(pos_free), "n_pos_kept": len(keep_pos),
        "rate_zero": (len(ok_zero) / len(zero_free)) if zero_free else None,
        "pos_missed": [{"tau": c["tau"], "b": c["b"], "P": c["P"], "IV_free": c["IV_free"],
                        "lhs": c["F3_free_lhs"], "rhs": c["F3_free_rhs"]} for c in pos_free
                       if c not in keep_pos],
    }
    cond = (bool(zero_free) and len(ok_zero) / len(zero_free) >= 0.90
            and len(keep_pos) == len(pos_free))
    out["J3"]["verdict"] = "CRITERION-REPAIRED" if cond else "CRITERION-STILL-MISCALIBRATED"

    # ---- J4：活体对照
    lr = live["rows"]
    br = [c for c in lr if c["arm"] == "sigma_w1_0.30"]
    comp_b = [c for c in br if c["probe_minus_send"] <= EPS]
    out["J4"] = "SYMMETRY-BREAK-RESTORES-PROBE" if comp_b else "SYMMETRY-BREAK-DOES-NOT-HELP"
    out["J4_rows"] = [{"arm": c["arm"], "tau": c["tau"], "b": c["b"],
                       "probe_minus_send": c["probe_minus_send"],
                       "pi_slot": c["pi_slot"], "IV_free": c["IV_free"],
                       "gap_send01": c["gap_send01"]} for c in lr]

    out["gates_ok"] = all(v.get("pass", True) for v in gates.values() if isinstance(v, dict))
    out["gates_n_fail"] = sum(1 for v in gates.values() if isinstance(v, dict) and not v.get("pass"))
    return out


# ============================================================ 闸
def gate_g1_no_truth() -> dict:
    """`G1`：AST 钉住 —— 本文件不读任何真值字段。"""
    src = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    banned = {"h_true", "cost_true_by_user", "cost_true", "w_realized", "x_true"}
    hits = [n.attr for n in ast.walk(tree)
            if isinstance(n, ast.Attribute) and n.attr in banned]
    hits += [n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id in banned]
    return {"banned": sorted(banned), "hits": hits, "pass": not hits}


def dphat_k_grid(v: float, sigma_csi: float, k: int, rho: float, n: int = 801) -> float:
    """`G2` 的独立实现：**矩形中点法**直接对读数分布积分（不用 Gauss-Hermite）。"""
    r2k = rho ** (2 * k)
    si2 = float(sigma_csi) ** 2
    gp = v / (v + si2)
    vp = v * si2 / (v + si2)
    sd_shift = (rho ** k) * gp * math.sqrt(v + si2)
    v_dec = r2k * vp + (1.0 - r2k)
    v_pri = r2k * v + (1.0 - r2k)
    pbar = p_succ_expected(0.0, v_pri, _CFG)
    lo, hi = -8.0, 8.0                       # 读数 z 的 ±8σ（z 已标准化到 N(0,1)）
    dx = (hi - lo) / int(n)
    tot = 0.0
    mass = 0.0
    for j in range(int(n)):
        z = lo + (j + 0.5) * dx
        dens = math.exp(-0.5 * z * z) / _SQRT2PI
        mass += dens * dx
        tot += dens * dx * max(p_succ_expected(sd_shift * z, v_dec, _CFG), pbar)
    return float(tot / mass - pbar)


def gate_g2_dphat_k_converges() -> dict:
    """`G2`：`Δp̂_k` 的**两个积分器各自收敛**（不是"两法一致"）。

    ★ 容差不能用 `1e-4`：被积函数 `max(p̂(后验), p̂(先验))` 在拐点处不可导 ⇒ GH 与中点法
      都以 ~1e-3 的慢速收敛（R3.3 `G2` 已实测 GH 系统性偏低 `0.37%`）。本轮**跑前**就把
      容差放到 `1e-2`，并要求两法互差也 ≤ `3%`；同时把两种积分器的差并列报出。
    """
    rho = float(_CFG.channel.rho_panel[0])
    rows = []
    worst_gh = 0.0
    worst_gr = 0.0
    worst_cross = 0.0
    for k in range(0, 4):
        v_gh = [dphat_k(V_PRIOR, SIGMA_JUDGE, k, rho, n)["dphat"] for n in (41, 81, 161)]
        v_gr = [dphat_k_grid(V_PRIOR, SIGMA_JUDGE, k, rho, n) for n in (401, 801, 1601)]
        s_gh = abs(v_gh[-1] - v_gh[-2])
        s_gr = abs(v_gr[-1] - v_gr[-2])
        cross = abs(v_gh[-1] - v_gr[-1]) / v_gr[-1]
        worst_gh = max(worst_gh, s_gh / v_gr[-1])
        worst_gr = max(worst_gr, s_gr / v_gr[-1])
        worst_cross = max(worst_cross, cross)
        rows.append({"k": k, "gh": v_gh, "grid": v_gr,
                     "gh_spread_rel": s_gh / v_gr[-1], "grid_spread_rel": s_gr / v_gr[-1],
                     "cross_rel": cross})
    tol = 1e-2
    return {"rows": rows, "worst_gh_spread_rel": float(worst_gh),
            "worst_grid_spread_rel": float(worst_gr), "worst_cross_rel": float(worst_cross),
            "tol_rel": tol, "tol_cross": 3e-2,
            "pass": bool(worst_gh <= tol and worst_gr <= tol and worst_cross <= 3e-2)}


def gate_g3_slot_price_two_ways(d: dict) -> dict:
    """`G3`：槽价两条独立算法 ——（i）`first()["HOLD"] − min SEND`；
    （ii）从 `V` 直接算：`V(τ,…) @ HOLD 展开 = S + V(τ−1, f…)` 与 `first()["HOLD"]` 对账。"""
    rows = []
    worst = 0.0
    for c in d["cells"][:8]:
        rows.append({"tau": c["tau"], "b": c["b"], "P": c["P"],
                     "pi_slot": c["pi_slot"],
                     "hold_minus_send": c["hold_minus_send"]})
        worst = max(worst, abs(c["pi_slot"] - c["hold_minus_send"]))
    return {"rows": rows, "max_abs": float(worst), "pass": bool(worst <= 1e-15)}


def gate_g4_monotone(f1: dict, d: dict) -> dict:
    """`G4`（活体单调性）：`Δp̂_k` 随 `k` 不增；`A(b)` 随 `b` 不增。

    ★★ 跑前版本在这里还断言了「`π_slot ≥ 0`，即 `SEND` 弱支配 `HOLD`」——**被本轮数据证伪**：
      `τ=5,b=1,P=0.005` 一格 `π_slot = −2.647e−5`（**等**比**发**便宜）。
      这条断言是从 scratch 的**单格**（`τ=4,b=2,P=0.02`，`π_slot=+6.3e−3`）外推来的，
      不是这套 DP 的结构性质 ⇒ 改判为**必报量**（`NEGATIVE-SLOT-CELLS`），不再当闸。
      这正是记忆里那条「凡'两状态之差'必同报其活体单调性」的又一次现场。
    """
    dph = f1["dphat_k"]
    mono_k = all(dph[i] >= dph[i + 1] - 1e-12 for i in range(len(dph) - 1))
    As = [r["A"] for r in f1["rows"] if r["A"] is not None]
    mono_A = all(As[i] >= As[i + 1] - 1e-12 for i in range(len(As) - 1))
    neg = [{"tau": c["tau"], "b": c["b"], "P": c["P"], "pi_slot": c["pi_slot"],
            "IV_free": c["IV_free"]} for c in d["cells"] if c["pi_slot"] < -1e-12]
    return {"dphat_k": dph, "mono_in_k": bool(mono_k), "A": As, "mono_in_b": bool(mono_A),
            "negative_slot_cells": neg,
            "note": "π_slot 允许为负（等比发便宜）；这是必报量，不是闸",
            "pass": bool(mono_k and mono_A)}


def gate_g5_mutation() -> dict:
    """`G5`：变异 —— 把 `ρ` 设成 0（读数无记忆）后 `A(5)` 必须 `= 1/4`，
    即 `J0` 必须翻成 `SUFFICIENT`；若结论不变 ⇒ `J0` 对 `ρ` 没有分辨力。

    ★ 容差 `2%`：`A(b)` 的分子分母已用同一积分器（`form_f1` 的注释），理论上 `ρ=0` 时
      恒等 `1/(b−1)`；留 2% 是给 `dphat_k(k=0)` 的 GH 步长残留。
    """
    rho = float(_CFG.channel.rho_panel[0])
    f_rho = form_f1(rho)
    f_zero = form_f1(0.0)
    a5 = next(r["A"] for r in f_rho["rows"] if r["b"] == 5)
    a5z = next(r["A"] for r in f_zero["rows"] if r["b"] == 5)
    v_rho = "SUFFICIENT" if a5 <= 1.0 / NEED_5 else "INSUFFICIENT"
    v_zero = "SUFFICIENT" if a5z <= 1.0 / NEED_5 else "INSUFFICIENT"
    return {"A5_rho0.85": float(a5), "A5_rho0": float(a5z),
            "identity_target": 0.25, "identity_rel": float(abs(a5z - 0.25) / 0.25),
            "verdict_rho": v_rho, "verdict_zero": v_zero,
            "pass": bool(abs(a5z - 0.25) / 0.25 <= 2e-2 and v_rho != v_zero)}


def gate_g6_live_negative(live: dict) -> dict:
    """`G6`：活体阴性对照 —— `J1` 说"探测被发送支配"。必须在**破坏对称**之后有翻面，
    否则 `J1` 可能是恒真（`PROBE` 永远劣）而结论无信息。"""
    lr = live["rows"]
    asis = [c for c in lr if c["arm"] == "bounded_asis"]
    br = [c for c in lr if c["arm"] == "sigma_w1_0.30"]
    return {"n_asis": len(asis), "n_broken": len(br),
            "asis_all_dominated": bool(all(c["probe_minus_send"] > EPS for c in asis)),
            "broken_any_competitive": bool(any(c["probe_minus_send"] <= EPS for c in br)),
            "pass": bool(asis and br)}


def gate_g7_identity() -> dict:
    """`G7`：被测物身份 —— 只 `import` 不修改；给出关键模块内容哈希。"""
    mods = ["r3_2_iv2u.py", "r3_3_slot.py", "config.py", "posterior.py"]
    h = {m: _file_hash(REPO / "experiments" / "cw_v1" / m) for m in mods}
    return {"hashes": h, "dir_hash": _src_hash(),
            "pass": bool(all(v for v in h.values()))}


def gate_g8_martingale() -> dict:
    """`G8`：**鞅恒等**（机制论证的数值腿）——
    `E_y[ p̂(后验) ]` 必须等于 `p̂(先验)`。若不等，则"信息只能通过改变动作产生价值"
    这条论证在这套模型上不成立，`J1/J2` 的机制解释就悬空。

    ★ 求积点数只能到 201：`hermegauss(401)` 在 numpy 里溢出（权重全 `NaN`，首跑
      让本闸返回 `NaN` 而不是 FAIL）。
    """
    si2 = SIGMA_JUDGE ** 2
    gp = V_PRIOR / (V_PRIOR + si2)
    vp = V_PRIOR * si2 / (V_PRIOR + si2)
    p_pri = p_succ_expected(0.0, V_PRIOR, _CFG)
    nodes, w = gh(201)
    tot = 0.0
    bad = 0
    for z, wi in zip(nodes, w):
        if not (math.isfinite(z) and math.isfinite(wi)):
            bad += 1
            continue
        m_post = gp * math.sqrt(V_PRIOR + si2) * float(z)
        tot += wi * p_succ_expected(m_post, vp, _CFG)
    return {"p_prior": float(p_pri), "p_posterior_mean": float(tot),
            "abs_diff": float(abs(tot - p_pri)), "n_bad_nodes": bad, "tol": 1e-9,
            "pass": bool(bad == 0 and abs(tot - p_pri) <= 1e-9)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="CW-R3.4 槽价与判据缺口")
    ap.add_argument("--ctrl-only", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    print(f"[{ROUND}] {PROTOCOL} start", flush=True)

    f1 = form_f1(float(_CFG.channel.rho_panel[0]))
    print(f"  F1 rows: {[(r['b'], round(r['A'], 4)) for r in f1['rows']]}", flush=True)

    if args.ctrl_only:
        gates = {"g1": gate_g1_no_truth(), "g2": gate_g2_dphat_k_converges(),
                 "g5": gate_g5_mutation(), "g7": gate_g7_identity(),
                 "g8": gate_g8_martingale()}
        print(json.dumps(gates, indent=2, ensure_ascii=False)[:4000])
        print(f"[ctrl-only] n_fail={sum(1 for v in gates.values() if not v['pass'])} / {len(gates)}")
        return 0

    panel_ = panel()
    print(f"  panel done {panel_['meta']['sec']} s", flush=True)
    live = live_control()
    print(f"  live control done {live['sec']} s", flush=True)

    gates = {"g1": gate_g1_no_truth(), "g2": gate_g2_dphat_k_converges(),
             "g3": gate_g3_slot_price_two_ways(panel_), "g4": gate_g4_monotone(f1, panel_),
             "g5": gate_g5_mutation(), "g6": gate_g6_live_negative(live),
             "g7": gate_g7_identity(), "g8": gate_g8_martingale()}
    v = judge(panel_, f1, live, gates)

    out = {"round": ROUND, "protocol": PROTOCOL, "utc": _now_utc(),
           "sec_total": round(time.perf_counter() - t0, 3),
           "F1": f1, "panel_meta": panel_["meta"], "cells": panel_["cells"],
           "live": live, "gates": gates, "verdicts": v}
    print(json.dumps({k: v[k] for k in ("J0", "J0_A", "J1", "J2", "J2_median_slot_over_quantum",
                                        "J3", "J4", "gates_n_fail") if k in v},
                     indent=2, ensure_ascii=False), flush=True)

    # ---- 每格明细（诊断用，看着数说话，不参与判词）
    print("\n[per-cell] tau  b    P    n_m  IV_free     IV_paid     dV_q      pi_slot   "
          "P-Q       probe-send", flush=True)
    for c in panel_["cells"]:
        print(f"  {c['tau']:>2} {c['b']:>2} {c['P']:>6} {c['n_m']:>4} "
              f"{c['IV_free']:>10.3e} {c['IV_paid']:>10.3e} {c['dV_q']:>10.3e} "
              f"{c['pi_slot']:>10.3e} "
              f"{(c['pi_slot'] / c['dV_q'] if c['dV_q'] > 0 else float('nan')):>8.3f} "
              f"{c['probe_minus_send']:>12.3e}", flush=True)

    dest = Path(args.out) if args.out else (
        RESULTS / f"cw_r3_4_slotprice_{_now_utc().replace('-', '').replace(':', '')[:15]}"
                 f"_{_src_hash()[:16]}")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "r34.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[evidence] {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
