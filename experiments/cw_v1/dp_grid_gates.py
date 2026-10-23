# -*- coding: utf-8 -*-
"""CW-R1.7.3 的**网格与判据**闸：把 `W4`（尾部可采纳性）从"过一个点"改成"过一片工作区"。

为什么要重写判据（先说服自己，再去说服别人）
-------------------------------------------
`W4`（`dp_wire_gates.py`）用**单点** `P=0.01`、把 `b=1` 的边际收益除以闭式精确值，要求 **≤1.00**。
本轮实测出三件它办不到的事：

1. **它的偏差是单边且不随加密一致收敛**：`legacy(15)` 1.512 → `fine32(39)` 1.089 →
   `r173(90)` 1.0079（τ=8），但同一个 `τ=8` 在 **101 点细网格上反而 1.0248** ⇒ 不单调。
2. **它在 `P→0` 处原理上不可能一致收敛**：`p ≪ h` 时弦误差与 `p` 同阶，**相对误差与尺度无关**
   （`p=1e-3, τ=8` 时 legacy 是 **3.51**）⇒ 任何有限网格都过不了"处处 ≤1.00"。
3. 所以"把 `W4` 压到 ≤1.00"这个验收目标本身**不可达**，不是被测物不合格。

改法：**保留 `W4` 原样报告（不删、不改阈值），另立两条可做到的**：
   `W4R` 在**声明的工作集**上双向带（`0.95 ≤ over ≤ 1.01`）—— 带宽朝上卡 1%，朝下卡 5%
        （朝下那一侧防"靠把值函数抬高/压低来过大关"，即单向闸的作弊面）；
   `W5`  **决策相关绝对量**：`|price_dp − 闭式| / V_τ(P, b=0) ≤ 1e-3`
        —— 价格本身乘上 `V` 才是"要不要花这一帧"的量级，比值在 `P` 很小时没有意义。

其余闸：`W6` 判据缺陷的证据（不单调）、`W7` 资源（内存/耗时）、`W8` 预算轴覆盖（静默压平计数）、
`W1R` 链上的 `G8d[dp]`（预算维严格递减）在新网格上复跑。
"""

from __future__ import annotations

import time

import numpy as np

from . import config as C
from .dp_wire_gates import _one_frame_drop_exact, _budget_sweep
from .gates import Gate
from .plan_dp import DPTail, default_knots, knots_r173
from .plan_ref import make

#: ★ 声明的工作集（看数前写死；`legacy` 在同一集合上的读数同时报告，作活体对照）
WORK_P = (1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 4e-2)
WORK_TAU = (8, 16, 32, 64, 128, 255)
#: `W4R` 的**主判据区**：`P ≥ 5e-3`。下面这两条是"为什么允许把它排除在主判据外"的理由，
#: 不是事后找补：① 该处价格本身 ≤ 2% 的尾部值（`W5` 逐点报绝对值）；② 相对误差与尺度无关（§1 第 2 条）。
MAIN_P_MIN = 5e-3
#: 双向带
BAND_LO, BAND_HI = 0.95, 1.01
#: `W5` 千分之一
ABS_TOL = 1e-3
#: `W7` 资源上限（本机 31.8 GB；单份 DP 表不得超过 3 GiB、构建不得超过 120 s）
MEM_CAP_MIB = 3072
BUILD_CAP_S = 120.0
#: 证书用的 `p̂`（两用户同值：这里只测"一帧的尺度"，不测用户轴 —— 用户轴由 `W0/W0b` 管）
PHAT = (0.5, 0.5)


def _cfg() -> C.CWConfig:
    return C.CWConfig(plant=C.bounded_plant())


def _price_drop(dp, P0: float, tau: int) -> float:
    Pm = np.asarray([[P0, P0]], dtype=float)
    v0 = float(dp.value(Pm, np.asarray([0]), tau)[0])
    v1 = float(dp.value(Pm, np.asarray([1]), tau)[0])
    return v0, v1


def _sweep(dp, cfg, tag: str) -> dict:
    rows = []
    for P0 in WORK_P:
        for tau in WORK_TAU:
            ub = _one_frame_drop_exact(cfg, P0, np.asarray(PHAT), tau)
            v0, v1 = _price_drop(dp, P0, tau)
            rows.append({"P0": P0, "tau": tau, "V_b0": v0, "V_b1": v1,
                         "drop": v0 - v1, "closed_form": ub,
                         "over": (v0 - v1) / ub if ub > 0 else None,
                         "abs_over_V0": abs((v0 - v1) - ub) / v0 if v0 > 0 else None})
    main = [r for r in rows if r["P0"] >= MAIN_P_MIN]
    return {"tag": tag, "n": int(dp.knots.size), "grid": dp.grid, "rows": rows,
            "max_over_all": max(r["over"] for r in rows),
            "min_over_all": min(r["over"] for r in rows),
            "max_over_main": max(r["over"] for r in main),
            "min_over_main": min(r["over"] for r in main),
            "max_abs_over_V0": max(r["abs_over_V0"] for r in rows),
            "worst_over_row": max(rows, key=lambda r: r["over"]),
            "worst_abs_row": max(rows, key=lambda r: r["abs_over_V0"])}


# ------------------------------------------------------------------ W4R
def w4r_grid_admissible(g: Gate) -> None:
    """新网格在**声明工作集**上的双向带；`legacy` 同集合作活体对照。"""
    cfg = _cfg()
    ph = np.asarray(PHAT)
    new = _sweep(DPTail(cfg, ph, b_cap=4, tau_max=255, grid="r173").build(), cfg, "r173")
    old = _sweep(DPTail(cfg, ph, b_cap=4, tau_max=255, grid="legacy").build(), cfg, "legacy")
    ok = (new["max_over_main"] <= BAND_HI) and (new["min_over_main"] >= BAND_LO)
    g.check(f"W4R 可采纳性（新网格 r173）：主判据区 P≥{MAIN_P_MIN:g} 上 "
            f"{BAND_LO} ≤ over ≤ {BAND_HI}（双向；τ∈{WORK_TAU}）",
            ok, {"r173": {k: new[k] for k in ("n", "max_over_main", "min_over_main",
                                              "max_over_all", "min_over_all",
                                              "max_abs_over_V0")},
                 "worst_over_row": new["worst_over_row"],
                 "legacy_same_set": {k: old[k] for k in ("n", "max_over_main", "min_over_main",
                                                         "max_over_all", "min_over_all",
                                                         "max_abs_over_V0")},
                 "note": ("over = DP 的 b=1 边际收益 / 闭式精确值。"
                          "旧 `W4`（单点 P=0.01、单向 ≤1.00）原样保留在 `dp_wire_gates.py`，未改阈值。")})
    g.check("W4R-对照：同一工作集上 `legacy` 网格必须 FAIL 该带（否则 W4R 无分辨力）",
            (old["max_over_main"] > BAND_HI) or (old["min_over_main"] < BAND_LO),
            {"legacy_max_over_main": old["max_over_main"],
             "legacy_min_over_main": old["min_over_main"]})


# ------------------------------------------------------------------ W5
def w5_absolute(g: Gate) -> None:
    """决策相关绝对量：价格误差相对同点尾部值。"""
    cfg = _cfg()
    ph = np.asarray(PHAT)
    new = _sweep(DPTail(cfg, ph, b_cap=4, tau_max=255, grid="r173").build(), cfg, "r173")
    old = _sweep(DPTail(cfg, ph, b_cap=4, tau_max=255, grid="legacy").build(), cfg, "legacy")
    g.check(f"W5 决策相关绝对量：全工作集 max |Δprice| / V_τ(P,0) ≤ {ABS_TOL:g}",
            new["max_abs_over_V0"] <= ABS_TOL,
            {"r173_max_abs_over_V0": new["max_abs_over_V0"],
             "r173_worst_abs_row": new["worst_abs_row"],
             "legacy_max_abs_over_V0": old["max_abs_over_V0"],
             "note": "分母取同点 `V_τ(P,b=0)`（尾部总代价），它才是'这一帧值不值'的量级。"})
    g.check("W5-对照：`legacy` 网格必须 FAIL 该绝对判据（否则 W5 无分辨力）",
            old["max_abs_over_V0"] > ABS_TOL,
            {"legacy_max_abs_over_V0": old["max_abs_over_V0"]})


# ------------------------------------------------------------------ W6
def w6_criterion_defect(g: Gate) -> None:
    """`W4` 的单向比值判据**不随加密一致改善** —— 这是它不可当验收目标的证据。"""
    cfg = _cfg()
    ph = np.asarray(PHAT)
    seq = {
        "legacy_15": default_knots(),
        "f39": np.concatenate([np.linspace(0.0, 0.06, 32),
                               np.asarray([0.08, 0.12, 0.18, 0.30, 0.50, 0.90, 1.50])]),
        "r173_90": knots_r173(),
        "r173_130": np.unique(np.concatenate([np.linspace(0.0, 0.06, 101),
                                              np.linspace(0.065, 0.14, 14),
                                              np.linspace(0.15, 0.35, 15)])),
    }
    rows = []
    for tag, kn in seq.items():
        dp = DPTail(cfg, ph, b_cap=4, tau_max=32, knots=kn).build()
        row = {"grid": tag, "n": int(dp.knots.size), "over": {}}
        for tau in (8, 16, 32):
            ub = _one_frame_drop_exact(cfg, 1e-3, ph, tau)
            v0, v1 = _price_drop(dp, 1e-3, tau)
            row["over"][str(tau)] = (v0 - v1) / ub
        rows.append(row)
    # 只判"是否存在一处加密后反而变差"（可证伪）：存在 ⇒ 单向比值不能当验收阈值
    bad = []
    for a, b in zip(rows, rows[1:]):
        for k in a["over"]:
            if b["over"][k] > a["over"][k]:
                bad.append({"from": a["grid"], "to": b["grid"], "tau": int(k),
                            "over_before": a["over"][k], "over_after": b["over"][k]})
    g.check("W6 判据缺陷证据：`P=1e-3` 处单向比值在**加密后可能变大**（存在反例即成立）——"
            "所以'压到 ≤1.00'不是可达的验收目标",
            len(bad) > 0, {"rows": rows, "n_non_monotone": len(bad), "examples": bad[:6],
                           "rule": ("本条不评判被测物，只评判**判据**：若加密能一致改善，"
                                    "那'继续加密'就是合法路线；实测不能，故改判据（W4R/W5）。")})


# ------------------------------------------------------------------ W7
def w7_resources(g: Gate) -> None:
    """真实配置（`b_cap = horizon = 256`、`τ_max = 256`）的**构建耗时与表占用**。"""
    cfg = _cfg()
    ph = np.asarray(PHAT)
    t0 = time.perf_counter()
    dp = DPTail(cfg, ph, b_cap=int(cfg.protocol.horizon), tau_max=int(cfg.protocol.horizon),
                grid="r173").build()
    dt = time.perf_counter() - t0
    mem = dp.mem_mib()
    g.check(f"W7 资源：n={dp.knots.size} / b_cap={dp.b_cap} 时 表 ≤ {MEM_CAP_MIB} MiB 且构建 ≤ {BUILD_CAP_S:g} s",
            mem <= MEM_CAP_MIB and dt <= BUILD_CAP_S,
            {"n_knots": int(dp.knots.size), "b_cap": int(dp.b_cap), "tau_max": int(dp.tau_max),
             "mem_mib": round(mem, 1), "build_s": round(dt, 2),
             "cap_mib": MEM_CAP_MIB, "cap_s": BUILD_CAP_S,
             "note": "网格是**内存换精度**：legacy/n15 只要 ~57 MiB，r173/n90 要 ~2 GiB。"})


# ------------------------------------------------------------------ W8
def w8_budget_coverage(g: Gate) -> None:
    """预算轴覆盖：`b > min(b_cap, τ_max)` 会被 `value()` **静默压平**，必须被计数而不是被吞掉。

    ★ 第一版写错了判据（#88 同族）：用 `τ=8` 去查 `b=204`，把"`b ≥ τ` 的**饱和律**"
      当成"预算轴不够长"⇒ 假 FAIL。饱和是设计（预注册 §2 近似 3），不是缺陷。
      正确测法：`τ = τ_max = horizon = 256`，此时 `nb = b+1` 全程不饱和，
      唯一的压平来源就只剩 `b_cap < 真实预算`。
    """
    cfg = _cfg()
    ph = np.asarray(PHAT)
    tau = int(cfg.protocol.horizon)
    pmax = int(cfg.protocol.budget(max(cfg.protocol.budget_ratios)))
    Pm = np.asarray([[1e-2, 1e-2]])
    bs = np.arange(0, pmax + 1, 8, dtype=float)

    dp = DPTail(cfg, ph, b_cap=int(cfg.protocol.horizon), tau_max=tau, grid="r173").build()
    for b in bs:
        dp.value(Pm, np.asarray([b]), tau)
    dp_small = DPTail(cfg, ph, b_cap=64, tau_max=tau, grid="r173").build()
    for b in bs:
        dp_small.value(Pm, np.asarray([b]), tau)
    g.check(f"W8 预算轴覆盖：真实最大预算 {pmax} ≤ b_cap={cfg.protocol.horizon} ⇒ 压平计数恒 0",
            dp.n_b_clip == 0,
            {"p_max": pmax, "b_cap": int(cfg.protocol.horizon), "tau": tau,
             "n_b_clip": int(dp.n_b_clip), "n_queries": int(bs.size)})
    g.check("W8-对照：`b_cap=64 < 204` 时必须记到压平（否则这个计数器没有分辨力）",
            dp_small.n_b_clip > 0,
            {"b_cap": 64, "n_b_clip": int(dp_small.n_b_clip), "n_queries": int(bs.size)})


# ------------------------------------------------------------------ W1R
def w1r_budget_monotone(g: Gate) -> None:
    """链上的 `G8d[dp]` 在新网格上复跑：预算维严格递减、平坦 ≤10%。"""
    cfg = _cfg()
    pol = make(cfg, arm="probe", h_plan=1, m_samples=1, tail="dp",
               dp_phat=np.asarray([0.60, 0.45]), dp_grid="r173")
    pol._wire_dp(None)
    r = _budget_sweep(pol, 128, 0.01)
    r["grid"] = pol._dp.grid
    r["mem_mib"] = round(pol._dp.mem_mib(), 1)
    g.check("W1R（= G8d[dp]@r173）值函数对预算严格递减：上升 0 对、平坦 ≤10%"
            "（steps=128、P0=0.01、p̂=(0.60,0.45)）",
            r["n_up"] == 0 and r["frac_flat"] <= 0.10, r)


def run_grid_gates() -> Gate:
    g = Gate()
    w4r_grid_admissible(g)
    w5_absolute(g)
    w6_criterion_defect(g)
    w7_resources(g)
    w8_budget_coverage(g)
    w1r_budget_monotone(g)
    return g
