# -*- coding: utf-8 -*-
"""`CW-R3.3`：把「槽」与「量子」分开的**可算论证** —— 探测到底买不买得起、在什么窗口下买得起。

预注册：`docs/research/CW-R3.3-槽与量子-预注册-v0.1.md`（写于 2026-10-26 06:3x，**最终面板跑之前**）

为什么是这一轮
--------------
R3.2 判「探测拿不到价值」，并把机制叙述成**「探测的代价是一个槽，不是一量子」**。
但那条叙述**没有量**：没人测过"一量子值多少""一次成功值多少""信息值多少"。
本轮把三样都测出来，再给一条**能被数据推翻的判据**：

    买得起 ⇔ (b − c_probe) · Δp̂_UB > p̂

推导（本轮的解析主件）
----------------------
DP 在每次发送处对 `p̂` **线性**（`S + p̂·V_s + (1−p̂)·V_f`），故
    增益 = Δp̂ · ΔV_succ · (受影响的发送次数) ≤ (b − c_probe) · Δp̂_UB · ΔV_succ
    代价 = 一个量子 = ΔV_q，而 `ΔV_q ≈ p̂·ΔV_succ`（`J-A` 实测，不是假设）
两边除 `ΔV_succ` ⇒ `(b−c_probe)·Δp̂_UB > p̂`。**槽不进这条判据**：
槽的作用只体现在 `UB` 与实测的差（`J-E` 的 slack）里。

三条纪律
--------
1. **零 episode、零训练、零参数更新**，纯 CPU；被测物一字节不动（只 `import`）。
2. 判词与否定条件见预注册 §4，**跑前写死**；`judge()` 只按表取，不许看结果改。
3. ★ 本轮**不定义** `price_t`：所有动作都吃一个槽，槽数由视界固定 ⇒ 槽没有独立可识别的边际
   （预注册 `§7-A1`）。凡引用"槽价"，必须先读这条。
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import time
from pathlib import Path

import numpy as np

from .config import mainline_config
from .posterior import p_succ_expected
from .r1_14_exactbase import _now_utc, _src_hash
from .r3_2_iv2u import SIGMA_JUDGE, build_engine  # 只复用引擎，不改它

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results" / "cw_v1"

PROTOCOL = "CW-POMDP-v1.1"
ROUND = "CW-R3.3"

#: 判据面板（预注册 §5）
TAU_LIST: tuple[int, ...] = (2, 3, 4, 5, 6)
B_MAX = 6
P_LIST: tuple[float, ...] = (0.005, 0.02, 0.05, 1.0)
P_JUDGE_MAX = 0.05
V_PRIOR = 1.0
N_M_JUDGE = 129
N_M_COARSE = 33

#: `Δp̂` 的敏感性阶梯
V_LIST: tuple[float, ...] = (1.0, 0.25, 0.05, 0.01)
SIG_LIST: tuple[float, ...] = (1e-3, 0.05, 0.2, 0.5)

#: 判据/容差（预注册 §4）
IV_NOISE = 1e-12
RATIO_TOL = 0.10
SLACK_TIGHT = 10.0
STATE_CAP = 6_000_000
SEC_CAP = 900.0

_SQRT2PI = math.sqrt(2.0 * math.pi)


# ==================================================================== Δp̂（零 DP）
def _gh(n: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.hermite_e.hermegauss(int(n))
    return nodes, weights / _SQRT2PI


def dphat_probe(v: float, sigma_csi: float, gh_n: int = 81) -> dict:
    """探**一个**用户、另一个停在先验时，**一次发送**的有效成功率期望提升（上界形态）。

    读数 `y = log h + ε`，`ε ~ N(0, σ_csi²)`；读数先验 `N(0, v)` ⇒ 后验 `N(g_p·y, v_p)`，
    `g_p = v/(v+σ²)`、`v_p = vσ²/(v+σ²)`。决策侧：`max(p̂(后验), p̂(先验))`。
    ⇒ `Δp̂ = E[max] − p̂`。**这是"探一个用户"能给出的最好情况**（它假设读数无量化、无延迟）。
    """
    nodes, w = _gh(gh_n)
    m = 0.0
    pbar = p_succ_expected(m, v, _CFG)
    si2 = float(sigma_csi) ** 2
    gp = v / (v + si2)
    vp = v * si2 / (v + si2)
    sd = math.sqrt(v)
    tot = 0.0
    for z, wi in zip(nodes, w):
        h_prior = m + sd * float(z)          # 真值的**先验**抽样点（不是任何真值字段）
        m_hat = m + gp * (h_prior - m)
        tot += float(wi) * max(p_succ_expected(m_hat, vp, _CFG), pbar)
    return {"v": float(v), "sigma_csi": float(sigma_csi), "p_hat": pbar,
            "v_post": vp, "dphat": float(tot - pbar)}


def dphat_grid(v: float, sigma_csi: float, n: int = 401) -> float:
    """`G4` 的独立实现：**矩形网格直接积分**（不用 Gauss-Hermite），返回 `Δp̂`。"""
    m = 0.0
    pbar = p_succ_expected(m, v, _CFG)
    si2 = float(sigma_csi) ** 2
    gp = v / (v + si2)
    vp = v * si2 / (v + si2)
    sd = math.sqrt(v)
    lo, hi = -8.0 * sd, 8.0 * sd
    dx = (hi - lo) / int(n)
    tot = 0.0
    mass = 0.0
    for k in range(int(n)):
        z = lo + (k + 0.5) * dx
        dens = math.exp(-0.5 * z * z) / _SQRT2PI
        mass += dens * dx
        # ★ 首版漏了 `sd`：后验均值是 `gp·h`，而 `h ~ N(0, v)` ⇒ 必须 `gp·sd·z`。
        #   漏掉后 `Δp̂` 与 `v` 无关（`v=0.05` 和 `v=1` 给出同一个数），当场暴露。
        tot += dens * dx * max(p_succ_expected(gp * sd * z, vp, _CFG), pbar)
    # 归一到截断质量（矩形中点法在 ±8σ 上的截断误差 ~1e-15）
    return float(tot / mass - pbar)


# ==================================================================== 面板
_CFG = mainline_config()


def build_trio(cfg, n_m: int):
    """三个臂：禁探测 / 免量子探测 / 付量子探测。返回 `(v_no, v_fr, v_pd, meta)`。"""
    t0 = time.perf_counter()
    out = []
    meta = {"n_m": int(n_m), "sec": {}}
    for name, kw in (("noprobe", {"probe_enabled": False}),
                     ("free", {"probe_cost": 0}),
                     ("paid", {"probe_cost": 1})):
        t1 = time.perf_counter()
        out.append(build_engine(cfg, SIGMA_JUDGE, n_m, **kw))
        meta["sec"][name] = round(time.perf_counter() - t1, 3)
        print(f"  [engine {name} n_m={n_m}] {meta['sec'][name]} s", flush=True)
    meta["sec"]["total"] = round(time.perf_counter() - t0, 3)
    return (*out, meta)


def panel_exact(cfg, n_m: int = N_M_JUDGE, tau_max: int = 5, tau6_nm: int = 0) -> dict:
    """精确面板：每一格给出 `ΔV_q`、`ΔV_succ`、`IV_free`、`IV_paid` 与解析上界。

    ★ `ΔV_succ` 取**根状态**一次成功 vs 失败的成本差（`V(P_f) − V(P_s)`，在同一 `(t−1,b)` 上）。
      `ΔV_q` 取 `V(b−1) − V(b)`（成本降幅 = 量子的价值）。
    """
    p = cfg.plant
    a2 = tuple(float(a) ** 2 for a in p.A)
    sw2 = tuple(float(s) ** 2 for s in p.sigma_w)
    sm2 = float(p.sigma_meas) ** 2
    rho = float(cfg.channel.rho_panel[0])
    rho2 = rho * rho
    infl = 1.0 - rho2
    v_hold = rho2 * V_PRIOR + infl
    v_no, v_fr, v_pd, meta = build_trio(cfg, n_m)

    cells: list[dict] = []
    # ★ τ≥6 在 `n_m=129` 上**撞 R3.2 引擎的 `STATE_CAP=6e6`**（实测 6.000010e6 ⇒ NaN）
    #   ⇒ 主环封顶 τ=5；τ=6 只能降分辨率另跑（`--tau6-nm`），且必须标注口径。
    tau_hi = min(int(tau_max), 5)
    for tau in range(2, tau_hi + 1):
        for b in range(1, min(int(tau), B_MAX) + 1):
            for P in P_LIST:
                a = v_no(tau, P, P, 0.0, V_PRIOR, 0.0, V_PRIOR, b)
                f = v_fr(tau, P, P, 0.0, V_PRIOR, 0.0, V_PRIOR, b)
                d = v_pd(tau, P, P, 0.0, V_PRIOR, 0.0, V_PRIOR, b)
                dq = (v_no(tau, P, P, 0.0, V_PRIOR, 0.0, V_PRIOR, b - 1) - a
                      if b >= 1 else float("nan"))
                g = P / (P + sm2)
                ps_ = a2[0] * (1.0 - g) * P + sw2[0]
                pf_ = a2[0] * P + sw2[0]
                dvs = (v_no(tau - 1, pf_, P, rho * 0.0, v_hold, rho * 0.0, v_hold, b)
                       - v_no(tau - 1, ps_, P, rho * 0.0, v_hold, rho * 0.0, v_hold, b))
                dpv = dphat_ref(V_PRIOR, SIGMA_JUDGE)
                dp = {"p_hat": p_succ_expected(0.0, V_PRIOR, _CFG), "dphat": dpv}
                cells.append({
                    "n_m": int(n_m), "tau": int(tau), "b": int(b), "P": float(P),
                    "sigma_csi": SIGMA_JUDGE, "v": V_PRIOR,
                    "V_noprobe": float(a), "V_free": float(f), "V_paid": float(d),
                    "IV_free": float(a - f), "IV_paid": float(a - d),
                    "dV_q": float(dq), "dV_succ": float(dvs),
                    "p_hat": dp["p_hat"], "dphat_ub": dp["dphat"],
                    "UB_free": float(b * dp["dphat"] * dvs),
                    "UB_paid": float((b - 1) * dp["dphat"] * dvs),
                })
        print(f"  [panel tau={tau}] {len(cells)} cells", flush=True)

    # ---- 阈值探针（可选）：`τ=6, b∈{5,6}` —— 判据预测 `b≥5` 才买得起。
    #      ★ `n_m=129` 撞 `STATE_CAP`（实测 6.000010e6，查询返回 NaN）⇒ 只能在 `n_m=65` 上跑，
    #        而 65 的 `snap` 步长 `0.125` 大于 `σ_csi=0.05` ⇒ 读数的分辨率先被网格吃掉，
    #        结论**偏向"探测没用"**（混淆项，必须在交付稿里写明）。
    if int(tau6_nm) > 0:
        v_no6, v_fr6, v_pd6, meta6 = build_trio(cfg, int(tau6_nm))
        for b in (5, 6):
            for P in (0.005, 0.02):
                a = v_no6(6, P, P, 0.0, V_PRIOR, 0.0, V_PRIOR, b)
                f = v_fr6(6, P, P, 0.0, V_PRIOR, 0.0, V_PRIOR, b)
                d = v_pd6(6, P, P, 0.0, V_PRIOR, 0.0, V_PRIOR, b)
                dq = v_no6(6, P, P, 0.0, V_PRIOR, 0.0, V_PRIOR, b - 1) - a
                g = P / (P + sm2)
                ps_ = a2[0] * (1.0 - g) * P + sw2[0]
                pf_ = a2[0] * P + sw2[0]
                dvs = (v_no6(5, pf_, P, 0.0, v_hold, 0.0, v_hold, b)
                       - v_no6(5, ps_, P, 0.0, v_hold, 0.0, v_hold, b))
                dpv = dphat_ref(V_PRIOR, SIGMA_JUDGE)
                cells.append({
                    "n_m": int(tau6_nm), "tau": 6, "b": int(b), "P": float(P),
                    "sigma_csi": SIGMA_JUDGE, "v": V_PRIOR, "coarse": True,
                    "V_noprobe": float(a), "V_free": float(f), "V_paid": float(d),
                    "IV_free": float(a - f), "IV_paid": float(a - d),
                    "dV_q": float(dq), "dV_succ": float(dvs),
                    "p_hat": p_succ_expected(0.0, V_PRIOR, _CFG), "dphat_ub": dpv,
                    "UB_free": float(b * dpv * dvs),
                    "UB_paid": float((b - 1) * dpv * dvs),
                })
        meta["n_states_tau6"] = {"noprobe": v_no6.n_states(), "free": v_fr6.n_states(),
                                 "paid": v_pd6.n_states()}
        meta["capped_tau6"] = bool(v_no6.capped() or v_fr6.capped() or v_pd6.capped())
        meta["sec"]["tau6"] = meta6["sec"]["total"]
        print(f"  [panel tau=6 coarse n_m={tau6_nm}] states={meta['n_states_tau6']} "
              f"capped={meta['capped_tau6']}", flush=True)
    meta["n_states"] = {"noprobe": v_no.n_states(), "free": v_fr.n_states(),
                        "paid": v_pd.n_states()}   # ★ 查询之后才取（R3.2 自伤第 1 条）
    meta["capped"] = bool(v_no.capped() or v_fr.capped() or v_pd.capped())
    return {"cells": cells, "meta": meta}


# ==================================================================== 判据表（零 DP）
def criterion_table() -> dict:
    """`Δp̂_UB` 的敏感性 + 阈值 `b* = 1 + p̂/Δp̂`（付量子）、`b*_free = p̂/Δp̂`（免量子）。"""
    rows = []
    for v in V_LIST:
        for sig in SIG_LIST:
            r = dphat_probe(v, sig)
            ref = dphat_ref(v, sig)
            dp = ref                       # ★ 用网格参考值（GH 系统性偏低 0.37%）
            pbar = r["p_hat"]
            rows.append({**r, "dphat": float(dp), "dphat_gh": float(r["dphat"]),
                         "b_star_paid": (1.0 + pbar / dp) if dp > 0 else None,
                         "b_star_free": (pbar / dp) if dp > 0 else None})
    return {"rows": rows}


# ==================================================================== 判词
def judge(exact: dict, crit: dict, gates: dict) -> dict:
    out: dict = {}
    # ★ 撞 `STATE_CAP` 的格会返回 `NaN`（预注册 §6：没算完就如实写，不许当 0）
    raw = exact["cells"]
    allc = [c for c in raw if all(isinstance(c[k], float) and math.isfinite(c[k])
                                 for k in ("IV_free", "IV_paid", "dV_q", "dV_succ"))]
    out["n_cells_raw"] = len(raw)
    out["n_cells_finite"] = len(allc)
    out["n_cells_dropped_nan"] = len(raw) - len(allc)
    cells = [c for c in allc if c["P"] <= P_JUDGE_MAX]

    # ---- J-A：一个量子 ≈ 一次发送的期望收益（只在 `b=1` 且 `ΔV_q>0` 的格上判）
    rows = [c for c in allc if c["b"] == 1 and c["dV_q"] > 0]
    ratios = [{"tau": c["tau"], "P": c["P"],
               "ratio": c["dV_q"] / (c["p_hat"] * c["dV_succ"])} for c in rows]
    ok = all(abs(r["ratio"] - 1.0) <= RATIO_TOL for r in ratios) and ratios
    out["J-A"] = "QUANTUM-EQUALS-ONE-SEND" if ok else "QUANTUM-NOT-ONE-SEND"
    out["J-A_ratios"] = ratios

    # ---- J-B：上界不许被破
    viol_free = [c for c in allc if c["IV_free"] > c["UB_free"] + 1e-15]
    viol_paid = [c for c in allc if c["IV_paid"] > c["UB_paid"] + 1e-15]
    out["J-B"] = "BOUND-VALID" if not (viol_free or viol_paid) else "BOUND-VIOLATED"
    out["J-B_viol_free"] = [{"tau": c["tau"], "b": c["b"], "P": c["P"],
                             "IV": c["IV_free"], "UB": c["UB_free"]} for c in viol_free]
    out["J-B_viol_paid"] = [{"tau": c["tau"], "b": c["b"], "P": c["P"],
                             "IV": c["IV_paid"], "UB": c["UB_paid"]} for c in viol_paid]

    # ---- J-C：证书方向（上界 < 代价 ⇒ 实测必须为 0）
    cert_cells = [c for c in allc if c["dV_q"] > 0 and c["UB_paid"] < c["dV_q"]]
    bad = [c for c in cert_cells if c["IV_paid"] > IV_NOISE]
    out["J-C"] = "CERTIFICATE-FAILED" if bad else "NO-PAID-PROBE-CERTIFIED"
    out["J-C_n_cert_cells"] = len(cert_cells)
    out["J-C_bad"] = [{"tau": c["tau"], "b": c["b"], "P": c["P"],
                       "IV_paid": c["IV_paid"], "UB_paid": c["UB_paid"],
                       "dV_q": c["dV_q"]} for c in bad]

    # ---- J-D：阈值 vs 面板（`v=1, σ=0.05` 行）
    row = next(r for r in crit["rows"] if r["v"] == V_PRIOR
               and abs(r["sigma_csi"] - SIGMA_JUDGE) < 1e-15)
    b_star = 1.0 + row["p_hat"] / row["dphat"]
    out["J-D_b_star"] = float(b_star)
    out["J-D_b_star_msg"] = f"b* = {b_star:.3f} ⇒ 付量子臂需 b ≥ {math.ceil(b_star)}"
    meas_pos = [c for c in allc if c["IV_paid"] > IV_NOISE]
    out["J-D_measured_paid_positive"] = [{"tau": c["tau"], "b": c["b"], "P": c["P"],
                                          "IV_paid": c["IV_paid"],
                                          "UB_paid": c["UB_paid"], "dV_q": c["dV_q"]}
                                         for c in meas_pos]
    hi = [c for c in allc if c["b"] >= 5]
    coarse = [c for c in hi if c.get("coarse")]
    out["J-D_n_cells_b_ge_5"] = len(hi)
    out["J-D_n_cells_b_ge_5_coarse"] = len(coarse)
    out["J-D_coarse_positive"] = [{"tau": c["tau"], "b": c["b"], "P": c["P"],
                                   "IV_paid": c["IV_paid"],
                                   "UB_over_price": (c["UB_paid"] / c["dV_q"]
                                                     if c["dV_q"] > 0 else None)}
                                  for c in coarse]
    verdict_hi = [c for c in hi if c["IV_paid"] > IV_NOISE]
    if any(c["IV_paid"] > IV_NOISE for c in hi if not c.get("coarse")):
        out["J-D"] = "THRESHOLD-CONFIRMED-BY-DP"
    elif verdict_hi:
        out["J-D"] = "THRESHOLD-HINT-AT-COARSE-GRID"
    elif hi and all(not c.get("coarse") for c in hi):
        out["J-D"] = "THRESHOLD-CONFIRMED-NEGATIVE-AT-5" if b_star > 5.0 else "THRESHOLD-REFUTED-AT-5"
    elif coarse:
        out["J-D"] = "THRESHOLD-UNCONFIRMED-AT-COARSE-GRID"
    elif b_star > 4.0:
        out["J-D"] = "THRESHOLD-UNVERIFIED-AT-JUDGE-RES"
    else:
        out["J-D"] = "THRESHOLD-INSIDE-PANEL"

    # ---- J-E：上界紧度（`IV_free/UB_free`，只看 `UB_free>0` 的格）
    slacks = [c["UB_free"] / c["IV_free"] for c in allc
              if c["IV_free"] > IV_NOISE and c["UB_free"] > 0]
    med = float(np.median(slacks)) if slacks else None
    out["J-E_median_slack"] = med
    out["J-E_n"] = len(slacks)
    out["J-E"] = ("BOUND-TOO-LOOSE" if (med is not None and med > SLACK_TIGHT)
                  else "BOUND-TIGHT")
    out["gates_ok"] = all(v.get("pass", True) for v in gates.values() if isinstance(v, dict))
    return out


# ==================================================================== 闸
def gate_g1_no_truth() -> dict:
    """`G1`：AST 钉住 —— 本文件不读任何真值字段（`h_true`/`cost_true*`/`w_realized`）。"""
    src = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    banned = {"h_true", "cost_true_by_user", "cost_true", "w_realized", "x_true"}
    hits = [n.attr for n in ast.walk(tree)
            if isinstance(n, ast.Attribute) and n.attr in banned]
    hits += [n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id in banned]
    return {"banned": sorted(banned), "hits": hits, "pass": not hits}


def dphat_ref(v: float, sigma_csi: float) -> float:
    """`Δp̂` 的**参考值**：矩形网格 1601 点（`G2` 实测其 801→1601 的漂移 `3.6e-8`）。

    ★ 为什么不用 Gauss-Hermite：被积函数在 `max(·)` 的拐点处不可导 ⇒ GH 慢收敛且**系统性偏低**
      （`GH@321` 比网格低 `0.37%`）。上界必须取大者，故用网格。
    """
    return dphat_grid(v, sigma_csi, 1601)


def gate_g2_dphat_two_ways() -> dict:
    """`G2`：`Δp̂` 的**数值收敛**（GH 点数阶梯 vs 矩形网格阶梯）。

    ★ 跑前设的 `1e-4` 容差是错的：被积函数 `max(p̂(后验), p̂(先验))` 在**拐点处不可导**，
      Gauss-Hermite 与中点法都以 `~1%` 的慢速收敛（首跑实测两法差 `1.46%`）。
      改正为：**两条阶梯各自收敛**，取极限值之差当**数值不确定度**，并把该不确定度
      传进 `b*`（`b*=1+p̂/Δp̂`）后检查判词是否仍在同一侧。
    """
    ladder = []
    for v in V_LIST:
        for sig in SIG_LIST:
            gh = [dphat_probe(v, sig, n)["dphat"] for n in (41, 81, 161, 321)]
            gr = [dphat_grid(v, sig, n) for n in (401, 801, 1601)]
            ladder.append({"v": v, "sigma_csi": sig, "gh": gh, "grid": gr,
                           "gh_spread": abs(gh[-1] - gh[-2]),
                           "grid_spread": abs(gr[-1] - gr[-2]),
                           "cross": abs(gh[-1] - gr[-1])})
    band = max(r["grid_spread"] for r in ladder)
    row = next(r for r in ladder if r["v"] == V_PRIOR and abs(r["sigma_csi"] - SIGMA_JUDGE) < 1e-15)
    dp = row["grid"][-1]
    gh_bias = abs(row["gh"][-1] - dp) / dp
    b_lo = 1.0 + 0.5 / (dp + band)
    b_hi = 1.0 + 0.5 / max(1e-12, dp - band)
    robust = math.ceil(b_hi) == math.ceil(b_lo) == 5
    return {"ladder": ladder, "grid_band_abs": float(band),
            "grid_band_reldiff": float(band / dp),
            "gh_bias_at_321": float(gh_bias),
            "dphat_final": float(dp), "b_star_band": [float(b_lo), float(b_hi)],
            "tol_grid": 1e-4, "tol_gh_bias": 1e-2, "verdict_robust": bool(robust),
            "pass": bool(band / dp <= 1e-4 and gh_bias <= 1e-2 and robust)}


def gate_g3_p_hat_identity() -> dict:
    """`G3`：`p_succ_expected(0,v)` 与独立积分一致（`m=0` 时应恒为 0.5，对称）。"""
    rows = []
    worst = 0.0
    nodes, w = _gh(201)
    for v in (1.0, 0.25, 0.05):
        a = p_succ_expected(0.0, v, _CFG)
        sd = math.sqrt(v)
        tot = sum(float(wi) * 1.0 / (1.0 + math.exp(-2.0 * (sd * float(z))))
                  for z, wi in zip(nodes, w))
        d = abs(a - tot)
        worst = max(worst, d)
        rows.append({"v": v, "posterior": a, "independent": float(tot), "abs": d})
    return {"rows": rows, "max_abs": float(worst), "tol": 1e-9, "pass": bool(worst <= 1e-9)}


def gate_g4_monotone(exact: dict) -> dict:
    """`G4`：活体单调性 ——（i）`V(b)` 随 `b` **非增**（多一个量子不会让成本变大）；
    （ii）`ΔV_q ≥ 0`；（iii）`Δp̂` 随 `v` 单调**不增**、随 `σ_csi` 单调**不增**。"""
    rows = []
    bad = []
    for c in exact["cells"]:
        if c["b"] >= 2:
            prev = next((x for x in exact["cells"] if x["tau"] == c["tau"]
                         and x["b"] == c["b"] - 1 and x["P"] == c["P"]), None)
            if prev is not None:
                ok = c["V_noprobe"] <= prev["V_noprobe"] + 1e-15
                rows.append({"tau": c["tau"], "b": c["b"], "P": c["P"], "ok": ok})
                if not ok:
                    bad.append(rows[-1])
    dv_bad = [{"tau": c["tau"], "b": c["b"], "P": c["P"], "dV_q": c["dV_q"]}
              for c in exact["cells"] if c["dV_q"] < -1e-15]
    dph = [dphat_probe(v, s)["dphat"] for v in V_LIST for s in (0.05,)]
    mono_v = all(dph[i] >= dph[i + 1] - 1e-15 for i in range(len(dph) - 1))
    dph2 = [dphat_probe(1.0, s)["dphat"] for s in (1e-3, 0.05, 0.2, 0.5)]
    mono_s = all(dph2[i] >= dph2[i + 1] - 1e-15 for i in range(len(dph2) - 1))
    return {"n_checked": len(rows), "bad": bad, "dV_q_negative": dv_bad,
            "dphat_by_v": dph, "mono_in_v": bool(mono_v),
            "dphat_by_sigma": dph2, "mono_in_sigma": bool(mono_s),
            "pass": bool(not bad and not dv_bad and mono_v and mono_s)}


def gate_g5_mutation(exact: dict) -> dict:
    """`G5`：**判据的分辨力**（变异测试）——把 `Δp̂` 置 `0` 必须判"全都不买"，
    置成 `2`（远超物理可能）必须判"全都买"。若变异后结论不变 ⇒ 判据是恒真/恒假，没有分辨力。"""
    cells = [c for c in exact["cells"] if c["b"] >= 2]
    def pays(dp: float) -> list[bool]:
        return [((c["b"] - 1) * dp * c["dV_succ"] > c["dV_q"]) for c in cells]
    lo = pays(0.0)
    hi = pays(2.0)
    return {"n_cells": len(cells),
            "zero_says_buy": int(sum(lo)), "huge_says_buy": int(sum(hi)),
            "pass": bool(sum(lo) == 0 and sum(hi) == len(cells))}


def gate_g6_negative_control(cfg) -> dict:
    """`G6`：**活体阴性对照** —— 探测读数被丢弃（`drop_obs=True`）时，
    探测动作**退化成一个纯占槽的 HOLD** ⇒ `V_noprobe − V_drop` 必须**逐位 0**；
    而 `V_noprobe − V_free > 0` 必须**严格为正**（增益真的来自读数）。

    ★ 首版把两个臂写反了（`V_free − V_drop`），跑出 `−4.79e-5` 当成"对照失败" —— 判据缺陷，非引擎缺陷。
    ★ 同时记一条 R3.2 引擎的**辅助函数**缺陷：`build_engine(...).first()` 的 `PROBE` 分支
      **不认 `drop_obs`**（照样加读数），所以 `first()` 在 `drop_obs` 臂上是错的；`V()` 不受影响。
      本轮不改被测物，只记录（`G6.first_bug`）。
    """
    v_no = build_engine(cfg, SIGMA_JUDGE, N_M_COARSE, probe_enabled=False)
    v_fr = build_engine(cfg, SIGMA_JUDGE, N_M_COARSE, probe_cost=0)
    v_dr = build_engine(cfg, SIGMA_JUDGE, N_M_COARSE, probe_cost=0, drop_obs=True)
    rows = []
    worst_drop = 0.0
    best_gain = 0.0
    for tau in (3, 4):
        for b in (1, 2):
            for P in (0.005, 0.02):
                args = (tau, P, P, 0.0, V_PRIOR, 0.0, V_PRIOR, b)
                a = v_no(*args)
                f = v_fr(*args)
                d = v_dr(*args)
                rows.append({"tau": tau, "b": b, "P": P,
                             "IV_free": a - f, "IV_drop": a - d})
                worst_drop = max(worst_drop, abs(a - d))
                best_gain = max(best_gain, a - f)
    fb = v_dr.first(4, 0.005, 0.005, 0.0, V_PRIOR, 0.0, V_PRIOR, 1)
    ff = v_fr.first(4, 0.005, 0.005, 0.0, V_PRIOR, 0.0, V_PRIOR, 1)
    return {"rows": rows, "max_abs_drop_effect": float(worst_drop),
            "max_positive_gain": float(best_gain),
            "first_bug": {"drop_PROBE_1": fb.get("PROBE_1"), "free_PROBE_1": ff.get("PROBE_1"),
                          "same": bool(fb.get("PROBE_1") == ff.get("PROBE_1"))},
            "tol_drop": 1e-12, "pass": bool(worst_drop <= 1e-12 and best_gain > 1e-12)}


def gate_g7_b_star_arithmetic(crit: dict) -> dict:
    """`G7`：`b*` 的算术复算（表里值 vs 现算），防止手抄错。"""
    worst = 0.0
    for r in crit["rows"]:
        recomputed = 1.0 + r["p_hat"] / r["dphat"]
        worst = max(worst, abs(recomputed - r["b_star_paid"]))
    return {"max_abs": float(worst), "tol": 1e-12, "pass": bool(worst <= 1e-12)}


def gate_g8_r32_reconcile(cfg) -> dict:
    """`G8`：与 R3.2 判据单元对账（`τ=4,b=1,P=0.005,n_m=129` 的 `IV_free`）。
    R3.2 报 `4.600e-5`；本轮同引擎同格必须同量级（相对差 `≤ 25%`，`n_m` 同为 129 ⇒ 应逐位近同）。"""
    v_no = build_engine(cfg, SIGMA_JUDGE, N_M_JUDGE, probe_enabled=False)
    v_fr = build_engine(cfg, SIGMA_JUDGE, N_M_JUDGE, probe_cost=0)
    a = v_no(4, 0.005, 0.005, 0.0, 1.0, 0.0, 1.0, 1)
    f = v_fr(4, 0.005, 0.005, 0.0, 1.0, 0.0, 1.0, 1)
    got = a - f
    ref = 4.600e-5
    rel = abs(got - ref) / ref
    return {"got": float(got), "r3_2_ref": ref, "rel": float(rel), "tol": 0.25,
            "pass": bool(rel <= 0.25)}


# ==================================================================== 主流程
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau-max", type=int, default=6)
    ap.add_argument("--n-m", type=int, default=N_M_JUDGE)
    ap.add_argument("--tau6-nm", type=int, default=0, help="τ=6 粗网格档位（0=不跑）")
    ap.add_argument("--skip-gates", action="store_true")
    ap.add_argument("--ctrl-only", action="store_true", help="只跑闸，不跑面板")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args(argv)

    cfg = mainline_config()
    t_all = time.perf_counter()
    stamp = _now_utc()
    src_hash = _src_hash()
    rid = f"cw_r3_3_slot_{stamp}_{src_hash}"
    outdir = Path(args.out) if args.out else (RESULTS / rid)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"[RUN] {ROUND} protocol={PROTOCOL} out={outdir}", flush=True)

    gates: dict = {}
    if not args.ctrl_only:
        print("[1/3] panel_exact ...", flush=True)
        exact = panel_exact(cfg, args.n_m, args.tau_max, args.tau6_nm)
    else:
        exact = {"cells": [], "meta": {}}

    print("[2/3] criterion_table ...", flush=True)
    crit = criterion_table()

    print("[3/3] gates ...", flush=True)
    gates["G1_no_truth"] = gate_g1_no_truth()
    gates["G2_dphat"] = gate_g2_dphat_two_ways()
    gates["G3_phat"] = gate_g3_p_hat_identity()
    gates["G4_monotone"] = gate_g4_monotone(exact) if exact["cells"] else {"pass": True, "skip": True}
    gates["G5_mutation"] = gate_g5_mutation(exact) if exact["cells"] else {"pass": True, "skip": True}
    gates["G6_negctrl"] = gate_g6_negative_control(cfg)
    gates["G7_arith"] = gate_g7_b_star_arithmetic(crit)
    gates["G8_r32"] = gate_g8_r32_reconcile(cfg)
    n_fail = sum(1 for v in gates.values() if isinstance(v, dict) and v.get("pass") is False)

    jd = judge(exact, crit, gates) if exact["cells"] else {}
    payload = {
        "round": ROUND, "protocol": PROTOCOL, "run_id": rid,
        "utc": stamp, "local": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "src_hash": src_hash, "args": vars(args),
        "config": {"plant": {"A": list(cfg.plant.A), "sigma_w": list(cfg.plant.sigma_w),
                             "sigma_meas": cfg.plant.sigma_meas, "q_cost": cfg.plant.q_cost},
                   "channel": {"rho_panel": list(cfg.channel.rho_panel),
                               "kappa": cfg.channel.kappa, "log_h_star": cfg.channel.log_h_star,
                               "ack_mode": cfg.channel.ack_mode}},
        "exact": exact, "criterion": crit, "gates": gates,
        "n_fail": n_fail, "judge": jd,
        "sec": round(time.perf_counter() - t_all, 2),
    }
    (outdir / "r33.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                                     encoding="utf-8")
    print(f"\n[JUDGE] {json.dumps(jd, ensure_ascii=False, indent=1)}", flush=True)
    print(f"\n[GATES] n_fail={n_fail} / {len(gates)}", flush=True)
    for k, v in gates.items():
        if isinstance(v, dict) and v.get("pass") is False:
            print(f"  ! FAIL {k}: {json.dumps(v, ensure_ascii=False)[:400]}", flush=True)
    print(f"[DONE] {payload['sec']} s -> {outdir}", flush=True)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
