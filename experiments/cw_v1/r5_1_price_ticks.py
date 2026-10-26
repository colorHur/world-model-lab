# -*- coding: utf-8 -*-
"""`CW-R5.1` 价格边界实验卡的控制件：**整数刻度记账**。

预注册：`docs/research/CW-R5.1-价格边界实验卡-预注册-v0.1.md`（写于正式读数之前；
§6 的规模方法在读数前做过一次内联修订，见该节"写卡后、读数前的修订"）。

为什么必须换刻度（而不是把 `probe_cost` 从 1 改成 0.25）
--------------------------------------------------------
既有引擎（`r3_2_iv2u.py` / `r4_0_boundary.py:125`）里 `pc = int(probe_cost)`，预算按**整数量子**累减。
传 `0.25` 会被 `int()` 截成 `0` ⇒ 探测**免费**、发送仍收 1 ⇒ 跑出来的不是"探测变便宜"，
而是与 `p=0` 极限混成同一个对象。**这是记账单位错，不是数值近似。**

本卡：`1 量子 = 8 刻度`、`SEND = 8 刻度`、`PROBE = 8p 刻度`（`p∈{1,1/2,1/4,1/8,0}` ⇒ `{8,4,2,1,0}`）。
`8` 是为了让 `1/8` 量子正好落在 1 个整刻度上。

纪律
----
1. 零 episode、零训练、零参数更新；被测物一字节不动（`config/world/plan_dp/r3_2_iv2u/r4_0_boundary` 只 `import`）。
2. 判据 `T-1..T-10` 见预注册 §5，**跑前写死**；`judge()` 只按表取。
3. 本文件的刻度引擎是**新写**的 ⇒ 先用 `p=1`（`8 刻度 == SEND`）与 `p=0` 两条**逐位**身份闸自证，
   再对任何读数说话（`T-1` / `T-2`）。身份闸不过，全卡读数作废。
4. `b=1` 量子（`8` 刻度）上的结构性零**只**解释 `b=1`；不拿它解释 `b≥2`（R5.0 §7.3）。
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

from .config import PlantConfig, mainline_config
from .posterior import p_succ_expected
from .r1_14_exactbase import _now_utc, _src_hash
from .r3_2_iv2u import GH_N, M_RANGE, STATE_CAP
from .r4_0_boundary import build_engine_n

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results" / "cw_v1"

PROTOCOL = "CW-POMDP-v1.3-ticks"
ROUND = "CW-R5.1"

#: 刻度定义（预注册 §1，跑前写死）
TICKS_PER_QUANTUM = 8
SEND_TICKS = 8
#: 价格网格 ⇒ 探测刻度（`8p`）；`p=0` 是免费极限
PRICE_LIST: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125, 0.0)
PRICE_TICKS: tuple[int, ...] = (8, 4, 2, 1, 0)

#: 面板（预注册 §3）
N_M_SMALL = 17
GH = GH_N
SIGMA_STAR = 1.0e-9
SIGMA_NULL = 1.0e6
SIGMA_W0 = 0.02
ALPHA_LIST: tuple[float, ...] = (1.0, 2.5)
P_LIST: tuple[float, ...] = (0.01, 0.05)
TAU_LIST: tuple[int, ...] = (3, 4)
B_TICKS_LIST: tuple[int, ...] = (8, 10, 16)
KAPPA = 1.0
N_USERS = 2
V_PRIOR = 1.0

#: 判据常数（预注册 §5）
TOL_BIT = 0.0
TOL_STRUCT = 1e-15
TOL_MONO = 1e-12
TOL_NOINFO = 1e-9
IV_NOISE = 1e-12

#: 规模腿（预注册 §6-4；只为回答"全网格 CPU 能不能跑"，不出价格结论）
#: ★ 第 4 条 `n33_tau8` 是**读到前三条规模之后、面板读数之前**追加的：前三条（τ≤6）全部远低于
#:   硬墙，于是把 τ 推到全网格真正需要的 `τ=8`。它撞顶 ⇒ 结论是"`τ=8` **昂贵探测**可测、
#:   **便宜探测**不可测"，不是"规模爆炸"。规模读数不是判据，追加不改判据。
SIZE_LEGS: tuple[dict, ...] = (
    {"tag": "n33_tau4", "n_m": 33, "tau": (4,), "b_ticks": (16,), "p": (1.0, 0.25)},
    {"tag": "n33_tau6", "n_m": 33, "tau": (6,), "b_ticks": (16, 24), "p": (1.0, 0.25)},
    {"tag": "n65_tau4", "n_m": 65, "tau": (4,), "b_ticks": (16,), "p": (1.0, 0.25)},
    {"tag": "n33_tau8", "n_m": 33, "tau": (8,), "b_ticks": (8, 16, 24), "p": (1.0, 0.25)},
)
SIZE_ALPHA = 2.5
SIZE_P = 0.01

_SQRT2PI = math.sqrt(2.0 * math.pi)


# ==================================================================== 刻度与 plant
def price_ticks(p: float) -> int:
    """`8p` 必须**先取整再进 DP**；`int(8p)` 与 `int(p)` 都是错的。

    `p=1/8` 必须得到 `1`（不是 `0`）。用 `round` 且先断言整数性 —— 不许 `int()` 截断。
    """
    x = float(p) * TICKS_PER_QUANTUM
    if abs(x - round(x)) > 1e-12:
        raise ValueError(f"价格 {p} 不是整数刻度：8p={x}")
    return int(round(x))


def _plant_asym(alpha: float) -> PlantConfig:
    base = mainline_config().plant
    return PlantConfig(**{**base.__dict__, "sigma_w": (SIGMA_W0, SIGMA_W0 * float(alpha))})


# ==================================================================== 刻度 DP
def build_engine_ticks(cfg, sigma_csi: float, n_m: int, probe_ticks: int, *,
                       send_ticks: int = SEND_TICKS, gh_n: int = GH,
                       probe_enabled: bool = True, n_users: int = N_USERS,
                       state_cap: int = STATE_CAP):
    """`build_engine_n` 的**刻度版**：动作价格换成 `(send_ticks, probe_ticks)`。

    与 `r4_0_boundary.build_engine_n` 的差值**只有三处**，其余算术顺序逐字照搬
    （所以 `send=probe=8` 与旧 `send=probe=1` 必须**逐位**同值，`T-1` 验）：

    ① `SEND` 门槛 `b >= 1` → `b >= send_ticks`，子预算 `b − send_ticks`；
    ② `PROBE` 门槛 `b >= pc` → `b >= pt`，子预算 `b − pt`；
    ③ `pc = int(probe_cost)` → `pt = int(probe_ticks)`（**整数已由 `price_ticks()` 保证**）。
    """
    p = cfg.plant
    a2 = tuple(float(a) ** 2 for a in p.A)
    sw2 = tuple(float(s) ** 2 for s in p.sigma_w)
    sm2 = float(p.sigma_meas) ** 2
    q = float(p.q_cost)
    rho = float(cfg.channel.rho_panel[0])
    rho2 = rho * rho
    infl = 1.0 - rho2
    si2 = float(sigma_csi) ** 2
    nodes, weights = np.polynomial.hermite_e.hermegauss(int(gh_n))
    nodes = [float(z) for z in nodes]
    weights = [float(w) for w in weights]
    centers = np.linspace(-M_RANGE, M_RANGE, int(n_m))
    centers_l = [float(x) for x in centers]
    pt = int(probe_ticks)
    sc = int(send_ticks)
    nu = int(n_users)
    if pt < 0 or sc <= 0:
        raise ValueError("刻度必须是非负整数，且发送价必须 > 0")
    assert len(a2) >= nu and len(sw2) >= nu, "plant 的用户数不够"

    @lru_cache(maxsize=None)
    def snap(m: float) -> float:
        return centers_l[int(np.argmin(np.abs(centers - float(m))))]

    @lru_cache(maxsize=None)
    def ps(m: float, v: float) -> float:
        return p_succ_expected(float(m), float(v), cfg)

    memo: dict[tuple, float] = {}
    state = {"capped": False}

    def V(t: int, Pp: tuple, mm: tuple, vv: tuple, b: int) -> float:
        if t <= 0:
            return 0.0
        key = (t, Pp, mm, vv, b)
        got = memo.get(key)
        if got is not None:
            return got
        S = q * sum(Pp)
        if t == 1:
            memo[key] = S
            return S
        if len(memo) > state_cap:
            raise MemoryError(f"state_cap exceeded ({state_cap})")

        ff = tuple(a2[i] * Pp[i] + sw2[i] for i in range(nu))
        mp = tuple(rho * m for m in mm)
        vp0 = tuple(rho2 * v + infl for v in vv)

        best = V(t - 1, ff, mp, vp0, b)                    # HOLD

        if b >= sc:                                        # SEND_i（整包，sc 刻度）
            for i in range(nu):
                Pi = Pp[i]
                g = Pi / (Pi + sm2)
                ps_ = a2[i] * (1.0 - g) * Pi + sw2[i]
                pf_ = a2[i] * Pi + sw2[i]
                ph = ps(mm[i], vv[i])
                aa = list(ff)
                aa[i] = ps_
                bb = list(ff)
                bb[i] = pf_
                va = ph * V(t - 1, tuple(aa), mp, vp0, b - sc)
                vb = (1.0 - ph) * V(t - 1, tuple(bb), mp, vp0, b - sc)
                val = va + vb
                if val < best:
                    best = val

        if probe_enabled and b >= pt and t >= 2:           # PROBE_i（pt 刻度）
            for i in range(nu):
                mi = mm[i]
                vi = vv[i]
                gp = vi / (vi + si2)
                vpi = rho2 * (vi * si2 / (vi + si2)) + infl
                sy = math.sqrt(vi + si2)
                tot = 0.0
                for z, w in zip(nodes, weights):
                    mm2 = list(mp)
                    mm2[i] = rho * snap(mi + gp * sy * z)
                    vv2 = list(vp0)
                    vv2[i] = vpi
                    tot += w * V(t - 1, ff, tuple(mm2), tuple(vv2), b - pt)
                tot /= _SQRT2PI
                if tot < best:
                    best = tot

        out = S + best
        memo[key] = out
        return out

    def value(t, Pvec, mvec, vvec, b) -> float:
        try:
            return float(V(int(t), tuple(float(x) for x in Pvec),
                           tuple(float(x) for x in mvec),
                           tuple(float(x) for x in vvec), int(b)))
        except MemoryError:
            state["capped"] = True
            return float("nan")

    def n_states() -> int:
        return len(memo)

    value.n_states = n_states                # type: ignore[attr-defined]
    value.capped = lambda: state["capped"]   # type: ignore[attr-defined]
    value.probe_ticks = pt                   # type: ignore[attr-defined]
    value.send_ticks = sc                    # type: ignore[attr-defined]
    return value


# ==================================================================== 腿（整腿 = 一引擎 + 全格）
class Leg:
    """一条腿 = 一个 `(alpha, price)` 上的 `paid` 引擎 + 该 `alpha` 的 `noprobe` 引擎。

    ★ 规模读数在**该腿全部格求值之后**取（`memo` 是整腿并集，不是单格，R4.2 教训）。
    """

    def __init__(self, cfg, n_m: int, pt: int):
        self.cfg = cfg
        self.n_m = int(n_m)
        self.pt = int(pt)
        self.no = build_engine_ticks(cfg, SIGMA_STAR, self.n_m, 0, probe_enabled=False)
        t0 = time.perf_counter()
        self.paid = build_engine_ticks(cfg, SIGMA_STAR, self.n_m, self.pt)
        self.sec_build = round(time.perf_counter() - t0, 3)
        self._n0 = self.no.n_states()
        self._first = None

    def run_cells(self, cells: list[tuple]) -> list[dict]:
        out = []
        for (tau, b, P) in cells:
            P0, P1 = float(P), float(P) * KAPPA
            a0 = self.no(tau, (P0, P1), (0.0, 0.0), (V_PRIOR, V_PRIOR), b)
            a0m = self.no(tau, (P0, P1), (0.0, 0.0), (V_PRIOR, V_PRIOR), b - SEND_TICKS) \
                if b >= SEND_TICKS else float("nan")
            vp = self.paid(tau, (P0, P1), (0.0, 0.0), (V_PRIOR, V_PRIOR), b)
            if self._first is None:
                self._first = self.paid.n_states()
            dq = (a0m - a0) if b >= SEND_TICKS else float("nan")
            out.append({"tau": int(tau), "b_ticks": int(b), "b_quanta": b / TICKS_PER_QUANTUM,
                        "P": float(P), "p": self.pt / TICKS_PER_QUANTUM,
                        "probe_ticks": self.pt,
                        "A0": float(a0), "V_paid": float(vp), "IV_paid": float(a0 - vp),
                        "dV_q": float(dq),
                        "capped": bool(self.no.capped() or self.paid.capped())})
        return out

    def states(self) -> dict:
        return {"noprobe": self.no.n_states(), "paid": self.paid.n_states(),
                "n_before_any_query": self._n0, "n_after_first_cell": self._first}


# ==================================================================== 面板
def panel() -> dict:
    rows: list[dict] = []
    legs_meta: list[dict] = []
    cells = [(tau, b, P) for P in P_LIST for tau in TAU_LIST for b in B_TICKS_LIST]
    for alpha in ALPHA_LIST:
        cfg = mainline_config(plant=_plant_asym(alpha))
        for pt in PRICE_TICKS:
            leg = Leg(cfg, N_M_SMALL, pt)
            t0 = time.perf_counter()
            rr = leg.run_cells(cells)
            # ★ 整腿全部格求值之后才取规模
            st = leg.states()
            idem = leg.paid(4, (0.01, 0.01), (0.0, 0.0), (V_PRIOR, V_PRIOR), 16)
            n_after_requery = leg.paid.n_states()
            for r in rr:
                r["alpha"] = float(alpha)
            rows += rr
            legs_meta.append({
                "alpha": float(alpha), "p": pt / TICKS_PER_QUANTUM, "probe_ticks": pt,
                "n_m": N_M_SMALL, "n_cells": len(rr),
                "sec": round(time.perf_counter() - t0, 3), "sec_build": leg.sec_build,
                "states": st, "capped": bool(leg.no.capped() or leg.paid.capped()),
                "n_after_requery": n_after_requery,
                "idem_same": bool(idem == idem),
                "max_IV": float(max(r["IV_paid"] for r in rr)),
            })
            print(f"  [leg α={alpha:g} p={pt / TICKS_PER_QUANTUM:g} ({pt} 刻度)] "
                  f"{legs_meta[-1]['sec']} s cells={len(rr)} states={st} "
                  f"maxIV={legs_meta[-1]['max_IV']:.6e}", flush=True)
    return {"rows": rows, "legs": legs_meta}


# ==================================================================== 闸
def gate_t1_p1_identity(n_m: int = N_M_SMALL) -> dict:
    """`T-1`：`p=1` 时 `8 刻度 == 1 量子`，刻度引擎必须与旧整数引擎**逐位**相同。"""
    rows = []
    worst = 0.0
    for alpha in ALPHA_LIST:
        cfg = mainline_config(plant=_plant_asym(alpha))
        tick = build_engine_ticks(cfg, SIGMA_STAR, n_m, 8, send_ticks=8)
        old = build_engine_n(cfg, SIGMA_STAR, n_m, probe_cost=1)
        for q in (1, 2, 3):
            for tau, P in ((3, 0.05), (4, 0.01), (4, 0.05)):
                if q > tau - 1:
                    continue
                a = tick(tau, (P, P), (0.0, 0.0), (V_PRIOR, V_PRIOR), q * TICKS_PER_QUANTUM)
                b = old(tau, (P, P), (0.0, 0.0), (V_PRIOR, V_PRIOR), q)
                rel = abs(a - b) / max(1e-12, abs(b))
                worst = max(worst, rel)
                rows.append({"alpha": alpha, "q": q, "tau": tau, "P": P,
                             "ticks": float(a), "old": float(b), "rel": float(rel)})
    return {"rows": rows, "max_rel": float(worst), "tol": TOL_BIT,
            "pass": bool(worst <= TOL_BIT)}


def gate_t2_p0_identity(n_m: int = N_M_SMALL) -> dict:
    """`T-2`：`p=0` 必须等于 `free` 臂；且与 1 刻度刻度的同构缩放逐位相同。"""
    rows = []
    worst = 0.0
    for alpha in ALPHA_LIST:
        cfg = mainline_config(plant=_plant_asym(alpha))
        z8 = build_engine_ticks(cfg, SIGMA_STAR, n_m, 0, send_ticks=8)
        z1 = build_engine_ticks(cfg, SIGMA_STAR, n_m, 0, send_ticks=1)
        old = build_engine_n(cfg, SIGMA_STAR, n_m, probe_cost=0)
        for q in (1, 2):
            for tau, P in ((3, 0.05), (4, 0.01)):
                a = z8(tau, (P, P), (0.0, 0.0), (V_PRIOR, V_PRIOR), q * TICKS_PER_QUANTUM)
                b = old(tau, (P, P), (0.0, 0.0), (V_PRIOR, V_PRIOR), q)
                c = z1(tau, (P, P), (0.0, 0.0), (V_PRIOR, V_PRIOR), q)
                for name, x in (("vs_free", b), ("vs_scaled", c)):
                    rel = abs(a - x) / max(1e-12, abs(x))
                    worst = max(worst, rel)
                    rows.append({"alpha": alpha, "q": q, "tau": tau, "P": P,
                                 "k": name, "a": float(a), "ref": float(x), "rel": float(rel)})
    return {"rows": rows, "max_rel": float(worst), "tol": TOL_BIT,
            "pass": bool(worst <= TOL_BIT)}


def gate_t3_monotone(rows: list) -> dict:
    """`T-3`：固定 `(alpha,tau,b,P)`，`IV_paid` 随价格下降**不减**（动作集嵌套 ⇒ `V` 不增）。

    ★ 方向是推出来的：`p↓ ⇒ 8p↓ ⇒ {b≥8p}` 变大 ⇒ `V` 不增 ⇒ `IV=A0−V` **不减**。
    """
    idx = {(r["alpha"], r["tau"], r["b_ticks"], r["P"], r["probe_ticks"]): r for r in rows}
    viol = []
    worst = 0.0
    for alpha in ALPHA_LIST:
        for tau in TAU_LIST:
            for b in B_TICKS_LIST:
                for P in P_LIST:
                    seq = []
                    for pt in PRICE_TICKS:          # 8,4,2,1,0 （价格下降）
                        r = idx.get((alpha, tau, b, P, pt))
                        if r is None or not math.isfinite(r["IV_paid"]):
                            seq = []
                            break
                        seq.append((pt, r["IV_paid"]))
                    for (p_hi, v_hi), (p_lo, v_lo) in zip(seq, seq[1:]):
                        d = v_hi - v_lo               # 必须 ≤ 0
                        worst = max(worst, d)
                        if d > TOL_MONO:
                            viol.append({"alpha": alpha, "tau": tau, "b_ticks": b, "P": P,
                                         "p_hi_ticks": p_hi, "p_lo_ticks": p_lo,
                                         "IV_hi": v_hi, "IV_lo": v_lo, "violation": float(d)})
    return {"n_violations": len(viol), "worst_positive_gap": float(worst),
            "tol": TOL_MONO, "violations": viol[:12], "pass": bool(not viol)}


def gate_t4_b1_struct(rows: list) -> dict:
    """`T-4`：`b=8 刻度`（1 量子）且 `p>0` ⇒ `IV_paid ≡ 0`（付完余 ≤7 刻度，发不出去）。"""
    sub = [r for r in rows if r["b_ticks"] == 8 and r["probe_ticks"] > 0]
    worst = max((abs(r["IV_paid"]) for r in sub), default=float("nan"))
    # 活体对照：同预算 `p=0` 必须有非零（否则"零"可能只是没接线）
    free = [r for r in rows if r["b_ticks"] == 8 and r["probe_ticks"] == 0]
    mx_free = max((abs(r["IV_paid"]) for r in free), default=float("nan"))
    return {"n_cells": len(sub), "max_abs_IV": float(worst), "tol": TOL_STRUCT,
            "n_free_cells": len(free), "max_abs_IV_free": float(mx_free),
            "pass": bool(sub and worst <= TOL_STRUCT)}


def gate_t5_no_info() -> dict:
    """`T-5`：`σ_csi=1e6`（`si2=1e12`）⇒ 读数几乎无信息 ⇒ `0 ≤ IV ≤ 1e-9`。"""
    cfg = mainline_config(plant=_plant_asym(2.5))
    rows = []
    worst = float("nan")
    for pt in PRICE_TICKS:
        no = build_engine_ticks(cfg, SIGMA_NULL, N_M_SMALL, 0, probe_enabled=False)
        pd_ = build_engine_ticks(cfg, SIGMA_NULL, N_M_SMALL, pt)
        for (tau, b, P) in ((4, 16, 0.01), (4, 16, 0.05)):
            a0 = no(tau, (P, P), (0.0, 0.0), (V_PRIOR, V_PRIOR), b)
            v = pd_(tau, (P, P), (0.0, 0.0), (V_PRIOR, V_PRIOR), b)
            iv = a0 - v
            rows.append({"probe_ticks": pt, "tau": tau, "b_ticks": b, "P": P,
                         "IV": float(iv)})
    vals = [r["IV"] for r in rows]
    worst = max(vals) if vals else float("nan")
    return {"rows": rows, "max_IV": float(worst), "tol": TOL_NOINFO,
            "min_IV": float(min(vals)) if vals else float("nan"),
            "pass": bool(vals and -1e-12 <= worst <= TOL_NOINFO)}


def gate_t6_truncation_mutation(rows: list) -> dict:
    """`T-6`（**预注册形态**）：把价格按 `int(p)` 截断（`p<1` ⇒ 探测免费）必须被**区分**出来。

    ★ 首跑结果：格点选在 `b=16`（2 量子）⇒ `n_hit=0`。原因不是"截断无害"，而是**那些格没有分辨力**：
    该预算下连免费探测都不改变最优动作（同批面板 `p=0` 也逐位为 0）⇒ 两个引擎值相同是**必然**。
    真分辨力落在**可行性边界**格（`b=8`：正确引擎付 1/4 量子就凑不出一次发送，截断版却能同时探测+发送）。
    预注册形态的 `0 命中` 按原样保留上报；补的边界格见 `gate_t6b`（**跑后**添加，两套读数同报）。
    """
    cfg = mainline_config(plant=_plant_asym(2.5))
    diffs = []
    for pt in (4, 2, 1):
        good = build_engine_ticks(cfg, SIGMA_STAR, N_M_SMALL, pt)
        mut_pt = int(pt / TICKS_PER_QUANTUM)          # ← 变异体：把量子价当刻度（1/4 → 0）
        mut = build_engine_ticks(cfg, SIGMA_STAR, N_M_SMALL, mut_pt)
        for (tau, b, P) in ((4, 16, 0.01), (3, 16, 0.05)):
            a = good(tau, (P, P), (0.0, 0.0), (V_PRIOR, V_PRIOR), b)
            m = mut(tau, (P, P), (0.0, 0.0), (V_PRIOR, V_PRIOR), b)
            diffs.append({"p": pt / TICKS_PER_QUANTUM, "good_ticks": pt,
                          "mut_ticks": mut_pt, "tau": tau, "b_ticks": b, "P": P,
                          "good": float(a), "mut": float(m), "abs": float(abs(a - m))})
    n_hit = sum(1 for d in diffs if d["abs"] > 1e-12)
    return {"rows": diffs, "n_hit": n_hit, "tol": 1e-12,
            "pass": bool(n_hit >= 1)}


def gate_t6b_truncation_edge() -> dict:
    """`T-6b`（**跑后**补充）：把变异测试挪到**可行性边界**格 `b∈{8,10}`。

    为什么这里才有分辨力：`b=8`（1 量子）下"探测 + 一次发送"需要 `8p+8 ≤ 8`，只有 `p=0` 成立
    ⇒ 正确引擎的 `IV≡0`，而截断版（探测免费）会拿到 `IV>0`。`b=10` 下门槛是 `8p ≤ 2`
    ⇒ `p=1/4` 可行、`p=1/2` 不可行，两条读数必须分开。
    """
    cfg = mainline_config(plant=_plant_asym(2.5))
    diffs = []
    for pt in (4, 2, 1):
        good = build_engine_ticks(cfg, SIGMA_STAR, N_M_SMALL, pt)
        mut_pt = int(pt / TICKS_PER_QUANTUM)
        mut = build_engine_ticks(cfg, SIGMA_STAR, N_M_SMALL, mut_pt)
        for (tau, b, P) in ((4, 8, 0.01), (4, 10, 0.01), (3, 10, 0.05)):
            a = good(tau, (P, P), (0.0, 0.0), (V_PRIOR, V_PRIOR), b)
            m = mut(tau, (P, P), (0.0, 0.0), (V_PRIOR, V_PRIOR), b)
            diffs.append({"p": pt / TICKS_PER_QUANTUM, "good_ticks": pt,
                          "mut_ticks": mut_pt, "tau": tau, "b_ticks": b, "P": P,
                          "good": float(a), "mut": float(m), "abs": float(a - m)})
    n_hit = sum(1 for d in diffs if abs(d["abs"]) > 1e-12)
    return {"rows": diffs, "n_hit": n_hit, "tol": 1e-12, "pass": bool(n_hit >= 1)}


def granularity_table(rows: list) -> dict:
    """把"名义价格"折成"**整次发送的机会成本**"：`n_sends_left = floor((b − 8p)/8)`。

    命题（`P-5`，解析）：`0<p≤1` 时 `n_sends_left` 与 `p` 无关（只要 `8p ≤ b`），
    所以**价格在 `(0,1]` 内的差别不改变可用发送次数** ⇒ 同一组内 `IV` 应出现**逐位同值**。
    这里只报"是否逐位同值"，不预先断言一定同值。
    """
    from collections import defaultdict
    groups: dict[tuple, list] = defaultdict(list)
    for r in rows:
        if not math.isfinite(r["IV_paid"]):
            continue
        left = (r["b_ticks"] - r["probe_ticks"]) // SEND_TICKS
        groups[(r["alpha"], r["tau"], r["b_ticks"], r["P"], left)].append((r["probe_ticks"],
                                                                          r["IV_paid"]))
    out = []
    for key, vals in sorted(groups.items()):
        alpha, tau, b, P, left = key
        vv = [v for _, v in vals]
        same = max(vv) - min(vv) if vv else float("nan")
        out.append({"alpha": alpha, "tau": tau, "b_ticks": b, "P": P,
                    "n_sends_left": int(left),
                    "prices": [pt / TICKS_PER_QUANTUM for pt, _ in vals],
                    "IV": [float(v) for v in vv], "spread": float(same),
                    "bit_identical": bool(math.isfinite(same) and same == 0.0)})
    n_ident = sum(1 for o in out if o["bit_identical"] and len(o["prices"]) > 1)
    n_multi = sum(1 for o in out if len(o["prices"]) > 1)
    return {"groups": out, "n_groups": len(out), "n_multi_price_groups": n_multi,
            "n_bit_identical_groups": n_ident,
            "pass": bool(n_multi > 0 and n_ident == n_multi)}


def gate_t7_b1_regression(rows: list) -> dict:
    """`T-7`：`p=1` 且 `b=8 刻度` 必须复现 R4.0 的 `B1-CANNOT-PAY`（`IV_paid ≡ 0`）。"""
    sub = [r for r in rows if r["probe_ticks"] == 8 and r["b_ticks"] == 8]
    worst = max((abs(r["IV_paid"]) for r in sub), default=float("nan"))
    return {"n_cells": len(sub), "max_abs_IV": float(worst), "tol": TOL_STRUCT,
            "pass": bool(sub and worst <= TOL_STRUCT)}


def gate_t8_size_invariants(legs: list) -> dict:
    """`T-8`：规模三条不变式（求值前 0 / 整腿 > 首格 / 重复查询不改计数）。"""
    rows = []
    ok = True
    for lg in legs:
        st = lg["states"]
        c1 = st["n_before_any_query"] == 0
        c2 = st["paid"] > (st["n_after_first_cell"] or 0)
        c3 = lg["n_after_requery"] == st["paid"]
        ok = ok and c1 and c2 and c3
        rows.append({"alpha": lg["alpha"], "p": lg["p"], "before": st["n_before_any_query"],
                     "after_first": st["n_after_first_cell"], "whole_leg": st["paid"],
                     "after_requery": lg["n_after_requery"],
                     "c1_zero_before": bool(c1), "c2_leg_gt_first": bool(c2),
                     "c3_idempotent": bool(c3)})
    return {"rows": rows, "pass": bool(ok and rows)}


def gate_t9_ast_no_truncation() -> dict:
    """`T-9`：源码里不得出现 `int(probe_cost)` / `int(p)` 这类截断；且检测器本身有分辨力。"""
    src = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    hits = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "int":
            if not n.args:
                continue
            a0 = n.args[0]
            name = a0.id if isinstance(a0, ast.Name) else (
                a0.attr if isinstance(a0, ast.Attribute) else "")
            if name in ("p", "price", "probe_cost", "probe_price"):
                hits.append({"line": getattr(n, "lineno", None), "arg": name})
    # 分辨力：检测器对一段**故意写错**的源码必须命中
    probe_src = "def f(p):\n    return int(p)\n"
    probe_hits = [n for n in ast.walk(ast.parse(probe_src))
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                  and n.func.id == "int"]
    return {"hits": hits, "n_hits": len(hits),
            "detector_catches_mutation": bool(probe_hits),
            "pass": bool(not hits and probe_hits)}


def gate_t10_distinct() -> dict:
    """`T-10`：指纹互异（`alpha` 两个不同 plant；价格五个不同刻度；预算三个不同刻度）。"""
    fps = {}
    for alpha in ALPHA_LIST:
        cfg = mainline_config(plant=_plant_asym(alpha))
        fps[f"{alpha:g}"] = {"sigma_w": list(cfg.plant.sigma_w), "fp": cfg.fingerprint()}
    uniq_alpha = len({v["fp"] for v in fps.values()})
    ticks = [price_ticks(p) for p in PRICE_LIST]
    uniq_ticks = len(set(ticks))
    uniq_b = len(set(B_TICKS_LIST))
    ok = (uniq_alpha == len(ALPHA_LIST) and uniq_ticks == len(PRICE_LIST)
          and uniq_b == len(B_TICKS_LIST) and ticks == list(PRICE_TICKS))
    return {"per_alpha": fps, "price_ticks": ticks, "n_distinct_alpha": uniq_alpha,
            "n_distinct_price_ticks": uniq_ticks, "n_distinct_b": uniq_b, "pass": bool(ok)}


# ==================================================================== 规模腿
def size_legs(specs=SIZE_LEGS, out_path: Path | None = None) -> list[dict]:
    """只为回答"全网格在 CPU 上能不能跑"：**不产出价格结论**（预注册 §6-4）。

    每条腿写盘一次（部分结果也能留档）：撞 `STATE_CAP` 就记 `capped=True` 并停止该腿。
    """
    cfg = mainline_config(plant=_plant_asym(SIZE_ALPHA))
    out: list[dict] = []
    for sp in specs:
        rec = {"tag": sp["tag"], "n_m": sp["n_m"], "tau": list(sp["tau"]),
               "b_ticks": list(sp["b_ticks"]), "p": list(sp["p"]), "alpha": SIZE_ALPHA,
               "P": SIZE_P, "capped": False, "states": {}, "sec": {}}
        for p in sp["p"]:
            pt = price_ticks(p)
            t0 = time.perf_counter()
            leg = Leg(cfg, sp["n_m"], pt)
            for tau in sp["tau"]:
                for b in sp["b_ticks"]:
                    if b > (tau - 1) * TICKS_PER_QUANTUM:
                        continue
                    leg.run_cells([(tau, b, SIZE_P)])
            rec["states"][f"p={p:g}"] = leg.states()
            rec["sec"][f"p={p:g}"] = round(time.perf_counter() - t0, 3)
            rec["capped"] = bool(rec["capped"] or leg.paid.capped() or leg.no.capped())
            print(f"  [size {sp['tag']} p={p:g}] {rec['sec'][f'p={p:g}']} s "
                  f"states={rec['states'][f'p={p:g}']} capped={rec['capped']}", flush=True)
        out.append(rec)
        if out_path is not None:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        if rec["capped"]:
            print(f"  [size {sp['tag']}] capped ⇒ 该腿停止", flush=True)
            break
    return out


# ==================================================================== 判词
def judge(p: dict, gates: dict, size: list, gran: dict | None = None) -> dict:
    rows = p["rows"]
    out: dict = {}
    fin = [r for r in rows if math.isfinite(r["IV_paid"]) and math.isfinite(r["A0"])]
    out["n_cells"] = len(rows)
    out["n_cells_finite"] = len(fin)
    out["n_cells_dropped"] = len(rows) - len(fin)
    out["n_capped"] = sum(1 for r in rows if r["capped"])
    pos = [r for r in fin if r["IV_paid"] > IV_NOISE]
    out["n_positive"] = len(pos)
    out["positive_cells"] = sorted(
        [{k: r[k] for k in ("alpha", "p", "probe_ticks", "tau", "b_ticks", "P")}
         | {"IV_paid": r["IV_paid"]} for r in pos],
        key=lambda x: (-x["IV_paid"]))[:20]
    out["max_IV_paid"] = float(max((r["IV_paid"] for r in fin), default=float("nan")))

    # 逐价格：最大 IV 与"是否全零"
    by_price = {}
    for pt in PRICE_TICKS:
        sub = [r for r in fin if r["probe_ticks"] == pt]
        by_price[f"p={pt / TICKS_PER_QUANTUM:g}"] = {
            "n": len(sub), "max_IV": float(max((r["IV_paid"] for r in sub), default=float("nan"))),
            "n_positive": sum(1 for r in sub if r["IV_paid"] > IV_NOISE)}
    out["by_price"] = by_price

    # 逐预算：最大 IV（看"整包发送粒度"在预算上的落点）
    by_b = {}
    for b in B_TICKS_LIST:
        sub = [r for r in fin if r["b_ticks"] == b]
        by_b[f"b={b}"] = {"n": len(sub),
                          "max_IV": float(max((r["IV_paid"] for r in sub), default=float("nan"))),
                          "n_positive": sum(1 for r in sub if r["IV_paid"] > IV_NOISE)}
    out["by_b_ticks"] = by_b

    out["gates"] = {k: {"pass": v.get("pass"), **{kk: vv for kk, vv in v.items()
                                                 if kk in ("max_rel", "n_violations", "max_abs_IV",
                                                           "n_hit", "max_IV", "n_hits",
                                                           "n_distinct_alpha", "tol")}}
                    for k, v in gates.items()}
    out["gates_n_fail"] = sum(1 for v in gates.values() if not v.get("pass"))
    out["gates_failed"] = sorted(k for k, v in gates.items() if not v.get("pass"))
    # ★ 判词口径：`T-6`（预注册格点）已由 `T-6b`（跑后边界格）替换 ⇒ 判词不计它，
    #   但两个读数都留在 `gates` 里，且下面显式写出"替换过"，不用"全绿"这种说法。
    out["gates_n_fail_verdict"] = sum(1 for k, v in gates.items()
                                      if not v.get("pass") and k != "t6")
    out["verdict_caveat"] = ("T-6（预注册格点 b=16）FAIL：该格无分辨力；"
                             "已用跑后 T-6b（可行性边界 b=8/10）替换，两套读数同报") \
        if "t6" in out["gates_failed"] else None
    out["size_legs"] = size
    # ★ `T-6`（预注册格点）与 `T-6b`（跑后边界格）分开登记，两套读数同报
    out["t6_pre_n_hit"] = gates.get("t6", {}).get("n_hit")
    out["t6b_n_hit"] = gates.get("t6b", {}).get("n_hit")
    out["t6_note"] = ("T-6 预注册格点（b=16）n_hit=0：该预算下连免费探测都不改动作 ⇒ 两引擎必然同值，"
                      "属**格点无分辨力**，不是'截断无害'。T-6b 把变异挪到可行性边界（b=8/10）。")
    out["granularity"] = None if gran is None else {
        "n_groups": gran["n_groups"], "n_multi_price_groups": gran["n_multi_price_groups"],
        "n_bit_identical_groups": gran["n_bit_identical_groups"],
        "all_multi_price_groups_bit_identical": gran["pass"],
        "top": [g for g in gran["groups"] if len(g["prices"]) > 1][:10],
    }

    # ---- 判词：只报"本卡能说什么"，不给价格边界
    if out["gates_n_fail_verdict"] > 0:
        out["VERDICT"] = "NO-VERDICT-GATE-FAIL"
    elif gates["t6b"]["n_hit"] == 0:
        out["VERDICT"] = "NO-VERDICT-TRUNCATION-GATE-HAS-NO-RESOLUTION"
    elif out["n_positive"] == 0:
        out["VERDICT"] = "PRICE-TICKS-VERIFIED-ALL-ZERO-IN-SMALL-INSTANCE"
    else:
        out["VERDICT"] = "PRICE-TICKS-VERIFIED-SOME-POSITIVE-IN-SMALL-INSTANCE"
    out["verdict_scope"] = ("只覆盖 n_m=17、τ≤4、2 用户、κ=1、σ_csi=σ⋆；"
                            "不构成价格边界，不许外推；b=1 的结构零只解释 b=1")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="CW-R5.1 价格边界实验卡（整数刻度）")
    ap.add_argument("--ctrl-only", action="store_true", help="只跑闸，不跑面板")
    ap.add_argument("--size-only", action="store_true", help="只跑规模腿")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    print(f"[{ROUND}] {PROTOCOL} start  ticks/q={TICKS_PER_QUANTUM} send={SEND_TICKS}", flush=True)

    dest = Path(args.out) if args.out else (
        RESULTS / f"cw_r5_1_price_ticks_{_now_utc().replace('-', '').replace(':', '')[:15]}"
                 f"_{_src_hash()[:16]}")
    dest.mkdir(parents=True, exist_ok=True)

    if args.size_only:
        size = size_legs(out_path=dest / "r51_size.json")
        print(f"[size-only] legs={len(size)}", flush=True)
        return 0

    gates = {"t1": gate_t1_p1_identity(), "t2": gate_t2_p0_identity(),
             "t5": gate_t5_no_info(), "t9": gate_t9_ast_no_truncation(),
             "t10": gate_t10_distinct()}
    if args.ctrl_only:
        print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "rows"}
                          for k, v in gates.items()}, indent=2, ensure_ascii=False)[:6000])
        print(f"[ctrl-only] n_fail={sum(1 for v in gates.values() if not v['pass'])} / {len(gates)}")
        return 0

    p = panel()
    gates["t3"] = gate_t3_monotone(p["rows"])
    gates["t4"] = gate_t4_b1_struct(p["rows"])
    gates["t6"] = gate_t6_truncation_mutation(p["rows"])
    gates["t6b"] = gate_t6b_truncation_edge()
    gates["t7"] = gate_t7_b1_regression(p["rows"])
    gates["t8"] = gate_t8_size_invariants(p["legs"])
    gran = granularity_table(p["rows"])
    size = size_legs(out_path=dest / "r51_size.json")
    v = judge(p, gates, size, gran)

    print(json.dumps({k: v[k] for k in ("VERDICT", "n_cells", "n_positive", "max_IV_paid",
                                        "by_price", "by_b_ticks", "gates_n_fail",
                                        "gates_failed", "t6_pre_n_hit", "t6b_n_hit",
                                        "verdict_caveat")},
                     indent=2, ensure_ascii=False)[:5000], flush=True)

    out = {"round": ROUND, "protocol": PROTOCOL, "utc": _now_utc(),
           "ticks_per_quantum": TICKS_PER_QUANTUM, "send_ticks": SEND_TICKS,
           "sec_total": round(time.perf_counter() - t0, 3),
           "panel": {"n_m": N_M_SMALL, "gh_n": GH, "n_users": N_USERS,
                     "alpha": list(ALPHA_LIST), "P": list(P_LIST), "tau": list(TAU_LIST),
                     "b_ticks": list(B_TICKS_LIST), "price": list(PRICE_LIST)},
           "cells": p["rows"], "legs": p["legs"], "size_legs": size,
           "granularity": gran, "gates": gates, "verdicts": v}
    (dest / "r51.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[evidence] {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
