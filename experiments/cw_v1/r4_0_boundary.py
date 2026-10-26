# -*- coding: utf-8 -*-
"""`CW-R4.0`：「买不起」的**边界** —— 在什么条件下完美 CSI 才值一个量子。

预注册：`docs/research/CW-R4.0-买不起的边界-预注册-v0.1.md`（写于正式面板之前）

上游：`CW-R3.5` 在标准 plant 上判 `UB := IV_free(σ⋆) ≡ 0`（150/150 格完美 CSI 不被买）。
负结果的价值等于它的**边界**：本轮扫 `σ_w` 不对称度 `α`、先验不对称 `κ`、`P`、`τ/b`，
找使 `IV_paid(σ⋆) > IV_NOISE` 的**最小条件集**；找不到就写"扫过的空间内不存在"，
并报**未扫维度**（用户数 3 用通用 n 用户 DP 单独试算，爆了就报实测状态数）。

三条纪律
--------
1. 零 episode、零训练、零参数更新；被测物一字节不动（只 `import`）。
2. 判据 `B0–B4/L1` 见预注册 §4，**跑前写死**；`judge()` 只按表取。
3. 通用 n 用户 DP 是**本轮新写**的，必须先用 `n=1,2` 与既有 `build_engine` 逐点对账（`G1`），
   再用 `r3_5_ubcert.cell()` 对账本面板（`G2`），最后才对 `n=3` 说话。
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
from .r3_2_iv2u import GH_N, M_RANGE, STATE_CAP, build_engine

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results" / "cw_v1"

PROTOCOL = "CW-POMDP-v1.1"
ROUND = "CW-R4.0"

_SQRT2PI = math.sqrt(2.0 * math.pi)

#: 面板（预注册 §3，跑前写死；**不按结果移动**）
SIGMA_STAR = 1.0e-9
SIGMA_W0 = 0.02
ASYM_LIST: tuple[float, ...] = (1.0, 1.5, 2.5, 5.0, 10.0, 15.0)
#: ★ 研究基线 `BOUNDED_PLANT` 的 `sigma_w=(0.02, 0.03)` **本身就不对称** ⇒ 基线等价于 `α=1.5`。
#: 首跑把 `α=1.0` 写成"基线"、`α=1.5` 又用 `_plant_asym` 重建出同一个 `(0.02,0.03)`，
#: 导致 80 个格是**同一配置的重复单元**（`α=1.0` 那 80 格其实是 `(0.02,0.02)`，从未被跑过）。
BASELINE_ALPHA = 0.03 / SIGMA_W0
KAPPA_LIST: tuple[float, ...] = (1.0, 0.5, 0.2, 0.05)
P_LIST: tuple[float, ...] = (0.01, 0.05)
TAU_LIST: tuple[int, ...] = (2, 3, 4, 5)
B_MAX = 4                     # `b <= min(τ−1, 4)`：**排除 `b=τ`**（`ΔV_q≡0`）
N_M_MAIN = 65
V_PRIOR = 1.0

#: 判据常数（预注册 §4）
IV_NOISE = 1e-12
UB_FLOOR = 1e-9
B0_TOL = 1e-15
SPREAD_MAX = 0.30
IDENT_TOL = 1e-12

#: 3 用户试算（预注册 §3 末）
N3_USERS = 3
N3_TAU = (2, 3)
#: **跑后扩展（未预注册）**：`τ≤3` 只有 1 次探测机会，加 `τ=4` 去掉这个退化
N3_TAU_EXT = (4,)
N3_NM = (17, 33)

#: R3.5 已记录值（`G3` 复现用；写死路径，找不到就 FAIL 不降级）
R35_POSTHOC = None   # 运行时按 glob 解析，见 `gate_g3_reproduce`
R35_RATIO_65 = 0.031096313177363884
R35_TOL = 1e-9

_CFG = mainline_config()


def _file_hash(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def _plant_asym(alpha: float) -> PlantConfig:
    base = mainline_config().plant
    return PlantConfig(**{**base.__dict__, "sigma_w": (SIGMA_W0, SIGMA_W0 * float(alpha))})


def _plant_three(sw: tuple[float, ...] = (SIGMA_W0, SIGMA_W0, SIGMA_W0)) -> PlantConfig:
    """3 用户 plant：`A≡1`，`σ_w` 可指定，其余字段与研究基线一致。"""
    base = mainline_config().plant
    return PlantConfig(**{**base.__dict__, "A": (1.0, 1.0, 1.0), "sigma_w": sw})


# ==================================================================== 通用 n 用户 DP
def build_engine_n(cfg, sigma_csi: float, n_m: int, gh_n: int = GH_N, *,
                   probe_enabled: bool = True, probe_cost: int = 1,
                   drop_obs: bool = False, n_users: int = 2,
                   state_cap: int = STATE_CAP):
    """`n_users` 用户的**精确**窗口 DP（`(m,v)` 充分的前提同 R3.2：ACK 不进信道信念）。

    与 `r3_2_iv2u.build_engine` 的差别只有两处：
    ① 状态用元组 `(t, P_vec, m_vec, v_vec, b)`，用户可以 >2；
    ② 噪声/代价按 `i` 逐用户取（`a2[i]`、`sw2[i]`）—— R3.2 已如此，这里保持同一算术顺序，
       所以 `n=1,2` 应与它**逐位一致**（`G1` 会验）。

    ★ 本函数不改任何既有模块；它是**新的**实现，必须自证与被测物一致才能用。
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
    pc = int(probe_cost)
    nu = int(n_users)
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

        if b >= 1:                                        # SEND_i
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
                va = ph * V(t - 1, tuple(aa), mp, vp0, b - 1)
                vb = (1.0 - ph) * V(t - 1, tuple(bb), mp, vp0, b - 1)
                val = va + vb
                if val < best:
                    best = val

        if probe_enabled and b >= pc and t >= 2:           # PROBE_i
            for i in range(nu):
                mi = mm[i]
                vi = vv[i]
                if drop_obs:                               # 变异：读数被丢弃
                    vp = tuple(rho2 * v + infl for v in vv)
                    mm2 = list(mp)
                    mm2[i] = rho * mi
                    vv2 = list(vp)
                    vv2[i] = rho2 * vi + infl
                    tot = V(t - 1, ff, tuple(mm2), tuple(vv2), b - pc)
                else:
                    gp = vi / (vi + si2)
                    vpi = rho2 * (vi * si2 / (vi + si2)) + infl
                    sy = math.sqrt(vi + si2)
                    tot = 0.0
                    for z, w in zip(nodes, weights):
                        mm2 = list(mp)
                        mm2[i] = rho * snap(mi + gp * sy * z)
                        vv2 = list(vp0)
                        vv2[i] = vpi
                        tot += w * V(t - 1, ff, tuple(mm2), tuple(vv2), b - pc)
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

    def first(t, Pvec, mvec, vvec, b) -> dict:
        """首步各动作值（只用于机制量，不参与判词）。"""
        t = int(t)
        Pp = tuple(float(x) for x in Pvec)
        mm = tuple(float(x) for x in mvec)
        vv = tuple(float(x) for x in vvec)
        b = int(b)
        S = q * sum(Pp)
        ff = tuple(a2[i] * Pp[i] + sw2[i] for i in range(nu))
        mp = tuple(rho * m for m in mm)
        vp0 = tuple(rho2 * v + infl for v in vv)
        out = {"HOLD": S + V(t - 1, ff, mp, vp0, b)}
        for i in range(nu):
            if b >= 1:
                Pi = Pp[i]
                g = Pi / (Pi + sm2)
                ps_ = a2[i] * (1.0 - g) * Pi + sw2[i]
                pf_ = a2[i] * Pi + sw2[i]
                ph = ps(mm[i], vv[i])
                aa = list(ff)
                aa[i] = ps_
                bb = list(ff)
                bb[i] = pf_
                out[f"SEND_{i}"] = (S + ph * V(t - 1, tuple(aa), mp, vp0, b - 1)
                                    + (1.0 - ph) * V(t - 1, tuple(bb), mp, vp0, b - 1))
            if probe_enabled and b >= pc and t >= 2:
                mi, vi = mm[i], vv[i]
                gp = vi / (vi + si2)
                vpi = rho2 * (vi * si2 / (vi + si2)) + infl
                sy = math.sqrt(vi + si2)
                tot = 0.0
                for z, w in zip(nodes, weights):
                    mm2 = list(mp)
                    mm2[i] = rho * snap(mi + gp * sy * z)
                    vv2 = list(vp0)
                    vv2[i] = vpi
                    tot += w * V(t - 1, ff, tuple(mm2), tuple(vv2), b - pc)
                out[f"PROBE_{i}"] = S + tot / _SQRT2PI
        return out

    def n_states() -> int:
        return len(memo)

    value.n_states = n_states                # type: ignore[attr-defined]
    value.capped = lambda: state["capped"]   # type: ignore[attr-defined]
    value.first = first                      # type: ignore[attr-defined]
    value.n_users = nu                       # type: ignore[attr-defined]
    return value


# ==================================================================== 臂与格
class Arms:
    """一套 `(plant, n_m)` 上的三条臂：`noprobe` / `free(σ⋆)` / `paid(σ⋆)`。"""

    def __init__(self, cfg, n_m: int, n_users: int = 2):
        self.cfg = cfg
        self.n_m = int(n_m)
        self.nu = int(n_users)
        t0 = time.perf_counter()
        self.no = build_engine_n(cfg, SIGMA_STAR, self.n_m, probe_enabled=False, n_users=self.nu)
        self.fr = build_engine_n(cfg, SIGMA_STAR, self.n_m, probe_cost=0, n_users=self.nu)
        self.pd = build_engine_n(cfg, SIGMA_STAR, self.n_m, probe_cost=1, n_users=self.nu)
        self.sec = round(time.perf_counter() - t0, 3)

    def states(self) -> dict:
        return {"noprobe": self.no.n_states(), "free": self.fr.n_states(),
                "paid": self.pd.n_states()}

    def capped(self) -> bool:
        return bool(self.no.capped() or self.fr.capped() or self.pd.capped())


def cell2(arms: Arms, tau: int, b: int, P0: float, P1: float) -> dict:
    """2 用户一格：`A0 / free(σ⋆) / paid(σ⋆) / ΔV_q` + 机制量。"""
    vp = V_PRIOR
    a0 = arms.no(tau, (P0, P1), (0.0, 0.0), (vp, vp), b)
    a0m = arms.no(tau, (P0, P1), (0.0, 0.0), (vp, vp), b - 1) if b >= 1 else float("nan")
    f = arms.fr(tau, (P0, P1), (0.0, 0.0), (vp, vp), b)
    d = arms.pd(tau, (P0, P1), (0.0, 0.0), (vp, vp), b)
    dq = (a0m - a0) if b >= 1 else float("nan")

    # 机制量：free 臂首步各动作
    ft = arms.fr.first(tau, (P0, P1), (0.0, 0.0), (vp, vp), b)
    s_send = min(v for k, v in ft.items() if k.startswith("SEND_"))
    s_probe = min(v for k, v in ft.items() if k.startswith("PROBE_"))
    probe_arg = min((k for k in ft if k.startswith("PROBE_")), key=lambda k: ft[k])
    send_arg = min((k for k in ft if k.startswith("SEND_")), key=lambda k: ft[k])

    return {
        "tau": int(tau), "b": int(b), "P0": float(P0), "P1": float(P1), "n_m": arms.n_m,
        "A0": float(a0), "IV_free": float(a0 - f), "IV_paid": float(a0 - d),
        "dV_q": float(dq),
        "ratio_free": float((a0 - f) / dq) if (dq is not None and dq > 1e-12) else None,
        "ratio_paid": float((a0 - d) / dq) if (dq is not None and dq > 1e-12) else None,
        "deficit_free": float(s_probe - s_send),
        "probe_arg": probe_arg, "send_arg": send_arg,
        "probe_chosen": bool(s_probe <= s_send),
    }


# ==================================================================== 面板
def panel_two_user() -> dict:
    cfg0 = mainline_config()
    rows = []
    meta = []
    for alpha in ASYM_LIST:
        cfg = mainline_config(plant=_plant_asym(alpha))
        arms = Arms(cfg, N_M_MAIN, n_users=2)
        rows0 = len(rows)
        for kappa in KAPPA_LIST:
            for P in P_LIST:
                for tau in TAU_LIST:
                    for b in range(1, min(tau - 1, B_MAX) + 1):
                        c = cell2(arms, tau, b, P, P * kappa)
                        c["alpha"] = float(alpha)
                        c["is_base"] = bool(abs(alpha - BASELINE_ALPHA) < 1e-18)
                        c["kappa"] = float(kappa)
                        c["P"] = float(P)
                        rows.append(c)
        # ★ 状态计数必须在**求值之后**取（R3.2 自伤：取在惰性递归之前 ⇒ 全 0）
        st = arms.states()
        mine = rows[rows0:]
        mx_free = max(c["IV_free"] for c in mine)
        mx_paid = max(c["IV_paid"] for c in mine)
        nz = [c for c in mine if c["IV_free"] > 1e-9]
        meta.append({"alpha": alpha, "sec": arms.sec, "states": st,
                     "capped": arms.capped(), "n_rows": len(mine),
                     "max_IV_free": float(mx_free), "max_IV_paid": float(mx_paid),
                     "n_free_above_floor": len(nz),
                     "cells_free_positive": [{k: c[k] for k in
                                              ("kappa", "P", "tau", "b", "IV_free",
                                               "IV_paid", "ratio_free", "probe_chosen")}
                                             for c in nz][:8]})
        print(f"  [panel α={alpha:g}] rows={len(rows)} states={st} capped={arms.capped()} "
              f"maxIV_free={mx_free:.4e} maxIV_paid={mx_paid:.4e} n_pos_free={len(nz)}",
              flush=True)
    return {"rows": rows, "meta": meta}


def refine_top_free(rows: list, k: int = 3) -> dict:
    """**跑后补充（未预注册）**：`IV_paid` 全零 ⇒ 预注册的 `B3` 没有靶子。

    但主报数 `max IV_free/ΔV_q` 仍要过分辨率关，否则"0.106 个量子"可能是网格假象。
    对 `IV_free` 最大的 `k` 个格在 `n_m ∈ {65,129}` 上重算（`257` 在 `b≥3` 上按状态量
    不做，理由写进交付：`n_m=257` 每臂状态量按 `n_m/65` 的幂次涨，超时风险大于收益）。
    """
    cfg0 = mainline_config()
    fin = [c for c in rows if isinstance(c["IV_free"], float) and math.isfinite(c["IV_free"])]
    top = sorted(fin, key=lambda c: -c["IV_free"])[:k]
    out = []
    for h in top:
        alpha = h["alpha"]
        cfg = mainline_config(plant=_plant_asym(alpha))
        rr = []
        for n_m in (65, 129):
            arms = Arms(cfg, n_m, n_users=2)
            c = cell2(arms, h["tau"], h["b"], h["P"], h["P"] * h["kappa"])
            rr.append({"n_m": n_m, "IV_free": c["IV_free"], "IV_paid": c["IV_paid"],
                       "ratio_free": c["ratio_free"], "dV_q": c["dV_q"],
                       "capped": arms.capped()})
        vals = [r["ratio_free"] for r in rr if isinstance(r["ratio_free"], float)]
        mx = max(vals) if vals else 0.0
        spread = float((mx - min(vals)) / mx) if mx > 1e-15 else 0.0
        out.append({"cell": {k2: h[k2] for k2 in ("alpha", "kappa", "P", "tau", "b")},
                    "rows": rr, "spread_rel": spread})
        print(f"  [refine-free α={alpha:g} κ={h['kappa']:g} P={h['P']:g} τ={h['tau']} "
              f"b={h['b']}] ratio={[round(v, 4) for v in vals]} spread={spread:.3f}",
              flush=True)
    return {"top": out, "n_m": [65, 129],
            "max_spread_rel": float(max((o["spread_rel"] for o in out), default=float("nan")))}


def refine_positive(hits: list) -> list:
    """对每个转正格在 `n_m ∈ {65,129,257}` 上重算（`B3`）。"""
    cfg0 = mainline_config()
    out = []
    cache: dict = {}
    for h in hits:
        alpha = h["alpha"]
        if alpha not in cache:
            cfg = mainline_config(plant=_plant_asym(alpha))
            cache[alpha] = cfg
        cfg = cache[alpha]
        rr = []
        for n_m in (65, 129, 257):
            arms = Arms(cfg, n_m, n_users=2)
            c = cell2(arms, h["tau"], h["b"], h["P"], h["P"] * h["kappa"])
            rr.append({"n_m": n_m, "IV_paid": c["IV_paid"], "IV_free": c["IV_free"],
                       "dV_q": c["dV_q"], "ratio_paid": c["ratio_paid"],
                       "capped": arms.capped()})
        vals = [r["IV_paid"] for r in rr if isinstance(r["IV_paid"], float)
                and math.isfinite(r["IV_paid"])]
        mx = max(vals) if vals else 0.0
        spread = (float((mx - min(vals)) / mx) if mx > 1e-15 else 0.0) if vals else None
        out.append({"cell": {k: h[k] for k in ("alpha", "kappa", "P", "tau", "b")},
                    "rows": rr, "spread_rel": spread, "max": float(mx)})
        print(f"  [refine α={alpha:g} κ={h['kappa']:g} P={h['P']:g} "
              f"τ={h['tau']} b={h['b']}] spread={spread}", flush=True)
    return out


def panel_three_user(taus=N3_TAU, tag: str = "prereg",
                     sw=(SIGMA_W0, SIGMA_W0, SIGMA_W0)) -> dict:
    """`n=3` 试算：能跑完就给数据点，撞顶就报实测状态数（`B4`）。"""
    cfg = mainline_config(plant=_plant_three(tuple(sw)))
    out = {"n_users": N3_USERS, "rows": [], "capped": False, "tag": tag, "taus": list(taus), "sigma_w": list(sw),
           "states": {}, "note": "P0=P1=P2，先验对称；τ≤3 ⇒ 只有 1–2 次 PROBE 机会"}
    for n_m in N3_NM:
        t0 = time.perf_counter()
        arms = Arms(cfg, n_m, n_users=N3_USERS)
        cap = False
        for tau in taus:
            for b in range(1, min(tau - 1, 3) + 1):
                for P in (0.01, 0.05):
                    a0 = arms.no(tau, (P, P, P), (0.0, 0.0, 0.0), (V_PRIOR,) * 3, b)
                    a0m = arms.no(tau, (P, P, P), (0.0, 0.0, 0.0), (V_PRIOR,) * 3, b - 1)
                    fr = arms.fr(tau, (P, P, P), (0.0, 0.0, 0.0), (V_PRIOR,) * 3, b)
                    pd = arms.pd(tau, (P, P, P), (0.0, 0.0, 0.0), (V_PRIOR,) * 3, b)
                    bad = not all(math.isfinite(x) for x in (a0, a0m, fr, pd))
                    cap = bool(cap or bad or arms.capped())
                    out["rows"].append({
                        "n_m": n_m, "tau": tau, "b": b, "P": P,
                        "A0": float(a0), "IV_free": float(a0 - fr), "IV_paid": float(a0 - pd),
                        "dV_q": float(a0m - a0),
                        "capped": bool(bad or arms.capped()),
                    })
        st = arms.states()          # ★ 求值之后才取（否则全 0）
        cap = bool(cap or arms.capped())
        out["states"][str(n_m)] = st
        out["capped"] = bool(out["capped"] or cap)
        out.setdefault("sec", {})[str(n_m)] = round(time.perf_counter() - t0, 3)
        print(f"  [3U n_m={n_m}] {out['sec'][str(n_m)]} s states={st} capped={cap}",
              flush=True)
        if cap:
            out["capped_at"] = {"n_m": n_m, "states": st}
            break
    return out


# ==================================================================== 闸
def gate_g1_identity_engine() -> dict:
    """新写的通用 n 用户 DP 必须与既有 `build_engine` 在 `n=1,2` 上一致。"""
    cfg = mainline_config()
    rows = []
    worst = 0.0
    for nu in (1, 2):
        for n_m in (33,):
            mine = {k: build_engine_n(cfg, SIGMA_STAR, n_m, probe_enabled=(k != "no"),
                                      probe_cost=(0 if k == "fr" else 1), n_users=nu)
                    for k in ("no", "fr", "pd")}
            ref = {"no": build_engine(cfg, SIGMA_STAR, n_m, probe_enabled=False, n_users=nu),
                   "fr": build_engine(cfg, SIGMA_STAR, n_m, probe_cost=0, n_users=nu),
                   "pd": build_engine(cfg, SIGMA_STAR, n_m, probe_cost=1, n_users=nu)}
            for tau, b, P in ((3, 1, 0.05), (4, 2, 0.01), (4, 3, 0.1)):
                for k in ("no", "fr", "pd"):
                    if nu == 1:
                        a1 = mine[k](tau, (P,), (0.0,), (V_PRIOR,), b)
                        a2_ = ref[k](tau, P, 0.0, 0.0, V_PRIOR, 0.0, V_PRIOR, b)
                    else:
                        a1 = mine[k](tau, (P, P), (0.0, 0.0), (V_PRIOR, V_PRIOR), b)
                        a2_ = ref[k](tau, P, P, 0.0, V_PRIOR, 0.0, V_PRIOR, b)
                    rel = abs(a1 - a2_) / max(1e-12, abs(a2_))
                    worst = max(worst, rel)
                    rows.append({"n_users": nu, "arm": k, "tau": tau, "b": b, "P": P,
                                 "mine": float(a1), "ref": float(a2_), "rel": float(rel)})
    return {"rows": rows, "max_rel": float(worst), "tol": IDENT_TOL,
            "pass": bool(worst <= IDENT_TOL)}


def gate_g2_identity_panel(rows: list) -> dict:
    """本面板的对称子集（α=1, κ=1）必须与 `r3_5_ubcert.cell()` 一致。"""
    from .r3_5_ubcert import Arms as Arms35
    from .r3_5_ubcert import cell as cell35

    cfg = mainline_config()
    a35 = Arms35(cfg, N_M_MAIN)
    pairs = []
    worst = 0.0
    for c in rows:
        if not c.get("is_base") or abs(c["kappa"] - 1.0) > 0:
            continue
        o = cell35(a35, cfg, c["tau"], c["b"], c["P"], SIGMA_STAR)
        for k, k2 in (("IV_free", "IV_free_star"), ("IV_paid", "IV_paid_star"),
                      ("dV_q", "dV_q")):
            d = abs(c[k] - o[k2])
            rel = d / max(1e-12, abs(o[k2]))
            worst = max(worst, rel)
            pairs.append({"tau": c["tau"], "b": c["b"], "P": c["P"], "k": k,
                          "mine": c[k], "r35": o[k2], "abs": float(d), "rel": float(rel)})
    return {"n_pairs": len(pairs), "max_rel": float(worst), "tol": IDENT_TOL,
            "pass": bool(pairs) and worst <= IDENT_TOL}


def gate_g3_reproduce(rows: list) -> dict:
    """`B2` 的硬腿：`(α=1,κ=1,τ=5,b=1,P=0.01,n_m=65)` 的 `UB/ΔV_q` 必须等于 R3.5 记录值。"""
    glob = sorted((RESULTS).glob("cw_r3_5_ubcert_*/r35_posthoc.json"))
    found = bool(glob)
    hit = [c for c in rows if c.get("is_base") and abs(c["kappa"] - 1.0) < 1e-18
           and c["tau"] == 5 and c["b"] == 1 and abs(c["P"] - 0.01) < 1e-18]
    if not hit:
        return {"found": found, "cell_found": False, "pass": False}
    ratio = hit[0]["ratio_free"]
    d = abs(float(ratio) - R35_RATIO_65) if ratio is not None else float("nan")
    return {"found": found, "cell_found": True, "ratio_mine": float(ratio),
            "ratio_r35": R35_RATIO_65, "abs_diff": float(d), "tol": R35_TOL,
            "pass": bool(math.isfinite(d) and d <= R35_TOL)}


def gate_g8_distinct_configs() -> dict:
    """`G8`：面板里每个 `α` 必须是**互不相同**的 plant。

    这条闸是首跑的现场教训：`α=1.0` 走 `mainline_config()`（`σ_w=(0.02,0.03)`）、
    `α=1.5` 走 `_plant_asym(1.5)`（同样得到 `(0.02,0.03)`）⇒ **80 个格是同一配置的重复单元**，
    而 `α=1.0` 想表达的对称 plant `(0.02,0.02)` 一次都没跑过。
    判据：指纹数 = `α` 数，且基线 `α` 唯一。
    """
    fps = {}
    for alpha in ASYM_LIST:
        cfg = mainline_config(plant=_plant_asym(alpha))
        fps[f"{alpha:g}"] = {"sigma_w": list(cfg.plant.sigma_w), "fp": cfg.fingerprint()}
    uniq = len({v["fp"] for v in fps.values()})
    n_base = sum(1 for a in ASYM_LIST if abs(a - BASELINE_ALPHA) < 1e-18)
    return {"per_alpha": fps, "n_alpha": len(ASYM_LIST), "n_distinct": uniq,
            "n_baseline_alpha": n_base, "baseline_alpha": BASELINE_ALPHA,
            "pass": bool(uniq == len(ASYM_LIST) and n_base == 1)}


def gate_g4_no_truth() -> dict:
    src = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    banned = {"h_true", "cost_true_by_user", "cost_true", "w_realized", "x_true"}
    hits = [n.attr for n in ast.walk(tree)
            if isinstance(n, ast.Attribute) and n.attr in banned]
    hits += [n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id in banned]
    return {"banned": sorted(banned), "hits": hits, "pass": not hits}


def gate_g5_identity_files() -> dict:
    mods = ["r3_2_iv2u.py", "config.py", "posterior.py", "r3_5_ubcert.py"]
    h = {m: _file_hash(REPO / "experiments" / "cw_v1" / m) for m in mods}
    return {"hashes": h, "dir_hash": _src_hash(), "pass": bool(all(v for v in h.values()))}


def gate_g6_first_consistency() -> dict:
    cfg = mainline_config()
    arms = Arms(cfg, 33, n_users=2)
    rows = []
    worst = 0.0
    for tau, b, P in ((3, 1, 0.05), (4, 2, 0.01), (4, 3, 0.1)):
        for name, fn in (("noprobe", arms.no), ("free", arms.fr), ("paid", arms.pd)):
            v = fn(tau, (P, P), (0.0, 0.0), (V_PRIOR, V_PRIOR), b)
            f = fn.first(tau, (P, P), (0.0, 0.0), (V_PRIOR, V_PRIOR), b)
            d = abs(min(f.values()) - v)
            worst = max(worst, d)
            rows.append({"tau": tau, "b": b, "arm": name, "V": float(v),
                         "min_first": float(min(f.values())), "abs_diff": float(d)})
    return {"rows": rows, "max_abs": float(worst), "tol": 1e-15,
            "pass": bool(worst <= 1e-15)}


def gate_g7_drop_obs() -> dict:
    """变异：`free(σ⋆)` 臂把读数丢弃后必须与 `A0` 逐位相等（优势全来自读数）。"""
    cfg = mainline_config()
    no = build_engine_n(cfg, SIGMA_STAR, 65, probe_enabled=False, n_users=2)
    drop = build_engine_n(cfg, SIGMA_STAR, 65, probe_cost=0, drop_obs=True, n_users=2)
    rows = []
    worst = 0.0
    for tau, b, P in ((3, 1, 0.05), (4, 2, 0.01), (4, 3, 0.1)):
        a0 = no(tau, (P, P), (0.0, 0.0), (V_PRIOR, V_PRIOR), b)
        dd = drop(tau, (P, P), (0.0, 0.0), (V_PRIOR, V_PRIOR), b)
        d = abs(a0 - dd)
        worst = max(worst, d)
        rows.append({"tau": tau, "b": b, "P": P, "A0": float(a0), "drop": float(dd),
                     "abs_diff": float(d)})
    return {"rows": rows, "max_abs": float(worst), "tol": 1e-15,
            "pass": bool(worst <= 1e-15)}


# ==================================================================== 判词
def _fin(x) -> bool:
    return isinstance(x, float) and math.isfinite(x)


def judge(p: dict, gates: dict, refine: list, p3: dict,
          refine_free: dict | None = None, p3e: dict | None = None) -> dict:
    rows = p["rows"]
    out: dict = {}
    out["n_cells"] = len(rows)
    fin = [c for c in rows if _fin(c["IV_free"]) and _fin(c["IV_paid"]) and _fin(c["dV_q"])]
    out["n_cells_finite"] = len(fin)
    out["n_cells_dropped"] = len(rows) - len(fin)
    valid = [c for c in fin if c["dV_q"] > 1e-12]
    out["n_cells_valid"] = len(valid)
    out["n_cells_degenerate_dVq"] = len(fin) - len(valid)

    # ---- B0：b=1 结构性零
    b1 = [c for c in fin if c["b"] == 1]
    worst_b1 = max((abs(c["IV_paid"]) for c in b1), default=float("nan"))
    out["B0"] = "B1-CANNOT-PAY" if (b1 and worst_b1 <= B0_TOL) else "STRUCTURE-WRONG"
    out["B0_detail"] = {"n_b1": len(b1), "max_abs_IV_paid": float(worst_b1),
                        "tol": B0_TOL}

    # ---- B1：转正格与最小条件集
    hits = [c for c in fin if c["IV_paid"] > IV_NOISE]
    out["B1"] = "PAYABLE-BOUNDARY-FOUND" if hits else "NO-BOUNDARY-IN-SCANNED-SPACE"
    out["B1_detail"] = {
        "n_hits": len(hits), "noise": IV_NOISE,
        "cells": sorted([{k: c[k] for k in ("alpha", "kappa", "P", "tau", "b")}
                         | {"IV_paid": c["IV_paid"], "ratio_paid": c["ratio_paid"],
                            "IV_free": c["IV_free"]} for c in hits],
                        key=lambda x: (x["alpha"], x["kappa"], x["P"], x["tau"], x["b"]))[:40],
        "max_ratio_paid": (float(max(c["ratio_paid"] for c in hits
                                     if c["ratio_paid"] is not None)) if hits else None),
        "n_paid_exact_zero": int(sum(1 for c in fin if abs(c["IV_paid"]) <= B0_TOL)),
        "n_paid_positive": len(hits),
        "n_free_positive": int(sum(1 for c in fin if c["IV_free"] > UB_FLOOR)),
        "n_b_ge2_free_positive": int(sum(1 for c in fin
                                         if c["b"] >= 2 and c["IV_free"] > UB_FLOOR)),
    }
    # 逐维边际：每个取值上有多少转正格（"到哪一步转正"）
    marg = {}
    for dim in ("alpha", "kappa", "P", "tau", "b"):
        vals = sorted({c[dim] for c in fin})
        marg[dim] = {f"{v:g}": {"n_cells": sum(1 for c in fin if c[dim] == v),
                                "n_hits": sum(1 for c in hits if c[dim] == v)}
                     for v in vals}
    out["B1_marginals"] = marg

    # 单维转正点（其余维固定在基线：α=1, κ=1, P=0.01）
    single = {}
    base = [c for c in fin if c.get("is_base") and abs(c["kappa"] - 1.0) < 1e-18]
    single["alpha@(κ=1,P=0.01)"] = sorted({
        c["alpha"]: any(cc["IV_paid"] > IV_NOISE for cc in base if cc["alpha"] == c["alpha"])
        for c in base}.items())
    single["kappa@(α=1,P=0.01)"] = sorted({
        c["kappa"]: any(cc["IV_paid"] > IV_NOISE for cc in base if cc["kappa"] == c["kappa"])
        for c in base}.items())
    # 自由边际：固定其余维不变，单独放开一维时的首个转正值
    free_edge = {}
    for dim in ("alpha", "kappa", "P"):
        vals = sorted({c[dim] for c in fin})
        free_edge[dim] = [{"value": v,
                           "n_hits_at_value": sum(1 for c in hits if c[dim] == v)}
                          for v in vals]
    out["B1_single_dim"] = single
    out["B1_free_edge"] = free_edge

    # ---- B2：基线复现（对称档全零 + R3.5 记录格）
    sym = [c for c in fin if c.get("is_base") and abs(c["kappa"] - 1.0) < 1e-18]
    max_paid_sym = max((c["IV_paid"] for c in sym), default=float("nan"))
    g3 = gates.get("g3", {})
    out["B2"] = "BASELINE-REPRODUCED" if (
        math.isfinite(max_paid_sym) and max_paid_sym <= IV_NOISE and g3.get("pass")) \
        else "BASELINE-DIVERGED"
    out["B2_detail"] = {"n_sym": len(sym), "max_IV_paid_sym": float(max_paid_sym),
                        "g3": g3}

    # ---- B3：分辨率
    spreads = [r["spread_rel"] for r in refine if r["spread_rel"] is not None]
    out["B3"] = ("BOUNDARY-GRID-STABLE" if (spreads and max(spreads) <= SPREAD_MAX)
                 else ("BOUNDARY-GRID-DEPENDENT" if spreads and max(spreads) > SPREAD_MAX
                       else "NO-POSITIVE-CELL-TO-REFINE"))
    out["B3_detail"] = {"n_refined": len(refine), "max_spread_rel":
                        float(max(spreads)) if spreads else None, "threshold": SPREAD_MAX,
                        "rows": refine}

    # ---- B4：3 用户
    out["B4"] = "SCALE-EXCEEDS-3U" if p3.get("capped") else "3U-DATA-POINT"
    out["B4_detail"] = {"capped": p3.get("capped"), "states": p3.get("states"),
                        "capped_at": p3.get("capped_at"), "sec": p3.get("sec"),
                        "n_rows": len(p3.get("rows", [])),
                        "rows": p3.get("rows", [])[:12]}

    # ---- L1：活体分辨率（α>1 档必须有正 UB）
    hi = [c for c in fin if c["alpha"] > BASELINE_ALPHA]
    max_free_hi = max((c["IV_free"] for c in hi), default=float("nan"))
    out["L1"] = ("MACHINERY-CAN-BE-POSITIVE" if (hi and max_free_hi > UB_FLOOR)
                 else "MACHINERY-CANNOT-BE-POSITIVE")
    out["L1_detail"] = {"n_cells_alpha_gt1": len(hi), "max_IV_free": float(max_free_hi),
                        "floor": UB_FLOOR,
                        "n_free_above_floor": int(sum(1 for c in hi
                                                      if c["IV_free"] > UB_FLOOR))}

    # ---- B3b：毛价值主报数的分辨率（跑后补充）
    rf = refine_free or {}
    spread_f = rf.get("max_spread_rel")
    out["B3b"] = ("GROSS-VALUE-GRID-STABLE"
                  if isinstance(spread_f, float) and math.isfinite(spread_f)
                  and spread_f <= SPREAD_MAX else "GROSS-VALUE-GRID-DEPENDENT")
    out["B3b_detail"] = {"n_m": rf.get("n_m"), "max_spread_rel": spread_f,
                         "threshold": SPREAD_MAX, "top": rf.get("top")}

    # ---- 价格上限（**不是判词**，是负结果的"距离"）：毛价值最大折成量子数
    ratios = [c["ratio_free"] for c in fin if c["ratio_free"] is not None]
    mx_r = max(ratios) if ratios else None
    best = max((c for c in fin if c["ratio_free"] is not None),
               key=lambda c: c["ratio_free"]) if ratios else None
    out["price_ceiling"] = {
        "max_ratio_free": float(mx_r) if mx_r is not None else None,
        "cell": ({k: best[k] for k in ("alpha", "kappa", "P", "tau", "b")}
                 | {"IV_free": best["IV_free"], "dV_q": best["dV_q"],
                    "probe_chosen": best["probe_chosen"]}) if best else None,
        "meaning": "免量子时（perfect CSI）探测最多值这么多个量子 ⇒ 量子价格须降到 ≤此值才买得起"
                   "；**不外推**（α≤15、κ≥0.05、τ≤5、P≤0.05、2 用户）",
        "n_free_above_floor": int(sum(1 for c in fin if c["IV_free"] > UB_FLOOR)),
        "n_cells": len(fin),
    }

    # ---- 3 用户扩展（跑后）
    pe = p3e or {}
    out["B4_ext"] = {"tag": pe.get("tag"), "taus": pe.get("taus"),
                     "capped": pe.get("capped"), "states": pe.get("states"),
                     "rows": pe.get("rows", [])[:12],
                     "verdict": "SCALE-EXCEEDS-3U" if pe.get("capped") else "3U-DATA-POINT"}

    out["gates_n_fail"] = sum(1 for v in gates.values()
                              if isinstance(v, dict) and not v.get("pass", True))

    # ---- 汇总
    if out["gates_n_fail"] > 0:
        out["VERDICT"] = "NO-VERDICT-GATE-FAIL"
    elif out["L1"] != "MACHINERY-CAN-BE-POSITIVE":
        out["VERDICT"] = "NO-VERDICT-L1-MACHINERY-CANNOT-BE-POSITIVE"
    elif out["B2"] != "BASELINE-REPRODUCED":
        out["VERDICT"] = "NO-VERDICT-BASELINE-DIVERGED"
    elif out["B1"] == "NO-BOUNDARY-IN-SCANNED-SPACE":
        out["VERDICT"] = "NO-BOUNDARY-IN-SCANNED-SPACE"
    elif out["B3"] == "BOUNDARY-GRID-DEPENDENT":
        out["VERDICT"] = "BOUNDARY-GRID-DEPENDENT"
    else:
        out["VERDICT"] = "PAYABLE-BOUNDARY-FOUND"
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="CW-R4.0 买不起的边界")
    ap.add_argument("--ctrl-only", action="store_true")
    ap.add_argument("--skip-3u", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    print(f"[{ROUND}] {PROTOCOL} start", flush=True)

    if args.ctrl_only:
        g = {"g1": gate_g1_identity_engine(), "g4": gate_g4_no_truth(),
             "g5": gate_g5_identity_files(), "g6": gate_g6_first_consistency(),
             "g7": gate_g7_drop_obs(), "g8": gate_g8_distinct_configs()}
        print(json.dumps(g, indent=2, ensure_ascii=False)[:4000])
        print(f"[ctrl-only] n_fail={sum(1 for v in g.values() if not v['pass'])} / {len(g)}")
        return 0

    p = panel_two_user()
    print(f"  panel done rows={len(p['rows'])}", flush=True)

    fin = [c for c in p["rows"] if _fin(c["IV_paid"])]
    hits = [c for c in fin if c["IV_paid"] > IV_NOISE]
    print(f"  hits={len(hits)}", flush=True)
    refine = refine_positive(hits[:6]) if hits else []
    refine_free = refine_top_free(p["rows"], k=3)     # 跑后补充，见 docstring

    p3 = {"capped": True, "states": {}, "rows": [],
          "capped_at": {"n_m": "skipped"}} if args.skip_3u else panel_three_user()
    p3e = {"capped": True, "states": {}, "rows": [],
           "capped_at": {"n_m": "skipped"}} if args.skip_3u else \
        panel_three_user(taus=N3_TAU_EXT, tag="posthoc-3u-symmetric")
    p3e2 = {"capped": True, "states": {}, "rows": [],
            "capped_at": {"n_m": "skipped"}} if args.skip_3u else \
        panel_three_user(taus=(4,), tag="posthoc-3u-baseline-sw", sw=(0.02, 0.03, 0.03))

    gates = {"g1": gate_g1_identity_engine(), "g4": gate_g4_no_truth(),
             "g5": gate_g5_identity_files(), "g6": gate_g6_first_consistency(),
             "g7": gate_g7_drop_obs(), "g8": gate_g8_distinct_configs(),
             "g2": gate_g2_identity_panel(p["rows"]),
             "g3": gate_g3_reproduce(p["rows"])}
    v = judge(p, gates, refine, p3, refine_free, p3e)

    out = {"round": ROUND, "protocol": PROTOCOL, "utc": _now_utc(),
           "sec_total": round(time.perf_counter() - t0, 3),
           "panel_meta": p["meta"], "cells": p["rows"], "refine": refine,
           "refine_free": refine_free,
           "three_user": p3, "three_user_ext": p3e, "three_user_ext2": p3e2, "gates": gates, "verdicts": v}

    print(json.dumps({k: v[k] for k in ("VERDICT", "B0", "B0_detail", "B1", "B1_detail",
                                        "B1_marginals", "B2", "B3", "B3b", "B4",
                                        "L1", "L1_detail", "price_ceiling",
                                        "gates_n_fail")},
                     indent=2, ensure_ascii=False)[:9000], flush=True)

    dest = Path(args.out) if args.out else (
        RESULTS / f"cw_r4_0_boundary_{_now_utc().replace('-', '').replace(':', '')[:15]}"
                 f"_{_src_hash()[:16]}")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "r40.json").write_text(json.dumps(out, indent=2, ensure_ascii=False),
                                   encoding="utf-8")
    print(f"[evidence] {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
