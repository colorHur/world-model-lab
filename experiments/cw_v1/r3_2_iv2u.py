# -*- coding: utf-8 -*-
"""`CW-R3.2`：**2 用户窗口 DP 的信道信息定价** —— 在"发给谁"这个边际存在的最小实例上，
一次信道探测的毛价值是多少、买不买得起。

预注册：`docs/research/CW-R3.2-2用户窗口DP-预注册-v0.1.md`（写于 2026-10-26 04:0x，**跑前**）

为什么是这一轮
--------------
`CW-R3.1` 判 `IV_free ≡ 0`（1 用户 20 格里 19 格**逐位 0**），但同轮就把话说死了：
1 用户里 `p̂` 唯一能翻的决策是"发不发"，而 `SEND` **弱支配** `HOLD` ⇒ 探测永不改变动作。
**2 用户还有"发给谁"**（R1.15 实测 `E_user` 在 `ep101` 占 91.9%）。本轮只问这一件事。

信息模型（★★ 本文件最硬的边界，预注册 §2.3）
--------------------------------------------
**窗口内 `ACK/NACK` 不进入信道信念**：`SEND` 的成败只改任务侧 `P_i`，不改 `(m_i, v_i)`。
理由：R3.1 `J-B` 已证 `ack_mode="exact"` 下 `(m,v)` **不是**充分统计量 ⇒ 要合法用 ACK
就得携带整条 321 点网格后验，2 用户 × τ=4 不可算。关掉 ACK 之后（高斯先验 + 高斯似然封闭），
`(m,v)` **恰好**是充分统计量 ⇒ 本 DP 除 `m` 的网格外是**精确**的。
⇒ 本轮定价的是"**探测读数**"的信息价值，不是"全部可得信道信息"的价值。

引擎
----
    V_0(·) = 0
    V_t(P0,P1,m0,v0,m1,v1,b) = q·(P0+P1) + min{
        HOLD      : V_{t−1}(f0,f1, m̃0,ṽ0, m̃1,ṽ1, b)
        SEND_i    : p̂_i·V_{t−1}(d_i,f_j, m̃,ṽ, b−1) + (1−p̂_i)·V_{t−1}(f_i,f_j, m̃,ṽ, b−1)
        PROBE_i   : E_{y}[ V_{t−1}(f0,f1, ρ·snap(m_i+g_p(y−m_i)), ṽ'_i, … , b−1) ]
    }
`f_i(P)=A_i²P+σ_w,i²`、`d_i(P)=A_i²(1−g_i)P+σ_w,i²`、`g_i=P/(P+σ_meas²)`；
`m̃=ρ·m`、`ṽ=ρ²v+(1−ρ²)`。`snap` 只在**探测之后**用（与 R3.1 逐字同手法）。

两处**可证的**化简（不是近似，值逐位不变）：
  ① `t=1`：任何动作的后续值都是 `V_0=0` ⇒ `V_1 = q·(P0+P1)` 直接返回；
  ② `t=1` 时**不求值** `PROBE` 分支（探测后已无槽可用，其值与 `HOLD` 逐位相同）。

纪律
----
1. **零 episode、零训练、零参数更新**，纯 CPU。
2. **被测物一字节不动**：只 `import` 既有模块，不改任何 `plan_*/world/config/...`。
3. 阈值全部来自预注册 §4/§6；`judge()` 只按表取。
4. ★ 本文件**不**读任何真值字段；闸 `G9` 用 AST 钉住。
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import time
import tracemalloc
from functools import lru_cache
from pathlib import Path

import numpy as np

from .config import mainline_config
from .posterior import p_succ_expected
from .r1_14_exactbase import _now_utc, _src_hash

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results" / "cw_v1"

PROTOCOL = "CW-POMDP-v1.1"
ROUND = "CW-R3.2"

#: 预算档（预注册 §2.5，跑前写死；**不按结果移动**）
B_OF_TAU: dict[int, tuple[int, ...]] = {1: (1,), 3: (1, 2, 3), 4: (1, 2, 3, 4)}
#: `m` 量化阶梯（预注册 §2.4：τ=3 到 65，τ=4 只到 33）
N_M_TAU3: tuple[int, ...] = (9, 17, 33, 65, 129)
N_M_TAU4: tuple[int, ...] = (9, 17, 33, 65, 129)
#: 起始估计协方差（R3.0 实测面板稳态 `P ∈ [0.005,0.05]`；`prior_var=1.0` 只是 t=0 冷启动）
P_START: tuple[float, ...] = (0.005, 0.02, 0.05, 1.0)
#: ★ 判据单元只用**稳态档**（`P ≤ 0.05`）；冷启动 `P=1.0` 只报描述
P_JUDGE_MAX = 0.05
#: `σ_csi` 阶梯 + 零信息对照
SIGMA_LIST: tuple[float, ...] = (0.05, 2.0, 50.0)
SIGMA_NULL = 1.0e6
#: 判据单元的 `σ_csi`
SIGMA_JUDGE = 0.05
GH_N = 21
GH_N_CHECK = 41
M_RANGE = 4.0
#: ★ 判据/地板（预注册 §4，沿用 R3.1）
IV_NOISE = 1e-12
IV_TOL = 0.10
TRAC_N = 1e7
TRAC_GIB = 4.0
TRAC_SEC = 600.0
#: 状态上限（超了就 `capped`，该引擎的判词不许用）
STATE_CAP = 6_000_000

_SQRT2PI = math.sqrt(2.0 * math.pi)

#: `BOUNDED` plant 的 A≡1 是本轮精确递推的前提
assert all(abs(a - 1.0) < 1e-15 for a in mainline_config().plant.A), \
    "本轮精确 DP 只在 A≡1 上成立"


# ==================================================================== 引擎
def build_engine(cfg, sigma_csi: float, n_m: int, gh_n: int = GH_N, *,
                 probe_enabled: bool = True, probe_cost: int = 1,
                 drop_obs: bool = False, n_users: int = 2, tau_max: int = 4):
    """返回 `(value, n_states, capped)`。

    · `probe_enabled=False` ⇒ 动作集退回 `{HOLD, SEND_i}`（参照臂）；
    · `probe_cost=0` ⇒ **毛价值**口径（探测不花预算，但仍占一个槽）；
    · `drop_obs=True` ⇒ **变异**：探测读数被丢弃（`m'←m`，别的都不动），闸 `G6` 用；
    · `n_users=1` ⇒ 1 用户约化，闸 `G1` 与 R3.1 引擎逐位对账。
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

    @lru_cache(maxsize=None)
    def snap(m: float) -> float:
        return centers_l[int(np.argmin(np.abs(centers - float(m))))]

    @lru_cache(maxsize=None)
    def ps(m: float, v: float) -> float:
        return p_succ_expected(float(m), float(v), cfg)

    memo: dict[tuple, float] = {}

    def V(t: int, P0: float, P1: float, m0: float, v0: float,
          m1: float, v1: float, b: int) -> float:
        if t <= 0:
            return 0.0
        key = (t, P0, P1, m0, v0, m1, v1, b)
        got = memo.get(key)
        if got is not None:
            return got
        S = q * (P0 + P1) if nu == 2 else q * P0
        if t == 1:                       # 化简①：所有动作后续值都是 0
            memo[key] = S
            return S
        if len(memo) > STATE_CAP:
            raise MemoryError(f"state_cap exceeded ({STATE_CAP})")

        # 传播：`P` 与信道信念各走一步 AR
        f0 = a2[0] * P0 + sw2[0]
        f1 = (a2[1] * P1 + sw2[1]) if nu == 2 else 0.0
        mm0 = rho * m0
        vv0 = rho2 * v0 + infl
        mm1 = rho * m1
        vv1 = rho2 * v1 + infl

        # HOLD
        best = V(t - 1, f0, f1, mm0, vv0, mm1, vv1, b)

        # SEND_i
        if b >= 1:
            for i in (0, 1):
                if i == 1 and nu == 1:
                    continue
                Pi = P0 if i == 0 else P1
                g = Pi / (Pi + sm2)
                ps_ = a2[i] * (1.0 - g) * Pi + sw2[i]
                pf_ = a2[i] * Pi + sw2[i]
                ph = ps(m0 if i == 0 else m1, v0 if i == 0 else v1)
                if i == 0:
                    va = ph * V(t - 1, ps_, f1, mm0, vv0, mm1, vv1, b - 1)
                    vb = (1.0 - ph) * V(t - 1, pf_, f1, mm0, vv0, mm1, vv1, b - 1)
                else:
                    va = ph * V(t - 1, f0, ps_, mm0, vv0, mm1, vv1, b - 1)
                    vb = (1.0 - ph) * V(t - 1, f0, pf_, mm0, vv0, mm1, vv1, b - 1)
                val = va + vb
                if val < best:
                    best = val

        # PROBE_i（化简②：t=1 不求值）
        if probe_enabled and b >= pc and t >= 2:
            for i in (0, 1):
                if i == 1 and nu == 1:
                    continue
                mi = m0 if i == 0 else m1
                vi = v0 if i == 0 else v1
                if drop_obs:                     # 变异：读数被丢弃（信念不动，只花预算）
                    if i == 0:
                        tot = V(t - 1, f0, f1, rho * mi, rho2 * vi + infl,
                                mm1, vv1, b - pc)
                    else:
                        tot = V(t - 1, f0, f1, mm0, vv0, rho * mi,
                                rho2 * vi + infl, b - pc)
                else:
                    gp = vi / (vi + si2)
                    vp = rho2 * (vi * si2 / (vi + si2)) + infl
                    sy = math.sqrt(vi + si2)
                    tot = 0.0
                    if i == 0:
                        for z, w in zip(nodes, weights):
                            mp = rho * snap(mi + gp * sy * z)
                            tot += w * V(t - 1, f0, f1, mp, vp, mm1, vv1, b - pc)
                    else:
                        for z, w in zip(nodes, weights):
                            mp = rho * snap(mi + gp * sy * z)
                            tot += w * V(t - 1, f0, f1, mm0, vv0, mp, vp, b - pc)
                    tot /= _SQRT2PI
                if tot < best:
                    best = tot

        out = S + best
        memo[key] = out
        return out

    state = {"capped": False}

    def first(t, P0, P1, m0, v0, m1, v1, b) -> dict:
        """首步各**动作的值**（`V` 的同一展开，只多留一份候选表）。

        ★ 只用于**机制量**（探测是否真的被用）：判词不看它，但"探测有没有改变动作"
          必须能看到，否则 `IV_free>0` 只能说"值变了"、说不出"改了什么"。
        """
        t = int(t)
        P0, P1, m0, v0, m1, v1, b = (float(P0), float(P1), float(m0), float(v0),
                                     float(m1), float(v1), int(b))
        S = q * (P0 + P1) if nu == 2 else q * P0
        f0 = a2[0] * P0 + sw2[0]
        f1 = (a2[1] * P1 + sw2[1]) if nu == 2 else 0.0
        mm0, vv0 = rho * m0, rho2 * v0 + infl
        mm1, vv1 = rho * m1, rho2 * v1 + infl
        out: dict[str, float] = {"HOLD": S + V(t - 1, f0, f1, mm0, vv0, mm1, vv1, b)}
        for i in (0, 1):
            if i == 1 and nu == 1:
                continue
            Pi = P0 if i == 0 else P1
            g = Pi / (Pi + sm2)
            ps_ = a2[i] * (1.0 - g) * Pi + sw2[i]
            pf_ = a2[i] * Pi + sw2[i]
            ph = ps(m0 if i == 0 else m1, v0 if i == 0 else v1)
            if b >= 1:
                if i == 0:
                    out["SEND_0"] = S + ph * V(t - 1, ps_, f1, mm0, vv0, mm1, vv1, b - 1) \
                        + (1.0 - ph) * V(t - 1, pf_, f1, mm0, vv0, mm1, vv1, b - 1)
                else:
                    out["SEND_1"] = S + ph * V(t - 1, f0, ps_, mm0, vv0, mm1, vv1, b - 1) \
                        + (1.0 - ph) * V(t - 1, f0, pf_, mm0, vv0, mm1, vv1, b - 1)
            if probe_enabled and b >= pc and t >= 2:
                mi = m0 if i == 0 else m1
                vi = v0 if i == 0 else v1
                gp = vi / (vi + si2)
                vp = rho2 * (vi * si2 / (vi + si2)) + infl
                sy = math.sqrt(vi + si2)
                tot = 0.0
                if i == 0:
                    for z, w in zip(nodes, weights):
                        tot += w * V(t - 1, f0, f1, rho * snap(mi + gp * sy * z), vp,
                                     mm1, vv1, b - pc)
                else:
                    for z, w in zip(nodes, weights):
                        tot += w * V(t - 1, f0, f1, mm0, vv0, rho * snap(mi + gp * sy * z),
                                     vp, b - pc)
                out[f"PROBE_{i}"] = S + tot / _SQRT2PI
        return out

    def value(t, P0, P1, m0, v0, m1, v1, b) -> float:
        try:
            return float(V(int(t), float(P0), float(P1), float(m0), float(v0),
                           float(m1), float(v1), int(b)))
        except MemoryError:
            state["capped"] = True
            return float("nan")

    def n_states() -> int:
        return len(memo)

    value.n_states = n_states              # type: ignore[attr-defined]
    value.capped = lambda: state["capped"]  # type: ignore[attr-defined]
    value.first = first                    # type: ignore[attr-defined]
    return value


# ==================================================================== 面板
def panel_main(cfg) -> dict:
    """主面板：`IV_free / IV_net / price` 的三量对照（每个 `n_m`/`σ` 建一次引擎）。"""
    out: dict = {"cells": [], "meta": [], "sec": 0.0}
    t_all = time.perf_counter()

    for n_m in N_M_TAU4:
        t0 = time.perf_counter()
        v_no = build_engine(cfg, SIGMA_LIST[0], n_m, probe_enabled=False)
        v_fr = build_engine(cfg, SIGMA_LIST[0], n_m, probe_cost=0)
        v_pd = build_engine(cfg, SIGMA_LIST[0], n_m, probe_cost=1)
        for tau in (1, 3, 4):
            for b0 in B_OF_TAU[tau]:
                for p0 in P_START:
                    a = v_no(tau, p0, p0, 0.0, 1.0, 0.0, 1.0, b0)
                    f = v_fr(tau, p0, p0, 0.0, 1.0, 0.0, 1.0, b0)
                    d = v_pd(tau, p0, p0, 0.0, 1.0, 0.0, 1.0, b0)
                    pr = (v_no(tau, p0, p0, 0.0, 1.0, 0.0, 1.0, b0 - 1) - a
                          if b0 >= 1 else 0.0)
                    out["cells"].append({
                        "n_m": n_m, "sigma_csi": SIGMA_LIST[0], "tau": tau, "b0": b0,
                        "P0": p0, "m0": 0.0, "v0": 1.0,
                        "V_noprobe": a, "V_probe_free": f, "V_probe_paid": d,
                        "IV_free": a - f, "IV_net": a - d, "price": pr})
        # ★ 状态数必须在**查询之后**取（引擎是惰性递归）：首版在查询前取 ⇒ 全是 0。
        out["meta"].append({"n_m": n_m, "sigma_csi": SIGMA_LIST[0],
                            "n_states": {"noprobe": v_no.n_states(),
                                         "free": v_fr.n_states(),
                                         "paid": v_pd.n_states()},
                            "capped": bool(v_no.capped() or v_fr.capped() or v_pd.capped()),
                            "sec": round(time.perf_counter() - t0, 3)})
        print(f"[MAIN] n_m={n_m:>3} states={out['meta'][-1]['n_states']} "
              f"({out['meta'][-1]['sec']}s)", flush=True)

    # ---- σ 阶梯（只在 `n_m=33` 上做，判据单元同格）
    n_m = N_M_TAU4[-1]
    for sig in SIGMA_LIST[1:] + (SIGMA_NULL,):
        v_fr = build_engine(cfg, sig, n_m, probe_cost=0)
        for tau in (3, 4):
            for b0 in B_OF_TAU[tau]:
                for p0 in P_START:
                    a = next(c["V_noprobe"] for c in out["cells"]
                             if c["n_m"] == n_m and c["tau"] == tau
                             and c["b0"] == b0 and c["P0"] == p0)
                    f = v_fr(tau, p0, p0, 0.0, 1.0, 0.0, 1.0, b0)
                    out["cells"].append({
                        "n_m": n_m, "sigma_csi": sig, "tau": tau, "b0": b0,
                        "P0": p0, "m0": 0.0, "v0": 1.0,
                        "V_noprobe": a, "V_probe_free": f, "V_probe_paid": None,
                        "IV_free": a - f, "IV_net": None, "price": None})
        print(f"[MAIN] sigma={sig:<8} done ({time.perf_counter() - t_all:.1f}s)", flush=True)

    # ---- 非对称先验扫描（`Δm = m0 − m1`）：边际住在哪里
    # ★ 同时报 `alloc_gap = |V(SEND_0) − V(SEND_1)|`（根状态「发给谁」这个决策**值多少**）：
    #   它是 `IV_free` 的**上界形态**（信息最多只能把这笔差价赚回来），
    #   两个量并列就能回答"是没边际，还是有边际但不值钱"。
    v_no = build_engine(cfg, SIGMA_JUDGE, n_m, probe_enabled=False)
    v_fr = build_engine(cfg, SIGMA_JUDGE, n_m, probe_cost=0)
    out["dm_sweep"] = []
    for dm in (0.0, 0.5, 1.0, 2.0, 4.0):
        for p0 in (0.02, 1.0):
            a = v_no(4, p0, p0, dm, 1.0, 0.0, 1.0, 2)
            f = v_fr(4, p0, p0, dm, 1.0, 0.0, 1.0, 2)
            fv = v_no.first(4, p0, p0, dm, 1.0, 0.0, 1.0, 2)
            out["dm_sweep"].append({"dm": dm, "P0": p0, "V_noprobe": a,
                                    "V_probe_free": f, "IV_free": a - f,
                                    "alloc_gap": float(abs(fv["SEND_0"] - fv["SEND_1"])),
                                    "act": min(fv, key=lambda k: fv[k])})
    # `v` 也不对称的一格（`v0≠v1`），给 `G2` 的活体破坏用
    out["asym_v"] = {"V": v_no(4, 0.02, 0.02, 0.0, 0.5, 0.0, 1.0, 2)}

    # ---- 网格收敛（判据单元的 `n_m` 阶梯已在上面，这里配对）
    out["sec"] = round(time.perf_counter() - t_all, 3)
    return out


# ==================================================================== 暴力枚举（G4）
def brute_force(cfg, sigma_csi: float, n_m: int, tau: int, b0: int,
                gh_n: int = GH_N) -> float:
    """显式枚举**动作序列 × 结果树**，取最小期望成本。**不复用** `build_engine` 的递归。"""
    from itertools import product

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
    centers = np.linspace(-M_RANGE, M_RANGE, int(n_m))

    def snap(m: float) -> float:
        return float(centers[int(np.argmin(np.abs(centers - float(m))))])

    acts = ("H", "S0", "S1", "P0", "P1")

    def run(seq: tuple[str, ...], st: tuple) -> float:
        P0, P1, m0, v0, m1, v1, b = st
        if len(seq) == 0:
            return 0.0
        act, rest = seq[0], seq[1:]
        S = q * (P0 + P1)
        f0 = a2[0] * P0 + sw2[0]
        f1 = a2[1] * P1 + sw2[1]
        mm0, vv0 = rho * m0, rho2 * v0 + infl
        mm1, vv1 = rho * m1, rho2 * v1 + infl
        if act == "H" or (act[0] in "SP" and b < 1):
            return S + run(rest, (f0, f1, mm0, vv0, mm1, vv1, b))
        if act[0] == "S":
            i = int(act[1])
            Pi = P0 if i == 0 else P1
            g = Pi / (Pi + sm2)
            ph = p_succ_expected(m0 if i == 0 else m1, v0 if i == 0 else v1, cfg)
            ps_ = a2[i] * (1.0 - g) * Pi + sw2[i]
            pf_ = a2[i] * Pi + sw2[i]
            if i == 0:
                return S + ph * run(rest, (ps_, f1, mm0, vv0, mm1, vv1, b - 1)) \
                    + (1.0 - ph) * run(rest, (pf_, f1, mm0, vv0, mm1, vv1, b - 1))
            return S + ph * run(rest, (f0, ps_, mm0, vv0, mm1, vv1, b - 1)) \
                + (1.0 - ph) * run(rest, (f0, pf_, mm0, vv0, mm1, vv1, b - 1))
        i = int(act[1])
        mi = m0 if i == 0 else m1
        vi = v0 if i == 0 else v1
        gp = vi / (vi + si2)
        vp = vi * si2 / (vi + si2)
        sy = math.sqrt(vi + si2)
        tot = 0.0
        for z, w in zip(nodes, weights):
            mp = snap(mi + gp * (sy * float(z)))
            if i == 0:
                tot += float(w) * run(rest, (f0, f1, rho * mp, rho2 * vp + infl,
                                            mm1, vv1, b - 1))
            else:
                tot += float(w) * run(rest, (f0, f1, mm0, vv0, rho * mp,
                                            rho2 * vp + infl, b - 1))
        return S + tot / _SQRT2PI

    best = math.inf
    for seq in product(acts, repeat=int(tau)):
        n_q = sum(1 for a in seq if a[0] in "SP")
        if n_q > int(b0):
            continue
        val = run(seq, (1.0, 1.0, 0.0, 1.0, 0.0, 1.0, int(b0)))
        if val < best:
            best = val
    return float(best)


# ==================================================================== 判词
def judge(main: dict, gates: dict) -> dict:
    out: dict = {}
    cells = main["cells"]
    n_m_hi = N_M_TAU4[-1]

    # ---- J-A 规模
    meta_hi = [m for m in main["meta"] if m["n_m"] == n_m_hi]
    if not meta_hi or meta_hi[0]["capped"]:
        out["J-A"] = "SCALE-2U-MARGINAL"
        out["J-A_note"] = "n_m=33 触到 state_cap 或未跑"
    else:
        n_st = max(meta_hi[0]["n_states"].values())
        sec = max(mm["sec"] for mm in main["meta"])
        if n_st <= TRAC_N and sec <= TRAC_SEC:
            out["J-A"] = "SCALE-2U-TRACTABLE"
        elif n_st <= 1e9:
            out["J-A"] = "SCALE-2U-MARGINAL"
        else:
            out["J-A"] = "SCALE-2U-EXPLODES"
        out["J-A_states"] = int(n_st)
        out["J-A_sec"] = float(sec)

    # ---- 判据单元：`n_m=33`、`σ=0.05`、`P ≤ 0.05`、`b0` 紧（b0 < τ）
    judged = [c for c in cells if c["n_m"] == n_m_hi and c["sigma_csi"] == SIGMA_JUDGE
              and c["P0"] <= P_JUDGE_MAX and c["b0"] < c["tau"] and c["tau"] >= 3]
    pos = [c for c in judged if c["IV_free"] > IV_NOISE]
    out["n_judged_cells"] = len(judged)
    out["n_positive_IV_free"] = len(pos)
    out["J-B"] = "IV-2U-ABOVE-NOISE" if pos else "IV-2U-BELOW-NOISE"

    # ---- J-C 网格分辨率（对每个 `IV_free>0` 的 (τ,b0,P0) 找最小 n_m*）
    # ★★ 跑后修订 §7-B1：原实现只要求**某一格**落进 `10%`（"单点达标"），
    #   而阶梯可能整体在 ±50% 里震荡 ⇒ 那会把"没有平台"误判成"分辨出来了"（本机上真发生了：
    #   `n_m=33` 单点落进 10%，但 `129` 又差 16%）。改正为 **平台判据**：
    #   存在 `n_m*` 使得**所有** `n_m ≥ n_m*` 与 `IV(n_m*)` 的相对差都 ≤ `IV_TOL`。
    need: dict = {}
    ladder: dict = {}
    for c in pos:
        key = f"t{c['tau']}_b{c['b0']}_P{c['P0']}"
        seq = []
        for nm in N_M_TAU4:
            g = next((x for x in cells if x["n_m"] == nm and x["tau"] == c["tau"]
                      and x["b0"] == c["b0"] and x["P0"] == c["P0"]
                      and x["sigma_csi"] == SIGMA_JUDGE), None)
            seq.append(None if g is None else g["IV_free"])
        ladder[key] = [None if v is None else float(v) for v in seq]
        cand = None
        # ★ 平台至少要**两格**才能自证（最细的一格自己跟自己比恒过，那是最廉价的假阳性）
        for k, nm in enumerate(N_M_TAU4[:-1]):
            if seq[k] is None or seq[k] <= IV_NOISE:
                continue
            base = seq[k]
            tail = seq[k:]
            if all(v is not None and abs(v - base) <= IV_TOL * abs(base) for v in tail):
                cand = nm
                break
        need[key] = cand
    out["n_m_star"] = need
    out["IV_free_ladder"] = ladder
    if not pos:
        out["J-C"] = "IV-2U-NO-MARGIN"
    elif all(v is not None for v in need.values()):
        out["J-C"] = "IV-2U-RESOLVABLE"
    else:
        out["J-C"] = "IV-2U-UNRESOLVED-BY-GRID"

    # ---- J-D 可负担性
    net_pos = [c for c in pos if (c["IV_net"] is not None and c["IV_net"] > IV_NOISE
                                 and c["IV_free"] >= c["price"])]
    if not pos:
        out["J-D"] = "IV-2U-NO-MARGIN"
    elif net_pos:
        out["J-D"] = "IV-2U-AFFORDABLE"
    else:
        out["J-D"] = "IV-2U-GROSS-ONLY"
    # 比率（只描述，不承载判词）
    out["ratio_IVfree_over_price"] = [
        {"tau": c["tau"], "b0": c["b0"], "P0": c["P0"],
         "IV_free": c["IV_free"], "price": c["price"],
         "ratio": (c["IV_free"] / c["price"] if c["price"] > 0 else None),
         "IV_net": c["IV_net"]} for c in pos]

    # ---- J-E 根状态是否并列（对称先验、禁探测）
    out["J-E"] = gates.get("J-E", "NOT-RUN")
    return out


# ==================================================================== 闸
def gate_g1(cfg) -> dict:
    """`G1`：1 用户约化 vs R3.1 引擎（双向：`probe_cost ∈ {0,1}`）。"""
    from .r3_1_scale import build_dp as _dp31

    rows = []
    worst = 0.0
    for n_m in (9, 33):
        for pcost in (0, 1):
            mine = build_engine(cfg, SIGMA_JUDGE, n_m, probe_cost=pcost, n_users=1)
            theirs, _ = _dp31(cfg, SIGMA_JUDGE, n_m, GH_N, probe_cost=pcost)
            for tau in (1, 2, 3, 4):
                for p0 in (0.02, 1.0):
                    a = mine(tau, p0, 0.0, 0.0, 1.0, 0.0, 1.0, max(1, tau // 2))
                    b = theirs(tau, p0, 0.0, 1.0, max(1, tau // 2))
                    d = abs(a - b)
                    worst = max(worst, d)
                    rows.append({"n_m": n_m, "probe_cost": pcost, "tau": tau,
                                 "P0": p0, "mine": a, "r3_1": b, "abs": d})
    return {"rows": rows, "max_abs": float(worst), "tol": 1e-12,
            "pass": bool(worst <= 1e-12)}


def gate_g2(cfg) -> dict:
    """`G2`：**用户轴一致性**（对称 plant 上的交换不变 + 非对称 plant 上的活体破坏）。

    ★ 首版在**非对称** plant（`sigma_w=(0.02,0.03)`）上直接判"交换不变" ⇒ 必然 FAIL：
      两个用户的噪声不同，"只换状态不换 plant"根本不是对称操作。**判据缺陷，不是引擎错**。
      正确的两条腿：
        ① **对称 plant**（`sigma_w` 取同值）：交换 `(P0,P1,m0,v0,m1,v1)` ⇒ `V` 必须逐位不变
           —— 这条正好抓 R1.7.1 那种**轴错配**（把用户 i 的映射作用到用户 j 的 `P` 上）；
        ② **非对称 plant**：同一个交换**必须变**（活体破坏，证明①不是恒真）。
    """
    from .config import bounded_plant

    cfg_sym = mainline_config(plant=bounded_plant(sigma_w=(0.02, 0.02)))
    vs = build_engine(cfg_sym, SIGMA_JUDGE, 17, probe_cost=1)
    syms = []
    worst = 0.0
    for (P0, P1, m0, v0, m1, v1) in ((0.02, 0.05, 0.3, 0.8, -0.4, 1.0),
                                     (0.01, 0.30, -1.1, 0.4, 0.9, 1.6),
                                     (0.04, 0.04, 1.0, 1.0, -1.0, 1.0)):
        a = vs(4, P0, P1, m0, v0, m1, v1, 2)
        b = vs(4, P1, P0, m1, v1, m0, v0, 2)
        d = abs(a - b)
        worst = max(worst, d)
        syms.append({"P0": P0, "P1": P1, "a": a, "b": b, "abs": float(d)})

    va = build_engine(cfg, SIGMA_JUDGE, 17, probe_cost=1)
    a2 = va(4, 0.02, 0.05, 0.3, 0.8, -0.4, 1.0, 2)
    brk = va(4, 0.05, 0.02, -0.4, 1.0, 0.3, 0.8, 2)
    brk_abs = abs(a2 - brk)
    return {"sym_cells": syms, "sym_max_abs": float(worst), "sym_tol": 1e-15,
            "live_break_abs": float(brk_abs),
            "live_break_ok": bool(brk_abs > 1e-9),
            "pass": bool(worst <= 1e-15 and brk_abs > 1e-9)}


def gate_g3(cfg) -> dict:
    """`G3`：`V(b)` 对 `b` 非增；`b=0` == HOLD-only 闭式。"""
    v = build_engine(cfg, SIGMA_JUDGE, 17, probe_cost=1)
    rows = []
    ok_mono = True
    ok_closed = True
    for tau in (1, 3, 4):
        for p0 in (0.005, 0.02, 1.0):
            seq = [v(tau, p0, p0, 0.0, 1.0, 0.0, 1.0, b) for b in range(0, tau + 1)]
            # ★ `V` 对 `b` **非增**：违反 = 多给预算反而更差（`seq[i+1] > seq[i]`）。
            #   首版把方向写反（数"下降"当上升）—— 与 R3.1 `price` 差向写反同族，记入自伤。
            rises = sum(1 for i in range(len(seq) - 1) if seq[i + 1] > seq[i] + 1e-15)
            closed = 0.0
            for k in range(tau):
                closed += cfg.plant.q_cost * (p0 + k * cfg.plant.sigma_w[0] ** 2
                                             + p0 + k * cfg.plant.sigma_w[1] ** 2)
            d0 = abs(seq[0] - closed)
            rows.append({"tau": tau, "P0": p0, "seq": seq, "rises": int(rises),
                         "closed": closed, "abs_b0": d0})
            ok_mono &= (rises == 0)
            ok_closed &= (d0 <= 1e-12)
    return {"rows": rows, "mono_ok": bool(ok_mono), "closed_ok": bool(ok_closed),
            "pass": bool(ok_mono and ok_closed)}


def gate_g3b(cfg) -> dict:
    """`G3b`：`IV_free(τ=1) ≡ 0`（恒零控）。"""
    v_no = build_engine(cfg, SIGMA_JUDGE, 17, probe_enabled=False)
    v_fr = build_engine(cfg, SIGMA_JUDGE, 17, probe_cost=0)
    vals = []
    for p0 in (0.005, 0.02, 1.0):
        vals.append(float(v_no(1, p0, p0, 0.0, 1.0, 0.0, 1.0, 1)
                           - v_fr(1, p0, p0, 0.0, 1.0, 0.0, 1.0, 1)))
    return {"vals": vals, "pass": bool(max(abs(x) for x in vals) == 0.0)}


def gate_g4(cfg) -> dict:
    """`G4`：独立暴力枚举（不复用引擎递归）。"""
    rows = []
    worst = 0.0
    for b0 in (1, 2):
        for pcost in (0, 1):
            v = build_engine(cfg, SIGMA_JUDGE, 9, probe_cost=pcost)
            a = v(3, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, b0)
            # 暴力枚举里 `b0` 是总量子数；`pcost=0` 口径（探测免费但仍占槽）另算
            b = (brute_force(cfg, SIGMA_JUDGE, 9, 3, b0) if pcost == 1
                 else brute_force_free(cfg, SIGMA_JUDGE, 9, 3, b0))
            d = abs(a - b)
            worst = max(worst, d)
            rows.append({"b0": b0, "probe_cost": pcost, "dp": a, "brute": b, "abs": d})
    return {"rows": rows, "max_abs": float(worst), "tol": 1e-12,
            "pass": bool(worst <= 1e-12)}


def brute_force_free(cfg, sigma_csi: float, n_m: int, tau: int, b0: int,
                     gh_n: int = GH_N) -> float:
    """`probe_cost=0` 口径的暴力枚举（探测不花量子，但仍占一个槽）。"""
    from itertools import product

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
    centers = np.linspace(-M_RANGE, M_RANGE, int(n_m))

    def snap(m: float) -> float:
        return float(centers[int(np.argmin(np.abs(centers - float(m))))])

    acts = ("H", "S0", "S1", "P0", "P1")

    def run(seq: tuple[str, ...], st: tuple) -> float:
        P0, P1, m0, v0, m1, v1, b = st
        if not seq:
            return 0.0
        act, rest = seq[0], seq[1:]
        S = q * (P0 + P1)
        f0 = a2[0] * P0 + sw2[0]
        f1 = a2[1] * P1 + sw2[1]
        mm0, vv0 = rho * m0, rho2 * v0 + infl
        mm1, vv1 = rho * m1, rho2 * v1 + infl
        if act == "H":
            return S + run(rest, (f0, f1, mm0, vv0, mm1, vv1, b))
        if act[0] == "S":
            if b < 1:
                return S + run(rest, (f0, f1, mm0, vv0, mm1, vv1, b))
            i = int(act[1])
            Pi = P0 if i == 0 else P1
            g = Pi / (Pi + sm2)
            ph = p_succ_expected(m0 if i == 0 else m1, v0 if i == 0 else v1, cfg)
            ps_ = a2[i] * (1.0 - g) * Pi + sw2[i]
            pf_ = a2[i] * Pi + sw2[i]
            if i == 0:
                return S + ph * run(rest, (ps_, f1, mm0, vv0, mm1, vv1, b - 1)) \
                    + (1.0 - ph) * run(rest, (pf_, f1, mm0, vv0, mm1, vv1, b - 1))
            return S + ph * run(rest, (f0, ps_, mm0, vv0, mm1, vv1, b - 1)) \
                + (1.0 - ph) * run(rest, (f0, pf_, mm0, vv0, mm1, vv1, b - 1))
        i = int(act[1])
        mi = m0 if i == 0 else m1
        vi = v0 if i == 0 else v1
        gp = vi / (vi + si2)
        vp = vi * si2 / (vi + si2)
        sy = math.sqrt(vi + si2)
        tot = 0.0
        for z, w in zip(nodes, weights):
            mp = snap(mi + gp * (sy * float(z)))
            if i == 0:
                tot += float(w) * run(rest, (f0, f1, rho * mp, rho2 * vp + infl,
                                             mm1, vv1, b))
            else:
                tot += float(w) * run(rest, (f0, f1, mm0, vv0, rho * mp,
                                             rho2 * vp + infl, b))
        return S + tot / _SQRT2PI

    best = math.inf
    for seq in product(acts, repeat=int(tau)):
        n_send = sum(1 for a in seq if a[0] == "S")
        if n_send > int(b0):
            continue
        val = run(seq, (1.0, 1.0, 0.0, 1.0, 0.0, 1.0, int(b0)))
        if val < best:
            best = val
    return float(best)


def gate_g5(cfg) -> dict:
    """`G5`：零信息对照 `σ_csi=1e6` ⇒ `IV_free ≤ IV_NOISE`。"""
    v_no = build_engine(cfg, SIGMA_JUDGE, 17, probe_enabled=False)
    v_nu = build_engine(cfg, SIGMA_NULL, 17, probe_cost=0)
    vals = []
    for tau in (3, 4):
        for p0 in (0.02, 1.0):
            a = v_no(tau, p0, p0, 0.0, 1.0, 0.0, 1.0, min(2, tau - 1))
            f = v_nu(tau, p0, p0, 0.0, 1.0, 0.0, 1.0, min(2, tau - 1))
            vals.append({"tau": tau, "P0": p0, "IV_free_null": float(a - f)})
    worst = max(abs(r["IV_free_null"]) for r in vals)
    return {"rows": vals, "worst": float(worst), "pass": bool(worst <= IV_NOISE)}


def gate_g6(cfg) -> dict:
    """`G6` 变异（信息链）：探测读数被丢弃 ⇒ `V(允许探测) == V(禁探测)`。"""
    v_no = build_engine(cfg, SIGMA_JUDGE, 9, probe_enabled=False)
    v_drop = build_engine(cfg, SIGMA_JUDGE, 9, probe_cost=0, drop_obs=True)
    rows = []
    worst = 0.0
    for tau in (3, 4):
        for p0 in (0.02, 1.0):
            a = v_no(tau, p0, p0, 0.0, 1.0, 0.0, 1.0, 2)
            b = v_drop(tau, p0, p0, 0.0, 1.0, 0.0, 1.0, 2)
            worst = max(worst, abs(a - b))
            rows.append({"tau": tau, "P0": p0, "noprobe": a, "drop": b, "abs": abs(a - b)})
    return {"rows": rows, "max_abs": float(worst), "pass": bool(worst <= 1e-15)}


def gate_g7(cells: list[dict], n_m: int) -> dict:
    """`G7`：`IV_free` 对 `σ_csi` 单调不增。"""
    bad = []
    n_cell = 0
    for tau in (3, 4):
        for b0 in B_OF_TAU[tau]:
            for p0 in P_START:
                seq = []
                for sig in SIGMA_LIST:
                    g = next((c for c in cells if c["n_m"] == n_m and c["sigma_csi"] == sig
                              and c["tau"] == tau and c["b0"] == b0 and c["P0"] == p0), None)
                    seq.append(None if g is None else g["IV_free"])
                if any(x is None for x in seq):
                    continue
                n_cell += 1
                for i in range(len(seq) - 1):
                    if seq[i + 1] > seq[i] + 1e-12:
                        bad.append({"tau": tau, "b0": b0, "P0": p0, "seq": seq})
    return {"n_cell": int(n_cell), "bad": bad, "pass": bool(not bad)}


def gate_g8() -> dict:
    """`G8` 变异测试（闸的闸）：守卫换成恒放行 ⇒ 3/3 必须转红。"""
    cases = []

    def guard_sym(d: float) -> bool:
        return abs(d) <= 1e-15

    def guard_mono(rises: int) -> bool:
        return rises == 0

    def guard_null(worst: float) -> bool:
        return worst <= IV_NOISE

    known_bad = [(guard_sym, 1e-3), (guard_mono, 2), (guard_null, 1e-3)]
    for g, bad_val in known_bad:
        real = bool(g(bad_val))
        mut = bool((lambda *a, **k: True)(bad_val))
        cases.append({"guard": g.__name__, "known_bad": bad_val,
                      "real_rejects": (not real), "mutant_accepts": mut,
                      "caught": bool((not real) and mut)})
    return {"cases": cases, "n_caught": sum(1 for c in cases if c["caught"]),
            "pass": bool(all(c["caught"] for c in cases))}


def gate_je(cfg) -> dict:
    """`J-E`（§7-A1 跑前修订后的形态）：**"发给谁"是不是真的自由度**。

    原卡写的是"根状态两侧 `SEND` 的值并列 ⇒ 边际存在"。这条**在非对称 plant 上恒假**
    （`sigma_w=(0.02,0.03)` ⇒ 两条 `SEND` 的值不可能相等），而"边际"靠的是**信道随机性**
    而不是值并列 ⇒ 原判据是**缺陷判据**，跑主面板前改掉（§7-A1）。

    新形态两条腿，都不看并列：
      ① **对称 plant** 上两条 `SEND` 的值必须相等（`≤1e-15`）⇒ 证明"发给谁"确实是一个
         值上无差别、只能靠信息分开的自由度；
      ② **机制**：在判据单元上，把探测设为**免费**后**首步最优动作**是否从 `SEND_*`
         变成 `PROBE_*`（`n_probe_chosen`）⇒ 探测被真正使用的次数。
    """
    from .config import bounded_plant

    cfg_sym = mainline_config(plant=bounded_plant(sigma_w=(0.02, 0.02)))
    v_sym = build_engine(cfg_sym, SIGMA_JUDGE, 17, probe_enabled=False)
    rows = []
    tie = True
    for tau in (3, 4):
        for p0 in (0.02, 1.0):
            fv = v_sym.first(tau, p0, p0, 0.0, 1.0, 0.0, 1.0, 1)
            d = abs(fv["SEND_0"] - fv["SEND_1"])
            rows.append({"tau": tau, "P0": p0, "V_send0": fv["SEND_0"],
                         "V_send1": fv["SEND_1"], "abs": float(d)})
            tie &= (d <= 1e-15)
    # 非对称 plant 上两条 `SEND` 的差（描述量：边际的"大小"由信道跑出来，不由 plant 差出来）
    v_asym = build_engine(cfg, SIGMA_JUDGE, 17, probe_enabled=False)
    gap = []
    for tau in (3, 4):
        for p0 in (0.02, 1.0):
            fv = v_asym.first(tau, p0, p0, 0.0, 1.0, 0.0, 1.0, 1)
            gap.append({"tau": tau, "P0": p0,
                        "d_send": float(abs(fv["SEND_0"] - fv["SEND_1"]))})
    return {"sym_rows": rows, "tie": bool(tie),
            "asym_send_gap": gap,
            "verdict": "TIE-AT-SYMMETRIC-PRIOR" if tie else "NO-TIE"}


def gate_mechanism(cfg) -> dict:
    """机制量：**探测是否真的改变了首步动作**（`IV_free>0` 只说"值变了"，这里说"改了什么"）。"""
    out = []
    n_probe = 0
    n_judged = 0
    for n_m in (N_M_TAU4[-1],):
        v_no = build_engine(cfg, SIGMA_JUDGE, n_m, probe_enabled=False)
        v_fr = build_engine(cfg, SIGMA_JUDGE, n_m, probe_cost=0)
        for tau in (3, 4):
            for b0 in B_OF_TAU[tau]:
                if b0 >= tau:
                    continue
                for p0 in P_START:
                    if p0 > P_JUDGE_MAX:
                        continue
                    a = v_no.first(tau, p0, p0, 0.0, 1.0, 0.0, 1.0, b0)
                    f = v_fr.first(tau, p0, p0, 0.0, 1.0, 0.0, 1.0, b0)
                    act_no = min(a, key=lambda k: a[k])
                    act_fr = min(f, key=lambda k: f[k])
                    n_judged += 1
                    if act_fr.startswith("PROBE"):
                        n_probe += 1
                    out.append({"tau": tau, "b0": b0, "P0": p0,
                                "act_noprobe": act_no, "act_free": act_fr,
                                "gap": float(a[act_no] - f[act_fr]),
                                "first_noprobe": {k: float(v) for k, v in a.items()},
                                "first_free": {k: float(v) for k, v in f.items()}})
    return {"cells": out, "n_judged": int(n_judged), "n_probe_chosen": int(n_probe)}


_TRUTH_NAMES = ("_log" + "_h_true", "_x" + "_true", "cost" + "_true",
                "z" + "_meas", "log_h" + "_history")


def _ast_scan(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in _TRUTH_NAMES:
            hits.append(f"attr:{node.attr}@{node.lineno}")
        if isinstance(node, ast.Name) and node.id in _TRUTH_NAMES:
            hits.append(f"name:{node.id}@{node.lineno}")
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value in _TRUTH_NAMES:
                hits.append(f"str:{node.value}@{node.lineno}")
    return hits


# ==================================================================== 主流程
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="CW-R3.2 2 用户窗口 DP 的信道信息定价")
    ap.add_argument("--ctrl-only", action="store_true", help="只跑闸，不跑主面板")
    ap.add_argument("--skip-g4", action="store_true", help="跳过暴力枚举（调试用）")
    args = ap.parse_args(argv)

    cfg = mainline_config()
    issues = cfg.v11_validate()
    if issues:
        print("[FATAL] 配置自洽检查未过：")
        for s in issues:
            print("  -", s)
        return 2

    run = f"cw_r3_2_iv2u_{_now_utc()}_{_src_hash()}"
    outdir = RESULTS / run
    if not args.ctrl_only:
        outdir.mkdir(parents=True, exist_ok=True)
    print(f"[RUN] {run}\n[DIR] {outdir if not args.ctrl_only else '(ctrl-only，不落盘)'}")

    t0 = time.perf_counter()
    tracemalloc.start()

    gates: dict = {}
    main_panel: dict = {"cells": [], "meta": [], "dm_sweep": [], "asym_v": {}, "sec": 0.0}
    if not args.ctrl_only:
        main_panel = panel_main(cfg)
        (outdir / "cells_main.json").write_text(
            json.dumps(main_panel, ensure_ascii=False, indent=1), encoding="utf-8")

    gates["G1"] = {"name": "1 用户约化 == R3.1 引擎", **gate_g1(cfg)}
    print(f"[G1] max_abs={gates['G1']['max_abs']:.3e} pass={gates['G1']['pass']}", flush=True)
    gates["G2"] = {"name": "对称交换 + 活体破坏", **gate_g2(cfg)}
    gates["G3"] = {"name": "V(b) 非增 + b=0 闭式", **gate_g3(cfg)}
    gates["G3b"] = {"name": "IV_free(tau=1)==0", **gate_g3b(cfg)}
    if not args.skip_g4:
        gates["G4"] = {"name": "独立暴力枚举 == DP", **gate_g4(cfg)}
        print(f"[G4] max_abs={gates['G4']['max_abs']:.3e} pass={gates['G4']['pass']}", flush=True)
    gates["G5"] = {"name": "零信息对照 sigma=1e6 => IV_free<=floor", **gate_g5(cfg)}
    gates["G6"] = {"name": "变异：读数被丢弃 => V==禁探测", **gate_g6(cfg)}
    g7 = (gate_g7(main_panel["cells"], N_M_TAU4[-1]) if not args.ctrl_only
          else {"n_cell": 0, "bad": [], "pass": False})
    gates["G7"] = {"name": "IV_free 对 sigma 单调不增", **g7}
    gates["G8"] = {"name": "变异测试(闸的闸) 3/3", **gate_g8()}
    gates["G9"] = {"name": "GH21 vs GH41", **_gate_g9(cfg)}
    hits = _ast_scan(Path(__file__))
    gates["G9b"] = {"name": "AST 不读真值字段", "hits": hits, "pass": bool(not hits)}
    je = gate_je(cfg)
    gates["J-E"] = je["verdict"]
    mech = gate_mechanism(cfg) if not args.ctrl_only else {
        "cells": [], "n_judged": 0, "n_probe_chosen": 0}
    if not args.ctrl_only:
        (outdir / "mech.json").write_text(
            json.dumps({"je": je, "mechanism": mech}, ensure_ascii=False, indent=1),
            encoding="utf-8")
    gates["G11"] = {"name": "活体单调性 price>=0 且 IV_free>=0", **_gate_g11(main_panel)}

    try:
        from .gates import run_all as _run_all
        reg = _run_all()
        reg_n, n_fail_reg = int(reg["n_checks"]), int(reg["n_fail"])
    except Exception as e:                        # pragma: no cover
        reg_n, n_fail_reg = -1, -1
        print(f"[G10] 回归闸异常：{type(e).__name__}: {e}")
    gates["G10"] = {"name": "gates.run_all n_fail==0", "n_total": reg_n,
                    "n_fail": n_fail_reg, "pass": bool(n_fail_reg == 0)}

    verdict = judge(main_panel, gates) if not args.ctrl_only else {
        "J-A": "NOT-RUN", "J-B": "NOT-RUN", "J-C": "NOT-RUN", "J-D": "NOT-RUN",
        "J-E": gates.get("J-E")}
    cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    n_fail_all = sum(1 for k, v in gates.items()
                     if isinstance(v, dict) and not v.get("pass", True))

    summary = {
        "round": ROUND, "protocol": PROTOCOL, "src_hash": _src_hash(),
        "cfg": {"plant_A": list(cfg.plant.A), "sigma_w": list(cfg.plant.sigma_w),
                "sigma_meas": cfg.plant.sigma_meas, "q_cost": cfg.plant.q_cost,
                "rho": cfg.channel.rho_panel[0], "kappa": cfg.channel.kappa,
                "ack_mode": cfg.channel.ack_mode},
        "info_model": ("probe-only channel belief (ACK/NACK excluded, see pre-reg §2.3); "
                       "moment state (m,v) is sufficient under this restriction"),
        "run_episode": 0, "train_steps": 0,
        "main": {k: v for k, v in main_panel.items() if k != "cells"},
        "cells": main_panel["cells"],
        "mechanism": {k: v for k, v in mech.items() if k != "cells"},
        "je": je,
        "gates": gates, "n_gate_fail": int(n_fail_all),
        "verdicts": verdict,
        "wall_sec": round(time.perf_counter() - t0, 2),
        "peak_mib": round(peak / 1024.0 ** 2, 1),
    }
    if not args.ctrl_only:
        (outdir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")

    print("\n================ 判词 ================")
    for k, v in verdict.items():
        if k in ("n_m_star", "ratio_IVfree_over_price"):
            continue
        print(f"  {k}: {v}")
    print(f"  闸：{len([g for g in gates.values() if isinstance(g, dict)])} 项，"
          f"fail={n_fail_all}")
    for k, v in gates.items():
        if isinstance(v, dict):
            print(f"    {k:<4} {'PASS' if v['pass'] else 'FAIL'}  {v['name']}")
        else:
            print(f"    {k:<4} = {v}")
    print(f"[BUDGET] {summary['wall_sec']}s / peak {summary['peak_mib']} MiB / "
          f"run_episode=0 / train=0")
    print(f"[EVIDENCE] {outdir}")
    return 0 if n_fail_all == 0 else 1


def _gate_g9(cfg) -> dict:
    """`G9`：GH 21 vs 41 求积点数收敛（只在 τ=3 上做，41² 太贵）。"""
    a = build_engine(cfg, SIGMA_JUDGE, 9, GH_N, probe_cost=0)
    b = build_engine(cfg, SIGMA_JUDGE, 9, GH_N_CHECK, probe_cost=0)
    rows = []
    worst = 0.0
    for p0 in (0.02, 1.0):
        va = a(3, p0, p0, 0.0, 1.0, 0.0, 1.0, 1)
        vb = b(3, p0, p0, 0.0, 1.0, 0.0, 1.0, 1)
        d = abs(va - vb)
        worst = max(worst, d)
        rows.append({"P0": p0, "V21": va, "V41": vb, "abs": d})
    scale = max(1e-12, max(abs(r["V41"]) for r in rows))
    return {"rows": rows, "worst_abs": float(worst), "rel": float(worst / scale),
            "pass": bool(worst / scale < 1e-6)}


def _gate_g11(main_panel: dict) -> dict:
    """`G11`：活体单调性 —— `price ≥ 0`、`IV_free ≥ 0`（"两状态之差"必报单调性）。"""
    cells = main_panel.get("cells", [])
    bad_price = [c for c in cells if c.get("price") is not None and c["price"] < -1e-15]
    bad_free = [c for c in cells if c["IV_free"] < -1e-15]
    return {"n_cell": len(cells), "n_bad_price": len(bad_price),
            "n_bad_ivfree": len(bad_free),
            "worst_price": (min((c["price"] for c in cells if c.get("price") is not None),
                                default=None)),
            "worst_ivfree": (min((c["IV_free"] for c in cells), default=None)),
            "pass": bool(not bad_price and not bad_free and len(cells) > 0)}


if __name__ == "__main__":                        # pragma: no cover
    raise SystemExit(main())
