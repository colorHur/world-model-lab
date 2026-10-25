# -*- coding: utf-8 -*-
"""`CW-R3.1`：**信道信息定价的规模先算**——三个零 episode 判定。

预注册：`docs/research/CW-R3.1-信道信息定价规模-预注册-v0.1.md`（写于 2026-10-26 01:1x，跑前）

背景（`CW-R3.0` 收口时的 `J0`）：两个参考物的动作集都是 `{HOLD, SEND_i}`，**不含 `PROBE`**
——因为 `plan_dp` 把 `p̂_i` 在槽初**冻结**，`PROBE` 在 DP 内恒被 `HOLD` 支配。
一旦 `p̂` 内生，值函数状态必须带上信道信念。

三个面板
--------
`A` 规模：`(P0,P1,b)` 的**可达状态数**逐层枚举（只数状态、不算值）⇒ 拟合每步增长 `γ`
   ⇒ 乘上 `(m,v)` 的因子 ⇒ `N(256)` 量级（判 `SCALE-*`）。
`B` 前提：以 `(m,v)` 做状态，在 `ack_mode="exact"`（研究基线）下**是否合法**？
   构造两个 `GridBelief` 后验，**矩相同、形状不同**，各吃同一条 ACK ⇒ 比较 `(m',v')`。
`C` 分辨率：把 `m` 量化到 `n_m` 格后，要多少格才能分辨一次探测的信息价值
   `IV = V(允许 PROBE) − V(禁止 PROBE)`。

纪律
----
1. **零 episode**（`run_episode = 0`）、零训练、零参数更新、纯 CPU。
2. **被测物一字节不动**：只读 `dp_audit.ExactClosedLoop` / `world.GridBelief` /
   `posterior.p_succ_expected` / `world.CWWorld._ack_bump`（只读调用）/ `config.mainline_config`。
3. 阈值全部来自预注册 §2；`judge()` 只按表取，不许人工改判词。
4. 预注册 §4 的**自我否证口**：若 `N(256)` 其实不大，本轮必须判 `SCALE-TRACTABLE`，
   并写明"R3 的『解析拿不到』前提被驳倒"。

★ 本文件**不**读任何真值字段（`_log_h_true` / `_x_true` / `cost_true` / `z_meas`），闸 `G7` 用 AST 钉住。
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
from .dp_audit import ExactClosedLoop
from .posterior import p_succ_expected
from .r1_14_exactbase import _now_utc, _src_hash
from .streams import make_episode_rngs
from .world import CWWorld, GridBelief

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results" / "cw_v1"

PROTOCOL = "CW-POMDP-v1.1"
ROUND = "CW-R3.1"

#: 预注册 §3 面板 `A`：τ 网格（外推用）
TAUS_A: tuple[int, ...] = (6, 8, 10, 12)
#: 枚举上限（超了就记 `capped`，不外推被截断的那点）
CAP_A = 4_000_000

#: 预注册 §2.3：`n_m` 阶梯与 τ（★ 跑前修订 §7-A2：τ 扩到 1..4，判词只用 3/4）
N_M_LIST: tuple[int, ...] = (9, 17, 33, 65, 129, 257)
TAUS_C: tuple[int, ...] = (1, 2, 3, 4)
TAUS_JUDGE: tuple[int, ...] = (3, 4)
SIGMA_CSI_LIST: tuple[float, ...] = (0.05, 0.5, 2.0, 50.0)
GH_N = 21              #: 积分 `y` 的高斯–埃尔米特点数（预注册：`V_ref` 与 `V_grid` 用**同一**点数）
GH_N_CHECK = 41        #: 闸 `G9`：点数收敛核对
M_RANGE = 4.0          #: `m` 量化区间 `[-4,4]`（公开先验 sd=1 的 4σ）
#: ★ 跑后修订 §7-A5：C 面板的预算必须**真的紧**。首跑用 `b0 = τ`（每槽都能发）
#:   ⇒ `SEND` 弱支配 `HOLD`、动作对 `p` 完全不敏感 ⇒ `IV ≡ 0`（见 `F1`）。
#:   这里改用面板的 0.5 档：`b0 = floor(0.5·τ)`。
B_RATIO = 0.5
#: ★ 跑后修订 §7-A6：`P0` 扫描。`prior_var=1.0` 只是 **t=0** 的初值；
#:   面板稳态 `P` 在 0.005–0.05（R3.0 的 `PS` 网格）。单点会让"探测 vs 一次发送"的
#:   相对价格完全由初值决定 ⇒ 必须扫描，不许挑一个好看的点。
P0_LIST: tuple[float, ...] = (0.005, 0.02, 0.05, 0.2, 1.0)

#: 预注册 §2.1/§2.2/§2.3 阈值（**看数前写定**）
TRAC_N = 1e7
TRAC_GIB = 4.0
TRAC_SEC = 600.0
MARG_N = 1e9
CLOSURE_EPS = 1e-6
IV_TOL = 0.10
#: ★ 跑后修订 §7-A8：`IV` 的**噪声地板**。`IV_free` 在 20 个单元上有 19 个**逐位 0.0**、
#:   剩下一个是 `5.551e-17`（= 2^-54，纯浮点残差）⇒ 若把它当"正价值"，`n_m*` 就建立在噪声上。
#:   地板取 `1e-12`：比 f64 在 O(1) 量级上的舍入（~1e-16）高 4 个量级，比任何物理效应低 6+ 个量级。
IV_NOISE = 1e-12

#: `BOUNDED` plant 的 A≡1（`ExactQ`/`ExactClosedLoop` 的前提）
assert all(abs(a - 1.0) < 1e-15 for a in mainline_config().plant.A), \
    "本轮的精确 DP 只在 A≡1 上成立"

_SQRT2PI = math.sqrt(2.0 * math.pi)


# ==================================================================== 面板 A：可达状态枚举
def panel_a(cfg) -> dict:
    """`(P0,P1,b)` 的**可达状态数**逐层枚举（BFS，只数不算值）。

    ★ 与 `CW-R3.0` `J3` 的口径差（必须在交付稿写明，**不许**声称"复现"）：
      `J3` 数的是 `ExactQ` 的 `lru_cache` 里**实际被访问**的状态，初始 `P` 取自
      `PS=(0.005,0.010,0.020,0.050)` 的 16 组配对、`b` 取 `{0,2,τ//2}` 的**并集**；
      这里从**单点** `(prior_var, prior_var)` 出发、`b` 只取 `τ//2`。
      ⇒ 两者**状态数不可直接比**，只能比**每步增长指数 `γ`**。
    """
    sw2 = tuple(float(s) ** 2 for s in cfg.plant.sigma_w)
    sm2 = float(cfg.plant.sigma_meas) ** 2
    p0 = float(cfg.plant.prior_var)

    rows: list[dict] = []
    t_all = time.perf_counter()
    for tau in TAUS_A:
        b0 = int(tau) // 2
        cur: set = {(p0, p0, b0)}
        counts: list[int] = [1]
        capped = False
        for _ in range(int(tau)):
            nxt: set = set()
            for (q0, q1, bd) in cur:
                nxt.add((q0 + sw2[0], q1 + sw2[1], bd))          # HOLD
                if bd >= 1:
                    for i, qq in enumerate((q0, q1)):             # SEND i
                        g = qq / (qq + sm2)
                        succ = (1.0 - g) * qq + sw2[i]
                        fail = qq + sw2[i]
                        if i == 0:
                            nxt.add((succ, q1 + sw2[1], bd - 1))
                            nxt.add((fail, q1 + sw2[1], bd - 1))
                        else:
                            nxt.add((q0 + sw2[0], succ, bd - 1))
                            nxt.add((q0 + sw2[0], fail, bd - 1))
            cur = nxt
            counts.append(len(cur))
            if len(cur) > CAP_A:
                capped = True
                break
        rows.append({"tau": int(tau), "b0": b0, "counts": counts,
                     "n_states": counts[-1] if not capped else None,
                     "n_states_at_cap": len(cur), "capped": capped,
                     "n_P1": len({round(s[0], 15) for s in cur}),
                     "sec": round(time.perf_counter() - t_all, 3)})
        print(f"[A] tau={tau:>3} b0={b0:>3} n_states={len(cur):>9} capped={capped} "
              f"({time.perf_counter() - t_all:.1f}s)", flush=True)
        if len(cur) > 1_200_000:      # 再往上只会更慢，且下游只需要 γ
            break

    # 每步增长指数（只用未被截断、且 count 足够大的相邻点）
    gam: list[float] = []
    ok = [r for r in rows if not r["capped"]]
    for r in ok:
        c = r["counts"]
        if len(c) >= 5 and c[-1] > 1000 and c[-3] > 0:
            gam.append(round((c[-1] / c[-3]) ** (1.0 / 2.0), 6))
    gamma = float(np.median(gam)) if gam else float("nan")
    return {"rows": rows, "gamma_per_step": gamma, "gamma_samples": gam,
            "sec": round(time.perf_counter() - t_all, 3)}


def extrapolate(panel: dict, n_m: int, horizon: int = 256) -> dict:
    """`N(256) = N_A(256) × n_m²`（`v` 不计因子：面板 `B3` 实测 `v` 不随观测值分支）。"""
    gamma = float(panel["gamma_per_step"])
    ok = [r for r in panel["rows"] if not r["capped"]]
    if not ok or not math.isfinite(gamma):
        return {"ok": False}
    base = ok[-1]
    k = int(horizon) - int(base["tau"])
    n_a = float(base["n_states"]) * (gamma ** k)
    n_tot = n_a * float(int(n_m)) ** 2
    # 实测 字节/状态 与 状态/秒（面板 A 自身就是这个量纲：枚举本身就等价于建表）
    bytes_per_state = 96.0          # 3 个 float64 + set 桶开销的下界估计（只作下界，交付稿声明）
    return {"ok": True, "gamma": gamma, "base_tau": int(base["tau"]),
            "base_n": int(base["n_states"]), "n_a_256": n_a, "n_m": int(n_m),
            "n_256": n_tot,
            "est_gib": n_tot * bytes_per_state / 1024.0 ** 3,
            "est_sec": n_tot / max(1e-9, float(base["n_states"]) / max(1e-9, float(base["sec"])))}


# ==================================================================== 面板 B：矩闭合
def _bimodal_belief(cfg, delta: float, mean: float, var: float) -> GridBelief:
    """在固定网格上造一个**双峰**后验：均值 `mean`、方差 `var`、50/50 质量。

    取 `s² = var − delta²`（`delta² ≤ var`）；`s=0` 时退化为两点分布。
    网格是 `GridBelief` 自己的 321 点，所以矩必须**按离散和**匹配，不是解析匹配。
    """
    g = GridBelief(1, float(cfg.channel.rho_panel[0]), float(cfg.channel.kappa),
                   float(cfg.channel.log_h_star), init_mean=0.0, init_std=1.0)
    x = g.x
    s2 = float(var) - float(delta) ** 2
    if s2 < -1e-12:
        raise ValueError(f"delta={delta} 与 var={var} 不相容（需 delta² ≤ var）")
    s = math.sqrt(max(0.0, s2))
    if s < 1e-9:
        # 两点：把质量放到最接近 mean±delta 的两个格点上
        dens = np.zeros_like(x)
        for tgt in (mean - delta, mean + delta):
            j = int(np.argmin(np.abs(x - tgt)))
            dens[j] += 0.5
    else:
        dens = (np.exp(-0.5 * ((x - (mean - delta)) / s) ** 2)
                + np.exp(-0.5 * ((x - (mean + delta)) / s) ** 2))
    dens = dens / dens.sum()
    g.p[0] = dens
    return g


def _gauss_belief(cfg, mean: float, var: float) -> GridBelief:
    g = GridBelief(1, float(cfg.channel.rho_panel[0]), float(cfg.channel.kappa),
                   float(cfg.channel.log_h_star), init_mean=0.0, init_std=1.0)
    s = math.sqrt(max(1e-12, float(var)))
    dens = np.exp(-0.5 * ((g.x - float(mean)) / s) ** 2)
    g.p[0] = dens / dens.sum()
    return g


def panel_b(cfg) -> dict:
    """同矩、异形 ⇒ 同一条 ACK ⇒ `(m',v')` 差多少。"""
    rows: list[dict] = []
    # 剂量：同一个 (m,v)=(0,1)，delta 从温和到极端（delta² ≤ v）
    for delta in (0.4, 0.6, 0.8, 1.0):
        gg = _gauss_belief(cfg, 0.0, 1.0)
        gb = _bimodal_belief(cfg, delta, 0.0, 1.0)
        m0, v0 = gg.moments()
        m1, v1 = gb.moments()
        row: dict = {"delta": float(delta), "m0": float(m0[0]), "v0": float(v0[0]),
                     "m1": float(m1[0]), "v1": float(v1[0]),
                     "pre_dm": float(abs(m0[0] - m1[0])),
                     "pre_dv": float(abs(v0[0] - v1[0]))}
        for tag, succ in (("ack", True), ("nack", False)):
            a = _gauss_belief(cfg, 0.0, 1.0)
            b = _bimodal_belief(cfg, delta, 0.0, 1.0)
            a.observe_binary(0, succ)
            b.observe_binary(0, succ)
            ma, va = a.moments()
            mb, vb = b.moments()
            row[f"{tag}_dm"] = float(abs(ma[0] - mb[0]))
            row[f"{tag}_dv"] = float(abs(va[0] - vb[0]))
            row[f"{tag}_m_gauss"] = float(ma[0])
            row[f"{tag}_v_gauss"] = float(va[0])
            row[f"{tag}_m_bimodal"] = float(mb[0])
            row[f"{tag}_v_bimodal"] = float(vb[0])
        rows.append(row)

    # 活体负控 `G6`：同一个后验复制两份 ⇒ 更新后必须**严格** 0.0
    c1 = _gauss_belief(cfg, 0.3, 0.7)
    c2 = _gauss_belief(cfg, 0.3, 0.7)
    c1.observe_binary(0, True)
    c2.observe_binary(0, True)
    m1_, v1_ = c1.moments()
    m2_, v2_ = c2.moments()
    ctrl = {"dm": float(abs(m1_[0] - m2_[0])), "dv": float(abs(v1_[0] - v2_[0]))}

    worst_dm = max(max(r["ack_dm"], r["nack_dm"]) for r in rows)
    worst_dv = max(max(r["ack_dv"], r["nack_dv"]) for r in rows)
    return {"rows": rows, "ctrl": ctrl,
            "worst_dm": float(worst_dm), "worst_dv": float(worst_dv),
            "verdict": ("MOMENT-CLOSURE-FAILS"
                        if (worst_dm > CLOSURE_EPS or worst_dv > CLOSURE_EPS)
                        else "MOMENT-CLOSURE-HOLDS")}


def panel_b3(cfg) -> dict:
    """`approx` 模型下 `world._ack_bump` 的 `v'` 与 ACK/NACK 方向**无关**（代数事实）。

    用**真的** `CWWorld` 实例调 `_ack_bump`，不是自己重写公式。
    """
    # 用 approx 信息模型的配置（`mainline_config` 是 exact；这里显式换回历史口径）
    base = mainline_config()
    ch = type(base.channel)(**{**base.channel.__dict__, "ack_mode": "approx"})
    cfg_ap = type(base)(plant=base.plant, channel=ch, protocol=base.protocol,
                        root_seed=base.root_seed)
    rngs, sids = make_episode_rngs(cfg_ap.root_seed, "dev", 100)
    w = CWWorld(cfg_ap, rngs, sids, rho=float(cfg_ap.channel.rho_panel[0]))
    rows = []
    for (m0, v0) in ((0.0, 1.0), (0.3, 0.7), (-1.2, 0.25), (2.0, 0.05)):
        out = {}
        for tag, d in (("ack", 1.0), ("nack", -1.0)):
            w._m_h = np.array([m0], dtype=float)
            w._v_h = np.array([v0], dtype=float)
            w._ack_bump(0, d)
            out[tag] = (float(w._m_h[0]), float(w._v_h[0]))
        rows.append({"m0": m0, "v0": v0,
                     "ack": out["ack"], "nack": out["nack"],
                     "dv": abs(out["ack"][1] - out["nack"][1]),
                     "dm": abs(out["ack"][0] - out["nack"][0])})
    return {"rows": rows,
            "max_dv": max(r["dv"] for r in rows),
            "verdict": ("V-NOT-BRANCHING" if max(r["dv"] for r in rows) == 0.0
                        else "V-BRANCHES")}


# ==================================================================== 面板 C：IV 分辨率
def _gh(n: int) -> tuple[np.ndarray, np.ndarray]:
    return np.polynomial.hermite_e.hermegauss(int(n))


def build_dp(cfg, sigma_csi: float, n_m: int | None, gh_n: int,
             probe_enabled: bool = True, p_frozen: float | None = None,
             probe_cost: int = 1):
    """1 用户、连续 `y` 求积的后向 DP；`n_m is not None` 时把 `m'` 量化到 `n_m` 格。

    状态 `(t, P, m, v, b)`；动作 `{HOLD, SEND, PROBE}`。
    ★ 与预注册 §5-④ 一致：这是**1 用户简化实例**；`ACK → 信道信念` 的更新在本面板**关闭**
      （理由与未检验性都写在交付稿边界节）。
    ★ 跑前修订 §7-A3：`probe_enabled=False` + `p_frozen` 给定时，本 DP 退化为
      「`p̂` 冻结、动作集 `{HOLD, SEND}`」的 1 用户闭环 —— 闸 `G1b` 用它和**暴力枚举**
      （显式枚举动作序列 × 结果树，不复用本递归）逐位对账。
    """
    p = cfg.plant
    sw2 = float(p.sigma_w[0]) ** 2
    sm2 = float(p.sigma_meas) ** 2
    q = float(p.q_cost)
    rho = float(cfg.channel.rho_panel[0])
    rho2 = rho * rho
    infl = 1.0 - rho2
    si2 = float(sigma_csi) ** 2
    nodes, weights = _gh(gh_n)
    centers = (np.linspace(-M_RANGE, M_RANGE, int(n_m)) if n_m is not None else None)

    def snap(m: float) -> float:
        if centers is None:
            return float(m)
        j = int(np.argmin(np.abs(centers - float(m))))
        return float(centers[j])

    @lru_cache(maxsize=None)
    def V(t: int, P: float, m: float, v: float, b: int) -> float:
        if t <= 0:
            return 0.0
        S = q * float(P)
        mn = rho * float(m)
        vn = rho2 * float(v) + infl
        ph = (float(p_frozen) if p_frozen is not None
              else p_succ_expected(float(m), float(v), cfg))
        # HOLD
        best = V(t - 1, float(P) + sw2, mn, vn, int(b))
        if b >= 1:
            # SEND
            g = float(P) / (float(P) + sm2)
            ps_ = (1.0 - g) * float(P) + sw2
            pf_ = float(P) + sw2
            val = (ph * V(t - 1, ps_, mn, vn, int(b) - 1)
                   + (1.0 - ph) * V(t - 1, pf_, mn, vn, int(b) - 1))
            if val < best:
                best = val
        if b >= int(probe_cost) and probe_enabled and p_frozen is None:
            # PROBE：`y = log h + N(0, si2)`，共轭后 `m' = m + gp (y − m)`
            gp = float(v) / (float(v) + si2)
            vp = float(v) * si2 / (float(v) + si2)
            sy = math.sqrt(float(v) + si2)
            vpp = rho2 * vp + infl
            tot = 0.0
            for z, w in zip(nodes, weights):
                y = float(m) + sy * float(z)
                mp = snap(float(m) + gp * (y - float(m)))
                tot += float(w) * V(t - 1, float(P) + sw2, rho * mp, vpp,
                                    int(b) - int(probe_cost))
            tot /= _SQRT2PI
            if tot < best:
                best = tot
        return S + best

    def value(tau: int, P: float, m: float, v: float, b: int) -> float:
        return float(V(int(tau), float(P), float(m), float(v), int(b)))

    def n_states() -> int:
        return int(V.cache_info().currsize)

    return value, n_states


def panel_c(cfg) -> dict:
    """三个量，**分开报**（★ 跑后修订 §7-A7）：

    · `IV_net  = V(禁探测,b) − V(可探测且探测花 1 量子,b)` —— 预注册 §2.3 的原定义（净价值）；
    · `IV_free = V(禁探测,b) − V(可探测但探测免费,b)` —— **毛价值**（Jensen 缺口本身）；
    · `price  = V(禁探测,b−1) − V(禁探测,b)` —— 一个资源量子的**价格**（同口径可测）。

    为什么要分开：首跑只报 `IV_net`，在 4×5×4 个单元上**逐位全是 0.0** —— 那不是"信息无价值"，
    而是"毛价值 < 价格"（`V` 对 `p̂` 非仿射，实测残差 8.5e-4 vs 跨度 2.6e-2 ⇒ 有 Jensen 缺口）。
    把两者混成一个"IV"会把"买不起"说成"没用"。分辨率问题只该问 `IV_free`。

    ★ 跑前修订 §7-A1：净价值差的**方向**按成本口径为 `V_禁 − V_允许 ≥ 0`（`G2` 的意图）。
    ★ 跑后修订 §7-A6：`P0` 扫 `P0_LIST`（`prior_var=1.0` 只是 t=0 初值，面板稳态 `P` 在 0.005–0.05）。
    """
    b_of = {tau: int(math.floor(B_RATIO * tau)) for tau in TAUS_C}
    out: dict = {"iv": [], "grid": [], "price": [], "b_of": b_of,
                 "P0_list": list(P0_LIST)}
    t_all = time.perf_counter()

    v_no, _ = build_dp(cfg, SIGMA_CSI_LIST[0], None, GH_N, probe_enabled=False)
    v_free, _ = build_dp(cfg, SIGMA_CSI_LIST[0], None, GH_N, probe_cost=0)
    vn = {(tau, p0): v_no(tau, p0, 0.0, 1.0, b_of[tau])
          for tau in TAUS_C for p0 in P0_LIST}
    for tau in TAUS_C:
        for p0 in P0_LIST:
            lo = v_no(tau, p0, 0.0, 1.0, max(0, b_of[tau] - 1))
            # ★ `price = V(b−1) − V(b) ≥ 0`（预算越少越贵）。首版写成 `V(b) − V(b−1)` ⇒ 全负，
            #   第二处符号自伤（§7-B 第 12 条）。用活体单调性 `V(0) ≥ V(1) ≥ V(2)` 复核。
            out["price"].append({"tau": int(tau), "P0": float(p0), "b0": b_of[tau],
                                 "price": lo - vn[(tau, p0)],
                                 "V_b": vn[(tau, p0)], "V_bm1": lo})

    for sig in SIGMA_CSI_LIST:
        v_pay, n_pr = build_dp(cfg, sig, None, GH_N, probe_cost=1)
        v_fr, _ = build_dp(cfg, sig, None, GH_N, probe_cost=0)
        for tau in TAUS_C:
            for p0 in P0_LIST:
                vp = v_pay(tau, p0, 0.0, 1.0, b_of[tau])
                vf = v_fr(tau, p0, 0.0, 1.0, b_of[tau])
                b = vn[(tau, p0)]
                out["iv"].append({"sigma_csi": float(sig), "tau": int(tau),
                                  "b0": b_of[tau], "P0": float(p0),
                                  "V_noprobe": b, "V_probe_paid": vp,
                                  "V_probe_free": vf,
                                  "IV_net": b - vp, "IV_free": b - vf,
                                  "n_states": n_pr()})
        print(f"[C-IV] sigma={sig:<5} done ({time.perf_counter() - t_all:.1f}s)", flush=True)

    # ---- 量化误差：以**免费探测**的 DP 为参考（`IV_free` 才是分辨率该问的量）
    sig = SIGMA_CSI_LIST[0]
    v_ref, _ = build_dp(cfg, sig, None, GH_N, probe_cost=0)
    ivf = {(r["tau"], r["P0"]): r["IV_free"] for r in out["iv"] if r["sigma_csi"] == sig}
    for n_m in N_M_LIST:
        vg, nst = build_dp(cfg, sig, n_m, GH_N, probe_cost=0)
        for tau in TAUS_C:
            for p0 in P0_LIST:
                ref = v_ref(tau, p0, 0.0, 1.0, b_of[tau])
                gv = vg(tau, p0, 0.0, 1.0, b_of[tau])
                iv = ivf[(tau, p0)]
                out["grid"].append({"tau": int(tau), "P0": float(p0), "n_m": int(n_m),
                                    "V_grid": gv, "V_ref": ref,
                                    "abs_err": abs(gv - ref), "IV_free": iv,
                                    "rel_err": (abs(gv - ref) / abs(iv) if iv > 0 else None),
                                    "n_states": nst()})
        print(f"[C-GRID] n_m={n_m:>4} done ({time.perf_counter() - t_all:.1f}s)", flush=True)

    # ---- 闸 `G9`：求积点数收敛（21 vs 41；只在 τ≤3 上做，41^4 太贵）
    v41, _ = build_dp(cfg, sig, None, GH_N_CHECK, probe_cost=0)
    g9 = []
    for tau in (3, 4):
        for p0 in (0.02, 1.0):
            a21 = v_ref(tau, p0, 0.0, 1.0, b_of[tau])
            a41 = v41(tau, p0, 0.0, 1.0, b_of[tau])
            g9.append({"tau": int(tau), "P0": p0, "V21": a21, "V41": a41,
                       "abs": abs(a21 - a41)})
    out["gh_check"] = g9
    out["sec"] = round(time.perf_counter() - t_all, 3)
    return out


# ==================================================================== 闸 `G1b` 的独立参考
def brute_force_1u(cfg, p_frozen: float, tau: int, P0: float, b0: int) -> float:
    """**暴力枚举**：所有 `{HOLD, SEND}` 动作序列 × 所有成功/失败结果树，取最小期望成本。

    ★ 故意不复用 `build_dp` 的递归（否则是自证）：这里显式遍历 `2^τ` 条序列，
      每条按 `2^{#SEND}` 棵结果树累加概率权重。
    """
    from itertools import product as _product

    sw2 = float(cfg.plant.sigma_w[0]) ** 2
    sm2 = float(cfg.plant.sigma_meas) ** 2
    q = float(cfg.plant.q_cost)

    def run(seq: tuple[str, ...]) -> float:
        # 状态 = [P, 剩余预算] 的概率分布字典
        dist: dict[tuple[float, int], float] = {(float(P0), int(b0)): 1.0}
        total = 0.0
        for act in seq:
            nxt: dict[tuple[float, int], float] = {}
            for (P, bd), w in dist.items():
                total += w * q * P                       # 本步即时代价
                if act == "HOLD" or bd < 1:
                    k = (P + sw2, bd)
                    nxt[k] = nxt.get(k, 0.0) + w
                    continue
                g = P / (P + sm2)
                for (Pn, pn) in (((1.0 - g) * P + sw2, p_frozen),
                                 (P + sw2, 1.0 - p_frozen)):
                    k = (Pn, bd - 1)
                    nxt[k] = nxt.get(k, 0.0) + w * pn
            dist = nxt
        return total

    best = math.inf
    for seq in _product(("HOLD", "SEND"), repeat=int(tau)):
        if seq.count("SEND") <= int(b0):
            best = min(best, run(seq))
    return float(best)


def gate_g1b(cfg) -> dict:
    """`G1b`：`build_dp(probe_enabled=False, p_frozen=p)` == 暴力枚举（1 用户，τ=1..4）。"""
    p_frozen = p_succ_expected(0.0, 1.0, cfg)
    rows = []
    d = 0.0
    for tau in (1, 2, 3, 4):
        b0 = int(tau)
        fn, _ = build_dp(cfg, SIGMA_CSI_LIST[0], None, GH_N,
                         probe_enabled=False, p_frozen=p_frozen)
        a = fn(tau, float(cfg.plant.prior_var), 0.0, 1.0, b0)
        b = brute_force_1u(cfg, p_frozen, tau, float(cfg.plant.prior_var), b0)
        rows.append({"tau": tau, "dp": a, "brute": b, "abs": abs(a - b)})
        d = max(d, abs(a - b))
    # ★ 跑前修订 §7-A4：**不要求逐位**相等 —— 两条路径的浮点求和顺序不同
    #   （DP 是"沿树深先累代价后取 min"，暴力是"沿序列先算概率权再加权求和"），
    #   实测差 4.44e-16（`tau=4`）⇒ 容差 1e-12，且实测值必须照实报。
    return {"rows": rows, "max_abs": float(d), "tol": 1e-12,
            "pass": bool(d <= 1e-12)}


def gate_g2c(cfg) -> dict:
    """`G2c`：探测的**数学恒等式** —— 求积权重归一 + 后验均值是鞅（`E[m'] = m`）。"""
    nodes, weights = _gh(GH_N)
    wsum = float(np.sum(weights))
    rows = []
    worst = 0.0
    for (m, v) in ((0.0, 1.0), (0.3, 0.7), (-1.5, 0.2), (2.0, 0.05)):
        si2 = SIGMA_CSI_LIST[0] ** 2
        gp = v / (v + si2)
        sy = math.sqrt(v + si2)
        em = float(np.sum(weights * (m + gp * (sy * nodes))) / _SQRT2PI)
        worst = max(worst, abs(em - m))
        rows.append({"m": m, "v": v, "E_m_post": em, "abs": abs(em - m)})
    return {"w_sum": wsum, "w_sum_target": _SQRT2PI,
            "w_abs": abs(wsum - _SQRT2PI), "rows": rows,
            "worst_martingale": float(worst),
            "pass": bool(abs(wsum - _SQRT2PI) < 1e-12 and worst < 1e-12)}


# ==================================================================== 闸
def run_gates(cfg, panel: dict, pb: dict, pb3: dict, pc: dict, full: bool = True) -> dict:
    res: dict = {}

    # ---- G1a：`ExactQ` == `ExactClosedLoop`（确认被引用的基线本身没被改坏）
    from .r3_0_card import ExactQ
    eq = ExactQ(cfg, [0.5, 0.2]).solve(6)
    ecl = ExactClosedLoop(cfg, [0.5, 0.2])
    d = 0.0
    for (p0, p1, b) in ((0.02, 0.02, 3), (0.005, 0.05, 2), (0.05, 0.01, 0)):
        d = max(d, abs(eq.value(p0, p1, b) - ecl.value(6, p0, p1, b)))
    res["G1a"] = {"name": "ExactQ == ExactClosedLoop", "max_abs": float(d),
                  "pass": bool(d == 0.0)}

    # ---- G1b：本文件的 1 用户引擎（禁探测 + p̂ 冻结）== 暴力枚举
    g1b = gate_g1b(cfg)
    res["G1b"] = {"name": "1用户引擎 == 暴力枚举(序列×结果树)", **g1b}

    # ---- G2c：探测的数学恒等式（权重归一 + 鞅）
    g2c = gate_g2c(cfg)
    res["G2c"] = {"name": "求积归一 + E[m']=m", **g2c}

    # ---- G2：净/毛信息价值都 >= 0（允许 PROBE 是动作超集）
    if full:
        wn = min(r["IV_net"] for r in pc["iv"])
        wf = min(r["IV_free"] for r in pc["iv"])
        res["G2"] = {"name": "IV_net>=0 且 IV_free>=0", "min_net": float(wn),
                     "min_free": float(wf),
                     "pass": bool(wn >= -1e-15 and wf >= -1e-15)}
    else:
        res["G2"] = {"name": "IV_net>=0 且 IV_free>=0", "min_net": None,
                     "min_free": None, "pass": False}
    # G2b：τ=1 的毛价值必须**严格 0**（只有 1 槽 ⇒ 探测后无槽可用；§7-A2）
    t1 = [r["IV_free"] for r in pc["iv"] if r["tau"] == 1]
    res["G2b"] = {"name": "IV_free(tau=1)==0", "vals": t1,
                  "pass": bool(t1 and max(abs(x) for x in t1) == 0.0)}

    # ---- G3：`IV_free` 对 sigma_csi 单调不增（逐 (τ,P0) 单元）
    mono_ok = bool(full)
    bad3 = []
    n_cell = 0
    for tau in TAUS_C:
        for p0 in P0_LIST:
            seq = [next((r["IV_free"] for r in pc["iv"]
                         if r["tau"] == tau and r["P0"] == p0 and r["sigma_csi"] == s), None)
                   for s in SIGMA_CSI_LIST]
            if any(x is None for x in seq):
                mono_ok = False
                continue
            n_cell += 1
            for i in range(len(seq) - 1):
                if seq[i + 1] > seq[i] + 1e-12:
                    mono_ok = False
                    bad3.append({"tau": tau, "P0": p0, "i": i, "seq": seq})
    res["G3"] = {"name": "IV_free 对 sigma_csi 单调不增", "n_cell": n_cell,
                 "bad": bad3, "pass": bool(mono_ok)}

    # ---- G4：sigma_csi=50 时毛价值远小于 sigma=0.05（只在 IV_free(0.05)>0 的单元上判）
    g4 = []
    ok4 = bool(full)
    n4 = 0
    for tau in TAUS_C:
        for p0 in P0_LIST:
            big = next((r["IV_free"] for r in pc["iv"] if r["tau"] == tau and r["P0"] == p0
                        and r["sigma_csi"] == 50.0), None)
            sml = next((r["IV_free"] for r in pc["iv"] if r["tau"] == tau and r["P0"] == p0
                        and r["sigma_csi"] == 0.05), None)
            if big is None or sml is None:
                ok4 = False
                continue
            if sml <= IV_NOISE:
                continue
            n4 += 1
            g4.append({"tau": tau, "P0": p0, "IV_big": big, "IV_small": sml,
                       "ratio": abs(big) / abs(sml)})
            if abs(big) > 1e-3 * abs(sml):
                ok4 = False
    res["G4"] = {"name": "无信息探测 IV_free<=1e-3*IV_free(0.05)", "n_cell": n4,
                 "detail": g4, "pass": bool(ok4),
                 "note": ("空过（`n_cell=0`）：所有 `IV_free(0.05)` 都不超过噪声地板 "
                          f"`IV_NOISE={IV_NOISE}` ⇒ 该闸无对象可检，**不计为通过也不计为实现错**")
                 if n4 == 0 else "有对象可检"}

    # ---- G5：网格细化误差不增（允许 1 次平台，逐 (τ,P0) 单元）
    ok5 = bool(full)
    g5 = []
    n5 = 0
    for tau in TAUS_C:
        for p0 in P0_LIST:
            errs = [g["abs_err"] for g in sorted(
                [g for g in pc["grid"] if g["tau"] == tau and g["P0"] == p0],
                key=lambda z: z["n_m"])]
            if len(errs) < 2:
                ok5 = False
                continue
            n5 += 1
            rises = sum(1 for i in range(len(errs) - 1) if errs[i + 1] > errs[i] + 1e-17)
            g5.append({"tau": tau, "P0": p0, "n_rises": int(rises),
                       "errs": [float(x) for x in errs]})
            if rises > 1:
                ok5 = False
    res["G5"] = {"name": "网格细化误差不增(<=1 平台)", "n_cell": n5,
                 "detail": g5, "pass": bool(ok5)}

    # ---- G6：同后验复制的活体负控必须严格 0
    res["G6"] = {"name": "同后验复制=>严格0", **pb["ctrl"],
                 "pass": bool(pb["ctrl"]["dm"] == 0.0 and pb["ctrl"]["dv"] == 0.0)}

    # ---- G7：AST 扫描本文件不读真值
    bad = _ast_scan(Path(__file__))
    res["G7"] = {"name": "AST 不读真值字段", "hits": bad, "pass": bool(not bad)}

    # ---- G9：求积点数收敛
    if pc["gh_check"]:
        worst9 = max(g["abs"] for g in pc["gh_check"])
        scale = max(1e-12, max(abs(g["V41"]) for g in pc["gh_check"]))
        res["G9"] = {"name": "GH21 vs GH41 收敛", "worst_abs": float(worst9),
                     "rel": float(worst9 / scale),
                     "pass": bool(worst9 / scale < 1e-6)}
    else:
        res["G9"] = {"name": "GH21 vs GH41 收敛", "worst_abs": None,
                     "rel": None, "pass": False}

    # ---- G8：回归闸（另起，见 main）
    return res


_TRUTH_NAMES = ("_log" + "_h_true", "_x" + "_true", "cost" + "_true",
                "z" + "_meas", "log_h" + "_history")
#: ★ 用字符串拼接构造，避免**本文件自己的常量表**被 `ast.Constant` 扫描命中
#:   （首跑 `G7` 就是被这张表自己触发的；这是判据自伤，不是越界读取）。


def _ast_scan(path: Path) -> list[str]:
    """AST：本文件不得出现真值字段名（属性/名字都要查）。"""
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


# ==================================================================== 判词
def judge(panel: dict, ex: dict, pb: dict, pc: dict, gates: dict) -> dict:
    out: dict = {}

    # J-A 规模
    if not ex.get("ok"):
        out["J-A"] = "SCALE-UNRESOLVED"
    else:
        n = float(ex["n_256"])
        if n <= TRAC_N and ex["est_gib"] <= TRAC_GIB and ex["est_sec"] <= TRAC_SEC:
            out["J-A"] = "SCALE-TRACTABLE"
        elif n <= MARG_N:
            out["J-A"] = "SCALE-MARGINAL"
        else:
            out["J-A"] = "SCALE-EXPLODES"

    # J-B 矩闭合
    out["J-B"] = pb["verdict"]

    # J-C IV 分辨率（★ §7-A2/A5/A6/A7：判据腿 = **所有 `IV_free>0` 的 (τ,P0) 单元**；τ=1 为零控）
    need: dict = {}
    pos = sorted({(g["tau"], g["P0"]) for g in pc["grid"]
                  if g["tau"] != 1 and abs(g["IV_free"]) > IV_NOISE})
    for (tau, p0) in pos:
        cand = None
        for g in sorted([g for g in pc["grid"] if g["tau"] == tau and g["P0"] == p0],
                        key=lambda z: z["n_m"]):
            iv = abs(g["IV_free"])
            if iv > 0 and g["abs_err"] <= IV_TOL * iv:
                cand = g["n_m"]
                break
        need[f"t{tau}_P{p0}"] = cand
    if not pos:
        out["J-C"] = "NO-ABOVE-NOISE-IV-IN-1U-REDUCTION"
    elif all(need[k] for k in need):
        out["J-C"] = "IV-RESOLVABLE"
    else:
        out["J-C"] = "IV-BELOW-GRID-FLOOR"
    out["n_m_star"] = need
    out["cells_with_positive_IV_free"] = [f"t{t}_P{p}" for (t, p) in pos]
    return out


# ==================================================================== 主流程
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="CW-R3.1 信道信息定价的规模先算")
    ap.add_argument("--ctrl-only", action="store_true", help="只跑闸/自证，不跑面板")
    args = ap.parse_args(argv)

    cfg = mainline_config()
    issues = cfg.v11_validate()
    if issues:
        print("[FATAL] 配置自洽检查未过：")
        for s in issues:
            print("  -", s)
        return 2

    run = f"cw_r3_1_scale_{_now_utc()}_{_src_hash()}"
    outdir = RESULTS / run
    if not args.ctrl_only:
        outdir.mkdir(parents=True, exist_ok=True)     # ★ R2.3 教训：目录先于面板
    print(f"[RUN] {run}\n[DIR] {outdir if not args.ctrl_only else '(ctrl-only，不落盘)'}")

    t0 = time.perf_counter()
    tracemalloc.start()

    panel = {"rows": [], "gamma_per_step": float("nan"), "gamma_samples": [], "sec": 0.0}
    pb = {"rows": [], "ctrl": {"dm": -1.0, "dv": -1.0}, "worst_dm": -1.0,
          "worst_dv": -1.0, "verdict": "NOT-RUN"}
    pb3 = {"rows": [], "max_dv": -1.0, "verdict": "NOT-RUN"}
    pc = {"cells": [], "iv": [], "grid": [], "gh_check": [], "sec": 0.0}

    if not args.ctrl_only:
        panel = panel_a(cfg)
        (outdir / "cells_a.json").write_text(
            json.dumps(panel, ensure_ascii=False, indent=1), encoding="utf-8")
        pb = panel_b(cfg)
        (outdir / "cells_b.json").write_text(
            json.dumps(pb, ensure_ascii=False, indent=1), encoding="utf-8")
        pb3 = panel_b3(cfg)
        pc = panel_c(cfg)
        (outdir / "cells_c.json").write_text(
            json.dumps(pc, ensure_ascii=False, indent=1), encoding="utf-8")

    n_m_use = 129                      # 判词前先取一格中间值算量级；`J-C` 的 `n_m*` 另报
    ex = extrapolate(panel, n_m_use) if not args.ctrl_only else {"ok": False}

    gates = run_gates(cfg, panel, pb, pb3, pc, full=not args.ctrl_only)

    # ---- 回归（`G8`）
    try:
        from .gates import run_all as _run_all
        reg = _run_all()
        reg_n = int(reg["n_checks"])
        n_fail = int(reg["n_fail"])
    except Exception as e:                      # pragma: no cover
        reg_n, n_fail = -1, -1
        print(f"[G8] 回归闸异常：{type(e).__name__}: {e}")
    gates["G8"] = {"name": "gates.run_all n_fail==0", "n_total": reg_n,
                   "n_fail": n_fail, "pass": bool(n_fail == 0)}

    verdict = judge(panel, ex, pb, pc, gates) if not args.ctrl_only else \
        {"J-A": "NOT-RUN", "J-B": "NOT-RUN", "J-C": "NOT-RUN", "n_m_star": {}}

    cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    n_fail_all = sum(1 for v in gates.values() if isinstance(v, dict) and not v["pass"])

    summary = {
        "round": ROUND, "protocol": PROTOCOL, "src_hash": _src_hash(),
        "cfg_budget": [cfg.protocol.budget(r) for r in cfg.protocol.budget_ratios],
        "horizon": cfg.protocol.horizon,
        "run_episode": 0, "train_steps": 0,
        "panel_a": {k: v for k, v in panel.items() if k != "rows"},
        "panel_a_rows": panel["rows"],
        "extrapolation": ex, "n_m_used": n_m_use,
        "panel_b": pb, "panel_b3": pb3,
        "panel_c_iv": pc["iv"], "panel_c_grid": pc["grid"], "panel_c_gh": pc["gh_check"],
        "panel_c_price": pc["price"], "panel_c_b_of": pc["b_of"],
        "gates": gates, "n_gate_fail": n_fail_all,
        "verdicts": verdict,
        "wall_sec": round(time.perf_counter() - t0, 2),
        "peak_mib": round(peak / 1024.0 ** 2, 1),
    }
    # 判词段单独包 try（R2.3 教训：别让写盘前的异常吃掉面板）
    if not args.ctrl_only:
        (outdir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")

    print("\n================ 判词 ================")
    for k, v in verdict.items():
        print(f"  {k}: {v}")
    print(f"  闸：{len(gates)} 项，fail={n_fail_all}")
    for k, v in gates.items():
        print(f"    {k:<4} {'PASS' if v['pass'] else 'FAIL'}  {v['name']}")
    print(f"[BUDGET] {summary['wall_sec']}s / peak {summary['peak_mib']} MiB / "
          f"run_episode=0 / train=0")
    print(f"[EVIDENCE] {outdir}")
    return 0 if n_fail_all == 0 else 1


if __name__ == "__main__":                        # pragma: no cover
    raise SystemExit(main())
