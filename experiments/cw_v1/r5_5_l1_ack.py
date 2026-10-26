# -*- coding: utf-8 -*-
"""`CW-R5.5` 控制件：**L1 语义**（ACK 更新信道信念）下的四臂信息价值对照。

预注册：`docs/research/CW-R5.5-L1-ACK与探测价值-预注册-v0.1.md`（写于任何本轮实现与读数之前）。
判据方向、臂定义、闸、否定条件**全部跑前写死**，本文件只实现。

四个臂（预注册 §1）
-------------------
| 臂    | 探测动作            | ACK 更新信道信念 `(m,v)` | 含义 |
|-------|---------------------|--------------------------|------|
| `A0`  | 无                  | 否 | 历史全部 L2 面板的基线 |
| `A0p` | 无                  | 是 | 只有免费 ACK 的 L1 世界 |
| `VL2` | 有（`p=0`，占槽）    | 否 | L2 语义 |
| `VL1` | 有（`p=0`，占槽）    | 是 | L1 语义 |

`IV_probe_L2 = A0 − VL2`；`IV_probe_L1 = A0p − VL1`；`IV_ack = A0 − A0p`；
**主量 `survival = IV_probe_L1 / IV_probe_L2`**（"ACK 在场后探测价值剩几成"）。

纪律
----
1. 零 episode、零训练、零参数更新；`config.py` / `posterior.py` / `r3_2_iv2u.py` /
   `r5_1_price_ticks.py` / `r5_2_three_users.py` **一个字符不改**（只 `import`）。
2. `ack_updates=False` 时必须与 `r5_2.build_engine` **逐位**相同（`G1-copy`）——
   否则本文件所有读数都不是同一条协议下的读数。
3. **L1 的信道信念是假定密度滤波（ADF，单高斯矩匹配）**，不是精确贝叶斯最优控制。
   预注册 §6-1 已明令不许把 `VL1` 叫"L1 最优"。
4. `P` 的更新两个臂**逐字相同**（只差 `(m,v)` 是否被 ACK 更新）——这是 §1 的"唯一差别"。
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

from .config import ChannelConfig, bounded_plant, mainline_config
from .posterior import p_succ_expected
from .r1_14_exactbase import _now_utc, _src_hash
from .r3_2_iv2u import GH_N, M_RANGE, STATE_CAP
from .r5_1_price_ticks import SEND_TICKS, TICKS_PER_QUANTUM
from .r5_2_three_users import build_engine as r52_engine

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results" / "cw_v1"

PROTOCOL = "CW-POMDP-v1.4-ack"     # 只换 `rho_panel` + ACK 的信息权限（L1 vs L2）
ROUND = "CW-R5.5"

#: 面板常数（预注册 §2，跑前写死）
RHO = 0.999                        # 长记忆：ACK 的信息要活得过一个槽
P0 = 0.01
SIGMA_W1 = 0.02                    # n=1：单用户，无"噪声比"这一维
SIGMA_W1_2U = (0.02, 0.05)         # 备用实例 n=2（α=2.5，与历史同式）
GH_MAIN = 9                        # 主面板求积节点（分支 1+2+9=12/slot）
GH_DIAG = 21                       # 分辨率诊断
SIGMA_STAR = 1.0e-9
V_PRIOR = 1.0

TAU_LIST: tuple[int, ...] = (4, 5, 6)
B_LIST: tuple[int, ...] = (8, 16)
N_M_LIST: tuple[int, ...] = (9, 17, 33)
PROBE_TICKS_MAIN = 0               # p=0：免量子、不免槽

#: ADF 参考积分（预注册 §7-4）
#: ★ `±8` 而不是 `±6`：`±6` 截断使"无信息"极限的 `E[z²]` 少 `7.4e-8` 相对
#:   （`2(6φ(6)+Φ̄(6))`）⇒ `G3-noinfo` 会在 `1e-8` 量级假失败；`±8` 降到 `~1e-14`。
Z_HALF = 8.0
Z_DZ = 0.025                       # ⇒ 641 点
Z_DZ_REFINE = 0.0125               # ⇒ 1281 点（`G10-ck`）

#: 判据常数
IV_NOISE = 1e-12
TOL_BIT = 0.0
TOL_STRUCT = 1e-13
TOL_MONO_REL = 1e-6
SURVIVAL_SPREAD_MAX = 0.30
G2_TOL = 1e-3
G3_TOL = 1e-12


# ==================================================================== 配置
def ack_config(rho: float = RHO, sigma_w=SIGMA_W1) -> "object":
    """新协议变体 `CW-POMDP-v1.4-ack`：只换 `rho_panel`，其余与 v1.1 逐字段相同。"""
    sw = sigma_w if isinstance(sigma_w, (tuple, list)) else (float(sigma_w),)
    plant = bounded_plant(A=tuple(1.0 for _ in sw), sigma_w=tuple(float(s) for s in sw))
    base = mainline_config(plant=plant)
    ch = ChannelConfig(**{**base.channel.__dict__, "rho_panel": (float(rho),)})
    return mainline_config(plant=plant, channel=ch)


# ==================================================================== ADF（L1 的唯一新算术）
def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def ack_posterior_moments(m: float, v: float, success: bool, kappa: float,
                          log_h_star: float, *, z_half: float = Z_HALF,
                          z_dz: float = Z_DZ) -> tuple[float, float]:
    """一次 ACK/NACK 的**单高斯矩匹配**后验（ADF）。

    后验密度 `∝ N(x; m, v) · [σ(κ(x−c))]^y [1−σ]^(1−y)`，在**标准化坐标**
    `z = (x−m)/√v` 上积分（`x = m + √v·z`，`z∈[−z_half, z_half]`，步长 `z_dz`）：
    这样网格**随先验缩放**，不会因为 `v` 变小而丢支撑，也不会因为 `m` 移动而截断。

    ★ 这不是精确贝叶斯（后验一般不是高斯）；它是 L1 语义的**可算实现**。
    ★ 若后验质量下溢（`Z→0`），返回 `(m, v)`（不更新）并由调用方计数 —
      **不许静默变成 NaN**。
    """
    k = float(kappa)
    c = float(log_h_star)
    s = math.sqrt(max(0.0, float(v)))
    n = int(round(2.0 * z_half / z_dz)) + 1
    zs = np.linspace(-z_half, z_half, n)
    x = float(m) + s * zs
    u = k * (x - c)
    sig = 1.0 / (1.0 + np.exp(-np.clip(u, -700.0, 700.0)))
    lik = sig if success else (1.0 - sig)
    dens = np.exp(-0.5 * zs * zs) / math.sqrt(2.0 * math.pi)
    w = dens * lik
    # ★ 归一后权重之和为 1 ⇒ 期望**不再乘 `dz`**（首版乘了两次，`Ez2` 变成 `dz·1`，
    #   `v_post` 被压到 `v·dz`，`G3-noinfo` 当场抓到 `max_abs=1.95`）。
    #   等距网格上 `dz` 在"求和 / 归一"两步里已经相消。
    sw = float(w.sum())
    if not np.isfinite(sw) or sw <= 1e-300:
        return (float(m), float(v))
    w = w / sw
    Ez = float((w * zs).sum())
    Ez2 = float((w * zs * zs).sum())
    var_z = max(0.0, Ez2 - Ez * Ez)
    return (float(m) + s * Ez, float(v) * var_z)


def ack_posterior_moments_grid(m: float, v: float, success: bool, kappa: float,
                               log_h_star: float, n_grid: int = 641) -> tuple[float, float]:
    """**独立**参考实现：固定定义域网格 `[−8,8]`（`world.GridBelief` 的那一套）。

    `n_grid=321` 是 `world.py` 的原始分辨率（Δ=0.05）；`641`（Δ=0.025）是更细的参考。
    只用于 `G2-ackpost`（两条独立算路对账），不进 DP。
    ★ 321 点在 `v=0.1`（sd=0.316）时每 sd 只有 6.3 个格点，**参考实现自己**的矩误差就到 `1e-3` 量级
    ⇒ 闸用 641，321 的值一并报出（不隐藏参考口径）。
    """
    gx = np.linspace(-8.0, 8.0, int(n_grid))
    s2 = max(1e-300, float(v))
    dens = np.exp(-0.5 * (gx - float(m)) ** 2 / s2)
    sig = 1.0 / (1.0 + np.exp(-np.clip(float(kappa) * (gx - float(log_h_star)),
                                       -700.0, 700.0)))
    lik = sig if success else (1.0 - sig)
    p = dens * lik
    z = float(p.sum())
    if not np.isfinite(z) or z <= 1e-300:
        return (float(m), float(v))
    p = p / z
    mu = float((p * gx).sum())
    var = float((p * (gx - mu) ** 2).sum())
    return (mu, var)


# ==================================================================== 引擎
def build_engine_l1(cfg, sigma_csi: float, n_m: int, probe_ticks: int, *,
                    send_ticks: int = SEND_TICKS, gh_n: int = GH_MAIN,
                    probe_enabled: bool = True, ack_updates: bool = False,
                    n_users: int = 1, drop_obs: bool = False,
                    state_cap: int = STATE_CAP, z_dz: float = Z_DZ):
    """一用户/多用户的窗口 DP，`ack_updates` 是 L1 与 L2 的**唯一**差别。

    · `ack_updates=False` ⇒ 与 `r5_2.build_engine` 逐字同算术（`G1-copy` 钉住）。
    · `ack_updates=True`  ⇒ SEND 之后按 ACK/NACK 更新 `(m,v)`（ADF），再 AR 传播到下一槽。
      `P` 的更新两臂完全相同（`ps_`/`pf_` 不改）。
    · `drop_obs=True` 是**变异**：探测读数被丢弃 ⇒ 用于 `G5-dropobs`。

    ★ 时序（与 `P` 的处理同序）：观测发生在**当前槽** `t` ⇒ 先在 `t` 上更新，
      再传播到 `t−1`（`m'=ρ·m_post`，`v'=ρ²·v_post+(1−ρ²)`）。`P` 侧对应
      `a²(1−g)P+σ_w²`（成功）与 `a²P+σ_w²`（失败），即"更新后传播"与"只传播"。
    """
    p = cfg.plant
    a2 = tuple(float(a) ** 2 for a in p.A)
    sw2 = tuple(float(s) ** 2 for s in p.sigma_w)
    sm2 = float(p.sigma_meas) ** 2
    q = float(p.q_cost)
    kappa = float(cfg.channel.kappa)
    log_h_star = float(cfg.channel.log_h_star)
    rho = float(cfg.channel.rho_panel[0])
    rho2 = rho * rho
    infl = 1.0 - rho2
    si2 = float(sigma_csi) ** 2
    nodes, weights = np.polynomial.hermite_e.hermegauss(int(gh_n))
    nodes = [float(z) for z in nodes]
    weights = [float(w) for w in weights]
    centers = np.linspace(-M_RANGE, M_RANGE, int(n_m))
    centers_l = [float(x) for x in centers]
    c_lo, c_hi = centers_l[0], centers_l[-1]
    pt = int(probe_ticks)
    sc = int(send_ticks)
    nu = int(n_users)
    if pt < 0 or sc <= 0:
        raise ValueError("刻度必须是非负整数，且发送价必须 > 0")
    assert len(a2) >= nu and len(sw2) >= nu, "plant 的用户数不够"

    stat = {"n_snap_clamped": 0, "n_ackpost_fallback": 0, "n_ackpost_calls": 0}

    @lru_cache(maxsize=None)
    def snap(m: float) -> float:
        if m < c_lo or m > c_hi:
            stat["n_snap_clamped"] += 1
        return centers_l[int(np.argmin(np.abs(centers - float(m))))]

    @lru_cache(maxsize=None)
    def ps(m: float, v: float) -> float:
        return p_succ_expected(float(m), float(v), cfg)

    @lru_cache(maxsize=None)
    def apost(m: float, v: float, success: bool) -> tuple[float, float]:
        """ACK/NACK 后的 `(m,v)`（**当前槽**，未传播）。缓存键是 `(m,v,outcome)`。"""
        stat["n_ackpost_calls"] += 1
        out = ack_posterior_moments(m, v, success, kappa, log_h_star, z_dz=z_dz)
        if out[0] == m and out[1] == v:
            stat["n_ackpost_fallback"] += 1
        return out

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
                p_ok = a2[i] * (1.0 - g) * Pi + sw2[i]     # 成功后 P（更新后传播）
                p_no = a2[i] * Pi + sw2[i]                 # 失败后 P（只传播）
                ph = ps(mm[i], vv[i])
                aa = list(ff)
                aa[i] = p_ok
                bb = list(ff)
                bb[i] = p_no
                if ack_updates:
                    # ★ L1：ACK/NACK 更新 (m,v)，再传播到 t−1
                    m_ok, v_ok = apost(mm[i], vv[i], True)
                    ma = list(mp)
                    ma[i] = rho * snap(m_ok)
                    va = list(vp0)
                    va[i] = rho2 * v_ok + infl
                    m_no, v_no = apost(mm[i], vv[i], False)
                    mb = list(mp)
                    mb[i] = rho * snap(m_no)
                    vb = list(vp0)
                    vb[i] = rho2 * v_no + infl
                    val = (ph * V(t - 1, tuple(aa), tuple(ma), tuple(va), b - sc)
                           + (1.0 - ph) * V(t - 1, tuple(bb), tuple(mb), tuple(vb), b - sc))
                else:
                    # L2：ACK 不进信道信念（历史口径）
                    val = (ph * V(t - 1, tuple(aa), mp, vp0, b - sc)
                           + (1.0 - ph) * V(t - 1, tuple(bb), mp, vp0, b - sc))
                if val < best:
                    best = val

        if probe_enabled and b >= pt and t >= 2:           # PROBE_i（pt 刻度，占槽）
            for i in range(nu):
                mi = mm[i]
                vi = vv[i]
                gp = vi / (vi + si2)
                vpi = rho2 * (vi * si2 / (vi + si2)) + infl
                sy = math.sqrt(vi + si2)
                tot = 0.0
                for z, w in zip(nodes, weights):
                    mm2 = list(mp)
                    mm2[i] = rho * snap(mi) if drop_obs else rho * snap(mi + gp * sy * z)
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

    value.n_states = lambda: len(memo)          # type: ignore[attr-defined]
    value.capped = lambda: state["capped"]      # type: ignore[attr-defined]
    value.stat = lambda: dict(stat)             # type: ignore[attr-defined]
    value.probe_ticks = pt                      # type: ignore[attr-defined]
    value.send_ticks = sc                       # type: ignore[attr-defined]
    return value


_SQRT2PI = math.sqrt(2.0 * math.pi)


# ==================================================================== 一组四臂
class Quad:
    """一个 `(n_users, n_m)` 上的四个臂（预注册 §1），p=0。"""

    def __init__(self, cfg, n_m: int, *, gh_n: int = GH_MAIN, z_dz: float = Z_DZ,
                 probe_ticks: int = PROBE_TICKS_MAIN):
        self.cfg = cfg
        self.n_m = int(n_m)
        self.nu = int(cfg.plant.n_users)
        self.pt = int(probe_ticks)
        k = dict(gh_n=gh_n, z_dz=z_dz, n_users=self.nu)
        t0 = time.perf_counter()
        self.A0 = build_engine_l1(cfg, SIGMA_STAR, n_m, 0, probe_enabled=False,
                                  ack_updates=False, **k)
        self.A0p = build_engine_l1(cfg, SIGMA_STAR, n_m, 0, probe_enabled=False,
                                   ack_updates=True, **k)
        self.VL2 = build_engine_l1(cfg, SIGMA_STAR, n_m, probe_ticks, probe_enabled=True,
                                   ack_updates=False, **k)
        self.VL1 = build_engine_l1(cfg, SIGMA_STAR, n_m, probe_ticks, probe_enabled=True,
                                   ack_updates=True, **k)
        self.sec_build = round(time.perf_counter() - t0, 3)

    def _state(self, P, m, v, b, tau):
        zv = tuple(0.0 for _ in range(self.nu))
        vv = tuple(V_PRIOR for _ in range(self.nu))
        Pv = tuple(P for _ in range(self.nu))
        return Pv, zv, vv

    def cell(self, tau: int, b: int) -> dict:
        Pv, zv, vv = self._state(P0, 0.0, V_PRIOR, b, tau)
        a0 = self.A0(tau, Pv, zv, vv, b)
        a0p = self.A0p(tau, Pv, zv, vv, b)
        vl2 = self.VL2(tau, Pv, zv, vv, b)
        vl1 = self.VL1(tau, Pv, zv, vv, b)
        iv_p2 = a0 - vl2
        iv_p1 = a0p - vl1
        iv_ack = a0 - a0p
        surv = (iv_p1 / iv_p2) if iv_p2 > IV_NOISE else float("nan")
        return {
            "n_users": self.nu, "n_m": self.n_m, "tau": int(tau), "b_ticks": int(b),
            "b_quanta": b / TICKS_PER_QUANTUM, "probe_ticks": self.pt,
            "p": self.pt / TICKS_PER_QUANTUM, "rho": float(self.cfg.channel.rho_panel[0]),
            "P": P0, "A0": float(a0), "A0p": float(a0p), "VL2": float(vl2), "VL1": float(vl1),
            "IV_probe_L2": float(iv_p2), "IV_probe_L1": float(iv_p1),
            "IV_ack": float(iv_ack), "survival": float(surv),
            "mono_ok": bool(vl1 <= vl2 + TOL_MONO_REL * max(1e-30, abs(vl2))),
            "capped": bool(self.A0.capped() or self.A0p.capped()
                           or self.VL2.capped() or self.VL1.capped()),
        }

    def states(self) -> dict:
        return {"A0": self.A0.n_states(), "A0p": self.A0p.n_states(),
                "VL2": self.VL2.n_states(), "VL1": self.VL1.n_states(),
                "stat_A0p": self.A0p.stat(), "stat_VL1": self.VL1.stat(),
                "stat_VL2": self.VL2.stat(), "stat_A0": self.A0.stat(),
                "capped": bool(self.A0.capped() or self.A0p.capped()
                               or self.VL2.capped() or self.VL1.capped())}


# ==================================================================== 面板
def panel(n_m_list=N_M_LIST, tau_list=TAU_LIST, b_list=B_LIST, *,
          n_users: int = 1, sigma_w=SIGMA_W1, gh_n: int = GH_MAIN,
          z_dz: float = Z_DZ, tag: str = "main") -> dict:
    rows: list[dict] = []
    legs: list[dict] = []
    for n_m in n_m_list:
        cfg = ack_config(RHO, sigma_w if n_users == 1 else SIGMA_W1_2U)
        t0 = time.perf_counter()
        qd = Quad(cfg, n_m, gh_n=gh_n, z_dz=z_dz)
        for tau in tau_list:
            for b in b_list:
                r = qd.cell(tau, b)
                r["arm"] = tag
                rows.append(r)
        st = qd.states()
        legs.append({"tag": tag, "n_users": n_users, "n_m": n_m, "sec": round(
            time.perf_counter() - t0, 3), "sec_build": qd.sec_build, "states": st})
        print(f"  [{tag} nu={n_users} n_m={n_m}] {legs[-1]['sec']} s "
              f"states={ {k: st[k] for k in ('A0', 'A0p', 'VL2', 'VL1')} } "
              f"clamp={st['stat_VL1']['n_snap_clamped']} capped={st['capped']}", flush=True)
    return {"rows": rows, "legs": legs}


# ==================================================================== 闸
def g1_copy_identity() -> dict:
    """`G1-copy`：`ack_updates=False` 时必须与 `r5_2.build_engine` **逐位**相同。"""
    rows = []
    worst = 0.0
    for n_m, tau, b, pt in ((9, 5, 16, 0), (9, 6, 16, 0), (17, 4, 8, 0), (9, 5, 8, 0)):
        cfg = ack_config(RHO, SIGMA_W1)
        mine = build_engine_l1(cfg, SIGMA_STAR, n_m, pt, gh_n=GH_MAIN,
                               probe_enabled=True, ack_updates=False, n_users=1)
        ref = r52_engine(cfg, SIGMA_STAR, n_m, pt, gh_n=GH_MAIN,
                         probe_enabled=True, n_users=1)
        a = mine(tau, (P0,), (0.0,), (V_PRIOR,), b)
        c = ref(tau, (P0,), (0.0,), (V_PRIOR,), b)
        d = abs(a - c)
        worst = max(worst, d)
        rows.append({"n_m": n_m, "tau": tau, "b_ticks": b, "mine": float(a),
                     "ref": float(c), "abs": float(d)})
    return {"rows": rows, "max_abs": float(worst), "tol": TOL_BIT,
            "pass": bool(rows and worst <= TOL_BIT)}


def g2_ackpost() -> dict:
    """`G2-ackpost`：ADF 后验矩 vs 固定域参考网格（641 点）后验矩；且 `v_post < v`。

    ★ 321 点（`world.py` 原分辨率）的值也报，但**不进闸的判定**：它自己的矩误差在
      `v=0.1` 处就到 `1e-3` 量级（每 sd 6.3 格），用它判定等于拿参考物的粗格当被测物的错。
    """
    rows = []
    worst = 0.0
    worst321 = 0.0
    n_shrink_fail = 0
    for m, v in ((0.0, 1.0), (0.0, 0.5), (0.5, 0.25), (-1.0, 1.0), (1.5, 0.1)):
        for ok in (True, False):
            a_m, a_v = ack_posterior_moments(m, v, ok, 2.0, 0.0)
            r_m, r_v = ack_posterior_moments_grid(m, v, ok, 2.0, 0.0, n_grid=641)
            q_m, q_v = ack_posterior_moments_grid(m, v, ok, 2.0, 0.0, n_grid=321)
            dm = abs(a_m - r_m) / max(1e-12, abs(r_m) + 1.0)
            dv = abs(a_v - r_v) / max(1e-12, abs(r_v))
            dm321 = abs(a_m - q_m) / max(1e-12, abs(q_m) + 1.0)
            dv321 = abs(a_v - q_v) / max(1e-12, abs(q_v))
            shrink = a_v < v
            if not shrink:
                n_shrink_fail += 1
            worst = max(worst, dm, dv)
            worst321 = max(worst321, dm321, dv321)
            rows.append({"m": m, "v": v, "success": ok, "adf_m": a_m, "adf_v": a_v,
                         "grid641_m": r_m, "grid641_v": r_v,
                         "grid321_m": q_m, "grid321_v": q_v,
                         "rel_dm": float(dm), "rel_dv": float(dv),
                         "rel_dm_321": float(dm321), "rel_dv_321": float(dv321),
                         "shrink": bool(shrink)})
    return {"rows": rows, "max_rel": float(worst), "tol": G2_TOL,
            "max_rel_vs_321": float(worst321),
            "n_shrink_fail": int(n_shrink_fail),
            "pass": bool(worst <= G2_TOL and n_shrink_fail == 0)}


def g3_noinfo() -> dict:
    """`G3-noinfo`（活体负控）：`κ=0` ⇒ 似然恒 1/2 ⇒ 后验不动。"""
    rows = []
    worst = 0.0
    for m, v in ((0.0, 1.0), (0.7, 0.3), (-2.0, 2.0)):
        for ok in (True, False):
            a_m, a_v = ack_posterior_moments(m, v, ok, 0.0, 0.0)
            d = max(abs(a_m - m), abs(a_v - v))
            worst = max(worst, d)
            rows.append({"m": m, "v": v, "success": ok, "post_m": a_m, "post_v": a_v,
                         "abs": float(d)})
    return {"rows": rows, "max_abs": float(worst), "tol": G3_TOL,
            "pass": bool(worst <= G3_TOL)}


def g3b_clamp_count(legs) -> dict:
    """`G3b`：`m` 的 snap 截断次数必须可数并报出（不是闸的判定项，是分辨率报告）。"""
    tot = 0
    detail = {}
    for lg in legs:
        n = int(lg["states"]["stat_VL1"]["n_snap_clamped"]) + int(
            lg["states"]["stat_A0p"]["n_snap_clamped"])
        detail[f"n_m={lg['n_m']}"] = n
        tot += n
    return {"total": tot, "per_leg": detail, "pass": True}


def g_struct(rows) -> dict:
    """`G-struct`：`b=8` 上 `A0p==A0` 且 `VL1==VL2`（预注册 §3 `P-STRUCT`）。"""
    sub = [r for r in rows if r["b_ticks"] == 8]
    worst = 0.0
    rows_o = []
    for r in sub:
        d1 = abs(r["A0p"] - r["A0"])
        d2 = abs(r["VL1"] - r["VL2"])
        worst = max(worst, d1, d2)
        rows_o.append({"n_m": r["n_m"], "tau": r["tau"], "A0": r["A0"], "A0p": r["A0p"],
                       "VL2": r["VL2"], "VL1": r["VL1"], "abs_A0": float(d1),
                       "abs_V": float(d2), "IV_ack": r["IV_ack"],
                       "survival": r["survival"]})
    return {"rows": rows_o, "max_abs": float(worst), "tol": TOL_STRUCT,
            "pass": (bool(worst <= TOL_STRUCT) if rows_o else None),
            "note": ("该面板不含 b=8 ⇒ 不适用" if not rows_o else "")}


def g4_mono(rows) -> dict:
    """`G4-mono`：每格 `VL1 ≤ VL2`（信息单调性）；违例单列。"""
    viol = [r for r in rows if not r["mono_ok"]]
    worst = 0.0
    for r in rows:
        worst = max(worst, (r["VL1"] - r["VL2"]) / max(1e-30, abs(r["VL2"])))
    return {"n_cells": len(rows), "n_violations": len(viol), "worst_rel": float(worst),
            "tol_rel": TOL_MONO_REL,
            "violations": [{k: r[k] for k in ("n_m", "tau", "b_ticks", "VL1", "VL2")}
                           for r in viol[:10]],
            "pass": bool(not viol and rows)}


def g5_dropobs(n_m: int = 9) -> dict:
    """`G5-dropobs`：丢弃探测读数必须**改变数值**（变异测试）。

    ★ 读后修正（保留原 fixture 的结果，不删）：首版只用 `(n=1, τ=5, b=16)`，而该格
      `IV_probe_L2 ≡ 0`（探测**从不被最优策略选中**）⇒ 变异改不到任何东西，闸必然"失败"。
      这是 **空闸**（R4.1 教训：空过的闸不算闸；反过来，空 fixture 的失败也不算缺陷）。
      现加一条**活体 fixture** `(n=2, b=8, τ=5)`（该格 `IV_probe_L2=6.86e-4 > 0`，探测真的被用上）：
      判定只看活体那条，空闸那条作为读数一并报出。
    """
    out = []
    n_changed_live = 0
    fixtures = (("n1_empty", 1, 5, 16), ("n2_live", 2, 5, 8))
    for tag, nu, tau, b in fixtures:
        cfg = ack_config(RHO, SIGMA_W1 if nu == 1 else SIGMA_W1_2U)
        for ack in (False, True):
            good = build_engine_l1(cfg, SIGMA_STAR, n_m, 0, gh_n=GH_MAIN,
                                   probe_enabled=True, ack_updates=ack, n_users=nu)
            bad = build_engine_l1(cfg, SIGMA_STAR, n_m, 0, gh_n=GH_MAIN,
                                  probe_enabled=True, ack_updates=ack, n_users=nu,
                                  drop_obs=True)
            pv = tuple(P0 for _ in range(nu))
            zv = tuple(0.0 for _ in range(nu))
            vv = tuple(V_PRIOR for _ in range(nu))
            g = good(tau, pv, zv, vv, b)
            b_ = bad(tau, pv, zv, vv, b)
            changed = abs(g - b_) > 1e-15
            if tag == "n2_live":
                n_changed_live += int(changed)
            out.append({"fixture": tag, "n_users": nu, "tau": tau, "b_ticks": b,
                        "ack_updates": ack, "V_good": float(g), "V_dropobs": float(b_),
                        "dV": float(g - b_), "changed": bool(changed)})
    n_changed_all = sum(1 for r in out if r["changed"])
    return {"rows": out, "n_changed": int(n_changed_all),
            "n_changed_live": int(n_changed_live),
            "n_changed_empty_fixture": int(n_changed_all - n_changed_live),
            "pass": bool(n_changed_live >= 1)}


def g6_budget() -> dict:
    """`G6-budget`：`A0(b=8) > A0(b=16)` 严格（预算确实接进引擎）。"""
    cfg = ack_config(RHO, SIGMA_W1)
    e = build_engine_l1(cfg, SIGMA_STAR, 9, 0, gh_n=GH_MAIN,
                        probe_enabled=False, ack_updates=False, n_users=1)
    a8 = e(5, (P0,), (0.0,), (V_PRIOR,), 8)
    a16 = e(5, (P0,), (0.0,), (V_PRIOR,), 16)
    return {"A0_b8": float(a8), "A0_b16": float(a16), "gap": float(a8 - a16),
            "pass": bool(a16 < a8)}


def g7_size(legs) -> dict:
    """`G7-size`：每条腿四臂 `n_states>0`、无撞顶。"""
    rows = []
    ok = True
    for lg in legs:
        st = lg["states"]
        c1 = all(st[k] > 0 for k in ("A0", "A0p", "VL2", "VL1"))
        ok = ok and c1 and not st["capped"]
        rows.append({"tag": lg["tag"], "n_m": lg["n_m"],
                     "states": {k: st[k] for k in ("A0", "A0p", "VL2", "VL1")},
                     "c1_all_positive": bool(c1), "capped": st["capped"]})
    return {"rows": rows, "pass": bool(ok and rows)}


def g8_distinct(n_m_list) -> dict:
    """`G8-fp`：`n=1` 与 `n=2` 两个 plant 指纹互异；`n_m` 三点互异。"""
    fps = {}
    for tag, sw in (("1u", SIGMA_W1), ("2u", SIGMA_W1_2U)):
        cfg = ack_config(RHO, sw)
        fps[tag] = {"A": list(cfg.plant.A), "sigma_w": list(cfg.plant.sigma_w),
                    "fp": cfg.fingerprint()}
    uniq = len({v["fp"] for v in fps.values()})
    return {"per_plant": fps, "n_distinct": uniq, "n_plants": len(fps),
            "n_m": list(n_m_list), "n_distinct_n_m": len(set(n_m_list)),
            "pass": bool(uniq == len(fps) and len(set(n_m_list)) == len(n_m_list))}


def g9_ast() -> dict:
    """`G9-ast`：无 `int(p)/int(price)/int(probe_cost)`；检测器对故意变异命中。"""
    src = Path(__file__).read_text(encoding="utf-8")
    hits = []
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "int":
            if not n.args:
                continue
            a0 = n.args[0]
            name = a0.id if isinstance(a0, ast.Name) else (
                a0.attr if isinstance(a0, ast.Attribute) else "")
            if name in ("p", "price", "probe_cost", "probe_price"):
                hits.append({"line": getattr(n, "lineno", None), "arg": name})
    probe_src = "def f(p):\n    return int(p)\n"
    probe_hits = [n for n in ast.walk(ast.parse(probe_src))
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                  and n.func.id == "int"]
    return {"hits": hits, "n_hits": len(hits),
            "detector_catches_mutation": bool(probe_hits),
            "pass": bool(not hits and probe_hits)}


def g10_ck() -> dict:
    """`G10-ck`：ADF 参考积分加密（`dz 0.025 → 0.0125`）时后验矩的相对变化。"""
    rows = []
    worst = 0.0
    for m, v in ((0.0, 1.0), (1.5, 0.1), (-2.0, 2.0), (0.5, 0.02)):
        for ok in (True, False):
            a_m, a_v = ack_posterior_moments(m, v, ok, 2.0, 0.0, z_dz=Z_DZ)
            b_m, b_v = ack_posterior_moments(m, v, ok, 2.0, 0.0, z_dz=Z_DZ_REFINE)
            dm = abs(a_m - b_m) / max(1e-12, abs(b_m) + 1.0)
            dv = abs(a_v - b_v) / max(1e-12, abs(b_v))
            worst = max(worst, dm, dv)
            rows.append({"m": m, "v": v, "success": ok, "rel_dm": float(dm),
                         "rel_dv": float(dv)})
    return {"rows": rows, "max_rel": float(worst), "tol": 1e-3,
            "pass": bool(worst <= 1e-3)}


# ==================================================================== 判词
def judge(rows, gates, n_m_list) -> dict:
    out: dict = {"n_cells": len(rows),
                 "n_capped": sum(1 for r in rows if r["capped"]),
                 "n_unmeasured_survival": sum(1 for r in rows if not math.isfinite(r["survival"]))}
    out["gates"] = {k: {kk: vv for kk, vv in v.items()
                        if kk in ("pass", "max_abs", "max_rel", "n_violations",
                                  "n_changed", "n_shrink_fail", "gap", "tol", "total")}
                    for k, v in gates.items()}
    out["gates_failed"] = sorted(k for k, v in gates.items()
                                  if v.get("pass") is False)   # None = 不适用，不算失败
    out["gates_n_fail"] = len(out["gates_failed"])

    def cells(b=None):
        return [r for r in rows if b is None or r["b_ticks"] == b]

    b_all = sorted({r["b_ticks"] for r in rows})
    b_main = b_all[-1] if b_all else 16          # 主判用的预算 = 面板里最大的那条

    # ---- P-STRUCT（只在面板含 `b=8` 时才有对象）
    if 8 in b_all:
        out["P_STRUCT"] = ("PASS" if gates.get("g_struct", {}).get("pass")
                           else "FAIL-WIRING-SUSPECT-B8")
    else:
        out["P_STRUCT"] = "NOT-APPLICABLE-NO-B8"

    # ---- P-A：探测有没有价值（L2 口径）
    p2 = [r for r in cells(None) if r["IV_probe_L2"] > IV_NOISE]
    signs = {}
    for tau in sorted({r["tau"] for r in rows}):
        for b in sorted({r["b_ticks"] for r in rows}):
            sub = [r for r in cells(b) if r["tau"] == tau]
            pos = sum(1 for r in sub if r["IV_probe_L2"] > IV_NOISE)
            neg = sum(1 for r in sub if r["IV_probe_L2"] < -IV_NOISE)
            signs[f"tau{tau}_b{b}"] = {"n": len(sub), "pos": pos, "neg": neg}
    out["sign_profile_IV_probe_L2"] = signs
    out["P_A"] = "PROBE-WORTHLESS-AT-N1" if not p2 else (
        "GRID-UNRESOLVED" if any(v["pos"] > 0 and v["neg"] > 0 for v in signs.values())
        else "PROBE-HAS-VALUE")

    # ---- P-B：主判（b=16 的 survival）
    b16 = [r for r in cells(b_main) if r["b_ticks"] == b_main]
    surv_by_tau = {}
    for tau in sorted({r["tau"] for r in b16}):
        sub = sorted([r for r in b16 if r["tau"] == tau], key=lambda r: r["n_m"])
        vals = [r["survival"] for r in sub]
        four = [r for r in sub if math.isfinite(r["survival"])]
        spread = (max(r["survival"] for r in four) - min(r["survival"] for r in four)) if four else float("nan")
        surv_by_tau[f"tau{tau}"] = {
            "n": len(sub), "n_measured": len(four),
            "survival": vals,
            "survival_finest": (four[-1]["survival"] if four else float("nan")),
            "spread": float(spread),
            "IV_probe_L2": [r["IV_probe_L2"] for r in sub],
            "IV_probe_L1": [r["IV_probe_L1"] for r in sub],
            "IV_ack": [r["IV_ack"] for r in sub],
        }
    out["P_B_by_tau"] = surv_by_tau

    def verdict_from(s: dict) -> str:
        if s["n_measured"] == 0:
            return "UNMEASURED"
        v = s["survival_finest"]
        if not math.isfinite(v):
            return "UNMEASURED"
        if v > 1.0 + 1e-6:
            return "ADF-BREAKS-MONOTONICITY"
        if s["spread"] > SURVIVAL_SPREAD_MAX:
            return "GRID-UNRESOLVED"
        if v <= 0.10:
            return "ACK-EATS-PROBE"
        if v >= 0.90:
            return "ACK-DOES-NOT-EAT-PROBE"
        return "ACK-PARTIALLY-EATS-PROBE"

    out["P_B"] = {k: verdict_from(v) for k, v in surv_by_tau.items()}

    # ---- P-C：ACK 本身有没有价值
    b8 = [r for r in cells(8)]
    p_c8_ok = (all(abs(r["IV_ack"]) <= TOL_STRUCT for r in b8) if b8 else None)
    b16_ack = [r["IV_ack"] for r in b16]
    # ★ 读后追加（不改预注册规则，只补一项读数）：`IV_ack < −1e-12` 在预注册里**没有对应判词**，
    #   因为 §1 的推导说它不可能出现（信息更多不能更差）。它一旦出现，就是**近似误差**的直接证据，
    #   不许被 `≤ 1e-12 ⇒ ACK-WORTHLESS` 这条规则吞成"ACK 无用"。
    n_neg = sum(1 for v in b16_ack if v < -IV_NOISE)
    out["P_C"] = {"b8_ack_zero_as_predicted": bool(p_c8_ok),
                  "b16_IV_ack": b16_ack,
                  "b16_any_above_noise": bool(any(v > IV_NOISE for v in b16_ack)),
                  "b16_n_negative_impossible": int(n_neg),
                  "b16_min_IV_ack": float(min(b16_ack)) if b16_ack else float("nan"),
                  "verdict": ("ACK-HAS-VALUE-AT-B16" if any(v > IV_NOISE for v in b16_ack)
                              else ("ACK-NEGATIVE-IMPOSSIBLE-ADF-BIAS" if n_neg else
                                    "ACK-WORTHLESS-AT-B16"))}

    # ---- 总判词
    if out["gates_n_fail"] > 0:
        out["VERDICT"] = "NO-VERDICT-GATE-FAIL"
    elif out["P_STRUCT"] == "FAIL-WIRING-SUSPECT-B8":
        out["VERDICT"] = "WIRING-SUSPECT-B8"
    elif len(set(n_m_list)) < 3:
        out["VERDICT"] = "NO-VERDICT-INSUFFICIENT-GRID"
    elif out["P_A"] == "PROBE-WORTHLESS-AT-N1":
        out["VERDICT"] = "PROBE-WORTHLESS-AT-N1"
    else:
        vs = [v for v in out["P_B"].values() if v not in ("UNMEASURED",)]
        out["VERDICT"] = ("SURVIVAL-" + "/".join(sorted(set(vs)))) if vs else "UNMEASURED"
    out["verdict_scope"] = (
        "只覆盖 n=1（或备用 n=2）、ρ=0.999、P0=0.01、κ=2、σ_w=(0.02,)、p=0、"
        "τ∈{4,5,6}、b∈{8,16}、GH=9；L1 = ADF 近似，不是 L1 最优；不许跨 ρ 与 R4–R5 相减")
    return out


# ==================================================================== main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="CW-R5.5 L1 语义下 ACK vs PROBE")
    ap.add_argument("--size-only", action="store_true")
    ap.add_argument("--ctrl-only", action="store_true")
    ap.add_argument("--fallback-2u", action="store_true", help="备用实例 n=2（预注册 §4）")
    ap.add_argument("--diag-b24", action="store_true",
                    help="读后追加诊断：n=2、b=24（3 发送 / 2 用户，唯一允许同一用户重复发送的格）")
    ap.add_argument("--n-m", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    n_m_list = tuple(int(x) for x in args.n_m.split(",")) if args.n_m else N_M_LIST

    t_all = time.perf_counter()
    # ★ 横幅必须写**这次真的会跑**的规格：读后追加的 `--diag-b24` 曾让横幅写成 `tau=(4,5,6) b=(8,16)`，
    #   而实际跑的是 `(5,6)/(24,)` —— 日志与实测不符就是假证据。
    if args.diag_b24:
        _nu, _taus, _bs = 2, (5, 6), (24,)
    elif args.fallback_2u:
        _nu, _taus, _bs = 2, (4, 5), B_LIST
    else:
        _nu, _taus, _bs = 1, TAU_LIST, B_LIST
    print(f"[{ROUND}] {PROTOCOL} nu={_nu} rho={RHO} tau={_taus} b={_bs} p=0 n_m={n_m_list}",
          flush=True)

    # ★ 读后修正：`run-id` 只到秒，且前缀不含模式 ⇒ `main1u` 与 `fallback2u` 两次跑
    #   （间隔 <1 s，`src_hash` 相同）**撞同一个目录**，后者静默覆盖前者。
    #   修法：模式进目录名。`_SUPERSEDED` 的旧目录保留不动。
    _mode_tag = ("diag_b24" if args.diag_b24 else
                 ("fallback2u" if args.fallback_2u else
                  ("size" if args.size_only else ("ctrl" if args.ctrl_only else "main1u"))))
    dest = Path(args.out) if args.out else (
        RESULTS / f"cw_r5_5_l1_ack_{_mode_tag}_{_now_utc()}_{_src_hash()[:16]}")
    dest.mkdir(parents=True, exist_ok=True)

    if args.ctrl_only:
        gates = {"g1_copy": g1_copy_identity(), "g2_ackpost": g2_ackpost(),
                 "g3_noinfo": g3_noinfo(), "g6_budget": g6_budget(),
                 "g8_fp": g8_distinct(n_m_list), "g9_ast": g9_ast(),
                 "g10_ck": g10_ck()}
        print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "rows"}
                          for k, v in gates.items()}, indent=2, ensure_ascii=False)[:8000])
        print(f"[ctrl-only] n_fail={sum(1 for v in gates.values() if not v['pass'])} "
              f"/ {len(gates)}", flush=True)
        (dest / "r55_ctrl.json").write_text(
            json.dumps({"round": ROUND, "protocol": PROTOCOL, "utc": _now_utc(),
                        "gates": gates}, indent=2, ensure_ascii=False), encoding="utf-8")
        return 0

    if args.size_only:
        # 逐 n_m 升序：任一腿撞顶就停（n_m 单调）
        out = []
        for n_m in (n_m_list if args.n_m else (5, 9, 13, 17, 25, 33)):
            cfg = ack_config(RHO, SIGMA_W1)
            t0 = time.perf_counter()
            qd = Quad(cfg, n_m)
            for tau in TAU_LIST:
                for b in B_LIST:
                    qd.cell(tau, b)
            st = qd.states()
            rec = {"n_m": n_m, "sec": round(time.perf_counter() - t0, 3),
                   "states": {k: st[k] for k in ("A0", "A0p", "VL2", "VL1")},
                   "capped": st["capped"],
                   "clamp": st["stat_VL1"]["n_snap_clamped"],
                   "fallback": st["stat_VL1"]["n_ackpost_fallback"]}
            out.append(rec)
            print(f"  [size n_m={n_m}] {rec['sec']} s states={rec['states']} "
                  f"capped={rec['capped']} clamp={rec['clamp']}", flush=True)
            if rec["capped"]:
                print(f"  [size] n_m={n_m} 撞顶 ⇒ 停止", flush=True)
                break
        (dest / "r55_size.json").write_text(
            json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[evidence] {dest}", flush=True)
        return 0

    gates: dict = {"g1_copy": g1_copy_identity(), "g2_ackpost": g2_ackpost(),
                   "g3_noinfo": g3_noinfo(), "g6_budget": g6_budget(),
                   "g8_fp": g8_distinct(n_m_list), "g9_ast": g9_ast(),
                   "g10_ck": g10_ck()}
    print(f"[ctrl] {json.dumps({k: v.get('pass') for k, v in gates.items()})}", flush=True)

    t0 = time.perf_counter()
    if args.fallback_2u:
        pnl = panel(n_m_list, (4, 5), B_LIST, n_users=2, gh_n=GH_MAIN, tag="fallback2u")
    elif args.diag_b24:
        # ★ 读后追加（预注册 §9-2）：`n=2, b=24` = 3 次发送 / 2 用户，
        #   是"探测有价值（发送次数<用户数 不成立）"与"ACK 有下游用途（同一用户重复发送）"
        #   唯一可能同时成立的格。**探索性，不进主判据。**
        pnl = panel(n_m_list, (5, 6), (24,), n_users=2, gh_n=GH_MAIN, tag="diag_b24")
    else:
        pnl = panel(n_m_list, TAU_LIST, B_LIST, n_users=1, gh_n=GH_MAIN, tag="main1u")
    sec_panel = round(time.perf_counter() - t0, 3)

    gates["g_struct"] = g_struct(pnl["rows"])
    gates["g4_mono"] = g4_mono(pnl["rows"])
    gates["g5_dropobs"] = g5_dropobs(n_m=9)
    gates["g7_size"] = g7_size(pnl["legs"])
    gates["g3b_clamp"] = g3b_clamp_count(pnl["legs"])
    v = judge(pnl["rows"], gates, n_m_list)

    print(json.dumps({k: v[k] for k in ("VERDICT", "P_A", "P_B", "P_C", "P_STRUCT",
                                        "gates_n_fail", "gates_failed")},
                     indent=2, ensure_ascii=False), flush=True)
    for r in pnl["rows"]:
        print(f"  n_m={r['n_m']:>3} tau={r['tau']} b={r['b_ticks']:>3} "
              f"A0={r['A0']:.8e} A0p={r['A0p']:.8e} IVp2={r['IV_probe_L2']:+.4e} "
              f"IVp1={r['IV_probe_L1']:+.4e} IVack={r['IV_ack']:+.4e} "
              f"surv={r['survival']:.6f}", flush=True)

    out = {"round": ROUND, "protocol": PROTOCOL, "utc": _now_utc(),
           "ticks_per_quantum": TICKS_PER_QUANTUM, "send_ticks": SEND_TICKS,
           "sec_panel": sec_panel, "sec_total": round(time.perf_counter() - t_all, 3),
           "panel": {"rho": RHO, "P": P0, "tau": list(TAU_LIST), "b_ticks": list(B_LIST),
                     "n_m_main": list(n_m_list), "gh_n": GH_MAIN, "z_dz": Z_DZ,
                     "probe_ticks": PROBE_TICKS_MAIN,
                     "sigma_w": list(ack_config(RHO, SIGMA_W1).plant.sigma_w),
                     "fallback_2u": bool(args.fallback_2u)},
           "cells": pnl["rows"], "legs": pnl["legs"], "gates": gates, "verdicts": v}
    (dest / "r55.json").write_text(json.dumps(out, indent=2, ensure_ascii=False),
                                   encoding="utf-8")
    print(f"[evidence] {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
