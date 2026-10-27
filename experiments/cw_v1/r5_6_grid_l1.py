# -*- coding: utf-8 -*-
"""`CW-R5.6` 控制件：**网格信念（直方图）** 上的 L1 窗口 DP。

预注册：`docs/research/CW-R5.6-网格信念L1-预注册-v0.1.md`（写于任何本轮实现与读数之前）。
判据方向、动作空间限制、闸的**被测物**、否定条件全部跑前写死，本文件只实现。

为什么换表示
------------
R5.5 把 L1（ACK 更新信道信念）做成 **ADF（单高斯矩匹配）**，结果
`n=1,b=16` 上 `A0p > A0`（9/9 格）—— 信息更多反而更差，物理上不可能。
诊断实测：`(m,v)` 的矩对到 `9.4e-11`，但 DP 真正消费的 `ps(m,v)` 偏
`−4.25e-3`(NACK)/`+1.85e-3`(ACK) ⇒ **中间量对得上 ≠ 被测物正确**。
本轮换 **网格直方图 + 精确似然乘后归一**：`ps` 变成精确求和，没有矩匹配这一步。

与 v1.4-ack 的**唯一**差别（`CW-POMDP-v1.5-grid`）
------------------------------------------------
| 项 | v1.4-ack | v1.5-grid |
|---|---|---|
| 信念 | `(m, v)` 单高斯 | `n_w` 节点直方图 `p` |
| ACK | ADF 矩匹配 | `p ⊙ lik_y` 后归一（**精确**） |
| 传播 | `m'=ρm, v'=ρ²v+(1−ρ²)` | `p' = p @ T`，`T[i,j]∝φ((x_j−ρx_i)/√(1−ρ²))` 按行归一 |
| 探测 | 高斯观测 + GH 求积 | **点观测** `p ← δ_j`（`σ_csi→0`） |
| `ps` | 80 点 GH 积分 | `Σ_j p_j σ(κ(x_j−c))` **精确求和** |

`plant` / 动作集 / 预算刻度 / `ρ` / `κ` / 目标函数 **一字节不动**。

★ `P` 维度**不离散化**（沿用 R5.2/R5.5 的浮点递推）⇒ 本轮误差**只来自 `n_w`**，
  这是相对 R5.5（`m` 还要 snap 到 `n_m` 个中心）的一个净改进。

★ 动作空间限制（预注册 §2，**不是近似**）：`probe_max=1`（每用户至多探测一次），
  另在最小可算格上用 `probe_max=2` 量一次限制损失；算不动就写 `UNMEASURED`。
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import time
from pathlib import Path

import numpy as np

from .r1_14_exactbase import _now_utc, _src_hash
from .r3_2_iv2u import STATE_CAP
from .r5_1_price_ticks import SEND_TICKS, TICKS_PER_QUANTUM
from .r5_5_l1_ack import (P0, RHO, SIGMA_W1, SIGMA_W1_2U, V_PRIOR, ack_config)

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results" / "cw_v1"

PROTOCOL = "CW-POMDP-v1.5-grid"
ROUND = "CW-R5.6"

#: 面板常数（预注册 §4）
HALF = 4.0                                  # 网格支撑 `log h ∈ [-4,4]`
N_W_LIST: tuple[int, ...] = (129, 257, 513)  # 三点分辨率（Δ = 8/(n_w−1)）
N_W_SIZE: tuple[int, ...] = (33, 65, 129, 257)
TAU_LIST: tuple[int, ...] = (5, 6)
B_LIST: tuple[int, ...] = (8, 16, 24)
PROBE_TICKS_MAIN = 0                        # p=0：免量子、**不免槽**
PROBE_MAX_MAIN = 1                          # 预注册 §2-1 的动作空间限制

#: 判据常数（预注册 §3）
IV_NOISE = 1e-12
TOL_MART = 1e-9
TOL_PS_REF = 1e-3
TOL_NOINFO = 1e-12
TOL_STRUCT = 1e-13
TOL_NW3_REL = 0.30
SURVIVAL_SPREAD_MAX = 0.30


# ==================================================================== 信念网格
class GridCtx:
    """`n_w` 节点上的直方图信念 + **内插表**（`tuple→int id`）。

    ★ 为什么必须内插：`n_w=513` 的信念向量当 dict 键要哈希 513 个浮点，
      memo 一进百万状态就慢到不可用。内插后 DP 键里只有小整数 id。
    ★ 用**精确字节**做内插键（不四舍五入）：浮点运算是确定性的 ⇒ 同一观测历史
      得同一字节串；四舍五入会**静默合并**两个不同的信念（那是给数据造假）。
    """

    def __init__(self, n_w: int, rho: float, kappa: float, log_h_star: float,
                 half: float = HALF) -> None:
        self.n_w = int(n_w)
        x = np.linspace(-float(half), float(half), self.n_w)
        self.x = x
        self.dx = float(x[1] - x[0])
        dens = np.exp(-0.5 * x * x)
        dens = dens / dens.sum()
        self.prior = dens
        self.lik_ok = 1.0 / (1.0 + np.exp(-float(kappa) * (x - float(log_h_star))))
        self.lik_fail = 1.0 - self.lik_ok
        s = math.sqrt(max(1e-12, 1.0 - rho * rho))
        diff = (x[None, :] - float(rho) * x[:, None]) / s     # [源 i, 目标 j]
        T = np.exp(-0.5 * diff * diff)
        T = T / T.sum(axis=1, keepdims=True)                 # ★ 按**目标**行归一
        self.T = T
        self.s_eff = float(s)
        #: 内插表
        self._ids: dict[bytes, int] = {}
        self._arr: list[np.ndarray] = []
        #: 缓存
        self._ps: dict[int, float] = {}
        self._prop: dict[int, int] = {}
        self._ack: dict[tuple[int, bool], int] = {}

    # ---------------- 内插
    def intern(self, p: np.ndarray) -> int:
        a = np.ascontiguousarray(p, dtype=np.float64)
        key = a.tobytes()
        got = self._ids.get(key)
        if got is not None:
            return got
        i = len(self._arr)
        self._ids[key] = i
        self._arr.append(a)
        return i

    def arr(self, bid: int) -> np.ndarray:
        return self._arr[bid]

    def n_beliefs(self) -> int:
        return len(self._arr)

    # ---------------- 三个基本算子（与 `world.GridBelief` 同式）
    def ps_of(self, bid: int) -> float:
        got = self._ps.get(bid)
        if got is None:
            got = float(self._arr[bid] @ self.lik_ok)
            self._ps[bid] = got
        return got

    def prop(self, bid: int) -> int:
        got = self._prop.get(bid)
        if got is None:
            got = self.intern(self._arr[bid] @ self.T)
            self._prop[bid] = got
        return got

    def ack(self, bid: int, success: bool) -> int:
        k = (bid, bool(success))
        got = self._ack.get(k)
        if got is None:
            lik = self.lik_ok if success else self.lik_fail
            q = self._arr[bid] * lik
            z = float(q.sum())
            if not np.isfinite(z) or z <= 0.0:
                raise ValueError(f"ACK 后后验质量非法（{z}）—— 逐位零的格应报 unmeasured，不许出 NaN")
            got = self.intern(self.prop_raw(q / z))
            self._ack[k] = got
        return got

    def prop_raw(self, p: np.ndarray) -> np.ndarray:
        out = p @ self.T
        z = float(out.sum())
        if not np.isfinite(z) or z <= 0.0:
            raise ValueError("传播后质量非法")
        return out / z

    # ---------------- 先验 / 点观测
    def prior_id(self) -> int:
        return self.intern(self.prior)

    def delta_id(self, j: int) -> int:
        e = np.zeros(self.n_w)
        e[int(j)] = 1.0
        return self.intern(e)

    def probe_belief_ids(self) -> np.ndarray:
        """点观测到节点 `j` 再传播 ⇒ 信念 = `T[j]`（`n_w` 个，**与先验无关**）。

        ★ 这条结构性事实是规模能算得动的原因：所有"探测过头"的历史都落到同一批
          `n_w` 个信念上，不随历史膨胀。
        """
        if getattr(self, "_pbid", None) is None:
            self._pbid = np.array([self.intern(self.T[j]) for j in range(self.n_w)],
                                  dtype=np.int64)
        return self._pbid


# ==================================================================== 引擎
def build_engine_grid(gw: GridCtx, cfg, probe_ticks: int, *, ack_updates: bool,
                      probe_enabled: bool, n_users: int, probe_max: int | None,
                      probe_total: int | None = None,
                      drop_obs: bool = False, send_ticks: int = SEND_TICKS,
                      state_cap: int = STATE_CAP,
                      record_policy: dict | None = None):
    """窗口 DP。`ack_updates` 是 L1 / L2 的**唯一**差别。

    · HOLD      ：信念只传播（`p@T`），`P` 走 `a²P+σ_w²`。
    · SEND_i    ：先按当前信念抽 ACK/NACK（概率 `ps`），`P` 更新后传播；
                  `ack_updates=True` 时信念也按**精确似然**更新再传播。
    · PROBE_i   ：读到节点 `j`（概率 `p[j]`）⇒ 信念 `δ_j` 再传播 = `T[j]`；
                  `P` 只传播（不加测量更新），花 `probe_ticks` 刻度、占**一个槽**。

    ★ 两条**动作空间限制**（预注册 §2，必须与读数一起报）：
      `probe_max`   = 每用户至多探测一次；
      `probe_total` = 全局至多探测 `k` 次（`n=2` 上为压住状态爆炸而加，**不是近似**）。
    """
    p = cfg.plant
    a2 = tuple(float(a) ** 2 for a in p.A)
    sw2 = tuple(float(s) ** 2 for s in p.sigma_w)
    sm2 = float(p.sigma_meas) ** 2
    q = float(p.q_cost)
    nu = int(n_users)
    pt = int(probe_ticks)
    sc = int(send_ticks)
    if pt < 0 or sc <= 0:
        raise ValueError("刻度必须是非负整数，且发送价必须 > 0")
    assert len(a2) >= nu and len(sw2) >= nu, "plant 的用户数不够"
    pbid = gw.probe_belief_ids() if probe_enabled else None
    pmax = None if probe_max is None else int(probe_max)
    ptotal = None if probe_total is None else int(probe_total)

    memo: dict[tuple, float] = {}
    state = {"capped": False}
    _last: dict = {"act": ("H",)}

    def V(t: int, Pp: tuple, bb: tuple, b: int, pm: tuple, pn: int) -> float:
        if t <= 0:
            return 0.0
        key = (t, Pp, bb, b, pm, pn)
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
        pb = tuple(gw.prop(bb[i]) for i in range(nu))
        best = V(t - 1, ff, pb, b, pm, pn)                   # HOLD
        _last["act"] = ("H",)

        if b >= sc:                                          # SEND_i（整包 sc 刻度）
            for i in range(nu):
                Pi = Pp[i]
                g = Pi / (Pi + sm2)
                p_ok = a2[i] * (1.0 - g) * Pi + sw2[i]
                p_no = a2[i] * Pi + sw2[i]
                ph = gw.ps_of(bb[i])
                aa = list(ff)
                aa[i] = p_ok
                cc = list(ff)
                cc[i] = p_no
                if ack_updates:
                    bok = list(bb)
                    bok[i] = gw.ack(bb[i], True)
                    bno = list(bb)
                    bno[i] = gw.ack(bb[i], False)
                    val = (ph * V(t - 1, tuple(aa), tuple(bok), b - sc, pm, pn)
                           + (1.0 - ph) * V(t - 1, tuple(cc), tuple(bno), b - sc, pm, pn))
                else:
                    val = (ph * V(t - 1, tuple(aa), pb, b - sc, pm, pn)
                           + (1.0 - ph) * V(t - 1, tuple(cc), pb, b - sc, pm, pn))
                if val < best:
                    best = val
                    _last["act"] = ("S", i)

        allow_probe = probe_enabled and b >= pt and t >= 2
        if allow_probe and ptotal is not None and pn >= ptotal:
            allow_probe = False
        if allow_probe:                                      # PROBE_i（占槽）
            for i in range(nu):
                if pmax is not None and pm[i] >= pmax:
                    continue
                pi = gw.arr(bb[i])
                pm2 = list(pm)
                pm2[i] += 1
                pm2t = tuple(pm2)
                tot = 0.0
                for j in range(gw.n_w):
                    wj = float(pi[j])
                    if wj <= 0.0:
                        continue
                    bi = list(pb)
                    bi[i] = pb[i] if drop_obs else int(pbid[j])
                    tot += wj * V(t - 1, ff, tuple(bi), b - pt, pm2t, pn + 1)
                if tot < best:
                    best = tot

        if record_policy is not None:
            record_policy[(t, Pp, int(b))] = _last["act"]
        out = S + best
        memo[key] = out
        return out

    def value(t, Pvec, bvec, b, pmvec=None) -> float:
        pmv = tuple(0 for _ in range(nu)) if pmvec is None else tuple(int(x) for x in pmvec)
        try:
            return float(V(int(t), tuple(float(x) for x in Pvec),
                           tuple(int(x) for x in bvec), int(b), pmv, 0))
        except MemoryError:
            state["capped"] = True
            return float("nan")

    value.n_states = lambda: len(memo)          # type: ignore[attr-defined]
    value.capped = lambda: state["capped"]      # type: ignore[attr-defined]
    value.probe_ticks = pt                      # type: ignore[attr-defined]
    value.send_ticks = sc                       # type: ignore[attr-defined]
    value.probe_max = pmax                      # type: ignore[attr-defined]
    return value


# ==================================================================== 四臂
class Quad:
    """一个 `(n_users, n_w)` 上的四个臂，`p=0`、`probe_max` 可调。"""

    def __init__(self, cfg, n_w: int, gw: GridCtx, *, probe_max: int | None = PROBE_MAX_MAIN,
                 probe_total: int | None = None,
                 probe_ticks: int = PROBE_TICKS_MAIN, drop_obs: bool = False):
        self.cfg = cfg
        self.gw = gw
        self.nu = int(cfg.plant.n_users)
        self.n_w = int(n_w)
        self.pt = int(probe_ticks)
        self.pmax = probe_max
        self.ptot = probe_total
        k = dict(n_users=self.nu, probe_max=probe_max, probe_total=probe_total,
                 drop_obs=drop_obs)
        t0 = time.perf_counter()
        self.A0 = build_engine_grid(gw, cfg, 0, probe_enabled=False,
                                    ack_updates=False, **k)
        self.A0p = build_engine_grid(gw, cfg, 0, probe_enabled=False,
                                     ack_updates=True, **k)
        self.VL2 = build_engine_grid(gw, cfg, probe_ticks, probe_enabled=True,
                                     ack_updates=False, **k)
        self.VL1 = build_engine_grid(gw, cfg, probe_ticks, probe_enabled=True,
                                     ack_updates=True, **k)
        self.sec_build = round(time.perf_counter() - t0, 3)
        self.bid_prior = gw.prior_id()

    def _args(self, b, tau):
        Pv = tuple(P0 for _ in range(self.nu))
        bv = tuple(self.bid_prior for _ in range(self.nu))
        return Pv, bv, int(b)

    def cell(self, tau: int, b: int) -> dict:
        Pv, bv, b = self._args(b, tau)
        a0 = self.A0(tau, Pv, bv, b)
        a0p = self.A0p(tau, Pv, bv, b)
        vl2 = self.VL2(tau, Pv, bv, b)
        vl1 = self.VL1(tau, Pv, bv, b)
        iv_p2 = a0 - vl2
        iv_p1 = a0p - vl1
        iv_ack = a0 - a0p
        surv = (iv_p1 / iv_p2) if iv_p2 > IV_NOISE else float("nan")
        return {
            "n_users": self.nu, "n_w": self.n_w, "tau": int(tau), "b_ticks": int(b),
            "b_quanta": b / TICKS_PER_QUANTUM, "probe_ticks": self.pt,
            "p": self.pt / TICKS_PER_QUANTUM, "rho": float(self.cfg.channel.rho_panel[0]),
            "probe_max": (None if self.pmax is None else int(self.pmax)),
            "probe_total": (None if self.ptot is None else int(self.ptot)),
            "P": P0, "A0": float(a0), "A0p": float(a0p), "VL2": float(vl2), "VL1": float(vl1),
            "IV_probe_L2": float(iv_p2), "IV_probe_L1": float(iv_p1),
            "IV_ack": float(iv_ack), "survival": float(surv),
            "mono_ok": bool(a0p <= a0),
            "ack_nonneg": bool(iv_ack >= 0.0),
            "capped": bool(self.A0.capped() or self.A0p.capped()
                           or self.VL2.capped() or self.VL1.capped()),
        }

    def states(self) -> dict:
        return {"A0": self.A0.n_states(), "A0p": self.A0p.n_states(),
                "VL2": self.VL2.n_states(), "VL1": self.VL1.n_states(),
                "n_beliefs": self.gw.n_beliefs(),
                "capped": bool(self.A0.capped() or self.A0p.capped()
                               or self.VL2.capped() or self.VL1.capped())}


def make_gw(cfg, n_w: int) -> GridCtx:
    ch = cfg.channel
    return GridCtx(int(n_w), float(ch.rho_panel[0]), float(ch.kappa),
                   float(ch.log_h_star), half=HALF)


def panel(n_w_list, tau_list=TAU_LIST, b_list=B_LIST, *, n_users: int = 1,
          probe_max: int | None = PROBE_MAX_MAIN, probe_total: int | None = None,
          tag: str = "main1u") -> dict:
    rows: list[dict] = []
    legs: list[dict] = []
    for n_w in n_w_list:
        cfg = ack_config(RHO, SIGMA_W1 if n_users == 1 else SIGMA_W1_2U)
        gw = make_gw(cfg, n_w)
        t0 = time.perf_counter()
        qd = Quad(cfg, n_w, gw, probe_max=probe_max, probe_total=probe_total)
        for tau in tau_list:
            for b in b_list:
                r = qd.cell(tau, b)
                r["arm"] = tag
                rows.append(r)
                print(f"    n_w={n_w:>4} tau={tau} b={b:>3} A0={r['A0']:.8e} "
                      f"A0p={r['A0p']:.8e} IVp2={r['IV_probe_L2']:+.4e} "
                      f"IVp1={r['IV_probe_L1']:+.4e} IVack={r['IV_ack']:+.3e} "
                      f"surv={r['survival']:.6f}", flush=True)
        st = qd.states()
        legs.append({"tag": tag, "n_users": n_users, "n_w": int(n_w),
                     "sec": round(time.perf_counter() - t0, 3),
                     "sec_build": qd.sec_build, "states": st})
        print(f"  [{tag} nu={n_users} n_w={n_w}] {legs[-1]['sec']} s "
              f"states={ {k: st[k] for k in ('A0', 'A0p', 'VL2', 'VL1')} } "
              f"n_beliefs={st['n_beliefs']} capped={st['capped']}", flush=True)
    return {"rows": rows, "legs": legs}


# ==================================================================== 闸
def g_ps_ref(n_w_list=N_W_LIST) -> dict:
    """`g_ps_ref`：网格 `ps(先验)` vs 连续参考 `p_succ_expected(0,1)`。"""
    from .posterior import p_succ_expected
    cfg = ack_config(RHO, SIGMA_W1)
    ref = p_succ_expected(0.0, V_PRIOR, cfg)
    rows = []
    worst = 0.0
    for n_w in n_w_list:
        gw = make_gw(cfg, n_w)
        v = float(gw.prior @ gw.lik_ok)
        rel = abs(v - ref) / max(1e-30, abs(ref))
        worst = max(worst, rel)
        rows.append({"n_w": int(n_w), "dx": gw.dx, "grid_ps": v, "ref_ps": ref,
                     "rel": float(rel),
                     "mass_outside_support": float(
                         1.0 - gw.prior[(gw.x >= -HALF) & (gw.x <= HALF)].sum())})
    return {"rows": rows, "ref": ref, "max_rel": float(worst), "tol": TOL_PS_REF,
            "pass": bool(worst <= TOL_PS_REF)}


def _mart_batch(gw: GridCtx, P: np.ndarray) -> np.ndarray:
    """对一批信念（行）算 `ps` 层的**鞅残差**。

    `Σ_y P(y|b)·ps(prop(b⊙lik_y)) − ps(prop(b))` —— 观测不该改变 `ps` 的期望。
    这是**精确贝叶斯必须满足的恒等式**，也是 R5.5 ADF 偏置的直接探测器。
    """
    p_ok = P @ gw.lik_ok
    q_ok = P * gw.lik_ok
    q_ok = q_ok / q_ok.sum(axis=1, keepdims=True)
    q_no = P * gw.lik_fail
    q_no = q_no / q_no.sum(axis=1, keepdims=True)
    ps_ok = (q_ok @ gw.T) @ gw.lik_ok
    ps_no = (q_no @ gw.T) @ gw.lik_ok
    ps_prop = (P @ gw.T) @ gw.lik_ok
    return np.abs(p_ok * ps_ok + (1.0 - p_ok) * ps_no - ps_prop)


def g_mart(n_w_list=N_W_LIST) -> dict:
    """★★★ `g_mart`：`ps` 层鞅残差必须在**全部节点**上 ≤ `1e-9`（不只根节点）。

    三批：① `n_w` 个 `δ_j`；② `n_w` 行 `T[j]`（点观测后的信念，DP 真的会用到）；
    ③ 先验 + 两条一步 ACK 后验。
    """
    cfg = ack_config(RHO, SIGMA_W1)
    rows = []
    worst = 0.0
    for n_w in n_w_list:
        gw = make_gw(cfg, n_w)
        ident = np.eye(gw.n_w)
        r1 = _mart_batch(gw, ident)
        r2 = _mart_batch(gw, gw.T)
        base = np.vstack([gw.prior[None, :],
                          (gw.prior * gw.lik_ok)[None, :] / (gw.prior @ gw.lik_ok),
                          (gw.prior * gw.lik_fail)[None, :] / (gw.prior @ gw.lik_fail)])
        r3 = _mart_batch(gw, base)
        m = float(max(r1.max(), r2.max(), r3.max()))
        worst = max(worst, m)
        rows.append({"n_w": int(n_w), "max_delta": float(r1.max()),
                     "max_Trow": float(r2.max()), "max_prior_family": float(r3.max()),
                     "max_all": m})
    return {"rows": rows, "max_abs": float(worst), "tol": TOL_MART,
            "pass": bool(rows and worst <= TOL_MART)}


def g_noinfo(n_w: int = 129) -> dict:
    """`g_noinfo`（活体负控）：`κ=0` ⇒ 似然恒 1/2 ⇒ 后验不动 ⇒ 四臂应相等。"""
    cfg0 = ack_config(RHO, SIGMA_W1)
    from .config import ChannelConfig, mainline_config
    ch = ChannelConfig(**{**cfg0.channel.__dict__, "kappa": 0.0})
    cfg = mainline_config(plant=cfg0.plant, channel=ch)
    gw = make_gw(cfg, n_w)
    qd = Quad(cfg, n_w, gw)
    Pv = (P0,)
    bv = (gw.prior_id(),)
    vals = {name: float(getattr(qd, name)(5, Pv, bv, 16))
            for name in ("A0", "A0p", "VL2", "VL1")}
    worst = max(abs(vals["A0p"] - vals["A0"]), abs(vals["VL1"] - vals["VL2"]))
    return {"n_w": int(n_w), "kappa": 0.0, "values": vals, "max_abs": float(worst),
            "tol": TOL_NOINFO, "pass": bool(worst <= TOL_NOINFO)}


def g_budget(n_w: int = 129) -> dict:
    """`g_budget`：`A0(b)` 关于 `b` **严格**递减（预算确实接进引擎）。"""
    cfg = ack_config(RHO, SIGMA_W1)
    gw = make_gw(cfg, n_w)
    e = build_engine_grid(gw, cfg, 0, probe_enabled=False, ack_updates=False,
                          n_users=1, probe_max=None)
    bv = (gw.prior_id(),)
    v = {b: float(e(5, (P0,), bv, b)) for b in (8, 16, 24, 32)}
    ok = all(v[b] > v[b + 8] for b in (8, 16, 24))
    return {"n_w": int(n_w), "A0": {str(k): x for k, x in v.items()},
            "gaps": {f"{b}->{b + 8}": float(v[b] - v[b + 8]) for b in (8, 16, 24)},
            "pass": bool(ok)}


def g_ast() -> dict:
    """`g_ast`：源码无 `int(p)` / `int(probe_cost)` 一类刻度截断；检测器对变异命中。"""
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
    mutant = [x for x in ast.walk(ast.parse("def f(p):\n    return int(p)\n"))
              if isinstance(x, ast.Call) and isinstance(x.func, ast.Name) and x.func.id == "int"]
    return {"hits": hits, "n_hits": len(hits), "detector_catches_mutation": bool(mutant),
            "pass": bool(not hits and mutant)}


def g_fp() -> dict:
    """`g_fp`：`n=1` 与 `n=2` 的 plant 指纹互异；`n_w` 三点分辨率互异。"""
    fps = {}
    for tag, sw in (("1u", SIGMA_W1), ("2u", SIGMA_W1_2U)):
        cfg = ack_config(RHO, sw)
        fps[tag] = {"A": list(cfg.plant.A), "sigma_w": list(cfg.plant.sigma_w),
                    "fp": cfg.fingerprint()}
    dxs = {int(n): 2 * HALF / (int(n) - 1) for n in N_W_LIST}
    return {"per_plant": fps, "n_distinct": len({v["fp"] for v in fps.values()}),
            "n_plants": len(fps), "dx_by_nw": {str(k): v for k, v in dxs.items()},
            "n_distinct_dx": len(set(dxs.values())),
            "pass": bool(len({v["fp"] for v in fps.values()}) == len(fps)
                         and len(set(dxs.values())) == len(dxs))}


def g_dropobs(n_w: int = 65) -> dict:
    """`g_dropobs`（变异测试）：丢弃探测读数必须**改变数值**（活体 fixture）。

    ★ 空闸的两个方向都算过（R4.1 / R5.5）：空 fixture 的"通过"不算闸，
      空 fixture 的"失败"也不算缺陷 ⇒ 先证明被测动作在该格真的被用上（`IV_probe_L2>0`）。
    """
    out = []
    n_live = 0
    for tag, nu, tau, b in (("n1_empty", 1, 6, 16), ("n2_live", 2, 5, 16)):
        cfg = ack_config(RHO, SIGMA_W1 if nu == 1 else SIGMA_W1_2U)
        gw = make_gw(cfg, n_w)
        for ack in (True, False):
            good = build_engine_grid(gw, cfg, 0, probe_enabled=True, ack_updates=ack,
                                     n_users=nu, probe_max=PROBE_MAX_MAIN, drop_obs=False)
            bad = build_engine_grid(gw, cfg, 0, probe_enabled=True, ack_updates=ack,
                                    n_users=nu, probe_max=PROBE_MAX_MAIN, drop_obs=True)
            Pv = tuple(P0 for _ in range(nu))
            bv = tuple(gw.prior_id() for _ in range(nu))
            g = float(good(tau, Pv, bv, b))
            d = float(bad(tau, Pv, bv, b))
            changed = abs(g - d) > 1e-15
            if tag == "n2_live" and changed:
                n_live += 1
            out.append({"fixture": tag, "n_users": nu, "tau": tau, "b_ticks": b,
                        "ack_updates": ack, "V_good": g, "V_dropobs": d,
                        "dV": float(g - d), "changed": bool(changed)})
    return {"rows": out, "n_changed": sum(1 for r in out if r["changed"]),
            "n_changed_live": int(n_live), "pass": bool(n_live >= 1)}


def g_mono(rows) -> dict:
    """★★★ `g_mono`：**每格** `A0p ≤ A0`（零容差）。违例 = 接线/数值错，不是物理。"""
    viol = [r for r in rows if not r["mono_ok"]]
    viol2 = [r for r in rows if not r["ack_nonneg"]]
    worst = max([r["A0p"] - r["A0"] for r in rows], default=0.0)
    return {"n_cells": len(rows), "n_violations": len(viol),
            "n_neg_ack": len(viol2), "worst_excess": float(worst),
            "violations": [{k: r[k] for k in ("n_w", "tau", "b_ticks", "A0", "A0p")}
                           for r in viol[:10]],
            "pass": bool(rows and not viol and not viol2)}


def g_size(legs) -> dict:
    rows = []
    ok = True
    for lg in legs:
        st = lg["states"]
        c1 = all(st[k] > 0 for k in ("A0", "A0p", "VL2", "VL1"))
        ok = ok and c1 and not st["capped"]
        rows.append({"tag": lg["tag"], "n_users": lg["n_users"], "n_w": lg["n_w"],
                     "states": {k: st[k] for k in ("A0", "A0p", "VL2", "VL1")},
                     "n_beliefs": st["n_beliefs"], "capped": st["capped"],
                     "c1_all_positive": bool(c1)})
    return {"rows": rows, "pass": bool(ok and rows)}


def g_nw3(rows) -> dict:
    """`g_nw3`：三点 `n_w` 上 `A0p` 的相对变化 ≤ 30%（否则 `GRID-UNRESOLVED`）。"""
    by = {}
    for r in rows:
        by.setdefault((r["n_users"], r["tau"], r["b_ticks"]), {})[r["n_w"]] = r["A0p"]
    rows_o = []
    worst = 0.0
    for k, d in sorted(by.items()):
        if len(d) < 3:
            continue
        vs = [d[n] for n in sorted(d)]
        lo, hi = min(vs), max(vs)
        rel = (hi - lo) / max(1e-30, abs(lo))
        worst = max(worst, rel)
        rows_o.append({"cell": {"n_users": k[0], "tau": k[1], "b_ticks": k[2]},
                       "A0p_by_nw": {str(n): v for n, v in sorted(d.items())},
                       "rel_spread": float(rel)})
    return {"rows": rows_o, "n_cells": len(rows_o), "max_rel_spread": float(worst),
            "tol": TOL_NW3_REL,
            "pass": (bool(rows_o and worst <= TOL_NW3_REL) if rows_o else None)}


# ==================================================================== 事后诊断（跑后追加，带时点）
def eval_policy_tracked(gw: GridCtx, cfg, policy: dict, tau: int, b0: int,
                        n_users: int = 1) -> float:
    """在**跟踪信念**的递推里评估一个"只看 `(t, P, b)` 的动作规则"。

    用途：`A0`（L2 臂）的最优策略是**不看 ACK** 的；把它放到"信念照常更新"的递推里
    评估，就得到该策略在**真实测度**下的期望成本。若它高于 `A0` 报出的数，
    说明 `A0` 那个数**不是它自己策略的真实成本** ⇒ L1/L2 两臂根本不在同一测度上。
    """
    p = cfg.plant
    a2 = tuple(float(a) ** 2 for a in p.A)
    sw2 = tuple(float(s) ** 2 for s in p.sigma_w)
    sm2 = float(p.sigma_meas) ** 2
    q = float(p.q_cost)
    nu = int(n_users)
    memo: dict[tuple, float] = {}

    def V(t: int, Pp: tuple, bb: tuple, b: int) -> float:
        if t <= 0:
            return 0.0
        key = (t, Pp, bb, b)
        got = memo.get(key)
        if got is not None:
            return got
        S = q * sum(Pp)
        if t == 1:
            memo[key] = S
            return S
        act = policy.get((int(t), Pp, int(b)))
        if act is None:
            raise KeyError(f"策略表缺 ({t}, {Pp}, {b}) —— 评估与提取不在同一格集上")
        ff = tuple(a2[i] * Pp[i] + sw2[i] for i in range(nu))
        pb = tuple(gw.prop(bb[i]) for i in range(nu))
        if act[0] == "H":
            best = V(t - 1, ff, pb, b)
        elif act[0] == "S":
            i = int(act[1])
            Pi = Pp[i]
            g = Pi / (Pi + sm2)
            ph = gw.ps_of(bb[i])
            aa = list(ff)
            aa[i] = a2[i] * (1.0 - g) * Pi + sw2[i]
            cc = list(ff)
            cc[i] = a2[i] * Pi + sw2[i]
            bok = list(bb)
            bok[i] = gw.ack(bb[i], True)
            bno = list(bb)
            bno[i] = gw.ack(bb[i], False)
            best = (ph * V(t - 1, tuple(aa), tuple(bok), b - int(policy["_sc"]))
                    + (1.0 - ph) * V(t - 1, tuple(cc), tuple(bno),
                                     b - int(policy["_sc"])))
        else:
            raise ValueError("该评估只支持无探测的 `A0/A0p` 策略")
        out = S + best
        memo[key] = out
        return out

    return float(V(int(tau), tuple(P0 for _ in range(nu)),
                   tuple(gw.prior_id() for _ in range(nu)), int(b0)))


def diag_l2_true(n_w: int, tau: int, b: int, n_users: int = 1) -> dict:
    """★★★ **跑后诊断**：`A0`（L2）的最优策略放到**跟踪信念**的递推里值多少。

    若 `V_piL2 > A0` ⇒ `A0` 低估了自己策略的成本（它把"失败率高"与"后续成功率低"
    的相关性抹掉了）⇒ **L1 与 L2 不在同一测度**，`IV_ack = A0 − A0p` 不是信息价值，
    `A0p ≤ A0` 这条"信息单调性"**前提不成立**（本轮的预注册 `g_mono` 是种类错）。
    """
    cfg = ack_config(RHO, SIGMA_W1 if n_users == 1 else SIGMA_W1_2U)
    gw = make_gw(cfg, n_w)
    pol: dict = {"_sc": SEND_TICKS}
    e0 = build_engine_grid(gw, cfg, 0, probe_enabled=False, ack_updates=False,
                           n_users=n_users, probe_max=None, record_policy=pol)
    Pv = tuple(P0 for _ in range(n_users))
    bv = tuple(gw.prior_id() for _ in range(n_users))
    a0 = float(e0(tau, Pv, bv, b))
    v_pi = eval_policy_tracked(gw, cfg, pol, tau, b, n_users=n_users)
    e1 = build_engine_grid(gw, cfg, 0, probe_enabled=False, ack_updates=True,
                           n_users=n_users, probe_max=None)
    a0p = float(e1(tau, Pv, bv, b))
    return {"n_w": int(n_w), "n_users": int(n_users), "tau": int(tau), "b_ticks": int(b),
            "A0_as_reported": a0, "V_piL2_true_measure": v_pi, "A0p": a0p,
            "l2_self_bias": float(v_pi - a0),
            "l2_self_bias_rel": float((v_pi - a0) / max(1e-30, abs(a0))),
            "A0p_vs_VpiL2": float(a0p - v_pi),
            "mono_holds_under_true_measure": bool(a0p <= v_pi + 1e-15)}


# ==================================================================== 判词
def judge(rows, gates) -> dict:
    out: dict = {"n_cells": len(rows),
                 "n_capped": sum(1 for r in rows if r["capped"]),
                 "n_unmeasured_survival": sum(1 for r in rows
                                              if not math.isfinite(r["survival"]))}
    out["gates"] = {k: {kk: vv for kk, vv in v.items()
                        if kk in ("pass", "max_abs", "max_rel", "max_rel_spread",
                                  "n_violations", "n_neg_ack", "n_changed",
                                  "n_changed_live", "tol", "worst_excess")}
                    for k, v in gates.items()}
    out["gates_failed"] = sorted(k for k, v in gates.items() if v.get("pass") is False)
    out["gates_na"] = sorted(k for k, v in gates.items() if v.get("pass") is None)

    # ---- 结构：`b=8`（1 发送）上 `A0p` 是否仍等于 `A0`
    b8 = [r for r in rows if r["b_ticks"] == 8]
    out["P_STRUCT_b8"] = ({
        "max_abs_A0p_minus_A0": float(max(abs(r["A0p"] - r["A0"]) for r in b8)),
        "all_equal": bool(all(abs(r["A0p"] - r["A0"]) <= TOL_STRUCT for r in b8)),
        "n": len(b8)} if b8 else "NOT-APPLICABLE-NO-B8")

    # ---- 探测价值（L2 口径）符号谱
    signs = {}
    for tau in sorted({r["tau"] for r in rows}):
        for b in sorted({r["b_ticks"] for r in rows}):
            sub = [r for r in rows if r["tau"] == tau and r["b_ticks"] == b]
            if not sub:
                continue
            signs[f"tau{tau}_b{b}"] = {
                "n": len(sub),
                "pos": sum(1 for r in sub if r["IV_probe_L2"] > IV_NOISE),
                "neg": sum(1 for r in sub if r["IV_probe_L2"] < -IV_NOISE),
                "IV_probe_L2": [r["IV_probe_L2"] for r in sub]}
    out["sign_profile_IV_probe_L2"] = signs
    any_pos = any(v["pos"] > 0 for v in signs.values())
    any_neg = any(v["neg"] > 0 for v in signs.values())
    out["P_PROBE_L2"] = ("PROBE-WORTHLESS-GRID-L2" if not any_pos
                         else ("MIXED-SIGN" if any_neg else "PROBE-HAS-VALUE-GRID-L2"))

    # ---- 主判：最大 b、最大 tau 的 survival
    b_max = max((r["b_ticks"] for r in rows), default=None)
    t_max = max((r["tau"] for r in rows), default=None)
    main = sorted([r for r in rows if r["b_ticks"] == b_max and r["tau"] == t_max],
                  key=lambda r: r["n_w"])
    four = [r for r in main if math.isfinite(r["survival"])]
    spread = ((max(r["survival"] for r in four) - min(r["survival"] for r in four))
              if len(four) >= 2 else float("nan"))
    out["P_MAIN"] = {
        "cell": {"b_ticks": b_max, "tau": t_max, "n_users": (main[0]["n_users"] if main else None)},
        "n": len(main), "n_measured": len(four),
        "survival": [r["survival"] for r in main],
        "IV_probe_L2": [r["IV_probe_L2"] for r in main],
        "IV_probe_L1": [r["IV_probe_L1"] for r in main],
        "IV_ack": [r["IV_ack"] for r in main],
        "spread": float(spread),
        "survival_finest": (four[-1]["survival"] if four else float("nan")),
        "n_w_finest": (four[-1]["n_w"] if four else None)}
    if not four:
        out["VERDICT_MAIN"] = "UNMEASURED-DENOMINATOR-ZERO"
    elif out["P_MAIN"]["survival_finest"] > 1.0 + 1e-9:
        out["VERDICT_MAIN"] = "WIRING-SUSPECT"
    elif spread == spread and spread > SURVIVAL_SPREAD_MAX:
        out["VERDICT_MAIN"] = "GRID-UNRESOLVED"
    else:
        v = out["P_MAIN"]["survival_finest"]
        out["VERDICT_MAIN"] = ("ACK-EATS-PROBE" if v <= 0.10 else
                               ("ACK-DOES-NOT-EAT-PROBE" if v >= 0.90
                                else "ACK-PARTIALLY-EATS-PROBE"))

    # ---- 总判词
    if out["gates_failed"]:
        out["VERDICT"] = "NO-VERDICT-GATE-FAIL"
    elif out["n_capped"] > 0:
        out["VERDICT"] = "SCALE-WALL"
    else:
        out["VERDICT"] = out["VERDICT_MAIN"]
    out["verdict_scope"] = (
        "只覆盖网格信念（`CW-POMDP-v1.5-grid`）、ρ=0.999、P0=0.01、κ=2、p=0、"
        f"probe_max={PROBE_MAX_MAIN}、HALF={HALF}；`P` 未离散化；"
        "不许跨 ρ（R4–R5 的 0.85）或跨表示（ADF vs 网格）相减/相除")
    return out


# ==================================================================== main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="CW-R5.6 网格信念（直方图）L1 DP")
    ap.add_argument("--size-only", action="store_true")
    ap.add_argument("--ctrl-only", action="store_true")
    ap.add_argument("--nu", type=int, default=1)
    ap.add_argument("--nw", default="")
    ap.add_argument("--tau", default="")
    ap.add_argument("--b", default="")
    ap.add_argument("--pmax", default=str(PROBE_MAX_MAIN),
                    help="探测次数上限；留空字符串表示不限")
    ap.add_argument("--ptotal", default="",
                    help="全局探测次数上限；留空表示不限（n=2 压状态用）")
    ap.add_argument("--diag-bias", action="store_true",
                    help="跑后诊断：L2 臂报出的数 vs 它自己策略在真实测度下的成本")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    n_w_list = tuple(int(x) for x in args.nw.split(",")) if args.nw else N_W_LIST
    tau_list = tuple(int(x) for x in args.tau.split(",")) if args.tau else TAU_LIST
    b_list = tuple(int(x) for x in args.b.split(",")) if args.b else B_LIST
    pmax = None if args.pmax == "" else int(args.pmax)
    ptotal = None if args.ptotal == "" else int(args.ptotal)
    nu = int(args.nu)

    tag = args.tag or ("n%d_pmax%s" % (nu, "inf" if pmax is None else pmax))
    _mode = ("size" if args.size_only else ("ctrl" if args.ctrl_only else
             ("diag_bias" if args.diag_bias else tag)))
    print(f"[{ROUND}] {PROTOCOL} nu={nu} rho={RHO} tau={tau_list} b={b_list} p=0 "
          f"n_w={n_w_list} pmax={pmax} ptotal={ptotal} HALF={HALF}", flush=True)

    dest = Path(args.out) if args.out else (
        RESULTS / f"cw_r5_6_grid_l1_{_mode}_{_now_utc()}_{_src_hash()[:16]}")
    dest.mkdir(parents=True, exist_ok=True)
    t_all = time.perf_counter()

    if args.ctrl_only:
        gates = {"g_ps_ref": g_ps_ref(n_w_list), "g_mart": g_mart(n_w_list),
                 "g_noinfo": g_noinfo(min(n_w_list)), "g_budget": g_budget(min(n_w_list)),
                 "g_ast": g_ast(), "g_fp": g_fp()}
        print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "rows"}
                          for k, v in gates.items()}, indent=2, ensure_ascii=False)[:8000],
              flush=True)
        nf = sum(1 for v in gates.values() if v.get("pass") is False)
        print(f"[ctrl-only] n_fail={nf} / {len(gates)}", flush=True)
        (dest / "r56_ctrl.json").write_text(
            json.dumps({"round": ROUND, "protocol": PROTOCOL, "utc": _now_utc(),
                        "gates": gates}, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[evidence] {dest}", flush=True)
        return 0

    if args.diag_bias:
        rows = []
        for nu_i in ([nu] if nu else [1, 2]):
            for tau in tau_list:
                for b in b_list:
                    d = diag_l2_true(min(n_w_list), tau, b, n_users=nu_i)
                    rows.append(d)
                    print(f"  [diag nu={nu_i} tau={tau} b={b}] "
                          f"A0={d['A0_as_reported']:.10e} "
                          f"Vpi={d['V_piL2_true_measure']:.10e} "
                          f"A0p={d['A0p']:.10e} "
                          f"bias_rel={d['l2_self_bias_rel']:+.4f} "
                          f"IVack={d['A0_as_reported'] - d['A0p']:+.3e} "
                          f"IVack_corr={d['V_piL2_true_measure'] - d['A0p']:+.3e} "
                          f"Vpi==A0p:{d['V_piL2_true_measure'] == d['A0p']}", flush=True)
        n_bias = sum(1 for d in rows if d["l2_self_bias"] > 0)
        outd = {"round": ROUND, "protocol": PROTOCOL, "utc": _now_utc(),
                "post_hoc": True,
                "note": ("本模式在**看到主面板读数之后**写；不是预注册项。"
                         "用途：判 `A0` 报出的数是否等于它自己策略在真实测度下的成本。"),
                "n_w": int(min(n_w_list)), "rows": rows,
                "n_positive_self_bias": int(n_bias), "n_rows": len(rows)}
        (dest / "r56_bias.json").write_text(
            json.dumps(outd, indent=2, ensure_ascii=False, default=float),
            encoding="utf-8")
        print(f"[diag] n_positive_self_bias={n_bias}/{len(rows)}", flush=True)
        print(f"[evidence] {dest}", flush=True)
        return 0

    if args.size_only:
        # 逐 `n_w` 升序：任一腿撞顶就停（`n_w` 单调）
        out = []
        for n_w in (n_w_list if args.nw else N_W_SIZE):
            cfg = ack_config(RHO, SIGMA_W1 if nu == 1 else SIGMA_W1_2U)
            gw = make_gw(cfg, n_w)
            t0 = time.perf_counter()
            qd = Quad(cfg, n_w, gw, probe_max=pmax, probe_total=ptotal)
            for tau in tau_list:
                for b in b_list:
                    qd.cell(tau, b)
            st = qd.states()
            rec = {"n_w": int(n_w), "n_users": nu, "probe_max": pmax,
                   "probe_total": ptotal,
                   "sec": round(time.perf_counter() - t0, 3),
                   "states": {k: st[k] for k in ("A0", "A0p", "VL2", "VL1")},
                   "n_beliefs": st["n_beliefs"], "capped": st["capped"]}
            out.append(rec)
            print(f"  [size n_w={n_w} nu={nu} pmax={pmax}] {rec['sec']} s "
                  f"states={rec['states']} beliefs={rec['n_beliefs']} "
                  f"capped={rec['capped']}", flush=True)
            if rec["capped"]:
                print(f"  [size] n_w={n_w} 撞顶 ⇒ 停止", flush=True)
                break
        (dest / "r56_size.json").write_text(
            json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[evidence] {dest}", flush=True)
        return 0

    gates = {"g_ps_ref": g_ps_ref(n_w_list), "g_mart": g_mart(n_w_list),
             "g_noinfo": g_noinfo(min(n_w_list)), "g_budget": g_budget(min(n_w_list)),
             "g_ast": g_ast(), "g_fp": g_fp()}
    print(f"[ctrl] {json.dumps({k: v.get('pass') for k, v in gates.items()})}", flush=True)

    t0 = time.perf_counter()
    pnl = panel(n_w_list, tau_list, b_list, n_users=nu, probe_max=pmax,
                probe_total=ptotal, tag=tag)
    sec_panel = round(time.perf_counter() - t0, 3)

    gates["g_mono"] = g_mono(pnl["rows"])
    gates["g_nw3"] = g_nw3(pnl["rows"])
    gates["g_size"] = g_size(pnl["legs"])
    if args.size_only is False and len(pnl["rows"]) <= 24:
        gates["g_dropobs"] = g_dropobs(min(n_w_list))
    v = judge(pnl["rows"], gates)

    print(json.dumps({k: v[k] for k in ("VERDICT", "VERDICT_MAIN", "P_PROBE_L2",
                                        "P_MAIN", "P_STRUCT_b8")},
                     indent=2, ensure_ascii=False, default=float)[:6000], flush=True)
    print(json.dumps({"gates_failed": v["gates_failed"], "gates_na": v["gates_na"],
                      "verdict": v["VERDICT"]}, ensure_ascii=False), flush=True)

    out = {"round": ROUND, "protocol": PROTOCOL, "utc": _now_utc(),
           "ticks_per_quantum": TICKS_PER_QUANTUM, "send_ticks": SEND_TICKS,
           "half": HALF, "probe_max": pmax, "probe_total": ptotal, "n_users": nu,
           "sec_panel": sec_panel, "sec_total": round(time.perf_counter() - t_all, 3),
           "panel_spec": {"rho": RHO, "P": P0, "tau": list(tau_list),
                          "b_ticks": list(b_list), "n_w": list(n_w_list),
                          "probe_ticks": PROBE_TICKS_MAIN,
                          "sigma_w": list(ack_config(RHO, SIGMA_W1 if nu == 1
                                                     else SIGMA_W1_2U).plant.sigma_w)},
           "cells": pnl["rows"], "legs": pnl["legs"],
           "gates": gates, "verdicts": v}
    (dest / "r56.json").write_text(json.dumps(out, indent=2, ensure_ascii=False,
                                              default=float), encoding="utf-8")
    print(f"[evidence] {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
