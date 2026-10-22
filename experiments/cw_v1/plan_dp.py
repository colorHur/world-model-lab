# -*- coding: utf-8 -*-
"""CW-R1.7 尾部值函数：`(P_0, P_1, budget_left)` 上的**小状态后向 DP**。

本条线在 R1.5/R1.6/R1.6.1 里的尾部是**解析估计**（均匀配速 + 稳态循环均值闭式）。
R1.6.1 实测它把"多花一帧"的价格估成 **0.0225**，而一帧的收益上界只有 **0.006**（3.8×）
⇒ 规划器推迟花帧（`send_profile` 前松后紧、`G8a` FAIL、`G4` 崩）。本轮把尾部换成
**精确后向 DP**，让"这一帧值不值"由同一个 Bellman 递推给出，而不是由闭式拟合给出。

递推（预注册 `docs/research/CW-R1.7-小状态DP-预注册-v0.1.md` §2 逐字对应）
---------------------------------------------------------------------
    V_0(p0,p1,b) = 0
    V_τ(p0,p1,b) = q·(p0+p1) + min{
        HOLD        : V_{τ−1}(f0(p0), f1(p1), b)
        SEND i (b≥1): p̂_i·V_{τ−1}(d_i(p_i), f_j(p_j), b−1)
                    + (1−p̂_i)·V_{τ−1}(f_i(p_i), f_j(p_j), b−1)
    }
    f_i(p) = A_i²p + σ_w,i²                               （不刷新）
    d_i(p) = A_i²(1−g_i)p + σ_w,i²,  g_i = p/(p+σ_meas²)  （送达 + 量测更新）

`V` 只含**任务侧** `q·Σ P`；`q·x̂²` 与 `r·u²` 仍由 `plan_ref._terminal` 按名义 `α=A−BK` 解析外推
（同一分解，理由见 `plan_ref._terminal` 的 docstring）。

四个**明写的近似**（预注册 §2，任一被推翻则本轮结论作废）
------------------------------------------------------
1. **信道信念冻结**：`p̂_i` 取**槽初**值，整条尾部不变 ⇒ DP **不内生化 PROBE**
   （探测价值只能经外层 rollout 的 H 步条件信念采样体现）。这是本轮最硬的近似。
2. **DP 里没有 `PROBE` 动作**：它只花一帧、不动任务估计 ⇒ 在 DP 内恒被 `HOLD` 支配。
   外层 rollout 照样保留 `PROBE`；它是否被选是**实测量**。
3. **`b ≥ τ` 饱和**：动作空间是每槽**至多一个**资源动作（`PROBE i / SEND i / HOLD`），
   所以 `b ≥ τ` 时多出的预算用不掉 ⇒ `V_τ(·,b) = V_τ(·,τ)`。
   （★ 预注册 §2 原写 `2τ`，**跑数前**已改正为 `τ`，记在预注册 §7。）
4. **分段线性插值 + 出界夹紧**：`p` 轴用分段非均匀结点（细段覆盖 0–0.055 的实际工作区，
   粗段到 1.5 以覆盖 `t=0` 的 `P=prior_var=1.0`）。出界点计数到 `n_clip`，**不静默**。

★ R1.7.1 修复的**轴错配**（`axis_mode`，2026-10-23）
----------------------------------------------------
原 `build()` 里 SEND 分支写的是 `_bil(Vprev, m_del[i], m_hold[1-i])`。`_bil` 的第 1/2 个参数
**恒作用于第 0/1 条 P 轴**（= 用户 0 / 用户 1），所以 `i=0` 时恰好正确，
而 `i=1` 时把**用户 1 的送达映射作用到了用户 0 的 P 上**、用户 0 的 hold 作用到用户 1 上。

后果（实测）：`τ≤2` 无差（`V_1` 与 `P` 无关）；`τ≥3` 起**每层同号累加**，对角点 +0.44%…+4.8%，
**非对角点 +15%…+94%**（`(0.002, 0.05)`、`τ=8`），且**加密不收敛**（结构性错，不是离散化）。

`axis_mode="user_axis"`（默认）为修复版；`"legacy_swap"` **逐位复现**旧行为，
只作闸的**活体阴性对照**（本轮 `dp_audit.A3/A4` 依赖它）。

缓存
----
`V` 只依赖 `(plant, p̂, b_cap, 网格)`，**不依赖 episode** ⇒ 跨 episode 复用。
`p̂` 量化到 0.02 再作 key（预注册 §2 明写"量化会改模型，故只作缓存粒度"）。
LRU 上限 `_CACHE_MAX`，避免把 50–70 MB 的数组无限留在内存里。
"""

from __future__ import annotations

import numpy as np

from .config import CWConfig

#: 细段（实际工作区）+ 粗段（覆盖初始先验 `P=1.0` 与冷启动）
FINE = (0.0, 0.055, 8)
COARSE = (0.08, 0.12, 0.18, 0.30, 0.50, 0.90, 1.50)

#: `p̂` 的缓存粒度（只影响缓存命中，不参与任何判据）
PHAT_QUANT = 0.02

_CACHE_MAX = 3
_CACHE: "dict[tuple, DPTail]" = {}
_ORDER: "list[tuple]" = []


def default_knots() -> np.ndarray:
    lo, hi, n = FINE
    return np.concatenate([np.linspace(lo, hi, n), np.asarray(COARSE, dtype=float)])


def _loc(knots: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """分段线性插值定位：返回左端下标与权重（`p` 已夹到结点范围内）。"""
    idx = np.clip(np.searchsorted(knots, p, side="right") - 1, 0, knots.size - 2)
    w = (p - knots[idx]) / (knots[idx + 1] - knots[idx])
    return idx, np.clip(w, 0.0, 1.0)


def _plant_key(cfg: CWConfig) -> tuple:
    p = cfg.plant
    return (tuple(float(a) for a in p.A), float(p.B), tuple(float(s) for s in p.sigma_w),
            float(p.sigma_meas), float(p.q_cost), float(p.r_cost), int(cfg.protocol.horizon))


class DPTail:
    """`(p0, p1, b)` 网格上的后向 DP（**有限视界、给定 p̂** 的条件最优，不是全局上界）。"""

    def __init__(self, cfg: CWConfig, p_hat, *, b_cap: int | None = None,
                 tau_max: int | None = None, knots: np.ndarray | None = None,
                 axis_mode: str = "user_axis") -> None:
        if axis_mode not in ("user_axis", "legacy_swap"):
            raise ValueError(f"axis_mode={axis_mode!r} 非法（只允许 user_axis / legacy_swap）")
        self.axis_mode = axis_mode
        p = cfg.plant
        self.A = np.asarray(p.A, dtype=float)
        self.sw2 = np.asarray(p.sigma_w, dtype=float) ** 2
        self.sm2 = float(p.sigma_meas) ** 2
        self.q = float(p.q_cost)
        self.T = int(cfg.protocol.horizon)
        self.tau_max = int(tau_max if tau_max is not None else self.T)
        self.b_cap = int(b_cap if b_cap is not None else self.T)
        self.knots = default_knots() if knots is None else np.asarray(knots, dtype=float)
        ph = np.asarray(p_hat, dtype=float).reshape(-1)
        if ph.size != self.A.size:
            raise ValueError(f"p_hat 长度 {ph.size} != n_users {self.A.size}")
        self.p_hat = np.clip(ph, 1e-9, 1.0 - 1e-9)
        self.nb_max = int(min(self.b_cap, self.tau_max)) + 1
        self._maps = self._build_maps()
        self._V: list[np.ndarray] | None = None
        self.n_clip = 0            # 插值出界计数（诊断；正常应恒 0）
        self.n_lookup = 0

    # ------------------------------------------------------------ 网格映射
    def _build_maps(self) -> dict:
        k = self.knots
        n = k.size
        out: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {}
        for i in range(self.A.size):
            a2 = float(self.A[i] ** 2)
            s2 = float(self.sw2[i])
            g = k / (k + self.sm2)
            for tag, p2 in (("hold", a2 * k + s2),
                            ("deliv", a2 * (1.0 - g) * k + s2)):
                pc = np.clip(p2, k[0], k[-1])
                idx = np.clip(np.searchsorted(k, pc, side="right") - 1, 0, n - 2)
                w = np.clip((pc - k[idx]) / (k[idx + 1] - k[idx]), 0.0, 1.0)
                out[(i, tag)] = (idx, w)
        return out

    @staticmethod
    def _bil(Vprev: np.ndarray, m0: tuple, m1: tuple) -> np.ndarray:
        """对 `Vprev`（形状 `(n0,n1,nb)`）按两条轴上的映射做**双线性**插值。"""
        i0, w0 = m0
        i1, w1 = m1
        r = (1.0 - w0)[:, None, None] * Vprev[i0] + w0[:, None, None] * Vprev[i0 + 1]
        s = (1.0 - w1)[None, :, None] * r[:, i1] + w1[None, :, None] * r[:, i1 + 1]
        return s

    # ------------------------------------------------------------ 后向递推
    def build(self) -> "DPTail":
        if self._V is not None:
            return self
        n = self.knots.size
        k = self.knots
        imm = (self.q * (k[:, None] + k[None, :]))[:, :, None]     # 本步即时代价 q(p0+p1)
        m_hold = {i: self._maps[(i, "hold")] for i in range(self.A.size)}
        m_del = {i: self._maps[(i, "deliv")] for i in range(self.A.size)}
        Vs: list[np.ndarray] = [np.zeros((n, n, 1), dtype=float)]
        for tau in range(1, self.tau_max + 1):
            nb = int(min(self.nb_max, tau + 1))
            Vprev = Vs[-1]
            nbp = Vprev.shape[2]
            bh = np.clip(np.arange(nb), 0, nbp - 1)
            bs = np.clip(np.arange(nb) - 1, 0, nbp - 1)
            base = self._bil(Vprev, m_hold[0], m_hold[1])           # HOLD 的后续
            V0 = base[:, :, bh]
            best = V0
            if nb > 1:
                Vs_ = base[:, :, bs]                                # 送达失败支 = HOLD 映射
                feas = (np.arange(nb) >= 1)[None, None, :]
                cand = np.full(V0.shape, np.inf)
                for i in range(self.A.size):
                    if self.axis_mode == "legacy_swap":
                        # ★ 旧行为（轴错配），逐位复现，只作阴性对照
                        j = 1 - i
                        d = self._bil(Vprev, m_del[i], m_hold[j])[:, :, bs]
                    else:
                        # ★ 按用户轴放置：第 i 个用户用送达映射，其余用 hold
                        maps = [m_hold[0], m_hold[1]]
                        maps[i] = m_del[i]
                        d = self._bil(Vprev, maps[0], maps[1])[:, :, bs]
                    send = self.p_hat[i] * d + (1.0 - self.p_hat[i]) * Vs_
                    cand = np.where(feas, np.minimum(cand, send), cand)
                best = np.minimum(V0, cand)
            Vs.append((best + imm).astype(float))
        self._V = Vs
        return self

    # ------------------------------------------------------------ 查询
    def value(self, P: np.ndarray, bud, tau: int) -> np.ndarray:
        """在任意 `P`（`(M, n)`）与预算 `bud`（`(M,)`）处取 `V_tau`（双线性插值）。"""
        if self._V is None:
            self.build()
        assert self._V is not None
        tau = int(min(max(int(tau), 0), self.tau_max))
        Vt = self._V[tau]
        P = np.atleast_2d(np.asarray(P, dtype=float))
        M = P.shape[0]
        k = self.knots
        lo, hi = float(k[0]), float(k[-1])
        self.n_lookup += M
        out_of = np.sum((P < lo) | (P > hi))
        self.n_clip += int(out_of)
        Pc = np.clip(P, lo, hi)
        nb = Vt.shape[2]
        b = np.clip(np.asarray(bud, dtype=float).reshape(-1).astype(int), 0, nb - 1)
        if b.size == 1 and M > 1:
            b = np.full(M, int(b[0]), dtype=int)
        i0, w0 = _loc(k, Pc[:, 0])
        i1, w1 = _loc(k, Pc[:, 1])
        v00 = Vt[i0, i1, b]
        v01 = Vt[i0, i1 + 1, b]
        v10 = Vt[i0 + 1, i1, b]
        v11 = Vt[i0 + 1, i1 + 1, b]
        return ((1 - w0) * (1 - w1) * v00 + (1 - w0) * w1 * v01
                + w0 * (1 - w1) * v10 + w0 * w1 * v11)

    def price(self, p, tau: int, bud: int) -> float:
        """单帧价格 `V_τ(p, b) − V_τ(p, b−1)`（两用户同 `p`）。"""
        P = np.asarray([[p, p]], dtype=float)
        return float(self.value(P, [bud], tau)[0] - self.value(P, [bud - 1], tau)[0])


def get_dp(cfg: CWConfig, p_hat, *, b_cap: int, axis_mode: str = "user_axis") -> DPTail:
    """带 LRU 的构造：同一 `(plant, p̂ 量化, b_cap, axis_mode)` 只建一次。"""
    ph = np.asarray(p_hat, dtype=float).reshape(-1)
    key = (_plant_key(cfg), tuple(np.round(ph / PHAT_QUANT).astype(int)), int(b_cap), str(axis_mode))
    hit = _CACHE.get(key)
    if hit is not None:
        return hit
    obj = DPTail(cfg, ph, b_cap=b_cap, axis_mode=axis_mode).build()
    _CACHE[key] = obj
    _ORDER.append(key)
    while len(_ORDER) > _CACHE_MAX:
        old = _ORDER.pop(0)
        _CACHE.pop(old, None)
    return obj


def cache_stats() -> dict:
    return {"size": len(_CACHE), "cap": _CACHE_MAX,
            "keys": [{"p_hat_q": k[1], "b_cap": k[2], "axis_mode": k[3]} for k in _ORDER]}
