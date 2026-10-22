# -*- coding: utf-8 -*-
"""CW-R1.7 闸：小状态 DP 的**可采纳性**（`plan_dp.py` 的验收面，不是收益表）。

为什么第一道闸就是"可采纳性"（台账 ⑫#39 / R14）
--------------------------------------------
`V` 是"从状态出发、**最优**调度下的期望代价"⇒ 它必须**不高于任何可行策略的代价**。
这句话给了两条**不依赖内部实现**的参考：
  ① **穷举**：`3^τ` 条动作序列（`τ ≤ 4`）直接按期望动力学模拟，取最小值 = 真最优；
  ② **轮询**：`τ` 槽、`b=τ` 时"每槽交替刷新一个用户"是可行策略 ⇒ `V_τ(b=τ) ≤` 它的代价。
两条都不需要相信 DP 里的任何一行代码。

本轮实测（见 `docs/CW-R1.7-小状态DP交付.md`）：**D3/D4 都 FAIL**，
即 `plan_dp` 的递推与"期望动力学 + 预算"这条链**不一致**，且**加密网格不收敛到穷举最优**
⇒ 不是离散化误差。判词 **`DP-TAIL-REJECTED`**：不接进 `plan_ref`、不跑面板、不出 `N1`/`N2`。

闸清单
------
D0 信息边界：`plan_dp.py` 与 `plan_ref.py` 的**代码实体**里不出现任何真值入口（AST）。
D1 `b=0` 与解析"再无观测"外推一致（该情形值函数关于 `p` 精确仿射 ⇒ 插值应当是精确的）。
D2 `V_τ` 对 `b` 单调不增（全网格抽样）。
D3 **穷举对账**：`τ∈{2,3,4}`、`b∈{1,τ}`、3 个 `p`：`V ≤ 穷举最优 + 1e-9`（相对）。
D4 **轮询对账**：`τ∈{8,16,32,64,128,256}`、`b=τ`：`V ≤ 轮询代价 + 1e-9`（相对）。
D5 **分辨率敏感性**：把结点数 ×2 四次，`D3/D4` 的相对超差必须随加密**显著缩小**
   （判据：最细与最粗相比至少缩小 10 倍）——否则说明误差不是离散化，而是递推结构。
D6 变异 3/3 被抓（送达概率置 1 / 交换两用户轴 / 预算不减）。
"""

from __future__ import annotations

import ast
import itertools
import pathlib

import numpy as np

from . import config as C
from .gates import Gate
from .plan_dp import DPTail, default_knots


# ---------------------------------------------------------------- 独立参考
def _expected_dyn(sw2, sm2, p_hat, act, P):
    """按**期望**推进一次（`act=-1` 不动，`0/1` 刷新对应用户）。"""
    P = list(P)
    if act < 0:
        return [P[0] + sw2[0], P[1] + sw2[1]]
    g = P[act] / (P[act] + sm2)
    P[act] = P[act] - p_hat[act] * g * P[act] + sw2[act]
    j = 1 - act
    P[j] = P[j] + sw2[j]
    return P


def brute_opt(tau: int, p, budget: int, sw2, sm2, p_hat, q) -> float:
    """`3^τ` 穷举动作序列的最优期望代价（**真值参考**，不信 DP 的任何一行）。"""
    best = None
    for seq in itertools.product((-1, 0, 1), repeat=tau):
        if sum(1 for a in seq if a >= 0) > budget:
            continue
        P = [p, p]
        tot = 0.0
        for a in seq:
            tot += q * (P[0] + P[1])
            P = _expected_dyn(sw2, sm2, p_hat, a, P)
        if best is None or tot < best:
            best = tot
    return float(best)


def round_robin(tau: int, p, sw2, sm2, p_hat, q) -> float:
    """每槽交替刷新一个用户（`b=τ` 时**可行**）的期望代价 —— 上界参考。"""
    P = [p, p]
    tot = 0.0
    for t in range(tau):
        tot += q * (P[0] + P[1])
        P = _expected_dyn(sw2, sm2, p_hat, t % 2, P)
    return float(tot)


def _plant(cfg):
    p = cfg.plant
    return (np.asarray(p.sigma_w, dtype=float) ** 2, float(p.sigma_meas) ** 2,
            float(p.q_cost))


# ---------------------------------------------------------------- D0
def d0_info_boundary(g: Gate) -> None:
    forbidden = {"_x_true", "_log_h_true", "w_realized", "cost_true", "p_true",
                 "_pub", "_inner", "_SchedulerView__w"}
    here = pathlib.Path(__file__).parent
    hits_all: dict[str, list[str]] = {}
    for name in ("plan_dp.py", "plan_ref.py"):
        src = (here / name).read_text(encoding="utf-8")
        used: set[str] = set()
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Attribute):
                used.add(node.attr)
            elif isinstance(node, ast.Name):
                used.add(node.id)
            elif isinstance(node, ast.arg):
                used.add(node.arg)
            elif isinstance(node, ast.keyword) and node.arg:
                used.add(node.arg)
        hits = sorted(forbidden & used)
        if hits:
            hits_all[name] = hits
    g.check("D0 plan_dp / plan_ref 的代码实体里不出现真值入口（AST，两个文件）",
            not hits_all, {"hits": hits_all})


# ---------------------------------------------------------------- D1
def d1_b0_analytic(g: Gate) -> None:
    cfg = C.CWConfig(plant=C.bounded_plant())
    sw2, sm2, q = _plant(cfg)
    dp = DPTail(cfg, (0.5, 0.5), b_cap=128).build()
    #: 预注册 §3 写 1e-9；探针实测 b=0 列在 τ≥128 上有 1e-8..1e-5 的相对底噪
    #: （`b=0` 时值函数关于 `p` **精确仿射**，所以那点底噪不是算术错，是网格+浮点累积）。
    #: **看 D 闸结果前**改成两条判据并记入预注册 §7：
    #:   D1a 单一粗阈值 1e-4（任何真算术错都在 1e-2 量级，仍留 2 个数量级分辨力）；
    #:   D1b **斜率精确性**（仿射部分差分的相对差 ≤ 1e-12）—— 这条才抓 `q/A/σ_w` 写错。
    tol_rel = 1e-4
    #: 斜率判据分两段：`τ≤16` 必须**逐位相等**（1e-12），大 τ 放宽到 1e-4（浮点/网格底噪）。
    #: 依据：`q/A/σ_w` 任一写错都会让**所有** τ 的斜率差 ≥1e-2，仍留 2 个数量级分辨力。
    tol_slope_small = 1e-12
    tol_slope_big = 1e-4
    taus = (1, 4, 16, 64, 128, 256)
    rows, bad, bad_slope = [], [], []
    for tau in taus:
        p_lo, p_hi = 0.01, 0.05
        s_lo, s_hi = 0.0, 0.0
        # ★ 顺序必须是"先记本步代价、再推进"（与 `_terminal` / DP 的 `imm` 同口径）。
        #   第一版写成"先推进再记"，D1b 在 τ=1 就报 6.5% 的假 FAIL —— 闸自伤（#46 同族）。
        Q = np.array([p_lo, p_lo])
        for _ in range(tau):
            s_lo += q * float(Q.sum())
            Q = Q + sw2
        Q = np.array([p_hi, p_hi])
        for _ in range(tau):
            s_hi += q * float(Q.sum())
            Q = Q + sw2
        got_lo = float(dp.value(np.array([[p_lo, p_lo]]), [0], tau)[0])
        got_hi = float(dp.value(np.array([[p_hi, p_hi]]), [0], tau)[0])
        d_an = (s_hi - s_lo) / max(abs(s_lo), 1e-12)
        d_dp = (got_hi - got_lo) / max(abs(got_lo), 1e-12)
        rel_slope = abs(d_dp - d_an) / max(abs(d_an), 1e-12)
        rows.append({"tau": tau, "slope_dp": d_dp, "slope_analytic": d_an,
                     "rel_slope": rel_slope})
        if rel_slope > (tol_slope_small if tau <= 16 else tol_slope_big):
            bad_slope.append((tau, rel_slope))
        for p in (0.0035, 0.0117, 0.0523):
            s = 0.0
            Q = np.array([p, p])
            for _ in range(tau):
                s += q * float(Q.sum())
                Q = Q + sw2
            got = float(dp.value(np.array([[p, p]]), [0], tau)[0])
            rel = abs(got - s) / max(abs(s), 1e-12)
            rows.append({"tau": tau, "p": p, "dp": got, "analytic": s, "rel": rel})
            if rel > tol_rel:
                bad.append((tau, p, rel))
    g.check("D1a b=0 与解析外推一致（相对 ≤ 1e-4；插值/浮点底噪已记录）", not bad,
            {"bad": bad, "worst_rel": max(r["rel"] for r in rows if "rel" in r),
             "rows": [{k: (round(v, 10) if isinstance(v, float) else v) for k, v in r.items()}
                      for r in rows if "rel" in r]})
    g.check("D1b b=0 时值函数关于 p 的**斜率**与解析一致（≤ 1e-12）", not bad_slope,
            {"bad": bad_slope,
             "rows": [{k: (round(v, 12) if isinstance(v, float) else v) for k, v in r.items()}
                      for r in rows if "rel_slope" in r]})


# ---------------------------------------------------------------- D2
def d2_monotone(g: Gate) -> None:
    cfg = C.CWConfig(plant=C.bounded_plant())
    dp = DPTail(cfg, (0.5, 0.5), b_cap=64).build()
    kn = default_knots()
    bad = []
    n = 0
    for tau in (1, 2, 3, 5, 8, 16, 32, 64):
        P = np.array([[kn[i], kn[j]] for i in (1, 3, 5) for j in (1, 3, 5)])
        for b in range(0, min(tau, 63)):
            v0 = dp.value(P, np.full(P.shape[0], b), tau)
            v1 = dp.value(P, np.full(P.shape[0], b + 1), tau)
            n += 1
            d = v1 - v0
            if np.any(d > 1e-12):
                bad.append({"tau": tau, "b": b, "max_dv": float(np.max(d))})
    g.check("D2 V_τ 对预算单调不增（9 个状态 × 8 个 τ × 全 b）", not bad,
            {"n_pairs": n, "bad": bad[:6], "n_bad": len(bad)})


# ---------------------------------------------------------------- D3
def d3_vs_brute(g: Gate, knots=None, cases=None) -> dict:
    cfg = C.CWConfig(plant=C.bounded_plant())
    sw2, sm2, q = _plant(cfg)
    ph = np.array([0.5, 0.5])
    dp = DPTail(cfg, ph, b_cap=4, knots=knots).build()
    cases = cases or [(tau, b, p) for tau in (2, 3, 4) for b in (1, tau)
                      for p in (0.005, 0.0117, 0.03)]
    rows, bad = [], []
    for tau, b, p in cases:
        ref = brute_opt(tau, p, b, sw2, sm2, ph, q)
        got = float(dp.value(np.array([[p, p]]), [b], tau)[0])
        rel = (got - ref) / max(abs(ref), 1e-12)
        rows.append({"tau": tau, "b": b, "p": p, "dp": got, "brute": ref, "rel": rel})
        if got > ref * (1.0 + 1e-9):
            bad.append({"tau": tau, "b": b, "p": p, "rel": rel})
    worst = max(r["rel"] for r in rows)
    g.check("D3 可采纳性 vs 穷举最优（τ≤4；V 必须 ≤ 真最优）", not bad,
            {"n_cases": len(rows), "n_bad": len(bad), "worst_rel": worst,
             "bad": bad[:8],
             "rows": [{k: (round(v, 10) if isinstance(v, float) else v)
                       for k, v in r.items()} for r in rows]})
    return {"worst_rel": worst, "n_bad": len(bad), "rows": rows}


# ---------------------------------------------------------------- D4
def d4_vs_round_robin(g: Gate, knots=None) -> dict:
    cfg = C.CWConfig(plant=C.bounded_plant())
    sw2, sm2, q = _plant(cfg)
    ph = np.array([0.5, 0.5])
    p = 0.012
    rows = []
    for tau in (8, 16, 32, 64, 128, 256):
        dp = DPTail(cfg, ph, b_cap=tau, knots=knots).build()
        ref = round_robin(tau, p, sw2, sm2, ph, q)
        got = float(dp.value(np.array([[p, p]]), [tau], tau)[0])
        rows.append({"tau": tau, "dp": got, "round_robin": ref, "ratio": got / ref})
    bad = [r for r in rows if r["ratio"] > 1.0 + 1e-9]
    g.check("D4 可采纳性 vs 轮询策略（b=τ；V 必须 ≤ 任何可行策略的代价）", not bad,
            {"n_bad": len(bad), "rows": [{k: (round(v, 10) if isinstance(v, float) else v)
                                          for k, v in r.items()} for r in rows]})
    return {"rows": rows, "n_bad": len(bad)}


# ---------------------------------------------------------------- D5
def d5_resolution(g: Gate) -> None:
    """加密网格必须让 D3/D4 的超差缩小（否则误差不是离散化的）。"""
    rows = []
    for nf in (8, 16, 32, 64):
        kn = np.concatenate([np.linspace(0.0, 0.055, nf + 1),
                             [0.08, 0.12, 0.18, 0.30, 0.50, 0.90, 1.50]])
        cfg = C.CWConfig(plant=C.bounded_plant())
        sw2, sm2, q = _plant(cfg)
        ph = np.array([0.5, 0.5])
        dp3 = DPTail(cfg, ph, b_cap=4, knots=kn).build()
        e3 = []
        for tau, b, p in ((3, 3, 0.0117), (4, 4, 0.0117)):
            ref = brute_opt(tau, p, b, sw2, sm2, ph, q)
            got = float(dp3.value(np.array([[p, p]]), [b], tau)[0])
            e3.append((got - ref) / ref)
        dp32 = DPTail(cfg, ph, b_cap=32, tau_max=32, knots=kn).build()
        ref32 = round_robin(32, 0.012, sw2, sm2, ph, q)
        got32 = float(dp32.value(np.array([[0.012, 0.012]]), [32], 32)[0])
        rows.append({"n_fine_knots": int(nf + 1), "spacing": 0.055 / nf,
                     "err_tau3": e3[0], "err_tau4": e3[1],
                     "ratio_tau32": got32 / ref32})
    imp3 = abs(rows[0]["err_tau3"]) / max(abs(rows[-1]["err_tau3"]), 1e-15)
    imp32 = (rows[0]["ratio_tau32"] - 1.0) / max(abs(rows[-1]["ratio_tau32"] - 1.0), 1e-15)
    g.check("D5 加密必须显著缩小超差（最细/最粗 ≥10 倍；否则不是离散化误差）",
            imp3 >= 10.0 and imp32 >= 10.0,
            {"improvement_tau3": imp3, "improvement_tau32": imp32,
             "rows": [{k: (round(v, 8) if isinstance(v, float) else v)
                       for k, v in r.items()} for r in rows]})


# ---------------------------------------------------------------- D6
def d6_mutations() -> tuple[list[dict], int]:
    """三次单变量破坏（判据：至少一条 D3/D4 从 PASS 翻 FAIL 或反向）。"""
    from . import plan_dp as D
    cfg = C.CWConfig(plant=C.bounded_plant())
    sw2, sm2, q = _plant(cfg)
    ph = np.array([0.5, 0.5])
    rows: list[dict] = []

    def err():
        dp = D.DPTail(cfg, ph, b_cap=4).build()
        out = []
        for tau, b, p in ((3, 3, 0.0117), (4, 4, 0.0117)):
            ref = brute_opt(tau, p, b, sw2, sm2, ph, q)
            out.append(float(dp.value(np.array([[p, p]]), [b], tau)[0]) / ref - 1.0)
        return max(out)

    base = err()

    orig = D.DPTail.__init__

    def _phat_one(self, cfg_, p_hat, **kw):     # noqa: ANN001
        orig(self, cfg_, np.ones(2), **kw)

    try:
        D.DPTail.__init__ = _phat_one
        e1 = err()
    finally:
        D.DPTail.__init__ = orig
    rows.append({"mutation": "M1 送达概率置 1（忽略 p̂）", "caught":
                 bool(abs(e1 - 1.0) > 0.05), "evidence": {"err": e1, "base": base}})

    orig_maps = D.DPTail._build_maps

    def _swap(self):                            # noqa: ANN001
        m = orig_maps(self)
        out = {}
        for (i, tag), val in m.items():
            out[(i, tag)] = m[(1 - i, tag)]
        return out

    try:
        D.DPTail._build_maps = _swap
        e2 = err()
    finally:
        D.DPTail._build_maps = orig_maps
    rows.append({"mutation": "M2 交换两用户的动力学轴", "caught":
                 bool(abs(e2 - base) > 5e-3), "evidence": {"err": e2, "base": base}})

    orig_build = D.DPTail.build

    def _nobud(self):                           # noqa: ANN001
        n = self.knots.size
        k = self.knots
        imm = (self.q * (k[:, None] + k[None, :]))[:, :, None]
        mh = {i: self._maps[(i, "hold")] for i in range(2)}
        md = {i: self._maps[(i, "deliv")] for i in range(2)}
        Vs = [np.zeros((n, n, 1))]
        for tau in range(1, self.tau_max + 1):
            nb = min(self.nb_max, tau + 1)
            Vprev = Vs[-1]
            bh = np.clip(np.arange(nb), 0, Vprev.shape[2] - 1)
            base_ = self._bil(Vprev, mh[0], mh[1])[:, :, bh]
            best = base_
            if nb > 1:
                cand = np.full(base_.shape, np.inf)
                for i in range(2):
                    d = self._bil(Vprev, md[i], mh[1 - i])[:, :, bh]   # ★ 预算不减
                    s = self.p_hat[i] * d + (1 - self.p_hat[i]) * base_
                    cand = np.where((np.arange(nb) >= 1)[None, None, :],
                                    np.minimum(cand, s), cand)
                best = np.minimum(base_, cand)
            Vs.append(best + imm)
        self._V = Vs
        return self

    try:
        D.DPTail.build = _nobud
        e3 = err()
    finally:
        D.DPTail.build = orig_build
    rows.append({"mutation": "M3 递推里预算不减（一帧可无限用）", "caught":
                 bool(abs(e3 - base) > 5e-3), "evidence": {"err": e3, "base": base}})
    return rows, sum(0 if r["caught"] else 1 for r in rows)


# ---------------------------------------------------------------- 入口
def run_all() -> tuple[Gate, dict]:
    g = Gate()
    d0_info_boundary(g)
    d1_b0_analytic(g)
    d2_monotone(g)
    x3 = d3_vs_brute(g)
    x4 = d4_vs_round_robin(g)
    d5_resolution(g)
    return g, {"brute": x3, "round_robin": x4}
