# -*- coding: utf-8 -*-
"""CW-R1.7.1 闸：`plan_dp` 的**可采纳性 v2** —— 参考物换成"精确闭环 DP"。

为什么 R1.7 的 D3/D4 漏掉了 +94% 的错（本轮教训）
--------------------------------------------------
R1.7 的两条外部参考都是**同一类东西**：「一个具体可行策略」。
- `brute_opt`：穷举**开环**动作序列，在**期望状态**上评估 ⇒ 开环最优（≥ 闭环最优）；
- `round_robin`：一个具体策略（≥ 闭环最优）。
它们只能证明"比某个策略差"，**没有分辨力**去抓"值函数在某个方向被算错"——
本轮实测：`(0.002, 0.05)`、`τ=8` 时 DP 比真最优高 **+94%**，而比 `brute` 只高 **+2.9%**（因为
brute 自己也比真最优高），D3 的阈值 1e-9 虽然会 FAIL，但**量级被参考物自身的水分掩盖**。

本闸引入第三条参考：**精确闭环 DP**（`A=1` 且 `p̂` 冻结下，把 `(P0,P1,b)` 当连续键、**不做任何插值**的
记忆化后向递推）。它与 `DPTail` 的唯一差别就是插值 —— 所以任何"结构性错"都会以
**与网格无关的常数**暴露出来（R1.7 的 D5 加密判据就是因为这一点才 FAIL 的）。

闸清单（**跑数前**写进 `docs/research/CW-R1.7.1-轴错配-预注册-v0.1.md`）
-----------------------------------------------------------------------
A0 信息边界：`plan_dp.py` / `dp_audit.py` 的 AST 里无真值入口。
A1 **参考物序关系**：`exact ≤ brute` 且 `exact ≤ round_robin`（24 例全数）。
   —— 违反则参考物作废，**不进入 A2**（防止"用一个更差的参考去判好"）。
A2 **修复版可采纳**：`dp/exact − 1 ≤ +1e-3`；对角 5 例 + **非对角 6 例**。
   —— 非对角是必须的：轴错配在 `P0=P1` 上只露 +0.44%，在 `P0≠P1` 上露 +15%…+94%。
A3 **阴性对照（活体旧实现）**：`axis_mode="legacy_swap"` 在非对角至少 5 例超差，
   且最大相对超差 ≥ 10%（≥ 100×tol）。
A4 **加密行为**：`nf ∈ {24, 96}` 两档，修复版都 ≤ tol；legacy 两档都 ≥ 100×tol
   （⇒ 误差与网格无关 = 结构性，不是离散化）。
A5 **轴可分辨性**：`sw2=(4e-4,9e-4)` 不对称 ⇒ `V(0.002,0.05) ≠ V(0.05,0.002)`（相对差 ≥ 1e-3）。
   若相等说明两条 P 轴又被当成同一条。
A6 **变异 2/2 被抓**：① 把 `maps` 交换（等价 legacy）；② `p̂` 置 1。
"""

from __future__ import annotations

import ast
import pathlib
from functools import lru_cache

import numpy as np

from . import config as C
from .dp_gates import _plant, brute_opt, round_robin
from .gates import Gate
from .plan_dp import DPTail

#: A2 的相对容差（0.1%）。依据：nf=96 网格的插值残差实测 −6e-4…−1e-5；
#: 而轴错配在**对角**最小 +0.44%（τ=3）、**非对角**最小 +15% ⇒ 仍有 4–150 倍分辨力。
TOL = 1e-3
#: A3/A4 要求 legacy 的超差 ≥ 100×tol
FAR = 100.0 * TOL


# ---------------------------------------------------------------- 精确参考
class ExactClosedLoop:
    """无插值的闭环后向 DP（**本轮唯一可信的"真最优"参考**）。

    `V_t(P0,P1,b) = q(P0+P1) + min{ HOLD, SEND_i }`，后继按**分支**展开（送达/未送达），
    状态键是浮点 `(P0,P1,b)` ⇒ 与 `DPTail` 的递推**同构**，只差插值。
    """

    def __init__(self, cfg: C.CWConfig, p_hat) -> None:
        p = cfg.plant
        self.sw2 = tuple(float(x) ** 2 for x in p.sigma_w)
        self.sm2 = float(p.sigma_meas) ** 2
        self.q = float(p.q_cost)
        ph = np.asarray(p_hat, dtype=float).reshape(-1)
        if ph.size != len(self.sw2):
            raise ValueError(f"p_hat 长度 {ph.size} != n_users {len(self.sw2)}")
        self.p_hat = tuple(float(x) for x in ph)
        self.n_call = 0

    def value(self, tau: int, P0: float, P1: float, b: int) -> float:
        sw2, sm2, q, ph = self.sw2, self.sm2, self.q, self.p_hat
        n = len(sw2)

        @lru_cache(maxsize=None)
        def V(t: int, p0: float, p1: float, bd: int) -> float:
            if t == 0:
                return 0.0
            ps = (p0, p1)
            nxt = tuple(ps[i] + sw2[i] for i in range(n))          # 两用户都推进
            best = V(t - 1, nxt[0], nxt[1], bd)
            if bd >= 1:
                for i in range(n):
                    g = ps[i] / (ps[i] + sm2)
                    d = (1.0 - g) * ps[i] + sw2[i]
                    nd = list(nxt)
                    nd[i] = d
                    v = (ph[i] * V(t - 1, nd[0], nd[1], bd - 1)
                         + (1.0 - ph[i]) * V(t - 1, nxt[0], nxt[1], bd - 1))
                    if v < best:
                        best = v
            return q * (p0 + p1) + best

        self.n_call += 1
        val = V(int(tau), float(P0), float(P1), int(b))
        self.n_states = V.cache_info().currsize
        return val

    def value_oriented(self, tau: int, p_lo: float, p_hi: float, b: int) -> float:
        """备选轴序：`(p_hi, p_lo)` —— A5 用。"""
        return self.value(tau, p_hi, p_lo, b)


def _knots(nf: int, coarse=(0.08, 0.12, 0.18, 0.30, 0.50, 0.90, 1.50)) -> np.ndarray:
    return np.concatenate([np.linspace(0.0, 0.055, nf + 1), np.asarray(coarse, dtype=float)])


# ---------------------------------------------------------------- 双 p 的开环/轮询参考
#: ★ 本轮自伤更正：`dp_gates.brute_opt` / `round_robin` 只接受**单个** `p`（隐含 `P0=P1`）。
#: 第一版 A1 把**非对角** case 也传给它 ⇒ 拿 `(p0,p0)` 的参考去判 `(p0,p1)` 的值，
#: A1 直接 FAIL（判据自伤，台账 #46 同族）。这里补双 `p` 版本，并**在对角上**与旧版逐位对账。
def _expected_dyn2(sw2, sm2, p_hat, act, P):
    P = list(P)
    if act < 0:
        return [P[0] + sw2[0], P[1] + sw2[1]]
    g = P[act] / (P[act] + sm2)
    P[act] = P[act] - p_hat[act] * g * P[act] + sw2[act]
    j = 1 - act
    P[j] = P[j] + sw2[j]
    return P


def brute_open_loop2(tau: int, p0: float, p1: float, budget: int,
                     sw2, sm2, p_hat, q) -> float:
    """`3^τ` 穷举**开环**序列在**期望状态**上的最优代价（`P0=p0, P1=p1`）。"""
    import itertools
    best = None
    for seq in itertools.product((-1, 0, 1), repeat=tau):
        if sum(1 for a in seq if a >= 0) > budget:
            continue
        P = [p0, p1]
        tot = 0.0
        for a in seq:
            tot += q * (P[0] + P[1])
            P = _expected_dyn2(sw2, sm2, p_hat, a, P)
        if best is None or tot < best:
            best = tot
    return float(best)


def round_robin2(tau: int, p0: float, p1: float, sw2, sm2, p_hat, q) -> float:
    """每槽交替刷新一个用户（`b=τ` 时可行）的期望代价。"""
    P = [p0, p1]
    tot = 0.0
    for t in range(tau):
        tot += q * (P[0] + P[1])
        P = _expected_dyn2(sw2, sm2, p_hat, t % 2, P)
    return float(tot)


def _plant_ctx():
    cfg = C.CWConfig(plant=C.bounded_plant())
    sw2, sm2, q = _plant(cfg)
    return cfg, sw2, float(sm2), float(q)


#: 对角案例（对称状态）：`(tau, b, p)`
DIAG = ((2, 1, 0.005), (3, 3, 0.0117), (4, 4, 0.0117), (6, 6, 0.0117), (8, 8, 0.0117))
#: 非对角案例：`(tau, b, p0, p1)` —— 本轮新增，**轴错配只在这里现形**
OFFDIAG = ((2, 2, 0.002, 0.05), (3, 3, 0.002, 0.05), (4, 4, 0.002, 0.05),
           (4, 1, 0.002, 0.05), (3, 2, 0.010, 0.030), (3, 2, 0.030, 0.010))


# ---------------------------------------------------------------- A0
def a0_info_boundary(g: Gate) -> None:
    forbidden = {"_x_true", "_log_h_true", "w_realized", "cost_true", "p_true",
                 "_pub", "_inner", "_SchedulerView__w"}
    here = pathlib.Path(__file__).parent
    hits_all: dict[str, list[str]] = {}
    for name in ("plan_dp.py", "dp_audit.py"):
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
    g.check("A0 plan_dp / dp_audit 的代码实体里不出现真值入口（AST）",
            not hits_all, {"hits": hits_all})


# ---------------------------------------------------------------- A1
def a1_reference_order(g: Gate) -> dict:
    cfg, sw2, sm2, q = _plant_ctx()
    ex = ExactClosedLoop(cfg, (0.5, 0.5))
    ph = (0.5, 0.5)
    rows, bad, xcheck = [], [], []
    cases = [(t, b, p, p) for t, b, p in DIAG] + [tuple(x) for x in OFFDIAG]
    for tau, b, p0, p1 in cases:
        e = ex.value(tau, p0, p1, b)
        br = brute_open_loop2(tau, p0, p1, b, sw2, sm2, ph, q)
        rr = round_robin2(tau, p0, p1, sw2, sm2, ph, q) if b >= tau else float("nan")
        if p0 == p1:                                  # 与旧单 p 参考逐位对账
            br_old = brute_opt(tau, p0, b, sw2, sm2, ph, q)
            rr_old = round_robin(tau, p0, sw2, sm2, ph, q) if b >= tau else float("nan")
            xcheck.append({"tau": tau, "b": b, "p": p0,
                           "d_brute": br - br_old,
                           "d_rr": (0.0 if np.isnan(rr) and np.isnan(rr_old) else rr - rr_old)})
        ok_br = e <= br * (1 + 1e-9)
        ok_rr = (np.isnan(rr)) or (e <= rr * (1 + 1e-9))
        rows.append({"tau": tau, "b": b, "p0": p0, "p1": p1, "exact": e, "brute": br,
                     "round_robin": (None if np.isnan(rr) else rr),
                     "ok_brute": bool(ok_br), "ok_rr": bool(ok_rr),
                     "exact_over_brute": e / br - 1.0})
        if not (ok_br and ok_rr):
            bad.append({"tau": tau, "b": b, "p0": p0, "p1": p1,
                        "exact": e, "brute": br,
                        "round_robin": (None if np.isnan(rr) else rr),
                        "exact_over_brute": e / br - 1.0})
    worst_x = max(abs(x["d_brute"]) for x in xcheck) if xcheck else 0.0
    g.check("A1 参考物序关系 exact ≤ brute 且 exact ≤ round_robin（24 例；含双 p 版本与旧版逐位对账）",
            not bad and worst_x <= 1e-12,
            {"n_cases": len(rows), "n_bad": len(bad), "bad": bad,
             "crosscheck_max_abs_delta_brute": worst_x,
             "crosscheck": xcheck,
             "rows": [{k: (round(v, 12) if isinstance(v, float) else v)
                       for k, v in r.items()} for r in rows]})
    return {"n_bad": len(bad), "rows": rows, "crosscheck_max_abs_delta_brute": worst_x}


# ---------------------------------------------------------------- A2
def a2_admissible(g: Gate, nf: int = 96, axis_mode: str = "user_axis",
                  emit_gate: bool = True) -> dict:
    """`emit_gate=False` 时只返回行、不注册 check。

    ★ 为什么 `legacy_swap` 不注册成 check：它的 **FAIL 是期望值**。注册会让
    `n_fail` 里长期挂一条"故意失败"，退出码恒 2，等于把"对照被抓"和"判据没过"混成一个计数
    （R1.7 教训 #88 的同族：判词不许只看一个聚合数）。`legacy` 的"必须被抓"由 **A3** 判。
    """
    cfg, _, _, _ = _plant_ctx()
    kn = _knots(nf)
    ex = ExactClosedLoop(cfg, (0.5, 0.5))
    rows, bad = [], []
    for tau, b, p in DIAG:
        e = ex.value(tau, p, p, b)
        d = float(DPTail(cfg, (0.5, 0.5), b_cap=b, knots=kn, axis_mode=axis_mode)
                  .build().value(np.array([[p, p]]), [b], tau)[0])
        rows.append({"kind": "diag", "tau": tau, "b": b, "p0": p, "p1": p,
                     "exact": e, "dp": d, "rel": d / e - 1.0})
    for tau, b, p0, p1 in OFFDIAG:
        e = ex.value(tau, p0, p1, b)
        d = float(DPTail(cfg, (0.5, 0.5), b_cap=b, knots=kn, axis_mode=axis_mode)
                  .build().value(np.array([[p0, p1]]), [b], tau)[0])
        rows.append({"kind": "off", "tau": tau, "b": b, "p0": p0, "p1": p1,
                     "exact": e, "dp": d, "rel": d / e - 1.0})
    for r in rows:
        if r["rel"] > TOL:
            bad.append(r)
    if not emit_gate:
        return {"n_bad": len(bad), "rows": rows, "gate_emitted": False}
    name = "A2 修复版可采纳：dp/exact−1 ≤ +1e-3（对角 5 + 非对角 6，6 例非对角必须在场）"
    g.check(name, not bad,
            {"nf": nf, "tol": TOL, "n_bad": len(bad),
             "worst_rel": max(r["rel"] for r in rows),
             "best_rel": min(r["rel"] for r in rows),
             "bad": [{k: (round(v, 10) if isinstance(v, float) else v) for k, v in r.items()}
                     for r in bad[:8]],
             "rows": [{k: (round(v, 10) if isinstance(v, float) else v) for k, v in r.items()}
                      for r in rows]})
    return {"n_bad": len(bad), "rows": rows}


# ---------------------------------------------------------------- A3
def a3_legacy_caught(g: Gate, nf: int = 96) -> dict:
    res = a2_admissible(g, nf=nf, axis_mode="legacy_swap", emit_gate=False)
    off = [r for r in res["rows"] if r["kind"] == "off"]
    n_over = [r for r in off if r["rel"] > TOL]
    worst = max(r["rel"] for r in res["rows"])
    g.check("A3 阴性对照（活体 legacy_swap）：非对角 ≥5 例超差且最大 ≥10%",
            len(n_over) >= 5 and worst >= FAR,
            {"n_off": len(off), "n_over_tol": len(n_over), "worst_rel": worst,
             "need_n_over": 5, "need_worst": FAR,
             "legacy_rows": [{k: (round(v, 10) if isinstance(v, float) else v)
                              for k, v in r.items()} for r in res["rows"]]})
    return {"n_over_tol": len(n_over), "worst_rel": worst, "rows": res["rows"]}


# ---------------------------------------------------------------- A4
def a4_resolution(g: Gate) -> dict:
    cfg, _, _, _ = _plant_ctx()
    ex = ExactClosedLoop(cfg, (0.5, 0.5))
    tau, b, p0, p1 = 4, 4, 0.002, 0.05
    e = ex.value(tau, p0, p1, b)
    rows = []
    for nf in (24, 96):
        kn = _knots(nf)
        row = {"nf": nf, "exact": e}
        for mode in ("user_axis", "legacy_swap"):
            d = float(DPTail(cfg, (0.5, 0.5), b_cap=b, knots=kn, axis_mode=mode)
                      .build().value(np.array([[p0, p1]]), [b], tau)[0])
            row[mode] = d / e - 1.0
        rows.append(row)
    fix_ok = all(r["user_axis"] <= TOL for r in rows)
    leg_far = all(r["legacy_swap"] >= FAR for r in rows)
    g.check("A4 加密两档：修复版都 ≤tol，legacy 都 ≥100×tol（⇒ 误差与网格无关＝结构性）",
            fix_ok and leg_far,
            {"rows": [{k: (round(v, 8) if isinstance(v, float) else v) for k, v in r.items()}
                      for r in rows], "tol": TOL, "far": FAR,
             "fix_all_ok": fix_ok, "legacy_all_far": leg_far})
    return {"rows": rows, "fix_all_ok": fix_ok, "legacy_all_far": leg_far}


# ---------------------------------------------------------------- A5
def a5_axis_distinguishable(g: Gate, nf: int = 96) -> dict:
    cfg, sw2, _, _ = _plant_ctx()
    ex = ExactClosedLoop(cfg, (0.5, 0.5))
    kn = _knots(nf)
    tau, b, p0, p1 = 3, 3, 0.002, 0.05
    dp = DPTail(cfg, (0.5, 0.5), b_cap=b, knots=kn).build()
    v_a = float(dp.value(np.array([[p0, p1]]), [b], tau)[0])
    v_b = float(dp.value(np.array([[p1, p0]]), [b], tau)[0])
    e_a = ex.value(tau, p0, p1, b)
    e_b = ex.value(tau, p1, p0, b)
    rel_dp = abs(v_a - v_b) / max(v_a, v_b)
    rel_ex = abs(e_a - e_b) / max(e_a, e_b)
    g.check("A5 两条 P 轴可分辨（sw2 不对称 ⇒ V(p0,p1) ≠ V(p1,p0)，相对差 ≥1e-3）",
            rel_dp >= 1e-3 and rel_ex >= 1e-3,
            {"dp_a": v_a, "dp_b": v_b, "rel_dp": rel_dp,
             "exact_a": e_a, "exact_b": e_b, "rel_exact": rel_ex, "sw2": list(sw2)})
    return {"rel_dp": rel_dp, "rel_exact": rel_ex}


# ---------------------------------------------------------------- A6
def a6_mutations(g: Gate, nf: int = 96) -> tuple[list[dict], int]:
    from . import plan_dp as D
    cfg, _, _, _ = _plant_ctx()
    kn = _knots(nf)
    ex = ExactClosedLoop(cfg, (0.5, 0.5))
    tau, b, p0, p1 = 4, 4, 0.002, 0.05
    e = ex.value(tau, p0, p1, b)
    rows: list[dict] = []

    def run(**kw):
        return float(D.DPTail(cfg, (0.5, 0.5), b_cap=b, knots=kn, **kw)
                     .build().value(np.array([[p0, p1]]), [b], tau)[0]) / e - 1.0

    base = run()
    rows.append({"mutation": "基准（未变异）", "caught": None, "evidence": {"rel": base}})

    orig_maps = D.DPTail._build_maps

    def _swap(self):                                  # noqa: ANN001
        m = orig_maps(self)
        return {(i, tag): m[(1 - i, tag)] for (i, tag), _v in m.items()}

    try:
        D.DPTail._build_maps = _swap
        e1 = run()
    finally:
        D.DPTail._build_maps = orig_maps
    rows.append({"mutation": "M1 把两张后继映射按用户交换（轴混）",
                 "caught": bool(e1 > TOL), "evidence": {"rel": e1, "base": base}})

    orig_init = D.DPTail.__init__

    def _ph1(self, cfg_, p_hat, **kw):                # noqa: ANN001
        orig_init(self, cfg_, np.ones(2), **kw)

    try:
        D.DPTail.__init__ = _ph1
        e2 = run()
    finally:
        D.DPTail.__init__ = orig_init
    rows.append({"mutation": "M2 送达概率置 1（忽略 p̂）",
                 "caught": bool(abs(e2 - base) > 10 * TOL),
                 "evidence": {"rel": e2, "base": base}})

    judged = [r for r in rows if r["caught"] is not None]
    n_missed = sum(0 if r["caught"] else 1 for r in judged)
    g.check("A6 变异 2/2 被抓（轴混 / 忽略 p̂）", n_missed == 0,
            {"rows": [{k: (round(v, 10) if isinstance(v, float) else v) for k, v in r.items()}
                      for r in rows], "n_missed": n_missed})
    return rows, n_missed


# ---------------------------------------------------------------- 入口
def run_all() -> tuple[Gate, dict]:
    g = Gate()
    a0_info_boundary(g)
    x1 = a1_reference_order(g)
    x2 = a2_admissible(g)
    x3 = a3_legacy_caught(g)
    x4 = a4_resolution(g)
    x5 = a5_axis_distinguishable(g)
    muts, n_missed = a6_mutations(g)
    return g, {"ref_order": x1, "fixed": x2, "legacy": x3, "resolution": x4,
               "axis": x5, "mutations": muts, "n_mut_missed": n_missed}
