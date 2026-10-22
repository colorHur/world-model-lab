# -*- coding: utf-8 -*-
"""CW-R1 有限视界规划参考的**实现闸**（`plan_ref.py` 的验收面，不是收益表）。

为什么先写闸（卡 §6.2 / 台账 ⑫#39）
------------------------------------
"规划器跑出数了"不等于"它按我说的规则在规划"。本模块每条闸都写清
**什么错误能让它 FAIL**，并用三次单变量破坏证明闸有分辨力。

闸清单
------
G0 `_p_hat` 与 `posterior.p_succ_expected` 逐位一致（同点数、同归一化）。
   ⇒ FAIL 于：求积点数/归一化/温度参数写错 —— 那会让"期望成功率"整条链偏。
G1 信息边界（三条子检）：
   G1a AST/词法：`plan_ref.py` 不得出现任何真值入口名。
   G1b 同一份**公开快照**重建视图 ⇒ 动作逐位相同（证明动作只是快照的函数）。
   G1c 取视图后**改动世界隐藏真值**（x/h 各加常数）⇒ 动作不变；
       而**改动公开快照里的 P̂** ⇒ 动作必须变（阴性对照，证明 G1c 不是恒真）。
   ⇒ FAIL 于：任何"偷看真值 / 抓世界句柄 / 只在真值变化时才稳定"的实现。
G2 预算（两条子检）：
   G2a 外部：`budget ∈ {0,1,2,8}` 下 `n_forced_hold == 0`、12 项记账恒等式全过。
   G2b 内部：rollout 的 `bud` 每做一次资源动作必须恰好减 1（不多不少）。
   ⇒ FAIL 于：动作不受预算约束 / rollout 不扣预算（后者会让尾部高估可用帧）。
G3 `H=1` 退化为贪心（两条子检）：
   G3a 与**独立重写**的标量实现（显式循环算尾部，不用闭式解）在 4 个时点上
       给出同一个最优动作（在 1e-9 内同为最小）。
   G3b `H=1` 整集 `n_probe == 0`（结构事实：PROBE 不动任务估计 ⇒ 与 HOLD 同值）。
   ⇒ FAIL 于：闭式尾部写错、`H` 没真正接到 rollout、并列时乱花钱。
G4 `M` 加倍稳定：`M=16` vs `M=32` 逐槽动作一致率 ≥ 0.90、逐 episode 成本相对差中位 ≤ 10%。
   ⇒ FAIL 于：估计量方差爆炸（少采样就换动作）或有未播种的随机源。
G5 探测必须**接进后续**（四条子检）：
   G5a `n_probe > 0`（否则命中卡 §5 的 N3：实现缺陷，不是科学结论）。
   G5b 探测臂与禁探测臂的动作序列**必须不同**（否则探测没进决策）。
   G5c 探测之后**同一 episode 的更晚时隙**动作要与禁探测臂不同（证明续接耦合，
       而不是"本槽换个动作就完了"）。
   G5d 存在"探测的用户 == 本槽最大 P̂ 用户"的情形（证明探测价值不只在"本槽换人"）。
G6 指标口径更正的可复算依据：`HOLD` 的 `cost_est == 0` 而 `cost_true > 1`。
   ⇒ 这条不是"规划器的闸"，是把"主指标为什么从 `cost_est` 改成 `cost_true`"钉成可复算事实。
G7 规划器必须优于 HOLD（弱但必要：比什么都不做还差的"参考"没有比较价值）。
"""

from __future__ import annotations

import numpy as np

from . import config as C
from . import plan_ref as Pmodule
from .gates import Gate
from .plan_ref import PlanRef, make
from .posterior import p_succ_expected
from .runner import check_identities, run_episode
from .world import Action, CWWorld, SchedulerView
from .streams import make_episode_rngs

RHO_DEFAULT = 0.98


# ---------------------------------------------------------------- 小工具
class _Rec:
    """记录逐槽动作的包装器（不改变内层行为）。"""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.acts: list[tuple[int, str, int | None]] = []
        self.name = getattr(inner, "name", "rec")

    def __call__(self, v):
        a = self.inner(v)
        self.acts.append((int(v.t), a.kind, a.user))
        return a


def _cfg() -> C.CWConfig:
    return C.CWConfig(plant=C.bounded_plant())


def _hold_policy():
    class _H:
        name = "hold"

        def __call__(self, v):
            return Action("HOLD")

    return _H()


# ---------------------------------------------------------------- G0
def g0_p_hat(g: Gate) -> None:
    rng = np.random.default_rng(7)
    cfg = _cfg()
    pol = make(cfg, arm="probe", h_plan=1, m_samples=1)
    ms = rng.normal(0.0, 1.0, 200)
    vs = rng.uniform(0.0, 2.5, 200)
    a = np.array([float(pol._p_hat(np.array([[m]]), np.array([[v]]))[0, 0])
                  for m, v in zip(ms, vs)])
    b = np.array([p_succ_expected(float(m), float(v), cfg) for m, v in zip(ms, vs)])
    d = float(np.max(np.abs(a - b)))
    g.check("G0 _p_hat 与 posterior.p_succ_expected 逐位一致（≤1e-12）", d <= 1e-12,
            {"max_abs_diff": d, "n": int(ms.size)})


# ---------------------------------------------------------------- G1
def g1_info_boundary(g: Gate) -> None:
    import ast
    import pathlib

    src = pathlib.Path(__file__).with_name("plan_ref.py").read_text(encoding="utf-8")
    forbidden = {"_x_true", "_log_h_true", "w_realized", "cost_true", "p_true",
                 "_pub", "_inner", "_SchedulerView__w"}
    # ★ 必须走 AST 而不是字符串搜索：文档字符串里写着"不读 `_x_true`"是**允许**的，
    #   字符串搜索会把它判成泄漏（本轮第一次跑 G1a 就是这么误报的）。
    #   AST 只看**代码实体的名字**（属性/变量/关键字/形参），注释与 docstring 是
    #   `Constant` 节点，天然不算。
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
    g.check("G1a plan_ref 的**代码实体**里不出现任何真值入口（AST 检查）", not hits,
            {"hits": hits, "n_forbidden": len(forbidden)})

    cfg = _cfg()
    rngs, sids = make_episode_rngs(cfg.root_seed, "dev", 100)
    w = CWWorld(cfg, rngs, sids, rho=RHO_DEFAULT)
    w.budget_total = 64
    for _ in range(6):
        w.step(Action("HOLD"))
    v = w.view()

    seed = 424242
    a1 = make(cfg, arm="probe", seed=seed, rho=RHO_DEFAULT)(v)

    # G1b 同一份公开快照 → 重建视图 → 同一动作
    import copy
    pub = copy.deepcopy(v._pub)
    v2 = SchedulerView(pub)
    a2 = make(cfg, arm="probe", seed=seed, rho=RHO_DEFAULT)(v2)

    # G1c 改动隐藏真值后，**同一份视图**仍给同一动作
    w._x_true = w._x_true + 50.0
    w._log_h_true = w._log_h_true + 3.0
    a3 = make(cfg, arm="probe", seed=seed, rho=RHO_DEFAULT)(v)

    g.check("G1b 同一公开快照（深拷贝重建视图）⇒ 动作相同", (a1 == a2),
            {"a1": repr(a1), "a2": repr(a2)})
    g.check("G1c 仅改隐藏真值（同视图）⇒ 动作不变", (a1 == a3),
            {"a1": repr(a1), "a3": repr(a3), "delta_x": 50.0, "delta_logh": 3.0})

    # 阴性对照：改公开快照里的一格 P̂ 必须能改变动作（否则 G1c 没有分辨力）
    changed = 0
    for bump in (2.5, 12.0, 60.0):
        pub2 = copy.deepcopy(v._pub)
        tel = [dict(r) for r in pub2["telemetry"]]
        tel[0]["P_hat"] = float(tel[0]["P_hat"]) + bump
        pub2["telemetry"] = tel
        aa = make(cfg, arm="probe", seed=seed, rho=RHO_DEFAULT)(SchedulerView(pub2))
        if aa != a1:
            changed += 1
    g.check("G1d 阴性对照：改公开 P̂ 至少一次改变动作（G1c 不是恒真）", changed >= 1,
            {"n_changed": changed, "bumps": [2.5, 12.0, 60.0]})


# ---------------------------------------------------------------- G2
def g2_budget(g: Gate) -> None:
    cfg = _cfg()
    rows = []
    bad = []
    for budget in (0, 1, 2, 8):
        pol = make(cfg, arm="probe", seed=99, rho=RHO_DEFAULT)
        r = run_episode(cfg, pol, split="dev", episode=100, budget_total=budget,
                        rho=RHO_DEFAULT)
        ids = check_identities(r, cfg)
        rows.append({"budget": budget, "used": r.budget_used, "forced": r.n_forced_hold,
                     "probe": r.n_probe, "attempted": r.n_attempted, "ids": ids["pass"]})
        if not ids["pass"] or r.n_forced_hold != 0 or r.budget_used > budget:
            bad.append(budget)
    g.check("G2a 预算 0/1/2/8 下不越支、无强制 HOLD、恒等式全过", not bad,
            {"failed_budgets": bad, "rows": rows})

    # G2b rollout 内部预算会计：连做 3 次 SEND + 1 次 PROBE，bud 必须恰好减 4
    rngs, sids = make_episode_rngs(cfg.root_seed, "dev", 100)
    w = CWWorld(cfg, rngs, sids, rho=RHO_DEFAULT)
    w.budget_total = 64
    pol = make(cfg, arm="probe", seed=5, rho=RHO_DEFAULT)
    dd = Pmodule._Draws(np.random.default_rng(1), 8, pol.M, 2)
    st = pol._init_state(w.view(), dd)
    b0 = float(st.bud[0])
    for k in range(3):
        pol._advance(st, Action("SEND", 0), k=k, d=dd)
    pol._advance(st, Action("PROBE", 1), k=3, d=dd)
    pol._advance(st, Action("HOLD"), k=4, d=dd)
    d = b0 - float(st.bud[0])
    g.check("G2b rollout 内部预算：3 次 SEND + 1 次 PROBE + 1 次 HOLD ⇒ 预算恰减 4",
            abs(d - 4.0) < 1e-12, {"bud_before": b0, "bud_after": float(st.bud[0]),
                                   "decrement": d})


# ---------------------------------------------------------------- G3
def _independent_h1_choice(cfg: C.CWConfig, tel: list[dict], budget_left: int,
                           steps_left: int, acts: tuple[Action, ...]) -> tuple[Action, dict]:
    """**独立重写**的 H=1 一步值：尾部用显式循环（不用闭式解），标量逐用户算。"""
    p = cfg.plant
    A = np.asarray(p.A, dtype=float)
    B, q, r = float(p.B), float(p.q_cost), float(p.r_cost)
    sw2 = np.asarray(p.sigma_w, dtype=float) ** 2
    sm2 = float(p.sigma_meas) ** 2
    K = np.asarray([PlanRef(cfg, h_plan=1, m_samples=1)._riccati_k(i)
                    for i in range(p.n_users)])
    alpha = A - B * K
    xh = np.asarray([r0["x_hat"] for r0 in tel], dtype=float)
    P = np.asarray([r0["P_hat"] for r0 in tel], dtype=float)
    mh = np.asarray([r0["h_mean"] for r0 in tel], dtype=float)
    vh = np.asarray([r0["h_var"] for r0 in tel], dtype=float)

    def term(Pv: np.ndarray) -> float:
        acc = 0.0
        Pc = Pv.copy()
        xc = alpha * xh
        for _ in range(max(steps_left - 1, 0)):
            acc += q * float(np.sum(Pc + xc ** 2)) + r * float(np.sum((K * xc) ** 2))
            Pc = A ** 2 * Pc + sw2
            xc = alpha * xc
        return acc

    vals: dict[Action, float] = {}
    for a in acts:
        if a.kind == "SEND":
            i = a.user
            gk = P[i] / (P[i] + sm2)
            P_del = A[i] ** 2 * ((1 - gk) * P[i]) + sw2[i]
            P_nodel = A[i] ** 2 * P[i] + sw2[i]
            pk = p_succ_expected(float(mh[i]), float(vh[i]), cfg)
            P2 = P.copy()
            P2[i] = pk * P_del + (1 - pk) * P_nodel
            vals[a] = term(P2)
        else:
            vals[a] = term(A ** 2 * P + sw2)
    best = min(vals.values())
    pick = [a for a in acts if vals[a] <= best + 1e-9]
    return pick[0], {repr(k): v for k, v in vals.items()}


def g3_h1_degenerate(g: Gate) -> None:
    cfg = _cfg()
    rngs, sids = make_episode_rngs(cfg.root_seed, "dev", 100)
    w = CWWorld(cfg, rngs, sids, rho=RHO_DEFAULT)
    w.budget_total = 100
    rows, bad = [], []
    for k in range(4):
        for _ in range(3):
            w.step(Action("HOLD") if k % 2 == 0 else Action("SEND", k % 2))
        v = w.view()
        pol = make(cfg, arm="probe", h_plan=1, seed=1000 + k, rho=RHO_DEFAULT)
        chosen = pol(v)
        tel = v.telemetry
        acts = tuple(a for a in pol._cand)
        _, vals = _independent_h1_choice(cfg, tel, int(v.budget_left),
                                         int(cfg.protocol.horizon - v.t), acts)
        best = min(vals.values())
        mine = vals[repr(chosen)]
        rows.append({"t": int(v.t), "chosen": repr(chosen), "value": mine, "min": best})
        if mine > best + 1e-9:
            bad.append(int(v.t))
    g.check("G3a H=1 的根动作与独立重写的一步值最优一致（4 个时点）", not bad,
            {"failed_t": bad, "rows": rows})

    pol = make(cfg, arm="probe", h_plan=1, seed=321, rho=RHO_DEFAULT)
    r = run_episode(cfg, pol, split="dev", episode=100,
                    budget_total=cfg.protocol.budget(0.5), rho=RHO_DEFAULT)
    g.check("G3b H=1 整集 n_probe == 0（PROBE 与 HOLD 同值 ⇒ 不花钱）", r.n_probe == 0,
            {"n_probe": r.n_probe, "n_attempted": r.n_attempted})


# ---------------------------------------------------------------- G4
def g4_m_stability(g: Gate) -> None:
    cfg = _cfg()
    budget = cfg.protocol.budget(0.5)
    agree, n_tot, diffs = 0, 0, []
    per_ep = []
    for ep in (100, 101):
        outs = {}
        for M in (16, 32):
            pol = _Rec(make(cfg, arm="probe", h_plan=5, m_samples=M, seed=777,
                            rho=RHO_DEFAULT))
            r = run_episode(cfg, pol, split="dev", episode=ep, budget_total=budget,
                            rho=RHO_DEFAULT)
            outs[M] = (pol.acts, r.cost_true_per_step)
        a16, a32 = outs[16][0], outs[32][0]
        for (t1, k1, u1), (t2, k2, u2) in zip(a16, a32):
            n_tot += 1
            if (k1, u1) == (k2, u2):
                agree += 1
        diffs.append(abs(outs[32][1] - outs[16][1]) / max(outs[16][1], 1e-12))
        per_ep.append({"episode": ep, "c16": outs[16][1], "c32": outs[32][1],
                       "rel": diffs[-1]})
    rate = agree / max(n_tot, 1)
    med = float(np.median(diffs))
    g.check("G4 M 加倍（16→32）动作一致率 ≥ 0.90 且成本相对差中位 ≤ 10%",
            rate >= 0.90 and med <= 0.10,
            {"agreement": rate, "n_compared": n_tot, "median_rel_cost_diff": med,
             "per_episode": per_ep})


# ---------------------------------------------------------------- G5
def g5_probe_coupling(g: Gate) -> None:
    cfg = _cfg()
    budget = cfg.protocol.budget(0.5)
    stats = {"probe": 0, "send": 0, "hold": 0, "same_argmax": 0, "diff_argmax": 0}
    per_ep = []
    differ_any, later_diff = 0, 0
    for ep in (100, 101, 102):
        pa = _Rec(make(cfg, arm="probe", h_plan=5, m_samples=16, seed=ep * 13,
                       rho=RHO_DEFAULT))
        pb = _Rec(make(cfg, arm="noprobe", h_plan=5, m_samples=16, seed=ep * 13,
                       rho=RHO_DEFAULT))
        ra = run_episode(cfg, pa, split="dev", episode=ep, budget_total=budget,
                         rho=RHO_DEFAULT)
        rb = run_episode(cfg, pb, split="dev", episode=ep, budget_total=budget,
                         rho=RHO_DEFAULT)
        stats["probe"] += ra.n_probe
        stats["send"] += ra.n_attempted
        stats["hold"] += ra.n_idle
        stats["same_argmax"] += pa.inner.probe_same_argmax
        stats["diff_argmax"] += pa.inner.probe_diff_argmax
        if ra.policy_trace_hash != rb.policy_trace_hash:
            differ_any += 1
        # 探测之后的更晚时隙是否有动作差异
        probe_ts = [t for (t, k, _u) in pa.acts if k == "PROBE"]
        d16 = {t: (k, u) for (t, k, u) in pa.acts}
        d_other = {t: (k, u) for (t, k, u) in pb.acts}
        if probe_ts:
            last = min(probe_ts)
            if any(d16[t] != d_other[t] for t in d16 if t > last):
                later_diff += 1
        per_ep.append({"episode": ep, "probe": ra.n_probe, "send": ra.n_attempted,
                       "deliv": ra.n_delivered, "cost_true": ra.cost_true_per_step,
                       "cost_true_noprobe": rb.cost_true_per_step,
                       "trace_differs": ra.policy_trace_hash != rb.policy_trace_hash})

    g.check("G5a 探测臂 n_probe > 0（否则命中 N3：实现缺陷）", stats["probe"] > 0, stats)
    g.check("G5b 探测臂 / 禁探测臂动作序列不同（探测真的进了决策）",
            differ_any >= 1, {"n_episodes_diff": differ_any})
    g.check("G5c 探测之后更晚时隙的动作发生变化（续接耦合，不只本槽换人）",
            later_diff >= 1, {"n_episodes_late_diff": later_diff})
    g.check("G5d 存在「探测对象 == 本槽最大 P̂ 用户」（价值不只来自换人）",
            stats["same_argmax"] > 0, {"same": stats["same_argmax"],
                                       "diff": stats["diff_argmax"]})


# ---------------------------------------------------------------- G6
def g6_metric(g: Gate) -> None:
    cfg = _cfg()
    r = run_episode(cfg, _hold_policy(), split="dev", episode=100,
                    budget_total=cfg.protocol.budget(0.5), rho=RHO_DEFAULT)
    pol = make(cfg, arm="probe", h_plan=5, seed=11, rho=RHO_DEFAULT)
    K = pol.K
    g.check("G6 HOLD 的 cost_est ≡ 0 而 cost_true 明显 > 0（主指标口径更正的可复算依据）",
            abs(r.cost_est_per_step) < 1e-12 and r.cost_true_per_step > 1.0,
            {"hold_cost_est": r.cost_est_per_step, "hold_cost_true": r.cost_true_per_step,
             "A_minus_BK": [float(x) for x in (np.asarray(cfg.plant.A) - cfg.plant.B * K)],
             "reason": "A-BK≈0.1433 ⇒ 不发帧时 x̂ 几何衰减 ⇒ q·x̂²+r·u²→0"})


# ---------------------------------------------------------------- G7
def g7_beats_hold(g: Gate) -> None:
    cfg = _cfg()
    budget = cfg.protocol.budget(0.5)
    ch, cp = [], []
    for ep in (100, 101):
        ch.append(run_episode(cfg, _hold_policy(), split="dev", episode=ep,
                              budget_total=budget, rho=RHO_DEFAULT).cost_true_per_step)
        cp.append(run_episode(cfg, make(cfg, arm="probe", h_plan=5, seed=ep,
                                        rho=RHO_DEFAULT), split="dev", episode=ep,
                              budget_total=budget, rho=RHO_DEFAULT).cost_true_per_step)
    ok = all(p < h for p, h in zip(cp, ch))
    g.check("G7 规划器优于 HOLD（逐 episode）", ok,
            {"plan": cp, "hold": ch, "ratios": [p / h for p, h in zip(cp, ch)]})


# ---------------------------------------------------------------- 变异闸
def mutation_gates() -> tuple[list[dict], int]:
    """三次单变量破坏，要求指定闸从 PASS 翻成 FAIL。

    M1 PROBE 的测量**不进信念**（仍扣预算）⇒ `G5a`（n_probe>0）必须失败。
    M2 `_pick` 无视预算（照旧返回 SEND）⇒ `G2a`（n_forced_hold==0）必须失败。
    M3 rollout 不扣预算（`bud` 不减）⇒ `G2b`（预算恰减 4）必须失败。
    """
    from . import plan_ref as P

    rows: list[dict] = []
    cfg = _cfg()

    # M1
    orig = P.PlanRef._advance

    def _probe_noop(self, st, act, *, k, d):      # noqa: ANN001
        if act.kind != "PROBE":
            return orig(self, st, act, k=k, d=d)
        aff = st.bud >= 1
        st.bud = st.bud - np.where(aff, 1.0, 0.0)   # 钱照扣
        st.mh = self.rho * st.mh                     # 但什么都不学
        st.vh = self.rho * self.rho * st.vh + (1.0 - self.rho * self.rho)
        if self.sample_hidden:
            u = -self.K[None, :] * st.xh
            st.ht = (self.rho * st.ht
                     + np.sqrt(max(1e-12, 1.0 - self.rho ** 2)) * d.s_h[k])
            st.xt = self.A[None, :] * st.xt + self.B * u + d.w[k] * self.sw[None, :]

    try:
        P.PlanRef._advance = _probe_noop
        pol = P.make(cfg, arm="probe", h_plan=5, seed=17, rho=RHO_DEFAULT)
        r = run_episode(cfg, pol, split="dev", episode=100,
                        budget_total=cfg.protocol.budget(0.5), rho=RHO_DEFAULT)
        n_probe = r.n_probe
    finally:
        P.PlanRef._advance = orig
    rows.append({"mutation": "M1 PROBE 测量不进信念（钱照扣）", "target": "G5a",
                 "caught": bool(n_probe == 0), "evidence": {"n_probe": n_probe}})

    # M2
    orig_pick = P.PlanRef._pick

    def _reckless(self, values, budget_left):     # noqa: ANN001
        return Action("SEND", 0)

    try:
        P.PlanRef._pick = _reckless
        pol = P.make(cfg, arm="probe", h_plan=5, seed=18, rho=RHO_DEFAULT)
        r = run_episode(cfg, pol, split="dev", episode=100, budget_total=3,
                        rho=RHO_DEFAULT)
        forced = r.n_forced_hold
    finally:
        P.PlanRef._pick = orig_pick
    rows.append({"mutation": "M2 _pick 无视预算", "target": "G2a",
                 "caught": bool(forced > 0), "evidence": {"n_forced_hold": forced,
                                                          "budget": 3}})

    # M3
    orig_adv = P.PlanRef._advance

    def _free_advance(self, st, act, *, k, d):   # noqa: ANN001
        bud = st.bud.copy()
        orig_adv(self, st, act, k=k, d=d)
        st.bud = bud                              # 预算不扣

    try:
        P.PlanRef._advance = _free_advance
        rngs, sids = make_episode_rngs(cfg.root_seed, "dev", 100)
        w = CWWorld(cfg, rngs, sids, rho=RHO_DEFAULT)
        w.budget_total = 64
        pol = P.make(cfg, arm="probe", seed=5, rho=RHO_DEFAULT)
        dd = P._Draws(np.random.default_rng(1), 8, pol.M, 2)
        st = pol._init_state(w.view(), dd)
        b0 = float(st.bud[0])
        for k in range(3):
            pol._advance(st, Action("SEND", 0), k=k, d=dd)
        dec = b0 - float(st.bud[0])
    finally:
        P.PlanRef._advance = orig_adv
    rows.append({"mutation": "M3 rollout 不扣预算", "target": "G2b",
                 "caught": bool(abs(dec - 3.0) > 1e-12),
                 "evidence": {"decrement": dec, "expected": 3.0}})

    return rows, sum(0 if r["caught"] else 1 for r in rows)


# ---------------------------------------------------------------- 入口
def run_all() -> Gate:
    g = Gate()
    g0_p_hat(g)
    g1_info_boundary(g)
    g2_budget(g)
    g3_h1_degenerate(g)
    g4_m_stability(g)
    g5_probe_coupling(g)
    g6_metric(g)
    g7_beats_hold(g)
    return g
