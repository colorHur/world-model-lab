# -*- coding: utf-8 -*-
"""CW-R1.7.2 的**接线闸**：`DPTail` 接进 `plan_ref._terminal` 之后的验收面。

判据与否定条件在看数前写死在 `docs/research/CW-R1.7.2-DP接plan_ref-预注册-v0.1.md` §3。

四道闸（`W0/W0b/W1/W1b`）＋两个**诊断量**（`W2/W3`，不设阈值，只报数）
--------------------------------------------------------------------------
W0  `tail="dp"` 的 `_terminal` 与**独立重算**逐位一致（≤1e-12）。
    ★ 独立重算的三件事：增益 `K` 用**二分法**解同一个 DARE（不是规划器的
      不动点迭代）、`Σx̂²` 的几何和另写、`DPTail.value` 直接调。
    ★ **必须用不对称的 `p̂=(0.70,0.35)`、不对称的 `P`、不对称的 `bud`** ——
      R1.7.1 的教训：只要有一处取两用户对称，"用户轴与参数位置错配"就**恒等不可见**。
W0b 预算管线：`_expect_after` 必须把 **动作之后**的预算传给 `_terminal`
    （`HOLD` 传 `b`、`PROBE`/`SEND` 传 `b−1`）。这条专抓"bud 传成动作前"这一类接线错，
    （`q` 常数平移在成本比较里看不出来，但预算传错会系统性把一帧定价成 0）。
W1  值函数对预算**严格递减**（= 链上的 `G8d[dp]`）：`steps=128`、`P0=0.01`、`p̂=(0.60,0.45)`、
    `b∈[0,128]` ⇒ 上升 **0 对**、平坦比例 **≤10%**。
W1b `W1` 的**活体**阴性对照：`b_cap=1`（真实接线错误：让饱和律把 `b` 全夹成 0/1）
    ⇒ 平坦比例必须 **>90%**。若它不 FAIL，`W1` 的 ≤10% 就没有分辨力（台账 #39/#95）。

W2  价格对账（**核心科学量**）：`price_analytic` vs `price_dp`，同点同 `τ` 同 `b`。
W3  `p̂` 冻结代价：episode 起点先验 `p̂=0.5` vs 槽初 `p̂(m_h,v_h)`，报相对差。
"""

from __future__ import annotations

import numpy as np

from . import config as C
from . import plan_ref as Pmodule
from .gates import Gate
from .plan_dp import get_dp
from .plan_ref import make
from .world import Action

#: W2 的工作点（与 R1.6.1 的"一帧定价 0.0225"尽可能同口径：`bounded_plant`、`P0=0.01`）
W2_TAU = 128
W2_P0 = 0.01
W2_B = 64


def _cfg() -> C.CWConfig:
    return C.CWConfig(plant=C.bounded_plant())


def _dare_k_bisect(A: float, B: float, q: float, r: float) -> float:
    """**独立算法**（二分）解标量 DARE，返回增益 `K`。

    残差 `g(P) = q + A²P − (APB)²/(r+B²P) − P`：`g(0)=q>0`，从 0 往上单调降到负。
    用二分而不是规划器的 `P ← g(P)+P` 不动点迭代，是为了让 `W0` 抓得住
    "公式对、迭代实现被改坏"这一类；若两边共用同一个错公式，它会一起错 ——
    这一条由 `W0` 里额外对账 `pol.K` 的**数值**（已知量级）来兜。
    """
    def res(P: float) -> float:
        return q + A * A * P - (A * P * B) ** 2 / (r + B * B * P) - P

    lo, hi = 0.0, 1.0
    while res(hi) > 0.0 and hi < 1e12:
        hi *= 2.0
    for _ in range(400):
        mid = 0.5 * (lo + hi)
        if res(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    P = 0.5 * (lo + hi)
    return A * P * B / (r + B * B * P)


# ------------------------------------------------------------------ W0
def w0_interface(g: Gate) -> None:
    cfg = _cfg()
    p = cfg.plant
    q, r, B = float(p.q_cost), float(p.r_cost), float(p.B)
    Ks = np.asarray([_dare_k_bisect(float(a), B, q, r) for a in p.A], dtype=float)
    al = np.asarray(p.A, dtype=float) - B * Ks

    pol = make(cfg, arm="probe", h_plan=1, m_samples=1, tail="dp",
               dp_phat=np.asarray([0.70, 0.35], dtype=float))
    pol._wire_dp(None)

    # 先对账"独立 K"本身：若两边差很多，下面的"一致"就毫无意义
    k_rel = float(np.max(np.abs(Ks - pol.K) / np.maximum(np.abs(pol.K), 1e-12)))

    P = np.asarray([[0.004, 0.030], [0.020, 0.008]], dtype=float)   # 不对称，且两行不同
    xh = np.asarray([[-0.60, 0.35], [0.10, -1.20]], dtype=float)
    worst, worst_tag, n = 0.0, None, 0
    for steps in (1, 7, 128, 255):
        for bud in (np.asarray([5.0, 200.0]), np.asarray([0.0, 3.0])):
            got = pol._terminal(P, xh, steps, bud)
            a2 = (al ** 2)[None, :]
            geo = a2 * (1.0 - a2 ** steps) / (1.0 - a2)          # Σ_{k=1..steps} α^{2k}
            sX2 = (xh ** 2) * geo
            want = (pol._dp.value(P, bud, steps)
                    + q * np.sum(sX2, axis=1)
                    + r * np.sum((Ks ** 2)[None, :] * sX2, axis=1))
            rel = float(np.max(np.abs(got - want) / np.maximum(np.abs(want), 1e-12)))
            n += 1
            if rel > worst:
                worst, worst_tag = rel, {"steps": steps, "bud": bud.tolist()}

    g.check("W0 接线一致性：_terminal(tail=dp) vs 独立重算 ≤1e-12"
            "（不对称 p̂=(0.70,0.35)、不对称 P、不对称 bud）",
            (worst <= 1e-12) and (k_rel <= 1e-9),
            {"max_rel": worst, "worst_case": worst_tag, "n_cases": n,
             "k_rel_vs_planner": k_rel, "K": Ks.tolist(), "K_planner": pol.K.tolist(),
             "p_hat_asym": [0.70, 0.35]})


# ------------------------------------------------------------------ W0b
def w0b_budget_plumbing(g: Gate) -> None:
    cfg = _cfg()
    p = cfg.plant
    q, r = float(p.q_cost), float(p.r_cost)
    sm2 = float(p.sigma_meas) ** 2
    pol = make(cfg, arm="probe", h_plan=1, m_samples=1, tail="dp",
               dp_phat=np.asarray([0.50, 0.50], dtype=float))
    pol._wire_dp(None)

    a2 = (np.asarray(p.A, dtype=float) ** 2)[None, :]
    sw2 = (np.asarray(p.sigma_w, dtype=float) ** 2)[None, :]
    xh0 = np.asarray([[0.30, -0.20], [0.05, 0.40]], dtype=float)
    P0 = np.asarray([[0.010, 0.020], [0.006, 0.012]], dtype=float)
    mh0 = np.asarray([[-0.20, 0.30], [0.00, 0.10]], dtype=float)
    vh0 = np.asarray([[0.80, 0.60], [1.00, 1.00]], dtype=float)
    bud0 = np.asarray([7.0, 200.0])
    st = Pmodule._St(xh0, P0, mh0, vh0, xh0, mh0, bud0.copy())
    steps_after = 40

    worst, rows = 0.0, []
    for act in (Action("HOLD"), Action("PROBE", 0), Action("PROBE", 1),
                Action("SEND", 0), Action("SEND", 1)):
        got = pol._expect_after(st, act, steps_after)
        need = pol._need(act)
        bud_after = np.maximum(bud0 - float(need), 0.0)
        P2 = a2 * P0 + sw2
        if act.kind == "SEND":
            i = int(act.user)
            gmeas = P0[:, i] / (P0[:, i] + sm2)
            P_del = a2[:, i] * ((1.0 - gmeas) * P0[:, i]) + sw2[:, i]
            p_i = pol._p_hat(mh0, vh0)[:, i]
            P2 = P2.copy()
            P2[:, i] = p_i * P_del + (1.0 - p_i) * (a2[:, i] * P0[:, i] + sw2[:, i])
        want = pol._terminal(P2, pol.alpha[None, :] * xh0, steps_after, bud=bud_after)
        rel = float(np.max(np.abs(got - want) / np.maximum(np.abs(want), 1e-12)))
        worst = max(worst, rel)
        rows.append({"action": f"{act.kind}{'' if act.user is None else act.user}",
                     "need": need, "bud_after": bud_after.tolist(), "rel": rel})

    g.check("W0b 预算管线：_expect_after 传的是动作**之后**的预算（HOLD 传 b、PROBE/SEND 传 b−1）"
            "＋ 未被送用户也必须推进一槽（与 HOLD/PROBE 一致）",
            worst <= 1e-12, {"max_rel": worst, "rows": rows})

    # ★ W0c：`legacy` 的活体对照 —— 旧行为（未送用户不推进）**必须**在这里 FAIL
    pol_l = make(cfg, arm="probe", h_plan=1, m_samples=1, tail="dp",
                 dp_phat=np.asarray([0.50, 0.50], dtype=float),
                 send_other_advance="legacy")
    pol_l._wire_dp(None)
    st2 = Pmodule._St(xh0, P0, mh0, vh0, xh0, mh0, bud0.copy())
    rels = []
    for act in (Action("SEND", 0), Action("SEND", 1)):
        got = pol_l._expect_after(st2, act, steps_after)
        need = 1.0
        bud_after = np.maximum(bud0 - need, 0.0)
        i = int(act.user)
        gmeas = P0[:, i] / (P0[:, i] + sm2)
        P_del = a2[:, i] * ((1.0 - gmeas) * P0[:, i]) + sw2[:, i]
        p_i = pol_l._p_hat(mh0, vh0)[:, i]
        P2 = (a2 * P0 + sw2).copy()
        P2[:, i] = p_i * P_del + (1.0 - p_i) * (a2[:, i] * P0[:, i] + sw2[:, i])
        want = pol_l._terminal(P2, pol_l.alpha[None, :] * xh0, steps_after, bud=bud_after)
        rels.append(float(np.max(np.abs(got - want) / np.maximum(np.abs(want), 1e-12))))
    g.check("W0c 阴性对照：send_other_advance='legacy' ⇒ 同一对账必须 FAIL（相对差 >1e-3）"
            "（否则 W0b 无分辨力）",
            min(rels) > 1e-3, {"legacy_rel": rels})


# ------------------------------------------------------------------ W1 / W1b
def _budget_sweep(pol, steps: int, P0: float) -> dict:
    P = np.asarray([[P0, P0]], dtype=float)
    xh = np.zeros((1, 2), dtype=float)
    vals = []
    for b in range(steps + 1):
        vals.append(float(pol._terminal(P, xh, steps, np.asarray([float(b)]))[0]))
    v = np.asarray(vals)
    d = np.diff(v)                      # d[b] = V(b+1) − V(b)，"多花一帧"的**收益**（应为负）
    n_up = int(np.sum(d > 1e-12))
    n_flat = int(np.sum(np.abs(d) <= 1e-12))
    return {"steps": steps, "P0": P0, "n_pairs": int(d.size), "n_up": n_up,
            "n_flat": n_flat, "frac_flat": n_flat / max(d.size, 1),
            "example_flat_b": [int(b) for b in np.arange(d.size)[np.abs(d) <= 1e-12][:12]],
            "example_d_b1_2": [float(x) for x in d[:4]]}


def w1_budget_monotone(g: Gate) -> None:
    cfg = _cfg()
    pol = make(cfg, arm="probe", h_plan=1, m_samples=1, tail="dp",
               dp_phat=np.asarray([0.60, 0.45], dtype=float))
    pol._wire_dp(None)
    r = _budget_sweep(pol, 128, 0.01)
    g.check("W1（链上 G8d[dp]）值函数对预算严格递减：上升 0 对、平坦 ≤10%"
            "（steps=128、P0=0.01、p̂=(0.60,0.45)）",
            r["n_up"] == 0 and r["frac_flat"] <= 0.10, r)


def w1b_live_control(g: Gate) -> None:
    cfg = _cfg()
    pol = make(cfg, arm="probe", h_plan=1, m_samples=1, tail="dp",
               dp_phat=np.asarray([0.60, 0.45], dtype=float), dp_b_cap=1)
    pol._wire_dp(None)
    r = _budget_sweep(pol, 128, 0.01)
    r["note"] = ("b_cap=1 ⇒ nb_max=2 ⇒ value() 把 b 夹到 0/1 ⇒ 预算维应几乎全平；"
                 "这是**活体**接线错误（不是改阈值）")
    g.check("W1b 阴性对照：b_cap=1（饱和律/预算轴夹错）⇒ 平坦比例 >90%"
            "（否则 W1 的 ≤10% 无分辨力）",
            r["frac_flat"] > 0.90, r)


# ------------------------------------------------------------------ W2（诊断）
def price_recon() -> dict:
    """价格对账：解析尾部 vs DP 尾部，同点同 `τ` 同 `b`。

    ★ 符号约定（v1.7.2 更正）：`DPTail.price()` 返回 `V(b) − V(b−1)`，**预算越多代价越低**
      ⇒ 这个量是**负的**。这里统一报**边际价值** `mV = V(b−1) − V(b) > 0`，
      解析侧同口径 `q·(ΣP(b−1) − ΣP(b))`。两套尾部都按"多一帧值多少"报，避免符号打架。
    """
    out = {}
    for tag, cfg in (("bounded_plant", _cfg()),
                     ("r0_A_gt_1", C.CWConfig())):
        p = cfg.plant
        q = float(p.q_cost)
        sm2 = float(p.sigma_meas) ** 2
        tau, P0, b = W2_TAU, W2_P0, W2_B

        polp = make(cfg, arm="probe", h_plan=1, m_samples=1, tail="paced")
        Pm = np.asarray([[P0, P0]], dtype=float)
        s_b = float(polp._paced_sumP(Pm, tau, np.asarray([float(b)]))[0][0])
        s_b1 = float(polp._paced_sumP(Pm, tau, np.asarray([float(b - 1)]))[0][0])
        mv_an = q * (s_b1 - s_b)

        dp = get_dp(cfg, np.asarray([0.5, 0.5]), b_cap=int(cfg.protocol.horizon))
        mv_dp = -float(dp.price(P0, tau, b))          # 取正：多一帧的价值

        gmeas = P0 / (P0 + sm2)
        benefit_imm = q * 0.5 * gmeas * P0
        a2 = np.asarray(p.A, dtype=float) ** 2
        sw2 = np.asarray(p.sigma_w, dtype=float) ** 2
        P_del = a2 * (1.0 - gmeas) * P0 + sw2
        v_now = float(dp.value(Pm, np.asarray([b]), tau)[0])
        v_del = float(dp.value(P_del[None, :], np.asarray([b]), tau)[0])

        out[tag] = {
            "working_point": {"tau": tau, "P": P0, "b": b, "p_hat": [0.5, 0.5]},
            "marginal_value_analytic": mv_an,
            "marginal_value_dp": mv_dp,
            "ratio_analytic_over_dp": (mv_an / mv_dp) if mv_dp else float("inf"),
            "benefit_imm": benefit_imm,
            "benefit_deliv_dp": v_now - v_del,
            "marginal_value_dp_at_b1": -float(dp.price(P0, tau, 1)),
            "marginal_value_dp_at_bTau": -float(dp.price(P0, tau, tau)),
            "price_dp_over_benefit_imm": (mv_dp / benefit_imm) if benefit_imm else float("inf"),
        }
    out["rule"] = ("ratio = 解析边际价值 / DP 边际价值；≥2 ⇒ 解析高估 ≥2×（G8a 失败可归因算术）；"
                   "0.5–2 ⇒ 同量级（推迟是结构性的）；<0.5 ⇒ 解析低估 ≥2×。"
                   "★ DP 自身的可采纳性由 W4 单独判：W4 FAIL 时，ratio 只能当**两个都不可信**的对照读。")
    return out


# ------------------------------------------------------------------ W3（诊断）
def phat_freeze_cost() -> dict:
    """`p̂` 冻结在 episode 起点（先验 0.5）的代价：与槽初 `p̂` 相比差多少。"""
    cfg = _cfg()
    pol = make(cfg, arm="probe", h_plan=1, m_samples=1, tail="dp")
    tau, P0, b = W2_TAU, W2_P0, W2_B
    prior = float(get_dp(cfg, np.asarray([0.5, 0.5]),
                         b_cap=int(cfg.protocol.horizon)).price(P0, tau, b))
    states = {"prior(0,1)": (0.0, 1.0), "mh-0.5": (-0.5, 1.0),
              "mh+0.5": (0.5, 0.36), "mh+1.0/vh0.25": (1.0, 0.25)}
    rows = []
    for tag, (mh, vh) in states.items():
        ph = pol._p_hat(np.asarray([[mh]]), np.asarray([[vh]]))[0]
        ph = np.asarray([float(ph[0]), float(ph[0])])       # 两用户同值：只测 p̂ 的影响
        pr = float(get_dp(cfg, ph, b_cap=int(cfg.protocol.horizon)).price(P0, tau, b))
        rows.append({"tag": tag, "m_h": mh, "v_h": vh, "p_hat": float(ph[0]),
                     "price": pr, "rel_vs_prior": (pr - prior) / prior if prior else None})
    worst = max(abs(r["rel_vs_prior"]) for r in rows if r["rel_vs_prior"] is not None)
    return {"tau": tau, "P": P0, "b": b, "price_prior": prior, "rows": rows,
            "max_abs_rel_vs_prior": worst,
            "rule": ">30% ⇒ 预注册 §5.4 命中：p̂ 冻结不可接受，只报 W2 不报面板"}


# ------------------------------------------------------------------ W4
def _one_frame_ub(cfg: C.CWConfig, P0: float, phat: np.ndarray, tau: int) -> float:
    """`b=1` 时"多一帧"的**闭式**期望收益（= 精确值，不是松上界）。

    只用一帧 ⇒ 至多一次送达，且只影响一个用户。在时刻 `k` 送达用户 `i` 后，
    它与"不送"的 `P` 之差为 `g_i·P_i^hold(k)`，之后按 `A_i^{2j}` 衰减 ⇒

        saving(i,k) = q·p̂_i·g_i·P_i^hold(k)·Σ_{j=0..τ−1−k} A_i^{2j}

    因为代价对 `P` 线性可加、且 `V(·,0)` 只有一条路径（`HOLD`），
    `V(τ,0) − V(τ,1)` **恰好**等于 `max_{i,k} saving(i,k)`（网格插值误差除外）。
    """
    p = cfg.plant
    q = float(p.q_cost)
    A2 = np.asarray(p.A, dtype=float) ** 2
    sw2 = np.asarray(p.sigma_w, dtype=float) ** 2
    sm2 = float(p.sigma_meas) ** 2
    best = 0.0
    for i in range(A2.size):
        a2, s2 = float(A2[i]), float(sw2[i])
        for k in range(int(tau)):
            Pk = (a2 ** k) * P0 + (s2 * (a2 ** k - 1.0) / (a2 - 1.0)
                                   if abs(a2 - 1.0) > 1e-12 else s2 * k)
            geo = ((a2 ** (tau - k) - 1.0) / (a2 - 1.0)
                   if abs(a2 - 1.0) > 1e-12 else float(tau - k))
            g = Pk / (Pk + sm2)
            best = max(best, q * float(phat[i]) * g * Pk * geo)
    return float(best)


def w4_admissibility(g: Gate) -> None:
    """★ 可采纳性证书：DP 的"一帧价格"不得超过闭式值。

    ★ 为什么必须有这条：`W1` 只检查**平滑**（不上升、不平坦），它**看不见尺度**。
      R1.6.1 的 `G8d` 也是这个盲点。尺度错误只能拿已知解析量来代验（R12：
      判据写完先拿已知答案代验量级）。
    """
    from .plan_dp import DPTail

    cfg = _cfg()
    P0 = 0.01
    phat = np.asarray([0.5, 0.5], dtype=float)
    dp = DPTail(cfg, phat, b_cap=4, tau_max=255).build()
    Pm = np.asarray([[P0, P0]], dtype=float)
    rows, worst = [], 0.0
    for tau in (8, 16, 32, 64, 128, 255):
        v0 = float(dp.value(Pm, np.asarray([0]), tau)[0])
        v1 = float(dp.value(Pm, np.asarray([1]), tau)[0])
        ub = _one_frame_ub(cfg, P0, phat, tau)
        over = (v0 - v1) / ub if ub > 0 else float("inf")
        worst = max(worst, over)
        rows.append({"tau": tau, "V_b0": v0, "V_b1": v1, "drop": v0 - v1,
                     "closed_form": ub, "overshoot": over})
    g.check("W4 可采纳性：DP 的 b=1 边际收益 ≤ 闭式值（比值 ≤1.00）"
            "—— 超了就是插值过度乐观（把一帧定价太高）",
            worst <= 1.0,
            {"P0": P0, "p_hat": phat.tolist(), "max_overshoot": worst, "rows": rows,
             "note": ("闭式值 = max_{i,k} q·p̂_i·g_i·P_i(k)·Σ_j A^{2j}；"
                      "b=1 时它**等于**精确边际收益，所以 >1 必是实现/插值的乐观偏差")})


def run_wire_gates() -> Gate:
    g = Gate()
    w0_interface(g)
    w0b_budget_plumbing(g)
    w1_budget_monotone(g)
    w1b_live_control(g)
    w4_admissibility(g)
    return g