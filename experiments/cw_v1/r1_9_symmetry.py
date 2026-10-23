# -*- coding: utf-8 -*-
"""CW-R1.9 **零成本**诊断：`dp` 为什么"一律偏 u0" —— 值函数 vs 策略层。

预注册：`docs/research/CW-R1.9-策略层对称-预注册-v0.1.md`（写于本轮任何新计算之前）。

四条探针（`run_episode` 调用 **0** 次）
--------------------------------------
T1 根动作值向量的**置换等变**闸：对称 plant + **显式不对称**状态 + 面板用户列同步置换，
   要求 `V(π(S), π(D))[π(a)] == V(S, D)[a]` 逐位（rel ≤ 1e-12）。
T2 价值层边际收益：真实 plant（`σ_w=(0.02,0.03)`、`grid="r173"`，与 R1.8 的 `dp16` 同参数）上
   算 `benefit_i = V(P,b) − V(deliv_i(P), b−1)`，看值函数**认为**哪个用户更值得服务。
T3 两条活体负对照：①`legacy_swap` 必须让 T1 FAIL；②只换状态不换面板必须出现差。
T4 接线读数：`_advance(SEND(0))` 后未送用户的 `P` 是否推进（只记录，不判 PASS/FAIL）。

★ 被测物（`plan_ref` / `plan_dp` / `world` / `runner`）**零改动**；本文件只读它们。
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import config as C
from . import plan_ref as Pmodule
from .plan_dp import cache_stats, get_dp
from .world import Action, SchedulerView

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results" / "cw_v1"

#: 与 R1.8 面板完全一致的策略参数（`r1_8_mech.py:54-61`）
RHO = 0.98
H_PLAN = 5
M_MAIN = 16
RATIO = 0.5
TOL = 1e-12

#: T1 的两个**显式不对称**状态（P0≠P1 即"对称已被破坏"，R1.7.1 迁移纪律）
STATES = (
    {"tag": "S_main", "t": 8, "bud": 128, "P": (0.030, 0.012),
     "x": (0.30, -0.20), "mh": (0.40, -0.30), "vh": (0.80, 0.60)},
    {"tag": "S_lowbud", "t": 250, "bud": 2, "P": (0.050, 0.020),
     "x": (-0.15, 0.25), "mh": (-0.20, 0.35), "vh": (0.55, 0.90)},
)

#: T2 的 6 个判点（预注册 §2 T2 逐字）
T2_POINTS = ((0.010, 0.010, 205), (0.030, 0.030, 205), (0.060, 0.060, 205),
             (0.030, 0.060, 205), (0.060, 0.030, 205), (0.030, 0.060, 128))

_PERM = {"SEND(0)": "SEND(1)", "SEND(1)": "SEND(0)",
         "PROBE(0)": "PROBE(1)", "PROBE(1)": "PROBE(0)", "HOLD": "HOLD"}


# ---------------------------------------------------------------- 造视图
def _pub(st: dict) -> dict:
    """手搓一份**公开快照**（键与 `CWWorld.view()` 逐字一致）。"""
    n = len(st["P"])
    tel = []
    for i in range(n):
        tel.append({"i": i, "x_hat": float(st["x"][i]), "P_hat": float(st["P"][i]),
                    "stamp": -1, "task_aoi": -1,
                    "h_mean": float(st["mh"][i]), "h_var": float(st["vh"][i]),
                    "csi_stamp": -1, "csi_aoi": -1})
    return {"t": int(st["t"]), "n_users": n, "budget_left": int(st["bud"]),
            "budget_total": 128,
            "prior": {"x_mean": 0.0, "x_var": 1.0, "h_log_mean": 0.0, "h_log_var": 1.0},
            "telemetry": tel, "history": ()}


def _swapped(xs):
    return (xs[1], xs[0])


def _perm_state(st: dict) -> dict:
    return {"tag": st["tag"] + "|π", "t": st["t"], "bud": st["bud"],
            "P": _swapped(st["P"]), "x": _swapped(st["x"]),
            "mh": _swapped(st["mh"]), "vh": _swapped(st["vh"])}


# ---------------------------------------------------------------- 面板置换
class _PermPanel(Pmodule._Draws):
    """同一份随机数，把**用户列**交换（用于置换等变闸）。"""

    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)
        for f in ("udeliv", "eps_meas", "eps_csi", "s_h", "w"):
            setattr(self, f, np.ascontiguousarray(getattr(self, f)[:, :, ::-1]))
        self.init_x = np.ascontiguousarray(self.init_x[:, ::-1])
        self.init_h = np.ascontiguousarray(self.init_h[:, ::-1])


@contextlib.contextmanager
def _panel_patch(cls):
    orig = Pmodule._Draws
    Pmodule._Draws = cls
    try:
        yield
    finally:
        Pmodule._Draws = orig
    assert Pmodule._Draws is orig, "面板补丁未还原"


# ---------------------------------------------------------------- 值捕获
class _Cap(Pmodule.PlanRef):
    """只**观测**根动作值：`_pick` 在 `__call__` 里最后一次被调用即根节点。"""

    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)
        self.picks: list[dict] = []

    def _pick(self, values, budget_left):          # noqa: ANN001
        self.picks.append({repr(a): float(values[a]) for a in values})
        return super()._pick(values, budget_left)

    @property
    def root(self) -> dict:
        return self.picks[-1]


def _mk(cfg, seed: int, tail: str | None) -> _Cap:
    kw = dict(h_plan=H_PLAN, m_samples=M_MAIN, seed=seed, rho=RHO, pacing=True)
    if tail == "dp":
        kw.update(tail="dp", dp_grid="r173")
    return _Cap(cfg, use_probe=True, sample_hidden=True, **kw)


def _rel(a: float, b: float) -> float:
    if not (np.isfinite(a) and np.isfinite(b)):
        return 0.0 if (a == b) else float("inf")
    return abs(a - b) / max(1e-12, max(abs(a), abs(b)))


def _compare(root: dict, root_perm: dict) -> dict:
    """对每个动作比 `root[a]` 与 `root_perm[π(a)]`；返回逐动作相对差。"""
    rows = {}
    for a, v in root.items():
        pa = _PERM.get(a, a)
        pv = root_perm.get(pa, float("nan"))
        rows[a] = {"self": v, "perm_of": pa, "perm_val": pv, "rel": _rel(v, pv)}
    worst = max(rows.values(), key=lambda r: r["rel"])
    return {"rows": rows, "max_rel": float(worst["rel"]),
            "worst_action": next(k for k, r in rows.items() if r is worst)}


# ---------------------------------------------------------------- T1
def t1_equivariance(cfg_sym, tail: str, seed: int) -> dict:
    out = {"tail": tail, "states": []}
    for st in STATES:
        a = _mk(cfg_sym, seed, tail)
        act_a = a(SchedulerView(_pub(st)))
        b = _mk(cfg_sym, seed, tail)
        with _panel_patch(_PermPanel):
            act_b = b(SchedulerView(_pub(_perm_state(st))))
        cmp_ = _compare(a.root, b.root)
        # 负对照 A：只换状态、**不**换面板 ⇒ 必须出现差（否则闸对面板不敏感）
        c = _mk(cfg_sym, seed, tail)
        act_c = c(SchedulerView(_pub(_perm_state(st))))
        cmp_noPanel = _compare(a.root, c.root)
        out["states"].append({
            "tag": st["tag"], "budget_left": st["bud"], "t": st["t"],
            "cmp": cmp_, "action": repr(act_a), "action_perm": repr(act_b),
            "action_perm_ok": bool(repr(act_b) == _PERM.get(repr(act_a), "")),
            "control_no_panel": {"max_rel": cmp_noPanel["max_rel"],
                                 "action": repr(act_c)},
        })
    out["max_rel"] = max(s["cmp"]["max_rel"] for s in out["states"])
    out["equivariant"] = bool(out["max_rel"] <= TOL)
    return out


def t1_control_legacy(cfg_sym, seed: int) -> dict:
    """活体负对照：`legacy_swap`（R1.7.1 旧缺陷）必须让 T1 FAIL。"""
    st = STATES[0]
    a = _Cap(cfg_sym, use_probe=True, sample_hidden=True, h_plan=H_PLAN, m_samples=M_MAIN,
             seed=seed, rho=RHO, pacing=True, tail="dp", dp_grid="r173",
             dp_axis_mode="legacy_swap")
    a(SchedulerView(_pub(st)))
    b = _Cap(cfg_sym, use_probe=True, sample_hidden=True, h_plan=H_PLAN, m_samples=M_MAIN,
             seed=seed, rho=RHO, pacing=True, tail="dp", dp_grid="r173",
             dp_axis_mode="legacy_swap")
    with _panel_patch(_PermPanel):
        b(SchedulerView(_pub(_perm_state(st))))
    cmp_ = _compare(a.root, b.root)
    return {"max_rel": cmp_["max_rel"], "fails_as_expected": bool(cmp_["max_rel"] > TOL),
            "cmp": cmp_}


# ---------------------------------------------------------------- T2
def _benefit(dp, cfg, P: np.ndarray, b: int, i: int, tau: int) -> float:
    """`V(P,b) − V(deliv_i(P), b−1)`：发一帧给 i 的收益（含预算机会成本）。"""
    p = cfg.plant
    A = np.asarray(p.A, dtype=float)
    sw2 = np.asarray(p.sigma_w, dtype=float) ** 2
    sm2 = float(p.sigma_meas) ** 2
    Pn = A ** 2 * P + sw2                      # 未送用户：不刷新、老化一槽
    g = P[i] / (P[i] + sm2)
    Pn[i] = A[i] ** 2 * (1.0 - g) * P[i] + sw2[i]   # 被送用户：送达 + 量测更新 + 老化
    v_hi = float(dp.value(np.array([P]), [b], tau)[0])
    v_lo = float(dp.value(np.array([Pn]), [b - 1], tau)[0])
    return v_hi - v_lo


def t2_marginal(cfg, p_hat, grid: str, tag: str) -> dict:
    T = int(cfg.protocol.horizon)
    dp = get_dp(cfg, p_hat, b_cap=T, axis_mode="user_axis", grid=grid)
    rows = []
    for p0, p1, b in T2_POINTS:
        P = np.array([p0, p1], dtype=float)
        b0 = _benefit(dp, cfg, P, b, 0, T)
        b1 = _benefit(dp, cfg, P, b, 1, T)
        rows.append({"P0": p0, "P1": p1, "b": b, "benefit_0": b0, "benefit_1": b1,
                     "diff_1_minus_0": b1 - b0, "prefers": ("u1" if b1 > b0 else "u0")})
    return {"tag": tag, "grid": grid, "sigma_w": list(cfg.plant.sigma_w),
            "p_hat": [float(x) for x in p_hat], "rows": rows,
            "n_prefers_u1": sum(1 for r in rows if r["prefers"] == "u1"),
            "dp_diag": dp.diag()}


def t2a_witness() -> dict:
    """见证闸：对称 plant 上 `_benefit` 必须对换用户不变。

    ★ 首跑（03:1x）把**两个非对称点**（`(0.030,0.060)` / `(0.060,0.030)`）也算进
      "`benefit_0` 必须等于 `benefit_1`" ⇒ 必然 FAIL（两点本来就该不等）。
      这是**判据写错**，不是被测物有问题（#39 同族）。改写为两条**正确**的见证：
        ① `P0 == P1` 的三点：`benefit_0 == benefit_1`；
        ② 交叉对称：`benefit_0(P=(a,b)) == benefit_1(P=(b,a))`（对全部 5 个不同 P 对）。
    """
    cfg = C.CWConfig(plant=C.bounded_plant(sigma_w=(0.025, 0.025)))
    out = t2_marginal(cfg, np.array([0.5, 0.5]), "r173", "T2a_sym_witness")
    sym = [r for r in out["rows"] if r["P0"] == r["P1"]]
    sym_diff = max(abs(r["diff_1_minus_0"]) for r in sym)
    by_p = {(r["P0"], r["P1"], r["b"]): r for r in out["rows"]}
    cross = []
    for (a, b, bb), r in by_p.items():
        other = by_p.get((b, a, bb))
        if other is None:
            continue
        cross.append({"pair": [a, b, bb],
                      "benefit_0_at_ab": r["benefit_0"],
                      "benefit_1_at_ba": other["benefit_1"],
                      "abs_diff": abs(r["benefit_0"] - other["benefit_1"])})
    cross_diff = max((c["abs_diff"] for c in cross), default=0.0)
    out.update({"sym_rows": sym, "sym_max_abs_diff": sym_diff,
                "cross_rows": cross, "cross_max_abs_diff": cross_diff,
                "max_abs_diff": max(sym_diff, cross_diff)})
    out["ok"] = bool(out["max_abs_diff"] <= TOL)
    return out


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# ---------------------------------------------------------------- T4
def t4_wiring(cfg) -> dict:
    """`_advance(SEND(0))` 后未送用户是否老化（只记录）。"""
    pol = _mk(cfg, 20261022, None)
    st = STATES[0]
    d = Pmodule._Draws(np.random.default_rng(7), 2, M_MAIN, 2)
    s = pol._init_state(SchedulerView(_pub(st)), d)
    p_before = s.P[0].copy()
    pol._advance(s, Action("SEND", 0), k=0, d=d)
    A = np.asarray(cfg.plant.A, dtype=float)
    sw2 = np.asarray(cfg.plant.sigma_w, dtype=float) ** 2
    expect_f = A ** 2 * p_before + sw2
    return {"P_before": [float(x) for x in p_before],
            "P_after_send0": [float(x) for x in s.P[0]],
            "A2P_plus_sw2": [float(x) for x in expect_f],
            "user1_advanced": bool(abs(float(s.P[0, 1]) - float(expect_f[1])) <= 1e-12),
            "user1_frozen": bool(abs(float(s.P[0, 1]) - float(p_before[1])) <= 1e-12)}


# ---------------------------------------------------------------- T5（exploratory）
#: 预注册 §2.5（03:2x 修订 1 之后追加；加它时**未读** `t2.json` 的数值）
T5_STATES = ((0.01, 0.02), (0.02, 0.01), (0.02, 0.04), (0.04, 0.02),
             (0.01, 0.04), (0.04, 0.01))
T5_SEEDS = (2001, 2002, 2003)


def t5_rollout_vs_table(cfg, dp) -> dict:
    """**只报不判**：rollout 的根节点偏好 vs 值表的 `benefit` 偏好，在同一 P 上对照。"""
    T = int(cfg.protocol.horizon)
    rows = []
    for p0, p1 in T5_STATES:
        st = {"tag": f"T5_{p0}_{p1}", "t": 0, "bud": 128, "P": (p0, p1),
              "x": (0.0, 0.0), "mh": (0.0, 0.0), "vh": (1.0, 1.0)}
        per = []
        for sd in T5_SEEDS:
            pol = _mk(cfg, sd, "dp")
            act = pol(SchedulerView(_pub(st)))
            v0, v1 = pol.root["SEND(0)"], pol.root["SEND(1)"]
            dlt = v1 - v0                      # >0 ⇒ SEND(0) 更小 ⇒ 偏好 u0
            per.append({"seed": sd, "SEND0": v0, "SEND1": v1, "diff_1_minus_0": dlt,
                        "chosen": repr(act),
                        "pref": ("u0" if dlt > TOL else ("u1" if dlt < -TOL else "tie"))})
        P = np.array([p0, p1], dtype=float)
        b0 = _benefit(dp, cfg, P, 128, 0, T)
        b1 = _benefit(dp, cfg, P, 128, 1, T)
        tp = "u1" if b1 > b0 else ("u0" if b0 > b1 else "tie")
        rows.append({"P0": p0, "P1": p1, "per_seed": per,
                     "n_pref_u0": sum(1 for r in per if r["pref"] == "u0"),
                     "n_pref_u1": sum(1 for r in per if r["pref"] == "u1"),
                     "n_tie": sum(1 for r in per if r["pref"] == "tie"),
                     "table_benefit_0": b0, "table_benefit_1": b1,
                     "table_pref": tp,
                     "rollout_majority": ("u0" if sum(1 for r in per if r["pref"] == "u0") > 1.5
                                          else ("u1" if sum(1 for r in per if r["pref"] == "u1") > 1.5
                                                else "split"))})
    return {"states": rows, "note": "report-only（预注册 §2.5），不进判词"}


# ---------------------------------------------------------------- main
def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def main() -> int:
    t0 = time.perf_counter()
    gates: dict[str, dict] = {}

    def gate(name: str, ok: bool, detail: dict) -> None:
        gates[name] = {"pass": bool(ok), **detail}
        print(f"[{'PASS' if ok else 'FAIL'}] {name}  "
              f"{json.dumps(detail, ensure_ascii=False)[:220]}")

    cfg_sym = C.CWConfig(plant=C.bounded_plant(A=(1.0, 1.0), sigma_w=(0.025, 0.025)))
    cfg_real = C.CWConfig(plant=C.bounded_plant())
    print(f"[INFO] sigma_w sym={cfg_sym.plant.sigma_w} real={cfg_real.plant.sigma_w} "
          f"T={cfg_real.protocol.horizon} budget={cfg_real.protocol.budget(RATIO)}")

    ev: dict = {}

    # ---- T1
    ev["t1"] = {}
    for tail in ("paced", "dp"):
        r = t1_equivariance(cfg_sym, tail, 20261024)
        ev["t1"][tail] = r
        gate(f"T1_eq_{tail}", r["max_rel"] <= TOL,
             {"max_rel": r["max_rel"],
              "per_state": [{"tag": s["tag"], "max_rel": s["cmp"]["max_rel"],
                             "act": s["action"], "act_perm": s["action_perm"],
                             "perm_action_ok": s["action_perm_ok"],
                             "no_panel_rel": s["control_no_panel"]["max_rel"]}
                            for s in r["states"]]})
        # 负对照 A：只换状态不换面板 ⇒ 应有差
        np_rel = max(s["control_no_panel"]["max_rel"] for s in r["states"])
        gate(f"T1_ctl_stateOnly_{tail}", np_rel > TOL, {"max_rel": np_rel})

    # ---- T3 活体负对照
    ctl = t1_control_legacy(cfg_sym, 20261024)
    ev["t3_legacy"] = ctl
    gate("T3_ctl_legacy_swap_fails", ctl["fails_as_expected"],
         {"max_rel": ctl["max_rel"]})

    # ---- T2a 见证闸
    w = t2a_witness()
    ev["t2a"] = w
    gate("T2a_sym_witness", w["ok"], {"sym_max": w["sym_max_abs_diff"], "cross_max": w["cross_max_abs_diff"],
                                      "dp": w["dp_diag"]})

    # ---- T2 主：真实 plant
    m = t2_marginal(cfg_real, np.array([0.5, 0.5]), "r173", "T2_real_r173")
    ev["t2"] = m
    gate("T2_build_clean", m["dp_diag"]["n_clip"] == 0 and m["dp_diag"]["n_b_clip"] == 0,
         {"n_clip": m["dp_diag"]["n_clip"], "n_b_clip": m["dp_diag"]["n_b_clip"]})

    # ---- T4
    ev["t4"] = t4_wiring(cfg_real)

    # ---- T5（exploratory，report-only）
    dp_real = get_dp(cfg_real, np.array([0.5, 0.5]), b_cap=int(cfg_real.protocol.horizon),
                     axis_mode="user_axis", grid="r173")
    ev["t5"] = t5_rollout_vs_table(cfg_real, dp_real)

    n_fail = sum(0 if g["pass"] else 1 for g in gates.values())
    verdict = _verdict(gates, ev)
    out = {"kind": "CW-R1.9", "verdict": verdict, "n_fail": n_fail, "gates": gates,
           "t1": ev["t1"], "t2a": ev["t2a"], "t2": ev["t2"], "t3_legacy": ev["t3_legacy"],
           "t4": ev["t4"], "t5": ev["t5"], "cache": cache_stats(),
           "elapsed_s": round(time.perf_counter() - t0, 2),
           "n_run_episode": 0}

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    blob = "".join(_sha(REPO / p) for p in (
        "experiments/cw_v1/plan_ref.py", "experiments/cw_v1/plan_dp.py",
        "experiments/cw_v1/r1_9_symmetry.py",
        "docs/research/CW-R1.9-策略层对称-预注册-v0.1.md"))
    d = RESULTS / f"cw_r1_9_symmetry_{stamp}_{blob[:16]}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "gates.json").write_text(json.dumps(gates, ensure_ascii=False, indent=1, default=_json_default), encoding="utf-8")
    for k, v in ev.items():
        (d / f"{k}.json").write_text(json.dumps(v, ensure_ascii=False, indent=1, default=_json_default), encoding="utf-8")
    (d / "meta.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=_json_default), encoding="utf-8")
    print(f"\n[T2] prefers_u1 {m['n_prefers_u1']}/6")
    for r in m["rows"]:
        print(f"   P=({r['P0']:.3f},{r['P1']:.3f}) b={r['b']}  benefit0={r['benefit_0']:+.6f} "
              f"benefit1={r['benefit_1']:+.6f}  prefers={r['prefers']}")
    print("[T5] rollout vs table（report-only）")
    for r in ev["t5"]["states"]:
        print(f"   P=({r['P0']:.2f},{r['P1']:.2f})  rollout u0/u1/tie="
              f"{r['n_pref_u0']}/{r['n_pref_u1']}/{r['n_tie']}  table={r['table_pref']}")
    print(f"[T4] {ev['t4']}")

    print(f"\n[VERDICT] {verdict}  n_fail={n_fail}  elapsed={out['elapsed_s']}s")
    print(f"[EVIDENCE] {d}")
    return 0 if n_fail == 0 else 1


def _verdict(gates: dict, ev: dict) -> str:
    if not gates["T3_ctl_legacy_swap_fails"]["pass"] or not gates["T2a_sym_witness"]["pass"]:
        return "GATE-NO-RESOLUTION"
    if not gates["T1_ctl_stateOnly_paced"]["pass"] or not gates["T1_ctl_stateOnly_dp"]["pass"]:
        return "GATE-NO-RESOLUTION"
    t1_bad = [k for k in gates if k.startswith("T1_eq_") and not gates[k]["pass"]]
    n_u1 = ev["t2"]["n_prefers_u1"]
    if t1_bad:
        return "ROLLOUT-ASYMMETRIC"
    return "VALUE-PREFERS-U1" if n_u1 == len(T2_POINTS) else (
        "VALUE-PREFERS-U0" if n_u1 == 0 else "VALUE-MIXED")


if __name__ == "__main__":
    raise SystemExit(main())
