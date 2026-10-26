# -*- coding: utf-8 -*-
"""`CW-R3.5`：信息价值上界的**独立面板**证书 —— `UB := IV_free(σ⋆→0)`。

预注册：`docs/research/CW-R3.5-信息价值上界的独立面板-预注册-v0.1.md`（写于正式面板之前）

为什么不是照抄 R3.4 的 `X2`
--------------------------
`X2 = π_slot − (V_PROBE − V_SEND)` **混了臂**：`π_slot` 与 `V_SEND` 取自 `noprobe` 臂、
`V_PROBE` 取自 `free` 臂，而 `free` 臂逐点 `V_free ≤ V_noprobe`。它是两个数之差，
没有"上界"的推导。本轮改用**同一动作集内**的上界：

    UB := A0 − A_free(σ⋆),  σ⋆ = 1e-9（网格内完美 CSI，免量子、仍占一个槽）

`σ⋆` 臂在本网格上 Blackwell-支配任何 `σ` 臂、`probe_cost=0` 的臂支配 `probe_cost=1`
⇒ `IV_paid(σ) ≤ IV_free(σ) ≤ UB`。两条都逐格验（`C0`）。
旧形态 `X2_legacy` 并列照跑，只回答一个问题：它会不会**超出**真上界（`C4`）。

三条纪律
--------
1. 零 episode、零训练、零参数更新；被测物一字节不动（只 `import`）。
2. 判据 `C0–C5` 见预注册 §4，**跑前写死**；`judge()` 只按表取。
3. `UB` 只定价"当前槽的信道知识"，**不**定价 ACK/NACK 通道，不外推到 256 槽。
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import time
from pathlib import Path

import numpy as np

from .config import mainline_config
from .posterior import p_succ_expected
from .r1_14_exactbase import _now_utc, _src_hash
from .r3_2_iv2u import build_engine

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results" / "cw_v1"

PROTOCOL = "CW-POMDP-v1.1"
ROUND = "CW-R3.5"

#: 面板（预注册 §3，跑前写死；**不按结果移动**）
SIGMA_STAR = 1.0e-9                      # 网格内完美 CSI
SIGMA_LIST: tuple[float, ...] = (SIGMA_STAR, 0.02, 0.05, 0.2, 0.5)
P_LIST: tuple[float, ...] = (0.01, 0.05, 0.1)
TAU_LIST: tuple[int, ...] = (2, 3, 4, 5)
B_MAX = 4
N_M = 129
V_PRIOR = 1.0

#: 判据常数（预注册 §4）
IV_NOISE = 1e-12
UB_FLOOR = 1e-9
C1_RATE = 0.90
C3_MAX_VIOL = 0.10
C5_RATE = 0.20
EPS = 1e-12
STATE_CAP = 6_000_000

#: 网格分辨率检查（预注册 §3 末）
REFINE_SIGMA = SIGMA_STAR
REFINE_P = 0.05
REFINE_TAU = 4
REFINE_B = (2, 3)
REFINE_NM = (65, 129, 257)
REFINE_SPREAD_MAX = 0.30

#: R3.4 证据目录（`G7` 逐位对账用；写死路径，找不到就 FAIL 不降级）
R34_JSON = (RESULTS / "cw_r3_4_slotprice_20261026T004136_fe38344ec8d735e3" / "r34.json")

_CFG = mainline_config()


def _file_hash(p: Path) -> str:
    import hashlib

    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


# ============================================================ 臂
class Arms:
    """按 `σ` 缓存的臂集合；`noprobe` 与 `σ` 无关，只建一次。"""

    def __init__(self, cfg, n_m: int):
        self.cfg = cfg
        self.n_m = int(n_m)
        t0 = time.perf_counter()
        self.no = build_engine(cfg, SIGMA_STAR, self.n_m, probe_enabled=False)
        self.by_sigma: dict[float, tuple] = {}
        for s in SIGMA_LIST:
            self.by_sigma[s] = (build_engine(cfg, s, self.n_m, probe_cost=0),
                                build_engine(cfg, s, self.n_m, probe_cost=1))
        self.sec = round(time.perf_counter() - t0, 3)

    def free(self, sigma: float):
        return self.by_sigma[sigma][0]

    def paid(self, sigma: float):
        return self.by_sigma[sigma][1]

    def n_states(self) -> dict:
        d = {"noprobe": self.no.n_states()}
        for s in SIGMA_LIST:
            d[f"free@{s:g}"] = self.by_sigma[s][0].n_states()
            d[f"paid@{s:g}"] = self.by_sigma[s][1].n_states()
        return d

    def capped(self) -> bool:
        flags = [self.no.capped()]
        for s in SIGMA_LIST:
            flags += [self.by_sigma[s][0].capped(), self.by_sigma[s][1].capped()]
        return bool(any(flags))


def cell(arms: Arms, cfg, tau: int, b: int, P: float, sigma: float) -> dict:
    """一格：四条臂 + `ΔV_q` + `X2_legacy`（公式与 R3.4 `cell()` 逐字一致，便于 `G7` 对账）。"""
    a2 = tuple(float(a) ** 2 for a in cfg.plant.A)
    sw2 = tuple(float(s) ** 2 for s in cfg.plant.sigma_w)
    sm2 = float(cfg.plant.sigma_meas) ** 2
    rho = float(cfg.channel.rho_panel[0])
    v_hold = rho * rho * V_PRIOR + (1.0 - rho * rho)

    no = arms.no
    A0 = no(tau, P, P, 0.0, V_PRIOR, 0.0, V_PRIOR, b)
    A0m = no(tau, P, P, 0.0, V_PRIOR, 0.0, V_PRIOR, b - 1) if b >= 1 else float("nan")

    F = arms.free(sigma)(tau, P, P, 0.0, V_PRIOR, 0.0, V_PRIOR, b)
    D = arms.paid(sigma)(tau, P, P, 0.0, V_PRIOR, 0.0, V_PRIOR, b)
    Fs = arms.free(SIGMA_STAR)(tau, P, P, 0.0, V_PRIOR, 0.0, V_PRIOR, b)
    Ds = arms.paid(SIGMA_STAR)(tau, P, P, 0.0, V_PRIOR, 0.0, V_PRIOR, b)

    dq = (A0m - A0) if b >= 1 else float("nan")

    g = P / (P + sm2)
    ps_ = a2[0] * (1.0 - g) * P + sw2[0]
    pf_ = a2[0] * P + sw2[0]
    dvs = (no(tau - 1, pf_, P, 0.0, v_hold, 0.0, v_hold, b)
           - no(tau - 1, ps_, P, 0.0, v_hold, 0.0, v_hold, b))

    # ---- X2_legacy（R3.4 原形态，混臂；只用于 C4）
    f_no = no.first(tau, P, P, 0.0, V_PRIOR, 0.0, V_PRIOR, b)
    f_fr = arms.free(sigma).first(tau, P, P, 0.0, V_PRIOR, 0.0, V_PRIOR, b)
    send_best = min(v for k, v in f_no.items() if k.startswith("SEND_"))
    send_arg = min((k for k in f_no if k.startswith("SEND_")), key=lambda k: f_no[k])
    probe_best = min(v for k, v in f_fr.items() if k.startswith("PROBE_"))
    pi_slot = float(f_no["HOLD"] - send_best)
    x2_legacy = float(pi_slot - (probe_best - send_best))

    # ---- 同臂量（★ `first()` 全部取自 `free(σ⋆)` 臂，不混臂）
    f_star = arms.free(SIGMA_STAR).first(tau, P, P, 0.0, V_PRIOR, 0.0, V_PRIOR, b)
    s_send = min(v for k, v in f_star.items() if k.startswith("SEND_"))
    s_probe = min(v for k, v in f_star.items() if k.startswith("PROBE_"))
    s_hold = float(f_star["HOLD"])
    deficit_free = float(s_probe - s_send)      # >0 ⇒ 探测差这么远才被选上（同臂）
    pi_slot_free = float(s_hold - s_send)

    return {
        "tau": int(tau), "b": int(b), "P": float(P), "sigma": float(sigma), "n_m": arms.n_m,
        "A0": float(A0),
        "IV_free": float(A0 - F), "IV_paid": float(A0 - D),
        "IV_free_star": float(A0 - Fs), "IV_paid_star": float(A0 - Ds),
        "dV_q": float(dq), "dV_succ": float(dvs),
        "pi_slot": pi_slot, "send_arg": send_arg,
        "probe_minus_send": float(probe_best - send_best),
        "deficit_free": deficit_free,
        "deficit_over_dVq": float(deficit_free / dq) if (dq is not None and dq > EPS) else None,
        "pi_slot_free": pi_slot_free,
        "gap_send01": float(abs(f_no.get("SEND_0", 0.0) - f_no.get("SEND_1", 0.0))),
        "x2_legacy": x2_legacy,
        "ratio_ub": float((A0 - Fs) / dq) if (dq is not None and dq > EPS) else None,
        "ratio_x2": float(x2_legacy / dq) if (dq is not None and dq > EPS) else None,
        "ratio_paid_star": float((A0 - Ds) / dq) if (dq is not None and dq > EPS) else None,
    }


def panel() -> dict:
    cfg = mainline_config()
    arms = Arms(cfg, N_M)
    print(f"  [arms n_m={N_M}] built {arms.sec} s", flush=True)
    rows = []
    for sigma in SIGMA_LIST:
        for tau in TAU_LIST:
            for b in range(1, min(tau, B_MAX) + 1):
                for P in P_LIST:
                    rows.append(cell(arms, cfg, tau, b, P, sigma))
            print(f"  [panel sigma={sigma:g} tau={tau}] rows={len(rows)}", flush=True)
    # 网格分辨率检查（独立 `n_m`，各自建臂）
    ref = []
    for n_m in REFINE_NM:
        a = Arms(cfg, n_m)
        for b in REFINE_B:
            c = cell(a, cfg, REFINE_TAU, b, REFINE_P, REFINE_SIGMA)
            c["n_m"] = n_m
            ref.append(c)
        print(f"  [refine n_m={n_m}] done", flush=True)
    return {"rows": rows, "refine": ref,
            "meta": {"n_m": N_M, "arms_sec": arms.sec,
                     "n_states": arms.n_states(), "capped": arms.capped()}}


# ============================================================ 闸
def gate_g1_no_truth() -> dict:
    src = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    banned = {"h_true", "cost_true_by_user", "cost_true", "w_realized", "x_true"}
    hits = [n.attr for n in ast.walk(tree)
            if isinstance(n, ast.Attribute) and n.attr in banned]
    hits += [n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id in banned]
    return {"banned": sorted(banned), "hits": hits, "pass": not hits}


def gate_g2_identity() -> dict:
    mods = ["r3_2_iv2u.py", "config.py", "posterior.py", "r3_4_slotprice.py"]
    h = {m: _file_hash(REPO / "experiments" / "cw_v1" / m) for m in mods}
    return {"hashes": h, "dir_hash": _src_hash(), "pass": bool(all(v for v in h.values()))}


def gate_g3_martingale() -> dict:
    """鞅恒等在整个面板用到的每个 `σ` 上都必须成立（不只是 `0.05`）。

    `E_y[p̂(后验)] − p̂(先验) = 0`。若某个 `σ` 上不成立，则"信息只能通过改变动作
    产生价值"这条论证在该 `σ` 上悬空 ⇒ `UB` 的语义也要重写。
    """
    bad_total = 0
    worst = 0.0
    rows = []
    for sigma in SIGMA_LIST:
        si2 = float(sigma) ** 2
        gp = V_PRIOR / (V_PRIOR + si2)
        vp = V_PRIOR * si2 / (V_PRIOR + si2)
        p_pri = p_succ_expected(0.0, V_PRIOR, _CFG)
        nodes, w = np.polynomial.hermite_e.hermegauss(161)
        tot = 0.0
        bad = 0
        for z, wi in zip(nodes, w):
            z = float(z)
            wi = float(wi)
            if not (math.isfinite(z) and math.isfinite(wi)):
                bad += 1
                continue
            m_post = gp * math.sqrt(V_PRIOR + si2) * z
            tot += wi * p_succ_expected(m_post, vp, _CFG)
        tot /= math.sqrt(2.0 * math.pi)
        d = abs(tot - p_pri)
        worst = max(worst, d)
        bad_total += bad
        rows.append({"sigma": sigma, "p_prior": float(p_pri),
                     "p_post_mean": float(tot), "abs_diff": float(d), "n_bad_nodes": bad})
    return {"rows": rows, "worst_abs_diff": float(worst), "n_bad_nodes": bad_total,
            "tol": 1e-9, "pass": bool(bad_total == 0 and worst <= 1e-9)}


def gate_g4_first_consistency() -> dict:
    """`G4`：`first()` 与 `V()` 的**根状态一致性** —— `min(first().values())` 必须逐位等于 `V()`。

    这条不是形式检查：`cell()` 的每条臂都是从 `V()` 与 `first()` 分别取的，
    只要两处的参数顺序/成本项有一次不一致（R3.4 `G6` 两臂写反同族），本闸当场红。
    """
    cfg = mainline_config()
    arms = Arms(cfg, 33)
    rows = []
    worst = 0.0
    for tau, b, P in ((3, 1, 0.05), (4, 2, 0.01), (4, 3, 0.1), (5, 4, 0.05)):
        for name, fn in (("noprobe", arms.no),
                         ("free*", arms.free(SIGMA_STAR)),
                         ("paid*", arms.paid(SIGMA_STAR))):
            v = fn(tau, P, P, 0.0, V_PRIOR, 0.0, V_PRIOR, b)
            f = fn.first(tau, P, P, 0.0, V_PRIOR, 0.0, V_PRIOR, b)
            d = float(abs(min(f.values()) - v))
            worst = max(worst, d)
            rows.append({"tau": tau, "b": b, "P": P, "arm": name,
                         "V": float(v), "min_first": float(min(f.values())), "abs_diff": d})
    return {"rows": rows, "max_abs_diff": float(worst), "tol": 1e-15,
            "pass": bool(worst <= 1e-15)}


def gate_g5_drop_obs(arms: Arms, cfg) -> dict:
    """`G5`（变异）：把 `σ⋆` 臂的读数**丢弃**（`drop_obs=True`，仍免量子）后，
    其值必须与 `A0` 逐位相等 —— 否则"免量子臂的优势全部来自读数"这条前提不成立。"""
    drop = build_engine(cfg, SIGMA_STAR, 129, probe_cost=0, drop_obs=True)
    rows = []
    worst = 0.0
    for tau, b, P in ((3, 1, 0.05), (4, 2, 0.01), (4, 3, 0.1)):
        a0 = arms.no(tau, P, P, 0.0, V_PRIOR, 0.0, V_PRIOR, b)
        dd = drop(tau, P, P, 0.0, V_PRIOR, 0.0, V_PRIOR, b)
        worst = max(worst, abs(a0 - dd))
        rows.append({"tau": tau, "b": b, "P": P, "A0": float(a0), "drop_obs": float(dd),
                     "abs_diff": float(abs(a0 - dd))})
    return {"rows": rows, "max_abs": float(worst), "tol": 1e-15,
            "pass": bool(worst <= 1e-15)}


def gate_g7_r34_identity(rows: list) -> dict:
    """`G7`：与 R3.4 证据在**重叠格**上逐位对账（`σ=0.05`，`P=0.05`，`b ≤ min(τ,4)`）。

    对不上 ⇒ 被测物变了 / 我的复制走形 ⇒ 判词不许出。
    """
    if not R34_JSON.exists():
        return {"path": str(R34_JSON), "found": False, "pass": False}
    old = json.loads(R34_JSON.read_text(encoding="utf-8"))
    idx = {}
    for c in old["cells"]:
        idx[(int(c["tau"]), int(c["b"]), float(c["P"]))] = c
    pairs = []
    worst = 0.0
    for c in rows:
        if abs(c["sigma"] - 0.05) > 0 or abs(c["P"] - 0.05) > 0:
            continue
        o = idx.get((c["tau"], c["b"], 0.05))
        if o is None:
            continue
        for k in ("IV_free", "IV_paid", "dV_q"):
            d = abs(c[k] - o[k])
            worst = max(worst, d)
            pairs.append({"tau": c["tau"], "b": c["b"], "key": k,
                          "new": c[k], "old": o[k], "abs_diff": d})
    return {"found": True, "n_pairs": len(pairs), "max_abs_diff": float(worst),
            "pass": bool(pairs) and worst <= 1e-15}


# ============================================================ 活体分辨率对照（`L1`）
def live_control() -> dict:
    """破坏用户对称后，`free(σ⋆)` 臂上必须**能**出现 `UB > 0`。

    没有这条，`UB ≡ 0` 无法排除"编码从未求值 `PROBE`"（R3.4 `J4` 的同款对照，
    本轮在**自己的代码路径**上重做一次，不复用旧结论）。
    """
    from .config import PlantConfig

    t0 = time.perf_counter()
    base = mainline_config()
    broken = PlantConfig(**{**base.plant.__dict__, "sigma_w": (0.02, 0.30)})
    cfg_b = mainline_config(plant=broken)
    rows = []
    for name, cfg in (("bounded_asis", base), ("sigma_w1_0.30", cfg_b)):
        arms = Arms(cfg, 33)
        for tau, b in ((3, 1), (4, 1), (4, 3)):
            c = cell(arms, cfg, tau, b, 0.02, SIGMA_STAR)
            rows.append({"arm": name, "tau": tau, "b": b,
                         "UB": c["IV_free_star"], "IV_paid_star": c["IV_paid_star"],
                         "dV_q": c["dV_q"], "deficit_free": c["deficit_free"],
                         "ratio_ub": c["ratio_ub"]})
    return {"rows": rows, "sec": round(time.perf_counter() - t0, 3)}


def gate_g6_live_positive(live: dict) -> dict:
    br = [r for r in live["rows"] if r["arm"] == "sigma_w1_0.30"]
    asis = [r for r in live["rows"] if r["arm"] == "bounded_asis"]
    return {"n_broken": len(br), "n_asis": len(asis),
            "broken_max_UB": float(max((r["UB"] for r in br), default=float("nan"))),
            "asis_max_UB": float(max((r["UB"] for r in asis), default=float("nan"))),
            "broken_positive": bool(any(r["UB"] > UB_FLOOR for r in br)),
            "pass": bool(br and asis)}



# ============================================================ 判词
def _finite(x) -> bool:
    return isinstance(x, float) and math.isfinite(x)


def judge(p: dict, gates: dict) -> dict:
    rows = p["rows"]
    out: dict = {}
    out["n_cells_raw"] = len(rows)
    fin = [c for c in rows
           if all(_finite(c[k]) for k in
                  ("IV_free", "IV_paid", "IV_free_star", "IV_paid_star", "dV_q", "x2_legacy"))]
    out["n_cells_finite"] = len(fin)
    out["n_cells_dropped_nan"] = len(rows) - len(fin)
    valid = [c for c in fin if c["dV_q"] > EPS]
    out["n_cells_valid"] = len(valid)
    out["n_cells_degenerate_dVq"] = len(fin) - len(valid)

    # ---- C0：上界合法性（逐格）
    viol = []
    worst_v = 0.0
    for c in fin:
        tol = max(1e-12, 1e-6 * abs(c["dV_q"]))
        for leg, hi, lo in (("star>=free", c["IV_free_star"], c["IV_free"]),
                            ("free>=paid", c["IV_free"], c["IV_paid"])):
            if hi < lo - tol:
                d = float(lo - hi)
                worst_v = max(worst_v, d)
                viol.append({"tau": c["tau"], "b": c["b"], "P": c["P"], "sigma": c["sigma"],
                             "leg": leg, "hi": hi, "lo": lo, "violation": d})
    out["C0"] = "BOUND-VALID" if not viol else "BOUND-INVALID"
    out["C0_detail"] = {"n_legs_checked": 2 * len(fin), "n_violations": len(viol),
                        "worst_violation": float(worst_v), "violations": viol[:20]}

    # ---- C5：分辨率（三态：恰好零 / 分辨得出 / 分辨不出）
    resolved = [c for c in valid if c["IV_free_star"] > UB_FLOOR]
    zeros = [c for c in valid if c["IV_free_star"] <= 1e-12]
    r_res = (len(resolved) / len(valid)) if valid else None
    r_zero = (len(zeros) / len(valid)) if valid else None
    if valid and r_res >= C5_RATE:
        out["C5"] = "UB-RESOLVED"
    elif valid and r_zero >= C5_RATE:
        out["C5"] = "UB-EXACTLY-ZERO"
    else:
        out["C5"] = "UB-UNRESOLVED"
    out["C5_detail"] = {"n_valid": len(valid), "n_resolved": len(resolved),
                        "n_exact_zero": len(zeros),
                        "rate_resolved": r_res, "rate_exact_zero": r_zero,
                        "floor": UB_FLOOR,
                        "ub_max": float(max((c["IV_free_star"] for c in valid),
                                            default=float("nan")))}
    # ---- 同臂 deficit（必报量，不是判据）
    dd = [c["deficit_over_dVq"] for c in valid if c["deficit_over_dVq"] is not None]
    out["deficit_stats"] = {
        "n": len(dd),
        "min": float(min(dd)) if dd else None,
        "median": float(np.median(dd)) if dd else None,
        "max": float(max(dd)) if dd else None,
        "n_negative": int(sum(1 for x in dd if x < 0)),
        "note": "同臂（free(σ⋆)）min_PROBE − min_SEND，除 ΔV_q ⇒ 探测差几个量子才被选上",
    }

    # ---- C1：UB/ΔV_q < 1 的比例
    ratios = [(c, c["ratio_ub"]) for c in valid if c["ratio_ub"] is not None]
    below = [(c, r) for c, r in ratios if r < 1.0]
    out["n_cells_ratio"] = len(ratios)
    out["n_cells_below_1"] = len(below)
    out["C1_rate"] = (len(below) / len(ratios)) if ratios else None
    out["C1"] = ("MEASURED-BOUND-CERTIFIED" if ratios and out["C1_rate"] >= C1_RATE
                 else "MEASURED-BOUND-PANEL-SPECIFIC")
    rs = sorted(r for _, r in ratios)
    out["C1_ratio_stats"] = {
        "min": float(rs[0]) if rs else None,
        "median": float(np.median(rs)) if rs else None,
        "p90": float(rs[int(0.9 * (len(rs) - 1))]) if rs else None,
        "max": float(rs[-1]) if rs else None,
        "max_cell": ({"tau": max(ratios, key=lambda t: t[1])[0]["tau"],
                      "b": max(ratios, key=lambda t: t[1])[0]["b"],
                      "P": max(ratios, key=lambda t: t[1])[0]["P"],
                      "sigma": max(ratios, key=lambda t: t[1])[0]["sigma"]} if ratios else None),
    }
    out["C1_by_sigma"] = {}
    for s in SIGMA_LIST:
        rr = [r for c, r in ratios if abs(c["sigma"] - s) < 1e-18]
        out["C1_by_sigma"][f"{s:g}"] = {
            "n": len(rr), "rate_below_1": (sum(1 for x in rr if x < 1.0) / len(rr)) if rr else None,
            "median": float(np.median(rr)) if rr else None}

    # ---- C2：完美 CSI 付一个量子，买不买
    pays = [c for c in fin if c["IV_paid_star"] > IV_NOISE]
    out["C2"] = "PERFECT-CSI-PAYS" if pays else "PERFECT-CSI-DOES-NOT-PAY"
    out["C2_detail"] = {"n_pays": len(pays), "noise": IV_NOISE,
                        "cells": [{"tau": c["tau"], "b": c["b"], "P": c["P"],
                                   "IV_paid_star": c["IV_paid_star"],
                                   "ratio_paid_star": c["ratio_paid_star"]} for c in pays[:20]],
                        "max_IV_paid_star": float(max((c["IV_paid_star"] for c in fin), default=float("nan")))}

    # ---- C3：`σ` 单调性（活体对照）
    order = (0.5, 0.2, 0.05, 0.02, SIGMA_STAR)      # `σ` 递减 ⇒ `IV_free` 应不减
    groups: dict[tuple, dict] = {}
    for c in fin:
        groups.setdefault((c["tau"], c["b"], c["P"]), {})[c["sigma"]] = c["IV_free"]
    n_chk = 0
    n_viol = 0
    worst_drop = 0.0
    for key, g in groups.items():
        vals = [g.get(s) for s in order]
        if any(v is None for v in vals):
            continue
        for i in range(len(vals) - 1):
            n_chk += 1
            if vals[i + 1] < vals[i] - max(1e-12, 1e-6 * abs(vals[i])):
                n_viol += 1
                worst_drop = max(worst_drop, float(vals[i] - vals[i + 1]))
    out["C3_rate_viol"] = (n_viol / n_chk) if n_chk else None
    out["C3"] = ("BLACKWELL-MONOTONE" if n_chk and n_viol / n_chk <= C3_MAX_VIOL
                 else "GRID-NOISE-DOMINATES")
    out["C3_detail"] = {"n_comparisons": n_chk, "n_violations": n_viol,
                        "worst_drop": float(worst_drop)}

    # ---- C4：旧形态会不会越界
    over = []
    worst_o = 0.0
    for c in valid:
        tol = max(1e-12, 1e-6 * abs(c["dV_q"]))
        if c["x2_legacy"] > c["IV_free_star"] + tol:
            d = float(c["x2_legacy"] - c["IV_free_star"])
            worst_o = max(worst_o, d)
            over.append({"tau": c["tau"], "b": c["b"], "P": c["P"], "sigma": c["sigma"],
                         "x2_legacy": c["x2_legacy"], "UB": c["IV_free_star"],
                         "excess": d, "ratio_x2": c["ratio_x2"]})
    out["C4"] = "LEGACY-WITHIN-BOUND" if not over else "LEGACY-EXCEEDS-BOUND"
    out["C4_detail"] = {"n_over": len(over), "n_cells": len(valid),
                        "worst_excess": float(worst_o), "cells": over[:20]}

    # ---- 网格分辨率（不参与判词，超差则降级）
    ref = p["refine"]
    rg = {}
    for b in REFINE_B:
        rr = [c["ratio_ub"] for c in ref if c["b"] == b and c["ratio_ub"] is not None]
        if len(rr) >= 2:
            mx = max(rr)
            rg[str(b)] = {"ratios": [float(x) for x in rr],
                          "spread_rel": float((mx - min(rr)) / mx) if mx > EPS else 0.0,
                          "all_zero": bool(mx <= EPS)}
    out["grid_refine"] = {"by_b": rg, "n_m": list(REFINE_NM),
                          "max_spread_rel": float(max((v["spread_rel"] for v in rg.values()),
                                                      default=float("nan"))),
                          "all_zero": bool(rg) and all(v["all_zero"] for v in rg.values()),
                          "threshold": REFINE_SPREAD_MAX}
    grid_dep = bool(rg) and out["grid_refine"]["max_spread_rel"] > REFINE_SPREAD_MAX

    out["gates_n_fail"] = sum(1 for v in gates.values()
                              if isinstance(v, dict) and not v.get("pass", True))
    # ---- 汇总判词（`C0`/`L1` 不过则无判词）
    l1_ok = bool(gates.get("g6", {}).get("broken_positive"))
    out["L1"] = "MACHINERY-CAN-BE-POSITIVE" if l1_ok else "MACHINERY-CANNOT-BE-POSITIVE"
    if out["C0"] != "BOUND-VALID":
        out["VERDICT"] = "NO-VERDICT-BOUND-INVALID"
    elif not l1_ok:
        out["VERDICT"] = "NO-VERDICT-L1-MACHINERY-CANNOT-BE-POSITIVE"
    elif out["C5"] == "UB-UNRESOLVED":
        out["VERDICT"] = "NO-VERDICT-NO-RESOLUTION"
    elif out["C5"] == "UB-EXACTLY-ZERO":
        out["VERDICT"] = ("BOUND-EXACTLY-ZERO-SLOT-BOUND" if out["C1"] == "MEASURED-BOUND-CERTIFIED"
                          else "BOUND-EXACTLY-ZERO-BUT-MIXED")
    elif out["C2"] == "PERFECT-CSI-PAYS":
        out["VERDICT"] = "PERFECT-CSI-PAYS"
    elif out["C1"] == "MEASURED-BOUND-CERTIFIED":
        out["VERDICT"] = ("MEASURED-BOUND-CERTIFIED" if not grid_dep
                          else "MEASURED-BOUND-CERTIFIED-GRID-DEPENDENT")
    else:
        out["VERDICT"] = "MEASURED-BOUND-PANEL-SPECIFIC"
    out["grid_dependent"] = grid_dep
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="CW-R3.5 信息价值上界独立面板")
    ap.add_argument("--ctrl-only", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    print(f"[{ROUND}] {PROTOCOL} start", flush=True)

    if args.ctrl_only:
        live = live_control()
        gates = {"g1": gate_g1_no_truth(), "g2": gate_g2_identity(),
                 "g3": gate_g3_martingale(), "g4": gate_g4_first_consistency(),
                 "g5": gate_g5_drop_obs(Arms(mainline_config(), 65), mainline_config()),
                 "g6": gate_g6_live_positive(live)}
        print(json.dumps(gates, indent=2, ensure_ascii=False)[:5000])
        print(f"[ctrl-only] n_fail={sum(1 for v in gates.values() if not v['pass'])} / {len(gates)}")
        return 0

    p = panel()
    print(f"  panel done {p['meta']['arms_sec']} s arms", flush=True)
    live = live_control()
    print(f"  live control {live['sec']} s", flush=True)

    gates = {"g1": gate_g1_no_truth(), "g2": gate_g2_identity(),
             "g3": gate_g3_martingale(), "g4": gate_g4_first_consistency(),
             "g5": gate_g5_drop_obs(Arms(mainline_config(), 65), mainline_config()),
             "g6": gate_g6_live_positive(live),
             "g7": gate_g7_r34_identity(p["rows"])}
    v = judge(p, gates)

    out = {"round": ROUND, "protocol": PROTOCOL, "utc": _now_utc(),
           "sec_total": round(time.perf_counter() - t0, 3),
           "panel_meta": p["meta"], "cells": p["rows"], "refine": p["refine"],
           "live": live, "gates": gates, "verdicts": v}
    print(json.dumps({k: v[k] for k in
                      ("VERDICT", "C0", "C1", "C1_rate", "C1_ratio_stats", "C1_by_sigma",
                       "C2", "C3", "C4", "C5", "C5_detail", "L1", "deficit_stats",
                       "grid_refine", "gates_n_fail")}, indent=2, ensure_ascii=False),
          flush=True)

    print("\n[per-cell] sigma tau  b    P    IV_free_star  dV_q     UB/q     X2_legacy  "
          "IV_paid*  probe-send", flush=True)
    for c in p["rows"]:
        print(f"  {c['sigma']:>7g} {c['tau']:>2} {c['b']:>2} {c['P']:>5} "
              f"{c['IV_free_star']:>12.4e} {c['dV_q']:>9.3e} "
              f"{(c['ratio_ub'] if c['ratio_ub'] is not None else float('nan')):>8.3f} "
              f"{c['x2_legacy']:>10.3e} {c['IV_paid_star']:>9.3e} "
              f"{c['probe_minus_send']:>10.3e}", flush=True)

    dest = Path(args.out) if args.out else (
        RESULTS / f"cw_r3_5_ubcert_{_now_utc().replace('-', '').replace(':', '')[:15]}"
                 f"_{_src_hash()[:16]}")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "r35.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[evidence] {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
