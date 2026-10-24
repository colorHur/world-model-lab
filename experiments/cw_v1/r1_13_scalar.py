# -*- coding: utf-8 -*-
"""CW-R1.13 一个标量能不能复现整条精确曲线？—— `ACK_VAR ∈ {0.63, 1.000, 1.726}` × exact。

预注册：`docs/research/CW-R1.13-标量vs形状-预注册-v0.1.md`（§8 记看数后的修订）

四段
----
`Q1` **零 episode**：精确后验的 Fisher 信息 `I(m)=κ²p(1−p)` 与一阶近似隐含的**常数** `1/ACK_VAR`
   并列 ⇒ 求 `m*`（使两者相等的位置）、外生/内生两个工作域上的 `sup/min`、以及均值位移的符号翻转区间。
`P0` **前置闸**：① `G1/G2` 身份 —— `A0`(0.63) 与 `AX`(exact) 的 8 格必须与 R1.12 入库值**逐位**相同；
   ② 全量 `gates.py` `n_fail=0`；③ **活体负控** —— 把 `_ack_bump` 换回"读类常量"的旧实现，`4s` 必须 FAIL。
`PANEL` **32 格**：`arm ∈ {0.63, 1.000, 1.726, exact} × ep ∈ {100,101} × seed ∈ {2001..2004}`。
`Q2` **判词**：预注册 §6 的五档，**先判 VOID**，两臂都报，不许挑。

★ 本轮**只动** `ACK_VAR` 这个标量（方向量级 `±1.0/−1.5` 由闸 `4t` 钉住不许动）。
"""

from __future__ import annotations

import argparse
import dataclasses as _dc
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import config as C
from . import gates as G
from . import plan_ref as Pmodule
from .runner import check_identities, run_episode
from .world import CWWorld, GridBelief

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results" / "cw_v1"
R112_DIR = RESULTS / "cw_r1_12_ackscale_20261024T020453_92d6a8ccd0bc1a26"
R112_CELLS = R112_DIR / "cells.json"

RHO = 0.98
H_PLAN = 5
M_MAIN = 16
RATIO = 0.5
EPS = (100, 101)
SEEDS = (2001, 2002, 2003, 2004)
TAIL_DP = {"tail": "dp", "dp_grid": "r173"}

AV_BASE = 0.63       # `CWWorld.ACK_VAR`，基线
AV_FISHER = 1.000    # R1.12 §Q1 的 Fisher 口径（m=0）
AV_MOMENT = 1.726    # R1.12 §Q1 的矩匹配口径
ARMS = (("0.63", AV_BASE), ("1.000", AV_FISHER), ("1.726", AV_MOMENT), ("exact", None))

FLOOR = 0.0333        # 预注册 §4.2-P4（R1.12 同值，不许换窄口径）
REL_TOL = 0.20        # 预注册 §4.2-P1
ABS_TOL = 0.05        # 预注册 §4.2-P3
CAP_TOL = 0.20        # 预注册 §4.2-P2
SHAPE_TOL = 0.50      # 预注册 §6-4

KAPPA = 2.0
STAR = 0.0


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


def _src_hash() -> str:
    h = hashlib.sha256()
    for p in sorted((REPO / "experiments" / "cw_v1").glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def _git(args: list[str]) -> str:
    return subprocess.run(["D:\\devtools\\Git\\cmd\\git.exe", *args],
                          cwd=str(REPO), capture_output=True, text=True).stdout.strip()


# ==================================================================== Q1（零 episode）
def _p_of(m: float) -> float:
    return 1.0 / (1.0 + np.exp(-KAPPA * (m - STAR)))


def _info(m: float) -> float:
    p = _p_of(m)
    return KAPPA * KAPPA * p * (1.0 - p)


def _m_star(ack_var: float):
    """解 `I(m)==1/ACK_VAR`；无解返回 None（⇒ 该标量在任何位置都匹配不上精确信息）。"""
    target = 1.0 / ack_var
    q = target / (KAPPA * KAPPA)
    disc = 1.0 - 4.0 * q
    if disc < 0.0:
        return None
    r = (1.0 - np.sqrt(disc)) / 2.0          # 取 p<0.5 的支
    return float(0.5 * np.log(r / (1.0 - r)))


def q1_formula() -> dict:
    """外生工作域的纯公式部分（`m∈[−3,3]` 与 `[−1,1]`、`[−2,2]` 三档并列）。"""
    out = {"kappa": KAPPA, "log_h_star": STAR,
           "sup_info": float(_info(0.0)),
           "note_sup": "sup I = κ²·0.25 = 1.000，仅 m=0 取得 ⇒ 任何 ACK_VAR<1 隐含的 I>1 都不可达"}
    for lo, hi in ((3.0, 3.0), (1.0, 1.0), (2.0, 2.0)):
        ms = np.arange(-lo, hi + 1e-12, 0.001)
        I = np.array([_info(m) for m in ms])
        out[f"exo_domain_pm{hi:g}"] = {
            "m_lo": -lo, "m_hi": hi,
            "info_sup": float(I.max()), "info_min": float(I.min()),
            "sup_over_min": float(I.max() / I.min()),
            "r_eff_range": [float(1.0 / I.max()), float(1.0 / I.min())],
        }
    out["m_star"] = {f"{av:g}": _m_star(av) for av in (AV_BASE, AV_FISHER, AV_MOMENT)}
    out["m_star_note"] = ("0.63 ⇒ NONE（隐含 I=1.587 > sup I=1.000）；"
                          "1.000 ⇒ 0（只在 m=0）；1.726 ⇒ ±0.7728")
    # 均值位移：v=1 下 exact 的 Δm 与标量 g·direction 并列
    rows = []
    for m in (-3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
        gb_ref = GridBelief(1, rho=0.98, kappa=KAPPA, log_h_star=STAR,
                            init_mean=m, init_std=1.0)
        rec = {"m": m, "info": float(_info(m)), "p": float(_p_of(m))}
        for av in (AV_BASE, AV_FISHER, AV_MOMENT):
            g_sc = 1.0 / (1.0 + av)
            rec[f"scalar_{av:g}"] = {"ack_dm": g_sc * 1.0, "nack_dm": -g_sc * 1.5}
        for ack, tag in ((True, "ack"), (False, "nack")):
            gb = GridBelief(1, rho=0.98, kappa=KAPPA, log_h_star=STAR,
                            init_mean=m, init_std=1.0)
            gb.observe_binary(0, ack)
            em, ev = gb.moments()
            rec[f"exact_{tag}_dm"] = float(em[0]) - m
            rec[f"exact_{tag}_v"] = float(ev[0])
        rec["sign_flip_nack_scalar0.63"] = {
            "exact": rec["exact_nack_dm"], "scalar": rec["scalar_0.63"]["nack_dm"],
            "opposite": bool(np.sign(rec["exact_nack_dm"]) != np.sign(rec["scalar_0.63"]["nack_dm"]))}
        # ★ §8 事后追加（只增报道列，不改任何判据）：一阶近似把 |NACK|/|ACK| **钉死**在 1.5，
        #   而 exact 的该比值随 m 单调变化 ⇒ 这是"形状"最直观的一列（比符号翻转更早暴露）。
        rec["asym_ratio_exact"] = (abs(rec["exact_nack_dm"]) / abs(rec["exact_ack_dm"])
                                   if rec["exact_ack_dm"] else None)
        rec["asym_ratio_scalar_0.63"] = 1.5
        rows.append(rec)
    out["mean_shift_rows"] = rows
    flips = [r["m"] for r in rows if r["sign_flip_nack_scalar0.63"]["opposite"]]
    out["sign_flip_interval_nack"] = [min(flips), max(flips)] if flips else None
    out["max_abs_shift_err_0.63"] = float(max(
        max(abs(r["exact_ack_dm"] - r["scalar_0.63"]["ack_dm"]),
            abs(r["exact_nack_dm"] - r["scalar_0.63"]["nack_dm"])) for r in rows))
    return out


def q1_endogenous(m_domain: dict) -> dict:
    """预注册 §4.1-Q1b：内生工作域（A0 臂实测 m 轨迹的 5%–95% 分位）。"""
    lo, hi = m_domain["m_p05"], m_domain["m_p95"]
    ms = np.linspace(lo, hi, 2001)
    I = np.array([_info(m) for m in ms])
    return {"m_lo": lo, "m_hi": hi, "source": "arm 0.63 逐槽根节点 h_mean（两用户 × 8 格）",
            "info_sup": float(I.max()), "info_min": float(I.min()),
            "sup_over_min": float(I.max() / I.min()) if I.min() > 0 else None}


# ==================================================================== 记录器
class _Rec(Pmodule.PlanRef):
    """只观测：逐槽公开 telemetry + 根节点 p̂；不改任何内层行为。"""

    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)
        self.rows: list[dict] = []

    def __call__(self, v):                      # noqa: ANN001
        a = super().__call__(v)
        tel = v.telemetry
        mh = np.asarray([[float(r["h_mean"]) for r in tel]], dtype=float)
        vh = np.asarray([[float(r["h_var"]) for r in tel]], dtype=float)
        ph = self._p_hat(mh, vh)[0]
        self.rows.append({"t": int(v.t), "phat": [float(x) for x in ph],
                          "h_mean": [float(x) for x in mh[0]],
                          "h_var": [float(x) for x in vh[0]]})
        return a


def _cfg(arm: str, av) -> C.CWConfig:
    base = C.CWConfig(plant=C.bounded_plant())
    if arm == "exact":
        return _dc.replace(base, channel=_dc.replace(base.channel, ack_mode="exact"))
    return _dc.replace(base, channel=_dc.replace(base.channel, ack_mode="approx",
                                                 ack_var=float(av)))


# ==================================================================== 单格
def run_cell(ep: int, seed: int, arm: str, av) -> dict:
    cfg = _cfg(arm, av)
    budget = cfg.protocol.budget(RATIO)
    pol = _Rec(cfg, h_plan=H_PLAN, m_samples=M_MAIN, seed=seed, rho=RHO,
               pacing=True, **TAIL_DP)
    t0 = time.perf_counter()
    res = run_episode(cfg, pol, split="dev", episode=ep, budget_total=budget, rho=RHO)
    elapsed = time.perf_counter() - t0
    chk = check_identities(res, cfg)
    att = [int(x) for x in res.n_attempted_by_user]
    dlv = [int(x) for x in res.n_delivered_by_user]
    nk = [int(x) for x in (res.n_nack_by_user or [])]
    ak = [int(x) for x in (res.n_ack_by_user or [])]
    rows = pol.rows
    dphi = [r["phat"][0] - r["phat"][1] for r in rows]
    pmin = [min(r["phat"][i] for r in rows) for i in range(len(att))]
    pla = [rows[-1]["phat"][i] for i in range(len(att))]
    m_traj = [x for r in rows for x in r["h_mean"]]
    ident_ok = all(att[i] == dlv[i] + nk[i] for i in range(len(att)))
    return {
        "ep": int(ep), "seed": int(seed), "arm": str(arm),
        "ack_mode": str(cfg.channel.ack_mode),
        "ack_var": (None if arm == "exact" else float(av)),
        "n_attempted_by_user": att, "n_delivered_by_user": dlv,
        "n_ack_by_user": ak, "n_nack_by_user": nk,
        "attribution_identity_ok": bool(ident_ok),
        "n_probe": int(res.n_probe), "n_hold": int(res.n_idle),
        "n_forced_hold": int(res.n_forced_hold),
        "n_never_served": int(res.n_never_served),
        "aoi_tavg_by_user": res.aoi_tavg_by_user,
        "cost_est_per_step": res.cost_est_per_step,
        "cost_true_per_step": res.cost_true_per_step,
        "cost_true_by_user": res.cost_true_by_user,
        "phat_min_by_user": pmin, "phat_last_by_user": pla,
        "phat_diff_median": (float(np.median(dphi)) if dphi else None),
        "m_p05": float(np.percentile(m_traj, 5)), "m_p95": float(np.percentile(m_traj, 95)),
        "m_min": float(np.min(m_traj)), "m_max": float(np.max(m_traj)),
        "policy_trace_hash": res.policy_trace_hash,
        "identities_pass": bool(chk["pass"]),
        "identities_fail": [k for k, v in chk["items"].items() if not v],
        "elapsed_s": round(elapsed, 2),
    }


# ==================================================================== 前置闸
def identity_check(arm: str, ref_map: dict, cells: list[dict]) -> dict:
    """`G1`(arm=0.63 vs R1.12 approx) / `G2`(arm=exact vs R1.12 exact)。"""
    key = "approx" if arm == "0.63" else "exact"
    rows = []
    for c in cells:
        if c["arm"] != arm:
            continue
        r = ref_map[(c["ep"], c["seed"], key)]
        d = {"ep": c["ep"], "seed": c["seed"],
             "cost_true_identical": c["cost_true_per_step"] == r["cost_true_per_step"],
             "aoi_tavg_identical": c["aoi_tavg_by_user"] == r["aoi_tavg_by_user"],
             "attempted_identical": c["n_attempted_by_user"] == r["n_attempted_by_user"],
             "phat_min_identical": c["phat_min_by_user"] == r["phat_min_by_user"],
             "trace_hash_identical": c["policy_trace_hash"] == r["policy_trace_hash"]}
        d["all_identical"] = all(v for k, v in d.items() if k.endswith("_identical"))
        rows.append(d)
    return {"source": str(R112_CELLS.relative_to(REPO)), "arm": arm,
            "n_cells": len(rows), "n_identical": sum(1 for d in rows if d["all_identical"]),
            "all_identical": bool(rows) and all(d["all_identical"] for d in rows),
            "cells": rows}


def mutation_control() -> dict:
    """★ 活体负控：把 `_ack_bump` 换回"读类常量"的旧实现 ⇒ `4s` 必须 FAIL。

    ★ 只换实现、不换被测物语义；`finally` 里逐字节还原并断言还原后 `4s` 重新 PASS。
    """
    orig = CWWorld._ack_bump

    def legacy(self, i: int, direction: float) -> None:
        if self.ack_mode == "exact":
            self._ack_bump_exact(i, direction)
            return
        prior_var = float(self._v_h[i])
        y = float(self._m_h[i]) + direction
        g = prior_var / (prior_var + float(CWWorld.ACK_VAR))     # ← 缺陷：忽略配置
        self._m_h[i] = self._m_h[i] + g * (y - self._m_h[i])
        self._v_h[i] = (1.0 - g) * prior_var

    res = {}
    try:
        CWWorld._ack_bump = legacy                                  # type: ignore[assignment]
        g_neg = G.Gate()
        G.counterexample_4b2_ack_var(C.default_config(), g_neg)
        res["legacy"] = {r["name"][:2]: bool(r["pass"]) for r in g_neg.rows}
        res["legacy_4s_failed"] = not res["legacy"].get("4s", True)
        res["legacy_4r_passed"] = bool(res["legacy"].get("4r", False))
    finally:
        CWWorld._ack_bump = orig                                    # type: ignore[assignment]
    g_pos = G.Gate()
    G.counterexample_4b2_ack_var(C.default_config(), g_pos)
    res["restored"] = {r["name"][:2]: bool(r["pass"]) for r in g_pos.rows}
    res["restored_all_pass"] = all(res["restored"].values())
    res["restore_identical"] = CWWorld._ack_bump is orig
    res["has_discriminating_power"] = bool(res["legacy_4s_failed"] and res["restored_all_pass"]
                                          and res["restore_identical"])
    res["note"] = ("`4r` 在旧实现下也 PASS —— 它只看配置默认值与 validate，不看接线；"
                   "**只有 `4s` 能抓「配置没接线」**（这正是 R12 的落地形式）。")
    return res


# ==================================================================== 判词
def _pairs(cells: list[dict], arm_a: str, arm_x: str, arm_0: str, metric: str) -> list[dict]:
    """对齐 16 对（逐用户）或 8 对（逐格标量）。"""
    idx = {(c["arm"], c["ep"], c["seed"]): c for c in cells}
    out = []
    for ep in EPS:
        for s in SEEDS:
            a, x, b = idx[(arm_a, ep, s)], idx[(arm_x, ep, s)], idx[(arm_0, ep, s)]
            if metric == "cost_true":
                out.append({"tag": f"ep{ep}/s{s}", "a": a["cost_true_per_step"],
                            "x": x["cost_true_per_step"], "b": b["cost_true_per_step"]})
            else:
                for i in range(2):
                    va = (a["phat_min_by_user"][i] if metric == "pmin"
                          else a[metric][i])
                    vx = (x["phat_min_by_user"][i] if metric == "pmin"
                          else x[metric][i])
                    vb = (b["phat_min_by_user"][i] if metric == "pmin"
                          else b[metric][i])
                    out.append({"tag": f"ep{ep}/s{s}/u{i}", "a": va, "x": vx, "b": vb})
    return out


def q2_arm(cells: list[dict], arm_a: str) -> dict:
    out = {"arm": arm_a}
    # P1：p̂ 下限相对差
    pr = _pairs(cells, arm_a, "exact", "0.63", "pmin")
    rels = [abs(p["a"] - p["x"]) / p["x"] if p["x"] else float("inf") for p in pr]
    absd = [abs(p["a"] - p["x"]) for p in pr]
    out["P1"] = {"n_pairs": len(rels), "n_pass_rel20": sum(1 for r in rels if r < REL_TOL),
                 "pass_all": bool(all(r < REL_TOL for r in rels)),
                 "rel_median": float(np.median(rels)), "rel_max": float(max(rels)),
                 "rels": [round(r, 4) for r in rels]}
    out["P3"] = {"n_pass_abs005": sum(1 for d in absd if d <= ABS_TOL),
                 "abs_median": float(np.median(absd)), "abs_max": float(max(absd)),
                 "abs": [round(d, 4) for d in absd]}
    # P2：效应捕获率
    caps = {}
    for metric in ("pmin", "cost_true", "aoi_tavg_by_user", "n_attempted_by_user"):
        pr_m = _pairs(cells, arm_a, "exact", "0.63", metric)
        num = float(np.median([abs(p["a"] - p["x"]) for p in pr_m]))
        den = float(np.median([abs(p["x"] - p["b"]) for p in pr_m]))
        den_floor = FLOOR if metric == "cost_true" else 1e-3
        caps[metric] = {"n_pairs": len(pr_m), "numerator_median": num, "denominator_median": den,
                        "denominator_floor": den_floor,
                        "no_effect": bool(den < den_floor),
                        "capture_ratio": (None if den < den_floor else num / den)}
    out["P2"] = caps
    # P4：代价配对差
    pc = _pairs(cells, arm_a, "exact", "0.63", "cost_true")
    p4 = {}
    for ep in EPS:
        d = [p["a"] - p["x"] for p in pc if p["tag"].startswith(f"ep{ep}/")]
        d0 = [p["b"] for p in pc if p["tag"].startswith(f"ep{ep}/")]
        p4[f"ep{ep}"] = {"paired_delta_per_seed": d, "median_delta": float(np.median(d)),
                         "exact_median": float(np.median(d0)),
                         "resolved_vs_floor": bool(abs(float(np.median(d))) >= FLOOR)}
    p4["floor"] = FLOOR
    p4["both_eps_below_floor"] = bool(all(not p4[f"ep{ep}"]["resolved_vs_floor"] for ep in EPS))
    out["P4"] = p4
    return out


def verdict_of(q2: dict, wiring: dict, identity: dict) -> dict:
    """预注册 §6：顺序锁死，先判 VOID。"""
    bad_id = [k for k, v in identity.items() if not v["all_identical"]]
    if bad_id:
        return {"verdict": "VOID-IDENTITY", "reason": f"身份闸不过：{bad_id}"}
    if not wiring["has_discriminating_power"]:
        return {"verdict": "VOID-CONTROL", "reason": "活体负控没有分辨力 ⇒ 判据不可用"}
    if not wiring["wired"]:
        return {"verdict": "VOID-NOT-WIRED", "reason": wiring["detail"]}
    per = {a: q2[a] for a in ("1.000", "1.726")}
    mech = {a: bool(per[a]["P1"]["pass_all"]
                    and (per[a]["P2"]["pmin"]["capture_ratio"] or 9e9) <= CAP_TOL) for a in per}
    cost_ok = {a: bool(per[a]["P4"]["both_eps_below_floor"]) for a in per}
    full = [a for a in per if mech[a] and cost_ok[a]]
    if full:
        return {"verdict": "SCALAR-EXPLAINS-ALL", "arms": full,
                "detail": {a: {"P1": per[a]["P1"]["n_pass_rel20"],
                               "cap_pmin": per[a]["P2"]["pmin"]["capture_ratio"],
                               "P4": {ep: per[a]["P4"][f"ep{ep}"]["median_delta"] for ep in EPS}}
                           for a in full}}
    part = [a for a in per if mech[a]]
    if part:
        return {"verdict": "SCALAR-EXPLAINS-MECHANISM-ONLY", "arms": part,
                "detail": {a: {"P4": {ep: per[a]["P4"][f"ep{ep}"]["median_delta"] for ep in EPS},
                               "P4_resolved": {ep: per[a]["P4"][f"ep{ep}"]["resolved_vs_floor"]
                                               for ep in EPS}} for a in part}}
    caps = {a: per[a]["P2"]["pmin"]["capture_ratio"] for a in per}
    if all((caps[a] or 9e9) > SHAPE_TOL for a in per):
        return {"verdict": "SHAPE-NEEDED",
                "detail": {"cap_ratio": caps,
                           "P1_pass": {a: per[a]["P1"]["n_pass_rel20"] for a in per}}}
    return {"verdict": "PARTIAL",
            "detail": {"cap_ratio": caps,
                       "P1_pass": {a: per[a]["P1"]["n_pass_rel20"] for a in per},
                       "P3_abs": {a: per[a]["P3"]["n_pass_abs005"] for a in per},
                       "note": "必须逐条列出过了哪条、差多少，不许归并成一句"}}


# ==================================================================== 主流程
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="CW-R1.13 标量 vs 形状（预注册 v0.1）")
    ap.add_argument("--probe", action="store_true", help="只跑 2 格报预算，**不写任何证据**")
    ap.add_argument("--tag", default="main")
    args = ap.parse_args(argv)

    if args.probe:
        t0 = time.perf_counter()
        a = run_cell(100, 2001, "0.63", AV_BASE)
        b = run_cell(100, 2001, "1.726", AV_MOMENT)
        print(json.dumps({"0.63": {k: a[k] for k in ("cost_true_per_step", "n_attempted_by_user",
                                                     "phat_min_by_user", "elapsed_s")},
                          "1.726": {k: b[k] for k in ("cost_true_per_step", "n_attempted_by_user",
                                                      "phat_min_by_user", "elapsed_s")},
                          "trace_differs": a["policy_trace_hash"] != b["policy_trace_hash"],
                          "elapsed_total_s": round(time.perf_counter() - t0, 1)},
                         ensure_ascii=False, indent=1, default=float))
        return 0

    t_start = time.perf_counter()
    head = _git(["rev-parse", "HEAD"])
    wt_sha = _git(["hash-object", "experiments/cw_v1/world.py"])
    blob_sha = _git(["rev-parse", f"{head}:experiments/cw_v1/world.py"])
    dirty = _git(["status", "--porcelain"])
    srchash = _src_hash()
    outdir = RESULTS / f"cw_r1_13_scalar_{_now_utc()}_{srchash}"
    outdir.mkdir(parents=True, exist_ok=False)
    print(f"[OUT] {outdir}")

    # ---- Q1（零 episode，纯公式）
    q1 = q1_formula()
    print(f"[Q1] sup/min(@±3)={q1['exo_domain_pm3']['sup_over_min']:.2f} "
          f"m*={ {k: (None if v is None else round(v,4)) for k,v in q1['m_star'].items()} } "
          f"flip={q1['sign_flip_interval_nack']}")

    # ---- 闸 + 活体负控
    t1 = time.perf_counter()
    gate = G.run_all()
    gate_s = round(time.perf_counter() - t1, 2)
    (outdir / "gates.json").write_text(json.dumps(gate, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
    print(f"[GATE] n_checks={gate['n_checks']} n_fail={gate['n_fail']} ({gate_s}s)")
    ctrl = mutation_control()
    (outdir / "mutation_control.json").write_text(
        json.dumps(ctrl, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[CTRL] has_discriminating_power={ctrl['has_discriminating_power']} "
          f"legacy_4s_failed={ctrl['legacy_4s_failed']}")

    # ---- PANEL 32 格
    cells: list[dict] = []
    for arm, av in ARMS:
        for ep in EPS:
            for seed in SEEDS:
                t0 = time.perf_counter()
                c = run_cell(ep, seed, arm, av)
                c["wall_s"] = round(time.perf_counter() - t0, 2)
                cells.append(c)
                print(f"[CELL] {arm:>5}/ep{ep}/s{seed} att={c['n_attempted_by_user']} "
                      f"del={c['n_delivered_by_user']} pmin="
                      f"{[round(x,4) for x in c['phat_min_by_user']]} "
                      f"cost={c['cost_true_per_step']:.5f} ({c['wall_s']}s)")

    ref = json.loads(R112_CELLS.read_text(encoding="utf-8"))
    ref_map = {(r["ep"], r["seed"], r["ack_mode"]): r for r in ref}
    id_a0 = identity_check("0.63", ref_map, cells)
    id_ax = identity_check("exact", ref_map, cells)
    identity = {"0.63_vs_R112_approx": id_a0, "exact_vs_R112_exact": id_ax}
    (outdir / "identity.json").write_text(json.dumps(identity, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    print(f"[ID] 0.63 {id_a0['n_identical']}/{id_a0['n_cells']} · "
          f"exact {id_ax['n_identical']}/{id_ax['n_cells']}")

    # ---- 接线（换标量真的改了动作？）
    idx = {(c["arm"], c["ep"], c["seed"]): c for c in cells}
    wrows, n1, n2 = [], 0, 0
    for ep in EPS:
        for s in SEEDS:
            h0 = idx[("0.63", ep, s)]["policy_trace_hash"]
            h1 = idx[("1.000", ep, s)]["policy_trace_hash"]
            h2 = idx[("1.726", ep, s)]["policy_trace_hash"]
            n1 += int(h1 != h0)
            n2 += int(h2 != h0)
            wrows.append({"ep": ep, "seed": s, "h_0.63": h0, "h_1.000": h1, "h_1.726": h2,
                          "d1": bool(h1 != h0), "d2": bool(h2 != h0)})
    wiring = {"rows": wrows, "n_diff_1.000": n1, "n_diff_1.726": n2, "n_total": len(wrows),
              "wired": bool(n1 >= 7 and n2 >= 7),
              "detail": f"hash 不同格数 1.000:{n1}/8、1.726:{n2}/8（预注册要求各 ≥7/8）",
              "has_discriminating_power": ctrl["has_discriminating_power"]}

    # ---- Q1b 内生工作域
    m0 = [x for c in cells if c["arm"] == "0.63" for x in (c["m_min"], c["m_max"])]
    dom = {"m_p05": float(np.percentile([c["m_p05"] for c in cells if c["arm"] == "0.63"], 5)),
           "m_p95": float(np.percentile([c["m_p95"] for c in cells if c["arm"] == "0.63"], 95)),
           "m_abs_min": float(min(abs(x) for x in m0)),
           "m_abs_max": float(max(abs(x) for x in m0))}
    q1b = q1_endogenous(dom)
    q1["endogenous"] = {"domain": dom, **q1b}
    (outdir / "q1_formula.json").write_text(json.dumps(q1, ensure_ascii=False, indent=1,
                                                       default=float), encoding="utf-8")
    print(f"[Q1b] endo_domain=[{dom['m_p05']:.3f},{dom['m_p95']:.3f}] "
          f"sup/min={q1b['sup_over_min']:.2f}")

    # ---- Q2
    q2 = {a: q2_arm(cells, a) for a in ("1.000", "1.726")}
    vd = verdict_of(q2, wiring, identity)

    summary = {
        "protocol": "CW-POMDP-v1.1",
        "stage": "CW-R1.13",
        "kind": "SCALAR-VS-SHAPE",
        "prereg": "docs/research/CW-R1.13-标量vs形状-预注册-v0.1.md",
        "verdict": vd["verdict"], "verdict_detail": vd,
        "q1_sup_over_min_exo_pm3": q1["exo_domain_pm3"]["sup_over_min"],
        "q1_sup_over_min_endo": q1b["sup_over_min"],
        "q1_m_star": q1["m_star"],
        "q1_sign_flip_interval_nack": q1["sign_flip_interval_nack"],
        "q1_max_abs_shift_err_0.63": q1["max_abs_shift_err_0.63"],
        "identity": {k: {"n_identical": v["n_identical"], "n_cells": v["n_cells"],
                         "all_identical": v["all_identical"]} for k, v in identity.items()},
        "wiring": {"n_diff_1.000": n1, "n_diff_1.726": n2, "n_total": len(wrows),
                   "wired": wiring["wired"], "has_discriminating_power": ctrl["has_discriminating_power"]},
        "q2": q2,
        "n_cells": len(cells),
        "floor": FLOOR, "rel_tol": REL_TOL, "abs_tol": ABS_TOL, "cap_tol": CAP_TOL,
        "head": head, "world_py_worktree_sha": wt_sha, "world_py_blob_sha": blob_sha,
        "src_hash": srchash, "git_dirty_before_run": dirty.splitlines(),
        "note": ("四臂只差 ACK/NACK 的**方差标量**（exact 臂换成网格精确贝叶斯）；"
                 "同 seed 是配对，不是独立样本；有效独立单元 = 2 个 episode。"),
    }
    (outdir / "cells.json").write_text(json.dumps(cells, ensure_ascii=False, indent=1,
                                                  default=float), encoding="utf-8")
    (outdir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1,
                                                    default=float), encoding="utf-8")
    try:
        import psutil  # type: ignore
        rss = psutil.Process(os.getpid()).memory_info().rss / 1024**2
    except Exception:
        rss = None
    meta = {"command": "python -m experiments.cw_v1.r1_13_scalar --tag " + str(args.tag),
            "python": os.sys.version.split()[0], "numpy": np.__version__,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "head": head, "src_hash": srchash,
            "world_worktree_sha": wt_sha, "world_blob_sha": blob_sha,
            "gate_elapsed_s": gate_s, "total_elapsed_s": round(time.perf_counter() - t_start, 1),
            "peak_rss_mb": round(rss, 1) if rss else None,
            "n_fail_gates": gate["n_fail"], "n_checks_gates": gate["n_checks"],
            "n_fail_cells": sum(1 for c in cells if not c["identities_pass"])}
    (outdir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                      encoding="utf-8")

    n_fail = (gate["n_fail"] + sum(1 for c in cells if not c["identities_pass"])
              + (0 if id_a0["all_identical"] and id_ax["all_identical"] else 1)
              + (0 if ctrl["has_discriminating_power"] else 1))
    print(f"[DONE] {outdir.name} n_fail={n_fail} total={meta['total_elapsed_s']}s "
          f"peak_rss={meta['peak_rss_mb']}MB")
    print(json.dumps({"verdict": vd["verdict"], "detail": vd.get("detail"),
                      "q2": {a: {"P1": q2[a]["P1"]["n_pass_rel20"], "P3": q2[a]["P3"]["n_pass_abs005"],
                                 "cap": {m: q2[a]["P2"][m]["capture_ratio"] for m in q2[a]["P2"]},
                                 "P4": {ep: q2[a]["P4"][f"ep{ep}"]["median_delta"] for ep in EPS}}
                            for a in q2}},
                     ensure_ascii=False, indent=1, default=float))
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    import sys
    sys.exit(main())
