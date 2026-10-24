# -*- coding: utf-8 -*-
"""CW-R1.15 送达缺口归因 —— exact 基线"送达 −18.4%"是**选错帧**还是**选错用户**。

预注册：`docs/research/CW-R1.15-送达缺口归因-预注册-v0.1.md`

四段
----
`P0` **前置闸**：① 全量 `gates.py` `n_fail=0`（含新增 `4y/4z`）；
   ② 16 格身份 —— 与 R1.14 同格同臂 5 字段**逐位**相同（纯记录不得改数字）。
`CTRL` **合成已知答案对照** `C1–C6`（**零 episode**）：先证明判据有分辨力，再看面板。
`PANEL` **16 格**：`arm ∈ {approx, exact} × ep ∈ {100,101} × seed ∈ {2001..2004}`
    —— 与 R1.14 **同格同配置**（直接复用 `r1_14_exactbase` 的 `_cfg/_Rec/常量`）。
`DEC` **精确可加分解** `Δr = E_user + E_frame + E_within`（预注册 §3.2），两档分辨率（10 箱 / 5 箱嵌套）。

★ 本轮**不动**任何被测物参数（`LINE_EFFECT`/阈值/网格/`ack_var`/`κ`/`log_h*`），
  只加**纯记录聚合量**（`runner.p_true_aggregates`）与一个离线判据。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import gates as G
from .r1_14_exactbase import (ARMS, EPS, H_PLAN, M_MAIN, RATIO, REPO, RESULTS, RHO,
                              SEEDS, TAIL_DP, _Rec, _cfg, _git, _src_hash)
from .runner import check_identities, run_episode

#: R1.14 的**正式**证据目录（身份闸参照物；作废版 `…T063012…` 不在其中）
R114_DIR = RESULTS / "cw_r1_14_exactbase_20261024T063157_c835f84ff417e9d2"
R114_CELLS = R114_DIR / "cells.json"

#: 预注册 §3.3：份额门槛与最小可判差（`JOINT_MIN` 见 §8-1 修订）
SHARE_MIN = 0.60
JOINT_MIN = 0.40
DR_MIN = 0.010
#: 预注册 §3.5-4：残差份额超过它就降级
RESID_MAX = 0.40


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


# ==================================================================== 单格
def run_cell(ep: int, seed: int, arm: str) -> dict:
    cfg = _cfg(arm)
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
    rows = pol.rows
    pmin = [min(r["phat"][i] for r in rows) for i in range(len(att))]
    return {
        "ep": int(ep), "seed": int(seed), "arm": str(arm),
        "ack_mode": str(cfg.channel.ack_mode),
        "config_fingerprint": res.config_fingerprint,
        "n_attempted_by_user": att, "n_delivered_by_user": dlv,
        "n_ack_by_user": [int(x) for x in (res.n_ack_by_user or [])],
        "n_nack_by_user": nk,
        "n_probe": int(res.n_probe), "n_hold": int(res.n_idle),
        "aoi_tavg_by_user": res.aoi_tavg_by_user,
        "cost_est_per_step": res.cost_est_per_step,
        "cost_true_per_step": res.cost_true_per_step,
        "phat_min_by_user": pmin,
        "policy_trace_hash": res.policy_trace_hash,
        "identities_pass": bool(chk["pass"]),
        "identities_fail": [k for k, v in chk["items"].items() if not v],
        # ---- R1.15 纯记录聚合量 ----
        "attempt_p_true_mean": res.attempt_p_true_mean,
        "attempt_p_true_q": res.attempt_p_true_q,
        "attempt_p_true_mean_by_user": res.attempt_p_true_mean_by_user,
        "attempt_p_true_q_by_user": res.attempt_p_true_q_by_user,
        "p_true_bin_edges": res.p_true_bin_edges,
        "n_attempt_in_bin_by_user": res.n_attempt_in_bin_by_user,
        "n_delivered_in_bin_by_user": res.n_delivered_in_bin_by_user,
        "elapsed_s": round(elapsed, 2),
    }


# ==================================================================== 分解
def decompose(ni_e, nd_e, ni_a, nd_a) -> dict:
    """预注册 §3.2 的三项分解（**精确可加**，残差容差 1e-12）。

    `ni_*/nd_*`：`n_users × n_bins` 的尝试/送达计数矩阵（int）。
    """
    ni_e = np.asarray(ni_e, dtype=float)
    nd_e = np.asarray(nd_e, dtype=float)
    ni_a = np.asarray(ni_a, dtype=float)
    nd_a = np.asarray(nd_a, dtype=float)
    n_e, n_a = ni_e.sum(), ni_a.sum()
    k_e, k_a = nd_e.sum(), nd_a.sum()
    if n_e <= 0 or n_a <= 0:
        return {"status": "NO-ATTEMPT", "n_attempt_e": float(n_e), "n_attempt_a": float(n_a)}
    r_e, r_a = k_e / n_e, k_a / n_a
    dr = r_e - r_a
    rho_i = np.where(ni_a.sum(axis=1) > 0, nd_a.sum(axis=1) / np.maximum(ni_a.sum(axis=1), 1), r_a)
    rho_ib = np.where(ni_a > 0, nd_a / np.maximum(ni_a, 1.0), rho_i[:, None])
    m_i = ni_e.sum(axis=1) - ni_a.sum(axis=1)
    m_ib = ni_e - ni_a
    e_user = float(np.sum(m_i * (rho_i - r_a)) / n_e)
    e_frame = float(np.sum(m_ib * (rho_ib - rho_i[:, None])) / n_e)
    e_within = float(np.sum((nd_e - nd_a) - rho_ib * m_ib) / n_e)
    resid = abs(e_user + e_frame + e_within - dr)
    shares = {"user": e_user / dr if dr else float("nan"),
              "frame": e_frame / dr if dr else float("nan"),
              "within": e_within / dr if dr else float("nan")}
    return {
        "status": "OK", "n_attempt_e": float(n_e), "n_attempt_a": float(n_a),
        "n_deliver_e": float(k_e), "n_deliver_a": float(k_a),
        "rate_e": float(r_e), "rate_a": float(r_a), "dr": float(dr),
        "E_user": e_user, "E_frame": e_frame, "E_within": e_within,
        "shares": {k: float(v) for k, v in shares.items()},
        "additivity_residual": float(resid),
        "n_empty_cells": int(np.sum(ni_a == 0)),
        "n_cells": int(ni_a.size),
    }


def coarsen(mat):
    """10 箱 → 5 箱（**嵌套粗化**：偶/奇箱合并）。断言这是同一份记录的两种分辨率。"""
    m = np.asarray(mat)
    if m.shape[-1] != 10:
        raise ValueError(f"coarsen 只接受 10 箱，收到 {m.shape[-1]}")
    return m.reshape(m.shape[:-1] + (5, 2)).sum(axis=-1)


def ep_counts(cells, arm: str, ep: int) -> tuple[np.ndarray, np.ndarray]:
    sel = [c for c in cells if c["arm"] == arm and c["ep"] == ep]
    if not sel:
        raise ValueError(f"没有 {arm}/ep{ep} 的格")
    ni = np.zeros_like(np.asarray(sel[0]["n_attempt_in_bin_by_user"]), dtype=int)
    nd = np.zeros_like(np.asarray(sel[0]["n_delivered_in_bin_by_user"]), dtype=int)
    for c in sel:
        ni += np.asarray(c["n_attempt_in_bin_by_user"], dtype=int)
        nd += np.asarray(c["n_delivered_in_bin_by_user"], dtype=int)
    return ni, nd


def classify(d: dict) -> dict:
    """预注册 §3.3（**§8-1 修订后**）：单 ep、单分辨率的机制判定。

    ★ 修订（写在 §8）：原卡写"两机制都成立 ⇔ 各自 `sh≥0.60`"，而三项份额**和为 1**
      ⇒ 那个条件恒不可能成立（合成对照 `C4` 设计成 0.5/0.5 时当场暴露）。
      改为：**单机制** `sh≥0.60`；**联合**区间 双方 `sh≥0.40`。修订时只看到 `C1–C6`，未看面板。
    """
    if d.get("status") != "OK":
        return {"status": d.get("status"), "mechanism": d.get("status")}
    dr = d["dr"]
    if abs(dr) < DR_MIN:
        return {"status": "GAP-TOO-SMALL", "dr": dr, "shares": d["shares"],
                "mechanism": "GAP-TOO-SMALL"}

    su, sf = d["shares"]["user"], d["shares"]["frame"]

    def _dom(share: float, e: float) -> bool:
        return abs(share) >= SHARE_MIN and np.sign(e) == np.sign(dr)

    ok_user, ok_frame = _dom(su, d["E_user"]), _dom(sf, d["E_frame"])
    joint = (abs(su) >= JOINT_MIN and abs(sf) >= JOINT_MIN
             and np.sign(d["E_user"]) == np.sign(dr) and np.sign(d["E_frame"]) == np.sign(dr))
    if ok_user and ok_frame:            # 份额和为 1 ⇒ 实际不可达，保留以防份额口径被改
        mech = "BOTH-MECHANISMS"
    elif ok_frame:
        mech = "H-A-FRAME-SELECTION"
    elif ok_user:
        mech = "H-B-USER-ALLOCATION"
    elif joint:
        mech = "BOTH-MECHANISMS"
    else:
        mech = "DELIVERY-GAP-UNEXPLAINED"
    part = abs(d["shares"]["within"]) > RESID_MAX and mech in (
        "H-A-FRAME-SELECTION", "H-B-USER-ALLOCATION")
    return {"status": "OK", "dr": dr, "shares": d["shares"], "mechanism": mech,
            "partial": bool(part), "ok_user": bool(ok_user), "ok_frame": bool(ok_frame),
            "joint": bool(joint)}


def combine(per_ep: dict) -> dict:
    """两 ep 合成判词（预注册 §3.3 表）。"""
    ms = [v.get("mechanism") for v in per_ep.values()]
    if any(m is None for m in ms):
        return {"verdict": "NOT-CLASSIFIABLE", "per_ep_mechanism": ms}
    if len(set(ms)) > 1:
        return {"verdict": "EP-DEPENDENT", "per_ep_mechanism": ms}
    v = ms[0]
    if v not in ("DELIVERY-GAP-UNEXPLAINED", "BOTH-MECHANISMS"):
        if any(x.get("partial") for x in per_ep.values()):
            v = "PARTIAL:" + v
    return {"verdict": v, "per_ep_mechanism": ms}


# ==================================================================== 合成对照
def _case(ni_e, nd_e, ni_a, nd_a) -> tuple:
    return (np.asarray(ni_e), np.asarray(nd_e), np.asarray(ni_a), np.asarray(nd_a))


def synthetic_controls() -> dict:
    """预注册 §4：`C1–C6`，答案**手算**（两箱；u0 好用户 rho=(0.8|0.9,0.4|0.5)、u1 差用户）。

    ★ 零 episode：直接构造计数矩阵 ⇒ 判据的分辨力与"答案已知"的算术都在看面板**之前**定死。
    ★ 手算过程写在每个 case 的 `hand` 字段里（预注册要求"先拿已知答案代验量级"）：
      `E_user = Σ_i m_i(ρ_i−ρ)/N^E`、`E_frame = Σ_i,b m_ib(ρ_ib−ρ_i)/N^E`、
      `E_within = Σ_i,b [(K^E_ib−K^A_ib) − ρ_ib·m_ib]/N^E`，且三项和恒等于 `Δr`。
    """
    rows = {}
    # C1 尝试集**完全相同** ⇒ 三项全 0、Δr=0 ⇒ GAP-TOO-SMALL（先过前置条件，不判机制）
    rows["C1"] = {
        "args": _case([[25, 25], [25, 25]], [[20, 5], [5, 5]],
                      [[25, 25], [25, 25]], [[20, 5], [5, 5]]),
        "expect_mech": "GAP-TOO-SMALL",
        "expect_shares": None, "expect_dr": 0.0,
        "expect_zeros": True,
        "hand": "两臂计数矩阵逐格相同 ⇒ 三项恒 0"}
    # C2 **只改用户分配**（u0 40→60、u1 60→40；每用户只用一个箱）⇒ E_user/Δr = 1
    #   ρ_a=(0.8,0.2)、ρ=0.44、Δr=0.56−0.44=+0.12；E_user=[20(0.8−0.44)+(−20)(0.2−0.44)]/100=0.12
    rows["C2"] = {
        "args": _case([[60, 0], [0, 40]], [[48, 0], [0, 8]],
                      [[40, 0], [0, 60]], [[32, 0], [0, 12]]),
        "expect_mech": "H-B-USER-ALLOCATION",
        "expect_shares": {"user": 1.0, "frame": 0.0, "within": 0.0}, "expect_dr": 0.12,
        "hand": "每用户单箱 ⇒ E_frame 恒 0；分配差全部进 E_user"}
    # C3 **只改 p 箱组成**（每用户 good 25→15，用户分配不变）⇒ E_frame/Δr = 1
    #   ρ_i=(0.5,0.5)、ρ=0.5、Δr=0.38−0.50=−0.12；E_frame=2·[(−10)(0.3)+(10)(−0.3)]/100=−0.12
    rows["C3"] = {
        "args": _case([[15, 35], [15, 35]], [[12, 7], [12, 7]],
                      [[25, 25], [25, 25]], [[20, 5], [20, 5]]),
        "expect_mech": "H-A-FRAME-SELECTION",
        "expect_shares": {"user": 0.0, "frame": 1.0, "within": 0.0}, "expect_dr": -0.12,
        "hand": "m_i≡0 ⇒ E_user 恒 0；箱间重分配全部进 E_frame"}
    # C4 两者**各一半**：先搬 10 次 u1→u0，再在每用户内搬 5 次 bin1→bin0
    #   ρ_a=(0.7,0.3)、ρ=0.5、N^E=80、Δr=0.60−0.50=+0.10
    #   E_user=[10(0.7−0.5)+(−10)(0.3−0.5)]/80=0.05 → 0.5
    #   E_frame：u0 [5,5]·(0.2,−0.2)=0；u1 [5,−15]·(0.2,−0.2)=1+3=4 → 4/80=0.05 → 0.5
    rows["C4"] = {
        "args": _case([[25, 25], [25, 5]], [[22.5, 12.5], [12.5, 0.5]],
                      [[20, 20], [20, 20]], [[18, 10], [10, 2]]),
        "expect_mech": "BOTH-MECHANISMS",
        "expect_shares": {"user": 0.5, "frame": 0.5, "within": 0.0}, "expect_dr": 0.10,
        "hand": "C2 的一半 + C3 的一半（按 5 次粒度落在同一基座上）"
    }
    # C5 **活动对照**：C3 反向（更多尝试进 good 箱）⇒ 份额同为 1、`E_frame` 符号必须**跟着翻**
    rows["C5"] = {
        "args": _case([[35, 15], [35, 15]], [[28, 3], [28, 3]],
                      [[25, 25], [25, 25]], [[20, 5], [20, 5]]),
        "expect_mech": "H-A-FRAME-SELECTION",
        "expect_shares": {"user": 0.0, "frame": 1.0, "within": 0.0}, "expect_dr": 0.12,
        "hand": "Δr 翻号 (−0.12 → +0.12)，份额不变 ⇒ 判据必须仍指 H-a 且 E_frame 变正"
    }
    # C6 **阴性对照**：同箱同计数、只让箱内送达率不同 ⇒ 归因到箱内 ⇒ UNEXPLAINED（不许硬凑机制）
    rows["C6"] = {
        "args": _case([[50, 0], [50, 0]], [[25, 0], [25, 0]],
                      [[50, 0], [50, 0]], [[40, 0], [18, 0]]),
        "expect_mech": "DELIVERY-GAP-UNEXPLAINED",
        "expect_shares": None, "expect_dr": -0.08,
        "hand": "m_ib≡0 ⇒ E_user=E_frame=0；Δr 全落 E_within（份额 1.0）"
    }

    out = {"cases": {}, "all_pass": True}
    for name, spec in rows.items():
        ni_e, nd_e, ni_a, nd_a = spec["args"]
        d = decompose(ni_e, nd_e, ni_a, nd_a)
        c = classify(d)
        ok = (c.get("mechanism") == spec["expect_mech"]) if spec["expect_mech"] else True
        shr_ok = True
        if spec["expect_shares"] is not None:
            shr_ok = all(abs(d["shares"][k] - v) <= 0.02
                         for k, v in spec["expect_shares"].items())
        dr_ok = True if spec["expect_dr"] is None else abs(d["dr"] - spec["expect_dr"]) <= 1e-12
        if spec.get("expect_zeros"):
            # `Δr=0` 时份额是 0/0（nan）⇒ 改判"三项绝对值逐位为 0"
            shr_ok = all(abs(d[k]) <= 1e-15 for k in ("E_user", "E_frame", "E_within"))
        add_ok = d["additivity_residual"] <= 1e-12
        passed = bool(ok and shr_ok and dr_ok and add_ok)
        out["cases"][name] = {"expect": spec["expect_mech"], "hand": spec["hand"],
                              "got": c.get("mechanism"), "dr": d.get("dr"),
                              "shares": d.get("shares"), "residual": d.get("additivity_residual"),
                              "E": {"user": d.get("E_user"), "frame": d.get("E_frame"),
                                    "within": d.get("E_within")},
                              "shares_ok": bool(shr_ok), "dr_ok": bool(dr_ok),
                              "additivity_ok": bool(add_ok), "pass": passed}
        out["all_pass"] = out["all_pass"] and passed
    # C5 的"符号必须跟着翻"单列一条：比的是**绝对项** `E_frame`（份额是比值，符号会抵消）
    e3 = out["cases"]["C3"]["E"]["frame"] or 0.0
    e5 = out["cases"]["C5"]["E"]["frame"] or 0.0
    out["c5_sign_flips"] = bool(e3 * e5 < 0)
    out["c5_detail"] = {"E_frame_C3": e3, "E_frame_C5": e5}
    out["all_pass"] = out["all_pass"] and out["c5_sign_flips"]
    return out


# ==================================================================== 身份闸
REF_FIELDS = ("cost_true_per_step", "aoi_tavg_by_user", "n_attempted_by_user",
              "phat_min_by_user", "policy_trace_hash")


def identity_check(cells: list[dict]) -> dict:
    ref = json.loads(R114_CELLS.read_text(encoding="utf-8"))
    ref_map = {(r["ep"], r["seed"], r["arm"]): r for r in ref}
    rows = []
    for c in cells:
        r = ref_map[(c["ep"], c["seed"], c["arm"])]
        d = {f: (c[f] == r[f]) for f in REF_FIELDS}
        d.update({"ep": c["ep"], "seed": c["seed"], "arm": c["arm"]})
        d["all_identical"] = all(v for k, v in d.items() if k in REF_FIELDS)
        rows.append(d)
    return {"source": str(R114_CELLS.relative_to(REPO)), "n_cells": len(rows),
            "n_identical": sum(1 for d in rows if d["all_identical"]),
            "all_identical": bool(rows) and all(d["all_identical"] for d in rows),
            "cells": rows}


# ==================================================================== 主流程
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="CW-R1.15 送达缺口归因（预注册 v0.1）")
    ap.add_argument("--probe", action="store_true", help="只跑 2 格报预算，**不写任何证据**")
    ap.add_argument("--tag", default="main")
    args = ap.parse_args(argv)

    # ---- CTRL 先行（零 episode）：判据没分辨力就不许看面板
    t_c0 = time.perf_counter()
    ctrl = synthetic_controls()
    ctrl_s = round(time.perf_counter() - t_c0, 2)
    print(f"[CTRL] all_pass={ctrl['all_pass']} c5_sign_flips={ctrl['c5_sign_flips']} ({ctrl_s}s)")
    for k, v in ctrl["cases"].items():
        print(f"  {k}: got={v['got']:>24} expect={v['expect']:>24} "
              f"dr={v['dr']:+.4f} sh={ {kk: round(vv, 3) for kk, vv in (v['shares'] or {}).items()} }"
              f" pass={v['pass']}")

    if args.probe:
        t0 = time.perf_counter()
        a = run_cell(100, 2001, "approx")
        b = run_cell(100, 2001, "exact")
        print(json.dumps({"approx": {k: a[k] for k in ("attempt_p_true_mean", "elapsed_s")},
                          "exact": {k: b[k] for k in ("attempt_p_true_mean", "elapsed_s")},
                          "elapsed_2cells_s": round(time.perf_counter() - t0, 1)},
                         ensure_ascii=False, indent=1, default=float))
        return 0

    if not ctrl["all_pass"]:
        print("[ABORT] 合成对照未过 ⇒ 判据无分辨力，按预注册 §3.5-3 本轮结论作废（不看面板）")
        return 3

    t_start = time.perf_counter()
    head = _git(["rev-parse", "HEAD"])
    wt_sha = _git(["hash-object", "experiments/cw_v1/runner.py"])
    blob_sha = _git(["rev-parse", f"{head}:experiments/cw_v1/runner.py"])
    dirty = _git(["status", "--porcelain"])
    srchash = _src_hash()
    outdir = RESULTS / f"cw_r1_15_delivery_{_now_utc()}_{srchash}"
    outdir.mkdir(parents=True, exist_ok=False)
    print(f"[OUT] {outdir}")

    # ---- P0-a 全量闸
    t1 = time.perf_counter()
    gate = G.run_all()
    gate_s = round(time.perf_counter() - t1, 2)
    (outdir / "gates.json").write_text(json.dumps(gate, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
    print(f"[GATE] n_checks={gate['n_checks']} n_fail={gate['n_fail']} ({gate_s}s)")

    # ---- PANEL 16 格
    cells: list[dict] = []
    for arm in ARMS:
        for ep in EPS:
            for seed in SEEDS:
                t0 = time.perf_counter()
                c = run_cell(ep, seed, arm)
                c["wall_s"] = round(time.perf_counter() - t0, 2)
                cells.append(c)
                print(f"[CELL] {arm:>6}/ep{ep}/s{seed} att={c['n_attempted_by_user']} "
                      f"del={c['n_delivered_by_user']} p̄={c['attempt_p_true_mean']:.4f} "
                      f"cost={c['cost_true_per_step']:.5f} ({c['wall_s']}s)")
                if not c["identities_pass"]:
                    print(f"  [WARN] 恒等失败: {c['identities_fail']}")

    # ---- P0-b 身份（16/16）
    ident = identity_check(cells)
    (outdir / "identity.json").write_text(json.dumps(ident, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    print(f"[ID] {ident['n_identical']}/{ident['n_cells']} 与 R1.14 逐位相同")

    # ---- DEC 分解（两档分辨率）
    dec = {}
    for res_name, fn in (("10bins", lambda m: np.asarray(m)), ("5bins", coarsen)):
        for ep in EPS:
            ni_e, nd_e = ep_counts(cells, "exact", ep)
            ni_a, nd_a = ep_counts(cells, "approx", ep)
            d = decompose(fn(ni_e), fn(nd_e), fn(ni_a), fn(nd_a))
            d["cls"] = classify(d)
            dec.setdefault(res_name, {})[f"ep{ep}"] = d
    for res_name, per in dec.items():
        v = combine({k: x["cls"] for k, x in per.items()})
        dec[res_name]["_verdict"] = v
        print(f"[DEC/{res_name}] {v['verdict']} " +
              " | ".join(f"{k}: dr={x['dr']:+.4f} "
                         f"sh(u/f/w)={x['shares']['user']:+.2f}/{x['shares']['frame']:+.2f}/"
                         f"{x['shares']['within']:+.2f}" if x.get("status") == "OK" else f"{k}: {x.get('status')}"
                         for k, x in per.items() if k != "_verdict"))

    primary = dec["10bins"]["_verdict"]["verdict"]
    secondary = dec["5bins"]["_verdict"]["verdict"]
    resolution_sensitive = (primary != secondary)
    verdict = "RESOLUTION-SENSITIVE" if resolution_sensitive else primary

    # ---- 分布描述量（不作判据，但与判据同报）
    dist = {}
    for ep in EPS:
        row = {}
        for arm in ARMS:
            sel = [c for c in cells if c["arm"] == arm and c["ep"] == ep]
            n = sum(sum(c["n_attempt_in_bin_by_user"][i]) for c in sel for i in (0, 1))
            mean = (sum(c["attempt_p_true_mean"] * sum(sum(r) for r in c["n_attempt_in_bin_by_user"])
                        for c in sel) / n) if n else float("nan")
            row[arm] = {"p_mean": float(mean),
                        "p_mean_by_user": [float(np.mean([c["attempt_p_true_mean_by_user"][i]
                                                          for c in sel])) for i in (0, 1)],
                        "n_attempt": int(n)}
        dist[f"ep{ep}"] = row
    frame_says = all(dist[f"ep{ep}"]["exact"]["p_mean"] < dist[f"ep{ep}"]["approx"]["p_mean"]
                     for ep in EPS)
    contradicts = (verdict in ("H-A-FRAME-SELECTION", "BOTH-MECHANISMS") and not frame_says)

    # ---- 逐格 Δr（配对，描述）
    per_cell = []
    for ep in EPS:
        for seed in SEEDS:
            e = next(c for c in cells if c["arm"] == "exact" and c["ep"] == ep and c["seed"] == seed)
            a = next(c for c in cells if c["arm"] == "approx" and c["ep"] == ep and c["seed"] == seed)
            ne, na = sum(map(sum, e["n_attempt_in_bin_by_user"])), sum(map(sum, a["n_attempt_in_bin_by_user"]))
            de_, da = sum(map(sum, e["n_delivered_in_bin_by_user"])), sum(map(sum, a["n_delivered_in_bin_by_user"]))
            per_cell.append({"ep": ep, "seed": seed,
                             "rate_exact": de_ / ne, "rate_approx": da / na,
                             "dr": de_ / ne - da / na, "delivered_exact": de_, "delivered_approx": da})

    n_fail = (gate["n_fail"] + sum(1 for c in cells if not c["identities_pass"])
              + (0 if ident["all_identical"] else 1) + (0 if ctrl["all_pass"] else 1)
              + sum(1 for res in dec.values() for k, v in res.items()
                    if k != "_verdict" and (v.get("status") != "OK"
                                            or v["additivity_residual"] > 1e-12)))
    summary = {
        "protocol": "CW-POMDP-v1.1",
        "stage": "CW-R1.15",
        "kind": "DELIVERY-GAP-ATTRIBUTION",
        "prereg": "docs/research/CW-R1.15-送达缺口归因-预注册-v0.1.md",
        "verdict": verdict,
        "verdict_primary_10bins": primary,
        "verdict_5bins": secondary,
        "resolution_sensitive": bool(resolution_sensitive),
        "contradicts_distribution": bool(contradicts),
        "frame_says_exact_lower_p": bool(frame_says),
        "M1_identity": {"all_identical": bool(ident["all_identical"]),
                        "detail": f"{ident['n_identical']}/{ident['n_cells']}"},
        "CTRL": ctrl,
        "DEC": {r: {k: (v if k == "_verdict" else
                        {kk: v[kk] for kk in ("status", "dr", "E_user", "E_frame", "E_within",
                                              "shares", "additivity_residual", "n_empty_cells",
                                              "n_attempt_e", "n_attempt_a", "rate_e", "rate_a")
                         if kk in v})
                   for k, v in per.items()} for r, per in dec.items()},
        "DEC_CLS": {r: {k: v["cls"] for k, v in per.items() if k != "_verdict"}
                    for r, per in dec.items()},
        "distribution": dist,
        "per_cell_dr": per_cell,
        "gates": {"n_checks": gate["n_checks"], "n_fail": gate["n_fail"]},
        "n_cells": len(cells),
        "head": head, "src_hash": srchash,
        "runner_py_worktree_sha": wt_sha, "runner_py_blob_sha": blob_sha,
        "git_dirty_before_run": dirty.splitlines(),
        "note": ("两臂只差 ACK/NACK 的信息模型；同 seed 是配对不是独立样本；"
                 "有效独立单元 = 2 个 episode。`slot` 配对下同一 (t,i) 的 `p_true` 与丢包硬币"
                 "与策略无关 ⇒ 送达率差只能来自「在哪些 (t,i) 上发了」。"),
    }
    (outdir / "cells.json").write_text(json.dumps(cells, ensure_ascii=False, indent=1,
                                                  default=float), encoding="utf-8")
    (outdir / "decomp.json").write_text(json.dumps({"DEC": summary["DEC"],
                                                    "DEC_CLS": summary["DEC_CLS"],
                                                    "CTRL": ctrl},
                                                   ensure_ascii=False, indent=1, default=float),
                                        encoding="utf-8")
    (outdir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1,
                                                    default=float), encoding="utf-8")
    try:
        import psutil  # type: ignore
        rss = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
    except Exception:
        rss = None
    meta = {"command": "python -m experiments.cw_v1.r1_15_delivery_gap --tag " + str(args.tag),
            "python": os.sys.version.split()[0], "numpy": np.__version__,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "head": head, "src_hash": srchash,
            "runner_worktree_sha": wt_sha, "runner_blob_sha": blob_sha,
            "gate_elapsed_s": gate_s, "ctrl_elapsed_s": ctrl_s,
            "total_elapsed_s": round(time.perf_counter() - t_start, 1),
            "peak_rss_mb": round(rss, 1) if rss else None,
            "n_fail_gates": gate["n_fail"], "n_checks_gates": gate["n_checks"]}
    (outdir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    print(f"[DONE] {outdir.name} n_fail={n_fail} total={meta['total_elapsed_s']}s "
          f"peak_rss={meta['peak_rss_mb']}MB")
    print(json.dumps({"verdict": verdict, "primary": primary, "secondary": secondary,
                      "frame_lower_p": frame_says, "id": summary["M1_identity"]},
                     ensure_ascii=False, indent=1, default=float))
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    import sys
    sys.exit(main())
