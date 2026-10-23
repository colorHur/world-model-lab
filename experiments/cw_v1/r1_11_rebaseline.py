# -*- coding: utf-8 -*-
"""CW-R1.11 逐用户 ACK 归属修正 —— **新基线**。

预注册：`docs/research/CW-R1.11-归属修正新基线-预注册-v0.1.md`

三段
----
NEG **活体阴性对照**：把 `_ack_bump` 的**实参来源**换回旧实现的泄漏值（`n_users-1`），
  不改任何常量、不改被测函数体 ⇒ 走的是真的旧路径。要求 4j-2 FAIL（有分辨力），
  并**如实报告** 4l 在负控下仍然 PASS（⇒ 4l 对实参缺陷无分辨力，只封计数归属）。
PANEL **16 格**：`ep ∈ {100,101} × seed ∈ {2001..2004} × ack=on`，与 R1.8/R1.9/R1.10 面板同参。
TAB **新旧分表**：本表与 `cw_r1_10_lockin_*` 的旧读数并列，表头写明跨版本不可逐位比。
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
from . import world as W
from .runner import check_identities, run_episode

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results" / "cw_v1"

RHO = 0.98
H_PLAN = 5
M_MAIN = 16
RATIO = 0.5
EPS = (100, 101)
SEEDS = (2001, 2002, 2003, 2004)
TAIL_DP = {"tail": "dp", "dp_grid": "r173"}


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


# ------------------------------------------------------------------ 记录器
class _Rec(Pmodule.PlanRef):
    """只**观测**：逐槽公开 telemetry + 根节点值；不改任何内层行为。"""

    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)
        self.rows: list[dict] = []
        self.picks: list[dict] = []

    def _pick(self, values, budget_left):          # noqa: ANN001
        self.picks.append({repr(a): float(values[a]) for a in values})
        return super()._pick(values, budget_left)

    def __call__(self, v):                          # noqa: ANN001
        a = super().__call__(v)
        tel = v.telemetry
        mh = np.asarray([[float(r["h_mean"]) for r in tel]], dtype=float)
        vh = np.asarray([[float(r["h_var"]) for r in tel]], dtype=float)
        ph = self._p_hat(mh, vh)[0]
        self.rows.append({
            "t": int(v.t), "act": repr(a),
            "P": [float(r["P_hat"]) for r in tel],
            "phat": [float(x) for x in ph],
        })
        return a


def _cfg(ack: bool = True) -> C.CWConfig:
    base = C.CWConfig(plant=C.bounded_plant())
    return _dc.replace(base, channel=_dc.replace(base.channel, use_ack_for_csi=bool(ack)))


# ------------------------------------------------------------------ 单格
def run_cell(ep: int, seed: int, ack: bool = True) -> dict:
    cfg = _cfg(ack)
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
    ident_ok = all(att[i] == dlv[i] + nk[i] for i in range(len(att)))
    return {
        "ep": int(ep), "seed": int(seed), "ack": bool(ack),
        "n_attempted_by_user": att, "n_delivered_by_user": dlv,
        "n_ack_by_user": ak, "n_nack_by_user": nk,
        "att_minus_deliv": [att[i] - dlv[i] for i in range(len(att))],
        "attribution_identity_ok": bool(ident_ok),
        "delta_attempted": int(att[0] - att[1]),
        "delta_delivered": int(dlv[0] - dlv[1]),
        "ratio_attempted": (round(att[0] / att[1], 3) if att[1] > 0 else None),
        "n_probe": int(res.n_probe), "n_hold": int(res.n_idle),
        "n_forced_hold": int(res.n_forced_hold),
        "n_never_served": int(res.n_never_served),
        "aoi_tavg_by_user": res.aoi_tavg_by_user,
        "aoi_max_by_user": res.aoi_max_by_user,
        "cost_est_per_step": res.cost_est_per_step,
        "cost_true_per_step": res.cost_true_per_step,
        "cost_true_by_user": res.cost_true_by_user,
        "phat_diff_median": (float(np.median(dphi)) if dphi else None),
        "phat_diff_max_abs": (float(np.max(np.abs(dphi))) if dphi else None),
        "policy_trace_hash": res.policy_trace_hash,
        "identities_pass": bool(chk["pass"]),
        "identities_fail": [k for k, v in chk["items"].items() if not v],
        "elapsed_s": round(elapsed, 2),
    }


# ------------------------------------------------------------------ 活体阴性对照
_ORIG_BUMP = W.CWWorld._ack_bump


def _legacy_bump(self, i: int, direction: float) -> None:
    """★ 复现旧实现的**实参来源**（`world.py:428` 循环泄漏的 `i = n_users-1`）。

    只改实参从哪里来，函数体、常量、调用点一字不动 ⇒ 这是**活体**负控（台账 #95），
    不是"把某个阈值调回旧值"。
    """
    if direction < 0:
        i = self.n_users - 1
    return _ORIG_BUMP(self, i, direction)


def _gate_subset() -> dict:
    """只跑反例 4（含 4j/4k/4l），返回逐项结果。"""
    g = G.Gate()
    G.counterexample_4(C.default_config(), g)
    return {r["name"]: bool(r["pass"]) for r in g.rows}


def negative_control() -> dict:
    before = _gate_subset()
    W.CWWorld._ack_bump = _legacy_bump
    try:
        during = _gate_subset()
    finally:
        W.CWWorld._ack_bump = _ORIG_BUMP
    assert W.CWWorld._ack_bump is _ORIG_BUMP, "负控补丁未还原"
    after = _gate_subset()

    def pick(d: dict, key: str) -> bool | None:
        for k, v in d.items():
            if k.startswith(key):
                return v
        return None

    return {
        "fixed": before,
        "legacy": during,
        "restored": after,
        "restored_identical": before == after,
        "NEG_4j2_discriminates": (pick(before, "4j-2") is True) and (pick(during, "4j-2") is False),
        "NEG_4k_no_power": {"before": pick(before, "4k"), "legacy": pick(during, "4k"),
                            "reading": "旧实现在 4k 上恰好正确 ⇒ 单独无分辨力（预期）"},
        "NEG_4l_no_power_on_arg": {"before": pick(before, "4l"), "legacy": pick(during, "4l"),
                                   "reading": "4l 在负控下仍 PASS ⇒ 它只封计数归属，"
                                              "**不能**证明 `_ack_bump` 实参正确（与预注册 §3 一致）"},
        "NEG_4g_discriminates": (pick(before, "4g") is True) and (pick(during, "4g") is False),
    }


# ------------------------------------------------------------------ 主流程
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="CW-R1.11 归属修正新基线（预注册 v0.1）")
    ap.add_argument("--probe", action="store_true", help="只跑 1 格报预算，**不写任何证据**")
    ap.add_argument("--tag", default="main")
    args = ap.parse_args(argv)

    if args.probe:
        t0 = time.perf_counter()
        c = run_cell(100, 2001, True)
        print(json.dumps({"probe_cell": c, "elapsed_total_s": round(time.perf_counter() - t0, 1)},
                         ensure_ascii=False, indent=1, default=float))
        return 0

    t_start = time.perf_counter()
    head = _git(["rev-parse", "HEAD"])
    wt_sha = _git(["hash-object", "experiments/cw_v1/world.py"])
    blob_sha = _git(["rev-parse", f"{head}:experiments/cw_v1/world.py"])
    dirty = _git(["status", "--porcelain"])
    srchash = _src_hash()
    stamp = _now_utc()
    outdir = RESULTS / f"cw_r1_11_rebaseline_{stamp}_{srchash}"
    outdir.mkdir(parents=True, exist_ok=False)
    print(f"[OUT] {outdir}")

    # ---- NEG
    print("[NEG] 活体阴性对照（旧实参来源）...")
    t1 = time.perf_counter()
    neg = negative_control()
    neg_s = round(time.perf_counter() - t1, 2)
    print(f"[NEG] 4j-2 有分辨力={neg['NEG_4j2_discriminates']} "
          f"4g 有分辨力={neg['NEG_4g_discriminates']} "
          f"4l 负控下仍 PASS={neg['NEG_4l_no_power_on_arg']['legacy']} ({neg_s}s)")
    (outdir / "negative_control.json").write_text(
        json.dumps(neg, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- 闸（修复态全量）
    print("[GATE] 全量 eval_gates ...")
    gate = G.run_all()
    (outdir / "gates.json").write_text(json.dumps(gate, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
    print(f"[GATE] n_checks={gate['n_checks']} n_fail={gate['n_fail']}")

    # ---- PANEL（2 ep × 2 臂 × 4 seed = 16 格）
    cells: list[dict] = []
    for ep in EPS:
        for ack in (True, False):
            for seed in SEEDS:
                t0 = time.perf_counter()
                c = run_cell(ep, seed, ack)
                c["wall_s"] = round(time.perf_counter() - t0, 2)
                cells.append(c)
                print(f"[CELL] ep{ep}/ack={int(ack)}/seed{seed} "
                      f"att={c['n_attempted_by_user']} del={c['n_delivered_by_user']} "
                      f"nack={c['n_nack_by_user']} cost_true={c['cost_true_per_step']:.4f} "
                      f"({c['wall_s']}s)")

    # ---- 判据 P1..P4（+ P2b 事后追加，见预注册 §7 修订区）
    on = [c for c in cells if c["ack"]]
    p1 = [c for c in on if c["ep"] == 100 and c["seed"] == 2001][0]["phat_diff_median"]
    p2_hits = sum(1 for c in cells
                  if c["n_attempted_by_user"][1] >= c["n_attempted_by_user"][0] / 2.0)
    p4_ok = sum(1 for c in cells if c["attribution_identity_ok"])
    p3 = {}
    for ep in EPS:
        s_on = [c for c in cells if c["ep"] == ep and c["ack"]]
        s_off = [c for c in cells if c["ep"] == ep and not c["ack"]]
        m_on = float(np.median([c["cost_true_per_step"] for c in s_on]))
        m_off = float(np.median([c["cost_true_per_step"] for c in s_off]))
        p3[f"ep{ep}"] = {
            "cost_true_median_ack_on(new)": m_on,
            "cost_true_median_ack_off(new)": m_off,
            "delta_pct_on_vs_off": round(100.0 * (m_on - m_off) / m_off, 2) if m_off else None,
            "old_R110_ack_on": 0.1164 if ep == 100 else 0.1885,
            "old_R110_ack_off": 0.1025 if ep == 100 else 0.1010,
        }
    # P2b（**事后追加**，预注册 §7 留痕）：按 episode 分组看方向。
    #   依据 R1.7.7 的硬约束「逐用户量先查角色：ep100/ep101 的贵用户会互换」——
    #   P2 的原始形式（u1 的被服务量 ≥ u0 的一半）会把**物理角色差异**当成**下标偏好**。
    #   ⇒ 正确的问题是"方向是否由**下标**决定"，而不是"u1 是否被饿死"。
    p2b = {}
    for ep in EPS:
        sub = [c for c in on if c["ep"] == ep]
        ds = [c["delta_attempted"] for c in sub]
        signs = [1 if d > 0 else (-1 if d < 0 else 0) for d in ds]
        p2b[f"ep{ep}"] = {"delta_attempted": ds, "signs": signs,
                          "unanimous": len(set(signs)) == 1,
                          "preferred": ("u0" if signs[0] > 0 else "u1") if signs else None}
    p2b["cross_episode_preference_differs"] = (
        p2b["ep100"]["preferred"] != p2b["ep101"]["preferred"])
    p2b["reading"] = ("是 ⇒ 方向由 **episode**（物理角色）决定，不由下标决定 ⇒ "
                      "缺陷已消失（旧状态下两集都是 u1 被饿死）"
                      if p2b["cross_episode_preference_differs"]
                      else "否 ⇒ 两集同向 ⇒ 仍存在下标级偏好，需重开")
    verdicts = {
        "P1_abs_phat_diff_median_lt_0.05": {"value": p1, "pass": bool(p1 is not None and abs(p1) < 0.05)},
        "P2_attempt_balance_ge_12_of_16": {"hits": p2_hits, "n": len(cells), "pass": p2_hits >= 12,
                                           "★": "本条形式上与「逐用户量先查角色」冲突，见 P2b"},
        "P2b_episode_determined": p2b,
        "P3_cost_true_vs_same_version_ack_off": p3,
        "P4_attribution_identity_16_of_16": {"hits": p4_ok, "n": len(cells), "pass": p4_ok == len(cells)},
        "identities_all_pass": {"hits": sum(1 for c in cells if c["identities_pass"]), "n": len(cells)},
    }

    summary = {
        "protocol": "CW-POMDP-v1.1",
        "stage": "CW-R1.11",
        "baseline_declaration": "本表 = **新基线**（world.py NACK 归属已修）；与 R1.10 及更早的 "
                                "`ack=on` 数字**跨版本不可逐位比**",
        "head": head, "world_py_worktree_sha": wt_sha, "world_py_blob_sha": blob_sha,
        "src_hash": srchash, "git_dirty_before_run": dirty.splitlines(),
        "n_cells": len(cells), "verdicts": verdicts,
        "old_reference": {"result_A_ack_off": {"ep100_cost_true": 0.1025, "ep101_cost_true": 0.1010},
                          "old_ack_on": {"ep100_cost_true": 0.1164, "ep101_cost_true": 0.1885}},
        "wall_s_total": round(time.perf_counter() - t_start, 1),
        "peak_rss_mb_note": "见 meta.json（psutil 若可用）",
    }
    (outdir / "cells.json").write_text(json.dumps(cells, ensure_ascii=False, indent=1,
                                                  default=float), encoding="utf-8")
    (outdir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1,
                                                    default=float), encoding="utf-8")

    # ---- meta
    try:
        import psutil  # type: ignore
        rss = psutil.Process(os.getpid()).memory_info().rss / 1024**2
    except Exception:
        rss = None
    meta = {
        "command": "python -m experiments.cw_v1.r1_11_rebaseline --tag " + str(args.tag),
        "python": os.sys.version.split()[0],
        "numpy": np.__version__,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "head": head, "src_hash": srchash,
        "world_worktree_sha": wt_sha, "world_blob_sha": blob_sha,
        "neg_elapsed_s": neg_s, "total_elapsed_s": round(time.perf_counter() - t_start, 1),
        "peak_rss_mb": round(rss, 1) if rss else None,
        "n_fail_gates": gate["n_fail"], "n_checks_gates": gate["n_checks"],
        "n_fail_cells": sum(1 for c in cells if not c["identities_pass"]),
    }
    (outdir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                      encoding="utf-8")

    n_fail = gate["n_fail"] + sum(1 for c in cells if not c["identities_pass"])
    print(f"[DONE] {outdir.name} n_fail={n_fail} total={meta['total_elapsed_s']}s "
          f"peak_rss={meta['peak_rss_mb']}MB")
    print(json.dumps(verdicts, ensure_ascii=False, indent=1, default=float))
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    import sys
    sys.exit(main())
