# -*- coding: utf-8 -*-
"""CW-R1.6 落盘入口：配速修正的闸 + 开发诊断面板（**新证据目录，不覆盖 R1.5**）。

与 `r1_5_plan.py` 的差别（只有这两条）
------------------------------------
1. 面板里的两个规划臂用 `pacing=True`（v1.6 默认）：尾部估计把**剩余预算的机会成本**
   算进去（`plan_ref._paced_sumP`）。R1.5 的证据是 `pacing=False` 口径，两组**不得混表**。
2. 面板/证据前缀改成 `cw_r1_6_pacing_*`，并额外记录 T1–T5 五条预注册判据。

★ 本 run 的 `kind` 仍是 `PLANNING-REFERENCE-NOT-UPPER-BOUND`：有限视界 + Monte-Carlo，
  **不是上界、不是最优策略**。

★ 面板可分离计算（`--panel-from`）：面板耗时远大于闸，允许先单独跑面板落成 JSON，
  再由正式入口把「闸 + 该面板」合成一份证据。合成时校验 `src_hash`/`m_samples`/`episodes`
  与本次一致，不一致直接拒绝（防止拿改造前的面板冒充）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from . import config as C
from .evaluate import compare_cell, required_cells_from_design
from .plan_gates import _cv, _send_profile, mutation_gates, run_all as run_plan_gates
from .plan_ref import make
from .policies import PacedAge, PeriodicSend
from .runner import check_identities, run_episode

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results" / "cw_v1"

RHOS = (0.85, 0.98)
RATIOS = (0.5, 0.8)
EPISODES = tuple(range(100, 108))
H_PLAN = 5
M_SAMPLES = 32


def _src_hash() -> str:
    h = hashlib.sha256()
    for p in sorted((REPO / "experiments" / "cw_v1").glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def _git(args: list[str]) -> str:
    import subprocess

    return subprocess.run(["D:\\devtools\\Git\\cmd\\git.exe", *args],
                          cwd=str(REPO), capture_output=True, text=True).stdout.strip()


class _Rec:
    def __init__(self, inner) -> None:
        self.inner = inner
        self.name = getattr(inner, "name", "rec")
        self.acts: list[tuple[int, str, int | None]] = []

    def __call__(self, v):
        a = self.inner(v)
        self.acts.append((int(v.t), a.kind, a.user))
        return a


def _run(cfg: C.CWConfig, pol, *, budget: int, rho: float, ep: int) -> dict:
    rec = _Rec(pol)
    r = run_episode(cfg, rec, split="dev", episode=ep, budget_total=budget, rho=rho)
    ids = check_identities(r, cfg)
    return {
        "episode": ep,
        "cost_true": r.cost_true_per_step,
        "cost_est": r.cost_est_per_step,
        "n_probe": r.n_probe, "n_attempted": r.n_attempted,
        "n_delivered": r.n_delivered, "n_idle": r.n_idle,
        "n_forced_hold": r.n_forced_hold,
        "budget_used": r.budget_used, "budget_wasted": r.budget_wasted,
        "mean_task_aoi": r.mean_task_aoi, "mean_csi_aoi": r.mean_csi_aoi,
        "cost_true_by_user": list(r.cost_true_by_user),
        "send_by_window": _send_profile(rec.acts, cfg.protocol.horizon),
        "trace_hash": r.policy_trace_hash,
        "identities_pass": ids["pass"],
        "identities_fail": [k for k, v in ids.items() if k != "pass" and not v],
        "planner_probe_same_argmax": int(getattr(rec.inner, "probe_same_argmax", 0)),
        "planner_probe_diff_argmax": int(getattr(rec.inner, "probe_diff_argmax", 0)),
    }


def build_panel() -> dict:
    cfg = C.CWConfig(plant=C.bounded_plant())
    out: dict = {"cells": [], "config": cfg.to_dict(),
                 "h_plan": H_PLAN, "m_samples": M_SAMPLES,
                 "episodes": list(EPISODES), "src_hash": _src_hash(),
                 "arms": ["plan_probe", "plan_noprobe", "periodic2", "paced_age8"]}
    for ratio in RATIOS:
        budget = cfg.protocol.budget(ratio)
        for rho in RHOS:
            cell = {"rho": rho, "budget_ratio": ratio, "budget": int(budget), "arms": {}}
            mk = {
                "plan_probe": lambda ep, rho=rho: make(cfg, arm="probe", h_plan=H_PLAN,
                                                      m_samples=M_SAMPLES,
                                                      seed=1000 + ep, rho=rho, pacing=True),
                "plan_noprobe": lambda ep, rho=rho: make(cfg, arm="noprobe", h_plan=H_PLAN,
                                                        m_samples=M_SAMPLES,
                                                        seed=1000 + ep, rho=rho,
                                                        pacing=True),
                "periodic2": lambda ep, rho=rho: PeriodicSend(2),
                "paced_age8": lambda ep, rho=rho: PacedAge(8),
            }
            for name, f in mk.items():
                cell["arms"][name] = [_run(cfg, f(ep), budget=budget, rho=rho, ep=ep)
                                      for ep in EPISODES]
            cand = [r["cost_true"] for r in cell["arms"]["plan_probe"]]
            cell["compare"] = {}
            for base in ("plan_noprobe", "periodic2", "paced_age8"):
                b = [r["cost_true"] for r in cell["arms"][base]]
                cell["compare"][base] = compare_cell(
                    b, cand, cell=(round(rho, 2), ratio, base)).as_dict()
            cell["send_profile"] = {
                name: np.mean([r["send_by_window"] for r in rows], axis=0).round(3).tolist()
                for name, rows in cell["arms"].items()
            }
            cell["profile_cv"] = {"plan_probe": _cv(np.mean(
                [r["send_by_window"] for r in cell["arms"]["plan_probe"]], axis=0).tolist())}
            out["cells"].append(cell)
            print(f"  cell rho={rho} B/T={ratio} 计划器={np.median(cand):.4f} "
                  f"periodic2={np.median([r['cost_true'] for r in cell['arms']['periodic2']]):.4f} "
                  f"aoi={np.median([r['mean_task_aoi'] for r in cell['arms']['plan_probe']]):.1f}",
                  flush=True)
    return out


# ---------------------------------------------------------------- 判据 T1–T5
def _t5(panel: dict) -> dict:
    per = []
    for cell in panel["cells"]:
        c = cell["compare"]["periodic2"]
        per.append({"rho": cell["rho"], "ratio": cell["budget_ratio"],
                    "median_pairwise_ratio": c["median_pairwise_ratio"],
                    "status": c["status"],
                    "plan_median": c["candidate_median"],
                    "periodic2_median": c["baseline_median"]})
    ok = all(p["status"] == "ok" and np.isfinite(p["median_pairwise_ratio"])
             and p["median_pairwise_ratio"] <= 1.00 for p in per)
    return {"pass": bool(ok), "n_cells": len(per),
            "required_cells": required_cells_from_design(per), "cells": per,
            "rule": "四格 median_pairwise_ratio(plan_probe / periodic2) ≤ 1.00"}


def _verdict(g, panel: dict | None, mut_missed: int) -> dict:
    rows = {r["name"].split(" ")[0]: r for r in g.rows}
    t1 = rows.get("G8a", {}).get("pass", False)
    t2 = rows.get("G8b", {}).get("pass", False)
    t3 = rows.get("G8a2", {}).get("pass", False)
    t4 = rows.get("G4", {}).get("pass", False)
    t5 = _t5(panel) if panel else None
    return {
        "T1_pacing_cv": t1, "T2_negative_control": t2, "T3_aoi": t3,
        "T4_m_stability": t4, "T5_vs_periodic2": (None if t5 is None else t5["pass"]),
        "T5_detail": t5,
        "eligible_for_N1N2": bool(t4),
        "reason": ("T4 通过 ⇒ 可以读「探测 vs 禁探测」" if t4 else
                   "T4（M 16→32 稳定）未过 ⇒ 规划器自身方差与待测效应不可分离，"
                   "本轮**不出** N1/N2 科学判词；只报工程可比性（T5）与残差修正（T1–T3）"),
        "n_gate_fail": g.summary()["n_fail"], "n_mutation_missed": mut_missed,
        "note": "T5 是工程可比性判据，不是「主动测量有价值」的判词",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="CW-R1.6 配速修正：闸 + 开发诊断面板")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--skip-panel", action="store_true")
    ap.add_argument("--panel-from", default=None, help="复用已算好的面板 JSON（校验 src_hash）")
    ap.add_argument("--note", default="")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    g = run_plan_gates()
    mut_rows, mut_missed = mutation_gates()
    n_fail = g.summary()["n_fail"] + mut_missed
    print(f"[CW-R1.6] plan_gates: {g.summary()['n_checks']} checks, {g.summary()['n_fail']} fail")
    for r in g.rows:
        if not r["pass"]:
            print("  FAIL ", r["name"])
    print(f"[CW-R1.6] 变异闸: {len(mut_rows)} 次破坏, {mut_missed} 次未被抓")

    if args.skip_panel:
        panel = None
    elif args.panel_from:
        panel = json.loads(Path(args.panel_from).read_text(encoding="utf-8"))
        for k, mine in (("src_hash", _src_hash()), ("m_samples", M_SAMPLES),
                        ("episodes", list(EPISODES)), ("h_plan", H_PLAN)):
            if panel.get(k) != mine:
                raise SystemExit(f"[FATAL] 面板口径不一致：{k}={panel.get(k)!r} vs {mine!r}")
        print(f"[CW-R1.6] 复用面板 {args.panel_from}（src_hash 校验通过）")
    else:
        panel = build_panel()

    elapsed = round(time.perf_counter() - t0, 2)
    verdict = _verdict(g, panel, mut_missed)
    print(f"[CW-R1.6] T1={verdict['T1_pacing_cv']} T2={verdict['T2_negative_control']} "
          f"T3={verdict['T3_aoi']} T4={verdict['T4_m_stability']} T5={verdict['T5_vs_periodic2']}")
    print(f"[CW-R1.6] 判词资格: {verdict['eligible_for_N1N2']} —— {verdict['reason']}")
    print(f"[CW-R1.6] total {n_fail} fail / "
          f"{g.summary()['n_checks'] + len(mut_rows)} checks ({elapsed} s)")

    if not args.no_write:
        now = datetime.now(timezone(timedelta(hours=8)))
        cfg = C.CWConfig(plant=C.bounded_plant())
        run_id = (f"cw_r1_6_pacing_{now.strftime('%Y%m%dT%H%M%S')}_"
                  f"{cfg.fingerprint()}_{_src_hash()}")
        out_dir = RESULTS / run_id
        out_dir.mkdir(parents=True, exist_ok=True)

        def _dump(name, obj):
            (out_dir / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2),
                                        encoding="utf-8")

        _dump("plan_gates.json", g.summary())
        _dump("mutation_gates.json", {"n_mutations": len(mut_rows),
                                      "n_missed": mut_missed, "rows": mut_rows})
        _dump("panel.json", panel or {})
        _dump("verdict.json", verdict)
        _dump("meta.json", {
            "run_id": run_id,
            "kind": "PLANNING-REFERENCE-NOT-UPPER-BOUND",
            "protocol": C.CWConfig.PROTOCOL_VERSION,
            "pacing": True,
            "event_pairing": cfg.protocol.event_pairing,
            "h_plan": H_PLAN, "m_samples": M_SAMPLES,
            "episodes": list(EPISODES), "rhos": list(RHOS), "ratios": list(RATIOS),
            "config": cfg.to_dict(), "note": args.note,
            "panel_from_cache": bool(args.panel_from),
            "at_asia_shanghai": now.isoformat(), "elapsed_s": elapsed,
            "git_head": _git(["rev-parse", "HEAD"]),
            "git_status_porcelain": _git(["status", "--porcelain"]),
            "src_hash": _src_hash(),
            "supersedes": "results/cw_v1/cw_r1_5_plan_* 的 planning 面板（pacing=False 口径）",
            "disclaimer": (
                "有限视界 + Monte-Carlo 规划参考，不是上界、不是最优策略；8 集/格是开发诊断，"
                "不是确认集。T4（M 16→32 稳定）未过 ⇒ 不出「测量值不值」的科学判词。"
                "与 R1.5 的 planning 数字不同口径（pacing=True），不得混表。"
            ),
        })
        print(f"[CW-R1.6] 证据落盘 results/cw_v1/{run_id}")

    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
