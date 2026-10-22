# -*- coding: utf-8 -*-
"""CW-R1.6.1 落盘入口：连续预算定价的闸 + 同一开发诊断面板（**新证据目录，不覆盖 R1.6**）。

与 `r1_6_plan.py` 的差别（就这三条）
----------------------------------
1. 跑的是 v1.6.1 的 `plan_ref`（`m` 连续 + 三处 `A²≠1` 解析修正）；
   `cont_m=False` 的口径**只**出现在 `G8d2` 阴性对照里，不进面板。
2. 判词按**有顺序的验收链**结算：`G8d → G8d2 → G8c2(+G8c2b) → M1 → G4`；
   链在第一个失败处停住，之后的项标 `not_reached`（不写"通过"）。
3. 证据前缀 `cw_r1_6_1_contm_*`，并额外落 `tail_math.json`（逐预算档的三种口径 + 旧实现）。

★ `kind` 仍是 `PLANNING-REFERENCE-NOT-UPPER-BOUND`：有限视界 + Monte-Carlo，
  **不是上界、不是最优策略**；8 集/格是开发诊断，不是确认集。
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from . import config as C
from .plan_gates import _legacy_paced_sumP, mutation_gates, run_all as run_plan_gates
from .plan_ref import make
from .r1_6_plan import (EPISODES, H_PLAN, M_SAMPLES, RESULTS, _git, _src_hash,
                        build_panel)

# 预注册的验收链（顺序不可跳；判据原文见
# docs/research/CW-R1.6.1-连续预算定价-预注册-v0.1.md §3）
CHAIN = (
    ("G8d", "值函数对预算严格递减（平坦 ≤10%、无上升）"),
    ("G8d2", "阴性对照：m 取整 ⇒ 平坦 >30%"),
    ("G8c2", "A²≠1 支算术：闭式 vs 模拟 ≤1%"),
    ("G8c2b", "阴性对照：旧实现同口径差 >10%"),
    ("M1", "变异 M1（探测不学但照扣钱）⇒ n_probe == 0"),
    ("G4", "M 16→32 动作一致率 ≥0.90 且成本相对差 ≤10%"),
)


def _budget_sweep() -> dict:
    """逐预算档的尾部估计（三种口径），供审计与作图。"""
    out = {}
    for tag, cfg in (("bounded_A1", C.CWConfig(plant=C.bounded_plant())),
                     ("r0_A_gt_1", C.CWConfig())):
        pol = make(cfg, arm="probe", h_plan=1, m_samples=1)
        n = int(pol.A.size)
        steps, P0 = 128, 0.01
        lines = {}
        for mode in ("cont", "rounded", "legacy"):
            vals = []
            for b in range(0, steps + 1):
                Pmat = np.full((1, n), P0)
                if mode == "legacy":
                    v = _legacy_paced_sumP(pol, Pmat, steps, np.array([float(b)]))
                else:
                    p2 = make(cfg, arm="probe", h_plan=1, m_samples=1,
                              cont_m=(mode == "cont"))
                    v = p2._paced_sumP(Pmat, steps, np.array([float(b)]))
                vals.append(float(v[0][0]))
            d = np.diff(vals)
            lines[mode] = {"values": [round(x, 10) for x in vals],
                           "n_flat": int(np.sum(np.abs(d) <= 1e-12)),
                           "n_up": int(np.sum(d > 1e-12)),
                           "n_pairs": int(d.size)}
        out[tag] = {"steps": steps, "P0": P0, "n_users": n, "modes": lines}
    return out


def _settle_chain(g, mut_rows, mut_missed: int) -> dict:
    rows = {r["name"].split(" ")[0]: r for r in g.rows}
    got = {k: bool(rows.get(k, {}).get("pass", False)) for k, _ in CHAIN if k != "M1"}
    got["M1"] = bool(mut_missed == 0 and all(r["caught"] for r in mut_rows
                                             if r["target"] == "G5a"))
    chain, stopped = [], None
    for name, desc in CHAIN:
        if stopped is None:
            state = "pass" if got[name] else "fail"
            if not got[name]:
                stopped = name
        else:
            state = "not_reached"
        chain.append({"gate": name, "desc": desc, "state": state,
                      "observed": got[name]})
    return {"chain": chain, "stopped_at": stopped,
            "all_pass": stopped is None,
            "eligible_for_N1N2": stopped is None,
            "rule": "链在第一个 fail 停住；not_reached 一律**不写**成通过",
            "n_gate_fail": g.summary()["n_fail"],
            "n_mutation_missed": mut_missed,
            "mutation_rows": mut_rows}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="CW-R1.6.1 连续预算定价：闸 + 开发诊断面板")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--skip-panel", action="store_true")
    ap.add_argument("--note", default="")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    g = run_plan_gates()
    mut_rows, mut_missed = mutation_gates()
    n_fail = g.summary()["n_fail"] + mut_missed
    print(f"[CW-R1.6.1] plan_gates: {g.summary()['n_checks']} checks, "
          f"{g.summary()['n_fail']} fail")
    for r in g.rows:
        if not r["pass"]:
            print("  FAIL ", r["name"])
    print(f"[CW-R1.6.1] 变异闸: {len(mut_rows)} 次破坏, {mut_missed} 次未被抓 "
          f"(M1 evidence: {[r['evidence'] for r in mut_rows if r['target'] == 'G5a']})")

    verdict = _settle_chain(g, mut_rows, mut_missed)
    print(f"[CW-R1.6.1] 验收链停止于: {verdict['stopped_at']}；"
          f"eligible_for_N1N2={verdict['eligible_for_N1N2']}")

    panel = None if args.skip_panel else build_panel()

    sweep = _budget_sweep()
    elapsed = round(time.perf_counter() - t0, 2)

    if not args.no_write:
        now = datetime.now(timezone(timedelta(hours=8)))
        cfg = C.CWConfig(plant=C.bounded_plant())
        run_id = (f"cw_r1_6_1_contm_{now.strftime('%Y%m%dT%H%M%S')}_"
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
        _dump("tail_math.json", sweep)
        _dump("meta.json", {
            "run_id": run_id,
            "kind": "PLANNING-REFERENCE-NOT-UPPER-BOUND",
            "protocol": C.CWConfig.PROTOCOL_VERSION,
            "pacing": True, "cont_m": True,
            "event_pairing": cfg.protocol.event_pairing,
            "h_plan": H_PLAN, "m_samples": M_SAMPLES,
            "episodes": list(EPISODES), "config": cfg.to_dict(),
            "note": args.note,
            "at_asia_shanghai": now.isoformat(), "elapsed_s": elapsed,
            "git_head": _git(["rev-parse", "HEAD"]),
            "git_status_porcelain": _git(["status", "--porcelain"]),
            "src_hash": _src_hash(),
            "supersedes": ("results/cw_v1/cw_r1_6_pacing_* 的 planning 面板"
                           "（那份是 m 取整 + A²≠1 支算术错的实现）"),
            "disclaimer": (
                "有限视界 + Monte-Carlo 规划参考，不是上界、不是最优策略；"
                "8 集/格是开发诊断，不是确认集。本轮只改尾部估计的算术与连续性，"
                "不声称「主动测量有价值/无价值」。"),
        })
        print(f"[CW-R1.6.1] 证据落盘 results/cw_v1/{run_id}")

    print(f"[CW-R1.6.1] total {n_fail} fail / "
          f"{g.summary()['n_checks'] + len(mut_rows)} checks ({elapsed} s)")
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
