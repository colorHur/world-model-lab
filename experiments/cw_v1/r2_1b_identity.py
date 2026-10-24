# -*- coding: utf-8 -*-
"""CW-R2.1b：**跨轮身份**对账（零 episode，只读两份入库证据）。

为什么单独一个入口
------------------
`B3` 已证"`ModelPolicy(BeliefReadout)` ≡ 裸规划器"（同轮、同脚本内）。
但那只证明**包装层**不改数字。还需要一条更硬的：
**R2.1 的 `P1 / belief_readout / seed=2001` 是否就是 R1.16 的 `exact / seed=2001`？**

- 若是 ⇒ ① 本轮面板与 R1 面板是**同一个世界**（可比性成立）；
  ② 本轮"强基线"就是 R1 的既有策略，不是新造的东西；
  ③ 后续任何改动若破坏这条，会被这条对账立刻抓到。
- 参照物是**入库证据**（`cw_r1_16_n10_…/cells.json`），不重跑。
- `P2` 档无参照（`sigma_meas=0.05` 首次出现）⇒ 如实 `SKIP-NO-REFERENCE`，**不**用 P1 的结果外推。

★ 只读，不写 `results/` 之外的任何东西；不跑 `run_episode`（`n_run_episode = 0`）。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from .r1_14_exactbase import _git, _src_hash

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results" / "cw_v1"
R116 = RESULTS / "cw_r1_16_n10_20261024T105416_fc9f25e16d2c6307"

#: 对账字段（逐位，不设容差）
FIELDS = ("cost_true_per_step", "policy_trace_hash",
          "n_attempted_by_user", "n_delivered_by_user", "n_probe")


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


def _pick(cells: list[dict], **kw) -> dict | None:
    for c in cells:
        if all(c.get(k) == v for k, v in kw.items()):
            return c
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="CW-R2.1b 跨轮身份对账（零 episode）")
    ap.add_argument("--r2-dir", default=None, help="R2.1 证据目录名（默认取最新非 SUPERSEDED）")
    ap.add_argument("--tag", default="main")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    if args.r2_dir:
        r2dir = RESULTS / args.r2_dir
    else:
        cands = sorted([d for d in RESULTS.iterdir()
                        if d.is_dir() and d.name.startswith("cw_r2_1_sep_")
                        and "SUPERSEDED" not in d.name])
        if not cands:
            print("[ABORT] 找不到 R2.1 证据目录")
            return 3
        r2dir = cands[-1]

    r2 = json.loads((r2dir / "cells.json").read_text(encoding="utf-8"))
    r1 = json.loads((R116 / "cells.json").read_text(encoding="utf-8"))

    rows = []
    n_same = n = 0
    for plant in ("P1", "P2"):
        for ep in range(100, 110):
            c = _pick(r2, plant=plant, model="belief_readout", ep=ep, plan_seed=2001)
            if c is None:
                rows.append({"plant": plant, "ep": ep, "status": "MISSING-IN-R2.1"})
                continue
            ref = _pick(r1, ep=ep, seed=2001, arm="exact") if plant == "P1" else None
            if ref is None:
                rows.append({"plant": plant, "ep": ep,
                             "status": "SKIP-NO-REFERENCE" if plant != "P1" else "MISSING-IN-R1.16",
                             "why": ("P2（`sigma_meas=0.05`）在本仓首次出现，无同配置历史证据"
                                     if plant != "P1" else "参照证据缺该格")})
                continue
            diff = {}
            for f in FIELDS:
                a, b = ref.get(f), c.get(f)
                if isinstance(a, float):
                    same = abs(a - b) <= 0.0
                else:
                    same = a == b
                diff[f] = {"r1_16": a, "r2_1": b, "same": bool(same)}
            ok = all(v["same"] for v in diff.values())
            n += 1
            n_same += int(ok)
            rows.append({"plant": plant, "ep": ep, "status": "IDENTICAL" if ok else "DIFF",
                         "fields": diff})

    checked = [r for r in rows if r["status"] in ("IDENTICAL", "DIFF")]
    verdict = ("CROSS-ROUND-IDENTICAL" if n and n_same == n and
               all(r["status"] == "IDENTICAL" for r in checked) else
               ("CROSS-ROUND-DIFF" if n else "NO-COMPARABLE-CELL"))

    head = _git(["rev-parse", "HEAD"])
    dirty = _git(["status", "--porcelain"])
    srchash = _src_hash()
    outdir = RESULTS / f"cw_r2_1b_identity_{_now_utc()}_{srchash}"
    outdir.mkdir(parents=True, exist_ok=False)

    ident = {
        "kind": "CROSS-ROUND-IDENTITY-NOT-A-NEW-RESULT",
        "protocol": "CW-POMDP-v1.1",
        "r2_dir": r2dir.name, "r1_16_dir": R116.name,
        "fields": list(FIELDS), "tolerance": "0（逐位，不设容差）",
        "n_checked": n, "n_identical": n_same, "verdict": verdict,
        "scope": "仅 P1（`bounded_plant()`）与 R1.16 的 `exact` 臂同格同种子；"
                 "P2 无参照 ⇒ SKIP-NO-REFERENCE（不许用 P1 的结果外推 P2）",
        "rows": rows, "head": head, "src_hash": srchash,
        "n_run_episode": 0,
    }
    (outdir / "identity.json").write_text(
        json.dumps(ident, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    meta = {"command": "python -m experiments.cw_v1.r2_1b_identity --tag " + str(args.tag),
            "timestamp_local": datetime.now().astimezone().isoformat(),
            "python": os.sys.version.split()[0],
            "head": head, "src_hash": srchash, "git_dirty_before_run": dirty.splitlines(),
            "total_elapsed_s": round(time.perf_counter() - t0, 2), "n_run_episode": 0}
    (outdir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    print(f"[OUT] {outdir.name}")
    print(json.dumps({"verdict": verdict, "n_identical": n_same, "n_checked": n,
                      "skipped": sum(1 for r in rows if r["status"].startswith("SKIP"))},
                     ensure_ascii=False, indent=1))
    return 0 if verdict == "CROSS-ROUND-IDENTICAL" else 2


if __name__ == "__main__":
    import sys
    sys.exit(main())
