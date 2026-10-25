# -*- coding: utf-8 -*-
"""`CW-R2.5b`：给 R2.5 补 **R17 逐用户成本**（**零 episode**，只读已落盘证据）。

为什么需要
----------
R2.5 主判只用了陈旧度 `A_t`，而 R2.4 的 R17 结论是"陈旧度 60/60 改善，但成本 `r_i>1` 者 41/60"
⇒ **只能写交易**。R2.5 换了规划器种子，`r_i` 的分布**必须重算**，否则"种子稳健"这句话
会被误读成"换种子后不再有成本代价"。

定义（与 R2.4 同口径，不新造量）
--------------------------------
`r_i = cost_true_by_user_i(prior_only) / cost_true_by_user_i(belief_readout)`，**同种子同 episode**。
`r_i > 1` ⇒ 该用户成本上升。

★ 两个种子**分开报**，不做合并（合并会让"哪个种子更贵"变得不可见）。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
R24 = REPO / "results" / "cw_v1" / "cw_r2_4_aoi_20261025T023636_a83942167173392b"
R25 = REPO / "results" / "cw_v1" / "cw_r2_5_seed_20261025T045209_f00b22682d18e97f"


def _load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def _pair(strong: dict, weak: dict) -> list[dict]:
    out = []
    for i, (cs, cw) in enumerate(zip(strong["cost_true_by_user"], weak["cost_true_by_user"])):
        cs, cw = float(cs), float(cw)
        out.append({"i": i, "cost_strong": cs, "cost_weak": cw,
                    "r_i": (cw / cs if cs > 0 else None),
                    "abs_delta_cost": cw - cs})
    return out


def main() -> int:
    c24 = _load(R24 / "cells.json")
    n24 = _load(R24 / "null.json")["nulls"]
    c25 = _load(R25 / "cells.json")
    s25 = _load(R25 / "summary.json")
    idx = {(c["plant"], c["model"], int(c["ep"])): c for c in c24}
    idxn = {(c["plant"], int(c["ep"])): c for c in n24}
    idx3 = {(c["plant"], c["model"], int(c["ep"])): c for c in c25}
    plants = ["P1", "P2"]
    eps = list(range(100, 115))

    table, unit_rows = {}, []
    for tag, strong_of, weak_of in (
            ("seed2001", lambda p, e: idx[(p, "belief_readout", e)],
             lambda p, e: idx[(p, "prior_only", e)]),
            ("seed3102", lambda p, e: idxn[(p, e)],
             lambda p, e: idx3[(p, "prior_only", e)])):
        per = []
        for p in plants:
            for e in eps:
                s, w = strong_of(p, e), weak_of(p, e)
                rs = _pair(s, w)
                for r in rs:
                    r.update({"plant": p, "ep": e, "seed": tag})
                per.extend(rs)
                unit_rows.append({"seed": tag, "plant": p, "ep": e,
                                  "n_user": len(rs),
                                  "n_r_gt_1": int(sum(1 for r in rs
                                                      if r["r_i"] is not None and r["r_i"] > 1)),
                                  "sum_cost_strong": float(sum(r["cost_strong"] for r in rs)),
                                  "sum_cost_weak": float(sum(r["cost_weak"] for r in rs))})
        rv = [r["r_i"] for r in per if r["r_i"] is not None]
        table[tag] = {
            "n_user_cells": len(per), "n_r_gt_1": int(sum(1 for x in rv if x > 1)),
            "median_r": float(np.median(rv)), "min_r": float(np.min(rv)), "max_r": float(np.max(rv)),
            "n_units_with_any_r_gt_1": int(sum(1 for u in unit_rows
                                               if u["seed"] == tag and u["n_r_gt_1"] > 0)),
            "units_total": len(eps) * len(plants),
            "sum_cost_strong": float(sum(u["sum_cost_strong"] for u in unit_rows if u["seed"] == tag)),
            "sum_cost_weak": float(sum(u["sum_cost_weak"] for u in unit_rows if u["seed"] == tag)),
            "per_user": per}

    # 一致性：R2.4 曾报 41/60；这里必须逐位复现，否则我的口径与 R2.4 不同
    reconcile = {"expected_2001": 41, "got_2001": table["seed2001"]["n_r_gt_1"],
                 "ok": table["seed2001"]["n_r_gt_1"] == 41}

    out = {"stage": "CW-R2.5b", "kind": "R17-PER-USER-COST-ADDENDUM",
           "inputs": {str(R24.name): "cells.json+null.json", str(R25.name): "cells.json"},
           "r17_by_seed": {k: {kk: vv for kk, vv in v.items() if kk != "per_user"}
                           for k, v in table.items()},
           "per_user_by_seed": {k: v["per_user"] for k, v in table.items()},
           "unit_rows": unit_rows,
           "reconcile_with_r24": reconcile,
           "verdict_language": (
               ("只写交易：陈旧度 60/60 用户改善，但成本 r_i>1 者 seed2001=%d/60、seed3102=%d/60"
                % (table["seed2001"]["n_r_gt_1"], table["seed3102"]["n_r_gt_1"]))
               + "。**不许**写'全面优于弱基线'。"),
           "training": "none", "n_episodes_run": 0}
    outdir = REPO / "results" / "cw_v1" / "cw_r2_5b_r17_20261025_readonly"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "r17.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float),
                                     encoding="utf-8")
    print("[R17] " + json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "per_user"}
                                for k, v in table.items()}, ensure_ascii=False, default=float))
    print("[RECONCILE] " + json.dumps(reconcile, ensure_ascii=False))
    print(f"[OUT] {outdir.name}")
    return 0 if reconcile["ok"] else 2


if __name__ == "__main__":
    import sys
    sys.exit(main())
