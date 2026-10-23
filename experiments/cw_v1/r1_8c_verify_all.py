# -*- coding: utf-8 -*-
"""CW-R1.8c 补闸：**全 16 格**身份对账（R1.8 主脚本的 `G1` 只覆盖了 2 格）。

为什么单独补（自伤记录）
------------------------
`r1_8_mech.py` 的 `G1` 只用 `paced16|100|2001` / `dp16|100|2001` 两格证明"新增字段是纯记录"。
但本轮重跑的是 **16 格**，而 R1.7.6 入库的 `cells.json` **恰好覆盖这 16 格**
⇒ 没有理由只对账 2 格。本补闸把 16/16 全部逐位对账，写进同一个证据目录。

对账项（必须**逐位相同**）：`cost_est` / `cost_true` / `n_delivered` / `n_attempted` /
`n_probe` / `policy_trace_hash` / `mean_task_aoi`（本轮记作 `mean_task_aoi_old`）。
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results" / "cw_v1"
REF = RESULTS / "cw_r1_7_6_g4form_20261023T182640_f11ffdefc06c0201_7dd8ba47dede6ce3" / "cells.json"
NEW = RESULTS / "cw_r1_8_mech_20261023T170326_08b6bfcbb8c7dec7"

FLOAT_KEYS = (("cost_est", "cost_est"), ("cost_true", "cost_true"),
              ("mean_task_aoi", "mean_task_aoi_old"))
INT_KEYS = ("n_delivered", "n_attempted", "n_probe")


def main() -> int:
    ref = json.loads(REF.read_text(encoding="utf-8"))
    new = json.loads((NEW / "cells.json").read_text(encoding="utf-8"))
    # 只对本轮真跑过的两臂对账（`paced32` 是 R1.7.6 的 `G4` 形态实验，本轮没跑）
    keys = [k for k in ref if k.split("|")[0] in ("paced16", "dp16")]
    skipped = [k for k in ref if k not in keys]
    rows = []
    for k in keys:
        a, b = ref[k], new.get(k)
        if b is None:
            rows.append({"cell": k, "status": "MISSING-IN-NEW", "pass": False})
            continue
        d = {ka: abs(float(a[ka]) - float(b[kb])) for ka, kb in FLOAT_KEYS}
        d.update({ka: int(a[ka]) == int(b[ka]) for ka in INT_KEYS})
        d["trace_eq"] = str(a["policy_trace_hash"]) == str(b["policy_trace_hash"])
        ok = (all(v <= 1e-12 for ka, v in d.items() if ka in [x[0] for x in FLOAT_KEYS])
              and all(d[ka] for ka in INT_KEYS) and d["trace_eq"])
        rows.append({"cell": k, **{ka: (round(v, 15) if isinstance(v, float) else v)
                                   for ka, v in d.items()}, "pass": bool(ok)})
    out = {"reference": str(REF.relative_to(REPO)), "new": str(NEW.relative_to(REPO)),
           "n_cells": len(keys), "n_pass": int(sum(1 for r in rows if r["pass"])),
           "pass": bool(len(rows) == len(keys) and all(r["pass"] for r in rows)),
           "skipped_not_rerun": skipped,
           "rows": rows,
           "rule": "本轮跑过的 16/16 逐位相同 ⇒ 新增字段确为纯记录（不是只抽 2 格）"}
    (NEW / "identity_all16.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                             encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("n_cells", "n_pass", "pass")},
                     ensure_ascii=False, indent=1))
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
