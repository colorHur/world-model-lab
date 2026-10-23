# -*- coding: utf-8 -*-
"""CW-R1.9c **对称状态上的下标偏好**（T8，post-hoc，report-only，零 episode）。

预注册：`docs/research/CW-R1.9-策略层对称-预注册-v0.1.md` §7 修订 4。

为什么还要测：T1 已证"在我选的两个状态上 rollout 逐位等变"，但真实面板里策略在
`P` **更低**时仍偏好发它（T6/T7 实测 `P0−P1` 中位 −0.045/−0.052，`n_send0:n_send1` = 115:13 / 121:7）。
若在**完全对称**的 plant + **完全对称**的公开状态上仍系统性偏好 `SEND(0)`，
那就说明 T1 的**状态覆盖不足**，下标依赖藏在别处。

不判 PASS/FAIL。
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import config as C
from .r1_9_symmetry import _Cap, _mk, _pub
from .world import SchedulerView

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results" / "cw_v1"
TOL = 1e-12

STATES = (
    ("t0_exact", {"tag": "t0_exact", "t": 0, "bud": 128, "P": (1.0, 1.0),
                  "x": (0.0, 0.0), "mh": (0.0, 0.0), "vh": (1.0, 1.0)}),
    ("P003", {"tag": "P003", "t": 8, "bud": 128, "P": (0.03, 0.03),
              "x": (0.0, 0.0), "mh": (0.0, 0.0), "vh": (1.0, 1.0)}),
    ("P006", {"tag": "P006", "t": 8, "bud": 128, "P": (0.06, 0.06),
              "x": (0.0, 0.0), "mh": (0.0, 0.0), "vh": (1.0, 1.0)}),
)
SEEDS = tuple(range(2001, 2009))


def main() -> int:
    t0 = time.perf_counter()
    cfg = C.CWConfig(plant=C.bounded_plant(A=(1.0, 1.0), sigma_w=(0.025, 0.025)))
    rows = []
    for tail in ("paced", "dp"):
        for tag, st in STATES:
            ds = []
            for sd in SEEDS:
                pol = _mk(cfg, sd, None if tail == "paced" else "dp")
                pol(SchedulerView(_pub(st)))
                v0, v1 = pol.root["SEND(0)"], pol.root["SEND(1)"]
                ds.append({"seed": sd, "v0": v0, "v1": v1, "d": v1 - v0})
            n0 = sum(1 for r in ds if r["d"] > TOL)
            n1 = sum(1 for r in ds if r["d"] < -TOL)
            rows.append({"tail": tail, "state": tag, "ts": st["t"], "P": list(st["P"]),
                         "n_pref_u0": n0, "n_pref_u1": n1, "n_tie": len(ds) - n0 - n1,
                         "median_d": float(np.median([r["d"] for r in ds])),
                         "per_seed": ds})
    out = {"kind": "CW-R1.9c", "note": "report-only（T8）", "plant": "sym sigma_w=(0.025,0.025)",
           "p_hat": [0.5, 0.5], "seeds": list(SEEDS), "rows": rows,
           "elapsed_s": round(time.perf_counter() - t0, 1), "n_run_episode": 0}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    d = RESULTS / f"cw_r1_9c_symstate_{stamp}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "t8_symstate.json").write_text(json.dumps(out, ensure_ascii=False, indent=1,
                                                   default=float), encoding="utf-8")
    for r in rows:
        print(f"[{r['tail']:5s}] {r['state']:9s} P={r['P']}  u0/u1/tie = "
              f"{r['n_pref_u0']}/{r['n_pref_u1']}/{r['n_tie']}  median_d = {r['median_d']:+.6f}")
    print(f"[ELAPSED] {out['elapsed_s']}s")
    print(f"[EVIDENCE] {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
