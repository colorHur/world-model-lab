# -*- coding: utf-8 -*-
"""CW-R1.9d **自我否证**：T8 的"8/8 偏 u0"是不是小样本假象？

预注册：`docs/research/CW-R1.9-策略层对称-预注册-v0.1.md` §7 修订 5（T8b）。

T8 在**完全对称**的 plant + **完全对称**的公开状态上，用 8 个 seed 看到 `dp` 8/8、7/1、5/3
偏 `SEND(0)`、`paced` 7/1、5/3、6/2 —— 看起来像"下标依赖"。但 n=8 的 `8/8` 只有 `p=0.0078`
且**极易被抽样偶然凑出**。本脚本用 40 个 seed 重做，并加一条**逐 seed 的等变恒等式**：

    Δ(seed) = v1 − v0，      Δ_swap(seed) = 用**面板列置换**重跑的同一量
    要求    Δ_swap(seed) == −Δ(seed)        （逐位，容差 1e-12）

状态本来就对称 ⇒ 置换面板 = 纯等变性检验（不掺状态置换）。这一条若 40/40 成立，
说明**状态→动作映射没有任何下标依赖**，T8 的 8/8 必须按小样本假象处理。

report-only。
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import config as C
from .r1_9_symmetry import _mk, _panel_patch, _PermPanel, _pub
from .world import SchedulerView

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results" / "cw_v1"

STATES = (("P003", 8, 128, (0.03, 0.03)), ("P006", 8, 128, (0.06, 0.06)))
SEEDS = tuple(range(2001, 2041))


def main() -> int:
    t0 = time.perf_counter()
    cfg = C.CWConfig(plant=C.bounded_plant(A=(1.0, 1.0), sigma_w=(0.025, 0.025)))
    out = {"kind": "CW-R1.9d", "note": "T8b 自我否证（report-only）",
           "plant": "sym A=(1,1) sigma_w=(0.025,0.025)", "p_hat": [0.5, 0.5],
           "n_seeds": len(SEEDS), "rows": []}
    for tail in ("paced", "dp"):
        for tag, t, bud, P in STATES:
            st = {"tag": tag, "t": t, "bud": bud, "P": P, "x": (0.0, 0.0),
                  "mh": (0.0, 0.0), "vh": (1.0, 1.0)}
            ds, bad, per = [], [], []
            for sd in SEEDS:
                a = _mk(cfg, sd, None if tail == "paced" else "dp")
                a(SchedulerView(_pub(st)))
                d0 = a.root["SEND(1)"] - a.root["SEND(0)"]
                b = _mk(cfg, sd, None if tail == "paced" else "dp")
                with _panel_patch(_PermPanel):
                    b(SchedulerView(_pub(st)))
                d1 = b.root["SEND(1)"] - b.root["SEND(0)"]
                ds.append(d0)
                per.append({"seed": sd, "d": d0, "d_swap": d1, "sum": d0 + d1,
                            "anti_ok": bool(abs(d0 + d1) <= 1e-12 * max(1.0, abs(d0)))})
                if not per[-1]["anti_ok"]:
                    bad.append(sd)
            arr = np.asarray(ds)
            sdv = float(arr.std(ddof=1))
            row = {"tail": tail, "state": tag, "P": list(P), "n": int(arr.size),
                   "mean_d": float(arr.mean()), "median_d": float(np.median(arr)),
                   "sd_d": sdv,
                   "sem_t": float(arr.mean() / (sdv / np.sqrt(arr.size))) if sdv > 0 else 0.0,
                   "n_pos": int((arr > 0).sum()), "n_neg": int((arr < 0).sum()),
                   "n_anti_identity_fail": len(bad), "anti_identity_seeds_bad": bad[:5],
                   "per_seed": per}
            out["rows"].append(row)
            print(f"[{tail:5s}] {tag}  n={row['n']}  mean_d={row['mean_d']:+.6f} "
                  f"median={row['median_d']:+.6f}  t={row['sem_t']:+.2f}  "
                  f"pos/neg={row['n_pos']}/{row['n_neg']}  "
                  f"anti_identity_fail={row['n_anti_identity_fail']}")
    out["all_anti_identity_ok"] = all(r["n_anti_identity_fail"] == 0 for r in out["rows"])
    out["elapsed_s"] = round(time.perf_counter() - t0, 1)
    out["n_run_episode"] = 0
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    d = RESULTS / f"cw_r1_9d_t8b_{stamp}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "t8b.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float),
                                encoding="utf-8")
    print(f"[ALL_ANTI_IDENTITY_OK] {out['all_anti_identity_ok']}  elapsed={out['elapsed_s']}s")
    print(f"[EVIDENCE] {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
