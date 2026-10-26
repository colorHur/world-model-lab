# -*- coding: utf-8 -*-
"""`CW-R3.5` **跑后补充**（探索性，未预注册）：三项不参与判词的对账。

1. 唯一非零格 `(τ=5,b=1,P=0.01)` 的 `n_m` 分辨率 —— 预注册的 `grid_refine` 挑到的
   两格（`P=0.05,τ=4,b∈{2,3}`）跑出来全为 0，**对"0.038 这个数是不是网格假象"没有分辨力**；
2. 交换率 `ΔV_q/(p̂·ΔV_succ)` 在本面板的分布 —— R3.3 的 `QUANTUM-EQUALS-ONE-SEND`
   是在**另一张面板**上测的，用它当分母前必须查它转不转移；
3. `σ` 是否真的进决策（`IV_free` 在 5 个 `σ` 上是否为常数）—— 若为常数，
   `UB` 与 `σ` 无关就不是结构事实而是"读数没进 DP"。

★ 本文件的一切数字都带 `exploratory: true`，不许当确证引用。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import mainline_config
from .posterior import p_succ_expected
from .r1_14_exactbase import _now_utc
from .r3_5_ubcert import SIGMA_LIST, SIGMA_STAR, Arms, cell

REPO = Path(__file__).resolve().parents[2]


def main() -> int:
    cfg = mainline_config()
    out: dict = {"posthoc": True, "exploratory": True, "utc": _now_utc(),
                 "round": "CW-R3.5"}

    # ---- 1) 唯一非零格的 n_m 分辨率
    rows = []
    for n_m in (65, 129, 257):
        a = Arms(cfg, n_m)
        c = cell(a, cfg, 5, 1, 0.01, SIGMA_STAR)
        rows.append({"n_m": n_m, "UB": c["IV_free_star"], "dV_q": c["dV_q"],
                     "ratio_ub": c["ratio_ub"], "deficit_over_dVq": c["deficit_over_dVq"]})
    rr = [r["ratio_ub"] for r in rows]
    out["A_nm_refine_bestcell"] = {
        "cell": {"tau": 5, "b": 1, "P": 0.01, "sigma": SIGMA_STAR},
        "rows": rows,
        "spread_rel_129_vs_257": float(abs(rr[2] - rr[1]) / rr[2]),
        "note": "预注册 grid_refine 命中的是 P=0.05 的两格（全 0），对 0.038 无分辨力；本项补上",
    }

    # ---- 2) 交换率
    p_hat = p_succ_expected(0.0, 1.0, cfg)
    a129 = Arms(cfg, 129)
    exch = []
    for tau in (2, 3, 4, 5):
        for b in range(1, min(tau, 4) + 1):
            for P in (0.01, 0.05, 0.1):
                c = cell(a129, cfg, tau, b, P, SIGMA_STAR)
                if c["dV_q"] > 1e-12 and c["dV_succ"] > 1e-12:
                    exch.append({"tau": tau, "b": b, "P": P,
                                 "ratio": c["dV_q"] / (p_hat * c["dV_succ"])})
    vals = [e["ratio"] for e in exch]
    out["B_exchange_rate"] = {
        "p_hat_prior": float(p_hat), "n": len(vals),
        "min": float(min(vals)), "median": float(np.median(vals)), "max": float(max(vals)),
        "n_below_0.5": int(sum(1 for v in vals if v < 0.5)),
        "rows": exch,
        "note": "R3.3 在另一面板给 1.000–1.087；本面板中位 0.836 ⇒ 恒等式不转移。"
                "ΔV_q 偏小 ⇒ UB/ΔV_q 偏大 ⇒ 对本轮的负结论是保守方向。",
    }

    # ---- 3) σ 是否进决策
    sig_rows = []
    for s in SIGMA_LIST:
        c = cell(a129, cfg, 5, 1, 0.01, s)
        sig_rows.append({"sigma": s, "IV_free": c["IV_free"],
                         "deficit_free": c["deficit_free"]})
    out["C_sigma_enters"] = {
        "cell": {"tau": 5, "b": 1, "P": 0.01},
        "rows": sig_rows,
        "monotone_decreasing": bool(all(sig_rows[i]["IV_free"] >= sig_rows[i + 1]["IV_free"]
                                        for i in range(len(sig_rows) - 1))),
        "note": "IV_free 随 σ 单调降 ⇒ 读数确实进 DP；`ratio_ub` 与 σ 无关是定义使然（上界恒取 σ⋆ 臂）",
    }
    print(json.dumps(out, indent=2, ensure_ascii=False)[:2500])
    dests = sorted((REPO / "results" / "cw_v1").glob("cw_r3_5_ubcert_*"))
    if dests:
        (dests[-1] / "r35_posthoc.json").write_text(
            json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[posthoc evidence] {dests[-1] / 'r35_posthoc.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
