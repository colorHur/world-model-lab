# -*- coding: utf-8 -*-
"""`CW-R5.3b` 事后对照：`b × τ` 的 2×2 边界表（**跑后追加，不进主判据**）。

登记口径（必须原样引用）
------------------------
本文件是在**已经看到主面板**（`r53.json`：`τ≤3` 逐位零、`τ≥4` 正）之后写的。
它**只能削弱或加强**主判的机制解释，**不能**当预注册证据；所有读数在交付里单列一节，
并注明"看到主面板之后追加"。

问的问题
--------
主面板把边界钉在 `τ≥4`（= **3 个动作时隙**）。但 `b=8` 只有一个发送，
所以"`τ=3` 为零"有两种解释：
  (a) **时隙论**：2 个动作时隙本身就不够（无论预算多少）；
  (b) **延迟论**：`τ=3` 时探测必须把唯一一次发送推迟一格，代价太大；预算多就没问题。
⇒ 用 `b=16`（2 次发送）在同样的 `τ` 上跑一遍即可分开：若 `τ=3/b=16` 仍为零而 `τ=5/b=16` 为正，
   (a) 胜；若 `τ=3/b=16` 转正，(b) 胜。

同一引擎、同一 `n_m`、同一 plant（R5.2 的 `τ=4/b=16 = 0` 是**逐位零**，本文件要逐位复现它）。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .config import mainline_config
from .r1_14_exactbase import _now_utc, _src_hash
from .r5_2_three_users import P0, V_PRIOR, build_engine, cfg_for
from .r5_3_tau_boundary import B_ONE, PT_MAIN, SIGMA_STAR

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results" / "cw_v1"

#: 2×2 边界表：`b ∈ {8, 16}` × `τ ∈ {3,4,5,6}`
B_LIST = (B_ONE, 16)
TAU_LIST = (3, 4, 5, 6)


def run(n_m_list: tuple[int, ...]) -> dict:
    rows = []
    legs = []
    for n_m in n_m_list:
        t0 = time.perf_counter()
        cfg = cfg_for(2, "HH")
        no = build_engine(cfg, SIGMA_STAR, n_m, 0, probe_enabled=False, n_users=2)
        paid = build_engine(cfg, SIGMA_STAR, n_m, PT_MAIN, n_users=2)
        for b in B_LIST:
            for tau in TAU_LIST:
                a0 = no(tau, (P0, P0), (0.0, 0.0), (V_PRIOR, V_PRIOR), b)
                vp = paid(tau, (P0, P0), (0.0, 0.0), (V_PRIOR, V_PRIOR), b)
                rows.append({"n_users": 2, "n_m": n_m, "tau": tau, "b_ticks": b,
                             "b_quanta": b / 8, "p": 0.0,
                             "A0": float(a0), "V_paid": float(vp),
                             "IV_paid": float(a0 - vp),
                             "IV_share_A0": float((a0 - vp) / a0) if a0 else float("nan"),
                             "capped": bool(no.capped() or paid.capped())})
                print(f"  [b={b} tau={tau} n_m={n_m}] IV={rows[-1]['IV_paid']:.6e} "
                      f"A0={rows[-1]['A0']:.6f} capped={rows[-1]['capped']}", flush=True)
        legs.append({"n_m": n_m, "sec": round(time.perf_counter() - t0, 3),
                     "states": {"noprobe": no.n_states(), "paid": paid.n_states()},
                     "capped": bool(no.capped() or paid.capped())})
    return {"rows": rows, "legs": legs}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="CW-R5.3b b×τ 2x2 事后对照")
    ap.add_argument("--n-m", default="129,257")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    n_m_list = tuple(int(x) for x in args.n_m.split(","))
    t_all = time.perf_counter()
    dest = Path(args.out) if args.out else (
        RESULTS / f"cw_r5_3b_btau_{_now_utc()}_{_src_hash()[:16]}")
    dest.mkdir(parents=True, exist_ok=True)
    res = run(n_m_list)
    out = {"round": "CW-R5.3b", "protocol": "CW-POMDP-v1.3-ticks", "utc": _now_utc(),
           "post_hoc": True,
           "post_hoc_note": "看到 R5.3 主面板之后追加；不进主判据，只用于分开时隙论/延迟论",
           "sec_total": round(time.perf_counter() - t_all, 3),
           "panel": {"alpha": 2.5, "P": P0, "n_m": list(n_m_list),
                     "b_ticks": list(B_LIST), "taus": list(TAU_LIST), "p": 0.0},
           "cells": res["rows"], "legs": res["legs"]}
    (dest / "r53b.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[evidence] {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
