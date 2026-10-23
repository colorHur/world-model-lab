# -*- coding: utf-8 -*-
"""CW-R1.7.3 落盘入口：`P` 网格重铺 + `W4` 判据重写（**新证据目录**）。

判据见 `docs/research/CW-R1.7.3-网格与判据-预注册-v0.1.md`。

判词
-----
`GRID-REJECTED`（`W4R` 或 `W1R` FAIL —— 新网格不可采纳 / 预算维不平滑）
`GRID-ADMISSIBLE`（`W4R`+`W5`+`W1R` 全过 ⇒ 下一轮才允许跑 `G4/M1/G8a` 与开发面板）
★ 本轮**不跑面板、不出 `N1/N2`**：先证"尾部值函数的尺度与平滑都对"，再去量收益。
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta, timezone

import numpy as np

from . import config as C
from .dp_grid_gates import run_grid_gates
from .plan_dp import cache_stats
from .r1_6_plan import RESULTS, _git, _src_hash


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="CW-R1.7.3 网格与判据闸")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--note", default="")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    g = run_grid_gates()
    s = g.summary()
    elapsed = round(time.perf_counter() - t0, 2)

    print(f"[CW-R1.7.3] grid gates: {s['n_checks']} checks, {s['n_fail']} fail ({elapsed} s)")
    for r in g.rows:
        print(("  PASS " if r["pass"] else "  FAIL "), r["name"][:78])

    names = {r["name"].split(" ")[0]: bool(r["pass"]) for r in g.rows}
    # 关键闸：W4R（含对照）、W5、W1R；对照闸本身就是"必须 FAIL 被测物"的判据
    key = [k for k in names if k in ("W4R", "W4R-对照", "W5", "W5-对照", "W7", "W8", "W8-对照", "W1R")]
    n_key_fail = sum(1 for k in key if not names[k])
    verdict = "GRID-ADMISSIBLE" if n_key_fail == 0 else "GRID-REJECTED"
    print(f"[CW-R1.7.3] 判词 = {verdict}（关键闸失败 {n_key_fail}）")

    if not args.no_write:
        now = datetime.now(timezone(timedelta(hours=8)))
        cfg = C.CWConfig(plant=C.bounded_plant())
        run_id = (f"cw_r1_7_3_grid_{now.strftime('%Y%m%dT%H%M%S')}_"
                  f"{cfg.fingerprint()}_{_src_hash()}")
        out = RESULTS / run_id
        out.mkdir(parents=True, exist_ok=True)

        def _enc(o):
            if hasattr(o, "item"):
                return o.item()
            raise TypeError(f"{type(o).__name__} 不是 JSON 可序列化")

        def _dump(name, obj):
            (out / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2,
                                               default=_enc), encoding="utf-8")

        _dump("grid_gates.json", s)
        _dump("verdict.json", {"verdict": verdict, "n_key_fail": n_key_fail,
                               "key_gates": key, "panel_run": False,
                               "panel_reason": "本轮先证尺度与平滑；面板留给下一轮",
                               "eligible_for_chain": verdict == "GRID-ADMISSIBLE"})
        _dump("meta.json", {
            "run_id": run_id,
            "kind": "GRID-AND-CRITERION-GATES-ONLY（本轮未跑面板）",
            "protocol": C.CWConfig.PROTOCOL_VERSION,
            "grid_default_module": __import__("experiments.cw_v1.plan_dp",
                                              fromlist=["GRID_DEFAULT"]).GRID_DEFAULT,
            "grid_used": "r173",
            "b_cap": int(cfg.protocol.horizon),
            "event_pairing": cfg.protocol.event_pairing,
            "config": cfg.to_dict(),
            "dp_cache": cache_stats(),
            "at_asia_shanghai": now.isoformat(),
            "elapsed_s": elapsed,
            "git_head": _git(["rev-parse", "HEAD"]),
            "git_status_porcelain": _git(["status", "--porcelain"]),
            "src_hash": _src_hash(),
            "note": args.note,
            "disclaimer": ("本轮只判定'尾部值函数的尺度与平滑'；不出任何收益判词；"
                           "未跑 N1/N2；未重跑旧 P3；未训练模型；未改 plant/q/r/p̂。"),
        })
        print(f"[CW-R1.7.3] 证据落盘 results/cw_v1/{run_id}")

    return 0 if s["n_fail"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
