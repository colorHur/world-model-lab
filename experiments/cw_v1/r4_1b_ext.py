# -*- coding: utf-8 -*-
"""`CW-R4.1` 跑后补充：把天花板格的网格收敛推到 `n_m=257`。

**未在预注册里**（预注册 §3 的分辨率重算只写 `n_m ∈ {65,129}`）⇒ 本文件产出的一切
只作**描述**，不改 `R2–R6/L1` 任何判词，也不参与 `VERDICT`。

动机：正式面板跑完发现 `n_m=65 → 129` 让天花板**系统性上移** 5.7–7.2%
（`0.1056→0.1137`、`0.1525→0.1617`），这不是随机抖动而是**量化偏置**
（`snap` 把后验均值吸附到网格，粗网格把"知道得多准"压扁了）。
一个只知道 65/129 两点的读者无法判断 0.16 是不是还要再涨，所以补第三点。

纪律：只读上一轮的 `r41.json`，不覆盖、不改写；结果写进**同一个证据目录**的新文件
`r41b_ext.json`，并在交付稿里标 `posthoc-ext`。
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path

from .r1_14_exactbase import _src_hash
from .r4_1_rho import Arms, KAPPA, RESULTS, cell, rho_config

N_M_EXT = 257
EXT_FLOOR = 1e-3        # 只补 ratio 超过这个数的靶子（零目标补了也没意义）


def load_latest() -> tuple[Path, dict]:
    cands = sorted(RESULTS.glob("cw_r4_1_rho_*/r41.json"))
    if not cands:
        raise SystemExit("找不到 r41.json")
    return cands[-1], json.loads(cands[-1].read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    t0 = time.perf_counter()

    src, d = load_latest()
    print(f"[ext] base={src}", flush=True)
    prev = {(r["rho"], r["alpha"]): r for r in d["refine"] if "rows" in r}
    out = {"round": "CW-R4.1b", "tag": "posthoc-ext", "base_run": str(src),
           "src_hash": _src_hash(), "n_m_ext": N_M_EXT, "floor": EXT_FLOOR,
           "note": "未预注册的网格收敛外推点；只作描述，不改任何判词与 VERDICT",
           "rows": []}

    for m in d["panel_meta"]:
        am = m.get("argmax_ratio")
        if not am or not isinstance(am.get("ratio_free"), float):
            continue
        if am["ratio_free"] <= EXT_FLOOR:
            continue
        key = (m["rho"], m["alpha"])
        pr = prev.get(key)
        cfg = rho_config(m["rho"], m["alpha"])
        t1 = time.perf_counter()
        arms = Arms(cfg, N_M_EXT)
        c = cell(arms, am["tau"], am["b"], am["P"], am["P"] * KAPPA)
        st = arms.states()
        capped = bool(arms.capped())
        del arms
        gc.collect()
        seq = {"65": am["ratio_free"],
               "129": (pr["rows"][1]["ratio_free"] if pr and len(pr["rows"]) > 1 else None),
               str(N_M_EXT): c["ratio_free"]}
        vals = [v for v in seq.values() if isinstance(v, float)]
        gap = (abs(seq["129"] - c["ratio_free"]) / abs(c["ratio_free"])
               if isinstance(seq.get("129"), float) and abs(c["ratio_free"]) > 1e-15 else None)
        out["rows"].append({"rho": m["rho"], "alpha": m["alpha"],
                            "cell": {"P": am["P"], "tau": am["tau"], "b": am["b"]},
                            "ratio_by_nm": seq, "gap_129_257": gap,
                            "IV_free": c["IV_free"], "IV_paid": c["IV_paid"],
                            "states": st, "capped": capped,
                            "sec": round(time.perf_counter() - t1, 3)})
        print(f"  [ext ρ={m['rho']:g} α={m['alpha']:g} τ={am['tau']} b={am['b']} P={am['P']:g}] "
              f"ratio 65/129/257 = {[round(v, 4) if isinstance(v, float) else v for v in vals]} "
              f"gap(129,257)={gap if gap is None else round(gap, 4)}", flush=True)

    conv = [r["gap_129_257"] for r in out["rows"] if isinstance(r["gap_129_257"], float)]
    out["max_gap_129_257"] = float(max(conv)) if conv else None
    out["verdict_ext"] = ("GRID-CONVERGED-BY-129" if conv and max(conv) <= 0.05
                          else "GRID-STILL-MOVING")
    out["sec_total"] = round(time.perf_counter() - t0, 3)
    print(f"[ext] {out['verdict_ext']} max_gap={out['max_gap_129_257']} "
          f"{out['sec_total']} s", flush=True)

    dest = Path(args.out) if args.out else (src.parent / "r41b_ext.json")
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[ext evidence] {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
