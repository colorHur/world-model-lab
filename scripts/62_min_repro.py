#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P3-3b · CPU 最短复现：4 个 cell 重跑主结果，并与**入库证据逐位比对**。

复现什么
--------
`P3-0` 主确认表的核心那一格（`B=200 unpaced`，臂 `age`）：
    predictor ∈ {zoh, linear} × seed ∈ {100, 130} × 30 集 = **4 cell**
输出：每格的 `J_uav/J_cart`、总耗时，以及与 `results/runs/.../58_p3_confirmation.json`
对应 cell 的**逐 episode 逐位比对**（`max|Δ|` 与不一致个数）。

★ 这不是"一键复现全部"：只复现**一格**。P3-1/P3-2 的扫描（96/144 cell，共约 400 s）不在最短路径里。
★ 也不是**跨机验收**：本机同版本下才要求逐位一致（`docs/REPRODUCE.md` §6 的容差纪律）。

纪律
----
- **不写** `results/`（自检/复现入口不得触碰入库证据；#45 同族）。
- 需要落盘时只写 `outputs/62_min_repro.json`（`outputs/` 已 gitignore）。
- 证据文件缺失 ⇒ 打印 SKIP 并给出应存在的路径，不静默 PASS。

用法
----
    D:/devtools/miniconda3/envs/ai-lab/python.exe scripts/62_min_repro.py
    ... --seeds 100,130,160     # 三种子全跑（约 ×1.5 时间）
    ... --quick                 # 2 集冒烟（只验流程，不是复现）

退出码：0 = 与入库证据逐位一致；2 = 不一致（打印差异数与前几个位置）。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from wmlab.eval.results_io import latest_result_path  # noqa: E402

REF_NAME = "58_p3_confirmation"
BUDGET = 200
MODE = "unpaced"
N_EP = 30


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="100,130")
    ap.add_argument("--episodes", type=int, default=N_EP)
    ap.add_argument("--quick", action="store_true", help="2 集冒烟（不比对，因为证据是 30 集）")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    n_ep = 2 if args.quick else args.episodes

    T58 = _load(ROOT / "scripts" / "58_p3_confirmation.py", "t58_mod62")

    ref_path = latest_result_path(REF_NAME)
    ref = None
    if ref_path is None or not ref_path.exists():
        print(f"[SKIP] 找不到 {REF_NAME} 的入库结果；应位于 results/runs/<run_id>/{REF_NAME}.json")
        print("       只跑本脚本，不做比对。")
    else:
        print(f"[ref] {ref_path.relative_to(ROOT).as_posix()}")
        ref = json.loads(ref_path.read_text(encoding="utf-8"))

    print(f"[cfg] B={BUDGET} mode={MODE} arm=age episodes={n_ep} seeds={seeds}")
    t_all = time.perf_counter()
    rows = []
    for seed in seeds:
        for name in ("zoh", "linear"):
            t0 = time.perf_counter()
            # 直接复用 58 的 `run_cell`：配置（吸收项/预热/臂/探针）与主实验**完全同源**，
            # 否则"复现"就变成"另一个配置跑出相近的数"。
            c = T58.run_cell(BUDGET, MODE, name, seed, n_ep, timed=(name in ("zoh", "linear")))
            dt = time.perf_counter() - t0
            rows.append({"predictor": name, "seed": seed,
                         "J_per_episode": c["J_per_episode"], "J_mean": c["J_mean"],
                         "elapsed_s": round(dt, 3)})
            print(f"  B={BUDGET} {name:6s} seed={seed:3d}  "
                  f"J=({rows[-1]['J_mean'][0]:.6f}, {rows[-1]['J_mean'][1]:.6f})  {dt:.2f}s")
    total = time.perf_counter() - t_all
    print(f"[time] {len(rows)} cell 共 {total:.2f} s（{'冒烟' if args.quick else '完整'}）")

    if args.quick:
        print("\n[quick] 冒烟模式：不比对（证据是 30 集）。流程 OK。")
        return 0
    if ref is None:
        return 0

    # ---------------- 逐位比对 ----------------
    print("\n[compare] 与入库证据逐 episode 比对")
    bad = 0
    for r in rows:
        hit = None
        for c in ref["cells_J"]:
            if (c.get("seed_set") == "primary" and c["budget"] == BUDGET
                    and c["mode"] == MODE and c["predictor"] == r["predictor"]
                    and c["seed"] == r["seed"]):
                hit = c
                break
        if hit is None:
            print(f"  [SKIP] 证据里没有 (seed={r['seed']}, {r['predictor']}) 这一格")
            continue
        a = np.asarray(r["J_per_episode"], dtype=float)
        b = np.asarray(hit["J_per_episode"], dtype=float)
        if a.shape != b.shape:
            print(f"  [FAIL] seed={r['seed']} {r['predictor']}: 形状 {a.shape} vs {b.shape}")
            bad += 1
            continue
        d = np.abs(a - b)
        n_diff = int((d > 0).sum())
        print(f"  [{'PASS' if n_diff == 0 else 'FAIL'}] seed={r['seed']} {r['predictor']:6s} "
              f"n={a.size}  max|Δ|={d.max():.3e}  不一致 {n_diff} 个")
        bad += (n_diff != 0)

    out = ROOT / "outputs" / "62_min_repro.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"budget": BUDGET, "mode": MODE, "episodes": n_ep,
                               "rows": rows, "elapsed_s": round(total, 3),
                               "ref": ref_path.relative_to(ROOT).as_posix() if ref_path else None},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[out] {out.relative_to(ROOT).as_posix()}（outputs/ 不入库）")

    print(f"\n{'BITWISE MATCH' if bad == 0 else 'MISMATCH: ' + str(bad) + ' cell'}")
    return 0 if bad == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
