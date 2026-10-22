# -*- coding: utf-8 -*-
"""CW-R1-3 判词更正：**重分析**，不是新确认。

这脚本存在的原因
----------------
2026-10-22 验收发现 `r1_3_oracle.py` 的比值方向反了（存 `baseline/oracle`，
门槛按 `oracle/baseline < 0.9` 判），且"每格挑一个最好基线"的汇总方式把
"输给某条基线"藏掉，还写死了 `WIN_MIN_CELLS = 6`（设计只有 4 格，不可满足）。
结果 `SPACE-VOID` 这个判词同时被三处错误推动。

★ 本脚本**不重跑仿真**：它从旧 run 的逐 episode 记录里复原两臂成本，
  用新的评价器（`evaluate.compare_cell`，CW-EVAL-v1.1）重算一遍，
  **只用于量化旧判词错到什么程度**。
  ⇒ 它的性质是 **REANALYSIS / NOT-CONFIRMATION**：
    旧数据是在**旧协议**（确定初态、ACK 不进信念）下跑的，与新协议不同口径，
    不得当作 v1.1 的独立确认。新协议下的数字必须另跑。

复原方式（可验证）
------------------
旧 run 的 `matched.json` 每个 `samepot_pairwise` 条目带逐 episode `delta`
（= baseline − oracle），`oracle.json` 带同一 episode 的 oracle 成本 ⇒
`baseline = oracle + delta`。脚本会把"用旧公式重算的比值"与存盘值逐格对齐，
对不上就报错 —— 这一步保证复原是真的，而不是"我记得是这样"。
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from .evaluate import EVAL_VERSION, compare_cell

REPO = Path(__file__).resolve().parents[2]
CORRECTIONS = REPO / "results" / "cw_v1" / "corrections"

#: 待更正的旧 run（只读，一个字节不改）
OLD_RUN = "cw_r1_3_20261021T221429_47fbe96aa30f8a6f_07d37da14c67cf10"

RATIO_THRESHOLD = 0.9
WIN_THRESHOLD = 0.7


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def rebuild(old_dir: Path) -> tuple[list[dict], list[dict]]:
    """返回 (逐格逐基线配对统计, 一致性对账记录)。"""
    oracle = _load(old_dir / "oracle.json")
    matched = _load(old_dir / "matched.json")

    # oracle 逐 (rho, ratio, variant) -> {episode: cost}
    main_rows: dict[tuple, dict[int, float]] = {}
    for o in oracle:
        if o["variant"] != "main":
            continue
        main_rows[(o["rho"], o["budget_ratio"])] = {
            int(r["episode"]): float(r["cost_true_per_step"]) for r in o["rows"]
        }

    stats: list[dict] = []
    audit: list[dict] = []
    for m in matched:
        if m["mode"] != "samepot_pairwise":
            continue
        key = (m["rho"], m["budget_ratio"])
        ref = main_rows[key]
        eps = [int(d["episode"]) for d in m["deltas"]]
        base = np.array([ref[e] + float(d["delta"]) for d, e in zip(m["deltas"], eps)])
        cand = np.array([ref[e] for e in eps])

        st = compare_cell(base, cand, cell=[m["rho"], m["budget_ratio"], m["policy"]],
                          n_boot=4000)

        # ★ 对账：用**旧公式**重算，必须与存盘值一致（证明复原忠实）
        old_formula = float(np.median(base) / np.median(cand))
        stored = float(m["cost_ratio_median"])
        audit.append({"rho": m["rho"], "budget_ratio": m["budget_ratio"],
                      "policy": m["policy"], "stored_old_ratio": stored,
                      "recomputed_old_formula": round(old_formula, 6),
                      "abs_diff": abs(old_formula - stored),
                      "n_pairs": len(eps)})

        stats.append({
            "rho": m["rho"], "budget_ratio": m["budget_ratio"], "baseline": m["policy"],
            "n_pairs": st.n_pairs,
            "status": st.status,
            "ratio_oracle_over_baseline": st.median_pairwise_ratio,
            "ratio_of_medians_oracle_over_baseline": st.ratio_of_medians,
            "median_delta_baseline_minus_oracle": st.median_delta,
            "delta_ci": list(st.delta_ci),
            "win_fraction": st.win_fraction,
            "old_stored_baseline_over_oracle": stored,
            "pass_ratio": bool(st.median_pairwise_ratio < RATIO_THRESHOLD),
            "pass_win": bool(st.win_fraction >= WIN_THRESHOLD),
            "pass": bool(st.median_pairwise_ratio < RATIO_THRESHOLD
                         and st.win_fraction >= WIN_THRESHOLD),
        })
    return stats, audit


def summarize(stats: list[dict]) -> dict:
    cells = sorted({(s["rho"], s["budget_ratio"]) for s in stats})
    baselines = sorted({s["baseline"] for s in stats})
    per_baseline = []
    for b in baselines:
        rows = [s for s in stats if s["baseline"] == b]
        n_pass = sum(1 for s in rows if s["pass"])
        per_baseline.append({
            "baseline": b,
            "n_cells": len(rows),
            "n_pass": n_pass,
            "beats_this_baseline_in_all_cells": n_pass == len(cells),
            "ratio_range": [min(s["ratio_oracle_over_baseline"] for s in rows),
                            max(s["ratio_oracle_over_baseline"] for s in rows)],
            "win_range": [min(s["win_fraction"] for s in rows),
                          max(s["win_fraction"] for s in rows)],
        })
    return {
        "n_cells": len(cells), "cells": [list(c) for c in cells],
        "n_baselines": len(baselines),
        "n_baselines_beaten": sum(1 for p in per_baseline
                                  if p["beats_this_baseline_in_all_cells"]),
        "per_baseline": per_baseline,
    }


def gates(summary: dict, audit: list[dict], stats: list[dict]) -> dict:
    rows = []

    def _row(name, ok, detail):
        rows.append({"name": name, "pass": bool(ok), "detail": detail})

    # 容差按**存盘精度**定：旧 run 的比值是 round(..., 6) ⇒ 单边误差上界 5e-7。
    max_diff = max((a["abs_diff"] for a in audit), default=0.0)
    _row("C1 复原忠实：用旧公式重算的比值与存盘值逐格一致", max_diff <= 5.1e-7,
         {"n_cells": len(audit), "max_abs_diff": max_diff,
          "tolerance": "存盘 6 位小数 ⇒ 5.1e-7"})

    # ★ 方向修正的机器证据：**同类统计量**才可比。
    #   旧存盘值是"中位数之比"（ratio_of_medians），所以只能与新的
    #   `ratio_of_medians_oracle_over_baseline` 比倒数。
    #   `median_pairwise_ratio`（逐 episode 比值的中位）是**另一个量**，
    #   拿它去比会得到 0.47 这种数 —— 那是两个统计量之差，不是方向错（E8 已覆盖）。
    worst = 0.0
    for s in stats:
        prod = (s["ratio_of_medians_oracle_over_baseline"]
                * s["old_stored_baseline_over_oracle"])
        worst = max(worst, abs(prod - 1.0))
    _row("C2 中位数之比被翻成倒数（方向确实改对）", worst < 1e-5,
         {"max_abs_prod_minus_1": worst,
          "note": "只比 ratio_of_medians；median_pairwise_ratio 是另一个统计量"})

    # C2b：两个比值统计量在本数据上确实不同（否则上面那条换成哪个都一样，就白测了）
    gap = max((abs(s["ratio_oracle_over_baseline"]
                   - s["ratio_of_medians_oracle_over_baseline"]) for s in stats),
              default=0.0)
    _row("C2b 逐 episode 比值中位数 ≠ 中位数之比（两个统计量不可互换）", gap > 1e-3,
         {"max_abs_gap": gap})

    _row("C3 格数要求由设计推导（4 格 ⇒ 4），不含写死的 6",
         summary["n_cells"] == len(summary["cells"]) and summary["n_cells"] != 6,
         {"n_cells": summary["n_cells"]})

    n_undef = sum(1 for s in stats if s["status"] != "ok")
    _row("C4 所有配对格都能定义（无 NaN / 非正基线 / 口径冲突）", n_undef == 0,
         {"n_undefined": n_undef,
          "statuses": sorted({s["status"] for s in stats})})

    # ★ 允许"诊断臂并不优于所有基线"—— 这不再是 FAIL，而是一条**必须原样报出**的事实
    _row("C5 逐基线结果已完整报出（不作为研究判据）", True,
         {"n_baselines": summary["n_baselines"],
          "n_baselines_beaten": summary["n_baselines_beaten"],
          "note": "诊断臂输给某条基线 ⇒ 只能说'这个贪心臂不够好'，不能说'信息无价值'"})

    n_fail = sum(0 if r["pass"] else 1 for r in rows)
    return {"n_checks": len(rows), "n_fail": n_fail, "all_pass": n_fail == 0, "rows": rows}


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="CW-R1-3 判词更正（重分析，非新确认）")
    ap.add_argument("--old-run", default=OLD_RUN)
    args = ap.parse_args(argv)

    old_dir = REPO / "results" / "cw_v1" / args.old_run
    if not old_dir.is_dir():
        print(f"[CW-R1-3-CORR] 找不到旧 run：{old_dir}")
        return 2

    t0 = time.perf_counter()
    stats, audit = rebuild(old_dir)
    summary = summarize(stats)
    g = gates(summary, audit, stats)
    elapsed = round(time.perf_counter() - t0, 3)

    now = datetime.now(timezone(timedelta(hours=8)))
    run_id = f"cw_r1_3_corr_{now:%Y%m%dT%H%M%S}_{EVAL_VERSION.replace('.', '_')}"
    out = CORRECTIONS / run_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "gates.json").write_text(
        json.dumps(g, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "meta.json").write_text(json.dumps({
        "run_id": run_id,
        "kind": "REANALYSIS-NOT-CONFIRMATION",
        "eval_version": EVAL_VERSION,
        "source_run": args.old_run,
        "source_protocol": "CW-POMDP-v1（确定初态、ACK 不进信念）",
        "current_protocol": "CW-POMDP-v1.1（初态取自公开先验、ACK 接入信念）",
        "why_not_confirmation": (
            "本目录只把旧 run 的逐 episode 记录用新评价器重算，用于量化旧判词的错误幅度。"
            "旧数据来自不同协议口径，**不得**当作 v1.1 的独立确认；"
            "v1.1 的数字必须在新协议下重跑。"),
        "at_asia_shanghai": now.isoformat(),
        "elapsed_s": elapsed,
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"[CW-R1-3-CORR] run_id={run_id}")
    print(f"[CW-R1-3-CORR] gates: {g['n_checks']} checks, {g['n_fail']} fail -> "
          f"{'ALL PASS' if g['all_pass'] else 'FAIL'}")
    print(f"[CW-R1-3-CORR] 逐基线：{summary['n_baselines_beaten']}/{summary['n_baselines']} "
          f"条基线在全部 {summary['n_cells']} 格上被诊断臂压过")
    for p in summary["per_baseline"]:
        print(f"  {p['baseline']:14s} pass {p['n_pass']}/{p['n_cells']} "
              f"ratio[{p['ratio_range'][0]:.3f},{p['ratio_range'][1]:.3f}] "
              f"win[{p['win_range'][0]:.2f},{p['win_range'][1]:.2f}]")
    for r in g["rows"]:
        if not r["pass"]:
            print(f"  FAIL  {r['name']}: {json.dumps(r['detail'], ensure_ascii=False)[:300]}")
    return 0 if g["all_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
