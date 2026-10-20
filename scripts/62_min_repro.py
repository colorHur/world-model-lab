#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P3-3b · CPU 最短复现：4 个 cell 重跑主结果，并与**冻结证据逐位比对**。

复现什么
--------
`P3-0` 主确认表的核心那一格（`B=200 unpaced`，臂 `age`）：
    predictor ∈ {zoh, linear} × seed ∈ {100, 130} × 30 集 = **4 cell**
输出：每格的 `J_uav/J_cart`、耗时，以及与 `docs/evidence_manifest_p3.json` 钉死的那份
`58_p3_confirmation.json` 对应 cell 的**逐 episode 逐位比对**（`max|Δ|` 与不一致个数）。

★ 这不是"一键复现全部"：只复现**一格**。P3-1/P3-2 的扫描（96/144 cell，共约 400 s）不在最短路径里。
★ 也不是**跨机验收**：本机同版本下才要求逐位一致（`docs/REPRODUCE.md` §6 的容差纪律）。

★ 2026-10-20 复验 A 项的修补（这一段是这次改动的全部理由）
------------------------------------------------------------
旧版在 `--seeds 999`（证据里没有这个种子）时**两格 SKIP、仍然打印 BITWISE MATCH、退出码 0**。
根因不是"少了个 else"，而是判据写成了 `bad == 0` —— **"没有比较过"与"比较后相等"都被算成 0**。
同一族的坑还有两个：`reference` 全缺时直接 `return 0`；`np.abs(a-b)` 遇到 NaN 时
`NaN > 0` 为 False ⇒ **NaN 会被算成"一致"**。

现在的判据（可证伪，且错法都写在下面）：
- 先算**期望集合** `expected = {(seed, predictor)}`，逐格要求"跑通 ∧ 在证据里找到 ∧ 形状一致 ∧
  三次有限性检查通过 ∧ 逐位相等"，任何一环不满足就进 `skipped` 或 `failed`，**不进** `compared`；
- `BITWISE MATCH` 的充要条件：`expected > 0 ∧ compared == expected ∧ skipped == 0 ∧ failed == 0`；
- 非有限值（输入/本次输出/参考）一律 `failed`，**不允许**被 NaN 比较吞掉；
- 退出码分层，自动任务能区分：
  `0` BITWISE MATCH｜`2` MISMATCH｜`4` NOT VERIFIED（缺证据/缺格/非有限）｜`5` SMOKE ONLY。

纪律
----
- **不写** `results/`（自检/复现入口不得触碰入库证据；#45 同族）。
- 需要落盘时只写 `outputs/62_min_repro.json`（`outputs/` 已 gitignore）。
- 证据来源默认 = 冻结清单；`--ref` 显式覆盖（**反例注入只在隔离副本上用这个**）；
  `--latest` 才退回"取最新那份"。

用法
----
    python scripts/62_min_repro.py                       # 清单 + 默认种子 100,130
    python scripts/62_min_repro.py --seeds 100,130,160   # 三种子
    python scripts/62_min_repro.py --quick               # 2 集流程冒烟（**不是**复现）
    python scripts/62_min_repro.py --ref /tmp/x/58_....json   # 隔离副本（反例注入用）
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

from wmlab.eval.evidence import EvidenceError, resolve  # noqa: E402
from wmlab.eval.results_io import latest_result_path  # noqa: E402

REF_NAME = "58_p3_confirmation"
BUDGET = 200
MODE = "unpaced"
N_EP = 30
PREDS = ("zoh", "linear")

EXIT_MATCH, EXIT_MISMATCH, EXIT_NOT_VERIFIED, EXIT_SMOKE = 0, 2, 4, 5


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _finite(a: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(a)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="100,130")
    ap.add_argument("--episodes", type=int, default=N_EP)
    ap.add_argument("--quick", action="store_true", help="2 集流程冒烟（不比对，因为证据是 30 集）")
    ap.add_argument("--ref", default=None, help="显式指定参考 JSON（隔离副本；不校验清单指纹）")
    ap.add_argument("--latest", action="store_true", help="走 latest 取证据（默认读冻结清单）")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    n_ep = 2 if args.quick else args.episodes

    T58 = _load(ROOT / "scripts" / "58_p3_confirmation.py", "t58_mod62")

    # ---------------- 参考证据 ----------------
    ref, ref_path, ref_note = None, None, None
    if args.ref:
        ref_path = Path(args.ref)
        src_kind = "--ref（一次性覆盖，不是清单证据）"
    elif args.latest:
        ref_path = latest_result_path(REF_NAME)
        src_kind = "--latest（会随新 run 漂移）"
    else:
        ref_path = None
        src_kind = "冻结清单"
    try:
        if ref_path is not None:
            if not ref_path.exists():
                raise EvidenceError(f"参考文件不存在：{ref_path.as_posix()}")
            ref = json.loads(ref_path.read_text(encoding="utf-8"))
        else:
            r = resolve(REF_NAME, key="58")
            ref_path, ref = r["path"], json.loads(r["raw"].decode("utf-8"))
            ref_note = (f"run_id={r['run_id']} sha_norm={r['sha256_norm'][:12]}"
                        + (f"（{r['note']}）" if r.get("note") else ""))
    except EvidenceError as exc:
        print(f"[NOT VERIFIED] 拿不到参考证据（来源：{src_kind}）：{exc}")
        print(f"  expected={len(seeds) * len(PREDS)} compared=0 skipped={len(seeds) * len(PREDS)} failed=0")
        print("\nNOT VERIFIED: 0 cell 被比对（缺证据 ⇒ 不得声明匹配）")
        return EXIT_NOT_VERIFIED

    print(f"[ref] {ref_path.as_posix() if isinstance(ref_path, Path) else ref_path}"
          f"（来源：{src_kind}）")
    if ref_note:
        print(f"      {ref_note}")
    print(f"[cfg] B={BUDGET} mode={MODE} arm=age episodes={n_ep} seeds={seeds}")

    if "cells_J" not in ref:
        print("[NOT VERIFIED] 参考文件里没有 cells_J")
        return EXIT_NOT_VERIFIED

    # ---------------- 跑 ----------------
    t_all = time.perf_counter()
    rows = []
    for seed in seeds:
        for name in PREDS:
            t0 = time.perf_counter()
            # 直接复用 58 的 `run_cell`：配置（吸收项/预热/臂/探针）与主实验**完全同源**，
            # 否则"复现"就变成"另一个配置跑出相近的数"。
            c = T58.run_cell(BUDGET, MODE, name, seed, n_ep, timed=(name in PREDS))
            dt = time.perf_counter() - t0
            rows.append({"predictor": name, "seed": seed,
                         "J_per_episode": c["J_per_episode"], "J_mean": c["J_mean"],
                         "elapsed_s": round(dt, 3)})
            print(f"  B={BUDGET} {name:6s} seed={seed:3d}  "
                  f"J=({rows[-1]['J_mean'][0]:.6f}, {rows[-1]['J_mean'][1]:.6f})  {dt:.2f}s")
    total = time.perf_counter() - t_all
    print(f"[time] {len(rows)} cell 共 {total:.2f} s（{'冒烟' if args.quick else '完整'}）")

    expected = len(seeds) * len(PREDS)
    if args.quick:
        out = ROOT / "outputs" / "62_min_repro.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps({"verdict": "SMOKE_ONLY", "budget": BUDGET, "mode": MODE,
                                   "episodes": n_ep, "rows": rows,
                                   "elapsed_s": round(total, 3)}, ensure_ascii=False, indent=1),
                       encoding="utf-8")
        print(f"\n[quick] 冒烟：跑了 {len(rows)} cell，**没有比对**（证据是 30 集）⇒ 这只说明流程能跑通。")
        print("SMOKE ONLY: 流程冒烟，不是复现验收（不给出匹配判词）")
        return EXIT_SMOKE

    # ---------------- 逐位比对 ----------------
    print("\n[compare] 与冻结证据逐 episode 比对")
    compared = skipped = failed = invalid = 0
    reasons: list[str] = []
    for r in rows:
        tag = f"seed={r['seed']} {r['predictor']}"
        # 本次输出必须有限，否则不进比对（NaN 差值恒为 NaN，`> 0` 为 False，会被当成"一致"）
        a = np.asarray(r["J_per_episode"], dtype=float)
        if a.size == 0 or not _finite(a):
            invalid += 1
            reasons.append(f"{tag}: 本次输出含非有限值/空")
            print(f"  [INVALID] {tag}: 本次输出非有限或为空 ⇒ 不可比对")
            continue
        hit = None
        for c in ref["cells_J"]:
            if (c.get("seed_set") == "primary" and c.get("budget") == BUDGET
                    and c.get("mode") == MODE and c.get("predictor") == r["predictor"]
                    and c.get("seed") == r["seed"]):
                hit = c
                break
        if hit is None:
            skipped += 1
            reasons.append(f"{tag}: 冻结证据里没有这一格")
            print(f"  [SKIP] {tag}: 冻结证据里没有这一格 ⇒ 未比对")
            continue
        b = np.asarray(hit["J_per_episode"], dtype=float)
        if b.size == 0 or not _finite(b):
            invalid += 1
            reasons.append(f"{tag}: 参考含非有限值/空")
            print(f"  [INVALID] {tag}: 参考数组非有限或为空 ⇒ 不可比对")
            continue
        if a.shape != b.shape:
            failed += 1
            reasons.append(f"{tag}: 形状 {a.shape} vs {b.shape}")
            print(f"  [FAIL] {tag}: 形状 {a.shape} vs {b.shape}")
            continue
        d = np.abs(a - b)
        n_diff = int((d > 0).sum())
        compared += 1
        ok = n_diff == 0
        failed += (not ok)
        if not ok:
            reasons.append(f"{tag}: {n_diff} 个 episode 不一致（max|Δ|={d.max():.3e}）")
        print(f"  [{'PASS' if ok else 'FAIL'}] {tag:18s} n={a.size}  max|Δ|={d.max():.3e}  不一致 {n_diff} 个")

    out = ROOT / "outputs" / "62_min_repro.json"
    out.parent.mkdir(exist_ok=True)
    verdict = ("BITWISE MATCH"
               if (expected > 0 and compared == expected and skipped == 0
                   and failed == 0 and invalid == 0)
               else ("MISMATCH" if failed else "NOT VERIFIED"))
    out.write_text(json.dumps({"verdict": verdict, "expected": expected, "compared": compared,
                               "skipped": skipped, "failed": failed, "invalid": invalid,
                               "reasons": reasons,
                               "budget": BUDGET, "mode": MODE, "episodes": n_ep,
                               "rows": rows, "elapsed_s": round(total, 3),
                               "ref": ref_path.as_posix(), "ref_source": src_kind,
                               "ref_sha256_norm": ref_note},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[out] {out.relative_to(ROOT).as_posix()}（outputs/ 不入库）")

    print(f"\n[count] expected={expected} compared={compared} skipped={skipped} "
          f"failed={failed} invalid={invalid}")
    if reasons:
        print("  原因：" + "；".join(reasons[:6]))
    if verdict == "BITWISE MATCH":
        print("\nBITWISE MATCH")
        return EXIT_MATCH
    if failed:
        print(f"\nMISMATCH: {failed} 个 cell 不一致")
        return EXIT_MISMATCH
    print("\nNOT VERIFIED: 有格子没被比对/数据不可用（缺证据、缺格、非有限）⇒ 不得声明匹配")
    return EXIT_NOT_VERIFIED


if __name__ == "__main__":
    sys.exit(main())
