#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""T28 · θ=8 的独立确认（第四批初值 310–399）。

一句话
------
T27 在第三批初值（210–299）上扫出 θ=8 是**唯一两档都过**的点，但它是 4 点扫描的单点
⇒ 预注册 N3 只能写"探索性"。本轮把 θ 固定为 8、初值换成**从未出现过**的第四批，
只回答一个问题：**这个点在独立批上还成立吗？**

口径来源（★ 全部在看结果之前冻结）
------------------------------------
`04-实验与代码/2026-10-21-T28-θ8独立确认-预注册-v0.1.md` §0–§7。

为什么"复用"而不是"重写"
------------------------
物理与统计实现与 T27 必须是**同一份**，否则"确认"会退化成"两个实现各自给个数"。
因此本脚本 `importlib` 载入 `scripts/68_t27_gated_fallback.py`，只覆盖五个常量：
    SEEDS / THETAS / MAIN_THETA / ARMS / FOREIGN_SEEDS
并把这三份文件的 sha256 与 git HEAD 记进证据（G7）。**不改 `68`、不改 `wmlab/`**。

用法
----
    D:/devtools/miniconda3/envs/ai-lab/python.exe scripts/69_t28_confirm.py
    ... --skip-experiment      # 只跑 G0–G4 前置闸，results/ 与 outputs/ 零写入（G8）
    ... --smoke                # 2 集 × 1 种子，不写 results/

产物
----
    outputs/69_t28_confirm.json（仅真跑时写，不入库）
    results/runs/<run_id>/69_t28_confirm.json（**入库**，results_io 唯一入口）
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from wmlab.eval.results_io import write_result  # noqa: E402

# ================================================================ 常量（预注册 §2）
THETA = 8
SEEDS = (310, 340, 370)          # 310–339 / 340–369 / 370–399 = 90 个初值
N_EP = 30
BUDGETS = (200, 100)
ARMS = ("zoh", "linear", f"gated{THETA}")
#: 历史初值集合：开发集 / P3-0 主确认集 / T26 / T27 第三批
FOREIGN_SEEDS = ((0, 32), (100, 190), (200, 202), (210, 300))

CAL_BAND = (0.20, 0.45)          # 与 T27 同带
T1_UPPER = 0.337                 # = 主确认集 linear 的区间上界（P3-0）
T2_UPPER = 1.10                  # P3-0 C4 预注册的实际等价界
EPOCH_TOKENS = ("seed",)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


T27 = _load(ROOT / "scripts" / "68_t27_gated_fallback.py", "t27_mod69")

# ★ 只覆盖五个常量：物理/统计实现与 T27 完全同一份
T27.SEEDS = SEEDS
T27.THETAS = (THETA,)
T27.MAIN_THETA = THETA
T27.ARMS = ARMS
T27.FOREIGN_SEEDS = FOREIGN_SEEDS
T27.REF  # 保留引用（CAL 的对照值来自 P3-0，不改）

FAILS: list[str] = T27.FAILS          # 与 68 共用同一个失败列表（prechecks 往里写）


def check(tag: str, ok: bool, detail: str) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {tag}: {detail}")
    if not ok:
        FAILS.append(f"{tag}: {detail}")


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT),
                              capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as exc:                                    # pragma: no cover
        return f"<unavailable: {exc}>"


# ================================================================ 本轮专属闸
def gates_t28() -> dict:
    print("\n== 本轮专属闸 G6/G7（预注册 §6）==")
    fp = {s + e for s in SEEDS for e in range(N_EP)}
    g6_ok = (len(fp) == 90) and min(fp) == 310 and max(fp) == 399
    check("G6 指纹 = 90 个连续初值且闭区间恰为 [310,399]", g6_ok,
          f"n={len(fp)} min={min(fp)} max={max(fp)}")

    s68 = _sha256(ROOT / "scripts" / "68_t27_gated_fallback.py")
    spr = _sha256(ROOT / "wmlab" / "eval" / "predictors.py")
    check("G7 运行时 θ == 8 且复用 68（不改库）", T27.MAIN_THETA == THETA == 8,
          f"θ={T27.MAIN_THETA}｜68={s68[:12]}｜predictors={spr[:12]}")
    return {"G6": {"n": len(fp), "min": int(min(fp)), "max": int(max(fp))},
            "G7": {"theta": int(THETA), "sha256_68": s68, "sha256_predictors": spr,
                   "git_head": _git_head()}}


# ================================================================ 主实验
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=N_EP)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip-experiment", action="store_true")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--allow-overwrite", action="store_true")
    args = ap.parse_args()
    n_ep = 2 if args.smoke else args.episodes
    if n_ep != N_EP and not args.smoke:
        print(f"[WARN] episodes={n_ep} ≠ 预注册 {N_EP}：结果不得标「确认」")
    seeds = (SEEDS[0],) if args.smoke else SEEDS
    arms = ARMS
    budgets = (200,) if args.smoke else BUDGETS

    t_all = time.time()
    print(f"[69] T28 θ=8 独立确认｜臂={T27.ARM}｜unpaced｜θ={THETA}｜seeds={seeds}｜"
          f"episodes={n_ep}｜arms={arms}｜budgets={budgets}")

    gate = gates_t28()

    # G8：--skip-experiment 必须零写入 ⇒ 先把该目录的时间戳记下来
    res_before = sorted(p.name for p in (ROOT / "results" / "runs").glob("*")) \
        if (ROOT / "results" / "runs").exists() else []

    pre = T27.prechecks()

    if args.skip_experiment:
        res_after = sorted(p.name for p in (ROOT / "results" / "runs").glob("*")) \
            if (ROOT / "results" / "runs").exists() else []
        check("G8 `--skip-experiment` 不写 results/", res_before == res_after,
              f"runs 目录条目 {len(res_before)} → {len(res_after)}")
        print(f"\n{'前置闸全 PASS' if not FAILS else 'FAILED: ' + '; '.join(FAILS)}")
        return 0 if not FAILS else 2

    cells: list[dict] = []
    for budget in budgets:
        for arm in arms:
            for seed in seeds:
                c = T27.run_cell(budget, arm, seed, n_ep)
                cells.append(c)
                print(f"  B={budget:3d} {arm:8s} seed={seed} "
                      f"J=({c['J_mean'][0]:.3f},{c['J_mean'][1]:.3f}) "
                      f"served=({c['served'][0]},{c['served'][1]}) "
                      f"late=({c['served_late'][0]},{c['served_late'][1]}) "
                      f"hold={c['n_hold']} alive={c['alive_steps']} {c['elapsed_s']}s")

    def get(budget, arm, seed):
        for c in cells:
            if (c["budget"], c["arm"], c["seed"]) == (budget, arm, seed):
                return c
        return None

    # ---------- 统计：与 T27 同一份实现（配对 cluster bootstrap）----------
    stats: dict = {}
    for budget in budgets:
        cand = [a for a in arms if a != "zoh"]
        rows_J, rows_D, clusters = [], [], []
        for s in seeds:
            z = get(budget, "zoh", s)
            if z is None:
                continue
            for e in range(len(z["J_per_episode"])):
                rows_D.append([float(x) for x in z["J_per_episode"][e]])
                rows_J.append([[float(get(budget, a, s)["J_per_episode"][e][i]) for i in (0, 1)]
                               for a in cand])
                clusters.append(int(s) + e)
        Jnum = np.asarray(rows_J, dtype=float)
        Jden = np.asarray(rows_D, dtype=float)
        keys = tuple(cand)
        bs = T27.cluster_bootstrap_multi(Jnum, Jden, clusters, keys)
        per = {}
        for j, a in enumerate(keys):
            by_seed = {}
            for s in seeds:
                c, z = get(budget, a, s), get(budget, "zoh", s)
                if c is None or z is None:
                    continue
                jm = np.asarray(c["J_mean"], dtype=float)
                zm = np.asarray(z["J_mean"], dtype=float)
                by_seed[str(s)] = [float(x) for x in np.where(zm > 0, jm / np.where(zm > 0, zm, 1), np.nan)]
            per[a] = {
                "R": bs["point"][a], "ci95": bs["ci"][a],
                "per_seed_R": by_seed,
                "n_seeds_uav_below_1": int(sum(1 for v in by_seed.values() if v[0] < 1.0)),
                "n_seeds": len(by_seed),
                "J_mean": [float(np.mean([get(budget, a, s)["J_mean"][i] for s in seeds])) for i in (0, 1)],
                "served_mean": [float(np.mean([get(budget, a, s)["served"][i] for s in seeds])) for i in (0, 1)],
                "alive_mean": [float(np.mean([get(budget, a, s)["alive_steps"][i] for s in seeds])) for i in (0, 1)],
                "served_late_mean": float(np.mean([sum(get(budget, a, s)["served_late"]) for s in seeds])),
                "gate_share_theta": float(np.mean([get(budget, a, s)["gate_share_ge"][str(THETA)]["0"]
                                                   for s in seeds])) if a.startswith("gated") else None,
            }
        # 逐 cluster 比值（分布无关）
        for a in keys:
            by_cl: dict[int, list] = {}
            for s in seeds:
                c, z = get(budget, a, s), get(budget, "zoh", s)
                for e in range(len(z["J_per_episode"])):
                    jn = np.asarray(c["J_per_episode"][e], dtype=float)
                    jd = np.asarray(z["J_per_episode"][e], dtype=float)
                    by_cl.setdefault(int(s) + e, []).append(
                        [np.nan if x <= 0 else y / x for x, y in zip(jd, jn)])
            clm = np.asarray([np.nanmean(np.asarray(v, dtype=float), axis=0) for v in by_cl.values()])
            per[a]["frac_clusters_below_1"] = [float(np.mean(clm[:, i] < 1.0)) for i in (0, 1)]
            per[a]["cluster_ratio_median"] = [float(np.nanmedian(clm[:, i])) for i in (0, 1)]
        gk = f"gated{THETA}"
        if "linear" in keys and gk in keys:
            jl, jg = keys.index("linear"), keys.index(gk)
            per[f"delta_linear_minus_{gk}"] = {
                "point": [float(bs["point"]["linear"][i] - bs["point"][gk][i]) for i in (0, 1)],
                "ci95": T27.diff_ci(bs["draws"], jl, jg),
                "note": "同一次 cluster 重采样内相减（配对），uav=0/cart=1"}
        stats[str(budget)] = {"per_arm": per, "n_clusters": bs["n_clusters"],
                              "n_rows": bs["n_rows"], "n_boot": bs["n_boot"]}

    print("\n== 汇总：R_i = Σ J_i / Σ J_i(zoh)，95% cluster bootstrap（90 初值，第四批）==")
    for budget in budgets:
        for a, d in stats[str(budget)]["per_arm"].items():
            if "R" not in d:
                continue
            print(f"  B={budget:3d} {a:10s} R=({d['R'][0]:.3f},{d['R'][1]:.3f}) "
                  f"CI_uav=[{d['ci95'][0][0]:.3f},{d['ci95'][1][0]:.3f}] "
                  f"CI_cart=[{d['ci95'][0][1]:.3f},{d['ci95'][1][1]:.3f}] "
                  f"J_uav={d['J_mean'][0]:.3f} seeds<1={d['n_seeds_uav_below_1']}/{d['n_seeds']} "
                  f"alive_uav={d['alive_mean'][0]:.1f} late={d['served_late_mean']:.0f}")

    # ---------- 判据（预注册 §3）----------
    print("\n== 判据（预注册 §3，看数前定死）==")
    verdicts: dict = {}
    gk = f"gated{THETA}"
    p200 = stats.get("200", {}).get("per_arm", {})
    p100 = stats.get("100", {}).get("per_arm", {})

    cal = None
    if "linear" in p200:
        r_cal = p200["linear"]["R"][0]
        cal = {"R_uav_linear_batch4": r_cal, "band": list(CAL_BAND),
               "in_band": bool(CAL_BAND[0] <= r_cal <= CAL_BAND[1]),
               "ref_primary": T27.REF["linear_B200"]["R"],
               "ref_t27_batch3": 0.308}
        print(f"  CAL linear 第四批 B=200 R_uav={r_cal:.3f} ∈ [{CAL_BAND[0]},{CAL_BAND[1]}]: "
              f"{cal['in_band']}（主确认集 0.315 / T27 第三批 0.308）")
    verdicts["CAL"] = cal
    suffix = "" if (cal and cal["in_band"]) else "（DEV-ONLY：第四批分布与主集不同）"

    t1 = {"verdict": "N/A"}
    if gk in p200:
        lo, hi = p200[gk]["ci95"][0][0], p200[gk]["ci95"][1][0]
        t1 = {"verdict": "GOOD-KEPT" if hi <= T1_UPPER else "BROKEN",
              "R_uav": p200[gk]["R"][0], "ci_uav": [lo, hi], "upper_gate": T1_UPPER,
              "J_uav": p200[gk]["J_mean"][0],
              "J_uav_zoh": p200["zoh"]["J_mean"][0] if "zoh" in p200 else None,
              "alive_uav": p200[gk]["alive_mean"][0],
              "alive_uav_zoh": p200["zoh"]["alive_mean"][0] if "zoh" in p200 else None}
        print(f"  T1 B=200: R_gated8={t1['R_uav']:.3f} CI_uav=[{lo:.3f},{hi:.3f}] vs 门槛 {T1_UPPER}"
              f" ⇒ {t1['verdict']}{suffix}")
    else:
        print("  [SKIP] T1：缺 B=200 的 gated8")
    verdicts["T1"] = t1

    t2 = {"verdict": "N/A"}
    if gk in p100:
        lo, hi = p100[gk]["ci95"][0][0], p100[gk]["ci95"][1][0]
        pt = p100[gk]["R"][0]
        if hi < T2_UPPER and pt < 1.0:
            v = "HARM-CONTAINED"
        elif hi < T2_UPPER:
            v = "EQUIVALENT-NO-GAIN"
        else:
            v = "NOT-CONTAINED"
        lin = p100.get("linear", {})
        t2 = {"verdict": v, "R_uav": pt, "ci_uav": [lo, hi], "upper_gate": T2_UPPER,
              "J_uav": p100[gk]["J_mean"][0],
              "J_uav_zoh": p100["zoh"]["J_mean"][0] if "zoh" in p100 else None,
              "J_uav_linear": lin.get("J_mean", [None])[0],
              "R_uav_linear": lin.get("R", [None])[0],
              "alive_uav": p100[gk]["alive_mean"][0],
              "alive_uav_zoh": p100["zoh"]["alive_mean"][0] if "zoh" in p100 else None,
              "R_cart": p100[gk]["R"][1], "ci_cart": [p100[gk]["ci95"][0][1], p100[gk]["ci95"][1][1]]}
        print(f"  T2 B=100: R_gated8={pt:.3f} CI_uav=[{lo:.3f},{hi:.3f}]（门槛 上界<{T2_UPPER} 且 点<1.0）"
              f"｜linear={lin.get('R', [float('nan')])[0]:.3f} ⇒ {v}{suffix}")
    else:
        print("  [SKIP] T2：缺 B=100 的 gated8")
    verdicts["T2"] = t2

    # F3 口径的 CartPole 逐用户（必须能写出"交易"）
    f3 = {}
    for budget in budgets:
        for a in [x for x in arms if x != "zoh"]:
            d = stats[str(budget)]["per_arm"].get(a)
            if not d:
                continue
            f3[f"B{budget}/{a}"] = {"R_cart": d["R"][1],
                                    "ci_cart": [d["ci95"][0][1], d["ci95"][1][1]],
                                    "trade": bool(d["R"][1] > 1.0 and d["ci95"][0][1] > 1.0)}
    print("  CartPole 逐用户（点>1 且下界>1 ⇒ 必须写「交易」）：")
    for k, v in f3.items():
        print(f"      {k:16s} R_cart={v['R_cart']:.3f} CI=[{v['ci_cart'][0]:.3f},{v['ci_cart'][1]:.3f}] "
              f"trade={v['trade']}")
    verdicts["cartpole"] = f3

    # 综合判词（映射写死在预注册 §3）
    if not (cal and cal["in_band"]):
        base = "CAL-FAILS"
    elif t1["verdict"] == "GOOD-KEPT" and t2["verdict"] == "HARM-CONTAINED":
        base = "CONFIRMED"
    elif t1["verdict"] == "GOOD-KEPT" and t2["verdict"] == "EQUIVALENT-NO-GAIN":
        base = "EQUIVALENT-NO-GAIN"
    else:
        base = "FALLBACK-FAILS (confirm)"
    verdicts["overall"] = {"verdict": base + suffix,
                           "rule": "CAL 过 ∧ T1 过 ∧ T2 严格 ⇒ CONFIRMED；T2 只等价 ⇒ EQUIVALENT-NO-GAIN；否则 FAILS"}
    print(f"\n  ⇒ 综合判词：{verdicts['overall']['verdict']}")

    # ---------- 后置闸 G3/G5（同一份实现）----------
    print("\n== 后置闸 G3/G5 ==")
    bad1 = [c for c in cells if c["served_total"] != n_ep * c["budget"]]
    bad2 = [c for c in cells if c["budget_reserved"] != c["budget_attempted"] + c["budget_wasted"]]
    bad3 = [c for c in cells if c["budget"] < T27.WINDOW and sum(c["served_late"]) != 0]
    check("G3 记账：总发帧==episodes×budget、reserved==attempted+wasted、unpaced B<L 的 t>B 发帧恒 0",
          not bad1 and not bad2 and not bad3, f"{len(cells)} cells｜bad={len(bad1)}/{len(bad2)}/{len(bad3)}")
    bad5 = []
    for c in cells:
        j = np.asarray(c["J_per_episode"], dtype=float)
        if j.shape != (n_ep, 2) or not np.all(np.isfinite(j)) or np.any(j < 0):
            bad5.append(f"B{c['budget']}/{c['arm']}/s{c['seed']}:shape={j.shape}")
    check("G5 逐 episode J 形状 == (episodes,2)、全有限、非负", not bad5,
          "全 cell" if not bad5 else "；".join(bad5[:4]))

    payload = {"config": {
        "N": 2, "window": T27.WINDOW, "warmup": [T27.WARMUP_UAV, 0],
        "budgets": list(budgets), "episodes": n_ep, "seeds": list(seeds), "arms": list(arms),
        "mode": "unpaced", "theta": int(THETA), "arm_alloc": T27.ARM, "gate": T27.GATE,
        "absorbed": [float(x) for x in T27.ABSORBED], "n_boot": T27.N_BOOT, "boot_seed": T27.BOOT_SEED,
        "statistic": "R_i = Σ_e J_i / Σ_e J_i(zoh)，配对到 (seed,episode)；cluster bootstrap 单位=初值",
        "criteria": "预注册 §3：CAL(带) / T1(上界≤0.337) / T2(上界<1.10 且 点<1.0)",
        "prereg": "04-实验与代码/2026-10-21-T28-θ8独立确认-预注册-v0.1.md",
        "ref": T27.REF, "cal_band": list(CAL_BAND), "t1_upper": T1_UPPER, "t2_upper": T2_UPPER,
        "reuse_of": "scripts/68_t27_gated_fallback.py（同一份物理与统计实现，仅覆盖 5 个常量）",
    }, "gates": gate, "prechecks": pre, "checks_failed": list(FAILS),
        "summary": stats, "verdicts": verdicts}

    _dump(payload, cells, n_ep, run_id=args.run_id,
          allow_overwrite=args.allow_overwrite, persist=not args.smoke)
    print(f"\n{'ALL CHECKS PASS' if not FAILS else 'FAILED: ' + '; '.join(FAILS)}")
    print(f"总耗时 {round(time.time() - t_all, 1)}s")
    return 0 if not FAILS else 2


def _dump(payload, cells, n_ep, *, run_id=None, allow_overwrite=False, persist=True):
    out = dict(payload)
    (ROOT / "outputs").mkdir(exist_ok=True)
    (ROOT / "outputs" / "69_t28_confirm.json").write_text(
        json.dumps({**out, "cells": (cells or [])}, ensure_ascii=False, indent=2), encoding="utf-8")
    if cells is None or not persist:
        print("[smoke] 不写 results/（smoke 数字不进证据链）")
        return
    keep = ("budget", "mode", "arm", "seed", "J_per_episode", "J_mean", "served",
            "served_late", "served_total", "alive_steps", "frames", "budget_reserved",
            "budget_attempted", "budget_wasted", "n_hold", "n_extrap", "gate_share_ge",
            "n_decision", "elapsed_s")
    slim = [{k: c[k] for k in keep} for c in cells]
    dest = write_result("69_t28_confirm", {**out, "cells_J": slim},
                        run_id=run_id, allow_overwrite=allow_overwrite)
    print(f"[evidence] {dest.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    sys.exit(main())
