#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""T27 · 门控回退的闭环 A/B：`age >= θ → ZOH`，否则 `Linear`。

口径来源（★ 全部在看结果之前冻结）
------------------------------------
`04-实验与代码/2026-10-20-T27-门控回退闭环-预注册-v0.1.md` §0–§9。

一句话
------
T25 说"按误差门控触发率为 0"、T26 说"`age` 与危害同号且小 `age` 恒无害" ⇒
唯一的候选规则是 `age` 阈值。本轮**闭环测它**：能否把 `B=100` 的 22.3% 危害收窄，
同时不破坏 `B=200` 的 0.315 收益。

设计（预注册 §2）
-----------------
    臂 = age；mode = unpaced；budget ∈ {200, 100}
    预测器 = zoh / linear / gated(θ ∈ {2,4,6,8})     ← θ=4 为主，其余**只报不判**
    种子 {210, 240, 270} × 30 episodes = **90 个从未出现过的初值**（210–299 三段）
    episodes=30, max_steps=200, warmup=[300,0], absorbed=(25,1), GATE=0.90（全部不动）

判据（预注册 §5，**看数之前定死**）
-----------------------------------
    CAL  linear 新初值 B=200 `R_uav` ∈ [0.20, 0.45]，否则判词加 `DEV-ONLY`
    F1   B=100：`ΔR = R_linear − R_gated4` 的 95% 区间下界 > 0 ⇒ 显著收窄
    F2   B=200：`R_gated4` 的 95% 区间上界 ≤ 0.337 ⇒ 不破坏好档
    F3   CartPole 逐用户 `R_cart`，点估计 >1 且区间下界 >1 ⇒ 必须写"交易"
    F4   θ 敏感度只报告；仅单点过关不得写"门控有效"

用法
----
    D:/devtools/miniconda3/envs/ai-lab/python.exe scripts/68_t27_gated_fallback.py
    ... --skip-experiment      # 只跑 G0–G2/G4 前置闸（不写 outputs/ 与 results/）
    ... --smoke                # 2 集 × 1 种子，不写 results/

产物
----
    outputs/68_t27_gated_fallback.json（仅真跑时写，不入库）
    results/runs/<run_id>/68_t27_gated_fallback.json（**入库**，results_io 唯一入口）
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

from wmlab.eval.multiuser import default_uav_cost, delivery_rate, window_cost  # noqa: E402
from wmlab.eval.predictors import Gated, Linear, ZOH  # noqa: E402
from wmlab.eval.results_io import write_result  # noqa: E402

FAILS: list[str] = []

N_EP = 30
SEEDS = (210, 240, 270)               # 初值 210–239 / 240–269 / 270–299，90 个互不相同
FOREIGN_SEEDS = ((0, 32), (100, 190), (200, 202))   # 开发集 / P3-0 主确认集 / T26 用过的
THETAS = (2, 4, 6, 8)
MAIN_THETA = 4
ARMS = ("zoh", "linear") + tuple(f"gated{t}" for t in THETAS)
BUDGETS = (200, 100)
N_BOOT = 10000
BOOT_SEED = 20261020

#: 已发表参考（P3-0 `58ee7b7`，`results/runs/20261020-050355-613f4be/58_p3_confirmation.json`）
REF = {"linear_B200": {"R": 0.315, "ci": (0.295, 0.337)},
       "linear_B100": {"R": 1.223, "ci": (1.167, 1.279)}}
CAL_BAND = (0.20, 0.45)
F2_UPPER = 0.337
F2_MILD = 0.40
F1_STRICT_UPPER = 1.10


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


T12 = _load(ROOT / "scripts" / "42_t12_hetero_four_arms.py", "t12_mod68")
T52 = _load(ROOT / "scripts" / "52_t21_paced_matching_ablation.py", "t52_mod68")

WINDOW = T12.WINDOW
WARMUP_UAV = T12.WARMUP_UAV
GATE = T12.GATE
ABSORBED = [T12.ABSORBED_UAV, T12.ABSORBED_CART]
COSTS = [default_uav_cost, T12.cartpole_cost]
ARM = "age"


def check(tag: str, ok: bool, detail: str) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {tag}: {detail}")
    if not ok:
        FAILS.append(f"{tag}: {detail}")


# ================================================================ 装置
def _make(arm: str):
    """臂名 → 预测器实例（`gatedT` → `Gated(T)`）。"""
    if arm == "zoh":
        return ZOH()
    if arm == "linear":
        return Linear()
    if arm.startswith("gated"):
        return Gated(int(arm[5:]))
    raise ValueError(f"未知臂 {arm!r}")


class _Probe:
    """逐决策点记录 `age` 直方图（逐用户），用于 G2 的接线核验与 M5 的触发份额。"""

    def __init__(self) -> None:
        self.hist: dict[int, dict[int, int]] = {0: {}, 1: {}}

    def __call__(self, p: dict) -> None:
        for i in (0, 1):
            if not p["alive"][i]:
                continue
            a = int(p["age"][i])
            h = self.hist[i]
            h[a] = int(h.get(a, 0)) + 1

    def n_ge(self, theta: int) -> dict[int, int]:
        return {i: int(sum(v for a, v in self.hist[i].items() if a >= theta)) for i in (0, 1)}

    def n_all(self) -> dict[int, int]:
        return {i: int(sum(self.hist[i].values())) for i in (0, 1)}

    def share_ge(self, theta: int) -> dict[int, float]:
        out = {}
        for i in (0, 1):
            n = sum(self.hist[i].values())
            out[i] = float(sum(v for a, v in self.hist[i].items() if a >= theta) / n) if n else float("nan")
        return out


def _run(budget, seed, n_ep, *, predictor, probe):
    envs = [T12.make_env("uav", seed=seed, max_steps=WARMUP_UAV + WINDOW),
            T12.make_env("CartPole-v1", seed=seed)]
    ctrls = [T12.PDRelativeController(), T12.CartPoleLQR(_run.K)]
    alloc = T52.make_alloc(ARM, "unpaced", budget)
    try:
        return T12.run_multiuser(
            envs, ctrls, alloc, T12.SINRSuccess(T12.Q_REF),
            n_episodes=n_ep, seed=seed, max_steps=WINDOW,
            warmup_steps=[WARMUP_UAV, 0], budget_total=int(budget),
            csi=T12.make_csi(seed, n_ep), cost_fn=COSTS, diagnostic=False,
            counterfactual=False, predictor=predictor, pred_probe=probe)
    finally:
        for e in envs:
            e.close()


_run.K = T12.load_lqr_gain()


def _J(records, n_ep):
    return np.asarray(window_cost(records, 2, window=WINDOW, absorbed=ABSORBED,
                                  n_episodes=n_ep), dtype=float)


# ================================================================ 统计
def _mat(rows_cluster: list[int], maxs: int) -> np.ndarray:
    cl = sorted(set(rows_cluster))
    idx = {c: [] for c in cl}
    for k, c in enumerate(rows_cluster):
        idx[c].append(k)
    m = np.full((len(cl), maxs), -1, dtype=np.int64)
    for k, c in enumerate(cl):
        m[k, :len(idx[c])] = idx[c]
    return m


def cluster_bootstrap_multi(Jnum: np.ndarray, Jden: np.ndarray, clusters: list[int],
                            keys: tuple[str, ...], n_boot: int = N_BOOT,
                            seed: int = BOOT_SEED) -> dict:
    """多个分子共用**同一次重采样**（配对）⇒ `ΔR` 的区间才能做差。

    Args:
        Jnum: **(n_rows, K, 2)** —— 每个候选臂 × 每个用户的逐 episode 代价（K=候选臂数，
              最后一维是用户：0=UAV / 1=CartPole）。第一版写成 `(n_rows, 2K)` 会让
              `dn[None,:]` 与 `safe[None,:]` 广播失败 —— 形状一错就直接抛，而不是静默算错，
              这是好事，但构造阶段就该对齐。
        Jden: (n_rows, 2) 基线（zoh）。
        clusters: 每行所属初值（重采样单位）。
    """
    assert Jnum.ndim == 3 and Jnum.shape[2] == Jden.shape[1] == 2, \
        f"Jnum 必须是 (n,K,2)、Jden (n,2)，收到 {Jnum.shape} / {Jden.shape}"
    mat = _mat(list(clusters), max(len([c for c in clusters if c == k]) for k in set(clusters)))
    rng = np.random.default_rng(int(seed))
    n_cl = mat.shape[0]

    def _ratio(ii):
        dn = Jden[ii].sum(axis=0)                        # (2,)
        num = Jnum[ii].sum(axis=0)                       # (K,2)
        safe = np.where(dn > 0, dn, 1.0)
        return np.where(dn[None, :] > 0, num / safe[None, :], np.nan)

    point = _ratio(np.arange(Jnum.shape[0], dtype=np.int64))
    draws = np.full((n_boot, len(keys), 2), np.nan)
    for b in range(n_boot):
        pick = rng.integers(0, n_cl, n_cl)
        ii = mat[pick].ravel()
        draws[b] = _ratio(ii[ii >= 0])
    out = {"point": {k: [float(x) for x in point[j]] for j, k in enumerate(keys)},
           "ci": {}, "draws": draws, "keys": keys, "n_clusters": int(n_cl),
           "n_rows": int(Jnum.shape[0]), "n_boot": int(n_boot), "boot_seed": int(seed)}
    for j, k in enumerate(keys):
        lo = np.nanpercentile(draws[:, j, :], 2.5, axis=0)
        hi = np.nanpercentile(draws[:, j, :], 97.5, axis=0)
        out["ci"][k] = [[float(x) for x in lo], [float(x) for x in hi]]
    return out


def diff_ci(draws: np.ndarray, j_a: int, j_b: int) -> list[list[float]]:
    d = draws[:, j_a, :] - draws[:, j_b, :]
    return [[float(x) for x in np.nanpercentile(d, 2.5, axis=0)],
            [float(x) for x in np.nanpercentile(d, 97.5, axis=0)]]


# ================================================================ 前置自检
def prechecks(n_ep_small: int = 2) -> dict:
    print("\n== 前置闸 G0a/G0b/G0c/G1/G2/G4（预注册 §7；每轮写明能打死什么错）==")
    out: dict = {}
    s0 = SEEDS[0]

    j_zoh = _J(_run(200, s0, n_ep_small, predictor=ZOH(), probe=None)["records"], n_ep_small)
    j_lin = _J(_run(200, s0, n_ep_small, predictor=Linear(), probe=None)["records"], n_ep_small)

    # G0c 既有路径未被污染
    j_none = _J(_run(200, s0, n_ep_small, predictor=None, probe=None)["records"], n_ep_small)
    d = float(np.max(np.abs(j_none - j_zoh)))
    check("G0c `predictor=None` ≡ `ZOH` 逐位（本轮没改既有路径）", d == 0.0, f"max|Δ|={d:.3e}")

    # G0a θ=0 ⇒ 恒回退 ⇒ 必须 ≡ ZOH
    j_g0 = _J(_run(200, s0, n_ep_small, predictor=Gated(0), probe=None)["records"], n_ep_small)
    d = float(np.max(np.abs(j_g0 - j_zoh)))
    check("G0a `gated(θ=0)` ≡ `ZOH` 逐位（打「门控没接线 / 恒走外推」）", d == 0.0, f"max|Δ|={d:.3e}")

    # G0b θ=1e9 ⇒ 恒外推 ⇒ 必须 ≡ Linear
    j_g9 = _J(_run(200, s0, n_ep_small, predictor=Gated(10 ** 9), probe=None)["records"], n_ep_small)
    d = float(np.max(np.abs(j_g9 - j_lin)))
    check("G0b `gated(θ=10⁹)` ≡ `Linear` 逐位（打「门控恒开 / 外推分支写错」）", d == 0.0,
          f"max|Δ|={d:.3e}")

    # G1 连跑两次
    a1 = _J(_run(200, s0, 5, predictor=Gated(MAIN_THETA), probe=None)["records"], 5)
    a2 = _J(_run(200, s0, 5, predictor=Gated(MAIN_THETA), probe=None)["records"], 5)
    check("G1 同一 cell 连跑两次逐位相同", np.array_equal(a1, a2),
          f"max|Δ|={float(np.max(np.abs(a1 - a2))):.3e}")

    # G2 门控触发次数 == 探针 age>=θ 决策点数（**逐用户**）
    g2 = {}
    ok2 = True
    for budget in (200, 100):
        for th in THETAS:
            pr = _Probe()
            g = Gated(th)
            _run(budget, s0, n_ep_small, predictor=g, probe=pr)
            exp = pr.n_ge(th)
            got = {int(k): int(v) for k, v in g.describe()["n_hold_by_user"].items()}
            got = {i: int(got.get(i, 0)) for i in (0, 1)}
            ok = (exp == got) and int(sum(exp.values())) == int(g.describe()["n_hold"])
            g2[f"B{budget}/θ{th}"] = {"expected_by_user": {str(k): v for k, v in exp.items()},
                                      "got_by_user": {str(k): v for k, v in got.items()},
                                      "equal": bool(ok)}
            ok2 = ok2 and ok
    check("G2 `gated` 回退次数 == 探针统计的 `age>=θ` 决策点数（逐预算 × 逐 θ × 逐用户）", ok2,
          json.dumps({k: {"exp": v["expected_by_user"], "got": v["got_by_user"]}
                      for k, v in g2.items()}, ensure_ascii=False))
    out["G2"] = g2

    # G4 初值集合不相交（90 个互不相同）
    fp = {s + e for s in SEEDS for e in range(N_EP)}
    foreign = set()
    for lo, hi in FOREIGN_SEEDS:
        foreign |= set(range(lo, hi))
    inter = sorted(fp & foreign)
    check("G4 新初值 90 个互不相同、且与开发集/主确认集/T26 初值交集为空",
          len(fp) == 90 and not inter,
          f"n={len(fp)}；∩历史={inter[:8]}{'…' if len(inter) > 8 else ''}（历史集大小 {len(foreign)}）")
    out["G4"] = {"n": len(fp), "n_foreign": len(foreign), "intersection": inter}
    return out


# ================================================================ 单 cell
def run_cell(budget, arm, seed, n_ep):
    t0 = time.time()
    pred = _make(arm)
    pr = _Probe()
    r = _run(budget, seed, n_ep, predictor=pred, probe=pr)
    rec = r["records"]
    J = _J(rec, n_ep)
    dr = delivery_rate(rec, 2)
    served = [sum(1 for x in rec if x.attempted_idx == i) for i in range(2)]
    late = [sum(1 for x in rec if x.t > budget and x.attempted_idx == i) for i in range(2)]
    desc = pred.describe()
    return {"budget": int(budget), "arm": arm, "seed": int(seed), "mode": "unpaced",
            "J_per_episode": np.asarray(J, dtype=float).tolist(),
            "J_mean": np.asarray(J, dtype=float).mean(axis=0).tolist(),
            "served": served, "served_late": late, "served_total": int(sum(served)),
            "alive_steps": dr["alive_steps"], "frames": dr["frames"],
            "budget_reserved": dr["budget_reserved"],
            "budget_attempted": dr["budget_attempted"],
            "budget_wasted": dr["budget_wasted"],
            "n_hold": int(desc.get("n_hold", 0)), "n_extrap": int(desc.get("n_extrap", 0)),
            "gate_share_ge": {str(th): {str(k): v for k, v in pr.share_ge(th).items()}
                              for th in THETAS},
            "n_decision": {str(k): v for k, v in pr.n_all().items()},
            "elapsed_s": round(time.time() - t0, 2)}


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
    arms = ("zoh", "linear", f"gated{MAIN_THETA}") if args.smoke else ARMS
    budgets = (200,) if args.smoke else BUDGETS

    t_all = time.time()
    print(f"[68] T27 门控回退｜窗口 {WINDOW}｜臂={ARM}｜unpaced｜seeds={seeds}｜episodes={n_ep}｜"
          f"arms={arms}｜budgets={budgets}")
    pre = prechecks()

    if args.skip_experiment:
        print(f"\n{'前置闸全 PASS' if not FAILS else 'FAILED: ' + '; '.join(FAILS)}")
        return 0 if not FAILS else 2

    cells: list[dict] = []
    for budget in budgets:
        for arm in arms:
            for seed in seeds:
                c = run_cell(budget, arm, seed, n_ep)
                cells.append(c)
                print(f"  B={budget:3d} {arm:8s} seed={seed} "
                      f"J=({c['J_mean'][0]:.3f},{c['J_mean'][1]:.3f}) "
                      f"served=({c['served'][0]},{c['served'][1]}) "
                      f"late=({c['served_late'][0]},{c['served_late'][1]}) "
                      f"hold={c['n_hold']} {c['elapsed_s']}s")

    def get(budget, arm, seed):
        for c in cells:
            if (c["budget"], c["arm"], c["seed"]) == (budget, arm, seed):
                return c
        return None

    # ---------- 统计：每个预算一次联合 bootstrap（配对）----------
    stats: dict = {}
    for budget in budgets:
        cand = [a for a in arms if a != "zoh"]
        if not cand:
            continue
        rows_J: list[list[float]] = []
        rows_D: list[list[float]] = []
        clusters: list[int] = []
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
        bs = cluster_bootstrap_multi(Jnum, Jden, clusters, keys)
        per = {}
        for j, a in enumerate(keys):
            R = bs["point"][a]
            by_seed = {}
            for s in seeds:
                c = get(budget, a, s)
                z = get(budget, "zoh", s)
                if c is None or z is None:
                    continue
                jm = np.asarray(c["J_mean"], dtype=float)
                zm = np.asarray(z["J_mean"], dtype=float)
                by_seed[str(s)] = [float(x) for x in np.where(zm > 0, jm / np.where(zm > 0, zm, 1), np.nan)]
            pred_late = float(np.mean([sum(get(budget, a, s)["served_late"]) for s in seeds]))
            per[a] = {
                "R": R, "ci95": bs["ci"][a],
                "per_seed_R": by_seed,
                "n_seeds_uav_below_1": int(sum(1 for v in by_seed.values() if v[0] < 1.0)),
                "n_seeds": len(by_seed),
                "J_mean": [float(np.mean([get(budget, a, s)["J_mean"][i] for s in seeds]))
                           for i in (0, 1)],
                "served_mean": [float(np.mean([get(budget, a, s)["served"][i] for s in seeds]))
                                for i in (0, 1)],
                "alive_mean": [float(np.mean([get(budget, a, s)["alive_steps"][i] for s in seeds]))
                               for i in (0, 1)],
                "served_late_mean": pred_late,
                "gate_share": {th: float(np.mean([get(budget, a, s)["gate_share_ge"][str(th)]["0"]
                                                  for s in seeds])) for th in THETAS}
                if a.startswith("gated") else None,
            }
        # 逐 cluster 比值（分布无关）
        for a in keys:
            by_cl: dict[int, list] = {}
            for s in seeds:
                c = get(budget, a, s)
                z = get(budget, "zoh", s)
                for e in range(len(z["J_per_episode"])):
                    jn = np.asarray(c["J_per_episode"][e], dtype=float)
                    jd = np.asarray(z["J_per_episode"][e], dtype=float)
                    by_cl.setdefault(int(s) + e, []).append(
                        [np.nan if x <= 0 else y / x for x, y in zip(jd, jn)])
            clm = np.asarray([np.nanmean(np.asarray(v, dtype=float), axis=0) for v in by_cl.values()])
            per[a]["frac_clusters_below_1"] = [float(np.mean(clm[:, i] < 1.0)) for i in (0, 1)]
            per[a]["cluster_ratio_median"] = [float(np.nanmedian(clm[:, i])) for i in (0, 1)]
        # ΔR（配对）
        g4 = f"gated{MAIN_THETA}"
        if "linear" in keys and g4 in keys:
            jl, jg = keys.index("linear"), keys.index(g4)
            per["delta_linear_minus_gated4"] = {
                "point": [float(bs["point"]["linear"][i] - bs["point"][g4][i]) for i in (0, 1)],
                "ci95": diff_ci(bs["draws"], jl, jg),
                "note": "同一次 cluster 重采样内相减（配对），uav=0/cart=1"}
        stats[str(budget)] = {"per_arm": per, "n_clusters": bs["n_clusters"],
                              "n_rows": bs["n_rows"], "n_boot": bs["n_boot"]}

    print("\n== 汇总：R_i = Σ J_i / Σ J_i(zoh)，95% cluster bootstrap（90 初值）==")
    for budget in budgets:
        for a, d in stats[str(budget)]["per_arm"].items():
            if "R" not in d:
                continue
            print(f"  B={budget:3d} {a:22s} R=({d['R'][0]:.3f},{d['R'][1]:.3f}) "
                  f"CI_uav=[{d['ci95'][0][0]:.3f},{d['ci95'][1][0]:.3f}] "
                  f"CI_cart=[{d['ci95'][0][1]:.3f},{d['ci95'][1][1]:.3f}] "
                  f"J_uav={d['J_mean'][0]:.3f} seeds<1={d['n_seeds_uav_below_1']}/{d['n_seeds']} "
                  f"late={d['served_late_mean']:.0f}")

    # ---------- 判据 ----------
    print("\n== 判据（预注册 §5）==")
    verdicts: dict = {}
    g4 = f"gated{MAIN_THETA}"
    p200 = stats.get("200", {}).get("per_arm", {})
    p100 = stats.get("100", {}).get("per_arm", {})

    # CAL
    cal = None
    if "linear" in p200:
        r_cal = p200["linear"]["R"][0]
        cal = {"R_uav_linear_new_seeds": r_cal, "band": list(CAL_BAND),
               "in_band": bool(CAL_BAND[0] <= r_cal <= CAL_BAND[1]),
               "ref_primary": REF["linear_B200"]["R"]}
        print(f"  CAL linear 新初值 B=200 R_uav={r_cal:.3f} ∈ [{CAL_BAND[0]},{CAL_BAND[1]}]: "
              f"{cal['in_band']}（主确认集=0.315）")
    verdicts["CAL"] = cal
    suffix = "" if (cal and cal["in_band"]) else "（DEV-ONLY：新初值分布与主集不同）"

    # F1
    f1 = {"verdict": "N/A"}
    if g4 in p100 and "linear" in p100 and "delta_linear_minus_gated4" in p100:
        dr = p100["delta_linear_minus_gated4"]
        lo, hi = dr["ci95"][0][0], dr["ci95"][1][0]
        rg_hi = p100[g4]["ci95"][1][0]
        if lo > 0 and rg_hi < F1_STRICT_UPPER:
            v = "HARM-ELIMINATED"
        elif lo > 0:
            v = "HARM-REDUCED"
        elif hi < 0:
            v = "WORSE"
        else:
            v = "NO-EFFECT"
        f1 = {"verdict": v, "delta_R_uav": dr["point"][0], "delta_ci": [lo, hi],
              "R_gated_ci": [p100[g4]["ci95"][0][0], p100[g4]["ci95"][1][0]],
              "R_linear_ci": [p100["linear"]["ci95"][0][0], p100["linear"]["ci95"][1][0]],
              "R_linear": p100["linear"]["R"], "R_gated": p100[g4]["R"]}
        print(f"  F1 B=100: ΔR(linear−gated4)={dr['point'][0]:+.3f} CI=[{lo:+.3f},{hi:+.3f}]；"
              f"R_gated CI_uav=[{f1['R_gated_ci'][0]:.3f},{f1['R_gated_ci'][1]:.3f}]"
              f" ⇒ {v}{suffix}")
    else:
        print("  [SKIP] F1：缺 B=100 的 linear/gated4 配对")
    verdicts["F1"] = f1

    # F2
    f2 = {"verdict": "N/A"}
    if g4 in p200:
        hi = p200[g4]["ci95"][1][0]
        lo = p200[g4]["ci95"][0][0]
        if hi <= F2_UPPER:
            v = "NOT-BROKEN"
        elif hi <= F2_MILD:
            v = "DEGRADED-MILD"
        else:
            v = "BROKEN"
        if lo > F2_UPPER:
            v = "BROKEN"
        f2 = {"verdict": v, "ci_uav": [lo, hi], "R": p200[g4]["R"],
              "ref_linear":{"R": REF["linear_B200"]["R"], "ci": list(REF["linear_B200"]["ci"])}}
        print(f"  F2 B=200: R_gated4=({p200[g4]['R'][0]:.3f},{p200[g4]['R'][1]:.3f}) "
              f"CI_uav=[{lo:.3f},{hi:.3f}] vs 门槛 {F2_UPPER} ⇒ {v}{suffix}")
    else:
        print("  [SKIP] F2：缺 B=200 的 gated4")
    verdicts["F2"] = f2

    # F3（CartPole 逐用户）
    f3 = {}
    for budget in budgets:
        for a in [x for x in arms if x != "zoh"]:
            d = stats[str(budget)]["per_arm"].get(a)
            if not d:
                continue
            r = d["R"][1]
            ci_lo = d["ci95"][0][1]
            f3[f"B{budget}/{a}"] = {"R_cart": r, "ci_cart": [ci_lo, d["ci95"][1][1]],
                                    "trade": bool(r > 1.0 and ci_lo > 1.0)}
    print("  F3 CartPole 逐用户 R_cart（点估计>1 且区间下界>1 ⇒ 必须写「交易」）：")
    for k, v in f3.items():
        print(f"      {k:18s} R_cart={v['R_cart']:.3f} CI=[{v['ci_cart'][0]:.3f},"
              f"{v['ci_cart'][1]:.3f}] trade={v['trade']}")
    verdicts["F3"] = f3

    # F4（θ 敏感度，只报）
    f4 = {}
    for budget in budgets:
        for th in THETAS:
            a = f"gated{th}"
            d = stats[str(budget)]["per_arm"].get(a)
            if not d:
                continue
            f4[f"B{budget}/θ{th}"] = {"R_uav": d["R"][0],
                                      "ci_uav": [d["ci95"][0][0], d["ci95"][1][0]],
                                      "R_cart": d["R"][1],
                                      "ci_cart": [d["ci95"][0][1], d["ci95"][1][1]]}
    print("  F4 θ 敏感度（只报告，不判）：")
    for k, v in f4.items():
        print(f"      {k:12s} R_uav={v['R_uav']:.3f} CI_uav=[{v['ci_uav'][0]:.3f},"
              f"{v['ci_uav'][1]:.3f}]")
    n_f1_pass = sum(1 for th in THETAS
                    if f"B100/θ{th}" in f4 and f4[f"B100/θ{th}"]["ci_uav"][1] < 1.0)
    verdicts["F4"] = {"table": f4, "n_theta_b100_ci_upper_lt_1": n_f1_pass,
                      "note": "仅单点过关时只能写「该 θ 的探索性结果」"}

    overall = ("FALLBACK-WORKS" if (f1.get("verdict") in ("HARM-ELIMINATED", "HARM-REDUCED")
                                    and f2.get("verdict") == "NOT-BROKEN")
               else "FALLBACK-FAILS")
    verdicts["overall"] = {"verdict": overall + ("" if not suffix else " (DEV-ONLY)"),
                           "rule": "F1∈{HARM-ELIMINATED,HARM-REDUCED} 且 F2=NOT-BROKEN ⇒ WORKS"}
    print(f"\n  ⇒ 综合判词：{verdicts['overall']['verdict']}")

    # ---------- 后置闸 ----------
    print("\n== 后置闸 G3/G5 ==")
    bad1 = [c for c in cells if c["served_total"] != n_ep * c["budget"]]
    bad2 = [c for c in cells if c["budget_reserved"]
            != c["budget_attempted"] + c["budget_wasted"]]
    bad3 = [c for c in cells if c["budget"] < WINDOW and sum(c["served_late"]) != 0]
    check("G3 记账：总发帧==episodes×budget、reserved==attempted+wasted、"
          "unpaced B<L 的 t>B 发帧恒 0",
          not bad1 and not bad2 and not bad3,
          f"{len(cells)} cells｜bad={len(bad1)}/{len(bad2)}/{len(bad3)}")
    bad5 = []
    for c in cells:
        j = np.asarray(c["J_per_episode"], dtype=float)
        if j.shape != (n_ep, 2) or not np.all(np.isfinite(j)) or np.any(j < 0):
            bad5.append(f"B{c['budget']}/{c['arm']}/s{c['seed']}:shape={j.shape}")
    check("G5 逐 episode J 形状 == (episodes,2)、全有限、非负", not bad5,
          "全 cell" if not bad5 else "；".join(bad5[:4]))

    payload = {"config": {
        "N": 2, "window": WINDOW, "warmup": [WARMUP_UAV, 0], "budgets": list(budgets),
        "episodes": n_ep, "seeds": list(seeds), "arms": list(arms), "mode": "unpaced",
        "gate_theta_main": MAIN_THETA, "gate_theta_scan": list(THETAS),
        "arm_alloc": ARM, "gate": GATE, "absorbed": [float(x) for x in ABSORBED],
        "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
        "statistic": "R_i = Σ_e J_i / Σ_e J_i(zoh)，配对到 (seed,episode)；cluster bootstrap 单位=初值",
        "criteria": "预注册 §5：CAL / F1(ΔR>0) / F2(上界≤0.337) / F3(逐用户) / F4(只报)",
        "prereg": "04-实验与代码/2026-10-20-T27-门控回退闭环-预注册-v0.1.md",
        "ref": REF,
    }, "prechecks": pre, "checks_failed": list(FAILS), "summary": stats,
        "verdicts": verdicts}

    _dump(payload, cells, n_ep, run_id=args.run_id,
          allow_overwrite=args.allow_overwrite, persist=not args.smoke)
    print(f"\n{'ALL CHECKS PASS' if not FAILS else 'FAILED: ' + '; '.join(FAILS)}")
    print(f"总耗时 {round(time.time() - t_all, 1)}s")
    return 0 if not FAILS else 2


def _dump(payload, cells, n_ep, *, run_id=None, allow_overwrite=False, persist=True):
    out = dict(payload)
    (ROOT / "outputs").mkdir(exist_ok=True)
    (ROOT / "outputs" / "68_t27_gated_fallback.json").write_text(
        json.dumps({**out, "cells": (cells or [])}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    if cells is None or not persist:
        print("[smoke] 不写 results/（smoke 数字不进证据链）")
        return
    keep = ("budget", "mode", "arm", "seed", "J_per_episode", "J_mean", "served",
            "served_late", "served_total", "alive_steps", "frames", "budget_reserved",
            "budget_attempted", "budget_wasted", "n_hold", "n_extrap", "gate_share_ge",
            "n_decision", "elapsed_s")
    slim = [{k: c[k] for k in keep} for c in cells]
    dest = write_result("68_t27_gated_fallback", {**out, "cells_J": slim},
                        run_id=run_id, allow_overwrite=allow_overwrite)
    print(f"[evidence] {dest.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    sys.exit(main())
