#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""T21 · 配速化匹配消融 —— 判"同一批现成臂配上同一配速后，排序与 geo 是否改变"。

口径来源（★ 全部在看结果之前冻结）
------------------------------------
`04-实验与代码/2026-10-19-T21-配速化匹配消融-预注册-v0.1.md` §1–§7。

为什么有这一轮（不是"再换一个臂"）
----------------------------------
T16 已实测：裸臂只要 `budget_left > 0` 就每步派帧 ⇒ `t = B` 时预算归零、后 `L−B` 步零发帧
（`t>B` 发帧 = 0）⇒ **T12–T15 在 B=100 上的全部数字都混着"后半段断服"**。
但 T16 的修法（`QuotaPacing`）**同时换掉了"选谁"**（赤字 argmax），
所以它回答不了"同一批臂配上同一配速后排序是否不变"。
本轮用 `Paced(arm, budget, window)`：只按 `floor(B·t/L)` 开门/关门，**选谁完全交给原臂**。

系统（与 T12–T16 逐位同一套）
-----------------------------
    用户 0：`uav_track`，PD，warmup 300，窗口 200，代价 = 距离 [m]，absorbed 25 m
    用户 1：`CartPole-v1`，LQR，无 warmup，代价 = 0/1，        absorbed 1.0
    csi：AR(1)（rho=0.9, sd=0.10），均值 (0.90, 0.55)，clip [0.10,1.0]
    q = SINRSuccess 口径 A
★ 常量与生成器**直接 import `scripts/42`**；`D` 直接 import `scripts/47`（不复制）。

扫描：3 臂 {rr, age, na} × 2 模式 {unpaced, paced} × 2 档 {100, 200} × 3 种子
      ⇒ 36 cell，每 cell 30 集 × 200 步。

判据（冻结）
------------
  S1..S7 自检（含三条阴性对照：B=L 恒等、裸臂 t>B=0、闸门关闭必须不派）
  F1 配速协议下是否有臂同时 `geo ≤ 0.90` 且 `max r_i ≤ 1.10`（决定性，二值）
  F2 三配对排序符号是否保持（**不设容差**，幅度原样报）
  F3 逐臂配速效应 Δgeo / Δr_i（只有三臂同号且量级相近才允许写"主要来自时序"）
  R0 非配速分支与 `results/42`（rr/age）、`results/47`（na）**逐 episode J 逐位相同**

用法
----
    D:/devtools/miniconda3/envs/ai-lab/python.exe scripts/52_t21_paced_matching_ablation.py
    ... --skip-experiment      # 只跑自检
    ... --smoke                # 2 集 × 1 种子

产物
----
    outputs/52_t21_paced_matching_ablation.json   完整表（不入库）
    results/runs/<run_id>/52_t21_paced_matching_ablation.json   逐 episode J + 汇总（**入库**）
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

SMOKE_TAG = ""
sys.path.insert(0, str(ROOT))

from wmlab.eval.allocators import MaxAge, NormalizedAge, Paced, RoundRobin  # noqa: E402
from wmlab.eval.results_io import latest_result_path, write_result  # noqa: E402
from wmlab.eval.multiuser import (  # noqa: E402
    Allocation, SINRSuccess, UserSlot, View, default_uav_cost, delivery_rate,
    run_multiuser, window_cost,
)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


T12 = _load(ROOT / "scripts" / "42_t12_hetero_four_arms.py", "t12_mod")
T15 = _load(ROOT / "scripts" / "47_t15_normalized_age.py", "t15_mod")

WINDOW = T12.WINDOW
WARMUP_UAV = T12.WARMUP_UAV
BUDGETS = T12.BUDGETS
GATE = T12.GATE
ABSORBED = [T12.ABSORBED_UAV, T12.ABSORBED_CART]
COSTS = [default_uav_cost, T12.cartpole_cost]
D_MEASURED = tuple(T15.D_MEASURED)      # (53, 10)，唯一来源在 47（闭环实测）
ARMS = ("rr", "age", "na")
MODES = ("unpaced", "paced")

FAILS: list[str] = []


def check(tag: str, ok: bool, detail: str) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {tag}: {detail}")
    if not ok:
        FAILS.append(f"{tag}: {detail}")


# ================================================================ 臂工厂
def make_inner(arm: str):
    if arm == "rr":
        return RoundRobin()
    if arm == "age":
        return MaxAge()
    if arm == "na":
        return NormalizedAge(deadlines=D_MEASURED, name="na")
    raise ValueError(f"未知臂 {arm!r}")


def make_alloc(arm: str, mode: str, budget: int):
    inner = make_inner(arm)
    if mode == "paced":
        return Paced(inner, budget=int(budget), window=WINDOW, name=f"paced-{arm}")
    if mode == "unpaced":
        return inner
    raise ValueError(f"未知模式 {mode!r}")


# ================================================================ 自检装置
class _Spy:
    """记录每次被调用时收到的 View（**不做选择**，一律派给 idx 最小的存活者）。

    用来验证 `Paced` "只改时序、不改 view、不多调一次"：
    若包装器重建/修改了 view（例如把 age 清零、换掉 budget_left），这里立刻看得出。
    """

    def __init__(self) -> None:
        self.seen: list[View] = []

    def __call__(self, view: View, rng: np.random.Generator) -> Allocation:
        self.seen.append(view)
        n = len(view.slots)
        served = np.zeros(n, dtype=bool)
        alive = [s.idx for s in view.slots if s.alive]
        if alive:
            served[alive[0]] = True
        return Allocation(served=served, reason="spy")

    def reset(self) -> None:
        self.seen = []


def _mk_view(ages, alive=None, t=1, budget_left=100, budget_total=100):
    n = len(ages)
    alive = [True] * n if alive is None else list(alive)
    slots = tuple(UserSlot(idx=i, alive=bool(alive[i]), age=int(ages[i]),
                           last_obs=np.zeros(1, dtype=np.float32), n_frames=0,
                           n_attempts=0, csi=1.0, cost_est=0.0) for i in range(n))
    return View(t=int(t), slots=slots, budget_left=int(budget_left),
                budget_total=int(budget_total), privileged=None)


def _run(budget, arm, mode, seed, n_ep, K):
    envs = [T12.make_env("uav", seed=seed, max_steps=WARMUP_UAV + WINDOW),
            T12.make_env("CartPole-v1", seed=seed)]
    ctrls = [T12.PDRelativeController(), T12.CartPoleLQR(K)]
    alloc = make_alloc(arm, mode, budget)
    try:
        r = run_multiuser(envs, ctrls, alloc, SINRSuccess(T12.Q_REF), n_episodes=n_ep, seed=seed,
                          max_steps=WINDOW, warmup_steps=[WARMUP_UAV, 0],
                          budget_total=budget, csi=T12.make_csi(seed, n_ep), cost_fn=COSTS,
                          diagnostic=False, counterfactual=False)
    finally:
        for e in envs:
            e.close()
    return r


# ================================================================ 自检
def selfchecks() -> dict:
    print("\n== 自检 S1–S7（预注册 §5：每条都写明能打死什么错）==")
    rng = np.random.default_rng(0)
    out: dict = {}
    K = T12.load_lqr_gain()

    # ---------------- S1：B=L 时包装器必须恒等（打"偷改选择/时点"）----------------
    s1 = {}
    for arm in ("rr", "na"):
        a = _run(200, arm, "unpaced", 0, 2, K)["records"]
        b = _run(200, arm, "paced", 0, 2, K)["records"]
        seq_a = [r.served_idx for r in a]
        seq_b = [r.served_idx for r in b]
        Ja = np.asarray(window_cost(a, 2, window=WINDOW, absorbed=ABSORBED, n_episodes=2))
        Jb = np.asarray(window_cost(b, 2, window=WINDOW, absorbed=ABSORBED, n_episodes=2))
        s1[arm] = {"n_steps": len(seq_a), "same_seq": seq_a == seq_b,
                   "max_abs_dJ": float(np.max(np.abs(Ja - Jb)))}
    ok1 = all(v["same_seq"] and v["max_abs_dJ"] == 0.0 for v in s1.values())
    check("S1 B=L(200) 时 Paced 与裸臂 served_idx 逐位相同、逐集 J 逐位相同", ok1, str(s1))
    out["S1"] = s1

    # ---------------- S2：配速版每集发帧恰为 B（打"闸门分母/跨集泄漏"）----------------
    s2 = {}
    for arm in ARMS:
        rec = _run(100, arm, "paced", 0, 2, K)["records"]
        per_ep = {e: sum(1 for x in rec if x.episode == e and x.attempted_idx >= 0)
                  for e in sorted({x.episode for x in rec})}
        s2[arm] = per_ep
    ok2 = all(v == 100 for d in s2.values() for v in d.values()) and all(len(d) == 2 for d in s2.values())
    check("S2 配速版 B=100：每集发帧恰为 100（三臂都成立）", ok2, str(s2))
    out["S2"] = s2

    # ---------------- S3：配速把帧铺到 t>B；裸臂必须仍是 0（阴性对照）----------------
    s3 = {}
    for mode in MODES:
        rec = _run(100, "rr", mode, 0, 2, K)["records"]
        s3[mode] = sum(1 for x in rec if x.t > 100 and x.attempted_idx >= 0)
    check("S3 B=100：paced 在 t>100 有发帧，unpaced 恒为 0（配对阴性对照）",
          s3["paced"] > 0 and s3["unpaced"] == 0, f"{s3}")
    out["S3"] = s3

    # ---------------- S4：只改时序 —— 内层臂收到的 view 必须原样且只在放行步 ----------------
    spy = _Spy()
    p = Paced(spy, budget=100, window=WINDOW, name="spy-paced")
    expected_open = []
    mism = []
    for t in range(1, 21):
        v = _mk_view((t, 2 * t), t=t, budget_left=100, budget_total=100)
        before = len(spy.seen)
        allowance = (100 * t) // WINDOW
        want = (allowance - before) >= 1          # ★ 放行判据必须在调用**之前**算
        p(v, rng)
        called = len(spy.seen) > before
        if called:
            expected_open.append(t)
            got = spy.seen[-1]
            same = (got.t == v.t and got.slots[0].age == v.slots[0].age
                    and got.slots[1].age == v.slots[1].age
                    and got.budget_left == v.budget_left
                    and got.budget_total == v.budget_total
                    and got.privileged is None)
            if not same:
                mism.append(f"view 被改动@t={t}")
        if called != want:
            mism.append(f"闸门不符@t={t}: called={called} want={want} allowance={allowance}")
    # 手算：B=100/L=200 ⇒ 放行步 = t=2,4,6,…,20（每 2 步一次）
    ok4 = expected_open == list(range(2, 21, 2)) and not mism
    check("S4 放行步只有偶数步、且内层臂收到的 view 与构造的**逐字段相同**", ok4,
          f"放行步={expected_open}；不符={mism}")
    out["S4"] = {"open_steps": expected_open, "mismatch": [str(x) for x in mism]}

    # ---------------- S5：允许量不变量 used ≤ floor(B·t/L) ----------------
    rec = _run(100, "age", "paced", 0, 2, K)["records"]
    viol, used, cur_ep = [], 0, None
    for x in rec:
        if x.episode != cur_ep:
            cur_ep, used = x.episode, 0
        if x.attempted_idx >= 0:
            used += 1
        cap = (100 * x.t) // WINDOW
        if used > cap:
            viol.append((x.episode, x.t, used, cap))
    check("S5 任一步累计发帧 ≤ floor(B·t/L)（允许量差一就会静默超预算）",
          not viol, f"违例 {viol[:3]}" if viol else "全 400 步满足")
    out["S5"] = {"violations": viol[:5]}

    # ---------------- S6：跨集计数复位 ----------------
    counts = {}
    for e in sorted({x.episode for x in rec}):
        counts[e] = sum(1 for x in rec if x.episode == e and x.attempted_idx >= 0)
    check("S6 两集各自恰 100 帧（第二集没有继承第一集计数）",
          len(counts) == 2 and all(v == 100 for v in counts.values()), str(counts))
    out["S6"] = counts

    # ---------------- S7：口径外推必须抛错 ----------------
    bad = []
    for kw in ({"budget": -1, "window": 200}, {"budget": 201, "window": 200},
               {"budget": 100, "window": 0}, {"budget": 100.5, "window": 200},
               {"budget": True, "window": 200}):
        try:
            Paced(MaxAge(), name="bad", **kw)
            bad.append(f"{kw} 未抛")
        except ValueError:
            pass
    check("S7 budget<0 / budget>window / window≤0 / 非整数 全部抛 ValueError",
          not bad, "全部抛错" if not bad else f"漏网 {bad}")
    out["S7"] = {"leaks": bad}
    return out


# ================================================================ 单 cell
def run_cell(budget, arm, mode, seed, n_ep, K):
    t0 = time.time()
    rec = _run(budget, arm, mode, seed, n_ep, K)["records"]
    dr = delivery_rate(rec, 2)
    J = np.asarray(window_cost(rec, 2, window=WINDOW, absorbed=ABSORBED, n_episodes=n_ep))
    served = [sum(1 for x in rec if x.attempted_idx == i) for i in range(2)]
    late = [sum(1 for x in rec if x.t > budget and x.attempted_idx == i) for i in range(2)]
    per_ep = {int(e): sum(1 for x in rec if x.episode == e and x.attempted_idx >= 0)
              for e in sorted({x.episode for x in rec})}
    return {
        "budget": int(budget), "arm": arm, "mode": mode, "seed": int(seed),
        "J_per_episode": np.asarray(J, dtype=float).tolist(),
        "J_mean": np.asarray(J, dtype=float).mean(axis=0).tolist(),
        "served": served, "served_late": late,
        "served_total": int(sum(served)),
        "served_per_episode": {str(k): int(v) for k, v in per_ep.items()},
        "alive_steps": dr["alive_steps"], "frames": dr["frames"],
        "attempts": dr["attempts"], "link_q": dr["link_q"], "p_eff": dr["p_eff"],
        "budget_reserved": dr["budget_reserved"],
        "budget_attempted": dr["budget_attempted"],
        "budget_wasted": dr["budget_wasted"],
        "gap_uav": T15.interval_stats(rec, 0),
        "gap_cart": T15.interval_stats(rec, 1),
        "elapsed_s": round(time.time() - t0, 2),
    }


# ================================================================ 主实验
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip-experiment", action="store_true")
    ap.add_argument("--run-id", default=None, help="入库目录名；默认 <时间戳>-<commit>")
    ap.add_argument("--allow-overwrite", action="store_true",
                    help="仅用于**确有必要**重写同一 run-id 的旧证据（会先备份 .bak）")
    args = ap.parse_args()
    global SMOKE_TAG
    SMOKE_TAG = "_smoke" if args.smoke else ""
    n_ep = 2 if args.smoke else args.episodes
    seeds = [0] if args.smoke else args.seeds
    budgets = list(BUDGETS) if not args.smoke else [100, 200]

    t_all = time.time()
    print(f"[52] T21 配速化匹配消融｜N=2 异质｜窗口 {WINDOW}｜预算 {budgets}｜seeds={seeds}"
          f"｜episodes={n_ep}｜arms={ARMS}｜modes={MODES}｜D={D_MEASURED}")
    sc = selfchecks()

    if args.skip_experiment:
        _dump({"selfchecks": sc}, None, None, n_ep, seeds)
        print(f"\n{'ALL CHECKS PASS' if not FAILS else 'FAILED: ' + '; '.join(FAILS)}")
        return 0 if not FAILS else 2

    K = T12.load_lqr_gain()
    cells: list[dict] = []
    for budget in budgets:
        for mode in MODES:
            for arm in ARMS:
                for seed in seeds:
                    c = run_cell(budget, arm, mode, seed, n_ep, K)
                    cells.append(c)
                    print(f"  B={budget:3d} {mode:7s} {arm:3s} seed={seed} "
                          f"J=({c['J_mean'][0]:.3f},{c['J_mean'][1]:.3f}) "
                          f"served=({c['served'][0]},{c['served'][1]}) "
                          f"late=({c['served_late'][0]},{c['served_late'][1]}) "
                          f"alive=({c['alive_steps'][0]},{c['alive_steps'][1]}) "
                          f"{c['elapsed_s']}s")

    # ---------- 汇总 ----------
    def get(budget, mode, arm, seed):
        for c in cells:
            if (c["budget"], c["mode"], c["arm"], c["seed"]) == (budget, mode, arm, seed):
                return c
        return None

    summary: dict = {}
    for budget in budgets:
        summary[str(budget)] = {}
        for mode in MODES:
            ref = {s: np.asarray(get(budget, mode, "rr", s)["J_mean"], dtype=float) for s in seeds}
            per_arm = {}
            for arm in ARMS:
                rs = []
                for s in seeds:
                    j = np.asarray(get(budget, mode, arm, s)["J_mean"], dtype=float)
                    deg = bool(np.any(ref[s] <= 1e-12))
                    r = (j / ref[s]) if not deg else np.array([np.nan, np.nan])
                    rs.append({"seed": s, "r": r.tolist(),
                               "geo": float(np.sqrt(r[0] * r[1])) if not deg else float("nan")})
                geos = [x["geo"] for x in rs if np.isfinite(x["geo"])]
                rows = [get(budget, mode, arm, s) for s in seeds]
                per_arm[arm] = {
                    "per_seed": rs,
                    "r_mean": np.nanmean([[v for v in x["r"]] for x in rs], axis=0).tolist(),
                    "r_sd": np.nanstd([[v for v in x["r"]] for x in rs], axis=0, ddof=1).tolist(),
                    "geo_mean": float(np.mean(geos)) if geos else float("nan"),
                    "geo_sd": float(np.std(geos, ddof=1)) if len(geos) > 1 else 0.0,
                    "J_uav_mean": float(np.mean([x["J_mean"][0] for x in rows])),
                    "J_cart_mean": float(np.mean([x["J_mean"][1] for x in rows])),
                    "served_uav": float(np.mean([x["served"][0] for x in rows])),
                    "served_cart": float(np.mean([x["served"][1] for x in rows])),
                    "served_total": float(np.mean([x["served_total"] for x in rows])),
                    "late_total": float(np.mean([sum(x["served_late"]) for x in rows])),
                    "alive_uav": float(np.mean([x["alive_steps"][0] for x in rows])),
                    "alive_cart": float(np.mean([x["alive_steps"][1] for x in rows])),
                    "cart_gap_median": float(np.mean([x["gap_cart"].get("median", np.nan)
                                                      for x in rows])),
                    "cart_gap_frac_ge10": float(np.mean([x["gap_cart"].get("frac_ge_10",
                                                                           np.nan)
                                                         for x in rows])),
                }
            summary[str(budget)][mode] = per_arm

    print("\n== 汇总（r_i 相对**同档同模式**的 rr，同种子配对）==")
    for budget in budgets:
        for mode in MODES:
            for arm in ARMS:
                d = summary[str(budget)][mode][arm]
                print(f"  B={budget:3d} {mode:7s} {arm:3s} "
                      f"geo={d['geo_mean']:.3f}±{d['geo_sd']:.3f} "
                      f"r=({d['r_mean'][0]:.3f},{d['r_mean'][1]:.3f}) "
                      f"J=({d['J_uav_mean']:.3f},{d['J_cart_mean']:.3f}) "
                      f"served=({d['served_uav']:.0f},{d['served_cart']:.0f}) "
                      f"late={d['late_total']:.0f} alive_cart={d['alive_cart']:.1f} "
                      f"cart_gap_med={d['cart_gap_median']:.1f} "
                      f"ge10={d['cart_gap_frac_ge10']:.3f}")

    # ---------- 记账判据 ----------
    print("\n== 记账判据（P1–P4）==")
    bad_p1 = [c for c in cells if c["served_total"] != n_ep * c["budget"]]
    check("P1 每 cell 总发帧 == episodes × budget", not bad_p1,
          f"{len(cells)} cells" if not bad_p1 else
          f"{[(c['budget'], c['mode'], c['arm'], c['seed'], c['served_total']) for c in bad_p1[:4]]}")
    bad_p2 = [c for c in cells
              if c["budget_reserved"] != c["budget_attempted"] + c["budget_wasted"]]
    check("P2 reserved == attempted + wasted（三分账闭合）", not bad_p2,
          "全 cell" if not bad_p2 else str(bad_p2[:3]))
    bad_late = [c for c in cells if c["mode"] == "unpaced" and c["budget"] < WINDOW
                and sum(c["served_late"]) != 0]
    check("P3 裸臂在 B<L 档 t>B 发帧恒为 0（缺陷仍复现，阴性对照没有被悄悄修掉）",
          not bad_late, "全部 0" if not bad_late else str(bad_late[:3]))
    bad_pace = [c for c in cells if c["mode"] == "paced" and c["budget"] < WINDOW
                and sum(c["served_late"]) <= 0]
    check("P4 配速臂在 B<L 档 t>B 发帧 > 0（R17：配速确实生效）", not bad_pace,
          "全部 >0" if not bad_pace else str(bad_pace[:3]))

    # ---------- R0：非配速分支逐位回归 ----------
    print("\n== R0 与历史结果逐位回归（非配速分支）==")
    r0 = {"42": None, "47": None}
    for name, logi, arms in (("42", "42_t12_four_arms_summary", ("rr", "age")),
                             ("47", "47_t15_normalized_age", ("rr", "age", "na"))):
        path = latest_result_path(logi)
        print(f"  [ref-{name}] {path.relative_to(ROOT).as_posix() if path else '<缺失>'}")
        if path is None or not path.exists():
            r0[name] = {"compared": 0, "diffs": ["<无参考>"]}
            continue
        ref = json.loads(path.read_text(encoding="utf-8"))
        ref_map = {(c["budget"], c["arm"], c["seed"]): c["J_per_episode"]
                   for c in ref.get("cells_J", [])}
        diffs, compared = [], 0
        for c in cells:
            if c["mode"] != "unpaced" or c["arm"] not in arms:
                continue
            key = (c["budget"], c["arm"], c["seed"])
            if key not in ref_map:
                continue
            a = np.asarray(c["J_per_episode"], dtype=float)
            b = np.asarray(ref_map[key], dtype=float)
            if a.shape != b.shape:
                continue
            compared += 1
            if not np.array_equal(a, b):
                diffs.append(f"{key} max|Δ|={np.nanmax(np.abs(a - b)):.3e}")
        r0[name] = {"compared": compared, "diffs": diffs}
        check(f"R0[{name}] 非配速 {'/'.join(arms)} 逐 episode J 与历史逐位相同",
              compared > 0 and not diffs,
              (f"{compared} cells 逐位相同" if not diffs else "；".join(diffs[:4])) +
              ("" if compared else "（0 可比 cell ⇒ 不算通过）"))

    # ---------- F1：配速协议下有没有臂同时过 geo 与非牺牲 ----------
    print("\n== F1（决定性：配速协议下 geo ≤ GATE 且 max r_i ≤ 1.10）==")
    f1 = {"gate": GATE, "hits": [], "near": [], "by_budget": {}}
    for budget in budgets:
        f1["by_budget"][str(budget)] = {}
        for arm in ("age", "na"):
            d = summary[str(budget)]["paced"][arm]
            max_r = float(max(d["r_mean"]))
            hit = bool(d["geo_mean"] <= GATE and max_r <= 1.10)
            f1["by_budget"][str(budget)][arm] = {"geo": d["geo_mean"], "r": d["r_mean"],
                                                 "max_r": max_r, "hit": hit}
            if hit:
                f1["hits"].append(f"B={budget}/{arm}")
            elif d["geo_mean"] <= GATE:
                f1["near"].append(f"B={budget}/{arm}(geo {d['geo_mean']:.3f} 过闸但 max r "
                                  f"{max_r:.3f}>1.10)")
            print(f"  B={budget:3d} paced {arm:3s} geo={d['geo_mean']:.3f} "
                  f"r=({d['r_mean'][0]:.3f},{d['r_mean'][1]:.3f}) ⇒ "
                  f"{'命中' if hit else '未命中'}")
    f1["verdict"] = "HIT" if f1["hits"] else "NO-HIT"
    if f1["hits"]:
        b200_only = all(h.startswith("B=200") for h in f1["hits"])
        f1["note"] = ("配速协议下存在同时过闸且不牺牲用户的臂：" + "、".join(f1["hits"]) +
                      ("；★ 仅出现在 B=200，而 B=200 与 L 重合、历史上本就不存在该缺陷，"
                       "不得记为本轮配速的功劳" if b200_only else ""))
    else:
        f1["note"] = ("本配速协议 + 本实现 + 3 种子下，无任何臂同时满足 geo ≤ 0.90 与 max r_i ≤ 1.10"
                      + ("（另有 geo 过闸但牺牲用户的情形：" + "、".join(f1["near"]) + "）"
                         if f1["near"] else ""))
    print(f"  ⇒ {f1['verdict']}：{f1['note']}")

    # ---------- F2：排序符号是否保持（不设容差）----------
    print("\n== F2（次级：三配对 geo 差值的符号，paced vs unpaced）==")
    pairs = [("rr", "age"), ("rr", "na"), ("age", "na")]
    f2 = {"pairs": {}, "reversed": [], "verdict": None}
    for budget in budgets:
        f2["pairs"][str(budget)] = {}
        for a, b in pairs:
            dp = summary[str(budget)]["paced"][a]["geo_mean"] - \
                summary[str(budget)]["paced"][b]["geo_mean"]
            du = summary[str(budget)]["unpaced"][a]["geo_mean"] - \
                summary[str(budget)]["unpaced"][b]["geo_mean"]
            f2["pairs"][str(budget)][f"{a}-{b}"] = {"d_paced": float(dp), "d_unpaced": float(du),
                                                    "sign_same": bool(np.sign(dp) == np.sign(du))}
            if np.sign(dp) != np.sign(du):
                f2["reversed"].append(f"B={budget} {a}-{b}: unpaced {du:+.3f} → paced {dp:+.3f}")
            print(f"  B={budget:3d} {a}-{b}: unpaced {du:+.3f} → paced {dp:+.3f} "
                  f"{'（同号）' if np.sign(dp) == np.sign(du) else '（★反向）'}")
    f2["verdict"] = "ORDER-PRESERVED" if not f2["reversed"] else "ORDER-FLIPPED"
    print(f"  ⇒ {f2['verdict']}（不设容差，幅度原样报）")

    # ---------- F3：配速效应（逐臂、逐用户）----------
    print("\n== F3（归因：配速效应 Δgeo / Δr_i，同臂同档同种子）==")
    f3 = {}
    for budget in budgets:
        f3[str(budget)] = {}
        for arm in ARMS:
            pu = summary[str(budget)]["unpaced"][arm]
            pa = summary[str(budget)]["paced"][arm]
            d_geo = float(pa["geo_mean"] - pu["geo_mean"])
            d_r = [float(pa["r_mean"][i] - pu["r_mean"][i]) for i in range(2)]
            f3[str(budget)][arm] = {"d_geo": d_geo, "d_r": d_r,
                                    "geo_unpaced": pu["geo_mean"], "geo_paced": pa["geo_mean"]}
            print(f"  B={budget:3d} {arm:3s} geo {pu['geo_mean']:.3f} → {pa['geo_mean']:.3f} "
                  f"（Δ={d_geo:+.3f}）；Δr=({d_r[0]:+.3f},{d_r[1]:+.3f})")
    same_sign = all(np.sign(v["d_geo"]) == np.sign(f3[str(budgets[0])][ARMS[0]]["d_geo"])
                    for b in budgets for v in f3[str(b)].values())
    f3["same_sign_all"] = bool(same_sign)

    _dump({"selfchecks": sc}, cells,
          {"summary": summary, "F1": f1, "F2": f2, "F3": f3, "R0": r0},
          n_ep, seeds, run_id=args.run_id, allow_overwrite=args.allow_overwrite,
          persist=not args.smoke)
    print(f"\n{'ALL CHECKS PASS' if not FAILS else 'FAILED: ' + '; '.join(FAILS)}")
    print(f"总耗时 {round(time.time() - t_all, 1)}s")
    return 0 if not FAILS else 2


def _dump(payload, cells, extra, n_ep, seeds, *, run_id=None, allow_overwrite=False,
          persist=True):
    out = {"config": {
        "N": 2, "window": WINDOW, "warmup": [WARMUP_UAV, 0],
        "budgets": list(BUDGETS), "episodes": n_ep, "seeds": seeds, "gate": GATE,
        "absorbed": [float(x) for x in ABSORBED], "D_measured": list(D_MEASURED),
        "arms": list(ARMS), "modes": list(MODES),
        "pacing": "allowance(t)=floor(B*t/L); 放行 ⇔ allowance-used ≥ 1；选谁由原臂决定",
        "aggregation": "geo of per-user ratios vs rr (same budget, same mode, same seed)",
        "prereg": "04-实验与代码/2026-10-19-T21-配速化匹配消融-预注册-v0.1.md",
        "paired_baseline": "同档同模式 rr；另在 summary 里保留 unpaced 供历史口径对照"},
        "selfchecks": payload.get("selfchecks"),
        "checks_failed": list(FAILS)}
    (ROOT / "outputs").mkdir(exist_ok=True)
    (ROOT / "outputs" / ("52_t21_paced_matching_ablation" + SMOKE_TAG + ".json")).write_text(
        json.dumps({**out, **(extra or {}), "cells": (cells or [])},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    if cells is None or not persist:
        if cells is not None:
            print("[smoke] 不写 results/（smoke 数字不进证据链，也不抢 INDEX 的 current）")
        return
    slim = [{k: c[k] for k in ("budget", "arm", "mode", "seed", "J_per_episode", "J_mean",
                               "served", "served_late", "served_total", "served_per_episode",
                               "alive_steps", "frames", "attempts", "link_q",
                               "budget_reserved", "budget_attempted", "budget_wasted",
                               "gap_cart", "gap_uav", "elapsed_s")}
            for c in (cells or [])]
    dest = write_result("52_t21_paced_matching_ablation", {**out, **(extra or {}), "cells_J": slim},
                        run_id=run_id, allow_overwrite=allow_overwrite)
    print(f"[evidence] {dest.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    sys.exit(main())
