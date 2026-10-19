#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""T16 · 预算「配额 + 配速」—— 判 H5（**机制验证，不是算法主张**）。

口径来源（★ 全部在看结果之前冻结）
------------------------------------
`04-实验与代码/2026-10-19-T16-配额配速-预注册-v0.1.md` §1–§6（含 §3.1 修订 v0.1a）。

为什么有这一轮（不是"又换一个臂"）
----------------------------------
读代码 + 实测（预注册 §1）发现：`_exhausted` + 所有臂"只要 alive 就派帧"
⇒ 第 t 步累计已用 = t ⇒ **t = B 时预算归零，窗内剩下 L−B 步完全断服**。
B=100 / L=200 实测：`rr`、`age` 的最后发帧时刻精确为 t=100，`t>100` 发帧数 **0**。
⇒ 现有四臂在 B=100 上比的是"前半段怎么花完 100 帧"，不是"100 帧怎么铺满 200 步"。

系统（与 T12/T13/T14/T15 逐位同一套）
---------------------------------
    用户 0：`uav_track`，PD，warmup 300，窗口 200，代价 = 距离 [m]，absorbed 25 m
    用户 1：`CartPole-v1`，LQR，无 warmup，代价 = 0/1，             absorbed 1.0
    csi：AR(1)（rho=0.9, sd=0.10），均值 (0.90, 0.55)，clip [0.10,1.0]
    q   = SINRSuccess 口径 A
★ 常量与生成器**直接 import `scripts/42`**（不复制）。

扫描：`quotas = (B − n_1, n_1)`，Σ = B（配额硬上界 ⇒ 不可能 I1 违约）
    B=100：n_1 ∈ {0,20,40,50,60,80,100}
    B=200：n_1 ∈ {0,20,...,200}（步长 20）
两种收回口径（预注册 §3.1）：
    `reclaim=True`  **主口径**（死者未用配额按比例收回 ⇒ 总支出仍被 B 卡死，只改"给谁"）
    `reclaim=False` 消融（严格 I6 不转嫁 ⇒ 会少花帧，与"配速"混淆）

判据（冻结）
------------
  S1..S6 自检（预注册 §5；S1 是手算对照，不是经验预期）
  F1  geo*(B) = min_{n_1} geo（主口径 reclaim）；
      geo* ≤ 0.90 且 max_i r_i ≤ 1.10 ⇒ FEASIBLE-ISH
      geo* > 0.90                    ⇒ INFEASIBLE-IN-QUOTA-FAMILY（**不是严格不可达**）
      geo ≤ 0.90 但 max_i r_i > 1.10 ⇒ DEGENERATE（牺牲一个用户，不算通过）

用法
----
    D:/devtools/miniconda3/envs/ai-lab/python.exe scripts/48_t16_quota_pacing.py
    ... --skip-experiment      # 只跑自检
    ... --smoke                # 2 集 × 1 种子

产物
----
    outputs/48_t16_quota_pacing.json   完整表（不入库）
    results/48_t16_quota_pacing.json   逐 episode J + 汇总 + 判据（**入库**）
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

from wmlab.eval.allocators import MaxAge, QuotaPacing, RoundRobin  # noqa: E402
from wmlab.eval.results_io import latest_result_path, write_result  # noqa: E402  ← 入库唯一入口（P0-3）
from wmlab.eval.multiuser import (  # noqa: E402
    SINRSuccess, UserSlot, View, default_uav_cost, delivery_rate, run_multiuser, window_cost,
)


def _load_t12():
    spec = importlib.util.spec_from_file_location(
        "t12_mod", ROOT / "scripts" / "42_t12_hetero_four_arms.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["t12_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


T12 = _load_t12()
WINDOW = T12.WINDOW
WARMUP_UAV = T12.WARMUP_UAV
BUDGETS = T12.BUDGETS
GATE = T12.GATE
ABSORBED = [T12.ABSORBED_UAV, T12.ABSORBED_CART]
COSTS = [default_uav_cost, T12.cartpole_cost]

SPLITS = {100: [0, 20, 40, 50, 60, 80, 100],
          200: [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]}
# ★ 主口径 reclaim=True（预注册 §3.1：只有它做到"同帧数、只改时序/分配"）
RECLAIM_MODES = (True, False)


def _rname(rc: bool) -> str:
    return "reclaim" if rc else "strict"


FAILS: list[str] = []


def check(tag: str, ok: bool, detail: str) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {tag}: {detail}")
    if not ok:
        FAILS.append(f"{tag}: {detail}")


# ================================================================ 自检
def _mk_view(ages, alive=None, t=1, budget_left=200, budget_total=200):
    n = len(ages)
    alive = [True] * n if alive is None else list(alive)
    slots = tuple(UserSlot(idx=i, alive=bool(alive[i]), age=int(ages[i]),
                           last_obs=np.zeros(1, dtype=np.float32), n_frames=0,
                           n_attempts=0, csi=1.0, cost_est=0.0) for i in range(n))
    return View(t=int(t), slots=slots, budget_left=int(budget_left),
                budget_total=int(budget_total), privileged=None)


def selfchecks() -> dict:
    print("\n== 自检 S1–S7（预注册 §5 + §3.1）==")
    rng = np.random.default_rng(0)
    out: dict = {}

    # ---- S1：手算序列（预注册 §5 末表，逐位对照）----
    q = QuotaPacing(quotas=(50, 50), window=200)
    seq = []
    for t in range(1, 9):
        a = q(_mk_view((0, 0), t=t, budget_left=200), rng)
        seq.append(int(np.flatnonzero(a.served)[0]) if a.served.any() else -1)
    expect = [-1, 0, 1, -1, -1, 0, 1, -1]
    check("S1 (50,50)/L=200 前 8 步 served 与手算逐位相同", seq == expect,
          f"实得 {seq} vs 期望 {expect}")
    out["S1"] = {"seq": seq, "expect": expect}

    # ---- S2：极端配额 ----
    a = QuotaPacing(quotas=(0, 100), window=200)
    picks = []
    for t in range(1, 7):
        al = a(_mk_view((5, 0), t=t, budget_left=200), rng)
        picks.append(int(np.flatnonzero(al.served)[0]) if al.served.any() else -1)
    b = QuotaPacing(quotas=(100, 0), window=200)
    picks_b = []
    for t in range(1, 7):
        al = b(_mk_view((0, 5), t=t, budget_left=200), rng)
        picks_b.append(int(np.flatnonzero(al.served)[0]) if al.served.any() else -1)
    ok2 = all(p in (1, -1) for p in picks) and 1 in picks \
        and all(p in (0, -1) for p in picks_b) and 0 in picks_b
    check("S2 (0,B) 只选 1；(B,0) 只选 0（配额索引没接反）", ok2,
          f"(0,100)⇒{picks}；(100,0)⇒{picks_b}")
    out["S2"] = {"q0_0": picks, "q_b_0": picks_b}

    # ---- S3：非法参数 ----
    bads = []
    for kw in ({"quotas": (50,), "window": 200}, {"quotas": (-1, 10), "window": 200},
               {"quotas": (1.5, 10), "window": 200}, {"quotas": (), "window": 200},
               {"quotas": (10, 10), "window": 0}, {"quotas": (True, 10), "window": 200}):
        try:
            a = QuotaPacing(**kw)
            a(_mk_view((1, 1), t=1, budget_left=200), rng)   # 长度不符只能在 __call__ 抛
            bads.append(f"{kw} 未抛")
        except ValueError:
            pass
    check("S3 非法 quotas/window 全部抛 ValueError", not bads,
          "全部抛错" if not bads else f"漏网 {bads}")
    out["S3"] = {"leaks": bads}

    # ---- S4/S5/S7 用真实运行 ----
    K = T12.load_lqr_gain()

    def run(alloc, budget, seed, n_ep):
        envs = [T12.make_env("uav", seed=seed, max_steps=WARMUP_UAV + WINDOW),
                T12.make_env("CartPole-v1", seed=seed)]
        r = run_multiuser(envs, [T12.PDRelativeController(), T12.CartPoleLQR(K)],
                          alloc, SINRSuccess(T12.Q_REF), n_episodes=n_ep, seed=seed,
                          max_steps=WINDOW, warmup_steps=[WARMUP_UAV, 0],
                          budget_total=budget, csi=T12.make_csi(seed, n_ep), cost_fn=COSTS)
        for e in envs:
            e.close()
        return r

    r_pace = run(QuotaPacing(quotas=(50, 50), window=WINDOW), 100, 0, 2)
    r_rr = run(RoundRobin(), 100, 0, 2)
    r_rc = run(QuotaPacing(quotas=(50, 50), window=WINDOW, reclaim=True), 100, 0, 2)
    late_pace = sum(1 for x in r_pace["records"] if x.t > 100 and x.attempted_idx >= 0)
    late_rr = sum(1 for x in r_rr["records"] if x.t > 100 and x.attempted_idx >= 0)
    check("S4 配速生效：B=100 时 t>100 仍有发帧（对照 rr 实测 0）",
          late_pace > 0 and late_rr == 0,
          f"quota(50,50) 后半段发帧 {late_pace}；rr {late_rr}（2 集）")
    out["S4"] = {"late_pace": late_pace, "late_rr": late_rr}

    # S5 配额上界 + 三分账（严格口径）
    used = {}
    for x in r_pace["records"]:
        if x.attempted_idx >= 0:
            used[x.episode] = used.get(x.episode, 0) + 1
    dr = delivery_rate(r_pace["records"], 2)
    ok5 = dr["budget_reserved"] == dr["budget_attempted"] + dr["budget_wasted"] \
        and all(v <= 100 for v in used.values()) and dr["budget_reserved"] <= 2 * 100
    check("S5 严格口径配额上界与三分账闭合", ok5,
          f"逐集发帧 {sorted(used.items())} ≤100；reserved={dr['budget_reserved']} "
          f"attempted={dr['budget_attempted']} wasted={dr['budget_wasted']}")
    out["S5"] = {"per_ep_served": used, "reserved": dr["budget_reserved"]}

    # S7 reclaim：a) 全存活时与严格口径逐位相同；b) 真实运行下总帧数上界仍是 B
    qs = QuotaPacing(quotas=(50, 50), window=200, reclaim=True)
    seq_rc = []
    for t in range(1, 9):
        al = qs(_mk_view((0, 0), t=t, budget_left=200), rng)
        seq_rc.append(int(np.flatnonzero(al.served)[0]) if al.served.any() else -1)
    n_rc = sum(1 for x in r_rc["records"] if x.attempted_idx >= 0)
    n_st = sum(1 for x in r_pace["records"] if x.attempted_idx >= 0)
    check("S7 reclaim：全存活时与严格口径逐位相同 + 总帧数 ≤ 2B 且多于严格口径",
          seq_rc == expect and n_rc <= 2 * 100 and n_rc > n_st,
          f"序列同={seq_rc == expect}；reclaim 发 {n_rc} 帧 vs 严格 {n_st} 帧（2 集）")
    out["S7"] = {"seq": seq_rc, "served_reclaim": n_rc, "served_strict": n_st}

    # ---- S6：与 results/42 的 rr/age 逐位回归在 main 里做（需要全量 30 集）----
    return out


# ================================================================ 单 cell
def run_cell(budget, quotas, seed, n_ep, K, reclaim):
    t0 = time.time()
    envs = [T12.make_env("uav", seed=seed, max_steps=WARMUP_UAV + WINDOW),
            T12.make_env("CartPole-v1", seed=seed)]
    alloc = QuotaPacing(quotas=quotas, window=WINDOW, reclaim=bool(reclaim))
    r = run_multiuser(envs, [T12.PDRelativeController(), T12.CartPoleLQR(K)],
                      alloc, SINRSuccess(T12.Q_REF), n_episodes=n_ep, seed=seed,
                      max_steps=WINDOW, warmup_steps=[WARMUP_UAV, 0],
                      budget_total=budget, csi=T12.make_csi(seed, n_ep), cost_fn=COSTS)
    for e in envs:
        e.close()
    rec = r["records"]
    dr = delivery_rate(rec, 2)
    J = np.asarray(window_cost(rec, 2, window=WINDOW, absorbed=ABSORBED, n_episodes=n_ep))
    served = [sum(1 for x in rec if x.attempted_idx == i) for i in range(2)]
    late = [sum(1 for x in rec if x.t > budget and x.attempted_idx == i) for i in range(2)]
    viol = []
    for e in range(n_ep):
        for i in range(2):
            n_i = sum(1 for x in rec if x.episode == e and x.attempted_idx == i)
            cap = quotas[i] if not reclaim else budget     # reclaim 下上界是 B
            if n_i > cap:
                viol.append((e, i, n_i))
    return {"budget": budget, "seed": seed, "quotas": list(quotas), "n_cart": quotas[1],
            "reclaim": bool(reclaim),
            "J_per_episode": J.tolist(), "J_mean": J.mean(axis=0).tolist(),
            "served": served, "served_late": late,
            "alive_steps": dr["alive_steps"], "frames": dr["frames"],
            "attempts": dr["attempts"], "link_q": dr["link_q"],
            "budget_reserved": dr["budget_reserved"],
            "budget_attempted": dr["budget_attempted"],
            "budget_wasted": dr["budget_wasted"],
            "quota_violations": viol,
            "elapsed_s": round(time.time() - t0, 2)}


def run_baseline(budget, arm, seed, n_ep, K):
    t0 = time.time()
    envs = [T12.make_env("uav", seed=seed, max_steps=WARMUP_UAV + WINDOW),
            T12.make_env("CartPole-v1", seed=seed)]
    alloc = RoundRobin() if arm == "rr" else MaxAge()
    r = run_multiuser(envs, [T12.PDRelativeController(), T12.CartPoleLQR(K)],
                      alloc, SINRSuccess(T12.Q_REF), n_episodes=n_ep, seed=seed,
                      max_steps=WINDOW, warmup_steps=[WARMUP_UAV, 0],
                      budget_total=budget, csi=T12.make_csi(seed, n_ep), cost_fn=COSTS)
    for e in envs:
        e.close()
    rec = r["records"]
    dr = delivery_rate(rec, 2)
    J = np.asarray(window_cost(rec, 2, window=WINDOW, absorbed=ABSORBED, n_episodes=n_ep))
    return {"budget": budget, "arm": arm, "seed": seed,
            "J_per_episode": J.tolist(), "J_mean": J.mean(axis=0).tolist(),
            "served": [sum(1 for x in rec if x.attempted_idx == i) for i in range(2)],
            "served_late": [sum(1 for x in rec if x.t > budget and x.attempted_idx == i)
                            for i in range(2)],
            "alive_steps": dr["alive_steps"],
            "elapsed_s": round(time.time() - t0, 2)}


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
    n_ep = 2 if args.smoke else args.episodes
    seeds = [0] if args.smoke else args.seeds
    splits = {b: ([0, b] if args.smoke else SPLITS[b]) for b in BUDGETS}

    t_all = time.time()
    print(f"[48] T16 配额配速｜N=2 异质｜窗口 {WINDOW}｜预算 {BUDGETS}｜seeds={seeds}"
          f"｜episodes={n_ep}｜regime={{'reclaim','strict'}}｜splits={splits}")
    sc = selfchecks()

    if args.skip_experiment:
        _dump({"selfchecks": sc, "experiment": None}, None, None, n_ep, seeds)
        print(f"\n{'ALL CHECKS PASS' if not FAILS else 'FAILED: ' + '; '.join(FAILS)}")
        return 0 if not FAILS else 2

    K = T12.load_lqr_gain()

    # ---------- 基线（给配对比值 + R0 回归）----------
    base: list[dict] = []
    for budget in BUDGETS:
        for arm in ("rr", "age"):
            for seed in seeds:
                c = run_baseline(budget, arm, seed, n_ep, K)
                base.append(c)
                print(f"  base B={budget:3d} {arm:3s} seed={seed} "
                      f"J=({c['J_mean'][0]:.3f},{c['J_mean'][1]:.3f}) "
                      f"served={c['served']} late={c['served_late']} {c['elapsed_s']}s")

    # ---------- 扫描 ----------
    cells: list[dict] = []
    for budget in BUDGETS:
        for rc in RECLAIM_MODES:
            for n_cart in splits[budget]:
                for seed in seeds:
                    c = run_cell(budget, (budget - n_cart, n_cart), seed, n_ep, K, rc)
                    cells.append(c)
                    print(f"  quota[{_rname(rc)}] B={budget:3d} n_cart={n_cart:3d} "
                          f"seed={seed} J=({c['J_mean'][0]:.3f},{c['J_mean'][1]:.3f}) "
                          f"served={c['served']} late={c['served_late']} {c['elapsed_s']}s")

    # ---------- 汇总 ----------
    summary: dict = {}
    for budget in BUDGETS:
        rr = {c["seed"]: c for c in base if c["budget"] == budget and c["arm"] == "rr"}
        summary[budget] = {}
        for rc in RECLAIM_MODES:
            per_n: dict = {}
            for n_cart in splits[budget]:
                rows = [c for c in cells if c["budget"] == budget
                        and c["n_cart"] == n_cart and c["reclaim"] == rc]
                rs, geos = [], []
                for c in rows:
                    j = np.asarray(c["J_mean"], dtype=float)
                    jr = np.asarray(rr[c["seed"]]["J_mean"], dtype=float)
                    deg = bool(np.any(jr <= 1e-12))
                    r = (j / jr) if not deg else np.array([np.nan, np.nan])
                    g = float(np.sqrt(r[0] * r[1])) if not deg else float("nan")
                    rs.append({"seed": c["seed"], "r": r.tolist(), "geo": g})
                    if np.isfinite(g):
                        geos.append(g)
                per_n[str(n_cart)] = {
                    "per_seed": rs,
                    "r_mean": np.nanmean([[v for v in x["r"]] for x in rs], axis=0).tolist(),
                    "geo_mean": float(np.mean(geos)) if geos else float("nan"),
                    "geo_sd": float(np.std(geos, ddof=1)) if len(geos) > 1 else 0.0,
                    "J_uav_mean": float(np.mean([c["J_mean"][0] for c in rows])),
                    "J_cart_mean": float(np.mean([c["J_mean"][1] for c in rows])),
                    "served_uav": float(np.mean([c["served"][0] for c in rows])),
                    "served_cart": float(np.mean([c["served"][1] for c in rows])),
                    "served_total": float(np.mean([sum(c["served"]) for c in rows])),
                    "served_late_uav": float(np.mean([c["served_late"][0] for c in rows])),
                    "served_late_cart": float(np.mean([c["served_late"][1] for c in rows])),
                    "alive_uav": float(np.mean([c["alive_steps"][0] for c in rows])),
                    "alive_cart": float(np.mean([c["alive_steps"][1] for c in rows])),
                    "quota_violations": sum(len(c["quota_violations"]) for c in rows),
                }
            summary[budget][_rname(rc)] = per_n

    print("\n== 汇总（r_i 相对同档 rr，同种子配对）==")
    for budget in BUDGETS:
        for rc in RECLAIM_MODES:
            print(f"  --- B={budget}  [{_rname(rc)}] ---")
            for n_cart in splits[budget]:
                d = summary[budget][_rname(rc)][str(n_cart)]
                print(f"    n_cart={n_cart:3d} geo={d['geo_mean']:.3f}±{d['geo_sd']:.3f} "
                      f"r=({d['r_mean'][0]:.3f},{d['r_mean'][1]:.3f}) "
                      f"J=({d['J_uav_mean']:.3f},{d['J_cart_mean']:.3f}) "
                      f"served=({d['served_uav']:.0f},{d['served_cart']:.0f}) "
                      f"late=({d['served_late_uav']:.0f},{d['served_late_cart']:.0f}) "
                      f"alive=({d['alive_uav']:.0f},{d['alive_cart']:.0f})")

    # ---------- 记账判据 ----------
    print("\n== 记账判据 ==")
    nviol = sum(len(c["quota_violations"]) for c in cells)
    check("P1 配额上界（strict: served_i ≤ quotas_i；reclaim: ≤ B）", nviol == 0,
          f"违例 {nviol} 处，{len(cells)} cells")
    ok_res = all(c["budget_reserved"] == c["budget_attempted"] + c["budget_wasted"]
                 for c in cells)
    check("P2 reserved == attempted + wasted（三分账闭合）", ok_res, f"{len(cells)} cells")
    check("P3 预算不超过 B（每集 ≤ B）",
          all(c["budget_reserved"] <= c["budget"] * n_ep for c in cells),
          f"max reserved={max(c['budget_reserved'] for c in cells)}")

    # ---------- R0 回归 ----------
    print("\n== R0 与 results/42 的 rr/age 逐位回归 ==")
    # ★ P0-3（2026-10-19）：读"当前版本"（runs/ 最新优先，退回 v1 冻结件），并打印**实际**路径。
    #   硬编码 results/42_... 会在 P0-3 之后永远核对旧的冻结文件（自证伪 #46 同型）。
    ref_path = latest_result_path("42_t12_four_arms_summary")
    print(f"  [ref] {ref_path.relative_to(ROOT).as_posix() if ref_path else '<缺失>'}")
    ref_map = {}
    if ref_path is not None and ref_path.exists():
        ref = json.loads(ref_path.read_text(encoding="utf-8"))
        ref_map = {(c["budget"], c["arm"], c["seed"]): c["J_per_episode"]
                   for c in ref.get("cells_J", [])}
    diffs, compared = [], 0
    for c in base:
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
    if compared:
        check("R0 rr/age 与 T12 逐位相同", not diffs,
              ("逐位相同" if not diffs else "；".join(diffs[:4])) + f"（{compared} cells）")
    else:
        print("  [SKIP] R0：无可比 cell")
    rr_late = [(c["budget"], c["seed"], c["served_late"]) for c in base
               if c["arm"] == "rr" and c["budget"] == 100]
    check("S6' rr 在 B=100 的后半段发帧 == 0（预算在 t=B 花光）",
          all(sum(x[2]) == 0 for x in rr_late), f"{rr_late}")

    # ---------- F1 ----------
    print("\n== F1（geo* = min over n_cart；主口径 reclaim）==")
    f1 = {"gate": GATE, "primary_regime": "reclaim", "by_regime": {},
          "geo_star": {}, "argmin_n_cart": {}, "r_at_argmin": {},
          "max_r_at_argmin": {}, "degenerate_ns": {}, "verdict": {}, "note": {}}
    for budget in BUDGETS:
        ns = splits[budget]
        f1["by_regime"][str(budget)] = {}
        for rc in RECLAIM_MODES:
            rn = _rname(rc)
            tab = summary[budget][rn]
            cand = sorted((float(tab[str(n)]["geo_mean"]), n) for n in ns
                          if np.isfinite(tab[str(n)]["geo_mean"]))
            g_best, n_best = cand[0]
            r_best = tab[str(n_best)]["r_mean"]
            max_r = float(max(r_best))
            deg_ns = [n for n in ns
                      if np.isfinite(tab[str(n)]["geo_mean"])
                      and tab[str(n)]["geo_mean"] <= GATE
                      and max(tab[str(n)]["r_mean"]) > 1.10]
            f1["by_regime"][str(budget)][rn] = {
                "geo_star": g_best, "n_cart": int(n_best), "r": r_best,
                "max_r": max_r, "degenerate_ns": deg_ns,
                "served_total_at_argmin": tab[str(n_best)]["served_total"]}
            print(f"  B={budget:3d} [{rn:7s}] geo*={g_best:.3f} @ n_cart={n_best} "
                  f"r={tuple(round(x, 3) for x in r_best)} "
                  f"served={tab[str(n_best)]['served_total']:.0f}/{budget}")
            if rc:      # 只有主口径写进正式结论
                f1["geo_star"][str(budget)] = g_best
                f1["argmin_n_cart"][str(budget)] = int(n_best)
                f1["r_at_argmin"][str(budget)] = r_best
                f1["max_r_at_argmin"][str(budget)] = max_r
                f1["degenerate_ns"][str(budget)] = deg_ns
                if g_best <= GATE and max_r <= 1.10:
                    v = "FEASIBLE-ISH"
                    note = (f"配额族内最优 geo={g_best:.3f} ≤ {GATE}，"
                            f"且 max r={max_r:.3f} ≤ 1.10（n_cart={n_best}）")
                elif g_best <= GATE:
                    v = "DEGENERATE"
                    note = (f"geo={g_best:.3f} 过闸但 max r={max_r:.3f} > 1.10 "
                            f"⇒ 牺牲了某个用户，不算通过")
                else:
                    v = "INFEASIBLE-IN-QUOTA-FAMILY"
                    note = (f"配额族内最优 geo={g_best:.3f} > {GATE}（n_cart={n_best}）；"
                            f"★ 只证明族内不可达，不等于全局不可达")
                f1["verdict"][str(budget)] = v
                f1["note"][str(budget)] = note
                print(f"        ⇒ {v}：{note}")

    # ---------- 单调性（只报不判）----------
    mono = {}
    for budget in BUDGETS:
        mono[str(budget)] = {}
        for rc in RECLAIM_MODES:
            ns = splits[budget]
            tab = summary[budget][_rname(rc)]
            jc = [tab[str(n)]["J_cart_mean"] for n in ns]
            ju = [tab[str(n)]["J_uav_mean"] for n in ns]
            dc, du = np.diff(jc), np.diff(ju)
            mono[str(budget)][_rname(rc)] = {
                "n_cart": ns, "J_cart": jc, "J_uav": ju,
                "cart_max_rise": float(dc.max()) if dc.size else 0.0,   # >0 = 反向（变差）
                "uav_max_drop": float(du.min()) if du.size else 0.0,    # <0 = 反向（变好）
            }
            print(f"  B={budget} [{_rname(rc)}]: J_cart 最大反向增量 "
                  f"{mono[str(budget)][_rname(rc)]['cart_max_rise']:+.4f}；"
                  f"J_uav 最大反向降幅 {mono[str(budget)][_rname(rc)]['uav_max_drop']:+.4f}")

    # ---------- 端点 ----------
    ends = {}
    for budget in BUDGETS:
        tab = summary[budget]["reclaim"]
        ends[str(budget)] = {
            "uav_floor_candidate": tab["0"]["J_uav_mean"],
            "cart_floor_candidate": tab[str(budget)]["J_cart_mean"],
            "cart_floor_alive_steps": tab[str(budget)]["alive_cart"],
        }
        print(f"  B={budget} [reclaim]: UAV 端点 n_cart=0 J_uav="
              f"{ends[str(budget)]['uav_floor_candidate']:.3f}；"
              f"CartPole 端点 n_cart=B J_cart="
              f"{ends[str(budget)]['cart_floor_candidate']:.3f}"
              f"（存活 {ends[str(budget)]['cart_floor_alive_steps']:.0f}/200 步）")

    _dump({"selfchecks": sc, "experiment": {"cells": cells, "base": base}},
          cells, {"summary": summary, "f1": f1, "monotonicity": mono,
                  "endpoints": ends, "base": base}, n_ep, seeds,
          run_id=args.run_id, allow_overwrite=args.allow_overwrite, persist=not args.smoke)
    print(f"\n{'ALL CHECKS PASS' if not FAILS else 'FAILED: ' + '; '.join(FAILS)}")
    print(f"总耗时 {round(time.time() - t_all, 1)}s")
    return 0 if not FAILS else 2


def _dump(payload, cells, extra, n_ep, seeds, *, run_id=None, allow_overwrite=False,
          persist=True):
    out = {"config": {"N": 2, "window": WINDOW, "warmup": [WARMUP_UAV, 0],
                      "budgets": list(BUDGETS), "episodes": n_ep, "seeds": seeds,
                      "gate": GATE, "splits": {str(k): v for k, v in SPLITS.items()},
                      "absorbed": ABSORBED,
                      "regimes": ["reclaim", "strict"], "primary_regime": "reclaim",
                      "aggregation": "geo of per-user ratios vs rr (same budget, same seed)",
                      "prereg": "04-实验与代码/2026-10-19-T16-配额配速-预注册-v0.1.md"},
           "selfchecks": payload["selfchecks"],
           "checks_failed": list(FAILS)}
    (ROOT / "outputs").mkdir(exist_ok=True)
    (ROOT / "outputs" / "48_t16_quota_pacing.json").write_text(
        json.dumps({**out, **(extra or {}), "cells": (cells or [])},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    if cells is None or not persist:
        # `--skip-experiment` / `--smoke` 只跑自检或接线，**不得**写入库证据
        #（T16 在 47 上踩过这个坑：`--skip-experiment` 把 24 个 cell 覆盖成空表）。
        if cells is not None:
            print("[smoke] 不写 results/（smoke 数字不进证据链，也不抢 INDEX 的 current）")
        return
    slim = [{k: c[k] for k in ("budget", "n_cart", "quotas", "reclaim", "seed",
                               "J_per_episode", "J_mean", "served", "served_late",
                               "alive_steps", "frames", "attempts", "link_q",
                               "budget_reserved", "budget_attempted", "budget_wasted")}
            for c in (cells or [])]
    base_slim = [{k: c[k] for k in ("budget", "arm", "seed", "J_per_episode", "J_mean",
                                    "served", "served_late", "alive_steps")}
                 for c in ((extra or {}).get("base") or [])]
    # ★ P0-3（2026-10-19）：改走 run-id 分目录；同名 run-id 再写会 SystemExit(3)。
    dest = write_result("48_t16_quota_pacing",
                        {**out, **(extra or {}), "cells_J": slim, "base_J": base_slim},
                        run_id=run_id, allow_overwrite=allow_overwrite)
    print(f"[evidence] {dest.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    sys.exit(main())
