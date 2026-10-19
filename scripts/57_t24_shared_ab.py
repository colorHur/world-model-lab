#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""T24 · 共享 `(A,B_j,c)` 单模型 —— 覆盖率能不能抬起来，以及抬起来的代价是什么。

口径来源（★ 全部在看结果之前冻结）
------------------------------------
`04-实验与代码/2026-10-20-T24-共享AB单模型-预注册-v0.1.md` §0–§8。

一句话
------
把逐滞后阶的 `M_g` 换成"所有 `g` 共用一套 `(A,B_j,c)`"，`age≥1` 决策点上"有模型可用"的比例
能不能从 **17.6%** 抬到 **≥50%**，以及代价改善了多少、代价（高 `g` 的结构偏差）有多大。

设计（预注册 §3，与 T23-b 完全同一套）
--------------------------------------
    臂 = age；predictor ∈ {zoh, linear, shared, truedyn}
    mode ∈ {unpaced, paced}；budget ∈ {100, 200}；seed ∈ {0,1,2}   ⇒ 48 cell
    episodes=30, max_steps=200, absorbed=(25,1), GATE=0.90（全部不动）

★ 主表 = B=200 unpaced；B=100 只作次要分析且**必须**同报 `t>B` 发帧数（R17）。
★ `shared` 是**线性辨识**，不是世界模型；`TrueDyn` 一律标 `diagnostic`。
★ 共享口径对真实线性系统**局部误设**（`g≥2` 有结构偏差，预注册 §2b）⇒ 高 `g` 变差必须写"交易"。

用法
----
    D:/devtools/miniconda3/envs/ai-lab/python.exe scripts/57_t24_shared_ab.py
    ... --skip-experiment      # 只跑自检（C5 现跑 12 个 ZOH cell；不写 results/）
    ... --smoke                # 2 集 × 1 种子，不写 results/

产物
----
    outputs/57_t24_shared_ab[_smoke].json                完整表（不入库）
    results/runs/<run_id>/57_t24_shared_ab.json          **入库**（results_io 唯一入口）
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

from wmlab.eval.multiuser import default_uav_cost, delivery_rate, window_cost  # noqa: E402
from wmlab.eval.predictors import Linear, SharedLag, TrueDyn, make_predictor  # noqa: E402
from wmlab.eval.results_io import latest_result_path, write_result  # noqa: E402

FAILS: list[str] = []

#: age 分层（与 T23-b 同口径，便于与 17.6% 的分档直接比）
AGE_BUCKETS = (("1", 1, 1), ("2-3", 2, 3), (">=4", 4, 10 ** 6))


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


T12 = _load(ROOT / "scripts" / "42_t12_hetero_four_arms.py", "t12_mod57")
T52 = _load(ROOT / "scripts" / "52_t21_paced_matching_ablation.py", "t52_mod57")

WINDOW = T12.WINDOW
WARMUP_UAV = T12.WARMUP_UAV
BUDGETS = list(T12.BUDGETS)
GATE = T12.GATE
ABSORBED = [T12.ABSORBED_UAV, T12.ABSORBED_CART]
COSTS = [default_uav_cost, T12.cartpole_cost]
ARM = "age"                                   # ★ 预注册 §3：固定臂，只比预测
PREDICTORS = ("zoh", "linear", "shared", "truedyn")
MODES = ("unpaced", "paced")

#: ★ 覆盖率门：新门 50%（预注册 §4-3），旧门 20% **必须同时报**
COV_GATE_NEW = 0.50
COV_GATE_OLD = 0.20
#: T23-b 30 集实跑的合并覆盖率（只作对照打印，不当本轮结果）
COV_T23B_REF = 0.176


def check(tag: str, ok: bool, detail: str) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {tag}: {detail}")
    if not ok:
        FAILS.append(f"{tag}: {detail}")


# ================================================================ 装置
def _run(budget, mode, seed, n_ep, *, predictor=None, probe=None):
    envs = [T12.make_env("uav", seed=seed, max_steps=WARMUP_UAV + WINDOW),
            T12.make_env("CartPole-v1", seed=seed)]
    ctrls = [T12.PDRelativeController(), T12.CartPoleLQR(_run.K)]
    alloc = T52.make_alloc(ARM, mode, budget)
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


class _ErrProbe:
    """逐决策点采"预测误差 + age 分档"，只在内存里聚合（48 cell 的逐步原始值太大，不入库）。

    ★ 误差定义与 T23-b 逐位一致：`‖est_pre − true_pre‖₂`，`age ≥ 1` 才算预测。
    """

    def __init__(self) -> None:
        self.err: dict[int, dict[str, list]] = {i: {b[0]: [] for b in AGE_BUCKETS} for i in range(2)}
        self.n_age: dict[int, dict[int, int]] = {0: {}, 1: {}}
        self.n_pred_calls: dict[int, int] = {0: 0, 1: 0}
        self.max_abs = 0.0

    def __call__(self, p: dict) -> None:
        for i in (0, 1):
            if not p["alive"][i]:
                continue
            a = int(p["age"][i])
            self.n_age[i][a] = int(self.n_age[i].get(a, 0)) + 1
            if a == 0:
                continue                      # 送达值，不是预测
            self.n_pred_calls[i] += 1
            e = float(np.linalg.norm(np.asarray(p["est_pre"][i], dtype=np.float64)
                                     - np.asarray(p["true_pre"][i], dtype=np.float64)))
            self.max_abs = max(self.max_abs, e)
            for tag, lo, hi in AGE_BUCKETS:
                if lo <= a <= hi:
                    self.err[i][tag].append(e)
                    break

    def summary(self) -> dict:
        out = {}
        for i in (0, 1):
            buckets = {}
            for tag, _lo, _hi in AGE_BUCKETS:
                v = np.asarray(self.err[i][tag], dtype=float)
                buckets[tag] = ({"n": int(v.size)} if v.size == 0 else
                                {"n": int(v.size), "median": float(np.median(v)),
                                 "mean": float(v.mean()), "p90": float(np.percentile(v, 90)),
                                 "max": float(v.max())})
            out[str(i)] = {"buckets": buckets,
                           "n_age": {str(k): int(v) for k, v in sorted(self.n_age[i].items())},
                           "n_pred_calls": int(self.n_pred_calls[i])}
        return out


# ================================================================ 合成系统（C1–C3）
_SHUFFLE_SEED = 20261020


def _synth(seed: int = 11, dim: int = 3, act_dim: int = 2, n_steps: int = 400):
    rng = np.random.default_rng(seed)
    A = rng.normal(0.0, 0.35, (dim, dim))
    A = A / max(np.max(np.abs(np.linalg.eigvals(A))), 1e-12) * 0.9
    B = rng.normal(0.0, 0.5, (dim, act_dim))
    c = rng.normal(0.0, 0.2, dim)
    xs = [rng.normal(0.0, 1.0, dim)]
    acts = []
    for _ in range(n_steps):
        a = rng.normal(0.0, 1.0, act_dim)
        acts.append(a)
        xs.append(A @ xs[-1] + B @ a + c)
    return A, B, c, xs, acts


def _pairs_gaps(xs, acts, gaps):
    """按**一条送达时间线**切段：段首尾不重叠（与真实流一致；#51 的教训）。"""
    s, out = 0, []
    for g in gaps:
        s2 = s + g
        out.append((xs[s], [acts[j].copy() for j in range(s, s2)], xs[s2]))
        s = s2
    return out


def _feed(p, start_x, pairs):
    """按 runner 的调用顺序喂：先首帧，再逐段送达（`age_before = len(seq)−1` ⇒ `gap_prev = g`）。"""
    p.observe_delivery(0, start_x, age_before=0)
    for _x, seq, y in pairs:
        p.observe_delivery(0, y, age_before=len(seq) - 1, action=seq[-1], action_seq=seq)
    return p


def _pred_rel(p, x_q, acts_q, truth) -> float:
    pred = np.asarray(p.predict(0, age=len(acts_q), actions_since=acts_q), dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    return float(np.linalg.norm(pred - truth) / max(np.linalg.norm(truth), 1e-300))


# ================================================================ 自检
def selfchecks(n_ep_small: int = 2) -> dict:
    print("\n== 自检 C1–C4 / C8（预注册 §5：每条写明能打死什么错）==")
    out: dict = {}
    A, B, c, xs, acts = _synth()
    ad = B.shape[1]
    rng = np.random.default_rng(999)

    # ---------------- C1：无样本时 ≡ Linear（含 age>max_gap 的退化）----------------
    h = _feed(SharedLag(), xs[0], [])
    lin = _feed(Linear(), xs[0], [])
    d1, worst = {}, 0.0
    for age in range(0, 8):
        aq = [rng.normal(0.0, 1.0, ad) for _ in range(age)]
        o1 = np.asarray(h.predict(0, age=age, actions_since=aq))
        o2 = np.asarray(lin.predict(0, age=age, actions_since=aq))
        d = float(np.max(np.abs(o1 - o2)))
        d1[str(age)] = d
        worst = max(worst, d)
    check("C1 无任何送达对时，age∈{0..7} 与 Linear **逐位相同**（打「退化路径分叉」）",
          worst == 0.0, f"max|Δ|={worst:.3e}")
    out["C1"] = d1

    # ---------------- C2：gap==1 数据 ⇒ age=1 精确；打乱动作 ⇒ 崩 ----------------
    gaps1 = [1] * 44
    pairs1 = _pairs_gaps(xs, acts, gaps1)
    s1 = _feed(SharedLag(), xs[0], pairs1)
    perm = np.random.default_rng(_SHUFFLE_SEED).permutation(len(pairs1))
    pairs_shuf = [(pairs1[k][0], pairs1[int(perm[k])][1], pairs1[k][2]) for k in range(len(pairs1))]
    s_shuf = _feed(SharedLag(), xs[0], pairs_shuf)
    x_q = np.asarray(pairs1[-1][2], dtype=np.float64)
    c2 = {}
    for tag, p in (("shared", s1), ("shared_action_shuffled", s_shuf)):
        aq = [rng.normal(0.0, 1.0, ad)]
        truth = A @ x_q + B @ aq[0] + c
        c2[tag] = _pred_rel(p, x_q, aq, truth)
    c2["n_samples"] = int(s1.describe()["users"]["0"]["n_samples"])
    c2["n_needed"] = s1.describe()["users"]["0"]["n_needed"]
    ok2 = (c2["shared"] < 1e-3 and c2["shared_action_shuffled"] > 0.1
           and c2["n_samples"] >= c2["n_needed"])
    check("C2 合成线性系统 gap==1：age=1 相对误差 <1e-3；动作块打乱后 >0.1（打「没学到动作依赖」）",
          ok2, json.dumps(c2, ensure_ascii=False))
    out["C2"] = c2

    # ---------------- C3：混合 gap 只进**一套**正规方程，且过门后覆盖 1..6 ----------------
    g_mix = [1, 2, 3, 4, 5, 6] * 10
    s3 = _feed(SharedLag(), xs[0], _pairs_gaps(xs, acts, g_mix))
    d3 = s3.describe()["users"]["0"]
    used = set()
    for age in range(1, 7):
        aq = [rng.normal(0.0, 1.0, ad) for _ in range(age)]
        s3.predict(0, age=age, actions_since=aq)
    desc3 = s3.describe()["users"]["0"]
    used = {int(k) for k, v in desc3["used_model"].items() if int(v) > 0}
    ok3 = (len(s3._joint) == 1 and int(d3["n_samples"]) == len(g_mix)
           and set(int(k) for k in d3["gaps_seen"]) == {1, 2, 3, 4, 5, 6}
           and used >= {1, 2, 3, 4, 5, 6})
    check("C3 混合 gap∈{1..6} 只进一套系统（n=全部对数）、gaps_seen 全档可见、过门后 used_model ⊇ {1..6}",
          ok3, f"n={d3['n_samples']} need={d3['n_needed']} gaps={d3['gaps_seen']} used={sorted(used)}")
    out["C3"] = {"n_samples": d3["n_samples"], "n_needed": d3["n_needed"],
                 "gaps_seen": d3["gaps_seen"], "used_model_ages": sorted(used)}

    # ---------------- C4：TrueDyn 误差必须精确 0，ZOH 必须 >0 ----------------
    c4 = {}
    for name in ("truedyn", "zoh"):
        pr = _ErrProbe()
        p = make_predictor(name)
        _run(200, "unpaced", 0, n_ep_small, predictor=p, probe=pr)
        s = pr.summary()
        n = sum(v["buckets"][t]["n"] for v in s.values() for t, _lo, _hi in AGE_BUCKETS)
        mx = max([v["buckets"][t].get("max", 0.0) for v in s.values()
                  for t, _lo, _hi in AGE_BUCKETS] or [0.0])
        c4[name] = {"n_pred_points": n, "max_err": mx}
    ok4 = (c4["truedyn"]["max_err"] == 0.0 and c4["truedyn"]["n_pred_points"] > 0
           and c4["zoh"]["max_err"] > 0.0)
    check("C4 TrueDyn 在 age≥1 上误差**精确 0**（n>0）且 ZOH 误差 >0（打「静默退化成 ZOH」）",
          ok4, str(c4))
    out["C4"] = c4

    # ---------------- C8：覆盖率计数与 describe() 逐档一致（#54：必须用跨集累计）----------------
    sp = make_predictor("shared")
    pr2 = _ErrProbe()
    _run(200, "unpaced", 0, n_ep_small, predictor=sp, probe=pr2)
    d2 = sp.describe()["cum_stats"]
    c8, ok8 = {}, True
    for i in (0, 1):
        um = {int(k): int(v) for k, v in d2[str(i)]["used_model"].items()}
        dg = {int(k): int(v) for k, v in d2[str(i)]["used_degraded"].items()}
        calls = {a: um.get(a, 0) + dg.get(a, 0) for a in set(um) | set(dg)}
        probe_calls = {a: n for a, n in pr2.n_age[i].items()}
        c8[str(i)] = {"describe_calls_by_age": {str(k): v for k, v in sorted(calls.items())},
                      "probe_calls_by_age": {str(k): v for k, v in sorted(probe_calls.items())}}
        if {k: v for k, v in calls.items() if v} != {k: v for k, v in probe_calls.items() if v}:
            ok8 = False
    check("C8 describe(cum).逐 age 调用数与 pred_probe 记录的逐 age 决策点数**逐档相等**",
          ok8, json.dumps(c8, ensure_ascii=False)[:400])
    out["C8"] = c8
    return out


# ================================================================ C5：ZOH 逐位回归
def c5_zoh_regression(n_ep: int, seeds, budgets, cells=None) -> dict:
    """`ZOH` 必须与 `results/52` 的 `age` 臂逐 episode J **逐位相同**。

    ★ 同一区间纪律（#49/#52）：历史是 30 集，集数不同就不可比 ⇒ 非 30 集时打印 SKIP、不计 FAIL。
    """
    print("\n== C5 与 results/52 的 age 臂逐 episode J 逐位回归（ZOH = 现状）==")
    if int(n_ep) != 30:
        print(f"  [SKIP] 本 runner 集数 {n_ep} ≠ 历史 30 集 ⇒ 不可比（不是通过，也不是失败）")
        return {"skipped": f"episodes={n_ep} != 30"}
    if cells is None:
        cells = [run_cell(b, m, "zoh", s, n_ep) for b in budgets for m in MODES for s in seeds]
    path = latest_result_path("52_t21_paced_matching_ablation")
    print(f"  [ref-52] {path.relative_to(ROOT).as_posix() if path else '<缺失>'}")
    res = {"compared": 0, "diffs": [], "ref": (path.relative_to(ROOT).as_posix() if path else None)}
    if path is None or not path.exists():
        res["diffs"] = ["<无参考>"]
    else:
        ref = json.loads(path.read_text(encoding="utf-8"))
        ref_map = {(c["budget"], c["arm"], c["mode"], c["seed"]): c["J_per_episode"]
                   for c in ref.get("cells_J", [])}
        for c in cells:
            if c["predictor"] != "zoh":
                continue
            key = (c["budget"], ARM, c["mode"], c["seed"])
            if key not in ref_map:
                continue
            a = np.asarray(c["J_per_episode"], dtype=float)
            b = np.asarray(ref_map[key], dtype=float)
            if a.shape != b.shape:
                res["diffs"].append(f"{key} 形状 {a.shape} vs {b.shape}")
                continue
            res["compared"] += 1
            if not np.array_equal(a, b):
                res["diffs"].append(f"{key} max|Δ|={np.nanmax(np.abs(a - b)):.3e}")
    check("C5 ZOH 12 cell 与 results/52 的 age 臂逐 episode J **逐位相同**",
          res["compared"] > 0 and not res["diffs"],
          (f"{res['compared']} cells 逐位相同" if not res["diffs"] else "；".join(res["diffs"][:4])) +
          ("" if res["compared"] else "（0 可比 cell ⇒ 不算通过）"))
    return res


# ================================================================ 单 cell
def _coverage(pred) -> dict | None:
    """共享模型在 `age≥1` 决策点上的启用率（**跨集累计**，`reset_episode` 不清空）。"""
    d = pred.describe()
    cum = d["cum_stats"]
    cov = {}
    for i in (0, 1):
        um = {int(k): int(v) for k, v in cum[str(i)]["used_model"].items()}
        dg = {int(k): int(v) for k, v in cum[str(i)]["used_degraded"].items()}
        tot = {a: um.get(a, 0) + dg.get(a, 0) for a in set(um) | set(dg)}
        n_all = sum(v for a, v in tot.items() if a >= 1)
        n_used = sum(v for a, v in um.items() if a >= 1)
        by_bucket = {}
        for tag, lo, hi in AGE_BUCKETS:
            t_all = sum(v for a, v in tot.items() if lo <= a <= hi)
            t_used = sum(v for a, v in um.items() if lo <= a <= hi)
            by_bucket[tag] = {"n": int(t_all), "used": int(t_used),
                              "frac": (float(t_used) / t_all if t_all else None)}
        cov[str(i)] = {"n_decisions_age_ge1": int(n_all), "n_used_model": int(n_used),
                       "frac": (float(n_used) / n_all if n_all else None),
                       "by_age_bucket": by_bucket,
                       # ★ 作用域写进键名（#49 同族：这两个只是**最后一集**的）
                       "last_episode_n_samples": d["users"].get(str(i), {}).get("n_samples"),
                       "last_episode_n_needed": d["users"].get(str(i), {}).get("n_needed"),
                       "gaps_seen_cum": d["users"].get(str(i), {}).get("gaps_seen")}
    return cov


def run_cell(budget, mode, pred_name, seed, n_ep):
    t0 = time.time()
    pred = make_predictor(pred_name)
    probe = _ErrProbe()
    r = _run(budget, mode, seed, n_ep, predictor=pred, probe=probe)
    rec = r["records"]
    J = _J(rec, n_ep)
    dr = delivery_rate(rec, 2)
    served = [sum(1 for x in rec if x.attempted_idx == i) for i in range(2)]
    late = [sum(1 for x in rec if x.t > budget and x.attempted_idx == i) for i in range(2)]
    cov = _coverage(pred) if pred_name == "shared" else None
    return {
        "budget": int(budget), "mode": mode, "predictor": pred_name, "seed": int(seed),
        "J_per_episode": np.asarray(J, dtype=float).tolist(),
        "J_mean": np.asarray(J, dtype=float).mean(axis=0).tolist(),
        "served": served, "served_late": late, "served_total": int(sum(served)),
        "alive_steps": dr["alive_steps"], "frames": dr["frames"], "attempts": dr["attempts"],
        "link_q": dr["link_q"], "p_eff": dr["p_eff"],
        "budget_reserved": dr["budget_reserved"], "budget_attempted": dr["budget_attempted"],
        "budget_wasted": dr["budget_wasted"],
        "pred_error": probe.summary(), "coverage": cov,
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
    ap.add_argument("--allow-overwrite", action="store_true")
    args = ap.parse_args()
    global SMOKE_TAG
    SMOKE_TAG = "_smoke" if args.smoke else ""
    n_ep = 2 if args.smoke else args.episodes
    seeds = [0] if args.smoke else args.seeds
    budgets = list(BUDGETS)

    t_all = time.time()
    print(f"[57] T24 共享 (A,B) 单模型｜N=2｜窗口 {WINDOW}｜臂={ARM}｜预算 {budgets}｜"
          f"seeds={seeds}｜episodes={n_ep}｜predictors={PREDICTORS}｜modes={MODES}")
    sc = selfchecks()

    if args.skip_experiment:
        sc["C5"] = c5_zoh_regression(n_ep, seeds, budgets, None)
        _dump({"selfchecks": sc}, None, None, n_ep, seeds)
        print(f"\n{'ALL CHECKS PASS' if not FAILS else 'FAILED: ' + '; '.join(FAILS)}")
        return 0 if not FAILS else 2

    cells: list[dict] = []
    for budget in budgets:
        for mode in MODES:
            for pname in PREDICTORS:
                for seed in seeds:
                    c = run_cell(budget, mode, pname, seed, n_ep)
                    cells.append(c)
                    print(f"  B={budget:3d} {mode:7s} {pname:8s} seed={seed} "
                          f"J=({c['J_mean'][0]:.3f},{c['J_mean'][1]:.3f}) "
                          f"served=({c['served'][0]},{c['served'][1]}) "
                          f"late=({c['served_late'][0]},{c['served_late'][1]}) "
                          f"{c['elapsed_s']}s")

    # ---------- 汇总 ----------
    def get(budget, mode, pname, seed):
        for c in cells:
            if (c["budget"], c["mode"], c["predictor"], c["seed"]) == (budget, mode, pname, seed):
                return c
        return None

    def ratio_table(budget, mode, num, den, seeds_):
        """同 budget/mode/seed 配对的逐用户比值 r_i（分母 ≤1e-12 记 NaN，不硬除）。"""
        rows = []
        for s in seeds_:
            a = np.asarray(get(budget, mode, num, s)["J_mean"], dtype=float)
            b = np.asarray(get(budget, mode, den, s)["J_mean"], dtype=float)
            if np.any(b <= 1e-12):
                rows.append({"seed": s, "r": [float("nan")] * 2, "geo": float("nan")})
                continue
            r = a / b
            rows.append({"seed": s, "r": r.tolist(), "geo": float(np.sqrt(r[0] * r[1]))})
        rr = np.asarray([x["r"] for x in rows], dtype=float)
        gg = np.asarray([x["geo"] for x in rows], dtype=float)
        return {"per_seed": rows,
                "r_mean": np.nanmean(rr, axis=0).tolist(),
                "r_sd": (np.nanstd(rr, axis=0, ddof=1).tolist() if len(rows) > 1 else [0.0, 0.0]),
                "geo_mean": float(np.nanmean(gg)),
                "geo_sd": (float(np.nanstd(gg, ddof=1)) if len(rows) > 1 else 0.0),
                "n_seeds_both_below_1": int(np.sum(np.all(rr < 1.0, axis=1))),
                "n_seeds_any_above_1": int(np.sum(np.any(rr > 1.0, axis=1)))}

    summary: dict = {}
    for budget in budgets:
        summary[str(budget)] = {}
        for mode in MODES:
            per = {}
            for pname in PREDICTORS:
                rows = [get(budget, mode, pname, s) for s in seeds]
                d = ratio_table(budget, mode, pname, "zoh", seeds) if pname != "zoh" else None
                per[pname] = {
                    "vs_zoh": d,
                    "J_uav_mean": float(np.mean([x["J_mean"][0] for x in rows])),
                    "J_cart_mean": float(np.mean([x["J_mean"][1] for x in rows])),
                    "served_uav": float(np.mean([x["served"][0] for x in rows])),
                    "served_cart": float(np.mean([x["served"][1] for x in rows])),
                    "late_total": float(np.mean([sum(x["served_late"]) for x in rows])),
                    "alive_uav": float(np.mean([x["alive_steps"][0] for x in rows])),
                    "alive_cart": float(np.mean([x["alive_steps"][1] for x in rows])),
                    "elapsed_s_mean": float(np.mean([x["elapsed_s"] for x in rows])),
                }
            # ★ 机制对照：shared / linear（预注册 §4-2）
            per["_shared_vs_linear"] = ratio_table(budget, mode, "shared", "linear", seeds)
            summary[str(budget)][mode] = per

    print("\n== 汇总（r_i 相对**同档同模式同种子**的 ZOH）==")
    for budget in budgets:
        for mode in MODES:
            for pname in PREDICTORS:
                d = summary[str(budget)][mode][pname]
                if pname == "zoh":
                    print(f"  B={budget:3d} {mode:7s} zoh      "
                          f"J=({d['J_uav_mean']:.3f},{d['J_cart_mean']:.3f})（基线）"
                          f" late={d['late_total']:.0f} alive_cart={d['alive_cart']:.1f}")
                    continue
                v = d["vs_zoh"]
                print(f"  B={budget:3d} {mode:7s} {pname:8s} "
                      f"geo={v['geo_mean']:.3f}±{v['geo_sd']:.3f} "
                      f"r=({v['r_mean'][0]:.3f}±{v['r_sd'][0]:.3f},"
                      f"{v['r_mean'][1]:.3f}±{v['r_sd'][1]:.3f}) "
                      f"J=({d['J_uav_mean']:.3f},{d['J_cart_mean']:.3f})")
            sl = summary[str(budget)][mode]["_shared_vs_linear"]
            print(f"  B={budget:3d} {mode:7s} shared/linear geo={sl['geo_mean']:.3f} "
                  f"r=({sl['r_mean'][0]:.3f},{sl['r_mean'][1]:.3f})")

    # ---------- C6：覆盖率（★ 本轮主指标）----------
    print("\n== C6 覆盖率（shared，30 集累计；分母 = age≥1 决策点）==")
    cov = {}
    for budget in budgets:
        for mode in MODES:
            rows = [c for c in cells if c["budget"] == budget and c["mode"] == mode
                    and c["predictor"] == "shared"]
            if not rows:
                continue
            tot = sum(r0["coverage"][u]["n_decisions_age_ge1"] for r0 in rows for u in ("0", "1"))
            used = sum(r0["coverage"][u]["n_used_model"] for r0 in rows for u in ("0", "1"))
            b23 = {}
            for tag, _lo, _hi in AGE_BUCKETS:
                n_ = sum(r0["coverage"][u]["by_age_bucket"][tag]["n"]
                         for r0 in rows for u in ("0", "1"))
                u_ = sum(r0["coverage"][u]["by_age_bucket"][tag]["used"]
                         for r0 in rows for u in ("0", "1"))
                b23[tag] = {"n": int(n_), "used": int(u_), "frac": (float(u_) / n_ if n_ else None)}
            frac = (float(used) / tot if tot else None)
            cov[f"{budget}/{mode}"] = {
                "n_decisions_age_ge1": int(tot), "n_used": int(used), "frac": frac,
                "gate_50pct_new_met": bool(tot and frac >= COV_GATE_NEW),
                "gate_20pct_old_met": bool(tot and frac >= COV_GATE_OLD),
                "by_age_bucket": b23,
                # 逐 cell 的覆盖率（判断"是否是某一种子异常带起来的"）
                "per_cell_frac": [{"seed": r0["seed"], "uav": r0["coverage"]["0"]["frac"],
                                   "cart": r0["coverage"]["1"]["frac"]} for r0 in rows],
            }
            print(f"  B={budget:3d} {mode:7s} 合并 {used}/{tot} = "
                  f"{100.0 * frac if frac is not None else float('nan'):.1f}% "
                  f"（新门 50% {'过' if cov[f'{budget}/{mode}']['gate_50pct_new_met'] else '未过'}／"
                  f"旧门 20% {'过' if cov[f'{budget}/{mode}']['gate_20pct_old_met'] else '未过'}）"
                  f" 分档 age1={100.0 * (b23['1']['frac'] or 0):.1f}% "
                  f"age2-3={100.0 * (b23['2-3']['frac'] or 0):.1f}% "
                  f"age>=4={100.0 * (b23['>=4']['frac'] or 0):.1f}%")
    print(f"  [对照] T23-b（hybrid，30 集）= {COV_T23B_REF:.1%}；本轮分母不同 ⇒ 只作量级对照")

    c6_key = "200/unpaced"
    c6 = {"key": c6_key, **cov.get(c6_key, {})}
    check("C6 新门：B=200 unpaced 的合并覆盖率 ≥50%（旧门 20% 同时报）",
          bool(cov.get(c6_key, {}).get("gate_50pct_new_met")),
          f"{cov.get(c6_key, {}).get('n_used')}/{cov.get(c6_key, {}).get('n_decisions_age_ge1')} = "
          f"{100.0 * (cov.get(c6_key, {}).get('frac') or 0):.1f}%（旧门 20% "
          f"{'过' if cov.get(c6_key, {}).get('gate_20pct_old_met') else '未过'}）")

    # ---------- C7 记账判据 ----------
    print("\n== C7 记账判据 ==")
    bad_p1 = [c for c in cells if c["served_total"] != n_ep * c["budget"]]
    check("C7a 每 cell 总发帧 == episodes × budget", not bad_p1,
          f"{len(cells)} cells" if not bad_p1 else str(bad_p1[:3]))
    bad_p2 = [c for c in cells if c["budget_reserved"] != c["budget_attempted"] + c["budget_wasted"]]
    check("C7b reserved == attempted + wasted（三分账闭合）", not bad_p2,
          "全 cell" if not bad_p2 else str(bad_p2[:3]))
    bad_unp = [c for c in cells if c["mode"] == "unpaced" and c["budget"] < WINDOW
               and sum(c["served_late"]) != 0]
    check("C7c 裸臂 B<L 档 t>B 发帧恒 0（缺陷仍在，没被悄悄修掉）", not bad_unp,
          "全部 0" if not bad_unp else str(bad_unp[:3]))
    bad_pac = [c for c in cells if c["mode"] == "paced" and c["budget"] < WINDOW
               and sum(c["served_late"]) <= 0]
    check("C7d 配速臂 B<L 档 t>B 发帧 >0（R17：配速确实生效）", not bad_pac,
          "全部 >0" if not bad_pac else str(bad_pac[:3]))

    c5 = c5_zoh_regression(n_ep, seeds, budgets, cells)

    # ---------- F1：B=200 unpaced 下 shared 的两用户 r_i ----------
    print("\n== F1（主：B=200 unpaced，shared 的 r_i 相对 ZOH）==")
    f1 = {}
    for budget in budgets:
        b = str(budget)
        v = summary[b]["unpaced"]["shared"]["vs_zoh"]
        s = summary[b]["unpaced"]["shared"]
        z = summary[b]["unpaced"]["zoh"]
        f1[b] = {"r_mean": v["r_mean"], "r_sd": v["r_sd"], "geo": v["geo_mean"],
                 "geo_sd": v["geo_sd"], "per_seed": v["per_seed"],
                 "both_below_1_mean": bool(all(x < 1.0 for x in v["r_mean"])),
                 "any_above_1_mean": bool(any(x > 1.0 for x in v["r_mean"])),
                 "seeds_both_below_1": v["n_seeds_both_below_1"],
                 "seeds_any_above_1": v["n_seeds_any_above_1"],
                 "n_seeds": len(seeds),
                 "late": s["late_total"],
                 # ★ 帧数反馈（R17 同源）：存活变化会通过 `age` 调度改帧分配
                 "served_zoh": [z["served_uav"], z["served_cart"]],
                 "served_shared": [s["served_uav"], s["served_cart"]],
                 "alive_zoh": [z["alive_uav"], z["alive_cart"]],
                 "alive_shared": [s["alive_uav"], s["alive_cart"]]}
        print(f"  B={budget:3d} unpaced shared r=({v['r_mean'][0]:.3f},{v['r_mean'][1]:.3f}) "
              f"geo={v['geo_mean']:.3f} 均值both<1={f1[b]['both_below_1_mean']} "
              f"任一种子>1={v['n_seeds_any_above_1']}/{len(seeds)} "
              f"served {z['served_uav']:.0f}/{z['served_cart']:.0f}→"
              f"{s['served_uav']:.0f}/{s['served_cart']:.0f}")
    v200 = f1["200"]
    if v200["both_below_1_mean"] and v200["seeds_any_above_1"] == 0:
        f1["verdict"] = "BOTH-IMPROVED-ALL-SEEDS"
    elif v200["both_below_1_mean"]:
        f1["verdict"] = (f"MEAN-BOTH-IMPROVED-BUT-"
                         f"{v200['seeds_any_above_1']}/{v200['n_seeds']}-SEEDS-DEVIATE")
    elif v200["any_above_1_mean"]:
        f1["verdict"] = "TRADE-OFF"
    else:
        f1["verdict"] = "NO-IMPROVEMENT"
    print(f"  ⇒ {f1['verdict']}（#44/#56：只写「两用户都变好」或「交易」，均值与逐种子分开写）")

    # ---------- F2：机制对照 shared vs Linear ----------
    print("\n== F2（机制：shared/Linear，只差「用了共享模型」这一支）==")
    f2 = {}
    for budget in budgets:
        for mode in MODES:
            sl = summary[str(budget)][mode]["_shared_vs_linear"]
            a = summary[str(budget)][mode]["shared"]
            b_ = summary[str(budget)][mode]["linear"]
            f2[f"{budget}/{mode}"] = {"r_mean": sl["r_mean"], "geo": sl["geo_mean"],
                                      "per_seed": sl["per_seed"],
                                      "any_above_1": bool(any(x > 1.0 for x in sl["r_mean"])),
                                      "served_linear": [b_["served_uav"], b_["served_cart"]],
                                      "served_shared": [a["served_uav"], a["served_cart"]]}
            print(f"  B={budget:3d} {mode:7s} r=({sl['r_mean'][0]:.3f},{sl['r_mean'][1]:.3f}) "
                  f"geo={sl['geo_mean']:.3f} served(lin→sh)="
                  f"{b_['served_uav']:.0f}/{b_['served_cart']:.0f}→{a['served_uav']:.0f}/{a['served_cart']:.0f}")
    r2 = f2["200/unpaced"]["r_mean"]
    f2["negative_condition_hit"] = bool(any(x >= 1.0 for x in r2))
    f2["mixed_trade"] = bool(any(x >= 1.0 for x in r2) and any(x < 1.0 for x in r2))
    # ★ 预注册 §6 的否定条件只针对 **UAV 侧**：`r_uav ≥ 1.0` ⇒ 停止加模型复杂度
    f2["stop_on_uav"] = bool(r2[0] >= 1.0)
    print(f"  ⇒ r_uav(shared/linear)={r2[0]:.3f} ⇒ 否定条件（r_uav≥1）"
          f"{'命中：停止加模型复杂度，转 P3' if f2['stop_on_uav'] else '未命中'}")

    # ---------- F3：TrueDyn 诊断上界 ----------
    print("\n== F3（诊断：TrueDyn/ZOH 的量级参考，不得当可达收益）==")
    f3 = {str(b): {"r_mean": summary[str(b)]["unpaced"]["truedyn"]["vs_zoh"]["r_mean"],
                   "geo": summary[str(b)]["unpaced"]["truedyn"]["vs_zoh"]["geo_mean"]}
          for b in budgets}
    for b in budgets:
        print(f"  B={b:3d} unpaced truedyn r=({f3[str(b)]['r_mean'][0]:.3f},"
              f"{f3[str(b)]['r_mean'][1]:.3f}) geo={f3[str(b)]['geo']:.3f} [diagnostic]")

    # ---------- ERR：预测误差分层（★ 预测质量，不是收益）----------
    print("\n== ERR（B=200 unpaced，seed 0，按年龄三档中位；标「预测质量」不标收益）==")
    err = {}
    for pname in PREDICTORS:
        rows = [c for c in cells if c["budget"] == 200 and c["mode"] == "unpaced"
                and c["predictor"] == pname and c["seed"] == seeds[0]]
        if not rows:
            continue
        c = rows[0]
        err[pname] = {u: {t: c["pred_error"][u]["buckets"][t] for t, _lo, _hi in AGE_BUCKETS}
                      for u in ("0", "1")}
        print(f"  {pname:8s} UAV(age1)={c['pred_error']['0']['buckets']['1'].get('median', float('nan')):.4f} "
              f"UAV(age2-3)={c['pred_error']['0']['buckets']['2-3'].get('median', float('nan')):.4f} "
              f"Cart(age1)={c['pred_error']['1']['buckets']['1'].get('median', float('nan')):.4f}")

    _dump({"selfchecks": sc}, cells,
          {"summary": summary, "F1": f1, "F2": f2, "F3": f3, "ERR": err, "COV": cov, "C6": c6,
           "C5_zoh_regression": c5},
          n_ep, seeds, run_id=args.run_id, allow_overwrite=args.allow_overwrite,
          persist=not args.smoke)
    print(f"\n{'ALL CHECKS PASS' if not FAILS else 'FAILED: ' + '; '.join(FAILS)}")
    print(f"总耗时 {round(time.time() - t_all, 1)}s")
    return 0 if not FAILS else 2


def _dump(payload, cells, extra, n_ep, seeds, *, run_id=None, allow_overwrite=False,
          persist=True):
    cfg = {
        "N": 2, "window": WINDOW, "warmup": [WARMUP_UAV, 0], "budgets": list(BUDGETS),
        "episodes": n_ep, "seeds": seeds, "gate": GATE, "arm": ARM,
        "absorbed": [float(x) for x in ABSORBED], "predictors": list(PREDICTORS),
        "modes": list(MODES),
        "shared_definition": ("x_{k+g} ≈ A x_k + Σ_{j≤g} B_j a_{k+j} + c，A/B_j/c 全 g 共享；"
                              "max_gap=6；age>6 或 n<F_total 落 Linear（同一函数）"),
        "shared_bias_note": ("真实线性系统的 g 步映射系数是 A^{g-j}B ⇒ g≥2 结构偏差；"
                             "这是偏差-方差交换，高 g 变差写「交易」"),
        "coverage_gate": {"new": COV_GATE_NEW, "old_reported": COV_GATE_OLD,
                          "denominator": "age≥1 决策点（本轮 30 集 × 3 种子累计）"},
        "aggregation": "r_i = J_i(pred)/J_i(ZOH)，同 budget/mode/seed 配对；geo=sqrt(r0*r1)",
        "prereg": "04-实验与代码/2026-10-20-T24-共享AB单模型-预注册-v0.1.md",
    }
    out = {"config": cfg, "selfchecks": payload.get("selfchecks"),
           "checks_failed": list(FAILS)}
    (ROOT / "outputs").mkdir(exist_ok=True)
    (ROOT / "outputs" / ("57_t24_shared_ab" + SMOKE_TAG + ".json")).write_text(
        json.dumps({**out, **(extra or {}), "cells": (cells or [])},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    if cells is None or not persist:
        if cells is not None:
            print("[smoke] 不写 results/（smoke 数字不进证据链）")
        return
    keep = ("budget", "mode", "predictor", "seed", "J_per_episode", "J_mean", "served",
            "served_late", "served_total", "alive_steps", "frames", "attempts", "link_q",
            "budget_reserved", "budget_attempted", "budget_wasted", "pred_error",
            "coverage", "elapsed_s")
    slim = [{k: c[k] for k in keep} for c in (cells or [])]
    dest = write_result("57_t24_shared_ab", {**out, **(extra or {}), "cells_J": slim},
                        run_id=run_id, allow_overwrite=allow_overwrite)
    print(f"[evidence] {dest.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    sys.exit(main())
