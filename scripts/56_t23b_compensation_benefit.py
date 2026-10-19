#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""T23-b · 预测补偿收益表 —— 同一调度下 `ZOH / Linear / hybrid / TrueDyn` 的逐用户代价比。

口径来源（★ 全部在看结果之前冻结）
------------------------------------
`04-实验与代码/2026-10-20-T23b-补偿收益表-预注册-v0.1.md` §0–§8。

一句话
------
同一个调度（`age`）、同一份预算、同一个初始信息下，把接收端的零阶保持换成三种预测，
每个用户的任务代价 `J_i` 变小了还是变大了？**不回答**预测能否替代调度。

设计（预注册 §3）
-----------------
    臂 = age；predictor ∈ {zoh, linear, hybrid, truedyn}
    mode ∈ {unpaced, paced}；budget ∈ {100, 200}；seed ∈ {0,1,2}   ⇒ 48 cell
    episodes=30, max_steps=200, absorbed=(25,1), GATE=0.90（全部不动）

★ 主表是 B=200（该档 `t>B` 发帧恒 0，无 T16 记账缺陷）；B=100 只作次要分析且**必须**同报 `t>B` 发帧数。
★ `hybrid` = `M_2`/`M_3` + `Linear` 回退（`predictors.HybridLag`）。**不是**世界模型，是逐阶线性辨识基线。
★ `TrueDyn` 一律标 `diagnostic`（环境确定性 ⇒ 误差精确 0），只作量级上界，**不得**当可达收益。

判据（预注册 §6；本轮是**测量**，不设成功线）
---------------------------------------------
  C1..C8 自检（每条写明能打死什么错）
  F1   B=200 unpaced 下 `hybrid` 的两用户 `r_i`（<1 才叫"两用户都变好"，有一个 >1 就写"交易"）
  F2   `hybrid` vs `Linear` 的匹配对照 —— **唯一可辨识的机制分解**（只差 `age∈{2,3}` 这一支）
  F3   `TrueDyn` 的量级参考（diagnostic）
  F4   paced − unpaced 的差值（只作描述，不判"协同"）
  R0   `ZOH` 12 cell 与 `results/52` 的 `age` 臂**逐 episode J 逐位相同**

用法
----
    D:/devtools/miniconda3/envs/ai-lab/python.exe scripts/56_t23b_compensation_benefit.py
    ... --skip-experiment      # 只跑自检（不写 results/，C5 仍读历史证据）
    ... --smoke                # 2 集 × 1 种子，不写 results/

产物
----
    outputs/56_t23b_compensation_benefit[_smoke].json                完整表（不入库）
    results/runs/<run_id>/56_t23b_compensation_benefit.json          **入库**（results_io 唯一入口）
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
from wmlab.eval.predictors import HybridLag, Linear, MultiLag, TrueDyn, ZOH, make_predictor  # noqa: E402
from wmlab.eval.results_io import latest_result_path, write_result  # noqa: E402

FAILS: list[str] = []

#: age 分层（预注册 §4-3，口径照抄 T23-a-2 的实测分层）
AGE_BUCKETS = (("1", 1, 1), ("2-3", 2, 3), (">=4", 4, 10 ** 6))


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


T12 = _load(ROOT / "scripts" / "42_t12_hetero_four_arms.py", "t12_mod56")
T52 = _load(ROOT / "scripts" / "52_t21_paced_matching_ablation.py", "t52_mod56")

WINDOW = T12.WINDOW
WARMUP_UAV = T12.WARMUP_UAV
BUDGETS = list(T12.BUDGETS)
GATE = T12.GATE
ABSORBED = [T12.ABSORBED_UAV, T12.ABSORBED_CART]
COSTS = [default_uav_cost, T12.cartpole_cost]
ARM = "age"                                   # ★ 预注册 §3：固定臂，只比预测
PREDICTORS = ("zoh", "linear", "hybrid", "truedyn")
MODES = ("unpaced", "paced")


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
    """逐决策点采"预测误差 + age 分档"，**只在内存里聚合**（48 cell 的逐步原始值太大，不入库）。

    ★ 误差定义：`‖est_pre − true_pre‖₂`，`age ≥ 1` 才算预测（`age=0` 直接是送达值，误差结构性为 0）。
      `est_pre` 是 `predict()` 刷新**之后**、"步进之前"的估计；`true_pre` 同刻真值
      （= 上一次步进后的观测，与 `predictors.py` 模块头的时序约定逐位对齐）。
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


def _synth(seed: int = 11, dim: int = 3, act_dim: int = 2, n_steps: int = 200):
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


def _pred_err(p, x_q, acts_q, truth_fn) -> float:
    pred = np.asarray(p.predict(0, age=len(acts_q), actions_since=acts_q), dtype=np.float64)
    return float(np.linalg.norm(pred - truth_fn()) / max(np.linalg.norm(truth_fn()), 1e-300))


# ================================================================ 自检
def selfchecks(n_ep_small: int = 2) -> dict:
    print("\n== 自检 C1–C8（预注册 §5：每条写明能打死什么错）==")
    out: dict = {}
    A, B, c, xs, acts = _synth()
    ad = B.shape[1]

    # ---------------- C1：hybrid 无数据时 ≡ Linear ----------------
    zero_pairs = _pairs_gaps(xs, acts, [2] * 0)
    h = _feed(HybridLag(), xs[0], zero_pairs)
    lin = _feed(Linear(), xs[0], zero_pairs)
    d1, worst = {}, 0.0
    for age in range(0, 6):
        aq = [np.random.default_rng(100 + age).normal(0.0, 1.0, ad) for _ in range(age)]
        o1 = np.asarray(h.predict(0, age=age, actions_since=aq))
        o2 = np.asarray(lin.predict(0, age=age, actions_since=aq))
        d = float(np.max(np.abs(o1 - o2)))
        d1[str(age)] = d
        worst = max(worst, d)
    check("C1 hybrid 无任何样本时，age∈{0..5} 与 Linear **逐位相同**", worst == 0.0,
          f"max|Δ|={worst:.3e}，逐 age={ {k: f'{v:.1e}' for k, v in d1.items()} }")
    out["C1"] = d1

    # ---------------- C2：有了 g=2,3 样本后只在这些 age 上用模型 ----------------
    gaps = [2] * 20 + [3] * 13
    pairs = _pairs_gaps(xs, acts, gaps)
    h2 = _feed(HybridLag(), xs[0], pairs)
    l2 = _feed(Linear(), xs[0], pairs)
    rng = np.random.default_rng(999)
    x_q = np.asarray(pairs[-1][2], dtype=np.float64)
    c2 = {}
    for age in (2, 3):
        aq = [rng.normal(0.0, 2.0, ad) for _ in range(age)]
        # 真值：从 x_q 出发按 A/B/c 前推 age 步（与合成系统同一动力学）
        x_t = x_q.copy()
        for a in aq:
            x_t = A @ x_t + B @ a + c
        e_h = _pred_err(h2, x_q, aq, lambda x_t=x_t: x_t)
        e_l = _pred_err(l2, x_q, aq, lambda x_t=x_t: x_t)
        c2[str(age)] = {"rel_hybrid": e_h, "rel_linear": e_l,
                        "ratio": (e_l / e_h if e_h > 0 else float("inf"))}
    c2["other_ages_identical"] = {}
    for age in (1, 4, 5):
        aq = [rng.normal(0.0, 1.0, ad) for _ in range(age)]
        o1 = np.asarray(h2.predict(0, age=age, actions_since=aq))
        o2 = np.asarray(l2.predict(0, age=age, actions_since=aq))
        c2["other_ages_identical"][str(age)] = float(np.max(np.abs(o1 - o2)))
    ok2 = (all(v["rel_hybrid"] < 1e-3 and v["ratio"] >= 10.0 for k, v in c2.items()
               if k in ("2", "3"))
           and all(v == 0.0 for v in c2["other_ages_identical"].values()))
    check("C2 age=2,3 上 hybrid 相对误差 <1e-3 且比 Linear 好 ≥10×；age=1,4,5 仍逐位同 Linear",
          ok2, json.dumps(c2, ensure_ascii=False))
    out["C2"] = c2

    # ---------------- C3：不该用的滞后阶只计数、不入拟合 ----------------
    g_mix = [1, 4, 2, 3, 5, 2, 1, 4] + [2] * 12 + [3] * 12
    h3 = _feed(HybridLag(), xs[0], _pairs_gaps(xs, acts, g_mix))
    fitted = sorted(int(g) for g in h3._fit.get(0, {}))
    skipped = {str(k): int(v) for k, v in h3.describe()["users"]["0"].get("skipped_gap", {}).items()}
    ok3 = set(fitted) <= set(HybridLag.usable_gaps) and all(k in skipped for k in ("1", "4", "5"))
    check("C3 拟合的滞后阶 ⊆ {2,3}；gap∈{1,4,5} 的样本只进 `skipped_gap` 计数", ok3,
          f"fitted={fitted} skipped={skipped}")
    out["C3"] = {"fitted_gaps": fitted, "skipped_gap": skipped}

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

    # ---------------- C6：hybrid 的 used_model 年龄集合 ⊆ {2,3} ----------------
    hp = make_predictor("hybrid")
    _run(200, "unpaced", 0, n_ep_small, predictor=hp)
    hd = hp.describe()
    used = set()
    for u in hd["cum_stats"].values():
        used |= {int(k) for k, v in u["used_model"].items() if int(v) > 0}
    ok6 = used <= {0, 2, 3}
    check("C6 实跑中 hybrid 的 used_model 年龄 ⊆ {0,2,3}（0=送达值，不算预测）", ok6, f"used={sorted(used)}")
    out["C6"] = {"used_model_ages": sorted(used),
                 "usable_ages": {k: v.get("usable_ages") for k, v in hd["users"].items()}}

    # ---------------- C8：覆盖率计数与 describe() 逐档一致（#48：不用调用后计数）----------------
    hp2 = make_predictor("hybrid")
    pr2 = _ErrProbe()
    _run(200, "unpaced", 0, n_ep_small, predictor=hp2, probe=pr2)
    d2 = hp2.describe()["cum_stats"]          # ★ 必须是**跨集累计**：`_pred_stats` 逐集重置（同区间纪律）
    c8 = {}
    ok8 = True
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
    """`ZOH` 必须与 `results/52` 的 `age` 臂**逐 episode J 逐位相同**（ZOH = 现状 = 本轮基线）。

    ★ 同一区间纪律（#49/#52）：历史是 30 集，**集数不同就不可比** ⇒ 非 30 集时打印 SKIP、不计 FAIL。
      `cells=None` 时现跑（`--skip-experiment` 用），传入主实验 cell 时直接复用（不重复计算）。
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
    cov = None
    if pred_name == "hybrid":
        d = pred.describe()
        cum = d["cum_stats"]                    # ★ 跨集累计（`_pred_stats` 逐集重置，不能用）
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
                           # ★ 作用域写进键名（#49 同族：`_fit` 逐集重置，这两个只是**最后一集**的）
                           "last_episode_usable_ages": d["users"][str(i)].get("usable_ages"),
                           "last_episode_ages": d["users"][str(i)].get("ages"),
                           "skipped_gap_cum": d["users"][str(i)].get("skipped_gap")}
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
    print(f"[56] T23-b 预测补偿收益表｜N=2｜窗口 {WINDOW}｜臂={ARM}｜预算 {budgets}｜"
          f"seeds={seeds}｜episodes={n_ep}｜predictors={PREDICTORS}｜modes={MODES}")
    sc = selfchecks()

    if args.skip_experiment:
        sc["C5"] = c5_zoh_regression(n_ep, seeds, budgets, None)   # 现跑 12 个 ZOH cell 再比
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
                "r_sd": (np.nanstd(rr, axis=0, ddof=1).tolist() if len(rows) > 1
                         else [0.0, 0.0]),
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
            # ★ 机制对照：hybrid / linear（唯一可辨识的分解，预注册 §4-2）
            per["_hybrid_vs_linear"] = ratio_table(budget, mode, "hybrid", "linear", seeds)
            summary[str(budget)][mode] = per

    print("\n== 汇总（r_i 相对**同档同模式同种子**的 ZOH）==")
    for budget in budgets:
        for mode in MODES:
            for pname in PREDICTORS:
                d = summary[str(budget)][mode][pname]
                if pname == "zoh":
                    print(f"  B={budget:3d} {mode:7s} zoh      "
                          f"J=({d['J_uav_mean']:.3f},{d['J_cart_mean']:.3f})（基线）"
                          f" late={d['late_total']:.0f} "
                          f"alive_cart={d['alive_cart']:.1f} {d['elapsed_s_mean']:.1f}s")
                    continue
                v = d["vs_zoh"]
                print(f"  B={budget:3d} {mode:7s} {pname:8s} "
                      f"geo={v['geo_mean']:.3f}±{v['geo_sd']:.3f} "
                      f"r=({v['r_mean'][0]:.3f}±{v['r_sd'][0]:.3f},"
                      f"{v['r_mean'][1]:.3f}±{v['r_sd'][1]:.3f}) "
                      f"J=({d['J_uav_mean']:.3f},{d['J_cart_mean']:.3f}) "
                      f"late={d['late_total']:.0f} {d['elapsed_s_mean']:.1f}s")
            hl = summary[str(budget)][mode]["_hybrid_vs_linear"]
            print(f"  B={budget:3d} {mode:7s} hybrid/linear "
                  f"geo={hl['geo_mean']:.3f} r=({hl['r_mean'][0]:.3f},{hl['r_mean'][1]:.3f})")

    # ---------- 记账判据（C7）----------
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

    # ---------- C5：ZOH 与 results/52 的 age 臂逐位回归 ----------
    c5 = c5_zoh_regression(n_ep, seeds, budgets, cells)

    # ---------- F1：B=200 unpaced 下 hybrid 两用户是否都变好 ----------
    print("\n== F1（主：B=200 unpaced，hybrid 的 r_i 相对 ZOH）==")
    f1 = {}
    for budget in budgets:
        b = str(budget)
        v = summary[b]["unpaced"]["hybrid"]["vs_zoh"]
        s = summary[b]["unpaced"]["hybrid"]
        z = summary[b]["unpaced"]["zoh"]
        f1[b] = {"r_mean": v["r_mean"], "r_sd": v["r_sd"], "geo": v["geo_mean"],
                 "geo_sd": v["geo_sd"],
                 "both_below_1_mean": bool(all(x < 1.0 for x in v["r_mean"])),
                 "any_above_1_mean": bool(any(x > 1.0 for x in v["r_mean"])),
                 "seeds_both_below_1": v["n_seeds_both_below_1"],
                 "seeds_any_above_1": v["n_seeds_any_above_1"],
                 "n_seeds": len(seeds),
                 "late": s["late_total"],
                 # ★ 帧数反馈：存活变长 ⇒ `age` 调度自然给它更多帧 ⇒ 收益不能全归预测（R17 同源）
                 "served_zoh": [z["served_uav"], z["served_cart"]],
                 "served_pred": [s["served_uav"], s["served_cart"]],
                 "alive_zoh": [z["alive_uav"], z["alive_cart"]],
                 "alive_pred": [s["alive_uav"], s["alive_cart"]]}
        print(f"  B={budget:3d} unpaced hybrid r=({v['r_mean'][0]:.3f},{v['r_mean'][1]:.3f}) "
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
    print(f"  ⇒ {f1['verdict']}（只写「两用户都变好」或「交易」，见 #44）")

    # ---------- F2：机制对照 hybrid vs Linear ----------
    print("\n== F2（机制：hybrid/Linear，只差 age∈{2,3} 这一支）==")
    f2 = {}
    for budget in budgets:
        for mode in MODES:
            hl = summary[str(budget)][mode]["_hybrid_vs_linear"]
            a = summary[str(budget)][mode]["hybrid"]
            b_ = summary[str(budget)][mode]["linear"]
            f2[f"{budget}/{mode}"] = {"r_mean": hl["r_mean"], "geo": hl["geo_mean"],
                                      "any_above_1": bool(any(x > 1.0 for x in hl["r_mean"])),
                                      "served_linear": [b_["served_uav"], b_["served_cart"]],
                                      "served_hybrid": [a["served_uav"], a["served_cart"]]}
            print(f"  B={budget:3d} {mode:7s} r=({hl['r_mean'][0]:.3f},{hl['r_mean'][1]:.3f}) "
                  f"geo={hl['geo_mean']:.3f} served(lin→hyb)="
                  f"{b_['served_uav']:.0f}/{b_['served_cart']:.0f}→{a['served_uav']:.0f}/{a['served_cart']:.0f}")
    r200 = f2["200/unpaced"]["r_mean"]
    # ★ 预注册 §6-F2 的否定条件是「**两用户都不改善**（r_i ≥ 1）」，不是「任一用户 ≥ 1」。
    #   首版写成 any(...) ⇒ 把「UAV 变差 + CartPole 变好」误判成否定条件命中（#50 同族，本轮第 2 次）。
    f2["negative_condition_hit"] = bool(all(x >= 1.0 for x in r200))
    f2["mixed_trade"] = bool(any(x >= 1.0 for x in r200) and any(x < 1.0 for x in r200))
    if f2["negative_condition_hit"]:
        f2["note"] = ("否定条件命中：`age=2–3` 这一支两用户都不改善 ⇒ 记「该层无补偿收益」，"
                      "不再加同类预测器，转「共享 A,B + 滞后作特征」（另写预注册）")
    elif f2["mixed_trade"]:
        f2["note"] = ("否定条件**未**命中，但也不是两用户同向改善：`age=2–3` 这一支对两个用户方向相反"
                      "（见 r_i）⇒ 只能记「交易」，且必须同时报两侧的 served 变化（存活变长会通过"
                      "`age` 调度拿到更多帧，收益不全来自预测）")
    else:
        f2["note"] = "否定条件未命中且两用户同向改善（仍不等于已证收益：3 种子只够筛选）"
    print(f"  ⇒ negative_hit={f2['negative_condition_hit']} mixed_trade={f2['mixed_trade']}")

    # ---------- F3：TrueDyn 诊断上界 ----------
    print("\n== F3（诊断：TrueDyn/ZOH 的量级参考，不得当可达收益）==")
    f3 = {str(b): {"r_mean": summary[str(b)]["unpaced"]["truedyn"]["vs_zoh"]["r_mean"],
                   "geo": summary[str(b)]["unpaced"]["truedyn"]["vs_zoh"]["geo_mean"]}
          for b in budgets}
    for b in budgets:
        print(f"  B={b:3d} unpaced truedyn r=({f3[str(b)]['r_mean'][0]:.3f},"
              f"{f3[str(b)]['r_mean'][1]:.3f}) geo={f3[str(b)]['geo']:.3f} [diagnostic]")

    # ---------- F4：配速交互（描述）----------
    print("\n== F4（描述：paced − unpaced 的 r_i 差，不判协同）==")
    f4 = {}
    for budget in budgets:
        for pname in ("linear", "hybrid", "truedyn"):
            u = summary[str(budget)]["unpaced"][pname]["vs_zoh"]["r_mean"]
            p = summary[str(budget)]["paced"][pname]["vs_zoh"]["r_mean"]
            f4[f"{budget}/{pname}"] = {"d_r": [float(p[0] - u[0]), float(p[1] - u[1])]}
            print(f"  B={budget:3d} {pname:8s} Δr=(paced−unpaced)=({p[0]-u[0]:+.3f},{p[1]-u[1]:+.3f})")

    # ---------- ERR：预测误差分层（★ 预测质量，不是收益；与 F1/F2 的排序未必同向）----------
    print("\n== ERR（预测误差中位，B=200 unpaced，按年龄三档；标「预测质量」不标收益）==")
    err = {}
    for budget in (200,):
        for pname in PREDICTORS:
            rows = [c for c in cells if c["budget"] == budget and c["mode"] == "unpaced"
                    and c["predictor"] == pname and c["seed"] == seeds[0]]
            if not rows:
                continue
            c = rows[0]
            err[f"{budget}/{pname}"] = {u: {t: c["pred_error"][u]["buckets"][t]
                                            for t, _lo, _hi in AGE_BUCKETS}
                                        for u in ("0", "1")}
            print(f"  B={budget} {pname:8s} "
                  f"UAV(age2-3)={c['pred_error']['0']['buckets']['2-3'].get('median', float('nan')):.4f} "
                  f"Cart(age2-3)={c['pred_error']['1']['buckets']['2-3'].get('median', float('nan')):.4f}")

    # ---------- COV：混合覆盖率（★ 预注册引用的 20% 门在**本轮实跑**里是否成立）----------
    print("\n== COV（hybrid 的模型启用率，30 集累计；门=20%，只在 age≥1 决策点上算）==")
    cov = {}
    for budget in budgets:
        for mode in MODES:
            rows = [c for c in cells if c["budget"] == budget and c["mode"] == mode
                    and c["predictor"] == "hybrid"]
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
                b23[tag] = {"n": int(n_), "used": int(u_),
                            "frac": (float(u_) / n_ if n_ else None)}
            cov[f"{budget}/{mode}"] = {"n_decisions_age_ge1": int(tot), "n_used": int(used),
                                       "frac": (float(used) / tot if tot else None),
                                       "gate_20pct_met": bool(tot and used / tot >= 0.20),
                                       "by_age_bucket": b23}
            print(f"  B={budget:3d} {mode:7s} 合并 {used}/{tot} = "
                  f"{100.0 * used / tot if tot else float('nan'):.1f}% "
                  f"（门 20% {'过' if cov[f'{budget}/{mode}']['gate_20pct_met'] else '未过'}）"
                  f" age2-3={100.0 * (b23['2-3']['frac'] or 0):.1f}%")

    _dump({"selfchecks": sc}, cells,
          {"summary": summary, "F1": f1, "F2": f2, "F3": f3, "F4": f4, "ERR": err, "COV": cov,
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
        "hybrid_definition": "age∈{2,3} 用 M_g（逐阶线性辨识），其余 age 落 Linear（同一函数）",
        "aggregation": "r_i = J_i(pred)/J_i(ZOH)，同 budget/mode/seed 配对；geo=sqrt(r0*r1)",
        "prereg": "04-实验与代码/2026-10-20-T23b-补偿收益表-预注册-v0.1.md",
    }
    out = {"config": cfg, "selfchecks": payload.get("selfchecks"),
           "checks_failed": list(FAILS)}
    (ROOT / "outputs").mkdir(exist_ok=True)
    (ROOT / "outputs" / ("56_t23b_compensation_benefit" + SMOKE_TAG + ".json")).write_text(
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
    dest = write_result("56_t23b_compensation_benefit", {**out, **(extra or {}), "cells_J": slim},
                        run_id=run_id, allow_overwrite=allow_overwrite)
    print(f"[evidence] {dest.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    sys.exit(main())
