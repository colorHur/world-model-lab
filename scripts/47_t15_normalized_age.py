#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""T15 · 归一化年龄（`age_i/D_i`）—— 判 H4（**机制验证，不是算法主张**）。

口径来源（★ 全部在看结果之前冻结）
------------------------------------
`04-实验与代码/2026-10-19-T15-归一化年龄-预注册-v0.1.md` §2–§7。
前置检索：`03-文献与竞品/2026-10-19-T15-deadline与虚拟队列占位检索-v0.1.md`
⇒ 该检索已确认 `age/D` 排序与硬截止 AoI 调度**已被占位**，故本脚本只检验机制，
即使通过也**不得**表述为"提出了新调度算法"。

系统（与 T12/T13/T14 逐位同一套）
---------------------------------
    用户 0：`uav_track`，PD，warmup 300，窗口 200，代价 = 距离 [m]，D_0 = 53（实测）
    用户 1：`CartPole-v1`，LQR，无 warmup，代价 = 0/1，        D_1 = 10（实测）
    csi：AR(1)（rho=0.9, sd=0.10），均值 (0.90, 0.55)，clip [0.10,1.0]
    q = SINRSuccess 口径 A
★ 常量与生成器**直接 import `scripts/42`**（不复制），保证与 T12–T14 同源。

臂：`rr` / `age` / `na`（D=(53,10)）/ `na_swap`（D=(10,53)，**仅诊断，不进判据**）

判据（冻结）
------------
  S1..S7  自检（预注册 §5；S2/S3 是解析值，不是经验预期）
  P1/P2/P3 记账（同 42/45）
  R0' 同进程重跑 `rr`/`age`，逐 episode `J` 与 `results/42_*.json` **逐位相同**
  H4  geo(na) ≤ 0.90 两档 且 两档 ≤ geo(age) ⇒ SUPPORTED；否则 PARTIAL/FALSIFIED
      ★ 机制闸门：CartPole 送达间隔中位 ≥6 步 或 ≥10 步占比 >10% ⇒ 最高只能 PARTIAL

用法
----
    D:/devtools/miniconda3/envs/ai-lab/python.exe scripts/47_t15_normalized_age.py
    ... --skip-experiment      # 只跑自检
    ... --smoke                # 2 集 × 1 种子 × 2 档

产物
----
    outputs/47_t15_normalized_age.json   完整表（不入库）
    results/47_t15_normalized_age.json   逐 episode J + 间隔分布 + 判据（**入库**）
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

from wmlab.eval.allocators import MaxAge, NormalizedAge, RoundRobin  # noqa: E402
from wmlab.eval.results_io import latest_result_path, write_result  # noqa: E402  ← 入库唯一入口（P0-3）
from wmlab.eval.multiuser import (  # noqa: E402
    Allocation, MissingStateError, SINRSuccess, UserSlot, View, default_uav_cost,
    delivery_rate, run_multiuser, window_cost,
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

# ★ D 的唯一来源：`results/46` 的闭环实测（断服→任务终止步数）。
#   写死在这里（而不是运行时读 json），因为**读数本身就是口径的一部分**；
#   若 46 的结论变了，必须改这里并升版预注册，不许静默跟随。
D_MEASURED = (53.0, 10.0)       # (UAV, CartPole)
D_SWAPPED = (10.0, 53.0)        # na_swap 诊断臂

FAILS: list[str] = []


def check(tag: str, ok: bool, detail: str) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {tag}: {detail}")
    if not ok:
        FAILS.append(f"{tag}: {detail}")


# ================================================================ 自检
def _mk_view(ages, alive=None, budget_left=10):
    n = len(ages)
    alive = [True] * n if alive is None else list(alive)
    slots = tuple(UserSlot(idx=i, alive=bool(alive[i]), age=int(ages[i]),
                           last_obs=np.zeros(1, dtype=np.float32), n_frames=0,
                           n_attempts=0, csi=1.0, cost_est=0.0) for i in range(n))
    return View(t=1, slots=slots, budget_left=int(budget_left), budget_total=200,
                privileged=None)


def selfchecks() -> dict:
    print("\n== 自检 S1–S7（预注册 §5）==")
    rng = np.random.default_rng(0)
    out: dict = {}

    # ---- S2/S3：解析值（预注册 §2 公式直接算）----
    na = NormalizedAge(deadlines=D_MEASURED)
    a2 = na(_mk_view((4, 1)), rng)                 # 4/53=0.0755 < 1/10=0.1000 ⇒ 选 1
    a3 = na(_mk_view((6, 1)), rng)                 # 6/53=0.1132 > 0.1000      ⇒ 选 0
    i2, i3 = int(np.flatnonzero(a2.served)[0]), int(np.flatnonzero(a3.served)[0])
    age_2 = int(np.flatnonzero(MaxAge()(_mk_view((4, 1)), rng).served)[0])
    check("S2 age=(4,1) D=(53,10) ⇒ 选 1（0.0755<0.1000）",
          i2 == 1, f"na 选 {i2}；对照：age 会选 {age_2}（两者**必须不同**）")
    out["S2"] = {"chosen": i2, "age_chosen": age_2,
                 "scores": [4 / 53.0, 1 / 10.0]}
    check("S3 age=(6,1) D=(53,10) ⇒ 选 0（0.1132>0.1000）",
          i3 == 0, f"na 选 {i3}（无条件偏向小 D 也是错）")
    out["S3"] = {"chosen": i3, "scores": [6 / 53.0, 1 / 10.0]}

    # ---- S2b：平局与 alive 过滤 ----
    # age=(0,0) 全平 ⇒ 同分取 idx 小（0）
    tie = int(np.flatnonzero(na(_mk_view((0, 0)), rng).served)[0])
    # user0 已死、user1 的 score 更低也必须选 1
    dead = int(np.flatnonzero(na(_mk_view((50, 1), alive=[False, True]), rng).served)[0])
    check("S2b 平局取 idx 小 + 死人必被过滤", tie == 0 and dead == 1,
          f"平局选 {tie}；user0 死时选 {dead}")
    out["S2b"] = {"tie": tie, "dead_filter": dead}

    # ---- S4：非法 D ----
    bads = []
    for kw in ({"deadlines": (0.0, 10.0)}, {"deadlines": (-1.0, 10.0)},
               {"deadlines": (float("nan"), 10.0)}, {"deadlines": (53.0, float("inf"))},
               {"deadlines": ()}):
        try:
            NormalizedAge(**kw)
            bads.append(f"{kw} 未抛")
        except ValueError:
            pass
    # 长度不符：构造期不知道 N ⇒ 必须在 __call__ 抛
    try:
        NormalizedAge(deadlines=(53.0,))( _mk_view((1, 1)), rng)
        bads.append("长度不符未抛")
    except ValueError:
        pass
    check("S4 非法 deadlines（0/负/NaN/inf/空/长度不符）全部抛错", not bads,
          "全部抛 ValueError" if not bads else f"漏网 {bads}")
    out["S4"] = {"leaks": bads}

    # ---- S5：预算 0 ⇒ 停发 ----
    z = na(_mk_view((3, 3), budget_left=0), rng)
    check("S5 预算 0 ⇒ 停发（served 全 False）", not bool(z.served.any()),
          f"served={z.served.tolist()} reason={z.reason}")
    out["S5"] = {"served": z.served.tolist(), "reason": z.reason}

    # ---- S1/S7 / S6：真实运行 ----
    K = T12.load_lqr_gain()

    def run2(alloc, budget, seed, n_ep=2):
        envs = [T12.make_env("uav", seed=seed, max_steps=WARMUP_UAV + WINDOW),
                T12.make_env("CartPole-v1", seed=seed)]
        r = run_multiuser(envs, [T12.PDRelativeController(), T12.CartPoleLQR(K)],
                          alloc, SINRSuccess(T12.Q_REF), n_episodes=n_ep, seed=seed,
                          max_steps=WINDOW, warmup_steps=[WARMUP_UAV, 0],
                          budget_total=budget, csi=T12.make_csi(seed, n_ep), cost_fn=COSTS)
        for e in envs:
            e.close()
        return r

    r_eq = run2(NormalizedAge(deadlines=(5.0, 5.0), name="na_eq"), 100, 0)
    r_age = run2(MaxAge(), 100, 0)
    seq_eq = [x.served_idx for x in r_eq["records"]]
    seq_age = [x.served_idx for x in r_age["records"]]
    same_seq = seq_eq == seq_age
    check("S1 D 全等(5,5) ⇒ na 与 age 的 served_idx 序列逐位相同",
          same_seq, f"{len(seq_eq)} 步，首个差异位 "
                    f"{next((k for k, (a, b) in enumerate(zip(seq_eq, seq_age)) if a != b), None)}")
    J_eq = np.asarray(window_cost(r_eq["records"], 2, window=WINDOW, absorbed=ABSORBED,
                                 n_episodes=2))
    J_age = np.asarray(window_cost(r_age["records"], 2, window=WINDOW, absorbed=ABSORBED,
                                  n_episodes=2))
    check("S7 D 全等 ⇒ 逐 episode J 也逐位相同（array_equal）",
          np.array_equal(J_eq, J_age),
          f"max|ΔJ|={float(np.max(np.abs(J_eq - J_age))):.3e}")
    out["S1_S7"] = {"n_steps": len(seq_eq), "same_seq": same_seq,
                    "J_na_eq": J_eq.tolist(), "J_age": J_age.tolist()}

    r_me = run2(NormalizedAge(deadlines=D_MEASURED), 100, 0)
    seq_me = [x.served_idx for x in r_me["records"]]
    n_diff = sum(1 for a, b in zip(seq_me, seq_age) if a != b)
    check("S6 D=(53,10) ⇒ na 与 age 的选择序列**必须**不同",
          n_diff > 0, f"差异 {n_diff}/{len(seq_me)} 步")
    out["S6"] = {"n_diff": n_diff, "n_steps": len(seq_me),
                 "share_na": [sum(1 for x in r_me["records"] if x.attempted_idx == i)
                              for i in range(2)],
                 "share_age": [sum(1 for x in r_age["records"] if x.attempted_idx == i)
                               for i in range(2)]}
    return out


# ================================================================ 送达间隔
def interval_stats(records, i: int) -> dict:
    """用户 i 的送达间隔（同 episode 内相邻两次送达的步差）。"""
    gaps: list[int] = []
    eps = sorted({r.episode for r in records})
    n_deliv = 0
    for e in eps:
        ts = [r.t for r in records if r.episode == e and r.delivered_idx == i]
        n_deliv += len(ts)
        gaps += [b - a for a, b in zip(ts, ts[1:])]
    if not gaps:
        return {"n": 0, "n_deliv": n_deliv, "episodes": len(eps)}
    g = np.asarray(gaps, dtype=float)
    return {"n": int(g.size), "n_deliv": n_deliv, "episodes": len(eps),
            "median": float(np.median(g)), "mean": float(g.mean()),
            "p90": float(np.percentile(g, 90)), "max": float(g.max()),
            "frac_ge_6": float(np.mean(g >= 6)), "frac_ge_10": float(np.mean(g >= 10))}


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
    arms = ["rr", "age", "na", "na_swap"]

    t_all = time.time()
    print(f"[47] T15 归一化年龄｜N=2 异质｜窗口 {WINDOW}｜预算 {BUDGETS}｜seeds={seeds}"
          f"｜episodes={n_ep}｜D_measured={D_MEASURED}｜arm={arms}")
    sc = selfchecks()

    if args.skip_experiment:
        _dump({"selfchecks": sc, "experiment": None}, None, None, None, n_ep, seeds)
        print(f"\n{'ALL CHECKS PASS' if not FAILS else 'FAILED: ' + '; '.join(FAILS)}")
        return 0 if not FAILS else 2

    K = T12.load_lqr_gain()
    q_model = SINRSuccess(T12.Q_REF)
    cells: list[dict] = []
    for budget in BUDGETS:
        for arm in arms:
            for seed in seeds:
                t0 = time.time()
                envs = [T12.make_env("uav", seed=seed, max_steps=WARMUP_UAV + WINDOW),
                        T12.make_env("CartPole-v1", seed=seed)]
                ctrls = [T12.PDRelativeController(), T12.CartPoleLQR(K)]
                tr: list | None = None
                if arm == "rr":
                    alloc = RoundRobin()
                elif arm == "age":
                    alloc = MaxAge()
                elif arm == "na":
                    tr = []
                    alloc = NormalizedAge(deadlines=D_MEASURED, name="na", trace=tr)
                else:
                    tr = []
                    alloc = NormalizedAge(deadlines=D_SWAPPED, name="na_swap", trace=tr)
                csi = T12.make_csi(seed, n_ep)
                # ★ 全部非诊断：`na` 不需要特权信息（这正是它相对 lookahead 的可部署性优势）
                r = run_multiuser(envs, ctrls, alloc, q_model, n_episodes=n_ep, seed=seed,
                                  max_steps=WINDOW, warmup_steps=[WARMUP_UAV, 0],
                                  budget_total=budget, csi=csi, cost_fn=COSTS,
                                  diagnostic=False, counterfactual=False)
                dr = delivery_rate(r["records"], 2)
                J = np.asarray(window_cost(r["records"], 2, window=WINDOW,
                                           absorbed=ABSORBED, n_episodes=n_ep))
                cell = {
                    "budget": budget, "arm": arm, "seed": seed,
                    "J_per_episode": J.tolist(),
                    "J_mean": J.mean(axis=0).tolist(),
                    "alive_steps": dr["alive_steps"],
                    "frames": dr["frames"], "attempts": dr["attempts"],
                    "share_attempts": [dr["attempts"][i] / max(sum(dr["attempts"]), 1)
                                       for i in range(2)],
                    "p_eff": dr["p_eff"], "link_q": dr["link_q"],
                    "budget_reserved": dr["budget_reserved"],
                    "budget_attempted": dr["budget_attempted"],
                    "budget_wasted": dr["budget_wasted"],
                    "budget_lost": dr["budget_lost"],
                    "gap_uav": interval_stats(r["records"], 0),
                    "gap_cart": interval_stats(r["records"], 1),
                    "elapsed_s": round(time.time() - t0, 2),
                }
                cells.append(cell)
                print(f"  B={budget:3d} {arm:8s} seed={seed} "
                      f"J=({cell['J_mean'][0]:.4f},{cell['J_mean'][1]:.4f}) "
                      f"share=({cell['share_attempts'][0]:.3f},{cell['share_attempts'][1]:.3f}) "
                      f"cart_gap_med={cell['gap_cart'].get('median', float('nan')):.1f} "
                      f"ge10={cell['gap_cart'].get('frac_ge_10', float('nan')):.3f} "
                      f"{cell['elapsed_s']}s")
                for e in envs:
                    e.close()

    # ---------- 汇总：r_i / geo（同种子配对 rr）----------
    summary: dict = {}
    for budget in BUDGETS:
        sub = [c for c in cells if c["budget"] == budget]
        rr = {c["seed"]: c for c in sub if c["arm"] == "rr"}
        per_arm = {}
        for arm in arms:
            a = {c["seed"]: c for c in sub if c["arm"] == arm}
            rs = []
            for seed in seeds:
                j = np.asarray(a[seed]["J_mean"], dtype=float)
                jr = np.asarray(rr[seed]["J_mean"], dtype=float)
                deg = bool(np.any(jr <= 1e-12))
                r = (j / jr) if not deg else np.array([np.nan, np.nan])
                rs.append({"seed": seed, "r": r.tolist(),
                           "geo": float(np.sqrt(r[0] * r[1])) if not deg else float("nan")})
            geos = [x["geo"] for x in rs if np.isfinite(x["geo"])]
            per_arm[arm] = {
                "per_seed": rs,
                "r_mean": np.nanmean([[v for v in x["r"]] for x in rs], axis=0).tolist(),
                "geo_mean": float(np.mean(geos)) if geos else float("nan"),
                "geo_sd": float(np.std(geos, ddof=1)) if len(geos) > 1 else 0.0,
                "share_attempts_mean": np.mean([[c["share_attempts"][i] for c in sub
                                                 if c["arm"] == arm] for i in range(2)],
                                               axis=1).tolist(),
                "gap_cart_median_mean": float(np.mean(
                    [c["gap_cart"].get("median", np.nan) for c in sub if c["arm"] == arm])),
                "gap_cart_frac_ge10_mean": float(np.mean(
                    [c["gap_cart"].get("frac_ge_10", np.nan) for c in sub if c["arm"] == arm])),
                "gap_uav_median_mean": float(np.mean(
                    [c["gap_uav"].get("median", np.nan) for c in sub if c["arm"] == arm])),
            }
        summary[budget] = per_arm

    print("\n== 汇总（r_i 相对 rr，同种子配对）==")
    for budget in BUDGETS:
        for arm in arms:
            d = summary[budget][arm]
            print(f"  B={budget:3d} {arm:8s} r=({d['r_mean'][0]:.3f},{d['r_mean'][1]:.3f}) "
                  f"geo={d['geo_mean']:.3f}±{d['geo_sd']:.3f} "
                  f"cart_gap_med={d['gap_cart_median_mean']:.1f} "
                  f"ge10={d['gap_cart_frac_ge10_mean']:.3f}")

    # ---------- 记账判据 ----------
    print("\n== 记账判据（P1–P3）==")
    for budget in BUDGETS:
        sub = [c for c in cells if c["budget"] == budget]
        check(f"P1[{budget}] reserved == n_ep·B",
              all(c["budget_reserved"] == n_ep * budget for c in sub),
              f"{len(sub)} cells")
        check(f"P2[{budget}] reserved == attempted + wasted",
              all(c["budget_reserved"] == c["budget_attempted"] + c["budget_wasted"]
                  for c in sub), "三分账闭合")
        ok3, bad = True, []
        for c in sub:
            for i in range(2):
                if not c["attempts"][i]:
                    continue
                if not (np.isfinite(c["link_q"][i]) and 0.0 <= c["link_q"][i] <= 1.0):
                    ok3 = False
                    bad.append(f"{c['arm']}/s{c['seed']}/u{i}: link_q={c['link_q'][i]}")
                if c["frames"][i] > c["attempts"][i]:
                    ok3 = False
                    bad.append(f"{c['arm']}/s{c['seed']}/u{i}: frames>attempts")
        check(f"P3[{budget}] 0 ≤ link_q ≤ 1 且 frames ≤ attempts", ok3, bad[:3] or "全 cell")

    # ---------- R0' 与 42 逐位回归 ----------
    print("\n== R0' 与 results/42 的 rr/age 逐位回归 ==")
    # ★ P0-3（2026-10-19）：读"当前版本"（runs/ 最新优先，退回 v1 冻结件），并打印**实际**路径。
    #   硬编码 results/42_... 会在 P0-3 之后永远核对旧的冻结文件（自证伪 #46 同型）。
    ref_path = latest_result_path("42_t12_four_arms_summary")
    print(f"  [ref] {ref_path.relative_to(ROOT).as_posix() if ref_path else '<缺失>'}")
    r0_ok, r0_detail = True, "未找到 42 结果（跳过，不算通过）"
    if ref_path is not None and ref_path.exists():
        ref = json.loads(ref_path.read_text(encoding="utf-8"))
        ref_map = {(c["budget"], c["arm"], c["seed"]): c["J_per_episode"]
                   for c in ref.get("cells_J", [])}
        diffs, skipped, compared = [], 0, 0
        for c in cells:
            if c["arm"] not in ("rr", "age"):
                continue
            key = (c["budget"], c["arm"], c["seed"])
            if key not in ref_map:
                continue
            a = np.asarray(c["J_per_episode"], dtype=float)
            b = np.asarray(ref_map[key], dtype=float)
            if a.shape != b.shape:
                skipped += 1
                continue
            compared += 1
            if not np.array_equal(a, b):
                r0_ok = False
                diffs.append(f"{key} max|Δ|={np.nanmax(np.abs(a - b)):.3e}")
        if compared == 0:
            print(f"  [SKIP] R0'：0 个可比 cell（skipped={skipped}）")
        else:
            check("R0' rr/age 与 T12 结果逐位相同", r0_ok,
                  ("逐位相同" if r0_ok else "；".join(diffs[:4])) +
                  f"（{compared} cells 比较，{skipped} 跳过）")

    # ---------- H4 ----------
    print("\n== H4（归一化年龄）==")
    h4: dict = {"gate": GATE, "D_measured": list(D_MEASURED), "geo": {}, "r": {},
                "delta_vs_age": {}, "mechanism": {}, "verdict": None, "note": ""}
    for budget in BUDGETS:
        for arm in ("age", "na", "na_swap"):
            h4["geo"].setdefault(arm, {})[str(budget)] = summary[budget][arm]["geo_mean"]
            h4["r"].setdefault(arm, {})[str(budget)] = summary[budget][arm]["r_mean"]
    h4["delta_vs_age"] = {str(b): {a: float(summary[b][a]["geo_mean"]
                                            - summary[b]["age"]["geo_mean"])
                                   for a in ("na", "na_swap")} for b in BUDGETS}
    # 机制闸门：CartPole 送达间隔
    mech = {}
    for budget in BUDGETS:
        m = summary[budget]["na"]
        mech[str(budget)] = {
            "cart_gap_median": m["gap_cart_median_mean"],
            "cart_gap_frac_ge10": m["gap_cart_frac_ge10_mean"],
            "age_cart_gap_median": summary[budget]["age"]["gap_cart_median_mean"],
            "ok": bool(m["gap_cart_median_mean"] < 6.0 and m["gap_cart_frac_ge10_mean"] <= 0.10),
        }
    h4["mechanism"] = mech
    for budget in BUDGETS:
        print(f"  B={budget:3d}  na geo={h4['geo']['na'][str(budget)]:.3f}"
              f"  Δ(age)={h4['delta_vs_age'][str(budget)]['na']:+.3f}"
              f"  |  na_swap geo={h4['geo']['na_swap'][str(budget)]:.3f}"
              f"  Δ(age)={h4['delta_vs_age'][str(budget)]['na_swap']:+.3f}")
        print(f"        CartPole 送达间隔 中位 {mech[str(budget)]['cart_gap_median']:.1f} 步"
              f"（age 为 {mech[str(budget)]['age_cart_gap_median']:.1f}）"
              f"  ≥10 步占比 {mech[str(budget)]['cart_gap_frac_ge10']:.3f}")

    gate_ok = all(np.isfinite(h4["geo"]["na"][str(b)]) and h4["geo"]["na"][str(b)] <= GATE
                  for b in BUDGETS)
    better_than_age = all(h4["delta_vs_age"][str(b)]["na"] <= 0 for b in BUDGETS)
    mech_ok = all(mech[str(b)]["ok"] for b in BUDGETS)
    uav_sacrificed = any(h4["r"]["na"][str(b)][0] > 1.10 for b in BUDGETS)
    if gate_ok and better_than_age and mech_ok and not uav_sacrificed:
        h4["verdict"] = "H4-SUPPORTED"
        h4["note"] = "两档 geo ≤ 0.90 且不差于 age；机制闸门通过（CartPole 被压进窗口）"
    elif gate_ok and (not better_than_age or not mech_ok or uav_sacrificed):
        h4["verdict"] = "H4-PARTIAL"
        why = []
        if not better_than_age:
            why.append("至少一档不优于 age")
        if not mech_ok:
            why.append("机制未兑现（间隔仍超窗）⇒ 数字可能是巧合")
        if uav_sacrificed:
            why.append("UAV r_0>1.10（牺牲 UAV）")
        h4["note"] = "两档 ≤ 0.90 但 " + "；".join(why)
    else:
        h4["verdict"] = "H4-FALSIFIED"
        h4["note"] = f"有档位 geo > {GATE} ⇒ 约束满足型目标也没过 Gate"
    print(f"  ⇒ {h4['verdict']}：{h4['note']}")

    _dump({"selfchecks": sc, "experiment": {"cells": cells}}, cells, summary, h4, n_ep, seeds,
          run_id=args.run_id, allow_overwrite=args.allow_overwrite, persist=not args.smoke)
    print(f"\n{'ALL CHECKS PASS' if not FAILS else 'FAILED: ' + '; '.join(FAILS)}")
    print(f"总耗时 {round(time.time() - t_all, 1)}s")
    return 0 if not FAILS else 2


def _dump(payload, cells, summary, h4, n_ep, seeds, *, run_id=None, allow_overwrite=False,
          persist=True):
    out = {"config": {"N": 2, "window": WINDOW, "warmup": [WARMUP_UAV, 0],
                      "budgets": list(BUDGETS), "episodes": n_ep, "seeds": seeds,
                      "gate": GATE, "D_measured": list(D_MEASURED), "D_swapped": list(D_SWAPPED),
                      "aggregation": "geo of per-user ratios vs rr, same seed",
                      "prereg": "04-实验与代码/2026-10-19-T15-归一化年龄-预注册-v0.1.md",
                      "litcheck": "03-文献与竞品/2026-10-19-T15-deadline与虚拟队列占位检索-v0.1.md"},
           "selfchecks": payload["selfchecks"],
           "summary": summary, "h4": h4,
           "checks_failed": list(FAILS)}
    (ROOT / "outputs").mkdir(exist_ok=True)
    (ROOT / "outputs" / "47_t15_normalized_age.json").write_text(
        json.dumps({**out, "cells": (cells or [])}, ensure_ascii=False, indent=2), encoding="utf-8")
    if cells is None or not persist:
        # `--skip-experiment` / `--smoke` 只跑自检或接线，**不得**写入库证据。
        # 实测踩点（T16）：这条路径曾把 results/47 的 24 个 cell 写成空表，须 git checkout 才救回。
        if cells is not None:
            print("[smoke] 不写 results/（smoke 数字不进证据链，也不抢 INDEX 的 current）")
        return
    slim = [{k: c[k] for k in ("budget", "arm", "seed", "J_per_episode", "J_mean",
                               "share_attempts", "link_q", "gap_cart", "gap_uav",
                               "alive_steps", "frames", "attempts", "elapsed_s")}
            for c in (cells or [])]
    # ★ P0-3（2026-10-19）：改走 run-id 分目录；同名 run-id 再写会 SystemExit(3)。
    dest = write_result("47_t15_normalized_age", {**out, "cells_J": slim},
                        run_id=run_id, allow_overwrite=allow_overwrite)
    print(f"[evidence] {dest.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    sys.exit(main())
