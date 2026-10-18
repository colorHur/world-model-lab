#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""T14 · 多步前瞻（视界 H ∈ {1,5,20}）诊断 —— 判 H3。

口径来源（★ 全部在看结果之前冻结）
------------------------------------
`06-进度管理/自动推进/STATE.md` §7 + `04-实验与代码/2026-10-19-T14-多步前瞻视界-预注册-v0.1.md` §1/§2/§5。
本脚本**只实现**那套口径。GATE / 预算档 / 种子 / 集数 / 窗口 / csi 轨迹**一律不动**。

系统（与 T12/T13 逐位同一套）
------------------------------
    用户 0：`uav_track`，PD，warmup 300，窗口 200，代价 = 距离 [m]
    用户 1：`CartPole-v1`，LQR，无 warmup，代价 = 0/1
    csi：AR(1)（rho=0.9, sd=0.10），均值 (0.90, 0.55)，clip [0.10,1.0]，`default_rng([seed,7919])`
    q = SINRSuccess 口径 A
★ 直接 import `scripts/42` 的常量与生成器（不复制），保证 csi 轨迹、LQR 增益、代价函数同源。

臂：`rr` / `age` / `la1` / `la5` / `la20`（`laH` = 视界 H 的期望前瞻，`cf_horizon=H`）

判据（冻结，跑完不改）
----------------------
  P1 `budget_reserved == n_ep·B`
  P2 `reserved == attempted + wasted`
  P3 `link_q` 落在期望 q 的 4σ 带内
  R0 **回归**：`rr / age / la1` 的逐 episode `J` 必须与 `results/42_t12_four_arms_summary.json`
     **逐位相同**（跨轮同环境复现；不同则先查漂移，不许用新数覆盖旧结论）
  H3 中间量优先：`cart_gain(H)`；若 `cart_gain(20)/s_cart < 0.1` ⇒ 停，判"视界不是瓶颈"
     否则 `geo(la20) ≤ 0.90` 两档 ⇒ 成立，任一档 > 0.90 ⇒ 否证

用法
----
    D:/devtools/miniconda3/envs/ai-lab/python.exe scripts/45_t14_horizon_arms.py
    ... --smoke                       # 2 集 × 1 种子，只验接线
    ... --horizons 1 5                # 只跑 H=5（H=20 超预算时的降级）

产物
----
    outputs/45_t14_horizon_arms.json    完整逐 episode 表（不入库，可重跑）
    results/45_t14_horizon_arms.json    逐 episode J + 中间量 + 判据（**入库**）
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

from wmlab.eval.allocators import Lookahead, MaxAge, RoundRobin  # noqa: E402
from wmlab.eval.multiuser import (  # noqa: E402
    SINRSuccess, default_uav_cost, delivery_rate, run_multiuser, window_cost,
)


def _load_t12():
    """import `scripts/42`（有 __main__ 守卫，import 不跑实验）—— 常量/生成器同源。"""
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

FAILS: list[str] = []


def check(tag: str, ok: bool, detail: str) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {tag}: {detail}")
    if not ok:
        FAILS.append(f"{tag}: {detail}")


def gains(trace: list, n_users: int = 2) -> dict:
    """从 `Lookahead.trace` 抽中间量：`cart_gain(H)` / `uav_gain(H)`（逐决策平均）。

    定义（预注册 §2.3）：只取"两个用户都存活"的决策；单位 = 每步代价（矩阵已除过 H）。
    """
    two = [d for d in trace if all(d["alive"][:n_users])]
    out = {"n_decisions": len(trace), "n_two_cand": len(two),
           "choice_count": {str(k): sum(1 for d in trace if d["chosen"] == k)
                            for k in range(n_users)}}
    if not two:
        out.update({"uav_gain": float("nan"), "cart_gain": float("nan"),
                    "cart_gain_n_pos": 0, "cart_gain_n_neg": 0})
        return out
    m = np.asarray([d["mat"] for d in two], dtype=float)
    # ★ 符号约定（2026-10-19 T14 统一）：gain = "服务他人时的代价 − 服务自己时的代价" > 0 ⇒ 服务自己更好。
    #   与 `scripts/44` L7（`mat[0][0] < mat[1][0]`）同一约定。
    #   ⚠ T13 的 `scripts/42` 里 `la_cart_gain_raw = m[:,1,1] − m[:,0,1]` 是**相反**约定（−benefit），
    #   因为当时它恒等于 0（0 = −0）所以从未暴露。本轮按 L7 约定统一，见 T14 结果文档 §自证伪。
    out["uav_gain"] = float(np.mean(m[:, 1, 0] - m[:, 0, 0]))
    d_cart = m[:, 0, 1] - m[:, 1, 1]
    out["cart_gain"] = float(np.mean(d_cart))
    out["cart_gain_n_pos"] = int(np.sum(d_cart > 0))
    out["cart_gain_n_neg"] = int(np.sum(d_cart < 0))
    out["cart_gain_max"] = float(np.max(d_cart))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 20])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    n_ep = 2 if args.smoke else args.episodes
    seeds = [0] if args.smoke else args.seeds
    hs = list(args.horizons)
    arms = ["rr", "age"] + [f"la{h}" for h in hs]

    K = T12.load_lqr_gain()
    q_model = SINRSuccess(T12.Q_REF)
    t_all = time.time()
    print(f"[45] T14 多步前瞻｜N=2 异质｜窗口 {WINDOW}｜预算 {BUDGETS}｜seeds={seeds}"
          f"｜episodes={n_ep}｜H={hs}｜arm={arms}")

    # ---------- B=0 试点：逐用户尺度 s_i（与臂收益无关，口径同 42）----------
    pilot_J = []
    for seed in seeds:
        envs = [T12.make_env("uav", seed=seed, max_steps=WARMUP_UAV + WINDOW),
                T12.make_env("CartPole-v1", seed=seed)]
        r = run_multiuser(envs, [T12.PDRelativeController(), T12.CartPoleLQR(K)],
                          T12.NeverServe(), q_model, n_episodes=n_ep, seed=seed,
                          max_steps=WINDOW, warmup_steps=[WARMUP_UAV, 0],
                          budget_total=0, cost_fn=[default_uav_cost, T12.cartpole_cost],
                          csi=T12.make_csi(seed, n_ep))
        J = window_cost(r["records"], 2, window=WINDOW, absorbed=ABSORBED, n_episodes=n_ep)
        pilot_J.append(np.mean(np.asarray(J), axis=0).tolist())
        for e in envs:
            e.close()
    s_scale = np.mean(np.asarray(pilot_J), axis=0).tolist()
    print(f"[45] B=0 试点 s_i={np.round(s_scale, 4).tolist()}")

    def uav_pot(obs):
        return default_uav_cost(obs)

    def cart_pot(obs):
        o = np.asarray(obs, dtype=float).reshape(-1)
        return float(max(abs(o[0]) / 2.4, abs(o[2]) / 0.2095))

    cells: list[dict] = []
    for budget in BUDGETS:
        for arm in arms:
            for seed in seeds:
                t0 = time.time()
                envs = [T12.make_env("uav", seed=seed, max_steps=WARMUP_UAV + WINDOW),
                        T12.make_env("CartPole-v1", seed=seed)]
                ctrls = [T12.PDRelativeController(), T12.CartPoleLQR(K)]
                tr = None
                if arm == "rr":
                    alloc, diag, cf, hz = RoundRobin(), False, False, 1
                elif arm == "age":
                    alloc, diag, cf, hz = MaxAge(), False, False, 1
                else:
                    hz = int(arm[2:])
                    tr = []
                    alloc = Lookahead(scales=[s_scale[0], s_scale[1]], name=arm,
                                      horizon=hz, trace=tr)
                    diag, cf = True, True
                csi = T12.make_csi(seed, n_ep)
                r = run_multiuser(envs, ctrls, alloc, q_model, n_episodes=n_ep, seed=seed,
                                  max_steps=WINDOW, warmup_steps=[WARMUP_UAV, 0],
                                  budget_total=budget, csi=csi,
                                  cost_fn=[default_uav_cost, T12.cartpole_cost],
                                  diagnostic=diag, counterfactual=cf, cf_horizon=hz)
                dr = delivery_rate(r["records"], 2)
                J = np.asarray(window_cost(r["records"], 2, window=WINDOW,
                                           absorbed=ABSORBED, n_episodes=n_ep))
                q_exp = [0.0, 0.0]
                n_att = [0, 0]
                for rec in r["records"]:
                    i = int(rec.attempted_idx)
                    if i >= 0:
                        q_exp[i] += q_model.q(float(csi[rec.episode, rec.t - 1, i]))
                        n_att[i] += 1
                q_exp = [q_exp[i] / n_att[i] if n_att[i] else float("nan") for i in range(2)]
                cell = {
                    "budget": budget, "arm": arm, "horizon": hz, "seed": seed,
                    "J_per_episode": J.tolist(),
                    "J_mean": J.mean(axis=0).tolist(),
                    "alive_steps": dr["alive_steps"],
                    "survival_frac": [dr["alive_steps"][i] / (n_ep * WINDOW) for i in range(2)],
                    "frames": dr["frames"], "attempts": dr["attempts"],
                    "share_attempts": [dr["attempts"][i] / max(sum(dr["attempts"]), 1)
                                       for i in range(2)],
                    "p_eff": dr["p_eff"], "link_q": dr["link_q"], "q_expected": q_exp,
                    "budget_reserved": dr["budget_reserved"],
                    "budget_attempted": dr["budget_attempted"],
                    "budget_wasted": dr["budget_wasted"],
                    "budget_lost": dr["budget_lost"],
                    "elapsed_s": round(time.time() - t0, 2),
                }
                if tr is not None:
                    cell.update({f"la_{k}": v for k, v in gains(tr).items()})
                    cell["la_trace_n"] = len(tr)
                cells.append(cell)
                print(f"  B={budget:3d} {arm:5s} seed={seed} "
                      f"J=({cell['J_mean'][0]:.4f},{cell['J_mean'][1]:.4f}) "
                      f"share=({cell['share_attempts'][0]:.3f},{cell['share_attempts'][1]:.3f}) "
                      f"{cell['elapsed_s']}s"
                      + (f" cart_gain={cell['la_cart_gain']:.4f}" if tr is not None else ""))
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
                "J_mean": float(np.mean([np.mean(np.asarray(a[s]["J_mean"])) for s in seeds])),
            }
        summary[budget] = per_arm

    print("\n== 汇总（r_i 相对 rr，同种子配对）==")
    for budget in BUDGETS:
        for arm in arms:
            d = summary[budget][arm]
            print(f"  B={budget:3d} {arm:5s} r=({d['r_mean'][0]:.3f},{d['r_mean'][1]:.3f}) "
                  f"geo={d['geo_mean']:.3f}±{d['geo_sd']:.3f}")

    # ---------- 判据 ----------
    print("\n== 判据（测试台）==")
    for budget in BUDGETS:
        sub = [c for c in cells if c["budget"] == budget]
        check(f"P1[{budget}] reserved == n_ep·B",
              all(c["budget_reserved"] == n_ep * budget for c in sub),
              f"异常 {[(c['arm'], c['seed'], c['budget_reserved']) for c in sub if c['budget_reserved'] != n_ep * budget][:3]}"
              f"（{len(sub)} cells）")
        check(f"P2[{budget}] 三分账闭合",
              all(c["budget_reserved"] == c["budget_attempted"] + c["budget_wasted"] for c in sub),
              "reserved == attempted + wasted")
        ok3, bad = True, []
        for c in sub:
            for i in range(2):
                qe, lq, n = c["q_expected"][i], c["link_q"][i], c["attempts"][i]
                if n == 0:
                    continue
                if not (np.isfinite(qe) and np.isfinite(lq)):
                    ok3 = False
                    bad.append(f"{c['arm']}/s{c['seed']}/u{i}: 不可比")
                    continue
                band = 4.0 * np.sqrt(max(qe * (1 - qe), 1e-12) / n)
                if abs(lq - qe) > band:
                    ok3 = False
                    bad.append(f"{c['arm']}/s{c['seed']}/u{i}: |{lq:.4f}-{qe:.4f}|>{band:.4f}")
        check(f"P3[{budget}] link_q 在 4σ 带内", ok3, bad[:4] if bad else "全 cell 通过")

    # ---------- R0：与 T13 的 rr/age/la1 逐位回归 ----------
    print("\n== R0 与 results/42 的 rr/age/la1 逐位回归 ==")
    ref_path = ROOT / "results" / "42_t12_four_arms_summary.json"
    r0_ok, r0_detail = True, "未找到 42 结果（跳过，不算通过）"
    if ref_path.exists():
        ref = json.loads(ref_path.read_text(encoding="utf-8"))
        ref_map = {(c["budget"], c["arm"], c["seed"]): c["J_per_episode"]
                   for c in ref.get("cells_J", [])}
        diffs, skipped = [], 0
        for c in cells:
            arm_ref = "lookahead" if c["arm"] == "la1" else c["arm"]
            key = (c["budget"], arm_ref, c["seed"])
            if key not in ref_map:
                continue
            a = np.asarray(c["J_per_episode"], dtype=float)
            b = np.asarray(ref_map[key], dtype=float)
            if a.shape != b.shape:
                # 冒烟模式（集数不同）不做逐位比较 —— 跳过要**显式计数**，不许当"通过"。
                skipped += 1
                continue
            if not np.array_equal(a, b):
                r0_ok = False
                diffs.append(f"{key} max|Δ|={np.nanmax(np.abs(a - b)):.3e}")
        r0_detail = ("逐位相同" if r0_ok else "；".join(diffs[:4])) + \
                    (f"（{skipped} 个 cell 因集数不同跳过）" if skipped else "")
        if skipped and skipped == len([c for c in cells if c["arm"] in ("rr", "age", "la1")]):
            r0_detail = "全部跳过（集数与 T13 不同，冒烟模式）"
            print(f"  [SKIP] R0：{r0_detail}")
        else:
            check("R0 rr/age/la1 与 T13 结果逐位相同", r0_ok, r0_detail)

    # ---------- H3（假设检验，与 check() 分开记）----------
    print("\n== H3（多步前瞻）==")
    s_cart = float(s_scale[1])
    h3 = {"gate": GATE, "s_cart": s_cart, "s_uav": float(s_scale[0]),
          "gains": {}, "geo": {}, "delta_vs_age": {}, "delta_vs_la1": {},
          "verdict": None, "note": ""}
    for h in hs:
        g = [c for c in cells if c["arm"] == f"la{h}"]
        h3["gains"][f"H{h}"] = {
            "cart_gain": float(np.mean([c["la_cart_gain"] for c in g])),
            "uav_gain": float(np.mean([c["la_uav_gain"] for c in g])),
            "cart_gain_n_pos": int(sum(c["la_cart_gain_n_pos"] for c in g)),
            "cart_gain_n_neg": int(sum(c["la_cart_gain_n_neg"] for c in g)),
            "n_two_cand": int(sum(c["la_n_two_cand"] for c in g)),
            "n_decisions": int(sum(c["la_n_decisions"] for c in g)),
            "choice_count": {k: int(sum(c["la_choice_count"][k] for c in g)) for k in ("0", "1")},
        }
        h3["gains"][f"H{h}"]["cart_gain_over_s_cart"] = (
            h3["gains"][f"H{h}"]["cart_gain"] / s_cart if s_cart > 1e-12 else float("nan"))
        for budget in BUDGETS:
            h3["geo"].setdefault(str(budget), {})[f"H{h}"] = summary[budget][f"la{h}"]["geo_mean"]
    h3["delta_vs_age"] = {str(b): {f"H{h}": float(summary[b][f"la{h}"]["geo_mean"]
                                                  - summary[b]["age"]["geo_mean"]) for h in hs}
                          for b in BUDGETS}
    h3["delta_vs_la1"] = {str(b): {f"H{h}": float(summary[b][f"la{h}"]["geo_mean"]
                                                  - summary[b]["la1"]["geo_mean"]) for h in hs}
                          for b in BUDGETS}

    for h in hs:
        gg = h3["gains"][f"H{h}"]
        print(f"  H={h:2d}  cart_gain={gg['cart_gain']:.4f}"
              f"（{gg['cart_gain_over_s_cart']:.3f}·s_cart；"
              f"+{gg['cart_gain_n_pos']}/-{gg['cart_gain_n_neg']}）"
              f"  uav_gain={gg['uav_gain']:.4f}"
              f"  两候选 {gg['n_two_cand']}/{gg['n_decisions']} 次")
        for budget in BUDGETS:
            print(f"        B={budget:3d} geo={h3['geo'][str(budget)][f'H{h}']:.3f}"
                  f"  Δ(age)={h3['delta_vs_age'][str(budget)][f'H{h}']:+.3f}"
                  f"  Δ(la1)={h3['delta_vs_la1'][str(budget)][f'H{h}']:+.3f}")

    top = f"H{max(hs)}"
    g_top = h3["gains"][top]
    gate_ok = all(np.isfinite(h3["geo"][str(b)][top]) and h3["geo"][str(b)][top] <= GATE
                  for b in BUDGETS)
    worse_than_age = all(h3["delta_vs_age"][str(b)][top] > 0 for b in BUDGETS)
    if g_top["cart_gain_over_s_cart"] < 0.1:
        h3["verdict"] = "HORIZON-NOT-BOTTLENECK"
        h3["note"] = (f"cart_gain({top})={g_top['cart_gain']:.4f} < 0.1·s_cart="
                      f"{0.1 * s_cart:.4f} ⇒ 视界不是瓶颈，按预注册 §6 转测任务失效时间尺度")
    elif gate_ok and not worse_than_age:
        h3["verdict"] = "H3-SUPPORTED"
        h3["note"] = f"{top} 两档 geo ≤ {GATE} 且不差于 age"
    elif gate_ok:
        h3["verdict"] = "H3-PARTIAL"
        h3["note"] = f"{top} 两档 geo ≤ {GATE}，但仍差于 age（Δ>0 两档）"
    else:
        h3["verdict"] = "H3-FALSIFIED"
        h3["note"] = f"{top} 有档位 geo > {GATE}"
    print(f"  ⇒ {h3['verdict']}：{h3['note']}")

    out = {
        "config": {"N": 2, "window": WINDOW, "warmup": [WARMUP_UAV, 0],
                   "budgets": list(BUDGETS), "arms": arms, "horizons": hs,
                   "episodes": n_ep, "seeds": seeds, "gate": GATE,
                   "aggregation": "geo of per-user ratios vs rr, same seed",
                   "lookahead_note": "perfect-dynamics H-step one-step-deviation eval; "
                                     "continuation = max-age with deterministic delivery; "
                                     "per-step normalized; NOT an upper bound",
                   "prereg": "04-实验与代码/2026-10-19-T14-多步前瞻视界-预注册-v0.1.md"},
        "pilot_b0": {"per_seed": pilot_J, "s_mean": s_scale},
        "cells": cells,
        "summary": summary,
        "h3": h3,
        "checks_failed": FAILS,
        "elapsed_s": round(time.time() - t_all, 1),
    }
    (ROOT / "outputs").mkdir(exist_ok=True)
    (ROOT / "outputs" / "45_t14_horizon_arms.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "45_t14_horizon_arms.json").write_text(
        json.dumps({**{k: out[k] for k in ("config", "pilot_b0", "summary", "h3",
                                           "checks_failed", "elapsed_s")},
                    "cells_J": [{k: c[k] for k in ("budget", "arm", "horizon", "seed",
                                                   "J_per_episode", "J_mean", "survival_frac",
                                                   "share_attempts", "link_q", "elapsed_s")}
                                | ({k: v for k, v in c.items() if k.startswith("la_")})
                                for c in cells]}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"\n{'ALL CHECKS PASS' if not FAILS else 'FAILED: ' + '; '.join(FAILS)}")
    print(f"总耗时 {out['elapsed_s']}s ⇒ results/45_t14_horizon_arms.json")
    return 0 if not FAILS else 2


if __name__ == "__main__":
    sys.exit(main())
