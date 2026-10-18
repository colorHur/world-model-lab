#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""T12 · 异质链路下「四策略 × 两档预算 × 3 种子」前置诊断（Gate 2 的唯一数据来源）。

口径来源（★ 全部在看结果之前冻结）
------------------------------------
`04-实验与代码/2026-10-19-T12-口径修订与四臂预注册-v0.1.md` §3–§5。
本脚本**只实现**那套口径，不新增判据；脚本里出现的阈值一律从那套口径抄来，不在这里调。

系统（N=2，异质任务 + 异质链路）
--------------------------------
    用户 0：`uav_track`，PD 控制器，warmup 300，评估窗口 200                    代价 = 跟踪距离 [m]
    用户 1：`CartPole-v1`，LQR（与 `scripts/35` 同一个 DARE 增益），无 warmup      代价 = 0/1（倒=1）
    csi：AR(1)（rho=0.9, sd=0.10），均值 (0.90, 0.55)，clip [0.10, 1.0]，
         **逐臂共享同一条轨迹**（公共随机数）；绑定方式 = `default_rng([seed, 7919])`
    q = SINRSuccess 口径 A：q(csi) = exp(−κ/csi)，κ = −ln 0.7

四臂：`rr` / `age` / `chan` / `oracle`（oracle = myopic 真值误差，**不是上界**）
两档：`B=100`（p_eff≈0.82）/ `B=200`（p_eff≈0.65）

判据（预注册，跑完不改）
------------------------
  P1 `budget_reserved == 30·B`（每步恰好 1 帧，预算用尽后停发）
  P2 `Σframes(汇总) == 原始表 count(delivered_idx ≥ 0)`，且 `reserved == attempted + wasted`
  P3 `link_q_i` 与"按 csi 轨迹算的期望 q"在 4σ 二项带内（验 SINRSuccess 接线，R12）
  P4 `chan` 臂对用户 0 的 attempts 份额 > 0.9（否则异质链路没接上，臂是退化的）
  G2 每档 `mean_seed( geo(r_i) ) ≤ 0.90`；两档同向 ⇒ 通过
  R4 逐用户 `r_i` 必须报；任一 `r_i > 1.10` 标"该用户被牺牲"

统计口径（§5 已定）：`r_i(seed) = J_i(arm,seed) / J_i(rr,seed)`（同种子配对），
`geo(seed) = sqrt(r_0·r_1)`；报 3 个种子的均值 ± 样本标准差。**不做事后选种子。**

用法
----
    D:/devtools/miniconda3/envs/ai-lab/python.exe scripts/42_t12_hetero_four_arms.py
    ... --smoke            # 2 集 × 1 种子，只验接线
    ... --episodes 30 --seeds 0 1 2

产物
----
    outputs/42_t12_hetero_four_arms.json   完整逐 episode 表（不入库，可重跑）
    results/42_t12_four_arms_summary.json  逐 episode J + 汇总（**入库**，作为证据）
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

from wmlab.control import PDRelativeController  # noqa: E402
from wmlab.envs import make_env  # noqa: E402
from wmlab.eval.allocators import MaxAge, MaxChannel, RoundRobin, TrueError  # noqa: E402
from wmlab.eval.multiuser import (  # noqa: E402
    SINRSuccess, default_uav_cost, delivery_rate, run_multiuser, window_cost,
)

WINDOW = 200
WARMUP_UAV = 300
MUS = (0.90, 0.55)
RHO = 0.9
CSI_SD = 0.10
Q_REF = 0.7
BUDGETS = (100, 200)
ESCAPE_RADIUS = 25.0        # UavTrackAdapter.escape_radius：跟丢 ⇒ 终止
ABSORBED_UAV = ESCAPE_RADIUS  # 跟丢后窗口内每步按逃逸半径记（不是 0！0 会把"丢机"奖励成"零代价"）
ABSORBED_CART = 1.0           # 倒杆后窗口内每步记 1
ARMS = ("rr", "age", "chan", "oracle")
GATE = 0.90
FAILS: list[str] = []


def check(tag: str, ok: bool, detail: str) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {tag}: {detail}")
    if not ok:
        FAILS.append(f"{tag}: {detail}")


# ================================================================ 环境 / 控制器 / 代价
def load_lqr_gain():
    """复用 `scripts/35` 的 DARE 增益（不复制公式，避免两处漂移）。"""
    spec = importlib.util.spec_from_file_location("judge01", ROOT / "scripts"
                                                 / "35_judge_task_sensitivity.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["judge01"] = mod          # ★ dataclass 需要模块已注册（否则 _is_type 取不到 __dict__）
    spec.loader.exec_module(mod)          # 该脚本有 __main__ 守卫，import 不跑实验
    ch, _obs, K, _dt = mod.make_cartpole(seed=0, ep_seed=0)
    ch.close()
    return K


class CartPoleLQR:
    """与 `scripts/35:run_classic` 同一条规则：u = −K x，取符号成离散动作。"""

    def __init__(self, K: np.ndarray):
        self.K = np.asarray(K, dtype=float)

    def reset(self, est=None) -> None:
        pass

    def act(self, est) -> np.ndarray:
        u = float(-(self.K @ np.asarray(est, dtype=float))[0])
        return np.array([1 if u > 0 else 0], dtype=np.int64)


def cartpole_cost(obs: np.ndarray) -> float:
    """0 = 还立着；1 = 已倒（用 gymnasium CartPole-v1 的终止阈值，不是自定阈值）。

    ★ 必须与 gym 的 terminated 一致：x_threshold=2.4，theta_threshold=12° = 0.2095 rad。
      否则"代价说没倒、env 说倒了"，window_cost 的吸收补齐会把两者算成两件事。
    """
    o = np.asarray(obs, dtype=float).reshape(-1)
    return 1.0 if (abs(o[0]) > 2.4 or abs(o[2]) > 0.2095) else 0.0


def make_csi(seed: int, n_episodes: int, n_users: int = 2) -> np.ndarray:
    """AR(1) 异质链路：同种子同轨迹，**不消耗** runner 的任何 rng。"""
    rng = np.random.default_rng([int(seed), 7919])
    z = np.zeros((n_episodes, WINDOW, n_users), dtype=float)
    for e in range(n_episodes):
        eps = rng.standard_normal((WINDOW, n_users))
        v = np.zeros(n_users)
        for t in range(WINDOW):
            v = RHO * v + np.sqrt(1.0 - RHO ** 2) * eps[t]
            z[e, t] = v
    return np.clip(np.asarray(MUS)[None, None, :] + CSI_SD * z, 0.10, 1.0)


def build(seed: int):
    envs = [make_env("uav", seed=seed, max_steps=WARMUP_UAV + WINDOW),
            make_env("CartPole-v1", seed=seed)]
    ctrls = [PDRelativeController(), CartPoleLQR(load_lqr_gain())]
    return envs, ctrls


class NeverServe:
    """B=0 试点：一帧都不发（算 s_i，与任何臂的收益无关）。"""

    def __call__(self, view, rng):
        from wmlab.eval.multiuser import Allocation
        return Allocation(served=np.zeros(len(view.slots), dtype=bool), reason="pilot-b0")


# ================================================================ 主流程
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    n_ep = 2 if args.smoke else args.episodes
    seeds = [0] if args.smoke else args.seeds

    K = load_lqr_gain()
    q_model = SINRSuccess(Q_REF)
    t_all = time.time()
    print(f"[42] N=2 异质（uav + cartpole）｜窗口 {WINDOW}｜预算 {BUDGETS}｜"
          f"seeds={seeds}｜episodes={n_ep}")

    # ---------- 0) B=0 试点：逐用户尺度 s_i（§4，与臂收益无关）----------
    pilot_J = []
    for seed in seeds:
        envs = [make_env("uav", seed=seed, max_steps=WARMUP_UAV + WINDOW),
                make_env("CartPole-v1", seed=seed)]
        r = run_multiuser(envs, [PDRelativeController(), CartPoleLQR(K)], NeverServe(),
                          q_model, n_episodes=n_ep, seed=seed, max_steps=WINDOW,
                          warmup_steps=[WARMUP_UAV, 0],
                          budget_total=0, cost_fn=[default_uav_cost, cartpole_cost],
                          csi=make_csi(seed, n_ep))
        J = window_cost(r["records"], 2, window=WINDOW, absorbed=[ABSORBED_UAV, ABSORBED_CART], n_episodes=n_ep)
        pilot_J.append(np.mean(np.asarray(J), axis=0).tolist())
        for e in envs:
            e.close()
    s_scale = np.mean(np.asarray(pilot_J), axis=0).tolist()
    print(f"[42] B=0 试点 s_i（逐种子）={np.round(pilot_J, 4).tolist()} ⇒ s={np.round(s_scale, 4)}")
    if min(s_scale) <= 1e-9:
        print("[42] ★ s_i 含 0 ⇒ 零预算下任务代价为 0，oracle 的尺度退化；"
              "本轮仍可跑，但 oracle 的排序依据会失去意义（结果里标注）")

    # ---------- oracle 的"真值任务潜力"（v0.2，见预注册文档 §4 的修订记录）----------
    # ★ 为什么不用逐 step 官方代价：CartPole 的代价是 0/1 阶跃，活着的每一步都等于 0
    #   ⇒ myopic 贪心在"全都活着"时对用户 1 的信号恒为 0，oracle 必然退化成 max-channel
    #   （2 集冒烟实测：oracle 与 chan 的 J、份额逐位相同）。改用**归一化的失败接近度**，
    #   它仍是真值（完美观测），只是把"离失败还有多远"这个梯度暴露出来。
    def uav_pot(obs):
        return default_uav_cost(obs)

    def cart_pot(obs):
        o = np.asarray(obs, dtype=float).reshape(-1)
        return float(max(abs(o[0]) / 2.4, abs(o[2]) / 0.2095))

    cells: list[dict] = []
    for budget in BUDGETS:
        for arm in ARMS:
            for seed in seeds:
                t0 = time.time()
                envs = [make_env("uav", seed=seed, max_steps=WARMUP_UAV + WINDOW),
                        make_env("CartPole-v1", seed=seed)]
                ctrls = [PDRelativeController(), CartPoleLQR(K)]
                if arm == "rr":
                    alloc, diag = RoundRobin(), False
                elif arm == "age":
                    alloc, diag = MaxAge(), False
                elif arm == "chan":
                    alloc, diag = MaxChannel(), False
                else:
                    alloc, diag = TrueError([uav_pot, cart_pot], scales=[s_scale[0], s_scale[1]]), True
                csi = make_csi(seed, n_ep)
                r = run_multiuser(envs, ctrls, alloc, q_model, n_episodes=n_ep, seed=seed,
                                  max_steps=WINDOW, warmup_steps=[WARMUP_UAV, 0],
                                  budget_total=budget, csi=csi,
                                  cost_fn=[default_uav_cost, cartpole_cost],
                                  diagnostic=diag)
                dr = delivery_rate(r["records"], 2)
                J = np.asarray(window_cost(r["records"], 2, window=WINDOW,
                                           absorbed=[ABSORBED_UAV, ABSORBED_CART], n_episodes=n_ep))
                # 期望 q（按实际被派帧的步，用同一条 csi 轨迹算）
                q_exp = [0.0, 0.0]
                n_att = [0, 0]
                for rec in r["records"]:
                    i = int(rec.attempted_idx)
                    if i >= 0:
                        q_exp[i] += q_model.q(float(csi[rec.episode, rec.t - 1, i]))
                        n_att[i] += 1
                q_exp = [q_exp[i] / n_att[i] if n_att[i] else float("nan") for i in range(2)]
                alive0 = dr["alive_steps"][0]
                cell = {
                    "budget": budget, "arm": arm, "seed": seed,
                    "J_per_episode": J.tolist(),
                    "J_mean": J.mean(axis=0).tolist(),
                    "J_sd": J.std(axis=0, ddof=1).tolist() if n_ep > 1 else [0.0, 0.0],
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
                    "uav_alive_full": bool(dr["alive_steps"][0] == n_ep * WINDOW),
                    "uav_absorbed_used": bool(alive0 < n_ep * WINDOW),
                    "elapsed_s": round(time.time() - t0, 2),
                }
                cells.append(cell)
                print(f"  B={budget:3d} {arm:6s} seed={seed} "
                      f"J=({cell['J_mean'][0]:.4f},{cell['J_mean'][1]:.4f}) "
                      f"share=({cell['share_attempts'][0]:.3f},{cell['share_attempts'][1]:.3f}) "
                      f"link_q=({dr['link_q'][0]:.3f},{dr['link_q'][1]:.3f}) "
                      f"wasted={dr['budget_wasted']} {cell['elapsed_s']}s")
                for e in envs:
                    e.close()

    # ---------- 汇总 + 判据 ----------
    summary: dict = {}
    print("\n== 判据 ==")
    for budget in BUDGETS:
        sub = [c for c in cells if c["budget"] == budget]
        rr = {c["seed"]: c for c in sub if c["arm"] == "rr"}
        per_arm = {}
        for arm in ARMS:
            a = {c["seed"]: c for c in sub if c["arm"] == arm}
            rs = []
            for seed in seeds:
                j, jr = np.asarray(a[seed]["J_mean"]), np.asarray(rr[seed]["J_mean"])
                rr_deg = bool(np.any(jr <= 1e-12))
                r = (j / jr) if not rr_deg else np.array([np.nan, np.nan])
                geo = float(np.sqrt(r[0] * r[1])) if not rr_deg else float("nan")
                rs.append({"seed": seed, "r": r.tolist(), "geo": geo,
                           "degenerate_rr": rr_deg})
            geos = [x["geo"] for x in rs if np.isfinite(x["geo"])]
            per_arm[arm] = {
                "per_seed": rs,
                "r_mean": np.nanmean([[np.nan if v is None else v for v in x["r"]]
                                      for x in rs], axis=0).tolist(),
                "geo_mean": float(np.mean(geos)) if geos else float("nan"),
                "geo_sd": float(np.std(geos, ddof=1)) if len(geos) > 1 else 0.0,
                "degenerate": bool(not geos),
            }
        summary[budget] = per_arm

    for budget in BUDGETS:
        sub = [c for c in cells if c["budget"] == budget]
        n_cells = len(sub)
        check(f"P1[{budget}] reserved == n_ep·B",
              all(c["budget_reserved"] == n_ep * budget for c in sub),
              f"{[(c['arm'], c['seed'], c['budget_reserved']) for c in sub if c['budget_reserved'] != n_ep * budget][:3]}"
              f"（{n_cells} 个 cell）")
        ok2 = all(c["budget_reserved"] == c["budget_attempted"] + c["budget_wasted"] for c in sub)
        check(f"P2[{budget}] 三分账闭合", ok2, "reserved == attempted + wasted")
        ok3 = True
        detail3 = []
        for c in sub:
            for i in range(2):
                qe, lq, n = c["q_expected"][i], c["link_q"][i], c["attempts"][i]
                if n == 0:
                    # ★ 该用户一次都没被派过帧（如 max-channel 完全放弃用户 1）⇒ 不可比，
                    #   不是失败。是否"零服务"由 P4 与 R4 单独报，不在这里判。
                    continue
                if not (np.isfinite(qe) and np.isfinite(lq)):
                    ok3 = False
                    detail3.append(f"{c['arm']}/s{c['seed']}/u{i}: 不可比 qe={qe} lq={lq} n={n}")
                    continue
                band = 4.0 * np.sqrt(max(qe * (1 - qe), 1e-12) / n)
                if abs(lq - qe) > band:
                    ok3 = False
                    detail3.append(f"{c['arm']}/s{c['seed']}/u{i}: |{lq:.4f}-{qe:.4f}|>{band:.4f}")
        check(f"P3[{budget}] link_q 落在期望 q 的 4σ 带内", ok3,
              detail3[:4] if detail3 else "全 cell 通过")
        chan = [c for c in sub if c["arm"] == "chan"]
        check(f"P4[{budget}] max-channel 份额偏向用户0 (>0.9)",
              all(c["share_attempts"][0] > 0.9 for c in chan),
              f"share0={[round(c['share_attempts'][0], 3) for c in chan]}")

    # ★ S1 原写"UAV 不终止"，冒烟实测**会终止**（预算用尽后估计冻结，PD 把机飞出去，
    #   dist>25m 触发 escape）⇒ 判据改成"终止必须被正确计价"，而不是"不许终止"。
    lb_ok = all(c["J_mean"][0] >= ABSORBED_UAV * (1.0 - c["survival_frac"][0]) - 1e-9
                for c in cells)
    n_esc = sum(1 for c in cells if not c["uav_alive_full"])
    check("S1 UAV 逃逸被正确计价（J ≥ escape·未存活占比）", lb_ok,
          f"{n_esc}/{len(cells)} 个 cell 发生 UAV 逃逸；"
          f"最差 cell={min(cells, key=lambda c: c['survival_frac'][0])['survival_frac'][0]:.3f} 存活率")

    print("\n== 汇总（r_i 相对 rr，同种子配对）==")
    verdict = {}
    for budget in BUDGETS:
        for arm in ARMS:
            d = summary[budget][arm]
            print(f"  B={budget:3d} {arm:6s} r=({d['r_mean'][0]:.3f},{d['r_mean'][1]:.3f}) "
                  f"geo={d['geo_mean']:.3f}±{d['geo_sd']:.3f}")
        g = summary[budget]["oracle"]["geo_mean"]
        verdict[budget] = ("PASS" if g <= GATE else ("NO-SPACE" if 0.95 <= g <= 1.05 else "FAIL"))
        check(f"G2[{budget}] oracle geo ≤ {GATE}", g <= GATE, f"geo={g:.4f} ⇒ {verdict[budget]}")
    same_dir = (summary[200]["oracle"]["geo_mean"] - 1) * (summary[100]["oracle"]["geo_mean"] - 1) > 0
    check("G2 两档同向", same_dir, f"geo(100)={summary[100]['oracle']['geo_mean']:.3f}, "
                                   f"geo(200)={summary[200]['oracle']['geo_mean']:.3f}")
    sacrificed = [f"u{i} B={b}" for b in BUDGETS for i in range(2)
                  if summary[b]["oracle"]["r_mean"][i] > 1.10]
    print(f"  R4 被牺牲的用户：{sacrificed if sacrificed else '无'}")

    out = {
        "config": {"N": 2, "window": WINDOW, "warmup": [WARMUP_UAV, 0], "mus": MUS,
                   "rho": RHO, "csi_sd": CSI_SD, "q_ref": Q_REF, "budgets": list(BUDGETS),
                   "arms": list(ARMS), "episodes": n_ep, "seeds": seeds,
                   "cost": ["uav: dist[m]", "cartpole: 1 if fallen else 0"],
                   "absorbed": [0.0, 1.0], "gate": GATE,
                   "aggregation": "geo of per-user ratios vs rr, same seed",
                   "oracle_note": "myopic true-error, 1-step lookahead NOT implemented, NOT an upper bound",
                   "q4_signoff": "autonomous default (not human-signed), see doc §6"},
        "pilot_b0": {"per_seed": pilot_J, "s_mean": s_scale},
        "cells": cells,
        "summary": summary,
        "verdict": verdict,
        "sacrificed_users": sacrificed,
        "checks_failed": FAILS,
        "elapsed_s": round(time.time() - t_all, 1),
    }
    (ROOT / "outputs").mkdir(exist_ok=True)
    (ROOT / "outputs" / "42_t12_hetero_four_arms.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "42_t12_four_arms_summary.json").write_text(
        json.dumps({k: out[k] for k in ("config", "pilot_b0", "summary", "verdict",
                                        "sacrificed_users", "checks_failed", "elapsed_s")}
                   | {"cells_J": [{"budget": c["budget"], "arm": c["arm"], "seed": c["seed"],
                                   "J_per_episode": c["J_per_episode"],
                                   "survival_frac": c["survival_frac"],
                                   "share_attempts": c["share_attempts"],
                                   "link_q": c["link_q"], "p_eff": c["p_eff"]}
                                  for c in cells]},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{'ALL CHECKS PASS' if not FAILS else 'FAILED: ' + '; '.join(FAILS)}")
    print(f"总耗时 {out['elapsed_s']}s ⇒ outputs/42_t12_hetero_four_arms.json")
    return 0 if not FAILS else 2


if __name__ == "__main__":
    sys.exit(main())
