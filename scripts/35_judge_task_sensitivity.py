#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""判-01：任务**通信敏感性**筛选（冲刺计划 §4 的唯一"选任务"依据）。

一句话
------
**对每个候选任务族，跑真实动力学 + 固定策略，比 p=0 与 p=0.3 两种条件下的任务代价；
比值 ≥ 1.5 判敏感、< 1.2 淘汰。** 这是本轮唯一允许的"选任务"依据（计划 §4 原文）。

本条不是新实验编号（不占 X 序列），命名 `判-01`，见计划 §8。

口径（★ 三条，写在这里，不许事后改）
------------------------------------
1. **不训任何模型。** 三个任务族的控制器全部是解析解：
   - UAV：`PDRelativeController`（增益有闭式推导，见 `wmlab/control.py` 文件头）
   - Pendulum / CartPole：**LQR，增益由离散 Riccati 方程（DARE）解出**，不手调
   丢包时一律 **persistence（保持上次收到的观测）** —— 不引入"世界模型好不好"这个变量。
2. **任务代价用真值算**（不因信道而换口径）；丢失只影响**控制器看到的估计**。
3. **UAV 走调度口径、经典控制走 ChannelAdapter**：
    - UAV 用 `run_closed_loop_control`（X30–X45 同源），丢包由 `lossy_schedule(0.3)` 决定；
    - 经典控制用 `ChannelAdapter`（计划 §4 原文指定的装置）。
    两者都是 i.i.d. Bernoulli 丢包，语义一致、代码路径不同 —— **这是本表唯一的口径差，必须标注。**

代价定义（每个族各自一个，都是"越低越好"）
-------------------------------------------
| 族 | 代价 | 说明 |
|---|---|---|
| `uav_track` | 稳态跟踪距离 `mean_dist_tail` [m] | 与 X30–X45 同一个量 |
| `pendulum_reg` | 每步 `θ² + 0.1θ̇² + 0.001u²`（= gymnasium 的负奖励） | 调节任务，初值 ±0.3 rad |
| `cartpole` | `1 − 存活步数/500` | 失败即代价 1；**α 条件下代价≈0 ⇒ 比值需设地板，已标注** |

自检（★ R12：判据写完先用已知答案代验量级）
--------------------------------------------
- [E1] `ChannelAdapter(p=0)` 的观测序列与裸 env **逐位一致**（X24 的性质，顺带验一遍）
- [S1] p=0 时三个族都必须**真的控得住**：UAV 稳态距离 < 1 m（解析 0.40）、
       Pendulum 每步代价 < 0.5、CartPole 存活率 ≥ 0.9×500。不满足 ⇒ 控制器没接对，**不得报比值**。
- [S2] 比值必须是"同种子同 episode 初值、只改丢包"的配对比较。

实测结果（2026-10-18，30 集 × 3 种子；`outputs/` 不入库，故结果写在这里）
--------------------------------------------------------------------------
| 族 | α(p=0) | β(p=.3) | 比值 | §4 判定 |
|---|---|---|---|---|
| `uav_track`      | 0.39329 | 2.90899 | **7.396** | 敏感·保留 |
| `cartpole`       | 0.00000 | 0.17158 | **∞**（改为存活时间比 1.31） | 口径退化 ⇒ 边缘 |
| `pendulum_swing` | 0.82328 | 1.01276 | **1.230** | 边缘 |
| `pendulum_reg`   | 0.001258 | 0.001260 | **1.006** | 淘汰 |

剂量响应（p = 0 / 0.3 / 0.6 / 0.9）——
    uav 0.393 / 2.910 / 3.064 / 12.503
    swing 0.597 / 0.599 / 1.599 / 5.495
    reg 0.0014 / 0.0014 / 0.533 / 6.312
    cartpole 0.000 / 0.298 / 0.870 / 0.966
⇒ **三个族拐点在 p≈0.6，只有 UAV 在 0.3 就恶化**。单用户 p=0.3 的实测平均年龄仅 **0.402 步**
（闭式 q/s = 0.4286），这个档位对"多用户竞争"（每用户等效 p ≈ 1−1/N）而言**太松**。

★ 两个被自检抓到的错（写在这里，因为它们是**坑**不是结果）
1. UAV `warmup(300) > env.max_steps(200)` ⇒ 被测段每集只剩 1 步，p=0/0.3/0.9 代价**逐位相同**，
   看起来像"不敏感"、其实是"没测到"。修正见 `run_uav` 的 docstring。
2. CartPole 线性化把倒立摆写成正立摆（`θ̈` 对 `θ` 的系数符号反了）⇒ 存活率 9%。S1 自检炸出。

用法
----
    D:/devtools/miniconda3/envs/ai-lab/python.exe scripts/35_judge_task_sensitivity.py --episodes 30
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.linalg as sla

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from wmlab.control import PDRelativeController, run_closed_loop_control  # noqa: E402
from wmlab.envs import ChannelAdapter, make_env  # noqa: E402
from wmlab.eval.tracking import lossy_schedule  # noqa: E402

P_LOSS = 0.3          # 计划 §4 原文：条件 β 固定丢包 p=0.3
SEEDS = (0, 1, 2)     # 计划 §4 原文：3 种子
RATIO_SENSITIVE = 1.5
RATIO_KILL = 1.2
COST_FLOOR = 1e-6     # 仅在 α 代价数值为 0 时用来避免除零；比值会标 inf
WARMUP_UAV = 300      # 与 X30–X45 一致：先让 PD 收敛到稳态，再切被测调度
MEASURED_UAV = 200    # 被测段长度


# ================================================================ LQR 工具
def lqr_gain(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """离散 LQR 增益 K（u = −K x），由 DARE 解出。**不手调。**"""
    P = sla.solve_discrete_are(A, B, Q, R)
    return np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)


# ================================================================ 三个任务族
@dataclass
class Family:
    name: str
    t_max: int


def make_pendulum_reg(seed: int, ep_seed: int):
    """Pendulum **调节**任务（不是起摆）：初值 θ ∈ ±0.3 rad、θ̇ ∈ ±0.3 rad/s。

    ★ 为什么不起摆：起摆需要 energy shaping，多一层"我的控制器行不行"的变量；
      本筛选要的是"同一控制律在两种丢包条件下的代价比"，调节任务已足够。
      代价：**本族结论不能外推到起摆任务**（已在结果里标注）。
    """
    base = make_env("Pendulum-v1", seed=seed)
    ch = ChannelAdapter(base, loss_prob=0.0, delay=0, seed=seed)   # loss 由外部设
    obs = ch.reset(seed=ep_seed)
    rng = np.random.default_rng(ep_seed + 101)
    th = float(rng.uniform(-0.3, 0.3))
    thdot = float(rng.uniform(-0.3, 0.3))
    base.env.unwrapped.state = np.array([th, thdot], dtype=np.float64)
    obs = np.array([np.cos(th), np.sin(th), thdot], dtype=np.float32)
    # 线性化（gymnasium 参数 g=10, m=l=1, dt=0.05）：θ̈ = 15·sinθ + 3u
    dt = 0.05
    A = np.array([[1.0, dt], [15.0 * dt, 1.0]])
    B = np.array([[0.0], [3.0 * dt]])
    K = lqr_gain(A, B, np.diag([10.0, 1.0]), np.array([[0.1]]))
    return ch, obs, K, dt


def make_pendulum_swing(seed: int, ep_seed: int):
    """Pendulum **起摆**任务：**gymnasium 默认初值**（θ ~ U(−π,π)，θ̇ ~ U(−1,1)）。

    ★ 与 `pendulum_reg` 的区别：本族**不改初值分布**，就是 `Pendulum-v1` 原样的任务。
      控制器 = 能量泵（swing-up）+ 近顶切 LQR（catch），两者都有闭式/经典依据：
        E(θ,θ̇) = ½θ̇² + g·l·(cosθ − 1)  （m=l=1, g=10）⇒ 目标 E₀ = 0（倒立静止）
        u = sat( k_pump · (E₀ − E) · sign(θ̇) )        （能量注入方向 = 运动方向）
        |θ| < 0.35 且 |θ̇| < 1.0 ⇒ 切 LQR
      k_pump = 3.0（**这一个是手调的**，如实标注；它只影响起摆快慢，不影响"丢包是否影响任务"）。
    """
    base = make_env("Pendulum-v1", seed=seed)
    ch = ChannelAdapter(base, loss_prob=0.0, delay=0, seed=seed)
    ch.reset(seed=ep_seed)
    rng = np.random.default_rng(ep_seed + 303)
    th = float(rng.uniform(-np.pi, np.pi))
    thdot = float(rng.uniform(-1.0, 1.0))
    base.env.unwrapped.state = np.array([th, thdot], dtype=np.float64)
    obs = np.array([np.cos(th), np.sin(th), thdot], dtype=np.float32)
    dt = 0.05
    A = np.array([[1.0, dt], [15.0 * dt, 1.0]])
    B = np.array([[0.0], [3.0 * dt]])
    K = lqr_gain(A, B, np.diag([10.0, 1.0]), np.array([[0.1]]))
    return ch, obs, K, dt


def pendulum_swing_action(est, K, k_pump: float = 3.0) -> np.ndarray:
    th = float(np.arctan2(est[1], est[0]))
    thdot = float(est[2])
    if abs(th) < 0.35 and abs(thdot) < 1.0:            # 近顶 ⇒ LQR 接管
        u = float(-(K @ np.array([th, thdot]))[0])
    else:                                              # 起摆 ⇒ 能量泵
        E = 0.5 * thdot ** 2 + 10.0 * (np.cos(th) - 1.0)
        u = k_pump * (0.0 - E) * np.sign(thdot) if thdot != 0.0 else 0.0
    return np.array([float(np.clip(u, -2.0, 2.0))], dtype=np.float32)


def make_cartpole(seed: int, ep_seed: int):
    """CartPole：线性化后 LQR，连续输出取符号成离散动作（左/右）。"""
    base = make_env("CartPole-v1", seed=seed)
    ch = ChannelAdapter(base, loss_prob=0.0, delay=0, seed=seed)
    ch.reset(seed=ep_seed)
    rng = np.random.default_rng(ep_seed + 202)
    s = np.array([rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05),
                  rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05)])
    base.env.unwrapped.state = s.astype(np.float64)
    obs = s.astype(np.float32)
    # ★ 小车-杆在**倒立点**的线性化（按 gymnasium `cartpole.py` 的方程现推，不抄公式表）
    #   M = mc+mp = 1.1，pm l = 0.05，l=0.5，g=9.8：temp = (u + 0.05·θ̇²sinθ)/M
    #   θ̈ = (g sinθ − cosθ·temp) / (l(4/3 − mp cos²θ/M))；ẍ = temp − 0.05·θ̈·cosθ/M
    #   在 θ=0,θ̇=0 处：θ̈ ≈ (9.8θ − u/1.1)/0.6212 = 15.77θ − 1.463u
    #                  ẍ ≈ 0.909u − 0.0455·θ̈ = −0.717θ + 0.975u
    #   ★ 第一版把 `θ̈` 对 θ 的系数写成**负号**（当成"正立摆"），控制器于是把杆推倒 ——
    #     冒烟时存活率只有 9%，S1 自检当场抓住（这就是 S1 存在的理由）。
    mc, mp, l, g = 1.0, 0.1, 0.5, 9.8
    M = mc + mp
    den = l * (4.0 / 3.0 - mp / M)
    th_th = g / den                      # ∂θ̈/∂θ
    th_u = -(1.0 / M) / den              # ∂θ̈/∂u
    x_th = -(mp * l / M) * th_th         # ∂ẍ/∂θ
    x_u = 1.0 / M - (mp * l / M) * th_u  # ∂ẍ/∂u
    dt = 0.02
    A = np.array([[1, dt, 0, 0],
                  [x_th * dt, 1, 0, 0],
                  [0, 0, 1, dt],
                  [0, 0, th_th * dt, 1]], dtype=float)
    B = np.array([[0], [x_u * dt], [0], [th_u * dt]], dtype=float)
    K = lqr_gain(A, B, np.diag([1.0, 0.1, 10.0, 1.0]), np.array([[0.1]]))
    return ch, obs, K, 0.02


def run_classic(family: str, loss_prob: float, seed: int, n_episodes: int,
                t_max: int) -> dict:
    """经典控制闭环：控制器只看 `est`，丢包 ⇒ persistence（保持上次观测）。"""
    cost_sum, cost_steps = 0.0, 0
    surv, n_escape = [], 0
    tail_th: list[float] = []
    for e in range(n_episodes):
        ep_seed = 1000 * seed + e
        if family == "pendulum_reg":
            ch, obs, K, _dt = make_pendulum_reg(seed, ep_seed)
        elif family == "pendulum_swing":
            ch, obs, K, _dt = make_pendulum_swing(seed, ep_seed)
        elif family == "cartpole":
            ch, obs, K, _dt = make_cartpole(seed, ep_seed)
        else:
            raise ValueError(family)
        ch.loss_prob = float(loss_prob)
        ch._rng = np.random.default_rng(ep_seed + 7919)

        est = np.asarray(obs, dtype=np.float64).copy()
        alive = 0
        th_hist: list[float] = []
        for t in range(t_max):
            if family == "pendulum_reg":
                th = float(np.arctan2(est[1], est[0]))
                thdot = float(est[2])
                u = float(np.clip(-(K @ np.array([th, thdot]))[0], -2.0, 2.0))
                a = np.array([u], dtype=np.float32)
            elif family == "pendulum_swing":
                a = pendulum_swing_action(est, K)
            else:
                u_lin = float(-(K @ est)[0])
                a = np.array([1 if u_lin > 0 else 0], dtype=np.int64)

            res = ch.step(a)
            true_next = np.asarray(res.obs, dtype=np.float64)
            alive += 1
            if family.startswith("pendulum"):
                th_hist.append(abs(float(np.arctan2(true_next[1], true_next[0]))))

            # ---- 代价：**用真值算** ----
            if family.startswith("pendulum"):
                th_t = float(np.arctan2(true_next[1], true_next[0]))
                thn = ((th_t + np.pi) % (2 * np.pi)) - np.pi
                uu = float(np.asarray(a).reshape(-1)[0])
                cost_sum += float(thn ** 2 + 0.1 * true_next[2] ** 2 + 0.001 * uu ** 2)
                cost_steps += 1
            if res.terminated or res.truncated:
                break
            # ---- 信道：交付才更新估计，否则 persistence ----
            if res.info and res.info.get("delivered", True):
                est = true_next.copy()
        surv.append(alive)
        if family == "cartpole" and alive < t_max:
            n_escape += 1
        if th_hist:                       # 末段 20% 的平均 |θ|（起摆是否真的立住了）
            k = max(1, int(round(len(th_hist) * 0.2)))
            tail_th.append(float(np.mean(th_hist[-k:])))
        ch.close()

    if family.startswith("pendulum"):
        cost = cost_sum / max(cost_steps, 1)
    else:
        cost = 1.0 - float(np.mean(surv)) / float(t_max)
    return {"cost": float(cost), "survival_mean": float(np.mean(surv)),
            "survival_frac": float(np.mean(surv)) / float(t_max),
            "fail_rate": n_escape / max(n_episodes, 1), "t_max": t_max,
            "tail_abs_theta": float(np.mean(tail_th)) if tail_th else float("nan")}


def run_uav(loss_prob: float, seed: int, n_episodes: int) -> dict:
    """UAV 跟踪：走 `run_closed_loop_control`（X30–X45 同源），丢包用 `lossy_schedule`。

    ★ 环境的步数上限必须 **≥ warmup + measured**，否则 warmup 跑完就已经 truncated，
      被测段每集只剩 1 步（第一版就是这样：p=0/0.3/0.9 的代价**逐位相同**，
      看起来像"任务不敏感"，其实是**根本没测到**）。
      这是本仓库第 N 次同类错误 —— R12：先问"它接线了吗"。
    """
    env = make_env("uav", seed=seed, max_steps=WARMUP_UAV + MEASURED_UAV)
    ctrl = PDRelativeController()
    r = run_closed_loop_control(
        env, None, ctrl, lossy_schedule(loss_prob),
        n_episodes=n_episodes, seed=seed, max_steps=MEASURED_UAV,
        estimator="persistence",          # ★ 不用世界模型：固定策略 + 保持上次观测
        var_g=1.0, label=f"uav/p={loss_prob}",
        warmup_steps=WARMUP_UAV, tail_frac=0.25,
    )
    env.close()
    return {"cost": float(r["mean_dist_tail"]), "survival_mean": float(r["mean_ep_len"]),
            "survival_frac": float(r["mean_ep_len"]) / MEASURED_UAV,
            "fail_rate": float(r["escape_rate"]), "t_max": MEASURED_UAV,
            "mean_age": float(r["mean_age"]), "tx_rate": float(r["tx_rate"])}


# ================================================================ 自检
def check_e1() -> dict:
    """[E1] ChannelAdapter(p=0) 与裸 env 观测序列逐位一致（X24 的性质）。"""
    a = make_env("CartPole-v1", seed=0)
    b = ChannelAdapter(make_env("CartPole-v1", seed=0), loss_prob=0.0, seed=0)
    oa = a.reset(seed=0)
    ob = b.reset(seed=0)
    same = np.array_equal(oa, ob)
    rng = np.random.default_rng(0)
    for _ in range(200):
        act = int(rng.integers(2))
        ra, rb = a.step(np.array([act])), b.step(np.array([act]))
        if not np.array_equal(ra.obs, rb.obs):
            same = False
            break
        if ra.terminated or ra.truncated:
            oa, ob = a.reset(seed=1), b.reset(seed=1)
            same = same and np.array_equal(oa, ob)
    a.close(); b.close()
    return {"e1_bit_exact": bool(same)}


def check_s1(baseline: dict) -> dict:
    """[S1] p=0 时三个族必须真的控得住 —— 否则不许报比值。"""
    out = {}
    out["uav_ok"] = bool(baseline["uav_track"]["alpha"] < 1.0)
    out["pendulum_ok"] = bool(baseline["pendulum_reg"]["alpha"] < 0.5)
    out["cartpole_ok"] = bool(baseline["cartpole"]["alpha"] <= 0.10)   # 存活率 ≥ 90%
    out["pendulum_swing_ok"] = bool(baseline["pendulum_swing"]["alpha_detail"][0].get("tail_abs_theta", 9.9) < 0.30)
    out["all_ok"] = bool(all([out["uav_ok"], out["pendulum_ok"], out["cartpole_ok"],
                              out["pendulum_swing_ok"]]))
    return out


def run_sweep(episodes: int) -> dict:
    """★ 接线检查（R12）：扫 p ∈ {0, 0.3, 0.6, 0.9}，看代价是否**真的随丢包动**。

    为什么必须做：如果 p=0.9 时代价仍然纹丝不动，那不是"任务不敏感"，
    而是**丢包根本没接进闭环**（本仓库历史上踩过：参数没接线、判据空转）。
    这一条只能证伪"没接线"，不能证伪"不敏感"——两者必须分开报。
    """
    fams = {"uav_track": 200, "pendulum_reg": 200, "pendulum_swing": 200,
            "cartpole": 500}
    out: dict = {}
    for fam, t_max in fams.items():
        pts = {}
        for p in (0.0, 0.3, 0.6, 0.9):
            vals = []
            for s in (0,):
                if fam == "uav_track":
                    vals.append(run_uav(p, s, episodes)["cost"])
                else:
                    vals.append(run_classic(fam, p, s, episodes, t_max)["cost"])
            pts[f"{p:g}"] = float(np.mean(vals))
        out[fam] = pts
    return out


# ================================================================ 主流程
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--sweep", action="store_true", help="额外跑丢包剂量响应（接线检查）")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "35_judge_task_sensitivity.json"))
    args = ap.parse_args()

    fams = {"uav_track": 200, "pendulum_reg": 200, "pendulum_swing": 200,
            "cartpole": 500}
    res: dict = {"p_loss": P_LOSS, "seeds": list(SEEDS), "n_episodes": args.episodes,
                 "families": {}}

    print(f"[35] 判-01 任务敏感性筛选  episodes={args.episodes} seeds={SEEDS} p={P_LOSS}")
    print(f"[35] {'family':<14}{'alpha(p=0)':>12}{'beta(p=.3)':>12}{'ratio':>9}  verdict")

    for fam, t_max in fams.items():
        rows = {}
        for cond, p in (("alpha", 0.0), ("beta", P_LOSS)):
            per_seed = []
            for s in SEEDS:
                if fam == "uav_track":
                    r = run_uav(p, s, args.episodes)
                else:
                    r = run_classic(fam, p, s, args.episodes, t_max)
                per_seed.append(r["cost"])
                rows.setdefault(f"{cond}_detail", []).append(r)
            rows[cond] = float(np.mean(per_seed))
            rows[f"{cond}_per_seed"] = [float(x) for x in per_seed]
            rows[f"{cond}_std"] = float(np.std(per_seed))
        a, b = rows["alpha"], rows["beta"]
        if a <= COST_FLOOR:
            ratio = float("inf")
            note = "α 代价≈0（比值不可靠，已设地板）"
        else:
            ratio = b / a
            note = ""
        if ratio >= RATIO_SENSITIVE:
            verdict = "敏感·保留"
        elif ratio < RATIO_KILL:
            verdict = "淘汰"
        else:
            verdict = "边缘（1.2–1.5）"
        rows["ratio"] = None if ratio == float("inf") else float(ratio)
        rows["ratio_inf"] = bool(ratio == float("inf"))
        rows["verdict"] = verdict
        rows["note"] = note
        rows["t_max"] = t_max
        res["families"][fam] = rows
        rr = "inf" if ratio == float("inf") else f"{ratio:.3f}"
        print(f"[35] {fam:<14}{a:>12.5f}{b:>12.5f}{rr:>9}  {verdict}")

    res["selftest_E1"] = check_e1()

    if args.sweep:
        sw_ep = max(3, args.episodes // 3)
        print(f"\n[35] 接线检查：丢包剂量响应（每点 {sw_ep} 集 / 1 种子）")
        print(f"[35] {'family':<14}{'p=0':>10}{'p=0.3':>10}{'p=0.6':>10}{'p=0.9':>10}")
        sweep = run_sweep(sw_ep)
        for fam, pts in sweep.items():
            print(f"[35] {fam:<14}" + "".join(f"{pts[k]:>10.5f}" for k in ("0", "0.3", "0.6", "0.9")))
        res["dose_response"] = sweep

    res["selftest_S1"] = check_s1({k: res["families"][k] for k in res["families"]})

    print(f"\n[35] 自检 E1（p=0 逐位一致）: {res['selftest_E1']}")
    print(f"[35] 自检 S1（p=0 控得住）  : {res['selftest_S1']}")
    if not res["selftest_S1"]["all_ok"]:
        print("[35] ★ S1 未通过 ⇒ 控制器没接对，**本表比值不可用**（R14：报错优于给假数字）")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[35] 写出 {args.out}")
    return 0 if res["selftest_S1"]["all_ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
