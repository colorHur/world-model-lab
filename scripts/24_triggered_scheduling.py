# -*- coding: utf-8 -*-
"""★ X38「触发式调度：等传输预算下，周期 / 年龄阈值 / 不确定性触发谁更好？」

================================================================= ★ 由来（为什么必须有这一号）
X34 / X35 / X36 隐含了同一个假设：**发送是周期的**（每 T 步发一次）。
而母论文那句话是 *"schedules sensing updates based on task urgency and
**predictive uncertainty**"* —— 说的是**调度策略本身**，不是"周期取多少"。
⇒ 周期只是**基线**。这一号测的是：把周期换成"按年龄/按不确定性触发"，在
**同样的传输预算**下能换来多少。

================================================================= ★★ 先在文献上钉死归属（不许抢功，也不许装作没人做过）
「采样率约束下最优采样是**阈值策略**、且显式优于 uniform（周期）采样」
已由 Sun–Polyanskiy–Uysal-Biyikoglu 解决（arXiv:1701.06734、arXiv:1707.02531，
IEEE 版本见 Sun 等后续）。他们也**明确比较过** uniform / zero-wait / age-optimal。
⇒ 本实验 **P1 / P2（周期 vs 年龄阈值）不是新贡献**，定位是
   「复现 + 移到**闭环控制 + 学习得到的世界模型**这个场景，拿一个实测倍数」。

真正的新问题在他们自己划的那条线上：**signal-independent vs signal-dependent**。
  · 周期（只看时钟）与年龄阈值（只看年龄）都是 **signal-independent**；
  · 世界模型自报的**累积不确定度 U** 是 **signal-dependent** 的触发量。
⇒ **P3 就是测：给定年龄 h 之后，U 还剩多少信息。** 这是本号的真正贡献点，
  也是母论文那句话里 "predictive uncertainty" 能不能落地的关键。

================================================================= ★ 预写判据（跑前定死）
P1  等平均发送率下，阈值策略 E[age] ≤ 周期策略；
    (a) PER=0 时两者**恒等**（都退化成确定周期 K+1）；
    (b) 比值随 PER 呈 **U 形** ⇒ 存在唯一的最大增益点。
    ★ 解析式已知 ⇒ 这是"拿已知答案验量级"的 R12 检查，不是"跑完再看图"。
P2  闭环上同 tx_rate 下，阈值策略 est_nmse ≤ 周期策略；任务指标同向。
    ⚠️ X35 的教训：Spearman(NMSE, 距离)=0.778 ⇒ **很可能不同向 ⇒ 照报**。
P3  ★ 核心：给定 age = h 后，累积不确定度 U 与实测误差的
    **条件 Pearson ρ_cond** 与 **条件 AUC**（U 预测「误差是否超阈」）。
    预写：ρ_cond ≥ 0.3 且 AUC_cond ≥ 0.6 ⇒ U 携带年龄以外的信息 ⇒ σ 触发有意义。
    若 ρ_cond < 0.1 且 AUC_cond < 0.6 ⇒ **U 只是 age 的替身** ⇒
    ★ **世界模型在调度问题上没有额外价值**（负面结论，照报并写进 README）。
P4  ★ P3 成立才做：把触发量换成 U（`utrigger`），三方在**同一 tx_rate** 上比。
    预写：U 触发 ≤ 年龄阈值（因为 U 携带了年龄以外的状态信息）。
    若 U 触发**不优于**年龄阈值 ⇒ 说明"σ 有信息"≠"σ 有决策价值" ⇒ 照报
    （这是两种不同的失败，不许混为一谈）。

================================================================= ★★ 口径（必须先说清，否则整套数都错）
`run_closed_loop_control` 的 `tx_rate` 记的是**送达**次数/步，不是**尝试**次数/步。
  · 周期：attempt = 1/T，delivery = s/T
  · 阈值：attempt = 1/(sK+1)，delivery = s/(sK+1)
两者只差常数因子 s ⇒ **"等尝试率"与"等送达率"给出同一个配对 T = s·K + 1**。
⇒ 配对不必纠结口径，但报数时必须写明是哪个（本文报的是 **tx_rate = 送达率**）。

================================================================= 自检（不过就抛 —— R14）
S_a  阈值闭式（E[age]、送达率）与纯调度 MC 一致（多副本，重尾 ⇒ 容差随 E[age] 放）
S_b  PER=0 时阈值(K) 的年龄 pmf 与周期(T=K+1) **逐位一致**（<1e-12）
S_c  K=0 时阈值 pmf 与 i.i.d. 几何（`periodic_age_pmf(p, T=1)`）**逐位一致**
S_d  等预算下 阈值 E[age] / 周期 E[age] ≤ 1（解析全网格扫描）
S_e  ★ R12：σ 头必须是**真有变化**的（跨样本 std/mean > 1%），否则 U 是常数、
     P3 必然得 0 —— 那种 0 是"没训练出来"，不是"σ 无用"，**必须分开报**
S_f  全部有限（R14）；尾部质量 > 2% 的点直接剔除（与 X34/X35/X36 同口径）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from wmlab.control import (PDRelativeController, collect_controlled_episodes,
                           run_closed_loop_control)
from wmlab.data import split_episodes, transitions_from_episodes
from wmlab.envs import make_env
from wmlab.eval.conformal import coverage as conf_coverage
from wmlab.eval.conformal import envelope as conf_envelope
from wmlab.eval.conformal import fit_envelope
from wmlab.eval.pairing import (WindowNotEvaluable, expo_grid,  # noqa: F401
                               overlap_window)
from wmlab.eval.tracking import (periodic_age_pmf, periodic_age_tail,
                                 periodic_lossy_schedule, periodic_mean_age,
                                 threshold_age_pmf, threshold_age_tail,
                                 threshold_delivery_rate, threshold_equivalent_period,
                                 threshold_lossy_schedule, threshold_mean_age)
from wmlab.models.prob_world_model import GaussianWorldModel
from wmlab.train import train_world_model
from wmlab.utils import count_params, get_device, load_config, output_dir, set_seed
from wmlab.utils.plot import PALETTE, apply_style, save_fig


# ★★ 等预算配对的**公共守卫**（2026-09-29 修正，两处配对路径共用）
#   (1) `diverged` 必须对**所有被比较的族**统一施加
#       （原先只有 P4 路径筛、P2 路径不筛 ⇒ 口径不对称；离线复查：X38 的 P2 一字不变，
#        wind=4 的 P2 有 6/10 行小幅变化 0.827→0.813 / 0.618→0.642 / 0.445→0.475）
#   (2) 配对窗口必须够宽：原守卫 `hi > lo*1.0001` **太松** ——
#       X40 的 PER=0.7 档靠 `[0.1502, 0.1584]`（**5% 速率跨度**）造出了
#       「9/9 更优」的**假头条**（9 个网格点全挤在同一段 ⇒ 不是 9 个独立点）。
MIN_RATE_RANGE = 1.5      # 窗口 hi/lo 下限
MIN_FAMILY_PTS = 3        # 每族至少几个可比点


def _sim_hybrid(K: int, tau: float, per: float, U_pool: np.ndarray,
                u_hat: np.ndarray, u_scale: np.ndarray,
                n: int, seed: int) -> tuple[float, float]:
    """★ S_h：混合触发的**纯调度**仿真（U 从离线条件分布**重采样**，不含控制/环境）。

    用途只有一个：把三条退化逐位钉死（R12「拿已知答案验量级」）——
      τ=+∞ ⇒ 恒等纯年龄阈值 `threshold_lossy_schedule(K, per)`
      K=1   ⇒ 恒等 T=1（每步都发）
      τ=−∞  ⇒ 恒等 T=1
    ⚠️ U 逐步独立重采样 ⇒ **忽略了 U 的自相关** ⇒ 本仿真只用于退化检查，
      **不得用于报"混合策略的 E[age]"**（那要真闭环，见主流程）。
    """
    rng = np.random.default_rng(seed)
    p = float(min(max(per, 0.0), 1.0))
    n_h = int(u_hat.shape[0])
    age = 0
    tot = 0.0
    ntx = 0
    for t in range(1, n + 1):
        U = 0.0
        if age > 0:
            U = float(U_pool[rng.integers(U_pool.shape[0]),
                             min(max(age, 1), n_h) - 1])
        fire = age >= int(K)
        if not fire:
            h = min(max(age, 1), n_h)
            fire = ((U - float(u_hat[h - 1])) / float(u_scale[h - 1])) >= float(tau)
        if fire and rng.random() >= p:
            age = 0
            ntx += 1
        else:
            age += 1
        tot += age
    return tot / n, ntx / n


def parse_args():
    p = argparse.ArgumentParser(description="X38 触发式调度：等预算下周期 vs 阈值 vs 不确定性")
    p.add_argument("--config", default="configs/uav_triggered.yaml")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--tag", default="24_triggered_scheduling")
    p.add_argument("--pers", default=None)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--p3-only", action="store_true",
                   help="★ X38-b：只跑「训练 + P1 解析 + P3 条件信息量」，"
                        "跳过耗时的闭环 P2/P4 网格（约从 16 min 降到 3 min）")
    p.add_argument("--wind-amp", type=float, default=None,
                   help="★ X38-b：状态依赖阵风场幅度（0 = 逐位退化为原同方差环境）")
    p.add_argument("--pol", default="utrigger",
                   choices=["utrigger", "resid", "hybrid", "conformal"],
                   help="★ X40/X42：不确定性触发的**触发量**。"
                        "utrigger = 全局阈值 U≥thr（X38/X39）；"
                        "resid = 纯残差 z≥τ（诊断用，会撤掉安全网）；"
                        "hybrid = age≥K 或 z≥τ（决定性：残差只准加速）；"
                        "conformal = age≥K 或 Ê_α(age,U)≥tol（★ X42：共形校准的误差信封）")
    p.add_argument("--conf-target", default="pos", choices=["pos", "nmse"],
                   help="★ X42：共形信封标定的**误差量**。"
                        "nmse = 全维归一化 NMSE（与 X38/X40 的估计误差同口径）；"
                        "pos = **位置维**归一化误差（= 控制器实际吃进去的那个量，任务对齐）")
    p.add_argument("--conf-alpha", type=float, default=0.1,
                   help="★ X42：共形分位数水平 α（覆盖率目标 1−α）")
    return p.parse_args()


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


# ------------------------------------------------------------------ 小工具
def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 3 or float(a.std()) < 1e-15 or float(b.std()) < 1e-15:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _auc(score: np.ndarray, label: np.ndarray) -> float:
    """Mann–Whitney 形式的 AUC（不依赖 sklearn）。样本不足/单类 ⇒ NaN。"""
    s = np.asarray(score, dtype=float)
    y = np.asarray(label, dtype=bool)
    npos = int(y.sum())
    nneg = int((~y).sum())
    if npos < 5 or nneg < 5:
        return float("nan")
    order = np.argsort(np.argsort(s))          # 平均秩（并列用 argsort 两次 ≈ 打散）
    ranks = order.astype(float) + 1.0
    return float((ranks[y].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def _sample_windows(episodes, horizon: int, n_samples: int, seed: int, t0_min: int = 0):
    """抽 (obs_t, actions[t:t+H], obs[t+1:t+H+1]) 窗口。返回 numpy 三元组。"""
    rng = np.random.default_rng(seed)
    starts = []
    for ei, ep in enumerate(episodes):
        L = int(ep["length"])
        hi = L - horizon - 1
        if hi <= t0_min:
            continue
        starts.append((ei, hi))
    if not starts:
        raise AssertionError("★ 没有可用的 rollout 窗口（episode 太短）")
    obs0, acts, tgts = [], [], []
    for _ in range(int(n_samples)):
        ei, hi = starts[int(rng.integers(len(starts)))]
        t0 = int(rng.integers(t0_min, hi))
        ep = episodes[ei]
        obs0.append(ep["obs"][t0])
        acts.append(ep["act"][t0:t0 + horizon])
        tgts.append(ep["obs"][t0 + 1:t0 + horizon + 1])
    return (np.asarray(obs0, dtype=np.float32),
            np.asarray(acts, dtype=np.float32),
            np.asarray(tgts, dtype=np.float32))


@torch.no_grad()
def _rollout_sigma(model, obs0, acts, device):
    """批量 rollout，逐步返回 (NMSE_h, U_h)。U = sqrt(Σ_{i≤h} mean_d σ_i²)。

    ★ 用**均值**传播（与 `GaussianWorldModel.next_latent` 一致），
      σ 只是**伴随量**，不参与状态演化 ⇒ 与 X30/X35 的 rollout 口径严格一致。
    """
    o = torch.as_tensor(obs0, dtype=torch.float32, device=device)
    a = torch.as_tensor(acts, dtype=torch.float32, device=device)
    z = model.encode(o)
    H = int(a.shape[1])
    U2 = torch.zeros(o.shape[0], dtype=torch.float32, device=device)
    Us = []
    for h in range(H):
        mu, sd, _ = model.next_latent_dist(z, a[:, h])
        U2 = U2 + sd.pow(2).mean(dim=-1)
        z = mu
        Us.append(torch.sqrt(U2).detach().cpu().numpy())
    return np.stack(Us, axis=1)      # (N, H)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.seed is not None:
        cfg["seed"] = args.seed
    dz = cfg["design"]
    if args.pers:
        dz["per_list"] = [float(x) for x in args.pers.split(",")]
    if args.quick:
        cfg["train"]["epochs"] = min(int(cfg["train"]["epochs"]), 40)
        cfg["data"]["n_episodes"] = min(int(cfg["data"]["n_episodes"]), 40)
        dz["per_list"] = [0.0, 0.3, 0.7]
        dz["period_list"] = [1, 2, 4, 8]
        dz["threshold_list"] = [0, 1, 3, 8]
        cfg["task"]["n_episodes"] = 6
        cfg["eval"]["n_samples"] = 512
        dz["u_h_star_list"] = [1, 2, 4, 8]
        dz["probe_horizon"] = 16

    seed = int(cfg["seed"])
    set_seed(seed)
    apply_style()
    device = get_device(args.device or cfg.get("device"))
    out = output_dir(cfg)
    t0 = time.time()

    # ★ X40：触发量族名（"utrigger" / "resid"）。三族比较的第三族随此切换。
    UT = str(args.pol)

    pers = [float(x) for x in dz["per_list"]]
    T_list = [int(x) for x in dz["period_list"]]
    K_list = [int(x) for x in dz["threshold_list"]]
    max_tail = float(dz.get("max_tail_mass", 0.02))
    WARMUP = int(cfg["task"]["warmup_steps"])
    n_task_ep = int(cfg["task"]["n_episodes"])

    print("[24] " + "=" * 78)
    print("[24] ★ X38/X40 触发式调度：等传输预算下 周期 / 年龄阈值 / 不确定性触发")
    _pol_desc = {"resid": "纯残差 z≥τ，先条件掉 age（会撤掉安全网）",
                 "hybrid": "混合 age≥K 或 z≥τ（残差只准加速，安全网 K 不动）"}.get(
                     UT, "全局阈值 U≥thr")
    print(f"[24]   触发量族 = {UT}（{_pol_desc}）")
    print(f"[24]   PER ∈ {pers}    T ∈ {T_list}    K ∈ {K_list}")
    print("[24]   预写：P1 等预算阈值 E[age] ≤ 周期（PER=0 恒等、U 形）"
          "  P2 闭环同向  P3 ρ_cond≥0.3 且 AUC_cond≥0.6")

    # ============================================================ 1) 训练
    # ★ X38-b：阵风场幅度（状态依赖噪声）。优先级：命令行 > 配置 > 0（同方差）。
    wind_amp = float(args.wind_amp if args.wind_amp is not None
                     else cfg["env"].get("wind_amp", 0.0))
    env = make_env(cfg["env"]["id"], seed=seed,
                   noise_std=float(cfg["env"]["noise_std"]),
                   max_steps=int(cfg["env"]["max_steps"]),
                   wind_amp=wind_amp)
    print(f"[24]   环境：{env.name}（wind_amp={wind_amp:g}"
          + (" ⇒ **异方差**，X38-b 边界档" if wind_amp > 0 else " ⇒ 同方差，X38 原档")
          + "）")
    cc = cfg["controller"]
    ctrl = PDRelativeController(dt=env.dt, omega_n=float(cc["omega_n"]),
                                zeta=float(cc["zeta"]), kappa=float(env.kappa),
                                a_max=float(env.a_max))
    n_ep = int(cfg["data"]["n_episodes"])
    train_all = collect_controlled_episodes(env, ctrl, n_episodes=n_ep, seed=seed,
                                            max_steps=int(cfg["env"]["max_steps"]))
    train_eps, val_eps = split_episodes(train_all, float(cfg["train"]["val_ratio"]), seed)
    tr = tuple(torch.as_tensor(x) for x in transitions_from_episodes(train_eps))
    va = tuple(torch.as_tensor(x) for x in transitions_from_episodes(val_eps))
    obs_dim = int(train_all[0]["obs"].shape[1])
    model = GaussianWorldModel(
        obs_dim=obs_dim, act_dim=int(env.act_dim),
        latent_dim=int(cfg["model"]["latent_dim"]),
        hidden=int(cfg["model"]["hidden"]),
        discrete_act=bool(env.is_discrete),
        logvar_init=float(cfg["model"].get("logvar_init", 0.0))).to(device)
    hist = train_world_model(model, tr, va, cfg, device, verbose=False)
    print(f"[24] 训练完成：params={count_params(model):,} "
          f"val={hist['val_total'][-1]:.6f} ({time.time() - t0:.0f}s)")

    ev_seed = seed + int(cfg["data"]["eval_seed_offset"])
    ev_env = make_env(cfg["env"]["id"], seed=ev_seed,
                      noise_std=float(cfg["env"]["noise_std"]),
                      max_steps=int(cfg["env"]["max_steps"]),
                      wind_amp=wind_amp)
    eval_eps = collect_controlled_episodes(ev_env, ctrl,
                                           n_episodes=max(30, n_ep // 3), seed=ev_seed,
                                           max_steps=int(cfg["env"]["max_steps"]))
    ev_env.close()
    allobs = np.concatenate([e["obs"][WARMUP:] for e in eval_eps], axis=0).astype(np.float64)
    var_g = float(allobs.var())

    # ============================================================ 2) Part A：解析 P1
    print("[24] " + "-" * 78)
    print("[24] Part A · P1（解析 + 纯调度 MC）：等预算下 阈值 vs 周期 的 E[age]")
    KC = int(dz.get("max_age_K", 40))
    p1_rows = []
    for per in pers:
        s = 1.0 - per
        for K in K_list:
            T_eq = threshold_equivalent_period(K, per)
            a_thr = threshold_mean_age(K, per)
            a_per = periodic_mean_age(per, T_eq)
            r_ana = threshold_delivery_rate(K, per)
            # ★ S_a：纯调度空跑（不含控制/环境）复核闭式
            a_sim, r_sim = _sim_threshold(K, per, n=200000, seed=seed + 1000 + K)
            tol = max(0.05, 0.05 * a_thr)
            if abs(a_thr - a_sim) > tol:
                raise AssertionError(
                    f"★ S_a 未通过：K={K} PER={per} 阈值 E[age] 闭式 {a_thr:.4f} "
                    f"vs 仿真 {a_sim:.4f}（容差 ±{tol:.4f}）")
            if abs(r_ana - r_sim) > max(0.01, 0.05 * r_ana):
                raise AssertionError(
                    f"★ S_a 未通过：送达率闭式 {r_ana:.5f} vs 仿真 {r_sim:.5f}")
            # ★ S_d：等预算下阈值不得劣于周期
            if a_thr > a_per * (1.0 + 1e-9):
                raise AssertionError(
                    f"★ S_d 未通过：K={K} PER={per} 阈值 E[age] {a_thr:.4f} > "
                    f"等预算周期 {a_per:.4f}")
            p1_rows.append({"per": per, "K": K, "T_eq": float(T_eq),
                            "age_threshold": float(a_thr),
                            "age_periodic": float(a_per),
                            "ratio": float(a_thr / a_per) if a_per > 1e-12 else None,
                            "delivery_rate": float(r_ana),
                            "age_sim": float(a_sim), "rate_sim": float(r_sim)})
    # ★ S_b / S_c：退化必须逐位一致
    for K in (0, 1, 3, 8, 16):
        d = float(np.abs(threshold_age_pmf(K, 0.0, KC)
                         - periodic_age_pmf(0.0, K + 1, KC)).max())
        if d > 1e-12:
            raise AssertionError(f"★ S_b 未通过：PER=0 K={K} pmf 与周期 T={K+1} 差 {d:.3e}")
    for per in (0.2, 0.5, 0.9):
        d = float(np.abs(threshold_age_pmf(0, per, KC)
                         - periodic_age_pmf(per, 1, KC)).max())
        if d > 1e-12:
            raise AssertionError(f"★ S_c 未通过：K=0 PER={per} pmf 与 i.i.d. 差 {d:.3e}")
    print("[24]   S_b（PER=0 ⇒ 阈值 ≡ 周期 T=K+1）与 "
          "S_c（K=0 ⇒ ≡ i.i.d.）逐位一致 ✓")

    gain = {}
    for per in pers:
        sub = [r for r in p1_rows if abs(r["per"] - per) < 1e-12 and r["ratio"] is not None]
        if sub:
            best = min(sub, key=lambda r: r["ratio"])
            gain[per] = best
            print(f"[24]   PER={per:<4g} 最大增益 @K={best['K']:<3d} "
                  f"(等预算 T={best['T_eq']:.2f})：E[age] {best['age_threshold']:.3f} "
                  f"vs 周期 {best['age_periodic']:.3f}  ⇒ 比值 {best['ratio']:.3f}")
    p1_verdict = {"PER=0 恒等": (None if 0.0 not in gain else gain[0.0]),
                  "最佳增益": {str(k): v for k, v in gain.items()}}

    # ============================================================ ★★ P1(b) 形状：细网格解析扫描
    # ★★ 第 19 次自证伪的直接产物：闭环网格只有 6 档 PER（最右 0.9）且 K ≤ 32，
    #    在这个**截断网格**上比值看起来"单调下降"，我据此写了"不是 U 形、最优在边界"。
    #    但解析细网格显示：拐点就在 PER≈0.88–0.96（取决于 K 上限）⇒ 我的网格**恰好卡在拐点左边**。
    #    ⇒ 形状这种结论**必须**在解析式上用细网格扫（免费），不许从 6 个闭环点的趋势推。
    print("[24]   ★ P1(b) 形状复检（解析细网格，PER 0.02–0.98 共 49 档）：")
    qs = np.linspace(0.02, 0.98, 49)
    shape_rows = []
    for Kmax in (max(K_list), 64, 200):
        m_by_q = []
        for q in qs:
            m = 1.0
            kbest = 0
            for K in range(0, int(Kmax) + 1):
                T = threshold_equivalent_period(K, q)
                rr = threshold_mean_age(K, q) / max(periodic_mean_age(q, T), 1e-12)
                if rr < m:
                    m, kbest = rr, K
            m_by_q.append((float(m), kbest))
        arr = np.array([x[0] for x in m_by_q])
        k = int(np.argmin(arr))
        interior = bool(0 < k < len(arr) - 1)
        shape_rows.append({"K_max": int(Kmax), "q_star": float(qs[k]),
                           "ratio_star": float(arr[k]),
                           "ratio_at_q02": float(arr[0]), "ratio_at_q98": float(arr[-1]),
                           "K_star": int(m_by_q[k][1]), "interior": interior})
        print(f"[24]     K≤{int(Kmax):<4d} 最小比值 {arr[k]:.4f} @PER={qs[k]:.2f} "
              f"(K*={m_by_q[k][1]})｜两端 {arr[0]:.4f} / {arr[-1]:.4f} ⇒ "
              + ("**U 形（拐点在内部）**" if interior else "单调（最优在边界）"))
    if all(r["interior"] for r in shape_rows):
        print("[24]     ⇒ ✅ P1(b) **U 形成立**（三档 K 上限全部给出内部拐点）"
              "；闭环网格只覆盖到 PER=0.9 ⇒ 看到的是下降支，不能据此否定 U 形。")
    else:
        print("[24]     ⇒ ❌ P1(b) U 形不成立（至少一档 K 上限的最优落在端点）")
    p1_verdict["shape"] = shape_rows

    # ★ X38-b：--p3-only 时整段 Part B 跳过（16 min -> 3 min）
    if not args.p3_only:
        # ============================================================ 3) Part B：闭环 P2
        print("[24] " + "-" * 78)
        print("[24] Part B · P2（真闭环）：同 tx_rate 下比 est_nmse 与任务指标")
        online_seed = seed + int(cfg["data"].get("online_seed_offset", 2000))
        meas_steps = int(cfg["task"]["measured_steps"])
        rows = []
        for per in pers:
            for T in T_list:
                tail_mass = float(periodic_age_tail(per, T, KC))
                if tail_mass > max_tail:
                    continue
                r = run_closed_loop_control(
                    env, model, ctrl, periodic_lossy_schedule(T, per),
                    n_episodes=n_task_ep, seed=online_seed,
                    max_steps=meas_steps, device=device, estimator="model",
                    var_g=var_g, label=f"periodic/T={T}",
                    tail_frac=float(cfg["task"]["tail_frac"]), warmup_steps=WARMUP)
                rows.append(_mkrow("periodic", per, T, r, tail_mass, KC,
                                   _age_mc("periodic", per, T, n_task_ep, meas_steps), meas_steps=meas_steps))
            for K in K_list:
                tail_mass = float(threshold_age_tail(K, per, KC))
                if tail_mass > max_tail:
                    continue
                r = run_closed_loop_control(
                    env, model, ctrl, threshold_lossy_schedule(K, per),
                    n_episodes=n_task_ep, seed=online_seed,
                    max_steps=meas_steps, device=device, estimator="model",
                    var_g=var_g, label=f"threshold/K={K}",
                    tail_frac=float(cfg["task"]["tail_frac"]), warmup_steps=WARMUP)
                rows.append(_mkrow("threshold", per, K, r, tail_mass, KC,
                                   _age_mc("threshold", per, K, n_task_ep, meas_steps), meas_steps=meas_steps))
        for r in rows:
            ana = r["age_analytic"]
            print(f"[24]   PER={r['per']:<4g} {r['policy']:<9s} 旋钮={r['knob']:<3d} "
                  f"tx_rate={r['tx_rate']:.4f} E[age]={r['mean_age']:6.2f} "
                  f"(MC {r['age_mc']:6.2f}" +
                  (f"/闭式 {ana:6.2f})" if np.isfinite(ana) else ")      ") +
                  f" | NMSE={r['est_nmse']:.5f} "
                  f"距离={r['mean_dist_tail']:6.3f}m 逃逸={r['escape_rate']:.3f}")

        # ★ S_f 与年龄闭式在闭环里也要成立（容差用 3×SE，与 X35 同口径）
        chk = [r for r in rows if np.isfinite(r["age_dev_sigma"])]
        devs = np.array([r["age_dev_sigma"] for r in chk], dtype=float)
        wb = np.array([r["age_window_bias"] for r in chk
                       if r["age_window_bias"] is not None], dtype=float)
        print(f"[24]   S_f/年龄：{len(chk)}/{len(rows)} 点可比（utrigger 无闭式 ⇒ 不计），"
              f"|偏差| 中位 {np.median(np.abs(devs)):.2f}σ 最大 {np.abs(devs).max():.2f}σ")
        if wb.size:
            print(f"[24]     有限窗口偏差（MC−解析）中位 {np.median(wb):+.4f} 步 "
                  f"最大 {np.abs(wb).max():.4f} ⇒ 这是偏差不是噪声，已用 MC 作参照抵消")
        # ★ 删失点（episode 提前终止 ⇒ 长年龄被系统性切掉）只可能把均值**拉低**
        #   ⇒ 只做单侧断言（X35 已确立的口径）：负向不报警，正向仍报警。
        n_cens = sum(1 for r in chk if r["censored"])
        if n_cens:
            print(f"[24]     删失点 {n_cens}/{len(chk)}（ep_frac<0.98）⇒ 只做单侧断言")
        bad = [r for r in chk
               if (abs(r["age_dev_sigma"]) > 3.0
                   if not r["censored"] else r["age_dev_sigma"] > 3.0)]
        for r in bad:
            print(f"[24]     ⚠ 超 3σ：{r['policy']} PER={r['per']} "
                  f"旋钮={r['knob']} {r['age_dev_sigma']:+.2f}σ "
                  f"(实测 {r['mean_age']:.3f} vs MC {r['age_mc']:.3f}"
                  f"{'，删失点单侧' if r['censored'] else ''})")
        print(f"[24]   S_f/年龄结论：{len(chk) - len(bad)}/{len(chk)} 点通过"
              f"{'（全部通过 ⇒ 两种策略的年龄闭式在闭环里都成立）' if not bad else ''}")

        # --- 等 tx_rate 插值比较 ---
        cmp_rows = []
        for per in pers:
            pp = sorted([r for r in rows if r["policy"] == "periodic"
                         and abs(r["per"] - per) < 1e-12
                         and not r.get("diverged", False)], key=lambda r: r["tx_rate"])
            tt = sorted([r for r in rows if r["policy"] == "threshold"
                         and abs(r["per"] - per) < 1e-12
                         and not r.get("diverged", False)], key=lambda r: r["tx_rate"])
            try:
                lo, hi = overlap_window(
                    {"periodic": [r["tx_rate"] for r in pp],
                     "threshold": [r["tx_rate"] for r in tt]},
                    min_range=MIN_RATE_RANGE, min_pts=MIN_FAMILY_PTS)
            except WindowNotEvaluable as e:
                # ★ 不许静默跳过：把原因打出来（静默会被读成"这一档没问题"）
                print(f"[24]   ⚠ P2 该档不可评估：PER={per:g} —— {e}")
                continue
            grid = np.array(expo_grid(lo, hi, 9))
            for met, better in (("est_nmse", "low"), ("mean_dist_tail", "low")):
                xp = np.array([r["tx_rate"] for r in pp])
                yp = np.array([r[met] for r in pp])
                xt = np.array([r["tx_rate"] for r in tt])
                yt = np.array([r[met] for r in tt])
                if np.any(~np.isfinite(yp)) or np.any(~np.isfinite(yt)):
                    continue
                vp = np.interp(grid, xp, yp)
                vt = np.interp(grid, xt, yt)
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = np.where(np.isfinite(vp) & (vp > 1e-18), vt / np.maximum(vp, 1e-18),
                                     np.nan)
                ratio = ratio[np.isfinite(ratio)]
                if ratio.size == 0:
                    continue
                cmp_rows.append({"per": per, "metric": met, "n_grid": int(ratio.size),
                                 "rate_lo": float(lo), "rate_hi": float(hi),
                                 "ratio_median": float(np.median(ratio)),
                                 "ratio_min": float(ratio.min()),
                                 "ratio_max": float(ratio.max()),
                                 "n_threshold_better": int((ratio < 1.0).sum())})
        for c in cmp_rows:
            print(f"[24] ★ P2[{c['metric']}] PER={c['per']:<4g} "
                  f"阈值/周期 中位 {c['ratio_median']:.3f} "
                  f"（{c['n_threshold_better']}/{c['n_grid']} 个率点上阈值更优，"
                  f"区间 {c['rate_lo']:.3f}–{c['rate_hi']:.3f}）")

    # ============================================================ 4) Part C：P3 条件信息量
    print("[24] " + "-" * 78)
    print("[24] Part C · P3（核心）：给定 age=h 后，累积不确定度 U 还剩多少信息？")
    H = int(dz.get("probe_horizon", 32))
    n_samples = int(cfg["eval"]["n_samples"])
    p3_extra: dict = {}
    o0, acts, tgts = _sample_windows(eval_eps, H, n_samples, seed=seed + 7,
                                     t0_min=WARMUP)
    U = _rollout_sigma(model, o0, acts, device)                       # (N, H)
    with torch.no_grad():
        pred = []
        z = model.encode(torch.as_tensor(o0, dtype=torch.float32, device=device))
        a_t = torch.as_tensor(acts, dtype=torch.float32, device=device)
        for h in range(H):
            z = model.next_latent(z, a_t[:, h])
            pred.append(model.decode(z).detach().cpu().numpy())
    pred = np.stack(pred, axis=1)                                     # (N, H, obs_dim)
    err = ((pred - tgts) ** 2).mean(axis=2) / max(var_g, 1e-18)       # (N, H)

    # ★ S_e：σ 头必须真有变化（否则 U 是常数，P3 的 0 是"没训出来"不是"σ 无用"）
    u_flat = U.reshape(-1)
    u_cv = float(u_flat.std() / max(u_flat.mean(), 1e-18))
    sigma_step = U[:, 0]
    s_cv = float(sigma_step.std() / max(sigma_step.mean(), 1e-18))
    print(f"[24]   S_e：U 的变异系数 CV={u_cv:.4f}（单步 σ 的 CV={s_cv:.4f}）")
    if u_cv < 0.01:
        print("[24]   ★★ S_e 未通过：U 几乎是常数 ⇒ P3 的任何结果都是"
              "「没训练出来」，不是「σ 无用」。本号到此为止，不许拿 P3 下结论。")

    # ============================================================ ★ X38-b：σ 头**校准**的直接检验
    # ★★ X38 的 P3 只给了"U 与误差的相关"。但"U 只是 age 的替身"有两个可能来源：
    #   (a) 环境本身**同方差** ⇒ 根本没什么可学（X38 那 7.1% 可能就是这个）
    #   (b) 环境是异方差的，但 **σ 头没学会**状态依赖
    # 区分办法：拿**环境 ground-truth 的单步 σ**（`env.sigma_at(pos)`，X38-b 新增；
    # wind_amp=0 时恒等于 noise_std ⇒ 逐位退化）与模型预测的单步 σ 直接比。
    gt_fn = getattr(env, "sigma_at", None)
    if callable(gt_fn):
        s_true = np.asarray([float(gt_fn(np.asarray(o[:2], dtype=np.float64)))
                             for o in o0], dtype=np.float64)
        cv_true = float(s_true.std() / max(s_true.mean(), 1e-18))
        # ★ R14：真实 σ 为常数时相关系数**无定义** ⇒ 报 NaN 而不是报 0（0 会被读成"完全不相关"）
        rho_cal = (_pearson(sigma_step.astype(np.float64), s_true)
                   if cv_true > 1e-12 else float("nan"))
        rho_txt = (f"{rho_cal:+.4f}" if np.isfinite(rho_cal)
                   else "NaN（真实 σ 为常数 ⇒ 相关系数无定义）")
        print(f"[24]   ★ X38-b σ 校准：环境真实单步 σ 的 CV={cv_true:.4f}"
              f"（wind_amp={float(getattr(env, 'wind_amp', 0.0)):g}）"
              f"｜模型预测 σ 的 CV={s_cv:.4f}"
              f"｜**ρ(σ_pred, σ_true) = {rho_txt}**")
        if cv_true < 0.01:
            print("[24]     ⇒ 本环境**同方差**（真实 σ 无变化）⇒ σ 头的 CV 只可能来自拟合噪声"
                  "；X38 的 P3 **必须**加「本场景」限定。")
        elif rho_cal >= 0.3:
            print("[24]     ⇒ ★ σ 头**真的学到了**状态依赖（能预报真实 σ）"
                  "⇒ 若此时 ρ_cond 仍 ≈0，则负面结论**可以**推广。")
        else:
            print("[24]     ⇒ ⚠ 环境**是**异方差的，但 σ 头**没学出来**（ρ_calib < 0.3）"
                  "⇒ X38 的 P3 只能归因为「没训出来」，**不许**推广。")
        p3_extra["sigma_calib"] = {"cv_true": cv_true, "cv_pred": float(s_cv),
                                   "rho_calib": float(rho_cal),
                                   "wind_amp": float(getattr(env, "wind_amp", 0.0))}

    thr = float(cfg["eval"]["err_threshold"])
    rho_all = _pearson(U.reshape(-1), err.reshape(-1))
    per_h = []
    for h in range(H):
        u_h = U[:, h]
        e_h = err[:, h]
        rho_h = _pearson(u_h, e_h)
        auc_h = _auc(u_h, e_h > thr)
        # ★★ 必须同时报**条件内变异**：ρ_cond≈0 有两种完全不同的解释 ——
        #   (a) U 几乎是 h 的确定性函数（CV_cond 极小）⇒ "U 是 age 的替身"；
        #   (b) U 在给定 h 后仍有可观方差，但**与误差无关** ⇒ 更强的负面结论
        #       （模型自报的不确定度本身就没校准好）。两者不能混为一谈 ⇒ 都记。
        u_sd = float(np.std(u_h))
        per_h.append({"h": h + 1, "rho": rho_h, "auc": auc_h,
                      "err_mean": float(e_h.mean()),
                      "u_mean": float(u_h.mean()), "u_std": u_sd,
                      "u_cv_cond": float(u_sd / max(float(u_h.mean()), 1e-12)),
                      "n_pos": int((e_h > thr).sum()), "n": int(e_h.size)})
    rhos = np.array([x["rho"] for x in per_h], dtype=float)
    aucs = np.array([x["auc"] for x in per_h], dtype=float)
    ok_r = rhos[np.isfinite(rhos)]
    ok_a = aucs[np.isfinite(aucs)]
    rho_cond = float(np.median(ok_r)) if ok_r.size else float("nan")
    auc_cond = float(np.median(ok_a)) if ok_a.size else float("nan")
    print(f"[24] ★ P3：不条件 ρ(U, err) = {rho_all:+.3f}   "
          f"⇒ 给定 age=h 后 条件 ρ 中位 = {rho_cond:+.3f}、条件 AUC 中位 = {auc_cond:.3f}")
    cvs = np.array([x["u_cv_cond"] for x in per_h], dtype=float)
    # ★★ 判定"U 是不是 age 的替身"**不许**用「CV_cond vs 0.5×CV(U)」这种拍系数
    #    （第 20 次自证伪）：U 是 h 步累积量 ⇒ CV(U) 被 h 那一维撑大，
    #    0.5 这个阈值在 wind_amp=4 档把 CV_cond=0.539 也判成"接近常数"（实际并不）。
    #    正确的参照是**单步 σ 的 CV**：条件后剩下的若 ≈ 单步 σ 的抖动 ⇒ 真的退化；
    #    明显大于 ⇒ U 里确实累积了状态依赖成分。
    keep = float(np.median(cvs)) / max(u_cv, 1e-18)
    vs_step = float(np.median(cvs)) / max(s_cv, 1e-18)
    print(f"[24]   条件内 U 变异：CV_cond 中位 {np.median(cvs):.4f}"
          f"（不条件 CV={u_cv:.4f} ⇒ 保留 {keep:.3f}；"
          f"单步 σ 的 CV={s_cv:.4f} ⇒ CV_cond/单步 = {vs_step:.2f}×）"
          + ("★ U 在给定 h 后**没有**退化（残余变异 > 单步 σ 的抖动）"
             if vs_step > 1.2 else
             "U 在给定 h 后基本退化为 h 的函数（残余变异 ≈ 单步 σ 的抖动）"))
    # ★ 最终判定交给**直接校准量** ρ(σ_pred, σ_true)，不再由这条启发式下结论
    if np.isfinite(rho_cond) and abs(rho_cond) < 0.1 and np.isfinite(auc_cond) \
            and auc_cond < 0.6:
        print("[24]   ★★ 判据落空 ⇒ **U 只是 age 的替身** ⇒ "
              "世界模型在这个调度问题上**没有**超出「知道年龄」的额外价值。照报。")
    elif np.isfinite(rho_cond) and rho_cond >= 0.3 and np.isfinite(auc_cond) \
            and auc_cond >= 0.6:
        print("[24]   ★ 判据达成 ⇒ U 携带年龄以外的信息 ⇒ σ 触发**有**独立价值（P4 可做）。")
    else:
        print("[24]   ⚠ 落在中间地带（既没达 0.3/0.6，也没掉到 0.1/0.6 以下）"
              "⇒ **不得下结论**，P4 只作探索性对照，不作为主张依据。")

    # ============================================================ ★★ X42：共形误差信封
    # ★★ 动机（X41 §12.5 的「下一步」）：X40/X41 的结论只能写成「**未经校准的**自报不确定度
    #    没有边际任务价值」。C30（arXiv:2607.01537）用 calibrated rollout-drift envelope
    #    在等预算下赢过 expected-belief 调度 ⇒ 本轮把自报量换成**有覆盖保证**的校准量。
    #
    # ★★ 但**先把 C30 读准**（2026-09-29 核原文，纠正 X41 里的过度转述）：C30 自己写明
    #    *"in the short-horizon frozen VN-JEPA regime, empirical **conformal horizons match**
    #    the deployed clock on validity and budget"* ⇒ **朴素共形并没有赢**。
    #    ⇒ 「换成共形就有价值」是**没有根据的预期**，本号的任务是**把它测出来**。
    #
    # ★★ 概念限定（决定本号能说什么、不能说什么）：共形只给**边缘**覆盖保证
    #    `P(e ≤ Ê) ≥ 1−α`，**不制造排序能力**。若 U|h 的排序力（AUC≈0.674）本来就不转化
    #    为任务收益，校准**不会**凭空创造收益 ⇒ H0 是**先验上完全可能**的结果。
    #
    # ★ 三个坑（K1/K2/K3）见 `wmlab/eval/conformal.py` 文件头；K1/K2 已由自检 ㊷㊸ 钉死。
    conf_tab = None
    conf_diag: dict = {}
    if args.pol == "conformal":
        calib_off = int(cfg["data"].get("calib_seed_offset", 3000))
        n_cal = int(dz.get("calib_samples", 4096))
        want_usable = int(dz.get("calib_episodes", 24))
        if args.quick:
            n_cal, want_usable = min(n_cal, 768), min(want_usable, 12)
        # ★★ 冒烟实测（2026-09-29，一次真事故）：**先把 episode 筛选出来再切分**。
        #   第一版直接 `collect(n)` 然后 `// 2` 切分 ⇒ wind_amp=16 时 `_sample_windows`
        #   抛「没有可用窗口」。根因：**PD 控制器在 amp=16 的阵风下大量跟丢**，
        #   实测 seed_off=3000/n=8 的 8 条里只有 **1 条**长度 > WARMUP+H+1=417
        #   （中位长度只有 169）⇒ 一半的标定集是空的。
        #   ★ 这不是标定集的局部问题：**凡是 `t0_min=WARMUP` 的探针（X38-b/X39/X41 的 P3）
        #     都在"熬过 400 步"的子集上算**，短 episode 一步都不贡献 —— 幸存者偏差，
        #     已在引用限定里如实登记。
        #   ★ 另一个教训：**别在单个种子上赌**。改成跨多个种子批次收集、边收边筛。
        usable: list = []
        n_batch, k_batch, n_raw = 12, 0, 0
        while len(usable) < want_usable and k_batch < 24:
            off = calib_off + 100 * k_batch
            c_env = make_env(cfg["env"]["id"], seed=off,
                             noise_std=float(cfg["env"]["noise_std"]),
                             max_steps=int(cfg["env"]["max_steps"]),
                             wind_amp=wind_amp)
            c_ctrl = PDRelativeController(dt=c_env.dt, omega_n=float(cc["omega_n"]),
                                          zeta=float(cc["zeta"]), kappa=float(c_env.kappa),
                                          a_max=float(c_env.a_max))
            eps_ = collect_controlled_episodes(
                c_env, c_ctrl, n_episodes=n_batch, seed=off,
                max_steps=int(cfg["env"]["max_steps"]))
            c_env.close()
            n_raw += len(eps_)
            # ★ 只留"熬过预热 + 还装得下一个 H 窗口"的 episode
            usable += [e for e in eps_ if int(e["length"]) - H - 1 > WARMUP]
            k_batch += 1
        if len(usable) < 6:
            raise AssertionError(
                f"★ X42：可用标定 episode 只有 {len(usable)} 条"
                f"（收 {n_raw} 条，要求每条长度 > {WARMUP + H + 1}）⇒ "
                "episode 级 fit/留出切分做不了；**不许**降级成全量标定（那会失去覆盖性检验）")
        # ★ K3：**按 episode 切**，不按窗口切（窗口在 episode 内强自相关；
        #   随机切窗口会让标定与留出共享相邻步 ⇒ 覆盖率被高估）
        n_half = len(usable) // 2
        dims = slice(0, 2) if args.conf_target == "pos" else slice(None)
        e_parts, u_parts = {}, {}
        ep_seed = seed + 77
        for half, half_eps, n_half_s in (("fit", usable[:n_half], n_cal),
                                         ("test", usable[n_half:], max(n_cal // 2, 256))):
            o0c, actc, tgtc = _sample_windows(half_eps, H, n_half_s,
                                              seed=ep_seed, t0_min=WARMUP)
            u_parts[half] = _rollout_sigma(model, o0c, actc, device)
            with torch.no_grad():
                zz = model.encode(torch.as_tensor(o0c, dtype=torch.float32, device=device))
                aa = torch.as_tensor(actc, dtype=torch.float32, device=device)
                pdc = []
                for h in range(H):
                    zz = model.next_latent(zz, aa[:, h])
                    pdc.append(model.decode(zz).detach().cpu().numpy())
            pdc = np.stack(pdc, axis=1)
            de = (pdc - tgtc) ** 2
            e_parts[half] = de[:, :, dims].mean(axis=2) / max(var_g, 1e-18)
        conf_tab = fit_envelope(e_parts["fit"], u_parts["fit"],
                                alpha=float(args.conf_alpha),
                                n_knots=int(dz.get("conf_knots", 8)))
        cov = conf_coverage(e_parts["test"], u_parts["test"], conf_tab)
        # ★ S_c2：ĝ 随年龄不减（误差随年龄增长这条常识必须体现在拟合里）
        g_med = np.array([float(np.median(conf_tab["knots_v"][h,
                                                              :int(conf_tab["n_keep"][h])]))
                          for h in range(H)])
        n_drop = int((np.diff(g_med) < -1e-9).sum())
        # ★ tol 阶梯：**预先登记**（锚在配置里既有的 err_threshold=0.05，×2 阶梯）
        tol_list = [float(x) for x in dz.get("conf_tol_list", [0.05, 0.1, 0.2, 0.4])]
        # 每个 tol 的"隐含交叉年龄"：仅靠年龄就会触发的最早步（供可解释性核对）
        u_med_mid = np.array([float(np.median(u_parts["fit"][:, h])) for h in range(H)])
        cross = []
        for tol in tol_list:
            grid_h = np.arange(1, H + 1, dtype=float)
            e_curve = conf_envelope(conf_tab, grid_h, u_med_mid)
            hit = np.nonzero(e_curve >= tol)[0]
            cross.append(int(hit[0]) + 1 if hit.size else -1)
        conf_diag = {"alpha": float(args.conf_alpha), "target": args.conf_target,
                     "calib_seed_offsets": [calib_off + 100 * j for j in range(k_batch)],
                     "n_calib_episodes_raw": int(n_raw),
                     "n_calib_episodes_usable": int(len(usable)),
                     "n_fit_episodes": int(n_half),
                     "n_test_episodes": int(len(usable) - n_half),
                     "usable_ep_len_median": float(np.median(
                         [int(e["length"]) for e in usable])),
                     "n_fit": int(conf_tab["n_fit"]), "q": float(conf_tab["q"]),
                     "q_clipped": bool(conf_tab["clipped"]),
                     "q_unnorm": float(conf_tab["q_unnorm"]),
                     "s_h": conf_tab["s_h"].tolist(),
                     "g_age_median": g_med.tolist(),
                     "n_age_monotone_violations": n_drop,
                     "coverage": cov, "tol_list": tol_list,
                     "implied_cross_age": cross,
                     "u_med_fit": u_med_mid.tolist(),
                     "note": ("共形只给**边缘**覆盖保证；本表逐年龄覆盖率用于判断"
                              "该信封能否在本设定下迁移（K3：episode 级切分）。")}
        print(f"[24]   ★ X42 共形信封：target={args.conf_target} α={args.conf_alpha:g} "
              f"| 标定集 {k_batch} 批 × {n_batch} 条（seed 偏移 "
              f"{conf_diag['calib_seed_offsets'][0]}…+100）收回 {n_raw} 条，"
              f"**熬过预热**的可用 {len(usable)} 条"
              f"（长度中位 {conf_diag['usable_ep_len_median']:.0f}）"
              f"⇒ fit {n_half} / 留出 {len(usable) - n_half} 条")
        print(f"[24]     Q_α={conf_tab['q']:.4f}（未归一化对照 {conf_tab['q_unnorm']:.4f}"
              f" ⇒ Ŝ(h) 的归一化把尺度从 {conf_tab['q_unnorm']:.3f} 压到 "
              f"{conf_tab['q']:.3f}）"
              + ("  ⚠ Q 被样本量截断" if conf_tab["clipped"] else ""))
        print(f"[24]     S_c1 覆盖性（留出集，目标 {cov['target']:.2f}）："
              f"边缘 {cov['marginal']:.4f}｜逐年龄 中位 {cov['per_age_median']:.4f} "
              f"最小 {cov['per_age_min']:.4f}（3σ 容差 ±{cov['tol_3sigma']:.4f}）")
        print(f"[24]     S_c2 ĝ 随年龄单调：违约 {n_drop}/{H - 1} 段"
              f"｜ĝ 中位 {g_med[0]:.4f}→{g_med[-1]:.4f}（h=1→{H}）")
        print("[24]     tol 阶梯与**隐含交叉年龄**（仅靠年龄就会触发的最早步）："
              + "  ".join(f"tol={t:g}@{c if c > 0 else '>H'}"
                          for t, c in zip(tol_list, cross)))
        # ★★ S_c1 是**硬门**：校准本身没兑现 ⇒ 后面的调度结论一律不许下
        if cov["marginal"] < cov["target"] - cov["tol_3sigma"]:
            raise AssertionError(
                f"★ S_c1 未通过：留出集边缘覆盖率 {cov['marginal']:.4f} < "
                f"目标 {cov['target']:.4f} − 3σ {cov['tol_3sigma']:.4f}"
                " ⇒ 该信封在本设定下不可迁移，X42 的调度结论全部作废（不许照报）")
        if conf_tab["clipped"]:
            print("[24]     ⚠ Q_α 被样本量截断 ⇒ 覆盖率退化为 1（恒覆盖），"
                  "本节结论只能作**上界**用")
        if n_drop:
            print(f"[24]     ⚠ S_c2 有 {n_drop} 段 ĝ 随年龄下降 ⇒ 先查标定集是否有"
                  "分布漂移，别急着解释")

    # ============================================================ ★ X38-b 快捷出口
    # X38-b 只关心「σ 头在异方差环境下能不能学到状态依赖 ⇒ ρ_cond 是否上升」，
    # 不需要闭环 P2/P4 那 96+48 个工作点（16 min）⇒ 到此为止，落一个只含 P1/P3 的 JSON。
    if args.p3_only:
        payload_b = {
            "experiment": "X38-b：状态依赖噪声（阵风场）下，σ 头能否学到异方差？",
            "env": {"id": cfg["env"]["id"],
                    "noise_std": float(cfg["env"]["noise_std"]),
                    "wind_amp": float(wind_amp)},
            "var_g": var_g,
            "P1_verdict": p1_verdict,
            "P3_conditional": {"rho_unconditional": float(rho_all),
                               "rho_conditional_median": float(rho_cond),
                               "auc_conditional_median": float(auc_cond),
                               "u_cv": float(u_cv), "sigma_step_cv": float(s_cv),
                               "u_cv_conditional_median": float(np.median(cvs)),
                               "per_h": per_h, **p3_extra},
        }
        # ★ X42：`--p3-only` 也把共形标定/覆盖性诊断落盘（冒烟即可验证全链路）
        if conf_diag:
            payload_b["conformal"] = conf_diag
        # ★ 文件名必须带 wind_amp：否则 4 个剂量档互相覆盖，只剩最后一个（剂量—反应曲线就没了）
        tag = args.tag + f"_wind{wind_amp:g}"
        jpath = os.path.join(out, tag + ".json")
        with open(jpath, "w", encoding="utf-8") as f:
            json.dump(jsonable(payload_b), f, ensure_ascii=False, indent=2)
        print(f"[24] ★ X38-b 产物：{jpath}  ({time.time() - t0:.0f}s)")
        env.close()
        return

    # ============================================================ 4b) Part D：P4 不确定性触发
    print("[24] " + "-" * 78)
    print("[24] Part D · P4（真贡献）：把触发量从 age 换成**模型自报的累积不确定度 U**")
    # ★★ 阈值的尺度必须取**小 h 那一段**的 U，不能取全池（第一版的坑）：
    #    全池混进 h=32 的大 U ⇒ 分位阈值被抬高到"几乎永不触发"
    #    ⇒ 实测 E[age] 14–42、逃逸率 100%、NMSE 2–4（发散），整组作废。
    # ★★ 阈值怎么定（第一版踩了两个坑，都记下来）
    #   坑 1：取**全池** U 的分位点 ⇒ 混进 h=32 的大 U ⇒ 阈值被抬到"几乎永不触发"
    #        （实测 E[age] 14–42、逃逸率 100%、NMSE 2–4，整组发散作废）
    #   坑 2：取 h≤8 段的**分位点** ⇒ 尺度对了但**分布不均匀**（U≈U(1)·√h 是凹的），
    #        阈值稍微一大就直接跳到发散区。
    #   ⇒ 改成**按目标年龄标定**：thr = 离线 U 在 h* 步处的中位数。
    #     这样 h* 就是"名义触发年龄"，与周期 T / 年龄阈值 K **同量纲可比**，
    #     也方便直接看出"实际年龄/名义年龄"这个比值是否失控（σ 头睡着的信号）。
    h_stars = [int(x) for x in dz.get("u_h_star_list", [1, 2, 3, 4, 6, 8, 12, 16])]
    h_stars = [h for h in h_stars if 1 <= h <= H]
    # ★★ X40：触发量二选一（`--pol` 切），**唯一差别是「有没有先把 age 条件掉」**
    #   "utrigger"（X38/X39）：thr = median(U(h*)) —— 按名义年龄标定 ⇒ 数学上≈年龄阈值
    #   "resid"    （X40）    ：τ  = 分位点(U − Û(age)) —— 先条件掉 age，吃条件信息
    u_hat = None
    u_scale = None
    hybrid_spec: list[tuple[int, float]] = []
    if args.pol == "conformal":
        # ★★ X42：旋钮 = 安全网年龄 K × **误差容限 tol**。
        #   ★ 与 X40 的**唯一**差别是触发统计量：X40 卡的是「U 在给定年龄下是否异常」
        #     （分位点标定的 τ，是个**调度旋钮**）；X42 卡的是「**校准后的误差上界**是否
        #     超过容限」（tol，是**任务量纲**）。规则结构逐字相同：`age≥K 或 触发`。
        if conf_tab is None:
            raise AssertionError("★ X42：--pol conformal 必须先生成共形信封")
        knobs = [int(x) for x in dz.get("hybrid_K_list", [2, 4, 8, 16, 32])]
        tol_list = [float(x) for x in conf_diag["tol_list"]]
        u_thrs = [t for k in knobs for t in tol_list]
        knobs = [k for k in knobs for _ in tol_list]
        print(f"[24]   共形触发：age≥K 或 Ê_α(age,U)≥tol"
              f"（Ê 已按 (h, U) 条件化；α={args.conf_alpha:g}，目标量={args.conf_target}）")
        print(f"[24]     K ∈ {sorted(set(knobs))}   tol ∈ {tol_list}"
              f"  ⇒ {len(knobs)} 个工作点")
        print("[24]     退化钉死（S_h'）：tol=+∞ ⇒ ≡纯年龄阈值；K=1 或 tol=−∞ ⇒ ≡T=1")
        _K0, _per0 = int(knobs[0]), float(pers[0])
        a_inf, _ = _sim_conformal(_K0, float("inf"), _per0, U, conf_tab,
                                  40000, seed + 91)
        a_thr = float(threshold_mean_age(_K0, _per0))
        a_k1, _ = _sim_conformal(1, -1e9, _per0, U, conf_tab, 40000, seed + 92)
        a_ninf, _ = _sim_conformal(_K0, -1e9, _per0, U, conf_tab, 40000, seed + 93)
        a_t1 = float(periodic_mean_age(_per0, 1))
        print(f"[24]     S_h'：tol=+∞ ⇒ E[age] {a_inf:.4f} vs 纯阈值闭式 {a_thr:.4f}"
              f"（相对差 {abs(a_inf - a_thr) / max(a_thr, 1e-9):.2%}）")
        if abs(a_inf - a_thr) > max(0.05, 0.03 * a_thr):
            raise AssertionError(
                f"★ S_h' 未通过：tol=+∞ 未退化为纯年龄阈值（{a_inf:.4f} vs {a_thr:.4f}）")
        if abs(a_k1 - a_t1) > max(0.05, 0.03 * a_t1) or \
                abs(a_ninf - a_t1) > max(0.05, 0.03 * a_t1):
            raise AssertionError(
                f"★ S_h' 未通过：K=1 / tol=−∞ 未退化为 T=1"
                f"（{a_k1:.4f} / {a_ninf:.4f} vs {a_t1:.4f}）")
        print(f"[24]     S_h'：K=1 ⇒ {a_k1:.4f}、tol=−∞ ⇒ {a_ninf:.4f}"
              f" vs T=1 闭式 {a_t1:.4f} ⇒ 三条退化全部通过 ✓")
    elif args.pol in ("resid", "hybrid"):
        # ★★ z 分数标准化（第一版用原始残差 ⇒ 尺度随 age 变 ⇒ 冒烟给 4.77× 假结果）
        u_hat = np.median(U, axis=0)                        # (H,) Û(age) 条件中位数
        mad = np.median(np.abs(U - u_hat[None, :]), axis=0)  # 条件 MAD（抗尾）
        u_scale = np.maximum(1.4826 * mad, 1e-6)            # → 稳健标准差
        z_pool = ((U - u_hat[None, :]) / u_scale[None, :]).reshape(-1)
        tau_qs = [float(x) for x in dz.get("resid_quantiles",
                                           [0.6, 0.7, 0.8, 0.85, 0.9])]
        tau_vals = [float(np.quantile(z_pool, q)) for q in tau_qs]
        if args.pol == "resid":
            knobs = [int(round(q * 100)) for q in tau_qs]
            u_thrs = list(tau_vals)
            print("[24]   纯残差触发：τ 取 **z 分数** (U−Û(age))/Ŝ(age) 的分位点："
                  + "  ".join(f"q{q:g}:{v:+.3f}" for q, v in zip(tau_qs, tau_vals))
                  + f"   （Ŝ=条件 MAD×1.4826，中位 {float(np.median(u_scale)):.4f}）")
        else:
            hy_k = [int(x) for x in dz.get("hybrid_K_list", [4, 8, 16])]
            hy_q = [float(x) for x in dz.get("hybrid_tau_quantiles", [0.7, 0.9])]
            hybrid_spec = [(k, float(np.quantile(z_pool, q))) for k in hy_k for q in hy_q]
            knobs = [k for k, _ in hybrid_spec]
            u_thrs = [t for _, t in hybrid_spec]
            print("[24]   混合触发：age≥K 或 z≥τ（**残差只准加速**，安全网 K 不动）")
            print("[24]     K ∈ " + str(hy_k) + "   z 分位 " + str(hy_q)
                  + " ⇒ τ = " + ", ".join(f"{t:+.3f}" for _, t in hybrid_spec))
            print("[24]     退化钉死（S_h）：τ=+∞ ⇒ ≡纯年龄阈值；K=1 或 τ=−∞ ⇒ ≡T=1")
            # ★ S_h：三条退化必须逐位成立（拿已知答案验量级，R12）
            _K0, _per0 = int(hy_k[0]), float(pers[0])
            a_inf, _ = _sim_hybrid(_K0, float("inf"), _per0, U, u_hat, u_scale,
                                   40000, seed + 91)
            a_thr = float(threshold_mean_age(_K0, _per0))
            a_k1, _ = _sim_hybrid(1, -1e9, _per0, U, u_hat, u_scale, 40000, seed + 92)
            a_ninf, _ = _sim_hybrid(_K0, -1e9, _per0, U, u_hat, u_scale,
                                    40000, seed + 93)
            a_t1 = float(periodic_mean_age(_per0, 1))
            print(f"[24]     S_h：τ=+∞ ⇒ E[age] {a_inf:.4f} vs 纯阈值闭式 {a_thr:.4f}"
                  f"（相对差 {abs(a_inf - a_thr) / max(a_thr, 1e-9):.2%}）")
            if abs(a_inf - a_thr) > max(0.05, 0.03 * a_thr):
                raise AssertionError(
                    f"★ S_h 未通过：τ=+∞ 未退化为纯年龄阈值（{a_inf:.4f} vs {a_thr:.4f}）")
            if abs(a_k1 - a_t1) > max(0.05, 0.03 * a_t1) or \
                    abs(a_ninf - a_t1) > max(0.05, 0.03 * a_t1):
                raise AssertionError(
                    f"★ S_h 未通过：K=1 / τ=−∞ 未退化为 T=1"
                    f"（{a_k1:.4f} / {a_ninf:.4f} vs {a_t1:.4f}）")
            print(f"[24]     S_h：K=1 ⇒ {a_k1:.4f}、τ=−∞ ⇒ {a_ninf:.4f}"
                  f" vs T=1 闭式 {a_t1:.4f} ⇒ 三条退化全部通过 ✓")
    else:
        knobs = list(h_stars)
        u_thrs = [float(np.median(U[:, h - 1])) for h in h_stars]
        print("[24]   阈值按「名义触发年龄 h*」标定（thr = 离线 U(h*) 的中位数）："
              + "  ".join(f"h*={h}:{v:.4f}" for h, v in zip(h_stars, u_thrs)))
    n_div = 0
    conf_trace = None
    for per in pers:
        for knob, thr in zip(knobs, u_thrs):
            state = {"U": 0.0, "age": 0}
            # ★ X42 口径核对：对**第一个**工作点开逐步 trace，事后比
            #   「闭环 U|age」与「离线标定 U|age」是否同分布（R12：参数真的接线了吗）
            if args.pol == "conformal" and conf_trace is None:
                state["_trace"] = []
            sch = (_resid_schedule(thr, per, state, u_hat, u_scale)
                   if args.pol == "resid" else
                   _hybrid_schedule(knob, thr, per, state, u_hat, u_scale)
                   if args.pol == "hybrid" else
                   _conformal_schedule(knob, thr, per, state, conf_tab)
                   if args.pol == "conformal" else
                   _utrig_schedule(thr, per, state))
            r = run_closed_loop_control(
                env, model, ctrl, sch, n_episodes=n_task_ep, seed=online_seed,
                max_steps=meas_steps, device=device, estimator="model",
                var_g=var_g, label=f"{UT}/{args.pol}={thr:.4f}",
                tail_frac=float(cfg["task"]["tail_frac"]), warmup_steps=WARMUP,
                u_state=state)
            # ★ 发散点（阈值太高 ⇒ 几乎不发送 ⇒ rollout 炸掉）：记下来但**不进比较**
            #   ★ X39 修正：判定已上移到 `_mkrow`，对**三族统一**施加（不再只作用于 utrigger）
            row = _mkrow(UT, per, knob, r, float("nan"), KC,
                         (float("nan"), float("nan")), u_thr=float(thr),
                         meas_steps=meas_steps)
            n_div += int(row["diverged"])
            # ★ 名义 vs 实际（仅全局阈值有意义；残差触发的旋钮不是年龄，记 NaN）
            row["age_over_hstar"] = (float(r["mean_age"]) / float(knob)
                                     if args.pol != "resid" else float("nan"))
            row["trigger"] = args.pol
            rows.append(row)
            if args.pol == "conformal" and conf_trace is None \
                    and state.get("_trace") is not None:
                conf_trace = state["_trace"]
    env.close()
    if n_div:
        print(f"[24]   ⚠ {n_div}/{len(pers) * len(u_thrs)} 个 {UT} 触发点发散"
              f"（NMSE>1 或 E[age]>K={KC}）⇒ 已从等预算比较中剔除，不计入结论")
    for r in rows:
        if r["policy"] == UT:
            _ex = (f"(=名义 {r['age_over_hstar']:.2f}×) "
                   if np.isfinite(r["age_over_hstar"]) else "")
            _kb = (f"K={r['knob']:<3d} τ={r['u_thr']:+.3f}" if args.pol == "hybrid"
                   else (f"K={r['knob']:<3d} tol={r['u_thr']:g}"
                         if args.pol == "conformal"
                         else (f"τ={r['u_thr']:+.3f}   " if args.pol == "resid"
                               else f"h*={r['knob']:<3d} thr={r['u_thr']:+.4f}")))
            print(f"[24]   PER={r['per']:<4g} {UT:<9s} {_kb} "
                  f"tx_rate={r['tx_rate']:.4f} "
                  f"E[age]={r['mean_age']:6.2f} {_ex}| "
                  f"NMSE={r['est_nmse']:.5f} 距离={r['mean_dist_tail']:6.3f}m "
                  f"逃逸={r['escape_rate']:.3f}"
                  + ("   ← 发散，剔除" if r.get("diverged") else ""))
    # ★ R12：U 阈值必须真的接线（改阈值必须改变发送率），否则整个 P4 在空跑
    rates_by_per = {}
    for per in pers:
        sub = [r for r in rows if r["policy"] == UT and abs(r["per"] - per) < 1e-12]
        if sub:
            rates_by_per[per] = (min(r["tx_rate"] for r in sub),
                                 max(r["tx_rate"] for r in sub))
    n_wired = sum(1 for lo, hi in rates_by_per.values() if hi > lo * 1.05)
    if n_wired == 0:
        raise AssertionError(
            "★ S_g（R12）未通过：改 U 阈值**没有**改变发送率 ⇒ "
            "u_state 没接进 control.py，整个 P4 在空跑")
    print(f"[24]   S_g（R12）：{n_wired}/{len(rates_by_per)} 个 PER 档上改 U 阈值"
          f"使发送率变化 >5% ⇒ 不确定性触发已接线 ✓")

    # ★★ X42 S_c4：**这一臂到底在不在动**（R12 的对象换成"整个策略族"）。
    #   冒烟实测：K=8 时 tol=0.2 / 0.4 的 tx_rate 与**同 K 纯年龄阈值**逐位相同
    #   ⇒ 那些工作点是"安全网 K 主导、信封从未提前触发"。
    #   若这种点在整族里占多数，则 H1/H0 的比较实际上在比"两个年龄阈值"，
    #   **结论会被读错**（把"信封没生效"读成"信封无价值"）⇒ 必须机器数出来。
    conf_active = None
    if args.pol == "conformal":
        thr_map = {(r["per"], r["knob"]): r["tx_rate"]
                   for r in rows if r["policy"] == "threshold"}
        n_kdom = n_tot = 0
        for r in rows:
            if r["policy"] != UT:
                continue
            base = thr_map.get((r["per"], r["knob"]))
            if base is None:
                continue
            n_tot += 1
            if abs(float(r["tx_rate"]) - float(base)) <= 1e-9:
                n_kdom += 1
        conf_active = {"n_work_points": int(n_tot), "n_k_dominated": int(n_kdom),
                       "frac_active": (1.0 - n_kdom / max(n_tot, 1)),
                       "note": ("K 主导 = 该工作点上共形信封从未比安全网 K 更早触发；"
                                "此时该点等价于纯年龄阈值，不构成对 H1 的证据")}
        print(f"[24]   ★ S_c4：共形触发相对**同 K 纯年龄阈值**真的改变了发送率的比例 = "
              f"{conf_active['frac_active']:.2f}（{n_tot - n_kdom}/{n_tot} 个可比工作点）"
              + ("  ⇒ ⚠ 多数工作点被安全网 K 主导 ⇒ 该族的比较**大部分**在比两个年龄阈值，"
                 "H1 只能按**剩下那部分**工作点下结论"
                 if conf_active["frac_active"] < 0.4 else "  ⇒ 信封确实在起作用 ✓"))

    # ★★ X42 口径核对（R12 的同一句话，对象换成"标定集"）：
    #   信封是在**离线** rollout 上学出来的（U 由真观测序列累积），
    #   而闭环里的 U 由**漂移中的估计**累积 ⇒ 两者可以不同分布。
    #   若不同分布，信封在闭环上就**没有覆盖保证**（可交换性被破坏）⇒ 必须如实报。
    conf_align = None
    if conf_trace:
        tr = np.asarray(conf_trace, dtype=float)
        off = np.asarray(conf_diag.get("u_med_fit", []), dtype=float)
        al_rows, ratios = [], []
        for h_ref in (1, 2, 4, 8, 16):
            m = tr[:, 0] == h_ref
            if int(m.sum()) >= 20 and h_ref <= off.size:
                cm, om = float(np.median(tr[m, 1])), float(off[h_ref - 1])
                al_rows.append({"h": h_ref, "n": int(m.sum()),
                                "u_med_closed": cm, "u_med_offline": om,
                                "ratio": cm / max(om, 1e-12)})
                ratios.append(cm / max(om, 1e-12))
        if ratios:
            r_arr = np.asarray(ratios, dtype=float)
            conf_align = {"per_h": al_rows,
                          "ratio_median": float(np.median(r_arr)),
                          "ratio_min": float(r_arr.min()),
                          "ratio_max": float(r_arr.max())}
            print(f"[24]   ★ X42 口径核对（闭环 U|age 中位 ÷ 离线 U|age 中位）："
                  + "  ".join(f"h={r['h']}:{r['ratio']:.3f}" for r in al_rows))
            if not (0.8 <= float(np.median(r_arr)) <= 1.25):
                print("[24]     ⚠⚠ 两者差 >25% ⇒ 闭环 U 不在标定分布内 ⇒ "
                      "共形覆盖保证在闭环上**不成立**，本号只能作**探索性**对照（必须写进引用限定）")
            else:
                print("[24]     ✓ 闭环与离线同量级（±25% 内）⇒ 标定可迁移")

    # --- 三方等预算对比（在同一 tx_rate 网格上插值）---
    cmp3 = []
    for per in pers:
        fams = {p: sorted([r for r in rows if r["policy"] == p
                           and abs(r["per"] - per) < 1e-12
                           and not r.get("diverged", False)],
                          key=lambda r: r["tx_rate"])
                for p in ("periodic", "threshold", UT)}
        try:
            lo, hi = overlap_window(
                {p: [r["tx_rate"] for r in v] for p, v in fams.items()},
                min_range=MIN_RATE_RANGE, min_pts=MIN_FAMILY_PTS)
        except WindowNotEvaluable as e:
            print(f"[24]   ⚠ 该档不可评估：PER={per:g} —— {e}")
            continue
        grid = np.array(expo_grid(lo, hi, 9))
        for met in ("est_nmse", "mean_dist_tail"):
            vals = {}
            for p, v in fams.items():
                x = np.array([r["tx_rate"] for r in v])
                y = np.array([r[met] for r in v])
                if np.any(~np.isfinite(y)):
                    vals = {}
                    break
                vals[p] = np.interp(grid, x, y)
            if len(vals) != 3:
                # ★ 2026-09-29 补：这里原先是**静默 continue** ⇒ 输出里少一行，
                #   会被读成"这一档没问题"（铁律 17 ⑥：不可评估必须把**原因**打进 stdout）。
                #   实测本轮的触发路径是 `overlap_window` 的显式异常（有打印），
                #   但这条静默路径是**潜伏**的：只要某族某一列出现非有限值就会吃掉一整档。
                print(f"[24]   ⚠ 该档 {met} 不可评估：PER={per:g} —— "
                      f"三族中有族的「{met}」列含非有限值（len(vals)={len(vals)}≠3）"
                      " ⇒ 跳过该条，**不计入结论**")
                continue
            with np.errstate(divide="ignore", invalid="ignore"):
                r_ut = vals[UT] / np.maximum(vals["threshold"], 1e-18)
                r_tt = vals["threshold"] / np.maximum(vals["periodic"], 1e-18)
            r_ut = r_ut[np.isfinite(r_ut)]
            r_tt = r_tt[np.isfinite(r_tt)]
            if r_ut.size == 0 or r_tt.size == 0:
                continue
            cmp3.append({"per": per, "metric": met, "n_grid": int(r_ut.size),
                         "rate_lo": float(lo), "rate_hi": float(hi),
                         "utrigger_over_threshold_median": float(np.median(r_ut)),
                         "utrigger_better_n": int((r_ut < 1.0).sum()),
                         "threshold_over_periodic_median": float(np.median(r_tt)),
                         "threshold_better_n": int((r_tt < 1.0).sum())})
    for c in cmp3:
        print(f"[24] ★ P4[{c['metric']}] PER={c['per']:<4g} "
              f"U触发/年龄阈值 中位 {c['utrigger_over_threshold_median']:.3f}"
              f"（{c['utrigger_better_n']}/{c['n_grid']} 更优）| "
              f"年龄阈值/周期 中位 {c['threshold_over_periodic_median']:.3f}"
              f"（{c['threshold_better_n']}/{c['n_grid']} 更优）")

    # ============================================================ 5) 图
    # ★★ 图标题必须**跟着本次运行走**。原先硬编码 "X38 触发式调度" ⇒ X39/X40 的产物图
    #    全顶着 X38 的名字，**图与实验对不上**（`payload["experiment"]` 早就是派生的，图漏了）。
    #    payload 里仍保留更详细的那句描述，这里只负责「这张图是哪一次运行」。
    _fam = {"utrigger": "X38", "resid": "X40-纯残差", "hybrid": "X40",
            "conformal": "X42"}.get(args.pol, "X38")
    exp_title = (f"{_fam} 触发式调度（trigger={args.pol}，wind_amp={wind_amp:g}）："
                 "等传输预算下 周期 / 年龄阈值 / 不确定性触发")
    fig, axes = plt.subplots(2, 3, figsize=(17.5, 9.5))
    fig.suptitle(exp_title, fontsize=11)

    ax = axes[0, 0]
    for per in pers:
        sub = [r for r in p1_rows if abs(r["per"] - per) < 1e-12 and r["ratio"] is not None]
        if sub:
            ax.plot([r["K"] for r in sub], [r["ratio"] for r in sub], "o-",
                    label=f"PER={per:g}")
    ax.axhline(1.0, color="k", ls=":", lw=1)
    ax.set_xlabel("阈值 K（步）"); ax.set_ylabel("阈值 E[age] / 等预算周期 E[age]")
    ax.set_title("① P1：等预算年龄比（<1 = 阈值更优）")
    ax.legend(fontsize=7.5, ncol=2); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    for pol, col, mk in (("periodic", PALETTE["blue"], "o"),
                         ("threshold", PALETTE["red"], "s"),
                         (UT, PALETTE["purple"], "^")):
        sub = [r for r in rows if r["policy"] == pol]
        if sub:
            ax.plot([r["tx_rate"] for r in sub], [r["est_nmse"] for r in sub],
                    mk, color=col, alpha=0.75, label=pol)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("tx_rate（送达/步）"); ax.set_ylabel("闭环 est_nmse")
    ax.set_title("② P2/P4：率—误差平面（同率下谁更低）")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[0, 2]
    aa = np.array([r["age_mc"] for r in rows])
    ea = np.array([r["mean_age"] for r in rows])
    ax.plot(aa, ea, "o", color=PALETTE["green"], alpha=0.8)
    lim = [0, max(aa.max(), ea.max()) * 1.1]
    ax.plot(lim, lim, "k:", lw=1)
    ax.set_xlabel("同结构 MC 的 E[age]"); ax.set_ylabel("闭环实测 E[age]")
    ax.set_title("③ S_f：年龄过程与调度一致（偏差以 σ 计）")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    hs = [x["h"] for x in per_h]
    ax.plot(hs, [x["rho"] for x in per_h], "o-", color=PALETTE["blue"],
            label="条件 ρ(U, err | h)")
    ax.axhline(0.0, color="k", ls=":", lw=1)
    ax.axhline(rho_all, color=PALETTE["orange"], ls="--", lw=1,
               label=f"不条件 ρ={rho_all:+.2f}")
    ax.set_xlabel("age h（步）"); ax.set_ylabel("Pearson ρ")
    ax.set_title(f"④ P3：给定年龄后的残余相关性（中位 {rho_cond:+.3f}）")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(hs, [x["auc"] for x in per_h], "s-", color=PALETTE["red"],
            label="条件 AUC")
    ax.axhline(0.5, color="k", ls=":", lw=1)
    ax.set_xlabel("age h（步）"); ax.set_ylabel("AUC（U 预测误差超阈）")
    ax.set_title(f"⑤ P3：条件 AUC（中位 {auc_cond:.3f}）")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1, 2]
    ax.plot(hs, [x["err_mean"] for x in per_h], "o-", color=PALETTE["green"],
            label="实测 NMSE")
    ax.plot(hs, [x["u_mean"] for x in per_h], "s--", color=PALETTE["purple"],
            label="U（模型自报）")
    ax.set_xlabel("age h（步）"); ax.set_ylabel("量值（不同量纲）")
    ax.set_title("⑥ 误差与 U 随年龄的增长（形状对比）")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ★★ 产物名必须跟着 tag / wind_amp 走。
    #    【2026-09-29 实测事故】这里原先硬编码 `24_triggered_scheduling`，而 `--tag` 只在
    #    `--p3-only` 分支生效 ⇒ 一次 `--quick --wind-amp 4` 冒烟就把 X38 的完整
    #    JSON/PNG（09-27 的 143.9 KB 证据）覆盖成了 41.4 KB 的 quick 产物。
    #    ⇒ amp=0 仍得 `24_triggered_scheduling.*`（保持文档引用不变）；amp>0 自动加后缀。
    _stem = args.tag + (f"_wind{wind_amp:g}" if wind_amp > 0 else "")
    save_fig(fig, os.path.join(out, _stem + ".png"))
    plt.close(fig)

    # ★★ X42：共形诊断图（单独一张，别把主图塞满）—— 一图看清"信封长什么样"
    if conf_tab is not None:
        fig2, ax2 = plt.subplots(1, 3, figsize=(16.5, 4.6))
        fig2.suptitle(f"X42 共形误差信封（target={conf_diag['target']}，"
                      f"α={conf_diag['alpha']:g}，wind_amp={wind_amp:g}，"
                      f"trigger={args.pol}）", fontsize=11)
        hh = np.arange(1, H + 1)
        # ① 留出集逐年龄覆盖率（S_c1）
        pa = np.asarray(conf_diag["coverage"]["per_age"], dtype=float)
        ax2[0].bar(hh, pa, color=PALETTE["blue"], alpha=0.8)
        ax2[0].axhline(conf_diag["coverage"]["target"], color=PALETTE["red"], ls="--",
                       lw=1.2, label=f"目标 {conf_diag['coverage']['target']:.2f}")
        ax2[0].set_ylim(0, 1.02)
        ax2[0].set_xlabel("age h（步）"); ax2[0].set_ylabel("实测覆盖率")
        ax2[0].set_title("① S_c1：留出集逐年龄覆盖率（共形只保**边缘**）")
        ax2[0].legend(fontsize=8); ax2[0].grid(alpha=0.3)
        # ② 信封曲线（U 取三个分位）+ tol 线
        uf = np.asarray(conf_diag["u_med_fit"], dtype=float)
        u_sd = np.asarray(conf_tab["s_h"], dtype=float)
        for q, ls in ((0.1, ":"), (0.5, "-"), (0.9, "--")):
            uq = uf + (0.0 if q == 0.5 else (u_sd * (1.2816 if q == 0.9 else -1.2816)))
            ax2[1].plot(hh, conf_envelope(conf_tab, hh.astype(float), uq),
                        ls, color=PALETTE["purple"], alpha=0.9,
                        label=f"U @ q{q:g}")
        for tol in conf_diag["tol_list"]:
            ax2[1].axhline(tol, color=PALETTE["green"], lw=0.9, alpha=0.6)
        ax2[1].set_yscale("log")
        ax2[1].set_xlabel("age h（步）"); ax2[1].set_ylabel("校正后误差上界 $\\hat E_\\alpha$")
        ax2[1].set_title("② 信封随年龄的增长（横线 = tol 阶梯）")
        ax2[1].legend(fontsize=7.5); ax2[1].grid(alpha=0.3)
        # ③ 触发边界：给定 h，U 要多大才会触发
        for k, tol in enumerate(conf_diag["tol_list"]):
            us = []
            for h in range(H):
                nk = int(conf_tab["n_keep"][h])
                ku = conf_tab["knots_u"][h, :nk]
                kv = conf_tab["knots_v"][h, :nk] + conf_tab["s_h"][h] * conf_tab["q"]
                hit = np.nonzero(kv >= tol)[0]
                us.append(float(ku[hit[0]]) if hit.size else np.nan)
            ax2[2].plot(hh, us, "o-", ms=3, alpha=0.85, label=f"tol={tol:g}")
        ax2[2].set_xlabel("age h（步）"); ax2[2].set_ylabel("触发所需的最小 U")
        ax2[2].set_title("③ 触发边界 $u^*(h)$（越低 = 越容易提前发）")
        ax2[2].legend(fontsize=7.5); ax2[2].grid(alpha=0.3)
        fig2.tight_layout(rect=(0, 0, 1, 0.94))
        save_fig(fig2, os.path.join(out, _stem + "_conf.png"))
        plt.close(fig2)

    payload = {
        "experiment": {"resid": "X40 纯残差触发 z≥τ（诊断：会撤掉安全网）",
                       "hybrid": "X40 混合触发 age≥K 或 z≥τ（残差只准加速）",
                       "conformal": "X42 共形校准触发 age≥K 或 Ê_α(age,U)≥tol"
                                    "（把自报 σ 换成有覆盖保证的误差上界）",
                       }.get(args.pol,
                             "X38 触发式调度：等传输预算下 周期 / 年龄阈值 / 不确定性触发"),
        "trigger": args.pol,
        "conf_target": (args.conf_target if args.pol == "conformal" else None),
        "wind_amp": float(wind_amp),
        "config": {k: cfg[k] for k in ("env", "controller", "design", "eval", "task")},
        "obs_dim": int(obs_dim), "var_g": var_g,
        "P1_analytic": p1_rows, "P1_verdict": p1_verdict,
        "P2_closed_loop": rows, "P2_matched_rate": cmp_rows,
        "P4_u_thresholds": u_thrs, "P4_matched_rate": cmp3,
        "conformal": (conf_diag if conf_diag else None),
        "conformal_alignment": conf_align,
        "conformal_active": conf_active,
        "P3_conditional": {"rho_unconditional": float(rho_all),
                           "rho_conditional_median": float(rho_cond),
                           "auc_conditional_median": float(auc_cond),
                           "u_cv": float(u_cv), "sigma_step_cv": float(s_cv),
                           "per_h": per_h, **p3_extra},
        "caveat": ("★ P1/P2（周期 vs 年龄阈值）**不是新贡献**：Sun–Polyanskiy–"
                   "Uysal-Biyikoglu（arXiv:1701.06734 / 1707.02531）已在采样率约束下"
                   "证明阈值策略最优并显式比较过 uniform。本号只作复现 + 移到闭环控制场景。"
                   "★ tx_rate 是**送达率**，不是尝试率（两者只差因子 s，配对关系相同）。"
                   "★ 只用逐包独立丢包，不含突发信道。"
                   "★★ `diverged`（NMSE>1 或 E[age]>max_age_K）**对三族统一施加**"
                   "（2026-09-29 修正：原先只在 utrigger 分支算 ⇒ 口径不对称）。"
                   "★★ trigger='utrigger' 的阈值是 `thr=median(U[:,h*-1])`（按名义年龄标定），"
                   "而闭环 ρ(U,age)=0.9941 ⇒ **它数学上就≈年龄阈值**，不能用来检验"
                   "「U 有无边际价值」；要检验请用 trigger='hybrid'。"
                   + ("  ★★ trigger='conformal'（X42）：触发量 = **共形校准的误差上界**"
                      " `Ê_α(h,U)=ĝ(h,U)+Ŝ(h)Q_α`（ĝ = 逐年龄 PAVA 单调拟合，"
                      "Q_α = 分数 (e−ĝ)/Ŝ 的 ⌈(n+1)(1−α)⌉ 阶统计量）。"
                      "★ 共形只给**边缘**覆盖保证（`P(e≤Ê)≥1−α`），**不制造排序能力** "
                      "⇒ 若 U|h 的排序力本就不转化为任务收益，校准不会凭空创造收益。"
                      "★ 标定集是**独立 episode**（seed+3000）且**按 episode 切** fit/留出"
                      "（K3：窗口在 episode 内强自相关，随机切会高估覆盖率）。"
                      "★ S_c1（留出集覆盖率）是**硬门**：不达标则本号结论全部作废。"
                      "★ tol 阶梯 {0.05,0.1,0.2,0.4} 锚在配置既有的 err_threshold=0.05，"
                      "**跑前登记**；它是**任务量纲**、与环境尺度绑定，跨场景必须重标。"
                      "★ C30（arXiv:2607.01537）自己写明其短视界档"
                      "『empirical conformal horizons **match** the deployed clock』"
                      "⇒ 『换成共形就有价值』是**没有根据的预期**，本号只负责测出来。"
                      if args.pol == "conformal" else "")),
    }
    jpath = os.path.join(out, _stem + ".json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(jsonable(payload), f, ensure_ascii=False, indent=2)
    print(f"[24] 产物：{os.path.join(out, _stem + '.png')} / {jpath}  "
          f"({time.time() - t0:.0f}s)")


def _utrig_schedule(threshold_u: float, loss_prob: float, state: dict) -> object:
    """★ X38（P4）：**不确定性触发** —— 只有 U ≥ threshold_u 时才尝试发送。

    与年龄阈值的唯一差别：**触发量从 age（signal-independent）换成 U（signal-dependent）**。
    Sun 等（arXiv:1701.06734）把采样策略分成 signal-independent / signal-dependent 两类，
    并证明后者可以严格更好 —— 但他们假定**信号模型已知**（Wiener 过程）。
    这里 U 来自**学习得到的**世界模型的 σ 头 ⇒ 这是本号的真正增量。

    `state["U"]` 由 `run_closed_loop_control(u_state=state)` 每步写入。
    """
    thr = float(threshold_u)
    p = float(min(max(loss_prob, 0.0), 1.0))

    def _f(t: int, rng: np.random.Generator) -> bool:
        if float(state.get("U", 0.0)) < thr:
            return False
        return bool(rng.random() >= p)

    def _reset(rng: np.random.Generator | None = None) -> None:
        state["U"] = 0.0

    _f.reset = _reset                                   # type: ignore[attr-defined]
    _f.__name__ = f"utrig(thr={thr:.4f},p={p})"
    return _f


def _resid_schedule(tau: float, loss_prob: float, state: dict,
                    u_hat: np.ndarray, u_scale: np.ndarray) -> object:
    """★ X40：**残差（z 分数）触发** —— 只有 `(U − Û(age)) / Ŝ(age) ≥ τ` 时才尝试发送。

    与 X38/X39 的**全局阈值**触发的唯一差别：**先把 age 条件掉**。

    ★ 为什么必须做（X39 实测的硬动机）：闭环里 `ρ(U, age) = 0.9941`
      ⇒ 全局阈值 `U ≥ thr` 在数学上**就是**一个年龄阈值（X39 的 thr 甚至是
      「按名义年龄 h* 标定」的）⇒ 它**在原理上无法利用** P3 测到的条件信息
      （ρ_cond=0.347、AUC_cond=0.601 是**给定 age 之后**的残余信息）。

    ★★ 为什么必须**标准化**（第一版踩的坑，直接记下来）：先用「原始残差
      U − Û(age) 的全局分位点」当 τ ⇒ 冒烟直接给 U触发/年龄阈值 NMSE **4.77×**
      （0/9 更优）。根因不是"信号无用"，而是**残差的尺度随 age 变**：
      小 h 的残差量级本来就小、大 h 的本来就大 ⇒ 一个全局 τ 在**小 h 处等于
      永不触发、在大 h 处等于几乎总触发** ⇒ 实际退化成"只在老年龄才发"
      （与代码里早记过的 U 阈值「坑 2：尺度不均」是同一个坑）。
      ⇒ 改成 **z 分数** `(U − Û(age)) / Ŝ(age)`，Ŝ = 条件 MAD（1.4826×）。
        这样 τ 无量纲、可跨 age 比，才是真正的"给定年龄后，U 是否异常高"。

    `u_hat[h]` / `u_scale[h]` = 离线 U 在年龄 h 处的**条件中位数 / 条件 MAD**（形状 (H,)）。
    `state` 由 `run_closed_loop_control` 每步写入 `U` 与 `age`（control.py 已接线）。
    """
    t_ = float(tau)
    p = float(min(max(loss_prob, 0.0), 1.0))
    n_h = int(u_hat.shape[0])

    def _f(t: int, rng: np.random.Generator) -> bool:
        U = float(state.get("U", 0.0))
        age = int(state.get("age", 0))
        h = min(max(age, 1), n_h)              # age=0（刚收到）⇒ 用 h=1 的基准；U 也已归零
        z = (U - float(u_hat[h - 1])) / float(u_scale[h - 1])
        if z < t_:
            return False
        return bool(rng.random() >= p)

    def _reset(rng: np.random.Generator | None = None) -> None:
        state["U"] = 0.0
        state["age"] = 0

    _f.reset = _reset                                   # type: ignore[attr-defined]
    _f.__name__ = f"resid(tau={t_:.4f},p={p})"
    return _f


def _hybrid_schedule(K: int, tau: float, loss_prob: float, state: dict,
                     u_hat: np.ndarray, u_scale: np.ndarray) -> object:
    """★ X40（决定性版本）：**混合触发** —— `age ≥ K` **或** `z(age) ≥ τ`。

    ★ 为什么纯残差触发（`_resid_schedule`）不算决定性测试：
      冒烟实测它在**同率下 E[age] 7.9–14.6 vs 年龄阈值 4.0**（更差），
      机制是「低 U 轨迹被**无限期**推迟发送」—— 而低 U 恰恰是模型**自信**的地方，
      自信≠正确（P3 的 AUC 只有 0.601）⇒ 误差在那里累积。
      ⇒ 这是**「残差信息净有害」**的证据，但**不是「残差信息无边际价值」**的证据：
        前者把安全网也一并撤了，两件事混在一起。

    ★★ 混合触发把两件事分开：**年龄阈值 K 是安全网（保证不塌），残差 τ 只能**加速**。
      ⇒ 与**纯年龄阈值（同 K）** 在**等预算**下比：
          混合更优 ⇒ U 有**正的边际价值**（能提前把该发的发出去）
          混合不优 ⇒ U 的边际价值为 0 或负 ⇒ **X38 的负面结论升级为「与实现无关」**

    三条退化（S_h 自检钉死）：
      τ = +∞ ⇒ 恒等于纯年龄阈值 K
      K = 1  ⇒ 恒等于每步都发（T=1）
      τ = −∞ ⇒ 恒等于 T=1
    """
    K_ = int(K)
    t_ = float(tau)
    p = float(min(max(loss_prob, 0.0), 1.0))
    n_h = int(u_hat.shape[0])

    def _f(t: int, rng: np.random.Generator) -> bool:
        age = int(state.get("age", 0))
        if age < K_:                       # 未到安全网 ⇒ 看残差是否要求**提前**
            h = min(max(age, 1), n_h)
            z = (float(state.get("U", 0.0)) - float(u_hat[h - 1])) / float(u_scale[h - 1])
            if z < t_:
                return False
        return bool(rng.random() >= p)

    def _reset(rng: np.random.Generator | None = None) -> None:
        state["U"] = 0.0
        state["age"] = 0

    _f.reset = _reset                                   # type: ignore[attr-defined]
    _f.__name__ = f"hybrid(K={K_},tau={t_:.3f},p={p})"
    return _f


def _conformal_schedule(K: int, tol: float, loss_prob: float, state: dict,
                        tab: dict) -> object:
    """★ X42：**共形信封触发** —— `age ≥ K` **或** `Ê_α(age, U) ≥ tol`。

    ★ 与 X40 `_hybrid_schedule` 的**唯一**差别 = 触发统计量：
        X40：`z = (U − Û(age))/Ŝ(age) ≥ τ`，τ 取 **z 池的分位点**
             ⇒ 「给定年龄后，U 是否**异常**高」（尺度锚在**调度**上）
        X42：`Ê_α(age, U) ≥ tol`，Ê = **共形校准的归一化误差上界**
             ⇒ 「按校准后的上界，误差是否**要超容限**」（尺度锚在**任务**上）
      规则结构（安全网 K + 只准加速）逐字相同 ⇒ 这是**只换统计量**的消融。

    ★ 为什么 tol 是任务量纲而不是分位点：X40 的 τ 是**调出来的旋钮**，
      换个环境就没意义；tol 是「能容忍多大误差」，可以与环境无关地先写死。
      代价：tol 与场景尺度绑定 ⇒ 跨场景比较时必须重标（已在引用限定里写明）。

    ★ 退化（S_h' 钉死）：tol=+∞ ⇒ 恒等纯年龄阈值；K=1 或 tol=−∞ ⇒ 恒等 T=1。
    """
    K_ = int(K)
    t_ = float(tol)
    p = float(min(max(loss_prob, 0.0), 1.0))
    n_h = int(tab["H"])

    def _f(t: int, rng: np.random.Generator) -> bool:
        age = int(state.get("age", 0))
        if age < K_:
            h = min(max(age, 1), n_h)
            e_hat = float(conf_envelope(tab, np.array([h]),
                                        np.array([float(state.get("U", 0.0))]))[0])
            if e_hat < t_:
                return False
        return bool(rng.random() >= p)

    def _reset(rng: np.random.Generator | None = None) -> None:
        state["U"] = 0.0
        state["age"] = 0

    _f.reset = _reset                                   # type: ignore[attr-defined]
    _f.__name__ = f"conformal(K={K_},tol={t_:g},p={p})"
    return _f


def _sim_conformal(K: int, tol: float, per: float, U_pool: np.ndarray, tab: dict,
                   n: int, seed: int) -> tuple[float, float]:
    """★ S_h'：共形触发的**纯调度**仿真（U 从离线池逐步**独立重采样**）。

    ⚠️ 与 `_sim_hybrid` 同一限制：独立重采样**忽略了 U 的自相关**
      ⇒ 只用于**退化检查**（tol=+∞ ⇒ 纯阈值；K=1 / tol=−∞ ⇒ T=1），
      **不得**用于报"共形策略的 E[age]"（那要真闭环）。
    """
    rng = np.random.default_rng(seed)
    p = float(min(max(per, 0.0), 1.0))
    n_h = int(tab["H"])
    age = 0
    tot = 0.0
    ntx = 0
    for t in range(1, n + 1):
        U = 0.0
        if age > 0:
            U = float(U_pool[rng.integers(U_pool.shape[0]),
                             min(max(age, 1), n_h) - 1])
        fire = age >= int(K)
        if not fire:
            h = min(max(age, 1), n_h)
            fire = float(conf_envelope(tab, np.array([h]), np.array([U]))[0]) >= float(tol)
        if fire and rng.random() >= p:
            age = 0
            ntx += 1
        else:
            age += 1
        tot += age
    return tot / n, ntx / n


def _sim_threshold(K: int, per: float, n: int, seed: int) -> tuple[float, float]:
    """纯调度空跑：复核阈值的 E[age] 与送达率闭式（S_a）。"""
    sch = threshold_lossy_schedule(K, per)
    rng = np.random.default_rng(seed)
    sch.reset(rng)
    age = 0
    tot = 0.0
    ntx = 0
    for t in range(1, n + 1):
        if bool(sch(t, rng)):
            ntx += 1
            age = 0
        else:
            age += 1
        tot += age
    return tot / n, ntx / n


def _age_mc(policy: str, per: float, knob: int, n_ep: int, steps: int,
            n_rep: int = 10, seed0: int = 31337) -> tuple[float, float]:
    """★ 本点「闭环实测 E[age]」的 **MC 均值 + 采样标准误**（纯调度空跑，与 X35 同精神）。

    ★★ 必须**复刻闭环的采样结构**，否则标准误会被系统性低估（第 17 次自证伪的
    直接教训，本号第一版又犯了一次变体）：
      · 闭环是 **n_ep 条 episode × steps 步**，且 **每条 episode 开头 age 归零**；
      · 我第一版只跑 **1 条 12000 步的长序列** ⇒ PER=0 时年龄是确定的，
        std 直接塌成 0，落到 1e-3 地板上 ⇒ 把 0.0145 步的**有限窗口偏差**
        （300 步不是 T 的整数倍）报成 **−13σ / −20σ 的假数**。
    ⇒ 这里改成同结构 MC，并且**拿 MC 均值（不是解析值）当参照**：
      MC 与实测共享同一个有限窗口偏差 ⇒ 偏差抵消掉，剩下的才是真噪声。

    Returns: (mc_mean, se)
    """
    vals = []
    for i in range(n_rep):
        rng = np.random.default_rng(seed0 + 977 * (i + 1) + knob)
        tot = 0.0
        n = 0
        for _ in range(int(n_ep)):
            sch = (periodic_lossy_schedule(knob, per) if policy == "periodic"
                   else threshold_lossy_schedule(knob, per))
            reset = getattr(sch, "reset", None)
            if callable(reset):
                reset(rng)
            age = 0
            for t in range(1, int(steps) + 1):
                if bool(sch(t, rng)):
                    age = 0
                else:
                    age += 1
                tot += age
                n += 1
        vals.append(tot / max(n, 1))
    v = np.asarray(vals, dtype=float)
    return float(v.mean()), (float(v.std(ddof=1)) if v.size > 1 else 0.0)


def _mkrow(policy: str, per: float, knob: int, r: dict, tail_mass: float,
           KC: int, mc: tuple[float, float], label: str = "",
           u_thr: float | None = None, meas_steps: int = 0) -> dict:
    for k in ("est_nmse", "escape_rate", "mean_dist", "mean_dist_tail", "p95_dist",
              "mean_age", "max_age", "tx_rate", "n_steps", "mean_ep_len"):
        if not np.isfinite(float(r[k])):
            raise FloatingPointError(f"★ S_f（R14）：{policy}/{knob} 指标 {k} 非有限")
    if policy == "periodic":
        ana = float(np.dot(np.arange(KC + 1), periodic_age_pmf(per, knob, KC)))
    elif policy == "threshold":
        ana = float(np.dot(np.arange(KC + 1), threshold_age_pmf(knob, per, KC)))
    else:
        ana = float("nan")          # 不确定性触发：年龄分布无闭式（P4 只做实测对照）
    emp = float(r["mean_age"])
    mc_mean, mc_se = float(mc[0]), float(mc[1])
    # ★★ 容差地板（两条，各自独立，都不是"拍一个看着合理的数"）：
    #   (1) **随机性地板** = 3×SE。不许用固定百分比 —— 高丢包下 sqrt(Var(age))
    #       与 E[age] 同量级，8% 只有 1.8σ，会被纯噪声打穿（第 17 次自证伪）。
    #   (2) **结构性地板** = 0.05 步。PER=0 时调度**完全确定**（SE=0），
    #       但闭环 episode 长度 ≠ MC 的固定窗口 ⇒ 残留的是**窗口偏差**不是噪声；
    #       除以 1e-3 会造出 1e8σ 这种纯除零假数（X35 第一版报过 7.2e8σ）。
    #       0.05 步 = 采样步的 1/20，比本实验报告的任何效应（≥10%）小 2 个数量级。
    #   utrigger 没有年龄闭式/MC ⇒ 直接不参与该断言（NaN，不是 0）。
    se = max(mc_se, 0.05 / 3.0) if np.isfinite(mc_se) else float("nan")
    dev = ((emp - mc_mean) / se) if (np.isfinite(mc_mean) and np.isfinite(se)) \
        else float("nan")
    # ★ 删失：episode 因逃逸提前结束 ⇒ 长年龄被系统性切掉 ⇒ 只可能把均值**拉低**
    #   （X35 已确立的口径）⇒ 这类点只做**单侧**断言，不做双侧。
    ep_frac = float(r["mean_ep_len"]) / max(float(max(int(meas_steps), 1)), 1)
    censored = bool(ep_frac < 0.98)
    # ★★ X39 口径修正（2026-09-29）：`diverged` 必须**三族同标准**。
    #   原实现只在 utrigger 分支里算（旧第 652 行）⇒ 配对比较只剔 utrigger 的发散点、
    #   却把 threshold 的发散点留在曲线里（PER=0.3/K=32 的 NMSE=1040 就在里面）
    #   ⇒ **口径不对称**。已离线复核影响：
    #     X38（同方差）三族 0 发散 ⇒ 对 X38 零影响；
    #     X39（异方差）periodic 4 / threshold 14 / utrigger 8 ⇒ 对称化后 P4' 仍不成立。
    diverged = bool(float(r["est_nmse"]) > 1.0 or float(r["mean_age"]) > KC)
    return {"policy": policy, "per": float(per), "knob": int(knob),
            "diverged": diverged,
            "u_thr": (None if u_thr is None else float(u_thr)),
            "label": label,
            "tail_mass": float(tail_mass),
            "tx_rate": float(r["tx_rate"]),
            "est_nmse": float(r["est_nmse"]),
            "escape_rate": float(r["escape_rate"]),
            "mean_dist": float(r["mean_dist"]),
            "mean_dist_tail": float(r["mean_dist_tail"]),
            "p95_dist": float(r["p95_dist"]),
            "mean_age": emp, "max_age": int(r["max_age"]),
            "age_analytic": ana,
            # ★ 与 MC 均值比（同结构 ⇒ 有限窗口偏差抵消）⇒ 剩下的才是真噪声
            "age_mc": mc_mean, "age_se": float(se),
            "age_dev_sigma": float(dev),
            # 解析值 vs MC 均值的差 = 有限窗口偏差（不是噪声，单独记）
            "age_window_bias": ((float(mc_mean - ana)
                                 if (np.isfinite(ana) and np.isfinite(mc_mean)) else None)),
            "ep_frac": float(ep_frac), "censored": censored,
            "n_steps": int(r["n_steps"]),
            "mean_ep_len": float(r["mean_ep_len"])}


if __name__ == "__main__":
    main()
