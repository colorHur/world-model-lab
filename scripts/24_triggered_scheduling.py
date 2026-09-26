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
from wmlab.eval.tracking import (periodic_age_pmf, periodic_age_tail,
                                 periodic_lossy_schedule, periodic_mean_age,
                                 threshold_age_pmf, threshold_age_tail,
                                 threshold_delivery_rate, threshold_equivalent_period,
                                 threshold_lossy_schedule, threshold_mean_age)
from wmlab.models.prob_world_model import GaussianWorldModel
from wmlab.train import train_world_model
from wmlab.utils import count_params, get_device, load_config, output_dir, set_seed
from wmlab.utils.plot import PALETTE, apply_style, save_fig


def parse_args():
    p = argparse.ArgumentParser(description="X38 触发式调度：等预算下周期 vs 阈值 vs 不确定性")
    p.add_argument("--config", default="configs/uav_triggered.yaml")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--tag", default="24_triggered_scheduling")
    p.add_argument("--pers", default=None)
    p.add_argument("--quick", action="store_true")
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

    pers = [float(x) for x in dz["per_list"]]
    T_list = [int(x) for x in dz["period_list"]]
    K_list = [int(x) for x in dz["threshold_list"]]
    max_tail = float(dz.get("max_tail_mass", 0.02))
    WARMUP = int(cfg["task"]["warmup_steps"])
    n_task_ep = int(cfg["task"]["n_episodes"])

    print("[24] " + "=" * 78)
    print("[24] ★ X38 触发式调度：等传输预算下 周期 / 年龄阈值 / 不确定性触发")
    print(f"[24]   PER ∈ {pers}    T ∈ {T_list}    K ∈ {K_list}")
    print("[24]   预写：P1 等预算阈值 E[age] ≤ 周期（PER=0 恒等、U 形）"
          "  P2 闭环同向  P3 ρ_cond≥0.3 且 AUC_cond≥0.6")

    # ============================================================ 1) 训练
    env = make_env(cfg["env"]["id"], seed=seed,
                   noise_std=float(cfg["env"]["noise_std"]),
                   max_steps=int(cfg["env"]["max_steps"]))
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
                      max_steps=int(cfg["env"]["max_steps"]))
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
                     and abs(r["per"] - per) < 1e-12], key=lambda r: r["tx_rate"])
        tt = sorted([r for r in rows if r["policy"] == "threshold"
                     and abs(r["per"] - per) < 1e-12], key=lambda r: r["tx_rate"])
        if len(pp) < 2 or len(tt) < 2:
            continue
        lo = max(min(r["tx_rate"] for r in pp), min(r["tx_rate"] for r in tt))
        hi = min(max(r["tx_rate"] for r in pp), max(r["tx_rate"] for r in tt))
        if not (hi > lo * 1.0001):
            continue
        grid = np.exp(np.linspace(np.log(lo), np.log(hi), 9))
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
    print(f"[24]   条件内 U 变异：CV_cond 中位 {np.median(cvs):.4f}"
          f"（不条件 CV={u_cv:.4f}）⇒ "
          + ("★ U 在给定 h 后**几乎不退化为常数** ⇒ ρ_cond≈0 是「有方差但无预测力」，"
             "比「U 是 age 替身」更强的负面结论"
             if np.median(cvs) > 0.5 * u_cv else
             "U 在给定 h 后接近常数 ⇒ ρ_cond≈0 属「U 是 age 的替身」"))
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
    u_thrs = [float(np.median(U[:, h - 1])) for h in h_stars]
    print("[24]   阈值按「名义触发年龄 h*」标定（thr = 离线 U(h*) 的中位数）："
          + "  ".join(f"h*={h}:{v:.4f}" for h, v in zip(h_stars, u_thrs)))
    n_div = 0
    for per in pers:
        for qi, (h_star, thr) in enumerate(zip(h_stars, u_thrs)):
            state = {"U": 0.0}
            sch = _utrig_schedule(thr, per, state)
            r = run_closed_loop_control(
                env, model, ctrl, sch, n_episodes=n_task_ep, seed=online_seed,
                max_steps=meas_steps, device=device, estimator="model",
                var_g=var_g, label=f"utrigger/thr={thr:.4f}",
                tail_frac=float(cfg["task"]["tail_frac"]), warmup_steps=WARMUP,
                u_state=state)
            # ★ 发散点（阈值太高 ⇒ 几乎不发送 ⇒ rollout 炸掉）：记下来但**不进比较**
            diverged = (float(r["est_nmse"]) > 1.0 or float(r["mean_age"]) > KC)
            n_div += int(diverged)
            row = _mkrow("utrigger", per, h_star, r, float("nan"), KC,
                         (float("nan"), float("nan")), u_thr=float(thr),
                         meas_steps=meas_steps)
            row["diverged"] = bool(diverged)
            # ★ 名义 vs 实际：比值远大于 1 ⇒ 闭环里 σ 长得比离线慢 ⇒ "σ 头睡着"
            row["age_over_hstar"] = float(r["mean_age"]) / float(h_star)
            rows.append(row)
    env.close()
    if n_div:
        print(f"[24]   ⚠ {n_div}/{len(pers) * len(u_thrs)} 个 U 触发点发散"
              f"（NMSE>1 或 E[age]>K={KC}）⇒ 已从等预算比较中剔除，不计入结论")
    for r in rows:
        if r["policy"] == "utrigger":
            print(f"[24]   PER={r['per']:<4g} utrigger   h*={r['knob']:<3d} "
                  f"thr={r['u_thr']:.4f} tx_rate={r['tx_rate']:.4f} "
                  f"E[age]={r['mean_age']:6.2f} (=名义 {r['age_over_hstar']:.2f}×) | "
                  f"NMSE={r['est_nmse']:.5f} 距离={r['mean_dist_tail']:6.3f}m "
                  f"逃逸={r['escape_rate']:.3f}"
                  + ("   ← 发散，剔除" if r.get("diverged") else ""))
    # ★ R12：U 阈值必须真的接线（改阈值必须改变发送率），否则整个 P4 在空跑
    rates_by_per = {}
    for per in pers:
        sub = [r for r in rows if r["policy"] == "utrigger" and abs(r["per"] - per) < 1e-12]
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

    # --- 三方等预算对比（在同一 tx_rate 网格上插值）---
    cmp3 = []
    for per in pers:
        fams = {p: sorted([r for r in rows if r["policy"] == p
                           and abs(r["per"] - per) < 1e-12
                           and not r.get("diverged", False)],
                          key=lambda r: r["tx_rate"])
                for p in ("periodic", "threshold", "utrigger")}
        if any(len(v) < 2 for v in fams.values()):
            continue
        lo = max(min(r["tx_rate"] for r in v) for v in fams.values())
        hi = min(max(r["tx_rate"] for r in v) for v in fams.values())
        if not (hi > lo * 1.0001):
            continue
        grid = np.exp(np.linspace(np.log(lo), np.log(hi), 9))
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
                continue
            with np.errstate(divide="ignore", invalid="ignore"):
                r_ut = vals["utrigger"] / np.maximum(vals["threshold"], 1e-18)
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
    fig, axes = plt.subplots(2, 3, figsize=(17.5, 9.5))
    fig.suptitle("X38 触发式调度：等传输预算下 周期 / 年龄阈值 / 不确定性触发", fontsize=11)

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
                         ("utrigger", PALETTE["purple"], "^")):
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

    save_fig(fig, os.path.join(out, "24_triggered_scheduling.png"))
    plt.close(fig)

    payload = {
        "experiment": "X38 触发式调度：等传输预算下 周期 / 年龄阈值 / 不确定性触发",
        "config": {k: cfg[k] for k in ("env", "controller", "design", "eval", "task")},
        "obs_dim": int(obs_dim), "var_g": var_g,
        "P1_analytic": p1_rows, "P1_verdict": p1_verdict,
        "P2_closed_loop": rows, "P2_matched_rate": cmp_rows,
        "P4_u_thresholds": u_thrs, "P4_matched_rate": cmp3,
        "P3_conditional": {"rho_unconditional": float(rho_all),
                           "rho_conditional_median": float(rho_cond),
                           "auc_conditional_median": float(auc_cond),
                           "u_cv": float(u_cv), "sigma_step_cv": float(s_cv),
                           "per_h": per_h},
        "caveat": ("★ P1/P2（周期 vs 年龄阈值）**不是新贡献**：Sun–Polyanskiy–"
                   "Uysal-Biyikoglu（arXiv:1701.06734 / 1707.02531）已在采样率约束下"
                   "证明阈值策略最优并显式比较过 uniform。本号只作复现 + 移到闭环控制场景。"
                   "★ tx_rate 是**送达率**，不是尝试率（两者只差因子 s，配对关系相同）。"
                   "★ 只用逐包独立丢包，不含突发信道。"),
    }
    with open(os.path.join(out, "24_triggered_scheduling.json"), "w", encoding="utf-8") as f:
        json.dump(jsonable(payload), f, ensure_ascii=False, indent=2)
    print(f"[24] 产物：{os.path.join(out, '24_triggered_scheduling.png')} / .json  "
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
    return {"policy": policy, "per": float(per), "knob": int(knob),
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
