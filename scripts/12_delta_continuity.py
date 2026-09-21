#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""脚本 12 —— **X28：让 δ 从「恒等于 1」变成 (0,1) 上的连续量**。

★ 问题的来源（X22 的负面结论，本脚本是要救活它）
------------------------------------------------
C17（arXiv:2605.15960）的 safe horizon 里，模型误差定义为

    δ = ½ max_{s,a} ‖T(·|s,a) − T′(·|s,a)‖₁        （两个转移分布的**总变差**）

X22 核实原文后证明：本仓库「环境确定 + 模型确定」⇒ 两个转移都是点质量
⇒ TV ∈ {0,1} ⇒ 取 max 后 **δ ≡ 1** ⇒ `H(ε,1) = 1+ε`，定理条件退化成
`γ ≤ ε/(1+ε)`，对任何有意义的 γ 都不成立。**⇒ gap 这个量不可定义。**

补救路径 A（X22 记录里写的）：**给环境注入过程噪声，并让模型输出分布**。
本脚本就是路径 A。两个新件：
  - `wmlab/envs/noisy_pendulum.py`      环境侧：可重置、可关断的过程噪声
  - `wmlab/models/prob_world_model.py`  模型侧：转移输出对角高斯 N(μ, diag σ²)

★ 本脚本判据（不满足就是失败，不许粉饰）
----------------------------------------
| # | 判据 | 意义 |
|---|---|---|
| **D1** | σ_env = 0 时，δ 必须**仍然 ≈ 1** | 复现 X22 的退化，证明测法没错 |
| **D2** | σ_env > 0 时，δ 必须落在 **(0,1)** 且随 σ_env **单调上升** | δ 真的连续了 |
| **D3** | δ 必须随**训练**下降 | δ 是"模型误差"，不是"环境噪声"本身 |
| **D4** | MC 估计必须收敛（双向差 < 容差，样本数扫描稳定） | 否则数字不可引用 |

★ TV 的估计量选择（一个会让数字不可用的技术细节）
-------------------------------------------------
高维（latent_dim=32）下不能用 `½·E_P|1 − q/p|` 估计 TV：
`q/p` 的尾部无界，方差爆炸，估计量在低概率区被少数样本主导。
改用**有界**形式（数学等价）：

    TV(P,Q) = E_P[(1 − q/p)_+] = E_Q[(1 − p/q)_+]

其中 `(·)_+ = max(·, 0)`，被积函数 ∈ [0,1] ⇒ 方差有界。
两个方向各估一次，其差作为**估计质量的诊断**（不是装饰）。

运行
----
    python scripts/12_delta_continuity.py
    python scripts/12_delta_continuity.py --quick
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from wmlab.data import collect_random_episodes, transitions_from_episodes
from wmlab.envs.noisy_pendulum import NoisyPendulumAdapter, assert_matches_gymnasium
from wmlab.eval import reliable_horizon
from wmlab.eval.metrics import per_step_nmse
from wmlab.models.prob_world_model import GaussianWorldModel
from wmlab.rollout import multi_step_error_curve
from wmlab.train import train_world_model
from wmlab.utils import describe_device, get_device, load_config, output_dir, set_seed
from wmlab.utils.plot import apply_style, save_fig

SIGMA_GRID = [0.0, 0.01, 0.03, 0.1, 0.3]


@torch.no_grad()
def tv_diag_gaussian(m0, s0, m1, s1, n=100_000, seed=0, device="cpu"):
    """TV( N(m0, diag s0²), N(m1, diag s1²) ) 的**双向有界**蒙特卡洛估计。

    用 TV = E_P[(1 − q/p)_+] = E_Q[(1 − p/q)_+]（被积函数 ∈ [0,1]）。
    返回 (tv, dpq, dqp)：tv 为双向均值，后两者是两个方向的原始值（其差 = 诊断）。
    """
    d = int(m0.numel())
    c = 0.5 * d * math.log(2.0 * math.pi)

    def logpdf(x, m, s):
        return -0.5 * (((x - m) / s) ** 2).sum(-1) - torch.log(s).sum() - c

    g = torch.Generator(device=device).manual_seed(seed)

    def one_dir(mA, sA, mB, sB):
        x = mA + sA * torch.randn(n, d, generator=g, device=device)
        lr = logpdf(x, mB, sB) - logpdf(x, mA, sA)
        r = torch.exp(torch.clamp(lr, -60.0, 60.0))     # 上溢/下溢都安全：r→∞ 或 0
        return float(torch.mean(torch.clamp(1.0 - r, min=0.0)).item())

    dpq = one_dir(m0, s0, m1, s1)      # 从 P 采
    dqp = one_dir(m1, s1, m0, s0)      # 从 Q 采
    return 0.5 * (dpq + dqp), dpq, dqp


def obs_to_state(obs: np.ndarray) -> np.ndarray:
    """[cos th, sin th, thdot] -> [th, thdot]（θ 由 atan2 唯一恢复）。"""
    return np.array([math.atan2(float(obs[1]), float(obs[0])), float(obs[2])],
                    dtype=np.float32)


@torch.no_grad()
def measure_delta(model, env, episodes, device, n_points=40, n_noise=256,
                  n_mc=100_000, seed=0, mc_scan=None):
    """在若干 (s,a) 上估计 δ = max TV(T_true, T_model)。

    T_true 的估计：在同一个 (s,a) 上采样 `n_noise` 次过程噪声，
    把得到的 next obs 编码成 z，用**经验均值/标准差**拟合一个对角高斯。
    （不做线性化假设 —— 经验拟合对 encode 的非线性是自动成立的。）

    T_model：模型在该 (z,a) 上输出的 N(μ_θ, diag σ_θ²)。
    """
    rng = np.random.default_rng(seed)
    # 采样 (s, a) 点
    flat = [(i, t) for i, ep in enumerate(episodes)
            for t in range(len(ep["obs"]) - 1)]
    idx = rng.choice(len(flat), size=min(n_points, len(flat)), replace=False)
    pick = [flat[int(i)] for i in idx]

    tvs, mus, strues, sig_pred = [], [], [], []
    mc_diag_all = []
    for i, t in pick:
        obs_t = episodes[i]["obs"][t]
        act_t = episodes[i]["act"][t]
        state_t = obs_to_state(obs_t)

        # 真实转移的经验分布（潜空间）
        zs = []
        for _ in range(n_noise):
            xi = float(rng.normal(0.0, env.noise_std)) if env.noise_std > 0 else 0.0
            st = env.step_det(state_t, float(np.asarray(act_t).reshape(-1)[0]), noise=xi)
            th, thdot = float(st[0]), float(st[1])
            o = np.array([np.cos(th), np.sin(th), thdot], dtype=np.float32)
            zs.append(o)
        zs_t = torch.as_tensor(np.asarray(zs, dtype=np.float32), device=device)
        z_true = model.encode(zs_t)
        m_true = z_true.mean(0)
        s_true = z_true.std(0).clamp_min(1e-6)

        # 模型预测的分布
        z = model.encode(torch.as_tensor(obs_t[None], device=device))
        a = torch.as_tensor(np.asarray(act_t).reshape(1, -1), device=device)
        mu, sd, _ = model.next_latent_dist(z, a)
        mu, sd = mu[0], sd[0]

        tv, dpq, dqp = tv_diag_gaussian(m_true, s_true, mu, sd, n=n_mc,
                                        seed=int(rng.integers(1 << 30)), device=device)
        tvs.append(tv)
        mc_diag_all.append(abs(dpq - dqp))
        mus.append(float((mu - m_true).norm().item()))
        strues.append(float(s_true.mean().item()))
        sig_pred.append(float(sd.mean().item()))

        if mc_scan:                      # 收敛性：只扫第一个点，避免开销翻倍
            for n_s in mc_scan:
                tv_s, _, _ = tv_diag_gaussian(m_true, s_true, mu, sd, n=n_s,
                                              seed=7, device=device)
                mc_scan.setdefault(n_s, []).append(tv_s)

    return {
        "delta": float(np.max(tvs)),            # ★ C17 的定义取 max
        "tv_mean": float(np.mean(tvs)),
        "tv_median": float(np.median(tvs)),
        "tv_min": float(np.min(tvs)),
        "tv_values": [float(v) for v in tvs],
        "mu_gap_mean": float(np.mean(mus)),      # ‖μ_θ − m_true‖
        "sigma_true_mean": float(np.mean(strues)),
        "sigma_pred_mean": float(np.mean(sig_pred)),
        "mc_bidir_gap_max": float(np.max(mc_diag_all)),
        "mc_bidir_gap_mean": float(np.mean(mc_diag_all)),
        "n_points": len(tvs),
    }


def main():
    ap = argparse.ArgumentParser(description="X28 δ 连续化")
    ap.add_argument("--config", default="configs/pendulum.yaml")
    ap.add_argument("--sigmas", default=",".join(str(s) for s in SIGMA_GRID))
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--n-episodes", type=int, default=None)
    ap.add_argument("--n-points", type=int, default=40)
    ap.add_argument("--n-noise", type=int, default=256)
    ap.add_argument("--n-mc", type=int, default=100_000)
    ap.add_argument("--latent-dim", type=int, default=None,
                    help="覆盖 cfg 的 latent_dim。★ 诊断 δ 饱和是否为维度效应时用"
                         "（注意：与历史 H* 的可比性随维度改变而失效）")
    ap.add_argument("--device", default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--tag", default="12_delta_continuity")
    args = ap.parse_args()

    cfg = load_config(args.config)
    sigmas = [float(x) for x in args.sigmas.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    epochs = int(args.epochs or (20 if args.quick else cfg["train"]["epochs"]))
    cfg["train"]["epochs"] = epochs
    n_ep = int(args.n_episodes or (60 if args.quick else cfg["data"]["n_episodes"]))

    apply_style()
    device = get_device(args.device or cfg.get("device"))
    out = output_dir(cfg)
    horizons = list(cfg["eval"]["horizons"])
    thr = float(cfg["eval"]["err_threshold"])
    mode = cfg["eval"].get("threshold_mode", "rel")
    mcfg = cfg["model"]

    print(f"[12] X28 · δ 连续化   device={describe_device(device)}  "
          f"seeds={seeds}  epochs={epochs}  n_episodes={n_ep}")
    print(f"[12] σ_env 网格 = {sigmas}")

    # ---------- 阶段 0：σ=0 时必须与 gymnasium 逐位一致 ----------
    worst = assert_matches_gymnasium(seed=0, n_steps=50)
    print(f"[12] 阶段0 · 与 gymnasium Pendulum-v1 一致性: 最大偏差 {worst:.3e}  ✅")

    t0 = time.time()
    rows = []
    mc_scan = {}
    for sigma in sigmas:
        for seed in seeds:
            env = NoisyPendulumAdapter(seed=seed, noise_std=sigma,
                                       max_steps=int(cfg["env"].get("max_steps", 200)))
            eps = collect_random_episodes(env, n_episodes=n_ep, seed=seed,
                                          max_steps=int(cfg["env"].get("max_steps", 200)))
            set_seed(seed)
            m = GaussianWorldModel(obs_dim=env.obs_dim, act_dim=env.act_dim,
                                   latent_dim=int(args.latent_dim or mcfg["latent_dim"]),
                                   hidden=int(mcfg["hidden"]),
                                   discrete_act=False).to(device)
            tr = tuple(torch.as_tensor(x) for x in transitions_from_episodes(eps))
            va = tuple(torch.as_tensor(x) for x in transitions_from_episodes(eps[:6]))
            train_world_model(m, tr, va, cfg, device, verbose=False)
            m.eval()

            d = measure_delta(m, env, eps, device, n_points=args.n_points,
                              n_noise=args.n_noise, n_mc=args.n_mc, seed=seed,
                              mc_scan=(mc_scan if (sigma == sigmas[-1] and
                                                   seed == seeds[0]) else None))
            # 实测 H*（噪声环境下重测；口径与 X2 一致：开环 + 逐点）
            c = multi_step_error_curve(m, eps, horizons, device, seed=seed)
            dense = c["nmse_per_step"]
            vals = [float(dense[h - 1]) for h in horizons]
            h_star = reliable_horizon(horizons, vals, thr, mode)

            rows.append({"sigma_env": sigma, "seed": seed, **d,
                         "h_star": h_star, "nmse_h1": vals[0],
                         "nmse_max": float(np.nanmax(vals))})
            print(f"[12]   σ={sigma:<5g} seed={seed}  δ = {d['delta']:.4f}  "
                  f"(mean {d['tv_mean']:.4f} / median {d['tv_median']:.4f})  "
                  f"H* = {h_star if h_star is not None else float('nan'):7.2f}  "
                  f"[σ_true={d['sigma_true_mean']:.3e} σ_pred={d['sigma_pred_mean']:.3e} "
                  f"μ_gap={d['mu_gap_mean']:.3e} MC双向差={d['mc_bidir_gap_max']:.2e}]")
            env.close()

    # ---------- 判据对账 ----------
    print()
    print("[12] ==== 判据对账 ====")
    by_sigma = {}
    for r in rows:
        by_sigma.setdefault(r["sigma_env"], []).append(r["delta"])
    d_at_zero = by_sigma.get(0.0, [None])[0]
    print(f"[12] D1  σ_env=0 时 δ ≈ 1 ？  δ = {d_at_zero:.4f}   "
          f"{'✅ 复现退化' if d_at_zero is not None and d_at_zero > 0.95 else '❌ 未复现'}")
    mono = [float(np.mean(by_sigma[s])) for s in sigmas]
    inc = all(b >= a - 1e-3 for a, b in zip(mono, mono[1:]))
    print(f"[12] D2  δ 随 σ_env 单调上升？  {[round(x,4) for x in mono]}   "
          f"{'✅' if inc else '❌ 非单调'}")
    in01 = all(0.0 < m < 1.0 for m in mono[1:])
    print(f"[12] D2' σ_env>0 时 δ ∈ (0,1)？  {'✅' if in01 else '❌ 有格子落在端点'}")
    worst_gap = max(r["mc_bidir_gap_max"] for r in rows)
    print(f"[12] D4  MC 双向差最大 = {worst_gap:.2e}  "
          f"{'✅ 估计收敛' if worst_gap < 0.02 else '⚠️ 估计未收敛，数字需谨慎'}")
    if mc_scan:
        print("[12]     MC 样本数收敛扫描（最后一组，第一个点）:")
        for n_s in sorted(mc_scan):
            print(f"[12]       n={n_s:>8d}  TV = {np.mean(mc_scan[n_s]):.5f}")

    # ---------- 出图 ----------
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))

    ax = axes[0]
    xs = [r["sigma_env"] for r in rows]
    ax.plot(xs, [r["delta"] for r in rows], "o-", color="#E45756", label="delta (max TV)")
    ax.plot(xs, [r["tv_median"] for r in rows], "s--", color="#4C78A8",
            label="median TV")
    ax.axhline(1.0, color="k", ls=":", lw=1, label="delta = 1 (degenerate)")
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xlabel("process-noise sigma_env"); ax.set_ylabel("delta")
    ax.set_title("(1) delta vs process noise  (D1/D2)")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

    ax = axes[1]
    ax.plot(xs, [r["sigma_true_mean"] for r in rows], "o-", color="#4C78A8",
            label="sigma_true (empirical)")
    ax.plot(xs, [r["sigma_pred_mean"] for r in rows], "s-", color="#F58518",
            label="sigma_pred (model head)")
    ax.set_xscale("symlog", linthresh=0.01); ax.set_yscale("log")
    ax.set_xlabel("process-noise sigma_env"); ax.set_ylabel("latent sigma")
    ax.set_title("(2) is the variance head calibrated?\n(= B1 first brick)")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

    ax = axes[2]
    hs = [r["h_star"] if r["h_star"] is not None else float("nan") for r in rows]
    ax.plot(xs, hs, "o-", color="#72B7B2")
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xlabel("process-noise sigma_env"); ax.set_ylabel("H* (open loop, steps)")
    ax.set_title("(3) H* under process noise")
    ax.grid(alpha=0.25)

    fig.suptitle(f"wmlab X28 · making delta continuous · epochs={epochs} · "
                 f"n_points={args.n_points} · n_mc={args.n_mc}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    png = save_fig(fig, out / f"{args.tag}.png")
    plt.close(fig)

    summary = {"experiment": "X28 delta continuity", "epochs": epochs,
               "sigma_grid": sigmas, "seeds": seeds, "rows": rows,
               "mc_scan": {str(k): [float(v) for v in vs] for k, vs in mc_scan.items()},
               "gym_consistency_max_dev": worst,
               "seconds": time.time() - t0}
    js = out / f"{args.tag}.json"
    js.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[12] 用时 {time.time()-t0:.1f}s")
    print(f"[12] 图 -> {png}")
    print(f"[12] 指标 -> {js}")
    print("[12] OK")


if __name__ == "__main__":
    main()
