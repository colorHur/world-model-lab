#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""脚本 15 —— **X12 最小版：σ 头的校准检查**（B1 的第一块）。

★ 问题
------
`GaussianWorldModel` 的方差头（X28）宣称"转移是 N(μ, diag σ²)"。
**校准**问的是：这个 σ 说的是实话吗？

若校准，则**标准化残差** r_d = (z_target − μ_d) / σ_d 应服从 N(0,1)，即：
  - 每维 std(r) ≈ 1
  - |r| ≤ 1 的比例 ≈ 68.3%（±1σ 名义）
  - |r| ≤ 1.96 的比例 ≈ 95%（±2σ 名义）

系统性偏离有两种方向，处置完全不同：
  - std(r) > 1（σ 说小了，过度自信）⇒ 温度缩放 τ = std(r) > 1，乘上去即可；
  - std(r) < 1（σ 说大了，过度保守）⇒ τ < 1。
**τ 本身就是一个可报告的数字** —— 温度缩放的解析解（一维高斯 NLL 最优化）。

★ 口径声明（不许混）
--------------------
这里测的是**潜空间**单步转移的校准（对 `encode(obs_next)`），
**不是**观测空间多步预测的校准 —— 后者需要 obs 空间的方差头，是 X12 的完整版。
B1 的最终目标（决策用的可信不确定度）在 obs 空间；本脚本是通往那里的第一块。

运行
----
    python scripts/15_sigma_calibration.py --quick
"""

from __future__ import annotations

import argparse
import json
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
from wmlab.envs.noisy_pendulum import NoisyPendulumAdapter
from wmlab.models.prob_world_model import GaussianWorldModel
from wmlab.train import train_world_model
from wmlab.utils import describe_device, get_device, load_config, output_dir, set_seed
from wmlab.utils.plot import apply_style, save_fig

NOMINAL = [(0.5, 0.3829), (1.0, 0.6827), (1.5, 0.8664), (1.96, 0.95), (2.5, 0.9876)]


@torch.no_grad()
def standardized_residuals(model, obs, act, nxt, device, batch=8192):
    """返回逐维标准化残差 r = (z_target − μ)/σ，形状 (N, latent)。"""
    rs = []
    for i in range(0, obs.shape[0], batch):
        z = model.encode(obs[i:i + batch].to(device))
        mu, sd, _ = model.next_latent_dist(z, act[i:i + batch].to(device))
        z_t = model.encode(nxt[i:i + batch].to(device))
        rs.append(((z_t - mu) / sd).cpu())
    return torch.cat(rs).numpy()


def main():
    ap = argparse.ArgumentParser(description="X12 最小版：σ 头校准检查")
    ap.add_argument("--config", default="configs/pendulum.yaml")
    ap.add_argument("--sigma-env", type=float, default=0.3)
    ap.add_argument("--latent-dim", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--n-episodes", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--tag", default="15_sigma_calibration")
    args = ap.parse_args()
    if args.quick:
        args.epochs, args.n_episodes = 20, 60

    cfg = load_config(args.config)
    cfg["train"]["epochs"] = args.epochs
    apply_style()
    device = get_device(args.device or cfg.get("device"))
    out = output_dir(cfg)

    print(f"[15] X12 最小版 · σ 头校准检查  σ_env={args.sigma_env}  latent_dim={args.latent_dim}  "
          f"epochs={args.epochs}  device={describe_device(device)}")

    t0 = time.time()
    env = NoisyPendulumAdapter(seed=args.seed, noise_std=args.sigma_env,
                               max_steps=int(cfg["env"].get("max_steps", 200)))
    eps = collect_random_episodes(env, n_episodes=args.n_episodes, seed=args.seed,
                                  max_steps=int(cfg["env"].get("max_steps", 200)))
    # 切分：90% 训练 / 10% 校准检查（episode 级，避免同轨迹泄漏）
    n_val = max(10, len(eps) // 10)
    val_eps, train_eps = eps[:n_val], eps[n_val:]
    set_seed(args.seed)
    m = GaussianWorldModel(obs_dim=env.obs_dim, act_dim=env.act_dim,
                           latent_dim=args.latent_dim, hidden=int(cfg["model"]["hidden"]),
                           discrete_act=False).to(device)
    tr = tuple(torch.as_tensor(x) for x in transitions_from_episodes(train_eps))
    va = tuple(torch.as_tensor(x) for x in transitions_from_episodes(val_eps))
    train_world_model(m, tr, va, cfg, device, verbose=False)
    m.eval()

    obs = torch.as_tensor(np.concatenate([e["obs"][:-1] for e in val_eps]))
    act = torch.as_tensor(np.concatenate([e["act"] for e in val_eps]))
    nxt = torch.as_tensor(np.concatenate([e["obs"][1:] for e in val_eps]))
    r = standardized_residuals(m, obs, act, nxt, device)

    per_dim_std = r.std(axis=0)
    tau = float(np.mean(per_dim_std))          # 温度缩放解析解
    flat = r.flatten()

    print(f"[15] 校准集：{len(val_eps)} 集 / {flat.size} 个残差（潜空间，逐维标准化）")
    print(f"[15] 逐维 std(r) = {np.round(per_dim_std, 3).tolist()}")
    print(f"[15] ★ 温度缩放因子 τ = mean std(r) = {tau:.3f}"
          f"  ({'过度自信，σ 需放大' if tau > 1.05 else '过度保守，σ 需缩小' if tau < 0.95 else '校准良好'})")

    cov_rows = []
    print(f"[15] {'名义':>6s} {'实测覆盖率':>10s} {'τ 修正后':>10s}")
    for k, nominal in NOMINAL:
        emp = float(np.mean(np.abs(flat) <= k))
        emp_t = float(np.mean(np.abs(flat / tau) <= k))
        cov_rows.append({"k": k, "nominal": nominal, "empirical": emp,
                         "after_tau": emp_t})
        print(f"[15] {nominal * 100:5.1f}% {emp * 100:9.2f}% {emp_t * 100:9.2f}%")

    # 校准曲线（reliability diagram 的连续版）
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax = axes[0]
    ks = np.linspace(0.1, 3.0, 60)
    from math import erf, sqrt
    ax.plot(ks, [erf(k / sqrt(2)) for k in ks], "k--", lw=1.2, label="ideal (calibrated)")
    ax.plot(ks, [float(np.mean(np.abs(flat) <= k)) for k in ks], "-",
            color="#E45756", label=f"raw (tau={tau:.2f})")
    ax.plot(ks, [float(np.mean(np.abs(flat / tau) <= k)) for k in ks], "-",
            color="#4C78A8", label="after tau scaling")
    ax.set_xlabel("interval half-width k (in units of sigma)")
    ax.set_ylabel("empirical coverage")
    ax.set_title("(1) coverage of latent one-step transition")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

    ax = axes[1]
    ax.hist(flat, bins=80, range=(-4, 4), density=True, color="#72B7B2", alpha=0.8)
    xs = np.linspace(-4, 4, 200)
    ax.plot(xs, np.exp(-xs ** 2 / 2) / np.sqrt(2 * np.pi), "k--", lw=1.2,
            label="N(0,1)")
    ax.set_xlabel("standardized residual (z_target - mu) / sigma")
    ax.set_ylabel("density")
    ax.set_title(f"(2) residuals (tau = {tau:.2f})")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

    fig.suptitle(f"wmlab X12-min · sigma-head calibration · pendulum sigma_env="
                 f"{args.sigma_env} · latent={args.latent_dim}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    png = save_fig(fig, out / f"{args.tag}.png")
    plt.close(fig)

    summary = {"experiment": "X12-min sigma-head calibration",
               "sigma_env": args.sigma_env, "latent_dim": args.latent_dim,
               "epochs": args.epochs, "n_val_episodes": len(val_eps),
               "per_dim_std": [float(v) for v in per_dim_std],
               "tau": tau,
               "coverage": cov_rows, "seconds": time.time() - t0}
    js = out / f"{args.tag}.json"
    js.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    torch.save({"state_dict": m.state_dict(), "obs_dim": env.obs_dim,
                "act_dim": env.act_dim, "latent_dim": args.latent_dim,
                "hidden": int(cfg["model"]["hidden"]), "sigma_env": args.sigma_env},
               out / f"{args.tag}.pt")
    print(f"[15] 用时 {time.time()-t0:.1f}s")
    print(f"[15] 图 -> {png}")
    print(f"[15] 指标 -> {js}")
    print("[15] OK")


if __name__ == "__main__":
    main()
