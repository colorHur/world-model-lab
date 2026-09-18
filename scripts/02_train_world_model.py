#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""脚本 02 —— 训练一个最小世界模型，并测量它的「可靠视界」。

这是本仓库的核心脚本。它回答一个问题：

    世界模型在潜空间里 rollout 到第几步开始不可靠？

产出 4 张子图（输出为 outputs/02_world_model.png）：
  ① 训练/验证 loss 曲线            —— 确认训练确实收敛（不是"跑了但没学"）
  ② NMSE vs rollout 视界（开环/闭环）—— 核心研究图，误差随步数增长
  ③ 一条示例轨迹的真值 vs 多步预测    —— 直观展示"第几步开始飘"
  ④ 误差增长率（每步 NMSE 增量）      —— 判断误差是线性累积还是自我放大

并打印 reliable horizon（首次超阈的步数，线性插值）。

运行
----
    python scripts/02_train_world_model.py
    python scripts/02_train_world_model.py --epochs 60 --device cuda   # 云端
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

from wmlab.data import collect_random_episodes, split_episodes, transitions_from_episodes
from wmlab.envs import make_env
from wmlab.eval import reliable_horizon
from wmlab.models import MLPWorldModel
from wmlab.rollout import closed_loop_error_curve, imagine, multi_step_error_curve
from wmlab.utils import (count_params, describe_device, get_device, load_config,
                         output_dir, set_seed)
from wmlab.utils.plot import apply_style, save_fig


def parse_args():
    p = argparse.ArgumentParser(description="训练世界模型并测量可靠视界")
    p.add_argument("--config", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--tag", default="02_world_model", help="输出文件名前缀")
    return p.parse_args()


def train_world_model(model, tr, va, cfg, device, verbose=True):
    """监督训练：预测下一步观测。返回 loss 历史。"""
    tcfg = cfg["train"]
    obs, act, nxt = tr
    obs_v, act_v, nxt_v = va

    opt = torch.optim.Adam(model.parameters(), lr=float(tcfg["lr"]),
                           weight_decay=float(tcfg.get("weight_decay", 0.0)))
    n = obs.shape[0]
    bs = int(tcfg["batch_size"])
    epochs = int(tcfg["epochs"])

    hist = {"train_total": [], "train_recon": [], "train_latent": [], "val_total": []}
    g = torch.Generator(device="cpu").manual_seed(int(cfg["seed"]))

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=g)
        tot = rec = lat = 0.0
        nb = 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            b_obs = obs[idx].to(device)
            b_act = act[idx].to(device)
            b_nxt = nxt[idx].to(device)
            loss, parts = model.loss(b_obs, b_act, b_nxt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if tcfg.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(tcfg["grad_clip"]))
            opt.step()
            tot += float(loss.item()); rec += float(parts["recon"].item())
            lat += float(parts["latent"].item()); nb += 1
        hist["train_total"].append(tot / max(nb, 1))
        hist["train_recon"].append(rec / max(nb, 1))
        hist["train_latent"].append(lat / max(nb, 1))

        model.eval()
        with torch.no_grad():
            vloss, _ = model.loss(obs_v.to(device), act_v.to(device), nxt_v.to(device))
        hist["val_total"].append(float(vloss.item()))

        if verbose and (ep % max(1, epochs // 10) == 0 or ep == epochs - 1):
            print(f"  epoch {ep:4d}  train={hist['train_total'][-1]:.6f}  "
                  f"val={hist['val_total'][-1]:.6f}  (recon={hist['train_recon'][-1]:.6f} "
                  f"latent={hist['train_latent'][-1]:.6f})")
    return hist


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.episodes is not None:
        cfg["data"]["n_episodes"] = args.episodes
    if args.seed is not None:
        cfg["seed"] = args.seed

    seed = int(cfg["seed"])
    set_seed(seed)
    # 统一样式：中文字体由代码显式设置，不依赖 MPLCONFIGDIR 等进程环境
    apply_style()
    device = get_device(args.device or cfg.get("device"))
    out = output_dir(cfg)

    print(f"[02] env={cfg['env']['id']}  seed={seed}  device={describe_device(device)}")

    # ---------- 1) 采数据 ----------
    env = make_env(cfg["env"]["id"], seed=seed)
    episodes = collect_random_episodes(env, n_episodes=int(cfg["data"]["n_episodes"]),
                                       seed=seed, max_steps=cfg["env"].get("max_steps"))
    env.close()
    lens = np.asarray([e["length"] for e in episodes])
    print(f"[02] 采集 {len(episodes)} 条 episode，共 {lens.sum()} 步；"
          f"长度 mean={lens.mean():.1f} max={lens.max()}")

    train_eps, val_eps = split_episodes(episodes, float(cfg["train"]["val_ratio"]), seed)

    # ---------- 2) 建模型 ----------
    mcfg = cfg["model"]
    model = MLPWorldModel(
        obs_dim=env.obs_dim, act_dim=env.act_dim,
        latent_dim=int(mcfg["latent_dim"]), hidden=int(mcfg["hidden"]),
        discrete_act=bool(mcfg["discrete_act"]) and env.is_discrete,
    ).to(device)
    print(f"[02] 世界模型参数量 = {count_params(model):,}  "
          f"(obs_dim={env.obs_dim}, act_dim={env.act_dim}, latent={mcfg['latent_dim']})")

    # ---------- 3) 训练 ----------
    tr = tuple(torch.as_tensor(x) for x in transitions_from_episodes(train_eps))
    va = tuple(torch.as_tensor(x) for x in transitions_from_episodes(val_eps))
    print(f"[02] 训练样本 {tr[0].shape[0]} 条，验证样本 {va[0].shape[0]} 条")
    t0 = time.time()
    hist = train_world_model(model, tr, va, cfg, device)
    print(f"[02] 训练完成，用时 {time.time()-t0:.1f}s")

    # ---------- 4) 多步 rollout 误差曲线 ----------
    hs = list(cfg["eval"]["horizons"])
    thr = float(cfg["eval"]["err_threshold"])
    mode = cfg["eval"].get("threshold_mode", "rel")

    curve_open = multi_step_error_curve(model, val_eps, hs, device, seed=seed)
    curve_closed = closed_loop_error_curve(model, val_eps, hs, device, seed=seed)

    rh_open = reliable_horizon(curve_open["horizons"], curve_open["nmse"], thr, mode)
    rh_closed = reliable_horizon(curve_closed["horizons"], curve_closed["nmse"], thr, mode)
    print(f"[02] ★ reliable horizon（NMSE>{thr}）: 开环={rh_open}  闭环={rh_closed}")

    # ---------- 5) 出图 ----------
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5))

    # ① loss
    ax = axes[0, 0]
    ax.plot(hist["train_total"], label="train total", color="#4C78A8")
    ax.plot(hist["val_total"], label="val total", color="#E45756")
    ax.plot(hist["train_recon"], label="train recon", color="#54A24B", alpha=0.7, linestyle="--")
    ax.plot(hist["train_latent"], label="train latent", color="#F58518", alpha=0.7, linestyle="--")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (log)")
    ax.set_title("① World-model training loss")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

    # ② NMSE vs horizon
    ax = axes[0, 1]
    ax.plot(curve_open["horizons"], curve_open["nmse"], "o-", color="#4C78A8", label="open-loop")
    ax.plot(curve_closed["horizons"], curve_closed["nmse"], "s--", color="#E45756", label="closed-loop")
    ax.axhline(thr, color="#888", linestyle=":", linewidth=1.2,
               label=f"threshold = {thr}")
    if rh_open is not None:
        ax.axvline(rh_open, color="#4C78A8", linestyle=":", alpha=0.7)
        ax.annotate(f"H* = {rh_open:.1f}", xy=(rh_open, thr),
                    xytext=(rh_open + 0.6, thr * 4), color="#4C78A8", fontsize=9,
                    arrowprops=dict(arrowstyle="->", color="#4C78A8", lw=0.8))
    if rh_closed is not None:
        ax.annotate(f"H* = {rh_closed:.1f}", xy=(rh_closed, thr),
                    xytext=(rh_closed + 0.6, thr * 12), color="#E45756", fontsize=9,
                    arrowprops=dict(arrowstyle="->", color="#E45756", lw=0.8))
    ax.set_yscale("log")
    ax.set_xlabel("Rollout horizon (steps)"); ax.set_ylabel("NMSE (log)")
    ax.set_title("② Prediction error vs horizon  ★core result")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

    # ③ 示例轨迹
    ax = axes[1, 0]
    long_eps = sorted(val_eps, key=lambda e: e["length"], reverse=True)[0]
    H = min(hs[-1], long_eps["length"] - 2)
    o0 = long_eps["obs"][0]
    acts = long_eps["act"][:H]
    tgt = long_eps["obs"][1:1 + H]
    with torch.no_grad():
        pred = imagine(model,
                       torch.as_tensor(o0, device=device).unsqueeze(0),
                       torch.as_tensor(acts, device=device).unsqueeze(0))[0].cpu().numpy()
    n_show = min(2, env.obs_dim)
    for d in range(n_show):
        ax.plot(range(1, H + 1), tgt[:H, d], color="#4C78A8", alpha=0.9,
                label=f"truth (dim {d})" if d == 0 else None)
        ax.plot(range(1, H + 1), pred[:H, d], color="#E45756", linestyle="--",
                label=f"world-model (dim {d})" if d == 0 else None)
    ax.set_xlabel("Rollout step"); ax.set_ylabel("Observation value")
    ax.set_title("③ Open-loop rollout: prediction drifts from truth")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

    # ④ 每步误差增量
    ax = axes[1, 1]
    # ★ 口径修正（2026-09-18）：horizons 间隔不等（…2,2,3,5,5,10,10,25,25,50,40），
    #   直接 np.diff(nm) 会把「间隔变大」混进「误差增长加速」。按步数归一才是每步增量。
    hs_f = np.asarray(curve_open["horizons"], dtype=float)
    nm = np.asarray(curve_open["nmse"], dtype=float)
    inc = np.diff(nm) / np.diff(hs_f)
    centers = 0.5 * (hs_f[:-1] + hs_f[1:])
    ax.bar(centers, inc, width=0.6 * np.diff(hs_f), color="#72B7B2")
    ax.set_xlabel("Horizon (steps) · bar at interval midpoint, width ∝ interval")
    ax.set_ylabel("Δ NMSE per step (normalized)")
    ax.set_title("④ Per-step error growth rate (is it self-amplifying?)")
    ax.grid(alpha=0.25, axis="y")

    fig.suptitle(
        f"wmlab · world model on {cfg['env']['id']} · seed={seed} · "
        f"latent={mcfg['latent_dim']} · params={count_params(model):,} · device={device.type}",
        fontsize=10,
    )
    fig.tight_layout()
    png = save_fig(fig, out / f"{args.tag}.png")
    plt.close(fig)

    summary = {
        "config": cfg,
        "device": str(device),
        "n_params": count_params(model),
        "data": {"n_episodes": len(episodes), "total_steps": int(lens.sum()),
                 "len_mean": float(lens.mean()), "len_max": int(lens.max())},
        "final_loss": {"train": hist["train_total"][-1], "val": hist["val_total"][-1]},
        "curve_open_loop": curve_open,
        "curve_closed_loop": curve_closed,
        "reliable_horizon_open": rh_open,
        "reliable_horizon_closed": rh_closed,
        "threshold": thr,
        "threshold_mode": mode,
    }
    js = out / f"{args.tag}.json"
    js.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[02] 图已保存 -> {png}")
    print(f"[02] 指标已保存 -> {js}")
    print("[02] OK")


if __name__ == "__main__":
    main()
