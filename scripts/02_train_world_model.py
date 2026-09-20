#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""脚本 02 —— 训练一个最小世界模型，并测量它的「可靠视界」。

这是本仓库的核心脚本。它回答一个问题：

    世界模型在潜空间里 rollout 到第几步开始不可靠？

产出 4 张子图（输出为 outputs/02_world_model.png）：
  ① 训练/验证 loss 曲线            —— 确认训练确实收敛（不是"跑了但没学"）
  ② NMSE vs rollout 视界（开环/闭环）—— 核心研究图，误差随步数增长
  ③ 一条示例轨迹的真值 vs 多步预测    —— 直观展示"第几步开始飘"
  ④ 逐点误差 vs 视界                 —— 判断误差是线性累积、自我放大还是饱和

并打印 reliable horizon（首次超阈的步数，线性插值）。

★ 误差口径（2026-09-20 修正，由 X26 对账发现）
--------------------------------------------
H* 的定义是"误差**首次**超阈" ⇒ 必须用**逐点**口径（第 h 步自身的误差）。
历史上本脚本用的是**累积**口径（`mse(pred[:, :h], tgt[:, :h])`），它把前 h-1 步的误差
摊成均值 ⇒ **穿越更晚 ⇒ H\* 被系统性抬高**。实测更正：

| 环境 | 曲线 | 累积（历史） | 逐点（正确） |
|---|---|---|---|
| Pendulum | 开环 | 64.83 | **42.99** |
| Pendulum | 闭环 | 51.96 | **39.01** |
| CartPole | 开环 | `None`（测不出） | **12.95** |
| CartPole | 闭环 | 10.11 | **6.96** |

现在两个口径都算、都存 json，图中主线画逐点、累积作淡色参照。
**归一化分母 = 切片目标方差**（与历史口径同源）。

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
import warnings
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
from wmlab.train import train_world_model
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


# ★ 训练循环已抽到 `wmlab/train.py`（2026-09-19），本脚本改为导入，调用点不变。
#   抽取原因：X24 的等价性检验需要在同一进程里训练多个世界模型做逐位对比；
#   若训练实现留在脚本里就会变成两份代码，两次实验的模型将不再可比。
#   抽取后行为与抽取前逐行一致（同 seed 下逐位可复现，已由 04 脚本的 E2 检验覆盖）。


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

    # ★★ 口径修正（2026-09-20，由 X26 对账发现）：
    #   H* 的定义是"误差**首次**超阈"，因此必须用**逐点**口径（每一步自身的误差）。
    #   历史上这里用的是 `curve["nmse"]`（**累积**口径：mse(pred[:, :h], tgt[:, :h])），
    #   它被前 h-1 步的小误差摊薄 ⇒ 系统性高估 H*。
    #   见 wmlab/rollout/imagine.py 的口径说明。累积值仍然保留在 json 里，用于复现历史结论。
    #
    #   ★★ 单位必须与阈值同源（这一条踩过坑）：阈值 0.05 是**归一化**口径（nmse），
    #      所以逐点序列也必须取 `nmse_per_step`，**不能取 `mse_per_step`**。
    #      第一版写成 mse_per_step 后，CartPole 上 H* 反而比累积口径更大（14.91 vs 10.11），
    #      与"累积=逐点的运行平均 ≤ 逐点 ⇒ 累积 H* 必不小于逐点 H*"矛盾 —— 由此查出单位混用。
    def _pointwise(curve):
        """把稠密的逐点 **NMSE** 数组按 horizons 取出来（索引 i ↔ 第 h 步）。"""
        v = np.asarray(curve["nmse_per_step"], dtype=float)
        return [float(v[h - 1]) for h in curve["horizons"]]

    rh_open = reliable_horizon(curve_open["horizons"], _pointwise(curve_open), thr, mode)
    rh_closed = reliable_horizon(curve_closed["horizons"], _pointwise(curve_closed), thr, mode)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")          # 这里是有意复现历史口径，不必刷警告
        rh_open_cum = reliable_horizon(curve_open["horizons"], curve_open["nmse"], thr, mode,
                                       calibration="cumulative")
        rh_closed_cum = reliable_horizon(curve_closed["horizons"], curve_closed["nmse"], thr, mode,
                                         calibration="cumulative")
    print(f"[02] ★ reliable horizon（NMSE>{thr}，逐点口径）: 开环={rh_open}  闭环={rh_closed}")
    print(f"[02]   （累积口径，历史记录用）:                开环={rh_open_cum}  闭环={rh_closed_cum}")

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
    #   ★ 主线画**逐点**口径（H* 的定义口径）；累积口径画成淡色虚线 —— 
    #     两条线的间距就是这个口径选择的影响力，直接可视化了，不需要另开一张图解释。
    ax = axes[0, 1]
    ax.plot(curve_open["horizons"], curve_open["nmse"], ":", color="#4C78A8", alpha=0.35,
            label="open · cumulative (legacy)")
    ax.plot(curve_closed["horizons"], curve_closed["nmse"], ":", color="#E45756", alpha=0.35,
            label="closed · cumulative (legacy)")
    ax.plot(curve_open["horizons"], _pointwise(curve_open), "o-", color="#4C78A8",
            label="open-loop · per-step ★")
    ax.plot(curve_closed["horizons"], _pointwise(curve_closed), "s--", color="#E45756",
            label="closed-loop · per-step ★")
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
    ax.legend(fontsize=7); ax.grid(alpha=0.25)

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
    #   ★ 口径修正（2026-09-20）：这里原来画的是 `np.diff(累积NMSE) / np.diff(horizon)`，
    #     那是**累积曲线的斜率**，不是"第 h 步的误差"。既然逐点口径已经可用，
    #     直接画逐点 NMSE —— 它本身就是"每一步有多准"，不需要再差分（对应硬约定 R2）。
    ax = axes[1, 1]
    ax.plot(curve_open["horizons"], _pointwise(curve_open), "o-", color="#72B7B2",
            label="open-loop · per-step")
    ax.plot(curve_closed["horizons"], _pointwise(curve_closed), "s--", color="#B279A2",
            label="closed-loop · per-step")
    ax.set_xlabel("Rollout horizon (steps)")
    ax.set_ylabel("NMSE at step h (per-step)")
    ax.set_title("④ Per-step error: is it self-amplifying?")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

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
        # ★ H* 以**逐点**口径为准（"首次超阈"的定义）；累积口径单列，仅用于复现历史结论。
        "nmse_calibration": "per_step (pointwise)  —— H* 的定义口径",
        "reliable_horizon_open": rh_open,
        "reliable_horizon_closed": rh_closed,
        "reliable_horizon_open_cumulative": rh_open_cum,
        "reliable_horizon_closed_cumulative": rh_closed_cum,
        "threshold": thr,
        "threshold_mode": mode,
        "eval_set": {"n_val_episodes": len(val_eps),
                     "len_min": int(min(e["length"] for e in val_eps)),
                     "len_max": int(max(e["length"] for e in val_eps)),
                     "note": "评测集 = 训练数据的 10% 切分（短 episode）。"
                             "换评测集会改变 H*，跨结论比较前必须核对这个字段。"},
    }
    js = out / f"{args.tag}.json"
    js.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[02] 图已保存 -> {png}")
    print(f"[02] 指标已保存 -> {js}")
    print("[02] OK")


if __name__ == "__main__":
    main()
