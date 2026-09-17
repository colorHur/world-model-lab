#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""脚本 01 —— 随机策略 baseline，产出本仓库的第一张结果图。

用途
----
1. 验证环境接口（envs）通路：能 reset、能 step、能采动作
2. 给出随机策略的回报水平 —— 后面世界模型/策略改进都要以它为参照
3. 产出 1 张结果图（E3 验收标准之一）

运行
----
    python scripts/01_random_baseline.py
    python scripts/01_random_baseline.py --episodes 500 --seed 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 允许直接 `python scripts/xx.py` 运行（无需 pip install -e .）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")           # 无界面后端，服务器/远程终端也能出图
import matplotlib.pyplot as plt
import numpy as np

from wmlab.data import collect_random_episodes, episode_stats
from wmlab.envs import make_env
from wmlab.utils import describe_device, get_device, load_config, output_dir, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="随机策略 baseline")
    p.add_argument("--config", default=None, help="YAML 配置路径（默认 configs/default.yaml）")
    p.add_argument("--episodes", type=int, default=None, help="覆盖 episode 数")
    p.add_argument("--seed", type=int, default=None, help="覆盖随机种子")
    p.add_argument("--device", default=None, help="cpu / cuda / auto")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.episodes is not None:
        cfg["data"]["n_episodes"] = args.episodes
    if args.seed is not None:
        cfg["seed"] = args.seed

    seed = int(cfg["seed"])
    set_seed(seed)
    device = get_device(args.device or cfg.get("device"))
    out = output_dir(cfg)

    print(f"[01] env={cfg['env']['id']}  episodes={cfg['data']['n_episodes']}  seed={seed}")
    print(f"[01] device = {describe_device(device)}")

    env = make_env(cfg["env"]["id"], seed=seed)
    print(f"[01] obs_dim={env.obs_dim}  act_dim={env.act_dim}  discrete={env.is_discrete}")

    episodes = collect_random_episodes(
        env, n_episodes=int(cfg["data"]["n_episodes"]), seed=seed,
        max_steps=cfg["env"].get("max_steps"),
    )
    env.close()

    stats = episode_stats(episodes)
    print("[01] stats: " + json.dumps(stats, ensure_ascii=False, indent=2))

    # ---------- 出图 ----------
    returns = np.asarray([float(ep["reward"].sum()) for ep in episodes])
    lengths = np.asarray([ep["length"] for ep in episodes])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].hist(returns, bins=30, color="#4C78A8", edgecolor="white")
    axes[0].axvline(returns.mean(), color="#E45756", linestyle="--", linewidth=1.5,
                    label=f"mean = {returns.mean():.1f}")
    axes[0].set_xlabel("Episode return")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Random policy return distribution ({cfg['env']['id']})")
    axes[0].legend()

    win = max(5, len(lengths) // 20)
    kernel = np.ones(win) / win
    smooth = np.convolve(lengths, kernel, mode="valid")
    axes[1].plot(lengths, color="#BAB0AC", linewidth=0.8, alpha=0.7, label="per episode")
    axes[1].plot(np.arange(win - 1, len(lengths)), smooth, color="#4C78A8", linewidth=2,
                 label=f"moving avg (w={win})")
    axes[1].set_xlabel("Episode index")
    axes[1].set_ylabel("Episode length (steps)")
    axes[1].set_title("Episode length over time")
    axes[1].legend()

    fig.suptitle(
        f"wmlab · random-policy baseline · seed={seed} · n={len(returns)} "
        f"· device={device.type}",
        fontsize=10,
    )
    fig.tight_layout()
    png = out / "01_random_baseline.png"
    fig.savefig(png, dpi=150)
    plt.close(fig)

    js = out / "01_random_baseline.json"
    js.write_text(json.dumps({"config": cfg, "stats": stats, "device": str(device)},
                             ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[01] 图已保存 -> {png}")
    print(f"[01] 指标已保存 -> {js}")
    print("[01] OK")


if __name__ == "__main__":
    main()
