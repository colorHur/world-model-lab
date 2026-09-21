#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""脚本 09 —— X5：手写 PPO 在 CartPole-v1 上训练。

对应 **X5 / E1**。验收口径不是"跑通"，而是两条：

1. **回报达标**：CartPole-v1 的**确定性评估**回报达到 ≥ `target_return`（默认 450，上限 500）
2. **能逐项解释每个 loss 项** —— 解释写在 `wmlab/agents/ppo.py` 的文件头推导里（② 类证据）

★ 为什么必须先有 PPO
--------------------
世界模型的可信度（H\*）此前全部用**随机策略**采的数据测出来的。
而随机策略在 CartPole 上平均只能撑 ~24 步 —— 也就是说，
**我们从来没测过"模型在真实控制策略产生的数据上能推多远"**。
X6 会用这个 PPO 策略采长 episode，X7 把"随机 vs PPO"当作消融自变量。

运行
----
    python scripts/09_train_ppo.py                    # 完整训练（约 10 分钟，CPU）
    python scripts/09_train_ppo.py --quick            # 冒烟：2 万步，看趋势用
    python scripts/09_train_ppo.py --total-steps 300000

★ 硬约定 R7：任何"改个配置再跑一遍"之前，先极小规模冒烟一次。
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

from wmlab.agents import ActorCritic, collect_rollout, evaluate_policy, ppo_update
from wmlab.envs import make_env
from wmlab.utils import (count_params, describe_device, get_device, load_config,
                         output_dir, set_seed)
from wmlab.utils.plot import apply_style, save_fig


def parse_args():
    p = argparse.ArgumentParser(description="手写 PPO 训练 CartPole")
    p.add_argument("--config", default=None)
    p.add_argument("--env", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--total-steps", type=int, default=None)
    p.add_argument("--hidden", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--rollout-steps", type=int, default=None)
    p.add_argument("--minibatch-size", type=int, default=None)
    p.add_argument("--quick", action="store_true", help="冒烟：total_steps 压到 2 万")
    p.add_argument("--tag", default="09_ppo_cartpole")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    ppo = dict(cfg.get("ppo", {}))

    # ---- 超参优先级：命令行 > yaml 的 ppo 段 > 内置默认 ----
    seed = int(args.seed if args.seed is not None else cfg["seed"])
    env_id = args.env or cfg["env"]["id"]
    max_steps = int(cfg["env"].get("max_steps", 500))

    def pick(name, default, cli=None, cast=float):
        if cli is not None:
            return cast(cli)
        return cast(ppo.get(name, default))

    total_steps = int(pick("total_steps", 150_000, args.total_steps, int))
    if args.quick:
        total_steps = min(total_steps, 20_000)
    rollout_steps = int(pick("rollout_steps", 128, args.rollout_steps, int))
    update_epochs = int(pick("update_epochs", 4, None, int))
    minibatch_size = int(pick("minibatch_size", 32, args.minibatch_size, int))
    lr = pick("lr", 3e-4, args.lr)
    gamma = pick("gamma", 0.99)
    lam = pick("gae_lambda", 0.95)
    clip_eps = pick("clip_eps", 0.2)
    vf_coef = pick("vf_coef", 0.5)
    ent_coef = pick("ent_coef", 0.01)
    max_grad_norm = pick("max_grad_norm", 0.5)
    hidden = int(pick("hidden", 64, args.hidden, int))
    eval_every = int(pick("eval_every", 10, None, int))
    n_eval = int(pick("n_eval_episodes", 5, None, int))
    target_return = float(pick("target_return", 450.0))

    set_seed(seed)
    apply_style()
    device = get_device(args.device or cfg.get("device"))
    out = output_dir(cfg)

    env = make_env(env_id, seed=seed)
    net = ActorCritic(env.obs_dim, env.act_dim, hidden=hidden).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, eps=1e-5)

    n_updates = max(1, total_steps // rollout_steps)
    print(f"[09] X5 · 手写 PPO  env={env_id}  seed={seed}  device={describe_device(device)}")
    print(f"[09] 参数量 = {count_params(net):,}   total_steps={total_steps}  "
          f"rollout={rollout_steps}  epochs={update_epochs}  mb={minibatch_size}  lr={lr}")
    print(f"[09] gamma={gamma}  lambda={lam}  clip={clip_eps}  "
          f"vf_coef={vf_coef}  ent_coef={ent_coef}")

    hist = {"update": [], "train_ep_return": [], "eval_return": [], "eval_len": [],
            "policy_loss": [], "value_loss": [], "entropy": [],
            "approx_kl": [], "clip_frac": [], "lr": []}
    all_train_returns: list[float] = []

    t0 = time.time()
    for u in range(1, n_updates + 1):
        # 学习率线性衰减（PPO 的标准做法；不衰减后期会在最优点附近反复震荡）
        frac = 1.0 - (u - 1) / n_updates
        for g in opt.param_groups:
            g["lr"] = lr * frac

        r = collect_rollout(env, net, rollout_steps, gamma, lam, device)
        st = ppo_update(net, opt, r, device=device, update_epochs=update_epochs,
                        minibatch_size=minibatch_size, clip_eps=clip_eps,
                        vf_coef=vf_coef, ent_coef=ent_coef,
                        max_grad_norm=max_grad_norm)

        all_train_returns.extend(r.ep_returns)
        hist["update"].append(u)
        hist["policy_loss"].append(st["policy_loss"])
        hist["value_loss"].append(st["value_loss"])
        hist["entropy"].append(st["entropy"])
        hist["approx_kl"].append(st["approx_kl"])
        hist["clip_frac"].append(st["clip_frac"])
        hist["lr"].append(lr * frac)
        hist["train_ep_return"].append(float(np.mean(r.ep_returns)) if r.ep_returns else np.nan)

        if u % eval_every == 0 or u == 1:
            ev = evaluate_policy(env, net, n_eval, device, max_steps=max_steps, seed=10_000 + u)
            hist["eval_return"].append(float(ev["return_mean"]))
            hist["eval_len"].append(float(ev["len_mean"]))
            print(f"[09] update {u:4d}/{n_updates}  "
                  f"steps={u*rollout_steps:6d}  eval_return={ev['return_mean']:7.1f} "
                  f"(len {ev['len_mean']:6.1f})  "
                  f"pi={st['policy_loss']:+.4f} vf={st['value_loss']:.4f} "
                  f"H={st['entropy']:.3f} kl={st['approx_kl']:.4f} "
                  f"clip={st['clip_frac']:.3f}  [{time.time()-t0:.0f}s]")
        else:
            hist["eval_return"].append(np.nan)
            hist["eval_len"].append(np.nan)

    env.close()
    elapsed = time.time() - t0

    # ---- 最终评估（更多 episode，作为引用值）----
    env2 = make_env(env_id, seed=seed + 999)
    final = evaluate_policy(env2, net, 20, device, max_steps=max_steps, seed=20_000)
    env2.close()

    evals = [x for x in hist["eval_return"] if not np.isnan(x)]
    last5 = evals[-5:] if len(evals) >= 5 else evals
    last5_mean = float(np.mean(last5)) if last5 else float("nan")
    passed = bool(last5_mean >= target_return)

    print(f"[09] 训练用时 {elapsed:.1f}s")
    print(f"[09] ★ 最终确定性评估（20 条 episode）: return = "
          f"{final['return_mean']:.1f} ± {final['return_std']:.1f} "
          f"(min {final['return_min']:.0f} / max {final['return_max']:.0f})")
    print(f"[09] ★ 判据（最近 5 次评估均值 ≥ {target_return:.0f}）: "
          f"{last5_mean:.1f}  ->  {'PASS' if passed else 'FAIL'}")

    # ---- 出图（标签全 ASCII：matplotlib 在本机缺中文字体，中文会变方框）----
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0))

    ax = axes[0, 0]
    xs = [u * rollout_steps for u in hist["update"]]
    ax.plot(xs, hist["train_ep_return"], ".", color="#4C78A8", alpha=0.35,
            ms=3, label="train ep return (per update)")
    ev_x = [xs[i] for i in range(len(xs)) if not np.isnan(hist["eval_return"][i])]
    ev_y = [hist["eval_return"][i] for i in range(len(xs))
            if not np.isnan(hist["eval_return"][i])]
    ax.plot(ev_x, ev_y, "o-", color="#E45756", ms=4, lw=1.6, label=f"eval return ({n_eval} eps)")
    ax.axhline(target_return, color="#54A24B", ls="--", lw=1.2, label=f"target {target_return:.0f}")
    ax.axhline(500.0, color="#888", ls=":", lw=1.0, label="CartPole-v1 cap = 500")
    ax.set_xlabel("Environment steps"); ax.set_ylabel("Return")
    ax.set_title("(1) PPO return curve (hand-written)")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

    ax = axes[0, 1]
    ax.plot(xs, hist["policy_loss"], color="#4C78A8", label="policy loss (clipped surrogate)")
    ax.plot(xs, hist["value_loss"], color="#E45756", label="value loss (0.5*MSE)")
    ax.set_xlabel("Environment steps"); ax.set_ylabel("Loss")
    ax.set_title("(2) Loss terms")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

    ax = axes[1, 0]
    ax.plot(xs, hist["entropy"], color="#54A24B", label="policy entropy H")
    ax.set_xlabel("Environment steps"); ax.set_ylabel("H[pi] (nats)")
    ax.set_title("(3) Entropy: must NOT collapse to 0 too early")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

    ax = axes[1, 1]
    ax.plot(xs, hist["approx_kl"], color="#4C78A8", label="approx KL(old||new)")
    ax.plot(xs, hist["clip_frac"], color="#F58518", label="clip fraction")
    ax.axhline(clip_eps, color="#888", ls=":", lw=1.0)
    ax.set_xlabel("Environment steps"); ax.set_ylabel("value")
    ax.set_title("(4) PPO diagnostics: is the clip active?")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

    fig.suptitle(f"wmlab X5 · hand-written PPO on {env_id} · seed={seed} · "
                 f"steps={total_steps} · final={final['return_mean']:.1f}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    png = save_fig(fig, out / f"{args.tag}.png")
    plt.close(fig)

    # ---- 存 checkpoint（X6 要用它采长 episode）----
    ckpt = out / f"{args.tag}.pt"
    torch.save({"state_dict": net.state_dict(), "obs_dim": env.obs_dim,
                "act_dim": env.act_dim, "hidden": hidden, "seed": seed,
                "env_id": env_id}, ckpt)

    summary = {
        "experiment": "X5 (hand-written PPO)",
        "env": env_id, "seed": seed, "device": str(device.type),
        "n_params": count_params(net),
        "hyperparams": {"total_steps": total_steps, "rollout_steps": rollout_steps,
                        "update_epochs": update_epochs, "minibatch_size": minibatch_size,
                        "lr": lr, "gamma": gamma, "gae_lambda": lam, "clip_eps": clip_eps,
                        "vf_coef": vf_coef, "ent_coef": ent_coef,
                        "max_grad_norm": max_grad_norm, "hidden": hidden},
        "final_eval": final,
        "last5_eval_mean": last5_mean,
        "target_return": target_return,
        "criterion_passed": passed,
        "train_seconds": elapsed,
        "history": hist,
    }
    js = out / f"{args.tag}.json"
    js.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[09] 图 -> {png}")
    print(f"[09] 指标 -> {js}")
    print(f"[09] 权重 -> {ckpt}")
    print("[09] OK")


if __name__ == "__main__":
    main()
