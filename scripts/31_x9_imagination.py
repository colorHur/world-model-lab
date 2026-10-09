#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""脚本 31 —— **X9：想象训练**（在世界模型的潜空间里 rollout 并反传策略梯度）。

对应 **X9 / E2**（E2 验收条目 ③「想象训练后的策略回报曲线」的唯一载体）。

本脚本要回答的一件事
--------------------
    在一个**只在随机数据上训过**的世界模型里做想象训练，
    策略在**真实环境**里到底能不能变强？以及**强多少**？

★ 为什么主判据是「真环境」回报而不是「想象」回报
------------------------------------------------
策略在模型里优化 ⇒ 它**必然**去利用模型的误差（model exploitation）。
"想象回报上升"可能是**失败**的症状。
⇒ 本脚本同时记录两条曲线，并算出 gap = 想象 − 真实（判据 H3）。

★ 三档想象视界 H ∈ {8, 16, 32}（判据 H4）
-----------------------------------------
H\* ≈ 43 步（本轮实测，见输出 `x9_horizon`）。
H = 8 < H\*/3 ⇒ 若连它都不上升，则「视界太长」这一归因被排除。

★ 预算对齐（唯一的自变量是 H）
------------------------------
每轮想象样本数固定 `--samples-per-update`（默认 512），
⇒ 起点数 B = 512 / H 与 H 一一对应（H=8→64, 16→32, 32→16）。
真实 PPO 对照臂的 `rollout_steps` **同样取 512**，轮数相同，minibatch/epochs 相同。
⇒ 三条想象臂与真实臂在"每轮看到多少样本、更新多少轮"上完全一致；
   **唯一变化的是每条想象轨迹的长度**（这正是 H4 要隔离的）。

★ 运行
------
    python scripts/31_x9_imagination.py --quick          # 冒烟（几分钟内）
    python scripts/31_x9_imagination.py                  # 正式
    python scripts/31_x9_imagination.py --horizons 8,16,32 --imag-updates 120
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

from wmlab.agents import (ContinuousActorCritic, collect_rollout_continuous,
                          continuous_ppo_update, evaluate_policy_continuous,
                          imagine_batch)
from wmlab.data import (collect_random_episodes, split_episodes,
                        transitions_from_episodes)
from wmlab.envs import make_env
from wmlab.eval import reliable_horizon
from wmlab.models import MLPWorldModel
from wmlab.rollout import multi_step_error_curve
from wmlab.train import train_world_model
from wmlab.utils import (count_params, describe_device, get_device, load_config,
                         output_dir, set_seed)
from wmlab.utils.plot import PALETTE, apply_style, save_fig

# PPO 超参：三条想象臂与真实臂**完全共用**（否则对比无意义）
CLIP_EPS = 0.2
VF_COEF = 0.5
ENT_COEF = 0.01
MAX_GRAD_NORM = 0.5
GAMMA = 0.99
LAM = 0.95
MINIBATCH = 64
EPOCHS_PER_UPDATE = 4
LR = 3e-4
HIDDEN = 64


def parse_args():
    p = argparse.ArgumentParser(description="X9 想象训练（Pendulum）")
    p.add_argument("--config", default="configs/pendulum.yaml")
    p.add_argument("--tag", default="31_x9_imagination")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    # 世界模型
    p.add_argument("--n-episodes", type=int, default=None, help="世界模型训练数据（默认取 config）")
    p.add_argument("--world-epochs", type=int, default=None)
    # 想象训练
    p.add_argument("--horizons", default="8,16,32")
    p.add_argument("--imag-updates", type=int, default=120)
    p.add_argument("--samples-per-update", type=int, default=512)
    # 真实 PPO 对照
    p.add_argument("--real-updates", type=int, default=None, help="默认与 imag-updates 相同")
    p.add_argument("--no-real-arm", action="store_true", help="跳过真实 PPO 对照臂")
    # 评估
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--n-eval", type=int, default=20)
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


# --------------------------------------------------------------------------- 小工具
def encode_fn(model, device):
    """obs(np) -> z(np)。**这是 X9 的关键接线**：想象臂的策略吃潜状态。"""
    @torch.no_grad()
    def fn(o):
        t = torch.as_tensor(np.asarray(o, dtype=np.float32), device=device).reshape(1, -1)
        return model.encode(t).cpu().numpy().reshape(-1)
    return fn


def identity_fn(o):
    """obs(np) -> obs(np)。真实 PPO 对照臂的策略吃原始观测。"""
    return np.asarray(o, dtype=np.float32).reshape(-1)


def random_policy_return(env, n_episodes, device, seed, max_steps):
    """随机策略基线（Pendulum 上这是唯一的"什么都不学"参照）。"""
    rng = np.random.default_rng(seed)
    rets = []
    for i in range(n_episodes):
        o = env.reset(seed=seed + i)
        R, done, t = 0.0, False, 0
        while not done:
            a = env.sample_action(rng)
            res = env.step(a)
            R += float(res.reward)
            t += 1
            if res.done or (max_steps is not None and t >= max_steps):
                done = True
            else:
                o = res.obs
        rets.append(R)
    return {"return_mean": float(np.mean(rets)), "return_std": float(np.std(rets)),
            "returns": rets}


# --------------------------------------------------------------------------- 训练臂
def run_imagination_arm(model, env, obs_pool, horizon, args, device, seed):
    """一条想象臂：从零开始训一个策略，主判据是真环境回报。"""
    max_steps = int(args.config_max_steps)
    n_starts = max(2, int(args.samples_per_update) // int(horizon))
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # ★ 策略输入是**潜状态** ⇒ 维度 = latent_dim
    actor = ContinuousActorCritic(model.latent_dim, env.act_dim, hidden=HIDDEN,
                                 act_scale=float(np.max(env.act_high))).to(device)
    opt = torch.optim.Adam(actor.parameters(), lr=LR, eps=1e-5)
    reward_fn = env.analytic_reward
    ef = encode_fn(model, device)

    # ★★ 量纲对齐（冒烟发现的真问题）：`imag_return` 是 **H 步**的累计，
    #    `real_return` 是 **200 步**的累计 —— 两者**不同量纲**。
    #    直接在累计值上做差，gap 会被"步数比 200/H"放大（H=8 时约 25 倍），
    #    报出来就是假的。⇒ gap 一律在**每步平均奖励**上算。
    hist = {"update": [], "imag_return": [], "imag_per_step": [],
            "real_return": [], "real_per_step": [], "gap": [],
            "entropy": [], "std": [], "z_start_std": [], "z_end_std": [],
            "policy_loss": [], "value_loss": [], "real_env_steps": 0}
    t0 = time.time()
    for u in range(1, int(args.imag_updates) + 1):
        frac = 1.0 - (u - 1) / max(1, int(args.imag_updates))
        for g in opt.param_groups:
            g["lr"] = LR * frac

        batch, info = imagine_batch(model, actor, obs_pool, n_starts, int(horizon),
                                    reward_fn, device, rng, GAMMA, LAM)
        st = continuous_ppo_update(actor, opt, batch, device=device,
                                   update_epochs=EPOCHS_PER_UPDATE,
                                   minibatch_size=MINIBATCH, clip_eps=CLIP_EPS,
                                   vf_coef=VF_COEF, ent_coef=ENT_COEF,
                                   max_grad_norm=MAX_GRAD_NORM)

        imag_per_step = float(info["imag_return"]) / float(horizon)
        hist["update"].append(u)
        hist["imag_return"].append(info["imag_return"])
        hist["imag_per_step"].append(imag_per_step)
        hist["entropy"].append(st["entropy"])
        hist["std"].append(st["std"])
        hist["policy_loss"].append(st["policy_loss"])
        hist["value_loss"].append(st["value_loss"])
        hist["z_start_std"].append(info["z_start_std"])
        hist["z_end_std"].append(info["z_end_std"])

        if u % int(args.eval_every) == 0 or u == 1:
            ev = evaluate_policy_continuous(env, actor, ef, int(args.n_eval), device,
                                            max_steps=max_steps, seed=20_000 + u)
            real_per_step = float(ev["return_mean"]) / max(float(ev["len_mean"]), 1.0)
            hist["real_return"].append(float(ev["return_mean"]))
            hist["real_per_step"].append(real_per_step)
            hist["gap"].append(imag_per_step - real_per_step)
            print(f"[31]   H={horizon:3d} update {u:4d}/{args.imag_updates}  "
                  f"imag/step={imag_per_step:8.3f}  real/step={real_per_step:8.3f}  "
                  f"gap/step={imag_per_step-real_per_step:8.3f}  "
                  f"real_R={ev['return_mean']:8.1f}  "
                  f"std={st['std']:.3f} H={st['entropy']:.3f}  [{time.time()-t0:.0f}s]")
        else:
            hist["real_return"].append(np.nan)
            hist["real_per_step"].append(np.nan)
            hist["gap"].append(np.nan)

    return actor, hist, ef


def run_real_arm(env, args, device, seed, n_updates):
    """真实环境 PPO 对照臂（同样的每轮样本数与轮数）。"""
    max_steps = int(args.config_max_steps)
    torch.manual_seed(seed)
    actor = ContinuousActorCritic(env.obs_dim, env.act_dim, hidden=HIDDEN,
                                 act_scale=float(np.max(env.act_high))).to(device)
    opt = torch.optim.Adam(actor.parameters(), lr=LR, eps=1e-5)
    ifn = identity_fn

    hist = {"update": [], "real_return": [], "real_per_step": [], "entropy": [], "std": [],
            "policy_loss": [], "value_loss": [], "real_env_steps": 0}
    t0 = time.time()
    for u in range(1, n_updates + 1):
        frac = 1.0 - (u - 1) / max(1, n_updates)
        for g in opt.param_groups:
            g["lr"] = LR * frac

        r = collect_rollout_continuous(env, actor, int(args.samples_per_update),
                                       GAMMA, LAM, device, x_fn=ifn)
        st = continuous_ppo_update(actor, opt, r, device=device,
                                   update_epochs=EPOCHS_PER_UPDATE,
                                   minibatch_size=MINIBATCH, clip_eps=CLIP_EPS,
                                   vf_coef=VF_COEF, ent_coef=ENT_COEF,
                                   max_grad_norm=MAX_GRAD_NORM)
        hist["real_env_steps"] += int(args.samples_per_update)
        hist["update"].append(u)
        hist["entropy"].append(st["entropy"])
        hist["std"].append(st["std"])
        hist["policy_loss"].append(st["policy_loss"])
        hist["value_loss"].append(st["value_loss"])

        if u % int(args.eval_every) == 0 or u == 1:
            ev = evaluate_policy_continuous(env, actor, ifn, int(args.n_eval), device,
                                            max_steps=max_steps, seed=20_000 + u)
            hist["real_return"].append(float(ev["return_mean"]))
            hist["real_per_step"].append(float(ev["return_mean"]) / max(float(ev["len_mean"]), 1.0))
            print(f"[31]   REAL  update {u:4d}/{n_updates}  "
                  f"real/step={hist['real_per_step'][-1]:8.3f}  "
                  f"real_R={ev['return_mean']:8.1f}  "
                  f"env_steps={hist['real_env_steps']:7d}  [{time.time()-t0:.0f}s]")
        else:
            hist["real_return"].append(np.nan)
            hist["real_per_step"].append(np.nan)
    return actor, hist, ifn


# --------------------------------------------------------------------------- 主流程
def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.quick:
        cfg["data"]["n_episodes"] = min(int(cfg["data"]["n_episodes"]), 40)
        args.world_epochs = min(int(args.world_epochs or 999), 8)
        args.imag_updates = min(int(args.imag_updates), 12)
        args.samples_per_update = min(int(args.samples_per_update), 64)
        args.eval_every = min(int(args.eval_every), 4)
        args.n_eval = min(int(args.n_eval), 3)
        args.horizons = ",".join([h for h in args.horizons.split(",")][:2])
    if args.world_epochs is not None:
        cfg["train"]["epochs"] = int(args.world_epochs)
    if args.n_episodes is not None:
        cfg["data"]["n_episodes"] = int(args.n_episodes)

    seed = int(args.seed if args.seed is not None else cfg["seed"])
    set_seed(seed)
    apply_style()
    device = get_device(args.device or cfg.get("device"))
    out = output_dir(cfg)
    args.config_max_steps = int(cfg["env"].get("max_steps", 200))
    horizons = [int(h) for h in str(args.horizons).split(",") if h.strip()]

    env_id = cfg["env"]["id"]
    print(f"[31] X9 想象训练  env={env_id}  seed={seed}  device={describe_device(device)}")
    print(f"[31] horizons={horizons}  imag_updates={args.imag_updates}  "
          f"samples/update={args.samples_per_update}  minibatch={MINIBATCH} "
          f"epochs={EPOCHS_PER_UPDATE}  n_eval={args.n_eval}")

    # ---------- 1) 随机数据 ----------
    env = make_env(env_id, seed=seed)
    if not getattr(env, "has_analytic_reward", False):
        raise RuntimeError(f"{env_id} 不支持解析奖励重建 ⇒ 想象训练不适用（见 X9 记录 §0.2）")
    episodes = collect_random_episodes(env, n_episodes=int(cfg["data"]["n_episodes"]),
                                       seed=seed, max_steps=args.config_max_steps)
    train_eps, val_eps = split_episodes(episodes, float(cfg["train"]["val_ratio"]), seed)
    all_obs = np.concatenate([ep["obs"] for ep in train_eps], axis=0)
    print(f"[31] 随机数据 {len(episodes)} 条 / {sum(e['length'] for e in episodes)} 步；"
          f"起点池 {all_obs.shape[0]} 帧")

    # ---------- 2) 世界模型 ----------
    mcfg = cfg["model"]
    model = MLPWorldModel(obs_dim=env.obs_dim, act_dim=env.act_dim,
                          latent_dim=int(mcfg["latent_dim"]), hidden=int(mcfg["hidden"]),
                          discrete_act=bool(mcfg["discrete_act"]) and env.is_discrete).to(device)
    tr = tuple(torch.as_tensor(x) for x in transitions_from_episodes(train_eps))
    va = tuple(torch.as_tensor(x) for x in transitions_from_episodes(val_eps))
    t0 = time.time()
    wm_hist = train_world_model(model, tr, va, cfg, device)
    print(f"[31] 世界模型训练完成 {time.time()-t0:.1f}s  参数量={count_params(model):,}  "
          f"final val={wm_hist['val_total'][-1]:.6f}")

    # ---------- 3) H* 实测（给 H4 提供 H*/3 的锚点） ----------
    hs = list(cfg["eval"]["horizons"])
    curve = multi_step_error_curve(model, val_eps, hs, device, seed=seed)
    per_step = [float(np.asarray(curve["nmse_per_step"])[h - 1]) for h in curve["horizons"]]
    thr = float(cfg["eval"]["err_threshold"])
    h_star = reliable_horizon(curve["horizons"], per_step, thr,
                             cfg["eval"].get("threshold_mode", "rel"))
    print(f"[31] ★ 本轮实测 H*（逐点口径，阈值 {thr}，val 集） = {h_star}")

    # ---------- 4) 随机基线 ----------
    base = random_policy_return(env, int(args.n_eval), device, 30_000, args.config_max_steps)
    print(f"[31] 随机策略基线 return = {base['return_mean']:.1f} ± {base['return_std']:.1f}")

    # ---------- 5) 三条想象臂 ----------
    arms = {}
    for H in horizons:
        print(f"[31] === 想象臂 H={H} ===")
        actor, hist, _ = run_imagination_arm(model, env, all_obs, H, args, device,
                                             seed=1000 + H)
        arms[f"imag_H{H}"] = {"horizon": H, "hist": hist,
                              "final_real": hist["real_return"][-1],
                              "final_imag": hist["imag_return"][-1]}
        print(f"[31] H={H} 完成：末次 真环境={hist['real_return'][-1]:.1f}  "
              f"想象={hist['imag_return'][-1]:.1f}  真环境交互步数=0（训练期）")

    # ---------- 6) 真实 PPO 对照臂 ----------
    real = None
    if not args.no_real_arm:
        n_upd = int(args.real_updates or args.imag_updates)
        print(f"[31] === 真实 PPO 对照臂（rollout={args.samples_per_update} × {n_upd} 轮）===")
        _, rh, _ = run_real_arm(env, args, device, seed=2000, n_updates=n_upd)
        real = {"hist": rh, "final_real": rh["real_return"][-1],
                "real_env_steps": rh["real_env_steps"]}
        print(f"[31] 真实臂完成：末次 真环境={rh['real_return'][-1]:.1f}  "
              f"真环境交互步数={rh['real_env_steps']}")
    env.close()

    # ---------- 7) 出图（★ 标签全 ASCII；标题从 args 派生 —— 铁律 16） ----------
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0))
    cols = [PALETTE["blue"], PALETTE["red"], PALETTE["green"]]

    def _xy(h, key):
        xs = [h["update"][i] for i in range(len(h["update"])) if not np.isnan(h[key][i])]
        ys = [h[key][i] for i in range(len(h["update"])) if not np.isnan(h[key][i])]
        return xs, ys

    ax = axes[0, 0]
    for (name, a), c in zip(arms.items(), cols):
        xs, ys = _xy(a["hist"], "real_return")
        ax.plot(xs, ys, "o-", color=c, ms=3.5, lw=1.5, label=f"imagination H={a['horizon']}")
    if real is not None:
        xs, ys = _xy(real["hist"], "real_return")
        ax.plot(xs, ys, "s--", color=PALETTE["purple"], ms=3.5, lw=1.5,
                label="real-env PPO (same budget)")
    ax.axhline(base["return_mean"], color=PALETTE["grey"], ls=":", lw=1.4,
               label=f"random policy ({base['return_mean']:.0f})")
    ax.set_xlabel("PPO update"); ax.set_ylabel("TRUE-environment return")
    ax.set_title("(1) *MAIN* true-env return: does imagination help?")
    ax.legend(fontsize=7.5); ax.grid(alpha=0.25)

    ax = axes[0, 1]
    for (name, a), c in zip(arms.items(), cols):
        ax.plot(a["hist"]["update"], a["hist"]["imag_per_step"], "-", color=c, lw=1.5,
                label=f"imagined H={a['horizon']} (model)")
        xs, ys = _xy(a["hist"], "real_per_step")
        ax.plot(xs, ys, "o:", color=c, ms=3.5, lw=1.3, alpha=0.85,
                label=f"real (same policy) H={a['horizon']}")
    if real is not None:
        xs, ys = _xy(real["hist"], "real_per_step")
        ax.plot(xs, ys, "s--", color=PALETTE["purple"], ms=3.5, lw=1.5, label="real PPO")
    ax.set_xlabel("PPO update")
    ax.set_ylabel("reward per step  (imagined vs real, SAME units)")
    ax.set_title("(2) imagined vs real reward/step")
    ax.legend(fontsize=6.5); ax.grid(alpha=0.25)

    ax = axes[1, 0]
    for (name, a), c in zip(arms.items(), cols):
        xs, ys = _xy(a["hist"], "gap")
        ax.plot(xs, ys, "o-", color=c, ms=3, lw=1.4, label=f"H={a['horizon']}")
    ax.axhline(0.0, color=PALETTE["grey"], ls="--", lw=1.2)
    ax.set_xlabel("PPO update"); ax.set_ylabel("imagined - real  (reward/step)")
    ax.set_title("(3) model-exploitation gap per step (H3)")
    ax.legend(fontsize=7.5); ax.grid(alpha=0.25)

    ax = axes[1, 1]
    for (name, a), c in zip(arms.items(), cols):
        ax.plot(a["hist"]["update"], a["hist"]["std"], "-", color=c, lw=1.4,
                label=f"sigma H={a['horizon']}")
    if real is not None:
        ax.plot(real["hist"]["update"], real["hist"]["std"], "--",
                color=PALETTE["purple"], lw=1.4, label="sigma real PPO")
    ax.set_xlabel("PPO update"); ax.set_ylabel("policy sigma")
    ax.set_title("(4) action std (exploration); entropy is Gaussian-approx")
    ax.legend(fontsize=7.5); ax.grid(alpha=0.25)

    fig.suptitle(f"wmlab X9 · imagination training on {env_id} · seed={seed} · "
                 f"H*={h_star} · imag_updates={args.imag_updates} · "
                 f"samples/update={args.samples_per_update}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    png = save_fig(fig, out / f"{args.tag}.png")
    plt.close(fig)

    # ---------- 8) 汇总 json ----------
    def _trend(vals):
        """★ 冒烟发现的坑：原来的 `first5` / `last5` 在评估点 < 10 时**是同一个数组**
        ⇒ `delta` 恒为 0 —— 看着像"FAIL"，其实是**没有分辨力**（自证伪第 28 族的形态）。
        ⇒ 改为「首末差 + 线性趋势」，并要求 **n ≥ 3** 才判；不足则明确标 `evaluable=False`。
        """
        v = np.asarray([x for x in vals if not np.isnan(x)], dtype=float)
        n = v.size
        if n < 3:
            return {"n": n, "evaluable": False, "first": None, "last": None,
                    "delta": None, "slope": None, "pearson_r": None,
                    "series": [float(z) for z in v]}
        x = np.arange(n, dtype=float)
        return {"n": n, "evaluable": True,
                "first": float(v[0]), "last": float(v[-1]),
                "delta": float(v[-1] - v[0]),
                "slope_per_eval": float(np.polyfit(x, v, 1)[0]),
                "pearson_r": float(np.corrcoef(x, v)[0, 1]),
                "series": [float(z) for z in v]}

    def arm_summary(a):
        h = a["hist"]
        return {
            "horizon": a["horizon"],
            "n_starts_per_update": max(2, int(args.samples_per_update) // a["horizon"]),
            "samples_per_update": int(args.samples_per_update),
            "tot_imag_samples": int(args.samples_per_update) * int(args.imag_updates),
            "real_env_steps_during_training": 0,
            "final_imag_return": float(h["imag_return"][-1]),
            "imag_return_per_step_last": float(h["imag_per_step"][-1]),
            "real_return_trend": _trend(h["real_return"]),
            "real_per_step_trend": _trend(h["real_per_step"]),
            "gap_trend": _trend(h["gap"]),
            "entropy_last": float(h["entropy"][-1]), "std_last": float(h["std"][-1]),
            "z_start_std_last": float(h["z_start_std"][-1]),
            "z_end_std_last": float(h["z_end_std"][-1]),
        }

    summary = {
        "experiment": "X9 (imagination training)",
        "env": env_id, "seed": seed, "device": str(device.type),
        "criteria_legend": {
            "H1": "true-env return rises: last5 > first5",
            "H2": "compare vs real-env PPO under the SAME per-update budget",
            "H3": "gap (imagined - real) trend; |r| < 0.5 => indeterminate",
            "H4": "if H = 8 (< H*/3) still fails to rise, 'horizon too long' is ruled out",
        },
        "world_model": {
            "n_params": count_params(model),
            "final_train_loss": wm_hist["train_total"][-1],
            "final_val_loss": wm_hist["val_total"][-1],
            "latent_dim": int(mcfg["latent_dim"]),
            "n_train_episodes": len(train_eps), "n_val_episodes": len(val_eps),
        },
        "horizon_measurement": {
            "H_star": h_star, "threshold": thr,
            "calibration": "per_step (pointwise) - the H* definition",
            "eval_set": "train/val split of the random dataset (same as script 02)",
            "note": "H* 随评测集与归一化分母而变（R1）；本值只与同一份 val 集可比。",
        },
        "random_baseline": {"return_mean": base["return_mean"],
                            "return_std": base["return_std"]},
        "arms": {k: arm_summary(v) for k, v in arms.items()},
        "real_arm": (None if real is None else {
            "real_env_steps_total": real["real_env_steps"],
            "samples_per_update": int(args.samples_per_update),
            "n_updates": int(args.real_updates or args.imag_updates),
            "real_return_trend": _trend(real["hist"]["real_return"]),
            "real_per_step_trend": _trend(real["hist"]["real_per_step"]),
            "entropy_last": float(real["hist"]["entropy"][-1]),
            "std_last": float(real["hist"]["std"][-1]),
        }),
        "budget_note": ("想象臂训练期真环境交互 = 0（仅初始随机数据集 "
                        f"{sum(e['length'] for e in episodes)} 步，三条臂共用）；"
                        "真实臂的真环境交互 = samples_per_update × n_updates。"
                        "两者每轮样本数、轮数、minibatch、epoch 完全一致。"),
        "hyperparams": {"lr": LR, "gamma": GAMMA, "gae_lambda": LAM,
                        "clip_eps": CLIP_EPS, "vf_coef": VF_COEF, "ent_coef": ENT_COEF,
                        "minibatch": MINIBATCH, "epochs_per_update": EPOCHS_PER_UPDATE,
                        "hidden": HIDDEN},
    }
    js = out / f"{args.tag}.json"
    js.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------- 9) 判据判定（逐条打印，便于对账） ----------
    print("[31] ================= 判据对账 =================")
    for k, a in summary["arms"].items():
        t, p = a["real_return_trend"], a["real_per_step_trend"]
        if not t["evaluable"]:
            print(f"[31] H1 {k}: 评估点仅 {t['n']} 个（<3）=> **不可评估**（不许判）")
            continue
        ok = (t["delta"] > 0) and (t["slope_per_eval"] > 0)
        print(f"[31] H1 {k}: reward/step {p['first']:+.3f} -> {p['last']:+.3f}  "
              f"(Δ {p['delta']:+.3f}, slope {p['slope_per_eval']:+.4f}/eval, r {p['pearson_r']:+.3f})"
              f"  -> {'PASS' if ok else 'FAIL'}")
    if summary["real_arm"] is not None:
        best = max((a["real_per_step_trend"]["last"] or -1e9)
                   for a in summary["arms"].values())
        rl = summary["real_arm"]["real_per_step_trend"]["last"]
        print(f"[31] H2 best-imagination {best:.3f} vs real-PPO {rl:.3f} (reward/step)"
              f"  -> ratio {best / rl if rl else float('nan'):.3f}")
    print("[31] H3 per-step gap trend (Pearson r; |r| < 0.5 => indeterminate):")
    for k, a in summary["arms"].items():
        g = a["gap_trend"]
        rr = g["pearson_r"] if g["pearson_r"] is not None else float("nan")
        print(f"[31]    {k}: r = {rr:+.3f}  (first {g['first']}, last {g['last']})")
    print(f"[31] H4 anchor: H* = {h_star}  => H*/3 = "
          f"{h_star/3 if h_star else float('nan'):.1f}；本臂最小 H = {min(horizons)}")
    print(f"[31] 图 -> {png}")
    print(f"[31] 指标 -> {js}")
    print("[31] OK")


if __name__ == "__main__":
    main()
