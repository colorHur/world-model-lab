#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""脚本 10 —— X6 + X7：**行为策略消融**（D1 的第一个实证，DoD 里「≥1 条消融」那格）。

问题
----
世界模型的可靠视界 H\*，**是否依赖于产生训练数据的行为策略？**

这个问题不是自找的：此前**所有** H\* 都是在**随机策略**数据上测的。
随机策略在 CartPole 上平均只撑 ~22 步 ⇒ 我们从来没测过
「模型在真实控制策略产生的数据分布上能推多远」。X5 训练出了能撑满 500 步的 PPO 策略，
这个消融才第一次可做。

★ 三个必须封死的混淆变量（不封死，结论不可解释）
-------------------------------------------------
| # | 混淆变量 | 封法 |
|---|---|---|
| **C1** | **训练样本数** | 随机每集 ~22 步、PPO 每集 500 步。若按 episode 数对齐，样本数差 20 倍 ⇒ 测到的是"数据更多"而不是"分布不同"。**⇒ 按转移样本数对齐（`align_transitions`）** |
| **C2** | **评测集** | X3/X4 已证明 H\* 依赖评测集。若训练分布与评测集同时换，两个效应混在一起。**⇒ 做 2×2：训练分布 {随机, PPO} × 评测集 {随机, PPO}**，从而把"训练分布效应"与"评测集效应"分开 |
| **C3** | **种子方差** | X3 实测 H\* 的 CV = 0.42（5 种子）。**单次对比的差值完全落在噪声里**。**⇒ 3 个种子，且做配对比较**（同 seed 相减，消掉种子间的水平差异）|

主口径：**在 PPO 评测集上**（长 episode，H\* 不被数据上限截断），比较 random-trained 与 PPO-trained。

运行
----
    python scripts/10_behavior_ablation.py                 # 3 seeds × 2 模型
    python scripts/10_behavior_ablation.py --seeds 0       # 单种子快速看
    python scripts/10_behavior_ablation.py --quick         # 冒烟：1 seed + 20 epoch
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

from wmlab.agents import ActorCritic
from wmlab.data import (align_transitions, collect_policy_episodes,
                        collect_random_episodes, episode_stats,
                        transitions_from_episodes)
from wmlab.envs import make_env
from wmlab.eval import reliable_horizon
from wmlab.models import MLPWorldModel
from wmlab.rollout import closed_loop_error_curve, multi_step_error_curve
from wmlab.train import train_world_model
from wmlab.utils import (describe_device, get_device, load_config, output_dir,
                         set_seed)
from wmlab.utils.plot import apply_style, save_fig

# 预训练好的 PPO 权重（X5 产出；不存在时脚本会明确报错，不会静默退化成随机策略）
DEFAULT_CKPT = "outputs/09d_ppo_relu_500k.pt"


def parse_args():
    p = argparse.ArgumentParser(description="X6+X7 行为策略消融")
    p.add_argument("--config", default=None)
    p.add_argument("--ckpt", default=DEFAULT_CKPT, help="X5 产出的 PPO 权重")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--quick", action="store_true", help="冒烟：1 个 seed + 20 epoch")
    p.add_argument("--tag", default="10_behavior_ablation")
    return p.parse_args()


def pointwise(curve):
    """逐点 NMSE（H\* 的定义口径，见 scripts/02 的说明）。"""
    v = np.asarray(curve["nmse_per_step"], dtype=float)
    return [float(v[h - 1]) for h in curve["horizons"]]


def load_policy(path: str, device: torch.device):
    p = Path(path)
    if not p.is_absolute():
        p = Path(__file__).resolve().parents[1] / path
    if not p.exists():
        raise FileNotFoundError(
            f"找不到 PPO 权重 {p} —— 请先跑 scripts/09_train_ppo.py。"
            f"这里故意不做'退化为随机策略'的兜底：那会让消融静默失效。")
    ck = torch.load(p, map_location=device, weights_only=False)
    net = ActorCritic(ck["obs_dim"], ck["act_dim"], hidden=ck["hidden"]).to(device)
    net.load_state_dict(ck["state_dict"])
    net.eval()
    return net, ck


def main():
    args = parse_args()
    cfg = load_config(args.config)
    seeds = [int(s) for s in args.seeds.split(",")]
    if args.quick:
        seeds = seeds[:1]
    epochs = int(args.epochs or (20 if args.quick else cfg["train"]["epochs"]))
    cfg["train"]["epochs"] = epochs      # train_world_model 的 epoch 数来自 cfg

    set_seed(seeds[0])
    apply_style()
    device = get_device(args.device or cfg.get("device"))
    out = output_dir(cfg)
    horizons = list(cfg["eval"]["horizons"])
    # ★ 长评测集专用网格：PPO 评测集的 episode 有 ~300 步，若仍用 15 步的网格，
    #   PPO-trained 那格会一直"未超阈"⇒ 报不出 H*，也就**报不出差值**（X7 判据要求"两条 H* 及差值"）。
    #   注意：这只用于 PPO 评测集；随机评测集的 episode 只有 ~23 步，仍用 cfg 的原网格。
    #   ⇒ 代价：两格的视界网格不同，**跨评测集的 H* 数值不可直接相减**（本来也不可比，见 C2）。
    horizons_long = horizons + [20, 25, 30, 40, 50, 60]
    thr = float(cfg["eval"]["err_threshold"])
    mode = cfg["eval"].get("threshold_mode", "rel")

    net, ck = load_policy(args.ckpt, device)
    print(f"[10] X6+X7 · 行为策略消融  env={cfg['env']['id']}  "
          f"device={describe_device(device)}  seeds={seeds}  epochs={epochs}")
    print(f"[10] PPO 权重: {args.ckpt}")

    def policy_fn(obs):
        return net.act(obs, device, deterministic=False)[0]

    results = []      # 每个 (seed, train_dist, eval_dist) 一条
    data_stats = {}
    curves_seed0 = {}

    t0 = time.time()
    for seed in seeds:
        env = make_env(cfg["env"]["id"], seed=seed)

        # ---------- 采数据 ----------
        rnd_all = collect_random_episodes(env, n_episodes=400, seed=seed,
                                          max_steps=cfg["env"].get("max_steps"))
        ppo_all = collect_policy_episodes(env, policy_fn, n_episodes=40, seed=seed,
                                          max_steps=cfg["env"].get("max_steps"))
        # ★ C1：按转移样本数对齐（不是按 episode 数）
        n_rnd = int(np.sum([e["length"] for e in rnd_all]))
        n_ppo = int(np.sum([e["length"] for e in ppo_all]))
        target = min(n_rnd, n_ppo)
        train_rnd = align_transitions(rnd_all, target)
        train_ppo = align_transitions(ppo_all, target)

        data_stats[seed] = {
            "n_train_transitions": target,
            "random": {"n_episodes_raw": len(rnd_all), "steps_raw": n_rnd,
                       "n_episodes_used": len(train_rnd),
                       **{k: v for k, v in episode_stats(train_rnd).items()}},
            "ppo": {"n_episodes_raw": len(ppo_all), "steps_raw": n_ppo,
                    "n_episodes_used": len(train_ppo),
                    **{k: v for k, v in episode_stats(train_ppo).items()}},
        }
        print(f"[10] seed={seed}  对齐后的训练样本数 = {target} 条转移  "
              f"(随机 {len(train_rnd)} 集 × ~{target/max(1,len(train_rnd)):.0f} 步 / "
              f"PPO {len(train_ppo)} 集 × ~{target/max(1,len(train_ppo)):.0f} 步)")

        # ---------- 评测集（★ C2：两个评测集都用）----------
        eval_rnd = collect_random_episodes(env, n_episodes=60, seed=10_000 + seed,
                                           max_steps=cfg["env"].get("max_steps"))
        eval_ppo = collect_policy_episodes(env, policy_fn, n_episodes=20, seed=20_000 + seed,
                                           max_steps=cfg["env"].get("max_steps"))

        # ★ NMSE 的分母 = 切片目标方差。两个评测集的方差差一个量级以上 ⇒
        #   **跨评测集的 NMSE/H* 数值不可直接比较**（这正是 C2 要做 2×2 的原因）。
        var_rnd = float(np.var(np.concatenate([e["obs"] for e in eval_rnd], 0)))
        var_ppo = float(np.var(np.concatenate([e["obs"] for e in eval_ppo], 0)))
        data_stats[seed]["eval_obs_var"] = {"random": var_rnd, "ppo": var_ppo}
        print(f"[10]   评测集目标方差（= NMSE 的分母尺度）: random={var_rnd:.4g}  "
              f"PPO={var_ppo:.4g}  ⇒ 比值 {var_rnd/max(var_ppo,1e-12):.1f}x，"
              f"跨评测集的 NMSE 不可直接比较")

        # ---------- 训两个世界模型（只差训练数据分布）----------
        mcfg = cfg["model"]
        models = {}
        for name, eps in (("random", train_rnd), ("ppo", train_ppo)):
            set_seed(seed)
            m = MLPWorldModel(obs_dim=env.obs_dim, act_dim=env.act_dim,
                              latent_dim=int(mcfg["latent_dim"]), hidden=int(mcfg["hidden"]),
                              discrete_act=bool(mcfg["discrete_act"]) and env.is_discrete
                              ).to(device)
            tr = tuple(torch.as_tensor(x) for x in transitions_from_episodes(eps))
            va = tuple(torch.as_tensor(x) for x in
                       transitions_from_episodes(eval_rnd[:6]))
            train_world_model(m, tr, va, cfg, device, verbose=False)
            models[name] = m

        # ---------- 2×2 测 H* ----------
        for tname, m in models.items():
            for ename, ev in (("random", eval_rnd), ("ppo", eval_ppo)):
                hs_use = horizons_long if ename == "ppo" else horizons
                c_open = multi_step_error_curve(m, ev, hs_use, device, seed=seed)
                c_closed = closed_loop_error_curve(m, ev, hs_use, device, seed=seed)
                v_open = pointwise(c_open)
                ho = reliable_horizon(c_open["horizons"], v_open, thr, mode)
                hc = reliable_horizon(c_closed["horizons"], pointwise(c_closed), thr, mode)
                if ho is None:
                    vals = pointwise(c_open)
                    print(f"[10]     [debug] H*=None: 开环逐点 NMSE = "
                          f"{[round(float(v), 4) for v in vals]}"
                          f"  (含 nan={int(np.isnan(vals).sum())}, "
                          f"inf={int(np.isinf(vals).sum())})")
                results.append({"seed": seed, "train": tname, "eval": ename,
                                "h_open": ho, "h_closed": hc,
                                "nmse_h1": float(v_open[0]),
                                "nmse_max": float(np.nanmax(v_open)),
                                "horizon_grid_max": int(max(c_open["horizons"]))})
                print(f"[10]   train={tname:6s} eval={ename:6s} -> "
                      f"H*(open)={ho if ho is not None else float('nan'):6.2f}  "
                      f"H*(closed)={hc if hc is not None else float('nan'):6.2f}  "
                      f"[NMSE h1={v_open[0]:.4g} max={np.nanmax(v_open):.4g}]")
                if seed == seeds[0]:
                    curves_seed0[(tname, ename)] = {
                        "open": pointwise(c_open), "closed": pointwise(c_closed),
                        "horizons": list(c_open["horizons"])}

        env.close()

    # ---------- 汇总 ----------
    def agg(train, eval_, key):
        xs = [r[key] for r in results if r["train"] == train and r["eval"] == eval_]
        xs = [float(x) for x in xs if x is not None]
        if not xs:
            return {"mean": None, "std": None, "n": 0, "values": []}
        return {"mean": float(np.mean(xs)), "std": float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0,
                "n": len(xs), "values": xs}

    table = {}
    for t in ("random", "ppo"):
        for e in ("random", "ppo"):
            table[f"train_{t}__eval_{e}"] = {
                "h_open": agg(t, e, "h_open"), "h_closed": agg(t, e, "h_closed")}

    print()
    print("[10] ==== 2x2 汇总（开环 H*，均值 ± 标准差，n = 种子数）====")
    print(f"[10] {'train':>8s} | {'eval=random':>18s} | {'eval=PPO':>18s}")
    for t in ("random", "ppo"):
        a = table[f"train_{t}__eval_random"]["h_open"]
        b = table[f"train_{t}__eval_ppo"]["h_open"]
        fa = f"{a['mean']:.2f} ± {a['std']:.2f}" if a["n"] else "n/a"
        fb = f"{b['mean']:.2f} ± {b['std']:.2f}" if b["n"] else "n/a"
        print(f"[10] {t:>8s} | {fa:>18s} | {fb:>18s}")

    # ★ 主口径：PPO 评测集上的配对差（同 seed 相减，消掉种子间的水平差异）
    pairs = []
    for seed in seeds:
        a = next((r["h_open"] for r in results
                  if r["seed"] == seed and r["train"] == "random" and r["eval"] == "ppo"), None)
        b = next((r["h_open"] for r in results
                  if r["seed"] == seed and r["train"] == "ppo" and r["eval"] == "ppo"), None)
        if a is not None and b is not None:
            pairs.append({"seed": seed, "random_trained": float(a),
                          "ppo_trained": float(b), "diff": float(b - a)})
    if pairs:
        diffs = np.array([p["diff"] for p in pairs])
        print()
        print("[10] ★ 主口径（PPO 评测集，配对比较）: "
              f"PPO-trained − random-trained = {diffs.mean():+.2f} ± {diffs.std(ddof=1) if len(diffs)>1 else 0.0:.2f}"
              f"  (n={len(diffs)}, 逐 seed: {np.round(diffs, 2).tolist()})")
        # ★ 与种子间标准差对照：差值若远小于种子噪声，就不能宣称有效应
        pooled = np.array([p["random_trained"] for p in pairs] +
                          [p["ppo_trained"] for p in pairs])
        print(f"[10]   对照：同一格内的种子间标准差 = {pooled.std(ddof=1):.2f} "
              f"⇒ |差值| {'>' if abs(diffs.mean()) > pooled.std(ddof=1) else '<'} 种子噪声"
              f" ⇒ {'效应可能可辨' if abs(diffs.mean()) > pooled.std(ddof=1) else '效应落在噪声内，不可宣称'}")

    # ---------- 出图 ----------
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0))
    labels = {"random": "random policy", "ppo": "PPO policy"}

    ax = axes[0, 0]
    xs = np.arange(4)
    means, stds, names, na_flags, caps = [], [], [], [], []
    for t in ("random", "ppo"):
        for e in ("random", "ppo"):
            a = table[f"train_{t}__eval_{e}"]["h_open"]
            names.append(f"{t[:3]}->{e[:3]}")
            gmax = float(max(horizons_long)) if e == "ppo" else float(max(horizons))
            if a["n"]:
                means.append(a["mean"]); stds.append(a["std"] or 0.0); na_flags.append(False)
            else:
                # ★ n/a 的含义是"视界内从未超阈"，画成 0 会误导 ⇒ 画到网格上限并加阴影标注
                means.append(gmax); stds.append(0.0); na_flags.append(True)
            caps.append(gmax)
    colors = ["#4C78A8", "#72B7B2", "#E45756", "#F58518"]
    bars = ax.bar(xs, means, yerr=stds, capsize=4, color=colors)
    for b, flag in zip(bars, na_flags):
        if flag:
            b.set_hatch("///"); b.set_alpha(0.45)
    ax.set_xticks(xs); ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("H* (open loop, steps)")
    ax.set_title("(1) 2x2: train-dist -> eval-set\n(hatched = never exceeded thr within horizon)")
    for b, m, flag, cap in zip(bars, means, na_flags, caps):
        ax.text(b.get_x() + b.get_width() / 2, m,
                (f">{cap:.0f}" if flag else f"{m:.1f}"),
                ha="center", va="bottom", fontsize=8)
    ax.grid(alpha=0.25, axis="y")

    ax = axes[0, 1]
    if pairs:
        ax.plot([p["random_trained"] for p in pairs], [p["ppo_trained"] for p in pairs],
                "o", color="#4C78A8", ms=8)
        lo = min(min(p["random_trained"] for p in pairs), min(p["ppo_trained"] for p in pairs))
        hi = max(max(p["random_trained"] for p in pairs), max(p["ppo_trained"] for p in pairs))
        pad = max(1.0, (hi - lo) * 0.15)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1, label="y = x")
        for p in pairs:
            ax.annotate(f"seed{p['seed']}", (p["random_trained"], p["ppo_trained"]),
                        textcoords="offset points", xytext=(5, 4), fontsize=8)
        ax.set_xlabel("H* : trained on RANDOM data")
        ax.set_ylabel("H* : trained on PPO data")
        ax.set_title("(2) Paired comparison (eval = PPO set)")
        ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    for (t, e), c in curves_seed0.items():
        ax.plot(c["horizons"], c["open"], "o-" if t == "random" else "s--",
                ms=4, lw=1.4, label=f"train {t[:3]} / eval {e[:3]}")
    ax.axhline(thr, color="#888", ls=":", lw=1.2, label=f"threshold {thr}")
    ax.set_yscale("log")
    ax.set_xlabel("Rollout horizon (steps)"); ax.set_ylabel("NMSE (per-step)")
    ax.set_title(f"(3) Open-loop curves (seed={seeds[0]})")
    ax.legend(fontsize=7); ax.grid(alpha=0.25)

    ax = axes[1, 1]
    s0 = seeds[0]
    for name, st in (("random", data_stats[s0]["random"]), ("ppo", data_stats[s0]["ppo"])):
        ax.bar([name], [st["length_mean"]], yerr=[st["length_std"]], capsize=4,
               color="#4C78A8" if name == "random" else "#E45756")
        ax.text(name, st["length_mean"], f"{st['length_mean']:.0f}", ha="center",
                va="bottom", fontsize=9)
    ax.set_ylabel("Episode length (steps)")
    ax.set_title(f"(4) Data distribution: episode length (seed={s0})\n"
                 "sample count is EQUALIZED (not this!)")
    ax.grid(alpha=0.25, axis="y")

    fig.suptitle(f"wmlab X6+X7 · behavior-policy ablation · env={cfg['env']['id']} · "
                 f"seeds={seeds} · equal train samples", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    png = save_fig(fig, out / f"{args.tag}.png")
    plt.close(fig)

    summary = {
        "experiment": "X6+X7 behavior-policy ablation",
        "env": cfg["env"]["id"], "seeds": seeds, "epochs": epochs,
        "horizons": horizons, "threshold": thr, "threshold_mode": mode,
        "controls": {"C1_equal_train_samples": True, "C2_two_eval_sets": True,
                     "C3_multiple_seeds_paired": True},
        "data_stats": data_stats,
        "table": table,
        "paired_eval_ppo": pairs,
        "raw": results,
        "ppo_ckpt": str(args.ckpt),
        "seconds": time.time() - t0,
    }
    js = out / f"{args.tag}.json"
    js.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[10] 用时 {time.time()-t0:.1f}s")
    print(f"[10] 图 -> {png}")
    print(f"[10] 指标 -> {js}")
    print("[10] OK")


if __name__ == "__main__":
    main()
