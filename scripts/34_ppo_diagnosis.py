#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""脚本 34 —— **限时 PPO 诊断**：用成熟实现（SB3）作**诊断参照**，定位"真实臂学不动"的归属。

存在的理由（一句话）
--------------------
`scripts/32`（参照学习器审计）在 10-12 收口后重跑，结论仍是 **VOID**：真实 PPO 的
「末 k 次评测均值」跨 3 个种子为 **−6.449 ± 0.983**，门槛 **−5.159**，**0/3 过线**。
问题是：这到底是**我们这套自实现在某处有缺口**，还是 **PPO 在本任务 / 本预算下本来就学不动**？

本轮按华腾给定的交付顺序第 3 项：**限时诊断 PPO**（≤2 个工作时段），
引入**成熟实现**作诊断参照，**保留相同任务、评测种子与环境步预算**。

★ 本脚本的定位（必须随结果一起报）
----------------------------------
    对照**只用于定位**（"差异在实现还是在算法/任务"），
    **不用于宣称算法优劣** —— 单任务、单预算、3 个种子，不足以支撑任何
    "实现 A 比实现 B 强"的结论。R14：不把它写成名次表。

对齐口径（三条，缺一条比较就不成立）
------------------------------------
1. **同一任务**：`gymnasium` 的 `Pendulum-v1`（与 `wmlab.envs.GymClassicAdapter` 同一个
   `gym.make` 环境；每集 200 步上限）。
2. **同一评测初值**：`scripts/31:fixed_eval_seeds(n)`（**单一真源**）—— `500000+0..n-1`；
   随机基线 / 我们的臂 / SB3 臂**共用同一批** ⇒ 差值才是配对差值。
3. **同一环境步预算**：我们的臂 = `800 轮 × 512 步 = 409,600` 环境步（`scripts/32` 默认档），
   SB3 用 `total_timesteps = 409,600`。
   ★ 评测遵循**同一步数间隔**（每 12,800 环境步一次 = 我们那边 25 轮 × 512），
   这样"末 k 次评测"在两个实现里覆盖**同样多的环境步**。

我们不重跑自己的臂（R7：能机器数的绝不手记）
--------------------------------------------
我们的臂的**机器产物**就是 `scripts/32` 的 JSON（同代码、同配置、同机、同 CPU）。
本脚本默认**读它**（`--our-arm-json`），不重新训练 —— 这样本轮的算力全部花在
"新信息"上（SB3 臂）。若该 JSON 缺失，脚本**直接报错**并给出复跑命令，不静默降级。

运行
----
    # 冒烟（每种配置 1 个种子、极小预算）
    python scripts/34_ppo_diagnosis.py --smoke
    # 正式
    python scripts/34_ppo_diagnosis.py
    # 只跑对齐配置
    python scripts/34_ppo_diagnosis.py --sb3-configs aligned
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

from wmlab.envs import make_env
from wmlab.eval import reference_learner_verdict
from wmlab.utils import load_config, output_dir, set_seed
from wmlab.utils.plot import PALETTE, apply_style, save_fig

_HERE = Path(__file__).resolve()
REPO_ROOT = _HERE.parents[1]

# 评测间隔：与我们那边对齐（25 轮 × 512 步）
EVAL_EVERY_STEPS = 25 * 512

# SB3 的两套配置：`aligned` = 尽量对齐我们那边；`default` = SB3 出厂默认。
SB3_CONFIGS = {
    "aligned": dict(n_steps=512, batch_size=64, n_epochs=4,
                    learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
                    clip_range=0.2, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5),
    "default": dict(n_steps=2048, batch_size=64, n_epochs=10,
                    learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
                    clip_range=0.2, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5),
    # ★ 单旋钮消融：从 `aligned`（= 我方超参）**各改一个**旋钮，用来把差异定位到具体超参。
    #   `aligned` 与 `default` 之间有 3 处不同（n_steps / n_epochs / ent_coef），
    #   一次全改说不清是哪个在起作用 ⇒ 这三个配置各只动一处。
    "ab_nsteps": dict(n_steps=2048, batch_size=64, n_epochs=4,
                      learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
                      clip_range=0.2, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5),
    "ab_epochs": dict(n_steps=512, batch_size=64, n_epochs=10,
                      learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
                      clip_range=0.2, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5),
    "ab_ent": dict(n_steps=512, batch_size=64, n_epochs=4,
                   learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
                   clip_range=0.2, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5),
}

# 我们那边（scripts/31 + 32）的配置，仅用于**列差异表**
OURS = dict(n_steps=512, batch_size=64, n_epochs=4, learning_rate="3e-4 → 线性衰减到 0",
            gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01, vf_coef=0.5,
            max_grad_norm=0.5, hidden=64, activation="ReLU", log_std_init=-0.5,
            init="正交初始化", obs_norm="无")

# 配置差异表（逐项列出；华腾要求"配置差异逐项列出"）
CONFIG_DIFF = [
    ("rollout 步数 n_steps", "512", "512", "2048"),
    ("minibatch", "64", "64", "64"),
    ("epochs / update", "4", "4", "10"),
    ("学习率", "3e-4 → 线性衰减到 0", "3e-4 常数", "3e-4 常数"),
    ("ent_coef", "0.01", "0.01", "0.0（SB3 默认）"),
    ("vf_coef", "0.5", "0.5", "0.5"),
    ("max_grad_norm", "0.5", "0.5", "0.5"),
    ("gamma / gae_lambda", "0.99 / 0.95", "0.99 / 0.95", "0.99 / 0.95"),
    ("clip_range", "0.2", "0.2", "0.2"),
    ("策略网络隐藏层", "64（2 层）", "64（2 层）", "64（2 层）"),
    ("激活函数", "ReLU", "Tanh（SB3 默认）", "Tanh（SB3 默认）"),
    ("log_std 初值", "-0.5", "SB3 默认（0.0）", "SB3 默认（0.0）"),
    ("参数初始化", "正交（mean 头 gain 0.01）", "SB3 默认", "SB3 默认"),
    ("观测归一化", "无", "无（未用 VecNormalize）", "无（未用 VecNormalize）"),
    # ★ 2026-10-18 更正：SB3 默认 PPO('MlpPolicy') 的 use_sde=False、squash_output=False，
    #   动作分布是 DiagGaussian **直出**，再由 action_space.clip 截到 [-2,2]；
    #   我方是 2·tanh(u)（有界 squash）。两者**分布形状不同**（一个在边界堆积、一个两端梯度消失），
    #   原表写 "SB3 SquashedDiagGaussian / 同" 是错的 ⇒ 这一项属于**未对齐**，
    #   使 X9 的归因只能停在"超参足以解释 SB3 侧的失败"，不能扩成"我方失败纯属超参"。
    ("动作参数化", "2·tanh(u)（有界 squash）", "DiagGaussian 直出 + clip 到 [-2,2]", "同"),
]


def parse_args():
    p = argparse.ArgumentParser(description="限时 PPO 诊断（SB3 作诊断参照）")
    p.add_argument("--config", default="configs/pendulum.yaml")
    p.add_argument("--tag", default="34_ppo_diagnosis")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--n-eval", type=int, default=10)
    p.add_argument("--budget-steps", type=int, default=409_600,
                   help="与环境步预算对齐（800 轮 × 512）")
    p.add_argument("--eval-every-steps", type=int, default=EVAL_EVERY_STEPS)
    p.add_argument("--last-k", type=int, default=5)
    p.add_argument("--margin", type=float, default=1.5)
    p.add_argument("--our-arm-json", default="outputs/32_reference_learner_audit.json")
    p.add_argument("--sb3-configs", default="aligned,default")
    p.add_argument("--reuse-sb3-json", default="",
                   help="逗号分隔的既有 JSON 列表：其中**配置逐项相同 + 种子相同 + 预算相同**的 "
                        "SB3 臂**直接复用**（不重算）。用于把分批跑的配置合并成唯一权威产物。")
    p.add_argument("--smoke", action="store_true",
                   help="冒烟：每配置 1 种子、预算 12,800 步、n_eval 3")
    return p.parse_args()


# --------------------------------------------------------------------------- 评测
def eval_sb3(model, seeds, max_steps):
    """在**指定初值种子**上确定性评测 SB3 策略；返回逐回合 (return, length)。"""
    import gymnasium as gym
    rets, lens = [], []
    for s in seeds:
        env = gym.make("Pendulum-v1")
        obs, _ = env.reset(seed=int(s))
        R, t, done = 0.0, 0, False
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(a)
            R += float(r)
            t += 1
            done = bool(term or trunc) or (t >= max_steps)
        env.close()
        rets.append(float(R))
        lens.append(int(t))
    return np.asarray(rets, float), np.asarray(lens, float)


def per_step_of(rets, lens):
    """per-step 奖励序列（逐回合）。"""
    return np.asarray(rets, float) / np.maximum(np.asarray(lens, float), 1.0)


def build_reuse_cache(paths):
    """从既有 JSON 里收集**可复用**的 SB3 臂。

    ★ 复用条件（三条**同时**满足，任一不满足即不复用 —— R14：宁可重算，不可错配）：
      1. `config` 字典**逐项相同**（按 `sort_keys` 的 JSON 串比较）；
      2. 训练种子列表相同；
      3. 评测初值列表相同 **且** 环境步预算相同。
    """
    cache: dict[str, list[dict]] = {}
    for p in paths:
        pp = Path(p)
        if not pp.is_absolute():
            pp = REPO_ROOT / pp
        if not pp.exists():
            print(f"[34] （跳过不存在的复用源 {pp}）")
            continue
        d = json.loads(pp.read_text(encoding="utf-8"))
        ta = d.get("task_alignment") or {}
        for cname, arm in (d.get("sb3_arms") or {}).items():
            key = json.dumps(arm.get("config", {}), sort_keys=True)
            cache.setdefault(key, []).append({
                "source": pp.name, "name": cname, "arm": arm,
                "seeds": [int(z) for z in (d.get("seeds") or [])],
                "eval_seeds": ta.get("eval_seeds"),
                "budget": ta.get("budget_env_steps"),
            })
            print(f"[34] 复用源 {pp.name}：发现臂 {cname}"
                  f"（seeds={d.get('seeds')}，budget={ta.get('budget_env_steps')}）")
    return cache


def reuse_lookup(cache, cfg_kw, seeds, eval_seeds, budget, sd):
    """命中则返回该 seed 的既有结果，否则 `None`。"""
    key = json.dumps(dict(cfg_kw), sort_keys=True)
    for e in cache.get(key, []):
        if e["seeds"] != [int(z) for z in seeds]:
            continue
        if e["eval_seeds"] is not None and [int(z) for z in e["eval_seeds"]] != [int(z) for z in eval_seeds]:
            continue
        if e["budget"] is not None and int(e["budget"]) != int(budget):
            continue
        arm = e["arm"]
        if str(sd) not in arm.get("per_step_per_seed", {}):
            continue
        return {"source": e["source"], "name": e["name"],
                "per_step": arm["per_step_per_seed"][str(sd)],
                "seconds": float(arm.get("seconds_per_seed", {}).get(str(sd), float("nan"))),
                "eval_at_steps": arm.get("eval_at_steps"),
                "episode_returns": (arm.get("episode_returns_per_seed") or {}).get(str(sd))}
    return None


def run_sb3_arm(cfg_name, cfg_kw, seed, seeds_eval, args, max_steps):
    """跑一条 SB3 臂，返回 (hist, seconds)。"""
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback

    import gymnasium as gym

    set_seed(int(seed))                     # 与我们那边同语义：重置全部随机源
    env = gym.make("Pendulum-v1")
    model = PPO("MlpPolicy", env, seed=int(seed), verbose=0, device="cpu",
                **cfg_kw)
    store = {"per_step": [], "episode_returns": [], "episode_lengths": [],
             "eval_at_steps": [], "next_eval": int(args.eval_every_steps)}

    class EvalCB(BaseCallback):
        def _on_step(self) -> bool:
            while self.num_timesteps >= store["next_eval"]:
                rets, lens = eval_sb3(self.model, seeds_eval, max_steps)
                store["per_step"].append(float(per_step_of(rets, lens).mean()))
                store["episode_returns"].append([float(z) for z in rets])
                store["episode_lengths"].append([int(z) for z in lens])
                store["eval_at_steps"].append(int(self.num_timesteps))
                store["next_eval"] += int(args.eval_every_steps)
            return True

    t0 = time.time()
    model.learn(total_timesteps=int(args.budget_steps), callback=EvalCB(),
                progress_bar=False)
    secs = time.time() - t0
    env.close()
    return store, secs


# --------------------------------------------------------------------------- 主流程
def main():
    args = parse_args()
    cfg = load_config(args.config)
    apply_style()
    out = output_dir(cfg)
    max_steps = int(cfg["env"].get("max_steps", 200))
    env_id = cfg["env"]["id"]
    if not str(env_id).startswith("Pendulum"):
        raise RuntimeError(f"本诊断只针对 Pendulum（依赖解析动作空间/200 步），当前 {env_id}")

    seeds = [int(s) for s in str(args.seeds).split(",") if s.strip()]
    cfg_names = [c.strip() for c in str(args.sb3_configs).split(",") if c.strip()]
    for c in cfg_names:
        if c not in SB3_CONFIGS:
            raise ValueError(f"未知 SB3 配置 {c}；可选 {list(SB3_CONFIGS)}")

    if args.smoke:
        seeds = seeds[:1]
        args.budget_steps = min(int(args.budget_steps), 12_800)
        args.n_eval = min(int(args.n_eval), 3)
        args.eval_every_steps = min(int(args.eval_every_steps), 6_400)
        args.tag = args.tag + "_smoke"

    # ---------- 1) 我们自己的臂：读机器产物（R7，不重跑） ----------
    p_our = (REPO_ROOT / args.our_arm_json) if not Path(args.our_arm_json).is_absolute() \
        else Path(args.our_arm_json)
    if not p_our.exists():
        raise FileNotFoundError(
            f"找不到我们臂的机器产物 {p_our}。\n"
            f"请先跑：python scripts/32_reference_learner_audit.py "
            f"--seeds {args.seeds} --updates {args.budget_steps // 512} "
            f"--n-eval {args.n_eval} --last-k {args.last_k} --margin {args.margin}"
        )
    our = json.loads(p_our.read_text(encoding="utf-8"))
    # ★ 2026-10-18 修复：原守卫 `[int(z) for z in xs] != [int(z) for z in xs]` 左右是同一表达式，
    #   **条件恒假**，那行 raise 永远不会执行。真正起作用的是 `int(z)` 转换自身抛异常
    #   （所以"能不能转成 int"被隐式校验了，"是不是整数而不是 5.0"没被校验）。
    #   这里换成显式类型校验：bool 要单独排除（isinstance(True, int) 为真）。
    _raw_eval_seeds = our["eval_seeds"]
    if not isinstance(_raw_eval_seeds, (list, tuple)) or any(
            isinstance(z, bool) or not isinstance(z, (int, np.integer)) for z in _raw_eval_seeds):
        raise AssertionError(f"eval_seeds 应为整数列表，实际 {_raw_eval_seeds!r}")
    eval_seeds = [int(z) for z in _raw_eval_seeds]
    if len(eval_seeds) != int(args.n_eval):
        raise AssertionError(
            f"评测种子数不匹配：JSON={len(eval_seeds)} vs --n-eval={args.n_eval}"
            "（比较必须共用同一批初值，R14：报错优于给假数字）")
    if int(our["env_steps_per_seed"]) != int(args.budget_steps):
        raise AssertionError(
            f"环境步预算不匹配：JSON={our['env_steps_per_seed']} vs --budget-steps={args.budget_steps}")
    base_per_step = float(our["random_baseline_per_step"])
    bar = base_per_step + float(args.margin)
    our_seeds = [int(z) for z in our["seeds"]]
    if our_seeds != seeds:
        raise AssertionError(f"训练种子不匹配：JSON={our_seeds} vs --seeds={seeds}")

    print(f"[34] 限时 PPO 诊断（SB3 作诊断参照；**仅用于定位，不用于宣称算法优劣**）")
    print(f"[34] env={env_id}  max_steps={max_steps}  评测初值 {eval_seeds[0]}..{eval_seeds[-1]} (n={len(eval_seeds)})")
    print(f"[34] 预算 {args.budget_steps} 环境步 / 臂；评测每 {args.eval_every_steps} 步一次（与我方对齐）")
    print(f"[34] 随机基线 per-step {base_per_step:.3f}；门槛 bar = base + {args.margin} = {bar:.3f}")
    print(f"[34] ours（来自 {p_our.name}）：末{args.last_k}均值 "
          f"{np.round(our['last_k_mean_per_seed'], 3).tolist()}，过线 {our['n_clearing_bar']}/{len(our_seeds)}"
          f" ⇒ {our['verdict']}")

    # ---------- 2) SB3 各配置各跑一遍 ----------
    arms = {"ours": {"per_step_per_seed": {str(s): our["runs"][str(s)]["per_step"] for s in our_seeds},
                     "last_k_mean_per_seed": [float(z) for z in our["last_k_mean_per_seed"]],
                     "n_clearing_bar": int(our["n_clearing_bar"]),
                     "verdict": our["verdict"],
                     "last_k": int(our["last_k"]),
                     "origin": f"scripts/32 JSON ({p_our.name})"}}
    reuse_paths = [z.strip() for z in str(args.reuse_sb3_json).split(",") if z.strip()]
    cache = build_reuse_cache(reuse_paths) if reuse_paths else {}

    for cname in cfg_names:
        kw = dict(SB3_CONFIGS[cname])
        arms[cname] = {"per_step_per_seed": {}, "last_k_mean_per_seed": [],
                       "seconds_per_seed": {}, "eval_at_steps": None,
                       "episode_returns_per_seed": {}, "origin": {}}
        for sd in seeds:
            ru = reuse_lookup(cache, kw, seeds, eval_seeds, int(args.budget_steps), sd)
            if ru is not None:
                hist = {"per_step": ru["per_step"], "eval_at_steps": ru["eval_at_steps"],
                        "episode_returns": ru["episode_returns"]}
                secs = ru["seconds"]
                arms[cname]["origin"][str(sd)] = f"reused from {ru['source']}"
                print(f"[34] === SB3[{cname}] seed={sd} **复用** {ru['source']}"
                      f"（配置/种子/预算逐项相同）===")
            else:
                print(f"[34] === SB3[{cname}] seed={sd} 开始（{args.budget_steps} 步）===")
                hist, secs = run_sb3_arm(cname, kw, seed=2000 + 1000 * sd,
                                         seeds_eval=eval_seeds, args=args, max_steps=max_steps)
                arms[cname]["origin"][str(sd)] = "computed"
            ps = [float(z) for z in hist["per_step"]]
            k = int(min(int(args.last_k), len(ps)))
            if k < int(args.last_k):
                print(f"[34]  ⚠️ seed={sd} 只有 {len(ps)} 次评测（<last_k={args.last_k}）"
                      f" ⇒ 末 k 均值用现有 {k} 个（须在报告里说明）")
            lk = float(np.mean(ps[-k:])) if ps else float("nan")
            arms[cname]["per_step_per_seed"][str(sd)] = ps
            arms[cname]["last_k_mean_per_seed"].append(lk)
            arms[cname]["seconds_per_seed"][str(sd)] = float(secs)
            arms[cname]["eval_at_steps"] = [int(z) for z in hist["eval_at_steps"]]
            arms[cname]["episode_returns_per_seed"][str(sd)] = hist["episode_returns"]
            print(f"[34]  SB3[{cname}] seed={sd} 末{k}均值 {lk:+.3f} "
                  f"(Δ vs 基线 {lk - base_per_step:+.3f})  n_eval点={len(ps)}  [{secs:.0f}s]")
        lkv = np.asarray(arms[cname]["last_k_mean_per_seed"], dtype=float)
        valid, det = reference_learner_verdict(lkv, bar)
        arms[cname]["valid"] = bool(valid)
        arms[cname]["verdict"] = "VALID" if valid else "VOID"
        arms[cname]["n_clearing_bar"] = int(det["n_clearing_bar"])
        arms[cname]["last_k"] = int(args.last_k)

    # ---------- 3) 诊断结论（★ 只做归属，不做名次） ----------
    def _cleared(name):
        a = arms[name]
        lkv = np.asarray(a["last_k_mean_per_seed"], dtype=float)
        return int((lkv >= bar).sum()), int(lkv.size)

    sb3_names = list(cfg_names)
    valid_map = {c: bool(arms[c].get("valid")) for c in sb3_names}
    passing = [c for c in sb3_names if valid_map[c]]
    aligned_cfg = SB3_CONFIGS.get("aligned", {})
    # ★ 与「我方超参**逐项相同**」的配置（按字典相等判定，不靠名字）
    same_hyper = [c for c in sb3_names if SB3_CONFIGS[c] == aligned_cfg]
    same_hyper_ok = all(valid_map[c] for c in same_hyper) if same_hyper else None
    if not passing:
        attribution = (
            "**成熟实现在同任务 / 同评测初值 / 同环境步预算下也过不了线** ⇒ 该门槛在本设定下"
            "**不是「自实现独有」的问题** ⇒ **不能**把 X9 的 VOID 归因于我们的 PPO 实现缺陷；"
            "更可能是 **算法族 × 任务 × 预算**的组合问题（Pendulum 上 PPO 本就弱）。")
        action = "把本诊断作为 X9 VOID 的**解释**，暂停 X9（不再调 PPO 超参）"
    elif same_hyper and not same_hyper_ok:
        attribution = (
            "★★ **关键分层**：**同为成熟实现，「与我方超参逐项相同」的配置 "
            f"({same_hyper}) 与 ours 一样判 VOID**，而 **{passing} 过线** ⇒ "
            "归属是**超参配置**：**不是**「PPO 族在本任务本预算下不可达」，"
            "**也不是**「我们实现独有的缺陷」。⇒ X9 的 VOID **可以用超参解释**；"
            "修法是**只改超参**，不是重写实现。")
        action = ("按消融（`ab_nsteps` / `ab_epochs` / `ab_ent`）定位到**具体旋钮**后，"
                  "**只改这一处超参**重跑参照臂；**不重写实现**")
    else:
        attribution = (
            "**成熟实现在同任务 / 同评测初值 / 同环境步预算下**至少有一套配置过线 ⇒ "
            "该门槛**对 PPO 族可达** ⇒ 我们的失败**更可能**落在**实现 / 配置缺口**上"
            "（继续限时定位：网络 / 初始化 / log_std / lr 衰减 / GAE 掩码各查一遍）。"
            "★ 注意：这只说'可达'，**不是**'我们的实现一定错'。")
        action = "继续限时定位（≤2 工作时段），**未定位则暂停 X9**"

    print("[34] ================= 诊断结论（归属，非名次）================")
    for nm in ["ours"] + sb3_names:
        a = arms[nm]
        lkv = np.asarray(a["last_k_mean_per_seed"], dtype=float)
        print(f"[34] {nm:8s} 末{a['last_k']}均值 {np.round(lkv, 3).tolist()}  "
              f"过线 {int((lkv >= bar).sum())}/{lkv.size}  ⇒ {a.get('verdict','?')}")
    print(f"[34] 归属：{attribution}")
    print(f"[34] 建议动作：{action}")

    # ---------- 4) 出图 ----------
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.4))
    ax = axes[0]
    ours_mean = np.mean([np.asarray(arms["ours"]["per_step_per_seed"][str(s)], float)
                         for s in our_seeds], axis=0)
    ax.plot(np.arange(ours_mean.size), ours_mean, "-", lw=2.4, color=PALETTE["purple"],
            label="ours (script 31/32)")
    _pal = [PALETTE["blue"], PALETTE["red"], PALETTE["green"], PALETTE["purple"]]
    colors = {c: _pal[i % len(_pal)] for i, c in enumerate(cfg_names)}
    for cname in sb3_names:
        rows = [np.asarray(arms[cname]["per_step_per_seed"][str(s)], float) for s in seeds]
        m = np.mean(rows, axis=0)
        ax.plot(np.arange(m.size), m, "-", lw=2.2, color=colors.get(cname, PALETTE["green"]),
                label=f"SB3 PPO [{cname}]")
    ax.axhline(base_per_step, color=PALETTE["grey"], ls=":", lw=1.5,
               label=f"random policy ({base_per_step:.2f})")
    ax.axhline(bar, color=PALETTE["red"], ls="--", lw=1.3,
               label=f"VALID bar ({bar:.2f})")
    ax.set_xlabel("evaluation index (every 12,800 env steps)")
    ax.set_ylabel("true-env reward/step")
    ax.set_title("(1) learning curves, same task / eval seeds / budget")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

    ax = axes[1]
    names = ["ours"] + sb3_names
    xs = np.arange(len(names))
    bar_colors = [PALETTE["purple"]] + [colors.get(c, PALETTE["green"]) for c in sb3_names]
    for i, nm in enumerate(names):
        lkv = np.asarray(arms[nm]["last_k_mean_per_seed"], dtype=float)
        ax.bar(i, lkv.mean(), width=0.5, color=bar_colors[i], alpha=0.85)
        ax.scatter(np.full(lkv.size, i), lkv, color="black", zorder=3, s=22)
    ax.axhline(base_per_step, color=PALETTE["grey"], ls=":", lw=1.5, label="random")
    ax.axhline(bar, color=PALETTE["red"], ls="--", lw=1.3, label="VALID bar")
    ax.set_xticks(xs); ax.set_xticklabels(names)
    ax.set_ylabel(f"last-{args.last_k} mean reward/step")
    ax.set_title("(2) per-seed last-k mean vs VALID bar")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

    fig.suptitle(f"wmlab script 34 - limited PPO diagnosis (SB3 as diagnostic reference) - "
                 f"{env_id} - budget={args.budget_steps} env steps - seeds={seeds}",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    png = save_fig(fig, out / f"{args.tag}.png")
    plt.close(fig)

    # ---------- 5) 落 JSON ----------
    summary = {
        "experiment": "limited PPO diagnosis (script 34, SB3 as diagnostic reference)",
        "purpose": ("定位『真实臂学不动』的归属：自实现/配置缺口 vs PPO 在本任务本预算下本来就学不动。"
                    "★ 对照只用于定位，**不用于宣称算法优劣**（单任务/单预算/3 种子）。"),
        "env": env_id, "device": "cpu",
        "task_alignment": {"env": env_id, "max_steps": max_steps,
                           "eval_seeds": eval_seeds,
                           "eval_seed_source": "scripts/31:fixed_eval_seeds (single source of truth)",
                           "budget_env_steps": int(args.budget_steps),
                           "eval_every_steps": int(args.eval_every_steps)},
        "random_baseline_per_step": base_per_step,
        "margin": float(args.margin), "bar": bar,
        "seeds": seeds, "last_k": int(args.last_k),
        "our_arm": {"source": str(p_our), "last_k_mean_per_seed":
                    [float(z) for z in our["last_k_mean_per_seed"]],
                    "n_clearing_bar": int(our["n_clearing_bar"]),
                    "verdict": our["verdict"]},
        "sb3_arms": {c: {"config": SB3_CONFIGS[c],
                         "last_k_mean_per_seed": [float(z) for z in arms[c]["last_k_mean_per_seed"]],
                         "per_seed_clears_bar": [bool(z) for z in
                                                 (np.asarray(arms[c]["last_k_mean_per_seed"], float) >= bar)],
                         "n_clearing_bar": int(arms[c]["n_clearing_bar"]),
                         "verdict": arms[c]["verdict"],
                         "origin": arms[c].get("origin", {}),
                         "seconds_per_seed": arms[c]["seconds_per_seed"],
                         "per_step_per_seed": arms[c]["per_step_per_seed"],
                         "eval_at_steps": arms[c]["eval_at_steps"],
                         "episode_returns_per_seed": arms[c]["episode_returns_per_seed"]}
                     for c in sb3_names},
        "our_arm_config": OURS,
        "config_diff_columns": ["项", "我们（scripts/31）", "SB3 aligned", "SB3 default"],
        "config_diff": [list(t) for t in CONFIG_DIFF],
        "attribution": attribution,
        "recommended_action": action,
        "caveats": [
            "对照仅用于定位，不用于宣称算法优劣（单任务、单预算、3 种子）。",
            "『过线』= 末 k 次评测均值 ≥ 随机基线 per-step + margin(1.5)；margin 是事先约定的‘值得关注的最小改善’，不是显著性阈值。",
            "SB3 的 Tanh 激活 / 初始化 / log_std 初值 / 常数 lr 与我们不同 —— 即便 aligned 配置也不是逐位同构。",
            "不重跑我们的臂（读 scripts/32 的机器产物）；两侧同机同 CPU 同 torch 版本。",
        ],
    }
    js = out / f"{args.tag}.json"
    js.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[34] 图 -> {png}")
    print(f"[34] 指标 -> {js}")
    print("[34] OK")


if __name__ == "__main__":
    main()
