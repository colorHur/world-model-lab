#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""脚本 32 —— **参照学习器有效性审计**（reference-learner audit）。

存在的理由（一句话）
--------------------
X9（想象训练）跑完得到一个"全员平坦"的结果 —— 但**对照臂（真实 PPO）自己也没学会**，
于是那次比较**根本没有分辨力**。本脚本把"先证明参照臂自己过了随机线，再解释 A/B 比较"
这条**前置门槛**固化成可复跑的工具。

★ 方法论（来自第 30 次自证伪，见课题空间 `成果台账.md`）
------------------------------------------------------
    任何 A/B 方法比较，跑前必须登记：
        「参照臂须先过『比 trivial 基线好 X』的门槛，否则实验 **作废(VOID)**，不是负结论(NEGATIVE)」。
    本脚本就是这条门槛的执行器：它**只回答一件事** ——
        **给定配置下，参照学习器（真实环境 PPO）到底会不会学？**

判定
----
    随机基线 per-step = μ_random（本脚本同口径实测）。
    VALID   : 末次 per-step >= μ_random + `--margin` 且 与轮次的 Pearson r >= 0.5
    VOID    : 否则 ⇒ **任何基于该参照的 A/B 比较都不能下结论**。

用法
----
    python scripts/32_reference_learner_audit.py                      # 默认 800 轮
    python scripts/32_reference_learner_audit.py --updates 200
    python scripts/32_reference_learner_audit.py --reward-scale 0.1   # 只缩放奖励（薄包装）
    python scripts/32_reference_learner_audit.py --config configs/pendulum.yaml

★ 单实现纪律
------------
真实 PPO 的**训练循环**不在本文件里复制 —— 它从 `scripts/31_x9_imagination.py`
按路径加载并**复用 `run_real_arm`**。项目硬约定：PPO 那套循环只能有一份实现
（同 `agents/ppo.py:clipped_surrogate` 的教训），否则两条实现会漂移，
审计出来的行为就不再指向正式跑的那个臂。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wmlab.envs import make_env
from wmlab.envs.base import StepResult
from wmlab.utils import get_device, load_config, output_dir, set_seed
from wmlab.utils.plot import PALETTE, apply_style, save_fig

_HERE = Path(__file__).resolve()
REPO_ROOT = _HERE.parents[1]


def _load_x9_arm_module():
    """按路径加载 `scripts/31_x9_imagination.py`，只为**复用**其中的 `run_real_arm`。"""
    target = REPO_ROOT / "scripts" / "31_x9_imagination.py"
    if not target.exists():
        raise RuntimeError(f"找不到 {target} —— 脚本 32 复用它的 run_real_arm，不能被删/改名")
    spec = importlib.util.spec_from_file_location("_x9_arm_src", target)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)   # __name__ != '__main__' ⇒ 不会触发它的 main()
    if not hasattr(mod, "run_real_arm"):
        raise RuntimeError("scripts/31 里没有 run_real_arm —— 复用契约被破坏")
    return mod


class RewardScaledEnv:
    """薄包装：只把 `reward` 乘一个常数，其余逐字转交。

    ★ 用途：把"奖励尺度"做成**可复跑的旋钮**，而不是每次手改库代码。
    ★ 注意：`evaluate` 的 return 也同比例缩放 ⇒ 报出的 per-step 要**除以 scale 折回原单位**。
    """

    def __init__(self, base, scale: float):
        self.base = base
        self.scale = float(scale)
        self.obs_dim = base.obs_dim
        self.act_dim = base.act_dim
        self.act_low, self.act_high = base.act_low, base.act_high
        self.is_discrete = getattr(base, "is_discrete", False)
        self.name = f"{getattr(base, 'name', 'env')}_rscale{scale}"

    def reset(self, seed=None):
        return self.base.reset(seed)

    def step(self, action):
        r = self.base.step(action)
        return StepResult(r.obs, float(r.reward) * self.scale, r.terminated,
                          r.truncated, r.info)

    def sample_action(self, rng=None):
        return self.base.sample_action(rng)

    def close(self):
        self.base.close()


def parse_args():
    p = argparse.ArgumentParser(description="参照学习器有效性审计（真实 PPO on Pendulum）")
    p.add_argument("--config", default="configs/pendulum.yaml")
    p.add_argument("--tag", default="32_reference_learner_audit")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--updates", type=int, default=800, help="参照臂的训练轮数（每轮 512 步）")
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--n-eval", type=int, default=10)
    p.add_argument("--margin", type=float, default=1.5,
                   help="「比随机基线好多少」才算 VALID（per-step，原单位）")
    p.add_argument("--reward-scale", type=float, default=1.0,
                   help="把 reward 乘该常数（=1 表示不缩放）")
    return p.parse_args()


def random_baseline(env, n_episodes, seed, max_steps):
    """同口径随机基线（与 scripts/31 的 random_policy_return 算法一致）。"""
    rng = np.random.default_rng(seed)
    rets, per_step = [], []
    for i in range(n_episodes):
        env.reset(seed=seed + i)
        R, t, done = 0.0, 0, False
        while not done:
            a = env.sample_action(rng)
            res = env.step(a)
            R += float(res.reward)
            t += 1
            done = bool(res.done) or (max_steps is not None and t >= max_steps)
        rets.append(R)
        per_step.append(R / max(t, 1))
    return {"return_mean": float(np.mean(rets)), "return_std": float(np.std(rets)),
            "per_step_mean": float(np.mean(per_step))}


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(args.seed))
    apply_style()
    device = get_device(args.device or cfg.get("device"))
    out = output_dir(cfg)
    max_steps = int(cfg["env"].get("max_steps", 200))
    scale = float(args.reward_scale)

    x9mod = _load_x9_arm_module()
    env_raw = make_env(cfg["env"]["id"], seed=int(args.seed))
    env = env_raw if scale == 1.0 else RewardScaledEnv(env_raw, scale)

    # 基线也在**同一缩放下**测，然后折算回原单位
    base = random_baseline(env, int(args.n_eval), 30_000, max_steps)
    base_per_step = base["per_step_mean"] / scale
    print(f"[32] 参照有效性审计  env={cfg['env']['id']}  reward_scale={scale}  "
          f"updates={args.updates}×512")
    print(f"[32] 随机基线 return={base['return_mean'] * 1.0:.1f} (缩放单位) "
          f"=> per-step(原单位) {base_per_step:.3f}")

    arm_args = SimpleNamespace(samples_per_update=512, eval_every=int(args.eval_every),
                               n_eval=int(args.n_eval), config_max_steps=max_steps)
    t0 = time.time()
    _, hist, _ = x9mod.run_real_arm(env, arm_args, device, seed=2000,
                                    n_updates=int(args.updates))
    env.close()
    print(f"[32] 参照臂用时 {time.time() - t0:.0f}s  "
          f"（真环境步数 {int(args.updates) * 512}）")

    # 折算回原单位
    v = np.asarray([z for z in hist["real_per_step"] if not np.isnan(z)], dtype=float) / scale
    x = np.arange(v.size, dtype=float)
    r = float(np.corrcoef(x, v)[0, 1]) if v.size > 2 else float("nan")
    slope = float(np.polyfit(x, v, 1)[0]) if v.size > 2 else float("nan")
    bar = base_per_step + float(args.margin)
    valid = bool((v[-1] >= bar) and (r >= 0.5))

    print(f"[32] per-step(原单位): first {v[0]:+.3f}  last {v[-1]:+.3f}  "
          f"best {v.max():+.3f}  slope {slope:+.5f}/eval  r {r:+.3f}")
    print(f"[32] 门槛 = 随机基线 {base_per_step:+.3f} + margin {args.margin} = {bar:+.3f}")
    print("[32] ================= 参照有效性判定 =================")
    if valid:
        print("[32] VALID   —— 参照臂自己会学 ⇒ 可以据此解释 A/B 比较")
    else:
        print("[32] **VOID** —— 参照臂没过随机线 ⇒ **任何基于它的 A/B 比较都不能下结论**")
        print("[32] 注意：这是「作废(VOID)」，**不是**「负结论(NEGATIVE)」—— 两者必须分开报")

    summary = {
        "experiment": "reference-learner audit (script 32)",
        "env": cfg["env"]["id"], "seed": int(args.seed), "device": str(device.type),
        "reward_scale": scale, "n_updates": int(args.updates),
        "samples_per_update": 512, "env_steps": int(args.updates) * 512,
        "random_baseline_per_step": base_per_step,
        "per_step_series": [float(z) for z in v],
        "first": float(v[0]), "last": float(v[-1]), "best": float(v.max()),
        "slope_per_eval": slope, "pearson_r": r,
        "margin": float(args.margin), "bar": bar, "valid": valid,
        "verdict": "VALID" if valid else "VOID",
        "note": ("VOID = 参照臂无效 ⇒ 基于它的 A/B 比较作废；这不是『方法无效』的负结论。"
                 "本审计复用 scripts/31 的 run_real_arm（单实现纪律）。"),
    }
    js = out / f"{args.tag}.json"
    js.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.3))
    ax = axes[0]
    ax.plot(x, v, "o-", ms=3, lw=1.4, color=PALETTE["blue"], label="reference (real PPO)")
    ax.axhline(base_per_step, color=PALETTE["grey"], ls=":", lw=1.5,
               label=f"random policy ({base_per_step:.2f})")
    ax.axhline(bar, color=PALETTE["red"], ls="--", lw=1.4,
               label=f"VALID bar ({bar:.2f} = base + {args.margin})")
    ax.set_xlabel("PPO update"); ax.set_ylabel("true-env reward/step (original units)")
    ax.set_title(f"(1) reference learner: does it clear the random bar?  [{summary['verdict']}]")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    ax = axes[1]
    ax.plot(hist["update"], hist["std"], "-", color=PALETTE["purple"], lw=1.4)
    ax.set_xlabel("PPO update"); ax.set_ylabel("policy sigma")
    ax.set_title("(2) policy sigma (moves, but no return gain)")
    ax.grid(alpha=0.25)
    fig.suptitle(f"wmlab script 32 · reference-learner audit · {cfg['env']['id']} · "
                 f"reward_scale={scale} · updates={args.updates} · "
                 f"env_steps={int(args.updates) * 512}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    png = save_fig(fig, out / f"{args.tag}.png")
    plt.close(fig)
    print(f"[32] 图 -> {png}")
    print(f"[32] 指标 -> {js}")
    print("[32] OK")


if __name__ == "__main__":
    main()
