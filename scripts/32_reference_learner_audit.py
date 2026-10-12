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
    本仓库的 A/B 流程要求：跑 A/B 方法比较前先登记
        「参照臂须先过『比 trivial 基线好 X』的门槛，否则实验 **作废(VOID)**，不是负结论(NEGATIVE)」。
    ★ 这条是**本仓库的流程纪律**，**不是**"一切 A/B 比较的普适规则"（2026-10-12 收窄）——
      它只保证"拿去比的参照确实会学"，别的项目未必要照搬。

判定（v0.2 · 2026-10-12 改口径）
-------------------------------
    随机基线 per-step = μ_random（本脚本同口径实测，评测种子固定）。
    VALID : **每个训练种子**的「**末 k 次评测的均值**」都 ≥ μ_random + `--margin`
    VOID  : 否则 ⇒ **任何基于该参照的 A/B 比较都不能下结论**。

    ★ 为什么换成这个（v0.1 是"末次 ≥ bar **且** Pearson r ≥ 0.5"，两头都不稳）：
      1. **末次单点**受波动影响 ⇒ 改**末 k 次**（`--last-k`，默认 5）的均值。
      2. **单种子**分不清"真学会了"与"这一次运气好" ⇒ 改**多种子**（`--seeds`，
         默认 0,1,2），每种子都要过线。
      3. **Pearson r 不当判据**：它是**形状**量，不是"过/不过"的门槛。反例两个方向
         都有 —— 快速学会后长期平台的学习器，r 会被平台期噪声**压低而误杀**；
         缓慢线性上升但**远没到线**的学习器，r 反而可能很高。⇒ r 只打印、不判。
      4. 评测种子**固定且与训练种子无关**（`run_real_arm` 用 `20_000 + update`，不含
         训练种子）⇒ 各种子面对**同一批**评测回合，配对差值才有意义。

用法
----
    python scripts/32_reference_learner_audit.py                         # 3 种子 × 800 轮
    python scripts/32_reference_learner_audit.py --seeds 0,1,2,3,4
    python scripts/32_reference_learner_audit.py --updates 20 --seeds 0  # 冒烟
    python scripts/32_reference_learner_audit.py --reward-scale 0.1
    python scripts/32_reference_learner_audit.py --margin 2.0 --last-k 8

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
    p.add_argument("--seed", type=int, default=0, help="只用于基线取样；评测种子由 run_real_arm 内部固定")
    p.add_argument("--device", default=None)
    p.add_argument("--seeds", default="0,1,2", help="训练种子列表（逗号分隔）")
    p.add_argument("--updates", type=int, default=800, help="每种子的训练轮数（每轮 512 步）")
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--n-eval", type=int, default=10)
    p.add_argument("--last-k", type=int, default=5, help="判定用的「末 k 次评测」")
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
    seeds = [int(s) for s in str(args.seeds).split(",") if str(s).strip()]

    x9mod = _load_x9_arm_module()
    env_raw = make_env(cfg["env"]["id"], seed=int(args.seed))
    env = env_raw if scale == 1.0 else RewardScaledEnv(env_raw, scale)

    # 基线也在**同一缩放下**测，然后折算回原单位
    base = random_baseline(env, int(args.n_eval), 30_000, max_steps)
    base_per_step = base["per_step_mean"] / scale
    bar = base_per_step + float(args.margin)
    print(f"[32] 参照有效性审计 v0.2（多种子 + 末 k 均值；r 只做诊断）")
    print(f"[32] env={cfg['env']['id']}  reward_scale={scale}  "
          f"seeds={seeds}  updates={args.updates}×512  last_k={args.last_k}")
    print(f"[32] 随机基线 return={base['return_mean'] * 1.0:.1f} (缩放单位) "
          f"=> per-step(原单位) {base_per_step:.3f}；VALID 门槛 = {bar:.3f}")

    arm_args = SimpleNamespace(samples_per_update=512, eval_every=int(args.eval_every),
                               n_eval=int(args.n_eval), config_max_steps=max_steps)
    runs = {}
    for sd in seeds:
        t0 = time.time()
        _, hist, _ = x9mod.run_real_arm(env, arm_args, device,
                                        seed=2000 + 1000 * sd, n_updates=int(args.updates))
        # 折算回原单位
        v = np.asarray([z for z in hist["real_per_step"] if not np.isnan(z)],
                       dtype=float) / scale
        sig = np.asarray(hist["std"], dtype=float)
        x = np.arange(v.size, dtype=float)
        k = int(min(int(args.last_k), v.size))
        runs[sd] = {
            "seed": sd,
            "per_step": [float(z) for z in v],
            "sigma": [float(z) for z in sig],
            "last_k": k,
            "last_k_mean": float(v[-k:].mean()),
            "first_k_mean": float(v[:k].mean()),
            "last": float(v[-1]), "best": float(v.max()),
            "delta_vs_base": float(v[-k:].mean() - base_per_step),
            "slope_per_eval": float(np.polyfit(x, v, 1)[0]) if v.size > 2 else float("nan"),
            "pearson_r": float(np.corrcoef(x, v)[0, 1]) if v.size > 2 else float("nan"),
            "seconds": float(time.time() - t0),
            "env_steps": int(args.updates) * 512,
        }
        r = runs[sd]
        print(f"[32] seed={sd:3d}  末k均值 {r['last_k_mean']:+.3f} "
              f"(Δ vs 基线 {r['delta_vs_base']:+.3f})  last {r['last']:+.3f}  "
              f"best {r['best']:+.3f}  r {r['pearson_r']:+.3f}  [{r['seconds']:.0f}s]")
    env.close()

    lk = np.asarray([runs[s]["last_k_mean"] for s in seeds], dtype=float)
    dlt = np.asarray([runs[s]["delta_vs_base"] for s in seeds], dtype=float)
    rr = np.asarray([runs[s]["pearson_r"] for s in seeds], dtype=float)
    mean_lk = float(lk.mean())
    std_lk = float(lk.std(ddof=1)) if lk.size > 1 else 0.0
    all_above = bool((dlt > 0).all())
    valid = bool((mean_lk >= bar) and all_above)
    verdict = "VALID" if valid else "VOID"

    print("[32] ================= 参照有效性判定（多种子）================")
    print(f"[32] 末 {args.last_k} 次均值：跨种子 {mean_lk:+.3f} ± {std_lk:.3f} (n={lk.size})"
          f"；门槛 {bar:+.3f}")
    print(f"[32] 配对差值（每种子末k均值 − 基线）：{np.round(dlt, 3).tolist()}")
    print(f"[32] 方向一致性（所有种子都 > 基线）：{all_above}")
    print(f"[32] 诊断项（**不参与判定**）：Pearson r = {np.round(rr, 3).tolist()}"
          f"（形状量；快速学会后长期平台会把 r 压低，故不当门槛）")
    if valid:
        print("[32] VALID   —— 参照臂在**所有**训练种子下都自己会学 ⇒ 可据此解释 A/B 比较")
    else:
        print("[32] **VOID** —— 参照臂没过随机线 ⇒ **任何基于它的 A/B 比较都不能下结论**")
        print("[32] 注意：这是「作废(VOID)」，**不是**「负结论(NEGATIVE)」—— 两者必须分开报")

    summary = {
        "experiment": "reference-learner audit (script 32, v0.2 multi-seed)",
        "env": cfg["env"]["id"], "device": str(device.type),
        "reward_scale": scale,
        "seeds": seeds,
        "n_updates": int(args.updates),
        "samples_per_update": 512,
        "env_steps_per_seed": int(args.updates) * 512,
        "last_k": int(args.last_k),
        "random_baseline_per_step": base_per_step,
        "margin": float(args.margin), "bar": bar,
        "mean_last_k_over_seeds": mean_lk, "std_last_k_over_seeds": std_lk,
        "paired_deltas_vs_base": [float(z) for z in dlt],
        "all_seeds_above_base": all_above,
        "pearson_r_per_seed": [float(z) for z in rr],
        "pearson_r_is_diagnostic_only": True,
        "valid": valid, "verdict": verdict,
        "runs": {str(s): runs[s] for s in seeds},
        "note": ("v0.2 判据 = 每个训练种子的末 k 次评测均值都 ≥ 随机基线 + margin；"
                 "评测种子固定且与训练种子无关（20_000 + update）。"
                 "Pearson r 仅作诊断：它是形状量，快速学会后长期平台会把它压低 ⇒ 不当门槛。"
                 "VOID = 参照臂无效 ⇒ 基于它的 A/B 比较作废；这不是『方法无效』的负结论。"
                 "本审计复用 scripts/31 的 run_real_arm（单实现纪律）。"),
    }
    js = out / f"{args.tag}.json"
    js.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------------------------------------------------------------- 出图
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.3))
    ax = axes[0]
    for sd in seeds:
        v = np.asarray(runs[sd]["per_step"], dtype=float)
        ax.plot(np.arange(v.size), v, "o-", ms=2.5, lw=1.1, alpha=0.5, label=f"seed {sd}")
    vmean = np.mean([np.asarray(runs[sd]["per_step"], dtype=float) for sd in seeds], axis=0)
    ax.plot(np.arange(vmean.size), vmean, "-", lw=2.4, color=PALETTE["blue"],
            label="mean over seeds")
    ax.axhline(base_per_step, color=PALETTE["grey"], ls=":", lw=1.5,
               label=f"random policy ({base_per_step:.2f})")
    ax.axhline(bar, color=PALETTE["red"], ls="--", lw=1.4,
               label=f"VALID bar ({bar:.2f} = base + {args.margin})")
    kk = min(int(args.last_k), vmean.size)
    ax.axvspan(vmean.size - kk - 0.5, vmean.size - 0.5, color=PALETTE["red"], alpha=0.08)
    ax.set_xlabel("PPO update"); ax.set_ylabel("true-env reward/step (original units)")
    ax.set_title(f"(1) reference learner across {len(seeds)} seeds  [{verdict}]")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

    ax = axes[1]
    for sd in seeds:
        ax.plot(np.asarray(runs[sd]["sigma"], dtype=float), "-", lw=1.2, alpha=0.6,
                label=f"seed {sd}")
    ax.set_xlabel("PPO update"); ax.set_ylabel("policy sigma")
    ax.set_title("(2) policy sigma per seed")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

    fig.suptitle(f"wmlab script 32 v0.2 · reference-learner audit · {cfg['env']['id']} · "
                 f"reward_scale={scale} · seeds={seeds} · updates={args.updates} · "
                 f"last_k={args.last_k}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    png = save_fig(fig, out / f"{args.tag}.png")
    plt.close(fig)
    print(f"[32] 图 -> {png}")
    print(f"[32] 指标 -> {js}")
    print("[32] OK")


if __name__ == "__main__":
    main()
