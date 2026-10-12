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

判定（v0.3 · 2026-10-12 统一规则 + 配对化）
------------------------------------------
    随机基线 per-step = μ_random（本脚本同口径实测，**初值种子与策略评测共用同一批**）。
    bar  = μ_random + `--margin`（margin 是**事先约定的"值得关注的最小改善"**，
           不是用标准误反推的显著性阈值）。
    VALID: **每一个**训练种子的「**末 k 次评测均值**」都 ≥ bar
    VOID : 否则 ⇒ **任何基于该参照的 A/B 比较都不能下结论**。
    ⇒ ★ 规则**单一真源** = `wmlab/eval/audit.py:reference_learner_verdict`
      （代码 / 本说明 / JSON / 测试**都从它取**，杜绝"代码按 A 判、文字按 B 写"）。

    ★ 为什么是"每个种子都过线"而不是"跨种子均值过线"：
      多种子设计的全部意义就是防"某一种子运气好"；取均值会把它摊回来。
      反例：改善 [0.1, 0.1, 4.4]、margin 1.5 ⇒ 均值 1.533**过线**，但两个种子几乎没动。
      ⇒ 取**保守方向**（宁可真 VOID，不可假 VALID）。

    ★ v0.1 → v0.3 改动（理由都在实测里，不是风格偏好）：
      1. **末次单点 → 末 k 次均值**（`--last-k`，默认 5）：`last` 跨种子离散度是
         `best` 的 **4.3 倍** ⇒ 单点判据本身就抖。
      2. **单种子 → 多种子**（`--seeds`，默认 0,1,2），每种子都要过线。
      3. **Pearson r 从判据降为诊断**：r 是**形状**量，不是"过/不过"门槛 ——
         快速学会后长期平台的学习器，r 会被平台期**压低而误杀**
         （100 点、首点 0、其余 99 点为 1 ⇒ r ≈ **0.172**）；
         缓慢线性上升但**远没到线**的学习器，r 反而可能很高。两个方向都不成立。
      4. **配对评测**：随机基线与 PPO 改用**同一批固定初值种子**
         （`scripts/31:fixed_eval_seeds`）；保存**逐回合**回报 ⇒ 差值是真配对差值，
         不再把"场景波动"混进"策略改善"。
      5. **随机动作**走**独立生成器**（`action_seed`），与初值种子分离。

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
from wmlab.eval import reference_learner_verdict
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


def random_baseline(env, seeds, action_seed, max_steps):
    """同口径随机基线（与 scripts/31 的 `random_policy_return` 算法一致）。

    ★ 2026-10-12 配对化：
      - 用**显式初值种子列表** `seeds`（= PPO 评测用的同一批）⇒ 两边面对**同一批**
        初始条件，差值才是**配对差值**；
      - 随机动作走**独立生成器** `action_seed`（与初值种子分离）。
    """
    rng = np.random.default_rng(int(action_seed))
    rets, per_step, lens = [], [], []
    for s in seeds:
        env.reset(seed=int(s))
        R, t, done = 0.0, 0, False
        while not done:
            a = env.sample_action(rng)
            res = env.step(a)
            R += float(res.reward)
            t += 1
            done = bool(res.done) or (max_steps is not None and t >= max_steps)
        rets.append(float(R))
        per_step.append(float(R) / max(t, 1))
        lens.append(int(t))
    return {"return_mean": float(np.mean(rets)), "return_std": float(np.std(rets)),
            "per_step_mean": float(np.mean(per_step)),
            "returns": rets, "per_step": per_step, "lengths": lens}


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

    # ★ 固定评测初值（单一真源 = scripts/31）；随机基线与 PPO **共用同一批** ⇒ 配对
    eval_seeds = x9mod.fixed_eval_seeds(int(args.n_eval))
    base = random_baseline(env, eval_seeds, 30_000, max_steps)
    base_per_step = base["per_step_mean"] / scale
    base_ps = np.asarray(base["per_step"], dtype=float) / scale   # 逐回合 per-step（原单位）
    bar = base_per_step + float(args.margin)
    print(f"[32] 参照有效性审计 v0.3（配对初值 + 多种子 + 末 k 均值；r 只做诊断）")
    print(f"[32] env={cfg['env']['id']}  reward_scale={scale}  "
          f"seeds={seeds}  updates={args.updates}×512  last_k={args.last_k}")
    print(f"[32] 评测初值（随机与 PPO 共用）：{eval_seeds[0]}..{eval_seeds[-1]} "
          f"(n={len(eval_seeds)})")
    print(f"[32] 随机基线 per-step(原单位) {base_per_step:.3f}；VALID 门槛 = {bar:.3f}")

    arm_args = SimpleNamespace(samples_per_update=512, eval_every=int(args.eval_every),
                               n_eval=int(args.n_eval), config_max_steps=max_steps)
    runs = {}
    for sd in seeds:
        t0 = time.time()
        _, hist, _ = x9mod.run_real_arm(env, arm_args, device,
                                        seed=2000 + 1000 * sd, n_updates=int(args.updates),
                                        eval_seeds=eval_seeds)
        # ★ 逐回合 per-step（原单位）：与 base_ps 同长度、同顺序 ⇒ 可**配对**相减
        ev_rets, ev_lens = hist["eval_returns"], hist["eval_lengths"]

        def _ps(rets_row, lens_row):
            return (np.asarray(rets_row, dtype=float)
                    / np.maximum(np.asarray(lens_row, dtype=float), 1.0)) / scale

        per_eval_ps = np.asarray([float(np.mean(_ps(a, b)))
                                  for a, b in zip(ev_rets, ev_lens)], dtype=float)
        paired_per_eval = np.asarray([float(np.mean(_ps(a, b) - base_ps))
                                      for a, b in zip(ev_rets, ev_lens)], dtype=float)
        n_ev = per_eval_ps.size
        sig = np.asarray(hist["std"], dtype=float)
        x = np.arange(n_ev, dtype=float)
        k = int(min(int(args.last_k), n_ev))
        runs[sd] = {
            "seed": sd,
            "per_step": [float(z) for z in per_eval_ps],
            "sigma": [float(z) for z in sig],
            "last_k": k,
            "last_k_mean": float(per_eval_ps[-k:].mean()),
            "first_k_mean": float(per_eval_ps[:k].mean()),
            "last": float(per_eval_ps[-1]), "best": float(per_eval_ps.max()),
            "delta_vs_base": float(per_eval_ps[-k:].mean() - base_per_step),
            "paired_delta_vs_base": float(paired_per_eval[-k:].mean()),
            "slope_per_eval": (float(np.polyfit(x, per_eval_ps, 1)[0])
                               if n_ev > 2 else float("nan")),
            "pearson_r": (float(np.corrcoef(x, per_eval_ps)[0, 1])
                          if n_ev > 2 else float("nan")),
            "seconds": float(time.time() - t0),
            "env_steps": int(args.updates) * 512,
        }
        r = runs[sd]
        print(f"[32] seed={sd:3d}  末k均值 {r['last_k_mean']:+.3f} "
              f"(Δ均值 {r['delta_vs_base']:+.3f} / 配对Δ {r['paired_delta_vs_base']:+.3f})  "
              f"last {r['last']:+.3f}  best {r['best']:+.3f}  "
              f"r {r['pearson_r']:+.3f}  [{r['seconds']:.0f}s]")
    env.close()

    lk = np.asarray([runs[s]["last_k_mean"] for s in seeds], dtype=float)
    dlt = np.asarray([runs[s]["delta_vs_base"] for s in seeds], dtype=float)
    pdlt = np.asarray([runs[s]["paired_delta_vs_base"] for s in seeds], dtype=float)
    rr = np.asarray([runs[s]["pearson_r"] for s in seeds], dtype=float)
    # ★★ 判定走**单一真源**（wmlab/eval/audit.py），不再在脚本里另写一套
    valid, det = reference_learner_verdict(lk, bar)
    verdict = "VALID" if valid else "VOID"

    print("[32] ================= 参照有效性判定（单一规则 · 多种子）================")
    print(f"[32] 规则：{det['rule']}")
    print(f"[32] 末 {args.last_k} 次均值（逐种子）：{np.round(lk, 3).tolist()}")
    print(f"[32] 跨种子 {det['mean_last_k_over_seeds']:+.3f} ± "
          f"{det['std_last_k_over_seeds']:.3f} (n={det['n_seeds']})；门槛 {bar:+.3f}")
    print(f"[32] 过线种子数 {det['n_clearing_bar']}/{det['n_seeds']}"
          f"；逐种子过线 {det['per_seed_clears_bar']}")
    print(f"[32] 配对差值（**同初值逐回合**，末k均值）：{np.round(pdlt, 3).tolist()}")
    print(f"[32] 诊断项（**不参与判定**）：Pearson r = {np.round(rr, 3).tolist()}"
          f"（形状量；快速学会后长期平台会把 r 压低，故不当门槛）")
    if valid:
        print("[32] VALID   —— 参照臂在**所有**训练种子下都自己会学 ⇒ 可据此解释 A/B 比较")
    else:
        print("[32] **VOID** —— 参照臂没过随机线 ⇒ **任何基于它的 A/B 比较都不能下结论**")
        print("[32] 注意：这是「作废(VOID)」，**不是**「负结论(NEGATIVE)」—— 两者必须分开报")

    summary = {
        "experiment": "reference-learner audit (script 32, v0.3 paired-eval multi-seed)",
        "env": cfg["env"]["id"], "device": str(device.type),
        "reward_scale": scale,
        "seeds": seeds,
        "n_updates": int(args.updates),
        "samples_per_update": 512,
        "env_steps_per_seed": int(args.updates) * 512,
        "last_k": int(args.last_k),
        "eval_seed_base": int(x9mod.EVAL_SEED_BASE),
        "eval_seeds": [int(z) for z in eval_seeds],
        "paired_evaluation": True,
        "random_action_seed": 30_000,
        "random_baseline_per_step": base_per_step,
        "margin": float(args.margin), "bar": bar,
        "verdict_rule": det["rule"],
        "last_k_mean_per_seed": [float(z) for z in lk],
        "per_seed_clears_bar": det["per_seed_clears_bar"],
        "n_clearing_bar": det["n_clearing_bar"],
        "min_last_k_mean": det["min_last_k_mean"],
        "mean_last_k_over_seeds": det["mean_last_k_over_seeds"],
        "std_last_k_over_seeds": det["std_last_k_over_seeds"],
        "paired_deltas_vs_base": [float(z) for z in pdlt],
        "unpaired_delta_of_means": [float(z) for z in dlt],
        "pearson_r_per_seed": [float(z) for z in rr],
        "pearson_r_is_diagnostic_only": True,
        "valid": valid, "verdict": verdict,
        "runs": {str(s): runs[s] for s in seeds},
        "note": (f"v0.3 判据（单一真源 wmlab/eval/audit.py）：{det['rule']}"
                 "评测初值**固定**且随机基线 / PPO **共用同一批**（scripts/31:fixed_eval_seeds），"
                 "差值按**逐回合配对**计算（保存 per-episode 回报）；随机动作走独立生成器。"
                 "Pearson r 仅作诊断：它是形状量（快速学会后长期平台会把它压低）⇒ 不当门槛。"
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
    ax.set_xlabel("evaluation index"); ax.set_ylabel("true-env reward/step (original units)")
    ax.set_title(f"(1) reference learner across {len(seeds)} seeds  [{verdict}]")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

    ax = axes[1]
    for sd in seeds:
        ax.plot(np.asarray(runs[sd]["sigma"], dtype=float), "-", lw=1.2, alpha=0.6,
                label=f"seed {sd}")
    ax.set_xlabel("evaluation index"); ax.set_ylabel("policy sigma")
    ax.set_title("(2) policy sigma per seed")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

    fig.suptitle(f"wmlab script 32 v0.3 · reference-learner audit (paired eval) · "
                 f"{cfg['env']['id']} · reward_scale={scale} · seeds={seeds} · "
                 f"updates={args.updates} · last_k={args.last_k}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    png = save_fig(fig, out / f"{args.tag}.png")
    plt.close(fig)
    print(f"[32] 图 -> {png}")
    print(f"[32] 指标 -> {js}")
    print("[32] OK")


if __name__ == "__main__":
    main()
