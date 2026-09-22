#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""脚本 17 —— ★ X31：**突发信道**（Gilbert–Elliott）下的世界模型顶包

一句话
------
X24–X27 的信道是**无记忆伯努利丢包**。真实无线信道是**突发**的（一丢丢一串）。
X31 把信道换成 Gilbert–Elliott 两状态模型，回答一件事：

    **同样的平均丢包率下，"突发"到底是让误差变大还是变小？**

★ 为什么这个问题有悬念（不能想当然）
--------------------------------------
直觉：突发 ⇒ 长的信息年龄尾巴 ⇒ 误差更大。但由 Jensen 不等式，

    E[NMSE] = E_a[ f(a) ]，   f(a) = 离线闭环 NMSE 曲线在视界 a 处的值

- 若 f **凸** ⇒ age 分布变宽 ⇒ E[f] **变大** ⇒ 突发更糟（直觉成立）
- 若 f **凹** ⇒ age 分布变宽 ⇒ E[f] **变小** ⇒ **突发反而更好**

而 **X2 已经实测到 f 是饱和的（凹的）** —— 长视界处每步增量在下降。
所以"突发更糟"**不是显然的**，本实验的答案是开放的。

⇒ **本脚本先做 f 的凹凸性诊断（二阶差分），把 Jensen 的方向从数据推出来，
   再去看实测结果 —— 不许先看结果再编解释（规范 R2）。**

★★ 一个必须处理的混淆（本实验设计里最关键的一处）
--------------------------------------------------
固定平均丢包率 p̄ 扫突发长度 L 时，**E[age] 不是常数**：

    Gilbert 模型下 Bad 游程 n ~ Geom(β)，游程内 age 取 1…n，于是
        E[age] = p̄ · L               （推导见 GilbertElliottChannel 类 docstring）

即"突发性"与"平均信息年龄"**耦合**。直接报"L 变大 ⇒ 误差变大"是**没有控制变量**的结论。

**处置**：对每个 L 额外算一个**同均值几何对照**

    E_geo[NMSE] = Σ_h Geom(h; 均值 = 实测 E[age]) · f(h)

（把 age 分布换成"均值相同、但无记忆"的几何分布）。定义

    **burst_gain = 实测 E[NMSE] / E_geo[NMSE]**

> 1 ⇒ 同样平均年龄下，突发让误差**变大**；< 1 ⇒ **变小**（凹性反转）。

★ 与 X26 的关系（本实验的第二个产出）
--------------------------------------
X26 验证过解析式 `E[NMSE] = Σ_h P_age(h)·NMSE(h)`，但**只在 i.i.d. 丢包下**验过。
突发信道下 age 分布不再是几何分布 —— 这条式子还成立吗？
本实验用**实测的 age 直方图**代入，逐点对照实测 E[NMSE]。
若成立 ⇒ X26 的解析式从"i.i.d. 专用"推广为"**任意信道可用**"，这是真贡献。

★ 自检（不过就抛，不产出数字 —— 规范 R14「报错优于给假数字」）
----------------------------------------------------------------
S1  GE 的解析量：π_B = p̄、E[L] = L、ρ₁ = 1−α−β（闭式，零容差）
S2  经验丢包率 vs p̄，在 4σ 内。★ σ 由**有效样本量 = Bad 游程数**估计，
    **不能给固定百分比阈值** —— 长突发时游程数只有百量级，方差天然大
    （第一版用 1% 固定阈值，L=16 被 3σ 的纯噪声误杀）
S3  ★ 锚点：**L = L_iid 时 GE 的实测 E[NMSE] 必须与 i.i.d. 基线一致**
    （把「GE 实现」与「lossy_schedule 实现」两个独立实现接上的唯一锚点）
S4  闭式年龄分布的均值 = 实测 E[age]（4σ，同样按有效样本量给 σ）
S5  全部指标有限（无 NaN/Inf）
S6  ★★ 实测**条件**年龄分布必须 = Geom(β)（TV 距离，阈值随样本量收紧）
    —— 这是「一阶/二阶分解」成立的前提；不过则整个分解不可信

★★★ 本脚本实测踩到并修掉的两个口径陷阱（都是"看着完全合理"的假结果）
------------------------------------------------------------------------
**陷阱 1 · 离线曲线被自己的最大视界挤进瞬态区**
`_sample_windows` 的起点范围是 `t0 ∈ [0, T − h_max − 1)`：**h_max 越大，起点越少，
且全部挤在 episode 开头**。UAV 的捕获瞬态恰好占前 ~250 步 ⇒ 离线 f(h) 几乎只在
瞬态区取样，而在线跑满全程。表现：f(1) 高估 **49%**、f(6) 高估 11%，
且**偏差随 h 增大而减小**（瞬态的一步误差大，rollout 越长起点影响越被稀释）
—— 这种"随 h 收敛"的形状很容易被误读成机制。
修法：给 `closed_loop_error_curve` 加 `t0_min`，在线 `run_tracking` 用同样的
`warmup`，**两边都在稳态区**；episode 长度同时拉到 700（否则稳态起点只剩 10 个）。

**陷阱 2 · `n_tx` 与 `n_steps` 不在同一区间**
`run_tracking` 里 `n_tx` 在预热期也累加、`n_steps` 只记预热之后 ⇒
`tx_rate` 被系统性高估（L=1 实测 0.778 而非 0.5，差 23σ）。
X26/X27 因为 `warmup=0` 从未触发。修法：发送**行为**照常，计数只在记账区间累加。

运行
----
    python scripts/17_burst_channel.py --config configs/uav_burst.yaml
    python scripts/17_burst_channel.py --config configs/uav_burst.yaml --quick   # 冒烟
"""

from __future__ import annotations

import argparse
import csv
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

from wmlab.control import (PDRelativeController, collect_controlled_episodes,
                           run_closed_loop_control)
from wmlab.data import split_episodes, transitions_from_episodes
from wmlab.envs import make_env
from wmlab.eval import reliable_horizon
from wmlab.eval.tracking import (GilbertElliottChannel, lossy_schedule, run_tracking,
                                 run_length_goodness, simulate_bad_runs)
from wmlab.models import MLPWorldModel
from wmlab.rollout import closed_loop_error_curve
from wmlab.train import train_world_model
from wmlab.utils import (count_params, describe_device, get_device, load_config,
                         output_dir, set_seed)
from wmlab.utils.plot import PALETTE, apply_style, save_fig


def parse_args():
    p = argparse.ArgumentParser(description="X31 突发信道（Gilbert–Elliott）")
    p.add_argument("--config", default="configs/uav_burst.yaml")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--tag", default="17_burst_channel")
    p.add_argument("--bursts", default=None, help="逗号分隔的突发长度网格，覆盖 config")
    p.add_argument("--quick", action="store_true", help="冒烟：少数据/少集/3 个突发点")
    return p.parse_args()


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def full_age_pmf(p_loss: float, beta: float, k_max: int) -> np.ndarray:
    """★ Gilbert–Elliott 下 age 的**精确**分布（闭式，不是拟合）。

        P(age = 0) = 1 − p̄
        P(age = k) = p̄ · β · (1−β)^(k−1),   k = 1 … k_max

    推导见 `GilbertElliottChannel` 类 docstring 的「age 的精确分布」一节：
    条件年龄分布恒为 Geom(β)，**与 p̄ 无关**；p̄ 只决定 age=0 那一份权重。

    ★★ 为什么必须保留 P(age=0)=1−p̄ 这一项
    第一版我用"匹配 E[age] 的几何分布"当对照，结果 burst_gain 系统性 <1，
    一度被读成"突发反而更好"。**那是口径错误**：均值匹配的几何分布会把
    P(age=0) 从 1−p̄ 压到 1/(1+E[age])（L=4 时 0.5 → 0.36），
    凭空多出 14% 的"非零误差"样本 ⇒ 对照被抬高 ⇒ 比值被压低。
    **固定 p̄ 时 P(age=0) 是不变量，任何对照都不许改它。**

    ★ 尾巴：k>k_max 的质量压到 k_max（f 饱和区，偏差可忽略；占比会打印出来）。
    """
    pl = float(p_loss)
    b = float(min(max(beta, 1e-12), 1.0))
    pmf = np.zeros(k_max + 1, dtype=np.float64)
    pmf[0] = 1.0 - pl
    for k in range(1, k_max + 1):
        pmf[k] = pl * b * ((1.0 - b) ** (k - 1))
    tail = max(0.0, 1.0 - float(pmf[:k_max].sum()))
    pmf[k_max] += tail
    return pmf / pmf.sum()


def age_goodness(cond_emp: np.ndarray, cond_th: np.ndarray, n_pos: int) -> dict:
    """★ 条件年龄分布的拟合优度（硬判定用 KS，TV 只作报告量）。

    ★★★ 第 14 次自我修正：TV 阈值**不能用 3/√n**
    ------------------------------------------------------------------
    第一版（X31）我用 `tv_thresh = max(0.03, 3/√n_pos)`。这个形式来自
    "比较两个二项比例"的直觉，但 TV 是**多类别**之和：

        TV = 0.5 Σ_k |p̂_k − p_k|
        E[TV | H0] ≈ 0.5 · √(2/(π n)) · Σ_k √(p_k (1−p_k)) ≈ 0.5 · √(2/(π n)) · Σ_k √p_k

    而 Geom(β) 的 **Σ_k √p_k = √β / (1 − √(1−β))**，这个因子随突发长度**发散**：
        β=1/4  ⇒ 2.0      β=1/16 ⇒ 6.1      β=1/32 ⇒ **11.2**
    ⇒ n=6076、β=1/32 时 **E[TV] ≈ 0.057**，而 3/√n 只有 0.038
    ⇒ **阈值比纯噪声还小，判据必然误杀**。（X31 侥幸没触发，只因那一个点
    的实测 TV 恰好偏小；X31-b 把 p̄ 铺开后立刻在 p̄=0.35,L=32 处炸了。）

    ⇒ 硬判定改用 **KS**：临界值 1.36/√n 与分布形状无关；对离散分布 KS 检验是
    **保守**的（实际一类错误 ≤ 名义值）⇒ 拿它做"不通过就抛"不会误杀。
    """
    n = max(int(n_pos), 1)
    ce = np.asarray(cond_emp, dtype=np.float64)
    ct = np.asarray(cond_th, dtype=np.float64)
    s = float(ct.sum())
    if s <= 0:
        raise ValueError("理论条件分布质量为 0")
    ct = ct / s                                   # 归一化（尾巴截断后仍要归一）
    tv = float(0.5 * np.abs(ce - ct).sum())
    # 解析的纯噪声期望（只作参考，不作判据）
    tv_noise = float(0.5 * np.sum(np.sqrt(2.0 * ct * (1.0 - ct) / (np.pi * n))))
    fe = np.cumsum(ce)
    ft = np.cumsum(ct)
    ks = float(np.max(np.abs(fe - ft)))
    ks_crit = 1.36 / np.sqrt(n)                   # 95%，对离散分布保守
    return {"tv": tv, "tv_noise": tv_noise, "ks": ks, "ks_crit": ks_crit,
            "ok": bool(ks <= ks_crit)}


def age_matched_geom_pmf(mean_age: float, k_max: int) -> np.ndarray:
    """（保留作对照口径）**全分布**匹配 E[age] 的无记忆几何分布。

    ★ 只在明确要做"年龄匹配"对照时用，且**必须连同它的 P(age=0) 一起报**
    —— 因为它不再等于 1−p̄，这正是它与 `full_age_pmf` 的全部差异。
    """
    m = float(max(mean_age, 1e-9))
    s = 1.0 / (1.0 + m)
    q = 1.0 - s
    pmf = np.array([(q ** k) * s for k in range(k_max + 1)], dtype=np.float64)
    pmf[-1] += max(0.0, 1.0 - float(pmf.sum()))
    return pmf / pmf.sum()


# ================================================================== 主流程
def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.bursts:
        cfg["channel"]["burst_lens"] = [float(x) for x in args.bursts.split(",")]
    if args.quick:
        cfg["train"]["epochs"] = min(int(cfg["train"]["epochs"]), 3)
        cfg["data"]["n_episodes"] = min(int(cfg["data"]["n_episodes"]), 20)
        cfg["channel"]["burst_lens"] = [1.0, 4.0, 16.0]
        cfg["channel"]["n_track_episodes"] = 8
        cfg["task"]["n_episodes"] = 5
        cfg["eval"]["n_samples"] = 64

    seed = int(cfg["seed"])
    set_seed(seed)
    apply_style()
    device = get_device(args.device or cfg.get("device"))
    out = output_dir(cfg)
    t0 = time.time()

    ch_cfg = cfg["channel"]
    p_loss = float(ch_cfg["p_loss"])
    bursts = [float(x) for x in ch_cfg["burst_lens"]]
    l_iid = 1.0 / (1.0 - p_loss)

    # ---------- 1) 信道参数自检 S1（闭式，零容差）----------
    print(f"[17] ★ X31 突发信道：p̄={p_loss}  L_iid=1/(1−p̄)={l_iid:.3f}")
    print(f"[17]   {'L':>6}{'α':>9}{'β':>9}{'ρ₁':>9}{'π_B':>9}{'E[age]':>9}{'机制':>12}")
    chans = {}
    for L in bursts:
        try:
            ch = GilbertElliottChannel(p_loss, L, seed=seed + 7)
        except ValueError as e:
            print(f"[17]   L={L:g} 不可行：{e}")
            continue
        # ★ S1：π_B 与 E[L] 必须与构造参数严格一致（1e-12）
        assert abs(ch.pi_bad - p_loss) < 1e-12, f"S1 失败：π_B={ch.pi_bad} ≠ p̄={p_loss}"
        assert abs(ch.mean_burst_len - L) < 1e-12, f"S1 失败：E[L]={ch.mean_burst_len} ≠ {L}"
        assert abs(ch.rho1 - (1 - ch.alpha - ch.beta)) < 1e-15, "S1 失败：ρ₁ 闭式不符"
        mech = ("交替(ρ₁<0)" if ch.rho1 < -1e-9 else
                "★无记忆=i.i.d." if abs(ch.rho1) <= 1e-9 else "突发(ρ₁>0)")
        chans[L] = ch
        print(f"[17]   {L:>6.0f}{ch.alpha:>9.4f}{ch.beta:>9.4f}{ch.rho1:>+9.4f}"
              f"{ch.pi_bad:>9.4f}{ch.expected_age:>9.2f}{mech:>12}")

    # ---------- 2) 环境 + 控制器 + 训练（与 X30 同口径）----------
    env = make_env(cfg["env"]["id"], seed=seed,
                   noise_std=float(cfg["env"]["noise_std"]),
                   max_steps=int(cfg["env"]["max_steps"]))
    cc = cfg["controller"]
    ctrl = PDRelativeController(dt=env.dt, omega_n=float(cc["omega_n"]),
                                zeta=float(cc["zeta"]), kappa=float(env.kappa),
                                a_max=float(env.a_max))
    n_ep = int(cfg["data"]["n_episodes"])
    train_all = collect_controlled_episodes(env, ctrl, n_episodes=n_ep, seed=seed,
                                            max_steps=int(cfg["env"]["max_steps"]))
    train_eps, val_eps = split_episodes(train_all, float(cfg["train"]["val_ratio"]), seed)
    tr = tuple(torch.as_tensor(x) for x in transitions_from_episodes(train_eps))
    va = tuple(torch.as_tensor(x) for x in transitions_from_episodes(val_eps))
    obs_dim = int(train_all[0]["obs"].shape[1])
    model = MLPWorldModel(obs_dim=obs_dim, act_dim=int(env.act_dim),
                          latent_dim=int(cfg["model"]["latent_dim"]),
                          hidden=int(cfg["model"]["hidden"]),
                          discrete_act=bool(env.is_discrete)).to(device)
    hist = train_world_model(model, tr, va, cfg, device, verbose=False)
    print(f"[17] 训练完成：params={count_params(model):,} val_loss={hist['val_total'][-1]:.6f} "
          f"({time.time() - t0:.0f}s)")

    # ---------- 3) 离线曲线 f(h) ----------
    ev_seed = seed + int(cfg["data"]["eval_seed_offset"])
    # ★★ 稳态口径（本实验最关键的一处对齐）
    #   离线曲线的起点被限制在 t ≥ WARMUP（`t0_min`），在线跟踪也用 `warmup` 跳过
    #   同样多的步 ⇒ **两边都在稳态区**。若只改一边，报出来的 gap 全是假的
    #   （第一版就这样：离线只在瞬态区取样 ⇒ f(1) 高估 49%，闭式凭空高 33%）。
    WARMUP = int(cfg["task"]["warmup_steps"])
    ev_env = make_env(cfg["env"]["id"], seed=ev_seed,
                      noise_std=float(cfg["env"]["noise_std"]),
                      max_steps=int(cfg["env"]["max_steps"]))
    eval_eps = collect_controlled_episodes(ev_env, ctrl,
                                           n_episodes=max(30, n_ep // 3), seed=ev_seed,
                                           max_steps=int(cfg["env"]["max_steps"]))
    ev_env.close()
    # ★ 分母也必须用**稳态段**的观测方差 —— 在线只记稳态段，分母口径要跟着走
    allobs = np.concatenate([e["obs"][WARMUP:] for e in eval_eps], axis=0).astype(np.float64)
    var_g = float(allobs.var())
    hs = [int(h) for h in cfg["eval"]["horizons"]]
    hs = [h for h in hs if h <= max(e["length"] for e in eval_eps) - WARMUP - 2]
    curve = closed_loop_error_curve(model, eval_eps, hs, device,
                                    n_samples=int(cfg["eval"]["n_samples"]), seed=seed,
                                    t0_min=WARMUP)
    thr = float(cfg["eval"]["err_threshold"])
    # ★★ 两个分母口径必须分开（X30 已经踩过一次，这里再写一遍）：
    #   `nmse_per_step` = **每个切片自己的方差**做分母 ⇒ 短视界分母小、数值虚高
    #   `mse_per_step / var_g` = **全局 pooled 方差** ⇒ 与在线 run_tracking 同口径
    # 本实验要把离线 f(h) 代进在线误差的闭式里 ⇒ **必须用全局分母版本**。
    # （第一版用了 nmse_per_step，闭式预测比实测高 35% —— 全是分母口径造成的假缺口。）
    f_h_rel = np.asarray([float(np.asarray(curve["nmse_per_step"], dtype=float)[h - 1])
                          for h in hs])
    f_h = np.asarray([float(np.asarray(curve["mse_per_step"], dtype=float)[h - 1])
                      / var_g for h in hs])
    h_err = reliable_horizon(hs, [float(x) for x in f_h_rel], thr,
                             cfg["eval"].get("threshold_mode", "rel"))
    if not np.all(np.isfinite(f_h)) or not np.all(np.isfinite(f_h_rel)):
        raise FloatingPointError("离线曲线含非有限值（R14）")
    print(f"[17] 离线曲线：H*_err={h_err}（切片分母口径，沿用 X2/X14）")
    print(f"[17]   f(1)={f_h[0]:.5f}  f({hs[-1]})={f_h[-1]:.5f}  "
          f"★ 全局分母口径（与在线可比）；切片口径 f_rel(1)={f_h_rel[0]:.5f} "
          f"f_rel({hs[-1]})={f_h_rel[-1]:.5f}")

    # ---------- 4) ★★ 先做凹凸性诊断，把 Jensen 方向从数据推出来 ----------
    #   f 的对数-对数二阶差分：在 log h 网格上算 Δ²(log f)。
    #   凸（Δ²>0）⇒ 突发 worse；凹（Δ²<0）⇒ 突发 better。
    lx = np.log(np.asarray(hs, dtype=float))
    ly = np.log(np.maximum(f_h, 1e-12))
    d1 = np.diff(ly) / np.diff(lx)                 # 局部幂指数（增长率）
    d2 = np.diff(d1)                               # >0 加速（凸） / <0 减速（凹）
    # ★★★ 凹凸性诊断必须**局部化到相关区间**，不许对整条曲线取平均。
    #   本例实测：全区间 mean(Δ²) = **−0.12**（会判成"凹 ⇒ 突发更好"），
    #   但相关区间（h ≤ L_max）内 Δ² 几乎全为正（凸），而**实测 Jensen 项 J 全部 > 0**。
    #   判"凹"与实测直接矛盾 —— 原因：曲线末端进入饱和（幂指数跌到 −0.47），
    #   两个大负值把全区间均值拉翻了，而 age 分布几乎取不到那么大的 h。
    #   ⇒ 用全区间均值判定，就是规范 R2 说的"从图上看出一个趋势就直接解释"。
    h_rel = int(min(max(bursts), hs[-1]))
    mask = np.asarray(hs[1:-1], dtype=float) <= h_rel
    d2_rel = d2[mask] if mask.any() else d2
    conv = ("凸（⇒ Jensen 使突发更糟）" if d2_rel.mean() > 0
            else "凹（⇒ Jensen 使突发**更好**）")
    print(f"[17] ★★ 凹凸性诊断（log-log 二阶差分，**局部化到 h ≤ {h_rel}**）")
    print(f"[17]    相关区间 mean(Δ²)={d2_rel.mean():+.4f} 中位={np.median(d2_rel):+.4f} "
          f"⇒ {conv}")
    print(f"[17]    ⚠ 全区间 mean(Δ²)={d2.mean():+.4f}（末端饱和段把它拉翻了，"
          f"**不得用于判定**）")
    print(f"[17]    局部幂指数 d(log f)/d(log h)：{np.round(d1, 3).tolist()}")

    hs_arr = np.asarray(hs, dtype=float)
    f_arr = np.asarray(f_h, dtype=float)

    def f_at(h: float) -> float:
        """离线曲线在任意 h 处的值（log-log 线性插值；h≤0 ⇒ 0，刚收到真值）。"""
        if h <= 0:
            return 0.0
        return float(np.exp(np.interp(np.log(h), np.log(hs_arr),
                                      np.log(np.maximum(f_arr, 1e-12)))))

    f_grid = np.asarray([f_at(k) for k in range(int(hs[-1]) + 1)], dtype=np.float64)

    # ---------- 5) A 层：纯跟踪（固定 episode，轨迹与模型无关）----------
    n_track = int(ch_cfg["n_track_episodes"])
    track_seed = seed + int(cfg["data"]["eval_seed_offset"])
    track_eps = eval_eps[:n_track] if n_track <= len(eval_eps) else eval_eps
    rows_a = []
    # i.i.d. 基线（X26 用的那个实现）—— S3 的锚点
    base = run_tracking(model, track_eps, device, lossy_schedule(p_loss),
                        seed=track_seed, denom=var_g, label=f"i.i.d.(p={p_loss})",
                        warmup=WARMUP)
    rows_a.append({"kind": "iid", "L": l_iid, "nmse": base.nmse,
                   "age_mean": base.age_tx_mean, "age_std": base.age_std,
                   "age_p95": base.age_p95, "age_hist": base.age_hist,
                   "tx_rate": base.tx_rate, "label": base.label})
    print(f"[17] 基线 i.i.d.(p={p_loss})：E[NMSE]={base.nmse:.5f} "
          f"E[age]={base.age_tx_mean:.3f}（解析 {p_loss / (1 - p_loss):.3f}）")

    for L, ch in chans.items():
        r = run_tracking(model, track_eps, device, ch, seed=track_seed,
                         denom=var_g, label=f"GE(L={L:g})", warmup=WARMUP)
        if not np.isfinite(r.nmse) or not np.isfinite(r.age_tx_mean):
            raise FloatingPointError(f"GE(L={L:g}) 产出非有限值（R14）")
        K = len(f_grid) - 1
        hist_ = {int(k): int(v) for k, v in r.age_hist.items()}
        n_tot = sum(hist_.values())
        p_age_emp = np.zeros_like(f_grid)
        for k, c in hist_.items():
            p_age_emp[min(int(k), K)] += c / n_tot
        # ① 用**实测** age 直方图代入（X26 式子的直接推广，不假设任何分布形状）
        pred = float(np.dot(p_age_emp, f_grid))
        # ② ★ 用**闭式的精确分布**代入（Gilbert 下条件年龄恒为 Geom(β)）
        pmf_th = full_age_pmf(p_loss, ch.beta, K)
        pred_th = float(np.dot(pmf_th, f_grid))
        # ③ ★★ 一阶 / Jensen 二阶分解：E[NMSE] = p̄·[ f(L) + J(L) ]
        f_of_L = f_at(L)                                  # 一阶项：条件年龄均值 = L
        g_L = pred_th / max(p_loss, 1e-12)                # = E_{Geom(β)}[f]
        j_L = g_L - f_of_L                                # Jensen 项（凹凸决定符号）
        # ④ 年龄匹配对照（保留，但必须连同 P(age=0) 一起报 —— 见函数 docstring）
        pmf_g = age_matched_geom_pmf(r.age_tx_mean, K)
        geo = float(np.dot(pmf_g, f_grid))
        # ★ S6：实测**条件**年龄分布必须与 Geom(β) 吻合（验证上面的闭式推导）
        n_pos = sum(c for k, c in hist_.items() if k >= 1)
        # ★ 条件分布定义在 age>0 上 ⇒ k=0 处必须**置 0**，
        #   不能写成 hist_.get(0,0)/n_pos（第一版就这么错，L=1 时 TV 恒为 0.5）
        cond_emp = np.array([0.0] + [hist_.get(k, 0) / max(n_pos, 1)
                                     for k in range(1, K + 1)])
        cond_th = np.array([0.0] + [ch.beta * ((1.0 - ch.beta) ** (k - 1))
                                    for k in range(1, K + 1)])
        # ★ S6 改用 KS 作硬判定（TV 的噪声随突发长度发散，不能当判据 —— 见
        #   `age_goodness` docstring 的第 14 次自我修正）
        # ★ 有效样本量 = **游程数**而非 age 样本数：游程内 age 是 1…n 的确定性
        #   序列，一个游程只有 1 个独立样本（用 n_pos 会把功效高估 √L 倍）。
        n_eff = max(int(n_pos / max(L, 1.0)), 1)
        go = age_goodness(cond_emp, cond_th, n_eff)
        tv, tv_noise, tv_thresh = go["tv"], go["tv_noise"], go["ks_crit"]
        if not go["ok"]:
            raise AssertionError(
                f"★ S6 未通过：L={L:g} 条件年龄分布 KS={go['ks']:.4f} > 临界 "
                f"{go['ks_crit']:.4f}（n_pos={n_pos}；TV={tv:.4f} vs 纯噪声期望 "
                f"{tv_noise:.4f}）—— 「条件年龄恒为几何分布」的闭式推导与仿真不符，"
                f"整个一阶/二阶分解都不可信")
        # ★ S7：纯信道的**游程长度**分布（i.i.d. 样本 ⇒ 这才是严格的硬判定）
        #   在线 age 直方图自相关严重（游程内 age = 1…n 是确定性序列），
        #   只能作参考；Gilbert 链下游程长度 i.i.d. ~ Geom(β)，且仿真极便宜。
        runs = simulate_bad_runs(ch, 200000, np.random.default_rng(seed + 31))
        rl = run_length_goodness(runs, ch.beta, alpha=0.05 / max(len(chans), 1))
        if rl["ok"] is False:
            raise AssertionError(
                f"★ S7 未通过：L={L:g} 游程长度 KS={rl['ks']:.4f} > {rl['ks_crit']:.4f}"
                f" 或均值 {rl['mean_emp']:.3f} vs 闭式 {rl['mean_th']:.3f} "
                f"(z={rl['mean_z']:+.1f})（n_runs={rl['n_runs']}）⇒ 信道实现有误")

        # ★ 附加诊断：在线**按 age 分桶**的实测误差 vs 离线曲线 f(k)。
        #   这是 X26 解析式最直接的检验（不经过任何分布假设）。
        by_age = {int(k): (float(v[0]) / max(float(v[1]), 1e-12)) / var_g
                  for k, v in r.nmse_by_age.items()}
        # ★ S4：闭式分布的均值必须 = 实测 E[age]，但**按有效样本量给容差**。
        #   ★ 长突发时游程数很少（L=16、8 集 ⇒ 只有 ~112 个独立游程），
        #     E[age] 的估计标准差很大 —— 第一版给固定 8% 阈值，把统计噪声
        #     误报成了"尾部截断"的 bug（规范 R13：先确认两边各自收敛）。
        n_runs_est = max(1.0, n_tot * p_loss / max(L, 1.0))
        sigma_age = float(r.age_std / np.sqrt(n_runs_est))
        m_th = float(np.dot(np.arange(len(pmf_th)), pmf_th))
        if abs(m_th - r.age_tx_mean) > max(4.0 * sigma_age, 0.05 * m_th):
            raise AssertionError(
                f"★ S4 未通过：闭式分布均值 {m_th:.4f} 与实测 E[age] "
                f"{r.age_tx_mean:.4f} 差 {abs(m_th - r.age_tx_mean) / max(sigma_age, 1e-12):.1f}σ"
                f"（有效样本≈{n_runs_est:.0f} 游程，σ={sigma_age:.4f}）")
        # ★ S2：经验丢包率 vs p̄，σ 按**有效样本量 = 游程数**估计
        n_runs = max(1.0, n_tot * p_loss / max(L, 1.0))     # 独立 Bad 游程的个数
        sigma = float(np.sqrt(max(p_loss * (1 - p_loss), 1e-12) / n_runs))
        # ★ 丢包率直接从**记账区间**的 age 直方图算：P(age>0) = 丢包率。
        #   不用 tx_rate —— 它依赖 n_tx 与 n_steps 同区间（已修），但这里更直接。
        emp_p = 1.0 - hist_.get(0, 0) / max(n_tot, 1)
        if abs(emp_p - p_loss) > 4.0 * sigma:
            raise AssertionError(
                f"★ S2 未通过：L={L:g} 实测丢包率 {emp_p:.4f} vs p̄={p_loss} "
                f"偏差 {abs(emp_p - p_loss) / max(sigma, 1e-12):.1f}σ"
                f"（有效样本≈{n_runs:.0f} 个游程，σ={sigma:.4f}）")
        rows_a.append({"kind": "ge", "L": float(L), "nmse": r.nmse,
                       "age_mean": r.age_tx_mean, "age_std": r.age_std,
                       "age_p95": r.age_p95, "age_hist": hist_,
                       "tx_rate": r.tx_rate, "label": r.label,
                       "alpha": ch.alpha, "beta": ch.beta, "rho1": ch.rho1,
                       "p_age0": float(pmf_th[0]),
                       "analytic_pred": pred, "analytic_pred_closed": pred_th,
                       "geo_matched": geo, "geo_p_age0": float(pmf_g[0]),
                       "age_matched_gain": r.nmse / max(geo, 1e-12),
                       "first_order_f_of_L": f_of_L, "jensen_J": j_L,
                       "closed_gap": r.nmse / max(pred_th, 1e-12) - 1.0,
                       "pred_gap": r.nmse / max(pred, 1e-12) - 1.0,
                       "cond_tv": tv, "cond_tv_noise": float(tv_noise),
                       "cond_ks": float(go["ks"]), "cond_ks_crit": float(go["ks_crit"]),
                       "n_eff": int(n_eff),
                       "run_len": {k: (float(v) if isinstance(v, (int, float, np.floating))
                                       else v) for k, v in rl.items()},
                       "nmse_by_age_online": by_age,
                       "emp_p_loss": emp_p, "sigma_p": sigma})
        print(f"[17] GE(L={L:>4g}) ρ₁={ch.rho1:+.3f} E[age]={r.age_tx_mean:6.3f}"
              f"(解析 {p_loss * L:5.2f})  E[NMSE]={r.nmse:.5f}")
        print(f"[17]        闭式={pred_th:.5f}({r.nmse / max(pred_th, 1e-12) - 1:+.1%})  "
              f"实测直方图={pred:.5f}({r.nmse / max(pred, 1e-12) - 1:+.1%})  "
              f"KS={go['ks']:.4f}/{go['ks_crit']:.4f}  TV={tv:.4f}(噪声期望 {tv_noise:.4f})")
        print(f"[17]        分解 p̄·[f(L)+J] = {p_loss:.2f}·[{f_of_L:.5f}{j_L:+.5f}] "
              f"⇒ 一阶 f(L) 占 {abs(f_of_L) / max(abs(f_of_L) + abs(j_L), 1e-12):.1%}")
        cmp = [(k, v, f_at(k)) for k, v in sorted(by_age.items())
               if k >= 1 and v > 0 and k <= 32][:6]
        if cmp:
            print("[17]        在线实测 f_online(k) vs 离线 f(k)："
                  + "  ".join(f"k={k}: {v:.4f}/{o:.4f}" for k, v, o in cmp))

    # ★ S3：L = L_iid 的 GE 必须与 i.i.d. 基线统计一致
    ge_iid = [r for r in rows_a if r["kind"] == "ge" and abs(r["L"] - l_iid) < 1e-9]
    s3 = None
    if ge_iid and base.nmse > 1e-12:
        g = ge_iid[0]
        s3 = {"iid_nmse": float(base.nmse), "ge_at_l_iid_nmse": float(g["nmse"]),
              "ratio": float(g["nmse"] / base.nmse)}
        print(f"[17] ★ S3 锚点：L=L_iid={l_iid:.3f} 的 GE 与 i.i.d. 基线 "
              f"E[NMSE] 比值 = {s3['ratio']:.3f}（应 ≈1 ⇒ 两个实现接上了）")
        if not (0.6 < s3["ratio"] < 1.6):
            raise AssertionError(
                f"★ S3 未通过：无记忆点处 GE({g['nmse']:.5f}) 与 i.i.d.({base.nmse:.5f}) "
                f"差 {s3['ratio']:.2f} 倍 —— 两个实现没接上，后面的 burst_gain 不可信")

    # ---------- 6) B 层：任务层（闭环控制，轨迹被模型改变）----------
    tk = cfg["task"]
    online_seed = seed + int(cfg["data"]["online_seed_offset"])
    rows_b = []
    for L, ch in chans.items():
        r = run_closed_loop_control(
            env, model, ctrl, ch, n_episodes=int(tk["n_episodes"]),
            seed=online_seed, max_steps=int(tk["measured_steps"]), device=device,
            estimator="model", var_g=var_g, label=f"GE(L={L:g})",
            tail_frac=float(tk["tail_frac"]), warmup_steps=int(tk["warmup_steps"]))
        if not all(np.isfinite(float(r[k])) for k in
                   ("est_nmse", "escape_rate", "mean_dist_tail", "mean_age")):
            raise FloatingPointError(f"任务层 GE(L={L:g}) 产出非有限值（R14）")
        r["L"] = float(L)
        r["rho1"] = ch.rho1
        rows_b.append(r)
        print(f"[17] 任务层 GE(L={L:>4g})  escape={r['escape_rate']:.3f}  "
              f"dist={r['mean_dist_tail']:6.3f}  est_nmse={r['est_nmse']:.5f}  "
              f"E[age]={r['mean_age']:6.2f}(p95 {r['age_p95']:4.0f})")
    env.close()

    # ---------- 7) 出图 ----------
    fig, axes = plt.subplots(2, 3, figsize=(18.5, 9.4))

    # ① age 分布：i.i.d. vs 突发（★ 直观展示"分布变宽"）
    ax = axes[0, 0]
    kmax_plot = 40
    ax.bar(np.arange(len(base.age_hist)),
           [base.age_hist.get(k, 0) / max(sum(base.age_hist.values()), 1)
            for k in range(len(base.age_hist))],
           color=PALETTE["grey"], alpha=0.7, label=f"i.i.d.(p={p_loss})")
    for L, col in zip(sorted(chans, reverse=True),
                      [PALETTE["blue"], PALETTE["orange"], PALETTE["green"]][:len(chans)]):
        r = next((x for x in rows_a if x["kind"] == "ge" and x["L"] == L), None)
        if r is None:
            continue
        tot = max(sum(r["age_hist"].values()), 1)
        ks = np.arange(kmax_plot + 1)
        ax.step(ks, [r["age_hist"].get(int(k), 0) / tot for k in ks], where="mid",
                color=col, lw=1.6, label=f"GE(L={L:g}, ρ₁={r['rho1']:+.2f})")
    ax.set_xlim(-0.5, kmax_plot); ax.set_yscale("log")
    ax.set_xlabel("传输信息年龄 age (steps)"); ax.set_ylabel("P(age)")
    ax.set_title("① 突发让 age 分布长出重尾（同样平均丢包率）")
    ax.legend(fontsize=7.5)

    # ② ★★ 核心结论：突发 vs i.i.d.（同样平均丢包率）
    ax = axes[0, 1]
    ge_rows = sorted([r for r in rows_a if r["kind"] == "ge"], key=lambda r: r["L"])
    ls = [r["L"] for r in ge_rows]
    raw = [r["nmse"] for r in ge_rows]
    closed = [r["analytic_pred_closed"] for r in ge_rows]
    gains = [r["age_matched_gain"] for r in ge_rows]
    rel = [x / max(base.nmse, 1e-12) for x in raw]
    rel_c = [c / max(base.nmse, 1e-12) for c in closed]
    ax.plot(ls, rel, "o-", color=PALETTE["red"], lw=2, label="实测 E[NMSE] / i.i.d.")
    ax.plot(ls, rel_c, "s:", color=PALETTE["blue"], lw=1.5,
            label="闭式 p̄[f(L)+J] / i.i.d.")
    ax.axhline(1.0, color=PALETTE["grey"], ls=":", label="1.0（= i.i.d.）")
    ax.axvline(l_iid, color=PALETTE["green"], ls="--", alpha=0.8,
               label=f"L_iid={l_iid:.2f}（无记忆=i.i.d.）")
    ax.set_xscale("log"); ax.set_xlabel("平均突发长度 L (steps)")
    ax.set_ylabel("相对 i.i.d. 的跟踪误差倍数")
    ax.set_title("② ★ 突发 vs i.i.d.（同样平均丢包率 p̄）")
    ax.legend(fontsize=7.5)

    # ③ ★★ 一阶 / Jensen 二阶分解：E[NMSE]/p̄ = f(L) + J(L)
    ax = axes[0, 2]
    fo = [r["first_order_f_of_L"] for r in ge_rows]
    jj = [r["jensen_J"] for r in ge_rows]
    ax.plot(ls, fo, "o-", color=PALETTE["blue"], label="一阶项 f(L)（突发拉长条件年龄）")
    ax.plot(ls, jj, "s-", color=PALETTE["orange"], label="Jensen 项 J(L)（凹凸决定符号）")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("平均突发长度 L (steps)"); ax.set_ylabel("NMSE 贡献")
    ax.set_title(f"③ ★ 分解 E[NMSE]/p̄ = f(L) + J(L)（相关区间{conv[:1]}）")
    ax.legend(fontsize=7.5)

    # ④ ★ 闭式恒等式检验：实测 E[NMSE] vs p̄·[f(L)+J(L)]
    ax = axes[1, 0]
    preds = [r["analytic_pred_closed"] for r in ge_rows]
    gaps = [r["closed_gap"] for r in ge_rows]
    ax.plot(preds, raw, "o", color=PALETTE["blue"], label="各 L")
    for r, x, y in zip(ge_rows, preds, raw):
        ax.annotate(f"L={r['L']:g}", (x, y), fontsize=7, xytext=(4, 3),
                    textcoords="offset points")
    lo = min(min(preds), min(raw)) * 0.7
    hi = max(max(preds), max(raw)) * 1.4
    ax.plot([lo, hi], [lo, hi], "k:", lw=1, label="y=x（恒等式成立）")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("闭式预测 p̄·[f(L)+J(L)]"); ax.set_ylabel("实测 E[NMSE]")
    ax.set_title(f"④ ★ 闭式恒等式检验（相对偏差中位 {np.median(gaps):+.1%}）")
    ax.legend(fontsize=7.5)

    # ⑤ 离线曲线 + 凹凸性
    ax = axes[1, 1]
    ax.plot(hs, f_h_rel, "o-", color=PALETTE["blue"], label="f（切片分母，H*_err 口径）")
    ax.plot(hs, f_h, "s--", color=PALETTE["red"], alpha=0.8,
            label="f（全局分母，与在线可比）")
    ax.axhline(thr, color=PALETTE["grey"], ls=":", label=f"阈值 {thr}")
    if h_err is not None:
        ax.axvline(h_err, color=PALETTE["red"], ls="--", alpha=0.8, label=f"H*_err={h_err:g}")
    ax2 = ax.twinx()
    ax2.plot(hs[1:-1], d2, "s--", color=PALETTE["orange"], alpha=0.8,
             label="Δ²(log f)（>0 凸 / <0 凹）")
    ax2.axhline(0, color="k", lw=0.6)
    ax2.set_ylabel("log-log 二阶差分")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("视界 h"); ax.set_ylabel("离线闭环 NMSE")
    ax.set_title(f"⑤ 离线曲线与凹凸性（相关区间 mean Δ²={d2_rel.mean():+.3f} ⇒ {conv[:1]}；"
                 f"末端饱和段 d1 跌到 {d1[-1]:+.2f}）")
    ax.legend(fontsize=7.5, loc="upper left")

    # ⑥ 任务层
    ax = axes[1, 2]
    if rows_b:
        rb = sorted(rows_b, key=lambda r: r["L"])
        ax.plot([r["L"] for r in rb], [r["escape_rate"] for r in rb], "o-",
                color=PALETTE["red"], label="逃逸率（任务失败）")
        ax.set_xscale("log")
        ax.set_xlabel("平均突发长度 L (steps)"); ax.set_ylabel("逃逸率", color=PALETTE["red"])
        ax2 = ax.twinx()
        ax2.plot([r["L"] for r in rb], [r["mean_dist_tail"] for r in rb], "s-",
                 color=PALETTE["blue"], label="稳态跟踪距离")
        ax2.set_ylabel("稳态跟踪距离 (m)", color=PALETTE["blue"])
        ax.axvline(l_iid, color=PALETTE["green"], ls="--", alpha=0.7)
        ax.set_title("⑥ 任务层：突发对任务指标的影响")
        ax.legend(fontsize=7.5, loc="upper left")
    else:
        ax.text(0.5, 0.5, "无数据", ha="center", va="center")

    fig.suptitle(
        f"wmlab · X31 突发信道（Gilbert–Elliott） · env={cfg['env']['id']} "
        f"σ={cfg['env']['noise_std']} · p̄={p_loss} · L_iid={l_iid:.2f} · "
        f"H*_err={h_err} · seed={seed} · 相对 i.i.d. 的误差 "
        f"{min(rel):.2f}×–{max(rel):.2f}×",
        fontsize=10)
    fig.tight_layout()
    png = save_fig(fig, out / f"{args.tag}.png")

    # ---------- 8) 落盘 ----------
    csv_path = out / f"{args.tag}.csv"
    cols = ["kind", "L", "rho1", "alpha", "beta", "nmse", "age_mean", "age_std",
            "age_p95", "tx_rate", "analytic_pred", "geo_matched", "burst_gain",
            "pred_gap", "emp_p_loss"]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in sorted(rows_a, key=lambda r: r["L"]):
            w.writerow([r.get(c, "") if not isinstance(r.get(c), float) else f"{r[c]:.8f}"
                        for c in cols])
    csv_b = out / f"{args.tag}_task.csv"
    cols_b = ["L", "rho1", "escape_rate", "mean_dist", "mean_dist_tail", "est_nmse",
              "mean_age", "age_std", "age_p95", "tx_rate", "mean_ep_len"]
    with open(csv_b, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(cols_b)
        for r in sorted(rows_b, key=lambda r: r["L"]):
            w.writerow([r.get(c, "") if not isinstance(r.get(c), float) else f"{r[c]:.8f}"
                        for c in cols_b])

    # ★ 结论判定（判据在跑之前就写死在 docstring 里，跑完自动对号，不许事后编解释）
    fo_arr = np.asarray(fo, dtype=float)
    jj_arr = np.asarray(jj, dtype=float)
    first_share = float(np.mean(np.abs(fo_arr) /
                                np.maximum(np.abs(fo_arr) + np.abs(jj_arr), 1e-12)))
    gt = [(r["L"], r["nmse"]) for r in ge_rows if r["L"] > l_iid * (1 + 1e-9)]
    lt = [(r["L"], r["nmse"]) for r in ge_rows if r["L"] < l_iid * (1 - 1e-9)]
    ok_gt = all(v > base.nmse for _, v in gt)
    ok_lt = all(v < base.nmse for _, v in lt)
    if ok_gt and (not lt or ok_lt):
        direction = ("★ 方向由 L 相对 L_iid **一阶**决定：L>L_iid（真突发）更糟、"
                     "L<L_iid（通/丢交替）反而更好")
    else:
        direction = ("方向与 (L − L_iid) 的符号**不一致** ⇒ 存在一阶之外的效应，"
                     "需要单独查（先查采样收敛，再谈机制）")
    # ★ L=1 时条件分布是点质量 ⇒ J 恰好为 0（没有 Jensen 效应可言），
    #   不能用 `> 0` 判，否则会被误报成"符号混合"。
    j_sign = ("全部 ≥ 0（凸性，雪上加霜）" if np.all(jj_arr >= -1e-15) else
              "全部 ≤ 0（凹性，部分抵消一阶）" if np.all(jj_arr <= 1e-15) else "符号混合")
    verdict = (f"{direction}。一阶项 f(L) 平均占 {first_share:.0%} "
               f"⇒ 突发的影响{'主要' if first_share > 0.7 else '并不'}是"
               f"「一阶地拉长条件信息年龄」，而非 Jensen 高阶效应；"
               f"Jensen 项 J(L) {j_sign}；"
               f"闭式恒等式 E[NMSE]=p̄[f(L)+J] 的相对偏差中位 {np.median(gaps):+.1%}")

    summary = {
        "config": cfg,
        "device": str(device),
        "n_params": count_params(model),
        "p_loss": p_loss,
        "l_iid": l_iid,
        "h_star_err": h_err,
        "var_global": var_g,
        "offline_curve": {"horizons": hs,
                          "f_global_denom": [float(x) for x in f_h],
                          "f_slice_denom": [float(x) for x in f_h_rel],
                          "loglog_d1": [float(x) for x in d1],
                          "loglog_d2": [float(x) for x in d2],
                          "d2_mean_relevant": float(d2_rel.mean()),
                          "d2_mean_full": float(d2.mean()),
                          "relevant_h_max": h_rel,
                          "convexity": "convex" if d2_rel.mean() > 0 else "concave",
                          "warning": ("全区间 d2_mean 与相关区间符号相反 —— 末端饱和段"
                                      "污染了均值，判定一律以 d2_mean_relevant 为准")},
        "iid_baseline": {"nmse": float(base.nmse), "age_mean": float(base.age_tx_mean),
                         "age_std": float(base.age_std), "tx_rate": float(base.tx_rate)},
        "anchor_check_S3": s3,
        "tracking_layer": rows_a,
        "task_layer": [{k: v for k, v in r.items() if k != "age_hist"} for r in rows_b],
        "relative_to_iid": {"L": [float(x) for x in ls],
                            "ratio": [float(x) for x in rel],
                            "closed_ratio": [float(x) for x in rel_c]},
        "decomposition": {"L": [float(x) for x in ls],
                          "first_order_f_of_L": [float(x) for x in fo],
                          "jensen_J": [float(x) for x in jj],
                          "first_order_share": first_share},
        "age_matched_control": {"L": [float(x) for x in ls], "gain": [float(x) for x in gains],
                                "note": ("仅作对照：该口径会改变 P(age=0)，"
                                         "系统性低估突发的影响，不得当作主结论")},
        "closed_identity_gap_median": float(np.median(gaps)),
        "empirical_hist_gap_median": float(np.median([r["pred_gap"] for r in ge_rows])),
        "verdict": verdict,
        "caveat": ("★ ① P(age=0)=1−p̄ 在固定丢包率下是不变量，任何对照都不得改它；"
                   "② 条件年龄恒为 Geom(β) 是 Gilbert 模型的性质，换成其他信道模型（如"
                   "Markov–Arrival / 真实 trace）该恒等式未必成立；"
                   "③ 任务层的结果仍是 (模型, PD控制器, 信道) 三元组的属性。"),
        "elapsed_sec": time.time() - t0,
    }
    js = out / f"{args.tag}.json"
    js.write_text(json.dumps(jsonable(summary), ensure_ascii=False, indent=2, default=str),
                  encoding="utf-8")
    print(f"[17] ★★★ 结论：{verdict}")
    print(f"[17] 图 -> {png}\n[17] 表 -> {csv_path}\n[17] 表 -> {csv_b}\n[17] json -> {js}")
    print(f"[17] 用时 {summary['elapsed_sec']:.1f}s")
    print("[17] OK")


if __name__ == "__main__":
    main()
