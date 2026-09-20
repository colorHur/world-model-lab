#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""脚本 07 —— X3（种子重复）+ X4（阈值敏感性）：把 H* 从"一个点"变成"一个分布"。

为什么必须有这个脚本
--------------------
到 X27 为止，本仓库报过的 **6 个 H\* 全是单 seed 点值**：

    Pendulum 开环 42.99 / 闭环 39.01；CartPole 开环 12.95 / 闭环 6.96（逐点）
    Pendulum 开环 64.83 / 闭环 51.96；CartPole 闭环 10.11（累积）

**没有一条带方差**。而 H\* 是一个"首次穿越"统计量 —— 它是曲线的**极值泛函**，
比曲线本身对噪声敏感得多。**在没有方差的情况下引用任何一个 H\*，方法论上都不成立。**

本脚本补两块：

- **X3**：seed ∈ {0,1,2} 各走一遍完整流程（采数据 → 训练 → 测曲线）→ **H\* 均值 ± 样本标准差**
- **X4**：θ ∈ {0.01 … 0.20} 扫一遍 → **H\*(θ) 曲线**，并检查「阈值是否改变开环/闭环的排序」
  （总纲 §4.3 明确写出：排序反转是 X4 的**重要发现**，不是失败）

★ 前置：**Part A 的收敛性检验**（这是 R13 的直接应用）
------------------------------------------------
`_sample_windows` 从 episode 里随机截窗口，`n_samples` 就是窗口数。
**若 n_samples 不够，同一模型两次测出的 H\* 都会不同** —— 那么 Part B 里
"种子之间的差"就会混入采样噪声，σ 被系统性高估，结论全废。
所以**先扫 n_samples、确认收敛，再谈种子方差**。这一步不能省。

★ 两种 H\*（都报，因为有差别）
---------------------------
- `h_sparse`：在 config 的**稀疏** horizons 上插值 —— 与 X2/X26/X27 同口径，可比
- `h_dense` ：在**每一步**（稠密 `nmse_per_step`）上插值 —— 无网格误差，更接近真值
  两者的差量化了"horizons 网格稀疏度"带来的误差。horizons 在 100→150→190 处很稀，
  稀疏口径在这一段最多差几十步 —— **这是历史口径的一个已知精度缺陷，本次首次量化**。

★ 删失（censoring）必须显式计数
-----------------------------
θ 很小/很大时 H\* 会顶到量表的边界（H\*=1 或 H\*=None），此时多个种子的 H\* 会
**全部相同** ⇒ σ=0 ⇒ CV=0。**这会被误读成"H\* 极其稳定"**，而它其实是"全都撞墙了"。
本脚本对每个 (θ, 曲线) 记 `n_censored`，任何汇总统计都要连同这个数字一起读。

口径声明（R1 · 三件套）
----------------------
1. **逐点**（pointwise）口径 —— H\* 是"首次超阈"，必须用逐点
2. **评测集** = 每个 seed 自己的 val split（`val_ratio=0.1`，30 条 × 200 步）
3. **归一化分母** = 该步切片的目标方差（与 02 脚本同源，**不是** pooled 全局方差）

运行
----
    python scripts/07_seeds_threshold.py --quick     # 冒烟
    python scripts/07_seeds_threshold.py             # 全量（3 种子重训，约 10 min）
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from wmlab.data import collect_random_episodes, split_episodes, transitions_from_episodes
from wmlab.envs import make_env
from wmlab.eval import reliable_horizon
from wmlab.models import MLPWorldModel
from wmlab.rollout import closed_loop_error_curve, multi_step_error_curve
from wmlab.train import train_world_model
from wmlab.utils import (count_params, describe_device, get_device, load_config,
                         output_dir, set_seed)
from wmlab.utils.plot import PALETTE, apply_style, save_fig

# ---------------------------------------------------------------- 默认扫描网格
#: Part A：窗口采样数扫描点 —— 目的是找出"再翻倍 H* 也不动"的那个量级。
#: ★ 上限给到 4096 是被前两次事故逼出来的：X26 的离线曲线在 n_samples=64 时
#:   造出一个 **5.4 倍的假缺口**，X27 提到 2048 才与在线仿真对齐。
#:   所以这里宁可在 Part A 多花一分钟，也不要带着未收敛的估计去谈"种子方差"。
NS_GRID_FULL = (256, 512, 1024, 2048, 4096)
#: Part B：种子。★ 种子贯穿全流程（采数据 / 模型初始化 / 训练 shuffle / 窗口采样），
#:          所以这里的 σ 是"整套 pipeline 的不确定性"，不是某个环节的。
SEEDS_FULL = (0, 1, 2)
#: Part C：NMSE 阈值扫描点。0.05 是主口径（与 X2 可比），其余用于看弹性
THRESHOLDS_FULL = (0.01, 0.02, 0.05, 0.10, 0.20)
#: 主口径阈值（单独拎出来，用于收敛性判定与 headline 数字）
THR_MAIN = 0.05
#: 收敛判据：相邻两个 n_samples 之间 H* 的相对变化小于此值视为收敛
CONVERGE_TOL = 0.02


# ---------------------------------------------------------------- 工具
def _pointwise(curve) -> list[float]:
    """把稠密的逐点 **NMSE** 按稀疏 horizons 取出（索引 i ↔ 第 i+1 步）。

    ★ 单位必须与阈值同源：θ 是 NMSE 口径，所以这里只能取 `nmse_per_step`。
    """
    v = np.asarray(curve["nmse_per_step"], dtype=float)
    hs = list(curve["horizons"])
    return [float(v[h - 1]) for h in hs]


def h_star_dense(curve, thr: float, mode: str = "rel") -> float | None:
    """在**每一步**网格上求 H\*（不走 config 的稀疏 horizons）。

    `nmse_per_step` 是稠密数组，第 i 项就是第 i+1 步自身的 NMSE，
    直接在上面找首次穿越即可 —— 没有网格稀疏带来的插值误差。
    """
    arr = np.asarray(curve["nmse_per_step"], dtype=float)
    hs = list(range(1, len(arr) + 1))
    return reliable_horizon(hs, [float(x) for x in arr], thr, mode)


def _both_h_star(open_curve, closed_curve, thr: float, mode: str = "rel") -> dict:
    """对一个 (开环, 闭环) 曲线对，算出 4 个 H\*：稀疏×2 + 稠密×2。"""
    return {
        "open_sparse": reliable_horizon(open_curve["horizons"], _pointwise(open_curve),
                                        thr, mode),
        "closed_sparse": reliable_horizon(closed_curve["horizons"], _pointwise(closed_curve),
                                          thr, mode),
        "open_dense": h_star_dense(open_curve, thr, mode),
        "closed_dense": h_star_dense(closed_curve, thr, mode),
    }


def _train_one(seed: int, cfg: dict, device: torch.device, n_eps: int | None = None,
               epochs: int | None = None):
    """走一遍完整流程：采数据 → 切分 → 建模型 → 训练。返回 (model, train_eps, val_eps, hist)。"""
    c = json.loads(json.dumps(cfg))          # 深拷贝，避免污染上层 cfg
    c["seed"] = seed
    set_seed(seed)
    env = make_env(c["env"]["id"], seed=seed)
    episodes = collect_random_episodes(
        env, n_episodes=int(n_eps or c["data"]["n_episodes"]),
        seed=seed, max_steps=c["env"].get("max_steps"))
    env.close()
    train_eps, val_eps = split_episodes(episodes, float(c["train"]["val_ratio"]), seed)

    mcfg = c["model"]
    model = MLPWorldModel(
        obs_dim=env.obs_dim, act_dim=env.act_dim,
        latent_dim=int(mcfg["latent_dim"]), hidden=int(mcfg["hidden"]),
        discrete_act=bool(mcfg["discrete_act"]) and env.is_discrete,
    ).to(device)

    tr = tuple(torch.as_tensor(x) for x in transitions_from_episodes(train_eps))
    va = tuple(torch.as_tensor(x) for x in transitions_from_episodes(val_eps))
    if epochs is not None:
        c["train"]["epochs"] = int(epochs)
    hist = train_world_model(model, tr, va, c, device, verbose=False)
    return model, train_eps, val_eps, hist


def _curves(model, val_eps, horizons, device, n_samples: int, seed: int) -> tuple[dict, dict]:
    """同一个 n_samples 下算出开环/闭环两条曲线。"""
    co = multi_step_error_curve(model, val_eps, horizons, device,
                                n_samples=n_samples, seed=seed)
    cc = closed_loop_error_curve(model, val_eps, horizons, device,
                                 n_samples=n_samples, seed=seed)
    return co, cc


def _mean_std(xs: list[float]) -> tuple[float, float, int]:
    """对含 None / NaN 的样本做统计。返回 (mean, sample_std(ddof=1), 有效样本数)。"""
    v = [float(x) for x in xs if x is not None and not math.isnan(float(x))]
    if not v:
        return float("nan"), float("nan"), 0
    if len(v) == 1:
        return v[0], 0.0, 1
    return statistics.fmean(v), statistics.stdev(v), len(v)


def _pearson(xs: list[float], ys: list[float]) -> float:
    """皮尔逊相关。样本数 < 2 或某一维方差为 0 时返回 nan（不是 0 —— 那是"无关"的意思）。"""
    if len(xs) < 2:
        return float("nan")
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    num = sum(a * b for a, b in zip(dx, dy))
    den = math.sqrt(sum(a * a for a in dx) * sum(b * b for b in dy))
    return num / den if den > 0 else float("nan")


def _rank(v: list[float]) -> list[float]:
    """转成秩（并列取平均秩），用于 Spearman。"""
    order = sorted(range(len(v)), key=lambda i: v[i])
    out = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            out[order[k]] = avg
        i = j + 1
    return out


def _is_censored(h) -> bool:
    """是否撞到量表边界 —— `None`（全程未超阈）或落到第 1 步（第一步就超阈）。

    ★ 必须计数：删失的样本会**全部取同一个边界值** ⇒ σ=0 ⇒ CV=0，
    那会被误读成"H\* 极其稳定"。CV 一定要连同这个数字一起看。
    """
    if h is None:
        return True
    v = float(h)
    return math.isnan(v) or v <= 1.0 + 1e-9


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pendulum.yaml")
    ap.add_argument("--tag", default="07_seeds_threshold")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--n-episodes", type=int, default=None)
    ap.add_argument("--quick", action="store_true",
                    help="冒烟：1 个种子、少量 epoch/集数、3 个采样数、3 个阈值")
    args = ap.parse_args()

    cfg = load_config(args.config)
    apply_style()
    device = get_device(args.device or cfg.get("device"))
    out = output_dir(cfg)

    seed_list = list(args.seeds) if args.seeds else list(SEEDS_FULL)
    thr_list = list(THRESHOLDS_FULL)
    ns_grid = list(NS_GRID_FULL)
    horizons = list(cfg["eval"]["horizons"])
    mode = cfg["eval"].get("threshold_mode", "rel")
    if THR_MAIN not in thr_list:
        thr_list = sorted(set(thr_list) | {THR_MAIN})

    if args.quick:
        # ★ 保留 2 个种子（而非 1 个）：std / 相关性 / 6 子图这几条**多样本路径**
        #   在小样本下才最容易崩（stdev 要求 n≥2），冒烟必须覆盖到。
        seed_list = seed_list[:2] if len(seed_list) >= 2 else seed_list[:1]
        ns_grid = [128, 256, 512]
        thr_list = [0.02, 0.05, 0.10]

    print(f"[07] env={cfg['env']['id']}  seeds={seed_list}  device={describe_device(device)}")
    print(f"[07] horizons={horizons}")
    t_all = time.time()

    # ================================================================
    # Part A · 收敛性：窗口采样数 n_samples 够不够？
    #   ★ R13 的直接应用：不先确认这条，Part B 的"种子方差"就是假的。
    #   只用 seed_list[0] 这一个模型 —— 收敛性是关于**曲线估计**的，与哪个模型无关。
    # ================================================================
    print("\n[07] ============ Part A · 曲线估计的采样收敛性 ============")
    t0 = time.time()
    mA, _, valA, histA = _train_one(seed_list[0], cfg, device,
                                    n_eps=60 if args.quick else args.n_episodes,
                                    epochs=15 if args.quick else args.epochs)
    print(f"[07] Part A 模型训练完成（seed={seed_list[0]}），用时 {time.time()-t0:.1f}s，"
          f"val_loss={histA['val_total'][-1]:.6f}，n_val={len(valA)}")

    conv_rows = []
    for ns in ns_grid:
        co, cc = _curves(mA, valA, horizons, device, ns, seed_list[0])
        hs_d = _both_h_star(co, cc, THR_MAIN, mode)
        row = dict(n_samples=ns, **{k: v for k, v in hs_d.items()})
        conv_rows.append(row)
        print(f"[07]   n_samples={ns:5d}  H*_open 稀疏={hs_d['open_sparse']} "
              f"稠密={hs_d['open_dense']}   H*_closed 稀疏={hs_d['closed_sparse']} "
              f"稠密={hs_d['closed_dense']}")

    # 收敛判定：最大的两个采样数之间，H* 相对变化是否 < tol
    last2 = conv_rows[-2:]
    rel = {}
    for key in ("open_sparse", "closed_sparse", "open_dense", "closed_dense"):
        a, b = last2[0][key], last2[1][key]
        if a is None or b is None or (math.isnan(float(a)) if a is not None else True):
            rel[key] = float("nan")
            continue
        a, b = float(a), float(b)
        rel[key] = abs(b - a) / max(abs(a), 1e-9)
    worst_rel = max((v for v in rel.values() if not math.isnan(v)), default=float("nan"))
    ns_chosen = ns_grid[-1]
    converged = bool(not math.isnan(worst_rel) and worst_rel < CONVERGE_TOL)
    print(f"[07] ★ 最后两档之间的最大相对变化 = {worst_rel:.4f} "
          f"(判据 <{CONVERGE_TOL})  ⇒  "
          f"{'[OK] 收敛' if converged else '[WARN] 尚未收敛'}")
    if not converged:
        print(f"[07] [WARN] 未收敛：仍然使用最大档 n_samples={ns_chosen}，"
              f"但 Part B 的 σ 里**包含**这部分采样噪声 —— 结论要带上这个限定。")
    print(f"[07] 采用 n_samples = {ns_chosen} 进入 Part B / C")

    # ================================================================
    # Part B · X3 种子重复
    # ================================================================
    print("\n[07] ============ Part B · X3 种子重复 ============")
    per_seed = []
    for s in seed_list:
        t0 = time.time()
        if s == seed_list[0] and not args.quick:
            model, _, val_eps, hist = mA, None, valA, histA     # 复用 Part A 的模型
            print(f"[07] seed={s}: 复用 Part A 的模型")
        else:
            model, _, val_eps, hist = _train_one(
                s, cfg, device,
                n_eps=60 if args.quick else args.n_episodes,
                epochs=15 if args.quick else args.epochs)
            print(f"[07] seed={s}: 训练完成 {time.time()-t0:.1f}s, "
                  f"val_loss={hist['val_total'][-1]:.6f}")
        co, cc = _curves(model, val_eps, horizons, device, ns_chosen, s)
        open_pt = np.asarray(co["nmse_per_step"], dtype=float)
        closed_pt = np.asarray(cc["nmse_per_step"], dtype=float)
        per_seed.append(dict(
            seed=s, val_loss=float(hist["val_total"][-1]), n_val=len(val_eps),
            open_per_step=open_pt.tolist(), closed_per_step=closed_pt.tolist(),
            h_by_thr={str(t): _both_h_star(co, cc, float(t), mode) for t in thr_list},
        ))
        print(f"[07]   seed={s} done in {time.time()-t0:.1f}s")

    # ---- X3 的 headline：主口径 θ=0.05 下的均值 ± 标准差 ----
    print(f"\n[07] ---- X3 结果（θ={THR_MAIN}，逐点口径）----")
    head = {}
    for kind in ("open", "closed"):
        for form in ("sparse", "dense"):
            key = f"{kind}_{form}"
            vals = [d["h_by_thr"][str(THR_MAIN)][key] for d in per_seed]
            mu, sd, n = _mean_std(vals)
            cen = sum(1 for v in vals if _is_censored(v))
            head[key] = dict(values=vals, mean=mu, std=sd, n=n, n_censored=cen,
                             cv=(sd / mu) if (mu and not math.isnan(mu) and mu != 0) else float("nan"))
            print(f"[07]   H*_{kind:6s}({form:6s}) = {vals}  →  mean={mu:.2f}  "
                  f"std={sd:.2f}  CV={head[key]['cv']:.3f}  删失={cen}/{len(vals)}")

    # ================================================================
    # Part C · X4 阈值敏感性
    # ================================================================
    print("\n[07] ============ Part C · X4 阈值敏感性 ============")
    thr_rows = []
    for t in thr_list:
        row = dict(threshold=float(t))
        for kind in ("open", "closed"):
            for form in ("sparse", "dense"):
                key = f"{kind}_{form}"
                vals = [d["h_by_thr"][str(t)][key] for d in per_seed]
                mu, sd, n = _mean_std(vals)
                cen = sum(1 for v in vals if _is_censored(v))
                row[f"{key}_mean"] = mu
                row[f"{key}_std"] = sd
                row[f"{key}_cv"] = (sd / mu) if (mu and not math.isnan(mu) and mu != 0) else float("nan")
                row[f"{key}_n_censored"] = cen
        thr_rows.append(row)
        print(f"[07]   θ={t:<5}  H*_open  = {row['open_sparse_mean']} ± {row['open_sparse_std']}"
              f"  (CV={row['open_sparse_cv'] if not math.isnan(row['open_sparse_cv']) else 'nan'}, "
              f"删失 {row['open_sparse_n_censored']})")
        print(f"[07]           H*_closed= {row['closed_sparse_mean']} ± {row['closed_sparse_std']}"
              f"  (CV={row['closed_sparse_cv'] if not math.isnan(row['closed_sparse_cv']) else 'nan'}, "
              f"删失 {row['closed_sparse_n_censored']})")

    # ---- ★ 排序反转检查（总纲 §4.3 点名的"重要发现"）----
    flips = []
    signs = []
    for row in thr_rows:
        a, b = row["open_sparse_mean"], row["closed_sparse_mean"]
        if a is None or b is None or math.isnan(a) or math.isnan(b):
            signs.append(None)
            continue
        signs.append(1 if a > b else (-1 if a < b else 0))
    valid_signs = [s for s in signs if s is not None]
    for i in range(1, len(valid_signs)):
        if valid_signs[i] != valid_signs[i - 1]:
            flips.append((thr_list[i - 1], thr_list[i]))
    print(f"[07] ★ 开环/闭环 H* 的大小序：{[('open>closed' if s==1 else ('open<closed' if s==-1 else '=')) for s in signs]}")
    print(f"[07] ★ 排序反转点：{flips if flips else '无 —— 开环/闭环的序在所有 θ 下一致'}")

    # ---- ★ 幂律弹性：H*(θ) ≈ C·θ^(-α) ⇒ log H* = log C − α·log θ ----
    elastic = {}
    for kind in ("open", "closed"):
        xs, ys = [], []
        for row in thr_rows:
            h = row[f"{kind}_sparse_mean"]
            if h is None or math.isnan(h) or h <= 0:
                continue
            if row[f"{kind}_sparse_n_censored"] > 0:
                continue          # 删失点不参与拟合（它们贴着量表边界，不是幂律段）
            xs.append(math.log(float(row["threshold"])))
            ys.append(math.log(float(h)))
        if len(xs) >= 2:
            slope, intercept = np.polyfit(xs, ys, 1)
            pred = np.polyval([slope, intercept], xs)
            ss_res = float(np.sum((np.asarray(ys) - pred) ** 2))
            ss_tot = float(np.sum((np.asarray(ys) - np.mean(ys)) ** 2))
            r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
            elastic[kind] = dict(alpha=float(-slope), C=float(math.exp(intercept)),
                                 r2=float(r2), n_points=len(xs))
            print(f"[07] ★ H*_{kind}(θ) ≈ {math.exp(intercept):.2f}·θ^({-slope:.3f})   "
                  f"R²={r2:.4f}  （用 {len(xs)} 个未删失点拟合）")
        else:
            elastic[kind] = dict(alpha=float("nan"), C=float("nan"), r2=float("nan"),
                                 n_points=len(xs))
            print(f"[07] [WARN] {kind}: 未删失点不足 2 个（{len(xs)}），无法拟合幂律")

    # ---- ★★ 单步训练损失能不能预测长视界可靠性？ ----
    #  这是 X3 里比"H* 有方差"本身更重要的一个发现：
    #  如果 val_loss 与 H* 无关，说明**长视界可靠性是一个必须单独测量的量**，
    #  不能从训练曲线推断 —— 这恰好是本课题存在的理由。
    print("\n[07] ---- ★★ 单步训练损失能否预测长视界可靠性？ ----")
    corr = {}
    for kind in ("open", "closed"):
        for form in ("sparse", "dense"):
            key = f"{kind}_{form}"
            pairs = [(d["val_loss"], d["h_by_thr"][str(THR_MAIN)][key])
                     for d in per_seed
                     if d["h_by_thr"][str(THR_MAIN)][key] is not None
                     and not math.isnan(float(d["h_by_thr"][str(THR_MAIN)][key]))]
            if len(pairs) < 2:
                corr[key] = dict(pearson=float("nan"), spearman=float("nan"), n=len(pairs))
                continue
            xs = [float(p[0]) for p in pairs]
            ys = [float(p[1]) for p in pairs]
            r, rho = _pearson(xs, ys), _pearson(_rank(xs), _rank(ys))
            corr[key] = dict(pearson=r, spearman=rho, n=len(pairs),
                             val_loss=xs, h_star=ys)
            print(f"[07]   H*_{kind}({form}): Pearson r={r:+.3f}  Spearman ρ={rho:+.3f}  "
                  f"(n={len(pairs)})")
            if len(pairs) < 3:
                print(f"[07]     [WARN] n={len(pairs)} < 3 —— 两个点的散点必然落在一条直线上，"
                      f"|r| 恒等于 1，此值无参考价值，不要引用。")
    r_main = corr["open_sparse"]["pearson"]
    print(f"[07] ★ 判读：n 很小（n={corr['open_sparse']['n']}），相关系数本身极不可靠，"
          f"只作**定性**观察。")
    print(f"[07]   若 |r| 明显小于 1 且符号不稳 ⇒ **val_loss 不能预报 H***，"
          f"长视界可靠性必须单独测。")

    # ---- 稀疏 vs 稠密的网格误差 ----
    grid_err = []
    for d in per_seed:
        for t in thr_list:
            hh = d["h_by_thr"][str(t)]
            for kind in ("open", "closed"):
                a, b = hh[f"{kind}_sparse"], hh[f"{kind}_dense"]
                if a is None or b is None or math.isnan(float(a)) or math.isnan(float(b)):
                    continue
                grid_err.append(abs(float(a) - float(b)))
    grid_max = max(grid_err) if grid_err else float("nan")
    print(f"[07] ★ 稀疏网格 vs 稠密每步：最大绝对差 = {grid_max} 步 "
          f"（horizons 在 150→190 段很稀，这是历史口径的已知精度缺陷）")

    print(f"[07] 总用时 {time.time()-t_all:.1f}s")

    # ================================================================
    # 出图
    # ================================================================
    fig, axes = plt.subplots(2, 3, figsize=(17.5, 9.2))

    # ① 三种子的逐点 NMSE 曲线（±1σ 阴影）＋ 三个阈值线
    ax = axes[0, 0]
    steps = np.arange(1, horizons[-1] + 1, dtype=float)
    for kind, color, name in (("open", PALETTE["blue"], "open-loop"),
                              ("closed", PALETTE["red"], "closed-loop")):
        M = np.asarray([d[f"{kind}_per_step"] for d in per_seed], dtype=float)
        mu = M.mean(axis=0)
        sd = M.std(axis=0, ddof=1) if len(per_seed) > 1 else np.zeros_like(mu)
        ax.plot(steps, mu, "-", color=color, label=f"{name} mean (n={len(per_seed)})")
        ax.fill_between(steps, np.maximum(mu - sd, 1e-12), mu + sd,
                        color=color, alpha=0.18)
    for t in (THR_MAIN,):
        ax.axhline(t, color=PALETTE["grey"], linestyle=":", linewidth=1.2,
                   label=f"θ = {t}")
    ax.set_yscale("log")
    ax.set_xlabel("Rollout step (per-step grid)")
    ax.set_ylabel("NMSE at step h (pointwise, log)")
    ax.set_title("① Per-step NMSE ± 1σ across seeds")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.25)

    # ② H*(θ) —— X4 的核心图
    ax = axes[0, 1]
    for kind, color, name in (("open", PALETTE["blue"], "open-loop"),
                              ("closed", PALETTE["red"], "closed-loop")):
        mu = np.asarray([r[f"{kind}_sparse_mean"] for r in thr_rows], dtype=float)
        sd = np.asarray([r[f"{kind}_sparse_std"] for r in thr_rows], dtype=float)
        ax.errorbar([float(r["threshold"]) for r in thr_rows], mu, yerr=sd,
                    marker="o" if kind == "open" else "s", color=color,
                    capsize=3, label=f"H*_{name} ± 1σ")
    ax.set_xscale("log")
    ax.set_xlabel("NMSE threshold θ (log)")
    ax.set_ylabel("H* (steps)")
    ax.set_title(f"② X4 · H*(θ)   α_open={elastic['open']['alpha']:.2f}  "
                 f"α_closed={elastic['closed']['alpha']:.2f}")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.25, which="both")

    # ③ 收敛性证据
    ax = axes[0, 2]
    for kind, color, name in (("open", PALETTE["blue"], "open-loop"),
                              ("closed", PALETTE["red"], "closed-loop")):
        ys = [r[f"{kind}_sparse"] for r in conv_rows]
        ys = [(float(y) if y is not None else float("nan")) for y in ys]
        ax.plot([r["n_samples"] for r in conv_rows], ys,
                "o-" if kind == "open" else "s--", color=color, label=f"H*_{name}")
    ax.set_xscale("log")
    ax.set_xlabel("n_samples (rollout windows, log)")
    ax.set_ylabel(f"H* @ θ={THR_MAIN}")
    # ★ 标题里不放 emoji / ⚠ / ✱ —— Microsoft YaHei 缺这些字形，会渲染成方块（tofu）
    ax.set_title("③ Part A · sampling convergence  "
                 f"[{'converged' if converged else 'NOT converged'}]")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.25, which="both")

    # ④ CV(θ) —— 相对不确定性随阈值怎么变
    ax = axes[1, 0]
    for kind, color, name in (("open", PALETTE["blue"], "open-loop"),
                              ("closed", PALETTE["red"], "closed-loop")):
        cv = np.asarray([r[f"{kind}_sparse_cv"] for r in thr_rows], dtype=float)
        cen = [r[f"{kind}_sparse_n_censored"] for r in thr_rows]
        xs = [float(r["threshold"]) for r in thr_rows]
        ax.plot(xs, cv, "o-" if kind == "open" else "s--", color=color,
                label=f"CV {name}")
        for x, c, n in zip(xs, cv, cen):
            if n and not math.isnan(c):
                ax.annotate(f"*{n}", xy=(x, c), xytext=(0, 6),
                            textcoords="offset points", fontsize=7, color=color)
    ax.set_xscale("log")
    ax.set_xlabel("NMSE threshold θ (log)")
    ax.set_ylabel("CV = σ/μ  (seed-to-seed)")
    ax.set_title("④ X3 · seed-to-seed variability  [* = censored seeds]")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.25, which="both")

    # ⑤ ★★ 单步训练损失 vs 长视界可靠性 —— 本轮最重要的散点图
    ax = axes[1, 2]
    for kind, color, name, mk in (("open", PALETTE["blue"], "open", "o"),
                                  ("closed", PALETTE["red"], "closed", "s")):
        c = corr.get(f"{kind}_sparse", {})
        if c.get("n", 0) < 2:
            continue
        ax.scatter(c["val_loss"], c["h_star"], color=color, marker=mk, s=55,
                   zorder=3, label=f"{name} (r={c['pearson']:+.2f})")
        for v, h, s in zip(c["val_loss"], c["h_star"], [d["seed"] for d in per_seed]):
            ax.annotate(f"s{s}", xy=(v, h), xytext=(5, 4),
                        textcoords="offset points", fontsize=7.5, color=color)
    ax.set_xlabel("final val loss (single-step prediction)")
    ax.set_ylabel(f"H* @ θ={THR_MAIN} (steps)")
    ax.set_title("⑥ single-step loss vs horizon reliability  ★key")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.25)

    # ⑥ 稀疏 vs 稠密网格的插值误差 vs θ
    ax = axes[1, 1]
    for kind, color, name in (("open", PALETTE["blue"], "open-loop"),
                              ("closed", PALETTE["red"], "closed-loop")):
        xs, ys = [], []
        for t in thr_list:
            ds = [abs(float(d["h_by_thr"][str(t)][f"{kind}_sparse"])
                      - float(d["h_by_thr"][str(t)][f"{kind}_dense"]))
                  for d in per_seed
                  if d["h_by_thr"][str(t)][f"{kind}_sparse"] is not None
                  and d["h_by_thr"][str(t)][f"{kind}_dense"] is not None]
            if ds:
                xs.append(float(t))
                ys.append(statistics.fmean(ds))
        if xs:
            ax.plot(xs, ys, "o-" if kind == "open" else "s--", color=color,
                    label=f"{name} mean |sparse-dense|")
    ax.set_xscale("log")
    ax.set_xlabel("NMSE threshold θ (log)")
    ax.set_ylabel("|H*_sparse − H*_dense|  (steps)")
    ax.set_title("⑤ grid error of historical sparse horizons")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.25, which="both")

    fig.suptitle(
        f"wmlab · X3 seeds × X4 thresholds · {cfg['env']['id']} · "
        f"seeds={seed_list} · n_samples={ns_chosen} · device={device.type}",
        fontsize=10)
    fig.tight_layout()
    png = save_fig(fig, out / f"{args.tag}.png")
    plt.close(fig)

    # ================================================================
    # 落盘
    # ================================================================
    csv_path = out / f"{args.tag}.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        cols = ["seed", "threshold"]
        cols += [f"{k}_{t}" for k in ("open", "closed") for t in ("sparse", "dense")]
        w.writerow(cols)
        for d in per_seed:
            for t in thr_list:
                hh = d["h_by_thr"][str(t)]
                w.writerow([d["seed"], t] +
                           [hh[f"{k}_{tt}"] for k in ("open", "closed")
                            for tt in ("sparse", "dense")])

    summary = {
        "meta": {
            "experiment": "X3 (seed repetition) + X4 (threshold sensitivity)",
            "env": cfg["env"]["id"],
            "seeds": seed_list,
            "horizons": horizons,
            # ★ R1 口径三件套
            "nmse_calibration": "per_step (pointwise)",
            "eval_set": "每个 seed 自己的 val split（val_ratio=0.1）",
            "normalization_denominator": "该步切片的目标方差（与 02 脚本同源，非 pooled）",
            "threshold_mode": mode,
        },
        "part_a_convergence": {
            "ns_grid": ns_grid,
            "rows": conv_rows,
            "worst_rel_change_last_two": worst_rel,
            "tolerance": CONVERGE_TOL,
            "converged": converged,
            "n_samples_chosen": ns_chosen,
            "caveat": ("未收敛：Part B 的 σ 中包含采样噪声，不可当作纯种子方差"
                       if not converged else "已收敛"),
        },
        "x3_headline": head,
        "x3_val_loss_vs_h_star": corr,
        "x4_threshold_sweep": thr_rows,
        "x4_power_law": elastic,
        "x4_order_flip": {"signs": signs, "flip_points": [list(x) for x in flips]},
        "grid_error_sparse_vs_dense_max_steps": grid_max,
        "per_seed_val_loss": [dict(seed=d["seed"], val_loss=d["val_loss"], n_val=d["n_val"])
                              for d in per_seed],
        # 曲线本身也存下来（开环/闭环 × 每步），供后续复算任意阈值下的 H*
        "per_step_curves": {
            str(d["seed"]): {"open": d["open_per_step"], "closed": d["closed_per_step"]}
            for d in per_seed
        },
    }
    js = out / f"{args.tag}.json"
    js.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[07] 图 -> {png}")
    print(f"[07] csv -> {csv_path}")
    print(f"[07] json -> {js}")
    print("[07] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
