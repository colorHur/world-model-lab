#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""脚本 11 —— **X11：第三种曲线 —— 把「纠错频率」从混杂变量变成可扫描的自变量**。

★ 先修正总纲 X11 的定义（本脚本的第一个结论，写在最前面）
------------------------------------------------------------
总纲 §5.2 把第三种曲线描述为「潜空间闭环：在潜空间里逐步喂回」。
**这个定义与现存的开环曲线重合，没有独立意义** —— 因为
`rollout_latent()` 本来就是"在潜空间里逐步喂回上一步的预测潜状态"。

把两条现有曲线的递推式写出来，真正的差异才浮出来：

| 曲线 | 递推式 | 闭环在哪 |
|---|---|---|
| 开环 `multi_step_error_curve`  | `z_{t+1} = T(z_t, a_t)`              | 潜空间（无纠错） |
| 闭环 `closed_loop_error_curve` | `z_{t+1} = R(T(z_t, a_t))`           | 观测空间（每步往返） |
|                                | 其中 **R = encode ∘ decode**          | |

⇒ **两条曲线的唯一差异 = 是否每步施加往返映射 R。**
（本脚本用 assert 逐位验证这一点，不靠推理。）

于是"第三种曲线"的正确形态不是再画一条，而是**把 R 的施加频率变成自变量**：

    z_{t+1} = T(z_t, a_t)，每 **K** 步施加一次 R

- `K = 1`   ⇒ 每步往返   = 现有闭环（端点）
- `K = ∞`   ⇒ 从不往返   = 现有开环（端点）
- `K = 2,3,5,10,20` ⇒ **插值族**（这才是新东西）

它回答的问题是 09-18 那轮留下的悬案：
「两条曲线在 h≈100–150 处**交叉**」当时被记为"归因仍属推测"。
**交叉点就是"每步纠错的边际收益由负转正"的那一步** —— 用 K 族可以直接把它扫出来。

★ 可证伪的预测（先写下来，再看数据）
------------------------------------
R 有两个相反的效应：
- **代价**：`‖R(z) − z‖` 的位移，往轨迹里注入扰动 ⇒ 短视界有害
- **收益**：若 R 把漂出训练分布的 z 拉回来（`d_M` 下降）⇒ 长视界有益

预测：**交叉点 ≈ 开环轨迹的 Mahalanobis 距离首次超出训练分布的那一步。**
若实测交叉点与它不吻合 ⇒ 预测被推翻，照实记录。

运行
----
    python scripts/11_latent_roundtrip.py --config configs/pendulum.yaml
    python scripts/11_latent_roundtrip.py --quick          # 冒烟
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

from wmlab.data import collect_random_episodes, transitions_from_episodes
from wmlab.envs import make_env
from wmlab.eval import reliable_horizon
from wmlab.eval.metrics import per_step_nmse
from wmlab.models import MLPWorldModel
from wmlab.rollout import closed_loop_error_curve, multi_step_error_curve
from wmlab.train import train_world_model
from wmlab.utils import describe_device, get_device, load_config, output_dir, set_seed
from wmlab.utils.plot import apply_style, save_fig

# K = 每 K 步施加一次往返映射 R。None 表示 K = ∞（从不往返 = 开环）
K_GRID = [1, 2, 3, 5, 10, 20, None]


def k_label(k):
    return "inf(open)" if k is None else f"K={k}"


@torch.no_grad()
def rollout_k(model, obs0: torch.Tensor, acts: torch.Tensor, h_max: int,
              k: int | None, return_z: bool = False):
    """按"每 K 步往返一次"的递推做潜空间 rollout。

    Args:
        k: 每 k 步施加一次 R = encode∘decode；None = 从不施加（纯开环）
    Returns:
        (obs_hat, zs) 预测观测序列 (B,H,obs_dim) 与潜状态序列 (B,H,latent_dim)
    """
    z = model.encode(obs0)
    zs = []
    for t in range(h_max):
        # ★ 往返必须在**转移之前**施加 —— 顺序错了就不是闭环。
        #   闭环的语义是"用上一步的**预测观测**重新编码后再走一步"，即
        #       obs_hat -> encode(obs_hat) -> T(...) ，对应递推 z_t = T(R(z_{t-1}), a)
        #   若写成 R(T(z_{t-1}), a)（先转移后往返），得到的是另一条曲线：
        #   实测与闭环相差 1.6e-2（在 Pendulum 20 epoch 上），不是数值误差。
        #   ⇒ 这个顺序由 assert 端点一致性守着，改错会立刻失败。
        if k is not None and t > 0 and t % k == 0:
            z = model.encode(model.decode(z))      # ← 往返
        z = model.next_latent(z, acts[:, t])
        zs.append(z)
    zs = torch.stack(zs, dim=1)
    B, H, L = zs.shape
    obs_hat = model.decode(zs.reshape(B * H, L)).reshape(B, H, -1)
    return (obs_hat, zs) if return_z else (obs_hat, None)


def sample_windows(episodes, horizon, n_samples, rng):
    obs0, acts, targets = [], [], []
    valid = [i for i, ep in enumerate(episodes) if len(ep["obs"]) > horizon + 1]
    if not valid:
        raise ValueError(f"没有足够长的 episode 支持 horizon={horizon}")
    for _ in range(n_samples):
        ei = valid[rng.integers(len(valid))]
        ep = episodes[ei]
        T = len(ep["obs"])
        t0 = rng.integers(0, T - horizon - 1)
        obs0.append(ep["obs"][t0])
        acts.append(ep["act"][t0:t0 + horizon])
        targets.append(ep["obs"][t0 + 1:t0 + 1 + horizon])
    return (np.asarray(obs0, dtype=np.float32), np.asarray(acts),
            np.asarray(targets, dtype=np.float32))


def build_and_train(cfg, env, eps, device, seed):
    """按 cfg["train"]["epochs"] 训一个世界模型（阶段 A / 阶段 B 共用，避免两份实现）。

    ★ 这里**必须**允许梯度：本函数不接受 @torch.no_grad()。
      （曾误把装饰器插到它头上，症状是 loss.backward() 报
       "element 0 of tensors does not require grad"，且报错点在训练函数内部 ——
       单看报错会以为是模型写错了，实际是装饰器位置错。）
    """
    mcfg = cfg["model"]
    set_seed(seed)
    m = MLPWorldModel(obs_dim=env.obs_dim, act_dim=env.act_dim,
                      latent_dim=int(mcfg["latent_dim"]), hidden=int(mcfg["hidden"]),
                      discrete_act=bool(mcfg["discrete_act"]) and env.is_discrete
                      ).to(device)
    tr = tuple(torch.as_tensor(x) for x in transitions_from_episodes(eps))
    va = tuple(torch.as_tensor(x) for x in transitions_from_episodes(eps[:6]))
    train_world_model(m, tr, va, cfg, device, verbose=False)
    m.eval()
    return m


@torch.no_grad()
def latent_ref_stats(model, episodes, device, n_max=20000):
    """训练分布潜状态的均值与逐维标准差（用于 Mahalanobis 距离）。

    用**训练集**的观测过一遍 encoder 得到"模型见过的潜状态分布"。
    这不是训练集潜状态的目标值（那需要 encode(obs_next)），而是模型实际
    在 rollout 里会待着的那片区域 —— 漂移出它，转移模型就没见过了。
    """
    obs = np.concatenate([e["obs"] for e in episodes], 0)[:n_max]
    z = model.encode(torch.as_tensor(obs, device=device))
    return z.mean(0), z.std(0).clamp_min(1e-6)


@torch.no_grad()
def mahalanobis(z: torch.Tensor, mu: torch.Tensor, sd: torch.Tensor):
    """对角协方差下的 Mahalanobis 距离：sqrt(Σ ((z-μ)/σ)²)。"""
    return torch.sqrt(((z - mu) / sd) ** 2).pow(2).sum(-1).sqrt().mean(0)  # (H,)


def main():
    ap = argparse.ArgumentParser(description="X11 往返映射与纠错频率")
    ap.add_argument("--config", default="configs/pendulum.yaml")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--n-samples", type=int, default=512)
    ap.add_argument("--device", default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--scan-epochs", default=None,
                    help="阶段 B：逗号分隔的 epochs 网格，如 10,30,100,150")
    ap.add_argument("--tag", default="11_latent_roundtrip")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seeds = [int(s) for s in args.seeds.split(",")]
    if args.quick:
        seeds = seeds[:1]
    epochs = int(args.epochs or (20 if args.quick else cfg["train"]["epochs"]))
    cfg["train"]["epochs"] = epochs

    apply_style()
    device = get_device(args.device or cfg.get("device"))
    out = output_dir(cfg)
    horizons = list(cfg["eval"]["horizons"])
    h_max = max(horizons)
    thr = float(cfg["eval"]["err_threshold"])
    mode = cfg["eval"].get("threshold_mode", "rel")

    print(f"[11] X11 · 第三种曲线（纠错频率 K） env={cfg['env']['id']}  "
          f"device={describe_device(device)}  seeds={seeds}  epochs={epochs}")
    print(f"[11] K 网格 = {[k_label(k) for k in K_GRID]}   n_samples={args.n_samples}")

    t0 = time.time()
    per_seed = []
    diag_seed0 = {}

    for seed in seeds:
        env = make_env(cfg["env"]["id"], seed=seed)
        eps = collect_random_episodes(env, n_episodes=int(cfg["data"]["n_episodes"]),
                                      seed=seed, max_steps=cfg["env"].get("max_steps"))
        m = build_and_train(cfg, env, eps, device, seed)

        # ---- 评测窗口（固定 seed，保证各 K 之间可比）----
        rng = np.random.default_rng(10_000 + seed)
        obs0_np, acts_np, tgt_np = sample_windows(eps, h_max, args.n_samples, rng)
        obs0 = torch.as_tensor(obs0_np, device=device)
        acts = torch.as_tensor(acts_np, device=device)
        targets = torch.as_tensor(tgt_np, device=device)

        # ---------- 断言 1/2：两个端点必须与现有两条曲线逐位一致 ----------
        ref_open = multi_step_error_curve(m, eps, horizons, device,
                                          n_samples=args.n_samples, seed=10_000 + seed)
        ref_closed = closed_loop_error_curve(m, eps, horizons, device,
                                             n_samples=args.n_samples, seed=10_000 + seed)
        o_inf, _ = rollout_k(m, obs0, acts, h_max, None)
        o_k1, _ = rollout_k(m, obs0, acts, h_max, 1)
        nm_inf = per_step_nmse(o_inf, targets)
        nm_k1 = per_step_nmse(o_k1, targets)

        def at_h(dense):
            return [float(dense[h - 1]) for h in horizons]

        d_inf = max(abs(a - b) for a, b in zip(at_h(nm_inf),
                                               at_h(ref_open["nmse_per_step"])))
        d_k1 = max(abs(a - b) for a, b in zip(at_h(nm_k1),
                                              at_h(ref_closed["nmse_per_step"])))
        print(f"[11] seed={seed}  端点一致性: |K=inf − 开环| = {d_inf:.3e}   "
              f"|K=1 − 闭环| = {d_k1:.3e}")
        # 参考曲线用自己的窗口采样（内部 rng），只要 n_samples/seed 一致，窗口就一致
        assert d_inf < 1e-4, f"K=inf 与开环曲线不一致（差 {d_inf:.3e}）"
        assert d_k1 < 1e-4, f"K=1 与闭环曲线不一致（差 {d_k1:.3e}）"

        # ---------- 扫 K ----------
        mu, sd = latent_ref_stats(m, eps, device)
        row = {"seed": seed, "curves": {}, "h_star": {}}
        curves_dense = {}
        zdiag = {}
        for k in K_GRID:
            obs_hat, zs = rollout_k(m, obs0, acts, h_max, k, return_z=True)
            dense = per_step_nmse(obs_hat, targets)
            curves_dense[k_label(k)] = [float(v) for v in dense]
            vals = at_h(dense)
            h = reliable_horizon(horizons, vals, thr, mode)
            row["curves"][k_label(k)] = vals
            row["h_star"][k_label(k)] = h
            if seed == seeds[0]:
                zdiag[k_label(k)] = {
                    "mahal": [float(v) for v in mahalanobis(zs, mu, sd)],
                }
            print(f"[11]   seed={seed} {k_label(k):>10s}  H* = "
                  f"{h if h is not None else float('nan'):7.2f}   "
                  f"NMSE@1={vals[0]:.3e}  NMSE@{h_max}={vals[-1]:.3e}")
            # ★ R14 精神：发散不许静默通过。这不是 NaN，但 1e3 量级的 NMSE
            #   意味着 rollout 已经脱离物理范围 —— H* 虽然有限，却不可引用。
            peak = float(np.nanmax(vals))
            row.setdefault("diverged", {})[k_label(k)] = bool(peak > 1e3)
            if peak > 1e3:
                print(f"[11]     [WARN] {k_label(k)} 数值发散：峰值 NMSE = {peak:.2e} "
                      f"（>1e3）⇒ 该格的 H* 不可引用；"
                      f"通常是 epochs 不足导致转移模型在分布外外插失控")

        # ---------- 往返映射 R 的直接诊断（在纯开环轨迹上）----------
        with torch.no_grad():
            _, z_open = rollout_k(m, obs0, acts, h_max, None, return_z=True)
            z_rt = m.encode(m.decode(z_open.reshape(-1, z_open.shape[-1]))
                            ).reshape(z_open.shape)
        shift = (z_rt - z_open).flatten(2).norm(dim=-1).mean(0)        # (H,)
        znorm = z_open.flatten(2).norm(dim=-1).mean(0).clamp_min(1e-8)
        m_open = mahalanobis(z_open, mu, sd)
        m_rt = mahalanobis(z_rt, mu, sd)
        if seed == seeds[0]:
            diag_seed0 = {
                "shift_abs": [float(v) for v in shift],
                "shift_rel": [float(v) for v in (shift / znorm)],
                "mahal_open": [float(v) for v in m_open],
                "mahal_after_R": [float(v) for v in m_rt],
                "mahal_delta": [float(a - b) for a, b in zip(m_open, m_rt)],
            }
            # 训练分布自己的 Mahalanobis 尺度（作为"出分布"的参照线）
            z_ref = m.encode(torch.as_tensor(
                np.concatenate([e["obs"] for e in eps], 0)[:5000], device=device))
            with torch.no_grad():
                diag_seed0["mahal_train_ref"] = float(
                    mahalanobis(z_ref.detach(), mu, sd).mean().item())
        print(f"[11]   seed={seed}  往返位移 ‖R(z)−z‖/‖z‖: "
              f"h=1 {float((shift/znorm)[0]):.3e} → h={h_max} "
              f"{float((shift/znorm)[-1]):.3e}")
        print(f"[11]   seed={seed}  Mahalanobis(开环轨迹): "
              f"h=1 {float(m_open[0]):.2f} → h={h_max} {float(m_open[-1]):.2f}  "
              f"(训练分布参照 {diag_seed0.get('mahal_train_ref', float('nan')):.2f})")
        print(f"[11]   seed={seed}  R 是否拉回训练分布? "
              f"Δ d_M = d_M(z) − d_M(R(z)): h=1 {float(m_open[0]-m_rt[0]):+.3f} → "
              f"h={h_max} {float(m_open[-1]-m_rt[-1]):+.3f}"
              f"  ({'拉回' if float(m_open[-1]-m_rt[-1]) > 0 else '推离'})")

        per_seed.append(row)
        env.close()

    # ---------- 阶段 B：K 的效应是否随【模型质量】衰减 ----------
    # 起因：冒烟（20 epoch）时 H*(K) 从 14.36 单调涨到 28.15（2 倍），
    #       而全量（150 epoch）时几乎平坦（22.50 → 23.53，差 1 步，落在种子噪声内）。
    # ⇒ 假设：**K 的效应是模型质量的函数** —— 模型越差，"多久纠一次"越要紧。
    #   这正好与 X27 的结论同源（"forward 有效的前提是模型够好"）。
    quality = []
    if args.scan_epochs:
        egrid = [int(x) for x in args.scan_epochs.split(",")]
        ksub = [1, None]
        print()
        print("[11] ==== 阶段 B：K 的效应 vs 模型质量 ====")
        print(f"[11] epochs 网格 = {egrid}   K ∈ {[k_label(k) for k in ksub]}"
              f"   seeds={seeds}   （数据采集按 seed 缓存，只采一次）")
        eps_cache = {}
        for ep in egrid:
            cfg["train"]["epochs"] = ep
            rows = []
            for seed in seeds:
                if seed not in eps_cache:
                    e = make_env(cfg["env"]["id"], seed=seed)
                    eps_cache[seed] = collect_random_episodes(
                        e, n_episodes=int(cfg["data"]["n_episodes"]), seed=seed,
                        max_steps=cfg["env"].get("max_steps"))
                    e.close()
                eps_ = eps_cache[seed]
                # build_and_train 需要 env 提供 obs/act 维数 ⇒ 开一个 env，用完即关
                e = make_env(cfg["env"]["id"], seed=seed)
                m = build_and_train(cfg, e, eps_, device, seed)
                rng = np.random.default_rng(10_000 + seed)
                o0, ac, tg = sample_windows(eps_, h_max, args.n_samples, rng)
                obs0 = torch.as_tensor(o0, device=device)
                acts = torch.as_tensor(ac, device=device)
                targets = torch.as_tensor(tg, device=device)
                r = {"seed": seed, "epochs": ep}
                for k in ksub:
                    with torch.no_grad():
                        oh, _ = rollout_k(m, obs0, acts, h_max, k)
                    vals = at_h(per_step_nmse(oh, targets))
                    r[k_label(k)] = reliable_horizon(horizons, vals, thr, mode)
                    r[f"nmse1_{k_label(k)}"] = float(vals[0])
                rows.append(r)
                e.close()
            diffs, q1 = [], []
            for r in rows:
                a_, b_ = r[k_label(1)], r[k_label(None)]
                if a_ is not None and b_ is not None:
                    diffs.append(float(b_ - a_))
                q1.append(r[f"nmse1_{k_label(1)}"])
            dm = float(np.mean(diffs)) if diffs else float("nan")
            ds = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
            quality.append({"epochs": ep, "nmse_step1_mean": float(np.mean(q1)),
                            "h_k1": [r[k_label(1)] for r in rows],
                            "h_kinf": [r[k_label(None)] for r in rows],
                            "delta_mean": dm, "delta_std": ds,
                            "delta_values": diffs})
            print(f"[11]   epochs={ep:4d}  单步NMSE={np.mean(q1):.3e}  "
                  f"H*(K=1)={np.nanmean([r[k_label(1)] if r[k_label(1)] is not None else np.nan for r in rows]):7.2f}  "
                  f"H*(K=inf)={np.nanmean([r[k_label(None)] if r[k_label(None)] is not None else np.nan for r in rows]):7.2f}  "
                  f"ΔH* = {dm:+7.2f} ± {ds:5.2f}")
        print("[11]   ΔH* = H*(K=inf) − H*(K=1)  (>0 表示不纠错更长，即每步纠错是净损失)")
        cfg["train"]["epochs"] = epochs

    # ---------- 交叉点：K=1 与 K=inf 两条曲线的相对关系 ----------
    print()
    print("[11] ==== 交叉点分析（K=1 每步往返  vs  K=inf 纯开环）====")
    crosses = []
    for row in per_seed:
        a = np.asarray(row["curves"]["K=1"])
        b = np.asarray(row["curves"]["inf(open)"])
        diff = a - b                       # >0 表示闭环更差
        # ★ 端点处两条曲线的差恒为 0（第一步尚未施加往返）⇒ np.sign(0)=0 会被
        #   误判成一次符号翻转（冒烟时误报 h_cross=1.0）。
        #   处置：① 从第一个非零差开始 ② 内部的 0 填成前一个非零符号。
        nz = np.nonzero(np.abs(diff) > 0)[0]
        if len(nz) < 2:
            hc = None
        else:
            sub, hs_sub = diff[nz[0]:], horizons[nz[0]:]
            s_raw = np.sign(sub)
            sgn, fill = [], float(s_raw[0])
            for v in s_raw:
                if v != 0:
                    fill = float(v)
                sgn.append(fill)
            idx = np.where(np.diff(np.asarray(sgn)) != 0)[0]
            if len(idx):
                i = int(idx[0])
                h0, h1 = hs_sub[i], hs_sub[i + 1]
                f = abs(sub[i]) / (abs(sub[i]) + abs(sub[i + 1]) + 1e-30)
                hc = h0 + f * (h1 - h0)
            else:
                hc = None
        crosses.append({"seed": row["seed"], "h_cross": hc,
                        "closed_worse_early": bool(diff[len(diff) // 4] > 0),
                        "closed_better_late": bool(diff[-1] < 0),
                        "nmse_190_k1": float(a[-1]), "nmse_190_kinf": float(b[-1])})
        print(f"[11]   seed={row['seed']}  交叉点 h ≈ "
              f"{hc if hc is not None else float('nan'):.1f}   "
              f"(早期闭环更差={diff[0] > 0}, 末期闭环更好={diff[-1] < 0})")

    # ---------- H*(K) 汇总 ----------
    print()
    print("[11] ==== H* 随纠错频率 K 的变化（均值 ± 标准差，n=种子数）====")
    hk = {}
    for k in K_GRID:
        xs = [r["h_star"][k_label(k)] for r in per_seed]
        xs = [float(x) for x in xs if x is not None]
        hk[k_label(k)] = {"mean": float(np.mean(xs)) if xs else None,
                          "std": float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0,
                          "n": len(xs), "values": xs}
        v = hk[k_label(k)]["mean"]
        s = hk[k_label(k)]["std"]
        print(f"[11]   {k_label(k):>10s}  H* = "
              f"{v if v is not None else float('nan'):7.2f} ± {s:6.2f}")

    # ---------- 出图 ----------
    ncol = 3 if quality else 2
    fig, axes = plt.subplots(2, ncol, figsize=(12.5 if ncol == 2 else 18.0, 8.2))
    if ncol == 3:
        axes[1, 1].axis("off")          # 原 (4) 挪到右上，中间留空
    cmap = plt.get_cmap("viridis")

    ax = axes[0, 0]
    for i, k in enumerate(K_GRID):
        vals = np.asarray(per_seed[0]["curves"][k_label(k)])
        ax.plot(horizons, vals, "-", lw=1.8 if k in (1, None) else 1.1,
                color=cmap(0.15 + 0.75 * i / max(1, len(K_GRID) - 1)),
                label=k_label(k))
    ax.axhline(thr, color="#888", ls=":", lw=1.2, label=f"thr {thr}")
    ax.set_yscale("log"); ax.set_xlabel("rollout horizon (steps)")
    ax.set_ylabel("NMSE (per-step)")
    ax.set_title(f"(1) error curves by correction period K (seed={seeds[0]})")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.25)

    ax = axes[0, 1]
    ks = [k_label(k) for k in K_GRID]
    mv = [hk[x]["mean"] if hk[x]["mean"] is not None else 0.0 for x in ks]
    sv = [hk[x]["std"] for x in ks]
    ax.errorbar(range(len(ks)), mv, yerr=sv, fmt="o-", capsize=4, color="#4C78A8")
    ax.set_xticks(range(len(ks))); ax.set_xticklabels(ks, fontsize=8, rotation=20)
    ax.set_ylabel("H* (steps)"); ax.set_xlabel("correction period K  (1 = every step)")
    ax.set_title("(2) H* vs correction frequency")
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    if diag_seed0:
        h_range = np.arange(1, h_max + 1)
        ax.plot(h_range, diag_seed0["shift_rel"], color="#E45756",
                label="cost: ||R(z)-z|| / ||z||")
        ax2 = ax.twinx()
        ax2.plot(h_range, diag_seed0["mahal_open"], color="#4C78A8", ls="-",
                 label="d_M (open-loop traj)")
        ax2.plot(h_range, diag_seed0["mahal_after_R"], color="#72B7B2", ls="--",
                 label="d_M after R")
        ax2.axhline(diag_seed0["mahal_train_ref"], color="#999", ls=":",
                    label="d_M of training latent dist")
        ax2.set_ylabel("Mahalanobis distance to training latent dist")
        ax.set_xlabel("rollout step"); ax.set_ylabel("relative roundtrip shift")
        ax.set_title("(3) roundtrip: cost vs pull-back-to-distribution")
        ax.legend(fontsize=7, loc="upper left")
        ax2.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.25)

    ax = axes[0, 2] if quality else axes[1, 1]
    a = np.asarray(per_seed[0]["curves"]["K=1"])
    b = np.asarray(per_seed[0]["curves"]["inf(open)"])
    ax.plot(horizons, a - b, "o-", ms=3, color="#F58518")
    ax.axhline(0.0, color="k", lw=1)
    ax.set_xscale("log"); ax.set_xlabel("rollout horizon (steps)")
    ax.set_ylabel("NMSE(K=1) - NMSE(K=inf)")
    ax.set_title("(4) sign of the difference: >0 = closed-loop WORSE\n"
                 "crossing point = where per-step correction turns beneficial")
    ax.grid(alpha=0.25)

    if quality:
        ax = axes[1, 2]
        ep = [q["epochs"] for q in quality]
        dm = [q["delta_mean"] for q in quality]
        ds = [q["delta_std"] for q in quality]
        ax.errorbar(range(len(ep)), dm, yerr=ds, fmt="o-", capsize=4, color="#E45756")
        ax.axhline(0.0, color="k", lw=1)
        ax.set_xticks(range(len(ep))); ax.set_xticklabels([str(e) for e in ep])
        ax.set_xlabel("training epochs (= model quality proxy)")
        ax.set_ylabel("H*(K=inf) - H*(K=1)   [steps]")
        ax.set_title("(5) effect of correction frequency vs model quality\n"
                     ">0: no-correction rolls longer  => correction is net LOSS")
        ax.grid(alpha=0.25)

    fig.suptitle(f"wmlab X11 · third curve = correction period K · env={cfg['env']['id']} "
                 f"· seeds={seeds} · epochs={epochs}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    png = save_fig(fig, out / f"{args.tag}.png")
    plt.close(fig)

    summary = {
        "experiment": "X11 third curve: correction period K (roundtrip map R=enc o dec)",
        "env": cfg["env"]["id"], "seeds": seeds, "epochs": epochs,
        "horizons": horizons, "threshold": thr, "n_samples": args.n_samples,
        "k_grid": [k_label(k) for k in K_GRID],
        "endpoint_identity_asserted": True,
        "per_seed": per_seed, "h_star_by_K": hk, "crossings": crosses,
        "quality_scan": quality, "diag_seed0": diag_seed0,
        "seconds": time.time() - t0,
    }
    js = out / f"{args.tag}.json"
    js.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[11] 用时 {time.time()-t0:.1f}s")
    print(f"[11] 图 -> {png}")
    print(f"[11] 指标 -> {js}")
    print("[11] OK")


if __name__ == "__main__":
    main()
