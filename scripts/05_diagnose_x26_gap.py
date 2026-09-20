#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""脚本 05 —— X26 对账：离线 NMSE(h) 曲线到底能不能预测在线 NMSE(age)？

起因（2026-09-19 夜，04 首次完整跑）
---------------------------------
解析式系统性**低估**在线跟踪误差，中位相对偏差 **56%**（P 族高达 −77%）。

结论（2026-09-20，由本脚本 (c) 段的决定性复算给出）
--------------------------------------------------
那 56% **完全是口径造成的假缺口，不是模型或解析式的缺陷**：

  04 的离线曲线用**累积平均**口径 `mse(pred[:, :h], tgt[:, :h])`，
  而在线 `run_tracking` 累计的是**每一步自身**的误差（逐点口径）。
  前 h-1 步的小误差被摊进均值 ⇒ 离线曲线系统性偏低 ⇒ 解析式"低估"仿真。

决定性证据：把取样起点固定为 04 的方式（`random(04)`）重算，与 04 离线曲线逐点一致；
把在线实测与**同条件**离线对比，比值 = 1.041 / 1.004（≈1，映射成立）。

修正后（04 改用逐点口径）本脚本的定位从『诊断缺口』变为『验证修正』：
若『在线实测 vs 04 离线主曲线』的比值 ≈ 1.000，即说明离线↔在线的映射成立。

输出
----
  outputs/05_x26_gap_diagnosis.csv   逐 h 对比表
  outputs/05_x26_gap_diagnosis.png   四联图
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from wmlab.data import collect_random_episodes
from wmlab.envs import make_env
from wmlab.eval import (NmseCurve, expected_nmse_analytic, geometric_age_pmf,
                        uniform_age_pmf)
from wmlab.models import MLPWorldModel
from wmlab.utils import load_config
from wmlab.utils.plot import PALETTE, apply_style, save_fig

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TAG = "04_channel"


def load(tag: str = "04_channel") -> dict:
    return json.loads((OUT / f"{tag}.json").read_text(encoding="utf-8"))


# =====================================================================
# (c) 决定性复算：同一个模型、同一批数据，只改「窗口起点怎么取」
# =====================================================================
def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = MLPWorldModel(obs_dim=int(ck["obs_dim"]), act_dim=int(ck["act_dim"]),
                          latent_dim=int(ck["latent_dim"]), hidden=int(ck["hidden"]),
                          discrete_act=bool(ck["discrete_act"])).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck


@torch.no_grad()
def per_h_mse_closed_loop(model, obs0, acts, targets, device, batch=512):
    """给一批窗口，算「每步重编码」闭环预测的**逐 h** MSE（对 dim 与窗口平均）。

    ★ 为什么 h 这一维必须留下：本诊断比的就是"每一个 h 的误差"。
      04 里 `closed_loop_error_curve` 正是按 horizon 切片后各自求 mean，口径一致。
    """
    B, H, _ = targets.shape
    acc = []
    for i in range(0, B, batch):
        o = obs0[i:i + batch].to(device)
        a = acts[i:i + batch].to(device)
        tg = targets[i:i + batch].to(device)
        cur = o
        preds = []
        for t in range(H):
            cur = model.predict_next(cur, a[:, t])
            preds.append(cur)
        ph = torch.stack(preds, dim=1)
        acc.append(((ph - tg) ** 2).mean(dim=2).cpu().numpy())   # (b, H)
    return np.concatenate(acc, axis=0).mean(axis=0)              # (H,)


def build_windows(episodes, starts, h_max):
    """按显式起点列表构造 (obs0, acts, targets)。starts = [(ep_idx, t0), ...]"""
    obs0, acts, tgs = [], [], []
    for ei, t0 in starts:
        ep = episodes[ei]
        if t0 + h_max + 1 > len(ep["obs"]):
            continue
        obs0.append(ep["obs"][t0])
        acts.append(np.asarray(ep["act"])[t0:t0 + h_max])
        tgs.append(ep["obs"][t0 + 1:t0 + 1 + h_max])
    return (torch.as_tensor(np.asarray(obs0, dtype=np.float32)),
            torch.as_tensor(np.asarray(acts)),
            torch.as_tensor(np.asarray(tgs, dtype=np.float32)))


def run_sampling_probe(ckpt_path, hs, offline_nmse, offline_cum, sched, device):
    """三种取样方式各算一条闭环曲线，把 gap 拆到具体的一环上。

    `random(04)` 复刻 04 里 `_sample_windows` 的做法 —— 它必须与离线曲线基本一致，
    否则说明本次重建（模型 / 评测集）本身就对不上，后面的比较没有意义。

    ★ 2026-09-20：`offline_nmse` 改为**逐点口径**（04 修正后的主曲线）。
      它才是与在线 tracking 同口径、可逐点对账的曲线。累积口径作为 `offline_cum`
      单独传入，只用于复现"修正前会得到什么"。
    """
    model, ck = load_model(ckpt_path, device)
    cfg = load_config("configs/channel.yaml")
    ch = cfg["channel"]

    ev_env = make_env(cfg["env"]["id"], seed=int(ch["eval_seed"]),
                      max_episode_steps=int(ch["eval_steps"]))
    eval_eps = collect_random_episodes(ev_env, n_episodes=int(ch["eval_episodes"]),
                                       seed=int(ch["eval_seed"]), max_steps=None)
    ev_env.close()

    allobs = np.concatenate([e["obs"] for e in eval_eps], axis=0).astype(np.float64)
    var_rebuilt = float(allobs.var())
    var_g = float(ck["var_global"])
    print("\n---- (c) 决定性复算：只改「窗口起点怎么取」----")
    print(f"  重建评测集 var = {var_rebuilt:.6f}；checkpoint 记录 var = {var_g:.6f}；"
          f"相对差 = {abs(var_rebuilt - var_g) / var_g:.2e}")
    if abs(var_rebuilt - var_g) / var_g > 1e-6:
        print("  ★ 警告：重建的评测集与 04 不一致，下面的比较不可信")

    h_max = int(max(hs))
    rng = np.random.default_rng(0)                       # 与 04 的 seed 对齐
    valid = [i for i, ep in enumerate(eval_eps) if len(ep["obs"]) > h_max + 1]
    starts_rand = []
    for _ in range(256):
        ei = int(valid[rng.integers(len(valid))])
        T = len(eval_eps[ei]["obs"])
        starts_rand.append((ei, int(rng.integers(0, T - h_max - 1))))

    starts_all = [(i, t0) for i, ep in enumerate(eval_eps)
                  for t0 in range(0, len(ep["obs"]) - h_max - 1)]
    # 周期 T=8 的"成功接收时刻"：t0 ≡ 0 (mod 8)
    starts_per = [(i, t0) for i, ep in enumerate(eval_eps)
                  for t0 in range(0, len(ep["obs"]) - h_max - 1, 8)]

    curves = {}
    for name, starts in (("random(04)", starts_rand), ("all_starts", starts_all),
                         ("periodic_T=8", starts_per)):
        o, a, tg = build_windows(eval_eps, starts, h_max)
        mse = per_h_mse_closed_loop(model, o, a, tg, device)
        curves[name] = mse / var_g
        print(f"  {name:<14} 窗口数={len(o):>6}  NMSE(h=1)={curves[name][0]:.4e}  "
              f"NMSE(h=20)={curves[name][19]:.4e}  NMSE(h=50)={curves[name][49]:.4e}")

    print("\n  与 04 记录的逐点对比")
    curve = NmseCurve(hs, offline_nmse)      # ★ 04 修正后的**逐点**口径主曲线
    curve_cum = NmseCurve(hs, offline_cum)   # 累积口径（修正前的旧曲线，作对照）
    print("  ★ 对照关系：`04 离线(逐点)` 与该行的四列**同口径**（都只取第 h 步的误差）；")
    print("     `04 离线(累积·旧)` 是修正前的曲线（把 h 步一起平均），只用于解释历史缺口。")
    print(f"  {'h':>5} {'04离线(逐点)':>14} {'04离线(累积·旧)':>15} {'random(04)':>13} "
          f"{'all_starts':>13} {'online P(T=8)':>14} {'离线/全起点':>11}")
    by_age_p8 = {int(k): float(v)
                 for k, v in sched["P(T=8)"]["empirical_nmse_by_age"].items()}
    h2k = {int(h): k for k, h in enumerate(hs)}
    rows_c = []
    for h in hs:
        hh = int(h)
        k = h2k[hh]          # ★ offline 是"19 个离散点"，不是"每个整数 h 都有"
        i = hh - 1           # ★ 逐点曲线是"长度 190 的连续序列"
        r = {"h": hh, "offline_04_pointwise": float(offline_nmse[k]),
             "offline_04_cumulative": float(offline_cum[k]),
             "random_04": float(curves["random(04)"][i]),
             "all_starts": float(curves["all_starts"][i]),
             "periodic_T8": float(curves["periodic_T=8"][i]),
             "online_P_T8": by_age_p8.get(hh, float("nan"))}
        r["ratio_point_vs_allstarts"] = r["offline_04_pointwise"] / max(r["all_starts"], 1e-12)
        r["ratio_cum_vs_point"] = r["offline_04_cumulative"] / max(r["offline_04_pointwise"], 1e-12)
        rows_c.append(r)
        online_s = f"{r['online_P_T8']:.4e}" if np.isfinite(r["online_P_T8"]) else "-"
        print(f"  {h:>5} {r['offline_04_pointwise']:>14.4e} {r['offline_04_cumulative']:>15.4e} "
              f"{r['random_04']:>13.4e} {r['all_starts']:>13.4e} "
              f"{online_s:>14} {r['ratio_point_vs_allstarts']:>11.3f}")

    print("\n  关键比值（中位数，仅取分母 ≥ 1e-4 的点）")

    def med_ratio(a_key, b_key):
        v = [r[a_key] / r[b_key] for r in rows_c
             if r[b_key] >= 1e-4 and np.isfinite(r.get(a_key, float("nan")))]
        return (float(np.median(v)) if v else float("nan")), len(v)

    for a_key, b_key, why in (
        ("random_04", "offline_04_pointwise", "① 复现一致性（同口径，应 ≈ 1.000）"),
        ("all_starts", "random_04", "② 256 随机窗口 vs 全部起点（采样噪声）"),
        ("periodic_T8", "all_starts", "③ 周期起点 vs 全部起点（位置效应）"),
        ("online_P_T8", "periodic_T8", "④ 在线实测 vs 同条件离线（★ 应 ≈ 1.000）"),
        ("online_P_T8", "offline_04_pointwise", "⑤ 在线实测 vs 04 离线主曲线（★ 应 ≈ 1.000）"),
        ("offline_04_cumulative", "offline_04_pointwise", "⑥ 累积 vs 逐点（修正前的系统偏差，应 > 1）"),
    ):
        rr, n = med_ratio(a_key, b_key)
        print(f"    {why:<46} ratio = {rr:.3f}  (n={n})")

    # ---------- ★ 用逐点曲线重算 X26 解析，看 gap 是否收敛 ----------
    pt_nmse = [float(curves["all_starts"][int(h) - 1]) for h in hs]
    curve_pt = NmseCurve(hs, pt_nmse)
    print("\n  ★ 用「逐点」曲线重做 X26 解析对账（这才是正确的对账）")
    print(f"  {'调度':<12} {'仿真':>12} {'解析(逐点曲线)':>15} {'相对':>9} "
          f"{'解析(累积曲线,旧)':>17} {'相对':>9}")
    recon = []
    for lbl in sched:
        rec = sched[lbl]
        if lbl.startswith("P(T="):
            pmf = uniform_age_pmf(int(lbl[4:-1]))
        else:
            pmf = geometric_age_pmf(float(lbl[4:-1]), 1000)
        sim = float(rec["sim_nmse"])
        if sim <= 1e-6:
            continue
        e_pt = expected_nmse_analytic(curve_pt, pmf)
        e_cum = expected_nmse_analytic(curve_cum, pmf)
        recon.append({"label": lbl, "sim": sim, "analytic_pointwise": e_pt,
                      "rel_pointwise": (e_pt - sim) / sim,
                      "analytic_cumulative": e_cum,
                      "rel_cumulative": (e_cum - sim) / sim})
    for r in recon:
        print(f"  {r['label']:<12} {r['sim']:>12.4e} {r['analytic_pointwise']:>15.4e} "
              f"{r['rel_pointwise']:>+9.1%} {r['analytic_cumulative']:>17.4e} "
              f"{r['rel_cumulative']:>+9.1%}")
    rp = np.asarray([abs(r["rel_pointwise"]) for r in recon])
    rc = np.asarray([abs(r["rel_cumulative"]) for r in recon])
    print(f"  → 中位绝对偏差：逐点曲线 {np.median(rp):.1%}   累积曲线（旧，修正前的 04 所用） {np.median(rc):.1%}")

    # ---------- ★ 逐点口径下真实的 H* ----------
    thr = 0.05
    def first_cross(hs_list, ys, t):
        for j in range(len(ys)):
            if ys[j] > t:
                if j == 0:
                    return float(hs_list[0])
                h0, h1, e0, e1 = hs_list[j - 1], hs_list[j], ys[j - 1], ys[j]
                return float(h0 + (t - e0) / (e1 - e0) * (h1 - h0)) if e1 != e0 else float(h1)
        return None

    grid = np.arange(1, h_max + 1, dtype=float)
    h_star_pt = first_cross(grid, curves["all_starts"], thr)
    h_star_cum = first_cross(grid, np.interp(grid, [h for h in hs], offline_cum), thr)
    print("\n  ★★ H* 的两种口径对比（阈值 NMSE > 0.05）")
    print(f"     逐点口径（第 h 步自身的误差）      H* = {h_star_pt:.2f}   ← 正确值")
    print(f"     累积口径（修正前的 04 与 X2 所用） H* = {h_star_cum:.2f}   ← 被高估")
    if h_star_pt and h_star_cum:
        print(f"     ⇒ 累积口径把 H* **高估**了 {h_star_cum - h_star_pt:.2f} 步"
              f"（{(h_star_cum / h_star_pt - 1):.1%}）")

    return {"rows": rows_c, "recon": recon,
            "H_star_pointwise": h_star_pt, "H_star_cumulative": h_star_cum,
            "curves": {k: v.tolist() for k, v in curves.items()},
            "var_rebuilt": var_rebuilt, "var_ckpt": var_g}


def pchip(x, y, xq):
    """单调保形三次插值（PCHIP，Fritsch–Carlson）。

    为什么不用 scipy：本仓库刻意保持依赖最小（见 README §设计约定）。
    实现自标准公式，30 行，够用且可核。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    dx = np.diff(x)
    delta = np.diff(y) / dx

    d = np.zeros(n)
    d[0] = delta[0]
    d[-1] = delta[-1]
    for i in range(1, n - 1):
        if delta[i - 1] * delta[i] <= 0:
            d[i] = 0.0
        else:
            w1 = 2 * dx[i] + dx[i - 1]
            w2 = dx[i] + 2 * dx[i - 1]
            d[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i])

    # ★ 变量名必须分开：dx 是节点间距（长度 n-1），hk 是被查询点所在的间距（长度 len(xq)）。
    #   第一版两者同名 h，导致 h10*h 用了未索引的整条数组 → 广播报错 (len(xq),) vs (n-1,)。
    xq = np.asarray(xq, dtype=float)
    idx = np.clip(np.searchsorted(x, xq) - 1, 0, n - 2)
    hk = dx[idx]
    t = (xq - x[idx]) / hk
    y0, y1, d0, d1 = y[idx], y[idx + 1], d[idx], d[idx + 1]
    h00 = (1 + 2 * t) * (1 - t) ** 2
    h10 = t * (1 - t) ** 2
    h01 = t ** 2 * (3 - 2 * t)
    h11 = t ** 2 * (t - 1)
    return h00 * y0 + h10 * hk * d0 + h01 * y1 + h11 * hk * d1


def main() -> int:
    apply_style()
    J = load()
    oc = J["offline_curve_closed_global"]
    # ★ 2026-09-20：主曲线取**逐点**口径（与在线 tracking 同口径）。缺该键时退回累积。
    #   逐点数组是**稠密**的（长度 = max(horizons)），必须按 horizons 抽取再配对，
    #   否则 NmseCurve 会静默截断（见 NmseCurve 的 docstring）。
    key = "nmse_per_step_global" if "nmse_per_step_global" in oc else "nmse_global"
    hs = [int(h) for h in oc["horizons"]]

    def _aligned(arr):
        a = np.asarray(arr, dtype=float)
        return [float(v) for v in a] if a.size == len(hs) else [float(a[h - 1]) for h in hs]

    nm = _aligned(oc[key])
    nm_cum = _aligned(oc.get("nmse_global", oc[key]))
    curve = NmseCurve(hs, nm)

    print("=" * 96)
    print("X26 对账 —— 离线闭环 NMSE(h)  vs  在线实测 NMSE(age=h)")
    print("=" * 96)
    print(f"离线曲线点数 = {len(hs)}，h ∈ [{min(hs)}, {max(hs)}]")
    print(f"主曲线口径 = {key}  （{'逐点，与在线跟踪同口径 ✔' if key.startswith('nmse_per_step') else '★ 累积：与在线不同口径，仅作历史对照'})")
    print(f"var_global = {J['var_global']:.6f}（离线与在线共用同一分母，可比）")
    print("★ 本脚本在 2026-09-20 口径修正后的作用已从『诊断缺口』变为『验证修正』：")
    print("  若『在线实测 vs 04 离线主曲线』的比值 ≈ 1.000，说明离线↔在线的映射成立。")

    sched = {r["label"]: r for r in J["schedules"]}
    focus = ["P(T=8)", "P(T=20)", "P(T=50)", "P(T=100)", "R(p=0.9)", "R(p=0.98)", "R(p=0.99)"]
    focus = [f for f in focus if f in sched]

    # ---------- (a) 同一 h 上逐点对比 ----------
    rows = []
    for h in hs:
        row = {"h": h, "offline_nmse": float(curve(h)),
               "offline_cumulative": float(np.interp(h, hs, nm_cum))}
        for lbl in focus:
            byage = {int(k): float(v) for k, v in sched[lbl]["empirical_nmse_by_age"].items()}
            row[lbl] = byage.get(h, float("nan"))
        rows.append(row)

    print("\n---- (a) 同一 h 上的逐点对比（空白 = 该调度没有出现这个 age）----")
    head = f"{'h':>5} {'offline(逐点)':>14} " + " ".join(f"{l:>12}" for l in focus)
    print(head)
    for r in rows:
        line = f"{r['h']:>5} {r['offline_nmse']:>14.3e} "
        for l in focus:
            v = r[l]
            line += f"{'':>12} " if not np.isfinite(v) else f"{v:>12.3e} "
        print(line)

    # 比值（只看量级足够大的点，避免分母病态）
    print("\n---- 在线 / 离线(逐点) 的比值（仅统计 offline ≥ 1e-4 的点）----")
    ratios = {}
    for lbl in focus:
        v = [r[lbl] / r["offline_nmse"] for r in rows
             if np.isfinite(r[lbl]) and r["offline_nmse"] >= 1e-4]
        if v:
            ratios[lbl] = (float(np.median(v)), len(v))
            print(f"  {lbl:<14} 中位比值 = {np.median(v):.3f}  （n={len(v)} 个 h 点）"
                  f"{'   ← ≈1 说明映射成立' if abs(np.median(v) - 1) < 0.35 else ''}")

    # ---------- (b) 插值方式的影响 ----------
    print("\n---- (b) 线性 vs 保形(PCHIP) 插值对解析值的影响 ----")
    xk = np.asarray(curve.h, dtype=float)
    yk = np.asarray(curve.y, dtype=float)

    def analytic_pchip(pmf):
        g = np.arange(len(pmf), dtype=float)
        vals = pchip(xk, yk, np.clip(g, xk[0], xk[-1]))
        vals = np.where(g <= 0.0, 0.0, vals)
        vals = np.where(g > xk[-1], yk[-1], vals)
        return float(np.dot(pmf, vals))

    interp_table = []
    for lbl in focus:
        rec = sched[lbl]
        if lbl.startswith("P(T="):
            period = int(lbl[4:-1])
            pmf = uniform_age_pmf(period)
        else:
            pmf = geometric_age_pmf(float(lbl[4:-1]), 1000)
        lin = expected_nmse_analytic(curve, pmf)
        pch = analytic_pchip(pmf)
        sim = float(rec["sim_nmse"])
        interp_table.append({
            "label": lbl, "sim": sim, "analytic_linear": lin, "analytic_pchip": pch,
            "rel_linear": (lin - sim) / sim if sim > 1e-9 else float("nan"),
            "rel_pchip": (pch - sim) / sim if sim > 1e-9 else float("nan"),
        })
        print(f"  {lbl:<14} sim={sim:.4e}  linear={lin:.4e} ({interp_table[-1]['rel_linear']:+.1%})"
              f"   pchip={pch:.4e} ({interp_table[-1]['rel_pchip']:+.1%})")

    # ---------- (c) 决定性复算 ----------
    ckpt = OUT / f"{TAG}_model.pt"
    probe = None
    if ckpt.exists():
        probe = run_sampling_probe(ckpt, hs, nm, nm_cum, sched, torch.device("cpu"))
    else:
        print(f"\n---- (c) 跳过：未找到 {ckpt.name}"
              f"（旧版 04 只存数字不存权重，重跑一次 04 即可生成）----")

    # ---------- 落盘 ----------
    csv_path = OUT / "05_x26_gap_diagnosis.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["h", "offline_nmse", "offline_cumulative"] + focus,
                           extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if (isinstance(v, float) and not np.isfinite(v)) else v)
                        for k, v in r.items()})

    # ---------- 图 ----------
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.6))

    ax = axes[0, 0]
    ax.plot(hs, nm_cum, "k:", lw=1.6, alpha=0.45, label="offline · cumulative (legacy)")
    ax.plot(hs, nm, "k--", lw=2.0, label="offline closed-loop · per-step ★")
    for i, lbl in enumerate(focus):
        byage = sorted((int(k), float(v)) for k, v in sched[lbl]["empirical_nmse_by_age"].items())
        byage = [(a, v) for a, v in byage if a > 0 and v > 0]
        ax.plot([a for a, _ in byage], [v for _, v in byage], "o", ms=3.2, alpha=0.7,
                label=lbl, color=plt.rcParams["axes.prop_cycle"].by_key()["color"][i % 10])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("age / horizon h (steps)"); ax.set_ylabel("NMSE")
    ax.set_title("(a) Same h-axis: offline per-step curve vs online measured")
    ax.legend(fontsize=7)

    ax = axes[0, 1]
    for lbl, (med, n) in ratios.items():
        ax.barh(lbl, med, color=PALETTE["red"] if med > 1 else PALETTE["blue"], alpha=0.8)
    ax.axvline(1.0, color="k", ls="--", lw=1)
    ax.set_xlabel("median( online / offline )  at matched h")
    ax.set_title("(b) Pointwise ratio — is the mapping 1:1?")
    ax.tick_params(axis="y", labelsize=8)

    ax = axes[1, 0]
    lbls = [t["label"] for t in interp_table]
    x = np.arange(len(lbls)); w = 0.38
    ax.bar(x - w / 2, [t["rel_linear"] * 100 for t in interp_table], w,
           label="linear interp", color=PALETTE["orange"])
    ax.bar(x + w / 2, [t["rel_pchip"] * 100 for t in interp_table], w,
           label="PCHIP (shape-preserving)", color=PALETTE["green"])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(lbls, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("analytic vs sim  [%]")
    ax.set_title("(c) Does interpolation choice close the gap?")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    hh, d2 = curve.second_difference()
    ax.plot(hh, d2, color=PALETTE["green"], lw=1.8)
    ax.axhline(0, color="k", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("h (steps)"); ax.set_ylabel("Δ²NMSE")
    ax.set_title(f"(d) Per-step curve curvature: convex fraction = {curve.convex_fraction():.2%}\n"
                 f"convex -> linear interp underestimates")

    fig.suptitle("wmlab · X26 reconciliation (per-step calibration) · "
                 "offline NMSE(h) vs online NMSE(age)", fontsize=10)
    fig.tight_layout()
    png = save_fig(fig, OUT / "05_x26_gap_diagnosis.png")

    print(f"\n表 -> {csv_path}")
    print(f"图 -> {png}")
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
