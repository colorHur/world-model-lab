# -*- coding: utf-8 -*-
"""★ X45「相位效应的边界归因」：把 X44 的 T1 拆开 —— 真实相位效应，还是边界假象？

================================================================= 由来（为什么必须有这一号）
X44 的 T1 判据落空：相位扫描里任务距离最大偏差 **6.40%（T=8）/ 6.26%（T=16）> 5%**。
但 X44 §6 局限 1 自己写着：线性时刻表落在 300 步窗口里 ⇒ 尾部静默 = 19−Δ 步
（Δ=0 是 19、Δ=7 是 12），**随相位漂移** ⇒ 那 6.4% 是「**相位 + 尾部边界**」的
**合并效应**，**无法判断有多少是真实相位效应**。本号就做这件事。

★★ 为什么**不是**「把窗口做成循环」（X44 §8 的建议）—— 跑前审前提的结论（0 算力）
   设窗口 [1, M]、M = K·T，时刻表是周期 T 的均匀网格与窗口的交集。则
   **尾部静默随相位的变化幅度恒为 T−1**：
       · 线性表（X44 用）：tail 从 2T−1（Δ=0）降到 T（Δ=T−1） ⇒ 幅度 T−1
       · 循环表（§8 建议）：tail 从 T−1（Δ=0）降到 0（Δ=T−1） ⇒ 幅度 T−1
   ⇒ 「循环化」只是把**基线**挪了 T（因为它比线性**多发一次**：K vs K−1 个点），
     **幅度不变 ⇒ 零解耦**。数值证据见自检 [51]。⇒ 真正的自变量是**指标口径**。

================================================================= 做法（与 X44 逐位可比）
① 完全沿用 X44 设置：M=300、warmup=400、PER=0、n_ep=30、seed=0、**同一个模型缓存**
   ⇒ 相位 Δ ∈ X44 扫过的档必须**逐位复现** X44 的 est_nmse / mean_dist_tail（S_x1）。
② 相位扫描**全档** Δ=0..T−1（X44 只扫 subsample 5 档；边界效应在相邻档之间变化很细）。
③ `run_closed_loop_control(return_traces=True)` 拿**逐点 dist**，在**同一次运行**上算**四口径**：
   · `tail`       最后 25%，**= X44 口径**（含尾静默）
   · `core`       去掉首段 [t₁,t₂) 与末段 [t_K,M]（含首尾静默；**位置随 Δ 平移**）
   · `core_fixed` 去掉**固定 3T 步**首尾（**位置与相位无关**；3T > 最大尾静默）⇒ 诊断用
   · `full`       全段

================================================================= 预写判据（跑前定死，不许事后编解释）
S_x1 接线（★最强）：`tail` 口径**逐位复现 X44**（相位 Δ ∈ X44 扫过的档），
     且脚本算的 tail == control.py 报的 `mean_dist_tail`（同一次运行，逐位）。
S_x2 恒等式：线性表 (头静默, 尾静默) = (Δ, window−1−Δ−(N−1)T)，逐 Δ 机器核对。
S_x3 循环网格的 tail 幅度恒 = T−1（数值坐实「循环化不解耦」）。
S_x4 T=1 ⇒ est_nmse 恒 0。
H_c1 `tail` 相位偏差 >5%（复现 X44）**而** `core` 幅度 ≤5% ⇒ **边界归因成立**。
H_c2 `core` 偏差 >5% ⇒ 存在真实相位效应（**或** core 的位置平移在起作用，见 H_c3）。
H_c3 ★★ 诊断：`core_fixed`（位置固定）偏差
     · ≈ core ⇒ **真实相位效应**（排除"位置平移"这个伪影）；
     · ≪ core ⇒ core 的偏差主要来自**位置平移的伪影**，不是相位本身。
H_c4 报 `tail 偏差 / core_fixed 偏差` ⇒ 量化"尾部口径"相对"位置固定稳态"的放大倍数。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from wmlab.control import (PDRelativeController, collect_controlled_episodes,  # noqa: E402
                           run_closed_loop_control)
from wmlab.data import split_episodes, transitions_from_episodes  # noqa: E402
from wmlab.envs import make_env  # noqa: E402
from wmlab.eval.timing import (boundary_silence, cyclic_grid, expected_attempts,  # noqa: E402
                               lookup_schedule, metric_over, n_tx_budget, plan_times)
from wmlab.models.prob_world_model import GaussianWorldModel  # noqa: E402
from wmlab.train import train_world_model  # noqa: E402
from wmlab.utils import count_params, get_device, load_config, output_dir, set_seed  # noqa: E402
from wmlab.utils.plot import PALETTE, apply_style, save_fig  # noqa: E402

# ★ 复用 X44 的模型缓存（**逐位可比的前提**）：同一个 .pt ⇒ 世界模型逐位相同
X44_CACHE = "x44_model.pt"
X44_JSON = "28_x44_timing.json"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/uav_triggered.yaml")
    p.add_argument("--tag", default="29_x45_boundary")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--n-ep", type=int, default=None, help="闭环 episode 数（配对 ⇒ 同 seed）")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--retrain", action="store_true", help="忽略模型缓存、重训")
    p.add_argument("--periods", default=None, help="逗号分隔的 T 列表（覆盖网格）")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["train"]["epochs"] = int(args.epochs)
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    if args.quick:
        cfg["train"]["epochs"] = min(int(cfg["train"]["epochs"]), 30)
        cfg["data"]["n_episodes"] = min(int(cfg["data"]["n_episodes"]), 40)
        cfg["task"]["n_episodes"] = 8
        cfg["task"]["measured_steps"] = 120
        cfg["task"]["warmup_steps"] = 200

    seed = int(cfg["seed"])
    set_seed(seed)
    apply_style()
    device = get_device(args.device or cfg.get("device"))
    out = output_dir(cfg)
    tmp = out / "_temp"
    tmp.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    n_task_ep = int(args.n_ep or cfg["task"]["n_episodes"])
    meas_steps = int(cfg["task"]["measured_steps"])
    WARMUP = int(cfg["task"]["warmup_steps"])
    tail_frac = float(cfg["task"]["tail_frac"])
    online_seed = seed + int(cfg["data"].get("online_seed_offset", 2000))

    print("[29] " + "=" * 78)
    print("[29] ★ X45 相位效应的边界归因：tail(=X44) / core(去首尾段) / core_fixed(位置固定)")
    print(f"[29]   n_ep={n_task_ep}  meas_steps={meas_steps}  warmup={WARMUP}  "
          f"PER=0（沿用 X44，沿用同一模型缓存）")
    print("[29]   预写：H_c1『tail>5% 且 core≤5%』⇒ 边界归因；H_c3 core_fixed≈core ⇒ 真实相位效应")

    # ============================================================ 1) 环境 / 模型
    wind_amp = float(cfg["env"].get("wind_amp", 0.0))
    env = make_env(cfg["env"]["id"], seed=seed,
                   noise_std=float(cfg["env"]["noise_std"]),
                   max_steps=int(cfg["env"]["max_steps"]),
                   wind_amp=wind_amp)
    cc = cfg["controller"]
    ctrl = PDRelativeController(dt=env.dt, omega_n=float(cc["omega_n"]),
                                zeta=float(cc["zeta"]), kappa=float(env.kappa),
                                a_max=float(env.a_max))
    obs_dim = int(env.obs_dim)

    # ★★ 优先复用 X44 的缓存（存在即 load）⇒ 与 X44 逐位可比；否则新建
    #   ★ 冒烟必须用**另一个缓存名**：否则 `--quick`（30 epochs）会把 X44 的
    #     `x44_model.pt`（150 epochs）**覆盖掉**，正式跑就用了错的模型（铁律 16 同一家族）。
    cache = tmp / (X44_CACHE if not args.quick else "x45_smoke_model.pt")
    model = GaussianWorldModel(
        obs_dim=obs_dim, act_dim=int(env.act_dim),
        latent_dim=int(cfg["model"]["latent_dim"]),
        hidden=int(cfg["model"]["hidden"]),
        discrete_act=bool(env.is_discrete),
        logvar_init=float(cfg["model"].get("logvar_init", 0.0))).to(device)
    if cache.is_file() and not args.retrain:
        model.load_state_dict(torch.load(cache, map_location=device))
        print(f"[29] 载入模型缓存 {cache}" + ("（= X44 的，逐位可比）" if not args.quick else ""))
    else:
        n_ep = int(cfg["data"]["n_episodes"])
        train_all = collect_controlled_episodes(
            env, ctrl, n_episodes=n_ep, seed=seed,
            max_steps=int(cfg["env"]["max_steps"]))
        train_eps, val_eps = split_episodes(train_all, float(cfg["train"]["val_ratio"]), seed)
        tr = tuple(torch.as_tensor(x) for x in transitions_from_episodes(train_eps))
        va = tuple(torch.as_tensor(x) for x in transitions_from_episodes(val_eps))
        hist = train_world_model(model, tr, va, cfg, device, verbose=False)
        torch.save(model.state_dict(), cache)
        print(f"[29] 训练完成：params={count_params(model):,} "
              f"val={hist['val_total'][-1]:.6f} ({time.time() - t0:.0f}s) ⇒ 已缓存")

    ev_seed = seed + int(cfg["data"]["eval_seed_offset"])
    ev_env = make_env(cfg["env"]["id"], seed=ev_seed,
                      noise_std=float(cfg["env"]["noise_std"]),
                      max_steps=int(cfg["env"]["max_steps"]),
                      wind_amp=wind_amp)
    eval_eps = collect_controlled_episodes(
        ev_env, ctrl, n_episodes=max(30, int(cfg["data"]["n_episodes"]) // 3),
        seed=ev_seed, max_steps=int(cfg["env"]["max_steps"]))
    ev_env.close()
    allobs = np.concatenate([e["obs"][WARMUP:] for e in eval_eps], axis=0).astype(np.float64)
    var_g = float(allobs.var())
    print(f"[29]   var_g={var_g:.10f}（正式档应逐位 = X44 的 37.01283052768303）")

    def run(sched, with_traces=False):
        return run_closed_loop_control(
            env, model, ctrl, sched, n_episodes=n_task_ep, seed=online_seed,
            max_steps=meas_steps, device=device, estimator="model",
            var_g=var_g, label=getattr(sched, "__name__", "sched"),
            tail_frac=tail_frac, warmup_steps=WARMUP, return_traces=with_traces)

    # ============================================================ 2) X44 参照（S_x1）
    x44_path = out / X44_JSON
    x44 = json.loads(x44_path.read_text(encoding="utf-8")) if x44_path.is_file() else None
    if x44 is None:
        print(f"[29]  ⚠ 未找到 {x44_path} ⇒ S_x1 的『逐位复现 X44』降级为**仅自洽检查**")
    else:
        print(f"[29]    X44 参照：{x44_path.name}（var_g={x44['var_g']:.10f}）")

    # ============================================================ 3) 相位全档扫描
    print("[29] " + "-" * 78)
    print("[29] Part A · 相位**全档**扫描（Δ=0..T−1），四口径在同一次运行上重算")
    T_list = [int(x) for x in (args.periods.split(",") if args.periods else
                               ([8] if args.quick else [8, 16]))]
    rows = []
    checks = []
    BUDGET_TOL = 0.05
    for T in T_list:
        N = n_tx_budget(T, meas_steps)
        margin = 3 * T                        # ★ > 最大尾静默（M−1−(N−1)T）⇒ 静默被完全去掉
        # ★ S_x2 恒等式（机器核对，逐 Δ）：线性表 头静默 = Δ、尾静默 = M−1−Δ−(N−1)T
        for d in range(T):
            ts_d = plan_times("phase", T, d, meas_steps, N)
            h, tl = boundary_silence(ts_d, meas_steps)
            exp_tail = meas_steps - 1 - d - (N - 1) * T
            if h != d or tl != exp_tail:
                raise AssertionError(
                    f"★ S_x2 未通过：T={T} Δ={d} 实测(头,尾)=({h},{tl}) ≠ 解析({d},{exp_tail})")
        if margin <= max(meas_steps - 1 - (N - 1) * T, 0):
            raise AssertionError(
                f"★ margin={margin} ≤ 最大尾静默 ⇒ core_fixed 没去掉边界，诊断会失真")
        # ★ S_x3 循环网格的 tail 幅度恒为 T−1（数值坐实「循环化不解耦」）
        M_cyc = (meas_steps // T) * T          # 取能被 T 整除的最大窗口
        cyc_tails = [boundary_silence(cyclic_grid(T, M_cyc, d), M_cyc)[1] for d in range(T)]
        if max(cyc_tails) - min(cyc_tails) != T - 1:
            raise AssertionError(
                f"★ S_x3 未通过：循环网格 tail 幅度={max(cyc_tails) - min(cyc_tails)} ≠ T−1")
        print(f"[29]   T={T}  N={N}  margin={margin}  S_x2/S_x3 ✓（线性与循环的 tail 幅度"
              f"都 = {T - 1} ⇒ 循环化不解耦）")

        base_times = plan_times("phase", T, 0, meas_steps, N)
        r_base = run(lookup_schedule(base_times, f"phase(T={T},d=0)"), with_traces=True)
        exp_base = expected_attempts(base_times, r_base["ep_lens"])
        if int(r_base["n_tx"]) != exp_base:
            raise AssertionError(
                f"★ S_t1 未通过（基准）：n_tx={r_base['n_tx']} ≠ 反算 {exp_base}")
        lb = list(r_base["ep_lens"])
        b_full, _ = metric_over(r_base["ep_dists"], base_times, meas_steps, "full")
        b_tail, _ = metric_over(r_base["ep_dists"], base_times, meas_steps, "tail", tail_frac)
        b_core, n_cb = metric_over(r_base["ep_dists"], base_times, meas_steps, "core")
        b_cfix, _ = metric_over(r_base["ep_dists"], base_times, meas_steps, "core_fixed",
                                margin=margin)
        # ★ S_x1a：脚本 tail 必须 == control.py 报的 mean_dist_tail（逐位）
        if abs(b_tail - float(r_base["mean_dist_tail"])) > 1e-12:
            raise AssertionError(
                f"★ S_x1a 未通过：脚本 tail={b_tail!r} ≠ control 报的 "
                f"{r_base['mean_dist_tail']!r} ⇒ metric_over 的口径没接线")
        print(f"[29]   T={T}  基准 Δ=0：nmse={r_base['est_nmse']:.4f}  tail={b_tail:.4f}  "
              f"core={b_core:.4f}  cfix={b_cfix:.4f}  full={b_full:.4f}  "
              f"L̄={np.mean(lb):.1f}/{meas_steps}")
        rows.append({"kind": "base", "T": T, "delta": 0,
                     "head": 0, "tail_silence": boundary_silence(base_times, meas_steps)[1],
                     "est_nmse": float(r_base["est_nmse"]),
                     "dist_tail": float(b_tail), "dist_core": float(b_core),
                     "dist_core_fixed": float(b_cfix), "dist_full": float(b_full),
                     "n_core": int(n_cb), "n_tx": int(r_base["n_tx"]),
                     "tx_rate": float(r_base["tx_rate"])})

        for d in range(1, T):
            ts = plan_times("phase", T, d, meas_steps, N)
            h, tl = boundary_silence(ts, meas_steps)
            r = run(lookup_schedule(ts, f"phase(T={T},d={d})"), with_traces=True)
            exp = expected_attempts(ts, r["ep_lens"])
            if int(r["n_tx"]) != exp:
                raise AssertionError(
                    f"★ S_t1 未通过：T={T} Δ={d} n_tx={r['n_tx']} ≠ 反算 {exp}")
            v_full, _ = metric_over(r["ep_dists"], ts, meas_steps, "full")
            v_tail, _ = metric_over(r["ep_dists"], ts, meas_steps, "tail", tail_frac)
            v_core, n_c = metric_over(r["ep_dists"], ts, meas_steps, "core")
            v_cfix, _ = metric_over(r["ep_dists"], ts, meas_steps, "core_fixed", margin=margin)
            if abs(v_tail - float(r["mean_dist_tail"])) > 1e-12:
                raise AssertionError(
                    f"★ S_x1a 未通过：T={T} Δ={d} 脚本 tail={v_tail!r} ≠ "
                    f"control 报的 {r['mean_dist_tail']!r}")
            rate_dev = abs(float(r["tx_rate"]) / float(r_base["tx_rate"]) - 1.0)
            budget_ok = bool(rate_dev <= BUDGET_TOL)
            rows.append({"kind": "phase", "T": T, "delta": int(d),
                         "head": int(h), "tail_silence": int(tl),
                         "est_nmse": float(r["est_nmse"]),
                         "dist_tail": float(v_tail), "dist_core": float(v_core),
                         "dist_core_fixed": float(v_cfix), "dist_full": float(v_full),
                         "n_core": int(n_c), "n_tx": int(r["n_tx"]),
                         "tx_rate": float(r["tx_rate"]),
                         "budget_ok": budget_ok, "rate_dev": float(rate_dev)})
            flag = "" if budget_ok else "  ⚠不可评估(预算偏差)"
            print(f"[29]     Δ={d:<3d} 头静默={h:<3d} 尾静默={tl:<3d}  "
                  f"nmse={r['est_nmse']:.4f}  tail={v_tail:.4f}  core={v_core:.4f}  "
                  f"cfix={v_cfix:.4f}  full={v_full:.4f}  (core集数 {n_c}/{len(r['ep_dists'])}){flag}")
        checks.append({"T": T, "N": N, "margin": margin,
                       "cyc_tail_span": int(max(cyc_tails) - min(cyc_tails))})
    env.close()

    # ============================================================ 4) 判据结算
    print("[29] " + "-" * 78)
    print("[29] 判据结算")
    verdict = {}
    for T in T_list:
        base = next(r for r in rows if r["T"] == T and r["kind"] == "base")
        pert = [r for r in rows if r["T"] == T and r["kind"] == "phase" and r["budget_ok"]]

        def max_dev(key):
            return max(abs(r[key] / base[key] - 1.0) for r in pert) if pert else float("nan")

        dev_tail = max_dev("dist_tail")
        dev_core = max_dev("dist_core")
        dev_cfix = max_dev("dist_core_fixed")
        dev_full = max_dev("dist_full")

        # ★ S_x1b：与 X44 逐位复现（相位档 Δ ∈ X44 扫过的档）
        x44_max_diff = None
        # ★ 冒烟（--quick）参数与 X44 正式档不同 ⇒ 不可比，跳过 S_x1b（否则必然 raise）
        if x44 is not None and not args.quick:
            diffs = []
            for r in pert:
                ref = [q for q in x44["rows"] if q.get("T") == T
                       and q.get("mode") == "phase" and int(q.get("param", -1)) == r["delta"]]
                if not ref:
                    continue
                diffs.append(abs(float(ref[0]["est_nmse"]) - r["est_nmse"]))
                diffs.append(abs(float(ref[0]["mean_dist_tail"]) - r["dist_tail"]))
            x44_max_diff = max(diffs) if diffs else None
            if x44_max_diff is not None and x44_max_diff > 1e-9:
                raise AssertionError(
                    f"★ S_x1b 未通过：T={T} 与 X44 相位档最大逐位差={x44_max_diff:.3e} > 1e-9"
                    " ⇒ 本号与 X44 不是同一次设置，逐位可比性不成立")

        v = {
            "T": T,
            "base_dist_tail": base["dist_tail"], "base_dist_core": base["dist_core"],
            "base_dist_core_fixed": base["dist_core_fixed"],
            "base_est_nmse": base["est_nmse"],
            "n_phase_evaluable": len(pert),
            "phase_max_dev_tail": float(dev_tail),
            "phase_max_dev_core": float(dev_core),
            "phase_max_dev_core_fixed": float(dev_cfix),
            "phase_max_dev_full": float(dev_full),
            "ratio_tail_over_core_fixed": (float(dev_tail / dev_cfix)
                                           if dev_cfix and np.isfinite(dev_cfix) and dev_cfix > 1e-12
                                           else float("nan")),
            "x44_max_abs_diff": (None if x44_max_diff is None else float(x44_max_diff)),
            "H_c1_boundary_attribution": bool(dev_tail > 0.05 and dev_core <= 0.05),
            "H_c2_core_gt_5pct": bool(dev_core > 0.05),
            "H_c3_core_fixed_gt_5pct": bool(dev_cfix > 0.05),
        }
        verdict[str(T)] = v
        print(f"[29]   T={T}:  tail {dev_tail:.2%}  core {dev_core:.2%}  "
              f"core_fixed {dev_cfix:.2%}  full {dev_full:.2%}  (可评估 {len(pert)}/{T - 1} 档)")
        if x44_max_diff is not None:
            print(f"[29]     ✓ S_x1b 与 X44 逐位复现：最大绝对差 {x44_max_diff:.3e}")
        print(f"[29]     H_c1 边界归因={'成立' if v['H_c1_boundary_attribution'] else '不成立'}"
              f" | H_c2 core>5%={'是' if v['H_c2_core_gt_5pct'] else '否'}"
              f" | H_c3 core_fixed>5%={'是（⇒ 真实相位效应）' if v['H_c3_core_fixed_gt_5pct'] else '否（⇒ 位置平移在起作用）'}"
              f" | tail/cfix 放大 {v['ratio_tail_over_core_fixed']:.2f}×")

    # ============================================================ 5) 图 + JSON
    stem = args.tag
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for T in T_list:
        b = next(r for r in rows if r["T"] == T and r["kind"] == "base")
        pert = sorted([r for r in rows if r["T"] == T and r["kind"] == "phase"
                       and r["budget_ok"]], key=lambda r: r["delta"])
        ds = [0] + [r["delta"] for r in pert]
        for key, col, mk, lb in (("dist_tail", PALETTE["blue"], "o", "tail (last 25%, = X44)"),
                                 ("dist_core", PALETTE["red"], "s", "core (drop edge segs)"),
                                 ("dist_core_fixed", PALETTE["green"], "D", "core_fixed (fixed margin)"),
                                 ("dist_full", PALETTE["purple"], "^", "full")):
            ys = [1.0] + [r[key] / b[key] for r in pert]
            axes[0].plot(ds, ys, marker=mk, color=col, lw=1.4, ms=4, label=f"T={T} · {lb}")
        axes[1].plot(ds[1:], [r["head"] for r in pert], marker="v", color=PALETTE["orange"],
                     lw=1.4, ms=4, label=f"T={T} head silence")
        axes[1].plot(ds[1:], [r["tail_silence"] for r in pert], marker="^", color=PALETTE["grey"],
                     lw=1.4, ms=4, label=f"T={T} tail silence")
    axes[0].axhline(1.0, color="k", lw=0.9, alpha=0.5)
    axes[0].axhline(1.05, color="r", lw=0.8, ls="--", alpha=0.6)
    axes[0].axhline(0.95, color="r", lw=0.8, ls="--", alpha=0.6)
    axes[0].set_xlabel(r"phase offset $\Delta$")
    axes[0].set_ylabel("task distance / base ($\\Delta=0$)")
    axes[0].set_title("phase sensitivity by metric", fontsize=11)
    axes[0].legend(fontsize=7)
    axes[1].set_xlabel(r"phase offset $\Delta$")
    axes[1].set_ylabel("silence length (steps)")
    axes[1].set_title("head / tail silence vs phase", fontsize=11)
    axes[1].legend(fontsize=7)
    for T in T_list:
        b = next(r for r in rows if r["T"] == T and r["kind"] == "base")
        pert = [r for r in rows if r["T"] == T and r["kind"] == "phase" and r["budget_ok"]]
        axes[2].scatter([abs(r["dist_core_fixed"] / b["dist_core_fixed"] - 1.0) for r in pert],
                        [abs(r["dist_tail"] / b["dist_tail"] - 1.0) for r in pert],
                        s=42, alpha=0.85, label=f"T={T}")
    lim = 0.15
    axes[2].plot([0, lim], [0, lim], color="k", lw=0.9, ls=":", alpha=0.6)
    axes[2].set_xlim(0, lim)
    axes[2].set_ylim(0, lim)
    axes[2].set_xlabel("|core_fixed dev|  (position-fixed)")
    axes[2].set_ylabel("|tail dev|  (= X44 metric)")
    axes[2].set_title("above diagonal => tail metric inflated by edge", fontsize=11)
    axes[2].legend(fontsize=8)
    fig.suptitle(f"X45 phase-vs-boundary attribution  tag={args.tag}  T={T_list}  "
                 f"PER=0  n_ep={n_task_ep}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_fig(fig, out / f"{stem}.png")
    plt.close(fig)

    payload = {
        "experiment": "X45",
        "title": "相位效应的边界归因：tail(=X44) / core / core_fixed / full 四口径",
        "config": args.config, "seed": seed, "n_task_ep": n_task_ep,
        "meas_steps": meas_steps, "warmup_steps": WARMUP, "per": 0.0,
        "tail_frac": tail_frac, "var_g": var_g,
        "model_params": int(count_params(model)),
        "model_cache": str(cache),
        "reused_x44_cache": bool(not args.quick and (tmp / X44_CACHE).is_file()),
        "quick": bool(args.quick), "T_list": T_list,
        "rows": rows, "verdict": verdict, "checks": checks,
        "note_metric_core": "core = 去掉首段 [t1,t2) 与末段 [tK,M]（位置随 Δ 平移）",
        "note_metric_core_fixed": "core_fixed = 去掉固定 3T 步首尾（位置与相位无关）⇒ 诊断用",
        "note_why_not_cyclic": "循环化只把尾部静默基线挪 T（比线性多发一次），幅度恒为 T−1 ⇒ 零解耦",
    }
    (out / f"{stem}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[29] 产物：{out / (stem + '.json')} / {stem}.png   总耗时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
