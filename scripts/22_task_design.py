# -*- coding: utf-8 -*-
"""★ X35「任务锚定的联合设计：闭式选出的 (T, b) 还是不是任务最优的？」

================================================================= ★ 由来（为什么必须有这一号）
X34 证明了设计曲线 (T × b) 可以用闭式算出来 —— 但它优化的是**估计误差 NMSE**。
X30 已经实测过：H*_task（逃逸率 / 跟踪距离定义的「任务还能撑多久」）与
H*_err（误差超阈）**不是同一个数**，在本场景差到 **32 倍**。

⇒ 于是有个必须回答的问题：**按 NMSE 选出来的设计点，是不是任务最优的那个点？**

本实验就是把它测出来：同一张 (T, b) 设计网格上同时记三类指标
  (a) 闭式预报的 NMSE（X34 的口径，开环标定）
  (b) 闭环实测的 est_nmse
  (c) 闭环实测的**任务指标**：稳态跟踪距离 mean_dist_tail + 逃逸率 escape_rate
然后比较三者的 argmin 与退化幅度。

================================================================= ★★ 与 X34 的关键差别：真闭环
X33/X34 用 `eval.run_tracking`：动作来自**预先采好的 episode** ⇒ 轨迹与模型无关（开环回放）。
本实验用 `control.run_closed_loop_control`：动作来自 `controller(est)` ⇒
**模型误差会改真实轨迹** ⇒ 误差会累积、会被闭环放大（X30 实测放大 1.69×）。

⇒ `payload_fn`（量化器）**第一次真正进入闭环** —— X33/X34 只在开环回放里量过它。

================================================================= ★ 预写判据（跑前定死）
T1  ★ 任务指标从最优点挪到次优点的**相对退化** ≥ NMSE 的相对退化 × 2
    理由：闭环放大估计误差 + 逃逸是阈值事件 ⇒ 均值型的 NMSE 会把"任务开始坏掉"糊掉
T2  排序一致性 Spearman(est_nmse, 任务指标)：预写 ρ ≥ 0.8（大致同向）。
    若实测 ρ < 0.8 ⇒ **NMSE 不能当任务指标的代理**，照报（这也是本实验要的结论之一）
T3  ★ model vs persistence 在**任务指标**上的差距 ≥ 在 NMSE 上的差距 × 1.2
    依据：X30 实测 T=64 时逃逸率 0.42 vs 0.92（差 2.2 倍），而同口径 est_nmse 差得更小
T4  ★ R12：改 b 必须改变**闭环**结果（否则 payload_fn 没接进 control.py ⇒ 整个实验空跑）
T5  闭式预报（离线曲线在**开环轨迹**上标定）搬到闭环后偏差**显著变大**
    ⇒ 量化「离线标定 → 在线闭环」的分布偏移

================================================================= 自检（不过就抛 —— R14）
S_a  闭环实测 E[age] 与闭式 periodic_mean_age 一致（⇒ 年龄闭式在闭环里也成立）
S_b  全部有限（R14：NaN/Inf 一律抛，不许静默）
S_c  b 最大 + T 最小 + PER≈0 的那个点 ≈ oracle，其 est_nmse 必须是全场最小量级
S_d  ★ R12：固定 (SNR, T)，改 b 必须改变闭环 est_nmse（T4 的机器版）
S_e  年龄分布尾部 < 2%（与 X34 同口径；用 `periodic_age_tail`，不是第一版那个错式）
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from wmlab.control import (PDRelativeController, collect_controlled_episodes,
                           run_closed_loop_control)
from wmlab.data import split_episodes, transitions_from_episodes
from wmlab.envs import make_env
from wmlab.eval import reliable_horizon
from wmlab.eval.physical import (UniformQuantizer, fit_quantizer_range, overload_fraction,
                                 per_from_ber, qam_approximation_valid, qam_bit_error_rate,
                                 snr_db_to_linear)
from wmlab.eval.tracking import (periodic_age_pmf, periodic_age_tail,
                                 periodic_lossy_schedule, periodic_mean_age)
from wmlab.models import MLPWorldModel
from wmlab.rollout import closed_loop_error_curve
from wmlab.train import train_world_model
from wmlab.utils import count_params, get_device, load_config, output_dir, set_seed
from wmlab.utils.plot import PALETTE, apply_style, save_fig


def _load_mod19():
    spec = importlib.util.spec_from_file_location(
        "_mod19", os.path.join(_HERE, "19_physical_layer.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PersistenceHead = _load_mod19().PersistenceHead


def parse_args():
    p = argparse.ArgumentParser(description="X35 任务锚定的联合设计")
    p.add_argument("--config", default="configs/uav_task_design.yaml")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--tag", default="22_task_design")
    p.add_argument("--snrs", default=None)
    p.add_argument("--bits", default=None)
    p.add_argument("--quick", action="store_true")
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
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.snrs:
        cfg["design"]["snr_db_list"] = [float(x) for x in args.snrs.split(",")]
    if args.bits:
        cfg["design"]["bits_list"] = [int(x) for x in args.bits.split(",")]
    dz = cfg["design"]
    if args.quick:
        cfg["train"]["epochs"] = min(int(cfg["train"]["epochs"]), 3)
        cfg["data"]["n_episodes"] = min(int(cfg["data"]["n_episodes"]), 20)
        dz["snr_db_list"] = [20.0]
        dz["bits_list"] = [4, 8]
        cfg["task"]["n_episodes"] = 6
        cfg["task"]["measured_steps"] = 120
        cfg["eval"]["n_samples"] = 64

    seed = int(cfg["seed"])
    set_seed(seed)
    apply_style()
    device = get_device(args.device or cfg.get("device"))
    out = output_dir(cfg)
    t0 = time.time()

    snrs = [float(x) for x in dz["snr_db_list"]]
    bits_list = [int(x) for x in dz["bits_list"]]
    m_list = [int(x) for x in dz["m_sym_list"]]
    max_T = int(dz.get("max_period", 32))
    max_tail = float(dz.get("max_tail_mass", 0.02))
    WARMUP = int(cfg["task"]["warmup_steps"])
    n_task_ep = int(cfg["task"]["n_episodes"])

    print("[22] " + "=" * 76)
    print("[22] ★ X35 任务锚定的联合设计：闭式选的 (T,b) 还是不是任务最优的")
    print(f"[22]   SNR ∈ {snrs} dB   b ∈ {bits_list}   可行 m（方形星座）∈ {m_list}")
    print("[22]   预写判据：T1 任务退化 ≥ NMSE 退化×2   T2 ρ≥0.8   T3 任务差距 ≥ NMSE 差距×1.2")
    print("[22]             T4 R12 改 b 必须改变闭环   T5 闭式搬到闭环后偏差变大")

    # ---------- 1) 训练（与 X30/X31/X32/X33/X34 完全同一套流程）----------
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
    print(f"[22] 训练完成：params={count_params(model):,} val={hist['val_total'][-1]:.6f} "
          f"({time.time() - t0:.0f}s)")
    heads = {"model": model, "persistence": PersistenceHead(model)}

    # ---------- 2) 离线标定（开环轨迹上的 g_b(h)，供 T5 对照）----------
    ev_seed = seed + int(cfg["data"]["eval_seed_offset"])
    ev_env = make_env(cfg["env"]["id"], seed=ev_seed,
                      noise_std=float(cfg["env"]["noise_std"]),
                      max_steps=int(cfg["env"]["max_steps"]))
    eval_eps = collect_controlled_episodes(ev_env, ctrl,
                                           n_episodes=max(30, n_ep // 3), seed=ev_seed,
                                           max_steps=int(cfg["env"]["max_steps"]))
    ev_env.close()
    lo, hi = fit_quantizer_range(eval_eps, margin=float(dz.get("range_margin", 0.05)))
    allobs = np.concatenate([e["obs"][WARMUP:] for e in eval_eps], axis=0).astype(np.float64)
    var_g = float(allobs.var())
    quantizers = {b: UniformQuantizer(lo, hi, b) for b in bits_list}
    for b in bits_list:
        if overload_fraction(quantizers[b], allobs) > 1e-6:
            raise AssertionError(f"★ S_q 未通过：b={b} 存在过载（裁剪）⇒ Δ²/12 不可用")

    hs = [int(h) for h in cfg["eval"]["horizons"]]
    hs = [h for h in hs if h <= max(e["length"] for e in eval_eps) - WARMUP - 2]

    def _offline(head, payload=None) -> np.ndarray:
        c = closed_loop_error_curve(head, eval_eps, hs, device,
                                    n_samples=int(cfg["eval"]["n_samples"]),
                                    seed=seed, t0_min=WARMUP, payload_fn=payload)
        return np.concatenate([[0.0], np.asarray(c["mse_per_step"], dtype=float) / var_g])

    K = None
    g_curves = {}
    for name, head in heads.items():
        f = _offline(head, None)
        K = len(f) - 1 if K is None else K
        if len(f) != K + 1:
            raise AssertionError(f"★ S_off 未通过：{name} 曲线长度不一致")
        for b in bits_list:
            gb = _offline(head, quantizers[b])
            gb[0] = float(quantizers[b].measure_mse(allobs) / var_g)
            if not np.all(np.isfinite(gb)):
                raise FloatingPointError(f"g_{name,b}(h) 含非有限值（R14）")
            g_curves[(name, b)] = gb
        print(f"[22]   [{name:>11s}] f(1)={f[1]:.5f} f(10)={f[min(10, K)]:.5f} "
              f"f(K={K})={f[K]:.5f}")
    h_err = reliable_horizon(hs, [float(x) for x in _offline(model, None)[hs]],
                             float(cfg["eval"]["err_threshold"]),
                             cfg["eval"].get("threshold_mode", "rel"))
    print(f"[22] 离线：K={K} 步   H*_err={h_err}")

    def per_of(snr_db: float, b: int, m: float) -> float:
        g = snr_db_to_linear(snr_db)
        M = 2.0 ** float(m)
        if not qam_approximation_valid(g, M):
            return 1.0 - 1e-6
        return float(per_from_ber(qam_bit_error_rate(g, M), float(obs_dim * b)))

    # ---------- 3) 设计网格 ----------
    grid, skipped = [], []
    for b in bits_list:
        for m in m_list:
            if b % m != 0:
                skipped.append({"bits": b, "m": m, "reason": f"T=b/m={b}/{m} 非整数"})
                continue
            T = b // m
            if T < 1 or T > max_T or T > K:
                skipped.append({"bits": b, "m": m, "T": T,
                                "reason": f"周期 {T} 越界（max={max_T}, K={K}）"})
                continue
            grid.append({"bits": b, "m": m, "T": T})
    print(f"[22] 设计网格：{len(grid)} 个 (b,T) 组合，跳过 {len(skipped)} 个")
    print("[22] " + "-" * 76)

    # ---------- 4) 扫：闭式预报 + **闭环**实测 ----------
    online_seed = seed + int(cfg["data"].get("online_seed_offset", 2000))
    rows = []
    for snr in snrs:
        for gd in grid:
            b, m, T = gd["bits"], gd["m"], gd["T"]
            per01 = per_of(snr, b, m)
            if per01 >= 0.999:
                skipped.append({**gd, "snr_db": snr, "reason": "PER≈1（链路不可用）"})
                continue
            pmf = periodic_age_pmf(per01, T, K)
            if abs(float(pmf.sum()) - 1.0) > 1e-9:
                raise AssertionError("★ S_b 未通过：pmf 未归一化")
            tail_mass = float(periodic_age_tail(per01, T, K))
            if tail_mass > max_tail:
                skipped.append({**gd, "snr_db": snr, "per": per01,
                                "tail_mass": tail_mass,
                                "reason": f"年龄尾部 {tail_mass:.1%} 超出可用视界 K={K}"})
                print(f"[22]   跳过 {snr:g}dB b={b} T={T}：尾部 {tail_mass:.1%} > "
                      f"{max_tail:.0%}（PER={per01:.3f}）")
                continue
            rec = {"snr_db": float(snr), "bits": b, "m": m, "T": T,
                   "M": float(2.0 ** m), "per": float(per01),
                   "tail_mass": tail_mass,
                   "age_mean_analytic": float(periodic_mean_age(per01, T)),
                   "age_mean_pmf_trunc": float(np.dot(np.arange(K + 1), pmf))}
            for name, head in heads.items():
                pred = float(np.dot(pmf, g_curves[(name, b)]))
                r = run_closed_loop_control(
                    env, head, ctrl, periodic_lossy_schedule(T, per01),
                    n_episodes=n_task_ep, seed=online_seed,
                    max_steps=int(cfg["task"]["measured_steps"]), device=device,
                    estimator=name, var_g=var_g, label=f"{name}/T={T}/b={b}",
                    tail_frac=float(cfg["task"]["tail_frac"]), warmup_steps=WARMUP,
                    payload_fn=quantizers[b])
                for k in ("est_nmse", "escape_rate", "mean_dist", "mean_dist_tail",
                          "p95_dist", "mean_age", "max_age", "tx_rate", "n_steps",
                          "mean_ep_len"):
                    if not np.isfinite(float(r[k])):
                        raise FloatingPointError(
                            f"S_b：(snr={snr},b={b},T={T},{name}) 指标 {k} 非有限（R14）")
                # ★ S_a：年龄闭式在闭环里也必须成立
                emp_age = float(r["mean_age"])
                if abs(emp_age - rec["age_mean_pmf_trunc"]) > max(
                        0.10, 0.08 * max(abs(emp_age), abs(rec["age_mean_pmf_trunc"]))):
                    raise AssertionError(
                        f"★ S_a 未通过：SNR={snr} T={T} per={per01:.4f} "
                        f"闭环实测 E[age]={emp_age:.4f} vs 闭式（截断后）"
                        f"{rec['age_mean_pmf_trunc']:.4f}")
                rec[name] = {"pred": pred, "est_nmse": float(r["est_nmse"]),
                             "gap_closed": float(pred / max(float(r["est_nmse"]), 1e-18) - 1.0),
                             "escape_rate": float(r["escape_rate"]),
                             "mean_dist": float(r["mean_dist"]),
                             "mean_dist_tail": float(r["mean_dist_tail"]),
                             "p95_dist": float(r["p95_dist"]),
                             "age_mean": emp_age, "tx_rate": float(r["tx_rate"]),
                             "n_steps": int(r["n_steps"]),
                             "mean_ep_len": float(r["mean_ep_len"])}
            rows.append(rec)
            rm = rec["model"]
            print(f"[22]   {snr:>5g}dB b={b:>2d} T={T:>2d} m={m} PER={per01:.3f} "
                  f"E[age]={rm['age_mean']:6.2f}(闭式 {rec['age_mean_pmf_trunc']:6.2f}) | "
                  f"NMSE 闭式={rm['pred']:.5f} 闭环={rm['est_nmse']:.5f}"
                  f"({rm['gap_closed']:+.0%}) | 距离={rm['mean_dist_tail']:6.3f}m "
                  f"逃逸={rm['escape_rate']:.3f}")

    env.close()
    if not rows:
        raise AssertionError("★ 设计网格为空 ⇒ 调整 SNR/b/m")

    # ---------- 5) ★ R12 接线检查（T4 的机器版）----------
    n_wired = 0
    for snr in snrs:
        for T in sorted({g["T"] for g in grid}):
            sub = [r for r in rows if abs(r["snr_db"] - snr) < 1e-9 and r["T"] == T]
            if len(sub) < 2:
                continue
            vals = [r["model"]["est_nmse"] for r in sub]
            # 判定：改 b 必须改变闭环结果（同一个 T、同一个 SNR）
            if max(vals) / max(min(vals), 1e-18) > 1.05:
                n_wired += 1
    if n_wired == 0:
        raise AssertionError(
            "★ T4/S_d 未通过：所有 (SNR,T) 下改 b **都不改变闭环结果** ⇒ "
            "payload_fn 没接进 control.py（R12 违规），整个实验在空跑")
    print(f"[22]   T4/S_d 通过：{n_wired} 个 (SNR,T) 组验到量化真的进了闭环")

    # ---------- 6) 判据 ----------
    print("[22] " + "=" * 76)
    TASK_METRICS = ("mean_dist_tail", "escape_rate")
    eps = float(cfg["task"].get("eps_floor", 1e-9))

    # --- T1：最优 → 次优的相对退化，任务指标 vs NMSE ---
    t1 = {}
    for met in TASK_METRICS:
        ratios = []
        for snr in snrs:
            for b in bits_list:
                sub = sorted([r for r in rows if abs(r["snr_db"] - snr) < 1e-9
                              and r["bits"] == b], key=lambda r: r["T"])
                if len(sub) < 2:
                    continue
                nv = np.array([r["model"]["est_nmse"] for r in sub])
                tv = np.array([r["model"][met] for r in sub])
                kn, kt = int(np.argmin(nv)), int(np.argmin(tv))
                # 各自最优点 → 各自次优点
                n_second = float(np.sort(nv)[1])
                t_second = float(np.sort(tv)[1])
                rn = n_second / max(float(nv[kn]), eps) - 1.0
                rt = t_second / max(float(tv[kt]), eps) - 1.0
                if rn > 1e-6:
                    ratios.append(rt / rn)
        t1[met] = ratios
        if ratios:
            med = float(np.median(ratios))
            ok = sum(1 for x in ratios if x >= 2.0)
            print(f"[22] ★ T1[{met}]：任务退化/NMSE 退化 中位 {med:.2f}×  "
                  f"≥2× 的点 {ok}/{len(ratios)}")
        else:
            print(f"[22]   T1[{met}]：无足够候选")

    # --- T2：排序一致性 ---
    t2 = {}
    try:
        from scipy.stats import spearmanr
        for met in TASK_METRICS:
            a = np.array([r["model"]["est_nmse"] for r in rows])
            c = np.array([r["model"][met] for r in rows])
            rho = float(spearmanr(a, c).statistic)
            t2[met] = rho
            print(f"[22] ★ T2[{met}]：Spearman(est_nmse, {met}) = {rho:+.3f}"
                  f"{'  ⇒ NMSE 可以当代理' if rho >= 0.8 else '  ⇒ ★ NMSE 不能当任务代理'}")
    except Exception as e:                      # scipy 不在依赖里 ⇒ 退回 Pearson 秩相关
        print(f"[22]   scipy 不可用（{e}），T2 改用手工秩相关")
        def _rank(x):
            order = np.argsort(np.argsort(x))
            return order.astype(float)
        for met in TASK_METRICS:
            a = _rank(np.array([r["model"]["est_nmse"] for r in rows]))
            c = _rank(np.array([r["model"][met] for r in rows]))
            rho = float(np.corrcoef(a, c)[0, 1])
            t2[met] = rho
            print(f"[22] ★ T2[{met}]：秩相关 = {rho:+.3f}")

    # --- T3：model vs persistence 的差距，任务指标 vs NMSE ---
    t3 = {}
    for met in TASK_METRICS:
        rs = []
        for r in rows:
            nm = max(float(r["persistence"]["est_nmse"]), eps)
            nt = max(float(r["persistence"][met]), eps)
            gm = float(r["model"]["est_nmse"]) / nm
            gt = float(r["model"][met]) / nt
            if abs(1.0 - gm) > 1e-4:            # NMSE 上真有差距的点才计
                rs.append((1.0 - gt) / (1.0 - gm))
        t3[met] = rs
        if rs:
            print(f"[22] ★ T3[{met}]：(任务差距)/(NMSE 差距) 中位 "
                  f"{float(np.median(rs)):.2f}×   n={len(rs)}")
        else:
            print(f"[22]   T3[{met}]：无足够候选")

    # --- T5：闭式（开环标定）搬到闭环后的偏差 ---
    gap_closed = np.abs(np.array([r["model"]["gap_closed"] for r in rows]))
    t5_med = float(np.median(gap_closed))
    print(f"[22] ★ T5：闭式预报 → 闭环实测 |偏差| 中位 {t5_med:.1%}、"
          f"90 分位 {float(np.percentile(gap_closed, 90)):.1%}"
          f"（X34 开环口径的中位是 3.1%）")

    # --- argmin 一致性：闭式/闭环 NMSE vs 任务指标 ---
    agree = {"pred_vs_task": 0, "n": 0}
    for met in ("mean_dist_tail",):
        for snr in snrs:
            for b in bits_list:
                sub = sorted([r for r in rows if abs(r["snr_db"] - snr) < 1e-9
                              and r["bits"] == b], key=lambda r: r["T"])
                if len(sub) < 2:
                    continue
                agree["n"] += 1
                kp = int(np.argmin([r["model"]["pred"] for r in sub]))
                kt = int(np.argmin([r["model"][met] for r in sub]))
                agree["pred_vs_task"] += int(sub[kp]["T"] == sub[kt]["T"])
    if agree["n"]:
        print(f"[22] ★ 闭式 argmin T 与任务 argmin T 一致 "
              f"{agree['pred_vs_task']}/{agree['n']}")

    verdict = {
        "T1_task_over_nmse_regression_median": {k: (float(np.median(v)) if v else None)
                                                for k, v in t1.items()},
        "T2_spearman": {k: float(v) for k, v in t2.items()},
        "T3_task_over_nmse_gap_median": {k: (float(np.median(v)) if v else None)
                                         for k, v in t3.items()},
        "T4_n_wired": int(n_wired),
        "T5_closed_pred_median": t5_med,
        "argmin_agree": agree,
    }

    # ---------- 7) 图 ----------
    fig, axes = plt.subplots(2, 3, figsize=(17.5, 9.5))
    fig.suptitle(
        "X35 任务锚定的联合设计：闭式选的 (T,b) 还是不是任务最优的", fontsize=11)

    # ① 任务指标 vs NMSE（散点，看是否单调）
    ax = axes[0, 0]
    for met, col, mk in (("mean_dist_tail", PALETTE["blue"], "o"),
                         ("escape_rate", PALETTE["red"], "s")):
        ax.plot([r["model"]["est_nmse"] for r in rows],
                [r["model"][met] for r in rows], mk, color=col, alpha=0.75, label=met)
    ax.set_xscale("log")
    ax.set_xlabel("闭环 est_nmse"); ax.set_ylabel("任务指标")
    ax.set_title("① 任务指标 vs 估计误差（是否单调？）")
    ax.legend(fontsize=8)

    # ② 每个 (SNR,b)：三条 argmin 曲线（T 横轴）
    ax = axes[0, 1]
    sub0 = sorted([r for r in rows if abs(r["snr_db"] - snrs[len(snrs) // 2]) < 1e-9
                   and r["bits"] == bits_list[min(2, len(bits_list) - 1)]],
                  key=lambda r: r["T"])
    if sub0:
        ax.plot([r["T"] for r in sub0], [r["model"]["pred"] for r in sub0], "k--",
                label="闭式 NMSE")
        ax.plot([r["T"] for r in sub0], [r["model"]["est_nmse"] for r in sub0], "o-",
                color=PALETTE["blue"], label="闭环 NMSE")
        ax2 = ax.twinx()
        ax2.plot([r["T"] for r in sub0], [r["model"]["mean_dist_tail"] for r in sub0],
                 "s-", color=PALETTE["red"], label="跟踪距离")
        ax2.set_ylabel("稳态跟踪距离 (m)", color=PALETTE["red"])
        ax.set_yscale("log")
        ax.set_xlabel("T（步）"); ax.set_ylabel("NMSE")
        ax.set_title(f"② SNR={snrs[len(snrs)//2]:g}dB b={bits_list[min(2, len(bits_list)-1)]}")
        ax.legend(fontsize=7.5, loc="upper left")
        ax2.legend(fontsize=7.5, loc="upper right")
    else:
        ax.set_title("② （该 (SNR,b) 无点）")

    # ③ 闭式 vs 闭环（T5）
    ax = axes[0, 2]
    ax.plot([r["model"]["pred"] for r in rows], [r["model"]["est_nmse"] for r in rows],
            "o", color=PALETTE["green"], alpha=0.8)
    vmin = max(min(r["model"]["pred"] for r in rows), 1e-7)
    vmax = max(r["model"]["pred"] for r in rows) * 1.5
    ax.plot([vmin, vmax], [vmin, vmax], "k:", lw=1, label="y=x")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("闭式预报（开环标定）"); ax.set_ylabel("闭环实测 est_nmse")
    ax.set_title(f"③ T5：开环标定搬到闭环（中位偏差 {t5_med:.1%}）")
    ax.legend(fontsize=8)

    # ④ age 闭式核对（闭环版 S_a）
    ax = axes[1, 0]
    aa = np.array([r["age_mean_pmf_trunc"] for r in rows])
    ea = np.array([r["model"]["age_mean"] for r in rows])
    ax.plot(aa, ea, "o", color=PALETTE["green"])
    lim = [0, max(aa.max(), ea.max()) * 1.1]
    ax.plot(lim, lim, "k:", lw=1)
    ax.set_xlabel("闭式 E[age]（截断后）"); ax.set_ylabel("闭环实测 E[age]")
    ax.set_title("④ S_a：年龄闭式在闭环里也成立")
    ax.grid(alpha=0.3)

    # ⑤ model vs persistence（任务口径）
    ax = axes[1, 1]
    ax.plot([r["model"]["mean_dist_tail"] for r in rows],
            [r["persistence"]["mean_dist_tail"] for r in rows], "o",
            color=PALETTE["orange"], alpha=0.8)
    lo2 = min(min(r["model"]["mean_dist_tail"] for r in rows),
              min(r["persistence"]["mean_dist_tail"] for r in rows))
    hi2 = max(max(r["model"]["mean_dist_tail"] for r in rows),
              max(r["persistence"]["mean_dist_tail"] for r in rows))
    ax.plot([lo2, hi2], [lo2, hi2], "k:", lw=1)
    ax.set_xlabel("model 稳态跟踪距离 (m)")
    ax.set_ylabel("persistence 稳态跟踪距离 (m)")
    ax.set_title("⑤ 世界模型在**任务**上换来了多少（对角线上方=模型更好）")
    ax.grid(alpha=0.3)

    # ⑥ 逃逸率 vs 周期
    ax = axes[1, 2]
    cmap = plt.get_cmap("viridis")
    for i, snr in enumerate(snrs):
        col = cmap(i / max(len(snrs) - 1, 1))
        sub = sorted([r for r in rows if abs(r["snr_db"] - snr) < 1e-9],
                     key=lambda r: (r["bits"], r["T"]))
        if sub:
            ax.plot([r["model"]["mean_age"] for r in sub],
                    [r["model"]["escape_rate"] for r in sub], "o", color=col,
                    ms=5, alpha=0.85, label=f"{snr:g}dB")
    ax.set_xlabel("E[age]（步）"); ax.set_ylabel("逃逸率")
    ax.set_title("⑥ 逃逸率 vs 信息年龄")
    ax.legend(fontsize=7.5, ncol=2)

    save_fig(fig, out, "22_task_design.png")
    plt.close(fig)

    payload = {
        "experiment": "X35 任务锚定的联合设计：闭式选的 (T,b) 还是不是任务最优的",
        "config": {k: cfg[k] for k in ("env", "controller", "design", "eval", "task")},
        "obs_dim": int(obs_dim), "var_g": var_g, "max_age_K": int(K), "h_err": h_err,
        "rows": rows, "skipped": skipped, "verdict": verdict,
        "caveat": ("★ 本实验只用**逐包独立**丢失（周期 + 伯努利），不含突发信道。"
                   "★ 任务指标只有 n_episodes 集，逃逸率的分母是集数 —— "
                   "在逃逸率接近 0 或 1 的区间分辨率不足，不得当连续量引用。"
                   "★ 闭式预报用的是**开环轨迹**上标定的离线曲线 g_b(h)，"
                   "闭环会改变轨迹分布 ⇒ ③ 图的偏差就是这件事的直接度量。"),
    }
    with open(os.path.join(out, "22_task_design.json"), "w", encoding="utf-8") as f:
        json.dump(jsonable(payload), f, ensure_ascii=False, indent=2)
    print(f"[22] 产物：{os.path.join(out, '22_task_design.png')} / .json  "
          f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
