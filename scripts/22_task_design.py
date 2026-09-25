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

================================================================= ★★ 第 17 次自证伪（跑第一版时挂出来的，照记）
第一版 S_a 写成「实测 vs 闭式相对差 ≤ 8%」，在 SNR=14dB / b=6 / T=1（64-QAM，
PER=0.9502）这个点上被打穿：实测 17.51 vs 闭式 19.08（−8.2%）。
诊断（纯调度空跑 24 个种子，不含控制、不含删失）显示：
    该点 E[age]=19.08，但 **sqrt(Var(age))=19.6 —— 标准差与均值同量级**；
    40×450 步里只有约 900 次到达 ⇒ 单次实测标准误 ≈ 4.7% ⇒ −8% 只是 1.8σ 的噪声，
    纯调度无偏（−0.38% ± 0.89）—— **闭式没错，是我的容差拍得没有依据**。
⇒ 改为：容差 = 本点**采样标准误 × 3**（标准误由纯调度 MC 估出，与控制器无关），
   并对逃逸删失的点（episode 提前结束 ⇒ 长年龄被切掉）只做**单侧**断言。
⇒ 共因仍与前面 16 次相同：拿一个"看起来合理"的固定数当判据，没有先算它的量级（R2/R13）。

================================================================= 自检（不过就抛 —— R14）
S_a  闭环实测 E[age] 与闭式 periodic_mean_age 一致（⇒ 年龄闭式在闭环里也成立）
     容差 = 3 × 本点采样标准误（见上；**不是**固定百分比）
     ★ 删失点（episode 因逃逸提前结束）只断言「实测不会**偏高**」
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
                                 periodic_age_var, periodic_lossy_schedule,
                                 periodic_mean_age)
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
        # ★ 冒烟**不能**把 epoch 压到 3：闭环里模型误差会盖过量化误差，
        #   那样 T4（量化接线检查）必然假失败，还看不出任何设计效应。
        #   实测：3 epoch 时闭环 est_nmse 0.266 而闭式 0.0066（40×），全是模型误差。
        cfg["train"]["epochs"] = min(int(cfg["train"]["epochs"]), 60)
        cfg["data"]["n_episodes"] = min(int(cfg["data"]["n_episodes"]), 40)
        dz["snr_db_list"] = [20.0]
        dz["bits_list"] = [4, 8]
        cfg["task"]["n_episodes"] = 6
        cfg["eval"]["n_samples"] = 256

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
    g_curves, f_curves = {}, {}
    for name, head in heads.items():
        f = _offline(head, None)
        f_curves[name] = f
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
    h_err = reliable_horizon(hs, [float(f_curves["model"][h]) for h in hs],
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
    meas_steps = int(cfg["task"]["measured_steps"])
    rows = []
    censored = []

    def _age_se(T, per, n_rep=12):
        """★ 本点「闭环实测 E[age]」的**采样标准误**：纯调度空跑 n_rep 个种子取 std。

        年龄过程只由调度决定（与环境、控制器都无关 —— 只要 episode 没被删失），
        所以可以脱离仿真直接估。⇒ 比拍固定百分比诚实：
          PER=0.95 / T=1 ⇒ std ≈ 4.7%（E[age]=19 而 sqrt(Var)=19.6，同量级！）
          PER=0.20 / T=1 ⇒ std ≈ 0.1%
        """
        vals = []
        for i in range(n_rep):
            rng = np.random.default_rng(online_seed + 977 * (i + 1))
            sch = periodic_lossy_schedule(T, per)
            tot, n = 0.0, 0
            age = 0
            for _ in range(n_task_ep):
                age = 0
                for t in range(WARMUP + meas_steps):
                    if sch(t, rng):
                        age = 0
                    else:
                        age += 1
                    if t >= WARMUP:
                        tot += age
                        n += 1
            vals.append(tot / max(n, 1))
        return float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
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
                   "age_mean_pmf_trunc": float(np.dot(np.arange(K + 1), pmf)),
                   "age_sd": float(np.sqrt(periodic_age_var(per01, T))),
                   "age_se": float(_age_se(T, per01))}
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
                #   ★★ 容差 = 3 × 本点采样标准误（第 17 次自证伪：固定 8% 会被噪声打穿）
                emp_age = float(r["mean_age"])
                ana_age = rec["age_mean_pmf_trunc"]
                # ★★ se 必须有个**有量纲的地板**：PER≈0 时年龄是确定的（std=0），
                #    dev/1e-12 会造出 7.2e8 σ 这种**纯除零假数**（第一版就报了这个数）。
                #    地板取 1e-3 步 —— 远小于任何真实采样误差，但足以让比值有意义。
                se = max(rec["age_se"], 1e-3)
                tol = max(0.10, 3.0 * se)
                dev = emp_age - ana_age
                ep_frac = float(r["mean_ep_len"]) / max(meas_steps, 1)
                if ep_frac < 0.98:
                    # ★ 删失：episode 因逃逸提前结束 ⇒ 长年龄被系统性切掉
                    #   ⇒ 只可能把均值**拉低**，所以只断言上侧（上侧超了一定是真 bug）
                    censored.append({"snr_db": float(snr), "bits": b, "T": T,
                                     "per": float(per01), "ep_frac": ep_frac,
                                     "emp": emp_age, "ana": ana_age})
                    if dev > tol:
                        raise AssertionError(
                            f"★ S_a 未通过（删失点却偏高）：SNR={snr} T={T} per={per01:.4f} "
                            f"实测 {emp_age:.4f} vs 闭式 {ana_age:.4f}（+{dev/se:.1f}σ）"
                            f"—— 删失只会拉低均值，偏高说明调度没接上")
                elif abs(dev) > tol:
                    raise AssertionError(
                        f"★ S_a 未通过：SNR={snr} T={T} per={per01:.4f} "
                        f"闭环实测 E[age]={emp_age:.4f} vs 闭式（截断后）{ana_age:.4f} "
                        f"（偏差 {dev/se:+.1f}σ，容差 ±{tol:.4f} = 3×SE={se:.4f}）")
                rec[name] = {"pred": pred, "est_nmse": float(r["est_nmse"]),
                             "gap_closed": float(pred / max(float(r["est_nmse"]), 1e-18) - 1.0),
                             "escape_rate": float(r["escape_rate"]),
                             "mean_dist": float(r["mean_dist"]),
                             "mean_dist_tail": float(r["mean_dist_tail"]),
                             "p95_dist": float(r["p95_dist"]),
                             "age_mean": emp_age, "tx_rate": float(r["tx_rate"]),
                             "n_steps": int(r["n_steps"]),
                             "mean_ep_len": float(r["mean_ep_len"]),
                             "ep_frac": ep_frac,
                             "age_dev_sigma": float(dev / se)}
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

    # ---------- 4b) S_a 汇总：偏差到底多大、删失了多少点 ----------
    devs = np.array([r["model"]["age_dev_sigma"] for r in rows])
    print(f"[22] S_a：{len(rows)} 点，|偏差| 中位 {np.median(np.abs(devs)):.2f}σ "
          f"最大 {np.abs(devs).max():.2f}σ（容差 3σ；超出点数 "
          f"{int((np.abs(devs) > 3).sum())}）")
    if censored:
        ef = np.array([c["ep_frac"] for c in censored])
        print(f"[22] ★ 删失 {len(censored)}/{len(rows)} 点（episode 因逃逸提前结束，"
              f"ep_frac 中位 {np.median(ef):.2f} 最小 {ef.min():.2f}）"
              f"⇒ 这些点的 E[age] 与 NMSE 都是**偏低**的（长年龄被切掉、坏样本被提前终止）"
              f"⇒ T1/T2/T3 同时报全量口径与「未删失」口径")

    # ---------- 5) ★ R12 接线检查（T4 的机器版）----------
    # ★★ 不用"改 b 看比值"当主判据（第一版就是它，且会**假失败**）：
    #    闭环里一旦模型误差占大头，量化从 b=4 改到 b=8 的绝对变化被埋掉，
    #    比值 <5% ⇒ 判据报警，但 payload_fn 其实接得好好的。
    #    ⇒ 改成**绝对标定**：T=1 且 PER≈0 时，控制器每步拿到的就是量化后的载荷，
    #      所以闭环 est_nmse **必须等于**量化地板 `g_b(0) = MSE_q/var_g`（这是已知答案）。
    #      这条与模型好坏无关 ⇒ 才是 R12 该有的形态（"拿已知答案代进去验量级"）。
    floor_errs, n_wired = [], 0
    for r in rows:
        # ★ 用 1e-3 而不是 1e-9：PER 打印成 0.000 不代表它严格为 0，
        #   而 T=1 时 PER=1e-3 只让 0.1% 的步用上模型 ⇒ 对地板的影响远小于 10% 容差
        if r["T"] != 1 or r["per"] > 1e-3:
            continue
        for name in ("model", "persistence"):
            want = float(quantizers[r["bits"]].measure_mse(allobs) / var_g)
            got = float(r[name]["est_nmse"])
            floor_errs.append({"snr_db": r["snr_db"], "bits": r["bits"], "head": name,
                               "want": want, "got": got,
                               "rel": got / max(want, 1e-18) - 1.0})
    if floor_errs:
        worst = max(abs(x["rel"]) for x in floor_errs)
        if worst > 0.10:
            raise AssertionError(
                f"★ T4/S_d 未通过：T=1/PER=0 时闭环 est_nmse 与量化地板差 "
                f"{worst:.1%}（>10%）⇒ payload_fn 没接进 control.py 或口径错了"
                f"（R12 违规，整个 X35 在空跑）。明细：{floor_errs[:4]}")
        n_wired = len(floor_errs)
        print(f"[22]   T4/S_d 通过：{n_wired} 个 (T=1,PER=0) 点的闭环 est_nmse "
              f"落在量化地板上（最大偏差 {worst:.2%}）")
    # 次级：改 b 是否真的改变闭环结果（**只作参考**，不作通过条件，
    #        因为它会被模型误差埋掉 —— 第一版拿它当主判据 ⇒ 假失败）
    n_ratio = 0
    for snr in snrs:
        for T in sorted({g["T"] for g in grid}):
            sub = [r for r in rows if abs(r["snr_db"] - snr) < 1e-9 and r["T"] == T]
            if len(sub) < 2:
                continue
            vals = [r["model"]["est_nmse"] for r in sub]
            if max(vals) / max(min(vals), 1e-18) > 1.05:
                n_ratio += 1
    if n_ratio == 0:
        print("[22]   ⚠ 次级检查：改 b 未使闭环 est_nmse 变化 >5% "
              "（模型误差占主导时的正常现象，不是 bug）")
    else:
        print(f"[22]   次级检查：{n_ratio} 个 (SNR,T) 组里改 b 使闭环结果变化 >5%")

    # ---------- 6) 判据 ----------
    print("[22] " + "=" * 76)
    TASK_METRICS = ("mean_dist_tail", "escape_rate")
    eps = float(cfg["task"].get("eps_floor", 1e-9))

    def _spread(v) -> float:
        """相对极差 —— 用来判断这个指标在候选集上**有没有分辨率**。

        ★ 为什么必须查：escape_rate 是"每集是否逃逸"的 0/1 均值，在很多
          (SNR,b) 组里**全为 0**（分辨率不足）⇒ 拿它算"最优→次优退化"会得到
          0/0 或荒谬的负数（冒烟实测给出 −29×）。这种点必须剔除，不能进中位数。
        """
        v = np.asarray(v, dtype=float)
        return float((v.max() - v.min()) / max(abs(v.max()), abs(v.min()), eps))

    # --- T1：最优 → 次优的相对退化，任务指标 vs NMSE ---
    t1 = {}
    for met in TASK_METRICS:
        ratios = []
        n_degen = 0
        for snr in snrs:
            for b in bits_list:
                sub = sorted([r for r in rows if abs(r["snr_db"] - snr) < 1e-9
                              and r["bits"] == b], key=lambda r: r["T"])
                if len(sub) < 2:
                    continue
                nv = np.array([r["model"]["est_nmse"] for r in sub])
                tv = np.array([r["model"][met] for r in sub])
                if _spread(tv) < 1e-6 or _spread(nv) < 1e-6:
                    n_degen += 1
                    continue
                kn, kt = int(np.argmin(nv)), int(np.argmin(tv))
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
                  f"≥2× 的点 {ok}/{len(ratios)}"
                  f"（剔除分辨率不足的组 {n_degen} 个）")
        else:
            print(f"[22]   T1[{met}]：无足够候选（{n_degen} 组分辨率不足）")

    # --- T2：排序一致性 ---
    t2 = {}
    try:
        from scipy.stats import spearmanr
        for met in TASK_METRICS:
            a = np.array([r["model"]["est_nmse"] for r in rows])
            c = np.array([r["model"][met] for r in rows])
            if _spread(c) < 1e-9:
                t2[met] = None
                print(f"[22]   T2[{met}]：该指标在全场**无分辨率**（全等），不计 ρ")
                continue
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
            c0 = np.array([r["model"][met] for r in rows])
            if _spread(c0) < 1e-9:
                t2[met] = None
                print(f"[22]   T2[{met}]：该指标在全场**无分辨率**（全等），不计 ρ")
                continue
            a = _rank(np.array([r["model"]["est_nmse"] for r in rows]))
            c = _rank(c0)
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

    # --- ★ 删失敏感性：只在「未删失」子集上重算 T2 / T5 / argmin 一致性 ----------
    #    为什么要看：删失点的 NMSE 是**乐观**的（坏到要逃逸的样本被提前终止），
    #    而 NMSE 与任务指标的关系（T2）恰好最容易被这种乐观污染。
    unc = [r for r in rows if r["model"]["ep_frac"] >= 0.98]
    sens = None
    if 8 <= len(unc) < len(rows):
        sens = {"n_uncensored": len(unc), "n_total": len(rows)}
        try:
            from scipy.stats import spearmanr
            for met in TASK_METRICS:
                a = np.array([r["model"]["est_nmse"] for r in unc])
                c = np.array([r["model"][met] for r in unc])
                sens[f"T2_{met}"] = (None if _spread(c) < 1e-9
                                     else float(spearmanr(a, c).statistic))
        except Exception:
            pass
        g = np.abs(np.array([r["model"]["gap_closed"] for r in unc]))
        sens["T5_median"] = float(np.median(g))
        ag, n_ag = 0, 0
        for snr in snrs:
            for b in bits_list:
                sub = sorted([r for r in unc if abs(r["snr_db"] - snr) < 1e-9
                              and r["bits"] == b], key=lambda r: r["T"])
                if len(sub) < 2:
                    continue
                n_ag += 1
                kp = int(np.argmin([r["model"]["pred"] for r in sub]))
                kt = int(np.argmin([r["model"]["mean_dist_tail"] for r in sub]))
                ag += int(sub[kp]["T"] == sub[kt]["T"])
        sens["argmin_agree"] = f"{ag}/{n_ag}" if n_ag else None
        print(f"[22] ★ 删失敏感性（未删失 {len(unc)}/{len(rows)} 点）："
              f"T2[dist]={sens.get('T2_mean_dist_tail')} "
              f"T2[esc]={sens.get('T2_escape_rate')} "
              f"T5={sens['T5_median']:.1%} argmin={sens['argmin_agree']}")
    elif len(unc) == len(rows):
        print("[22]   （无删失点 ⇒ 不做敏感性对照）")

    verdict = {
        "T1_task_over_nmse_regression_median": {k: (float(np.median(v)) if v else None)
                                                for k, v in t1.items()},
        "T2_spearman": {k: (None if v is None else float(v)) for k, v in t2.items()},
        "T3_task_over_nmse_gap_median": {k: (float(np.median(v)) if v else None)
                                         for k, v in t3.items()},
        "T4_quant_floor_check": {"n_points": int(n_wired),
                                 "max_rel_dev": (max(abs(x["rel"]) for x in floor_errs)
                                                 if floor_errs else None),
                                 "detail": floor_errs},
        "T4b_ratio_check_n": int(n_ratio),
        "T5_closed_pred_median": t5_med,
        "argmin_agree": agree,
        "S_a_age_dev_sigma": {"median": float(np.median(np.abs(devs))),
                              "max": float(np.abs(devs).max()),
                              "n_over_3sigma": int((np.abs(devs) > 3).sum())},
        "censoring": {"n_censored": len(censored), "n_total": len(rows),
                      "detail": censored},
        "censoring_sensitivity": sens,
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
        # ★ 不用 twinx：本仓库的 savefig 链路在 twinx 上会炸
        #   （matplotlib 的 dpi_scale_trans 被污染：can't multiply sequence by ...）。
        #   ⇒ 改成各自按本组最大值归一化后画在同一轴上，并在标题里写明"已归一化"。
        nv = np.array([r["model"]["est_nmse"] for r in sub0], dtype=float)
        pv = np.array([r["model"]["pred"] for r in sub0], dtype=float)
        dv = np.array([r["model"]["mean_dist_tail"] for r in sub0], dtype=float)
        ax.plot([r["T"] for r in sub0], pv / max(pv.max(), 1e-18), "k--", label="闭式 NMSE")
        ax.plot([r["T"] for r in sub0], nv / max(nv.max(), 1e-18), "o-",
                color=PALETTE["blue"], label="闭环 NMSE")
        ax.plot([r["T"] for r in sub0], dv / max(dv.max(), 1e-18), "s-",
                color=PALETTE["red"], label="稳态跟踪距离")
        ax.set_xlabel("T（步）")
        ax.set_ylabel("各曲线按本组最大值归一化")
        ax.set_title(f"② SNR={snrs[len(snrs)//2]:g}dB "
                     f"b={bits_list[min(2, len(bits_list) - 1)]}（已归一化）")
        ax.legend(fontsize=7.5)
        ax.grid(alpha=0.3)
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
            ax.plot([r["model"]["age_mean"] for r in sub],
                    [r["model"]["escape_rate"] for r in sub], "o", color=col,
                    ms=5, alpha=0.85, label=f"{snr:g}dB")
    ax.set_xlabel("E[age]（步）"); ax.set_ylabel("逃逸率")
    ax.set_title("⑥ 逃逸率 vs 信息年龄")
    ax.legend(fontsize=7.5, ncol=2)

    save_fig(fig, os.path.join(out, "22_task_design.png"))
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
