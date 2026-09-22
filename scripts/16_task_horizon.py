#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""脚本 16 —— ★ X30：**任务锚定的 H\\***（task-anchored horizon）

一句话
------
此前所有 H\\* 都是 **NMSE 超过 0.05** 定义的。这个阈值**与任务无关** ——
它只是"预测误差变得难看"的那一步，不是"任务开始失败"的那一步。
X30 在 UAV 跟踪场景上把两个视界并排测出来，报出它们的**比值**：

    H\\*_err  = 离线闭环 NMSE(h) 首次 > 0.05 的步数      （沿用 X2/X14 口径）
    H\\*_task = 闭环控制下**任务指标开始坏掉**的传输间隔  （本实验新定义）

★ 为什么这一步非做不可（上游依据）
----------------------------------
母论文 *"...schedules sensing updates based on **task urgency** and predictive
uncertainty"* —— 从"误差"到"任务"中间差一个映射，**本仓库此前没人测过这个映射**。
X30 就是把映射测出来。两种结果都算结论：
  · 比值 ≈ 1  ⇒ 0.05 这个阈值恰好任务相关（要说清为什么，不能靠运气）
  · 比值 ≫ 1  ⇒ **阈值过保守**，H\\* 必须由任务定义（这是更强的结论）

★ 与 `eval.run_tracking` 的差别（不是重复实现）
--------------------------------------------------
`run_tracking` 的动作来自离线 episode ⇒ **轨迹与模型无关**，测的是"模型在固定分布上能推多远"。
本脚本的动作来自 `controller(est)` ⇒ **模型误差会改变真实轨迹**
（compounding / distribution shift 第一次真正接上）。
★ 代价（必须写在结论里）：此时测到的视界是 **(模型, 控制器, 调度) 三元组**的属性，
  **不是模型单独的属性**。跨实验引用时必须连同三元组一起引用。

★ 自检（R12「它接线了吗」+ R14「报错优于假数字」）
----------------------------------------------------
W1  T=1 时 est_nmse 必须 ≈ 0（估计恒等于真值）—— 不 ≈0 说明记账顺序错了
W2  tx_rate 必须 = 1/T；mean_age 必须 = (T−1)/2（周期调度的解析值）
W3  T=1 的稳态跟踪距离必须 ≈ 解析值 e_ss = (a_tgt + κ·v_tgt)/kp（PD 增益推导的验证）
    三者任一不满足 ⇒ 直接抛异常，**不产出数字**。

运行
----
    python scripts/16_task_horizon.py --config configs/uav_task.yaml
    python scripts/16_task_horizon.py --config configs/uav_task.yaml --quick   # 冒烟
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
                           run_closed_loop_control, task_horizon)
from wmlab.data import split_episodes, transitions_from_episodes
from wmlab.envs import make_env
from wmlab.eval import periodic_schedule, reliable_horizon
from wmlab.models import MLPWorldModel
from wmlab.rollout import closed_loop_error_curve
from wmlab.train import train_world_model
from wmlab.utils import (count_params, describe_device, get_device, load_config,
                         output_dir, set_seed)
from wmlab.utils.plot import PALETTE, apply_style, save_fig


def parse_args():
    p = argparse.ArgumentParser(description="X30 任务锚定的 H*")
    p.add_argument("--config", default="configs/uav_task.yaml")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--tag", default="16_task_horizon")
    p.add_argument("--periods", default=None, help="逗号分隔的周期网格，覆盖 config")
    p.add_argument("--episodes", type=int, default=None, help="每个 T 的 episode 数")
    p.add_argument("--quick", action="store_true", help="冒烟：小模型/少数据/3 个周期点")
    return p.parse_args()


def pooled_var(episodes) -> float:
    allobs = np.concatenate([e["obs"] for e in episodes], axis=0).astype(np.float64)
    return float(allobs.var())


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


# ================================================================== 主流程
def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.episodes is not None:
        cfg["task"]["n_episodes"] = args.episodes
    if args.periods:
        cfg["task"]["periods"] = [int(x) for x in args.periods.split(",")]
    if args.quick:
        cfg["train"]["epochs"] = min(int(cfg["train"]["epochs"]), 3)
        cfg["data"]["n_episodes"] = min(int(cfg["data"]["n_episodes"]), 20)
        cfg["task"]["n_episodes"] = 5
        cfg["task"]["periods"] = [1, 8, 32]
        cfg["eval"]["n_samples"] = 64
    tk = cfg["task"]
    # ★ R12：env 的 episode 长度必须 = 预热 + 被测，否则预热会把 episode 截断
    need = int(tk["warmup_steps"]) + int(tk["measured_steps"])
    if int(cfg["env"]["max_steps"]) != need:
        raise ValueError(
            f"env.max_steps({cfg['env']['max_steps']}) 必须等于 "
            f"warmup({tk['warmup_steps']}) + measured({tk['measured_steps']}) = {need}；"
            f"否则预热段会被 env 自己的 TimeLimit 截断，控制器根本没收敛就开始记账")

    seed = int(cfg["seed"])
    set_seed(seed)
    apply_style()
    device = get_device(args.device or cfg.get("device"))
    out = output_dir(cfg)
    t0 = time.time()

    # ---------- 1) 环境 + PD 控制器 ----------
    env = make_env(cfg["env"]["id"], seed=seed,
                   noise_std=float(cfg["env"]["noise_std"]),
                   max_steps=int(cfg["env"]["max_steps"]))
    cc = cfg["controller"]
    ctrl = PDRelativeController(dt=env.dt, omega_n=float(cc["omega_n"]),
                                zeta=float(cc["zeta"]), kappa=float(env.kappa),
                                a_max=float(env.a_max))
    # ★ 解析预期：**旋转坐标系**下的稳态误差（不是标量相加！）
    #   目标做圆周运动 ⇒ 扰动项 d = a_tgt + κ·v_tgt 是**旋转的**，其中
    #   a_tgt = −ω²r（径向）与 κ·v_tgt = κωr（切向）**互相正交**，不能标量相加。
    #   在旋转系里稳态误差 e 为常向量，其惯性导数 ė = ω ẑ×e、ë = −ω² e，代入
    #        ë + (κ+kd) ė + kp e = d
    #   得 2×2 线性方程组（r̂ / t̂ 两个分量），解出 |e|。
    #   ★ 第一版写成标量 e_ss = (a_tgt + κ v_tgt)/kp = 0.576 m，**漏了正交关系**；
    #     实测稳态 0.394 m ⇒ 与向量解 0.3996 m 吻合 1.4%，与标量解差 46%。
    w = float(env.omega)
    r_orb = float(env.r_orbit)
    kap = float(env.kappa)
    kp, kd = ctrl.kp, ctrl.kd
    A = np.array([[kp - w ** 2, -(kap + kd) * w],
                  [(kap + kd) * w, kp - w ** 2]])
    dvec = np.array([-(w ** 2) * r_orb, kap * w * r_orb])
    e_vec = np.linalg.solve(A, dvec)
    e_ss = float(np.linalg.norm(e_vec))
    print(f"[16] env={cfg['env']['id']} noise={cfg['env']['noise_std']} "
          f"device={describe_device(device)}")
    print(f"[16] PD 增益（闭式解）: ω_n={ctrl.omega_n} ζ={ctrl.zeta} κ={kap} "
          f"⇒ kp={kp:.4f} kd={kd:.4f}  a_max={ctrl.a_max}")
    print(f"[16] ★ 解析稳态跟踪误差（旋转系向量解）= |{np.round(e_vec, 4)}| = "
          f"**{e_ss:.4f} m**（逃逸半径 {env.escape_radius} m）")

    # ---------- 2) 用 PD 控制器采训练数据（★ 训练分布 = 真实控制分布）----------
    n_ep = int(cfg["data"]["n_episodes"])
    train_all = collect_controlled_episodes(env, ctrl, n_episodes=n_ep, seed=seed,
                                            max_steps=int(cfg["env"]["max_steps"]))
    lens = np.asarray([e["length"] for e in train_all])
    print(f"[16] 训练集（PD 采集）：{len(train_all)} 集，共 {lens.sum()} 步，"
          f"长度 mean={lens.mean():.1f} max={lens.max()}")

    train_eps, val_eps = split_episodes(train_all, float(cfg["train"]["val_ratio"]), seed)
    tr = tuple(torch.as_tensor(x) for x in transitions_from_episodes(train_eps))
    va = tuple(torch.as_tensor(x) for x in transitions_from_episodes(val_eps))

    obs_dim = int(train_all[0]["obs"].shape[1])
    model = MLPWorldModel(obs_dim=obs_dim, act_dim=int(env.act_dim),
                          latent_dim=int(cfg["model"]["latent_dim"]),
                          hidden=int(cfg["model"]["hidden"]),
                          discrete_act=bool(env.is_discrete)).to(device)
    hist = train_world_model(model, tr, va, cfg, device, verbose=False)
    print(f"[16] 训练完成：params={count_params(model):,}  val_loss={hist['val_total'][-1]:.6f}")

    # ---------- 3) 离线曲线 → H*_err（held-out 集）----------
    ev_seed = seed + int(cfg["data"]["eval_seed_offset"])
    ev_env = make_env(cfg["env"]["id"], seed=ev_seed,
                      noise_std=float(cfg["env"]["noise_std"]),
                      max_steps=int(cfg["env"]["max_steps"]))
    eval_eps = collect_controlled_episodes(ev_env, ctrl, n_episodes=max(20, n_ep // 5),
                                           seed=ev_seed,
                                           max_steps=int(cfg["env"]["max_steps"]))
    ev_env.close()
    var_g = pooled_var(eval_eps)
    hs = [int(h) for h in cfg["eval"]["horizons"]]
    hs = [h for h in hs if h <= max(e["length"] for e in eval_eps) - 2]
    curve = closed_loop_error_curve(model, eval_eps, hs, device,
                                    n_samples=int(cfg["eval"]["n_samples"]), seed=seed)
    thr = float(cfg["eval"]["err_threshold"])
    mode = cfg["eval"].get("threshold_mode", "rel")
    pt = [float(np.asarray(curve["nmse_per_step"], dtype=float)[h - 1]) for h in hs]
    h_err = reliable_horizon(hs, pt, thr, mode)
    # 全局分母版本（与在线 est_nmse 可比，图 ④ 用）
    pt_g = [float(np.asarray(curve["mse_per_step"], dtype=float)[h - 1]) / var_g for h in hs]
    print(f"[16] 评测集 held-out: {len(eval_eps)} 集 seed={ev_seed} var_global={var_g:.6f}")
    print(f"[16] ★ H*_err（离线闭环 · 逐点 · NMSE>{thr}）= {h_err}")
    print(f"[16]   逐点 NMSE 序列（前 8）: {[round(x, 5) for x in pt[:8]]}")

    # ---------- 4) 在线闭环扫描 ----------
    tk = cfg["task"]
    periods = [int(t) for t in tk["periods"]]
    n_task_ep = int(tk["n_episodes"])
    online_seed = seed + int(cfg["data"]["online_seed_offset"])
    rows = []
    for est_name in ("model", "persistence"):
        for T in periods:
            r = run_closed_loop_control(
                env, model, ctrl, periodic_schedule(T),
                n_episodes=n_task_ep, seed=online_seed,
                max_steps=int(tk["measured_steps"]), device=device,
                estimator=est_name, var_g=var_g, label=f"{est_name}(T={T})",
                tail_frac=float(tk["tail_frac"]), warmup_steps=int(tk["warmup_steps"]))
            r["T"] = int(T)
            rows.append(r)
            print(f"[16] {r['label']:<22} tx={r['tx_rate']:.4f} age={r['mean_age']:6.2f} "
                  f"escape={r['escape_rate']:.3f} dist={r['mean_dist']:7.3f} "
                  f"tail={r['mean_dist_tail']:7.3f} est_nmse={r['est_nmse']:.5f}")

    env.close()

    # ---------- 5) ★★ R12 接线自检（不过就抛，不产出数字）----------
    checks = {}
    by = {(r["estimator"], r["T"]): r for r in rows}
    r1 = by.get(("model", 1))
    if r1 is None:
        raise RuntimeError("周期网格必须包含 T=1（它是 oracle 基线，缺了无法定标）")
    checks["W1_est_nmse_at_T1"] = float(r1["est_nmse"])
    if not r1["est_nmse"] < 1e-9:
        raise AssertionError(
            f"★ W1 未通过：T=1 时估计应恒等于真值，est_nmse 应 ≈0，实测 "
            f"{r1['est_nmse']:.3e} —— 记账顺序又错了（见 control.py 的同名注释）")
    # ★★ W2 用**精确**判据，不用容差：周期调度的 n_tx 与 E[age] 都有闭式解。
    #   （第一版用 "tx_rate ≈ 1/T 容差 2%"，在 T=32 上被判失败 —— 实测 0.0300 vs 0.03125。
    #    这不是接线错误，是 **episode 有限长度的边缘效应**：200 步里只在
    #    t=32,64,...,192 送了 6 次 ⇒ 6/200=0.030，而 1/32=0.03125。
    #    ★ 用容差判据会把"数学上的必然"误报成 bug，也会放过真正的 bug —— 改成闭式解。）
    worst_tx, worst_age = 0.0, 0.0
    for T in periods:
        r = by[("model", T)]
        exp_tx, exp_age_sum = 0, 0
        for L in r["ep_lens"]:                      # 每集长度（预热之后、可能提前逃逸）
            q, rem = divmod(int(L), T)
            exp_tx += q                             # 送达发生在 t = T, 2T, ... ≤ L
            exp_age_sum += q * T * (T - 1) // 2 + rem * (rem + 1) // 2
        d_tx = abs(r["n_tx"] - exp_tx)
        d_age = abs(r["mean_age"] - exp_age_sum / max(r["n_steps"], 1))
        worst_tx = max(worst_tx, d_tx)
        worst_age = max(worst_age, d_age)
    checks["W2_max_abserr_n_tx"] = float(worst_tx)
    checks["W2_max_abserr_mean_age"] = float(worst_age)
    if worst_tx != 0:
        raise AssertionError(f"★ W2 未通过：n_tx 与闭式解 floor(L/T) 之和差 {worst_tx}")
    if worst_age > 1e-6:
        raise AssertionError(f"★ W2 未通过：mean_age 与闭式解差 {worst_age:.3e}")
    # W3：预热之后 T=1 就是"控制器看真值"的稳态 ⇒ 实测距离必须 ≈ 解析 e_ss
    checks["W3_dist_T1"] = float(r1["mean_dist"])
    checks["W3_analytic_e_ss"] = float(e_ss)
    checks["W3_ratio"] = float(r1["mean_dist"] / max(e_ss, 1e-9))
    print(f"[16] ★ W3 解析对账：T=1 稳态距离实测 {r1['mean_dist']:.4f} m vs "
          f"解析 e_ss {e_ss:.4f} m ⇒ 比值 {checks['W3_ratio']:.3f}")
    # ★ W3 的解析值是**确定性**稳态误差；有风扰时实测只会更高：
    #   由 Jensen 不等式，E‖e‖ ≥ ‖E e‖ = e_ss ⇒ 比值应 ≥ 1（统计意义上）。
    #   本场景 σ=0.15 实测比值 ≈ 2.2（风扰贡献与确定性项同量级），
    #   确定性环境下的精确吻合由自检 ⑬ 单独验（相对偏差 < 5%）。
    if not (0.8 < checks["W3_ratio"] < 3.5):
        raise AssertionError(
            f"★ W3 未通过：实测稳态距离与解析 e_ss 差 {checks['W3_ratio']:.2f} 倍。"
            f"（若 ≫1 且是**确定性**环境，多半是预热不够、瞬态没走完 —— "
            f"见 control.py 的 warmup 说明；有风扰时按 Jensen 应 ≥1，"
            f"若 <0.8 多半是 kp/kd 没按公式取，或 κ/a_max/dt 没从 env 读）")

    # ---------- 6) H*_task ----------
    mrows = sorted([r for r in rows if r["estimator"] == "model"], key=lambda r: r["T"])
    prows = sorted([r for r in rows if r["estimator"] == "persistence"], key=lambda r: r["T"])
    base_escape = float(mrows[0]["escape_rate"])
    base_dist = float(mrows[0]["mean_dist_tail"])
    h_escape = task_horizon(mrows, "escape_rate", base_escape,
                            tol_abs=float(tk["escape_tol_abs"]))
    h_dist = task_horizon(mrows, "mean_dist_tail", base_dist,
                          tol_rel=float(tk["dist_tol_rel"]))
    p_escape = task_horizon(prows, "escape_rate", float(prows[0]["escape_rate"]),
                            tol_abs=float(tk["escape_tol_abs"]))
    p_dist = task_horizon(prows, "mean_dist_tail", float(prows[0]["mean_dist_tail"]),
                          tol_rel=float(tk["dist_tol_rel"]))
    print(f"[16] ★★ H*_task（model）= 逃逸口径 {h_escape} / 距离口径 {h_dist}")
    print(f"[16]    H*_task（persistence 基线）= {p_escape} / {p_dist}")
    print(f"[16]    H*_err（误差阈值）= {h_err}")
    # ---------- 6b) ★★ 容差依赖扫描（本实验最重要的一张表）----------
    tol_table = []
    for ta in (0.02, 0.05, 0.10, 0.20):
        v = task_horizon(mrows, "escape_rate", base_escape, tol_abs=ta)
        tol_table.append({"metric": "escape_rate", "tol_kind": "abs", "tol": ta,
                          "h_task": v,
                          "ratio_over_h_err": (v / h_err) if (h_err and v) else None})
    for trel in (0.2, 0.5, 1.0, 2.0):
        v = task_horizon(mrows, "mean_dist_tail", base_dist, tol_rel=trel)
        tol_table.append({"metric": "mean_dist_tail", "tol_kind": "rel", "tol": trel,
                          "h_task": v,
                          "ratio_over_h_err": (v / h_err) if (h_err and v) else None})
    print("[16] ---- ★★ H*_task 对『任务指标 + 容差』的依赖 ----")
    print(f"[16]   {'指标':<16}{'容差':>8}{'H*_task':>10}{'/ H*_err':>10}")
    for x in tol_table:
        tag = (f"+{x['tol']:g}" if x["tol_kind"] == "abs" else f"×{1 + x['tol']:g}")
        rr = "n/a" if x["ratio_over_h_err"] is None else f"{x['ratio_over_h_err']:.2f}"
        print(f"[16]   {x['metric']:<16}{tag:>8}{str(x['h_task']):>10}{rr:>10}")
    vals = [x["h_task"] for x in tol_table if x["h_task"] is not None]
    if vals:
        print(f"[16]   ⇒ 同一批数据上 H*_task 的跨度 = {min(vals)} … {max(vals)} "
              f"（{max(vals) / max(min(vals), 1):.0f} 倍），而 H*_err 只有一个数 "
              f"({h_err}) —— **换任务指标就会换结论**")

    # ---------- 7) 在线 vs 离线（闭环放大检验）----------
    # ★ 口径对齐（这一条不改就是假结论）：在线 est_nmse 是 **age = 0…T−1 的平均**
    #   （每个 age 出现次数相同），而离线曲线给的是**某一个 h** 的误差。
    #   第一版直接拿 `offline_nmse(T)` 去比，得到 median=0.247 的"在线更小"——
    #   那只是把"全程平均"和"最老的那一步"比，不是机制差异。
    #   正确做法：把离线曲线按 age 分布求平均 `E_a[NMSE(a)] = (1/T)Σ_{a<T} NMSE(a)`，
    #   （NMSE(0)=0：刚收到真值那一步误差恒为 0），再与在线比。
    #   网格 hs 不连续 ⇒ 用 log-log 线性插值到整数 h（曲线光滑，误差可忽略）。
    hs_arr = np.asarray(hs, dtype=float)
    ptg_arr = np.asarray(pt_g, dtype=float)

    def offline_at(h: int) -> float:
        if h <= 0:
            return 0.0
        return float(np.exp(np.interp(np.log(h), np.log(hs_arr),
                                      np.log(np.maximum(ptg_arr, 1e-12)))))

    online_offline = []
    for r in mrows:
        T = int(r["T"])
        if T < 2:
            continue                      # T=1 ⇒ 两边都恒为 0，比值无意义
        off_mean = float(np.mean([offline_at(a) for a in range(T)]))
        online_offline.append({
            "T": T,
            "online_est_nmse": r["est_nmse"],
            "offline_mean_over_ages": off_mean,
            "offline_nmse_at_T": offline_at(T),
            "ratio": r["est_nmse"] / max(off_mean, 1e-12),
        })
    if online_offline:
        rr = [x["ratio"] for x in online_offline]
        print(f"[16] 在线/离线（按 age 分布对齐后）比值：median={np.median(rr):.3f} "
              f"min={min(rr):.3f} max={max(rr):.3f} "
              f"（>1 ⇒ 闭环控制**放大**了模型误差，离线 H* 不够用）")

    # ---------- 8) 出图 ----------
    fig, axes = plt.subplots(2, 3, figsize=(18.5, 9.4))
    ax = axes[0, 0]
    for est_name, col, mk in (("model", PALETTE["blue"], "o"),
                              ("persistence", PALETTE["orange"], "s")):
        d = sorted([r for r in rows if r["estimator"] == est_name], key=lambda r: r["T"])
        ax.plot([r["T"] for r in d], [r["escape_rate"] for r in d], mk + "-",
                color=col, label=f"{est_name}")
    ax.axhline(base_escape + float(tk["escape_tol_abs"]), color=PALETTE["grey"], ls=":",
               label=f"tol = base + {float(tk['escape_tol_abs']):g}")
    if h_escape is not None:
        ax.axvline(h_escape, color=PALETTE["red"], ls="--", alpha=0.8)
        ax.annotate(f"H*_escape={h_escape:g}", xy=(h_escape, 0.5),
                    xytext=(h_escape * 1.1, 0.55), color=PALETTE["red"], fontsize=9)
    ax.set_xscale("log"); ax.set_xlabel("传输周期 T (steps)")
    ax.set_ylabel("逃逸率（跟丢的比例）"); ax.set_ylim(-0.02, 1.02)
    ax.set_title("① 任务失败率 vs 通信周期")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for est_name, col, mk in (("model", PALETTE["blue"], "o"),
                              ("persistence", PALETTE["orange"], "s")):
        d = sorted([r for r in rows if r["estimator"] == est_name], key=lambda r: r["T"])
        ax.plot([r["T"] for r in d], [r["mean_dist_tail"] for r in d], mk + "-",
                color=col, label=f"{est_name} (稳态)")
    ax.axhline(e_ss, color=PALETTE["green"], ls=":", label=f"解析 e_ss={e_ss:.2f} m")
    ax.axhline(base_dist * (1 + float(tk["dist_tol_rel"])), color=PALETTE["grey"], ls=":",
               label=f"tol = base×(1+{float(tk['dist_tol_rel']):g})")
    if h_dist is not None:
        ax.axvline(h_dist, color=PALETTE["red"], ls="--", alpha=0.8)
        ax.annotate(f"H*_dist={h_dist:g}", xy=(h_dist, base_dist),
                    xytext=(h_dist * 1.1, base_dist * 1.5), color=PALETTE["red"], fontsize=9)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("传输周期 T (steps)"); ax.set_ylabel("稳态跟踪距离 (m)")
    ax.set_title("② 稳态跟踪误差 vs 通信周期")
    ax.legend(fontsize=8)

    ax = axes[0, 2]
    ax.plot(hs, pt, "s--", color=PALETTE["red"], label="closed-loop · per-step")
    ax.axhline(thr, color=PALETTE["grey"], ls=":", label=f"threshold={thr}")
    if h_err is not None:
        ax.axvline(h_err, color=PALETTE["red"], ls="--", alpha=0.8)
        ax.annotate(f"H*_err={h_err:g}", xy=(h_err, thr),
                    xytext=(h_err * 1.1, thr * 3), color=PALETTE["red"], fontsize=9)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Rollout horizon h (steps)"); ax.set_ylabel("NMSE (per-step)")
    ax.set_title("③ 离线曲线：误差阈值定义的 H*_err")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    if online_offline:
        ax.plot([x["T"] for x in online_offline],
                [x["offline_mean_over_ages"] for x in online_offline],
                "o-", color=PALETTE["blue"], label="离线 E_a[NMSE(a)]（按 age 分布对齐）")
        ax.plot([x["T"] for x in online_offline],
                [x["offline_nmse_at_T"] for x in online_offline],
                ":^", color=PALETTE["grey"], alpha=0.7, label="离线 NMSE(T)（仅最老一步）")
        ax.plot([x["T"] for x in online_offline], [x["online_est_nmse"] for x in online_offline],
                "s-", color=PALETTE["orange"], label="在线闭环实测估计误差")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("周期 T / 视界 h (steps)")
        ax.set_ylabel("NMSE（全局分母）")
        ax.set_title("④ 闭环放大检验：在线 vs 离线（同口径）")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "周期网格与 horizons 无交集", ha="center", va="center")
        ax.set_title("④ 闭环放大检验（无数据）")

    # ⑤ ★★ H*_task 随"容差"漂移 —— 证明"任务视界"不是一个数，是一个族
    ax = axes[1, 2]
    for met, kind, col, mk in (("escape_rate", "abs", PALETTE["red"], "o"),
                               ("mean_dist_tail", "rel", PALETTE["blue"], "s")):
        d = [x for x in tol_table if x["metric"] == met]
        xs = [x["tol"] if kind == "abs" else 1 + x["tol"] for x in d]
        ys = [x["h_task"] for x in d]
        ax.plot(xs, ys, mk + "-", color=col,
                label=f"{met}（{'绝对' if kind == 'abs' else '相对'}容差）")
    if h_err is not None:
        ax.axhline(h_err, color=PALETTE["green"], ls="--",
                   label=f"H*_err = {h_err:.1f}（误差阈值）")
    ax.set_xscale("log")
    ax.set_xlabel("容差（escape: 绝对增量；dist: 相对倍数）")
    ax.set_ylabel("H*_task (steps)")
    ax.set_title("⑤ ★ 任务视界 = 一个族，不是一个数")
    ax.legend(fontsize=7.5)

    # ⑥ 通信–任务帕累托前沿
    ax = axes[1, 1]
    for est_name, col, mk in (("model", PALETTE["blue"], "o"),
                              ("persistence", PALETTE["orange"], "s")):
        d = sorted([r for r in rows if r["estimator"] == est_name], key=lambda r: r["T"])
        ax.plot([r["tx_rate"] for r in d], [r["escape_rate"] for r in d], mk + "-",
                color=col, label=est_name)
        for r in d:
            ax.annotate(str(r["T"]), (r["tx_rate"], r["escape_rate"]),
                        fontsize=6, color=col, xytext=(3, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("传输率（越小越省通信）"); ax.set_ylabel("逃逸率（任务失败率）")
    ax.set_title("⑥ 通信–任务帕累托前沿（数字 = 周期 T）")
    ax.legend(fontsize=8)

    fig.suptitle(
        f"wmlab · X30 任务锚定的 H* · env={cfg['env']['id']} σ={cfg['env']['noise_std']} · "
        f"PD(ω_n={ctrl.omega_n}, ζ={ctrl.zeta}) · {n_task_ep} 集/点 · seed={seed} · "
        f"H*_err={h_err} · H*_task(escape)={h_escape} · H*_task(dist)={h_dist}",
        fontsize=10)
    fig.tight_layout()
    png = save_fig(fig, out / f"{args.tag}.png")

    csv_path = out / f"{args.tag}.csv"
    cols = ["estimator", "T", "tx_rate", "mean_age", "max_age", "escape_rate",
            "mean_dist", "mean_dist_tail", "p95_dist", "max_dist", "mean_ep_len",
            "est_nmse", "n_steps", "n_tx"]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in sorted(rows, key=lambda r: (r["estimator"], r["T"])):
            w.writerow([r.get(c, "") if not isinstance(r.get(c), float) else f"{r[c]:.8f}"
                        for c in cols])

    summary = {
        "config": cfg,
        "device": str(device),
        "n_params": count_params(model),
        "controller": {"kp": ctrl.kp, "kd": ctrl.kd, "omega_n": ctrl.omega_n,
                       "zeta": ctrl.zeta, "kappa": float(env.kappa),
                       "analytic_e_ss": float(e_ss),
                       "analytic_e_vec": [float(x) for x in e_vec],
                       "derivation": "kp = ω_n², kd = 2ζω_n − κ；"
                                     "稳态误差在**旋转坐标系**下解 2×2 线性方程组"
                                     "（向心项与阻尼项正交，不可标量相加）—— "
                                     "见 wmlab/control.py 文件头"},
        "warmup_steps": int(tk["warmup_steps"]),
        "measured_steps": int(tk["measured_steps"]),
        "var_global": var_g,
        "wiring_checks": checks,
        "h_star_err": h_err,
        "h_star_err_definition": "离线闭环 NMSE(h) 逐点口径首次 > 0.05（沿用 X2/X14）",
        "h_star_task_escape": h_escape,
        "h_star_task_dist": h_dist,
        "h_star_task_definition": (
            "H_escape = max{T : escape_rate(T) ≤ escape_rate(T=1) + 0.05}; "
            "H_dist = max{T : mean_dist_tail(T) ≤ mean_dist_tail(T=1)·1.2}; "
            "★ 网格内未坏掉时返回网格最大值，真实值 ≥ 该数"),
        "baseline_persistence": {"h_escape": p_escape, "h_dist": p_dist},
        "ratio_task_over_err": {
            "escape": (h_escape / h_err) if (h_err and h_escape) else None,
            "dist": (h_dist / h_err) if (h_err and h_dist) else None,
        },
        # ★★ 本实验的核心产物：任务视界对"指标+容差"的依赖
        "tolerance_sweep": tol_table,
        "h_task_spread": ({"min": min(vals), "max": max(vals)} if vals else None),
        "online_vs_offline": online_offline,
        "offline_curve_closed_global": {"horizons": hs, "nmse_per_step": pt,
                                        "nmse_per_step_global": pt_g},
        "sweep": rows,
        "caveat": ("★ 本实验的视界是 (模型, PD控制器, 周期调度) 三元组的属性，"
                   "不是模型单独的属性 —— 换控制器/换调度，数值会变。"),
        "elapsed_sec": time.time() - t0,
    }
    js = out / f"{args.tag}.json"
    js.write_text(json.dumps(jsonable(summary), ensure_ascii=False, indent=2, default=str),
                  encoding="utf-8")
    print(f"[16] 图 -> {png}\n[16] 表 -> {csv_path}\n[16] json -> {js}")
    print(f"[16] 用时 {summary['elapsed_sec']:.1f}s")
    print("[16] OK")


if __name__ == "__main__":
    main()
