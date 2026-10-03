# -*- coding: utf-8 -*-
"""★ X44「发送时机扰动」：**固定传输预算**下，任务指标 vs 估计指标对**发送时刻**的敏感度

================================================================= 由来（为什么必须有这一号）
X38–X43 把「换触发统计量」这条路走完了：
    周期 → 年龄阈值 → 自报 U → 残差 z → 共形信封 → **oracle（真值误差的上界）**
而 X43 的结论是 **O0 成立**：连「完美知道当前估计误差」都赢不了年龄阈值。
⇒ 定位为「**瓶颈在任务侧**」。但那是**间接**推断（换量打不破）。
本号把它变成**直接**测量 —— **固定发送预算，只动发送时刻**，看
    ① 估计指标 `est_nmse` 动多少    ② 任务指标 `mean_dist_tail` 动多少。
若「估计动得多、任务几乎不动」 ⇒ 「任务侧受限」被坐实
（= X35「NMSE 不能当任务代理」的直接延伸）。

================================================================= 与 X38–X43 的关系（消融封死混淆变量）
- **载体 = 周期调度**（`periodic_lossy`）：无状态、预算可**逐位固定**
  ⇒ 唯一自变量就是「时刻」。
- ★★ **PER = 0（无丢包）是刻意选择**：有丢包时，不同的时刻序列会消耗**不同的随机数流**
  ⇒ 丢包实现与「时机」产生**虚假相关**（混淆变量）。p=0 把时机**干净地**隔离出来。
  （"加丢包更真实"的直觉在这里是错的：它会引进一个与被测量无关的混淆项。）
- **预算硬约束**：同一 T 下**所有扰动的发送次数逐位相同**（= `n_tx`，由构造给出 + 逐 episode 断言）。
  ⇒ 这不是"大致相同"，是"逐位相同"（铁律 17 的精神：不可评估/不可比的要剔除，不许凑）。

================================================================= 三个旋钮（都保持恰好 N 次发送）
- **相位 Δ**（确定性整体提前/推后 Δ 步）：`t_k = Δ + k·T`，Δ ∈ [0, T)。
- **抖动 j**（每次发送提前/推后至多 j 步；有界 j ≤ (T−1)//2 ⇒ 天然不乱序）：`t_k = k·T + u_k`。
- **完全随机时机**（离散度上界）：在窗口 [0, N·T) 内取 N 个**均匀随机并排序**的时刻。
⇒ 「时刻离散度」的单调阶梯：**周期 → 抖动 → 随机**。

================================================================= 预写判据（跑前定死，不许事后编解释）
T1  相位免疫：相位扫描里 `max|dist/周期 − 1| ≤ 0.05`（周期内相位不影响任务）。
T2  ★核心：抖动/随机扫描里，「估计相对升幅 ≥ 15%」**而**「任务相对升幅 ≤ 5%」
    ⇒ 任务距离对发送时机近乎免疫 ⇒ **任务侧受限坐实**。
T2' 反例：若任务与估计**同量级**上升 ⇒ 时机敏感 ⇒ X43 的「打平」另有解释（照报）。
T3  预算曲线（参考）：T 从 1 到 32 的 `est_nmse` 与 `mean_dist_tail`，把
    「时机效应」与「预算效应」放到**同一尺度**上（给出全局兑换率）。

================================================================= 自检（不过就抛 —— R14）
S_t1 R12 接线①：**逐 episode 的发送次数**在全部扰动下**逐位相同**（= N）。
S_t2 R12 接线②：扰动的**送达时刻序列确实与周期基准不同**（否则 = 没接线）。
S_t3 周期自洽：相位 Δ=T ≡ Δ=0 **逐位相同**；抖动 j=0 ≡ 周期。
S_t4 R12 接线③：预算参考里 **T=1 ⇒ est_nmse ≈ 0**（每步送真值 ⇒ 误差恒为 0）。
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
from wmlab.eval.timing import (expected_attempts, lookup_schedule,  # noqa: E402
                               n_tx_budget, plan_times, subsample,
                               timing_dispersion)
from wmlab.eval.tracking import periodic_lossy_schedule  # noqa: E402
from wmlab.models.prob_world_model import GaussianWorldModel  # noqa: E402
from wmlab.train import train_world_model  # noqa: E402
from wmlab.utils import count_params, get_device, load_config, output_dir, set_seed  # noqa: E402
from wmlab.utils.plot import PALETTE, apply_style, save_fig  # noqa: E402


# ★ 时刻规划与接线检查已抽成库模块 `wmlab.eval.timing`
#   （`plan_times` / `n_tx_budget` / `expected_attempts` / `timing_dispersion`）
#   —— 见该模块头部说明：预算的**逐位相等**是 X44 全部结论的前提，必须被回归测试锁住。


# ================================================================= 主流程
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/uav_triggered.yaml")
    p.add_argument("--tag", default="28_x44_timing")
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

    print("[28] " + "=" * 78)
    print("[28] ★ X44 发送时机扰动：固定预算下 估计指标 vs 任务指标 对「发送时刻」的敏感度")
    print(f"[28]   T 网格（预算参考）  n_ep={n_task_ep}  meas_steps={meas_steps}  "
          f"warmup={WARMUP}  PER=0（刻意）")
    print("[28]   预写：T1 相位免疫(≤5%)  T2 抖动/随机『估计↑≥15% 而 任务↑≤5%』⇒ 任务侧受限")

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

    cache = tmp / "x44_model.pt"
    model = GaussianWorldModel(
        obs_dim=obs_dim, act_dim=int(env.act_dim),
        latent_dim=int(cfg["model"]["latent_dim"]),
        hidden=int(cfg["model"]["hidden"]),
        discrete_act=bool(env.is_discrete),
        logvar_init=float(cfg["model"].get("logvar_init", 0.0))).to(device)
    if cache.is_file() and not args.retrain:
        model.load_state_dict(torch.load(cache, map_location=device))
        print(f"[28] 载入模型缓存 {cache}")
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
        print(f"[28] 训练完成：params={count_params(model):,} "
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

    def run(sched):
        return run_closed_loop_control(
            env, model, ctrl, sched, n_episodes=n_task_ep, seed=online_seed,
            max_steps=meas_steps, device=device, estimator="model",
            var_g=var_g, label=getattr(sched, "__name__", "sched"),
            tail_frac=tail_frac, warmup_steps=WARMUP)

    # ============================================================ 2) Part D：预算参考
    print("[28] " + "-" * 78)
    print("[28] Part D · 预算参考曲线（周期 T 扫描）：把『时机效应』放到『预算效应』尺度上")
    T_ref = [int(x) for x in (args.periods.split(",") if args.periods else
                              ([1, 2, 4, 8, 16] if args.quick else [1, 2, 4, 8, 16, 32]))]
    budget_rows = []
    for T in T_ref:
        r = run(periodic_lossy_schedule(T, 0.0))
        budget_rows.append({"kind": "budget", "T": T, "est_nmse": float(r["est_nmse"]),
                            "mean_dist_tail": float(r["mean_dist_tail"]),
                            "mean_age": float(r["mean_age"]),
                            "tx_rate": float(r["tx_rate"]), "n_tx": int(r["n_tx"])})
        print(f"[28]   T={T:<3d} nmse={r['est_nmse']:.4f}  dist={r['mean_dist_tail']:.4f}  "
              f"E[age]={r['mean_age']:.3f}  tx={r['tx_rate']:.4f}  n_tx={r['n_tx']}")

    # ★ S_t4：T=1 ⇒ 每步送真值 ⇒ NMSE 恒 0
    nmse_t1 = [b["est_nmse"] for b in budget_rows if b["T"] == 1]
    if nmse_t1 and nmse_t1[0] > 1e-9:
        raise AssertionError(f"★ S_t4 未通过：T=1 的 est_nmse={nmse_t1[0]:.3e} 应恒为 0"
                             "（每步送真值）⇒ 接线有问题")
    print("[28]   ✓ S_t4：T=1 ⇒ est_nmse 恒 0（接线成立）")

    # ============================================================ 3) Part A/B：时机扰动
    print("[28] " + "-" * 78)
    print("[28] Part A/B · 时机扰动（相位 Δ / 抖动 j / 完全随机），标称预算逐位固定 = N")
    print("[28]   ⚠ 闭环 episode 会因跟丢**提前终止** ⇒ 实测次数 = Σ_e #{1 ≤ t_k ≤ L_e}"
          "（由 ep_lens 逐位核对，R12；注意调度首次被查询在 t=1，t=0 不出现）")
    T_list = [int(x) for x in (args.periods.split(",") if args.periods else
                               ([8] if args.quick else [8, 16]))]
    rows = []
    checks = []
    BUDGET_TOL = 0.05     # 实测送达率相对基准的容许偏差（铁律 17：超了 ⇒ 该档不可评估）
    for T in T_list:
        N = n_tx_budget(T, meas_steps)
        jmax = (T - 1) // 2
        base_times = plan_times("phase", T, 0, meas_steps, N)
        r_base = run(lookup_schedule(base_times, f"periodic(T={T})"))
        exp_base = expected_attempts(base_times, r_base["ep_lens"])
        if int(r_base["n_tx"]) != exp_base:
            raise AssertionError(
                f"★ S_t1 未通过（基准）：n_tx={r_base['n_tx']} ≠ 由 ep_lens 反算 {exp_base}"
                " ⇒ 时刻表没接线，或有别的量在偷偷改预算")
        lb = list(r_base["ep_lens"])
        print(f"[28]   T={T}  N={N}  基准：nmse={r_base['est_nmse']:.4f} "
              f"dist={r_base['mean_dist_tail']:.4f}  tx={r_base['tx_rate']:.5f}  "
              f"L̄={np.mean(lb):.1f}/{meas_steps} 短集={np.mean(np.array(lb) < meas_steps):.0%}")
        # ★ S_t3 之一：相位 Δ=T ≡ Δ=0（周期自洽）
        ts_T = plan_times("phase", T, T, meas_steps, N)
        if ts_T != base_times:
            raise AssertionError(f"★ S_t3 未通过：相位 Δ=T 的时刻序列 ≠ Δ=0（T={T}）")
        rows.append({"kind": "base", "T": T, "mode": "periodic", "param": 0,
                     "n_tx_target": N, "n_tx": int(r_base["n_tx"]),
                     "disp": timing_dispersion(base_times),
                     "budget_ok": True, "rate_dev": 0.0,
                     "ep_len_mean": float(np.mean(lb)),
                     "short_frac": float(np.mean(np.array(lb) < meas_steps)),
                     "est_nmse": float(r_base["est_nmse"]),
                     "mean_dist_tail": float(r_base["mean_dist_tail"]),
                     "mean_age": float(r_base["mean_age"]),
                     "tx_rate": float(r_base["tx_rate"])})

        specs = []
        for d in subsample(0, T - 1, 5):
            specs.append(("phase", d))
        for j in subsample(0, jmax, 5):
            specs.append(("jitter", j))
        for p in range(1 if args.quick else 3):
            specs.append(("random", p))

        for mode, param in specs:
            if mode == "random":
                ts = plan_times("random", T, param, meas_steps, N, pattern_seed=param + 1)
                name = f"rand(T={T},#{param})"
            else:
                ts = plan_times(mode, T, param, meas_steps, N)
                name = f"{mode}(T={T},{param})"
            # ★ S_t2：扰动时刻必须真的不同于基准
            #   （**恒等档**除外：phase Δ=0 与 jitter j=0 按定义 ≡ 周期，不许当"没接线"）
            is_identity = ((mode == "phase" and int(param) % T == 0)
                           or (mode == "jitter" and int(param) == 0))
            if not is_identity and ts == base_times:
                raise AssertionError(
                    f"★ S_t2 未通过：{name} 的时刻序列与基准**逐位相同** ⇒ 扰动没接线")
            r = run(lookup_schedule(ts, name))
            # ★ S_t1：次数必须**恰好**由「时刻表 × 各集长度」解释（R12 真接线）
            exp = expected_attempts(ts, r["ep_lens"])
            if int(r["n_tx"]) != exp:
                raise AssertionError(
                    f"★ S_t1 未通过：{name} n_tx={r['n_tx']} ≠ 由 ep_lens 反算 {exp}"
                    " ⇒ 有未接线的量在改预算")
            rate_dev = abs(float(r["tx_rate"]) / float(r_base["tx_rate"]) - 1.0)
            budget_ok = bool(rate_dev <= BUDGET_TOL)
            lp = list(r["ep_lens"])
            rows.append({"kind": "perturb", "T": T, "mode": mode, "param": int(param),
                         "n_tx_target": N, "n_tx": int(r["n_tx"]),
                         "disp": timing_dispersion(ts),
                         "budget_ok": budget_ok, "rate_dev": float(rate_dev),
                         "ep_len_mean": float(np.mean(lp)),
                         "short_frac": float(np.mean(np.array(lp) < meas_steps)),
                         "est_nmse": float(r["est_nmse"]),
                         "mean_dist_tail": float(r["mean_dist_tail"]),
                         "mean_age": float(r["mean_age"]),
                         "tx_rate": float(r["tx_rate"])})
            flag = "" if budget_ok else "  ⚠不可评估(预算偏差)"
            print(f"[28]     {name:<20s} nmse={r['est_nmse']:.4f} "
                  f"dist={r['mean_dist_tail']:.4f}  E[age]={r['mean_age']:.3f}  "
                  f"tx={r['tx_rate']:.5f}  Δrate={rate_dev:+.2%}{flag}")
            if not budget_ok:
                print(f"[28]       └ 原因：episode 提前终止使实测送达率偏离 {rate_dev:.1%}"
                      "（机制见 expected_attempts 注释）；该档**不计入判据**")
        # ★ S_t3 之二：抖动 j=0 ≡ 周期
        if plan_times("jitter", T, 0, meas_steps, N) != base_times:
            raise AssertionError(f"★ S_t3 未通过：jitter j=0 的时刻 ≠ 周期（T={T}）")
        checks.append({"T": T, "n_tx": int(r_base["n_tx"]), "N_target": N, "jmax": jmax})
    env.close()

    # ============================================================ 4) 判据结算
    print("[28] " + "-" * 78)
    print("[28] 判据结算")
    verdict = {}
    for T, ck in zip(T_list, checks):
        base = next(r for r in rows if r["T"] == T and r["kind"] == "base")
        b_nmse, b_dist = base["est_nmse"], base["mean_dist_tail"]
        allp = [r for r in rows if r["T"] == T and r["kind"] == "perturb"]
        pert = [r for r in allp if r["budget_ok"]]           # ★ 只算**可评估**的档
        n_dropped = len(allp) - len(pert)
        if n_dropped:
            print(f"[28]   T={T}: {n_dropped}/{len(allp)} 个扰动档因**预算偏差 >5%**"
                  " 被剔除（原因=episode 提前终止，非静默跳过）")
        ph = [r for r in pert if r["mode"] == "phase"]
        dj = [r for r in pert if r["mode"] in ("jitter", "random")]
        dev_ph_dist = max(abs(r["mean_dist_tail"] / b_dist - 1.0) for r in ph) if ph else float("nan")
        # 抖动/随机：取"估计升幅最大"那一档，看任务升幅
        worst = max(dj, key=lambda r: r["est_nmse"]) if dj else None
        nmse_up = (worst["est_nmse"] / b_nmse - 1.0) if worst else float("nan")
        dist_up = (worst["mean_dist_tail"] / b_dist - 1.0) if worst else float("nan")
        # ★ 兑换率：Δ任务 / Δ估计（斜率越小 ⇒ 任务侧越不敏感）
        slope = (dist_up / nmse_up) if (worst and abs(nmse_up) > 1e-12) else float("nan")
        # ★★ 稳健版（不依赖"最坏那一点"）：全部可评估扰动档上的离散度与相关系数
        nm_a = np.array([r["est_nmse"] for r in pert], dtype=float)
        ds_a = np.array([r["mean_dist_tail"] for r in pert], dtype=float)
        cv_nmse = float(nm_a.std() / nm_a.mean()) if nm_a.size >= 2 else float("nan")
        cv_dist = float(ds_a.std() / ds_a.mean()) if ds_a.size >= 2 else float("nan")
        ratio_cv = float(cv_dist / cv_nmse) if (cv_nmse and np.isfinite(cv_nmse)
                                                and cv_nmse > 1e-12) else float("nan")
        corr = float(np.corrcoef(nm_a, ds_a)[0, 1]) if nm_a.size >= 3 else float("nan")
        # ★ 判据分区（**完整划分**，跑正式版之前就定死；冒烟只用于确认机制、不作证据）
        # ★ 判据分区（**完整划分**：符号 × 幅度，五种情况互斥且穷尽）
        #   ★★ 2026-10-04 如实登记：**首跑之后**才发现分区缺一档 ——
        #      正式跑落在「**同向亚线性**」（估计 +518% 而任务只 +44%），
        #      既不是 T2（任务免疫 ≤5%）也不是 T2′（近线性）⇒ 老分区把它标成
        #      "弱／不可判"，那是**标注假象**，不是结论。
        #      ⇒ 教训（已入自证伪表）：判据分区**必须覆盖"连续量之间的定量关系"
        #        （斜率 / CV 比）**，只用"符号 + 拍一个阈值"的组合会漏掉最常见的那一档。
        same_dir = bool(nmse_up * dist_up > 0)
        if (not same_dir) and abs(dist_up) > 0.05:
            outcome = "T2″ 解耦（异号：估计更差反而任务更好）"
        elif nmse_up >= 0.15 and abs(dist_up) <= 0.05:
            outcome = "T2 任务免疫（估计动了、任务没动）"
        elif same_dir and dist_up > 0.5 * nmse_up:
            outcome = "T2′ 时机敏感（估计与任务同向近线性）"
        elif same_dir and dist_up > 0.05:
            outcome = "T2‴ 亚线性衰减（任务跟涨，但远小于估计）"
        else:
            outcome = "弱（|Δ任务| ≤5%，效应量不足）"
        v = {
            "T": T, "base_est_nmse": b_nmse, "base_dist": b_dist,
            "n_perturb_evaluable": len(pert), "n_perturb_dropped": int(n_dropped),
            "T1_phase_max_dist_dev": float(dev_ph_dist),
            "T1_pass": bool(dev_ph_dist <= 0.05),
            "worst_mode": (worst["mode"] + f"({worst['param']})") if worst else None,
            "nmse_rel_up": float(nmse_up), "dist_rel_up": float(dist_up),
            "exchange_slope": float(slope),
            "cv_nmse": cv_nmse, "cv_dist": cv_dist, "ratio_cv": ratio_cv,
            "corr_nmse_dist": corr, "outcome": outcome,
            "T2_pass": bool(nmse_up >= 0.15 and abs(dist_up) <= 0.05),
            "T2prime_pass": bool(same_dir and dist_up > 0.5 * nmse_up),
            "T2pp_pass": bool((not same_dir) and abs(dist_up) > 0.05),
        }
        verdict[str(T)] = v
        print(f"[28]   T={T}:  T1 相位最大任务偏差 {dev_ph_dist:.2%} ⇒ "
              f"{'PASS' if v['T1_pass'] else 'FAIL'}")
        print(f"[28]     抖动/随机最坏档 {v['worst_mode']}: 估计 {nmse_up:+.1%} / "
              f"任务 {dist_up:+.1%}（Δ任务/Δ估计={slope:+.3f}）⇒ {outcome}")
        print(f"[28]     稳健统计：CV(估计)={cv_nmse:.4f}  CV(任务)={cv_dist:.4f}  "
              f"比值={ratio_cv:.3f}  ρ(估计,任务)={corr:+.3f}")

    # ★ S_t1/S_t2/S_t3 汇总
    print("[28]   ✓ S_t1（次数=Σ_e#{t_k<L_e} 逐位）· S_t2（扰动确实改时刻）·"
          " S_t3（周期自洽）· S_t4（T=1 NMSE=0）全部通过")

    # ============================================================ 5) 图 + JSON
    stem = args.tag
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    T0 = T_list[0]
    for T in T_list:
        base = next(r for r in rows if r["T"] == T and r["kind"] == "base")
        pert = [r for r in rows
                if r["T"] == T and r["kind"] == "perturb" and r["budget_ok"]]
        xs = [r["disp"] for r in pert]
        axes[0].scatter(xs, [r["est_nmse"] / base["est_nmse"] for r in pert],
                        s=42, alpha=0.85, label=f"T={T}")
        axes[1].scatter(xs, [r["mean_dist_tail"] / base["mean_dist_tail"] for r in pert],
                        s=42, alpha=0.85, label=f"T={T}")
        axes[2].scatter([r["est_nmse"] / base["est_nmse"] for r in pert],
                        [r["mean_dist_tail"] / base["mean_dist_tail"] for r in pert],
                        s=42, alpha=0.85, label=f"T={T}")
    for ax, ttl, yl in zip(axes[:2],
                           ["est_nmse vs timing dispersion",
                            "task distance vs timing dispersion"],
                           ["est_nmse / periodic", "mean_dist_tail / periodic"]):
        ax.axhline(1.0, color="k", lw=0.9, alpha=0.5)
        ax.set_xlabel("CV(inter-arrival gaps)   -- fixed budget")
        ax.set_ylabel(yl)
        ax.set_title(ttl, fontsize=11)
        ax.legend(fontsize=8)
    axes[2].axhline(1.0, color="k", lw=0.9, alpha=0.5)
    axes[2].axvline(1.0, color="k", lw=0.9, alpha=0.5)
    axes[2].set_xlabel("est_nmse / periodic")
    axes[2].set_ylabel("mean_dist_tail / periodic")
    axes[2].set_title("estimation gain vs task gain", fontsize=11)
    axes[2].legend(fontsize=8)
    fig.suptitle(f"X44 transmission-timing perturbation (fixed budget)  "
                 f"tag={args.tag}  T={T_list}  PER=0  n_ep={n_task_ep}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_fig(fig, out / f"{stem}.png")
    plt.close(fig)

    payload = {
        "experiment": "X44",
        "title": "发送时机扰动：固定预算下 估计指标 vs 任务指标 对发送时刻的敏感度",
        "config": args.config, "seed": seed, "n_task_ep": n_task_ep,
        "meas_steps": meas_steps, "warmup_steps": WARMUP, "per": 0.0,
        "tail_frac": tail_frac, "var_g": var_g,
        "model_params": int(count_params(model)),
        "train_epochs": int(cfg["train"]["epochs"]),
        "data_n_episodes": int(cfg["data"]["n_episodes"]),
        "model_cache": str(cache), "quick": bool(args.quick),
        "T_list": T_list, "T_ref": T_ref,
        "budget_rows": budget_rows, "rows": rows, "verdict": verdict, "checks": checks,
        "note_timing_budget": "同一 T 下全部扰动共用同一 n_tx（逐 episode 断言）",
        "note_per0": "PER=0 是刻意选择：有丢包时随机数流与时刻序列混淆",
    }
    (out / f"{stem}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[28] 产物：{out / (stem + '.json')} / {stem}.png   总耗时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
