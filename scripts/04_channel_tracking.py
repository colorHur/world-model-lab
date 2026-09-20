#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""脚本 04 —— 把通信接进来（X24 / X25 / X26）

计划书：`04-实验与代码/2026-09-19-通信接入计划-v0.1.md`

它回答三个问题
--------------
X24  信道层**有没有偷偷改动力学**？            —— 三层等价性检验（E1 / E2 / E3）
X25  把 H\\* 当成"多久必须传一次"的参数，它值不值？ —— 传输率–NMSE 帕累托前沿（两个调度族）
X26  离线测的 NMSE(h) 曲线**能不能预测在线跟踪误差**？ —— 解析 vs 仿真 gap + Jensen 偏差

★ 三条口径声明（R1 / R2 / R3）
------------------------------
1. 在线跟踪用的"预测顶上"= `model.predict_next`（encode→transition→decode），
   与 X2 的**观测空间闭环**曲线机制逐次相同 ⇒ X26 的解析对账用的是**闭环 NMSE(h)**。
2. NMSE 一律用**全局 pooled 方差**归一（`denom=var_global`），
   否则离线曲线（按切片算方差）与在线跟踪（按整批算方差）数值不可比。
3. 所有结论先算数、再看图（R2）：关键量在 `outputs/04_channel_*.csv` 里逐行可核。

★★ 第四条口径（2026-09-20 补，X26 对账发现的真 bug）
---------------------------------------------------
在线 `run_tracking` 累计的是**每一步自身**的平方误差 ⇒ 属**逐点**口径。
离线曲线必须与之同口径，否则解析式会"凭空低估"仿真。历史上 04 用的是**累积**口径
（`mse(pred[:, :h], tgt[:, :h])`），前 h-1 步的小误差被摊进均值 ⇒ 离线曲线偏低，
于是得到"解析低估仿真 56%"的假缺口。改口径后缺口收敛到个位数百分比。
H* 同样改用逐点（『首次超阈』的定义口径），累计口径仅保留作历史对照。

运行
----
    python scripts/04_channel_tracking.py --config configs/channel.yaml
    python scripts/04_channel_tracking.py --config configs/channel.yaml --quick   # 冒烟
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from wmlab.data import collect_random_episodes, split_episodes, transitions_from_episodes
from wmlab.envs import make_env, make_env_with_channel
from wmlab.eval import (NmseCurve, geometric_age_pmf, geometric_age_stats, jensen_gap,
                        loss_prob_for_tail_risk, lossy_schedule, periodic_schedule,
                        reliable_horizon, run_tracking, uniform_age_pmf,
                        uniform_age_stats)
from wmlab.models import MLPWorldModel
from wmlab.rollout import closed_loop_error_curve, multi_step_error_curve
from wmlab.train import train_world_model
from wmlab.utils import (count_params, describe_device, get_device, load_config,
                         output_dir, set_seed)
from wmlab.utils.plot import PALETTE, apply_style, save_fig


# ===================================================================== 工具
def parse_args():
    p = argparse.ArgumentParser(description="通信接入：X24 等价性 / X25 调度 / X26 解析对账")
    p.add_argument("--config", default="configs/channel.yaml")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--tag", default="04_channel")
    p.add_argument("--quick", action="store_true",
                   help="冒烟模式：小模型 / 少数据 / 少调度点（只为验证管线跑通）")
    p.add_argument("--skip-equivalence", action="store_true",
                   help="跳过 X24 的 CartPole 等价性检验（它要额外训两个模型）")
    return p.parse_args()


def pooled_var(episodes) -> float:
    """整批观测的 pooled 方差 —— 全局唯一的 NMSE 归一化分母。"""
    allobs = np.concatenate([e["obs"] for e in episodes], axis=0).astype(np.float64)
    return float(allobs.var())


def clamp_horizons(horizons, episodes, label="") -> list[int]:
    """按 episode 实际长度裁剪 horizons。

    ★ 为什么必须显式做这件事（R3）：`_sample_windows` 要求 `T - horizon - 1 > 0`，
    否则直接抛异常。**而"视界被数据卡住"这件事本身会改变结论的含义** ——
    CartPole 那轮"开环 > 15 步"里的 15 就是数据上限，不是模型上限。
    所以这里**打印被丢弃的视界**，让每一个被截断的上限都留下痕迹，不许静默。
    """
    max_len = max(len(e["act"]) for e in episodes)
    limit = max_len - 2
    kept = [int(h) for h in horizons if int(h) <= limit]
    dropped = [int(h) for h in horizons if int(h) > limit]
    if dropped:
        print(f"[04] ★ horizons 受**数据上限**裁剪{('（' + label + '）') if label else ''}："
              f"最长 episode={max_len} 步 → 上限 {limit}，丢弃 {dropped}")
    if not kept:
        kept = [max(1, limit)]
    return sorted(kept)


def jsonable(o):
    """把 numpy 标量/数组转成原生类型 —— json 是给人读的产物，不该出现 "np.float64(...)"。"""
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


def train_new_model(cfg, env_id, episodes, device, seed, act_dim, discrete_act, label=""):
    """按给定 episode 训练一个新世界模型，返回 (model, hist, rh_open, rh_closed)。

    ★ `act_dim` / `discrete_act` 必须由**调用方从环境对象**读出后传入。
      不能从 episode 数据推断：`sample_action` 无论离散还是连续都返回形状 (1,) 的数组，
      推断出来的 act_dim 恒为 1 —— 对 CartPole（2 个动作）会让 one_hot 直接越界。
      （2026-09-19 首次完整跑时的崩溃点，冒烟没覆盖到，因为冒烟只走了 Pendulum 分支。）
    """
    set_seed(seed)
    train_eps, val_eps = split_episodes(episodes, float(cfg["train"]["val_ratio"]), seed)
    tr = tuple(torch.as_tensor(x) for x in transitions_from_episodes(train_eps))
    va = tuple(torch.as_tensor(x) for x in transitions_from_episodes(val_eps))

    obs_dim = int(episodes[0]["obs"].shape[1])

    model = MLPWorldModel(
        obs_dim=obs_dim, act_dim=int(act_dim),
        latent_dim=int(cfg["model"]["latent_dim"]), hidden=int(cfg["model"]["hidden"]),
        discrete_act=bool(discrete_act),
    ).to(device)

    if label:
        print(f"  [{label}] obs_dim={obs_dim} act_dim={act_dim} params={count_params(model):,} "
              f"train={tr[0].shape[0]} val={va[0].shape[0]}")
    hist = train_world_model(model, tr, va, cfg, device, verbose=False)

    hs = clamp_horizons(list(cfg["eval"]["horizons"]), val_eps, label=label)
    thr = float(cfg["eval"]["err_threshold"])
    mode = cfg["eval"].get("threshold_mode", "rel")
    co = multi_step_error_curve(model, val_eps, hs, device,
                                n_samples=int(cfg["eval"]["n_samples"]), seed=seed)
    cc = closed_loop_error_curve(model, val_eps, hs, device,
                                     n_samples=max(64, int(cfg["eval"]["n_samples"]) // 2), seed=seed)

    # ★ 口径：H* 用**逐点**（"首次超阈"的定义口径）；累积口径一并算出，
    #   只用于与历史记录（E2 的 reference_closed_h = 10.1073…）对齐 —— 历史值是累积口径。
    #   ★★ 单位必须与阈值同源：阈值 0.05 是归一化（nmse）口径 ⇒ 逐点序列取 `nmse_per_step`。
    #      第一版误取 `mse_per_step`，在 CartPole 上让逐点 H* 反而大于累积 H*，
    #      与"累积 = 逐点的运行平均"矛盾 —— 由此查出（详见 02 的同名注释）。
    def _pt(curve):
        v = np.asarray(curve["nmse_per_step"], dtype=float)
        return [float(v[h - 1]) for h in curve["horizons"]]

    rh = {"open_pointwise": reliable_horizon(co["horizons"], _pt(co), thr, mode),
          "closed_pointwise": reliable_horizon(cc["horizons"], _pt(cc), thr, mode)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")          # 有意复现历史口径，不刷警告
        rh["open_cumulative"] = reliable_horizon(co["horizons"], co["nmse"], thr, mode,
                                                 calibration="cumulative")
        rh["closed_cumulative"] = reliable_horizon(cc["horizons"], cc["nmse"], thr, mode,
                                                   calibration="cumulative")
    return model, hist, rh


# ===================================================== X24 · 等价性检验
def check_equivalence(cfg, device, seed):
    """三层等价性检验：信道层不得改变任何已有结论。

    E1  动力学逐位一致：p=0 时 ChannelAdapter 采出的 obs 序列与裸 env bit-exact
    E2  端到端可复现：用 p=0 信道重训一个 CartPole 世界模型，H* 必须与裸 env 逐位一致
    E3  记账正确：p=0 时 age 恒 0；有限 p 时实测 E[age] 收敛到理论 p/(1-p)
    """
    ec = cfg["channel"]["equivalence"]
    env_id = ec["env_id"]
    n_ep = int(ec["n_episodes"])
    print(f"[E] 等价性检验 env={env_id} n_episodes={n_ep}")

    # ---------- E1 动力学 bit-exact ----------
    def collect(env, n_ep, seed):
        rng = np.random.default_rng(seed)
        out = []
        for i in range(n_ep):
            seq = [env.reset(seed=seed + i)]
            while True:
                res = env.step(env.sample_action(rng))
                seq.append(res.obs)
                if res.done:
                    break
            out.append(np.asarray(seq, dtype=np.float64))
        return out

    e_bare = make_env(env_id, seed=seed)
    e_ch = make_env_with_channel(env_id, seed=seed, loss_prob=0.0, delay=0)
    A = collect(e_bare, 3, seed + 7)
    B = collect(e_ch, 3, seed + 7)
    e1_max = max(float(np.max(np.abs(a - b))) for a, b in zip(A, B))
    e1_bit = all(np.array_equal(a, b) for a, b in zip(A, B))
    e_bare.close(); e_ch.close()
    print(f"[E1] 动力学 bit-exact = {e1_bit}   max|diff| = {e1_max}")
    assert e1_bit and e1_max == 0.0, "★ E1 未通过 —— 信道层改变了动力学，后续全部作废"

    # ---------- E2 端到端复现 ----------
    # ★ act_dim / is_discrete 必须从环境对象读（见 train_new_model 的 docstring）
    env_bare = make_env(env_id, seed=seed)
    n_act_dim, is_disc = int(env_bare.act_dim), bool(env_bare.is_discrete)
    eps_bare = collect_random_episodes(env_bare, n_episodes=n_ep,
                                       seed=seed, max_steps=cfg["env"].get("max_steps"))
    env_bare.close()
    env_c = make_env_with_channel(env_id, seed=seed, loss_prob=0.0, delay=0)
    eps_ch = collect_random_episodes(env_c, n_episodes=n_ep,
                                     seed=seed, max_steps=cfg["env"].get("max_steps"))
    env_c.close()
    print(f"[E2] env={env_id} act_dim={n_act_dim} discrete={is_disc}")
    data_bit = all(np.array_equal(a["obs"], b["obs"]) and np.array_equal(a["act"], b["act"])
                   for a, b in zip(eps_bare, eps_ch))
    print(f"[E2] 采集数据 bit-exact = {data_bit}")

    ec_cfg = {"model": cfg["model"], "train": dict(cfg["train"]), "eval": cfg["eval"], "seed": seed}
    ec_cfg["train"]["epochs"] = int(ec["epochs"])
    m_bare, _, rh_b = train_new_model(
        ec_cfg, env_id, eps_bare, device, seed, act_dim=n_act_dim,
        discrete_act=is_disc, label="E2-bare")
    m_ch, _, rh_c = train_new_model(
        ec_cfg, env_id, eps_ch, device, seed, act_dim=n_act_dim,
        discrete_act=is_disc, label="E2-channel(p=0)")
    rho_b, rho_c = rh_b["open_pointwise"], rh_c["open_pointwise"]
    rhc_b, rhc_c = rh_b["closed_pointwise"], rh_c["closed_pointwise"]
    e2_open = (rho_b is not None and rho_c is not None and rho_b == rho_c)
    e2_closed = (rhc_b is not None and rhc_c is not None and rhc_b == rhc_c)
    # ★ 历史参考值 10.1073… 是**累积口径**下的数，因此对照必须用累积值。
    #   ★★ 2026-09-20 二次发现：即便用累积口径，本轮实测也是 10.763 而非 10.107 ——
    #      说明该"历史记录值"**没有绑定它的测量条件**（哪份 config 的 horizons / n_samples /
    #      eval 集？），因此它**不能作为 pass 判据**。E2 真正要验的是"p=0 信道 == 裸 env"
    #      的逐位一致（下面 e2_open/e2_closed），那一条是自洽的、与历史无关。
    #      参考值只作信息项打印。这与本次口径问题属于**同一类根因**：
    #      **数字被记下来时没带上它的测量条件**。
    ref = float(ec.get("reference_closed_h", float("nan")))
    rhc_b_cum = rh_b["closed_cumulative"]
    e2_ref_same = (rhc_b_cum is not None and abs(rhc_b_cum - ref) < 1e-9)
    print(f"[E2] 裸 env      H*: 开环={rho_b}  闭环={rhc_b}（逐点）"
          f"   [累积: 开环={rh_b['open_cumulative']} 闭环={rhc_b_cum}]")
    print(f"[E2] p=0 信道    H*: 开环={rho_c}  闭环={rhc_c}（逐点）"
          f"   [累积: 开环={rh_c['open_cumulative']} 闭环={rh_c['closed_cumulative']}]")
    print(f"[E2] ★ 逐位一致（裸 vs p=0 信道）= {e2_open and e2_closed}  ← 这才是 E2 的判据")
    print(f"[E2]   （信息项）与 config 里记录的历史值 {ref} 相同 = {e2_ref_same}，"
          f"本轮累积实测 = {rhc_b_cum}"
          f"（差 {abs(rhc_b_cum - ref) if rhc_b_cum is not None else float('nan'):.3e}）")
    print("[E2]   该历史值未绑定测量条件（config/horizons/n_samples/eval 集），"
          "不参与 pass 判据；如需长期回归基线，应在 config 里连同条件一起冻结。")

    # ---------- E3 AoI 记账 ----------
    # ★ 用 Pendulum（每集 200 步）而不是 CartPole：age 的稳态需要足够长的序列，
    #   CartPole 随机策略平均 ~22 步就 terminate，复位太频繁会放大统计噪声。
    # ★ 复位语义：env.reset() 返回初始观测 = 一次成功接收 ⇒ age 归 0（与 StateTracker 一致）。
    e3_env_id = cfg["env"]["id"]
    e3 = []
    for p, n_steps in ((0.0, 3000), (0.3, 6000), (0.5, 6000), (0.9, 20000)):
        # ★ 用一条**不被 TimeLimit 打断**的长序列（max_episode_steps 拉到 1e6）。
        #   第一版沿用了 200 步的自然边界，每集开头 age 归 0，导致 p=0.5/0.9 的
        #   实测 E[age] 系统性偏低 5%（6000 样本的标准误只有 0.018，装不下这个偏差）。
        #   记账正误应该用纯序列来验，不该掺进 episode 边界的语义。
        env = make_env_with_channel(e3_env_id, seed=seed, loss_prob=p, delay=0,
                                    max_episode_steps=10 ** 6)
        rng = np.random.default_rng(seed)
        env.reset(seed=seed)
        ages, delivered = [], []
        age = 0
        ep = 0
        for _ in range(n_steps):
            res = env.step(env.sample_action(rng))
            if res.delivered:
                age = 0
            else:
                age += 1
            ages.append(age)
            delivered.append(res.delivered)
            if res.done:
                ep += 1
                env.reset(seed=seed + 1000 + ep)
                age = 0                      # 复位 = 重新获得观测
        env.close()
        meas_mean = float(np.mean(ages))
        meas_rate = float(np.mean(delivered))
        theo_mean = float(geometric_age_stats(p)[0]) if p > 0 else 0.0
        rel = abs(meas_mean - theo_mean) / max(theo_mean, 1e-9)
        e3.append({"p": p, "measured_mean_age": meas_mean, "theory_mean_age": theo_mean,
                   "rel_err": float(rel), "measured_delivery_rate": meas_rate,
                   "n_steps": n_steps, "n_episodes": ep + 1})
        print(f"[E3] p={p:<5} 实测 E[age]={meas_mean:7.4f}  理论={theo_mean:7.4f}  "
              f"相对误差={rel:6.2%}  送达率={meas_rate:.4f}  (n_steps={n_steps}, "
              f"{ep + 1} 集)")

    return {
        "E1_dynamics_bit_exact": bool(e1_bit), "E1_max_abs_diff": e1_max,
        "E2_data_bit_exact": bool(data_bit),
        "E2_open_h_bare": rho_b, "E2_open_h_channel": rho_c,
        "E2_closed_h_bare": rhc_b, "E2_closed_h_channel": rhc_c,
        "E2_bit_exact": bool(e2_open and e2_closed),
        "E2_reference_closed_h": ref,
        "E2_reference_closed_h_cumulative_actual": rhc_b_cum,
        "E2_reference_matches": bool(e2_ref_same),
        "E2_reference_in_pass_criteria": False,
        "E2_calibration_note": "H* 为逐点口径；reference_closed_h 为历史累积口径值，"
                               "且未绑定测量条件（config/horizons/n_samples/eval 集），"
                               "故仅作信息项，不参与 all_pass。",
        "E3": e3,
        # ★ pass 判据只包含"等价性"本身：E1 动力学逐位、E2 数据逐位 + 裸/信道 H* 逐位一致。
        "all_pass": bool(e1_bit and data_bit and e2_open and e2_closed),
    }


# ===================================================== X25 · 调度扫描
def sweep_schedules(model, eval_eps, cfg, device, var_g, quick=False):
    ch = cfg["channel"]
    sd = int(ch["tracking_seed"])
    periods = ch["periodic"] if not quick else ch["periodic"][:5]
    probs = ch["loss_prob"] if not quick else ch["loss_prob"][:3]
    out = []
    for T in periods:
        r = run_tracking(model, eval_eps, device, periodic_schedule(T), seed=sd,
                         delay=int(ch.get("delay", 0)), label=f"P(T={T})", denom=var_g)
        out.append(r)
        print(f"[X25] {r.label:<14} N_tx={r.n_tx:6d}  tx_rate={r.tx_rate:.4f}  "
              f"NMSE={r.nmse:.6f}  E[age]={r.age_tx_mean:8.3f}  max_age={r.age_tx_max}")
    for p in probs:
        r = run_tracking(model, eval_eps, device, lossy_schedule(p), seed=sd,
                         delay=int(ch.get("delay", 0)), label=f"R(p={p})", denom=var_g)
        out.append(r)
        print(f"[X25] {r.label:<14} N_tx={r.n_tx:6d}  tx_rate={r.tx_rate:.4f}  "
              f"NMSE={r.nmse:.6f}  E[age]={r.age_tx_mean:8.3f}  max_age={r.age_tx_max}")
    return out


def find_knee(points, tiny=1e-12):
    """在 (平均传输间隔, NMSE) 曲线上找**边际收益拐点**（Kneedle 定义，不依赖阈值）。

    ★ 为什么不用"NMSE ≤ 1.05×最小值"这类阈值定义（第一版就是那样，冒烟时失效了）：
      P(T=1)（每步都传）的 NMSE 恒为 **0** ⇒ `1.05 × 0 = 0`，判据整条失效。
      角度法只依赖曲线形状，换数据不换判据。

    做法：把 (log 间隔, log NMSE) 归一化到单位方框，取离「首尾连线」垂距最大的点。
    另附两个可读参考量（相对**最密传输点**的 NMSE 倍数），用于人工核对：
      interval_2x / interval_10x —— NMSE 首次达到参考值 2 倍 / 10 倍的间隔
    """
    pts = sorted([(float(i), float(v)) for i, v in points], key=lambda t: t[0])
    pts = [(i, v) for i, v in pts if v > 0.0 and i > 0.0]   # 排除退化点（NMSE=0 / 间隔=0）
    if len(pts) < 3:
        return {"note": f"有效点不足（已排除 NMSE=0 的点，剩 {len(pts)}）"}
    xs = np.log(np.asarray([p[0] for p in pts], dtype=float))
    ys = np.log(np.maximum(np.asarray([p[1] for p in pts], dtype=float), tiny))
    xn = (xs - xs.min()) / max(xs.max() - xs.min(), tiny)
    yn = (ys - ys.min()) / max(ys.max() - ys.min(), tiny)
    dx, dy = xn[-1] - xn[0], yn[-1] - yn[0]
    norm = float(np.hypot(dx, dy)) or 1.0
    dist = np.abs(dy * (xn - xn[0]) - dx * (yn - yn[0])) / norm
    k = int(np.argmax(dist))
    ref = float(pts[0][1])                                   # 最密传输点的 NMSE
    return {
        "knee_interval": float(pts[k][0]),
        "knee_nmse": float(pts[k][1]),
        "knee_distance": float(dist[k]),
        "ref_nmse_densest": ref,
        "ref_interval_densest": float(pts[0][0]),
        "interval_2x": next((float(i) for i, v in pts if v >= 2.0 * ref), float("nan")),
        "interval_10x": next((float(i) for i, v in pts if v >= 10.0 * ref), float("nan")),
        "n_points_used": len(pts),
    }


# ===================================================== X26 · 解析对账
def analytic_reconcile(curve_closed_global, results, cfg, h_max=1000):
    """用离线闭环 NMSE(h) 曲线解析预测在线跟踪误差，与仿真比。

    ★ 口径必须与在线一致（2026-09-20 修正，这是 X26 那 56% 缺口的根因）：
      在线 `run_tracking` 累计的是**每一步自身的**平方误差 ⇒ 是**逐点**口径。
      离线曲线若用**累积**口径（前 h-1 步被摊进均值），就会系统性低于在线值，
      于是解析式"低估"仿真 —— 但这个偏差来自口径不匹配，不是解析模型错了。
      因此这里优先用 `nmse_per_step_global`；缺该键时退回累积并明确打印告警。
    """
    key = "nmse_per_step_global" if "nmse_per_step_global" in curve_closed_global \
        else "nmse_global"
    tag = "逐点 (per-step)" if key == "nmse_per_step_global" else "★ 累积（与在线不可比！）"
    print(f"[X26] 解析所用离线曲线口径 = {key}  →  {tag}")
    # ★ 对齐：逐点数组是**稠密**的（长度 = max(horizons)），必须按 horizons 抽取后再配对，
    #   否则 NmseCurve 会静默截断（2026-09-20 踩过，见 NmseCurve 的 docstring）。
    _hs = list(curve_closed_global["horizons"])
    _arr = np.asarray(curve_closed_global[key], dtype=float)
    if _arr.size == len(_hs):
        _ys = [float(v) for v in _arr]
    else:
        assert _arr.size >= max(_hs), \
            f"逐点数组长度 {_arr.size} 不足以覆盖 max(horizons)={max(_hs)}"
        _ys = [float(_arr[h - 1]) for h in _hs]
    curve = NmseCurve(_hs, _ys)
    rows = []
    for r in results:
        if r.label.startswith("P(T="):
            period = int(r.label[4:-1])
            pmf = uniform_age_pmf(period)
            theory_mean, theory_var = uniform_age_stats(period)
            family = "P"
        else:
            p = float(r.label[4:-1])
            pmf = geometric_age_pmf(p, h_max)
            theory_mean, theory_var = geometric_age_stats(p)
            family = "R"
        e_analytic, nmse_at_mean = jensen_gap(curve, pmf)
        # ★ 相对误差在 sim→0 时病态（P(T=1) 的 NMSE 恒为 0，会得到 ~1e13%）。
        #   这不是结论，是分母问题 —— 置为 NaN 并在统计时剔除，绝不放进结论。
        rel_gap = ((e_analytic - r.nmse) / r.nmse) if r.nmse > 1e-9 else float("nan")
        rows.append({
            "label": r.label, "family": family,
            "analytic_curve_calibration": key,
            "tx_rate": r.tx_rate, "mean_interval": 1.0 / max(r.tx_rate, 1e-12),
            "sim_nmse": r.nmse,
            "analytic_E_nmse": e_analytic,
            "analytic_nmse_at_mean_age": nmse_at_mean,
            "jensen_gap": e_analytic - nmse_at_mean,
            "abs_gap": e_analytic - r.nmse,
            "rel_analytic_vs_sim": rel_gap,
            "measured_mean_age": r.age_tx_mean,
            "theory_mean_age": theory_mean,
            "theory_var_age": theory_var,
            "measured_max_age": r.age_tx_max,
        })
    return curve, rows


def jensen_at_matched_age(curve, rows, n_grid=40):
    """★ Jensen 命题的直接检验：把两个调度族插值到**相同平均间隔**上比较 NMSE。

    命题：平均 AoI 相同、但 age **方差**更大的那一族（几何 / 丢包），
          实际误差更高（当 NMSE(h) 在相应区间为凸时）。
    """
    out = []
    fams = {}
    for r in rows:
        fams.setdefault(r["family"], []).append((r["mean_interval"], r["sim_nmse"]))
    for f in fams:
        fams[f] = sorted(fams[f])
    if "P" not in fams or "R" not in fams:
        return out
    xp = np.asarray([a for a, _ in fams["P"]]); yp = np.asarray([b for _, b in fams["P"]])
    xr = np.asarray([a for a, _ in fams["R"]]); yr = np.asarray([b for _, b in fams["R"]])
    lo = max(xp.min(), xr.min()); hi = min(xp.max(), xr.max())
    if hi <= lo:
        return out
    grid = np.exp(np.linspace(np.log(lo), np.log(hi), n_grid))
    ip = np.interp(grid, xp, yp)
    ir = np.interp(grid, xr, yr)
    for g, a, b in zip(grid, ip, ir):
        out.append({"mean_interval": float(g), "nmse_periodic": float(a),
                    "nmse_lossy": float(b), "diff_lossy_minus_periodic": float(b - a),
                    "ratio": float(b / max(a, 1e-12))})
    return out


# ===================================================== 出图
def plot_all(out_path, cfg, var_g, curve_open, curve_closed, results, rows,
             jensen_rows, eq, rh_ref):
    fig, axes = plt.subplots(2, 3, figsize=(17.5, 9.6))
    hs = curve_closed["horizons"]

    # ① 离线曲线（全局归一）
    #   ★ 主线画**逐点**（与在线跟踪同口径、X26 解析实际使用的那条）；
    #     累积口径画淡色虚线 —— X2 的历史数字来自它，差别一眼可见。
    ax = axes[0, 0]
    ax.plot(curve_open["horizons"], curve_open["nmse_global"], ":", color=PALETTE["blue"],
            alpha=0.35, label="open · cumulative (legacy)")
    ax.plot(hs, curve_closed["nmse_global"], ":", color=PALETTE["red"], alpha=0.35,
            label="closed · cumulative (legacy)")
    _op = np.asarray(curve_open["nmse_per_step_global"], dtype=float)
    _cp = np.asarray(curve_closed["nmse_per_step_global"], dtype=float)
    ax.plot(curve_open["horizons"], _op[[h - 1 for h in curve_open["horizons"]]], "o-",
            color=PALETTE["blue"], label="open-loop · per-step ★")
    ax.plot(hs, _cp[[h - 1 for h in hs]], "s--", color=PALETTE["red"],
            label="closed-loop · per-step ★ (used by X26)")
    thr = float(cfg["eval"]["err_threshold"])
    ax.axhline(thr, color=PALETTE["grey"], ls=":", lw=1.2, label=f"threshold={thr}")
    for h, c, nm in ((rh_ref.get("open"), PALETTE["blue"], "open"),
                     (rh_ref.get("closed"), PALETTE["red"], "closed")):
        if h is not None:
            ax.axvline(h, color=c, ls=":", alpha=0.6)
            ax.annotate(f"H*({nm})={h:.1f}", xy=(h, thr), xytext=(h + 3, thr * 30),
                        color=c, fontsize=8)
    ax.set_yscale("log"); ax.set_xscale("log")
    ax.set_xlabel("Rollout horizon h (steps)")
    ax.set_ylabel("NMSE (global pooled var)")
    ax.set_title("① Offline NMSE(h) — per-step vs cumulative")
    ax.legend(fontsize=7)

    # ② 传输率–NMSE 帕累托
    ax = axes[0, 1]
    for fam, col, mk, nm in (("P", PALETTE["blue"], "o", "periodic (deterministic age)"),
                             ("R", PALETTE["red"], "s", "lossy channel (geometric age)")):
        pts = sorted([(r["mean_interval"], r["sim_nmse"]) for r in rows if r["family"] == fam])
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], mk + "-", color=col, label=nm)
    for h, c in ((rh_ref.get("closed"), PALETTE["red"]), (rh_ref.get("open"), PALETTE["blue"])):
        if h is not None:
            ax.axvline(h, color=c, ls=":", alpha=0.5)
            ax.annotate(f"H*={h:.0f}", xy=(h, ax.get_ylim()[1]), xytext=(h * 1.05, 0.5),
                        color=c, fontsize=8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Mean transmission interval  (= 1 / tx_rate)  [steps]")
    ax.set_ylabel("Tracking NMSE")
    ax.set_title("② X25 Pareto: cost vs task error")
    ax.legend(fontsize=7.5)

    # ③ 解析 vs 仿真
    ax = axes[0, 2]
    for fam, col, mk, nm in (("P", PALETTE["blue"], "o", "periodic"),
                             ("R", PALETTE["red"], "s", "lossy")):
        d = [r for r in rows if r["family"] == fam]
        if d:
            ax.scatter([r["sim_nmse"] for r in d], [r["analytic_E_nmse"] for r in d],
                       color=col, marker=mk, s=26, label=nm)
    lims = [1e-5, 2.0]
    ax.plot(lims, lims, color=PALETTE["grey"], ls="--", lw=1, label="y = x (perfect)")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Simulated tracking NMSE")
    ax.set_ylabel("Analytic  E[NMSE(age)]")
    ax.set_title("③ X26 Analytic vs simulation")
    ax.legend(fontsize=7.5)

    # ④ Jensen gap
    ax = axes[1, 0]
    for fam, col, mk, nm in (("P", PALETTE["blue"], "o", "periodic"),
                             ("R", PALETTE["red"], "s", "lossy")):
        d = sorted([r for r in rows if r["family"] == fam], key=lambda r: r["mean_interval"])
        if d:
            ax.plot([r["mean_interval"] for r in d],
                    [r["jensen_gap"] / max(r["sim_nmse"], 1e-12) for r in d],
                    mk + "-", color=col, label=nm)
    ax.axhline(0.0, color=PALETTE["grey"], ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("Mean transmission interval [steps]")
    ax.set_ylabel("(E[NMSE(age)] − NMSE(E[age])) / NMSE_sim")
    ax.set_title("④ Jensen bias of 'average AoI' reasoning")
    ax.legend(fontsize=7.5)

    # ⑤ 曲率（二阶差分）—— 判定凸/凹段
    ax = axes[1, 1]
    hh, d2 = NmseCurve(hs, curve_closed["nmse_global"]).second_difference()
    ax.plot(hh, d2, color=PALETTE["green"], lw=1.8)
    ax.axhline(0.0, color=PALETTE["grey"], ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("Horizon h (steps)")
    ax.set_ylabel("Second difference  Δ²NMSE")
    frac = NmseCurve(hs, curve_closed["nmse_global"]).convex_fraction()
    ax.set_title(f"⑤ Curvature: convex (>0) fraction = {frac:.2%}")

    # ⑥ 实测 NMSE(age) vs 离线曲线
    ax = axes[1, 2]
    ax.plot(hs, curve_closed["nmse_global"], "--", color=PALETTE["grey"], lw=2.0,
            label="offline closed-loop NMSE(h)", alpha=0.9)
    for fam, col, mk, nm in (("P", PALETTE["blue"], "o", "periodic"),
                             ("R", PALETTE["red"], "s", "lossy")):
        first = True
        for r in [x for x in rows if x["family"] == fam]:
            byage = [(a, v) for a, v in sorted(r["empirical_nmse_by_age"].items()) if a > 0]
            if not byage:
                continue
            ax.plot([a for a, _ in byage], [v for _, v in byage], mk, color=col,
                    ms=3.5, alpha=0.6, label=nm if first else None)
            first = False
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Age h (steps)")
    ax.set_ylabel("NMSE")
    ax.set_title("⑥ Online NMSE(age) vs offline curve")
    ax.legend(fontsize=7.5)

    fig.suptitle(
        f"wmlab · X24/X25/X26 channel access · env={cfg['env']['id']} · seed={cfg['seed']} · "
        f"eval={cfg['channel']['eval_episodes']}×{cfg['channel']['eval_steps']} steps · "
        f"var_global={var_g:.5f} · equivalence={'PASS' if eq.get('all_pass') else 'n/a'}",
        fontsize=10)
    fig.tight_layout()
    return save_fig(fig, out_path)


# ===================================================== main
def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.episodes is not None:
        cfg["data"]["n_episodes"] = args.episodes
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.quick:
        cfg["train"]["epochs"] = min(int(cfg["train"]["epochs"]), 3)
        cfg["data"]["n_episodes"] = min(int(cfg["data"]["n_episodes"]), 20)
        cfg["channel"]["eval_episodes"] = 3
        cfg["channel"]["eval_steps"] = 120
        cfg["eval"]["n_samples"] = 64
        args.skip_equivalence = True

    seed = int(cfg["seed"])
    set_seed(seed)
    apply_style()
    device = get_device(args.device or cfg.get("device"))
    out = output_dir(cfg)
    t_start = time.time()

    print(f"[04] env={cfg['env']['id']}  seed={seed}  device={describe_device(device)}")
    print(f"[04] 曲线口径声明：在线跟踪用 predict_next（encode→transition→decode）"
          f"= 观测空间闭环 ⇒ X26 对账用闭环 NMSE(h)")

    # ---------- X24 等价性 ----------
    eq = {"all_pass": None}
    if not args.skip_equivalence:
        eq = check_equivalence(cfg, device, seed)

    # ---------- 训练主模型（与 X2 同配置） ----------
    env = make_env(cfg["env"]["id"], seed=seed)
    act_dim_env, disc_env = int(env.act_dim), bool(env.is_discrete)
    train_episodes = collect_random_episodes(env, n_episodes=int(cfg["data"]["n_episodes"]),
                                             seed=seed, max_steps=cfg["env"].get("max_steps"))
    env.close()
    lens = np.asarray([e["length"] for e in train_episodes])
    print(f"[04] 训练集：{len(train_episodes)} 条 episode，共 {lens.sum()} 步，"
          f"长度 mean={lens.mean():.1f} max={lens.max()}  "
          f"(act_dim={act_dim_env}, discrete={disc_env})")

    m0, hist, rh = train_new_model(
        cfg, cfg["env"]["id"], train_episodes, device, seed,
        act_dim=act_dim_env, discrete_act=disc_env, label="X2-同配置")
    print(f"[04] ★ 与 X2 对照（逐点口径）：开环 H*={rh['open_pointwise']}  "
          f"闭环 H*={rh['closed_pointwise']}")
    print(f"[04]   （历史累积口径：开环 {rh['open_cumulative']}  闭环 {rh['closed_cumulative']}）")
    print(f"[04]    最终 val loss = {hist['val_total'][-1]:.6f}  "
          f"params={count_params(m0):,}")

    # ---------- 长评测集（held-out） ----------
    ch = cfg["channel"]
    ev_seed = int(ch["eval_seed"])
    ev_env = make_env(cfg["env"]["id"], seed=ev_seed,
                      max_episode_steps=int(ch["eval_steps"]))
    eval_eps = collect_random_episodes(ev_env, n_episodes=int(ch["eval_episodes"]),
                                       seed=ev_seed, max_steps=None)
    ev_env.close()
    ev_lens = np.asarray([e["length"] for e in eval_eps])
    var_g = pooled_var(eval_eps)
    print(f"[04] 评测集：{len(eval_eps)} 条 × {ev_lens.mean():.0f} 步"
          f"（seed={ev_seed}，与训练集不重叠）  var_global={var_g:.6f}")

    # ---------- 离线曲线（全局归一） ----------
    hs = clamp_horizons(list(cfg["eval"]["horizons"]), eval_eps, label="eval set")
    cfg["eval"]["horizons"] = hs
    co = multi_step_error_curve(m0, eval_eps, hs, device,
                                n_samples=int(cfg["eval"]["n_samples"]), seed=seed)
    cc = closed_loop_error_curve(m0, eval_eps, hs, device,
                                     n_samples=max(64, int(cfg["eval"]["n_samples"]) // 2),
                                     seed=seed)
    for c in (co, cc):
        c["nmse_global"] = [m / var_g for m in c["mse"]]
        # ★ 逐点口径的 uniform 分母版本 —— X26 解析对账必须用它（与在线 tracking 同分母同口径）
        c["nmse_per_step_global"] = [m / var_g for m in c["mse_per_step"]]

    def _pt_of(curve, key):
        arr = np.asarray(curve[key], dtype=float)
        return [float(arr[h - 1]) for h in curve["horizons"]]

    thr_eval = float(cfg["eval"]["err_threshold"])
    h_open_pt = reliable_horizon(co["horizons"], _pt_of(co, "nmse_per_step_global"),
                                 thr_eval, "rel")
    h_pt = reliable_horizon(cc["horizons"], _pt_of(cc, "nmse_per_step_global"), thr_eval, "rel")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        h_cum = reliable_horizon(cc["horizons"], cc["nmse_global"], thr_eval, "rel",
                                 calibration="cumulative")
    print(f"[04] 评测集闭环 H*（逐点 · global 分母）= {h_pt}   [累积口径对照 = {h_cum}]")
    print(f"[04] 评测集开环 H*（逐点 · global 分母）= {h_open_pt}")

    # ---------- X25 调度扫描 ----------
    print("[04] ---- X25 调度扫描 ----")
    results = sweep_schedules(m0, eval_eps, cfg, device, var_g, quick=args.quick)

    # ---------- X26 解析对账 ----------
    print("[04] ---- X26 解析 vs 仿真 ----")
    curve, rows = analytic_reconcile(cc, results, cfg)
    by_label = {r.label: r for r in results}
    for r in rows:
        src = by_label[r["label"]].nmse_by_age
        r["empirical_nmse_by_age"] = {
            int(h): (v[0] / max(v[1], 1)) / var_g for h, v in src.items()
        }
    rel = np.asarray([abs(r["rel_analytic_vs_sim"]) for r in rows])
    finite = rel[np.isfinite(rel)]
    print(f"[04] 解析 vs 仿真 相对偏差（剔除 NMSE≈0 的退化点，有效 n={finite.size}/{len(rows)}）："
          f"median={np.median(finite):.2%}  p90={np.percentile(finite, 90):.2%}  "
          f"max={finite.max():.2%}")
    for r in rows:
        rs = (f"{r['rel_analytic_vs_sim']:+.2%}"
              if np.isfinite(r["rel_analytic_vs_sim"]) else "   n/a")
        print(f"[04]   {r['label']:<14} sim={r['sim_nmse']:.6f}  "
              f"analytic={r['analytic_E_nmse']:.6f}  rel={rs}  "
              f"jensen={r['jensen_gap']:+.6f}")

    jr = jensen_at_matched_age(curve, rows)
    if jr:
        mid = jr[len(jr) // 2]
        pos = sum(1 for x in jr if x["diff_lossy_minus_periodic"] > 0)
        print(f"[04] Jensen@matched-age: {pos}/{len(jr)} 个点上『丢包族误差更高』；"
              f"中位点 interval={mid['mean_interval']:.2f} ratio={mid['ratio']:.3f}")

    # ---------- 决策量：H* 落在哪 ----------
    #   ★ 用**评测集上、逐点口径**的闭环 H*（h_pt）—— 它才是与 X25 曲线同一套口径的量。
    #     训练集短 val 集上的 rh 只用于与 X2 对齐，不参与决策。
    p_points = [(r["mean_interval"], r["sim_nmse"]) for r in rows if r["family"] == "P"]
    knee = find_knee(p_points)
    print(f"[04] 周期族 knee（Kneedle 角度法）: 间隔={knee.get('knee_interval')}  "
          f"NMSE={knee.get('knee_nmse')}  |  参考点=最密传输 "
          f"(间隔 {knee.get('ref_interval_densest')}, NMSE {knee.get('ref_nmse_densest')})")
    print(f"[04]    NMSE 达参考 2 倍的间隔 = {knee.get('interval_2x')}；"
          f"10 倍 = {knee.get('interval_10x')}")
    h_ref = h_pt if h_pt is not None else rh["closed_pointwise"]
    p_star = float("nan")
    if h_ref is not None and "knee_interval" in knee:
        kr = float(knee["knee_interval"])
        side = ("右侧（更省通信，但 NMSE 已明显高于拐点处）" if h_ref > kr
                else "左侧（比拐点更密：误差更低，代价是更多传输 —— 即 H* 是保守的）")
        print(f"[04] H*(闭环)={h_ref:.2f} vs knee={kr:.1f} → H* 落在拐点{side}")
        p_star = loss_prob_for_tail_risk(h_ref, 0.01)
        print(f"[04] 容忍丢包率 p*(H={h_ref:.1f}, δ=1%) = {p_star:.4f} —— "
              f"数值越大说明通信越不构成瓶颈（计划书 §3 的 p*≈0.93 由此而来）")

    # ---------- 出图 + 落盘 ----------
    rh_ref = {"open": h_open_pt, "closed": h_pt}
    png = plot_all(out / f"{args.tag}.png", cfg, var_g, co, cc, results, rows, jr, eq, rh_ref)

    csv_path = out / f"{args.tag}_schedules.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["label", "family", "analytic_curve_calibration", "n_tx", "n_steps",
                    "tx_rate", "mean_interval",
                    "sim_nmse", "analytic_E_nmse", "analytic_nmse_at_mean_age",
                    "jensen_gap", "abs_gap", "rel_analytic_vs_sim", "measured_mean_age",
                    "theory_mean_age", "theory_var_age", "measured_max_age"])
        for r in rows:
            w.writerow([r["label"], r["family"], r["analytic_curve_calibration"],
                        next(x.n_tx for x in results if x.label == r["label"]),
                        next(x.n_steps for x in results if x.label == r["label"]),
                        f"{r['tx_rate']:.8f}", f"{r['mean_interval']:.6f}",
                        f"{r['sim_nmse']:.8f}", f"{r['analytic_E_nmse']:.8f}",
                        f"{r['analytic_nmse_at_mean_age']:.8f}", f"{r['jensen_gap']:.8f}",
                        f"{r['abs_gap']:.8f}", f"{r['rel_analytic_vs_sim']:.6f}",
                        f"{r['measured_mean_age']:.4f}",
                        f"{r['theory_mean_age']:.4f}", f"{r['theory_var_age']:.4f}",
                        r["measured_max_age"]])

    jcsv = out / f"{args.tag}_jensen_matched.csv"
    with open(jcsv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["mean_interval", "nmse_periodic", "nmse_lossy",
                    "diff_lossy_minus_periodic", "ratio"])
        for x in jr:
            w.writerow([f"{x['mean_interval']:.6f}", f"{x['nmse_periodic']:.8f}",
                        f"{x['nmse_lossy']:.8f}", f"{x['diff_lossy_minus_periodic']:.8f}",
                        f"{x['ratio']:.6f}"])

    summary = {
        "config": cfg, "device": str(device), "n_params": count_params(m0),
        "var_global": var_g,
        "eval_set": {"n_episodes": len(eval_eps), "steps_each": int(ev_lens.mean()),
                     "seed": ev_seed},
        "x2_reference": [rh["open_pointwise"], rh["closed_pointwise"],
                         "（同一配置重训；对应 2026-09-18 X2，但 X2 当时用的是训练集切分出的"
                         "短 val 集 + 累积口径，故数值不同不是复现失败，见 x2_reference_detail）"],
        "x2_reference_detail": {
            "train_val_set": {"open_pointwise": rh["open_pointwise"],
                              "closed_pointwise": rh["closed_pointwise"],
                              "open_cumulative": rh["open_cumulative"],
                              "closed_cumulative": rh["closed_cumulative"],
                              "denominator": "切片目标方差（与 X2 历史口径同源，故可与 64.8/52.0 对照）",
                              "note": "训练集 10% 切分出的短 val 集（与 X2 的评测集一致）"},
            "eval_set": {"open_pointwise": h_open_pt, "closed_pointwise": h_pt,
                         "closed_cumulative": h_cum,
                         "denominator": "全局 pooled 方差（与在线 tracking / X25-X26 同源）",
                         "note": "新采的 20×1000 held-out 长集"},
            "why_they_differ": "同一模型下，H* 随『评测集』『口径』『归一化分母』三者变化。"
                              "跨结论引用 H* 必须同时核对这三个字段。",
        },
        "nmse_calibration": {
            "h_star": "per_step (pointwise) —— 『首次超阈』的定义口径",
            "cumulative_reported_for": "复现历史结论 / 与 X2 的旧数字对照",
            "overestimate_measured": "Pendulum 实测：开环 +52.0%、闭环 +51.1%",
        },
        "h_star_eval_set": {"open_pointwise": h_open_pt, "closed_pointwise": h_pt,
                            "closed_cumulative": h_cum, "eval_seed": ev_seed},
        "equivalence": eq,
        "offline_curve_open_global": co,
        "offline_curve_closed_global": cc,
        # ★ 保留 empirical_nmse_by_age：X26 的 gap 诊断（scripts/05）要用它
        #   做「离线曲线 vs 在线实测」的逐 h 对比，缺了它下游只能重跑实验。
        "schedules": rows,
        "jensen_matched_age": jr,
        "knee": knee,
        "loss_prob_star_for_H": p_star,
        "elapsed_sec": time.time() - t_start,
    }
    # ---------- 保存模型权重（★ 2026-09-20 加）----------
    # 起因：X26 诊断（scripts/05）发现"离线曲线 vs 在线实测"在同一个 h 上差 2–3 倍，
    # 要定位就必须能在**不重训**的前提下复算任意口径的曲线。只存 json 数字做不到这件事。
    ckpt = out / f"{args.tag}_model.pt"
    torch.save({
        "state_dict": m0.state_dict(),
        "obs_dim": int(eval_eps[0]["obs"].shape[1]),
        "act_dim": act_dim_env,
        "discrete_act": disc_env,
        "latent_dim": int(cfg["model"]["latent_dim"]),
        "hidden": int(cfg["model"]["hidden"]),
        "var_global": var_g,
        "eval_seed": ev_seed,
        "eval_episodes": len(eval_eps),
        "eval_steps": int(ev_lens.mean()),
        "train_seed": seed,
        "train_episodes": int(cfg["data"]["n_episodes"]),
        "env_id": cfg["env"]["id"],
    }, ckpt)

    js = out / f"{args.tag}.json"
    js.write_text(json.dumps(jsonable(summary), ensure_ascii=False, indent=2, default=str),
                  encoding="utf-8")

    print(f"[04] 图   -> {png}")
    print(f"[04] 表   -> {csv_path}")
    print(f"[04] 表   -> {jcsv}")
    print(f"[04] 权重 -> {ckpt}")
    print(f"[04] json -> {js}")
    print(f"[04] 用时 {summary['elapsed_sec']:.1f}s")
    print("[04] OK")


if __name__ == "__main__":
    main()
