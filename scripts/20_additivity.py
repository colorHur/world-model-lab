# -*- coding: utf-8 -*-
"""★ X33「丢包 × 量化：跟踪误差能不能被**分解**、进而被**预报**」

================================================================= 为什么必须做这个实验
X31/X31-b 解决了"丢包结构"那一半：闭式 `E[NMSE] = Σ_h P_age(h)·f(h)` 已验证到 % 级。
X32 解决了"物理层"那一半：给定 SNR 与量化比特 b，速率—可靠权衡存在 U 形最优 b\*。

但两个各自gitlab的结论**加不起来**就是两张皮 —— 本实验要问的正是"怎么加"：

    N_full(信道, 量化)  ?=?  N_drop(同信道, 无量化)  +  N_quant(完美信道, 有量化)

如果等号成立 ⇒ 任意新配置可以用**两次便宜的测量**预报，联合设计（X34）才有可能。

================================================================= ★★ 推导：为什么预感等号**不**成立
一次丢包后，估计值是"上次收到的那个**被量化过的**状态"再 rollout h 步：

    x̂_t  = F^h( x_{t−h} + q )          q = 量化误差
    x_t  = F^h( x_{t−h} ) + 过程噪声

    ⇒ 误差 ≈ [F^h(x+q) − F^h(x)] − [x_t − F^h(x)]
           = **被传播了 h 步的量化误差** + 模型 rollout 误差

    静态量化地板 N_quant 只在 h≈0 时成立；h 越大，q 越有机会被 F 的雅可比放大/扭曲。

        ⇒ 定义 **放大倍数 A = (N_full − N_drop) / N_quant**

预写判据（跑前定死）：
    V1 可加      —— max |交互项 I| / N_full < 10%
    V2 弱可加    —— < 30%
    V3 不可加    —— ≥ 30%（⇒ 必须报 A，而不是假装可加）
    H1 A > 1 普遍成立，且与 E[age] **正相关**（Spearman ρ > 0.5）
    H2 persistence 与 model 的 A **不同** —— persistence 是零阶保持，
       q 不会被"传播"，只会在原地累积 ⇒ 预写 **A_persist 更接近 1 / 更小**
    ★ 注意：我不能排除反方向。若实测相反，必须照报并解释，不许改判据。

================================================================= ★ 预报能力（X34 的前提）
把所有网格点按 E[age] 排序，用 **留一插值**（LOO-np.interp）从其余点推出 Â，
再预报 N̂_full = N_drop + Â·N_quant，报出相对误差的中位与 90 分位。
⇒ 这个数字就是"能不能用便宜测量预报没跑过的配置"的直接回答。

================================================================= 自检（不过就抛 —— 规范 R14）
S1  完美信道下 model 与 persistence 必须**逐位一致**（每步都到 ⇒ 从不调用预测）
S2  N_full 不得显著低于 N_drop（>5% ⇒ 有 bug 或存在"噪声反而更好"的反常）
S3  N_quant 与静态量化的 ground truth `qmse/var_g` 同量级（相差 >2×  ⇒ 口径错了）
S4  全部有限（R14）
S5  R12：量化确实改变结果（N_full ≠ N_drop，PER 未饱和处）
S6  N_quant 对 b 单调减
S7  同一 (SNR,b,L) 下两个 head 的 E[age] **必须相同** —— 信道的随机数消耗
    与估计器无关。这条若破，"同龄对照"就失去了意义（X26 的 R13 教训）
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
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

from wmlab.control import PDRelativeController, collect_controlled_episodes
from wmlab.data import split_episodes, transitions_from_episodes
from wmlab.envs import make_env
from wmlab.eval import reliable_horizon
from wmlab.eval.physical import (PhysicalChannel, UniformQuantizer, fit_quantizer_range,
                                 overload_fraction, per_from_ber, qam_approximation_valid,
                                 qam_bit_error_rate, snr_db_to_linear)
from wmlab.eval.tracking import GilbertElliottChannel, lossy_schedule, run_tracking
from wmlab.models import MLPWorldModel
from wmlab.rollout import closed_loop_error_curve
from wmlab.train import train_world_model
from wmlab.utils import count_params, get_device, load_config, output_dir, set_seed
from wmlab.utils.plot import PALETTE, apply_style, save_fig


def _load_mod19():
    """复用 scripts/19 里**已验证过**的 `PersistenceHead`（19 有 __main__ 保护）。

    ★ 为什么要复用而不是复制：persistence 的语义（零阶保持）必须与 X32 逐字一致，
      否则本实验与 X19 的对照不可比。
    """
    spec = importlib.util.spec_from_file_location(
        "_mod19", os.path.join(_HERE, "19_physical_layer.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_mod19 = _load_mod19()
PersistenceHead = _mod19.PersistenceHead


def _load_mod17():
    """复用 X31（`scripts/17`）里**已验证过**的 Gilbert 年龄分布闭式。

    ★ 必须复用： appropriation 公式 `P(age=k) = p̄·β·(1−β)^(k−1)` 已在 X31/X31-b 里
      被 KS/TV/游程长度三重检验过；口径必须逐字一致，本实验的结论才与 X31 可比。
    """
    spec = importlib.util.spec_from_file_location(
        "_mod17", os.path.join(_HERE, "17_burst_channel.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


full_age_pmf = _load_mod17().full_age_pmf


def parse_args():
    p = argparse.ArgumentParser(description="X33 丢包 × 量化：可分解性与预报")
    p.add_argument("--config", default="configs/uav_additivity.yaml")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--tag", default="20_additivity")
    p.add_argument("--snrs", default=None)
    p.add_argument("--bits", default=None)
    p.add_argument("--burst", default=None, help="逗号分隔突发长度；'none'=无记忆")
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


def per_at(g_db: float, b: int, d: int) -> float:
    """给定 (SNR, 每维比特数 b, 状态维数 d) 的包错误率。

    失效区（`qam_approximation_valid` 为假）按"链路不可用 PER≈1"处理 ——
    理由见 `wmlab/eval/physical.py`，第一版没这样处理时 PER 对 b 非单调。
    """
    g = snr_db_to_linear(g_db)
    if not qam_approximation_valid(g, 2.0 ** b):
        return 1.0 - 1e-6
    return per_from_ber(qam_bit_error_rate(g, 2.0 ** b), float(d * b))


def snr_for_target_per(b: int, d: int, target: float,
                       lo_db: float = -60.0, hi_db: float = 90.0) -> float | None:
    """二分反解：哪个 SNR 能让 (b, d) 这一档调制得到目标 PER。

    ★★ 为什么不直接扫 SNR 网格（第一版的错误，记在这里）
    第一版按 `snr_db_list × bits_list` 扫，结果 **20 个点里 19 个是 PER≈0**：
    4-QAM @25dB 的 BER 是 Q(√316) ≈ 1e-71，float64 下 PER 直接下溢成 **0.0**，
    ⇒ E[age]=0 ⇒ 没有丢包 ⇒ A 恒等于 1、交互项恒等于 0。
    这些点同时印证不了任何东西，还把 V/H 判据全拉向"可加"的假结论。

    ⇒ 真实的自变量是 **PER（它决定信息年龄结构）**，SNR 只是达到 PER 的手段。
       所以网格应当 **按目标 PER 反解 SNR**，且这几个 "PER 精确到具体值" 的
       工作点依然是物理上合法的 (SNR, b) 组合。
    """
    f_lo = per_at(lo_db, b, d) - target
    f_hi = per_at(hi_db, b, d) - target
    if f_lo < 0.0:
        return None          # 即使 SNR 低到 lo_db，PER 仍 > 目标（impossible to be cleaner）
    if f_hi > 0.0:
        return None          # 即使 SNR 高到 hi_db，PER 仍 < 目标
    a, bb = lo_db, hi_db
    for _ in range(200):
        m = 0.5 * (a + bb)
        if per_at(m, b, d) - target > 0.0:
            a = m
        else:
            bb = m
    return 0.5 * (a + bb)


def spearman(x, y):
    """秩相关（仓库不装 scipy，手写）。输入可为含 NaN 的数组 ⇒ 自动剔除。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 4:
        return float("nan"), int(x.size)

    def rank(v):
        order = np.argsort(v, kind="mergesort")
        r = np.empty(v.size, dtype=float)
        r[order] = np.arange(1, v.size + 1)
        # 并列取平均秩
        for u in np.unique(v):
            sel = v == u
            if sel.sum() > 1:
                r[sel] = r[sel].mean()
        return r

    rx, ry = rank(x), rank(y)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = float(np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))
    return (float((rx * ry).sum() / denom) if denom > 0 else float("nan")), int(x.size)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.snrs:
        cfg["physical"]["snr_db_list"] = [float(x) for x in args.snrs.split(",")]
    if args.bits:
        cfg["physical"]["bits_list"] = [int(x) for x in args.bits.split(",")]
    if args.burst:
        cfg["physical"]["burst_lens"] = [
            None if s.strip().lower() in ("none", "null", "") else float(s)
            for s in args.burst.split(",")]
    ph0 = cfg["physical"]
    if args.quick:
        cfg["train"]["epochs"] = min(int(cfg["train"]["epochs"]), 3)
        cfg["data"]["n_episodes"] = min(int(cfg["data"]["n_episodes"]), 20)
        ph0["per_targets"] = [0.05, 0.4]
        ph0["bits_list"] = [2, 6]
        ph0["burst_lens"] = [None, 8]
        ph0["n_track_episodes"] = 8
        cfg["eval"]["n_samples"] = 64

    seed = int(cfg["seed"])
    set_seed(seed)
    apply_style()
    device = get_device(args.device or cfg.get("device"))
    out = output_dir(cfg)
    t0 = time.time()

    ph = cfg["physical"]
    bits_list = [int(x) for x in ph["bits_list"]]
    bursts = list(ph["burst_lens"])
    per_targets = [float(x) for x in ph["per_targets"]]
    n_track = int(ph["n_track_episodes"])

    print("[20] " + "=" * 76)
    print("[20] ★ X33 丢包 × 量化：可分解性检验")
    print(f"[20]   目标 PER ∈ {per_targets}（★ 反解 SNR，而不是直接扫 SNR）"
          f"   b ∈ {bits_list}   L ∈ {bursts}")
    print("[20]   预写判据：V1 可加(|I|<10%) / V2 弱可加(<30%) / V3 不可加")
    print("[20]             H1 A>1 且与 E[age] 正相关   H2 A_persist ≠ A_model")

    # ---------- 1) 环境 ----------
    env = make_env(cfg["env"]["id"], seed=seed,
                   noise_std=float(cfg["env"]["noise_std"]),
                   max_steps=int(cfg["env"]["max_steps"]))
    # ---------- 2) 训练 ----------
    cc = cfg["controller"]
    ctrl = PDRelativeController(dt=env.dt, omega_n=float(cc["omega_n"]),
                                zeta=float(cc["zeta"]), kappa=float(env.kappa),
                                a_max=float(env.a_max))
    n_ep = int(cfg["data"]["n_episodes"])
    train_all = collect_controlled_episodes(env, ctrl, n_episodes=n_ep, seed=seed,
                                            max_steps=int(cfg["env"]["max_steps"]))
    env.close()
    train_eps, val_eps = split_episodes(train_all, float(cfg["train"]["val_ratio"]), seed)
    tr = tuple(torch.as_tensor(x) for x in transitions_from_episodes(train_eps))
    va = tuple(torch.as_tensor(x) for x in transitions_from_episodes(val_eps))
    obs_dim = int(train_all[0]["obs"].shape[1])
    model = MLPWorldModel(obs_dim=obs_dim, act_dim=int(env.act_dim),
                          latent_dim=int(cfg["model"]["latent_dim"]),
                          hidden=int(cfg["model"]["hidden"]),
                          discrete_act=bool(env.is_discrete)).to(device)
    hist = train_world_model(model, tr, va, cfg, device, verbose=False)
    print(f"[20] 训练完成：params={count_params(model):,} val={hist['val_total'][-1]:.6f} "
          f"({time.time() - t0:.0f}s)")

    # ---------- 3) 量化器 + 跟踪数据 ----------
    WARMUP = int(cfg["task"]["warmup_steps"])
    ev_seed = seed + int(cfg["data"]["eval_seed_offset"])
    ev_env = make_env(cfg["env"]["id"], seed=ev_seed,
                      noise_std=float(cfg["env"]["noise_std"]),
                      max_steps=int(cfg["env"]["max_steps"]))
    eval_eps = collect_controlled_episodes(ev_env, ctrl,
                                           n_episodes=max(30, n_ep // 3), seed=ev_seed,
                                           max_steps=int(cfg["env"]["max_steps"]))
    ev_env.close()
    lo, hi = fit_quantizer_range(eval_eps, margin=float(ph.get("range_margin", 0.05)))
    allobs = np.concatenate([e["obs"][WARMUP:] for e in eval_eps], axis=0).astype(np.float64)
    var_g = float(allobs.var())
    quantizers = {b: UniformQuantizer(lo, hi, b) for b in bits_list}

    track_seed = seed + int(cfg["data"].get("online_seed_offset", 2000))
    track_eps = eval_eps[:n_track] if n_track <= len(eval_eps) else eval_eps
    print(f"[20] 跟踪：{len(track_eps)} 集 × {len(track_eps[0]['obs'])} 步  "
          f"var_g={var_g:.4f}")

    # ---------- 4) 离线曲线：f(h) 与 **g_b(h)**（★ 本实验的核心观测量）----------
    #   f(h)   = 从真值出发闭环外推 h 步的误差        —— X31 那条（无量化）
    #   g_b(h) = 从**被量化过的状态**出发外推 h 步的误差 —— X33 新增
    #   ⇒ 数字化 floor 只在 h=0 处注入，之后由模型自己传播；这正是"跨层联合"的本质。
    #
    # ★★ 为什么改用离线曲线而不是"在线做差"（第 15 次自我修正）
    #   第一版我用 N_full − N_drop 去定义量化贡献 A。问题：
    #     · 差值常常只有总量的 2%（b=6 时），而两次在线运行的噪声就在这个量级；
    #     · 实测 b=2 给 A=0.52、b=6 给 A=4.69（**同 PER 同 L 却反号**）
    #       ⇒ 不是机制，是噪声，却很容易被读成"量化被 rollout 洗掉"。
    #   g_b(h) 是**离线**测的（n_samples 个窗口，每个窗口独立），统计量好 1–2 个量级，
    #   而且它顺带把"可加性"这个问题从"做差"变成了"看曲线形状"。
    heads = {"model": model, "persistence": PersistenceHead(model)}
    hs = [int(h) for h in cfg["eval"]["horizons"]]
    hs = [h for h in hs if h <= max(e["length"] for e in eval_eps) - WARMUP - 2]

    def _offline(head, payload=None) -> np.ndarray:
        c = closed_loop_error_curve(head, eval_eps, hs, device,
                                    n_samples=int(cfg["eval"]["n_samples"]),
                                    seed=seed, t0_min=WARMUP, payload_fn=payload)
        d = np.asarray(c["mse_per_step"], dtype=float) / var_g
        return np.concatenate([[0.0], d])

    K = None
    f_curves, g_curves = {}, {}
    for name, head in heads.items():
        f_curves[name] = _offline(head, None)
        K = int(len(f_curves[name])) - 1 if K is None else K
        if len(f_curves[name]) != K + 1:
            raise AssertionError(f"★ S_off 未通过：{name} 的 f 长度不一致")
        g0_ref = {}
        for b in bits_list:
            gb = _offline(head, quantizers[b])
            if len(gb) != K + 1:
                raise AssertionError(f"★ S_off 未通过：({name},b={b}) 的 g 长度 ≠ f")
            gb[0] = float(quantizers[b].measure_mse(allobs) / var_g)   # h=0：片子本身被量化
            if not np.all(np.isfinite(gb)):
                raise FloatingPointError(f"g_{name,b}(h) 含非有限值（R14）")
            g_curves[(name, b)] = gb
            g0_ref[b] = gb[0]
        print(f"[20]   [{name:>11s}] f(1)={f_curves[name][1]:.5f} "
              f"f(10)={f_curves[name][min(10, K)]:.5f} f({K})={f_curves[name][K]:.5f} | "
              + "  ".join(f"g{b}(0)={g_curves[(name, b)][0]:.3e}" for b in bits_list))
    print(f"[20] 离线曲线：K={K} 步（每条稠密到 h={K}）")

    # ★ 可加性的**离线**判据：Δ_b(h) = g_b(h) − f(h)
    #   平（≈ g_b(0)）⇒ 简单可加；随 h 下降 ⇒ 量化被 rollout 洗掉；上升 ⇒ 被放大
    delta_rows = []
    for b in bits_list:
        d = g_curves[("model", b)] - f_curves["model"]
        rel = d / max(d[0], 1e-18)
        slope = float(np.polyfit(np.arange(1, min(30, K) + 1),
                                 d[1:min(30, K) + 1] / max(d[0], 1e-18), 1)[0])
        delta_rows.append({"bits": b, "g0": float(d[0]),
                           "rel_at_K": float(rel[min(K, len(rel) - 1)]),
                           "rel_at_10": float(rel[min(10, len(rel) - 1)]),
                           "slope_per_step": slope})
        print(f"[20]   b={b:>2d}: Δ(h)/Δ(0) @h=10 → {rel[min(10, len(rel) - 1)]:.2f}  "
              f"@h={K} → {rel[min(K, len(rel) - 1)]:.2f}  "
              f"（斜率 {slope:+.3f}/步）")

    h_err = reliable_horizon(hs, [float(x) for x in f_curves["model"][[h for h in hs]]],
                             float(cfg["eval"]["err_threshold"]),
                             cfg["eval"].get("threshold_mode", "rel"))
    print(f"[20] 离线曲线：H*_err={h_err}（背景参考）")

    # ---------- 5) 静态量化地板 N_quant（只依赖 b，扫一次）----------
    print("[20] " + "-" * 76)
    print("[20] (a) 完美信道下的静态量化地板 N_quant")
    quant_pos = {}
    for b in bits_list:
        q = quantizers[b]
        for name, head in heads.items():
            r = run_tracking(head, track_eps, device, lossy_schedule(0.0),
                             seed=track_seed, denom=var_g, warmup=WARMUP,
                             label=f"{name}/quant(b={b})", payload_fn=q)
            if not np.isfinite(r.nmse):
                raise FloatingPointError(f"N_quant(b={b},{name}) 非有限值（R14）")
            quant_pos[(b, name)] = {"nmse": float(r.nmse), "age_mean": float(r.age_tx_mean)}
        m_, p_ = quant_pos[(b, "model")]["nmse"], quant_pos[(b, "persistence")]["nmse"]
        print(f"[20]   b={b:>2d}  N_quant = {m_:.6e}  (persistence {p_:.6e})  "
              f"静态 qmse/var = {quantizers[b].measure_mse(allobs) / var_g:.6e}")

    # ★ S1：完美信道下两 head 必须逐位一致（每步都到 ⇒ 从不调用预测）
    for b in bits_list:
        m_ = quant_pos[(b, "model")]["nmse"]
        p_ = quant_pos[(b, "persistence")]["nmse"]
        if abs(m_ - p_) > max(1e-15, 1e-9 * abs(m_)):
            raise AssertionError(
                f"★ S1 未通过：完美信道下 model={m_:.6e} 与 persistence={p_:.6e} 不一致 "
                f"⇒ 完美信道里仍在调用预测（或 payload 接线不对称），随后的分解不可信")
    print("[20]   S1 通过：完美信道下两 head 逐位一致（确认从不调用预测）")
    # ★ S6：N_quant 对 b 单调减
    qq = [quant_pos[(b, "model")]["nmse"] for b in bits_list]
    if not all(qq[i] >= qq[i + 1] * (1 - 1e-9) for i in range(len(qq) - 1)):
        raise AssertionError(f"★ S6 未通过：N_quant 对 b 非单调减 {qq}")
    print("[20]   S6 通过：静态量化地板对 b 单调递减")
    # ★ S3：与 ground truth 同量级
    for b in bits_list:
        ref = quantizers[b].measure_mse(allobs) / var_g
        got = quant_pos[(b, "model")]["nmse"]
        ratio = got / max(ref, 1e-18)
        if not (0.5 <= ratio <= 2.0):
            raise AssertionError(
                f"★ S3 未通过：b={b} N_quant={got:.4e} 与静态 qmse/var={ref:.4e} "
                f"相差 {ratio:.2f}×（>2× ⇒ 归一化口径或量化接线有问题）")
    print("[20]   S3 通过：N_quant 与静态量化 MSE 同量级")

    # ---------- 6) 主扫 (目标PER × b × L × head) ----------
    print("[20] " + "-" * 76)
    print("[20] (b) 构造工作点：按目标 PER 反解 SNR")
    grid, skipped = [], []
    for L in bursts:
        for b in bits_list:
            for target in per_targets:
                g_db = snr_for_target_per(b, obs_dim, target)
                if g_db is None:
                    skipped.append({"bits": b, "L": L, "target_per": target,
                                    "reason": "该调制档在任何 SNR 下都达不到目标 PER"})
                    print(f"[20]   跳过 b={b} target={target:g}：反解失败（不可达）")
                    continue
                per01 = per_at(g_db, b, obs_dim)
                # ★ PER 极端时跳过：~0 等价完美信道 ⇒ Gilbert 构造会拒绝 p̄=0，
                #   且这样的点 E[age]=0，对分解检验**零信息**（第一版的教训）
                if not (1e-4 <= per01 <= 0.995):
                    skipped.append({"bits": b, "L": L, "target_per": target,
                                    "snr_db": g_db, "per": per01,
                                    "reason": f"PER={per01:.3e} 落在极端区（无信息或链路全断）"})
                    print(f"[20]   跳过 b={b} target={target:g}：反解 PER={per01:.3e} 在极端区")
                    continue
                eff_L = L if L is not None else 1.0 / max(1.0 - per01, 1e-9)
                ok, why = True, ""
                try:
                    GilbertElliottChannel(per01, eff_L, seed=seed + 11)
                except ValueError as e:
                    ok, why = False, str(e)
                if not ok:
                    skipped.append({"bits": b, "L": L, "target_per": target,
                                    "snr_db": g_db, "per": per01,
                                    "reason": f"突发长度不可行：{why}"})
                    print(f"[20]   跳过 b={b} target={target:g} L={L}：PER={per01:.4f} "
                          f"需要 L ≥ {per01 / max(1 - per01, 1e-9):.1f}")
                    continue
                grid.append({"snr_db": float(g_db), "bits": b, "L": L,
                             "target_per": target, "per": float(per01),
                             "eff_L": float(eff_L),
                             "M": float(2.0 ** b),
                             "formula_valid": bool(qam_approximation_valid(
                                 snr_db_to_linear(g_db), 2.0 ** b))})
    print(f"[20]   有效工作点 {len(grid)} 个，跳过 {len(skipped)} 个")
    for g in grid:
        print(f"[20]     b={g['bits']:>2d} L={str(g['L']):>4s} "
              f"targetPER={g['target_per']:<5g} ⇒ SNR={g['snr_db']:>7.2f}dB "
              f"(实际 PER={g['per']:.4f}, M={g['M']:>4.0f}, E[age]≈{g['per'] * g['eff_L']:.1f})")

    print("[20] (c) 三种变体测量：full / drop / 静态地板")
    rows = []
    for g in grid:
        g_db, b, L = g["snr_db"], g["bits"], g["L"]
        rec = dict(g)
        pmf_th = full_age_pmf(g["per"], 1.0 / g["eff_L"], K)
        for name, head in heads.items():
            ch_full = PhysicalChannel(snr_db=g_db, bits=b, obs_dim=obs_dim,
                                      quantizer=quantizers[b], burst_len=L,
                                      seed=seed + 11)
            r_full = run_tracking(head, track_eps, device, ch_full,
                                  seed=track_seed, denom=var_g, warmup=WARMUP,
                                  label=f"{name}/full", payload_fn=ch_full.payload_fn())
            ch_noq = PhysicalChannel(snr_db=g_db, bits=b, obs_dim=obs_dim,
                                     quantizer=None, burst_len=L, seed=seed + 11)
            r_noq = run_tracking(head, track_eps, device, ch_noq, seed=track_seed,
                                 denom=var_g, warmup=WARMUP, label=f"{name}/drop",
                                 payload_fn=None)
            if not np.isfinite(r_full.nmse) or not np.isfinite(r_noq.nmse):
                raise FloatingPointError(
                    f"S4：(b={b},PER={rec['per']:.3f},L={L},{name}) 产出非有限值（R14）")
            n_q = quant_pos[(b, name)]["nmse"]
            inter = float(r_full.nmse - r_noq.nmse - n_q)
            ampl = float((r_full.nmse - r_noq.nmse) / max(n_q, 1e-18))
            # ------------------------------------------------ 闭式预报 ----------
            # ★ 两类 grandfather 代入：① 闭式精确分布（真正能用来做设计）
            #                        ② 实测年龄直方图（不依赖任何分布假设）
            hist_ = {int(k): int(v) for k, v in r_full.age_hist.items()}
            n_tot = max(sum(hist_.values()), 1)
            p_emp = np.zeros_like(f_curves[name])
            for k, c in hist_.items():
                p_emp[min(int(k), K)] += c / n_tot
            curve_g = g_curves[(name, b)]
            curve_f = f_curves[name]
            n_full = float(r_full.nmse)
            n_drop = float(r_noq.nmse)
            rec[name] = {"nmse_full": n_full,
                         "nmse_drop": n_drop,
                         "nmse_quant": float(n_q),
                         "interaction": inter,
                         "interaction_rel": float(inter / max(n_full, 1e-18)),
                         "amplification": ampl,
                         "age_mean": float(r_full.age_tx_mean),
                         "quant_share": float((n_full - n_drop) / max(n_full, 1e-18)),
                         # ★ 预报 vs 实测
                         "pred_full_analytic": float(np.dot(pmf_th, curve_g)),
                         "pred_full_empirical": float(np.dot(p_emp, curve_g)),
                         "pred_drop_analytic": float(np.dot(pmf_th, curve_f)),
                         "gap_full_analytic": float(np.dot(pmf_th, curve_g)
                                                    / max(n_full, 1e-18) - 1.0),
                         "gap_full_empirical": float(np.dot(p_emp, curve_g)
                                                     / max(n_full, 1e-18) - 1.0),
                         "gap_drop_analytic": float(np.dot(pmf_th, curve_f)
                                                    / max(n_drop, 1e-18) - 1.0)}
        rows.append(rec)
        rm, rp = rec["model"], rec["persistence"]
        print(f"[20]   SNR={g_db:>6.1f}dB b={b:>2d} L={str(L):>4s} PER={rec['per']:.4f} "
              f"E[age]={rm['age_mean']:5.2f} | model full={rm['nmse_full']:.5f} "
              f"预报={rm['pred_full_analytic']:.5f}({rm['gap_full_analytic']:+.1%}) "
              f"drop 预报偏差={rm['gap_drop_analytic']:+.1%} "
              f"| persist full={rp['nmse_full']:.5f}"
              f"({rp['gap_full_analytic']:+.1%})")

    if not rows:
        raise AssertionError("★ 所有网格点都被跳过（突发长度全不可行）⇒ 网格要重设")

    # ---------- 7) 自检 S2 / S5 / S7 ----------
    n_wired = 0
    for r in rows:
        if r["per"] > 0.999:
            continue                      # 链路全丢 ⇒ 量化没机会生效
        for name in ("model", "persistence"):
            d = r[name]
            # S2：加一层量化只会让误差更糟；显著变好 ⇒ 有 bug 或反常，必须报
            if d["nmse_full"] < d["nmse_drop"] * 0.95:
                raise AssertionError(
                    f"★ S2 未通过：{r['snr_db']}dB b={r['bits']} L={r['L']} {name} "
                    f"full={d['nmse_full']:.5f} < drop={d['nmse_drop']:.5f}×0.95 "
                    f"⇒ 加了量化反而更好，检查口径")
            # S5（R12）：量化必须真的改变了结果
            if abs(d["nmse_full"] - d["nmse_drop"]) < 1e-12:
                continue
            n_wired += 1
        # S7：同一信道下两 head 的年龄序列必须相同
        if abs(r["model"]["age_mean"] - r["persistence"]["age_mean"]) > 1e-9:
            raise AssertionError(
                f"★ S7 未通过：{r['snr_db']}dB b={r['bits']} 两 head 的 E[age] 不同 "
                f"({r['model']['age_mean']:.4f} vs {r['persistence']['age_mean']:.4f}) "
                f"⇒ 信道的随机数消耗依赖了估计器，同龄对照失效")
    if n_wired == 0:
        raise AssertionError(
            "★ S5 未通过：所有非饱和点的 full / drop 都逐位相同 "
            "⇒ payload_fn 没接线（R12 违规）")
    print(f"[20]   S2/S5/S7 通过（{n_wired} 个点验到量化接线，年龄序列与 head 无关）")

    # ---------- 8) 判据 ----------
    print("[20] " + "=" * 76)
    valid = [r for r in rows if r["per"] <= 0.999]

    # ===== 主判据 W：闭式预报（★ 只用离线曲线 + 信道参数，没有用该点的在线实测）=====
    def _errs(key, name="model"):
        return np.array([r[name][key] for r in valid], dtype=float)

    gap_full = np.abs(_errs("gap_full_analytic"))
    gap_emp = np.abs(_errs("gap_full_empirical"))
    gap_drop = np.abs(_errs("gap_drop_analytic"))
    w_med, w_p90 = float(np.median(gap_full)), float(np.percentile(gap_full, 90))
    if w_med < 0.10 and w_p90 < 0.25:
        verdict_w = (f"W1 ★闭式预报成立：|偏差| 中位 {w_med:.1%}、90 分位 {w_p90:.1%} "
                     f"⇒ 任意 (b, PER, L) 都可只凭离线曲线 + 信道参数预报")
    elif w_med < 0.30:
        verdict_w = (f"W2 定性成立、定量不够：|偏差| 中位 {w_med:.1%}、90 分位 {w_p90:.1%}")
    else:
        verdict_w = (f"W3 ★闭式预报失效：|偏差| 中位 {w_med:.1%}、90 分位 {w_p90:.1%} "
                     f"⇒ 不能只用离线曲线预报联合误差")
    print(f"[20] ★ {verdict_w}")
    print(f"[20]   对照：用**实测**年龄直方图代入 ⇒ 中位 {np.median(gap_emp):.1%}"
          f"（若这条也差 ⇒ 不是分布假设的问题，是 Σ_h P(h)·g(h) 这个分解本身不成立）")
    print(f"[20]   对照：同一套机械**不带量化**（即 X31 的旧结论）⇒ 中位 "
          f"{np.median(gap_drop):.1%}")

    # ===== 可加性（离线版，比在线做差可靠）=====
    flat = [d for d in delta_rows if abs(d["rel_at_K"] - 1.0) <= 0.2]
    wash = [d for d in delta_rows if d["rel_at_K"] < 0.8]
    grow = [d for d in delta_rows if d["rel_at_K"] > 1.25]
    if len(flat) == len(delta_rows):
        verdict_add = (f"A1 可加：Δ_b(h)/Δ_b(0) 在所有 b 上都≈1（±20% 内）"
                       f"⇒ 量化贡献与时间无关，简单相加成立")
    elif len(wash) >= len(grow):
        verdict_add = (f"A2 ★不可加：{len(wash)}/{len(delta_rows)} 个 b 的量化贡献"
                       f"随年龄**衰减**到 {min(d['rel_at_K'] for d in wash):.0%}"
                       f"⇒ 量化误差被 rollout 洗掉了（不是被放大）")
    else:
        verdict_add = (f"A3 ★不可加：{len(grow)}/{len(delta_rows)} 个 b 的量化贡献"
                       f"随年龄**放大**到 {max(d['rel_at_K'] for d in grow):.1f}×")
    print(f"[20] ★ {verdict_add}")

    # ===== 次级诊断（噪声大，仅作参考，不进主结论）=====
    amp_m = np.array([r["model"]["amplification"] for r in valid])
    amp_p = np.array([r["persistence"]["amplification"] for r in valid])
    irel_max = float(np.max(np.abs([r["model"]["interaction_rel"] for r in valid])))
    print(f"[20]   次级（在线做差，噪声大）：|I|/N_full 最大 {irel_max:.1%}；"
          f"A_model 中位 {np.median(amp_m):.2f} / A_persist {np.median(amp_p):.2f}")

    # ---------- 9) 图 ----------
    fig, axes = plt.subplots(2, 3, figsize=(17.5, 9.5))
    fig.suptitle(
        f"X33 丢包 × 量化：闭式预报与可分解性  ·  {verdict_w}\n{verdict_add}",
        fontsize=10.5)

    # ① ★ 闭式预报 vs 实测（主图）
    ax = axes[0, 0]
    for name, col, mk in (("model", PALETTE["blue"], "o"),
                          ("persistence", PALETTE["orange"], "s")):
        xs = [r[name]["pred_full_analytic"] for r in valid]
        ys = [r[name]["nmse_full"] for r in valid]
        ax.plot(xs, ys, mk, color=col, alpha=0.8, label=name)
    vmin = min(min(r[name]["nmse_full"] for r in valid for name in ("model", "persistence")),
               min(r[name]["pred_full_analytic"] for r in valid
                   for name in ("model", "persistence")))
    vmin = max(vmin, 1e-7)
    vmax = max(r["model"]["nmse_full"] for r in valid) * 1.5
    ax.plot([vmin, vmax], [vmin, vmax], "k:", lw=1, label="y=x（预报成立）")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("闭式预报 Σ_h P_age(h)·g_b(h)")
    ax.set_ylabel("实测 E[NMSE]")
    ax.set_title(f"① ★ 跨层闭式预报（|偏差| 中位 {w_med:.1%}）")
    ax.legend(fontsize=8)

    # ② ★ 离线侧的可加性诊断：Δ_b(h) = g_b(h) − f(h) 随年龄如何变化
    ax = axes[0, 1]
    ax.plot(range(K + 1), f_curves["model"], "-", color=PALETTE["grey"], lw=2,
            label="f(h)（无量化，X31 那条）")
    for b in bits_list:
        gb = g_curves[("model", b)]
        ax.plot(range(K + 1), gb, "-", lw=1.3, label=f"g_b, b={b}")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_yscale("log")
    ax.set_xlabel("信息年龄 h（步）")
    ax.set_ylabel("归一化 MSE（全局分母）")
    ax.set_title("② 离线曲线：g_b(h) vs f(h)")
    ax.legend(fontsize=7, ncol=2)

    # ③ 可加性量的形状：Δ_b(h)/Δ_b(0)
    ax = axes[0, 2]
    for b in bits_list:
        d = g_curves[("model", b)] - f_curves["model"]
        rel = d / max(d[0], 1e-18)
        ax.plot(range(K + 1), rel, "-", lw=1.5, label=f"b={b}")
    ax.axhline(1.0, color=PALETTE["grey"], ls=":", lw=1.2,
               label="1.0 = 量化贡献与时间无关（可加）")
    hi = max(float(np.max((g_curves[("model", b)] - f_curves["model"])
                          / max(abs((g_curves[("model", b)] - f_curves["model"])[0]), 1e-18)))
             for b in bits_list)
    ax.set_xlabel("信息年龄 h（步）")
    ax.set_ylabel("量化贡献 / Δ_b(0)")
    ax.set_ylim(0.0, min(max(2.0, 1.2 * hi), 6.0))
    ax.set_title("③ ★ 量化贡献是否随年龄变化")
    ax.legend(fontsize=8)

    # ④ 预报误差分解：用**实测**直方图 vs 用**闭式**分布
    ax = axes[1, 0]
    xs = np.arange(len(valid))
    ax.plot(xs, 100 * _errs("gap_full_analytic"), "o-", color=PALETTE["blue"], ms=4,
            label="闭式 pmf（可预报 ⇒ 能做设计）")
    ax.plot(xs, 100 * _errs("gap_full_empirical"), "s--", color=PALETTE["green"], ms=4,
            label="实测 pmf（检验分解本身）")
    ax.plot(xs, 100 * _errs("gap_drop_analytic"), "^:", color=PALETTE["grey"], ms=4,
            label="无量化（X31 旧结论）")
    ax.axhspan(-10, 10, color=PALETTE["grey"], alpha=0.15)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("网格点（按 b × PER × L 排序）")
    ax.set_ylabel("预报相对偏差 (%)")
    ax.set_title("④ 预报偏差逐点（±10% 带）")
    ax.legend(fontsize=7.5)

    # ⑤ 偏差 vs E[age]：偏差集中在哪个区间
    ax = axes[1, 1]
    for name, col in (("model", PALETTE["blue"]), ("persistence", PALETTE["orange"])):
        xs2 = np.array([r["model"]["age_mean"] for r in valid])
        ys2 = 100 * np.abs(_errs("gap_full_analytic", name))
        ax.plot(xs2, ys2, "o" if name == "model" else "s", color=col, ms=4, alpha=0.8,
                label=f"{name}（中位 {np.median(ys2):.1f}%）")
    ax.set_xlabel("E[age]（步）"); ax.set_ylabel("|预报偏差| (%)")
    ax.set_title("⑤ 偏差是否集中在特定年龄区间")
    ax.legend(fontsize=8)

    # ⑥ 两个预测器的 map：同一套物理层缺陷，代价是否不同
    ax = axes[1, 2]
    xs2 = [r["model"]["nmse_full"] for r in valid]
    ys2 = [r["persistence"]["nmse_full"] for r in valid]
    ax.plot(xs2, ys2, "o", color=PALETTE["purple"], ms=5)
    lims = [min(min(xs2), min(ys2)) * 0.7, max(max(xs2), max(ys2)) * 1.4]
    ax.plot(lims, lims, "k:", lw=1, label="相等")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("世界模型 E[NMSE]"); ax.set_ylabel("persistence E[NMSE]")
    ax.set_title("⑥ 同一组配置下两个预测器（点在下方 ⇒ 模型更差）")
    ax.legend(fontsize=8)

    save_fig(fig, os.path.join(out, args.tag + ".png"))
    plt.close(fig)

    payload = {
        "experiment": "X33 丢包 × 量化：跨层闭式预报与误差可分解性",
        "config": {k: cfg[k] for k in ("env", "controller", "physical", "eval", "task")},
        "obs_dim": obs_dim, "var_g": var_g, "h_err": h_err, "max_age_K": K,
        "quant_floor": {str(b): quant_pos[(b, "model")]["nmse"] for b in bits_list},
        "static_quant_mse": {str(b): float(quantizers[b].measure_mse(allobs))
                             for b in bits_list},
        "offline_delta": delta_rows,
        "rows": rows,
        "skipped": skipped,
        "prediction_error": {
            "full_analytic_median": w_med,
            "full_analytic_p90": w_p90,
            "full_empirical_median": float(np.median(gap_emp)),
            "drop_analytic_median": float(np.median(gap_drop)),
            "n_points": int(len(valid))},
        "diagnostic_noisy": {
            "interaction_rel_max": float(irel_max),
            "amp_median": {"model": float(np.median(amp_m)),
                           "persistence": float(np.median(amp_p))}},
        "verdict": {"prediction": verdict_w, "additivity": verdict_add},
        "caveat": ("★ 主结论（W 判据）只用了「离线曲线 g_b(h)」+「信道参数 (p̄, L)」，"
                   "没有使用被预报配置的任何在线实测 ⇒ 它本身就是「预报未跑过的配置」。"
                   "★ 次级诊断 A = (N_full−N_drop)/N_quant 是两次在线运行的差分，"
                   "在 N_quant 远小于 N_drop 时信噪比很差，**不得单独引用**。"
                   "★ BER/PER 闭式仍是教科书近似（见 physical.py 的「待核」声明）："
                   "本实验的结论依赖其定性结构，任何「SNR = X dB 时 PER = Y」的定量"
                   "表述在核对前不得写进论文。"),
    }
    with open(os.path.join(out, args.tag + ".json"), "w", encoding="utf-8") as fp:
        json.dump(jsonable(payload), fp, ensure_ascii=False, indent=2)
    print(f"[20] 产物：{os.path.join(out, args.tag)}.png / .json  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
