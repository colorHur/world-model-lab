# -*- coding: utf-8 -*-
"""★ X36「突发信道下的联合设计：丢了『突发结构』还能不能做 (T, b) 设计？」

================================================================= ★ 由来（为什么必须有这一号）
X31 实测：平均丢包率相同（50%）时，突发结构能让跟踪误差跨 **570 倍**。
但 X34 的联合设计（周期 T × 精度 b）**只用了逐包独立丢包** —— 在它的闭式里，
信道被压缩成了一个数 PER。

⇒ 必须回答：**做 (T,b) 联合设计时，PER（乃至总丢包率）够不够用？**
   判法：固定设计网格，只改突发的**结构**（保持总丢包率不变），看最优周期 T\* 动不动。

================================================================= ★ 物理模型（三条假设，跑前定死）
1) 链路层**阻塞** = Gilbert–Elliott 两状态链，**按「尝试」演化**（每 T 步走一步链）。
   Good 全通、Bad 全丢 —— 与 X31 同口径（最简 Gilbert）。
2) 非阻塞期再叠加**物理层逐包噪声丢包** PER(b, m, SNR) —— 与 X32/X34 同口径。
3) 一次尝试成功 ⇔ (信道 Good) 且 (无噪声错包) ⇒ s = (1−p̄)(1−PER)。
⇒ p̄（阻塞率）是**环境参数**、PER 是**设计参数**，两者独立。

================================================================= ★ 新闭式（本次推导，已验量级）
以「尝试次数」为单位的到达间隔 D_a 由 2×2 矩阵给出（精确，不靠仿真）：
    P(D_a = d) = v0 · F^{d−1} · s,   v0 = Pm[Good, :],  F = Pm·(1−s_i)（行乘）
年龄分布用**更新过程的平稳年龄公式**（注意是长度偏采样，不是"间隔内均匀"）：
    P(age = h) = P(D_a ≥ ⌊h/T⌋+1) / (T·E[D_a])
    E[age]     = T·(E[D_a²]−E[D_a])/(2E[D_a]) + (T−1)/2
四条已验的退化（自检 ㉝/㉞ + 本脚本 S_iid）：
  · L = L_iid ⇒ GE 无记忆 ⇒ 与 `periodic_age_pmf(q_total,T)` **逐点相等**（2.6e-12）
  · T=1 且 PER=0 ⇒ E[age] = p̄·L（X31 的闭式，1e-10）
  · pmf 归一化且 pmf 均值 == 闭式均值（1e-9）
  · 调度仿真直方图 vs 闭式 pmf：TV ≈ 0.003

================================================================= ★ 预写判据（跑前定死）
E1  ★ 同一 (SNR,b) 下把 L 从 L_iid 拉到 16·L_iid，**闭式选出的 T\* 发生偏移（≥1 档）**
     的比例 ≥ 50% ⇒ 「总丢包率不足以做设计」的定量证据
E2  突发感知年龄分布的预报 |偏差| 中位 < 15%；**无记忆**年龄分布（同总丢包率）的
     预报偏差**显著更大**（两者中位之比 ≥ 1.5×）
E3  ★ 方向：预写 **T\* 随突发变长而单调不增**
     （理由：坏串里"多给几次机会"比"提高单次可靠性"更值 —— 坏串里提高可靠性也送不出去）
     判据：在 T\* 发生变化的 (SNR,b) 组里，≥70% 是「L 变大 ⇒ T\* 变小」
E4  ★ R12：同 (SNR,b,T) 下改突发长度 L 必须改变实测 NMSE（≥5%），否则 GE 没接进调度
E5  S_a：实测 E[age] 与闭式一致，容差 = **3 × 本点采样标准误**
     （第 17 次自证伪的教训：固定百分比会在高丢包点被采样噪声打穿）

================================================================= 自检（不过就抛 —— R14）
S_a    见 E5
S_b    全部有限（R14）
S_iid  ★ L = L_iid 时，突发管线的 pred/nmse 必须与**无记忆**管线一致（相差 <0.5%）
S_c    b 最小 + T 最小 + 阻塞率最小的点，NMSE 必须接近全场最小
S_d    ★ R12：改 L 必须改变实测结果（E4 的机器版）
"""
from __future__ import annotations

import argparse
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

from wmlab.control import PDRelativeController, collect_controlled_episodes
from wmlab.data import split_episodes, transitions_from_episodes
from wmlab.envs import make_env
from wmlab.eval import reliable_horizon
from wmlab.eval.physical import (UniformQuantizer, fit_quantizer_range, overload_fraction,
                                 per_from_ber, qam_approximation_valid, qam_bit_error_rate,
                                 snr_db_to_linear)
from wmlab.eval.tracking import (burst_periodic_age_pmf, burst_periodic_interarrival,
                                 burst_periodic_lossy_schedule, burst_periodic_mean_age,
                                 periodic_age_pmf, periodic_mean_age, run_tracking)
from wmlab.models import MLPWorldModel
from wmlab.rollout import closed_loop_error_curve
from wmlab.train import train_world_model
from wmlab.utils import count_params, get_device, load_config, output_dir, set_seed
from wmlab.utils.plot import PALETTE, apply_style, save_fig


def jsonable(o):
    """把 numpy / 非 JSON 原生类型递归转成可序列化对象（与 20/21/22 同口径）。"""
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    return o


class PersistenceHead:
    """基线预测器：预测"下一帧 = 上一帧"（不含任何学习）。"""

    def __init__(self, ref):
        self.ref = ref

    def eval(self):
        return self

    def to(self, *a, **k):
        return self

    def predict_next(self, obs, act):
        return obs


def _burst_tail(p_bar: float, L: float, per: float, T: int, K: int) -> float:
    """P(age > K)：用 4 倍视界算一遍，取 K 之外的质量（避免另写一套尾部闭式）。"""
    big = burst_periodic_age_pmf(p_bar, L, per, T, 4 * K + 4)
    return float(1.0 - big[:K + 1].sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(_ROOT, "configs", "uav_burst_design.yaml"))
    ap.add_argument("--quick", action="store_true", help="极小规模冒烟")
    args = ap.parse_args()

    cfg = load_config(args.config)
    dz = cfg["design"]
    seed = int(cfg.get("seed", 0))
    set_seed(seed)
    device = get_device(cfg.get("device", "auto"))
    out = output_dir(cfg)
    t0 = time.time()

    snrs = [float(s) for s in dz["snr_db_list"]]
    bits_list = [int(b) for b in dz["bits_list"]]
    m_list = [int(m) for m in dz["m_sym_list"]]
    p_bars = [float(p) for p in dz["p_bar_list"]]
    k_list = [float(k) for k in dz["burst_k_list"]]
    max_T = int(dz.get("max_period", 32))
    n_track = int(dz.get("n_track_episodes",
                         cfg["eval"].get("n_track_episodes", 40)))

    if args.quick:
        cfg["train"]["epochs"] = min(int(cfg["train"]["epochs"]), 3)
        cfg["data"]["n_episodes"] = min(int(cfg["data"]["n_episodes"]), 20)
        snrs = [snrs[0]]
        bits_list = [4, 8]
        p_bars = [p_bars[0]]
        k_list = [1.0, 4.0]
        cfg["eval"]["n_samples"] = 64
        n_track = 12

    print("[23] " + "=" * 76)
    print("[23] ★ X36 突发信道下的联合设计：丢了『突发结构』还能不能做 (T,b) 设计")
    print(f"[23]   SNR ∈ {snrs} dB   b ∈ {bits_list}   m ∈ {m_list}")
    print(f"[23]   阻塞率 p̄ ∈ {p_bars}   突发长度 L = k×L_iid，k ∈ {k_list}")
    print("[23]   预写判据：E1 T* 随 L 偏移的比例 ≥50%   E2 突发感知预报更准（≥1.5×）   "
          "E3 T* 随 L 单调不增")

    # ---------- 1) 训练（与 X30–X35 同一套流程）----------
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
    print(f"[23] 训练完成：params={count_params(model):,} val={hist['val_total'][-1]:.6f} "
          f"({time.time() - t0:.0f}s)")
    heads = {"model": model, "persistence": PersistenceHead(model)}

    # ---------- 2) 数据 / 量化器 / 离线曲线 ----------
    WARMUP = int(cfg["task"]["warmup_steps"])
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
    track_seed = seed + int(cfg["data"].get("online_seed_offset", 2000))
    track_eps = eval_eps[:n_track] if n_track <= len(eval_eps) else eval_eps

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
        f_curves[name] = _offline(head, None)
        K = len(f_curves[name]) - 1 if K is None else K
        if len(f_curves[name]) != K + 1:
            raise AssertionError(f"★ S_off 未通过：{name} 曲线长度不一致")
        for b in bits_list:
            gb = _offline(head, quantizers[b])
            gb[0] = float(quantizers[b].measure_mse(allobs) / var_g)
            if not np.all(np.isfinite(gb)):
                raise FloatingPointError(f"g_{name,b}(h) 含非有限值（R14）")
            g_curves[(name, b)] = gb
        print(f"[23]   [{name:>11s}] f(1)={f_curves[name][1]:.5f} "
              f"f(10)={f_curves[name][min(10, K)]:.5f} f(K={K})={f_curves[name][K]:.5f}")
    print(f"[23] 离线：K={K} 步")

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
                skipped.append({"bits": b, "m": m, "T": T, "reason": "周期越界"})
                continue
            grid.append({"bits": b, "m": m, "T": T})

    chans = []
    for pb in p_bars:
        l_iid = 1.0 / (1.0 - pb)
        for k in k_list:
            L = k * l_iid
            if L < pb / (1.0 - pb) - 1e-12:
                skipped.append({"p_bar": pb, "k": k, "reason": "L 低于 GE 可行域"})
                continue
            chans.append({"p_bar": pb, "k": k, "L": float(L), "l_iid": float(l_iid)})
    print(f"[23] 设计网格：{len(grid)} 个 (b,T) × {len(chans)} 个信道档 "
          f"× {len(snrs)} 个 SNR ⇒ {len(grid)*len(chans)*len(snrs)} 点（跳过 {len(skipped)}）")

    steps_per_point = int(sum(min(e["length"], int(cfg["env"]["max_steps"]))
                              for e in track_eps))

    def _age_se(T, pb, L, per, n_rep=12) -> float:
        """★ 本点 E[age] 的**采样标准误**（纯调度空跑，不含环境）。

        ★★ 必须**逐条 episode 模仿 run_tracking**：同样的 episode 长度、每集重新
        从 age=0 开始、**跳过 warmup**。第一版没跳 warmup ⇒ 样本数按 700 算而实际
        只有 300 ⇒ SE 低估约 1.5 倍，冒烟里直接把 −1.8σ 的噪声判成了"闭式错"。
        （诊断脚本证明：纯调度在 200 条 episode 时收敛到闭式 3.21±0.25 vs 3.278
        ⇒ 闭式没错，是样本量。）
        """
        vals = []
        ep_lens = [min(int(e["length"]), int(cfg["env"]["max_steps"])) for e in track_eps]
        for i in range(n_rep):
            rng = np.random.default_rng(track_seed + 977 * (i + 1))
            sch = burst_periodic_lossy_schedule(T, pb, L, per, seed=track_seed + i)
            tot, n = 0.0, 0
            for L_ep in ep_lens:
                age = 0
                for t in range(L_ep):
                    if sch(t, rng):
                        age = 0
                    else:
                        age = min(age + 1, K)
                    if t >= WARMUP:              # ★ 与 run_tracking 的 warmup 同口径
                        tot += age
                        n += 1
            vals.append(tot / max(n, 1))
        return float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

    # ---------- 4) 扫 ----------
    print("[23] " + "-" * 76)
    rows = []
    for snr in snrs:
        for ch in chans:
            pb, L, k = ch["p_bar"], ch["L"], ch["k"]
            for gd in grid:
                b, m, T = gd["bits"], gd["m"], gd["T"]
                per01 = per_of(snr, b, m)
                q_tot = 1.0 - (1.0 - pb) * (1.0 - per01)
                if q_tot >= 0.999:
                    skipped.append({**gd, **ch, "snr_db": snr, "reason": "总丢包率≈1"})
                    continue
                tail = _burst_tail(pb, L, per01, T, K)
                if tail > float(dz.get("max_tail_mass", 0.02)):
                    skipped.append({**gd, **ch, "snr_db": snr, "tail": tail,
                                    "reason": "年龄尾部超可用视界"})
                    continue
                pmf = burst_periodic_age_pmf(pb, L, per01, T, K)
                pmf_mem = periodic_age_pmf(q_tot, T, K)
                se = _age_se(T, pb, L, per01)
                rec = {"snr_db": float(snr), "bits": b, "m": m, "T": T,
                       "p_bar": float(pb), "L": float(L), "k": float(k),
                       "l_iid": float(ch["l_iid"]), "per": float(per01),
                       "q_total": float(q_tot), "tail_mass": float(tail),
                       "age_mean_analytic": float(burst_periodic_mean_age(pb, L, per01, T)),
                       "age_mean_pmf_trunc": float(np.dot(np.arange(K + 1), pmf)),
                       "age_mean_memless": float(periodic_mean_age(q_tot, T)),
                       "age_se": float(se)}
                for name, head in heads.items():
                    gb = g_curves[(name, b)]
                    pred = float(np.dot(pmf, gb))
                    pred_mem = float(np.dot(pmf_mem, gb))
                    r = run_tracking(head, track_eps, device,
                                     burst_periodic_lossy_schedule(T, pb, L, per01,
                                                                   seed=track_seed),
                                     seed=track_seed, denom=var_g, warmup=WARMUP,
                                     label=f"{name}/T={T}/L={L:g}",
                                     payload_fn=quantizers[b])
                    if not np.isfinite(r.nmse):
                        raise FloatingPointError(
                            f"S_b：(snr={snr},b={b},T={T},L={L},{name}) NMSE 非有限（R14）")
                    emp = float(r.age_tx_mean)
                    tol = max(0.10, 3.0 * max(se, 1e-3))
                    if abs(emp - rec["age_mean_pmf_trunc"]) > tol:
                        raise AssertionError(
                            f"★ S_a 未通过：SNR={snr} T={T} p̄={pb} L={L:g} per={per01:.4f} "
                            f"实测 E[age]={emp:.4f} vs 闭式 {rec['age_mean_pmf_trunc']:.4f}"
                            f"（容差 {tol:.4f} = 3×SE={se:.4f}）")
                    rec[name] = {"nmse": float(r.nmse), "pred": pred, "pred_memless": pred_mem,
                                 "gap": float(pred / max(r.nmse, 1e-18) - 1.0),
                                 "gap_memless": float(pred_mem / max(r.nmse, 1e-18) - 1.0),
                                 "age_mean": emp}
                rows.append(rec)
                rm = rec["model"]
                print(f"[23] {snr:>4g}dB b={b} T={T} p̄={pb:.2f} L={L:>6.2f}(k={k:>4g}) "
                      f"PER={per01:.4f} qt={rec['q_total']:.4f} | "
                      f"E[age]={rm['age_mean']:6.2f}(闭式{rec['age_mean_pmf_trunc']:6.2f}/"
                      f"无记忆{rec['age_mean_memless']:6.2f}) | "
                      f"NMSE 实测={rm['nmse']:.5f} 突发预报={rm['pred']:.5f}"
                      f"({rm['gap']:+.0%}) 无记忆预报({rm['gap_memless']:+.0%})")

    if not rows:
        raise AssertionError("★ 网格为空 ⇒ 调整 SNR / b / m / p̄ / k")

    # ---------- 5) E4 / S_d（R12）：改突发长度必须改变实测 ----------
    n_wired = 0
    for snr in snrs:
        for gd in grid:
            for pb in p_bars:
                sub = [r for r in rows if abs(r["snr_db"] - snr) < 1e-9
                       and r["bits"] == gd["bits"] and r["T"] == gd["T"]
                       and abs(r["p_bar"] - pb) < 1e-12]
                if len(sub) < 2:
                    continue
                vals = [r["model"]["nmse"] for r in sub]
                if max(vals) / max(min(vals), 1e-18) > 1.05:
                    n_wired += 1
    if n_wired == 0:
        raise AssertionError("★ E4/S_d 未通过：改突发长度 L **不改变实测 NMSE** "
                             "⇒ GE 没接进调度（R12 违规），整个实验在空跑")
    print(f"[23]   E4/S_d 通过：{n_wired} 个 (SNR,b,T,p̄) 组验到突发真的起作用")

    # ---------- 6) 判据 ----------
    print("[23] " + "=" * 76)
    gap = np.abs(np.array([r["model"]["gap"] for r in rows]))
    gap_m = np.abs(np.array([r["model"]["gap_memless"] for r in rows]))
    e2_med, e2_med_mem = float(np.median(gap)), float(np.median(gap_m))
    ratio = e2_med_mem / max(e2_med, 1e-12)
    print(f"[23] ★ E2：突发感知预报 |偏差| 中位 {e2_med:.1%}（90 分位 "
          f"{float(np.percentile(gap, 90)):.1%}）｜无记忆口径 {e2_med_mem:.1%} "
          f"（90 分位 {float(np.percentile(gap_m, 90)):.1%}）⇒ 倍数 {ratio:.2f}×"
          f"{'  ✅' if ratio >= 1.5 and e2_med < 0.15 else '  ★ 未达标'}")
    e2_ok = bool(ratio >= 1.5 and e2_med < 0.15)

    # T* 随 L 的移动
    moves = []
    for snr in snrs:
        for b in bits_list:
            for pb in p_bars:
                sub = [r for r in rows if abs(r["snr_db"] - snr) < 1e-9
                       and r["bits"] == b and abs(r["p_bar"] - pb) < 1e-12]
                ks = sorted({r["k"] for r in sub})
                if len(ks) < 2:
                    continue
                tstar = {}
                for k in ks:
                    cand = sorted([r for r in sub if abs(r["k"] - k) < 1e-9],
                                  key=lambda r: r["T"])
                    if not cand:
                        continue
                    best = min(cand, key=lambda r: r["model"]["pred"])
                    tstar[k] = best["T"]
                if len(tstar) < 2:
                    continue
                kk = sorted(tstar)
                moves.append({"snr_db": snr, "bits": b, "p_bar": pb,
                              "T_by_k": {float(k): int(tstar[k]) for k in kk},
                              "moved": int(tstar[kk[0]] != tstar[kk[-1]]),
                              "delta": int(tstar[kk[-1]] - tstar[kk[0]])})
    n_moved = sum(m["moved"] for m in moves)
    frac = n_moved / max(len(moves), 1)
    print(f"[23] ★ E1：突发从 L_iid 拉到最大档后，闭式选出的 T* 发生偏移 "
          f"{n_moved}/{len(moves)}（{frac:.0%}）{'  ✅' if frac >= 0.5 else '  ★ 未达标'}")
    e1_ok = bool(frac >= 0.5)

    changed = [m for m in moves if m["moved"]]
    n_down = sum(1 for m in changed if m["delta"] < 0)
    e3_frac = n_down / max(len(changed), 1)
    print(f"[23] ★ E3：T* 变化的 {len(changed)} 组里，L 变大 ⇒ T* 变小 的 "
          f"{n_down}/{len(changed)}（{e3_frac:.0%}）"
          f"{'  ✅' if e3_frac >= 0.7 else '  ★ 预写方向不成立'}")
    e3_ok = bool(e3_frac >= 0.7)
    for m in moves:
        print(f"[23]     SNR={m['snr_db']:>4g}dB b={m['bits']} p̄={m['p_bar']:.2f}  "
              f"T*(k) = {m['T_by_k']}")

    # S_iid：k=1（无记忆）时突发管线必须与无记忆管线一致
    iid = [r for r in rows if abs(r["k"] - 1.0) < 1e-12]
    if iid:
        d = max(abs(r["model"]["pred"] - r["model"]["pred_memless"])
                / max(r["model"]["pred_memless"], 1e-18) for r in iid)
        if d > 5e-3:
            raise AssertionError(f"★ S_iid 未通过：L=L_iid 时突发口径与无记忆口径差 "
                                 f"{d:.2%}（>0.5%）⇒ 新闭式退化不成立")
        print(f"[23]   S_iid 通过：{len(iid)} 个 L=L_iid 点与无记忆口径一致（最大差 {d:.2%}）")

    # ---------- 7) 图 ----------
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.0))
    fig.suptitle("X36 突发信道下的联合设计：总丢包率够不够做设计？", fontsize=11)
    ax = axes[0, 0]
    for pb in p_bars:
        for b in bits_list:
            sub = [r for r in rows if abs(r["p_bar"] - pb) < 1e-12 and r["bits"] == b
                   and abs(r["snr_db"] - snrs[len(snrs) // 2]) < 1e-9]
            if not sub:
                continue
            sub = sorted(sub, key=lambda r: (r["T"], r["k"]))
            ax.plot([r["L"] for r in sub], [r["model"]["nmse"] for r in sub], "o-",
                    label=f"p̄={pb} b={b}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("突发长度 L（步）")
    ax.set_ylabel("闭环 NMSE")
    ax.set_title("① 同丢包率下改突发结构")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(gap_m, gap, "o", alpha=0.7)
    lim = [0, max(float(gap.max()), float(gap_m.max())) * 1.1]
    ax.plot(lim, lim, "k:", lw=1)
    ax.set_xlabel("无记忆口径 \|预报偏差\|")
    ax.set_ylabel("突发感知口径 \|预报偏差\|")
    ax.set_title(f"② E2：点落在对角线下 ⇒ 突发感知更准（倍数 {ratio:.2f}×）")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    for m in moves:
        kk = sorted(m["T_by_k"])
        ax.plot(kk, [m["T_by_k"][k] for k in kk], "o-",
                label=f"{m['snr_db']:g}dB b={m['bits']} p̄={m['p_bar']}")
    ax.set_xscale("log")
    ax.set_xlabel("L / L_iid")
    ax.set_ylabel("闭式选出的 T*")
    ax.set_title(f"③ E1/E3：T* 是否随突发移动（{n_moved}/{len(moves)} 组移动）")
    ax.legend(fontsize=6.5)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    aa = np.array([r["age_mean_pmf_trunc"] for r in rows])
    ea = np.array([r["model"]["age_mean"] for r in rows])
    ax.plot(aa, ea, "o", alpha=0.7)
    lim = [0, max(float(aa.max()), float(ea.max())) * 1.1]
    ax.plot(lim, lim, "k:", lw=1)
    ax.set_xlabel("闭式 E[age]（突发感知）")
    ax.set_ylabel("实测 E[age]")
    ax.set_title("④ S_a：新闭式与仿真一致")
    ax.grid(alpha=0.3)

    save_fig(fig, str(out / "23_burst_design.png"))

    payload = {
        "experiment": "X36 突发信道下的联合设计",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "K": int(K),
        "n_points": len(rows),
        "prewritten": {
            "E1": "同一 (SNR,b) 下 L 从 L_iid 拉到最大档，闭式 T* 偏移比例 ≥50%",
            "E2": "突发感知预报 |偏差| 中位 <15% 且比无记忆口径好 ≥1.5×",
            "E3": "T* 随 L 单调不增（变化的组里 ≥70% 是变小）",
            "E4": "R12：改 L 必须改变实测 NMSE",
        },
        "verdict": {"E1_frac_moved": float(frac), "E1_ok": e1_ok,
                    "E2_median": e2_med, "E2_median_memless": e2_med_mem,
                    "E2_ratio": float(ratio), "E2_ok": e2_ok,
                    "E3_frac_down": float(e3_frac), "E3_n_changed": len(changed),
                    "E3_ok": e3_ok, "E4_n_wired": int(n_wired),
                    "moves": moves},
        "rows": rows,
        "skipped": skipped,
        "caveat": ("★ 阻塞（GE）与物理层噪声丢包是**两个独立机制**：本实验把它们叠加，"
                   "但**没有**把 SNR 映射进 GE 的状态转移（即没有做『阴影衰落 ⇒ SNR 下降 "
                   "⇒ PER 上升』的联合物理模型）。⇒ 结论讲的是『结构』，"
                   "不是『某个具体场景的绝对数值』。"),
    }
    with open(out / "23_burst_design.json", "w", encoding="utf-8") as f:
        json.dump(jsonable(payload), f, ensure_ascii=False, indent=2)
    print(f"[23] 产物：{out / '23_burst_design.png'} / .json  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
