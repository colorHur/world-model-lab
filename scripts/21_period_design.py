# -*- coding: utf-8 -*-
"""★ X34「联合设计：发送周期 T × 量化精度 b —— 闭式选点 + 仿真核实」

================================================================= 为什么这是收口的那一步
X33 已经验证了一个跨层预报式：

        **E[NMSE] = Σ_h P_age(h) · g_b(h)**

其中 g_b(h) 是离线测的「从被量化的状态出发外推 h 步」的误差曲线。**只要年龄分布
P_age 有闭式，整个联合误差就可以不跑闭环仿真地算出来** —— 那就可以用来**设计**。

本实验补上最后一块：**周期发送**下的年龄分布闭式。

================================================================= ★★ 推导（写死在这里，据此自检）
每个时隙有 `d` 个符号；状态 `d` 维、每维 `b` 比特 ⇒ 一包 `d·b` 比特；
每 **T** 步才发一次 ⇒ 这一包摊到 `T·d` 个符号 ⇒ **每符号 m = b/T 比特 ⇒ M = 2^m**。

    T ↑  ⇒ m ↓ 星座稀 ⇒ 抗噪（PER 低）   但两次到达间隔拉长 ⇒ 信息更旧
    T ↓  ⇒ m ↑ 星座密 ⇒ PER 飙升         但一旦到了就是新的

--- 周期发送 + 逐包独立丢失时的年龄分布 ---
每次尝试成功概率 s = 1 − PER（逐包独立，无信道编码）。尝试发生在
t ≡ 0 (mod T)。成功之间的尝试次数 K ~ Geom(s) ⇒ **到达间隔 D = T·K**，E[D] = T/s。

平稳更新过程里 P(age = h) = P(D > h) / E[D]，而

    P(D > h) = P(K > h/T) = (1−s)^⌊h/T⌋

        ⇒ **P(age = h) = (s/T) · (1−s)^⌊h/T⌋**,   h = 0, 1, 2, …

    ★ 归一化自检：Σ_h (s/T)·q^⌊h/T⌋ = s·Σ_j q^j = s/(1−q) = 1  ✓  (q = 1−s)

    ★ 均值闭式（S_a 就是拿它跟实测 E[age] 对）：
        E[age] = T·q/s + (T−1)/2
      校验 T=1 ⇒ q/s = PER/(1−PER)（与 X26/X31 的 i.i.d. 结果一致）；
      校验 s=1 ⇒ (T−1)/2（无丢包时在 0…T−1 上均匀）✓

================================================================= ★ 预写判据
D1  |闭式预报 − 实测| 中位 < 15% ⇒ 「可以不仿真地选 T」成立
D2  闭式 argmin T*_pred 与仿真 argmin T*_sim 一致或相邻
D3  ★ T*_model ≥ T*_persistence —— 世界模型能 rollout ⇒ 应当能拉长发送周期
    （这是把 X25「H\* 不能直接当发送周期」的负面结论翻成正面设计规则的关键一步）

================================================================= 自检
S_a  闭式 E[age] 与实测 E[age] 一致（⇒ 上面的推导与 `periodic_pmf` 都对）
S_b  pmf 归一化；截断质量 < 1%
S_c  全部有限（R14）
S_d  ★ R12：改周期必须改变结果（⇒ 周期性真的接进了 schedule，不是空跑）
S_e  只看 budget：T 必须 ≤ 铺设的最大视界 K（否则分布被截断污染）
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
from wmlab.eval.physical import (UniformQuantizer, fit_quantizer_range, overload_fraction,
                                 per_from_ber, qam_approximation_valid, qam_bit_error_rate,
                                 qam_symbol_error_rate, snr_db_to_linear)
from wmlab.eval.tracking import (periodic_age_pmf, periodic_age_tail,
                                 periodic_lossy_schedule,
                                 periodic_mean_age, run_tracking)
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


# ---------------------------------------------------------------- 闭式
periodic_pmf = periodic_age_pmf          # 库里的实现即本脚本用的口径（唯一一份）
def parse_args():
    p = argparse.ArgumentParser(description="X34 联合设计：周期 T × 精度 b")
    p.add_argument("--config", default="configs/uav_period.yaml")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--tag", default="21_period_design")
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
        dz["n_track_episodes"] = 8
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
    max_T = int(dz.get("max_period", 64))
    n_track = int(dz["n_track_episodes"])

    print("[21] " + "=" * 76)
    print("[21] ★ X34 联合设计：发送周期 T × 量化精度 b")
    print(f"[21]   SNR ∈ {snrs} dB   b ∈ {bits_list}   可行 m（方形星座）∈ {m_list}")
    print("[21]   预写判据：D1 |偏差|中位 <15%   D2 T*_pred ≈ T*_sim   "
          "D3 T*_model ≥ T*_persist")

    # ---------- 1) 训练（复用 X30/X31/X32/X33 的同一套流程）----------
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
    print(f"[21] 训练完成：params={count_params(model):,} val={hist['val_total'][-1]:.6f} "
          f"({time.time() - t0:.0f}s)")

    heads = {"model": model, "persistence": PersistenceHead(model)}

    # ---------- 2) 数据与量化器 ----------
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

    # ---------- 3) 离线曲线 g_b(h)（两个预测器各一份）----------
    hs = [int(h) for h in cfg["eval"]["horizons"]]
    hs = [h for h in hs if h <= max(e["length"] for e in eval_eps) - WARMUP - 2]

    def _offline(head, payload=None) -> np.ndarray:
        c = closed_loop_error_curve(head, eval_eps, hs, device,
                                    n_samples=int(cfg["eval"]["n_samples"]),
                                    seed=seed, t0_min=WARMUP, payload_fn=payload)
        return np.concatenate([[0.0], np.asarray(c["mse_per_step"], dtype=float) / var_g])

    K = None
    f_curves, g_curves = {}, {}
    for name, head in heads.items():
        f_curves[name] = _offline(head, None)
        K = len(f_curves[name]) - 1 if K is None else K
        if len(f_curves[name]) != K + 1:
            raise AssertionError(f"★ S_off 未通过：{name} 的曲线长度不一致")
        for b in bits_list:
            gb = _offline(head, quantizers[b])
            gb[0] = float(quantizers[b].measure_mse(allobs) / var_g)
            if not np.all(np.isfinite(gb)):
                raise FloatingPointError(f"g_{name,b}(h) 含非有限值（R14）")
            g_curves[(name, b)] = gb
        print(f"[21]   [{name:>11s}] f(1)={f_curves[name][1]:.5f} "
              f"f(10)={f_curves[name][min(10, K)]:.5f} f(K={K})={f_curves[name][K]:.5f}")
    h_err = reliable_horizon(hs, [float(x) for x in f_curves["model"][hs]],
                             float(cfg["eval"]["err_threshold"]),
                             cfg["eval"].get("threshold_mode", "rel"))
    print(f"[21] 离线：K={K} 步   H*_err={h_err}")

    def per_of(snr_db: float, b: int, m: float) -> float:
        g = snr_db_to_linear(snr_db)
        M = 2.0 ** float(m)
        if not qam_approximation_valid(g, M):
            return 1.0 - 1e-6
        return float(per_from_ber(qam_bit_error_rate(g, M), float(obs_dim * b)))

    # ---------- 4) 构造设计网格：(b, m) ⇒ T = b/m（必须整除）----------
    grid = []
    skipped = []
    for b in bits_list:
        for m in m_list:
            if b % m != 0:
                skipped.append({"bits": b, "m": m, "reason": f"T = b/m = {b}/{m} 非整数"})
                continue
            T = b // m
            if T < 1 or T > max_T:
                skipped.append({"bits": b, "m": m, "T": T, "reason": "周期超出范围"})
                continue
            if T > K:
                skipped.append({"bits": b, "m": m, "T": T,
                                "reason": f"T={T} > 可用视界 K={K}（分布会被截断污染）"})
                continue
            grid.append({"bits": b, "m": m, "T": T})
    print(f"[21] 设计网格：{len(grid)} 个 (b,T) 组合，跳过 {len(skipped)} 个")
    for g in grid:
        print(f"[21]   b={g['bits']:>2d} m={g['m']} ⇒ T={g['T']:>2d} 步 "
              f"（M={2.0 ** g['m']:>4.0f}-QAM）")

    # ---------- 5) 扫：闭式预报 + 仿真实测 ----------
    print("[21] " + "-" * 76)
    rows = []
    for snr in snrs:
        for gd in grid:
            b, m, T = gd["bits"], gd["m"], gd["T"]
            per01 = per_of(snr, b, m)
            if per01 >= 0.999:
                skipped.append({**gd, "snr_db": snr, "reason": "PER≈1（链路不可用）"})
                continue
            pmf = periodic_pmf(per01, T, K)
            if abs(float(pmf.sum()) - 1.0) > 1e-9:
                raise AssertionError(f"★ S_b 未通过：pmf 未归一化 {pmf.sum()}")
            tail_mass = float(periodic_age_tail(per01, T, K))
            # ★ 截断判据：年龄分布的尾巴必须落在可用视界内，否则"平稳分布"根本没形成。
            #   （第一版没排除：PER=0.987 / T=3 ⇒ 解析 E[age]=237 而 K=190，
            #     45% 质量被压到边界，实测 209 vs 解析 237 —— 那是截断，不是推导错。）
            #
            # ★★ 第二版修 bug（本仓库第 16 次自证伪，R12）：
            #   第一版写成 `1 − (1−per)^{K/T}` —— 那是「至少一次失败」的概率，
            #   而这里要的是「**全部**失败」的概率 per^{K/T}。两者都随丢包率增大，
            #   但量级差约 160 个数量级（per=0.019/T=2/K=190：错式 83.8%，正确 ≈1e−160）
            #   ⇒ 第一版把 PER>0 的点**全部**判成截断丢掉，网格里只剩 PER≈0 的退化点，
            #     D1/D2/D3 是在那个退化区拿到的，不能当作"联合设计"的结论。
            #   ⇒ 现在由库函数 `periodic_age_tail` 给出，并在自检 ㉛ 里与暴力求和逐点对齐、
            #     同时钉死方向（随丢包率单调增）与量级（0.019/2/190 必须 < 1e−100）。
            if tail_mass > 0.02:
                skipped.append({**gd, "snr_db": snr, "per": per01, "tail_mass": tail_mass,
                                "reason": (f"年龄分布有 {tail_mass:.1%} 质量超出可用视界 "
                                           f"K={K}（链路太差/周期太长）⇒ 不进设计网格")})
                print(f"[21]   跳过 SNR={snr:g}dB b={b} T={T}：尾巴 {tail_mass:.1%} > 2% "
                      f"（PER={per01:.3f}）")
                continue
            rec = {"snr_db": float(snr), "bits": b, "m": m, "T": T,
                   "M": float(2.0 ** m), "per": float(per01),
                   "tail_mass": tail_mass,
                   "age_mean_analytic": float(periodic_mean_age(per01, T)),
                   # ★ 与"仿真能实现的"比：截断后的 pmf 均值（不是无界解析均值）
                   "age_mean_pmf_trunc": float(np.dot(np.arange(K + 1), pmf))}
            for name, head in heads.items():
                curve_g = g_curves[(name, b)]
                pred = float(np.dot(pmf, curve_g))
                r = run_tracking(head, track_eps, device,
                                 periodic_lossy_schedule(T, per01),
                                 seed=track_seed, denom=var_g, warmup=WARMUP,
                                 label=f"{name}/T={T}", payload_fn=quantizers[b])
                if not np.isfinite(r.nmse):
                    raise FloatingPointError(
                        f"S_c：(snr={snr},b={b},T={T},{name}) 非有限值（R14）")
                emp_age = float(r.age_tx_mean)
                # ★ S_a：闭式的均值必须等于实测 —— 与**截断后的** pmf 均值比
                if abs(emp_age - rec["age_mean_pmf_trunc"]) > max(
                        0.05, 0.05 * max(abs(emp_age), abs(rec["age_mean_pmf_trunc"]))):
                    raise AssertionError(
                        f"★ S_a 未通过：SNR={snr} T={T} per={per01:.4f} "
                        f"实测 E[age]={emp_age:.4f} vs 闭式（截断后）"
                        f"{rec['age_mean_pmf_trunc']:.4f} ⇒ 「周期发送的年龄分布」"
                        f"推导与仿真不一致，闭式不可用")
                rec[name] = {"nmse": float(r.nmse), "pred": pred,
                             "gap": float(pred / max(r.nmse, 1e-18) - 1.0),
                             "age_mean": emp_age}
            rows.append(rec)
            rm, rp = rec["model"], rec["persistence"]
            print(f"[21]   {snr:>5g}dB b={b:>2d} T={T:>2d} m={m} PER={per01:.4f} "
                  f"E[age]={rm['age_mean']:6.2f}(闭式 {rec['age_mean_pmf_trunc']:6.2f}) | "
                  f"model 实测={rm['nmse']:.5f} 预报={rm['pred']:.5f}"
                  f"({rm['gap']:+.1%}) | persist 实测={rp['nmse']:.5f}"
                  f"({rp['gap']:+.1%})")

    if not rows:
        raise AssertionError("★ 设计网格为空（PER 全饱和或 T 全超视界）⇒ 调整 SNR/b/m")

    # ★ S_d（R12）：改周期必须改变结果
    n_distinct = 0
    for snr in snrs:
        for b in bits_list:
            sub = [r for r in rows if abs(r["snr_db"] - snr) < 1e-9 and r["bits"] == b]
            if len(sub) < 2:
                continue
            vals = [r["model"]["nmse"] for r in sub]
            if max(vals) / max(min(vals), 1e-18) > 1.05:
                n_distinct += 1
    if n_distinct == 0:
        raise AssertionError(
            "★ S_d 未通过：所有 SNR 下改周期**都不改变结果** ⇒ 周期性没接进 "
            "schedule（R12 违规），整个实验在空跑")
    print(f"[21]   S_a/S_b/S_c/S_d 通过（{n_distinct} 个 (SNR,b) 组验到周期真的起作用）")

    # ---------- 6) 判据 ----------
    print("[21] " + "=" * 76)

    def _arr(key, name="model"):
        return np.array([r[name][key] for r in rows], dtype=float)

    gap = np.abs(_arr("gap"))
    d1_med, d1_p90 = float(np.median(gap)), float(np.percentile(gap, 90))
    if d1_med < 0.15:
        verdict_d1 = (f"D1 ★成立：闭式预报 |偏差| 中位 {d1_med:.1%}、90 分位 {d1_p90:.1%} "
                      f"⇒ **设计曲线可以不跑闭环仿真地算出来**")
    elif d1_med < 0.35:
        verdict_d1 = f"D2 仅定性可用：|偏差| 中位 {d1_med:.1%}、90 分位 {d1_p90:.1%}"
    else:
        verdict_d1 = (f"D1 ★不成立：|偏差| 中位 {d1_med:.1%} ⇒ 闭式只能定性，"
                      f"选工作点必须仿真")
    print(f"[21] ★ {verdict_d1}")

    # T* 对比
    tstar = []
    for snr in snrs:
        for b in bits_list:
            sub = sorted([r for r in rows if abs(r["snr_db"] - snr) < 1e-9
                          and r["bits"] == b], key=lambda r: r["T"])
            if len(sub) < 2:
                continue
            for name in ("model", "persistence"):
                sim = [r[name]["nmse"] for r in sub]
                prd = [r[name]["pred"] for r in sub]
                ks, kp = int(np.argmin(sim)), int(np.argmin(prd))
                tstar.append({"snr_db": snr, "bits": b, "head": name,
                              "T_sim": sub[ks]["T"], "T_pred": sub[kp]["T"],
                              "nmse_at_sim": float(sim[ks]), "pred_at_pred": float(prd[kp]),
                              "n_options": len(sub),
                              "all_T": [r["T"] for r in sub],
                              "nmse_curve": [float(x) for x in sim],
                              "pred_curve": [float(x) for x in prd]})
    match = sum(1 for t in tstar if t["T_sim"] == t["T_pred"])
    adja = sum(1 for t in tstar
               if abs(t["all_T"].index(t["T_sim"]) - t["all_T"].index(t["T_pred"])) <= 1)
    if tstar:
        print(f"[21] ★ D2：闭式选出的 T* 与仿真一致 {match}/{len(tstar)}，"
              f"相邻（±1 档）{adja}/{len(tstar)}")
        for t in tstar:
            if t["head"] == "model":
                print(f"[21]     SNR={t['snr_db']:>5g}dB b={t['bits']:>2d}  "
                      f"候选 T={t['all_T']}  仿真最优 T*={t['T_sim']}  "
                      f"闭式最优 T*={t['T_pred']}")
    verdict_d2 = (f"D2 一致 {match}/{len(tstar)}；相邻 {adja}/{len(tstar)}"
                  if tstar else "D2 无足够候选（网格太粗）")

    tm = [t["T_sim"] for t in tstar if t["head"] == "model"]
    tp = [t["T_sim"] for t in tstar if t["head"] == "persistence"]
    if tm and tp:
        frac = float(np.mean(np.asarray(tm) >= np.asarray(tp)))
        if frac >= 0.75:
            verdict_d3 = (f"D3 ★成立：{frac:.0%} 的配置里 T*_model ≥ T*_persistence"
                          f"（中位 {np.median(tm):g} vs {np.median(tp):g} 步）"
                          f"⇒ 世界模型确实换来了更长的发送周期")
        elif frac <= 0.25:
            verdict_d3 = (f"D3 ★反向：只有 {frac:.0%} 的配置满足 T*_model ≥ T*_persist"
                          f"（中位 {np.median(tm):g} vs {np.median(tp):g}）")
        else:
            verdict_d3 = f"D3 混合：{frac:.0%} 的配置满足（无稳定结论）"
        print(f"[21] ★ {verdict_d3}")
    else:
        verdict_d3 = "D3 无法比较（缺候选）"

    print(f"[21]   参照：离线 H*_err={h_err}（T* 与之相比是同一量级？见 ③ 图）")

    # ---------- 7) 图 ----------
    fig, axes = plt.subplots(2, 3, figsize=(17.5, 9.5))
    fig.suptitle(f"X34 联合设计：周期 T × 精度 b  ·  {verdict_d1}\n"
                 f"{verdict_d2}\n{verdict_d3}", fontsize=10.5)

    # ① 预报 vs 实测
    ax = axes[0, 0]
    for name, col, mk in (("model", PALETTE["blue"], "o"),
                          ("persistence", PALETTE["orange"], "s")):
        ax.plot([r[name]["pred"] for r in rows], [r[name]["nmse"] for r in rows],
                mk, color=col, alpha=0.8, label=name)
    vmin = max(min(min(r[n]["nmse"] for r in rows) for n in ("model", "persistence")), 1e-7)
    vmax = max(max(r["model"]["nmse"] for r in rows) * 1.5, vmin * 10)
    ax.plot([vmin, vmax], [vmin, vmax], "k:", lw=1, label="y=x")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("闭式预报 Σ_h P_age(h)·g_b(h)")
    ax.set_ylabel("实测 E[NMSE]")
    ax.set_title(f"① 闭式预报精度（中位 {d1_med:.1%}）")
    ax.legend(fontsize=8)

    # ② 设计曲线：每个 SNR 一行，误差 vs T（★ 是否 U 形）
    ax = axes[0, 1]
    cmap = plt.get_cmap("viridis")
    for i, snr in enumerate(snrs):
        col = cmap(i / max(len(snrs) - 1, 1))
        for b in bits_list:
            sub = sorted([r for r in rows if abs(r["snr_db"] - snr) < 1e-9
                          and r["bits"] == b], key=lambda r: r["T"])
            if len(sub) < 2:
                continue
            ax.plot([r["T"] for r in sub],
                    [r["model"]["nmse"] for r in sub], "o-", color=col, ms=4,
                    alpha=0.85, label=f"{snr:g}dB" if b == bits_list[0] else None)
            ax.plot([r["T"] for r in sub],
                    [r["model"]["pred"] for r in sub], ":", color=col, lw=1, alpha=0.8)
    ax.set_yscale("log"); ax.set_xscale("log")
    ax.set_xlabel("发送周期 T（步）"); ax.set_ylabel("E[NMSE]（实线=实测，虚线=闭式）")
    ax.set_title("② 设计曲线：U 形最优是否存在")
    ax.legend(fontsize=7.5, ncol=2)

    # ③ T* 与 H* 的关系
    ax = axes[0, 2]
    for snr in snrs:
        for b in bits_list:
            sub = [r for r in rows if abs(r["snr_db"] - snr) < 1e-9 and r["bits"] == b]
            if not sub:
                continue
            for name, col in (("model", PALETTE["blue"]),
                              ("persistence", PALETTE["orange"])):
                vals = [r[name]["nmse"] for r in sub]
                k = int(np.argmin(vals))
                ax.plot(snr, sub[k]["T"], "o" if name == "model" else "s", color=col,
                        ms=6, alpha=0.8,
                        label=name if (snr == snrs[0] and b == bits_list[0]) else None)
    if h_err is not None:
        ax.axhline(float(h_err), color=PALETTE["red"], ls="--", lw=1.5,
                   label=f"H*_err={float(h_err):.1f} 步")
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("最优发送周期 T*（步）")
    ax.set_title("③ ★ T* vs SNR（与离线 H*_err 对照）")
    ax.legend(fontsize=8)

    # ④ age 闭式核对
    ax = axes[1, 0]
    aa = np.array([r["age_mean_pmf_trunc"] for r in rows])
    ea = np.array([r["model"]["age_mean"] for r in rows])
    ax.plot(aa, ea, "o", color=PALETTE["green"])
    lim = [0, max(aa.max(), ea.max()) * 1.1]
    ax.plot(lim, lim, "k:", lw=1)
    ax.set_xlabel("闭式 E[age]（截断后）= Σ_h h·P(h)")
    ax.set_ylabel("实测 E[age]")
    ax.set_title("④ S_a：周期发送的年龄闭式核对")
    ax.grid(alpha=0.3)

    # ⑤ 量化精度：同等周期下 b 的效应
    ax = axes[1, 1]
    for i, snr in enumerate(snrs[:2]):          # 只画前两个 SNR，否则轴标签挤成一团
        col = cmap(i / max(len(snrs) - 1, 1))
        sub = sorted([r for r in rows if abs(r["snr_db"] - snr) < 1e-9],
                     key=lambda r: (r["bits"], r["T"]))
        if sub:
            xs2 = [f"b{r['bits']}\nT{r['T']}" for r in sub]
            ax.plot(xs2, [r["model"]["nmse"] for r in sub], "o-", color=col, ms=4,
                    label=f"{snr:g}dB")
    ax.set_yscale("log"); ax.tick_params(axis="x", labelsize=6, rotation=90)
    ax.set_ylabel("E[NMSE]（model）")
    ax.set_title("⑤ 精度 vs 周期：全部组合一览")
    ax.legend(fontsize=7.5, ncol=2)

    # ⑥ 两个预测器的最优差
    ax = axes[1, 2]
    xs3 = np.arange(len(tstar) // 2)
    for name, col in (("model", PALETTE["blue"]), ("persistence", PALETTE["orange"])):
        vals = [t["T_sim"] for t in tstar if t["head"] == name]
        if vals:
            ax.plot(xs3[:len(vals)], vals, "o-", color=col, ms=5, label=f"T*_{name}")
    ax.set_xlabel("配置编号（SNR × b）"); ax.set_ylabel("最优周期 T*（步）")
    ax.set_title("⑥ ★ 世界模型是否换来更长的发送周期")
    ax.legend(fontsize=8)

    save_fig(fig, os.path.join(out, args.tag + ".png"))
    plt.close(fig)

    payload = {
        "experiment": "X34 联合设计：发送周期 T × 量化精度 b（闭式选点 + 仿真核实）",
        "config": {k: cfg[k] for k in ("env", "controller", "design", "eval", "task")},
        "obs_dim": obs_dim, "var_g": var_g, "max_age_K": int(K), "h_err": h_err,
        "rows": rows, "skipped": skipped, "tstar": tstar,
        "prediction_error": {"median": d1_med, "p90": d1_p90},
        "verdict": {"D1": verdict_d1, "D2": verdict_d2, "D3": verdict_d3},
        "caveat": ("★ 本实验只用**逐包独立**丢失（周期 + 伯努利），不含突发信道；"
                   "突发版本需要另一条年龄闭式，列为后续。"
                   "★ 可行调制限制为方形星座（m 偶数）且 T=b/m 必须整除 ⇒ T 的可选集很粗，"
                   "T* 只能在候选集里取 argmin，不得当作连续最优点引用。"
                   "★ BER/PER 闭式仍是教科书近似（physical.py 的「待核」声明）。"),
    }
    with open(os.path.join(out, args.tag + ".json"), "w", encoding="utf-8") as fp:
        json.dump(jsonable(payload), fp, ensure_ascii=False, indent=2)
    print(f"[21] 产物：{os.path.join(out, args.tag)}.png / .json  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
