# -*- coding: utf-8 -*-
"""★ X31-b「多丢包率扫描」：验证突发信道的误差是否可分解为 p̄ × G(L)

================================================================= 为什么做这个
X31（scripts/17）只在 **p̄ = 0.5** 一个点上验过闭式恒等式
``E[NMSE] = p̄ · [ f(L) + J(L) ]``。

★ 单点值不可引用 —— 这是 X3/X4 用种子方差换来的教训（H\* 随种子抖到不可引）。
本脚本把 p̄ 铺开成网格，看那个恒等式是**结构性的**还是**恰好在 0.5 处成立**。

================================================================= ★★ 待验的强预言
Gilbert–Elliott 下 age 的精确分布是

    P(age = 0) = 1 − p̄
    P(age = k) = p̄ · β · (1−β)^(k−1),  k ≥ 1,  β = 1/L

**条件**年龄分布（k≥1 部分）恒为 Geom(β)，**与 p̄ 无关**；p̄ 只决定 age=0 那一份权重。
又 f(0) = 0（刚收到真值，估计无误差），于是

    E[NMSE] = Σ_k P(age=k)·f(k) = p̄ · Σ_{k≥1} β(1−β)^(k−1)·f(k) = **p̄ · G(L)**

★ 两个可直接证伪的推论（本脚本的主检验）：

  **C1 数据塌缩**  固定 L，``E[NMSE] / p̄ = G(L)`` 与 p̄ 无关
                   ⇒ 把不同 p̄ 的 ``E[NMSE]/p̄ vs L`` 画在一起，**曲线必须重合**
  **C2 线性于 p̄**  固定 L，``E[NMSE] ∝ p̄``（过原点直线）

若成立 ⇒「丢包率」与「突发结构」是**解耦的两个因子**：
丢包率由链路预算/功率决定，突发长度由信道时变性/移动速度决定，
跟踪误差可以分开设计。**这就是本实验对通信侧的意义所在。**

若 C1/C2 被证伪 ⇒ 说明 f 之外还有 p̄ 依赖的机制（例如连续丢包之间的
**误差 compound**：上次没纠上的误差会被带进下一次），那本身就是更有价值的发现。

================================================================= 控制变量
★ **模型只训练一次，所有 (p̄, L) 共享同一个模型。**
  若每个 p̄ 各训一次，"模型质量"会混进 p̄ 的效应里 ⇒ C1 的塌缩检验直接失效。
  同理离线曲线 f(h) 也只算一次。

================================================================= 网格可行性
α = p̄/(L(1−p̄)) ≤ 1 ⇒ **L ≥ p̄/(1−p̄)**。p̄=0.8 时 L≥4（L=1,2 无解）。
公共可行域取 L ∈ {2,4,8,16,32}，p̄ ∈ {0.2,0.35,0.5,0.65}。

用法：
    python scripts/18_burst_sweep.py --config configs/uav_burst_sweep.yaml
    python scripts/18_burst_sweep.py --quick          # 冒烟
"""
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
from wmlab.eval.tracking import (GilbertElliottChannel, lossy_schedule, run_tracking,
                                 run_length_goodness, simulate_bad_runs)
from wmlab.models import MLPWorldModel
from wmlab.rollout import closed_loop_error_curve
from wmlab.train import train_world_model
from wmlab.utils import (count_params, get_device, load_config, output_dir, set_seed)
from wmlab.utils.plot import PALETTE, apply_style, save_fig

def _load_mod17():
    """从 scripts/17_burst_channel.py 加载**已验证过的** pmf 函数。

    ★ 为什么用 importlib 而不是复制一份：口径必须**逐字一致**，
      否则本脚本的 G(L) 与 X31 的不可比，塌缩检验就失去对照。
      17 有 `if __name__ == \"__main__\"` 保护 ⇒ 加载不会跑实验。
    """
    import importlib.util
    path = os.path.join(_HERE, "17_burst_channel.py")
    spec = importlib.util.spec_from_file_location("_mod17", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_mod17 = _load_mod17()
full_age_pmf = _mod17.full_age_pmf


def parse_args():
    p = argparse.ArgumentParser(description="X31-b 多丢包率扫描（p̄ × L 网格）")
    p.add_argument("--config", default="configs/uav_burst_sweep.yaml")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--tag", default="18_burst_sweep")
    p.add_argument("--ps", default=None, help="逗号分隔的丢包率网格，覆盖 config")
    p.add_argument("--bursts", default=None, help="逗号分隔的突发长度网格，覆盖 config")
    p.add_argument("--seeds", type=int, default=None,
                   help="★ 每个 (p̄,L) 重复几个随机种子（用于把噪声和真偏离分开）")
    p.add_argument("--quick", action="store_true", help="冒烟：少数据/少集/小网格")
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
    if args.ps:
        cfg["channel"]["p_loss_list"] = [float(x) for x in args.ps.split(",")]
    if args.bursts:
        cfg["channel"]["burst_lens"] = [float(x) for x in args.bursts.split(",")]
    if args.quick:
        cfg["train"]["epochs"] = min(int(cfg["train"]["epochs"]), 3)
        cfg["data"]["n_episodes"] = min(int(cfg["data"]["n_episodes"]), 20)
        cfg["channel"]["p_loss_list"] = [0.2, 0.5]
        cfg["channel"]["burst_lens"] = [2.0, 8.0]
        cfg["channel"]["n_track_episodes"] = 6
        cfg["eval"]["n_samples"] = 64

    seed = int(cfg["seed"])
    set_seed(seed)
    apply_style()
    device = get_device(args.device or cfg.get("device"))
    out = output_dir(cfg)
    t0 = time.time()

    ch_cfg = cfg["channel"]
    p_list = [float(x) for x in ch_cfg["p_loss_list"]]
    bursts = [float(x) for x in ch_cfg["burst_lens"]]
    # ★ 种子重复：没有它就无法区分「真偏离」与「采样噪声」
    #   （第一版单种子跑出 C1 最大 CV 16.7%，判"弱成立"，但其实无从判断
    #    这 16.7% 是机制还是噪声 —— 单个 G 只由 ~590 个游程估计出来。）
    n_seeds = int(args.seeds) if args.seeds else int(ch_cfg.get("n_seeds", 1))
    n_seeds = max(1, n_seeds)
    if args.quick:
        n_seeds = max(1, min(n_seeds, 2))

    print(f"[18] ★ X31-b 多丢包率扫描：p̄ ∈ {p_list}  L ∈ {bursts}")
    print(f"[18]   待验：C1 数据塌缩（E[NMSE]/p̄ = G(L) 与 p̄ 无关）"
          f" / C2 线性于 p̄")

    # ---------- 1) 信道构造 + 可行性 ----------
    # ★ L ≥ p̄/(1−p̄) 才有解；不可行的组合**跳过并记账**，不许静默丢
    grid = {}          # (p̄, L) -> channel
    infeasible = []
    for pl in p_list:
        l_min = pl / (1.0 - pl)
        for L in bursts:
            if L < l_min - 1e-12:
                infeasible.append((pl, L, l_min))
                continue
            ch = GilbertElliottChannel(pl, L, seed=seed + 7)
            assert abs(ch.pi_bad - pl) < 1e-12, f"S1 失败 π_B={ch.pi_bad} ≠ {pl}"
            assert abs(ch.mean_burst_len - L) < 1e-12, f"S1 失败 E[L]={ch.mean_burst_len}"
            assert abs(ch.rho1 - (1 - ch.alpha - ch.beta)) < 1e-15, "S1 失败 ρ₁"
            grid[(pl, L)] = ch
    if infeasible:
        print(f"[18]   ⚠ {len(infeasible)} 个组合超出可行域 L ≥ p̄/(1−p̄)，已跳过：")
        for pl, L, lm in infeasible:
            print(f"[18]     p̄={pl:g} L={L:g}（需 L ≥ {lm:.3f}）")
    print(f"[18]   可行网格 {len(grid)} / {len(p_list) * len(bursts)} 个点")

    # ---------- 2) 环境 + 控制器 + 训练（★ 一次，全网格共享）----------
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
    print(f"[18] 训练完成：params={count_params(model):,} val_loss={hist['val_total'][-1]:.6f} "
          f"({time.time() - t0:.0f}s)")

    # ---------- 3) 离线曲线 f(h)（★ 一次，全网格共享）----------
    WARMUP = int(cfg["task"]["warmup_steps"])
    ev_seed = seed + int(cfg["data"]["eval_seed_offset"])
    ev_env = make_env(cfg["env"]["id"], seed=ev_seed,
                      noise_std=float(cfg["env"]["noise_std"]),
                      max_steps=int(cfg["env"]["max_steps"]))
    n_ev = max(30, n_ep // 3)
    eval_eps = collect_controlled_episodes(ev_env, ctrl, n_episodes=n_ev, seed=ev_seed,
                                           max_steps=int(cfg["env"]["max_steps"]))
    ev_env.close()
    allobs = np.concatenate([e["obs"][WARMUP:] for e in eval_eps], axis=0).astype(np.float64)
    var_g = float(allobs.var())
    hs = [int(h) for h in cfg["eval"]["horizons"]]
    hs = [h for h in hs if h <= max(e["length"] for e in eval_eps) - WARMUP - 2]
    curve = closed_loop_error_curve(model, eval_eps, hs, device,
                                    n_samples=int(cfg["eval"]["n_samples"]),
                                    seed=seed, t0_min=WARMUP)
    thr = float(cfg["eval"]["err_threshold"])
    f_h_rel = np.asarray([float(np.asarray(curve["nmse_per_step"], dtype=float)[h - 1])
                          for h in hs])
    f_h = np.asarray([float(np.asarray(curve["mse_per_step"], dtype=float)[h - 1])
                      / var_g for h in hs])
    h_err = reliable_horizon(hs, [float(x) for x in f_h_rel], thr,
                             cfg["eval"].get("threshold_mode", "rel"))
    if not np.all(np.isfinite(f_h)):
        raise FloatingPointError("离线曲线含非有限值（R14）")
    print(f"[18] 离线曲线：H*_err={h_err}  f(1)={f_h[0]:.5f}  f({hs[-1]})={f_h[-1]:.5f}"
          f"（全局分母口径，与在线可比）")

    hs_arr = np.asarray(hs, dtype=float)
    f_arr = np.asarray(f_h, dtype=float)

    def f_at(h: float) -> float:
        if h <= 0:
            return 0.0
        return float(np.exp(np.interp(np.log(h), np.log(hs_arr),
                                      np.log(np.maximum(f_arr, 1e-12)))))

    K = int(hs[-1])
    f_grid = np.asarray([f_at(k) for k in range(K + 1)], dtype=np.float64)

    # ---------- 4) 扫网格 ----------
    n_track = int(ch_cfg["n_track_episodes"])
    track_seed = seed + int(cfg["data"]["eval_seed_offset"])
    track_eps = eval_eps[:n_track] if n_track <= len(eval_eps) else eval_eps
    rows = []
    baselines = {}          # p̄ -> i.i.d. 基线 nmse

    for pl in p_list:
        base = run_tracking(model, track_eps, device, lossy_schedule(pl),
                            seed=track_seed, denom=var_g,
                            label=f"i.i.d.(p={pl})", warmup=WARMUP)
        baselines[pl] = float(base.nmse)
        rows.append({"kind": "iid", "p": pl, "L": 1.0 / (1.0 - pl),
                     "nmse": float(base.nmse), "age_mean": float(base.age_tx_mean),
                     "tx_rate": float(base.tx_rate)})
        print(f"[18] p̄={pl:g}  i.i.d. 基线 E[NMSE]={base.nmse:.5f} "
              f"E[age]={base.age_tx_mean:.3f}（解析 {pl / (1 - pl):.3f}）")

        for L in bursts:
            ch = grid.get((pl, L))
            if ch is None:
                continue
            r = run_tracking(model, track_eps, device, ch, seed=track_seed,
                             denom=var_g, label=f"GE(p̄={pl:g},L={L:g})", warmup=WARMUP)
            if not np.isfinite(r.nmse) or not np.isfinite(r.age_tx_mean):
                raise FloatingPointError(f"GE(p̄={pl:g},L={L:g}) 产出非有限值（R14）")
            hist_ = {int(k): int(v) for k, v in r.age_hist.items()}
            n_tot = sum(hist_.values())

            # 闭式精确预测（条件年龄恒为 Geom(β)）
            pmf_th = full_age_pmf(pl, ch.beta, K)
            pred_th = float(np.dot(pmf_th, f_grid))
            # 实测直方图预测（不假设分布形状）
            p_age_emp = np.zeros_like(f_grid)
            for k, c in hist_.items():
                p_age_emp[min(int(k), K)] += c / n_tot
            pred_emp = float(np.dot(p_age_emp, f_grid))
            # 一阶 / Jensen 分解
            f_of_L = f_at(L)
            g_L = pred_th / max(pl, 1e-12)
            j_L = g_L - f_of_L

            # ★ S2：经验丢包率 vs p̄（σ 按有效样本量 = 游程数）
            n_runs = max(1.0, n_tot * pl / max(L, 1.0))
            sigma = float(np.sqrt(max(pl * (1 - pl), 1e-12) / n_runs))
            emp_p = 1.0 - hist_.get(0, 0) / max(n_tot, 1)
            if abs(emp_p - pl) > 4.0 * sigma:
                raise AssertionError(
                    f"★ S2 未通过：p̄={pl:g} L={L:g} 实测丢包率 {emp_p:.4f} vs {pl} "
                    f"偏差 {abs(emp_p - pl) / max(sigma, 1e-12):.1f}σ")
            # ★ S6：实测条件年龄分布 vs Geom(β)
            n_pos = sum(c for k, c in hist_.items() if k >= 1)
            cond_emp = np.array([0.0] + [hist_.get(k, 0) / max(n_pos, 1)
                                         for k in range(1, K + 1)])
            cond_th = np.array([0.0] + [ch.beta * ((1.0 - ch.beta) ** (k - 1))
                                        for k in range(1, K + 1)])
            # ★ S6 硬判定用 KS（TV 的噪声随突发长度发散 ⇒ 不能当阈值，
            #   见 17 里 `age_goodness` 的第 14 次自我修正）
            # ★ 有效样本量 = 游程数（游程内 age = 1…n 是确定性序列，只有 1 个独立样本）
            n_eff = max(int(n_pos / max(L, 1.0)), 1)
            go = _mod17.age_goodness(cond_emp, cond_th, n_eff)
            tv = go["tv"]
            if not go["ok"]:
                raise AssertionError(
                    f"★ S6 未通过：p̄={pl:g} L={L:g} 条件年龄分布 KS={go['ks']:.4f} > "
                    f"临界 {go['ks_crit']:.4f}（n_eff={n_eff}；TV={tv:.4f} vs 噪声期望 "
                    f"{go['tv_noise']:.4f}）")
            # ★ S7：纯信道游程长度分布（i.i.d. 样本 ⇒ 严格硬判定）
            runs = simulate_bad_runs(ch, 200000, np.random.default_rng(seed + 31))
            rl = run_length_goodness(runs, ch.beta, alpha=0.05 / max(len(grid), 1))
            if rl["ok"] is False:
                raise AssertionError(
                    f"★ S7 未通过：p̄={pl:g} L={L:g} 游程长度 KS={rl['ks']:.4f} > "
                    f"{rl['ks_crit']:.4f} 或均值 {rl['mean_emp']:.3f} vs 闭式 "
                    f"{rl['mean_th']:.3f} (z={rl['mean_z']:+.1f})")
            # ★ S4：闭式分布均值 vs 实测 E[age]（按有效样本量给容差）
            sigma_age = float(r.age_std / np.sqrt(max(n_runs, 1.0)))
            m_th = float(np.dot(np.arange(len(pmf_th)), pmf_th))
            if abs(m_th - r.age_tx_mean) > max(4.0 * sigma_age, 0.05 * m_th):
                raise AssertionError(
                    f"★ S4 未通过：p̄={pl:g} L={L:g} 闭式均值 {m_th:.4f} vs 实测 "
                    f"{r.age_tx_mean:.4f} 差 {abs(m_th - r.age_tx_mean) / max(sigma_age, 1e-12):.1f}σ")

            # ★ 种子重复（自检已在 si=0 做过一次即可 —— 它验的是**信道实现**，
            #   与评估种子无关；重复的种子只用来估计 G 的采样噪声）
            nmse_s = [float(r.nmse)]
            for si in range(1, n_seeds):
                r2 = run_tracking(model, track_eps, device, ch,
                                  seed=track_seed + 1000 * si, denom=var_g,
                                  label=f"GE(p̄={pl:g},L={L:g})#s{si}", warmup=WARMUP)
                if not np.isfinite(r2.nmse):
                    raise FloatingPointError(f"GE(p̄={pl:g},L={L:g}) 种子 {si} 非有限值（R14）")
                nmse_s.append(float(r2.nmse))
            nmse_mean = float(np.mean(nmse_s))
            nmse_se = (float(np.std(nmse_s, ddof=1) / np.sqrt(len(nmse_s)))
                       if len(nmse_s) > 1 else 0.0)

            g_meas = nmse_mean / max(pl, 1e-12)       # ★ 实测的 G(L)
            rows.append({"kind": "ge", "p": pl, "L": float(L), "nmse": nmse_mean,
                         "nmse_se": nmse_se, "nmse_samples": [float(x) for x in nmse_s],
                         "n_seeds": int(n_seeds),
                         "age_mean": float(r.age_tx_mean), "age_std": float(r.age_std),
                         "tx_rate": float(r.tx_rate), "alpha": ch.alpha, "beta": ch.beta,
                         "rho1": ch.rho1, "pred_closed": pred_th, "pred_emp": pred_emp,
                         "closed_gap": r.nmse / max(pred_th, 1e-12) - 1.0,
                         "emp_gap": r.nmse / max(pred_emp, 1e-12) - 1.0,
                         "f_of_L": f_of_L, "jensen_J": j_L,
                         "G_meas": g_meas, "G_closed": g_L,
                         "first_share": abs(f_of_L) / max(abs(f_of_L) + abs(j_L), 1e-12),
                         "vs_iid": r.nmse / max(baselines[pl], 1e-12),
                         "cond_tv": tv, "cond_tv_noise": float(go["tv_noise"]),
                         "cond_ks": float(go["ks"]), "cond_ks_crit": float(go["ks_crit"]),
                         "n_eff": int(n_eff),
                         "run_len": {k: (float(v) if isinstance(v, (int, float, np.floating))
                                         else v) for k, v in rl.items()},
                         "emp_p": emp_p, "n_pos": int(n_pos)})
            print(f"[18]   p̄={pl:g} L={L:>4g} ρ₁={ch.rho1:+.3f}  E[NMSE]={r.nmse:.5f} "
                  f"闭式={pred_th:.5f}({r.nmse / max(pred_th, 1e-12) - 1:+.1%})  "
                  f"★G(实测)={g_meas:.5f} G(闭式)={g_L:.5f}  ×i.i.d.={rows[-1]['vs_iid']:.2f}")

    ge = [r for r in rows if r["kind"] == "ge"]
    if not ge:
        raise RuntimeError("网格全空，什么都测不到")

    # ---------- 5) ★★ C1 / C2 检验 ----------
    print("[18] " + "=" * 74)
    print("[18] ★★ C1 数据塌缩检验：同一 L 下 G(L)=E[NMSE]/p̄ 是否**与 p̄ 无关**")
    c1_rows = []
    for L in bursts:
        sub = [r for r in ge if abs(r["L"] - L) < 1e-9]
        vals = np.array([r["G_meas"] for r in sub])
        if vals.size < 2:
            continue
        cv = float(vals.std() / max(vals.mean(), 1e-12))
        # ★ 种子内噪声：G 均值的标准误 / 均值 ⇒ 这就是"纯噪声能造成的离散度"
        cv_noise = float(np.mean([r["nmse_se"] / max(r["nmse"], 1e-12) for r in sub]))
        ratio = cv / max(cv_noise, 1e-12)
        c1_rows.append({"L": float(L), "n_p": int(vals.size), "G_mean": float(vals.mean()),
                        "G_std": float(vals.std()), "cv": cv, "cv_noise": cv_noise,
                        "cv_ratio": float(ratio),
                        "spread": float(vals.max() / max(vals.min(), 1e-12))})
        print(f"[18]   L={L:>4g}  G = {vals.mean():.5f} ± {vals.std():.5f} "
              f"(跨 p̄ 的 CV {cv:.1%}；种子内噪声 CV {cv_noise:.1%}；"
              f"**比值 {ratio:.2f}**, n_p={vals.size})")
    cv_max = max(r["cv"] for r in c1_rows) if c1_rows else float("nan")
    ratio_max = max(r["cv_ratio"] for r in c1_rows) if c1_rows else float("nan")
    print(f"[18]   ⇒ 最大 CV = {cv_max:.1%}；最大 CV/噪声比值 = {ratio_max:.2f}")
    if n_seeds < 2:
        print("[18]   ⚠ n_seeds=1 ⇒ **无法区分真偏离与噪声**，"
              "CV/噪声比值不可用（请用 --seeds 3 重跑）")

    print("[18] ★★ C2 线性检验：固定 L，E[NMSE] 是否 ∝ p̄（过原点）")
    c2_rows = []
    for L in bursts:
        ps = np.array([r["p"] for r in ge if abs(r["L"] - L) < 1e-9])
        ys = np.array([r["nmse"] for r in ge if abs(r["L"] - L) < 1e-9])
        if ps.size < 2:
            continue
        slope = float(np.dot(ps, ys) / max(np.dot(ps, ps), 1e-12))   # 过原点最小二乘
        resid = ys - slope * ps
        rel = float(np.max(np.abs(resid) / np.maximum(ys, 1e-12)))
        c2_rows.append({"L": float(L), "slope": slope, "max_rel_resid": rel})
        print(f"[18]   L={L:>4g}  斜率 G(L)={slope:.5f}  过原点直线最大相对残差 {rel:.1%}")

    # 判定（预写，跑完自动对号）
    # ★★ 判据必须以「CV / 种子内噪声」为准 —— 光看 CV 本身无法区分
    #   真偏离与采样噪声（单种子版就卡在这里：CV 16.7%，不知该说是还是不是）
    if n_seeds < 2:
        verdict_c1 = (f"C1 无法判定：n_seeds=1，最大 CV {cv_max:.1%} 但**没有噪声基准**"
                      f" ⇒ 不得据此下结论（请 --seeds 3）")
    elif ratio_max < 1.5:
        verdict_c1 = (f"C1 成立：跨 p̄ 的离散度不超过种子内噪声的 {ratio_max:.2f} 倍"
                      f" ⇒ 差异**可完全由噪声解释**，G(L) 与 p̄ 无关 ⇒ 丢包率 × 突发结构**解耦**")
    elif ratio_max < 2.5:
        verdict_c1 = (f"C1 弱成立：CV/噪声 = {ratio_max:.2f}（1.5–2.5，"
                      f"存在残余 p̄ 依赖，但量级接近噪声）")
    else:
        verdict_c1 = (f"C1 ★被证伪：CV/噪声 = {ratio_max:.2f} > 2.5 ⇒ "
                      f"存在**超出噪声**的 p̄ 依赖机制")
    rel_max = max(r["max_rel_resid"] for r in c2_rows) if c2_rows else float("nan")
    verdict_c2 = ("C2 成立（E[NMSE] ∝ p̄）" if rel_max < 0.15 else
                  f"C2 偏离：最大相对残差 {rel_max:.1%}")
    print(f"[18] ⇒ {verdict_c1}")
    print(f"[18] ⇒ {verdict_c2}")

    gaps = [r["closed_gap"] for r in ge]
    print(f"[18] 闭式恒等式 E[NMSE]=p̄·G(L) 相对偏差：中位 {np.median(gaps):+.1%} "
          f"最大 {max(abs(g) for g in gaps):.1%}（{len(gaps)} 个点）")

    # ---------- 6) 图 ----------
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))
    cmap_ps = plt.cm.viridis(np.linspace(0.05, 0.85, len(p_list)))

    # ① ★ 数据塌缩主图
    ax = axes[0, 0]
    for pl, col in zip(p_list, cmap_ps):
        ls = [r["L"] for r in ge if abs(r["p"] - pl) < 1e-9]
        gs = [r["G_meas"] for r in ge if abs(r["p"] - pl) < 1e-9]
        ax.plot(ls, gs, "o-", color=col, lw=2, ms=5, label=f"p̄={pl:g}")
    if c1_rows:
        ax.plot([r["L"] for r in c1_rows], [r["G_mean"] for r in c1_rows],
                "k--", lw=1.2, alpha=0.7, label="各 L 的均值（若塌缩则点在线上）")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("平均突发长度 L (steps)")
    ax.set_ylabel(r"$G(L)\;=\;E[\mathrm{NMSE}]\,/\,\bar{p}$")
    ax.set_title(f"① ★ C1 数据塌缩检验（CV {cv_max:.1%}，CV/噪声 {ratio_max:.2f}，"
                 f"{n_seeds} 种子）")
    ax.legend(fontsize=7.5)

    # ② C2：E[NMSE] vs p̄（过原点直线？）
    ax = axes[0, 1]
    cmap_L = plt.cm.plasma(np.linspace(0.05, 0.85, len(bursts)))
    for L, col in zip(bursts, cmap_L):
        ps = [r["p"] for r in ge if abs(r["L"] - L) < 1e-9]
        ys = [r["nmse"] for r in ge if abs(r["L"] - L) < 1e-9]
        ax.plot(ps, ys, "o-", color=col, lw=1.8, ms=5, label=f"L={L:g}")
        if len(ps) >= 2:
            sl = [r["slope"] for r in c2_rows if abs(r["L"] - L) < 1e-9][0]
            xx = np.linspace(0, max(ps) * 1.05, 20)
            ax.plot(xx, sl * xx, ":", color=col, lw=1, alpha=0.6)
    ax.set_xlabel(r"平均丢包率 $\bar{p}$"); ax.set_ylabel("E[NMSE]")
    ax.set_yscale("log")
    ax.set_title(f"② C2 线性于 p̄？（虚线=过原点拟合，最大残差 {rel_max:.1%}）")
    ax.legend(fontsize=7.5)

    # ③ 闭式偏差热力图 [p̄ × L]
    ax = axes[0, 2]
    Ls_sorted = sorted({r["L"] for r in ge})
    M = np.full((len(p_list), len(Ls_sorted)), np.nan)
    for i, pl in enumerate(p_list):
        for j, L in enumerate(Ls_sorted):
            m = [r for r in ge if abs(r["p"] - pl) < 1e-9 and abs(r["L"] - L) < 1e-9]
            if m:
                M[i, j] = m[0]["closed_gap"]
    im = ax.imshow(M, origin="lower", aspect="auto", cmap="RdBu_r",
                   vmin=-max(0.3, np.nanmax(np.abs(M))), vmax=max(0.3, np.nanmax(np.abs(M))))
    ax.set_xticks(range(len(Ls_sorted)))
    ax.set_xticklabels([f"{L:g}" for L in Ls_sorted])
    ax.set_yticks(range(len(p_list)))
    ax.set_yticklabels([f"{pl:g}" for pl in p_list])
    ax.set_xlabel("突发长度 L"); ax.set_ylabel("丢包率 p̄")
    ax.set_title("③ 闭式偏差（实测/闭式−1）")
    for i in range(len(p_list)):
        for j in range(len(Ls_sorted)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:+.0%}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.04)

    # ④ 相对 i.i.d. 的倍数
    ax = axes[1, 0]
    for pl, col in zip(p_list, cmap_ps):
        ls = [r["L"] for r in ge if abs(r["p"] - pl) < 1e-9]
        vs = [r["vs_iid"] for r in ge if abs(r["p"] - pl) < 1e-9]
        ax.plot(ls, vs, "s-", color=col, lw=1.8, ms=4, label=f"p̄={pl:g}")
    ax.axhline(1.0, color=PALETTE["grey"], ls=":", label="1.0（= i.i.d.）")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("平均突发长度 L"); ax.set_ylabel("E[NMSE] / E[NMSE]_{i.i.d.}")
    ax.set_title("④ ★ 同丢包率下突发相对 i.i.d. 的倍数")
    ax.legend(fontsize=7.5)

    # ⑤ 一阶占比
    ax = axes[1, 1]
    for pl, col in zip(p_list, cmap_ps):
        ls = [r["L"] for r in ge if abs(r["p"] - pl) < 1e-9]
        fs = [r["first_share"] for r in ge if abs(r["p"] - pl) < 1e-9]
        ax.plot(ls, fs, "^--", color=col, lw=1.5, ms=4, label=f"p̄={pl:g}")
    ax.set_xscale("log")
    ax.set_xlabel("平均突发长度 L"); ax.set_ylabel("一阶项 f(L) 占比")
    ax.set_ylim(0, 1.05)
    ax.set_title("⑤ 分解：一阶 vs Jensen 二阶占比")
    ax.legend(fontsize=7.5)

    # ⑥ 离线曲线
    ax = axes[1, 2]
    ax.plot(hs, f_h, "o-", color=PALETTE["blue"])
    ax.axhline(thr, color=PALETTE["grey"], ls=":", label=f"阈值 {thr}")
    if h_err is not None:
        ax.axvline(h_err, color=PALETTE["red"], ls="--", alpha=0.8, label=f"H*_err={h_err:g}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("视界 h (steps)"); ax.set_ylabel("f(h)（全局分母口径）")
    ax.set_title("⑥ 离线曲线（全网格共享一次）")
    ax.legend(fontsize=7.5)

    fig.suptitle(f"wmlab · X31-b 多丢包率扫描 · env={cfg['env']['id']} "
                 f"σ={cfg['env']['noise_std']} · L∈{bursts} · p̄∈{p_list} · "
                 f"H*_err={h_err} · seed={seed}", fontsize=10)
    save_fig(fig, os.path.join(out, args.tag + ".png"))
    plt.close(fig)

    # ---------- 7) json ----------
    payload = {
        "experiment": "X31-b 多丢包率扫描（p̄ × L 网格）",
        "config": jsonable(cfg),
        "infeasible": [{"p": a, "L": b, "L_min": c} for a, b, c in infeasible],
        "offline_curve": {"horizons": hs, "f_global_denom": [float(x) for x in f_h],
                          "f_slice_denom": [float(x) for x in f_h_rel],
                          "h_star_err": h_err},
        "grid": jsonable(rows),
        "baselines": {f"{k:g}": v for k, v in baselines.items()},
        "C1_collapse": jsonable(c1_rows),
        "C2_linearity": jsonable(c2_rows),
        "verdict": {"C1": verdict_c1, "C2": verdict_c2, "n_seeds": int(n_seeds),
                    "closed_gap_median": float(np.median(gaps)),
                    "closed_gap_max_abs": float(max(abs(g) for g in gaps))},
        "caveat": ("★ C1/C2 是「E[NMSE] = p̄·G(L)」这个闭式的推论，前提是"
                   "误差只依赖当前信息年龄 f(k) 而不依赖历史。若存在 compound"
                   "（上次没纠上的误差被带进下一次），C1 会被证伪 —— 那也是结论。"),
        "elapsed_sec": float(time.time() - t0),
    }
    with open(os.path.join(out, args.tag + ".json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[18] 产物：{args.tag}.png / .json  ({time.time() - t0:.0f}s)")
    print("[18] OK")


if __name__ == "__main__":
    main()
