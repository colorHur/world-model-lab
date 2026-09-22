# -*- coding: utf-8 -*-
"""★ X32「物理层接入」：固定时隙下量化精度与调制阶数的速率—可靠权衡

================================================================= 实验设定
每个时隙只能传 `N_sym` 个符号（带宽 × 时隙长度，固定不变）。
要传 `d` 维状态、每维 `b` 比特 ⇒ 一包 `n_bits = d·b` 比特
⇒ 每个符号必须承载 `m = n_bits / N_sym` 比特 ⇒ **调制阶数 M = 2^m**。

取 `N_sym = d` 时 `m = b`，于是 b = 2/4/6/8 恰好是 **4/16/64/256-QAM**。

================================================================= ★★ 权衡的两个方向
    b ↑  量化更精细（误差 ∝ 2^(−2b)）   但 M ↑ ⇒ 星座更密 ⇒ BER ↑ ⇒ PER ↑ ⇒ 丢包更多
    b ↓  星座抗噪（PER 低）             但量化粗糙（**每次到达都注入**误差）

⇒ 总误差对 b 是 **U 形**，存在最优 b\*。

================================================================= ★★★ 预写判据（跑前写定，不许事后编）
问：**世界模型在环时，最优量化精度 b\* 会往哪边移？**

一个诱人但**错误**的答案是"模型能补信息损失 ⇒ 可以用更粗的量化 ⇒ b\* 下移"。
错在哪：量化误差是**每次到达都注入**的**地板** —— 收到的那一刻误差就已经在了，
世界模型无论多准都补不掉它（它补的是**时间**上的空缺，不是精度上的）。

正确的是另一个方向：

    模型能补的是**丢包**（时间维度的空缺）
    ⇒ 模型在环 ⇒ 更能扛住高 PER
    ⇒ 可以选**更大**的 b（更高阶调制、更精细量化），用"敢丢包"换"量化准"

    ⇒ **判据 P1：b\*_model ≥ b\*_persistence**
       （严格大于 = 世界模型把最优工作点推向更高阶调制）

    P2（相等）：权衡由量化地板主导，模型不改变工作点
    P3（b\*_model < b\*_persistence）：与上面推理相反，必须给出解释才能写进结论

================================================================= 消融：把两种损失分开
① 量化误差  = **确定性**、每次到达都有、模型补不掉
② 丢包误差  = **随机**、只在丢时出现、模型能补
⇒ 对每个 (SNR, b) 额外跑一次 **关掉量化、但保留同样的 PER** 的配置
   （`PhysicalChannel(quantizer=None)`）⇒ 两者之差 = 量化那一份。

================================================================= ⚠ 待核
M-QAM 误码率用的是教科书近似式，**绝对数值未与标准表核对**。
本实验的结论（b\* 的存在与移动方向）只依赖"PER 随 b 单调增"这个定性结构。
任何"在 SNR=X dB 时 PER=Y"的定量表述，核对前不得写进论文。
"""
import argparse
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
                                 qam_bit_error_rate, qam_symbol_error_rate,
                                 snr_db_to_linear)
from wmlab.eval.tracking import run_tracking
from wmlab.models import MLPWorldModel
from wmlab.rollout import closed_loop_error_curve
from wmlab.train import train_world_model
from wmlab.utils import count_params, get_device, load_config, output_dir, set_seed
from wmlab.utils.plot import PALETTE, apply_style, save_fig


class PersistenceHead:
    """★ 零侵入的 persistence 对照：把模型的"预测"换成**恒等**（零阶保持）。

    `StateTracker` 只会调 `model.predict_next(o, a)`，所以包一层即可，
    不必改 wmlab 里任何既有代码（既有实验逐位不变）。
    语义：丢包时**保持上次收到的值不动** —— 这就是 persistence 基线。
    """

    def __init__(self, model):
        self._m = model

    def eval(self):
        self._m.eval()
        return self

    def to(self, *a, **k):
        return self

    def predict_next(self, obs, act):
        return obs


def parse_args():
    p = argparse.ArgumentParser(description="X32 物理层（M-QAM + 量化）")
    p.add_argument("--config", default="configs/uav_physical.yaml")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--tag", default="19_physical_layer")
    p.add_argument("--snrs", default=None, help="逗号分隔 SNR(dB) 网格，覆盖 config")
    p.add_argument("--bits", default=None, help="逗号分隔量化比特网格，覆盖 config")
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
        cfg["physical"]["snr_db_list"] = [float(x) for x in args.snrs.split(",")]
    if args.bits:
        cfg["physical"]["bits_list"] = [int(x) for x in args.bits.split(",")]
    if args.quick:
        cfg["train"]["epochs"] = min(int(cfg["train"]["epochs"]), 3)
        cfg["data"]["n_episodes"] = min(int(cfg["data"]["n_episodes"]), 20)
        cfg["physical"]["snr_db_list"] = [0.0, 15.0]
        cfg["physical"]["bits_list"] = [2, 8]
        cfg["physical"]["n_track_episodes"] = 6
        cfg["eval"]["n_samples"] = 64

    seed = int(cfg["seed"])
    set_seed(seed)
    apply_style()
    device = get_device(args.device or cfg.get("device"))
    out = output_dir(cfg)
    t0 = time.time()

    ph = cfg["physical"]
    snrs = [float(x) for x in ph["snr_db_list"]]
    bits_list = [int(x) for x in ph["bits_list"]]
    n_track = int(ph["n_track_episodes"])
    burst_len = ph.get("burst_len", None)

    print(f"[19] ★ X32 物理层：SNR ∈ {snrs} dB   b ∈ {bits_list}")
    print(f"[19]   待验：P1 b*_model ≥ b*_persistence（模型能扛 PER ⇒ 敢用更高阶调制）")

    # ---------- 1) 环境（先拿到 obs_dim，后面的自检才知道一包多少比特）----------
    env = make_env(cfg["env"]["id"], seed=seed,
                   noise_std=float(cfg["env"]["noise_std"]),
                   max_steps=int(cfg["env"]["max_steps"]))
    obs_dim = int(env.obs_dim) if hasattr(env, "obs_dim") else 6
    print(f"[19] obs_dim={obs_dim}  N_sym=d={obs_dim} ⇒ 每符号比特数 m = b，M = 2^b")

    # ---------- 2) 物理层公式自检（闭式退化 / 单调性）----------
    # ★ M=4（QPSK）时公式必须退化为**精确**误码率 Q(√γ_s) —— 唯一能自证的一条。
    #   ⚠ 检验必须避开 SER 饱和区：低 SNR 时近似式会算出 >1−1/M 而被 clip，
    #     那时差异来自 clip 而非公式本身（第一版在 −10dB 就是这样误报的）。
    #     ⇒ 只在 γ ≥ 0dB（BER ≲ 0.16，远低于饱和 0.75）上比。
    for g_db in (0.0, 3.0, 6.0, 10.0, 20.0):
        g = snr_db_to_linear(g_db)
        ber = qam_bit_error_rate(g, 4.0)
        exact = 0.5 * math.erfc(math.sqrt(g / 2.0))          # Q(√γ_s)
        if abs(ber - exact) > 1e-12:
            raise AssertionError(
                f"★ S_a 未通过：M=4 时 BER={ber:.6e} 与 QPSK 精确值 {exact:.6e} 不符")
    # ★ 另外确认饱和 clip 真的只在极端 SNR 触发（否则会污染正常区间）
    sat = [g_db for g_db in np.arange(-20, 45, 0.5)
           if qam_symbol_error_rate(snr_db_to_linear(g_db), 4.0) >= 0.75 - 1e-12]
    print(f"[19] S_a 通过：M=4 在 γ≥0dB 上等于 QPSK 精确式；"
          f"4-QAM 的饱和 clip 仅在 SNR ≤ {max(sat) if sat else 'n/a'}dB 触发")
    # ★ PER 单调：同 SNR 下 b 越大（M 越大）PER 越大；同 b 下 SNR 越大 PER 越小
    #   ⚠ 公式失效区（SER 饱和）按"链路不可用 PER≈1"处理，否则会非单调 ——
    #     见 `qam_approximation_valid` 的说明。
    def per_at(g_db: float, b: int) -> float:
        g = snr_db_to_linear(g_db)
        if not qam_approximation_valid(g, 2.0 ** b):
            return 1.0 - 1e-6
        return per_from_ber(qam_bit_error_rate(g, 2.0 ** b), float(obs_dim * b))

    for g_db in snrs:
        pers = [per_at(g_db, b) for b in bits_list]
        if not all(np.diff(pers) >= -1e-12):
            raise AssertionError(f"★ S_b 未通过：SNR={g_db}dB 时 PER 对 b 非单调增 {pers}")
    for b in bits_list:
        pers = [per_at(g, b) for g in snrs]
        if not all(np.diff(pers) <= 1e-12):
            raise AssertionError(f"★ S_b 未通过：b={b} 时 PER 对 SNR 非单调减 {pers}")
    print("[19] S_b 通过：PER 对 b 单调增、对 SNR 单调减")

    # ---------- 3) 训练 ----------
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
    if obs_dim != int(getattr(env, "obs_dim", obs_dim)):
        raise AssertionError("obs_dim 在环境与实际数据间不一致")
    model = MLPWorldModel(obs_dim=obs_dim, act_dim=int(env.act_dim),
                          latent_dim=int(cfg["model"]["latent_dim"]),
                          hidden=int(cfg["model"]["hidden"]),
                          discrete_act=bool(env.is_discrete)).to(device)
    hist = train_world_model(model, tr, va, cfg, device, verbose=False)
    print(f"[19] 训练完成：params={count_params(model):,} val_loss={hist['val_total'][-1]:.6f} "
          f"({time.time() - t0:.0f}s)")

    # ---------- 3) 量化量程（★ 由数据统计，不拍脑袋）----------
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

    quantizers = {}
    for b in bits_list:
        q = UniformQuantizer(lo, hi, b)
        quantizers[b] = q
    # ★ S_c：量化 MSE 必须对 b 单调减；且无过载时接近 Δ²/12
    qmse = {b: q.measure_mse(allobs) for b, q in quantizers.items()}
    if not all(qmse[bits_list[i]] >= qmse[bits_list[i + 1]] * (1 - 1e-9)
               for i in range(len(bits_list) - 1)):
        raise AssertionError(f"★ S_c 未通过：量化 MSE 对 b 非单调减 {qmse}")
    ovl = {b: overload_fraction(quantizers[b], allobs) for b in bits_list}
    if max(ovl.values()) > 1e-6:
        raise AssertionError(f"★ S_e 未通过：存在过载（裁剪）{ovl} ⇒ Δ²/12 不可用，"
                             f"请加大 range_margin")
    print("[19] S_c/S_e 通过：量化 MSE 单调递减、无过载")
    print("[19]   量化 MSE（实测）：" +
          "  ".join(f"b={b}: {qmse[b]:.3e}(理论 {quantizers[b].theoretical_mse():.3e})"
                    for b in bits_list))

    # ---------- 4) 离线曲线（H*_err，沿用 X30/X31 口径）----------
    hs = [int(h) for h in cfg["eval"]["horizons"]]
    hs = [h for h in hs if h <= max(e["length"] for e in eval_eps) - WARMUP - 2]
    curve = closed_loop_error_curve(model, eval_eps, hs, device,
                                    n_samples=int(cfg["eval"]["n_samples"]),
                                    seed=seed, t0_min=WARMUP)
    f_h_rel = np.asarray([float(np.asarray(curve["nmse_per_step"], dtype=float)[h - 1])
                          for h in hs])
    h_err = reliable_horizon(hs, [float(x) for x in f_h_rel],
                             float(cfg["eval"]["err_threshold"]),
                             cfg["eval"].get("threshold_mode", "rel"))
    print(f"[19] 离线曲线：H*_err={h_err}")

    # ---------- 5) 扫 (SNR × b × estimator) ----------
    track_seed = seed + int(cfg["data"]["eval_seed_offset"])
    track_eps = eval_eps[:n_track] if n_track <= len(eval_eps) else eval_eps
    heads = {"model": model, "persistence": PersistenceHead(model)}
    rows = []

    for g_db in snrs:
        for b in bits_list:
            rec = {"snr_db": g_db, "bits": b}
            for name, head in heads.items():
                # (a) 完整：量化 + 由 (SNR,b) 决定的 PER
                ch_full = PhysicalChannel(snr_db=g_db, bits=b, obs_dim=obs_dim,
                                          quantizer=quantizers[b], burst_len=burst_len,
                                          seed=seed + 11)
                r_full = run_tracking(head, track_eps, device, ch_full, seed=track_seed,
                                      denom=var_g, label=f"{name}/full(b={b})",
                                      warmup=WARMUP, payload_fn=ch_full.payload_fn())
                # (b) 消融：同样的 PER，但**关掉量化** ⇒ 分离出量化那一份
                ch_noq = PhysicalChannel(snr_db=g_db, bits=b, obs_dim=obs_dim,
                                         quantizer=None, burst_len=burst_len,
                                         seed=seed + 11)
                r_noq = run_tracking(head, track_eps, device, ch_noq, seed=track_seed,
                                     denom=var_g, label=f"{name}/noquant(b={b})",
                                     warmup=WARMUP, payload_fn=None)
                if not np.isfinite(r_full.nmse) or not np.isfinite(r_noq.nmse):
                    raise FloatingPointError(f"({g_db}dB,b={b},{name}) 产出非有限值（R14）")
                rec[name] = {"nmse": float(r_full.nmse), "nmse_noquant": float(r_noq.nmse),
                             "quant_share": float((r_full.nmse - r_noq.nmse)
                                                  / max(r_full.nmse, 1e-12)),
                             "age_mean": float(r_full.age_tx_mean),
                             "emp_per": float(1.0 - r_full.tx_rate)}
            rec["M"] = float(2.0 ** b)
            rec["n_bits"] = int(obs_dim * b)
            rec["per"] = float(per_at(g_db, b))
            rec["formula_valid"] = bool(qam_approximation_valid(
                snr_db_to_linear(g_db), 2.0 ** b))
            rec["ber"] = float(qam_bit_error_rate(snr_db_to_linear(g_db), 2.0 ** b))
            rec["quant_mse"] = float(qmse[b])
            rows.append(rec)
            print(f"[19] SNR={g_db:>5g}dB b={b:>2d} M={rec['M']:>6.0f} "
                  f"BER={rec['ber']:.2e} PER={rec['per']:.4f}  "
                  f"model={rec['model']['nmse']:.5f}(量化占 {rec['model']['quant_share']:+.0%})  "
                  f"persist={rec['persistence']['nmse']:.5f}")

    # ★ S_d（R12）：payload_fn 真的接线了吗 —— 量化必须让结果变
    n_checked = 0
    for g_db in snrs:
        sub = [r for r in rows if abs(r["snr_db"] - g_db) < 1e-9]
        for r in sub:
            # ★ PER≈1 时**所有包都丢**，量化根本没有机会生效 ⇒ 跳过
            #   （第一版没排除，把"链路完全不可用"误报成了"payload_fn 没接线"）
            if r["per"] > 0.999:
                continue
            n_checked += 1
            if abs(r["model"]["nmse"] - r["model"]["nmse_noquant"]) < 1e-12:
                raise AssertionError(
                    f"★ S_d 未通过：SNR={g_db}dB b={r['bits']} 开关量化后结果**逐位不变** "
                    f"⇒ payload_fn 没接线（R12 违规），整个实验在空跑")
    if n_checked == 0:
        raise AssertionError(
            "★ S_d 未通过：所有网格点的 PER 都 ≈1（链路全不可用）⇒ 根本没验到接线。"
            "请调高 snr_db_list 或减小 bits_list")
    print(f"[19] S_d 通过：量化开关确实改变结果（payload_fn 已接线，{n_checked} 个点有效）")

    # ---------- 6) b* 判定 ----------
    print("[19] " + "=" * 74)
    bstar = {}
    for g_db in snrs:
        sub = [r for r in rows if abs(r["snr_db"] - g_db) < 1e-9]
        sub.sort(key=lambda r: r["bits"])
        for name in ("model", "persistence"):
            vals = [r[name]["nmse"] for r in sub]
            k = int(np.argmin(vals))
            bstar[(g_db, name)] = sub[k]["bits"]
            print(f"[19] SNR={g_db:>5g}dB  b*_{name} = {sub[k]['bits']:>2d}  "
                  f"(误差 {vals[k]:.5f}，曲线 " +
                  " ".join(f"{r['bits']}:{r[name]['nmse']:.4f}" for r in sub) + ")")

    diffs = [bstar[(g, "model")] - bstar[(g, "persistence")] for g in snrs]
    if all(d > 0 for d in diffs):
        verdict = (f"P1 成立：b*_model 在每个 SNR 上都 **大于** b*_persistence"
                   f"（差 {diffs}）⇒ 世界模型把最优工作点推向**更高阶调制**："
                   f"它补的是丢包不是精度，所以敢用「易错但精细」的配置")
    elif all(d == 0 for d in diffs):
        verdict = "P2：b* 与估计器无关 ⇒ 权衡由量化地板主导，模型不改变工作点"
    elif all(d < 0 for d in diffs):
        verdict = (f"P3 ★反直觉：b*_model < b*_persistence（差 {diffs}）—— 与"
                   f"「模型补丢包不补精度」的推理相反，必须给出机制解释才能写进结论")
    else:
        verdict = f"P1/P3 混合：差 {diffs}（随 SNR 变号）"
    print(f"[19] ⇒ {verdict}")

    # ★★ 补充结论（这才是真正被数据支持的那个）：模型的价值**随 PER 增长**
    #   PER≈0 ⇒ 误差全是量化地板 ⇒ 两个估计器相同（模型补不了精度）
    #   PER 高 ⇒ 误差主要是丢包 ⇒ 模型把它压下来
    gains = []
    for r in rows:
        if r["per"] > 0.999 or r["model"]["nmse"] < 1e-12:
            continue
        gains.append((r["per"], r["persistence"]["nmse"] / r["model"]["nmse"],
                      r["snr_db"], r["bits"]))
    gains.sort()
    print("[19] ★★ 模型增益（persistence/model）随 PER 的变化：")
    for per_, g, sd, b in gains:
        print(f"[19]     PER={per_:.4f} (SNR={sd:g}dB, b={b}): ×{g:.2f}")
    lo = [g for per_, g, _, _ in gains if per_ < 0.05]
    hi = [g for per_, g, _, _ in gains if per_ > 0.3]
    if lo and hi:
        print(f"[19]   ⇒ 低 PER(<0.05) 平均 ×{np.mean(lo):.2f}；"
              f"高 PER(>0.3) 平均 ×{np.mean(hi):.2f}")
        verdict_gain = (f"模型增益从低 PER 的 ×{np.mean(lo):.2f} 升到高 PER 的 "
                        f"×{np.mean(hi):.2f} ⇒ 模型补的是**丢包**，不是**量化精度**")
    else:
        verdict_gain = "样本不足以比较高低 PER 两档"
    print(f"[19] ⇒ {verdict_gain}")
    verdict = verdict + "  ｜  " + verdict_gain

    # ---------- 7) 图 ----------
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))
    cmap_s = plt.cm.viridis(np.linspace(0.05, 0.85, len(snrs)))

    # ① 总误差 vs b（U 形？）
    for name, ax, mk in (("model", axes[0, 0], "o-"), ("persistence", axes[1, 0], "s--")):
        for g_db, col in zip(snrs, cmap_s):
            sub = sorted([r for r in rows if abs(r["snr_db"] - g_db) < 1e-9],
                         key=lambda r: r["bits"])
            ax.plot([r["bits"] for r in sub], [r[name]["nmse"] for r in sub],
                    mk, color=col, lw=1.8, ms=4, label=f"{g_db:g}dB")
            ax.scatter([bstar[(g_db, name)]],
                       [min(r[name]["nmse"] for r in sub)],
                       marker="*", s=140, color=col, zorder=5)
        ax.set_yscale("log")
        ax.set_xlabel("量化比特 b（= 每符号比特数 ⇒ M=2^b）")
        ax.set_ylabel("E[NMSE]")
        ax.set_title(f"{'①' if name == 'model' else '④'} ★ 总误差 vs b"
                     f"（★ = b*{'' if name == 'model' else '_persistence'}）")
        ax.legend(fontsize=7, ncol=2)

    # ② PER / BER vs b
    ax = axes[0, 1]
    for g_db, col in zip(snrs, cmap_s):
        sub = sorted([r for r in rows if abs(r["snr_db"] - g_db) < 1e-9],
                     key=lambda r: r["bits"])
        ax.plot([r["bits"] for r in sub], [r["per"] for r in sub], "o-",
                color=col, lw=1.8, ms=4, label=f"{g_db:g}dB")
    ax.set_yscale("log")
    ax.set_xlabel("量化比特 b"); ax.set_ylabel("包错误率 PER")
    ax.set_title("② PER 随 b 单调增（星座变密）")
    ax.legend(fontsize=7, ncol=2)

    # ③ 量化 MSE vs b（与 PER 对照：一个降一个升 = U 形的两个来源）
    ax = axes[0, 2]
    ax.plot(bits_list, [qmse[b] for b in bits_list], "o-", color=PALETTE["blue"],
            label="量化 MSE（实测，单调降）")
    ax.set_yscale("log")
    ax.set_xlabel("量化比特 b"); ax.set_ylabel("量化 MSE")
    ax2 = ax.twinx()
    for g_db, col in zip(snrs, cmap_s):
        sub = sorted([r for r in rows if abs(r["snr_db"] - g_db) < 1e-9],
                     key=lambda r: r["bits"])
        ax2.plot([r["bits"] for r in sub], [r["per"] for r in sub], "s:",
                 color=col, lw=1.2, alpha=0.8)
    ax2.set_yscale("log"); ax2.set_ylabel("PER（虚线，单调升）")
    ax.set_title("③ U 形的两个来源：量化降 / PER 升")
    ax.legend(fontsize=7)

    # ⑤ b* vs SNR（两种估计器）
    ax = axes[1, 1]
    ax.plot(snrs, [bstar[(g, "model")] for g in snrs], "o-", color=PALETTE["red"],
            lw=2, ms=6, label="b*_model（世界模型在环）")
    ax.plot(snrs, [bstar[(g, "persistence")] for g in snrs], "s--", color=PALETTE["grey"],
            lw=2, ms=6, label="b*_persistence（零阶保持）")
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("最优量化比特 b*")
    ax.set_title("⑤ ★ b* 随 SNR 的移动（判据 P1/P2/P3）")
    ax.legend(fontsize=7.5)

    # ⑥ ★★ 核心：模型增益 vs PER（"模型补丢包、不补量化"的直接证据）
    ax = axes[1, 2]
    for g_db, col in zip(snrs, cmap_s):
        sub = [r for r in rows if abs(r["snr_db"] - g_db) < 1e-9]
        xs = [max(r["per"], 1e-6) for r in sub]
        ys = [r["persistence"]["nmse"] / max(r["model"]["nmse"], 1e-12) for r in sub]
        ax.plot(xs, ys, "o", color=col, ms=6, label=f"{g_db:g}dB")
        for r, x, y in zip(sub, xs, ys):
            ax.annotate(f"b={r['bits']}", (x, y), fontsize=6.5,
                        xytext=(4, 3), textcoords="offset points")
    ax.axhline(1.0, color=PALETTE["grey"], ls=":", label="1.0（模型无增益）")
    ax.set_xscale("log")
    ax.set_xlabel("包错误率 PER（对数轴）")
    ax.set_ylabel("persistence 误差 / model 误差")
    ax.set_yscale("log")
    ax.set_title("⑥ ★ 模型增益随 PER 增长（PER≈0 时 =1：量化地板补不了）")
    ax.legend(fontsize=7, ncol=2)

    fig.suptitle(f"wmlab · X32 物理层（M-QAM + 量化） · env={cfg['env']['id']} "
                 f"σ={cfg['env']['noise_std']} · N_sym=d={obs_dim} · H*_err={h_err} · "
                 f"seed={seed}", fontsize=10)
    save_fig(fig, os.path.join(out, args.tag + ".png"))
    plt.close(fig)

    payload = {
        "experiment": "X32 物理层：量化精度 × 调制阶数的速率-可靠权衡",
        "config": jsonable(cfg),
        "setup": {"n_sym": obs_dim, "obs_dim": obs_dim, "M": "2^b",
                  "formula_source": "Proakis/Goldsmith 教科书近似 —— 定性结构可信，"
                                    "绝对数值【待核】，未与标准表核对"},
        "quantizer": {"range_lo": [float(x) for x in lo], "range_hi": [float(x) for x in hi],
                      "mse_by_bits": {str(b): float(qmse[b]) for b in bits_list},
                      "theoretical_mse": {str(b): float(quantizers[b].theoretical_mse())
                                          for b in bits_list},
                      "overload_fraction": {str(b): float(ovl[b]) for b in bits_list}},
        "grid": jsonable(rows),
        "b_star": {f"{g:g}dB": {"model": bstar[(g, "model")],
                                "persistence": bstar[(g, "persistence")],
                                "diff": bstar[(g, "model")] - bstar[(g, "persistence")]}
                   for g in snrs},
        "verdict": verdict,
        "gain_vs_per": [{"per": float(a), "gain": float(b), "snr_db": float(c),
                         "bits": int(d)} for a, b, c, d in gains],
        "offline_h_star_err": h_err,
        "elapsed_sec": float(time.time() - t0),
    }
    with open(os.path.join(out, args.tag + ".json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[19] 产物：{args.tag}.png / .json  ({time.time() - t0:.0f}s)")
    print("[19] OK")


if __name__ == "__main__":
    main()
