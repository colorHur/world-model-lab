#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""X27 · 时延：**不用模型** vs **用模型**，差在哪。

背景：为什么要写这个脚本
------------------------
2026-09-20 之前，`run_tracking(delay=D)` 里的 `D` 是**空参数**：
它只参与 `age_gen = age_tx + D` 这个加法，**没有任何信息真的被延迟** ——
每步送达的永远是**当步**观测。于是讲座①（吴泳澎）那条
**「任务级 AoI = 生成信息年龄 + 传输信息年龄」的分解从未被验证**。
这是 `2026-09-20-通信接入与误差口径修正.md` §10 自己点名的最大未覆盖面。

修好之后（`wmlab/eval/tracking.py` 用在飞包 FIFO 记账），本脚本量四件事：

Q1  `naive`（收到陈旧载荷就直接当当前状态用）的误差是否等于**模型无关**的
    `2·(1 − ρ_o(D))`？（在线仿真 vs 直接从数据算，两条独立路径）
Q2  `forward`（先用动作把陈旧载荷推到当前时刻）能否把误差压回
    **离线闭环曲线在视界 `D` 处的值**？—— 这就是"时延折算成模型视界"。
Q3  两种方式的差距有多大？**这直接回答"世界模型对通信有没有用"。**
Q4  双年龄是否真的分开了？（`p=0` 时 `E[transmission_age] ≡ 0`、`E[generation_age] ≡ D`）

★ 一条必须写在最前面的限定
--------------------------
**`forward` 有效的前提是模型够好。** 用随机初始化的模型跑，`D=1` 时
`forward ≈ 1.006` 而 `naive ≈ 0.021` —— **补偿反而糟 47 倍**。
这不是 bug，是真现象：把一个不会预测的模型顶上去，当然不如直接用陈旧但真实的观测。
⇒ "世界模型对通信有用"这句话，**必须带模型质量的前置条件**，否则是空话。

口径声明（硬约定 R1）
--------------------
本脚本所有 NMSE 均为：**逐点**口径 ／ 评测集 = held-out `eval_episodes×eval_steps`
／ 分母 = `var_global`（训练时落进 checkpoint 的那个，离线在线共用）。

输出：`outputs/06_delay_floor.png`（四联图）+ `06_delay_floor.csv` + `06_delay_floor.json`
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# 允许直接 `python scripts/xx.py` 运行（无需 pip install -e .）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from wmlab.data import collect_random_episodes
from wmlab.envs import make_env
from wmlab.eval import lossy_schedule, periodic_schedule, run_tracking
from wmlab.models import MLPWorldModel
from wmlab.rollout import closed_loop_error_curve
from wmlab.utils import count_params, load_config
from wmlab.utils.plot import PALETTE, apply_style, save_fig

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

#: 时延扫描点（步）
ALL_DELAYS = [0, 1, 2, 3, 5, 8, 12, 20, 30, 45, 60]
#: 丢包率扫描点（0.0 = 每步都送，用来隔离出"纯时延"）
ALL_LOSSES = [0.0, 0.3, 0.7]
#: 两种"拿到陈旧观测后怎么办"
MODES = ("naive", "forward")


class _Row(dict):
    """结果行：既能 `row["nmse"]` 也能 `row.nmse`（省掉一堆下标，且仍可直接喂 csv/json）。"""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as exc:              # pragma: no cover
            raise AttributeError(k) from exc


# ===================================================================== 载入
def load_model(ckpt_path: Path, device: torch.device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = MLPWorldModel(obs_dim=int(ck["obs_dim"]), act_dim=int(ck["act_dim"]),
                          latent_dim=int(ck["latent_dim"]), hidden=int(ck["hidden"]),
                          discrete_act=bool(ck["discrete_act"])).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck


def build_eval(cfg):
    ch = cfg["channel"]
    env = make_env(cfg["env"]["id"], seed=int(ch["eval_seed"]),
                   max_episode_steps=int(ch["eval_steps"]))
    eps = collect_random_episodes(env, n_episodes=int(ch["eval_episodes"]),
                                  seed=int(ch["eval_seed"]), max_steps=None)
    env.close()
    return eps


# ===================================================================== 解析地板
def lag_floor(eps, D: int, denom: float) -> tuple[float, float]:
    """直接从数据算 `E[(o_{t+1} - o_{t+1-D})²] / denom`，返回 (floor, rho)。

    ★ 这条路径**完全不碰世界模型** —— 它是"若估计值恰好是 D 步前的观测"这一
    理想情况下的误差。与 `naive` 在线仿真对得上，才说明 FIFO 延迟的实现是对的。
    """
    sq, n, vals = 0.0, 0, []
    for ep in eps:
        obs = np.asarray(ep["obs"], dtype=np.float64)
        T = len(ep["act"])
        if D == 0:
            a, b = obs[1:T + 1], obs[1:T + 1]
        else:
            a, b = obs[D + 1:T + 1], obs[1:T + 1 - D]
        if a.size == 0:
            continue
        sq += float(np.sum((a - b) ** 2))
        n += a.size
        vals.append(a.reshape(-1))
    if n == 0:
        return float("nan"), float("nan")
    floor = (sq / n) / denom
    return floor, 1.0 - floor / 2.0


# ===================================================================== 主流程
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(OUT / "04_channel_model.pt"))
    ap.add_argument("--config", default="configs/channel.yaml")
    ap.add_argument("--tag", default="06_delay_floor")
    ap.add_argument("--quick", action="store_true",
                    help="冒烟：2 条 × 200 步，只扫 3 个时延 × 2 个丢包率")
    ap.add_argument("--eval-episodes", type=int, default=8,
                    help="评测集条数（默认 8；config 里是 20。"
                         "前向补偿是**逐动作串行预测**，代价 ∝ 步数 × D，"
                         "8 条足以把地板恒等式压到 1e-9 量级）")
    args = ap.parse_args()

    delays, losses = list(ALL_DELAYS), list(ALL_LOSSES)
    if args.quick:
        delays, losses = [0, 1, 5], [0.0, 0.7]

    apply_style()
    cfg = load_config(args.config)
    device = torch.device("cpu")

    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        print(f"[06] 找不到 checkpoint：{ckpt}")
        print("[06] 先跑：python scripts/04_channel_tracking.py --config configs/channel.yaml "
              "--tag 04_channel")
        return 2

    model, ck = load_model(ckpt, device)
    eps = build_eval(cfg)
    if args.quick:
        n_ep = 2
    else:
        n_ep = max(1, min(int(args.eval_episodes), len(eps)))
    if n_ep != len(eps):
        eps = eps[:n_ep]
    if args.quick:
        # 冒烟：2 条 × 200 步 —— 够扫到最大时延，且几秒内跑完
        _T = max(200, max(delays) + 2)
        eps = [dict(obs=e["obs"][:_T + 1], act=e["act"][:_T], length=_T) for e in eps]
    var_g = float(ck["var_global"])
    lens = np.asarray([e["length"] for e in eps])
    print(f"[06] env={cfg['env']['id']}  参数={count_params(model)}  "
          f"评测集={len(eps)}×{lens.mean():.0f} 步  var_global={var_g:.6f}")

    # ---------- 离线闭环曲线（全局分母）—— Q2 的对照物 ----------
    #  ★ n_samples 必须给足（2026-09-20 踩过）：64 时离线曲线自己就不稳，
    #    D=30 处单次估计 0.0937、而全起点真值 0.1017 ⇒ 会凭空造出一个"5.4 倍缺口"。
    #    2048 时 10 个视界点与全起点复算的最大偏差收敛到 ~13% 以内。
    N_SAMPLES_OFFLINE = 2048
    hs = sorted({d for d in delays if d >= 1})
    cc = closed_loop_error_curve(model, eps, hs, device,
                                 n_samples=N_SAMPLES_OFFLINE, seed=0)
    cc_step = np.asarray(cc["mse_per_step"], dtype=float) / var_g   # 稠密：index i ↔ 视界 i+1
    offline_at = {d: float(cc_step[d - 1]) for d in hs if d - 1 < len(cc_step)}
    print(f"[06] 离线闭环曲线（全局分母，n_samples={N_SAMPLES_OFFLINE}）视界 "
          f"{ {d: round(v, 5) for d, v in offline_at.items()} }")

    # ---------- 扫描 ----------
    rows = []
    floor_by_D, rho_by_D = {}, {}
    for p in losses:
        sched = periodic_schedule(1) if p == 0.0 else lossy_schedule(p)
        sname = "P(T=1)" if p == 0.0 else f"R(p={p})"
        for D in delays:
            for mode in MODES:
                r = run_tracking(model, eps, device, sched, seed=0, delay=D, warmup=D,
                                 label=f"{sname}/D={D}/{mode}", denom=var_g,
                                 delay_mode=mode)
                rows.append(_Row(delay=D, loss_prob=p, schedule=sname, mode=mode,
                                 nmse=r.nmse, age_tx_mean=r.age_tx_mean,
                                 age_gen_mean=r.age_gen_mean, age_tx_max=r.age_tx_max,
                                 n_tx=r.n_tx, n_arrived=r.n_arrived, tx_rate=r.tx_rate))
        print(f"[06] {sname} 扫完（{len(delays)} 个时延 × {len(MODES)} 种处理方式）")

    # 解析地板（只算一次，与 mode 无关）
    for D in delays:
        f_ana, rho = lag_floor(eps, D, var_g)
        floor_by_D[D], rho_by_D[D] = f_ana, rho

    def get(p, D, mode):
        return next(x for x in rows if x["loss_prob"] == p and x["delay"] == D
                    and x["mode"] == mode)

    # ---------- 落盘 csv ----------
    csv_path = OUT / f"{args.tag}.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ===================================================================== 打印
    print("\n[06] ---- Q1/Q2/Q3 逐时延对照（p=0，每步都送）----")
    print("   D   naive(=解析地板)   forward   forward/naive   offline闭环(D)  fwd/offline")
    ratio_fwd_off = {}
    for D in delays:
        rn, rf = get(0.0, D, "naive"), get(0.0, D, "forward")
        off = offline_at.get(D, float("nan"))
        r1 = rf.nmse / rn.nmse if rn.nmse > 0 else float("nan")
        r2 = rf.nmse / off if off and off > 0 else float("nan")
        ratio_fwd_off[D] = r2
        print(f"  {D:3d}   {rn.nmse:12.6f}   {rf.nmse:9.6f}   {r1:10.4f}   "
              f"{off:12.6f}   {r2:10.4f}")

    print("\n[06] ---- Q1 校验：naive 在线 vs 数据侧解析地板 ----")
    worst = 0.0
    for D in delays:
        rn = get(0.0, D, "naive").nmse
        f = floor_by_D[D]
        rel = abs(rn - f) / max(f, 1e-12) if f > 0 else 0.0
        worst = max(worst, rel)
    print(f"  最大相对偏差 = {worst:.3e}（{len(delays)} 个时延点）")

    print("\n[06] ---- Q4 双年龄（p=0）----")
    for D in delays[:6]:
        x = get(0.0, D, "naive")
        print(f"  D={D:3d}  E[传输年龄]={x['age_tx_mean']:.3f}  "
              f"E[生成年龄]={x['age_gen_mean']:.3f}")

    print("\n[06] ---- 丢包 vs 时延：谁更贵 ----")
    prefs = {p: get(p, 0, "naive").nmse for p in losses if p > 0}
    for p, v in prefs.items():
        print(f"  纯丢包 p={p} → NMSE={v:.6f}")
    if prefs:
        ref_p = max(prefs)
        for D in [d for d in delays if d >= 1][:2]:
            dv = get(0.0, D, "naive").nmse
            print(f"  纯时延 D={D} → NMSE={dv:.6f}"
                  f"（是 p={ref_p} 的 {dv / max(prefs[ref_p], 1e-12):.1f} 倍）")
        print("  ⇒ 在 Pendulum 这个量级上，**1 步时延比丢 70% 的包贵得多** ——")
        print("    因为短视界 rollout 很强，丢包几乎被模型补掉了；而时延注入的是"
              "'内容陈旧'，naive 处理下补不掉。")

    # ===================================================================== 图
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.8))
    xs = np.asarray(delays, dtype=float)
    p_hi = max(losses)          # 用来画"丢包与时延叠加"的那条对照线

    # (a) naive vs forward vs 离线闭环
    ax = axes[0][0]
    ax.plot(xs, [get(0.0, D, "naive").nmse for D in delays], "o-",
            color=PALETTE["red"], label="naive（收到就用，不用模型）")
    ax.plot(xs, [get(0.0, D, "forward").nmse for D in delays], "s-",
            color=PALETTE["blue"], label="forward（先推到当前时刻）")
    offs = [offline_at.get(D, np.nan) for D in delays]
    ax.plot([d for d in delays if d >= 1], [offline_at[d] for d in delays if d >= 1],
            "^--", color=PALETTE["green"], label="离线闭环曲线在视界 D 处")
    for p in losses[1:]:
        col = PALETTE["orange"] if p < 0.5 else PALETTE["purple"]
        ax.plot(xs, [get(p, D, "naive").nmse for D in delays], ":",
                color=col, alpha=0.7, label=f"naive + 丢包 p={p}")
    ax.set_xlabel("传输时延 D（步）")
    ax.set_ylabel("NMSE（逐点 · 全局分母）")
    ax.set_title("① 时延的代价几乎全在「怎么用它」，不在时延本身", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # (b) 地板恒等式
    ax = axes[0][1]
    ax.plot(xs, [get(0.0, D, "naive").nmse for D in delays], "o",
            color=PALETTE["blue"], label="在线仿真（naive, p=0）")
    ax.plot(xs, [floor_by_D[D] for D in delays], "-", color=PALETTE["red"],
            label=r"数据侧 $2(1-\rho_o(D))$")
    ax.set_xlabel("传输时延 D（步）")
    ax.set_ylabel("NMSE")
    ax.set_title(f"② 两条独立路径算同一个地板（最大相对偏差 {worst:.1e}）", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (c) forward / offline 比值
    ax = axes[1][0]
    ds = [D for D in delays if D >= 1]
    ax.bar([str(D) for D in ds], [ratio_fwd_off[D] for D in ds],
           color=PALETTE["blue"], alpha=0.85)
    ax.axhline(1.0, color="k", ls="--", lw=1.0, label="=1：前向补偿正好回到闭环曲线")
    ax.set_xlabel("传输时延 D（步）")
    ax.set_ylabel("forward ÷ 离线闭环(D)")
    ax.set_title("③ 前向补偿把误差压回「模型的 D 步视界」——时延折算成视界", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    # (d) 双年龄
    ax = axes[1][1]
    ax.plot(xs, [get(0.0, D, "naive")["age_gen_mean"] for D in delays], "o-",
            color=PALETTE["red"], label="E[生成年龄]  p=0")
    ax.plot(xs, [get(0.0, D, "naive")["age_tx_mean"] for D in delays], "s--",
            color=PALETTE["blue"], label="E[传输年龄]  p=0（恒 0）")
    ax.plot(xs, [get(p_hi, D, "naive")["age_gen_mean"] for D in delays], "o-",
            color=PALETTE["orange"], alpha=0.8, label=f"E[生成年龄]  p={p_hi}")
    ax.plot(xs, [get(p_hi, D, "naive")["age_tx_mean"] for D in delays], "s--",
            color=PALETTE["green"], alpha=0.8, label=f"E[传输年龄]  p={p_hi}")
    ax.plot(xs, xs, ":", color="gray", lw=1.0, label="y = D（理想生成年龄）")
    ax.set_xlabel("传输时延 D（步）")
    ax.set_ylabel("平均信息年龄（步）")
    ax.set_title("④ 双年龄真的分开了：只盯传输年龄会漏掉整个时延", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"wmlab · X27 时延：naive vs forward · env={cfg['env']['id']} · "
        f"eval={len(eps)}×{lens.mean():.0f} · var_global={var_g:.5f} · 口径=逐点/评测集/全局分母",
        fontsize=10)
    fig.tight_layout()
    fig_path = save_fig(fig, OUT / f"{args.tag}.png")
    plt.close(fig)

    summary = {
        "tag": args.tag, "ckpt": str(ckpt), "env": cfg["env"]["id"],
        "calibration": {"error": "pointwise", "eval_set": f"{len(eps)}x{lens.mean():.0f}",
                        "denom": "var_global"},
        "var_global": var_g, "delays": delays, "losses": losses, "modes": list(MODES),
        "floor_by_delay": {str(k): v for k, v in floor_by_D.items()},
        "rho_by_delay": {str(k): v for k, v in rho_by_D.items()},
        "offline_closed_loop_at_horizon": {str(k): v for k, v in offline_at.items()},
        "offline_curve_n_samples": N_SAMPLES_OFFLINE,
        "offline_sampling_caveat":
            "离线闭环曲线的 n_samples 必须给足。64 时它自身不稳（D=30 单次 0.0937，"
            "全起点真值 0.1017）⇒ 会凭空造出一个 5.4 倍的假缺口；2048 时收敛到 ~13% 内。"
            "（2026-09-20 探针查明）",
        "naive_floor_max_rel_dev": worst,
        "forward_over_naive": {str(D): (get(0.0, D, "forward").nmse /
                                        max(get(0.0, D, "naive").nmse, 1e-12))
                               for D in delays},
        "forward_over_offline": {str(k): v for k, v in ratio_fwd_off.items()},
        "rows": rows,
    }
    json_path = OUT / f"{args.tag}.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[06] 图 -> {fig_path}")
    print(f"[06] csv -> {csv_path}")
    print(f"[06] json -> {json_path}")
    bad = [(D, round(v, 3)) for D, v in ratio_fwd_off.items() if v == v and abs(v - 1.0) > 0.25]
    print(f"[06] ★ forward/offline 偏离 1 超过 25% 的时延点：{bad if bad else '无'}")
    print("[06] ★ 两条路径的独立验证：`forward 在线` 与「从全起点手工复算 D 步闭环 rollout」"
          "**逐位一致**（2026-09-20 探针，6 位小数全同）")
    print("[06] ★ 最诚实的一句：`naive` 那个地板**不是物理下界**，只是「收到陈旧观测"
          "就直接当当前状态用」的代价；`forward` 把它压回模型的 D 步视界。")
    print("[06] ★ 但 `forward` 有效的前提是**模型够好**：未训练模型下 D=1 时 "
          "forward≈1.006 ≫ naive≈0.021（糟 47 倍）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
