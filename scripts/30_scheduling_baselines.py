# -*- coding: utf-8 -*-
"""调度基线库 v0.1 的驱动入口：把**六个调度族**放进同一张表，跑通其中信号无关的三族。

================================================================= 为什么有这一号
X38–X45 把「换触发统计量」这条路走完了（周期 → 年龄阈值 → 自报 U → 残差 z →
共形信封 → oracle），结论是**本场景下触发量不是瓶颈**。
但这条链的**工程副产物**一直散在 `scripts/24..29` 里：每轮都要重写一遍
「等预算配对 + 发散过滤 + 结果落盘」，同一种比较出现过 4 份实现，其中 2 次
因为守卫写错产出假头条（X40 的「9/9 更优」、X45 的「max 口径 FAIL / CV 口径 PASS」）。
⇒ 本脚本把这件事收成 `wmlab/eval/scheduling.py` + 这一个入口。

================================================================= 本版本**做到了什么**（诚实边界）
✅ `periodic` / `threshold` / `timing` 三族：**端到端跑通**
   —— 它们信号无关，可以喂进纯调度空跑，直接给 E[age] 与发送率，
   并与 `tracking.py` 的**精确闭式**对账（`--n` 越大越紧）。
✅ 统一 schema（`Row`）+ 批量 JSON 导出 + 等预算配对（走 `pairing.overlap_window`）。
✅ 注册表把六族的**可部署性**写进代码：`oracle` 标 `diagnostic=True, deployable=False`。
⛔ `selfreport` / `conformal` / `oracle` 三族：**只注册、未在此跑**
   —— 它们要读闭环里每步写入的 `state`（模型自报 U / 真值误差）。
   `simulate_schedule()` 对它们**直接 raise**，而不是用空 state 凑一个数。
   这三族的历史结果在 `scripts/24..27`（X38–X43）。

================================================================= 自检（不过就抛 —— R14）
S_s1 闭式对账：三族空跑的 E[age] 与 `tracking` 的**精确闭式**对齐，判据是
     **5×批量均值标准误**（99 个点同时判 ⇒ 做多重比较修正），**不是固定百分比**
     —— 因为 p=0.7/T=16 档的 MC 标准误本身就有百分之几，拍 1% 会误杀正确实现。
S_s2 ★ 预算接线（X44 的前提）：`timing` 六个档的 `n_tx` **逐位相等** = `n_tx_budget`。
S_s3 退化：`periodic(T=1)` ≡ 每步发（`tx_rate = 1`）；`threshold(K=0)` ≡ 每步尝试。
S_s4 等预算配对的可评估性：`periodic` vs `threshold` 在公共发送率窗口上
     **必须**满足 `pairing` 的两条守卫，否则打印原因并**不产出比值**（不许静默 continue）。
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

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from wmlab.eval import pairing, scheduling  # noqa: E402
from wmlab.eval.timing import n_tx_budget, plan_times, timing_dispersion  # noqa: E402
from wmlab.eval.tracking import (periodic_mean_age,  # noqa: E402
                                 threshold_mean_age)
from wmlab.utils import load_config, output_dir  # noqa: E402
from wmlab.utils.plot import PALETTE, apply_style, save_fig  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/uav_triggered.yaml")
    p.add_argument("--n", type=int, default=100000, help="空跑步数（越大闭式对账越紧）")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tmax", type=int, default=16, help="periodic/threshold 的旋钮上限")
    p.add_argument("--loss-probs", type=float, nargs="+", default=[0.0, 0.3, 0.7])
    p.add_argument("--tag", default="30_scheduling_baselines")
    return p.parse_args()


# ------------------------------------------------------------------ Part 0：注册表
def print_registry() -> None:
    print("[30] 调度族注册表（v0.1）")
    for fam in scheduling.FAMILIES:
        sp = scheduling.REGISTRY[fam]
        flag = "可部署" if sp.deployable else "★不可部署(诊断用)"
        wired = "空跑已接通" if fam not in scheduling.SIGNAL_DEPENDENT else "★需闭环，本脚本不跑"
        print(f"  · {fam:<11} 旋钮={sp.knob:<12} {flag:<16} {wired}")
        print(f"      needs={sp.needs}")


# ------------------------------------------------------------------ Part 1：闭式对账
def check_closed_forms(n: int, seed: int, loss_probs: list[float],
                       tmax: int) -> dict:
    """S_s1/S_s3：逐点把空跑结果与 `tracking` 精确闭式对账。

    ★ 判据 = **5×批量均值标准误**（`scheduling.check_tolerance`），**不是固定百分比**。
      理由（真实测量，不是理论顾虑）：`age` 是重尾 + 强自相关的量，p=0.7、T=16 档的
      再生周期 E[D] ≈ 53 步 ⇒ n=5e4 下**蒙特卡洛标准误本身就有百分之几**。
      若拍一个 1% 的容差，这条判据会在**实现完全正确**的情况下失败 ——
      那会把"判据口径错"误报成"代码错"（X45 的同族错误：口径反转结论）。
    ★ 为什么是 5σ 而不是 4σ：**这里是 99 个点同时判**（多重比较）。
      4σ 的单点 p ≈ 6e-5 ⇒ 99 点的族错误率 ≈ 0.6%（实测最差比值 0.995，擦线通过）；
      5σ 的单点 p ≈ 6e-7 ⇒ 族错误率 ≈ 6e-5 ⇒ 重跑稳定。
      （不拍百分比，但也不假装"单点阈值可以直接搬到 99 点上"。）
    """
    K_SIGMA = 5.0
    worst = 0.0
    worst_key = ""
    rows = []
    for p in loss_probs:
        for T in range(1, int(tmax) + 1):
            r = scheduling.simulate_schedule(
                scheduling.build("periodic", loss_prob=p, period=T), n, seed)
            ref = periodic_mean_age(p, T)
            ratio = scheduling.check_tolerance(abs(r["mean_age"] - ref),
                                               r["mean_age_se"], k_sigma=K_SIGMA)
            rows.append(("periodic", p, T, r["mean_age"], ref, ratio))
            if ratio > worst:
                worst, worst_key = ratio, f"periodic(p={p},T={T})"
        for K in range(0, int(tmax) + 1):
            r = scheduling.simulate_schedule(
                scheduling.build("threshold", loss_prob=p, threshold=K), n, seed)
            ref = threshold_mean_age(K, p)
            ratio = scheduling.check_tolerance(abs(r["mean_age"] - ref),
                                               r["mean_age_se"], k_sigma=K_SIGMA)
            rows.append(("threshold", p, K, r["mean_age"], ref, ratio))
            if ratio > worst:
                worst, worst_key = ratio, f"threshold(p={p},K={K})"
    r1 = scheduling.simulate_schedule(
        scheduling.build("periodic", loss_prob=0.0, period=1), n, seed)
    deg_T1 = abs(r1["tx_rate"] - 1.0)
    r2 = scheduling.simulate_schedule(
        scheduling.build("threshold", loss_prob=0.0, threshold=0), n, seed)
    deg_K0 = abs(r2["tx_rate"] - 1.0)
    print(f"[30] S_s1 闭式对账：最差判据比值 diff/(5·SE) = {worst:.3f}"
          f"（≤1 通过；共 {len(rows)} 点，n={n}；最差点 {worst_key}）")
    print(f"[30] S_s3 退化：periodic(T=1) tx_rate−1 = {deg_T1:.3e}；"
          f"threshold(K=0) tx_rate−1 = {deg_K0:.3e}")
    if worst > 1.0:
        raise AssertionError(
            f"★ S_s1 失败：{worst_key} 的 |MC−闭式| 超过 4σ（{worst:.3f}×）"
            " ⇒ 闭式与实现确实不一致，先别往下跑（分根因：闭式错 / 调度器错 / 口径错）")
    if deg_T1 > 1e-12 or deg_K0 > 1e-12:
        raise AssertionError(f"★ S_s3 失败：退化不成立（{deg_T1:.3e} / {deg_K0:.3e}）")
    return {"worst_ratio_over_5sigma": worst, "worst_key": worst_key,
            "k_sigma": K_SIGMA, "n_points": len(rows),
            "deg_periodic_T1": deg_T1, "deg_threshold_K0": deg_K0}


# ------------------------------------------------------------------ Part 2：扫描
def sweep(n: int, seed: int, loss_probs: list[float], tmax: int) -> list[dict]:
    """信号无关三族的旋钮扫描 ⇒ 统一 `scheduling.Row`。"""
    rows: list[dict] = []
    for p in loss_probs:
        for T in range(1, int(tmax) + 1):
            r = scheduling.simulate_schedule(
                scheduling.build("periodic", loss_prob=p, period=T), n, seed)
            rows.append(dict(family="periodic", knob_axis="period", knob_value=T,
                             loss_prob=p, seed=seed, tx_rate=r["tx_rate"],
                             mean_age=r["mean_age"], metric="mean_age",
                             metric_value=r["mean_age"], n_eval=1))
        for K in range(0, int(tmax) + 1):
            r = scheduling.simulate_schedule(
                scheduling.build("threshold", loss_prob=p, threshold=K), n, seed)
            rows.append(dict(family="threshold", knob_axis="threshold", knob_value=K,
                             loss_prob=p, seed=seed, tx_rate=r["tx_rate"],
                             mean_age=r["mean_age"], metric="mean_age",
                             metric_value=r["mean_age"], n_eval=1))
    # timing：固定 PER=0（X44 的刻意选择：有丢包时时刻序列会消耗不同随机数流 ⇒ 虚假相关）
    T0 = 8
    N0 = n_tx_budget(T0, n)
    for mode, params in (("phase", list(range(T0))),
                         ("jitter", list(range((T0 - 1) // 2 + 1))),
                         ("random", list(range(4)))):
        for prm in params:
            s = scheduling.build("timing", loss_prob=0.0, mode=mode, period=T0,
                                 param=prm, n_steps=n, n_tx=N0)
            r = scheduling.simulate_schedule(s, n, seed)
            ts = plan_times(mode, T0, prm, n, N0)
            rows.append(dict(family="timing", knob_axis="param", knob_value=prm,
                             loss_prob=0.0, seed=seed, tx_rate=r["tx_rate"],
                             mean_age=r["mean_age"], metric="mean_age",
                             metric_value=r["mean_age"], n_eval=1,
                             extra={"mode": mode, "T": T0, "n_tx": r["n_tx"],
                                    "cv_gap": timing_dispersion(ts)}))
    return rows


def to_rows(rows: list[dict]) -> list[scheduling.Row]:
    out = []
    for d in rows:
        d = {k: v for k, v in d.items() if k != "extra"}
        sp = scheduling.REGISTRY[d["family"]]
        out.append(scheduling.Row(deployable=sp.deployable,
                                  diagnostic=sp.diagnostic, **d))
    return out


# ------------------------------------------------------------------ Part 3：等预算
def pair_periodic_threshold(rows: list[dict], p: float) -> dict | None:
    """S_s4：`threshold` vs `periodic` 在公共发送率窗口上的等预算比较。

    ★ 比值口径：`ratio = E[age](threshold) / E[age](periodic)` ⇒ **< 1 表示阈值更优**。
      （参数顺序就是结论方向：把 threshold 传成 `a`、periodic 传成 `b`。
        反了不会报错，只会把结论标签写反 —— 这类"方向标签"错误在本仓库出现过，
        所以在打印里把分子分母**写死成字符串**，不靠读者脑补。）
    ★ 已知答案（PER=0）：两族在等预算下 **E[age] 恒等 ⇒ ratio ≡ 1.0000**
      （无丢包时"失败后重试"没有机会发生 ⇒ 阈值策略退化为确定周期）。
      这条既是自检、也是 X38「PER=0 恒等」的独立复现。
    """
    a = to_rows([r for r in rows if r["family"] == "threshold"
                 and abs(r["loss_prob"] - p) < 1e-12])
    b = to_rows([r for r in rows if r["family"] == "periodic"
                 and abs(r["loss_prob"] - p) < 1e-12])
    try:
        res = scheduling.equal_budget_pair(a, b, metric="mean_age")
    except pairing.WindowNotEvaluable as e:
        # ★ 不许静默 continue —— 静默会被读成"这一档没问题"（pairing.py 的设计意图）
        print(f"[30] S_s4 p={p}：窗口不可评估 ⇒ **不产出比值**。原因：{e}")
        return None
    print(f"[30] S_s4 p={p}：公共窗口 tx_rate∈[{res['lo']:.4f},{res['hi']:.4f}]"
          f"（{res['hi'] / res['lo']:.2f}×；原始窗口 [{res['lo_raw']:.4f},"
          f"{res['hi_raw']:.4f}]）"
          f"  E[age] 比值中位 threshold/periodic = {res['ratio_median']:.4f}"
          f"（<1 ⇒ 阈值更优；网格 {res['n_grid_used']}/{res['n_grid_raw']} 点，"
          f"n_a={res['n_a']} n_b={res['n_b']}）")
    if p == 0.0 and abs(res["ratio_median"] - 1.0) > 1e-9:
        raise AssertionError(
            f"★ 已知答案失败：PER=0 时两族等预算 E[age] 必须**恒等**（ratio≡1），"
            f"实测 {res['ratio_median']:.6f}（⇒ 先查『等预算配对』的口径，别急着解释差异）")
    return res


# ------------------------------------------------------------------ Part 4：图
def make_figure(rows: list[dict], pairs: dict, outdir, tag: str):
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))

    # (a) E[age] vs 发送率：两族 + 闭式虚线
    ax = axes[0]
    _cols = (PALETTE["blue"], PALETTE["red"], PALETTE["green"])
    for p, col in zip((0.0, 0.3, 0.7), _cols):
        for fam, mk, ls in (("periodic", "o", "-"), ("threshold", "s", "--")):
            rs = sorted([r for r in rows if r["family"] == fam
                         and abs(r["loss_prob"] - p) < 1e-12],
                        key=lambda r: r["tx_rate"])
            xs = [r["tx_rate"] for r in rs]
            ys = [r["mean_age"] for r in rs]
            ax.plot(xs, ys, ls, marker=mk, ms=4, lw=1.5, color=col,
                    label=f"{fam} p={p}")
        for T in range(1, 17):
            ax.plot(1.0 / T, periodic_mean_age(p, T), "x", ms=4,
                    color=col, alpha=0.35)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("发送率 tx_rate（每步送达概率）")
    ax.set_ylabel("E[age]（步）")
    ax.set_title("(a) 周期 vs 年龄阈值：E[age]–预算前沿\n(× = 闭式，与实测重合 ⇒ 接线正确)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7, ncol=2)

    # (b) 固定预算下的时机离散度 vs E[age]
    ax = axes[1]
    for mode, col in zip(("phase", "jitter", "random"),
                         (PALETTE["orange"], PALETTE["purple"], PALETTE["grey"])):
        rs = [r for r in rows if r["family"] == "timing"
              and r.get("extra", {}).get("mode") == mode]
        rs.sort(key=lambda r: r["extra"]["cv_gap"])
        xs = [r["extra"]["cv_gap"] for r in rs]
        ys = [r["mean_age"] for r in rs]
        ax.plot(xs, ys, "o-", ms=5, lw=1.5, color=col, label=f"{mode}")
    ax.set_xlabel("时机离散度 CV(相邻间隔)  ← 只吃『时机』，不吃『预算』")
    ax.set_ylabel("E[age]（步）")
    ax.set_title("(b) 固定预算 + PER=0：只动发送时刻\n（T=8；预算逐位相等 ⇒ 差异纯来自时机）")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # (c) 注册表（六族 × 可部署性）
    ax = axes[2]
    ax.axis("off")
    fams = list(scheduling.FAMILIES)
    ax.text(0.0, 1.0, "(c) 调度族注册表 v0.1", fontsize=12, va="top")
    y = 0.90
    for fam in fams:
        sp = scheduling.REGISTRY[fam]
        wired = fam not in scheduling.SIGNAL_DEPENDENT
        mark = "●" if wired else "○"
        tagl = "空跑已接通" if wired else "需闭环（本脚本不跑）"
        dep = "可部署" if sp.deployable else "不可部署·诊断用"
        ax.text(0.0, y, f"{mark} {fam}", fontsize=11, va="top",
                color="#3a7" if wired else "#a33")
        ax.text(0.36, y, f"旋钮 {sp.knob}", fontsize=9, va="top")
        ax.text(0.66, y, dep, fontsize=9, va="top",
                color="#333" if sp.deployable else "#a33")
        ax.text(0.0, y - 0.045, f"    {tagl}", fontsize=8, va="top", alpha=0.75)
        y -= 0.155
    ax.text(0.0, y, "★ 六族同一张表；oracle 在代码里被标为不可部署。",
            fontsize=8, va="top", alpha=0.8)

    fig.tight_layout()
    png = save_fig(fig, os.path.join(str(outdir), f"{tag}.png"))
    return png


# ------------------------------------------------------------------ 主流程
def main():
    args = parse_args()
    t0 = time.time()
    cfg = load_config(args.config)
    out = output_dir(cfg)
    print_registry()

    checks = scheduling.self_check(verbose=False, n=args.n, seed=args.seed)
    print("[30] scheduling.self_check()（库内置；含预算逐位相等）：")
    for k, v in checks.items():
        print(f"      {k}: 实测={v[0]:.6f} 参考={v[1]:.6f} 差={v[2]:.3e} "
              f"se={v[3]:.2e} 判据比值={scheduling.check_tolerance(v[2], v[3]):.3f}")
    bad = [k for k, v in checks.items()
           if scheduling.check_tolerance(v[2], v[3]) > 1.0]
    if bad:
        raise AssertionError(f"★ 库内置自检未通过（差 > 4σ 或构造等式不成立）：{bad}")

    closed = check_closed_forms(args.n, args.seed, args.loss_probs, args.tmax)
    rows = sweep(args.n, args.seed, args.loss_probs, args.tmax)
    pairs = {}
    for p in args.loss_probs:
        r = pair_periodic_threshold(rows, p)
        pairs[f"p={p}"] = r

    png = make_figure(rows, pairs, out, args.tag)
    jpath = os.path.join(str(out), f"{args.tag}.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump({"schema": "scheduling-v0.1",
                   "meta": {"n": args.n, "seed": args.seed, "tmax": args.tmax,
                            "loss_probs": args.loss_probs, "n_grid": len(rows),
                            "closed_form_check": closed,
                            "self_check": {k: list(v) for k, v in checks.items()},
                            "pairs": pairs,
                            "wired": [f for f in scheduling.FAMILIES
                                      if f not in scheduling.SIGNAL_DEPENDENT],
                            "not_wired": sorted(scheduling.SIGNAL_DEPENDENT)},
                   "rows": rows}, f, ensure_ascii=False, indent=2)
    print(f"[30] 产物：{png} / {jpath}  ({time.time() - t0:.1f}s)")
    print("[30] ⛔ 未接通（只注册）：" + ", ".join(sorted(scheduling.SIGNAL_DEPENDENT))
          + " —— 它们需要闭环 state；历史结果见 scripts/24..27。")


if __name__ == "__main__":
    main()
