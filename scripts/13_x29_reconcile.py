#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""脚本 13 —— **X29：δ 有分辨力之后的 C17 对账**（X22 判定的死路，第二次尝试）。

★ 与 X22 的关系
---------------
X22 证明：确定性设定下 δ ≡ 1 ⇒ H(ε,1) = 1+ε ⇒ 代入不成立 ⇒ "gap 不可定义"。
X28 找到了 δ 有分辨力的条件：**latent_dim=4**（D≥8 时 δ 饱和回 1）。
本脚本在 D=4 的条件下第一次把对账做出来。

★ 对账用的公式（X22 已核实到原文，此处沿用 scripts/08 的实现）

    H(ε,δ) = [ (1+ε) + sqrt((1−ε)² + 4ε/δ) ] / 2

    δ → 1 时退化为 1+ε（X22 的闭式，已用两条独立路径验证到 2.2e-16）；
    δ → 0 时 H → ∞（模型完美 ⇒ 视界无穷）—— 方向自洽。

★★ 必须带着走的断裂点声明（不许删）
------------------------------------
| # | 断裂点 | 内容 |
|---|---|---|
| ① | **measurand 不同** | C17 的 H 约束「策略序反转幅度 ≤ ε」（含回报、折扣 γ）；实测 H\* 是「预测误差首次超阈的步数」（不含策略与价值）⇒ **gap 数值只能定性解读（方向），不能当倍数引用** |
| ② | **空间不同** | C17 的 δ 定义在状态-动作空间；这里测的是**潜空间**转移 |
| ③ | **原文限定 finite MDP** | Pendulum 是连续状态 |

⇒ 本脚本报的每个 gap 都要读作：**"在 δ 可测的设定下，理论保证与实测能力之间
存在数量级差"**，而不是"理论界保守了 N 倍"。

★ 运行
------
    python scripts/13_x29_reconcile.py            # 读 X28 的 json，出表出图
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wmlab.utils import output_dir

ROOT = Path(__file__).resolve().parents[1]
EPS = 0.05


def H(eps: float, delta: float) -> float:
    """C17 safe horizon（X22 核实版，与 scripts/08 逐字一致）。"""
    if delta <= 0:
        return float("inf")
    return ((1.0 + eps) + math.sqrt((1.0 - eps) ** 2 + 4.0 * eps / delta)) / 2.0


def load(tag: str) -> dict:
    p = ROOT / "outputs" / f"{tag}.json"
    if not p.exists():
        raise FileNotFoundError(f"缺少 {p} —— 先跑 scripts/12_delta_continuity.py")
    return json.loads(p.read_text(encoding="utf-8"))


def main():
    print(f"[13] X29 · δ 有分辨力后的 C17 对账   ε = {EPS}")
    print("[13] ★ 断裂点声明：measurand（序反转 vs 误差超阈）/ 空间（状态 vs 潜空间）/"
          "原文限定 finite MDP —— gap 只能定性解读，不能当倍数引用")

    src = {
        "D=4 (delta resolvable)": load("12_delta_ldim4_full"),
        "D=32 (default, saturates)": load("12_delta_continuity"),
    }

    rows = []
    for name, s in src.items():
        by_sigma = {}
        for r in s["rows"]:
            by_sigma.setdefault(r["sigma_env"], []).append(r)
        for sigma, rs in sorted(by_sigma.items()):
            delta_max = float(np.mean([r["delta"] for r in rs]))
            delta_med = float(np.mean([r["tv_median"] for r in rs]))
            hs = [r["h_star"] for r in rs if r.get("h_star") is not None]
            h_star = float(np.mean(hs)) if hs else None
            rows.append({
                "config": name, "sigma_env": sigma,
                "delta_max": delta_max, "delta_median": delta_med,
                "H_max": H(EPS, delta_max), "H_median": H(EPS, delta_med),
                "h_star": h_star,
                "ratio_med": (h_star / H(EPS, delta_med))
                             if (h_star and math.isfinite(H(EPS, delta_med))) else None,
            })

    print()
    print(f"[13] {'config':<26s} {'σ_env':>6s} {'δ_max':>7s} {'δ_med':>7s} "
          f"{'H(ε,δ_med)':>10s} {'H*(实测)':>9s} {'H*/H':>7s}")
    for r in rows:
        fmt = lambda v: f"{v:7.4f}" if isinstance(v, float) else "      —"
        hm = f"{r['H_median']:10.2f}" if math.isfinite(r["H_median"]) else "       inf"
        hs = f"{r['h_star']:9.2f}" if r["h_star"] is not None else "      n/a"
        ra = f"{r['ratio_med']:7.1f}" if r["ratio_med"] else "      —"
        print(f"[13] {r['config']:<26s} {r['sigma_env']:6.2g} {fmt(r['delta_max'])} "
              f"{fmt(r['delta_median'])} {hm} {hs} {ra}")

    # ---------- 图 ----------
    out = output_dir({"output": {"dir": "outputs"}})
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    colors = {"D=4 (delta resolvable)": "#4C78A8",
              "D=32 (default, saturates)": "#E45756"}

    ax = axes[0]
    for name, _ in src.items():
        rs = [r for r in rows if r["config"] == name]
        ax.plot([r["sigma_env"] for r in rs], [r["delta_median"] for r in rs],
                "o-", color=colors[name], label=name)
    ax.axhline(1.0, color="k", ls=":", lw=1)
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xlabel("process-noise sigma_env"); ax.set_ylabel("delta (median TV)")
    ax.set_title("(1) delta: resolvable only at latent_dim=4")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

    ax = axes[1]
    for name, _ in src.items():
        rs = [r for r in rows if r["config"] == name and r["h_star"]]
        if not rs:
            continue
        xs = [r["sigma_env"] for r in rs]
        ax.plot(xs, [r["h_star"] for r in rs], "s-", color=colors[name],
                label=f"H* measured ({name.split()[0]})")
        ax.plot(xs, [min(r["H_median"], 200) for r in rs], "^--", color=colors[name],
                alpha=0.7, label=f"H(eps,delta) theory ({name.split()[0]})")
    ax.set_yscale("log")
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xlabel("process-noise sigma_env")
    ax.set_ylabel("horizon (steps)")
    ax.set_title("(2) theory vs measurement\n(measurand mismatch: qualitative only)")
    ax.legend(fontsize=7); ax.grid(alpha=0.25)

    fig.suptitle(f"wmlab X29 · C17 reconcile after delta made resolvable · eps={EPS}",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    png = out / "13_x29_reconcile.png"
    fig.savefig(png, dpi=150)
    plt.close(fig)

    (out / "13_x29_reconcile.json").write_text(
        json.dumps({"eps": EPS, "rows": rows,
                    "caveats": ["measurand mismatch", "state-vs-latent space",
                                "finite-MDP assumption"]},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[13] 图 -> {png}")
    print("[13] OK")


if __name__ == "__main__":
    main()
