#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P3-1 主图：`R_uav` 对实测刷新陈旧度的失配边界（+ 绝对代价面板）。

数据来源（只读，不改任何证据）：
    results/.../59_p3_boundary.json           主轮 96 cell（6 预算 × 2 模式 × 2 预测器 × 3 种子 + 确认集）
    results/.../59b_p3_boundary_refine.json   补点 18 cell（unpaced B∈{160,175,190}）

★ 只画点估计与 95% cluster bootstrap 区间；不做任何拟合曲线（预注册 §6：这是刻画，不是检验）。
★ 标签用英文（图要能进论文）；说明写在结果文档里。

用法：
    D:/devtools/miniconda3/envs/ai-lab/python.exe scripts/59c_plot_p3_boundary.py
产物：
    assets/59_p3_boundary.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from wmlab.eval.results_io import latest_result_path  # noqa: E402


def _load(name: str) -> dict:
    p = latest_result_path(name)
    if p is None or not p.exists():
        raise SystemExit(f"找不到 {name} 的入库结果")
    print(f"[in] {p.relative_to(ROOT).as_posix()}")
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> int:
    main59 = _load("59_p3_boundary")
    ref59b = _load("59b_p3_boundary_refine")

    pts = list(main59["points"])
    fine = list(ref59b["points"])
    for r in fine:
        r["_fine"] = True

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), gridspec_kw={"width_ratios": [1.15, 1]})

    # ---------------- Panel A：R_uav vs x ----------------
    ax = axes[0]
    style = {"unpaced": dict(marker="o", color="#c0392b", label="unpaced (budget spent early)"),
             "paced": dict(marker="^", color="#1f6fb4", label="paced (spread over window)")}
    for mode, st in style.items():
        sub = [r for r in pts if r["mode"] == mode]
        sub.sort(key=lambda r: r["x"])
        x = [r["x"] for r in sub]
        y = [r["R_uav"] for r in sub]
        lo = [r["ci_uav"][0] for r in sub]
        hi = [r["ci_uav"][1] for r in sub]
        ax.errorbar(x, y, yerr=[[yi - l for yi, l in zip(y, lo)],
                                [h - yi for yi, h in zip(y, hi)]],
                    fmt=st["marker"], color=st["color"], ms=7, lw=1.4, capsize=3,
                    label=st["label"], zorder=3)
        dy = 6 if mode == "paced" else -12
        for r in sub:
            ax.annotate(f"B={r['budget']}", (r["x"], r["R_uav"]), textcoords="offset points",
                        xytext=(4, dy), fontsize=7, color=st["color"])
    # 补点（只 unpaced）
    fx = sorted(fine, key=lambda r: r["x"])
    ax.errorbar([r["x"] for r in fx], [r["R_uav"] for r in fx],
                yerr=[[r["R_uav"] - r["ci_uav"][0] for r in fx],
                      [r["ci_uav"][1] - r["R_uav"] for r in fx]],
                fmt="s", color="#e67e22", ms=5.5, lw=1.1, capsize=2.5, zorder=2,
                label="unpaced, refinement points")
    for r in fx:
        ax.annotate(f"B={r['budget']}", (r["x"], r["R_uav"]), textcoords="offset points",
                    xytext=(5, -13), fontsize=7, color="#b9770e")

    # 交点区间（预注册 §6-F2 的 bracket：refine 后为 (5.83, 7.57)）
    x_lo, x_hi = 5.827, 7.571
    ax.axvspan(x_lo, x_hi, color="#999999", alpha=0.16, lw=0, zorder=0)
    ax.axhline(1.0, color="k", ls="--", lw=1.0, zorder=1)
    ax.text(x_hi * 1.04, 1.005, "R = 1 (no benefit)", fontsize=7.5, va="bottom")
    ax.text((x_lo * x_hi) ** 0.5, 0.36, "crossing\nbracket", fontsize=7.5,
            ha="center", color="#444444")

    ax.set_xscale("log")
    ax.set_xlabel("measured staleness  $x$  (mean UAV age at decision points, $t\\geq41$)")
    ax.set_ylabel("$R_{uav}$ = $\\Sigma J_{uav}$(Linear) / $\\Sigma J_{uav}$(ZOH)")
    ax.set_title("(a) Benefit flips sign with refresh staleness", fontsize=10.5)
    ax.grid(alpha=0.28, which="both", lw=0.5)
    ax.legend(fontsize=7.5, loc="upper left", framealpha=0.9)
    ax.set_ylim(0.25, 1.45)

    # ---------------- Panel B：绝对代价（unpaced）----------------
    ax2 = axes[1]
    un = [r for r in pts if r["mode"] == "unpaced"] + fx
    un.sort(key=lambda r: r["x"])
    ax2.plot([r["x"] for r in un], [r["J_zoh"][0] for r in un], "o-", color="#7f8c8d",
             ms=5, lw=1.3, label="$J_{uav}$ under ZOH (baseline)")
    ax2.plot([r["x"] for r in un], [r["J"][0] for r in un], "s-", color="#c0392b",
             ms=5, lw=1.3, label="$J_{uav}$ under Linear")
    for r in un:
        if r["budget"] in (60, 100, 150, 200):
            ax2.annotate(f"B={r['budget']}", (r["x"], r["J"][0]), textcoords="offset points",
                         xytext=(4, -11), fontsize=7, color="#c0392b")
    ax2.axvspan(x_lo, x_hi, color="#999999", alpha=0.16, lw=0, zorder=0)
    ax2.set_xscale("log")
    ax2.set_xlabel("measured staleness  $x$  (same axis as (a))")
    ax2.set_ylabel("$J_{uav}$  [m]  (window cost, lower is better)")
    ax2.set_title("(b) Ratio hides the absolute damage at low budget", fontsize=10.5)
    ax2.grid(alpha=0.28, which="both", lw=0.5)
    ax2.legend(fontsize=7.5, loc="upper left", framealpha=0.9)

    fig.suptitle("P3-1 mismatch boundary — frozen method: Linear predictor, arm = age, "
                 "30 episodes x 3 seeds (dev set)", fontsize=10.5, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.955))

    out = ROOT / "assets" / "59_p3_boundary.png"
    fig.savefig(out, dpi=170)
    print(f"[out] {out.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
