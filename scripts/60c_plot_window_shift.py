#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P3-2 主图：失效窗口改变后，`R_uav`–陈旧度曲线与交点是否平移。

数据来源（只读，不改任何证据）：
    results/.../60_p3_window_shift.json   三档窗口 × 8 预算 × 2 预测器 × 3 种子 = 144 cell

左：三档窗口的 `R_uav`–`x` 曲线（`x` = UAV `age` 均值，实测），阴影 = 交点相邻区间。
右：交点区间 vs **实测失效窗口**；虚线 = "按窗口成比例"的参考（过原点、锚在基线窗口）。
★ 只画点估计与 95% cluster bootstrap 区间；不做拟合曲线之外的任何平滑。
★ 标签用英文（图要能进论文）；解释写在结果文档里。

用法：
    D:/devtools/miniconda3/envs/ai-lab/python.exe scripts/60c_plot_window_shift.py
产物：
    assets/60_window_shift.png
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

STYLE = {"short": ("#2b7bba", "o", "R_fail=16 m  (W=35)"),
         "base": ("#444444", "s", "R_fail=25 m  (W=53)"),
         "long": ("#c0392b", "^", "R_fail=35 m  (W=104)")}


def main() -> int:
    p = latest_result_path("60_p3_window_shift")
    if p is None or not p.exists():
        raise SystemExit("找不到 60_p3_window_shift 的入库结果")
    print(f"[in] {p.relative_to(ROOT).as_posix()}")
    d = json.loads(p.read_text(encoding="utf-8"))
    summ = d["summary"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.7))

    # ---------------- 左：R – x ----------------
    for w, (c, m, lab) in STYLE.items():
        pts = sorted(summ[w]["points"], key=lambda r: r["x"])
        xs = [r["x"] for r in pts]
        ys = [r["R_uav"] for r in pts]
        lo = [r["ci_uav"][0] for r in pts]
        hi = [r["ci_uav"][1] for r in pts]
        ax1.plot(xs, ys, "-" + m, color=c, ms=5, lw=1.4, label=lab)
        ax1.fill_between(xs, lo, hi, color=c, alpha=0.13, lw=0)
        br = summ[w]["bracket"]
        if br["kind"] == "BRACKETED":
            ax1.axvspan(br["bracket"][0], br["bracket"][1], color=c, alpha=0.10, lw=0)
            ax1.plot([br["x_star"]], [1.0], marker="|", color=c, ms=14, mew=2)
    ax1.axhline(1.0, color="k", lw=1.0, ls="--")
    ax1.set_xlabel("measured staleness  $x$ = mean UAV age  (steps)")
    ax1.set_ylabel("$R_{uav}$ = $J_{uav}$(Linear) / $J_{uav}$(ZOH)")
    ax1.set_title("(a) extrapolation harm vs staleness, three failure windows", fontsize=10)
    ax1.legend(fontsize=8, loc="lower right", frameon=False)
    ax1.grid(alpha=0.25, lw=0.5)

    # ---------------- 右：交点区间 vs 实测窗口 ----------------
    Ws, xs, elo, ehi, labs = [], [], [], [], []
    for w in ("short", "base", "long"):
        br = summ[w]["bracket"]
        Ws.append(summ[w]["window_expected"])
        xs.append(br["x_star"])
        elo.append(br["x_star"] - br["bracket"][0])
        ehi.append(br["bracket"][1] - br["x_star"])
        labs.append(f"{br['x_star'] / summ[w]['window_expected']:.3f}")
    for w, (c, m, lab) in STYLE.items():
        br = summ[w]["bracket"]
        ax2.errorbar([summ[w]["window_expected"]], [br["x_star"]], yerr=[
            [br["x_star"] - br["bracket"][0]], [br["bracket"][1] - br["x_star"]]],
            fmt=m, color=c, ms=8, capsize=6, lw=1.6, label=lab)
        ax2.annotate(f"$x^*$/W = {br['x_star'] / summ[w]['window_expected']:.3f}",
                     (summ[w]["window_expected"], br["x_star"]), textcoords="offset points",
                     xytext=(-8, 8) if w == "long" else (6, 8), fontsize=8, color=c,
                     ha="right" if w == "long" else "left")
    # 参考：按窗口成比例（过原点、锚在基线）
    slope = xs[1] / Ws[1]
    ax2.plot([0, max(Ws) * 1.15], [0, slope * max(Ws) * 1.15], color="#888888", ls=":",
             lw=1.2, label="proportional to window (anchor = baseline)")
    ax2.set_xlim(0, max(Ws) * 1.15)
    ax2.set_ylim(0, max(xs) * 1.45)
    ax2.set_xlabel("measured failure window $W$  (steps without any update)")
    ax2.set_ylabel("crossing $x^*$  (mean age where $R_{uav}$ passes 1)")
    ax2.set_title("(b) crossing moves far less than proportionally", fontsize=10)
    ax2.legend(fontsize=8, loc="upper left", frameon=False)
    ax2.grid(alpha=0.25, lw=0.5)

    fig.tight_layout()
    out = ROOT / "assets" / "60_window_shift.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=170)
    print(f"[out] {out.relative_to(ROOT).as_posix()}")
    print("verdict:", d["F1p"]["verdict"], "|", d["F1p"]["reason"])
    for w in ("short", "base", "long"):
        br = summ[w]["bracket"]
        print(f"  {w:5s} W={summ[w]['window_expected']:3d} bracket={br['bracket']} x*={br['x_star']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
