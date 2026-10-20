#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P3-3b · 论文图（2×2）：把 P3-0/P3-1/P3-2 的三块证据合成一张，**只从入库 JSON 重画**。

四格
----
(a) 适用域：`R_uav` vs 实测陈旧度 `x`，`paced` / `unpaced` 两支分开，交点区间加阴影。
(b) 绝对代价：同一批点上 `J_uav`（Linear 与 ZOH 两条），log 纵轴 —— 说明"比值趋平 ≠ 危害趋平"。
(c) 窗口交点：三档失效窗口的 `x*` 与区间（误差棒 = 交点所在相邻两点，**不是统计置信区间**）。
(d) `x*/W` vs `W`：比例参考线（过原点、锚在基线）+ 三档实测点。

数据入口与 `scripts/61_build_tables.py` **完全相同**（`results_io.latest_result_path`），
不硬编码任何数字 ⇒ 证据变了图会跟着变。

用法：
    D:/devtools/miniconda3/envs/ai-lab/python.exe scripts/61c_plot_paper_fig.py
产物：
    assets/paper_p3.png
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


T61 = _load(ROOT / "scripts" / "61_build_tables.py", "t61_mod")

C_UNPACED = "#c0392b"
C_PACED = "#2b7bba"
C_WIN = {"short": "#2b7bba", "base": "#444444", "long": "#c0392b"}
M_WIN = {"short": "o", "base": "s", "long": "^"}


def main() -> int:
    data, meta = T61.load_all()
    for k in ("59", "60"):
        if k not in data:
            raise SystemExit(f"缺 {k} 的入库结果，画不了")
    d59, d60 = data["59"], data["60"]
    ref59 = d59.get("points", []) + (data["59b"].get("points", []) if "59b" in data else [])

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.2))
    ax = axes[0][0]

    # ---------------- (a) 适用域 ----------------
    for mode, color, lab in (("unpaced", C_UNPACED, "unpaced (budget exhausted at $t=B$)"),
                             ("paced", C_PACED, "paced (budget spread over window)")):
        pts = sorted((p for p in ref59 if p["mode"] == mode), key=lambda p: p["x"])
        xs = [p["x"] for p in pts]
        ys = [p["R_uav"] for p in pts]
        lo = [p["ci_uav"][0] for p in pts]
        hi = [p["ci_uav"][1] for p in pts]
        ax.plot(xs, ys, "-o", color=color, ms=4.5, lw=1.5, label=lab)
        ax.fill_between(xs, lo, hi, color=color, alpha=0.16, lw=0)
    for c in d59["F2"]["unpaced"]["crossings"]:
        ax.axvspan(c["between"][0], c["between"][1], color=C_UNPACED, alpha=0.09, lw=0)
        ax.axvline(c["x"], color=C_UNPACED, ls=":", lw=1.1)
        ax.text(0.03, 0.50,
                f"crossing bracket ({c['between'][0]:.2f}, {c['between'][1]:.2f})\n"
                "= gap between grid points, not a CI",
                transform=ax.transAxes, fontsize=7, color=C_UNPACED, va="top",
                bbox=dict(fc="white", ec="none", alpha=0.75, pad=2.0))
    ax.axhline(1.0, color="k", lw=1.0, ls="--")
    ax.set_xlabel("measured staleness  $x$ = mean UAV age (steps)")
    ax.set_ylabel("$R_{uav}=J_{uav}(\\mathrm{Linear})/J_{uav}(\\mathrm{ZOH})$")
    ax.set_title("(a) compensation benefit vs measured staleness", fontsize=10)
    ax.legend(fontsize=7.5, loc="lower right", frameon=False)
    ax.grid(alpha=0.25, lw=0.5)

    # ---------------- (b) 绝对代价 ----------------
    ax = axes[0][1]
    for mode, ls in (("unpaced", "-"), ("paced", "--")):
        pts = sorted((p for p in ref59 if p["mode"] == mode), key=lambda p: p["x"])
        ax.plot([p["x"] for p in pts], [p["J"][0] for p in pts], ls, color=C_UNPACED,
                marker="o", ms=3.5, lw=1.3, label=f"$J_{{uav}}$ Linear, {mode}")
        ax.plot([p["x"] for p in pts], [p["J_zoh"][0] for p in pts], ls, color="#777777",
                marker="x", ms=3.5, lw=1.0, alpha=0.85, label=f"$J_{{uav}}$ ZOH, {mode}")
    ax.set_yscale("log")
    ax.set_xlabel("measured staleness  $x$ (steps)")
    ax.set_ylabel("$J_{uav}$  (m, log scale)")
    ax.set_title("(b) the ratio flattens, the absolute cost does not", fontsize=10)
    ax.legend(fontsize=6.6, loc="upper left", frameon=False)
    ax.grid(alpha=0.25, lw=0.5, which="both")

    # ---------------- (c) 窗口交点 ----------------
    ax = axes[1][0]
    for w in ("short", "base", "long"):
        br = d60["brackets"][w]
        W = d60["windows"][w]["W_expected"]
        ax.errorbar([W], [br["x_star"]],
                    yerr=[[br["x_star"] - br["bracket"][0]], [br["bracket"][1] - br["x_star"]]],
                    fmt=M_WIN[w], color=C_WIN[w], ms=8, capsize=6, lw=1.6,
                    label=f"$R_{{fail}}$={d60['windows'][w]['radius']:.0f} m, $W$={W}")
    ax.set_xlabel("measured failure window $W$ (steps without update)")
    ax.set_ylabel("crossing $x^*$ (mean age where $R_{uav}$ passes 1)")
    ax.set_title("(c) crossing vs window — intervals overlap ($\\it{INCONCLUSIVE}$)", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper left", frameon=False)
    ax.grid(alpha=0.25, lw=0.5)

    # ---------------- (d) x*/W ----------------
    ax = axes[1][1]
    Ws = [d60["windows"][w]["W_expected"] for w in ("short", "base", "long")]
    rs = [d60["x_star_over_W"][w]["x_star_over_W"] for w in ("short", "base", "long")]
    los = [d60["x_star_over_W"][w]["bracket_over_W"][0] for w in ("short", "base", "long")]
    his = [d60["x_star_over_W"][w]["bracket_over_W"][1] for w in ("short", "base", "long")]
    for w, W, r, lo, hi in zip(("short", "base", "long"), Ws, rs, los, his):
        ax.errorbar([W], [r], yerr=[[r - lo], [hi - r]], fmt=M_WIN[w], color=C_WIN[w],
                    ms=8, capsize=6, lw=1.6, label=f"$W$={W}")
    anchor = rs[1]
    ax.plot([0, max(Ws) * 1.1], [anchor, anchor], color="#888888", ls=":",
            lw=1.2, label="proportional to window (anchor = baseline)")
    ax.set_xlim(0, max(Ws) * 1.1)
    ax.set_ylim(0, max(rs) * 1.45)
    ax.set_xlabel("measured failure window $W$ (steps)")
    ax.set_ylabel("$x^*/W$")
    ax.set_title("(d) the threshold does not scale with the window", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper right", frameon=False)
    ax.grid(alpha=0.25, lw=0.5)

    fig.suptitle("Prediction-compensation safety domain under bandwidth limits "
                 "(dev-set scan; cluster bootstrap on reset initial states)", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = ROOT / "assets" / "paper_p3.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=170)
    print(f"[out] {out.relative_to(ROOT).as_posix()}")
    print("[data]", {k: meta[k]["path"] for k in ("59", "59b", "60") if k in meta})
    return 0


if __name__ == "__main__":
    sys.exit(main())
