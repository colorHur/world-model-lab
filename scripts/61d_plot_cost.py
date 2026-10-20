#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P3-3c · 资源与耗时图：`assets/paper_p3_cost.png`（只从入库 JSON 重画）。

为什么单独画
------------
v2 §P3 要"资源与耗时图"，而 `58` 里只有 `TIMING`（每次调用中位 µs + 调用数）与
`cells_J[*].elapsed_s`（每 cell 墙钟）两列数字。单看 µs 会被误读成"预测器很便宜所以可以随便加"，
所以图必须**两格一起给**：单次调用开销 vs 整轮墙钟，后者由环境步进主导。

两格
----
(a) 每次调用中位耗时（µs，条形）+ 调用数 —— 只用于比较**相对**开销。
(b) 每 cell 墙钟时间（按 预算×时序 分组，逐 cell 散点）—— 说明预测器开销在整轮里是小量。

★ 图上强制标注「Python 层实现开销，不是实时预算结论」（与源 JSON 的 `note` 同字面）。

用法：
    D:/devtools/miniconda3/envs/ai-lab/python.exe scripts/61d_plot_cost.py
产物：
    assets/paper_p3_cost.png
不写 `results/`（只读证据）。
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


T61 = _load(ROOT / "scripts" / "61_build_tables.py", "t61_cost")

C_ZOH = "#777777"
C_LIN = "#c0392b"
NOTE = "Python-level implementation overhead — not a real-time budget statement"


def main() -> int:
    data, meta = T61.load_all()
    if "58" not in data:
        raise SystemExit("缺 58 的入库结果，画不了")
    d = data["58"]
    cells = d["cells_J"]
    tm = d["TIMING"]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.1))

    # ---------------- (a) 每次调用开销 ----------------
    ax = axes[0]
    names, meds, ncall = [], [], []
    for pred, lab in (("zoh", "ZOH"), ("linear", "Linear")):
        names.append(lab)
        meds.append(tm[f"primary/{pred}"]["median_us"])
        ncall.append(tm[f"primary/{pred}"]["n_calls"])
    bars = ax.bar(names, meds, color=[C_ZOH, C_LIN], width=0.55)
    for b, v, n in zip(bars, meds, ncall):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f} µs\nn={n}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, max(meds) * 1.35)
    ax.set_ylabel("median per-call cost (µs)")
    ax.set_title("(a) predictor cost per decision step", fontsize=10)
    ax.grid(alpha=0.25, lw=0.5, axis="y")
    ax.text(0.5, -0.22, NOTE, transform=ax.transAxes, ha="center", va="top",
            fontsize=8, color="#c0392b")

    # ---------------- (b) 每 cell 墙钟 ----------------
    ax = axes[1]
    groups = [(200, "unpaced"), (100, "unpaced"), (100, "paced")]
    labels = [f"B={b}\n{m}" for b, m in groups]
    for k, (pred, color, off) in enumerate((("zoh", C_ZOH, -0.13), ("linear", C_LIN, 0.13))):
        for gi, (b, m) in enumerate(groups):
            vals = [c["elapsed_s"] for c in cells
                    if c["predictor"] == pred and c["seed_set"] == "primary"
                    and c["budget"] == b and c["mode"] == m]
            if not vals:
                continue
            xs = [gi + off] * len(vals)
            ax.scatter(xs, vals, s=26, color=color, alpha=0.85,
                       label=("ZOH" if k == 0 else "Linear") if gi == 0 else None, zorder=3)
            mean = sum(vals) / len(vals)
            ax.hlines(mean, gi + off - 0.10, gi + off + 0.10, color=color, lw=2.0, zorder=4)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("wall-clock per cell (s), 30 episodes")
    ax.set_title("(b) whole-cell cost — the environment dominates", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.grid(alpha=0.25, lw=0.5, axis="y")

    fig.suptitle("Resource cost of prediction compensation (confirmation run, single core CPU)",
                 fontsize=9.5)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    out = ROOT / "assets" / "paper_p3_cost.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=170)
    print(f"[out] {out.relative_to(ROOT).as_posix()}")
    print(f"[data] {meta['58']['path']}")
    print(f"[note] {NOTE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
