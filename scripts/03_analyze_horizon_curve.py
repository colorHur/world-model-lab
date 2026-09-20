#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""脚本 03 —— 从 02 产出的 json 里，把「误差增长的形态」算清楚。

为什么需要这个脚本
------------------
02 的图②画的是 NMSE 随视界增长的**曲线本身**，图④画的是每步增量。
但要回答「误差是线性累积、自我放大、还是饱和」，光看图不够 —— 需要看
**每步增量的峰值出现在哪、峰值之后是继续涨还是掉头**。

上一轮（2026-09-17）的结论「误差自我放大」是**看图得出的**，且用了错误口径
（`np.diff(nm)` 对不等间隔的 horizons 直接差分）。这个脚本把正确的口径固定下来：

    inc(h) = ΔNMSE / Δhorizon        # 相邻视界点之间的斜率，已按步数归一

★ 2026-09-20 口径修正（X26 对账发现）
------------------------------------
序列本身改用 **逐点**口径（第 h 步自身的误差，`nmse_per_step`），不再用累积口径。
累积口径是逐点的运行平均，斜率会被"分母在变大"扭曲 —— 它反映的是平滑化过程，
不是误差增长本身。旧格式 JSON 无该字段时回落到累积并在输出里标明 `cumulative(fallback)`。

并给出形态判定（三选一）：
    - monotone_up : 增量一路上升（自我放大 / 发散）
    - saturating  : 增量先升后降，末端趋平（饱和，存在有界特征时间 H*）
    - linear      : 增量基本恒定（线性累积）

运行
----
    python scripts/03_analyze_horizon_curve.py outputs/02_world_model_pendulum.json
    python scripts/03_analyze_horizon_curve.py a.json b.json --csv out.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def per_step(nmse: list[float], horizons: list[float]) -> list[float]:
    """相邻视界点之间的斜率：ΔNMSE / Δhorizon（按步数归一，消除不等间隔的影响）。"""
    return [(nmse[i + 1] - nmse[i]) / (horizons[i + 1] - horizons[i])
            for i in range(len(nmse) - 1)]


def shape(inc: list[float]) -> tuple[str, str]:
    """判定增量序列的形态。返回 (标签, 一句依据)。"""
    k = len(inc)
    if k < 3:
        return "undetermined", f"只有 {k} 个区间，不足以判定形态"
    peak = max(range(k), key=lambda i: inc[i])
    tail = inc[peak:]
    # 峰值之后的最大回撤比例
    drop = (max(tail) - min(tail)) / max(tail) if max(tail) > 0 else 0.0
    # 首段（峰值之前）的上升倍数
    head_gain = inc[peak] / inc[0] if inc[0] > 0 else float("inf")

    if peak == k - 1:
        return "monotone_up", (f"峰值落在最后一段区间（第 {peak + 1}/{k} 段），"
                               f"全程上升 {head_gain:.1f}× —— 未见饱和")
    if drop >= 0.15:
        return "saturating", (f"峰值在第 {peak + 1}/{k} 段，之后最大回撤 {drop:.0%}"
                              f"（上升 {head_gain:.1f}× 后掉头）—— 误差增长饱和")
    if head_gain < 1.5:
        return "linear", f"全程增量变化 < 1.5×（{head_gain:.2f}×）—— 近似线性累积"
    return "saturating_mild", (f"峰值在第 {peak + 1}/{k} 段，之后回撤仅 {drop:.0%}"
                               f"（上升 {head_gain:.1f}×）—— 趋于饱和但未明显掉头")


def analyze(js: Path) -> dict:
    d = json.loads(js.read_text(encoding="utf-8"))
    out = {"file": js.name, "env": d["config"]["env"]["id"],
           "H_open": d["reliable_horizon_open"],
           "H_closed": d["reliable_horizon_closed"],
           "H_open_cum": d.get("reliable_horizon_open_cumulative"),
           "H_closed_cum": d.get("reliable_horizon_closed_cumulative"),
           "calibration": d.get("nmse_calibration", "cumulative (旧格式 JSON，无该字段)"),
           "threshold": d["threshold"], "curves": {}}
    for key, label in (("curve_open_loop", "open"), ("curve_closed_loop", "closed")):
        c = d[key]
        hs = [float(h) for h in c["horizons"]]
        # ★ 口径修正（2026-09-20）：优先用**逐点**序列（第 h 步自身的误差）。
        #   逐点序列本身就是"每步有多准"，相邻点斜率 = 误差增长是否在加速。
        #   旧格式 JSON 没有该字段时回落到累积口径，并在输出里标明。
        if "nmse_per_step" in c:
            ps_dense = [float(v) for v in c["nmse_per_step"]]
            series = [ps_dense[h - 1] for h in c["horizons"]]
            source = "per_step"
        else:
            series = [float(v) for v in c["nmse"]]
            source = "cumulative(fallback)"
        inc = per_step(series, hs)
        lab, why = shape(inc)
        out["curves"][label] = {
            "source": source,
            "horizons": hs, "nmse": series, "inc": inc,
            "nmse_cumulative": [float(v) for v in c["nmse"]],
            "mid": [0.5 * (hs[i] + hs[i + 1]) for i in range(len(hs) - 1)],
            "shape": lab, "why": why,
            "peak_mid": 0.5 * (hs[inc.index(max(inc))] + hs[inc.index(max(inc)) + 1]),
            "nan": any(v != v or v in (float("inf"), float("-inf")) for v in series + inc),
        }
    return out


def main():
    p = argparse.ArgumentParser(description="分析误差曲线的增长形态")
    p.add_argument("jsons", nargs="+", help="02 脚本产出的 json 路径")
    p.add_argument("--csv", default=None, help="可选：把每步增量表导出为 csv")
    args = p.parse_args()

    rows = []
    for j in args.jsons:
        a = analyze(Path(j))
        print("=" * 74)
        print(f"{a['file']}   env={a['env']}   threshold={a['threshold']}")
        print(f"  H* 开环 = {a['H_open']}    H* 闭环 = {a['H_closed']}   "
              f"（口径：{a['calibration']}）")
        if a["H_open_cum"] is not None or a["H_closed_cum"] is not None:
            print(f"  累积口径对照：H* 开环 = {a['H_open_cum']}    "
                  f"H* 闭环 = {a['H_closed_cum']}   ← 系统性高估，仅作历史对照")
        for label in ("open", "closed"):
            c = a["curves"][label]
            name = "开环" if label == "open" else "闭环"
            print(f"\n  [{name}] 序列口径={c['source']}   形态 = {c['shape']}   "
                  f"NaN/Inf = {c['nan']}")
            print(f"    {c['why']}")
            print("    horizon区间中点   每步ΔNMSE")
            for mid, v in zip(c["mid"], c["inc"]):
                bar = "#" * max(1, int(38 * v / max(c["inc"])))
                print(f"    {mid:8.1f}   {v:.3e}  {bar}")
                rows.append([a["file"], a["env"], label, mid, v])
        print()

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["file", "env", "loop", "horizon_mid", "nmse_inc_per_step"])
            w.writerows(rows)
        print(f"csv 已写出 -> {args.csv}")


if __name__ == "__main__":
    main()
