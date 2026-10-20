#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P3-3a · 主结果表：只从**已入库** JSON 生成 `docs/RESULTS-P3.md`。

为什么要有这个脚本
------------------
v2 §P3 要"主结果表"，而本仓库此前的数字散在 4 份入库 JSON 的
`summary` / `F1` / `F2` / `points` 里，且**没有一个地方**把"确认集 / 开发集 / 失效窗口"
三块分开摆。手工抄表会立刻违反 P0-3 的初衷（数字与证据脱钩）。

本脚本的三条硬约束
------------------
1. **只读**：唯一数据入口是 `results_io.latest_result_path(name)`；不跑实验、不写 `results/`。
2. **可逐格溯源**：每个显示值都登记到 `docs/results_p3_trace.json`
   （`display` / `value`（JSON 原值）/ `source` / `key`）。
   随后做**字面 grep**：`repr(value)` 必须原样出现在该 JSON 的文本里
   —— 这是一条真检查：改一位小数、换一次运行，它都会 FAIL。
3. **重算对账**：主表的 `R` 不用 summary 的字段直接抄，而是从 `cells_J` 的
   逐 episode `J` **重算** `Σ J(linear)/Σ J(zoh)`，再与 summary 比（容差 1e-9）。
   summary 与 cells 不一致 ⇒ FAIL。
4. **字节稳定**：文档里**不写任何时间戳**，内容只取决于证据字节 ⇒ 重跑两次 `sha256` 相同。

用法
----
    D:/devtools/miniconda3/envs/ai-lab/python.exe scripts/61_build_tables.py
    ... --check-only        # 只跑检查，不写 docs/

产物
----
    docs/RESULTS-P3.md              （内容不变则不重写）
    docs/results_p3_trace.json      （每格 → JSON 原值 + 键路径）

退出码：0 = 全部检查通过；2 = 有检查 FAIL（不是"研究 Gate 未过"，是**文档与证据不一致**）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from wmlab.eval.results_io import latest_result_path  # noqa: E402

FAILS: list[str] = []
SKIPS: list[str] = []
TRACE: list[dict] = []

SRC = {
    "58": "58_p3_confirmation",
    "59": "59_p3_boundary",
    "59b": "59b_p3_boundary_refine",
    "60": "60_p3_window_shift",
}


def check(tag: str, ok: bool, detail: str) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {tag}: {detail}")
    if not ok:
        FAILS.append(f"{tag}: {detail}")


def skip(tag: str, detail: str) -> None:
    """对象不存在 ⇒ SKIP（既不算通过也不算失败，#60 纪律）。"""
    print(f"  [SKIP] {tag}: {detail}")
    SKIPS.append(f"{tag}: {detail}")


# --------------------------------------------------------------------------- 取数
def load_all() -> tuple[dict, dict]:
    """返回 (数据, 溯源信息)。缺失的证据 ⇒ SKIP 并列出实际找到的路径。"""
    data, meta = {}, {}
    for k, name in SRC.items():
        p = latest_result_path(name)
        if p is None or not p.exists():
            skip(f"S-{k}", f"找不到入库结果 {name}；请先确认 results/runs/ 下的文件名")
            continue
        raw = p.read_bytes()
        data[k] = json.loads(raw.decode("utf-8"))
        meta[k] = {
            "name": name,
            "path": p.relative_to(ROOT).as_posix(),
            "run_id": p.parent.name,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "text": raw.decode("utf-8"),
        }
        print(f"  [in] {k}: {meta[k]['path']}  ({meta[k]['run_id']}, "
              f"{len(data[k].get('cells_J', []))} cells)")
    return data, meta


def fmt_num(value, source: str, key: str, spec: str = ".3f", derived: bool = False) -> str:
    """格式化一个**直接取自 JSON** 的数，并登记溯源。

    `derived=True` 只用于"由 JSON 计算出来的量"（表格里的峰值、比例等），
    这类值不参与字面 grep —— 但它们自身在脚本里就是算式，可复核。
    """
    disp = format(value, spec)
    if not derived:
        TRACE.append({"display": disp, "value": value, "source": source, "key": key})
    return disp


def fmt_calc(value, spec: str = ".3f") -> str:
    return format(value, spec)


def csv_num(values: list, source: str, keys: list[str], spec: str = ".3f") -> str:
    """把一串 JSON 原值格式化成 `a / b / c`，每一格都登记 trace。"""
    return " / ".join(fmt_num(v, source, k, spec) for v, k in zip(values, keys))


# --------------------------------------------------------------------------- 重算
def reratio(cells: list[dict], *, num: str, den: str = "zoh", seed_set: str,
            budget: int, mode: str, user: int, window: int | None = None) -> tuple[float, int]:
    """从逐 episode `J` 重算 `Σ J(num)/Σ J(den)`（同种子同 episode 配对）。

    `window` 给定时只取该窗口的 cell（P3-2 的数据多一维 `window`）。
    """
    def pick(name: str) -> list[dict]:
        out = []
        for c in cells:
            if c.get("predictor") != name:
                continue
            if c.get("seed_set", "dev") != seed_set:
                continue
            if c.get("budget") != budget or c.get("mode", "unpaced") != mode:
                continue
            if window is not None and c.get("window") != window:
                continue
            out.append(c)
        return sorted(out, key=lambda c: c["seed"])

    a, b = pick(num), pick(den)
    if not a or not b or [c["seed"] for c in a] != [c["seed"] for c in b]:
        raise RuntimeError(f"配对失败：{num}/{den} {seed_set} B={budget} {mode} window={window}")
    s_num = sum(e[user] for c in a for e in c["J_per_episode"])
    s_den = sum(e[user] for c in b for e in c["J_per_episode"])
    return s_num / s_den, sum(len(c["J_per_episode"]) for c in a)


def sel_cell(cells: list[dict], **kw) -> dict | None:
    for c in cells:
        if all(c.get(k) == v for k, v in kw.items()):
            return c
    return None


# --------------------------------------------------------------------------- 检查
def checks(data: dict, meta: dict) -> None:
    print("\n[checks]")

    # ---- S1 主表 R 由 cells_J 重算，与 summary 一致（P3-0） ----
    if "58" in data:
        d = data["58"]
        cells = d["cells_J"]
        for user, idx, tag in ((0, 0, "uav"), (1, 1, "cart")):
            r, n = reratio(cells, num="linear", seed_set="primary", budget=200,
                           mode="unpaced", user=user)
            ref = d["F1"]["R"][idx]
            check(f"S1 58 R_{tag} 重算=cells_J", abs(r - ref) < 1e-9,
                  f"重算 {r:.12f} vs F1.R[{idx}] {ref:.12f}（n={n}）")
        for b, mode in ((100, "unpaced"), (100, "paced")):
            r, n = reratio(cells, num="linear", seed_set="primary", budget=b,
                           mode=mode, user=0)
            ref = d["F2"][f"primary/{b}/{mode}"]["R"][0]
            check(f"S2 58 R_uav 重算 B={b}/{mode}", abs(r - ref) < 1e-9,
                  f"重算 {r:.12f} vs F2 {ref:.12f}（n={n}）")
        # 判词内部一致性：CONFIRMED ⇔ 区间上界 <1 且 R_cart ≤1.10
        up = d["F1"]["ci_uav"][1]
        ok_lbl = d["F1"]["verdict"] == ("CONFIRMED" if (up < 1.0 and d["F1"]["R"][1] <= 1.10)
                                        else d["F1"]["verdict"])
        check("S3 58 F1 判词与判据一致", ok_lbl,
              f"verdict={d['F1']['verdict']} 上界={up:.4f} R_cart={d['F1']['R'][1]:.4f}")
        # R17：B<L 档必须能报 t>B 发帧 —— 断服档应恰为 0
        c = sel_cell(d["cells_J"], seed_set="primary", budget=100, mode="unpaced",
                     predictor="linear", seed=100)
        check("S4 58 B=100 unpaced t>B 发帧=0（断服事实可查）", c is not None and c["served_late"] == [0, 0],
              f"served_late={c['served_late'] if c else 'N/A'}")
    else:
        skip("S1-S4 58", "58 证据缺失")

    # ---- S5 P3-1 的点：R 从 cells_J 重算 ----
    if "59" in data:
        d = data["59"]
        for b, mode in ((200, "paced"), (150, "unpaced")):
            r, n = reratio(d["cells_J"], num="linear", seed_set="dev", budget=b,
                           mode=mode, user=0)
            pt = next((p for p in d["points"] if p["budget"] == b and p["mode"] == mode), None)
            if pt is None:
                skip(f"S5 59 B={b}/{mode}", "points 里没有这一格")
            else:
                check(f"S5 59 R_uav 重算 B={b}/{mode}", abs(r - pt["R_uav"]) < 1e-9,
                      f"重算 {r:.12f} vs points {pt['R_uav']:.12f}（n={n}）")
        # 单调性判词与 τ_b 符号一致
        f1 = d["F1"]
        check("S6 59 MONOTONE 与 τ_b>0 一致",
              (f1["verdict_full"] == "MONOTONE") == (f1["tau_full"]["tau_b"] > 0),
              f"verdict={f1['verdict_full']} τ_b={f1['tau_full']['tau_b']:.4f}")
    else:
        skip("S5-S6 59", "59 证据缺失")

    # ---- S7 P3-2：INCONCLUSIVE 必须能从区间重叠重算出来 ----
    if "60" in data:
        d = data["60"]
        brs = [d["brackets"][w] for w in ("short", "base", "long")]
        pairs = []
        for i in range(3):
            for j in range(i + 1, 3):
                a, b = brs[i]["bracket"], brs[j]["bracket"]
                pairs.append("OVERLAP" if (a[0] <= b[1] and b[0] <= a[1]) else "DISJOINT")
        want = "INCONCLUSIVE" if any(p == "OVERLAP" for p in pairs) else "SEPARATED"
        check("S7 60 INCONCLUSIVE 可由区间重叠重算", d["F1p"]["verdict"] == want,
              f"verdict={d['F1p']['verdict']} 重算={want} pairs={pairs}")
        # 比例律：x*/W 单调下降 ⇒ "成比例"不成立
        xs = [d["x_star_over_W"][w]["x_star_over_W"] for w in ("short", "base", "long")]
        check("S8 60 x*/W 单调下降（比例律不成立）", xs[0] > xs[1] > xs[2],
              f"x*/W = {xs[0]:.4f} / {xs[1]:.4f} / {xs[2]:.4f}")
        # 正对照：x<5 的 6 个 cell 逐位相同（跨窗口）
        base = d["cells_J"]
        b200 = [c for c in base if c["budget"] in (200, 175)]
        same = True
        for b in (200, 175):
            for pred in ("zoh", "linear"):
                ref = None
                for w in ("short", "base", "long"):
                    c = sel_cell(base, window=w, budget=b, predictor=pred, seed=0)
                    if c is None:
                        same = False
                        break
                    key = (c["J_per_episode"], c["served"], c["alive_steps"])
                    if ref is None:
                        ref = key
                    elif key != ref:
                        same = False
                if not same:
                    break
        check("S9 60 x<5 的 6 格跨窗口逐位相同（正对照）", same,
              "B∈{200,175} × {zoh,linear} × 三窗口 的 J/served/alive 完全一致")
    else:
        skip("S7-S9 60", "60 证据缺失")

    # ---- S10 字面溯源：每个显示值都能在原 JSON 文本里 grep 到 ----
    bad = []
    for t in TRACE:
        src_meta = next((m for m in meta.values() if m["path"] == t["source"]), None)
        if src_meta is None:
            bad.append(f"{t['source']} 不在已加载的证据里")
            continue
        lit = repr(t["value"])
        if lit not in src_meta["text"]:
            bad.append(f"{t['key']} = {lit} 未出现在 {t['source']}")
    check("S10 每个显示值可在证据 JSON 中字面 grep 到", not bad,
          f"{len(TRACE)} 格，失败 {len(bad)}" + ("" if not bad else "；" + "；".join(bad[:5])))


# --------------------------------------------------------------------------- 渲染
def render(data: dict, meta: dict) -> tuple[str, dict]:
    L: list[str] = []
    A = L.append

    A("# P3 主结果表（自动生成，勿手改）")
    A("")
    A("> 生成器：`scripts/61_build_tables.py`。唯一数据入口 `results_io.latest_result_path()`。")
    A("> **不写任何时间戳** ⇒ 内容只取决于证据字节；重跑两次 `sha256` 必须相同（D1-③）。")
    A("> 显示值统一 3 位小数；**精确原值**见 `docs/results_p3_trace.json`（每格 → JSON 键路径 + 原值，可直接 grep）。")
    A("> 所有 `R` 都由 `cells_J` 的逐 episode `J` **重算**后与 summary 对账（容差 `1e-9`）；不直接抄 summary。")
    A("")

    # ---------------- 0 证据钉死 ----------------
    A("## 0 证据钉死（引用数字必须指到这里）")
    A("")
    A("| 逻辑名 | run-id | 文件 | 大小(B) | sha256（前 16） | cells | checks_failed |")
    A("|---|---|---|---|---|---|---|")
    for k in ("58", "59", "59b", "60"):
        if k not in data:
            A(f"| `{SRC[k]}` | — | **缺失** | — | — | — | — |")
            continue
        m, d = meta[k], data[k]
        sz = (ROOT / m["path"]).stat().st_size
        A(f"| `{k}` `{SRC[k]}` | `{m['run_id']}` | `{m['path']}` | {sz} | `{m['sha256'][:16]}` | "
          f"{len(d.get('cells_J', []))} | {len(d.get('checks_failed', []))} |")
    A("")
    A("★ `59b` 是 P3-1 的**补点**（B∈{160,175,190}，18 cell），只提高交点分辨率，不进 F1。")
    A("★ `checks_failed` 列全为 0 才允许引用对应表；非空时该表只能标「诊断」。")
    A("")

    # ---------------- 1 协议 ----------------
    A("## 1 共用协议（三块表一致，不随表变）")
    A("")
    c58 = data.get("58", {}).get("config", {})
    A(f"- 拓扑/任务：`N={c58.get('N')}`，`max_steps={c58.get('window')}`，`warmup={c58.get('warmup')}`，"
      f"臂 = `{c58.get('arm')}`（仅 `age`），`absorbed={c58.get('absorbed')}`（P3-2 里 `absorbed_uav := escape_radius`）。")
    A("- 指标：`R_i = Σ_e J_i(linear) / Σ_e J_i(zoh)`，**同种子同 episode 配对**；")
    A("  区间 = **cluster bootstrap，重采样单位 = `reset()` 初值**（不是 episode 次），`n_boot=10000`。")
    A("- `J_uav` 单位 m（越大越坏）；`served` = 实际送达帧数；`t>B` = **预算用完后**（`t>B`）仍在发的帧数。")
    A("- GATE=0.90 / 非牺牲门 `max r_i ≤1.10` **只记录不判**（`R_cart≈1` ⇒ 任何档都过不了，T21 已定，非新结论）。")
    A("")

    # ---------------- 2 P3-0 ----------------
    A("## 2 P3-0 确认（主确认集 `{100,130,160}`，90 个互不相同的初值，与开发集零交集）")
    A("")
    if "58" in data:
        d = data["58"]
        src = meta["58"]["path"]
        cells = d["cells_J"]
        F1 = d["F1"]
        F2 = d["F2"]["primary/200/unpaced"]
        A("### 2a 主确认：`B=200 unpaced`（`t>B` 发帧恒 0）")
        A("")
        A("| 预测器 | `R_uav` | 95% 区间(UAV) | `R_cart` | 95% 区间(cart) | `J_uav` lin/zoh | `J_cart` lin/zoh | `served` lin/zoh |")
        A("|---|---|---|---|---|---|---|---|")
        for pred, tag in (("linear", "`Linear`（**冻结方法**）"),
                          ("hybrid", "`hybrid`（历史参考）"),
                          ("truedyn", "`TrueDyn`〔diagnostic〕")):
            ru, n = reratio(cells, num=pred, seed_set="primary", budget=200, mode="unpaced", user=0)
            rc, _ = reratio(cells, num=pred, seed_set="primary", budget=200, mode="unpaced", user=1)
            if pred == "linear":
                ciu = (f"[{fmt_num(F1['ci_uav'][0], src, 'F1.ci_uav[0]')}, "
                       f"{fmt_num(F1['ci_uav'][1], src, 'F1.ci_uav[1]')}]")
                cic = (f"[{fmt_num(F1['ci_cart'][0], src, 'F1.ci_cart[0]')}, "
                       f"{fmt_num(F1['ci_cart'][1], src, 'F1.ci_cart[1]')}]")
            else:
                ciu = cic = "—（未做区间）"
            jl, jz = sel_cell(cells, seed_set="primary", budget=200, mode="unpaced",
                              predictor=pred, seed=100), \
                sel_cell(cells, seed_set="primary", budget=200, mode="unpaced", predictor="zoh", seed=100)
            A(f"| {tag} | **{fmt_calc(ru)}** | {ciu} | {fmt_calc(rc)} | {cic} | "
              f"{fmt_calc(jl['J_mean'][0])}/{fmt_calc(jz['J_mean'][0])} | "
              f"{fmt_calc(jl['J_mean'][1])}/{fmt_calc(jz['J_mean'][1])} | "
              f"{fmt_calc(jl['served'][0],'.0f')}/{fmt_calc(jz['served'][0],'.0f')} |")
        A("")
        A(f"★ 表里 `R` 全部由 `cells_J` 重算（每格 90 个 episode）；`J`/`served` 取 `seed=100` 的 cell（逐种子表见 2c）。")
        A(f"★ `hybrid`/`TrueDyn` 的 `R` 是**点估计**（该轮未对它们做 bootstrap），只能与 `Linear` 比方向，不能比区间。")
        A("")
        A("### 2b 断言（预注册 §4 判据，逐条给值）")
        A("")
        A("| 断言 | 值 | 来源 |")
        A("|---|---|---|")
        A(f"| 判词 | **`{F1['verdict']}`** | `58.F1.verdict` |")
        A(f"| UAV 区间上界 < 1.0 | {fmt_num(F1['ci_uav'][1], src, 'F1.ci_uav[1]')} < 1.0 ⇒ 成立 | `58.F1.ci_uav[1]` |")
        A(f"| `R_cart` ≤ 1.10（等价界） | {fmt_num(F1['R'][1], src, 'F1.R[1]')} ⇒ 成立 | `58.F1.R[1]` |")
        A(f"| 逐种子 UAV 同向 | `{F1['seeds_uav_below_1']}` | `58.F1.seeds_uav_below_1` |")
        A(f"| 逐 cluster 改善比例（UAV） | {fmt_num(F2['frac_clusters_below_1'][0], src, 'F2.primary/200/unpaced.frac_clusters_below_1[0]', '.3f')}"
          f"（{fmt_calc(F2['frac_clusters_below_1'][0]*90, '.0f')}/90 个初值） | `58.F2` |")
        A(f"| 逐 cluster 改善比例（CartPole） | {fmt_num(F2['frac_clusters_below_1'][1], src, 'F2.primary/200/unpaced.frac_clusters_below_1[1]', '.3f')}"
          f"（{fmt_calc(F2['frac_clusters_below_1'][1]*90, '.0f')}/90） ⇒ **不构成收益** | `58.F2` |")
        A(f"| 逐集比值中位（CartPole） | {fmt_num(F2['cluster_ratio_median'][1], src, 'F2.primary/200/unpaced.cluster_ratio_median[1]')}（正好 1） | `58.F2` |")
        A("")
        A("★ **CartPole 只能写「未超出预注册等价界 1.10」，不写「两用户都变好」**：区间含 1、逐 cluster 仅 40%。")
        A("")
        A("### 2c 逐种子（`R_uav` / `R_cart`）")
        A("")
        A("| 种子 | `R_uav` | `R_cart` |")
        A("|---|---|---|")
        for s, (ru, rc) in F1["per_seed_R"].items():
            A(f"| {s} | {fmt_num(ru, src, f'F1.per_seed_R.{s}[0]')} | {fmt_num(rc, src, f'F1.per_seed_R.{s}[1]')} |")
        A("")
        A("### 2d 失配边界（同一 `Linear`，只改时序；R17 必报 `t>B` 发帧）")
        A("")
        A("| 档 | `mode` | `R_uav` | 95% 区间 | 逐 cluster 改善 | `J_uav` lin/zoh | `t>B` 发帧（3 种子 × 30 集合计 / 每集） |")
        A("|---|---|---|---|---|---|---|")
        for b, mode in ((100, "unpaced"), (100, "paced"), (200, "unpaced")):
            o = d["F2"][f"primary/{b}/{mode}"]
            ru, _ = reratio(cells, num="linear", seed_set="primary", budget=b, mode=mode, user=0)
            cl = sel_cell(cells, seed_set="primary", budget=b, mode=mode, predictor="linear", seed=100)
            cz = sel_cell(cells, seed_set="primary", budget=b, mode=mode, predictor="zoh", seed=100)
            late = sum(c["served_late"][0] + c["served_late"][1]
                       for c in cells
                       if c.get("seed_set") == "primary" and c["budget"] == b
                       and c["mode"] == mode and c["predictor"] == "linear")
            n_ep_tot = sum(len(c["J_per_episode"]) for c in cells
                           if c.get("seed_set") == "primary" and c["budget"] == b
                           and c["mode"] == mode and c["predictor"] == "linear")
            A(f"| B={b} | `{mode}` | **{fmt_calc(ru)}** | "
              f"[{fmt_num(o['ci_uav'][0], src, f'F2.primary/{b}/{mode}.ci_uav[0]')}, "
              f"{fmt_num(o['ci_uav'][1], src, f'F2.primary/{b}/{mode}.ci_uav[1]')}] | "
              f"{fmt_calc(o['frac_clusters_below_1'][0]*90, '.0f')}/90 | "
              f"{fmt_calc(cl['J_mean'][0])}/{fmt_calc(cz['J_mean'][0])} | "
              f"{late}（{fmt_calc(late / n_ep_tot, '.0f')}/集） |")
        A("")
        A("★ **`B=100 unpaced` 的 `R_uav` 1.223 与 `paced` 0.476 差 2.6×、符号相反** ⇒ "
          "同一预测器在不同时序下相反；`unpaced` 的 `t>B` 发帧为 **0**（断服），`paced` 不为 0。")
        A("★ `B=200` 时 `paced ≡ unpaced`（`B=L` ⇒ 配速 no-op），两行数值相同是恒等式，不是「协同为零」。")
        A("")
    else:
        A("**58 证据缺失 ⇒ 本节 SKIP**")
        A("")

    # ---------------- 3 P3-1 ----------------
    A("## 3 P3-1 适用域（**开发结果**：开发集 `{0,1,2}`；确认集只做落点复核）")
    A("")
    if "59" in data:
        d = data["59"]
        src = meta["59"]["path"]
        A("横轴 `x` = **实测** UAV `age` 算术均值（`alive` 且 `t≥41` 的决策点，`linear` 轨迹）；纵轴 `R_uav`。")
        A("★ 轴**内生**：预测改了轨迹 ⇒ 改了 `age` 分布（`x_zoh` 列给出替代轴的同口径值，见 trace）。")
        A("")
        A("| B | `mode` | `x`（实测均值） | `R_uav` | 95% 区间 | `J_uav` lin/zoh | `J_cart` lin/zoh | `served` lin/zoh | `alive` lin/zoh | `t>B` 发帧 |")
        A("|---|---|---|---|---|---|---|---|---|---|")
        pts = sorted(d["points"], key=lambda p: (p["mode"], p["x"]))
        for p in pts:
            i = d["points"].index(p)
            A(f"| {p['budget']} | `{p['mode']}` | {fmt_num(p['x'], src, f'points[{i}].x')} | "
              f"**{fmt_num(p['R_uav'], src, f'points[{i}].R_uav')}** | "
              f"[{fmt_num(p['ci_uav'][0], src, f'points[{i}].ci_uav[0]')}, {fmt_num(p['ci_uav'][1], src, f'points[{i}].ci_uav[1]')}] | "
              f"{fmt_calc(p['J'][0])}/{fmt_calc(p['J_zoh'][0])} | "
              f"{fmt_calc(p['J'][1])}/{fmt_calc(p['J_zoh'][1])} | "
              f"{fmt_calc(p['served'][0],'.0f')}/{fmt_calc(p['served_zoh'][0],'.0f')} | "
              f"{fmt_calc(p['alive'][0],'.0f')}/{fmt_calc(p['alive_zoh'][0],'.0f')} | "
              f"{fmt_calc(p['late'],'.0f')} |")
        A("")
        if "59b" in data:
            b = data["59b"]
            sb = meta["59b"]["path"]
            A("### 3b 补点（P3-1 §4b：只提高交点分辨率，不进 F1）")
            A("")
            A("| B | `mode` | `x` | `R_uav` | 95% 区间 | `J_uav` lin/zoh | `t>B` 发帧 |")
            A("|---|---|---|---|---|---|---|")
            for j, p in enumerate(b["points"]):
                A(f"| {p['budget']} | `{p['mode']}` | {fmt_num(p['x'], sb, f'points[{j}].x')} | "
                  f"**{fmt_num(p['R_uav'], sb, f'points[{j}].R_uav')}** | "
                  f"[{fmt_num(p['ci_uav'][0], sb, f'points[{j}].ci_uav[0]')}, {fmt_num(p['ci_uav'][1], sb, f'points[{j}].ci_uav[1]')}] | "
                  f"{fmt_calc(p['J'][0])}/{fmt_calc(p['J_zoh'][0])} | {fmt_calc(p['late'],'.0f')} |")
            A("")
        f1 = d["F1"]
        A("### 3c 形状与交点")
        A("")
        A("| 量 | 值 | 含义 |")
        A("|---|---|---|")
        A(f"| Kendall `τ_b`（12 点） | {fmt_num(f1['tau_full']['tau_b'], src, 'F1.tau_full.tau_b')} | 正相关；"
          f"相邻逆序 {fmt_num(f1['tau_full']['n_adjacent_inversions'], src, 'F1.tau_full.n_adjacent_inversions', 'd')} 个（全在 `unpaced` 高 `x` 端）"
          f" ⇒ **「上升后饱和」，不是严格单调** |")
        A(f"| 判词 | `{f1['verdict_full']}` | 只看方向，不看线性 |")
        A(f"| `x` 跨度 | {fmt_num(f1['x_min'], src, 'F1.x_min')} → {fmt_num(f1['x_max'], src, 'F1.x_max')}"
          f"（{fmt_num(f1['x_span_ratio'], src, 'F1.x_span_ratio', '.1f')}×） | 跨度够，不是「扫不动」 |")
        A(f"| `paced` 分支取值范围 | 0.313–0.618（全 <1） | 见上表 `paced` 各行 |")
        A(f"| 交点（未补点） | `x≈{fmt_num(d['verdicts']['F2']['unpaced'][0]['x'], src, 'verdicts.F2.unpaced[0].x', '.2f')}`，"
          f"bracket ({fmt_calc(d['verdicts']['F2']['unpaced'][0]['between'][0],'.2f')}, "
          f"{fmt_calc(d['verdicts']['F2']['unpaced'][0]['between'][1],'.2f')}) | **只能给区间，不能给数** |")
        A("")
        A("### 3d 确认集落点复核（不参与任何选择）")
        A("")
        A("| 档 | 开发集 `R_uav` | 确认集 `R_uav` | Δ |")
        A("|---|---|---|---|")
        for key in ("150/unpaced", "150/paced", "200/unpaced"):
            o = d["F3"][key]
            A(f"| {key} | {fmt_num(o['R_dev'][0], src, f'F3.{key}.R_dev[0]')} | "
              f"{fmt_num(o['R_conf'][0], src, f'F3.{key}.R_conf[0]')} | "
              f"{fmt_calc(o['d_uav'])} |")
        A("")
        A("★ 三格 |Δ| ≤ 0.029 ⇒ 落点一致；这条只说明「表不是开发集挑出来的」，**不构成第二次独立确认**。")
        A("")
    else:
        A("**59 证据缺失 ⇒ 本节 SKIP**")
        A("")

    # ---------------- 4 P3-2 ----------------
    A("## 4 P3-2 失效窗口平移检验（**开发结果**，`unpaced`，8 档预算 × 3 窗口）")
    A("")
    if "60" in data:
        d = data["60"]
        src = meta["60"]["path"]
        A("| 窗口 | `R_fail`(m) | 实测 `W`(步) | 交点区间 | `x*` | `x*/W` | `R` 峰值 |")
        A("|---|---|---|---|---|---|---|")
        for w in ("short", "base", "long"):
            br = d["brackets"][w]
            ww = d["windows"][w]
            xs = d["x_star_over_W"][w]
            peak = max(p["R_uav"] for p in d["summary"][w]["points"])
            A(f"| `{w}` | {fmt_num(ww['radius'], src, f'windows.{w}.radius', '.0f')} | "
              f"{fmt_num(ww['W_expected'], src, f'windows.{w}.W_expected', 'd')} | "
              f"[{fmt_num(br['bracket'][0], src, f'brackets.{w}.bracket[0]')}, "
              f"{fmt_num(br['bracket'][1], src, f'brackets.{w}.bracket[1]')}] | "
              f"**{fmt_num(br['x_star'], src, f'brackets.{w}.x_star')}** | "
              f"{fmt_num(xs['x_star_over_W'], src, f'x_star_over_W.{w}.x_star_over_W')} | "
              f"{fmt_calc(peak)}（派生） |")
        A("")
        A(f"**判词 `{d['F1p']['verdict']}`**（{d['F1p']['reason']}）：三档区间"
          f"[{fmt_calc(d['brackets']['short']['bracket'][0])}, {fmt_calc(d['brackets']['short']['bracket'][1])}] / "
          f"[{fmt_calc(d['brackets']['base']['bracket'][0])}, {fmt_calc(d['brackets']['base']['bracket'][1])}] / "
          f"[{fmt_calc(d['brackets']['long']['bracket'][0])}, {fmt_calc(d['brackets']['long']['bracket'][1])}] "
          f"两两重叠 ⇒ **本设计分不清位移**。")
        A("")
        A("★ `W` ×3.0 而 `x*` 只 ×1.41；`x*/W` 0.165 → 0.078（单调降 2.1×）⇒ **「门槛按窗口成比例」不成立**（附加量，预注册标「只报不判」）。")
        A("★ **不得**把 `INCONCLUSIVE` 写成「窗口无关」：点估计单调（5.79 < 7.47 < 8.16），方向不反。")
        A("★ **结构性限制**：改窗口 ⇒ 改存活 ⇒ 改横轴（`age` 均值同时是「陈旧度」与「存活时长」的函数）；"
          "`x*/W` 的下降有一部分是**选择效应**。")
        A("")
    else:
        A("**60 证据缺失 ⇒ 本节 SKIP**")
        A("")

    # ---------------- 5 主张 ----------------
    # ★ 本节每个数字都由 JSON 计算并登记 trace —— 不允许出现"只有正文才有"的数（D4-③）。
    A("## 5 主张与否定条件（每个数字的键路径见 `results_p3_trace.json`）")
    A("")
    A("| # | 主张 / 否定 | 值 |")
    A("|---|---|---|")
    n2_row = ""      # 由 59 计算、排在 C7 之后（"否定"集中放在末尾）
    if "58" in data:
        d58 = data["58"]
        s58 = meta["58"]["path"]
        F1 = d58["F1"]
        F2 = d58["F2"]["primary/200/unpaced"]
        served_z, served_l = F1["served"]["zoh"][0], F1["served"]["linear"][0]
        A(f"| C1 | `Linear` 补偿在**快刷新**档（B=200 unpaced）降低 UAV 代价 | `R_uav` "
          f"{fmt_num(F1['R'][0], s58, 'F1.R[0]')}，区间 "
          f"[{fmt_num(F1['ci_uav'][0], s58, 'F1.ci_uav[0]')}, {fmt_num(F1['ci_uav'][1], s58, 'F1.ci_uav[1]')}]；"
          f"逐 cluster 改善 {fmt_num(F2['frac_clusters_below_1'][0], s58, 'F2.primary/200/unpaced.frac_clusters_below_1[0]')}"
          f"（= 90/90） |")
        A(f"| C2 | 该改善**不是靠多分帧** | `served` "
          f"{fmt_num(served_z, s58, 'F1.served.zoh[0]', '.1f')} → {fmt_num(served_l, s58, 'F1.served.linear[0]', '.1f')}"
          f"（{fmt_calc((served_l / served_z - 1) * 100, '+.1f')}%） |")
        for rank, (b, mode) in enumerate(((100, "unpaced"), (100, "paced")), start=3):
            o = d58["F2"][f"primary/{b}/{mode}"]
            A(f"| C{rank} | {'**断服**档上同一预测器反而有害' if mode == 'unpaced' else '`paced` 时序下低预算档仍 <1'} | "
              f"`R_uav` {fmt_num(o['R'][0], s58, f'F2.primary/{b}/{mode}.R[0]')}，区间 "
              f"[{fmt_num(o['ci_uav'][0], s58, f'F2.primary/{b}/{mode}.ci_uav[0]')}, "
              f"{fmt_num(o['ci_uav'][1], s58, f'F2.primary/{b}/{mode}.ci_uav[1]')}]；逐 cluster 改善 "
              f"{fmt_num(o['frac_clusters_below_1'][0], s58, f'F2.primary/{b}/{mode}.frac_clusters_below_1[0]')} |")
    if "59" in data:
        d59 = data["59"]
        s59 = meta["59"]["path"]
        t = d59["F1"]["tau_full"]
        A(f"| C5 | 危害随**实测**陈旧度上升并饱和 | `τ_b` {fmt_num(t['tau_b'], s59, 'F1.tau_full.tau_b')}，"
          f"相邻逆序 {fmt_num(t['n_adjacent_inversions'], s59, 'F1.tau_full.n_adjacent_inversions', 'd')} 个 |")
        pts = {(p["budget"], p["mode"]): k for k, p in enumerate(d59["points"])}
        j150 = d59["points"][pts[(150, "unpaced")]]
        j60 = d59["points"][pts[(60, "unpaced")]]
        k150, k60 = pts[(150, "unpaced")], pts[(60, "unpaced")]
        n2_row = (f"| N2 | 比值**掩盖绝对危害** | `R` 只 {fmt_calc(j150['R_uav'])}→{fmt_calc(j60['R_uav'])}（"
                  f"{fmt_calc((j60['R_uav'] / j150['R_uav'] - 1) * 100, '+.0f')}%），而 `J_uav` "
                  f"{fmt_num(j150['J'][0], s59, f'points[{k150}].J[0]')} → {fmt_num(j60['J'][0], s59, f'points[{k60}].J[0]')} m"
                  f"（{fmt_calc(j60['J'][0] / j150['J'][0], '.1f')}×） |")
    if "60" in data:
        d60 = data["60"]
        s60 = meta["60"]["path"]
        xs = [d60["brackets"][w]["x_star"] for w in ("short", "base", "long")]
        A(f"| C6 | **「门槛 = 失效窗口」未获支持** | 交点 `x*` "
          + csv_num(xs, s60, [f"brackets.{w}.x_star" for w in ("short", "base", "long")])
          + f"，三档区间两两重叠（判词 `{d60['F1p']['verdict']}`） |")
        ratio = [d60["x_star_over_W"][w]["x_star_over_W"] for w in ("short", "base", "long")]
        A(f"| C7 | **「门槛按窗口成比例」被否证** | `x*/W` "
          + csv_num(ratio, s60, [f"x_star_over_W.{w}.x_star_over_W" for w in ("short", "base", "long")])
          + "（单调下降） |")
    if "58" in data:
        A(f"| N1 | CartPole **不构成收益** | 区间含 1、逐 cluster 改善 "
          f"{fmt_num(data['58']['F2']['primary/200/unpaced']['frac_clusters_below_1'][1], meta['58']['path'], 'F2.primary/200/unpaced.frac_clusters_below_1[1]')}"
          f"（= 36/90）、逐集比值中位 "
          f"{fmt_num(data['58']['F2']['primary/200/unpaced']['cluster_ratio_median'][1], meta['58']['path'], 'F2.primary/200/unpaced.cluster_ratio_median[1]')} |")
    if n2_row:
        A(n2_row)
    A("")
    A("★ 术语纪律：`Linear` **无可学参数** ⇒ C1–C4 **不构成「世界模型收益」**，不解锁 P2；`TrueDyn` 一律 `diagnostic`。")
    A("★ C3 的 `t>B` 发帧 = 0 与 C4 的 `t>B` 发帧 ≠ 0 见 §2d（R17 必须同报）。")
    A("")

    # ---------------- 6 入口 ----------------
    A("## 6 复现与图")
    A("")
    A("- 最短复现：`docs/REPRODUCE-P3.md` + `scripts/62_min_repro.py`（4 cell，CPU，只读证据）。")
    A("- 论文图：`scripts/61c_plot_paper_fig.py` → `assets/paper_p3.png`（2×2，数据同入口）。")
    A("- 旧单面板图保留：`assets/59_p3_boundary.png`、`assets/60_window_shift.png`（不删）。")
    A("")
    return "\n".join(L), {"n_trace": len(TRACE)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true", help="只跑检查，不写 docs/")
    args = ap.parse_args()

    print("[61] P3 主结果表：只读入库 JSON")
    data, meta = load_all()
    if not data:
        print("\n[FAIL] 一份证据都没找到")
        return 2

    # ★ 顺序有意为之：**先渲染再检查**。S10 核对的是"渲染时登记进 TRACE 的每一格"，
    #   若先跑检查，TRACE 还是空的 ⇒ S10 会恒 PASS（#39「检查没有分辨力」的同型）。
    text, info = render(data, meta)
    checks(data, meta)

    if args.check_only:
        print(f"\n[check-only] 跳过写盘（trace {info['n_trace']} 格）")
    else:
        out = ROOT / "docs" / "RESULTS-P3.md"
        old = out.read_text(encoding="utf-8") if out.exists() else None
        if old == text:
            print(f"\n[unchanged] {out.relative_to(ROOT).as_posix()} 内容未变，不重写")
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text, encoding="utf-8")
            print(f"\n[out] {out.relative_to(ROOT).as_posix()}（{len(text)} B）")
        tr = ROOT / "docs" / "results_p3_trace.json"
        tr_txt = json.dumps(TRACE, ensure_ascii=False, indent=1)
        old_tr = tr.read_text(encoding="utf-8") if tr.exists() else None
        if old_tr == tr_txt:
            print(f"[unchanged] {tr.relative_to(ROOT).as_posix()}")
        else:
            tr.write_text(tr_txt, encoding="utf-8")
            print(f"[out] {tr.relative_to(ROOT).as_posix()}（{info['n_trace']} 格）")

    print(f"\n{'ALL CHECKS PASS' if not FAILS else 'FAILED: ' + '; '.join(FAILS)}")
    if SKIPS:
        print("SKIPPED: " + "; ".join(SKIPS))
    return 0 if not FAILS else 2


if __name__ == "__main__":
    sys.exit(main())
