# -*- coding: utf-8 -*-
"""CW-R0 交付文档的数字可追溯性检查（**只读**，不写 results/）。

为什么需要它
------------
`docs/CW-R0-协议交付.md` 里的 §5 面板表是**手抄**的。手抄就会错 ——
第一次写这张表时我就把 `csi_sensitive_4 @ ρ=0.85,B=0.8` 的 AoI 写成 7.5（实际 7.8，
7.5 是 `ρ=0.98` 那行的值），并且漏了一整行。这类错误不会让任何闸变红，
只会让读者按错表去解释。

做法
----
1. 从证据目录读 `panel_agg.json`（24 行）与 `gates.json`/`meta.json`；
2. 解析交付文档 §5 的 markdown 表；
3. **逐格比对**：每一行的 (ρ, B/T, policy) 必须存在，且 cost / probe / att / del / fhold / aoi 必须对得上；
4. 文档里**声称的**聚合数（49 项、0 失败、72 行、1.25 s）必须与 meta/gates 一致；
5. 阴性对照：把文档里任一格的数改一位小数后重跑，必须 FAIL。

用法::

    python -m experiments.cw_v1.doccheck <run_id>            # 检查
    python -m experiments.cw_v1.doccheck <run_id> --nctl     # 只跑阴性对照

退出码：0 = 全部对上；2 = 有对不上的格（打印行号与差值）。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "results" / "cw_v1" / "runs"
DOC = ROOT / "docs" / "CW-R0-协议交付.md"

ROW_RE = re.compile(
    r"^\|\s*(?P<rho>0\.\d+)\s*\|\s*(?P<bt>0\.\d+)\s*\|\s*`?(?P<policy>[a-z0-9_]+)`?\s*\|"
    r"\s*\*{0,2}(?P<cost>[\d.]+)\*{0,2}\s*\|\s*(?P<probe>\d+)\s*\|\s*(?P<att>\d+)\s*\|"
    r"\s*(?P<del>[\d.]+)\s*\|\s*\*{0,2}(?P<fhold>\d+)\*{0,2}\s*\|\s*(?P<aoi>-?[\d.]+)\s*\|"
)

TOL = {
    # 文档写 3 位小数 ⇒ 舍入误差上界 5e-4；留 6e-4 吸收末位进位（0.0755→0.076 差 5.0e-4）。
    # ★ 不要再放大：1e-3 量级的错（0.076 vs 0.086）必须仍被抓到。
    "cost": 6e-4,
    "del": 6e-2,    # 1 位小数
    "aoi": 6e-2,    # 1 位小数
}


def load_run(run_id: str) -> tuple[dict, list[dict], dict]:
    d = RUNS / run_id
    if not d.is_dir():
        raise SystemExit(f"[doccheck] 证据目录不存在：{d}")
    agg = json.loads((d / "panel_agg.json").read_text(encoding="utf-8"))
    gates = json.loads((d / "gates.json").read_text(encoding="utf-8"))
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    return meta, agg, gates


def parse_rows(text: str) -> list[tuple[int, dict]]:
    rows = []
    for ln, line in enumerate(text.splitlines(), start=1):
        m = ROW_RE.match(line.strip())
        if m:
            rows.append((ln, m.groupdict()))
    return rows


def compare(row: dict, src: dict) -> list[str]:
    bad = []
    # 计数列必须完全相等（整数）
    for dst, key in (("probe", "probe_mean"), ("att", "attempted_mean"), ("fhold", "forced_hold_mean")):
        a, b = int(float(row[dst])), int(round(float(src[key])))
        if a != b:
            bad.append(f"{dst}: 文档 {a} vs 证据 {b}")
    for dst, key, tol in (("cost", "cost_mean", TOL["cost"]),
                          ("del", "delivered_mean", TOL["del"]),
                          ("aoi", "aoi_mean", TOL["aoi"])):
        a, b = float(row[dst]), float(src[key])
        if abs(a - b) > tol:
            bad.append(f"{dst}: 文档 {a} vs 证据 {b:.4f}（差 {abs(a-b):.2e} > {tol:g}）")
    return bad


def run_checks(run_id: str, *, doc_text: str) -> tuple[int, list[str]]:
    meta, agg, gates = load_run(run_id)
    idx = {(round(float(r["rho"]), 2), round(float(r["budget_ratio"]), 2), r["policy"]): r for r in agg}
    fails: list[str] = []
    rows = parse_rows(doc_text)
    if not rows:
        fails.append("§5 面板表一行都没解析出来（判据失效，不是表没问题）")

    for ln, row in rows:
        key = (round(float(row["rho"]), 2), round(float(row["bt"]), 2), row["policy"])
        src = idx.get(key)
        if src is None:
            fails.append(f"L{ln}: 证据里没有这一行 {key}")
            continue
        for msg in compare(row, src):
            fails.append(f"L{ln} {key}: {msg}")

    # 聚合口径
    if f"{gates['n_checks']} 项" not in doc_text:
        fails.append(f"文档未写闸数 {gates['n_checks']}")
    if f"n_fail={gates['n_fail']}" not in doc_text:
        fails.append(f"文档未写 n_fail={gates['n_fail']}")
    if f"n_panel_rows={meta['n_panel_rows']}" not in doc_text:
        fails.append(f"文档未写 n_panel_rows={meta['n_panel_rows']}")
    if f"elapsed_s={meta['elapsed_s']}" not in doc_text:
        fails.append(f"文档未写 elapsed_s={meta['elapsed_s']}")
    if len(agg) != 24:
        fails.append(f"证据聚合行数 {len(agg)} ≠ 24")
    return len(rows), fails


def _mutate(text: str, ln: int, old: str, new: str) -> str:
    lines = text.splitlines()
    assert old in lines[ln - 1], f"L{ln} 不含 {old!r}"
    lines[ln - 1] = lines[ln - 1].replace(old, new, 1)
    return "\n".join(lines)


def negative_control(run_id: str) -> list[str]:
    """四个扰动都必须被抓到；任何一个漏掉 ⇒ 检查无分辨力。

    ★ 只有「末位 +1」是不够的对照：它可能恰好仍在容差内，抓到也说明不了阈值合理。
      所以这里分两层：①连续量超容差 3×；②整数计数 ±1（容差为 0，必须逐位相等）。
    """
    text = DOC.read_text(encoding="utf-8")
    rows = parse_rows(text)
    if not rows:
        return ["阴性对照无法构造：表解析失败"]
    ln, row = rows[0]
    problems: list[str] = []

    # ① cost 超容差 3 倍
    mut = _mutate(text, ln, row["cost"], f"{float(row['cost']) + 0.002:.3f}")
    if not run_checks(run_id, doc_text=mut)[1]:
        problems.append("cost +0.002 未被抓到")
    # ② delivered 超容差 3 倍
    mut = _mutate(text, ln, row["del"], f"{float(row['del']) + 0.2:.1f}")
    if not run_checks(run_id, doc_text=mut)[1]:
        problems.append("delivered +0.2 未被抓到")
    # ③ fhold 整数 +1
    mut = _mutate(text, ln, f"| {row['fhold']} |", f"| {int(row['fhold']) + 1} |")
    if not run_checks(run_id, doc_text=mut)[1]:
        problems.append("forced_hold +1 未被抓到")
    # ④ 串格：把第 2 行的 policy 名换成第 1 行的，数字不换（"数字对但条件错"）
    if len(rows) >= 2:
        ln2, row2 = rows[1]
        if row2["policy"] != row["policy"]:
            mut = _mutate(text, ln2, row2["policy"], row["policy"])
            if not run_checks(run_id, doc_text=mut)[1]:
                problems.append("policy 串格未被抓到")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--nctl", action="store_true", help="只跑阴性对照")
    a = ap.parse_args(argv)

    text = DOC.read_text(encoding="utf-8")
    if a.nctl:
        bad = negative_control(a.run_id)
        if bad:
            print("[NCTL FAIL]", *bad, sep="\n  ")
            return 2
        print("[NCTL PASS] 4 个扰动（cost+0.002 / del+0.2 / fhold+1 / policy 串格）全部被抓到"
              " ⇒ 检查有分辨力")
        return 0

    n, fails = run_checks(a.run_id, doc_text=text)
    print(f"[doccheck] run={a.run_id}  解析面板表 {n} 行")
    if fails:
        print(f"[doccheck] FAIL {len(fails)} 项")
        for f in fails:
            print("  -", f)
        return 2
    print("[doccheck] PASS：面板表逐格与证据一致；聚合口径一致")
    return 0


if __name__ == "__main__":
    sys.exit(main())
