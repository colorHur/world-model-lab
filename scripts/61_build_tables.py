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

P3-3c 追加的三条（都在检查里，不是口号）
--------------------------------------
- `S10c` **派生值可重算**：`fmt_derived` 登记操作数键 + 左结合算子，`fmt_agg` 登记聚合口径，
  `fmt_reratio` 登记配对重算口径；S10c 逐个重算，差 >1e-9 或格式化后与 `display` 不符 ⇒ FAIL。
  （这条抓到过一次真错：登记的操作数顺序与算式相反。）
- `S11` **初稿数字漂移闸**：抽初稿里"整段只由数字/单位/分隔符组成"的 inline code，
  其中每个小数都必须在 trace 的 `display` 里；抽到的数 <15 个 ⇒ FAIL（非退化守卫）；
  另有**阴性对照**（改一位小数必须被抓到），不抓到 ⇒ 说明闸没有分辨力，如实记。
- **§7 资源与耗时**：只从 `58.TIMING` 与 `cells_J[*].elapsed_s` 出，图上/文里强制写明
  "Python 层实现开销，不是实时预算结论"。

用法
----
    D:/devtools/miniconda3/envs/ai-lab/python.exe scripts/61_build_tables.py
    ... --check-only        # 只跑检查，不写 docs/
    ... --draft <path>      # 指定初稿（默认科研空间里的 2026-10-20-P3-初稿-v0.1.md；缺失 ⇒ S11 SKIP）

产物
----
    docs/RESULTS-P3.md              （内容不变则不重写）
    docs/results_p3_trace.json      （每格 → JSON 原值 + 键路径；派生量带可重算的操作数/公式）

退出码：0 = 全部检查通过；2 = 有检查 FAIL（不是"研究 Gate 未过"，是**文档与证据不一致**）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from wmlab.eval.evidence import (  # noqa: E402
    EvidenceError, resolve, sha256_norm as evidence_sha, load_manifest, NEG_MANIFEST_PATH,  # noqa: F401
)
from wmlab.eval.results_io import latest_result_path  # noqa: E402

# 论文正文的**唯一发布版本**在本仓库（v0.3 起为 `docs/PAPER-P3-v0.3.md`；v0.2 保留为历史，
# 科研空间里的 v0.1 亦保留为历史）——S11 闸门默认核这一份。
# 为什么要搬进来：复验 E 项点名"初稿不在 Git 提交内 ⇒ 论文尚无版本链"。两份正文同时改会漂移，
# 所以只认仓库这份；科研空间不再维护第二份正文。
DEFAULT_DRAFT = ROOT / "docs" / "PAPER-P3-v0.3.md"

FAILS: list[str] = []
SKIPS: list[str] = []
TRACE: list[dict] = []          # 直接取自 JSON 的值（S10 字面 grep）
TRACE_DERIVED: list[dict] = []  # 由 JSON 值算出来的量（S10c 用公式重算复核）
DRAFT_PATH: str = str(DEFAULT_DRAFT)
DATA: dict = {}                 # render 期间可查的 JSON（claim 绑定用）
CLAIMS: list[dict] = []         # claim-id → (source, JSON 键, 条件, 显示精度, 值)
USE_LATEST = False              # True ⇒ 退回 latest（会漂移），仅诊断用

SRC = {
    "58": "58_p3_confirmation",
    "59": "59_p3_boundary",
    "59b": "59b_p3_boundary_refine",
    "60": "60_p3_window_shift",
}

# ★ v0.3 追加：**否定线**（T25–T28）。走**独立**冻结清单 `docs/evidence_manifest_p3_neg.json`
# （`scripts/63 --set neg` 生成），主清单一个字节不动。
# 为什么要单独一份：主清单钉的是主表/图/复现的四份输入，给它换版本号会让那几处的
# "清单未换"记录一并作废；而否定线是正文 §6.2 的**新增**引用面，单独钉、单独校验更干净。
NEG_SRC = {
    "66": "66_identifiability",
    "67": "67_cf_action",
    "68": "68_t27_gated_fallback",
    "69": "69_t28_confirm",
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
    """返回 (数据, 溯源信息)。

    默认走**冻结清单**（`docs/evidence_manifest_p3.json`）：`run_id` 与内容指纹都钉死，
    新增 run 不会改变本次输入。`--latest` 才退回"取最新那份"（诊断用，会漂移）。
    缺证据/与清单不符 ⇒ SKIP（不判 PASS），并在渲染里写成"缺失"。
    """
    data, meta = {}, {}
    for k, name in SRC.items():
        if USE_LATEST:
            p = latest_result_path(name)
            if p is None or not p.exists():
                skip(f"S-{k}", f"（--latest）找不到 {name}")
                continue
            raw = p.read_bytes()
            src = {"path": p, "raw": raw, "sha256_norm": evidence_sha(raw),
                   "run_id": p.parent.name, "pinned": False, "note": "--latest，非清单证据"}
        else:
            try:
                src = resolve(name, key=k)
            except EvidenceError as exc:
                skip(f"S-{k}", str(exc))
                continue
        p, raw = src["path"], src["raw"]
        data[k] = json.loads(raw.decode("utf-8"))
        meta[k] = {
            "name": name,
            "path": p.relative_to(ROOT).as_posix(),
            "run_id": src["run_id"],
            "sha256": evidence_sha(raw),
            "sha256_bytes": hashlib.sha256(raw).hexdigest(),
            "pinned": src["pinned"],
            "note": src.get("note", ""),
            "text": raw.decode("utf-8"),
        }
        print(f"  [in] {k}: {meta[k]['path']}  ({meta[k]['run_id']}, "
              f"{len(data[k].get('cells_J', []))} cells, "
              f"{'清单钉死' if meta[k]['pinned'] else '非清单'})")
        if meta[k]["note"]:
            print(f"       ★ {meta[k]['note']}")

    # ---- 否定线（T25–T28）：独立清单，主清单不受影响 ----
    try:
        neg_m = load_manifest(NEG_MANIFEST_PATH)
    except EvidenceError as exc:
        for k in NEG_SRC:
            skip(f"S-{k}", str(exc))
        neg_m = None
    if neg_m is not None:
        for k, name in NEG_SRC.items():
            try:
                src = resolve(name, key=k, manifest=neg_m)
            except EvidenceError as exc:
                skip(f"S-{k}", str(exc))
                continue
            raw = src["raw"]
            data[k] = json.loads(raw.decode("utf-8"))
            meta[k] = {
                "name": name,
                "path": src["path"].relative_to(ROOT).as_posix(),
                "run_id": src["run_id"],
                "sha256": evidence_sha(raw),
                "sha256_bytes": hashlib.sha256(raw).hexdigest(),
                "pinned": src["pinned"],
                "note": src.get("note", ""),
                "text": raw.decode("utf-8"),
            }
            print(f"  [neg] {k}: {meta[k]['path']}  ({meta[k]['run_id']}, 否定线清单钉死)")
    return data, meta


def fmt_num(value, source: str, key: str, spec: str = ".3f", derived: bool = False) -> str:
    """格式化一个**直接取自 JSON** 的数，并登记溯源。

    `derived=True` 只用于"由 JSON 计算出来的量"（表格里的峰值、比例等），
    这类值不参与字面 grep —— 但它们自身在脚本里就是算式，可复核。
    """
    disp = format(value, spec)
    if not derived:
        TRACE.append({"kind": "direct", "display": disp, "value": value, "source": source, "key": key})
    return disp


def fmt_calc(value, spec: str = ".3f") -> str:
    return format(value, spec)


def fmt_derived(value: float, operands: list[str], ops: list[str], spec: str = ".2f") -> str:
    """登记一个**由 JSON 值算出来**的量。

    `operands` 是 trace 里已经存在的键（必须能在 `TRACE` 里查到 → S10c 会解析它们），
    `ops` 是左结合运算符（长度 = len(operands)-1）。这样"派生值"也能被机器重算，
    不是"脚本里写了算式、嘴上说可复核"。
    """
    disp = format(value, spec)
    TRACE_DERIVED.append({"kind": "derived", "display": disp, "value": value, "spec": spec,
                          "operands": operands, "ops": ops})
    return disp


def fmt_agg(value: float, data_key: str, field: str, flt: dict, stat: str = "mean",
            spec: str = ".2f", key: str | None = None) -> str:
    """登记一个**聚合量**（如"每 cell 墙钟均值"），S10c 会从原始 cells 重新聚合复核。

    `key` 只是给下游（演示页）取用的**语义标签**，不参与溯源判定。
    """
    disp = format(value, spec)
    row = {"kind": "derived", "display": disp, "value": value, "spec": spec,
           "agg": {"data": data_key, "field": field, "filter": flt, "stat": stat}}
    if key:
        row["key"] = key
    TRACE_DERIVED.append(row)
    return disp


def fmt_reratio(value: float, data_key: str, num: str, den: str, seed_set: str, budget: int,
                mode: str, user: int, spec: str = ".3f") -> str:
    """登记一个**重算出来的比值**（ΣJ(num)/ΣJ(den)），S10c 会用同样的配对规则重算复核。"""
    disp = format(value, spec)
    TRACE_DERIVED.append({"kind": "derived", "display": disp, "value": value, "spec": spec,
                          "reratio": {"data": data_key, "num": num, "den": den, "seed_set": seed_set,
                                      "budget": budget, "mode": mode, "user": user}})
    return disp


def fmt_agg_ep(value: float, data_key: str, field: str, idx: int, flt: dict,
               stat: str = "mean", spec: str = ".3f", key: str | None = None,
               nest: str = "ep") -> str:
    """登记一个**逐 episode 聚合量**（如"90 集的 `J_uav` 均值"）。

    与 `fmt_agg` 的区别：值本身是 list（`J_per_episode` / `served` / `alive_steps`），
    要先按 `idx` 取用户，再跨 cell 把所有 episode 摊平求统计量。
    ★ 这条存在的原因：复验 C 项里主表把"90 个初值重算的 `R`"与"seed=100 单格的 `J`"放在同一行，
      读者用同行 `J` 相除得不到同行 `R`（0.919/2.983≈0.308 ≠ 0.315）。统一口径后必须能同行复核。
    """
    disp = format(value, spec)
    row = {"kind": "derived", "display": disp, "value": value, "spec": spec,
           "agg_ep": {"data": data_key, "field": field, "idx": idx, "filter": flt,
                      "stat": stat, "nest": nest}}
    if key:
        row["key"] = key
    TRACE_DERIVED.append(row)
    return disp


def agg_ep(cells: list[dict], field: str, idx: int, flt: dict, stat: str = "mean",
           nest: str = "ep") -> tuple[float, int]:
    """跨 cell 聚合一个"每用户"的量。

    `nest="ep"`：`c[field]` 是**每集的二元组列表**（`J_per_episode`）⇒ 先摊平 episode，再取 `idx`；
    `nest="user"`：`c[field]` 是**每用户一个数**（`served` / `alive_steps`）⇒ 直接取 `[idx]`。
    """
    vals: list[float] = []
    for c in cells:
        if not all(c.get(k) == v for k, v in flt.items()):
            continue
        raw = c[field]
        if nest == "ep":
            vals.extend(e[idx] for e in raw)
        else:
            vals.append(raw[idx])
    if not vals:
        raise RuntimeError(f"逐集聚合无样本：{field}[{idx}] {flt}")
    return (sum(vals) / len(vals) if stat == "mean" else sum(vals)), len(vals)


def agg_cells(cells: list[dict], field: str, flt: dict, stat: str = "mean") -> tuple[float, int]:
    vals = [c[field] for c in cells if all(c.get(k) == v for k, v in flt.items())]
    if not vals:
        raise RuntimeError(f"聚合无样本：{field} {flt}")
    return (sum(vals) / len(vals) if stat == "mean" else sum(vals)), len(vals)


def apply_op(acc: float, op: str, x: float) -> float:
    if op == "+":
        return acc + x
    if op == "-":
        return acc - x
    if op == "*":
        return acc * x
    if op == "/":
        return acc / x
    if op == "max":
        return max(acc, x)
    if op == "maxabs":
        return max(abs(acc), abs(x))
    raise RuntimeError(f"未知算子 {op}")


# --------------------------------------------------- claim 绑定（复验 B 项的核心）
# 复验实测：把摘要里的 `0.315`（B200/unpaced）换成另一个**真实存在**的数 `1.223`（B100/unpaced），
# 97 个数全命中、S11a/S11b 都 PASS。⇒ "数字在结果池里出现过"不能证明它被引用到**正确条件**。
# 补法（不开发大型文稿检查框架）：正文里每个核心数字后面跟一个 claim-id 标记
#     `0.315`<!--C1.R_uav-->
# claim-id 绑定 (证据源, JSON 键, 条件)，显示值**由该键在证据里现算**。
# 于是"把正确数字放到错误条件"必然不一致 ⇒ 抓得住；而错别字/语义错误仍须人工读，不在此闸范围。
CLAIM_MARK = re.compile(r"`([^`\n]+)`[ \t]*<!--([A-Za-z][\w.\-]*)-->")
#: 一个 inline code 片段可以跟**多个** claim 标记（区间 `[a, b]` 就是这种），
#: 按"第 k 个小数 ↔ 第 k 个标记"绑定；数目不等 ⇒ 绑定不唯一 ⇒ FAIL。
CLAIM_SPAN = re.compile(r"`([^`\n]+)`((?:[ \t]*<!--[A-Za-z][\w.\-]*-->)+)")
CLAIM_ID = re.compile(r"<!--([A-Za-z][\w.\-]*)-->")
#: 协议常量不是"结果数"，不进结果闸（但它们确实不在证据里，所以必须显式列出来）。
#: 只放判据阈值/等价带这类**预注册常量**；任何新增都要在这里留痕。
PROTOCOL_CONSTS = {"0.90", "0.95", "1.10", "0.9", "1.1", "0.05"}
CLAIM_INDEX: dict[str, dict] = {}


def json_path(obj, key: str):
    """按点分路径取值，支持 `a.b[0].c` 与 `F2.primary/100/unpaced.R[0]`（段内可含 `/`）。"""
    cur = obj
    for tok in key.split("."):
        m = re.match(r"^([^\[\]]*)((?:\[\d+\])*)$", tok)
        if not m:
            raise RuntimeError(f"键路径写法不认识：{key}（段 `{tok}`）")
        name, idxs = m.group(1), m.group(2)
        if name:
            cur = cur[name]
        for i in re.findall(r"\[(\d+)\]", idxs):
            cur = cur[int(i)]
    return cur


def _reg_claim(cid: str, value, spec: str, source: str, binding: dict, note: str) -> str:
    if cid in CLAIM_INDEX:
        raise RuntimeError(f"claim-id 重复：{cid}")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise RuntimeError(f"claim {cid} 绑定的不是标量：{value!r}")
    disp = format(value, spec)
    row = {"id": cid, "display": disp, "value": value, "spec": spec,
           "source": source, "note": note, **binding}
    CLAIM_INDEX[cid] = row
    CLAIMS.append(row)
    # 同时进 TRACE：这样它既受 S10 的"字面可 grep"约束，也能按 JSON 键供下游（演示页）取用。
    if "json_key" in binding:
        TRACE.append({"kind": "direct", "display": disp, "value": value,
                      "source": source, "key": binding["json_key"]})
    return disp


def claim(cid: str, source: str, json_key: str, spec: str = ".3f", note: str = "") -> str:
    """登记"正文数字 ↔ 证据键"，返回显示串（正文/表格直接用它，不手写数字）。"""
    return _reg_claim(cid, json_path(DATA[source], json_key), spec, source,
                      {"json_key": json_key}, note or json_key)


def claim_calc(cid: str, source: str, keys: list[str], ops: list[str],
               spec: str = ".2f", note: str = "") -> str:
    """由若干证据键**算出**的派生量（求和、比值…），同样绑定到键而不是绑定到"某个数"。"""
    vals = [json_path(DATA[source], k) for k in keys]
    acc = vals[0]
    for op, x in zip(ops, vals[1:]):
        acc = apply_op(acc, op, x)
    return _reg_claim(cid, acc, spec, source, {"keys": keys, "ops": ops},
                      note or " = ".join([keys[0]] + [f"{o} {k}" for o, k in zip(ops, keys[1:])]))


def cdisp(cid: str) -> str:
    """取已登记 claim 的显示串（同一数字在表里被多处引用时用，避免重复登记）。"""
    return CLAIM_INDEX[cid]["display"]


def claim_num(cid: str, source: str, json_key: str, spec: str = ".3f", note: str = "") -> str:
    """同 `claim()`，但允许证据里存的是**字符串数字**。

    66–69（否定线）的证据把标量写成了 `str`（脚本内部 `_s()` 口径），而 58–60 是 float。
    这里不做数据类型"兼容"式的宽容：解析不出数、或不是有限值，**直接抛错**——
    拿不到数就不许往正文写数（#14 报错优于给假数字）。
    """
    raw = json_path(DATA[source], json_key)
    try:
        v = float(raw)
    except (TypeError, ValueError):
        raise RuntimeError(f"claim {cid} 绑定的不是数值：{source}.{json_key} → {raw!r}") from None
    if not math.isfinite(v):
        raise RuntimeError(f"claim {cid} 绑定的不是有限值：{source}.{json_key} → {v!r}")
    if spec == "d":
        if not float(v).is_integer():
            raise RuntimeError(f"claim {cid} 声称是计数却不是整数：{source}.{json_key} → {v!r}")
        v = int(v)
    return _reg_claim(cid, v, spec, source, {"json_key": json_key}, note or json_key)


def merged_crossing(pts_a: list[dict], pts_b: list[dict], mode: str = "unpaced") -> dict | None:
    """把基线扫描与**补点**合起来排 `x`，找 `R_uav` 跨过 1 的相邻两格 ⇒ 网格括区间。

    ★ 复验 D 项：图 (a) 画了补点，却仍标基线扫描的括区间 `(0.92, 7.57)` ——
      因为基线在 `x=0.921` 与 `x=7.571` 之间没有设计点。补点把这段填上后，括区间应收窄到
      `[5.827, 7.571]`。括区间仍是**网格间隔**，不是置信区间（不在交点处做重采样）。
    """
    merged = []
    for tag, pts in (("59", pts_a), ("59b", pts_b)):
        for p in pts:
            if p.get("mode") == mode:
                merged.append({"x": p["x"], "R": p["R_uav"], "budget": p["budget"], "src": tag})
    merged.sort(key=lambda p: (p["x"], p["budget"]))
    for i in range(len(merged) - 1):
        a, b = merged[i], merged[i + 1]
        if a["R"] < 1.0 <= b["R"]:
            return {"lo": a, "hi": b, "n_pts": len(merged), "idx_lo": i, "idx_hi": i + 1}
    return None


def check_claims_in_text(text: str) -> tuple[list[str], int]:
    """核对正文里每个 `数字`<!--claim-id-->：数字必须**逐字等于**该 claim 的显示值。

    支持一个片段跟多个标记（区间）：按位置一一绑定；数目不等即判"绑定不唯一"。
    """
    bad: list[str] = []
    n = 0
    for m in CLAIM_SPAN.finditer(text):
        span = m.group(1).strip()
        ids = CLAIM_ID.findall(m.group(2))
        nums = SIGNED_DEC_RE.findall(span)
        # ★ v0.3：计数类 claim（"反例 9 个"）的显示串是**整数**，没有小数点。
        # 旧口径会把它读成"0 个数却跟了 1 个标记"⇒ 假 FAIL。这里把"整段就是一个整数"当唯一数。
        # 注意：S11a 的**覆盖闸仍只抽小数**，所以这条放宽不会让凭空写的整数蒙混过关。
        if not nums and re.fullmatch(r"-?\d+", span):
            nums = [span]
        n += len(ids)
        line = text.count("\n", 0, m.start()) + 1
        if len(nums) != len(ids):
            bad.append(f"L{line}: `{span}` 有 {len(nums)} 个数却跟了 {len(ids)} 个 claim 标记 ⇒ 绑定不唯一")
            continue
        for num, cid in zip(nums, ids):
            c = CLAIM_INDEX.get(cid)
            if c is None:
                bad.append(f"L{line}: 未知 claim-id `{cid}`（不在登记表里）")
                continue
            if num != c["display"]:
                bad.append(f"L{line}: `{num}` ≠ claim {cid} 的 {c['display']}（绑定 {c['note']}）")
    return bad, n


def _first_mark_span(text: str) -> tuple[int, int, str, str] | None:
    m = CLAIM_MARK.search(text)
    if not m:
        return None
    return m.start(1), m.end(1), m.group(1).strip(), m.group(2)


def mutate_claim_digit(text: str) -> str:
    """阴性对照 ①：把第一个**带 claim 标记**的数字末位改掉。必须被抓。"""
    hit = _first_mark_span(text)
    if hit is None:
        return text
    s0, s1, span, _cid = hit
    m = DEC_RE.search(span)
    if not m:
        return text
    i = m.end() - 1
    new = "1" if span[i] != "1" else "2"
    return text[:s0] + span[:i] + new + span[i + 1:] + text[s1:]


def mutate_claim_condition(text: str) -> str:
    """阴性对照 ②：把一个 claim 的数字换成**另一个真实存在**的数（错条件/错用户）。

    这正是复验里"0.315 被换成 1.223 仍然通过"的复现 —— 改后必须 FAIL，否则闸没有分辨力。
    """
    hit = _first_mark_span(text)
    if hit is None:
        return text
    s0, s1, span, cid = hit
    # 只挑"整个显示串就是一个数"的 claim，避免把区间串拼进来
    others = [c for c in CLAIMS
              if c["id"] != cid and c["display"] != span and re.fullmatch(r"-?\d+\.\d+", c["display"])]
    if not others or not re.fullmatch(r"-?\d+\.\d+", span):
        return text
    base = abs(float(span))
    # 优先挑"量级相近但确实不同"的那个：换成一个 6000 这种量级差太远的数，
    # 虽然也是"真实数放错处"，但不像复验里 0.315→1.223 那种**看起来完全合理**的错。
    cand = [c for c in others if 0.05 <= abs(float(c["display"])) <= 3.0] or others
    pool = sorted(cand, key=lambda c: -abs(abs(float(c["display"])) - base))
    return text[:s0] + pool[0]["display"] + text[s1:]


# ---------------- 初稿数字闸（S11）
# 规则：**凡当结果引用的数**在初稿里写成 inline code；S11 抽"整段只由数字/单位/分隔符组成"的片段，
# 再要求其中**每一个小数**都能在 trace 的 display 里找到（整数与协议量不在闸内，见预注册 §2）。
SPAN_OK = re.compile(r"^[\d.,\s\[\]()–—\-+×%µms步]{1,48}$")
DEC_RE = re.compile(r"\d+\.\d+")
#: v0.3：正文要引**带符号**的区间（`ΔR` 的下界可以为负）⇒ 取数时带符号。
#: 否则 `-0.0033` 会被抽成 `0.0033`，既追不到 claim、也追不到 trace（假 FAIL）。
SIGNED_DEC_RE = re.compile(r"-?\d+\.\d+")


def extract_result_tokens(text: str) -> list[tuple[str, int, str]]:
    """返回 [(片段, 行号, 小数)] —— 一个片段里有几个小数就产生几条（每个都要能溯源）。"""
    out: list[tuple[str, int, str]] = []
    for m in re.finditer(r"`([^`\n]+)`", text):
        s = m.group(1).strip()
        if not SPAN_OK.match(s):
            continue
        line = text.count("\n", 0, m.start()) + 1
        for num in SIGNED_DEC_RE.findall(s):
            out.append((s, line, num.replace("−", "-")))
    return out


def mutate_first_token(text: str) -> str:
    """把第一个受闸小数的末位数字改掉 —— 阴性对照用（#39：先问什么错误能打死这条检查）。"""
    for m in re.finditer(r"`([^`\n]+)`", text):
        s = m.group(1).strip()
        if not SPAN_OK.match(s):
            continue
        hit = DEC_RE.search(s)
        if not hit:
            continue
        i = hit.end() - 1
        new = "1" if s[i] != "1" else "2"
        return text[: m.start(1)] + s[:i] + new + s[i + 1:] + text[m.end(1):]
    return text


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
        if t.get("kind", "direct") != "direct":
            continue
        src_meta = meta.get(t["source"]) or next(
            (m for m in meta.values() if m["path"] == t["source"]), None)
        if src_meta is None:
            bad.append(f"{t['source']} 不在已加载的证据里")
            continue
        lit = repr(t["value"])
        if lit not in src_meta["text"]:
            bad.append(f"{t['key']} = {lit} 未出现在 {t['source']}")
    check("S10 每个显示值可在证据 JSON 中字面 grep 到", not bad,
          f"{sum(1 for t in TRACE if t.get('kind', 'direct') == 'direct')} 格，失败 {len(bad)}"
          + ("" if not bad else "；" + "；".join(bad[:5])))

    # ---- S10c 派生值：由它自己声明的操作数键重算一遍 ----
    km = {t["key"]: t["value"] for t in TRACE}
    bad_d: list[str] = []
    for dv in TRACE_DERIVED:
        try:
            if "agg" in dv:
                a = dv["agg"]
                base = data.get(a["data"], {}).get("cells_J")
                if not base:
                    bad_d.append(f"聚合对象 {a['data']}.cells_J 不存在")
                    continue
                v, _n = agg_cells(base, a["field"], a["filter"], a["stat"])
            elif "agg_ep" in dv:
                a = dv["agg_ep"]
                base = data.get(a["data"], {}).get("cells_J")
                if not base:
                    bad_d.append(f"逐集聚合对象 {a['data']}.cells_J 不存在")
                    continue
                v, _n = agg_ep(base, a["field"], a["idx"], a["filter"], a["stat"],
                               a.get("nest", "ep"))
            elif "reratio" in dv:
                rr = dv["reratio"]
                base = data.get(rr["data"], {}).get("cells_J")
                if not base:
                    bad_d.append(f"重算对象 {rr['data']}.cells_J 不存在")
                    continue
                v, _n = reratio(base, num=rr["num"], den=rr["den"], seed_set=rr["seed_set"],
                                budget=rr["budget"], mode=rr["mode"], user=rr["user"])
            else:
                miss = [k for k in dv["operands"] if k not in km]
                if miss:
                    bad_d.append(f"{dv['display']} 的操作数未登记：{miss}")
                    continue
                vals = [km[k] for k in dv["operands"]]
                acc = vals[0]
                for op, x in zip(dv["ops"], vals[1:]):
                    acc = apply_op(acc, op, x)
                v = acc
        except Exception as exc:  # noqa: BLE001
            bad_d.append(f"{dv['display']} 重算异常：{exc}")
            continue
        if abs(v - dv["value"]) > 1e-9:
            bad_d.append(f"{dv['display']} 重算 {v!r} ≠ 登记 {dv['value']!r}")
        elif format(v, dv["spec"]) != dv["display"]:
            bad_d.append(f"{dv['display']} 与重算格式化 {format(v, dv['spec'])} 不一致")
    check("S10c 每个派生值都能由它的操作数重算（含聚合）", not bad_d,
          f"{len(TRACE_DERIVED)} 个派生量，失败 {len(bad_d)}" + ("" if not bad_d else "；" + "；".join(bad_d[:5])))

    # ---- S0 证据来源必须钉死（不是 latest 漂移） ----
    unpinned = [k for k, m in meta.items() if not m["pinned"]]
    check("S0 证据来自冻结清单（不是 latest 漂移）", not unpinned,
          f"{len(meta)} 份证据均已钉死" if not unpinned else f"未钉死：{unpinned}（--latest？）")

    # ---- S12 主表同行自洽：同一批 90 集均值的 J 相除必须等于 R ----
    if "58" in data:
        c58 = data["58"]["cells_J"]
        fb0 = {"seed_set": "primary", "budget": 200, "mode": "unpaced"}
        worst, rows_dbg = 0.0, []
        for pred in ("linear", "hybrid", "truedyn"):
            for user, tag in ((0, "uav"), (1, "cart")):
                jl, _n = agg_ep(c58, "J_per_episode", user, {**fb0, "predictor": pred})
                jz, _ = agg_ep(c58, "J_per_episode", user, {**fb0, "predictor": "zoh"})
                r, _ = reratio(c58, num=pred, seed_set="primary", budget=200, mode="unpaced", user=user)
                worst = max(worst, abs(jl / jz - r))
                rows_dbg.append(f"{pred}/{tag} {jl / jz:.6f} vs {r:.6f}")
        check("S12 主表同行 `J` 相除 = 同行 `R`（同批 90 集）", worst < 1e-9,
              f"max|Δ|={worst:.2e}；{rows_dbg[0]}；{rows_dbg[2]}")
        # 阴性对照：换成旧版做法（seed=100 单格）必须被抓 —— 否则说明 S12 没有分辨力
        fb1 = {**fb0, "seed": 100}
        jl1, _ = agg_ep(c58, "J_per_episode", 0, {**fb1, "predictor": "linear"})
        jz1, _ = agg_ep(c58, "J_per_episode", 0, {**fb1, "predictor": "zoh"})
        r1, _ = reratio(c58, num="linear", seed_set="primary", budget=200, mode="unpaced", user=0)
        check("S12b 阴性对照：只取 seed=100 时 S12 必须失败", abs(jl1 / jz1 - r1) > 1e-4,
              f"seed=100 时 J 比 {jl1 / jz1:.6f} vs R {r1:.6f}（差 {abs(jl1 / jz1 - r1):.2e}）")
    else:
        skip("S12", "58 证据缺失")

    # ---- S10d claim 绑定：键必须解析得出，且原值能在该证据里字面 grep 到 ----
    bad_c: list[str] = []
    for c in CLAIMS:
        try:
            if "json_key" in c:
                v = json_path(data[c["source"]], c["json_key"])
            else:
                vals = [json_path(data[c["source"]], k) for k in c["keys"]]
                v = vals[0]
                for op, x in zip(c["ops"], vals[1:]):
                    v = apply_op(v, op, x)
        except Exception as exc:  # noqa: BLE001
            bad_c.append(f"{c['id']}: 键取不到（{exc}）")
            continue
        src_text = meta.get(c["source"], {}).get("text", "")
        if "json_key" in c and repr(v) not in src_text:
            bad_c.append(f"{c['id']}: {repr(v)} 不在 {c['source']} 的文本里（键指错文件？）")
        elif abs(v - c["value"]) > 1e-12 or format(v, c["spec"]) != c["display"]:
            bad_c.append(f"{c['id']}: 重算 {v!r} ≠ 登记 {c['display']}")
    check("S10d 每条 claim 的绑定键都能解析并复现显示值", not bad_c,
          f"{len(CLAIMS)} 条 claim，失败 {len(bad_c)}"
          + ("" if not bad_c else "；" + "；".join(bad_c[:5])))

    # ---- S11 初稿：claim 绑定闸（强）+ 数字覆盖（弱）+ 两条阴性对照 ----
    dp = Path(DRAFT_PATH)
    if not dp.exists():
        skip("S11 初稿 claim 闸", f"初稿不存在：{dp.as_posix()}（对象不存在 ⇒ SKIP，不判 PASS/FAIL）")
    else:
        text = dp.read_text(encoding="utf-8")
        bad_m, n_mark = check_claims_in_text(text)
        check("S11c 正文每个 claim 标记的数字 = 该 claim 的绑定值", (not bad_m) and n_mark >= 20,
              f"{n_mark} 个 claim 标记，失败 {len(bad_m)}"
              + ("" if not bad_m else "；" + "；".join(bad_m[:6]))
              + ("" if n_mark >= 20 else "；★ 标记 <20 ⇒ 非退化守卫不满足"))
        bad1, _ = check_claims_in_text(mutate_claim_digit(text))
        check("S11d 阴性对照①：改一位小数必须被抓", len(bad1) >= 1,
              f"变异后失败 {len(bad1)} 处" + (f"（{bad1[0]}）" if bad1 else "；★ 闸没有分辨力"))
        bad2, _ = check_claims_in_text(mutate_claim_condition(text))
        check("S11e 阴性对照②：**真实数字放到错误条件**必须被抓", len(bad2) >= 1,
              f"变异后失败 {len(bad2)} 处" + (f"（{bad2[0]}）" if bad2 else "；★ 这正是不换 claim-id 时抓不到的那类错"))
        # 弱覆盖：正文里**所有** inline code 小数仍须能在 trace/claim 里找到。
        # ★ 这条只保证"不是凭空写的数"，**不保证条件正确** —— 条件正确由 S11c 的 claim 绑定负责。
        disp = ({t["display"] for t in TRACE} | {d["display"] for d in TRACE_DERIVED}
                | {c["display"] for c in CLAIMS})
        toks = extract_result_tokens(text)
        missing = [(s, ln) for (s, ln, num) in toks if num not in disp and num not in PROTOCOL_CONSTS]
        n_proto = sum(1 for (_s, _ln, num) in toks if num in PROTOCOL_CONSTS)
        check("S11a 覆盖（弱）：正文出现的每个小数都能在 trace/claim 找到", (not missing) and len(toks) >= 15,
              f"抽出 {len(toks)} 个结果数（另有 {n_proto} 个协议常量免检），未命中 {len(missing)}"
              + ("" if not missing else "；" + "；".join(f"L{ln}:`{s}`" for s, ln in missing[:6]))
              + ("" if len(toks) >= 15 else "；★ 抽到的数 <15 ⇒ 非退化守卫不满足"))
        mut = mutate_first_token(text)
        toks2 = extract_result_tokens(mut)
        miss2 = [(s, ln) for (s, ln, num) in toks2 if num not in disp]
        check("S11b 阴性对照（弱闸）：改一位小数必须被 S11a 抓住", len(miss2) >= 1,
              f"变异后未命中 {len(miss2)} 个"
              + (f"（如 L{miss2[0][1]}:`{miss2[0][0]}`）" if miss2 else "；★ 说明弱闸没有分辨力"))


# --------------------------------------------------------------------------- 渲染
def render(data: dict, meta: dict) -> tuple[str, dict]:
    global DATA
    DATA = data          # claim 绑定要用：注册时立刻从证据取值，不读手写表
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
    A("| 逻辑名 | run-id | 文件 | 大小(B) | sha256_norm（前 16，行尾归一） | cells | checks_failed |")
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
    A("★ **输入由 `docs/evidence_manifest_p3.json` 钉死**：以上四份就是要引用的那四份，"
      "新跑一个 run 不会静默换掉它们。换证据必须 `python scripts/63_freeze_manifest.py --allow-update`，"
      "旧版会另存为 `evidence_manifest_p3.v<N>.json`。")
    A("★ 指纹用 `sha256_norm`（行尾 `\\r\\n`→`\\n` 后计算）：本机 `write_text` 写 CRLF、"
      "`core.autocrlf=input` 提交时又规范成 LF ⇒ **工作树字节 sha 与 git blob sha 必然不同**，"
      "不能拿字节 sha 当「被篡改」的证据；原始字节 sha 另记在清单里备查。")
    A("")

    # ---------------- 1 协议 ----------------
    A("## 1 共用协议（三块表一致，不随表变）")
    A("")
    c58 = data.get("58", {}).get("config", {})
    A(f"- 拓扑/任务：`N={c58.get('N')}`，`max_steps={c58.get('window')}`，`warmup={c58.get('warmup')}`，"
      f"臂 = `{c58.get('arm')}`（仅 `age`），`absorbed={c58.get('absorbed')}`（P3-2 里 `absorbed_uav := escape_radius`）。")
    A("- 指标：`R_i = Σ_e J_i(linear) / Σ_e J_i(zoh)`，**同种子同 episode 配对**；")
    A("  区间 = **cluster bootstrap，重采样单位 = `reset()` 初值**（不是 episode 次），`n_boot=10000`。")
    A("- `J_uav` 单位 m（越大越坏）；`served` = **发送尝试**帧数（按 `attempted_idx` 计数，"
      "**不是成功送达**；送达数见 `delivery_rate()[\"frames\"]`）；"
      "`t>B` = **预算用完后**（`t>B`）仍在发的帧数。")
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
        A("**一整行同口径**：`R` / `J` / `served` 全部由**同一批 90 个初值**（3 种子 × 30 集）聚合 ——")
        A("`J` = 90 集逐集均值（m）；`served` = 每种子 30 集合计、再对 3 种子取均值（预算打满 ⇒ 两用户合计恒 `6000`）。")
        A("⇒ **同行 `J` 相除必须等于同行 `R`**（S12 逐行核；旧版 `J` 取 `seed=100` 单格，`0.919/2.983≈0.308≠0.315`，已改）。")
        A("")
        A("| 预测器 | `R_uav` | 95% 区间(UAV) | `R_cart` | 95% 区间(cart) | "
          "`J_uav` lin/zoh（90 集均值, m） | `J_cart` lin/zoh（90 集均值） | `served` u0/u1（30 集合计均值） |")
        A("|---|---|---|---|---|---|---|---|")
        fb = {"seed_set": "primary", "budget": 200, "mode": "unpaced"}
        jz0, _n90 = agg_ep(cells, "J_per_episode", 0, {**fb, "predictor": "zoh"})
        jz1, _ = agg_ep(cells, "J_per_episode", 1, {**fb, "predictor": "zoh"})
        sz0, _ = agg_ep(cells, "served", 0, {**fb, "predictor": "zoh"}, nest="user")
        sz1, _ = agg_ep(cells, "served", 1, {**fb, "predictor": "zoh"}, nest="user")
        jz0_d = fmt_agg_ep(jz0, "58", "J_per_episode", 0, {**fb, "predictor": "zoh"},
                           key="58.J.zoh.uav")
        jz1_d = fmt_agg_ep(jz1, "58", "J_per_episode", 1, {**fb, "predictor": "zoh"},
                           key="58.J.zoh.cart")
        sz0_d = fmt_agg_ep(sz0, "58", "served", 0, {**fb, "predictor": "zoh"},
                           spec=".1f", key="58.served.zoh.uav", nest="user")
        sz1_d = fmt_agg_ep(sz1, "58", "served", 1, {**fb, "predictor": "zoh"},
                           spec=".1f", key="58.served.zoh.cart", nest="user")
        for pred, tag in (("linear", "`Linear`（**冻结方法**）"),
                          ("hybrid", "`hybrid`（历史参考）"),
                          ("truedyn", "`TrueDyn`〔diagnostic：特权参考〕")):
            ru, _ = reratio(cells, num=pred, seed_set="primary", budget=200, mode="unpaced", user=0)
            rc, _ = reratio(cells, num=pred, seed_set="primary", budget=200, mode="unpaced", user=1)
            if pred == "linear":
                ciu = (f"[{claim('C1.ci_uav_lo', '58', 'F1.ci_uav[0]', '.3f', 'UAV R 区间下界')}, "
                       f"{claim('C1.ci_uav_hi', '58', 'F1.ci_uav[1]', '.3f', 'UAV R 区间上界')}]")
                cic = (f"[{claim('C1.ci_cart_lo', '58', 'F1.ci_cart[0]', '.3f', 'CartPole R 区间下界')}, "
                       f"{claim('C1.ci_cart_hi', '58', 'F1.ci_cart[1]', '.3f', 'CartPole R 区间上界')}]")
                ru_disp = f"**{claim('C1.R_uav', '58', 'F1.R[0]', '.3f', 'B200/unpaced/linear/uav/primary')}**"
                rc_disp = claim('C1.R_cart', '58', 'F1.R[1]', '.3f', 'B200/unpaced/linear/cart/primary')
            else:
                ciu = cic = "—（未做区间）"
                ru_disp = fmt_reratio(ru, "58", pred, "zoh", "primary", 200, "unpaced", 0)
                rc_disp = fmt_reratio(rc, "58", pred, "zoh", "primary", 200, "unpaced", 1)
            jl0, _ = agg_ep(cells, "J_per_episode", 0, {**fb, "predictor": pred})
            jl1, _ = agg_ep(cells, "J_per_episode", 1, {**fb, "predictor": pred})
            sl0, _ = agg_ep(cells, "served", 0, {**fb, "predictor": pred}, nest="user")
            sl1, _ = agg_ep(cells, "served", 1, {**fb, "predictor": pred}, nest="user")
            A(f"| {tag} | {ru_disp} | {ciu} | {rc_disp} | {cic} | "
              f"{fmt_agg_ep(jl0, '58', 'J_per_episode', 0, {**fb, 'predictor': pred}, key=f'58.J.{pred}.uav')}"
              f"/{jz0_d} | "
              f"{fmt_agg_ep(jl1, '58', 'J_per_episode', 1, {**fb, 'predictor': pred}, key=f'58.J.{pred}.cart')}"
              f"/{jz1_d} | "
              f"{fmt_agg_ep(sl0, '58', 'served', 0, {**fb, 'predictor': pred}, spec='.1f', nest='user', key=f'58.served.{pred}.uav')}"
              f"/{fmt_agg_ep(sl1, '58', 'served', 1, {**fb, 'predictor': pred}, spec='.1f', nest='user', key=f'58.served.{pred}.cart')}"
              f"（zoh {sz0_d}/{sz1_d}） |")
        A("")
        A("★ 上表每一行的 `J` 都是 **90 集均值**（`n=90`，不是 `seed=100` 的 30 集），因此 `J_lin/J_zoh` 与 `R` 逐行自洽。")
        A("★ `hybrid`/`TrueDyn` 的 `R` 是**点估计**（该轮未对它们做 bootstrap），只能与 `Linear` 比方向，不能比区间。")
        A("★ `served` 是**任务级发送尝试**计数（用户 0 / 用户 1；按 `attempted_idx`，非送达），"
          "两用户合计恒为 `6000`（= 30 集 × `B=200` 的预算上限）⇒")
        A("  它**不是**「总资源可自由伸缩」的证据：总量被预算钉死，变化只能在两个用户之间挪。")
        A("  汇总尝试数恒等只说明「预算打满」，**不说明送达**：真正送达 = 尝试 − `budget_lost`，"
          "两者差额随链路成功率变化。")
        A("")
        A("### 2b 断言（预注册 §4 判据，逐条给值）")
        A("")
        A("| 断言 | 值 | 来源 |")
        A("|---|---|---|")
        A(f"| 判词 | **`{F1['verdict']}`** | `58.F1.verdict` |")
        A(f"| UAV 区间上界 < 1.0 | {fmt_num(F1['ci_uav'][1], src, 'F1.ci_uav[1]')} < 1.0 ⇒ 成立 | `58.F1.ci_uav[1]` |")
        A(f"| `R_cart` ≤ 1.10（**预设不牺牲判据**，点估计） | {fmt_num(F1['R'][1], src, 'F1.R[1]')} ⇒ 点估计在带内 | `58.F1.R[1]` |")
        A(f"| 逐种子 UAV 同向 | `{F1['seeds_uav_below_1']}` | `58.F1.seeds_uav_below_1` |")
        A(f"| 逐 cluster 改善比例（UAV） | {fmt_num(F2['frac_clusters_below_1'][0], src, 'F2.primary/200/unpaced.frac_clusters_below_1[0]', '.3f')}"
          f"（{fmt_calc(F2['frac_clusters_below_1'][0]*90, '.0f')}/90 个初值） | `58.F2` |")
        A(f"| 逐 cluster 改善比例（CartPole） | {fmt_num(F2['frac_clusters_below_1'][1], src, 'F2.primary/200/unpaced.frac_clusters_below_1[1]', '.3f')}"
          f"（{fmt_calc(F2['frac_clusters_below_1'][1]*90, '.0f')}/90） ⇒ **不构成收益** | `58.F2` |")
        A(f"| 逐集比值中位（CartPole） | {fmt_num(F2['cluster_ratio_median'][1], src, 'F2.primary/200/unpaced.cluster_ratio_median[1]')}（正好 1） | `58.F2` |")
        A("")
        A("★ **CartPole 只能写「未超出预设不牺牲判据 1.10」，不写「两用户都变好」，也不写「已证明等价」**：")
        A("  区间含 1、逐 cluster 仅 40%；且 `R_cart` 的 95% 区间是 `[0.821, 1.028]`，**并没有完整落在**对称等价带")
        A("  `[0.9, 1.1]` 里（下界 `0.821` 已经低出 0.9 之外 —— 那是「变好」的一侧，但仍然说明这不是一个")
        A("  双侧被夹住的等价结论）。⇒ 现有材料只支持**点估计落在预设带上**，不构成等价性证明。")
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
        _ru_unp = d["F2"]["primary/100/unpaced"]["R"][0]
        _ru_pac = d["F2"]["primary/100/paced"]["R"][0]
        _ratio = fmt_derived(_ru_unp / _ru_pac,
                             ["F2.primary/100/unpaced.R[0]", "F2.primary/100/paced.R[0]"], ["/"])
        A(f"★ **`B=100 unpaced` 的 `R_uav` {fmt_num(_ru_unp, src, 'F2.primary/100/unpaced.R[0]')} 与 `paced` "
          f"{fmt_num(_ru_pac, src, 'F2.primary/100/paced.R[0]')} 差 {_ratio}×、符号相反** ⇒ "
          "同一预测器在不同时序下相反；`unpaced` 的 `t>B` 发帧为 **0**（断服），`paced` 不为 0。")
        A("★ `B=200` 时 `paced ≡ unpaced`（`B=L` ⇒ 配速 no-op），两行数值相同是恒等式，不是「协同为零」。")
        A("")
    else:
        A("**58 证据缺失 ⇒ 本节 SKIP**")
        A("")

    # ---------------- 3 P3-1 ----------------
    A("## 3 P3-1 收益/退化范围（**开发结果**：开发集 `{0,1,2}`；确认集只做落点复核）")
    A("")
    A("★ 措辞纪律（复验 D）：本节给的是**当前扫描条件下的收益/退化范围**，**不是「安全域」**——")
    A("  没有安全性证明，也没有可外推的阈值；横轴 `age` 内生且受存活选择影响。")
    A("")
    if "59" in data:
        d = data["59"]
        src = meta["59"]["path"]
        A("横轴 `x` = **实测** UAV `age` 算术均值（`alive` 且 `t≥41` 的决策点，`linear` 轨迹）；纵轴 `R_uav`。")
        A("★ 轴**内生**：预测改了轨迹 ⇒ 改了 `age` 分布（`x_zoh` 列给出替代轴的同口径值，见 trace）。")
        A("")
        A("| B | `mode` | `x`（实测均值） | `R_uav` | 95% 区间 | `J_uav` lin/zoh | `J_cart` lin/zoh | `served` lin/zoh（尝试） | `alive` lin/zoh | `t>B` 发帧 |")
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
        # `paced` 分支的取值范围由 points 现算（不再手写 0.313–0.618）
        dpts = d["points"]
        _pi = [i for i, p in enumerate(dpts) if p["mode"] == "paced"]
        rmin = min(_pi, key=lambda i: dpts[i]["R_uav"])
        rmax = max(_pi, key=lambda i: dpts[i]["R_uav"])
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
        A(f"| `paced` 分支取值范围 | {fmt_num(dpts[rmin]['R_uav'], src, f'points[{rmin}].R_uav')}"
          f"–{fmt_num(dpts[rmax]['R_uav'], src, f'points[{rmax}].R_uav')}（全 <1） | 见上表 `paced` 各行 |")
        _mc = merged_crossing(dpts, data["59b"]["points"] if "59b" in data else [])
        if _mc:
            _i59 = {(p["budget"], p["mode"]): i for i, p in enumerate(dpts)}
            _i59b = {(p["budget"], p["mode"]): i for i, p in enumerate(data["59b"]["points"])}
            _klo = "points[" + str(_i59b[(_mc["lo"]["budget"], "unpaced")]) + "].x"
            _khi = "points[" + str(_i59[(_mc["hi"]["budget"], "unpaced")]) + "].x"
            _lo = claim("C5.br_lo", "59b", _klo, ".3f", "交点网格下界（补点 B=160 的 x）")
            _hi = claim("C5.br_hi", "59", _khi, ".3f", "交点网格上界（B=150 的 x）")
            A(f"| 交点（**补点后**的网格括区间） | `[{_lo}, {_hi}]` | 相邻设计点之间 `R_uav` 跨过 1；"
              f"**括区间 = 网格间隔，不是置信区间**（交点处未做重采样） |")
            A(f"★ 旧图标的 `(0.92, 7.57)` 是**基线扫描**的括区间（`x=0.92` 与 `7.57` 之间没有设计点）；"
              f"补点（`B∈{{160,175,190}}`）把这段填上后应改为 `[{_lo}, {_hi}]`。图 (a) 已改。")
        else:
            A("| 交点 | — | 合并补点后仍未跨过 1 ⇒ 无括区间 |")
        A("")
        A("### 3d 确认集落点复核（不参与任何选择）")
        A("")
        A("| 档 | 开发集 `R_uav` | 确认集 `R_uav` | Δ |")
        A("|---|---|---|---|")
        for key in ("150/unpaced", "150/paced", "200/unpaced"):
            o = d["F3"][key]
            A(f"| {key} | {fmt_num(o['R_dev'][0], src, f'F3.{key}.R_dev[0]')} | "
              f"{fmt_num(o['R_conf'][0], src, f'F3.{key}.R_conf[0]')} | "
              f"{fmt_num(o['d_uav'], src, f'F3.{key}.d_uav')} |")
        A("")
        _deltas = [abs(d["F3"][k]["d_uav"]) for k in ("150/unpaced", "150/paced", "200/unpaced")]
        A(f"★ 三格 |Δ| 的最大者 = {fmt_derived(max(_deltas), ['F3.150/unpaced.d_uav', 'F3.150/paced.d_uav', 'F3.200/unpaced.d_uav'], ['maxabs', 'maxabs'], '.3f')}"
          "（逐格值见上表）⇒ 落点一致；这条只说明「表不是开发集挑出来的」，**不构成第二次独立确认**。")
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
        _W_s, _W_l = d["windows"]["short"]["W_expected"], d["windows"]["long"]["W_expected"]
        _xs_s, _xs_l = d["brackets"]["short"]["x_star"], d["brackets"]["long"]["x_star"]
        _rat_s, _rat_l = d["x_star_over_W"]["short"]["x_star_over_W"], d["x_star_over_W"]["long"]["x_star_over_W"]
        A(f"★ `W` ×{fmt_derived(_W_l / _W_s, ['windows.long.W_expected', 'windows.short.W_expected'], ['/'], '.1f')} "
          f"而 `x*` 只 ×{fmt_derived(_xs_l / _xs_s, ['brackets.long.x_star', 'brackets.short.x_star'], ['/'])}；"
          f"`x*/W` {fmt_num(_rat_s, src, 'x_star_over_W.short.x_star_over_W')} → "
          f"{fmt_num(_rat_l, src, 'x_star_over_W.long.x_star_over_W')}"
          f"（单调降 {fmt_derived(_rat_s / _rat_l, ['x_star_over_W.short.x_star_over_W', 'x_star_over_W.long.x_star_over_W'], ['/'], '.1f')}×）"
          "⇒ **探索性比例偏离**（预注册标「只报不判」，未作确认性判决）。")
        A("★ 复验 D 已把这一条从「比例律被否证」降级：`W` 与 `x*` 都是**点估计**，而交点的")
        A("  括区间是**网格间隔不是置信区间** ⇒ 不能拿区间不重叠当比例律检验；`x*/W` 也在分母里含选择效应。")
        A("★ **不得**把 `INCONCLUSIVE` 写成「窗口无关」：点估计单调（5.79 < 7.47 < 8.16），方向不反。")
        A("★ **结构性限制**：改窗口 ⇒ 改存活 ⇒ 改横轴（`age` 均值同时是「陈旧度」与「存活时长」的函数）；"
          "`x*/W` 的下降有一部分是**选择效应**。")
        A("")
    else:
        A("**60 证据缺失 ⇒ 本节 SKIP**")
        A("")

    # ---------------- 5 主张 ----------------
    # ★ 复验 D 项：把越过证据的措辞逐条收紧（比例律/安全域/等价/上界/多分帧归因/「必要」）。
    A("## 5 主张与否定条件（数字 ↔ 条件的绑定见 `docs/claims_p3.json`）")
    A("")
    A("| # | 主张 / 否定 | 值 | claim-id |")
    A("|---|---|---|---|")
    n2_row = ""      # 由 59 计算、排在 C7 之后（"否定"集中放在末尾）
    if "58" in data:
        d58 = data["58"]
        F2 = d58["F2"]["primary/200/unpaced"]
        A(f"| C1 | `Linear` 补偿在**快刷新**档（B=200 unpaced）降低 UAV 代价 | `R_uav` "
          f"{cdisp('C1.R_uav')}，区间 [{cdisp('C1.ci_uav_lo')}, {cdisp('C1.ci_uav_hi')}]；逐 cluster 改善 "
          f"{claim('C1.frac_uav', '58', 'F2.primary/200/unpaced.frac_clusters_below_1[0]', '.3f', '90 个初值里 R<1 的比例（UAV）')}"
          f"（= 90/90） | `C1.*` |")
        _sz0 = claim('C2.sz0', '58', 'F1.served.zoh[0]', '.1f', 'zoh UAV 发送尝试（30 集合计均值）')
        _sz1 = claim('C2.sz1', '58', 'F1.served.zoh[1]', '.1f', 'zoh CartPole 发送尝试')
        _sl0 = claim('C2.sl0', '58', 'F1.served.linear[0]', '.1f', 'linear UAV 发送尝试')
        _sl1 = claim('C2.sl1', '58', 'F1.served.linear[1]', '.1f', 'linear CartPole 发送尝试')
        _tot_z = claim_calc('C2.total_zoh', '58', ['F1.served.zoh[0]', 'F1.served.zoh[1]'], ['+'], '.1f', 'zoh 两用户尝试合计')
        _tot_l = claim_calc('C2.total_lin', '58', ['F1.served.linear[0]', 'F1.served.linear[1]'], ['+'], '.1f', 'linear 两用户尝试合计')
        _r_u = claim_calc('C2.uav_ratio', '58', ['F1.served.linear[0]', 'F1.served.zoh[0]'], ['/'], '.3f', '尝试数比 UAV linear/zoh')
        _r_c = claim_calc('C2.cart_ratio', '58', ['F1.served.linear[1]', 'F1.served.zoh[1]'], ['/'], '.3f', '尝试数比 CartPole linear/zoh')
        A(f"| C2 | 该设置下**汇总**发送尝试数未增加（任务间重分配未隔离；送达数未单独列出） | 两用户合计 "
          f"`{_tot_z}` → `{_tot_l}`（= 30 集 × B = 预算上限，左右恒等）；但**按用户拆开不是常数**："
          f"UAV `{_sz0}`→`{_sl0}`（×`{_r_u}`）、CartPole `{_sz1}`→`{_sl1}`（×`{_r_c}`） | `C2.*` |")
        for rank, (b, mode) in enumerate(((100, "unpaced"), (100, "paced")), start=3):
            k = f"F2.primary/{b}/{mode}"
            A(f"| C{rank} | {'**断服**档上同一预测器反而有害' if mode == 'unpaced' else '`paced` 时序下低预算档仍 <1'} | "
              f"`R_uav` {claim(f'C{rank}.R_uav', '58', f'{k}.R[0]', '.3f', f'B={b}/{mode}/linear/uav/primary')}，区间 "
              f"[{claim(f'C{rank}.ci_lo', '58', f'{k}.ci_uav[0]')}, "
              f"{claim(f'C{rank}.ci_hi', '58', f'{k}.ci_uav[1]')}]；逐 cluster 改善 "
              f"{claim(f'C{rank}.frac', '58', f'{k}.frac_clusters_below_1[0]', '.3f')} | `C{rank}.*` |")
    if "59" in data:
        d59 = data["59"]
        t = d59["F1"]["tau_full"]
        i59 = {(p["budget"], p["mode"]): i for i, p in enumerate(d59["points"])}
        d59b = data.get("59b", {"points": []})
        A(f"| C5 | 危害随**实测**陈旧度上升，高 `x` 端饱和 | `τ_b` "
          f"{claim('C5.tau', '59', 'F1.tau_full.tau_b', '.3f', '12 点 Kendall τ_b')}，相邻逆序 "
          f"{claim('C5.inv', '59', 'F1.tau_full.n_adjacent_inversions', 'd')} 个；交点只给**网格括区间** "
          f"[{cdisp('C5.br_lo')}, {cdisp('C5.br_hi')}]（补点后收窄；括区间不是置信区间） | `C5.*` |")
        k150, k60 = i59[(150, "unpaced")], i59[(60, "unpaced")]
        n2_row = (f"| N2 | 比值**掩盖绝对危害** | `R` 只 "
                  f"{claim('N2.R150', '59', f'points[{k150}].R_uav', '.3f')}→"
                  f"{claim('N2.R60', '59', f'points[{k60}].R_uav', '.3f')}"
                  f"（×{claim_calc('N2.R_ratio', '59', [f'points[{k60}].R_uav', f'points[{k150}].R_uav'], ['/'], '.2f', 'R 比值 B60/B150')}），而 `J_uav` "
                  f"{claim('N2.J150', '59', f'points[{k150}].J[0]', '.3f')} → "
                  f"{claim('N2.J60', '59', f'points[{k60}].J[0]', '.3f')} m"
                  f"（{claim_calc('N2.J_ratio', '59', [f'points[{k60}].J[0]', f'points[{k150}].J[0]'], ['/'], '.1f', 'J 比值 B60/B150')}×） | `N2.*` |")
        i_pmin = min((i for i, p in enumerate(d59["points"]) if p["mode"] == "paced"),
                     key=lambda i: d59["points"][i]["R_uav"])
        k_pmin = "points[" + str(i_pmin) + "].R_uav"
        A(f"| C5b | `paced` 分支在本扫描内全程 <1 | 见 §3a `paced` 各行（最小值 "
          + claim('C5b.min', '59', k_pmin, '.3f', 'paced 分支最小 R_uav') + "） | `C5b.*` |")
    if "60" in data:
        d60 = data["60"]
        A(f"| C6 | **「门槛 = 失效窗口」未获支持** | 交点 `x*` "
          + " / ".join(claim(f"C6.x_{w}", "60", f"brackets.{w}.x_star", ".3f", f"{w} 窗口交点")
                       for w in ("short", "base", "long"))
          + f"，三档网格括区间两两重叠（判词 `{d60['F1p']['verdict']}`） | `C6.*` |")
        A(f"| C7 | **比例偏离（探索性，report-only）** | 预注册即标「只报不判」的派生量：`x*/W` "
          f"{claim('C7.xow_s', '60', 'x_star_over_W.short.x_star_over_W', '.3f')} → "
          f"{claim('C7.xow_m', '60', 'x_star_over_W.base.x_star_over_W', '.3f')} → "
          f"{claim('C7.xow_l', '60', 'x_star_over_W.long.x_star_over_W', '.3f')}；`W` ×"
          f"{claim_calc('C7.W_ratio', '60', ['windows.long.W_expected', 'windows.short.W_expected'], ['/'], '.1f', 'W 长/短')} 而 `x*` 只 ×"
          f"{claim_calc('C7.xs_ratio', '60', ['brackets.long.x_star', 'brackets.short.x_star'], ['/'], '.2f', 'x* 长/短')}"
          " ⇒ **只写「探索性比例偏离、未作确认性判决」**；不能用括区间不重叠代替比例律检验 | `C7.*` |")
    if "58" in data:
        A(f"| N1 | CartPole **不构成收益**（也不构成等价证明） | 区间含 1、逐 cluster 改善 "
          f"{claim('N1.frac_cart', '58', 'F2.primary/200/unpaced.frac_clusters_below_1[1]', '.3f', '90 个初值里 R<1 的比例（CartPole）')}"
          f"（= 36/90）、逐集比值中位 "
          f"{claim('N1.median_cart', '58', 'F2.primary/200/unpaced.cluster_ratio_median[1]', '.3f')} | `N1.*` |")
    if n2_row:
        A(n2_row)
    A("")
    A("★ 术语纪律：`Linear` **无可学参数** ⇒ C1–C4 **不构成「世界模型收益」**，不解锁 P2。")
    A("★ `TrueDyn` 一律写 **「特权诊断参考」**（能读仿真真值），**不是**「诊断上界」——本文没有最优性或上界证明。")
    A("★ C3 的 `t>B` 发帧 = 0 与 C4 的 `t>B` 发帧 ≠ 0 见 §2d（R17 必须同报）。")
    A("★ C2 的边界：**汇总帧数不变 ≠ 没有重分配**。两个用户的 `served` 一降一升，"
      "而且存活时长也在变 ⇒ 想作因果归因（补偿 vs 调度 vs 存活）必须另做固定发送时序的对照，本文未做。")
    A("★ 另注（2026-10-21 更正）：C2 里的 `served` 是**发送尝试**（按 `attempted_idx`），不是送达。"
      "「尝试总数恒为预算上限」只说明预算打满；**本行没有论证送达数不变** —— "
      "送达需另看 `delivery_rate()[\"frames\"]`（= 尝试 − `budget_lost`）。")
    A("")

    # ---------------- 6 入口 ----------------
    A("## 6 复现与图")
    A("")
    A("- 最短复现：`docs/REPRODUCE-P3.md` + `scripts/62_min_repro.py`（4 cell，CPU，只读证据）。")
    A("- 论文图：`scripts/61c_plot_paper_fig.py` → `assets/paper_p3.png`（2×2，数据同入口）。")
    A("- 旧单面板图保留：`assets/59_p3_boundary.png`、`assets/60_window_shift.png`（不删）。")
    A("")

    # ---------------- 7 资源与耗时 ----------------
    A("## 7 资源与耗时（`58`；**Python 层实现开销，不是实时预算结论**）")
    A("")
    if "58" in data:
        d7 = data["58"]
        s7 = meta["58"]["path"]
        tm = d7["TIMING"]
        c7 = d7["cells_J"]
        A("| 预测器 | 每次调用中位耗时(µs) | 调用数 | 每 cell 墙钟(s)（`B=200 unpaced`，n=3 种子） |")
        A("|---|---|---|---|")
        for pred in ("zoh", "linear"):
            med = tm[f"primary/{pred}"]["median_us"]
            ncall = tm[f"primary/{pred}"]["n_calls"]
            flt = {"predictor": pred, "seed_set": "primary", "budget": 200, "mode": "unpaced"}
            em, n_cell = agg_cells(c7, "elapsed_s", flt)
            A(f"| `{pred}` | {fmt_num(med, s7, f'TIMING.primary/{pred}.median_us', '.1f')} | "
              f"{fmt_num(ncall, s7, f'TIMING.primary/{pred}.n_calls', 'd')} | "
              f"{fmt_agg(em, '58', 'elapsed_s', flt, 'mean', '.2f', f'58.elapsed.{pred}.mean')}（n={n_cell}） |")
        A("")
        A("★ 源 JSON 自带的 `predict_timing.note` 原文写明：**「Python 层实现开销，不是实时预算结论」** —— "
          "这里的 µs 只用于比较两种预测器的**相对**开销，不能当成上下行实时性预算。")
        A("★ 每 cell 墙钟时间由环境步进主导（见 `assets/paper_p3_cost.png` 面板 b 的逐 cell 散点），"
          "预测器开销在总量里是小量；**不得**据此声称「预测可以免费实时跑」。")
        A("")
    else:
        A("**58 证据缺失 ⇒ 本节 SKIP**")
        A("")
    # ---------------- 8 否定线（T25–T28） ----------------
    # ★ v0.3：正文 §6.2「否定结果」的引用面。四行 = 四次尝试，各带判词与可点的 JSON 键。
    A("## 8 否定线（T25–T28）：四次尝试、四个判词")
    A("")
    A("正文 §6.2 的引用面。证据走**独立**冻结清单 `docs/evidence_manifest_p3_neg.json`"
      "（`scripts/63_freeze_manifest.py --set neg` 生成），主清单（`…_p3.json`）与主表/图/复现的输入未动。")
    A("")
    A("| # | 问的问题 | 判词 | 关键数 | claim-id | 证据（run-id / 键） |")
    A("|---|---|---|---|---|---|")
    if "66" in data:
        A(f"| T25 | 现有样本量 + 预测误差，够不够做「误差门控」 | `DO-NOT-START` | "
          f"反例 {claim_num('D1', '66', 'I2.n_counterexamples_substantive', 'd', '实质反例数')} 个；"
          f"8 层里 τ<0 的 "
          f"{claim_num('D2', '66', 'I2.n_tau_negative', 'd', '误差-危害 τ 为负的层数')} 层；"
          f"伤害最大格（B=100/unpaced/uav）`J/ZOH` = "
          f"{claim_num('D3', '66', 'I2.harm_without_trigger[0][3].J_rel_vs_zoh', '.3f', '伤害最大格的 J 比')}，"
          f"而该格误差门控触发率 **0** | `D1`–`D3` | `66_identifiability`（`I2.*`） |")
        A(f"| T25b | 样本量能否分辨小效应 | 只能看大效应 | uav −4.3% 那格需 "
          f"{claim_num('D4', '66', 'I1.worst_case.n_needed', 'd', 'worst_case 需要的配对数')} 对，"
          f"实有 {claim_num('D5', '66', 'I1.worst_case.n_have', 'd')} | `D4`–`D5` | `66`（`I1.worst_case`） |")
    if "67" in data:
        A(f"| T26 | `age` 能不能当**阈值**（逐决策反事实） | `NO-ROOM` | "
          f"B=100/uav 上 `τ(age, ΔJ)` = "
          f"{claim_num('D6', '67', 'cells.B100|u0.tau_age', '.3f', 'age 与动作效应的秩相关')}，"
          f"但「换动作会**增加**代价」的决策占 "
          f"{claim_num('D7', '67', 'cells.B100|u0.share_dJ_pos', '.3f', 'ΔJ>0 的决策份额')}"
          f"（探针 {claim_num('D8', '67', 'checks.probe_n', 'd', '逐决策探针数')} 个） | `D6`–`D8` | `67_cf_action`（`cells.B100\\|u0`） |")
    if "68" in data:
        A(f"| T27 | 一个固定 θ 的门控回退能否**两档通吃**（开发批 210–299） | `FALLBACK-FAILS` | "
          f"θ=4：B=100 压到 {claim_num('D9', '68', 'verdicts.F1.R_gated[0]', '.3f', 'B100/gated4 的 R_uav')}、"
          f"但 B=200 从 0.308 变成 {claim_num('D10', '68', 'verdicts.F2.R[0]', '.3f', 'B200/gated4 的 R_uav')}；"
          f"θ=2 在 B=200 反而 {claim_num('D11', '68', 'verdicts.F4.table.B200/θ2.R_uav', '.3f', 'θ 扫描最左点')} | "
          f"`D9`–`D11` | `68_t27_gated_fallback`（`F1/F2/F4`） |")
        A(f"| T27b | 危害**减少了多少** | 减少不归零 | B=100 `ΔR` = "
          f"{claim_num('D12', '68', 'verdicts.F1.delta_R_uav', '.3f', 'B100 上 linear−gated4')}，"
          f"区间 [{claim_num('D13', '68', 'verdicts.F1.delta_ci[0]', '.3f')}, "
          f"{claim_num('D14', '68', 'verdicts.F1.delta_ci[1]', '.3f')}] | `D12`–`D14` | `68`（`F1.delta_*`） |")
    if "69" in data:
        A(f"| T28 | θ=8 在**独立批 310–399** 上是否成立 | `FALLBACK-FAILS (confirm)` | "
          f"同批 `linear` = {claim_num('D15', '69', 'verdicts.CAL.R_uav_linear_batch4', '.3f', '第四批 linear B200')}；"
          f"B=200 `gated8` = {claim_num('D16', '69', 'verdicts.T1.R_uav', '.3f', 'B200/gated8')}"
          f"（区间 [{claim_num('D17', '69', 'verdicts.T1.ci_uav[0]', '.3f')}, "
          f"{claim_num('D18', '69', 'verdicts.T1.ci_uav[1]', '.3f')}]）⇒ 与 `linear` 配对 `ΔR` = "
          f"{claim_num('D19', '69', 'summary.200.per_arm.delta_linear_minus_gated8.point[0]', '.4f', '配对差（同一次重采样内相减）')}"
          f" 区间 [{claim_num('D20', '69', 'summary.200.per_arm.delta_linear_minus_gated8.ci95[0][0]', '.4f')}, "
          f"{claim_num('D21', '69', 'summary.200.per_arm.delta_linear_minus_gated8.ci95[1][0]', '.4f')}] **含 0** | "
          f"`D15`–`D21` | `69_t28_confirm` |")
        A(f"| T28b | 好档「不坏」是怎么来的 | 几乎不起作用 | B=200 门控触发份额 = "
          f"{claim_num('D22', '69', 'summary.200.per_arm.gated8.gate_share_theta', '.4f', '触发份额（比例，非百分数）')}"
          f"（≈0.34%）；B=100 同臂触发份额 = "
          f"{claim_num('D23', '69', 'summary.100.per_arm.gated8.gate_share_theta', '.4f')} | `D22`–`D23` | `69` |")
        A(f"| T28d | 绝对门槛在本批是否批不变 | **不是** | 同批 `linear` 的区间上界 = "
          f"{claim_num('D31', '69', 'summary.200.per_arm.linear.ci95[1][0]', '.3f', '第四批 linear B200 的 95% 区间上界')}"
          f" > 预注册门槛 0.337 | `D31` | `69`（`summary.200.per_arm.linear`） |")
        A(f"| T28c | 毒档是否**归零** | `NOT-CONTAINED` | B=100 `gated8` = "
          f"{claim_num('D24', '69', 'verdicts.T2.R_uav', '.4f', 'B100/gated8')}"
          f"（区间 [{claim_num('D25', '69', 'verdicts.T2.ci_uav[0]', '.4f')}, "
          f"{claim_num('D26', '69', 'verdicts.T2.ci_uav[1]', '.4f')}] **不含 1**）；"
          f"相对 `linear`（{claim_num('D27', '69', 'verdicts.T2.R_uav_linear', '.4f')}）`ΔR` = "
          f"{claim_num('D28', '69', 'summary.100.per_arm.delta_linear_minus_gated8.point[0]', '.4f')}"
          f" 区间 [{claim_num('D29', '69', 'summary.100.per_arm.delta_linear_minus_gated8.ci95[0][0]', '.4f')}, "
          f"{claim_num('D30', '69', 'summary.100.per_arm.delta_linear_minus_gated8.ci95[1][0]', '.4f')}] | "
          f"`D24`–`D30` | `69`（`T2.*`） |")
    A("")
    A("四条**写作纪律**（每条都由上表的数直接决定，不能反着写）：")
    A("")
    A("1. **判词按更强的那句写**：T28 的毒档区间**不含 1** ⇒ 只能写「仍显著差于 `zoh`」，"
      "不得写成「与 `zoh` 等价、无收益」（教训 #73：等价与更差不能并成一格）。")
    A("2. **好档「不坏」不等于门控有效**：触发份额 0.0034 ⇒ 打平的原因是**几乎没起作用**，"
      "不是「补偿被正确抑制」的证明。")
    A("3. **绝对门槛不是批不变量**：T28 的 T1 门槛 `0.337` 取自主集 `linear` 的区间上界，"
      "而同批 `linear` 自己的上界是 0.349 ⇒ 门槛原地不动也可能 FAIL（教训 #72）。"
      "所以 T1 的 FAIL **不得**反读成「好档被弄坏」。")
    A("4. **一次性的 `<1` 不算复现**：T27 的 B=100 单点曾落到 0.988（<1），T28 在独立批上是 1.050"
      " ⇒ 那个点是批噪声，不能引用为「已归零」。")
    A("")
    A("★ 与 §5 的关系：T25 的「误差不能排序危害」与 T23-b 同源；T26 的 `τ(age,ΔJ)` 与 C5 的 `τ_b` 同号"
      "但**层级不同**（C5 是「点与点之间」，T26 是「逐决策」），不得互相代替。")
    A("")

    return "\n".join(L), {"n_trace": len(TRACE)}


def main() -> int:
    global DRAFT_PATH, USE_LATEST
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true", help="只跑检查，不写 docs/")
    ap.add_argument("--draft", default=str(DEFAULT_DRAFT), help="论文正文路径（S11 用）")
    ap.add_argument("--latest", action="store_true",
                    help="退回 latest 取证据（会随新 run 漂移；默认读冻结清单）")
    args = ap.parse_args()
    DRAFT_PATH = args.draft
    USE_LATEST = args.latest

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
        print(f"\n[check-only] 跳过写盘（trace {info['n_trace']} + {len(TRACE_DERIVED)} 派生）")
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
        rows = TRACE + TRACE_DERIVED
        tr_txt = json.dumps(rows, ensure_ascii=False, indent=1)
        old_tr = tr.read_text(encoding="utf-8") if tr.exists() else None
        if old_tr == tr_txt:
            print(f"[unchanged] {tr.relative_to(ROOT).as_posix()}")
        else:
            tr.write_text(tr_txt, encoding="utf-8")
            print(f"[out] {tr.relative_to(ROOT).as_posix()}（{len(rows)} 格："
                  f"direct {len(TRACE)} + derived {len(TRACE_DERIVED)}）")
        cl = ROOT / "docs" / "claims_p3.json"
        cl_txt = json.dumps(CLAIMS, ensure_ascii=False, indent=1)
        old_cl = cl.read_text(encoding="utf-8") if cl.exists() else None
        if old_cl == cl_txt:
            print(f"[unchanged] {cl.relative_to(ROOT).as_posix()}")
        else:
            cl.write_text(cl_txt, encoding="utf-8")
            print(f"[out] {cl.relative_to(ROOT).as_posix()}（{len(CLAIMS)} 条 claim："
                  f"id → 证据源 + JSON 键 + 条件）")

    print(f"\n{'ALL CHECKS PASS' if not FAILS else 'FAILED: ' + '; '.join(FAILS)}")
    if SKIPS:
        print("SKIPPED: " + "; ".join(SKIPS))
    return 0 if not FAILS else 2


if __name__ == "__main__":
    sys.exit(main())
