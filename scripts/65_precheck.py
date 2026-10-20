#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""投稿前自查闸（v3.1 §2 第二轮）：把"自查清单"里能用机器判的那几条变成可失败检查。

为什么另起一个脚本而不塞进 `scripts/61`
--------------------------------------
`61` 的职责是**从证据 JSON 生成主表**（S10/S10c/S12 都在核"表对不对"）。
本轮要核的是**文稿自身的投稿前条件**：文献来源是否有一手页面、摘要里有没有"没绑定条件"的数字、
禁用断言有没有漏进来、图资产是否真存在。两类判据的生命周期不同（证据变 vs 文稿变），
混在一起会让"表没问题"掩盖"文有问题"。

五组检查（每条都能 FAIL，且各有阴性对照）
----------------------------------------
- `R1–R3` 文献：正文每个 `arXiv:` 编号都要在 `docs/refs_p3.json` 有条目；
  条目的题名必须在正文逐字出现（空白归一后比对）；`volume_issue_pages` 写「待核」的，
  正文对应位置 ±400 字窗内必须也出现「待核」。
- `A1–A2` 摘要绑定：摘要里**每个** inline 结果数后面必须紧跟 `<!--claim-id-->`，
  绑定率必须 100%（不是"数字在 trace 里出现过"，那是 `61` 的 S11a 弱闸已经管的事）；
  结论节（§7）同规则，单列一条，便于定位。
- `L1–L2` 禁用断言：硬禁词（已投稿/已接收/已联系/外部复现通过/外部验收）出现即 FAIL；
  软禁词（安全域/等价证明/诊断上界）出现时同行必须带否定标记（不/非/无/未/不得/不算/不是）。
- `F1` 图资产：正文「图」节里每个 `assets/...` 路径必须真实存在。
- `N1–N3` 阴性对照：注入假编号、删条目、删标记、插禁用词 —— 对应检查必须报错，
  否则说明该检查没有分辨力（自证伪 #39）。

用法
----
    D:/devtools/miniconda3/envs/ai-lab/python.exe scripts/65_precheck.py
    ... --paper <path> --refs <path>

退出码：0 = 全部通过；2 = 有检查 FAIL。**只读**：不写 `results/`、不改任何文稿。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PAPER = ROOT / "docs" / "PAPER-P3-v0.3.md"
DEFAULT_REFS = ROOT / "docs" / "refs_p3.json"
DEFAULT_CLAIMS = ROOT / "docs" / "claims_p3.json"

RESULTS: list[tuple[str, str, str]] = []   # (tag, verdict, detail)


def check(tag: str, ok: bool, detail: str) -> None:
    RESULTS.append((tag, "PASS" if ok else "FAIL", detail))


def skip(tag: str, detail: str) -> None:
    RESULTS.append((tag, "SKIP", detail))


# ---------------------------------------------------------------- 文本工具
ARXIV = re.compile(r"arXiv:\s*(\d{4}\.\d{4,5})")
INLINE = re.compile(r"`([^`\n]+)`")
# 「数字 + 可选的单位词（写在 code span 外）+ 紧跟的 claim 标记」都算已绑定：
# 中文行文里常写「`3` 个<!--C5.inv-->」，单位在 span 外是正常写法，不该被当成未绑定。
MARK = re.compile(
    r"`[^`\n]+`[ \t]*(?:步|倍|个|格|次|页|集|条|µs|us|ms|s|m|dB|%)?"
    r"((?:[ \t]*<!--[A-Za-z][\w.\-]*-->)+)"
)
CLAIM_ID = re.compile(r"<!--([A-Za-z][\w.\-]*)-->")

# 允许作为结果数的"单位后缀"：数字后面可以直接跟这些词，仍算一个结果数
UNITS = r"(?:µs|us|ms|s|m|dB|%|步|倍|个|格|次|页|集|条)"
NUM = re.compile(r"^[+\-−]?\d+(?:[.,]\d+)?(?:\s*[×x]\s*\d+(?:\.\d+)?)?" + UNITS + r"?$")


def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def section(text: str, header: str) -> str:
    """取 `## header` 到下一个同级 `## ` 之间的正文（找不到 ⇒ 空串）。"""
    m = re.search(rf"^##\s*{re.escape(header)}\s*$", text, re.M)
    if not m:
        return ""
    rest = text[m.end():]
    nxt = re.search(r"^##\s", rest, re.M)
    return rest[: nxt.start()] if nxt else rest


def is_number_span(content: str) -> bool:
    """inline code 内容是否"是一个结果数"（纯数字 + 单位 + 分隔符）。"""
    c = content.strip()
    if not c or not re.search(r"\d", c):
        return False
    if re.search(r"[A-Za-z_]", c.replace("µs", "us")):
        # 允许 1e-9 这种科学计数
        if not re.fullmatch(r"[+\-]?\d+(?:\.\d+)?[eE][+\-]?\d+", c):
            return False
    return bool(NUM.match(c))


def numbers_without_claim(seg: str) -> list[str]:
    """返回 seg 里**没有紧跟 claim 标记**的结果数（含原文位置）。"""
    bad: list[str] = []
    # 先把"数字 + 紧跟标记"的位置标出来。
    # ★ 这里踩过一次：`MARK` 正则本身以反引号开头，所以 `m.start()` 就是该 inline code 的**起点**；
    #   第一版又往回找了两次反引号，把"已绑定"误判为"未绑定"（5/10 假 FAIL）——先怀疑判据（#39）。
    covered = {m.start() for m in MARK.finditer(seg)}
    for m in INLINE.finditer(seg):
        if is_number_span(m.group(1)) and m.start() not in covered:
            bad.append(m.group(1))
    return bad


def count_claim_bound(seg: str) -> int:
    n = 0
    for m in INLINE.finditer(seg):
        if is_number_span(m.group(1)):
            tail = seg[m.end(): m.end() + 80]
            if re.match(r"^\s*(?:<!--[A-Za-z][\w.\-]*-->\s*)+", tail):
                n += 1
    return n


def count_numbers(seg: str) -> int:
    return sum(1 for m in INLINE.finditer(seg) if is_number_span(m.group(1)))


# ---------------------------------------------------------------- R 组：文献
def run_refs(paper: str, refs: dict) -> None:
    entries = refs.get("entries", [])
    by_id = {e["arxiv_id"]: e for e in entries}
    paper_ids = sorted(set(ARXIV.findall(paper)))

    missing = [i for i in paper_ids if i not in by_id]
    extra = [i for i in by_id if i not in paper_ids]
    check("R1 正文 arXiv 编号 ↔ refs_p3.json 条目双向一致",
          (not missing) and (not extra) and len(entries) >= 3,
          f"正文 {len(paper_ids)} 个编号 / 清单 {len(entries)} 条"
          + (f"；清单缺：{missing}" if missing else "")
          + (f"；清单多余（未被引）：{extra}" if extra else "")
          + ("" if len(entries) >= 3 else "；★ 条目 <3 ⇒ 非退化守卫不满足"))

    np_ = norm_ws(paper)
    bad_title, bad_meta = [], []
    for e in entries:
        t = norm_ws(e.get("title", ""))
        if not t or t not in np_:
            bad_title.append(e["arxiv_id"])
        for f in ("source_url", "submitted", "venue_status", "volume_issue_pages", "mechanism_relation"):
            if not e.get(f):
                bad_meta.append(f"{e['arxiv_id']}:{f}")
        if e.get("venue_status") not in refs.get("status_vocabulary", {}):
            bad_meta.append(f"{e['arxiv_id']}:venue_status 不在词表")
    check("R2 每条清单的题名在正文逐字出现，且关键字段非空", (not bad_title) and (not bad_meta),
          f"题名未命中 {len(bad_title)}；字段缺失/越界 {len(bad_meta)}"
          + (f"；{bad_title}" if bad_title else "")
          + (f"；{bad_meta}" if bad_meta else ""))

    # 待核传播：卷期写「待核」的条目，正文该条附近必须也写「待核」
    bad_prop = []
    for e in entries:
        vip = e.get("volume_issue_pages", "")
        if "待核" not in vip:
            continue
        t = norm_ws(e.get("title", ""))
        pos = np_.find(t)
        if pos < 0:
            bad_prop.append(e["arxiv_id"])
            continue
        window = np_[max(0, pos - 400): pos + 400]
        if "待核" not in window:
            bad_prop.append(e["arxiv_id"])
    check("R3 清单标『待核』的卷期，正文对应位置也必须标『待核』", not bad_prop,
          f"{sum(1 for e in entries if '待核' in e.get('volume_issue_pages', ''))} 条待核；未传播 {len(bad_prop)}"
          + (f"；{bad_prop}" if bad_prop else ""))


# ---------------------------------------------------------------- A 组：摘要/结论绑定
def run_binding(paper: str, claims: list[dict]) -> None:
    ids = {c["id"] for c in claims}
    abstract = section(paper, "摘要")
    concl = section(paper, "7. 结论") or section(paper, "7 结论")

    if not abstract:
        skip("A1 摘要绑定率", "找不到 `## 摘要` 段（对象不存在 ⇒ SKIP，不判 PASS/FAIL）")
    else:
        n, bad = count_numbers(abstract), numbers_without_claim(abstract)
        check("A1 摘要里每个结果数都绑定 claim-id（100%）", (not bad) and n >= 6,
              f"结果数 {n} 个，未绑定 {len(bad)}"
              + (f"：{bad[:6]}" if bad else "")
              + ("" if n >= 6 else "；★ 抽到的数 <6 ⇒ 非退化守卫不满足"))

    if not concl:
        skip("A2 结论节绑定率", "找不到 `## 7. 结论` 段")
    else:
        n2, bad2 = count_numbers(concl), numbers_without_claim(concl)
        check("A2 结论节里每个结果数都绑定 claim-id", (not bad2) and n2 >= 4,
              f"结果数 {n2} 个，未绑定 {len(bad2)}"
              + (f"：{bad2[:6]}" if bad2 else "")
              + ("" if n2 >= 4 else "；★ 抽到的数 <4 ⇒ 非退化守卫不满足"))

    # 标记引用的 claim-id 必须都存在（否则是"绑了个不存在的条件"）
    all_ids = set(CLAIM_ID.findall(paper))
    unknown = sorted(i for i in all_ids if i not in ids)
    check("A3 正文里引用的 claim-id 都存在于 claims_p3.json", not unknown,
          f"引用 {len(all_ids)} 个不同 claim-id，未知 {len(unknown)}"
          + (f"：{unknown[:6]}" if unknown else ""))


# ---------------------------------------------------------------- L 组：禁用断言
HARD_BAN = ["已投稿", "已接收", "已被接收", "已联系", "外部复现通过", "外部验收通过"]
SOFT_BAN = ["安全域", "等价证明", "诊断上界"]
NEG_MARK = ["不", "非", "无", "未", "不得", "不算", "不是", "没有"]


def run_wording(paper: str) -> None:
    hits = []
    for w in HARD_BAN:
        for m in re.finditer(re.escape(w), paper):
            line = paper[paper.rfind("\n", 0, m.start()) + 1: paper.find("\n", m.start())]
            if not any(n in line for n in NEG_MARK):
                hits.append(f"{w}@{line.strip()[:40]}")
    check("L1 硬禁断言（已投稿/已接收/已联系/外部验收）不得以肯定语气出现", not hits,
          f"命中 {len(hits)}" + (f"：{hits[:4]}" if hits else ""))

    soft = []
    for w in SOFT_BAN:
        for m in re.finditer(re.escape(w), paper):
            line = paper[paper.rfind("\n", 0, m.start()) + 1: paper.find("\n", m.start())]
            if not any(n in line for n in NEG_MARK):
                soft.append(f"{w}@{line.strip()[:40]}")
    check("L2 软禁断言（安全域/等价证明/诊断上界）必须处于否定语境", not soft,
          f"命中 {len(soft)}" + (f"：{soft[:4]}" if soft else ""))


# ---------------------------------------------------------------- F 组：图资产
def run_assets(paper: str) -> None:
    paths = sorted(set(re.findall(r"`(assets/[^`\s]+\.(?:png|jpg|svg|html))`", paper)))
    missing = [p for p in paths if not (ROOT / p).exists()]
    check("F1 正文引用的图/资产文件都存在", (not missing) and len(paths) >= 2,
          f"引用 {len(paths)} 个资产，缺失 {len(missing)}"
          + (f"：{missing}" if missing else "")
          + ("" if len(paths) >= 2 else "；★ 引用 <2 ⇒ 非退化守卫不满足"))


# ---------------------------------------------------------------- N 组：阴性对照
def run_negative(paper: str, refs: dict) -> None:
    # N1：注入一个不存在的编号 ⇒ R1 必须报"清单缺"
    fake = paper.replace("arXiv:2002.07987", "arXiv:2002.07987", 1)
    fake = fake + "\n\n（对照注入）见 arXiv:9912.99999 的相关讨论。\n"
    ids = set(ARXIV.findall(fake))
    by_id = {e["arxiv_id"] for e in refs.get("entries", [])}
    check("N1 阴性对照：注入不存在的 arXiv 编号必须被 R1 抓到",
          any(i not in by_id for i in ids), f"注入后未在清单的编号 {sorted(i for i in ids if i not in by_id)}")

    # N2：删掉一个条目的题名 ⇒ R2 必须报"题名未命中"
    ents = [dict(e) for e in refs.get("entries", [])]
    if ents:
        ents[0]["title"] = ents[0]["title"] + "（被改动）"
        np_ = norm_ws(paper)
        miss = [e["arxiv_id"] for e in ents if norm_ws(e["title"]) not in np_]
        check("N2 阴性对照：清单题名被改动后 R2 必须抓到", len(miss) >= 1, f"未命中 {miss}")
    else:
        skip("N2 阴性对照", "清单为空")

    # N3：把摘要里第一个 claim 标记删掉 ⇒ A1 必须报"未绑定"
    abstract = section(paper, "摘要")
    if abstract:
        mutated = abstract.replace("<!--C1.R_uav-->", "", 1)
        bad = numbers_without_claim(mutated)
        check("N3 阴性对照：删掉摘要里一个 claim 标记必须被 A1 抓到", len(bad) >= 1,
              f"变异后未绑定 {bad[:3]}")
    else:
        skip("N3 阴性对照", "无摘要段")

    # N4：插一个硬禁词 ⇒ L1 必须抓到
    mutated = paper + "\n（对照注入）本工作已投稿至某刊。\n"
    hits = [w for w in HARD_BAN if re.search(re.escape(w), mutated)]
    check("N4 阴性对照：插入硬禁词必须被 L1 抓到", bool(hits), f"命中 {hits}")


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", default=str(DEFAULT_PAPER))
    ap.add_argument("--refs", default=str(DEFAULT_REFS))
    ap.add_argument("--claims", default=str(DEFAULT_CLAIMS))
    a = ap.parse_args()

    paper_p, refs_p, claims_p = Path(a.paper), Path(a.refs), Path(a.claims)
    for p in (paper_p, refs_p, claims_p):
        if not p.exists():
            print(f"[FATAL] 缺文件：{p.as_posix()}")
            return 2

    paper = paper_p.read_text(encoding="utf-8")
    refs = json.loads(refs_p.read_text(encoding="utf-8"))
    claims = json.loads(claims_p.read_text(encoding="utf-8"))

    print(f"[precheck] 正文 = {paper_p.relative_to(ROOT).as_posix()}  清单 = {refs_p.name}  "
          f"claim = {len(claims)} 条  正文字符 = {len(paper)}")
    run_refs(paper, refs)
    run_binding(paper, claims)
    run_wording(paper)
    run_assets(paper)
    run_negative(paper, refs)

    w = max(len(t) for t, _, _ in RESULTS)
    for tag, verdict, detail in RESULTS:
        print(f"[{verdict:4s}] {tag.ljust(w)}  {detail}")
    nfail = sum(1 for _, v, _ in RESULTS if v == "FAIL")
    nskip = sum(1 for _, v, _ in RESULTS if v == "SKIP")
    npass = len(RESULTS) - nfail - nskip
    print(f"\n[summary] PASS {npass} / FAIL {nfail} / SKIP {nskip}")
    if nfail:
        print("[verdict] 投稿前自查未通过：先看 FAIL 的对象是否真的存在（R14 自检失败先怀疑判据）")
        return 2
    print("[verdict] 投稿前自查（机器可判部分）通过；人判部分见 05-论文与产出 的自查文档")
    return 0


if __name__ == "__main__":
    sys.exit(main())
