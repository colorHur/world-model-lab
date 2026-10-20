#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P3 · 闸门自检：**用复验报告列出的错误反例**逐条打一遍（不是只看正常路径）。

为什么单独一支脚本
------------------
正常路径通过只能说明"没坏"，说明不了"能分辨"。2026-10-20 复验抓到两处正是这个问题：
- `62 --seeds 999` 两格 SKIP 却打印 `BITWISE MATCH`、退出码 0；
- 把摘要的 `0.315` 换成另一个**真实存在**的 `1.223`，S11 全部 PASS。

所以本脚本的判据不是"跑通"，而是"**每个反例都必须以指定的方式失败**"：
期望成功的一行要 exit 0；期望失败的一行要给出对应的非 0 码，且**输出里不能出现** `BITWISE MATCH`。

纪律
----
- 所有注入都在 `tempfile` 的**隔离副本**上做：不碰 `results/`、不碰 `docs/PAPER-P3-v0.2.md`。
- 不改原始证据；证据删除类反例用"删副本里的一格"实现。
- 每一项都打印"期望 / 实际"，失败项直接让整支脚本 exit 2。

用法
----
    python scripts/64_gate_selftest.py            # 全量（含 4 次真实 4-cell 复现，约 25 s）
    python scripts/64_gate_selftest.py --fast     # 跳过耗时的 62 复现用例，只验 61 的数字闸
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from wmlab.eval.evidence import resolve  # noqa: E402

PAPER = ROOT / "docs" / "PAPER-P3-v0.2.md"
PY = sys.executable


def run(args: list[str], timeout: int = 300) -> tuple[int, str]:
    p = subprocess.run([PY] + args, cwd=str(ROOT), capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=timeout)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


class Case:
    """一条反例：期望退出码 + 输出里**不该**出现的字样 + 输出里**应该**出现的字样。"""

    def __init__(self, name: str, expect_code: int, want: list[str] | None = None,
                 forbid: list[str] | None = None):
        self.name, self.expect_code = name, expect_code
        self.want = want or []
        self.forbid = forbid or []

    def judge(self, code: int, out: str) -> tuple[bool, str]:
        if code != self.expect_code:
            return False, f"退出码 {code} ≠ 期望 {self.expect_code}"
        miss = [w for w in self.want if w not in out]
        if miss:
            return False, f"输出缺少 {miss}"
        hit = [f for f in self.forbid if f in out]
        if hit:
            return False, f"输出出现了不该有的 {hit}"
        return True, "OK"


def inject_remove_cell(src: Path, dst: Path) -> str:
    """删掉 copy 里的一个 primary 格（seed=130/linear）—— 模拟"证据少一格"。"""
    d = json.loads(src.read_text(encoding="utf-8"))
    keep = [c for c in d["cells_J"]
            if not (c.get("seed_set") == "primary" and c.get("budget") == 200
                    and c.get("mode") == "unpaced" and c.get("predictor") == "linear"
                    and c.get("seed") == 130)]
    removed = len(d["cells_J"]) - len(keep)
    d["cells_J"] = keep
    dst.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    return f"删掉 {removed} 个 cell"


def inject_nan(src: Path, dst: Path) -> str:
    """把 copy 里某个 cell 的一个 episode 代价改成 NaN —— 模拟"证据里出现非有限值"。"""
    d = json.loads(src.read_text(encoding="utf-8"))
    n = 0
    for c in d["cells_J"]:
        if (c.get("seed_set") == "primary" and c.get("budget") == 200
                and c.get("mode") == "unpaced" and c.get("predictor") == "zoh"
                and c.get("seed") == 100):
            c["J_per_episode"][3][0] = float("nan")
            n += 1
    dst.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    return f"注入 {n} 处 NaN"


def inject_wrong_condition(paper: Path, dst: Path) -> str:
    """把摘要里第一个 `0.315` 换成**另一个真实结果** `1.223`（复验 B 项的原反例）。"""
    text = paper.read_text(encoding="utf-8")
    old = "`0.315`<!--C1.R_uav-->"
    if old not in text:
        raise SystemExit("论文里找不到 `0.315`<!--C1.R_uav-->，无法注入错条件反例")
    new = "`1.223`<!--C1.R_uav-->"
    dst.write_text(text.replace(old, new, 1), encoding="utf-8")
    return "0.315→1.223（真实数字，错条件）"


def inject_last_digit(paper: Path, dst: Path) -> str:
    text = paper.read_text(encoding="utf-8")
    dst.write_text(text.replace("`0.315`<!--C1.R_uav-->", "`0.311`<!--C1.R_uav-->", 1),
                   encoding="utf-8")
    return "0.315→0.311（改末位）"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="跳过 62 的真实复现用例")
    args = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="p3-gate-"))
    ev = resolve("58_p3_confirmation", key="58")
    ref = ev["path"]
    print(f"[env] 隔离目录 {tmp.as_posix()}")
    print(f"[env] 冻结证据 {ref.relative_to(ROOT).as_posix()}（run_id={ev['run_id']}）")

    cases: list[tuple[Case, int, str]] = []

    def add(c: Case, args_list: list[str]) -> None:
        code, out = run(args_list)
        ok, why = c.judge(code, out)
        cases.append((c, code, why if not ok else "OK"))
        print(f"  [{'PASS' if ok else 'FAIL'}] {c.name}: 退出码 {code}；{why}")

    if not args.fast:
        # ---------------- 62 复现闸 ----------------
        add(Case("62 正常四格（清单证据）", 0, want=["BITWISE MATCH", "compared=4"]), 
            ["scripts/62_min_repro.py"])
        add(Case("62 未知种子 seed=999（复验原反例）", 4, want=["NOT VERIFIED"],
                 forbid=["BITWISE MATCH"]),
            ["scripts/62_min_repro.py", "--seeds", "999"])
        add(Case("62 --quick 只能叫流程冒烟", 5, want=["SMOKE ONLY"], forbid=["BITWISE MATCH"]),
            ["scripts/62_min_repro.py", "--quick"])
        add(Case("62 参考文件不存在", 4, want=["NOT VERIFIED"], forbid=["BITWISE MATCH"]),
            ["scripts/62_min_repro.py", "--ref", str(tmp / "nope.json")])

        cut = tmp / "58_cut.json"
        inject_remove_cell(ref, cut)
        add(Case("62 证据少一格（跳过比对）", 4, want=["NOT VERIFIED"], forbid=["BITWISE MATCH"]),
            ["scripts/62_min_repro.py", "--ref", str(cut)])

        nanf = tmp / "58_nan.json"
        inject_nan(ref, nanf)
        add(Case("62 证据含 NaN（非有限值）", 4, want=["NOT VERIFIED"], forbid=["BITWISE MATCH"]),
            ["scripts/62_min_repro.py", "--ref", str(nanf)])

    # ---------------- 61 数字闸 ----------------
    add(Case("61 正文正常（claim 绑定全部命中）", 0, want=["ALL CHECKS PASS"]),
        ["scripts/61_build_tables.py", "--check-only"])

    p1 = tmp / "paper_wrong_cond.md"
    note1 = inject_wrong_condition(PAPER, p1)
    print(f"  [inject] 错条件：{note1}")
    add(Case("61 真实数字放到错误条件（复验 B 项原反例）", 2,
             want=["S11c"], forbid=["ALL CHECKS PASS"]),
        ["scripts/61_build_tables.py", "--check-only", "--draft", str(p1)])

    p2 = tmp / "paper_last_digit.md"
    note2 = inject_last_digit(PAPER, p2)
    print(f"  [inject] 改末位：{note2}")
    add(Case("61 改一位小数", 2, want=["S11c"], forbid=["ALL CHECKS PASS"]),
        ["scripts/61_build_tables.py", "--check-only", "--draft", str(p2)])

    # ---------------- 汇总 ----------------
    bad = [c for c, _code, why in cases if why != "OK"]
    print("\n| # | 反例 | 期望退出码 | 实际 | 判定 |")
    print("|---|---|---|---|---|")
    for i, (c, code, why) in enumerate(cases, 1):
        print(f"| {i} | {c.name} | {c.expect_code} | {code} | {'PASS' if why == 'OK' else 'FAIL：' + why} |")

    shutil.rmtree(tmp, ignore_errors=True)
    if bad:
        print(f"\nGATE SELFTEST FAILED：{len(bad)} 条反例没有按预期失败/通过")
        return 2
    print(f"\nGATE SELFTEST PASS：{len(cases)} 条反例全部按预期（失败的都是该失败的）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
