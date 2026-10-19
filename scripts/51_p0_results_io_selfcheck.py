#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""T20 / P0-3 自检：`results/` 不可覆盖（run-id 分目录 + 拒绝覆盖 + INDEX 单一真源）。

为什么要有这个文件
------------------
P0-3 修的是"证据会不会被同名重跑静默覆盖"。**"我改成写 runs/ 了" 不是证据**：
证据只能是"同一个名字写第二次，程序必须拒绝，且旧文件字节不变"，
以及"一个**故意的错误实现**在这条检查下必须 FAIL"（自证伪 #39：先问什么错误能让它 FAIL）。

七条检查
--------
R1 落点与命名：`write_result` 落到 `results/runs/<run_id>/<name>.json`；默认 run_id 形状
   `<8 位日期>-<6 位时刻>-<7 位 commit>`。
   ★ 错误实现会 FAIL：仍旧写 `results/<name>.json`（固定路径）。
R2 拒绝覆盖（核心反例）：同 run_id 同名再写 ⇒ `SystemExit(3)`，且旧文件 sha256 **不变**；
   同目录下用"裸 write_text"做**阴性对照**，断言那样写 sha256 **确实会变**
   —— 否则"sha 没变"可能只是因为路径写错了、根本没写进去（#46 同型）。
   ★ 错误实现会 FAIL：不检查就写 ⇒ 不抛异常、sha 变。
R3 显式覆盖要留备份：`allow_overwrite=True` ⇒ 新内容生效，且恰好一个 `*.bak.<时间戳>`，
   备份内容 == **旧**内容。
   ★ 错误实现会 FAIL：覆盖不备份（无 .bak）；备份的是新内容（内容比对不过）。
R4 INDEX 语义：临时根里造 1 个 legacy v1 + 同名新 run + 另一个只在 legacy 的逻辑名 ⇒
   新 run = `current`、旧 run = `superseded`、被新 run 取代的 legacy = `superseded`、
   独占的 legacy = `v1(frozen)`。
   ★ 错误实现会 FAIL：INDEX 手写/不重建 ⇒ 行缺失；把第一行当 current ⇒ 状态错。
R5 自检快照的判定守卫：判定不变 ⇒ 不重写（sha 不变）；判定翻转 ⇒ 重写且 `VERDICTS.json`
   记下 `changed_from`。
   ★ 错误实现会 FAIL：无条件写 ⇒ sha 变；翻转不记录 ⇒ 无 `changed_from`。
R6 生产脚本**真的**接了线（静态 + 反例）：42/43/44/45/46/47/48/50 都 import 了 results_io，
   且源码里不再有"固定路径写 results/"。同时用一段**人造违规源码**喂同一个检测函数，
   必须被抓到 —— 证明这条静态检查有分辨力。
   ★ 错误实现会 FAIL：漏改任一脚本 ⇒ 命中；检测正则写废 ⇒ 人造违规抓不到。
R7 `latest_result_path` 优先级：`runs/` 里最新的优先于 v1 冻结文件。
   ★ 错误实现会 FAIL：永远返回 legacy ⇒ `scripts/50` 会一直核对旧文件（#46 事故重演）。

用法
----
    D:/devtools/miniconda3/envs/ai-lab/python.exe scripts/51_p0_results_io_selfcheck.py

所有写操作都在 `tempfile.mkdtemp()` 里做，**绝不碰真 `results/`**（写完删临时目录）。
退出码：0=全过 / 2=有 FAIL。
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from wmlab.eval import results_io as rio  # noqa: E402

RESULTS: list[dict] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append({"name": name, "ok": bool(ok), "detail": detail})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return bool(ok)


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- R1
def r1_layout(tmp: Path) -> dict:
    rid = "20260101-120000-abcdef0"
    p = rio.write_result("demo", {"a": 1}, run_id=rid, root=tmp)
    rel = p.relative_to(tmp).as_posix()
    check("R1a 落到 results/runs/<run_id>/<name>.json",
          rel == f"results/runs/{rid}/demo.json", f"path={rel}")
    check("R1b 没有顺手在 results/ 根目录留一份副本",
          not (tmp / "results" / "demo.json").exists(),
          f"results/ 根目录 json = {[x.name for x in (tmp / 'results').glob('*.json')]}")

    p2 = rio.write_result("demo_default_id", {"a": 2}, root=tmp)
    rid2 = p2.parent.name
    import re
    ok_shape = bool(re.fullmatch(r"\d{8}-\d{6}-[0-9a-f]{7}", rid2))
    check("R1c 默认 run_id 形如 <8位日期>-<6位时刻>-<7位commit>", ok_shape, f"run_id={rid2}")
    check("R1d 内容可回读", json.loads(p.read_text(encoding="utf-8")) == {"a": 1})
    return {"rid": rid, "path": rel}


# --------------------------------------------------------------------------- R2
def r2_refuse_overwrite(tmp: Path, rid: str) -> dict:
    dest = tmp / "results" / "runs" / rid / "demo.json"
    before = _sha(dest)
    code = None
    try:
        rio.write_result("demo", {"a": 999}, run_id=rid, root=tmp)
        code = 0  # 没抛 = 错误实现
    except SystemExit as e:
        code = e.code
    after = _sha(dest)
    check("R2a 同名同 run_id 再写 ⇒ SystemExit(3)", code == rio.RESULT_FROZEN_EXIT,
          f"exit={code}")
    check("R2b 旧文件字节不变", after == before, f"sha {before[:12]} → {after[:12]}")
    check("R2c 旧内容仍是旧值", json.loads(dest.read_text(encoding="utf-8")) == {"a": 1},
          f"content={json.loads(dest.read_text(encoding='utf-8'))}")

    # ---- 阴性对照：证明"sha 不变"不是因为根本没写进去 ----
    ctrl = tmp / "control" / "demo.json"
    ctrl.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(dest, ctrl)
    h0 = _sha(ctrl)
    ctrl.write_text(json.dumps({"a": 999}, ensure_ascii=False, indent=2), encoding="utf-8")
    check("R2d 阴性对照：裸 write_text 确实会改 sha（证明 R2b 有分辨力）",
          _sha(ctrl) != h0, f"对照 sha {h0[:12]} → {_sha(ctrl)[:12]}")
    return {"before": before}


# --------------------------------------------------------------------------- R3
def r3_allow_overwrite(tmp: Path, rid: str) -> dict:
    dest = tmp / "results" / "runs" / rid / "demo.json"
    old_text = dest.read_text(encoding="utf-8")
    rio.write_result("demo", {"a": 7}, run_id=rid, root=tmp, allow_overwrite=True)
    baks = sorted(dest.parent.glob("demo.json.bak.*"))
    check("R3a 显式覆盖后新内容生效",
          json.loads(dest.read_text(encoding="utf-8")) == {"a": 7})
    check("R3b 恰好生成 1 个 .bak", len(baks) == 1, f"baks={[b.name for b in baks]}")
    check("R3c 备份内容 == 旧内容",
          bool(baks) and baks[0].read_text(encoding="utf-8") == old_text,
          f"bak={baks[0].name if baks else None}")
    return {"bak": baks[0].name if baks else None}


# --------------------------------------------------------------------------- R4
def r4_index(tmp: Path) -> dict:
    # legacy：模拟 P0-3 之前写在 results/ 根目录的 v1
    (tmp / "results" / "legacy_only.json").write_text('{"v": 1}', encoding="utf-8")
    (tmp / "results" / "42_t12_four_arms_summary.json").write_text('{"v": 1}', encoding="utf-8")
    rio.write_result("42_t12_four_arms_summary", {"v": 2}, run_id="20260101-000000-aaaaaaa",
                     root=tmp)
    rio.write_result("42_t12_four_arms_summary", {"v": 3}, run_id="20260102-000000-bbbbbbb",
                     root=tmp)
    idx = (tmp / "results" / "INDEX.md").read_text(encoding="utf-8")

    def status_of(path_frag: str) -> str:
        for line in idx.splitlines():
            if path_frag in line and line.startswith("| `"):
                cells = [c.strip() for c in line.split("|")]
                return cells[3]
        return "<缺失>"

    s_new = status_of("runs/20260102-000000-bbbbbbb/42_t12_four_arms_summary.json")
    s_old = status_of("runs/20260101-000000-aaaaaaa/42_t12_four_arms_summary.json")
    s_v1 = status_of("`results/42_t12_four_arms_summary.json`")
    s_leg = status_of("`results/legacy_only.json`")
    check("R4a 最新 run = current", s_new.strip("*") == "current", f"got {s_new}")
    check("R4b 旧 run = superseded", s_old.strip("*") == "superseded", f"got {s_old}")
    check("R4c 被取代的 v1 = superseded", s_v1.strip("*") == "superseded", f"got {s_v1}")
    check("R4d 独占的 v1 = v1(frozen)", s_leg.strip("*") == "v1(frozen)", f"got {s_leg}")
    check("R4e 旧文件没被删（只标 superseded）",
          (tmp / "results" / "42_t12_four_arms_summary.json").exists())
    return {"current": s_new, "v1": s_leg}


# --------------------------------------------------------------------------- R5
def r5_selfcheck_guard(tmp: Path) -> dict:
    p = rio.write_selfcheck("demo_check", {"checks": [{"name": "a", "ok": True}]}, root=tmp)
    h0 = _sha(p)
    p2 = rio.write_selfcheck("demo_check", {"checks": [{"name": "a", "ok": True}]}, root=tmp)
    check("R5a 判定不变 ⇒ 不重写（sha 不变）", _sha(p2) == h0, f"{h0[:12]} → {_sha(p2)[:12]}")
    p3 = rio.write_selfcheck("demo_check", {"checks": [{"name": "a", "ok": False}]}, root=tmp)
    check("R5b 判定翻转 ⇒ 重写", _sha(p3) != h0, f"{h0[:12]} → {_sha(p3)[:12]}")
    hist = json.loads((tmp / "results" / "selfcheck" / "VERDICTS.json").read_text(encoding="utf-8"))
    row = hist.get("demo_check", {})
    check("R5c VERDICTS 记下翻转来源",
          row.get("verdict") == "FAIL" and row.get("changed_from") == "PASS", f"row={row}")
    return {"verdict_after": row.get("verdict")}


# --------------------------------------------------------------------------- R6
def find_fixed_results_writes(src: str) -> list[str]:
    """找"固定路径写 results/"的写法。

    两种形态（都对应 P0-3 之前的真实代码）：
      ① `(ROOT / "results" / "x.json").write_text(...)`
      ② `ROOT / "results" / "x.json"`（把路径取出来再写）
    ★ 已知局限：`dest = ROOT / "results"; (dest / "x.json").write_text(...)` 这种经中间变量的
      写法**抓不到**（R6d 显式断言这个局限，而不是假装检测完备）。
    """
    import re
    hits: list[str] = []
    pat1 = re.compile(r'\(\s*ROOT\s*/\s*"results"\s*/\s*[^)\n]*\)\s*\.write_text')
    pat2 = re.compile(r'ROOT\s*/\s*"results"\s*/\s*"[^"\n]+"')
    for pat in (pat1, pat2):
        for m in pat.finditer(src):
            frag = m.group(0).replace("\n", " ")[:90]
            if frag not in hits:
                hits.append(frag)
    return hits


def r6_wiring() -> dict:
    targets = ["42_t12_hetero_four_arms.py", "43_t9_fix_abc_selfcheck.py",
               "44_t13_lookahead_selfcheck.py", "45_t14_horizon_arms.py",
               "46_t14_failure_timescale.py", "47_t15_normalized_age.py",
               "48_t16_quota_pacing.py", "50_p0_wiring_selfcheck.py"]
    no_import, still_fixed = [], []
    for t in targets:
        src = (ROOT / "scripts" / t).read_text(encoding="utf-8")
        if "results_io" not in src:
            no_import.append(t)
        hits = find_fixed_results_writes(src)
        if hits:
            still_fixed.append(f"{t}:{hits[0]}")
    check("R6a 八个脚本都 import 了 results_io", not no_import, f"缺={no_import}")
    check("R6b 没有脚本再用固定路径写 results/", not still_fixed, f"命中={still_fixed}")

    # 反例：检测函数必须能抓到人造违规（否则 R6b 恒 PASS 且毫无信息）
    fake = '    (ROOT / "results" / "99_fake.json").write_text(json.dumps(x), encoding="utf-8")'
    fake2 = '    dest = ROOT / "results"\n    (dest / "99_fake.json").write_text(t)'
    caught = bool(find_fixed_results_writes(fake))
    missed_var = bool(find_fixed_results_writes(fake2))
    check("R6c 人造违规被 R6b 的检测函数抓到", caught, f"命中={find_fixed_results_writes(fake)}")
    check("R6d 已知局限：经中间变量的写法**抓不到**（记录下来，不假装没漏洞）",
          not missed_var, f"命中={find_fixed_results_writes(fake2)}")
    return {"targets": len(targets)}


# --------------------------------------------------------------------------- R7
def r7_latest(tmp: Path) -> dict:
    p = rio.latest_result_path("42_t12_four_arms_summary", root=tmp)
    got = p.relative_to(tmp).as_posix() if p else "<None>"
    check("R7 runs/ 最新优先于 v1",
          got.endswith("runs/20260102-000000-bbbbbbb/42_t12_four_arms_summary.json"), f"got={got}")
    p2 = rio.latest_result_path("legacy_only", root=tmp)
    check("R7b 只有 legacy 时回退到 v1",
          p2 is not None and p2.name == "legacy_only.json", f"got={p2}")
    check("R7c 不存在的逻辑名 ⇒ None", rio.latest_result_path("nope", root=tmp) is None)
    return {"latest": got}


def main() -> int:
    print(f"[51] results_io 自检｜ROOT={ROOT}")
    tmp = Path(tempfile.mkdtemp(prefix="wmlab_rio_"))
    try:
        d1 = r1_layout(tmp)
        d2 = r2_refuse_overwrite(tmp, d1["rid"])
        d3 = r3_allow_overwrite(tmp, d1["rid"])
        d4 = r4_index(tmp)
        d5 = r5_selfcheck_guard(tmp)
        d6 = r6_wiring()
        d7 = r7_latest(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    ok = sum(1 for r in RESULTS if r["ok"])
    out = {"checks": RESULTS, "failed": [r["name"] for r in RESULTS if not r["ok"]],
           "detail": {"r1": d1, "r3": d3, "r4": d4, "r5": d5, "r6": d6, "r7": d7}}
    # ★ 本脚本只用临时目录写；快照本身走 results/selfcheck/（判定不变则不重写）
    rio.write_selfcheck("51_p0_results_io_selfcheck", out)
    print(f"\n{'ALL CHECKS PASS' if ok == len(RESULTS) else 'FAILED'} ({ok}/{len(RESULTS)})")
    return 0 if ok == len(RESULTS) else 2


if __name__ == "__main__":
    sys.exit(main())
