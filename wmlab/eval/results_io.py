"""入库结果的**唯一**写入口：run-id 分目录 + 拒绝覆盖 + INDEX 单一真源。

★ 为什么必须收敛到一个入口
--------------------------
2026-10-19 验收（P0-3）点名的缺陷：`scripts/42/43/44/45/46/47/48/50` 各自
`(ROOT / "results" / "<固定名>.json").write_text(...)`。同名重跑 = **静默覆盖**，
覆盖后**没有任何痕迹**能说明"这个数字是哪一版代码、哪一次配置跑出来的"。
T16 已经踩过一次更轻的版本（`scripts/47 --skip-experiment` 把 24 个 cell 写成空表，
靠 `git checkout` 救回，自证伪 #45）——那次还能救，是因为 git 里有旧版；
一旦覆盖未提交的产物，就**不可恢复**。

⇒ 本模块是唯一实现，脚本**不得**自己拼 `results/` 路径。规则四条：

1. **新结果写 `results/runs/<run_id>/<name>.json`**，
   `run_id = <YYYYmmdd-HHMMSS>-<commit 前 7 位>`。同一轮多个脚本可共享 run_id
   （设环境变量 `WMLAB_RUN_ID`），便于"一轮 = 一个目录"。
2. **拒绝覆盖**：目标已存在 ⇒ 打印原因并 `SystemExit(3)`。
   只有显式 `allow_overwrite=True`（脚本的 `--allow-overwrite`）才写，
   且写前把旧文件复制成 `<name>.json.bak.<时间戳>`。**任何分支都不删除历史。**
3. **`results/INDEX.md` 由 `rebuild_index()` 生成**（不是手写）：同一逻辑名一行一版本，
   最新 = `current`，其余 = `superseded`；`results/` 根目录已有的 v1 文件**原地保留**，
   登记为 `v1(frozen)`（若已有更新版本则降为 `superseded`）。
   INDEX 里**不写"本次生成时间"**，只写每个文件的 mtime ⇒ 文件集合不变时 INDEX 逐字节不变，
   不会让每次回归都产生一个"只改了时间戳"的脏提交。
4. **自检快照**走另一条路径 `results/selfcheck/<name>.json`：它们不是"主张的证据"，
   而是"某次检查的断言"，且每轮回归都会重跑 ⇒ **只在判定发生变化时**才更新快照，
   判定变化记进 `results/selfcheck/VERDICTS.json`。
   这样既不会静默抹掉 "PASS → FAIL"，也不会让每次回归产生噪声提交。

★ 设计取舍（为什么自检不也进 runs/）
------------------------------------
若自检也进 `runs/<ts>-<commit>/`，则同一 commit 每回归一次就多一个目录（本机每 2 h 一轮），
很快会把 `results/` 变成"目录垃圾场"，反而让真正的证据（42/45/47/48）淹没。
自检的语义是**断言**不是**测量**：同一个 commit 上重复跑出同一个 PASS，信息量为 0；
有信息量的是"判定何时翻转"，所以用 VERDICTS.json 记住翻转点。
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path

__all__ = [
    "ROOT",
    "RESULT_FROZEN_EXIT",
    "ResultFrozen",
    "current_run_id",
    "write_result",
    "write_selfcheck",
    "rebuild_index",
    "derive_verdict",
    "sha256_text",
    "latest_result_path",
]

#: 本仓库根（`wmlab/eval/results_io.py` 往上三层）
ROOT = Path(__file__).resolve().parents[2]

#: 拒绝覆盖已冻结结果时的退出码（与"研究 Gate 未过 = 2"区分开）
RESULT_FROZEN_EXIT = 3

_TZ = timezone(timedelta(hours=8))


class ResultFrozen(RuntimeError):
    """目标已存在且未显式允许覆盖。"""


# --------------------------------------------------------------------------- 工具
def _now() -> datetime:
    return datetime.now(_TZ)


def _stamp(fmt: str = "%Y%m%d-%H%M%S") -> str:
    return _now().strftime(fmt)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def short_commit(cwd: Path | None = None) -> str:
    """当前 HEAD 前 7 位；不在 git 仓库里时返回 `nogit`（不抛，避免测试环境失败）。"""
    try:
        p = subprocess.run(["git", "rev-parse", "--short=7", "HEAD"],
                           cwd=str(cwd or ROOT), capture_output=True, text=True, timeout=20)
        out = (p.stdout or "").strip()
        if p.returncode == 0 and out:
            return out.splitlines()[0]
    except Exception:  # noqa: BLE001 —— 缺 git 不应让实验失败
        pass
    return "nogit"


def current_run_id() -> str:
    """`<时间戳>-<commit>`；可用 `WMLAB_RUN_ID` 让同一轮的多份产物共用一个目录。"""
    env = (os.environ.get("WMLAB_RUN_ID") or "").strip()
    if env:
        return env
    return f"{_stamp()}-{short_commit()}"


def derive_verdict(payload) -> str | None:
    """从产物里抽出**可比较的判定**（PASS / FAIL），抽不出就返回 None。

    顺序有意为之：先看显式判定字段，再看"失败清单"，最后才按 checks 列表全否。
    有 checks 但元素没有判定字段 ⇒ 视为 FAIL（保守：宁可报错也不默认通过）。
    """
    if not isinstance(payload, dict):
        return None
    v = payload.get("verdict")
    if isinstance(v, bool):
        return "PASS" if v else "FAIL"
    if isinstance(v, str) and v:
        return v
    for key in ("pass", "ok"):
        v = payload.get(key)
        if isinstance(v, bool):
            return "PASS" if v else "FAIL"
    for key in ("fails", "failed", "checks_failed", "failures"):
        v = payload.get(key)
        if isinstance(v, (list, tuple)):
            return "PASS" if len(v) == 0 else "FAIL"
    checks = payload.get("checks")
    if isinstance(checks, (list, tuple)) and checks:
        return "PASS" if all(_one_check_ok(c) for c in checks) else "FAIL"
    return None


def _one_check_ok(c) -> bool:
    if isinstance(c, bool):
        return c
    if isinstance(c, dict):
        for key in ("ok", "pass", "passed"):
            if key in c:
                return bool(c[key])
        return False  # 有 check 却没判定字段 ⇒ 保守判 FAIL
    return bool(c)


# --------------------------------------------------------------------------- 证据
def write_result(name: str, payload, *, run_id: str | None = None,
                 allow_overwrite: bool = False, root: Path | None = None) -> Path:
    """把**证据类**产物写进 `results/runs/<run_id>/<name>.json`，并重建 INDEX。

    Raises
    ------
    SystemExit(3)
        目标已存在且 `allow_overwrite=False`。**这是设计行为，不是 bug**：
        已冻结的证据只能另起一个 run，不能被同名覆盖。
    """
    root = Path(root or ROOT)
    res = root / "results"
    rid = run_id or current_run_id()
    dest_dir = res / "runs" / rid
    dest = dest_dir / f"{name}.json"
    text = json.dumps(payload, ensure_ascii=False, indent=2)

    if dest.exists():
        if not allow_overwrite:
            # 打印原因 + 非 0 退出码（3）。用"先 print 再 raise int"而不是 raise 字符串，
            # 否则 `SystemExit.code` 会是一段文字，调用方无法按码分支。
            print(f"[FROZEN] {dest} 已存在，拒绝覆盖（run_id={rid}）。\n"
                  f"         换一个 --run-id / WMLAB_RUN_ID 重跑；确有正当理由要覆盖时，\n"
                  f"         显式加 --allow-overwrite（会先备份 *.bak.<时间戳>）。")
            raise SystemExit(RESULT_FROZEN_EXIT)
        bak = dest.with_suffix(dest.suffix + f".bak.{_stamp()}")
        shutil.copy2(dest, bak)
        print(f"[OVERWRITE] 旧证据已备份 → {bak.name}")

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8")
    rebuild_index(root=root)
    return dest


# --------------------------------------------------------------------------- 自检
def write_selfcheck(name: str, payload, *, root: Path | None = None,
                    force: bool = False) -> Path:
    """把**自检快照**写进 `results/selfcheck/<name>.json`，判定不变则不重写。

    Returns
    -------
    Path
        快照路径（无论本次是否真的写了）。

    Notes
    -----
    - `force=True` 时无条件写（给 `scripts/51` 做反例用）。
    - 判定翻转（PASS ↔ FAIL）或首次出现 ⇒ 写快照，并在 `VERDICTS.json` 里追加/更新
      `since` 与 `changed_from`；判定不变 ⇒ 完全不碰磁盘（避免回归噪声）。
    """
    root = Path(root or ROOT)
    d = root / "results" / "selfcheck"
    dest = d / f"{name}.json"
    verdict = derive_verdict(payload)
    text = json.dumps(payload, ensure_ascii=False, indent=2)

    prev = None
    if dest.exists():
        try:
            prev = json.loads(dest.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 —— 旧文件坏了就当作没有，不会阻止写入
            prev = None
    prev_verdict = derive_verdict(prev) if prev is not None else None

    if dest.exists() and not force and prev_verdict == verdict:
        return dest

    d.mkdir(parents=True, exist_ok=True)
    if dest.exists() and force:
        bak = dest.with_suffix(dest.suffix + f".bak.{_stamp()}")
        shutil.copy2(dest, bak)
        print(f"[OVERWRITE] 旧快照已备份 → {bak.name}")
    dest.write_text(text, encoding="utf-8")

    # 判定历史：只在"出现过"或"翻转"时改这一行，保证文件内容稳定
    hist = d / "VERDICTS.json"
    table = {}
    if hist.exists():
        try:
            table = json.loads(hist.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            table = {}
    row = table.get(name) or {}
    if row.get("verdict") != verdict:
        table[name] = {"verdict": verdict,
                       "changed_from": row.get("verdict"),
                       "since": _now().strftime("%Y-%m-%d %H:%M:%S"),
                       "run_id": current_run_id()}
        hist.write_text(json.dumps(dict(sorted(table.items())), ensure_ascii=False, indent=2),
                        encoding="utf-8")
    rebuild_index(root=root)  # 快照变了，INDEX 里的 sha 也要跟着变
    return dest


# --------------------------------------------------------------------------- 索引
def latest_result_path(name: str, root: Path | None = None) -> Path | None:
    """某逻辑名的**当前版本**路径（`runs/` 里最新的优先，退回到 v1 冻结文件）。

    ★ 为什么需要它：P0-3 之后 42/45/47/48 的**新**数字都写进 `runs/`，
    但 `results/` 根目录的 v1 文件还在。任何"读 42 结果做核对"的自检如果继续硬编码
    `results/42_....json`，就会永远核对**旧的冻结版**、并因为旧文件一直存在而恒 PASS
    （这正是自证伪 #46「验收点指错对象」的同型事故）。
    """
    root = Path(root or ROOT)
    res = root / "results"
    runs = res / "runs"
    if runs.is_dir():
        cands = sorted(p for p in runs.glob(f"*/{name}.json"))
        if cands:
            return cands[-1]  # run_id 前缀是时间戳 ⇒ 字典序 = 时间序
    legacy = res / f"{name}.json"
    return legacy if legacy.exists() else None


_LEGEND = (
    "| 状态 | 含义 |\n"
    "|---|---|\n"
    "| `current` | 该逻辑名的**最新**版本（引用数字请指这里） |\n"
    "| `superseded` | 旧版本，**保留不删**，仅为可追溯 |\n"
    "| `v1(frozen)` | P0-3 之前写在 `results/` 根目录的 v1 产物，**原地冻结**，未经 run-id 复跑 |\n"
)


def _collect(root: Path) -> list[dict]:
    res = root / "results"
    rows: list[dict] = []
    for p in sorted(res.glob("*.json")):
        st = p.stat()
        rows.append({"name": p.stem, "version": "v1", "sort": "0000-00-00 00:00:00+v1",
                     "path": p.relative_to(root).as_posix(), "kind": "legacy",
                     "mtime": st.st_mtime, "size": st.st_size, "sha": _sha256_file(p)})
    runs = res / "runs"
    if runs.is_dir():
        for run_dir in sorted(x for x in runs.iterdir() if x.is_dir()):
            for p in sorted(run_dir.glob("*.json")):
                st = p.stat()
                # run_id 以 <YYYYmmdd-HHMMSS>-<commit> 开头 ⇒ 字典序 = 时间序
                rows.append({"name": p.stem, "version": run_dir.name, "sort": run_dir.name,
                             "path": p.relative_to(root).as_posix(), "kind": "run",
                             "mtime": st.st_mtime, "size": st.st_size, "sha": _sha256_file(p)})
    return rows


def rebuild_index(root: Path | None = None) -> Path:
    """重扫 `results/` 并重写 `results/INDEX.md`（内容仅取决于文件集合与文件本身）。"""
    root = Path(root or ROOT)
    res = root / "results"
    res.mkdir(parents=True, exist_ok=True)
    rows = _collect(root)

    by_name: dict[str, list[dict]] = {}
    for r in rows:
        by_name.setdefault(r["name"], []).append(r)
    for items in by_name.values():
        items.sort(key=lambda x: x["sort"])
        for i, it in enumerate(items):
            newest = i == len(items) - 1
            if it["kind"] == "legacy":
                it["status"] = "superseded" if not newest else "v1(frozen)"
            else:
                it["status"] = "current" if newest else "superseded"

    lines = [
        "# results 索引（自动生成，勿手改）",
        "",
        "由 `wmlab/eval/results_io.py::rebuild_index()` 生成。手改会被下一次写入覆盖。",
        "",
        _LEGEND,
        "## 证据（`results/runs/<run_id>/` 与冻结的 v1）",
        "",
        "| 逻辑名 | 版本 | 状态 | 文件时间 | 大小 | sha256 | 路径 |",
        "|---|---|---|---|---|---|---|",
    ]
    for name in sorted(by_name):
        for it in by_name[name]:
            t = datetime.fromtimestamp(it["mtime"], _TZ).strftime("%Y-%m-%d %H:%M")
            lines.append(f"| `{name}` | {it['version']} | **{it['status']}** | {t} | "
                         f"{it['size']} B | `{it['sha'][:12]}` | `{it['path']}` |")

    sc = res / "selfcheck"
    if sc.is_dir():
        lines += ["", "## 自检快照（`results/selfcheck/`，判定不变则不重写）", "",
                  "| 快照 | 文件时间 | 大小 | sha256 |", "|---|---|---|---|"]
        for p in sorted(sc.glob("*.json")):
            if p.stem == "VERDICTS":  # 判定历史的元数据，不是快照本身
                continue
            st = p.stat()
            t = datetime.fromtimestamp(st.st_mtime, _TZ).strftime("%Y-%m-%d %H:%M")
            lines.append(f"| `{p.stem}` | {t} | {st.st_size} B | "
                         f"`{_sha256_file(p)[:12]}` |")

    dest = res / "INDEX.md"
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dest
