"""证据冻结：把"引用哪份入库 JSON"从 `latest` 自动漂移改成**清单钉死**。

★ 为什么需要它（2026-10-20 复验 E 项）
------------------------------------
`latest_result_path()` 取 `results/runs/` 里字典序最大的那份。它能保证"读到的是最新的"，
但**不能保证"下次读到的还是同一份"**：明天再跑一个 run，主表/复现的输入就静默换了，
而生成出来的文档看起来毫无变化（脚本会打印来源，但那只在运行当时的终端里）。
复验报告的原话是：*"生成后记录来源是有用的，但未来新增 run 会改变输入，不等于事先固定证据。"*

所以本模块提供第二层：`docs/evidence_manifest_p3.json`。
- 生成/复现脚本**默认只读清单**，清单里写死 `run_id` + 路径 + 内容指纹。
- 清单**不会**自动更新：`scripts/63_freeze_manifest.py --allow-update` 才写，且先把旧版
  另存为 `docs/evidence_manifest_p3.v<N>.json`（"更新清单必须另留版本"）。
- 想临时指向别的证据 ⇒ 显式 `--ref` / `--latest`（命令行上看得见），不靠隐式漂移。

★ 两种指纹，别混用（同一条复验 E 项）
------------------------------------
本机 `Path.write_text()` 在 Windows 上把 `\\n` 写成 `\\r\\n`，而 `core.autocrlf=input` 提交时
又把 CRLF 规范化成 LF ⇒ **同一个 JSON 的"工作树字节 sha256"与"git blob sha256"必然不同**。
- `sha256_bytes`：工作树原始字节。用来发现"这份文件被改过"。
- `sha256_norm`：把 `\\r\\n` 归一成 `\\n` 之后的字节。**跨检出设置稳定**，是权威指纹。
清单两个都存；校验优先用 `sha256_norm`，`sha256_bytes` 不符只记为提示（换行差异不算篡改）。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

__all__ = [
    "ROOT",
    "MANIFEST_PATH",
    "EvidenceError",
    "sha256_bytes",
    "sha256_norm",
    "load_manifest",
    "resolve",
    "verify_entry",
]

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "docs" / "evidence_manifest_p3.json"


class EvidenceError(RuntimeError):
    """证据缺失 / 与清单不符 / 清单本身不可用。"""


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_norm(b: bytes) -> str:
    """行尾归一后的内容指纹（`\\r\\n` → `\\n`）。跨平台/跨检出设置稳定。"""
    return hashlib.sha256(b.replace(b"\r\n", b"\n")).hexdigest()


def load_manifest(path: str | Path | None = None) -> dict:
    p = Path(path or MANIFEST_PATH)
    if not p.exists():
        raise EvidenceError(f"冻结清单不存在：{p.as_posix()}")
    try:
        m = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise EvidenceError(f"冻结清单不是合法 JSON：{p.as_posix()}（{exc}）") from exc
    if not isinstance(m.get("entries"), dict) or not m["entries"]:
        raise EvidenceError(f"冻结清单没有 entries：{p.as_posix()}")
    return m


def verify_entry(entry: dict, raw: bytes) -> tuple[bool, str]:
    """清单条目 vs 实际字节。返回 (是否一致, 说明)。

    判据：`sha256_norm` 必须一致（权威）；`sha256_bytes` 不一致但 norm 一致 ⇒ 只提示换行差异。
    """
    got_norm = sha256_norm(raw)
    got_bytes = sha256_bytes(raw)
    want_norm = entry.get("sha256_norm")
    want_bytes = entry.get("sha256_bytes")
    if want_norm is None:
        return False, "清单条目缺少 sha256_norm（旧格式）"
    if got_norm != want_norm:
        return False, (f"内容指纹不符：清单 {want_norm[:16]} vs 实际 {got_norm[:16]}"
                       f"（若确为正当换证据，请用 scripts/63 --allow-update 另起版本）")
    if want_bytes is not None and got_bytes != want_bytes:
        return True, f"仅行尾/字节差异（norm 一致；bytes {want_bytes[:12]}≠{got_bytes[:12]}），不算篡改"
    return True, ""


def resolve(name: str, *, key: str | None = None, manifest: dict | None = None,
            path_override: str | Path | None = None, root: Path | None = None) -> dict:
    """**唯一**的证据解析入口。

    Parameters
    ----------
    name
        逻辑名（`58_p3_confirmation`）。
    key
        清单里的键（`"58"`）。给定时要求清单条目的 `name` 与之一致，避免键名串台。
    path_override
        显式覆盖路径（`--ref`）。**不再验证清单指纹**，但会记录 `pinned=False`，
        供调用方在输出里标"这是一次性覆盖，不是清单证据"。
    """
    root = Path(root or ROOT)
    if path_override is not None:
        p = Path(path_override)
        if not p.is_absolute():
            p = root / p
        if not p.exists():
            raise EvidenceError(f"--ref 指定的文件不存在：{p.as_posix()}")
        raw = p.read_bytes()
        return {"path": p, "raw": raw, "sha256_bytes": sha256_bytes(raw),
                "sha256_norm": sha256_norm(raw), "run_id": p.parent.name,
                "pinned": False, "entry": None}

    m = manifest or load_manifest()
    entries = m["entries"]
    if key is not None:
        if key not in entries:
            raise EvidenceError(f"清单里没有键 {key}；现有键：{sorted(entries)}")
        e = entries[key]
        if e.get("name") != name:
            raise EvidenceError(f"清单键 {key} 指向 {e.get('name')}，与请求的 {name} 不符")
    else:
        hits = [(k, v) for k, v in entries.items() if v.get("name") == name]
        if not hits:
            raise EvidenceError(f"清单里没有 {name}；现有：{sorted(e['name'] for e in entries.values())}")
        if len(hits) > 1:
            raise EvidenceError(f"清单里 {name} 有多个键 {[k for k, _ in hits]}，请用 key= 指定")
        k, e = hits[0]

    p = root / e["path"]
    if not p.exists():
        raise EvidenceError(f"清单钉死的证据文件不存在：{p.as_posix()}（run_id={e.get('run_id')}）")
    raw = p.read_bytes()
    ok, detail = verify_entry(e, raw)
    if not ok:
        raise EvidenceError(f"{p.as_posix()}：{detail}")
    return {"path": p, "raw": raw, "sha256_bytes": sha256_bytes(raw),
            "sha256_norm": sha256_norm(raw), "run_id": e.get("run_id"),
            "pinned": True, "entry": e, "note": detail}
