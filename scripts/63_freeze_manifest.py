#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P3 · 冻结证据清单：`docs/evidence_manifest_p3.json`（生成/复现默认只读它）。

为什么
------
`latest_result_path()` = "现在最新的一份"，**不等于**"这次引用、下次还引用的同一份"。
主表（61）、论文图（61c/61d）、最短复现（62）如果各自取 latest，任何一天新跑一个 run，
输入就静默换了，而文档看起来没变。复验报告 E 项要的是**事先固定**，不是事后记录。

本脚本做三件事
--------------
1. 把当前 latest 的 `run_id` / 路径 / 内容指纹（`sha256_norm`，行尾归一）/
   工作树字节指纹（`sha256_bytes`）/ cells 数 / `checks_failed` 写进清单；
2. 顺带抄下**协议**（budget、mode、seed_set、seeds、episodes、n_boot），
   这样读清单的人不必再猜"0.315 是哪一档";
3. 已存在时**拒绝直接覆盖**：必须 `--allow-update`，且旧版先另存为
   `docs/evidence_manifest_p3.v<N>.json`（"更新清单必须另留版本"）。

用法
----
    python scripts/63_freeze_manifest.py                 # 首次冻结 / 只读校验（dry）
    python scripts/63_freeze_manifest.py --allow-update   # 换证据（会先归档旧版）

退出码：0 = 清单与当前 latest 一致或已写入；2 = 有键缺失。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from wmlab.eval.evidence import (  # noqa: E402
    MANIFEST_PATH, load_manifest, resolve, sha256_bytes, sha256_norm,
)
from wmlab.eval.results_io import latest_result_path  # noqa: E402

SRC = {"58": "58_p3_confirmation", "59": "59_p3_boundary",
       "59b": "59b_p3_boundary_refine", "60": "60_p3_window_shift"}

PROTOCOL_KEYS = ("budget", "mode", "seed_set", "seeds", "episodes", "n_boot", "arm", "absorbed")


def entry_from(path: Path, name: str) -> dict:
    raw = path.read_bytes()
    d = json.loads(raw.decode("utf-8"))
    cfg = d.get("config") or {}
    proto = {k: cfg[k] for k in ("N", "window", "episodes", "n_boot", "arm", "absorbed",
                                 "budgets", "modes", "seed_sets") if k in cfg}
    return {
        "name": name,
        "run_id": path.parent.name,
        "path": path.relative_to(ROOT).as_posix(),
        "sha256_norm": sha256_norm(raw),
        "sha256_bytes": sha256_bytes(raw),
        "cells": len(d.get("cells_J", []) or []),
        "checks_failed": len(d.get("checks_failed", []) or []),
        "protocol": proto,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-update", action="store_true", help="允许换证据（旧版另存为 .v<N>.json）")
    ap.add_argument("--check", action="store_true", help="只校验清单与磁盘一致，不写")
    args = ap.parse_args()

    cur: dict = {"manifest_version": 1,
                 "purpose": "P3 生成/复现的固定输入（不随 latest 漂移）",
                 "generated_by": "scripts/63_freeze_manifest.py",
                 "note": ("生成/复现脚本默认只读本清单。换证据必须 --allow-update，"
                          "旧版另存 evidence_manifest_p3.v<N>.json；不得手改。"),
                 "hash_scheme": ("sha256_norm = 行尾归一(\\r\\n→\\n)后的 sha256，权威；"
                                 "sha256_bytes = 工作树原始字节 sha256，仅记录（与 git blob 因换行不同）"),
                 "entries": {}}

    missing: list[str] = []
    for k, name in SRC.items():
        p = latest_result_path(name)
        if p is None or not p.exists():
            missing.append(f"{k}:{name}")
            print(f"[MISS] {k} {name} 在 results/ 下找不到")
            continue
        cur["entries"][k] = entry_from(p, name)
        print(f"[ok] {k} {name} ← {cur['entries'][k]['path']}")

    if missing:
        print("\n[FAIL] 缺证据：" + ", ".join(missing))
        return 2

    if MANIFEST_PATH.exists():
        old = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        same = all(old["entries"].get(k, {}).get("sha256_norm") == v["sha256_norm"]
                   for k, v in cur["entries"].items())
        if args.check:
            print(f"\n[check] 清单 {'与当前 latest 一致' if same else '与当前 latest 不一致'}")
            return 0 if same else 2
        if not same and not args.allow_update:
            print("\n[FROZEN] 清单已存在且与当前 latest 不同；换证据请加 --allow-update"
                  "（旧版会另存为 evidence_manifest_p3.v<N>.json）")
            return 2
        if same:
            print("\n[unchanged] 清单与当前 latest 一致，不重写")
            return 0
        ver = int(old.get("manifest_version", 1))
        bak = MANIFEST_PATH.with_name(f"evidence_manifest_p3.v{ver}.json")
        bak.write_text(json.dumps(old, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        cur["manifest_version"] = ver + 1
        cur["supersedes"] = bak.name
        print(f"[history] 旧版另存 → {bak.relative_to(ROOT).as_posix()}")

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(cur, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"[out] {MANIFEST_PATH.relative_to(ROOT).as_posix()}（v{cur['manifest_version']}，"
          f"{len(cur['entries'])} 条）")

    # 写完立刻用统一入口回读一遍：清单自己也要能被校验（不然只是又一份手写表）
    for k, name in SRC.items():
        r = resolve(name, key=k)
        print(f"[verify] {k} pinned={r['pinned']} run_id={r['run_id']} {r['note'] or 'OK'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
