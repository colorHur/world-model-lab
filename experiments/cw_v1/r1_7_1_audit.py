# -*- coding: utf-8 -*-
"""CW-R1.7.1 落盘入口：`plan_dp` 的**轴错配**复审 + 参考物更正。

本轮不出面板、不接 `plan_ref`（预注册 §3 明写：接 `plan_ref` 另立一轮）。
`kind` 由闸链决定：
- `AXIS-SWAP-CONFIRMED`（A1–A6 全过）⇒ 根因坐实、修复版在同一套判据下可采纳；
- `AXIS-SWAP-UNCONFIRMED`（任一 FAIL）⇒ 只交现状，不写"修好了"。
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import config as C
from .dp_audit import run_all
from .plan_dp import cache_stats

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results" / "cw_v1"

#: 验收链（预注册 §2 的顺序：先证参考物、再证修复、再证闸有分辨力、最后才证轴可分辨）
CHAIN = (
    ("A0", "信息边界（AST）"),
    ("A1", "参考物序关系 exact ≤ brute ≤ round_robin"),
    ("A2", "修复版可采纳（对角 5 + 非对角 6，≤1e-3）"),
    ("A3", "阴性对照（活体 legacy_swap）非对角 ≥5 例超差且最大 ≥10%"),
    ("A4", "加密两档：修复版 ≤tol、legacy ≥100×tol"),
    ("A5", "两条 P 轴可分辨（V(p0,p1) ≠ V(p1,p0)）"),
    ("A6", "变异 2/2 被抓"),
)


def _src_hash() -> str:
    import hashlib
    h = hashlib.sha256()
    for p in sorted((REPO / "experiments" / "cw_v1").glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def _git(args: list[str]) -> str:
    import subprocess
    return subprocess.run(["D:\\devtools\\Git\\cmd\\git.exe", *args], cwd=str(REPO),
                          capture_output=True, text=True).stdout.strip()


def _settle(g, n_mut_missed: int) -> dict:
    key = {r["name"].split(" ")[0]: r for r in g.rows}
    chain, stopped = [], None
    for name, desc in CHAIN:
        ok = bool(key.get(name, {}).get("pass", False))
        if stopped is None:
            state = "pass" if ok else "fail"
            if not ok:
                stopped = name
        else:
            state = "not_reached"
        chain.append({"gate": name, "desc": desc, "state": state, "observed": ok})
    ok_all = stopped is None
    return {
        "chain": chain,
        "stopped_at": stopped,
        "all_pass": ok_all,
        "kind": "AXIS-SWAP-CONFIRMED" if ok_all else "AXIS-SWAP-UNCONFIRMED",
        "fixed_impl_admissible_under_this_prereg": ok_all,
        "legacy_impl_admissible": False,          # R1.7 判词，本轮不撤销
        "r17_verdict_preserved": "DP-TAIL-REJECTED",
        "plan_ref_wired": False,
        "panel_run": False,
        "eligible_for_N1N2": False,
        "n_gate_fail": g.summary()["n_fail"],
        "n_checks": g.summary()["n_checks"],
        "n_mutation_missed": n_mut_missed,
        "note": ("本轮只做「根因 + 参考物更正」；不接 plan_ref、不跑面板、不出 N1/N2。"
                 "`R1.7` 的 `DP-TAIL-REJECTED` 指那一版实现，本轮不撤销它。"),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="CW-R1.7.1 轴错配复审")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--note", default="")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    g, extra = run_all()
    verdict = _settle(g, extra.get("n_mut_missed", 99))
    elapsed = round(time.perf_counter() - t0, 2)

    print(f"[R1.7.1] dp_audit {g.summary()['n_checks']} checks / {g.summary()['n_fail']} fail")
    for r in g.rows:
        print(("  PASS " if r["pass"] else "  FAIL ") + r["name"])
    print(f"[R1.7.1] 判词 {verdict['kind']}；链停止于 {verdict['stopped_at']}")

    if not args.no_write:
        now = datetime.now(timezone(timedelta(hours=8)))
        cfg = C.CWConfig(plant=C.bounded_plant())
        run_id = (f"cw_r1_7_1_axis_{now.strftime('%Y%m%dT%H%M%S')}_"
                  f"{cfg.fingerprint()}_{_src_hash()}")
        out = RESULTS / run_id
        out.mkdir(parents=True, exist_ok=True)

        def _dump(name, obj):
            (out / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2),
                                    encoding="utf-8")

        _dump("audit_gates.json", g.summary())
        _dump("verdict.json", verdict)
        _dump("details.json", extra)
        _dump("meta.json", {
            "run_id": run_id, "kind": verdict["kind"],
            "protocol": C.CWConfig.PROTOCOL_VERSION,
            "config": cfg.to_dict(),
            "code": ["experiments/cw_v1/plan_dp.py", "experiments/cw_v1/dp_audit.py",
                     "experiments/cw_v1/r1_7_1_audit.py"],
            "prereg": "docs/research/CW-R1.7.1-轴错配-预注册-v0.1.md",
            "does_not_touch": ["wmlab/", "plan_ref 的默认行为", "results/cw_v1/cw_r1_7_dp_*（旧 3 份）"],
            "note": args.note,
            "at_asia_shanghai": now.isoformat(),
            "elapsed_s": elapsed,
            "git_head": _git(["rev-parse", "HEAD"]),
            "git_status_porcelain": _git(["status", "--porcelain"]),
            "src_hash": _src_hash(),
            "dp_cache": cache_stats(),
        })
        print(f"[R1.7.1] 证据落盘 results/cw_v1/{run_id}")
    print(f"[R1.7.1] 总耗时 {elapsed} s")
    return 0 if g.summary()["n_fail"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
