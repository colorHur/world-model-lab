# -*- coding: utf-8 -*-
"""CW-R1.7 落盘入口：小状态 DP 的**可采纳性闸**（本轮不出面板，理由见下）。

为什么**不跑面板**
----------------
预注册 `docs/research/CW-R1.7-小状态DP-预注册-v0.1.md` §4 否定条件：
`D3`/`D4` 任一 FAIL 且不是"两条链都不自洽"的边界情形 ⇒ **停，只交"DP 未成立"，不出面板数字**。
本轮 `D3`（vs 穷举最优）与 `D4`（vs 轮询策略）都 FAIL，且 `D5` 证明加密网格**不**收敛
⇒ DP 既可能高于真最优、也可能**低于**它（插值在折点抹平后可以"作弊"）⇒
把它当尾部值函数会把偏差直接灌进动作选择。**因此不接 `plan_ref`、不跑面板、不出 `N1`/`N2`。**

`kind=DP-TAIL-REJECTED`：本轮产出是"计划中的主件被它自己的可采纳性闸判掉"，不是收益证据。
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import config as C
from .dp_gates import d6_mutations, run_all
from .plan_dp import cache_stats

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results" / "cw_v1"

#: 验收链（预注册 §3 的顺序：先算术、再可采纳性、最后才谈面板）
CHAIN = (
    ("D0", "信息边界（AST，两个文件）"),
    ("D1a", "b=0 与解析外推一致（≤1e-4）"),
    ("D1b", "b=0 的斜率与解析一致（≤1e-12）"),
    ("D2", "对预算单调不增"),
    ("D3", "可采纳性 vs 穷举最优（τ≤4）"),
    ("D4", "可采纳性 vs 轮询策略（b=τ）"),
    ("D5", "加密必须缩小超差（否则不是离散化）"),
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


def _settle(g, mut_rows, mut_missed) -> dict:
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
    return {"chain": chain, "stopped_at": stopped, "all_pass": stopped is None,
            "usable_as_tail": stopped is None,
            "eligible_for_N1N2": False if stopped else None,
            "n_gate_fail": g.summary()["n_fail"],
            "n_checks": g.summary()["n_checks"],
            "n_mutation_missed": mut_missed, "mutation_rows": mut_rows,
            "note": ("链在第一个 fail 停住；`not_reached` 一律不写成通过。"
                     "`usable_as_tail=False` ⇒ 本轮不接 `plan_ref`、不跑面板、不出 N1/N2。")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="CW-R1.7 小状态 DP：可采纳性闸")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--note", default="")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    g, extra = run_all()
    mut_rows, mut_missed = d6_mutations()
    verdict = _settle(g, mut_rows, mut_missed)
    elapsed = round(time.perf_counter() - t0, 2)

    print(f"[R1.7] dp_gates {g.summary()['n_checks']} checks / {g.summary()['n_fail']} fail")
    for r in g.rows:
        print(("  PASS " if r["pass"] else "  FAIL ") + r["name"])
    print(f"[R1.7] 变异 {len(mut_rows)} 次 / {mut_missed} 次未被抓")
    print(f"[R1.7] 链停止于 {verdict['stopped_at']}；usable_as_tail={verdict['usable_as_tail']}")

    if not args.no_write:
        now = datetime.now(timezone(timedelta(hours=8)))
        cfg = C.CWConfig(plant=C.bounded_plant())
        run_id = (f"cw_r1_7_dp_{now.strftime('%Y%m%dT%H%M%S')}_"
                  f"{cfg.fingerprint()}_{_src_hash()}")
        out = RESULTS / run_id
        out.mkdir(parents=True, exist_ok=True)

        def _dump(name, obj):
            (out / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2),
                                    encoding="utf-8")

        _dump("dp_gates.json", g.summary())
        _dump("mutation_gates.json", {"n_mutations": len(mut_rows),
                                      "n_missed": mut_missed, "rows": mut_rows})
        _dump("admissibility.json", extra)
        _dump("verdict.json", verdict)
        _dump("meta.json", {
            "run_id": run_id, "kind": "DP-TAIL-REJECTED",
            "protocol": C.CWConfig.PROTOCOL_VERSION,
            "event_pairing": cfg.protocol.event_pairing,
            "config": cfg.to_dict(),
            "panel_run": False,
            "panel_skip_reason": ("D3/D4 未过 ⇒ DP 不可采纳；把它当尾部会把偏差灌进动作选择"
                                  "（预注册 §4 否定条件 1）"),
            "code": ["experiments/cw_v1/plan_dp.py", "experiments/cw_v1/dp_gates.py",
                     "experiments/cw_v1/r1_7_plan.py"],
            "prereg": "docs/research/CW-R1.7-小状态DP-预注册-v0.1.md",
            "note": args.note,
            "at_asia_shanghai": now.isoformat(),
            "elapsed_s": elapsed,
            "git_head": _git(["rev-parse", "HEAD"]),
            "git_status_porcelain": _git(["status", "--porcelain"]),
            "src_hash": _src_hash(),
            "dp_cache": cache_stats(),
            "does_not_touch": ["wmlab/", "results/（旧）", "plan_ref 的默认行为"],
        })
        print(f"[R1.7] 证据落盘 results/cw_v1/{run_id}")
    print(f"[R1.7] 总耗时 {elapsed} s")
    return 0 if g.summary()["n_fail"] == 0 and mut_missed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
