# -*- coding: utf-8 -*-
"""CW-R1.7.2 落盘入口：`DPTail` 接进 `plan_ref` 的接线闸 + 价格对账（**新证据目录**）。

判据与否定条件见 `docs/research/CW-R1.7.2-DP接plan_ref-预注册-v0.1.md`。

本轮**跑什么 / 不跑什么**
------------------------
跑：`W0/W0b/W0c/W1/W1b/W4` 六道接线闸（本机 ~2 s）＋ `W2` 价格对账 ＋ `W3` p̂ 冻结代价。
不跑：`G4/M1/G8a` 与开发面板 —— 因为 `W4`（可采纳性）在链的第 0 步就 FAIL：
      DP 尾部把"多一帧"的价值高估到闭式精确值的 **1.32×**（τ=16），
      在这种值函数上跑收益表，等于用**已知偏乐观的价格**做决策，数字无法解释。

判词（预注册 §4 + §7 更正 3）
-----------------------------
`DP-WIRING-REJECTED`（`W0/W0b/W1/W1b` 任一 FAIL）
`DP-TAIL-INADMISSIBLE`（`W4` FAIL：尾部不是可采纳的代价估计）
`DP-WIRED-PRICE-*`（只有前两组全过才可能）
"""

from __future__ import annotations

import argparse
import re
import json
import time
from datetime import datetime, timedelta, timezone

import numpy as np

from . import config as C
from . import plan_ref as Pmodule
from .dp_wire_gates import (grid_refinement_scan, phat_freeze_cost, price_recon,
                            run_wire_gates)
from .plan_dp import cache_stats
from .r1_6_plan import RESULTS, _git, _src_hash

#: 验收链（顺序不可跳；`panel` 表示该步之后的收益表是否允许跑）
CHAIN = (
    ("W0", "接线一致性（不对称 p̂/P/bud）"),
    ("W0b", "预算管线 + 未送用户推进一槽"),
    ("W0c", "legacy 阴性对照（必须 FAIL）"),
    ("W1", "= G8d[dp]：预算维严格递减"),
    ("W1b", "W1 的活体阴性对照（必须 FAIL 才算有分辨力）"),
    ("W4b", "证书自检：闭式 vs 逐步模拟 ≤1e-12"),
    ("W4", "可采纳性：b=1 边际收益 ≤ 闭式精确值"),
)


def _settle(g, price: dict) -> dict:
    def _find(code: str):
        # ★ 按"代码后不是单词字符"匹配：`W1` 不能吃掉 `W1b`，`W0` 不能吃掉 `W0b`。
        pat = re.compile(rf"^{code}(?!\w)")
        return next((r for r in g.rows if pat.match(r["name"])), None)

    rows = {k: _find(k) for k, _ in CHAIN}
    got = {k: bool(rows.get(k, {}).get("pass", False)) for k, _ in CHAIN}
    # `W0c`/`W1b` 是**阴性对照**：它们的 PASS 判据就是"被对照项 FAIL"
    chain, stopped = [], None
    for name, desc in CHAIN:
        if stopped is None:
            state = "pass" if got[name] else "fail"
            if not got[name]:
                stopped = name
        else:
            state = "not_reached"
        chain.append({"gate": name, "desc": desc, "state": state, "observed": got[name]})

    if not all(got[k] for k in ("W0", "W0b", "W0c", "W1", "W1b")):
        verdict = "DP-WIRING-REJECTED"
    elif not got["W4"]:
        verdict = "DP-TAIL-INADMISSIBLE"
    else:
        r = price["bounded_plant"]["ratio_analytic_over_dp"]
        verdict = ("DP-WIRED-PRICE-OVERSTATED" if r >= 2.0
                   else "DP-WIRED-PRICE-STRUCTURAL" if r >= 0.5
                   else "DP-WIRED-PRICE-UNDERSTATED")
    return {"chain": chain, "stopped_at": stopped, "verdict": verdict,
            "panel_run": False,
            "panel_reason": ("链在 W4 停住：DP 尾部不满足可采纳性（比闭式精确值乐观 ≥8%），"
                             "在其上跑收益表无法归因") if verdict == "DP-TAIL-INADMISSIBLE" else "",
            "eligible_for_N1N2": False,
            "n_gate_fail": g.summary()["n_fail"]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="CW-R1.7.2 DP 接线闸 + 价格对账")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--note", default="")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    prev = Pmodule.TAIL_DEFAULT
    Pmodule.TAIL_DEFAULT = "dp"
    try:
        g = run_wire_gates()
        price = price_recon()
        w3 = phat_freeze_cost()
        grid = grid_refinement_scan()
    finally:
        Pmodule.TAIL_DEFAULT = prev

    print(f"[CW-R1.7.2] wire gates: {g.summary()['n_checks']} checks, "
          f"{g.summary()['n_fail']} fail；TAIL_DEFAULT 已恢复={Pmodule.TAIL_DEFAULT}")
    for r in g.rows:
        print(("  PASS " if r["pass"] else "  FAIL "), r["name"][:60])
    verdict = _settle(g, price)
    print(f"[CW-R1.7.2] 判词 = {verdict['verdict']}；链停在 {verdict['stopped_at']}；"
          f"panel_run={verdict['panel_run']}")
    bp = price["bounded_plant"]
    print(f"[CW-R1.7.2] 价格对账(bounded)：解析 {bp['marginal_value_analytic']:.6f} vs "
          f"DP {bp['marginal_value_dp']:.6f}，ratio={bp['ratio_analytic_over_dp']:.3f}")
    print(f"[CW-R1.7.2] W3 p̂ 冻结最大相对差 = {w3['max_abs_rel_vs_prior']:.3f}")
    for r in grid["rows"]:
        print(f"[CW-R1.7.2] 网格 {r['grid']:20s} n={r['n']:3d} " +
              " ".join(f"tau{k}:{v:.3f}" for k, v in r["over"].items()))
    elapsed = round(time.perf_counter() - t0, 2)

    if not args.no_write:
        now = datetime.now(timezone(timedelta(hours=8)))
        cfg = C.CWConfig(plant=C.bounded_plant())
        run_id = (f"cw_r1_7_2_dpwire_{now.strftime('%Y%m%dT%H%M%S')}_"
                  f"{cfg.fingerprint()}_{_src_hash()}")
        out = RESULTS / run_id
        out.mkdir(parents=True, exist_ok=True)

        def _enc(o):
            # numpy 标量（int64/float64）不是 JSON 原生类型，统一降级成 Python 标量
            if hasattr(o, "item"):
                return o.item()
            raise TypeError(f"{type(o).__name__} 不是 JSON 可序列化")

        def _dump(name, obj):
            (out / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2,
                                               default=_enc), encoding="utf-8")

        _dump("wire_gates.json", g.summary())
        _dump("price_recon.json", price)
        _dump("phat_freeze.json", w3)
        _dump("grid_scan.json", grid)
        _dump("verdict.json", verdict)
        _dump("meta.json", {
            "run_id": run_id,
            "kind": "WIRING-GATES-ONLY（本轮未跑面板）",
            "protocol": C.CWConfig.PROTOCOL_VERSION,
            "tail": "dp", "tail_default_after_run": Pmodule.TAIL_DEFAULT,
            "send_other_advance": "fixed",
            "dp_b_cap": int(cfg.protocol.horizon),
            "dp_axis_mode": "user_axis",
            "phat_freeze": "episode_prior",
            "event_pairing": cfg.protocol.event_pairing,
            "config": cfg.to_dict(),
            "dp_cache": cache_stats(),
            "at_asia_shanghai": now.isoformat(),
            "elapsed_s": elapsed,
            "git_head": _git(["rev-parse", "HEAD"]),
            "git_status_porcelain": _git(["status", "--porcelain"]),
            "src_hash": _src_hash(),
            "note": args.note,
            "disclaimer": ("本轮只判定'DP 尾部能不能接进规划器'，**不出任何收益判词**；"
                           "未跑 N1/N2；未重跑旧 P3；未训练任何模型。"),
        })
        print(f"[CW-R1.7.2] 证据落盘 results/cw_v1/{run_id}")

    print(f"[CW-R1.7.2] total {g.summary()['n_fail']} fail / {g.summary()['n_checks']} checks "
          f"({elapsed} s)")
    return 0 if g.summary()["n_fail"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
