# -*- coding: utf-8 -*-
"""CW-R1.10b **缺陷定位**：NACK 被记到"最后一个用户"头上。

背景（R1.10 面板读数）
----------------------
`ep100/seed2001/ack=on` 与 R1.9b 逐位相同（`n_attempted=[115,13]`），但本轮的逐槽记录显示
`p̂` 差中位 **0.949**、`m_h` 绝对值到 **8.2** —— 这不可能是"偏好"，只能来自信念被单向污染。

本文件只做三件事（**不改任何被测文件**）
----------------------------------------
A 插桩计数：把 `CWWorld._ack_bump(i, direction)` 的真实入参与"该次 SEND 的真实用户"对账。
B 源码级复核：断言 `world.py` 的 `for i in range(self.n_users)` 循环位于成功/失败两支**之前**
  （循环变量 `i` 泄漏到 `elif act.kind == "SEND"` 分支）。
C **对照（诊断补丁，内存内，不落盘）**：把 NACK 改成归到 `act.user`，看失衡是否消失。
  ★ 这不是"修好"，也不是新基线；只是把"缺陷是不是原因"这一个问题答掉。

判据（写死在代码里，跑完不改）
------------------------------
A1 Σ成功计数 == `n_delivered`；Σ失败计数 == `n_nack`
A2 失败计数在用户 0 上为 **0**，而用户 0 的真实失败数 `= n_attempted[0] − n_delivered[0]` **= 71**
   ⇒ `MISATTRIBUTED`（这是本轮的硬结论）
A3 全部失败计数的下标 ∈ {`n_users − 1`}
A4 补丁版：逐用户失败计数 == 真实失败数（证明补丁本身正确）
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import config as C
from . import world as W
from .r1_10_lockin import _mk, first_probe_index
from .runner import run_episode

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results" / "cw_v1"

RHO = 0.98
RATIO = 0.5
EP, SEED = 100, 2001


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


def _src_hash() -> str:
    h = hashlib.sha256()
    for p in sorted((REPO / "experiments" / "cw_v1").glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def _git(args: list[str]) -> str:
    return subprocess.run(["D:\\devtools\\Git\\cmd\\git.exe", *args],
                          cwd=str(REPO), capture_output=True, text=True).stdout.strip()


# ------------------------------------------------------------------ 插桩
BUMPS: list[tuple[int, int, float]] = []      # (t, i, direction)
CUR: dict = {"user": None}

_ORIG_BUMP = W.CWWorld._ack_bump
_ORIG_STEP = W.CWWorld.step


def _bump_logged(self, i, direction):        # noqa: ANN001
    BUMPS.append((int(self.t), int(i), float(direction)))
    return _ORIG_BUMP(self, i, direction)


def _bump_fixed(self, i, direction):         # noqa: ANN001
    """诊断补丁：失败分支用**真实动作用户**。"""
    if direction < 0 and CUR["user"] is not None:
        i = int(CUR["user"])
    BUMPS.append((int(self.t), int(i), float(direction)))
    return _ORIG_BUMP(self, i, direction)


def _step_tagged(self, act, *, n_substeps: int = 1):   # noqa: ANN001
    CUR["user"] = int(act.user) if (act.kind == "SEND" and act.user is not None) else None
    return _ORIG_STEP(self, act, n_substeps=n_substeps)


def _install(bump_fn) -> None:
    W.CWWorld._ack_bump = bump_fn
    W.CWWorld.step = _step_tagged


def _uninstall() -> None:
    W.CWWorld._ack_bump = _ORIG_BUMP
    W.CWWorld.step = _ORIG_STEP
    assert W.CWWorld._ack_bump is _ORIG_BUMP and W.CWWorld.step is _ORIG_STEP, "补丁未还原"


def _run(patched: bool) -> tuple[dict, list[tuple[int, int, float]]]:
    BUMPS.clear()
    _install(_bump_fixed if patched else _bump_logged)
    try:
        cfg = C.CWConfig(plant=C.bounded_plant())
        budget = cfg.protocol.budget(RATIO)
        pol = _mk(cfg, SEED)
        t0 = time.perf_counter()
        res = run_episode(cfg, pol, split="dev", episode=EP, budget_total=budget, rho=RHO)
        elapsed = time.perf_counter() - t0
        rows = pol.rows
        dphi = [r["phat"][0] - r["phat"][1] for r in rows]
        out = {
            "patched": bool(patched),
            "n_attempted_by_user": [int(x) for x in res.n_attempted_by_user],
            "n_delivered_by_user": [int(x) for x in res.n_delivered_by_user],
            "n_ack": int(res.n_ack), "n_nack": int(res.n_nack),
            "n_probe": int(res.n_probe),
            "aoi_tavg_by_user": res.aoi_tavg_by_user,
            "aoi_max_by_user": res.aoi_max_by_user,
            "cost_true_per_step": res.cost_true_per_step,
            "phat_diff_median": float(np.median(dphi)) if dphi else None,
            "phat_tail_u1": [float(x) for x in rows[-1]["phat"]] if rows else None,
            "mh_min": [float(min(r["mh"][i] for r in rows)) for i in range(2)] if rows else None,
            "mh_max": [float(max(r["mh"][i] for r in rows)) for i in range(2)] if rows else None,
            "policy_trace_hash": res.policy_trace_hash,
            "elapsed_s": round(elapsed, 2),
        }
        return out, list(BUMPS)
    finally:
        _uninstall()


def _counts(bumps: list[tuple[int, int, float]]) -> dict:
    ok = collections.Counter(i for _, i, d in bumps if d > 0)
    bad = collections.Counter(i for _, i, d in bumps if d < 0)
    return {"ack_by_user": {str(k): int(v) for k, v in sorted(ok.items())},
            "nack_by_user": {str(k): int(v) for k, v in sorted(bad.items())},
            "n_ack": int(sum(ok.values())), "n_nack": int(sum(bad.values())),
            "nack_indices": sorted(set(bad))}


def _source_facts() -> dict:
    """机器复核源码事实（不靠"我读过"）。"""
    src = (REPO / "experiments" / "cw_v1" / "world.py").read_text(encoding="utf-8").splitlines()
    line_for = next((k + 1 for k, ln in enumerate(src)
                     if ln.strip() == "for i in range(self.n_users):"), None)
    line_nack = next((k + 1 for k, ln in enumerate(src)
                      if "_ack_bump(i, -1.5)" in ln), None)
    line_ack = next((k + 1 for k, ln in enumerate(src)
                     if "_ack_bump(i, +1.0)" in ln), None)
    line_user = next((k + 1 for k, ln in enumerate(src)
                      if ln.strip() == "i = act.user"), None)
    return {"for_loop_line": line_for, "nack_call_line": line_nack,
            "ack_call_line": line_ack, "first_i_assign_line": line_user,
            "loop_before_nack": bool(line_for and line_nack and line_for < line_nack),
            "rule": "`for i in range(n_users)` 在失败分支之前 ⇒ `i` 此时 = n_users−1（泄漏）；"
                    "成功分支在 `i = act.user` 之后 ⇒ 成功归属正确"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="CW-R1.10b NACK 归属缺陷定位（诊断）")
    ap.add_argument("--tag", default="main")
    args = ap.parse_args(argv)
    t0 = time.perf_counter()
    head = _git(["rev-parse", "HEAD"])
    dirty = _git(["status", "--porcelain"])

    asis, b_asis = _run(patched=False)
    fixed, b_fixed = _run(patched=True)

    ca, cf = _counts(b_asis), _counts(b_fixed)
    true_fail = [asis["n_attempted_by_user"][i] - asis["n_delivered_by_user"][i] for i in range(2)]
    true_del = asis["n_delivered_by_user"]
    # ★ 自伤 1（本脚本第一版）：A4 拿**补丁版**的计数去比**原版**的真实成败 ⇒ 判错对象
    #   （#46 同族）。补丁会改变轨迹，必须用补丁版自己的 `attempted − delivered`。
    fix_fail = [fixed["n_attempted_by_user"][i] - fixed["n_delivered_by_user"][i] for i in range(2)]
    fix_del = fixed["n_delivered_by_user"]

    gates = {
        "A1_totals": {
            "counted_ack": ca["n_ack"], "counted_nack": ca["n_nack"],
            "res_ack": asis["n_ack"], "res_nack": asis["n_nack"],
            "pass": bool(ca["n_ack"] == asis["n_ack"] and ca["n_nack"] == asis["n_nack"]),
            "rule": "Σ计数 == 记账总数（聚合层恒等 ⇒ 缺陷在聚合层看不见，这正是它活到今天的原因）",
        },
        "A2_nack_user0": {
            "counted_nack_u0": ca["nack_by_user"].get("0", 0),
            "true_fail_u0": int(true_fail[0]),
            "pass": bool(ca["nack_by_user"].get("0", 0) == 0 and true_fail[0] > 0),
            "verdict_if_pass": "MISATTRIBUTED：用户 0 的 71 次失败**一次都没记到用户 0**",
        },
        "A3_all_on_last_index": {
            "nack_indices": ca["nack_indices"], "n_users": 2,
            "pass": bool(ca["nack_indices"] == [1]),
            "rule": "全部 NACK 落在下标 n_users−1（泄漏的循环变量）",
        },
        "A4_patch_correct": {
            "patched_nack_by_user": cf["nack_by_user"], "patched_true_fail": [int(x) for x in fix_fail],
            "patched_ack_by_user": cf["ack_by_user"], "patched_true_delivered": [int(x) for x in fix_del],
            "pass": bool(all(cf["nack_by_user"].get(str(i), 0) == int(fix_fail[i]) for i in range(2))
                         and all(cf["ack_by_user"].get(str(i), 0) == int(fix_del[i])
                                 for i in range(2))),
            "rule": "补丁版逐用户 ACK/NACK 计数 == **补丁版自己的**逐用户成败（证补丁本身正确）",
        },
        "A5_source_facts": _source_facts(),
    }
    gates["A5_source_facts"]["pass"] = bool(gates["A5_source_facts"]["loop_before_nack"])

    n_fail = sum(0 if g.get("pass") else 1 for g in gates.values())
    out = {
        "kind": "CW-R1.10b-DIAGNOSTIC", "tag": args.tag,
        "note": "**诊断**：不是结果表，不是新基线；补丁只在内存里、已还原",
        "head": head, "dirty": dirty, "src_hash": _src_hash(),
        "ep": EP, "seed": SEED, "rho": RHO, "ratio": RATIO,
        "asis": asis, "fixed_attr": fixed,
        "counts_asis": ca, "counts_fixed": cf, "true_fail_by_user": [int(x) for x in true_fail],
        "gates": gates, "n_fail": int(n_fail),
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }
    d = RESULTS / f"cw_r1_10b_nackattr_{args.tag}_{_now_utc()}_{out['src_hash']}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "nack_attr.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float),
                                      encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("counts_asis", "counts_fixed", "true_fail_by_user",
                                          "asis", "fixed_attr", "n_fail")},
                     ensure_ascii=False, indent=1, default=float))
    print(json.dumps(gates["A2_nack_user0"], ensure_ascii=False))
    print(f"[EVIDENCE] {d}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
