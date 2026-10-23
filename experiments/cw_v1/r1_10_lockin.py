# -*- coding: utf-8 -*-
"""CW-R1.10 `dp` 的**锁死通道**：切断 ACK × 换策略 seed。

预注册：`docs/research/CW-R1.10-锁死通道-预注册-v0.1.md`（写于本轮任何新计算之前）。

两件事（同一轮内，按预注册 §2 的顺序）
--------------------------------------
A **切断信息**：`use_ack_for_csi=False` ⇒ 世界与规划器都不再把 ACK 折进信道信念
  ⇒ `p̂ ≡ 0.5`（G0 用**逐槽最大偏差**钉住，且要求 `ack=on` 时该检查必须 FAIL，否则开关没接线）。
B **换硬币**：同一 episode 换 8 个策略 seed（`2001..2008`），看"偏 u0"是 8/8 还是分裂。

判据口径（预注册 §3）
--------------------
有效独立单元 = **episode 数（2）**；8 个 seed 只回答"这一集里方向是否确定"，
**不**把 `16/16` 当 16 个独立样本报显著性。

被测物 `plan_ref` / `plan_dp` / `world` / `runner` / `config` **零改动**。
"""

from __future__ import annotations

import argparse
import dataclasses as _dc
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import config as C
from . import plan_ref as Pmodule
from .runner import check_identities, run_episode

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results" / "cw_v1"

#: 与 R1.8 / R1.9 面板逐字一致的策略参数（不调参）
RHO = 0.98
H_PLAN = 5
M_MAIN = 16
RATIO = 0.5
EPS = (100, 101)
SEEDS = (2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008)
TAIL_DP = {"tail": "dp", "dp_grid": "r173"}

#: 预注册 §2 G2 的对照物（R1.9b 已入库）
R19B = RESULTS / "cw_r1_9b_trace_20261023T192329" / "t6_trace.json"
EXPECT_ATT = [115, 13]
EXPECT_DEL = [44, 7]

#: G0 的容差：`m_h` 是 0.0 乘 rho 的结果，应逐位 0；`v_h` 有 1e-16 级舍入
TOL_MH = 0.0
TOL_VH = 1e-12
#: `ack=on` 必须有分辨力：`max|m_h|` 超过这个数才算"真的动过"
LINE_MH_ON = 0.01


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


# ------------------------------------------------------------------ 记录器
class _Rec(Pmodule.PlanRef):
    """只**观测**：逐槽公开 telemetry + 根节点值；不改任何内层行为。"""

    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)
        self.rows: list[dict] = []
        self.picks: list[dict] = []

    def _pick(self, values, budget_left):          # noqa: ANN001
        self.picks.append({repr(a): float(values[a]) for a in values})
        return super()._pick(values, budget_left)

    def __call__(self, v):                          # noqa: ANN001
        a = super().__call__(v)
        tel = v.telemetry
        mh = np.asarray([[float(r["h_mean"]) for r in tel]], dtype=float)
        vh = np.asarray([[float(r["h_var"]) for r in tel]], dtype=float)
        ph = self._p_hat(mh, vh)[0]
        root = self.picks[-1] if self.picks else {}
        self.rows.append({
            "t": int(v.t), "bud": int(v.budget_left), "act": repr(a),
            "P": [float(r["P_hat"]) for r in tel],
            "x": [float(r["x_hat"]) for r in tel],
            "mh": [float(r["h_mean"]) for r in tel],
            "vh": [float(r["h_var"]) for r in tel],
            "phat": [float(x) for x in ph],
            "v_send0": root.get("SEND(0)"), "v_send1": root.get("SEND(1)"),
        })
        return a


def _mk(cfg, seed: int) -> _Rec:
    return _Rec(cfg, h_plan=H_PLAN, m_samples=M_MAIN, seed=seed, rho=RHO,
                pacing=True, **TAIL_DP)


def _cfg(ack: bool) -> C.CWConfig:
    base = C.CWConfig(plant=C.bounded_plant())
    return _dc.replace(base, channel=_dc.replace(base.channel, use_ack_for_csi=bool(ack)))


# ------------------------------------------------------------------ 单格
def run_cell(ep: int, seed: int, ack: bool, *, rows_out: bool = False) -> dict:
    cfg = _cfg(ack)
    budget = cfg.protocol.budget(RATIO)
    pol = _mk(cfg, seed)
    t0 = time.perf_counter()
    res = run_episode(cfg, pol, split="dev", episode=ep, budget_total=budget, rho=RHO)
    elapsed = time.perf_counter() - t0
    chk = check_identities(res, cfg)
    rows = pol.rows
    att = [int(x) for x in res.n_attempted_by_user]
    dele = [int(x) for x in res.n_delivered_by_user]
    mh_abs = [max(abs(r["mh"][i]) for r in rows) for i in range(2)] if rows else [0.0, 0.0]
    vh_dev = [max(abs(r["vh"][i] - 1.0) for r in rows) for i in range(2)] if rows else [0.0, 0.0]
    dphi = [r["phat"][0] - r["phat"][1] for r in rows]
    dP = [r["P"][0] - r["P"][1] for r in rows]
    dv = [r["v_send1"] - r["v_send0"] for r in rows
          if r["v_send0"] is not None and r["v_send1"] is not None]
    send0 = [r for r in rows if r["act"] == "SEND(0)"]
    send1 = [r for r in rows if r["act"] == "SEND(1)"]
    out = {
        "ep": int(ep), "seed": int(seed), "ack": bool(ack),
        "n_attempted_by_user": att, "n_delivered_by_user": dele,
        "delta_delivered": int(dele[0] - dele[1]),
        "delta_attempted": int(att[0] - att[1]),
        "ratio_delivered": (float(dele[0] / dele[1]) if dele[1] > 0 else None),
        "n_probe": int(res.n_probe), "n_hold": int(res.n_idle),
        "n_forced_hold": int(res.n_forced_hold),
        "n_never_served": int(res.n_never_served),
        "aoi_tavg_by_user": res.aoi_tavg_by_user,
        "aoi_max_by_user": res.aoi_max_by_user,
        "cost_est_per_step": res.cost_est_per_step,
        "cost_true_per_step": res.cost_true_per_step,
        "cost_true_by_user": res.cost_true_by_user,
        "mh_abs_max": mh_abs, "vh_dev_max": vh_dev,
        "phat_diff_median": float(np.median(dphi)) if dphi else None,
        "phat_diff_max_abs": float(np.max(np.abs(dphi))) if dphi else None,
        "P_diff_median": float(np.median(dP)) if dP else None,
        "P0_gt_P1_frac": float(np.mean([d > 0 for d in dP])) if dP else None,
        "v_send_diff_median": float(np.median(dv)) if dv else None,
        "v_send_pref_u0_frac": float(np.mean([d > 0 for d in dv])) if dv else None,
        "send0_when_phat0_gt_phat1": (float(np.mean([r["phat"][0] > r["phat"][1] for r in send0]))
                                     if send0 else None),
        "send1_when_phat1_gt_phat0": (float(np.mean([r["phat"][1] > r["phat"][0] for r in send1]))
                                     if send1 else None),
        "policy_trace_hash": res.policy_trace_hash,
        "identities_pass": bool(chk["pass"]),
        "identities_fail": [k for k, v in chk["items"].items() if not v],
        "elapsed_s": round(elapsed, 2),
    }
    if rows_out:
        out["rows"] = rows
    return out


def first_probe_index(rows: list[dict]) -> int:
    for k, r in enumerate(rows):
        if r["act"].startswith("PROBE"):
            return k
    return len(rows)


# ------------------------------------------------------------------ 主流程
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="CW-R1.10 dp 锁死通道（预注册 v0.1）")
    ap.add_argument("--probe", action="store_true", help="只跑 1 格报预算，**不写任何证据**")
    ap.add_argument("--tag", default="main")
    args = ap.parse_args(argv)

    if args.probe:
        t0 = time.perf_counter()
        c = run_cell(100, 2001, True)
        print(json.dumps({"probe_cell": c, "elapsed_total_s": round(time.perf_counter() - t0, 1)},
                         ensure_ascii=False, indent=1, default=float))
        return 0

    t_start = time.perf_counter()
    head = _git(["rev-parse", "HEAD"])
    dirty = _git(["status", "--porcelain"])
    srchash = _src_hash()

    gates: dict = {}
    cells: list[dict] = []
    traces: dict = {}

    # ---- 闸格：ep100/seed2001，ack=on 跑两次（G1）+ ack=off 一次（G0），带 rows
    a1 = run_cell(100, 2001, True, rows_out=True)
    a2 = run_cell(100, 2001, True, rows_out=True)
    b1 = run_cell(100, 2001, False, rows_out=True)

    # G0 接线闸
    def _pre(c: dict) -> list[dict]:
        rows = c["rows"]
        return rows[:first_probe_index(rows)]

    def _g0_side(c: dict) -> dict:
        pre = _pre(c)
        mh = max((max(abs(x) for x in r["mh"]) for r in pre), default=0.0)
        vh = max((max(abs(x - 1.0) for x in r["vh"]) for r in pre), default=0.0)
        return {"n_rows_checked": len(pre), "max_abs_mh": float(mh), "max_abs_vh_dev": float(vh),
                "n_probe_total": c["n_probe"]}

    g0_off = _g0_side(b1)
    g0_on = _g0_side(a1)
    g0_off_ok = bool(g0_off["max_abs_mh"] <= TOL_MH and g0_off["max_abs_vh_dev"] <= TOL_VH)
    g0_on_moves = bool(g0_on["max_abs_mh"] > LINE_MH_ON)
    gates["G0_switch_wired"] = {
        "ack_off_prefix": g0_off, "ack_on_prefix": g0_on,
        "rule_off_identity": "ack=off 且首个 PROBE 之前：max|m_h| == 0 且 max|v_h−1| ≤ 1e-12",
        "rule_on_discriminates": f"ack=on：max|m_h| > {LINE_MH_ON}（否则开关没接线 ⇒ 本轮作废）",
        "pass": bool(g0_off_ok and g0_on_moves),
    }

    # G1 确定性闸
    keys = ("n_attempted_by_user", "n_delivered_by_user", "n_probe", "n_hold",
            "cost_est_per_step", "cost_true_per_step", "policy_trace_hash")
    g1_same = all(a1[k] == a2[k] for k in keys)
    gates["G1_deterministic"] = {
        "keys": list(keys),
        "a1": {k: a1[k] for k in keys}, "a2": {k: a2[k] for k in keys},
        "pass": bool(g1_same),
    }

    # G2 与 R1.9b 对账
    ref = json.loads(R19B.read_text(encoding="utf-8")) if R19B.exists() else {}
    g2_ok = bool(a1["n_attempted_by_user"] == EXPECT_ATT
                 and a1["n_delivered_by_user"] == EXPECT_DEL
                 and ref.get("n_attempted_by_user") == EXPECT_ATT)
    gates["G2_reproduce_R19b"] = {
        "expect_attempted": EXPECT_ATT, "expect_delivered": EXPECT_DEL,
        "got_attempted": a1["n_attempted_by_user"], "got_delivered": a1["n_delivered_by_user"],
        "ref_file": str(R19B), "ref_attempted": ref.get("n_attempted_by_user"),
        "pass": bool(g2_ok),
    }

    n_fail_gates = sum(0 if g.get("pass") else 1 for g in gates.values())
    traces["ep100_s2001_ackon"] = {"cell": {k: v for k, v in a1.items() if k != "rows"},
                                  "rows": a1["rows"]}
    traces["ep100_s2001_ackoff"] = {"cell": {k: v for k, v in b1.items() if k != "rows"},
                                   "rows": b1["rows"]}

    if n_fail_gates:
        # 闸没过 ⇒ 不许把面板数字当结论（预注册 §5）
        d = RESULTS / f"cw_r1_10_lockin_{args.tag}_{_now_utc()}_{srchash}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "gates.json").write_text(json.dumps(gates, ensure_ascii=False, indent=1,
                                                 default=float), encoding="utf-8")
        (d / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, default=float),
                                       encoding="utf-8")
        print(json.dumps({"n_fail_gates": n_fail_gates, "gates": gates},
                         ensure_ascii=False, indent=1, default=float))
        print(f"[ABORT] 闸未过，面板不跑。证据 {d}")
        return 1

    cells.append({k: v for k, v in a1.items() if k != "rows"})
    cells.append({k: v for k, v in b1.items() if k != "rows"})

    # ---- 面板：其余 30 格（无 rows，只留摘要）
    for ep in EPS:
        for ack in (True, False):
            for seed in SEEDS:
                if ep == 100 and seed == 2001:
                    continue                      # 已在闸格里跑过
                cells.append(run_cell(ep, seed, ack))

    # ---- 判词（预注册 §2 T2）
    def _verdict(ep: int, ack: bool) -> dict:
        sub = [c for c in cells if c["ep"] == ep and c["ack"] is ack]
        sub.sort(key=lambda c: c["seed"])
        ds = [c["delta_delivered"] for c in sub]
        k = sum(1 for d in ds if d > 0)
        n = len(ds)
        if k == n and n >= 4:
            lab = "ALL-U0"
        elif k == 0:
            lab = "ALL-U1"
        elif 2 <= k <= n - 2:
            lab = "SPLIT"
        else:
            lab = "NEAR-ALL"        # 7/8、6/8 这类：n=8 上不显著，单列
        return {"ep": int(ep), "ack": bool(ack), "n_seeds": n, "k_u0": int(k),
                "label": lab, "deltas": ds,
                "sum_attempted": [sum(c["n_attempted_by_user"][i] for c in sub) for i in range(2)],
                "sum_delivered": [sum(c["n_delivered_by_user"][i] for c in sub) for i in range(2)],
                "n_probe_total": int(sum(c["n_probe"] for c in sub)),
                "aoi_tavg_median": [_med([c["aoi_tavg_by_user"][i] for c in sub]) for i in range(2)],
                "n_never_served_any": int(sum(c["n_never_served"] for c in sub))}

    verdicts = {f"ep{ep}_{'on' if ack else 'off'}": _verdict(ep, ack)
                for ep in EPS for ack in (True, False)}

    judges = {}
    for ep in EPS:
        on = verdicts[f"ep{ep}_on"]
        off = verdicts[f"ep{ep}_off"]
        first = {"ALL-U0": "SYSTEMATIC-GIVEN-EPISODE", "SPLIT": "COIN-FLIP",
                 "ALL-U1": "ALL-U1-ANOMALY", "NEAR-ALL": "NEAR-ALL-NOT-SIGNIFICANT"}[on["label"]]
        second = {"ALL-U0": "LOCK-ACK-INDEPENDENT", "SPLIT": "LOCK-ACK-FEEDBACK",
                  "ALL-U1": "ANOMALY-CHECK-IMPL", "NEAR-ALL": "NEAR-ALL-NOT-SIGNIFICANT"}[off["label"]]
        judges[f"ep{ep}"] = {"systematic_or_coin": first, "ack_channel": second,
                             "note": ("有效独立单元 = 2 个 episode；8 个 seed 只说明集内确定性，"
                                      "不构成 8 个独立样本")}

    elapsed = round(time.perf_counter() - t_start, 1)
    d = RESULTS / f"cw_r1_10_lockin_{args.tag}_{_now_utc()}_{srchash}"
    d.mkdir(parents=True, exist_ok=True)
    meta = {
        "kind": "CW-R1.10", "tag": args.tag, "protocol": C.CWConfig.PROTOCOL_VERSION,
        "prereg": "docs/research/CW-R1.10-锁死通道-预注册-v0.1.md",
        "head": head, "dirty": dirty, "src_hash": srchash,
        "params": {"rho": RHO, "h_plan": H_PLAN, "m_samples": M_MAIN, "ratio": RATIO,
                   "eps": list(EPS), "seeds": list(SEEDS), "tail": TAIL_DP,
                   "budget": int(C.CWConfig(plant=C.bounded_plant()).protocol.budget(RATIO))},
        "cfg_ack_on": _cfg(True).to_dict(), "cfg_ack_off": _cfg(False).to_dict(),
        "n_cells": len(cells), "elapsed_s": elapsed,
        "command": "python -m experiments.cw_v1.r1_10_lockin",
    }
    (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1,
                                            default=float), encoding="utf-8")
    (d / "gates.json").write_text(json.dumps(gates, ensure_ascii=False, indent=1,
                                             default=float), encoding="utf-8")
    (d / "cells.json").write_text(json.dumps(cells, ensure_ascii=False, indent=1,
                                             default=float), encoding="utf-8")
    (d / "verdicts.json").write_text(json.dumps({"verdicts": verdicts, "judges": judges},
                                                ensure_ascii=False, indent=1,
                                                default=float), encoding="utf-8")
    (d / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, default=float),
                                   encoding="utf-8")

    print(json.dumps({"n_fail_gates": 0, "verdicts": verdicts, "judges": judges,
                      "elapsed_s": elapsed, "n_cells": len(cells)},
                     ensure_ascii=False, indent=1, default=float))
    print(f"[EVIDENCE] {d}")
    return 0


def _med(xs):
    xs = [x for x in xs if x is not None]
    return float(np.median(xs)) if xs else None


if __name__ == "__main__":
    raise SystemExit(main())
