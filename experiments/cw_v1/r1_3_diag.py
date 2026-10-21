# -*- coding: utf-8 -*-
"""CW-R1-3 的**诊断**：为什么"全知"也赢不了 `periodic4`？——离线、不落盘。

这是 `r1_3_oracle.py` 的配套诊断（教训 #39：结论出来先问"被测物有没有算对"）。
本文件**不改任何实验语义**，只把同一集的三条臂的**逐槽真值轨迹**并排打印：
1. `periodic4` 在 `B=64`（它在 `B=128` 时只用掉 64 ⇒ 这是它的**满配**）；
2. `ORACLE-KNOWN` 在 `B=64`（同预算，同样每 4 槽一帧）；
3. `ORACLE-KNOWN` 在 `B=128`（把手上的钱花光）。

要回答的三个问题（写于看数之前）
--------------------------------
Q1 同一预算下，全知臂**挑谁**是否比轮转更好？（`Σ delivered`, `Σ p_true`）
Q2 `HOLD` 的"零成本"到底是多少？—— 若 `HOLD` 已经比"每 4 槽发一帧"更便宜，
   那么"多发一帧"本身就是**负价值**，全知臂"看见谁的信道好"就更没意义。
Q3 `x_t` 的量级是否把所有帧都推到"低价值"区（`|x|` 大 ⇒ `ΔV` 小）⇒ 全知臂的
   "挑帧"决策面被压平？

★ 判词口径：本文件只输出**事实**，"为什么"写进 `docs/` 交付文档，且必须能对回这里的行。
"""

from __future__ import annotations

import json
import sys

import numpy as np

from . import config as C
from .oracle_known import OracleKnown
from .r1_baselines import _make_policy
from .runner import run_episode
from .world import Action


class _Recorder:
    """包一层策略，把每槽的 `(act, telemetry, 真值)` 记下来。"""

    def __init__(self, inner, world_holder, name=None):
        self.inner = inner
        self.name = name or getattr(inner, "name", "rec")
        self.wh = world_holder
        self.log = []

    def attach_truth(self, w):
        if hasattr(self.inner, "attach_truth"):
            self.inner.attach_truth(w)

    def __call__(self, v):
        act = self.inner(v)
        tel = v.telemetry
        w = self.wh.get("w")
        rec = {"t": int(v.t), "act": str(act),
               "b_left": int(v.budget_left),
               "x_hat": [round(float(r["x_hat"]), 6) for r in tel],
               "P_hat": [round(float(r["P_hat"]), 6) for r in tel]}
        if w is not None:
            rec["x_true"] = [round(float(x), 6) for x in w._x_true]
            rec["log_h_true"] = [round(float(x), 6) for x in w._log_h_true]
        self.log.append(rec)
        return act


def _run(cfg, make_policy, *, ep, budget, rho, needs_truth):
    from . import runner as R
    holder = {}
    pol = make_policy()
    rec = _Recorder(pol, holder)

    if not needs_truth:
        res = run_episode(cfg, rec, split="dev", episode=ep, budget_total=budget, rho=rho)
        return res, rec.log

    orig = R.CWWorld

    class _W(orig):  # type: ignore[misc,valid-type]
        def __init__(self, *a, **kw):
            kw["allow_truth_keys"] = True
            super().__init__(*a, **kw)
            holder["w"] = self
            if hasattr(pol, "attach_truth"):
                pol.attach_truth(self)

    R.CWWorld = _W
    try:
        res = R.run_episode(cfg, rec, split="dev", episode=ep, budget_total=budget,
                            rho=rho, allow_truth_keys=True)
    finally:
        R.CWWorld = orig
    return res, rec.log


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="CW-R1-3 诊断（不落盘）")
    ap.add_argument("--episode", type=int, default=100)
    ap.add_argument("--rho", type=float, default=0.85)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    cfg = C.CWConfig(plant=C.bounded_plant())
    ep, rho = args.episode, args.rho

    arms = [
        ("periodic4@B64", lambda: _make_policy("periodic4", 2), 64, False),
        ("oracle@B64", lambda: OracleKnown(cfg), 64, True),
        ("oracle@B128", lambda: OracleKnown(cfg), 128, True),
        ("periodic1@B128", lambda: _make_policy("periodic1", 2), 128, False),
        ("hold", lambda: (lambda v: Action("HOLD")), 128, False),
    ]

    out = {"episode": ep, "rho": rho, "arms": []}
    for name, mk, budget, truth in arms:
        res, log = _run(cfg, mk, ep=ep, budget=budget, rho=rho, needs_truth=truth)
        rec = {
            "arm": name, "budget": budget,
            "cost_true": res.cost_true_per_step,
            "cost_by_user": [round(float(x), 6) for x in res.cost_true_by_user],
            "n_attempted": res.n_attempted, "n_delivered": res.n_delivered,
            "n_probe": res.n_probe, "budget_used": res.budget_used,
            "delivery_rate": (round(res.n_delivered / res.n_attempted, 4)
                              if res.n_attempted else None),
            "n_slots": len(log),
            "nodes_sent": sorted({r["act"] for r in log}),
        }
        # 每槽的真值 |x| 典型尺度（用来判断"帧是不是都被推到低价值区"）
        xs = [abs(x) for r in log if "x_true" in r for x in r["x_true"]]
        rec["truestate_abs_median"] = round(float(np.median(xs)), 4) if xs else None
        rec["truestate_abs_p90"] = round(float(np.percentile(xs, 90)), 4) if xs else None
        out["arms"].append(rec)

    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0

    print(f"[diag] episode={ep} rho={rho}  （BOUNDED；episode 100 的两个车道均值不同）")
    hdr = (f"{'arm':16s} {'B':>4s} {'cost':>10s} {'u0':>9s} {'u1':>9s} "
           f"{'att':>4s} {'del':>4s} {'probe':>5s} {'used':>5s} {'|x|med':>8s} {'|x|p90':>8s}")
    print(hdr)
    for r in out["arms"]:
        print(f"{r['arm']:16s} {r['budget']:4d} {r['cost_true']:10.6f} "
              f"{r['cost_by_user'][0]:9.6f} {r['cost_by_user'][1]:9.6f} "
              f"{r['n_attempted']:4d} {r['n_delivered']:4d} {r['n_probe']:5d} "
              f"{r['budget_used']:5d} "
              f"{str(r['truestate_abs_median']):>8s} {str(r['truestate_abs_p90']):>8s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
