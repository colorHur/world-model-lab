# -*- coding: utf-8 -*-
"""CW-R1.9b **单 episode 轨迹**：`dp` 在真实面板上到底看着什么在选人。

预注册：`docs/research/CW-R1.9-策略层对称-预注册-v0.1.md` §7 修订 2（T6，post-hoc，report-only）。

只做一件事：与 R1.8 面板**同配置**地跑 1 个 episode，逐槽记下
公开 `P_hat`、所选动作、以及根节点 `vals[SEND(0)] − vals[SEND(1)]`。

★ 不判 PASS/FAIL（判词已在 `r1_9_symmetry.py` 里定完）；本脚本的输出只用来
  **检验**"小幅价值倾斜 + 反馈回路"能否解释 13:1 的送达失衡。
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import config as C
from .r1_9_symmetry import _Cap, _mk
from .runner import run_episode

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results" / "cw_v1"

RHO = 0.98
RATIO = 0.5
EP = 100
SEED = 2001


class _Trace:
    """记录公开状态与根节点值（不改内层行为）。"""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.name = getattr(inner, "name", "trace")
        self.rows: list[dict] = []

    def __call__(self, v):
        a = self.inner(v)
        tel = v.telemetry
        r = self.inner.root
        self.rows.append({"t": int(v.t), "bud": int(v.budget_left),
                          "P0": float(tel[0]["P_hat"]), "P1": float(tel[1]["P_hat"]),
                          "x0": float(tel[0]["x_hat"]), "x1": float(tel[1]["x_hat"]),
                          "act": repr(a),
                          "v_send0": r.get("SEND(0)"), "v_send1": r.get("SEND(1)"),
                          "d_1_minus_0": (r["SEND(1)"] - r["SEND(0)"])})
        return a


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="CW-R1.9b 单 episode 轨迹（report-only）")
    ap.add_argument("--sw", default="0.02,0.03", help="sigma_w，逗号分隔（默认面板值）")
    ap.add_argument("--tail", default="dp", choices=("dp", "paced"))
    ap.add_argument("--ep", type=int, default=EP)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--tag", default="main")
    args = ap.parse_args(argv)
    sw = tuple(float(x) for x in args.sw.split(","))
    t0 = time.perf_counter()
    cfg = C.CWConfig(plant=C.bounded_plant(sigma_w=sw))
    budget = cfg.protocol.budget(RATIO)
    pol = _Trace(_mk(cfg, args.seed, None if args.tail == "paced" else "dp"))
    res = run_episode(cfg, pol, split="dev", episode=args.ep, budget_total=budget, rho=RHO)
    rows = pol.rows

    send0 = [r for r in rows if r["act"] == "SEND(0)"]
    send1 = [r for r in rows if r["act"] == "SEND(1)"]
    dP = [r["P0"] - r["P1"] for r in rows]
    roll_u0 = sum(1 for r in rows if r["d_1_minus_0"] > 1e-12)
    roll_u1 = sum(1 for r in rows if r["d_1_minus_0"] < -1e-12)
    out = {
        "kind": "CW-R1.9b", "note": "report-only（预注册 §7 修订 2 T6）",
        "tag": args.tag, "sigma_w": list(sw), "tail": args.tail,
        "episode": args.ep, "seed": args.seed, "budget": budget, "rho": RHO,
        "n_steps": len(rows),
        "n_send0": len(send0), "n_send1": len(send1),
        "n_probe": sum(1 for r in rows if r["act"].startswith("PROBE")),
        "n_hold": sum(1 for r in rows if r["act"] == "HOLD"),
        "n_delivered_by_user": res.n_delivered_by_user,
        "n_attempted_by_user": res.n_attempted_by_user,
        "aoi_tavg_by_user": res.aoi_tavg_by_user,
        "cost_true_by_user": res.cost_true_by_user,
        "P_diff": {"median": float(np.median(dP)), "min": float(np.min(dP)),
                   "max": float(np.max(dP)),
                   "frac_P0_gt_P1": float(np.mean([d > 0 for d in dP])),
                   "frac_abs_lt_1e-3": float(np.mean([abs(d) < 1e-3 for d in dP]))},
        "send0_when_P0_gt_P1": (sum(1 for r in send0 if r["P0"] > r["P1"]) / len(send0)
                                if send0 else None),
        "send1_when_P1_gt_P0": (sum(1 for r in send1 if r["P1"] > r["P0"]) / len(send1)
                                if send1 else None),
        "rollout_pref": {"u0": roll_u0, "u1": roll_u1, "tie_or_nan": len(rows) - roll_u0 - roll_u1},
        "median_abs_d": float(np.median([abs(r["d_1_minus_0"]) for r in rows
                                         if np.isfinite(r["d_1_minus_0"])])),
        "elapsed_s": round(time.perf_counter() - t0, 1),
        "n_run_episode": 1,
        "episode_result": {"cost_true_per_step": res.cost_true_per_step,
                           "cost_est_per_step": res.cost_est_per_step,
                           "budget_used": res.budget_used, "n_attempted": res.n_attempted},
        "trace": rows,
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    d = RESULTS / f"cw_r1_9b_trace_{args.tag}_{stamp}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "t6_trace.json").write_text(
        json.dumps({k: v for k, v in out.items() if k != "trace"},
                   ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    (d / "t6_rows.json").write_text(
        json.dumps(rows, ensure_ascii=False, default=float), encoding="utf-8")
    summary = {"tag": args.tag, "sigma_w": out["sigma_w"], "tail": args.tail}
    summary.update({k: out[k] for k in ("n_send0", "n_send1", "n_probe", "n_hold",
                                   "n_delivered_by_user", "n_attempted_by_user",
                                   "aoi_tavg_by_user", "P_diff", "send0_when_P0_gt_P1",
                                   "send1_when_P1_gt_P0", "rollout_pref", "median_abs_d",
                                   "elapsed_s")})
    print(json.dumps(summary, ensure_ascii=False, indent=1, default=float))
    print(f"[EVIDENCE] {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
