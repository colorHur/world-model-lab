# -*- coding: utf-8 -*-
"""CW-R1.7.4b 三臂的**两个必要补充**（不改任何判据、不改上一份证据）。

为什么需要（都是 R1.7.4 主跑暴露出来的缺口，不是新问题）
------------------------------------------------------
A. **送达口径**（台账 R17）：主跑只记了 `n_attempted`（尝试次数），没记 `n_delivered`
   （真正送达）。而本轮的核心观察是"dp 臂的 `mean_task_aoi` 比 `paced` 更差、发帧更多"
   ⇒ 按 R17「报改善必须同报绝对代价 / 实际送达用 frames」，必须把
   `attempted / delivered / wasted / forced_hold / 逐用户 cost` 一起报出来。
   ★ 这也是旧 58/68「served 按 attempted_idx 计数」那条教训的同型风险：不能再拿
   尝试次数冒充送达。

B. **`p̂` 混淆的单变量排查**（台账 #35 归属未定配单变量对照）：
   `tail="dp"` 的 `p̂` 由 `_wire_dp` **冻结在 episode 起点的公开先验**
   （`m_h=0, v_h=1 ⇒ p̂=0.5`，见 `plan_ref._wire_dp` docstring 与 `W3`），
   而 `tail="paced"` 的 `p̂` 是**逐槽** `_p_hat(st.mh, st.vh)`。
   ⇒ 三臂差异同时含两个变量：①值函数来源 ②用哪个 `p̂`。
   本补充用显式 `dp_phat ∈ {(0.3,0.3),(0.5,0.5),(0.7,0.7)}` 把②单独扫一遍：
   若 `G8a` 的 CV 在三个 `p̂` 上都落在同一档，则"配速改善"不是 `p̂=0.5` 这个选择造成的。

判据（看数前写死，只报不判，**不改主跑判词**）
------------------------------------------
- A 段：只报数。附加一条**自检**：`delivered ≤ attempted`、`delivered + wasted + idle ≈ 预算`
  这类恒等式由 `runner.check_identities` 给出，必须全过（否则口径错，数不可用）。
- B 段：判定"是否对 `p̂` 敏感"= 三个 `p̂` 的 CV 极差 / 中位 > 0.20。
  - 不敏感 ⇒ 主跑的 `TAILS-SWITCH` 归因可写成"值函数来源（含价格），不是 p̂ 选点"。
  - 敏感 ⇒ 归因**不成立**，必须在交付里把主结论降级为"含未分离的 p̂ 混淆"。

★ 本模块**不重写**主跑判词、不改阈值、不读 `N1/N2`。
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta, timezone

import numpy as np

from . import config as C
from .plan_gates import _Rec, _cv, _send_profile
from .plan_ref import make
from .r1_7_4_tails import (ARMS, PANEL_CFG, PANEL_EPS, REPO, RESULTS,
                           _git, _src_hash, _tail_ctx)
from .runner import check_identities, run_episode


def _one(cfg, pol, *, budget: int, rho: float, ep: int) -> dict:
    r = run_episode(cfg, pol, split="dev", episode=ep, budget_total=budget, rho=rho)
    ids = check_identities(r, cfg)
    prof = _send_profile(pol.acts, cfg.protocol.horizon)
    return {
        "episode": ep, "cv": _cv(prof), "profile": prof,
        "n_probe": int(r.n_probe),
        "n_attempted": int(r.n_attempted), "n_delivered": int(r.n_delivered),
        "n_wasted": int(getattr(r, "budget_wasted", -1)),
        "n_forced_hold": int(r.n_forced_hold),
        "budget_used": int(r.budget_used),
        "cost_true": float(r.cost_true_per_step),
        "cost_true_by_user": [float(x) for x in r.cost_true_by_user],
        "mean_task_aoi": float(r.mean_task_aoi),
        "mean_csi_aoi": float(r.mean_csi_aoi),
        "delivery_rate": (float(r.n_delivered) / max(int(r.n_attempted), 1)),
        "identities_pass": bool(ids["pass"]),
        "identities_fail": [k for k, v in ids.items() if k != "pass" and not v],
    }


def _seg_a(cfg, budget: int, rho: float) -> dict:
    """A 段：8 集 × 三臂，补齐送达口径。"""
    out = {}
    for tag, tail, grid in ARMS:
        rows = []
        t0 = time.perf_counter()
        with _tail_ctx(tail, grid):
            for ep in PANEL_EPS:
                pol = _Rec(make(cfg, arm="probe", h_plan=PANEL_CFG["h_plan"],
                                m_samples=PANEL_CFG["m_samples"],
                                seed=PANEL_CFG["seed"] - 100 + ep,
                                rho=rho, pacing=True))
                rows.append(_one(cfg, pol, budget=budget, rho=rho, ep=ep))
        out[tag] = {
            "rows": rows,
            "attempted_median": float(np.median([r["n_attempted"] for r in rows])),
            "delivered_median": float(np.median([r["n_delivered"] for r in rows])),
            "delivery_rate_median": float(np.median([r["delivery_rate"] for r in rows])),
            "wasted_median": float(np.median([r["n_wasted"] for r in rows])),
            "forced_hold_median": float(np.median([r["n_forced_hold"] for r in rows])),
            "aoi_median": float(np.median([r["mean_task_aoi"] for r in rows])),
            "cost_median": float(np.median([r["cost_true"] for r in rows])),
            "identities_all_pass": bool(all(r["identities_pass"] for r in rows)),
            "n_identity_fail": int(sum(len(r["identities_fail"]) for r in rows)),
            "elapsed_s": round(time.perf_counter() - t0, 2),
        }
        print(f"  A[{tag}] attempted={out[tag]['attempted_median']:.0f} "
              f"delivered={out[tag]['delivered_median']:.0f} "
              f"rate={out[tag]['delivery_rate_median']:.3f} "
              f"aoi={out[tag]['aoi_median']:.2f} cost={out[tag]['cost_median']:.4f} "
              f"ids_ok={out[tag]['identities_all_pass']}", flush=True)
    return out


def _seg_b(cfg, budget: int, rho: float, eps=(100, 101, 102, 103)) -> dict:
    """B 段：`dp@r173` + 显式 `p̂`（三个档），分离"值函数来源"与"p̂ 选点"。"""
    out = {"phis": {}, "eps": list(eps)}
    for pv in (0.3, 0.5, 0.7):
        rows = []
        t0 = time.perf_counter()
        with _tail_ctx("dp", "r173"):
            for ep in eps:
                pol = _Rec(make(cfg, arm="probe", h_plan=PANEL_CFG["h_plan"],
                                m_samples=PANEL_CFG["m_samples"],
                                seed=PANEL_CFG["seed"] - 100 + ep, rho=rho,
                                pacing=True, dp_phat=np.array([pv, pv])))
                rows.append(_one(cfg, pol, budget=budget, rho=rho, ep=ep))
        cvs = [r["cv"] for r in rows]
        out["phis"][str(pv)] = {
            "rows": rows, "cv_median": float(np.median(cvs)),
            "cv_min": float(min(cvs)), "cv_max": float(max(cvs)),
            "aoi_median": float(np.median([r["mean_task_aoi"] for r in rows])),
            "cost_median": float(np.median([r["cost_true"] for r in rows])),
            "delivered_median": float(np.median([r["n_delivered"] for r in rows])),
            "probe_median": float(np.median([r["n_probe"] for r in rows])),
            "elapsed_s": round(time.perf_counter() - t0, 2),
        }
        print(f"  B[p̂={pv}] cv_med={out['phis'][str(pv)]['cv_median']:.4f} "
              f"[{out['phis'][str(pv)]['cv_min']:.4f},{out['phis'][str(pv)]['cv_max']:.4f}] "
              f"aoi={out['phis'][str(pv)]['aoi_median']:.2f}", flush=True)
    meds = [v["cv_median"] for v in out["phis"].values()]
    rng = max(meds) - min(meds)
    out["phi_sensitivity"] = {
        "cv_medians": meds, "range": float(rng),
        "rel_range": float(rng / max(float(np.median(meds)), 1e-12)),
        "sensitive": bool(rng / max(float(np.median(meds)), 1e-12) > 0.20),
        "rule": "三个 p̂ 的 CV 中位相对极差 >0.20 ⇒ 对 p̂ 敏感 ⇒ 主结论须降级为含混淆",
    }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="CW-R1.7.4b 送达口径 + p̂ 混淆排查")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--seg", default="ab", help="跑哪段：a / b / ab")
    ap.add_argument("--note", default="")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    cfg = C.CWConfig(plant=C.bounded_plant())
    budget = cfg.protocol.budget(PANEL_CFG["ratio"])
    rho = PANEL_CFG["rho"]

    seg_a = _seg_a(cfg, budget, rho) if "a" in args.seg else {}
    seg_b = _seg_b(cfg, budget, rho) if "b" in args.seg else {}

    elapsed = round(time.perf_counter() - t0, 2)
    print(f"[R1.7.4b] 完成 {elapsed} s", flush=True)
    if seg_b:
        fs = seg_b["phi_sensitivity"]
        print(f"  p̂ 敏感性：rel_range={fs['rel_range']:.3f} sensitive={fs['sensitive']}")

    if not args.no_write:
        now = datetime.now(timezone(timedelta(hours=8)))
        run_id = (f"cw_r1_7_4b_delivery_{now.strftime('%Y%m%dT%H%M%S')}_"
                  f"{cfg.fingerprint()}_{_src_hash()}")
        out_dir = RESULTS / run_id
        out_dir.mkdir(parents=True, exist_ok=True)

        def _dump(name, obj):
            (out_dir / name).write_text(
                json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

        _dump("delivery.json", seg_a)
        _dump("phi_sensitivity.json", seg_b)
        _dump("meta.json", {
            "run_id": run_id, "kind": "SUPPLEMENT-NOT-A-NEW-VERDICT",
            "parent": "results/cw_v1/cw_r1_7_4_tails_20261023T130519_f11ffdefc06c0201_232550606b401065",
            "why": ("① 主跑漏记 n_delivered（R17：实际送达用 frames）；"
                    "② tail=dp 的 p̂ 冻结在 0.5 而 paced 逐槽，三臂含两个变量（#35）"),
            "segments": args.seg,
            "panel_cfg": PANEL_CFG, "panel_eps": list(PANEL_EPS),
            "config": cfg.to_dict(), "note": args.note,
            "at_asia_shanghai": now.isoformat(), "elapsed_s": elapsed,
            "git_head": _git(["rev-parse", "HEAD"]),
            "src_hash": _src_hash(),
            "disclaimer": ("本目录**不改**主跑判词 TAILS-SWITCH，只补两项必要口径与归因排查；"
                           "仍然不读 N1/N2、不宣称「测量值不值」。"),
        })
        print(f"[R1.7.4b] 证据落盘 results/cw_v1/{run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
