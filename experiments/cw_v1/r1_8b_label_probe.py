# -*- coding: utf-8 -*-
"""CW-R1.8b **探索性**探针：`dp` 臂为什么把 u1 饿死 —— 跟"下标"走还是跟"噪声"走。

★ 性质：**探索性**（预注册 §9 之外）。不承载任何判词，不进主表；
  唯一目的是把 R1.8 交付里那条"dp 只服务 u0"的观察从"现象"推进到"可判别的假设"。

背景（R1.8 实测，`cw_r1_8_mech_20261023T170326_08b6bfcbb8c7dec7`）
------------------------------------------------------------------
`dp16` 在 8/8 对上把 96–121 次 SEND 给了 u0、只给 u1 **6–21** 次（送达 u1 只有 2–8 帧）。
而目标函数是 `q·Σ_i x_i²`（**逐用户等权**，`q_cost=1.0`），`sigma_w=(0.02, 0.03)`
⇒ u1 的过程噪声**更大**、留它陈旧**更贵** ⇒ 一个正确的最小化者应当**更多**服务 u1。
实测方向相反。本探针区分两个假设：

- H1（下标驱动）：把 `sigma_w` 换成 `(0.03, 0.02)`（只交换噪声，不换任何代码），
  若 `dp` **仍然**把 u1 饿死 ⇒ 行为跟**用户下标**走 ⇒ 疑为轴/下标不对称缺陷（R1.7.1 同族）。
- H2（噪声驱动）：若交换后 `dp` 改成饿死 u0 ⇒ 行为跟**物理**走，属"分配偏好"，
  但那样它与等权目标仍矛盾，须另找理由。

另做零成本前提：`p̂` 在起点必须**逐用户相同**（先验 `m_h=0, v_h=1`）——
若 `p̂` 本身不对称，"饿死"就只是 `p_hat` 接线问题。
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import config as C
from . import plan_gates as G
from .plan_ref import make
from .runner import run_episode

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results" / "cw_v1"

EP, SEED, RHO, RATIO, H_PLAN, M = 100, 2001, 0.98, 0.5, 5, 16
TAIL_DP = {"tail": "dp", "dp_grid": "r173"}


def _one(cfg, budget, tail):
    kw = dict(tail) if tail else {}
    pol = G._Rec(make(cfg, arm="probe", h_plan=H_PLAN, m_samples=M, seed=SEED,
                      rho=RHO, pacing=True, **kw))
    r = run_episode(cfg, pol, split="dev", episode=EP, budget_total=budget, rho=RHO)
    return {"n_attempted_by_user": r.n_attempted_by_user,
            "n_delivered_by_user": r.n_delivered_by_user,
            "aoi_tavg_by_user": r.aoi_tavg_by_user,
            "aoi_max_by_user": r.aoi_max_by_user,
            "mean_task_aoi_old": float(r.mean_task_aoi),
            "n_probe": int(r.n_probe),
            "policy_trace_hash": str(r.policy_trace_hash),
            "elapsed_s": float(r.elapsed_s)}


def main() -> int:
    t0 = time.perf_counter()
    # ---- P0 零成本：起点的 p̂ 是否逐用户相同
    cfg0 = C.CWConfig(plant=C.bounded_plant())
    pol0 = G._Rec(make(cfg0, arm="probe", h_plan=H_PLAN, m_samples=M, seed=SEED,
                       rho=RHO, pacing=True, **TAIL_DP))
    ph = np.asarray(pol0.inner._p_hat(np.zeros((1, 2)), np.full((1, 2), 1.0))[0], dtype=float)
    p0 = {"p_hat_prior": [float(x) for x in ph],
          "symmetric": bool(abs(ph[0] - ph[1]) <= 1e-12),
          "rule": "先验 m_h=0,v_h=1 ⇒ 两用户 p̂ 必须相同；不同则饿死只是 p̂ 接线问题"}

    # ---- 主探针：只交换 sigma_w，其它一个不改
    rows = {}
    for tag, sw in (("base_(0.02,0.03)", (0.02, 0.03)), ("swap_(0.03,0.02)", (0.03, 0.02))):
        cfg = C.CWConfig(plant=C.bounded_plant(sigma_w=sw))
        budget = cfg.protocol.budget(RATIO)
        rows[tag] = {"sigma_w": list(sw)}
        for arm, tail in (("paced16", None), ("dp16", TAIL_DP)):
            rows[tag][arm] = _one(cfg, budget, tail)

    def _verdict() -> dict:
        base = rows["base_(0.02,0.03)"]["dp16"]["n_attempted_by_user"]
        swap = rows["swap_(0.03,0.02)"]["dp16"]["n_attempted_by_user"]
        starved_base = int(np.argmin(base))
        starved_swap = int(np.argmin(swap))
        # base 上 sigma_w 小的是 u0；swap 上 sigma_w 小的是 u1
        low_noise_base = int(np.argmin([0.02, 0.03]))
        low_noise_swap = int(np.argmin([0.03, 0.02]))
        follows_label = bool(starved_base == starved_swap)
        follows_physics = bool(starved_base == low_noise_base and starved_swap == low_noise_swap)
        if follows_label and not follows_physics:
            v = "LABEL-DRIVEN"
        elif follows_physics and not follows_label:
            v = "PHYSICS-DRIVEN"
        else:
            v = "AMBIGUOUS"
        return {"starved_base": starved_base, "starved_swap": starved_swap,
                "attempts_base": base, "attempts_swap": swap,
                "follows_label": follows_label, "follows_physics": follows_physics,
                "verdict": v,
                "rule": "H1 下标驱动 = 两次都饿死同一个下标；H2 噪声驱动 = 每次都饿死低噪声用户"}

    out = {"kind": "EXPLORATORY-NOT-A-VERDICT", "round": "CW-R1.8b",
           "premise_p0": p0, "rows": rows, "probe": _verdict(),
           "elapsed_s": round(time.perf_counter() - t0, 1),
           "note": "探索性探针：不承载判词、不进主表；只用来区分 下标驱动 / 噪声驱动"}
    print(json.dumps(out, ensure_ascii=False, indent=1))

    d = RESULTS / f"cw_r1_8b_labelprobe_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_EXPLORATORY"
    d.mkdir(parents=True, exist_ok=True)
    (d / "probe.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[EVIDENCE] {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
