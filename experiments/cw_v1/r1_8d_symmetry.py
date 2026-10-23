# -*- coding: utf-8 -*-
"""CW-R1.8d **零成本**对称闸：DP 值表在"完全对称 plant"下必须满足 `V(P0,P1) == V(P1,P0)`。

动机（R1.8 主跑的观察 + R1.8b 探针）
------------------------------------
R1.8 实测：`dp16` 把 114 次 SEND 中位数给 u0、13.5 次给 u1；R1.8b 交换 `sigma_w` 后
**饿死的仍是 u1**（`LABEL-DRIVEN`），且 `p̂` 起点两用户同为 0.5。

⇒ 下一步要区分"值表本身对 P1 轴不对称"与"策略层选动作时不对称"。
本闸只查**值表**，且不跑任何 episode：

  把 plant 设成**逐用户完全对称**（`sigma_w=(s,s)`，`A=(1,1)`，`p̂=(0.5,0.5)`），
  于是唯一可能的非对称来源就是**实现**（轴/映射/广播）。
  `V_tau[i,j,b]` 与 `V_tau[j,i,b]` 必须逐位相同；任何偏差都只能归因于实现。

★ 判定：
  · `max_asym ≤ 1e-12` ⇒ 值表对称，饿死只能来自**策略层**（候选枚举 / argmax / `_need`）。
  · `max_asym` 显著 >0 ⇒ 值表本身不对称 ⇒ 属 R1.7.1 同族缺陷，须先修值表。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import config as C
from .plan_dp import get_dp

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results" / "cw_v1"


def main() -> int:
    t0 = time.perf_counter()
    cfg = C.CWConfig(plant=C.bounded_plant(sigma_w=(0.025, 0.025)))
    dp = get_dp(cfg, np.array([0.5, 0.5]), b_cap=int(cfg.protocol.horizon), axis_mode="user_axis",
                grid=None).build()
    V = dp._V
    rows = []
    for tau, Vt in enumerate(V):
        n0, n1, nb = Vt.shape
        m = min(n0, n1)
        sub = Vt[:m, :m, :]
        asym = np.abs(sub - np.transpose(sub, (1, 0, 2)))
        rel = float(np.max(asym) / max(1e-12, float(np.max(np.abs(sub)))))
        rows.append({"tau": tau, "max_abs_asym": float(np.max(asym)),
                     "max_rel_asym": rel, "scale": float(np.max(np.abs(sub)))})
    worst = max(rows, key=lambda r: r["max_abs_asym"])
    out = {"kind": "SYMMETRY-GATE", "plant": "sigma_w=(0.025,0.025) 完全对称",
           "p_hat": [0.5, 0.5], "n_tau": len(V), "rows": rows,
           "max_abs_asym": worst["max_abs_asym"], "worst_tau": worst["tau"],
           "symmetric": bool(worst["max_abs_asym"] <= 1e-12),
           "verdict": ("TABLE-SYMMETRIC" if worst["max_abs_asym"] <= 1e-12
                       else "TABLE-ASYMMETRIC"),
           "elapsed_s": round(time.perf_counter() - t0, 1),
           "rule": "对称 plant 下 V(P0,P1) 与 V(P1,P0) 必须逐位相同；偏差只能归因于实现"}
    print(json.dumps({k: out[k] for k in ("max_abs_asym", "worst_tau", "symmetric",
                                          "verdict", "elapsed_s")},
                     ensure_ascii=False, indent=1))
    d = RESULTS / "cw_r1_8d_symmetry"
    d.mkdir(parents=True, exist_ok=True)
    (d / "symmetry.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                     encoding="utf-8")
    print(f"[EVIDENCE] {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
