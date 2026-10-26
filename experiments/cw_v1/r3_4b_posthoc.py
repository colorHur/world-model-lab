# -*- coding: utf-8 -*-
"""`CW-R3.4b` **跑后分析**（明确标注：探索性，**不是**预注册判据）。

读 `r34.json`，只做三件事，全部是"表上看出来的"，跑前没写：

`X1` 首步恒等：`IV_free == max(0, −(V_PROBE − V_SEND))`？
     —— 若成立，则面板上的"信息价值"**全部**来自根状态首步比较，不含深层策略效应。
`X2` 实测上界 `F4`：`VoI ≤ π_slot − (V_PROBE − V_SEND)`（`PROBE` 被支配时），
     `VoI = IV_free`（`PROBE` 被选上时）。**这条是跑后构造的，只能算探索性**。
`X3` `J2` 的中位比为什么会是 1.000：`b=1` 的格上 `π_slot ≡ ΔV_q`（代数退化），
     以及 `π_slot < 0` 的格（"等"比"发"便宜 ⇒ 不是价格）。

★ 本文件不改 `r34.json`，只额外写 `r34_posthoc.json`。
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
EPS = 1e-12


def flip_and_snap() -> dict:
    """`X4`：R3.3 §7.1-4 把过估归因到 ②翻面概率 `≪1` ③`snap` 量化。这两条成立吗？

    **诊断单格**（`τ=4,b=2,P=0.02,n_m=129`），不产生研究结论：
    - ② 加权翻面率：读数让 `t−1` 时刻的首步 argmin（`SEND_0` vs `SEND_1`）改变的概率；
    - ③ `snap` 相对误差：网格步长 `2·M_RANGE/(n_m−1)` ÷ 读数导致的**后验均值位移 sd**。
    """
    from experiments.cw_v1.config import mainline_config
    from experiments.cw_v1.r3_2_iv2u import SIGMA_JUDGE, build_engine

    cfg = mainline_config()
    rho = float(cfg.channel.rho_panel[0])
    tau, b, P, n_m = 4, 2, 0.02, 129
    si2 = SIGMA_JUDGE ** 2
    gp = 1.0 / (1.0 + si2)
    vp = rho ** 2 * (si2 / (1.0 + si2)) + (1.0 - rho ** 2)
    vv = rho ** 2 * 1.0 + (1.0 - rho ** 2)
    sd_shift = rho * gp * math.sqrt(1.0 + si2)

    eng = build_engine(cfg, SIGMA_JUDGE, n_m, probe_enabled=False)
    f_prior = eng.first(tau - 1, P, P, 0.0, vv, 0.0, vv, b)
    act_prior = min(f_prior, key=lambda k: f_prior[k])
    nodes, w = np.polynomial.hermite_e.hermegauss(21)
    w = w / math.sqrt(2.0 * math.pi)
    p_flip = 0.0
    n_flip_nodes = 0
    for z, wi in zip(nodes, w):
        mp = sd_shift * float(z)
        fz = eng.first(tau - 1, P, P, mp, vp, 0.0, vv, b)
        if min(fz, key=lambda k: fz[k]) != act_prior:
            p_flip += float(wi)
            n_flip_nodes += 1
    step = 2.0 * 4.0 / (n_m - 1)                    # M_RANGE=4.0，n_m 点
    return {"cell": {"tau": tau, "b": b, "P": P, "n_m": n_m},
            "act_prior": act_prior,
            "p_flip_weighted": float(p_flip),
            "n_flip_nodes": int(n_flip_nodes), "n_nodes": int(len(nodes)),
            "sd_posterior_mean_shift": float(sd_shift),
            "grid_step": float(step),
            "snap_rel": float(step / sd_shift),
            "grid_step_nm65": float(2.0 * 4.0 / 64),
            "snap_rel_nm65": float((2.0 * 4.0 / 64) / sd_shift)}


def main(argv: list[str]) -> int:
    src = Path(argv[0]) if argv else sorted((REPO / "results" / "cw_v1").glob(
        "cw_r3_4_slotprice_*/r34.json"))[-1]
    d = json.loads(Path(src).read_text(encoding="utf-8"))
    cells = d["cells"]

    # ---- X1：首步恒等
    x1 = []
    for c in cells:
        pred = max(0.0, -(c["probe_minus_send"]))
        x1.append({"tau": c["tau"], "b": c["b"], "P": c["P"], "n_m": c["n_m"],
                   "IV_free": c["IV_free"], "pred": pred,
                   "abs": abs(c["IV_free"] - pred)})
    worst1 = max(r["abs"] for r in x1)
    n_pos = sum(1 for r in x1 if r["pred"] > EPS)
    x1_ok = worst1 <= 1e-15

    # ---- X2：实测上界 F4
    f4 = []
    for c in cells:
        dom = c["probe_minus_send"] > EPS
        voi_ub = (c["pi_slot"] - c["probe_minus_send"]) if dom else c["IV_free"]
        f4.append({"tau": c["tau"], "b": c["b"], "P": c["P"], "n_m": c["n_m"],
                   "dominated": bool(dom), "VoI_ub": float(voi_ub),
                   "IV_free": c["IV_free"], "dV_q": c["dV_q"],
                   "ub_over_price": (float(voi_ub / c["dV_q"]) if c["dV_q"] > 0 else None),
                   "says_buy": bool(c["dV_q"] > 0 and voi_ub > c["dV_q"])})
    buys = [r for r in f4 if r["says_buy"]]
    ratios = [r["ub_over_price"] for r in f4 if r["ub_over_price"] is not None]
    med_ratio = float(np.median(ratios)) if ratios else None
    max_ratio = float(max(ratios)) if ratios else None

    # ---- X3：J2 退化的来源
    b1 = [c for c in cells if c["b"] == 1]
    b1_ident = all(abs(c["pi_slot"] - c["dV_q"]) <= 1e-12 for c in b1)
    ge2 = [c for c in cells if c["b"] >= 2 and c["dV_q"] > EPS and c["pi_slot"] > EPS]
    ge2_ratio = [c["pi_slot"] / c["dV_q"] for c in ge2]
    neg = [{"tau": c["tau"], "b": c["b"], "P": c["P"], "pi_slot": c["pi_slot"],
            "IV_free": c["IV_free"]} for c in cells if c["pi_slot"] < -EPS]

    # ---- X4：R3.3 那三条归因里的 ②③ 到底成不成立（**诊断单格**，不产生研究结论）
    #   ② 翻面概率：读数让首步 argmin 改变的概率（GH 权重加权，不是数节点）
    #   ③ snap 量化：网格步长 / 读数导致的后验均值位移 sd
    x4 = flip_and_snap()

    out = {
        "source": str(src).replace("\\", "/"),
        "X1": {"identity": "IV_free == max(0, -(V_PROBE - V_SEND))",
               "worst_abs": float(worst1), "n_cells": len(x1),
               "n_pred_positive": int(n_pos), "ok": bool(x1_ok), "rows": x1},
        "X2": {"bound": "VoI_ub = pi_slot - (V_PROBE - V_SEND) [dominated] / IV_free [chosen]",
               "n_cells": len(f4), "n_says_buy": len(buys),
               "says_buy_cells": [r for r in f4 if r["says_buy"]],
               "median_ub_over_price": med_ratio, "max_ub_over_price": max_ratio,
               "rows": f4},
        "X3": {"n_b1": len(b1), "b1_pi_equals_dVq": bool(b1_ident),
               "n_b_ge2": len(ge2), "median_ratio_b_ge2": float(np.median(ge2_ratio)) if ge2_ratio else None,
               "min_ratio_b_ge2": float(min(ge2_ratio)) if ge2_ratio else None,
               "max_ratio_b_ge2": float(max(ge2_ratio)) if ge2_ratio else None,
               "negative_pi_slot": neg},
        "X4": x4,
    }
    dest = Path(src).parent / "r34_posthoc.json"
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[X1] IV_free == max(0, -（PROBE-SEND)) : {x1_ok}  worst={worst1:.3e} "
          f"正格={n_pos}/{len(x1)}")
    print(f"[X2] F4 判\"买得起\"的格: {len(buys)}/{len(f4)} ; "
          f"中位 UB/价={med_ratio:.3f} ; 最大={max_ratio:.3f}")
    for r in buys:
        print(f"      tau={r['tau']} b={r['b']} P={r['P']} VoI_ub={r['VoI_ub']:.3e} "
              f"dV_q={r['dV_q']:.3e} ratio={r['ub_over_price']:.2f}")
    print(f"[X3] b=1 上 pi_slot==dV_q: {b1_ident} (n={len(b1)}) ; "
          f"b>=2 比值 中位={out['X3']['median_ratio_b_ge2']} "
          f"范围=[{out['X3']['min_ratio_b_ge2']}, {out['X3']['max_ratio_b_ge2']}]")
    print(f"     π_slot<0 的格: {neg}")
    print(f"[X4] 翻面率(加权)={x4['p_flip_weighted']:.3f} "
          f"(节点 {x4['n_flip_nodes']}/{x4['n_nodes']}, act_prior={x4['act_prior']}) ; "
          f"snap 相对误差 n_m=129: {x4['snap_rel']:.3f}, n_m=65: {x4['snap_rel_nm65']:.3f}")
    print(f"[out] {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
