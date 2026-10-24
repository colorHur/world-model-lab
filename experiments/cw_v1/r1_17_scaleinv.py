# -*- coding: utf-8 -*-
"""CW-R1.17 把机制判据换成**尺度不变**形态 —— 零新 episode，只对 R1.16 的 `cells.json` 重算。

预注册：`docs/research/CW-R1.17-尺度不变机制判据-预注册-v0.1.md`

为什么换
--------
R1.16 的 `share = E_term / Δr` 在 `|Δr|` 小时爆（`sh_user` 最高 2.76、`sh_frame` 1.92），
`Spearman(|Δr|, max|share|) = −0.7333` ⇒ "机制归因"这一问在 `|Δr|` 小的格上不可读。

换法：把含混的一问拆成三问（预注册 §1）——
`Q_dir` 方向（逐 ep 符号 + 符号计数）/ `Q_mag` 强度（`ku=E_user/Mu`）/ `Q_share` 解释度（降级为描述）。

★ 尺度不变得以成立的**严格理由**（不是断言）
--------------------------------------------
`ku = E_user/Mu = Σ_i w_i σ_i (ρ_i − ρ_a)`，其中 `w_i=|m_i|/Σ|m_i|`（和为 1）、`σ_i=sign(m_i)`。
⇒ **`|ku| ≤ max_i |ρ_i − ρ_a| ≤ 1`**，即 `ku` 恒有界；且 `E_user` 与 `Mu` **都只依赖 `ni_*` 与 `nd_a`**，
不含 `nd_e` ⇒ 只改送达结果（`nd_e`）时 `ku` **逐位不变**，而 `Δr` 会变 ⇒ `share` 会变。
`S1`/`S6` 就是这个构造的实测版。

被测物：`plan_ref/plan_dp/world/runner/config` **一字节不动**；本文件只读 `cells.json` 重算。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import gates as G
from .mech import (KAPPA_MIN, SIGN_MIN, binom_two_sided, combine_dir,  # noqa: F401
                   dir_verdict, mech_quantities)
from .r1_14_exactbase import RESULTS, _git, _src_hash
from .r1_15_delivery_gap import ep_counts
from .r1_16_n10 import boot_ci, perm_p, spearman

#: R1.16 的正式证据（**只读输入**；不许改它一个字节）
R116_DIR = RESULTS / "cw_r1_16_n10_20261024T105416_fc9f25e16d2c6307"
R116_CELLS = R116_DIR / "cells.json"

EPS10 = tuple(range(100, 110))

#: 预注册 §8-2：分母下限（`Mu`/`Mf` 太小则机制未发生）。实测最小值 0.046 ⇒ 本面板上不起筛选作用
MU_MIN = 0.01
#: 预注册 §3.3：**描述**子集的 `|Δr|` 门槛（不参与判词）
DR_DESC = 0.02
#: 阈值敏感性网格（预注册 §4.1）
KAPPA_GRID = (0.005, 0.01, 0.02)
DESC_GRID = (0.01, 0.02, 0.05)
#: 预测因子门槛（沿用 R1.16 一字不改）
RHO_SUP, RHO_REJ, MIN_POINTS = 0.60, 0.40, 6
SEED_STAT = 20261024
PERM_N = 100_000


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")



def verdict_predictor(rho: float, p: float, n_pts: int) -> dict:
    if n_pts < MIN_POINTS or not np.isfinite(rho):
        return {"verdict": "NOT-ENOUGH-POINTS", "rho": rho, "p_perm": p, "n_points": n_pts}
    if abs(rho) < RHO_REJ:
        v = "PREDICTOR-REJECTED"
    elif rho >= RHO_SUP:
        v = "PREDICTOR-SUPPORTED" if p < 0.05 else "PREDICTOR-SUPPORTED-DIRECTION-ONLY"
    elif rho <= -RHO_SUP:
        v = "PREDICTOR-SUPPORTED-NEGATIVE" if p < 0.05 else "PREDICTOR-DIRECTION-ONLY-NEGATIVE"
    else:
        v = "PREDICTOR-INCONCLUSIVE"
    return {"verdict": v, "rho": rho, "p_perm": p, "n_points": n_pts}


# ============================================================ 判据自证（零 episode）
def _counts_from(rows_ni, rows_nd):
    ni = np.asarray(rows_ni, dtype=float)
    nd = np.asarray(rows_nd, dtype=float)
    return ni, nd


def _synth_unbiased():
    """`S2` 无偏搬运：4 用户、m 与 ρ 无关（两进两出、符号对称）。手算 `ku = 0`。"""
    ni_a = np.array([[100, 0], [100, 0], [100, 0], [100, 0]], dtype=float)
    nd_a = np.array([[90, 0], [70, 0], [30, 0], [10, 0]], dtype=float)   # ρ = .9/.7/.3/.1，ρ_a=0.5
    m = np.array([40, -40, -40, 40], dtype=float)                        # 两进两出、对称
    ni_e = ni_a + m[:, None]
    nd_e = nd_a.copy()                                                   # 送达率不变 ⇒ E_within=0
    return ni_e, nd_e, ni_a, nd_a


def _synth_biased(sign: int = -1):
    """`S3`/`S5` 系统性搬运：`sign=-1` 全搬向差用户；`sign=+1` 全搬向好用户。`|ku|=0.6`。"""
    ni_a = np.array([[100, 0], [100, 0], [100, 0], [100, 0]], dtype=float)
    nd_a = np.array([[90, 0], [70, 0], [30, 0], [10, 0]], dtype=float)
    m = np.array([-40, -40, 40, 40], dtype=float) * (1 if sign < 0 else -1)
    ni_e = ni_a + m[:, None]
    nd_e = nd_a.copy()
    return ni_e, nd_e, ni_a, nd_a


def _shift_delivered(ni_e, nd_e, ni_a, nd_a, target_total: int):
    """只改 `nd_e` 的**整数**扰动，使 `Σ nd_e → target_total`（逐格 `0 ≤ nd ≤ ni_e`）。"""
    nd = np.asarray(nd_e, dtype=int).copy()
    ni = np.asarray(ni_e, dtype=int)
    cur = int(nd.sum())
    if target_total < cur:
        need = cur - target_total
        for i in range(nd.shape[0]):
            for b in range(nd.shape[1]):
                if need <= 0:
                    break
                take = min(need, int(nd[i, b]))
                nd[i, b] -= take
                need -= take
    elif target_total > cur:
        need = target_total - cur
        for i in range(nd.shape[0]):
            for b in range(nd.shape[1]):
                if need <= 0:
                    break
                add = min(need, int(ni[i, b]) - int(nd[i, b]))
                nd[i, b] += add
                need -= add
    return nd


def controls(cells_json: Path = R116_CELLS) -> dict:
    """预注册 §5 的 `S1–S9`。**零 episode**：合成构造 + 复用 R1.16 的真实计数矩阵。"""
    out: dict = {}

    # ---------------- S1/S6 尺度不变性：只改 nd_e
    cells = json.loads(cells_json.read_text(encoding="utf-8"))
    ni_e0, nd_e0 = ep_counts(cells, "exact", 100)
    ni_a0, nd_a0 = ep_counts(cells, "approx", 100)
    q0 = mech_quantities(ni_e0, nd_e0, ni_a0, nd_a0)

    r_a = q0["rate_a"]
    n_e = int(ni_e0.sum())
    tgt = int(round(n_e * r_a))
    nd_e1 = _shift_delivered(ni_e0, nd_e0, ni_a0, nd_a0, tgt)
    q1 = mech_quantities(ni_e0, nd_e1, ni_a0, nd_a0)

    d_ku = abs(q1["ku"] - q0["ku"])
    d_mu = abs(q1["Mu"] - q0["Mu"])
    ratio_sh = (abs(q1["shares"]["user"]) / abs(q0["shares"]["user"])
                if abs(q0["shares"]["user"]) > 0 else float("nan"))
    out["S1"] = {"ep": 100, "dr_orig": q0["dr"], "dr_shifted": q1["dr"],
                 "ku_orig": q0["ku"], "ku_shifted": q1["ku"], "d_ku": d_ku, "d_Mu": d_mu,
                 "share_orig": q0["shares"]["user"], "share_shifted": q1["shares"]["user"],
                 "share_ratio": ratio_sh,
                 "pass": bool(d_ku < 1e-12 and d_mu < 1e-12 and abs(q1["dr"]) < abs(q0["dr"])
                              and (ratio_sh > 10 or not np.isfinite(ratio_sh)))}

    # ---------------- S6 极端：Δr 落到一个送达的分辨率 ⇒ 旧判据**符号翻转**，ku 仍逐位不变
    #   ★ 修订（见预注册 §8-3）：原断言写 "|share| > 1e3" —— 在整数送达数下**不可达**
    #     （`share ≈ |E_user|·N^E/1 ≈ 13` 是上界）。改为"符号翻转 + ≥10× + ku 不变"，
    #     这既不依赖拍的阈值，又正是旧判据失效的可操作定义。
    tgt0 = None
    q_ex = None
    for cand in (tgt, tgt + 1, tgt - 1, tgt + 2, tgt - 2):
        nd_c = _shift_delivered(ni_e0, nd_e0, ni_a0, nd_a0, cand)
        qc = mech_quantities(ni_e0, nd_c, ni_a0, nd_a0)
        if abs(qc["dr"]) <= 2.0 / n_e:      # 落在 1–2 个送达的分辨率内
            tgt0, q_ex = cand, qc
            break
    if tgt0 is None:
        tgt0, q_ex = tgt, q1
    sh0 = q0["shares"]["user"]
    sh1 = q_ex["shares"]["user"]
    finite1 = bool(np.isfinite(sh1))
    sign_flip = bool(finite1 and np.sign(sh1) != np.sign(sh0))
    big = bool((not finite1) or abs(sh1) > 10 * abs(sh0))
    out["S6"] = {"target_total": int(tgt0), "dr": q_ex["dr"],
                 "share_user": sh1, "share_user_orig": sh0,
                 "sign_flip": sign_flip, "share_ge_10x_or_nan": big,
                 "ku": q_ex["ku"], "d_ku": abs(q_ex["ku"] - q0["ku"]),
                 "pass": bool(big and (sign_flip or not finite1)
                              and abs(q_ex["ku"] - q0["ku"]) < 1e-12)}

    # ---------------- S2 无偏搬运 ⇒ ku = 0 ⇒ UNRESOLVED
    q = mech_quantities(*_synth_unbiased())
    v = dir_verdict([q["ku"]] * 10)
    out["S2"] = {"ku": q["ku"], "E_user": q["E_user"], "Mu": q["Mu"], "verdict": v["verdict"],
                 "pass": bool(abs(q["ku"]) < 1e-12
                              and v["verdict"] == "DIRECTION-UNRESOLVED")}

    # ---------------- S3 全搬向差用户 ⇒ ku = −0.6 ⇒ CONFIRMED
    q = mech_quantities(*_synth_biased(-1))
    v = dir_verdict([q["ku"]] * 10)
    out["S3"] = {"ku": q["ku"], "expect_ku": -0.6, "verdict": v["verdict"],
                 "n_neg": v["n_neg"], "p": v["p_binom_null50"],
                 # ★ 修订（预注册 §8-3）：首版断言 `p == 0.021484375`，那是 **9/10** 的值；
                 #   10/10 的正确值是 `2/2^10 = 0.001953125`（手算错，判据自伤，非被测物错）
                 "expect_p": 0.001953125,
                 "pass": bool(abs(q["ku"] + 0.6) < 1e-12
                              and v["verdict"] == "DIRECTION-CONFIRMED:USER-NEGATIVE"
                              and v["n_neg"] == 10
                              and abs(v["p_binom_null50"] - 0.001953125) < 1e-9)}

    # ---------------- S4 两臂分配完全相同 ⇒ Mu=0 ⇒ ku=NaN ⇒ UNRESOLVED（不是"确认"）
    ni_a = np.array([[100, 0], [100, 0]], dtype=float)
    nd_a = np.array([[80, 0], [20, 0]], dtype=float)
    q = mech_quantities(ni_a, nd_a, ni_a, nd_a)
    v = dir_verdict([q["ku"]] * 10)
    out["S4"] = {"Mu": q["Mu"], "ku": q["ku"], "is_nan": bool(np.isnan(q["ku"])),
                 "n_nan": v["n_nan"], "verdict": v["verdict"],
                 "pass": bool(q["Mu"] == 0.0 and np.isnan(q["ku"])
                              and v["n_nan"] == 10 and v["verdict"] == "DIRECTION-UNRESOLVED")}

    # ---------------- S5 反向 ⇒ ku = +0.6 ⇒ POSITIVE 支
    q = mech_quantities(*_synth_biased(+1))
    v = dir_verdict([q["ku"]] * 10)
    out["S5"] = {"ku": q["ku"], "verdict": v["verdict"],
                 "pass": bool(abs(q["ku"] - 0.6) < 1e-12
                              and v["verdict"] == "DIRECTION-CONFIRMED:USER-POSITIVE")}

    # ---------------- S7 KAPPA_MIN 分辨力（单元级：直接喂已知 ku）
    v_lo = dir_verdict([0.009] * 10, kappa_min=0.01)
    v_hi = dir_verdict([-0.011] * 10, kappa_min=0.01)
    out["S7"] = {"ku_0.009": {"n_neu": v_lo["n_neu"], "n_neg": v_lo["n_neg"],
                              "verdict": v_lo["verdict"]},
                 "ku_-0.011": {"n_neu": v_hi["n_neu"], "n_neg": v_hi["n_neg"],
                               "verdict": v_hi["verdict"]},
                 "pass": bool(v_lo["n_neu"] == 10 and v_lo["n_neg"] == 0
                              and v_lo["verdict"] == "DIRECTION-UNRESOLVED"
                              and v_hi["n_neu"] == 0 and v_hi["n_neg"] == 10
                              and v_hi["verdict"] == "DIRECTION-CONFIRMED:USER-NEGATIVE")}

    # ---------------- S8 阈值敏感性 3×3：判词不得翻转
    q = mech_quantities(*_synth_biased(-1))
    grid = {}
    flip = False
    base = None
    for km in KAPPA_GRID:
        for dd in DESC_GRID:
            vv = dir_verdict([q["ku"]] * 10, kappa_min=km)
            key = f"kappa={km}/desc={dd}"
            grid[key] = {"verdict": vv["verdict"], "n_neg": vv["n_neg"]}
            if base is None:
                base = vv["verdict"]
            elif vv["verdict"] != base:
                flip = True
    out["S8"] = {"grid": grid, "flip": bool(flip), "pass": bool(not flip)}

    # ---------------- S9 结构性对账：5 箱与 10 箱的 E_user/Mu **逐位相同**
    rec = {}
    for ep in (100, 105, 108):
        ni_e, nd_e = ep_counts(cells, "exact", ep)
        ni_a, nd_a = ep_counts(cells, "approx", ep)

        def _c5(m):
            m = np.asarray(m)
            return m.reshape(m.shape[:-1] + (5, 2)).sum(axis=-1)

        q10 = mech_quantities(ni_e, nd_e, ni_a, nd_a)
        q5 = mech_quantities(_c5(ni_e), _c5(nd_e), _c5(ni_a), _c5(nd_a))
        rec[f"ep{ep}"] = {"d_E_user": abs(q10["E_user"] - q5["E_user"]),
                          "d_Mu": abs(q10["Mu"] - q5["Mu"]),
                          "d_ku": abs(q10["ku"] - q5["ku"]),
                          "d_dr": abs(q10["dr"] - q5["dr"])}
    ok9 = all(v["d_E_user"] < 1e-12 and v["d_Mu"] < 1e-12 and v["d_ku"] < 1e-12
              and v["d_dr"] < 1e-12 for v in rec.values())
    out["S9"] = {"per_ep": rec, "pass": bool(ok9)}

    # ---------------- S10 与 R1.15 `decompose` 的三项逐位一致
    from .r1_15_delivery_gap import decompose
    dmax = 0.0
    for ep in EPS10:
        ni_e, nd_e = ep_counts(cells, "exact", ep)
        ni_a, nd_a = ep_counts(cells, "approx", ep)
        q = mech_quantities(ni_e, nd_e, ni_a, nd_a)
        d = decompose(ni_e, nd_e, ni_a, nd_a)
        dmax = max(dmax, abs(q["E_user"] - d["E_user"]), abs(q["E_frame"] - d["E_frame"]),
                   abs(q["E_within"] - d["E_within"]), abs(q["dr"] - d["dr"]))
    out["S10"] = {"max_abs_diff_vs_R115_decompose": dmax, "pass": bool(dmax < 1e-12)}

    out["all_pass"] = all(out[k]["pass"] for k in
                          ("S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10"))
    return out


# ============================================================ 主流程
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="CW-R1.17 尺度不变机制判据（零 episode）")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--ctrl-only", action="store_true", help="只跑判据自证，不写证据")
    args = ap.parse_args(argv)

    t_start = time.perf_counter()

    # ---- CTRL 先行
    t0 = time.perf_counter()
    ctrl = controls()
    ctrl_s = round(time.perf_counter() - t0, 2)
    print(f"[CTRL] all_pass={ctrl['all_pass']} " +
          " ".join(f"{k}={ctrl[k]['pass']}" for k in
                   ("S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10")) +
          f" ({ctrl_s}s)")
    if not ctrl["all_pass"]:
        print("[ABORT] 判据自证未过 ⇒ 按预注册 §5 不看面板")
        return 3
    if args.ctrl_only:
        print(json.dumps(ctrl, ensure_ascii=False, indent=1, default=float))
        return 0

    # ---- P0 全量闸
    t1 = time.perf_counter()
    gate = G.run_all()
    gate_s = round(time.perf_counter() - t1, 2)
    print(f"[GATE] n_checks={gate['n_checks']} n_fail={gate['n_fail']} ({gate_s}s)")

    head = _git(["rev-parse", "HEAD"])
    wt_sha = _git(["hash-object", "experiments/cw_v1/r1_17_scaleinv.py"])
    # ★ 运行时本文件通常**尚未提交** ⇒ 显式 allow_fail，拿 None 而不是假哈希（见 `_git` 的注释）
    blob_sha = _git(["rev-parse", f"{head}:experiments/cw_v1/r1_17_scaleinv.py"],
                    allow_fail=True)
    dirty = _git(["status", "--porcelain"])
    srchash = _src_hash()
    outdir = RESULTS / f"cw_r1_17_scaleinv_{_now_utc()}_{srchash}"
    outdir.mkdir(parents=True, exist_ok=False)
    print(f"[OUT] {outdir.name}")

    # ---- 读 R1.16 证据（只读）
    cells = json.loads(R116_CELLS.read_text(encoding="utf-8"))
    if len(cells) != 80:
        print(f"[ABORT] R1.16 cells 应为 80 格，实为 {len(cells)}")
        return 4

    # ---- 重算 10 个 ep
    per_ep = {}
    for ep in EPS10:
        ni_e, nd_e = ep_counts(cells, "exact", ep)
        ni_a, nd_a = ep_counts(cells, "approx", ep)
        q = mech_quantities(ni_e, nd_e, ni_a, nd_a)
        q["ep"] = ep
        q["abs_delta_p_bad"] = None  # 稍后从 R1.16 summary 取
        per_ep[f"ep{ep}"] = q
        print(f"[EP] {ep} dr={q['dr']:+.5f} E_u={q['E_user']:+.5f} Mu={q['Mu']:.4f} "
              f"ku={q['ku']:+.4f} E_f={q['E_frame']:+.5f} Mf={q['Mf']:.4f} kf={q['kf']:+.4f}")

    # ---- Q_dir
    ku_all = [v["ku"] for v in per_ep.values()]
    kf_all = [v["kf"] for v in per_ep.values()]
    vu = dir_verdict(ku_all, KAPPA_MIN, "user")
    vf = dir_verdict(kf_all, KAPPA_MIN, "frame")
    v_dir = combine_dir(vu, vf)
    print(f"[Q_dir] user: {vu['verdict']} ({vu['n_neg']}neg/{vu['n_pos']}pos/{vu['n_neu']}neu, "
          f"p50={vu['p_binom_null50']:.4g}) | frame: {vf['verdict']} "
          f"({vf['n_neg']}neg/{vf['n_pos']}pos/{vf['n_neu']}neu) ⇒ {v_dir}")

    # ---- Q_mag
    def _mag(vals):
        v = np.array([x for x in vals if np.isfinite(x)], dtype=float)
        if v.size == 0:
            return {"n": 0}
        b = boot_ci(v)
        return {"n": int(v.size), "mean": float(v.mean()), "median": float(np.median(v)),
                "ci_lo": b["ci_lo"], "ci_hi": b["ci_hi"],
                "min": float(v.min()), "max": float(v.max()),
                "n_negative": int((v < 0).sum()), "n_positive": int((v > 0).sum())}

    mag = {"ku": _mag(ku_all), "kf": _mag(kf_all)}

    # ---- Q_share（描述，不判）
    drs = [abs(v["dr"]) for v in per_ep.values()]
    # ★ 口径修正（见预注册 §8-4）：R1.16 记的 `−0.7333` 是 `max` 在**三项**（含 within）上取；
    #   只在 `user+frame` 两项上取是 `−0.6121`。两者都对，但对不同对象 ⇒ **两个都报**。
    maxshare3 = [max(abs(v["shares"][t]) for t in ("user", "frame", "within"))
                 for v in per_ep.values()]
    maxshare2 = [max(abs(v["shares"]["user"]), abs(v["shares"]["frame"]))
                 for v in per_ep.values()]
    sub = [k for k, v in per_ep.items() if abs(v["dr"]) >= DR_DESC]
    share_desc = {
        "subset_def": f"|dr| >= {DR_DESC}（**描述**门槛，不参与判词）",
        "subset_eps": sub, "n_subset": len(sub),
        "spearman_absdr_maxshare_all10": float(spearman(drs, maxshare2)),
        "spearman_absdr_maxshare3_all10": float(spearman(drs, maxshare3)),
        "spearman_absdr_maxshare_subset": (
            float(spearman([abs(per_ep[k]["dr"]) for k in sub],
                           [max(abs(per_ep[k]["shares"]["user"]),
                                abs(per_ep[k]["shares"]["frame"])) for k in sub]))
            if len(sub) >= 3 else float("nan")),
        "spearman_absdr_abs_ku_all10": float(spearman(drs, [abs(x) for x in ku_all])),
        "maxshare_max_ep": max({k: maxshare2[i] for i, k in enumerate(per_ep)}.items(),
                               key=lambda kv: kv[1])[0],
        "per_ep_shares": {k: v["shares"] for k, v in per_ep.items()},
        "note_max_scope": ("`maxshare_all10` = max(|sh_user|,|sh_frame|)（user+frame 两项，"
                           "= R1.16 判词实际用的那两项）；`maxshare3` 另含 |sh_within|，"
                           "R1.16 交付里的 −0.7333 是这个口径。两个都报，避免与 R1.16 对不上。"),
    }
    print(f"[Q_share] 描述子集 n={len(sub)}/{len(per_ep)}；"
          f"Sp(|dr|,max|share|) 两项={share_desc['spearman_absdr_maxshare_all10']:+.4f} "
          f"三项={share_desc['spearman_absdr_maxshare3_all10']:+.4f} "
          f"子集={share_desc['spearman_absdr_maxshare_subset']:+.4f}；"
          f"Sp(|dr|,|ku|)={share_desc['spearman_absdr_abs_ku_all10']:+.4f}")

    # ---- M6' 预测因子（|Δp_bad| → ku，对称汇报旧因变量）
    r116 = json.loads((R116_DIR / "summary.json").read_text(encoding="utf-8"))
    apb = {f"ep{r['ep']}": r["abs_delta_p_bad"] for r in r116["per_ep"]}
    pts = []
    for k, v in per_ep.items():
        if apb.get(k) is None or not np.isfinite(v["ku"]):
            continue
        pts.append({"ep": v["ep"], "abs_delta_p_bad": float(apb[k]),
                    "ku": float(v["ku"]), "sh_user": float(v["shares"]["user"])})
    if len(pts) >= 3:
        xs = [p["abs_delta_p_bad"] for p in pts]
        rho_new = float(spearman(xs, [p["ku"] for p in pts]))
        p_new = perm_p(xs, [p["ku"] for p in pts], rho_new, n=PERM_N, seed=SEED_STAT)
        rho_old = float(spearman(xs, [p["sh_user"] for p in pts]))
        p_old = perm_p(xs, [p["sh_user"] for p in pts], rho_old, n=PERM_N, seed=SEED_STAT)
        rho_shift = float(spearman(xs, np.roll([p["ku"] for p in pts], 1)))
    else:
        rho_new = p_new = rho_old = p_old = rho_shift = float("nan")
    pred = verdict_predictor(rho_new, p_new, len(pts))
    # ★ 必报的混淆诊断（预注册 §4.4 精神：一个漂亮的 ρ 必须先被拆开）：
    #   `ku = E_user/Mu` ⇒ 检查 `|Δp_bad|` 是与**分子**相关、与**分母**相关、还是与 `Δr` 相关。
    #   若 `|Δp_bad|` 主要与 `|E_user|` 相关，则 `ku` 的预测力只是"分子大"的镜像。
    diag = {}
    if len(pts) >= 3:
        xs = [p["abs_delta_p_bad"] for p in pts]
        diag = {
            "rho_absdp_vs_abs_Euser": float(spearman(xs, [abs(per_ep[f"ep{p['ep']}"]["E_user"])
                                                       for p in pts])),
            "rho_absdp_vs_Mu": float(spearman(xs, [per_ep[f"ep{p['ep']}"]["Mu"] for p in pts])),
            "rho_absdp_vs_abs_dr": float(spearman(xs, [abs(per_ep[f"ep{p['ep']}"]["dr"])
                                                    for p in pts])),
            "rho_absdp_vs_kf": float(spearman(xs, [per_ep[f"ep{p['ep']}"]["kf"] for p in pts])),
            "note": ("`ku` 与 `|Δp_bad|` 共享『用户间信道差异』这一信息源 ⇒ ρ 大**部分**是结构性的；"
                     "上面四项用来判断主要走分子还是分母。"),
        }
        print(f"[M6'-diag] ρ(|Δp|,|E_u|)={diag['rho_absdp_vs_abs_Euser']:+.4f} "
              f"ρ(|Δp|,Mu)={diag['rho_absdp_vs_Mu']:+.4f} "
              f"ρ(|Δp|,|dr|)={diag['rho_absdp_vs_abs_dr']:+.4f} "
              f"ρ(|Δp|,kf)={diag['rho_absdp_vs_kf']:+.4f}")
    pred.update({"y_new": "ku", "y_old": "sh_user", "rho_old_y": rho_old, "p_old_y": p_old,
                 "rho_shifted_true": rho_shift, "points": pts, "confound_diag": diag,
                 "no_info_flag": bool(np.isfinite(rho_shift) and np.isfinite(rho_new)
                                      and abs(rho_shift) >= abs(rho_new))})
    print(f"[M6'] n={len(pts)} ρ(ku)={rho_new:+.4f} p={p_new:.4f} | "
          f"ρ(sh_user)={rho_old:+.4f} p={p_old:.4f} | ρ_shift={rho_shift:+.4f} "
          f"⇒ {pred['verdict']}")

    # ---- 阈值敏感性（真实数据，不是构造）
    sens = {}
    for km in KAPPA_GRID:
        vu_k = dir_verdict(ku_all, km, "user")
        vf_k = dir_verdict(kf_all, km, "frame")
        sens[f"kappa={km}"] = {"user": vu_k["verdict"], "frame": vf_k["verdict"],
                               "combined": combine_dir(vu_k, vf_k),
                               "n_neg_u": vu_k["n_neg"], "n_neg_f": vf_k["n_neg"]}
    print("[SENS] " + json.dumps({k: v["combined"] for k, v in sens.items()},
                                 ensure_ascii=False))

    verdicts = {
        "Q_dir": {"combined": v_dir, "user": vu, "frame": vf},
        "Q_mag": mag,
        "Q_share": share_desc,
        "M6p": pred,
        "sens_kappa": sens,
        "mu_min": MU_MIN,
        "mu_min_effective": bool(min(v["Mu"] for v in per_ep.values()) < MU_MIN
                                 or min(v["Mf"] for v in per_ep.values()) < MU_MIN),
        "note_q_dir_vs_q_share": (
            "Q_dir 与 Q_share 答不同问题：前者'搬运方向是否系统性'，后者'解释了多少净缺口'。"
            "两者可以给出不同判词，必须同报。"),
    }

    n_fail = gate["n_fail"] + (0 if ctrl["all_pass"] else 1)
    (outdir / "analysis.json").write_text(
        json.dumps(per_ep, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    (outdir / "verdicts.json").write_text(
        json.dumps(verdicts, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    (outdir / "ctrl.json").write_text(
        json.dumps(ctrl, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    (outdir / "gates.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=1), encoding="utf-8")

    try:
        import psutil  # type: ignore
        rss = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
    except Exception:
        rss = None
    meta = {"command": "python -m experiments.cw_v1.r1_17_scaleinv --tag " + str(args.tag),
            "timestamp_local": datetime.now().astimezone().isoformat(),
            "python": os.sys.version.split()[0], "numpy": np.__version__,
            "head": head, "src_hash": srchash,
            "worktree_sha_r1_17": wt_sha, "blob_sha_r1_17": blob_sha,
            "blob_sha_note": ("运行时该文件尚未提交 ⇒ blob_sha 记 None（不再记参数串）；"
                              "提交后的真实 blob 见 _POST_COMMIT.json"
                              if blob_sha is None else "运行前该文件已在 HEAD 上"),
            "git_dirty_before_run": dirty.splitlines(),
            "input_cells": str(R116_CELLS), "n_cells_in": len(cells),
            "n_run_episode": 0,
            "ctrl_elapsed_s": ctrl_s, "gate_elapsed_s": gate_s,
            "total_elapsed_s": round(time.perf_counter() - t_start, 2),
            "peak_rss_mb": round(rss, 1) if rss else None,
            "n_checks_gates": gate["n_checks"], "n_fail_gates": gate["n_fail"]}
    (outdir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"[DONE] {outdir.name} n_fail={n_fail} total={meta['total_elapsed_s']}s "
          f"peak_rss={meta['peak_rss_mb']}MB")
    print(json.dumps({"Q_dir": v_dir, "M6p": pred["verdict"],
                      "sens": {k: v["combined"] for k, v in sens.items()}},
                     ensure_ascii=False, indent=1))
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    import sys
    sys.exit(main())
