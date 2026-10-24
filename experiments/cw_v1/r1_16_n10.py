# -*- coding: utf-8 -*-
"""CW-R1.16 把「有效独立单元」从 2 提到 10 —— 送达率缺口的稳定性 + 机制分岔的预测因子。

预注册：`docs/research/CW-R1.16-N10独立单元-预注册-v0.1.md`（`run_episode`=0 时写）

五段
----
`P0` **前置闸**：全量 `gates.py` `n_fail=0`。
`CTRL` **判据自证**（零 episode、**看面板之前**）：`S1` bootstrap / `S2` Spearman / `S3` 置换 p /
    `S4` 机制计数 / `S5` 活体负控（错位配对必须掉下来）。判据没分辨力就不许看面板。
`PANEL` **80 格**：`arm ∈ {approx, exact} × ep ∈ {100..109} × seed ∈ {2001..2004}`。
    其中 `ep ∈ {100,101}` 的 16 格与 R1.14 同格同配置 ⇒ 身份闸（`N4`：不逐位相同则本轮作废）。
`M` **主量/机制**：`M1` 逐 ep 送达率差 → `M4` bootstrap CI；`M5` 复用 R1.15 的 `decompose/classify` 数机制计数。
`P` **预测因子**：`M6` `Spearman(|Δp_bad|, sh_user)` + 置换 p；`S5` 的另一半（真实配对）在此。

★ 被测物**一字节不动**；本文件只做"跑格 + 统计"，判据函数一律 `import` R1.15（同一份代码）。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import gates as G
from . import r1_15_delivery_gap as R15
from .r1_14_exactbase import (ARMS, H_PLAN, M_MAIN, RATIO, REPO, RESULTS, RHO, SEEDS,
                              TAIL_DP, _Rec, _cfg, _git, _src_hash)
from .runner import check_identities, run_episode

#: 10 个 episode（`ep100/101` = R1.14/R1.15 的旧格，`ep102…109` 全新）
EPS10 = tuple(range(100, 110))
#: 与 R1.14 有参照物的那两集（身份闸只对它们有意义）
EPS_REF = (100, 101)

#: 预注册 §4.2 / §4.5
BOOT_N = 10_000
PERM_N = 100_000
SEED_STAT = 20261024
CI = (2.5, 97.5)
#: 预注册 §4.4 / §4.5 的门槛（与 R1.15 同值，一字不改）
DR_MIN = R15.DR_MIN
RHO_SUP = 0.60
RHO_REJ = 0.40
MIN_POINTS = 6


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


# ==================================================================== 统计工具
def spearman(x, y) -> float:
    """Spearman ρ（**并列取平均秩**，无 scipy 依赖）。任一列完全常数 ⇒ NaN。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size != y.size or x.size < 2:
        return float("nan")

    def _rank(v):
        order = np.argsort(v, kind="mergesort")
        r = np.empty(v.size, dtype=float)
        r[order] = np.arange(v.size, dtype=float)
        # 并列：同一值的秩取平均
        sv = v[order]
        i = 0
        while i < sv.size:
            j = i
            while j + 1 < sv.size and sv[j + 1] == sv[i]:
                j += 1
            if j > i:
                r[order[i:j + 1]] = (i + j) / 2.0
            i = j + 1
        return r

    rx, ry = _rank(x), _rank(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    if den == 0:
        return float("nan")
    return float((rx * ry).sum() / den)


def perm_p(x, y, rho_obs: float, n: int = PERM_N, seed: int = SEED_STAT) -> float:
    """双侧置换 p：打乱 `y` 重算 `ρ`，数 `|ρ| ≥ |ρ_obs|` 的比例（含 +1 平滑）。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n):
        if abs(spearman(x, rng.permutation(y))) >= abs(rho_obs) - 1e-15:
            cnt += 1
    return float((cnt + 1) / (n + 1))


def boot_ci(vals, n: int = BOOT_N, seed: int = SEED_STAT) -> dict:
    """配对差 bootstrap（对 **episode** 有放回重采样，不重采样 seed）。"""
    v = np.asarray(vals, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n, v.size))
    means = v[idx].mean(axis=1)
    lo, hi = np.percentile(means, CI)
    return {"n_ep": int(v.size), "n_boot": int(n), "mean": float(v.mean()),
            "median": float(np.median(v)), "ci_lo": float(lo), "ci_hi": float(hi),
            "frac_boot_lt0": float((means < 0).mean()),
            "min": float(v.min()), "max": float(v.max())}


def verdict_delivery(b: dict) -> str:
    if b["ci_hi"] < 0:
        return "DELIVERY-GAP-CONFIRMED-AT-N10"
    if b["ci_lo"] > 0:
        return "DELIVERY-GAP-REVERSED-AT-N10"
    return "UNRESOLVED-AT-N10"


def verdict_mech(counts: dict) -> dict:
    ha = counts.get("H-A-FRAME-SELECTION", 0)
    hb = counts.get("H-B-USER-ALLOCATION", 0)
    sm = counts.get("GAP-TOO-SMALL", 0)
    un = counts.get("DELIVERY-GAP-UNEXPLAINED", 0)
    n = sum(counts.values())
    if ha >= 3 and hb >= 3:
        v = "MIXED-MECHANISM-AT-N10"
    elif ha >= 8:
        v = "MECHANISM-DOMINANT:H-A-FRAME-SELECTION"
    elif hb >= 8:
        v = "MECHANISM-DOMINANT:H-B-USER-ALLOCATION"
    elif sm + un >= 4 and ha < 3 and hb < 3:
        v = "MECHANISM-UNRESOLVED-AT-N10"
    else:
        v = "MIXED-WEAK"
    return {"verdict": v, "n_ep": n, "counts": counts}


def verdict_predictor(rho: float, p: float, n_pts: int) -> dict:
    if n_pts < MIN_POINTS:
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


# ==================================================================== CTRL（零 episode）
def controls() -> dict:
    """预注册 §7 的 `S1–S5`，答案**手算/构造已知**。"""
    out: dict = {}

    # ---- S1 bootstrap
    s1 = {}
    v_pos = [0.10] * 10
    b = boot_ci(v_pos)
    s1["degenerate_pos"] = {"ci": [b["ci_lo"], b["ci_hi"]], "pass":
                            bool(b["ci_lo"] > 0 and abs(b["ci_hi"] - b["ci_lo"]) < 1e-12)}
    v_alt = [0.10, -0.10] * 5
    b = boot_ci(v_alt)
    s1["alternating"] = {"ci": [b["ci_lo"], b["ci_hi"]], "verdict": verdict_delivery(b),
                         "pass": verdict_delivery(b) == "UNRESOLVED-AT-N10"}
    b = boot_ci([0.0] * 10)
    s1["all_zero"] = {"ci": [b["ci_lo"], b["ci_hi"]], "verdict": verdict_delivery(b),
                      "pass": bool(verdict_delivery(b) == "UNRESOLVED-AT-N10"
                                   and b["ci_lo"] == 0.0 and b["ci_hi"] == 0.0)}
    # 单侧全负（与 R1.15 观察到的量级同阶）：CI 必须判 CONFIRMED
    v_neg = [-0.12, -0.14, -0.09, -0.13, -0.11, -0.15, -0.10, -0.12, -0.16, -0.08]
    b = boot_ci(v_neg)
    s1["all_negative"] = {"ci": [b["ci_lo"], b["ci_hi"]], "verdict": verdict_delivery(b),
                          "pass": verdict_delivery(b) == "DELIVERY-GAP-CONFIRMED-AT-N10"}
    out["S1"] = s1
    out["S1_pass"] = all(v["pass"] for v in s1.values())

    # ---- S2 Spearman
    s2 = {}
    x = list(range(10))
    s2["monotone"] = {"rho": spearman(x, [2 * v + 1 for v in x]), "pass":
                      abs(spearman(x, [2 * v + 1 for v in x]) - 1.0) < 1e-12}
    s2["anti"] = {"rho": spearman(x, [-3 * v for v in x]), "pass":
                  abs(spearman(x, [-3 * v for v in x]) + 1.0) < 1e-12}
    # 含并列：x 有并列、y 单调增 ⇒ 手算 ρ = 80/√(80·82.5) = 0.9847319…
    #   （rx 偏差 = (−4,−4,−2,−2,0,0,2,2,4,4)；ry 偏差 = (−4.5…4.5)；Σrx·ry=80、Σry²=82.5）
    #   ★ 预注册 §8-1 修订：首版写 0.9877 是**我手算错**，实为 0.98473（判据自伤，非被测物错）
    xt = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    yt = list(range(10))
    rho_t = spearman(xt, yt)
    s2["ties"] = {"rho": rho_t, "expect": 0.9847319, "pass": abs(rho_t - 0.9847319) < 1e-3}
    # 打乱对照（**显式**置换，不用 rng ⇒ 自证必须可复现）：|ρ| 必须掉下来
    perm = [3, 7, 0, 9, 2, 5, 8, 1, 4, 6]
    rho_sh = spearman(x, [x[i] for i in perm])
    s2["shuffled"] = {"rho": rho_sh, "perm": perm, "pass": abs(rho_sh) < 0.9}
    # 常数列 ⇒ NaN（不许静默当 0）
    s2["constant_nan"] = {"rho": spearman(x, [1.0] * 10),
                          "pass": bool(np.isnan(spearman(x, [1.0] * 10)))}
    out["S2"] = s2
    out["S2_pass"] = all(v["pass"] for v in s2.values())

    # ---- S3 置换 p（只跑 2000 次控时；正式面板用 10^5）
    y_alt = [1, -1, 1, -1, 1, -1, 1, -1, 1, -1]
    p1 = perm_p(x, x, 1.0, n=2000)
    rho_alt = spearman(x, y_alt)
    p0 = perm_p(x, y_alt, rho_alt, n=2000)
    out["S3"] = {"p_perfect": p1, "rho_null": rho_alt, "p_null": p0,
                 "pass": bool(p1 < 0.01 and p0 > 0.4)}
    out["S3_pass"] = out["S3"]["pass"]

    # ---- S4 机制计数
    s4 = {}
    mix = {"H-A-FRAME-SELECTION": 5, "H-B-USER-ALLOCATION": 5}
    s4["mixed_5_5"] = {"got": verdict_mech(mix)["verdict"],
                       "pass": verdict_mech(mix)["verdict"] == "MIXED-MECHANISM-AT-N10"}
    dom = {"H-A-FRAME-SELECTION": 10}
    s4["dominant_10"] = {"got": verdict_mech(dom)["verdict"],
                         "pass": verdict_mech(dom)["verdict"]
                         == "MECHANISM-DOMINANT:H-A-FRAME-SELECTION"}
    unres = {"GAP-TOO-SMALL": 2, "DELIVERY-GAP-UNEXPLAINED": 3,
             "H-A-FRAME-SELECTION": 3, "H-B-USER-ALLOCATION": 2}
    s4["unresolved"] = {"got": verdict_mech(unres)["verdict"],
                        "pass": verdict_mech(unres)["verdict"] == "MIXED-WEAK"}
    out["S4"] = s4
    out["S4_pass"] = all(v["pass"] for v in s4.values())

    # ---- S5 活体负控（★ 预注册 §8-2 修订：原写"循环移位必须 <0.4"—— 对平滑单调序列移位
    #        本身就是 ρ=−0.60，那个阈值是拍的；改为三条**方向明确**的断言）
    xa = np.array([0.26, 0.43, 0.10, 0.55, 0.31, 0.08, 0.49, 0.22, 0.37, 0.60])
    ya = np.array([0.20, 0.92, 0.15, 0.80, 0.35, 0.10, 0.70, 0.25, 0.55, 0.85])
    rho_ok = spearman(xa, ya)
    rho_rev = spearman(xa, -ya)
    rho_shift = spearman(xa, np.roll(ya, 1))
    out["S5"] = {"rho_aligned": rho_ok, "rho_reversed": rho_rev, "rho_rolled": rho_shift,
                 "pass": bool(rho_ok >= 0.9 and rho_rev <= -0.9 and rho_shift < 0.4)}
    out["S5_pass"] = out["S5"]["pass"]

    out["all_pass"] = bool(out["S1_pass"] and out["S2_pass"] and out["S3_pass"]
                           and out["S4_pass"] and out["S5_pass"])
    return out


# ==================================================================== 主流程
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="CW-R1.16 N=10 独立单元（预注册 v0.1）")
    ap.add_argument("--probe", action="store_true",
                    help="只跑 3 格报**耗时/DP 构建数**（不打印任何指标、不写证据）")
    ap.add_argument("--tag", default="main")
    args = ap.parse_args(argv)

    import experiments.cw_v1.plan_dp as PD

    if args.probe:
        t0 = time.perf_counter()
        n0 = PD.N_BUILDS
        R15.run_cell(102, 2001, "approx")
        t1 = time.perf_counter()
        R15.run_cell(102, 2001, "exact")
        t2 = time.perf_counter()
        R15.run_cell(105, 2001, "approx")
        t3 = time.perf_counter()
        print(json.dumps({
            "ep102/approx_s": round(t1 - t0, 1),
            "ep102/exact_s": round(t2 - t1, 1),
            "ep105/approx_s": round(t3 - t2, 1),
            "n_dp_builds_total": int(PD.N_BUILDS - n0),
            "mean_3cells_s": round((t3 - t0) / 3, 2),
            "est_80cells_s": round((t3 - t0) / 3 * 80, 1),
            "cache_stats": PD.cache_stats()["cap"],
        }, ensure_ascii=False, indent=1))
        return 0

    # ---- CTRL 先行（零 episode）
    t_c0 = time.perf_counter()
    ctrl = controls()
    ctrl_s = round(time.perf_counter() - t_c0, 1)
    print(f"[CTRL] all_pass={ctrl['all_pass']} "
          f"S1={ctrl['S1_pass']} S2={ctrl['S2_pass']} S3={ctrl['S3_pass']} "
          f"S4={ctrl['S4_pass']} S5={ctrl['S5_pass']} ({ctrl_s}s)")
    if not ctrl["all_pass"]:
        print("[ABORT] 判据自证未过 ⇒ 按预注册 §7 不看面板")
        return 3

    t_start = time.perf_counter()
    head = _git(["rev-parse", "HEAD"])
    wt_sha = _git(["hash-object", "experiments/cw_v1/r1_16_n10.py"])
    blob_sha = _git(["rev-parse", f"{head}:experiments/cw_v1/r1_16_n10.py"])
    dirty = _git(["status", "--porcelain"])
    srchash = _src_hash()
    outdir = RESULTS / f"cw_r1_16_n10_{_now_utc()}_{srchash}"
    outdir.mkdir(parents=True, exist_ok=False)
    print(f"[OUT] {outdir}")

    # ---- P0 全量闸
    t1 = time.perf_counter()
    gate = G.run_all()
    gate_s = round(time.perf_counter() - t1, 2)
    (outdir / "gates.json").write_text(json.dumps(gate, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
    print(f"[GATE] n_checks={gate['n_checks']} n_fail={gate['n_fail']} ({gate_s}s)")

    # ---- PANEL 80 格
    n_builds0 = PD.N_BUILDS
    cells: list[dict] = []
    for arm in ARMS:
        for ep in EPS10:
            for seed in SEEDS:
                c = R15.run_cell(ep, seed, arm)
                cells.append(c)
                print(f"[CELL] {arm:>6}/ep{ep}/s{seed} att={c['n_attempted_by_user']} "
                      f"del={c['n_delivered_by_user']} p̄={c['attempt_p_true_mean']:.4f} "
                      f"({c['elapsed_s']}s)")
                if not c["identities_pass"]:
                    print(f"  [WARN] 恒等失败: {c['identities_fail']}")
    panel_s = round(time.perf_counter() - t1, 1)
    print(f"[PANEL] {len(cells)} 格 {panel_s}s dp_builds={PD.N_BUILDS - n_builds0}")

    # ---- N4 身份闸（只对 ep100/101 的 16 格有意义）
    ref_cells = [c for c in cells if c["ep"] in EPS_REF]
    ident = R15.identity_check(ref_cells)
    (outdir / "identity.json").write_text(json.dumps(ident, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    print(f"[ID] {ident['n_identical']}/{ident['n_cells']} 与 R1.14 逐位相同")

    # ---- 重复性（ep105/seed2001/approx 连跑两次）
    rep = {}
    a = next(c for c in cells if c["ep"] == 105 and c["seed"] == 2001 and c["arm"] == "approx")
    b2 = R15.run_cell(105, 2001, "approx")
    for f in ("n_attempted_by_user", "n_delivered_by_user", "cost_true_per_step",
              "policy_trace_hash", "aoi_tavg_by_user"):
        rep[f] = bool(a[f] == b2[f])
    rep["all_identical"] = all(rep.values())

    # ---- M1 逐 ep 逐 seed 送达率差
    per_ep = []
    for ep in EPS10:
        seeds = []
        for s in SEEDS:
            e = next(c for c in cells if c["ep"] == ep and c["seed"] == s and c["arm"] == "exact")
            p = next(c for c in cells if c["ep"] == ep and c["seed"] == s and c["arm"] == "approx")
            ne = float(sum(map(sum, e["n_attempt_in_bin_by_user"])))
            na = float(sum(map(sum, p["n_attempt_in_bin_by_user"])))
            de = float(sum(map(sum, e["n_delivered_in_bin_by_user"])))
            da = float(sum(map(sum, p["n_delivered_in_bin_by_user"])))
            seeds.append({"seed": s, "attempt_exact": ne, "attempt_approx": na,
                          "deliver_exact": de, "deliver_approx": da,
                          "rate_exact": de / ne if ne else float("nan"),
                          "rate_approx": da / na if na else float("nan"),
                          "dr": (de / ne - da / na) if (ne and na) else float("nan")})
        drs = np.array([x["dr"] for x in seeds], dtype=float)

        def _cell(ep_: int, s_: int, arm_: str) -> dict:
            return next(x for x in cells
                        if x["ep"] == ep_ and x["seed"] == s_ and x["arm"] == arm_)

        def _med_idx(field: str, arm_: str, i: int) -> float:
            return float(np.median([float(_cell(ep, s, arm_)[field][i]) for s in SEEDS]))

        def _med_scalar(field: str, arm_: str) -> float:
            return float(np.median([float(_cell(ep, s, arm_)[field]) for s in SEEDS]))

        row = {
            "ep": int(ep), "per_seed": seeds,
            "dr_ep": float(drs.mean()), "dr_ep_median": float(np.median(drs)),
            "dr_seed_min": float(drs.min()), "dr_seed_max": float(drs.max()),
            "n_seeds_negative": int((drs < 0).sum()),
            "aoi_tavg_by_user": {"exact": [_med_idx("aoi_tavg_by_user", "exact", i) for i in (0, 1)],
                                 "approx": [_med_idx("aoi_tavg_by_user", "approx", i) for i in (0, 1)]},
            "attempt_by_user": {"exact": [_med_idx("n_attempted_by_user", "exact", i) for i in (0, 1)],
                                "approx": [_med_idx("n_attempted_by_user", "approx", i) for i in (0, 1)]},
            "deliver_by_user": {"exact": [_med_idx("n_delivered_by_user", "exact", i) for i in (0, 1)],
                                "approx": [_med_idx("n_delivered_by_user", "approx", i) for i in (0, 1)]},
            "n_probe_median": {"exact": _med_scalar("n_probe", "exact"),
                               "approx": _med_scalar("n_probe", "approx")},
            "cost_true": {"exact": _med_scalar("cost_true_per_step", "exact"),
                          "approx": _med_scalar("cost_true_per_step", "approx")},
            "cost_est": {"exact": _med_scalar("cost_est_per_step", "exact"),
                         "approx": _med_scalar("cost_est_per_step", "approx")},
            "p_true_mean_by_user_approx": [
                float(np.mean([_cell(ep, s, "approx")["attempt_p_true_mean_by_user"][i]
                               for s in SEEDS])) for i in (0, 1)],
        }
        # `n_never_served` 不在 R1.15.run_cell 的导出字段里 ⇒ 用**代理**并如实标注：
        #   4-seed 中位送达数为 0 的用户数（不是 runner 的逐格 `n_never_served`）
        row["n_never_served"] = {
            "note": "代理：逐用户 4-seed 中位送达数 == 0 的用户数（run_cell 未导出该字段）",
            "exact": int(sum(1 for i in (0, 1) if row["deliver_by_user"]["exact"][i] == 0)),
            "approx": int(sum(1 for i in (0, 1) if row["deliver_by_user"]["approx"][i] == 0))}
        # 预测因子候选：approx 臂 p̄_u0 − p̄_u1
        p0, p1 = row["p_true_mean_by_user_approx"]
        row["delta_p_bad"] = float(p0 - p1)
        row["abs_delta_p_bad"] = float(abs(p0 - p1))
        per_ep.append(row)

    dr_eps = [r["dr_ep"] for r in per_ep]
    boot = boot_ci(dr_eps)
    v_del = verdict_delivery(boot)
    print(f"[M1] mean Δr = {boot['mean']:+.4f} CI [{boot['ci_lo']:+.4f}, {boot['ci_hi']:+.4f}] "
          f"⇒ {v_del}")

    # ---- M5 机制（两档分辨率）
    dec = {}
    for res_name, fn in (("10bins", lambda m: np.asarray(m)), ("5bins", R15.coarsen)):
        per = {}
        for ep in EPS10:
            ni_e, nd_e = R15.ep_counts(cells, "exact", ep)
            ni_a, nd_a = R15.ep_counts(cells, "approx", ep)
            d = R15.decompose(fn(ni_e), fn(nd_e), fn(ni_a), fn(nd_a))
            d["cls"] = R15.classify(d)
            per[f"ep{ep}"] = d
        counts = {}
        for d in per.values():
            m = d["cls"].get("mechanism")
            counts[m] = counts.get(m, 0) + 1
        vm = verdict_mech(counts)
        dec[res_name] = {"per_ep": {k: {"cls": v["cls"],
                                        "shares": v.get("shares"),
                                        "dr": v.get("dr"),
                                        "residual": v.get("additivity_residual")}
                                    for k, v in per.items()},
                         "counts": counts, "verdict": vm}
        print(f"[M5/{res_name}] {vm['verdict']} counts={counts}")

    # ---- M6 预测因子（10 bins 主口径）
    pts = []
    for r, (k, d) in zip(per_ep, dec["10bins"]["per_ep"].items()):
        cls = d["cls"]
        if cls.get("status") != "OK" or abs(r["dr_ep"]) < DR_MIN:
            continue
        sh = cls["shares"]["user"]
        if sh is None or not np.isfinite(sh):
            continue
        pts.append({"ep": r["ep"], "dr_ep": r["dr_ep"], "sh_user": float(sh),
                    "abs_delta_p_bad": r["abs_delta_p_bad"]})
    if len(pts) >= 2:
        xs = [p["abs_delta_p_bad"] for p in pts]
        ys = [p["sh_user"] for p in pts]
        rho = spearman(xs, ys)
        p_perm = perm_p(xs, ys, rho)
    else:
        rho, p_perm = float("nan"), float("nan")
    pred = verdict_predictor(rho, p_perm, len(pts))
    # S5 的真实配对版本：错位一位（只在 n≥3 时有意义）
    rho_shift = float("nan")
    if len(pts) >= 3:
        rho_shift = spearman([p["abs_delta_p_bad"] for p in pts],
                             np.roll([p["sh_user"] for p in pts], 1))
    pred.update({"points": pts, "rho_shifted_true": rho_shift,
                 "pooling": "p̄_ui = mean_{4 seeds}; Δp_bad = p̄_u0 − p̄_u1 (approx 臂)"})
    print(f"[M6] n_pts={len(pts)} ρ={rho:+.4f} p_perm={p_perm:.4f} ρ_shift={rho_shift:+.4f} "
          f"⇒ {pred['verdict']}")

    # ---- 汇总
    n_fail = (gate["n_fail"] + sum(1 for c in cells if not c["identities_pass"])
              + (0 if ident["all_identical"] else 1) + (0 if ctrl["all_pass"] else 1)
              + (0 if rep["all_identical"] else 1))
    summary = {
        "protocol": "CW-POMDP-v1.1", "stage": "CW-R1.16", "kind": "N10-INDEPENDENT-UNITS",
        "prereg": "docs/research/CW-R1.16-N10独立单元-预注册-v0.1.md",
        "eps": list(EPS10), "seeds": list(SEEDS), "arms": list(ARMS),
        "n_cells": len(cells),
        "verdict": {"delivery": v_del,
                    "mechanism_10bins": dec["10bins"]["verdict"]["verdict"],
                    "mechanism_5bins": dec["5bins"]["verdict"]["verdict"],
                    "predictor": pred["verdict"]},
        "M1_boot": boot,
        "M4_verdict": v_del,
        "M5": {r: {"counts": dec[r]["counts"], "verdict": dec[r]["verdict"]} for r in dec},
        "M6": pred,
        "per_ep": per_ep,
        "identity": {"all_identical": bool(ident["all_identical"]),
                     "detail": f"{ident['n_identical']}/{ident['n_cells']}",
                     "scope": "仅 ep100/101 的 16 格（其余 64 格无参照物）"},
        "repeatability": rep,
        "CTRL": ctrl,
        "gates": {"n_checks": gate["n_checks"], "n_fail": gate["n_fail"]},
        "head": head, "src_hash": srchash,
        "n10_note": ("有效独立单元 = 10 个 episode；seed 只用于配对，不当样本。"
                     "`Δp_bad` 内生于 approx 臂的 SEND 选择，不是外生信道参数。"),
    }
    (outdir / "cells.json").write_text(json.dumps(cells, ensure_ascii=False, indent=1,
                                                  default=float), encoding="utf-8")
    (outdir / "decomp.json").write_text(json.dumps(dec, ensure_ascii=False, indent=1,
                                                   default=float), encoding="utf-8")
    (outdir / "predictor.json").write_text(json.dumps(pred, ensure_ascii=False, indent=1,
                                                      default=float), encoding="utf-8")
    (outdir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1,
                                                    default=float), encoding="utf-8")
    try:
        import psutil  # type: ignore
        rss = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
    except Exception:
        rss = None
    meta = {"command": "python -m experiments.cw_v1.r1_16_n10 --tag " + str(args.tag),
            "python": os.sys.version.split()[0], "numpy": np.__version__,
            "timestamp_local": datetime.now().astimezone().isoformat(),
            "head": head, "src_hash": srchash,
            "worktree_sha_r1_16": wt_sha, "blob_sha_r1_16": blob_sha,
            "git_dirty_before_run": dirty.splitlines(),
            "gate_elapsed_s": gate_s, "ctrl_elapsed_s": ctrl_s,
            "panel_elapsed_s": panel_s, "total_elapsed_s": round(time.perf_counter() - t_start, 1),
            "dp_builds_panel": int(PD.N_BUILDS - n_builds0),
            "peak_rss_mb": round(rss, 1) if rss else None,
            "n_checks_gates": gate["n_checks"], "n_fail_gates": gate["n_fail"]}
    (outdir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    print(f"[DONE] {outdir.name} n_fail={n_fail} total={meta['total_elapsed_s']}s "
          f"peak_rss={meta['peak_rss_mb']}MB builds={meta['dp_builds_panel']}")
    print(json.dumps(summary["verdict"], ensure_ascii=False, indent=1))
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    import sys
    sys.exit(main())
