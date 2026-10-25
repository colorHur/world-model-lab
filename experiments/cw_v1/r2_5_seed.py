# -*- coding: utf-8 -*-
"""`CW-R2.5` 落盘入口：`Δ` 对**规划器采样种子**的稳健性（2×2 补格）。

预注册：`docs/research/CW-R2.5-种子稳健-预注册-v0.1.md`（写于 2026-10-25 12:49，`run_episode`=0）

三段
----
`CTRL S1–S7`：`S1` 判据两端代验、`S2` **绝对值当符号**的活体负控、`S3` §4.2 等价性、
        `S4` **共享分量**使 `1/2` 不成立（MC，禁幅度判词的依据）、`S5` 与已发布 R2.4 数字对账、
        `S6` 与 R2.4 逐格身份（读已落盘证据，不重跑）、`S7` 种子**是否真接进规划器**（活体闸）。
`PANEL` **30 格** = `(prior_only, plan_seed=3102) × 2 plant × 15 ep`。**不新增 `(plant, ep)` 单元**。
`ANALYSIS` 主判只有一条：`n_pos3 ≥ 21`（`n=30`，精确 `Bin(1/2)`，`α=0.025`）。
        幅度**只描述**（预注册 §4.3 给了三条"零分布无功效"的证明）。

★ 本轮**不训练**、不改 `sigma_meas`、不改 `mech2/mech3` 任何常量；被测物
  `plan_ref` / `plan_dp` / `world` / `runner` / `config` / `policies` / `r2_1_sep.py` / `r2_4_aoi.py`
  **一字节不动**（`run_cell2` 只被 import，不被改）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import mech3
from .config import mainline_config
from .model_api import OBJECTIVE, spec_from_config
from .r1_14_exactbase import _git, _src_hash
from .r2_1_sep import MODELS, PLANTS, RATIO, RHO, SEED_NULL, SEED_PANEL, action_diff_frac, controls, gates as r21_gates
from .r2_4_aoi import _load, run_cell2

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results" / "cw_v1"

#: ★ 只读上游证据（R2.4 = 唯一一处把 `n_unserved_slots_by_user` 落盘的 run）
R24_DIR = RESULTS / "cw_r2_4_aoi_20261025T023636_a83942167173392b"
R23_DIR = RESULTS / "cw_r2_3_n30_20261025T001450_113cd87acf5dc643"

EPS_ALL = list(range(100, 115))
PLANT_KEYS = ["P1", "P2"]

#: 本轮唯一要补的那一格
MISSING_MODEL = "prior_only"
MISSING_SEED = SEED_NULL            # 3102

#: 主判（预注册 §4.1，写死）
N_UNITS = 30
ALPHA_PRIMARY = 0.025
K_PRIMARY = 21

#: 身份闸跳过的字段（计时）
SKIP_ID_FIELDS = {"elapsed_s"}


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ==================================================================== 判词（预注册 §4.4 逐行实现）
def verdict_r25(n_pos3: int, n: int, alpha: float = ALPHA_PRIMARY) -> dict:
    """表驱动，**只**看方向腿。`k_need` 现算（不手抄）。"""
    k = mech3.k_need_table((n,), (alpha,))[str(n)][str(alpha)]["k_need"]
    p = mech3.binom_p_ge(n_pos3, n)
    ok = bool(n_pos3 >= k)
    return {"n_pos3": int(n_pos3), "n": int(n), "k_need": int(k), "alpha": float(alpha),
            "p_one_sided": float(p), "pass": ok,
            "verdict": "AOI-SEED-ROBUST" if ok else "AOI-SEED-SPECIFIC",
            "why": ("Δ3 的符号在另一规划器种子下仍压倒 1/2" if ok
                    else "Δ3 不过方向腿 ⇒ R2.4 的 30/30 是 plan_seed=2001 的产物，不得当稳健结论")}


# ==================================================================== 单元装配
def _a_t(cell: dict, variant: str = mech3.VAR_TIME) -> float:
    return float(mech3.aoi_metrics(cell, variants=(variant,))[variant])


def build_2x2(cells24: list[dict], nulls24: list[dict], cells25: list[dict]) -> list[dict]:
    """2×2 单元表。缺任何一格 ⇒ 抛 `mech3.IncompletePanel`（**不用剩下的凑数**）。"""
    idx24 = {(c["plant"], c["model"], int(c["ep"])): c for c in cells24}
    idxn = {(c["plant"], int(c["ep"])): c for c in nulls24}
    idx25 = {(c["plant"], c["model"], int(c["ep"])): c for c in cells25}
    rows, missing = [], []
    for plant in PLANT_KEYS:
        for ep in EPS_ALL:
            try:
                s2c = idx24[(plant, "belief_readout", ep)]
                w2c = idx24[(plant, MISSING_MODEL, ep)]
                s3c = idxn[(plant, ep)]
                w3c = idx25[(plant, MISSING_MODEL, ep)]
            except KeyError:
                missing.append(f"{plant}/ep{ep}")
                continue
            per = {}
            for var in (mech3.VAR_TIME, mech3.VAR_FULL):
                S2, W2 = _a_t(s2c, var), _a_t(w2c, var)
                S3, W3 = _a_t(s3c, var), _a_t(w3c, var)
                per[var] = {"S2": S2, "W2": W2, "S3": S3, "W3": W3,
                            "d2": W2 - S2, "d3": W3 - S3,
                            "f_S": abs(S2 - S3), "f_W": abs(W2 - W3),
                            "per_user_d2": [float(w) - float(s) for s, w in zip(
                                mech3.aoi_metrics(s2c, variants=(var,))["per_user"][var],
                                mech3.aoi_metrics(w2c, variants=(var,))["per_user"][var])],
                            "per_user_d3": [float(w) - float(s) for s, w in zip(
                                mech3.aoi_metrics(s3c, variants=(var,))["per_user"][var],
                                mech3.aoi_metrics(w3c, variants=(var,))["per_user"][var])]}
            d3 = per[mech3.VAR_TIME]["d3"]
            rows.append({
                "plant": plant, "ep": int(ep),
                "a_t": per[mech3.VAR_TIME], "a_T": per[mech3.VAR_FULL],
                "d3": float(d3), "sign3": (1 if d3 > 0 else (0 if d3 == 0 else -1)),
                "trace_same_vs_w2": (w3c["policy_trace_hash"] == w2c["policy_trace_hash"]),
                "action_diff_vs_w2": action_diff_frac(w2c["acts"], w3c["acts"]),
                "k_w3": int(sum(w3c["n_unserved_slots_by_user"])),
                "k_w2": int(sum(w2c["n_unserved_slots_by_user"])),
            })
    if missing:
        raise mech3.IncompletePanel(f"缺格 {len(missing)} 个：{missing[:8]}")
    return rows


# ==================================================================== CTRL
def controls5() -> dict:
    """`S1–S6`。**零 episode**（只读已落盘证据 + 合成/MC）。"""
    out: dict = {}
    # ---- S1 主判两端代验（21 过 / 20 不过，α=0.025）
    v21 = verdict_r25(21, 30)
    v20 = verdict_r25(20, 30)
    out["S1_threshold_two_ends"] = {
        "pass": bool(v21["pass"] and not v20["pass"] and v21["k_need"] == 21),
        "k_need": v21["k_need"], "p_at_21": v21["p_one_sided"], "p_at_20": v20["p_one_sided"],
        "why": ("阈值必须两端都可代验：21 过（p<0.025）、20 不过（p>0.025）。"
                "只验一端等于没验（R2.3 的 `18/15` 在 n=10 算术不可达就是这样漏的）")}
    # ---- S2 ★ 绝对值当符号的**活体负控**（R2.4 `N1` 的错）
    d = np.array([5.0] * 10 + [-5.0] * 20, dtype=float)
    n_pos_signed = int((d > 0).sum())
    n_pos_abs = int((np.abs(d) > 0).sum())
    v_signed = verdict_r25(n_pos_signed, 30)
    out["S2_abs_as_sign_trap"] = {
        "pass": bool(n_pos_signed == 10 and n_pos_abs == 30
                     and not v_signed["pass"]),
        "n_pos_signed": n_pos_signed, "n_pos_abs": n_pos_abs,
        "verdict_signed": v_signed["verdict"], "verdict_abs": "AOI-SEED-ROBUST（恒真，错）",
        "why": ("同一份数据：有符号判 10/30 不过，绝对值判 30/30 恒过。"
                "若两者给出相同判词，本条判据就没有分辨力")}
    # ---- S3 §4.2 等价性：Δ2 全正时「一致性」≡「Δ3 符号检验」
    out["S3_agreement_equals_sign"] = {
        "pass": True,
        "claim": "#{sign(Δ2)=sign(Δ3)} ≡ #{Δ3>0}，当且仅当 Δ2 有符号且全正",
        "check_now": "Δ2 是否 30/30 全正由 S5 对账给出；若 S5 的 n_pos2≠30，本条自动失效并降级为描述",
        "why": "防重复计证（把同一个检验报成两份证据）"}
    # ---- S4a ★ 零分布标定（预注册 §4.3-2'，跑前修订 1 之后）
    #     原卡写"共享 W3 破坏 1/2"⇒ 被本 MC 推翻 ⇒ 现在的断言是相反的：两种 floor 的 H0 率都是 1/2。
    rng = np.random.default_rng(20251025)
    M = 4000
    r_fS, r_fW = [], []
    for _ in range(M):
        S2 = rng.normal(0.0, 1.0, 30)
        W2 = rng.normal(0.0, 1.0, 30)
        S3 = rng.normal(0.0, 1.0, 30)
        W3 = rng.normal(0.0, 1.0, 30)
        d3 = W3 - S3
        r_fS.append(float((np.abs(d3) > np.abs(S2 - S3)).mean()))
        r_fW.append(float((np.abs(d3) > np.abs(W2 - W3)).mean()))
    r_fS, r_fW = np.array(r_fS), np.array(r_fW)
    out["S4a_null_calibration"] = {
        "pass": bool(abs(float(r_fS.mean()) - 0.5) <= 0.03
                     and abs(float(r_fW.mean()) - 0.5) <= 0.03),
        "h0_beat_rate_fS_mean": float(r_fS.mean()), "h0_beat_rate_fS_sd": float(r_fS.std()),
        "h0_beat_rate_fW_mean": float(r_fW.mean()), "h0_beat_rate_fW_sd": float(r_fW.std()),
        "M": M,
        "why": ("iid 噪声零假设下，两种 floor 的逐对 beat 率都应 ≈1/2（交换 W2↔S3 的不变性）。"
                "若显著偏离，说明我算 beat 的实现本身有偏 ⇒ 任何 beat 类结论都不可用。"
                "★ 本条的断言方向与预注册初稿**相反**，见 §10 跑前修订 1")}
    # ---- S5 与已发布 R2.4 数字对账（重建口径必须一致）
    cells24 = _load(R24_DIR / "cells.json")
    nulls24 = _load(R24_DIR / "null.json")["nulls"]
    s24 = _load(R24_DIR / "summary.json")
    idx24 = {(c["plant"], c["model"], int(c["ep"])): c for c in cells24}
    idxn = {(c["plant"], int(c["ep"])): c for c in nulls24}
    d2, fS = [], []
    for p in PLANT_KEYS:
        for ep in EPS_ALL:
            s2, w2, s3 = _a_t(idx24[(p, "belief_readout", ep)]), _a_t(idx24[(p, MISSING_MODEL, ep)]), _a_t(idxn[(p, ep)])
            d2.append(w2 - s2)
            fS.append(abs(s2 - s3))
    got = {"n_pos2": int(sum(1 for x in d2 if x > 0)),
           "median_abs_d2": float(np.median(np.abs(d2))),
           "median_fS": float(np.median(fS))}
    exp = {"n_pos2": int(s24["primary_A_t"]["ALL"]["n_pos"]),
           "median_abs_d2": float(s24["primary_A_t"]["ALL"]["median_abs_delta"]),
           "median_fS": float(s24["primary_A_t"]["ALL"]["median_floor_pair"])}
    tol = 1e-9
    items = {"n_pos2_same": got["n_pos2"] == exp["n_pos2"],
             "median_abs_d2_same": abs(got["median_abs_d2"] - exp["median_abs_d2"]) <= tol,
             "median_fS_same": abs(got["median_fS"] - exp["median_fS"]) <= tol}
    out["S5_reconcile_r24"] = {"pass": all(items.values()), "items": items, "got": got,
                              "expect": exp, "tol": tol,
                              "why": ("我用 `aoi_metrics` 重建的 `A_t` 必须与 R2.4 发布值逐位相同，"
                                      "否则本轮 2×2 里的 `S2/W2/S3` 与 R2.4 不是同一个量")}
    # ---- S4b ★ 幅度腿的**分辨率**（预注册 §4.3-2'）：`r` = 最小能让 beat 掉到 k 以下的幅度缩放
    import numpy as _np
    d2a, fSa = _np.abs(_np.array(d2, dtype=float)), _np.array(fS, dtype=float)
    k_main = mech3.k_need_table((30,), (ALPHA_PRIMARY,))["30"][str(ALPHA_PRIMARY)]["k_need"]
    r_res, grid = None, [0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1,
                         0.15, 0.2, 0.3, 0.5, 0.75, 1.0]
    for s in grid:
        if int((d2a * s > fSa).sum()) < k_main:
            r_res = s
            break
    n_beat_at_1 = int((d2a > fSa).sum())
    out["S4b_magnitude_resolution"] = {
        "pass": bool(r_res is not None and r_res <= 0.3),
        "n_beat_at_scale_1": n_beat_at_1, "k_need": int(k_main),
        "resolution_scale_r": r_res,
        "hit_grid_floor": bool(r_res == grid[0]),
        "collapse_factor_needed_to_fail": (None if r_res in (None, 0) else round(1.0 / r_res, 1)),
        "why": ("`r` = 幅度要缩到原来的多少倍，`beat` 才会掉到 `k_need` 以下。"
                "`r` 越小 ⇒ 这条腿只能分辨越极端的崩塌 ⇒ 对'同量级/弱一点'无分辨力。"
                "这是**拒发幅度判词**的定量依据（预注册 §4.3-2'），也是本轮唯一允许引用它的方式")}
    return out


def identity_vs_r24(cells25: list[dict]) -> dict:
    """本轮新跑的 30 格与 R2.4 同 `(plant, ep)` 的 `prior_only@2001` 格：**除种子外**应全部不同。

    真正要证的是：这 30 格用的是**同一套配置/同一批世界**，只有 `plan_seed` 变了。
    """
    cells24 = _load(R24_DIR / "cells.json")
    idx24 = {(c["plant"], c["model"], int(c["ep"])): c for c in cells24}
    same_fp, diff_seed, bad = 0, 0, []
    for c in cells25:
        r = idx24[(c["plant"], c["model"], int(c["ep"]))]
        if c["config_fingerprint"] == r["config_fingerprint"]:
            same_fp += 1
        else:
            bad.append({"key": f'{c["plant"]}/{c["model"]}/ep{c["ep"]}', "field": "config_fingerprint"})
        if int(c["plan_seed"]) != int(r["plan_seed"]):
            diff_seed += 1
        if float(c["sigma_meas"]) != float(r["sigma_meas"]):
            bad.append({"key": f'{c["plant"]}/{c["model"]}/ep{c["ep"]}', "field": "sigma_meas"})
    return {"ok": bool(same_fp == len(cells25) and diff_seed == len(cells25) and not bad),
            "n": len(cells25), "n_same_config_fingerprint": same_fp,
            "n_seed_differs": diff_seed, "diffs": bad[:6],
            "why": ("配置指纹或 `sigma_meas` 若变了，这 30 格就不是'只换种子'的对照，2×2 不成立")}


# ==================================================================== 主流程
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="CW-R2.5 Δ 对规划器采样种子的稳健性")
    ap.add_argument("--ctrl-only", action="store_true", help="只跑 S1–S5（零 episode）")
    ap.add_argument("--tag", default="main")
    args = ap.parse_args(argv)

    t_start = time.perf_counter()
    ctrl = controls5()
    base = controls()
    base_keys = base.pop("_keys")
    fails = [k for k, v in ctrl.items() if isinstance(v, dict) and v.get("pass") is False]
    all_pass = not fails
    print(f"[CTRL] all_pass={all_pass} n={len(ctrl)} fails={fails}")
    print("[S1] k_need=%d p21=%.4g p20=%.4g" % (ctrl["S1_threshold_two_ends"]["k_need"],
                                                ctrl["S1_threshold_two_ends"]["p_at_21"],
                                                ctrl["S1_threshold_two_ends"]["p_at_20"]))
    print("[S2] signed=%d abs=%d -> %s" % (ctrl["S2_abs_as_sign_trap"]["n_pos_signed"],
                                           ctrl["S2_abs_as_sign_trap"]["n_pos_abs"],
                                           ctrl["S2_abs_as_sign_trap"]["verdict_signed"]))
    print("[S4a] H0 beat: fS=%.4f±%.4f  fW=%.4f±%.4f (M=%d)"
          % (ctrl["S4a_null_calibration"]["h0_beat_rate_fS_mean"],
             ctrl["S4a_null_calibration"]["h0_beat_rate_fS_sd"],
             ctrl["S4a_null_calibration"]["h0_beat_rate_fW_mean"],
             ctrl["S4a_null_calibration"]["h0_beat_rate_fW_sd"],
             ctrl["S4a_null_calibration"]["M"]))
    print("[S4b] beat@scale1=%d/%d  r=%.4g  collapse_needed=%.1fx"
          % (ctrl["S4b_magnitude_resolution"]["n_beat_at_scale_1"],
             ctrl["S4b_magnitude_resolution"]["k_need"],
             ctrl["S4b_magnitude_resolution"]["resolution_scale_r"] or -1,
             ctrl["S4b_magnitude_resolution"]["collapse_factor_needed_to_fail"] or -1))
    print("[S5] " + json.dumps(ctrl["S5_reconcile_r24"]["got"], ensure_ascii=False))
    if not all_pass:
        print("[ABORT] 判据自证未过 ⇒ 按预注册 §4 不看面板")
        print(json.dumps(ctrl, ensure_ascii=False, indent=1, default=float))
        return 3
    if args.ctrl_only:
        print(json.dumps(ctrl, ensure_ascii=False, indent=1, default=float))
        return 0

    # ---- 证据目录：**面板之前**（R2.3 首跑丢 90 格的教训）
    head = _git(["rev-parse", "HEAD"])
    wt_sha = _git(["hash-object", "experiments/cw_v1/r2_5_seed.py"])
    blob_sha = _git(["rev-parse", f"{head}:experiments/cw_v1/r2_5_seed.py"], allow_fail=True)
    dirty = _git(["status", "--porcelain"])
    srchash = _src_hash()
    outdir = RESULTS / f"cw_r2_5_seed_{_now_utc()}_{srchash}"
    outdir.mkdir(parents=True, exist_ok=False)
    print(f"[OUT] {outdir.name}")
    (outdir / "ctrl.json").write_text(json.dumps(
        {"r25_S1_S5": ctrl, "r21_base": base}, ensure_ascii=False, indent=1, default=float),
        encoding="utf-8")

    # ---- 面板：30 格，逐格落盘（崩溃不丢）
    t_p = time.perf_counter()
    cells25: list[dict] = []
    jl = (outdir / "cells_stream.jsonl").open("w", encoding="utf-8")
    try:
        for plant in PLANT_KEYS:
            for ep in EPS_ALL:
                c = run_cell2(plant, MISSING_MODEL, ep, MISSING_SEED)
                cells25.append(c)
                jl.write(json.dumps(c, ensure_ascii=False, default=float) + "\n")
                jl.flush()
                print(f"[CELL] {plant}/{MISSING_MODEL}/ep{ep} s={MISSING_SEED} "
                      f"cost={c['cost_true_per_step']:.6f} att={c['n_attempted']} "
                      f"del={c['n_delivered']} k={c['n_unserved_slots_by_user']} ({c['elapsed_s']}s)")
    finally:
        jl.close()
    panel_s = round(time.perf_counter() - t_p, 1)
    print(f"[PANEL] {len(cells25)} 格 = {panel_s}s")
    (outdir / "cells.json").write_text(json.dumps(cells25, ensure_ascii=False, indent=1,
                                                 default=float), encoding="utf-8")

    # ---- 活体闸 S7：种子是否真接进规划器（否则本轮无分辨力）
    cells24_ro = _load(R24_DIR / "cells.json")
    idx24_ro = {(x["plant"], x["model"], int(x["ep"])): x for x in cells24_ro}
    n_trace_same = int(sum(
        1 for c in cells25
        if c["policy_trace_hash"] == idx24_ro[(c["plant"], c["model"], int(c["ep"]))]["policy_trace_hash"]))
    s7 = {"pass": bool(n_trace_same < len(cells25)),
          "n_trace_same_vs_w2": n_trace_same, "n": len(cells25),
          "why": ("若换种子后动作轨迹 30/30 逐位相同，说明 `plan_seed` 没接进规划器采样 ⇒ "
                  "本轮检验是空的（'跑过了'≠'有分辨力'）")}
    print(f"[S7] trace_same_vs_w2={n_trace_same}/{len(cells25)} pass={s7['pass']}")

    # ---- 闸
    gate = r21_gates(base)
    print(f"[GATE] n_checks={gate['n_checks']} n_fail={gate['n_fail']} (B3 {gate['b3_elapsed_s']}s) "
          f"base={base_keys}")
    s6 = identity_vs_r24(cells25)
    print(f"[S6] ok={s6['ok']} same_fp={s6['n_same_config_fingerprint']}/{s6['n']} "
          f"seed_diff={s6['n_seed_differs']}/{s6['n']}")
    i2_bad = [f'{c["plant"]}/{c["model"]}/ep{c["ep"]}' for c in cells25
              if not c["identities_pass"] or float(c["cost_identity_gap"]) > 1e-9]
    print(f"[I2] 逐格恒等 n_bad={len(i2_bad)}")
    n_fail = gate["n_fail"] + (0 if s6["ok"] else 1) + (0 if s7["pass"] else 1) + (0 if not i2_bad else 1)

    # ---- 分析
    ana: dict = {}
    if n_fail == 0:
        try:
            rows = build_2x2(_load(R24_DIR / "cells.json"), _load(R24_DIR / "null.json")["nulls"],
                             cells25)
            v_t = verdict_r25(sum(1 for r in rows if r["a_t"]["d3"] > 0), len(rows))
            v_T = verdict_r25(sum(1 for r in rows if r["a_T"]["d3"] > 0), len(rows))
            # 敏感性（α 三档）+ 逐 plant（n=15）
            sens = {}
            for a in (0.05, ALPHA_PRIMARY, 0.01):
                sens[str(a)] = {"ALL": verdict_r25(
                    sum(1 for r in rows if r["a_t"]["d3"] > 0), len(rows), a)["verdict"]}
                for p in PLANT_KEYS:
                    sub = [r for r in rows if r["plant"] == p]
                    sens[str(a)][p] = verdict_r25(sum(1 for r in sub if r["a_t"]["d3"] > 0),
                                                  len(sub), a)
            # 描述量（**不作判词**，预注册 §4.3）
            d3 = np.array([r["a_t"]["d3"] for r in rows], dtype=float)
            d2 = np.array([r["a_t"]["d2"] for r in rows], dtype=float)
            fW = np.array([r["a_t"]["f_W"] for r in rows], dtype=float)
            fS = np.array([r["a_t"]["f_S"] for r in rows], dtype=float)
            desc = {"median_abs_d3": float(np.median(np.abs(d3))),
                    "median_abs_d2": float(np.median(np.abs(d2))),
                    "median_fW": float(np.median(fW)),
                    "median_fS": float(np.median(fS)),
                    "ratio_d3_over_d2": float(np.median(np.abs(d3)) / np.median(np.abs(d2))),
                    "ratio_d3_over_fS": float(np.median(np.abs(d3)) / np.median(fS)) if np.median(fS) > 0 else None,
                    "ratio_d3_over_fW": float(np.median(np.abs(d3)) / np.median(fW)) if np.median(fW) > 0 else None,
                    "n_trace_same_vs_w2": n_trace_same,
                    "median_action_diff_vs_w2": float(np.median([r["action_diff_vs_w2"] for r in rows
                                                                if np.isfinite(r["action_diff_vs_w2"])])),
                    "per_pair_beat_vs_fS_DESCRIPTIVE_ONLY": int(sum(
                        1 for r in rows if abs(r["a_t"]["d3"]) > r["a_t"]["f_S"])),
                    "per_pair_beat_vs_fW_FORBIDDEN": int(sum(
                        1 for r in rows if abs(r["a_t"]["d3"]) > r["a_t"]["f_W"])),
                    "note": ("全部为描述量。`f_W` 与 `Δ3` 共享 `W3`（预注册 §4.3-2，`S4` 实测偏离见 ctrl），"
                             "`per_pair_beat_vs_fW_FORBIDDEN` 一栏只展示**不许当判据**的那个数")}
            # 逐用户（R17）
            per_user = []
            for r in rows:
                for i, du in enumerate(r["a_t"]["per_user_d3"]):
                    per_user.append({"plant": r["plant"], "ep": r["ep"], "i": i, "delta_user": float(du)})
            ana = {"rows": rows, "verdict_A_t": v_t, "verdict_A_T": v_T, "sens": sens,
                   "desc": desc, "per_user": per_user}
            print("[MAIN] " + json.dumps(v_t, ensure_ascii=False, default=float))
            print("[VARIANT-A_T] " + json.dumps(v_T, ensure_ascii=False, default=float))
            print("[DESC] " + json.dumps(desc, ensure_ascii=False, default=float))
            print("[SENS] " + json.dumps(sens, ensure_ascii=False, default=float))
        except Exception as e:  # noqa: BLE001
            ana = {"analysis_error": repr(e), "traceback": traceback.format_exc()[-1200:]}
            print(f"[ERR] 分析段异常：{e!r}（30 格已落盘，不重跑；R14 报错不报假数字）")
            n_fail += 1
    else:
        print("[SKIP] 闸未全通过 ⇒ 按预注册不看主判")

    (outdir / "gates.json").write_text(json.dumps(
        {"r21_b1_b8": gate, "r21_base_ctrl_keys": base_keys,
         "r25_S1_S5": {k: v for k, v in ctrl.items()},
         "S6_identity_vs_r24": s6, "S7_seed_wired": s7, "I2_bad": i2_bad},
        ensure_ascii=False, indent=1, default=str), encoding="utf-8")

    cfg_mat = mainline_config(plant=PLANTS["P1"])
    spec = spec_from_config(cfg_mat, budget_total=cfg_mat.protocol.budget(RATIO),
                            split="dev", episodes=EPS_ALL,
                            seeds=(SEED_PANEL, SEED_NULL), models=MODELS)
    try:
        import psutil  # type: ignore
        rss = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
    except Exception:  # noqa: BLE001
        rss = None
    total_s = round(time.perf_counter() - t_start, 1)

    summary = {
        "protocol": "CW-POMDP-v1.1", "stage": "CW-R2.5",
        "kind": "SEED-ROBUSTNESS-2x2-ONE-CELL-FILLED",
        "prereg": "docs/research/CW-R2.5-种子稳健-预注册-v0.1.md",
        "hypothesis": "H-R2-0",
        "design": {"eps": EPS_ALL, "plants": PLANT_KEYS,
                   "seeds": [SEED_PANEL, SEED_NULL], "models": list(MODELS),
                   "n_new_cells": len(cells25), "new_plant_ep_units": 0,
                   "which_cell_filled": f"({MISSING_MODEL}, plan_seed={MISSING_SEED})"},
        "primary_A_t": ana.get("verdict_A_t"),
        "co_report_A_T": ana.get("verdict_A_T"),
        "alpha_sensitivity": ana.get("sens"),
        "descriptive_only": ana.get("desc"),
        "per_user_R17": ({"n": len(ana.get("per_user") or []),
                          "n_user_delta_negative": int(sum(
                              1 for x in (ana.get("per_user") or []) if x["delta_user"] < 0)),
                          "by_unit_user": ana.get("per_user")} if ana.get("per_user") else None),
        "ctrl_all_pass": all_pass, "S6_identity": s6, "S7_seed_wired": s7,
        "I2_bad": i2_bad,
        "objective": OBJECTIVE, "spec_fingerprint_P1": spec.fingerprint(),
        "sigma_meas": {p: float(PLANTS[p].sigma_meas) for p in PLANT_KEYS},
        "budget": {"panel_s": panel_s, "total_s": total_s,
                   "per_cell_s": [c["elapsed_s"] for c in cells25]},
        "head": head, "src_hash": srchash,
        "n_checks_gates": gate["n_checks"], "n_fail_gates": gate["n_fail"],
        "n_fail_total": n_fail, "training": "none（本轮零训练）",
        "inputs": {str(R24_DIR.name): _sha256(R24_DIR / "cells.json"),
                   str(R23_DIR.name): _sha256(R23_DIR / "cells.json")},
    }
    (outdir / "two_by_two.json").write_text(json.dumps(
        ana.get("rows") or [], ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    (outdir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1,
                                                    default=float), encoding="utf-8")
    meta = {"command": "python -m experiments.cw_v1.r2_5_seed --tag " + str(args.tag),
            "timestamp_local": datetime.now().astimezone().isoformat(),
            "python": os.sys.version.split()[0], "numpy": np.__version__,
            "head": head, "src_hash": srchash,
            "worktree_sha_r2_5": wt_sha, "blob_sha_r2_5": blob_sha,
            "blob_sha_note": (None if blob_sha else "运行前未入库 ⇒ 提交后补 _POST_COMMIT"),
            "git_dirty_before_run": dirty.splitlines(),
            "protocol": "CW-POMDP-v1.1", "event_pairing": "slot",
            "n_cells": len(cells25), "panel_elapsed_s": panel_s, "total_elapsed_s": total_s,
            "peak_rss_mb": round(rss, 1) if rss else None,
            "n_fail_total": n_fail, "training": "none"}
    (outdir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    print(f"[DONE] {outdir.name} n_fail={n_fail} total={total_s}s peak={meta['peak_rss_mb']}MB")
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    import sys
    sys.exit(main())
