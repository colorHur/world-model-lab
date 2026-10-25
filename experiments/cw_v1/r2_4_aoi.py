# -*- coding: utf-8 -*-
"""`CW-R2.4` 落盘入口：把**非删除型**任务陈旧度立为 R2 主指标（先算条件化零分布）。

预注册：`docs/research/CW-R2.4-AoI主指标-预注册-v0.1.md`（写于 2026-10-25 10:4x，`run_episode`=0）

五段
----
`CTRL` 判据自证 `C1–C11`：`C1–C7` 合成三腿、`C8` 与 R2.3 已发布数字对账、
        `C9` **删窗口伪影活体负控**、`C10` 前缀性质 + 重建公式（真跑 2 格）、`C11` 分解恒等式。
`GATE` `B1–B8`：复用 `r2_1_sep.gates()`（11 项，含 `B3` 直通恒等 + 活体负控）。
`I1`  身份闸：90 格 vs R2.3 `cells.json`/`null.json`，**除 `elapsed_s` 外全部共有字段**逐位相同。
`PANEL` 90 格 = 面板 `2 × 2 × 15` + 零控 `2 × 15`（**不新增任何 `(plant, ep)` 单元**）。
`ANALYSIS` 主判 `A_t` + 同报 `A_T` + `N1/N2/N3` + `alpha` 敏感性 + 逐用户 `R17`。

★ 本轮**不训练**、不改 `sigma_meas`、不改 `mech2.py` 任何常量；被测物
  `plan_ref` / `plan_dp` / `world` / `runner` / `config` / `policies` / `r2_1_sep.py` **一字节不动**。
★ 重跑同一批格子的**唯一**原因：`aoi_tavg_by_user` 与 `aoi_max_by_user` 之外还需要
  `n_unserved_slots_by_user`，而 R2.3 的 `cells.json` 没写它（`run_cell` 不落这个字段）。
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
from .model_policy import ModelPolicy
from .r1_14_exactbase import _git, _src_hash
from .r2_1_sep import (MODELS, PLANTS, RATIO, RHO, SEED_NULL, SEED_PANEL, _MODEL_CTOR,
                       _factory, action_diff_frac, controls, gates as r21_gates, run_cell)
from .runner import check_identities, run_episode
from .world import CWWorld

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results" / "cw_v1"

#: ★ 写死的上游证据（`C8` 对账 + `I1` 身份闸都用它）
R23_DIR = RESULTS / "cw_r2_3_n30_20261025T001450_113cd87acf5dc643"

EPS_ALL = list(range(100, 115))
PLANT_KEYS = ["P1", "P2"]

#: 身份闸：**除计时外**全部共有字段逐位相同
SKIP_ID_FIELDS = {"elapsed_s"}


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


# ==================================================================== 单格（run_cell + 2 字段）
def run_cell2(plant: str, model: str, ep: int, plan_seed: int) -> dict:
    """与 `r2_1_sep.run_cell` **逐字同源**，只多写 `n_slots` 与 `n_unserved_slots_by_user`。

    ★ 复制而不是改 `r2_1_sep.py`：那是 R2.1 的证据入口，动了它 R2.1 的 `summary.json`
      就失去自证能力。这里多出来的两个字段是**纯记录**，`I1` 会证明其余字段逐位未变。
    """
    cfg = mainline_config(plant=PLANTS[plant])
    budget = cfg.protocol.budget(RATIO)
    fac = _factory(cfg, plan_seed)
    if model == "raw":
        policy = fac()
    else:
        policy = ModelPolicy(_MODEL_CTOR[model](), fac)
    t0 = time.perf_counter()
    res = run_episode(cfg, policy, split="dev", episode=int(ep),
                      budget_total=budget, rho=RHO)
    elapsed = time.perf_counter() - t0
    chk = check_identities(res, cfg)
    acts = (policy.acts() if isinstance(policy, ModelPolicy)
            else list(getattr(policy, "acts", []) or []))
    cost_by_user = [float(x) for x in res.cost_true_by_user]
    return {
        "plant": plant, "model": model, "ep": int(ep), "plan_seed": int(plan_seed),
        "sigma_meas": float(cfg.plant.sigma_meas),
        "config_fingerprint": res.config_fingerprint,
        "cost_true_per_step": float(res.cost_true_per_step),
        "cost_est_per_step": float(res.cost_est_per_step),
        "cost_true_by_user": cost_by_user,
        "cost_est_by_user": [float(x) for x in res.cost_est_by_user],
        "n_attempted_by_user": [int(x) for x in res.n_attempted_by_user],
        "n_delivered_by_user": [int(x) for x in res.n_delivered_by_user],
        "n_attempted": int(res.n_attempted), "n_delivered": int(res.n_delivered),
        "n_probe": int(res.n_probe), "n_idle": int(res.n_idle),
        "n_forced_hold": int(res.n_forced_hold),
        "aoi_tavg_by_user": list(res.aoi_tavg_by_user),
        "aoi_max_by_user": list(res.aoi_max_by_user),
        "n_never_served": int(res.n_never_served),
        "policy_trace_hash": res.policy_trace_hash,
        "readout_signature": (policy.readout_signature() if isinstance(policy, ModelPolicy) else None),
        "n_model_calls": (policy.n_model_calls if isinstance(policy, ModelPolicy) else 0),
        "acts": [[t, k, u] for (t, k, u) in acts],
        "identities_pass": bool(chk["pass"]),
        "identities_fail": [k for k, v in chk["items"].items() if not v],
        "cost_identity_gap": abs(float(sum(cost_by_user)) - float(res.cost_true_per_step)),
        "elapsed_s": round(elapsed, 2),
        # ---- 本轮新增（纯记录）----
        "n_slots": int(res.n_slots),
        "n_unserved_slots_by_user": [int(x) for x in res.n_unserved_slots_by_user],
    }


# ==================================================================== 合成行
def _rows(deltas, floors, plants=None) -> list[dict]:
    out = []
    for i, (d, f) in enumerate(zip(deltas, floors)):
        pl = (plants[i] if plants else ("P1" if i % 2 == 0 else "P2"))
        out.append({"plant": pl, "ep": 100 + i // 2, "variant": "SYN",
                    "a_strong": 0.0, "a_weak": float(d), "a_null": 0.0,
                    "delta": float(d), "floor_pair": float(f),
                    "sign": (1 if d > 0 else (0 if d == 0 else -1)),
                    "beat": (1 if abs(d) > f else 0),
                    "ratio_desc": (abs(d) / f if f > 0 else None),
                    "k_strong": 0, "k_weak": 0, "k_null": 0, "per_user_delta": []})
    return out


# ==================================================================== CTRL
def controls2() -> dict:
    """`C1–C11`。零 episode（`C10` 例外：2 格真跑，单独计数）。"""
    out: dict = {}
    n = 30
    # C1 全零差
    r = _rows([0.0] * n, [1.0] * n)
    out["C1_all_zero_diff"] = {"pass": mech3.block3(r, n)["verdict"] == mech3.V_NONE,
                               "verdict": mech3.block3(r, n)["verdict"]}
    # C2 反向全大
    r = _rows([-5.0] * n, [1.0] * n)
    b = mech3.block3(r, n)
    out["C2_reversed"] = {"pass": b["verdict"] == mech3.V_SEP_REV, "verdict": b["verdict"]}
    # C3 正向全大
    r = _rows([5.0] * n, [1.0] * n)
    b = mech3.block3(r, n)
    out["C3_separates"] = {"pass": b["verdict"] == mech3.V_SEP, "verdict": b["verdict"]}
    # C4 缺格
    r = _rows([5.0] * (n - 1), [1.0] * (n - 1))
    b = mech3.block3(r, n)
    out["C4_incomplete"] = {"pass": b["verdict"] == mech3.V_INCOMPLETE, "verdict": b["verdict"]}
    # C5 非有限
    r = _rows([5.0] * n, [1.0] * n)
    r[7]["floor_pair"] = float("nan")
    b = mech3.block3(r, n)
    out["C5_nonfinite"] = {"pass": b["verdict"] == mech3.V_UNDEF, "verdict": b["verdict"],
                           "named": b["nonfinite_cells"][:3]}
    # C6 符号过、幅度不过
    r = _rows([5.0] * 20 + [-5.0] * 10, [100.0] * n)
    b = mech3.block3(r, n)
    out["C6_sign_only"] = {"pass": b["verdict"] == mech3.V_SIGN, "verdict": b["verdict"]}
    # C7 n=30 下 20+20 不可达（代验，不是断言）
    out["C7_pos_neg_infeasible"] = {"pass": (20 + 20 > n),
                                    "n_pos+n_neg": 40, "n": n,
                                    "why": "Δ 的符号互斥 ⇒ n_pos+n_neg ≤ n"}
    # C8 与 R2.3 已发布数字对账（aoi_tavg 口径）
    s23 = _load(R23_DIR / "summary.json")["secondary_mean_task_aoi"]
    cells23 = _load(R23_DIR / "cells.json")
    nulls23 = _load(R23_DIR / "null.json")["nulls"]
    idx = {(c["plant"], c["model"], int(c["ep"])): c for c in cells23}
    idxn = {(c["plant"], int(c["ep"])): c for c in nulls23}

    def _mean_or_none(c):
        v = list(c["aoi_tavg_by_user"])
        return None if (not v or any(x is None for x in v)) else float(np.mean([float(x) for x in v]))

    ds, fs, la = [], [], []
    for p in PLANT_KEYS:
        for ep in EPS_ALL:
            mb = _mean_or_none(idx[(p, "belief_readout", ep)])
            mw = _mean_or_none(idx[(p, "prior_only", ep)])
            mn = _mean_or_none(idxn[(p, ep)])
            if None in (mb, mw, mn):
                continue
            ds.append(mw - mb)
            fs.append(abs(mb - mn))
            la.append(mb)
    got = {"n_pos": int(sum(1 for x in ds if x > 0)),
           "median_abs_delta": float(np.median(np.abs(ds))),
           "median_floor": float(np.median(fs))}
    exp = {"n_pos": int(s23["n30"]["n_pos"]),
           "median_abs_delta": float(s23["n30"]["median_abs_delta"]),
           "median_floor": float(s23["n30"]["median_floor"])}
    tol = 1e-9
    c8_items = {"n_pos_same": got["n_pos"] == exp["n_pos"],
                "median_abs_delta_same": abs(got["median_abs_delta"] - exp["median_abs_delta"]) <= tol,
                "median_floor_same": abs(got["median_floor"] - exp["median_floor"]) <= tol}
    out["C8_reconcile_r23"] = {"pass": all(c8_items.values()), "items": c8_items,
                               "got": got, "expect": exp, "tol": tol,
                               "why": "重建口径若与 R2.3 `secondary_aoi` 不一致，本轮的'改写'就无从对照"}
    # C9 ★ 删窗口伪影活体负控（合成格，走真实 `aoi_metrics` 代码路径）
    T = 256
    c_strong = {"plant": "P1", "model": "belief_readout", "ep": 0, "n_slots": T,
                "aoi_tavg_by_user": [0.0, 0.0],
                "n_unserved_slots_by_user": [T - 1, T - 1],   # 只在最后一槽送达
                "aoi_max_by_user": [0, 0], "n_never_served": 0}
    c_weak = {"plant": "P1", "model": "prior_only", "ep": 0, "n_slots": T,
              "aoi_tavg_by_user": [20.0, 20.0],
              "n_unserved_slots_by_user": [0, 0],            # 从第一槽起就在服务
              "aoi_max_by_user": [40, 40], "n_never_served": 0}
    ms, mw = mech3.aoi_metrics(c_strong), mech3.aoi_metrics(c_weak)
    d_old = float(np.mean(c_weak["aoi_tavg_by_user"])) - float(np.mean(c_strong["aoi_tavg_by_user"]))
    d_new = mw[mech3.VAR_TIME] - ms[mech3.VAR_TIME]
    out["C9_window_artifact"] = {
        "pass": bool(d_old > 0 and d_new < 0),
        "delta_aoi_tavg": d_old, "delta_A_t": d_new,
        "A_t_strong": ms[mech3.VAR_TIME], "A_t_weak": mw[mech3.VAR_TIME],
        "why": ("旧口径把'拖到最后一槽才服务'判成更好（窗口只剩 1 槽、均值 0）；"
                "新口径把它判成更差 ⇒ 若新口径也判更好，修补就无分辨力，本轮全部作废")}
    # C11 分解恒等式（§5）
    def _dec(levels, deltas, floors):
        return {"lvl": float(np.median(levels)), "d": float(np.median(np.abs(deltas))),
                "f": float(np.median(floors))}
    cd = _dec([idx[(p, "belief_readout", ep)]["cost_true_per_step"]
               for p in PLANT_KEYS for ep in EPS_ALL],
              [idx[(p, "prior_only", ep)]["cost_true_per_step"]
               - idx[(p, "belief_readout", ep)]["cost_true_per_step"]
               for p in PLANT_KEYS for ep in EPS_ALL],
              [abs(idx[(p, "belief_readout", ep)]["cost_true_per_step"]
                   - idxn[(p, ep)]["cost_true_per_step"])
               for p in PLANT_KEYS for ep in EPS_ALL])
    ad = _dec(la, ds, fs)
    lhs_c = cd["d"] / cd["f"]
    rhs_c = (cd["d"] / cd["lvl"]) / (cd["f"] / cd["lvl"])
    lhs_a = ad["d"] / ad["f"]
    rhs_a = (ad["d"] / ad["lvl"]) / (ad["f"] / ad["lvl"])
    out["C11_decomposition_identity"] = {
        "pass": bool(abs(lhs_c - rhs_c) <= 1e-9 and abs(lhs_a - rhs_a) <= 1e-9),
        "cost": {**cd, "ratio": lhs_c}, "aoi": {**ad, "ratio": lhs_a},
        "factor_effect": (ad["d"] / ad["lvl"]) / (cd["d"] / cd["lvl"]),
        "factor_floor": (cd["f"] / cd["lvl"]) / (ad["f"] / ad["lvl"]),
        "why": "两因子乘积必须等于比值之比（代数恒等式；不闭合说明报告里的数抄错）"}
    return out


# ==================================================================== C10（真跑 2 格）
def c10_prefix_and_rebuild() -> dict:
    """前缀性质 + `(aoi_tavg, k)` 重建公式的正确性。**真跑 2 格**。"""
    import experiments.cw_v1.runner as R
    orig = R.CWWorld
    cap: list = []

    class _Cap(orig):  # type: ignore[misc,valid-type]
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            cap.append(self)

    R.CWWorld = _Cap
    try:
        c1 = run_cell2("P1", "belief_readout", 100, SEED_PANEL)
    finally:
        R.CWWorld = orig
    c2 = run_cell("P1", "belief_readout", 100, SEED_PANEL)   # 未包装，做身份对照
    w = cap[-1]
    hist = w.aoi_history
    T = len(hist)
    prefix_ok = all(
        max([t for t in range(T) if hist[t][i] < 0], default=-1)
        < min([t for t in range(T) if hist[t][i] >= 0], default=T)
        for i in range(w.n_users))
    direct, rebuilt, diffs = [], [], []
    direct_A_t, rebuilt_A_t, direct_A_T, rebuilt_A_T = [], [], [], []
    for var in (mech3.VAR_TIME, mech3.VAR_FULL):
        direct, rebuilt = [], []
        for i in range(w.n_users):
            col = [hist[t][i] for t in range(T)]
            impute = (lambda t: float(t)) if var == mech3.VAR_TIME else (lambda t: float(T))
            d = float(np.mean([(a if a >= 0 else impute(t)) for t, a in enumerate(col)]))
            m = c1["aoi_tavg_by_user"][i]
            k = c1["n_unserved_slots_by_user"][i]
            r = mech3._user_aoi(k, m, T, var)
            direct.append(d)
            rebuilt.append(r)
            if abs(d - r) > 1e-12:
                diffs.append({"variant": var, "user": i, "direct": d, "rebuilt": r})
        if var == mech3.VAR_TIME:
            direct_A_t, rebuilt_A_t = direct, rebuilt
        else:
            direct_A_T, rebuilt_A_T = direct, rebuilt
    id_diffs = []
    for f in c1:
        if f in SKIP_ID_FIELDS or f in ("n_slots", "n_unserved_slots_by_user"):
            continue
        a, b = c1[f], c2[f]
        same = (float(a) == float(b)) if isinstance(a, float) else (a == b)
        if not same:
            id_diffs.append({"field": f})
    return {"pass": bool(prefix_ok and not diffs and not id_diffs and T == int(c1["n_slots"])),
            "T": T, "prefix_ok": bool(prefix_ok),
            "n_rebuild_diff": len(diffs), "rebuild_diffs": diffs[:4],
            "n_identity_diff": len(id_diffs), "identity_diffs": id_diffs[:4],
            "direct_A_t": direct_A_t, "rebuilt_A_t": rebuilt_A_t,
            "direct_A_T": direct_A_T, "rebuilt_A_T": rebuilt_A_T,
            "why": ("前缀性质若被破坏，`(aoi_tavg,k)` 就无法重建全时隙均值；"
                    "重建若不等，主指标算错，本轮全部作废"),
            "cost": "2 格真跑（包装 1 + 未包装 1）"}


# ==================================================================== 身份闸 I1
def identity_gate_23(cells: list[dict], nulls: list[dict]) -> dict:
    cells23 = _load(R23_DIR / "cells.json")
    nulls23 = _load(R23_DIR / "null.json")["nulls"]
    ref = {(c["plant"], c["model"], int(c["ep"])): c for c in cells23}
    refn = {(c["plant"], int(c["ep"])): c for c in nulls23}
    diffs, checked = [], 0
    for cur, table, fmt in ((cells, ref, "cell"), (nulls, refn, "null")):
        for c in cur:
            key = ((c["plant"], c["model"], int(c["ep"])) if fmt == "cell"
                   else (c["plant"], int(c["ep"])))
            r = table.get(key)
            if r is None:
                diffs.append({"key": str(key), "why": "R2.3 无此格"})
                continue
            checked += 1
            for f in c:
                if f in SKIP_ID_FIELDS or f not in r:
                    continue
                a, b = c[f], r[f]
                same = (float(a) == float(b)) if isinstance(a, float) else (a == b)
                if not same:
                    diffs.append({"key": str(key), "field": f, "now": a, "r23": b})
    n_common = sum(1 for f in cells[0] if f not in SKIP_ID_FIELDS and f in cells23[0])
    return {"ok": not diffs, "n_checked": checked, "n_diff": len(diffs), "diffs": diffs[:10],
            "n_fields_compared": n_common,
            "fields_skipped": sorted(SKIP_ID_FIELDS),
            "why": "同 seed 同配置重跑若读数变了，90 格就不属于同一个世界，主指标作废"}


# ==================================================================== 主流程
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="CW-R2.4 非删除型 AoI 主指标")
    ap.add_argument("--ctrl-only", action="store_true", help="只跑 C1–C11（C10 会真跑 2 格）")
    ap.add_argument("--tag", default="main")
    args = ap.parse_args(argv)

    t_start = time.perf_counter()
    ctrl = controls2()
    c10 = c10_prefix_and_rebuild()
    ctrl["C10_prefix_and_rebuild"] = c10
    kk = mech3.k_need_table()
    ctrl["k_need"] = kk
    # ★ `B1–B8` 用的是 `r2_1_sep.controls()` 的 `C1–C8`（那条实现守的是模型层边界），
    #   与本轮 `C1–C11`（守的是 AoI 判据）**是两套**，绝不混表。
    base = controls()
    base_keys = base.pop("_keys")

    fails = [k for k, v in ctrl.items() if isinstance(v, dict) and v.get("pass") is False]
    all_pass = not fails
    print(f"[CTRL] all_pass={all_pass} n={len([k for k in ctrl if k.startswith('C')])} "
          f"fails={fails}")
    print("[K_NEED] " + json.dumps({n: {a: kk[n][a]["k_need"] for a in kk[n]} for n in kk},
                                  ensure_ascii=False))
    print("[C8] " + json.dumps(ctrl["C8_reconcile_r23"]["got"], ensure_ascii=False))
    print("[C9] delta_aoi_tavg=%.4f delta_A_t=%.4f T=%d"
          % (ctrl["C9_window_artifact"]["delta_aoi_tavg"],
             ctrl["C9_window_artifact"]["delta_A_t"], c10["T"]))
    print("[C11] factor_effect=%.4f factor_floor=%.4f"
          % (ctrl["C11_decomposition_identity"]["factor_effect"],
             ctrl["C11_decomposition_identity"]["factor_floor"]))

    if not all_pass:
        print("[ABORT] 判据自证未过 ⇒ 按预注册 §7-1 不看闸与面板")
        print(json.dumps(ctrl, ensure_ascii=False, indent=1, default=float))
        return 3
    if args.ctrl_only:
        print(json.dumps(ctrl, ensure_ascii=False, indent=1, default=float))
        return 0

    # ---- 证据目录：**面板之前**（R2.3 首跑丢 90 格的教训）
    head = _git(["rev-parse", "HEAD"])
    wt_sha = _git(["hash-object", "experiments/cw_v1/r2_4_aoi.py"])
    blob_sha = _git(["rev-parse", f"{head}:experiments/cw_v1/r2_4_aoi.py"], allow_fail=True)
    dirty = _git(["status", "--porcelain"])
    srchash = _src_hash()
    outdir = RESULTS / f"cw_r2_4_aoi_{_now_utc()}_{srchash}"
    outdir.mkdir(parents=True, exist_ok=False)
    print(f"[OUT] {outdir.name}")
    (outdir / "ctrl.json").write_text(json.dumps(
        {"r24_C1_C11": ctrl, "r21_base": base, "k_need": kk},
        ensure_ascii=False, indent=1, default=float), encoding="utf-8")

    # ---- 面板
    t_p = time.perf_counter()
    cells, nulls = [], []
    for plant in PLANT_KEYS:
        for model in MODELS:
            for ep in EPS_ALL:
                c = run_cell2(plant, model, ep, SEED_PANEL)
                cells.append(c)
                print(f"[CELL] {plant}/{model:>14}/ep{ep} cost={c['cost_true_per_step']:.6f} "
                      f"att={c['n_attempted']} del={c['n_delivered']} "
                      f"k={c['n_unserved_slots_by_user']} ({c['elapsed_s']}s)")
    for plant in PLANT_KEYS:
        for ep in EPS_ALL:
            c = run_cell2(plant, "belief_readout", ep, SEED_NULL)
            nulls.append(c)
            print(f"[NULL] {plant}/ep{ep} cost={c['cost_true_per_step']:.6f} "
                  f"k={c['n_unserved_slots_by_user']} ({c['elapsed_s']}s)")
    panel_s = round(time.perf_counter() - t_p, 1)
    print(f"[PANEL] {len(cells)} 格 + {len(nulls)} 零控 = {panel_s}s")
    (outdir / "cells.json").write_text(json.dumps(cells, ensure_ascii=False, indent=1,
                                                 default=float), encoding="utf-8")
    (outdir / "null.json").write_text(json.dumps({"nulls": nulls}, ensure_ascii=False,
                                                 indent=1, default=float), encoding="utf-8")

    # ---- 闸
    gate = r21_gates(base)
    print(f"[GATE] n_checks={gate['n_checks']} n_fail={gate['n_fail']} (B3 {gate['b3_elapsed_s']}s) "
          f"base={base_keys}")
    i1 = identity_gate_23(cells, nulls)
    print(f"[I1] ok={i1['ok']} checked={i1['n_checked']} fields={i1['n_fields_compared']} "
          f"diff={i1['n_diff']}")
    i2_bad = [f'{c["plant"]}/{c["model"]}/ep{c["ep"]}' for c in cells + nulls
              if not c["identities_pass"] or float(c["cost_identity_gap"]) > 1e-9]
    print(f"[I2] 逐格恒等 n_bad={len(i2_bad)}")
    n_fail = gate["n_fail"] + (0 if i1["ok"] else 1) + (0 if not i2_bad else 1)

    # ---- 分析（异常不许吞掉已落盘的 90 格）
    ana: dict = {}
    if n_fail == 0:
        try:
            rt = mech3.build_units(cells, nulls, PLANT_KEYS, EPS_ALL, mech3.VAR_TIME)
            rT = mech3.build_units(cells, nulls, PLANT_KEYS, EPS_ALL, mech3.VAR_FULL)
            vt = mech3.verdicts3(rt, PLANT_KEYS, EPS_ALL, variant=mech3.VAR_TIME)
            vT = mech3.verdicts3(rT, PLANT_KEYS, EPS_ALL, variant=mech3.VAR_FULL)
            nd = mech3.null_diagnostics(rt)
            ag = mech3.variant_agreement(vt, vT)
            sens = {}
            for a in (0.05, 0.025, 0.01):
                sens[str(a)] = {b: mech3.block3([r for r in rt if b == "ALL" or r["plant"] == b],
                                                (len(rt) if b == "ALL" else len(EPS_ALL)),
                                                alpha=a)["verdict"] for b in ("ALL", "P1", "P2")}
            # 逐用户（R17）
            per_user = []
            for r in rt:
                for i, du in enumerate(r["per_user_delta"]):
                    per_user.append({"plant": r["plant"], "ep": r["ep"], "i": i,
                                     "delta_user": float(du)})
            # 动作差（H-R24-2）
            diff = []
            for r in rt:
                cb = next(c for c in cells if c["plant"] == r["plant"]
                          and c["model"] == "belief_readout" and int(c["ep"]) == r["ep"])
                cw = next(c for c in cells if c["plant"] == r["plant"]
                          and c["model"] == "prior_only" and int(c["ep"]) == r["ep"])
                diff.append({"plant": r["plant"], "ep": r["ep"],
                             "action_diff_frac": action_diff_frac(cb["acts"], cw["acts"]),
                             "trace_same": cb["policy_trace_hash"] == cw["policy_trace_hash"]})
            dvec = [x["action_diff_frac"] for x in diff
                    if np.isfinite(x["action_diff_frac"])]
            ana = {"rows": rt, "rows_full": rT, "verdicts": vt, "verdicts_full": vT,
                   "agreement": ag, "null_diag": nd, "sens": sens,
                   "per_user": per_user, "action": {
                       "n_trace_same": int(sum(1 for x in diff if x["trace_same"])),
                       "n": len(diff),
                       "median_action_diff_frac": float(np.median(dvec)) if dvec else None,
                       "min": float(np.min(dvec)) if dvec else None,
                       "max": float(np.max(dvec)) if dvec else None}}
            print("[A_t-ALL] " + json.dumps(
                {k: vt["ALL"][k] for k in ("verdict", "n_pos", "n_neg", "n_beat",
                                           "binom_p_sign", "binom_p_neg", "binom_p_beat",
                                           "median_abs_delta", "median_floor_pair",
                                           "ratio_of_medians")}, ensure_ascii=False))
            print("[A_T-ALL] " + json.dumps(
                {k: vT["ALL"][k] for k in ("verdict", "n_pos", "n_neg", "n_beat",
                                           "binom_p_sign", "binom_p_beat",
                                           "ratio_of_medians")}, ensure_ascii=False))
            print("[VARIANT] " + json.dumps(ag, ensure_ascii=False))
            print("[NULL-DIAG] " + json.dumps(nd["N1"], ensure_ascii=False))
            print("[SENS] " + json.dumps(sens, ensure_ascii=False))
            print("[ACTION] " + json.dumps(ana["action"], ensure_ascii=False))
        except Exception as e:  # noqa: BLE001
            ana = {"analysis_error": repr(e), "traceback": traceback.format_exc()[-1200:]}
            print(f"[ERR] 分析段异常：{e!r}（90 格已落盘，不重跑；按 R14 报错不报假数字）")
            n_fail += 1
    else:
        print("[SKIP] 闸未全通过 ⇒ 按预注册不看主判")

    (outdir / "gates.json").write_text(json.dumps(
        {"r21_b1_b8": gate, "r21_base_ctrl_keys": base_keys,
         "r24_C1_C11": {k: v for k, v in ctrl.items() if k.startswith("C")},
         "I1": i1, "I2_bad": i2_bad},
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
        "protocol": "CW-POMDP-v1.1", "stage": "CW-R2.4",
        "kind": "NON-DELETIVE-AOI-PRIMARY-WITH-CONDITIONAL-NULL",
        "prereg": "docs/research/CW-R2.4-AoI主指标-预注册-v0.1.md",
        "hypothesis": "H-R2-0",
        "design": {"eps": EPS_ALL, "plants": PLANT_KEYS, "models": list(MODELS),
                   "seed_panel": SEED_PANEL, "seed_null": SEED_NULL,
                   "n_cells": len(cells), "n_null": len(nulls),
                   "new_plant_ep_units": 0,
                   "why_rerun": "R2.3 cells.json 缺 n_unserved_slots_by_user，无法重建非删除型均值"},
        "primary_A_t": ana.get("verdicts"),
        "co_report_A_T": ana.get("verdicts_full"),
        "variant_agreement": ana.get("agreement"),
        "null_diagnostics": ana.get("null_diag"),
        "alpha_sensitivity": ana.get("sens"),
        "action_diagnostic_H_R24_2": ana.get("action"),
        "per_user_R17": ({"n": len(ana.get("per_user") or []),
                          "n_user_delta_negative": int(sum(
                              1 for x in (ana.get("per_user") or [])
                              if x["delta_user"] < 0)),
                          "by_unit_user": ana.get("per_user")} if ana.get("per_user") else None),
        "decomposition_C11": ctrl["C11_decomposition_identity"],
        "ctrl_all_pass": all_pass,
        "identity_I1": i1,
        "identity_I2_n_bad": i2_bad,
        "objective": OBJECTIVE,
        "spec_fingerprint_P1": spec.fingerprint(),
        "sigma_meas": {p: float(PLANTS[p].sigma_meas) for p in PLANT_KEYS},
        "budget": {"panel_s": panel_s, "total_s": total_s,
                   "per_cell_s": [c["elapsed_s"] for c in cells]},
        "head": head, "src_hash": srchash,
        "n_checks_gates": gate["n_checks"], "n_fail_gates": gate["n_fail"],
        "n_fail_total": n_fail, "training": "none（本轮零训练）",
    }
    (outdir / "rows.json").write_text(json.dumps(
        {"rows_A_t": ana.get("rows") or [], "rows_A_T": ana.get("rows_full") or [],
         "inputs": {"r23": {"path": str((R23_DIR / "cells.json").relative_to(REPO)),
                            "sha256": _sha256(R23_DIR / "cells.json")}}},
        ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    (outdir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1,
                                                    default=float), encoding="utf-8")
    meta = {"command": "python -m experiments.cw_v1.r2_4_aoi --tag " + str(args.tag),
            "timestamp_local": datetime.now().astimezone().isoformat(),
            "python": os.sys.version.split()[0], "numpy": np.__version__,
            "head": head, "src_hash": srchash,
            "worktree_sha_r2_4": wt_sha, "blob_sha_r2_4": blob_sha,
            "blob_sha_note": (None if blob_sha else "运行前未入库 ⇒ 提交后补 _POST_COMMIT"),
            "git_dirty_before_run": dirty.splitlines(),
            "protocol": "CW-POMDP-v1.1", "event_pairing": "slot",
            "n_cells": len(cells), "n_null": len(nulls),
            "panel_elapsed_s": panel_s, "total_elapsed_s": total_s,
            "peak_rss_mb": round(rss, 1) if rss else None,
            "n_fail_total": n_fail, "training": "none"}
    (outdir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    print(f"[DONE] {outdir.name} n_fail={n_fail} total={total_s}s peak={meta['peak_rss_mb']}MB")
    if ana.get("verdicts"):
        print(json.dumps({"A_t": {k: v["verdict"] for k, v in ana["verdicts"].items()},
                          "A_T": {k: v["verdict"] for k, v in ana["verdicts_full"].items()},
                          "i1_ok": i1["ok"], "n_fail": n_fail}, ensure_ascii=False, indent=1))
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    import sys
    sys.exit(main())
