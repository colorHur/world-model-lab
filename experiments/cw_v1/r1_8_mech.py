# -*- coding: utf-8 -*-
"""CW-R1.8 机械量判据：把判据从 `cost_est/cost_true` 换到**机械量**，并补逐用户。

对应预注册：`docs/research/CW-R1.8-机械量判据-预注册-v0.1.md`（**看新数之前**写死）。

为什么需要（STATE §7.2 唯一主件）
--------------------------------
R1.7.8 的四口径矩阵证 `cost_est`/`cost_true` 在这块面板上**全部不可分辨**（8 个比值 <2），
且加 seed 提不了信噪比。⇒ 必须在**不经 `x̂`** 的量上重问一次。

本轮同时修旧量的两处**机械缺陷**（预注册 §2，均可机械证明）：
  D1 `mean_task_aoi` 是**末端单帧**快照（两个用户 ⇒ 两个整数）；
  D2 它用 `if a >= 0` **剔除**从未送达的用户 ⇒ 用户越被饿死越不进指标（R17 的机制级版本）。

入口结构
--------
- `G0` 合成语义闸：用**合成序列**证明 D1/D2 真实存在（不跑 episode）。
- `G1` 身份闸（U2）：重跑 `paced16|100|2001` / `dp16|100|2001`，必须与 R1.7.6 入库 cell
  的 `cost_est/cost_true/n_delivered/n_attempted/n_probe/policy_trace_hash` **逐位相同**
  ⇒ 证明新增字段是**纯记录**。
- `G2` 口径闸（U1）：`mean(aoi_final_by_user>=0)` 必须逐位等于旧 `mean_task_aoi`。
- `G3` 计数恒等式：`Σ n_delivered_by_user == n_delivered`、`Σ n_attempted_by_user == n_attempted`。
- 面板：`2 episodes × 4 seeds × {paced16, dp16}` = 16 格（复用 G1 的 2 格）。
- 分析：按预注册 §4 的 U3/U4/U5/U6 出判词。

★ 不改 `plan_ref` / `plan_dp` / `plan_gates` / 阈值 / 协议参数；不读 `N1/N2`。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
import tracemalloc
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from . import config as C
from . import plan_gates as G
from .plan_ref import make
from .runner import run_episode, check_identities

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results" / "cw_v1"

#: R1.7.6 入库证据（身份闸的对照物）
R176_DIR = RESULTS / "cw_r1_7_6_g4form_20261023T182640_f11ffdefc06c0201_7dd8ba47dede6ce3"
R176_CELLS = R176_DIR / "cells.json"

EPS = (100, 101)
SEEDS = (2001, 2002, 2003, 2004)
RHO = float(G.RHO_DEFAULT)      # 0.98，与 R1.7.6 面板一致
RATIO = 0.5
H_PLAN = 5
M_MAIN = 16
ARM = "probe"
TAIL_DP = {"tail": "dp", "dp_grid": "r173"}

#: 预注册 §4 的线（看数前写死）
LINE_EFFECT = 0.20      # |中位 r − 1| ≥ 0.20
N_BOOT = 20000
BOOT_SEED = 20261024
#: R1.7.8 的外生角色重标号（`paced16` 基线臂判定，8/8）
ROLE = {100: 1, 101: 0}   # ep → 贵用户下标


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


def _src_hash() -> str:
    h = hashlib.sha256()
    for p in sorted((REPO / "experiments" / "cw_v1").glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def _git(args: list[str]) -> str:
    return subprocess.run(["D:\\devtools\\Git\\cmd\\git.exe", *args],
                          cwd=str(REPO), capture_output=True, text=True).stdout.strip()


# ------------------------------------------------------------------ G0 合成语义闸
def _old_mean_aoi(aoi_vals: list[int]) -> float:
    """**逐字复制** `runner.py` 旧式（R1.7.6 时的实现），用来证明 D1/D2。"""
    return float(np.mean([a for a in aoi_vals if a >= 0])) if any(a >= 0 for a in aoi_vals) else -1.0


def _new_time_avg(hist: list[tuple[int, ...]], i: int) -> tuple[float | None, int, int]:
    col = [h[i] for h in hist]
    served = [a for a in col if a >= 0]
    return (float(np.mean(served)) if served else None,
            int(len(col) - len(served)),
            int(col[-1]) if col else -1)


def g0_synthetic_semantics() -> dict:
    """★ 合成数据自检（#103）：先按分布算期望，再验两个收敛。

    S1（D2）：用户 1 全期未送达 ⇒ 旧式**把它删掉**，答案 = 用户 0 的值；
             新式报 `n_never_served=1` 且该用户 `tavg=None`。
    S2（D1）：末端 vs 时间平均在同一序列上给出不同数（旧式取末端，新式取均值）。
    S3（阴性对照）：两个用户都正常送达时，两边**必须**一致（否则新式自己算错了）。
    """
    # S1：历史只有用户 0 送达
    hist1 = [(0, -1), (1, -1), (2, -1), (3, -1)]
    old1 = _old_mean_aoi([hist1[-1][0], hist1[-1][1]])
    t0, uns0, fin0 = _new_time_avg(hist1, 0)
    t1, uns1, fin1 = _new_time_avg(hist1, 1)
    s1 = {"old": old1, "tavg_u0": t0, "tavg_u1": t1, "unserved_u1": uns1,
          "final_u1": fin1,
          "pass": bool(old1 == 3.0 and t1 is None and uns1 == 4),
          "rule": "旧式把从未送达的用户从分母删掉（它只报用户 0 的 3.0），新式必须报 None+未服务槽数"}

    # S2：末端 1，时间平均 (0+2+5+1)/4 = 2.0
    hist2 = [(0, 4), (2, 5), (5, 6), (1, 7)]
    old2 = _old_mean_aoi([hist2[-1][0], hist2[-1][1]])
    a0, _, f0 = _new_time_avg(hist2, 0)
    s2 = {"old_terminal": old2, "tavg_u0": a0, "final_u0": f0,
          "pass": bool(old2 == 4.0 and a0 == 2.0 and f0 == 1),
          "rule": "末端快照 ≠ 时间平均（4.0 vs 2.0）：旧量丢掉了全部时间信息"}

    # S3：两用户都送达时，旧式（末端均值）与新式（末端均值）必须逐位相同
    hist3 = [(0, 3), (4, 2), (7, 9)]
    old3 = _old_mean_aoi([hist3[-1][0], hist3[-1][1]])
    f_0, _, fin3_0 = _new_time_avg(hist3, 0)
    f_1, _, fin3_1 = _new_time_avg(hist3, 1)
    new3 = float(np.mean([fin3_0, fin3_1]))
    s3 = {"old": old3, "new_final_mean": new3, "pass": bool(abs(old3 - new3) <= 1e-12),
          "rule": "阴性对照：都送达时两个口径必须一致（否则新式自己错了）"}

    return {"S1_drop_unserved": s1, "S2_terminal_vs_tavg": s2, "S3_control": s3,
            "pass": bool(s1["pass"] and s2["pass"] and s3["pass"])}


# ------------------------------------------------------------------ 跑一格
def _run(cfg, budget, ep: int, seed: int, tail: dict | None = None) -> dict:
    kw = dict(tail) if tail else {}
    pol = G._Rec(make(cfg, arm=ARM, h_plan=H_PLAN, m_samples=M_MAIN, seed=seed,
                      rho=RHO, pacing=True, **kw))
    r = run_episode(cfg, pol, split="dev", episode=ep, budget_total=budget, rho=RHO)
    ids = check_identities(r, cfg)
    return {
        "cost_est": float(r.cost_est_per_step),
        "cost_true": float(r.cost_true_per_step),
        "n_probe": int(r.n_probe), "n_attempted": int(r.n_attempted),
        "n_delivered": int(r.n_delivered),
        "mean_task_aoi_old": float(r.mean_task_aoi),
        "aoi_tavg_by_user": r.aoi_tavg_by_user,
        "n_unserved_slots_by_user": r.n_unserved_slots_by_user,
        "aoi_final_by_user": r.aoi_final_by_user,
        "aoi_max_by_user": r.aoi_max_by_user,
        "n_delivered_by_user": r.n_delivered_by_user,
        "n_attempted_by_user": r.n_attempted_by_user,
        "n_never_served": int(r.n_never_served),
        "policy_trace_hash": str(r.policy_trace_hash),
        "identities_ok": bool(ids.get("pass")),
        "identities_fail": [k for k, v in ids.get("items", {}).items() if not v],
        "elapsed_s": float(r.elapsed_s),
    }


# ------------------------------------------------------------------ 尺子
def sym(a: float, b: float) -> float:
    return 2.0 * abs(b - a) / (abs(a) + abs(b) + 1e-300)


def signed(a: float, b: float) -> float:
    return (1.0 if b >= a else -1.0) * sym(a, b)


def _boot_median_ci(vals: list[float]) -> dict:
    rng = np.random.default_rng(BOOT_SEED)
    v = np.asarray(vals, dtype=float)
    meds = np.array([np.median(rng.choice(v, size=len(v), replace=True))
                     for _ in range(N_BOOT)])
    lo, hi = float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))
    return {"n": int(len(v)), "median": float(np.median(v)), "ci95": [lo, hi],
            "excludes_zero": bool(lo > 0 or hi < 0),
            "n_distinct_medians": int(len(np.unique(np.round(meds, 12)))),
            "n_boot": N_BOOT, "boot_seed": BOOT_SEED}


# ------------------------------------------------------------------ 分析
def analyse(cells: dict) -> dict:
    """`cells` 键：`f"{arm}|{ep}|{seed}"`。全部按预注册 §4 的 U3–U6 判。"""
    out: dict = {"cells": {}, "per_user": {}, "floor": {}, "trade": {}}

    # ---- U6 地板：同臂（paced16）跨 seed 的 tavg 对称相对差
    floor_rows = []
    for ep in EPS:
        for i in (0, 1):
            vals = [cells[f"paced16|{ep}|{s}"]["aoi_tavg_by_user"][i] for s in SEEDS]
            if any(v is None for v in vals):
                floor_rows.append({"episode": ep, "user": i, "status": "SKIP-NO-OBJECT",
                                   "vals": vals})
                continue
            pairs = [sym(vals[a], vals[b]) for a in range(4) for b in range(a + 1, 4)]
            floor_rows.append({"episode": ep, "user": i, "vals": vals,
                               "median_sym": float(np.median(pairs)), "n_pairs": len(pairs)})
    out["floor"] = {"rows": floor_rows,
                    "rule": "同臂跨 seed = 噪声地板（不配对、不换臂）；与效应用同一 sym 尺度"}

    # ---- U3 主判据：逐 (ep, user) 的 dp/paced 配对
    for ep in EPS:
        for i in (0, 1):
            r_vals, e_vals, rows = [], [], []
            for s in SEEDS:
                a = cells[f"paced16|{ep}|{s}"]["aoi_tavg_by_user"][i]
                b = cells[f"dp16|{ep}|{s}"]["aoi_tavg_by_user"][i]
                if a is None or b is None:
                    rows.append({"episode": ep, "user": i, "seed": s,
                                 "paced": a, "dp": b, "note": "SKIP-NO-OBJECT"})
                    continue
                r_vals.append(b / a if a else float("inf"))
                e_vals.append(signed(a, b))
                rows.append({"episode": ep, "user": i, "seed": s, "paced": a, "dp": b,
                             "r": r_vals[-1], "e": e_vals[-1]})
            if len(r_vals) < 4:
                out["per_user"][f"{ep}|u{i}"] = {"status": "SKIP-NO-OBJECT", "rows": rows,
                                                 "note": "有格未服务（tavg=None）⇒ 不承载结论"}
                continue
            fl = next((f.get("median_sym") for f in floor_rows
                       if f["episode"] == ep and f["user"] == i), None)
            n_pos = sum(1 for e in e_vals if e > 0)
            n_neg = sum(1 for e in e_vals if e < 0)
            sign_all = bool(n_pos == 4 or n_neg == 4)
            boot = _boot_median_ci(e_vals)
            med_r = float(np.median(r_vals))
            eff_ok = bool(abs(med_r - 1.0) >= LINE_EFFECT)
            r_over_floor = (abs(boot["median"]) / fl) if fl else float("inf")
            floor_ok = bool(fl is not None and abs(boot["median"]) > fl)
            sep = bool(sign_all and boot["excludes_zero"] and eff_ok and floor_ok)
            status = ("DP-WORSE" if med_r > 1 else "DP-BETTER") if sep else "UNRESOLVED"
            out["per_user"][f"{ep}|u{i}"] = {
                "status": status, "n": 4, "median_r": med_r,
                "n_dp_worse": n_pos, "n_dp_better": n_neg, "sign_all_same": sign_all,
                "boot_effect": boot, "floor_median_sym": fl,
                "effect_over_floor": r_over_floor,
                "pass_sign": sign_all, "pass_boot": boot["excludes_zero"],
                "pass_effect_line": eff_ok, "pass_floor": floor_ok,
                "role": "EXPENSIVE" if ROLE[ep] == i else "CHEAP",
                "rows": rows}

    # ---- U4 交易：同 (ep) 两用户方向相反
    for ep in EPS:
        a = out["per_user"].get(f"{ep}|u0", {})
        b = out["per_user"].get(f"{ep}|u1", {})
        dirs = {}
        for u, d in (("u0", a), ("u1", b)):
            if d.get("status", "").startswith("DP-"):
                dirs[u] = d["status"]
            elif d.get("rows"):
                rr = [x["r"] for x in d["rows"] if "r" in x]
                dirs[u] = ("DP-WORSE" if np.median(rr) > 1 else "DP-BETTER") if rr else "ND"
            else:
                dirs[u] = "ND"
        n_opp = 0
        for s in SEEDS:
            r0 = next((x.get("r") for x in a.get("rows", []) if x["seed"] == s), None)
            r1 = next((x.get("r") for x in b.get("rows", []) if x["seed"] == s), None)
            if r0 is not None and r1 is not None and (r0 - 1) * (r1 - 1) < 0:
                n_opp += 1
        out["trade"][f"ep{ep}"] = {"dirs": dirs, "n_seed_pairs_opposite": n_opp,
                                   "is_trade": bool(n_opp >= 2)}

    # ---- U5 副量同向性：delivered 的比值符号必须与 aoi 比值相反
    u5_rows = []
    for ep in EPS:
        for i in (0, 1):
            rs, rd = [], []
            for s in SEEDS:
                a = cells[f"paced16|{ep}|{s}"]["aoi_tavg_by_user"][i]
                b = cells[f"dp16|{ep}|{s}"]["aoi_tavg_by_user"][i]
                da = cells[f"paced16|{ep}|{s}"]["n_delivered_by_user"][i]
                db = cells[f"dp16|{ep}|{s}"]["n_delivered_by_user"][i]
                if a is None or b is None:
                    continue
                rs.append(b / a)
                rd.append(db / da if da else float("inf"))
            if not rs:
                u5_rows.append({"episode": ep, "user": i, "status": "SKIP-NO-OBJECT"})
                continue
            m_r, m_d = float(np.median(rs)), float(np.median(rd))
            consistent = bool((m_r - 1) * (m_d - 1) < 0)
            u5_rows.append({"episode": ep, "user": i,
                            "median_r_aoi": m_r, "median_r_delivered": m_d,
                            "consistent": consistent,
                            "rule": "送达少 ⇔ 陈旧高（同向）；同号 ⇒ MECH-INCONSISTENT"})
    out["U5_delivered_consistency"] = {
        "rows": u5_rows,
        "n_inconsistent": int(sum(1 for r in u5_rows if r.get("consistent") is False))}

    # ---- R17：绝对量（不许只报比值）
    abs_rows = {}
    for arm in ("paced16", "dp16"):
        cells_a = [v for k, v in cells.items() if k.startswith(arm)]
        abs_rows[arm] = {
            "n": len(cells_a),
            "aoi_tavg_by_user_median": [
                float(np.median([c["aoi_tavg_by_user"][i] for c in cells_a])) for i in (0, 1)],
            "aoi_max_by_user_median": [
                float(np.median([c["aoi_max_by_user"][i] for c in cells_a])) for i in (0, 1)],
            "n_delivered_by_user_median": [
                float(np.median([c["n_delivered_by_user"][i] for c in cells_a])) for i in (0, 1)],
            "n_attempted_by_user_median": [
                float(np.median([c["n_attempted_by_user"][i] for c in cells_a])) for i in (0, 1)],
            "n_never_served": int(sum(c["n_never_served"] for c in cells_a)),
            "mean_task_aoi_old_median": float(np.median([c["mean_task_aoi_old"] for c in cells_a])),
            "mean_task_aoi_old_values": sorted(c["mean_task_aoi_old"] for c in cells_a),
        }
    out["R17_absolute"] = abs_rows
    return out


def verdict_of(ana: dict) -> dict:
    st = {k: v.get("status") for k, v in ana["per_user"].items()}
    n_unres = sum(1 for v in st.values() if v == "UNRESOLVED")
    trades = [k for k, v in ana["trade"].items() if v["is_trade"]]
    n_inc = ana["U5_delivered_consistency"]["n_inconsistent"]
    if n_inc >= 2:
        v, reason = "MECH-INCONSISTENT", f"{n_inc} 个 (ep,user) 格的 aoi 与送达计数同号 ⇒ aoi 暂不可当唯一任务读数"
    elif len(trades) >= 1:
        v, reason = "MECH-TRADE", f"{trades} 上两用户方向相反（≥2 个 seed 对）⇒ 只能写交易"
    elif n_unres >= 3:
        v, reason = "MECH-UNRESOLVED", f"{n_unres}/4 格未分辨 ⇒ 机械量在这块面板上同样测不出"
    else:
        dirs = sorted({s for s in st.values() if s and s.startswith("DP-")})
        v, reason = "MECH-" + "/".join(dirs) if dirs else "MECH-UNRESOLVED", "逐格可分"
    return {"verdict": v, "reason": reason, "per_user_status": st,
            "n_unresolved": n_unres, "trades": trades,
            "n_u5_inconsistent": n_inc,
            "thresholds": {"LINE_EFFECT": LINE_EFFECT, "N_BOOT": N_BOOT,
                           "BOOT_SEED": BOOT_SEED, "require_sign": "4/4"}}


# ------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    tracemalloc.start()
    git_head = _git(["rev-parse", "HEAD"])
    dirty = _git(["status", "--porcelain"])
    srchash = _src_hash()

    cfg = C.CWConfig(plant=C.bounded_plant())
    budget = cfg.protocol.budget(RATIO)

    gates: dict = {}
    # ---- G0 合成语义闸（不跑 episode）
    gates["G0_synthetic_semantics"] = g0_synthetic_semantics()

    # ---- G1 身份闸（U2）：先跑 1 paced + 1 dp，核"纯记录"
    ref = json.loads(R176_CELLS.read_text(encoding="utf-8"))
    cells: dict = {}
    t_probe = time.perf_counter()
    first = {}
    for tag, tail in (("paced16", None), ("dp16", TAIL_DP)):
        c = _run(cfg, budget, 100, 2001, tail=tail)
        cells[f"{tag}|100|2001"] = c
        first[tag] = c
    probe_wall = time.perf_counter() - t_probe
    cmp_rows = []
    for tag in ("paced16", "dp16"):
        k = f"{tag}|100|2001"
        a, b = ref[k], first[tag]
        rows = {
            "cost_est": abs(a["cost_est"] - b["cost_est"]),
            "cost_true": abs(a["cost_true"] - b["cost_true"]),
            "n_delivered_int_eq": int(a["n_delivered"]) == int(b["n_delivered"]),
            "n_attempted_int_eq": int(a["n_attempted"]) == int(b["n_attempted"]),
            "n_probe_int_eq": int(a["n_probe"]) == int(b["n_probe"]),
            "trace_eq": str(a["policy_trace_hash"]) == str(b["policy_trace_hash"]),
            "mean_task_aoi_old_abs_diff": abs(float(a["mean_task_aoi"]) - b["mean_task_aoi_old"]),
        }
        ok = (rows["cost_est"] <= 1e-12 and rows["cost_true"] <= 1e-12
              and rows["n_delivered_int_eq"] and rows["n_attempted_int_eq"]
              and rows["n_probe_int_eq"] and rows["trace_eq"])
        cmp_rows.append({"cell": k, **rows, "pass": bool(ok)})
    gates["G1_identity_pure_recording"] = {
        "rows": cmp_rows, "pass": bool(all(r["pass"] for r in cmp_rows)),
        "rule": "重跑必须与 R1.7.6 入库逐位相同 ⇒ 新增字段是纯记录；不等则停（预注册 N3）"}

    # ---- G2 口径闸（U1）+ G3 计数恒等式
    u1_rows = []
    for k, c in cells.items():
        fin = [x for x in c["aoi_final_by_user"] if x >= 0]
        new_old = float(np.mean(fin)) if fin else -1.0
        u1_rows.append({"cell": k, "old": c["mean_task_aoi_old"], "new_final_mean": new_old,
                        "abs_diff": abs(c["mean_task_aoi_old"] - new_old)})
    gates["G2_u1_recompute_old"] = {
        "rows": u1_rows,
        "pass": bool(all(r["abs_diff"] <= 1e-12 for r in u1_rows)),
        "rule": "aoi_final_by_user 的 >=0 分量均值必须逐位等于旧 mean_task_aoi"}

    id_rows = []
    for k, c in cells.items():
        id_rows.append({"cell": k,
                        "sum_delivered": int(sum(c["n_delivered_by_user"])),
                        "n_delivered": int(c["n_delivered"]),
                        "sum_attempted": int(sum(c["n_attempted_by_user"])),
                        "n_attempted": int(c["n_attempted"]),
                        "pass": bool(sum(c["n_delivered_by_user"]) == int(c["n_delivered"])
                                     and sum(c["n_attempted_by_user"]) == int(c["n_attempted"]))})
    gates["G3_count_identities"] = {"rows": id_rows,
                                    "pass": bool(all(r["pass"] for r in id_rows)),
                                    "rule": "逐用户计数之和必须等于 runner 的总计数"}

    # ---- 预算纪律（#98）：先按 2 格实测外推，再决定是否跑满
    per_cell = probe_wall / 2.0
    n_rest = 14
    est_rest = per_cell * n_rest
    budget_note = {"probe_wall_s": round(probe_wall, 2), "per_cell_s": round(per_cell, 2),
                   "n_rest_cells": n_rest, "est_rest_s": round(est_rest, 1),
                   "rule": "单格 > 40 s 则缩到 2 episodes × 2 seeds"}
    shrink = bool(per_cell > 40.0)
    seeds_use = SEEDS[:2] if shrink else SEEDS

    # ---- 面板
    for ep in EPS:
        for s in seeds_use:
            if f"paced16|{ep}|{s}" in cells:
                continue
            cells[f"paced16|{ep}|{s}"] = _run(cfg, budget, ep, s, tail=None)
    for ep in EPS:
        for s in seeds_use:
            if f"dp16|{ep}|{s}" in cells:
                continue
            cells[f"dp16|{ep}|{s}"] = _run(cfg, budget, ep, s, tail=TAIL_DP)

    wall = time.perf_counter() - t0
    peak_mib = tracemalloc.get_traced_memory()[1] / 2 ** 20
    tracemalloc.stop()

    # 数据完整性：每个 (ep,seed) 必须两臂都有
    missing = [f"{ep}|{s}" for ep in EPS for s in seeds_use
               if f"paced16|{ep}|{s}" not in cells or f"dp16|{ep}|{s}" not in cells]
    gates["G4_panel_complete"] = {"missing": missing, "pass": bool(not missing),
                                  "seeds_used": list(seeds_use), "shrunk": shrink}

    # ---- G5 记账恒等式（每格）
    bad = {k: c["identities_fail"] for k, c in cells.items() if not c["identities_ok"]}
    gates["G5_identities_all_cells"] = {
        "n_cells": len(cells), "bad_cells": bad, "pass": bool(not bad),
        "rule": "runner.check_identities 全过；不过则数不可用（与判词无关，是记账错）"}

    ana = analyse(cells) if gates["G4_panel_complete"]["pass"] else {}
    ver = verdict_of(ana) if ana else {"verdict": "NO-VERDICT", "reason": "面板不完整"}
    if not gates["G1_identity_pure_recording"]["pass"]:
        ver = {"verdict": "STOP-IMPURE", "reason": "身份闸不过：改动不是纯记录（预注册 N3）"}

    n_fail = sum(1 for g in gates.values() if g.get("pass") is False)
    out = {"gates": gates, "budget": budget_note, "cells": cells, "analysis": ana,
           "verdict": ver, "n_fail": int(n_fail),
           "elapsed_s": round(wall, 1), "peak_mib": round(peak_mib, 1)}

    print(json.dumps({"gates_pass": {k: v.get("pass") for k, v in gates.items()},
                      "verdict": ver["verdict"], "reason": ver["reason"],
                      "n_fail": n_fail, "elapsed_s": round(wall, 1),
                      "peak_mib": round(peak_mib, 1)}, ensure_ascii=False, indent=1))

    if args.no_write:
        print("[NO-WRITE] 未写证据目录")
        return 0 if n_fail == 0 else 1

    d = RESULTS / f"cw_r1_8_mech_{_now_utc()}_{srchash}"
    d.mkdir(parents=True, exist_ok=False)
    (d / "cells.json").write_text(json.dumps(cells, ensure_ascii=False, indent=1), encoding="utf-8")
    (d / "gates.json").write_text(json.dumps(gates, ensure_ascii=False, indent=1), encoding="utf-8")
    (d / "floor.json").write_text(json.dumps(ana.get("floor", {}), ensure_ascii=False, indent=1),
                                  encoding="utf-8")
    (d / "verdict.json").write_text(json.dumps(ver, ensure_ascii=False, indent=1), encoding="utf-8")
    (d / "analysis.json").write_text(json.dumps(ana, ensure_ascii=False, indent=1), encoding="utf-8")
    meta = {
        "round": "CW-R1.8", "prereg": "docs/research/CW-R1.8-机械量判据-预注册-v0.1.md",
        "protocol": "CW-POMDP-v1.1", "git_head": git_head,
        "git_status_porcelain": dirty, "src_hash16": srchash,
        "started_utc": _now_utc(), "elapsed_s": round(wall, 1),
        "peak_mib": round(peak_mib, 1), "budget": budget_note,
        "panel": {"eps": list(EPS), "seeds": list(seeds_use), "arms": ["paced16", "dp16"],
                  "h_plan": H_PLAN, "m": M_MAIN, "rho": RHO, "ratio": RATIO,
                  "tail_dp": TAIL_DP},
        "n_run_episode": len(cells), "reference": str(R176_CELLS.relative_to(REPO)),
        "python": "miniconda ai-lab 3.11 / numpy", "note": "预注册 §4 判据，未改任何阈值",
    }
    (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[EVIDENCE] {d}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
