#!/usr/bin/env python
"""CW-R1.7.7 · 逐用户判据（零成本分解）

预注册：`docs/research/CW-R1.7.7-逐用户判据-预注册-v0.1.md`（含 §3.1 卡内修订 1 的 `G2b`）

本入口**不跑任何 episode**，只读 R1.7.6 已入库的 `cells.json` / `acts.json` / `stats.json`，
回答一件事：R1.7.6 §7.1-4 的聚合结论「`Δcost_est` 符号 3 正 5 负、CI 含 0」到底是
①两个用户反向抵消 ②每用户都无方向 ③一个稳一个不稳 ④同向且稳。

用法：
    python experiments/cw_v1/r1_7_7_peruser.py            # 写证据
    python experiments/cw_v1/r1_7_7_peruser.py --no-write # 自检，不落盘
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SRC176 = (REPO / "results" / "cw_v1"
          / "cw_r1_7_6_g4form_20261023T182640_f11ffdefc06c0201_7dd8ba47dede6ce3")
TZ = timezone(timedelta(hours=8))

EPS = (100, 101)
SEEDS = (2001, 2002, 2003, 2004)
PAIRS = [(ep, s) for ep in EPS for s in SEEDS]

LINE_CONSISTENT = 7 / 8      # 预注册 §4：一致率阈值（照抄 STATE §7.2）
TOL_IDENT = 1e-12            # 恒等式 / 对账容差


# ------------------------------------------------------------------ 工具
def sym(a: float, b: float) -> float:
    """对称相对差 —— 与 `r1_7_6_g4form.py` 逐字一致。"""
    return 2.0 * abs(b - a) / (abs(a) + abs(b) + 1e-300)


def signed(a: float, b: float) -> float:
    """带符号的对称相对差：`b` 相对 `a` —— 与 `r1_7_6_g4form.py` 逐字一致。"""
    return (1.0 if b >= a else -1.0) * sym(a, b)


def rel_hi(c_lo: float, c_hi: float) -> float:
    """`G4` 原式 —— 与 `r1_7_6_g4form.py` 逐字一致。"""
    return abs(c_hi - c_lo) / max(c_lo, 1e-12)


def agree(a_acts: list, b_acts: list) -> tuple[float, int]:
    """动作一致率 —— 与 `r1_7_6_g4form.py::_agree` 逐字一致。"""
    if len(a_acts) != len(b_acts):
        raise ValueError(f"动作序列长度不等 {len(a_acts)} vs {len(b_acts)}")
    same = sum(1 for x, y in zip(a_acts, b_acts)
               if (x[0], x[1]) == (y[0], y[1]))
    return same / max(len(a_acts), 1), len(a_acts)


def sign_test_p(k: int, n: int) -> float:
    """精确符号检验（两侧，H0: p=0.5）。`n` 为剔除 TIE 后的对数。"""
    if n <= 0:
        return 1.0
    k = max(k, n - k)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / (2.0 ** n)
    return float(min(1.0, 2.0 * tail))


def sha256_16(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def git_head() -> dict:
    cands = ["git", r"D:/devtools/Git/cmd/git.exe"]
    for exe in cands:
        try:
            head = subprocess.run([exe, "rev-parse", "--short", "HEAD"],
                                  cwd=str(REPO), capture_output=True,
                                  text=True, timeout=20).stdout.strip()
            dirty = subprocess.run([exe, "status", "--porcelain"],
                                   cwd=str(REPO), capture_output=True,
                                   text=True, timeout=20).stdout.strip()
            return {"head": head, "dirty_paths": dirty.splitlines()}
        except Exception:  # noqa: BLE001
            continue
    return {"head": None, "dirty_paths": [], "note": "git 不可用"}


# ------------------------------------------------------------------ 分类器
def consistency(deltas: list[float], n_denom: int) -> dict:
    """预注册 §4：`c_u = max(#pos,#neg)/8`，`TIE` 占分母但不进分子。"""
    n_pos = sum(1 for d in deltas if d > 0)
    n_neg = sum(1 for d in deltas if d < 0)
    n_tie = sum(1 for d in deltas if d == 0)
    c = max(n_pos, n_neg) / max(n_denom, 1)
    dom = 0 if n_pos == n_neg else (1 if n_pos > n_neg else -1)
    n_eff = n_pos + n_neg
    k = max(n_pos, n_neg)
    return {"n_pos": n_pos, "n_neg": n_neg, "n_tie": n_tie,
            "n_denom": n_denom, "consistency": float(c),
            "dominant_sign": int(dom), "n_eff": n_eff,
            "sign_test_p": sign_test_p(k, n_eff)}


def classify(d0: list[float], d1: list[float]) -> dict:
    """预注册 §4 的四分支（互斥，按序命中即停）。"""
    k0, k1 = consistency(d0, len(d0)), consistency(d1, len(d1))
    c0, c1 = k0["consistency"], k1["consistency"]
    ok0, ok1 = c0 >= LINE_CONSISTENT, c1 >= LINE_CONSISTENT
    if ok0 and ok1 and k0["dominant_sign"] * k1["dominant_sign"] < 0:
        v = "AGGREGATE-CANCELS"
    elif ok0 and ok1 and k0["dominant_sign"] * k1["dominant_sign"] > 0:
        v = "SAME-DIRECTION"
    elif ok0 != ok1:
        v = "ONE-SIDED-AT-8"
    else:
        v = "NO-DIRECTION-AT-8"
    return {"verdict": v, "user0": k0, "user1": k1,
            "line_consistent": LINE_CONSISTENT}


# ------------------------------------------------------------------ 主流程
def run(no_write: bool) -> int:
    t0 = datetime.now(TZ)
    cells = json.loads((SRC176 / "cells.json").read_text(encoding="utf-8"))
    acts = json.loads((SRC176 / "acts.json").read_text(encoding="utf-8"))
    stats176 = json.loads((SRC176 / "stats.json").read_text(encoding="utf-8"))

    key = lambda arm, ep, s: f"{arm}|{ep}|{s}"  # noqa: E731
    n_users = len(cells[key("dp16", 100, 2001)]["cost_est_by_user"])
    gates: dict = {}

    # ---------------- G0 恒等式
    worst0 = 0.0
    for k, c in cells.items():
        s = float(sum(c["cost_est_by_user"]))
        worst0 = max(worst0, abs(c["cost_est"] - s) / max(abs(c["cost_est"]), 1e-300))
    gates["G0_identity_cost_est_eq_sum_by_user"] = {
        "pass": bool(worst0 <= TOL_IDENT), "max_rel_err": worst0,
        "n_cells": len(cells), "n_users": n_users, "tol": TOL_IDENT,
        "rule": "cost_est 必须是 cost_est_by_user 的同一分解（逐位）"}

    # ---------------- G1 配对同一性
    mism = [f"{ep}|{s}" for ep, s in PAIRS
            if cells[key("dp16", ep, s)]["stream_ids"]["spawn_keys"]
            != cells[key("paced16", ep, s)]["stream_ids"]["spawn_keys"]]
    gates["G1_pairing_same_source"] = {
        "pass": bool(not mism), "n_pairs": len(PAIRS), "mismatched": mism,
        "rule": "同 (ep,seed) 下 dp16 与 paced16 外生流 spawn_keys 必须逐字相同"}

    # ---------------- G2 复算 B2（paced16 vs paced32）
    rels, ags = [], []
    for ep, s in PAIRS:
        a, b = cells[key("paced16", ep, s)], cells[key("paced32", ep, s)]
        rels.append(rel_hi(a["cost_est"], b["cost_est"]))
        ag, _ = agree(acts[key("paced16", ep, s)], acts[key("paced32", ep, s)])
        ags.append(ag)
    got_rel, got_ag = float(np.median(rels)), float(np.median(ags))
    ref = stats176["B2_g4_original_form_cost_est"]
    d_rel = abs(got_rel - ref["rel_median"])
    d_ag = abs(got_ag - ref["agree_median"])
    gates["G2_reproduce_B2"] = {
        "pass": bool(d_rel <= TOL_IDENT and d_ag <= TOL_IDENT),
        "got": {"rel_median": got_rel, "agree_median": got_ag},
        "ref": {"rel_median": ref["rel_median"], "agree_median": ref["agree_median"]},
        "abs_diff": {"rel": d_rel, "agree": d_ag}, "tol": TOL_IDENT,
        "rule": "从 cells.json+acts.json 复算的 B2 必须与 R1.7.6 入库值逐位相同"}

    # ---------------- G2b 复算 B3（dp16 vs paced16，本轮对象）
    eff = [signed(cells[key("paced16", ep, s)]["cost_est"],
                  cells[key("dp16", ep, s)]["cost_est"]) for ep, s in PAIRS]
    floor_pooled = float(stats176["B1_floor"]["pooled"]["median_sym_est"])
    med_eff = float(np.median(np.abs(np.asarray(eff))))
    R_got = med_eff / floor_pooled
    sign_frac_got = max(sum(1 for e in eff if e > 0), sum(1 for e in eff if e < 0)) / len(eff)
    ref3 = stats176["B3_resolution_est"]
    dR = abs(R_got - ref3["R"])
    dsf = abs(sign_frac_got - ref3["sign_frac"])
    dme = abs(med_eff - ref3["median_abs_effect"])
    gates["G2b_reproduce_B3"] = {
        "pass": bool(dR <= TOL_IDENT and dsf <= TOL_IDENT and dme <= TOL_IDENT),
        "got": {"R": R_got, "sign_frac": sign_frac_got, "median_abs_effect": med_eff},
        "ref": {"R": ref3["R"], "sign_frac": ref3["sign_frac"],
                "median_abs_effect": ref3["median_abs_effect"]},
        "abs_diff": {"R": dR, "sign_frac": dsf, "median_abs_effect": dme},
        "n_pos": int(sum(1 for e in eff if e > 0)),
        "n_neg": int(sum(1 for e in eff if e < 0)),
        "tol": TOL_IDENT,
        "rule": "本轮要分解的『3 正 5 负』必须与 R1.7.6 §7.1-4 是同一个数"}

    # ---------------- 逐用户主分析
    dlt = {u: [] for u in range(n_users)}
    rows = []
    for ep, s in PAIRS:
        a, b = cells[key("paced16", ep, s)], cells[key("dp16", ep, s)]
        row = {"episode": ep, "seed": s}
        for u in range(n_users):
            pa, pb = a["cost_est_by_user"][u], b["cost_est_by_user"][u]
            du = pb - pa
            dlt[u].append(du)
            row[f"u{u}"] = {"paced": pa, "dp": pb, "delta": du,
                            "ratio": (pb / pa) if pa > 0 else None,
                            # ★ 自伤 #2（本轮，落盘前自检抓住）：地板是**相对**量，
                            #   效应必须同尺度。这里存每个配对的相对效应，供 `R_u_vs_floor` 用。
                            "rel_effect": sym(pa, pb)}
        rows.append(row)

    per_user = {}
    for u in range(n_users):
        ds = dlt[u]
        cons = consistency(ds, len(ds))
        ratios = [r[f"u{u}"]["ratio"] for r in rows if r[f"u{u}"]["ratio"] is not None]
        r_med = float(np.median(ratios)) if ratios else None
        r_cross = (float(np.median([r[f"u{u}"]["dp"] for r in rows]))
                   / float(np.median([r[f"u{u}"]["paced"] for r in rows])))
        per_user[f"u{u}"] = {
            **cons,
            "delta_median": float(np.median(ds)),
            "delta_median_abs": float(np.median(np.abs(np.asarray(ds)))),
            "rel_effect_median_abs": float(np.median(
                [abs(r[f"u{u}"]["rel_effect"]) for r in rows])),
            "paced_median": float(np.median([r[f"u{u}"]["paced"] for r in rows])),
            "dp_median": float(np.median([r[f"u{u}"]["dp"] for r in rows])),
            "R_u_median_of_ratios": r_med,
            "R_u_ratio_of_medians": r_cross,
            "ratio_stable": bool(r_med is not None and r_cross is not None
                                 and (r_med - 1) * (r_cross - 1) > 0),
            "deltas": ds}

    # ---------------- 逐用户地板（跨 seed，同 ep）
    floor_by_arm: dict = {}
    for arm in ("paced16", "dp16"):
        arm_floor = {}
        for u in range(n_users):
            vals = []
            for ep in EPS:
                for i in range(len(SEEDS)):
                    for j in range(i + 1, len(SEEDS)):
                        a = cells[key(arm, ep, SEEDS[i])]["cost_est_by_user"][u]
                        b = cells[key(arm, ep, SEEDS[j])]["cost_est_by_user"][u]
                        vals.append(sym(a, b))
            arm_floor[f"u{u}"] = {"n": len(vals),
                                  "median": float(np.median(vals)) if vals else None}
        floor_by_arm[arm] = arm_floor
    floor_final = {}
    for u in range(n_users):
        m = [floor_by_arm[arm][f"u{u}"]["median"] for arm in ("paced16", "dp16")]
        floor_final[f"u{u}"] = {"paced16": m[0], "dp16": m[1],
                                "taken_max": float(max(m)),
                                "which": "dp16" if m[1] >= m[0] else "paced16"}
        # R_u 与地板之比（**同尺度**：分子分母都是 `sym` 相对量）。只报不判：
        # 8 对定不出方向时比值同样不可引用，且地板走的是"跨 seed"轴、效应走的是"跨臂"轴。
        e = per_user[f"u{u}"]["rel_effect_median_abs"]
        per_user[f"u{u}"]["R_u_vs_floor"] = (e / floor_final[f"u{u}"]["taken_max"]
                                            if floor_final[f"u{u}"]["taken_max"] > 0 else None)

    # ---------------- G3 分辨力（活体对照，答案已知）
    # ★ 自伤 #1（本轮，`--no-write` 抓住，未落盘）：原 `E_tie_all` 的**期望值**写成
    #   `NO-DIRECTION-AT-8`，实测给 `ONE-SIDED-AT-8` —— 不是分类器缺陷：§4 规定
    #   `c_u = max(#pos,#neg)/8`，全 TIE 用户 `c_0 = 0` ⇒ 落第三分支。错在**我**把
    #   "该用户无方向"误当成"整体无方向"。处置：**不放宽**，把 `E` 换成两条**更严**的
    #   机械断言（直接钉 TIE 记账），不再依赖判词命名。
    one = [1.0] * len(PAIRS)
    alt = [1.0 if i % 2 == 0 else -1.0 for i in range(len(PAIRS))]
    five_neg = [-1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0]  # 5 负 3 正 ⇒ c=0.625
    controls = {
        "A_opposite": {"expect": "AGGREGATE-CANCELS",
                       "got": classify(one, [-x for x in one])["verdict"]},
        "B_same": {"expect": "SAME-DIRECTION",
                   "got": classify([-x for x in one], [-x for x in one])["verdict"]},
        "C_no_direction": {"expect": "NO-DIRECTION-AT-8",
                           "got": classify(alt, [1.0, 1.0, -1.0, -1.0,
                                                 1.0, 1.0, -1.0, -1.0])["verdict"]},
        "D_one_sided_5of8": {"expect": "ONE-SIDED-AT-8",
                             "got": classify(one, five_neg)["verdict"]},
        "E1_tie_all_accounting": {"expect": {"n_tie": 8, "n_pos": 0, "n_neg": 0,
                                             "consistency": 0.0},
                                  "got": consistency([0.0] * len(PAIRS), len(PAIRS))},
        "E2_tie_counts_in_denom": {"expect": 1.0 / 8,
                                   "got": consistency([1.0] + [0.0] * 7, len(PAIRS))
                                   ["consistency"]},
    }
    for k, v in controls.items():
        if isinstance(v["expect"], dict):
            v["pass"] = bool(all(v["got"].get(a) == b for a, b in v["expect"].items()))
        elif isinstance(v["expect"], float):
            v["pass"] = bool(abs(v["got"] - v["expect"]) <= 1e-15)
        else:
            v["pass"] = bool(v["got"] == v["expect"])
    gates["G3_classifier_discriminative"] = {
        "pass": bool(all(v["pass"] for v in controls.values())),
        "controls": controls,
        "rule": "五个已知答案的合成输入必须各给出对应判词；任一不中 ⇒ 分类器无分辨力"}

    # ---------------- G4 地板自检（自己对自己必为 0）
    self_zero = all(sym(x, x) == 0.0 for x in
                    [cells[key("dp16", 100, 2001)]["cost_est_by_user"][u]
                     for u in range(n_users)])
    gates["G4_floor_self_zero"] = {"pass": bool(self_zero),
                                  "rule": "同值与自身的对称相对差必须恒为 0"}

    # ---------------- 聚合符号的机械来源（G0 恒等式的直接推论）
    # `cost_est == Σ_u cost_est_by_user` ⇒ `Δ_total == Σ_u Δ_u` ⇒ 聚合符号 = sign(Δ_0+Δ_1)，
    # 不是"符号投票"。若两者计数不等 ⇒ 我对聚合量的理解有误。
    tot = [sum(r[f"u{u}"]["delta"] for u in range(n_users)) for r in rows]
    agg_pos = sum(1 for t in tot if t > 0)
    agg_neg = sum(1 for t in tot if t < 0)
    ref_pos = int(sum(1 for e in eff if e > 0))
    ref_neg = int(sum(1 for e in eff if e < 0))
    gates["G5_aggregate_sign_is_sum_not_vote"] = {
        "pass": bool(agg_pos == ref_pos and agg_neg == ref_neg),
        "delta_sum": {"n_pos": agg_pos, "n_neg": agg_neg},
        "B3_effect": {"n_pos": ref_pos, "n_neg": ref_neg},
        "rule": "聚合 Δ 的符号分布必须等于 Σ_u Δ_u 的符号分布（不是两用户符号投票相加）"}

    gate_all = all(bool(g["pass"]) for g in gates.values())

    # ---------------- 判词
    cls = classify(dlt[0], dlt[1]) if n_users == 2 else {"verdict": "VOID-NU"}
    if not gate_all:
        verdict = "VOID-GATE"
        failed = [k for k, g in gates.items() if not g["pass"]]
    else:
        verdict = cls["verdict"]
        failed = []

    # ---------------- 写证据
    src = Path(__file__).resolve()
    meta = {
        "exp": "CW-R1.7.7 逐用户判据（零成本分解）",
        "prereg": "docs/research/CW-R1.7.7-逐用户判据-预注册-v0.1.md",
        "started_at": t0.isoformat(), "tz": "Asia/Shanghai (+08:00)",
        "run_episode_calls": 0,
        "src": str(src.relative_to(REPO)), "src_hash": sha256_16(src),
        "inputs": {f.name: {"path": str(f.relative_to(REPO)), "sha256_16": sha256_16(f)}
                   for f in (SRC176 / "cells.json", SRC176 / "acts.json",
                             SRC176 / "stats.json")},
        "python": {"exe": sys.executable, "version": platform.python_version(),
                   "numpy": np.__version__},
        "git": {**git_head(), "note": "本入口零执行，只读 R1.7.6 入库证据"},
        "command": " ".join([Path(sys.executable).name, "experiments/cw_v1/r1_7_7_peruser.py"]
                            + (["--no-write"] if no_write else [])),
        "arms": {"candidate": "dp16", "baseline": "paced16"},
        "pairs": [f"{ep}|{s}" for ep, s in PAIRS],
    }
    elapsed = (datetime.now(TZ) - t0).total_seconds()
    result = {"verdict": verdict, "failed_gates": failed,
              "classifier": cls, "per_user": per_user, "floor": floor_final,
              "floor_raw": floor_by_arm, "rows": rows, "elapsed_s": elapsed}

    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("rows", "floor_raw")},
                     ensure_ascii=False, indent=1))
    print("\n--- gates ---")
    for k, g in gates.items():
        print(f"  {k:36s} pass={g['pass']}")
    print(f"\nVERDICT = {verdict}")
    print(f"elapsed  = {elapsed:.2f} s")

    if no_write:
        return 0 if gate_all else 4

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out = REPO / "results" / "cw_v1" / f"cw_r1_7_7_peruser_{stamp}_{meta['src_hash']}"
    if out.exists():
        print(f"[REFUSE] 目录已存在，拒绝覆盖：{out}", file=sys.stderr)
        return 3
    out.mkdir(parents=True)
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
    (out / "gates.json").write_text(json.dumps(gates, ensure_ascii=False, indent=1),
                                    encoding="utf-8")
    (out / "peruser.json").write_text(json.dumps(
        {"per_user": per_user, "floor": floor_final, "floor_raw": floor_by_arm,
         "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "verdict.json").write_text(json.dumps(
        {k: v for k, v in result.items() if k not in ("rows", "floor_raw")},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[WROTE] {out.relative_to(REPO)}")
    return 0 if gate_all else 4


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-write", action="store_true", help="自检：不落盘")
    a = ap.parse_args()
    sys.exit(run(a.no_write))
