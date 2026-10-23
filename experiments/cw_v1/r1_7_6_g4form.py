# -*- coding: utf-8 -*-
"""CW-R1.7.6 `G4` 改形态：先量噪声地板，再定"可达形态"。

上一轮（R1.7.5）钉住了"`G4` 的 10% 线在 `M=16` 上不可达"，但**没回答换成什么形态才可达**。
本轮按预注册 `docs/research/CW-R1.7.6-G4改形态-预注册-v0.1.md` 的顺序做三件事：

1. `A0` **前提闸**：固定 `(ep, M, arm)` 只改规划器 `seed` ⇒ 外生流必须逐位相同、
   成本必须不同；且本轮重跑的 `paced@M=16` 必须与 R1.7.5 入库 `cells.json` 的
   `cost_true` **逐位相同**（否则 `dp` 臂与 `paced` 臂差的就不止 `tail` 一个因素）。
2. `A1`（零成本）：从 `cells.json` 复算 `cost_true` 的**跨 seed 噪声地板**
   —— 这是"换 seed 也过不了 10% 线"的直接检验。
3. `B1/B2/B3`（26 次 `run_episode`）：把 `G4` 的被比较量从 **`cost_true`（诊断）**
   换成 **`cost_est`（`runner.py` 里的主指标）**，原阈值 `0.90/0.10` 一个不改地再问一次；
   另加"效应/地板 ≥ 3 且符号一致 ≥7/8"的分辨率形态 `G4R`。

★ 本入口**不碰** `plan_ref` / `plan_dp` / `plan_gates` 一个字节：只调用它们的公开构造函数。
★ 不读 `N1/N2`，不跑面板，不改任何阈值，不训练模型。
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import time
import tracemalloc
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from . import config as C
from . import plan_gates as G
from .plan_ref import make
from .runner import run_episode

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results" / "cw_v1"

#: R1.7.5 已入库证据（`A1` 零成本复算 + `A0c` 口径恒等）
R175_DIR = RESULTS / "cw_r1_7_5_reach_20261023T160758_f11ffdefc06c0201_876e6de425c68273"
R175_CELLS = R175_DIR / "cells.json"

# ---- 参数：一律取判据实现自身，不挑
EPS = (100, 101)
EPS_HIST = (100, 101, 102, 103)          # R1.7.5 的 4 集（只给零成本的 A1 用）
SEEDS = (2001, 2002, 2003, 2004)
RHO = float(G.RHO_DEFAULT)               # 0.98
RATIO = 0.5
H_PLAN = 5
M_LIST = (16, 32)
ARM = "probe"
TAIL_DP = {"tail": "dp", "dp_grid": "r173"}   # R1.7.3 网格 + R1.7.1 修复后的 user_axis

#: 阈值：`0.90` / `0.10` 是 `G4` 原线（**不许改**）；`3.0` 是预注册 §2.5 的分离倍数
LINE_REL = 0.10
LINE_AGREE = 0.90
LINE_R = 3.0
LINE_SIGN = 7 / 8
N_BOOT = 10000
BOOT_SEED = 777

#: `C2` 活体对照的参数（`H=1 ⇒ PROBE≡HOLD`，G3b 结构事实）
C2_EP, C2_SEED, C2_H = 100, 2001, 1


def _src_hash() -> str:
    h = hashlib.sha256()
    for p in sorted((REPO / "experiments" / "cw_v1").glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def _git(args: list[str]) -> str:
    import subprocess

    return subprocess.run(["D:\\devtools\\Git\\cmd\\git.exe", *args],
                          cwd=str(REPO), capture_output=True, text=True).stdout.strip()


# ---------------------------------------------------------------- 尺度
def sym(a: float, b: float) -> float:
    """对称相对差。★ 地板与效应**必须**用同一个尺度，否则比值没有意义。"""
    return 2.0 * abs(b - a) / (abs(a) + abs(b) + 1e-300)


def rel_hi(c_lo: float, c_hi: float) -> float:
    """`G4` 原式：|c_hi − c_lo| / c_lo。`B2` 必须与判据实现逐字一致。"""
    return abs(c_hi - c_lo) / max(c_lo, 1e-12)


def signed(a: float, b: float) -> float:
    """带符号的对称相对差：`b` 相对 `a`。"""
    return (1.0 if b >= a else -1.0) * sym(a, b)


def _boot_median_ci(vals: list[float], n_boot: int = N_BOOT) -> dict:
    rng = np.random.default_rng(BOOT_SEED)
    v = np.asarray(vals, dtype=float)
    meds = np.array([np.median(rng.choice(v, size=len(v), replace=True))
                     for _ in range(n_boot)])
    lo, hi = float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))
    return {"n": int(len(v)), "median": float(np.median(v)), "ci95": [lo, hi],
            "excludes_zero": bool(lo > 0 or hi < 0),
            "n_boot": n_boot, "boot_seed": BOOT_SEED}


# ---------------------------------------------------------------- 跑一格
def _run(cfg, budget, ep: int, seed: int, M: int, h_plan: int = H_PLAN,
         tail: dict | None = None) -> dict:
    kw = dict(tail) if tail else {}
    pol = G._Rec(make(cfg, arm=ARM, h_plan=h_plan, m_samples=M, seed=seed,
                      rho=RHO, pacing=True, **kw))
    r = run_episode(cfg, pol, split="dev", episode=ep, budget_total=budget, rho=RHO)
    return {
        "cost_est": float(r.cost_est_per_step),
        "cost_true": float(r.cost_true_per_step),
        "cost_est_by_user": [float(x) for x in r.cost_est_by_user],
        "cost_true_by_user": [float(x) for x in r.cost_true_by_user],
        "n_probe": int(r.n_probe), "n_attempted": int(r.n_attempted),
        "n_delivered": int(r.n_delivered), "n_idle": int(r.n_idle),
        "mean_task_aoi": float(r.mean_task_aoi),
        "policy_trace_hash": str(r.policy_trace_hash),
        "stream_ids": {k: str(v) for k, v in dict(r.stream_ids).items()},
        "acts": [(str(k), (None if u is None else int(u))) for (_t, k, u) in pol.acts],
        "n_acts": len(pol.acts),
        "elapsed_s": float(r.elapsed_s),
    }


def _agree(a: dict, b: dict) -> tuple[float, int]:
    aa, bb = a["acts"], b["acts"]
    if len(aa) != len(bb):
        raise ValueError(f"动作序列长度不等 {len(aa)} vs {len(bb)}")
    same = sum(1 for (k1, u1), (k2, u2) in zip(aa, bb) if (k1, u1) == (k2, u2))
    return same / max(len(aa), 1), len(aa)


# ---------------------------------------------------------------- A1（零成本）
def a1_floor_true_from_history() -> dict:
    """从 R1.7.5 入库 `cells.json` 复算：同 `M` 跨 seed 的 `cost_true` 相对差。"""
    cells = json.loads(R175_CELLS.read_text(encoding="utf-8"))
    out: dict = {"source": str(R175_CELLS.relative_to(REPO)),
                 "n_seeds": len(SEEDS), "eps": list(EPS_HIST), "panels": {}}
    for mode, M in (("indep", 16), ("indep", 32), ("nested", 16)):
        rows = []
        for ep in EPS_HIST:
            vals = {}
            for s in SEEDS:
                k = f"{mode}|{ep}|{s}|{M}"
                if k not in cells:
                    raise KeyError(f"R1.7.5 证据缺格 {k}")
                vals[s] = float(cells[k]["cost_true"])
            for s1, s2 in itertools.combinations(SEEDS, 2):
                rows.append({"episode": ep, "s1": s1, "s2": s2,
                             "rel_hi": rel_hi(vals[s1], vals[s2]),
                             "sym": sym(vals[s1], vals[s2])})
        a = np.array([r["sym"] for r in rows], dtype=float)
        out["panels"][f"{mode}_M{M}"] = {
            "n_pairs": len(a), "median_sym": float(np.median(a)),
            "p10": float(np.percentile(a, 10)), "p90": float(np.percentile(a, 90)),
            "frac_gt_line": float((a > LINE_REL).mean()),
            "rows": rows,
        }
    out["all_medians_gt_line"] = bool(all(v["median_sym"] > LINE_REL
                                          for v in out["panels"].values()))
    out["rule"] = ("同一策略、同一 M、只换规划器 seed，成本相对差中位已 > 10% ⇒ "
                   "任何『两批抽样成本差 ≤10%』的相等型判据在 cost_true 上不可达")
    return out


# ---------------------------------------------------------------- A0 前提闸
def a0_premise(paced16: dict, hist_cells: dict) -> dict:
    out = {}
    a = paced16[f"{EPS[0]}|{SEEDS[0]}"]
    b = paced16[f"{EPS[0]}|{SEEDS[1]}"]
    out["A0a_stream_ids_seed_invariant"] = {
        "pass": a["stream_ids"] == b["stream_ids"],
        "stream_ids_a": a["stream_ids"], "stream_ids_b": b["stream_ids"],
        "rule": "外生流由 (root_seed, split, episode) 决定 ⇒ 只换规划器 seed 必须逐位相同",
    }
    out["A0b_seed_is_wired"] = {
        "pass": bool(a["policy_trace_hash"] != b["policy_trace_hash"]
                     and abs(a["cost_true"] - b["cost_true"]) > 0),
        "cost_true_a": a["cost_true"], "cost_true_b": b["cost_true"],
        "trace_a": a["policy_trace_hash"], "trace_b": b["policy_trace_hash"],
        "rule": "只换 seed 动作/成本必须变化（否则『跨 seed 地板』其实是 0）",
    }
    diffs, missing = {}, []
    for ep in EPS:
        for s in SEEDS:
            hk = f"indep|{ep}|{s}|16"
            nk = f"{ep}|{s}"
            if hk not in hist_cells or nk not in paced16:
                missing.append(hk)
                continue
            diffs[nk] = abs(paced16[nk]["cost_true"]
                            - float(hist_cells[hk]["cost_true"]))
    out["A0c_identity_with_r175"] = {
        "pass": bool((not missing) and diffs and max(diffs.values()) == 0.0),
        "abs_diff": diffs, "missing": missing,
        "rule": "本轮 paced@M=16 的 cost_true 必须与 R1.7.5 入库 cells.json 逐位相同"
                "（否则 dp 臂与 paced 臂差的就不止 tail 一个因素）",
    }
    return out


# ---------------------------------------------------------------- 统计
def _floor_rows(cells: dict, eps) -> dict:
    """同臂同 `M`、只换 seed 的对。★ `eps` 必须是**去重后**的集（否则每格会重复算 4 次）。"""
    rows = []
    for ep in eps:
        for s1, s2 in itertools.combinations(SEEDS, 2):
            a, b = cells[f"{ep}|{s1}"], cells[f"{ep}|{s2}"]
            rows.append({"episode": ep, "s1": s1, "s2": s2,
                         "sym_est": sym(a["cost_est"], b["cost_est"]),
                         "sym_true": sym(a["cost_true"], b["cost_true"])})
    return {"rows": rows, "n": len(rows),
            "median_sym_est": float(np.median([r["sym_est"] for r in rows])),
            "median_sym_true": float(np.median([r["sym_true"] for r in rows]))}


def _resolution(eff: list[float], floor_med: float, tag: str) -> dict:
    ae = np.abs(np.asarray(eff, dtype=float))
    arr = np.asarray(eff, dtype=float)
    med = float(np.median(ae))
    n_pos, n_neg = int((arr > 0).sum()), int((arr < 0).sum())
    sign_frac = max(n_pos, n_neg) / max(len(eff), 1)
    R = med / floor_med if floor_med > 0 else float("inf")
    boot = _boot_median_ci(list(eff))
    pass_R, pass_sign, pass_boot = (R >= LINE_R, sign_frac >= LINE_SIGN,
                                    boot["excludes_zero"])
    return {"tag": tag, "n": len(eff), "median_abs_effect": med,
            "floor_median": floor_med, "R": R,
            "n_pos": n_pos, "n_neg": n_neg, "sign_frac": sign_frac, "boot": boot,
            "pass_R": bool(pass_R), "pass_sign": bool(pass_sign),
            "pass_boot": bool(pass_boot),
            "pass": bool(pass_R and pass_sign and pass_boot)}


def analyse(paced16: dict, paced32: dict, dp16: dict) -> dict:
    """键：`paced16/paced32/dp16` 均为 `f"{ep}|{seed}"`。"""
    keys = [(ep, s) for ep in EPS for s in SEEDS]
    out: dict = {}

    # ---- P-3：cost_est 必须策略相关
    d = [abs(dp16[k]["cost_est"] - paced16[k]["cost_est"]) for k in
         (f"{ep}|{s}" for ep, s in keys)]
    out["P3_cost_est_policy_dependent"] = {
        "pass": bool(max(d) > 0), "max_abs_diff": float(max(d)),
        "rule": "两臂 cost_est 若逐位相同 ⇒ 它不是可比较量，全部 B* 作废（VOID）",
    }

    # ---- B1 地板（同为 M=16，只在臂内换 seed）
    fl_p = _floor_rows(paced16, EPS)
    fl_d = _floor_rows(dp16, EPS)
    pooled = {"n": fl_p["n"] + fl_d["n"],
              "median_sym_est": float(np.median([r["sym_est"] for r in
                                                 fl_p["rows"] + fl_d["rows"]])),
              "median_sym_true": float(np.median([r["sym_true"] for r in
                                                  fl_p["rows"] + fl_d["rows"]]))}
    out["B1_floor"] = {"paced": fl_p, "dp": fl_d, "pooled": pooled, "line": LINE_REL,
                       "paced_below_line": bool(fl_p["median_sym_est"] <= LINE_REL)}

    # ---- B3 分辨率（效应 = dp − paced，配对同 (ep, seed)）
    eff_est, eff_true, rows = [], [], []
    for ep, s in keys:
        a, b = paced16[f"{ep}|{s}"], dp16[f"{ep}|{s}"]
        e_est = signed(a["cost_est"], b["cost_est"])
        e_true = signed(a["cost_true"], b["cost_true"])
        eff_est.append(e_est)
        eff_true.append(e_true)
        rows.append({"episode": ep, "seed": s,
                     "c_est_paced": a["cost_est"], "c_est_dp": b["cost_est"],
                     "e_est": e_est,
                     "c_true_paced": a["cost_true"], "c_true_dp": b["cost_true"],
                     "e_true": e_true,
                     "r_i_est": b["cost_est"] / a["cost_est"] if a["cost_est"] else None})
    out["B3_effect_rows"] = rows
    out["B3_resolution_est"] = _resolution(eff_est, pooled["median_sym_est"], "cost_est")
    out["B3_resolution_true"] = _resolution(eff_true, pooled["median_sym_true"],
                                            "cost_true")

    # ---- C1 活体阴性对照：同臂跨 seed 的对，当成 effect 喂给同一个函数（n 相同）
    fake = []
    for ep, s in keys:
        s2 = SEEDS[(SEEDS.index(s) + 1) % len(SEEDS)]
        fake.append(signed(paced16[f"{ep}|{s}"]["cost_est"],
                           paced16[f"{ep}|{s2}"]["cost_est"]))
    out["C1_noise_as_effect"] = _resolution(fake, pooled["median_sym_est"],
                                           "NOISE-as-effect")
    out["C1_ok"] = bool(out["C1_noise_as_effect"]["R"] < LINE_R)

    # ---- B2 `G4` 原形态（阈值逐字不动）
    def _b2(metric: str, hi: dict) -> dict:
        rels, ags, rr = [], [], []
        for ep, s in keys:
            a, b = paced16[f"{ep}|{s}"], hi[f"{ep}|{s}"]
            ag, n = _agree(a, b)
            rels.append(rel_hi(a[metric], b[metric]))
            ags.append(ag)
            rr.append({"episode": ep, "seed": s, "agree": ag, "n_acts": n,
                       "c16": a[metric], "c32": b[metric], "rel": rels[-1]})
        rels_a, ags_a = np.asarray(rels), np.asarray(ags)
        return {"n": len(rr), "rel_median": float(np.median(rels_a)),
                "rel_p90": float(np.percentile(rels_a, 90)),
                "rel_frac_gt_line": float((rels_a > LINE_REL).mean()),
                "agree_median": float(np.median(ags_a)),
                "agree_min": float(ags_a.min()),
                "agree_frac_ge_line": float((ags_a >= LINE_AGREE).mean()),
                "pass": bool(np.median(rels_a) <= LINE_REL
                             and np.median(ags_a) >= LINE_AGREE),
                "rows": rr}
    out["B2_g4_original_form_cost_est"] = _b2("cost_est", paced32)
    out["B2_g4_original_form_cost_true"] = _b2("cost_true", paced32)

    # ---- 绝对代价与逐用户（R17：报比值必须同报绝对代价）
    def _abs(cells: dict) -> dict:
        vals = list(cells.values())
        return {"n": len(vals),
                "cost_est_median": float(np.median([c["cost_est"] for c in vals])),
                "cost_true_median": float(np.median([c["cost_true"] for c in vals])),
                "n_delivered_median": float(np.median([c["n_delivered"] for c in vals])),
                "n_attempted_median": float(np.median([c["n_attempted"] for c in vals])),
                "n_probe_median": float(np.median([c["n_probe"] for c in vals])),
                "aoi_median": float(np.median([c["mean_task_aoi"] for c in vals])),
                "cost_est_by_user_median": [
                    float(np.median([c["cost_est_by_user"][i] for c in vals]))
                    for i in range(len(vals[0]["cost_est_by_user"]))],
                "cost_true_by_user_median": [
                    float(np.median([c["cost_true_by_user"][i] for c in vals]))
                    for i in range(len(vals[0]["cost_true_by_user"]))]}
    out["R17_absolute"] = {"paced16": _abs(paced16), "paced32": _abs(paced32),
                           "dp16": _abs(dp16)}
    return out


# ---------------------------------------------------------------- C2
def skip_semantics(n_object: int, n_violation: int, rule: str) -> dict:
    """`SKIP-NO-OBJECT` 语义（R1.7.5 教训 #100：对象为 0 时返回 `pass=True` 是空过）。"""
    if n_object == 0:
        return {"status": "SKIP-NO-OBJECT", "pass": None, "n_object": 0,
                "n_violation": int(n_violation), "rule": rule,
                "note": "没有检验对象 ⇒ 不承载任何结论（不是 PASS）"}
    return {"status": "EVALUATED", "pass": bool(n_violation == 0),
            "n_object": int(n_object), "n_violation": int(n_violation), "rule": rule}


def c2_control(cfg, budget) -> dict:
    a = _run(cfg, budget, C2_EP, C2_SEED, 16, h_plan=C2_H)
    b = _run(cfg, budget, C2_EP, C2_SEED, 32, h_plan=C2_H)
    ag, n = _agree(a, b)
    same_trace = a["policy_trace_hash"] == b["policy_trace_hash"]
    obj = bool(ag >= 1.0 - 1e-12)
    return {
        "runs": {"M16": {k: a[k] for k in ("cost_est", "cost_true",
                                           "policy_trace_hash", "n_probe")},
                 "M32": {k: b[k] for k in ("cost_est", "cost_true",
                                           "policy_trace_hash", "n_probe")}},
        "agree": ag, "n_acts": n, "same_trace": same_trace,
        "object_created": obj,
        "P0d_live": skip_semantics(int(obj), int(obj and
                                            abs(a["cost_est"] - b["cost_est"]) > 0),
                                   "agree==1.0 ⇒ 两档成本必须逐位相同"),
        "P0d2_live": skip_semantics(int(same_trace),
                                    int(same_trace and a["cost_est"] != b["cost_est"]),
                                    "trace 相同 ⇒ 成本必须相同"),
        "rule": ("`H=1 ⇒ PROBE≡HOLD`，两档 M 应退化成同一策略 ⇒ 不变式**有对象**。"
                 "若仍无对象，记 `C2-NO-OBJECT`，不许改 h_plan 再试"),
    }


# ---------------------------------------------------------------- 判词
def verdict_of(checks: dict, an: dict, c2: dict) -> dict:
    b2 = an["B2_g4_original_form_cost_est"]
    r3 = an["B3_resolution_est"]
    if any(checks[k]["pass"] is False for k in
           ("A0a_stream_ids_seed_invariant", "A0b_seed_is_wired",
            "A0c_identity_with_r175")):
        v, why = "PREMISE-BROKEN", "A0 前提闸未过 ⇒ 不解读任何 B*"
    elif not an["P3_cost_est_policy_dependent"]["pass"]:
        v, why = "VOID-COST-EST-POLICY-BLIND", "两臂 cost_est 逐位相同 ⇒ 不是可比较量"
    elif not an["C1_ok"]:
        v, why = "G4R-NO-POWER", "纯噪声喂进分辨率判据也会过 ⇒ B3 的 PASS 不可采信"
    elif b2["pass"] and r3["pass"]:
        v, why = ("G4-MAIN-METRIC-REACHABLE",
                  f"主指标上原阈值可过（rel 中位 {b2['rel_median']:.4f} ≤10%、"
                  f"agree 中位 {b2['agree_median']:.3f} ≥0.90）且效应/地板 "
                  f"R={r3['R']:.2f} ≥3")
    elif (not b2["pass"]) and r3["pass"]:
        v, why = ("EQUALITY-UNREACHABLE-RESOLVABLE",
                  f"相等形态不过（rel 中位 {b2['rel_median']:.4f}）但可分辨 "
                  f"（R={r3['R']:.2f}）")
    elif b2["pass"] and (not r3["pass"]):
        v, why = ("EQUALITY-OK-RESOLUTION-LOW",
                  f"相等形态过但臂间效应分不出（R={r3['R']:.2f}）"
                  "⇒ 不许用成本差给尾部臂排序")
    else:
        v, why = ("G4-UNREACHABLE-ON-BOTH-METRICS",
                  f"两个形态都不过（rel 中位 {b2['rel_median']:.4f}、R={r3['R']:.2f}）"
                  "⇒ 下一轮不再调参，改问『面板该比什么量』")
    return {"verdict": v, "reason": why,
            "n_fail": sum(1 for c in checks.values() if c.get("pass") is False),
            "n_skip": sum(1 for c in checks.values() if c.get("pass") is None),
            "n_checks": len(checks),
            "c2_object_created": bool(c2.get("object_created")),
            "lines": {"rel": LINE_REL, "agree": LINE_AGREE, "R": LINE_R,
                      "sign": LINE_SIGN},
            "no_n1_n2": True, "no_threshold_change": True,
            "rule": "判词按预注册 §3 映射；0.90/0.10/3.0 一个不改"}


# ---------------------------------------------------------------- 主流程
def main(argv: list[str] | None = None) -> int:
    global SEEDS
    ap = argparse.ArgumentParser(description="CW-R1.7.6 G4 改形态（地板 + 主指标 + 分辨率）")
    ap.add_argument("--quick", action="store_true", help="2 seed（烟雾，不出判词）")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args(argv)
    if args.quick:
        SEEDS = SEEDS[:2]

    t0 = time.perf_counter()
    tracemalloc.start()
    cfg = C.CWConfig(plant=C.bounded_plant())
    budget = cfg.protocol.budget(RATIO)
    print(f"[R1.7.6] cfg={cfg.fingerprint()} rho={RHO} B/T={RATIO} h={H_PLAN} "
          f"seeds={len(SEEDS)} eps={EPS}", flush=True)
    hist = json.loads(R175_CELLS.read_text(encoding="utf-8"))

    # ---- 1) paced@M=16（先跑：A0 前提闸要用）
    paced16: dict = {}
    for ep in EPS:
        for s in SEEDS:
            t = time.perf_counter()
            c = _run(cfg, budget, ep, s, 16)
            paced16[f"{ep}|{s}"] = c
            print(f"  paced M=16 ep={ep} seed={s} {time.perf_counter() - t:5.1f}s "
                  f"c_est={c['cost_est']:.6f} c_true={c['cost_true']:.6f}", flush=True)

    print("[R1.7.6] A0 前提闸", flush=True)
    a0 = a0_premise(paced16, hist)
    for k, val in a0.items():
        print(f"  {val['pass']}  {k}", flush=True)
    a1 = a1_floor_true_from_history()
    for k, val in a1["panels"].items():
        print(f"  A1 {k:12s} n={val['n_pairs']} median_sym={val['median_sym']:.4f} "
              f"frac>10%={val['frac_gt_line']:.3f}", flush=True)

    paced32: dict = {}
    dp16: dict = {}
    c2: dict = {}
    an: dict = {}
    if any(val["pass"] is False for val in a0.values()):
        print("[R1.7.6] A0 FAIL ⇒ 停，不跑 B*", flush=True)
    else:
        for ep in EPS:
            for s in SEEDS:
                t = time.perf_counter()
                c = _run(cfg, budget, ep, s, 32)
                paced32[f"{ep}|{s}"] = c
                print(f"  paced M=32 ep={ep} seed={s} {time.perf_counter() - t:5.1f}s "
                      f"c_est={c['cost_est']:.6f}", flush=True)
        for ep in EPS:
            for s in SEEDS:
                t = time.perf_counter()
                c = _run(cfg, budget, ep, s, 16, tail=TAIL_DP)
                dp16[f"{ep}|{s}"] = c
                print(f"  dp@r173 M=16 ep={ep} seed={s} {time.perf_counter() - t:5.1f}s "
                      f"c_est={c['cost_est']:.6f}", flush=True)
        an = analyse(paced16, paced32, dp16)
        print("[R1.7.6] C2 活体对照（h_plan=1）", flush=True)
        c2 = c2_control(cfg, budget)
        print(f"  agree={c2['agree']:.4f} same_trace={c2['same_trace']} "
              f"object_created={c2['object_created']}", flush=True)

    checks: dict = dict(a0)
    checks["A1_floor_true_gt_line"] = {
        "pass": bool(a1["all_medians_gt_line"]),
        "medians": {k: v["median_sym"] for k, v in a1["panels"].items()},
        "rule": "三个面板的跨 seed 相对差中位都 > 10%",
    }
    if an:
        checks.update({
            "P3_cost_est_policy_dependent": an["P3_cost_est_policy_dependent"],
            "B2_g4_original_form_cost_est": {
                "pass": an["B2_g4_original_form_cost_est"]["pass"],
                "rel_median": an["B2_g4_original_form_cost_est"]["rel_median"],
                "agree_median": an["B2_g4_original_form_cost_est"]["agree_median"],
                "rule": "原 G4 阈值（0.90/0.10）换主指标 cost_est"},
            "B3_resolution_est": {
                "pass": an["B3_resolution_est"]["pass"],
                "R": an["B3_resolution_est"]["R"],
                "sign_frac": an["B3_resolution_est"]["sign_frac"],
                "rule": "效应/地板 ≥3 且符号一致 ≥7/8 且 bootstrap CI 不含 0"},
            "C1_noise_as_effect": {
                "pass": an["C1_ok"], "R": an["C1_noise_as_effect"]["R"],
                "rule": "把同臂跨 seed 的对当 effect ⇒ R 必须 <3（否则判据无分辨力）"},
            "C2_object_created": {
                "pass": c2["object_created"], "agree": c2["agree"],
                "rule": "h_plan=1 下 M=16/32 必须退化成同一策略 ⇒ P0d/P0d2 有对象"},
        })
        v = verdict_of(checks, an, c2)
    else:
        v = {"verdict": "PREMISE-BROKEN", "reason": "A0 前提闸未过", "n_fail": 1,
             "n_skip": 0, "n_checks": len(checks), "no_n1_n2": True,
             "no_threshold_change": True,
             "lines": {"rel": LINE_REL, "agree": LINE_AGREE, "R": LINE_R}}

    elapsed = round(time.perf_counter() - t0, 2)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"[R1.7.6] 判词 {v['verdict']}（{v['reason']}）；"
          f"{v['n_fail']} fail / {v['n_checks']} checks；{elapsed} s；"
          f"peak {peak / 2 ** 20:.0f} MiB", flush=True)
    if an:
        for tag in ("B2_g4_original_form_cost_est", "B2_g4_original_form_cost_true"):
            b = an[tag]
            print(f"  {tag:32s} rel 中位={b['rel_median']:.4f} "
                  f"agree 中位={b['agree_median']:.3f} pass={b['pass']}")
        for tag in ("B3_resolution_est", "B3_resolution_true", "C1_noise_as_effect"):
            r = an[tag]
            print(f"  {tag:24s} R={r['R']:.3f} 效应={r['median_abs_effect']:.4f} "
                  f"地板={r['floor_median']:.4f} 符号={r['n_pos']}/{r['n']} "
                  f"CI={np.round(r['boot']['ci95'], 4)} pass={r['pass']}")
        print(f"  地板 pooled est={an['B1_floor']['pooled']['median_sym_est']:.4f} "
              f"true={an['B1_floor']['pooled']['median_sym_true']:.4f}")

    if not args.no_write:
        now = datetime.now(timezone(timedelta(hours=8)))
        run_id = (f"cw_r1_7_6_g4form_{now.strftime('%Y%m%dT%H%M%S')}_"
                  f"{cfg.fingerprint()}_{_src_hash()}")
        out_dir = RESULTS / run_id
        out_dir.mkdir(parents=True, exist_ok=True)

        def _dump(name, obj):
            (out_dir / name).write_text(
                json.dumps(obj, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8")

        all_cells = {}
        for tag, group in (("paced16", paced16), ("paced32", paced32), ("dp16", dp16)):
            for k, val in group.items():
                all_cells[f"{tag}|{k}"] = val
        _dump("cells.json", {k: {kk: vv for kk, vv in val.items() if kk != "acts"}
                             for k, val in all_cells.items()})
        _dump("acts.json", {k: val["acts"] for k, val in all_cells.items()})
        _dump("a0_premise.json", a0)
        _dump("a1_floor_true.json", a1)
        _dump("stats.json", an or {"skipped": "A0 FAIL"})
        _dump("controls.json", c2 or {"skipped": "A0 FAIL"})
        _dump("checks.json", checks)
        _dump("verdict.json", v)
        _dump("meta.json", {
            "run_id": run_id,
            "kind": "G4-FORM-DIAGNOSIS-NOT-PANEL-CONCLUSION",
            "protocol": C.CWConfig.PROTOCOL_VERSION,
            "prereg": "docs/research/CW-R1.7.6-G4改形态-预注册-v0.1.md",
            "params": {"eps": list(EPS), "seeds": list(SEEDS), "M_list": list(M_LIST),
                       "rho": RHO, "ratio": RATIO, "h_plan": H_PLAN, "arm": ARM,
                       "tail_dp": TAIL_DP,
                       "c2": {"ep": C2_EP, "seed": C2_SEED, "h_plan": C2_H},
                       "lines": {"rel": LINE_REL, "agree": LINE_AGREE,
                                 "R": LINE_R, "sign": LINE_SIGN}},
            "quick": bool(args.quick),
            "config": cfg.to_dict(),
            "at_asia_shanghai": now.isoformat(),
            "elapsed_s": elapsed,
            "peak_tracemalloc_mib": round(peak / 2 ** 20, 1),
            "n_cells": len(all_cells), "n_extra_runs": len(c2.get("runs", {})),
            "hist_evidence": str(R175_CELLS.relative_to(REPO)),
            "git_head": _git(["rev-parse", "HEAD"]),
            "git_status_porcelain": _git(["status", "--porcelain"]),
            "src_hash": _src_hash(),
            "change_to_subject": ("无 —— 本入口只调用 plan_ref.make / runner.run_episode / "
                                  "plan_gates._Rec，未修改任何被测文件一个字节"),
            "disclaimer": ("本 run 只回答『G4 该用什么形态』：不是面板科学结论；不读 N1/N2；"
                           "不改任何阈值；不宣称『世界模型/主动测量有价值或无价值』。"
                           "2 ep × 4 seed 是开发诊断，不是确认集。"),
        })
        print(f"[R1.7.6] 证据落盘 results/cw_v1/{run_id}")

    ok = v["verdict"] in ("G4-MAIN-METRIC-REACHABLE", "EQUALITY-UNREACHABLE-RESOLVABLE",
                          "EQUALITY-OK-RESOLUTION-LOW", "G4-UNREACHABLE-ON-BOTH-METRICS")
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
