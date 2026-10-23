#!/usr/bin/env python
"""CW-R1.7.8 · 判据形态（零成本：钉分母 / 角色 / 幅度 / 前提）

预注册：`docs/research/CW-R1.7.8-判据形态-预注册-v0.1.md`

本入口**不跑任何 episode**，只读 R1.7.6 已入库的 `cells.json`，回答四件事：
  P1 分母 —— 判"效应是否可辨"的标准误该来自哪条轴（配对差 vs 臂内地板）
  P2 角色 —— `ep100`/`ep101` 用户角色互换后，逐用户量还能不能跨 ep 汇总
  P3 幅度 —— 符号计数对幅度盲，用什么尺度分层
  P4 前提 —— `cost_est` 是信念侧成本，臂间差里有多少是"估计偏差的差"

用法：
    python experiments/cw_v1/r1_7_8_metric_form.py            # 写证据
    python experiments/cw_v1/r1_7_8_metric_form.py --no-write # 自检，不落盘
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SRC176 = (REPO / "results" / "cw_v1"
          / "cw_r1_7_6_g4form_20261023T182640_f11ffdefc06c0201_7dd8ba47dede6ce3")
TZ = timezone(timedelta(hours=8))

EPS = (100, 101)
SEEDS = (2001, 2002, 2003, 2004)

B_BOOT = 20000
RNG_SEED = 20261023
CI_LO, CI_HI = 2.5, 97.5

# 预注册 §4.3 分层阈值（写死）
BAND_NULL = 0.05
BAND_SMALL = 0.20
# 预注册 §3 D5 角色分离度
ROLE_SEP_MIN = 1.5
TOL_IDENT = 1e-12
TOL_RECON = 1e-9

# 预注册 §4.2：旧"效应/地板 >= 3"仅作历史对照列
LEGACY_LINE = 3.0


# ------------------------------------------------------------------ 工具
def sym(a: float, b: float) -> float:
    """对称相对差 —— 与 r1_7_7 逐字一致（仅用于历史对照列）。"""
    return 2.0 * abs(b - a) / (abs(a) + abs(b) + 1e-300)


def band(x: float) -> str:
    """预注册 §4.3：按 |δ| 分三档。"""
    ax = abs(x)
    if ax < BAND_NULL:
        return "approx0"
    if ax < BAND_SMALL:
        return "small"
    return "large"


def sha256_16_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def sha256_16_path(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def git_exe() -> str | None:
    for exe in ("git", r"D:/devtools/Git/cmd/git.exe"):
        try:
            subprocess.run([exe, "--version"], capture_output=True, timeout=20)
            return exe
        except Exception:  # noqa: BLE001
            continue
    return None


def git_head() -> dict:
    exe = git_exe()
    if exe is None:
        return {"head": None, "dirty_paths": [], "note": "git 不可用"}
    head = subprocess.run([exe, "rev-parse", "--short", "HEAD"], cwd=str(REPO),
                          capture_output=True, text=True, timeout=20).stdout.strip()
    dirty = subprocess.run([exe, "status", "--porcelain"], cwd=str(REPO),
                           capture_output=True, text=True, timeout=20).stdout.strip()
    return {"head": head, "dirty_paths": dirty.splitlines()}


def blob_sha16(rel: str) -> dict:
    """★ R1.7.7 §9.1 教训：`core.autocrlf=input` ⇒ 工作树(CRLF) 与 blob(LF) 哈希不同，必须双记。"""
    out = {"path": rel}
    p = REPO / rel
    out["worktree_sha"] = sha256_16_path(p)
    exe = git_exe()
    if exe is None:
        out["blob_sha"] = None
        return out
    r = subprocess.run([exe, "cat-file", "blob", f"HEAD:{rel}"], cwd=str(REPO),
                       capture_output=True, timeout=60)
    out["blob_sha"] = sha256_16_bytes(r.stdout) if r.returncode == 0 else None
    return out


# ------------------------------------------------------------------ 判据（预注册 §5）
def judge(deltas, baselines, rng) -> dict:
    """对一格 `Δ` 落判词。`Δ = cost(dp) − cost(paced)` ⇒ `Δ<0` = 候选臂成本更低。

    映射（与预注册 §5 的 NEG/POS 一一对应，仅改名以免把 NEG 读成"坏"）：
        NEG → DP-LOWER，POS → DP-HIGHER
    """
    d = np.asarray(deltas, dtype=float)
    b = np.asarray(baselines, dtype=float)
    n = d.size
    med = float(np.median(d))
    rel = d / b
    med_rel = float(np.median(rel))

    idx = rng.integers(0, n, size=(B_BOOT, n))
    bs = np.median(d[idx], axis=1)
    lo = float(np.percentile(bs, CI_LO))
    hi = float(np.percentile(bs, CI_HI))
    n_distinct = int(np.unique(bs).size)  # n=4 时最多 256 ⇒ CI 是离散的（诚实性用）

    if lo > 0.0:
        v = "DP-HIGHER"
    elif hi < 0.0:
        v = "DP-LOWER"
    elif abs(med_rel) < BAND_NULL:
        v = "NULL"
    else:
        v = "UNRESOLVED"

    return {
        "n": n, "verdict": v, "median_delta": med, "median_rel": med_rel,
        "ci95": [lo, hi], "ci_includes_zero": bool(lo <= 0.0 <= hi),
        "bootstrap": {"B": B_BOOT, "n_distinct_medians": n_distinct,
                      "method": "percentile", "rng_seed": RNG_SEED},
        "baseline_median": float(np.median(b)),
        "band_median_rel": band(med_rel),
        "n_pos": int((d > 0).sum()), "n_neg": int((d < 0).sum()),
        "n_approx0": int(sum(1 for r in rel if band(float(r)) == "approx0")),
        "n_small": int(sum(1 for r in rel if band(float(r)) == "small")),
        "n_large": int(sum(1 for r in rel if band(float(r)) == "large")),
        "bands": [band(float(r)) for r in rel],
    }


def combine(v1: str, v2: str) -> str:
    """预注册 §5 的合并判词（两 episode 是否同向）。"""
    fixed = {"DP-LOWER", "DP-HIGHER"}
    a, b = v1, v2
    if a in fixed and b in fixed:
        return "CONSISTENT-LOWER" if a == b == "DP-LOWER" else (
            "CONSISTENT-HIGHER" if a == b == "DP-HIGHER" else "EPISODE-CONFLICT")
    if a in fixed and b not in fixed:
        return f"ONE-EP-{a}"
    if b in fixed and a not in fixed:
        return f"ONE-EP-{b}"
    return "UNRESOLVED-BOTH"


# ------------------------------------------------------------------ 主流程
def run(no_write: bool) -> int:
    t0 = datetime.now(TZ)
    cells = json.loads((SRC176 / "cells.json").read_text(encoding="utf-8"))
    key = lambda arm, ep, s: f"{arm}|{ep}|{s}"  # noqa: E731

    n_users = len(cells[key("dp16", 100, 2001)]["cost_est_by_user"])
    gates: dict = {}

    # ============================== D0 恒等式
    worst0 = 0.0
    for c in cells.values():
        s = float(sum(c["cost_est_by_user"]))
        worst0 = max(worst0, abs(c["cost_est"] - s) / max(abs(c["cost_est"]), 1e-300))
    gates["D0_identity"] = {"pass": bool(worst0 <= TOL_IDENT), "max_rel_err": worst0,
                            "n_cells": len(cells), "n_users": n_users,
                            "rule": "cost_est 必须是 cost_est_by_user 的同一分解"}

    # ============================== D1 配对同一性
    pairs = [(ep, s) for ep in EPS for s in SEEDS]
    mism = [f"{ep}|{s}" for ep, s in pairs
            if cells[key("dp16", ep, s)]["stream_ids"]["spawn_keys"]
            != cells[key("paced16", ep, s)]["stream_ids"]["spawn_keys"]]
    gates["D1_pairing"] = {"pass": bool(not mism), "n_pairs": len(pairs),
                           "mismatched": mism,
                           "rule": "同 (ep,seed) 两臂 spawn_keys 逐字相同"}

    # ============================== 逐对取值（原始用户编号）
    raw = []
    for ep, s in pairs:
        a = cells[key("paced16", ep, s)]
        b = cells[key("dp16", ep, s)]
        row = {"episode": ep, "seed": s, "u": {}}
        for u in range(n_users):
            pa, pb = a["cost_est_by_user"][u], b["cost_est_by_user"][u]
            ta, tb = a["cost_true_by_user"][u], b["cost_true_by_user"][u]
            row["u"][u] = {"paced": pa, "dp": pb, "delta": pb - pa,
                           "paced_true": ta, "dp_true": tb,
                           "delta_true": tb - ta,
                           # gap = est - true；Δ_gap = Δ_est − Δ_true（恒等式见 D6）
                           "gap_paced": pa - ta, "gap_dp": pb - tb}
        raw.append(row)

    # ============================== D2 对账 R1.7.7 入库的逐用户量与 c_u
    recon = {}
    for u in range(n_users):
        ds = [r["u"][u]["delta"] for r in raw]
        got_med = float(np.median(ds))
        got_pos = int(sum(1 for d in ds if d > 0))
        got_neg = int(sum(1 for d in ds if d < 0))
        recon[f"u{u}"] = {"median_delta": got_med, "n_pos": got_pos, "n_neg": got_neg,
                          "consistency": max(got_pos, got_neg) / len(ds)}
    # R1.7.7 交付 §4 / §9.1 的入库值
    REF = {"u0": {"median_delta": 4.84742e-05, "n_pos": 5, "n_neg": 3, "consistency": 0.625},
           "u1": {"median_delta": -1.50730e-03, "n_pos": 1, "n_neg": 7, "consistency": 0.875}}
    diffs = {}
    ok_d2 = True
    for k, ref in REF.items():
        g = recon[k]
        dd = abs(g["median_delta"] - ref["median_delta"])
        # 参考值只有 6 位有效数字 ⇒ 容差按 5e-9 判（不是 1e-12，那对 6 位有效数字不可能）
        tol = 5e-9
        ok = bool(dd <= tol and g["n_pos"] == ref["n_pos"] and g["n_neg"] == ref["n_neg"]
                  and abs(g["consistency"] - ref["consistency"]) <= TOL_RECON)
        diffs[k] = {"median_delta_absdiff": dd, "tol": tol, "match": ok}
        ok_d2 = ok_d2 and ok
    gates["D2_recon_R1_7_7"] = {"pass": bool(ok_d2), "got": recon, "ref": REF,
                                "diff": diffs,
                                "rule": "复算 R1.7.7 入库的逐用户中位 Δ 与 c_u（6 位有效，容差 5e-9）"}

    # ============================== P2 角色重标号（用基线臂，外生）
    sep = []
    for r in raw:
        p0, p1 = r["u"][0]["paced"], r["u"][1]["paced"]
        hi_u = 0 if p0 >= p1 else 1
        lo_u = 1 - hi_u
        ratio = max(p0, p1) / max(min(p0, p1), 1e-300)
        r["exp_u"], r["cheap_u"] = hi_u, lo_u
        r["role_sep"] = ratio
        sep.append({"episode": r["episode"], "seed": r["seed"],
                    "exp_u": hi_u, "cheap_u": lo_u, "sep": ratio,
                    "ok": bool(ratio >= ROLE_SEP_MIN)})
    n_role_ok = sum(1 for x in sep if x["ok"])
    gates["D5_role_separation"] = {
        "pass": bool(n_role_ok == len(sep)), "n_ok": n_role_ok, "n_pairs": len(sep),
        "min_sep": float(min(x["sep"] for x in sep)), "line": ROLE_SEP_MIN,
        "detail": sep,
        "rule": "逐配对 max/min(paced 两用户成本) >= 1.5，否则角色标签不可靠"}
    # 角色是否在 ep 内稳定（同一 ep 的 4 个 seed 是否给出同一个 exp_u）
    role_stable = {}
    for ep in EPS:
        us = [x["exp_u"] for x in sep if x["episode"] == ep]
        role_stable[f"ep{ep}"] = {"exp_u_per_seed": us,
                                  "stable": bool(len(set(us)) == 1),
                                  "exp_u": us[0] if len(set(us)) == 1 else None}
    gates["D5b_role_stable_within_ep"] = {
        "pass": bool(all(v["stable"] for v in role_stable.values())),
        "detail": role_stable,
        "rule": "同 episode 的 4 个 seed 必须给出同一个『贵用户』编号"}

    # ============================== D6 恒等式 Δ_gap = Δ_est − Δ_true
    worst6 = 0.0
    for r in raw:
        for u in range(n_users):
            x = r["u"][u]
            lhs = x["gap_dp"] - x["gap_paced"]
            rhs = x["delta"] - x["delta_true"]
            worst6 = max(worst6, abs(lhs - rhs))
    gates["D6_gap_identity"] = {"pass": bool(worst6 <= 1e-12), "max_abs_err": worst6,
                                "rule": "Δ_gap = Δ_est − Δ_true 必须恒成立（分解定义）"}

    # ============================== P3 幅度分层 + 主分析
    rng = np.random.default_rng(RNG_SEED)
    cells_out: dict = {}
    for role in ("EXP", "CHEAP"):
        for ep in list(EPS) + ["POOLED"]:
            sel = [r for r in raw if ep == "POOLED" or r["episode"] == ep]
            ds, bs = [], []
            for r in sel:
                u = r["exp_u"] if role == "EXP" else r["cheap_u"]
                ds.append(r["u"][u]["delta"])
                bs.append(r["u"][u]["paced"])
            cells_out[f"{role}|ep{ep}"] = {
                **judge(ds, bs, np.random.default_rng(RNG_SEED)),
                "note": ("跨初值混合，不代表任何单一场景" if ep == "POOLED" else ""),
                "deltas": [float(x) for x in ds],
                "baselines": [float(x) for x in bs],
                "rel": [float(d / b) for d, b in zip(ds, bs)],
            }

    # 合并判词（预注册 §5：由两个 ep 是否同向决定）
    combined = {}
    for role in ("EXP", "CHEAP"):
        v1 = cells_out[f"{role}|ep100"]["verdict"]
        v2 = cells_out[f"{role}|ep101"]["verdict"]
        combined[role] = {"ep100": v1, "ep101": v2, "combined": combine(v1, v2),
                          "pooled": cells_out[f"{role}|epPOOLED"]["verdict"]}

    # ============================== P4 前提：Δ_est 里有多少是估计偏差
    p4 = {}
    for role in ("EXP", "CHEAP"):
        rows = []
        for r in raw:
            u = r["exp_u"] if role == "EXP" else r["cheap_u"]
            x = r["u"][u]
            rows.append({"episode": r["episode"], "seed": r["seed"],
                         "delta_est": x["delta"], "delta_true": x["delta_true"],
                         "delta_gap": x["gap_dp"] - x["gap_paced"],
                         "ratio_gap_over_est": (abs(x["gap_dp"] - x["gap_paced"])
                                                / max(abs(x["delta"]), 1e-300))})
        n_dom = sum(1 for x in rows if abs(x["delta_gap"]) >= abs(x["delta_est"]))
        p4[role] = {
            "rows": rows,
            "n_pairs": len(rows),
            "n_gap_dominates": int(n_dom),
            "frac_gap_dominates": n_dom / len(rows),
            "median_delta_est": float(np.median([x["delta_est"] for x in rows])),
            "median_delta_true": float(np.median([x["delta_true"] for x in rows])),
            "median_ratio": float(np.median([x["ratio_gap_over_est"] for x in rows])),
            "est_usable": bool(n_dom * 2 < len(rows)),  # 预注册 §4.4：多数配对 |Δgap| ≥ |Δest| ⇒ 降级
        }

    # ============================== 诊断（post-hoc，**不作判词**；预注册未含，见交付 §8 自伤）
    diag: dict = {}

    # S1 CI 端点离 0 有多近 —— 判 DP-LOWER/DP-HIGHER 的格必须看这条
    for k, v in cells_out.items():
        if v["verdict"] in ("DP-LOWER", "DP-HIGHER"):
            edge = v["ci95"][1] if v["verdict"] == "DP-LOWER" else v["ci95"][0]
            diag.setdefault("S1_ci_edge_margin", {})[k] = {
                "verdict": v["verdict"], "edge": edge,
                "margin_over_abs_median": abs(edge) / max(abs(v["median_delta"]), 1e-300),
                "n_distinct_medians": v["bootstrap"]["n_distinct_medians"],
                "note": "端点越贴近 0、离散中位值越少 ⇒ 该判词越不稳健"}

    # S2 幅度敏感性：把 `approx0` 档的配对剔除后重算（只对已定格的格）
    for k, v in cells_out.items():
        if v["verdict"] not in ("DP-LOWER", "DP-HIGHER"):
            continue
        keep = [(d, b) for d, b, bd in zip(v["deltas"], v["baselines"], v["bands"])
                if bd != "approx0"]
        if len(keep) < 3:
            diag.setdefault("S2_drop_approx0", {})[k] = {
                "n_kept": len(keep), "status": "SKIP-NO-OBJECT",
                "why": f"剔除 approx0 后只剩 {len(keep)} 对，<3 ⇒ 不给 CI（#100：空过不算过）"}
            continue
        jd = judge([d for d, _ in keep], [b for _, b in keep],
                   np.random.default_rng(RNG_SEED))
        diag["S2_drop_approx0"][k] = {
            "n_kept": len(keep), "verdict_after_drop": jd["verdict"],
            "median_delta": jd["median_delta"], "ci95": jd["ci95"],
            "changed": bool(jd["verdict"] != v["verdict"])}

    # S3 压缩倍数：`cost_est` 的臂间差相对 `cost_true` 被压缩多少
    #   ★ 预注册 §4.4 的判法（"|Δgap| ≥ |Δest| ⇒ 主要来自估计偏差"）在恒等式
    #     `Δgap = Δest − Δtrue` 下是**循环的**（|Δtrue| 大则 |Δgap| 自动大）。
    #     正确的问题是"压缩/放大多少倍"，见交付 §8 自伤 #1。
    for role in ("EXP", "CHEAP"):
        rr = [abs(x["delta_true"]) / max(abs(x["delta_est"]), 1e-300)
              for x in p4[role]["rows"]]
        signs = [1 if (x["delta_true"] * x["delta_est"]) > 0 else
                 (-1 if (x["delta_true"] * x["delta_est"]) < 0 else 0)
                 for x in p4[role]["rows"]]
        diag.setdefault("S3_compression", {})[role] = {
            "median_abs_true_over_abs_est": float(np.median(rr)),
            "p10": float(np.percentile(rr, 10)), "p90": float(np.percentile(rr, 90)),
            "n_same_sign": int(sum(1 for s in signs if s > 0)),
            "n_flip_sign": int(sum(1 for s in signs if s < 0)),
            "n_zero": int(sum(1 for s in signs if s == 0)),
            "note": "符号一致数 > 翻转数 ⇒ 方向可信、幅度被压缩；否则方向也不可信"}

    # S4 判"符号翻转"之前必须先问：两个量各自的效应有没有超过各自的噪声？
    #   否则"翻转"只是两边都在噪声里，不能推出"cost_est 方向不可信"。
    #   ★ 地板口径与 legacy 列**完全一致**（同 ep、同角色、跨 seed 两两配对、各臂各算一套、
    #     **取较大者** = R1.7.7 §4 的保守原则），只是把 `cost_est` 换成 `cost_true` 再算一遍。
    #   ★★ 相对地板在"值小"时会饱和（`sym` 在两值都接近 0 时趋 2）⇒ 再加一套**绝对量**地板
    #     （|跨 seed 差| 的绝对中位），四种口径一起报：{est, true} × {相对 sym, 绝对}。
    #     结论必须对口径**不敏感**才算稳。
    def _floor_role_abs(role: str, fields: tuple[str, str]) -> dict:
        vals = {fields[0]: [], fields[1]: []}
        for ep in EPS:
            rs = [r for r in raw if r["episode"] == ep]
            for i in range(len(rs)):
                for j in range(i + 1, len(rs)):
                    if rs[i]["exp_u"] != rs[j]["exp_u"]:
                        continue
                    u = rs[i]["exp_u"] if role == "EXP" else rs[i]["cheap_u"]
                    for arm in fields:
                        vals[arm].append(abs(rs[i]["u"][u][arm] - rs[j]["u"][u][arm]))
        med = {k: float(np.median(v)) for k, v in vals.items() if v}
        return {"per_arm": med, "taken_max": max(med.values()),
                "n": len(next(iter(vals.values())))}

    def _eff_abs(role: str, fld: tuple[str, str]) -> float:
        out = []
        for r in raw:
            u = r["exp_u"] if role == "EXP" else r["cheap_u"]
            out.append(abs(r["u"][u][fld[1]] - r["u"][u][fld[0]]))
        return float(np.median(out))


    def _floor_role(role: str, fields: tuple[str, str]) -> dict:
        vals = {fields[0]: [], fields[1]: []}
        for ep in EPS:
            rs = [r for r in raw if r["episode"] == ep]
            for i in range(len(rs)):
                for j in range(i + 1, len(rs)):
                    if rs[i]["exp_u"] != rs[j]["exp_u"]:
                        continue
                    u = rs[i]["exp_u"] if role == "EXP" else rs[i]["cheap_u"]
                    for arm in fields:
                        vals[arm].append(sym(rs[i]["u"][u][arm], rs[j]["u"][u][arm]))
        med = {k: float(np.median(v)) for k, v in vals.items() if v}
        return {"per_arm": med, "n": len(next(iter(vals.values()))),
                "taken_max": max(med.values())}

    for role in ("EXP", "CHEAP"):
        fl_est = _floor_role(role, ("paced", "dp"))
        fl_true = _floor_role(role, ("paced_true", "dp_true"))
        rels = []
        for r in raw:
            u = r["exp_u"] if role == "EXP" else r["cheap_u"]
            x = r["u"][u]
            rels.append({"episode": r["episode"], "seed": r["seed"],
                         "rel_est": sym(x["paced"], x["dp"]),
                         "rel_true": sym(x["paced_true"], x["dp_true"])})
        m_est = float(np.median([x["rel_est"] for x in rels]))
        m_true = float(np.median([x["rel_true"] for x in rels]))
        r_est = m_est / fl_est["taken_max"]
        r_true = m_true / fl_true["taken_max"]
        diag.setdefault("S4_effect_vs_floor", {})[role] = {
            "median_rel_effect_est": m_est, "median_rel_effect_true": m_true,
            "floor_est": fl_est, "floor_true": fl_true,
            "ratio_est": r_est, "ratio_true": r_true,
            "est_resolvable": bool(r_est >= 1.0),
            "true_resolvable": bool(r_true >= 1.0),
            "note": "两量同口径（取较大者=保守）；两个 ratio 都 <1 ⇒ 这块面板测不出任何东西，"
                    "此时 S3 的『符号翻转』不能反推 cost_est 不可信"}
        # 绝对量口径（不受小值饱和影响）
        fa_est = _floor_role_abs(role, ("paced", "dp"))
        fa_true = _floor_role_abs(role, ("paced_true", "dp_true"))
        ea_est = _eff_abs(role, ("paced", "dp"))
        ea_true = _eff_abs(role, ("paced_true", "dp_true"))
        mat = {"est_rel": r_est, "est_abs": ea_est / fa_est["taken_max"],
               "true_rel": r_true, "true_abs": ea_true / fa_true["taken_max"]}
        diag["S4_effect_vs_floor"][role].update({
            "floor_abs": {"est": fa_est, "true": fa_true},
            "effect_abs": {"est": ea_est, "true": ea_true},
            "matrix": mat,
            "max_ratio_over_all_scales": max(mat.values()),
            "any_scale_reaches_2": bool(max(mat.values()) >= 2.0),
        })



    # ============================== D3 分辨力（活体对照，答案已知）
    base8 = [0.005, 0.011, 0.004, 0.003, 0.017, 0.002, 0.003, 0.002]
    ctrls = {}
    # (a) Δ≡0 ⇒ 必须含 0 且 NULL
    z = judge([0.0] * 8, base8, np.random.default_rng(RNG_SEED))
    ctrls["A_identical"] = {"expect": "NULL", "got": z["verdict"],
                            "ci_includes_zero": z["ci_includes_zero"]}
    # (b) +20% 恒定 ⇒ 必须不含 0 且 DP-HIGHER
    up = judge([0.2 * b for b in base8], base8, np.random.default_rng(RNG_SEED))
    ctrls["B_plus20pct"] = {"expect": "DP-HIGHER", "got": up["verdict"],
                            "ci_includes_zero": up["ci_includes_zero"]}
    # (c) 交替 ±30% ⇒ 中位≈0，必须含 0
    alt_d = [0.30 * b * (1 if i % 2 == 0 else -1) for i, b in enumerate(base8)]
    alt = judge(alt_d, base8, np.random.default_rng(RNG_SEED))
    ctrls["C_alternating30"] = {"expect_includes_zero": True, "got": alt["verdict"],
                                "ci_includes_zero": alt["ci_includes_zero"],
                                "median_rel": alt["median_rel"]}
    # (d) (b) 取反 ⇒ 必须 DP-LOWER
    dn = judge([-0.2 * b for b in base8], base8, np.random.default_rng(RNG_SEED))
    ctrls["D_minus20pct"] = {"expect": "DP-LOWER", "got": dn["verdict"],
                             "ci_includes_zero": dn["ci_includes_zero"]}
    # (e) n=4 单格的已知答案（主表真实规模）：+25% 恒定，必须不含 0
    up4 = judge([0.25 * b for b in base8[:4]], base8[:4], np.random.default_rng(RNG_SEED))
    ctrls["E_n4_plus25pct"] = {"expect": "DP-HIGHER", "got": up4["verdict"],
                               "ci_includes_zero": up4["ci_includes_zero"]}
    # (f) n=4 全同 ⇒ 必须含 0（退化：CI 是点）
    z4 = judge([0.0] * 4, base8[:4], np.random.default_rng(RNG_SEED))
    ctrls["F_n4_identical"] = {"expect": "NULL", "got": z4["verdict"],
                               "ci_includes_zero": z4["ci_includes_zero"]}

    def _ck(v):
        if "expect" in v:
            return v["got"] == v["expect"]
        if "expect_includes_zero" in v:
            return v["ci_includes_zero"] == v["expect_includes_zero"]
        return False

    for k, v in ctrls.items():
        v["pass"] = bool(_ck(v))
    gates["D3_discriminative"] = {"pass": bool(all(v["pass"] for v in ctrls.values())),
                                 "controls": ctrls,
                                 "rule": "六个已知答案的合成输入必须各给出对应判词/CI 性质"}

    # ============================== D4 bootstrap 可复现（同 rng seed 两遍逐位）
    r1 = judge([x["u"][0]["delta"] for x in raw], [x["u"][0]["paced"] for x in raw],
               np.random.default_rng(RNG_SEED))
    r2 = judge([x["u"][0]["delta"] for x in raw], [x["u"][0]["paced"] for x in raw],
               np.random.default_rng(RNG_SEED))
    gates["D4_bootstrap_deterministic"] = {
        "pass": bool(r1["ci95"] == r2["ci95"]), "ci_a": r1["ci95"], "ci_b": r2["ci95"],
        "rule": "同一 rng seed 两次 bootstrap 的 CI 端点必须逐位相同"}

    # ============================== 历史对照列（效应/地板）—— 单一来源：复用 S4 的 `floor_est`
    legacy = {}
    s4 = diag["S4_effect_vs_floor"]
    stats176 = None
    try:
        stats176 = json.loads((SRC176 / "stats.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        stats176 = None
    for role in ("EXP", "CHEAP"):
        fl = s4[role]["floor_est"]
        taken = fl["taken_max"]
        eff = s4[role]["median_rel_effect_est"]
        legacy[role] = {"floor_paced": fl["per_arm"]["paced"],
                        "floor_dp": fl["per_arm"]["dp"],
                        "floor_taken_max": taken, "n_floor_pairs": fl["n"],
                        "median_abs_rel_effect": eff,
                        "effect_over_floor": (eff / taken if taken > 0 else None),
                        "line": LEGACY_LINE,
                        "passes_legacy_line": bool(taken > 0 and eff / taken >= LEGACY_LINE),
                        "status": "历史对照列，不作判词（预注册 §4.1）",
                        "note": "同口径：同 ep、同角色、跨 seed 两两配对、各臂各算、取较大者"}


    # ============================== 判词
    gate_all = all(bool(g["pass"]) for g in gates.values())
    verdict = {
        "P1_denominator": {
            "primary": "配对差 95% bootstrap CI（百分位法）",
            "arm_floor_status": "diagnostic-only",
            "legacy_line_status": "historical-column-only (>=3 无理论依据)",
        },
        "P2_role": {"label_source": "paced16 (baseline, exogenous)",
                    "primary_report": "per-episode, no cross-episode pooling",
                    "role_stable": role_stable,
                    "combined": combined},
        "P3_band": {"scale": "delta / cost_paced (per pair)",
                    "thresholds": {"approx0": BAND_NULL, "small": BAND_SMALL}},
        "P4_premise": {k: {kk: vv for kk, vv in v.items() if kk != "rows"}
                       for k, v in p4.items()},
        "cells": {k: {kk: vv for kk, vv in v.items()
                      if kk not in ("deltas", "baselines", "rel")}
                  for k, v in cells_out.items()},
        "diagnostics_posthoc": diag,
    }
    if not gate_all:
        verdict["verdict"] = "VOID-GATE"
        verdict["failed_gates"] = [k for k, g in gates.items() if not g["pass"]]
    else:
        verdict["verdict"] = "METRIC-FORM-PINNED"
        verdict["failed_gates"] = []

    elapsed = (datetime.now(TZ) - t0).total_seconds()

    # ============================== 输出
    print(json.dumps(verdict, ensure_ascii=False, indent=1))
    print("\n--- gates ---")
    for k, g in gates.items():
        print(f"  {k:30s} pass={g['pass']}")
    print("\n--- legacy column (不作判词) ---")
    print(json.dumps(legacy, ensure_ascii=False, indent=1))
    print(f"\nVERDICT = {verdict['verdict']}")
    print(f"elapsed  = {elapsed:.2f} s")

    if no_write:
        return 0 if gate_all else 4

    # ============================== 写证据
    src = Path(__file__).resolve()
    rel_src = str(src.relative_to(REPO)).replace("\\", "/")
    rel_cells = str((SRC176 / "cells.json").relative_to(REPO)).replace("\\", "/")
    meta = {
        "exp": "CW-R1.7.8 判据形态（零成本：分母/角色/幅度/前提）",
        "prereg": "docs/research/CW-R1.7.8-判据形态-预注册-v0.1.md",
        "started_at": t0.isoformat(), "tz": "Asia/Shanghai (+08:00)",
        "run_episode_calls": 0,
        "src": rel_src, "src_hash": sha256_16_path(src),
        "inputs": {"cells.json": blob_sha16(rel_cells)},
        "python": {"exe": sys.executable, "version": platform.python_version(),
                   "numpy": np.__version__},
        "git": {**git_head(), "note": "零执行，只读 R1.7.6 入库证据"},
        "command": " ".join([Path(sys.executable).name, rel_src]
                            + (["--no-write"] if no_write else [])),
        "arms": {"candidate": "dp16", "baseline": "paced16"},
        "pairs": [f"{ep}|{s}" for ep, s in pairs],
        "bootstrap": {"B": B_BOOT, "rng_seed": RNG_SEED, "method": "percentile",
                      "note": "n=4 时重采样组合仅 4^4=256 种 ⇒ CI 离散，单格 CI 极宽"},
    }
    out = REPO / "results" / "cw_v1" / (
        f"cw_r1_7_8_metricform_{datetime.now(TZ).strftime('%Y%m%dT%H%M%S')}"
        f"_{meta['src_hash']}")
    if out.exists():
        print(f"[REFUSE] 目录已存在，拒绝覆盖：{out}", file=sys.stderr)
        return 3
    out.mkdir(parents=True)
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
    (out / "gates.json").write_text(json.dumps(gates, ensure_ascii=False, indent=1),
                                    encoding="utf-8")
    (out / "cells_by_role.json").write_text(json.dumps(
        {"cells": cells_out, "combined": combined, "role_detail": sep,
         "legacy_column": legacy}, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "premise_p4.json").write_text(json.dumps(p4, ensure_ascii=False, indent=1),
                                         encoding="utf-8")
    (out / "verdict.json").write_text(json.dumps(
        {**verdict, "elapsed_s": elapsed}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[WROTE] {out.relative_to(REPO)}")
    return 0 if gate_all else 4


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-write", action="store_true", help="自检：不落盘")
    a = ap.parse_args()
    sys.exit(run(a.no_write))
