# -*- coding: utf-8 -*-
"""CW-R1.12 ACK 尺度校准 —— 精确贝叶斯 vs 一阶伪观测。

预注册：`docs/research/CW-R1.12-ACK尺度校准-预注册-v0.1.md`（§8 记录看数后的修订）

三段
----
Q1 **零 episode**：`ACK_VAR=0.63` 与精确后验的等效量测方差并列（三个口径：
   Fisher 局部、Laplace 一阶、按精确后验方差反解的"矩匹配"），并报 Laplace **自身**的误差。
P0 **前置闸**：① 新代码不许污染默认路径 —— `ack_mode="approx"` 重跑 R1.11 的两格，
   与入库值**逐位**对账（这是"两臂只差信息模型"的背书）；② 全量 `gates.py` `n_fail=0`。
PANEL **16 格**：`ep ∈ {100,101} × mode ∈ {approx,exact} × seed ∈ {2001..2004}`，
   与 R1.11 同参（`rho=0.98`、`B=floor(0.5T)`、`h_plan=5`、`M=16`、`tail=dp@r173`）。

★ 判词用**预注册的地板**（0.0333，R1.11 四组 MAD 的最大值）。文档里不许拿窄口径让效应"显著"。
"""

from __future__ import annotations

import argparse
import dataclasses as _dc
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import config as C
from . import gates as G
from . import plan_ref as Pmodule
from .runner import check_identities, run_episode
from .world import GridBelief

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results" / "cw_v1"
R111 = RESULTS / "cw_r1_11_rebaseline_20261023T235001_3ef5b3de20cd3be4" / "cells.json"

RHO = 0.98
H_PLAN = 5
M_MAIN = 16
RATIO = 0.5
EPS = (100, 101)
SEEDS = (2001, 2002, 2003, 2004)
TAIL_DP = {"tail": "dp", "dp_grid": "r173"}
ACK_VAR = 0.63                      # 现状常量（不许改，改另立卡）
FLOOR = 0.0333                      # 预注册 §5：R1.11 四组 MAD 的最大值
FLOOR_NARROW = {"ep100": 0.01272, "ep101": 0.01213}   # on/on 两组，仅作并列汇报


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


# ================================================================== Q1（零 episode）
def q1_table(kappa: float = 2.0, star: float = 0.0) -> dict:
    """三个口径并列：Fisher 局部 / Laplace 一阶 / 矩匹配（用精确后验方差反解）。"""
    ms = (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
    vs = (1.0, 0.5)
    rows = []
    for v in vs:
        for m in ms:
            p = 1.0 / (1.0 + np.exp(-kappa * (m - star)))
            info = kappa * kappa * p * (1.0 - p)
            r_fisher = 1.0 / info if info > 0 else float("inf")
            # 精确后验（网格；与闸 4n 同一条实现）
            gb = GridBelief(1, rho=0.98, kappa=kappa, log_h_star=star,
                            init_mean=m, init_std=np.sqrt(v))
            out = {}
            for ack, tag in ((True, "ack"), (False, "nack")):
                gb2 = GridBelief(1, rho=0.98, kappa=kappa, log_h_star=star,
                                 init_mean=m, init_std=np.sqrt(v))
                gb2.observe_binary(0, ack)
                em, ev = gb2.moments()
                em, ev = float(em[0]), float(ev[0])
                # 矩匹配：把精确后验方差反解成"等效伪观测方差"
                r_mm = (v * ev / (v - ev)) if (v - ev) > 0 else float("inf")
                # Laplace（一阶高斯）
                lap_m = m + v * kappa * ((1.0 - p) if ack else -p) / (1.0 + v * info)
                lap_v = v / (1.0 + v * info)
                # 现状实现（伪观测方差 ACK_VAR）
                gg = v / (v + ACK_VAR)
                cod_m = m + gg * 1.0 if ack else m - gg * 1.5
                cod_v = (1.0 - gg) * v
                out[tag] = {
                    "exact_mean": em, "exact_var": ev,
                    "delta_mean_exact": em - m, "var_ratio_exact": ev / v,
                    "r_eff_moment_matched": r_mm,
                    "r_mm_over_ACK_VAR": r_mm / ACK_VAR if np.isfinite(r_mm) else None,
                    "laplace_mean": lap_m, "laplace_var": lap_v,
                    "laplace_err_mean": lap_m - em, "laplace_err_var": lap_v - ev,
                    "code_mean": cod_m, "code_var": cod_v,
                    "code_err_mean": cod_m - em, "code_var_over_exact": cod_v / ev,
                }
            rows.append({"m": m, "v": v, "p": p,
                         "fisher_info": info, "r_eff_fisher": r_fisher,
                         "r_fisher_over_ACK_VAR": r_fisher / ACK_VAR,
                         **out})
    # ---- 判词（预注册 §4-Q1）：Fisher 口径
    off = [r for r in rows if abs(r["m"]) > 1e-12]
    ratio0 = [r["r_fisher_over_ACK_VAR"] for r in rows if abs(r["m"]) < 1e-12
              and abs(r["v"] - 1.0) < 1e-12][0]
    all_gt1 = all(r["r_fisher_over_ACK_VAR"] > 1.0 for r in off)
    if ratio0 >= 1.2 and all_gt1:
        verdict = "ACK-VAR-OVERCONFIDENT"
    elif all(r["r_fisher_over_ACK_VAR"] < 1.0 for r in off):
        verdict = "ACK-VAR-UNDERCONFIDENT"
    else:
        verdict = "ACK-VAR-SCALE-OK"
    # 额外口径（看数后追加，写进预注册 §8）：矩匹配比值
    mm_ratio = [r["ack"]["r_mm_over_ACK_VAR"] for r in rows
                if abs(r["m"]) < 1e-12 and abs(r["v"] - 1.0) < 1e-12][0]
    return {"verdict": verdict,
            "criterion": {"r_eff_fisher(0,1)/ACK_VAR": ratio0, "all_m_ne_0_gt_1": all_gt1},
            "extra_caliber_r_moment_matched_over_ACK_VAR_at_m0_v1": mm_ratio,
            "note": "两个口径（Fisher=1.587、矩匹配=2.74）**同向**：现状后验方差收缩过快。"
                    "两口径都报，不挑好看的那个（R1.7.8 纪律）。",
            "rows": rows}


# ================================================================== 记录器
class _Rec(Pmodule.PlanRef):
    """只观测：逐槽公开 telemetry + 根节点值；不改任何内层行为。"""

    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)
        self.rows: list[dict] = []

    def __call__(self, v):                      # noqa: ANN001
        a = super().__call__(v)
        tel = v.telemetry
        mh = np.asarray([[float(r["h_mean"]) for r in tel]], dtype=float)
        vh = np.asarray([[float(r["h_var"]) for r in tel]], dtype=float)
        ph = self._p_hat(mh, vh)[0]
        self.rows.append({"t": int(v.t), "phat": [float(x) for x in ph],
                          "h_mean": [float(x) for x in mh[0]],
                          "h_var": [float(x) for x in vh[0]]})
        return a


def _cfg(mode: str) -> C.CWConfig:
    base = C.CWConfig(plant=C.bounded_plant())
    return _dc.replace(base, channel=_dc.replace(base.channel, ack_mode=str(mode)))


# ================================================================== 单格
def run_cell(ep: int, seed: int, mode: str) -> dict:
    cfg = _cfg(mode)
    budget = cfg.protocol.budget(RATIO)
    pol = _Rec(cfg, h_plan=H_PLAN, m_samples=M_MAIN, seed=seed, rho=RHO,
               pacing=True, **TAIL_DP)
    t0 = time.perf_counter()
    res = run_episode(cfg, pol, split="dev", episode=ep, budget_total=budget, rho=RHO)
    elapsed = time.perf_counter() - t0
    chk = check_identities(res, cfg)
    att = [int(x) for x in res.n_attempted_by_user]
    dlv = [int(x) for x in res.n_delivered_by_user]
    nk = [int(x) for x in (res.n_nack_by_user or [])]
    ak = [int(x) for x in (res.n_ack_by_user or [])]
    rows = pol.rows
    dphi = [r["phat"][0] - r["phat"][1] for r in rows]
    pmin = [min(r["phat"][i] for r in rows) for i in range(len(att))]
    pla = [rows[-1]["phat"][i] for i in range(len(att))]
    ident_ok = all(att[i] == dlv[i] + nk[i] for i in range(len(att)))
    return {
        "ep": int(ep), "seed": int(seed), "ack_mode": str(mode),
        "n_attempted_by_user": att, "n_delivered_by_user": dlv,
        "n_ack_by_user": ak, "n_nack_by_user": nk,
        "attribution_identity_ok": bool(ident_ok),
        "delta_attempted": int(att[0] - att[1]),
        "delta_delivered": int(dlv[0] - dlv[1]),
        "n_probe": int(res.n_probe), "n_hold": int(res.n_idle),
        "n_forced_hold": int(res.n_forced_hold),
        "n_never_served": int(res.n_never_served),
        "aoi_tavg_by_user": res.aoi_tavg_by_user,
        "aoi_max_by_user": res.aoi_max_by_user,
        "cost_est_per_step": res.cost_est_per_step,
        "cost_true_per_step": res.cost_true_per_step,
        "cost_true_by_user": res.cost_true_by_user,
        "phat_min_by_user": pmin, "phat_last_by_user": pla,
        "phat_diff_median": (float(np.median(dphi)) if dphi else None),
        "phat_diff_max_abs": (float(np.max(np.abs(dphi))) if dphi else None),
        "policy_trace_hash": res.policy_trace_hash,
        "identities_pass": bool(chk["pass"]),
        "identities_fail": [k for k, v in chk["items"].items() if not v],
        "elapsed_s": round(elapsed, 2),
    }


# ================================================================== 前置：默认路径逐位未变
def isolation_check() -> dict:
    """`ack_mode="approx"` 重跑 R1.11 的两格，与**入库证据**逐位对账。"""
    ref = json.loads(R111.read_text(encoding="utf-8"))
    rmap = {(c["ep"], c["seed"], c["ack"]): c for c in ref}
    out = {"source": str(R111.relative_to(REPO)), "cells": []}
    for ep, seed in ((100, 2001), (101, 2003)):
        c = run_cell(ep, seed, "approx")
        r = rmap[(ep, seed, True)]
        d = {
            "ep": ep, "seed": seed,
            "cost_true_new": c["cost_true_per_step"], "cost_true_old": r["cost_true_per_step"],
            "cost_true_identical": c["cost_true_per_step"] == r["cost_true_per_step"],
            "aoi_tavg_identical": c["aoi_tavg_by_user"] == r["aoi_tavg_by_user"],
            "attempted_identical": c["n_attempted_by_user"] == r["n_attempted_by_user"],
            "trace_hash_identical": c["policy_trace_hash"] == r["policy_trace_hash"],
            "phat_diff_median_identical": c["phat_diff_median"] == r["phat_diff_median"],
        }
        out["cells"].append(d)
    out["all_identical"] = all(
        d["cost_true_identical"] and d["aoi_tavg_identical"] and d["attempted_identical"]
        and d["trace_hash_identical"] and d["phat_diff_median_identical"]
        for d in out["cells"])
    return out


# ================================================================== 主流程
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="CW-R1.12 ACK 尺度校准（预注册 v0.1）")
    ap.add_argument("--probe", action="store_true", help="只跑 1 格报预算，**不写任何证据**")
    ap.add_argument("--tag", default="main")
    args = ap.parse_args(argv)

    if args.probe:
        t0 = time.perf_counter()
        a = run_cell(100, 2001, "approx")
        b = run_cell(100, 2001, "exact")
        print(json.dumps({"approx": {k: a[k] for k in
                                     ("cost_true_per_step", "n_attempted_by_user",
                                      "n_probe", "elapsed_s", "policy_trace_hash")},
                          "exact": {k: b[k] for k in
                                    ("cost_true_per_step", "n_attempted_by_user",
                                     "n_probe", "elapsed_s", "policy_trace_hash")},
                          "elapsed_total_s": round(time.perf_counter() - t0, 1)},
                         ensure_ascii=False, indent=1, default=float))
        return 0

    t_start = time.perf_counter()
    head = _git(["rev-parse", "HEAD"])
    wt_sha = _git(["hash-object", "experiments/cw_v1/world.py"])
    blob_sha = _git(["rev-parse", f"{head}:experiments/cw_v1/world.py"])
    dirty = _git(["status", "--porcelain"])
    srchash = _src_hash()
    stamp = _now_utc()
    outdir = RESULTS / f"cw_r1_12_ackscale_{stamp}_{srchash}"
    outdir.mkdir(parents=True, exist_ok=False)
    print(f"[OUT] {outdir}")

    # ---- Q1（零 episode）
    q1 = q1_table()
    (outdir / "q1_table.json").write_text(json.dumps(q1, ensure_ascii=False, indent=1,
                                                    default=float), encoding="utf-8")
    print(f"[Q1] verdict={q1['verdict']} "
          f"fisher_ratio={q1['criterion']['r_eff_fisher(0,1)/ACK_VAR']:.3f} "
          f"mm_ratio={q1['extra_caliber_r_moment_matched_over_ACK_VAR_at_m0_v1']:.3f}")

    # ---- 闸
    t1 = time.perf_counter()
    gate = G.run_all()
    gate_s = round(time.perf_counter() - t1, 2)
    (outdir / "gates.json").write_text(json.dumps(gate, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
    print(f"[GATE] n_checks={gate['n_checks']} n_fail={gate['n_fail']} ({gate_s}s)")

    # ---- 隔离（默认路径逐位未变）
    iso = isolation_check()
    (outdir / "isolation.json").write_text(json.dumps(iso, ensure_ascii=False, indent=1),
                                           encoding="utf-8")
    print(f"[ISO] all_identical={iso['all_identical']}")

    # ---- PANEL
    cells: list[dict] = []
    for ep in EPS:
        for mode in ("approx", "exact"):
            for seed in SEEDS:
                t0 = time.perf_counter()
                c = run_cell(ep, seed, mode)
                c["wall_s"] = round(time.perf_counter() - t0, 2)
                cells.append(c)
                print(f"[CELL] ep{ep}/{mode}/seed{seed} att={c['n_attempted_by_user']} "
                      f"del={c['n_delivered_by_user']} cost_true={c['cost_true_per_step']:.5f} "
                      f"aoi={[None if x is None else round(x,2) for x in c['aoi_tavg_by_user']]} "
                      f"({c['wall_s']}s)")

    # ---- Q2 判词
    def med(xs):
        return float(np.median(xs))

    q2 = {}
    for ep in EPS:
        a = {c["seed"]: c for c in cells if c["ep"] == ep and c["ack_mode"] == "approx"}
        b = {c["seed"]: c for c in cells if c["ep"] == ep and c["ack_mode"] == "exact"}
        paired = [b[s]["cost_true_per_step"] - a[s]["cost_true_per_step"] for s in SEEDS]
        rel = [d / a[s]["cost_true_per_step"] for d, s in zip(paired, SEEDS)]
        diff_a = med([a[s]["cost_true_per_step"] for s in SEEDS])
        diff_b = med([b[s]["cost_true_per_step"] for s in SEEDS])
        q2[f"ep{ep}"] = {
            "cost_true_median_approx": diff_a, "cost_true_median_exact": diff_b,
            "delta_median_of_medians": diff_b - diff_a,
            "paired_delta_per_seed": paired,
            "paired_rel_per_seed": rel,
            "paired_median_delta": med(paired),
            "paired_mean_delta": float(np.mean(paired)),
            "floor_used": FLOOR, "floor_narrow": FLOOR_NARROW[f"ep{ep}"],
            "resolved_vs_floor": bool(abs(med(paired)) >= FLOOR),
            "resolved_vs_narrow": bool(abs(med(paired)) >= FLOOR_NARROW[f"ep{ep}"]),
        }
    signs = [np.sign(q2[f"ep{ep}"]["paired_median_delta"]) for ep in EPS]
    q2["verdict"] = ("EXACT-VS-APPROX-RESOLVED" if all(q2[f"ep{ep}"]["resolved_vs_floor"] for ep in EPS)
                     else ("PANEL-CANNOT-RESOLVE" if not any(q2[f"ep{ep}"]["resolved_vs_floor"]
                                                             for ep in EPS)
                           else "MIXED-BY-EPISODE"))
    q2["signs_consistent"] = bool(len({float(s) for s in signs}) == 1)

    # 逐用户（R17：任一 r_i > 1 必须写"交易"）
    per_user = {}
    for ep in EPS:
        a = [c for c in cells if c["ep"] == ep and c["ack_mode"] == "approx"]
        b = [c for c in cells if c["ep"] == ep and c["ack_mode"] == "exact"]
        rows = []
        for i in range(2):
            ra = [x["cost_true_by_user"][i] for x in a]
            rb = [x["cost_true_by_user"][i] for x in b]
            rows.append({"user": i, "median_approx": med(ra), "median_exact": med(rb),
                         "ratio_exact_over_approx": med(rb) / med(ra) if med(ra) else None,
                         "attempted_approx": [x["n_attempted_by_user"][i] for x in a],
                         "attempted_exact": [x["n_attempted_by_user"][i] for x in b],
                         "nack_approx": [x["n_nack_by_user"][i] for x in a],
                         "nack_exact": [x["n_nack_by_user"][i] for x in b],
                         "phat_min_approx": [x["phat_min_by_user"][i] for x in a],
                         "phat_min_exact": [x["phat_min_by_user"][i] for x in b]})
        per_user[f"ep{ep}"] = rows
    trade = any((r["ratio_exact_over_approx"] or 0) > 1.0
                for ep in EPS for r in per_user[f"ep{ep}"])

    # ---- 接线：两臂的轨迹必须不同（否则精确版没影响动作 ⇒ 效应全假）
    def _hash(ep: int, seed: int, mode: str) -> str:
        for c in cells:
            if c["ep"] == ep and c["seed"] == seed and c["ack_mode"] == mode:
                return c["policy_trace_hash"]
        return ""

    wiring = {}
    n_diff = 0
    for ep in EPS:
        rows_w = []
        for s in SEEDS:
            ha, hx = _hash(ep, s, "approx"), _hash(ep, s, "exact")
            n_diff += int(ha != hx)
            rows_w.append({"seed": s, "hash_approx": ha, "hash_exact": hx,
                           "different": bool(ha != hx)})
        wiring[f"ep{ep}"] = rows_w
    wiring["n_trace_diff"] = n_diff
    wiring["n_total"] = len(EPS) * len(SEEDS)
    wiring["wired"] = bool(n_diff == wiring["n_total"])

    summary = {
        "protocol": "CW-POMDP-v1.1",
        "stage": "CW-R1.12",
        "kind": "INFO-MODEL-CALIBRATION",
        "prereg": "docs/research/CW-R1.12-ACK尺度校准-预注册-v0.1.md",
        "q1_verdict": q1["verdict"],
        "q1_criterion": q1["criterion"],
        "q1_extra_caliber": q1["extra_caliber_r_moment_matched_over_ACK_VAR_at_m0_v1"],
        "q2_verdict": q2["verdict"],
        "q2": q2,
        "per_user": per_user,
        "trade_any_user_worse": trade,
        "wiring": {k: v for k, v in wiring.items()},
        "isolation_all_identical": iso["all_identical"],
        "n_cells": len(cells),
        "floor": FLOOR,
        "head": head, "world_py_worktree_sha": wt_sha, "world_py_blob_sha": blob_sha,
        "src_hash": srchash, "git_dirty_before_run": dirty.splitlines(),
        "wall_s_total": round(time.perf_counter() - t_start, 1),
        "note": ("`approx` 臂与 R1.11 同参（`ack=True`）；`exact` 臂只把 ACK/NACK 的"
                 "信息模型换成网格精确贝叶斯（PROBE 一并进网格，见 world.step）。"
                 "两臂同种子消耗的随机数位置相同 ⇒ 同 seed 是**配对**，不是独立样本。"),
    }
    (outdir / "cells.json").write_text(json.dumps(cells, ensure_ascii=False, indent=1,
                                                  default=float), encoding="utf-8")
    (outdir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1,
                                                    default=float), encoding="utf-8")

    try:
        import psutil  # type: ignore
        rss = psutil.Process(os.getpid()).memory_info().rss / 1024**2
    except Exception:
        rss = None
    meta = {
        "command": "python -m experiments.cw_v1.r1_12_ackscale --tag " + str(args.tag),
        "python": os.sys.version.split()[0], "numpy": np.__version__,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "head": head, "src_hash": srchash,
        "world_worktree_sha": wt_sha, "world_blob_sha": blob_sha,
        "gate_elapsed_s": gate_s, "total_elapsed_s": round(time.perf_counter() - t_start, 1),
        "peak_rss_mb": round(rss, 1) if rss else None,
        "n_fail_gates": gate["n_fail"], "n_checks_gates": gate["n_checks"],
        "n_fail_cells": sum(1 for c in cells if not c["identities_pass"]),
    }
    (outdir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                      encoding="utf-8")

    n_fail = gate["n_fail"] + sum(1 for c in cells if not c["identities_pass"]) \
        + (0 if iso["all_identical"] else 1) + (0 if wiring["wired"] else 1)
    print(f"[DONE] {outdir.name} n_fail={n_fail} total={meta['total_elapsed_s']}s "
          f"peak_rss={meta['peak_rss_mb']}MB")
    print(json.dumps({"q1": summary["q1_verdict"], "q2": summary["q2_verdict"],
                      "iso": iso["all_identical"], "wired": wiring["wired"],
                      "trade": trade, "floor": FLOOR,
                      "ep100_delta": q2["ep100"]["paired_median_delta"],
                      "ep101_delta": q2["ep101"]["paired_median_delta"]},
                     ensure_ascii=False, indent=1, default=float))
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    import sys
    sys.exit(main())
