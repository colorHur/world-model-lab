# -*- coding: utf-8 -*-
"""CW-R1.14 精确贝叶斯升为**研究基线** —— 身份闸 + 标定（calibration）。

预注册：`docs/research/CW-R1.14-精确基线-预注册-v0.1.md`（§8 记 3 条，均写于见数之前）

三段
----
`P0` **前置闸**：① 全量 `gates.py` `n_fail=0`（含新增 `4u/4v/4w/4x`）；
   ② 16 格身份 —— `approx`/`exact` 两臂各 8 格必须与 R1.13 同名臂**逐位**相同（5 字段）。
`PANEL` **16 格**：`arm ∈ {approx(0.63 旧基线), exact(新主线)} × ep ∈ {100,101} × seed ∈ {2001..2004}`。
`CAL` **标定量**：`R_var = mean((H−m)²)/mean(v)`（理论值 1，期望上恒等）、`mean_z2`、
   `P(|z|>2)/P(|z|>3)`、逐用户、逐 ep、逐格、全池化 —— **五个层级全报**。
`V` **判词**：预注册 §4.2 + §8-1（主口径 = 8 格中位数；三口径方向不一致 ⇒ `POOLING-CONFLICT`）。

★ 本轮**不动** `0.63` / `±1.0/−1.5`（`4t` 钉住）、不改 `ChannelConfig.ack_mode` 默认值、
  不改网格（`4n/4q` 钉住）、不跑 `cost_true` 的优劣比较（分母 < floor）。
"""

from __future__ import annotations

import argparse
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

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results" / "cw_v1"
R113_DIR = RESULTS / "cw_r1_13_scalar_20261024T041643_4bae71ec8c126d1d"
R113_CELLS = R113_DIR / "cells.json"

RHO = 0.98
H_PLAN = 5
M_MAIN = 16
RATIO = 0.5
EPS = (100, 101)
SEEDS = (2001, 2002, 2003, 2004)
TAIL_DP = {"tail": "dp", "dp_grid": "r173"}
ARMS = ("approx", "exact")

#: 预注册 §4.2 的判定容差：`|R_var − 1|` 的比较是**严格小于**，不设"等价带"（等号即否定条件命中）
#: ★ 不设等价带的理由：设了等价带就等于允许"两臂都差不多"这种不可证伪的结论。


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


# ==================================================================== 配置（两臂）
def _cfg(arm: str) -> C.CWConfig:
    if arm == "exact":
        return C.mainline_config()                     # ★ 新基线：BOUNDED + exact
    # 旧基线：与 R1.13 的 `_cfg("0.63")` **逐字段**同构（`bounded_plant()` + 默认 channel）
    return C.CWConfig(plant=C.bounded_plant())


# ==================================================================== 记录器
class _Rec(Pmodule.PlanRef):
    """只观测：逐槽公开 telemetry（`h_mean/h_var`）+ 根节点 `p̂`；不改任何内层行为。"""

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


# ==================================================================== 标定
def calibration(res, rows: list[dict]) -> dict:
    """`R_var` 等标定量。★ 分母口径 = `mean(v)`（预注册 §4.2），理由见卡内。"""
    H = np.asarray(res.log_h_hist, dtype=float)
    if H.ndim != 2 or H.shape[0] == 0:
        return {"status": "SKIP-NO-OBJECT", "n_slots": 0}
    T, n = H.shape
    if len(rows) != T:
        # ★ 不对齐就**不许算**（先对账再比，教训 #49/#54）
        return {"status": "ALIGN-MISMATCH", "n_slots": T, "n_rows": len(rows)}
    m = np.asarray([[r["h_mean"][i] for i in range(n)] for r in rows], dtype=float)
    v = np.asarray([[r["h_var"][i] for i in range(n)] for r in rows], dtype=float)
    err2 = (H - m) ** 2
    z2 = err2 / v
    z = (H - m) / np.sqrt(v)
    mean_err2 = float(err2.mean())
    mean_v = float(v.mean())
    out = {
        "status": "OK", "n_slots": int(T), "n_users": int(n),
        # 主口径：均值的比值（只有它才有 E[(H−m)²]=E[v] 这个恒等式）
        "R_var": mean_err2 / mean_v,
        "mean_err2": mean_err2, "mean_v": mean_v,
        # 描述量：比值的均值（非高斯下本来就 >1，不许当"标定失败"）
        "mean_z2": float(z2.mean()),
        "z_mean": float(z.mean()),
        "p_abs_z_gt2": float((np.abs(z) > 2).mean()),
        "p_abs_z_gt3": float((np.abs(z) > 3).mean()),
        "by_user": {}, "by_ep": int(res.episode),
    }
    for i in range(n):
        out["by_user"][f"u{i}"] = {
            "R_var": float(err2[:, i].mean() / v[:, i].mean()),
            "mean_z2": float(z2[:, i].mean()),
            "p_abs_z_gt2": float((np.abs(z[:, i]) > 2).mean()),
            "mean_v": float(v[:, i].mean()),
            "mean_err2": float(err2[:, i].mean()),
        }
    return out


# ==================================================================== 单格
def run_cell(ep: int, seed: int, arm: str) -> dict:
    cfg = _cfg(arm)
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
    pmin = [min(r["phat"][i] for r in rows) for i in range(len(att))]
    cal = calibration(res, rows)
    return {
        "ep": int(ep), "seed": int(seed), "arm": str(arm),
        "ack_mode": str(cfg.channel.ack_mode),
        "config_fingerprint": res.config_fingerprint,
        "n_attempted_by_user": att, "n_delivered_by_user": dlv,
        "n_ack_by_user": ak, "n_nack_by_user": nk,
        "attribution_identity_ok": bool(all(att[i] == dlv[i] + nk[i] for i in range(len(att)))),
        "n_probe": int(res.n_probe), "n_hold": int(res.n_idle),
        "n_never_served": int(res.n_never_served),
        "aoi_tavg_by_user": res.aoi_tavg_by_user,
        "cost_est_per_step": res.cost_est_per_step,
        "cost_true_per_step": res.cost_true_per_step,
        "cost_true_by_user": res.cost_true_by_user,
        "phat_min_by_user": pmin,
        "policy_trace_hash": res.policy_trace_hash,
        "identities_pass": bool(chk["pass"]),
        "identities_fail": [k for k, v in chk["items"].items() if not v],
        "calib": cal,
        "elapsed_s": round(elapsed, 2),
    }


# ==================================================================== 身份闸
#: 本轮的臂名 → R1.13 `cells.json` 里的 `arm` 字段。
#: ★★ 首跑 `approx 0/8` 的**根因就是这张表**：第一版用 `ack_mode` 当键，而 R1.13 的
#:   `0.63`/`1.000`/`1.726` 三臂的 `ack_mode` 都是 `"approx"` ⇒ 字典只留最后一个
#:   （`1.726`）⇒ 拿 1.726 的读数去对 0.63 的读数，**必然 0/8**。
#:   ⇒ 参照物必须按**臂名**索引，不许按"模式名"索引（#85「比错对象」同族）。
REF_ARM = {"approx": "0.63", "exact": "exact"}


def identity_check(arm: str, ref_map: dict, cells: list[dict]) -> dict:
    """与 R1.13 同名臂逐位比对（5 字段）。`ref_map` 的键 = `(ep, seed, arm名)`。"""
    rows = []
    for c in cells:
        if c["arm"] != arm:
            continue
        r = ref_map[(c["ep"], c["seed"], REF_ARM[arm])]
        d = {"ep": c["ep"], "seed": c["seed"],
             "cost_true_identical": c["cost_true_per_step"] == r["cost_true_per_step"],
             "aoi_tavg_identical": c["aoi_tavg_by_user"] == r["aoi_tavg_by_user"],
             "attempted_identical": c["n_attempted_by_user"] == r["n_attempted_by_user"],
             "phat_min_identical": c["phat_min_by_user"] == r["phat_min_by_user"],
             "trace_hash_identical": c["policy_trace_hash"] == r["policy_trace_hash"]}
        d["all_identical"] = all(v for k, v in d.items() if k.endswith("_identical"))
        rows.append(d)
    return {"source": str(R113_CELLS.relative_to(REPO)), "arm": arm,
            "n_cells": len(rows), "n_identical": sum(1 for d in rows if d["all_identical"]),
            "all_identical": bool(rows) and all(d["all_identical"] for d in rows),
            "cells": rows}


# ==================================================================== 判词
def verdict_M2(cells: list[dict]) -> dict:
    """预注册 §4.2 + §8-1：三口径（格中位 / 全池化 / 逐 ep 中位）全报，方向不一致 ⇒ POOLING-CONFLICT。"""
    per = {}
    for arm in ARMS:
        cs = [c for c in cells if c["arm"] == arm and c["calib"].get("status") == "OK"]
        if not cs:
            return {"verdict": "VOID-NO-CALIB", "detail": f"{arm} 无可算标定的格"}
        per[arm] = {
            "n_cells": len(cs),
            "R_var_median": float(np.median([c["calib"]["R_var"] for c in cs])),
            "R_var_cells": [c["calib"]["R_var"] for c in cs],
            "mean_z2_median": float(np.median([c["calib"]["mean_z2"] for c in cs])),
            "p_abs_z_gt2_median": float(np.median([c["calib"]["p_abs_z_gt2"] for c in cs])),
            "p_abs_z_gt3_median": float(np.median([c["calib"]["p_abs_z_gt3"] for c in cs])),
            "pooled_R_var": (float(sum(c["calib"]["mean_err2"] * c["calib"]["n_slots"]
                                       for c in cs))
                             / float(sum(c["calib"]["mean_v"] * c["calib"]["n_slots"]
                                         for c in cs))),
            "by_ep_median_R_var": {
                f"ep{ep}": float(np.median([c["calib"]["R_var"] for c in cs
                                            if c["ep"] == ep]))
                for ep in EPS if any(c["ep"] == ep for c in cs)},
            "by_user_R_var_median": {
                f"u{i}": float(np.median([c["calib"]["by_user"][f"u{i}"]["R_var"] for c in cs]))
                for i in range(len(cs[0]["calib"]["by_user"]))},
        }
    d_med = {a: abs(per[a]["R_var_median"] - 1.0) for a in ARMS}
    d_pool = {a: abs(per[a]["pooled_R_var"] - 1.0) for a in ARMS}
    d_ep = {}
    for ep in EPS:
        k = f"ep{ep}"
        vals = {a: per[a]["by_ep_median_R_var"].get(k) for a in ARMS}
        if all(v is not None for v in vals.values()):
            d_ep[k] = {a: abs(vals[a] - 1.0) for a in ARMS}
    signs = {a: bool(d_med[a] < d_med["approx"]) for a in ("exact",)}
    sign_pool = bool(d_pool["exact"] < d_pool["approx"])
    sign_ep = {k: bool(v["exact"] < v["approx"]) for k, v in d_ep.items()}
    all_signs = [signs["exact"], sign_pool, *sign_ep.values()]
    if all(all_signs):
        v = "EXACT-CALIBRATED-BETTER"
    elif not any(all_signs):
        v = "CALIBRATION-NOT-BETTER"          # 预注册否定条件命中 ⇒ A 的措辞必须降级
    else:
        v = "POOLING-CONFLICT"
    return {
        "verdict": v,
        "per_arm": per,
        "abs_dev_from_1": {"cell_median": d_med, "pooled": d_pool, "by_ep": d_ep},
        "sign_exact_better": {"cell_median": signs["exact"], "pooled": sign_pool,
                              "by_ep": sign_ep},
        "detail": ("主口径 = 8 格中位；三口径（格中位/全池化/逐 ep）全报。"
                   "`mean_z2` 是描述量：非高斯后验下本来就 >1，不许当标定失败。"),
        "caveat": ("有效独立单元 = 2 个 episode；槽间强自相关 ⇒ 不许用 256×2×8 个槽谈显著性。"),
    }


# ==================================================================== 主流程
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="CW-R1.14 精确基线（预注册 v0.1）")
    ap.add_argument("--probe", action="store_true", help="只跑 2 格报预算，**不写任何证据**")
    ap.add_argument("--tag", default="main")
    args = ap.parse_args(argv)

    if args.probe:
        t0 = time.perf_counter()
        a = run_cell(100, 2001, "approx")
        b = run_cell(100, 2001, "exact")
        print(json.dumps({"approx": {k: a[k] for k in ("cost_true_per_step", "n_attempted_by_user",
                                                        "phat_min_by_user", "elapsed_s")},
                          "exact": {k: b[k] for k in ("cost_true_per_step", "n_attempted_by_user",
                                                      "phat_min_by_user", "elapsed_s")},
                          "calib_approx": a["calib"].get("R_var"),
                          "calib_exact": b["calib"].get("R_var"),
                          "trace_differs": a["policy_trace_hash"] != b["policy_trace_hash"],
                          "elapsed_total_s": round(time.perf_counter() - t0, 1)},
                         ensure_ascii=False, indent=1, default=float))
        return 0

    t_start = time.perf_counter()
    head = _git(["rev-parse", "HEAD"])
    wt_sha = _git(["hash-object", "experiments/cw_v1/world.py"])
    blob_sha = _git(["rev-parse", f"{head}:experiments/cw_v1/world.py"])
    dirty = _git(["status", "--porcelain"])
    srchash = _src_hash()
    outdir = RESULTS / f"cw_r1_14_exactbase_{_now_utc()}_{srchash}"
    outdir.mkdir(parents=True, exist_ok=False)
    print(f"[OUT] {outdir}")

    # ---- P0-a 全量闸
    t1 = time.perf_counter()
    gate = G.run_all()
    gate_s = round(time.perf_counter() - t1, 2)
    (outdir / "gates.json").write_text(json.dumps(gate, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
    print(f"[GATE] n_checks={gate['n_checks']} n_fail={gate['n_fail']} ({gate_s}s)")

    # ---- PANEL 16 格
    cells: list[dict] = []
    for arm in ARMS:
        for ep in EPS:
            for seed in SEEDS:
                t0 = time.perf_counter()
                c = run_cell(ep, seed, arm)
                c["wall_s"] = round(time.perf_counter() - t0, 2)
                cells.append(c)
                cal = c["calib"]
                print(f"[CELL] {arm:>6}/ep{ep}/s{seed} att={c['n_attempted_by_user']} "
                      f"del={c['n_delivered_by_user']} pmin="
                      f"{[round(x, 4) for x in c['phat_min_by_user']]} "
                      f"R_var={cal.get('R_var'):.4f} z2={cal.get('mean_z2'):.3f} "
                      f"cost={c['cost_true_per_step']:.5f} ({c['wall_s']}s)")

    # ---- P0-b 身份（16/16）
    ref = json.loads(R113_CELLS.read_text(encoding="utf-8"))
    ref_map = {(r["ep"], r["seed"], r["arm"]): r for r in ref}
    identity = {a: identity_check(a, ref_map, cells) for a in ARMS}
    (outdir / "identity.json").write_text(json.dumps(identity, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    print("[ID] " + " · ".join(f"{a} {v['n_identical']}/{v['n_cells']}"
                               for a, v in identity.items()))

    # ---- V 判词
    vd = verdict_M2(cells)
    (outdir / "calib.json").write_text(json.dumps(vd, ensure_ascii=False, indent=1,
                                                  default=float), encoding="utf-8")
    print(f"[M2] {vd['verdict']} | abs_dev cell_median="
          f"{ {a: round(x, 4) for a, x in vd['abs_dev_from_1']['cell_median'].items()} } "
          f"pooled={ {a: round(x, 4) for a, x in vd['abs_dev_from_1']['pooled'].items()} }")

    # ---- 逐用户机械量表（描述，不作判据）
    split_tbl = {}
    for arm in ARMS:
        cs = [c for c in cells if c["arm"] == arm]
        split_tbl[arm] = {
            "att_by_user": [sum(c["n_attempted_by_user"][i] for c in cs) for i in range(2)],
            "del_by_user": [sum(c["n_delivered_by_user"][i] for c in cs) for i in range(2)],
            "probe_total": int(sum(c["n_probe"] for c in cs)),
            "never_served_total": int(sum(c["n_never_served"] for c in cs)),
            "cost_true_median": float(np.median([c["cost_true_per_step"] for c in cs])),
            "cost_est_median": float(np.median([c["cost_est_per_step"] for c in cs])),
            "phat_min_median": [float(np.median([c["phat_min_by_user"][i] for c in cs]))
                                for i in range(2)],
            "R_var_cells": [c["calib"].get("R_var") for c in cs],
        }

    m1_ok = all(v["all_identical"] for v in identity.values())
    n_fail = (gate["n_fail"] + sum(1 for c in cells if not c["identities_pass"])
              + sum(1 for c in cells if c["calib"].get("status") != "OK")
              + (0 if m1_ok else 1))
    summary = {
        "protocol": "CW-POMDP-v1.1",
        "stage": "CW-R1.14",
        "kind": "EXACT-AS-MAINLINE",
        "prereg": "docs/research/CW-R1.14-精确基线-预注册-v0.1.md",
        "decision": "A",
        "mainline": {"MAINLINE_ACK_MODE": str(C.MAINLINE_ACK_MODE),
                     "default_channel_ack_mode": str(C.ChannelConfig().ack_mode),
                     "plant": "BOUNDED_PLANT"},
        "M1_identity": {"all_identical": bool(m1_ok),
                        "detail": {a: f"{v['n_identical']}/{v['n_cells']}"
                                   for a, v in identity.items()}},
        "M2_verdict": vd["verdict"],
        "M2_per_arm": {a: {k: v[k] for k in ("n_cells", "R_var_median", "R_var_cells",
                                             "mean_z2_median", "p_abs_z_gt2_median",
                                             "p_abs_z_gt3_median", "pooled_R_var",
                                             "by_ep_median_R_var", "by_user_R_var_median")}
                       for a, v in vd["per_arm"].items()},
        "M3_split_table": split_tbl,
        "gates": {"n_checks": gate["n_checks"], "n_fail": gate["n_fail"]},
        "n_cells": len(cells),
        "head": head, "src_hash": srchash,
        "world_py_worktree_sha": wt_sha, "world_py_blob_sha": blob_sha,
        "git_dirty_before_run": dirty.splitlines(),
        "note": ("两臂只差 ACK/NACK 的**信息模型**（approx = 伪观测一阶近似；exact = 网格精确贝叶斯）；"
                 "同 seed 是配对不是独立样本；有效独立单元 = 2 个 episode。"
                 "`cost_true` 只报不比（R1.13 已证分母 < floor）。"),
    }
    (outdir / "cells.json").write_text(json.dumps(cells, ensure_ascii=False, indent=1,
                                                  default=float), encoding="utf-8")
    (outdir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1,
                                                    default=float), encoding="utf-8")
    try:
        import psutil  # type: ignore
        rss = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
    except Exception:
        rss = None
    meta = {"command": "python -m experiments.cw_v1.r1_14_exactbase --tag " + str(args.tag),
            "python": os.sys.version.split()[0], "numpy": np.__version__,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "head": head, "src_hash": srchash,
            "world_worktree_sha": wt_sha, "world_blob_sha": blob_sha,
            "gate_elapsed_s": gate_s, "total_elapsed_s": round(time.perf_counter() - t_start, 1),
            "peak_rss_mb": round(rss, 1) if rss else None,
            "n_fail_gates": gate["n_fail"], "n_checks_gates": gate["n_checks"],
            "n_fail_cells": sum(1 for c in cells if not c["identities_pass"])}
    (outdir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    print(f"[DONE] {outdir.name} n_fail={n_fail} total={meta['total_elapsed_s']}s "
          f"peak_rss={meta['peak_rss_mb']}MB")
    print(json.dumps({"M1": summary["M1_identity"], "M2": vd["verdict"],
                      "abs_dev": vd["abs_dev_from_1"]},
                     ensure_ascii=False, indent=1, default=float))
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    import sys
    sys.exit(main())
