# -*- coding: utf-8 -*-
"""`CW-R2.3` 落盘入口：把面板从 `n=20` 扩到 `n=30`（`ep110–114` 为**全新**单元）。

预注册：`docs/research/CW-R2.3-扩N到30-预注册-v0.1.md`（写于 2026-10-25 08:08，`run_episode`=0）

五段
----
`CTRL` 判据自证 `C1–C8`：**复用** `r2_1_sep.controls()`（零 episode）。
`C9`  子集对账：用 `mech2` 重算 `ep100–109` 的 20 单位，必须**逐字**复现 R2.2 三判词。
`GATE` `B1–B8`：**复用** `r2_1_sep.gates()`（11 项，含 `B3` 直通恒等 + 活体负控）。
`I1`  身份闸：`ep100–109` 的 40 格 × 5 字段与 R2.1 `cells.json` **逐位相同**（20 零控同）。
`PANEL` **90 格**：面板 `2 × 2 × 15 = 60` + 零控 `2 × 15 = 30`。

★ 本轮**不训练**、不改 `sigma_meas`、不改 `mech2.py` 任何常量；被测物
  `plan_ref` / `plan_dp` / `world` / `runner` / `config` / `policies` / `r2_1_sep.py` **一字节不动**。
★ 主判是 **`FRESH` 腿（`ep110–114`，`n=10`，`k_need=9`）**，不是合并 `n=30` —— 理由见预注册 §2.1：
  扩样由 14/20 触发，合并检验的条件化 I 类错误率 ≈ 0.3770（名义 0.05 的 7.5 倍）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import gates as G
from . import mech2
from .model_api import OBJECTIVE, spec_from_config
from .r1_14_exactbase import _git, _src_hash
from .r2_1_sep import (MODELS, PLANTS, RATIO, SEED_NULL, SEED_PANEL, controls,
                       gates as r21_gates, run_cell)

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results" / "cw_v1"

#: ★ 写死的输入（身份闸 `I1` 与子集对账 `C9` 都用它）
R21_DIR = RESULTS / "cw_r2_1_sep_20261024T194859_3c71623d6afe07e4"
R22_DIR = RESULTS / "cw_r2_2_criteria_20261024T215859_2d54069c2b4d767e"

EPS_REUSE = list(range(100, 110))          # 复用（R2.1 已跑）
EPS_FRESH = list(range(110, 115))          # 全新
EPS_ALL = EPS_REUSE + EPS_FRESH            # 15
PLANT_KEYS = ["P1", "P2"]

#: `alpha`（与 `mech2.ALPHA` 一致；**本文件不改它**，只用于写判词表）
ALPHA = mech2.ALPHA

#: R2.2 的三判词 + 数字（**写死**，`C9` 逐字对账）
R22_EXPECT = {
    "ALL": {"verdict": "SIGN-ONLY-NO-MAGNITUDE", "n_pos": 17, "n_neg": 3, "n_beat": 14,
            "median_abs_delta": 0.04563087743827283, "median_floor_pair": 0.02215760874931358},
    "P1": {"verdict": "CANNOT-SEPARATE", "n_pos": 8, "n_neg": 2, "n_beat": 8,
           "median_abs_delta": 0.07067629706211606, "median_floor_pair": 0.02215760874931358},
    "P2": {"verdict": "SIGN-ONLY-NO-MAGNITUDE", "n_pos": 9, "n_neg": 1, "n_beat": 6,
           "median_abs_delta": 0.040086324746157274, "median_floor_pair": 0.02445862844931726},
}

#: 身份闸逐位比较的 5 个字段（预注册 §6-4）
ID_FIELDS = ("cost_true_per_step", "cost_est_per_step", "policy_trace_hash",
             "n_attempted", "n_delivered")


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


# ==================================================================== 判词映射
def fresh_verdict(b: dict) -> dict:
    """预注册 §5.1 的映射（**写死**）。`n=10` 时 `h_beat ⇔ n_beat ≥ 9`。"""
    h_sign = bool(b["binom_p_sign"] < ALPHA)
    h_neg = bool(b["binom_p_neg"] < ALPHA)
    h_beat = bool(b["binom_p_beat"] < ALPHA)
    if h_beat and h_sign:
        v, nxt = "FRESH-MAGNITUDE-SEPARATES", "ENTER-TRAINING"
    elif h_beat and h_neg:
        v, nxt = "FRESH-SEPARATES-REVERSED", "STOP-AND-REPORT-REVERSED"
    elif h_sign:
        v, nxt = "FRESH-SIGN-ONLY-NO-MAGNITUDE", "SWITCH-METRIC:mean_task_aoi"
    elif h_neg:
        v, nxt = "FRESH-SIGN-ONLY-REVERSED", "SWITCH-METRIC:mean_task_aoi (反向必报)"
    else:
        v, nxt = "FRESH-MAGNITUDE-UNRESOLVED", "SWITCH-METRIC:mean_task_aoi"
    return {"verdict": v, "next_step": nxt, "h_sign": h_sign, "h_neg": h_neg,
            "h_beat": h_beat,
            "note": ("`n_beat<9` **不等于**无效应：真实 beat 率 0.70 时 n=10 的功效仅 0.1493"
                     "（预注册 §2.3）")}


def k_need_table() -> dict:
    """给 `n ∈ {10,20,30}` 在各 `alpha` 下的最小显著计数（写交付时用，跑前一并算）。"""
    out = {}
    for n in (10, 20, 30):
        row = {}
        for a in (0.05, 0.025, 0.01):
            ks = [k for k in range(n + 1) if mech2.binom_p_ge(k, n) < a]
            row[str(a)] = {"k_need": (min(ks) if ks else None),
                           "p_at_k_need": (mech2.binom_p_ge(min(ks), n) if ks else None),
                           "p_at_k_need_minus_1": (mech2.binom_p_ge(min(ks) - 1, n)
                                                   if ks and min(ks) > 0 else None)}
        # 合并规则在"已见 14/20"条件下的实效 I 类错误率（预注册 §2.1）
        row["pooled_conditional_typeI"] = mech2.binom_p_ge(max(0, 20 - 14), 10)
        out[str(n)] = row
    return out


# ==================================================================== 子集对账 C9
def subset_reconcile(rows30: list[dict]) -> dict:
    """`ep100–109` 的 20 单位用**同一判据**重算，必须逐字复现 R2.2。"""
    rows20 = [r for r in rows30 if int(r["ep"]) in EPS_REUSE]
    v = mech2.verdicts_v2(rows20, PLANT_KEYS, EPS_REUSE)
    det: dict = {"n_units": len(rows20), "got": {}, "expect": R22_EXPECT, "items": {}}
    ok_all = True
    for blk, exp in R22_EXPECT.items():
        got = v[blk]
        det["got"][blk] = {"verdict": got["verdict"], "n_pos": got["n_pos"],
                           "n_neg": got["n_neg"], "n_beat": got["n_beat"],
                           "median_abs_delta": got["median_abs_delta"],
                           "median_floor_pair": got["median_floor_pair"]}
        items = {
            "verdict_same": got["verdict"] == exp["verdict"],
            "n_pos_same": int(got["n_pos"]) == int(exp["n_pos"]),
            "n_neg_same": int(got["n_neg"]) == int(exp["n_neg"]),
            "n_beat_same": int(got["n_beat"]) == int(exp["n_beat"]),
            "median_abs_delta_same": abs(float(got["median_abs_delta"])
                                         - float(exp["median_abs_delta"])) <= 1e-12,
            "median_floor_same": abs(float(got["median_floor_pair"])
                                     - float(exp["median_floor_pair"])) <= 1e-12,
        }
        det["items"][blk] = items
        ok_all = ok_all and all(items.values())
    det["ok"] = bool(ok_all and len(rows20) == 20)
    return det


# ==================================================================== 身份闸 I1
def identity_gate(cells: list[dict], nulls: list[dict]) -> dict:
    """`ep100–109` 的 40 格 + 20 零控与 R2.1 落盘逐位相同（容差 0）。"""
    r21 = _load(R21_DIR / "cells.json")
    r21n = _load(R21_DIR / "null.json")["nulls"]
    idx21 = {(c["plant"], c["model"], int(c["ep"])): c for c in r21}
    idx21n = {(c["plant"], int(c["ep"])): c for c in r21n}
    diffs: list[dict] = []
    checked = 0
    for c in cells:
        if int(c["ep"]) not in EPS_REUSE:
            continue
        ref = idx21.get((c["plant"], c["model"], int(c["ep"])))
        if ref is None:
            diffs.append({"key": f'{c["plant"]}/{c["model"]}/ep{c["ep"]}', "why": "R2.1 无此格"})
            continue
        checked += 1
        for f in ID_FIELDS:
            a, b = c[f], ref[f]
            same = (a == b) if not isinstance(a, float) else (float(a) == float(b))
            if not same:
                diffs.append({"key": f'{c["plant"]}/{c["model"]}/ep{c["ep"]}',
                              "field": f, "now": a, "r21": b})
    for c in nulls:
        if int(c["ep"]) not in EPS_REUSE:
            continue
        ref = idx21n.get((c["plant"], int(c["ep"])))
        if ref is None:
            diffs.append({"key": f'{c["plant"]}/null/ep{c["ep"]}', "why": "R2.1 无此零控"})
            continue
        checked += 1
        for f in ID_FIELDS:
            a, b = c[f], ref[f]
            same = (a == b) if not isinstance(a, float) else (float(a) == float(b))
            if not same:
                diffs.append({"key": f'{c["plant"]}/null/ep{c["ep"]}',
                              "field": f, "now": a, "r21": b})
    return {"ok": not diffs, "n_checked": checked, "n_diff": len(diffs),
            "diffs": diffs[:20],
            "expect_n": len(PLANT_KEYS) * len(EPS_REUSE) * (len(MODELS) + 1),
            "why": "复用单元若是另一个世界的读数 ⇒ 合并 n=30 与 floor 全部作废"}


# ==================================================================== 次级指标（描述）
def _aoi_mean(c: dict):
    """`mean_task_aoi` 的**逐单位**版本。

    ★★ `aoi_tavg_by_user[i] is None` ⇔ 该用户**全程从未送达**（`runner.py:278`：
       `aoi_tavg = mean(served)`，`served` 为空则 `None`）。此时单位均值**无定义**，
       返回 `None`，**不许**只对剩下的用户取平均 —— 那正是 R1.8 判死的
       「删未送达用户」偏差。调用方必须把该单位标 `undefined` 并计数。
    """
    vals = list(c["aoi_tavg_by_user"])
    if not vals or any(v is None for v in vals):
        return None
    return float(np.mean([float(v) for v in vals]))


def _aoi_stats(rows: list[dict]) -> dict:
    ok = [r for r in rows if r["d"] is not None]
    und = [f'{r["plant"]}/ep{r["ep"]}' for r in rows if r["d"] is None]
    if not ok:
        return {"n": 0, "n_undefined": len(und), "undefined": und}
    d = np.array([r["d"] for r in ok], dtype=float)
    f = np.array([r["floor"] for r in ok], dtype=float)
    return {"n": int(d.size), "n_undefined": len(und), "undefined": und,
            "n_pos": int((d > 0).sum()), "n_neg": int((d < 0).sum()),
            "n_beat": int((np.abs(d) > f).sum()),
            "median_abs_delta": float(np.median(np.abs(d))),
            "median_floor": float(np.median(f))}


def secondary_aoi(cells: list[dict], nulls: list[dict]) -> dict:
    """预注册 §5.4：`mean_task_aoi` 逐单位配对差。**不出判词**。"""
    rows = []
    for plant in PLANT_KEYS:
        for ep in EPS_ALL:
            b = next(c for c in cells if c["plant"] == plant
                     and c["model"] == "belief_readout" and int(c["ep"]) == ep)
            w = next(c for c in cells if c["plant"] == plant
                     and c["model"] == "prior_only" and int(c["ep"]) == ep)
            n2 = next(c for c in nulls if c["plant"] == plant and int(c["ep"]) == ep)
            mb, mw, mn = _aoi_mean(b), _aoi_mean(w), _aoi_mean(n2)
            undef = (mb is None) or (mw is None) or (mn is None)
            rows.append({"plant": plant, "ep": int(ep),
                         "d": None if undef else (mw - mb),
                         "floor": None if undef else abs(mb - mn),
                         "undefined_side": ([s for s, v in (("strong", mb), ("weak", mw),
                                                           ("null", mn)) if v is None]
                                            if undef else []),
                         # 伴随量：**永远有定义**，用来看排除是否选择性
                         "never_served_users": int(sum(int(c["n_never_served"])
                                                       for c in (b, w, n2)))})
    fresh = [r for r in rows if int(r["ep"]) in EPS_FRESH]
    return {"rows": rows, "n30": _aoi_stats(rows), "fresh10": _aoi_stats(fresh),
            "n_units_undefined": int(sum(1 for r in rows if r["d"] is None)),
            "n_units_unserved_any": int(sum(1 for r in rows if r["undefined_side"])),
            "note": ("描述量，不进判词；单位内任一臂有用户从未送达 ⇒ 该单位 `mean_task_aoi` 无定义"
                     "（不删分母），`never_served_users` 作伴随量")}


# ==================================================================== 主流程
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="CW-R2.3 面板 n=20 → n=30")
    ap.add_argument("--ctrl-only", action="store_true", help="只跑判据自证 + 子集对账（零 episode）")
    ap.add_argument("--no-fresh", action="store_true",
                    help="只重跑复用单元并做身份闸（调试用；不进判词）")
    ap.add_argument("--tag", default="main")
    args = ap.parse_args(argv)

    t_start = time.perf_counter()

    # ---- 1 CTRL（零 episode）
    ctrl = controls()
    kk = ctrl.pop("_keys")
    print(f"[CTRL] all_pass={ctrl['all_pass']} " +
          " ".join(f"{k}={ctrl[k]['pass']}" for k in kk))
    kn = k_need_table()
    print("[K_NEED] " + json.dumps({n: {a: kn[n][a]["k_need"] for a in ("0.05", "0.025", "0.01")}
                                    for n in kn}, ensure_ascii=False))
    print("[POOLED-RULE-TYPE-I] " + json.dumps(
        {n: kn[n]["pooled_conditional_typeI"] for n in kn}, ensure_ascii=False))

    if not ctrl["all_pass"]:
        print("[ABORT] 判据自证未过 ⇒ 按预注册 §6-1 不看闸与面板")
        return 3

    # ---- 2 面板（CTRL 过后才跑；--ctrl-only 在此返回）
    if args.ctrl_only:
        print(json.dumps({"ctrl": ctrl, "k_need": kn}, ensure_ascii=False, indent=1, default=float))
        return 0

    # ---- 证据目录：**面板之前**就建好。任何后续崩溃都不许丢 90 格读数
    #      （本轮首跑正是反面教材：次级指标段崩在落盘之前 ⇒ 被迫重跑 4.5 min）
    head = _git(["rev-parse", "HEAD"])
    wt_sha = _git(["hash-object", "experiments/cw_v1/r2_3_n30.py"])
    blob_sha = _git(["rev-parse", f"{head}:experiments/cw_v1/r2_3_n30.py"], allow_fail=True)
    dirty = _git(["status", "--porcelain"])
    srchash = _src_hash()
    outdir = RESULTS / f"cw_r2_3_n30_{_now_utc()}_{srchash}"
    outdir.mkdir(parents=True, exist_ok=False)
    print(f"[OUT] {outdir.name}")
    (outdir / "ctrl.json").write_text(json.dumps({"ctrl": ctrl, "k_need": kn},
                                                 ensure_ascii=False, indent=1,
                                                 default=float), encoding="utf-8")

    eps_run = EPS_REUSE if args.no_fresh else EPS_ALL
    t_p = time.perf_counter()
    cells: list[dict] = []
    for plant in PLANT_KEYS:
        for model in MODELS:
            for ep in eps_run:
                c = run_cell(plant, model, ep, SEED_PANEL)
                cells.append(c)
                tag = "FRESH" if ep in EPS_FRESH else "REUSE"
                print(f"[CELL:{tag}] {plant}/{model:>14}/ep{ep} "
                      f"cost={c['cost_true_per_step']:.6f} att={c['n_attempted']} "
                      f"del={c['n_delivered']} probe={c['n_probe']} ({c['elapsed_s']}s)")
    nulls: list[dict] = []
    for plant in PLANT_KEYS:
        for ep in eps_run:
            c = run_cell(plant, "belief_readout", ep, SEED_NULL)
            nulls.append(c)
            tag = "FRESH" if ep in EPS_FRESH else "REUSE"
            print(f"[NULL:{tag}] {plant}/ep{ep} cost={c['cost_true_per_step']:.6f} "
                  f"({c['elapsed_s']}s)")
    panel_s = round(time.perf_counter() - t_p, 1)
    print(f"[PANEL] {len(cells)} 格 + {len(nulls)} 零控 = {panel_s}s")
    # 立即落盘（后续段崩了也不丢）
    (outdir / "cells.json").write_text(json.dumps(cells, ensure_ascii=False, indent=1,
                                                 default=float), encoding="utf-8")
    (outdir / "null.json").write_text(json.dumps({"nulls": nulls}, ensure_ascii=False,
                                                 indent=1, default=float), encoding="utf-8")

    # ---- 3 闸（CTRL 之后、判词之前）
    gate = r21_gates(ctrl)                      # B1–B8（含 3 格 B3 组 + run_all 84）
    print(f"[GATE] n_checks={gate['n_checks']} n_fail={gate['n_fail']} (B3 {gate['b3_elapsed_s']}s)")
    i1 = identity_gate(cells, nulls)
    print(f"[I1] ok={i1['ok']} checked={i1['n_checked']}/{i1['expect_n']} diff={i1['n_diff']}")
    i2_bad = [f'{c["plant"]}/{c["model"]}/ep{c["ep"]}' for c in cells
              if not c["identities_pass"] or float(c["cost_identity_gap"]) > 1e-9]
    print(f"[I2] 逐格恒等 n_bad={len(i2_bad)}")

    n_fail_gate = gate["n_fail"] + (0 if i1["ok"] else 1) + (0 if not i2_bad else 1)

    # ---- 4 判词（★ 任何异常都不许吞掉已落盘的 90 格；首跑教训写在预注册 §9）
    ana: dict = {}
    if n_fail_gate == 0:
        rows30 = mech2.build_units(cells, nulls, PLANT_KEYS, EPS_ALL)
        pooled = mech2.verdicts_v2(rows30, PLANT_KEYS, EPS_ALL)
        c9 = subset_reconcile(rows30)
        print(f"[C9] ok={c9['ok']} " + json.dumps(
            {k: v["verdict"] for k, v in c9["got"].items()}, ensure_ascii=False))
        if not c9["ok"]:
            n_fail_gate += 1
    else:
        rows30, pooled, c9 = [], {}, {"ok": False, "why": "闸未过，未装配单位"}
        print("[SKIP] 闸未全通过 ⇒ 按预注册不看判词")

    ana = {"rows30": rows30, "pooled": pooled, "c9": c9,
           "fresh": {"verdict": "BLOCKED-BY-GATE"} if n_fail_gate else None}
    if n_fail_gate == 0:
      try:
        rows_f = [r for r in rows30 if int(r["ep"]) in EPS_FRESH]
        b_f = mech2._block(rows_f, len(PLANT_KEYS) * len(EPS_FRESH))
        fv = fresh_verdict(b_f)
        strata = mech2.stratify(rows30)
        coup = mech2.floor_cost_coupling(rows30)
        sec = secondary_aoi(cells, nulls)
        r_all = []
        for r in rows30:
            b = next(c for c in cells if c["plant"] == r["plant"]
                     and c["model"] == "belief_readout" and int(c["ep"]) == r["ep"])
            w = next(c for c in cells if c["plant"] == r["plant"]
                     and c["model"] == "prior_only" and int(c["ep"]) == r["ep"])
            for i in range(len(b["cost_true_by_user"])):
                den = float(b["cost_true_by_user"][i])
                num = float(w["cost_true_by_user"][i])
                if den > 0:
                    r_all.append({"plant": r["plant"], "ep": r["ep"], "i": i,
                                  "r_i": num / den, "fresh": int(r["ep"]) in EPS_FRESH})
        vals = [x["r_i"] for x in r_all]
        ana.update({
            "fresh_block": b_f, "fresh": fv, "strata": strata, "coupling": coup,
            "secondary": sec,
            "r_i": {"n": len(vals),
                    "median": float(np.median(vals)) if vals else None,
                    "min": float(np.min(vals)) if vals else None,
                    "max": float(np.max(vals)) if vals else None,
                    "n_gt1": int(sum(1 for v in vals if v > 1.0)),
                    "n_units_with_any_gt1": int(sum(
                        1 for r in rows30
                        if any(x["r_i"] > 1.0 for x in r_all
                               if x["plant"] == r["plant"] and x["ep"] == r["ep"]))),
                    "by_cell": r_all}})
        print("[FRESH] " + json.dumps(
            {"verdict": fv["verdict"], "next": fv["next_step"],
             "n_pos": b_f["n_pos"], "n_neg": b_f["n_neg"], "n_beat": b_f["n_beat"],
             "p_beat": b_f["binom_p_beat"], "p_sign": b_f["binom_p_sign"]},
            ensure_ascii=False))
        print("[POOLED] " + json.dumps(
            {k: {"verdict": v["verdict"], "n_pos": v["n_pos"], "n_beat": v["n_beat"],
                 "p_beat": v["binom_p_beat"]} for k, v in pooled.items()}, ensure_ascii=False))
        print("[SECONDARY-AOI] " + json.dumps(
            {"n30": sec["n30"], "fresh10": sec["fresh10"],
             "n_undef": sec["n_units_undefined"]}, ensure_ascii=False))
      except Exception as e:  # noqa: BLE001
        ana["analysis_error"] = repr(e)
        ana["fresh"] = {"verdict": "ANALYSIS-ERROR", "error": repr(e)}
        print(f"[ERR] 判词段异常：{e!r}（90 格已落盘，不重跑；按 R14 报错不报假数字）")

    # ---- 5 落盘（cells/null/ctrl 已在前面写过）
    (outdir / "gates.json").write_text(json.dumps({"r21_b1_b8": gate, "I1": i1,
                                                   "I2_bad": i2_bad},
                                                  ensure_ascii=False, indent=1, default=str),
                                       encoding="utf-8")
    (outdir / "ctrl.json").write_text(json.dumps({"ctrl": ctrl, "c9": ana.get("c9"),
                                                  "k_need": kn},
                                                 ensure_ascii=False, indent=1, default=float),
                                      encoding="utf-8")

    cfg = PLANTS["P1"]
    from .config import mainline_config
    cfg_mat = mainline_config(plant=cfg)
    spec = spec_from_config(cfg_mat, budget_total=cfg_mat.protocol.budget(RATIO),
                            split="dev", episodes=EPS_ALL,
                            seeds=(SEED_PANEL, SEED_NULL), models=MODELS)

    try:
        import psutil  # type: ignore
        rss = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
    except Exception:
        rss = None
    total_s = round(time.perf_counter() - t_start, 1)

    summary = {
        "protocol": "CW-POMDP-v1.1", "stage": "CW-R2.3",
        "kind": "PANEL-EXTENSION-N30-FRESH-PRIMARY",
        "prereg": "docs/research/CW-R2.3-扩N到30-预注册-v0.1.md",
        "hypothesis": "H-R2-0",
        "design": {"eps_reuse": EPS_REUSE, "eps_fresh": EPS_FRESH, "eps_all": EPS_ALL,
                   "plants": PLANT_KEYS, "models": list(MODELS),
                   "seed_panel": SEED_PANEL, "seed_null": SEED_NULL,
                   "n_cells": len(cells), "n_null": len(nulls)},
        "primary_fresh": ana.get("fresh"),
        "fresh_block": ana.get("fresh_block"),
        "pooled_descriptive": {k: v for k, v in (ana.get("pooled") or {}).items()},
        "pooled_caveat": ("名义 alpha=0.05 无效：扩样由 ep100–109 的 14/20 触发，"
                          "条件化后实效 I 类错误率 = P(X≥6|n=10,1/2) = 0.3770"),
        "k_need_table": kn,
        "subset_reconcile_c9": ana.get("c9"),
        "identity_I1": i1,
        "identity_I2_n_bad": i2_bad,
        "strata": ana.get("strata"),
        "coupling": ana.get("coupling"),
        "secondary_mean_task_aoi": ana.get("secondary"),
        "r_i_summary": ana.get("r_i"),
        "objective": OBJECTIVE,
        "spec_fingerprint_P1": spec.fingerprint(),
        "sigma_meas": {p: float(PLANTS[p].sigma_meas) for p in PLANT_KEYS},
        "budget": {"panel_s": panel_s, "total_s": total_s,
                   "per_cell_s": [c["elapsed_s"] for c in cells]},
        "head": head, "src_hash": srchash,
        "n_checks_gates": gate["n_checks"], "n_fail_gates": gate["n_fail"],
        "n_fail_total": n_fail_gate,
        "training": "none（本轮零训练）",
    }

    (outdir / "rows30.json").write_text(json.dumps(
        {"rows": ana.get("rows30") or [], "inputs": {
            "r21": {"path": str((R21_DIR / "cells.json").relative_to(REPO)),
                    "sha256": _sha256(R21_DIR / "cells.json")},
            "r22": {"path": str((R22_DIR / "summary.json").relative_to(REPO)),
                    "sha256": _sha256(R22_DIR / "summary.json")}}},
        ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    (outdir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1,
                                                   default=float), encoding="utf-8")

    meta = {"command": "python -m experiments.cw_v1.r2_3_n30 --tag " + str(args.tag),
            "timestamp_local": datetime.now().astimezone().isoformat(),
            "python": os.sys.version.split()[0], "numpy": np.__version__,
            "head": head, "src_hash": srchash,
            "worktree_sha_r2_3": wt_sha, "blob_sha_r2_3": blob_sha,
            "blob_sha_note": (None if blob_sha else "运行前未入库 ⇒ 下一轮补 _POST_COMMIT"),
            "git_dirty_before_run": dirty.splitlines(),
            "protocol": "CW-POMDP-v1.1", "event_pairing": "slot",
            "n_cells": len(cells), "n_null": len(nulls),
            "panel_elapsed_s": panel_s, "total_elapsed_s": total_s,
            "peak_rss_mb": round(rss, 1) if rss else None,
            "n_fail_total": n_fail_gate, "training": "none"}
    (outdir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                      encoding="utf-8")

    print(f"[DONE] {outdir.name} n_fail={n_fail_gate} total={total_s}s "
          f"peak={meta['peak_rss_mb']}MB")
    print(json.dumps({"fresh": (ana.get("fresh") or {}).get("verdict"),
                      "fresh_next": (ana.get("fresh") or {}).get("next_step"),
                      "pooled": {k: v["verdict"] for k, v in (ana.get("pooled") or {}).items()},
                      "c9_ok": (ana.get("c9") or {}).get("ok"),
                      "i1_ok": i1["ok"], "n_fail": n_fail_gate},
                     ensure_ascii=False, indent=1))
    return 0 if n_fail_gate == 0 else 2


if __name__ == "__main__":
    import sys
    sys.exit(main())
