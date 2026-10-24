# -*- coding: utf-8 -*-
"""`CW-R2.2` 落盘入口：**判据先修**（零新 episode，只重算 R2.1 的 60 份落盘）。

预注册：`docs/research/CW-R2.2-判据形态缩放-预注册-v0.1.md`

三段
----
`CTRL` 判据自证 `C1–C9`（全零 episode；`C9` 是**真实性闸**：复算 R2.1 的三个旧判词，逐字相同）。
`GATE` `C0` 协议回归 `gates.run_all()`（84 项）+ `C1–C9` 全过才出判词。
`PANEL` **不跑**：直接读 R2.1 的 `cells.json`（40 格）+ `null.json`（20 格），`run_episode` = **0**。

★ 本轮**不训练、不新增 episode、不改 `sigma_meas`**；被测物
  `plan_ref` / `plan_dp` / `world` / `runner` / `config` / `policies` **一字节不动**，
  R2.1 的 `summary.json` 也**不改**（旧判词是历史证据）。
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
from .r1_14_exactbase import _git, _src_hash

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results" / "cw_v1"

#: ★ 写死的输入（预注册 §4 `C9`）：R2.1 的主面板证据目录
R21_DIR = RESULTS / "cw_r2_1_sep_20261024T194859_3c71623d6afe07e4"
PLANTS = ["P1", "P2"]
EPS = list(range(100, 110))


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ==================================================================== 输入
def load_inputs() -> dict:
    cells_p = R21_DIR / "cells.json"
    null_p = R21_DIR / "null.json"
    sum_p = R21_DIR / "summary.json"
    for p in (cells_p, null_p, sum_p):
        if not p.exists():
            raise FileNotFoundError(f"缺少 R2.1 证据：{p}")
    cells = json.loads(cells_p.read_text(encoding="utf-8"))
    nul = json.loads(null_p.read_text(encoding="utf-8"))
    summ = json.loads(sum_p.read_text(encoding="utf-8"))
    return {"cells": cells, "nulls": nul["nulls"], "floor_by_pair": nul["floor_by_pair"],
            "r21_summary": summ,
            "src": {"cells": {"path": str(cells_p.relative_to(REPO)),
                              "sha256": _sha256(cells_p), "n": len(cells)},
                    "null": {"path": str(null_p.relative_to(REPO)),
                             "sha256": _sha256(null_p), "n": len(nul["nulls"])},
                    "summary": {"path": str(sum_p.relative_to(REPO)),
                                "sha256": _sha256(sum_p)}}}


# ==================================================================== 合成构造（判据自证）
def synth(n: int, n_pos: int, n_beat: int, mag: float = 0.10) -> list[dict]:
    """造 `n` 个单位：前 `n_pos` 个 `Δ=+mag`，其余 `Δ=−mag`；前 `n_beat` 个 `|Δ|>floor`。"""
    rows = []
    for i in range(n):
        d = mag if i < n_pos else -mag
        f = 0.5 * mag if i < n_beat else 3.0 * mag
        rows.append({"plant": "SYN", "ep": 900 + i, "cost_strong": 1.0,
                     "cost_weak": 1.0 + d, "delta": d, "floor_pair": f,
                     "sign": (1 if d > 0 else (-1 if d < 0 else 0)),
                     "beat": (1 if abs(d) > f else 0),
                     "ratio_desc": (abs(d) / f) if f > 0 else None})
    return rows


def controls(inp: dict) -> dict:
    """`C1–C9`：每条都写"什么错误能让它 FAIL"（预注册 §4 表）。"""
    chk: list[dict] = []

    def add(cid: str, ok: bool, detail=None) -> None:
        chk.append({"id": cid, "ok": bool(ok), "detail": detail})

    # ---- C1 全成功 ⇒ SEPARATES
    r1 = synth(10, 10, 10)
    v1 = mech2._block(r1, 10)
    add("C1", v1["verdict"] == mech2.V_SEP,
        {"got": v1["verdict"], "n_pos": v1["n_pos"], "n_beat": v1["n_beat"],
         "why": "全正 + 全压过地板时判不出 SEPARATES ⇒ 判据无阳性"})

    # ---- C2 反向（幅度不变、符号全负）⇒ 显式 REVERSED，绝不得 SEPARATES
    r2 = synth(10, 0, 10)
    v2 = mech2._block(r2, 10)
    add("C2", v2["verdict"] == mech2.V_SEP_REV,
        {"got": v2["verdict"], "why": "★ 两条都要：①不得判 SEPARATES（否则方向语义被抹掉）；"
                                     "②也不得折进 CANNOT/幅度档（那会把'强基线更差'说轻）"})

    # ---- C3 幅度 9/10 但符号 5/10 ⇒ MAGNITUDE-ONLY（旧判据会读成"有分离"）
    r3 = synth(10, 5, 9)
    v3 = mech2._block(r3, 10)
    add("C3", v3["verdict"] == mech2.V_MAG_ONLY,
        {"got": v3["verdict"], "n_pos": v3["n_pos"], "n_beat": v3["n_beat"],
         "why": "这条数据的方向是掷硬币，只有幅度过地板 ⇒ 若判 SEPARATES，就是把'幅度大'当'方向对'"})

    # ---- C4 缺格 ⇒ 拒答
    ok4 = False
    det4 = {}
    cells_hole = [c for c in inp["cells"] if not (c["plant"] == "P1" and c["model"] == "prior_only"
                                                 and int(c["ep"]) == 105)]
    try:
        mech2.build_units(cells_hole, inp["nulls"], PLANTS, EPS)
        det4["build_units"] = "未抛异常（缺格被静默吞掉）"
    except mech2.IncompletePanel as exc:
        det4["build_units"] = f"IncompletePanel: {exc}"
        ok4 = True
    rows_real = mech2.build_units(inp["cells"], inp["nulls"], PLANTS, EPS)
    v4b = mech2._block(rows_real[:-1], len(rows_real))
    det4["verdict_on_short"] = v4b["verdict"]
    ok4 = ok4 and v4b["verdict"] == mech2.V_INCOMPLETE
    add("C4", ok4, {**det4, "why": "9/10 却出判词 ⇒ 分母被删（R1.8 同族）"})

    # ---- C5 NaN ⇒ UNDEFINED 且点名
    r5 = synth(10, 8, 8)
    r5[3]["delta"] = float("nan")
    v5 = mech2._block(r5, 10)
    add("C5", v5["verdict"] == mech2.V_UNDEF and v5["nonfinite_cells"] == ["SYN/ep903"],
        {"got": v5["verdict"], "named": v5["nonfinite_cells"],
         "why": "NaN 被当中位数算 ⇒ 判词无意义却看起来正常"})

    # ---- C6 floor=0 合法且不除零（synth 只让前 3 格过地板，再把第 4 格地板置 0 ⇒ n_beat=4）
    r6 = synth(10, 9, 3)
    r6[3]["floor_pair"] = 0.0
    r6[3]["beat"] = 1 if abs(r6[3]["delta"]) > 0 else 0
    r6[3]["ratio_desc"] = None
    v6 = mech2._block(r6, 10)
    add("C6", v6["verdict"] != mech2.V_UNDEF and v6["n_zero_floor"] == 1
        and v6["n_beat"] == 4 and np.isfinite(v6["delta_median"]),
        {"got": v6["verdict"], "n_zero_floor": v6["n_zero_floor"], "n_beat": v6["n_beat"],
         "ratio_desc": v6.get("ratio_desc"),
         "why": "地板恰为 0 的格在整数送达下不罕见；除零会造 inf/NaN 或悄悄跳过"})

    # ---- C7 阈值随 n 缩放：同比例 0.80
    p8 = mech2.binom_p_ge(8, 10)
    p12 = mech2.binom_p_ge(12, 15)
    v7a = mech2._block(synth(10, 8, 8), 10)
    v7b = mech2._block(synth(15, 12, 12), 15)
    add("C7", (p8 >= mech2.ALPHA) and (p12 < mech2.ALPHA)
        and v7a["verdict"] != mech2.V_SEP and v7b["verdict"] == mech2.V_SEP,
        {"p_8of10": p8, "p_12of15": p12, "v_8of10": v7a["verdict"], "v_12of15": v7b["verdict"],
         "why": "同一比例 0.80 在 n=10 不过、n=15 过 ⇒ 证明阈值是按 n 缩放的；"
                "若两者同判，说明还残留计数阈值（旧 `18/15` 在 n=10 上算术不可达）"})

    # ---- C8 旧判据活体对照：同一构造两套判词必须不同
    r8 = synth(20, 17, 17)
    old8 = mech2.legacy_block(r8, 20)
    new8 = mech2._block(r8, 20)
    add("C8", old8["verdict"] == "SEPARATES-MAGNITUDE-ONLY"
        and new8["verdict"] == mech2.V_SEP,
        {"legacy": old8["verdict"], "v2": new8["verdict"],
         "why": "两套判词在同一构造上给出同一结果 ⇒ '新判据'只是旧判据换名"})

    # ---- C9 真实性闸：复算 R2.1 三个旧判词（逐字 + 1e-12）
    summ = inp["r21_summary"]
    det9 = {}
    ok9 = True
    for key in ("ALL", "P1", "P2"):
        sel = rows_real if key == "ALL" else [r for r in rows_real if r["plant"] == key]
        n_units = len(sel)
        got = mech2.legacy_block(sel, n_units)
        exp = summ["blocks"][key]
        d = {
            "verdict": [got["verdict"], exp["verdict"]],
            "median_abs_delta_absdiff": abs(got["median_abs_delta"] - exp["median_abs_delta"]),
            "floor_absdiff": abs(got["floor_R2"] - exp["floor_R2"]),
            "n_pos": [got["n_pos"], exp["n_pos"]],
            "n_units": [n_units, exp["n_units"]],
        }
        good = (got["verdict"] == exp["verdict"]
                and d["median_abs_delta_absdiff"] <= 1e-12
                and d["floor_absdiff"] <= 1e-12
                and got["n_pos"] == exp["n_pos"] and n_units == exp["n_units"])
        det9[key] = {**d, "ok": good}
        ok9 = ok9 and good
    add("C9", ok9, {**det9,
                    "why": "★ 唯一的真实性闸：我算的 Δ/floor 必须就是 R2.1 那 40+20 格，"
                           "不是另一批数（否则新旧对照无意义）"})

    return {"checks": chk, "n_checks": len(chk),
            "n_fail": sum(1 for c in chk if not c["ok"]),
            "all_pass": all(c["ok"] for c in chk)}


# ==================================================================== 分析
def analyse(rows: list[dict], inp: dict) -> dict:
    v2 = mech2.verdicts_v2(rows, PLANTS, EPS)
    legacy = {"ALL": mech2.legacy_block(rows, len(rows))}
    for p in PLANTS:
        legacy[p] = mech2.legacy_block([r for r in rows if r["plant"] == p], len(EPS))
    strat = mech2.stratify(rows)

    # 逐用户 r_i（R17：必报）
    r_i = {}
    for p in PLANTS:
        for ep in EPS:
            b = next(c for c in inp["cells"] if c["plant"] == p
                     and c["model"] == "belief_readout" and int(c["ep"]) == ep)
            w = next(c for c in inp["cells"] if c["plant"] == p
                     and c["model"] == "prior_only" and int(c["ep"]) == ep)
            r_i[f"{p}/ep{ep}"] = [
                (float(w["cost_true_by_user"][i]) / float(b["cost_true_by_user"][i])
                 if float(b["cost_true_by_user"][i]) > 0 else None)
                for i in range(len(b["cost_true_by_user"]))]
    all_r = [v for vs in r_i.values() for v in vs if v is not None]
    r_summary = {"n": len(all_r),
                 "median": float(np.median(all_r)) if all_r else None,
                 "min": float(np.min(all_r)) if all_r else None,
                 "max": float(np.max(all_r)) if all_r else None,
                 "n_gt1": int(sum(1 for v in all_r if v > 1.0))}

    # 逐格新旧对照
    per_cell = []
    for r in rows:
        key = f'{r["plant"]}/ep{r["ep"]}'
        per_cell.append({"cell": key, "delta": r["delta"], "floor_pair": r["floor_pair"],
                         "sign": r["sign"], "beat": int(r["beat"]),
                         "ratio_desc": r["ratio_desc"], "r_i": r_i.get(key),
                         "stratum": ("LOW" if key in strat["low_keys"] else "HIGH")})

    changed = {k: (legacy[k]["verdict"] != v2[k]["verdict"]) for k in ("ALL", "P1", "P2")}
    return {"v2": v2, "legacy": legacy, "strata": strat,
            "per_cell": per_cell, "r_i_by_cell": r_i, "r_i_summary": r_summary,
            "coupling": mech2.floor_cost_coupling(rows),
            "verdict_changed": changed,
            "n_changed": int(sum(changed.values()))}


# ==================================================================== 主流程
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="CW-R2.2 判据先修（零新 episode）")
    ap.add_argument("--ctrl-only", action="store_true", help="只跑判据自证 + 读输入（零 episode）")
    ap.add_argument("--tag", default="main")
    args = ap.parse_args(argv)

    t_start = time.perf_counter()
    inp = load_inputs()
    print(f"[INPUT] cells={inp['src']['cells']['n']} null={inp['src']['null']['n']} "
          f"cells_sha={inp['src']['cells']['sha256'][:16]} null_sha={inp['src']['null']['sha256'][:16]}")

    ctrl = controls(inp)
    print(f"[CTRL] all_pass={ctrl['all_pass']} " +
          " ".join(f"{c['id']}={c['ok']}" for c in ctrl["checks"]))
    if not ctrl["all_pass"]:
        print("[ABORT] 判据自证未过 ⇒ 按预注册 §4 不看面板、不出判词")
        for c in ctrl["checks"]:
            if not c["ok"]:
                print(f"  FAIL {c['id']}: {json.dumps(c['detail'], ensure_ascii=False)[:400]}")
        return 3
    if args.ctrl_only:
        print(json.dumps(ctrl, ensure_ascii=False, indent=1, default=float))
        return 0

    # ---- C0 协议回归
    t1 = time.perf_counter()
    reg = G.run_all()
    reg_s = round(time.perf_counter() - t1, 2)
    print(f"[C0] gates.run_all n_checks={reg['n_checks']} n_fail={reg['n_fail']} ({reg_s}s)")

    rows = mech2.build_units(inp["cells"], inp["nulls"], PLANTS, EPS)
    ana = analyse(rows, inp)

    print("[V2] " + " | ".join(f"{k}={ana['v2'][k]['verdict']}"
                              f"(pos={ana['v2'][k]['n_pos']}/{ana['v2'][k]['n_units']},"
                              f"beat={ana['v2'][k]['n_beat']},p_s={ana['v2'][k]['binom_p_sign']:.4g},"
                              f"p_b={ana['v2'][k]['binom_p_beat']:.4g})"
                              for k in ("ALL", "P1", "P2")))
    print("[OLD] " + " | ".join(f"{k}={ana['legacy'][k]['verdict']}" for k in ("ALL", "P1", "P2")))
    print(f"[STRAT] cut={ana['strata']['cut_median_floor']:.6f} "
          f"LOW={ana['strata']['LOW']['verdict']} HIGH={ana['strata']['HIGH']['verdict']} "
          f"(n_low={ana['strata']['n_low']} n_high={ana['strata']['n_high']})")

    # ---- 落盘
    head = _git(["rev-parse", "HEAD"])
    wt = _git(["hash-object", "experiments/cw_v1/r2_2_criteria.py"])
    blob = _git(["rev-parse", f"{head}:experiments/cw_v1/r2_2_criteria.py"], allow_fail=True)
    dirty = _git(["status", "--porcelain"])
    srchash = _src_hash()
    outdir = RESULTS / f"cw_r2_2_criteria_{_now_utc()}_{srchash}"
    outdir.mkdir(parents=True, exist_ok=False)
    print(f"[OUT] {outdir.name}")

    summary = {
        "protocol": "CW-POMDP-v1.1", "stage": "CW-R2.2",
        "kind": "CRITERIA-FIX-ZERO-NEW-EPISODE",
        "prereg": "docs/research/CW-R2.2-判据形态缩放-预注册-v0.1.md",
        "inputs": inp["src"], "plants": PLANTS, "eps": EPS,
        "n_units": len(rows), "n_new_episodes": 0,
        "verdicts_v2": {k: v["verdict"] for k, v in ana["v2"].items()},
        "verdicts_legacy": {k: v["verdict"] for k, v in ana["legacy"].items()},
        "verdict_changed": ana["verdict_changed"], "n_changed": ana["n_changed"],
        "blocks_v2": ana["v2"], "blocks_legacy": ana["legacy"],
        "strata": {k: ana["strata"][k] for k in
                   ("cut_median_floor", "LOW", "HIGH", "n_low", "n_high",
                    "low_keys", "high_keys")},
        "coupling": ana["coupling"], "r_i_summary": ana["r_i_summary"],
        "r_i_by_cell": ana["r_i_by_cell"],
        "alpha": mech2.ALPHA,
        "head": head, "src_hash": srchash,
        "n_checks_ctrl": ctrl["n_checks"], "n_fail_ctrl": ctrl["n_fail"],
        "n_checks_regression": reg["n_checks"], "n_fail_regression": reg["n_fail"],
        "training": "none（本轮零训练、零新 episode）",
        "codes": {"V_SEP": mech2.V_SEP, "V_SEP_REV": mech2.V_SEP_REV,
                  "V_SIGN_ONLY": mech2.V_SIGN_ONLY, "V_SIGN_ONLY_REV": mech2.V_SIGN_ONLY_REV,
                  "V_MAG_ONLY": mech2.V_MAG_ONLY, "V_CANNOT": mech2.V_CANNOT,
                  "V_UNDEF": mech2.V_UNDEF, "V_INCOMPLETE": mech2.V_INCOMPLETE,
                  "V_NO_CLAIM": mech2.V_NO_CLAIM},
    }
    (outdir / "cells_v2.json").write_text(
        json.dumps(ana["per_cell"], ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    (outdir / "ctrl.json").write_text(
        json.dumps(ctrl, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    (outdir / "gates.json").write_text(
        json.dumps({"C0_regression": {"n_checks": reg["n_checks"], "n_fail": reg["n_fail"],
                                      "elapsed_s": reg_s},
                    "C1_C9": ctrl}, ensure_ascii=False, indent=1, default=str),
        encoding="utf-8")

    try:
        import psutil  # type: ignore
        rss = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
    except Exception:
        rss = None
    total_s = round(time.perf_counter() - t_start, 1)
    summary["budget"] = {"total_s": total_s,
                         "peak_rss_mb": round(rss, 1) if rss else None}
    (outdir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1, default=float), encoding="utf-8")

    meta = {"command": "python -m experiments.cw_v1.r2_2_criteria --tag " + str(args.tag),
            "timestamp_local": datetime.now().astimezone().isoformat(),
            "python": os.sys.version.split()[0], "numpy": np.__version__,
            "head": head, "src_hash": srchash,
            "worktree_sha_r2_2": wt, "blob_sha_r2_2": blob,
            "blob_sha_note": (None if blob is None else "运行前已在 HEAD 上"),
            "git_dirty_before_run": dirty.splitlines(),
            "protocol": "CW-POMDP-v1.1", "event_pairing": "slot",
            "n_new_episodes": 0, "total_elapsed_s": total_s,
            "peak_rss_mb": round(rss, 1) if rss else None,
            "n_checks_ctrl": ctrl["n_checks"], "n_fail_ctrl": ctrl["n_fail"],
            "n_checks_regression": reg["n_checks"], "n_fail_regression": reg["n_fail"],
            "training": "none"}
    (outdir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                      encoding="utf-8")

    n_fail = ctrl["n_fail"] + int(reg["n_fail"])
    print(f"[DONE] {outdir.name} n_fail={n_fail} total={total_s}s peak={meta['peak_rss_mb']}MB")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
