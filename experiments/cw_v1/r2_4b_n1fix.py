# -*- coding: utf-8 -*-
"""`CW-R2.4b`：修正 `N1`（条件化零分布）的**恒真判据**缺陷，并**零 episode** 重算。

## 为什么有这个文件

`CW-R2.4` 的预注册 §4.1 把 `N1` 定义为
`p0 = #{floor_u > 0} / 30`，而 `floor_u = |A(strong,2001) − A(strong,3102)|` 是**绝对值**。
于是 `p0` 恒等于 1.0（30/30，`p=1.86e-9` "显著偏离 1/2"），条件化后 `p=1.0` ——
按预注册 §7-3 的字面规则，这会把 `AOI-SEPARATES` 直接翻成"符号腿不过"。

这是个**恒真判据**：它测的是"绝对值非零"，与符号毫无关系。它不是数据的问题，是判据的问题。
本文件把它改成**有符号**的 `s_u = A(strong,2001) − A(strong,3102)`，并：
1. 用 `K1–K3` 证明"绝对值版"与"有符号版"在**同一份数据**上给出不同答案（修补有分辨力）；
2. 用 `K4` 从已落盘的 90 格重算真正的 `N1`（**零 episode**）；
3. 用 `K5` 证明本次重算与 `CW-R2.4` 的**主判逐字相同**（同一份数据，只换诊断）。

★ 旧证据目录**不改**（`cells/null/summary` 全部保留）；本文件只新增一份更正后的诊断块。
★ 预注册 §10 会记：这是**看数之后**的判据修正，因此 `N1` 的新旧两版**都报**，
  并明确"旧版作废、新版只在 `deviates=False` 时允许沿用朴素 1/2 零分布"。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import mech3
from .r1_14_exactbase import _git, _src_hash

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results" / "cw_v1"
R24_DIR = RESULTS / "cw_r2_4_aoi_20261025T023636_a83942167173392b"
EPS_ALL = list(range(100, 115))
PLANT_KEYS = ["P1", "P2"]


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


def _load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def _syn(s_values: list[float], delta: float = 5.0, floor: float = 1.0) -> list[dict]:
    """合成行：`s = a_strong − a_null` 由调用方钉死；`floor=|s|` 或指定值。"""
    rows = []
    for i, s in enumerate(s_values):
        rows.append({"plant": "P1" if i % 2 == 0 else "P2", "ep": 100 + i // 2,
                     "variant": "SYN", "a_strong": 10.0 + float(s), "a_null": 10.0,
                     "a_weak": 10.0 + delta, "delta": float(delta),
                     "floor_pair": float(abs(s) if floor is None else floor),
                     "sign": (1 if delta > 0 else (0 if delta == 0 else -1)),
                     "beat": (1 if abs(delta) > (abs(s) if floor is None else floor) else 0),
                     "ratio_desc": None, "k_strong": 0, "k_weak": 0, "k_null": 0,
                     "per_user_delta": []})
    return rows


def _abs_version(rows: list[dict]) -> dict:
    """**复刻旧版**（`floor_pair > 0` 当符号率）——只用于证明它恒真。"""
    f = np.array([r["floor_pair"] for r in rows], dtype=float)
    n = len(f)
    k0 = int((f > 0).sum())
    p0 = k0 / float(n)
    p_up = mech3.binom_p_ge(k0, n)
    p_lo = mech3.binom_p_ge(n - k0, n)
    return {"n_abs_pos": k0, "n": n, "p0": p0,
            "p_two_sided_vs_half": min(1.0, 2.0 * min(p_up, p_lo)),
            "deviates_at_alpha": bool(min(1.0, 2.0 * min(p_up, p_lo)) < mech3.ALPHA)}


def gates() -> dict:
    out: dict = {}
    # K1 对称有符号零控 ⇒ p0 ≈ 1/2，不偏离
    alt = [(-1.0) ** i for i in range(30)]
    rows = _syn(alt, floor=1.0)
    nd = mech3.null_diagnostics(rows)["N1"]
    out["K1_symmetric_signed"] = {
        "pass": bool(abs(nd["p0_signed_sign_rate"] - 0.5) <= 1e-12
                     and nd["deviates_at_alpha"] is False),
        "p0": nd["p0_signed_sign_rate"], "deviates": nd["deviates_at_alpha"]}
    # K2 全正有符号零控 ⇒ p0 = 1，偏离
    out["K2_all_positive_signed"] = {
        "pass": bool(mech3.null_diagnostics(_syn([1.0] * 30))["N1"]["deviates_at_alpha"] is True),
        "p0": mech3.null_diagnostics(_syn([1.0] * 30))["N1"]["p0_signed_sign_rate"]}
    # K3 ★ 分辨力：同一份数据，绝对值版 1.0 vs 有符号版 0.5
    ab = _abs_version(rows)
    out["K3_abs_vs_signed_discriminates"] = {
        "pass": bool(ab["p0"] == 1.0 and nd["p0_signed_sign_rate"] == 0.5),
        "abs": ab, "signed": {"p0": nd["p0_signed_sign_rate"]},
        "why": ("同一份对称数据：绝对值版必得 30/30（恒真），有符号版得 15/30。"
                "两版若相同 ⇒ 修补无效")}
    # K4 真实数据上的有符号 N1
    cells = _load(R24_DIR / "cells.json")
    nulls = _load(R24_DIR / "null.json")["nulls"]
    rt = mech3.build_units(cells, nulls, PLANT_KEYS, EPS_ALL, mech3.VAR_TIME)
    real = mech3.null_diagnostics(rt)
    out["K4_real_signed_null"] = {"pass": bool(np.isfinite(real["N1"]["p0_signed_sign_rate"])),
                                  "N1": real["N1"], "N1_abs_vacuous": real["N1_abs_vacuous"]}
    # K5 主判逐字复现
    vt = mech3.verdicts3(rt, PLANT_KEYS, EPS_ALL, variant=mech3.VAR_TIME)
    old = _load(R24_DIR / "summary.json")["primary_A_t"]
    items = {}
    for blk in ("ALL", "P1", "P2"):
        a, b = vt[blk], old[blk]
        items[blk] = {"verdict_same": a["verdict"] == b["verdict"],
                      "n_pos_same": int(a["n_pos"]) == int(b["n_pos"]),
                      "n_neg_same": int(a["n_neg"]) == int(b["n_neg"]),
                      "n_beat_same": int(a["n_beat"]) == int(b["n_beat"]),
                      "p_sign_same": abs(float(a["binom_p_sign"])
                                         - float(b["binom_p_sign"])) <= 1e-15,
                      "p_beat_same": abs(float(a["binom_p_beat"])
                                         - float(b["binom_p_beat"])) <= 1e-15}
    out["K5_reproduce_primary"] = {"pass": all(all(v.values()) for v in items.values()),
                                   "items": items}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="CW-R2.4b N1 修正（零 episode）")
    ap.add_argument("--tag", default="n1fix")
    args = ap.parse_args(argv)
    t0 = time.perf_counter()
    g = gates()
    fails = [k for k, v in g.items() if not v.get("pass")]
    print(f"[GATES] n={len(g)} fails={fails}")
    print("[K3] abs_p0=%.3f signed_p0=%.3f"
          % (g["K3_abs_vs_signed_discriminates"]["abs"]["p0"],
             g["K3_abs_vs_signed_discriminates"]["signed"]["p0"]))
    print("[K4] " + json.dumps(g["K4_real_signed_null"]["N1"], ensure_ascii=False))
    if fails:
        print("[ABORT] 判据自证未过 ⇒ 不写更正")
        return 3

    cells = _load(R24_DIR / "cells.json")
    nulls = _load(R24_DIR / "null.json")["nulls"]
    rt = mech3.build_units(cells, nulls, PLANT_KEYS, EPS_ALL, mech3.VAR_TIME)
    rT = mech3.build_units(cells, nulls, PLANT_KEYS, EPS_ALL, mech3.VAR_FULL)
    nd_t = mech3.null_diagnostics(rt)
    nd_T = mech3.null_diagnostics(rT)

    head = _git(["rev-parse", "HEAD"])
    srchash = _src_hash()
    outdir = RESULTS / f"cw_r2_4b_n1fix_{_now_utc()}_{srchash}"
    outdir.mkdir(parents=True, exist_ok=False)
    summary = {
        "protocol": "CW-POMDP-v1.1", "stage": "CW-R2.4b",
        "kind": "N1-CRITERIA-FIX-ZERO-EPISODE",
        "corrects": str(R24_DIR.name),
        "corrects_scope": ("只更正 `null_diagnostics.N1`；`cells/null/primary_A_t/co_report_A_T/"
                           "I1/I2` 全部逐字沿用（`K5` 证明）"),
        "prereg_r24": "docs/research/CW-R2.4-AoI主指标-预注册-v0.1.md",
        "bug": ("预注册 §4.1 用 `floor_u>0`（绝对值）当符号率 ⇒ 恒真（30/30, p=1.86e-9），"
                "条件化后 p=1.0，按 §7-3 字面规则会误杀 `AOI-SEPARATES`"),
        "fix": "改用有符号 `s_u = A(strong,2001) − A(strong,3102)`",
        "N1_A_t": nd_t["N1"], "N1_A_T": nd_T["N1"],
        "N1_abs_vacuous": nd_t["N1_abs_vacuous"], "N2": nd_t["N2"],
        "gates_K1_K5": g,
        "n_run_episode": 0,
        "head": head, "src_hash": srchash,
        "elapsed_s": round(time.perf_counter() - t0, 2),
    }
    (outdir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1,
                                                    default=float), encoding="utf-8")
    print(f"[DONE] {outdir.name} elapsed={summary['elapsed_s']}s")
    print("[N1-A_t] " + json.dumps(nd_t["N1"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
