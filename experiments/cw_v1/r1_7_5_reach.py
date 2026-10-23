# -*- coding: utf-8 -*-
"""CW-R1.7.5 `G4` 的可达性下界与「嵌套采样」单变量对照。

回答 STATE §7.2 留下的唯一问题：**`G4` 的 10% 线在 `M=16` 上到底能不能过。**
在改实现之前先回答"能不能过"，否则又会花一轮追一个不存在的目标（R1.7.3 的
`W4` 就是这样被否掉的）。

预注册：`docs/research/CW-R1.7.5-G4可达性-预注册-v0.1.md`（15:17:42 写，早于本文件任何数字）

四条闸 / 三个面板
-----------------
- `P0a` 结构：`_Draws(M=16)` 与 `_Draws(M=32)[:16]` **不相等** ⇒ CRN 嵌套在 `0426fcb` 不可得。
- `P0b` 活体阴性对照：`_Draws(M=16, pool_M=32)` 与 `_Draws(M=32, pool_M=32)[:16]`
  **必须逐位相同**（证明"嵌套"这个概念在本环境**可判定**，不是我的检查没分辨力）。
- `P0c` 口径恒等：默认路径下 `plan_gates.g4_m_stability` 的四个标量必须与
  R1.7.4 证据里的 `bench.json` **逐位相同**（改被测物不改默认路径）。
- `P0d` 不变式：`agree==1.0` 的 episode，`c16` 与 `c32` 必须逐位相同
  （`cost_true` 是真实过程成本，两条轨迹动作相同时不可能不同 —— 预注册 §1.2）。

- `P1`（`indep`）：`8 seed × 4 ep × {16,32}` 的 `rel` 分布（现状口径 = 两次抽样不配对）。
- `P3`（`nested`）：同 `seed/ep`，`M=16` 改用 `pool_M=32` ⇒ 与 `M=32` 共享前 16 条实现，
  差**只**来自新增的 16 条路径。**单变量对照：只改采样结构，不改策略。**
- `P2`（`M∈{16,32,64}`）：跨 8 个 seed 的 `cost_true` 样本标准差对 `M` 的 log-log 斜率。
  ★ 预注册 §1.3：**不许**算"内层 M 个样本均值的标准差"（那按构造就是 `c/√M`，恒等式）。

本轮不读 `N1/N2`、不跑面板、不改任何既有阈值。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import tracemalloc
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from . import config as C
from . import plan_gates as G
from . import plan_ref as P
from .plan_ref import _Draws, make
from .runner import run_episode

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results" / "cw_v1"

# ---- 参数（预注册 §4 写死，跑前不改）
EPS = (100, 101, 102, 103)
SEEDS = tuple(range(2001, 2009))       # 8 个
M_LIST = (16, 32)                       # P1 / P3
M_VAR = (16, 32, 64)                    # P2
RHO = float(G.RHO_DEFAULT)              # 0.98，取自判据实现
RATIO = 0.5
H_PLAN = 5
ARM = "probe"
LINE_REL = 0.10                         # G4 的原线，不许改
LINE_AGREE = 0.90                       # G4 的原线，不许改

#: R1.7.4 主证据（`bench.json` 的 `G4` 标量，用来做 `P0c` 口径恒等闸）
R174_BENCH = (RESULTS / "cw_r1_7_4_tails_20261023T130519_f11ffdefc06c0201_232550606b401065"
              / "bench.json")


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


# ---------------------------------------------------------------- P0
def p0_structural() -> dict:
    """三条结构性闸 + 一条不变式的前置（不变式本体在 P1 里查）。"""
    out = {}
    seed, H, n = 777, H_PLAN, 2

    # ---- P0a：现状两档不是嵌套
    d16 = _Draws(np.random.default_rng(seed), H, 16, n)
    d32 = _Draws(np.random.default_rng(seed), H, 32, n)
    per = {}
    for name in ("udeliv", "eps_meas", "eps_csi", "s_h", "w"):
        a, b = getattr(d16, name), getattr(d32, name)[:, :16, :]
        per[name] = {"equal": bool(np.array_equal(a, b)),
                     "maxdiff": float(np.abs(a - b).max())}
    for name in ("init_x", "init_h"):
        a, b = getattr(d16, name), getattr(d32, name)[:16, :]
        per[name] = {"equal": bool(np.array_equal(a, b)),
                     "maxdiff": float(np.abs(a - b).max())}
    none_equal = all(not v["equal"] for v in per.values())
    out["P0a_not_nested"] = {
        "pass": none_equal, "per_array": per,
        "rule": "7 个数组必须全部不等 ⇒ `0426fcb` 的 M=16/M=32 是两批独立样本，不是嵌套",
    }

    # ---- P0b（活体阴性对照）：开了 pool_M 之后必须嵌套
    e16 = _Draws(np.random.default_rng(seed), H, 16, n, pool_M=32)
    e32 = _Draws(np.random.default_rng(seed), H, 32, n, pool_M=32)
    per2 = {}
    for name in ("udeliv", "eps_meas", "eps_csi", "s_h", "w"):
        a, b = getattr(e16, name), getattr(e32, name)[:, :16, :]
        per2[name] = {"equal": bool(np.array_equal(a, b)),
                      "maxdiff": float(np.abs(a - b).max())}
    for name in ("init_x", "init_h"):
        a, b = getattr(e16, name), getattr(e32, name)[:16, :]
        per2[name] = {"equal": bool(np.array_equal(a, b)),
                      "maxdiff": float(np.abs(a - b).max())}
    out["P0b_nested_available"] = {
        "pass": all(v["equal"] for v in per2.values()), "per_array": per2,
        "rule": "`pool_M=32` 时 M=16 必须逐位等于 M=32 的前 16 条（否则『嵌套』不可判定 ⇒ NO-POWER）",
    }

    # ---- P0b2：纯解析复算旧公式（便宜的双保险，不依赖任何历史文件）
    rng_a = np.random.default_rng(seed)
    old = {
        "udeliv": rng_a.random((H, 16, n)),
        "eps_meas": rng_a.standard_normal((H, 16, n)),
        "eps_csi": rng_a.standard_normal((H, 16, n)),
        "s_h": rng_a.standard_normal((H, 16, n)),
        "w": rng_a.standard_normal((H, 16, n)),
        "init_x": rng_a.standard_normal((16, n)),
        "init_h": rng_a.standard_normal((16, n)),
    }
    new = _Draws(np.random.default_rng(seed), H, 16, n)      # pool_M=None
    per3 = {k: {"equal": bool(np.array_equal(old[k], getattr(new, k))),
                "maxdiff": float(np.abs(old[k] - getattr(new, k)).max())} for k in old}
    out["P0b2_formula_identical"] = {
        "pass": all(v["equal"] for v in per3.values()), "per_array": per3,
        "rule": "`pool_M=None` 必须与旧公式（0426fcb 的 7 条 rng 调用）逐位相同",
    }
    return out


# ---------------------------------------------------------------- 跑一格
def _run(cfg, budget, ep: int, seed: int, M: int, pool: int | None) -> dict:
    pol = G._Rec(make(cfg, arm=ARM, h_plan=H_PLAN, m_samples=M, seed=seed,
                      rho=RHO, pacing=True, draws_pool_m=pool))
    r = run_episode(cfg, pol, split="dev", episode=ep, budget_total=budget, rho=RHO)
    return {"cost_true": float(r.cost_true_per_step),
            "n_probe": int(r.n_probe), "n_attempted": int(r.n_attempted),
            "n_delivered": int(r.n_delivered),
            "n_idle": int(r.n_idle),
            "policy_trace_hash": str(r.policy_trace_hash),
            "acts": [(str(k), (None if u is None else int(u))) for (_t, k, u) in pol.acts],
            "n_acts": len(pol.acts)}


def _agree(a: dict, b: dict) -> tuple[float, int]:
    aa, bb = a["acts"], b["acts"]
    if len(aa) != len(bb):
        raise ValueError(f"动作序列长度不等 {len(aa)} vs {len(bb)}")
    n = len(aa)
    same = sum(1 for (k1, u1), (k2, u2) in zip(aa, bb) if (k1, u1) == (k2, u2))
    return same / max(n, 1), n


def _rel(c_lo: float, c_hi: float) -> float:
    return abs(c_hi - c_lo) / max(c_lo, 1e-12)


# ---------------------------------------------------------------- 主流程
def build(seeds, eps, quick: bool) -> dict:
    cfg = C.CWConfig(plant=C.bounded_plant())
    budget = cfg.protocol.budget(RATIO)
    tag = "QUICK" if quick else "FULL"
    print(f"[R1.7.5] {tag} seeds={len(seeds)} eps={len(eps)}", flush=True)

    cells: dict = {}
    t0 = time.perf_counter()
    total = len(seeds) * len(eps) * (len(M_LIST) + 1 + (0 if quick else 1))
    i = 0

    for ep in eps:
        for seed in seeds:
            # ---- indep：M=16 / M=32（现状口径）
            for M in M_LIST:
                i += 1
                t = time.perf_counter()
                cells[(f"indep|{ep}|{seed}|{M}")] = _run(cfg, budget, ep, seed, M, None)
                print(f"  [{i:3d}/{total}] indep ep={ep} seed={seed} M={M:2d} "
                      f"{time.perf_counter() - t:5.1f}s", flush=True)
            # ---- nested：M=16 但池子 = 32（M=32 与 indep 的同档**逐位相同**，直接复用）
            i += 1
            t = time.perf_counter()
            cells[(f"nested|{ep}|{seed}|16")] = _run(cfg, budget, ep, seed, 16, 32)
            print(f"  [{i:3d}/{total}] nested ep={ep} seed={seed} M=16(pool=32) "
                  f"{time.perf_counter() - t:5.1f}s", flush=True)
            # ---- P2：M=64（只补这一档，16/32 复用上面）
            if not quick:
                i += 1
                t = time.perf_counter()
                cells[(f"indep|{ep}|{seed}|64")] = _run(cfg, budget, ep, seed, 64, None)
                print(f"  [{i:3d}/{total}] indep ep={ep} seed={seed} M=64 "
                      f"{time.perf_counter() - t:5.1f}s", flush=True)

    print(f"[R1.7.5] 采样完成 {len(cells)} 格 / {time.perf_counter() - t0:.1f}s", flush=True)
    return {"cells": cells, "elapsed_s": round(time.perf_counter() - t0, 2),
            "n_cells": len(cells)}


def analyse(raw: dict, seeds, eps, quick: bool) -> dict:
    cells = raw["cells"]
    out: dict = {"n_cells": len(cells), "elapsed_s": raw["elapsed_s"]}
    invariants: dict = {}

    # ---- P1 / P3 逐对
    def _pairs(mode: str, m_lo: int, m_hi: int) -> dict:
        rows = []
        for ep in eps:
            for seed in seeds:
                key_lo = f"{mode}|{ep}|{seed}|{m_lo}"
                key_hi = f"indep|{ep}|{seed}|{m_hi}"     # M=32 一律取 indep 档（与 nested 同池）
                if key_lo not in cells or key_hi not in cells:
                    continue
                a, b = cells[key_lo], cells[key_hi]
                ag, n = _agree(a, b)
                rows.append({"episode": ep, "seed": seed, "agree": ag, "n_acts": n,
                             "c_lo": a["cost_true"], "c_hi": b["cost_true"],
                             "rel": _rel(a["cost_true"], b["cost_true"]),
                             "n_probe_lo": a["n_probe"], "n_probe_hi": b["n_probe"]})
        rel = np.array([r["rel"] for r in rows], dtype=float)
        ag = np.array([r["agree"] for r in rows], dtype=float)
        return {"rows": rows, "n": len(rows),
                "rel_median": float(np.median(rel)) if len(rel) else None,
                "rel_mean": float(rel.mean()) if len(rel) else None,
                "rel_p90": float(np.percentile(rel, 90)) if len(rel) else None,
                "rel_p10": float(np.percentile(rel, 10)) if len(rel) else None,
                "rel_frac_gt_line": float((rel > LINE_REL).mean()) if len(rel) else None,
                "agree_median": float(np.median(ag)) if len(ag) else None,
                "agree_min": float(ag.min()) if len(ag) else None,
                "frac_agree_ge_line": float((ag >= LINE_AGREE).mean()) if len(ag) else None}

    out["P1_indep"] = _pairs("indep", 16, 32)
    out["P3_nested"] = _pairs("nested", 16, 32)

    # ---- P0d 不变式：agree==1.0 ⇒ 成本逐位相同（用 P1 的 32 对）
    viol = [r for r in out["P1_indep"]["rows"]
            if r["agree"] >= 1.0 - 1e-12 and abs(r["c_lo"] - r["c_hi"]) > 0]
    exact = [r for r in out["P1_indep"]["rows"] if r["agree"] >= 1.0 - 1e-12]
    invariants["P0d_agree1_cost_equal"] = {
        "pass": len(viol) == 0,
        "n_agree1": len(exact), "n_violation": len(viol),
        "violations": viol[:5],
        "rule": "agree==1.0 的 episode 里 c16 必须逐位等于 c32（cost_true 是真实过程成本）",
    }

    # ---- 独立性交叉检查：indep 的两档 trace 若相同也必须同成本
    same_hash, same_hash_bad = 0, []
    for ep in eps:
        for seed in seeds:
            a = cells.get(f"indep|{ep}|{seed}|16")
            b = cells.get(f"indep|{ep}|{seed}|32")
            if a and b and a["policy_trace_hash"] == b["policy_trace_hash"]:
                same_hash += 1
                if a["cost_true"] != b["cost_true"]:
                    same_hash_bad.append({"episode": ep, "seed": seed,
                                          "c16": a["cost_true"], "c32": b["cost_true"]})
    invariants["P0d2_same_trace_same_cost"] = {
        "pass": len(same_hash_bad) == 0,
        "n_same_trace": same_hash, "violations": same_hash_bad[:5],
        "rule": "policy_trace_hash 相同 ⇒ cost_true 必须相同（否则 cost 里有规划器采样依赖 ⇒ 破坏 §1.2）",
    }

    # ---- P2 方差斜率（跨 seed 的 cost_true 标准差 vs M）
    p2 = {"per_episode": {}, "slopes": []}
    for ep in eps:
        s = {}
        for M in (M_VAR if not quick else M_LIST):
            vals = [cells[f"indep|{ep}|{seed}|{M}"]["cost_true"]
                    for seed in seeds if f"indep|{ep}|{seed}|{M}" in cells]
            if len(vals) >= 3:
                s[M] = {"n": len(vals), "mean": float(np.mean(vals)),
                        "sd": float(np.std(vals, ddof=1)),
                        "min": float(np.min(vals)), "max": float(np.max(vals))}
        if len(s) >= 2:
            xs = np.log(np.array([M for M in s], dtype=float))
            ys = np.log(np.array([s[M]["sd"] for M in s], dtype=float))
            ok = np.isfinite(ys) & (np.array([s[M]["sd"] for M in s]) > 0)
            slope = float(np.polyfit(xs[ok], ys[ok], 1)[0]) if ok.sum() >= 2 else None
        else:
            slope = None
        p2["per_episode"][str(ep)] = {"by_M": {str(k): v for k, v in s.items()},
                                      "slope_logsd_vs_logM": slope}
        if slope is not None:
            p2["slopes"].append(slope)
    p2["slope_median"] = float(np.median(p2["slopes"])) if p2["slopes"] else None
    n_m_points = min(len(p2["per_episode"][k]["by_M"]) for k in p2["per_episode"]) \
        if p2["per_episode"] else 0
    p2["n_M_points"] = n_m_points
    if p2["slope_median"] is None:
        p2["band"] = "NO-DATA"
    elif n_m_points < 3:
        # ★ 两点拟合 log(sd)~log(M) **没有自由度**：斜率完全由两个点决定，
        #   而 sd 在 n=4 个 seed 上的抽样误差本身 ~40%。⇒ 本轮（预算砍到 4 seed、
        #   放弃 M=64）**不允许**把斜率映射成 C-SQRT-M / VARIANCE-FLOOR / VARIANCE-STEP。
        #   判 `TWO-POINT-NO-DF`：只报数，不下判。预注册 §2 的 `P2` 因此**未按原设计执行**。
        p2["band"] = "TWO-POINT-NO-DF"
    elif -0.65 <= p2["slope_median"] <= -0.35:
        p2["band"] = "C-SQRT-M"            # 符合 c/√M
    elif p2["slope_median"] > -0.35:
        p2["band"] = "VARIANCE-FLOOR"      # 有偏或未播种随机源
    else:
        p2["band"] = "VARIANCE-STEP"       # 决策阶梯化（不是缺陷，但须记）
    out["P2_slope"] = p2

    # ---- 预算内一致性：n_probe 随 M 的变化（只报不判）
    out["probe_counts"] = {
        "indep_M16_median": float(np.median([cells[f"indep|{ep}|{seed}|16"]["n_probe"]
                                             for ep in eps for seed in seeds])),
        "indep_M32_median": float(np.median([cells[f"indep|{ep}|{seed}|32"]["n_probe"]
                                             for ep in eps for seed in seeds])),
        "nested_M16_median": float(np.median([cells[f"nested|{ep}|{seed}|16"]["n_probe"]
                                              for ep in eps for seed in seeds])),
    }
    return {"stats": out, "invariants": invariants}


def settle(p0: dict, an: dict) -> dict:
    inv = an["invariants"]
    st = an["stats"]
    checks = {}
    checks["P0a_not_nested"] = p0["P0a_not_nested"]
    checks["P0b_nested_available"] = p0["P0b_nested_available"]
    checks["P0b2_formula_identical"] = p0["P0b2_formula_identical"]
    checks.update(inv)

    # ---- P0c 口径恒等：与 R1.7.4 的 bench.json 逐位比
    try:
        b = json.loads(R174_BENCH.read_text(encoding="utf-8"))
        det = b["G4"]["detail"]
        ref = {"agreement": float(det["agreement"]),
               "median_rel": float(det["median_rel_cost_diff"])}
        checks["P0c_bench_ref_loaded"] = {
            "pass": True, "ref": ref,
            "file": str(R174_BENCH.relative_to(REPO)),
            "note": "参照值已载入；实际比对在 g4_identity 里做",
        }
    except Exception as exc:                                  # noqa: BLE001
        checks["P0c_bench_ref_loaded"] = {
            "pass": False, "error": repr(exc), "file": str(R174_BENCH),
            "rule": "读不到 R1.7.4 的 bench.json ⇒ 不能做口径恒等闸",
        }
    checks["P0c_g4_identity"] = an.get("g4_identity", {"pass": False,
                                                      "error": "未执行"})
    return checks


def g4_identity() -> dict:
    """默认路径下重跑 `G4`，与 R1.7.4 证据的 `bench.json` 逐位比。

    ★ 这是"改被测物没改默认路径"最硬的一条：`g4_m_stability` 是**原函数**，
      参数全部由它自己决定（seed=777 / ep∈{100,101} / M∈{16,32} / rho=RHO_DEFAULT）。
    """
    g = G.Gate()
    G.g4_m_stability(g)
    row = [r for r in g.rows if r["name"].startswith("G4")][0]
    now = {"agreement": float(row["detail"]["agreement"]),
           "median_rel": float(row["detail"]["median_rel_cost_diff"]),
           "per_episode": row["detail"]["per_episode"],
           # ★ 键名必须与"恒等闸的 pass"分开：`g4_gate_pass` 是 **G4 判据本身**过不过
           #   （本轮预期 False，因为它就是要被诊断的那条线），
           #   `pass`（外层）才是"现在与 0426fcb 是否逐位相同"。
           #   第一版两者同名，打印出来读作 PASS 而 detail 里 False，是 #32 同型自伤。
           "g4_gate_pass": bool(row["pass"]), "n_compared": int(row["detail"]["n_compared"])}
    try:
        b = json.loads(R174_BENCH.read_text(encoding="utf-8"))
        det = b["G4"]["detail"]
        ref = {"agreement": float(det["agreement"]),
               "median_rel": float(det["median_rel_cost_diff"])}
        d = {k: abs(now[k] - ref[k]) for k in ref}
        return {"pass": max(d.values()) <= 1e-12, "now": now, "ref": ref,
                "abs_diff": d,
                "rule": "默认路径必须与 0426fcb 的 G4 逐位相同（rel ≤1e-12）"}
    except Exception as exc:                                  # noqa: BLE001
        return {"pass": False, "now": now, "error": repr(exc)}


def verdict_of(checks: dict, an: dict) -> dict:
    st = an["stats"]
    ind = st["P1_indep"]
    ned = st["P3_nested"]
    if not checks["P0c_g4_identity"]["pass"]:
        v, why = "HARNESS-MISMATCH", "默认路径的 G4 与 0426fcb 不一致 ⇒ 改坏了被测物"
    elif not checks["P0b_nested_available"]["pass"]:
        v, why = "NO-POWER", "`pool_M` 开了也不嵌套 ⇒ 『嵌套』不可判定，检查无分辨力"
    elif not checks["P0d_agree1_cost_equal"]["pass"] or not checks["P0d2_same_trace_same_cost"]["pass"]:
        v, why = "CONTRACT-VIOLATION", "动作/trace 相同但成本不同 ⇒ cost 里有规划器采样依赖（§1.2 被破）"
    elif ned["rel_median"] is None or ind["rel_median"] is None:
        v, why = "NO-DATA", "格不足"
    elif ned["rel_median"] > LINE_REL:
        v, why = ("G4-UNREACHABLE-AT-M16",
                  f"嵌套下新增 16 条路径的 rel 中位 {ned['rel_median']:.4f} > 10% ⇒ 线本身不可达")
    elif ind["rel_median"] > LINE_REL:
        v, why = ("G4-CONFOUNDED-BY-DECORRELATED-DRAWS",
                  f"嵌套 rel 中位 {ned['rel_median']:.4f} ≤10% 但独立 rel 中位 "
                  f"{ind['rel_median']:.4f} >10% ⇒ G4 崩来自两次抽样不配对，不是 M 不够")
    elif ind["agree_median"] >= LINE_AGREE:
        v, why = ("G4-PASSES-UNDER-REPEAT",
                  "两档 rel 中位都 ≤10%、agree 中位 ≥0.90 ⇒ 原 FAIL 是单点采样")
    else:
        v, why = ("G4-AGREE-FAILS-ALONE",
                  f"rel 中位都 ≤10% 但 agree 中位 {ind['agree_median']:.3f} <0.90 ⇒ 一致率是独立障碍")
    return {"verdict": v, "reason": why,
            "n_fail": sum(1 for c in checks.values() if not c.get("pass", False)),
            "n_checks": len(checks),
            "no_n1_n2": True, "no_threshold_change": True,
            "lines": {"rel": LINE_REL, "agree": LINE_AGREE},
            "rule": "判词按预注册 §3 映射；不改 0.90/10%；不读 N1/N2"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="CW-R1.7.5 G4 可达性 + 嵌套采样对照")
    ap.add_argument("--quick", action="store_true", help="4 个 seed、不跑 M=64（烟雾）")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--skip-g4", action="store_true")
    ap.add_argument("--note", default="")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    tracemalloc.start()

    seeds = SEEDS[:4] if args.quick else SEEDS
    eps = EPS
    print(f"[R1.7.5] 0/4 P0 结构性闸", flush=True)
    p0 = p0_structural()
    for k, v in p0.items():
        print(f"  {'PASS' if v['pass'] else 'FAIL'}  {k}", flush=True)

    print(f"[R1.7.5] 1/4 G4 口径恒等（默认路径重跑原函数）", flush=True)
    g4id = {"pass": False, "error": "skipped"} if args.skip_g4 else g4_identity()
    print(f"  {'PASS' if g4id['pass'] else 'FAIL'}  P0c_g4_identity  "
          f"{g4id.get('now', {})}", flush=True)

    print(f"[R1.7.5] 2/4 采样面板", flush=True)
    raw = build(seeds, eps, args.quick)

    print(f"[R1.7.5] 3/4 统计", flush=True)
    an = analyse(raw, seeds, eps, args.quick)
    an["g4_identity"] = g4id

    checks = settle(p0, an)
    v = verdict_of(checks, an)

    elapsed = round(time.perf_counter() - t0, 2)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"[R1.7.5] 4/4 判词 {v['verdict']}（{v['reason']}）；"
          f"{v['n_fail']} fail / {v['n_checks']} checks；{elapsed} s；peak {peak / 2 ** 20:.0f} MiB")
    for k, c in checks.items():
        print(f"  {'PASS' if c.get('pass') else 'FAIL'}  {k}")
    for tag in ("P1_indep", "P3_nested"):
        s = an["stats"][tag]
        print(f"  {tag:10s} n={s['n']:3d} rel 中位={s['rel_median']:.4f} "
              f"p90={s['rel_p90']:.4f} >10% 占比={s['rel_frac_gt_line']:.3f} "
              f"agree 中位={s['agree_median']:.3f}")
    p2 = an["stats"]["P2_slope"]
    print(f"  P2 斜率中位={p2['slope_median']} → {p2['band']}")

    if not args.no_write:
        now = datetime.now(timezone(timedelta(hours=8)))
        cfg = C.CWConfig(plant=C.bounded_plant())
        run_id = (f"cw_r1_7_5_reach_{now.strftime('%Y%m%dT%H%M%S')}_"
                  f"{cfg.fingerprint()}_{_src_hash()}")
        out_dir = RESULTS / run_id
        out_dir.mkdir(parents=True, exist_ok=True)

        def _dump(name, obj):
            (out_dir / name).write_text(
                json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

        _dump("p0_structural.json", p0)
        _dump("g4_identity.json", g4id)
        # 逐格原始（动作序列另存，避免主表过大）
        cells_light = {k: {kk: vv for kk, vv in v.items() if kk != "acts"}
                       for k, v in raw["cells"].items()}
        _dump("cells.json", cells_light)
        _dump("acts.json", {k: v["acts"] for k, v in raw["cells"].items()})
        _dump("stats.json", an["stats"])
        _dump("invariants.json", an["invariants"])
        _dump("checks.json", checks)
        _dump("verdict.json", v)
        _dump("meta.json", {
            "run_id": run_id,
            "kind": "G4-REACHABILITY-NOT-PANEL-CONCLUSION",
            "protocol": C.CWConfig.PROTOCOL_VERSION,
            "prereg": "docs/research/CW-R1.7.5-G4可达性-预注册-v0.1.md",
            "params": {"eps": list(eps), "seeds": list(seeds), "M_list": list(M_LIST),
                       "M_var": list(M_VAR), "rho": RHO, "ratio": RATIO,
                       "h_plan": H_PLAN, "arm": ARM, "pacing": True,
                       "lines": {"rel": LINE_REL, "agree": LINE_AGREE}},
            "quick": bool(args.quick),
            "config": cfg.to_dict(),
            "at_asia_shanghai": now.isoformat(),
            "elapsed_s": elapsed,
            "peak_tracemalloc_mib": round(peak / 2 ** 20, 1),
            "n_cells": raw["n_cells"],
            "seed_time_s": raw["elapsed_s"],
            "git_head": _git(["rev-parse", "HEAD"]),
            "git_status_porcelain": _git(["status", "--porcelain"]),
            "src_hash": _src_hash(),
            "change_to_subject": (
                "plan_ref._Draws / PlanRef.__init__ / make 各加可选参数 `pool_M` / "
                "`draws_pool_m`，默认 None ⇒ 默认路径逐位不变（由 P0c 与 P0b2 双闸钉住）"),
            "disclaimer": (
                "本轮只回答『G4 的 10% 线在 M=16 上能不能过』，不是科学结论；"
                "不读 N1/N2；不改任何阈值；不宣称『测量值不值』。"
                "4 集 × 8 seed 是开发诊断，不是确认集。"),
            "note": args.note,
        })
        print(f"[R1.7.5] 证据落盘 results/cw_v1/{run_id}")

    ok = v["verdict"] in ("G4-UNREACHABLE-AT-M16", "G4-CONFOUNDED-BY-DECORRELATED-DRAWS",
                          "G4-PASSES-UNDER-REPEAT", "G4-AGREE-FAILS-ALONE")
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
