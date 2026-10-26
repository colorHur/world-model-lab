# -*- coding: utf-8 -*-
"""`CW-R4.1`：「读一次能用几次」—— `ρ`（信道记忆长度）杠杆。

预注册：`docs/research/CW-R4.1-读一次能用几次-预注册-v0.1.md`（写于正式面板之前）
协议变体：`docs/research/CW-POMDP-v1.2-rho-协议变体.md`

上游：`CW-R4.0` 在 `ρ=0.85` 的 480 格上判 `IV_paid(σ⋆) ≡ 0`，毛价值最大 **0.1056 个量子**
（`α=2.5, κ=1, P=0.01, τ=5, b=1`）⇒ 边界落在**价格**上。本轮问它的**横轴**：
把一次读数的有效期从 3.6 个槽拉到 25 / 100 / 500 个槽，那个 `0.106` 会变成多少？

两条互相排斥的跑前预测（预注册 §1）：
`P-A` 有效期定律（价值 ∝ `1/(1−ρ²)`，跨 1 量子 ⇒ 能买）
`P-B` 窗口饱和（读数比窗口长之后不再增量 ⇒ 天花板 `<1`）
数据只须二选一。

纪律
----
1. 零 episode、零训练、零参数更新；被测物一字节不动（只 `import`，`ρ` 在构造配置时替换）。
2. 判据见预注册 §4，**跑前写死**；`judge()` 只按表取。
3. `ρ` 只允许进「读数之后信念怎么衰减」：根状态 `m=0,v=1` 是 `ρ` 映射的不动点
   ⇒ `A0` 与 `ΔV_q` 必须**逐位与 `ρ` 无关**（`A0` 闸）。
"""

from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import json
import math
import time
from pathlib import Path

from .config import ChannelConfig, mainline_config
from .r1_14_exactbase import _now_utc, _src_hash
from .r4_0_boundary import _plant_asym, build_engine_n

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results" / "cw_v1"

PROTOCOL = "CW-POMDP-v1.2-rho"
UPSTREAM_PROTOCOL = "CW-POMDP-v1.1"
ROUND = "CW-R4.1"

SIGMA_STAR = 1.0e-9
V_PRIOR = 1.0

#: 面板（预注册 §3，跑前写死；**不按结果移动**）
RHO_MAIN: tuple[float, ...] = (0.85, 0.90, 0.95, 0.98, 0.995)
RHO_STATIC = 0.999                 # 静态极限腿（只跑 α=2.5, P=0.01）
ALPHA_LIST: tuple[float, ...] = (1.0, 2.5)
KAPPA = 1.0                        # 固定（未扫：见交付「未扫维度」）
P_LIST: tuple[float, ...] = (0.01, 0.05)
TAU_B: tuple[tuple[int, int], ...] = ((2, 1), (3, 1), (3, 2), (4, 1), (4, 2),
                                      (4, 3), (5, 1), (5, 2), (5, 3), (5, 4))
N_M_MAIN = 65
N_M_REFINE = (65, 129)

#: 判据常数（预注册 §4）
IV_NOISE = 1e-12
UB_FLOOR = 1e-9
IDENT_TOL = 1e-13                  # `A0` 闸：跨 ρ 的 `A0/ΔV_q` 相对差
ANCHOR_TOL = 1e-9
LAW_TOL = 0.30
SAT_TOL = 0.10
SPREAD_MAX = 0.30
FIRST_TOL = 1e-15

#: R4.0 已记录值（锚点；写死，找不到就 FAIL 不降级）
R40_ANCHOR_RATIO = 0.10560288560562463
R40_ANCHOR_CELL = {"alpha": 2.5, "kappa": 1.0, "P": 0.01, "tau": 5, "b": 1}

_CFG_BASE = mainline_config()


# ==================================================================== 配置与臂
def rho_config(rho: float, alpha: float):
    """只替换 `rho_panel`（**改世界**，协议 v1.2-rho），其余与 v1.1 逐字段相同。"""
    plant = _plant_asym(float(alpha))
    ch = ChannelConfig(**{**_CFG_BASE.channel.__dict__, "rho_panel": (float(rho),)})
    return mainline_config(plant=plant, channel=ch)


class Arms:
    """一套 `(ρ, α, n_m)` 上的三条臂：`noprobe` / `free(σ⋆)` / `paid(σ⋆)`。"""

    def __init__(self, cfg, n_m: int):
        self.cfg = cfg
        self.n_m = int(n_m)
        t0 = time.perf_counter()
        self.no = build_engine_n(cfg, SIGMA_STAR, self.n_m, probe_enabled=False, n_users=2)
        self.fr = build_engine_n(cfg, SIGMA_STAR, self.n_m, probe_cost=0, n_users=2)
        self.pd = build_engine_n(cfg, SIGMA_STAR, self.n_m, probe_cost=1, n_users=2)
        self.sec_build = round(time.perf_counter() - t0, 3)

    def states(self) -> dict:
        return {"noprobe": self.no.n_states(), "free": self.fr.n_states(),
                "paid": self.pd.n_states()}

    def capped(self) -> bool:
        return bool(self.no.capped() or self.fr.capped() or self.pd.capped())


def cell(arms: Arms, tau: int, b: int, P0: float, P1: float) -> dict:
    """一格：`A0 / free / paid / ΔV_q` + 机制量（与 R4.0 `cell2` 逐字同口径）。"""
    vp = V_PRIOR
    a0 = arms.no(tau, (P0, P1), (0.0, 0.0), (vp, vp), b)
    a0m = arms.no(tau, (P0, P1), (0.0, 0.0), (vp, vp), b - 1) if b >= 1 else float("nan")
    f = arms.fr(tau, (P0, P1), (0.0, 0.0), (vp, vp), b)
    d = arms.pd(tau, (P0, P1), (0.0, 0.0), (vp, vp), b)
    dq = (a0m - a0) if b >= 1 else float("nan")

    ft = arms.fr.first(tau, (P0, P1), (0.0, 0.0), (vp, vp), b) if all(
        math.isfinite(x) for x in (a0, f, d)) else {}
    s_send = min((v for k, v in ft.items() if k.startswith("SEND_")), default=float("nan"))
    s_probe = min((v for k, v in ft.items() if k.startswith("PROBE_")), default=float("nan"))
    probe_arg = min((k for k in ft if k.startswith("PROBE_")), key=lambda k: ft[k]) if ft else None
    send_arg = min((k for k in ft if k.startswith("SEND_")), key=lambda k: ft[k]) if ft else None

    return {
        "tau": int(tau), "b": int(b), "P0": float(P0), "P1": float(P1), "n_m": arms.n_m,
        "A0": float(a0), "IV_free": float(a0 - f), "IV_paid": float(a0 - d),
        "dV_q": float(dq),
        "ratio_free": float((a0 - f) / dq) if (dq is not None and dq > 1e-12) else None,
        "ratio_paid": float((a0 - d) / dq) if (dq is not None and dq > 1e-12) else None,
        "deficit_free": float(s_probe - s_send),
        "probe_arg": probe_arg, "send_arg": send_arg,
        "probe_chosen": bool(s_probe <= s_send) if ft else None,
    }


# ==================================================================== 面板
def panel() -> dict:
    rows: list[dict] = []
    meta: list[dict] = []
    plan = [(r, a, P_LIST) for r in RHO_MAIN for a in ALPHA_LIST]
    plan += [(RHO_STATIC, 2.5, (0.01,))]        # 静态极限腿
    for rho, alpha, plist in plan:
        cfg = rho_config(rho, alpha)
        arms = Arms(cfg, N_M_MAIN)
        rows0 = len(rows)
        capped = False
        for P in plist:
            for tau, b in TAU_B:
                c = cell(arms, tau, b, P, P * KAPPA)
                c.update({"rho": float(rho), "alpha": float(alpha), "kappa": float(KAPPA),
                          "P": float(P)})
                rows.append(c)
                if arms.capped():
                    capped = True
                    break
            if capped:
                break
        mine = rows[rows0:]
        fin = [c for c in mine if isinstance(c["ratio_free"], float)]
        mx_free = max((c["IV_free"] for c in mine if math.isfinite(c["IV_free"])),
                      default=float("nan"))
        mx_paid = max((c["IV_paid"] for c in mine if math.isfinite(c["IV_paid"])),
                      default=float("nan"))
        mx_ratio = max((c["ratio_free"] for c in fin), default=float("nan"))
        # ★ 靶子必须**显式绑定到 max 那一格**：首版这里用了外层循环变量 `c`（残留值 = 最后一格
        #   `τ=5,b=4,P=0.05`，读数恰为 0）⇒ refine 挑中全 0 格、"spread=0.000" 假通过
        #   （"无靶子"同族第 6 次）。改用 `max(..., key=)` 并用 `g_argmax` 闸自一致。
        argmax = max(fin, key=lambda c: c["ratio_free"]) if fin else None
        st = arms.states()                     # ★ 求值之后才取（R3.2/R4.0 同族教训）
        meta.append({"rho": float(rho), "alpha": float(alpha),
                     "fp": cfg.fingerprint(), "sigma_w": list(cfg.plant.sigma_w),
                     "sec_build": arms.sec_build, "states": st,
                     "capped": bool(capped or arms.capped()),
                     "n_rows": len(mine), "n_nan_ratio": len(mine) - len(fin),
                     "max_IV_free": float(mx_free), "max_IV_paid": float(mx_paid),
                     "max_ratio_free": float(mx_ratio),
                     "argmax_ratio": ({k: argmax[k] for k in ("P", "tau", "b")}
                                      | {"ratio_free": argmax["ratio_free"],
                                         "IV_free": argmax["IV_free"],
                                         "dV_q": argmax["dV_q"]}) if argmax else None})
        print(f"  [ρ={rho:g} α={alpha:g}] rows={len(mine)} states={st} "
              f"capped={meta[-1]['capped']} maxIV_free={mx_free:.4e} "
              f"max_ratio={mx_ratio:.4f}", flush=True)
        del arms
        gc.collect()
    return {"rows": rows, "meta": meta}


def refine_argmax(rows: list, meta: list) -> list:
    """每个 `ρ` 上 `ratio_free` 最大的格在 `n_m ∈ {65,129}` 重算（预注册 §3）。

    ★ 靶子来自 `meta["argmax_ratio"]`（`max(..., key=)` 绑定），不是循环残留变量。
    ★ 靶子为空/低于地板 ⇒ **显式记 `skipped`，`spread_rel=None`**，不许把 0/0 记成 0.000
      （那会让"无靶子"伪装成"网格稳定"，R4.0 已踩过一次）。
    """
    out = []
    for m in meta:
        am = m.get("argmax_ratio")
        tgt = am.get("ratio_free") if am else None
        if not am or not isinstance(tgt, float) or not math.isfinite(tgt):
            out.append({"rho": m["rho"], "alpha": m["alpha"], "skipped": "no-finite-cell",
                        "spread_rel": None})
            continue
        if tgt <= UB_FLOOR:
            out.append({"rho": m["rho"], "alpha": m["alpha"],
                        "skipped": "no-target-below-floor", "target_ratio": float(tgt),
                        "spread_rel": None})
            print(f"  [refine ρ={m['rho']:g} α={m['alpha']:g}] SKIP no-target "
                  f"(max ratio {tgt:.3e} ≤ floor {UB_FLOOR:g})", flush=True)
            continue
        rho, alpha = float(m["rho"]), float(m["alpha"])
        cfg = rho_config(rho, alpha)
        rr = []
        t0 = time.perf_counter()
        for n_m in N_M_REFINE:
            arms = Arms(cfg, n_m)
            c = cell(arms, am["tau"], am["b"], am["P"], am["P"] * KAPPA)
            rr.append({"n_m": n_m, "ratio_free": c["ratio_free"],
                       "IV_free": c["IV_free"], "IV_paid": c["IV_paid"],
                       "dV_q": c["dV_q"], "capped": arms.capped()})
            del arms
            gc.collect()
        vals = [r["ratio_free"] for r in rr if isinstance(r["ratio_free"], float)]
        mx = max(vals) if vals else 0.0
        spread = float((mx - min(vals)) / mx) if mx > 1e-15 else None
        ident = abs(rr[0]["ratio_free"] - tgt) if isinstance(rr[0]["ratio_free"], float) else None
        out.append({"rho": rho, "alpha": alpha,
                    "cell": {"P": am["P"], "tau": am["tau"], "b": am["b"]},
                    "panel_ratio": float(tgt), "rows": rr, "spread_rel": spread,
                    "identity_abs_diff_nm65": ident,
                    "sec": round(time.perf_counter() - t0, 3)})
        print(f"  [refine ρ={rho:g} α={alpha:g} τ={am['tau']} b={am['b']} P={am['P']:g}] "
              f"ratio={[round(v, 4) if isinstance(v, float) else v for v in vals]} "
              f"spread={spread} ident={ident} {out[-1]['sec']}s", flush=True)
    return out


# ==================================================================== 闸
def gate_a0_rho_invariance(rows: list) -> dict:
    """`A0` 闸：`ρ` 只进信念 ⇒ 同一格 `A0`/`ΔV_q` 跨 `ρ` 必须一致。

    根状态 `m=0, v=V_PRIOR=1` 是 `ρ` 映射的不动点（`ρ²·1+(1−ρ²)=1`），
    无探测臂从不产生读数 ⇒ 它的整棵树与 `ρ` 无关。这条闸同时是**接线检查**：
    `ρ` 若从别处漏进 DP，这里会立刻转红。
    """
    by = {}
    for c in rows:
        key = (c["alpha"], c["P"], c["tau"], c["b"])
        by.setdefault(key, []).append(c)
    worst_a0 = worst_dq = 0.0
    n_cmp = 0
    bad = []
    for key, cs in by.items():
        if len(cs) < 2:
            continue
        ref = cs[0]
        for c in cs[1:]:
            n_cmp += 1
            for fld, store in (("A0", "a0"), ("dV_q", "dq")):
                a, b_ = float(ref[fld]), float(c[fld])
                if not (math.isfinite(a) and math.isfinite(b_)):
                    continue
                rel = abs(a - b_) / max(1e-12, abs(a))
                if fld == "A0":
                    worst_a0 = max(worst_a0, rel)
                else:
                    worst_dq = max(worst_dq, rel)
                if rel > IDENT_TOL:
                    bad.append({"key": list(key), "field": fld, "ref": a, "cur": b_,
                                "rel": rel, "rho_ref": ref["rho"], "rho_cur": c["rho"]})
    worst = max(worst_a0, worst_dq)
    return {"n_cmp": n_cmp, "max_rel_A0": float(worst_a0), "max_rel_dVq": float(worst_dq),
            "max_rel": float(worst), "tol": IDENT_TOL, "bad": bad[:10],
            "pass": bool(n_cmp > 0 and worst <= IDENT_TOL)}


def gate_g_argmax_consistent(rows: list, meta: list) -> dict:
    """靶子自一致：`meta["argmax_ratio"]["ratio_free"]` 必须**等于**该档面板里 `ratio_free` 的 max。

    ★ 这条闸是首跑的现场教训：`argmax_ratio` 写成 `{k: c[k] ...} if fin` 时用的是外层循环
      残留变量 ⇒ 记录成"最后一格"（`τ=5,b=4,P=0.05`，读数 0），refine 于是挑中全 0 格、
      还把 `0/0` 报成 `spread=0.000`（= "网格稳定"）。**归零的靶子 = 没有靶子**。
    """
    rows_bad = []
    for m in meta:
        cs = [c for c in rows if abs(c["rho"] - m["rho"]) < 1e-12
              and abs(c["alpha"] - m["alpha"]) < 1e-12
              and isinstance(c["ratio_free"], float)]
        if not cs:
            continue
        mx = max(c["ratio_free"] for c in cs)
        am = m.get("argmax_ratio")
        got = am.get("ratio_free") if am else float("nan")
        d = abs(float(got) - mx) if isinstance(got, float) else float("inf")
        if not (d <= 1e-15):
            rows_bad.append({"rho": m["rho"], "alpha": m["alpha"], "max": mx, "recorded": got,
                             "abs_diff": d})
    return {"n_bad": len(rows_bad), "bad": rows_bad[:10],
            "pass": bool(len(rows_bad) == 0)}


def gate_r1_anchor(rows: list) -> dict:
    hit = [c for c in rows if abs(c["rho"] - 0.85) < 1e-12
           and abs(c["alpha"] - R40_ANCHOR_CELL["alpha"]) < 1e-12
           and abs(c["P"] - R40_ANCHOR_CELL["P"]) < 1e-12
           and c["tau"] == R40_ANCHOR_CELL["tau"] and c["b"] == R40_ANCHOR_CELL["b"]]
    if not hit:
        return {"cell_found": False, "pass": False}
    r = hit[0]["ratio_free"]
    d = abs(float(r) - R40_ANCHOR_RATIO) if isinstance(r, float) else float("nan")
    return {"cell_found": True, "ratio_mine": r, "ratio_r40": R40_ANCHOR_RATIO,
            "abs_diff": float(d), "tol": ANCHOR_TOL,
            "pass": bool(math.isfinite(d) and d <= ANCHOR_TOL)}


def gate_g_fingerprint(meta: list) -> dict:
    """`G8` 同族：每个 `(ρ, α)` 必须是**互不相同**的配置（R4.0 首跑 80 格重复单元的教训）。"""
    fps = {}
    for m in meta:
        fps[f"rho={m['rho']:g},alpha={m['alpha']:g}"] = m["fp"]
    uniq = len(set(fps.values()))
    return {"n_config": len(fps), "n_distinct": uniq, "fps": fps,
            "pass": bool(uniq == len(fps))}


def gate_g_no_truth() -> dict:
    src = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    banned = {"h_true", "cost_true_by_user", "cost_true", "w_realized", "x_true"}
    hits = [n.attr for n in ast.walk(tree)
            if isinstance(n, ast.Attribute) and n.attr in banned]
    hits += [n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id in banned]
    return {"banned": sorted(banned), "hits": hits, "pass": not hits}


def _file_hash(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def gate_g_identity_files() -> dict:
    mods = ["r4_0_boundary.py", "r3_2_iv2u.py", "config.py", "posterior.py"]
    h = {m: _file_hash(REPO / "experiments" / "cw_v1" / m) for m in mods}
    return {"hashes": h, "dir_hash": _src_hash(), "pass": bool(all(v for v in h.values()))}


def gate_g_first_consistency() -> dict:
    """`first()` 的首步最优必须等于 `V`（`n_m=33`，`ρ=0.995`，最便宜的接线检查）。"""
    cfg = rho_config(0.995, 2.5)
    arms = Arms(cfg, 33)
    rows, worst = [], 0.0
    for tau, b, P in ((3, 1, 0.05), (4, 2, 0.01), (4, 3, 0.05)):
        for name, fn in (("noprobe", arms.no), ("free", arms.fr), ("paid", arms.pd)):
            v = fn(tau, (P, P), (0.0, 0.0), (V_PRIOR, V_PRIOR), b)
            f = fn.first(tau, (P, P), (0.0, 0.0), (V_PRIOR, V_PRIOR), b)
            d = abs(min(f.values()) - v)
            worst = max(worst, d)
            rows.append({"tau": tau, "b": b, "arm": name, "V": float(v),
                         "min_first": float(min(f.values())), "abs_diff": float(d)})
    return {"rows": rows, "max_abs": float(worst), "tol": FIRST_TOL,
            "pass": bool(worst <= FIRST_TOL)}


def gate_g_drop_obs() -> dict:
    """变异：把读数丢弃后必须**逐位回到 `A0`**（`ρ=0.995`，即读数最耐放的一档）。

    ★ 这条是 `L1` 的活体对照：若在"信息最有用"的世界里丢读数都不掉分，
      说明面板根本没在用读数，`R5` 的全零不能当证据。
    """
    cfg = rho_config(0.995, 2.5)
    no = build_engine_n(cfg, SIGMA_STAR, 33, probe_enabled=False, n_users=2)
    drop = build_engine_n(cfg, SIGMA_STAR, 33, probe_cost=0, drop_obs=True, n_users=2)
    rows, worst = [], 0.0
    for tau, b, P in ((3, 1, 0.05), (4, 2, 0.01), (4, 3, 0.05)):
        a0 = no(tau, (P, P), (0.0, 0.0), (V_PRIOR, V_PRIOR), b)
        dd = drop(tau, (P, P), (0.0, 0.0), (V_PRIOR, V_PRIOR), b)
        d = abs(a0 - dd)
        worst = max(worst, d)
        rows.append({"tau": tau, "b": b, "P": P, "A0": float(a0), "drop": float(dd),
                     "abs_diff": float(d)})
    return {"rows": rows, "max_abs": float(worst), "tol": FIRST_TOL,
            "pass": bool(worst <= FIRST_TOL)}


def gate_g_blackwell_sigma() -> dict:
    """`σ` 越大信息越少 ⇒ `IV_free(σ=0.05) ≤ IV_free(σ⋆)`（`ρ=0.995`，`n_m=33`）。"""
    cfg = rho_config(0.995, 2.5)
    no = build_engine_n(cfg, SIGMA_STAR, 33, probe_enabled=False, n_users=2)
    f_star = build_engine_n(cfg, SIGMA_STAR, 33, probe_cost=0, n_users=2)
    f_noisy = build_engine_n(cfg, 0.05, 33, probe_cost=0, n_users=2)
    rows, bad = [], 0
    for tau, b, P in ((4, 2, 0.01), (5, 2, 0.01), (5, 3, 0.05)):
        a0 = no(tau, (P, P), (0.0, 0.0), (V_PRIOR, V_PRIOR), b)
        s = a0 - f_star(tau, (P, P), (0.0, 0.0), (V_PRIOR, V_PRIOR), b)
        n_ = a0 - f_noisy(tau, (P, P), (0.0, 0.0), (V_PRIOR, V_PRIOR), b)
        ok = bool(n_ <= s + 1e-15)
        bad += 0 if ok else 1
        rows.append({"tau": tau, "b": b, "P": P, "IV_star": float(s), "IV_sigma05": float(n_),
                     "ok": ok})
    return {"rows": rows, "n_violation": bad, "pass": bool(bad == 0)}


# ==================================================================== 判词
def _fin(x) -> bool:
    return isinstance(x, float) and math.isfinite(x)


def judge(p: dict, refine: list, gates: dict) -> dict:
    rows = p["rows"]
    out: dict = {"n_cells": len(rows)}
    fin = [c for c in rows if _fin(c["ratio_free"]) and _fin(c["IV_paid"])]
    out["n_cells_finite"] = len(fin)
    out["n_cells_dropped"] = len(rows) - len(fin)
    rhos = sorted({c["rho"] for c in rows})

    # ---- 每个 ρ 的上限（毛价值，折成量子）
    ceil = {}
    for r in rhos:
        cs = [c for c in fin if abs(c["rho"] - r) < 1e-12]
        if not cs:
            ceil[f"{r:g}"] = None
            continue
        bst = max(cs, key=lambda c: c["ratio_free"])
        ceil[f"{r:g}"] = {
            "max_ratio_free": float(bst["ratio_free"]),
            "cell": {k: bst[k] for k in ("alpha", "P", "tau", "b")},
            "IV_free": float(bst["IV_free"]), "dV_q": float(bst["dV_q"]),
            "probe_chosen": bst["probe_chosen"],
            "n_free_above_floor": int(sum(1 for c in cs if c["IV_free"] > UB_FLOOR)),
            "n_cells": len(cs),
            "n_paid_positive": int(sum(1 for c in cs if c["IV_paid"] > IV_NOISE)),
            "max_IV_paid": float(max(c["IV_paid"] for c in cs)),
        }
    out["ceiling_by_rho"] = ceil

    # ---- R5 主判：有没有任何格买得起
    hits = [c for c in fin if c["IV_paid"] > IV_NOISE]
    out["R5"] = "RHO-BUYS" if hits else "NO-RHO-RESCUE"
    out["R5_detail"] = {
        "n_hits": len(hits), "noise": IV_NOISE,
        "cells": sorted([{k: c[k] for k in ("rho", "alpha", "P", "tau", "b")}
                         | {"IV_paid": c["IV_paid"], "ratio_paid": c["ratio_paid"],
                            "IV_free": c["IV_free"]} for c in hits],
                        key=lambda x: -x["IV_paid"])[:20],
        "n_paid_exact_zero": int(sum(1 for c in fin if abs(c["IV_paid"]) <= 1e-15)),
    }

    # ---- R2 单调：同一格 ratio_free 随 ρ 非降
    by = {}
    for c in fin:
        by.setdefault((c["alpha"], c["P"], c["tau"], c["b"]), []).append(c)
    viol = []
    n_cmp = 0
    for key, cs in by.items():
        cs = sorted(cs, key=lambda c: c["rho"])
        for a, b_ in zip(cs, cs[1:]):
            n_cmp += 1
            if b_["ratio_free"] < a["ratio_free"] - 1e-12:
                viol.append({"key": list(key), "rho_from": a["rho"], "rho_to": b_["rho"],
                             "ratio_from": a["ratio_free"], "ratio_to": b_["ratio_free"]})
    out["R2"] = "MONOTONE-IN-MEMORY" if (n_cmp and not viol) else (
        "NON-MONOTONE" if viol else "NO-COMPARABLE-PAIR")
    out["R2_detail"] = {"n_cmp": n_cmp, "n_violation": len(viol), "violations": viol[:10]}

    # ---- R3 有效期定律：跨 ρ 的倍数 vs 1/(1−ρ²) 的倍数
    base_key = "0.85"
    base = ceil.get(base_key)
    law = {}
    for r in rhos:
        c = ceil.get(f"{r:g}")
        if not c or not base or base["max_ratio_free"] <= 1e-15:
            law[f"{r:g}"] = None
            continue
        pred = (1.0 / (1.0 - r * r)) / (1.0 / (1.0 - 0.85 ** 2))
        meas = c["max_ratio_free"] / base["max_ratio_free"]
        ratio = meas / pred
        law[f"{r:g}"] = {"pred_factor": float(pred), "measured_factor": float(meas),
                         "measured_over_pred": float(ratio),
                         "holds": bool(abs(ratio - 1.0) <= LAW_TOL)}
    out["R3_detail"] = {"law": law, "tol": LAW_TOL, "base_rho": 0.85}
    holds = [v for v in law.values() if v]
    out["R3"] = ("LIFETIME-LAW-HOLDS" if holds and all(v["holds"] for v in holds)
                 else "LIFETIME-LAW-BREAKS")
    out["R3_partial"] = [k for k, v in law.items() if v and v["holds"]]

    # ---- R4 饱和：0.995 / 0.999 相对 0.98
    c98 = ceil.get("0.98")
    sat = {}
    for k in ("0.995", "0.999"):
        c = ceil.get(k)
        if not c or not c98 or c98["max_ratio_free"] <= 1e-15:
            sat[k] = None
            continue
        rel = abs(c["max_ratio_free"] - c98["max_ratio_free"]) / c98["max_ratio_free"]
        sat[k] = {"ratio": c["max_ratio_free"], "rel_diff_vs_098": float(rel),
                  "saturated": bool(rel <= SAT_TOL)}
    out["R4_detail"] = {"vs": "0.98", "tol": SAT_TOL, "sat": sat}
    sv = [v for v in sat.values() if v]
    out["R4"] = ("SATURATED-AT-WINDOW" if sv and all(v["saturated"] for v in sv)
                 else "STILL-GROWING")

    # ---- R6 分辨率（★ "无靶子" 与 "网格稳定" 必须分开报）
    spreads = [r.get("spread_rel") for r in refine if isinstance(r.get("spread_rel"), float)]
    ident_bad = [r for r in refine
                 if isinstance(r.get("identity_abs_diff_nm65"), float)
                 and r["identity_abs_diff_nm65"] > 1e-12]
    skipped = [r for r in refine if "skipped" in r]
    if ident_bad:
        out["R6"] = "REFINE-IDENTITY-BROKEN"
    elif len(spreads) >= 2 and max(spreads) <= SPREAD_MAX:
        out["R6"] = "GRID-STABLE"
    elif spreads:
        out["R6"] = "GRID-DEPENDENT"
    else:
        out["R6"] = "NO-REFINE-TARGET"
    out["R6_detail"] = {"n_refined": len([r for r in refine if "rows" in r]),
                        "max_spread_rel": float(max(spreads)) if spreads else None,
                        "threshold": SPREAD_MAX, "skipped": skipped,
                        "identity_bad": ident_bad}

    # ---- L1 活体
    top = ceil.get("0.999") or ceil.get("0.995")
    live_val = top["max_ratio_free"] if top else float("nan")
    drop = gates.get("g_drop", {})
    out["L1"] = ("MACHINERY-CAN-BE-POSITIVE"
                 if (top and live_val > UB_FLOOR and drop.get("pass"))
                 else "MACHINERY-CANNOT-BE-POSITIVE")
    out["L1_detail"] = {"rho_static_max_ratio": live_val, "floor": UB_FLOOR,
                        "drop_obs_pass": drop.get("pass")}

    out["gates_n_fail"] = sum(1 for v in gates.values()
                              if isinstance(v, dict) and not v.get("pass", True))

    # ---- 汇总（预注册 §4 末表：组合唯一）
    if out["gates_n_fail"] > 0:
        out["VERDICT"] = "NO-VERDICT-GATE-FAIL"
    elif not gates.get("g_r1_anchor", {}).get("pass"):
        out["VERDICT"] = "NO-VERDICT-ANCHOR-DIVERGED"
    elif out["L1"] != "MACHINERY-CAN-BE-POSITIVE":
        out["VERDICT"] = "NO-VERDICT-L1-MACHINERY-CANNOT-BE-POSITIVE"
    elif out["R5"] == "RHO-BUYS":
        out["VERDICT"] = "RHO-BUYS-AT-SLOW-CHANNEL"
    elif out["R4"] == "SATURATED-AT-WINDOW":
        out["VERDICT"] = "NO-RHO-RESCUE-SATURATED-AT-WINDOW"
    elif out["R3"] == "LIFETIME-LAW-HOLDS":
        out["VERDICT"] = "LIFETIME-LAW-HOLDS-BUT-BELOW-ONE"
    else:
        out["VERDICT"] = "NO-RHO-RESCUE-LAW-BREAKS"
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="CW-R4.1 读一次能用几次（ρ 杠杆）")
    ap.add_argument("--ctrl-only", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    print(f"[{ROUND}] {PROTOCOL} start", flush=True)

    if args.ctrl_only:
        g = {"g_no_truth": gate_g_no_truth(),
             "g_identity": gate_g_identity_files(), "g_first": gate_g_first_consistency(),
             "g_drop": gate_g_drop_obs(), "g_blackwell": gate_g_blackwell_sigma()}
        print(json.dumps(g, indent=2, ensure_ascii=False)[:5000])
        print(f"[ctrl-only] n_fail={sum(1 for v in g.values() if not v['pass'])} / {len(g)}")
        return 0

    p = panel()
    print(f"  panel done rows={len(p['rows'])}", flush=True)
    refine = refine_argmax(p["rows"], p["meta"])

    gates = {"a0": gate_a0_rho_invariance(p["rows"]),
             "g_r1_anchor": gate_r1_anchor(p["rows"]),
             "g_argmax": gate_g_argmax_consistent(p["rows"], p["meta"]),
             "g_fp": gate_g_fingerprint(p["meta"]),
             "g_no_truth": gate_g_no_truth(),
             "g_identity": gate_g_identity_files(),
             "g_first": gate_g_first_consistency(),
             "g_drop": gate_g_drop_obs(),
             "g_blackwell": gate_g_blackwell_sigma()}
    v = judge(p, refine, gates)

    out = {"round": ROUND, "protocol": PROTOCOL, "protocol_upstream": UPSTREAM_PROTOCOL,
           "utc": _now_utc(), "sec_total": round(time.perf_counter() - t0, 3),
           "panel_meta": p["meta"], "cells": p["rows"], "refine": refine,
           "gates": gates, "verdicts": v}

    print(json.dumps({k: v[k] for k in ("VERDICT", "R5", "R5_detail", "R2", "R3", "R3_detail",
                                        "R4", "R4_detail", "R6", "L1", "ceiling_by_rho",
                                        "gates_n_fail")},
                     indent=2, ensure_ascii=False)[:12000], flush=True)

    dest = Path(args.out) if args.out else (
        RESULTS / f"cw_r4_1_rho_{_now_utc()}_{_src_hash()[:16]}")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "r41.json").write_text(json.dumps(out, indent=2, ensure_ascii=False),
                                   encoding="utf-8")
    print(f"[evidence] {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
