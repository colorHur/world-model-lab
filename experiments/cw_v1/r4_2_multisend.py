# -*- coding: utf-8 -*-
"""`CW-R4.2`：「一次读数能服务几次发送」—— `b`（可用量子数）× `τ`（窗口）双轴。

预注册：`docs/research/CW-R4.2-一次读数服务几次发送-预注册-v0.1.md`（含 §3.2 可行性修订）
协议：`CW-POMDP-v1.2-rho`（沿用 R4.1；本轮**不改任何世界参数**）

上游矛盾（本轮要判的）
----------------------
R4.1 §7.1-4 写「峰值格是单次发送时机问题，读数一覆盖窗口就没有**第二次发射机会**」，
但同一证据文件里 `ρ=0.999, α=2.5, P=0.01, τ=5` 的 `ratio_free` 随 `b` 是
`0.15254 (b=1) → 0.04399 (b=2) → 0.0 (b=3) → 0.0 (b=4)`，且 `b≥3` 上最优策略**连免费探测都不做**。
若"没有第二次发射机会"是卡点，`b=3`（探测后还剩 2 次发送）应当更值钱，而不是归零。

三条跑前预测（预注册 §1）：`Q-A` 多服务放大 / `Q-B` 稀缺替代 / `Q-C` 窗口主导。

分腿（预注册 §3.2，实测状态数决定）
-----------------------------------
`A`：`n_m=33, τ∈{5,6,7}, b∈{1,2,3,4}`（主判：`b` 阶梯 + `τ` 趋势）
`B`：`n_m=17, τ∈{5,6,7,8}, b∈{1,2,3,4}`（唯一能到 `τ=8` 的自洽网格）
`C`：`n_m=65, τ∈{5,6,7}, b∈{1,2,3}`（分辨率核对；`b=4` 在 65 上撞顶 ⇒ 未扫）

纪律
----
1. 零 episode、零训练、零参数更新；被测物一字节不动（只 `import`）。
2. `b` 轴/`τ` 轴的比较**只在同一腿内**做；跨腿只做方向核对，不把两腿的数相除当效应量。
3. `b=1` 上付费探测**结构上无发送可用**（探测吃掉唯一量子后 `b-1=0` ⇒ 无 SEND 分支）
   ⇒ `IV_paid(b=1) ≡ 0`；R4.0/R4.1 的"价格距离"恰是用这一格算的，本轮改在 `b≥2` 重算。
4. 凡 `IV_free == 0` 的格必须过 `g_probe_live`：`PROBE_i` 键**存在**且其值 `> min(first)`
   —— 零必须是"拒绝"，不能是"没接线"。
"""

from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import json
import math
import sys
import time
from pathlib import Path

from .config import ChannelConfig, mainline_config
from .r1_14_exactbase import _now_utc, _src_hash
from .r4_0_boundary import _plant_asym, build_engine_n

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results" / "cw_v1"

PROTOCOL = "CW-POMDP-v1.2-rho"
ROUND = "CW-R4.2"

SIGMA_STAR = 1.0e-9
V_PRIOR = 1.0

#: 面板（预注册 §3 + §3.2 实测修订；**不按结果移动**）
RHO_LIST: tuple[float, ...] = (0.98, 0.999)
ALPHA = 2.5
P_VALUE = 0.01
KAPPA = 1.0
N_USERS = 2
LEGS: tuple[dict, ...] = (
    {"id": "A", "n_m": 33, "taus": (5, 6, 7), "bs": (1, 2, 3, 4)},
    {"id": "B", "n_m": 17, "taus": (5, 6, 7, 8), "bs": (1, 2, 3)},
)
N_M_REFINE: tuple[int, ...] = (17, 33, 65)
REFINE_CELLS: tuple[tuple[int, int], ...] = ((7, 2), (7, 3), (8, 2))

#: 判据常数（预注册 §4）
IV_NOISE = 1e-12
UB_FLOOR = 1e-9
ANCHOR_TOL = 1e-9
FIRST_TOL = 1e-15
W_INTERP = 1.60
W_SAT = 1.30
MULTI_RATIO = 0.10

#: R4.1 已记录值（`r41.json`，`τ=5, α=2.5, P=0.01, n_m=65`；**写死，找不到就 FAIL 不降级**）
ANCHOR41: dict[tuple[float, int], float] = {
    (0.98, 1): 0.14607796192918882, (0.98, 2): 0.03363912786276052,
    (0.98, 3): 0.0, (0.98, 4): 0.0,
    (0.999, 1): 0.15253986342060927, (0.999, 2): 0.04399015647560396,
    (0.999, 3): 0.0, (0.999, 4): 0.0,
}

_CFG_BASE = mainline_config()


# ==================================================================== 配置
def rho_config(rho: float):
    plant = _plant_asym(ALPHA)
    ch = ChannelConfig(**{**_CFG_BASE.channel.__dict__, "rho_panel": (float(rho),)})
    return mainline_config(plant=plant, channel=ch)


class Arms:
    def __init__(self, cfg, n_m: int):
        self.cfg = cfg
        self.n_m = int(n_m)
        t0 = time.perf_counter()
        self.no = build_engine_n(cfg, SIGMA_STAR, self.n_m, probe_enabled=False,
                                 n_users=N_USERS)
        self.fr = build_engine_n(cfg, SIGMA_STAR, self.n_m, probe_cost=0,
                                 n_users=N_USERS)
        self.pd = build_engine_n(cfg, SIGMA_STAR, self.n_m, probe_cost=1,
                                 n_users=N_USERS)
        self.sec_build = round(time.perf_counter() - t0, 3)

    def states(self) -> dict:
        return {"noprobe": self.no.n_states(), "free": self.fr.n_states(),
                "paid": self.pd.n_states()}

    def capped(self) -> bool:
        return bool(self.no.capped() or self.fr.capped() or self.pd.capped())


def cell(arms: Arms, tau: int, b: int) -> dict:
    """一格：`A0 / free / paid / ΔV_q` ＋ 机制量（`first()` 的探测间隙）。"""
    vp = V_PRIOR
    t0 = time.perf_counter()
    a0 = arms.no(tau, (P_VALUE, P_VALUE), (0.0, 0.0), (vp, vp), b)
    a0m = arms.no(tau, (P_VALUE, P_VALUE), (0.0, 0.0), (vp, vp), b - 1) if b >= 1 else float("nan")
    f = arms.fr(tau, (P_VALUE, P_VALUE), (0.0, 0.0), (vp, vp), b)
    d = arms.pd(tau, (P_VALUE, P_VALUE), (0.0, 0.0), (vp, vp), b)
    dq = (a0m - a0) if b >= 1 else float("nan")
    sec = round(time.perf_counter() - t0, 3)

    ft = arms.fr.first(tau, (P_VALUE, P_VALUE), (0.0, 0.0), (vp, vp), b) if all(
        math.isfinite(x) for x in (a0, f, d)) else {}
    probe_keys = sorted(k for k in ft if k.startswith("PROBE_"))
    s_min = min(ft.values()) if ft else float("nan")
    s_probe = min((ft[k] for k in probe_keys), default=float("nan"))
    probe_gap = float(s_probe - s_min) if ft and probe_keys else float("nan")

    iv_free = float(a0 - f)
    iv_paid = float(a0 - d)
    return {
        "tau": int(tau), "b": int(b), "P": float(P_VALUE), "n_m": arms.n_m,
        "A0": float(a0), "V_free": float(f), "V_paid": float(d),
        "IV_free": iv_free, "IV_paid": iv_paid, "dV_q": float(dq),
        "ratio_free": float(iv_free / dq) if (dq and dq > 1e-12) else None,
        "ratio_paid": float(iv_paid / dq) if (dq and dq > 1e-12) else None,
        "frac_task": float(iv_free / a0) if a0 > 1e-12 else None,
        "bound_a0_over_dq": float(a0 / dq) if (dq and dq > 1e-12) else None,
        "probe_chosen": bool(s_probe <= s_min) if ft else None,
        "probe_keys": probe_keys,
        "probe_gap": probe_gap,
        "sec": sec,
        "capped": bool(arms.capped()),
    }


# ==================================================================== 面板
def run_leg(cfg, rho: float, leg: dict) -> dict:
    n_m = int(leg["n_m"])
    arms = Arms(cfg, n_m)
    rows: list[dict] = []
    stop = False
    for tau in sorted(leg["taus"], reverse=True):
        for b in sorted(leg["bs"], reverse=True):
            if b > tau - 1:
                continue
            c = cell(arms, tau, b)
            c.update({"rho": float(rho), "alpha": ALPHA, "kappa": KAPPA,
                      "leg": leg["id"], "n_m": n_m})
            rows.append(c)
            if c["capped"]:
                print(f"  [CAP] leg={leg['id']} ρ={rho:g} τ={tau} b={b} "
                      f"states={arms.states()}", flush=True)
                stop = True
                break
        if stop:
            break
    meta = {"leg": leg["id"], "rho": float(rho), "n_m": n_m, "alpha": ALPHA,
            "fp": cfg.fingerprint(), "sigma_w": list(cfg.plant.sigma_w),
            "sec_build": arms.sec_build, "capped": bool(arms.capped()),
            "n_rows": len(rows),
            "taus": list(leg["taus"]), "bs": list(leg["bs"])}
    print(f"  [leg {leg['id']} ρ={rho:g} n_m={n_m}] rows={len(rows)} "
          f"capped={meta['capped']} states={arms.states()}", flush=True)
    del arms
    gc.collect()
    return {"rows": rows, "meta": meta}


def panel() -> dict:
    rows: list[dict] = []
    meta: list[dict] = []
    for leg in LEGS:
        for rho in RHO_LIST:
            p = run_leg(rho_config(rho), rho, leg)
            rows += p["rows"]
            meta.append(p["meta"])
    return {"rows": rows, "meta": meta}


# ==================================================================== 闸
def gate_anchor41() -> dict:
    """`τ=5, n_m=65` 的 8 个 R4.1 锚点格必须**逐格复现**（`≤1e-9`）。"""
    out, bad = [], []
    cache: dict[float, Arms] = {}
    for (rho, b), want in sorted(ANCHOR41.items()):
        if rho not in cache:
            cache[rho] = Arms(rho_config(rho), 65)
        got = cell(cache[rho], 5, b)["ratio_free"]
        d = abs(float(got) - want) if isinstance(got, float) else float("inf")
        rec = {"rho": rho, "b": b, "want": want, "got": got, "abs_diff": d}
        out.append(rec)
        if not (d <= ANCHOR_TOL):
            bad.append(rec)
        print(f"  [anchor ρ={rho:g} b={b}] got={got!r} want={want!r} diff={d:.3e}",
              flush=True)
    del cache
    gc.collect()
    return {"rows": out, "n_bad": len(bad), "bad": bad, "tol": ANCHOR_TOL,
            "pass": bool(len(bad) == 0 and len(out) == len(ANCHOR41))}


def gate_paid_le_free(rows: list) -> dict:
    """同 `b` 下付费臂策略集 ⊆ 免量子臂策略集 ⇒ `V_free ≤ V_paid` ⇒ `IV_paid ≤ IV_free`。"""
    bad = [{"leg": c["leg"], "rho": c["rho"], "tau": c["tau"], "b": c["b"],
            "IV_free": c["IV_free"], "IV_paid": c["IV_paid"]}
           for c in rows
           if isinstance(c["IV_free"], float) and isinstance(c["IV_paid"], float)
           and c["IV_paid"] > c["IV_free"] + 1e-15]
    return {"n_violation": len(bad), "bad": bad[:8], "pass": bool(not bad)}


def gate_bound_a0(rows: list) -> dict:
    """`IV_free ≤ A0`（因 `V_free ≥ 0`）。"""
    bad = [{"leg": c["leg"], "rho": c["rho"], "tau": c["tau"], "b": c["b"],
            "A0": c["A0"], "IV_free": c["IV_free"]}
           for c in rows
           if isinstance(c["IV_free"], float) and c["IV_free"] > c["A0"] + 1e-12]
    slack = [c["ratio_free"] / c["bound_a0_over_dq"] for c in rows
             if isinstance(c.get("ratio_free"), float)
             and isinstance(c.get("bound_a0_over_dq"), float)
             and c["bound_a0_over_dq"] > 1e-12]
    return {"n_violation": len(bad), "bad": bad[:8],
            "max_slack_used": float(max(slack)) if slack else None,
            "pass": bool(not bad)}


def gate_probe_live(rows: list) -> dict:
    """★ 每个 `IV_free == 0` 的格：`PROBE_i` 键必须**存在**且其值 `> min(first)`。"""
    zeros = [c for c in rows
             if isinstance(c["IV_free"], float) and c["IV_free"] <= IV_NOISE
             and not c["capped"]]
    bad = []
    for c in zeros:
        if not c["probe_keys"]:
            bad.append({"leg": c["leg"], "rho": c["rho"], "tau": c["tau"],
                        "b": c["b"], "why": "no-PROBE-key"})
        elif not (isinstance(c["probe_gap"], float) and c["probe_gap"] > 0):
            bad.append({"leg": c["leg"], "rho": c["rho"], "tau": c["tau"],
                        "b": c["b"], "why": "probe-not-rejected",
                        "gap": c["probe_gap"]})
    return {"n_zero_cells": len(zeros), "n_bad": len(bad), "bad": bad[:8],
            "pass": bool(not bad)}


def gate_g_drop_obs() -> dict:
    """变异：丢弃读数后必须**逐位回到 `A0`**（`ρ=0.999, n_m=33`）。"""
    cfg = rho_config(0.999)
    no = build_engine_n(cfg, SIGMA_STAR, 33, probe_enabled=False, n_users=N_USERS)
    drop = build_engine_n(cfg, SIGMA_STAR, 33, probe_cost=0, drop_obs=True,
                          n_users=N_USERS)
    rows, worst = [], 0.0
    for tau, b in ((5, 2), (6, 2), (6, 3)):
        a0 = no(tau, (P_VALUE, P_VALUE), (0.0, 0.0), (V_PRIOR, V_PRIOR), b)
        dd = drop(tau, (P_VALUE, P_VALUE), (0.0, 0.0), (V_PRIOR, V_PRIOR), b)
        d = abs(a0 - dd)
        worst = max(worst, d)
        rows.append({"tau": tau, "b": b, "A0": float(a0), "drop": float(dd),
                     "abs_diff": float(d)})
    return {"rows": rows, "max_abs": float(worst), "tol": FIRST_TOL,
            "pass": bool(worst <= FIRST_TOL)}


def gate_g_blackwell() -> dict:
    """`σ` 越大信息越少 ⇒ `IV_free(σ=0.05) ≤ IV_free(σ⋆)`（`ρ=0.999, n_m=17`）。

    ★ 首版放在 `n_m=33` 上，`(τ=7,b=2)` 的 `σ=0.05` 臂**撞 `STATE_CAP`** 返回 `NaN`，
      而 `NaN <= x` 是 `False` ⇒ 被记成"Blackwell 违例"。**`NaN` 是"没测到"，不是"违例"**
      （R3.4 `hermegauss` 溢出同族）。现改成：`n_m=17`（两个 `σ` 都可行）＋
      `capped/NaN` 显式记成 `unmeasured` 并让闸**失败**（测不到就不许判过）。
    """
    n_m = 17
    cfg = rho_config(0.999)
    no = build_engine_n(cfg, SIGMA_STAR, n_m, probe_enabled=False, n_users=N_USERS)
    f_star = build_engine_n(cfg, SIGMA_STAR, n_m, probe_cost=0, n_users=N_USERS)
    f_noisy = build_engine_n(cfg, 0.05, n_m, probe_cost=0, n_users=N_USERS)
    rows = []
    bad = unmeasured = 0
    for tau, b in ((5, 2), (6, 2), (7, 2)):
        a0 = no(tau, (P_VALUE, P_VALUE), (0.0, 0.0), (V_PRIOR, V_PRIOR), b)
        s = a0 - f_star(tau, (P_VALUE, P_VALUE), (0.0, 0.0), (V_PRIOR, V_PRIOR), b)
        n_ = a0 - f_noisy(tau, (P_VALUE, P_VALUE), (0.0, 0.0), (V_PRIOR, V_PRIOR), b)
        fin = all(isinstance(x, float) and math.isfinite(x) for x in (s, n_))
        ok = bool(fin and n_ <= s + 1e-15)
        if not fin:
            unmeasured += 1
        elif not ok:
            bad += 1
        rows.append({"tau": tau, "b": b, "IV_star": float(s), "IV_sigma05": float(n_),
                     "measured": fin, "ok": ok})
    return {"rows": rows, "n_m": n_m, "n_violation": bad, "n_unmeasured": unmeasured,
            "pass": bool(bad == 0 and unmeasured == 0)}


def gate_g_first() -> dict:
    """`min(first()) == V`（`n_m=17, ρ=0.999`）。"""
    arms = Arms(rho_config(0.999), 17)
    rows, worst = [], 0.0
    for tau, b in ((5, 2), (6, 3), (7, 4)):
        for name, fn in (("noprobe", arms.no), ("free", arms.fr), ("paid", arms.pd)):
            v = fn(tau, (P_VALUE, P_VALUE), (0.0, 0.0), (V_PRIOR, V_PRIOR), b)
            f = fn.first(tau, (P_VALUE, P_VALUE), (0.0, 0.0), (V_PRIOR, V_PRIOR), b)
            d = abs(min(f.values()) - v)
            worst = max(worst, d)
            rows.append({"tau": tau, "b": b, "arm": name, "V": float(v),
                         "min_first": float(min(f.values())), "abs_diff": float(d)})
    return {"rows": rows, "max_abs": float(worst), "tol": FIRST_TOL,
            "pass": bool(worst <= FIRST_TOL)}


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
    return {"hashes": h, "dir_hash": _src_hash(), "pass": bool(all(h.values()))}


def gate_g_fp(meta: list) -> dict:
    """配置指纹互异：**键只用 `ρ`**（指纹只依赖配置，同一 `ρ` 的不同腿本来就同配置）。

    ★ 首版把键写成 `(leg, ρ)` ⇒ 同 `ρ` 的两条腿指纹必然相同 ⇒ `n_distinct < n_config`
      判成假失败。**闸的键集必须与被测对象的真身份一致**（R4.0 `G8` 同族）。
    """
    by_rho: dict[str, set] = {}
    for m in meta:
        by_rho.setdefault(f"{m['rho']:g}", set()).add(m.get("fp"))
    bad = {k: sorted(v) for k, v in by_rho.items() if len(v) != 1 or None in v}
    distinct = sorted({x for v in by_rho.values() for x in v if x})
    return {"n_rho": len(by_rho), "n_distinct_fp": len(distinct),
            "fp_by_rho": {k: (sorted(v)[0] if len(v) == 1 else sorted(v))
                          for k, v in by_rho.items()},
            "inconsistent": bad,
            "pass": bool(len(by_rho) > 0 and not bad
                         and len(distinct) == len(by_rho))}


# ==================================================================== 分辨率
def refine_cells() -> list:
    """`(τ,b) ∈ {(7,2),(7,3),(8,2)}` × `n_m ∈ {17,33,65}`（三点起步；撞顶要报出来）。"""
    out = []
    for rho in RHO_LIST:
        cfg = rho_config(rho)
        for tau, b in REFINE_CELLS:
            rr = []
            for n_m in N_M_REFINE:
                a = Arms(cfg, n_m)
                c = cell(a, tau, b)
                rr.append({"n_m": n_m, "ratio_free": c["ratio_free"],
                           "IV_free": c["IV_free"], "capped": c["capped"],
                           "sec": c["sec"]})
                del a
                gc.collect()
            vals = [(r["n_m"], r["ratio_free"]) for r in rr
                    if isinstance(r["ratio_free"], float) and not r["capped"]]
            spread = None
            if len(vals) >= 2 and max(v for _, v in vals) > 1e-15:
                spread = float((max(v for _, v in vals) - min(v for _, v in vals))
                               / max(v for _, v in vals))
            seg = None
            if len(vals) == 3 and abs(vals[0][1]) > 1e-15 and abs(vals[1][1]) > 1e-15:
                seg = {"17_to_33": abs(vals[1][1] - vals[0][1]) / abs(vals[0][1]),
                       "33_to_65": abs(vals[2][1] - vals[1][1]) / abs(vals[1][1])}
            out.append({"rho": rho, "tau": tau, "b": b, "rows": rr,
                        "n_feasible": len(vals), "spread_rel": spread, "seg": seg,
                        "grade": ("3-point" if len(vals) >= 3
                                  else "2-point-ONLY(cannot-declare-stable)")})
            print(f"  [refine ρ={rho:g} τ={tau} b={b}] "
                  f"{[(r['n_m'], r['ratio_free'], r['capped']) for r in rr]} "
                  f"spread={spread} seg={seg}", flush=True)
    return out


# ==================================================================== 判词
def _fin(x) -> bool:
    return isinstance(x, float) and math.isfinite(x)


def _leg_view(rows: list, leg: str, rho: float) -> dict:
    cs = [c for c in rows if c["leg"] == leg and abs(c["rho"] - rho) < 1e-12
          and not c["capped"] and _fin(c["ratio_free"])]
    by = {(c["tau"], c["b"]): c for c in cs}
    return {"cells": cs, "by": by}


def judge(p: dict, refine: list, gates: dict) -> dict:
    rows = [c for c in p["rows"] if not c["capped"]]
    out: dict = {"n_cells": len(p["rows"]), "n_cells_usable": len(rows),
                 "n_cells_capped": len(p["rows"]) - len(rows)}
    out["legs"] = {}
    all_multi, any_pos_multi = [], []
    best_buy = None

    for leg in LEGS:
        lid = leg["id"]
        out["legs"][lid] = {"n_m": leg["n_m"]}
        for rho in RHO_LIST:
            v = _leg_view(rows, lid, rho)
            by = v["by"]
            # --- b 阶梯
            ladder = []
            for tau in leg["taus"]:
                seq = [by.get((tau, b)) for b in leg["bs"]]
                ladder.append({"tau": tau,
                               "ratio_free": [c["ratio_free"] if c else None for c in seq],
                               "IV_free": [c["IV_free"] if c else None for c in seq],
                               "probe_chosen": [c["probe_chosen"] if c else None for c in seq]})
            # --- 单腿内 τ 趋势（b=2）
            wv = [(tau, by[(tau, 2)]["ratio_free"]) for tau in leg["taus"]
                  if (tau, 2) in by]
            # --- 稀缺替代律：同 (ρ,τ) 上 IV_free 对 b 非增
            viol, n_cmp = [], 0
            for tau in leg["taus"]:
                prev = None
                for b in sorted(leg["bs"]):
                    c = by.get((tau, b))
                    if not c:
                        continue
                    if prev is not None:
                        n_cmp += 1
                        if c["IV_free"] > prev["IV_free"] + 1e-15:
                            viol.append({"tau": tau, "b_from": b - 1, "b_to": b,
                                         "iv_from": prev["IV_free"],
                                         "iv_to": c["IV_free"]})
                    prev = c
            # --- b≥3 相对 b=2
            multi = []
            for tau in leg["taus"]:
                base = by.get((tau, 2))
                for b in leg["bs"]:
                    if b < 3 or (tau, b) not in by:
                        continue
                    c = by[(tau, b)]
                    multi.append({"rho": rho, "leg": lid, "tau": tau, "b": b,
                                  "IV_free": c["IV_free"],
                                  "ratio_free": c["ratio_free"],
                                  "base_IV_b2": base["IV_free"] if base else None,
                                  "probe_chosen": c["probe_chosen"]})
            all_multi += multi
            any_pos_multi += [m for m in multi if m["IV_free"] > IV_NOISE]
            # --- 购买区最大
            for c in v["cells"]:
                if c["b"] >= 2 and (best_buy is None
                                    or c["ratio_free"] > best_buy["ratio_free"]):
                    best_buy = c
            out["legs"][lid][f"rho_{rho:g}"] = {
                "ladder": ladder, "tau_b2": wv,
                "H1": {"n_cmp": n_cmp, "n_violation": len(viol), "violations": viol[:6],
                       "holds": bool(n_cmp and not viol)},
                "multi": multi,
                "max_ratio_free_b_ge_2": max((c["ratio_free"] for c in v["cells"]
                                              if c["b"] >= 2), default=None),
                "max_IV_paid": max((c["IV_paid"] for c in v["cells"]), default=None),
            }
        print(f"  [judge leg {lid}] done", flush=True)

    # ---- S1（全腿汇总；★ 三分类，跑后修订：原二分是"种类错"）
    amp = [m for m in all_multi
           if m["base_IV_b2"] and m["IV_free"] > m["base_IV_b2"] + IV_NOISE]
    nonvanish = [m for m in all_multi if m["IV_free"] > IV_NOISE]
    if amp:
        out["S1"] = "MULTI-SEND-AMPLIFIES"
    elif nonvanish:
        out["S1"] = "MULTI-SEND-NON-VANISHING"
    else:
        out["S1"] = "MULTI-SEND-VANISHES"
    out["S1_detail"] = {
        "criterion_note": ("跑后修订：原「b>=3 的毛价值 > 10% 的 b=2」把「不放大也不消失」"
                           "错记成 AMPLIFIES（种类错）；现分三类：AMPLIFIES（b>=3 严格大于 b=2）"
                           "/ NON-VANISHING（正但不超过 b=2）/ VANISHES（逐位零）。"
                           "原二分读数同报于 n_old_definition_hits。"),
        "n_cells_b_ge_3": len(all_multi),
        "n_positive": len(nonvanish),
        "n_amplify": len(amp),
        "n_old_definition_hits": len([m for m in all_multi
                                      if m["base_IV_b2"] and m["base_IV_b2"] > 1e-15
                                      and m["IV_free"] > max(IV_NOISE,
                                                             MULTI_RATIO * m["base_IV_b2"])]),
        "positive_cells": sorted(
            [{k: m[k] for k in ("rho", "leg", "tau", "b")}
             | {"IV_free": m["IV_free"],
                "ratio_free": m["ratio_free"]} for m in nonvanish],
            key=lambda x: -x["ratio_free"])[:16],
        "amplify_cells": amp[:8],
    }

    # ---- H1（全腿）
    h1 = [out["legs"][l["id"]][f"rho_{r:g}"]["H1"] for l in LEGS for r in RHO_LIST]
    n_v = sum(h["n_violation"] for h in h1)
    out["H1"] = ("SCARCITY-SUBSTITUTION-HOLDS" if (n_v == 0 and all(h["n_cmp"] for h in h1))
                 else "SCARCITY-SUBSTITUTION-BREAKS")
    out["H1_detail"] = {"n_violation_total": n_v,
                        "per_leg": {f"{l['id']}@{r:g}": out["legs"][l["id"]][f"rho_{r:g}"]["H1"]
                                    for l in LEGS for r in RHO_LIST}}

    # ---- W1（腿 B 是唯一能到 τ=8 的网格）
    w1 = {}
    for rho in RHO_LIST:
        pts = out["legs"]["B"][f"rho_{rho:g}"]["tau_b2"]
        r_w = (pts[-1][1] / pts[0][1]) if (len(pts) >= 2 and pts[0][1] > 1e-15) else None
        mono = (all(pts[i][1] < pts[i + 1][1] for i in range(len(pts) - 1))
                if len(pts) >= 2 else None)
        if r_w is None:
            w = "WINDOW-NO-DATA"
        elif r_w >= W_INTERP and mono:
            w = "WINDOW-STILL-GROWS"
        elif r_w < W_SAT:
            w = "WINDOW-SATURATES"
        else:
            w = "WINDOW-MIXED"
        w1[f"{rho:g}"] = {"points": pts, "R_W_8_over_5": r_w, "monotone": mono, "W": w}
    cross = {}
    for lid in ("A",):
        for rho in RHO_LIST:
            pts = out["legs"][lid][f"rho_{rho:g}"]["tau_b2"]
            r7 = (pts[-1][1] / pts[0][1]) if (len(pts) >= 2 and pts[0][1] > 1e-15) else None
            cross[f"{lid}@{rho:g}"] = {"points": pts, "R_last_over_5": r7,
                                       "monotone": (all(pts[i][1] < pts[i + 1][1]
                                                        for i in range(len(pts) - 1))
                                                    if len(pts) >= 2 else None)}
    out["W1_detail"] = {"leg_B": w1, "cross_leg_direction": cross,
                        "interp_threshold": W_INTERP, "sat_threshold": W_SAT}
    ws = [v["W"] for v in w1.values()]
    out["W1"] = ("WINDOW-SATURATES" if all(w == "WINDOW-SATURATES" for w in ws)
                 else "WINDOW-STILL-GROWS" if all(w == "WINDOW-STILL-GROWS" for w in ws)
                 else "WINDOW-MIXED")

    # ---- P1 / D1 / L1
    out["P1_detail"] = ({"leg": best_buy["leg"], "n_m": best_buy["n_m"],
                         "rho": best_buy["rho"], "tau": best_buy["tau"],
                         "b": best_buy["b"], "ratio_free": best_buy["ratio_free"],
                         "IV_free": best_buy["IV_free"],
                         "frac_task": best_buy["frac_task"],
                         "bound_a0_over_dq": best_buy["bound_a0_over_dq"]}
                        if best_buy else None)
    out["P1"] = ("BUYS-WITH-SPARE-SLOTS"
                 if best_buy and best_buy["ratio_free"] > 1.0 else "NOT-BUYS")

    # ---- D1 价格距离：**只能在可购买区（b>=2）算**
    b1 = [c for c in rows if c["b"] == 1]
    cz = [c for c in rows if c["b"] >= 2]
    mx_ratio_b2 = max((c["ratio_free"] for c in cz), default=float("nan"))
    out["D1_detail"] = {
        "max_ratio_free_b1": float(max((c["ratio_free"] for c in b1), default=float("nan"))),
        "max_ratio_free_b_ge_2": float(mx_ratio_b2),
        "price_distance_x_b_ge_2": float(1.0 / mx_ratio_b2) if mx_ratio_b2 and mx_ratio_b2 > 1e-15 else None,
        "b1_purchasable": "NO (整数记账：探测吃掉唯一量子 ⇒ 之后无发送 ⇒ IV_paid(b=1) 逐位零)",
        "note": ("R4.0/R4.1 报的 8.8×/6.2× 取的是 b=1 那格（0.1137/0.1617）—— 而 b=1 在整数"
                 "记账下**买不到**（付 1 个量子后剩 0 个发送）。可购买区的正确距离由本节给。"),
        "max_IV_paid_all": float(max((c["IV_paid"] for c in rows), default=float("nan"))),
        "n_paid_positive": int(sum(1 for c in rows if c["IV_paid"] > IV_NOISE)),
    }

    live = [c for c in rows if c["leg"] == "A" and c["tau"] == 7 and c["b"] == 2]
    gl = gates.get("g_probe_live", {})
    out["L1"] = ("MACHINERY-CAN-BE-POSITIVE"
                 if (live and live[0]["IV_free"] > UB_FLOOR and gl.get("pass"))
                 else "MACHINERY-CANNOT-BE-POSITIVE")
    out["L1_detail"] = {"live_cell": "leg A, τ=7, b=2, ρ=0.98",
                        "IV_free": live[0]["IV_free"] if live else None,
                        "floor": UB_FLOOR, "probe_live_pass": gl.get("pass")}

    # ---- H0（**跑后**，未预注册）：负载率 b/τ 与闲置槽 slack=τ−b
    post = []
    for leg in LEGS:
        for rho in RHO_LIST:
            for c in rows:
                if c["leg"] != leg["id"] or abs(c["rho"] - rho) > 1e-12:
                    continue
                post.append({"leg": leg["id"], "n_m": c["n_m"], "rho": rho,
                             "tau": c["tau"], "b": c["b"], "slack": c["tau"] - c["b"],
                             "load": c["b"] / c["tau"], "ratio_free": c["ratio_free"],
                             "IV_free": c["IV_free"]})
    peak_by_tau = {}
    for rho in RHO_LIST:
        for tau in sorted({p_["tau"] for p_ in post}):
            cs = [p_ for p_ in post if abs(p_["rho"] - rho) < 1e-12 and p_["tau"] == tau
                  and p_["leg"] == "A"]
            if cs:
                pk = max(cs, key=lambda x: x["ratio_free"])
                peak_by_tau[f"{rho:g}@τ={tau}"] = {"b_star": pk["b"], "n_m": pk["n_m"],
                                                   "ratio_free": pk["ratio_free"],
                                                   "slack": pk["slack"]}
    out["H0_posthoc"] = {"tag": "posthoc-not-preregistered", "points": post,
                         "peak_b_by_tau": peak_by_tau}

    out["gates_n_fail"] = sum(1 for v in gates.values()
                              if isinstance(v, dict) and not v.get("pass", True))

    if out["gates_n_fail"] > 0:
        out["VERDICT"] = "NO-VERDICT-GATE-FAIL"
    elif not gates.get("g_anchor41", {}).get("pass"):
        out["VERDICT"] = "NO-VERDICT-ANCHOR-DIVERGED"
    elif out["L1"] != "MACHINERY-CAN-BE-POSITIVE":
        out["VERDICT"] = "NO-VERDICT-MACHINERY"
    elif out["P1"] == "BUYS-WITH-SPARE-SLOTS":
        out["VERDICT"] = "BUYS-WITH-SPARE-SLOTS"
    elif out["S1"] == "MULTI-SEND-NON-VANISHING" and out["W1"] in (
            "WINDOW-SATURATES", "WINDOW-MIXED") and out["H1"] == "SCARCITY-SUBSTITUTION-BREAKS":
        out["VERDICT"] = "MULTI-READING-SERVES-SENDS-BUT-BELOW-ONE"
    elif out["S1"] == "MULTI-SEND-VANISHES" and out["W1"] == "WINDOW-SATURATES":
        out["VERDICT"] = "NO-MULTI-SEND-AMPLIFICATION-SCARCITY-BOUND"
    elif out["S1"] == "MULTI-SEND-VANISHES" and out["W1"] == "WINDOW-STILL-GROWS":
        out["VERDICT"] = "WINDOW-GROWS-BUT-SPARES-DONT-HELP"
    else:
        out["VERDICT"] = f"MIXED-S1-{out['S1']}-W1-{out['W1']}"
    return out


# ==================================================================== main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="CW-R4.2 一次读数服务几次发送")
    ap.add_argument("--ctrl-only", action="store_true", help="只跑闸（含 anchor41）")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    print(f"[{ROUND}] {PROTOCOL} start", flush=True)

    if args.ctrl_only:
        g = {"g_anchor41": gate_anchor41(), "g_first": gate_g_first(),
             "g_drop": gate_g_drop_obs(), "g_blackwell": gate_g_blackwell(),
             "g_no_truth": gate_g_no_truth(), "g_identity": gate_g_identity_files()}
        print(json.dumps({k: {"pass": v.get("pass"), "detail": {
            kk: vv for kk, vv in v.items() if kk in
            ("max_abs", "n_bad", "n_violation", "max_slack_used", "hashes", "dir_hash")}}
            for k, v in g.items()}, indent=2, ensure_ascii=False)[:6000], flush=True)
        print(f"[ctrl-only] n_fail={sum(1 for v in g.values() if not v['pass'])} / {len(g)}",
              flush=True)
        return 0

    p = panel()
    print(f"  panel done rows={len(p['rows'])}", flush=True)
    refine = refine_cells()

    gates = {"g_anchor41": gate_anchor41(),
             "g_paid_le_free": gate_paid_le_free(p["rows"]),
             "g_bound_a0": gate_bound_a0(p["rows"]),
             "g_probe_live": gate_probe_live(p["rows"]),
             "g_fp": gate_g_fp(p["meta"]),
             "g_first": gate_g_first(),
             "g_drop": gate_g_drop_obs(),
             "g_blackwell": gate_g_blackwell(),
             "g_no_truth": gate_g_no_truth(),
             "g_identity": gate_g_identity_files()}
    v = judge(p, refine, gates)

    out = {"round": ROUND, "protocol": PROTOCOL, "utc": _now_utc(),
           "sec_total": round(time.perf_counter() - t0, 3),
           "argv": sys.argv, "python": sys.executable, "cwd": str(Path.cwd()),
           "panel": {"rho": list(RHO_LIST), "alpha": ALPHA, "P": P_VALUE,
                     "kappa": KAPPA, "n_users": N_USERS,
                     "legs": [dict(l) for l in LEGS],
                     "refine_cells": [list(x) for x in REFINE_CELLS],
                     "refine_n_m": list(N_M_REFINE)},
           "panel_meta": p["meta"], "cells": p["rows"], "refine": refine,
           "gates": gates, "verdicts": v}

    print(json.dumps({k: v[k] for k in ("VERDICT", "W1", "W1_detail", "S1", "S1_detail",
                                        "H1", "P1", "P1_detail", "D1_detail", "L1",
                                        "n_cells_usable", "gates_n_fail")},
                     indent=2, ensure_ascii=False)[:16000], flush=True)

    dest = Path(args.out) if args.out else (
        RESULTS / f"cw_r4_2_multisend_{_now_utc()}_{_src_hash()[:16]}")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "r42.json").write_text(json.dumps(out, indent=2, ensure_ascii=False),
                                   encoding="utf-8")
    (dest / "commands.txt").write_text(
        f"cwd={Path.cwd()}\npython={sys.executable}\nargv={sys.argv}\n"
        f"utc={_now_utc()}\nsrc_hash={_src_hash()}\n", encoding="utf-8")
    print(f"[evidence] {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
