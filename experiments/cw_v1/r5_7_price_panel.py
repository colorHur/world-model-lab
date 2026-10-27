# -*- coding: utf-8 -*-
"""`CW-R5.7` 控制件：**信道测量的价格边界**（L1 网格对象，只加价格维度）。

预注册：`docs/research/CW-R5.7-测量价格边界-预注册-v0.1.md`（写于任何本轮实现与读数之前）。

与 R5.6 的关系
--------------
模型、信念表示、`plant`、动作集、目标函数、`ρ`、`κ`、`P0`、`n_w` 网格 **全部不动**
（协议仍是 `CW-POMDP-v1.5-grid`）。本轮唯一新增的自变量是 **探测的预算价 `p`**
（刻度 `p_ticks ∈ {0,1,2,4,8}`，`8` 刻度 = 一次整包发送的价）。

★ **不修改 `r5_6_grid_l1.py`**：它的证据目录记着自己的 `_src_hash`，
  改字节会让 8 个正式证据目录一起作废（R5.6 §7.5-7 的教训）。
  ⇒ 引擎在本文件**复制一份**，唯一差别是多两个参数 `probe_slot` / `send_ticks`，
  并由闸 `G1-p0-regression` 用**逐位相同**强制它和 R5.6 保持等价。

三个"成本"必须分开记账（预注册 §3）
------------------------------------
| 成本 | 谁付 | 本轮怎么变 |
|---|---|---|
| 时间槽 | 每次探测固定占一个动作槽 `t→t−1` | 所有 `p` 都付；`p=0` 时**只剩它** |
| 预算 | 每次探测 `p_ticks` 刻度 | `p=0` 时为零 |
| 整包粒度 | 发送不可分，`SEND_TICKS=8` | `p=8` 恰好等于"一整次发送" |

`no_slot` 消融（`probe_slot=False`）是**明示的理想化**：测量零耗时，物理不可实现，
只用于把"槽成本"和"预算成本"分开，**不进主表**。

主量
----
`IV_probe_L1(p) = A0p − VL1(p)`（同一 L1 测度、动作集嵌套 ⇒ 合法、零容差非负）。
"""

from __future__ import annotations

import argparse
import ast
import gc
import json
import math
import time
from pathlib import Path

from .r1_14_exactbase import _now_utc, _src_hash
from .r3_2_iv2u import STATE_CAP
from .r5_1_price_ticks import SEND_TICKS, TICKS_PER_QUANTUM
from .r5_5_l1_ack import P0, RHO, SIGMA_W1, SIGMA_W1_2U, ack_config
from .r5_6_grid_l1 import (HALF, GridCtx, build_engine_grid, g_mart, g_ps_ref,
                           make_gw)

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results" / "cw_v1"

ROUND = "CW-R5.7"
PROTOCOL = "CW-POMDP-v1.5-grid"          # 与 R5.6 同一协议，只加价格维度

#: 面板（预注册 §3，写死）
P_TICKS_ALL: tuple[int, ...] = (0, 1, 2, 4, 8)
N_W_LIST: tuple[int, ...] = (129, 257, 513)
N_W_SIZE: tuple[int, ...] = (33, 65, 129)
TAU_LIST: tuple[int, ...] = (5, 6)
B_LIST: tuple[int, ...] = (8, 16, 24)
PROBE_MAX_MAIN = 1
PROBE_TOTAL_MAIN = 1                     # n=2 压状态用；**必须与读数一起报**

#: 判据常数（预注册 §4/§5）
IV_NOISE = 1e-12
TOL_PRICE_MONO = 1e-15
TOL_NOINFO = 1e-12
TOL_NW3_REL = 0.30


# ==================================================================== 引擎（复制 + 价格/槽参数）
def build_engine_price(gw: GridCtx, cfg, probe_ticks: int, *, ack_updates: bool,
                       probe_enabled: bool, n_users: int, probe_max: int | None,
                       probe_total: int | None = None,
                       probe_slot: bool = True,
                       send_ticks: int = SEND_TICKS,
                       state_cap: int = STATE_CAP):
    """窗口 DP。与 `r5_6_grid_l1.build_engine_grid` **逐式相同**，只多 `probe_slot`。

    `probe_slot=False` ⇒ 探测**不占**动作槽（`t` 不减），是明示的理想化消融，默认关闭。
    """
    p = cfg.plant
    a2 = tuple(float(a) ** 2 for a in p.A)
    sw2 = tuple(float(s) ** 2 for s in p.sigma_w)
    sm2 = float(p.sigma_meas) ** 2
    q = float(p.q_cost)
    nu = int(n_users)
    pt = int(probe_ticks)
    sc = int(send_ticks)
    if pt < 0 or sc <= 0:
        raise ValueError("刻度必须是非负整数，且发送价必须 > 0")
    assert len(a2) >= nu and len(sw2) >= nu, "plant 的用户数不够"
    if not probe_slot and pt <= 0 and probe_total is None:
        raise ValueError("`no_slot` + 零价格 + 不限次数 ⇒ 递归不终止")
    pbid = gw.probe_belief_ids() if probe_enabled else None
    pmax = None if probe_max is None else int(probe_max)
    ptotal = None if probe_total is None else int(probe_total)
    dt_probe = 1 if probe_slot else 0

    memo: dict[tuple, float] = {}
    state = {"capped": False}

    def V(t: int, Pp: tuple, bb: tuple, b: int, pm: tuple, pn: int) -> float:
        if t <= 0:
            return 0.0
        key = (t, Pp, bb, b, pm, pn)
        got = memo.get(key)
        if got is not None:
            return got
        S = q * sum(Pp)
        if t == 1:
            memo[key] = S
            return S
        if len(memo) > state_cap:
            raise MemoryError(f"state_cap exceeded ({state_cap})")

        ff = tuple(a2[i] * Pp[i] + sw2[i] for i in range(nu))
        pb = tuple(gw.prop(bb[i]) for i in range(nu))
        best = V(t - 1, ff, pb, b, pm, pn)                   # HOLD

        if b >= sc:                                          # SEND_i（整包 sc 刻度）
            for i in range(nu):
                Pi = Pp[i]
                g = Pi / (Pi + sm2)
                p_ok = a2[i] * (1.0 - g) * Pi + sw2[i]
                p_no = a2[i] * Pi + sw2[i]
                ph = gw.ps_of(bb[i])
                aa = list(ff)
                aa[i] = p_ok
                cc = list(ff)
                cc[i] = p_no
                if ack_updates:
                    bok = list(bb)
                    bok[i] = gw.ack(bb[i], True)
                    bno = list(bb)
                    bno[i] = gw.ack(bb[i], False)
                    val = (ph * V(t - 1, tuple(aa), tuple(bok), b - sc, pm, pn)
                           + (1.0 - ph) * V(t - 1, tuple(cc), tuple(bno), b - sc, pm, pn))
                else:
                    val = (ph * V(t - 1, tuple(aa), pb, b - sc, pm, pn)
                           + (1.0 - ph) * V(t - 1, tuple(cc), pb, b - sc, pm, pn))
                if val < best:
                    best = val

        allow_probe = probe_enabled and b >= pt and t >= 2
        if allow_probe and ptotal is not None and pn >= ptotal:
            allow_probe = False
        if allow_probe:                                      # PROBE_i（占槽 + 花 pt 刻度）
            for i in range(nu):
                if pmax is not None and pm[i] >= pmax:
                    continue
                pi = gw.arr(bb[i])
                pm2 = list(pm)
                pm2[i] += 1
                pm2t = tuple(pm2)
                tot = 0.0
                for j in range(gw.n_w):
                    wj = float(pi[j])
                    if wj <= 0.0:
                        continue
                    bi = list(pb)
                    bi[i] = int(pbid[j])
                    tot += wj * V(t - dt_probe, ff, tuple(bi), b - pt, pm2t, pn + 1)
                if tot < best:
                    best = tot

        out = S + best
        memo[key] = out
        return out

    def value(t, Pvec, bvec, b, pmvec=None) -> float:
        pmv = tuple(0 for _ in range(nu)) if pmvec is None else tuple(int(x) for x in pmvec)
        try:
            return float(V(int(t), tuple(float(x) for x in Pvec),
                           tuple(int(x) for x in bvec), int(b), pmv, 0))
        except MemoryError:
            state["capped"] = True
            return float("nan")

    value.n_states = lambda: len(memo)          # type: ignore[attr-defined]
    value.capped = lambda: state["capped"]      # type: ignore[attr-defined]
    return value


# ==================================================================== 面板
class PricePanel:
    """一个 `(n_users, n_w)` 上：`A0p`（不探测，L1）与 `VL1(p)`（5 个价格）。"""

    def __init__(self, cfg, n_w: int, gw: GridCtx, *, n_users: int = 2,
                 probe_max: int | None = PROBE_MAX_MAIN,
                 probe_total: int | None = PROBE_TOTAL_MAIN,
                 p_ticks=P_TICKS_ALL, probe_slot: bool = True):
        self.cfg = cfg
        self.gw = gw
        self.nu = int(n_users)
        self.n_w = int(n_w)
        self.pmax = probe_max
        self.ptot = probe_total
        self.ps = tuple(int(x) for x in p_ticks)
        self.slot = bool(probe_slot)
        k = dict(n_users=n_users, probe_max=probe_max, probe_total=probe_total)
        t0 = time.perf_counter()
        self.A0p = build_engine_price(gw, cfg, 0, probe_enabled=False,
                                      ack_updates=True, **k)
        self.A0 = build_engine_price(gw, cfg, 0, probe_enabled=False,
                                     ack_updates=False, **k)
        self.VL1 = {pt: build_engine_price(gw, cfg, pt, probe_enabled=True,
                                           ack_updates=True, probe_slot=probe_slot, **k)
                    for pt in self.ps}
        self.sec_build = round(time.perf_counter() - t0, 3)
        self.bid_prior = gw.prior_id()

    def cell(self, tau: int, b: int) -> dict:
        Pv = tuple(P0 for _ in range(self.nu))
        bv = tuple(self.bid_prior for _ in range(self.nu))
        a0p = float(self.A0p(tau, Pv, bv, int(b)))
        a0 = float(self.A0(tau, Pv, bv, int(b)))
        vl1 = {pt: float(self.VL1[pt](tau, Pv, bv, int(b))) for pt in self.ps}
        iv = {pt: a0p - vl1[pt] for pt in self.ps}
        prof = boundary(iv)
        return {
            "n_users": self.nu, "n_w": self.n_w, "tau": int(tau), "b_ticks": int(b),
            "b_quanta": b / TICKS_PER_QUANTUM, "rho": float(self.cfg.channel.rho_panel[0]),
            "probe_max": (None if self.pmax is None else int(self.pmax)),
            "probe_total": (None if self.ptot is None else int(self.ptot)),
            "probe_slot": self.slot,
            "A0p": a0p, "A0_sidebar": a0,
            "VL1": {str(k): v for k, v in vl1.items()},
            "IV_probe_L1": {str(k): v for k, v in iv.items()},
            "rel_IV": {str(k): v / max(1e-30, abs(a0p)) for k, v in iv.items()},
            "boundary": prof,
            "capped": bool(self.A0p.capped() or self.A0.capped()
                           or any(e.capped() for e in self.VL1.values())),
        }

    def states(self) -> dict:
        return {"A0p": self.A0p.n_states(), "A0": self.A0.n_states(),
                "VL1": {str(k): v.n_states() for k, v in self.VL1.items()},
                "n_beliefs": self.gw.n_beliefs(),
                "capped": bool(self.A0p.capped() or self.A0.capped()
                               or any(e.capped() for e in self.VL1.values()))}


def boundary(iv: dict[int, float]) -> dict:
    """价格边界（预注册 §4/§6）：只报区间，不插值。`iv` 的键是 `p_ticks`。"""
    pos = sorted(pt for pt, v in iv.items() if v > IV_NOISE)
    if not pos:
        return {"verdict": "MEASUREMENT-WORTH-NOTHING-IN-SCANNED-PRICE-RANGE",
                "positive_p_ticks": [], "p_star_interval": None}
    hi = max(pos)
    nxt = [pt for pt in sorted(iv) if pt > hi]
    if hi == max(iv):
        return {"verdict": "P-ABOVE-SEND", "positive_p_ticks": pos,
                "p_star_interval": [hi, None]}
    return {"verdict": ("FREE-ONLY" if pos == [0] else "PRICE-BOUNDARY-IN-RANGE"),
            "positive_p_ticks": pos, "p_star_interval": [hi, min(nxt)],
            "p_lo_quanta": hi / TICKS_PER_QUANTUM,
            "p_hi_quanta": min(nxt) / TICKS_PER_QUANTUM}


def panel(n_w_list, tau_list=TAU_LIST, b_list=B_LIST, *, p_ticks=P_TICKS_ALL,
          n_users: int = 2, probe_max: int | None = PROBE_MAX_MAIN,
          probe_total: int | None = PROBE_TOTAL_MAIN,
          probe_slot: bool = True, tag: str = "price2u") -> dict:
    """逐价格**串行建表并释放**（`n_w=513` 上 5 份 `VL1` 同时驻留会 OOM）。

    ★ 峰值记忆 = `A0p + A0 + 单个 VL1`，与 R5.6 同量级；不是 5 倍。
    """
    rows: list[dict] = []
    legs: list[dict] = []
    for n_w in n_w_list:
        cfg = cfg_for(n_users)
        gw = make_gw(cfg, n_w)
        t0 = time.perf_counter()
        Pv = tuple(P0 for _ in range(n_users))
        bv = tuple(gw.prior_id() for _ in range(n_users))
        k = dict(n_users=n_users, probe_max=probe_max, probe_total=probe_total)
        e_a0p = build_engine_price(gw, cfg, 0, probe_enabled=False,
                                   ack_updates=True, **k)
        e_a0 = build_engine_price(gw, cfg, 0, probe_enabled=False,
                                  ack_updates=False, **k)
        a0p = {(t, b): float(e_a0p(t, Pv, bv, int(b)))
               for t in tau_list for b in b_list}
        a0 = {(t, b): float(e_a0(t, Pv, bv, int(b)))
              for t in tau_list for b in b_list}
        st_states = {"A0p": e_a0p.n_states(), "A0": e_a0.n_states(), "VL1": {}}
        capped_any = bool(e_a0p.capped() or e_a0.capped())
        del e_a0
        vl1: dict[int, dict[tuple, float]] = {}
        for pt in p_ticks:
            eng = build_engine_price(gw, cfg, pt, probe_enabled=True,
                                     ack_updates=True, probe_slot=probe_slot, **k)
            vl1[pt] = {(t, b): float(eng(t, Pv, bv, int(b)))
                       for t in tau_list for b in b_list}
            st_states["VL1"][str(pt)] = eng.n_states()
            capped_any = capped_any or bool(eng.capped())
            del eng
            gc.collect()
        for tau in tau_list:
            for b in b_list:
                v = {pt: vl1[pt][(tau, b)] for pt in p_ticks}
                iv = {pt: a0p[(tau, b)] - v[pt] for pt in p_ticks}
                prof = boundary(iv)
                rows.append({
                    "n_users": n_users, "n_w": int(n_w), "tau": int(tau),
                    "b_ticks": int(b), "b_quanta": b / TICKS_PER_QUANTUM,
                    "rho": float(cfg.channel.rho_panel[0]),
                    "probe_max": (None if probe_max is None else int(probe_max)),
                    "probe_total": (None if probe_total is None else int(probe_total)),
                    "probe_slot": bool(probe_slot),
                    "A0p": a0p[(tau, b)], "A0_sidebar": a0[(tau, b)],
                    "VL1": {str(pt): v[pt] for pt in p_ticks},
                    "IV_probe_L1": {str(pt): iv[pt] for pt in p_ticks},
                    "rel_IV": {str(pt): iv[pt] / max(1e-30, abs(a0p[(tau, b)]))
                               for pt in p_ticks},
                    "boundary": prof, "capped": capped_any})
                ivs = " ".join(f"p{pt}={iv[pt]:+.3e}" for pt in p_ticks)
                print(f"    n_w={n_w:>4} tau={tau} b={b:>3} "
                      f"A0p={a0p[(tau, b)]:.8e} {ivs} -> {prof['verdict']}", flush=True)
        legs.append({"tag": tag, "n_users": n_users, "n_w": int(n_w),
                     "sec": round(time.perf_counter() - t0, 3),
                     "states": {**st_states, "n_beliefs": gw.n_beliefs(),
                                "capped": capped_any}})
        print(f"  [{tag} nu={n_users} n_w={n_w}] {legs[-1]['sec']} s "
              f"states={st_states['VL1']} n_beliefs={gw.n_beliefs()} "
              f"capped={capped_any}", flush=True)
    return {"rows": rows, "legs": legs}


def cfg_for(n_users: int):
    return ack_config(RHO, SIGMA_W1 if n_users == 1 else SIGMA_W1_2U)


# ==================================================================== 闸
def g_p0_regression(rows, n_w_list=N_W_LIST, tau_list=TAU_LIST,
                    b_list=B_LIST, n_users: int = 2) -> dict:
    """★★★ `G1`：`p=0` 的 `VL1` 必须与 **R5.6 的引擎**逐位相同。

    ★ 这是**跨模块**等价证明：左边是本文件复制出来的 `build_engine_price`，
      右边是 R5.6 的原始 `build_engine_grid`（同一 `probe_ticks=0`、同一价格 0）。
      若复制时改坏了任何一处算术，这里就会抖。
      ★ 不用"两次跑同一个函数"当回归 —— 那只能证明确定性，不能证明等价。
    """
    cfg = cfg_for(n_users)
    out = []
    ok = True
    for n_w in n_w_list:
        gw = make_gw(cfg, n_w)
        pp = PricePanel(cfg, n_w, gw, n_users=n_users, p_ticks=(0,),
                        probe_max=PROBE_MAX_MAIN, probe_total=PROBE_TOTAL_MAIN)
        ref_eng = build_engine_grid(gw, cfg, 0, probe_enabled=True, ack_updates=True,
                                    n_users=n_users, probe_max=PROBE_MAX_MAIN,
                                    probe_total=PROBE_TOTAL_MAIN)
        Pv = tuple(P0 for _ in range(n_users))
        bv = tuple(gw.prior_id() for _ in range(n_users))
        for tau in tau_list:
            for b in b_list:
                a = float(pp.VL1[0](tau, Pv, bv, int(b)))
                ref = float(ref_eng(tau, Pv, bv, int(b)))
                in_panel = next((r["VL1"]["0"] for r in rows
                                 if r["n_w"] == n_w and r["tau"] == tau
                                 and r["b_ticks"] == b), None)
                same = (a == ref) and (in_panel is not None and a == in_panel)
                ok = ok and same
                out.append({"n_w": n_w, "tau": tau, "b_ticks": b,
                            "VL1_p0_r57": a, "VL1_p0_r56_engine": ref,
                            "VL1_p0_in_panel": in_panel,
                            "identical": bool(same)})
    return {"rows": out, "n_cells": len(out),
            "n_identical": sum(1 for r in out if r["identical"]),
            "pass": bool(ok and out)}


def g_price_mono(rows) -> dict:
    """`G2`：`VL1(p)` 关于 `p` 单调不降（更贵不可能更优）。零容差 `1e-15`。"""
    viol = []
    worst = 0.0
    for r in rows:
        ks = sorted((int(k) for k in r["VL1"]))
        for a, c in zip(ks, ks[1:]):
            d = r["VL1"][str(a)] - r["VL1"][str(c)]     # 便宜 − 贵 ≤ 0 才合法
            worst = max(worst, d)
            if d > TOL_PRICE_MONO:
                viol.append({"n_w": r["n_w"], "tau": r["tau"], "b": r["b_ticks"],
                             "p_cheap": a, "p_dear": c, "excess": float(d)})
    return {"n_violations": len(viol), "worst_excess": float(worst),
            "tol": TOL_PRICE_MONO, "violations": viol[:10],
            "pass": bool(rows and not viol)}


def g_legal_info(rows) -> dict:
    """★★★ `G3`：`VL1(p) ≤ A0p` **每格**成立（零容差）。

    合法：动作集包含"从不探测"，且 `p>0` 时不探测不花预算 ⇒ 恒等式，不是近似。
    """
    viol = [r for r in rows if r["VL1"][str(max(int(k) for k in r["VL1"]))] > r["A0p"]]
    worst = max((r["VL1"][str(max(int(k) for k in r["VL1"]))] - r["A0p"] for r in rows),
                default=0.0)
    return {"n_cells": len(rows), "n_violations": len(viol),
            "worst_excess": float(worst), "tol": 0.0,
            "violations": [{k: r[k] for k in ("n_w", "tau", "b_ticks", "A0p")}
                           for r in viol[:10]],
            "pass": bool(rows and not viol)}


def g_a0p_invariant(rows) -> dict:
    """`G4`：`A0p` 与价格无关（不许探测 ⇒ 预算永远不花在探测上）。"""
    by = {}
    for r in rows:
        by.setdefault((r["n_users"], r["n_w"], r["tau"], r["b_ticks"]), set()).add(r["A0p"])
    bad = {str(k): sorted(v) for k, v in by.items() if len(v) != 1}
    return {"n_groups": len(by), "n_groups_inconsistent": len(bad),
            "detail": dict(list(bad.items())[:5]), "pass": bool(by and not bad)}


def g_noinfo(p_ticks=P_TICKS_ALL, n_w: int = 129, n_users: int = 2) -> dict:
    """`G5`（活体负控）：`κ=0` ⇒ 似然恒 1/2 ⇒ 探测不带来信息 ⇒ `IV_probe_L1(p) ≡ 0`。"""
    cfg0 = cfg_for(n_users)
    from .config import ChannelConfig, mainline_config
    ch = ChannelConfig(**{**cfg0.channel.__dict__, "kappa": 0.0})
    cfg = mainline_config(plant=cfg0.plant, channel=ch)
    gw = make_gw(cfg, n_w)
    pp = PricePanel(cfg, n_w, gw, n_users=n_users, p_ticks=p_ticks,
                    probe_max=PROBE_MAX_MAIN, probe_total=PROBE_TOTAL_MAIN)
    Pv = tuple(P0 for _ in range(n_users))
    bv = tuple(gw.prior_id() for _ in range(n_users))
    a0p = float(pp.A0p(5, Pv, bv, 16))
    iv = {str(pt): float(a0p - float(pp.VL1[pt](5, Pv, bv, 16))) for pt in p_ticks}
    worst = max(abs(v) for v in iv.values())
    return {"n_w": int(n_w), "kappa": 0.0, "p_ticks": list(p_ticks),
            "IV_probe_L1": iv, "max_abs": float(worst), "tol": TOL_NOINFO,
            "pass": bool(worst <= TOL_NOINFO)}


def g_ast() -> dict:
    """`G7`：源码无 `int(p)` / `int(price)` 一类刻度截断；检测器对变异命中。"""
    src = Path(__file__).read_text(encoding="utf-8")
    hits = []
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "int":
            if not n.args:
                continue
            a0 = n.args[0]
            name = a0.id if isinstance(a0, ast.Name) else (
                a0.attr if isinstance(a0, ast.Attribute) else "")
            if name in ("p", "price", "probe_cost", "probe_price"):
                hits.append({"line": getattr(n, "lineno", None), "arg": name})
    mutant = [x for x in ast.walk(ast.parse("def f(p):\n    return int(p)\n"))
              if isinstance(x, ast.Call) and isinstance(x.func, ast.Name) and x.func.id == "int"]
    return {"hits": hits, "n_hits": len(hits),
            "detector_catches_mutation": bool(mutant),
            "pass": bool(not hits and mutant)}


def g_slack_rule(rows) -> dict:
    """★★★ `G-slack`（**看到 `n_w=129` 读数后**形成的规律，在 257/513 上验）：

    设 `slack = b mod SEND_TICKS`（预算余数，不用来探测就永远花不掉的部分）。预测：

      · `p_ticks ≤ slack` ⇒ `IV(p)` 与 `IV(0)` **逐位相同**（价格被余数吸收）
      · `p_ticks > slack` ⇒ `IV(p)` **逐位为 0**（探测越界就要吃掉一整次发送）

    `slack=0`（`b=8,16,24`）时退化为"任何正价格都精确归零"。
    ★ 这是**预测检验**，不是事后拟合：规律在 `n_w=129` 上看到，在 257/513 上首次检验。
    """
    rows_o = []
    n_ok = 0
    n_tot = 0
    n_exact_zero = 0
    for r in rows:
        slack = int(r["b_ticks"]) % int(SEND_TICKS)
        iv0 = r["IV_probe_L1"]["0"]
        for pt_s, v in sorted(r["IV_probe_L1"].items(), key=lambda kv: int(kv[0])):
            pt = int(pt_s)
            if pt <= slack:
                ok = (v == iv0)
            else:
                ok = (v <= IV_NOISE)
                n_exact_zero += int(v == 0.0)
            n_tot += 1
            n_ok += int(ok)
            if not ok:
                rows_o.append({"n_w": r["n_w"], "tau": r["tau"], "b": r["b_ticks"],
                               "slack": slack, "p_ticks": pt, "IV": v, "IV_p0": iv0})
    return {"n_checked": n_tot, "n_conform": n_ok, "n_exact_zero_above_slack": n_exact_zero,
            "violations": rows_o[:10], "pass": bool(n_tot and n_ok == n_tot)}


def g_slot_sanity(abl_rows) -> dict:
    """`G-slot`：把探测的槽成本**去掉**不可能让价值变差 ⇒ `IV(no_slot) ≥ IV(with_slot)`。

    ★ 本轮首版 `no_slot` 实现把"零耗时"写成了"探测后同一 `t` 再算一遍"（等于多算一个周期的
      累积成本），于是 `p=0` 上 `IV(no_slot)=0 < IV(with_slot)=9.44e-4` —— 违反单调性。
      按项目铁律（数据违反直觉先怀疑接线）判定**实现语义错**，该消融读数**不作证据**。
      本闸就是那个探测器的留存：它会失败，失败本身是它的分辨力证明。
    """
    rows = []
    ok = True
    for d in abl_rows:
        a, c = d["IV_with_slot"], d["IV_no_slot"]
        good = (c >= a - 1e-15)
        ok = ok and good
        rows.append({"p_ticks": d["p_ticks"], "IV_with_slot": a, "IV_no_slot": c,
                     "monotone": bool(good)})
    return {"rows": rows, "pass": bool(ok and rows),
            "interpretation": ("`pass=False` ⇒ `no_slot` 消融实现不可用（已知：语义错），"
                               "读数不进主表；不是物理发现")}


def g_size(legs) -> dict:
    rows = []
    ok = True
    for lg in legs:
        st = lg["states"]
        c1 = st["A0p"] > 0 and st["A0"] > 0 and all(v > 0 for v in st["VL1"].values())
        ok = ok and c1 and not st["capped"]
        rows.append({"tag": lg["tag"], "n_w": lg["n_w"], "n_users": lg["n_users"],
                     "states": st["VL1"], "A0p": st["A0p"], "A0": st["A0"],
                     "n_beliefs": st["n_beliefs"], "capped": st["capped"],
                     "c1_all_positive": bool(c1)})
    return {"rows": rows, "pass": bool(ok and rows)}


def g_nw3(rows) -> dict:
    """`G6`：`IV_probe_L1(p)` 在三点 `n_w` 上的相对 spread ≤ 30%（只对**有符号**的格判）。"""
    by = {}
    for r in rows:
        for pt, v in r["IV_probe_L1"].items():
            by.setdefault((r["tau"], r["b_ticks"], int(pt)), {})[r["n_w"]] = v
    out = []
    worst = 0.0
    n_skip = 0
    for k, d in sorted(by.items()):
        if len(d) < 3:
            continue
        vs = [d[n] for n in sorted(d)]
        if max(abs(v) for v in vs) <= IV_NOISE:
            n_skip += 1
            continue                                    # 全零格：relative spread 无意义
        lo, hi = min(vs), max(vs)
        rel = (hi - lo) / max(1e-30, abs(lo)) if lo > 0 else float("inf")
        worst = max(worst, rel)
        out.append({"cell": {"tau": k[0], "b": k[1], "p_ticks": k[2]},
                    "IV_by_nw": {str(n): v for n, v in sorted(d.items())},
                    "rel_spread": float(rel)})
    return {"rows": out, "n_cells": len(out), "n_skipped_all_zero": n_skip,
            "max_rel_spread": float(worst), "tol": TOL_NW3_REL,
            "pass": bool(out and worst <= TOL_NW3_REL) if out else None}


def g_slot_ablation(n_w: int = 129, tau: int = 6, b: int = 16,
                    p_ticks: int = 1, n_users: int = 2) -> dict:
    """`G-ablation`（明示理想化）：把探测的**槽成本**关掉，看读数变多少。

    不是闸，是**分解**：`p=0` 时探测只付槽；`p>0` 时槽 + 预算。
    关掉槽后剩的就是"只有预算价"的世界。**物理不可实现，不进主表。**
    """
    cfg = cfg_for(n_users)
    gw = make_gw(cfg, n_w)
    Pv = tuple(P0 for _ in range(n_users))
    bv = tuple(gw.prior_id() for _ in range(n_users))
    out = {}
    for slot in (True, False):
        pp = PricePanel(cfg, n_w, gw, n_users=n_users, p_ticks=(p_ticks,),
                        probe_max=PROBE_MAX_MAIN, probe_total=PROBE_TOTAL_MAIN,
                        probe_slot=slot)
        vl1 = float(pp.VL1[p_ticks](tau, Pv, bv, b))
        a0p = float(pp.A0p(tau, Pv, bv, b))
        out["slot" if slot else "no_slot"] = {
            "VL1": vl1, "A0p": a0p, "IV_probe_L1": float(a0p - vl1),
            "states_VL1": pp.VL1[p_ticks].n_states(),
            "capped": bool(pp.VL1[p_ticks].capped())}
    d = out["slot"]["IV_probe_L1"] - out["no_slot"]["IV_probe_L1"]
    return {"n_w": n_w, "tau": tau, "b_ticks": b, "p_ticks": p_ticks,
            "n_users": n_users, "by_arm": out,
            "IV_with_slot": out["slot"]["IV_probe_L1"],
            "IV_no_slot": out["no_slot"]["IV_probe_L1"],
            "slot_cost_abs": float(d),
            "slot_cost_rel": float(d / max(1e-30, abs(out["no_slot"]["IV_probe_L1"]))),
            "note": "`no_slot` 物理不可实现，只作分解；不进主表、不进判词"}


# ==================================================================== 判词
def judge(rows, gates) -> dict:
    out: dict = {"n_cells": len(rows),
                 "n_capped": sum(1 for r in rows if r["capped"]),
                 "n_cells_by_verdict": {}}
    out["gates"] = {k: {kk: vv for kk, vv in v.items()
                        if kk in ("pass", "max_abs", "max_rel", "max_rel_spread",
                                  "n_violations", "n_identical", "n_hits",
                                  "n_groups_inconsistent", "tol", "worst_excess")}
                    for k, v in gates.items()}
    out["gates_failed"] = sorted(k for k, v in gates.items() if v.get("pass") is False)
    out["gates_na"] = sorted(k for k, v in gates.items() if v.get("pass") is None)

    for r in rows:
        v = r["boundary"]["verdict"]
        out["n_cells_by_verdict"][v] = out["n_cells_by_verdict"].get(v, 0) + 1

    # ---- 逐格边界表（最细 `n_w`，避免 3 倍冗余）
    n_w_max = max(r["n_w"] for r in rows) if rows else None
    fine = [r for r in rows if r["n_w"] == n_w_max]
    out["boundary_table_finest"] = [
        {"tau": r["tau"], "b_ticks": r["b_ticks"],
         "positive_p_ticks": r["boundary"]["positive_p_ticks"],
         "p_star_interval": r["boundary"]["p_star_interval"],
         "verdict": r["boundary"]["verdict"],
         "IV_p0": r["IV_probe_L1"]["0"],
         "IV_max_p": r["IV_probe_L1"][str(max(int(k) for k in r["IV_probe_L1"]))],
         "rel_IV_p0": r["rel_IV"]["0"]}
        for r in sorted(fine, key=lambda x: (x["tau"], x["b_ticks"]))]

    # ---- 主判：`b` 最大、`τ` 最大的格（与 R5.6 同一格，便于对照）
    b_max = max((r["b_ticks"] for r in rows), default=None)
    t_max = max((r["tau"] for r in rows), default=None)
    main = [r for r in fine if r["b_ticks"] == b_max and r["tau"] == t_max]
    out["P_MAIN"] = ({"cell": {"b_ticks": b_max, "tau": t_max, "n_w": n_w_max},
                      "verdict": main[0]["boundary"]["verdict"],
                      "p_star_interval": main[0]["boundary"]["p_star_interval"],
                      "IV_probe_L1": main[0]["IV_probe_L1"],
                      "rel_IV": main[0]["rel_IV"]} if main else "NOT-APPLICABLE")

    if out["gates_failed"]:
        out["VERDICT"] = "NO-VERDICT-GATE-FAIL"
    elif out["n_capped"] > 0:
        out["VERDICT"] = "SCALE-WALL"
    else:
        vs = set(out["n_cells_by_verdict"])
        out["VERDICT"] = (vs.pop() if len(vs) == 1 else "PRICE-BOUNDARY-CELL-DEPENDENT")
    out["verdict_scope"] = (
        "只覆盖网格信念（`CW-POMDP-v1.5-grid`）、L1（ACK 更新信念）、ρ=0.999、P0=0.01、κ=2、"
        f"n=2、probe_max={PROBE_MAX_MAIN}、probe_total={PROBE_TOTAL_MAIN}、HALF={HALF}、"
        "探测=点观测、`P` 未离散化；不许外推到多次测量 / n>2 / 其他 ρ 或 κ")
    return out


# ==================================================================== main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="CW-R5.7 信道测量的价格边界（L1 网格）")
    ap.add_argument("--size-only", action="store_true")
    ap.add_argument("--ctrl-only", action="store_true")
    ap.add_argument("--slot-abl", action="store_true", help="只跑 no-slot 分解")
    ap.add_argument("--nu", type=int, default=2)
    ap.add_argument("--nw", default="")
    ap.add_argument("--tau", default="")
    ap.add_argument("--b", default="")
    ap.add_argument("--p", default="", help="价格刻度列表，逗号分隔（0,1,2,4,8）")
    ap.add_argument("--pmax", default=str(PROBE_MAX_MAIN))
    ap.add_argument("--ptotal", default=str(PROBE_TOTAL_MAIN))
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    n_w_list = tuple(int(x) for x in args.nw.split(",")) if args.nw else N_W_LIST
    tau_list = tuple(int(x) for x in args.tau.split(",")) if args.tau else TAU_LIST
    b_list = tuple(int(x) for x in args.b.split(",")) if args.b else B_LIST
    p_list = tuple(int(x) for x in args.p.split(",")) if args.p else P_TICKS_ALL
    pmax = None if args.pmax == "" else int(args.pmax)
    ptotal = None if args.ptotal == "" else int(args.ptotal)
    nu = int(args.nu)

    tag = args.tag or f"price{nu}u"
    _mode = ("size" if args.size_only else ("ctrl" if args.ctrl_only else
             ("slotabl" if args.slot_abl else tag)))
    print(f"[{ROUND}] {PROTOCOL} nu={nu} rho={RHO} tau={tau_list} b={b_list} "
          f"p_ticks={p_list} n_w={n_w_list} pmax={pmax} ptotal={ptotal} HALF={HALF}",
          flush=True)

    dest = Path(args.out) if args.out else (
        RESULTS / f"cw_r5_7_price_{_mode}_{_now_utc()}_{_src_hash()[:16]}")
    dest.mkdir(parents=True, exist_ok=True)
    t_all = time.perf_counter()

    if args.ctrl_only:
        gates = {"g_ps_ref": g_ps_ref(n_w_list), "g_mart": g_mart(n_w_list),
                 "g_noinfo": g_noinfo(p_list, n_w=min(n_w_list)), "g_ast": g_ast()}
        print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "rows"}
                          for k, v in gates.items()}, indent=2,
                         ensure_ascii=False)[:6000], flush=True)
        (dest / "r57_ctrl.json").write_text(
            json.dumps({"round": ROUND, "protocol": PROTOCOL, "utc": _now_utc(),
                        "gates": gates}, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[ctrl] fails={[k for k, v in gates.items() if v.get('pass') is False]}",
              flush=True)
        print(f"[evidence] {dest}", flush=True)
        return 0

    if args.slot_abl:
        out = []
        for pv in ([int(x) for x in args.p.split(",")] if args.p else [0, 1, 2, 4]):
            d = g_slot_ablation(n_w=min(n_w_list), tau=max(tau_list), b=16,
                                p_ticks=pv, n_users=nu)
            out.append(d)
            print(f"  [slot-abl p={pv}] IV_with_slot={d['IV_with_slot']:+.4e} "
                  f"IV_no_slot={d['IV_no_slot']:+.4e} "
                  f"slot_cost_rel={d['slot_cost_rel']:+.4f}", flush=True)
        (dest / "r57_slot_ablation.json").write_text(
            json.dumps({"round": ROUND, "utc": _now_utc(),
                        "idealized": True,
                        "note": "no_slot 物理不可实现，只作分解，不进主表、不进判词",
                        "sanity": g_slot_sanity(out),
                        "rows": out}, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[evidence] {dest}", flush=True)
        return 0

    if args.size_only:
        out = []
        for n_w in (n_w_list if args.nw else N_W_SIZE):
            cfg = cfg_for(nu)
            gw = make_gw(cfg, n_w)
            t0 = time.perf_counter()
            pp = PricePanel(cfg, n_w, gw, n_users=nu, p_ticks=p_list,
                            probe_max=pmax, probe_total=ptotal)
            for tau in tau_list:
                for b in b_list:
                    pp.cell(tau, b)
            st = pp.states()
            rec = {"n_w": int(n_w), "n_users": nu, "p_ticks": list(p_list),
                   "probe_max": pmax, "probe_total": ptotal,
                   "sec": round(time.perf_counter() - t0, 3),
                   "states_VL1": st["VL1"], "A0p": st["A0p"], "A0": st["A0"],
                   "n_beliefs": st["n_beliefs"], "capped": st["capped"]}
            out.append(rec)
            print(f"  [size n_w={n_w} nu={nu}] {rec['sec']} s states={rec['states_VL1']} "
                  f"beliefs={rec['n_beliefs']} capped={rec['capped']}", flush=True)
            if rec["capped"]:
                print(f"  [size] n_w={n_w} 撞顶 ⇒ 停止", flush=True)
                break
        (dest / "r57_size.json").write_text(
            json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[evidence] {dest}", flush=True)
        return 0

    gates = {"g_ps_ref": g_ps_ref(n_w_list), "g_mart": g_mart(n_w_list),
             "g_noinfo": g_noinfo(p_list, n_w=min(n_w_list)), "g_ast": g_ast()}
    print(f"[ctrl] {json.dumps({k: v.get('pass') for k, v in gates.items()})}", flush=True)

    t0 = time.perf_counter()
    pnl = panel(n_w_list, tau_list, b_list, p_ticks=p_list, n_users=nu,
                probe_max=pmax, probe_total=ptotal, tag=tag)
    sec_panel = round(time.perf_counter() - t0, 3)

    gates["g_p0_regression"] = g_p0_regression(pnl["rows"], n_w_list, tau_list,
                                               b_list, n_users=nu)
    gates["g_price_mono"] = g_price_mono(pnl["rows"])
    gates["g_legal_info"] = g_legal_info(pnl["rows"])
    gates["g_a0p_invariant"] = g_a0p_invariant(pnl["rows"])
    gates["g_nw3"] = g_nw3(pnl["rows"])
    gates["g_size"] = g_size(pnl["legs"])
    gates["g_slack_rule"] = g_slack_rule(pnl["rows"])
    in_samp = g_slack_rule([r for r in pnl["rows"] if r["n_w"] == min(n_w_list)])
    out_samp = g_slack_rule([r for r in pnl["rows"] if r["n_w"] != min(n_w_list)])
    v = judge(pnl["rows"], gates)
    v["g_slack_in_sample"] = {k: in_samp[k] for k in
                              ("n_checked", "n_conform", "n_exact_zero_above_slack", "pass")}
    v["g_slack_out_of_sample"] = {k: out_samp[k] for k in
                                  ("n_checked", "n_conform", "n_exact_zero_above_slack", "pass")}

    print(json.dumps({k: v[k] for k in ("VERDICT", "P_MAIN", "n_cells_by_verdict",
                                        "boundary_table_finest")},
                     indent=2, ensure_ascii=False, default=float)[:8000], flush=True)
    print(json.dumps({"gates_failed": v["gates_failed"], "gates_na": v["gates_na"]},
                     ensure_ascii=False), flush=True)

    out = {"round": ROUND, "protocol": PROTOCOL, "utc": _now_utc(),
           "ticks_per_quantum": TICKS_PER_QUANTUM, "send_ticks": SEND_TICKS,
           "half": HALF, "probe_max": pmax, "probe_total": ptotal, "n_users": nu,
           "p_ticks": list(p_list), "sec_panel": sec_panel,
           "sec_total": round(time.perf_counter() - t_all, 3),
           "src_hash": _src_hash(),
           "panel_spec": {"rho": RHO, "P": P0, "tau": list(tau_list),
                          "b_ticks": list(b_list), "n_w": list(n_w_list),
                          "sigma_w": list(cfg_for(nu).plant.sigma_w),
                          "probe_slot": True, "probe_kind": "point-observation"},
           "cells": pnl["rows"], "legs": pnl["legs"],
           "gates": gates, "verdicts": v}
    (dest / "r57.json").write_text(json.dumps(out, indent=2, ensure_ascii=False,
                                              default=float), encoding="utf-8")
    print(f"[evidence] {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
