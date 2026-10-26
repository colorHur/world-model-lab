# -*- coding: utf-8 -*-
"""`CW-R5.2` 三用户 `b=16`（2 发送 < 3 用户）与「收件人选择自由度」的控制件。

预注册：`docs/research/CW-R5.2-三用户与收件人选择-预注册-v0.1.md`（写于任何面板读数之前；
§5 的规模规则在读数前执行一次，选中值记进该节"读面板前选定"）。

本文件做什么
------------
只做一件事：把 R5.1 的**同一条记账协议**（`CW-POMDP-v1.3-ticks`）从 2 用户搬到 3 用户，
在**发送次数完全相同**（`b=16` = 2 次发送）的条件下问：读数的价值会不会从 0 翻成正。

为什么这样问（§0/§1）
----------------------
R5.1 已落盘的读数里，2 用户 `τ=4, P=0.01, α=2.5`：
`b=8`（1 发送 < 2 用户）`p=0` ⇒ `IV=2.544453e-4 > 0`；`b=16`（2 发送 = 2 用户）`p=0` ⇒ `IV=0`。
⇒ 零不是"服务覆盖满"独有的新鲜事，正确形态是 `发送次数/用户数` 的比值：`<1` 有值、`=1` 为零。
本轮加一个比值点 `2/3`，并给镜像点 `3/3`。

纪律
----
1. 零 episode、零训练、零参数更新；`config.py` / `posterior.py` / `r3_2_iv2u.py` /
   `r4_0_boundary.py` / `r5_1_price_ticks.py` **一个字符不改**（只 `import`）。
2. 本文件的引擎是 R5.1 引擎的**带变异副本**（多一个 `drop_obs`）⇒ 必须先用 `G1b`
   证明 `drop_obs=False` 时与 R5.1 引擎**逐位**相同，再对任何读数说话。
3. 判据 `P-A..P-F` / 闸 `G1..G9` 见预注册 §3/§6，**跑前写死**。
4. `IV` 只报符号与区间；`n_m` 必须三点起（R4.1 教训）。
5. 2 用户与 3 用户的目标函数是 `q·ΣP_i`，用户数不同 ⇒ **跨实例，不许相减**。
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

from .config import PlantConfig, bounded_plant, mainline_config
from .posterior import p_succ_expected
from .r1_14_exactbase import _now_utc, _src_hash
from .r3_2_iv2u import GH_N, M_RANGE, STATE_CAP
from .r5_1_price_ticks import (
    PRICE_TICKS,
    SEND_TICKS,
    TICKS_PER_QUANTUM,
    build_engine_ticks as r51_engine,
)

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results" / "cw_v1"

PROTOCOL = "CW-POMDP-v1.3-ticks"          # 沿用 R5.1，不新立变体
ROUND = "CW-R5.2"

#: 面板常数（预注册 §2，跑前写死）
ALPHA = 2.5
P0 = 0.01
KAPPA_P = 1.0
TAU = 4
GH = GH_N
SIGMA_STAR = 1.0e-9
SIGMA_NULL = 1.0e6
SIGMA_W0 = 0.02
V_PRIOR = 1.0

#: 价格三点（探测刻度 8p）：1、1/4、0 ⇒ (8, 2, 0)
PRICE_PTS: tuple[int, ...] = (8, 2, 0)

#: 预算（刻度）
B_MAIN = 16        # 2 量子 = 2 次发送（2 用户 ⇒ 2/2；3 用户 ⇒ 2/3）
B_MIRROR = 24      # 3 量子 = 3 次发送（3 用户 ⇒ 3/3）
B_SCARCE = 8       # 1 量子 = 1 次发送（2 用户 ⇒ 1/2，R5.1 的稀缺锚）

#: `n_m` 四点：由 §5 规模规则定（`--size-only` 实测后填；见预注册 §9 的跑前修订 1）。
#: 实测：3 用户 `b=16` 整组在 `n_m=257` 上只有 70,290 状态 / 0.26 s ⇒ 3e6 的余量根本没用上，
#: 于是按「覆盖 R4.1 的 65/129 分辨率对 + 两端各再拉一格」取四点，而不是取"能算的最大三点"。
N_M_MAIN: tuple[int, ...] = (33, 65, 129, 257)
N_M_ONE = 129      # 只给 `G4/G5/G6/G7b` 这类单点闸用

#: 判据常数
IV_NOISE = 1e-12
TOL_BIT = 0.0
TOL_MONO = 1e-12
TOL_NOINFO = 1e-9
#: R5.1 已落盘的 2 用户锚（逐位对账用；`p=1` 与 `p=0` 都是这一对）
R51_ANCHOR = {
    "A0_b16_tau4_P001": 9.142381e-02,
    "IV_b16_tau4_P001_all_p": 0.0,
    "IV_b8_p0": 2.544453e-04,
    "A0_b8_tau4_P001": 9.440000e-02,
}

_SQRT2PI = math.sqrt(2.0 * math.pi)


# ==================================================================== plant
def plant_for(n_users: int, mode: str = "HH") -> PlantConfig:
    """按预注册 §2 造 plant。`A≡1`、`B/σ_meas/q` 全部沿用 `bounded_plant()`。

    · `n_users=2`：`sigma_w=(0.02, 0.02*alpha)` —— **与 R5.1 的 `_plant_asym` 同式**（G1a 锚）
    · `n_users=3, mode="HH"`：`(0.02, 0.05, 0.05)` —— 第 3 个是**第二个高噪用户**
      ⇒ 用户 2 与 3 在 `t=0` 完全并列，收件人决定权只能来自信道读数（§2.3）
    · `n_users=3, mode="HL"`：`(0.02, 0.05, 0.02)` —— 构成臂，回答"是不是换了哪个人"
    """
    if n_users == 2:
        sw = (SIGMA_W0, SIGMA_W0 * ALPHA)
    elif n_users == 3 and mode == "HH":
        sw = (SIGMA_W0, SIGMA_W0 * ALPHA, SIGMA_W0 * ALPHA)
    elif n_users == 3 and mode == "HL":
        sw = (SIGMA_W0, SIGMA_W0 * ALPHA, SIGMA_W0)
    else:
        raise ValueError(f"未登记的 (n_users={n_users}, mode={mode})")
    return bounded_plant(A=tuple(1.0 for _ in range(n_users)), sigma_w=sw)


def cfg_for(n_users: int, mode: str = "HH"):
    return mainline_config(plant=plant_for(n_users, mode))


# ==================================================================== 引擎（R5.1 的带变异副本）
def build_engine(cfg, sigma_csi: float, n_m: int, probe_ticks: int, *,
                 send_ticks: int = SEND_TICKS, gh_n: int = GH,
                 probe_enabled: bool = True, n_users: int = 2,
                 drop_obs: bool = False, state_cap: int = STATE_CAP):
    """与 `r5_1_price_ticks.build_engine_ticks` **逐字同算术**，只多一个 `drop_obs`。

    `drop_obs=True` 是**变异**：探测读数被丢弃（`m' ← rho*snap(m)`，没有 `gp*sy*z` 项），
    其余（价格、预算、网格、GH 节点、动作集）一律不动 ⇒ 用于 `G5`：
    「若 IV 不依赖读到的值，那 IV 不是读数带来的」。

    ★ 副本本身是被测物 ⇒ `G1b` 必须先证明 `drop_obs=False` 与 R5.1 引擎**逐位**相同。
    """
    p = cfg.plant
    a2 = tuple(float(a) ** 2 for a in p.A)
    sw2 = tuple(float(s) ** 2 for s in p.sigma_w)
    sm2 = float(p.sigma_meas) ** 2
    q = float(p.q_cost)
    rho = float(cfg.channel.rho_panel[0])
    rho2 = rho * rho
    infl = 1.0 - rho2
    si2 = float(sigma_csi) ** 2
    nodes, weights = np.polynomial.hermite_e.hermegauss(int(gh_n))
    nodes = [float(z) for z in nodes]
    weights = [float(w) for w in weights]
    centers = np.linspace(-M_RANGE, M_RANGE, int(n_m))
    centers_l = [float(x) for x in centers]
    pt = int(probe_ticks)
    sc = int(send_ticks)
    nu = int(n_users)
    if pt < 0 or sc <= 0:
        raise ValueError("刻度必须是非负整数，且发送价必须 > 0")
    assert len(a2) >= nu and len(sw2) >= nu, "plant 的用户数不够"

    @lru_cache(maxsize=None)
    def snap(m: float) -> float:
        return centers_l[int(np.argmin(np.abs(centers - float(m))))]

    @lru_cache(maxsize=None)
    def ps(m: float, v: float) -> float:
        return p_succ_expected(float(m), float(v), cfg)

    memo: dict[tuple, float] = {}
    state = {"capped": False}

    def V(t: int, Pp: tuple, mm: tuple, vv: tuple, b: int) -> float:
        if t <= 0:
            return 0.0
        key = (t, Pp, mm, vv, b)
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
        mp = tuple(rho * m for m in mm)
        vp0 = tuple(rho2 * v + infl for v in vv)

        best = V(t - 1, ff, mp, vp0, b)                    # HOLD

        if b >= sc:                                        # SEND_i（整包，sc 刻度）
            for i in range(nu):
                Pi = Pp[i]
                g = Pi / (Pi + sm2)
                ps_ = a2[i] * (1.0 - g) * Pi + sw2[i]
                pf_ = a2[i] * Pi + sw2[i]
                ph = ps(mm[i], vv[i])
                aa = list(ff)
                aa[i] = ps_
                bb = list(ff)
                bb[i] = pf_
                va = ph * V(t - 1, tuple(aa), mp, vp0, b - sc)
                vb = (1.0 - ph) * V(t - 1, tuple(bb), mp, vp0, b - sc)
                val = va + vb
                if val < best:
                    best = val

        if probe_enabled and b >= pt and t >= 2:           # PROBE_i（pt 刻度）
            for i in range(nu):
                mi = mm[i]
                vi = vv[i]
                gp = vi / (vi + si2)
                vpi = rho2 * (vi * si2 / (vi + si2)) + infl
                sy = math.sqrt(vi + si2)
                tot = 0.0
                for z, w in zip(nodes, weights):
                    mm2 = list(mp)
                    mm2[i] = rho * snap(mi) if drop_obs else rho * snap(mi + gp * sy * z)
                    vv2 = list(vp0)
                    vv2[i] = vpi
                    tot += w * V(t - 1, ff, tuple(mm2), tuple(vv2), b - pt)
                tot /= _SQRT2PI
                if tot < best:
                    best = tot

        out = S + best
        memo[key] = out
        return out

    def value(t, Pvec, mvec, vvec, b) -> float:
        try:
            return float(V(int(t), tuple(float(x) for x in Pvec),
                           tuple(float(x) for x in mvec),
                           tuple(float(x) for x in vvec), int(b)))
        except MemoryError:
            state["capped"] = True
            return float("nan")

    value.n_states = lambda: len(memo)          # type: ignore[attr-defined]
    value.capped = lambda: state["capped"]      # type: ignore[attr-defined]
    value.probe_ticks = pt                      # type: ignore[attr-defined]
    value.send_ticks = sc                       # type: ignore[attr-defined]
    return value


# ==================================================================== 组（整腿）
class Group:
    """`(n_users, mode, n_m)` 一组 = 一条 `noprobe` 引擎 + 每价格一条 `paid` 引擎。

    ★ 规模在**该组全部格求值之后**取（`memo` 是整组并集，不是单格）。
    """

    def __init__(self, n_users: int, mode: str, n_m: int):
        self.n_users = int(n_users)
        self.mode = str(mode)
        self.n_m = int(n_m)
        self.cfg = cfg_for(self.n_users, self.mode)
        t0 = time.perf_counter()
        self.no = build_engine(self.cfg, SIGMA_STAR, self.n_m, 0,
                               probe_enabled=False, n_users=self.n_users)
        self.sec_no = round(time.perf_counter() - t0, 3)
        self.paid: dict[int, object] = {}
        self._first = None

    def _P(self) -> tuple:
        return tuple(P0 for _ in range(self.n_users))

    def paid_engine(self, pt: int):
        if pt not in self.paid:
            t0 = time.perf_counter()
            e = build_engine(self.cfg, SIGMA_STAR, self.n_m, pt, n_users=self.n_users)
            self.paid[pt] = e
            self._sec_build = {**getattr(self, "_sec_build", {}), pt: round(time.perf_counter() - t0, 3)}
        return self.paid[pt]

    def run(self, cells: list[dict]) -> list[dict]:
        out = []
        zero = tuple(0.0 for _ in range(self.n_users))
        vp0 = tuple(V_PRIOR for _ in range(self.n_users))
        for cell in cells:
            tau, b = int(cell["tau"]), int(cell["b_ticks"])
            pt = int(cell["probe_ticks"])
            eng = self.paid_engine(pt)
            a0 = self.no(tau, self._P(), zero, vp0, b)
            a0m = (self.no(tau, self._P(), zero, vp0, b - SEND_TICKS)
                   if b >= SEND_TICKS else float("nan"))
            vp = eng(tau, self._P(), zero, vp0, b)
            if self._first is None:
                self._first = eng.n_states()
            out.append({
                "n_users": self.n_users, "mode": self.mode, "n_m": self.n_m,
                "tau": tau, "b_ticks": b, "b_quanta": b / TICKS_PER_QUANTUM,
                "p": pt / TICKS_PER_QUANTUM, "probe_ticks": pt,
                "P": P0, "kappa": KAPPA_P, "alpha": ALPHA,
                "A0": float(a0), "V_paid": float(vp), "IV_paid": float(a0 - vp),
                "dV_send": float(a0m - a0) if math.isfinite(a0m) else float("nan"),
                "capped": bool(self.no.capped() or eng.capped()),
            })
        return out

    def states(self) -> dict:
        return {"noprobe": self.no.n_states(),
                "paid": {str(k): v.n_states() for k, v in self.paid.items()},
                "n_after_first_cell": self._first,
                "capped": bool(self.no.capped() or any(v.capped() for v in self.paid.values()))}


# ==================================================================== 面板
def panel(n_m_list: tuple[int, ...], n_m_one: int) -> dict:
    rows: list[dict] = []
    meta: list[dict] = []

    def do(g: Group, cells: list[dict], tag: str) -> None:
        t0 = time.perf_counter()
        rr = g.run(cells)
        for r in rr:
            r["arm"] = tag
        st = g.states()
        rows.extend(rr)
        meta.append({"arm": tag, "n_users": g.n_users, "mode": g.mode, "n_m": g.n_m,
                     "sec": round(time.perf_counter() - t0, 3), "sec_no": g.sec_no,
                     "sec_build": getattr(g, "_sec_build", {}), "states": st,
                     "max_IV": float(max((r["IV_paid"] for r in rr), default=float("nan")))})
        print(f"  [{tag} nu={g.n_users} n_m={g.n_m}] {meta[-1]['sec']} s "
              f"states={st['paid']} maxIV={meta[-1]['max_IV']:.6e} capped={st['capped']}",
              flush=True)

    # ---- 主面板：2 用户对照 + 3 用户主件，b=16，三价格，四点 n_m
    for n_m in n_m_list:
        cells_main = [{"tau": TAU, "b_ticks": B_MAIN, "probe_ticks": pt} for pt in PRICE_PTS]
        do(Group(2, "HH", n_m), cells_main, "ctrl2u_b16")
        do(Group(3, "HH", n_m), cells_main, "main3u_b16")
        # 稀缺锚：2 用户 b=8（1 发送 < 2 用户），R5.1 的 2.544453e-4 多点复算
        do(Group(2, "HH", n_m), [{"tau": TAU, "b_ticks": B_SCARCE, "probe_ticks": 0}],
           "anchor2u_b8_p0")
        # ★ 3 用户稀缺锚（**活体对照**，跑中追加并记入预注册 §9-3）：
        #   若 3 用户 `b=16` 为零，必须先证明"3 用户这套引擎能出现非零"，
        #   否则那个零可能只是"3 用户分支坏了"。`b=8`（1 发送 / 3 用户）就是这个活体。
        do(Group(3, "HH", n_m), [{"tau": TAU, "b_ticks": B_SCARCE, "probe_ticks": 0}],
           "anchor3u_b8_p0")
        # 镜像臂：3 发送 = 3 用户（3/3）与 3 发送 > 2 用户
        do(Group(3, "HH", n_m),
           [{"tau": TAU, "b_ticks": B_MIRROR, "probe_ticks": pt} for pt in (0, 2)],
           "mirror3u_b24")
        do(Group(2, "HH", n_m),
           [{"tau": TAU, "b_ticks": B_MIRROR, "probe_ticks": 0}],
           "mirror2u_b24")
        # 构成臂：第 3 个用户取低噪（`(0.02,0.05,0.02)`），探索，不进主判据
        do(Group(3, "HL", n_m),
           [{"tau": TAU, "b_ticks": B_MAIN, "probe_ticks": pt} for pt in (0, 2)],
           "comp3u_b16_HL")

    return {"rows": rows, "legs": meta}


# ==================================================================== 闸
def _r51_evidence() -> Path:
    """找 R5.1 的**正式**（非 `_SUPERSEDED`）证据 JSON。

    ★ 锚值必须从产物取，不许手抄（STATE §7.5-6）。首版把 `9.142381e-02` 写成常量，它是
    JSON 里打印的 7 位有效数字，与实际值 `0.09142380952380953` 差 `4.76e-10`
    ⇒ 拿它做"逐位"锚会必然 FAIL（这是**我自己的**锚错，不是引擎错）。
    """
    cands = [p for p in sorted(RESULTS.glob("cw_r5_1_price_ticks_*/r51.json"))
             if "SUPERSEDED" not in str(p)]
    if not cands:
        raise FileNotFoundError("找不到 R5.1 正式证据 r51.json")
    return cands[-1]


def g1a_anchor_vs_r51() -> dict:
    """`G1a`：2 用户 `n_m=17` 必须复现 R5.1 的 `A0` 与 `IV_paid`（**逐位**，取自产物 JSON）。"""
    ref_path = _r51_evidence()
    ref = json.loads(ref_path.read_text(encoding="utf-8"))
    ref_cells = [r for r in ref["cells"]
                 if abs(r["alpha"] - 2.5) < 1e-12 and r["tau"] == TAU and r["P"] == P0]
    want = {}
    for r in ref_cells:
        if r["b_ticks"] in (B_MAIN, B_SCARCE):
            want[(r["b_ticks"], r["probe_ticks"])] = r
    g = Group(2, "HH", 17)
    cells = [{"tau": TAU, "b_ticks": b, "probe_ticks": pt}
             for b, pt in sorted(want)]
    mine = {(r["b_ticks"], r["probe_ticks"]): r for r in g.run(cells)}
    rows = []
    worst = 0.0
    for key, w in sorted(want.items()):
        m = mine[key]
        d_a0 = abs(m["A0"] - w["A0"])
        d_iv = abs(m["IV_paid"] - w["IV_paid"])
        worst = max(worst, d_a0, d_iv)
        rows.append({"b_ticks": key[0], "probe_ticks": key[1],
                     "A0_mine": float(m["A0"]), "A0_ref": float(w["A0"]),
                     "IV_mine": float(m["IV_paid"]), "IV_ref": float(w["IV_paid"]),
                     "abs_dA0": float(d_a0), "abs_dIV": float(d_iv)})
    return {"ref_file": str(ref_path.relative_to(REPO)).replace("\\", "/"),
            "rows": rows, "n_cells": len(rows), "max_abs": float(worst),
            "tol": TOL_BIT, "pass": bool(rows and worst <= TOL_BIT)}


def g1b_copy_identity() -> dict:
    """`G1b`：本文件的引擎副本（`drop_obs=False`）必须与 R5.1 引擎**逐位**相同。

    若这条不过，本文件所有读数都不是 R5.1 那条协议下的读数。
    """
    rows = []
    worst = 0.0
    cases = [(2, "HH", 17, 8, 16), (2, "HH", 17, 0, 8), (3, "HH", 9, 2, 16),
             (3, "HH", 9, 0, 16), (3, "HL", 9, 0, 16), (2, "HH", 9, 2, 24)]
    for nu, mode, n_m, pt, b in cases:
        cfg = cfg_for(nu, mode)
        mine = build_engine(cfg, SIGMA_STAR, n_m, pt, n_users=nu)
        ref = r51_engine(cfg, SIGMA_STAR, n_m, pt, n_users=nu)
        Pv = tuple(P0 for _ in range(nu))
        zv = tuple(0.0 for _ in range(nu))
        vv = tuple(V_PRIOR for _ in range(nu))
        a = mine(TAU, Pv, zv, vv, b)
        c = ref(TAU, Pv, zv, vv, b)
        d = abs(a - c)
        worst = max(worst, d)
        rows.append({"n_users": nu, "mode": mode, "n_m": n_m, "probe_ticks": pt,
                     "b_ticks": b, "mine": float(a), "ref": float(c), "abs": float(d)})
    return {"rows": rows, "max_abs": float(worst), "tol": TOL_BIT, "pass": bool(worst <= TOL_BIT)}


def g2_no_leak() -> dict:
    """`G2`：3 用户 plant 上取 `n_users=2` ≡ 2 用户 plant 上 `n_users=2`（第 3 条 `sigma_w` 不泄漏）。"""
    rows = []
    worst = 0.0
    for pt in (8, 2, 0):
        c2 = cfg_for(2, "HH")
        c3 = cfg_for(3, "HH")
        e2 = build_engine(c2, SIGMA_STAR, 9, pt, n_users=2)
        e3 = build_engine(c3, SIGMA_STAR, 9, pt, n_users=2)
        a = e2(TAU, (P0, P0), (0.0, 0.0), (V_PRIOR, V_PRIOR), B_MAIN)
        c = e3(TAU, (P0, P0), (0.0, 0.0), (V_PRIOR, V_PRIOR), B_MAIN)
        d = abs(a - c)
        worst = max(worst, d)
        rows.append({"probe_ticks": pt, "two_plant": float(a), "three_plant_nu2": float(c),
                     "abs": float(d)})
    return {"rows": rows, "max_abs": float(worst), "tol": TOL_BIT, "pass": bool(worst <= TOL_BIT)}


def g3_monotone(rows: list) -> dict:
    """`G3`（`P-E`）：固定 `(arm, n_m, b)`，`IV_paid` 随价格下降**不减**。

    ★ 方向先推：`p↓ ⇒ 8p↓ ⇒ {b ≥ 8p}` 变大 ⇒ 可行集嵌套 ⇒ `V` 不增 ⇒ `IV = A0 − V` **不减**。
    """
    from collections import defaultdict
    grp: dict[tuple, list] = defaultdict(list)
    for r in rows:
        if math.isfinite(r["IV_paid"]):
            grp[(r["arm"], r["n_m"], r["b_ticks"])].append((r["probe_ticks"], r["IV_paid"]))
    viol = []
    worst = 0.0
    n_pairs = 0
    for key, vals in grp.items():
        vals.sort(key=lambda x: -x[0])          # 价格下降方向
        for (p_hi, v_hi), (p_lo, v_lo) in zip(vals, vals[1:]):
            n_pairs += 1
            d = v_hi - v_lo                     # 必须 ≤ 0
            worst = max(worst, d)
            if d > TOL_MONO:
                viol.append({"arm": key[0], "n_m": key[1], "b_ticks": key[2],
                             "p_hi_ticks": p_hi, "p_lo_ticks": p_lo,
                             "IV_hi": v_hi, "IV_lo": v_lo, "violation": float(d)})
    return {"n_groups": len(grp), "n_pairs": n_pairs, "n_violations": len(viol),
            "worst_positive_gap": float(worst), "tol": TOL_MONO,
            "violations": viol[:12], "pass": bool(not viol and n_pairs > 0)}


def g4_no_info(n_m: int) -> dict:
    """`G4`：`σ_csi=1e6`（读数几乎无信息）⇒ `0 ≥ IV ≥ −1e-12` 且 `≤ 1e-9`。"""
    rows = []
    for nu in (2, 3):
        cfg = cfg_for(nu, "HH")
        no = build_engine(cfg, SIGMA_NULL, n_m, 0, probe_enabled=False, n_users=nu)
        for pt in (8, 2):
            pd_ = build_engine(cfg, SIGMA_NULL, n_m, pt, n_users=nu)
            Pv = tuple(P0 for _ in range(nu))
            zv = tuple(0.0 for _ in range(nu))
            vv = tuple(V_PRIOR for _ in range(nu))
            iv = no(TAU, Pv, zv, vv, B_MAIN) - pd_(TAU, Pv, zv, vv, B_MAIN)
            rows.append({"n_users": nu, "probe_ticks": pt, "IV": float(iv)})
    mx = max(r["IV"] for r in rows)
    mn = min(r["IV"] for r in rows)
    return {"rows": rows, "max_IV": float(mx), "min_IV": float(mn), "tol": TOL_NOINFO,
            "pass": bool(mx <= TOL_NOINFO and mn >= -1e-12)}


def g5_drop_obs(n_m: int) -> dict:
    """`G5`：`drop_obs` 变异**有分辨力**（`n_changed ≥ 1`）。

    ★ 口径修正（见预注册 §9 的跑前修订 1）：`G5` 原来同时管"塌零"与"有分辨力"，
    但预注册 §4 第一行写"任一 `G` 未过 ⇒ `NO-VERDICT`"，而 §4 末行又要求"塌零"参与判词
    ⇒ 两条自相矛盾。现改：`G5` 只判"变异确实改变数值"（闸的属性），
    **"正格是否塌零"移到判词**（`IV-POSITIVE-NOT-FROM-READING`），两套读数都报。
    """
    rows = []
    n_collapse = 0
    n_pos = 0
    n_changed = 0
    for nu, b in ((2, B_SCARCE), (3, B_MAIN)):
        cfg = cfg_for(nu, "HH")
        no = build_engine(cfg, SIGMA_STAR, n_m, 0, probe_enabled=False, n_users=nu)
        good = build_engine(cfg, SIGMA_STAR, n_m, 0, n_users=nu)
        bad = build_engine(cfg, SIGMA_STAR, n_m, 0, n_users=nu, drop_obs=True)
        Pv = tuple(P0 for _ in range(nu))
        zv = tuple(0.0 for _ in range(nu))
        vv = tuple(V_PRIOR for _ in range(nu))
        a0 = no(TAU, Pv, zv, vv, b)
        v_good = good(TAU, Pv, zv, vv, b)
        v_bad = bad(TAU, Pv, zv, vv, b)
        iv_good = a0 - v_good
        iv_bad = a0 - v_bad
        changed = abs(v_good - v_bad) > 1e-15
        positive = abs(iv_good) > IV_NOISE
        collapsed = abs(iv_bad) <= IV_NOISE
        n_changed += int(changed)
        n_pos += int(positive)
        n_collapse += int(positive and collapsed)
        rows.append({"n_users": nu, "b_ticks": b, "IV_good": float(iv_good),
                     "IV_drop_obs": float(iv_bad), "dV": float(v_good - v_bad),
                     "positive": bool(positive), "changed": bool(changed),
                     "collapsed": bool(collapsed)})
    return {"rows": rows, "n_changed": n_changed, "n_positive_cells": n_pos,
            "n_collapse": n_collapse, "pass": bool(n_changed >= 1)}


def g6_budget_moved(n_m: int) -> dict:
    """`G6`：预算确实接进引擎 —— `A0(b=24) < A0(b=16)` 严格。"""
    cfg = cfg_for(3, "HH")
    no = build_engine(cfg, SIGMA_STAR, n_m, 0, probe_enabled=False, n_users=3)
    Pv = (P0, P0, P0)
    zv = (0.0, 0.0, 0.0)
    vv = (V_PRIOR, V_PRIOR, V_PRIOR)
    a16 = no(TAU, Pv, zv, vv, B_MAIN)
    a24 = no(TAU, Pv, zv, vv, B_MIRROR)
    return {"A0_b16": float(a16), "A0_b24": float(a24), "gap": float(a16 - a24),
            "pass": bool(a24 < a16)}


def g7_size_invariants(legs: list) -> dict:
    """`G7`：规模两条不变式（首格已建 >0 / 整组每条 `paid` 引擎都有状态）+ 无撞顶。"""
    rows = []
    ok = True
    for lg in legs:
        st = lg["states"]
        paid = list(st["paid"].values())
        c1 = st["n_after_first_cell"] is not None and st["n_after_first_cell"] > 0
        c2 = bool(paid) and all(v > 0 for v in paid)
        ok = ok and c1 and c2 and not st["capped"]
        rows.append({"arm": lg["arm"], "n_m": lg["n_m"], "paid": st["paid"],
                     "n_after_first_cell": st["n_after_first_cell"],
                     "c1_first_cell_nonzero": bool(c1), "c2_all_legs_built": bool(c2),
                     "capped": st["capped"]})
    return {"rows": rows, "pass": bool(ok and rows)}


def g7b_requery_idempotent(n_m: int) -> dict:
    """`G7b`：重复查询不改状态计数（memo 幂等）。"""
    cfg = cfg_for(3, "HH")
    e = build_engine(cfg, SIGMA_STAR, n_m, 2, n_users=3)
    Pv = (P0, P0, P0)
    zv = (0.0, 0.0, 0.0)
    vv = (V_PRIOR, V_PRIOR, V_PRIOR)
    v1 = e(TAU, Pv, zv, vv, B_MAIN)
    n1 = e.n_states()
    v2 = e(TAU, Pv, zv, vv, B_MAIN)
    n2 = e.n_states()
    return {"v1": float(v1), "v2": float(v2), "n1": int(n1), "n2": int(n2),
            "pass": bool(v1 == v2 and n1 == n2)}


def g8_distinct(n_m_list: tuple[int, ...]) -> dict:
    """`G8`：三个 plant 指纹两两不同；`n_m` 三点互异。"""
    fps = {}
    for tag, (nu, mode) in (("2u", (2, "HH")), ("3u_HH", (3, "HH")), ("3u_HL", (3, "HL"))):
        cfg = cfg_for(nu, mode)
        fps[tag] = {"A": list(cfg.plant.A), "sigma_w": list(cfg.plant.sigma_w),
                    "fp": cfg.fingerprint()}
    uniq = len({v["fp"] for v in fps.values()})
    return {"per_plant": fps, "n_plants": len(fps), "n_distinct": uniq,
            "n_m": list(n_m_list), "n_distinct_n_m": len(set(n_m_list)),
            "pass": bool(uniq == len(fps) and len(set(n_m_list)) == len(n_m_list))}


def g9_ast() -> dict:
    """`G9`：源码无 `int(p)` / `int(price)` / `int(probe_cost)` 截断；检测器对故意写错命中。"""
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
    probe_src = "def f(p):\n    return int(p)\n"
    probe_hits = [n for n in ast.walk(ast.parse(probe_src))
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "int"]
    return {"hits": hits, "n_hits": len(hits),
            "detector_catches_mutation": bool(probe_hits),
            "pass": bool(not hits and probe_hits)}


def diag_gh(n_m: int) -> dict:
    """GH 节点数敏感性（21 vs 41）——**分辨率诊断，不是判据**（预注册 §9 跑前修订 2）。

    R5.0 §1.4 已实测 `21→41` 会让某格值变 `1.01e-3` 相对 ⇒ 求积节点也是近似，不是"只有网格一种近似"。
    这里只回答：**符号会不会被节点数翻面**。不进 `gates_n_fail`。
    """
    rows = []
    for nu, b, pts in ((3, B_MAIN, (0, 2)), (2, B_SCARCE, (0,))):
        cfg = cfg_for(nu, "HH")
        Pv = tuple(P0 for _ in range(nu))
        zv = tuple(0.0 for _ in range(nu))
        vv = tuple(V_PRIOR for _ in range(nu))
        no21 = build_engine(cfg, SIGMA_STAR, n_m, 0, probe_enabled=False, n_users=nu, gh_n=21)
        no41 = build_engine(cfg, SIGMA_STAR, n_m, 0, probe_enabled=False, n_users=nu, gh_n=41)
        a21 = no21(TAU, Pv, zv, vv, b)
        a41 = no41(TAU, Pv, zv, vv, b)
        for pt in pts:
            e21 = build_engine(cfg, SIGMA_STAR, n_m, pt, n_users=nu, gh_n=21)
            e41 = build_engine(cfg, SIGMA_STAR, n_m, pt, n_users=nu, gh_n=41)
            iv21 = a21 - e21(TAU, Pv, zv, vv, b)
            iv41 = a41 - e41(TAU, Pv, zv, vv, b)
            rows.append({"n_users": nu, "b_ticks": b, "probe_ticks": pt,
                         "IV_gh21": float(iv21), "IV_gh41": float(iv41),
                         "rel_dV": float(abs(e21(TAU, Pv, zv, vv, b) - e41(TAU, Pv, zv, vv, b))
                                         / max(1e-30, abs(a21))),
                         "same_sign": bool((iv21 > IV_NOISE) == (iv41 > IV_NOISE))})
    return {"n_m": int(n_m), "rows": rows,
            "same_sign_all": bool(all(r["same_sign"] for r in rows)),
            "note": "诊断，不是判据；符号一致 ≠ 幅度可信"}


# ==================================================================== 判词
def _pick(rows: list, arm: str) -> list[dict]:
    return [r for r in rows if r["arm"] == arm and math.isfinite(r["IV_paid"])]


def judge(rows: list, gates: dict, n_m_list: tuple[int, ...]) -> dict:
    out: dict = {"n_cells": len(rows),
                 "n_cells_dropped": sum(1 for r in rows if not math.isfinite(r["IV_paid"])),
                 "n_capped": sum(1 for r in rows if r["capped"])}
    by_arm: dict = {}
    for arm in ("ctrl2u_b16", "main3u_b16", "anchor2u_b8_p0", "anchor3u_b8_p0",
                "mirror3u_b24", "mirror2u_b24", "comp3u_b16_HL"):
        sub = _pick(rows, arm)
        by_arm[arm] = {
            "n": len(sub),
            "max_IV": float(max((r["IV_paid"] for r in sub), default=float("nan"))),
            "cells": [{k: r[k] for k in ("n_m", "b_ticks", "p", "A0", "IV_paid", "dV_send")}
                      for r in sorted(sub, key=lambda x: (x["n_m"], -x["probe_ticks"]))],
        }
    out["by_arm"] = by_arm

    # 主件：3 用户 b=16，逐价格看三点 n_m 的符号
    def sign_profile(arm: str, pt: int | None = None) -> dict:
        sub = [r for r in _pick(rows, arm) if pt is None or r["probe_ticks"] == pt]
        pos = sum(1 for r in sub if r["IV_paid"] > IV_NOISE)
        neg = sum(1 for r in sub if r["IV_paid"] < -IV_NOISE)
        z = len(sub) - pos - neg
        return {"n": len(sub), "pos": pos, "zero": z, "neg": neg,
                "min": float(min((r["IV_paid"] for r in sub), default=float("nan"))),
                "max": float(max((r["IV_paid"] for r in sub), default=float("nan")))}

    main_free = sign_profile("main3u_b16", 0)
    main_q = sign_profile("main3u_b16", 2)
    main_p1 = sign_profile("main3u_b16", 8)
    out["main3u_b16"] = {"p0": main_free, "p0.25": main_q, "p1": main_p1}
    out["ctrl2u_b16"] = {f"p{pt}": sign_profile("ctrl2u_b16", pt) for pt in PRICE_PTS}
    out["anchor2u_b8_p0"] = sign_profile("anchor2u_b8_p0")
    out["anchor3u_b8_p0"] = sign_profile("anchor3u_b8_p0")
    out["mirror3u_b24"] = {"p0": sign_profile("mirror3u_b24", 0),
                           "p0.25": sign_profile("mirror3u_b24", 2)}
    out["mirror2u_b24"] = {"p0": sign_profile("mirror2u_b24", 0)}
    out["comp3u_b16_HL"] = {"p0": sign_profile("comp3u_b16_HL", 0),
                            "p0.25": sign_profile("comp3u_b16_HL", 2)}

    out["gates"] = {k: {"pass": v.get("pass"),
                        **{kk: vv for kk, vv in v.items()
                           if kk in ("max_abs", "abs_dA0", "max_abs_dIV", "n_violations",
                                     "max_IV", "min_IV", "n_changed", "n_collapse",
                                     "n_distinct", "n_hits", "gap", "tol")}}
                    for k, v in gates.items()}
    out["gates_failed"] = sorted(k for k, v in gates.items() if not v.get("pass"))
    out["gates_n_fail"] = len(out["gates_failed"])

    # ---- 主判词（逐条按预注册 §4）
    if out["gates_n_fail"] > 0:
        out["VERDICT"] = "NO-VERDICT-GATE-FAIL"
    elif len(set(n_m_list)) < 3:
        out["VERDICT"] = "NO-VERDICT-INSUFFICIENT-GRID"
    else:
        sig = [sign_profile("main3u_b16", pt) for pt in (0, 2)]
        all_pos = all(s["n"] == len(n_m_list) and s["pos"] == len(n_m_list) for s in sig)
        all_zero = all(s["n"] == len(n_m_list) and s["pos"] == 0 for s in sig)
        if all_zero:
            out["VERDICT"] = "IV-ZERO-3U-B16"
        elif all_pos:
            g5 = gates.get("g5", {})
            npos = int(g5.get("n_positive_cells", 0))
            ncol = int(g5.get("n_collapse", 0))
            out["collapse_check"] = {"n_positive_cells": npos, "n_collapse": ncol}
            out["VERDICT"] = ("IV-POSITIVE-3U-B16" if npos > 0 and ncol == npos
                              else "IV-POSITIVE-NOT-FROM-READING")
        else:
            out["VERDICT"] = "IV-MIXED-3U-B16"

    # ---- 机制判词（组合）
    ctrl_zero = all(v["pos"] == 0 for v in out["ctrl2u_b16"].values())
    mir_zero = (out["mirror3u_b24"]["p0"]["pos"] == 0 and out["mirror3u_b24"]["p0"]["n"] > 0)
    if out["VERDICT"] == "IV-POSITIVE-3U-B16" and ctrl_zero and mir_zero:
        out["MECHANISM"] = "MECHANISM-RECIPIENT-SELECTION-SUPPORTED"
    elif out["VERDICT"] == "IV-ZERO-3U-B16" and ctrl_zero:
        out["MECHANISM"] = "MECHANISM-NOT-SELECTION"
    else:
        out["MECHANISM"] = "MECHANISM-PARTIAL"
    out["mechanism_parts"] = {
        "ctrl2u_b16_all_zero": bool(ctrl_zero),
        "mirror3u_b24_zero": bool(mir_zero),
        "mirror3u_b24_measured": bool(out["mirror3u_b24"]["p0"]["n"] > 0),
        "anchor2u_b8_p0_positive": bool(out["anchor2u_b8_p0"]["pos"] > 0),
        "anchor2u_b8_p0_n": int(out["anchor2u_b8_p0"]["n"]),
        "anchor3u_b8_p0_positive": bool(out["anchor3u_b8_p0"]["pos"] > 0),
        "anchor3u_b8_p0_n": int(out["anchor3u_b8_p0"]["n"]),
        "comp3u_p0_positive": bool(out["comp3u_b16_HL"]["p0"]["pos"] > 0),
    }
    out["verdict_scope"] = ("只覆盖 α=2.5, P=0.01, κ=1, τ=4, b∈{8,16,24}, p∈{1,1/4,0}, "
                            "σ_csi=σ⋆, 3 用户 (0.02,0.05,0.05)；跨实例（2u vs 3u）不相减；"
                            "不是价格边界，不许外推")
    return out


# ==================================================================== 规模
def size_probe(specs=None, out_path: Path | None = None) -> list[dict]:
    """规模探针（**不是判据**）：定 `n_m` 三点用（预注册 §5）。

    逐 `n_m` 升序；某 `n_m` 的 3 用户整组撞 `STATE_CAP` 就停（`n_m` 单调 ⇒ 更大也不行）。
    """
    if specs is None:
        specs = [5, 7, 9, 11, 13, 17]
    out: list[dict] = []
    for n_m in specs:
        rec = {"n_m": int(n_m), "alpha": ALPHA, "P": P0, "capped": False, "legs": {}}
        for tag, nu, mode, cells in (
            ("3u_b16", 3, "HH", [{"tau": TAU, "b_ticks": B_MAIN, "probe_ticks": pt}
                                 for pt in PRICE_PTS]),
            ("2u_b16", 2, "HH", [{"tau": TAU, "b_ticks": B_MAIN, "probe_ticks": pt}
                                 for pt in PRICE_PTS]),
            ("3u_b24", 3, "HH", [{"tau": TAU, "b_ticks": B_MIRROR, "probe_ticks": pt}
                                 for pt in (0, 2)]),
        ):
            t0 = time.perf_counter()
            g = Group(nu, mode, n_m)
            g.run(cells)
            st = g.states()
            rec["legs"][tag] = {"sec": round(time.perf_counter() - t0, 3),
                                "states": st, "capped": st["capped"]}
            rec["capped"] = bool(rec["capped"] or st["capped"])
            print(f"  [size n_m={n_m} {tag}] {rec['legs'][tag]['sec']} s "
                  f"states={st['paid']} capped={st['capped']}", flush=True)
        out.append(rec)
        if out_path is not None:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        if rec["capped"]:
            print(f"  [size] n_m={n_m} 撞顶 ⇒ 停止（n_m 单调）", flush=True)
            break
    return out


# ==================================================================== main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="CW-R5.2 三用户 b=16 与收件人选择")
    ap.add_argument("--size-only", action="store_true", help="只跑规模探针")
    ap.add_argument("--ctrl-only", action="store_true", help="只跑不依赖面板的闸")
    ap.add_argument("--n-m", default="", help="覆盖主面板 n_m 三点，逗号分隔")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    n_m_list = N_M_MAIN if not args.n_m else tuple(int(x) for x in args.n_m.split(","))
    n_m_one = n_m_list[len(n_m_list) // 2] if not args.n_m else n_m_list[0]

    t_all = time.perf_counter()
    print(f"[{ROUND}] {PROTOCOL}  nu∈{{2,3}} tau={TAU} b∈{{{B_MAIN},{B_MIRROR},{B_SCARCE}}} "
          f"p∈{{1,1/4,0}} n_m={n_m_list}", flush=True)

    dest = Path(args.out) if args.out else (
        RESULTS / f"cw_r5_2_three_users_{_now_utc()}_{_src_hash()[:16]}")
    dest.mkdir(parents=True, exist_ok=True)

    if args.size_only:
        specs = [int(x) for x in args.n_m.split(",")] if args.n_m else None
        size = size_probe(specs=specs, out_path=dest / "r52_size.json")
        print(f"[size-only] n_m 探测 {len(size)} 档；最后一条 capped={size[-1]['capped']}", flush=True)
        return 0

    gates: dict = {"g1a": g1a_anchor_vs_r51(), "g1b": g1b_copy_identity(),
                   "g2": g2_no_leak(), "g6": g6_budget_moved(n_m_one),
                   "g7b": g7b_requery_idempotent(n_m_one),
                   "g8": g8_distinct(n_m_list), "g9": g9_ast()}
    print(f"[ctrl] {json.dumps({k: v.get('pass') for k, v in gates.items()})}", flush=True)
    if args.ctrl_only:
        print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'rows'}
                          for k, v in gates.items()}, indent=2, ensure_ascii=False)[:6000])
        print(f"[ctrl-only] n_fail={sum(1 for v in gates.values() if not v['pass'])} / {len(gates)}")
        return 0

    t0 = time.perf_counter()
    pnl = panel(n_m_list, n_m_one)
    sec_panel = round(time.perf_counter() - t0, 3)

    gates["g3"] = g3_monotone(pnl["rows"])
    gates["g4"] = g4_no_info(n_m_one)
    gates["g5"] = g5_drop_obs(n_m_one)
    gates["g7"] = g7_size_invariants(pnl["legs"])
    dgh = diag_gh(n_m_one)
    v = judge(pnl["rows"], gates, n_m_list)

    print(json.dumps({k: v[k] for k in ("VERDICT", "MECHANISM", "mechanism_parts",
                                        "gates_n_fail", "gates_failed")},
                     indent=2, ensure_ascii=False), flush=True)
    for arm in ("ctrl2u_b16", "main3u_b16", "anchor2u_b8_p0", "anchor3u_b8_p0",
                "mirror3u_b24", "mirror2u_b24", "comp3u_b16_HL"):
        blk = v[arm]
        if blk and all(isinstance(x, dict) for x in blk.values()):
            txt = " | ".join(f"{k}={vv['pos']}/{vv['n']}pos max={vv['max']:.4e}"
                             for k, vv in blk.items())
        else:
            txt = (f"pos={blk['pos']}/{blk['n']} zero={blk['zero']} "
                   f"min={blk['min']:.4e} max={blk['max']:.4e}")
        print(f"  {arm}: {txt}", flush=True)

    out = {"round": ROUND, "protocol": PROTOCOL, "utc": _now_utc(),
           "ticks_per_quantum": TICKS_PER_QUANTUM, "send_ticks": SEND_TICKS,
           "sec_panel": sec_panel,
           "sec_total": round(time.perf_counter() - t_all, 3),
           "panel": {"alpha": ALPHA, "P": P0, "kappa": KAPPA_P, "tau": TAU, "gh_n": GH,
                     "n_m_main": list(n_m_list), "n_m_one": n_m_one,
                     "price_ticks": list(PRICE_PTS), "b_ticks": [B_MAIN, B_MIRROR, B_SCARCE],
                     "sigma_w": {f"{nu}{m}": list(plant_for(nu, m).sigma_w)
                                 for nu, m in ((2, "HH"), (3, "HH"), (3, "HL"))}},
           "cells": pnl["rows"], "legs": pnl["legs"], "gates": gates, "verdicts": v,
           "diag_gh": dgh, "anchor_r51": R51_ANCHOR}
    (dest / "r52.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[evidence] {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
