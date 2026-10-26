# -*- coding: utf-8 -*-
"""`CW-R5.3` 把「单次发送区」的边界从 `τ` 那一侧钉住的控制件。

预注册：`docs/research/CW-R5.3-单次发送区时间边界-预注册-v0.1.md`（写于任何面板读数之前；
§5 的规模规则在读数前执行一次，选中值记进该节"读面板前选定"）。

本文件做什么
------------
只动一个轴：**窗口长度 `τ`**。把 `b=8`（恰好 1 次发送）与 `p=0`（只占时隙、不占量子）钉死，
扫 `τ ∈ {2,3,4,5,6}`，问「单次发送区」的正例能不能在 `τ` 上活下来、以及它随窗口怎么变。

为什么（§0/§1）
---------------
1. ★ 上游把槽数写成 `τ`，但 `V()` 里 `t == 1` 直接返回终值（`r5_2_three_users.py:177`）⇒
   **可动作时隙 = `τ−1`**。所以 `τ=2` 只有 1 个动作时隙（探测必然挤掉唯一发送）⇒ 那里的零是
   **结构性零**，只能当负控用，不当信息价值证据。
2. `b=8` 的正例只在 `τ=4` 被测过。它有两种解释：靠"1 探测 + 1 发送"这条时隙结构（H1，则 `τ≥3`
   都该有且随 `τ` 非降），还是 `τ=4` 的巧合（H2，则换 `τ` 会翻零）。只钉死 `b,p` 扫 `τ`
   是唯一能分开二者的单变量设计。

纪律
----
1. 零 episode、零训练、零参数更新；`config.py` / `posterior.py` / `r3_2_iv2u.py` /
   `r4_0_boundary.py` / `r5_1_price_ticks.py` / `r5_2_three_users.py` **一个字符不改**（只 `import`）。
2. 引擎直接用 R5.2 的 `build_engine`（含 `drop_obs` 变异），**不新写副本** ⇒ 跨轮身份靠 `G1`
   （对 R5.2 产物 `r52.json` 逐位）钉住，而不是靠"我又写了一遍同样的算术"。
3. 判据 `P-A'..P-E` / 闸 `G1..G9` 见预注册 §3/§6，**跑前写死**。
4. `IV = A0 − V_paid ≥ 0` 恒成立（动作集嵌套）⇒ 唯一能推翻"窗口越长越值钱"的形态是**零或下降**。
5. 结构性零（`τ=2`、`p=1`）单列，不进"信息价值"一侧。
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import time
from pathlib import Path

from .config import bounded_plant, mainline_config
from .r1_14_exactbase import _now_utc, _src_hash
from .r3_2_iv2u import STATE_CAP
from .r5_1_price_ticks import SEND_TICKS, TICKS_PER_QUANTUM
from .r5_2_three_users import (
    ALPHA,
    GH,
    KAPPA_P,
    P0,
    SIGMA_NULL,
    SIGMA_STAR,
    SIGMA_W0,
    V_PRIOR,
    build_engine,
    cfg_for,
    plant_for,
)

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results" / "cw_v1"

PROTOCOL = "CW-POMDP-v1.3-ticks"          # 沿用 R5.1/R5.2，不新立变体
ROUND = "CW-R5.3"

#: 面板常数（预注册 §2，跑前写死）
TAU_LIST: tuple[int, ...] = (2, 3, 4, 5, 6)
B_ONE = 8                                  # 1 量子 = 恰好 1 次发送
PT_MAIN = 0                                # p=0：只占时隙，不占量子
PT_STRUCT = 8                              # p=1：与 SEND 同价 ⇒ 探测与发送互斥（结构负控）
TAU_STRUCT: tuple[int, ...] = (2, 3, 4, 6)
TAU_SYM: tuple[int, ...] = (3, 4, 6)
#: 3 用户腿的 `τ` 上限（由 `--size-only` 实测：`τ=6` 在 `n_m=65` 撞 6.0e6 ⇒ 未测，不当零）
TAU_MAX_3U = 5
TAU_K4: tuple[int, ...] = (4,)

#: 主面板 `n_m` 三点：由预注册 §5 的规模规则定（`--size-only` 实测后写死，见 §9-1）
N_M_MAIN: tuple[int, ...] = (65, 129, 257)

#: 判据常数（预注册 §3）
IV_NOISE = 1e-12
TOL_BIT = 0.0
TOL_STRUCT = 1e-15
TOL_NOINFO = 1e-9

#: R5.2 正式证据（`G1` 跨轮身份锚；**从产物 JSON 读，不手抄**）
R52_EVID = "cw_r5_2_three_users_20261026T162053_ed05ba8445708e56"

ARMS_MAIN = ("ctrl2u_b8_p0", "main3u_b8_p0")


# ==================================================================== plant
def plant_sym2():
    """对称 plant：两用户同噪 ⇒ 「选人」无差别，只剩「择时」（预注册 §2.2 的 `sym2u`）。"""
    return bounded_plant(A=(1.0, 1.0), sigma_w=(SIGMA_W0, SIGMA_W0))


def plant_4u():
    """4 用户：1 低噪 + 3 个并列高噪 ⇒ 回答"分母还能不能继续加"（探索，不进主判据）。"""
    return bounded_plant(A=(1.0,) * 4,
                         sigma_w=(SIGMA_W0, SIGMA_W0 * ALPHA, SIGMA_W0 * ALPHA, SIGMA_W0 * ALPHA))


def cfg_sym2():
    return mainline_config(plant=plant_sym2())


def cfg_4u():
    return mainline_config(plant=plant_4u())


# ==================================================================== 组（整腿）
class Grp:
    """一条 `noprobe` 引擎 + 每价格一条 `paid` 引擎；`memo` 在**该组全部格**上累加。

    ★ 规模必须在**该组全部格求值之后**取（R4.2：单格探针系统性低估）。
    """

    def __init__(self, tag: str, cfg, n_users: int, n_m: int):
        self.tag = str(tag)
        self.n_users = int(n_users)
        self.n_m = int(n_m)
        self.cfg = cfg
        t0 = time.perf_counter()
        self.no = build_engine(cfg, SIGMA_STAR, self.n_m, 0,
                              probe_enabled=False, n_users=self.n_users)
        self.sec_no = round(time.perf_counter() - t0, 3)
        self.paid: dict[int, object] = {}
        self.sec_build: dict[str, float] = {}

    def _P(self) -> tuple:
        return tuple(P0 for _ in range(self.n_users))

    def paid_engine(self, pt: int):
        if pt not in self.paid:
            t0 = time.perf_counter()
            self.paid[pt] = build_engine(self.cfg, SIGMA_STAR, self.n_m, int(pt),
                                        n_users=self.n_users)
            self.sec_build[str(pt)] = round(time.perf_counter() - t0, 3)
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
            vp = eng(tau, self._P(), zero, vp0, b)
            out.append({
                "arm": cell["arm"], "n_users": self.n_users, "n_m": self.n_m,
                "tau": tau, "b_ticks": b, "b_quanta": b / TICKS_PER_QUANTUM,
                "p": pt / TICKS_PER_QUANTUM, "probe_ticks": pt,
                "P": P0, "alpha": ALPHA if self.tag != "sym2u" else 1.0, "kappa": KAPPA_P,
                "A0": float(a0), "V_paid": float(vp), "IV_paid": float(a0 - vp),
                "IV_share_A0": float((a0 - vp) / a0) if a0 else float("nan"),
                "capped": bool(self.no.capped() or eng.capped()),
            })
        return out

    def states(self) -> dict:
        return {"noprobe": self.no.n_states(),
                "paid": {str(k): v.n_states() for k, v in self.paid.items()},
                "capped": bool(self.no.capped() or any(v.capped() for v in self.paid.values()))}


# ==================================================================== 面板
def _cells(arm: str, taus) -> list[dict]:
    pt = PT_STRUCT if arm.endswith("struct_p1") else PT_MAIN
    return [{"arm": arm, "tau": int(t), "b_ticks": B_ONE, "probe_ticks": pt} for t in taus]


def panel(n_m_list: tuple[int, ...], taus: tuple[int, ...]) -> dict:
    rows: list[dict] = []
    meta: list[dict] = []

    def do(g: Grp, cells: list[dict]) -> None:
        t0 = time.perf_counter()
        rr = g.run(cells)
        st = g.states()
        rows.extend(rr)
        meta.append({"tag": g.tag, "n_users": g.n_users, "n_m": g.n_m,
                     "sec": round(time.perf_counter() - t0, 3), "sec_no": g.sec_no,
                     "sec_build": g.sec_build, "states": st,
                     "n_cells": len(rr),
                     "max_IV": float(max((r["IV_paid"] for r in rr), default=float("nan")))})
        print(f"  [{g.tag} n_m={g.n_m}] {meta[-1]['sec']} s states={st['paid']} "
              f"noprobe={st['noprobe']} maxIV={meta[-1]['max_IV']:.6e} capped={st['capped']}",
              flush=True)

    for n_m in n_m_list:
        # 主面板 + 结构负控共用一个 2 用户组（同 plant 同 n_m ⇒ 同一条腿）
        g2 = Grp("ctrl2u_b8", cfg_for(2, "HH"), 2, n_m)
        do(g2, _cells("ctrl2u_b8_p0", taus) + _cells("ctrl2u_b8_struct_p1", TAU_STRUCT))
        do(Grp("main3u_b8", cfg_for(3, "HH"), 3, n_m),
           _cells("main3u_b8_p0", tuple(t for t in taus if t <= TAU_MAX_3U)))
        do(Grp("sym2u", cfg_sym2(), 2, n_m), _cells("sym2u_b8_p0", TAU_SYM))
        do(Grp("k4u", cfg_4u(), 4, n_m), _cells("k4u_b8_p0", TAU_K4))

    return {"rows": rows, "legs": meta}


# ==================================================================== 闸
def _r52_json() -> dict:
    p = RESULTS / R52_EVID / "r52.json"
    if not p.exists():
        raise FileNotFoundError(f"找不到 R5.2 正式证据：{p}")
    return json.loads(p.read_text(encoding="utf-8"))


def g1_anchor_vs_r52(rows: list, n_m_list: tuple[int, ...]) -> dict:
    """`G1`：`τ=4, b=8, p=0` 的 `A0` 与 `IV_paid` 必须与 R5.2 产物**逐位**相同（锚从 JSON 取）。"""
    ref = _r52_json()
    want = {}
    for r in ref["cells"]:
        if r["arm"] in ("anchor2u_b8_p0", "anchor3u_b8_p0") and r["n_m"] in set(n_m_list):
            key = (2 if r["arm"].startswith("anchor2u") else 3, r["n_m"])
            want[key] = r
    mine = {}
    for r in rows:
        if r["tau"] == 4 and r["probe_ticks"] == PT_MAIN and r["arm"] in ARMS_MAIN:
            mine[(r["n_users"], r["n_m"])] = r
    tbl, worst = [], 0.0
    for key in sorted(set(want) & set(mine)):
        w, m = want[key], mine[key]
        d_a0 = abs(m["A0"] - w["A0"])
        d_iv = abs(m["IV_paid"] - w["IV_paid"])
        worst = max(worst, d_a0, d_iv)
        tbl.append({"n_users": key[0], "n_m": key[1],
                    "A0_mine": float(m["A0"]), "A0_ref": float(w["A0"]),
                    "IV_mine": float(m["IV_paid"]), "IV_ref": float(w["IV_paid"]),
                    "abs_dA0": float(d_a0), "abs_dIV": float(d_iv)})
    return {"ref": f"{R52_EVID}/r52.json", "rows": tbl, "n_cells": len(tbl),
            "n_missing": len(set(want) ^ set(mine)), "max_abs": float(worst),
            "tol": TOL_BIT, "pass": bool(tbl and len(tbl) == len(want) and worst <= TOL_BIT)}


def g2_no_leak(n_m: int) -> dict:
    """`G2`：3 用户 plant 上取 `n_users=2` ≡ 2 用户 plant 上 `n_users=2`（第 3 条 `sigma_w` 不泄漏）。"""
    c2, c3 = cfg_for(2, "HH"), cfg_for(3, "HH")
    tbl, worst = [], 0.0
    for tau in (3, 4):
        e2 = build_engine(c2, SIGMA_STAR, n_m, PT_MAIN, n_users=2)
        e3 = build_engine(c3, SIGMA_STAR, n_m, PT_MAIN, n_users=2)
        a = e2(tau, (P0, P0), (0.0, 0.0), (V_PRIOR, V_PRIOR), B_ONE)
        c = e3(tau, (P0, P0), (0.0, 0.0), (V_PRIOR, V_PRIOR), B_ONE)
        worst = max(worst, abs(a - c))
        tbl.append({"tau": tau, "two_plant": float(a), "three_plant_nu2": float(c),
                    "abs": float(abs(a - c))})
    return {"rows": tbl, "max_abs": float(worst), "tol": TOL_BIT, "pass": bool(worst <= TOL_BIT)}


def g3_slots(n_m: int) -> dict:
    """`G3`：**时隙记账** —— `τ=2` 上 `b∈{8,16,24}` 必须**逐位同值**（只有 1 个动作时隙）；
    `τ=3` 上 `A0(b=8) > A0(b=16)` 严格（2 个动作时隙 ⇒ 第 2 次发送有用）。

    ★ 变异（把 `t==1` 改成允许动作）会让第一半 FAIL；这是本闸有分辨力的理由。
    """
    cfg = cfg_for(2, "HH")
    no = build_engine(cfg, SIGMA_STAR, n_m, 0, probe_enabled=False, n_users=2)
    P2, Z2, V2 = (P0, P0), (0.0, 0.0), (V_PRIOR, V_PRIOR)
    a2 = {b: no(2, P2, Z2, V2, b) for b in (8, 16, 24)}
    a3 = {b: no(3, P2, Z2, V2, b) for b in (8, 16, 24)}
    a4 = {b: no(4, P2, Z2, V2, b) for b in (8, 16, 24)}
    d2 = max(abs(a2[8] - a2[16]), abs(a2[8] - a2[24]))
    ok = (d2 <= TOL_STRUCT) and (a3[8] - a3[16] > 0.0)
    return {"tau2": {f"b{b}": float(v) for b, v in a2.items()}, "tau2_max_spread": float(d2),
            "tau3": {f"b{b}": float(v) for b, v in a3.items()},
            "tau3_gap_8_minus_16": float(a3[8] - a3[16]),
            "tau4": {f"b{b}": float(v) for b, v in a4.items()},
            "tau4_gap_8_minus_16": float(a4[8] - a4[16]),
            "tol": TOL_STRUCT, "pass": bool(ok),
            "note": "τ=4 的两项是诊断；判定只用 τ=2 逐位同值 + τ=3 严格下降"}


def g4_no_info(n_m: int) -> dict:
    """`G4`：`σ_csi=1e6`（读数几乎无信息）⇒ `−1e-12 ≤ IV ≤ 1e-9`。"""
    tbl = []
    for nu in (2, 3):
        cfg = cfg_for(nu, "HH")
        no = build_engine(cfg, SIGMA_NULL, n_m, 0, probe_enabled=False, n_users=nu)
        pd_ = build_engine(cfg, SIGMA_NULL, n_m, PT_MAIN, n_users=nu)
        P, Z, V = tuple(P0 for _ in range(nu)), tuple(0.0 for _ in range(nu)), tuple(V_PRIOR for _ in range(nu))
        for tau in (3, 4):
            iv = no(tau, P, Z, V, B_ONE) - pd_(tau, P, Z, V, B_ONE)
            tbl.append({"n_users": nu, "tau": tau, "IV": float(iv)})
    mx = max(r["IV"] for r in tbl)
    mn = min(r["IV"] for r in tbl)
    return {"rows": tbl, "max_IV": float(mx), "min_IV": float(mn), "tol": TOL_NOINFO,
            "pass": bool(mx <= TOL_NOINFO and mn >= -1e-12)}


def g5_drop_obs(n_m: int) -> dict:
    """`G5`：`drop_obs` 变异**有分辨力**（`n_changed ≥ 1`）；正格塌零率登记到判词。"""
    tbl, n_changed, n_pos, n_collapse = [], 0, 0, 0
    for nu in (2, 3):
        cfg = cfg_for(nu, "HH")
        no = build_engine(cfg, SIGMA_STAR, n_m, 0, probe_enabled=False, n_users=nu)
        good = build_engine(cfg, SIGMA_STAR, n_m, PT_MAIN, n_users=nu)
        bad = build_engine(cfg, SIGMA_STAR, n_m, PT_MAIN, n_users=nu, drop_obs=True)
        P, Z, V = tuple(P0 for _ in range(nu)), tuple(0.0 for _ in range(nu)), tuple(V_PRIOR for _ in range(nu))
        for tau in (3, 4, 5):
            a0 = no(tau, P, Z, V, B_ONE)
            vg, vb = good(tau, P, Z, V, B_ONE), bad(tau, P, Z, V, B_ONE)
            ivg, ivb = a0 - vg, a0 - vb
            changed = abs(vg - vb) > 1e-15
            pos = abs(ivg) > IV_NOISE
            n_changed += int(changed)
            n_pos += int(pos)
            n_collapse += int(pos and abs(ivb) <= IV_NOISE)
            tbl.append({"n_users": nu, "tau": tau, "IV_good": float(ivg), "IV_drop_obs": float(ivb),
                        "dV": float(vg - vb), "positive": bool(pos), "changed": bool(changed)})
    return {"rows": tbl, "n_changed": n_changed, "n_positive_cells": n_pos,
            "n_collapse": n_collapse, "pass": bool(n_changed >= 1)}


def g6g8_fingerprints() -> dict:
    """`G6`/`G8`：四个 plant 的 `fingerprint()` 两两不同（`α` 标度错位曾让两档跑成同一 plant）。"""
    fps = {}
    for tag, cfg in (("2u_asym", cfg_for(2, "HH")), ("3u_HH", cfg_for(3, "HH")),
                     ("2u_sym", cfg_sym2()), ("4u", cfg_4u())):
        fps[tag] = {"A": list(cfg.plant.A), "sigma_w": list(cfg.plant.sigma_w),
                    "fp": cfg.fingerprint()}
    uniq = len({v["fp"] for v in fps.values()})
    return {"per_plant": fps, "n_plants": len(fps), "n_distinct": uniq,
            "pass": bool(uniq == len(fps))}


def g7_size(legs: list) -> dict:
    """`G7`：整腿 `n_states>0` / 无撞顶。"""
    tbl = []
    ok = True
    for lg in legs:
        st = lg["states"]
        paid = list(st["paid"].values())
        c = bool(paid) and all(v > 0 for v in paid) and st["noprobe"] > 0 and not st["capped"]
        ok = ok and c
        tbl.append({"tag": lg["tag"], "n_m": lg["n_m"], "noprobe": st["noprobe"],
                    "paid": st["paid"], "capped": st["capped"], "ok": bool(c)})
    return {"rows": tbl, "pass": bool(ok and tbl)}


def g7b_idempotent(n_m: int) -> dict:
    """`G7b`：重复查询不改状态计数（memo 幂等）。"""
    e = build_engine(cfg_for(2, "HH"), SIGMA_STAR, n_m, PT_MAIN, n_users=2)
    P2, Z2, V2 = (P0, P0), (0.0, 0.0), (V_PRIOR, V_PRIOR)
    v1 = e(4, P2, Z2, V2, B_ONE)
    n1 = e.n_states()
    v2 = e(4, P2, Z2, V2, B_ONE)
    n2 = e.n_states()
    return {"v1": float(v1), "v2": float(v2), "n1": int(n1), "n2": int(n2),
            "pass": bool(v1 == v2 and n1 == n2)}


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
    probe_hits = [n for n in ast.walk(ast.parse("def f(p):\n    return int(p)\n"))
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "int"]
    return {"hits": hits, "n_hits": len(hits), "detector_catches_mutation": bool(probe_hits),
            "pass": bool(not hits and probe_hits)}


def diag_gh(n_m: int) -> dict:
    """GH `21` vs `41`：**分辨率诊断，不是判据**（R5.0 §1.4 已证节点数会改读数）。"""
    tbl = []
    cfg = cfg_for(2, "HH")
    P2, Z2, V2 = (P0, P0), (0.0, 0.0), (V_PRIOR, V_PRIOR)
    for tau in (3, 4, 5):
        no21 = build_engine(cfg, SIGMA_STAR, n_m, 0, probe_enabled=False, n_users=2, gh_n=21)
        no41 = build_engine(cfg, SIGMA_STAR, n_m, 0, probe_enabled=False, n_users=2, gh_n=41)
        e21 = build_engine(cfg, SIGMA_STAR, n_m, PT_MAIN, n_users=2, gh_n=21)
        e41 = build_engine(cfg, SIGMA_STAR, n_m, PT_MAIN, n_users=2, gh_n=41)
        a21, a41 = no21(tau, P2, Z2, V2, B_ONE), no41(tau, P2, Z2, V2, B_ONE)
        iv21, iv41 = a21 - e21(tau, P2, Z2, V2, B_ONE), a41 - e41(tau, P2, Z2, V2, B_ONE)
        tbl.append({"tau": tau, "IV_gh21": float(iv21), "IV_gh41": float(iv41),
                    "rel_dV": float(abs(a21 - a41) / max(1e-30, abs(a21))),
                    "same_sign": bool((iv21 > IV_NOISE) == (iv41 > IV_NOISE))})
    return {"n_m": int(n_m), "rows": tbl,
            "same_sign_all": bool(all(r["same_sign"] for r in tbl)),
            "note": "诊断，不是判据；符号一致 ≠ 幅度可信"}


# ==================================================================== 判词
def _grid(rows: list, arm: str, n_m_list: tuple[int, ...]) -> list[dict]:
    """按 `n_m` 取该臂的全部 `τ` 行，`τ` 升序。"""
    out = []
    for n_m in n_m_list:
        sub = sorted((r for r in rows if r["arm"] == arm and r["n_m"] == n_m),
                     key=lambda r: r["tau"])
        if sub:
            out.append({"n_m": n_m, "tau": [r["tau"] for r in sub],
                        "IV": [float(r["IV_paid"]) for r in sub],
                        "A0": [float(r["A0"]) for r in sub],
                        "share": [float(r["IV_share_A0"]) for r in sub]})
    return out


def _res(rows: list, arm: str, tau: int, n_m_list: tuple[int, ...]) -> float | None:
    """`τ` 上的**网格分辨率** = 同一 `(arm, τ)` 在各 `n_m` 之间的最大相对差。

    ★ `max ≤ IV_NOISE`（全零）时返回 `None` —— **"无靶子"不许写成 `spread=0`**（R4.1 教训，同族第 6 次）。
    """
    vals = [r["IV_paid"] for r in rows
            if r["arm"] == arm and r["tau"] == tau and r["n_m"] in set(n_m_list)
            and math.isfinite(r["IV_paid"])]
    if len(vals) < 2 or max(vals) <= IV_NOISE:
        return None
    return float((max(vals) - min(vals)) / max(vals))


def judge(rows: list, gates: dict, n_m_list: tuple[int, ...], tau_max: int,
          taus_all: tuple[int, ...] = TAU_LIST) -> dict:
    out: dict = {"n_cells": len(rows),
                 "n_cells_dropped": sum(1 for r in rows if not math.isfinite(r["IV_paid"])),
                 "n_capped": sum(1 for r in rows if r["capped"]),
                 "tau_max_measured": int(tau_max)}
    taus = [t for t in taus_all if t <= tau_max]
    grids = {arm: _grid(rows, arm, n_m_list) for arm in
             ("ctrl2u_b8_p0", "main3u_b8_p0", "sym2u_b8_p0", "k4u_b8_p0", "ctrl2u_b8_struct_p1")}
    out["grids"] = grids

    res = {(arm, t): _res(rows, arm, t, n_m_list) for arm in ARMS_MAIN for t in taus}
    out["resolution"] = {f"{a}@{t}": v for (a, t), v in res.items()}

    # ---- τ=2 结构负控
    tau2 = [r for r in rows if r["tau"] == 2 and r["arm"] in ARMS_MAIN and math.isfinite(r["IV_paid"])]
    out["tau2"] = {"n": len(tau2), "max_abs_IV": float(max((abs(r["IV_paid"]) for r in tau2), default=float("nan"))),
                   "all_bit_zero": bool(tau2 and all(abs(r["IV_paid"]) <= TOL_STRUCT for r in tau2))}

    # ---- τ≥3 主判
    ge3 = [r for r in rows if r["tau"] >= 3 and r["tau"] <= tau_max
           and r["arm"] in ARMS_MAIN and math.isfinite(r["IV_paid"])]
    n_pos = sum(1 for r in ge3 if r["IV_paid"] > IV_NOISE)
    out["ge3"] = {"n": len(ge3), "n_pos": n_pos, "n_zero": len(ge3) - n_pos,
                  "per_arm": {arm: {"n": len([r for r in ge3 if r["arm"] == arm]),
                                    "n_pos": sum(1 for r in ge3 if r["arm"] == arm and r["IV_paid"] > IV_NOISE),
                                    "min": float(min((r["IV_paid"] for r in ge3 if r["arm"] == arm),
                                                     default=float("nan"))),
                                    "max": float(max((r["IV_paid"] for r in ge3 if r["arm"] == arm),
                                                     default=float("nan")))}
                              for arm in ARMS_MAIN}}
    out["ge3"]["tau_by_arm"] = {arm: sorted({r["tau"] for r in ge3 if r["arm"] == arm})
                                for arm in ARMS_MAIN}

    # ---- 单调性：最细网格上逐 τ 对
    mono = {"pairs": [], "n_drop_beyond_res": 0}
    fine = max(n_m_list)
    for arm in ARMS_MAIN:
        g = [x for x in grids[arm] if x["n_m"] == fine]
        if not g:
            continue
        g = g[0]
        for i in range(len(g["tau"]) - 1):
            t0_, t1_ = g["tau"][i], g["tau"][i + 1]
            if t0_ < 3:
                continue
            v0, v1 = g["IV"][i], g["IV"][i + 1]
            drop = (v0 - v1) / max(v0, v1) if max(v0, v1) > IV_NOISE else 0.0
            r0, r1 = res.get((arm, t0_)), res.get((arm, t1_))
            tol = max([x for x in (r0, r1) if x is not None], default=None)
            beyond = bool(tol is not None and drop > tol)
            mono["n_drop_beyond_res"] += int(beyond)
            mono["pairs"].append({"arm": arm, "n_m": fine, "tau_from": t0_, "tau_to": t1_,
                                  "IV_from": float(v0), "IV_to": float(v1),
                                  "rel_drop": float(drop), "res": tol, "beyond_res": beyond})
    out["monotone"] = mono

    # ---- 结构负控（p=1）与对称臂/4 用户臂（描述）
    st_p1 = [r for r in rows if r["arm"] == "ctrl2u_b8_struct_p1" and math.isfinite(r["IV_paid"])]
    out["struct_p1"] = {"n": len(st_p1),
                        "max_abs_IV": float(max((abs(r["IV_paid"]) for r in st_p1), default=float("nan"))),
                        "all_bit_zero": bool(st_p1 and all(abs(r["IV_paid"]) <= TOL_STRUCT for r in st_p1))}
    sym = [r for r in rows if r["arm"] == "sym2u_b8_p0" and math.isfinite(r["IV_paid"])]
    out["sym2u"] = {"n": len(sym), "max_abs_IV": float(max((abs(r["IV_paid"]) for r in sym), default=float("nan"))),
                    "all_le_noise": bool(sym and all(abs(r["IV_paid"]) <= IV_NOISE for r in sym))}
    k4 = [r for r in rows if r["arm"] == "k4u_b8_p0" and math.isfinite(r["IV_paid"])]
    out["k4u"] = {"n": len(k4), "max_abs_IV": float(max((abs(r["IV_paid"]) for r in k4), default=float("nan"))),
                  "n_pos": sum(1 for r in k4 if r["IV_paid"] > IV_NOISE)}

    out["gates"] = {k: {kk: vv for kk, vv in v.items()
                        if kk in ("pass", "max_abs", "n_missing", "n_changed", "n_collapse",
                                  "max_IV", "min_IV", "n_distinct", "n_hits", "tau2_max_spread",
                                  "tau3_gap_8_minus_16", "tol")}
                    for k, v in gates.items()}
    out["gates_failed"] = sorted(k for k, v in gates.items() if not v.get("pass"))
    out["gates_n_fail"] = len(out["gates_failed"])

    # ---- 主判词（预注册 §4，顺序取第一个命中）
    if out["gates_n_fail"] > 0:
        out["VERDICT"] = "NO-VERDICT-GATE-FAIL"
    elif len(set(n_m_list)) < 3:
        out["VERDICT"] = "NO-VERDICT-INSUFFICIENT-GRID"
    elif not out["tau2"]["all_bit_zero"]:
        out["VERDICT"] = "SLOT-COUNTING-WRONG"
    elif out["ge3"]["n_pos"] == 0:
        out["VERDICT"] = "SINGLE-SEND-ZERO-ABOVE-TAU2"
    elif out["ge3"]["n_zero"] > 0:
        out["VERDICT"] = "SINGLE-SEND-VALUE-PARTIAL"
    elif mono["n_drop_beyond_res"] > 0:
        out["VERDICT"] = "SINGLE-SEND-VALUE-NONMONOTONE"
    else:
        out["VERDICT"] = "SINGLE-SEND-VALUE-GROWS-WITH-WINDOW"

    # ---- 机制判词（组合）
    g5 = gates.get("g5", {})
    collapse_all = int(g5.get("n_positive_cells", 0)) > 0 and \
        int(g5.get("n_collapse", 0)) == int(g5.get("n_positive_cells", 0))
    out["collapse_check"] = {"n_positive_cells": int(g5.get("n_positive_cells", 0)),
                             "n_collapse": int(g5.get("n_collapse", 0)),
                             "all_collapsed": bool(collapse_all)}
    if out["VERDICT"] == "SINGLE-SEND-VALUE-GROWS-WITH-WINDOW" and collapse_all and out["sym2u"]["all_le_noise"]:
        out["MECHANISM"] = "MECHANISM-SELECTION-ACROSS-WINDOWS"
    elif out["VERDICT"] == "SINGLE-SEND-VALUE-GROWS-WITH-WINDOW" and not out["sym2u"]["all_le_noise"]:
        out["MECHANISM"] = "MECHANISM-SELECTION-PLUS-TIMING"
    elif out["VERDICT"] in ("SINGLE-SEND-ZERO-ABOVE-TAU2", "SINGLE-SEND-VALUE-PARTIAL"):
        out["MECHANISM"] = "MECHANISM-WINDOW-DEPENDENT"
    else:
        out["MECHANISM"] = "MECHANISM-PARTIAL"
    out["mechanism_parts"] = {
        "tau2_all_bit_zero": out["tau2"]["all_bit_zero"],
        "struct_p1_all_bit_zero": out["struct_p1"]["all_bit_zero"],
        "sym2u_all_le_noise": out["sym2u"]["all_le_noise"],
        "drop_obs_all_collapsed": collapse_all,
        "drop_obs_changed_cells": int(g5.get("n_changed", 0)),
        "k4u_n_pos": out["k4u"]["n_pos"],
    }
    out["verdict_scope"] = (
        f"只覆盖 α=2.5 (2u (0.02,0.05) / 3u (0.02,0.05,0.05) / 4u +3 高噪), P=0.01, κ=1, b=8, p=0, "
        f"τ∈{{2..{tau_max}}}, σ_csi=σ⋆, gh_n=21；跨实例（2u/3u/4u）不相减；"
        "不是价格边界；不外推 τ≥7 或 p>0")
    return out


# ==================================================================== 规模探针
def size_probe(specs, taus, out_path: Path | None = None) -> list[dict]:
    """规模探针（**不是判据**）：逐 `n_m` 升序、逐 `τ` 升序单跑，撞顶即停。

    只打印 `n_states` / `sec` / `capped`，**不打印 IV**（保持面板读数盲区）。
    """
    out: list[dict] = []
    for n_m in specs:
        rec = {"n_m": int(n_m), "legs": {}, "capped": False}
        for tag, cfg, nu in (("2u_b8_p0", cfg_for(2, "HH"), 2),
                             ("3u_b8_p0", cfg_for(3, "HH"), 3)):
            g = Grp(tag, cfg, nu, n_m)
            per = []
            for tau in taus:
                t0 = time.perf_counter()
                g.run([{"arm": tag, "tau": tau, "b_ticks": B_ONE, "probe_ticks": PT_MAIN}])
                st = g.states()
                per.append({"tau": tau, "sec": round(time.perf_counter() - t0, 3),
                            "states_after": int(sum(st["paid"].values())) + int(st["noprobe"]),
                            "capped": bool(st["capped"])})
                print(f"  [size n_m={n_m} {tag} tau={tau}] {per[-1]['sec']} s "
                      f"states={per[-1]['states_after']} capped={per[-1]['capped']}", flush=True)
                if st["capped"]:
                    break
            rec["legs"][tag] = per
            rec["capped"] = bool(rec["capped"] or any(p["capped"] for p in per))
        out.append(rec)
        if out_path is not None:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        if rec["capped"]:
            print(f"  [size] n_m={n_m} 撞顶 ⇒ 停止（规模随 n_m 单调以外，τ 也已到顶）", flush=True)
            break
    return out


# ==================================================================== main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="CW-R5.3 单次发送区的窗口边界")
    ap.add_argument("--size-only", action="store_true")
    ap.add_argument("--ctrl-only", action="store_true")
    ap.add_argument("--n-m", default="")
    ap.add_argument("--taus", default="", help="覆盖 τ 列表，逗号分隔")
    ap.add_argument("--tau-max", type=int, default=0, help="判定用的 τ 上限（撞顶后降）")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    n_m_list = N_M_MAIN if not args.n_m else tuple(int(x) for x in args.n_m.split(","))
    taus = TAU_LIST if not args.taus else tuple(int(x) for x in args.taus.split(","))
    tau_max = int(args.tau_max) if args.tau_max else max(taus)

    t_all = time.perf_counter()
    print(f"[{ROUND}] {PROTOCOL}  b={B_ONE} p=0 tau={taus} n_m={n_m_list}", flush=True)

    dest = Path(args.out) if args.out else (
        RESULTS / f"cw_r5_3_tau_boundary_{_now_utc()}_{_src_hash()[:16]}")
    dest.mkdir(parents=True, exist_ok=True)

    if args.size_only:
        size = size_probe(list(n_m_list), list(taus), out_path=dest / "r53_size.json")
        print(f"[size-only] n_m 探测 {len(size)} 档；最后一条 capped={size[-1]['capped']}", flush=True)
        return 0

    n_m_one = n_m_list[len(n_m_list) // 2] if len(n_m_list) >= 3 else n_m_list[0]
    gates: dict = {"g2": g2_no_leak(n_m_one), "g3": g3_slots(n_m_one),
                   "g6g8": g6g8_fingerprints(), "g7b": g7b_idempotent(n_m_one), "g9": g9_ast()}
    print(f"[ctrl] {json.dumps({k: v.get('pass') for k, v in gates.items()})}", flush=True)
    if args.ctrl_only:
        print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'rows'}
                          for k, v in gates.items()}, indent=2, ensure_ascii=False)[:5000])
        print(f"[ctrl-only] n_fail={sum(1 for v in gates.values() if not v['pass'])} / {len(gates)}")
        return 0

    t0 = time.perf_counter()
    pnl = panel(n_m_list, taus)
    sec_panel = round(time.perf_counter() - t0, 3)

    gates["g1"] = g1_anchor_vs_r52(pnl["rows"], n_m_list)
    gates["g4"] = g4_no_info(n_m_one)
    gates["g5"] = g5_drop_obs(n_m_one)
    gates["g7"] = g7_size(pnl["legs"])
    dgh = diag_gh(n_m_one)
    v = judge(pnl["rows"], gates, n_m_list, tau_max, taus)

    print(json.dumps({k: v[k] for k in ("VERDICT", "MECHANISM", "mechanism_parts",
                                        "gates_n_fail", "gates_failed", "tau2", "ge3",
                                        "struct_p1", "sym2u", "k4u")},
                     indent=2, ensure_ascii=False), flush=True)
    for arm, gs in v["grids"].items():
        for g in gs:
            print(f"  {arm} n_m={g['n_m']}: " +
                  " ".join(f"τ{t}={iv:.4e}" for t, iv in zip(g["tau"], g["IV"])), flush=True)

    out = {"round": ROUND, "protocol": PROTOCOL, "utc": _now_utc(),
           "ticks_per_quantum": TICKS_PER_QUANTUM, "send_ticks": SEND_TICKS,
           "sec_panel": sec_panel, "sec_total": round(time.perf_counter() - t_all, 3),
           "panel": {"alpha": ALPHA, "P": P0, "kappa": KAPPA_P, "gh_n": GH,
                     "n_m_main": list(n_m_list), "n_m_one": n_m_one, "taus": list(taus),
                     "b_ticks": B_ONE, "p": 0.0, "p_struct": 1.0,
                     "tau_sym": list(TAU_SYM), "tau_k4": list(TAU_K4),
                     "tau_max_3u": TAU_MAX_3U,
                     "sigma_w": {"2u_asym": list(plant_for(2, "HH").sigma_w), "2u_sym": list(plant_sym2().sigma_w),
                                 "3u_HH": list(cfg_for(3, "HH").plant.sigma_w),
                                 "4u": list(plant_4u().sigma_w)}},
           "cells": pnl["rows"], "legs": pnl["legs"], "gates": gates, "verdicts": v,
           "diag_gh": dgh}
    (dest / "r53.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[evidence] {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
