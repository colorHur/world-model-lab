# -*- coding: utf-8 -*-
"""`CW-R3.0`：**R3 立项卡**——「排序误差是瓶颈」的可证伪判据 ＋ 三个零 episode 判定。

预注册：`docs/research/CW-R3.0-立项卡-预注册-v0.1.md`（写于 2026-10-25 22:35，跑前）

三个判定（对应 v4 §8 表 `R3` 行："**先证明排序误差是瓶颈**，再检验机制是否改善闭环成本"）
--------------------------------------------------------------------------------------
`J1` 排序遗憾：把 `plan_ref._expect_after` 的解析排序（`tail="paced"`，研究基线）与
     **精确闭环最优排序**（`A=1`、`p̂` 冻结、动作集 `{HOLD, SEND_i}`）比。
`J2` 解析路线的自查能力：换 `DPTail`（grid=`r173`）再排一次，看它能不能自己把缺口关掉。
     —— 若解析能关掉 ⇒ "学习"没有可赢的对手。
`J3` 真值可算到多大：`ExactClosedLoop` 的状态数/耗时随 τ 的**实测**增长。
     —— 面板是 τ=256，若真值在 τ≈10 就爆掉，则"在面板尺度上验证学习声明"没有参考物。

★★ `J0`（结构事实，闸 `C5` 用 AST 钉住）：`plan_dp` / `dp_audit` 的动作集**不含 `PROBE`**
   ⇒ J1/J2 只对「`HOLD` vs `SEND_i`」发言，**不得**外推到"信道信息（探测）的定价也没问题"。

本文件纪律
----------
1. **零 episode**（`run_episode = 0`）、零训练、零参数更新、纯 CPU。
2. **被测物一字节未动**：只读 `plan_ref.PlanRef._expect_after` / `plan_dp.DPTail` /
   `dp_audit.ExactClosedLoop`；本文件是**独立重写**的精确 DP（带 Q 值），并**必须**与
   `ExactClosedLoop.value` 逐位相等（闸 `C1`），否则本轮不出判词（预注册 §6-①）。
3. 阈值全部来自预注册 §2.4（看数前写定）；`judge()` 只按表取，不许人工改判词。
"""

from __future__ import annotations

import argparse
import ast
import gc
import json
import math
import time
from functools import lru_cache
from itertools import product
from pathlib import Path

import numpy as np

from .config import mainline_config
from .dp_audit import ExactClosedLoop
from .plan_dp import DPTail
from .plan_ref import PlanRef, _St, _TIE_EPS
from .r1_14_exactbase import _now_utc, _src_hash
from .world import Action

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results" / "cw_v1"

PROTOCOL = "CW-POMDP-v1.1"
ROUND = "CW-R3.0"

#: 预注册 §3 网格
TAUS: tuple[int, ...] = (4, 6, 8, 10, 12, 14)   # ★ 12/14 为跑后扩展，见预注册 §7-1
PS: tuple[float, ...] = (0.005, 0.010, 0.020, 0.050)
PHATS: tuple[float, ...] = (0.5, 0.2)

#: 预注册 §2.4 阈值（**看数前写定**）
THRESH_NB_REGRET = 5e-3
THRESH_B_REGRET = 5e-2
AGREE_OK = 0.95
AGREE_BAD = 0.85
CLOSE_OK = 0.8
CLOSE_BAD = 0.5

#: 预注册 §3-J3：τ=10 若超这两个限，就声明"面板尺度无可信真值"
J3_MAX_STATES = 2e5
J3_MAX_SECONDS = 120.0

#: 三动作固定序（与 `plan_ref._pick` 的 tie-break 同向：HOLD 优先、再按用户下标）
ORDER: tuple[str, ...] = ("HOLD", "SEND0", "SEND1")


# ==================================================================== 精确最优（独立重写）
class ExactQ:
    """`(P0,P1,b)` 上的**无插值**闭环后向 DP，返回三动作的 Q 值与 `V*`。

    递推与 `dp_audit.ExactClosedLoop` **同构**（`A=1`、`p̂` 冻结、动作集 `{HOLD,SEND_i}`），
    但额外给出每个动作的 Q 值 —— 排序遗憾必须要有 Q，只有 `min` 是不够的。
    **闸 `C1` 要求本类的 `V` 与 `ExactClosedLoop.value` 逐位相等**（同为 0.0，不是 1e-12）。
    """

    def __init__(self, cfg, p_hat) -> None:
        p = cfg.plant
        a = np.asarray(p.A, dtype=float)
        if not np.all(a == 1.0):
            raise ValueError(f"ExactQ 只在 A≡1 上成立，实为 A={tuple(a)}（预注册 §6-①）")
        self.sw2 = tuple(float(x) ** 2 for x in p.sigma_w)
        self.sm2 = float(p.sigma_meas) ** 2
        self.q = float(p.q_cost)
        ph = np.asarray(p_hat, dtype=float).reshape(-1)
        if ph.size != len(self.sw2):
            raise ValueError(f"p_hat 长度 {ph.size} != n_users {len(self.sw2)}")
        self.p_hat = tuple(float(x) for x in ph)
        self.V = None
        self.n_states = 0

    def solve(self, tau: int) -> "ExactQ":
        """建 `V(0..tau)` 的缓存（**每 τ 一份**：p̂ 进递推，不能跨 p̂ 复用）。"""
        sw2, sm2, q, ph = self.sw2, self.sm2, self.q, self.p_hat
        n = len(sw2)

        @lru_cache(maxsize=None)
        def V(t: int, p0: float, p1: float, bd: int) -> float:
            if t == 0:
                return 0.0
            ps = (p0, p1)
            nxt = tuple(ps[i] + sw2[i] for i in range(n))
            best = V(t - 1, nxt[0], nxt[1], bd)
            if bd >= 1:
                for i in range(n):
                    g = ps[i] / (ps[i] + sm2)
                    d = (1.0 - g) * ps[i] + sw2[i]
                    nd = list(nxt)
                    nd[i] = d
                    v = (ph[i] * V(t - 1, nd[0], nd[1], bd - 1)
                         + (1.0 - ph[i]) * V(t - 1, nxt[0], nxt[1], bd - 1))
                    if v < best:
                        best = v
            return q * (p0 + p1) + best

        self._V = V
        self.tau = int(tau)
        return self

    def value(self, P0: float, P1: float, b: int) -> float:
        v = self._V(self.tau, float(P0), float(P1), int(b))
        self.n_states = self._V.cache_info().currsize
        return float(v)

    def q_values(self, P0: float, P1: float, b: int) -> dict[str, float]:
        """三动作的精确 Q（含本步即时代价 `q·(P0+P1)`）；买不起 ⇒ `inf`。"""
        sw2, sm2, q, ph = self.sw2, self.sm2, self.q, self.p_hat
        t, bd = self.tau, int(b)
        S = q * (P0 + P1)
        ps = (float(P0), float(P1))
        nxt = tuple(ps[i] + sw2[i] for i in range(len(sw2)))
        out: dict[str, float] = {"HOLD": S + self._V(t - 1, nxt[0], nxt[1], bd)}
        if bd >= 1:
            for i in range(len(sw2)):
                g = ps[i] / (ps[i] + sm2)
                d = (1.0 - g) * ps[i] + sw2[i]
                nd = list(nxt)
                nd[i] = d
                out[f"SEND{i}"] = (S
                                   + ph[i] * self._V(t - 1, nd[0], nd[1], bd - 1)
                                   + (1.0 - ph[i]) * self._V(t - 1, nxt[0], nxt[1], bd - 1))
        else:
            out["SEND0"] = out["SEND1"] = math.inf
        self.n_states = self._V.cache_info().currsize
        return out


def argmin3(vals: dict[str, float], budget: int = 1) -> str:
    """固定序 `HOLD < SEND0 < SEND1` 的 argmin（与 `_pick` 的 tie-break 同向）。

    ★ **可行性必须先筛**（闸 `C6` 首跑抓到的自伤）：`_expect_after` 在 `b=0` 时把
      `bud_after = max(0−1, 0) = 0`，于是 SEND 看起来"免费且刷新"⇒ 排序器会选它。
      生产的 `_pick` 靠 `budget_left >= need` 先筛掉不可行动作，所以生产是对的；
      本函数的 argmin 也必须筛，否则量的是我自己漏掉的一步。
    """
    feas = [k for k in ORDER if (int(budget) >= 1 or k == "HOLD") and np.isfinite(vals[k])]
    if not feas:
        return "HOLD"
    best = min(vals[k] for k in feas)
    for k in feas:
        if vals[k] <= best + _TIE_EPS:
            return k
    return feas[0]


# ==================================================================== 排序器（解析、只读）
def invert_p_hat(pr: PlanRef, target: float) -> float:
    """反解 `m_h` 使 `_p_hat(m_h, v_h=0) = target`（`vh=0` 时求积退化为 `sigmoid`）。

    用 200 次二分（`_p_hat` 对 `m_h` 严格单调）。★ 不假设"求积归一化恰好等于 1"：
    反过来把 `_p_hat` 的实际输出当基准（闸 `C2` 核残差 ≤1e-12）。
    """
    lo, hi = -40.0, 40.0
    z = np.zeros((1, 1))
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if float(pr._p_hat(np.array([[mid]]), z)[0, 0]) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def make_ranker(cfg, p_hat_val: float, tail: str, tau_max: int) -> tuple[PlanRef, float]:
    """构造一个**只读**排序器：返回 `(PlanRef, m_h)`；`p̂` 由 `_p_hat` 反解后**逐位一致**。"""
    ph = [float(p_hat_val)] * int(cfg.plant.n_users)
    pr = PlanRef(cfg, h_plan=5, m_samples=1, tail=tail, pacing=True, cont_m=True,
                 dp_phat=(ph if tail == "dp" else None),
                 dp_grid=("r173" if tail == "dp" else None))
    mh = invert_p_hat(pr, float(p_hat_val))
    p_real = float(pr._p_hat(np.array([[mh]]), np.zeros((1, 1)))[0, 0])
    if tail == "dp":
        # ★ 直接用 DPTail 构造（不经 `get_dp`）：`get_dp` 会把 `p̂` 量化到 0.02，
        #   那样两侧的 `p̂` 就不是同一个数了。`tau_max` 只截到本轮需要的层数
        #   （闸 `C4` 核"截断不改用到的层"）。
        pr._dp = DPTail(cfg, ph, b_cap=int(tau_max), tau_max=int(tau_max),
                        grid="r173").build()
    return pr, mh


def ranker_values(pr: PlanRef, mh: float, P0: float, P1: float, b: float,
                  steps_after: int, kinds: tuple[str, ...] = ORDER) -> dict[str, float]:
    """`_expect_after` 的三动作值（**不含**本步即时代价 —— 它不随动作变）。"""
    M = 1
    st = _St(np.zeros((M, 2)),
             np.array([[float(P0), float(P1)]]),
             np.array([[float(mh), float(mh)]]),
             np.zeros((M, 2)), np.zeros((M, 2)), np.zeros((M, 2)),
             np.array([float(b)]))
    out: dict[str, float] = {}
    for k in kinds:
        act = Action("HOLD") if k == "HOLD" else Action("SEND", int(k[-1]))
        out[k] = float(pr._expect_after(st, act, int(steps_after))[0])
    return out


# ==================================================================== J1/J2 面板
def panel_j1(cfg) -> dict:
    """逐 (τ, p̂, P0, P1, b) 比两套解析排序器的 argmin 与精确 Q 的 argmin。"""
    rows: list[dict] = []
    t_all = time.perf_counter()
    for tau in TAUS:
        b_list = sorted({0, 2, tau // 2})
        for phat in PHATS:
            ex = ExactQ(cfg, [phat] * int(cfg.plant.n_users)).solve(tau)
            rankers = {name: make_ranker(cfg, phat, tail, tau)
                       for name, tail in (("paced", "paced"), ("dp_r173", "dp"))}
            for P0, P1 in product(PS, PS):
                for b in b_list:
                    qe = ex.q_values(P0, P1, b)
                    vstar = min(qe.values())
                    a_star = argmin3(qe, budget=int(b))
                    rec = {"tau": int(tau), "p_hat": float(phat), "P0": float(P0),
                           "P1": float(P1), "b": int(b), "Vstar": vstar,
                           "a_star": a_star,
                           "Q_exact": {k: qe[k] for k in ORDER},
                           "rankers": {}}
                    for name, (pr, mh) in rankers.items():
                        vr = ranker_values(pr, mh, P0, P1, b, tau - 1)
                        a_r = argmin3(vr, budget=int(b))
                        # 解析值是"动作之后的尾部"，精确 Q 含本步即时代价 ⇒ 加同一个常数再比
                        S = cfg.plant.q_cost * (P0 + P1)
                        vr_full = {k: vr[k] + S for k in ORDER}
                        rec["rankers"][name] = {
                            "a": a_r, "agree": bool(a_r == a_star),
                            "Q_rank": {k: vr_full[k] for k in ORDER},
                            "regret_abs": float(qe[a_r] - vstar),
                            "regret_rel": float((qe[a_r] - vstar) / vstar) if vstar > 0 else None,
                            "budget_used": float(b),
                        }
                    # 描述性：含 PROBE 的 5 动作集里 PROBE 是否被选（J0：不入判词）
                    prp, mhp = rankers["paced"]
                    v5 = ranker_values(prp, mhp, P0, P1, b, tau - 1,
                                       kinds=("HOLD", "SEND0", "SEND1"))
                    M = 1
                    st = _St(np.zeros((M, 2)), np.array([[float(P0), float(P1)]]),
                             np.array([[float(mhp), float(mhp)]]), np.zeros((M, 2)),
                             np.zeros((M, 2)), np.zeros((M, 2)), np.array([float(b)]))
                    if b >= 1:
                        for u in (0, 1):
                            v5[f"PROBE{u}"] = float(
                                prp._expect_after(st, Action("PROBE", u), tau - 1)[0])
                    rec["probe_pref"] = bool(b >= 1 and min(
                        v5[f"PROBE{u}"] for u in (0, 1)) <= min(v5[k] for k in ORDER) - 1e-9)
                    rows.append(rec)
            del rankers, ex
            gc.collect()
    return {"rows": rows, "seconds": time.perf_counter() - t_all}


# ==================================================================== J3 规模
def panel_j3(cfg) -> dict:
    """`ExactClosedLoop`（真值）与 `DPTail`（近似）的**实测**规模标度。"""
    ex_pts: list[dict] = []
    for tau in (4, 6, 8, 10, 12, 14):
        b = max(1, tau // 2)
        t0 = time.perf_counter()
        ex = ExactQ(cfg, [0.5] * int(cfg.plant.n_users)).solve(tau)
        v = ex.value(0.01, 0.01, b)
        dt = time.perf_counter() - t0
        npair = 3 ** tau
        ex_pts.append({"tau": int(tau), "b": int(b), "V": v,
                       "n_states": int(ex.n_states), "seconds": round(dt, 3),
                       "pairs_bound": int(npair), "states_per_pair": round(ex.n_states / npair, 3)})
        del ex
        gc.collect()
        if dt > J3_MAX_SECONDS or ex_pts[-1]["n_states"] > 4 * J3_MAX_STATES:
            ex_pts[-1]["truncated_after"] = True
            break

    dp_pts: list[dict] = []
    for tau_max in (10, 32):
        for grid in ("legacy", "r173"):
            t0 = time.perf_counter()
            dp = DPTail(cfg, [0.5, 0.5], b_cap=tau_max, tau_max=tau_max, grid=grid).build()
            dt = time.perf_counter() - t0
            dp_pts.append({"tau_max": int(tau_max), "grid": grid,
                           "n_knots": int(dp.knots.size), "mem_mib": round(dp.mem_mib(), 2),
                           "seconds": round(dt, 3), "n_b": int(dp.nb_max)})
            del dp
            gc.collect()

    # τ=256 的**结构外推**（不是实测）：分支因子 3（HOLD / SEND0 / SEND1）、预算轴 b+1
    grow = 3.0
    tmax = max(p["tau"] for p in ex_pts)
    return {"exact": ex_pts, "dp": dp_pts,
            "extrapolation": {"branch_factor": grow, "max_tau_computed": int(tmax),
                              "note": "★ 外推非实测：ExactClosedLoop 的 (P0,P1) 状态数按每步 3 分支增长",
                              "log10_ratio_to_panel": round((256 - tmax) * math.log10(grow), 3)}}


# ==================================================================== 判词
def _label(med: float, agree: float) -> str:
    """预注册 §2.4 的**原样**判据（一字未改；闭合率口径的漏洞只影响 overall，不影响它）。"""
    if med <= THRESH_NB_REGRET and agree >= AGREE_OK:
        return "NOT-BOTTLENECK"
    if med >= THRESH_B_REGRET or agree <= AGREE_BAD:
        return "IS-BOTTLENECK"
    return "INCONCLUSIVE-BY-MARGIN"


def _stats(rows: list[dict], name: str) -> dict:
    import statistics as st
    rel = [r["rankers"][name]["regret_rel"] for r in rows
           if r["rankers"][name]["regret_rel"] is not None]
    agree = [1.0 if r["rankers"][name]["agree"] else 0.0 for r in rows]
    dis = [r["rankers"][name]["regret_rel"] for r in rows if not r["rankers"][name]["agree"]]
    arr = np.asarray(rel, dtype=float)
    return {"n": int(len(rel)),
            "median_regret_rel": float(st.median(rel)),
            "mean_regret_rel": float(np.mean(arr)),
            "p90_regret_rel": float(np.percentile(arr, 90)),
            "max_regret_rel": float(np.max(arr)),
            "share_ge_1pct": float(np.mean(arr >= 1e-2)),
            "share_ge_5pct": float(np.mean(arr >= 5e-2)),
            "zero_share": float(np.mean(arr == 0.0)),
            "agree": float(np.mean(agree)),
            "n_disagree": int(len(dis)),
            "median_regret_when_disagree": (float(st.median(dis)) if dis else None)}


def judge(panel: dict, scal: dict) -> dict:
    out: dict = {}
    for name in ("paced", "dp_r173"):
        st_ = _stats(panel["rows"], name)
        st_["label"] = _label(st_["median_regret_rel"], st_["agree"])
        out[name] = st_

    # ---------------- 闭合率：预注册口径（中位比）＋ 跑后修订口径（均值比 / 同意率补集）
    mp, md = out["paced"]["median_regret_rel"], out["dp_r173"]["median_regret_rel"]
    if mp >= THRESH_NB_REGRET and mp > 0:
        c = 1.0 - md / mp
        out["closure_prereg"] = {
            "value": float(c),
            "verdict": ("CLOSABLE" if c >= CLOSE_OK else "PERSISTS" if c < CLOSE_BAD else "PARTIAL")}
    else:
        out["closure_prereg"] = {
            "value": None, "verdict": "UNDEFINED",
            "why": ("median(regret_paced)==0.0。★ 这是**形式缺陷**，不是效应小："
                    "只要排序器在 >50% 的状态上与最优同解，zero-inflated 变量的中位必为 0，"
                    "分母恒为 0 —— 与 R2.4「只用中位」同族。见预注册 §7-2。")}
    mup, mud = out["paced"]["mean_regret_rel"], out["dp_r173"]["mean_regret_rel"]
    ap, ad = out["paced"]["agree"], out["dp_r173"]["agree"]
    c_mean = 1.0 - (mud / mup) if mup > 0 else None
    c_agree = 1.0 - ((1.0 - ad) / (1.0 - ap)) if ap < 1.0 else None
    cand = [x for x in (c_mean, c_agree) if x is not None]
    cmin = min(cand) if cand else None
    out["closure_revised"] = {
        "by_mean": c_mean, "by_agree": c_agree,
        "value": cmin,
        "verdict": ("CLOSABLE" if cmin is not None and cmin >= CLOSE_OK
                    else "PERSISTS" if cmin is not None and cmin < CLOSE_BAD
                    else "PARTIAL" if cmin is not None else "UNDEFINED"),
        "note": "两个口径都报，取**较小者**（保守）；均为跑后定义，见预注册 §7-2"}

    def _overall(closure_key: str) -> str:
        if out["paced"]["label"] == "NOT-BOTTLENECK":
            return "NO-WINNABLE-OPPONENT-BY-RANKING"
        v = out[closure_key]["verdict"]
        if v == "CLOSABLE":
            return "NO-WINNABLE-OPPONENT-BY-RANKING"
        if v == "PERSISTS":
            return "RANKING-OFFERS-VERIFIABLE-GAP"
        return "INCONCLUSIVE-BY-MARGIN"

    out["overall_prereg"] = _overall("closure_prereg")
    out["overall"] = _overall("closure_revised")      # ★ 对外用这一条（与 R2.8 的双标签同一手法）

    # ---------------- J3 附判：按预注册 §3-J3 的写法，**在 τ=10 上**判
    t10 = next((p for p in scal["exact"] if p["tau"] == 10), None)
    big = max(scal["exact"], key=lambda x: x["tau"])
    ok10 = (t10 is not None and t10["n_states"] <= J3_MAX_STATES
            and t10["seconds"] <= J3_MAX_SECONDS)
    out["truth_window"] = {
        "tau10_states": None if t10 is None else int(t10["n_states"]),
        "tau10_seconds": None if t10 is None else float(t10["seconds"]),
        "max_tau_computed": int(big["tau"]),
        "max_tau_states": int(big["n_states"]),
        "max_tau_seconds": float(big["seconds"]),
        "log10_ratio_to_panel": scal["extrapolation"]["log10_ratio_to_panel"],
        "panel_tau": 256,
        "verdict": ("TRUTH-OK-AT-TAU10-PANEL-INFEASIBLE" if ok10
                    else "TRUTH-INFEASIBLE-AT-TAU10"),
        "why": ("τ=10 的真值可算（状态数/耗时都在限内）⇒ **可验证窗口**存在；"
                "但 τ=256 的状态数按 3^Δτ 增长（见 log10_ratio_to_panel）⇒ "
                "**面板尺度上没有真值**，任何在 τ=256 上的'学习更好'都只能对基线验证，不能对最优验证。")}
    return out


# ==================================================================== 闸
def _probe_not_in_code(path: Path, names: tuple[str, ...] = ("PROBE",)) -> dict:
    """AST：`PROBE` 是否只出现在 docstring/注释里（注释不在 AST 中，只需排 docstring）。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docs = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            d = ast.get_docstring(node, clean=False)
            if d is not None:
                docs.add(d)
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value in docs:
                continue
            for nm in names:
                if nm in node.value:
                    hits.append(f"{path.name}:{node.lineno}")
    return {"file": path.name, "hits": hits}


def run_gates(cfg) -> tuple[dict, dict]:
    gates: dict[str, dict] = {}

    def gate(name: str, ok: bool, **kw):
        gates[name] = {"pass": bool(ok), **kw}

    # ---------------- C0：A≡1（否则精确参考不成立）
    gate("C0", all(float(a) == 1.0 for a in cfg.plant.A),
         A=[float(a) for a in cfg.plant.A], need="A≡1")

    # ---------------- C1：独立重写 == ExactClosedLoop（**逐位**）
    cases = [(2, 0.01, 0.02, 1), (3, 0.005, 0.05, 1), (4, 0.02, 0.01, 2),
             (4, 0.05, 0.05, 0), (6, 0.01, 0.01, 3), (6, 0.005, 0.02, 1),
             (8, 0.02, 0.005, 4), (10, 0.01, 0.01, 5)]
    diffs = []
    for tau, p0, p1, b in cases:
        mine = ExactQ(cfg, [0.5, 0.5]).solve(tau).value(p0, p1, b)
        ref = ExactClosedLoop(cfg, [0.5, 0.5]).value(tau, p0, p1, b)
        diffs.append(abs(mine - ref))
    gate("C1", max(diffs) == 0.0, max_abs_diff=max(diffs), n=len(cases), need="==0.0（逐位）")

    # ---------------- C1b：分辨力（负控）——把送达更新换成"不加 g"，必须不同
    ex = ExactQ(cfg, [0.5, 0.5]).solve(6)
    v_ok = ex.value(0.01, 0.02, 3)
    sw2, sm2, q = ex.sw2, ex.sm2, ex.q

    @lru_cache(maxsize=None)
    def V_bad(t, p0, p1, bd):
        if t == 0:
            return 0.0
        ps = (p0, p1)
        nxt = tuple(ps[i] + sw2[i] for i in range(2))
        best = V_bad(t - 1, nxt[0], nxt[1], bd)
        if bd >= 1:
            for i in range(2):
                # ★ 变异：漏掉 (1−g) 因子（等价"量测没降方差"）
                d = ps[i] + sw2[i]
                nd = list(nxt)
                nd[i] = d
                v = (0.5 * V_bad(t - 1, nd[0], nd[1], bd - 1)
                     + 0.5 * V_bad(t - 1, nxt[0], nxt[1], bd - 1))
                if v < best:
                    best = v
        return q * (p0 + p1) + best

    v_bad = V_bad(6, 0.01, 0.02, 3)
    gate("C1b", abs(v_bad - v_ok) > 1e-6, v_ok=v_ok, v_bad=v_bad, need=">1e-6")

    # ---------------- C2：p̂ 反解
    resid = []
    for target in PHATS:
        pr, mh = make_ranker(cfg, target, "paced", 10)
        got = float(pr._p_hat(np.array([[mh]]), np.zeros((1, 1)))[0, 0])
        resid.append(abs(got - target))
    gate("C2", max(resid) <= 1e-12, max_resid=max(resid), need="<=1e-12")

    # ---------------- C3：结构不变量（单调性）
    ex3 = ExactQ(cfg, [0.5, 0.5]).solve(6)
    mono_p = ex3.value(0.02, 0.01, 3) > ex3.value(0.01, 0.01, 3)
    mono_b = ex3.value(0.01, 0.02, 3) < ex3.value(0.01, 0.02, 1)
    gate("C3", bool(mono_p and mono_b), mono_in_P=bool(mono_p), mono_in_b=bool(mono_b),
         need="P↑⇒V↑ 且 b↑⇒V↓")

    # ---------------- C4：DP 截断不改用到的层
    a = DPTail(cfg, [0.5, 0.5], b_cap=10, tau_max=10, grid="r173").build()
    b10 = DPTail(cfg, [0.5, 0.5], b_cap=12, tau_max=12, grid="r173").build()
    d4 = float(np.max(np.abs(a._V[10] - b10._V[10])))
    gate("C4", d4 == 0.0, max_abs_diff=d4, need="==0.0（逐位）")

    # ---------------- C5：PROBE 不在代码路径里（J0 的结构依据）
    hits = []
    for fn in ("plan_dp.py", "dp_audit.py"):
        r = _probe_not_in_code(REPO / "experiments" / "cw_v1" / fn)
        hits.extend(r["hits"])
    gate("C5", len(hits) == 0, hits=hits, need="代码字符串里不出现 PROBE（只在 docstring）")

    # ---------------- C6：我的 argmin == `_pick`（生产排序）
    pr_np = PlanRef(cfg, h_plan=5, m_samples=1, use_probe=False, tail="paced",
                    pacing=True, cont_m=True)
    mh = invert_p_hat(pr_np, 0.5)
    bad = []
    for P0, P1, b, tau in [(0.01, 0.02, 3, 6), (0.005, 0.05, 5, 10), (0.02, 0.02, 0, 6)]:
        vr = ranker_values(pr_np, mh, P0, P1, b, tau - 1)
        mine = argmin3(vr, budget=int(b))
        picks = pr_np._pick({Action("HOLD"): vr["HOLD"],
                             Action("SEND", 0): vr["SEND0"],
                             Action("SEND", 1): vr["SEND1"]}, budget_left=int(b))
        theirs = "HOLD" if picks.kind == "HOLD" else f"SEND{int(picks.user)}"
        if mine != theirs:
            bad.append({"P0": P0, "P1": P1, "b": b, "mine": mine, "pick": theirs})
    gate("C6", len(bad) == 0, mismatches=bad, n=3, need="3/3 一致")

    # ---------------- C7：变异测试——把排序反向，遗憾必须变坏
    pr, mh_ = make_ranker(cfg, 0.5, "paced", 10)
    ex7 = ExactQ(cfg, [0.5, 0.5]).solve(10)
    rev_reg, ok_reg = [], []
    for P0, P1 in [(0.005, 0.05), (0.02, 0.01), (0.01, 0.01), (0.05, 0.005)]:
        qe = ex7.q_values(P0, P1, 5)
        vstar = min(qe.values())
        vr = ranker_values(pr, mh_, P0, P1, 5, 9)
        a_ok = argmin3(vr, budget=5)
        a_bad = max(ORDER, key=lambda k: vr[k])
        ok_reg.append((qe[a_ok] - vstar) / vstar)
        rev_reg.append((qe[a_bad] - vstar) / vstar)
    gate("C7", float(np.median(rev_reg)) >= THRESH_B_REGRET,
         median_ok=float(np.median(ok_reg)), median_reversed=float(np.median(rev_reg)),
         ratio=float(np.median(rev_reg) / max(np.median(ok_reg), 1e-12)),
         need="反向排序的中位遗憾 ≥ 5e-2（= `IS-BOTTLENECK` 门槛）⇒ 指标在该量级有分辨力")

    return gates, {"C1_cases": len(cases)}


# ==================================================================== 主函数
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="CW-R3.0 立项卡：排序遗憾 vs 精确最优（零 episode）")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--ctrl-only", action="store_true")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    srchash = _src_hash()
    cfg = mainline_config()
    gates, gmeta = run_gates(cfg)

    if args.ctrl_only:
        n_fail = sum(1 for g in gates.values() if not g["pass"])
        print(json.dumps({"stage": "ctrl", "gates": gates, "n_fail": n_fail},
                         ensure_ascii=False, indent=2))
        return 0 if n_fail == 0 else 1

    outdir = Path(args.outdir) if args.outdir else RESULTS / f"cw_r3_0_card_{_now_utc()}_{srchash}"
    outdir.mkdir(parents=True, exist_ok=False)

    summary: dict = {"round": ROUND, "protocol": PROTOCOL, "src_hash": srchash,
                     "plant": {"A": [float(x) for x in cfg.plant.A], "B": float(cfg.plant.B),
                               "sigma_w": [float(x) for x in cfg.plant.sigma_w],
                               "sigma_meas": float(cfg.plant.sigma_meas),
                               "q": float(cfg.plant.q_cost), "r": float(cfg.plant.r_cost)},
                     "grid": {"taus": list(TAUS), "P": list(PS), "p_hat": list(PHATS)},
                     "thresholds": {"NB_regret": THRESH_NB_REGRET, "B_regret": THRESH_B_REGRET,
                                    "agree_ok": AGREE_OK, "agree_bad": AGREE_BAD,
                                    "close_ok": CLOSE_OK, "close_bad": CLOSE_BAD}}

    t1 = time.perf_counter()
    panel = panel_j1(cfg)
    t1 = time.perf_counter() - t1
    (outdir / "panel_j1.json").write_text(
        json.dumps(panel, ensure_ascii=False, indent=2), "utf-8")

    t2 = time.perf_counter()
    scal = panel_j3(cfg)
    t2 = time.perf_counter() - t2
    (outdir / "scaling.json").write_text(json.dumps(scal, ensure_ascii=False, indent=2), "utf-8")

    j = judge(panel, scal)

    # ---------------- 闸（跑后追加）
    gates["C8"] = {"pass": panel["seconds"] < 1500.0, "seconds": round(panel["seconds"], 2),
                   "need": "<1500 s（预注册 §5 上界 25 min）"}
    gates["C9"] = {"pass": all(math.isfinite(r["Vstar"]) and r["Vstar"] > 0 for r in panel["rows"]),
                   "n": len(panel["rows"]), "need": "V* 有限且 >0"}
    gates["C10"] = {"pass": j["truth_window"]["verdict"].startswith("TRUTH-"),
                    "verdict": j["truth_window"]["verdict"]}
    gates["C11"] = {"pass": j["overall"] in (
        "NO-WINNABLE-OPPONENT-BY-RANKING", "RANKING-OFFERS-VERIFIABLE-GAP",
        "INCONCLUSIVE-BY-MARGIN"), "overall": j["overall"]}

    n_fail = sum(1 for g in gates.values() if not g["pass"])
    (outdir / "gates.json").write_text(json.dumps(gates, ensure_ascii=False, indent=2), "utf-8")

    summary.update({"judge": j, "n_fail": n_fail, "seconds_panel": round(t1, 2),
                    "seconds_scaling": round(t2, 2),
                    "seconds_total": round(time.perf_counter() - t0, 2),
                    "run_episode": 0, "train_steps": 0, "n_states_rows": len(panel["rows"])})
    (outdir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), "utf-8")
    keep = ("median_regret_rel", "mean_regret_rel", "p90_regret_rel", "max_regret_rel",
            "share_ge_1pct", "share_ge_5pct", "agree", "n_disagree",
            "median_regret_when_disagree", "label", "n")
    print(json.dumps({"outdir": str(outdir), "n_fail": n_fail,
                      "overall": j["overall"], "overall_prereg": j["overall_prereg"],
                      "paced": {k: j["paced"][k] for k in keep},
                      "dp_r173": {k: j["dp_r173"][k] for k in keep},
                      "closure_prereg": j["closure_prereg"],
                      "closure_revised": j["closure_revised"],
                      "truth_window": j["truth_window"],
                      "seconds": summary["seconds_total"]},
                     ensure_ascii=False, indent=2))
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
