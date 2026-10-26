# -*- coding: utf-8 -*-
"""`CW-R5.0` 控制件：免费探测臂的**一次探测上限**对照。

预注册：`docs/research/CW-R5.0-一次探测上限-预注册-v0.1.md`（写于正式读数之前）

为什么要做
----------
R4.2 的 `free` 臂 `probe_cost=0`，而动作空间是**每槽至多一个**资源动作
（`PROBE_i` / `SEND_i` / `HOLD`）⇒ 一个 τ 槽窗口里最多探测 `τ−1` 次（`t≥2` 才有意义）。
所以 `IV_free = A0 − V_free` 度量的是「**无限次免费探测**」的价值，而 R4.2 把它读成
「**一次**读数服务了多次发送」。这是**读解**，不是实测。验收报告 2026-10-26 §4.5 点名这一条。

本文件只做一件事：在**同一递推、同一世界、同一动作集**上加一个探测计数器 `k`，
令 `PROBE` 只在 `k < probe_max` 时可用，得到 `IV_free(k)`。于是

    m_1 = IV_free(1) − IV_free(0)        ← 「一次读数」的价值
    m_2 = IV_free(2) − IV_free(1)        ← 第二次读数的边际价值

`k=0` 必须逐位等于 `probe_enabled=False`；`k=∞` 必须逐位等于 `build_engine_n(probe_cost=0)`。
两条都过，才允许解释 `m_k`。

不改被测物：`r4_0_boundary.py` / `r3_2_iv2u.py` / `plan_dp.py` 一字节不动，只 `import`。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

from .config import ChannelConfig, PlantConfig, mainline_config
from .posterior import p_succ_expected
from .r1_14_exactbase import _now_utc, _src_hash
from .r3_2_iv2u import GH_N, M_RANGE, STATE_CAP
from .r4_0_boundary import build_engine_n

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results" / "cw_v1"

PROTOCOL = "CW-POMDP-v1.1"
ROUND = "CW-R5.0"
SIGMA_STAR = 1.0e-9
SIGMA_W0 = 0.02
ALPHA = 2.5
P_VALUE = 0.01
KAPPA = 1.0
N_USERS = 2
RHO = 0.999
V_PRIOR = 1.0

#: 三格（预注册 §4）：`(tau, b, n_m)`
CELLS: tuple[tuple[int, int, int], ...] = ((7, 2, 33), (6, 2, 33), (8, 3, 17))
#: 探测上限（`None` = 无限，等价既有 free 臂）
CAPS: tuple[int | None, ...] = (None, 1, 2, 3)

TOL_IDENT = 0.0          # 逐位
TOL_MONO = 1e-15
TOL_ANCHOR = 1e-12
_SQRT2PI = math.sqrt(2.0 * math.pi)


def _plant_asym(alpha: float) -> PlantConfig:
    base = mainline_config().plant
    return PlantConfig(**{**base.__dict__, "sigma_w": (SIGMA_W0, SIGMA_W0 * float(alpha))})


def rho_config(rho: float = RHO):
    plant = _plant_asym(ALPHA)
    ch = ChannelConfig(**{**mainline_config().channel.__dict__, "rho_panel": (float(rho),)})
    return mainline_config(plant=plant, channel=ch)


# ==================================================================== 带探测计数的 DP
def build_engine_cap(cfg, sigma_csi: float, n_m: int, *, probe_max: int | None,
                     probe_cost: int = 0, gh_n: int = GH_N, n_users: int = N_USERS,
                     state_cap: int = STATE_CAP):
    """`build_engine_n` 的**加法版**：多一个探测计数 `k`，`PROBE` 仅当 `k < probe_max`。

    `probe_max=None` ⇒ 与 `build_engine_n(probe_cost=0)` 应**逐位相同**（`C-1` 验）。
    递推、插值、映射、GH 节点、量化 `snap()` 全部逐字照搬，只增一个整数维度。
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
    pc = int(probe_cost)
    nu = int(n_users)
    k_max = None if probe_max is None else int(probe_max)
    assert len(a2) >= nu and len(sw2) >= nu, "plant 的用户数不够"

    @lru_cache(maxsize=None)
    def snap(m: float) -> float:
        return centers_l[int(np.argmin(np.abs(centers - float(m))))]

    @lru_cache(maxsize=None)
    def ps(m: float, v: float) -> float:
        return p_succ_expected(float(m), float(v), cfg)

    memo: dict[tuple, float] = {}
    state = {"capped": False}

    def V(t: int, Pp: tuple, mm: tuple, vv: tuple, b: int, k: int) -> float:
        if t <= 0:
            return 0.0
        key = (t, Pp, mm, vv, b, k)
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

        best = V(t - 1, ff, mp, vp0, b, k)                  # HOLD

        if b >= 1:                                          # SEND_i
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
                va = ph * V(t - 1, tuple(aa), mp, vp0, b - 1, k)
                vb = (1.0 - ph) * V(t - 1, tuple(bb), mp, vp0, b - 1, k)
                val = va + vb
                if val < best:
                    best = val

        allowed = k_max is None or k < k_max
        if allowed and b >= pc and t >= 2:                  # PROBE_i
            for i in range(nu):
                mi = mm[i]
                vi = vv[i]
                gp = vi / (vi + si2)
                vpi = rho2 * (vi * si2 / (vi + si2)) + infl
                sy = math.sqrt(vi + si2)
                tot = 0.0
                for z, w in zip(nodes, weights):
                    mm2 = list(mp)
                    mm2[i] = rho * snap(mi + gp * sy * z)
                    vv2 = list(vp0)
                    vv2[i] = vpi
                    tot += w * V(t - 1, ff, tuple(mm2), tuple(vv2), b - pc, k + 1)
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
                           tuple(float(x) for x in vvec), int(b), 0))
        except MemoryError:
            state["capped"] = True
            return float("nan")

    def n_states() -> int:
        return len(memo)

    value.n_states = n_states                # type: ignore[attr-defined]
    value.capped = lambda: state["capped"]   # type: ignore[attr-defined]
    value.n_users = nu                       # type: ignore[attr-defined]
    return value


# ==================================================================== 读数
def read_cell(cfg, tau: int, b: int, n_m: int, probe_max: int | None) -> dict:
    t0 = time.perf_counter()
    val = build_engine_cap(cfg, SIGMA_STAR, n_m, probe_max=probe_max)
    v = val(tau, (P_VALUE, P_VALUE), (0.0, 0.0), (V_PRIOR, V_PRIOR), b)
    return {"tau": int(tau), "b": int(b), "n_m": int(n_m), "probe_max": probe_max,
            "V": float(v), "n_states": int(val.n_states()), "capped": bool(val.capped()),
            "sec": round(time.perf_counter() - t0, 3)}


def anchor_from_r42() -> dict:
    """读 R4.2 已提交证据中的三格（同 `ρ=0.999, α=2.5, P=0.01`）。"""
    cand = sorted(RESULTS.glob("cw_r4_2_multisend_*/r42.json"))
    cand = [p for p in cand if "_SUPERSEDED" not in str(p)]
    if not cand:
        return {}
    d = json.loads(cand[-1].read_text(encoding="utf-8"))
    out = {}
    for c in d["cells"]:
        if abs(float(c["rho"]) - RHO) < 1e-12 and abs(float(c["P"]) - P_VALUE) < 1e-12:
            out[f'{int(c["tau"])},{int(c["b"])},{int(c["n_m"])}'] = {
                "A0": float(c["A0"]), "V_free": float(c["V_free"]),
                "IV_free": float(c["IV_free"]), "ratio_free": c["ratio_free"]}
    return {"path": str(cand[-1].relative_to(REPO)), "cells": out}


def panel() -> dict:
    cfg = rho_config(RHO)
    rows: list[dict] = []
    for (tau, b, n_m) in CELLS:
        # `A0` 用该格自己的 `n_m`（`noprobe` 臂与 `n_m` 无关，但保持同参调用）
        npb = build_engine_n(cfg, SIGMA_STAR, n_m, probe_enabled=False, n_users=N_USERS)
        noprobe_v = npb(tau, (P_VALUE, P_VALUE), (0.0, 0.0), (V_PRIOR, V_PRIOR), b)
        a0m = npb(tau, (P_VALUE, P_VALUE), (0.0, 0.0), (V_PRIOR, V_PRIOR), b - 1)
        dq = float(a0m - noprobe_v)
        for cap in CAPS:
            r = read_cell(cfg, tau, b, n_m, cap)
            r["A0"] = float(noprobe_v)
            r["dV_q"] = dq
            if math.isfinite(r["V"]) and dq > 1e-12:
                r["IV"] = float(noprobe_v - r["V"])
                r["ratio"] = float(r["IV"] / dq)
            else:
                r["IV"] = None
                r["ratio"] = None
            rows.append(r)
            # ★ `probe_max=None` 在我的实现里也要多带一维 `k` ⇒ 状态数约为既有 free 臂的 `τ` 倍，
            #   `(τ=8,b=3,n_m=17)` 会撞 `6e6`。撞顶时**不放弃该格**：`∞` 值改用 R4.2 已提交读数
            #   （同一对象，`C-1` 在另两格已逐位验证），再继续跑 `k=1,2,3`。
            if r["capped"] and cap is not None:
                break
    return {"cells": rows}


def judge(p: dict, anchor: dict) -> dict:
    rows = p["cells"]
    by = {(r["tau"], r["b"], r["n_m"], r["probe_max"]): r for r in rows}

    # `C-1` 无限上限 vs 既有 free 臂（用已提交 R4.2 值作锚）
    ident, ident_bad = [], []
    for key, ref in (anchor.get("cells") or {}).items():
        tau, b, n_m = (int(x) for x in key.split(","))
        r = by.get((tau, b, n_m, None))
        if r is None or not math.isfinite(r["V"]):
            continue
        d = abs(r["V"] - ref["V_free"])
        rec = {"cell": [tau, b, n_m], "ours": r["V"], "r42": ref["V_free"], "diff": d}
        (ident if d <= TOL_IDENT else ident_bad).append(rec)
    # `A0` 锚
    a0_bad = []
    for key, ref in (anchor.get("cells") or {}).items():
        tau, b, n_m = (int(x) for x in key.split(","))
        r = by.get((tau, b, n_m, None))
        if r is None:
            continue
        d = abs(r["A0"] - ref["A0"])
        if d > TOL_ANCHOR:
            a0_bad.append({"cell": [tau, b, n_m], "ours": r["A0"], "r42": ref["A0"], "diff": d})

    # `C-2` k=0 分支：`probe_max=0` 未在本面板跑，改用 `IV(∞)` 与 `noprobe` 的 `A0` 关系间接验
    #        （`probe_max=0` 与 `probe_enabled=False` 递推不同路径，单独跑一格更硬）
    cfg = rho_config(RHO)
    tau0, b0, nm0 = CELLS[0]
    val0 = build_engine_cap(cfg, SIGMA_STAR, nm0, probe_max=0)
    v0 = val0(tau0, (P_VALUE, P_VALUE), (0.0, 0.0), (V_PRIOR, V_PRIOR), b0)
    a0_ref = by[(tau0, b0, nm0, None)]["A0"]
    c2 = {"ours": float(v0), "A0": float(a0_ref), "diff": abs(float(v0) - a0_ref)}

    # `C-3` 单调性 + `C-4` 上限是否 binding
    mono_pre, mono_post, binding, curves = [], [], [], {}
    for (tau, b, n_m) in CELLS:
        key = f"{tau},{b},{n_m}"
        seq = [(r["probe_max"], r["IV"]) for r in rows
               if (r["tau"], r["b"], r["n_m"]) == (tau, b, n_m) and r["IV"] is not None]
        # `∞` 未测到（撞顶）⇒ 用 R4.2 已提交读数补齐，并记 `inf_source`
        ref = (anchor.get("cells") or {}).get(key)
        if not any(k is None for k, _ in seq) and ref is not None and ref.get("IV_free") is not None:
            seq = [(None, float(ref["IV_free"]))] + seq
            inf_source = "r42"
        else:
            inf_source = "ours"
        # ★ 顺序必须放在补 `∞` **之后**：`None` 是最大的 k，要排最后
        seq.sort(key=lambda z: (1, 0.0) if z[0] is None else (0, float(z[0])))
        curves[key] = {"inf_source": inf_source, "seq": [[k, v] for k, v in seq]}
        for (k1, v1), (k2, v2) in zip(seq, seq[1:]):
            # 预注册写的是「不增」（**方向写反**：选项集嵌套 ⇒ V 不增 ⇒ IV 不减）
            if v2 > v1 + TOL_MONO:
                mono_pre.append({"cell": [tau, b, n_m], "k1": k1, "k2": k2, "v1": v1, "v2": v2})
            # 理论正确方向：`IV(k)` 随 k **不递减**
            if v2 < v1 - TOL_MONO:
                mono_post.append({"cell": [tau, b, n_m], "k1": k1, "k2": k2, "v1": v1, "v2": v2})
        inf = next((v for k, v in seq if k is None), None)
        one = next((v for k, v in seq if k == 1), None)
        if inf is not None and one is not None:
            binding.append({"cell": [tau, b, n_m], "IV_inf": inf, "IV_1": one,
                            "gap": inf - one, "binding": (inf - one) > 1e-12,
                            "kept": (one / inf) if inf > 1e-12 else None})

    gates = {
        "C1_identity_inf_vs_r42": {"n_ok": len(ident), "n_bad": len(ident_bad), "bad": ident_bad},
        "C1b_anchor_A0": {"n_bad": len(a0_bad), "bad": a0_bad},
        "C2_probe_max_zero": {"diff": c2["diff"], "ok": c2["diff"] <= TOL_ANCHOR},
        "C3pre_monotone_as_preregistered": {"n_bad": len(mono_pre), "bad": mono_pre,
                                            "note": "预注册方向写反；本项预计 FAIL，属规格错误"},
        "C3b_monotone_corrected_posthoc": {"n_bad": len(mono_post), "bad": mono_post,
                                           "note": "跑后修正：选项集嵌套 ⇒ V 不增 ⇒ IV 不减"},
        "C4_cap_binds": {"cells": binding,
                         "n_binding": sum(1 for x in binding if x["binding"])},
    }
    n_fail_pre = (len(ident_bad) + len(a0_bad) + (0 if gates["C2_probe_max_zero"]["ok"] else 1)
                  + len(mono_pre) + (0 if gates["C4_cap_binds"]["n_binding"] > 0 else 1))
    n_fail = (len(ident_bad) + len(a0_bad) + (0 if gates["C2_probe_max_zero"]["ok"] else 1)
              + len(mono_post) + (0 if gates["C4_cap_binds"]["n_binding"] > 0 else 1))
    return {"gates": gates, "curves": curves, "n_fail": n_fail, "n_fail_prereg": n_fail_pre,
            "verdict": ("CAP-EXPERIMENT-VALID" if n_fail == 0 else "CAP-EXPERIMENT-INVALID"),
            "verdict_note": ("C3 方向为本人写反（规格错误，已披露）；结论只依赖 C1/C1b/C2/C3b/C4")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--rejudge", default=None,
                    help="只对既有 r50.json 的 panel 重算判据（不改原始读数），写出 r50_judge2.json")
    args = ap.parse_args()

    if args.rejudge:
        src = Path(args.rejudge)
        d = json.loads(src.read_text(encoding="utf-8"))
        j = judge(d["panel"], d["anchor"])
        out = src.parent / "r50_judge2.json"
        out.write_text(json.dumps({
            "round": ROUND, "protocol": PROTOCOL, "utc": _now_utc(),
            "note": "跑后重算判据：首版 `seq.sort` 把 `probe_max=None`（k=∞）排在最前，"
                    "导致单调性两两比较把 (∞,1) 当成一对 ⇒ C3b 假失败。**原始读数未改**，只改排序。",
            "source": str(src.name), "judge": j,
            "src_hash": _src_hash(),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"out": str(out), "n_fail": j["n_fail"],
                          "n_fail_prereg": j["n_fail_prereg"], "verdict": j["verdict"],
                          "curves": j["curves"]}, ensure_ascii=False, indent=2))
        return 0 if j["n_fail"] == 0 else 3

    utc = _now_utc()
    t0 = time.perf_counter()
    anchor = anchor_from_r42()
    p = panel()
    j = judge(p, anchor)
    payload = {
        "round": ROUND, "protocol": PROTOCOL, "utc": utc,
        "argv": ["experiments.cw_v1.r5_0_probcap"] + ([] if not args.out else ["--out", args.out]),
        "panel_meta": {"rho": RHO, "alpha": ALPHA, "P": P_VALUE, "kappa": KAPPA,
                       "n_users": N_USERS, "sigma_star": SIGMA_STAR,
                       "cells": [list(c) for c in CELLS], "caps": [str(c) for c in CAPS],
                       "gh_n": GH_N, "n_m_range": M_RANGE, "state_cap": STATE_CAP,
                       "V_prior": V_PRIOR},
        "anchor": anchor, "panel": p, "judge": j,
        "src_hash": _src_hash(),
        "sec_total": round(time.perf_counter() - t0, 3),
    }
    out = Path(args.out) if args.out else (RESULTS / f"cw_r5_0_probcap_{utc.replace('-', '').replace(':', '')}"
                                           f"_{payload['src_hash'][:16]}" / "r50.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(out), "n_fail": j["n_fail"], "verdict": j["verdict"],
                      "curves": j["curves"], "gates": j["gates"]},
                     ensure_ascii=False, indent=2))
    return 0 if j["n_fail"] == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
