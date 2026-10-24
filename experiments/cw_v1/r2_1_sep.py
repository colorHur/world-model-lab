# -*- coding: utf-8 -*-
"""CW-R2.1 落盘入口：零训练双基线分辨力（先测 `floor_R2`）。

预注册：`docs/research/CW-R2.1-双基线分辨力-预注册-v0.1.md`

四段
----
`CTRL` **判据自证**（零 episode，合成视图 + 1 次真实视图构造）：`C1–C8`。
`GATE` **闸** `B1–B8`：其中 `B3/B3b` 需要真跑 3 格（直通恒等 + 活体负控）。
`PANEL` **40 格**：`2 plant × 2 model × 10 ep`，规划器种子 `2001` 两模型相同（严格 CRN）。
`NULL` **20 格**：同模型 `belief_readout`、只把规划器种子换成 `3102` ⇒ 与面板的 `2001` 臂配对算 `floor_R2`。

★ 本轮**不训练**；被测物 `plan_ref` / `plan_dp` / `world` / `runner` / `config` / `policies`
  **一字节不动**。新增的只有 `model_policy.py`（模型→视图重绑定）与本入口。
★ 判词只看预注册 §3.1/§3.2 的两个量；其余全是描述量。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from math import comb
from pathlib import Path

import numpy as np

from . import gates as G
from .config import bounded_plant, mainline_config
from .model_api import (OBJECTIVE, deploy, require_view, scan_source, spec_from_config)
from .model_policy import ModelPolicy, rebind_equivalence, rebind_view
from .r1_14_exactbase import _Rec, _git, _src_hash
from .r2_models import BeliefReadout, PriorOnly, TruthDiagnostic
from .runner import check_identities, run_episode
from .streams import make_episode_rngs
from .world import CWWorld, SchedulerView

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results" / "cw_v1"
MODEL_POLICY_SRC = REPO / "experiments" / "cw_v1" / "model_policy.py"

# ==================================================================== 面板常量（预注册 §2）
PLANTS: dict[str, object] = {
    "P1": bounded_plant(),                       # sigma_meas = 0.20（R1 基线）
    "P2": bounded_plant(sigma_meas=0.05),        # 任务观测更准
}
MODELS = ("belief_readout", "prior_only")
EPS = tuple(range(100, 110))
SEED_PANEL = 2001
SEED_NULL = 3102
RHO = 0.98
H_PLAN = 5
M_MAIN = 16
RATIO = 0.5
TAIL_DP = {"tail": "dp", "dp_grid": "r173"}

#: 判词阈值（预注册 §3.3，写死）
N_UNITS = len(PLANTS) * len(EPS)             # 20
N_POS_SEP = 18
N_POS_WEAK_LO = 15
ALPHA = 0.01


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


def binom_p(n_pos: int, n: int = N_UNITS) -> float:
    """单侧 `P(X ≥ n_pos | n, 1/2)`，整数组合和精确算（不引 scipy）。"""
    return sum(comb(n, k) for k in range(int(n_pos), n + 1)) / float(2 ** n)


def null_pair(c1: float, c2: float) -> float:
    """零控的定义（预注册 §3.2）：同模型同 episode，只换规划器采样种子。"""
    return float(abs(float(c1) - float(c2)))


def verdict(med_abs: float, floor: float, n_pos: int, n_units: int) -> dict:
    if not np.isfinite(med_abs) or not np.isfinite(floor):
        return {"verdict": "UNDEFINED", "why": "median 或 floor 非有限"}
    p = binom_p(n_pos, n_units)
    if med_abs <= floor:
        v = "CANNOT-SEPARATE-BASELINES"
    elif n_pos >= N_POS_SEP and p < ALPHA:
        v = "PANEL-SEPARATES"
    elif n_pos >= N_POS_WEAK_LO:
        v = "SEPARATES-MAGNITUDE-ONLY"
    else:
        v = "CANNOT-SEPARATE-BY-SIGN"
    return {"verdict": v, "median_abs_delta": med_abs, "floor_R2": floor,
            "ratio": (med_abs / floor) if floor else float("inf"),
            "n_pos": int(n_pos), "n_units": int(n_units), "binom_p": p}


# ==================================================================== 策略
class _RecAct(_Rec):
    """`_Rec` + 动作序列（`B3` 的对照物与 `action_diff_frac` 都要它）。"""

    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)
        self.acts: list[tuple] = []

    def __call__(self, v):  # noqa: ANN001
        a = super().__call__(v)
        self.acts.append((int(v.t), str(a.kind), -1 if a.user is None else int(a.user)))
        return a


def _factory(cfg, plan_seed: int):
    def f():
        return _RecAct(cfg, h_plan=H_PLAN, m_samples=M_MAIN, seed=plan_seed,
                       rho=RHO, pacing=True, **TAIL_DP)
    return f


_MODEL_CTOR = {"belief_readout": BeliefReadout, "prior_only": PriorOnly}


def _acts(policy) -> list[tuple]:
    if isinstance(policy, ModelPolicy):
        return policy.acts()
    return list(getattr(policy, "acts", []) or [])


def action_diff_frac(a: list[tuple], b: list[tuple]) -> float:
    if not a or not b or len(a) != len(b):
        return float("nan")
    return float(sum(1 for x, y in zip(a, b) if x != y) / len(a))


# ==================================================================== 单格
def run_cell(plant: str, model: str, ep: int, plan_seed: int) -> dict:
    """`model == "raw"` ⇒ 策略就是规划器本身（`B3` 的直通对照）。"""
    cfg = mainline_config(plant=PLANTS[plant])
    budget = cfg.protocol.budget(RATIO)
    fac = _factory(cfg, plan_seed)
    if model == "raw":
        policy = fac()
    else:
        policy = ModelPolicy(_MODEL_CTOR[model](), fac)
    t0 = time.perf_counter()
    res = run_episode(cfg, policy, split="dev", episode=int(ep),
                      budget_total=budget, rho=RHO)
    elapsed = time.perf_counter() - t0
    chk = check_identities(res, cfg)
    acts = _acts(policy)
    cost_by_user = [float(x) for x in res.cost_true_by_user]
    return {
        "plant": plant, "model": model, "ep": int(ep), "plan_seed": int(plan_seed),
        "sigma_meas": float(cfg.plant.sigma_meas),
        "config_fingerprint": res.config_fingerprint,
        "cost_true_per_step": float(res.cost_true_per_step),
        "cost_est_per_step": float(res.cost_est_per_step),
        "cost_true_by_user": cost_by_user,
        "cost_est_by_user": [float(x) for x in res.cost_est_by_user],
        "n_attempted_by_user": [int(x) for x in res.n_attempted_by_user],
        "n_delivered_by_user": [int(x) for x in res.n_delivered_by_user],
        "n_attempted": int(res.n_attempted), "n_delivered": int(res.n_delivered),
        "n_probe": int(res.n_probe), "n_idle": int(res.n_idle),
        "n_forced_hold": int(res.n_forced_hold),
        "aoi_tavg_by_user": list(res.aoi_tavg_by_user),
        "aoi_max_by_user": list(res.aoi_max_by_user),
        "n_never_served": int(res.n_never_served),
        "policy_trace_hash": res.policy_trace_hash,
        "readout_signature": (policy.readout_signature() if isinstance(policy, ModelPolicy) else None),
        "n_model_calls": (policy.n_model_calls if isinstance(policy, ModelPolicy) else 0),
        "acts": [[t, k, u] for (t, k, u) in acts],
        "identities_pass": bool(chk["pass"]),
        "identities_fail": [k for k, v in chk["items"].items() if not v],
        "cost_identity_gap": abs(float(sum(cost_by_user)) - float(res.cost_true_per_step)),
        "elapsed_s": round(elapsed, 2),
    }


# ==================================================================== 合成视图（CTRL 用）
def synth_view(*, t: int = 5) -> SchedulerView:
    """一个**非平凡**的合法视图：`t>0`、`x_hat≠0`、`P_hat≠prior_var`。

    ★ 为什么必须合造：`t=0` 的真实视图上 `x_hat≡0`、`P_hat≡prior_var`，
      恰好与 `PriorOnly` 的输出**逐位相同** ⇒ 在那个点比"两模型是否不同"会假 PASS
      （判据自伤，写卡时就预见到）。
    """
    tel = [
        {"i": 0, "x_hat": 0.70, "P_hat": 0.20, "stamp": 3, "task_aoi": 2,
         "h_mean": 0.10, "h_var": 0.50, "csi_stamp": 4, "csi_aoi": 1},
        {"i": 1, "x_hat": -0.30, "P_hat": 0.90, "stamp": -1, "task_aoi": -1,
         "h_mean": -0.20, "h_var": 0.80, "csi_stamp": -1, "csi_aoi": -1},
    ]
    return SchedulerView({
        "t": int(t), "n_users": 2, "budget_left": 100, "budget_total": 128,
        "prior": {"x_mean": 0.0, "x_var": 1.0, "h_log_mean": 0.0, "h_log_var": 1.0},
        "telemetry": tel, "history": (),
    })


def _xP(view):
    return (np.array([float(r["x_hat"]) for r in view.telemetry]),
            np.array([float(r["P_hat"]) for r in view.telemetry]))


# ==================================================================== CTRL（零 episode）
def controls() -> dict:
    out: dict = {}
    v = synth_view()

    # ---- C1 重绑定后仍是合法视图，且顶层键集不变
    x, P = _xP(v)
    v2 = rebind_view(v, x * 2.0, P * 1.5)
    try:
        require_view(v2)
        c1 = tuple(sorted(v2._pub)) == tuple(sorted(v._pub))
        d1 = {"keys_equal": bool(c1)}
    except Exception as e:  # noqa: BLE001
        c1, d1 = False, {"err": repr(e)}
    out["C1"] = {"pass": bool(c1), "detail": d1}

    # ---- C2 用**原值**重绑定 ⇒ 逐位零改动（重建没丢/串字段）
    v_same = rebind_view(v, x, P)
    eq = rebind_equivalence(v, v_same)
    out["C2"] = {"pass": bool(eq["ok"]), "detail": eq}

    # ---- C3 只改 x_hat/P_hat：不一致项必须**恰好**是这两个键（逐用户）
    v_chg = rebind_view(v, x + 1.0, P + 0.5)
    bad = set(rebind_equivalence(v, v_chg)["bad"])
    expect = {f"telemetry[{i}].{k}" for i in range(int(v.n_users)) for k in ("x_hat", "P_hat")}
    out["C3"] = {"pass": bad == expect, "detail": {"bad": sorted(bad), "expect": sorted(expect)}}

    # ---- C4 原视图**未被污染**
    x_after, P_after = _xP(v)
    out["C4"] = {"pass": bool(np.array_equal(x, x_after) and np.array_equal(P, P_after)),
                 "detail": {"x_hat_unchanged": bool(np.array_equal(x, x_after)),
                            "P_hat_unchanged": bool(np.array_equal(P, P_after))}}

    # ---- C5 活体负控：坏 rebind（预算 +1）必须被 C2/C3 抓到
    def _bad_rebind(view, xh, Ph):
        good = rebind_view(view, xh, Ph)
        tel = [dict(r) for r in good.telemetry]
        return SchedulerView({"t": int(good.t), "n_users": int(good.n_users),
                              "budget_left": int(good.budget_left) + 1,
                              "budget_total": int(good.budget_total),
                              "prior": dict(good.prior), "telemetry": tel,
                              "history": tuple(good.history)})

    eq_bad = rebind_equivalence(v, _bad_rebind(v, x, P))
    out["C5"] = {"pass": (not eq_bad["ok"]) and ("budget_left" in eq_bad["bad"]),
                 "detail": {"bad": eq_bad["bad"],
                            "why": "闸必须能 FAIL：用原值重绑定 + 预算改 1 ⇒ 必须报 budget_left"}}

    # ---- C6 两模型在非平凡视图上**确实**不同（接线分辨力）
    br = BeliefReadout().predict(v)
    po = PriorOnly().predict(v)
    out["C6"] = {"pass": bool(not np.allclose(br.x_hat, po.x_hat)
                              and not np.allclose(br.P, po.P)),
                 "detail": {"belief": [br.x_hat.tolist(), br.P.tolist()],
                            "prior": [po.x_hat.tolist(), po.P.tolist()]}}

    # ---- C7 特权臂必须被 deploy 拒绝
    td = TruthDiagnostic(x=np.zeros(2), log_h=np.zeros(2))
    try:
        deploy(td)
        c7, note = False, "deploy 接受了特权臂（不该）"
    except PermissionError:
        c7, note = True, "PermissionError"
    out["C7"] = {"pass": c7, "detail": note}

    # ---- C8 null_pair 公式自检（已知答案）
    ok8 = (null_pair(3.0, 3.0) == 0.0 and null_pair(3.0, 1.0) == 2.0
           and null_pair(1.0, 3.0) == 2.0)
    out["C8"] = {"pass": bool(ok8),
                 "detail": {"same": null_pair(3.0, 3.0), "fwd": null_pair(3.0, 1.0),
                            "rev": null_pair(1.0, 3.0)}}

    ks = [f"C{i}" for i in range(1, 9)]
    out["all_pass"] = all(out[k]["pass"] for k in ks)
    out["_keys"] = ks
    return out


# ==================================================================== 闸
def gates(ctrl: dict) -> dict:
    chk: list[dict] = []

    def add(gid: str, ok: bool, detail=None) -> None:
        chk.append({"id": gid, "ok": bool(ok), "detail": detail})

    # ---- B1 部署路径 AST 干净
    s = scan_source(MODEL_POLICY_SRC.read_text(encoding="utf-8"))
    add("B1", not (s["names"] or s["attrs"] or s["imports"]),
        {"file": MODEL_POLICY_SRC.name, "names": s["names"], "attrs": s["attrs"],
         "imports": s["imports"], "why": "模型层若出现真值名/_pub/越界 import ⇒ 非法通道"})

    # ---- B2/B2b/B2c 直接绑 CTRL 的三条（守的是同一实现，不重复写）
    add("B2", ctrl["C2"]["pass"],
        {"from": "C2", "why": "重建时漏字段/串字段 ⇒ 用原值重绑定后不逐位相同",
         "bad": ctrl["C2"]["detail"]["bad"]})
    add("B2b", ctrl["C4"]["pass"],
        {"from": "C4", "why": "直接改原 dict ⇒ 世界状态被模型写回（边界破）"})
    add("B2c", ctrl["C5"]["pass"],
        {"from": "C5", "why": "活体负控：坏 rebind 若不被抓 ⇒ 本组闸没有分辨力"})

    # ---- B3 / B3b 直通恒等 + 活体负控（需真跑）
    t0 = time.perf_counter()
    raw = run_cell("P1", "raw", 100, SEED_PANEL)
    wrp = run_cell("P1", "belief_readout", 100, SEED_PANEL)
    b3 = (raw["policy_trace_hash"] == wrp["policy_trace_hash"]
          and raw["cost_true_per_step"] == wrp["cost_true_per_step"]
          and raw["cost_true_by_user"] == wrp["cost_true_by_user"])
    add("B3", b3, {"raw": {"trace": raw["policy_trace_hash"],
                          "cost": raw["cost_true_per_step"]},
                   "wrapped": {"trace": wrp["policy_trace_hash"],
                               "cost": wrp["cost_true_per_step"]},
                   "why": "包装层引入扰动 ⇒ 面板量的不是模型效应"})

    alt = run_cell("P1", "prior_only", 100, SEED_PANEL)
    b3b = (alt["policy_trace_hash"] != raw["policy_trace_hash"])
    add("B3b", b3b, {"raw_trace": raw["policy_trace_hash"],
                     "prior_trace": alt["policy_trace_hash"],
                     "why": "若换模型也不改动作 ⇒ B3 的'恒等'可能只是两边都空跑",
                     "note": ("若本项 FAIL，不是故障而是**结果**：该格上 prior_only 与"
                              "接收端后验给出同一动作序列 ⇒ 该格无区分力，须在交付里如实写")})
    b3_elapsed = round(time.perf_counter() - t0, 1)

    # ---- B4 接线：模型输出**真的**进了 rebind（digest 非空且不同）
    add("B4", bool(wrp["n_model_calls"] > 0 and alt["n_model_calls"] > 0
                   and wrp["readout_signature"] and alt["readout_signature"]
                   and wrp["readout_signature"] != alt["readout_signature"]),
        {"belief_calls": wrp["n_model_calls"], "prior_calls": alt["n_model_calls"],
         "belief_sig": wrp["readout_signature"], "prior_sig": alt["readout_signature"],
         "why": "模型没被调用 / 输出没进 rebind / 两个模型喂了同一串信念"})

    # ---- B5 特权臂（CTRL C7，重述进闸表以保"一处不漏"）
    add("B5", ctrl["C7"]["pass"], {"from": "C7"})

    # ---- B6 null_pair 公式
    add("B6", ctrl["C8"]["pass"], {"from": "C8"})

    # ---- B7 逐用户成本恒等（B3 的三格）
    gaps = {f"{c['model']}@ep{c['ep']}": c["cost_identity_gap"] for c in (raw, wrp, alt)}
    add("B7", all(float(g) <= 1e-9 for g in gaps.values()),
        {"gaps": gaps, "why": "逐用户量定义漂了（R1.8：聚合口径先查分母）"})

    # ---- B8 协议回归
    t1 = time.perf_counter()
    reg = G.run_all()
    add("B8", int(reg["n_fail"]) == 0,
        {"n_checks": reg["n_checks"], "n_fail": reg["n_fail"],
         "elapsed_s": round(time.perf_counter() - t1, 2)})

    return {"checks": chk, "n_checks": len(chk),
            "n_fail": sum(1 for c in chk if not c["ok"]),
            "b3_cells": [raw, wrp, alt], "b3_elapsed_s": b3_elapsed,
            "run_all": {"n_checks": reg["n_checks"], "n_fail": reg["n_fail"]}}


# ==================================================================== 面板
def panel() -> tuple[list[dict], list[dict]]:
    cells: list[dict] = []
    for plant in PLANTS:
        for model in MODELS:
            for ep in EPS:
                c = run_cell(plant, model, ep, SEED_PANEL)
                cells.append(c)
                print(f"[CELL] {plant}/{model:>14}/ep{ep} cost={c['cost_true_per_step']:.6f} "
                      f"att={c['n_attempted']} del={c['n_delivered']} "
                      f"probe={c['n_probe']} ({c['elapsed_s']}s)")
    nulls: list[dict] = []
    for plant in PLANTS:
        for ep in EPS:
            c = run_cell(plant, "belief_readout", ep, SEED_NULL)
            nulls.append(c)
            print(f"[NULL] {plant}/belief_readout/ep{ep} s={SEED_NULL} "
                  f"cost={c['cost_true_per_step']:.6f} ({c['elapsed_s']}s)")
    return cells, nulls


def _cell(cells: list[dict], plant: str, model: str, ep: int) -> dict:
    return next(c for c in cells if c["plant"] == plant and c["model"] == model and c["ep"] == ep)


def analyse(cells: list[dict], nulls: list[dict]) -> dict:
    rows = []
    for plant in PLANTS:
        for ep in EPS:
            b = _cell(cells, plant, "belief_readout", ep)
            p = _cell(cells, plant, "prior_only", ep)
            n2 = next(c for c in nulls if c["plant"] == plant and c["ep"] == ep)
            delta = float(p["cost_true_per_step"]) - float(b["cost_true_per_step"])
            floor = null_pair(b["cost_true_per_step"], n2["cost_true_per_step"])
            r_i = []
            for i in range(len(b["cost_true_by_user"])):
                den = float(b["cost_true_by_user"][i])
                num = float(p["cost_true_by_user"][i])
                r_i.append(num / den if den > 0 else float("nan"))
            rows.append({
                "plant": plant, "ep": int(ep),
                "cost_belief": float(b["cost_true_per_step"]),
                "cost_prior": float(p["cost_true_per_step"]),
                "delta": delta, "floor_pair": floor,
                "r_i": r_i, "n_pos_any_r_i_gt1": int(sum(1 for v in r_i if v > 1.0)),
                "action_diff_frac": action_diff_frac(
                    [tuple(a) for a in b["acts"]], [tuple(a) for a in p["acts"]]),
                "belief": {"n_attempted": b["n_attempted"], "n_delivered": b["n_delivered"],
                           "n_probe": b["n_probe"],
                           "att_by_user": b["n_attempted_by_user"],
                           "del_by_user": b["n_delivered_by_user"],
                           "aoi_tavg_by_user": b["aoi_tavg_by_user"],
                           "n_never_served": b["n_never_served"]},
                "prior": {"n_attempted": p["n_attempted"], "n_delivered": p["n_delivered"],
                          "n_probe": p["n_probe"],
                          "att_by_user": p["n_attempted_by_user"],
                          "del_by_user": p["n_delivered_by_user"],
                          "aoi_tavg_by_user": p["aoi_tavg_by_user"],
                          "n_never_served": p["n_never_served"]},
            })

    def _block(sel: list[dict]) -> dict:
        d = np.array([r["delta"] for r in sel], dtype=float)
        f = np.array([r["floor_pair"] for r in sel], dtype=float)
        med_abs = float(np.median(np.abs(d)))
        floor = float(np.median(f))
        n_pos = int((d > 0).sum())
        v = verdict(med_abs, floor, n_pos, len(sel))
        v.update({"delta_median": float(np.median(d)),
                  "delta_iqr": [float(np.percentile(d, 25)), float(np.percentile(d, 75))],
                  "abs_delta_iqr": [float(np.percentile(np.abs(d), 25)),
                                    float(np.percentile(np.abs(d), 75))]})
        return v

    blocks = {"ALL": _block(rows)}
    for plant in PLANTS:
        blocks[plant] = _block([r for r in rows if r["plant"] == plant])

    # 必报（R17）：逐用户 r_i 的分布 + 交易计数
    r_all = [v for r in rows for v in r["r_i"] if np.isfinite(v)]
    trade = [{"plant": r["plant"], "ep": r["ep"], "r_i": r["r_i"]}
             for r in rows if any((v > 1.0) for v in r["r_i"] if np.isfinite(v))]
    return {"rows": rows, "blocks": blocks,
            "r_i_summary": {"n": len(r_all),
                            "median": float(np.median(r_all)) if r_all else None,
                            "min": float(np.min(r_all)) if r_all else None,
                            "max": float(np.max(r_all)) if r_all else None,
                            "n_gt1_users": int(sum(1 for v in r_all if v > 1.0))},
            "trades": trade,
            "note_sign": "Δ = cost(prior_only) − cost(belief_readout)；Δ>0 ⇒ 弱基线更贵 ⇒ 强基线更好",
            "action_diff_median": float(np.median([r["action_diff_frac"] for r in rows
                                                   if np.isfinite(r["action_diff_frac"])]))
            if any(np.isfinite(r["action_diff_frac"]) for r in rows) else None}


# ==================================================================== 主流程
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="CW-R2.1 零训练双基线分辨力")
    ap.add_argument("--ctrl-only", action="store_true", help="只跑判据自证（零 episode）")
    ap.add_argument("--probe", action="store_true", help="只跑 2 格计时（不写证据）")
    ap.add_argument("--no-null", action="store_true", help="跳过零控（只作调试，不进判词）")
    ap.add_argument("--tag", default="main")
    args = ap.parse_args(argv)

    t_start = time.perf_counter()
    ctrl = controls()
    kk = ctrl.pop("_keys")
    print(f"[CTRL] all_pass={ctrl['all_pass']} " +
          " ".join(f"{k}={ctrl[k]['pass']}" for k in kk))
    if not ctrl["all_pass"]:
        print("[ABORT] 判据自证未过 ⇒ 按预注册 §4 不看闸与面板")
        return 3
    if args.ctrl_only:
        print(json.dumps(ctrl, ensure_ascii=False, indent=1, default=float))
        return 0

    if args.probe:
        a = run_cell("P1", "belief_readout", 100, SEED_PANEL)
        b = run_cell("P2", "prior_only", 100, SEED_PANEL)
        print(json.dumps({"cell1_s": a["elapsed_s"], "cell2_s": b["elapsed_s"],
                          "mean_s": round((a["elapsed_s"] + b["elapsed_s"]) / 2, 2),
                          "est_61_cells_s": round((a["elapsed_s"] + b["elapsed_s"]) / 2 * 61, 1)},
                         ensure_ascii=False, indent=1))
        return 0

    gate = gates(ctrl)
    print(f"[GATE] n_checks={gate['n_checks']} n_fail={gate['n_fail']} "
          f"(B3 组 {gate['b3_elapsed_s']}s)")
    for c in gate["checks"]:
        print(f"  {'PASS' if c['ok'] else 'FAIL'}  {c['id']}")

    cells, nulls = ([], [])
    if gate["n_fail"] == 0:
        t_p = time.perf_counter()
        cells, nulls = panel()
        panel_s = round(time.perf_counter() - t_p, 1)
        print(f"[PANEL] {len(cells)} 格 + {len(nulls)} 零控 = {panel_s}s")
        ana = analyse(cells, nulls)
    else:
        panel_s = 0.0
        ana = {"blocks": {"ALL": {"verdict": "BLOCKED-BY-GATE"}}, "rows": []}
        print("[SKIP] 闸未全通过 ⇒ 按预注册不看面板")

    # ---- 落盘
    head = _git(["rev-parse", "HEAD"])
    wt_sha = _git(["hash-object", "experiments/cw_v1/r2_1_sep.py"])
    blob_sha = _git(["rev-parse", f"{head}:experiments/cw_v1/r2_1_sep.py"], allow_fail=True)
    dirty = _git(["status", "--porcelain"])
    srchash = _src_hash()
    outdir = RESULTS / f"cw_r2_1_sep_{_now_utc()}_{srchash}"
    outdir.mkdir(parents=True, exist_ok=False)
    print(f"[OUT] {outdir.name}")

    cfg_mat = {p: mainline_config(plant=PLANTS[p]) for p in PLANTS}
    spec = spec_from_config(cfg_mat["P1"], budget_total=cfg_mat["P1"].protocol.budget(RATIO),
                            split="dev", episodes=EPS, seeds=(SEED_PANEL, SEED_NULL),
                            models=MODELS)
    idx = {r["plant"]: r["ep"] for r in ana["rows"]} if ana["rows"] else {}
    for plant in PLANTS:
        if plant in idx:
            pass

    summary = {
        "protocol": "CW-POMDP-v1.1", "stage": "CW-R2.1", "kind": "NON-TRAINED-BASELINE-SEPARABILITY",
        "prereg": "docs/research/CW-R2.1-双基线分辨力-预注册-v0.1.md",
        "hypothesis": "H-R2-0",
        "panels": {"plants": list(PLANTS), "models": list(MODELS), "eps": list(EPS),
                   "seed_panel": SEED_PANEL, "seed_null": SEED_NULL,
                   "n_cells": len(cells), "n_null": len(nulls)},
        "verdicts": {k: v["verdict"] for k, v in ana["blocks"].items()},
        "blocks": ana["blocks"],
        "r_i_summary": ana.get("r_i_summary"),
        "trades": ana.get("trades"),
        "action_diff_median": ana.get("action_diff_median"),
        "note_sign": ana.get("note_sign"),
        "rows": ana.get("rows", []),
        "objective": OBJECTIVE,
        "spec_fingerprint_P1": spec.fingerprint(),
        "sigma_meas": {p: float(PLANTS[p].sigma_meas) for p in PLANTS},
        "budget": {"total_s": None, "panel_s": panel_s, "per_cell_s_observed":
                   [c["elapsed_s"] for c in cells]},
        "head": head, "src_hash": srchash,
        "n_checks_gates": gate["n_checks"], "n_fail_gates": gate["n_fail"],
        "training": "none（本轮零训练）",
    }

    (outdir / "cells.json").write_text(json.dumps(cells, ensure_ascii=False, indent=1,
                                                 default=float), encoding="utf-8")
    (outdir / "null.json").write_text(json.dumps(
        {"nulls": nulls, "floor_by_pair": [
            {"plant": r["plant"], "ep": r["ep"], "floor_pair": r["floor_pair"]}
            for r in ana.get("rows", [])]}, ensure_ascii=False, indent=1, default=float),
        encoding="utf-8")
    (outdir / "gates.json").write_text(json.dumps(gate, ensure_ascii=False, indent=1,
                                                 default=str), encoding="utf-8")
    (outdir / "ctrl.json").write_text(json.dumps(ctrl, ensure_ascii=False, indent=1,
                                                default=float), encoding="utf-8")

    try:
        import psutil  # type: ignore
        rss = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
    except Exception:
        rss = None
    total_s = round(time.perf_counter() - t_start, 1)
    summary["budget"]["total_s"] = total_s
    (outdir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1,
                                                   default=float), encoding="utf-8")

    meta = {"command": "python -m experiments.cw_v1.r2_1_sep --tag " + str(args.tag),
            "timestamp_local": datetime.now().astimezone().isoformat(),
            "python": os.sys.version.split()[0], "numpy": np.__version__,
            "head": head, "src_hash": srchash,
            "worktree_sha_r2_1": wt_sha, "blob_sha_r2_1": blob_sha,
            "blob_sha_note": (None if blob_sha is None else "运行前已在 HEAD 上"),
            "git_dirty_before_run": dirty.splitlines(),
            "protocol": "CW-POMDP-v1.1", "event_pairing": "slot",
            "n_cells": len(cells), "n_null": len(nulls),
            "panel_elapsed_s": panel_s, "total_elapsed_s": total_s,
            "peak_rss_mb": round(rss, 1) if rss else None,
            "n_checks_gates": gate["n_checks"], "n_fail_gates": gate["n_fail"],
            "training": "none"}
    (outdir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                      encoding="utf-8")

    n_fail = gate["n_fail"] + (0 if ctrl["all_pass"] else 1)
    print(f"[DONE] {outdir.name} n_fail={n_fail} total={total_s}s peak={meta['peak_rss_mb']}MB")
    print(json.dumps({"verdicts": summary["verdicts"],
                      "floor_R2": ana["blocks"]["ALL"].get("floor_R2"),
                      "median_abs_delta": ana["blocks"]["ALL"].get("median_abs_delta"),
                      "n_pos": ana["blocks"]["ALL"].get("n_pos"),
                      "r_i_summary": ana.get("r_i_summary", {}).get("median"),
                      "trades": len(ana.get("trades") or [])},
                     ensure_ascii=False, indent=1))
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    import sys
    sys.exit(main())
