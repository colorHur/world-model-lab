#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""T26 · 补偿动作的反事实可分辨性（逐决策日志 + 单步分支，**不训模型**）。

口径来源（★ 全部在看数之前冻结）
--------------------------------
`04-实验与代码/2026-10-20-T26-补偿动作反事实-预注册-v0.1.md` §2–§6。

一句话
------
在**同一个真实状态**上比较两个候选动作的 h 步代价：
    P 支（补偿）= 按 `Linear` 外推估计决策；Z 支（保持）= 按最近送达估计决策；
    `ΔJ = J_P − J_Z`（>0 ⇒ 补偿更差）。
主指标不是"新方法收益"，而是 **`ΔJ` 能否被决策时点可得信息排序**（A2 的 Spearman τ）。

为什么需要探针
--------------
T25 的 I4 = SKIP：入库 cell 只有 episode 级聚合，没有逐决策 `(obs, age, 估计)`。
本轮给 `run_multiuser` 加**只读** `decision_probe`（在 `act()` 之前调用），
探针内 `copy.deepcopy` env + controller（rng 一并复制 ⇒ 两支同噪声）。

纪律（G 系列，预注册 §6）
-------------------------
    G0 P 支对 P 支 ⇒ `ΔJ` 逐位 0（分支里没有未受控随机性）
    G1 `predictor=ZOH` ⇒ 两支退化 ⇒ 全部 `ΔJ` == 0（真在比两个动作）
    G2 `cost_fn=const` ⇒ `ΔJ` 全 0 **且**判据必须判 NO-ROOM（闸有分辨力）
    G3 Z 支 `age` 偏 1 ⇒ 非零比例 > 50%（分支真的分化）
    G4 探针关/开 ⇒ 逐 episode `J` 逐位相同（探针不改主轨迹）
    G5 同参数重跑 ⇒ 逐位相同（抽样顺序没被探针参数改掉）

用法
----
    D:/devtools/miniconda3/envs/ai-lab/python.exe scripts/67_cf_action.py
    ... --skip-experiment      # 只跑自检（不写 results/）
    ... --smoke                # 2 集 × 1 种子 × 1 预算，不写 results/

产物
----
    outputs/67_cf_action[_smoke].json                  明细（不入库）
    results/runs/<run_id>/67_cf_action.json            **入库**（results_io 唯一入口）
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from wmlab.eval.multiuser import default_uav_cost, window_cost  # noqa: E402
from wmlab.eval.predictors import Linear, ZOH  # noqa: E402
from wmlab.eval.results_io import write_result  # noqa: E402

FAILS: list[str] = []


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


T12 = _load(ROOT / "scripts" / "42_t12_hetero_four_arms.py", "t12_mod67")
T52 = _load(ROOT / "scripts" / "52_t21_paced_matching_ablation.py", "t52_mod67")

WINDOW = T12.WINDOW
WARMUP_UAV = T12.WARMUP_UAV
COSTS = [default_uav_cost, T12.cartpole_cost]
ABSORBED = [T12.ABSORBED_UAV, T12.ABSORBED_CART]
ARM = "age"
MODE = "unpaced"
BUDGETS = (200, 100)          # 200 = 主表口径（有益）；100 = 边界口径（变毒）
PRED = "linear"
H = 5                         # 分支视界（预注册 §2 写死）
SEEDS = (200, 201)            # ★ 全新初值（开发 0–31 / 确认 100–189 之外）
N_EP = 8
AGE_BUCKETS = (("1", 1, 1), ("2-3", 2, 3), (">=4", 4, 10 ** 6))

TAU_STRONG = 0.30             # 预注册 §5
TAU_NULL = 0.20
SPREAD_STRONG = 0.20
SPREAD_NULL = 0.10


def check(tag: str, ok: bool, detail: str) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {tag}: {detail}")
    if not ok:
        FAILS.append(f"{tag}: {detail}")


# ================================================================ 装置
def _mk_pred(name: str):
    if name is None:
        return None
    return ZOH() if name == "zoh" else Linear()


def _run(budget, seed, n_ep, *, pred_name=PRED, decision_probe=None, pred=None):
    envs = [T12.make_env("uav", seed=seed, max_steps=WARMUP_UAV + WINDOW),
            T12.make_env("CartPole-v1", seed=seed)]
    ctrls = [T12.PDRelativeController(), T12.CartPoleLQR(_run.K)]
    alloc = T52.make_alloc(ARM, MODE, budget)
    p = pred if pred is not None else _mk_pred(pred_name)
    try:
        return T12.run_multiuser(
            envs, ctrls, alloc, T12.SINRSuccess(T12.Q_REF),
            n_episodes=n_ep, seed=seed, max_steps=WINDOW,
            warmup_steps=[WARMUP_UAV, 0], budget_total=int(budget),
            csi=T12.make_csi(seed, n_ep), cost_fn=COSTS, diagnostic=False,
            counterfactual=False, predictor=p, pred_probe=None,
            decision_probe=decision_probe)
    finally:
        for e in envs:
            e.close()


_run.K = T12.load_lqr_gain()


def _J(records, n_ep):
    return np.asarray(window_cost(records, 2, window=WINDOW, absorbed=ABSORBED,
                                  n_episodes=n_ep), dtype=float)


class _Anchor:
    """决策时点快照 → 两支 h 步分支。

    Args:
        pred:      **runner 正在用的那个**预测器实例（只读调用 `predict`）
        const_cost: 非 None ⇒ 分支代价恒为该值（G2 分辨力对照）
        z_age_bias: Z 支的年龄偏移（G3 只用于证明分支真的分化）
        self_compare: True ⇒ P 支对 P 支（G0）
        only_user:  只处理该用户（None ⇒ 两个用户都处理）
    """

    def __init__(self, pred, *, const_cost=None, z_age_bias=0,
                 self_compare=False, only_user=None, absorb=False):
        self.pred = pred
        self.const_cost = const_cost
        self.z_bias = int(z_age_bias)
        self.self_compare = bool(self_compare)
        self.only_user = only_user
        #: 事后追加口径（**不是预注册主口径**）：任一支提前终止时，用终止步代价填满余下步数，
        #  两支同一规则 ⇒ 仍可配对比较。预注册 §3 的主口径是"终止即排除"，本标志只用于
        #  诊断"被排除的那些点里有没有信号"（CartPole 的代价是 0/1，5 步内几乎只有"死/不死"两种）。
        self.absorb = bool(absorb)
        self.rows: list[dict] = []
        self.n_probe = 0
        self.n_skipped_age0 = 0
        self.n_invalid = 0

    # ---------- 单支：从同一快照走 H 步 ----------
    def _branch(self, env, ctrl, last_obs, age0, cost_fn, user, mode: str):
        e = copy.deepcopy(env)
        c = copy.deepcopy(ctrl)
        J = 0.0
        a_first = None
        terminated = False
        for k in range(1, H + 1):
            if mode == "pred":
                age_k = age0 + k - 1
                est = (np.asarray(self.pred.predict(user, age=age_k,
                                                    actions_since=[np.zeros(2)] * age_k),
                                  dtype=np.float32)
                       if age_k > 0 else np.asarray(last_obs, dtype=np.float32))
            else:
                est = np.asarray(last_obs, dtype=np.float32)
            a = c.act(est)
            if k == 1:
                a_first = np.asarray(a, dtype=np.float64).reshape(-1).copy()
            res = e.step(a)
            cval = (float(self.const_cost) if self.const_cost is not None
                    else float(cost_fn(np.asarray(res.obs))))
            J += cval
            if res.terminated or res.truncated:
                terminated = True
                if self.absorb:
                    J += cval * (H - k)      # 两支同一吸收规则（事后口径，见 __init__）
                break
        return J, a_first, terminated

    def __call__(self, p: dict) -> None:
        user = int(p["user"])
        if self.only_user is not None and user != self.only_user:
            return
        age0 = int(p["age"])
        if age0 < 1:
            self.n_skipped_age0 += 1
            return                     # 估计 == 送达值时两支必然相同，无信息
        self.n_probe += 1
        last_obs = np.asarray(p["last_obs"], dtype=np.float32)
        # ★ 接线守卫：探针 reconstruct 的 P 支估计必须 == runner 本步要用的估计
        est_pred = (np.asarray(self.pred.predict(user, age=age0,
                                                 actions_since=[np.zeros(2)] * age0),
                               dtype=np.float64) if age0 > 0 else last_obs.astype(np.float64))
        if not np.allclose(np.asarray(p["est"], dtype=np.float64), est_pred, atol=1e-6, rtol=0):
            raise RuntimeError(
                f"探针 est 与预测器重建值不一致（ep={p['episode']} t={p['t']} u={user}）"
                f"：此时两支比的不是真实执行的动作")
        cost_fn = p["cost_fn"]
        jp, ap1, tp = self._branch(p["env"], p["controller"], last_obs, age0, cost_fn, user, "pred")
        if self.self_compare:
            jz, az1, tz = self._branch(p["env"], p["controller"], last_obs, age0, cost_fn, user, "pred")
        else:
            jz, az1, tz = self._branch(p["env"], p["controller"], last_obs,
                                       age0 + self.z_bias, cost_fn, user, "zoh")
        if (tp or tz) and not self.absorb:
            self.n_invalid += 1        # 预注册 §3：任一支 h 步内终止 ⇒ 该点排除（两支必须同长度）
            return
        self.rows.append({
            "episode": int(p["episode"]), "t": int(p["t"]), "user": user, "age": age0,
            "J_P": float(jp), "J_Z": float(jz), "dJ": float(jp - jz),
            "d_act": float(np.linalg.norm(ap1 - az1)),
            "d_est": float(np.linalg.norm(est_pred - np.asarray(last_obs, dtype=np.float64))),
        })


# ================================================================ 统计
def _rank(v: np.ndarray) -> np.ndarray:
    order = np.argsort(v, kind="mergesort")
    sv = v[order]
    r = np.empty(len(v), dtype=float)
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return r


def spearman(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 3 or x.size != y.size:
        return float("nan")
    rx, ry = _rank(x), _rank(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    den = float(np.sqrt((rx * rx).sum() * (ry * ry).sum()))
    return float("nan") if den == 0 else float((rx * ry).sum() / den)


def analyse(rows: list[dict]) -> dict:
    out: dict = {"n": len(rows)}
    if not rows:
        return out
    dJ = np.asarray([r["dJ"] for r in rows], dtype=float)
    JZ = np.asarray([r["J_Z"] for r in rows], dtype=float)
    age = np.asarray([r["age"] for r in rows], dtype=float)
    dact = np.asarray([r["d_act"] for r in rows], dtype=float)
    dest = np.asarray([r["d_est"] for r in rows], dtype=float)
    out.update({
        "share_dJ_pos": float((dJ > 0).mean()),
        "median_dJ": float(np.median(dJ)),
        "median_rel": float(np.median(dJ / np.maximum(JZ, 1e-9))),
        "mean_rel": float(np.mean(dJ / np.maximum(JZ, 1e-9))),
        "n_nonzero_dJ": int((np.abs(dJ) > 1e-12).sum()),
        "max_abs_dJ": float(np.max(np.abs(dJ))),
        "tau_age": spearman(dJ, age),
        "tau_dact": spearman(dJ, dact),
        "tau_dest": spearman(dJ, dest),
        "age_median": float(np.median(age)),
        "buckets": {},
    })
    for tag, lo, hi in AGE_BUCKETS:
        m = (age >= lo) & (age <= hi)
        if m.sum() == 0:
            out["buckets"][tag] = {"n": 0}
            continue
        out["buckets"][tag] = {
            "n": int(m.sum()),
            "share_dJ_pos": float((dJ[m] > 0).mean()),
            "median_dJ": float(np.median(dJ[m])),
            "median_rel": float(np.median(dJ[m] / np.maximum(JZ[m], 1e-9))),
        }
    return out


def verdict(cells: dict) -> dict:
    """预注册 §5。`cells` = {"B|uK": analyse(...)}。"""
    taus = {"tau_age": [], "tau_dact": [], "tau_dest": []}
    for c in cells.values():
        if "tau_age" not in c:
            continue
        for k in taus:
            taus[k].append(float(c[k]))
    if not taus["tau_age"]:
        return {"verdict": "NO-DATA", "detail": "无可用 cell"}
    per_cov = {}
    for k, v in taus.items():
        v = np.asarray(v, dtype=float)
        v = v[np.isfinite(v)]
        per_cov[k] = {
            "n_cells": int(v.size),
            "tau": [float(x) for x in v],
            "max_abs": float(np.max(np.abs(v))) if v.size else 0.0,
            "same_sign": bool(v.size and (np.all(v > 0) or np.all(v < 0))),
        }
    shares = [c["buckets"]["2-3"]["share_dJ_pos"] for c in cells.values()
              if c.get("buckets", {}).get("2-3", {}).get("n", 0) > 0]
    spread = float(max(shares) - min(shares)) if len(shares) >= 2 else 0.0

    strong = [k for k, v in per_cov.items()
              if v["max_abs"] >= TAU_STRONG and v["same_sign"] and spread >= SPREAD_STRONG]
    if strong:
        return {"verdict": "SURVIVABLE", "covariates": strong,
                "per_cov": per_cov, "spread_2_3": spread}
    all_weak = all(v["max_abs"] < TAU_NULL for v in per_cov.values())
    sign_flip = any((len(v["tau"]) >= 2 and min(v["tau"]) < 0 < max(v["tau"]))
                    for v in per_cov.values())
    if all_weak or sign_flip or spread < SPREAD_NULL:
        return {"verdict": "NO-ROOM", "per_cov": per_cov, "spread_2_3": spread,
                "why": {"all_weak": bool(all_weak), "sign_flip": bool(sign_flip),
                        "spread_small": bool(spread < SPREAD_NULL)}}
    return {"verdict": "INCONCLUSIVE", "per_cov": per_cov, "spread_2_3": spread}


def _verdict_tau_only(cells: dict) -> dict:
    """修订判定（**只读 A2**，不改任何阈值）：至少一个协变量在所有非空 cell 上
    `|τ| ≥ 0.30` 且**同号** ⇒ 可排序性成立；若所有 `|τ| < 0.20` ⇒ 成立不了；否则 INCONCLUSIVE。

    ★ 为什么不直接用预注册的 `verdict()`：它把 `spread`（固定分档 "2-3" 的份额极差）
      并进了成立条件，而 A2 主指标是 τ。两者在"分档退化"（某用户 ΔJ 恒 0）时会打架 ——
      首跑即暴露（`B200|u1` 的 ΔJ 全为 0 ⇒ τ=NaN ⇒ 分档份额恒 0 ⇒ spread 恒 0）。
      处置：**两个判定都写进证据**，不删预注册那一条。
    """
    per_cov: dict = {}
    for k in ("tau_age", "tau_dact", "tau_dest"):
        v = np.asarray([float(c[k]) for c in cells.values() if k in c], dtype=float)
        v = v[np.isfinite(v)]
        per_cov[k] = {"n_cells": int(v.size),
                      "tau": [float(x) for x in v],
                      "max_abs": float(np.max(np.abs(v))) if v.size else 0.0,
                      "same_sign": bool(v.size and (np.all(v > 0) or np.all(v < 0)))}
    strong = [k for k, d in per_cov.items() if d["max_abs"] >= TAU_STRONG and d["same_sign"]]
    if strong:
        out = "SURVIVABLE"
    elif all(d["max_abs"] < TAU_NULL for d in per_cov.values()):
        out = "NO-ROOM"
    else:
        out = "INCONCLUSIVE"
    return {"verdict": out, "covariates": strong, "per_cov": per_cov,
            "rule": "|tau|>=0.30 & same_sign（只看 A2；不含 spread 条件）"}


# ================================================================ 自检
def selfchecks() -> dict:
    print("\n== 自检 G0–G5（每条写明打死什么错；预注册 §6）==")
    out: dict = {}
    b, s, n = 200, SEEDS[0], 2

    base = _run(b, s, n, pred_name=PRED)
    probed = _Anchor(_mk_pred(PRED), only_user=0)
    hot = _run(b, s, n, pred=probed.pred, decision_probe=probed)
    j0, j1 = _J(base["records"], n), _J(hot["records"], n)
    d = float(np.max(np.abs(j0 - j1)))
    check("G4 探针不改主轨迹", d == 0.0, f"max|ΔJ_ep|={d:.3e}（n_probe={probed.n_probe}）")
    out["probe_n"] = int(probed.n_probe)

    again = _run(b, s, n, pred_name=PRED)
    j2 = _J(again["records"], n)
    d = float(np.max(np.abs(j1 - j2)))
    check("G5 同参数重跑确定性", d == 0.0, f"max|ΔJ_ep|={d:.3e}")

    g0 = _Anchor(_mk_pred(PRED), self_compare=True)
    _run(b, s, n, pred=g0.pred, decision_probe=g0)
    d0 = np.asarray([r["dJ"] for r in g0.rows], dtype=float)
    check("G0 P 支对 P 支", d0.size > 0 and float(np.max(np.abs(d0))) == 0.0,
          f"n={d0.size} max|ΔJ|={float(np.max(np.abs(d0))) if d0.size else float('nan'):.3e}")

    g1 = _Anchor(_mk_pred("zoh"))
    _run(b, s, n, pred=g1.pred, decision_probe=g1)
    d1 = np.asarray([r["dJ"] for r in g1.rows], dtype=float)
    check("G1 ZOH 下两支退化", d1.size > 0 and float(np.max(np.abs(d1))) == 0.0,
          f"n={d1.size} max|ΔJ|={float(np.max(np.abs(d1))) if d1.size else float('nan'):.3e}")

    g3 = _Anchor(_mk_pred(PRED), z_age_bias=1)
    _run(b, s, n, pred=g3.pred, decision_probe=g3)
    d3 = np.asarray([r["dJ"] for r in g3.rows], dtype=float)
    nz = float((np.abs(d3) > 1e-12).mean()) if d3.size else 0.0
    check("G3 分支真的分化", nz > 0.5, f"非零比例={nz:.3f}（n={d3.size}）")

    g2 = _Anchor(_mk_pred(PRED), const_cost=1.0)
    _run(b, s, n, pred=g2.pred, decision_probe=g2)
    d2 = np.asarray([r["dJ"] for r in g2.rows], dtype=float)
    v2 = verdict({"only": analyse(g2.rows)})
    check("G2 常数代价 ⇒ ΔJ≡0 且判 NO-ROOM",
          d2.size > 0 and float(np.max(np.abs(d2))) == 0.0 and v2["verdict"] == "NO-ROOM",
          f"max|ΔJ|={float(np.max(np.abs(d2))) if d2.size else float('nan'):.3e}, "
          f"verdict={v2['verdict']}")
    out["g2_verdict"] = v2["verdict"]
    return out


# ================================================================ 主流程
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-experiment", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    t0 = time.time()
    chk = selfchecks()
    print(f"\n[自检耗时] {time.time() - t0:.1f} s")
    if args.skip_experiment:
        print(f"\n== 只跑自检：{'ALL CHECKS PASS' if not FAILS else 'FAIL'} ==")
        return 0 if not FAILS else 4

    seeds = (SEEDS[0],) if args.smoke else SEEDS
    n_ep = 2 if args.smoke else N_EP
    budgets = (200,) if args.smoke else BUDGETS

    rows_by_cell: dict[str, list] = {}
    meta: dict = {}
    t1 = time.time()
    for absorb in (False, True):
        tag = "|absorb" if absorb else ""
        for b in budgets:
            for s in seeds:
                for u in (0, 1):
                    key = f"B{b}|u{u}{tag}"
                    probe = _Anchor(_mk_pred(PRED), only_user=u, absorb=absorb)
                    _run(b, s, n_ep, pred=probe.pred, decision_probe=probe)
                    rows_by_cell.setdefault(key, []).extend(probe.rows)
                    m = meta.setdefault(key, {"budget": b, "user": u, "seeds": [],
                                              "absorb": bool(absorb),
                                              "n_probe": 0, "n_invalid": 0, "n_age0": 0})
                    m["seeds"].append(int(s))
                    m["n_probe"] += int(probe.n_probe)
                    m["n_invalid"] += int(probe.n_invalid)
                    m["n_age0"] += int(probe.n_skipped_age0)
    el = time.time() - t1

    cells = {k: analyse(v) for k, v in rows_by_cell.items()}
    primary = {k: c for k, c in cells.items() if not k.endswith("|absorb")}
    posthoc = {k: c for k, c in cells.items() if k.endswith("|absorb")}
    v = verdict(primary)
    v_post = verdict(posthoc)
    # ★ 预注册判据缺陷（首跑即暴露，见结果文档 §5）：`spread` 用的是固定分档 "2-3" 的份额极差，
    #   与 A2（τ）不是同一个命题；u1 全域 ΔJ≡0 时分档退化。因此另给"只按 A2 读"的修订判定，
    #   但**不覆盖**预注册判定（两者都写在证据里）。
    v_fix = _verdict_tau_only(primary)
    payload = {
        "unit": "T26 补偿动作反事实可分辨性",
        "protocol": {"arm": ARM, "mode": MODE, "predictor": PRED, "h": H,
                     "seeds": [int(x) for x in seeds], "n_episode": int(n_ep),
                     "window": WINDOW, "warmup": [WARMUP_UAV, 0],
                     "budgets": [int(x) for x in budgets],
                     "note": "seed 200/201 = 全新初值（开发 0–31、确认 100–189 之外）"},
        "gates": {"fails": list(FAILS), "n_fail": len(FAILS)},
        "checks": chk,
        "cells": cells,
        "verdict": v,
        "verdict_tau_only": v_fix,
        "verdict_posthoc_absorb": v_post,
        "meta": meta,
        "elapsed_s": float(el),
        "dJ_definition": "J_P(Linear 外推) − J_Z(最近送达冻结)；h=5 步、期间无送达、两支同噪声",
    }
    out = ROOT / "outputs" / ("67_cf_action_smoke.json" if args.smoke
                              else "67_cf_action.json")
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[outputs] {out}")
    if not args.smoke:
        p = write_result("67_cf_action", payload, run_id=args.run_id)
        print(f"[results] {p}")

    print("\n== 结果摘要 ==")
    for k, c in cells.items():
        if not c.get("n"):
            print(f"  {k}: n=0")
            continue
        print(f"  {k}: n={c['n']} nz={c['n_nonzero_dJ']} share(+)={c['share_dJ_pos']:.3f} "
              f"med_rel={c['median_rel']:+.4f} τ_age={c['tau_age']:+.3f} "
              f"τ_dact={c['tau_dact']:+.3f} τ_dest={c['tau_dest']:+.3f}")
    print(f"  verdict(预注册) = {v['verdict']}  why={v.get('why')}")
    print(f"  verdict(只读 τ)  = {v_fix['verdict']}  cov={v_fix['covariates']}")
    print(f"  verdict(事后 absorb) = {v_post['verdict']}  cov={v_post.get('covariates')}")
    print(f"\n== {('ALL CHECKS PASS' if not FAILS else 'FAIL: ' + '; '.join(FAILS))} ==")
    return 0 if not FAILS else 4


if __name__ == "__main__":
    raise SystemExit(main())
