# -*- coding: utf-8 -*-
"""`CW-R2.7` 开训入口：**第一次真正的参数更新**（只训读出，不训策略）。

预注册：`docs/research/CW-R2.7-开训卡-预注册-v0.1.md`

四段（预注册 §7 的顺序门）
--------------------------
`CTRL` 零 episode 自证 + 闸 `G1–G8`（`G9` 走 `gates.run_all`）。
`DATA` 采训练集（`train` ep 0–69）与 dev 特征（ep 100–114）：**行为策略 = 基线规划器本身**。
`P1` 离线：3 种子训练 → dev 上 `MSE_learned / MSE_base`（逐 ep 配对）。
`P2` 闭环：仅当 P1 过 `GATE-OFFLINE`（中位比 ≤1.05 且逐 ep 方向 ≥9/15）才跑。

三条纪律
--------
1. ★ 本文件**不是**部署路径（`G3` 只扫 `r2_7_learn.py`）：这里持有**标签通道**
   （`w._x_true`）与优化器。标签只进 loss，**永不进特征**（`G5b` 钉）。
2. ★ 训练集与 dev **零交集**（`assert_disjoint`）。
3. ★ 不用 dev 调任何超参：`lr/步数/宽度/正则` 全部来自预注册 §5。
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
from . import mech3
from .config import mainline_config
from .model_api import FIELD_PERMISSION, FORBIDDEN_NAMES, OBJECTIVE, scan_source
from .model_policy import ModelPolicy
from .r1_14_exactbase import _git, _src_hash
from .r2_1_sep import PLANTS, RHO, SEED_PANEL, _factory
from .r2_7_learn import (GLOBAL_FEATURES, N_FEATURES, PER_USER_FEATURES,
                         ResidualReadout)
from .r2_models import BeliefReadout
from .r2_train_spec import (DEV_EPS, EVAL_PRIMARY, TRAIN_EPS, TRAIN_SEEDS,
                            assert_capacity, assert_disjoint, assert_objective,
                            count_params)
from .runner import check_identities, run_episode
from .streams import make_episode_rngs
from .world import Action, BudgetExhausted, CWWorld, SchedulerView

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results" / "cw_v1"
LEARN_SRC = REPO / "experiments" / "cw_v1" / "r2_7_learn.py"

# ==================================================================== 预算（预注册 §5）
N_STEPS = 12_000
BATCH = 512
LR = 3e-3
WD = 1e-4
HIDDEN = (32, 32)
PLANT = "P1"
RATIO = 0.5
DEV_LIST = tuple(int(e) for e in DEV_EPS)
TRAIN_LIST = tuple(int(e) for e in TRAIN_EPS)
OFFLINE_RATIO_MAX = 1.05
OFFLINE_MIN_DIR = 9


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


def binom_p(n_pos: int, n: int) -> float:
    return sum(comb(n, k) for k in range(int(n_pos), n + 1)) / float(2 ** n)


# ==================================================================== 数据采集
def _new_world(cfg, split: str, ep: int, budget: int):
    rngs, sids = make_episode_rngs(cfg.root_seed, split, ep)
    w = CWWorld(cfg, rngs, sids, rho=RHO)
    w.budget_total = int(budget)
    w._x_hat[:] = 0.0
    w._P_hat[:] = float(cfg.plant.prior_var)
    w._stamp[:] = -1
    w._m_h[:] = 0.0
    w._v_h[:] = float(cfg.channel.h_log_init_std) ** 2
    w._csi_stamp[:] = -1
    return w


def collect(cfg, split: str, eps, *, plan_seed: int = SEED_PANEL,
            with_labels: bool = True, verbose: bool = False) -> dict:
    """按 `runner.run_episode` 的同一顺序跑，逐槽记 `(特征, 基线信念, 标签)`。

    ★ 与 `runner` 的**唯一**差别是多记了三样东西（特征、基线信念、标签）；推进顺序、
      预算截断的降级路径（`BudgetExhausted` ⇒ HOLD）、复位初值全部照抄。
    """
    budget = int(cfg.protocol.budget(RATIO))
    T = int(cfg.protocol.horizon)
    ex = ResidualReadout()          # 只借它的 features()（未训练、无副作用）
    X, Y, XB, EP, US = [], [], [], [], []
    t0 = time.perf_counter()
    eps = list(eps)
    for ep in eps:
        w = _new_world(cfg, split, ep, budget)
        policy = _factory(cfg, plan_seed)()
        for _ in range(T):
            v = w.view()
            X.append(ex.features(v))
            XB.append(np.asarray([float(r["x_hat"]) for r in v.telemetry], dtype=float))
            if with_labels:
                Y.append(np.asarray(w._x_true, dtype=float).copy())
            EP.append(np.full(int(v.n_users), int(ep), dtype=int))
            US.append(np.arange(int(v.n_users), dtype=int))
            act = policy(v)
            n_sub = max(int(cfg.protocol.c_probe) if act.kind == "PROBE" else
                        int(cfg.protocol.c_data) if act.kind == "SEND" else 1, 1)
            try:
                w.step(act, n_substeps=n_sub)
            except BudgetExhausted:
                w.budget_blocked += 1
                w.step(Action("HOLD"), n_substeps=1)
        if verbose and (ep % 10 == 0 or ep == eps[-1]):
            print(f"  [DATA] {split}/ep{ep} ({time.perf_counter() - t0:.0f}s)", flush=True)
    out = {"X": np.concatenate(X, axis=0), "xb": np.concatenate(XB, axis=0),
           "ep": np.concatenate(EP, axis=0), "u": np.concatenate(US, axis=0),
           "elapsed_s": round(time.perf_counter() - t0, 1),
           "n_samples": int(sum(len(a) for a in X))}
    if with_labels:
        out["y"] = np.concatenate(Y, axis=0)
    return out


# ==================================================================== 训练（Adam + 手写反传）
def _forward(params, Z):
    # `ResidualReadout.parameters()` 的顺序是 `[*W, *b]`（与 set_parameters 同序）
    W1, W2, W3, b1, b2, b3 = params
    h1 = np.tanh(Z @ W1 + b1)
    h2 = np.tanh(h1 @ W2 + b2)
    o = h2 @ W3 + b3
    return o, (Z, h1, h2)


def _loss_grad(params, Z, target):
    """`L = mean((o − target)²)`；`target` 是**残差**标签（= 真值 − 基线信念）。"""
    o, (Z_, h1, h2) = _forward(params, Z)
    d = o.reshape(-1) - target
    loss = float(np.mean(d * d))
    do = (2.0 * d / d.size).reshape(-1, 1)
    W1, W2, W3, b1, b2, b3 = params
    gW3 = h2.T @ do
    gb3 = do.sum(axis=0)
    da2 = (do @ W3.T) * (1.0 - h2 * h2)
    gW2 = h1.T @ da2
    gb2 = da2.sum(axis=0)
    da1 = (da2 @ W2.T) * (1.0 - h1 * h1)
    gW1 = Z_.T @ da1
    gb1 = da1.sum(axis=0)
    grads = [gW1 + WD * W1, gW2 + WD * W2, gW3 + WD * W3, gb1, gb2, gb3]
    return loss, grads


def train_fit(X, xb, y, *, seed: int, n_steps: int = N_STEPS, batch: int = BATCH,
              lr: float = LR, log_every: int = 50, verbose: bool = False):
    """固定预算训练（**无早停**：预算由预注册写死）。返回 `(model, history)`。"""
    rng = np.random.default_rng(seed)
    model = ResidualReadout(hidden=HIDDEN, seed=seed)
    mu = np.asarray(X, dtype=float).mean(axis=0)
    sd = np.asarray(X, dtype=float).std(axis=0)
    sd[~np.isfinite(sd) | (sd <= 0)] = 1.0
    model.set_standardizer(mu, sd)
    Z = (np.asarray(X, dtype=float) - mu) / sd
    targ = (np.asarray(y, dtype=float) - np.asarray(xb, dtype=float)).reshape(-1)
    n = Z.shape[0]
    params = [np.array(p, dtype=float, copy=True) for p in model.parameters()]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    b1_, b2_, eps = 0.9, 0.999, 1e-8
    hist: list[dict] = []
    t0 = time.perf_counter()
    for step in range(1, int(n_steps) + 1):
        idx = rng.integers(0, n, size=int(batch))
        _, grads = _loss_grad(params, Z[idx], targ[idx])
        for j in range(len(params)):
            m[j] = b1_ * m[j] + (1 - b1_) * grads[j]
            v[j] = b2_ * v[j] + (1 - b2_) * (grads[j] ** 2)
            mh = m[j] / (1 - b1_ ** step)
            vh = v[j] / (1 - b2_ ** step)
            params[j] -= lr * mh / (np.sqrt(vh) + eps)
        if step == 1 or step % log_every == 0 or step == int(n_steps):
            full, _ = _loss_grad(params, Z, targ)
            hist.append({"step": int(step), "loss_full": float(full)})
            if verbose and step % 2000 == 0:
                print(f"  [TRAIN seed={seed}] step={step} loss={full:.6f} "
                      f"({time.perf_counter() - t0:.0f}s)", flush=True)
    model.set_parameters(params)
    model.trained = True
    model.train_meta = {"seed": int(seed), "n_steps": int(n_steps), "batch": int(batch),
                        "lr": float(lr), "wd": float(WD), "n_train_samples": int(n),
                        "elapsed_s": round(time.perf_counter() - t0, 1),
                        "final_loss_full": float(hist[-1]["loss_full"])}
    return model, hist


def plateau_stats(hist: list[dict], *, frac: float = 0.2, thresh: float = 0.01) -> dict:
    """预注册 §4：末 `frac` 预算内训练损失相对下降 `< thresh` ⇒ 平台。"""
    if not hist:
        return {"plateau": False, "why": "空曲线"}
    last = int(hist[-1]["step"])
    lo = int(round((1.0 - frac) * last))
    win = [h for h in hist if h["step"] >= lo]
    if len(win) < 2:
        return {"plateau": False, "why": f"窗口内只有 {len(win)} 个点"}
    a, b = float(win[0]["loss_full"]), float(win[-1]["loss_full"])
    rel = (a - b) / a if a > 0 else float("inf")
    return {"plateau": bool(rel < thresh), "rel_drop_last20pct": float(rel),
            "loss_start_win": a, "loss_end": b, "window_steps": [win[0]["step"], win[-1]["step"]]}


# ==================================================================== 离线评价
def offline_eval(model, data: dict) -> dict:
    """dev 上 `MSE_learned / MSE_base`（逐 ep 配对）。代价空间只作诊断（预注册 §2.1）。"""
    X, xb, y, ep = data["X"], data["xb"], data["y"], data["ep"]
    d = model.residual(X).reshape(-1)
    pred = xb + d
    per_ep = []
    for e in sorted(set(int(v) for v in ep)):
        m = ep == e
        err_b = (xb[m] - y[m]) ** 2
        err_l = (pred[m] - y[m]) ** 2
        cb = (xb[m] ** 2 - y[m] ** 2) ** 2
        cl = (pred[m] ** 2 - y[m] ** 2) ** 2
        per_ep.append({"ep": int(e), "mse_base": float(err_b.mean()),
                       "mse_learned": float(err_l.mean()),
                       "ratio": float(err_l.mean() / err_b.mean()) if err_b.mean() > 0 else None,
                       "mse_cost_base": float(cb.mean()), "mse_cost_learned": float(cl.mean()),
                       "cost_ratio": float(cl.mean() / cb.mean()) if cb.mean() > 0 else None,
                       "max_abs_residual": float(np.abs(d[m]).max())})
    ratios = np.asarray([r["ratio"] for r in per_ep], dtype=float)
    return {"per_ep": per_ep,
            "median_ratio": float(np.median(ratios)),
            "n_ratio_lt1": int((ratios < 1.0).sum()),
            "n_ep": int(len(ratios)),
            "mse_base_all": float(((xb - y) ** 2).mean()),
            "mse_learned_all": float(((pred - y) ** 2).mean()),
            "cost_ratio_median": float(np.median([r["cost_ratio"] for r in per_ep])),
            "max_abs_residual": float(np.abs(d).max()),
            "sign_corr": float(np.corrcoef(pred.reshape(-1), y.reshape(-1))[0, 1])
            if pred.std() > 0 else None}


# ==================================================================== 闭环
def run_cell(model, ep: int, plan_seed: int, *, label: str) -> dict:
    cfg = mainline_config(plant=PLANTS[PLANT])
    budget = int(cfg.protocol.budget(RATIO))
    fac = _factory(cfg, plan_seed)
    if model is None:
        policy = fac()
        sig, n_calls = None, 0
    else:
        policy = ModelPolicy(model, fac)
    t0 = time.perf_counter()
    res = run_episode(cfg, policy, split="dev", episode=int(ep),
                      budget_total=budget, rho=RHO)
    elapsed = time.perf_counter() - t0
    chk = check_identities(res, cfg)
    if model is not None:
        sig, n_calls = policy.readout_signature(), int(policy.n_model_calls)
    cell = {"plant": PLANT, "model": label, "ep": int(ep), "plan_seed": int(plan_seed),
            "n_slots": int(res.n_slots),
            "aoi_tavg_by_user": list(res.aoi_tavg_by_user),
            "n_unserved_slots_by_user": [int(x) for x in res.n_unserved_slots_by_user],
            "cost_true_per_step": float(res.cost_true_per_step),
            "cost_true_by_user": [float(x) for x in res.cost_true_by_user],
            "cost_est_per_step": float(res.cost_est_per_step),
            "n_attempted": int(res.n_attempted), "n_delivered": int(res.n_delivered),
            "n_probe": int(res.n_probe), "n_forced_hold": int(res.n_forced_hold),
            "identities_pass": bool(chk["pass"]),
            "identities_fail": [k for k, v in chk["items"].items() if not v],
            "policy_trace_hash": res.policy_trace_hash,
            "readout_signature": sig, "n_model_calls": n_calls,
            "elapsed_s": round(elapsed, 2)}
    a = mech3.aoi_metrics(cell, variants=(mech3.VAR_TIME,))
    cell["A_t"] = float(a[mech3.VAR_TIME])
    cell["A_t_by_user"] = [float(v) for v in a["per_user"][mech3.VAR_TIME]]
    return cell


def closed_loop(model, eps, plan_seed: int = SEED_PANEL) -> tuple[list[dict], list[dict]]:
    cells, base = [], []
    for ep in eps:
        b = run_cell(None, ep, plan_seed, label="baseline")
        base.append(b)
        c = run_cell(model, ep, plan_seed, label="learned")
        cells.append(c)
        print(f"  [CLOSED] ep{ep} A_t base={b['A_t']:.4f} learned={c['A_t']:.4f} "
              f"({b['elapsed_s']}s/{c['elapsed_s']}s)", flush=True)
    return cells, base


def closed_analysis(cells: list[dict], base: list[dict]) -> dict:
    rows = []
    for c in cells:
        b = next(x for x in base if x["ep"] == c["ep"])
        r_i = []
        for i in range(len(b["cost_true_by_user"])):
            den = float(b["cost_true_by_user"][i])
            num = float(c["cost_true_by_user"][i])
            r_i.append(num / den if den > 0 else float("nan"))
        rows.append({"ep": int(c["ep"]),
                     "A_t_base": float(b["A_t"]), "A_t_learned": float(c["A_t"]),
                     "A_t_ratio": float(c["A_t"] / b["A_t"]) if b["A_t"] > 0 else None,
                     "delta_A_t": float(c["A_t"] - b["A_t"]),
                     "cost_base": float(b["cost_true_per_step"]),
                     "cost_learned": float(c["cost_true_per_step"]),
                     "cost_ratio": float(c["cost_true_per_step"] / b["cost_true_per_step"])
                     if b["cost_true_per_step"] > 0 else None,
                     "r_i": r_i,
                     "n_r_i_gt1": int(sum(1 for v in r_i if np.isfinite(v) and v > 1.0)),
                     "action_diff_trace": (c["policy_trace_hash"] != b["policy_trace_hash"]),
                     "n_model_calls": int(c["n_model_calls"])})
    ratios = np.asarray([r["A_t_ratio"] for r in rows], dtype=float)
    n_up = int((ratios > 1.0).sum())
    med = float(np.median(ratios))
    r_all = [v for r in rows for v in r["r_i"] if np.isfinite(v)]
    if med > OFFLINE_RATIO_MAX:
        v = "LEARNER-FALLS-SHORT"
    elif med < 0.95:
        v = "SUSPECT-LEAK-OR-BUG"
    else:
        v = "LEARNER-MATCHES-WITHIN-5"
    return {"rows": rows, "median_A_t_ratio": med, "n_ratio_gt1": n_up, "n_ep": len(rows),
            "binom_p_two_sided_extreme": binom_p(max(n_up, len(rows) - n_up), len(rows)),
            "verdict": v,
            "cost_ratio_median": float(np.median([r["cost_ratio"] for r in rows])),
            "cost_ratio_n_gt_1": int(sum(1 for r in rows if r["cost_ratio"] > 1.0)),
            "action_diff_n": int(sum(1 for r in rows if r["action_diff_trace"])),
            "r_i_summary": {"n": len(r_all),
                            "median": float(np.median(r_all)) if r_all else None,
                            "n_gt1_users": int(sum(1 for v_ in r_all if v_ > 1.0)),
                            "max": float(np.max(r_all)) if r_all else None},
            "note": "A_t 比 >1 ⇒ 学习臂更陈旧（更差）；R17：∃ r_i>1 ⇒ 只许写'交易'"}


# ==================================================================== 合成视图 / 闸
def synth_view(*, t: int = 5, n: int = 2) -> SchedulerView:
    tel = [
        {"i": 0, "x_hat": 0.70, "P_hat": 0.20, "stamp": 3, "task_aoi": 2,
         "h_mean": 0.10, "h_var": 0.50, "csi_stamp": 4, "csi_aoi": 1},
        {"i": 1, "x_hat": -0.30, "P_hat": 0.90, "stamp": -1, "task_aoi": -1,
         "h_mean": -0.20, "h_var": 0.80, "csi_stamp": -1, "csi_aoi": -1},
    ][:n]
    return SchedulerView({
        "t": int(t), "n_users": n, "budget_left": 100, "budget_total": 128,
        "prior": {"x_mean": 0.0, "x_var": 1.0, "h_log_mean": 0.0, "h_log_var": 1.0},
        "telemetry": tel, "history": (),
    })


class _SignFlip:
    """`G6` 的假模型：把基线信念**翻符号**。`x²` 不变 ⇒ 代价空间损失应逐位不变。"""

    name = "sign_flip"
    privileged = False

    def predict(self, view):
        out = BeliefReadout().predict(view)
        return type(out)(x_hat=-np.asarray(out.x_hat), P=np.asarray(out.P),
                         label=self.name).validate(int(view.n_users))


def gates_zero_episode() -> list[dict]:
    chk: list[dict] = []

    def add(gid: str, ok: bool, detail=None) -> None:
        chk.append({"id": gid, "ok": bool(ok), "detail": detail})

    # G1 —— C1 裁定
    try:
        obj = assert_objective("A_t")
        g1 = (EVAL_PRIMARY == "A_t" and obj["allowed"]
              and obj["evidence_side_only"] is False
              and OBJECTIVE["primary"] == "cost_true_per_step_median")
    except Exception as e:  # noqa: BLE001
        g1, obj = False, {"err": repr(e)}
    add("G1", g1, {"EVAL_PRIMARY": EVAL_PRIMARY, "assert_objective_A_t": obj,
                   "OBJECTIVE_primary": OBJECTIVE["primary"],
                   "why": "C1 裁定：主量 = A_t，OBJECTIVE 不改（只记 legacy）"})

    # G2 —— 零步恒等
    v = synth_view()
    fresh = ResidualReadout(hidden=HIDDEN, seed=0)
    p_new = fresh.predict(v)
    p_base = BeliefReadout().predict(v)
    g2 = (np.array_equal(p_new.x_hat, p_base.x_hat) and np.array_equal(p_new.P, p_base.P))
    add("G2", g2, {"max_abs_diff": float(np.abs(p_new.x_hat - p_base.x_hat).max()),
                   "why": "末层全零 ⇒ 未训练必须与强基线逐位相同；否则'训练有没有帮上忙'失去共同起点",
                   "n_params": int(count_params(fresh))})

    # G3 —— 部署路径 AST 干净
    s = scan_source(LEARN_SRC.read_text(encoding="utf-8"))
    add("G3", not (s["names"] or s["attrs"] or s["imports"]),
        {"file": LEARN_SRC.name, "names": s["names"], "attrs": s["attrs"],
         "imports": s["imports"]})

    # G4 —— 容量（两条腿 + 两条活体负控）
    #   ★ 本模型的 `d_out=1`（只出残差），而 `r2_train_spec.assert_capacity` 的
    #   `width/depth` 分支按 `d_out=2` 造形状（R2.6 骨架的输出是二维）。
    #   所以这里用**本模型的**形状函数 + 那两条守卫本身，而不是硬套 2 维形状。
    from .r2_7_learn import layer_shapes as _shapes1
    from .r2_train_spec import assert_param_budget, assert_structure
    sh = _shapes1(N_FEATURES, HIDDEN, 1)
    cap = {**assert_structure(list(HIDDEN)),
           **assert_param_budget(int(sum(a * b + b for a, b in sh))),
           "shapes": sh, "source": "shapes(d_out=1)"}
    bad = {}
    try:
        assert_structure([65, 32])
        bad["width65"] = "accepted"
    except Exception as e:  # noqa: BLE001
        bad["width65"] = type(e).__name__
    try:
        assert_structure([32, 32, 32])
        bad["depth3"] = "accepted"
    except Exception as e:  # noqa: BLE001
        bad["depth3"] = type(e).__name__
    n_meas = int(count_params(ResidualReadout(hidden=HIDDEN)))
    g4 = (cap["n_params"] == n_meas and cap["n_params"] == 1601
          and bad.get("width65") == "CapacityExceeded"
          and bad.get("depth3") == "CapacityExceeded")
    add("G4", g4, {"measured": cap, "n_params_model": n_meas,
                   "negative_controls": bad,
                   "why": "上限必须能被触发；1601 是形状函数算出来的，不是硬编码"})

    # G5 / G5b —— 数据不相交 + 特征不含真值
    d = assert_disjoint(TRAIN_LIST, DEV_LIST)
    try:
        assert_disjoint((100, 101), (101,), label="neg")
        neg = "accepted"
    except AssertionError:
        neg = "AssertionError"
    add("G5", d["n_inter"] == 0 and neg == "AssertionError",
        {"disjoint": d, "negative_control": neg})
    feats = list(PER_USER_FEATURES) + list(GLOBAL_FEATURES)

    def _perm_key(name: str) -> str:
        """特征名 → 权限表键（表里是 `telemetry.* / prior.* / view.*` 三类前缀）。"""
        if name in ("t", "budget_left", "budget_total"):
            return f"view.{name}"
        if name.startswith("prior."):
            return name
        return f"telemetry.{name}"

    bad_feats = [f for f in feats
                 if f in FORBIDDEN_NAMES
                 or FIELD_PERMISSION.get(_perm_key(f), {}).get("visibility") != "all"]
    missing_keys = [f for f in feats if _perm_key(f) not in FIELD_PERMISSION]
    add("G5b", (len(feats) == N_FEATURES) and not bad_feats and not missing_keys,
        {"n_features": N_FEATURES, "not_all_visible": bad_feats,
         "missing_in_permission_table": missing_keys,
         "perm_keys": [_perm_key(f) for f in feats][:15],
         "why": "标签只进 loss；特征列必须全部在权限表里可见性 = all"})

    # G6 —— 符号敏感性（§2.1 裁定的可检查性）
    y = np.array([0.8, -0.5])
    xb = np.array([0.6, 0.2])
    se_b = float(np.mean((xb - y) ** 2))
    se_f = float(np.mean((-xb - y) ** 2))
    ce_b = float(np.mean((xb ** 2 - y ** 2) ** 2))
    ce_f = float(np.mean(((-xb) ** 2 - y ** 2) ** 2))
    add("G6", (se_f > se_b) and (abs(ce_f - ce_b) < 1e-15),
        {"state_mse_base": se_b, "state_mse_flipped": se_f,
         "cost_mse_base": ce_b, "cost_mse_flipped": ce_f,
         "why": "只翻符号 ⇒ 状态空间损失变差、代价空间损失逐位不变；"
                "若两者都不变，说明状态空间损失也不看符号（§2.1 的裁定就成了空话）"})

    # G7 —— 平台判据有分辨力
    flat = [{"step": s_, "loss_full": 1.0 - 1e-5 * (s_ / 1000)} for s_ in range(100, 1001, 50)]
    drop = [{"step": s_, "loss_full": 1.0 * (0.5 ** (s_ / 200))} for s_ in range(100, 1001, 50)]
    pf, pd = plateau_stats(flat), plateau_stats(drop)
    add("G7", pf["plateau"] and not pd["plateau"],
        {"flat": pf, "dropping": pd, "why": "判据恒 PASS 的话，'未平台不许报 dev' 就没有约束力"})

    # G8 —— 训练确定性（同种子、同数据、同种子下重训两次）
    rng = np.random.default_rng(0)
    Xs = rng.normal(size=(600, N_FEATURES))
    ys = Xs[:, 0] * 0.3 + rng.normal(scale=0.05, size=600)
    xbs = np.zeros(600)
    m1, _ = train_fit(Xs, xbs, ys, seed=11, n_steps=120, log_every=40)
    m2, _ = train_fit(Xs, xbs, ys, seed=11, n_steps=120, log_every=40)
    same = all(np.array_equal(a, b) for a, b in zip(m1.parameters(), m2.parameters()))
    m3, _ = train_fit(Xs, xbs, ys, seed=12, n_steps=120, log_every=40)
    diff = any(not np.allclose(a, b) for a, b in zip(m1.parameters(), m3.parameters()))
    add("G8", same and diff,
        {"same_seed_bitwise_equal": bool(same), "other_seed_differs": bool(diff),
         "why": "同种子必须逐位可复现；换种子必须真的换（否则'三种子分层'是空的）"})

    return chk


# ==================================================================== 主流程
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="CW-R2.7 开训（只训读出）")
    ap.add_argument("--ctrl-only", action="store_true")
    ap.add_argument("--skip-closed", action="store_true", help="只做 P1（调试用，不尽判）")
    ap.add_argument("--train-eps", type=int, default=70)
    ap.add_argument("--steps", type=int, default=N_STEPS)
    ap.add_argument("--tag", default="main")
    args = ap.parse_args(argv)

    t_start = time.perf_counter()
    chk = gates_zero_episode()
    n_fail0 = sum(1 for c in chk if not c["ok"])
    for c in chk:
        print(f"  {'PASS' if c['ok'] else 'FAIL'}  {c['id']}")
    if n_fail0:
        print(f"[ABORT] 零 episode 闸 {n_fail0} 项 FAIL ⇒ 按预注册不看数据")
        return 3
    if args.ctrl_only:
        print(json.dumps(chk, ensure_ascii=False, indent=1, default=str))
        return 0

    cfg = mainline_config(plant=PLANTS[PLANT])
    tr_list = TRAIN_LIST[: int(args.train_eps)]

    t_d = time.perf_counter()
    tr = collect(cfg, "train", tr_list, verbose=True)
    dv = collect(cfg, "dev", DEV_LIST, verbose=True)
    data_s = round(time.perf_counter() - t_d, 1)
    print(f"[DATA] train={tr['X'].shape} dev={dv['X'].shape} {data_s}s")

    results = {}
    models = {}
    for seed in TRAIN_SEEDS:
        m, hist = train_fit(tr["X"], tr["xb"], tr["y"], seed=int(seed),
                            n_steps=int(args.steps), verbose=True)
        models[int(seed)] = m
        off = offline_eval(m, dv)
        plat = plateau_stats(hist)
        results[int(seed)] = {"offline": off, "plateau": plat,
                             "curve": hist, "meta": m.train_meta}
        print(f"[P1 seed={seed}] median_ratio={off['median_ratio']:.4f} "
              f"dir={off['n_ratio_lt1']}/{off['n_ep']} plateau={plat['plateau']} "
              f"rel_drop={plat['rel_drop_last20pct']:.4%}")

    ratios = np.asarray([results[s]["offline"]["median_ratio"] for s in TRAIN_SEEDS])
    dirs = [results[s]["offline"]["n_ratio_lt1"] for s in TRAIN_SEEDS]
    plat_all = all(results[s]["plateau"]["plateau"] for s in TRAIN_SEEDS)
    med_ratio = float(np.median(ratios))
    min_dir = int(min(dirs))
    offline_gate = (med_ratio <= OFFLINE_RATIO_MAX) and (min_dir >= OFFLINE_MIN_DIR)
    if not plat_all:
        offline_verdict = "NOT-CONVERGED"
    elif offline_gate:
        offline_verdict = "OFFLINE-PASSES"
    else:
        offline_verdict = "OFFLINE-FALLS-SHORT"
    print(f"[P1] median_ratio={med_ratio:.4f} min_dir={min_dir}/{len(DEV_LIST)} "
          f"plateau_all={plat_all} ⇒ {offline_verdict}")

    closed, closed_ana = None, None
    if offline_verdict == "OFFLINE-PASSES" and not args.skip_closed:
        seed_best = int(TRAIN_SEEDS[int(np.argmin(ratios))])
        per_seed = {}
        for seed in TRAIN_SEEDS:
            c, b = closed_loop(models[int(seed)], DEV_LIST)
            per_seed[int(seed)] = closed_analysis(c, b)
            print(f"[P2 seed={seed}] {per_seed[int(seed)]['verdict']} "
                  f"median_A_t_ratio={per_seed[int(seed)]['median_A_t_ratio']:.4f}")
        closed = per_seed
        closed_ana = {"selected_seed_by_median_offline_ratio": seed_best,
                      "verdicts_by_seed": {str(s): v["verdict"] for s, v in per_seed.items()}}
    elif not args.skip_closed:
        # 预注册 §11-A1：P1 不过时**仍跑**一次 `EXPLORATORY` 闭环（不影响 H-R2-1 判词）
        per_seed = {}
        for seed in TRAIN_SEEDS:
            c, b = closed_loop(models[int(seed)], DEV_LIST)
            ana_s = closed_analysis(c, b)
            ana_s["verdict_raw"] = ana_s["verdict"]
            ana_s["verdict"] = f"EXP-{ana_s['verdict']}"
            per_seed[int(seed)] = ana_s
            print(f"[EXP seed={seed}] {ana_s['verdict']} "
                  f"median_A_t_ratio={ana_s['median_A_t_ratio']:.4f}")
        closed = per_seed
        closed_ana = {"exploratory": True,
                      "why": f"P1 判 {offline_verdict} ⇒ H-R2-1 未过；本次闭环只作"
                             f"「决策层对信念扰动的敏感性」描述（预注册 §11-A1）",
                      "verdicts_by_seed": {str(s): v["verdict"] for s, v in per_seed.items()}}

    # ---- 落盘
    head = _git(["rev-parse", "HEAD"])
    srchash = _src_hash()
    wt = _git(["hash-object", "experiments/cw_v1/r2_7_learn.py"])
    blob = _git(["rev-parse", f"{head}:experiments/cw_v1/r2_7_learn.py"], allow_fail=True)
    dirty = _git(["status", "--porcelain"])
    outdir = RESULTS / f"cw_r2_7_train_{_now_utc()}_{srchash}"
    outdir.mkdir(parents=True, exist_ok=False)
    print(f"[OUT] {outdir.name}")

    reg = G.run_all()
    g9 = int(reg["n_fail"]) == 0
    print(f"[G9] run_all n_checks={reg['n_checks']} n_fail={reg['n_fail']}")

    try:
        import psutil  # type: ignore
        rss = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
    except Exception:  # noqa: BLE001
        rss = None
    total_s = round(time.perf_counter() - t_start, 1)

    summary = {
        "protocol": "CW-POMDP-v1.1", "stage": "CW-R2.7", "kind": "FIRST-PARAM-UPDATE",
        "prereg": "docs/research/CW-R2.7-开训卡-预注册-v0.1.md",
        "hypothesis": "H-R2-1",
        "model": {"name": "residual_readout", "hidden": list(HIDDEN),
                  "n_params": int(count_params(ResidualReadout(hidden=HIDDEN))),
                  "features": list(PER_USER_FEATURES) + list(GLOBAL_FEATURES),
                  "variance_policy": "frozen_at_base", "loss": "MSE(x_hat, label) state space"},
        "budget": {"n_steps": int(args.steps), "batch": BATCH, "lr": LR, "wd": WD,
                   "train_eps": list(tr_list), "dev_eps": list(DEV_LIST),
                   "seeds": list(TRAIN_SEEDS), "data_s": data_s, "total_s": total_s},
        "objective": {"eval_primary": EVAL_PRIMARY, "legacy_primary": OBJECTIVE["primary"],
                      "secondary": list(OBJECTIVE["secondary"])},
        "offline": {str(s): {"median_ratio": results[s]["offline"]["median_ratio"],
                             "n_ratio_lt1": results[s]["offline"]["n_ratio_lt1"],
                             "n_ep": results[s]["offline"]["n_ep"],
                             "cost_ratio_median": results[s]["offline"]["cost_ratio_median"],
                             "max_abs_residual": results[s]["offline"]["max_abs_residual"],
                             "plateau": results[s]["plateau"],
                             "meta": results[s]["meta"],
                             "per_ep": results[s]["offline"]["per_ep"]}
                    for s in TRAIN_SEEDS},
        "offline_gate": {"median_ratio": med_ratio, "min_dir": min_dir,
                         "n_ep": len(DEV_LIST), "threshold": OFFLINE_RATIO_MAX,
                         "min_dir_required": OFFLINE_MIN_DIR,
                         "plateau_all": plat_all, "verdict": offline_verdict},
        "closed": closed_ana,
        "closed_by_seed": ({str(s): {k: v for k, v in per_seed[s].items() if k != "rows"}
                            for s in per_seed} if closed else None),
        "closed_rows": ({str(s): per_seed[s]["rows"] for s in per_seed} if closed else None),
        "gates": {"zero_episode": chk, "n_fail_zero_episode": n_fail0,
                  "run_all": {"n_checks": reg["n_checks"], "n_fail": reg["n_fail"]},
                  "G9": g9},
        "head": head, "src_hash": srchash,
        "peak_rss_mb": round(rss, 1) if rss else None,
    }
    (outdir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False,
                                                    indent=1, default=float), encoding="utf-8")
    (outdir / "gates.json").write_text(json.dumps({"zero_episode": chk, "G9": {
        "n_checks": reg["n_checks"], "n_fail": reg["n_fail"]}}, ensure_ascii=False,
        indent=1, default=str), encoding="utf-8")
    (outdir / "curves.json").write_text(json.dumps(
        {str(s): results[s]["curve"] for s in TRAIN_SEEDS}, ensure_ascii=False,
        indent=1, default=float), encoding="utf-8")

    meta = {"command": f"python -m experiments.cw_v1.r2_7_train --steps {args.steps} "
                       f"--tag {args.tag}",
            "timestamp_local": datetime.now().astimezone().isoformat(),
            "python": os.sys.version.split()[0], "numpy": np.__version__,
            "head": head, "src_hash": srchash,
            "worktree_sha_r2_7_learn": wt, "blob_sha_r2_7_learn": blob,
            "git_dirty_before_run": dirty.splitlines()[:40],
            "protocol": "CW-POMDP-v1.1", "event_pairing": "slot",
            "n_train_samples": int(tr["n_samples"]), "n_dev_samples": int(dv["n_samples"]),
            "data_elapsed_s": data_s, "total_elapsed_s": total_s,
            "peak_rss_mb": round(rss, 1) if rss else None,
            "training": "residual readout only (no policy training)",
            "gates_n_fail": n_fail0 + (0 if g9 else 1)}
    (outdir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                     encoding="utf-8")

    print(f"[DONE] {outdir.name} total={total_s}s peak={meta['peak_rss_mb']}MB "
          f"g9={g9} offline={offline_verdict}")
    print(json.dumps({"offline_verdict": offline_verdict, "median_ratio": med_ratio,
                      "dir": f"{min_dir}/{len(DEV_LIST)}",
                      "plateau_all": plat_all,
                      "closed": (closed_ana if closed is None else
                                 {k: v for k, v in closed_ana.items()})},
                     ensure_ascii=False, indent=1, default=str))
    return 0 if (n_fail0 == 0 and g9) else 2


if __name__ == "__main__":
    import sys
    sys.exit(main())
