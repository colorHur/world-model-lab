# -*- coding: utf-8 -*-
"""`CW-R2.8`：**「学习空间到底在哪里」** —— 两个零训练、零规划器的判定。

预注册：`docs/research/CW-R2.8-学习空间判定-预注册-v0.1.md`（写于 2026-10-25 19:49，跑前）

两个问题
--------
**Q-A**：`SchedulerView.telemetry.x_hat/P_hat` 能不能**只靠公开历史 + 协议公开常量逐位重建**？
  能 ⇒ R2.7 §7.2 的 A 方案（"去掉 `x̂/P̂` 输入让模型从历史重建"）**信息缺口 = 0**：
  目标是输入的**确定性函数**，学习只能是重复实现，不是"学到新信息"。

**Q-B**：把"已知生成参数"换成"**参数未知**"（v4 §4 允许的入口）后，参数不确定性值多少 MSE？

★ 本文件**不训练**、**不跑规划器**（`run_episode` = 0）、**不改被测物一字节**、不改任何常量。
★ A 支直接驱动 `CWWorld.step`（1536 步）；B 支是**合成标量 AR**，不碰 `CWWorld`。

B 支为什么关掉控制（`u ≡ 0`）
------------------------------
若三臂各自用 `u_t = −K·x̂_t`，则"估计不同 ⇒ 控制不同 ⇒ **真值轨迹不同**"，
估计差与闭环差会混在一个 MSE 里，判词就说不清是谁的功劳。
本支只问"参数不确定性值多少"，⇒ 开环纯滤波。闭环效应属下一轮，不在本轮判词内。
（这条是**写码时**发现的，已登记在预注册 §9 `A1`，跑任何 B 支数字之前。）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path

import numpy as np

from .config import mainline_config
from .r1_14_exactbase import _git, _now_utc, _src_hash
from .streams import make_episode_rngs
from .world import PUBLIC_HISTORY_FIELDS, Action, CWWorld

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results" / "cw_v1"
HERE = Path(__file__).resolve().parent
SELF = Path(__file__).resolve()

PROTOCOL = "CW-POMDP-v1.1"
ROUND = "CW-R2.8"

#: 重建器**只允许**读的公开历史键（⊆ `PUBLIC_HISTORY_FIELDS`；闸 `S6` 双向核对）
READ_KEYS: tuple[str, ...] = (
    "t", "action", "user", "data_delivered", "ack", "y_t", "budget_left",
)
assert set(READ_KEYS) <= set(PUBLIC_HISTORY_FIELDS), "READ_KEYS 必须是公开历史白名单的子集"

TOL = 1e-12

#: A 支动作脚本（3 档）
SCRIPTS: dict[str, str] = {
    "dense": "每槽 SEND user=(t%2)",
    "sparse": "仅 t%8==0 时 SEND user=(t%2)",
    "mixed": "t%3==0→PROBE；t%3==1→SEND user=((t//3)%2)；否则 HOLD",
}
EPISODES: tuple[int, ...] = (100, 101)

#: B 支网格（预注册 §9-A1 修订后：剂量 = 参数不确定性宽度 `w_A` × 送达率 `p`）
B_W_A: tuple[float, ...] = (0.005, 0.02)
B_P_DELIVER: tuple[float, ...] = (0.5, 0.1, 0.02)
B_SIGMA_W: float = 0.02          # 与 `BOUNDED_PLANT.sigma_w[0]` 同量级，全格固定
B_N_EP = 1000
B_T = 64
B_THETA = (0.0, 0.25, 0.5, 0.75, 1.0)   # 网格位置（0..1），实值 = 1 + w·(2u−1)
H_PRESENT, H_NEGLIGIBLE = 0.05, 0.01


# ==================================================================== 公开常量
def riccati_k(A: float, B: float, q: float, r: float) -> float:
    """标量离散 LQR 增益（**独立重算**，与 `CWWorld.nominal_gain` 对账 = 闸 `S11`）。"""
    P = float(q)
    for _ in range(200000):
        P_new = q + A * A * P - (A * P * B) ** 2 / (r + B * B * P)
        if abs(P_new - P) < 1e-15:
            P = P_new
            break
        P = P_new
    return float(A * P * B / (r + B * B * P))


def public_consts(cfg, w: CWWorld) -> dict:
    """协议 v1.1 下**所有可部署方法共享**的生成参数（v4.1："已知生成参数"）。"""
    A = [float(a) for a in cfg.plant.A]
    return {
        "A": A,
        "B": float(cfg.plant.B),
        "sigma_w": [float(s) for s in cfg.plant.sigma_w],
        "sigma_meas": float(cfg.plant.sigma_meas),
        "prior_mean": 0.0,
        "prior_var": float(cfg.plant.prior_var),
        "K": [riccati_k(a, float(cfg.plant.B), float(cfg.plant.q_cost), float(cfg.plant.r_cost))
              for a in A],
        "K_world": [float(w.nominal_gain(i)) for i in range(cfg.plant.n_users)],
    }


# ==================================================================== 重建器（A 支）
class RecReader:
    """把公开记录包成**只能读白名单键**的只读通道（闸 `S6` 的实现）。

    ★ 故意做成"读一次记一次"：这样"重建器到底读了哪些键"是**运行期可观测**的，
      不靠人读源码保证。
    """

    def __init__(self, records: list[dict]) -> None:
        self._recs = records
        self.touched: set[str] = set()

    def __len__(self) -> int:
        return len(self._recs)

    def get(self, k: int, key: str):
        if key not in PUBLIC_HISTORY_FIELDS:
            raise PermissionError(f"键 {key!r} 不在公开历史白名单内（重建器无权读）")
        self.touched.add(key)
        return self._recs[k][key]


def reconstruct(recs: RecReader, cs: dict, n_users: int, *,
                y_perm: np.ndarray | None = None,
                drop_k: int | None = None,
                sigma_meas: float | None = None,
                gain_mode: str = "exact") -> tuple[np.ndarray, np.ndarray]:
    """从公开历史重建**每槽开始**的 `x̂/P̂`（返回形状 `(T+1, n_users)`）。

    `y_perm`    : 送达记录的 `y_t` 置换（负控 `S2`）
    `drop_k`    : 把第 k 条记录当作"没送达"（负控 `S3`）
    `sigma_meas`: 覆盖量测噪声 sd（负控 `S4`）
    `gain_mode` : `"exact"` / `"zero"` / `"one"`（变异测试 `S7`）
    """
    T = len(recs)
    A = cs["A"]
    B = cs["B"]
    sw = cs["sigma_w"]
    K = cs["K"]
    sm = float(cs["sigma_meas"] if sigma_meas is None else sigma_meas)
    var_m = sm * sm

    xs = np.zeros((T + 1, n_users))
    Ps = np.zeros((T + 1, n_users))
    xs[0, :] = float(cs["prior_mean"])
    Ps[0, :] = float(cs["prior_var"])

    # 先把 y 序列取出来（置换一次，之后按位置用）
    ys = [recs.get(k, "y_t") for k in range(T)]
    if y_perm is not None:
        idx = [k for k in range(T) if ys[k] is not None]
        if len(idx) > 1:
            src = np.asarray([ys[k] for k in idx], dtype=float)[np.asarray(y_perm)]
            for pos, k in enumerate(idx):
                ys[k] = float(src[pos])

    for k in range(T):
        x_t = xs[k]
        P_t = Ps[k]
        u = -np.asarray(K) * x_t
        # 名义滚动（无观测路径）
        x_next = np.asarray(A) * x_t + B * u
        P_next = np.asarray(A) ** 2 * P_t + np.asarray(sw) ** 2
        # 观测路径
        delivered = int(recs.get(k, "data_delivered") or 0)
        if drop_k is not None and k == drop_k:
            delivered = 0
        if delivered and ys[k] is not None:
            i = int(recs.get(k, "user"))
            if gain_mode == "zero":
                g = 0.0
            elif gain_mode == "one":
                g = 1.0
            else:
                g = P_t[i] / (P_t[i] + var_m)
            xc = x_t[i] + g * (float(ys[k]) - x_t[i])
            Pc = (1.0 - g) * P_t[i]
            x_next[i] = float(A[i]) * xc + B * u[i]
            P_next[i] = float(A[i]) ** 2 * Pc + float(sw[i]) ** 2
        xs[k + 1] = x_next
        Ps[k + 1] = P_next
    return xs, Ps


# ==================================================================== A 支面板
def run_script(cfg, script: str, episode: int) -> dict:
    """按脚本驱动 `CWWorld`，返回 `views`（每槽开始）与 `public`（每槽末追加）。"""
    T = int(cfg.protocol.horizon)
    rngs, sids = make_episode_rngs(cfg.root_seed, "dev", episode)
    w = CWWorld(cfg, rngs, sids, rho=float(cfg.channel.rho_panel[0]))
    w.budget_total = int(T)
    w._x_hat[:] = 0.0
    w._P_hat[:] = float(cfg.plant.prior_var)
    w._stamp[:] = -1

    views = []
    n_steps = 0
    for t in range(T):
        views.append(w.view())          # 槽开始视图（比对目标的 t 时刻那一份）
        if script == "dense":
            act = Action(kind="SEND", user=t % cfg.plant.n_users)
        elif script == "sparse":
            act = (Action(kind="SEND", user=t % cfg.plant.n_users) if t % 8 == 0
                   else Action(kind="HOLD"))
        elif script == "mixed":
            if t % 3 == 0:
                act = Action(kind="PROBE", user=t % cfg.plant.n_users)
            elif t % 3 == 1:
                act = Action(kind="SEND", user=(t // 3) % cfg.plant.n_users)
            else:
                act = Action(kind="HOLD")
        else:  # pragma: no cover
            raise KeyError(script)
        w.step(act)
        n_steps += 1
    views.append(w.view())              # 末槽之后（t = T）那一份
    public = [{k: r[k] for k in PUBLIC_HISTORY_FIELDS} for r in w._public_records]
    return {"views": views, "public": public, "n_steps": n_steps,
            "stream_ids": sids.as_record(), "n_ack": w.n_ack, "n_nack": w.n_nack, "n_probe": w.n_probe,
            "t_final": w.t}


def telemetry_arrays(view, n_users: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray([float(tel["x_hat"]) for tel in view.telemetry], dtype=float)
    P = np.asarray([float(tel["P_hat"]) for tel in view.telemetry], dtype=float)
    return x, P


def panel_a(cfg) -> tuple[dict, dict]:
    """A 支：6 跑 × (恒等 + 4 负控 + 2 变异)。"""
    n_users = int(cfg.plant.n_users)
    runs, controls = {}, {}
    total_steps = 0
    for script in SCRIPTS:
        for ep in EPISODES:
            key = f"{script}@{ep}"
            r = run_script(cfg, script, ep)
            total_steps += r["n_steps"]
            cs = public_consts(cfg, CWWorld(cfg, *make_episode_rngs(cfg.root_seed, "dev", ep),
                                            rho=float(cfg.channel.rho_panel[0])))
            recs = RecReader(r["public"])

            xs, Ps = reconstruct(recs, cs, n_users)
            tgt_x = np.asarray([telemetry_arrays(v, n_users)[0] for v in r["views"]])
            tgt_P = np.asarray([telemetry_arrays(v, n_users)[1] for v in r["views"]])
            dx = float(np.max(np.abs(xs - tgt_x)))
            dP = float(np.max(np.abs(Ps - tgt_P)))
            # 初态对账（闸 S8）
            dx0 = float(np.max(np.abs(xs[0] - tgt_x[0])))
            dP0 = float(np.max(np.abs(Ps[0] - tgt_P[0])))

            # ---- 负控 S2：打乱 y_t
            delivered_idx = [k for k in range(len(r["public"])) if r["public"][k]["y_t"] is not None]
            perm = np.arange(len(delivered_idx))[::-1].copy()   # 确定性"逆序"= 一次无随机置换
            xs_s, Ps_s = reconstruct(RecReader(r["public"]), cs, n_users, y_perm=perm)
            # ---- 负控 S3：丢掉第一条送达观测
            drop_k = delivered_idx[0] if delivered_idx else None
            xs_d, _ = reconstruct(RecReader(r["public"]), cs, n_users, drop_k=drop_k)
            # ---- 负控 S4：σ_meas × 1.5
            xs_m, Ps_m = reconstruct(RecReader(r["public"]), cs, n_users,
                                     sigma_meas=1.5 * cs["sigma_meas"])
            # ---- 变异 S7
            xs_g0, Ps_g0 = reconstruct(RecReader(r["public"]), cs, n_users, gain_mode="zero")
            xs_g1, Ps_g1 = reconstruct(RecReader(r["public"]), cs, n_users, gain_mode="one")

            runs[key] = {
                "script": script, "episode": ep,
                "max_abs_dx": dx, "max_abs_dP": dP,
                "max_abs_dx_init": dx0, "max_abs_dP_init": dP0,
                "n_delivered_records": len(delivered_idx),
                "n_ack": r["n_ack"], "n_nack": r["n_nack"], "n_probe": r["n_probe"],
                "keys_touched": sorted(recs.touched),
                "stream_ids": r["stream_ids"],
            }
            controls[key] = {
                "shuffle_y_max_abs_dx": float(np.max(np.abs(xs_s - tgt_x))),
                "shuffle_y_max_abs_dP": float(np.max(np.abs(Ps_s - tgt_P))),
                "drop_one_max_abs_dx": float(np.max(np.abs(xs_d - tgt_x))),
                "sigma15_max_abs_dx": float(np.max(np.abs(xs_m - tgt_x))),
                "sigma15_max_abs_dP": float(np.max(np.abs(Ps_m - tgt_P))),
                "mut_gain0_max_abs_dx": float(np.max(np.abs(xs_g0 - tgt_x))),
                "mut_gain0_max_abs_dP": float(np.max(np.abs(Ps_g0 - tgt_P))),
                "mut_gain1_max_abs_dx": float(np.max(np.abs(xs_g1 - tgt_x))),
                "mut_gain1_max_abs_dP": float(np.max(np.abs(Ps_g1 - tgt_P))),
            }
    return {"runs": runs, "n_world_steps": total_steps}, controls


# ==================================================================== B 支：MMKF
def b_cell(w_A: float, p_deliver: float, sigma_w: float, *, n_ep: int, T: int,
           seed: int, cfg) -> dict:
    """参数未知（`A`）下的三臂滤波：`plug` / `mmkf` / `oracle`。**开环**（`u ≡ 0`）。

    ★ 为什么开环：闭环时"各臂控制不同 ⇒ 真值轨迹不同"，估计差与闭环差会混在一个 MSE 里。
      开环下 `x_t` 由 `(A_true, x0, w)` 完全决定、**与任何滤波器无关** ⇒ 三臂共用同一条真值轨迹，
      MSE 差只可能是估计器差。（预注册 §9 `A1` 的修订理由。）
    ★ 为什么 `w_A ≤ 0.02`：开环 `u≡0` 时 `|x_T| ~ A^T`；`T=64` 下 `A=1.02` 已放大 3.5×，
      再宽就让 MSE 被"轨迹尺度"主导而不是估计质量。每格都报 `max|x|` 作稳定性守卫。
    """
    lo, hi = 1.0 - float(w_A), 1.0 + float(w_A)
    theta = np.linspace(lo, hi, len(B_THETA))
    rng = np.random.default_rng(seed)
    A_true = rng.uniform(lo, hi, n_ep)
    x0 = rng.standard_normal(n_ep) * float(np.sqrt(cfg.plant.prior_var))
    W = rng.standard_normal((n_ep, T)) * float(sigma_w)
    Z = rng.standard_normal((n_ep, T)) * float(cfg.plant.sigma_meas)
    Z2 = rng.standard_normal((n_ep, T)) * float(cfg.plant.sigma_meas)
    D = rng.random((n_ep, T)) < float(p_deliver)

    var_m = float(cfg.plant.sigma_meas) ** 2
    var_w = float(sigma_w) ** 2
    K = len(theta)

    # ---- 真值轨迹（开环 ⇒ 与滤波器无关，三臂共用）
    x = np.empty((n_ep, T))
    xt = x0.copy()
    for t in range(T):
        x[:, t] = xt
        xt = A_true * xt + W[:, t]

    def arm(a_val, Zuse: np.ndarray) -> np.ndarray:
        """标量卡尔曼（`a_val` 可为标量或 `(n_ep,)`）。"""
        m = np.zeros(n_ep)
        P = np.full(n_ep, float(cfg.plant.prior_var))
        out = np.empty((n_ep, T))
        a = np.asarray(a_val, dtype=float)
        for t in range(T):
            m = a * m
            P = a * a * P + var_w
            y = x[:, t] + Zuse[:, t]
            g = P / (P + var_m)
            m = np.where(D[:, t], m + g * (y - m), m)
            P = np.where(D[:, t], (1.0 - g) * P, P)
            out[:, t] = m
        return out

    def arm_mmkf() -> np.ndarray:
        m = np.zeros((n_ep, K))
        P = np.full((n_ep, K), float(cfg.plant.prior_var))
        logw = np.full((n_ep, K), -np.log(K))
        out = np.empty((n_ep, T))
        for t in range(T):
            m = theta[None, :] * m
            P = theta[None, :] ** 2 * P + var_w
            y = (x[:, t] + Z[:, t])[:, None]
            S = P + var_m
            ll = -0.5 * ((y - m) ** 2 / S + np.log(S))
            logw = np.where(D[:, t][:, None], logw + ll, logw)
            logw = logw - logw.max(axis=1, keepdims=True)
            w = np.exp(logw)
            w = w / w.sum(axis=1, keepdims=True)
            logw = np.log(w + 1e-300)
            g = P / S
            m = np.where(D[:, t][:, None], m + g * (y - m), m)
            P = np.where(D[:, t][:, None], (1.0 - g) * P, P)
            out[:, t] = (w * m).sum(axis=1)
        return out

    mse = lambda est: ((est - x) ** 2).mean(axis=1)      # 逐 episode
    e_plug = mse(arm(1.00, Z))
    e_mmkf = mse(arm_mmkf())
    e_orc = mse(arm(A_true, Z))
    e_null = mse(arm(1.00, Z2))                          # 地板：同滤波器、另一条观测噪声流

    rg = np.random.default_rng(seed + 7)
    boots = np.empty(1000)
    for b in range(1000):
        idx = rg.integers(0, n_ep, n_ep)
        boots[b] = 1.0 - e_mmkf[idx].mean() / e_plug[idx].mean()
    lo_ci, hi_ci = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    h = float(1.0 - e_mmkf.mean() / e_plug.mean())
    floor_rel = float(abs(e_plug.mean() - e_null.mean()) / e_plug.mean())
    return {
        "w_A": float(w_A), "p_deliver": float(p_deliver), "sigma_w": float(sigma_w),
        "mse_plug": float(e_plug.mean()), "mse_mmkf": float(e_mmkf.mean()),
        "mse_oracle": float(e_orc.mean()),
        "max_abs_x": float(np.abs(x).max()),
        "h_B": h, "h_B_ci95": [lo_ci, hi_ci], "floor_rel": floor_rel,
        "theta_grid": [float(v) for v in theta],
        "n_ep": int(n_ep), "T": int(T), "seed": int(seed),
    }


# ==================================================================== 判词
def judge(a: dict, ctrl: dict, b: list[dict]) -> dict:
    worst_dx = max(r["max_abs_dx"] for r in a["runs"].values())
    worst_dP = max(r["max_abs_dP"] for r in a["runs"].values())
    # 负控必须"超差"才叫有分辨力
    ctrl_min_dx = min(v["shuffle_y_max_abs_dx"] for v in ctrl.values())
    ctrl_drop_min = min(v["drop_one_max_abs_dx"] for v in ctrl.values())
    ctrl_sig_dx = min(v["sigma15_max_abs_dx"] for v in ctrl.values())
    ctrl_sig_dP = min(v["sigma15_max_abs_dP"] for v in ctrl.values())
    mut_min = min(min(v["mut_gain0_max_abs_dx"], v["mut_gain1_max_abs_dx"]) for v in ctrl.values())

    a_exact = (worst_dx <= TOL and worst_dP <= TOL
               and ctrl_min_dx > 1e-3 and ctrl_drop_min > 1e-6
               and ctrl_sig_dx > 1e-6 and ctrl_sig_dP > 1e-6)
    a_judge = "A-RECONSTRUCT-EXACT" if a_exact else "A-RECONSTRUCT-FAILS"

    any_present = any(c["h_B"] >= H_PRESENT and c["h_B_ci95"][0] > 0.0
                      and c["h_B"] > 5.0 * c["floor_rel"] for c in b)
    all_neg = all(c["h_B"] < H_NEGLIGIBLE for c in b)
    if any_present:
        b_judge = "B-HEADROOM-PRESENT"
    elif all_neg:
        b_judge = "B-HEADROOM-NEGLIGIBLE"
    else:
        b_judge = "B-HEADROOM-UNRESOLVED"

    if a_exact and b_judge == "B-HEADROOM-NEGLIGIBLE":
        overall_prereg = "R2-BELIEF-CLOSED-FORM"
    elif a_exact and b_judge == "B-HEADROOM-PRESENT":
        overall_prereg = "LEARNING-SPACE-AT-UNKNOWN-PARAMS"
    elif a_exact and b_judge == "B-HEADROOM-UNRESOLVED":
        overall_prereg = "B-UNRESOLVED-KEEP-R2"
    else:
        overall_prereg = "A-STILL-OPEN"
    # ★ 预注册 §9-A3 的更正：`PRESENT` 只说明"参数自适应估计器优于名义参数估计器"（本卡 §7-3 原话），
    #   而 MMKF 在离散 θ 下是**精确**贝叶斯 ⇒ 学习模型的期望上界是"打平"。
    #   所以"信念层无信息缺口"一旦成立，R2 在信念层的结论就与 B 的取值无关。
    overall_corrected = "R2-BELIEF-CLOSED-FORM" if a_exact else "A-STILL-OPEN"
    return {"A": a_judge, "B": b_judge,
            "overall_prereg": overall_prereg, "overall_corrected": overall_corrected,
            "overall": overall_corrected,
            "worst_abs_dx": worst_dx, "worst_abs_dP": worst_dP,
            "ctrl_shuffle_min_dx": ctrl_min_dx, "ctrl_drop_min_dx": ctrl_drop_min,
            "ctrl_sigma15_min_dx": ctrl_sig_dx, "ctrl_sigma15_min_dP": ctrl_sig_dP,
            "mut_min_dx": mut_min}


# ==================================================================== 主
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="CW-R2.8 学习空间判定（零训练、零规划器）")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--b-ep", type=int, default=B_N_EP)
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    srchash = _src_hash()
    outdir = Path(args.outdir) if args.outdir else RESULTS / f"cw_r2_8_space_{_now_utc()}_{srchash}"
    outdir.mkdir(parents=True, exist_ok=False)
    cfg = mainline_config()

    gates: dict[str, dict] = {}
    summary: dict = {"round": ROUND, "protocol": PROTOCOL, "src_hash": srchash}

    def gate(name: str, ok: bool, **kw):
        gates[name] = {"pass": bool(ok), **kw}

    try:
        # ---------------- A 支
        ta = time.perf_counter()
        a, ctrl = panel_a(cfg)
        ta = time.perf_counter() - ta
        (outdir / "panel_a.json").write_text(json.dumps({"runs": a["runs"], "controls": ctrl},
                                                        ensure_ascii=False, indent=2), "utf-8")

        # ---------------- B 支
        tb = time.perf_counter()
        b = []
        for i, wA in enumerate(B_W_A):
            for j, p in enumerate(B_P_DELIVER):
                b.append(b_cell(wA, p, B_SIGMA_W, n_ep=int(args.b_ep), T=B_T,
                                seed=20261025 + 1000 * i + 10 * j, cfg=cfg))
        tb = time.perf_counter() - tb
        (outdir / "panel_b.json").write_text(json.dumps(b, ensure_ascii=False, indent=2), "utf-8")

        j = judge(a, ctrl, b)

        # ---------------- 闸
        gate("S1", all(r["max_abs_dx"] <= TOL and r["max_abs_dP"] <= TOL for r in a["runs"].values()),
             worst_dx=j["worst_abs_dx"], worst_dP=j["worst_abs_dP"], tol=TOL)
        gate("S2", j["ctrl_shuffle_min_dx"] > 1e-3, min_dx=j["ctrl_shuffle_min_dx"], need=">1e-3")
        gate("S3", j["ctrl_drop_min_dx"] > 1e-6, min_dx=j["ctrl_drop_min_dx"], need=">1e-6")
        gate("S4", j["ctrl_sigma15_min_dx"] > 1e-6 and j["ctrl_sigma15_min_dP"] > 1e-6,
             min_dx=j["ctrl_sigma15_min_dx"], min_dP=j["ctrl_sigma15_min_dP"], need=">1e-6")
        # S5：P̂ 只依赖送达指示 ⇒ 打乱 y 后 P̂ 仍逐位等于视图
        s5 = max(v["shuffle_y_max_abs_dP"] for v in ctrl.values())
        gate("S5", s5 <= TOL, worst_abs_dP=s5, tol=TOL)
        # S6：白名单（键集 ⊆ READ_KEYS 且 ⊆ PUBLIC_HISTORY_FIELDS）
        touched = set()
        for r in a["runs"].values():
            touched |= set(r["keys_touched"])
        gate("S6", touched <= set(READ_KEYS),
             touched=sorted(touched), allowed=sorted(READ_KEYS))
        # S7：变异必须有牙（把 gain 换掉 ⇒ S1 必须红）
        gate("S7", j["mut_min_dx"] > TOL, min_mut_dx=j["mut_min_dx"],
             note="gain∈{0,1} 两种变异里最小的那个都必须让恒等失败")
        # S8：初态对账
        d8 = max(max(r["max_abs_dx_init"], r["max_abs_dP_init"]) for r in a["runs"].values())
        gate("S8", d8 <= TOL, worst_abs=d8, tol=TOL)
        # S9：没有规划器
        gate("S9", a["n_world_steps"] == len(SCRIPTS) * len(EPISODES) * int(cfg.protocol.horizon),
             n_world_steps=a["n_world_steps"],
             expect=len(SCRIPTS) * len(EPISODES) * int(cfg.protocol.horizon))
        # S10：MC 地板必须 < 5%（诊断门槛；"h_B > 5×floor" 只作 judge() 的判词条件）
        s10 = all(c["floor_rel"] < 0.05 for c in b)
        gate("S10", s10, detail=[{"cell": f"w={c['w_A']},p={c['p_deliver']}",
                                  "floor_rel": c["floor_rel"], "h_B": c["h_B"],
                                  "max_abs_x": c["max_abs_x"]} for c in b])
        # S11：K 独立重算 == 世界内部
        cs = public_consts(cfg, CWWorld(cfg, *make_episode_rngs(cfg.root_seed, "dev", 100),
                                        rho=float(cfg.channel.rho_panel[0])))
        d11 = max(abs(x - y) for x, y in zip(cs["K"], cs["K_world"]))
        gate("S11", d11 <= 1e-12, worst_abs=d11, K=cs["K"], K_world=cs["K_world"])

        n_fail = sum(1 for g in gates.values() if not g["pass"])
        summary.update({
            "judgement": j,
            "gates": {"n": len(gates), "n_fail": n_fail},
            "elapsed_s": {"panel_a": ta, "panel_b": tb, "total": time.perf_counter() - t0},
            "config": {"plant": {k: v for k, v in cfg.plant.__dict__.items()},
                       "channel": {k: v for k, v in cfg.channel.__dict__.items()},
                       "protocol": {k: v for k, v in cfg.protocol.__dict__.items()}},
            "counts": {"run_episode": 0, "world_steps": a["n_world_steps"],
                       "train_steps": 0, "grad_steps": 0},
            "git_head": _git(["rev-parse", "HEAD"], allow_fail=True),
            "src_hash_self": hashlib.sha256(SELF.read_bytes()).hexdigest()[:16],
        })
    except Exception as exc:                                  # noqa: BLE001 —— 失败也要留证据
        summary["exception"] = {"type": type(exc).__name__, "msg": str(exc),
                                "trace": traceback.format_exc()[-2000:]}
        n_fail = -1

    (outdir / "gates.json").write_text(json.dumps(gates, ensure_ascii=False, indent=2), "utf-8")
    (outdir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps({"outdir": str(outdir), "n_fail": n_fail,
                      "judgement": summary.get("judgement", {}),
                      "elapsed_s": summary.get("elapsed_s", {})},
                     ensure_ascii=False, indent=2))
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
