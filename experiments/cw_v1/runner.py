# -*- coding: utf-8 -*-
"""CW-POMDP-v1 的 episode 运行器：一个策略在一条外生轨迹上跑完 T 个时隙。

★★ 配对的设计（v4 §7「同一轨迹配对」）
    外生子流 `plant` / `channel` / `packet` 由 (root_seed, split, episode) 决定，
    **与策略无关**。所以同一个 (split, episode) 下，不同策略面对的是**同一串**
    过程噪声、同一条信道序列、同一串「如果发得出去」的随机数。
    策略的动作只改变**消费哪些**，不改变**未来有什么**。
    ⚠️ 但要注意：`packet` 流是**按发送次数**取值而不是按时隙，
    所以不同发送次数会读取流的不同位置 —— 这不是泄漏，但也**不能**说
    "两策略的丢包序列逐位相同"。这一点必须在证据里写明，不能含糊。

★ 恒等关系（`check_identities`）：一个 episode 的记账必须满足
        n_slots     = n_probe + n_attempted + n_idle
        budget_used = c_probe*n_probe + c_data*n_attempted        (资源只花在主动动作上)
        budget_used <= budget_total
        n_ack + n_nack = n_attempted
        n_delivered    = n_ack
        budget_wasted  = c_data*(n_attempted - n_delivered)       (失败也扣)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, asdict

import numpy as np

from .config import CWConfig
from .streams import make_episode_rngs
from .world import Action, CWWorld, BudgetExhausted, action_space


@dataclass
class EpisodeResult:
    split: str
    episode: int
    policy: str
    budget_total: int
    rho: float
    T: int
    # ---- 主指标 ----
    cost_est_per_step: float      # 先验下任务成本均值（所有策略同口径）
    cost_true_per_step: float     # 真状态成本均值（诊断，不进主表）
    # ---- 逐用户 ----
    cost_est_by_user: list[float]
    cost_true_by_user: list[float]
    # ---- 记账 ----
    n_slots: int
    n_probe: int
    n_attempted: int
    n_delivered: int
    n_ack: int
    n_nack: int
    n_idle: int
    n_forced_hold: int      # 预算被拒后强制 HOLD 的时隙数（>0 表示策略被预算截断）
    probe_resource: int
    attempted_resource: int
    delivered_resource: int
    budget_used: int
    budget_attempted: int
    budget_wasted: int
    budget_left: int
    # ---- 诊断 ----
    mean_task_aoi: float
    mean_csi_aoi: float
    final_est_err: float
    stream_ids: dict
    seed_plan: dict
    elapsed_s: float
    config_fingerprint: str
    policy_trace_hash: str


def _trace_hash(recs) -> str:
    """动作序列的内容哈希：用来判断"只改真值时动作是否变了"。"""
    import hashlib

    blob = "|".join(f"{r.t}:{r.action}:{r.user}" for r in recs).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def run_episode(
    cfg: CWConfig,
    policy,
    *,
    split: str,
    episode: int,
    budget_total: int,
    rho: float,
    allow_truth_keys: bool = False,
    correct_receiver: bool = True,
    perturb_truth: str | None = None,
) -> EpisodeResult:
    """跑一个 episode。

    `perturb_truth`：`"x"` / `"h"` 在**运行时**把真值换成一个完全不同的序列，
    用于反例 1 的"只改真值 ⇒ 动作应逐位不变"。这只改真值，不改任何子流。
    """
    t_start = time.perf_counter()
    rngs, sids = make_episode_rngs(cfg.root_seed, split, episode)
    w = CWWorld(cfg, rngs, sids, allow_truth_keys=allow_truth_keys,
                correct_receiver=correct_receiver, rho=rho)
    w.budget_total = int(budget_total)

    # 显式重置到公开先验（构造时已是初值，这里写出来是为了防止后续改动漏掉）
    w._x_hat[:] = 0.0
    w._P_hat[:] = float(cfg.plant.prior_var)
    w._stamp[:] = -1
    w._m_h[:] = 0.0
    w._v_h[:] = float(cfg.channel.h_log_init_std) ** 2
    w._csi_stamp[:] = -1

    T = int(cfg.protocol.horizon)
    for step_i in range(T):
        # ★ 扰动必须在**推进之前**注入，且要覆盖推进后新生成的值：
        #   `w.step` 末尾会 `_advance_physics` / `_advance_channel` 重写真值。
        #   第一版把注入放在 step **之前**，于是扰动只影响"本槽读取"，
        #   而本槽的真实演化又被覆盖 ⇒ 扰动实际错位一步 ⇒ 反例 1d FAIL 误判。
        #   正确做法：在 step 内部推进之后注入。见 `CWWorld.step(..., truth_override=)`。
        w.truth_override = (_truth_series(perturb_truth, step_i, w.n_users)
                            if perturb_truth is not None else None)
        v = w.view()
        try:
            act = policy(v)
        except AttributeError:
            # 策略试图读真值 —— 直接让整个 episode 失败，这是期望行为（反例 1）
            raise
        if not isinstance(act, Action):
            raise TypeError(f"策略 {getattr(policy, 'name', '?')} 返回了 {type(act)}，应为 Action")
        # c_probe / c_data > 1 时，一次主动动作真的占多个物理时隙
        n_sub = max(int(cfg.protocol.c_probe) if act.kind == "PROBE" else
                    int(cfg.protocol.c_data) if act.kind == "SEND" else 1, 1)
        try:
            w.step(act, n_substeps=n_sub)
        except BudgetExhausted:
            # 预算已尽：协议退化为 HOLD（不是程序错误）。
            # ★ 这是**静默降级**，必须计数 —— 否则"策略被预算截断后完全失控"
            #   会伪装成"该策略本来就这么差"（R0 面板里 always_probe 的 1224.8 正是如此）。
            w.budget_blocked += 1
            w.step(Action("HOLD"), n_substeps=1)

    elapsed = time.perf_counter() - t_start
    recs = w.records
    n_slots = len(recs)
    cost_est = float(np.mean([r.cost_est for r in recs])) if recs else float("nan")
    cost_true = float(np.mean([r.cost_true for r in recs])) if recs else float("nan")

    # 逐用户成本：由逐槽、逐用户成本累加后取均值（不是把范数当成本）
    est_by_user = [float(np.mean([r.cost_est_by_user[i] for r in recs])) for i in range(w.n_users)] \
        if recs else []
    true_by_user = [float(np.mean([r.cost_true_by_user[i] for r in recs])) for i in range(w.n_users)] \
        if recs else []

    aoi_vals = [x["task_aoi"] for x in w.view().telemetry]
    csi_aois = [x["csi_aoi"] for x in w.view().telemetry]

    res = EpisodeResult(
        split=split, episode=int(episode), policy=getattr(policy, "name", type(policy).__name__),
        budget_total=int(budget_total), rho=float(rho), T=T,
        cost_est_per_step=cost_est, cost_true_per_step=cost_true,
        cost_est_by_user=est_by_user, cost_true_by_user=true_by_user,
        n_slots=n_slots,
        n_probe=w.n_probe, n_attempted=w.budget_attempted // max(1, int(cfg.protocol.c_data)),
        n_delivered=w.n_ack, n_ack=w.n_ack, n_nack=w.n_nack, n_idle=w.n_idle,
        n_forced_hold=w.budget_blocked,
        probe_resource=int(cfg.protocol.c_probe) * w.n_probe,
        attempted_resource=w.budget_attempted,
        delivered_resource=int(cfg.protocol.c_data) * w.n_ack,
        budget_used=w.budget_used, budget_attempted=w.budget_attempted,
        budget_wasted=w.budget_wasted, budget_left=w.budget_total - w.budget_used,
        mean_task_aoi=float(np.mean([a for a in aoi_vals if a >= 0])) if any(a >= 0 for a in aoi_vals) else -1.0,
        mean_csi_aoi=float(np.mean([a for a in csi_aois if a >= 0])) if any(a >= 0 for a in csi_aois) else -1.0,
        final_est_err=float(recs[-1].est_norm) if recs else float("nan"),
        stream_ids=sids.as_record(),
        seed_plan={"root_seed": int(cfg.root_seed), "split": split, "episode": int(episode)},
        elapsed_s=round(elapsed, 4),
        config_fingerprint=cfg.fingerprint(),
        policy_trace_hash=_trace_hash(recs),
    )
    return res


def _truth_series(which: str | None, step_i: int, n_users: int) -> dict | None:
    """反例 1 用的"替换真值序列"：与任何子流**完全无关**的确定性序列。

    ★ 只替换真值本身，一个随机数都不额外消耗。
    """
    if which is None:
        return None
    vals = np.sin(1e6 + 7919.0 * step_i + np.arange(n_users))
    if which == "x":
        return {"x": 50.0 * vals}       # 量级差 100 倍以上
    if which == "h":
        return {"h": 5.0 * vals}        # log h ±5 ⇒ 成功率几乎 0 或 1
    raise ValueError(f"未知扰动目标 {which!r}")


def check_identities(res: EpisodeResult, cfg: CWConfig) -> dict:
    """对账一个 episode 的全部恒等关系。返回逐项 bool 与总判。"""
    c_p = int(cfg.protocol.c_probe)
    c_d = int(cfg.protocol.c_data)
    items = {
        # 时隙守恒：每个时隙恰好被一个类别占用（探测/尝试/空闲）
        "slots_eq_probe_plus_attempt_plus_idle":
            res.n_slots == res.n_probe + res.n_attempted + res.n_idle,
        # 资源只花在主动动作上
        "budget_used_eq_probe_plus_attempt_rsc":
            res.budget_used == c_p * res.n_probe + c_d * res.n_attempted,
        "budget_within_total": res.budget_used <= res.budget_total,
        "budget_left_nonneg": res.budget_left >= 0,
        # ACK 记账
        "ack_plus_nack_eq_attempted": res.n_ack + res.n_nack == res.n_attempted,
        "delivered_eq_ack": res.n_delivered == res.n_ack,
        "delivered_le_attempted": res.n_delivered <= res.n_attempted,
        # 失败也扣钱
        "wasted_eq_data_cost_times_failed":
            res.budget_wasted == c_d * (res.n_attempted - res.n_delivered),
        "attempted_resource_eq_cd_times_attempted":
            res.attempted_resource == c_d * res.n_attempted,
        # 探测与数据不能同槽（N=2 时动作空间本来就是互斥的，这里查记账层面）
        "probe_and_data_mutually_exclusive_in_slots":
            res.n_probe + res.n_attempted <= res.n_slots,
        # ★ 强制 HOLD 只在预算耗尽后出现；它算进 idle，因此时隙守恒仍成立
        "forced_hold_within_idle": res.n_forced_hold <= res.n_idle,
        "forced_hold_only_when_budget_nearly_spent":
            res.n_forced_hold == 0 or res.budget_left < 1,
    }
    return {"items": items, "pass": all(items.values())}
