# -*- coding: utf-8 -*-
"""CW-POMDP-v1 的 episode 运行器：一个策略在一条外生轨迹上跑完 T 个时隙。

★★ 配对的设计（v4 §7「同一轨迹配对」）
    外生子流 `plant` / `channel` / `packet` / `meas` / `csi` 由 (root_seed, split, episode)
    决定，**与策略无关** ⇒ 同一个 (split, episode) 下，不同策略面对同一串过程噪声与同一条
    信道序列。策略的动作只改变**消费哪些**，不改变**未来有什么**。

    ★★ 但"同一串"到什么粒度，取决于 `config.ProtocolConfig.event_pairing`：
      · `"slot"`（默认，2026-10-22 18:xx 起）：`packet`/`meas`/`csi` 按 `(t, user)`
        预生成 ⇒ 两策略若在同一 (t,i) 上都发了，成败与量测噪声**逐位相同**（真配对）。
      · `"attempt"`（18:xx 前的全部 v1.1 证据）：按动作次数取值 ⇒ 发 128 次与发 64 次
        的臂读的是流的**不同位置**。分布相同，但**不能**声称"丢包序列逐时隙相同"。
      ★ 两种口径的数字**不得混表**；`meta.json` 的 `event_pairing` 字段是唯一凭据。

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
from dataclasses import dataclass, field, asdict

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
    #: ★ 预测器口径（"nominal" / "exact"）。进 EpisodeResult ⇒ 进证据，
    #:   不同口径的结果不会被误当成同一方法的两次重复。
    predictor: str = "nominal"
    # ---- ★ R1.8 逐用户机械量（**纯记录**；不改任何已有字段的语义）----
    #: 用户 i 的时间平均任务陈旧度（只统计首次送达之后的时隙）；全期未送达 ⇒ None
    aoi_tavg_by_user: list = field(default_factory=list)
    #: 从未送达过（`stamp<0`）的时隙数
    n_unserved_slots_by_user: list = field(default_factory=list)
    #: 末端快照的陈旧度（`-1` = 未送达）—— 旧 `mean_task_aoi` 的两个分量
    aoi_final_by_user: list = field(default_factory=list)
    #: 期内最大陈旧度（饥饿守卫）
    aoi_max_by_user: list = field(default_factory=list)
    #: 逐用户送达 / 尝试计数（由 SlotRecord 计；`n_delivered` 是 ACK 总数）
    n_delivered_by_user: list = field(default_factory=list)
    n_attempted_by_user: list = field(default_factory=list)
    #: 全期从未送达的用户个数（>0 时旧 `mean_task_aoi` 会把它从分母里删掉）
    n_never_served: int = 0
    # ---- ★ R1.11 逐用户**归属**计数（纯记录）----
    #: ★★ 存在的理由：`Σ_i 计数 = 总数` 对"记到了谁头上"恒等不可见。
    #:   R1.10 的缺陷里 `Σ NACK = 77` 全对，而 u0 的 71 次真实失败一次也没记到 u0。
    #:   ⇒ 必须把**逐用户**这一维变成可观测的量，否则归属缺陷只能靠插桩发现。
    #:   ★ 与 `n_attempted_by_user - n_delivered_by_user` 是**两处独立代码**，
    #:     两者对账（闸 4l）只封"计数归属"，实参正确性由闸 4j/4k 的行为观察钉死。
    n_ack_by_user: list = field(default_factory=list)
    n_nack_by_user: list = field(default_factory=list)
    # ---- ★ R1.14 标定诊断（**纯记录**）----
    #: 逐槽、逐用户的真实 `log h`（时点 = 槽开始，与 `telemetry` 的 `h_mean/h_var` 同刻）。
    #: 只用于离线算 `R_var = mean((H−m)²)/mean(v)`；**不进任何决策路径**。
    log_h_hist: list = field(default_factory=list)


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
    predictor: str = "nominal",
) -> EpisodeResult:
    """跑一个 episode。

    `perturb_truth`：`"x"` / `"h"` 在**运行时**把真值换成一个完全不同的序列，
    用于反例 1 的"只改真值 ⇒ 动作应逐位不变"。这只改真值，不改任何子流。
    `predictor`：**只剩 `"nominal"`**。2026-10-22 18:xx 撤回 `"exact"`：
    本环境一槽只推进一次物理，名义滚动 `A x̂ + B u` / `A²P + σ_w²` 本身就是**精确**的
    一步预测；旧 `"exact"` 分支把控制量与过程噪声各算两遍（零噪声下偏差 0.5997），
    是缺陷不是改进 ⇒ 传 `"exact"` 直接报错，不给"打开就更精确"的错觉。
    """
    if predictor != "nominal":
        raise ValueError(
            f"predictor={predictor!r} 不被支持：`exact` 分支已于 2026-10-22 撤回"
            f"（它不是精确预测器，见 experiments/cw_v1/pred_gates.py）。只允许 'nominal'。"
        )
    t_start = time.perf_counter()
    rngs, sids = make_episode_rngs(cfg.root_seed, split, episode)
    w = CWWorld(cfg, rngs, sids, allow_truth_keys=allow_truth_keys,
                correct_receiver=correct_receiver, rho=rho,
                exact_predictor=False)
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

    # ---- ★ R1.8 逐用户机械量（纯记录；不参与任何决策）----
    #  旧 `mean_task_aoi` 的两个机械缺陷（预注册 §2）：
    #    D1 它只是**末端单帧**的快照（两个用户 ⇒ 两个整数）；
    #    D2 它用 `if a >= 0` **剔除**从未送达的用户 ⇒ 用户越被饿死越不进指标。
    hist = w.aoi_history
    n_u = w.n_users
    aoi_tavg: list = []
    n_unserved: list = []
    aoi_final: list = []
    aoi_max: list = []
    for i in range(n_u):
        col = [h[i] for h in hist]
        served = [a for a in col if a >= 0]
        n_unserved.append(int(len(col) - len(served)))
        aoi_tavg.append(float(np.mean(served)) if served else None)
        aoi_max.append(int(max(served)) if served else None)
        aoi_final.append(int(col[-1]) if col else -1)
    deliv_u = [0] * n_u
    att_u = [0] * n_u
    for r in recs:
        if r.action == "SEND" and r.user is not None:
            att_u[r.user] += 1
            if r.data_delivered:
                deliv_u[r.user] += 1
    n_never = int(sum(1 for i in range(n_u) if aoi_tavg[i] is None))

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
        predictor=predictor,
        aoi_tavg_by_user=aoi_tavg,
        n_unserved_slots_by_user=n_unserved,
        aoi_final_by_user=aoi_final,
        aoi_max_by_user=aoi_max,
        n_delivered_by_user=deliv_u,
        n_attempted_by_user=att_u,
        n_never_served=n_never,
        n_ack_by_user=[int(x) for x in w.n_ack_by_user],
        n_nack_by_user=[int(x) for x in w.n_nack_by_user],
        log_h_hist=[[float(y) for y in row] for row in w.log_h_history],
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
