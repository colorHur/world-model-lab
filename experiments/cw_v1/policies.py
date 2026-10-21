# -*- coding: utf-8 -*-
"""CW-POMDP-v1 的策略。R0 只要求能验证通道接线的策略，不追求收益。

★ 这里所有策略都只接收 `SchedulerView`。唯一的例外是 `TruthGreedy` 与 `FuturePeek`，
  它们**故意越界**，用于反例 1 证明"非法接入会被拒绝 / 动作不变"。
  这两者的名字带前缀 `ILLEGAL_`，且不允许出现在任何结果表里。

R0 的策略清单（用途）
---------------------
· `AlwaysProbe`   每槽探测一个用户（轮转）——测"探测只花钱"（反例 6）
· `AlwaysSend`    每槽发一个用户（轮转）——最朴素，用尽预算
· `PeriodicSend`  固定周期发送（paced），不探测 —— B1 基线原型
· `PacedAge`      在每用户 AoI 达阈值时发送（最旧优先）—— B1 的任务 AoI 版本
· `CsiSensitive`  **信息敏感策略**：按信道信念选用户发送 —— 反例 2/6 的阴性对照来源
· `CsiBlind`      与 `CsiSensitive` 结构完全相同，但**把 CSI 内容替换成常数**
                  —— 若两者动作逐位相同，说明 CSI 没有真正进入决策（阴性对照）

★★ 反例 2 的做法：`CsiSensitive` 与 `CsiBlind` 的**唯一差别**是读不读 CSI 值。
   前者随测量变化、后者不随测量变化 ⇒ 证明"通道存在"。这只证明通道，
   **不声称为闭环收益**（首轮卡原文）。
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .world import Action, SchedulerView


Policy = Callable[[SchedulerView], Action]


def _user_send_priority(tel: list[dict], order: str) -> list[int]:
    """返回按优先级排好的用户下标。

    ★ `task_aoi == -1` 表示**从未收到过观测**，它的陈旧度是"无穷大"而不是 -1。
      这里统一用 `_aoi_key()` 转换成可比较的量，避免把 -1 当成"最新"。
    """
    if order == "aoi":
        return sorted(range(len(tel)), key=lambda i: _aoi_key(tel[i]), reverse=True)
    if order == "var":
        return sorted(range(len(tel)), key=lambda i: tel[i]["P_hat"], reverse=True)
    return list(range(len(tel)))


def _aoi_key(row: dict) -> float:
    """用户 i 的陈旧度（越大越该发）。从未观测 ⇒ 无穷大。"""
    return float("inf") if row["task_aoi"] < 0 else float(row["task_aoi"])


def _needs_update(tel: list[dict], theta: int) -> list[int]:
    """返回需要更新的用户下标（AoI ≥ theta，或从未观测）。"""
    return [i for i, x in enumerate(tel) if x["task_aoi"] < 0 or x["task_aoi"] >= theta]


class AlwaysProbe:
    """每槽探测，按用户轮转。用于验证"探测只保留开销"。"""

    name = "always_probe"

    def __init__(self, n_users: int = 2) -> None:
        self.n = n_users

    def __call__(self, v: SchedulerView) -> Action:
        return Action("PROBE", v.t % self.n)


class AlwaysSend:
    name = "always_send"

    def __init__(self, n_users: int = 2) -> None:
        self.n = n_users

    def __call__(self, v: SchedulerView) -> Action:
        return Action("SEND", v.t % self.n)


class PeriodicSend:
    """每 `period` 槽发一次，目标用户按 `order` 选；其余 HOLD。不探测。"""

    def __init__(self, period: int, order: str = "aoi", n_users: int = 2) -> None:
        if period < 1:
            raise ValueError("period 必须 ≥1")
        self.period = int(period)
        self.order = order
        self.n = n_users
        self.name = f"periodic{period}_{order}"

    def __call__(self, v: SchedulerView) -> Action:
        if v.budget_left <= 0 or v.t % self.period != 0:
            return Action("HOLD")
        tel = v.telemetry
        return Action("SEND", _user_send_priority(tel, self.order)[0])


class PacedAge:
    """AoI 阈值触发：任一用户需更新（从未观测，或 AoI ≥ theta）就发最旧用户，否则 HOLD。

    ★ 不加探测。这是 B1（固定周期/AoI）基线的原型。
    """

    def __init__(self, theta: int, n_users: int = 2) -> None:
        self.theta = int(theta)
        self.n = n_users
        self.name = f"paced_age{theta}"

    def __call__(self, v: SchedulerView) -> Action:
        if v.budget_left <= 0:
            return Action("HOLD")
        tel = v.telemetry
        if not _needs_update(tel, self.theta):
            return Action("HOLD")
        return Action("SEND", _user_send_priority(tel, "aoi")[0])


class CsiSensitive:
    """信道信念敏感：按 `h_mean - h_var` 选"最可能还便宜"的用户；**只在需要更新时发送**。

    结构 = PacedAge 的触发条件 + 按 CSI 选目标。
    `use_csi=False` 时退回"最旧优先"（与 PacedAge 同目标选择）——
    ★ 这就是阴性对照：结构相同、**只有 CSI 这一项输入被关掉**。
    """

    def __init__(self, theta: int, use_csi: bool = True, n_users: int = 2) -> None:
        self.theta = int(theta)
        self.use_csi = bool(use_csi)
        self.n = n_users
        self.name = ("csi_sensitive" if use_csi else "csi_blind") + f"_{theta}"

    def __call__(self, v: SchedulerView) -> Action:
        if v.budget_left <= 0:
            return Action("HOLD")
        tel = v.telemetry
        need = _needs_update(tel, self.theta)
        if not need:
            return Action("HOLD")
        if self.use_csi:
            need.sort(key=lambda i: -(tel[i]["h_mean"] - 0.5 * tel[i]["h_var"]))
        else:
            need.sort(key=lambda i: _aoi_key(tel[i]), reverse=True)
        return Action("SEND", need[0])


class ProbeThenSend:
    """信道信念不新鲜就探测，否则发送。用于反例 3 的探测记账检查。"""

    def __init__(self, csi_max_age: int = 4, n_users: int = 2) -> None:
        self.csi_max_age = int(csi_max_age)
        self.n = n_users
        self.name = f"probe_then_send_{csi_max_age}"

    def __call__(self, v: SchedulerView) -> Action:
        if v.budget_left <= 0:
            return Action("HOLD")
        tel = v.telemetry
        # 目标用户 = 最旧；它的 CSI 太旧就先探测它
        i = _user_send_priority(tel, "aoi")[0]
        csi_aoi = tel[i]["csi_aoi"]
        if csi_aoi < 0 or csi_aoi >= self.csi_max_age:
            return Action("PROBE", i)
        return Action("SEND", i)


# ---------------------------------------------------------------- 故意越界（反例 1）
class ILLEGAL_TruthGreedy:
    """★ 只有 `allow_truth_keys=True` 才能构造。

    用途：作为「试图用真值做决策」的样本，用来验证**它确实拿不到真值**。
    实现上它通过 `vars(view)` / 属性探测去找真值入口；`SchedulerView` 用 `__slots__`
    且 `__getattr__` 一律抛错，所以这个策略必定在第一次调用时就失败。
    ★ 它的"失败"就是反例 1 的通过条件 —— 不要给它加任何 fallback。
    """

    name = "ILLEGAL_truth_greedy"

    def __init__(self, allow_truth_keys: bool = False) -> None:
        if not allow_truth_keys:
            raise PermissionError(
                "ILLEGAL_TruthGreedy 读真值，只允许 allow_truth_keys=True（诊断/反例）下构造。"
            )

    def __call__(self, v: SchedulerView) -> Action:
        # 枚举一切可访问入口，找一个能给出真状态的东西
        for probe in ("_x_true", "_w", "true_x", "world", "env", "_world"):
            obj = getattr(v, probe)   # SchedulerView.__getattr__ 抛 AttributeError
            if obj is not None:       # pragma: no cover
                break
        raise AttributeError("未找到任何真值入口（期望行为）")  # pragma: no cover
