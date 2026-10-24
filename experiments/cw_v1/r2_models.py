# -*- coding: utf-8 -*-
"""CW-R2 的**参考模型实现**：只吃 `SchedulerView`，不碰世界。

★ 闸 `A3` 会扫本文件的 AST：这里**不许**出现 `TRUTH_FIELDS` / `_z_*` 等名字，
  也不许 `view._pub` 直取快照容器；import 只许 `.world: SchedulerView` 与 `.model_api`。
★ 唯一例外是 `TruthDiagnostic`（诊断臂，整类被 `scan_source` 跳过）——
  它由 `deploy()` 的 `PermissionError` 拦住，闸 `A5` 钉住，**不得**进入任何结果表。

三个参考件（R2.1 的对照基础，本轮**不跑**）
--------------------------------------
`BeliefReadout`   R2 的**强基线**：直接读 `telemetry` 的 `x_hat/P_hat`。
                  它不是"学习模型"——是环境自己的名义闭环信念（R1.4 已证一步预测**精确**：
                  零噪声逐位 0、独立 Kalman 复算 160 值逐位 0）。任何"学习模型"必须先
                  在**这一步预测**上追平它，才有资格谈任务代价。
`PriorOnly`       R2 的**弱基线**：无视全部历史，只报公开先验。用来钉"信息到底有没有被用上"。
`TruthDiagnostic` 真值辅助诊断臂。**不是上界**（R1.1 已撤回该名称与它承载的推论：
                  同一格给它更多真信息反而更差 4.26× ⇒ 那是决策规则差，不是信息无用）。
"""

from __future__ import annotations

import numpy as np

from .model_api import ModelOutput, require_view
from .world import SchedulerView


class BeliefReadout:
    """强基线：把接收端后验原样搬出来（零参数、零训练）。"""

    name = "belief_readout"
    privileged = False

    def predict(self, view: SchedulerView) -> ModelOutput:
        v = require_view(view)
        x = np.array([float(t["x_hat"]) for t in v.telemetry], dtype=float)
        P = np.array([float(t["P_hat"]) for t in v.telemetry], dtype=float)
        return ModelOutput(x_hat=x, P=P, label=self.name).validate(v.n_users)

    def describe(self) -> dict:
        return {"name": self.name, "n_params": 0, "learned": False,
                "uses": ["telemetry.x_hat", "telemetry.P_hat"]}


class PriorOnly:
    """弱基线：只报公开先验（不看任何观测/ACK）。"""

    name = "prior_only"
    privileged = False

    def predict(self, view: SchedulerView) -> ModelOutput:
        v = require_view(view)
        pr = v.prior
        x = np.full(int(v.n_users), float(pr["x_mean"]), dtype=float)
        P = np.full(int(v.n_users), float(pr["x_var"]), dtype=float)
        return ModelOutput(x_hat=x, P=P, label=self.name).validate(v.n_users)

    def describe(self) -> dict:
        return {"name": self.name, "n_params": 0, "learned": False,
                "uses": ["prior.x_mean", "prior.x_var"]}


class TruthDiagnostic:
    """诊断臂：真值由**外部显式传入**（不来自视图，视图里也没有）。

    ★ `privileged=True` ⇒ `deploy()` 必抛 `PermissionError`；只能用 `deploy_diagnostic()`。
    ★ 名字里的 "Diagnostic" 是**实质约束**：它不评估"测一下值不值"、不评估剩余预算的
      续接价值 ⇒ 它**不是**最优策略、**不是**上界（见 R1.1 交付 §1）。
    """

    name = "truth_diagnostic"
    privileged = True

    def __init__(self, *, x: np.ndarray, log_h: np.ndarray) -> None:
        self._x = np.asarray(x, dtype=float).reshape(-1)
        self._log_h = np.asarray(log_h, dtype=float).reshape(-1)

    def predict(self, view: SchedulerView) -> ModelOutput:
        v = require_view(view)   # 仍然要求视图合法（防止拿它当"绕过守卫"的先例）
        n = int(v.n_users)
        if self._x.shape != (n,):
            raise ValueError(f"真值长度 {self._x.shape} 与 n_users={n} 不符")
        return ModelOutput(x_hat=self._x.copy(), P=np.full(n, 1e-12),
                           label=self.name, privileged=True).validate(n)

    def describe(self) -> dict:
        return {"name": self.name, "n_params": 0, "learned": False,
                "privileged": True,
                "uses": ["外部传入的真状态（不来自视图）"],
                "not_an_upper_bound": "R1.1 撤回；单步贪心 + 无前瞻"}
