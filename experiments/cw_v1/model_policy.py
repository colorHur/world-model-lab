# -*- coding: utf-8 -*-
"""CW-R2.1：把**模型输出**接到**规划器输入**的唯一实现。

为什么需要这一层
----------------
`PlanRef` 只吃 `SchedulerView`，并且是从 `telemetry` 的 `x_hat/P_hat` 读任务状态信念
（`plan_ref.py:585`、`:638`）。所以"换模型"这件事在实现上只有一条合法路径：
**用模型的输出重建一个合法视图**，再把该视图交给同一个规划器。

三条硬约束
----------
1. 模型只看**原始**视图；它拿不到任何真值/未来（`model_api.require_view` + AST 闸 `B1`）。
2. 重建后的视图必须**仍然通过** `require_view`（类型恒等 + 键集相等）——
   否则这一层就成了绕过守卫的口子。
3. 重建**只**覆盖 `telemetry.x_hat` 与 `telemetry.P_hat`，其余字段逐位原样；
   且**不回写**原视图（世界自己的后验不被模型污染）。

★ 本文件是部署路径：`B1` 用 `model_api.scan_source` 扫它。
  import 白名单 = `.world: SchedulerView` + `.model_api` + numpy/typing/hashlib。
  **不许** import 任何策略模块（`plan_ref` / `plan_dp` / `runner`）——
  规划器由调用方**注入**，这样"模型层"与"策略层"在 import 图上就是分开的。
"""

from __future__ import annotations

import numpy as np

from .model_api import TELEMETRY_KEYS, require_view
from .world import SchedulerView

#: 重绑定时**允许**被覆盖的两个 telemetry 键（其余一律逐位原样）。
REBIND_KEYS: tuple[str, ...] = ("x_hat", "P_hat")


def rebind_view(view: SchedulerView, x_hat, P_hat) -> SchedulerView:
    """用 `(x_hat, P_hat)` 覆盖 `telemetry`，返回**新的**合法视图。

    ★ 用**公开属性**重建（`t/n_users/budget_left/budget_total/prior/telemetry/history`），
      不读 `_pub` —— 直取快照容器的写法在本项目里是被 AST 闸禁的（`FORBIDDEN_ATTRS`）。
    ★ 校验放在前面：形状/有限性/正性不合格就抛，不留给下游（`ModelOutput.validate` 同口径）。
    """
    v = require_view(view)
    n = int(v.n_users)
    x = np.asarray(x_hat, dtype=float).reshape(-1)
    P = np.asarray(P_hat, dtype=float).reshape(-1)
    if x.shape != (n,):
        raise ValueError(f"x_hat 形状应为 ({n},)，收到 {x.shape}")
    if P.shape != (n,):
        raise ValueError(f"P_hat 形状应为 ({n},)，收到 {P.shape}")
    if not np.all(np.isfinite(x)):
        raise ValueError("x_hat 含 NaN/Inf")
    if not np.all(np.isfinite(P)) or np.any(P <= 0.0):
        raise ValueError("P_hat 必须是正的有限方差")

    tel = []
    for i, row in enumerate(v.telemetry):
        new = dict(row)
        if tuple(sorted(new)) != tuple(sorted(TELEMETRY_KEYS)):
            raise ValueError(f"telemetry[{i}] 键集与声明不符：{sorted(new)}")
        new["x_hat"] = float(x[i])
        new["P_hat"] = float(P[i])
        tel.append(new)

    return SchedulerView({
        "t": int(v.t),
        "n_users": n,
        "budget_left": int(v.budget_left),
        "budget_total": int(v.budget_total),
        "prior": dict(v.prior),
        "telemetry": tel,
        "history": tuple(v.history),
    })


class ModelPolicy:
    """`model.predict(view)` → `rebind_view` → `planner(v2)`。

    ★ `planner_factory` 是**无参可调用**，每次 episode 由调用方新建一个
      （`PlanRef` 持有 `n_probe` 等诊断计数与惰性 DP，不能跨 episode 复用）。
    ★ `deploy()` 在这里把模型的权限钉住：`privileged=True` 直接 `PermissionError`。
    ★ `readout_signature`：把模型每步的 `(x̂,P̂)` **累积和 + 首末值**写成可读串 ⇒
      "两个模型喂给规划器的信念串确实不同" 可被机器/人直接核对，不用存逐槽大表。
      ★ 为什么不用 `hashlib`：`model_api.ALLOWED_ABS_IMPORTS` 的白名单只列了
        numpy/typing/dataclasses/math/__future__/json，闸 `B1` 首跑就拦下了 `hashlib`
        （**闸是对的**）。按"要兼容就改调用方"的纪律改这里，**不**放松白名单。
    """

    def __init__(self, model, planner_factory, *, label: str | None = None) -> None:
        from .model_api import deploy

        self.model = deploy(model)
        self.model_name = str(getattr(model, "name", type(model).__name__))
        self.name = f"model_policy[{self.model_name}]" if label is None else str(label)
        self._factory = planner_factory
        self._planner = None
        self.n_model_calls = 0
        self._acc_x = None
        self._acc_P = None
        self._first = None
        self._last = None

    # ---- 惰性规划器 ----
    @property
    def planner(self):
        if self._planner is None:
            self._planner = self._factory()
        return self._planner

    def _accumulate(self, x, P) -> None:
        x = np.asarray(x, dtype=float).reshape(-1)
        P = np.asarray(P, dtype=float).reshape(-1)
        if self._acc_x is None:
            self._acc_x = np.zeros_like(x)
            self._acc_P = np.zeros_like(P)
            self._first = (x.tolist(), P.tolist())
        self._acc_x = self._acc_x + x
        self._acc_P = self._acc_P + P
        self._last = (x.tolist(), P.tolist())

    def readout_signature(self) -> str:
        if self._acc_x is None:
            return ""
        return (f"n={self.n_model_calls};"
                f"sum_x={[round(v, 9) for v in self._acc_x.tolist()]};"
                f"sum_P={[round(v, 9) for v in self._acc_P.tolist()]};"
                f"first={self._first};last={self._last}")

    def __call__(self, view):
        out = self.model.predict(view)                      # 模型只看原始视图
        out.validate(int(view.n_users))
        self.n_model_calls += 1
        self._accumulate(out.x_hat, out.P)
        return self.planner(rebind_view(view, out.x_hat, out.P))

    # ---- 诊断代理（证据用；不影响决策）----
    @property
    def n_probe(self) -> int:
        return int(getattr(self._planner, "n_probe", 0))

    @property
    def n_send(self) -> int:
        return int(getattr(self._planner, "n_send", 0))

    @property
    def n_hold(self) -> int:
        return int(getattr(self._planner, "n_hold", 0))

    def acts(self):
        """规划器若记录了动作序列（`_RecAct`），返回它；否则空表。"""
        p = self._planner
        return list(getattr(p, "acts", []) or [])


def rebind_equivalence(view: SchedulerView, other: SchedulerView,
                       *, tol: float = 0.0) -> dict:
    """逐字段核对两个视图（`B2` 的实现）：返回不一致项清单。"""
    bad: list[str] = []
    for k in ("t", "n_users", "budget_left", "budget_total"):
        if int(getattr(view, k)) != int(getattr(other, k)):
            bad.append(k)
    if dict(view.prior) != dict(other.prior):
        bad.append("prior")
    if tuple(view.history) != tuple(other.history):
        bad.append("history")
    if len(view.telemetry) != len(other.telemetry):
        bad.append("telemetry.len")
        return {"ok": False, "bad": bad}
    for i, (a, b) in enumerate(zip(view.telemetry, other.telemetry)):
        if tuple(sorted(a)) != tuple(sorted(b)):
            bad.append(f"telemetry[{i}].keys")
            continue
        for k in a:
            if float(a[k]) != float(b[k]):
                if tol > 0.0 and abs(float(a[k]) - float(b[k])) <= tol:
                    continue
                bad.append(f"telemetry[{i}].{k}")
    return {"ok": not bad, "bad": bad}
