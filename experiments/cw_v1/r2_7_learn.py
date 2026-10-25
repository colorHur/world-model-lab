# -*- coding: utf-8 -*-
"""`CW-R2.7`：**公开视图上的残差读出**（本轮唯一的学习对象）。

它在 R2 里承担的角​色
--------------------
`belief_readout`（R2 的强基线）把接收端后验原样搬给规划器。本文件实现的学习模型
**不改方差**，只在均值上加一个小残差：`x̂ = x̂_base + Δ`、`P̂ = P̂_base`。

为什么是**残差**而不是"从头预测"
--------------------------------
① 末层全零初始化 ⇒ `Δ ≡ 0` ⇒ 未训练的模型与强基线**逐位相同**（闸 `G2`）。这让
   "训练到底有没有帮上忙"变成一个**从同一个点出发**的问题，而不是"两个不同模型比大小"。
② §2.2 的 headroom 事实：在本环境里 `x̂_base` 已是该信息集上的 MMSE 估计 ⇒ 学习器
   的期望上界是**打平**。残差参数化把"打平"变成默认解的邻域，而不是一个需要恰好命中的点。

为什么**只学均值**
------------------
方差校准需要另一套标签（误差平方 / NLL），本轮预算里同时学两个自由度会把归因搅浑。
冻结 `P̂ = P̂_base` ⇒ 闭环里唯一变化的自由度是"均值纠正"，单变量对照干净。

★ 本文件是**部署路径**：闸 `G3` 用 `model_api.scan_source` 扫它（禁用名 / `_pub` / 越界 import）。
  所以这里**不许**出现任何真值字段名，也不许 import 训练器（标签只在训练器里）。
★ 本文件**不训练**：没有 `fit()`、没有梯度、没有 loss；`set_weights()` 只做"把参数装进来"，
  优化器在 `r2_7_train.py` 里（训练器不部署、也不被 AST 闸扫）。
"""

from __future__ import annotations

import math

import numpy as np

from .model_api import ModelOutput, require_view
from .world import SchedulerView

#: 逐用户特征（顺序即列顺序，**改顺序等于改模型语义**）
PER_USER_FEATURES: tuple[str, ...] = (
    "x_hat", "P_hat", "stamp", "task_aoi", "h_mean", "h_var", "csi_stamp", "csi_aoi",
)
#: 全局特征（每行广播同一份）
GLOBAL_FEATURES: tuple[str, ...] = (
    "prior.x_mean", "prior.x_var", "prior.h_log_mean", "prior.h_log_var",
    "t", "budget_left", "budget_total",
)
N_FEATURES = len(PER_USER_FEATURES) + len(GLOBAL_FEATURES)

#: 标签尺度用的参考量（公开先验方差 = 1.0 时残差应在这个量级；只在训练器里用于加权）
RESIDUAL_CLIP = 3.0


def layer_shapes(d_in: int = N_FEATURES, hidden: tuple[int, ...] = (32, 32),
                 d_out: int = 1) -> list[tuple[int, int]]:
    dims = [int(d_in), *[int(h) for h in hidden], int(d_out)]
    return [(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]


def count_params(shapes) -> int:
    return int(sum(a * b + b for a, b in shapes))


class ResidualReadout:
    """`(x̂_base, P̂_base)` + 一个零初始化的残差网络 `Δ(features)`。

    - `predict(view)` 只看公开视图；`P̂` 逐位等于 `x̂` 同源的 `telemetry.P_hat`；
    - `dtype=float64`：与世界的后验同精度，闸 `G2` 要求"未训练时逐位相同"；
    - `trained` 由训练器显式置位（`describe()` 里透出），本文件不会自己变 True。
    """

    name = "residual_readout"
    privileged = False

    def __init__(self, *, hidden: tuple[int, ...] = (32, 32), seed: int = 0,
                 d_in: int = N_FEATURES) -> None:
        self.hidden = tuple(int(h) for h in hidden)
        self.d_in = int(d_in)
        self.seed = int(seed)
        self.shapes = layer_shapes(self.d_in, self.hidden, 1)
        self.n_params = count_params(self.shapes)
        rng = np.random.default_rng(self.seed)
        self._w: list[np.ndarray] = []
        self._b: list[np.ndarray] = []
        for k, (a, b) in enumerate(self.shapes):
            if k == len(self.shapes) - 1:
                # ★ 末层恒零 ⇒ Δ ≡ 0 ⇒ 未训练 == 基线（闸 G2 的全部理由）
                self._w.append(np.zeros((a, b), dtype=float))
                self._b.append(np.zeros(b, dtype=float))
            else:
                self._w.append(rng.normal(0.0, 1.0 / math.sqrt(a), size=(a, b)))
                self._b.append(np.zeros(b, dtype=float))
        # 标准化统计：未训练时 = 恒等（mean 0 / std 1）
        self.feat_mean = np.zeros(self.d_in, dtype=float)
        self.feat_std = np.ones(self.d_in, dtype=float)
        self.trained = False
        self.train_meta: dict = {}

    # ---------------- 参数存取（训练器用；本文件不做优化）
    def parameters(self) -> list[np.ndarray]:
        return [*self._w, *self._b]

    def set_parameters(self, params) -> None:
        n = len(self._w) + len(self._b)
        vals = list(params)
        if len(vals) != n:
            raise ValueError(f"参数个数不符：给 {len(vals)}，要 {n}")
        for i, v in enumerate(vals):
            arr = np.asarray(v, dtype=float)
            target = self._w[i] if i < len(self._w) else self._b[i - len(self._w)]
            if arr.shape != target.shape:
                raise ValueError(f"参数 {i} 形状 {arr.shape} ≠ {target.shape}")
            target[...] = arr

    def set_standardizer(self, mean, std) -> None:
        m = np.asarray(mean, dtype=float).reshape(-1)
        s = np.asarray(std, dtype=float).reshape(-1)
        if m.shape != (self.d_in,) or s.shape != (self.d_in,):
            raise ValueError("标准化统计维度与 d_in 不符")
        if np.any(s <= 0):
            raise ValueError("标准化 std 必须为正")
        self.feat_mean, self.feat_std = m, s

    def describe(self) -> dict:
        return {"name": self.name, "n_params": int(self.n_params),
                "hidden": list(self.hidden), "d_in": self.d_in,
                "trained": bool(self.trained), "learned": bool(self.trained),
                "only_mean": True, "variance_policy": "frozen_at_base",
                "uses": list(PER_USER_FEATURES) + list(GLOBAL_FEATURES),
                "train_meta": dict(self.train_meta)}

    # ---------------- 前向
    def features(self, view: SchedulerView) -> np.ndarray:
        """原始（未标准化）特征矩阵，形状 `(n_users, N_FEATURES)`。"""
        v = require_view(view)
        g = [float(v.prior["x_mean"]), float(v.prior["x_var"]),
             float(v.prior["h_log_mean"]), float(v.prior["h_log_var"]),
             float(v.t), float(v.budget_left), float(v.budget_total)]
        rows = []
        for tel in v.telemetry:
            rows.append([float(tel[k]) for k in PER_USER_FEATURES] + g)
        return np.asarray(rows, dtype=float).reshape(int(v.n_users), self.d_in)

    def residual(self, feats: np.ndarray) -> np.ndarray:
        """标准化后的前向：返回 `(n, 1)`（`N_FEATURES` 与 `feats` 列数必须一致）。"""
        f = np.asarray(feats, dtype=float)
        if f.ndim != 2 or f.shape[1] != self.d_in:
            raise ValueError(f"特征形状应为 (n, {self.d_in})，收到 {f.shape}")
        z = (f - self.feat_mean) / self.feat_std
        h = z
        for k, (w, b) in enumerate(zip(self._w, self._b)):
            h = h @ w + b
            if k < len(self._w) - 1:
                h = np.tanh(h)
        return np.clip(h, -RESIDUAL_CLIP, RESIDUAL_CLIP)

    def predict(self, view: SchedulerView) -> ModelOutput:
        v = require_view(view)
        n = int(v.n_users)
        xb = np.asarray([float(t["x_hat"]) for t in v.telemetry], dtype=float)
        Pb = np.asarray([float(t["P_hat"]) for t in v.telemetry], dtype=float)
        d = self.residual(self.features(v)).reshape(-1)
        if d.shape != (n,):
            raise ValueError(f"残差形状应为 ({n},)，收到 {d.shape}")
        return ModelOutput(x_hat=xb + d, P=Pb, label=self.name).validate(n)
