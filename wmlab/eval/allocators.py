# -*- coding: utf-8 -*-
"""四个分配臂（T11 / Gate 2 的对照臂）。

为什么单独一个文件
------------------
臂的**规则**是研究口径的一部分（预注册于 `04-实验与代码/2026-10-19-T12-口径修订与四臂预注册-v0.1.md` §4），
必须与 runner 分开：runner 管记账，臂只管"派给谁"。这样换臂不动记账、改记账不动臂。

四条共同纪律（写死，别在子类里破）
----------------------------------
1. **只在 `alive` 的用户里选**（2026-10-19 验收 §3-B）：派给死人的份额会空转，
   若某些臂过滤、某些不过滤，则"各臂实际可用预算"不等 ⇒ 比值不是算法收益。
   所有臂都走 `_pick_alive()`，它只有一个出口。
2. **不读 `StepRecord`、不读 env**：只读 `view`（I4）。臂拿不到真值，除非 `diagnostic=True` 且走
   `require_privileged`。
3. **同分取 idx 小**：否则同分时的选择依赖 `argmax` 实现细节，跨 numpy 版本可能不同。
4. **不消耗 `rng`**：四个臂都是确定性的（这是故意的 —— 有随机性的臂要额外控制随机源，
   本轮不加，避免"臂间差 = 随机数差"）。

`oracle` 的诚实标注
-------------------
`true_error()` 用 `privileged["true_obs"]` 算**当前**真值代价，选最大者。它 **没有前瞻**：
服务一个用户会刷新它的控制估计，从而影响**下一步**代价，而本臂不看这一步。
⇒ **它不是上界**，只回答"把 age/csi 换成真值任务误差，分配会变好吗"。
`scales` 参数是逐任务尺度（`s_i`），来源是 B=0 零预算试点（与任何臂的收益无关）；
比较的是 `cost_i / s_i`，因为跨任务量纲不同（m vs 0/1）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from .multiuser import Allocation, MissingStateError, View, require_privileged

__all__ = ["RoundRobin", "MaxAge", "MaxChannel", "TrueError", "ALIVE_ONLY_NOTE"]


ALIVE_ONLY_NOTE = (
    "四个臂都只在 alive 用户里选（验收 §3-B）；"
    "否则死人份额空转会让各臂实际可用预算不等。"
)


def _pick_alive(view: View, key: Callable[[int], float]) -> int:
    """在存活用户里选 `key` 最大者；同分取 idx 小；没人活着返回 -1。

    ★ 唯一的"选人"出口：想加新臂就写 `key`，不要自己写 argmax —— 那样很容易忘掉 alive 过滤，
      而忘掉它的后果（预算空转）在单臂单跑时**看不出来**，只在跨臂比较时变成假收益。
    """
    best, best_key = -1, -np.inf
    for s in view.slots:
        if not s.alive:
            continue
        k = float(key(s.idx))
        if k > best_key:                 # 严格大于 ⇒ 同分保留更小的 idx
            best, best_key = s.idx, k
    return best


def _alloc(n_users: int, idx: int, reason: str) -> Allocation:
    served = np.zeros(n_users, dtype=bool)
    if idx >= 0:
        served[idx] = True
    return Allocation(served=served, reason=reason)


def _exhausted(view: View) -> Allocation | None:
    """预算用完 ⇒ 必须停发（I1）。**四个臂都要过这一关**，否则 runner 会抛 I1 违约。

    ★ 这不是"臂的智能"，是记账契约：`budget_left` 是 `View` 里唯一与预算有关的字段，
      臂不读它就等于假设预算无限（单臂单跑看不出来，两档预算一比就炸）。
    """
    if int(view.budget_left) <= 0:
        return _alloc(len(view.slots), -1, "budget-exhausted")
    return None


@dataclass
class RoundRobin:
    """轮流服务：在 `alive` 用户里按 idx 轮转（不是按 t 取模 —— 那样死人会吃掉轮次）。"""
    _ptr: int = field(default=0, init=False, repr=False)

    def __call__(self, view: View, rng: np.random.Generator) -> Allocation:
        n = len(view.slots)
        stop = _exhausted(view)
        if stop is not None:
            return stop
        alive = [s.idx for s in view.slots if s.alive]
        if not alive:
            return _alloc(n, -1, "rr:no-alive")
        # 从 _ptr 起找下一个存活用户（环形）
        start = self._ptr % n
        for k in range(n):
            i = (start + k) % n
            if view.slots[i].alive:
                self._ptr = (i + 1) % n
                return _alloc(n, i, "rr:alive-rotation")
        return _alloc(n, -1, "rr:unreachable")


@dataclass
class MaxAge:
    """最旧优先：`age` 最大者。age 只随**成功送达**归零（与 `UserSlot.age` 定义一致）。"""

    def __call__(self, view: View, rng: np.random.Generator) -> Allocation:
        stop = _exhausted(view)
        if stop is not None:
            return stop
        i = _pick_alive(view, lambda j: view.slots[j].age)
        return _alloc(len(view.slots), i, "max-age")


@dataclass
class MaxChannel:
    """信道最好优先：`csi` 最大者。

    ★ 用**常数** csi 时本臂退化为"恒选某一个人"（不是信道择优）；要它真的是择优，
      必须喂时变 csi 轨迹（`run_multiuser(csi=3维数组)`）。
    """

    def __call__(self, view: View, rng: np.random.Generator) -> Allocation:
        stop = _exhausted(view)
        if stop is not None:
            return stop
        i = _pick_alive(view, lambda j: view.slots[j].csi)
        return _alloc(len(view.slots), i, "max-channel")


@dataclass
class TrueError:
    """真值任务误差最大者（**诊断臂**，myopic，不是上界）。见模块 docstring 的标注。

    Args:
        cost_fn: 真值代价函数。可调用 ⇒ 广播给所有用户；序列 ⇒ **逐用户**
            （异质任务必须逐用户：UAV 的 obs 是 6 维距离量、CartPole 是 4 维 0/1，
             拿 UAV 的 cost 去算 CartPole 的 obs 会直接下标越界 —— 或者更糟，算出一个假数）。
        scales: 逐用户尺度 `s_i`（`None` ⇒ 全 1.0，即直接比原单位，只在同族任务时合法）。
        name: 写进 `Allocation.reason`，便于逐 step 归因。
    """

    cost_fn: Callable[[np.ndarray], float] | Sequence[Callable[[np.ndarray], float]]
    scales: Sequence[float] | None = None
    name: str = "oracle"

    def __post_init__(self) -> None:
        if callable(self.cost_fn):
            self._cfs = None                      # 延迟到第一次调用才知道 N
        else:
            self._cfs = list(self.cost_fn)
            if not all(callable(f) for f in self._cfs):
                raise TypeError("cost_fn 序列里每一项都必须是可调用对象")
        if self.scales is not None:
            self.scales = [float(s) for s in self.scales]
            if any((not np.isfinite(s)) or s <= 0.0 for s in self.scales):
                raise ValueError(f"scales 必须是有限正数，收到 {self.scales}")

    def __call__(self, view: View, rng: np.random.Generator) -> Allocation:
        stop = _exhausted(view)
        if stop is not None:
            return stop
        obs = require_privileged(view, "true_obs")     # I3：拿不到就抛，不静默降级
        if len(obs) != len(view.slots):
            raise MissingStateError(f"true_obs 长度 {len(obs)} != N {len(view.slots)}")
        n = len(view.slots)
        cfs = self._cfs
        if cfs is None:
            cfs = [self.cost_fn] * n
        elif len(cfs) != n:
            raise MissingStateError(f"cost_fn 序列长度 {len(cfs)} != N {n}")
        if self.scales is not None and len(self.scales) != n:
            raise MissingStateError(f"scales 长度 {len(self.scales)} != N {n}")
        scaled = {}
        for s in view.slots:
            c = float(cfs[s.idx](np.asarray(obs[s.idx], dtype=np.float32)))
            if not np.isfinite(c):
                raise MissingStateError(f"user {s.idx} 的真值代价不是有限值：{c}")
            scaled[s.idx] = c / (1.0 if self.scales is None else self.scales[s.idx])
        i = _pick_alive(view, lambda j: scaled[j])
        return _alloc(len(view.slots), i, f"{self.name}:true-error-greedy")
