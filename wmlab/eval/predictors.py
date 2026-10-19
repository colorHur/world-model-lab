# -*- coding: utf-8 -*-
"""接收端预测器（T22 / P1-2，2026-10-19）。

为什么单独一个模块
------------------
T12–T21 的全部数字都建立在**一件事**上：接收端的估计观测 `est[i]` 只在**送达**时被替换，
其余时刻保持上一次送达值（ZOH，零阶保持）。于是"谁更该被服务"恒等于"谁的观测更旧"。
把 `est` 换成**预测**之后，同一条调度规则面对的输入就变了 —— 这是"预测补偿与调度协同"
这个主问题的第一个直接实验。

信息集边界（**这是本模块的核心，不是实现细节**）
------------------------------------------------
每个预测器只允许用下面这张表里的东西，越界就是"偷真值"：

    ZOH      最后一次送达的观测
    Linear   最近两次送达的观测（差分估速度）
    TrueDyn  `deepcopy` 的 env + **自己实际施加过的动作**（诊断特权，只作上界）

三者**都不读 `View` / `View.privileged`**：预测器拿不到 `privileged`，这是结构性保证
（`predict()` 的签名里根本没有 view），不是"我们保证不读"。`scripts/53` 的 S3 另外用
"`diagnostic=True/False` 下轨迹逐位相同"再验一次。

时序约定（与 `multiuser.run_multiuser` 逐位对齐；差一步就会让"预测 d 步"变成"d−1 步"）
------------------------------------------------------------------------------------
决策时点 `t`、`age = d` ⇒ 最后一次送达发生在第 `t-1-d` 步，手里的观测是 `x_{t-1-d}`，
真值是 `x_{t-1}` ⇒ 需要补 `d` 步；而 `actions_since` 恰好有 `d` 项
（第 `t-d, …, t-1` 步实际施加的动作）。第 `t` 步的动作在预测时**还没产生**，不得在缓冲里。

三条实现纪律
------------
1. **送达即重置**：`observe_delivery()` 里清空动作缓冲的语义由调用方负责（runner 负责清）；
   本模块只负责"快照换成新观测"。
2. **不得触碰主 env**：`TrueDyn` 只对 `deepcopy` 出来的影子前推；影子有**自己的** rng，
   主轨迹与主 rng 一位都不变（`scripts/53` 的 S7 用"影子跑但丢输出"做阴性对照）。
3. **不许静默降级**：缺 env / `actions_since` 太短 / `age < 0` 一律 `ValueError`
   （同 `multiuser.MissingStateError` 的理由：静默降级会跑出一个"看起来正常"的数字）。
"""

from __future__ import annotations

import copy
from typing import Sequence

import numpy as np

__all__ = ["PredictorBase", "ZOH", "Linear", "TrueDyn", "make_predictor"]


class PredictorBase:
    """预测器基类：管每用户的历史（`last_obs` / `prev_obs` / `gap_prev`），子类只写 `_predict`。

    ★ 为什么历史放在预测器里而不是 runner 里：runner 已经维护 `last_obs`，但 `prev_obs` 与
      "两次送达之间隔了几步"（`gap_prev`）只有预测器要用。放在这里可以让 runner 只有
      两行接线，也避免"runner 里多存一份、两边不同步"。

    ★ `gap_prev` 的定义：两次送达相隔的**步数**（不是帧数）。送达在第 τ1、τ2 步
      ⇒ `gap_prev = τ2 − τ1 = age_before(τ2) + 1`（`age_before` 是决策时点的年龄，
      送达发生在该步**之后**）。用它把观测差分换算成"每步速度"，否则间隔一变速度就差一个因子。
    """

    name: str = "base"
    #: 是否需要在送达时拿到 env 引用（只有 `TrueDyn` 为 True；其余为 False ⇒ 结构上不可能偷真值）
    consumes_env: bool = False

    def __init__(self) -> None:
        self._st: dict[int, dict] = {}

    # ------------------------------------------------------------ 生命周期
    def reset_episode(self) -> None:
        """每集开头清空全部用户状态（不跨集泄漏，等价于 I6 的"跨集不继承"）。"""
        self._st = {}

    def observe_delivery(self, i: int, obs, *, age_before: int, env=None) -> None:
        """一帧送达：把历史前移一格，并把新观测记为 `last_obs`。

        Args:
            i: 用户序号。
            obs: 本步送达的观测（= 该步 `env.step()` 返回的 `nxt`）。
            age_before: **决策时点**的年龄（该步尚未结算），用来推 `gap_prev`。
            env: 仅 `consumes_env=True` 的预测器需要（快照源）。
        """
        i = int(i)
        st = self._st.setdefault(i, {})
        prev = st.get("last_obs")
        st["prev_obs"] = None if prev is None else np.asarray(prev, dtype=np.float64).copy()
        st["gap_prev"] = (int(age_before) + 1) if prev is not None else 0
        st["last_obs"] = np.asarray(obs, dtype=np.float32).reshape(-1).copy()
        st["n_deliveries"] = int(st.get("n_deliveries", 0)) + 1
        if self.consumes_env and env is None:
            raise ValueError(f"{self.name}：consumes_env=True 但 observe_delivery 收到 env=None"
                             f"（没有快照源就无法前推；不许静默退化成 ZOH）")
        self._on_delivery(i, st, env)

    # ------------------------------------------------------------ 预测
    def predict(self, i: int, *, age: int, actions_since: Sequence) -> np.ndarray:
        """返回该用户在当前决策时点的**估计观测**（形状同 `last_obs`）。"""
        i = int(i)
        age = int(age)
        if age < 0:
            raise ValueError(f"{self.name}：age 必须 ≥ 0，收到 {age}")
        st = self._st.get(i)
        if st is None or "last_obs" not in st:
            raise ValueError(f"{self.name}：用户 {i} 还没有任何送达观测，无法预测"
                             f"（先调 observe_delivery；不许拿 0 向量顶替）")
        acts = list(actions_since)
        if len(acts) != age:
            # ★ 长度不等 = 时序接错（多/少一步），必须炸；否则会静默地"预测错步数"
            raise ValueError(f"{self.name}：actions_since 长度 {len(acts)} ≠ age {age}"
                             f"（时序接错会让预测差一步，且结果看起来完全正常）")
        out = self._predict(i, st, age=age, actions_since=acts)
        out = np.asarray(out, dtype=np.float32).reshape(-1)
        if not np.all(np.isfinite(out)):
            raise ValueError(f"{self.name}：预测输出含 NaN/inf：{out.tolist()}")
        return out

    # ------------------------------------------------------------ 子类钩子
    def _on_delivery(self, i: int, st: dict, env) -> None:  # pragma: no cover - 默认空
        pass

    def _predict(self, i: int, st: dict, *, age: int, actions_since: list) -> np.ndarray:
        raise NotImplementedError

    # ------------------------------------------------------------ 自检辅助
    def state_refs(self) -> list:
        """摊平全部内部状态值（`scripts/53` 用它验证"状态里没有 env 对象"）。"""
        out: list = []
        for st in self._st.values():
            out.extend(st.values())
        return out


class ZOH(PredictorBase):
    """零阶保持 = **现状**（`83340e1` 的行为）。它的存在意义是回归红线。

    ★ 它必须与 `predictor=None` **逐位相同**；任何不同都说明接线改变了既有轨迹。
    """

    name = "zoh"

    def _predict(self, i: int, st: dict, *, age: int, actions_since: list) -> np.ndarray:
        return st["last_obs"]


class Linear(PredictorBase):
    """局部线性外推：`x̂ = x_last + age · v`，`v = (x_last − x_prev) / gap_prev`。

    ★ 只用**观测**，**不用动作**（这是刻意的，也是本轮明确登记的结构性弱点）：
      稳定控制器会持续反向修正，"按惯性外推"必然过冲。因此若 T23 出现 `Linear` 劣于 `ZOH`，
      只能说明"这个弱预测器不行"，**不能**说明"预测补偿没有空间"（预注册 §5-R2）。

    ★ 观测都是**全状态**（含速度分量）⇒ 差分得到的 `v` 近似"每步位移"，对本系统是对的量纲。
      换到部分观测的场景（只有位置）时，本类需要重新定义 —— 见模块头的"信息集"表，
      它只承诺"最近两次送达观测"，不承诺任何任务知识。
    """

    name = "linear"

    def _predict(self, i: int, st: dict, *, age: int, actions_since: list) -> np.ndarray:
        prev, gap = st.get("prev_obs"), int(st.get("gap_prev", 0))
        if prev is None or gap <= 0 or age == 0:
            return st["last_obs"]          # 无速度估计 ⇒ 退化为 ZOH（预注册 §2）
        v = (np.asarray(st["last_obs"], dtype=np.float64) - prev) / float(gap)
        return st["last_obs"] + float(age) * v


class TrueDyn(PredictorBase):
    """**真实动力学预测（诊断特权）**：从送达时刻的 env 快照出发，用自己实际施加的动作前推。

    ★★ 这不是可部署方法：它需要一份与真实环境同构的仿真器，且与真实 env 状态同步。
      它只回答一个问题——"如果预测**完全准**，调度/补偿能拿到多少"。
      ⇒ 结果里必须标 `diagnostic`，**不得**写成"我们的世界模型"。

    ★★ 影子必须是 `deepcopy`：`copy.deepcopy(env)` 连 `env._rng` 一起复制，因此影子前推
      **不消耗主 env 的任何随机数**、也不改主状态。这条由 `scripts/53` 的 S7 做阴性对照
      （"影子照跑但输出丢弃" ⇒ 主轨迹必须与 `predictor=None` 逐位相同）。

    ★ 终止语义：影子一旦终止，冻结在终止观测（与 `window_cost` 的 absorbed 同向：不再变化），
      并且不再步进（gym 环境在 done 之后步进是未定义行为）。
    """

    name = "truedyn"
    consumes_env = True

    def _on_delivery(self, i: int, st: dict, env) -> None:
        st["shadow"] = copy.deepcopy(env)
        st["shadow_k"] = 0
        st["shadow_obs"] = np.asarray(st["last_obs"], dtype=np.float32).copy()
        st["shadow_dead"] = False

    def _predict(self, i: int, st: dict, *, age: int, actions_since: list) -> np.ndarray:
        if age == 0:
            return st["shadow_obs"]
        if st.get("shadow_dead"):
            return st["shadow_obs"]
        k0 = int(st["shadow_k"])
        if age < k0:
            # age 回退只可能来自"送达但没重置"，属于接线错误 ⇒ 不猜，直接炸
            raise ValueError(f"TrueDyn：age={age} 小于已前推步数 {k0}"
                             f"（送达后必须重建快照；此刻状态已对不上）")
        shadow = st["shadow"]
        for j in range(k0, age):
            res = shadow.step(actions_since[j])
            st["shadow_obs"] = np.asarray(res.obs, dtype=np.float32).reshape(-1).copy()
            if res.terminated or res.truncated:
                st["shadow_dead"] = True
                st["shadow_k"] = age
                return st["shadow_obs"]
        st["shadow_k"] = age
        return st["shadow_obs"]


_PREDICTORS = {"zoh": ZOH, "linear": Linear, "truedyn": TrueDyn}


def make_predictor(name: str | None) -> PredictorBase | None:
    """按名字造预测器；`None` / `"none"` ⇒ `None`（= 现状 ZOH，但**不经过**预测器路径）。

    ★ 为什么不把 `None` 直接映射成 `ZOH`：`predictor=None` 走的是既有代码路径（逐位回归的
      真实对象），`ZOH` 走的是新路径。"两者逐位相同"是 `scripts/53` 的 S1，而不是把两者
      合并成同一个东西 —— 合并了就永远测不出"新路径改了行为"。
    """
    if name is None or str(name).lower() in ("none", ""):
        return None
    key = str(name).lower()
    if key not in _PREDICTORS:
        raise ValueError(f"未知预测器 {name!r}；可选 {sorted(_PREDICTORS)} 或 None")
    return _PREDICTORS[key]()
