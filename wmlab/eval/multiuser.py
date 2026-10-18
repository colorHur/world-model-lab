# -*- coding: utf-8 -*-
"""多用户共享预算资源层 —— **接口契约**（T8 / 2026-10-18）。

本模块只定义"谁在什么时候能看见什么、拿什么做决定、预算怎么记"，
**不含求解器、不含 SINR 公式、不含环境步进**（T9 才写）。

为什么必须先把接口钉死（而不是边写边调）
------------------------------------------------------------------
单用户那条链（X24–X45）里，`loss_prob` 是**外生常数**（envs/channel.py:41-105），
所以"通信约束有多紧"是一个**输入**；多用户一上来，"通信约束有多紧"变成**决策的后果**：
    分配 → 每用户实得帧占比 → 等效丢包 p_eff → 年龄分布 → 任务代价
一旦 `p_eff` 是被调出来的（而不是被记出来的），Gate 2 的比值就可以靠
"把预算调松/调紧"任意操纵。因此本接口把三件事写成**硬结构**，不给调用方自由度：

  H1 **预算只认整数帧**（`budget_total`，单位 = 帧），逐 episode 精确递减，不插值不对齐。
  H2 **`p_eff` 必须由实测的送达计数算出**（`delivery_rate`），不许由档位反推。
     判-01 §4 的 `p_eff ≈ 1−1/N` 仅当每次传输都成功（q=1）成立，
     而精确关系是 `p_eff = 1 − q·(实得帧占比)` ⇒ `1−1/N` 是 p_eff 的**乐观下界**。
  H3 **特权信息只有一个入口**（`View.privileged`），且用到它的臂必须
     `diagnostic=True`（对齐 eval/scheduling.py:24-31 的 C3：「特权信息基线，不主张上界」）。

时序（★ 与单用户口径逐位对齐的关键）
------------------------------------------------------------------
    第 t 步：
      1. `view = build_view(t, users, ledger)`      # 只含 t−1 及之前**已送达**的内容
      2. `alloc = allocator(view, rng)`             # 分配决策（★ 不得触碰 env）
      3. 对每个用户 i：`ok_i ~ Bernoulli(success_prob(view, alloc)[i])`
      4. 对 `ok_i` 为真的用户：帧送达 ⇒ `age_i ← 0`；否则 `age_i ← age_i + 1`
      5. 所有存活用户各走一步 env（动作由各用户自己的控制器/策略产生）

  ★ 第 2 步必须在**任何** env.step() 之前发生。单用户链里 `t` 从 1 起、
    t=0 永不出现（scheduling.py:349-351 记过这个坑），本接口沿用：`t ∈ {1,…,L}`。

不变式（每条都有一个能打死它的检查，见 T9 清单）
------------------------------------------------------------------
  I1 预算守恒：`served.sum() ≤ 1` 每步；逐 episode `Σ_t served.sum() == budget_total`（整数字面相等）。
  I2 不可见真值：`UserSlot.last_obs` 是**最后一次成功送达**的那份观测，
     不是 env 当前观测；`View.privileged is None` 时分配器拿不到任何真值。
  I3 缺字段报错：分配器若读到 `None` 形态的必需字段 ⇒ 抛 `MissingStateError`，
     **不许**当 0 / False（理由同 scheduling.py:55-92）。
  I4 决策不扰动动力学：`allocator(view, rng)` 只能消耗**自己的** rng，
     不得读写任何 env 的内部 rng（否则"换臂"会改轨迹）。
  I5 一帧不可拆：不存在"半个用户"；`served` 是 bool 向量。
  I6 终止用户不转嫁预算：`alive=False` 的用户不再被服务，其份额**空转**（不转给存活用户）。
     ★ 这条会让 oracle 的优势变小（更保守），是刻意选的保守方向；
     它同时是一个待拍板项（见 T8 文档 §6-Q2）。

状态（2026-10-18）：**接口已钉死，实现为零**。本文件所有可执行体都 `raise NotImplementedError`，
唯一通过的自检是「能 import」（T8 文档 §7 第 3 步）。T8 文档 §3.1 表里 T8 只能记「部分」：
Q1–Q4 未拍板，其中 **Q4（N 个用户的任务代价怎么合成一个比）未定前 T9 不得开工**。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

__all__ = ["UserSlot", "View", "Allocation", "Allocator", "StepRecord", "delivery_rate"]


@dataclass(frozen=True)
class UserSlot:
    """第 i 个用户在第 t 步**分配器可见**的全部信息。

    这是整个接口的核心：**这张表就是"信息约束"本身**。
    往里加字段 = 放宽信息约束（等于给所有臂更多特权），必须走变更记录。

    Attributes:
        idx: 用户序号（0-based）。
        alive: 本步是否还能被服务。False ⇒ 已 terminated（CartPole 掉杆 / UAV 跟丢）。
            ★ 由 runner 写，不由分配器改。
        age: 距上次**成功送达**的步数（0 = 刚收到）。递推口径与单用户完全一致：
            `age ← 0 if delivered else age + 1`（scheduling.py:348-356）。
        last_obs: 最后一次成功送达的观测，形状 `(obs_dim,)`。
            ★ **不是** env 当前观测。未收到过任何帧时（t=1 首步）为 `reset()` 的观测，
            并置 `n_frames == 0` —— 分配器可以用 `n_frames == 0` 区分"初始值"与"真收到"。
        n_frames: 累计成功送达帧数（整数）。
        n_attempts: 累计被派帧数（整数）。`n_frames <= n_attempts` 恒成立。
        csi: 归一化信道质量 ∈ [0, 1]（1 = 最好）。
            ⚠ 待核（T8 §6-Q1）：CSI 是否"免费可得"会改变所有臂的信息集，
            在 Q1 拍板前，T9 只许把它当**已知量**并在结果里显式声明。
        cost_last: 上一步该用户的任务代价（标量，**逐用户量纲**）。
            ★★ 不同任务族的量纲不同（UAV = 稳态距离 m，CartPole = 0/1 存活），
            **不可跨用户相加**。聚合口径见 T8 §6-Q4，未定之前不许用它算总分。
    """

    idx: int
    alive: bool
    age: int
    last_obs: np.ndarray
    n_frames: int
    n_attempts: int
    csi: float
    cost_last: float


@dataclass(frozen=True)
class View:
    """一次分配决策能看到的**全部**信息。除 `privileged` 外不含任何真值。

    Attributes:
        t: 当前步号，`t ∈ {1, …, L}`（t=0 不出现，对齐 scheduling.py:349-351）。
        slots: 长度 N 的 `UserSlot` 元组，按 `idx` 升序。
        budget_left: 本 episode 剩余帧数（整数，含本步可用的这 1 帧）。
        budget_total: 本 episode 总帧数（整数）。两档预算 = 两个不同的 `budget_total`。
        privileged: **仅诊断臂**为 dict，其余臂必须为 `None`。允许键：
            `"true_obs"`（长度 N，真值观测）、`"true_cost_next"`（N×N 矩阵：
            第 i 行 = 服务用户 i 后**下一步**各用户的真实代价）、
            `"true_age"`。★ 除这三个键外的键一律 raise（防止随手塞新特权量）。
    """

    t: int
    slots: tuple[UserSlot, ...]
    budget_left: int
    budget_total: int
    privileged: dict | None


@dataclass(frozen=True)
class Allocation:
    """一次分配的结果。

    Attributes:
        served: 长度 N 的 bool 数组，**至多一个 True**（I1/I5）。
        reason: 人读的一句话（写进逐 episode 原始表，便于事后归因）。
    """

    served: np.ndarray
    reason: str


Allocator = Callable[[View, np.random.Generator], Allocation]
"""`(View, rng) -> Allocation`。

★ 与单用户 `Schedule = Callable[[int, rng], bool]`（eval/tracking.py:87）**不同型**，
  两者不兼容、不互相包装：单用户的 `True/False` 是"这一步发不发"，
  多用户的 `Allocation` 是"N 个人里派谁"。任何"把 Schedule 套 N 层当多用户用"的做法
  都会失去预算控制（N 个 Schedule 各自以为预算无限），本接口拒绝这种用法。

实现约束（T9 起）：
  · 只读 `view`，不得回写；状态（如轮转指针）放分配器**自己**的属性里。
  · 只能消耗传入的 `rng`（I4）。
  · 读必需字段用 `view.slots[i].xxx`；若该字段可能为 `None`（目前只有 `privileged`），
    必须显式判 `is None` 并抛 `MissingStateError`，不许 `or 0`。
"""


@dataclass(frozen=True)
class StepRecord:
    """**逐步原始表**的一行 —— Gate 2 要的就是这张表，不是均值。

    为什么强制逐 episode 逐 step 落盘（而不是只存均值）：
      · 判-01 §5 已声明 30 集 × 3 种子只够筛选，不够 Gate 2 的统计判断；
      · X45 的教训是**同一份数据换口径就能反转结论**（max 口径 T=8 FAIL / CV 口径 PASS），
        ⇒ 口径必须在**同一次运行**的原始表上离线重算，不许重跑（重跑会变 rng 路径）。

    Attributes:
        episode: episode 序号（0-based）。
        t: 步号。
        served_idx: 本步被服务的用户（-1 = 空转）。
        delivered_idx: 本步**真正送达**的用户（-1 = 没送达）。
            `served_idx != delivered_idx` 时（served 但丢包）两者不同，这是 T9 SINR 层的证据。
        ages: 长度 N 的年龄（**决策前**的值，即分配器看到的那个）。
        costs: 长度 N 的该步任务代价。
        alive: 长度 N 的存活标志。
    """

    episode: int
    t: int
    served_idx: int
    delivered_idx: int
    ages: tuple[int, ...]
    costs: tuple[float, ...]
    alive: tuple[bool, ...]


def delivery_rate(records: Sequence[StepRecord], n_users: int) -> dict:
    """**实测**每用户送达率与等效丢包（H2 的唯一合法来源）。

    Returns:
        {"per_user": [frames_i / alive_steps_i, ...],   # 分母 = 该用户**存活**步数
         "p_eff": [1 − rate_i, ...],
         "budget_used": Σ served_idx >= 0,
         "budget_total": len(records)}

    ★ 为什么分母是"存活步数"而不是总步数：CartPole 掉杆后 `alive=False`（I6 不转嫁），
      拿总步数做分母会把"已经不用服务"算成"丢包"⇒ p_eff 虚高。这是 I6 的直接后果，
      两处必须一致，否则 T12 报出来的 p_eff 和实际预算对不上。

    ★★ 报 p_eff 时必须同时报 `budget_total` 与 N：
      单看 p_eff = 0.5 无法判断是"N=2 且每次成功"还是"N=4 且一半丢失"。
      判-01 §4 的风险正是"档位看着紧、其实松"，唯一防法是这两个数一起报。
    """
    raise NotImplementedError("T9 才实现：本文件只是接口契约（T8），不许在这里落数字")
