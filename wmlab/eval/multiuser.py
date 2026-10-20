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

状态（2026-10-18 T9 起）：**接口已钉死；资源层已到"分配决定成功概率"这一层**。
- 已实现：`delivery_rate`、`run_multiuser`（闭环骨架 + `csi` 接线）、`ConstantSuccess`（**仅 E1 对齐用**）、
  `SINRSuccess`（csi → Rayleigh 中断概率，锚定口径见其 docstring）、`require_privileged`（I3 的唯一合法读法）。
- 2026-10-19（验收报告 §3 修复）：
  A 信息边界 —— `UserSlot.cost_last` 改名 `cost_est`，**只由已送达观测算出**，真值只进 `StepRecord.costs`；
  B 死亡记账 —— `StepRecord` 增 `attempted_idx`，五类事件（reserved/attempted/delivered/valid/wasted）同义；
  C 终止代价 —— 新增 `window_cost()`（定长窗口 + 吸收态补齐），旧尾均值 `per_user_cost` 只留作历史对照；
  另加：逐用户 `warmup_steps`、3 维时变 `csi` 轨迹、`diagnostic` 开关（`privileged` 只给诊断臂）。
- **仍未实现**：`max-channel` 在**常数** csi 下退化为"恒选最优"（要时变轨迹才是信道择优；本版已支持喂轨迹，
  但轨迹由调用方生成）；I2 的逐条自检只在 `scripts/43` 里；全仓 `pytest` 在本机跑不了（无 pytest）。
- ★ 2026-10-18 20:15 起，「Q4（N 个用户的任务代价怎么合成一个比）未定前 T9 不得开工」**作废**：
  Q4 约束的是**报出来的那个比怎么合成**，T9–T11 的代码不编码任何聚合口径，聚合在离线做。
  边界改为：**T12 跑出任何 `oracle`/`round-robin` 比值之前，Q4 签注框必须已勾**（计划 §3.1）。
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

__all__ = ["UserSlot", "View", "Allocation", "Allocator", "StepRecord", "delivery_rate",
           "MissingStateError", "SuccessModel", "ConstantSuccess", "SINRSuccess",
           "default_uav_cost", "require_privileged", "run_multiuser", "window_cost",
           "counterfactual_next_costs", "counterfactual_horizon_costs"]


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
        cost_est: 由**最后一次成功送达的观测**算出的任务代价（标量，**逐用户量纲**）。
            ★★ 2026-10-19 验收修复（A 项）：这里**只允许**是 `cost_fn(last_obs)`，
            **不是** env 当前真值。旧字段名 `cost_last` 装的是"上一步真值"，等于给所有臂
            免费一条紧迫性遥测通道（验收 §3-A 的缺陷）。改名就是为了让旧语义无处藏身。
            从未送达时 = `cost_fn(reset 观测)`，可用 `n_frames == 0` 区分。
            ★★ 不同任务族的量纲不同（UAV = 稳态距离 m，CartPole = 0/1 存活），
            **不可跨用户相加**。聚合口径见 T8 §6-Q4；评估一律走逐用户比值。
    """

    idx: int
    alive: bool
    age: int
    last_obs: np.ndarray
    n_frames: int
    n_attempts: int
    csi: float
    cost_est: float


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
        served_idx: 本步**分配器选中**的用户（-1 = 空转）。
            ★ 语义是 `reserved`（派了份额），**不代表真的发了帧**：若该用户已死，
              份额空转（I6），`attempted_idx == -1`。这里沿用旧名字只为兼容既有脚本，
              新代码请读 `attempted_idx` / `delivered_idx`。
        attempted_idx: 本步**真的发出**了帧的用户（-1 = 没发）。
            `= served_idx if alive_before[served_idx] else -1`。
            ★ 2026-10-19 验收修复（B 项）：旧实现把"派给死人"直接写成 `delivered_idx`，
              实测 2116 条死者被记送达，而汇总又把这 2116 条剔除 ⇒ 原始表与汇总词义不一致。
        delivered_idx: 本步**物理送达**的用户（-1 = 没送达/没发）。
            `attempted_idx == i` 且 `u_i >= 1 − q_i` 才成立 ⇒ 死者**不可能**出现在这里。
            ★ 本轮 `valid ≡ delivered`（不给尸体发帧）；字段留给将来"链路可送达但任务已终止"。
        ages: 长度 N 的年龄（**决策前**的值，即分配器看到的那个）。
        costs: 长度 N 的该步**真值**任务代价（`alive=False` 的用户填 `nan`）。
            ★ 只进离线评估，**任何分配器都读不到**（信息边界见 UserSlot.cost_est）。
        alive: 长度 N 的存活标志（**决策前**的值 —— 与 `ages` 同一时点）。
            ★ 为什么必须与 `ages` 同时点：若记成"步进之后"的值，
              本步刚死掉的用户会被 `delivery_rate` 从分母里剔掉，
              而它本步确实被派过帧 ⇒ 送达率被低估（I6 的直接后果）。
    """

    episode: int
    t: int
    served_idx: int
    attempted_idx: int
    delivered_idx: int
    ages: tuple[int, ...]
    costs: tuple[float, ...]
    alive: tuple[bool, ...]


def delivery_rate(records: Sequence[StepRecord], n_users: int) -> dict:
    """**实测**每用户送达率与等效丢包（H2 的唯一合法来源）。

    2026-10-19 验收修复（B 项）后，五类事件分开统计，**同一词义贯穿原始表与汇总**：

    | 键 | 含义 | 分母 |
    |---|---|---|
    | `reserved` | Σ served_idx≥0（派了份额，含派给死人的空转） | — |
    | `attempted` | Σ attempted_idx≥0（真发了帧） | — |
    | `frames` (= `valid`) | Σ delivered_idx==i（物理送达且接收方存活） | — |
    | `wasted` | reserved − attempted（派给死人） | — |
    | `lost` | attempted − frames（发了没送到） | — |

    Returns:
        {"per_user": [frames_i / alive_steps_i, ...],       # 业务送达率（含份额损失）
         "p_eff": [1 − rate_i, ...],                        # 等效丢包（Gate 口径，定义不变）
         "link_q": [frames_i / attempts_i, ...],            # 纯链路成功率，只含信道损失
         "budget_reserved": int, "budget_attempted": int, "budget_wasted": int,
         "budget_lost": int,
         "budget_used": int,        # = budget_reserved（兼容旧键，语义未变）
         "budget_total": int,       # = len(records)，帧位
         "frames", "attempts", "alive_steps", "n_users"}

    ★ 为什么分母是"存活步数"而不是总步数：CartPole 掉杆后 `alive=False`（I6 不转嫁），
      拿总步数做分母会把"已经不用服务"算成"丢包"⇒ p_eff 虚高。这是 I6 的直接后果，
      两处必须一致，否则 T12 报出来的 p_eff 和实际预算对不上。

    ★★ 报 p_eff 时必须同时报 `budget_total` 与 N：
      单看 p_eff = 0.5 无法判断是"N=2 且每次成功"还是"N=4 且一半丢失"。

    ★ `link_q` 是 2026-10-19 新增的**诊断量**，不是新判据：`per_user` 里混了"份额少"与"链路差"
      两种损失，`link_q` 把两者分开。判据 P3 用的是 `link_q`（纯信道），不是 `per_user`。
    """
    frames = [0] * n_users
    attempts = [0] * n_users
    alive_steps = [0] * n_users
    reserved = attempted = 0
    for r in records:
        reserved += int(r.served_idx >= 0)
        attempted += int(getattr(r, "attempted_idx", -1) >= 0)
        for i in range(n_users):
            if i < len(r.alive) and bool(r.alive[i]):
                alive_steps[i] += 1
                if int(getattr(r, "attempted_idx", -1)) == i:
                    attempts[i] += 1
                if int(r.delivered_idx) == i:
                    frames[i] += 1
    per_user = [(frames[i] / alive_steps[i]) if alive_steps[i] else float("nan")
                for i in range(n_users)]
    p_eff = [(1.0 - x) if x == x else float("nan") for x in per_user]
    link_q = [(frames[i] / attempts[i]) if attempts[i] else float("nan")
              for i in range(n_users)]
    return {
        "per_user": per_user,
        "p_eff": p_eff,
        "link_q": link_q,
        "budget_reserved": int(reserved),
        "budget_attempted": int(attempted),
        "budget_wasted": int(reserved - attempted),
        "budget_lost": int(attempted - sum(frames)),
        "budget_used": int(reserved),
        # ★ 按本函数契约，这里是"总帧位"（= records 数），不是计划的 `budget_total`。
        #   两者只有在 budget_total == max_steps 且无提前终止时相等，别混用。
        "budget_total": len(records),
        "n_users": int(n_users),
        "frames": [int(x) for x in frames],
        "attempts": [int(x) for x in attempts],
        "alive_steps": [int(x) for x in alive_steps],
    }


# ================================================================ 资源层（T9）
class MissingStateError(RuntimeError):
    """分配器/资源层读到 `None` 形态的必需字段（I3）。**不许拿 0 / False 顶替。**"""


SuccessModel = Callable[[View, Allocation], np.ndarray]
"""`(View, Allocation) -> 长度 N 的每用户成功概率 ∈ [0,1]`。

★ 为什么成功概率要依赖 `Allocation`：预算按帧记（I1），每步**至多一个**用户被派帧（I5）
  ⇒ 本步没有同频干扰，SINR 只由"被服务的是谁"决定（该用户的 `csi`）。
  这是 T8 接口的直接后果，不是简化假设；若将来放宽到多帧并发，签名不用改，只改实现。
"""


@dataclass(frozen=True)
class ConstantSuccess:
    """每用户恒定成功概率 —— **E1 专用对齐装置，不是物理信道**。

    存在的唯一理由：让"多用户资源层 + N=1"与单用户 `lossy_schedule(p)` 走**同一条**
    随机数路径（同样每步 1 次抽样、同样判决式 `u >= 1−q`），E1 才能是逐位回归。
    `q = 0.7` ⇔ `lossy_schedule(0.3)`。

    ★★ 不得用它产出任何 Gate 数字：它把"通信约束"又写回外生常数，
       正是 T9 要消灭的那种做法。它只允许出现在自检脚本里。
    """
    q: float

    def __call__(self, view: "View", alloc: "Allocation") -> np.ndarray:
        return np.full(len(view.slots), float(self.q), dtype=float)


_PRIVILEGED_KEYS = ("true_obs", "true_cost_next", "true_age")


def require_privileged(view: "View", key: str):
    """**I3 唯一合法的读法**：读诊断臂的特权字段。拿不到就抛，不许 `or 0` / `or []`。

    为什么要有这个函数而不是让各臂自己写 `if view.privileged:`：
      `scheduling.py:55-92` 已经吃过一次亏 —— 把 `None` 当"没有特权"而落到默认分支，
      于是诊断臂在**没有真值**的情况下照样跑出一个数字，且不报错。
      Gate 2 要拿 `oracle-alloc` 当诊断上界，这种"静默降级"会让上界变成假的。

    Raises:
        ValueError: `key` 不在 `View.privileged` 的白名单里（防止随手塞新特权量）。
        MissingStateError: `view.privileged is None`、缺该键、或该键的值是 `None`。
    """
    if key not in _PRIVILEGED_KEYS:
        raise ValueError(f"未知特权键 {key!r}；允许 {list(_PRIVILEGED_KEYS)}")
    if view.privileged is None:
        raise MissingStateError(
            f"I3：本臂要求特权字段 {key!r}，但 view.privileged is None（非诊断臂或未接线）")
    if key not in view.privileged:
        raise MissingStateError(f"I3：view.privileged 缺键 {key!r}，现有 {sorted(view.privileged)}")
    val = view.privileged[key]
    if val is None:
        raise MissingStateError(f"I3：view.privileged[{key!r}] 是 None，不许当空值继续")
    return val


@dataclass(frozen=True)
class SINRSuccess:
    """Rayleigh 中断 → 每用户成功概率。**T9 要的就是这一层：q 由"服务谁"决定。**

    映射（★ 口径 A：2026-10-18 选定，见 `04-实验与代码/2026-10-18-T9-N2冒烟-v0.1.md` §6.1）
    -----------------------------------------------------------------------------
        κ      := −ln(q_ref)          # 默认 q_ref = 0.7 ⇒ κ = 0.3566749…
        q(csi) := exp(−κ / csi)       # csi ∈ (0, 1]
        q(0)   := 0.0                 # 全中断（按 csi→0+ 的极限取，不是除以 0）

    于是 `csi = 1 ⇔ q = q_ref`，`csi ↑ ⇒ q ↑`（单调），`csi → 0+ ⇒ q → 0`。

    ★★ 为什么锚在 q_ref 而不是锚在 γ̄/γ_th（口径 B 被否的理由）
      T8 的契约是 `csi ∈ [0,1]` 且 **1 = 最好**（`UserSlot` docstring）。若按
      `csi = γ̄/γ_th` 归一化，则 `q = exp(−1/csi)`，`csi = 1 ⇒ q = 0.3679` —— 想表达
      β 档的 `q = 0.7` 需要 `csi = 2.8037 > 1`，**越出契约**。口径 A 把 `csi = 1` 定义成
      "判-01 β 档那条信道"，代价是**"最好"是相对于 β 档而言的，不是物理最优**
      ⇒ 论文/汇报里必须这么写，不许说成"csi=1 即理想信道"。

    ★ 口径 A 的实际好处（不是审美）：`q(1) ≡ q_ref` ⇒ N=1 时 `SINRSuccess(0.7)` 与
      `ConstantSuccess(0.7)` 数值等价，E1 的三个锚点**一位都不用改**仍可当护栏。

    ★ 未做的边界：**csi 只支持"每用户一个常数"**（`run_multiuser(csi=…)`），
      逐 step 衰落未实现。影响：T11 的 `max-channel` 在静态 csi 下退化为
      "恒选 csi 最大的那个用户"，不是信道择优 —— 要真做信道择优，先补时变 csi。
    """

    q_ref: float = 0.7

    def __post_init__(self) -> None:
        if not (0.0 < float(self.q_ref) < 1.0):
            raise ValueError(f"q_ref 必须在 (0,1)，收到 {self.q_ref}")

    @property
    def kappa(self) -> float:
        """κ = −ln(q_ref)，即"参考档的 γ_th/γ̄"（口径 A 下 `csi = 1` 就取这个值）。"""
        return float(-np.log(float(self.q_ref)))

    def q(self, csi: float) -> float:
        c = float(csi)
        if not np.isfinite(c) or c < 0.0 or c > 1.0:
            raise ValueError(f"csi 必须落在 [0,1] 且有限，收到 {csi!r}")
        if c == 0.0:
            return 0.0
        return float(np.exp(-self.kappa / c))

    def __call__(self, view: "View", alloc: "Allocation") -> np.ndarray:
        return np.array([self.q(s.csi) for s in view.slots], dtype=float)


def default_uav_cost(obs: np.ndarray) -> float:
    """UAV 任务代价 = `‖p_drone − p_target‖`，与 `control.py` 的 `dist` 逐位同口径。"""
    o = np.asarray(obs, dtype=np.float32)
    return float(np.linalg.norm(o[0:2] - o[4:6]))


def window_cost(
    records: Sequence[StepRecord],
    n_users: int,
    *,
    window: int,
    absorbed,
    n_episodes: int | None = None,
) -> list[list[float]]:
    """**定长窗口 + 终止吸收代价**的逐集逐用户任务代价 `J_i(e)`（2026-10-19 验收修复 C 项）。

    定义（预注册于 `04-实验与代码/2026-10-19-T12-口径修订与四臂预注册-v0.1.md` §3）::

        J_i(e) = (1/L) · [ Σ_{t≤T_e} c_i(t)          # 仅存活步
                         + absorbed_i · (L − T_e) ]  # 终止/截断后按吸收态补齐

    L = `window`（评估窗口，两任务同一个 L）；T_e = 该集实际记录步数。

    ★★ 为什么必须这么算（验收 §3-C）：旧口径"对终止前已有样本取尾均值"下，
      **更早失败 ⇒ 样本更少 ⇒ 坏样本更少 ⇒ 代价更低**，把"死得快"奖励成了"成本低"。
      本函数里早死一定更贵：第 50 步死 ⇒ 至少 (L−50)/L 的窗口被 absorbed 填满。

    ★★★ 2026-10-19 自证伪：第一版用**逐集**的 `T_e = max(t)` 做补齐边界，于是
      "用户 1 第 12 步死、用户 0 活到 200"时，用户 1 的 188 个死步**一步都没补**，
      代价只有 0.005（等于把缺陷原样保留）。单用户自检全过，因为**单用户时 T_e 就是它自己的存活步数**。
      修正：补齐边界必须是**逐用户**的 `alive_steps_i`，不是 episode 的最后一步。

    ★ 为什么不做跨任务归一化：Gate 2 只用**逐用户比值** `r_i = J_i(arm)/J_i(rr)`，
      量纲自动约掉；一旦做绝对归一化，就得选一个"1 m 误差 = 多少失败概率"的汇率，
      而这个汇率能单独决定 Gate 2 生死（与 X45 同型风险）。

    Args:
        records: `run_multiuser` 的原始表（`costs` 是真值，`alive` 是决策前时点）。
        absorbed: 标量或长度 N 的吸收态代价（死亡后每步记多少）。
        n_episodes: 若给定，缺失的 episode 也按"全窗口 absorbed"补齐（不静默丢集）。

    Returns:
        `[episode][user]` 的列表，长度 = `n_episodes`（或实际出现过的集数）。
    """
    ab = ([float(absorbed)] * n_users) if np.isscalar(absorbed) else [float(x) for x in absorbed]
    if len(ab) != n_users:
        raise ValueError(f"absorbed 长度必须是 {n_users}，收到 {len(ab)}")
    if int(window) <= 0:
        raise ValueError("window 必须 > 0")

    n_eps = int(n_episodes) if n_episodes is not None else (max((r.episode for r in records), default=-1) + 1)
    sums = [[0.0] * n_users for _ in range(n_eps)]
    alive_cnt = [[0] * n_users for _ in range(n_eps)]   # ★ 逐用户，不是逐集
    for r in records:
        e = int(r.episode)
        if not (0 <= e < n_eps):
            raise ValueError(f"episode {e} 越界（n_episodes={n_eps}）")
        if int(r.t) > int(window):
            raise ValueError(f"记录步数超过窗口：t={r.t} > window={window}")
        for i in range(n_users):
            if i < len(r.alive) and bool(r.alive[i]):
                c = float(r.costs[i])
                if not np.isfinite(c):
                    raise ValueError(f"episode {e} t={r.t} user {i} 存活但代价不是有限值：{c}")
                sums[e][i] += c
                alive_cnt[e][i] += 1

    out: list[list[float]] = []
    for e in range(n_eps):
        out.append([(sums[e][i] + ab[i] * (int(window) - alive_cnt[e][i])) / float(window)
                    for i in range(n_users)])
    return out


def counterfactual_next_costs(
    envs: Sequence,
    controllers: Sequence,
    est: Sequence[np.ndarray],
    alive: Sequence[bool],
    q: Sequence[float],
    cfs: Sequence[Callable[[np.ndarray], float]],
) -> np.ndarray:
    """**期望 1 步前瞻**的代价矩阵 `Ĉ`（2026-10-19 T13；填 `View.privileged["true_cost_next"]`）。

    定义（预注册：`04-实验与代码/2026-10-19-T13-lookahead臂实现口径-预注册-v0.1.md` §2）::

        Ĉ[j][i] = q_i · c_j(t+1 | 本步派给 i 且送达) + (1−q_i) · c_j(t+1 | 派给 i 但丢失)

    ★ 索引顺序是 `[候选分配 i][用户 j]`（与 `View.privileged["true_cost_next"]` 的"第 i 行 =
      服务用户 i 后下一步各用户的真实代价"一致），**不是** `[用户][候选]`。L2 自检专门验这一条。

    ★★ 为什么必须走**两步**（这里最容易接错，2026-10-19 实测踩点）
      runner 的时序是 `build_view → alloc → 抽样 → 各用户 step → 送达者 est ← nxt`。
      ⇒ 本步的动作在**送达之前**就算好了，所以"派给谁"对**本步**代价没有任何影响；
        它只通过 `est` 影响 **t+1** 的代价。因此：
            第 1 步（所有分支共用，与决策无关）：用当前 est 各走一步 ⇒ nxt
            第 2 步（分支）：送达 ⇒ est_i ← nxt_i；丢失 ⇒ est 不变；再各走一步 ⇒ c(t+1)
      若只走一步，"下一步代价"对每个候选恒等 ⇒ 臂退化成"恒选 idx 小"，且**不会报错**
      （`scripts/44` 的 L8 就是为打死这种接法写的）。

    ★ 为什么按 `q_i` 取期望而不是"假设一定送达"：`q_i` 由**公开**的 `csi` 与公开的 SINR 映射
      算出，不是特权量；而乐观分支会让臂系统性高估低 q 用户（本场景 ≈0.53）的收益，
      那不是前瞻，是错误的信念。

    ★ 反事实**只读**原对象：env / controller / est 全部 `deepcopy` 后再步进，
      主轨迹、控制器内部状态（`PDRelativeController._tgt_prev`）与任何 rng 都不被触碰。
      这条由 `scripts/44` 的 L5（`rr` 带/不带反事实必须逐位相同）打死。

    Args:
        envs: 长度 N 的 env 适配器（只读，内部会被 deepcopy）。
        controllers: 长度 N 的控制器（**状态会被 deepcopy**，`act()` 的副作用不落到原件）。
        est: 长度 N 的估计观测（控制器实际看到的）。
        alive: 长度 N 的存活标志（**决策前**时点）。
        q: 长度 N 的每用户成功概率（本步 `success(view, alloc)` 的输出）。
        cfs: 长度 N 的逐用户真值代价函数。

    Returns:
        `(N, N)` 数组；`Ĉ[i][j]` = 若本步派给 i，用户 j 在 t+1 的期望代价。
        **死亡候选的行全 NaN**（不该被选；臂必须只在 alive 行里取 argmin）。
    """
    n = len(envs)
    if not (len(controllers) == len(est) == len(alive) == len(q) == len(cfs) == n):
        raise ValueError(
            f"counterfactual_next_costs 的输入长度不一致：envs={len(envs)} "
            f"controllers={len(controllers)} est={len(est)} alive={len(alive)} "
            f"q={len(q)} cfs={len(cfs)}")

    # ---------- 第 1 步：与决策无关的公共未来 ----------
    cf_envs = [copy.deepcopy(e) if alive[i] else None for i, e in enumerate(envs)]
    cf_ctrls = [copy.deepcopy(c) if alive[i] else None for i, c in enumerate(controllers)]
    nxt: list[np.ndarray | None] = [None] * n
    alive_cf = [bool(alive[i]) for i in range(n)]
    c_term: list[float] = [float("nan")] * n      # 第 1 步就终止的用户：t+1 用终止观测的代价
    for i in range(n):
        if not alive_cf[i]:
            continue
        res = cf_envs[i].step(cf_ctrls[i].act(est[i]))
        obs = np.asarray(res.obs, dtype=np.float32)
        nxt[i] = obs
        if res.terminated or res.truncated:
            alive_cf[i] = False
            c_term[i] = float(cfs[i](obs))

    # ---------- 第 2 步：逐候选 × 逐送达分支 ----------
    mat = np.full((n, n), np.nan, dtype=float)
    for cand in range(n):
        if not alive[cand]:
            continue
        row = np.zeros(n, dtype=float)
        for delivered in (True, False):
            w = float(q[cand]) if delivered else 1.0 - float(q[cand])
            if w <= 0.0:                      # q=0/1 的退化分支权重为 0，不算（省一半 clone）
                continue
            est2 = [np.asarray(e, dtype=np.float64).reshape(-1).copy() for e in est]
            if delivered:
                est2[cand] = nxt[cand].astype(np.float64)
            # 第 1 步已终止的用户在第 2 步不步进，直接记终止代价（与 absorbed 同值，见 §2.4）
            for j in range(n):
                if alive[j] and not alive_cf[j]:
                    row[j] += w * c_term[j]
            step2 = [j for j in range(n) if alive_cf[j]]
            if not step2:
                continue
            e2 = {j: copy.deepcopy(cf_envs[j]) for j in step2}
            c2 = {j: copy.deepcopy(cf_ctrls[j]) for j in step2}
            for j in step2:
                res = e2[j].step(c2[j].act(est2[j]))
                row[j] += w * float(cfs[j](np.asarray(res.obs, dtype=np.float32)))
        mat[cand] = row
    return mat


def counterfactual_horizon_costs(
    envs: Sequence,
    controllers: Sequence,
    est: Sequence[np.ndarray],
    alive: Sequence[bool],
    q: Sequence[float],
    cfs: Sequence[Callable[[np.ndarray], float]],
    *,
    horizon: int = 1,
    ages: Sequence[int] | None = None,
) -> np.ndarray:
    """**H 步前瞻**的逐步平均期望代价矩阵（2026-10-19 T14；填 `View.privileged["true_cost_next"]`）。

    预注册：`04-实验与代码/2026-10-19-T14-多步前瞻视界-预注册-v0.1.md` §2。定义::

        Ĉ_H[i][j] = (1/H) · Σ_{k=1..H} E[ c_j(t+k) | 本步派给 i ]

    索引顺序仍是 `[候选 i][用户 j]`（与 `counterfactual_next_costs` 一致）。

    ★★ 为什么除以 H（不是可选项）
      不除的话量纲是"H 步代价和"，与 `s_j`（每步代价）不同量纲 ⇒ `cart_gain(H)` 会随 H
      机械增长，`0.1·s_cart` 这条阈值就成了随 H 走的橡皮筋。除以 H 后：① 所有 H 同量纲；
      ② **H=1 时与 `counterfactual_next_costs` 逐位相同**（L9 的逐位回归就靠这条）。

    ★★★ 基准策略（本函数唯一的新假设，预注册 §2.4）
      t+1 之后（k ≥ 2）没有人再"前瞻"，按 **max-age** 选人、且**确定性送达**（q=1）。
      选 max-age：本臂回答的是"相对 age 基线，此刻偏离一步、往后看 H 步是否划算"
      = 策略迭代里的**一步偏离评估**。
      选确定性送达：把 `2^H` 的分支压成 2（只在 t+1 分支），H=20 才跑得动；
      **偏差方向是低估"此刻服务它"的价值**（未来太乐观）⇒ 对臂保守，不虚报收益。
      ⇒ 标签是"完美动力学 + H 步一步偏离评估"，**不是**上界，**不是** model-based。

    ★ 终止（预注册 §2.5）：反事实里终止的用户，从终止那步起在**剩余步数上冻结**
      终止观测的代价（与 `window_cost` 的 absorbed 同值：25 m / 1.0）。

    ★ 时序（与 runner 逐位对齐，见 `counterfactual_next_costs` 的两步说明）
      反事实第 1 步 = runner 的第 t 步（与决策无关，所有分支共用）；
      第 2 步 = t+1（分支：送达/丢失，权重 q_i / 1−q_i）；
      第 k 步（k ≥ 3）= t+k−1，按基准策略。
      ⇒ H=1 只累计第 2 步，与 `counterfactual_next_costs` 是**同一个对象**。

    Args:
        envs, controllers, est, alive, q, cfs: 同 `counterfactual_next_costs`。
        horizon: H ≥ 1。
        ages: **决策时点**的年龄（与 `StepRecord.ages` 同一时点）。H=1 时可不给；
            H>1 必须给（基准策略要用），缺了直接 `ValueError`（不许默认 0 蒙过去）。

    Returns:
        `(N, N)` 数组；死亡候选的行全 NaN。
    """
    n = len(envs)
    if not (len(controllers) == len(est) == len(alive) == len(q) == len(cfs) == n):
        raise ValueError(
            f"counterfactual_horizon_costs 的输入长度不一致：envs={len(envs)} "
            f"controllers={len(controllers)} est={len(est)} alive={len(alive)} "
            f"q={len(q)} cfs={len(cfs)}")
    H = int(horizon)
    if H < 1:
        raise ValueError(f"horizon 必须 ≥ 1，收到 {horizon!r}")
    if ages is None:
        if H > 1:
            # ★ 不许默认成 0：ages 全 0 时 max-age 基准策略恒选 idx 小，
            #   会静默把"基准策略"变成"恒选用户 0"，而结果看起来完全正常。
            raise ValueError("horizon > 1 需要 ages（基准策略 max-age 依赖它），收到 None")
        ag0 = [0] * n
    else:
        if len(ages) != n:
            raise ValueError(f"ages 长度必须是 {n}，收到 {len(ages)}")
        ag0 = [int(a) for a in ages]

    # ---------- 反事实第 1 步：与决策无关的公共未来 ----------
    cf_envs = [copy.deepcopy(e) if alive[i] else None for i, e in enumerate(envs)]
    cf_ctrls = [copy.deepcopy(c) if alive[i] else None for i, c in enumerate(controllers)]
    nxt: list[np.ndarray | None] = [None] * n
    alive1 = [bool(alive[i]) for i in range(n)]
    c_term = [float("nan")] * n
    for i in range(n):
        if not alive1[i]:
            continue
        res = cf_envs[i].step(cf_ctrls[i].act(est[i]))
        o = np.asarray(res.obs, dtype=np.float32)
        nxt[i] = o
        if res.terminated or res.truncated:
            alive1[i] = False
            c_term[i] = float(cfs[i](o))

    # ---------- 第 2 步（分支）… 第 H+1 步（基准策略）----------
    mat = np.full((n, n), np.nan, dtype=float)
    for cand in range(n):
        if not alive[cand]:
            continue
        row = np.zeros(n, dtype=float)
        for delivered in (True, False):
            w = float(q[cand]) if delivered else 1.0 - float(q[cand])
            if w <= 0.0:                      # q=0/1 的退化分支权重为 0，不算
                continue
            est_cur = [np.asarray(e, dtype=np.float64).reshape(-1).copy() for e in est]
            if delivered:
                est_cur[cand] = nxt[cand].astype(np.float64)
            al = list(alive1)
            ag = list(ag0)
            for j in range(n):
                if alive[j]:                  # 第 t 步的送达 ⇒ t+1 决策时点的 age
                    ag[j] = 0 if (delivered and j == cand) else ag[j] + 1
            acc = np.zeros(n, dtype=float)
            frozen: dict[int, float] = {}
            for j in range(n):
                if alive[j] and not al[j]:    # 第 1 步就终止：整段窗口按终止代价
                    acc[j] += w * c_term[j] * H
                    frozen[j] = c_term[j]
            e_k = {j: copy.deepcopy(cf_envs[j]) for j in range(n) if al[j]}
            c_k = {j: copy.deepcopy(cf_ctrls[j]) for j in range(n) if al[j]}
            obs_prev = {j: nxt[j] for j in range(n) if al[j]}

            for s in range(2, H + 2):
                live = [j for j in range(n) if al[j]]
                if s >= 3 and live:
                    # 上一步的基准策略决策：max-age（同分取 idx 小）+ 确定性送达
                    u = live[0]
                    for j in live[1:]:
                        if ag[j] > ag[u]:
                            u = j
                    est_cur[u] = np.asarray(obs_prev[u], dtype=np.float64).reshape(-1).copy()
                    for j in live:
                        ag[j] = 0 if j == u else ag[j] + 1
                for j in range(n):
                    if not alive[j]:
                        continue                  # 决策时点已死：不进矩阵（列为 0，与旧实现一致）
                    if not al[j]:
                        acc[j] += w * frozen[j]   # 已终止：本步记冻结代价
                        continue
                    res = e_k[j].step(c_k[j].act(est_cur[j]))
                    o = np.asarray(res.obs, dtype=np.float32)
                    acc[j] += w * float(cfs[j](o))
                    obs_prev[j] = o
                    if res.terminated or res.truncated:
                        al[j] = False
                        frozen[j] = float(cfs[j](o))
            row += acc
        mat[cand] = row / float(H)
    return mat


def run_multiuser(
    envs: Sequence,
    controllers: Sequence,
    allocator: Allocator,
    success: SuccessModel,
    *,
    n_episodes: int = 30,
    seed: int = 0,
    max_steps: int = 200,
    warmup_steps: int | Sequence[int] = 0,
    budget_total: int | None = None,
    cost_fn: Callable[[np.ndarray], float] | Sequence[Callable[[np.ndarray], float]] | None = None,
    tail_frac: float = 0.25,
    csi: float | Sequence[float] | np.ndarray | None = None,
    diagnostic: bool = False,
    counterfactual: bool = False,
    cf_horizon: int = 1,
    predictor=None,
    pred_probe: Callable[[dict], None] | None = None,
    decision_probe: Callable[[dict], None] | None = None,
) -> dict:
    """多用户闭环骨架：**分配 → 成功概率 → 送达 → 步进 → 记账**，逐 step 落原始表。

    时序（与 `multiuser.py` 模块头的第 1–5 步一一对应）
    ------------------------------------------------------------------
        第 t 步：build_view(t) → allocator(view) → 抽信道 → 各存活用户 step → 写 StepRecord

    ★ 与 `control.py:run_closed_loop_control` 的唯一差别：**谁被服务是决策**，不是外生调度。
      N=1 + 恒成功率时两者必须**逐位一致**（E1 的判据）；不一致只可能是抽样次数/顺序错。

    ★★ 两条随机数纪律（I4 的实现方式，写死以免日后被"优化"掉）    ------------------------------------------------------------------
    1. **信道流与分配器流分开**：`rng_ch = default_rng(seed)` 只管送达判决；
       `rng_alloc = default_rng([seed, 2**31−1])` 才交给分配器。
       混用一个流 ⇒ ε-greedy 类臂多抽一次就改掉后面所有送达骰子，"换臂"变成"换轨迹"。
    2. **信道每步固定抽 `n_users` 个 double，与"谁被服务"无关**。两个后果都是要的：
       ① N=1 时抽样次数 = 每步 1 次，与 `lossy_schedule` 逐位相同（E1 的前提）；
       ② 同种子跨臂 ⇒ **公共随机数**：第 t 步、第 i 个用户的骰子与臂无关，
          于是"臂 A 与臂 B 的差"不再混入信道噪声。
       代价：未服务用户的骰子被浪费掉。这个浪费是**刻意的**，不要改成"只给被派帧的人抽"。

    ★ `csi` 参数（2026-10-18 T9 新增，2026-10-19 扩展）：`None` ⇒ 每用户 1.0（与 E1 逐位可比）；
    标量 ⇒ 所有用户同值；长度 N 的序列 ⇒ 逐用户常数；
    **形状 `(n_episodes, max_steps, N)` 的数组 ⇒ 逐 step 时变 csi（T12 的异质链路）**。
    时变分支只从数组**读**，不抽随机数 ⇒ 同一条轨迹可以喂给所有臂（公共随机数，臂间差里不含信道噪声）。

    ★ `warmup_steps`（2026-10-19）：标量 ⇒ 所有用户同样预热；序列 ⇒ 逐用户。
    **为什么要逐用户**：UAV 需要 300 步让 PD 收敛，CartPole 不需要；给 CartPole 预热 300 步
    等于在第 300 步前就掉杆（判-01 已踩过"预热把被测段吃掉"的坑）。

    ★ `diagnostic`（2026-10-19）：只有诊断臂为 True。为 True 时 `View.privileged` 里放
    `{"true_obs": ..., "true_age": ...}`（T8 白名单内的键，**不新增键**）。
    非诊断臂 `privileged is None`，且 `UserSlot.cost_est` **只随送达更新**（A 项修复）。

    ★ `counterfactual`（2026-10-19 T13）：为 True 时**额外**算 `privileged["true_cost_next"]`
    （`(N,N)` 的期望前瞻代价矩阵，见 `counterfactual_next_costs`），供 `lookahead` 臂使用。
    **必须与 `diagnostic=True` 同时使用**（它本身就是特权量）；`counterfactual=True, diagnostic=False`
    ⇒ 直接 `ValueError`，不许静默忽略（否则"以为接线了其实没接"会悄悄跑出一个正常数字）。

    ★ `cf_horizon`（2026-10-19 T14）：矩阵的视界 H，走 `counterfactual_horizon_costs`。
    `H=1` ⇒ 与 T13 的 `counterfactual_next_costs` **逐位相同**（L9 回归）；
    `H>1` ⇒ 逐步平均的 H 步前瞻（t+1 之后按 max-age 基准策略、确定性送达）。
    若 `allocator` 带 `horizon` 属性且与 `cf_horizon` 不一致 ⇒ `ValueError`（R12 接线守卫）。

    ★ `predictor`（2026-10-19 T22）：接收端的估计观测 `est[i]` 从"零阶保持"改为"预测"。
    `None` ⇒ **完全走既有路径**（逐位回归红线：T12–T21 的全部数字都是这条路径）。
    非 None 时每步在**动作产生之前**用 `predictor.predict(i, age, actions_since)` 刷新 `est[i]`，
    送达时 `est ← nxt` 并 `observe_delivery` + 清空该用户的动作缓冲。
    预测器实现见 `wmlab/eval/predictors.py`（信息集边界写在那里）。

    ★★ 预测**不得**进入分配器的信息集：`View.slots[i].last_obs` / `cost_est` 仍是送达值。
      否则等于给所有臂免费一条"预测真值"通道，T22 就变成"偷偷放宽信息约束"。
      `scripts/53` 的 S3/S9 就是为打死这种接法写的。

    ★ `pred_probe`（诊断回调，默认 None）：每步调用一次，payload 见函数末尾的组装代码。
      只给自检脚本用；不参与任何决策、不写 `results/`。

    Returns:
        {"records": [StepRecord…],                 # Gate 2 要的原始表（真值代价 + 五类事件）
         "per_user_cost": [均值(逐集尾均)…],        # 旧口径，**仅供历史对照，不得用于 Gate 2**
         "per_user_survival": [均值(存活步数)…],
         "budget_reserved": int, "budget_attempted": int, "budget_wasted": int,
         "budget_cap_per_episode": int,
         "n_steps": int, "n_users": int, "window": int}

    ★★ 为什么还留着 `per_user_cost`（尾均值）：`scripts/39/40/41` 的护栏用它。
      **Gate 2 一律用 `window_cost()`**（定长窗口 + 吸收代价），它在"早死"时给出更高代价，
      而尾均值给出更低 —— 两者结论方向不同，混用就是 X45 那种口径反转。
    """
    n_users = len(envs)
    if len(controllers) != n_users:
        raise ValueError(f"envs/controllers 数量不等：{n_users} vs {len(controllers)}")
    # ★ 2026-10-19：cost_fn 支持**逐用户**（异质任务必须逐用户定义，UAV 是米、CartPole 是 0/1）。
    #   标量函数 ⇒ 广播给所有用户（旧行为，E1/39-41 逐位不变）。
    if cost_fn is None:
        cfs: list[Callable[[np.ndarray], float]] = [default_uav_cost] * n_users
    elif callable(cost_fn):
        cfs = [cost_fn] * n_users
    else:
        cfs = list(cost_fn)
        if len(cfs) != n_users:
            raise ValueError(f"cost_fn 序列长度必须是 {n_users}，收到 {len(cfs)}")
        if not all(callable(f) for f in cfs):
            raise TypeError("cost_fn 序列里每一项都必须是可调用对象")
    cost_fn = cfs[0] if n_users == 1 else cost_fn   # 仅为兼容旧调试打印，不参与计算
    budget_cap = int(max_steps if budget_total is None else budget_total)
    if budget_cap < 0:
        raise ValueError("budget_total 必须 ≥ 0（单位 = 帧）")
    if counterfactual and not diagnostic:
        # ★ 不许静默忽略：`counterfactual` 产出的矩阵是特权量，非诊断运行里给它等于放宽信息集。
        raise ValueError("counterfactual=True 必须配 diagnostic=True（true_cost_next 是特权量，"
                         "见 View.privileged 白名单）")
    if int(cf_horizon) < 1:
        raise ValueError(f"cf_horizon 必须 ≥ 1，收到 {cf_horizon!r}")
    if counterfactual:
        # ★ R12 接线守卫（2026-10-19 T14）：臂自报的视界与 runner 实际算的视界必须一致。
        #   不一致时不会报错，只会让"以为在看 20 步"的臂拿到 1 步的矩阵 —— 结果依旧"正常"，
        #   这是最典型的一类静默错。没有 horizon 属性的臂（如 rr/age）不检查。
        want = getattr(allocator, "horizon", None)
        if want is not None and int(want) != int(cf_horizon):
            raise ValueError(
                f"接线不一致：allocator.horizon={want} 与 cf_horizon={cf_horizon} 不同"
                f"（臂会拿到错误的视界，且不会报错）")

    if predictor is not None:
        # ★ 结构检查：缺方法就炸，别让"接了一半"静默跑成 ZOH（那会让 T22 的结论完全落空）
        for meth in ("reset_episode", "observe_delivery", "predict"):
            if not callable(getattr(predictor, meth, None)):
                raise TypeError(f"predictor 缺方法 {meth}（收到 {type(predictor).__name__}）")
        if getattr(predictor, "consumes_env", False) and not hasattr(predictor, "_on_delivery"):
            raise TypeError(f"predictor {type(predictor).__name__} 声称 consumes_env 却没实现快照钩子")

    warm = ([int(warmup_steps)] * n_users) if np.isscalar(warmup_steps) \
        else [int(x) for x in warmup_steps]
    if len(warm) != n_users:
        raise ValueError(f"warmup_steps 长度必须是 {n_users}，收到 {len(warm)}")

    rng_ch = np.random.default_rng(int(seed))
    rng_alloc = np.random.default_rng([int(seed), 2 ** 31 - 1])

    # ---------- csi 解析（2026-10-18 常数 / 2026-10-19 时变轨迹）----------
    csi_traj = None
    csi_per_user = None
    if csi is None:
        csi_per_user = [1.0] * n_users
    elif np.isscalar(csi):
        csi_per_user = [float(csi)] * n_users
    else:
        arr = np.asarray(csi, dtype=float)
        if arr.ndim == 3:
            if arr.shape[0] != int(n_episodes) or arr.shape[1] != int(max_steps) \
                    or arr.shape[2] != n_users:
                raise ValueError(
                    f"csi 轨迹形状必须是 ({int(n_episodes)}, {int(max_steps)}, {n_users})，"
                    f"收到 {arr.shape}")
            if not np.all(np.isfinite(arr)) or arr.min() < 0.0 or arr.max() > 1.0:
                raise ValueError("csi 轨迹必须落在 [0,1] 且有限（契约见 UserSlot.csi）")
            csi_traj = arr
        elif arr.ndim == 1:
            csi_per_user = [float(x) for x in arr]
            if len(csi_per_user) != n_users:
                raise ValueError(f"csi 长度必须是 {n_users}，收到 {len(csi_per_user)}")
        else:
            raise ValueError(f"csi 只接受标量 / 长度 {n_users} 序列 / 3 维轨迹，收到 ndim={arr.ndim}")
    if csi_per_user is not None:
        for i, c in enumerate(csi_per_user):
            if not np.isfinite(c) or c < 0.0 or c > 1.0:
                raise ValueError(f"csi[{i}] 必须落在 [0,1] 且有限，收到 {c!r}（契约见 UserSlot.csi）")

    records: list[StepRecord] = []
    tail: list[list[float]] = [[] for _ in range(n_users)]
    surv: list[list[int]] = [[] for _ in range(n_users)]
    budget_used = 0
    n_steps = 0

    for e in range(int(n_episodes)):
        obs = [np.asarray(env.reset(seed=int(seed) + e), dtype=np.float32).copy()
               for env in envs]
        est = [o.copy() for o in obs]
        for i, c in enumerate(controllers):
            c.reset(obs[i])

        alive = [True] * n_users
        age = [0] * n_users
        n_frames = [0] * n_users
        n_attempts = [0] * n_users
        last_obs = [o.copy() for o in obs]
        true_obs = [o.copy() for o in obs]       # 评估侧真值，只用于 privileged / 记录
        # ★ A 项：cost_est 只能从"已送达的观测"算。t=0 的 reset 观测在开局即已知，
        #   算进去是合规的（它不需要传输），用 n_frames==0 可区分。
        cost_est = [float(cfs[i](o)) for i, o in enumerate(obs)]
        ep_dist: list[list[float]] = [[] for _ in range(n_users)]
        ep_alive = [0] * n_users

        # ---------- 预热：每步送真值，控制器先收敛到稳态（不消耗任何 rng）----------
        for _ in range(max(warm) if n_users else 0):
            for i in range(n_users):
                if not alive[i] or _ >= warm[i]:
                    continue
                res = envs[i].step(controllers[i].act(est[i]))
                est[i] = np.asarray(res.obs, dtype=np.float32).copy()
                true_obs[i] = est[i].copy()
                if res.terminated or res.truncated:
                    alive[i] = False

        # ★ T19/P0-2：预热末态同步。预热按定义"每步送真值"⇒ 末次观测是**免费可得**的，
        #   所以 `last_obs`/`cost_est` 必须跟上 `est`，否则主循环第一步的 `cost_est`
        #   还停在 t=0 的 reset 观测量纲上（UAV 预热 25 步，差得很远）。
        #   旧版只更新 est/true_obs ⇒ "预热说免费给了真值，但估计代价没跟着走"，
        #   属于接线不一致（R12）。当前没有任何臂读这两字段，故数值不变；P1 的预测臂会读。
        for i in range(n_users):
            last_obs[i] = est[i].copy()
            cost_est[i] = float(cfs[i](est[i]))

        # ★ T22：预测器初态 = "预热末次完整观测免费可得"（与上面 P0-2 的预热同步同一口径，
        #   否则 `Linear` 会把 t=0 的 reset 观测当成上一个送达点，速度量纲直接错）。
        actions_since: list[list] = [[] for _ in range(n_users)]
        if predictor is not None:
            predictor.reset_episode()
            for i in range(n_users):
                if alive[i]:
                    predictor.observe_delivery(i, last_obs[i], age_before=0, env=envs[i])

        t = 0
        used_ep = 0
        while t < int(max_steps) and any(alive):
            t += 1
            alive_before = list(alive)
            ages_before = list(age)
            csi_step = ([float(csi_per_user[i]) for i in range(n_users)]
                        if csi_traj is None else
                        [float(csi_traj[e, t - 1, i]) for i in range(n_users)])
            slots = tuple(
                UserSlot(idx=i, alive=bool(alive[i]), age=int(age[i]),
                         last_obs=last_obs[i], n_frames=int(n_frames[i]),
                         n_attempts=int(n_attempts[i]), csi=float(csi_step[i]),
                         cost_est=float(cost_est[i]))
                for i in range(n_users)
            )
            priv = ({"true_obs": tuple(o.copy() for o in true_obs),
                     "true_age": tuple(int(x) for x in ages_before)}
                    if diagnostic else None)
            if counterfactual:
                # ★ q 必须在本步**决策之前**知道（矩阵要进 view），而矩阵又是分配器的输入
                #   ⇒ 只能用"与 alloc 无关"的方式取 q。契约上 SuccessModel 只依赖 csi
                #   （SINRSuccess.__call__ 只读 slots.csi），所以这里传一个零分配探针。
                #   若将来 success 真的依赖 alloc，"取 q 用零分配"就成了错误假设 ⇒ 必须在
                #   预注册 §6 里记并改（不许在这里静默）。
                _probe = Allocation(served=np.zeros(n_users, dtype=bool), reason="cf-q-probe")
                q_cf = np.asarray(success(
                    View(t=int(t), slots=slots, budget_left=int(budget_cap - used_ep),
                         budget_total=int(budget_cap), privileged=None), _probe), dtype=float)
                cf_mat = counterfactual_horizon_costs(
                    envs, controllers, est, alive_before, q_cf, cfs,
                    horizon=int(cf_horizon), ages=ages_before)
                priv["true_cost_next"] = cf_mat
            view = View(t=int(t), slots=slots, budget_left=int(budget_cap - used_ep),
                        budget_total=int(budget_cap), privileged=priv)

            alloc = allocator(view, rng_alloc)
            served = np.asarray(alloc.served, dtype=bool).reshape(-1)
            if served.shape != (n_users,):
                raise ValueError(f"served 形状必须是 ({n_users},)，收到 {served.shape}")
            n_served = int(served.sum())
            if n_served > 1:
                raise RuntimeError(f"I1/I5 违约：一步派出 {n_served} 帧（上限 1）")
            if n_served and used_ep >= budget_cap:
                raise RuntimeError(f"I1 违约：预算 {budget_cap} 帧已用尽仍要派帧")
            used_ep += n_served

            probs = np.asarray(success(view, alloc), dtype=float).reshape(-1)
            if probs.shape != (n_users,):
                raise ValueError(f"success 返回形状必须是 ({n_users},)，收到 {probs.shape}")
            # ★ T19/P0-2：NaN 能通过下面的区间检查（NaN 的比较恒为 False）⇒ 会静默算成"丢包"。
            #   这类错误不报错、只让结果悄悄变差，属于最难查的一类，必须在入口拦死。
            if not np.all(np.isfinite(probs)):
                raise ValueError(f"成功概率含 NaN/inf：{probs.tolist()} "
                                 f"（NaN 会躲过 [0,1] 区间检查，必须显式拦）")
            if np.any((probs < 0.0) | (probs > 1.0)):
                raise ValueError(f"成功概率越界：{probs.tolist()}")

            u = rng_ch.random(n_users)      # ★ 固定抽 n_users 个，见 docstring 纪律 2
            delivered = np.zeros(n_users, dtype=bool)
            si = -1
            ai = -1
            if n_served:
                si = int(np.flatnonzero(served)[0])
                # ★ B 项：派给死人 = 份额空转，**不发帧**（不可能送达）。
                if alive_before[si]:
                    ai = si
                    n_attempts[ai] += 1
                    delivered[ai] = bool(u[ai] >= 1.0 - probs[ai])

            # ★ T22：预测必须在 `act()` 之前刷新 `est`（否则控制器看到的仍是上一拍的保持值），
            #   且**不得**改 View —— 分配器看到的是 `last_obs`，不是预测值。
            #   此刻 `actions_since[i]` 恰好有 `age[i]` 项（第 t−d … t−1 步实际施加的动作）。
            if predictor is not None:
                for i in range(n_users):
                    if alive_before[i]:
                        est[i] = predictor.predict(i, age=int(age[i]),
                                                  actions_since=actions_since[i])
            if pred_probe is not None:
                est_pre = [o.copy() for o in est]
                true_pre = [o.copy() for o in true_obs]
                n_since_before = [int(len(actions_since[i])) for i in range(n_users)]
                acts_now: list = [None] * n_users
                acts_after: list = [None] * n_users

            costs_step = [float("nan")] * n_users
            for i in range(n_users):
                if not alive_before[i]:
                    continue
                if decision_probe is not None:
                    # ★ T26：**决策时点**的只读探针（在 `act()` 之前，控制器状态尚未被本步推进）。
                    #   用途 = 在同一真实状态上做"补偿动作 vs 保持动作"的反事实分支。
                    #   契约：探针**不得**修改 env / controller / est（会静默改掉主轨迹）；
                    #   需要快照请自己 `copy.deepcopy`。默认 None ⇒ 零行为变化（G4 逐位核）。
                    decision_probe({
                        "episode": int(e), "t": int(t), "user": int(i),
                        "env": envs[i], "controller": controllers[i],
                        "est": np.asarray(est[i], dtype=np.float64).copy(),
                        "last_obs": np.asarray(last_obs[i], dtype=np.float64).copy(),
                        "age": int(age[i]), "cost_fn": cfs[i],
                    })
                act_i = controllers[i].act(est[i])
                res = envs[i].step(act_i)
                nxt = np.asarray(res.obs, dtype=np.float32)
                c = float(cfs[i](nxt))
                costs_step[i] = c
                true_obs[i] = nxt.copy()
                ep_dist[i].append(c)
                ep_alive[i] += 1
                if pred_probe is not None:
                    acts_now[i] = np.asarray(act_i).copy()
                # ★ 实际施加的动作**无条件**入缓冲（与是否送达无关：动作用的是旧估计，照样执行）
                actions_since[i].append(np.asarray(act_i).copy())
                if delivered[i]:
                    est[i] = nxt.copy()
                    last_obs[i] = nxt.copy()
                    cost_est[i] = float(cfs[i](nxt))   # ★ A 项：只有送达才更新估计代价
                    n_frames[i] += 1
                    age[i] = 0
                    if predictor is not None:
                        # 快照必须取"本步已步进"之后的 env（a_t 已烘进状态）⇒ 缓冲清空不重放
                        # ★ T23-a：把**本步实际施加**的动作交给预测器（`ARX` 学一步映射要它）。
                        #   此刻缓冲还没清空，最后一项就是 a_t（上面无条件 append 的那一条）。
                        _a_deliv = (np.asarray(actions_since[i][-1]).copy()
                                    if actions_since[i] else None)
                        # ★ T23-a-2：整段动作窗口（长度 == gap_prev）交给预测器，
                        #   `MultiLag` 的多步模型需要它；`ARX` 仍只读 `action=`（行为不变）。
                        _a_seq = ([np.asarray(a).copy() for a in actions_since[i]]
                                  if actions_since[i] else None)
                        predictor.observe_delivery(i, nxt, age_before=int(ages_before[i]),
                                                   env=envs[i], action=_a_deliv,
                                                   action_seq=_a_seq)
                        actions_since[i] = []
                else:
                    age[i] += 1
                if pred_probe is not None:
                    acts_after[i] = int(len(actions_since[i]))
                if res.terminated or res.truncated:
                    alive[i] = False

            if pred_probe is not None:
                pred_probe({
                    "episode": int(e), "t": int(t),
                    "age": [int(x) for x in ages_before],
                    "alive": [bool(x) for x in alive_before],
                    "est_pre": est_pre, "true_pre": true_pre,
                    "est_post": [o.copy() for o in est],
                    "true_post": [o.copy() for o in true_obs],
                    "delivered": [bool(x) for x in delivered],
                    "actions": acts_now,
                    "n_since_before": n_since_before,
                    "n_since_after": acts_after,
                })

            records.append(StepRecord(
                episode=int(e), t=int(t), served_idx=int(si),
                attempted_idx=int(ai),
                delivered_idx=int(ai if (ai >= 0 and delivered[ai]) else -1),
                ages=tuple(int(x) for x in ages_before),
                costs=tuple(costs_step),
                alive=tuple(bool(x) for x in alive_before),
            ))

        for i in range(n_users):
            d = ep_dist[i]
            if d:
                k = max(1, int(round(len(d) * float(tail_frac))))
                tail[i].append(float(np.mean(d[-k:])))
            else:
                tail[i].append(float("nan"))
            surv[i].append(int(ep_alive[i]))
        budget_used += used_ep
        n_steps += t

    dr = delivery_rate(records, n_users)
    return {
        "records": records,
        "per_user_cost": [float(np.mean(x)) if x else float("nan") for x in tail],
        "per_user_survival": [float(np.mean(x)) if x else float("nan") for x in surv],
        "budget_reserved": int(dr["budget_reserved"]),
        "budget_attempted": int(dr["budget_attempted"]),
        "budget_wasted": int(dr["budget_wasted"]),
        "budget_used": int(budget_used),
        "budget_cap_per_episode": int(budget_cap),
        "n_steps": int(n_steps),
        "n_users": int(n_users),
        "window": int(max_steps),
    }
