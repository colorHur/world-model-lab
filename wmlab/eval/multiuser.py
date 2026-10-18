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
- 未实现：**时变** `csi`（现在只支持"每用户一个常数"，逐 step 衰落未做 ⇒ T11 的 `max-channel` 臂
  在静态 csi 下退化为"恒选最优用户"，不是信道择优）；I1–I6 的逐条自检只做了 I1/I5/I6（`scripts/40`）；
  I2/I3 只有 `scripts/41` 的一条反例；全仓 `pytest` 在本机跑不了（无 pytest）。
- ★ 2026-10-18 20:15 起，「Q4（N 个用户的任务代价怎么合成一个比）未定前 T9 不得开工」**作废**：
  Q4 约束的是**报出来的那个比怎么合成**，T9–T11 的代码不编码任何聚合口径，聚合在离线做。
  边界改为：**T12 跑出任何 `oracle`/`round-robin` 比值之前，Q4 签注框必须已勾**（计划 §3.1）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

__all__ = ["UserSlot", "View", "Allocation", "Allocator", "StepRecord", "delivery_rate",
           "MissingStateError", "SuccessModel", "ConstantSuccess", "SINRSuccess",
           "default_uav_cost", "require_privileged", "run_multiuser"]


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
        costs: 长度 N 的该步任务代价（`alive=False` 的用户填 `nan`）。
        alive: 长度 N 的存活标志（**决策前**的值 —— 与 `ages` 同一时点）。
            ★ 为什么必须与 `ages` 同时点：若记成"步进之后"的值，
              本步刚死掉的用户会被 `delivery_rate` 从分母里剔掉，
              而它本步确实被派过帧 ⇒ 送达率被低估（I6 的直接后果）。
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
    frames = [0] * n_users
    alive_steps = [0] * n_users
    budget_used = 0
    for r in records:
        budget_used += int(r.served_idx >= 0)
        for i in range(n_users):
            if i < len(r.alive) and bool(r.alive[i]):
                alive_steps[i] += 1
                if int(r.delivered_idx) == i:
                    frames[i] += 1
    per_user = [(frames[i] / alive_steps[i]) if alive_steps[i] else float("nan")
                for i in range(n_users)]
    p_eff = [(1.0 - x) if x == x else float("nan") for x in per_user]
    return {
        "per_user": per_user,
        "p_eff": p_eff,
        "budget_used": int(budget_used),
        # ★ 按本函数契约，这里是"总帧位"（= records 数），不是计划的 `budget_total`。
        #   两者只有在 budget_total == max_steps 且无提前终止时相等，别混用。
        "budget_total": len(records),
        "n_users": int(n_users),
        "frames": [int(x) for x in frames],
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


def run_multiuser(
    envs: Sequence,
    controllers: Sequence,
    allocator: Allocator,
    success: SuccessModel,
    *,
    n_episodes: int = 30,
    seed: int = 0,
    max_steps: int = 200,
    warmup_steps: int = 0,
    budget_total: int | None = None,
    cost_fn: Callable[[np.ndarray], float] | None = None,
    tail_frac: float = 0.25,
    csi: float | Sequence[float] | None = None,
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

    ★ `csi` 参数（2026-10-18 T9 新增）：`None` ⇒ 每用户 1.0（与 E1 逐位可比）；
    标量 ⇒ 所有用户同值；长度 N 的序列 ⇒ 逐用户常数。**逐 step 衰落未实现**。
    有了它，`SINRSuccess` 才能让"派给谁"改变本步的成功概率 —— 这是 T9 的交付物本身。

    Returns:
        {"records": [StepRecord…],                 # Gate 2 要的原始表
         "per_user_cost": [均值(逐集尾均)…],        # 与 `mean_dist_tail` 同口径
         "per_user_survival": [均值(存活步数)…],
         "budget_used": int, "budget_cap_per_episode": int,
         "n_steps": int, "n_users": int}
    """
    n_users = len(envs)
    if len(controllers) != n_users:
        raise ValueError(f"envs/controllers 数量不等：{n_users} vs {len(controllers)}")
    cost_fn = cost_fn or default_uav_cost
    budget_cap = int(max_steps if budget_total is None else budget_total)
    if budget_cap < 0:
        raise ValueError("budget_total 必须 ≥ 0（单位 = 帧）")

    rng_ch = np.random.default_rng(int(seed))
    rng_alloc = np.random.default_rng([int(seed), 2 ** 31 - 1])

    # ---------- csi 解析（★ 2026-10-18 T9：把"信道质量"从常量变成每用户的输入）----------
    # None ⇒ 全 1.0（= 之前 E1/冒烟的世界，保持逐位可比）；标量 ⇒ 广播；序列 ⇒ 逐用户。
    # ★ 只支持"每用户一个常数"：逐 step 衰落未实现，别在文档里说成"有时变信道"。
    if csi is None:
        csi_per_user = [1.0] * n_users
    elif np.isscalar(csi):
        csi_per_user = [float(csi)] * n_users
    else:
        csi_per_user = [float(x) for x in csi]
        if len(csi_per_user) != n_users:
            raise ValueError(f"csi 长度必须是 {n_users}，收到 {len(csi_per_user)}")
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
        cost_last = [float("nan")] * n_users
        ep_dist: list[list[float]] = [[] for _ in range(n_users)]
        ep_alive = [0] * n_users

        # ---------- 预热：每步送真值，控制器先收敛到稳态（不消耗任何 rng）----------
        for _ in range(int(warmup_steps)):
            for i in range(n_users):
                if not alive[i]:
                    continue
                res = envs[i].step(controllers[i].act(est[i]))
                est[i] = np.asarray(res.obs, dtype=np.float32).copy()
                if res.terminated or res.truncated:
                    alive[i] = False

        t = 0
        used_ep = 0
        while t < int(max_steps) and any(alive):
            t += 1
            alive_before = list(alive)
            ages_before = list(age)
            slots = tuple(
                UserSlot(idx=i, alive=bool(alive[i]), age=int(age[i]),
                         last_obs=last_obs[i], n_frames=int(n_frames[i]),
                         n_attempts=int(n_attempts[i]), csi=float(csi_per_user[i]),
                         cost_last=float(cost_last[i]))
                for i in range(n_users)
            )
            view = View(t=int(t), slots=slots, budget_left=int(budget_cap - used_ep),
                        budget_total=int(budget_cap), privileged=None)

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
            if np.any((probs < 0.0) | (probs > 1.0)):
                raise ValueError(f"成功概率越界：{probs.tolist()}")

            u = rng_ch.random(n_users)      # ★ 固定抽 n_users 个，见 docstring 纪律 2
            delivered = np.zeros(n_users, dtype=bool)
            si = -1
            if n_served:
                si = int(np.flatnonzero(served)[0])
                n_attempts[si] += 1
                delivered[si] = bool(u[si] >= 1.0 - probs[si])

            costs_step = [float("nan")] * n_users
            for i in range(n_users):
                if not alive_before[i]:
                    continue
                res = envs[i].step(controllers[i].act(est[i]))
                nxt = np.asarray(res.obs, dtype=np.float32)
                c = float(cost_fn(nxt))
                costs_step[i] = c
                ep_dist[i].append(c)
                ep_alive[i] += 1
                cost_last[i] = c
                if delivered[i]:
                    est[i] = nxt.copy()
                    last_obs[i] = nxt.copy()
                    n_frames[i] += 1
                    age[i] = 0
                else:
                    age[i] += 1
                if res.terminated or res.truncated:
                    alive[i] = False

            records.append(StepRecord(
                episode=int(e), t=int(t), served_idx=int(si),
                delivered_idx=int(si if (si >= 0 and delivered[si]) else -1),
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

    return {
        "records": records,
        "per_user_cost": [float(np.mean(x)) if x else float("nan") for x in tail],
        "per_user_survival": [float(np.mean(x)) if x else float("nan") for x in surv],
        "budget_used": int(budget_used),
        "budget_cap_per_episode": int(budget_cap),
        "n_steps": int(n_steps),
        "n_users": int(n_users),
    }
