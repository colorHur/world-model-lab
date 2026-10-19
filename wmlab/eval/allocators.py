# -*- coding: utf-8 -*-
"""分配臂（T11 / Gate 2 的对照臂）。

为什么单独一个文件
------------------
臂的**规则**是研究口径的一部分（预注册于 `04-实验与代码/2026-10-19-T12-口径修订与四臂预注册-v0.1.md` §4，
前瞻臂见 `2026-10-19-T13-lookahead臂实现口径-预注册-v0.1.md` §2），
必须与 runner 分开：runner 管记账，臂只管"派给谁"。这样换臂不动记账、改记账不动臂。

五个臂：`rr` / `age` / `chan` / `oracle`（myopic 真值）/ `lookahead`（期望 1 步前瞻，2026-10-19 T13 新增）。
2026-10-19 T15 新增第六个：`NormalizedAge`（`age_i/D_i`，**无特权信息**，定位=机制验证臂）。

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

__all__ = ["RoundRobin", "MaxAge", "MaxChannel", "TrueError", "Lookahead", "NormalizedAge",
           "QuotaPacing", "ALIVE_ONLY_NOTE"]


ALIVE_ONLY_NOTE = (
    "四个臂都只在 alive 用户里选（验收 §3-B）；"
    "否则死人份额空转会让各臂实际可用预算不等。"
)


def _pick_alive(view: View, key: Callable[[int], float], *, maximize: bool = True) -> int:
    """在存活用户里选 `key` 最大（`maximize=False` 时最小）者；同分取 idx 小；没人活着返回 -1。

    ★ 唯一的"选人"出口：想加新臂就写 `key`，不要自己写 argmax —— 那样很容易忘掉 alive 过滤，
      而忘掉它的后果（预算空转）在单臂单跑时**看不出来**，只在跨臂比较时变成假收益。

    ★ 2026-10-19 T13：加 `maximize` 开关（`Lookahead` 要的是 argmin）。**同分仍取 idx 小**：
      最小分支用严格 `<`，所以并列时保留先遇到的（idx 更小）。
    """
    best, best_key = -1, (-np.inf if maximize else np.inf)
    for s in view.slots:
        if not s.alive:
            continue
        k = float(key(s.idx))
        if not np.isfinite(k):
            # ★ I3 的延伸：NaN 参与比较会让 argmin 静默吞掉这个候选（nan < x 恒 False），
            #   于是"算不出来"变成"没被选"，与"明确知道它不重要"无法区分。
            raise MissingStateError(f"user {s.idx} 的选择键不是有限值：{k}")
        better = (k > best_key) if maximize else (k < best_key)
        if better:                                      # 严格 ⇒ 同分保留更小的 idx
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
class NormalizedAge:
    """**归一化年龄**：选 `age_i / D_i` 最大者（`D_i` = 该任务的失效窗口）。

    规则（预注册：`04-实验与代码/2026-10-19-T15-归一化年龄-预注册-v0.1.md` §2）::

        score_i = age_i / D_i          （D_i > 0，已知任务参数）
        selected = argmax_{i ∈ alive} score_i      （同分取 idx 小）

    ★★ 它**不是**新算法，别当新算法写（预注册 §0/§7）
      1. `D` 全相等时 `argmax age_i/D ≡ argmax age_i` ⇒ 与 `MaxAge` **逐位相同**。
         若所有任务共用同一个截止期，归一化**不改变任何决策**（这是恒等式，不是近似）。
      2. 归一化只在 `D_i` 逐任务不同时才改变行为 —— 本系统 10 vs 53（5.3×），
         于是它实际作用是**把预算系统性偏向 `D` 小的任务**（CartPole）。
         ⇒ 若它变好，正确结论是"按失效窗口倾斜是对的"，**不是**"归一化年龄更优"。
      3. 实时调度里的 EDF/LLF、AoI 约束调度里的"最接近违约阈值者优先"已占位
         （见 `03-文献与竞品/2026-10-19-T15-deadline与虚拟队列占位检索-v0.1.md`）。

    ★ 信息边界（可部署性）：只读 `slots[i].age` + 自身 `deadlines`，**无特权信息**。
      `D_i` 视为**已知任务元数据**（任务规格书上的"容忍陈旧时长"），
      来源是 `results/46_t14_failure_timescale.json` 的闭环实测（53 / 10 步），
      与任何臂的收益无关 ⇒ 不是用结果调出来的参数。
      `D_i` 若需在线估计，属另一条线，本臂不支持（不许在这里偷偷用 `cost_est` 推）。

    ★ 为什么长度不符抛 `ValueError` 而不是 `MissingStateError`
      这是**接线错误**（构造参数与系统不匹配），不是"运行中缺了一个状态量"。
      `Lookahead.scales` 长度不符抛 `MissingStateError` 是历史遗留的不一致；
      本臂按预注册 §5-S4 的原文抛 `ValueError`，差异是刻意的，不unify以免掩盖。
    """

    deadlines: Sequence[float]
    name: str = "na"
    trace: list | None = None
    """诊断轨迹（默认 None）。给了 list 就逐步 append `{"t","scores","chosen","alive"}`。"""

    def __post_init__(self) -> None:
        d = [float(x) for x in self.deadlines]
        if not d:
            raise ValueError("deadlines 不能为空")
        bad = [(k, v) for k, v in enumerate(d) if (not np.isfinite(v)) or v <= 0.0]
        if bad:
            raise ValueError(f"deadlines 必须是有限正数（单位 = 步），非法项 {bad}")
        self.deadlines = d

    def __call__(self, view: View, rng: np.random.Generator) -> Allocation:
        stop = _exhausted(view)
        if stop is not None:
            return stop
        n = len(view.slots)
        if len(self.deadlines) != n:
            raise ValueError(f"deadlines 长度 {len(self.deadlines)} != N {n}（接线错误）")
        i = _pick_alive(view, lambda j: view.slots[j].age / self.deadlines[j])
        if self.trace is not None:
            self.trace.append({
                "t": int(view.t), "chosen": int(i),
                "scores": {int(s.idx): float(s.age / self.deadlines[s.idx]) for s in view.slots},
                "ages": {int(s.idx): int(s.age) for s in view.slots},
                "alive": [bool(s.alive) for s in view.slots],
            })
        return _alloc(n, i, f"{self.name}:argmax-age-over-D")


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


@dataclass
class Lookahead:
    """**期望 1 步前瞻**（诊断臂）：选让"下一步各用户归一化代价之和"最小的那个用户。

    规则（预注册：`04-实验与代码/2026-10-19-T13-lookahead臂实现口径-预注册-v0.1.md` §2）::

        score(i) = Σ_{j ∈ A} Ĉ[j][i] / s_j        ← 注意索引：`Ĉ[cand][user]`
        lookahead := argmin_{i ∈ A} score(i)      （并列取 idx 小）

    `Ĉ` 由 runner 通过 `View.privileged["true_cost_next"]` 提供（`counterfactual_next_costs`
    算出，已把 `q_i` 的送达期望算进去）。

    ★★ 三条诚实标注（写进论文/汇报，不许省）
    1. 它是 **privileged**：动力学是真实 env 的 clone，不是学出来的模型
       ⇒ 标签是"完美信息 1 步前瞻"，**不是** "model-based"，更**不是** 上界。
    2. 视界只有 1 步 ⇒ 对"后果滞后"的任务（CartPole 掉杆要好几步）先天看不够。
       这正是 H2 要被检验的地方；失败也不奇怪。
    3. 它**不知道**本步骰子（按 `q_i` 取期望），所以不是 clairvoyant：
       信息集里没有"未来"。

    ★ 与 `TrueError` 的差别只有一条：`TrueError` 看**当前**代价，`Lookahead` 看**服务之后**的
      下一步代价。其余（alive 过滤、预算守卫、同分取 idx 小、不消耗 rng）完全一致 ——
      这样两者之差可以归因到"前瞻"这一个变量上。

    ★ `horizon`（2026-10-19 T14）：**不改变本臂的算法**（永远是"对给定矩阵取行和的 argmin"），
      只是声明"我以为矩阵是几步的"。runner 会用 `run_multiuser(cf_horizon=...)` 校验两者一致，
      不一致直接 `ValueError` —— 否则"臂以为在看 20 步、runner 只给 1 步"不会报错，
      只会安静地跑出一个看起来正常的数字（R12 接线事故的标准形态）。
    """

    scales: Sequence[float] | None = None
    name: str = "lookahead"
    horizon: int = 1
    trace: list | None = None
    """诊断轨迹（默认 None = 不记录）。给了 list 就逐步 append `{"t","scores","chosen","gap"}`。

    ★ 为什么需要它：冒烟实测 `lookahead` 的选择与 `max-channel` **逐位相同**（恒选 UAV）。
      这可能是"1 步视界真的没分辨力"，也可能是"矩阵算错了但恰好退化"。
      只有把 `gap = 次优 − 最优` 的实际量级记下来，才能区分两者 ——
      这是**解释性诊断量**，不是新判据（GATE / geo / 种子 / 预算一律不动）。
    """

    def __post_init__(self) -> None:
        if self.scales is not None:
            self.scales = [float(s) for s in self.scales]
            if any((not np.isfinite(s)) or s <= 0.0 for s in self.scales):
                raise ValueError(f"scales 必须是有限正数，收到 {self.scales}")

    def __call__(self, view: View, rng: np.random.Generator) -> Allocation:
        stop = _exhausted(view)
        if stop is not None:
            return stop
        mat = np.asarray(require_privileged(view, "true_cost_next"), dtype=float)
        n = len(view.slots)
        if mat.shape != (n, n):
            raise MissingStateError(f"true_cost_next 形状必须是 ({n},{n})，收到 {mat.shape}")
        if self.scales is not None and len(self.scales) != n:
            raise MissingStateError(f"scales 长度 {len(self.scales)} != N {n}")
        s = [1.0] * n if self.scales is None else self.scales
        score = {}
        for i in range(n):
            if not view.slots[i].alive:
                continue
            total = 0.0
            for j in range(n):
                if not view.slots[j].alive:
                    continue          # 死者不计价（NaN 也在这里被跳过）
                v = float(mat[i, j])
                if not np.isfinite(v):
                    raise MissingStateError(
                        f"true_cost_next[{i}][{j}] 不是有限值（候选 {i} 存活、用户 {j} 存活）")
                total += v / s[j]
            score[i] = total
        if not score:                 # 没有存活候选 ⇒ _pick_alive 返回 -1，与其余臂一致
            return _alloc(n, -1, f"{self.name}:no-alive")
        i = _pick_alive(view, lambda j: score[j], maximize=False)
        if self.trace is not None:
            order = sorted(score.items(), key=lambda kv: kv[1])
            gap = (order[1][1] - order[0][1]) if len(order) > 1 else float("inf")
            self.trace.append({"t": int(view.t), "scores": {int(k): float(v) for k, v in order},
                               "chosen": int(i), "gap": float(gap),
                               "mat": mat.tolist(),
                               "alive": [bool(s.alive) for s in view.slots]})
        return _alloc(n, i, f"{self.name}:argmin-expected-cost-H{int(self.horizon)}")


@dataclass
class QuotaPacing:
    """**配额 + 配速**：把 `quotas[i]` 帧按 `B/L` 的速率**铺满整个窗口**，而不是先到先用完。

    规则（预注册：`04-实验与代码/2026-10-19-T16-配额配速-预注册-v0.1.md` §3）::

        d_i(t) = eff_i(t) · t / L − used_i          # 累计配额赤字（L = window）
        eff_i(t) = quotas[i]                       # reclaim=False（严格 I6）
                 = quotas[i] + dead_q·quotas[i]/Σ_alive quotas   # reclaim=True
        cand   = argmax_{i ∈ alive} d_i(t)           # 同分取 idx 小
        serve  = cand  若 d_cand(t) ≥ 0.5  否则不派
        used_cand += 1

    ★★ 存在的理由（2026-10-19 T16 实测，不是设想）
      现有四个臂一旦 `budget_left > 0` 就每步派帧 ⇒ **第 t 步累计已用 = t** ⇒
      `t = B` 时预算归零，此后窗内剩下的 `L − B` 步**完全断服**。
      B=100 / L=200 时实测：`rr`、`age` 的「最后一次发帧」精确在 t=100，`t>100` 发帧数 **0**
      （2 集 × seed 0，见 T16 预注册 §1）。
      ⇒ 那一档的 `J` 混着"窗口后半段没人服务"，不能直接当"策略能力上限"读。
      本臂就是把这个缺陷单独拿出来当一个变量：**同样的帧数，改成匀速铺开**。

    ★ 三条硬性质
      1. **配额硬上界（解析）**：`d ≥ 0.5` ⇒ `used_i ≤ quotas[i]·t/L − 0.5 < quotas[i]`
         （因 `t ≤ L`）⇒ 只要 `Σ quotas ≤ budget_total`，**不可能**触发 I1 预算违约。
      2. **均匀铺开**：`Σ quotas = B` 时支出速率被 `B/L` 卡住 ⇒ 帧铺满 `[1, L]`。
      3. **不转嫁（沿用 I6，默认）**：`alive=False` 不进候选，其配额**空转**。
         `reclaim=True` 时才按比例收回（理由见 `reclaim` 字段 docstring：
         不收回会把"配速"与"总帧数更少"混成一个变量）。

    ★ 信息边界：只读 `view.t`、`view.budget_left`、`slots[i].alive` 与自身计数器。
      **无特权信息**（不读 `privileged` / `cost_est`）⇒ `diagnostic=False` 可跑。
      ⚠ `View.t` 是公开字段（runner 每步给），不是特权量。

    ★ 诚实标注：**不是新算法主张**。加权轮询 / deficit round robin 是网络调度的标准件；
      本臂的定位是**机制验证**——"配速 vs 前紧后松"这一个变量值多少 `geo`。

    ★ `_used` 的复位：靠"`t` 不递增 ⇒ 新 episode"判定（runner 每集 t 从 1 重置）。
      不用 `t == 1` 单条件，因为**首集首步与后续集首步无法区分**；用 `t <= _last_t` 才唯一。
    """

    quotas: Sequence[int]
    window: int
    name: str = "quota"
    reclaim: bool = False
    """死亡用户的**未用配额**是否按存活用户的配额比例收回（默认 False = 严格 I6 不转嫁）。

    ★ 为什么必须有这个开关（2026-10-19 T16 自检实测，看结果之前决定的）
      严格 I6 下，CartPole 第 ~40 步死掉后它的 50 帧配额**永远空转**，
      实测 B=100 `(50,50)` 每集只花 **54/100** 帧。于是"配速"这个变量会与
      "总共花了更少的帧"混在一起，而后者本身就会让结果变差 —— 归属未定（同 #35 型）。
      开 `reclaim=True` 时把死者的未用配额按比例分给存活用户，
      `Σ eff_i = Σ quotas_i ≤ budget_total` ⇒ **总支出仍被 B 卡死**，只改"给谁"。
      两条硬性质仍成立：`used_i ≤ eff_i`（逐时刻），且 `eff_i` 只随死亡单调增 ⇒ 不会 I1 违约。
    """
    _used: list = field(default_factory=list, init=False, repr=False)
    _last_t: int = field(default=-1, init=False, repr=False)

    def __post_init__(self) -> None:
        q = []
        for k, v in enumerate(self.quotas):
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ValueError(f"quotas[{k}] 必须是整数，收到 {v!r}")
            if float(v) != int(v):
                raise ValueError(f"quotas[{k}] 必须是整数（不许静默截断），收到 {v!r}")
            if int(v) < 0:
                raise ValueError(f"quotas[{k}] 必须 ≥ 0，收到 {v!r}")
            q.append(int(v))
        if not q:
            raise ValueError("quotas 不能为空")
        if int(self.window) <= 0:
            raise ValueError(f"window 必须 > 0，收到 {self.window!r}")
        self.quotas = q
        self.window = int(self.window)

    @property
    def used(self) -> list[int]:
        """本 episode 已派帧数（逐用户）。诊断用，**不要**拿它当判据。"""
        return list(self._used)

    def __call__(self, view: View, rng: np.random.Generator) -> Allocation:
        n = len(view.slots)
        if len(self.quotas) != n:
            raise ValueError(f"quotas 长度 {len(self.quotas)} != N {n}（接线错误）")

        t = int(view.t)
        if not self._used:
            self._used = [0] * n
        elif t <= self._last_t:                 # t 不递增 ⇒ 新一集开始
            self._used = [0] * n
        self._last_t = t
        L = int(self.window)

        if int(view.budget_left) <= 0:
            return _alloc(n, -1, "quota:budget-exhausted")
        if t > L:                               # 窗口外不再花（否则"铺满窗口"的口径失效）
            return _alloc(n, -1, "quota:beyond-window")

        deficit = {}
        alive_idx = [s.idx for s in view.slots if s.alive]
        if not alive_idx:
            return _alloc(n, -1, "quota:no-alive")
        dead_q = sum(q for i, q in enumerate(self.quotas) if i not in alive_idx)
        tot_alive_q = sum(self.quotas[i] for i in alive_idx)
        for i in alive_idx:
            eff = self.quotas[i]
            if self.reclaim and dead_q > 0:
                eff = self.quotas[i] + (dead_q * self.quotas[i] / tot_alive_q
                                        if tot_alive_q > 0 else dead_q / float(len(alive_idx)))
            deficit[i] = eff * (t / float(L)) - self._used[i]

        i = _pick_alive(view, lambda j: deficit[j], maximize=True)
        if not (deficit[i] >= 0.5):             # ← 配速闸门：还没到该发的时点
            return _alloc(n, -1, "quota:pace-hold")
        self._used[i] += 1
        return _alloc(n, i, f"{self.name}:quota-{self.quotas[i]}-used-{self._used[i]}")

