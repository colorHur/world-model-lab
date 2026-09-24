"""本地状态跟踪器（X24）—— H\* 获得通信含义的地方。

一句话
------
> **H\* 步之内丢包无所谓，超过 H\* 步必须重传。**
> 到这里，H\* 从"模型准不准"变成"**多久必须传一次**"。

机制（与通信里的真实做法同源）
------------------------------
| 本步信道状态 | 本地怎么做                                   | transmission age |
|---|---|---|
| 收到真实观测 | ŝ_t ← o_t（**重置**）                        | 归 0             |
| 丢包         | ŝ_t ← world_model(ŝ_{t-1}, a_{t-1})（**预测顶上**） | +1               |

这等价于 **丢包隐藏（packet loss concealment）**，也是 Kalman 滤波在丢包时的"预测步"。
**不是自造概念。**

★ 关键自洽性（X26 的立论基础）
---------------------------
这里的"预测顶上"用的正是 `model.predict_next`（encode → transition → decode），
与 X2 里那条**观测空间闭环 rollout 曲线**的机制**逐次相同**。

⇒ 于是：**X2 离线测得的闭环 NMSE(h) 曲线，应该能预测这里的在线跟踪误差。**
这条命题可以证伪（X26 就是去验它），一旦成立，
就把"离线指标"与"在线系统性能"接上了 —— 这是整个通信侧论证的地基。

指标口径
--------
NMSE = mean_{t,dim} (ŝ - o)² / Var(o)，**Var 在整批评测观测上 pooled 计算**。
与 `wmlab.eval.nmse` 的分切片口径不同，此处必须用 pooled 口径，
否则不同调度下参与 pooling 的样本集不同，数值不可比。

★ 时延 `delay` 的语义（X27，2026-09-20 补）
----------------------------------------
**修正前的实现是错的（空参数）**：`delay` 只参与 `age_gen = age_tx + delay` 这个算术加法，
**没有任何信息真的被延迟** —— 送达的永远是**当前这一步**的观测。
于是 `generation_age` 只是 `transmission_age` 平移了一个常数，双年龄分解**没有被验证**。

修正后：在飞包用 FIFO 记账（`inflight[arrival_step] = payload`），
- `t` 步成功发送 ⇒ 载荷 `obs[t+1]` 在 **`t + delay` 步到达**（`delay=0` 时当步到达，与旧行为逐位一致）；
- 一步内到达多个包时**取最新生成的那个**（后写覆盖先写）；
- 该步若无到达包 ⇒ 用 `predict_next` 顶上。

★ 由此得到一条可检验的推论（X27 的分析式），**但它有一个前提**：
--------------------------------------------------------------
前提是**收到陈旧载荷就直接当当前状态用**（`delay_mode="naive"`，本仓库的默认）。
这时 `p=0`（每步都送达）下估计值**恰好等于 `obs[t+1-delay]`**，与模型无关 ⇒

    NMSE_naive(d) = E[(o_{t+1} − o_{t+1−d})²] / Var(o) = 2·(1 − ρ_o(d))

其中 `ρ_o(d)` 是**观测过程**的 lag-d 自相关（平稳假设）。**再多的通信也压不掉它。**

★ **但它不是物理下界，只是"不用模型"的代价。** 换 `delay_mode="forward"`，
载荷 `obs[t-d+1]` 会先用动作 `a_{t-d+1}…a_t` 向前推进 `d` 步再落定 ⇒
误差变成**模型的 `d` 步 rollout 误差**，即离线**闭环**曲线在视界 `d` 处的值。
两者之差 = **被白白浪费掉的预测能力**，而且这个差随 `d` 增大而急剧扩大。

**实测（2026-09-20，X27，Pendulum / 8×1000 / 全局分母）：**

| d | naive（=地板） | forward | forward/naive | 离线闭环(d) | forward/离线 |
|---|---|---|---|---|---|
| 1 | 0.023044 | 0.000038 | 0.0016 | 0.000034 | 1.12 |
| 5 | 0.491322 | 0.000729 | 0.0015 | 0.000737 | 0.99 |
| 12 | 1.958980 | 0.006384 | 0.0033 | 0.006774 | 0.94 |
| 30 | 1.567637 | 0.102739 | 0.0655 | 0.101686 | 1.01 |
| 60 | 2.263172 | 0.381736 | 0.1687 | 0.391933 | 0.97 |

两条独立路径都验过：①`naive` 在线仿真 ≡ 数据侧 `2(1-ρ_o(d))`，最大相对偏差 **1.9e-9**；
②`forward` 在线 ≡ 「全起点手工复算 `d` 步闭环 rollout」**逐位一致**。

> ⚠️ **两个必须一起说的前提**：
> **(a) 模型够好。** 用随机初始化的模型跑，`d=1` 时 `forward≈1.006 ≫ naive≈0.021`
> （**糟 47 倍**）—— 把不会预测的模型顶上去，当然不如用陈旧但真实的观测。
> **(b) 离线曲线的采样数要给足。** `n_samples=64` 时离线值自身不稳，会凭空造出
> 一个 5.4 倍的假缺口；`2048` 才收敛（见 `scripts/06_delay_floor.py`）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import torch

Schedule = Callable[[int, np.random.Generator], bool]


# ---------------------------------------------------------------- 调度 / 信道
def periodic_schedule(period: int) -> Schedule:
    """主动周期调度：每 `period` 步传输一次（t % T == 0 ⇒ age 在 0..T-1 循环）。"""
    T = max(1, int(period))

    def _f(t: int, rng: np.random.Generator) -> bool:
        return (t % T) == 0

    _f.__name__ = f"periodic(T={T})"
    return _f


def periodic_age_pmf(loss_prob: float, period: int, k_max: int) -> np.ndarray:
    """★ 周期发送 + 逐包独立丢包下的**平稳年龄分布**（截断到 [0, k_max]）。

        P(age = h) = (s/T) · (1−s)^⌊h/T⌋ ,   s = 1 − p,   h = 0 … k_max

    推导（写在这里，因为它就是本仓库的 ② 类证据）：
      每次尝试独立成功 w.p. s ⇒ 两次到达之间的**尝试次数** K ~ Geom(s)
      ⇒ 到达间隔 D = T·K，E[D] = T/s。平稳更新过程里 P(age=h) = P(D>h)/E[D]，而
      P(D > h) = P(K > h/T) = (1−s)^⌊h/T⌋  ⇒ 上式。
      归一化自检：Σ_h (s/T)·q^⌊h/T⌋ = s·Σ_j q^j = s/(1−q) = 1。
    """
    s = 1.0 - float(min(max(loss_prob, 0.0), 1.0 - 1e-12))
    T = int(max(period, 1))
    K = int(max(k_max, 1))
    h = np.arange(K + 1, dtype=np.int64)
    pmf = (s / float(T)) * ((1.0 - s) ** (h // T))
    tail = max(0.0, 1.0 - float(pmf[:K].sum()))      # h > K 的质量压到 K
    pmf[K] += tail
    return pmf / pmf.sum()


def periodic_mean_age(loss_prob: float, period: int) -> float:
    """E[age] = T·q/s + (T−1)/2（**无截断**的闭式）。

    由 Σ_h h·q^⌊h/T⌋ = T²q/s² + T(T−1)/(2s) 再除以 E[D]=T/s 得到。
    两个必须成立的退化：
      T=1 ⇒ q/s = p/(1−p)        （与 `lossy_schedule` 的 i.i.d. 结果一致）
      s=1 ⇒ (T−1)/2              （无丢包时 age 在 0…T−1 上均匀 —— 周期本身的地板）
    """
    q = float(min(max(loss_prob, 0.0), 1.0 - 1e-12))
    s = 1.0 - q
    T = float(max(period, 1))
    return T * q / s + (T - 1.0) / 2.0


def periodic_age_tail(loss_prob: float, period: int, k_max: int) -> float:
    """★ X34-b2：周期发送年龄分布**超出 [0, k_max] 的尾部质量** P(age > k_max)（精确闭式）。

        P(age > K) = (s/T)·q^{k0}·r + q^{k0+1},
        k0 = (K+1)//T ,  r = (k0+1)·T − (K+1)

    推导：`periodic_age_pmf` 里每一「块」k 覆盖 h ∈ [kT, (k+1)T−1]，块内共 T 个 h、
    每点质量 (s/T)·q^k ⇒ 整块质量 s·q^k。
      · 第 k0 块被 K 截断后只剩 r 个点         ⇒ 贡献 (s/T)·q^{k0}·r
      · 之后所有块 Σ_{k≥k0+1} s·q^k = q^{k0+1} ⇒ 贡献 q^{k0+1}
    两个必须成立的退化：q→0 ⇒ 0；T=1 ⇒ q^{K+1}（与 i.i.d. 几何分布一致）。

    ★★ 为什么单列一条（本仓库第 16 次自证伪的记录）：
    第一版判据写成了 `1 − (1−q)^{K/T}` —— 那是「**至少一次失败**」的概率，不是「**全部失败**」
    的概率（把"至少一次"当成"全部"）。两者**都随 q 增大**，但量级差了约 160 个数量级：
      q=0.019, T=2, K=190 ⇒ 错式 1−0.981^95 = 83.8%   正确式 0.019^95 ≈ 1e−160
    于是第一版把 PER>0 的**全部**点判成"截断"而丢弃，网格里只剩 PER≈0 的退化点
    —— 而那正是本实验最该研究的高丢包区间。
    ⇒ 教训仍是 R12：判据写完后必须拿一个已知答案代进去验量级，不能只看它"跑得动"。
    （顺带：我在写这条注释时一度把方向也写反，是自检 ㉛ 的单调性断言把它抓出来的。）
    """
    q = float(min(max(loss_prob, 0.0), 1.0 - 1e-12))
    s = 1.0 - q
    T = int(max(period, 1))
    K = int(max(k_max, 1))
    n = K + 1
    k0 = n // T
    r = (k0 + 1) * T - n
    return float((s / float(T)) * (q ** k0) * r + q ** (k0 + 1))


def periodic_lossy_schedule(period: int, loss_prob: float) -> Schedule:
    """★ X34 新增：**周期发送 + 逐包独立丢包**（`periodic_schedule` 的带损版本）。

    与 `lossy_schedule` 的差别只在"只在 t % T == 0 时才尝试发送"：
    非发送步不消耗随机数 ⇒ 每次**尝试**独立地以 `p` 丢掉。

    ★★ 为什么必须为它单独写一条年龄分布闭式（见 `scripts/21_period_design.py`）
    周期发送让 age 有了**地板**：即使一个包都不丢，age 也在 0…T−1 上循环，
    E[age] = (T−1)/2 ⇒ 发送周期本身就是一种信息代价，与丢包无关。

        P(age = h) = (s/T)·(1−s)^⌊h/T⌋ ,   s = 1 − p
        E[age]     = T·(1−s)/s + (T−1)/2

    `T=1` 时退化为 `lossy_schedule(p)`（P(age=h) = s·(1−s)^h，E[age] = p/(1−p)），
    这条退化关系在自检 ㉗ 里被逐位检查。
    """
    T = max(1, int(period))
    p = float(min(max(loss_prob, 0.0), 1.0))

    def _f(t: int, rng: np.random.Generator) -> bool:
        if t % T != 0:
            return False
        return bool(rng.random() >= p)

    _f.__name__ = f"periodic_lossy(T={T},p={p})"
    return _f


def lossy_schedule(loss_prob: float) -> Schedule:
    """被动丢包信道：每步独立地以概率 `p` 丢弃。

    ★ 独立性假设（必须在记录里写明）：无记忆伯努利丢包。
    真实无线信道有突发性（burst）—— **X31 用下面的 `GilbertElliottChannel` 补上了**。
    """
    p = float(loss_prob)

    def _f(t: int, rng: np.random.Generator) -> bool:
        return bool(rng.random() >= p)

    _f.__name__ = f"lossy(p={p})"
    return _f


# ================================================================ 突发信道（X31）
class GilbertElliottChannel:
    """★ Gilbert–Elliott 两状态突发信道（X31）。

    Good 状态**全通**、Bad 状态**全丢**（Gilbert 模型的最简形式，也是通信教科书里
    描述突发误码的标准模型）。状态转移概率：

        α = P(Good → Bad)      β = P(Bad → Good)

    ★★ 全部解析量（本类的自检锚点 —— 都不靠仿真估计，有闭式解）
    ------------------------------------------------------------------
    | 量 | 闭式解 | 含义 |
    |---|---|---|
    | 稳态 Bad 概率 | `π_B = α/(α+β)` | **平均丢包率** |
    | 平均突发长度 | `E[L_bad] = 1/β` | 连续丢多少步 |
    | 平均无丢长度 | `E[L_good] = 1/α` | 连续通多少步 |
    | **记忆性** | `ρ₁ = 1 − α − β` | 信道状态的 lag-1 自相关 |

    给定 (平均丢包率 `p̄`, 平均突发长度 `L`) 反解（★ 本类的对外参数化）：

        β = 1/L,      α = p̄·β/(1−p̄) = p̄ / (L·(1−p̄))

    可行域（数学必然，不是 bug）：α ≤ 1 ⇒ **L ≥ p̄/(1−p̄)**。
    丢包率越高，能做出的"最短突发"越长（p̄=0.8 ⇒ L ≥ 4）。

    ★★★ 本实验最反直觉、也最关键的一点
    --------------------------------------------------------------
    "无记忆"（= 退化成 i.i.d. 伯努利）的条件是 **ρ₁ = 0 ⇔ α + β = 1**，
    代入上面的反解 ⇒ **L_iid = 1/(1 − p̄)**，**不是 L = 1**。

    | L 相对 L_iid | ρ₁ | 信道长什么样 |
    |---|---|---|
    | L < L_iid | **< 0** | **负相关** —— 通/丢趋于**交替**，比 i.i.d. 更"规律" |
    | L = L_iid | 0 | 无记忆 ⇒ 与 `lossy_schedule(p̄)` 统计等价 |
    | L > L_iid | **> 0** | **正相关的突发** —— 一丢丢一串 |

    于是扫描 L **连续地穿过**「交替 → 无记忆 → 强突发」三种机制。
    （★ 第一版我默认 L=1 就是 i.i.d.，那是错的：L=1 ⇒ β=1、α=1 ⇒ 确定性交替。）

    ★★ 固定 p̄ 扫 L 时，**E[age] 不是常数**（这一点决定了实验的分析口径）
    ------------------------------------------------------------------
    Bad 游程长度 n ~ Geometric(β)，游程内 age 取 1…n。随机挑一个 Bad 时刻，

        E[age | Bad] = E[n(n+1)/2] / E[n] = 1/β = L
        ⇒ **E[age] = p̄ · L**

    也就是说"突发性"和"平均信息年龄"是**耦合**的：L 变大，age 均值同时变大。

    ★★★ age 的**精确分布**（2026-09-22 推导，X31 的核心结果）
    ------------------------------------------------------------------
    固定 p̄ 时，处于 Good 的概率恒为 1−p̄ ⇒ **P(age = 0) = 1 − p̄ 是常数**。
    而给定 age>0，落在长度 n 的 Bad 游程里的概率 ∝ n·P(n)，游程内 age 取 1…n 均匀：

        P(age = k | age > 0) = P(n ≥ k) / E[n] = β·(1−β)^(k−1)        ← **几何分布，参数 β**

    于是完整分布为：

        **P(age = 0) = 1 − p̄,    P(age = k) = p̄ · β · (1−β)^(k−1),  k ≥ 1**

    ⇒ 两个立刻可用的推论：
      1. **条件年龄分布恒为几何分布，唯一参数就是突发长度 L = 1/β。**
         突发并没有引入"新的分布形状"，只是把条件年龄的均值从 L_iid 拉到 L。
      2. 取 β = 1 − p̄（即 L = L_iid）时条件分布恰是 i.i.d. 丢包的条件分布
         ⇒ **i.i.d. 是「L = L_iid 的 Gilbert 信道」的特例**，两者不是两类东西。

    ⇒★ 因此**跟踪误差有闭式分解**（f = 离线闭环 NMSE 曲线）：

        **E[NMSE] = p̄ · [ f(L) + J(L) ]**
        J(L) = Σ_{k≥1} Geom(k; β)·f(k) − f(L)        （Jensen 项）

    一阶项 f(L)：突发把条件年龄拉长到 L，f 递增 ⇒ 误差变大（**这是主效应**）。
    二阶项 J(L)：纯粹由 f 的**凹凸**决定符号（f 凸 ⇒ J>0 ⇒ 雪上加霜）。
    ⇒ **"突发更糟"是一阶效应，不是 Jensen 高阶效应** —— 这个区分是本实验的答案。
    """

    GOOD, BAD = 0, 1

    def __init__(self, p_loss: float, burst_len: float, seed: int | None = None) -> None:
        pl = float(p_loss)
        L = float(burst_len)
        if not 0.0 < pl < 1.0:
            raise ValueError(f"p_loss 必须落在 (0, 1)，收到 {p_loss}")
        if L < 1.0:
            raise ValueError(f"burst_len 必须 ≥ 1，收到 {burst_len}")
        beta = 1.0 / L
        alpha = pl * beta / (1.0 - pl)
        # ★ 边界处（L = p̄/(1−p̄) ⇒ α 恰为 1）浮点会给出 1.0000000000000002，
        #   直接判 `> 1.0` 会把**恰好可行**的点误杀（p̄=0.8/L=4 就是这样被拒的）。
        if alpha > 1.0 + 1e-9:
            raise ValueError(
                f"参数不可行：p̄={pl}, L={L} ⇒ α={alpha:.4f} > 1。"
                f"（由 π_B = α/(α+β) = p̄ 且 β = 1/L 解得 α = p̄/(L(1−p̄))；"
                f"可行域为 L ≥ p̄/(1−p̄) = {pl / (1.0 - pl):.2f}）")
        alpha = min(alpha, 1.0)
        self.p_loss = pl
        self.burst_len = L
        self.alpha = float(alpha)
        self.beta = float(beta)
        self._rng = None if seed is None else np.random.default_rng(int(seed))
        self.state = self.GOOD
        self.reset()
        self.__name__ = f"GE(p̄={pl:g},L={L:g})"

    # ---------- 解析量（★ 自检用，全部闭式）----------
    @property
    def pi_bad(self) -> float:
        """稳态 Bad 概率 = 平均丢包率。**必须恒等于构造时给的 p_loss**（自检 A1）。"""
        return self.alpha / (self.alpha + self.beta)

    @property
    def mean_burst_len(self) -> float:
        """平均突发长度 1/β。**必须恒等于构造时给的 burst_len**（自检 A2）。"""
        return 1.0 / self.beta

    @property
    def rho1(self) -> float:
        """信道状态 lag-1 自相关 = 1 − α − β。=0 ⇒ 无记忆 ⇒ 退化成 i.i.d.。"""
        return 1.0 - self.alpha - self.beta

    @property
    def l_iid(self) -> float:
        """该丢包率下"无记忆"对应的突发长度 1/(1−p̄)。"""
        return 1.0 / (1.0 - self.p_loss)

    @property
    def expected_age(self) -> float:
        """E[age] = p̄·L（见类 docstring 的推导）—— 用于构造同均值几何对照。"""
        return self.p_loss * self.burst_len

    # ---------- 采样 ----------
    def reset(self, rng: np.random.Generator | None = None) -> None:
        """把信道状态按**稳态分布**重采样。

        ★ 为什么必须按稳态采样（而不是一律置 Good）：一律置 Good 会让每条 episode
        开头都处于"刚从好状态起步"，引入一个**与突发性无关**的瞬态偏置。
        """
        r = rng if rng is not None else self._rng
        if r is None:
            self.state = self.GOOD
        else:
            self.state = self.BAD if r.random() < self.pi_bad else self.GOOD

    def __call__(self, t: int, rng: np.random.Generator) -> bool:
        if self.state == self.GOOD:
            if rng.random() < self.alpha:
                self.state = self.BAD
        else:
            if rng.random() < self.beta:
                self.state = self.GOOD
        return self.state == self.GOOD

    def __repr__(self) -> str:  # pragma: no cover
        return (f"<GilbertElliottChannel p̄={self.p_loss:g} L={self.burst_len:g} "
                f"α={self.alpha:.4f} β={self.beta:.4f} ρ₁={self.rho1:+.4f}>")


# ---------------------------------------------------------------- 信道层自检
def simulate_bad_runs(channel: "GilbertElliottChannel", n_steps: int,
                      rng: np.random.Generator, burn_in: int = 2000) -> np.ndarray:
    """★ 纯信道仿真：抽出 **Bad 游程长度**序列（不需要世界模型，很便宜）。

    ★★ 为什么分布检验要用游程长度、而不是在线的 age 直方图（第 14 次自我修正）
    ------------------------------------------------------------------
    age 序列是**强自相关**的：一个长度为 n 的 Bad 游程里，age 就是 1,2,…,n
    这个**确定性**序列。⇒ 一个游程只贡献 **1 个独立样本**，而不是 n 个。
    把 n_pos 当成样本量，会把检验功效高估约 √L 倍。

    实测症状（X31-b，p̄=0.2/L=16）：用 n_pos=3665 算 KS 临界值 0.0225，
    实测 KS=0.0235 判"显著"；但同一份数据的 TV=0.0278 却**低于**纯噪声期望
    0.0512。两个指标互相矛盾 ⇒ 一定是样本量口径错了。

    游程长度在 Gilbert 链下是 **i.i.d. Geometric(β)** —— 真正独立的样本，
    而且可以廉价地跑几十万步 ⇒ 检验既**严格**又**便宜**。

    ★ 末尾未闭合的游程被丢弃（会轻微低估长游程）；n_steps 足够大时可忽略。
    """
    channel.reset(rng)
    for _ in range(int(burn_in)):
        channel(0, rng)
    runs: list[int] = []
    cur = 0
    for t in range(int(n_steps)):
        if not channel(t, rng):
            cur += 1
        elif cur > 0:
            runs.append(cur)
            cur = 0
    return np.asarray(runs, dtype=np.int64)


def run_length_goodness(runs: np.ndarray, beta: float, alpha: float = 0.05) -> dict:
    """检验游程长度是否 ~ Geometric(β)（**i.i.d. 样本**，KS 临界值可用）。

    同时报均值（闭式 E[n] = 1/β = L），均值检验对长尾比 KS 更敏感。

    ★★★ `alpha` 必须做**多重比较校正**（第 14 次自我修正的续集）
    KS 的临界值是 c(α)/√n，其中 c(α) = √( −0.5·ln(α/2) )：
        α=0.05 ⇒ c=1.36    α=0.0025 ⇒ c=1.83
    一个网格扫 20 个点、每点各做一次 α=0.05 的检验 ⇒ 至少一个误报的概率
    **1 − 0.95^20 = 64%**。实测正是这样：20 点网格跑到第 13 个点时
    KS=0.0127 判"显著"（临界 0.0122），只超 4%，而同一份数据的均值检验
    z=+2.3 完全正常 ⇒ 典型的多重比较假阳性。
    ⇒ 调用方应当传 `alpha = 0.05 / n_tests`（Bonferroni）。
    """
    r = np.asarray(runs, dtype=np.int64)
    n = int(r.size)
    if n < 50:
        return {"n_runs": n, "ok": None, "reason": "样本太少，不判定"}
    b = float(min(max(beta, 1e-12), 1.0))
    kmax = int(max(r.max(), int(np.ceil(3.0 / b)) + 10))
    ks_ = np.arange(1, kmax + 1)
    # Geom(β) 的 CDF（支持集 1,2,…）
    cdf_th = 1.0 - (1.0 - b) ** ks_
    xs = np.sort(r)
    cdf_emp = np.searchsorted(xs, ks_, side="right") / float(n)
    ks = float(np.max(np.abs(cdf_emp - cdf_th)))
    a = float(min(max(alpha, 1e-9), 0.5))
    ks_crit = math.sqrt(-0.5 * math.log(a / 2.0)) / math.sqrt(n)
    mean_emp = float(r.mean())
    mean_th = 1.0 / b
    sd = float(np.sqrt((1.0 - b) / (b * b) / n))       # Geom 均值的标准误
    z = (mean_emp - mean_th) / max(sd, 1e-12)
    return {"n_runs": n, "ks": ks, "ks_crit": ks_crit, "alpha": a,
            "mean_emp": mean_emp, "mean_th": mean_th, "mean_z": float(z),
            "ok": bool(ks <= ks_crit and abs(z) <= 4.0)}


# ---------------------------------------------------------------- 结果容器
@dataclass
class TrackingResult:
    """一次跟踪仿真的聚合结果。"""

    label: str
    n_tx: int                 # 成功传输次数
    n_steps: int              # 总步数
    mse: float
    nmse: float
    age_tx_mean: float        # transmission age 均值
    age_tx_max: int
    age_gen_mean: float       # generation age 均值
    nmse_by_age: dict         # {transmission age: (累积平方误差和, 样本数)} —— X26 用
    #: {generation age: (累积平方误差和, 样本数)} —— X27 用。
    #: `delay=0` 时与 `nmse_by_age` 逐位相同；`delay>0` 时才是"误差真正对应的信息年龄"。
    nmse_by_gen_age: dict
    #: 实际到达的包数（`delay>0` 时，episode 末尾发出的包到不了 ⇒ 小于 `n_tx`）
    n_arrived: int = 0
    #: ★ X31：age 的**完整经验分布** {age: 出现步数}。
    #: 突发信道下 age 不再是无记忆几何分布 —— 这张直方图是"突发性"的唯一可观测量，
    #: 也是把 X26 解析式 `E[NMSE] = Σ_h P_age(h)·NMSE(h)` 从 i.i.d. 推广到任意信道的依据。
    age_hist: dict = field(default_factory=dict)
    age_std: float = float("nan")
    age_p95: float = float("nan")
    #: ★ 同均值几何对照：`Σ_h Geom_{p_eff}(h)·NMSE(h)`，其中几何分布的均值
    #: 被调成与实测 `E[age]` 相同。**把"突发效应"从"年龄变大了"里剥离出来**（见
    #: `GilbertElliottChannel` 类 docstring 的 E[age] = p̄·L 推导）。
    #: 由 `scripts/17` 填；本函数只负责给出 `age_hist`。
    geo_matched_nmse: float | None = None

    @property
    def tx_rate(self) -> float:
        """传输率（越小越省通信）。"""
        return self.n_tx / max(self.n_steps, 1)


# ---------------------------------------------------------------- 跟踪器
class StateTracker:
    """本地状态估计器：丢包时用世界模型 rollout 顶上，同时记账 AoI。"""

    def __init__(self, model, device: torch.device, obs_dim: int) -> None:
        self.model = model
        self.device = device
        self.obs_dim = obs_dim
        self.est: np.ndarray | None = None
        self.age_tx: int = 0

    def reset(self, obs: np.ndarray) -> None:
        self.est = np.asarray(obs, dtype=np.float32).reshape(-1).copy()
        self.age_tx = 0

    @torch.no_grad()
    def _predict(self, action) -> None:
        """推进一步：`est ← world_model(est, action)`。"""
        o = torch.as_tensor(self.est, device=self.device).unsqueeze(0)
        a = torch.as_tensor(np.asarray(action).reshape(-1), device=self.device).unsqueeze(0)
        self.est = self.model.predict_next(o, a)[0].detach().cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def update(
        self,
        obs_or_none: np.ndarray | None,
        action: np.ndarray,
        delay: int = 0,
        replay: Sequence | None = None,
    ) -> tuple[np.ndarray, int, int]:
        """推进一步本地估计。

        Args:
            obs_or_none: 本步**到达**的观测载荷（`None` = 本步没有包到达）。
                ★ 时延 `d>0` 时它是 `d` 步前的观测 —— 见 `replay`。
            action: 本步动作
            delay: 传输时延，只用于算 `generation_age = transmission_age + delay`
            replay: 收到**陈旧**载荷时的"前向补偿"动作序列。

                - `replay=None`（**naive**，默认）：把陈旧载荷**直接当当前状态**。
                  这是不做补偿的朴素做法，误差会出现一个**与模型无关**的下限
                  `2(1-ρ_o(d))`（见模块 docstring）。
                - `replay=[a_s, ..., a_t]`（**forward**）：先用这些动作把载荷向前推到
                  当前时刻再落定 ⇒ 误差变成**模型 `d` 步 rollout 误差**，
                  即离线闭环曲线在视界 `d` 处的值。**这才是"用上世界模型"的做法。**
                  两者之差 = 白白浪费掉的预测能力。

        Returns:
            (估计值, transmission_age, generation_age)
        """
        if obs_or_none is not None:
            self.est = np.asarray(obs_or_none, dtype=np.float32).reshape(-1).copy()
            if replay:
                for a_k in replay:
                    self._predict(a_k)
            self.age_tx = 0
        else:
            self._predict(action)
            self.age_tx += 1
        return self.est, self.age_tx, self.age_tx + int(delay)


# ---------------------------------------------------------------- 主循环
def run_tracking(
    model,
    episodes: Sequence[dict],
    device: torch.device,
    schedule: Schedule,
    seed: int = 0,
    delay: int = 0,
    label: str = "",
    max_steps: int | None = None,
    denom: float | None = None,
    warmup: int = 0,
    delay_mode: str = "naive",
    payload_fn: Callable[[np.ndarray], np.ndarray] | None = None,
) -> TrackingResult:
    """按给定调度跑一遍"符真—预测—跟踪"，返回聚合指标。

    Args:
        episodes: 已采集好的 episode（**必须是 held-out 的**，否则测的是训练集）
        schedule: (t, rng) -> delivered
        delay: 传输时延步数（★ X27 起**真的延迟**：见模块 docstring）
        max_steps: 每条 episode 最多跑多少步（None = 跑满）
        denom: NMSE 的归一化分母（观测方差）。★ 传入**外部的全局方差**时，
            本函数与离线 `closed_loop_error_curve` 共用同一分母 ⇒ 两者数值严格可比。
            None 时用本批数据自算（与外部口径会差一个常数倍，不可直接并排）。
        warmup: 每条 episode 前多少步**只跑不记**。`delay>0` 时开头 `delay` 步里
            还没有任何包到达（在飞），那一段是暂态；不计入统计以免污染均值。
        delay_mode: 收到**陈旧**载荷时怎么办 ——
            `"naive"`（默认）直接当当前状态用；`"forward"` 先用动作把它向前推进到
            当前时刻（即"真的用上世界模型"）。两者的差 = 被浪费掉的预测能力。
            `delay=0` 时两种模式**逐位相同**（没有陈旧可言）。
        payload_fn: ★ X32 新增。发送端对**载荷**的变换（例如均匀量化）。
            None（默认）⇒ 恒等 ⇒ **既有全部实验（X24/X26/X27/X30/X31）逐位不变**。
            这是 X24「信道层零侵入」原则的延续：新能力以可选参数接入，
            不传就等价于没这回事。★ 自检见 tests 第 39 项（R12：改掉它，看输出是否真变）。
    """
    if delay_mode not in ("naive", "forward"):
        raise ValueError(f"delay_mode 必须是 'naive' 或 'forward'，收到 {delay_mode!r}")
    model.eval()
    rng = np.random.default_rng(seed)
    D = int(delay)
    tracker = StateTracker(model, device, int(episodes[0]["obs"].shape[1]))

    sq_err_sum = 0.0
    n_elem = 0
    n_tx = 0
    n_arrived = 0
    n_steps = 0
    age_sum = 0
    age_gen_sum = 0
    age_max = 0
    err_by_age: dict[int, list] = {}
    err_by_gen: dict[int, list] = {}
    #: ★ X31：age 的**步数**直方图（与 err_by_age 的区别：后者记的是元素数，
    #: 要除以 obs_dim 才是步数 —— 单独记一份避免口径混淆）
    age_counts: dict[int, int] = {}
    true_sq_sum = 0.0
    true_sum = 0.0
    #: ★ X31：有状态信道（GilbertElliottChannel）每条 episode 开头要按稳态重采样，
    #: 否则 channel 状态会跨 episode 延续，各 episode 不再独立。
    #: 无状态 schedule（`periodic_schedule` / `lossy_schedule`）没有 `reset` ⇒ 零影响。
    reset_fn = getattr(schedule, "reset", None)

    for ep in episodes:
        tracker.reset(ep["obs"][0])
        if callable(reset_fn):
            reset_fn(rng)
        # ★ 在飞包：arrival_step -> 载荷。**每条 episode 独立**，包不跨 episode 存活。
        inflight: dict[int, np.ndarray] = {}
        T = len(ep["act"])
        if max_steps is not None:
            T = min(T, max_steps)
        for t in range(T):
            # ① 远端本步是否发真值：发了 ⇒ 在 `t + D` 步到达
            if schedule(t, rng):
                # ★ 发送**行为**在预热期照常发生（否则延迟到达的包会在预热结束后
                #   凭空出现），但**计数**只在记账区间内累加 ——
                #   否则 `tx_rate = n_tx/n_steps` 的分子分母不同区间，会被系统性高估。
                #   （X31 实测：L=1 时算出丢包率 0.222 而非 0.5，差 23σ。
                #    X26/X27 因为 warmup=0 从未触发过这个 bug。）
                if t >= warmup:
                    n_tx += 1
                # ★ X32：发送端变换（量化）。None ⇒ 恒等，逐位等价于旧行为。
                payload = ep["obs"][t + 1]
                inflight[t + D] = payload if payload_fn is None else payload_fn(payload)
            # ② 本步到达的包（可能没有）。固定时延下发送与到达一一对应；
            #    若信道升级为变时延，后来的包覆盖先到的 ⇒ 天然"取最新生成的那个"。
            rx = inflight.pop(t, None)
            if rx is not None and t >= warmup:
                n_arrived += 1

            a = ep["act"][t]
            o_true = ep["obs"][t + 1]

            # ③ 陈旧载荷的**前向补偿**（`delay_mode="forward"`）
            #    载荷是 obs[t-D+1]（D=0 时即 obs[t+1]）；用 a_{t-D+1} … a_t 把它推到当前时刻，
            #    应用次数恰好 = D ⇒ 落定后的误差 = 模型的 D 步 rollout 误差。
            replay = None
            if rx is not None and D > 0 and delay_mode == "forward":
                replay = [ep["act"][k] for k in range(t - D + 1, t + 1)]

            est, a_tx, a_gen = tracker.update(rx, a, D, replay=replay)

            d = est - o_true
            if t >= warmup:
                sq_err_sum += float(np.sum(d ** 2))
                n_elem += d.size
                true_sq_sum += float(np.sum(o_true.astype(np.float64) ** 2))
                true_sum += float(np.sum(o_true.astype(np.float64)))
                n_steps += 1
                age_sum += a_tx
                age_gen_sum += a_gen
                age_max = max(age_max, a_tx)
                acc = err_by_age.setdefault(a_tx, [0.0, 0])
                acc[0] += float(np.sum(d ** 2))
                acc[1] += int(d.size)
                accg = err_by_gen.setdefault(a_gen, [0.0, 0])
                accg[0] += float(np.sum(d ** 2))
                accg[1] += int(d.size)
                age_counts[a_tx] = age_counts.get(a_tx, 0) + 1

    mse = sq_err_sum / max(n_elem, 1)
    mean_true = true_sum / max(n_elem, 1)
    var_true = true_sq_sum / max(n_elem, 1) - mean_true ** 2
    scale = float(denom) if denom is not None else max(var_true, 1e-8)
    nmse = mse / max(scale, 1e-8)

    # ★ X31：age 分布的分位数用**完整样本**算（不能对已聚合的直方图再取分位数，
    #   那样是按 age 值加权、不是按步数加权 —— 正好把权重搞反）
    _hist = {k: v for k, v in sorted(age_counts.items())}
    _n = sum(_hist.values())
    if _n:
        _vals = np.repeat(np.asarray(list(_hist.keys()), dtype=float),
                          [int(v) for v in _hist.values()])
        age_std = float(_vals.std())
        age_p95 = float(np.percentile(_vals, 95))
    else:
        age_std, age_p95 = float("nan"), float("nan")

    return TrackingResult(
        label=label,
        n_tx=n_tx,
        n_steps=n_steps,
        mse=mse,
        nmse=nmse,
        age_tx_mean=age_sum / max(n_steps, 1),
        age_tx_max=int(age_max),
        age_gen_mean=age_gen_sum / max(n_steps, 1),
        nmse_by_age={k: v for k, v in sorted(err_by_age.items())},
        nmse_by_gen_age={k: v for k, v in sorted(err_by_gen.items())},
        n_arrived=n_arrived,
        age_hist=_hist,
        age_std=age_std,
        age_p95=age_p95,
    )
