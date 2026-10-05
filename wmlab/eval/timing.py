# -*- coding: utf-8 -*-
"""★ X44（2026-10-04）：**发送时机扰动**的时刻规划库。

把「**固定传输预算、只动发送时刻**」这件构件事抽成可复用模块 —— 与
`pairing.py` / `conformal.py` / `oracle.py` 同属「实验共用的口径构件」。

★★ 为什么必须抽成库函数（而不是留在脚本里）
    X44 的全部结论都建立在**唯一自变量是"时刻"**这一点上，而它依赖
    「同一 T 下所有扰动的发送次数**逐位相等**」。
    写成**纯函数 + 反向断言**（喂乱序 / 越界 / 重复输入必须 raise），
    才能用回归测试锁住；否则"预算被悄悄改掉"会**静默**地把结论变成
    「预算效应冒充时机效应」——而症状是"结论看起来完全合理"。

三个旋钮（都保持恰好 N 次发送）：
  · **phase**  `t_k = Δ + k·T`          确定性整体提前/推后 Δ 步（Δ ∈ [0, T)）
  · **jitter** `t_k = k·T + u_k`        每次提前/推后至多 j 步（有界 ⇒ 不乱序）
  · **random** 窗口 [0, N·T) 内 N 个均匀随机时刻排序（离散度上界）
"""
from __future__ import annotations

import numpy as np

TIMING_MODES = ("phase", "jitter", "random")


def n_tx_budget(period: int, n_steps: int) -> int:
    """★ 同一 T 下**所有扰动共用**的发送次数（预算固定 = 构造上相等）。

    ★★ 时刻从 **1** 起（不是 0）—— `run_closed_loop_control` 里 `t += 1` 在
       `schedule(t, rng)` **之前** ⇒ 调度被查询的时刻是 **1, 2, …, L**，**`t=0` 永不出现**。
       （这是本仓库一个不显眼但致命的时序细节：把网格起点写成 0，第一个发送点会被静默丢掉，
        于是各相位档的次数差 1 ⇒ 预算不等，而症状看起来像"时机有影响"。）
    取最保守的那个（相位 Δ 可到 T−1、抖动可到 ±(T−1)//2）：
        N·T ≤ n_steps − 1   ⇒   N = (n_steps − T) // T
    ⇒ 相位 / 抖动 / 随机的时刻全部落在 [1, N·T] ⊆ [1, n_steps)。
    """
    T = int(period)
    if T < 1:
        raise ValueError(f"period 必须 ≥1，收到 {period}")
    return max(1, (int(n_steps) - T) // T)


def plan_times(mode: str, period: int, param: int, n_steps: int, n_tx: int,
               pattern_seed: int = 0) -> list[int]:
    """生成**恰好 n_tx 个**发送时刻（预算的硬保证就在这里）。见模块头。

    ★ 网格起点为 1（见 `n_tx_budget` 的说明）；返回的时刻满足 `1 ≤ t_k < n_steps`。
    """
    if mode not in TIMING_MODES:
        raise ValueError(f"未知 mode={mode!r}，应为 {TIMING_MODES}")
    T = int(period)
    N = int(n_tx)
    if N < 1:
        raise ValueError(f"n_tx 必须 ≥1，收到 {n_tx}")

    if mode == "phase":
        d = int(param) % T
        ts = [1 + d + k * T for k in range(N)]
    elif mode == "jitter":
        j = int(param)
        if j < 0:
            raise ValueError(f"jitter 必须 ≥0，收到 {j}")
        if j > (T - 1) // 2:
            raise ValueError(
                f"jitter={j} 超过 (T−1)//2={(T - 1) // 2} ⇒ 会乱序（时刻语义被破坏）")
        rng = np.random.default_rng(int(pattern_seed) * 1000003 + T * 101 + j)
        u = rng.integers(-j, j + 1, size=N) if j > 0 else np.zeros(N, dtype=np.int64)
        # ★ 起点抬到 `1 + j`：抖动可以是负的，若从 1 起会把首个时刻推到 ≤0
        #   （实测症状：j=2 时首个时刻 = −1 ⇒ 越界 AssertionError —— 而不是"悄悄少发一次"）
        ts = [1 + j + k * T + int(u[k]) for k in range(N)]
    else:  # random
        rng = np.random.default_rng(int(pattern_seed) * 1000003 + T * 101 + 7)
        ts = sorted(int(x) for x in rng.choice(np.arange(1, N * T + 1),
                                               size=N, replace=False))

    # ---- 预算与合法性硬约束（任一破就抛，不给假数字 —— R14）----
    if len(ts) != N:
        raise AssertionError(f"★ 时刻数 {len(ts)} ≠ 目标预算 {N}（mode={mode}）")
    if len(set(ts)) != N:
        raise AssertionError(f"★ 时刻有重复（mode={mode}）⇒ 预算会被悄悄削掉")
    if any(not (0 <= x < n_steps) for x in ts):
        raise AssertionError(f"★ 时刻越界 [0,{n_steps})（mode={mode}, 最大 {max(ts)}）")
    if any(ts[k + 1] <= ts[k] for k in range(N - 1)):
        raise AssertionError(f"★ 时刻未严格递增（mode={mode}）")
    return ts


def lookup_schedule(times: list[int], name: str):
    """把一组绝对时刻包成 `Schedule`（无状态查表；不需要 `reset`）。"""
    s = set(int(x) for x in times)

    def _f(t: int, rng: np.random.Generator) -> bool:
        return t in s

    _f.__name__ = name
    return _f


def timing_dispersion(times: list[int]) -> float:
    """★ 时刻的**离散度**：相邻间隔的 `std/mean`（CV）。

    这是把三个旋钮放在**同一根轴**上的量：
      · 周期（含任意相位）⇒ **0**（间隔恒为 T）
      · 抖动         ⇒ 随 j 单调上升
      · 完全随机     ⇒ 最大
    ★ 与 E[age] 的区别：E[age] 同时被"预算"和"时机"影响，而 CV(间隔) **只吃时机**
      ⇒ 它才是本实验的**纯净自变量**（预算固定后）。
    """
    ts = np.asarray(sorted(int(x) for x in times), dtype=np.float64)
    if ts.size < 2:
        return 0.0
    gaps = np.diff(ts)
    m = float(gaps.mean())
    return float(gaps.std() / m) if m > 0 else 0.0


def expected_attempts(times: list[int], ep_lens: list[int]) -> int:
    """★ R12 接线检查：由时刻表 + **每集实测长度**反算**应有**的发送次数。

    ★★ 区间是 **`1 ≤ t ≤ L`**，不是 `0 ≤ t < L`：
      `run_closed_loop_control` 里 `t += 1` 发生在 `schedule(t, rng)` **之前**
      ⇒ 调度被查询的时刻是 **1, 2, …, L**，**`t=0` 永不出现**。
      实测代价：第一版写成 `t < L`，基准档反算 101 vs 实测 **93**（8 集各丢一次）
      —— 若没有这条反算检查，这个**差 1 的预算偏差**会被读成"相位改变了发送次数"
      （= 一个不存在的"时机效应"）。

    ★ 为什么不能直接断言 `n_tx == n_ep · N`：闭环 episode 会因**跟丢目标提前终止**，
      每集长度 `L_e` 不同 ⇒ 次数 = `Σ_e #{1 ≤ t_k ≤ L_e}`。
      这不是 bug，是**任务侧效应**本身（调度差 ⇒ 跟丢 ⇒ episode 短）。
    """
    return int(sum(sum(1 for x in times if 1 <= x <= int(L)) for L in ep_lens))


def subsample(lo: int, hi: int, max_n: int) -> list[int]:
    """在 [lo, hi] 上取 ≤ max_n 个等距整数（含两端）。"""
    if hi <= lo:
        return [int(lo)]
    n = min(int(max_n), int(hi) - int(lo) + 1)
    vals = np.unique(np.round(np.linspace(lo, hi, n)).astype(int)).tolist()
    return [int(v) for v in vals]


# ======================================================================== X45
# ★ X45（2026-10-06）：把 X44 的「相位免疫」判据拆开 —— 到底是**真实相位效应**，
#   还是「相位 → 首尾静默分配」×「指标只看最后 25%」的**耦合假象**？
#
# ★★ 为什么不能靠「把窗口做成循环」来修（X44 §8 的建议，经推导**达不到目的**）：
#   设窗口 [1, M]、M = K·T，时刻表是周期 T 的均匀网格与窗口的交集（恰好 K 个点）。
#   则恒有
#        头部静默 (t₁−1) + 尾部静默 (M−t_K) = M − 1 − (K−1)·T = T − 1        (†)
#   与相位**无关**。⇒ 循环化只是把 (T−1, 0) 重新分配给 (0, T−1)，**两者之和不变**；
#   边界效应没被消除，只是从尾部搬到了头部。
#   ⇒ 真正的自变量是**指标口径**：`tail`（最后 25%）吃尾部静默，
#     而 `core`（去掉首段与尾段）与静默无关。见 `scripts/29_*`。
#   ★ 反过来说：只要指标**只取尾部**，相位就与它强耦合 —— 这不是 bug，是**口径**。


def boundary_silence(times: list[int], window: int) -> tuple[int, int]:
    """★ X45：时刻表在窗口 `[1, window]` 里留下的 **(头部静默, 尾部静默)**。

    = `(min(t) − 1, window − max(t))`。

    ★★ 恒等式 (†)（本号的数学基础）：当 `window = K·T` 且 `times` 是周期 T 的
       均匀网格与窗口的交集（恰好 K 个点）时，`head + tail ≡ T − 1`，**与相位无关**。
       ⇒ 「窗口做成循环」**不能**消除边界效应（见模块头）。
    ★ 输入必须严格递增：乱序会让 `min/max` 仍然对、但语义已坏 ⇒ 直接 raise，
      不给"看起来对"的数（R14）。
    """
    ts = [int(x) for x in times]
    if not ts:
        raise ValueError("时刻表为空")
    if any(ts[k + 1] <= ts[k] for k in range(len(ts) - 1)):
        raise ValueError("时刻表必须**严格递增**（乱序 ⇒ 边界语义坏掉，不许静默继续）")
    w = int(window)
    if ts[0] < 1:
        raise ValueError(f"首时刻必须 ≥1（调度首次被查询在 t=1）：{ts[0]}")
    if ts[-1] > w:
        raise ValueError(f"时刻超出窗口：max={ts[-1]} > window={w}")
    return (ts[0] - 1, w - ts[-1])


def cyclic_grid(period: int, window: int, phase: int) -> list[int]:
    """`plan_times` 的**循环版**：周期 T 的网格**平移 phase 后环绕**整个窗口。

    即 `t_k = 1 + ((k·T + phase) mod window)`，取 `k = 0 … window/T − 1`。

    ★ 用途（X45）：**用数值证据坐实「循环化不消除边界效应」** ——
      对任意 phase，`boundary_silence` 都满足 `head + tail = T − 1`（自检 [51]）。
      换言之它**保留了「相位驱动边界分配」这个自由度**，所以不能用来做分离。
    ★ `window` 必须是 `period` 的整数倍：否则各相位的点数不等 ⇒ 预算随相位变
      （那就是 X44 花了大力气封死的混淆变量）。
    """
    T = int(period)
    M = int(window)
    if T < 1:
        raise ValueError(f"period 必须 ≥1，收到 {period}")
    if M <= 0:
        raise ValueError(f"window 必须 ≥1，收到 {window}")
    if M % T != 0:
        raise ValueError(
            f"window={M} 必须是 period={T} 的整数倍（否则各相位的点数不等 ⇒ 预算随相位变）")
    K = M // T
    ph = int(phase) % T
    ts = sorted(1 + ((k * T + ph) % M) for k in range(K))
    if len(set(ts)) != K:
        raise AssertionError(f"★ 循环网格出现重复时刻（T={T}, M={M}, phase={ph}）")
    if any(not (1 <= x <= M) for x in ts):
        raise AssertionError(f"★ 循环网格越界 [1,{M}]：{[min(ts), max(ts)]}")
    return ts


def core_slice(ep_len: int, times: list[int]) -> tuple[int, int] | None:
    """★ X45：**「去边界」口径**在该集上的 0-based 索引区间 `[lo, hi)`。

    定义：保留被测段里 **`[t₂, t_K)`** 这一段 —— 即**丢掉首段** `[t₁, t₂)`
    （含头部静默 `t₁−1`）与**末段** `[t_K, M]`（含尾部静默 `M−t_K`），
    只留**完整中间周期**。

    ★ **不是 `[t₂, t_{K−1})`**：那个写法对只有 3 个时刻的集是**空区间**
      （自检 [52] 首跑就是这么炸的 —— 幸亏写成了断言而不是"看似合理"的数）。
    ★ 索引换算：`ep_dists[e][i]` 对应时刻 `t = i+1`（见 control.py 的记账位置）
      ⇒ 时刻 t ⇔ 索引 t−1。
    ★ 该集**有效时刻 < 3 个**（提前终止太早）⇒ 返回 `None`（**该集不参与** core 口径，
      而不是用短段凑一个数 —— 那会把"集太短"混进"口径差异"）。
    ★ 与 `tail` 口径的分工：`tail`（最后 25%）**吃尾部静默**；`core` 与静默无关。
      两者在同一次运行上比较 ⇒ 可把相位效应归因到边界（X45 的核心手法）。
    """
    ts = [int(x) for x in times if 1 <= int(x) <= int(ep_len)]
    if len(ts) < 3:
        return None
    return (ts[1] - 1, ts[-1] - 1)


def metric_over(traces: list[list[float]], times: list[int], window: int,
                kind: str, tail_frac: float = 0.25,
                margin: int = 0) -> tuple[float, int]:
    """★ X45：从**同一份逐点 trace** 算指定口径的**跨集平均**距离。

    Returns:
        (值, 参与集数)。全部集都不合格 ⇒ `(nan, 0)`（**不抛**，由调用方判定）。

    kind:
      · `"full"` —— 每集全段均值（含首尾静默）
      · `"tail"` —— 每集**最后 `tail_frac`**（**= X44 的 `mean_dist_tail`**，
                    接线检查：它必须与 `run_closed_loop_control` 报的值逐位一致）
      · `"core"` —— 每集**去掉首段与末段**（`core_slice`）；不合格的集**跳过**。
                    ⚠ **位置随相位平移** ⇒ 与 `tail` 的差异里同时含"去边界"与"换位置"两项。
      · `"core_fixed"` —— 每集**去掉固定 `margin` 步**的首尾（`[margin, len−margin)`）。
                    **位置与相位无关** ⇒ 只去掉固定边界，用于**把"位置平移"这个混杂
                    从 `core` 里剥出来**（X45 诊断：若 core_fixed ≪ core 则 core 的
                    偏差主要来自位置平移这个伪影，而不是相位本身）。

    ★ 为什么三口径必须来自**同一次运行**：重跑会因 rng 消耗路径不同得到不同轨迹
      ⇒ 跨运行比较会把"重跑噪声"混进"口径差异"（X44 用 PER=0 封死的同类混淆）。
    """
    if kind not in ("full", "tail", "core", "core_fixed"):
        raise ValueError(f"未知口径 kind={kind!r}")
    if kind == "core_fixed" and int(margin) <= 0:
        raise ValueError("core_fixed 必须给 margin > 0（否则它与 full 逐位相同）")
    times = [int(x) for x in times]
    if times and any(x < 1 or x > int(window) for x in times):
        raise ValueError(
            f"时刻表与窗口不符：window={window}, times∈[{min(times)},{max(times)}]"
            "（⇒ 口径会与静默对不上，不许静默继续）")
    vals: list[float] = []
    for tr in traces:
        if not tr:
            continue
        if kind == "full":
            vals.append(float(np.mean(tr)))
        elif kind == "tail":
            k = max(1, int(round(len(tr) * float(tail_frac))))
            vals.append(float(np.mean(tr[-k:])))
        elif kind == "core":
            sl = core_slice(len(tr), times)
            if sl is None:
                continue
            lo, hi = sl
            hi = min(hi, len(tr))          # 防御：hi 不该超长，超了也不静默取空段
            if hi <= lo:
                continue
            vals.append(float(np.mean(tr[lo:hi])))
        else:                              # core_fixed
            m = int(margin)
            if len(tr) <= 2 * m:
                continue
            vals.append(float(np.mean(tr[m:len(tr) - m])))
    if not vals:
        return (float("nan"), 0)
    return (float(np.mean(vals)), len(vals))
