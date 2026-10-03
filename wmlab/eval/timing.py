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
