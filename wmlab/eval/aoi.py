"""AoI 分析：由离线 NMSE(h) 曲线**解析地**预测在线跟踪误差（X26）。

这是本通信接入路线上唯一能在 CPU 上几分钟内产出 **② 类证据（数学推导）** 的地方。

核心恒等式
----------
丢包率 p 下，每步独立丢弃，稳态 age 服从几何分布：

    P(age = h) = (1 - p) · p^h ,   h = 0, 1, 2, ...
    E[age]     = p / (1 - p)
    Var[age]   = p / (1 - p)²

周期传输 T 下，age 在 {0, …, T-1} 上均匀：

    P(age = h) = 1 / T
    E[age]     = (T - 1) / 2
    Var[age]   = (T² - 1) / 12

把 **X2 实测的闭环 NMSE(h) 曲线**当作已知函数代入，跟踪误差的期望即可解析求出：

    E[NMSE] = Σ_h P(age = h) · NMSE(ĥ)

★ Jensen 命题（可写进论文的陈述句）
-----------------------------------
`E[NMSE(age)]` 与 `NMSE(E[age])` 的差，完全由 NMSE 曲线在相应区间的**凹凸性**决定：

- 曲线**凸**（二阶差分 > 0，误差加速上升段）⇒ Jensen gap > 0，**用平均 AoI 会低估误差**；
- 曲线**凹**（二阶差分 < 0，饱和段）⇒ Jensen gap < 0，**用平均 AoI 会高估误差**。

⇒ 推论：**平均 AoI 相同的两种调度，落在凸段的那个实际误差更高。**
这条可以用 X25 的仿真直接检验 —— 因为几何分布（重尾、大方差）与均匀分布（有界）
可以在同一个 E[age] 上配起来，而它们的 NMSE 不同。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


# ---------------------------------------------------------------- age 分布
def geometric_age_pmf(p: float, h_max: int) -> np.ndarray:
    """独立丢包下的稳态 age 分布（截断到 [0, h_max] 并重新归一）。

    P(age=h) = (1-p)·p^h。返回长度 h_max+1 的数组。
    """
    p = float(p)
    if not 0.0 <= p < 1.0:
        raise ValueError(f"p 必须落在 [0, 1)，收到 {p}")
    hs = np.arange(h_max + 1, dtype=np.float64)
    pmf = (1.0 - p) * np.power(p, hs)
    return pmf / pmf.sum()


def uniform_age_pmf(period: int) -> np.ndarray:
    """周期传输下的 age 分布：{0, …, T-1} 上均匀。"""
    T = max(1, int(period))
    return np.full(T, 1.0 / T, dtype=np.float64)


def geometric_age_stats(p: float) -> tuple[float, float]:
    """几何分布 age 的 (均值, 方差)。"""
    p = float(p)
    m = p / (1.0 - p)
    v = p / (1.0 - p) ** 2
    return m, v


def uniform_age_stats(period: int) -> tuple[float, float]:
    """周期调度 age 的 (均值, 方差)。"""
    T = max(1, int(period))
    return (T - 1) / 2.0, (T ** 2 - 1) / 12.0


# ---------------------------------------------------------------- 曲线插值
class NmseCurve:
    """由离散实测点构造连续 NMSE(h)（线性插值 + 区间外常数外推）。

    为什么常数外推：X2 已测得 NMSE **饱和**（不是发散），且 NMSE 有自然上界 1
    （= 完全不可预测时等于目标方差）。外推到未测区间时取端点值是最保守的处理。

    ★ `anchor_zero`（2026-09-19 由 X26 对账发现，实测差异最大 94%）
    ---------------------------------------------------------
    实测曲线从 h=1 起步，若直接插值，`NMSE(0)` 会被常数外推填成 `NMSE(1)`。
    **这是错的**：age=0 表示本地刚刚收到真实观测，状态估计 ≡ 真值，**误差恒为 0**。
    不修正会系统性高估短间隔调度的误差 —— 在 P(T=2)（age 在 {0,1} 交替）上实测
    解析预测高出仿真 94%，修正后降到个位数百分比。

    ⇒ 默认在曲线最左端插入物理锚点 `(0, 0)`。

    ★ 长度必须一致（2026-09-20 由 X26 修正时发现的静默 bug）
    ------------------------------------------------------
    `horizons` 与 `nmse` 必须**一一对应**。`np.asarray(nmse)[order]` 在两者长度不同时
    会**静默截断**（不报错），把"第 1..k 步的值"错配到"视界 h₁..h_k"上，结论全错。
    实测踩到的例子：把**稠密**的 `nmse_per_step`（长度 = max(horizons) = 190）
    直接传进来，而 `horizons` 只有 19 个 ⇒ 解析式把 age=100 处的 NMSE 当成第 10 步的值，
    大间隔调度的解析值离谱地小（T=200 时解析/仿真 = 0.016/0.703）。
    ⇒ 这里直接抛错，不让它静默通过。
    """

    def __init__(self, horizons: Sequence[float], nmse: Sequence[float],
                 anchor_zero: bool = True) -> None:
        if len(horizons) != len(nmse):
            raise ValueError(
                f"NmseCurve: horizons 与 nmse 长度不一致（{len(horizons)} vs {len(nmse)}）。"
                "常见原因：把**稠密**的逐点数组（长度 = max(horizons)）与**稀疏**的 "
                "horizons 配了对；请先按 horizons 抽取：`[dense[h-1] for h in horizons]`。"
            )
        order = np.argsort(horizons)
        h = np.asarray(horizons, dtype=np.float64)[order]
        y = np.asarray(nmse, dtype=np.float64)[order]
        if anchor_zero and h[0] > 0.0:
            h = np.concatenate([[0.0], h])
            y = np.concatenate([[0.0], y])
        self.h = h
        self.y = y

    def __call__(self, h):
        return np.interp(np.asarray(h, dtype=np.float64), self.h, self.y)

    def second_difference(self) -> tuple[np.ndarray, np.ndarray]:
        """按等间距重采样后的二阶差分，用于判定凹凸。

        Returns:
            (h_centers, d2) —— d2 > 0 凸（误差加速上升），d2 < 0 凹（趋于饱和）
        """
        hs = np.arange(self.h[0], self.h[-1] + 1e-9, 1.0)
        ys = self(hs)
        d2 = ys[2:] - 2 * ys[1:-1] + ys[:-2]
        return hs[1:-1], d2

    def convex_fraction(self, tol: float = 1e-12) -> float:
        """二阶差分 > 0 的点占比 —— 「有多少比例的视界上误差还在加速上升」。

        ★ 为什么需要 `tol`（数值容差，2026-09-19 由自检测试发现）：
          二阶差分在**严格线性**段上理论值为 0，但浮点运算给的是 ±1e-17 级别的噪声。
          实测踩到的例子：`0.9 - 2×0.5 + 0.1 = 1.4e-17 > 0`，
          于是一条纯直线被判成"有一半的点在加速上升"。
          凹凸是**定性**判断，不该被舍入误差左右 ⇒ 容差按曲线自身的量级取相对值。
        """
        _, d2 = self.second_difference()
        if d2.size == 0:
            return float("nan")
        yscale = max(float(np.max(np.abs(self.y))) if self.y.size else 1.0, 1e-12)
        return float(np.mean(d2 > tol * yscale))


# ---------------------------------------------------------------- 解析期望
def expected_nmse_analytic(curve: NmseCurve, pmf: np.ndarray) -> float:
    """E[NMSE(age)] = Σ_h pmf(h) · NMSE(h)。"""
    hs = np.arange(len(pmf), dtype=np.float64)
    return float(np.dot(pmf, curve(hs)))


def nmse_at_mean_age(curve: NmseCurve, pmf: np.ndarray) -> float:
    """NMSE(E[age]) —— 用平均 AoI 单点估计的误差（工程上常见的偷懒做法）。"""
    hs = np.arange(len(pmf), dtype=np.float64)
    mean_age = float(np.dot(pmf, hs))
    return float(curve(mean_age))


def jensen_gap(curve: NmseCurve, pmf: np.ndarray) -> tuple[float, float]:
    """返回 (E[NMSE(age)], NMSE(E[age]))。

    gap = E - NMSE(E[age])：> 0 说明用平均 AoI 会**低估**误差（曲线凸）。
    """
    e = expected_nmse_analytic(curve, pmf)
    n = nmse_at_mean_age(curve, pmf)
    return e, n


# ---------------------------------------------------------------- 逆问题
def loss_prob_for_mean_age(mean_age: float) -> float:
    """要让平均 age 达到 `mean_age`，需要的丢包率 p = m / (1 + m)。"""
    m = float(mean_age)
    if m < 0:
        raise ValueError("mean_age 必须非负")
    return m / (1.0 + m)


def loss_prob_for_tail_risk(horizon: float, delta: float = 0.01) -> float:
    """容忍 "连续丢失超过 H 步" 的风险 ≤ δ 所需的最大丢包率。

    P(age > H) = p^(H+1) ≤ δ  ⇒  p ≤ δ^(1/(H+1))
    """
    H = float(horizon)
    if H <= 0:
        raise ValueError("horizon 必须为正")
    return float(delta ** (1.0 / (H + 1.0)))
