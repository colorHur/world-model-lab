# -*- coding: utf-8 -*-
"""调度器的**后验期望成功率**——R1 第 2 步真正需要改变动作的那一层。

为什么需要这个文件（而不是在策略里现算）
----------------------------------------
`policies.CsiSensitive` 只把 `h_mean - 0.5*h_var` 当**排序键**，它不进入任何
"值 vs 成本"的比较 ⇒ 探测多少次都不改变动作（STATE §3 / B4 判据实测：csi_aoi
64→8 后 `attempted` 与 `cost_true` **逐位相同**）。要造第二个 CSI 消费入口，
必须有一条**连续**的量把信道信念映射到"这次 SEND 期望值不值"。

定义（一次性写定，先于任何收益观测）
------------------------------------
`p_hat_{i,t} = E_{log h ~ N(m_i, v_i)}[ sigmoid(kappa*(log h - log h*)) ]`

· 用 9 点 Gauss–Hermite 求积（`np.polynomial.hermite_e`），吸收 `1/sqrt(2pi)`
  后再做方差缩放：`x = nu * sqrt(v)`，`p ≈ Σ w_nu * sigmoid(kappa*(m + x - c)) / sqrt(2π)`。
· 与 `world.steady_state_success` 同一被积函数、同一归一化 ⇒ 当 `m=0, v=1`
  （公开先验）时两条路必须**逐位一致**；`sched_gates.py` 的 S1 判据就是这个对账
  （不是"看起来差不多"）。
· 纯函数、不改任何状态 ⇒ 调度器读它不越过 `SchedulerView` 的信息边界。

★ 这不是精确贝叶斯。`m_i, v_i` 的更新仍在 `world.py` 里（探测 = 高斯共轭；
  ACK/NACK 若开启则是 `_ack_bump` 的一阶近似）。本文件只做**从信念到期望成功率**
  的映射，不声称"信念本身是后验最优"。
"""

from __future__ import annotations

import numpy as np

from .config import CWConfig

#: 求积点数。★ 2026-10-21 20:5x 实测：9 点在 `m=0.25, v=1` 处与精确积分差
#: **4.37e-4**（Gauss–Hermite 在"宽后验 + sigmoid 饱和"处收敛慢），
#: 80 点与精确积分差 2e-12 ⇒ 决策用 80 点。`sched_gates.S2` 用梯形法则
#: 做独立核对（不是自己跟自己比），容差 1e-9。
NODES9, WEIGHTS9 = np.polynomial.hermite_e.hermegauss(9)
NODES80, WEIGHTS80 = np.polynomial.hermite_e.hermegauss(80)
#: 决策用点数（进 `p_hat_vector`）
NODES_DECISION = 80


def p_succ_expected(m: float, v: float, cfg: CWConfig, *, n80: bool = True) -> float:
    """在 `log h ~ N(m, v)` 下 `sigmoid(kappa*(log h - log h*))` 的期望。

    ★ 默认用 80 点求积（见上面 9 点误差的实测）。`n80=False` 只用于展示
      低阶求积的误差量级，**不得**用于决策。
    """
    k = float(cfg.channel.kappa)
    c = float(cfg.channel.log_h_star)
    nodes, weights = (NODES80, WEIGHTS80) if n80 else (NODES9, WEIGHTS9)
    s = float(np.sqrt(max(0.0, v)))
    z = float(m) + nodes * s
    # 数值稳定 sigmoid
    sig = np.where(z * k - k * c > 0,
                   1.0 / (1.0 + np.exp(-(k * (z - c)))),
                   np.exp(k * (z - c)) / (1.0 + np.exp(k * (z - c))))
    return float(np.sum(weights * sig) / np.sqrt(2.0 * np.pi))


def p_hat_vector(telemetry: list[dict], cfg: CWConfig) -> np.ndarray:
    """逐用户 `p_hat`。`telemetry` 直接来自 `SchedulerView.telemetry`（可部署）。"""
    return np.array([p_succ_expected(float(r["h_mean"]), float(r["h_var"]), cfg)
                     for r in telemetry], dtype=float)
