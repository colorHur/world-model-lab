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
"""

from __future__ import annotations

from dataclasses import dataclass
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


def lossy_schedule(loss_prob: float) -> Schedule:
    """被动丢包信道：每步独立地以概率 `p` 丢弃。

    ★ 独立性假设（必须在记录里写明）：无记忆伯努利丢包。
    真实无线信道有突发性（burst），需换成 Gilbert–Elliott 后补。
    """
    p = float(loss_prob)

    def _f(t: int, rng: np.random.Generator) -> bool:
        return bool(rng.random() >= p)

    _f.__name__ = f"lossy(p={p})"
    return _f


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
    nmse_by_age: dict         # {age: 该 age 下的 (累积平方误差和, 样本数)} —— X26 用

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
    def update(
        self,
        obs_or_none: np.ndarray | None,
        action: np.ndarray,
        delay: int = 0,
    ) -> tuple[np.ndarray, int, int]:
        """推进一步本地估计。

        Returns:
            (估计值, transmission_age, generation_age)
        """
        if obs_or_none is not None:
            self.est = np.asarray(obs_or_none, dtype=np.float32).reshape(-1).copy()
            self.age_tx = 0
        else:
            o = torch.as_tensor(self.est, device=self.device).unsqueeze(0)
            a = torch.as_tensor(np.asarray(action).reshape(-1), device=self.device).unsqueeze(0)
            self.est = self.model.predict_next(o, a)[0].detach().cpu().numpy().astype(np.float32)
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
) -> TrackingResult:
    """按给定调度跑一遍"符真—预测—跟踪"，返回聚合指标。

    Args:
        episodes: 已采集好的 episode（**必须是 held-out 的**，否则测的是训练集）
        schedule: (t, rng) -> delivered
        max_steps: 每条 episode 最多跑多少步（None = 跑满）
        denom: NMSE 的归一化分母（观测方差）。★ 传入**外部的全局方差**时，
            本函数与离线 `closed_loop_error_curve` 共用同一分母 ⇒ 两者数值严格可比。
            None 时用本批数据自算（与外部口径会差一个常数倍，不可直接并排）。
    """
    model.eval()
    rng = np.random.default_rng(seed)
    tracker = StateTracker(model, device, int(episodes[0]["obs"].shape[1]))

    sq_err_sum = 0.0
    n_elem = 0
    n_tx = 0
    n_steps = 0
    age_sum = 0
    age_gen_sum = 0
    age_max = 0
    err_by_age: dict[int, list] = {}
    true_sq_sum = 0.0
    true_sum = 0.0

    for ep_i, ep in enumerate(episodes):
        tracker.reset(ep["obs"][0])
        T = len(ep["act"])
        if max_steps is not None:
            T = min(T, max_steps)
        for t in range(T):
            delivered = schedule(t, rng)
            a = ep["act"][t]
            o_true = ep["obs"][t + 1]

            if delivered:
                n_tx += 1
                est, a_tx, a_gen = tracker.update(o_true, a, delay)
            else:
                est, a_tx, a_gen = tracker.update(None, a, delay)

            d = est - o_true
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

    mse = sq_err_sum / max(n_elem, 1)
    mean_true = true_sum / max(n_elem, 1)
    var_true = true_sq_sum / max(n_elem, 1) - mean_true ** 2
    scale = float(denom) if denom is not None else max(var_true, 1e-8)
    nmse = mse / max(scale, 1e-8)

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
    )
