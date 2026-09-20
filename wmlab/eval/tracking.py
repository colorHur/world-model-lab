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
    nmse_by_age: dict         # {transmission age: (累积平方误差和, 样本数)} —— X26 用
    #: {generation age: (累积平方误差和, 样本数)} —— X27 用。
    #: `delay=0` 时与 `nmse_by_age` 逐位相同；`delay>0` 时才是"误差真正对应的信息年龄"。
    nmse_by_gen_age: dict
    #: 实际到达的包数（`delay>0` 时，episode 末尾发出的包到不了 ⇒ 小于 `n_tx`）
    n_arrived: int = 0

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
    true_sq_sum = 0.0
    true_sum = 0.0

    for ep in episodes:
        tracker.reset(ep["obs"][0])
        # ★ 在飞包：arrival_step -> 载荷。**每条 episode 独立**，包不跨 episode 存活。
        inflight: dict[int, np.ndarray] = {}
        T = len(ep["act"])
        if max_steps is not None:
            T = min(T, max_steps)
        for t in range(T):
            # ① 远端本步是否发真值：发了 ⇒ 在 `t + D` 步到达
            if schedule(t, rng):
                n_tx += 1
                inflight[t + D] = ep["obs"][t + 1]
            # ② 本步到达的包（可能没有）。固定时延下发送与到达一一对应；
            #    若信道升级为变时延，后来的包覆盖先到的 ⇒ 天然"取最新生成的那个"。
            rx = inflight.pop(t, None)
            if rx is not None:
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
        nmse_by_gen_age={k: v for k, v in sorted(err_by_gen.items())},
        n_arrived=n_arrived,
    )
