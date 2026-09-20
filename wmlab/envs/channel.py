"""有损信道适配层（X24）。

把"通信"接入本仓库的最小装置 —— **不换动力学、不改世界模型**，
只在「远端被控对象」和「本地状态估计」之间插一层可达性判定：

    远端 env  --(每步以概率 p 丢弃)-->  本地 StateTracker
                    (时延 d 由 run_tracking 的在飞包 FIFO 实现)

设计要点
--------
1. **底层动力学一个字节都不动。** `step()` 直接返回底层 `StepResult` 的内容，
   只往 `info` 里追加 `delivered` / `delay`。
   ⇒ `loss_prob=0` 时 ChannelAdapter 与裸 env **逐位一致**（这是 X24 的等价性检验 E1）。
2. **信道有独立的 rng**（`seed + 7919`）。丢包不消耗底层 env 的动作采样 rng，
   保证"加信道"这件事**不会改变被采样到的轨迹本身**，只改变"哪些步能看见"。
   ⇒ 这是把通信从动力学里彻底剥离的关键，也是复现性前提。
3. **观测值本身不因信道而失真**（不量化、不加噪）。
   信道只决定"这一步是否被本地看见"。
   ★ 观测的**陈旧性**由两层承担：本模块记录 `delay` 参数，
   真正的"延迟到达"由 `wmlab.eval.tracking.run_tracking` 的**在飞包 FIFO** 实现
   （X27，2026-09-20 修）。**修正前 `delay` 只是算术偏移、不延迟任何信息**，
   详见 `tracking.py` 模块 docstring。

AoI 两个分量（直接对应讲座①吴泳澎的拆法）
-----------------------------------
- `transmission_age`：自上一次成功**接收**起的步数 → **传输信息年龄**
- `generation_age = transmission_age + delay`  → **生成信息年龄**
  （`delay` 真的延迟到达之后，这条式子才第一次有了内容 —— X27 已验证：
  `p=0` 时 `E[transmission_age] ≡ 0` 而 `E[generation_age] ≡ delay`。）

未接信道时 `StepResult.info is None`，`delivered` 恒为 True —— 完美观测。
"""

from __future__ import annotations

import numpy as np

from .base import EnvAdapter, StepResult


class ChannelAdapter(EnvAdapter):
    """把任意 EnvAdapter 包一层有损信道。底层动力学零改动。"""

    #: 信道 rng 相对环境 seed 的偏移（避免与动力学共用随机流）
    _CH_SEED_OFFSET = 7919

    def __init__(
        self,
        base: EnvAdapter,
        loss_prob: float = 0.0,
        delay: int = 0,
        seed: int = 0,
        name: str | None = None,
    ) -> None:
        if not 0.0 <= loss_prob < 1.0:
            raise ValueError(f"loss_prob 必须落在 [0, 1)，收到 {loss_prob}")
        if delay < 0:
            raise ValueError(f"delay 必须非负，收到 {delay}")

        self.base = base
        self.loss_prob = float(loss_prob)
        self.delay = int(delay)
        self.seed = int(seed)
        self._rng = np.random.default_rng(seed + self._CH_SEED_OFFSET)

        # 透传底层能力描述（让上层代码不需要知道是否被包过信道）
        self.name = name or f"{base.name}+ch(p={loss_prob},d={delay})"
        self.obs_dim = base.obs_dim
        self.act_dim = base.act_dim
        self.is_discrete = base.is_discrete
        self.act_low = base.act_low
        self.act_high = base.act_high

        #: 上一步的信道记账（无信道时为 {"delivered": True, "delay": 0}）
        self.last_info: dict = {"delivered": True, "delay": 0}

    # ---------- 信道行为 ----------
    def _draw_delivered(self) -> bool:
        """本步是否送达。loss_prob=0 时恒为 True（且不消耗 rng，保证完全等价）。"""
        if self.loss_prob == 0.0:
            return True
        return bool(self._rng.random() >= self.loss_prob)

    # ---------- EnvAdapter 接口 ----------
    def reset(self, seed: int | None = None) -> np.ndarray:
        obs = self.base.reset(seed=seed)
        self.last_info = {"delivered": True, "delay": self.delay}
        return obs

    def step(self, action) -> StepResult:
        res = self.base.step(action)
        delivered = self._draw_delivered()
        self.last_info = {
            "delivered": delivered,
            "delay": self.delay,
            "loss_prob": self.loss_prob,
        }
        # ★ 返回值内容与底层完全一致（含 reward / terminated），只多带一个 info
        return StepResult(
            obs=res.obs,
            reward=res.reward,
            terminated=res.terminated,
            truncated=res.truncated,
            info=self.last_info,
        )

    def sample_action(self, rng: np.random.Generator | None = None) -> np.ndarray:
        """动作采样**直接透传给底层** —— 信道不得干预被采样到的动力学样本。"""
        return self.base.sample_action(rng)

    def close(self) -> None:
        self.base.close()

    def __repr__(self) -> str:  # pragma: no cover
        return (f"<{type(self).__name__} base={self.base.name} "
                f"loss_prob={self.loss_prob} delay={self.delay}>")


def wrap_channel(
    env: EnvAdapter,
    loss_prob: float = 0.0,
    delay: int = 0,
    seed: int = 0,
) -> EnvAdapter:
    """给环境套一层信道。loss_prob=0 且 delay=0 时等价于原环境（逐位一致）。"""
    return ChannelAdapter(env, loss_prob=loss_prob, delay=delay, seed=seed)
