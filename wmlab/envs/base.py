"""环境适配层的统一接口。

设计动机
--------
世界模型真正依赖的只有一件事：
    (obs_t, action_t) -> (obs_{t+1}, reward_t, done_t)

把这个契约固定下来之后，**换场景（CartPole -> UAV 轨迹 -> 无线信道）
只需要新增一个 Adapter 子类，models / rollout / eval 三层一行都不用改。**
这是本仓库把"无线场景"与"通用世界模型"接起来的关键结构。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class StepResult:
    """统一的一步转移结果。"""

    obs: np.ndarray
    reward: float
    terminated: bool
    truncated: bool

    @property
    def done(self) -> bool:
        return bool(self.terminated or self.truncated)


class EnvAdapter(ABC):
    """所有环境适配器的基类。"""

    #: 人类可读的环境名（会写进实验记录）
    name: str = "env"
    #: 观测维度（扁平化后）
    obs_dim: int = 0
    #: 动作维度；离散动作时表示可选动作个数
    act_dim: int = 0
    #: 动作是否为离散
    is_discrete: bool = True
    #: 连续动作的上下界（仅 is_discrete=False 时有效）
    act_low: np.ndarray | None = None
    act_high: np.ndarray | None = None

    @abstractmethod
    def reset(self, seed: int | None = None) -> np.ndarray:
        """重置环境，返回初始观测。"""

    @abstractmethod
    def step(self, action) -> StepResult:
        """执行一步动作。"""

    @abstractmethod
    def sample_action(self, rng: np.random.Generator | None = None) -> np.ndarray:
        """采样一个随机动作（用于收集 baseline 数据）。"""

    def close(self) -> None:  # pragma: no cover - 多数适配器无需清理
        """释放底层资源。"""

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{type(self).__name__} name={self.name} obs_dim={self.obs_dim} act_dim={self.act_dim}>"
