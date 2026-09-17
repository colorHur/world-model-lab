"""gymnasium 经典控制任务适配器（CartPole / Pendulum 等）。

这是"把训练循环跑通"用的最小场景，不是研究目标 ——
研究目标场景在 envs/ 下另加 Adapter（UAV 轨迹 / 无线信道）。
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from .base import EnvAdapter, StepResult


class GymClassicAdapter(EnvAdapter):
    """把 gymnasium 的经典控制环境包装成统一接口。"""

    def __init__(self, env_id: str = "CartPole-v1", seed: int = 0, **kwargs):
        self.env = gym.make(env_id, **kwargs)
        self.name = env_id
        self._rng = np.random.default_rng(seed)

        obs_space = self.env.observation_space
        act_space = self.env.action_space

        self.obs_dim = int(np.prod(obs_space.shape))
        self.is_discrete = isinstance(act_space, gym.spaces.Discrete)
        if self.is_discrete:
            self.act_dim = int(act_space.n)
        else:
            self.act_dim = int(np.prod(act_space.shape))
            self.act_low = np.asarray(act_space.low, dtype=np.float32).reshape(-1)
            self.act_high = np.asarray(act_space.high, dtype=np.float32).reshape(-1)

    def reset(self, seed: int | None = None) -> np.ndarray:
        obs, _ = self.env.reset(seed=seed)
        return np.asarray(obs, dtype=np.float32).reshape(-1)

    def step(self, action) -> StepResult:
        if self.is_discrete:
            a = int(np.asarray(action).reshape(-1)[0])
        else:
            a = np.asarray(action, dtype=np.float32).reshape(self.env.action_space.shape)
        obs, reward, terminated, truncated, _ = self.env.step(a)
        return StepResult(
            obs=np.asarray(obs, dtype=np.float32).reshape(-1),
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
        )

    def sample_action(self, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng if rng is not None else self._rng
        if self.is_discrete:
            return np.array([rng.integers(self.act_dim)], dtype=np.int64)
        return rng.uniform(self.act_low, self.act_high).astype(np.float32)

    def close(self) -> None:
        self.env.close()
