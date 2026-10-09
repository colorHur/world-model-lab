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

    # ---------- ★ X9 新增（2026-10-09）：解析奖励重建 ----------
    @property
    def has_analytic_reward(self) -> bool:
        """本环境是否支持「从观测 + 动作解析重建奖励」。

        只有奖励是 (obs, action) 的**确定性函数**时才为真 —— 这是想象训练的硬前提：
        `MLPWorldModel` 没有 reward 头，想象轨迹的奖励只能从**想象出的观测**反算。
        """
        return self.name.startswith("Pendulum")

    def analytic_reward(self, obs, action):
        """从观测 + 动作重建奖励。形状自适应：(d,) -> float；(B,d) -> (B,)。

        ★ Pendulum 的奖励（本机 gymnasium 源码 `pendulum.py:136` 核实，非二手）：
              costs  = angle_normalize(θ)² + 0.1·θ̇² + 0.001·u²
              reward = −costs
          其中 **u 先经 `np.clip(u, −2, 2)`**（同源 `pendulum.py:134`）。
        ★ 观测是 `[cos θ, sin θ, θ̇]`（`pendulum.py:171`）⇒ θ 只能由 `atan2` 反解。
          见模块级 `_pendulum_reward_from_obs` 的说明。
        """
        if not self.has_analytic_reward:
            raise NotImplementedError(
                f"{self.name} 的奖励不是 (obs, action) 的确定性函数，"
                f"不支持解析重建；想象训练在此环境上不适用（CartPole 因奖励恒为 1，"
                f"想象轨迹之间无可优化差异）。"
            )
        return _pendulum_reward_from_obs(obs, action)


def _pendulum_reward_from_obs(obs, action):
    """Pendulum-v1 的解析奖励重建。**纯函数**（自检 [57] 独立验证它的正确性）。

    ★ 为什么 θ 能反解：观测给出 (cos θ, sin θ)，`atan2(sin θ, cos θ) ∈ [−π, π]`，
      而 `angle_normalize` 的值域**恰好**就是 [−π, π] ⇒ 反解出的角度与
      `angle_normalize(真实 θ)` **在所有情况下相等**（θ 的 2π 周期性被吸收掉了）。

    ★ 隐式代价（写进 X9 记录）：`atan2` 只用方向，**丢弃半径**（cos²+sin² 是否为 1）。
      对真实观测无害；对**模型预测的**观测则是一次隐式归一化 ⇒
      预测观测整体偏大/偏小不影响奖励，只要**方向角**对。这是有意选择。
    """
    obs = np.asarray(obs, dtype=np.float64)
    action = np.asarray(action, dtype=np.float64)
    single = (obs.ndim == 1)
    if single:
        obs = obs[None, :]
    # 动作三种形状都要吃：标量 (0维) / (A,) / (B, A)
    if action.ndim == 0:
        action = action.reshape(1, 1)
    elif action.ndim == 1:
        action = action.reshape(1, -1)
    th = np.arctan2(obs[:, 1], obs[:, 0])
    th_norm = ((th + np.pi) % (2.0 * np.pi)) - np.pi
    thdot = obs[:, 2]
    u = np.clip(action[:, 0], -2.0, 2.0)
    r = -(th_norm ** 2 + 0.1 * thdot ** 2 + 0.001 * u ** 2)
    return float(r[0]) if single else r
