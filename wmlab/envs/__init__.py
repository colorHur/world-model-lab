"""环境适配层。

换场景的入口：在 `_REGISTRY` 里加一条映射即可，
models / rollout / eval 三层保证不做任何改动。
"""

from __future__ import annotations

from .base import EnvAdapter, StepResult
from .channel import ChannelAdapter, wrap_channel
from .classic import GymClassicAdapter
from .noisy_pendulum import NoisyPendulumAdapter
from .uav_track import UavTrackAdapter

__all__ = [
    "EnvAdapter", "StepResult", "GymClassicAdapter", "ChannelAdapter", "wrap_channel",
    "NoisyPendulumAdapter", "UavTrackAdapter",
    "make_env", "make_env_with_channel",
]

#: 后续接入研究场景时在此注册
_REGISTRY: dict[str, type[EnvAdapter]] = {
    "classic": GymClassicAdapter,
    "gym": GymClassicAdapter,
}


def make_env(env_id: str = "CartPole-v1", seed: int = 0, **kwargs) -> EnvAdapter:
    """按环境 id 构造适配器。

    分派规则（前缀匹配）：
      "uav*"   -> UavTrackAdapter（X14 场景迁移；**不**接收 env_id）
      其他     -> GymClassicAdapter（转发 env_id）
    """
    key = env_id.lower()
    if key.startswith("uav"):
        return UavTrackAdapter(seed=seed, **kwargs)
    return _REGISTRY["classic"](env_id=env_id, seed=seed, **kwargs)


def make_env_with_channel(env_id: str = "CartPole-v1", seed: int = 0,
                          loss_prob: float = 0.0, delay: int = 0,
                          **kwargs) -> EnvAdapter:
    """构造「环境 + 信道」。loss_prob=0 时与 `make_env` 逐位等价（X24 的 E1 检验对象）。"""
    return ChannelAdapter(make_env(env_id, seed=seed, **kwargs),
                          loss_prob=loss_prob, delay=delay, seed=seed)
