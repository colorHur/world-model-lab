"""环境适配层。

换场景的入口：在 `_REGISTRY` 里加一条映射即可，
models / rollout / eval 三层保证不做任何改动。
"""

from __future__ import annotations

from .base import EnvAdapter, StepResult
from .classic import GymClassicAdapter

__all__ = ["EnvAdapter", "StepResult", "GymClassicAdapter", "make_env"]

#: 后续接入研究场景时在此注册，例如 "uav": UAVTrajectoryAdapter, "channel": WirelessChannelAdapter
_REGISTRY: dict[str, type[EnvAdapter]] = {
    "classic": GymClassicAdapter,
    "gym": GymClassicAdapter,
}


def make_env(env_id: str = "CartPole-v1", seed: int = 0, **kwargs) -> EnvAdapter:
    """按环境 id 构造适配器。

    目前所有 gymnasium 经典控制任务都走 GymClassicAdapter；
    一旦 `uav-*` / `channel-*` 这类 id 出现，就在这里分派到专用 Adapter。
    """
    key = "classic"
    if env_id.lower().startswith(("uav", "channel", "wireless")):
        raise KeyError(
            f"环境 '{env_id}' 的适配器尚未实现。"
            f"请在 wmlab/envs/ 下新增 Adapter 并在 _REGISTRY 注册 —— "
            f"这正是本仓库为'换场景'预留的扩展点。"
        )
    return _REGISTRY[key](env_id=env_id, seed=seed, **kwargs)
