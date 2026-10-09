"""RL 算法模块（X5 起）。

仓库此前没有任何 RL 算法（`wmlab/{envs,models,rollout,eval,utils}` 全是世界模型侧）。
本目录是"能力线"的落点：X5（PPO）→ X6（用策略采长 episode）→ X7（行为策略消融）
→ ★ X9（连续动作 + 想象训练）。
"""

from .ppo import (ActorCritic, Rollout, clipped_surrogate, collect_rollout,
                  evaluate_policy, ppo_update)
from .continuous import (ContinuousActorCritic, ContinuousRollout,
                         collect_rollout_continuous, compute_gae,
                         continuous_ppo_update, evaluate_policy_continuous,
                         squash_correction)
from .imagination import compute_gae_2d, imagine_batch

__all__ = [
    # X5 / X6 / X7（离散）
    "ActorCritic", "Rollout", "clipped_surrogate", "collect_rollout",
    "evaluate_policy", "ppo_update",
    # X9（连续 + 想象）
    "ContinuousActorCritic", "ContinuousRollout", "squash_correction",
    "continuous_ppo_update", "collect_rollout_continuous",
    "evaluate_policy_continuous", "compute_gae",
    "imagine_batch", "compute_gae_2d",
]
