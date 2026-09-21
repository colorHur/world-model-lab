"""RL 算法模块（X5 起）。

仓库此前没有任何 RL 算法（`wmlab/{envs,models,rollout,eval,utils}` 全是世界模型侧）。
本目录是"能力线"的落点：X5（PPO）→ X6（用策略采长 episode）→ X7（行为策略消融）。
"""

from .ppo import ActorCritic, Rollout, collect_rollout, evaluate_policy, ppo_update

__all__ = ["ActorCritic", "Rollout", "collect_rollout", "evaluate_policy", "ppo_update"]
