"""数据采集与切分。

世界模型学的是"环境怎么演化"，与策略优劣无关，因此用**随机策略**采集即可 ——
好处是数据分布固定、结论可复现，也避免把策略学习的好坏混进世界模型的评测里。

后续换成 UAV / 信道场景时，这里的接口不变：只要 Adapter 实现了
reset/step/sample_action，数据采集代码一行都不用改。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .envs.base import EnvAdapter


def collect_random_episodes(
    env: EnvAdapter,
    n_episodes: int = 100,
    seed: int = 0,
    max_steps: int | None = None,
) -> list[dict]:
    """用随机策略跑若干 episode。

    Returns:
        list of {"obs": (T, obs_dim), "act": (T-1,), "reward": (T-1,), "length": int}
    """
    rng = np.random.default_rng(seed)
    episodes: list[dict] = []
    for i in range(n_episodes):
        obs = env.reset(seed=seed + i)
        obs_list = [obs]
        act_list, rew_list = [], []
        t = 0
        while True:
            a = env.sample_action(rng)
            res = env.step(a)
            obs_list.append(res.obs)
            act_list.append(a)
            rew_list.append(res.reward)
            t += 1
            if res.done or (max_steps is not None and t >= max_steps):
                break
        episodes.append({
            "obs": np.asarray(obs_list, dtype=np.float32),
            "act": np.asarray(act_list),
            "reward": np.asarray(rew_list, dtype=np.float32),
            "length": len(act_list),
        })
    return episodes


def split_episodes(episodes: Sequence[dict], val_ratio: float = 0.1, seed: int = 0):
    """按 episode 切分训练/验证集（不按时间步切，避免同一 episode 泄漏）。"""
    idx = np.random.default_rng(seed).permutation(len(episodes))
    n_val = max(1, int(round(len(episodes) * val_ratio)))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    train = [episodes[i] for i in train_idx]
    val = [episodes[i] for i in val_idx]
    return train, val


def transitions_from_episodes(episodes: Sequence[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把 episode 列表摊平成 (obs_t, act_t, obs_{t+1}) 监督样本。"""
    obs, act, nxt = [], [], []
    for ep in episodes:
        o = ep["obs"]
        a = ep["act"]
        if len(a) == 0:
            continue
        obs.append(o[:-1])
        act.append(a)
        nxt.append(o[1:])
    return (np.concatenate(obs, axis=0),
            np.concatenate(act, axis=0),
            np.concatenate(nxt, axis=0))


def episode_stats(episodes: Sequence[dict]) -> dict:
    """episode 层面的统计量（写进实验记录，便于核对数据是否正常）。"""
    lengths = np.asarray([ep["length"] for ep in episodes], dtype=np.float64)
    returns = np.asarray([float(ep["reward"].sum()) for ep in episodes], dtype=np.float64)
    return {
        "n_episodes": int(len(episodes)),
        "total_steps": int(lengths.sum()),
        "length_mean": float(lengths.mean()),
        "length_std": float(lengths.std()),
        "length_max": int(lengths.max()),
        "return_mean": float(returns.mean()),
        "return_std": float(returns.std()),
        "return_max": float(returns.max()),
    }
