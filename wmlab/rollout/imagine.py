"""潜空间 rollout，以及"世界模型第几步开始不准"的定量测量。

这是本仓库的核心可验证能力。它对应课题里最重要的一句判断（面试素材）：

    "我知道我的世界模型从第几步开始不可靠，也知道为什么。"

三层递进
--------
1. `imagine()`                —— 给定初始观测与动作序列，用潜空间 rollout 预测未来观测
2. `multi_step_error_curve()` —— 误差随 rollout 步数的增长曲线（本仓库的第一张研究图）
3. `reliable_horizon()`       —— 从曲线上取出"首次超阈"的那一步（在 eval/metrics.py）

注意这里是**开环** rollout：从真实观测出发，之后只吃模型自己的预测。
闭环 rollout（把预测当作下一轮的输入）误差增长更快，
两者之差本身就是"预测该不该做"（D1）的一个直接证据。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from ..eval.metrics import mse, nmse


@torch.no_grad()
def imagine(model, obs0: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """用世界模型在潜空间推演 H 步，返回每步预测的观测。

    Args:
        model: MLPWorldModel
        obs0: (B, obs_dim)
        actions: (B, H)（离散）或 (B, H, act_dim)（连续）
    Returns:
        (B, H, obs_dim) 预测观测序列
    """
    z0 = model.encode(obs0)
    zs = model.rollout_latent(z0, actions)          # (B, H, latent)
    B, H, L = zs.shape
    obs_hat = model.decode(zs.reshape(B * H, L)).reshape(B, H, -1)
    return obs_hat


def _sample_windows(episodes: Sequence[dict], horizon: int, n_samples: int,
                    rng: np.random.Generator):
    """从 episode 里随机截取长度为 horizon 的窗口（保证不跨 episode 边界）。"""
    obs0, acts, targets = [], [], []
    valid = [i for i, ep in enumerate(episodes) if len(ep["obs"]) > horizon + 1]
    if not valid:
        raise ValueError(
            f"没有足够长的 episode 支持 horizon={horizon}；"
            f"最长 episode 长度 = {max((len(e['obs']) for e in episodes), default=0)}"
        )
    for _ in range(n_samples):
        ei = valid[rng.integers(len(valid))]
        ep = episodes[ei]
        T = len(ep["obs"])
        t0 = rng.integers(0, T - horizon - 1)
        obs0.append(ep["obs"][t0])
        acts.append(ep["act"][t0:t0 + horizon])
        targets.append(ep["obs"][t0 + 1:t0 + 1 + horizon])
    return (np.asarray(obs0, dtype=np.float32),
            np.asarray(acts),
            np.asarray(targets, dtype=np.float32))


@torch.no_grad()
def multi_step_error_curve(
    model,
    episodes: Sequence[dict],
    horizons: Sequence[int],
    device: torch.device,
    n_samples: int = 512,
    seed: int = 0,
) -> dict:
    """统计多步预测误差随 rollout 步数的增长。

    一次取最大视界的窗口，做完 rollout 后按各 horizon 切片评估 —— 比逐 horizon 重跑快得多。

    Returns:
        {
          "horizons": [...],
          "mse":   [...],     # 绝对误差
          "nmse":  [...],     # 归一化误差（除以该步的目标方差）
        }
    """
    model.eval()
    horizons = sorted(int(h) for h in horizons)
    h_max = horizons[-1]
    rng = np.random.default_rng(seed)

    obs0_np, acts_np, targets_np = _sample_windows(episodes, h_max, n_samples, rng)
    obs0 = torch.as_tensor(obs0_np, device=device)
    acts = torch.as_tensor(acts_np, device=device)
    targets = torch.as_tensor(targets_np, device=device)

    obs_hat = imagine(model, obs0, acts)              # (B, H, obs_dim)

    mses, nmses = [], []
    for h in horizons:
        pred = obs_hat[:, :h, :]
        tgt = targets[:, :h, :]
        mses.append(mse(pred, tgt))
        nmses.append(nmse(pred, tgt))
    return {"horizons": horizons, "mse": mses, "nmse": nmses}


@torch.no_grad()
def closed_loop_error_curve(
    model,
    episodes: Sequence[dict],
    horizons: Sequence[int],
    device: torch.device,
    n_samples: int = 256,
    seed: int = 0,
) -> dict:
    """闭环 rollout 的误差曲线：把模型自己的预测当作下一步输入。

    与开环曲线的差距，就是"误差自我放大"的代价 —— 这是 D1（该不该预测）的实证依据。
    当前用解码后的观测再编码回潜空间（因为潜空间转移训练时只见过真实观测的编码）。
    """
    model.eval()
    horizons = sorted(int(h) for h in horizons)
    h_max = horizons[-1]
    rng = np.random.default_rng(seed)
    obs0_np, acts_np, targets_np = _sample_windows(episodes, h_max, n_samples, rng)

    obs0 = torch.as_tensor(obs0_np, device=device)
    acts = torch.as_tensor(acts_np, device=device)
    targets = torch.as_tensor(targets_np, device=device)

    preds = []
    cur = obs0
    for t in range(h_max):
        nxt = model.predict_next(cur, acts[:, t])
        preds.append(nxt)
        cur = nxt                                     # ← 闭环：吃自己的预测
    obs_hat = torch.stack(preds, dim=1)

    mses, nmses = [], []
    for h in horizons:
        mses.append(mse(obs_hat[:, :h, :], targets[:, :h, :]))
        nmses.append(nmse(obs_hat[:, :h, :], targets[:, :h, :]))
    return {"horizons": horizons, "mse": mses, "nmse": nmses}
