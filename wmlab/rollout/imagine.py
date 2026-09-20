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

★★ 误差口径：两种，别混用（2026-09-20 由 X26 对账发现）
-----------------------------------------------------
| 口径 | 写法 | 回答的问题 |
|---|---|---|
| **累积平均** `nmse` | `mse(pred[:, :h], tgt[:, :h])` | "平均到第 h 步，误差水平是多少" |
| **逐点** `nmse_per_step` | `mse(pred[:, h-1], tgt[:, h-1])` | "**第 h 步本身**有多不准" |

`reliable_horizon` 的定义是"误差**首次**超过阈值的那一步" ⇒ **必须用逐点口径**。
累积口径会先把前 h-1 步的误差摊成均值，因此**穿越阈值更晚** ⇒ H\* 被**系统性抬高**，
极端情况下直接给出 `None`（"在测得视界内始终可靠"），而逐点口径能给出有限值。
实测（Pendulum / CartPole，阈值 NMSE=0.05，评测集 = 训练切分出的短 val 集，
归一化分母 = 切片目标方差）：

| 环境 | 曲线 | 累积口径 H\* | 逐点口径 H\* | 累积的相对偏差 |
|---|---|---|---|---|
| Pendulum | 开环 | 64.83 | **42.99** | **+50.8%** |
| Pendulum | 闭环 | 51.96 | **39.01** | **+33.2%** |
| CartPole | 开环 | `None`（测不出） | **12.95** | —（累积直接测不出） |
| CartPole | 闭环 | 10.11 | **6.96** | **+45.3%** |

（04 的 held-out 长集、global 分母下：闭环 34.85 → **25.57**，即 +36.3%。）

★ 偏差幅度**随评测集与归一化分母而变** ⇒ **任何报出的 H\* 都必须同时声明
「口径（逐点/累积）」「评测集」「归一化分母」三个字段，缺一不可。**

⇒ **X2（2026-09-18）报出的 H\*（开环 64.8 / 闭环 52.0）是累积口径，按『首次超阈』的
准确定义应更正为开环 42.99 / 闭环 39.01。**更正后**"开环 > 闭环"这一方向性结论不变**，
"误差增长饱和"的形态判定也不变，变的是数值。

★ 另有一次**单位混用**的教训（同轮自查发现）：阈值 0.05 是**归一化**口径，
逐点序列必须取 `nmse_per_step` 而非 `mse_per_step`。误用后者会让 H\* 随环境的
目标方差漂移（CartPole 上甚至得到"逐点 > 累积"的反向结果，与"累积=逐点的运行平均"矛盾，
正是靠这个矛盾查出来的）。

本模块**两个口径都返回**，下游必须显式声明用了哪个（对应硬约定 R1）。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from ..eval.metrics import mse, nmse, per_step_mse, per_step_nmse


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
          "horizons":       [...],  # 与下面 mse/nmse 对齐
          "mse":            [...],  # 累积口径：mse(pred[:, :h], tgt[:, :h])
          "nmse":           [...],  # 累积口径，除以该切片的目标方差
          "mse_per_step":   [...],  # ★ 逐点口径，**稠密**，长度 = max(horizons)，索引 i ↔ 第 i+1 步
          "nmse_per_step":  [...],  # ★ 同上，除以该步的目标方差
        }

    ★ 两个口径都返回（对应硬约定 R1：报 H* 必须声明曲线定义）。
      稠密形式的自洽性（冒烟已验证）：mean(mse_per_step[:h]) ≡ mse(h)，对任意 h 成立。
      累积 ≈ 逐点的运行平均，长视界处被前段小误差摊平 ⇒ 会高估 H*。
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
        mses.append(mse(obs_hat[:, :h, :], targets[:, :h, :]))
        nmses.append(nmse(obs_hat[:, :h, :], targets[:, :h, :]))
    return {"horizons": horizons, "mse": mses, "nmse": nmses,
            "mse_per_step": per_step_mse(obs_hat, targets),
            "nmse_per_step": per_step_nmse(obs_hat, targets)}


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

    与开环曲线的差距，**不是**"误差自我放大"的代价 —— 两条曲线语义不同、不能相减
    （第三次自证伪的结论：闭环曲线是"每步重编码"的隐式纠错，不是一个误差放大过程）。
    闭环曲线本身的用途是：它是**在线 tracking 的离线对应物** —— 在线 tracking 里
    "没有新观测时用模型外推"就是这里的闭环机制，所以 X26 的解析对账要拿它比。

    当前用解码后的观测再编码回潜空间（因为潜空间转移训练时只见过真实观测的编码）。
    返回字段与 `multi_step_error_curve` 完全一致（逐点口径同样为**稠密**数组）。
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
    return {"horizons": horizons, "mse": mses, "nmse": nmses,
            "mse_per_step": per_step_mse(obs_hat, targets),
            "nmse_per_step": per_step_nmse(obs_hat, targets)}
