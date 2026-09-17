"""最小世界模型。

结构（刻意保持最小）：

    obs_t  --E-->  z_t  --T(·, a_t)-->  z_{t+1}  --D-->  obs_{t+1} 的估计
    (编码器)        (潜空间转移)                    (解码器)

训练目标 = 重构误差 + 潜空间一致性
    重构误差：保证 z 确实携带重建观测所需的信息（否则潜空间可以退化成常数）
    潜空间一致性：保证 z 是"可预测的"，而不是每步乱跳
                  —— 这一项是多步 rollout 能成立的前提

为什么现在就固定这个接口
------------------------
B1（混合潜空间的不确定性校准）要做的，正是把 `transition` 换成
"确定性 H_t + 随机 Z_t"的因子化形式，并在 rollout 层度量可靠视界。
本文件已把 `latent_dim / hidden / discrete_act` 与 `next_latent()` 暴露出来，
届时只需替换 transition 与 loss —— rollout / eval 两层无需改动。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(sizes: list[int], activation: type[nn.Module] = nn.SiLU,
        activate_last: bool = False) -> nn.Sequential:
    """按层宽构造 MLP。最后一层默认不加激活（回归/编码都需要线性输出）。"""
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        is_last = i == len(sizes) - 2
        if not is_last or activate_last:
            layers.append(activation())
    return nn.Sequential(*layers)


class MLPWorldModel(nn.Module):
    """潜空间世界模型：编码器 + 转移模型 + 解码器。"""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        latent_dim: int = 32,
        hidden: int = 128,
        discrete_act: bool = True,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.latent_dim = latent_dim
        self.discrete_act = discrete_act

        self.encoder = mlp([obs_dim, hidden, hidden, latent_dim])
        self.decoder = mlp([latent_dim, hidden, hidden, obs_dim])
        self.transition = mlp([latent_dim + act_dim, hidden, hidden, latent_dim])

    # ---------- 基本构件 ----------
    def action_features(self, action: torch.Tensor) -> torch.Tensor:
        """把动作变成网络输入：(B,) 或 (B, act_dim) -> (B, act_dim)。"""
        if action.dim() == 1:
            action = action.unsqueeze(-1)
        if self.discrete_act:
            return F.one_hot(action.squeeze(-1).long(), self.act_dim).float()
        return action.float()

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def next_latent(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.transition(torch.cat([z, self.action_features(action)], dim=-1))

    # ---------- 前向与损失 ----------
    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        """返回 (z_t, z_{t+1}, obs_{t+1} 的估计)。"""
        z = self.encode(obs)
        z_next = self.next_latent(z, action)
        return z, z_next, self.decode(z_next)

    def loss(self, obs: torch.Tensor, action: torch.Tensor, obs_next: torch.Tensor):
        """训练损失。返回 (总损失, 各分项字典) 便于记录曲线。"""
        z, z_next, obs_next_hat = self.forward(obs, action)
        with torch.no_grad():
            z_next_target = self.encode(obs_next)  # 编码器同步更新，故 detach

        rec = F.mse_loss(obs_next_hat, obs_next)
        latent = F.mse_loss(z_next, z_next_target)
        total = rec + latent
        return total, {"recon": rec.detach(), "latent": latent.detach()}

    # ---------- 推理 ----------
    @torch.no_grad()
    def predict_next(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """单步预测：obs_{t+1} 的估计。"""
        z = self.encode(obs)
        z_next = self.next_latent(z, action)
        return self.decode(z_next)

    @torch.no_grad()
    def rollout_latent(self, z0: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """在潜空间里连续推演。

        Args:
            z0: (B, latent_dim) 起始潜状态
            actions: (B, H) 或 (B, H, act_dim) 动作序列
        Returns:
            (B, H, latent_dim) 每步推演出的潜状态
        """
        zs: list[torch.Tensor] = []
        z = z0
        H = actions.shape[1]
        for t in range(H):
            z = self.next_latent(z, actions[:, t])
            zs.append(z)
        return torch.stack(zs, dim=1)
