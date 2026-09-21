"""带**方差头**的世界模型：转移输出 N(μ, diag σ²)。

★ 存在的理由（X22 → X28，这是本文件的全部动机）
------------------------------------------------
X22 核实到 C17（arXiv:2605.15960）的 safe horizon 原文后，得到一个负面结论：

    δ = ½ max_{s,a} ‖T(·|s,a) − T′(·|s,a)‖₁    —— 两个**转移分布**的总变差距离

而本仓库「环境确定（gym 的 step 无噪声）+ 模型确定（`transition` 是 MLP 点估计）」
⇒ 两个转移都是**点质量** ⇒ `TV ∈ {0,1}` ⇒ 取 max 后 **δ ≡ 1**
⇒ `H(ε,1) = 1+ε`（闭式解），定理条件退化成 `γ ≤ ε/(1+ε)`，对任何有意义的 γ 都不成立。

结论是：**在该配置下，"理论界 vs 实测 gap" 这个量不可定义。**
要救活它，必须让 δ 能取 (0,1) 里的连续值，而这需要两侧都是分布：
  - 环境侧 → `envs/noisy_pendulum.py`（过程噪声）
  - 模型侧 → **本文件**（转移输出高斯）

★ 与 `MLPWorldModel` 的关系（刻意做成子类）
------------------------------------------
`encode / decode / rollout_latent / predict_next` 全部继承、一行不改，
**只替换 transition 与 loss**。这样：
  ① `rollout` / `eval` 两层零改动 ⇒ 实测 H\* 的口径与 X2/X3/X4 完全一致，可比；
  ② 历史数字不受影响（`MLPWorldModel` 本身没动）；
  ③ 唯一的语义差别是 `next_latent` 返回**均值**，与确定性模型的返回值同义。

★ 训练目标
----------
    recon : MSE(decode(μ_{t+1}), obs_{t+1})        —— 保证 z 携带重建信息
    nll   : 0.5·mean( (z_target−μ)²/σ² + log σ² )  —— 潜空间高斯的负对数似然

nll 取代了原来的 `MSE(z_next, z_target)`。这一改动是**有代价的**，必须知道：
MSE 只逼 μ 去拟合目标；NLL 同时逼 **σ 去拟合残差的尺度**。
⇒ 训练充分的模型，σ 应当收敛到「真实残差的标准差」——
  这件事本身就是 **B1（不确定性校准）的第一块**，并且是**可检验的**
  （见 scripts/12：σ 预测值 vs 实测残差的对照）。

★ rollout 为什么用均值而不是采样
--------------------------------
`next_latent` 返回 μ，rollout 是确定性的。
这样实测 H\* 与 X2 以来的口径**严格一致**（否则 H\* 里会混进一条采样噪声）。
若要测"随机 rollout"的 H\*，应另开实验并显式声明 —— 不能在同一个 H\* 里混。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .world_model import MLPWorldModel, mlp

LOGVAR_MIN, LOGVAR_MAX = -10.0, 10.0


class GaussianWorldModel(MLPWorldModel):
    """潜空间转移为对角高斯的 world model。"""

    def __init__(self, obs_dim: int, act_dim: int, latent_dim: int = 32,
                 hidden: int = 128, discrete_act: bool = True,
                 logvar_init: float = 0.0) -> None:
        super().__init__(obs_dim, act_dim, latent_dim, hidden, discrete_act)
        # 覆盖父类的确定性转移：共享 trunk + μ/σ 两个头
        # （两个头比"一个 MLP 输出 2L"更稳：σ 不会被 μ 的梯度尺度带偏）
        self.transition = mlp([latent_dim + act_dim, hidden, hidden],
                              activate_last=True)
        self.mu_head = nn.Linear(hidden, latent_dim)
        self.logvar_head = nn.Linear(hidden, latent_dim)
        nn.init.zeros_(self.logvar_head.weight)
        nn.init.constant_(self.logvar_head.bias, float(logvar_init))

    # ---------- 转移分布 ----------
    def _trans_feat(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.transition(torch.cat([z, self.action_features(action)], dim=-1))

    def next_latent_dist(self, z: torch.Tensor, action: torch.Tensor):
        """返回 (μ, σ, log σ²)。σ = exp(0.5·logvar)，已 clamp 防下溢。"""
        h = self._trans_feat(z, action)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h).clamp(LOGVAR_MIN, LOGVAR_MAX)
        return mu, torch.exp(0.5 * logvar), logvar

    def next_latent(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """★ rollout 用**均值**（保证与 MLPWorldModel 的 rollout 口径一致）。"""
        return self.next_latent_dist(z, action)[0]

    # ---------- 损失 ----------
    def loss(self, obs: torch.Tensor, action: torch.Tensor, obs_next: torch.Tensor):
        z = self.encode(obs)
        mu, sd, logvar = self.next_latent_dist(z, action)
        with torch.no_grad():
            z_target = self.encode(obs_next)      # 编码器同步更新，故 detach

        rec = F.mse_loss(self.decode(mu), obs_next)
        # 对角高斯的 NLL（省掉常数项 0.5·log(2π)）
        nll = 0.5 * torch.mean((z_target - mu) ** 2 * torch.exp(-logvar) + logvar)
        total = rec + nll
        return total, {"recon": rec.detach(), "latent": nll.detach(),
                       "sigma_mean": sd.detach().mean()}
