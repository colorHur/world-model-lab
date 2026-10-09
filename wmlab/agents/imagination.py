"""想象训练：在世界模型的**潜空间**里 rollout，并反传策略梯度（Dreamer 系最小闭环）。

对应 **X9 / E2**。判据是「**真环境**策略回报曲线上升」—— 不是"想象回报上升"，
理由见下。

★ 为什么主判据必须是真环境回报（本实验最核心的一句）
----------------------------------------------------
策略是在世界模型里优化的 ⇒ 它**必然**会去利用世界模型的误差
（model exploitation：模型说那里奖励高，策略就往那里走，而真实环境可能完全相反）。
⇒ "想象回报上升"**不能**作为成功证据，它甚至可能是**失败**的症状。
  这正是判据 H3（报出想象回报与真环境回报的 gap）存在的理由。

★ 三个容易错的地方（代码里都有标记）
------------------------------------
1. **奖励用「当前」z 的 decode，不是下一步的**。
   与 gym 的 `step` 语义对齐：`PendulumEnv.step(u)` 在内部用 **step 之前**的 (θ, θ̇)
   计算 `costs`，返回的却是**新**观测（本机源码 `pendulum.py:127,136,147` 核实）。
2. **全程不 re-encode**：`z_{t+1} = next_latent(z_t, a_t)`，纯潜空间滚动 ——
   这就是"想象"的定义（不再看真实观测）。decode 只用于取奖励。
3. **末端必须 bootstrap `V(z_H)`**：Pendulum **永不 terminated**（只有 TimeLimit 截断），
   所以没有任何一步可以当作"此后无回报"（X5 推导 (4) 的 ★）。
   ⇒ 本模块的 GAE **不带 terminated 掩码**，前提是 `horizon < 200`（有断言卡住）。

★ 依赖注入：`reward_fn`
-----------------------
世界模型没有 reward 头（`MLPWorldModel.loss` 只有 recon + latent）。
奖励由调用方注入，签名 `reward_fn(obs: (B, obs_dim) np, a: (B, act_dim) np) -> (B,) np`。
**本模块不提供默认值** —— 缺了必须报错，不允许"悄悄用 0 当奖励"跑出一张好看的曲线。
"""

from __future__ import annotations

import numpy as np
import torch

from .continuous import ContinuousRollout

MAX_TRUNCATION_STEPS = 200      # Pendulum-v1 的 TimeLimit


# --------------------------------------------------------------------------- GAE (2D)
def compute_gae_2d(reward: np.ndarray, value: np.ndarray, last_value: np.ndarray,
                   gamma: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """按「每条想象轨迹独立」计算的 GAE(γ, λ)。

    Args:
        reward, value: (B, H)
        last_value:    (B,)  —— V(z_H)，末端 bootstrap 项
    Returns:
        adv (B, H)（**未归一化** —— 归一化在展平之后做，与 ppo.py 同口径）、
        ret (B, H) = adv + value

    ★ 与 `criteria.compute_gae`（1D 版）的关系：本函数是它在 H 维上的向量化，
      语义逐条一致。选择 2D 是因为想象 rollout 天然是 (B, H) 形状，
      逐条循环 B 次会让 noise 掩盖形状错误。

    ★ 无 terminated 掩码：见模块 docstring 第 3 条。
    """
    B, H = reward.shape
    adv = np.zeros((B, H), dtype=np.float32)
    last = np.zeros(B, dtype=np.float32)
    next_v = np.asarray(last_value, dtype=np.float32).reshape(B)
    for t in reversed(range(H)):
        delta = reward[:, t] + gamma * next_v - value[:, t]
        last = delta + gamma * lam * last
        adv[:, t] = last
        next_v = value[:, t]
    return adv, adv + value


# --------------------------------------------------------------------------- rollout
@torch.no_grad()
def imagine_batch(model, actor, obs_pool: np.ndarray, n_starts: int, horizon: int,
                  reward_fn, device: torch.device, rng: np.random.Generator,
                  gamma: float, lam: float) -> tuple[ContinuousRollout, dict]:
    """从真实观测池随机取 B 个起点，各在潜空间展开 `horizon` 步。

    ★ 起点来自**真实数据**（`model.encode(真实的 obs0)`），之后完全由模型自己滚动。
      这是 Dreamer 的标准做法：起点分布对，之后的漂移是模型自己的事（→ H3 的 gap）。

    Returns:
        (ContinuousRollout 已展平为 (B*H, ...), info)
        info 含 `imag_return`（本批想象回报均值）、`z_start_std`、`z_end_std`
        —— 后两个用于量化"训练时的 z 分布 vs 评估时 encoder(真实 obs) 的 z 分布"
        这个**表征漂移**（X9 记录 §0.2 决策 B）。
    """
    if horizon >= MAX_TRUNCATION_STEPS:
        raise ValueError(
            f"horizon={horizon} >= Pendulum 的截断步数 {MAX_TRUNCATION_STEPS}："
            f"此时'永不 terminate'的假设不再成立，GAE 必须带掩码。"
        )
    obs_pool = np.asarray(obs_pool, dtype=np.float32)
    idx = rng.integers(0, obs_pool.shape[0], size=int(n_starts))
    obs0 = torch.as_tensor(obs_pool[idx], device=device)

    z = model.encode(obs0)                                   # (B, latent)
    z_start_std = float(z.std().item())

    zs, us, aas, logps, vals, rews = [], [], [], [], [], []
    for _ in range(int(horizon)):
        out = actor.sample(z)                                # u,a,logp,value,ent
        # ★ 易错点 1：用**当前** z 的 decode 取奖励（与 gym.step 的语义对齐）
        obs_cur = model.decode(z)
        r = reward_fn(obs_cur.detach().cpu().numpy(), out["a"].detach().cpu().numpy())

        zs.append(z.detach().cpu().numpy())
        us.append(out["u"].detach().cpu().numpy())
        aas.append(out["a"].detach().cpu().numpy())
        logps.append(out["logp"].detach().cpu().numpy())
        vals.append(out["value"].detach().cpu().numpy())
        rews.append(np.asarray(r, dtype=np.float32))

        # ★ 易错点 2：不 re-encode，直接潜空间滚动
        z = model.next_latent(z, out["a"])

    z_end_std = float(z.std().item())
    last_v = actor.forward(z)[2].detach().cpu().numpy()       # ★ 易错点 3：末端 bootstrap

    stack = lambda L: np.stack(L, axis=1).astype(np.float32)   # (B, H, ...)
    Z, U, A = stack(zs), stack(us), stack(aas)
    LP, V, R = stack(logps), stack(vals), np.stack(rews, axis=1).astype(np.float32)

    adv, ret = compute_gae_2d(R, V, last_v, gamma, lam)
    B, H = R.shape
    flat = lambda x: x.reshape(B * H, -1)
    adv_flat = adv.reshape(-1)
    if adv_flat.size > 1:
        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)   # 推导 (6).1

    rollout = ContinuousRollout(
        x=flat(Z), u=flat(U), a=flat(A), logp=LP.reshape(-1),
        value=V.reshape(-1), reward=R.reshape(-1), adv=adv_flat,
        ret=ret.reshape(-1), ep_returns=[],
    )
    info = {
        "imag_return": float(R.sum(axis=1).mean()),
        "imag_return_std": float(R.sum(axis=1).std()),
        "z_start_std": z_start_std,
        "z_end_std": z_end_std,
        "n_starts": int(n_starts), "horizon": int(horizon),
    }
    return rollout, info
