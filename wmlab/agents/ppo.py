"""PPO（Proximal Policy Optimization）—— **手写实现，不抄 CleanRL**。

为什么手写
----------
- 对应 **X5 / E1**：验收口径是「能跑通 + **能逐项解释每一个 loss 项**」。
  抄一个实现能拿到回报曲线，但拿不到 ② 类证据（数学推导）。
- 本文件因此**把推导写在实现旁边**：每一段代码对应上面的哪一条公式，必须能指向。

★ 数学推导（② 类证据 · 逐项对应下面代码）
---------------------------------------
记策略 π_θ(a|s)，折扣回报 J(θ) = E[Σ_t γ^t r_t]，优势 A(s,a) = Q(s,a) − V(s)。

**(1) 策略梯度定理（REINFORCE 的起点）**

    ∇J(θ) = E_t[ ∇_θ log π_θ(a_t | s_t) · A_t ]

直觉：把"带来正优势的动作"的概率往上推，负优势往下压，步长 ∝ |A|。
问题：直接用它做梯度上升，**步长一大策略就崩**（一次坏更新会把 π 推到很远的区域，
而采样数据是按旧 π 采的，新旧分布一错配，梯度方向就失效）。

**(2) 重要性采样（把"旧数据"复用到新策略上）**

    r_t(θ) = π_θ(a_t | s_t) / π_θ_old(a_t | s_t)
    L^CPI(θ) = E_t[ r_t(θ) · A_t ]          ← CPI = conservative policy iteration

r_t = 1 时它退化成 (1)。允许在同一批数据上做**多轮**更新（update_epochs），样本效率因此提高。
但它没有约束 r_t 能走多远 —— 这就是 PPO 要补的地方。

**(3) ★ 裁剪代理目标（PPO 的核心一行）**

    L^CLIP(θ) = E_t[ min( r_t(θ)·A_t ,  clip(r_t(θ), 1−ε, 1+ε)·A_t ) ]

为什么是 `min` 而不是直接 clip：**它构造的是 L^CPI 的一个下界（悲观界）**。
分两种情形看（这一步是理解 PPO 的关键，面试也常问）：

- **A_t > 0**（这个动作比平均好，想提高它的概率）：
  `min` 取的是"被截断的那一支" ⇒ r_t 一旦超过 1+ε，**目标不再上升**。
  ⇒ 阻止"因为某一步优势特别大就把 π 一把推很远"。
- **A_t < 0**（这个动作比平均差，想压低它的概率）：
  r_t 一旦低于 1−ε，被截断的那一支停在 (1−ε)·A_t，而 `min` 取**更负的那一支**…… 
  注意 A_t<0 时 r_t·A_t 随 r_t 减小而**增大**（负得更少），所以 `min` 取的是未截断项
  ⇒ r_t 一旦低于 1−ε，目标**不再下降**。
  ⇒ 阻止"把某个坏动作的概率一把压到 0"（一旦压到 0，就再也采不到它，探索被永久关闭）。

一句总结：**PPO 不是禁止策略变化，而是禁止"单步变化过大"，且上下两个方向都禁。**

**(4) 优势估计 GAE(γ, λ)**

    δ_t     = r_t + γ·V(s_{t+1})·(1 − done_t) − V(s_t)        （TD 残差）
    A_t^GAE = Σ_{l=0}^{∞} (γλ)^l · δ_{t+l}

λ 是**偏差–方差旋钮**：λ→0 退化为一步 TD（低方差、高偏差，因为信 V 不信真实回报）；
λ→1 退化为蒙特卡洛（无偏、高方差，因为整段采样的随机性都进来了）。常用 0.95。
★ **done 的语义**：只有**真终止**（terminated，杆真倒了）才截断 bootstrap；
时间上限截断（truncated，CartPole 到 500 步）**不是**"此后无价值"，必须用 V(s_T) 兜底，
否则智能体会学到"撑到 500 步和倒下差不多"—— 这是一个容易静默发生的坑。

**(5) 三个 loss 项与它们的符号**

    L(θ) = L^CLIP  +  c_vf · L^VF  −  c_ent · H[π]

| 项 | 目标 | 优化方向 | 进 optimizer 时的符号 |
|---|---|---|---|
| `L^CLIP` | 最大化代理目标 | **上升** | 取负（optimizer 只会下降） |
| `L^VF = ½(V − V_target)²` | 减小价值估计误差 | **下降** | 正（MSE 本身就是"越小越好"） |
| `H[π] = −Σ_a π(a) log π(a)` | **增大**熵（保持探索） | **上升** | **取负** |

★ 两个符号是最容易写错的地方：**策略项与熵项都要"上升" ⇒ 进 loss 时都带负号**，
而价值项是唯一"下降"的。写错熵的符号，模型会**过早收敛到确定性策略**（CartPole 上表现为
"训练中期就卡在 200 分上不去"）—— 症状很像超参问题，实际是符号问题。

**(6) 其他三处工程细节（不是数学，但不做会不收敛）**

1. **优势归一化**：`A ← (A − mean) / (std + 1e-8)`（每轮 update 做一次）。
   它不改变策略梯度的期望方向，但把尺度拉到 1 附近 ⇒ 学习率可以跨任务复用。
2. **梯度裁剪** `‖g‖ ≤ 0.5`：策略梯度天然方差大，个别 batch 会出现巨大梯度。
3. **正交初始化**：actor 头用 `gain=0.01`（初始策略接近均匀 ⇒ 熵大 ⇒ 探索充分），
   critic 头用 `gain=1.0`（价值尺度要跟回报对齐）。这个细节来自实践，不影响正确性但影响收敛速度。

范围
----
★ **只实现离散动作（Categorical）** —— X5/X6/X7 全在 CartPole-v1 上。
连续动作（Gaussian + tanh squash）待 X9（想象训练，需要 Pendulum）时再补，
**现在不写未测过的代码**（R12 精神：没接线、没验证的东西不进仓库）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


# --------------------------------------------------------------------------- 网络
class ActorCritic(nn.Module):
    """共享 trunk 的 Actor-Critic（离散动作）。

    obs -> trunk(64,64) -> actor_head -> logits
                        -> critic_head -> V(s)

    共享 trunk 的理由：CartPole 这类任务的"状态特征"对策略与价值是共用的，
    分开两套网络只是多一倍参数、多一倍过拟合风险。

    ★★ 激活函数为什么是 ReLU 而不是 tanh（2026-09-21 实测改的，**这一条踩过坑**）
    ------------------------------------------------------------------------
    第一版用 `Tanh`，训练 15 万步后回报卡在 **141 / 209**（判据 450），
    且 value loss 长期停在 50~70 不降。诊断脚本（`outputs/_temp/_x5_diag.py`）实测：

        h1 层 |a|>0.95 占比 0.000     （正常）
        h2 层 |a|>0.95 占比 0.796     ← 八成激活贴在 ±1
        critic explained variance = 0.2366   （学好应 0.8+）
        ret-V 残差 std 11.2 vs ret std 12.85  ← critic 几乎没有解释力

    tanh 一旦饱和，局部梯度 ≈ 0 ⇒ **梯度流被掐断** ⇒ critic 学不动 ⇒
    优势估计退化为噪声 ⇒ 策略梯度没有方向 ⇒ 熵一直停在 0.60（几乎均匀）。
    **症状看起来像"超参没调好"，实际是激活饱和** —— 这正是"先诊断再扫参"的价值：
    盲目扫 batch / lr 只会得到一堆同样卡住的数字。

    ReLU 在正半轴梯度恒为 1，不存在饱和（代价是可能出现死神经元，
    本网络只有 2 层、维度 64，实测未出现）。
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.actor = nn.Linear(hidden, act_dim)
        self.critic = nn.Linear(hidden, 1)

        # 见推导 (6).3：actor 小 gain（初始近均匀）、critic gain=1、trunk 用 sqrt(2)
        # （正交初始化；对 ReLU 与 Tanh 都是保持激活方差的标准增益）
        for m in self.trunk:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=float(np.sqrt(2)))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(obs)
        return self.actor(h), self.critic(h).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: np.ndarray, device: torch.device, deterministic: bool = False):
        """采样一个动作。返回 (action, logp, value, entropy)。"""
        o = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        logits, value = self.forward(o)
        dist = Categorical(logits=logits)
        a = logits.argmax(dim=-1) if deterministic else dist.sample()
        return (int(a.item()), float(dist.log_prob(a).item()),
                float(value.item()), float(dist.entropy().item()))

    def evaluate(self, obs: torch.Tensor, act: torch.Tensor):
        """给定 (obs, act) 重算 logp / entropy / value —— 用于 update 阶段的多次前向。"""
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        return dist.log_prob(act), dist.entropy(), value


# --------------------------------------------------------------------------- 采样
@dataclass
class Rollout:
    """一批 on-policy 数据（全部为 numpy，进入 update 时才转 tensor）。"""

    obs: np.ndarray        # (T, obs_dim)
    act: np.ndarray        # (T,)
    logp: np.ndarray       # (T,)   旧策略的对数概率
    value: np.ndarray      # (T,)   旧价值估计
    adv: np.ndarray        # (T,)   GAE 优势（已归一化）
    ret: np.ndarray        # (T,)   adv + value，作为价值回归目标
    ep_returns: list       # 本批内完整结束的 episode 回报（仅用于监控）


def collect_rollout(env, net: ActorCritic, n_steps: int, gamma: float, lam: float,
                    device: torch.device, max_episode_steps: int | None = None) -> Rollout:
    """与环境交互 n_steps 步，按推导 (4) 计算 GAE。

    ★ done 的两种语义在这里必须分开：
      - `terminated`（真终止）→ 不 bootstrap（此后确实没有未来回报）
      - `truncated`（到步数上限）→ **要 bootstrap** V(s_T)
      这决定了 GAE 里的 `nonterminal` 掩码用哪一个。
    """
    obs_dim = env.obs_dim
    obs_buf = np.zeros((n_steps, obs_dim), dtype=np.float32)
    act_buf = np.zeros(n_steps, dtype=np.int64)
    logp_buf = np.zeros(n_steps, dtype=np.float32)
    val_buf = np.zeros(n_steps, dtype=np.float32)
    rew_buf = np.zeros(n_steps, dtype=np.float32)
    term_buf = np.zeros(n_steps, dtype=np.float32)   # 只记 terminated

    ep_returns: list[float] = []
    cur_ret = 0.0

    o = env.reset()
    for t in range(n_steps):
        a, logp, v, _ = net.act(o, device)
        obs_buf[t] = o
        act_buf[t] = a
        logp_buf[t] = logp
        val_buf[t] = v
        res = env.step(np.array([a], dtype=np.int64))
        rew_buf[t] = res.reward
        term_buf[t] = float(res.terminated)
        cur_ret += float(res.reward)
        if res.done:
            ep_returns.append(cur_ret)
            cur_ret = 0.0
            o = env.reset()
        else:
            o = res.obs

    # bootstrap：用最后一步之后的状态价值兜底（推导 (4) 的 ★）
    last_v = float(net.act(o, device)[2])
    # 若最后一步是 terminated，则不再有未来回报
    last_nonterminal = 0.0 if term_buf[-1] > 0.5 else 1.0

    adv = np.zeros(n_steps, dtype=np.float32)
    last_gae = 0.0
    next_v, next_nonterm = last_v, last_nonterminal
    for t in reversed(range(n_steps)):
        nonterm = 1.0 - term_buf[t]
        delta = rew_buf[t] + gamma * next_v * next_nonterm - val_buf[t]
        adv[t] = last_gae = delta + gamma * lam * next_nonterm * last_gae
        next_v, next_nonterm = val_buf[t], nonterm
    ret = adv + val_buf

    # 推导 (6).1：优势归一化（不改变期望方向，只把尺度拉回 1 附近）
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    return Rollout(obs_buf, act_buf, logp_buf, val_buf, adv, ret, ep_returns)


# --------------------------------------------------------------------------- 更新
def ppo_update(net: ActorCritic, opt: torch.optim.Optimizer, r: Rollout, *,
               device: torch.device, update_epochs: int, minibatch_size: int,
               clip_eps: float, vf_coef: float, ent_coef: float,
               max_grad_norm: float) -> dict:
    """在同一批数据上跑 update_epochs 轮小批量更新（推导 (2)(3)(5)）。"""
    obs = torch.as_tensor(r.obs, dtype=torch.float32, device=device)
    act = torch.as_tensor(r.act, dtype=torch.int64, device=device)
    logp_old = torch.as_tensor(r.logp, dtype=torch.float32, device=device)
    adv = torch.as_tensor(r.adv, dtype=torch.float32, device=device)
    ret = torch.as_tensor(r.ret, dtype=torch.float32, device=device)

    n = obs.shape[0]
    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
             "approx_kl": 0.0, "clip_frac": 0.0, "n_minibatch": 0}

    for _ in range(update_epochs):
        idx = np.random.permutation(n)
        for s in range(0, n, minibatch_size):
            mb = torch.as_tensor(idx[s:s + minibatch_size], dtype=torch.int64, device=device)
            logp, entropy, value = net.evaluate(obs[mb], act[mb])

            # 推导 (2)：重要性比
            ratio = torch.exp(logp - logp_old[mb])
            a_mb = adv[mb]

            # 推导 (3)：clip 双分支 + min ⇒ 悲观下界
            pg1 = -a_mb * ratio
            pg2 = -a_mb * torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
            policy_loss = torch.max(pg1, pg2).mean()

            # 推导 (5)：价值项下降（正号）、熵项上升（负号）
            value_loss = 0.5 * ((value - ret[mb]) ** 2).mean()
            entropy_mean = entropy.mean()
            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy_mean

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)   # 推导 (6).2
            opt.step()

            with torch.no_grad():
                stats["policy_loss"] += float(policy_loss.item())
                stats["value_loss"] += float(value_loss.item())
                stats["entropy"] += float(entropy_mean.item())
                stats["approx_kl"] += float((logp_old[mb] - logp).mean().item())
                stats["clip_frac"] += float(
                    ((ratio - 1.0).abs() > clip_eps).float().mean().item())
                stats["n_minibatch"] += 1

    k = max(1, stats.pop("n_minibatch"))
    return {kk: v / k for kk, v in stats.items()}


def evaluate_policy(env, net: ActorCritic, n_episodes: int, device: torch.device,
                    max_steps: int | None = None, seed: int = 0) -> dict:
    """确定性（argmax）评估 —— 训练过程监控用，不参与梯度。"""
    rets, lens = [], []
    for i in range(n_episodes):
        o = env.reset(seed=seed + i)
        done, R, t = False, 0.0, 0
        while not done:
            a, _, _, _ = net.act(o, device, deterministic=True)
            res = env.step(np.array([a], dtype=np.int64))
            R += float(res.reward)
            t += 1
            if res.done or (max_steps is not None and t >= max_steps):
                done = True
            else:
                o = res.obs
        rets.append(R)
        lens.append(t)
    return {"return_mean": float(np.mean(rets)), "return_std": float(np.std(rets)),
            "return_min": float(np.min(rets)), "return_max": float(np.max(rets)),
            "len_mean": float(np.mean(lens)), "returns": rets}
