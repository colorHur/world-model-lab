"""连续动作 Actor-Critic（Gaussian 策略 + tanh squash）。

★ 为什么这个文件现在才出现
--------------------------
`agents/ppo.py:83` 早就写着「连续动作（Gaussian + tanh squash）**待 X9**
（想象训练，需要 Pendulum）时再补 —— 现在不写未测过的代码」。**本轮 X9 就是那个时点。**

★ 数学推导（② 类证据 · 与下面代码逐段对应）
============================================

记策略 π_θ(u|s) = N(u; μ_θ(s), σ²)，σ 由独立的 `log_std` 参数给出（不依赖 s）。

**(1) 为什么必须 squash**

Pendulum 的动作有界：|a| ≤ 2，而高斯分布的支撑是整条实轴。
若直接令 a = u，会大量采样到 |u| > 2 的区域，**而 gym 会把它 clip 到 ±2**
⇒ 实际执行的动作与 log π 计算所用的动作**不是同一个数** ⇒ 策略梯度**有偏**，
且偏差集中在"动作幅度"这个维度上。

⇒ 用 `a = s · tanh(u)`（s = 动作上界，此处 s = 2）。因为 |tanh| < 1 严格成立，
  所以 |a| < s **恒成立 ⇒ 永不需要 clip ⇒ 不存在 clip 修正项**。
  （若改用 clip，就必须补一条 ∂clip/∂u 的修正，否则同一个坑会以另一种形式回来。）

**(2) ★ log-prob 的换元修正（本文件最容易写错的一行）**

对可逆变换 a = g(u)，密度按雅可比变换：

    π(a) = π(u) · |du/da| = π(u) / |g'(u)|
    ⇒ **log π(a) = log π(u) − log|g'(u)|**

本例 g(u) = s·tanh(u)，g'(u) = s·(1 − tanh²u) = s·sech²u，于是

    **log π(a) = log N(u; μ, σ) − log s − log(1 − tanh²u)**

★ 漏掉这一项的后果**不是"稍微不准"，而是系统性偏**：|g'(u)| 在 |u| 大时趋 0，
  修正项 → −∞，它起的作用正是**惩罚极端动作**。漏掉 ⇒ 智能体会把 σ 推大、
  输出饱和动作（因为饱和动作在"未修正"的 log-prob 下不再被惩罚）。
  ⇒ [58] 用数值积分卡死这一项（**不经过本函数**的另一条路径）。

**(3) 熵（★ 这是一个近似，必须标注）**

squashed 分布 `a = s·tanh(u)` 的**真熵没有闭式解**（需对 log|g'| 求期望）。
本实现报的是**未 squash 的高斯熵**

    H = ½·log(2πe·σ²)

⇒ 它**略高于** squashed 分布的真熵。图与文档中一律标为 `entropy (Gaussian, approx)`。
   这只是监控量（看探索是否塌缩），不进任何判据。

**(4) 工程细节（沿 X5 的两条实测教训）**

1. **ReLU trunk，不用 tanh**：X5 实测 tanh 第二层 79.6% 激活贴 ±1 ⇒ 梯度断流
   ⇒ critic explained variance 0.24 ⇒ 回报卡在 141/209。
2. **正交初始化**：mean 头 gain=0.01（初始策略≈确定性中值，σ 靠 log_std 提供探索）、
   critic 头 gain=1.0。
3. `log_std` 初始 −0.5（σ ≈ 0.61，对 [−2,2] 的动作范围是合理的初始探索）。

范围
----
本文件只服务 **X9（想象训练）与其真实环境对照（H2）**，都是 Pendulum-v1。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from .ppo import clipped_surrogate

LOG_STD_MIN, LOG_STD_MAX = -5.0, 1.0
_LOG2 = float(np.log(2.0))


def squash_correction(u: torch.Tensor) -> torch.Tensor:
    """`log(1 − tanh²u)` 的**数值稳定**形式（推导 (2) 的那一项）。

    恒等变形（可逐点代入验证）：

        log(1 − tanh²u) = log(sech²u) = −2·log(cosh u)
                        = 2·(log 2 − u − softplus(−2u))

    验算 u = 0：右侧 = 2·(0.6931 − 0 − 0.6931) = 0 = log(1 − 0) ✓
    验算 u = ±1：右侧 ≈ −0.8676，`log(1 − tanh²1) = log(0.4200) = −0.8675` ✓

    ★ 为什么不直接写 `torch.log(1 - torch.tanh(u)**2)`：
      float32 下 |u| ≳ 9 时 `tanh(u)²` 舍入成 1.0 ⇒ `1 − 1 = 0` ⇒ `log(0) = −inf`
      ⇒ **整条 loss 变 NaN**（且 NaN 又被铁律 9 描述的 `NaN <= 阈值 ≡ False` 静默吃掉）。
      上式在 u = 20 时给出 −38.6（有限）。[58] 用 |u| = 20 卡死。
    """
    return 2.0 * (_LOG2 - u - F.softplus(-2.0 * u))


class ContinuousActorCritic(nn.Module):
    """共享 trunk 的连续动作 Actor-Critic。

        z (或 obs) -> trunk(64,64,ReLU) -> mean_head -> μ
                                        -> log_std (独立参数) -> σ
                                        -> critic -> V

    ★ 输入维度名叫 `obs_dim`，但 X9 里传的是**潜状态 z**（latent_dim）——
      Dreamer 的想象训练就在潜空间里做。这不是笔误，见 X9 记录 §0.2 决策 B。
    """

    def __init__(self, obs_dim: int, act_dim: int = 1, hidden: int = 64,
                 act_scale: float = 2.0, log_std_init: float = -0.5):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.act_scale = float(act_scale)

        self.trunk = nn.Sequential(
            nn.Linear(self.obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden, self.act_dim)
        self.critic = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.full((self.act_dim,), float(log_std_init)))

        for m in self.trunk:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=float(np.sqrt(2)))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    # ---------------------------------------------------------------- 前向
    def forward(self, x: torch.Tensor):
        """返回 (μ, σ, V)。σ 由 log_std 给出并广播到 batch。"""
        h = self.trunk(x)
        mean = self.mean_head(h)
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)).expand_as(mean)
        return mean, std, self.critic(h).squeeze(-1)

    def to_action(self, u: torch.Tensor) -> torch.Tensor:
        """u（无界）→ a（有界，|a| < act_scale）。推导 (1)。"""
        return self.act_scale * torch.tanh(u)

    def log_prob(self, u: torch.Tensor, mean: torch.Tensor,
                 std: torch.Tensor) -> torch.Tensor:
        """★ squash 后的 log π(a) —— 推导 (2)。

        注意入参是 **u（未 squash 的动作）** 而不是 a：
        更新阶段要重算 log-prob，而从 a 反解 u = atanh(a/s) 在 |a|→s 时数值爆炸；
        直接存 u 则完全精确（这也是 Rollout 里存 u 而不只存 a 的原因）。
        """
        base = Normal(mean, std).log_prob(u)
        return base - _LOG2 - np.log(self.act_scale) - squash_correction(u)

    def entropy(self, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """未 squash 的高斯熵（近似，见推导 (3)）。(B, act_dim)"""
        return Normal(mean, std).entropy()

    def evaluate(self, x: torch.Tensor, u: torch.Tensor):
        """给定 (x, u) 重算 logp / entropy / value —— 更新阶段用。"""
        mean, std, value = self.forward(x)
        logp = self.log_prob(u, mean, std).sum(-1)
        ent = self.entropy(mean, std).sum(-1)
        return logp, ent, value

    # ---------------------------------------------------------------- 采样
    @torch.no_grad()
    def sample(self, x: torch.Tensor, deterministic: bool = False):
        """给定状态张量（B, d）采样动作。

        Returns: dict(u=(B,A), a=(B,A), logp=(B,), value=(B,), ent=(B,))
        """
        mean, std, value = self.forward(x)
        u = mean if deterministic else Normal(mean, std).sample()
        a = self.to_action(u)
        logp = self.log_prob(u, mean, std).sum(-1)
        ent = self.entropy(mean, std).sum(-1)
        return {"u": u, "a": a, "logp": logp, "value": value, "ent": ent}

    @torch.no_grad()
    def act_np(self, x_np, device: torch.device, deterministic: bool = False):
        """单条 numpy 状态 -> (u, a, logp, value, ent)，供与真环境交互的循环调用。"""
        x = torch.as_tensor(np.asarray(x_np, dtype=np.float32),
                            device=device).reshape(1, -1)
        out = self.sample(x, deterministic=deterministic)
        return (out["u"].reshape(-1).cpu().numpy(),
                out["a"].reshape(-1).cpu().numpy(),
                float(out["logp"].item()), float(out["value"].item()),
                float(out["ent"].item()))


# --------------------------------------------------------------------------- 数据
@dataclass
class ContinuousRollout:
    """一批连续动作数据（全部 numpy；进 update 时才转 tensor）。

    ★ 同时存 `u`（未 squash）与 `a`（squash 后）：
      - `a` 给**世界模型**用（它是世界模型见过的动作空间）
      - `u` 给**log-prob 重算**用（推导 (2) 的入参必须是 u）
    """

    x: np.ndarray          # (T, d)   状态（真实环境里是 obs，想象里是 z）
    u: np.ndarray          # (T, A)   未 squash 的动作
    a: np.ndarray          # (T, A)   squash 后的动作
    logp: np.ndarray       # (T,)
    value: np.ndarray      # (T,)
    reward: np.ndarray     # (T,)
    adv: np.ndarray        # (T,)  GAE，已归一化
    ret: np.ndarray        # (T,)  adv + value
    ep_returns: list       # 本批内完整结束的 episode 回报（仅真实环境有；想象里为空）


def compute_gae(reward: np.ndarray, value: np.ndarray, last_value: float,
                gamma: float, lam: float, terminated: np.ndarray | None = None) -> tuple:
    """GAE(γ, λ)（推导 (4)）。

    ★ done 的两种语义（X5 的同一处分叉）：
      `terminated`（真终止）→ 不 bootstrap；truncated（到步数上限）→ **要** bootstrap V(s_T)。
      Pendulum **永远不会 terminated**（只有 TimeLimit 截断）⇒ 调用方传 None，
      表示"全程 nonterminal"，最后一步用 `last_value` 兜底。
    """
    T = len(reward)
    if terminated is None:
        nonterm = np.ones(T, dtype=np.float32)
    else:
        nonterm = (1.0 - np.asarray(terminated, dtype=np.float32))
    adv = np.zeros(T, dtype=np.float32)
    last = 0.0
    next_v, next_nt = float(last_value), (0.0 if (terminated is not None and terminated[-1]) else 1.0)
    for t in reversed(range(T)):
        delta = reward[t] + gamma * next_v * next_nt - value[t]
        adv[t] = last = delta + gamma * lam * next_nt * last
        next_v, next_nt = value[t], nonterm[t]
    ret = adv + value
    if T > 1:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)   # 推导 (6).1
    return adv, ret


# --------------------------------------------------------------------------- 更新
def continuous_ppo_update(net: ContinuousActorCritic, opt: torch.optim.Optimizer,
                          r: ContinuousRollout, *, device: torch.device,
                          update_epochs: int, minibatch_size: int, clip_eps: float,
                          vf_coef: float, ent_coef: float,
                          max_grad_norm: float) -> dict:
    """与 `ppo_update` 逐行同构，只把 log-prob 换成连续版（推导 (5) 同式）。

    ★ 裁剪代理目标调用的是**同一个** `clipped_surrogate`（不是复制一份）——
      这样 X5（离散）与 X9（连续）在"PPO 的核心那一行"上不可能漂移。
    """
    x = torch.as_tensor(r.x, dtype=torch.float32, device=device)
    u = torch.as_tensor(r.u, dtype=torch.float32, device=device)
    logp_old = torch.as_tensor(r.logp, dtype=torch.float32, device=device)
    adv = torch.as_tensor(r.adv, dtype=torch.float32, device=device)
    ret = torch.as_tensor(r.ret, dtype=torch.float32, device=device)

    n = x.shape[0]
    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
             "approx_kl": 0.0, "clip_frac": 0.0, "std": 0.0, "n_minibatch": 0}

    for _ in range(update_epochs):
        idx = np.random.permutation(n)
        for s in range(0, n, minibatch_size):
            mb = torch.as_tensor(idx[s:s + minibatch_size], dtype=torch.int64, device=device)
            logp, ent, value = net.evaluate(x[mb], u[mb])

            ratio = torch.exp(logp - logp_old[mb])
            policy_loss = clipped_surrogate(logp, logp_old[mb], adv[mb], clip_eps)

            value_loss = 0.5 * ((value - ret[mb]) ** 2).mean()
            entropy_mean = ent.mean()
            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy_mean

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
            opt.step()

            with torch.no_grad():
                stats["policy_loss"] += float(policy_loss.item())
                stats["value_loss"] += float(value_loss.item())
                stats["entropy"] += float(entropy_mean.item())
                stats["approx_kl"] += float((logp_old[mb] - logp).mean().item())
                stats["clip_frac"] += float(((ratio - 1.0).abs() > clip_eps).float().mean().item())
                stats["std"] += float(torch.exp(net.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)).mean().item())
                stats["n_minibatch"] += 1

    k = max(1, stats.pop("n_minibatch"))
    return {kk: v / k for kk, v in stats.items()}


# --------------------------------------------------------------------------- 评测
@torch.no_grad()
def evaluate_policy_continuous(env, net: ContinuousActorCritic, x_fn,
                               n_episodes: int, device: torch.device,
                               max_steps: int | None = None, seed: int = 0,
                               deterministic: bool = True) -> dict:
    """真环境评估。

    Args:
        x_fn: obs(np.ndarray) -> 策略输入。**这是 X9 的关键接线点**：
              真实环境 PPO 传 `lambda o: o`（策略吃观测）；
              想象训练传 `lambda o: encode(o)`（策略吃潜状态）——
              两条路都必须能被同一个评估函数评，才谈得上比较。
    """
    rets, lens = [], []
    for i in range(n_episodes):
        o = env.reset(seed=seed + i)
        done, R, t = False, 0.0, 0
        while not done:
            x = x_fn(o)
            _, a, _, _, _ = net.act_np(x, device, deterministic=deterministic)
            res = env.step(a.astype(np.float32))
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


@torch.no_grad()
def collect_rollout_continuous(env, net: ContinuousActorCritic, n_steps: int,
                               gamma: float, lam: float, device: torch.device,
                               x_fn=None, max_episode_steps: int | None = None,
                               seed: int | None = None) -> ContinuousRollout:
    """在**真实环境**里采 n_steps 步（H2 的真实 PPO 对照臂用）。"""
    x_fn = x_fn if x_fn is not None else (lambda o: o)
    A = env.act_dim
    x_buf = np.zeros((n_steps, 0), dtype=np.float32)
    u_buf = np.zeros((n_steps, A), dtype=np.float32)
    a_buf = np.zeros((n_steps, A), dtype=np.float32)
    logp_buf = np.zeros(n_steps, dtype=np.float32)
    val_buf = np.zeros(n_steps, dtype=np.float32)
    rew_buf = np.zeros(n_steps, dtype=np.float32)
    term_buf = np.zeros(n_steps, dtype=np.float32)

    ep_returns: list[float] = []
    cur_ret = 0.0
    o = env.reset() if seed is None else env.reset(seed=seed)
    for t in range(n_steps):
        x = np.asarray(x_fn(o), dtype=np.float32)
        if t == 0:
            x_buf = np.zeros((n_steps, x.shape[-1]), dtype=np.float32)
        u, a, logp, v, _ = net.act_np(x, device)
        x_buf[t], u_buf[t], a_buf[t] = x, u, a
        logp_buf[t], val_buf[t] = logp, v
        res = env.step(a.astype(np.float32))
        rew_buf[t] = res.reward
        term_buf[t] = float(res.terminated)
        cur_ret += float(res.reward)
        if res.done:
            ep_returns.append(cur_ret)
            cur_ret = 0.0
            o = env.reset()
        else:
            o = res.obs

    x_last = np.asarray(x_fn(o), dtype=np.float32)
    last_v = float(net.act_np(x_last, device)[3])
    adv, ret = compute_gae(rew_buf, val_buf, last_v, gamma, lam, terminated=term_buf)
    return ContinuousRollout(x_buf, u_buf, a_buf, logp_buf, val_buf, rew_buf,
                             adv, ret, ep_returns)
