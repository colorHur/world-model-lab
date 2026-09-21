"""Pendulum + **可重置、可关断**的过程噪声。

★ 为什么自己实现，而不是包装 gymnasium 的 `env.step()`
-------------------------------------------------------
X22 证明了 C17 的 δ 在本仓库恒等于 1：

    δ = ½ max_{s,a} ‖T(·|s,a) − T′(·|s,a)‖₁   （两个**转移分布**的总变差距离）

而「环境确定 + 模型确定」⇒ 两个转移都是**点质量** ⇒ `TV ∈ {0,1}` ⇒ 取 max 后 **δ ≡ 1**
⇒ `H(ε,1) = 1+ε`，定理条件退化为 `γ ≤ ε/(1+ε)`，对任何有意义的 γ 都不成立。

要让 δ 落进 (0,1) 的连续区间，必须同时做两件事：
  ① **环境带过程噪声**  ⇒ T 变成真正的分布
  ② **模型输出分布**     ⇒ T′ 也是分布（见 `models/prob_world_model.py`）

而"测两个分布的距离"要求能在**同一个 (s,a)** 上重复采样 next state。
`gymnasium` 的 `env.step()` 做不到：它既不能从任意 state 出发，也不能临时关掉噪声。
⇒ 所以这里自己实现动力学，暴露 `set_state()` 与 `step_det()`。

★ 一致性硬约束（不是"尽量"，是 assert）
--------------------------------------
`noise_std = 0` 时，本适配器必须与 `gymnasium Pendulum-v1` **逐位一致**，
否则所有历史 H\*（Pendulum 42.99 / 39.01 等）与这里的数字不可比。
动力学按 gymnasium 源码复刻：

    newthdot = thdot + (−3g/(2l)·sin(th+π) + 3/(m l²)·u)·dt
    newthdot = clip(newthdot, −8, 8)
    newth    = th + newthdot·dt
    obs      = [cos(th), sin(th), thdot]

（g = 10.0, m = l = 1.0, dt = 0.05, max_torque = 2.0, max_speed = 8.0）

★ 噪声加在哪一层 —— 一个会影响结论的选择
------------------------------------------
加在**角速度 θ̇**上（过程噪声，物理意义 = 未建模的扰动/阵风），
**不加在观测上**。理由：加在观测上会破坏 `cos²θ + sin²θ = 1` 的约束，
把状态推出合法流形 —— 那样测到的"分布距离"里混着一个纯属人为的流形外分量。
"""

from __future__ import annotations

import numpy as np

from .base import EnvAdapter, StepResult


class NoisyPendulumAdapter(EnvAdapter):
    """可重置、噪声强度可调的 Pendulum。"""

    #: 与 gymnasium Pendulum-v1 完全一致的物理参数
    g: float = 10.0
    m: float = 1.0
    l: float = 1.0
    dt: float = 0.05
    max_torque: float = 2.0
    max_speed: float = 8.0

    def __init__(self, seed: int = 0, noise_std: float = 0.0,
                 max_steps: int = 200):
        self.noise_std = float(noise_std)
        self.max_steps = int(max_steps)
        self.name = f"NoisyPendulum-v0(sigma={self.noise_std:g})"
        self.obs_dim = 3
        self.act_dim = 1
        self.is_discrete = False
        self.act_low = np.array([-self.max_torque], dtype=np.float32)
        self.act_high = np.array([self.max_torque], dtype=np.float32)
        self._rng = np.random.default_rng(seed)
        self._t = 0
        # ★ 内部状态用 **float64**：gymnasium 的 Pendulum 也是 float64 存 state。
        #   用 float32 存会在 50 步内累积出 3.3e-6 的偏差（实测），
        #   看似很小但会让"与历史 H* 可比"这句话失效 —— 一致性检验就是为抓它而设。
        self.state = np.zeros(2, dtype=np.float64)
        self.reset(seed=seed)

    # ---------- 基本接口 ----------
    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        th = float(self._rng.uniform(-np.pi, np.pi))
        thdot = float(self._rng.uniform(-1.0, 1.0))
        self.state = np.array([th, thdot], dtype=np.float64)
        self._t = 0
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        th, thdot = float(self.state[0]), float(self.state[1])
        return np.array([np.cos(th), np.sin(th), thdot], dtype=np.float32)

    def step(self, action) -> StepResult:
        u = float(np.asarray(action).reshape(-1)[0])
        n = float(self._rng.normal(0.0, self.noise_std)) if self.noise_std > 0 else 0.0
        self.state = self.step_det(self.state, u, noise=n)
        self._t += 1
        return StepResult(obs=self._get_obs(), reward=0.0,
                          terminated=False,
                          truncated=bool(self._t >= self.max_steps))

    def sample_action(self, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = self._rng if rng is None else rng
        return rng.uniform(self.act_low, self.act_high).astype(np.float32)

    def close(self) -> None:
        pass

    # ---------- ★ 测 δ 必需的额外能力 ----------
    def step_det(self, state, u, noise: float = 0.0) -> np.ndarray:
        """从**指定** state 走一步；`noise=0` 时即 gymnasium 的确定性动力学。"""
        th, thdot = float(state[0]), float(state[1])
        u = float(np.clip(u, -self.max_torque, self.max_torque))
        newthdot = thdot + (-3.0 * self.g / (2.0 * self.l) * np.sin(th + np.pi)
                            + 3.0 / (self.m * self.l ** 2) * u) * self.dt
        newthdot += float(noise)
        newthdot = float(np.clip(newthdot, -self.max_speed, self.max_speed))
        newth = th + newthdot * self.dt
        return np.array([newth, newthdot], dtype=np.float64)

    def set_state(self, state) -> None:
        self.state = np.asarray(state, dtype=np.float64).copy()

    def get_state(self) -> np.ndarray:
        return self.state.copy()


def assert_matches_gymnasium(seed: int = 0, n_steps: int = 50,
                             atol: float = 1e-6) -> float:
    """一致性检验：σ=0 时与 gymnasium Pendulum-v1 逐位比对。

    做法：用 gym env 的初始 state 喂给本适配器，跑**同一串动作**，
    逐步比对观测。返回最大绝对偏差（供 assert 使用）。

    ★ 为什么不能直接比 reset：gym 用 `np_random`（legacy seeding），
    本适配器用 `default_rng`，两者的随机序列不同 —— 所以要**从同一个 state 出发**比。
    """
    import gymnasium as gym

    g_env = gym.make("Pendulum-v1")
    g_env.reset(seed=seed)
    ours = NoisyPendulumAdapter(seed=seed, noise_std=0.0)
    ours.set_state(g_env.unwrapped.state.copy())

    rng = np.random.default_rng(12345)
    worst = 0.0
    for _ in range(n_steps):
        act = rng.uniform(-2.0, 2.0, size=(1,)).astype(np.float32)
        o_gym, _, _, _, _ = g_env.step(act)
        o_ours = ours.step(act).obs
        worst = max(worst, float(np.max(np.abs(o_gym - o_ours))))
    g_env.close()
    ours.close()
    if worst > atol:
        raise AssertionError(
            f"NoisyPendulumAdapter 与 gymnasium Pendulum-v1 不一致："
            f"最大偏差 {worst:.3e} > {atol:.3e}。"
            f"若放任不管，本适配器上的所有 H* 与历史数字不可比。")
    return worst
