"""2D UAV 目标跟踪环境 —— **X14 场景迁移的第一块**（路线 A：自主可控的自建场景）。

★ 为什么是它（四条理由，前两条来自看板，后两条来自本仓库自己的实验）
------------------------------------------------------------------
① 确定性控制环境上通信压力显不出来（X25：1 步时延 = 丢 70% 包的 59.5 倍）；
② 玩具环境上 H\* 的种子方差高达 ±40%（X3/X4），需要更有结构的场景；
③ "倒立摆"在简历筛选环节不加分（§12.2）；
④ ★ X28 新增：δ 要有分辨力，环境必须带过程噪声 —— 无人系统的"风扰"是
   物理上最自然的过程噪声来源。

★ 本文件同时是 `envs/base.py` 架构声明的第一次实测
------------------------------------------------
base.py 写着："换场景只需新增一个 Adapter 子类，models / rollout / eval 三层一行不改。"
这句话此前**从未被验证过**。本文件就是验证：实现后直接跑 `scripts/02` 的训练与
H\* 测量管线，若需要改动上层代码，声明即被证伪（照实记录）。

★ 场景设计（刻意保持可解析，先立方法再上真实信道）
--------------------------------------------------
- UAV：2D 双积分器 + 速度阻尼 —— p' = p + v·dt；v' = (1−κ·dt)·v + a·dt + w_t
- 目标：已知角速度的圆周运动（**部分可预测**：位置可解析预测，
  但 UAV 自身动力学带风扰 ⇒ 误差有两部分来源，正好可分离研究）
- 观测 = [x, y, vx, vy, tx, ty]（6 维），动作 = 加速度 (ax, ay) ∈ [−a_max, a_max]²
- 过程噪声 w_t ~ N(0, σ²I₂) 加在**速度**上（风扰），σ 可调 —— 继承 X28 的需求
- 终止：UAV 与目标距离超过 escape_radius（跟丢了）⇒ terminated

★ 与 NoisyPendulum 的同一约束：σ=0 时动力学**确定性**、可解析复算，
`step_det()` 暴露无噪声一步（测 δ 必需，与 noisy_pendulum 同一套接口约定）。
"""

from __future__ import annotations

import numpy as np

from .base import EnvAdapter, StepResult


class UavTrackAdapter(EnvAdapter):
    """2D UAV 跟踪圆周运动目标。"""

    dt: float = 0.1
    kappa: float = 0.5          # 速度阻尼系数（1/s）
    a_max: float = 3.0          # 加速度上限 (m/s²)
    v_max: float = 8.0          # 速度上限 (m/s)
    r_orbit: float = 10.0       # 目标圆周半径 (m)
    omega: float = 0.4          # 目标角速度 (rad/s)
    escape_radius: float = 25.0 # 跟丢半径 (m)

    # 阵风场的空间周期（m）：σ(p) = σ0·(1 + amp·g(p))
    wind_period: float = 12.0

    def __init__(self, seed: int = 0, noise_std: float = 0.0, max_steps: int = 200,
                 wind_amp: float = 0.0):
        self.noise_std = float(noise_std)
        self.wind_amp = float(wind_amp)
        self.max_steps = int(max_steps)
        self.name = (f"UavTrack-v0(sigma={self.noise_std:g}"
                     f"{'' if self.wind_amp == 0 else f',wind={self.wind_amp:g}'})")
        self.obs_dim = 6
        self.act_dim = 2
        self.is_discrete = False
        self.act_low = np.full(2, -self.a_max, dtype=np.float32)
        self.act_high = np.full(2, self.a_max, dtype=np.float32)
        self._rng = np.random.default_rng(seed)
        self._t = 0
        self.uav_p = np.zeros(2)
        self.uav_v = np.zeros(2)
        self.tgt_phase = 0.0
        self.reset(seed=seed)

    # ---------- 几何 ----------
    def target_pos(self, phase: float) -> np.ndarray:
        return np.array([self.r_orbit * np.cos(phase),
                         self.r_orbit * np.sin(phase)])

    # ---------- 基本接口 ----------
    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        # UAV 初始在目标圆内随机位置，速度为零；目标相位随机
        r = float(self._rng.uniform(0.0, 0.5 * self.r_orbit))
        ang = float(self._rng.uniform(-np.pi, np.pi))
        self.uav_p = np.array([r * np.cos(ang), r * np.sin(ang)])
        self.uav_v = np.zeros(2)
        self.tgt_phase = float(self._rng.uniform(0.0, 2.0 * np.pi))
        self._t = 0
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        tp = self.target_pos(self.tgt_phase)
        return np.concatenate([self.uav_p, self.uav_v, tp]).astype(np.float32)

    # ---------- ★ X38-b：状态依赖的阵风场（异方差的唯一来源）----------
    def gust_factor(self, pos: np.ndarray) -> float:
        """位置 p 处的阵风放大系数 ∈ [1, 1+wind_amp]。

        设计要点（跑前定死，不许事后改）：
          · 只用**位置**（不用速度/时间）⇒ 阵风场在环境中是**静态**的，
            σ 头有足够信息从观测里把它学出来；
          · 光滑正弦场 ⇒ 不需要很强的函数逼近能力；
          · `wind_amp = 0` 时**逐位退化**为原同方差环境（自检 ㊳）。
        """
        if self.wind_amp == 0.0:
            return 1.0
        k = 2.0 * np.pi / self.wind_period
        g = 0.5 * (1.0 + float(np.sin(k * pos[0]) * np.sin(k * pos[1])))
        return 1.0 + self.wind_amp * g

    def sigma_at(self, pos: np.ndarray) -> float:
        """该位置的**真实**单步噪声标准差（X38-b 的 ground truth）。"""
        return self.noise_std * self.gust_factor(pos)

    def step(self, action) -> StepResult:
        a = np.clip(np.asarray(action, dtype=np.float64).reshape(2),
                    -self.a_max, self.a_max)
        sg = self.sigma_at(self.uav_p)
        w = (self._rng.normal(0.0, sg, size=2) if sg > 0 else np.zeros(2))
        self.uav_p, self.uav_v = self.step_det(
            np.concatenate([self.uav_p, self.uav_v]), a, noise=w)
        self.tgt_phase += self.omega * self.dt
        self._t += 1
        dist = float(np.linalg.norm(self.uav_p - self.target_pos(self.tgt_phase)))
        return StepResult(obs=self._get_obs(), reward=float(-dist),
                          terminated=bool(dist > self.escape_radius),
                          truncated=bool(self._t >= self.max_steps))

    def sample_action(self, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = self._rng if rng is None else rng
        return rng.uniform(self.act_low, self.act_high).astype(np.float32)

    def close(self) -> None:
        pass

    # ---------- 测 δ 必需：无噪声一步 + 状态重置 ----------
    def step_det(self, state, action, noise=None):
        """从指定 state 走一步（无噪声或给定噪声向量）。

        state = [x, y, vx, vy]；noise = 2 维风扰向量（或 None）。
        返回 (p', v')。目标相位由调用方自行推进（它不含随机性）。
        """
        state = np.asarray(state, dtype=np.float64)
        p, v = state[:2], state[2:4]
        a = np.clip(np.asarray(action, dtype=np.float64).reshape(2),
                    -self.a_max, self.a_max)
        w = np.zeros(2) if noise is None else np.asarray(noise, dtype=np.float64)
        v_new = (1.0 - self.kappa * self.dt) * v + a * self.dt + w
        speed = float(np.linalg.norm(v_new))
        if speed > self.v_max:
            v_new = v_new * (self.v_max / speed)
        p_new = p + v_new * self.dt
        return p_new, v_new
