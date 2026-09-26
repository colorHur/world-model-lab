#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""闭环控制与任务级评估 —— **X30 任务锚定的 H\\*** 用。

★ 为什么要有这个文件（它补的是此前所有 H\\* 实验共同缺的一环）
----------------------------------------------------------------
X2 / X14 / X25 / X26 里，**动作是从离线 episode 里读出来的** —— 也就是说，
无论世界模型预测得多离谱，**真实轨迹都不会因此改变**。测出来的 H\\* 因此只是
"模型在一个固定数据分布上能推多远"，**不是"用这个模型做决策会怎样"**。

X30 要的就是后者：控制器**只看得见估计状态**（收不到包时用世界模型 rollout 顶上），
于是模型误差会通过动作**反过来改变真实轨迹**（ compounding / distribution shift
第一次真正接上）。本文件提供这条闭环链上缺的两块：

1. `PDRelativeController` —— 解析 PD 跟踪控制器（无训练、可复现、增益可推导）
2. `run_closed_loop_control` —— 带调度 + 估计器的闭环仿真，直接产出**任务指标**

★★ 为什么用解析 PD 而不是 PPO（诚实交代，不是偷懒）
------------------------------------------------------
`wmlab/agents/ppo.py` 只支持离散动作（连续动作的版本在文件头标着"待 X9"）。
而就算补上连续版，也会引入一个新变量：**PPO 收敛得好不好**。
X30 要测的是"视界 → 任务失败"这一条因果链，**任何额外的方差源都会污染它**。
PD 控制器的增益有闭式解（见下），同一 seed 逐位可复现 —— 这是本实验想要的。

★ PD 增益的闭式推导（②类证据：写进文档的数学推导）
----------------------------------------------------
UAV 动力学（见 `wmlab/envs/uav_track.py`）：

    ṗ = v,     v̇ = −κ·v + a + w          （κ = 0.5 阻尼，w = 风扰）

控制律（对**相对**误差做 PD，v̂_tgt 由估计的目标位置差分得到）：

    a = kp·(tgt − p) + kd·(v̂_tgt − v)

令 e = tgt − p，则 ė = v_tgt − v，ë = a_tgt − p̈。代入：

    ë = a_tgt + κ·v − kp·e − kd·ė − w
      = a_tgt + κ·(v_tgt − ė) − kp·e − kd·ė − w
    ⇒  ë + (κ + kd)·ė + kp·e = a_tgt + κ·v_tgt − w

特征多项式 s² + (κ+kd)s + kp。取阻尼比 ζ、自然频率 ω_n：

    **kp = ω_n²,    kd = 2ζω_n − κ**

稳态误差：**目标做圆周运动 ⇒ 扰动项是旋转的，必须在旋转坐标系里解**，
不能把向心项与阻尼项标量相加（第一版就是这么错的，见下）：

    旋转系下稳态误差 e 为常向量 ⇒ 惯性导数 ė = ω·ẑ×e，ë = −ω²·e，代入上式：

        [ kp−ω²      −(κ+kd)ω ] [e_r]   [ −ω²r   ]      (径向：向心加速度)
        [ (κ+kd)ω     kp−ω²   ] [e_t] = [  κ·ω·r ]      (切向：阻尼补偿)

本仓库 UAV 场景：r=10 m、ω=0.4 rad/s、κ=0.5。取 ω_n=2.5、ζ=1 ⇒ kp=6.25、kd=4.5：

    **e = (−0.1398, 0.3743)，|e| = 0.3996 m**，远小于逃逸半径 25 m。

★ 这一条踩过坑（2026-09-22）：文件头第一版写成标量式
`e_ss = (a_tgt + κ·v_tgt)/kp = (1.6+2.0)/6.25 = 0.576 m` ——
**错在把正交的向心项与切向项直接相加**。诊断脚本实测稳态 **0.394 m**，
与向量解 0.3996 m 吻合到 1.4%，与标量解差 46% ⇒ 向量解才是对的。
`scripts/16` 的 W3 自检就是用这个数判"控制器有没有接对"。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import torch

from .envs.base import EnvAdapter


# ============================================================ 控制器
@dataclass
class PDRelativeController:
    """相对运动 PD 跟踪控制器。

    Args:
        omega_n: 期望闭环自然频率 (rad/s)
        zeta:    期望阻尼比
        kappa:   环境真实阻尼系数（**必须从 env 读，不能猜** —— 猜了增益就错）
        a_max:   加速度上限（同 env）
        kp/kd:   显式给定则覆盖上面三个参数推导出的值（用于消融）
    """

    dt: float = 0.1
    omega_n: float = 2.5
    zeta: float = 1.0
    kappa: float = 0.5
    a_max: float = 3.0
    kp: float | None = None
    kd: float | None = None
    _tgt_prev: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self):
        # 见文件头推导：kp = ω_n², kd = 2ζω_n − κ
        if self.kp is None:
            self.kp = float(self.omega_n ** 2)
        if self.kd is None:
            self.kd = float(2.0 * self.zeta * self.omega_n - self.kappa)

    # ---------- 生命周期 ----------
    def reset(self, est: np.ndarray | None = None) -> None:
        """新 episode 开头调用：`est` 为该 episode 的初始估计观测。"""
        self._tgt_prev = (None if est is None
                          else np.asarray(est, dtype=np.float64).reshape(-1)[4:6].copy())

    # ---------- 决策 ----------
    def act(self, est) -> np.ndarray:
        """由估计状态 [px,py,vx,vy,tx,ty] 给出加速度指令 (2,)。

        ★ 目标速度由**差分**估计，不用任何 env 内部量 —— 否则控制器就拿到了
          真值特权（privileged info），"世界模型顶上"这件事就测不出来了。
        """
        e = np.asarray(est, dtype=np.float64).reshape(-1)
        p, v, tgt = e[0:2], e[2:4], e[4:6]
        if self._tgt_prev is None:
            v_tgt = np.zeros(2)
        else:
            v_tgt = (tgt - self._tgt_prev) / self.dt
        self._tgt_prev = tgt.copy()
        a = self.kp * (tgt - p) + self.kd * (v_tgt - v)
        nrm = float(np.linalg.norm(a))
        if nrm > self.a_max:                       # 逐元素 clip 会改变方向 ⇒ 按范数缩放
            a = a * (self.a_max / nrm)
        return a.astype(np.float32)


# ============================================================ 数据采集
def collect_controlled_episodes(
    env: EnvAdapter,
    controller: PDRelativeController,
    n_episodes: int = 100,
    seed: int = 0,
    max_steps: int | None = None,
    perfect_state: bool = True,
) -> list[dict]:
    """用 PD 控制器采集 episode（`data.collect_policy_episodes` 只支持离散动作）。

    Args:
        perfect_state: True ⇒ 控制器看**真值**（采集"标称分布"的训练数据）；
            False ⇒ 控制器看自己的估计（用于造分布漂移的数据，X30 暂不用）。
    """
    episodes: list[dict] = []
    for i in range(n_episodes):
        obs = env.reset(seed=seed + i)
        controller.reset(obs if perfect_state else obs)
        obs_list = [np.asarray(obs, dtype=np.float32)]
        act_list, rew_list = [], []
        t = 0
        while True:
            a = controller.act(obs)
            res = env.step(a)
            obs = res.obs
            obs_list.append(np.asarray(obs, dtype=np.float32))
            act_list.append(a)
            rew_list.append(res.reward)
            t += 1
            if res.done or (max_steps is not None and t >= max_steps):
                break
        episodes.append({
            "obs": np.asarray(obs_list, dtype=np.float32),
            "act": np.asarray(act_list, dtype=np.float32),
            "reward": np.asarray(rew_list, dtype=np.float32),
            "length": len(act_list),
        })
    return episodes


# ============================================================ 闭环仿真
def _predict_next(model, est: np.ndarray, act: np.ndarray, device) -> np.ndarray:
    """世界模型单步预测：numpy -> numpy。（`model.predict_next` 自带 no_grad）"""
    o = torch.as_tensor(est, dtype=torch.float32, device=device).reshape(1, -1)
    a = torch.as_tensor(act, dtype=torch.float32, device=device).reshape(1, -1)
    out = model.predict_next(o, a).reshape(-1).detach().cpu().numpy()
    # ★★ R14（硬约定）：**报错优于给假数字。** 闭环 rollout 一旦发散出 NaN/Inf，
    #   后续"误差超阈"类判据会因为 NaN <= 阈值 恒为 False 而把发散判成"没超阈"，
    #   返回一个**虚假偏大**的 H* —— 看着完全合理，实际是 bug。这里直接抛。
    if not np.all(np.isfinite(out)):
        raise FloatingPointError(
            f"世界模型闭环 rollout 发散：预测值含非有限值 "
            f"(nan={int(np.isnan(out).sum())}, inf={int(np.isinf(out).sum())})；"
            f"输入 est={np.asarray(est).ravel()[:6]}，act={np.asarray(act).ravel()}")
    return out


def _predict_next_u(model, est: np.ndarray, act: np.ndarray, device) -> tuple[np.ndarray, float]:
    """★ X38（P4）：世界模型单步预测，**顺带返回本步的自报不确定度**。

    与 `_predict_next` 的差别只有一处：还要读 σ 头。
        U² 累加量 = mean_d(σ²)      （d 为潜空间维度；与 X38 Part C 的 U 定义一致）

    ⇒ 确定性模型（`MLPWorldModel`）没有 `next_latent_dist` ⇒ 这里直接抛，
      不许静默返回 0（R14：NaN/静默 0 会让"不确定性触发"悄悄退化成"永不触发"）。
    """
    if not hasattr(model, "next_latent_dist"):
        raise AttributeError(
            "u_state 需要带方差头的世界模型（GaussianWorldModel），"
            f"收到 {type(model).__name__}（无 next_latent_dist）")
    o = torch.as_tensor(est, dtype=torch.float32, device=device).reshape(1, -1)
    a = torch.as_tensor(act, dtype=torch.float32, device=device).reshape(1, -1)
    with torch.no_grad():
        z = model.encode(o)
        mu, sd, _ = model.next_latent_dist(z, a)
        out = model.decode(mu).reshape(-1).detach().cpu().numpy()
        u2 = float(sd.pow(2).mean().item())
    if not np.all(np.isfinite(out)) or not np.isfinite(u2):
        raise FloatingPointError("世界模型闭环 rollout 发散（含 σ 头）")
    return out, u2


def run_closed_loop_control(
    env: EnvAdapter,
    model,
    controller: PDRelativeController,
    schedule: Callable[[int, np.random.Generator], bool],
    *,
    n_episodes: int = 100,
    seed: int = 0,
    max_steps: int = 200,
    device=None,
    estimator: str = "model",
    var_g: float = 1.0,
    label: str = "",
    tail_frac: float = 0.25,
    warmup_steps: int = 0,
    payload_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    u_state: dict | None = None,
) -> dict:
    """★ 闭环控制仿真：**估计状态驱动控制器，控制器改变真实轨迹**。

    ★ X35 新增 `payload_fn`：给**送达的载荷**加一道变换（X33/X34 里是均匀量化器）。
    语义与 `eval.run_tracking` / `rollout.closed_loop_error_curve` 完全一致：
    只改"送过来的那份观测"，不改世界模型、不改控制器。
    ⇒ **量化误差第一次真正进入闭环**（X33/X34 只在开环回放里量过它）。

    与 `eval.run_tracking` 的关键差别（这不是重复实现）：
      `run_tracking` 里动作来自离线 episode ⇒ 轨迹与模型无关；
      这里动作来自 `controller(est)` ⇒ **模型误差会改变真实轨迹**。

    Args:
        estimator: "model" —— 丢包时用世界模型 rollout 顶上（本实验的实验组）
                   "persistence" —— 丢包时**保持上次收到的观测**（基线，X15 的 persistence）
        u_state: ★ X38（P4）**可选**：传一个空 dict，本函数会**每步把当前累积不确定度
                 U 写进去**（`u_state["U"]`），供**不确定性触发**的 schedule 读取。
                 不传 ⇒ 行为与 X30/X35 完全一致（零影响、零开销）。
                 ★ 为什么用 dict 而不是回调：`Schedule` 的签名只有 `(t, rng)`，
                   拿不到 tracker 内部状态 ⇒ 用一个**共享可变容器**做单向传递，
                   比改签名干净，也不破坏既有调用点。
        var_g: 观测 pooled 方差，用于把估计误差归一成 NMSE（与 X2/X14 同口径）
        warmup_steps: ★★ **预热步数**（隔离"初始捕获"瞬态）。

    ★★ 为什么必须有 warmup（2026-09-22 冒烟实测发现，差点当成 bug）
    ------------------------------------------------------------------
    UAV 初始点最远离目标 15 m，而 a_max=3 m/s² ⇒ 加速度**全程饱和**，
    捕获瞬态要 **~400 步**才衰减掉，可 episode 只有 200 步。
    冒烟时 T=1（控制器看真值、与世界模型无关）的稳态距离实测 **2.86 m**，
    而解析 e_ss = **0.40 m** —— 差 7 倍。诊断脚本把 episode 拉到 400 步后，
    t=399 实测 **0.394 m**，与解析 0.3996 m 吻合到 1.4% ⇒ **解析推导没错，
    是瞬态没走完**。

    但"初始捕获"与"通信周期"**无关** —— 把它混进指标，等于给所有 T 加同一个
    常数偏置，会把"任务开始坏掉"的拐点糊掉。所以：预热段用 **T=1（每步送真值）**
    把控制器先收敛到稳态，**预热段不计入任何指标**，之后才切到被测调度。
    """
    if estimator not in ("model", "persistence"):
        raise ValueError(f"estimator 必须是 model/persistence，收到 {estimator!r}")
    rng = np.random.default_rng(seed)
    #: ★ X31：有状态信道（`GilbertElliottChannel`）每条 episode 开头按稳态重采样。
    #: 无状态 schedule（`periodic_schedule`）没有 `reset` ⇒ 零影响（X30 结果不变）。
    reset_fn = getattr(schedule, "reset", None)

    n_tx = 0
    n_steps = 0
    n_escape = 0
    dists: list[float] = []
    err_sq_sum = 0.0
    ages: list[int] = []
    ep_lens: list[int] = []
    tail_dists: list[float] = []
    n_warmup_escape = 0

    for i in range(n_episodes):
        obs = env.reset(seed=seed + i)
        est = np.asarray(obs, dtype=np.float32).copy()
        controller.reset(est)
        if callable(reset_fn):
            reset_fn(rng)

        # ---------- 预热：每步送真值，控制器先收敛到稳态 ----------
        for _ in range(int(warmup_steps)):
            a = controller.act(est)
            res = env.step(a)
            est = np.asarray(res.obs, dtype=np.float32).copy()
            if res.terminated:            # 预热期（真值 + 无丢包）不该跟丢
                n_warmup_escape += 1
                break
            if res.truncated:
                break

        # ---------- 被测段 ----------
        age = 0
        t = 0
        escaped = False
        ep_dist: list[float] = []
        u2 = 0.0
        if u_state is not None:
            u_state["U"] = 0.0
        while t < max_steps:
            a = controller.act(est)                 # est ≈ obs_t（age 已知）
            res = env.step(a)
            true_next = np.asarray(res.obs, dtype=np.float32)
            t += 1

            # ---- 调度：obs_t 这一步的观测是否送达 ----
            # ★ 顺序不能反：必须**先更新估计**，再记账误差。
            #   第一版写成"先记 err = est − true_next 再更新 est"，
            #   那样记到的是 **obs_t − obs_t+1**（一步状态变化量），不是估计误差
            #   ⇒ T=1 时恒等于一步位移，NMSE 永远不为 0，R12 的接线检查会立刻炸。
            delivered = bool(schedule(t, rng))
            if delivered:
                n_tx += 1
                if payload_fn is None:
                    est = true_next.copy()
                else:
                    est = np.asarray(payload_fn(true_next), dtype=np.float32).copy()
                age = 0
                u2 = 0.0                # 收到新观测 ⇒ 累积不确定度清零
            else:
                if estimator == "model":
                    if u_state is not None:
                        est, du2 = _predict_next_u(model, est, a, device)
                        u2 += du2
                    else:
                        est = _predict_next(model, est, a, device)
                else:
                    est = est.copy()    # persistence：原样保持
                age += 1

            # ★ 把「当前估计的累积不确定度」交给调度器（供 P4 的触发策略读）
            if u_state is not None:
                u_state["U"] = float(np.sqrt(max(u2, 0.0)))

            # ---- 记账：est（对 obs_t 的估计） vs 真值 ----
            d = est - true_next
            err_sq_sum += float(np.mean(d * d))
            dist = float(np.linalg.norm(true_next[0:2] - true_next[4:6]))
            dists.append(dist)
            ep_dist.append(dist)
            ages.append(age)
            n_steps += 1

            if res.terminated:
                escaped = True
                break
            if res.truncated:
                break
        ep_lens.append(t)
        n_escape += int(escaped)
        if ep_dist:      # 稳态跟踪误差：只取每集**最后 tail_frac** 段（排除起始瞬态）
            k = max(1, int(round(len(ep_dist) * tail_frac)))
            tail_dists.append(float(np.mean(ep_dist[-k:])))

    dists_a = np.asarray(dists, dtype=float)
    ages_a = np.asarray(ages, dtype=float)
    return {
        "label": label or f"{estimator}/{getattr(schedule, '__name__', 'sched')}",
        "estimator": estimator,
        "n_episodes": n_episodes,
        "n_steps": n_steps,
        "n_tx": n_tx,
        "tx_rate": n_tx / max(n_steps, 1),
        "mean_age": float(ages_a.mean()) if ages_a.size else float("nan"),
        "max_age": int(ages_a.max()) if ages_a.size else 0,
        # ★ X31：突发信道下 age 的**分布形状**才是观测量（均值相同、方差可以差很多）
        "age_std": float(ages_a.std()) if ages_a.size else float("nan"),
        "age_p95": float(np.percentile(ages_a, 95)) if ages_a.size else float("nan"),
        "age_hist": {int(k): int(v) for k, v in
                     zip(*np.unique(ages_a.astype(int), return_counts=True))}
        if ages_a.size else {},
        # ---- ★ 任务指标（H*_task 就定义在它上面）----
        "escape_rate": n_escape / max(n_episodes, 1),
        "mean_dist": float(dists_a.mean()) if dists_a.size else float("nan"),
        "p95_dist": float(np.percentile(dists_a, 95)) if dists_a.size else float("nan"),
        "max_dist": float(dists_a.max()) if dists_a.size else float("nan"),
        # ★ 稳态跟踪误差（排除起始瞬态）—— 与文件头 e_ss 解析值对账用
        "mean_dist_tail": float(np.mean(tail_dists)) if tail_dists else float("nan"),
        "tail_frac": float(tail_frac),
        "mean_ep_len": float(np.mean(ep_lens)) if ep_lens else float("nan"),
        # ---- 估计精度（与离线曲线同分母，供交叉核对）----
        "est_nmse": err_sq_sum / max(n_steps, 1) / float(var_g),
        # ★ 预热段之后的每集长度 —— W2 的解析接线检查要用它精确复算 n_tx / E[age]
        "ep_lens": [int(x) for x in ep_lens],
        "warmup_steps": int(warmup_steps),
        "n_warmup_escape": int(n_warmup_escape),
    }


def task_horizon(rows: Sequence[dict], key: str, base: float, tol_abs: float | None = None,
                 tol_rel: float | None = None) -> float | None:
    """★ 由**任务指标**定义可靠视界：网格上最后一个"还没坏掉"的传输间隔。

    定义：`H_task = max{ T : metric(T) ≤ base + tol_abs }`
          （或 `≤ base · (1 + tol_rel)`，二选一，不可同时给）

    Args:
        rows: `run_closed_loop_control` 的结果列表，**必须含 T 字段并按 T 升序**
        key:  任务指标名（"escape_rate" / "mean_dist"）
        base: T=1（= oracle，每步都传）时的指标值
    Returns:
        最大的合格 T；若全部合格 ⇒ 返回网格最大值（并调用方应标注 "≥"）；
        若 T=1 本身就不合格 ⇒ None（说明控制器/场景没配好）。
    """
    if (tol_abs is None) == (tol_rel is None):
        raise ValueError("tol_abs 与 tol_rel 必须且只能给一个")
    rows = sorted(rows, key=lambda r: float(r["T"]))
    thr = (base + tol_abs) if tol_abs is not None else base * (1.0 + tol_rel)
    if not rows:
        return None
    if float(rows[0][key]) > thr:
        return None
    good = [float(r["T"]) for r in rows if float(r[key]) <= thr]
    return max(good) if good else None
