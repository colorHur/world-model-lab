# -*- coding: utf-8 -*-
"""CW-R1 有限视界规划参考（Planning Reference）——**不是上界，不是最优策略**。

问题（`docs/research/CW-R1-有限视界参考-预注册-v0.1.md` §1）
------------------------------------------------------------
在"预算有限、信道未知、状态未知"的**同一信息集**下，"现在花一帧预算去测信道"
能不能换来更低的后续任务代价？

本文件的纪律（违反任一条，结果作废）
------------------------------------
1. **只吃 `SchedulerView`**：`prior` / `telemetry` / `history` / `budget_left` / `t`。
   不读 `_x_true` / `_log_h_true` / `w_realized` / `cost_true*` / 未来随机数。
   `plan_gates.G1` 用 AST + 深拷贝重建视图两道检查钉住这条。
2. **已知生成参数**：`A, B, σ_w, σ_meas, ρ, κ, log h*, σ_csi, q, r` 来自 `cfg`。
   "上帝知道参数、但不知道当前实现"——这**不等于**读真状态。
3. **条件信念采样**（卡 §2.3）：隐藏量从**当前后验**采样
   `x_true ~ N(x̂, P̂)`、`log h_true ~ N(m_h, v_h)`，不是从先验重采样。
   ★ 这一步是**探测价值的唯一来源**：若不采样隐藏量（把信念当真相），
     投递概率恒等于 `p̂` 本身，"测得更准"在数值上不可能改变任何期望 ⇒
     探测价值恒为 0（`plan_gates.M3` 就是把这个采样关掉，要求闸 FAIL）。

目标函数（看数前写定，理由是可解析的，不是结果驱动的）
------------------------------------------------------
规划器最小化**给定信念下真实代价的贝叶斯期望**：

    E[q·Σ_i x_i² + r·Σ_i u_i² | 信念] = q·Σ_i (P_i + x̂_i²) + r·Σ_i u_i²

★ 为什么必须含 `P`：`cost_est`（世界记的 `q·x̂² + r·u²`）在 `HOLD` 下几何衰减
  （`A−BK = 0.143335`，`u = -K x̂` ⇒ `x̂_{t+1} = 0.143 x̂_t`）⇒ **什么都不做的策略
  `cost_est` 恒等于 0**（实测：40 集 `HOLD`，`cost_est = 0.000000`，
  而 `cost_true = 3.076`）。用 `cost_est` 当主指标会奖励"不发帧"，
  所以本线主指标改为 `cost_true_per_step`，`cost_est` 只作诊断。
  若目标函数也写成 `q·x̂²`（不含 `P`），SEND 的价值恒为 0 ⇒ 规划器退化成 HOLD，
  那正是卡 §5 的 N3（实现缺陷），不是科学结论。

结构（对齐卡 §3，一处**明写的偏离**）
------------------------------------
· 根节点：5 个动作各做 `M` 条条件信念采样，每条走 `H` 步 → 取均值最小者；
· 尾部：每个采样分支按**同一条 1 步贪心**续接（卡 §3「续接」）；
· `H` 步之后给一个**预算感知的尾部估计**（v1.6 起，见 `_paced_sumP`）：
  假设**剩余预算 `b` 按可负担速率均匀铺开**（每用户刷新间隔 `m = steps·n/b`），
  `P` 走一个稳态循环（`P_post ↔ P_b`），`x̂_k = α^k x̂`（`α = A − B·K`）。
  **这是估计，不是最优值函数**；`b = 0` 时退回"再无观测"的解析外推。
  ★ v1.6.1：`m` 取**实数**（原来取整 ⇒ 78.1% 的相邻预算档估计值逐位相同），
  并修正三处只在 `A²≠1` 才走到的解析错误（见 `_paced_sumP` 的"更正记录"与 `G8c2`）。
· ★ **为什么必须这样（R1.5 的实测结论，不是审美）**：R1.5 的尾部假设"今后**再无**观测"，
  `A=1` 时 `ΣP_k = steps·P + σ_w²·steps(steps+1)/2` **二次增长** ⇒
  ① 任何"现在发一帧"都显得极贵；② 花掉的预算在尾部估计里**没有价格**
  ⇒ 规划器把 128 帧在**前 128 槽**花光（`send_profile` 前 4 窗 31.25/31.25/31.13/29.75，
  后 4 窗 4.63/0/0/0），`mean_task_aoi` **122.8**，比 `periodic2` 差 1.37–2.02 倍。
  ⇒ 配速残差比"探测 vs 不探测"的待测效应大一个数量级，**必须先把配速修掉**。
  推导与量级代验见 `docs/research/CW-R1.6-配速-预注册-v0.1.md` §5（**看数前**写定）；
  实测见 `docs/CW-R1.6-配速交付.md`。
  ⚠️ 仍然保留的近似：均匀配速是**假定**的分摊方式（不按 `P_i` 加权），
  所以它是"会配速的估计"，不是"最优配速"。

信息时序（严格对齐 `world.step` 的顺序，`pred_gates` 已验证该顺序）
------------------------------------------------------------------
一个槽：① 用**槽初**信念算 `u = -K x̂`；② 若 SEND，用**当前真值**判成败；
③ 若 PROBE，用**当前真值**加噪读数；④ 记本槽代价（`x_true` 与 `u` 都是 t 时刻的）；
⑤ 用带时间戳 t 的观测校正**t 时刻**的信念，再按已知 `u` 推到 t+1；
⑥ 信道信念按 `ρ` 归一演化；⑦ 隐藏量推进到 t+1。

★ 公共随机数（CRN）：同一个槽里 5 个根动作**共用同一段随机数**
  （按 `(base_seed, 调用序号)` 播种）⇒ 动作之间的比较是配对的，
  不是"5 次独立抽样比大小"。这是压低 Monte-Carlo 噪声的必要手段，不是调参。
"""

from __future__ import annotations

import numpy as np

from .config import CWConfig
from .posterior import NODES80, WEIGHTS80
from .world import Action

#: 尾部终止估计的步数上限（防止数值溢出；真实剩余步数 ≤ horizon=256）
_TERM_MAX_STEPS = 400

#: 值相等判定容差（`_pick` 的并列处理用它）
_TIE_EPS = 1e-12

#: ★ v1.7.2：`tail` 的**模块级默认值**，恒为 `"paced"`（R1.6.1 的解析尾部）。
#:  只有 `r1_7_2_dp_wire.py` 在闸/面板期间临时置 `"dp"`，并在 `finally` 恢复。
#:  这样 `r1_5_plan` / `r1_6_plan` / `r1_6_1_plan` / `r1_7_1_audit` 的行为**零改动**。
TAIL_DEFAULT = "paced"


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """数值稳定 sigmoid（与 `world.p_success_from_log_h` 同一形状）。"""
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


class _St:
    """一条采样分支的信念/隐藏量状态（全部是 `(M, n)` 数组）。"""

    __slots__ = ("xh", "P", "mh", "vh", "xt", "ht", "bud")

    def __init__(self, xh, P, mh, vh, xt, ht, bud) -> None:
        self.xh = xh      # 信念均值（策略可见）
        self.P = P        # 信念方差（策略可见）
        self.mh = mh      # 信道信念均值
        self.vh = vh      # 信道信念方差
        self.xt = xt      # ★ 采样出来的隐藏真状态（**本规划器自己的抽样**，不是世界的）
        self.ht = ht      # ★ 采样出来的隐藏真 log h（同上）
        self.bud = bud    # 每分支剩余预算

    def copy(self) -> "_St":
        return _St(self.xh.copy(), self.P.copy(), self.mh.copy(),
                   self.vh.copy(), self.xt.copy(), self.ht.copy(), self.bud.copy())


class _Draws:
    """一个时隙内**所有根动作共用**的随机数面板（严格公共随机数 CRN）。

    ★ 为什么必须这样（这是 G4 第一次失败后改的，实测记录在交付文档）：
      最初的做法是"每个根动作用同一 seed 新建生成器"。那只配对到**第一步**：
      第一步之后各分支消耗随机数的次数不同，从第二步起就各走各的路径 ⇒
      根动作之间的差值不是配对差，M=16 时动作一致率只有 **65.6%**、M=128 仍不稳。
      改成"按 `(步 k, 样本 m, 用户 i)` 预生成、所有分支按同一索引取"之后，
      任意两个根动作在同一 `(k, m)` 上看到**同一个**世界实现 ⇒ 比较变成配对差，
      方差坍缩。**这不是调参，是让 Monte-Carlo 比较成立的必要条件。**
    """

    __slots__ = ("udeliv", "eps_meas", "eps_csi", "s_h", "w", "init_x", "init_h")

    def __init__(self, rng: np.random.Generator, H: int, M: int, n: int,
                 pool_M: int | None = None) -> None:
        #: ★ R1.7.5：`pool_M` = 先把随机数按**更大的样本数**抽出来，再取前 `M` 条。
        #:  `None` ⇒ `pool_M = M` ⇒ 抽取的形状与 `0426fcb` **逐位相同**（默认路径不变，
        #:  由 `P0c` 闸钉住）。`pool_M > M` 时，`M` 与 `pool_M` 两档共享前 `M` 条实现
        #:  ⇒ 两档的差只来自"新增的 `pool_M − M` 条路径"，这才是配对的 CRN 边际。
        #:  没有这个开关，`M=16` 与 `M=32` 抽的是**不同长度**的数组，两批样本无包含关系
        #:  （见 `docs/research/CW-R1.7.5-G4可达性-预注册-v0.1.md` §1、`P0a`）。
        M_draw = int(M if pool_M is None else pool_M)
        if M_draw < int(M):
            raise ValueError(f"pool_M={M_draw} < M={int(M)}：切不出前 M 条")
        # `.copy()`：切片是视图，复制一份保证这条支路拿到的内存与旧实现一样是独占的
        # （值不变；`P0c` 闸逐位比对默认路径）。
        self.udeliv = rng.random((H, M_draw, n))[:, :M, :].copy()
        self.eps_meas = rng.standard_normal((H, M_draw, n))[:, :M, :].copy()
        self.eps_csi = rng.standard_normal((H, M_draw, n))[:, :M, :].copy()
        self.s_h = rng.standard_normal((H, M_draw, n))[:, :M, :].copy()
        self.w = rng.standard_normal((H, M_draw, n))[:, :M, :].copy()
        self.init_x = rng.standard_normal((M_draw, n))[:M, :].copy()
        self.init_h = rng.standard_normal((M_draw, n))[:M, :].copy()


class PlanRef:
    """有限视界规划参考。`use_probe=False` 是**同代码路径**的消融臂。"""

    def __init__(self, cfg: CWConfig, *, h_plan: int = 5, m_samples: int = 16,
                 seed: int = 20261022, use_probe: bool = True,
                 sample_hidden: bool = True, rho: float | None = None,
                 pacing: bool = True, cont_m: bool = True,
                 tail: str | None = None, dp_phat=None,
                 dp_b_cap: int | None = None,
                 dp_axis_mode: str = "user_axis",
                 dp_grid: str | None = None,
                 send_other_advance: str = "fixed",
                 draws_pool_m: int | None = None) -> None:
        self.cfg = cfg
        self.H = int(h_plan)
        self.M = int(m_samples)
        self.use_probe = bool(use_probe)
        self.sample_hidden = bool(sample_hidden)
        #: ★ v1.6：尾部估计是否把**剩余预算的机会成本**算进去（`_paced_sumP`）。
        #:  `False` = 退回 v1.5 的"再无观测"外推 ⇒ 只作 `G8b` 阴性对照，**不进结果表**。
        self.pacing = bool(pacing)
        #: ★ v1.6.1：刷新间隔 `m` 是否取**实数**（`True` = 值函数对预算连续）。
        #:  `False` = 旧的 `round(steps·n/b)` 口径，**只作 `G8d2` 单变量阴性对照**，不进结果表。
        self.cont_m = bool(cont_m)
        #: ★ v1.7.2：尾部用哪一套值函数。`"paced"` = R1.6.1 解析（默认，行为不变）；
        #:  `"dp"` = `plan_dp.DPTail`（`(P_0,P_1,b)` 后向 DP）。
        #:  `None` ⇒ 取模块级 `TAIL_DEFAULT`（只有 R1.7.2 入口临时改为 `"dp"`）。
        self.tail = str(TAIL_DEFAULT if tail is None else tail)
        if self.tail not in ("paced", "dp"):
            raise ValueError(f"tail={self.tail!r} 非法（只允许 paced / dp）")
        #: 诊断闸可以把 `p̂` 钉成指定值（`W0` 用它造**不对称**用户轴）；正常运行留 `None`。
        self._dp_phat = None if dp_phat is None else np.asarray(dp_phat, dtype=float).reshape(-1)
        #: DP 的预算轴上限。饱和律是 `b ≥ τ ⇒ V_τ(·,b)=V_τ(·,τ)`，
        #: 所以必须 `b_cap ≥ horizon`，否则 `b∈(b_cap, τ)` 会被 `value()` **静默压平**。
        self._dp_b_cap = (int(dp_b_cap) if dp_b_cap is not None
                          else int(cfg.protocol.horizon))
        self._dp_axis_mode = str(dp_axis_mode)
        #: ★ R1.7.3：DP 的 `P` 网格。`None` ⇒ 取 `plan_dp.GRID_DEFAULT`（默认 `"legacy"`，
        #:  与 `f719669`…`4e2b2c0` 的证据逐位对应）；`"r173"` 只在 R1.7.3 起的入口显式打开。
        self._dp_grid = (None if dp_grid is None else str(dp_grid))
        #: ★ v1.7.2：`_expect_after` 的 SEND 分支是否也给**未被送用户**推进一槽。
        #:  `"fixed"`（默认）= `P2 = A²P+σ_w²` 再覆盖被送用户 —— 与 HOLD/PROBE 一致；
        #:  `"legacy"` = R1.5–R1.6.1 的旧行为（未送用户原地不动），**只作活体对照**。
        self.send_other_advance = str(send_other_advance)
        if self.send_other_advance not in ("fixed", "legacy"):
            raise ValueError(f"send_other_advance={self.send_other_advance!r} 非法")
        self._dp = None            # 惰性：首个 `__call__` 用当次公开 telemetry 建一次
        self.n_dp_requests = 0
        self.base_seed = int(seed)
        #: ★ R1.7.5：让 `M` 档位共享更大的抽取池（`pool = 32` ⇒ `M=16` 与 `M=32`
        #:  共享前 16 条实现）。`None` = 旧行为（每档独立抽，与 `0426fcb` 逐位相同）。
        #:  **只改采样结构，不改策略**；只在 `r1_7_5_reach.py` 的 `P3` 对照里打开。
        self.draws_pool_m = (None if draws_pool_m is None else int(draws_pool_m))
        if self.draws_pool_m is not None and self.draws_pool_m < self.M:
            raise ValueError(f"draws_pool_m={self.draws_pool_m} < m_samples={self.M}")
        self.rho = float(rho if rho is not None else cfg.channel.rho_panel[0])
        self.name = (f"plan_ref_H{self.H}_M{self.M}"
                     f"{'' if self.use_probe else '_noprobe'}"
                     f"{'' if self.pacing else '_nopace'}"
                     f"{'_dp' if self.tail == 'dp' else ''}")
        # 诊断计数（每 episode 新建实例；runner 不 reset 它）
        self.n_probe = 0
        self.n_send = 0
        self.n_hold = 0
        #: 根节点选到探测时，本槽"P 最大"的用户是否就是被探测的用户
        #: （回答"探测价值是否只来自本槽换人"；验收 §3 P0-C 明令不许只这样写）
        self.probe_same_argmax = 0
        self.probe_diff_argmax = 0
        self._call_i = 0
        self._steps_left = int(cfg.protocol.horizon)   # 每次 `__call__` 覆盖

        p = cfg.plant
        self.A = np.asarray(p.A, dtype=float)
        self.B = float(p.B)
        self.sw = np.asarray(p.sigma_w, dtype=float)
        self.q = float(p.q_cost)
        self.r = float(p.r_cost)
        self.smeas2 = float(p.sigma_meas) ** 2
        self.scsi2 = float(cfg.channel.sigma_csi) ** 2
        self.kappa = float(cfg.channel.kappa)
        self.logh_star = float(cfg.channel.log_h_star)
        self.ack_var = 0.63          # 与 `world.CWWorld.ACK_VAR` 同一常数（其注释有依据）
        self.use_ack = bool(cfg.channel.use_ack_for_csi)
        self.T = int(cfg.protocol.horizon)
        self.K = np.asarray([self._riccati_k(i) for i in range(p.n_users)], dtype=float)
        self.alpha = self.A - self.B * self.K
        self._cand = self._build_candidates()

    # ------------------------------------------------------------ 标量解析量
    def _riccati_k(self, i: int) -> float:
        """与 `world.CWWorld.nominal_gain` 同一个 DARE 不动点（此处独立重算以便对账）。"""
        A, B = float(self.A[i]), self.B
        P = self.q
        for _ in range(100000):
            Pn = self.q + A * A * P - (A * P * B) ** 2 / (self.r + B * B * P)
            if abs(Pn - P) < 1e-15:
                P = Pn
                break
            P = Pn
        return float(A * P * B / (self.r + B * B * P))

    def _p_hat(self, mh: np.ndarray, vh: np.ndarray) -> np.ndarray:
        """`E[sigmoid(κ(log h − log h*))]`，`log h ~ N(mh, vh)`，80 点求积。

        与 `posterior.p_succ_expected` **同点数、同归一化**
        （`plan_gates.G0` 对账：随机 200 组 `(m, v)` 逐位一致）。
        这里向量化成 `(M, n, 80)`，避免逐元素 Python 循环。
        """
        z = (mh[:, :, None]
             + np.sqrt(np.maximum(vh, 0.0))[:, :, None] * NODES80[None, None, :])
        sig = _sigmoid(self.kappa * (z - self.logh_star))
        return np.sum(WEIGHTS80[None, None, :] * sig, axis=2) / np.sqrt(2.0 * np.pi)

    def _immediate_cost(self, st: _St) -> np.ndarray:
        """本槽代价的贝叶斯期望 `q·Σ(P + x̂²) + r·Σ u²`，`u = -K x̂`。"""
        u = -self.K[None, :] * st.xh
        return (self.q * np.sum(st.P + st.xh ** 2, axis=1)
                + self.r * np.sum(u * u, axis=1))

    # ------------------------------------------------------------ 尾部估计
    @staticmethod
    def _geo_sum(a2, w) -> np.ndarray:
        """`Σ_{k=0}^{w-1} A^{2k} = (A^{2w} − 1)/(A² − 1)`，`w` 可为**实数**。

        `A² ≈ 1` 取极限 `w`（阈值 1e-9：此时二阶项 `w(w−1)(A²−1)/2` 远小于 1e-9·w²，
        在 `steps ≤ 400` 的量级下可忽略）。用 `expm1` 保证 `w` 很小时不丢精度。
        """
        ww = np.asarray(w, dtype=float)
        safe = np.abs(a2 - 1.0) > 1e-9
        num = np.expm1(ww * np.log(a2))
        return np.where(safe, num / np.where(safe, a2 - 1.0, 1.0), ww)

    def _paced_sumP(self, P: np.ndarray, steps: int, bud) -> np.ndarray:
        """`Σ_{k=1..steps} P_k` 的**预算感知**估计（`(M, n)` 向量化）。

        模型（推导写在 `docs/research/CW-R1.6-配速-预注册-v0.1.md` §5，先于运行；
        解析修正记录在 `docs/research/CW-R1.6.1-连续预算定价-预注册-v0.1.md` §2）：
        剩余预算 `b` 按可负担速率铺开 ⇒ 每个用户每 `m = steps·n_users / b` 步被刷新一次
        （`m ≥ n_users`，因为一槽最多一帧）。`m` 是**实数**（v1.6.1 起，见下"更正 4"），
        稳态循环满足

            `P_post = P_b·c/(P_b+c) + σ_w²`（送达 + 量测更新后）
            `P_b    = A^{2(m-1)}·P_post + σ_w²·Σ_{k=0}^{m-2} A^{2k}`

        令 `k = A^{2(m-1)}`、`G = σ_w²·(A^{2(m-1)}−1)/(A²−1)`，消元得二次式 ⇒ **闭式**

            `P_b = ½[W + sqrt(W² + 4(G + kσ_w²)·c)]`, `W = G + kσ_w² + (k−1)·c`

        循环均值取**逐相位平均** `P̄ = (1/m)·Σ_{k=0}^{m-1} P_k`（闭式见函数体）——
        `A=1` 时等于 `(P_post + P_b)/2`，`A>1` 时**不等于**（几何增长是凸的）。

        ★ 更正记录（v1.6.1，逐条都是"只在 `A²≠1` 时才走到的分支"，`A=1` 的 `G8c` 看不见）：
          ① `G` 的分母原写成 `1−A²` ⇒ `A>1` 时 `G<0`（噪声和恒正）⇒ 循环水位低估
             40–67%（`R0_DEFAULT` plant 实测；见 `G8c2`）。
          ② 暖入段 `warm_geo` 的噪声项原写成 `s2(X−w)/(A²−1)`，正确是 `s2(Y−w)/(A²−1)`
             （`X = A²(A^{2w}−1)/(1−A²) = −Y`）⇒ `A>1` 时会给出**负**的暖入代价。
          ③ 循环均值原来取端点平均 `(P_post+P_b)/2` ⇒ `A>1` 时系统偏低（凸性）。
          ④ `m` 原来取整（`round(steps·n/b)`）⇒ `b` 在一个区间内变化根本不改 `m`，
             `steps=128` 时有 **100/128（78.1%）** 个相邻预算档的估计值**逐位相同**、0 对上升
             ⇒ "花一帧的机会成本"在值函数里是 0。v1.6.1 起 `m` 用实数（`cont_m=True`），
             `cont_m=False` 只保留给 `G8d2` 做**单变量阴性对照**。

        当前 `P` 低于 `P_b` 时先用 `w` 步"走到 `P_b`"，其余按 `P̄` 计；
        `P ≥ P_b`（比如刚换了信道还没收敛）时 `w=0`，首步仍按 `P` 计 ——
        这一项就是"刚发过一帧、暂时不值得再发"的价格。

        ★ `bud = 0` 是**真的**没有钱 ⇒ 退回"再无观测"的解析外推，不是缺陷。
        """
        steps = int(min(max(int(steps), 0), _TERM_MAX_STEPS))
        if steps <= 0:
            return np.zeros(P.shape[0])
        a2 = (self.A ** 2)[None, :]
        s2 = (self.sw ** 2)[None, :]
        c = float(self.smeas2)
        n = P.shape[1]

        bud_arr = np.asarray(bud, dtype=float).reshape(-1, 1)

        # ---- 旧口径（无观测外推）：bud=0 时兜底
        one_minus = 1.0 - a2
        safe = np.abs(one_minus) > 1e-12
        Pinf = np.where(safe, s2 / np.where(safe, one_minus, 1.0), 0.0)
        geo = np.where(safe, a2 * (1.0 - a2 ** steps) / np.where(safe, one_minus, 1.0),
                       float(steps))
        sumP_free = (P - Pinf) * geo + Pinf * float(steps)
        sumP_free = np.where(safe, sumP_free,
                             float(steps) * P + s2 * (steps * (steps + 1) / 2.0))

        # ---- 刷新间隔：**实数**（v1.6.1）。`cont_m=False` 仅为 `G8d2` 阴性对照保留。
        m_raw = float(steps) * n / np.maximum(bud_arr, 1.0)
        m = np.clip(m_raw, float(n), float(steps))
        if not self.cont_m:
            m = np.round(m)
        kk = np.where(safe, a2 ** (m - 1.0), 1.0)
        G = s2 * self._geo_sum(a2, m - 1.0)          # ★ 更正 ①：分母 A²−1
        W = G + kk * s2 + (kk - 1.0) * c
        Pb = 0.5 * (W + np.sqrt(np.maximum(W * W + 4.0 * (G + kk * s2) * c, 0.0)))
        Ppost = (Pb - G) / np.maximum(kk, 1e-12)
        Sa_m = self._geo_sum(a2, m)
        # ★ 更正 ③：逐相位平均，而不是端点平均
        Pbar = (Ppost * Sa_m + s2 * (Sa_m - m) / np.where(safe, a2 - 1.0, 1.0)) / m
        Pbar = np.where(safe, Pbar, Ppost + s2 * (m - 1.0) / 2.0)

        # ---- 从当前 P 走到 P_b 需要几步（实数解；`A²≈1` 时退化为线性解）
        den = P - Pinf
        ok = safe & (den > 0.0) & (Pb > Pinf)
        ratio = np.where(ok, (Pb - Pinf) / np.where(ok, den, 1.0), 1.0)
        lin = (Pb - P) / np.maximum(s2, 1e-300)
        with np.errstate(divide="ignore", invalid="ignore"):
            logw = np.log(ratio) / np.where(ok, np.log(a2), 1.0)
        w = np.where(ok, logw, lin)
        w = np.where(np.isfinite(w), w, lin)
        w = np.clip(w, 0.0, np.minimum(m, float(steps)))

        # ---- 暖入段（逐值增长；`A²≈1` 时是等差）+ 循环均值段
        warm_lin = w * P + s2 * w * (w + 1.0) / 2.0
        Y = a2 * self._geo_sum(a2, w)                # Σ_{k=1..w} A^{2k}
        # ★ 更正 ②：`P·Y + s2·(Y−w)/(A²−1)`
        warm_geo = P * Y + s2 * (Y - w) / np.where(safe, a2 - 1.0, 1.0)
        warm = np.where(safe, warm_geo, warm_lin)
        rest = np.maximum(float(steps) - w, 0.0) * Pbar
        paced = warm + rest + np.where(w <= 0.0, P - Pbar, 0.0)

        # ★ 只要还有预算就走配速分支（`m` 被夹到 `steps` 时配速分支自己退化成
        #   "一次刷新 + 之外全自由增长"，与无观测外推只差一次刷新）。
        #   **不要**写"只在某条件下用配速"—— 那会让口径随预算档跳变，制造假分辨力。
        return np.where(bud_arr > 0.0, paced, sumP_free)

    # ------------------------------------------------------------ DP 尾部（v1.7.2）
    def _dp_for(self, phat):
        """按给定 `p̂`（每用户一个标量）取 `DPTail`（`plan_dp` 内有 LRU）。"""
        from . import plan_dp
        from .plan_dp import get_dp
        self.n_dp_requests += 1
        grid = plan_dp.GRID_DEFAULT if self._dp_grid is None else self._dp_grid
        return get_dp(self.cfg, phat, b_cap=self._dp_b_cap,
                      axis_mode=self._dp_axis_mode, grid=grid)

    def _wire_dp(self, tel) -> None:
        """用**公开 telemetry** 在 episode 起点建一次 DP。

        ★ `p̂` 冻结在 episode 起点的公开信念（先验 `m_h=0, v_h=1` ⇒ `p̂=0.5`），
          不是逐槽 `p̂`。理由与代价写在预注册卡 §2.3 / `W3`：
          `_terminal` 每槽被调 10² 次，逐槽重建不可行（0.174 s × 10²/槽）。
        """
        if self._dp_phat is not None:
            ph = self._dp_phat
        else:
            if tel is None:
                raise ValueError("_wire_dp 需要 telemetry，或构造时显式给 dp_phat")
            mh = np.asarray([[float(r["h_mean"]) for r in tel]], dtype=float)
            vh = np.asarray([[float(r["h_var"]) for r in tel]], dtype=float)
            ph = self._p_hat(mh, vh)[0]
        self._dp = self._dp_for(ph)

    def _dp_value(self, P: np.ndarray, bud, steps: int) -> np.ndarray:
        """DP 尾部的 `q·Σ_{k=1..steps} Σ_i P_i`（`DPTail.V` **已含** `q`）。"""
        b = np.zeros(P.shape[0]) if bud is None else np.asarray(bud, dtype=float).reshape(-1)
        return self._dp.value(np.maximum(P, 0.0), b, int(steps))

    def _terminal(self, P: np.ndarray, xh: np.ndarray, steps: int, bud=None) -> np.ndarray:
        """H 步之后剩 `steps` 步、`bud` 帧预算的期望尾部代价（**估计**，不是最优值函数）。

        `P` 走 `_paced_sumP`（预算感知）；`x̂` 按名义 `α^k` 衰减。
        ★ 为什么 `x̂` 不跟 `P` 一起"循环"：`E[q·x²|信念] = q·(P + x̂²)`，
          `x̂` 在 `α=0.14` 下一两步就衰减掉，`Σ x̂²` 与刷新间隔**近似无关**
          （`1/(1−α²)=1.021` 倍），所以"陈旧度"的全部预算敏感性都在 `P` 上。
          这个分解在 `A=1`、`α≈0.14` 的本面板上是数量级正确的（见预注册 §5 的代验）。
        """
        steps = int(min(max(steps, 0), _TERM_MAX_STEPS))
        if steps <= 0:
            return np.zeros(P.shape[0])
        if self.tail == "dp":
            if self._dp is None:
                # ★ 不许静默退回解析尾部：否则 `W0` 会"通过"而实际没接线。
                raise RuntimeError(
                    "tail='dp' 但 DP 未初始化 —— 必须由 `__call__` 用首个时隙的公开 telemetry 建")
            task_P = self._dp_value(P, bud, steps)          # 已含 q
        else:
            sumP = (self._paced_sumP(P, steps, bud) if bud is not None
                    else self._paced_sumP(P, steps, np.zeros(P.shape[0])))
            task_P = self.q * np.sum(sumP, axis=1)
        al2 = (self.alpha ** 2)[None, :]
        one_minus_a = 1.0 - al2
        safe_a = np.abs(one_minus_a) > 1e-12
        geo_a = np.where(safe_a,
                         al2 * (1.0 - al2 ** steps) / np.where(safe_a, one_minus_a, 1.0),
                         float(steps))
        sumX2 = (xh ** 2) * geo_a
        return (task_P
                + self.q * np.sum(sumX2, axis=1)
                + self.r * np.sum((self.K ** 2)[None, :] * sumX2, axis=1))


    def _expect_after(self, st: _St, act: Action, steps_after: int) -> np.ndarray:
        """动作 `act` 之后、还剩 `steps_after` 步时的期望尾部代价（对投递做**解析**期望）。

        ★ 近似（明写）：送达会把 `x̂` 拉向观测值，但 `E[y] = x̂`（一阶无偏），
          所以这里只用 `P` 的下降刻画收益，`x̂` 一律按名义 `α·x̂` 外推。
          这个近似对本问题是**保守**的（不虚增 SEND 的收益）。
        """
        a2 = (self.A ** 2)[None, :]
        # ★ 尾部估计必须看到**本动作之后**的剩余预算（v1.6）：
        #   花掉一帧 ⇒ 之后可负担的刷新率下降 ⇒ 尾部 `P̄` 上升。
        #   这是"配速"进入值函数的唯一入口；不给它，SEND/PROBE 在尾部就是免费的。
        bud_after = np.maximum(st.bud - float(self._need(act)), 0.0)
        if not self.pacing:
            # ★ 阴性对照（`G8b`）：强制退回"再无观测"外推。单变量 —— 只动这一处。
            bud_after = np.zeros_like(bud_after)
        if act.kind == "SEND":
            i = int(act.user)
            g = st.P[:, i] / (st.P[:, i] + self.smeas2)
            P_del = a2[:, i] * ((1.0 - g) * st.P[:, i]) + self.sw[i] ** 2
            P_nodel = a2[:, i] * st.P[:, i] + self.sw[i] ** 2
            p = self._p_hat(st.mh, st.vh)[:, i]
            if self.send_other_advance == "legacy":
                # ★ 旧行为（R1.5–R1.6.1 全部面板都是这个口径）：只改被送用户，
                #   **未被送用户的 `P` 原地不动**（没有 `A²P+σ_w²` 推进一槽）。
                #   与 HOLD / PROBE 两支不一致 ⇒ SEND 的尾部成本被系统性低估
                #   （实测差 ~1% 的尾部值，比"一帧价格"本身大一个量级）。
                #   **只作活体对照**，不得进结果表。
                P2 = st.P.copy()
            else:
                P2 = a2 * st.P + (self.sw ** 2)[None, :]
            P2[:, i] = p * P_del + (1.0 - p) * P_nodel
            return self._terminal(P2, self.alpha[None, :] * st.xh, steps_after,
                                  bud=bud_after)
        # HOLD / PROBE：本槽不动任务估计（PROBE 只动信道信念），但 PROBE 一样花钱
        P2 = a2 * st.P + (self.sw ** 2)[None, :]
        return self._terminal(P2, self.alpha[None, :] * st.xh, steps_after,
                              bud=bud_after)

    # ------------------------------------------------------------ 动作选择
    def _build_candidates(self) -> tuple[Action, ...]:
        acts = []
        if self.use_probe:
            acts += [Action("PROBE", 0), Action("PROBE", 1)]
        acts += [Action("SEND", 0), Action("SEND", 1), Action("HOLD")]
        return tuple(acts)

    @staticmethod
    def _need(act: Action) -> int:
        return 1 if act.kind in ("PROBE", "SEND") else 0

    def _pick(self, values: dict[Action, float], budget_left: int) -> Action:
        """取最小者；并列（±1e-12）时**先选不花预算的**，再按固定顺序。

        ★ 这条 tie-break 就是"没有证明的收益就不花钱"。
          v1.5（尾部不认预算）时 `H=1` 的 PROBE 与 HOLD 估计值**严格相等**
          （PROBE 不动任务估计）⇒ 靠 tie-break 才选 HOLD。
          v1.6 起（`_expect_after` 把预算计进尾部）PROBE 已经**严格更差**：
          同样不动任务估计，却让尾部可负担刷新率下降 ⇒ 不需要 tie-break 也不选它。
          两种口径都不允许"因为探测看起来免费而探测"。
        """
        feas = [a for a in self._cand if budget_left >= self._need(a) and np.isfinite(values[a])]
        if not feas:
            return Action("HOLD")
        best = min(values[a] for a in feas)
        near = [a for a in feas if values[a] <= best + _TIE_EPS]
        near.sort(key=lambda a: (0 if a.kind == "HOLD" else 1,
                                 0 if a.kind == "SEND" else 1,
                                 -1 if a.user is None else a.user))
        return near[0]

    def _greedy(self, st: _St) -> Action:
        """同一规划器的 1 步贪心（尾部续接用）。

        ★ 买不起某动作的**分支**退化为 HOLD 的值（不是 +inf）—— 那些分支本来
          就只能 HOLD，把它们的值扔掉会让其余分支的动作被"最少预算的分支"绑架。
        """
        hold_v = self._expect_after(st, Action("HOLD"), max(0, self._steps_left - 1))
        vals: dict[Action, float] = {}
        for a in self._cand:
            need = self._need(a)
            tval = self._expect_after(st, a, max(0, self._steps_left - 1))
            eff = np.where(st.bud >= need, tval, hold_v)
            vals[a] = float(np.mean(eff))
        return self._pick(vals, int(np.max(st.bud)))

    # ------------------------------------------------------------ 单步推进（采样）
    def _advance(self, st: _St, act: Action, *, k: int, d: _Draws) -> None:
        """按 `world.step` 的顺序推进一条分支（原地改 `st`）。

        `k` = 本时隙内的第几步；`d` = 本时隙的公共随机数面板（见 `_Draws`）。
        """
        u = -self.K[None, :] * st.xh
        i = act.user

        if act.kind == "SEND":
            aff = st.bud >= 1
            src_h = st.ht if self.sample_hidden else st.mh
            p = _sigmoid(self.kappa * (src_h[:, i] - self.logh_star))
            deliv = (d.udeliv[k, :, i] < p) & aff
            src_x = st.xt if self.sample_hidden else st.xh
            y = src_x[:, i] + np.sqrt(self.smeas2) * d.eps_meas[k, :, i]
            g = st.P[:, i] / (st.P[:, i] + self.smeas2)
            xc = np.where(deliv, st.xh[:, i] + g * (y - st.xh[:, i]), st.xh[:, i])
            Pc = np.where(deliv, (1.0 - g) * st.P[:, i], st.P[:, i])
            st.xh[:, i] = self.A[i] * xc + self.B * u[:, i]
            st.P[:, i] = self.A[i] ** 2 * Pc + self.sw[i] ** 2
            if self.use_ack:
                dd = np.where(deliv, 1.0, -1.5) * aff
                gg = st.vh[:, i] / (st.vh[:, i] + self.ack_var)
                st.mh[:, i] = st.mh[:, i] + gg * dd
                st.vh[:, i] = (1.0 - gg) * st.vh[:, i]
            st.bud = st.bud - np.where(aff, 1.0, 0.0)
        elif act.kind == "PROBE":
            aff = st.bud >= 1
            src_h = st.ht if self.sample_hidden else st.mh
            y = src_h[:, i] + np.sqrt(self.scsi2) * d.eps_csi[k, :, i]
            g = st.vh[:, i] / (st.vh[:, i] + self.scsi2)
            st.mh[:, i] = np.where(aff, st.mh[:, i] + g * (y - st.mh[:, i]), st.mh[:, i])
            st.vh[:, i] = np.where(aff, (1.0 - g) * st.vh[:, i], st.vh[:, i])
            st.bud = st.bud - np.where(aff, 1.0, 0.0)

        # 信道信念按 ρ 归一演化（所有用户、每个时隙）
        st.mh = self.rho * st.mh
        st.vh = self.rho * self.rho * st.vh + (1.0 - self.rho * self.rho)

        # 隐藏量推进（只有采样臂需要；无采样臂的隐藏量不参与任何计算）
        if self.sample_hidden:
            st.ht = (self.rho * st.ht
                     + np.sqrt(max(1e-12, 1.0 - self.rho ** 2)) * d.s_h[k])
            st.xt = self.A[None, :] * st.xt + self.B * u + d.w[k] * self.sw[None, :]

    # ------------------------------------------------------------ 根节点
    def _init_state(self, v, d: _Draws) -> _St:
        """从**公开快照**构造信念，并做条件信念采样。"""
        tel = v.telemetry
        n = v.n_users
        M = self.M
        xh = np.tile(np.asarray([r["x_hat"] for r in tel], dtype=float), (M, 1))
        P = np.tile(np.asarray([r["P_hat"] for r in tel], dtype=float), (M, 1))
        mh = np.tile(np.asarray([r["h_mean"] for r in tel], dtype=float), (M, 1))
        vh = np.tile(np.asarray([r["h_var"] for r in tel], dtype=float), (M, 1))
        if self.sample_hidden:
            xt = xh + np.sqrt(np.maximum(P, 0.0)) * d.init_x
            ht = mh + np.sqrt(np.maximum(vh, 0.0)) * d.init_h
        else:
            # ★ 诊断臂：把信念当真相（= `plan_gates.M1` 的破坏对象；**不得**进结果表）
            xt, ht = xh.copy(), mh.copy()
        return _St(xh, P, mh, vh, xt, ht, np.full(M, int(v.budget_left), dtype=float))

    def __call__(self, v) -> Action:
        self._call_i += 1
        if self.tail == "dp" and self._dp is None:
            self._wire_dp(v.telemetry)
        steps_left = int(self.T - v.t)
        H = int(min(self.H, max(steps_left, 1)))
        self._steps_left = steps_left
        seed = (self.base_seed * 1_000_003 + self._call_i * 7919) % (2 ** 63 - 1)
        acts = [a for a in self._cand if int(v.budget_left) >= self._need(a)]
        if not acts:
            self.n_hold += 1
            return Action("HOLD")

        # ★ 一个时隙只抽一次随机数面板，5 个根动作共用（严格 CRN 配对）
        d = _Draws(np.random.default_rng(seed), H, self.M, int(v.n_users),
                   pool_M=self.draws_pool_m)
        vals: dict[Action, float] = {}
        for a in acts:
            st = self._init_state(v, d)
            total = np.zeros(self.M)
            act = a
            for k in range(H):
                total = total + self._immediate_cost(st)
                left_after = max(steps_left - (k + 1), 0)
                if k == H - 1:
                    total = total + self._expect_after(st, act, left_after)
                else:
                    self._advance(st, act, k=k, d=d)
                    # ★ 尾部贪心的终止估计必须用**当前**剩余步数：
                    #   `_steps_left` 是根节点剩余时隙，若沿用到尾部会把终止权重
                    #   系统性放大（→ 尾部更爱发帧）。第一次实现就写错了这一处。
                    self._steps_left = max(steps_left - (k + 1), 1)
                    act = self._greedy(st)
            self._steps_left = steps_left
            vals[a] = float(np.mean(total))
        for a in self._cand:
            vals.setdefault(a, float("inf"))

        chosen = self._pick(vals, int(v.budget_left))
        if chosen.kind == "PROBE":
            self.n_probe += 1
            argmax_P = int(np.argmax([r["P_hat"] for r in v.telemetry]))
            if argmax_P == chosen.user:
                self.probe_same_argmax += 1
            else:
                self.probe_diff_argmax += 1
        elif chosen.kind == "SEND":
            self.n_send += 1
        else:
            self.n_hold += 1
        return chosen


def make(cfg: CWConfig, *, arm: str = "probe", h_plan: int = 5, m_samples: int = 16,
         seed: int = 20261022, rho: float | None = None,
         pacing: bool = True, cont_m: bool = True,
         tail: str | None = None, dp_phat=None, dp_b_cap: int | None = None,
         dp_axis_mode: str = "user_axis", dp_grid: str | None = None,
         send_other_advance: str = "fixed",
         draws_pool_m: int | None = None) -> PlanRef:
    """按臂名构造：`probe` / `noprobe` / `nohidden`（后者只给变异闸用）。"""
    kw = dict(h_plan=h_plan, m_samples=m_samples, seed=seed, rho=rho, pacing=pacing,
              cont_m=cont_m, tail=tail, dp_phat=dp_phat, dp_b_cap=dp_b_cap,
              dp_axis_mode=dp_axis_mode, dp_grid=dp_grid,
              send_other_advance=send_other_advance,
              draws_pool_m=draws_pool_m)
    if arm == "probe":
        return PlanRef(cfg, use_probe=True, sample_hidden=True, **kw)
    if arm == "noprobe":
        return PlanRef(cfg, use_probe=False, sample_hidden=True, **kw)
    if arm == "nohidden":
        return PlanRef(cfg, use_probe=True, sample_hidden=False, **kw)
    raise ValueError(f"未知臂 {arm!r}")
