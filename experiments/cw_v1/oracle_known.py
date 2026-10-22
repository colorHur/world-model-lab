# -*- coding: utf-8 -*-
"""CW-R1 第 2 步第 1 项：`ORACLE-KNOWN` —— **真值辅助的单步贪心诊断臂**。

★★ 2026-10-22 更正（v4.1 验收 §2）：**撤回"上界臂"这个名称与它承载的推论。**
  它不是最优策略，也不是任何上界：单步贪心、无前瞻、不评估"测一下值不值"、
  不评估剩余预算的续接价值。
  **决定性证据（在本文件自己的产物里）**：同一格 `(ρ=0.85, B/T=0.5)`，
  给它**更多**真信息反而**更差** ——
      变体 `no_x`（真 h、假 x）均值成本 **0.10698**
      变体 `main`（真 h + 真 x）均值成本 **0.45563**（4.26×）
  "加上真 x 就变差"只可能说明**决策规则差**，不可能推出"信息没有价值"。
  见 `results/cw_v1/cw_r1_3_20261022T162956_…/oracle.json` 与
  `docs/CW-R1.1-纠错交付-1.md` §1。

  另有一条必须一起报：本臂 **`probe_mean = 0.0`**（4 格全 0）—— 它拿到真状态后没有理由
  花钱测信道 ⇒ **它对"测量的价值"没有信息量**。要回答那个问题需要有限视界规划参考，
  不是这个臂。

  下面保留的是当时的原始推理（历史记录，不删），但**结论段已经失效**。

---

原始文档（2026-10-21，其中"上界"的用法已撤回）
------------------------------------------------
`ORACLE-KNOWN` —— 全知模型 + 全知信道 的**自我标榜**上界臂。

它要回答的问题（写于看任何结果之前）
------------------------------------
> 在 CW-POMDP-v1 里，"知道模型 + 知道信道"这条信息路径，**最多**能把成本压多少？
> 如果连它都赢不了最好基线 10%（中位比 < 0.9）且逐 episode 胜率 ≥ 0.7，
> 那么**后面 R2 的系统辨识 / 学习就没有可赚的空间**，本课题在 R1 就该收口
> （预注册判词 `SPACE-VOID`）。

★★ 它是**故意越界**的上界臂，不是候选方法
------------------------------------------
- 它读**真信道** `log h_t`（策略本来只能靠 PROBE 花钱去买）⇒ 探测成本被"看到"取代。
- 它还读**真状态** `x_t` 的**模** `|x_t|`（策略本来只能靠送达帧去估）⇒ 决定"这一帧值不值"。
  ★ 先论证为什么这个越界**不额外送钱**（否则上界就是假的）：
  帧的边际价值 `v_i(x) = κ·(√(x²+P_ss) − √(x²+P_ss(1−f)))`，`f = P_ss/(P_ss+σ_meas²)`。
  估值误差对它的初值敏感度 `d/dx = κ·x·(1/√(x²+P) − 1/√(x²+P(1−f)))`，
  在 `x ≲ √P_ss ≈ 1.01` 时最大约 **0.45/单位 x**；而完整模型下的后验误差 `P_ss^{1/2}≈1.01`
  ⇒ 估值误差最多带来 **~0.45/槽** 的值估计噪声，落在 `HOLD` 基线 `0.0891/槽` 的同一量级
  ⇒ 「精度」本身**不是**这条臂显著赢的主要来源（这是本轮的**保守结论**，不是已证事实，
  见 `r1_3_oracle.py` 的 K3 判据：两个变体的差必须落在噪声内）。
- 越界通过**显式白名单**实现：`run_episode(allow_truth_keys=True)` + 本文件自己拿真值句柄，
  **不**通过 `view._inner` 偷看（该入口已于 v1.1 删除；`SchedulerView` 现在只返回公开快照）。

决策规则（一次写定，不看收益）
------------------------------
记 `u = x − x̂`（真值与接收端估计之差）。在 t 槽**必须发出**一帧时的期望增量成本（nats，越大越好）
由 LQR 一阶扰动推出（推导见 `docs/` 交付文档 §2）：

    ΔV_i(x) ≈ κ·( √(x² + P) − √(x² + P(1−f)) ),  κ = ½(q + rK²),  f = P/(P+σ_meas²)
    送达概率 = p_i = sigmoid(κ_ch·(log h_i − log h*))       （真信道，免费）

    v_send(i) = p_i·ΔV_i(x_i)
    v_hold    = 0
    v_probe(i) = 0   ← ★ 全知臂**不需要**探测：它已经知道 `log h`

    a* = argmax{ v_send(i), 0 }，余额不足则 `HOLD`。

为什么 `v_probe ≡ 0` 是一个**可检验的预言**（而不是偷懒）
---------------------------------------------------------
本轮解析量级（BOUNDED，`P_ss^{1/2}≈1.01`、`σ_meas=0.2`、`σ_csi=0.05`）：
- 一条 CSI 读数对帧的**直接**改价上界 `≈ ½(q+rK²)·σ_csi` ≈ **0.03 nats**（`P_ss` 远大于 `σ_meas²`）；
- 但更根本：`x_{t+1}=A x_t+B u_t+w_t` 且 `u_t=−K x̂_t` ⇒ 偏差 `u` 的前向系数
  `|A−BK| = 0.1433`，而**单步测量噪声只让 `u` 的方差增加 `σ_w² = 4e-4`** ⇒ 一条此刻的读数
  对下一槽之后几乎无信息（`α ≈ σ_w²/P ≈ 3.8e-4`）⇒ 一条 CSI 读数的总价值上界
  `≈ ½·κ·σ_csi ≈ 0.03` nats，**比一次 SEND 的机会成本（≈0.34 nats）低一个数量级**。
⇒ 预言：**在 CW-POMDP-v1 的 BOUNDED plant 上，最优调度不会花预算去探测。**
  这不是"探测没用"，而是"**这个 plant 的 plant-noise 太小**，信道信息对任务几乎无价值"。

★ ★ 这条预言如果成立，会给 R2 一个**确凿的负面前提**（要改的不是学习方法，而是 plant 的 SNR）。
  如果不成立（`n_probe > 0` 且更便宜），说明上面的量级算错了 —— 两者都必须原样报。

阴性对照（本文件自带，判据在 `sched_gates_oracle.py`）
------------------------------------------------------
- `use_true_h=True, use_true_x=True`（主臂）
- `use_true_h=False`：把真信道替换成**信念** `p_hat`（= 没有 CSI 特权的盲版）
- `use_true_x=False`：用 `x_hat` 而不是 `x_true` 算 `ΔV`（= 没有状态特权的盲版）
两者必须让动作序列改变，否则说明"特权"根本没接线（R12 / 教训 #39）。

★ 本类**不声称可部署**。任何引用它的结论必须同时写明"这是上界臂"。
"""

from __future__ import annotations

import numpy as np

from .posterior import p_hat_vector
from .world import Action, SchedulerView


class OracleKnown:
    """全知模型 + 全知信道的上界臂（见模块 docstring）。"""

    def __init__(self, cfg, *, use_true_h: bool = True, use_true_x: bool = True,
                 n_users: int = 2, kappa_scale: float = 0.5) -> None:
        self.cfg = cfg
        self.n = int(n_users)
        self.use_true_h = bool(use_true_h)
        self.use_true_x = bool(use_true_x)
        #: `κ = kappa_scale·(q + rK²)`。推导给 ½；留一个乘子做**敏感性**（不进主判据）
        self.kappa_scale = float(kappa_scale)

        var = ("h" if use_true_h else "bh") + ("x" if use_true_x else "bx")
        self.name = f"oracle_known[{var}]"

        P = cfg.plant
        self.q = float(P.q_cost)
        self.r = float(P.r_cost)
        self.B = float(P.B)
        self.A = [float(a) for a in P.A]
        self.sigma_w = [float(s) for s in P.sigma_w]
        self.sigma_meas2 = float(P.sigma_meas) ** 2
        self.K = [self._lqr_gain(i) for i in range(self.n)]
        self.kappa = [self.kappa_scale * (self.q + self.r * k * k) for k in self.K]

        #: 后验方差的**稳态**假设（`P_ss = q + A²P_ss(1 − P_ss/(P_ss+σ_meas²))` 的不动点）。
        #: ★ 不读 `view.P_hat`：全知臂连"接收端估计得准不准"都不该假设 —— 用解析稳态。
        self.P_ss = [self._posterior_ss(i) for i in range(self.n)]

        #: 诊断计数
        self.n_probe_actions = 0
        self.n_send_actions = 0
        self.n_hold_actions = 0
        self.truth_handle = None          # 由 runner 侧显式注入（见 `attach_truth`）
        self._truth_calls = 0

    # ---------------------------------------------------------------- 解析
    def _lqr_gain(self, i: int) -> float:
        """与 `CWWorld.nominal_gain` **同一个** DARE（刻意重复实现，便于逐位对账）。"""
        A = self.A[i]
        B, q, r = self.B, self.q, self.r
        P = q
        for _ in range(100000):
            P_new = q + A * A * P - (A * P * B) ** 2 / (r + B * B * P)
            if abs(P_new - P) < 1e-15:
                P = P_new
                break
            P = P_new
        return float(A * P * B / (r + B * B * P))

    def _posterior_ss(self, i: int) -> float:
        """稳态后验方差（§2 递推的不动点）：`P = q + A²P(1 − P/(P+σ_meas²))`。

        ★ 取 `α = P/(P+σ_meas²)` 与 `world.step` 第 6 步的卡尔曼增益**同一形式**
          （`P ← P − (P²)/(P+σ_meas²)`）；`σ_w` 只进滚动项 `A²P + σ_w²`。
        """
        A = self.A[i]
        sw2 = self.sigma_w[i] ** 2
        P = self.q
        for _ in range(200000):
            P_roll = A * A * P + sw2
            P_new = P_roll - (P_roll * P_roll) / (P_roll + self.sigma_meas2)
            if abs(P_new - P) < 1e-16:
                return float(P_new)
            P = P_new
        return float(P)

    def frame_value(self, i: int, x: float) -> float:
        """`ΔV_i(x) = κ·(√(x²+P) − √(x²+P(1−f)))`，`f = P/(P+σ²)`。"""
        P = self.P_ss[i]
        f = P / (P + self.sigma_meas2)
        a = float(np.hypot(x, np.sqrt(P)))
        b = float(np.hypot(x, np.sqrt(P * (1.0 - f))))
        return self.kappa[i] * (a - b)

    # ---------------------------------------------------------------- 真值白名单
    def attach_truth(self, world) -> None:
        """显式注入真值句柄。**唯一**的越界入口，必须由 runner 侧 `allow_truth_keys=True` 保证。

        ★ 不用 `view._inner`：那是 R0 反例 1 判死的通道。
        """
        if not getattr(world, "allow_truth_keys", False):
            raise PermissionError(
                "OracleKnown 需要 allow_truth_keys=True 的 CWWorld；"
                "本臂是上界臂，必须显式白名单，不得在默认配置下构造。"
            )
        self.truth_handle = world

    # ---------------------------------------------------------------- 决策
    def __call__(self, v: SchedulerView) -> Action:
        return self._decide(v)

    def _decide(self, v: SchedulerView) -> Action:
        tel = v.telemetry
        c_d = max(1, int(self.cfg.protocol.c_data))
        n = len(tel)

        if self.use_true_h:
            if self.truth_handle is None:
                raise RuntimeError("use_true_h=True 但未 attach_truth(world)")
            self._truth_calls += 1
            log_h = np.asarray(self.truth_handle._log_h_true, dtype=float).copy()
            k = float(self.cfg.channel.kappa)
            c = float(self.cfg.channel.log_h_star)
            p = 1.0 / (1.0 + np.exp(-k * (log_h - c)))
        else:
            p = p_hat_vector(tel, self.cfg)

        if self.use_true_x:
            if self.truth_handle is None:
                raise RuntimeError("use_true_x=True 但未 attach_truth(world)")
            x = np.asarray(self.truth_handle._x_true, dtype=float).copy()
        else:
            x = np.array([float(r["x_hat"]) for r in tel], dtype=float)

        vals = np.array([p[i] * self.frame_value(i, float(x[i])) for i in range(n)])

        if v.budget_left < c_d:
            self.n_hold_actions += 1
            return Action("HOLD")
        i_best = int(np.argmax(vals))
        if vals[i_best] <= 0.0:
            self.n_hold_actions += 1
            return Action("HOLD")
        self.n_send_actions += 1
        return Action("SEND", i_best)


class OracleKnownProbeable(OracleKnown):
    """`ORACLE-KNOWN` 的**对照变体**：同样知道信道，但允许它用**别人要花钱的办法**去测。

    存在的唯一目的：证明"主臂不探测"不是因为**没实现探测**，而是因为**探了不划算**。

    规则（一次性写定）
    ------------------
    `v_probe(i) = ½·κ_i·σ_csi`（一条读数的改价上界，推导见模块 docstring）；
    当 `max_i v_probe(i) > max_j v_send(j)` 时执行 `PROBE(argmax v_probe)`，否则走主臂规则。

    ★ 这个阈值**极难触发**：`½κσ_csi ≈ 0.029` vs `v_send ≈ 0.34`（`P_ss^{1/2}≈1.01` 时）
      ⇒ 若它真触发，说明模块 docstring 的量级推导错了，必须报告而不是调阈值。
    ★ 它的 `n_probe` 若为 0，**不能**用来证明"探测通道已关闭"——那是本类的设计结果，
      真正证明通道存在的是 `sched_gates.py` 的 S4/S5（BuyInfoThenSend 的 A/B 对照）。
    """

    def __init__(self, cfg, **kw) -> None:
        super().__init__(cfg, **kw)
        self.name = self.name.replace("oracle_known", "oracle_probeable")

    def probe_bound(self, i: int) -> float:
        """一条 CSI 读数对帧价值的**上界**（`P_ss ≫ σ_meas²` 时线性化）。"""
        return 0.5 * self.kappa[i] * float(self.cfg.channel.sigma_csi)

    def _decide(self, v: SchedulerView) -> Action:
        tel = v.telemetry
        c_p = max(1, int(self.cfg.protocol.c_probe))
        c_d = max(1, int(self.cfg.protocol.c_data))

        if self.use_true_h:
            if self.truth_handle is None:
                raise RuntimeError("use_true_h=True 但未 attach_truth(world)")
            log_h = np.asarray(self.truth_handle._log_h_true, dtype=float).copy()
            k = float(self.cfg.channel.kappa)
            c = float(self.cfg.channel.log_h_star)
            p = 1.0 / (1.0 + np.exp(-k * (log_h - c)))
        else:
            p = p_hat_vector(tel, self.cfg)
        if self.use_true_x:
            if self.truth_handle is None:
                raise RuntimeError("use_true_x=True 但未 attach_truth(world)")
            x = np.asarray(self.truth_handle._x_true, dtype=float).copy()
        else:
            x = np.array([float(r["x_hat"]) for r in tel], dtype=float)

        vals = np.array([p[i] * self.frame_value(i, float(x[i])) for i in range(len(tel))])
        vb = np.array([self.probe_bound(i) for i in range(len(tel))])

        can_probe = v.budget_left >= c_p
        can_send = v.budget_left >= c_d
        i_best = int(np.argmax(vals))
        i_pb = int(np.argmax(vb))

        if can_probe and float(vb[i_pb]) > max(float(vals[i_best]), 0.0):
            self.n_probe_actions += 1
            return Action("PROBE", i_pb)
        if can_send and float(vals[i_best]) > 0.0:
            self.n_send_actions += 1
            return Action("SEND", i_best)
        self.n_hold_actions += 1
        return Action("HOLD")
