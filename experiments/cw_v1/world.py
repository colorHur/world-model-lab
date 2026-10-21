# -*- coding: utf-8 -*-
"""CW-POMDP-v1 的环境、外生信道与接收端估计器。

信息边界（v4 §3.3 是硬约束，代码层面用**物理隔离**实现而不是靠自觉）
--------------------------------------------------------------------------
· `CWWorld`：持有**真状态 x** 与**真信道 h**。只在仿真内部使用。
· `SchedulerView`：调度器**唯一**拿到的对象。它**只有**历史观测/测量/ACK/已执行动作，
  没有任何字段能读到 x 或 h。反例 1 就是靠枚举它的可访问面来证明"读不到"。
· `Receiver`：接收端控制器。只用**已送达**的带时间戳观测 + 已知动作做修正与预测。

★ 与旧仓库的关键差异
  旧 `wmlab.control.run_closed_loop_control` 是「每步都尝试发送、按丢包率决定是否送达」。
  v4 要求「每槽三选一（PROBE/SEND/HOLD）、探测与数据争用同一预算」。
  两者不是同一协议的参数化，**不能包装**，只能新写。

★★ 接收端的两种实现，用途严格分开
  · `Receiver`（correct=True）：**可部署**——观测按时间戳校正，槽末观测先用于 t 时刻、
    再随系统滚动到 t+1。所有真实策略用它。
  · `LeakyReceiver`（correct=False，`allow_truth_keys=True` 才可构造）：
    ★★ 这是**故意的错误实现**，用于反例 5。它把槽末的 x_t 观测直接写进接收端，
    当作已经推进到 t+1 的状态。用它跑出来的成本**不得**进任何结果表。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import CWConfig
from .streams import StreamIDs


# ---------------------------------------------------------------- 动作
class Action:
    """5 个动作（N=2）：PROBE(0) PROBE(1) SEND(0) SEND(1) HOLD。"""

    KINDS = ("PROBE", "SEND", "HOLD")
    __slots__ = ("kind", "user")

    def __init__(self, kind: str, user: int | None = None) -> None:
        if kind not in self.KINDS:
            raise ValueError(f"未知动作类型 {kind!r}")
        if kind == "HOLD":
            if user is not None:
                raise ValueError("HOLD 不接受 user 参数")
        else:
            if user not in (0, 1):
                raise ValueError(f"{kind} 需要 user ∈ {{0,1}}，收到 {user!r}")
        self.kind = kind
        self.user = None if user is None else int(user)

    def __eq__(self, other) -> bool:
        return isinstance(other, Action) and (self.kind, self.user) == (other.kind, other.user)

    def __hash__(self) -> int:
        return hash((self.kind, self.user))

    def __repr__(self) -> str:
        return "HOLD" if self.kind == "HOLD" else f"{self.kind}({self.user})"

    def __str__(self) -> str:
        return repr(self)


def action_space(n_users: int) -> tuple[Action, ...]:
    """固定顺序的动作空间（顺序进证据，防止不同轮次顺序漂移）。"""
    acts: list[Action] = []
    for k in ("PROBE", "SEND"):
        for i in range(n_users):
            acts.append(Action(k, i))
    acts.append(Action("HOLD"))
    return tuple(acts)


def action_cost(act: Action, cfg: CWConfig) -> int:
    """占用多少个资源量子。

    ★ 与"消耗几个物理时隙"是两件事：默认 c_probe=c_data=1 ⇒ 1 量子 = 1 时隙。
      若 c_probe=2，本函数返回 2，`CWWorld.step` 会**真的**推进 2 次物理时间
      （`n_substeps=2`），不是只改账。
    """
    if act.kind == "PROBE":
        return int(cfg.protocol.c_probe)
    if act.kind == "SEND":
        return int(cfg.protocol.c_data)
    return 0


# ---------------------------------------------------------------- 信道
def p_success_from_log_h(log_h: float, cfg: CWConfig) -> float:
    """发包成功概率 p = sigmoid(kappa * (log h - log h*))。

    ★ 抽象，不是真实 5G 的 BLER 曲线拟合。
    """
    k = float(cfg.channel.kappa)
    z = -k * (float(log_h) - cfg.channel.log_h_star)
    # 数值稳定写法（z 很大时 1+exp(z) 溢出）
    if z > 700:
        return 0.0
    if z < -700:
        return 1.0
    return float(1.0 / (1.0 + np.exp(z)))


def steady_state_success(cfg: CWConfig) -> float:
    """rho 无关的稳态平均成功率（log h ~ N(0, s^2) 下的解析值）。

    E[sigmoid(k(z-c))], z~N(0,s^2)。用 Hermite 数值积分，无 scipy 依赖。
    """
    from math import sqrt

    k = float(cfg.channel.kappa)
    c = float(cfg.channel.log_h_star)
    s = float(cfg.channel.h_log_init_std)
    nodes, weights = np.polynomial.hermite_e.hermegauss(80)
    z = nodes * s
    sig = 1.0 / (1.0 + np.exp(-k * (z - c)))
    return float(np.sum(weights * sig) / sqrt(2 * np.pi))


# ---------------------------------------------------------------- 记录
@dataclass
class SlotRecord:
    """一个时隙的完整记账（一行 = 一个时隙，恒等关系可对账）。"""

    t: int
    action: str
    user: int | None
    resource_spent: int
    probe_done: int
    data_attempted: int
    data_delivered: int
    ack: int | None
    csi_reading: float | None
    #: ★ SEND 时被采样的 y_t（含测量噪声）。诊断与结构检验需要它；
    #:   它不是可部署信息之外的泄漏 —— 接收端在 ACK 时本来就能拿到同一个值。
    y_t: float | None
    p_true: float | None
    #: ★ 本槽推进时抽到的过程噪声（诊断：结构检验与误差分解要用；
    #:   调度器拿不到它 —— 它不在 `SchedulerView.history` 的可见字段里）
    w_realized: tuple[float, ...]
    cost_est: float
    cost_true: float
    cost_est_by_user: tuple[float, ...]
    cost_true_by_user: tuple[float, ...]
    est_norm: float
    true_norm: float
    budget_left: int


# ---------------------------------------------------------------- 世界
class CWWorld:
    """真实系统 + 接收端 + 调度可视图，一次 `step` 走完一个时隙。

    `correct_receiver=False` 只能在 `allow_truth_keys=True` 下使用（反例 5 的错误实现）。
    """

    def __init__(self, cfg: CWConfig, rngs: dict[str, np.random.Generator],
                 stream_ids: StreamIDs, *, allow_truth_keys: bool = False,
                 correct_receiver: bool = True, rho: float | None = None) -> None:
        if not correct_receiver and not allow_truth_keys:
            raise ValueError(
                "LeakyReceiver（槽末观测当下一步）只允许在 allow_truth_keys=True（诊断/反例）下构造，"
                "不得进入任何结果表。"
            )
        self.cfg = cfg
        self.rngs = rngs
        self.stream_ids = stream_ids
        self.allow_truth_keys = bool(allow_truth_keys)
        self.correct_receiver = bool(correct_receiver)
        self.rho = float(rho if rho is not None else cfg.channel.rho_panel[0])

        n = cfg.plant.n_users
        self.n_users = n
        self.action_space = action_space(n)

        # ★ 真值：只在内部
        self._x_true = np.zeros(n, dtype=float)
        self._log_h_true = np.zeros(n, dtype=float)

        # 接收端信念
        self._x_hat = np.zeros(n, dtype=float)
        self._P_hat = np.full(n, float(cfg.plant.prior_var))
        self._stamp = np.full(n, -1, dtype=int)

        # 信道信念
        self._m_h = np.zeros(n, dtype=float)
        self._v_h = np.full(n, float(cfg.channel.h_log_init_std) ** 2)
        self._csi_stamp = np.full(n, -1, dtype=int)

        # 记账
        self.t = 0
        self.budget_total = 0
        self.budget_used = 0
        self.budget_attempted = 0
        self.budget_wasted = 0
        self.n_probe = 0
        self.n_idle = 0
        #: 被预算拒绝、被迫退化为 HOLD 的次数（静默降级必须可见）
        self.budget_blocked = 0
        self.n_ack = 0
        self.n_nack = 0
        self.control_history: list[np.ndarray] = []
        self.records: list[SlotRecord] = []
        #: 反例 1 的真值替换：{"x": array} / {"h": array} / None
        self.truth_override: dict | None = None
        self._w_last = np.zeros(n, dtype=float)
        self._K = np.array([self.nominal_gain(i) for i in range(n)])

    # ---------------- 控制器（名义 LQR，固定，所有方法同一套）
    def nominal_gain(self, user: int) -> float:
        """标量离散 LQR：min Σ q x² + r u²，A=cfg.A[i], B=cfg.B。

        Riccati：P = q + A²P - (A P B)²/(r + B²P)，K = A P B /(r + B²P)，u = -K x̂。
        ★ 不 import `wmlab.control`：那个是单链路单发送语义，形状与用途都不同。
        """
        A = float(self.cfg.plant.A[user])
        B = float(self.cfg.plant.B)
        q = float(self.cfg.plant.q_cost)
        r = float(self.cfg.plant.r_cost)
        P = q
        for _ in range(100000):
            P_new = q + A * A * P - (A * P * B) ** 2 / (r + B * B * P)
            if abs(P_new - P) < 1e-15:
                P = P_new
                break
            P = P_new
        return float(A * P * B / (r + B * B * P))

    def control_from_estimate(self) -> np.ndarray:
        """u_t = -K_i x̂_i，用**槽开始时刻**的估计。"""
        return -self._K * self._x_hat

    # ---------------- 时隙推进
    def step(self, act: Action, *, n_substeps: int = 1) -> SlotRecord:
        """执行一个时隙。顺序严格按 v4 §3.3。

        `self.truth_override`（由 runner 在反例 1 中设置）在推进之后覆盖真值，
        用于验证"只改真值 ⇒ 动作不变"。
        """
        cfg = self.cfg
        t = self.t
        spend = action_cost(act, cfg)

        # ★ 反例 1 的真值替换必须发生在**本槽任何读数之前**：
        #   采样 x_t、读 CSI、判成败都要看见同一份替换后的真值。
        #   （第一版只在推进后替换，于是本槽仍读旧真值 ⇒ 扰动错位一步 ⇒ 1d 误判。）
        if self.truth_override:
            if "x" in self.truth_override:
                self._x_true = np.asarray(self.truth_override["x"], dtype=float).copy()
            if "h" in self.truth_override:
                self._log_h_true = np.asarray(self.truth_override["h"], dtype=float).copy()

        if spend > 0 and self.budget_used + spend > self.budget_total:
            raise BudgetExhausted(
                f"t={t} 动作 {act} 需 {spend} 量子，余额 {self.budget_total - self.budget_used}"
            )

        # 1) 控制器用上一时隙末的信息算 u_t（本槽尚无任何新信息）
        u = self.control_from_estimate()
        self.control_history.append(u.copy())

        # 2) 若是 SEND：采样 x_t（采样时刻 = t，送达时刻 = 槽末）
        send_sample = None
        if act.kind == "SEND":
            i = act.user
            send_sample = float(self._x_true[i]
                                + cfg.plant.sigma_meas * self.rngs["meas"].standard_normal())

        # 3) 若是 PROBE：有噪 CSI 读数（y = log h_t + N(0, sigma_csi²)）
        csi_reading = None
        if act.kind == "PROBE":
            i = act.user
            csi_reading = float(self._log_h_true[i]
                                + cfg.channel.sigma_csi * self.rngs["csi"].standard_normal())

        # 4) 成败由**真信道**在 t 时刻决定（策略不可见）
        p_true = None
        delivered = 0
        ack: int | None = None
        if act.kind == "SEND":
            i = act.user
            p_true = p_success_from_log_h(float(self._log_h_true[i]), cfg)
            delivered = int(self.rngs["packet"].random() < p_true)
            ack = delivered
            self.budget_attempted += spend
            self.budget_used += spend

        # 5) 物理推进：x_t → x_{t+1}，h_t → h_{t+1}（这是接下去一切的时间基准）
        cost_est = 0.0
        cost_true = 0.0
        est_u = np.zeros(self.n_users)
        true_u = np.zeros(self.n_users)
        for _ in range(int(n_substeps)):
            est_u += cfg.plant.q_cost * self._x_hat ** 2 + cfg.plant.r_cost * u ** 2
            true_u += cfg.plant.q_cost * self._x_true ** 2 + cfg.plant.r_cost * u ** 2
            cost_est += float(np.sum(est_u))
            cost_true += float(np.sum(true_u))
        cost_est_by_user = tuple(float(x) for x in est_u)
        cost_true_by_user = tuple(float(x) for x in true_u)

        # 6) 接收端：先按名义动力学滚动到 t+1（无观测时的"继续预测"）
        #    ★ 滚动前先把 t 时刻的先验存下来 —— 槽末观测属于 t 时刻，
        #      要校正的对象是它，不是已经推进到 t+1 的值。
        x_prior_t = self._x_hat.copy()
        P_prior_t = self._P_hat.copy()
        for i in range(self.n_users):
            self._x_hat[i] = float(self._nominal_map(i, self._x_hat[i], u[i]))
            self._P_hat[i] = float(self._nominal_map_var(i, self._P_hat[i]))

        # 7) 槽末拿到带时间戳 t 的观测/测量 —— 校正**t 时刻**，再按已知 u_t 推到 t+1
        if act.kind == "SEND" and delivered:
            i = act.user
            noise_var = float(cfg.plant.sigma_meas) ** 2
            gain = P_prior_t[i] / (P_prior_t[i] + noise_var)
            x_corrected_t = x_prior_t[i] + gain * (send_sample - x_prior_t[i])
            P_corrected_t = (1.0 - gain) * P_prior_t[i]
            if self.correct_receiver:
                # 正确：校正到 t，再用已知 u_t 推到 t+1
                self._x_hat[i] = self._nominal_map(i, x_corrected_t, u[i])
                self._P_hat[i] = self._nominal_map_var(i, P_corrected_t)
                self._stamp[i] = t
            else:
                # ★★ 故意的错误实现（反例 5）：把 x_t 当成 x_{t+1}
                self._x_hat[i] = x_corrected_t
                self._P_hat[i] = P_corrected_t
                self._stamp[i] = t
            self.n_ack += 1
            # ★ ACK 本身也是信息（v4 §3.3-4）：成功说明当时的 h 不差。
            #   但默认**不**用它更新信念 —— 见 config.use_ack_for_csi 的说明。
            if cfg.channel.use_ack_for_csi:
                self._ack_bump(i, +1.0)
        elif act.kind == "SEND":
            self.budget_wasted += spend
            self.n_nack += 1
            if cfg.channel.use_ack_for_csi:
                self._ack_bump(i, -1.5)

        if act.kind == "PROBE":
            i = act.user
            self.budget_used += spend
            self.n_probe += 1
            obs_var = float(cfg.channel.sigma_csi) ** 2
            g = self._v_h[i] / (self._v_h[i] + obs_var)
            self._m_h[i] = self._m_h[i] + g * (csi_reading - self._m_h[i])
            self._v_h[i] = (1.0 - g) * self._v_h[i]
            self._csi_stamp[i] = t
            # ★ 不触碰任务估计与时间戳（反例 3）
        elif act.kind == "HOLD":
            self.n_idle += 1   # ★ 不产生 ACK（反例 4）

        # 8) 信道信念随 AR 演化（所有用户；用真实 rho，不用面板首项）
        self._m_h = self.rho * self._m_h
        self._v_h = self.rho * self.rho * self._v_h + (1.0 - self.rho * self.rho)

        # 9) 物理与信道真的推进 n_substeps 次
        for _ in range(int(n_substeps)):
            self._advance_physics()
            self._advance_channel()
            self.t += 1
            # ★ 反例 1 的真值替换必须发生在**推进之后**，否则会被推进覆盖 ⇒ 错位一步
            if self.truth_override:
                if "x" in self.truth_override:
                    self._x_true = np.asarray(self.truth_override["x"], dtype=float).copy()
                if "h" in self.truth_override:
                    self._log_h_true = np.asarray(self.truth_override["h"], dtype=float).copy()

        rec = SlotRecord(
            t=t, action=act.kind, user=act.user, resource_spent=spend,
            probe_done=int(act.kind == "PROBE"), data_attempted=int(act.kind == "SEND"),
            data_delivered=delivered, ack=ack, csi_reading=csi_reading,
            y_t=send_sample, p_true=p_true,
            w_realized=tuple(float(x) for x in self._w_last),
            cost_est=cost_est, cost_true=cost_true,
            cost_est_by_user=cost_est_by_user, cost_true_by_user=cost_true_by_user,
            est_norm=float(np.linalg.norm(self._x_hat)),
            true_norm=float(np.linalg.norm(self._x_true)),
            budget_left=self.budget_total - self.budget_used,
        )
        self.records.append(rec)
        return rec

    # ---------------- 内部
    def _nominal_map(self, i: int, x: float, u: float) -> float:
        return float(self.cfg.plant.A[i]) * x + float(self.cfg.plant.B) * u

    def _nominal_map_var(self, i: int, P: float) -> float:
        A = float(self.cfg.plant.A[i])
        return A * A * P + float(self.cfg.plant.sigma_w[i]) ** 2

    def _advance_physics(self) -> None:
        cfg = self.cfg
        A = np.asarray(cfg.plant.A, dtype=float)
        u = self.control_history[-1] if self.control_history else np.zeros(self.n_users)
        w = self.rngs["plant"].standard_normal(self.n_users) * np.asarray(cfg.plant.sigma_w)
        self._w_last = w.copy()
        self._x_true = A * self._x_true + float(cfg.plant.B) * u + w

    def _advance_channel(self) -> None:
        s = self.rngs["channel"].standard_normal(self.n_users)
        self._log_h_true = (self.rho * self._log_h_true
                            + np.sqrt(max(1e-12, 1.0 - self.rho * self.rho)) * s)

    #: ACK 更新的"等效量测方差"（大 ⇒ 每次 ACK 只把信念挪一点）
    ACK_VAR = 0.63   # ≈ π²/3·0.2，对应 1 次二值结果只看清 ±1.5 个 log 单位

    def _ack_bump(self, i: int, direction: float) -> None:
        """用 ACK/NACK 更新用户 i 的信道信念（高斯近似，二值观测）。

        ★ 这不是精确贝叶斯（精确做法要对 `log h` 的连续后验乘上
          `p_succ` 或 `1-p_succ` 的似然并归一化），而是**一阶近似**：
          把"成功/失败"折算成一条方差为 `ACK_VAR` 的伪观测。
          之所以先用近似：R0 只要求 ACK **进入**信念更新且方向正确（反例 4g），
          精确贝叶斯是 R1 的事，写在这里以免被误当成已实现。
        """
        prior_var = float(self._v_h[i])
        y = float(self._m_h[i]) + direction    # 伪观测落在先验附近 ±direction
        g = prior_var / (prior_var + self.ACK_VAR)
        self._m_h[i] = self._m_h[i] + g * (y - self._m_h[i])
        self._v_h[i] = (1.0 - g) * prior_var

    def view(self) -> "SchedulerView":
        return SchedulerView(self)


class BudgetExhausted(RuntimeError):
    """主动动作被预算拒绝。★ 这是协议行为，不是程序错误。"""


class SchedulerView:
    """调度器**能看到的全部信息**。没有 x、没有 h、没有未来。

    ★ 内部世界用**双下划线**存（`__w` ⇒ 被改名为 `_SchedulerView__w`），
      并且 `__slots__` 只放这一个名字。这样 `view._w`、`view.world`、`view.env`
      一律 AttributeError —— 反例 1 的可访问面因此是"空"。
      （单下划线 `_w` 是**能**被直接访问的，第一版正是因此漏了。）
    """

    __slots__ = ("__w",)

    def __init__(self, w: CWWorld) -> None:
        self.__w = w

    @property
    def _inner(self) -> CWWorld:  # pragma: no cover - 仅供本类内部/测试显式取用
        return self.__w

    @property
    def t(self) -> int:
        return self.__w.t

    @property
    def n_users(self) -> int:
        return self.__w.n_users

    @property
    def budget_left(self) -> int:
        return self.__w.budget_total - self.__w.budget_used

    @property
    def budget_total(self) -> int:
        return self.__w.budget_total

    @property
    def telemetry(self) -> list[dict]:
        """逐用户可部署状态：全部由"收到的观测 + 已执行动作 + ACK"推出。"""
        w = self.__w
        out = []
        for i in range(w.n_users):
            stamp = int(w._stamp[i])
            csi_stamp = int(w._csi_stamp[i])
            out.append({
                "i": i,
                "x_hat": float(w._x_hat[i]),
                "P_hat": float(w._P_hat[i]),
                "stamp": stamp,
                "task_aoi": (w.t - stamp) if stamp >= 0 else -1,
                "h_mean": float(w._m_h[i]),
                "h_var": float(w._v_h[i]),
                "csi_stamp": csi_stamp,
                "csi_aoi": (w.t - csi_stamp) if csi_stamp >= 0 else -1,
            })
        return out

    @property
    def history(self) -> tuple[dict, ...]:
        """已发生的记账（可部署部分）。真值字段已剔除。"""
        return tuple(
            {k: v for k, v in r.__dict__.items()
             if k not in ("p_true", "csi_reading", "cost_true", "true_norm")}
            for r in self.__w.records
        )

    def __getattr__(self, name: str):
        raise AttributeError(
            f"SchedulerView 没有属性 {name!r}。真状态/真信道/未来随机数不向调度器开放。"
        )
