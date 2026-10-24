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

★★ 一步预测（2026-10-22 18:xx 核实，`pred_gates.py` 钉死）
  本环境**一个时隙只推进一次物理**（`_advance_physics`）⇒ 接收端的一步预测就是
        x̂_{t+1} = A x̂_t + B u_t ,  P_{t+1} = A² P_t + σ_w²
  这两条是**精确**的（零噪声逐位 0；独立 Kalman 复算逐位 0；1000 episode 的
  `E[e²]/E[P̂]` = 1.024 与 0.986，落在 1±3σ 内）。原先名为 `exact` 的分支是
  "把控制量算两遍 + 把过程噪声加两遍"，已删除。

★★ 外生事件索引（2026-10-22 18:xx）
  `event_pairing="slot"`（默认）：`packet`/`meas`/`csi` 三条流按 `(t, user)` 预生成 ⇒
  两策略在同一 (t,i) 上看到**同一枚硬币**，配对是构造出来的，不是事后对齐的。
  `event_pairing="attempt"`：按动作次数取流（18:xx 前全部证据），只能重放，不许混表。
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
    #: ★ SEND 时被采样的 y_t（含测量噪声）。**只有送达时非 None**：
    #:   未送达的 SEND 没有任何样本到接收端，那时 `y_t` 就是真值 `x_t + 噪声`，
    #:   公开它等于绕开整个信道（2026-10-22 修正：旧版无论成没成都写进去）。
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


#: ★★ 策略可读的历史字段**白名单**（v1.1）。
#:
#: 判别标准只有一条：**这个量是不是"策略自己做过的事 + 它收到的回执"**。
#:   - `cost_est*` 在白名单里：用**自己的估计** `x_hat` 算的，不是真值。
#:   - `cost_true*` / `w_realized` / `p_true` / `true_norm` 不在：它们是世界内部状态。
#:   - `y_t` 只在送达时非 None（见 `SchedulerView.history`）。
#: 新增 `SlotRecord` 字段时**默认不进**这份白名单 —— 想进必须改这里，并写清理由。
PUBLIC_HISTORY_FIELDS: tuple[str, ...] = (
    "t", "action", "user", "resource_spent",
    "probe_done", "data_attempted", "data_delivered", "ack",
    "csi_reading", "y_t",
    "cost_est", "cost_est_by_user",
    "budget_left",
)

#: 真值字段清单（反例闸用它做"白名单之外一律不可见"的对照）。
TRUTH_FIELDS: tuple[str, ...] = (
    "p_true", "w_realized", "cost_true", "cost_true_by_user", "true_norm",
)


# ---------------------------------------------------------------- 网格后验（R1.12 exact 臂）
class GridBelief:
    """一维网格上的 `log h` 后验 —— **精确贝叶斯**信道信念（R1.12）。

    为什么需要它
    ------------
    `_ack_bump` 把一次二值结果折算成"方差 `ACK_VAR=0.63` 的伪观测"，这是一阶近似
    （代码注释自承）。精确做法要对 `log h` 的连续后验乘伯努利似然再归一化 —— 没有闭式，
    只能数值做。R1.12 问的正是"这条伪更新的等效方差该是多少"以及"换成精确贝叶斯会不会
    改变任务代价"。

    为什么是**固定网格**而不是粒子/高斯和
    --------------------------------------
    ① 一维、定义域窄（稳态 `N(0,1)` 且更新带强收缩）⇒ `[-8,8]` / Δ=0.05（321 点）足够，
       矩的离散误差 ~1e-13（闸 4m/4n 用解析与数值积分钉住）；
    ② 传播是**固定**的高斯核卷积 ⇒ 转移矩阵建一次复用，每步只是
       `(n_users, G) @ (G, G)`（≈2e5 flops，256 步 ~50 ms）；
    ③ **确定性**（无采样噪声）⇒ 同种子逐位可复现，`ack_mode` 的模式隔离可逐位对账。

    ★ 信息边界：网格只被"自己的动作 + 收到的 ACK/NACK + PROBE 读数"更新，
      从不接触 `_log_h_true`。它比一阶近似**多**知道的只有"似然的形状"，不多个比特的真值。
    """

    N_GRID = 321
    HALF = 8.0

    def __init__(self, n_users: int, rho: float, kappa: float, log_h_star: float,
                 init_mean: float = 0.0, init_std: float = 1.0) -> None:
        x = np.linspace(-self.HALF, self.HALF, self.N_GRID)
        self.x = x
        self.dx = float(x[1] - x[0])
        self.rho = float(rho)
        #: 初始后验 = 公开先验（与 `world._m_h/_v_h` 的初值同一分布）
        dens = np.exp(-0.5 * ((x - float(init_mean)) / float(init_std)) ** 2)
        dens = dens / dens.sum()
        self.p = np.tile(dens, (int(n_users), 1))
        #: 伯努利似然的两支（在网格上一次算好）
        sig = 1.0 / (1.0 + np.exp(-float(kappa) * (x - float(log_h_star))))
        self.lik_ok = sig
        self.lik_fail = 1.0 - sig
        #: AR 传播核：`T[i,j] ∝ φ((x_j − ρ·x_i)/s)`，`[i,j]` = (源, 目标)，**按行归一**。
        #: ★ 三处必须做对，否则网格会往里掺假信息（闸 4q 抓的是这个）：
        #:   ① 归一化轴：要对**目标**求和（`axis=1`）。按源求和（`axis=0`）会把反向核当权重，
        #:      实测把 5 步稳压从 1.0 推到 1.4478（每步 +8%）—— 看起来像"信道变差"，其实是数值错。
        #:   ② 乘法方向：`p @ T`（`p_new[k]=Σ_j p[j]T[j,k]`）。写成 `p @ T.T` 因 `ρ≠1` 使 `T` 非对称，
        #:      实测 50 步把方差推到 11.55（发了散）。
        #:   ③ ★ **不要**做 `s_eff = sqrt(s²−dx²/12)` 的"Sheppard 修正"：实测本实现的离散二阶矩
        #:      **恒等于 `s²`**（`corr=0` 时 200 步最大偏差 9.9e-14；修了反而稳态掉到 0.9947，−0.53%）。
        #:      原因：节点质量按"密度×dx、再整列归一到 1"生成，`dx²/12` 项不出现。这条是踩过的坑，留着。
        s = np.sqrt(max(1e-12, 1.0 - self.rho * self.rho))
        diff = (x[None, :] - self.rho * x[:, None]) / s        # [源 i, 目标 j]
        T = np.exp(-0.5 * diff * diff)
        T = T / T.sum(axis=1, keepdims=True)
        self.T = T
        self.s_eff = float(s)

    # ---------------- 更新
    def _normalize(self, i: int) -> None:
        z = float(self.p[i].sum())
        if not np.isfinite(z) or z <= 0.0:
            raise ValueError(f"用户 {i} 的后验质量非法（{z}）—— 数值下溢，网格需扩域或换尺度")
        self.p[i] = self.p[i] / z

    def observe_binary(self, i: int, success: bool) -> None:
        """一次 ACK(True)/NACK(False) —— **精确**乘似然后归一。"""
        self.p[i] = self.p[i] * (self.lik_ok if success else self.lik_fail)
        self._normalize(i)

    def observe_gauss(self, i: int, y: float, var: float) -> None:
        """一次 PROBE 读数 `y = log h + N(0, var)` —— 高斯似然，同样精确。"""
        self.p[i] = self.p[i] * np.exp(-0.5 * (self.x - float(y)) ** 2 / float(var))
        self._normalize(i)

    def propagate(self) -> None:
        """一步 AR：后验与 `N(0, 1−ρ²)` 卷积（**不引入任何信息**，闸 4q 钉住这一点）。"""
        # p_new[i,k] = Σ_j p[i,j]·T[j,k] ⇒ 右乘 T（T 因 ρ≠1 而**非对称**，写成 `@ T.T` 会扩散爆炸）
        self.p = self.p @ self.T
        for i in range(self.p.shape[0]):
            self._normalize(i)

    # ---------------- 矩（对外只暴露这两个量，与一阶近似同一消费面）
    def moments(self) -> tuple[np.ndarray, np.ndarray]:
        m = self.p @ self.x
        d = self.x[None, :] - m[:, None]
        v = np.sum(self.p * d * d, axis=1)
        return m, v

    def mass_outside(self, lo: float, hi: float) -> np.ndarray:
        """落在 `[lo,hi]` 之外的质量（闸 4m 用它证明截断无害）。"""
        inside = (self.x >= lo) & (self.x <= hi)
        return self.p[:, ~inside].sum(axis=1)


# ---------------------------------------------------------------- 世界
class CWWorld:
    """真实系统 + 接收端 + 调度可视图，一次 `step` 走完一个时隙。

    `correct_receiver=False` 只能在 `allow_truth_keys=True` 下使用（反例 5 的错误实现）。
    """

    def __init__(self, cfg: CWConfig, rngs: dict[str, np.random.Generator],
                 stream_ids: StreamIDs, *, allow_truth_keys: bool = False,
                 correct_receiver: bool = True, rho: float | None = None,
                 exact_predictor: bool = False) -> None:
        if not correct_receiver and not allow_truth_keys:
            raise ValueError(
                "LeakyReceiver（槽末观测当下一步）只允许在 allow_truth_keys=True（诊断/反例）下构造，"
                "不得进入任何结果表。"
            )
        if exact_predictor:
            # ★★ 2026-10-22 18:xx 撤回（见 `experiments/cw_v1/pred_gates.py` 与
            #   `docs/CW-R1.1-纠错交付-2.md`）：这个"精确预测器"**不是精确的**。
            #   它把一步推进算成两步（先 u_{t-1} 再 u_t），并把过程噪声加了两次
            #   （A²σ_w² + σ_w²）。零噪声 HOLD 下名义预测器逐位为 0，
            #   它却从第 2 步起给出 **B·K·x₀ = 0.599666** 的偏差
            #   （BOUNDED，x0=[0.7,-0.4]；解析值与实测差 1e-16，见 pred_gates.P7）。
            #   ⇒ 直接拒绝构造，不留"打开就变精确"的错觉。
            raise ValueError(
                "exact_predictor 已撤回：本环境的物理推进就是一步一次噪声，"
                "名义预测器 x̂'=A x̂+B u、P'=A²P+σ_w² 才是**精确**的一步预测"
                "（零噪声逐位 0，独立 Kalman 复算逐位 0）。见 pred_gates.py。"
            )
        self.cfg = cfg
        self.rngs = rngs
        self.stream_ids = stream_ids
        self.allow_truth_keys = bool(allow_truth_keys)
        self.correct_receiver = bool(correct_receiver)
        #: ★ 只由 `runner.run_episode(predictor="exact")` 置真；默认 False ⇒ 历史数值不变
        self.exact_predictor = False   # ★ 2026-10-22 撤回：True 已在上面直接拒绝
        self.rho = float(rho if rho is not None else cfg.channel.rho_panel[0])
        #: ★★ 外生事件的索引方式（`config.ProtocolConfig.event_pairing`）
        self.pairing = str(cfg.protocol.event_pairing)

        n = cfg.plant.n_users
        self.n_users = n
        self.action_space = action_space(n)

        # ★ 真值：只在内部
        #    v1.1：从**公开先验**采样，而不是确定取 0。
        #    旧版 `x_0 ≡ 0` 而策略被告知先验 `N(0, prior_var)` ⇒ 策略"以为在猜、
        #    其实精确知道"，首次观测前的决策被系统性高估（2026-10-22 验收判为缺陷）。
        self._x_true = np.zeros(n, dtype=float)
        self._log_h_true = np.zeros(n, dtype=float)
        self.init_record: dict = {}
        self._sample_init()

        # 接收端信念
        self._x_hat = np.zeros(n, dtype=float)
        self._P_hat = np.full(n, float(cfg.plant.prior_var))
        self._stamp = np.full(n, -1, dtype=int)

        # 信道信念
        self._m_h = np.zeros(n, dtype=float)
        self._v_h = np.full(n, float(cfg.channel.h_log_init_std) ** 2)
        self._csi_stamp = np.full(n, -1, dtype=int)
        #: ★★ R1.12：信道信念的**信息模型**（`"approx"` = 一阶伪观测，默认；`"exact"` = 网格精确贝叶斯）
        #:   `"approx"` 是历史全部证据的口径 ⇒ 默认值**不许**改，改了旧 16 格就不再逐位复现。
        self.ack_mode = str(getattr(cfg.channel, "ack_mode", "approx"))
        if self.ack_mode not in ("approx", "exact"):
            raise ValueError(f"ack_mode={self.ack_mode!r} 未实现（只允许 'approx' / 'exact'）")
        #: exact 模式的网格后验；`None` ⇒ 走 `_ack_bump` 的老路径（逐位不变）
        self._grid: GridBelief | None = None
        #: ★★ R1.13：一阶近似的伪观测方差 —— 从配置读，默认 = 类常量 `ACK_VAR`（0.63）。
        #:   ★ 为什么保留类常量：`gates 4j-1/4p` 用 `CWWorld.ACK_VAR` 做解析对账，
        #:     删掉常量就等于删掉那两条闸的参照物；`4r` 钉死"默认值 == 常量"。
        self.ack_var = float(getattr(cfg.channel, "ack_var", type(self).ACK_VAR))
        if not (self.ack_var > 0.0):
            raise ValueError(f"ack_var={self.ack_var} 必须为正")
        if self.ack_mode == "exact":
            self._grid = GridBelief(
                n_users=n, rho=self.rho,
                kappa=float(cfg.channel.kappa), log_h_star=float(cfg.channel.log_h_star),
                init_mean=0.0, init_std=float(cfg.channel.h_log_init_std))
            _gm, _gv = self._grid.moments()
            self._m_h = np.asarray(_gm, dtype=float).copy()
            self._v_h = np.asarray(_gv, dtype=float).copy()

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
        #: ★★ R1.11 新增（**纯记录**）：逐用户的 ACK / NACK 归属计数。
        #:   存在的唯一理由：`Σ_i 计数 = 总数` 这类**聚合恒等**对"记到了谁头上"恒等不可见
        #:   （R1.10：`Σ NACK = 77` 全对，而 u0 的 71 次真实失败一次也没记到 u0）。
        #:   ⇒ 必须把**逐用户**这一维显式建成可观测量，否则归属缺陷永远只能靠插桩发现。
        #:   不进决策、不消耗随机数、不改任何既有字段。
        self.n_ack_by_user = [0] * n
        self.n_nack_by_user = [0] * n
        self.control_history: list[np.ndarray] = []
        self.records: list[SlotRecord] = []
        #: ★ R1.8：逐槽的**逐用户**任务陈旧度快照 `(task_aoi_0, ..., task_aoi_{n-1})`，
        #:   与 `view().telemetry[i]["task_aoi"]` 逐位同式（`-1` = 该用户从未送达）。
        #:   **纯记录**：不参与决策、不消耗随机数、不改任何记账量。
        self._aoi_hist: list[tuple[int, ...]] = []
        #: ★★ R1.14：逐槽的**真实** `log h`（`(log_h_0, ..., log_h_{n-1})`），时点 = 槽开始、
        #:   即与本槽 `view().telemetry[i]["h_mean"/"h_var"]` **同一时刻**。
        #:   存在的唯一理由：标定量 `R_var = mean((H−m)²)/mean(v)` 需要真值，而真值不许进视图。
        #:   **纯记录**：不参与决策、不消耗随机数、不改任何记账量（闸 `4x` 钉住）。
        #:   ★ 记录位置在 `truth_override` **之后** —— 记的是"本槽真的被读到的那份真值"（闸 `4w`）。
        self._log_h_hist: list[tuple[float, ...]] = []
        #: ★ v1.1：与 `records` 同长，但只含白名单字段 —— 视图只引用这一份，
        #:   这样"视图里能读到什么"由**追加处**决定，不依赖读的人自觉过滤。
        self._public_records: list[dict] = []
        #: 反例 1 的真值替换：{"x": array} / {"h": array} / None
        self.truth_override: dict | None = None
        self._w_last = np.zeros(n, dtype=float)
        self._K = np.array([self.nominal_gain(i) for i in range(n)])

        # ★★ 外生事件的预生成（只在 pairing="slot" 下；见 config.ProtocolConfig.event_pairing）
        #    形状 (horizon+1, n)：下标就是 (t, user)。
        #    `+1` 只是为越界留一格；正常 loop 的 t 落在 [0, horizon-1]。
        self._z_meas: np.ndarray | None = None
        self._z_csi: np.ndarray | None = None
        self._u_packet: np.ndarray | None = None
        if self.pairing == "slot":
            H = int(cfg.protocol.horizon) + 1
            self._z_meas = self.rngs["meas"].standard_normal((H, n))
            self._z_csi = self.rngs["csi"].standard_normal((H, n))
            self._u_packet = self.rngs["packet"].random((H, n))

    # ---------------- 外生抽样（两种索引方式）
    def _draw_meas(self, i: int) -> float:
        """任务观测噪声。`slot` ⇒ 由 (t,i) 决定；`attempt` ⇒ 按发送次数取流。"""
        if self._z_meas is not None:
            return float(self._z_meas[self.t, i])
        return float(self.rngs["meas"].standard_normal())

    def _draw_csi(self, i: int) -> float:
        if self._z_csi is not None:
            return float(self._z_csi[self.t, i])
        return float(self.rngs["csi"].standard_normal())

    def _draw_packet(self, i: int) -> float:
        """发包成败用的均匀数。`slot` ⇒ 由 (t,i) 决定 ⇒ 两策略在同 (t,i) 上同一枚硬币。"""
        if self._u_packet is not None:
            return float(self._u_packet[self.t, i])
        return float(self.rngs["packet"].random())

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

    # ---------------- 初态（v1.1）
    def _sample_init(self) -> None:
        """按**公开先验**采样真初态。

        公开先验（所有方法共享、也写进 `telemetry` 的初始 `P_hat`/`h_var`）：
            `x_i(0) ~ N(0, prior_var)`，`log h_i(0) ~ N(0, h_log_init_std²)`。

        ★ 为什么单独一条 `init` 子流：`SeedSequence.spawn(n)` 的第 i 个子序列
          与 n 无关，所以把 `init` **追加在最后**不会挪动 plant/channel/meas/csi/packet
          的 spawn_key ⇒ 旧证据的随机序列仍能原样重放。
        ★ `init_from_prior=False` 时**不消耗**该子流的任何数，真值保持 0 ——
          这是复现 v1 旧数字的唯一开关，不进新结果表。
        """
        n = self.n_users
        if not self.cfg.protocol.init_from_prior:
            self.init_record = {"mode": "zero", "x0": [0.0] * n, "h0": [0.0] * n}
            return
        g = self.rngs["init"]
        x0 = g.standard_normal(n) * float(np.sqrt(self.cfg.plant.prior_var))
        h0 = g.standard_normal(n) * float(self.cfg.channel.h_log_init_std)
        self._x_true = np.asarray(x0, dtype=float)
        self._log_h_true = np.asarray(h0, dtype=float)
        self.init_record = {"mode": "prior",
                            "x0": [float(v) for v in x0],
                            "h0": [float(v) for v in h0]}

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

        # ★★ R1.14：纯记录本槽开始时刻的真 `log h`（在 override 之后 ⇒ 记的是"真的被读到的那份"）。
        self._log_h_hist.append(tuple(float(x) for x in self._log_h_true))

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
                                + cfg.plant.sigma_meas * self._draw_meas(i))

        # 3) 若是 PROBE：有噪 CSI 读数（y = log h_t + N(0, sigma_csi²)）
        csi_reading = None
        if act.kind == "PROBE":
            i = act.user
            csi_reading = float(self._log_h_true[i]
                                + cfg.channel.sigma_csi * self._draw_csi(i))

        # 4) 成败由**真信道**在 t 时刻决定（策略不可见）
        p_true = None
        delivered = 0
        ack: int | None = None
        if act.kind == "SEND":
            i = act.user
            p_true = p_success_from_log_h(float(self._log_h_true[i]), cfg)
            delivered = int(self._draw_packet(i) < p_true)
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
        # ★★ 2026-10-22 18:xx：「精确预测器」分支已**删除**。
        #   它原来在这里做「A(A x̂ + B u_{t-1}) + B u_t」并把噪声加两次，
        #   理由是"名义滚动漏了槽内的 A·w_t"。那个理由是错的：本环境的
        #   `_advance_physics()` **一个时隙只推进一次、只加一次 w**，
        #   所以名义滚动 `A x̂ + B u`、`A²P + σ_w²` 就是**精确**的一步预测。
        #   实测（`pred_gates.py`）：零噪声 HOLD 下名义误差逐位 0，被删掉的
        #   那条分支给出 0.599666 / 0.685619；有噪声时按用户 RMS 误差
        #   0.1042 → 0.1514（BOUNDED）、0.0494 → 0.1132（R0_DEFAULT）。
        #   ⇒ 它是"把控制量算两遍"的缺陷，不是更精确的预测器。

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
            # ★★ R1.11：逐用户计数 + 实参显式化成 `act.user`。
            #   旧写法 `_ack_bump(i, ...)` 在本支**恰好**正确（443 行刚把 i 赋成 act.user），
            #   但那是隐式依赖 —— 同一函数里 467 行就因为没有这层保护而错了 6 轮。
            self.n_ack_by_user[int(act.user)] += 1
            # ★ ACK 本身也是信息（v4 §3.3-4）：成功说明当时的 h 不差。
            #   但默认**不**用它更新信念 —— 见 config.use_ack_for_csi 的说明。
            if cfg.channel.use_ack_for_csi:
                self._ack_bump(int(act.user), +1.0)
        elif act.kind == "SEND":
            self.budget_wasted += spend
            self.n_nack += 1
            # ★★ R1.11 行为修复：旧版这里写 `self._ack_bump(i, -1.5)`，
            #   而 `i` 已被 428 行 `for i in range(self.n_users)` 泄漏成 `n_users-1`
            #   ⇒ 所有 NACK 的信道信念更新都落到下标最大的用户头上。
            #   n_users==1 时 n_users-1==0==act.user ⇒ 缺陷恒等不可见（R0 `4g` 就是单用户口径）。
            self.n_nack_by_user[int(act.user)] += 1
            if cfg.channel.use_ack_for_csi:
                self._ack_bump(int(act.user), -1.5)

        if act.kind == "PROBE":
            i = act.user
            self.budget_used += spend
            self.n_probe += 1
            obs_var = float(cfg.channel.sigma_csi) ** 2
            if self._grid is not None:
                # ★ R1.12：exact 模式下 PROBE 也必须进网格 ——
                #   否则下一步 `_sync_h_from_grid` 会把解调器的读数**覆盖掉**，
                #   于是两臂除了"ACK 的算法"还差"PROBE 有没有用"，变成双变量混淆。
                self._grid.observe_gauss(i, csi_reading, obs_var)
                self._sync_h_from_grid()
            else:
                g = self._v_h[i] / (self._v_h[i] + obs_var)
                self._m_h[i] = self._m_h[i] + g * (csi_reading - self._m_h[i])
                self._v_h[i] = (1.0 - g) * self._v_h[i]
            self._csi_stamp[i] = t
            # ★ 不触碰任务估计与时间戳（反例 3）
        elif act.kind == "HOLD":
            self.n_idle += 1   # ★ 不产生 ACK（反例 4）

        # 8) 信道信念随 AR 演化（所有用户；用真实 rho，不用面板首项）
        if self._grid is not None:
            # ★ R1.12：exact 模式用**网格卷积**做这次演化（闭式矩递推会与网格不一致）
            self._grid.propagate()
            self._sync_h_from_grid()
        else:
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
            y_t=(send_sample if delivered else None), p_true=p_true,
            w_realized=tuple(float(x) for x in self._w_last),
            cost_est=cost_est, cost_true=cost_true,
            cost_est_by_user=cost_est_by_user, cost_true_by_user=cost_true_by_user,
            est_norm=float(np.linalg.norm(self._x_hat)),
            true_norm=float(np.linalg.norm(self._x_true)),
            budget_left=self.budget_total - self.budget_used,
        )
        self.records.append(rec)
        self._public_records.append({f: getattr(rec, f) for f in PUBLIC_HISTORY_FIELDS})
        # ★ R1.8：逐槽快照（与 `view().telemetry[i]["task_aoi"]` 同式）。纯记录。
        self._aoi_hist.append(tuple(
            (int(self.t) - int(self._stamp[i])) if int(self._stamp[i]) >= 0 else -1
            for i in range(self.n_users)))
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
    #: ★ R1.13 起这只是**默认值**：真正的取值来自 `cfg.channel.ack_var`（`self.ack_var`），
    #:   两条必须相等，由闸 `4r` 钉死；保留常量是为了给 `4j-1/4p` 提供解析参照物。
    ACK_VAR = 0.63   # ≈ π²/3·0.2，对应 1 次二值结果只看清 ±1.5 个 log 单位

    def _ack_bump(self, i: int, direction: float) -> None:
        """用 ACK/NACK 更新用户 i 的信道信念（高斯近似，二值观测）。

        ★ 这不是精确贝叶斯（精确做法要对 `log h` 的连续后验乘上
          `p_succ` 或 `1-p_succ` 的似然并归一化），而是**一阶近似**：
          把"成功/失败"折算成一条方差为 `ACK_VAR` 的伪观测。
          之所以先用近似：R0 只要求 ACK **进入**信念更新且方向正确（反例 4g），
          精确贝叶斯是 R1 的事，写在这里以免被误当成已实现。

        ★★ R1.12：精确版已实现，但**不替换这里**，而是由 `ack_mode` 分派 ——
          `"approx"`（默认）逐位走下面这段老代码，`"exact"` 走网格后验。
          调用点（SEND/ACK 支与 SEND/NACK 支）一字未改，两臂只差信息模型。

        ★★ R1.13：伪观测方差改从 `self.ack_var`（配置项）读，**默认 = `ACK_VAR`（0.63）**
          ⇒ 默认路径逐位不变（由 `4p` 与 G1 身份闸双封）。方向量级 `±1.0/−1.5` 仍是字面常量，
          本轮不允许动（`4t` 钉住）。
        """
        if self.ack_mode == "exact":
            self._ack_bump_exact(i, direction)
            return
        prior_var = float(self._v_h[i])
        y = float(self._m_h[i]) + direction    # 伪观测落在先验附近 ±direction
        g = prior_var / (prior_var + self.ack_var)
        self._m_h[i] = self._m_h[i] + g * (y - self._m_h[i])
        self._v_h[i] = (1.0 - g) * prior_var

    # ---------------- R1.12：精确贝叶斯支（网格后验）
    def _ack_bump_exact(self, i: int, direction: float) -> None:
        """精确贝叶斯：ACK(`direction>0`) / NACK(`direction<0`) 作为伯努利似然乘上去。

        ★ 与一阶近似的三处结构性差别（预注册 §2）：
          ① 位移幅度依赖 `(m, v)`，不是只依赖 `v`；
          ② 后验方差收缩量 = Fisher 信息 `κ²p(1−p)` 决定的量，而不是常数 `ACK_VAR`；
          ③ 后验可以非高斯（`_m_h/_v_h` 只是它的矩）⇒ "偏度"第一次进入决策链。
        """
        if self._grid is None:      # pragma: no cover - 构造函数已保证
            raise RuntimeError("ack_mode='exact' 但网格后验未初始化")
        self._grid.observe_binary(int(i), bool(float(direction) > 0.0))
        self._sync_h_from_grid()

    def _sync_h_from_grid(self) -> None:
        """把网格后验的矩写回 `_m_h/_v_h` —— 这是 exact 模式对外**唯一**的暴露面。

        ★ `_m_h/_v_h` 在 exact 模式下是**导出量**，不再是独立状态：
          任何绕过网格去改它们的代码都会在下一次 sync 被覆盖（这正是我们要的）。
        """
        if self._grid is None:      # pragma: no cover
            return
        m, v = self._grid.moments()
        self._m_h = np.asarray(m, dtype=float)
        self._v_h = np.asarray(v, dtype=float)

    @property
    def aoi_history(self) -> list[tuple[int, ...]]:
        """★ R1.8 新增（**纯记录**）：逐槽的逐用户任务陈旧度，`-1` = 该用户当时尚未送达过。

        与 `view().telemetry[i]["task_aoi"]` 逐位同式；长度 = `len(self.records)`。
        ★ 这是**评估侧**量，不进 `SchedulerView._pub`：视图能读到什么由 `view()` 的白名单决定，
          与这个属性无关（它挂在 `CWWorld` 上，策略拿不到 `CWWorld`）。
        """
        return self._aoi_hist

    @property
    def log_h_history(self) -> list[tuple[float, ...]]:
        """★ R1.14 新增（**纯记录**）：逐槽、逐用户的真实 `log h`，时点 = 槽开始。

        与 `view().telemetry[i]["h_mean"]/["h_var"]` 同一时刻 ⇒ 可直接算
        `z = (H − m)/sqrt(v)`。★ 这是**评估侧**量：不进 `SchedulerView._pub`，
        策略拿不到 `CWWorld`，所以拿不到它。
        """
        return self._log_h_hist

    def view(self) -> "SchedulerView":
        """构造**公开信息快照**。不含 self，也不含任何真值字段。

        ★ 这是 v1.1 的信息边界实现：边界不靠"藏引用"，靠"只拷公开数值"。
          即使策略把 `view._pub` 全打印出来，也只有这些键。
        """
        tel = []
        for i in range(self.n_users):
            stamp = int(self._stamp[i])
            csi_stamp = int(self._csi_stamp[i])
            tel.append({
                "i": i,
                "x_hat": float(self._x_hat[i]),
                "P_hat": float(self._P_hat[i]),
                "stamp": stamp,
                "task_aoi": (self.t - stamp) if stamp >= 0 else -1,
                "h_mean": float(self._m_h[i]),
                "h_var": float(self._v_h[i]),
                "csi_stamp": csi_stamp,
                "csi_aoi": (self.t - csi_stamp) if csi_stamp >= 0 else -1,
            })
        return SchedulerView({
            "t": int(self.t),
            "n_users": int(self.n_users),
            "budget_left": int(self.budget_total - self.budget_used),
            "budget_total": int(self.budget_total),
            "prior": {
                "x_mean": 0.0,
                "x_var": float(self.cfg.plant.prior_var),
                "h_log_mean": 0.0,
                "h_log_var": float(self.cfg.channel.h_log_init_std) ** 2,
            },
            "telemetry": tel,
            "history": tuple(self._public_records),
        })


class BudgetExhausted(RuntimeError):
    """主动动作被预算拒绝。★ 这是协议行为，不是程序错误。"""


class SchedulerView:
    """调度器能看到的**全部**信息 —— 一个纯快照，不含任何世界句柄。

    ★★ v1.1 重新设计（2026-10-22 验收缺陷）：旧版持有 `__w` 指向 `CWWorld`。
      `__slots__ = ("__w",)` 只是**名字改写**（`_SchedulerView__w`），
      `view._SchedulerView__w._x_true` 一步就能读到真值 —— 反例 1k 就是这么抓到的。
      "Python 里用下划线藏引用"不是信息边界，改名不是安全机制。
      ⇒ 现在 `CWWorld.view()` 只传**公开数值的快照**；即便把 `_pub` 整个打印出来，
        里面也没有任何真值字段（1i 用白名单逐字段验）。
    ★ 快照的构造代价是每槽 O(n_users)，`history` 只是把已追加的公开记录元组化。
    """

    __slots__ = ("_pub",)

    def __init__(self, pub: dict) -> None:
        self._pub = pub

    @property
    def t(self) -> int:
        return self._pub["t"]

    @property
    def n_users(self) -> int:
        return self._pub["n_users"]

    @property
    def budget_left(self) -> int:
        return self._pub["budget_left"]

    @property
    def budget_total(self) -> int:
        return self._pub["budget_total"]

    @property
    def prior(self) -> dict:
        """**公开先验**：所有方法共享，且与真初态的采样分布一致（v1.1）。"""
        return self._pub["prior"]

    @property
    def telemetry(self) -> list[dict]:
        """逐用户可部署状态：全部由"收到的观测 + 已执行动作 + ACK"推出。"""
        return self._pub["telemetry"]

    @property
    def history(self) -> tuple[dict, ...]:
        """已发生的记账 —— **白名单**输出（见 `PUBLIC_HISTORY_FIELDS`）。

        ★★ 为什么是白名单：旧版是黑名单 `{k: v for k, v in r.__dict__.items()
          if k not in (...)}`。黑名单的失效模式是"加字段就泄漏"：`SlotRecord`
          后来加了 `w_realized`（实现的过程噪声）与 `cost_true_by_user`（逐用户真成本），
          都不在旧的 4 元黑名单里 ⇒ 通过 `view.history` 直接可读。
        ★ `y_t` 只在**送达**时非 None（未送达的 SEND 没有样本到接收端，
          那时它是 `x_t + 噪声`，公开它等于绕过整个信道）。
        ★ `csi_reading` 在白名单里是**对的**：那是调度器自己花钱测的读数。
        """
        return self._pub["history"]

    def __getattr__(self, name: str):
        raise AttributeError(
            f"SchedulerView 没有属性 {name!r}。真状态/真信道/未来随机数不向调度器开放。"
        )


#: 视图对外声明的全部成员（闸用它做"没有多余入口"的对照）
VIEW_MEMBERS: tuple[str, ...] = (
    "t", "n_users", "budget_left", "budget_total", "prior", "telemetry", "history",
)
