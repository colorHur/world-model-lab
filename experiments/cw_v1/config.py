# -*- coding: utf-8 -*-
"""CW-POMDP-v1 配置：所有常量集中在此，不散落在代码里。

★ 本文件里的数字全部是**设计预算**，不是实测结果。任何报告引用时必须标注来源文件。
★ 首轮不做扫描：`CHANNEL_PANEL` 是**预先声明的开发面板**，不得按收益来回移动。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass(frozen=True)
class PlantConfig:
    """任务（被控对象）配置。标量解析系统，用于验证协议而非证明学习必要性。

    动力学：`x_{i,t+1} = A_i * x_{i,t} + B * u_{i,t} + w_{i,t}`，
    `w_{i,t} ~ N(0, sigma_w_i^2)`。B 固定为 0.1（v4 §3 / 首轮卡给定量）。
    """

    A: tuple[float, ...] = (1.01, 1.03)
    B: float = 0.1
    sigma_w: tuple[float, ...] = (0.02, 0.03)
    #: 初态共同先验 N(0, prior_var)，所有方法共享
    prior_var: float = 1.0
    #: 送达观测的噪声标准差（任务状态量测）
    sigma_meas: float = 0.01
    #: 代价权重：cost = q * sum_i x_i^2 + r * sum_i u_i^2（不按最终收益调权）
    q_cost: float = 1.0
    r_cost: float = 0.1

    @property
    def n_users(self) -> int:
        return len(self.A)


#: ★★ 开发 plant 的两个预设（2026-10-21 18:0x 实测判定后新增，判据见
#: `experiments/cw_v1/diag_plant_stability.py`）。
#:
#: `R0_DEFAULT` = 上面 dataclass 的默认值 `A=(1.01,1.03)`：**开环不稳定**。
#:   `x_0` 确定 = 0（见 `CWWorld.__init__` 的 `_x_true = zeros`，`prior_var` **只**进
#:   接收端信念、不进真值），`E[x_t²] = σ_w²·Σ A^{2(t-1-k)}` 按 `A^{2t}` 指数爆
#:   ⇒ 256 步 `HOLD` 成本末端 15627.85 / 中段 8.10 = **1929 倍**，均值 763~3998 完全由
#:   最后几十步决定。在这种 plant 上"平均任务成本"不是可比较量，任何基线比较都是噪声。
#:   它只能用作「协议载体验收」的环境（R0 的 49 项闸就够用），**不得**用来做方法比较。
#:
#: `BOUNDED` = `A=(1.00,1.00)`（临界稳定）+ `sigma_meas=0.20` + `B=0.32` + `r_cost=0.02`。
#:
#:   ★ 2026-10-21 18:1x 剂量实测定档（40 episode，dev 100–139，rho=0.85，B/T=0.8）：
#:     hold 0.1639 / paced1 0.0377 / paced16 0.0536（逐 episode 37/40 优于 hold，
#:     中位比 0.302）⇒ **SEND 确实有价值，且基线之间的差异不是噪声**。
#:
#:   ★★ 为什么改 `B` 与 `r_cost`（这两个 R0 之前没人动过）：
#:     R0 的 `B=0.1, r=0.1` 下 Riccati 解出 `K≈2.79/2.98`、`BK≈0.28-0.30` ⇒
#:     控制器每步只抵消 30% 误差，开环虽稳定但**闭环接近开环** ⇒ 送达观测几乎
#:     无法通过控制改善任务成本；实测（A=(1.0,1.0), sigma_meas=0.2）加 `B=0.32`
#:     + `r_cost=0.02` 后 `BK` 才接近 1（"能压住"）。这两个数是**设计值**，
#:     不是从任何真实系统拟合的。
#:
#:   ★ 必须同时报告的三条（R13/P3-1 纪律）：
#:     ① `HOLD` 在这套 plant 上会**漂移**（A=1 随机游走，方差线性增长，末/前 64 步比
#:        7.43 见 D4）⇒ 不许把 HOLD 当"零成本"基线；
#:     ② `paced16` 与 `paced1` 差距 1.42×，说明**发送频率**在这里是真实权衡；
#:     ③ 面板主表以 `cost_true_per_step` 的**中位数**为主、均值并报（`A=1` 时均值被
#:        长尾拖），另报逐 user 与逐 episode 比值。
BOUNDED_PLANT = PlantConfig(
    A=(1.00, 1.00), B=0.32, sigma_w=(0.02, 0.03), sigma_meas=0.20,
    q_cost=1.0, r_cost=0.02,
)



def bounded_plant(**overrides: Any) -> PlantConfig:
    """`BOUNDED` 预设 + 覆盖（用于剂量实验，不改动 dataclass 的默认值）。"""
    import dataclasses as _dc

    return _dc.replace(BOUNDED_PLANT, **overrides)


@dataclass(frozen=True)
class ChannelConfig:
    """外生信道：每个用户一条独立 AR(1) 幅值过程。

    ★ 这是**开发面板的抽象**，不是真实 5G 拟合参数（v4 §3.1 明令不得如此称呼）。

    `log h_{i,t+1} = rho * log h_{i,t} + sqrt(1-rho^2) * s_i`, `s_i ~ N(0, 1)`。
    稳态 `log h ~ N(0,1)` ⇒ 中位 h=1、`P(h>0.5)=0.7572`。
    发包成功概率 `p_succ = sigmoid(KAPPA * (log h - LOG_H_STAR))`。
    归一化方式刻意选择为「不依赖 rho」，这样不同 rho 档的**平均成功率可比**。

    ★★ 稳态标准差（2026-10-22 更正）：**就是 1.0，且与 rho 无关**。
      因为更新式已归一（新息系数 `sqrt(1-rho^2)`、新息 sd 为 1），
      `Var(∞) = (1-rho²)/(1-rho²)·1 = 1`。
      旧 STATE 里写的「ρ=0.98 时稳态 sd 应为 5.03」是**把未归一 AR 的公式
      `1/sqrt(1-rho²)` 套到已归一的式子上**算出来的，错。5.025 只对
      `log h_{t+1} = rho·log h_t + s_t`（新息 sd=1、无 `sqrt(1-rho²)` 因子）成立。
      ⇒ `h_log_init_std` 不是"一个可调参数"，它是**该稳态 sd 的声明值**，
        因此必须恒等于 1.0；`protocol.v11_validate()` 会检查这一点。
    """

    rho_panel: tuple[float, ...] = (0.85, 0.98)
    kappa: float = 2.0
    log_h_star: float = 0.0
    #: 探测测量的对数信道噪声标准差（只影响 PROBE 读到的值，不影响真实 h）
    sigma_csi: float = 0.05
    #: 真值 h 的初始条件：稳态分布 log h ~ N(0,1)（所有方法共享同一公开先验）
    h_log_init_std: float = 1.0
    #: ★ ACK/NACK 是否用于更新**信道信念**。
    #: v1.1 起默认 **True**：v4 §3.3 要求 ACK 是合法信息，任何部署方法都该能用它。
    #: 旧版默认 False 的唯一理由是"为了让反例 1d（只改真信道 ⇒ 动作不变）通过"——
    #: 那是**关掉信息去迁就判据**，2026-10-22 验收判为不合理；1d 已改写成
    #: "公开历史相同 ⇒ 当前动作相同"（见 `gates.counterexample_1`）。
    #: 置 False 只用于单独对照"没有 ACK 通道"的世界，**不得**进主表。
    use_ack_for_csi: bool = True

    #: ★★ R1.12（2026-10-24）：ACK/NACK 更新信道信念的**信息模型**。
    #:
    #:   `"approx"`（默认，历史全部证据的口径）：把一次二值结果折算成一条
    #:     方差 `ACK_VAR=0.63` 的伪观测（`world._ack_bump`）—— 一阶近似，代码自承不精确。
    #:   `"exact"`：对 `log h` 的一维网格后验乘伯努利似然（`world.GridBelief`）。
    #:
    #:   ★ 为什么默认仍是 `"approx"`：R1.7.6/R1.8/R1.11 的 16 格逐位身份都建立在这条路径上，
    #:     改默认值 = 旧证据全部失效。精确版是**另一条臂**，只在新证据目录里出现。
    #:   ★ 两臂的差别**仅限信息模型**：动作空间、预算、外生事件、规划器、目标函数都不变。
    ack_mode: str = "approx"

    #: ★★ R1.13（2026-10-24）：一阶近似里那条伪观测的**等效量测方差**（唯一标量自由度）。
    #:
    #:   只有 `ack_mode="approx"` 读它；`exact` 模式走网格，本字段无作用（但保留以便并列配置）。
    #:   ★ 默认 **必须** 等于 `world.CWWorld.ACK_VAR`（=0.63）：R1.7.6→R1.12 的全部 16 格
    #:     逐位身份都建立在这条路径上，默认值一改旧证据全部失效（闸 `4r` 钉死这条等式）。
    #:   ★ 本轮只允许调这个标量，**不许**顺手动 `±1.0/−1.5` 的方向量级（那是另一件事，闸 `4t` 钉住）。
    ack_var: float = 0.63


@dataclass(frozen=True)
class ProtocolConfig:
    """协议与预算。"""

    horizon: int = 256
    c_probe: int = 1
    c_data: int = 1
    #: 预算档：floor(0.5T) / floor(0.8T)
    budget_ratios: tuple[float, ...] = (0.5, 0.8)

    #: ★★ v1.1（2026-10-22）：真初态是否从**公开先验**采样。
    #:   旧版真值 `x_0 ≡ 0`、`log h_0 ≡ 0`（确定值），而策略被告知先验是
    #:   `N(0, prior_var)` / `N(0, h_log_init_std²)` ⇒ **公开先验与真实初态不一致**：
    #:   策略以为自己在猜，其实是精确已知 0。这会让"首次观测前"的所有决策被系统性高估。
    #:   True = 用 `init` 子流按公开先验采样（v1.1 默认，进新证据目录）；
    #:   False = 复现 v1 旧证据的确定初态（**只用于重放，不进新结果表**）。
    init_from_prior: bool = True

    #: ★★ v1.1-2（2026-10-22 18:xx）：**外生事件按什么索引**。
    #:
    #:   `"slot"`（默认，新证据一律用它）：`packet` / `meas` / `csi` 三条子流在
    #:     episode 构造时**一次性预生成** `(horizon, n_users)` 矩阵，取值下标是
    #:     `(t, user)`。于是"第 t 槽、第 i 个用户的成败/量测噪声"**只由 (seed, split,
    #:     episode, t, i) 决定，与策略做过什么无关** ⇒ 两策略在同一 (t,i) 上如果都
    #:     发了，看到的是**同一个**成败与同一条量测噪声。这才是"同一轨迹配对"。
    #:
    #:   `"attempt"`（v1.1 前 18:xx 的全部证据都是这个）：一次 SEND 才从 `packet`
    #:     流里取一个数。于是发 128 次的臂与发 64 次的臂读的是流的**不同位置** ——
    #:     分布相同，但**不能**声称"两臂的丢包序列逐时隙相同"，也不能把两次 SEND
    #:     的成败当成同一枚硬币的两次投掷来配对。
    #:
    #:   ★ 为什么默认改成 `"slot"`：配对是"同轨迹比较"的定义前提，不是可选项。
    #:     代价是**数值不再与 18:xx 前的 v1.1 证据逐位相同**（分布相同、实现不同），
    #:     因此 `attempt` 保留为显式开关，只用于重放旧数字，**不得与新表混**。
    #:   ★ 归属已核：改这条**只**改外生随机数的取值方式，不改动力学、预算、代价定义。
    event_pairing: str = "slot"

    #: ★ v1.1 只支持 `c_probe == c_data == 1`。多时隙配置（如 (2,1)）在本版
    #:   有已知的"标称 T 槽实际 T·n_substeps 槽"缺陷：horizon 按**动作数**计而
    #:   物理时间按子步计，两者不等价 ⇒ 不同配置的"平均每步成本"不可比。
    #:   `v11_validate()` 会显式拒绝非 (1,1) 的配置，而不是让它安静地跑出错误对比。
    #:   修法设计见 `docs/research/CW-R1-有限视界参考-预注册-v0.1.md` §4
    #:   （固定物理时间窗 + 逐子步计成本 + 动作完成才揭示测量），修好前不得启用。
    allow_multislot: bool = False

    def budget(self, ratio: float) -> int:
        """`floor(ratio * T)`，用有理数整数运算实现以避免浮点误差。

        ★ 必须走分数：`0.8 * 256 = 204.80000000000001`，而 `(256*4)//5 = 204`。
          两者此处恰好同值，但换成 T=250 就会分叉，所以从一开始就不用浮点乘法。
        """
        from fractions import Fraction

        frac = Fraction(str(ratio))  # '0.5' -> 1/2，'0.8' -> 4/5（字符串入口保证十进制语义）
        if frac not in (Fraction(1, 2), Fraction(4, 5)):
            raise ValueError(f"未预注册的预算档 {ratio}（只允许 0.5 / 0.8）")
        return (self.horizon * frac.numerator) // frac.denominator


@dataclass(frozen=True)
class CWConfig:
    plant: PlantConfig = field(default_factory=PlantConfig)
    channel: ChannelConfig = field(default_factory=ChannelConfig)
    protocol: ProtocolConfig = field(default_factory=ProtocolConfig)
    #: 随机流的根种子（一个整数，由 SeedSequence 展开为子流，见 streams.py）
    root_seed: int = 20261021

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["protocol_name"] = "CW-POMDP-v1.1"
        d["budget_values"] = [self.protocol.budget(r) for r in self.protocol.budget_ratios]
        return d

    #: 协议版本号。写进 run-id 与证据目录，禁止用 v1 与 v1.1 的数字混表。
    PROTOCOL_VERSION = "CW-POMDP-v1.1"

    def v11_validate(self) -> list[str]:
        """v1.1 的**结构**自洽检查。返回问题清单（空 = 通过）。

        ★ 这些不是"数值容差"，是**定义级恒等式**：
          ① 归一 AR(1) 的稳态方差 = 1.0（新息系数 `sqrt(1-rho²)`、新息 sd=1）；
             写成 `1/(1-rho²)` 是把**未归一**的 AR 公式套过来，与更新式矛盾。
          ② `h_log_init_std` 是上面这个稳态 sd 的**声明值**，不是自由参数 ⇒ 必须 = 1.0。
             若它 ≠ 稳态 sd，信道信念的初始方差与真信道分布不一致，
             `P(h>h*)` 之类的概率从一开始就是错的。
          ③ 初态先验 `prior_var` 与 `init_from_prior=True` 必须同时成立，
             否则又回到"策略以为在猜、其实精确已知 0"。
          ④ 多时隙在 v1.1 未修，`(c_probe, c_data) != (1,1)` 必须被拒绝。
          ⑤ 估计协方差的地板：`P_hat ≥ sigma_w²`（过程噪声方差，由 `A²P+σ_w²` 决定），
             与 Riccati 的稳态解 `P_ss` 不是同一个量 —— 后者是**控制代价**的稳态值，
             随 `q/r` 变；前者是**估计误差**的稳态值，与 `q/r` 无关。混用会让
             "估计已经收敛到什么程度"被系统性夸大或低估。
        """
        p, ch, pr = self.plant, self.channel, self.protocol
        bad: list[str] = []

        # ① 用**与实际代码同一个递推**算稳态方差，而不是背公式：
        #    `world._v_h ← rho²·v + (1-rho²)`。从 v=1 出发迭代 ⇒ 应恒为 1。
        for rho in ch.rho_panel:
            v = 1.0
            for _ in range(500):
                v = rho * rho * v + (1.0 - rho * rho)
            if abs(v - 1.0) > 1e-12:
                bad.append(f"rho={rho}：方差递推稳态 {v}，应为 1.0")
        if abs(float(ch.h_log_init_std) - 1.0) > 1e-12:
            bad.append(
                f"h_log_init_std={ch.h_log_init_std} 必须 =1.0（归一 AR 的稳态 sd）；"
                f"旧写法 1/sqrt(1-rho^2) 只在未归一 AR 下成立")

        if pr.init_from_prior and float(p.prior_var) <= 0.0:
            bad.append(f"init_from_prior=True 要求 prior_var>0，实为 {p.prior_var}")
        if (not pr.init_from_prior) and float(p.prior_var) > 0.0:
            # 不是错，而是必须意识到：确定初态 + 非零公开先验 = 先验与真值不一致
            bad.append(
                "init_from_prior=False（复现 v1 旧证据）与 prior_var>0 并存 ⇒ "
                "公开先验与真实初态不一致，只允许用于重放旧数字")

        if (int(pr.c_probe), int(pr.c_data)) != (1, 1) and not pr.allow_multislot:
            bad.append(
                f"c_probe={pr.c_probe}, c_data={pr.c_data}：v1.1 只支持 (1,1)。"
                f"多时隙的时间轴未与 horizon 对齐（标 T 槽实际 T·n_substeps 物理步），"
                f"不同配置的平均成本不可比。修好前请保持 (1,1)。")

        # ⑥ 配对口径必须是已实现的两个之一；`slot` 的预生成矩阵按 (horizon, n) 定形，
        #    多时隙下 `t` 会在一个动作里推进多次 ⇒ 只在 (1,1) 下语义清楚。
        if pr.event_pairing not in ("slot", "attempt"):
            bad.append(f"event_pairing={pr.event_pairing!r} 未实现（只允许 'slot' / 'attempt'）")
        if pr.event_pairing == "slot" and (int(pr.c_probe), int(pr.c_data)) != (1, 1):
            bad.append("event_pairing='slot' 的预生成矩阵按 (horizon, n_users) 定形，只支持 (1,1)")

        for i, sw in enumerate(p.sigma_w):
            if float(sw) <= 0.0:
                bad.append(f"sigma_w[{i}]={sw} 必须为正（估计协方差地板依赖它）")

        # ⑦ ACK 信息模型必须是已实现的两个之一（R1.12）。写错的名字会静默退回近似口径。
        if ch.ack_mode not in ("approx", "exact"):
            bad.append(f"ack_mode={ch.ack_mode!r} 未实现（只允许 'approx' / 'exact'）")
        # ⑧ R1.13：伪观测方差必须为正有限值。写成 0/负数时 `g=v/(v+ack_var)` 会退化成
        #   "完全相信单次二值结果"（g=1）或**负增益**（g>1 ⇒ 信念反向），是静默的灾难。
        if not (float(ch.ack_var) > 0.0) or not math.isfinite(float(ch.ack_var)):
            bad.append(f"ack_var={ch.ack_var} 必须是正的有限值（伪观测方差）")
        return bad

    def fingerprint(self) -> str:
        """配置内容哈希：run-id 与证据记录的组成部分。"""
        import hashlib

        blob = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]


def default_config(**overrides) -> CWConfig:
    """构造默认配置；overrides 只接受 plant/channel/protocol/root_seed 级替换。"""
    base = CWConfig()
    if not overrides:
        return base
    kwargs: dict[str, Any] = {}
    for k, v in overrides.items():
        kwargs[k] = v
    return CWConfig(**kwargs)


# ============================================================ ★★ R1.14 研究基线
#: 研究基线（新证据，`results/cw_v1/cw_r1_14_*` 起）的 ACK 信息模型。
#:
#: 为什么是 `"exact"`（预注册 `docs/research/CW-R1.14-精确基线-预注册-v0.1.md` §2）：
#:   ① 精确贝叶斯**没有自由参数**；`"approx"` 有 `ACK_VAR=0.63` 与方向量级 `±1.0/−1.5`
#:      三个手调常数，其中方向比 1.5 在它**自己的标定点** `m=0` 就错 52%（真值 1.000）；
#:   ② `"approx"` 有"跑飞"通道：`p→1` 时精确 Fisher 信息 `I→0`，精确后验几乎不动，
#:      而 `"approx"` 仍按 `g·(+1)` 继续推均值、继续缩方差 ⇒ 越自信越推。R1.12 观测到的
#:      `p̂` 下限 1e-3 量级（以及 R1.10 的 `p̂→1.7e-7` 自锁）是这条通道的产物；
#:   ③ R1.13 已证明标量层面**治不好**（0/16 过、解析层面 `I ≤ 1.000` 不可达）。
#:
#: ★★ 为什么**不**把 `ChannelConfig.ack_mode` 的默认值直接改掉：
#:   `4j-1`/`4p` 用 `CWWorld.ACK_VAR` 做一阶近似的**解析对账**；R1.7.6→R1.13 的 32 格
#:   逐位身份也建立在那条路径上。默认值翻掉 = 旧证据全部失效 + 两条闸的参照物消失。
#:   ⇒ 默认值保持 `"approx"`，性质同 `protocol.init_from_prior=False`：
#:     **只用于重放旧证据的开关，不得进新表**。基线靠这个常量显式表达，不靠默认值。
MAINLINE_ACK_MODE = "exact"


def mainline_config(plant: PlantConfig | None = None, **overrides) -> CWConfig:
    """**研究基线配置**（R1.14 起）：`BOUNDED` plant + `ack_mode=MAINLINE_ACK_MODE`。

    ★ `plant` 默认取 `bounded_plant()`（`A=1` 有界档），**不是** dataclass 默认的
      `R0_DEFAULT`（`A=(1.01,1.03)` 开环不稳定）—— CW 线 R1 之后的全部面板（R1.7.6→R1.13）
      都在 `BOUNDED` 上跑，基线必须跟着面板走，否则"基线"与"面板"不是同一个世界。
    除 `channel.ack_mode` 外与同 plant 的默认配置逐字段相同（闸 `4u` 钉住）。
    新证据目录一律由它构造配置；`default_config()` 保留给历史身份重放。
    """
    base = default_config(plant=(bounded_plant() if plant is None else plant), **overrides)
    ch = type(base.channel)(**{**base.channel.__dict__, "ack_mode": MAINLINE_ACK_MODE})
    return CWConfig(plant=base.plant, channel=ch, protocol=base.protocol,
                    root_seed=base.root_seed)


