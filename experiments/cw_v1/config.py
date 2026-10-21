# -*- coding: utf-8 -*-
"""CW-POMDP-v1 配置：所有常量集中在此，不散落在代码里。

★ 本文件里的数字全部是**设计预算**，不是实测结果。任何报告引用时必须标注来源文件。
★ 首轮不做扫描：`CHANNEL_PANEL` 是**预先声明的开发面板**，不得按收益来回移动。
"""

from __future__ import annotations

import json
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
    """

    rho_panel: tuple[float, ...] = (0.85, 0.98)
    kappa: float = 2.0
    log_h_star: float = 0.0
    #: 探测测量的对数信道噪声标准差（只影响 PROBE 读到的值，不影响真实 h）
    sigma_csi: float = 0.05
    #: 真值 h 的初始条件：稳态分布 log h ~ N(0,1)（所有方法共享同一公开先验）
    h_log_init_std: float = 1.0
    #: ★ ACK/NACK 是否用于更新**信道信念**。
    #: R0 默认 False —— 理由见 `gates.counterexample_1` 的长注释：
    #:   一旦 ACK 进信念，"只改真信道 ⇒ 动作不变"（反例 1d）就不可能成立，
    #:   因为 ACK 是慢信道的合法结果。ACK→信念属于预测/学习（R2），不是 R0 协议验收面。
    #: 置 True 只用于单独验 ACK 的方向性（反例 4g）。
    use_ack_for_csi: bool = False


@dataclass(frozen=True)
class ProtocolConfig:
    """协议与预算。"""

    horizon: int = 256
    c_probe: int = 1
    c_data: int = 1
    #: 预算档：floor(0.5T) / floor(0.8T)
    budget_ratios: tuple[float, ...] = (0.5, 0.8)

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
        d["protocol_name"] = "CW-POMDP-v1"
        d["budget_values"] = [self.protocol.budget(r) for r in self.protocol.budget_ratios]
        return d

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
