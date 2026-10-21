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
