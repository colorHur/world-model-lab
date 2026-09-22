# -*- coding: utf-8 -*-
"""★ X32「物理层接入」：固定时隙下的速率—可靠权衡（M-QAM + 均匀量化）

================================================================= 为什么不是"SNR→丢包率"查表
如果只是把丢包率换成一个由 SNR 算出来的数，那对 X31 **零增量**
（X31 已经把任意 p̄ 都覆盖了一遍，而 X31-b 还证明了 p̄ 与突发结构解耦）。

本模块做的是**真 tradeoff**：

    每个时隙只能传 `N_sym` 个符号（带宽 × 时隙，固定）
    要传的状态有 `d` 维，每维 `b` 比特 ⇒ 一包 `n_bits = d·b` 比特
    ⇒ 每个符号必须承载 `m = n_bits / N_sym` 比特 ⇒ **调制阶数 M = 2^m**

    b ↑  ⇒  量化更精细（误差 ∝ 2^(−2b)）   但  m ↑ ⇒ 星座更密 ⇒ 误码率飙升 ⇒ 丢包变多
    b ↓  ⇒  星座抗噪（PER 低）             但  量化粗糙

★ 取 `N_sym = d` 时 `m = b`，于是 b=2/4/6/8 恰好是 **4/16/64/256-QAM** ——
  干净的教科书设定，且 b 与调制阶数一一对应，便于解释。

================================================================= ★ 公式来源与「待核」声明
方形 M-QAM 在 AWGN 下的**符号**错误率近似（Proakis《数字通信》 / Goldsmith
《无线通信》中的标准近似式）：

    P_s ≈ 4·(1 − 1/√M) · Q( √( 3·γ_s / (M − 1) ) )        γ_s = 每符号信噪比（线性）
    P_b ≈ P_s / log2(M)                                     （Gray 编码近似）
    PER = 1 − (1 − P_b)^(n_bits)                            （**无信道编码**）

⚠ **待核**：上面两式是凭教科书记忆写下的，本仓库没有做与仿真/标准表的核对。
   已做的自证只有以下几条（见 `tests` 第 40–42 项）：
     · M=4 时公式退化为 `P_b = Q(√γ_s)`，与 QPSK 的**精确**误码率一致
     · PER 对 M 单调增、对 γ_s 单调减、落在 [0,1]
    ⇒ 也就是说：**公式的定性结构（b↑⇒PER↑）可信，绝对数值待核**。
   本实验的结论（`b*` 的存在与移动方向）只依赖定性结构，不依赖绝对精度；
   但任何"在 SNR = X dB 时 PER = Y"的**定量**表述在核对前不得写进论文。

================================================================= 与既有实验的关系
★ 本模块**不改** `run_tracking` 的默认行为：
  · 丢包仍走 `schedule` 接口（这里复用 X31 的 Gilbert–Elliott，PER 当 p̄）
  · 量化走 `payload_fn` 可选参数（默认 None ⇒ 恒等 ⇒ X24/X26/X27/X30/X31 逐位不变）
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from .tracking import GilbertElliottChannel, Schedule


def qfunc(x: float) -> float:
    """Q(x) = 0.5·erfc(x/√2)（标准库实现，避免给仓库加 scipy 依赖）。"""
    return 0.5 * math.erfc(float(x) / math.sqrt(2.0))


def qam_symbol_error_rate(gamma_s: float, M: float) -> float:
    """方形 M-QAM 在 AWGN 下的符号错误率近似。见模块 docstring 的「待核」声明。"""
    if M <= 1:
        raise ValueError(f"M 必须 > 1，收到 {M}")
    g = float(max(gamma_s, 0.0))
    if g <= 0.0:
        return 1.0 - 1.0 / M          # 无信噪比 ⇒ 完全随机猜，错误率 = 1 − 1/M
    # ★ 近似式在"高阶调制 + 低 SNR"下会**溢出 1**（例如 256-QAM @15dB 算出 1.016）。
    #   物理上 SER 的上限是 1 − 1/M（完全随机猜，仍有 1/M 的概率猜中）⇒ 饱和到那里。
    #   ⚠ 这个饱和区间意味着"链路已不可用"，不是精确值 —— 见模块 docstring 的待核声明。
    arg = math.sqrt(3.0 * g / (M - 1.0))
    ser = 4.0 * (1.0 - 1.0 / math.sqrt(M)) * qfunc(arg)
    return min(ser, 1.0 - 1.0 / M)


#: ★ 近似式 `BER ≈ SER/log2(M)` 的有效上限（SER 超过它 ⇒ 换算不再成立）
#: 依据见 `qam_bit_error_rate` 的说明。
SER_VALID_MAX = 0.5


def qam_approximation_valid(gamma_s: float, M: float) -> bool:
    """`BER ≈ SER/log2(M)` 这个换算是否还成立。

    ★★ 为什么必须有这个判据（X32 实测踩出来的）
    该换算依赖"Gray 编码下**每次符号错误只错 1 个比特**" —— 这只在高 SNR 成立。
    低 SNR 时星座完全不可分辨，一个符号错误会错**多个**比特，真实 BER 趋向 0.5。

    失效症状（SNR=5dB）：64-QAM 与 256-QAM 的 SER 都饱和到 1−1/M，
    于是 BER = (1−1/M)/log2(M)：
        64-QAM  ⇒ 0.984/6 = 0.164
        256-QAM ⇒ 0.996/8 = **0.124**   ← 调制阶数越高，BER 反而越低？
    ⇒ PER 对 b **非单调**（0.99842 → 0.99831）。这是公式的人工产物，不是物理。

    ⇒ SER > 0.5 时判定为"公式失效"，该 (SNR, M) 组合按**链路不可用**处理。
    """
    return qam_symbol_error_rate(gamma_s, M) < SER_VALID_MAX


def qam_bit_error_rate(gamma_s: float, M: float) -> float:
    """由 SER 折算出 BER（Gray 编码，BER ≈ SER/log2 M）。

    ⚠ 只在 `qam_approximation_valid` 为真时才可信 —— 见那里的说明。
    """
    ser = qam_symbol_error_rate(gamma_s, M)
    return ser / max(math.log2(M), 1e-12)


def per_from_ber(ber: float, n_bits: float) -> float:
    """无信道编码时，n_bits 个比特里错一个就算整包错。"""
    b = float(min(max(ber, 0.0), 1.0))
    return 1.0 - (1.0 - b) ** float(max(n_bits, 0.0))


def snr_db_to_linear(snr_db: float) -> float:
    return 10.0 ** (float(snr_db) / 10.0)


# ---------------------------------------------------------------- 量化
class UniformQuantizer:
    """每维独立的均匀（mid-rise）量化器。

    Args:
        lo, hi: 量程（标量或 per-dim 数组）。
            ★ 量程必须由**训练数据**统计得到，不能拍脑袋 —— 过载（超出量程被裁剪）
            会引入一个与 b 无关的误差地板，把 b 的效应吃掉。
        bits: 每维比特数 b ⇒ 2^b 个电平，步长 Δ = (hi−lo)/2^b

    理论量化噪声功率（每维）≈ Δ²/12 —— **仅在无过载、且信号在电平间均匀分布时成立**。
    实测值由 `measure_mse` 给出，脚本应当报**实测**值而不是这个式子。
    """

    def __init__(self, lo, hi, bits: int):
        self.bits = int(bits)
        if self.bits < 1:
            raise ValueError(f"bits 必须 ≥1，收到 {self.bits}")
        self.lo = np.asarray(lo, dtype=np.float64)
        self.hi = np.asarray(hi, dtype=np.float64)
        if np.any(self.hi <= self.lo):
            raise ValueError("量化量程必须满足 hi > lo")
        self.levels = 2 ** self.bits
        self.delta = (self.hi - self.lo) / float(self.levels)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """量化（mid-rise）：映射回电平中心。超出量程会**裁剪**。"""
        x = np.asarray(x, dtype=np.float64)
        q = np.floor((x - self.lo) / self.delta)
        q = np.clip(q, 0.0, float(self.levels) - 1.0)
        return (self.lo + (q + 0.5) * self.delta).astype(np.float32)

    def theoretical_mse(self) -> float:
        """Δ²/12 的 per-dim 平均（**理论**值，未计过载）。"""
        return float(np.mean(self.delta ** 2) / 12.0)

    def measure_mse(self, data: Sequence[np.ndarray] | np.ndarray) -> float:
        """在给定数据上**实测**的量化 MSE（per element，含过载效应）。"""
        if isinstance(data, np.ndarray):
            arr = np.asarray(data, dtype=np.float64)
        else:
            arr = np.concatenate([np.asarray(d, dtype=np.float64).ravel() for d in data])
        q = self(arr.reshape(-1, self.lo.size if self.lo.ndim else 1))
        return float(np.mean((q - arr.reshape(q.shape)) ** 2))


def fit_quantizer_range(episodes: Sequence[dict], margin: float = 0.05) -> tuple:
    """从 episode 的观测里统计量程（per-dim min/max，各外扩 `margin` 比例）。

    ★ 为什么外扩：min/max 是样本极值，直接拿它当量程会让边界样本正好落在
    裁剪线上；外扩一点把"过载"压到可忽略。**但要报出实际过载比例**
    （见 `overload_fraction`），因为它决定了 Δ²/12 这个式子能不能用。
    """
    obs = np.concatenate([np.asarray(e["obs"], dtype=np.float64) for e in episodes], axis=0)
    lo = obs.min(axis=0)
    hi = obs.max(axis=0)
    span = np.maximum(hi - lo, 1e-12)
    return lo - margin * span, hi + margin * span


def overload_fraction(quant: UniformQuantizer, data: np.ndarray) -> float:
    """落在量程之外的元素占比（>0 ⇒ 存在裁剪，Δ²/12 低估了真实误差）。"""
    arr = np.asarray(data, dtype=np.float64)
    lo = quant.lo.reshape(1, -1) if quant.lo.ndim else quant.lo
    hi = quant.hi.reshape(1, -1) if quant.hi.ndim else quant.hi
    return float(np.mean((arr < lo) | (arr > hi)))


# ---------------------------------------------------------------- 物理层信道
@dataclass
class PhysicalChannel:
    """固定时隙的物理链路：**量化 + 由 SNR 决定的调制阶数与丢包率**。

    组成（每一块都可单独关掉做消融）：
      ① 发送端量化  `payload_fn`  —— 误差 ∝ 2^(−2b)，**确定性**损失
      ② 调制阶数    M = 2^(d·b/N_sym)   —— b 越大星座越密
      ③ 包错误率    PER = 1−(1−BER)^(d·b)   —— **随机**损失
      ④ 丢包过程    复用 X31 的 Gilbert–Elliott（PER 当平均丢包率 p̄）

    ★★ ①与③是两种**性质不同**的信息损失：
       量化误差**每次到达都有**，且不可通过"多发几次"消除；
       丢包误差**只在丢时出现**，可以通过预测（世界模型）填补。
       ⇒ 世界模型对两者的补偿能力理应不同 —— 这是本实验要问的核心问题。
    """

    snr_db: float
    bits: int
    obs_dim: int
    n_sym: int | None = None          # None ⇒ 取 obs_dim（于是 m = b，M = 2^b）
    burst_len: float | None = None    # None ⇒ L = 1/(1−PER)，即无记忆 i.i.d.
    seed: int = 0
    quantizer: UniformQuantizer | None = None
    iid: bool = False                 # True ⇒ 用无记忆伯努利而非 Gilbert（消融用）

    #: 由 (SNR, b) 决定的派生量，构造时算好并**冻结**
    bits_per_symbol: float = field(init=False)
    M: float = field(init=False)
    ber: float = field(init=False)
    per: float = field(init=False)
    n_bits: int = field(init=False)
    formula_valid: bool = field(init=False)

    def __post_init__(self):
        d = int(self.obs_dim)
        b = int(self.bits)
        self.n_sym = int(self.n_sym) if self.n_sym is not None else d
        self.n_bits = d * b
        self.bits_per_symbol = self.n_bits / float(self.n_sym)
        if self.bits_per_symbol <= 0:
            raise ValueError("每符号比特数必须 > 0")
        self.M = 2.0 ** self.bits_per_symbol
        gamma = snr_db_to_linear(self.snr_db)
        self.formula_valid = qam_approximation_valid(gamma, self.M)
        self.ber = qam_bit_error_rate(gamma, self.M)
        # ★ 公式失效区（SER 饱和）⇒ 链路**不可用**，按"几乎全丢"处理。
        #   不能用失效区算出的 PER（它会让 PER 对 b 非单调 —— 见
        #   `qam_approximation_valid` 的说明）。
        self.per = (per_from_ber(self.ber, self.n_bits) if self.formula_valid
                    else 1.0 - 1e-6)
        # ★ PER 必须落在 (0,1) 内的合法概率区间，否则 Gilbert 构造会炸
        self.per = float(min(max(self.per, 1e-9), 1.0 - 1e-9))
        L = self.burst_len if self.burst_len is not None else 1.0 / (1.0 - self.per)
        if self.iid:
            self._chan = None
            self._p = self.per
        else:
            self._chan = GilbertElliottChannel(self.per, L, seed=int(self.seed))
        self._rng = np.random.default_rng(int(self.seed))

    # ---- 发送端变换 ----
    def payload_fn(self) -> Callable[[np.ndarray], np.ndarray] | None:
        if self.quantizer is None:
            return None
        return self.quantizer

    # ---- 链路层丢包 ----
    def schedule(self, t: int, rng: np.random.Generator) -> bool:
        if self._chan is not None:
            return self._chan(t, rng)
        return bool(rng.random() >= self._p)

    def __call__(self, t: int, rng: np.random.Generator) -> bool:
        return self.schedule(t, rng)

    def reset(self, rng: np.random.Generator) -> None:
        """有状态信道（Gilbert）每条 episode 开头按稳态重采样。"""
        if self._chan is not None:
            self._chan.reset(rng)

    @property
    def quant_mse_theoretical(self) -> float:
        return 0.0 if self.quantizer is None else self.quantizer.theoretical_mse()

    def __repr__(self) -> str:
        return (f"<PhysicalChannel SNR={self.snr_db:g}dB b={self.bits} "
                f"M={self.M:g} BER={self.ber:.3e} PER={self.per:.4f} "
                f"n_bits={self.n_bits} "
                f"{'公式有效' if self.formula_valid else '★公式失效/链路不可用'}"
                f" L={getattr(self._chan, 'burst_len', float('nan')):g}>")
