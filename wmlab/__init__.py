"""world-model-lab: 世界模型 rollout 与可靠视界评测的最小可复现框架。

分层约定（换场景时只动第一层）：
    envs     -> 环境适配：统一 (obs, action) -> (next_obs, reward, done)
                + 有损信道（channel.py，X24）：只判定"这一步是否送达"，不动动力学
    models   -> 世界模型：编码器 / 潜空间转移 / 解码器
    rollout  -> 潜空间多步 rollout 与误差随视界增长的测量
    eval     -> 指标：NMSE、可靠视界、覆盖率
                + 在线状态跟踪（tracking.py，X24）：丢包时用世界模型顶替，记 AoI
                + AoI 解析（aoi.py，X26）：由离线 NMSE(h) 曲线解析预测在线误差
    train    -> 监督训练循环（脚本 02 与 04 共用，避免两份实现漂移）
    utils    -> 设备选择、随机种子等横切能力

核心问题始终是一个：**世界模型 rollout 到第几步开始不可靠（H\\*）**，
以及**这个 H\\* 在通信里意味着什么**（多久必须传一次）。
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
