"""world-model-lab: 世界模型 rollout 与可靠视界评测的最小可复现框架。

分层约定（换场景时只动第一层）：
    envs     -> 环境适配：统一 (obs, action) -> (next_obs, reward, done)
    models   -> 世界模型：编码器 / 潜空间转移 / 解码器
    rollout  -> 潜空间多步 rollout 与误差随视界增长的测量
    eval     -> 指标：NMSE、可靠视界、覆盖率
    utils    -> 设备选择、随机种子等横切能力
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
