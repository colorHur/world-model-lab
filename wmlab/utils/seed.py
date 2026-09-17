"""随机种子：曲线可复现的前提。

世界模型的评测里，"可靠视界"是一个单点数值（第几步误差超阈），
对随机性极敏感。不固定种子得出来的结论没有意义，也无法被面试官复现。
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 0, deterministic: bool = False) -> int:
    """一次性设置 python / numpy / torch（含 CUDA）的随机种子。"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        # 关掉 cudnn 自动调优，换取逐位可复现（会慢一些）
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed


def make_rng(seed: int = 0) -> np.random.Generator:
    """独立的 numpy 随机数发生器，避免全局 np.random 被别处污染。"""
    return np.random.default_rng(seed)
