"""设备选择：本地（无独显，CPU）与云端（租用 GPU）之间无痛切换的关键。

约定：仓库内任何 `.to(...)` 都必须走 `get_device()`，
代码里不允许出现硬编码的 `.cuda()`。这样从本机搬到云 GPU 时，
只需要 `--device cuda`（或 `device: cuda` in yaml），不改一行模型代码。
"""

from __future__ import annotations

import torch


def get_device(spec: str | None = None) -> torch.device:
    """解析设备字符串。

    Args:
        spec: None / "auto" / "" -> 有 CUDA 用 CUDA，否则 CPU；
              "cpu" / "cuda" / "cuda:1" -> 显式指定。

    Raises:
        RuntimeError: 显式要求 CUDA 但当前机器没有可用的 CUDA。
    """
    if spec is None or str(spec).strip().lower() in ("auto", ""):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dev = torch.device(str(spec).strip())
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"指定了 device={spec}，但当前环境 torch.cuda.is_available() == False。\n"
            f"  本机（工位机）无 NVIDIA 独显，请用 device=cpu；\n"
            f"  若已在云 GPU 上，请确认装了 CUDA 版 torch：\n"
            f"    pip install torch --index-url https://download.pytorch.org/whl/cu121"
        )
    return dev


def describe_device(dev: torch.device) -> str:
    """返回一行人类可读的设备描述，用于日志与实验记录。"""
    if dev.type != "cuda":
        import platform

        return f"cpu ({platform.machine()}, torch {torch.__version__}, threads={torch.get_num_threads()})"
    idx = dev.index or 0
    name = torch.cuda.get_device_name(idx)
    total = torch.cuda.get_device_properties(idx).total_memory / 2**30
    return f"cuda:{idx} ({name}, {total:.1f} GB, torch {torch.__version__})"


def count_params(module: torch.nn.Module) -> int:
    """可训练参数量。写进日志，方便对比不同规模的世界模型。"""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
