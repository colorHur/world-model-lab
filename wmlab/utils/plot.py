# -*- coding: utf-8 -*-
"""统一绘图样式。

为什么要有这个模块
------------------
中文字体依赖 ``MPLCONFIGDIR`` 环境变量 + matplotlibrc 才能生效，但那依赖进程环境：
换一台机器、换一个 IDE 终端、上云跑实验，中文标签立刻变成方块（tofu）。

所以样式由**代码显式设置**，不依赖任何外部环境。任何脚本画图前调一次
``apply_style()``，就能保证：中文正常、负号正常、DPI 与留白一致。

用法::

    from wmlab.utils.plot import apply_style, save_fig, PALETTE
    apply_style()
    ...
    save_fig(fig, path)
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
from matplotlib import font_manager  # 必须显式导入：import matplotlib 不会带上 font_manager

# 中文字体候选表：按优先级从左到右。每台机器命中的那个不同，逐个探测。
_CJK_CANDIDATES = (
    "Microsoft YaHei",   # Windows 简体中文最常见
    "PingFang SC",       # macOS
    "Noto Sans CJK SC",  # Linux / 容器
    "Source Han Sans SC",
    "DengXian",
    "SimHei",
    "SimSun",
    "Microsoft JhengHei",
    "KaiTi",
)

# 统一配色（色盲友好，浅色主题下可读）
PALETTE = {
    "blue": "#4C78A8",
    "red": "#E45756",
    "green": "#54A24B",
    "orange": "#F58518",
    "purple": "#B279A2",
    "grey": "#8C8C8C",
    "yellow": "#ECA03B",
}
_COLOR_CYCLE = [PALETTE["blue"], PALETTE["red"], PALETTE["green"],
                PALETTE["orange"], PALETTE["purple"], PALETTE["yellow"], PALETTE["grey"]]

_STYLE_APPLIED = False


def pick_cjk_font() -> str | None:
    """返回本机第一个可用的中文字体名；一个都没有则返回 None。"""
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in _CJK_CANDIDATES:
        if name in available:
            return name
    return None


def apply_style(dpi: int = 120) -> str:
    """套用统一样式。返回实际选中的中文字体名（None 表示没找到中文字体）。"""
    global _STYLE_APPLIED
    font = pick_cjk_font()

    sans = list(_CJK_CANDIDATES)
    # 中文优先，但保留 DejaVu 兜底（西文/数字/符号仍用 DejaVu 更好看）
    sans += ["DejaVu Sans", "Bitstream Vera Sans", "sans-serif"]

    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": sans,
        "axes.unicode_minus": False,   # 否则负号渲染成方块
        "figure.dpi": dpi,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
        "axes.prop_cycle": matplotlib.cycler(color=_COLOR_CYCLE),
        "axes.grid": True,
        "grid.alpha": 0.25,
        "figure.autolayout": False,
        "agg.path.chunksize": 20000,   # 长序列折线加速
    })
    _STYLE_APPLIED = True
    return font


def save_fig(fig, path, dpi: int = 160) -> Path:
    """保存图片，自动创建父目录，并返回路径。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def style_summary() -> str:
    """一行状态串，自检脚本用来确认字体链路正常。"""
    return (f"font={pick_cjk_font()} minus={matplotlib.rcParams['axes.unicode_minus']} "
            f"applied={_STYLE_APPLIED}")
