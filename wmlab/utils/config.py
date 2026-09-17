"""配置加载。超参一律外置到 configs/*.yaml，便于扫参与云端批量复现。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

#: 仓库根目录（wmlab/utils/config.py -> wmlab/utils -> wmlab -> 仓库根）
REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | os.PathLike | None = None, overrides: dict | None = None) -> dict:
    """载入 YAML 配置，并用 overrides 覆盖（形如 {"train.epochs": 10}）。"""
    if path is None:
        path = REPO_ROOT / "configs" / "default.yaml"
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    with open(path, "r", encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f) or {}

    for k, v in (overrides or {}).items():
        node = cfg
        parts = k.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = v
    cfg["_config_path"] = str(path)
    return cfg


def output_dir(cfg: dict) -> Path:
    """结果输出目录（绝对路径，自动创建）。"""
    d = Path(cfg.get("output", {}).get("dir", "outputs"))
    if not d.is_absolute():
        d = REPO_ROOT / d
    d.mkdir(parents=True, exist_ok=True)
    return d
