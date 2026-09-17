"""横切工具：设备选择、随机种子、配置加载、参数量统计。"""

from .config import REPO_ROOT, load_config, output_dir
from .device import count_params, describe_device, get_device
from .seed import make_rng, set_seed

__all__ = [
    "get_device",
    "describe_device",
    "count_params",
    "set_seed",
    "make_rng",
    "load_config",
    "output_dir",
    "REPO_ROOT",
]
