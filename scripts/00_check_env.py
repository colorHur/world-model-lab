#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""脚本 00 —— 环境自检。

换机器、上云、或者怀疑环境坏了的时候，先跑这个。
它不训练任何东西（5 秒内出结果），只回答一个问题：

    这套环境到底能不能跑本仓库？

检查项：
  ① Python 与平台信息
  ② 关键依赖版本
  ③ 计算设备（本机 CPU / 云端 CUDA）
  ④ gymnasium 真跑一步（接口层）
  ⑤ torch 真做一次反向传播（训练层）
  ⑥ 配置能加载、wmlab 各层能 import（本仓库结构完整）

运行：
    python scripts/00_check_env.py
"""

from __future__ import annotations

import importlib
import platform
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OK, BAD, WARN = "  [OK]  ", "  [!!]  ", "  [--]  "
results: list[tuple[str, bool]] = []


def line(msg: str) -> None:
    print(msg, flush=True)


def check(name: str, fn) -> bool:
    try:
        detail = fn()
        line(f"{OK}{name:32s} {detail if detail else ''}")
        results.append((name, True))
        return True
    except Exception as e:
        line(f"{BAD}{name:32s} {type(e).__name__}: {e}")
        results.append((name, False))
        return False


def main() -> int:
    line("=" * 78)
    line("world-model-lab · 环境自检")
    line("=" * 78)

    # ---------- ① Python / 平台 ----------
    line("\n【1】运行环境")
    line(f"      解释器 : {sys.executable}")
    line(f"      Python : {sys.version.split()[0]}")
    line(f"      平台   : {platform.platform()} / {platform.machine()}")
    py_ok = sys.version_info[:2] >= (3, 9)
    line(f"{OK if py_ok else BAD}Python >= 3.9 要求    当前 {sys.version_info.major}.{sys.version_info.minor}")
    results.append(("python>=3.9", py_ok))
    if sys.version_info[:2] >= (3, 13):
        line(f"{WARN}注意：Python {sys.version_info.major}.{sys.version_info.minor} 偏新，"
             f"部分深度学习依赖可能没有预编译轮子。建议 3.11。")

    # ---------- ② 依赖 ----------
    line("\n【2】关键依赖")
    deps = ["torch", "numpy", "matplotlib", "yaml", "gymnasium", "tqdm"]
    optional = ["scipy", "pandas", "sklearn", "stable_baselines3", "diffusers", "tensorboard"]

    def dep_ver(mod: str):
        def _f():
            m = importlib.import_module(mod)
            return getattr(m, "__version__", "(no __version__)")
        return _f

    for d in deps:
        check(d, dep_ver(d))
    line("      ---- 可选（缺了不影响本仓库核心流程）----")
    for d in optional:
        try:
            m = importlib.import_module(d)
            line(f"{OK}{d:32s} {getattr(m, '__version__', '?')}")
        except Exception:
            line(f"{WARN}{d:32s} 未安装")

    # ---------- ③ 设备 ----------
    line("\n【3】计算设备")
    try:
        import torch
        from wmlab.utils import describe_device, get_device

        dev = get_device()
        line(f"{OK}get_device() -> {dev.type}      {describe_device(dev)}")
        line(f"      cuda.is_available() = {torch.cuda.is_available()}")
        line(f"      torch threads = {torch.get_num_threads()}")
        if dev.type == "cpu":
            line(f"{WARN}当前是 CPU 模式。若你已在云 GPU 上，说明装的是 CPU 版 torch，请换：")
            line("            pip install torch --index-url https://download.pytorch.org/whl/cu121")
        results.append(("get_device()", True))
    except Exception as e:
        line(f"{BAD}设备检查失败: {e}")
        results.append(("get_device()", False))

    # ---------- ④ gymnasium 真跑一步 ----------
    line("\n【4】环境接口（真跑一步）")

    def _env():
        from wmlab.envs import make_env
        env = make_env("CartPole-v1", seed=0)
        obs = env.reset(seed=0)
        res = env.step(env.sample_action())
        env.close()
        return f"obs_dim={env.obs_dim} act_dim={env.act_dim} reward={res.reward:.1f} done={res.done}"

    check("make_env + reset + step", _env)

    # ---------- ⑤ torch 真做一次反传 ----------
    line("\n【5】训练通路（真做一次反向传播）")

    def _bp():
        import torch
        from wmlab.utils import get_device
        dev = get_device()
        net = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.SiLU(), torch.nn.Linear(16, 1)).to(dev)
        x = torch.randn(64, 8, device=dev)
        y = torch.randn(64, 1, device=dev)
        opt = torch.optim.Adam(net.parameters(), lr=1e-2)
        losses = []
        for _ in range(50):
            loss = torch.nn.functional.mse_loss(net(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        return f"loss {losses[0]:.4f} -> {losses[-1]:.4f}（下降 {losses[0]/max(losses[-1],1e-9):.1f}x）"

    check("forward/backward/Adam", _bp)

    # ---------- ⑥ 本仓库结构 ----------
    line("\n【6】本仓库结构")

    def _cfg():
        from wmlab.utils import load_config, output_dir
        cfg = load_config()
        d = output_dir(cfg)
        return f"configs 载入 OK；输出目录 {d}"

    check("配置加载", _cfg)

    def _model():
        import torch
        from wmlab.models import MLPWorldModel
        from wmlab.utils import count_params, get_device
        dev = get_device()
        m = MLPWorldModel(obs_dim=4, act_dim=2, latent_dim=8, hidden=32).to(dev)
        o = torch.randn(16, 4, device=dev)
        a = torch.randint(0, 2, (16,), device=dev)
        n = torch.randn(16, 4, device=dev)
        loss, parts = m.loss(o, a, n)
        return f"参数量={count_params(m)} loss={float(loss.detach()):.4f}"

    check("世界模型前向+损失", _model)

    def _rollout():
        import torch
        from wmlab.models import MLPWorldModel
        from wmlab.rollout import imagine
        from wmlab.utils import get_device
        dev = get_device()
        m = MLPWorldModel(obs_dim=4, act_dim=2, latent_dim=8, hidden=32).to(dev)
        out = imagine(m, torch.randn(4, 4, device=dev), torch.randint(0, 2, (4, 6), device=dev))
        return f"rollout 输出形状 {tuple(out.shape)}（期望 (4, 6, 4)）"

    check("潜空间 rollout", _rollout)

    def _metrics():
        from wmlab.eval import reliable_horizon
        h = reliable_horizon([1, 2, 3, 4, 5], [0.01, 0.02, 0.04, 0.09, 0.2], 0.05)
        return f"reliable_horizon 插值 = {h}（期望约 3.2）"

    check("可靠视界指标", _metrics)

    # ---------- 汇总 ----------
    line("\n" + "=" * 78)
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    line(f"结果：{passed}/{total} 项通过")
    failed = [n for n, ok in results if not ok]
    if failed:
        line("未通过：" + ", ".join(failed))
        line("→ 未通过的项直接决定本仓库能不能跑，先修它。")
        return 1
    line("环境可用。下一步：")
    line("    python scripts/01_random_baseline.py     # 产出第一张图")
    line("    python scripts/02_train_world_model.py   # 训练 + 可靠视界")
    line("=" * 78)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        line("\n中断")
        sys.exit(130)
    except Exception:
        traceback.print_exc()
        sys.exit(2)
