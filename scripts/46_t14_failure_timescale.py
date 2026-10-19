#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""T14 · 副产物：**任务失效的时间尺度**（预注册 §6 的触发项）。

为什么要有这个脚本
------------------
T14 主实验的结论是 `cart_gain(20) = 0.030·s_cart < 0.1·s_cart` ⇒ 按预注册判定"视界不是瓶颈"。
但这个判定有**两种互斥解释**，只靠 `cart_gain` 分不开：

  (a) 视界仍然不够（失效发生在 20 步之后）⇒ 继续加大 H 有意义；
  (b) 视界够了（失效发生在 20 步之内），但 **max-age 基准策略本来就会在那之前救它**
      ⇒ 偏离一步没有价值。此时"加大 H"是死路，而"age 已接近最优"才是真相。

本脚本量 (a) 的那一半：**完全不服务**时，任务还能活多少步。
若中位数 ≪ 20，则 H=20 已覆盖整个失效窗口 ⇒ 支持 (b)，排除 (a)。

口径（与主实验同一 env / 同一控制器 / 同一代价定义）
----------------------------------------------------
    CartPole：reset 后 est 冻结在 reset 观测（= 一帧都不送），记 `terminated` 的步数。
    UAV：先按 warmup=300 用真值喂到稳态，再冻结 est，记 `default_uav_cost > escape_radius` 的步数。

★ 这是**诊断量**，不是判据：它不进入任何 Gate，也不改任何臂。逐集原始值全部落盘。

用法
----
    D:/devtools/miniconda3/envs/ai-lab/python.exe scripts/46_t14_failure_timescale.py
产物
----
    results/46_t14_failure_timescale.json
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from wmlab.control import PDRelativeController  # noqa: E402
from wmlab.envs import make_env  # noqa: E402
from wmlab.eval.multiuser import default_uav_cost  # noqa: E402
from wmlab.eval.results_io import write_result  # noqa: E402  ← 入库唯一入口（P0-3）

N_EP = 200
MAX_STEPS = 200
WARMUP_UAV = 300
ESCAPE = 25.0


def _lqr_gain():
    spec = importlib.util.spec_from_file_location(
        "j35", ROOT / "scripts" / "35_judge_task_sensitivity.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["j35"] = m
    spec.loader.exec_module(m)
    ch, _obs, K, _dt = m.make_cartpole(seed=0, ep_seed=0)
    ch.close()
    return K


class CartLQR:
    def __init__(self, K):
        self.K = np.asarray(K, dtype=float)

    def reset(self, est=None):
        pass

    def act(self, est):
        u = float(-(self.K @ np.asarray(est, dtype=float))[0])
        return np.array([1 if u > 0 else 0], dtype=np.int64)


def stats(x: list[int]) -> dict:
    a = np.asarray(x, dtype=float)
    return {"n": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)),
            "p10": float(np.percentile(a, 10)), "p90": float(np.percentile(a, 90)),
            "min": float(a.min()), "max": float(a.max()),
            "frac_le_5": float((a <= 5).mean()), "frac_le_10": float((a <= 10).mean()),
            "frac_le_20": float((a <= 20).mean())}


def main() -> int:
    K = _lqr_gain()
    print(f"[46] 任务失效时间尺度｜N={N_EP}｜完全不服务（est 冻结）")

    # ---------- CartPole ----------
    cp_steps = []
    for e in range(N_EP):
        env = make_env("CartPole-v1", seed=e)
        ctrl = CartLQR(K)
        obs = env.reset(seed=e)
        est = np.asarray(obs, dtype=np.float32).copy()   # ★ 冻结，永不刷新
        ctrl.reset(est)
        k = 0
        while k < MAX_STEPS:
            res = env.step(ctrl.act(est))
            k += 1
            if res.terminated or res.truncated:
                break
        cp_steps.append(k)
        env.close()
    cp = stats(cp_steps)
    print(f"  CartPole 断服存活步数：median={cp['median']:.0f} "
          f"p10={cp['p10']:.0f} p90={cp['p90']:.0f} "
          f"⇒ ≤10 步占 {cp['frac_le_10']:.1%}，≤20 步占 {cp['frac_le_20']:.1%}")

    # ---------- UAV ----------
    ua_steps = []
    for e in range(N_EP):
        env = make_env("uav", seed=e, max_steps=WARMUP_UAV + MAX_STEPS)
        ctrl = PDRelativeController()
        obs = np.asarray(env.reset(seed=e), dtype=np.float32).copy()
        ctrl.reset(obs)
        # warmup：每步都送真值（与主实验 warmup 口径一致）
        for _ in range(WARMUP_UAV):
            res = env.step(ctrl.act(obs))
            obs = np.asarray(res.obs, dtype=np.float32).copy()
            if res.terminated or res.truncated:
                break
        est = obs.copy()                                # ★ 之后冻结
        k = 0
        while k < MAX_STEPS:
            res = env.step(ctrl.act(est))
            k += 1
            if default_uav_cost(np.asarray(res.obs)) > ESCAPE or res.terminated or res.truncated:
                break
        ua_steps.append(k)
        env.close()
    ua = stats(ua_steps)
    print(f"  UAV 断服到逃逸（>25 m）：median={ua['median']:.0f} "
          f"p10={ua['p10']:.0f} p90={ua['p90']:.0f} "
          f"⇒ ≤20 步占 {ua['frac_le_20']:.1%}")

    out = {"config": {"N_episodes": N_EP, "max_steps": MAX_STEPS,
                      "warmup_uav": WARMUP_UAV, "escape_radius": ESCAPE,
                      "note": "est frozen (no delivery); diagnostic only, not a gate"},
           "cartpole_steps_to_fall": {"summary": cp, "per_episode": cp_steps},
           "uav_steps_to_escape": {"summary": ua, "per_episode": ua_steps},
           "interpretation": {
               "cartpole": ("H=20 覆盖了 %.0f%% 的失效窗口（median %.0f 步）"
                            % (100 * cp["frac_le_20"], cp["median"])),
               "uav": ("H=20 只覆盖 %.0f%%（median %.0f 步）"
                       % (100 * ua["frac_le_20"], ua["median"]))}}
    # ★ P0-3（2026-10-19）：改走 run-id 分目录，拒绝覆盖固定路径。
    dest = write_result("46_t14_failure_timescale", out)
    print(f"  ⇒ {out['interpretation']['cartpole']}；{out['interpretation']['uav']}")
    print(f"JSON → {dest.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
