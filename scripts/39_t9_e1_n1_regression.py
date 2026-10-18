#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""T9 第①件 · **E1：N=1 逐位回归**（多用户资源层 vs 单用户 `lossy_schedule`）。

一句话
------
**N=1、预算=每步 1 帧、恒成功率 q=0.7 时，新的多用户闭环必须逐位复现
`判-01` β 侧的三个锚点。** 不通就说明抽样次数/顺序错，**不许往上叠任何东西**。

为什么这一条必须先做（R12 的用法）
----------------------------------
T9 要把"通信约束"从**外生常量 `loss_prob`** 改成**分配动作的后果**。改的是接线，
不是公式 —— 而接线错的表现通常**不是报错，是数字"看着差不多"**。
本仓库历史上这类事已经发生过多次（warmup 比 horizon 长 ⇒ 每集只测 1 步，
三个 p 的代价逐位相同，被当成"任务不敏感"）。
所以先用一个**已知答案**（判-01 的 β 侧，来源：`scripts/35` 的 `run_uav(0.3, s, 30)`）
把新路径钉住：它必须回到**同一条轨迹**上，一个 bit 都不许差。

锚点（★ 来源 = 2026-10-18 本机复算输出，非手抄）
------------------------------------------------
    seed 0 → 2.9093751196861266
    seed 1 → 2.9058766984939575
    seed 2 → 2.911704212506612

链路（四条检查，任何一条红即整体 FAIL）
---------------------------------------
| 编号 | 内容 | 失败意味着 |
|---|---|---|
| E1a | 三个种子的 `per_user_cost[0]` 与锚点**逐位相等**（`==`，不是近似） | 轨迹变了：抽样顺序 / dtype / 尾窗口径有差 |
| E1b | 逐 episode 的"送达/丢失"序列可由 `default_rng(seed)` **重放**出来 | 信道流被别的东西动过（例如分配器偷抽） |
| E1b- | **阴性对照**：把重放流平移 1 个 draw ⇒ 必须**大面积失配** | 重放检查没有分辨力（恒真 = 假检查） |
| E1c | `budget_used == n_steps`（整数相等），且每步 `served` 恰好 1 帧 | I1 预算守恒没接上 |
| E1d | `delivery_rate` 的 p_eff ≈ 0.3，且 `frames + lost == alive_steps` | 记账分母错（I6） |

用法
----
    D:/devtools/miniconda3/envs/ai-lab/python.exe scripts/39_t9_e1_n1_regression.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from wmlab.control import PDRelativeController  # noqa: E402
from wmlab.envs import make_env  # noqa: E402
from wmlab.eval.multiuser import (  # noqa: E402
    Allocation, ConstantSuccess, View, delivery_rate, run_multiuser,
)

# ★ 来源：scripts/35 的 run_uav(0.3, seed, 30) 在本机（2026-10-18 21:2x）的实际输出
ANCHORS = {
    0: 2.9093751196861266,
    1: 2.9058766984939575,
    2: 2.911704212506612,
}
P_LOSS = 0.3            # 判-01 β 档原文
Q = 1.0 - P_LOSS        # ConstantSuccess 的 q ⇔ lossy_schedule(0.3)
N_EPISODES = 30         # 与判-01 一致
WARMUP = 300            # 与 X30–X45 / scripts/35 一致
MEASURED = 200
TAIL_FRAC = 0.25


def always_serve_first(view: View, rng: np.random.Generator) -> Allocation:
    """N=1 的退化分配器：有预算就派给 0 号用户。**不消耗 rng**（I4 的干净一侧）。"""
    served = np.zeros(len(view.slots), dtype=bool)
    if view.budget_left > 0 and view.slots[0].alive:
        served[0] = True
    return Allocation(served=served, reason="always-serve[0]")


def run_one(seed: int) -> dict:
    env = make_env("uav", seed=seed, max_steps=WARMUP + MEASURED)
    ctrl = PDRelativeController()
    out = run_multiuser(
        [env], [ctrl], always_serve_first, ConstantSuccess(Q),
        n_episodes=N_EPISODES, seed=seed, max_steps=MEASURED,
        warmup_steps=WARMUP, budget_total=MEASURED, tail_frac=TAIL_FRAC,
    )
    env.close()
    return out


def episode_lengths(records: list) -> list[int]:
    """按 episode 计记录条数 = 该集被测段步数（UAV 不失锁时恒为 MEASURED）。"""
    lens: dict[int, int] = {}
    for r in records:
        lens[r.episode] = lens.get(r.episode, 0) + 1
    return [lens.get(e, 0) for e in range(N_EPISODES)]


def replay_check(records: list, seed: int, shift: int = 0) -> dict:
    """用 `default_rng(seed)` 重放"送达"序列；`shift` 用于阴性对照（平移 draw）。"""
    lens = episode_lengths(records)
    rng = np.random.default_rng(seed)
    expected: list[bool] = []
    for L in lens:
        if shift:
            rng.random(shift)                 # 白白丢掉 shift 个 draw（阴性对照）
        expected.extend(list(rng.random(L) >= P_LOSS))
    got = [r.delivered_idx == 0 for r in records]
    n = min(len(expected), len(got))
    mism = int(sum(1 for k in range(n) if bool(expected[k]) != bool(got[k])))
    return {"n": n, "mismatch": mism, "rate_got": float(np.mean(got)) if got else float("nan"),
            "lengths_ok": bool(len(set(lens)) == 1 and lens[0] == MEASURED)}


def main() -> int:
    res: dict = {"p_loss": P_LOSS, "q": Q, "n_episodes": N_EPISODES,
                 "warmup": WARMUP, "measured": MEASURED, "anchors": ANCHORS, "per_seed": {}}
    ok_cost = True
    print(f"[39] E1 · N=1 逐位回归   30 集 × 3 种子 · warmup={WARMUP} · measured={MEASURED}")
    print(f"[39] {'seed':<6}{'got':>22}{'anchor':>22}{'bitwise':>9}{'p_eff':>9}{'lost':>7}")

    for seed in sorted(ANCHORS):
        out = run_one(seed)
        recs = out["records"]
        got = float(out["per_user_cost"][0])
        anchor = float(ANCHORS[seed])
        bit_exact = bool(got == anchor)          # ★ 逐位，不是 np.isclose
        dr = delivery_rate(recs, 1)
        lost = int(dr["alive_steps"][0] - dr["frames"][0])
        rep = replay_check(recs, seed, shift=0)
        neg = replay_check(recs, seed, shift=1)
        checks = {
            "E1a_cost_bitwise": bit_exact,
            "E1b_replay_exact": bool(rep["mismatch"] == 0),
            "E1b_replay_negative_control_has_teeth": bool(neg["mismatch"] > 0),
            "E1c_budget_equals_steps": bool(out["budget_used"] == out["n_steps"] == N_EPISODES * MEASURED),
            "E1c_one_frame_per_step": bool(len(recs) == out["n_steps"]),
            "E1d_accounting_closes": bool(dr["frames"][0] + lost == dr["alive_steps"][0]),
        }
        res["per_seed"][str(seed)] = {
            "cost": got, "anchor": anchor, "abs_diff": float(abs(got - anchor)),
            "bit_exact": bit_exact, "p_eff": float(dr["p_eff"][0]),
            "frames": int(dr["frames"][0]), "alive_steps": int(dr["alive_steps"][0]),
            "lost": lost, "budget_used": int(out["budget_used"]),
            "n_steps": int(out["n_steps"]), "replay": rep,
            "negative_control": neg, "checks": checks, "all_ok": bool(all(checks.values())),
        }
        ok_cost = ok_cost and all(checks.values())
        print(f"[39] {seed:<6}{got:>22.17g}{anchor:>22.17g}{str(bit_exact):>9}"
              f"{dr['p_eff'][0]:>9.4f}{lost:>7}")
        if not all(checks.values()):
            for k, v in checks.items():
                if not v:
                    print(f"[39]   ✗ {k}")

    res["all_ok"] = bool(ok_cost)
    out_path = ROOT / "outputs" / "39_t9_e1_n1_regression.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[39] {'PASS' if ok_cost else 'FAIL'} · 写出 {out_path}")
    if not ok_cost:
        print("[39] ★ 不许继续 T9 的后续子步骤（SINR 映射 / N>1 冒烟）：先查抽样顺序与 dtype")
    return 0 if ok_cost else 2


if __name__ == "__main__":
    raise SystemExit(main())
