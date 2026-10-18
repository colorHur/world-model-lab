#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""T12 前置自检 · 验收报告 §3 的三项修复（A 信息边界 / B 死亡记账 / C 终止代价）。

为什么这脚本必须先跑
--------------------
三项修复都会**改变 Gate 2 的比值**：
  A 让所有臂丢掉"真值紧迫性遥测"，oracle 相对优势会下降；
  B 让死者不再被记送达，`p_eff` 与预算对得上，跨臂预算才可比；
  C 让"早死"变贵，避免把失败奖励成低成本。
如果它们在代码里没接线（R12），后面跑出来的四臂数字就全是替代品。

每条检查都配**阴性对照**：一个恒真的检查不是检查。

用法
----
    D:/devtools/miniconda3/envs/ai-lab/python.exe scripts/43_t9_fix_abc_selfcheck.py

退出码：0 = 全过；2 = 有真失败。JSON 落 `results/43_fix_abc_selfcheck.json`。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from wmlab.envs.base import StepResult  # noqa: E402
from wmlab.eval.allocators import RoundRobin  # noqa: E402
from wmlab.eval.multiuser import (  # noqa: E402
    Allocation, ConstantSuccess, StepRecord, delivery_rate, run_multiuser, window_cost,
)

WINDOW = 200
FAILS: list[str] = []


def check(tag: str, ok: bool, detail: str) -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {tag}: {detail}")
    if not ok:
        FAILS.append(f"{tag}: {detail}")


# ================================================================ 合成环境（确定性，不碰 gym）
class CountingEnv:
    """每步 obs[0] += 1，可选在第 `term_step` 步终止。cost_fn = obs[0]。

    ★ 用合成环境而不是 gym：本脚本要验的是**记账/信息边界**，不是动力学；
      合成环境的真值序列是解析已知的，任何偏差都能定位到 runner。
    """

    name = "counting"
    obs_dim = 1
    act_dim = 1
    is_discrete = True
    act_low = None
    act_high = None

    def __init__(self, term_step: int | None = None, start: float = 0.0):
        self.term_step = term_step
        self.start = float(start)
        self._t = 0
        self._v = self.start

    def reset(self, seed: int | None = None) -> np.ndarray:
        self._t = 0
        self._v = self.start
        return np.array([self._v], dtype=np.float32)

    def step(self, action) -> StepResult:
        self._t += 1
        self._v += 1.0
        term = self.term_step is not None and self._t >= self.term_step
        return StepResult(obs=np.array([self._v], dtype=np.float32), reward=0.0,
                          terminated=bool(term), truncated=False)

    def close(self) -> None:
        pass


class ConstCtrl:
    def reset(self, est=None) -> None:
        pass

    def act(self, est) -> np.ndarray:
        return np.array([0], dtype=np.int64)


class NeverServe:
    """一帧都不发（预算 0 的等价物）。"""

    def __call__(self, view, rng):
        return Allocation(served=np.zeros(len(view.slots), dtype=bool), reason="never")


class AlwaysServe:
    def __call__(self, view, rng):
        s = np.zeros(len(view.slots), dtype=bool)
        s[0] = True
        return Allocation(served=s, reason="always-0")


class BlindRR:
    """**故意不过滤 alive** 的盲轮转（阴性对照用）。"""

    def __init__(self) -> None:
        self.k = 0

    def __call__(self, view, rng):
        n = len(view.slots)
        s = np.zeros(n, dtype=bool)
        s[self.k % n] = True
        self.k += 1
        return Allocation(served=s, reason="blind-rr")


class Probe:
    """记录每一步分配器**实际看到**的 cost_est / privileged，用来验信息边界。"""

    def __init__(self, inner=None) -> None:
        self.inner = inner or AlwaysServe()
        self.seen_cost: list[list[float]] = []
        self.seen_priv: list[bool] = []

    def __call__(self, view, rng):
        self.seen_cost.append([float(s.cost_est) for s in view.slots])
        self.seen_priv.append(view.privileged is not None)
        return self.inner(view, rng)


cost_obs0 = lambda o: float(np.asarray(o).reshape(-1)[0])  # noqa: E731


# ================================================================ A · 信息边界
def section_a() -> dict:
    print("\n== A · 信息边界（cost_est 只随送达更新）==")
    # A1：持续丢包（一帧不发）⇒ cost_est 恒定，而真值代价在涨
    probe = Probe(NeverServe())
    r = run_multiuser([CountingEnv()], [ConstCtrl()], probe, ConstantSuccess(0.7),
                      n_episodes=1, seed=0, max_steps=30, budget_total=0, cost_fn=cost_obs0)
    est = np.array(probe.seen_cost)[:, 0]
    true_costs = [rec.costs[0] for rec in r["records"]]
    check("A1 cost_est 恒定", float(est.max() - est.min()) == 0.0,
          f"30 步 cost_est 范围 [{est.min():.1f},{est.max():.1f}]（期望 0.0）")
    check("A1- 真值在变（检查有分辨力）", float(max(true_costs) - min(true_costs)) > 0.0,
          f"真值代价 {min(true_costs):.1f}→{max(true_costs):.1f}")
    check("A1b 非诊断臂看不到 privileged", not any(probe.seen_priv),
          f"privileged 非 None 的步数 = {sum(probe.seen_priv)}")

    # A2（阴性对照）：送达 ⇒ cost_est 必须跟着真值走，否则 A1 是"恒不变"的假检查
    probe2 = Probe(AlwaysServe())
    run_multiuser([CountingEnv()], [ConstCtrl()], probe2, ConstantSuccess(1.0),
                  n_episodes=1, seed=0, max_steps=30, budget_total=30, cost_fn=cost_obs0)
    est2 = np.array(probe2.seen_cost)[:, 0]
    check("A2 送达时 cost_est 跟着真值变", float(est2[-1] - est2[0]) >= 29.0,
          f"cost_est {est2[0]:.1f}→{est2[-1]:.1f}（期望 +29 以上）")

    # A3 结构：诊断臂拿得到 true_obs，且它来自**独立通道**（不与 cost_est 同源）
    probe3 = Probe(AlwaysServe())
    run_multiuser([CountingEnv()], [ConstCtrl()], probe3, ConstantSuccess(0.0),
                  n_episodes=1, seed=0, max_steps=10, budget_total=10, cost_fn=cost_obs0,
                  diagnostic=True)
    check("A3 诊断臂 privileged 已接线", all(probe3.seen_priv),
          f"privileged 非 None 的步数 = {sum(probe3.seen_priv)}/10")
    return {"A1_cost_est_range": float(est.max() - est.min()),
            "A1_true_range": float(max(true_costs) - min(true_costs)),
            "A2_delta": float(est2[-1] - est2[0])}


# ================================================================ B · 死亡记账
def section_b() -> dict:
    print("\n== B · 死亡记账（reserved / attempted / delivered / wasted）==")
    envs = [CountingEnv(), CountingEnv(term_step=1)]     # 1 号在第一个被测步后终止
    r = run_multiuser(envs, [ConstCtrl(), ConstCtrl()], BlindRR(), ConstantSuccess(1.0),
                      n_episodes=1, seed=0, max_steps=100, budget_total=100, cost_fn=cost_obs0)
    dr = delivery_rate(r["records"], 2)
    recs = r["records"]
    reserved = sum(int(x.served_idx >= 0) for x in recs)
    attempted = sum(int(x.attempted_idx >= 0) for x in recs)
    dead_delivered = sum(1 for x in recs
                         if x.delivered_idx == 1 and not x.alive[1])
    check("B1 reserved == 预算", reserved == 100, f"reserved={reserved}（期望 100）")
    check("B2 死者的 delivered 条数 == 0", dead_delivered == 0,
          f"原始表里 delivered_idx==1 且 alive[1]=False 的条数 = {dead_delivered}")
    check("B3 attempted < reserved（空转被记出来）", attempted < reserved,
          f"attempted={attempted}, wasted={reserved - attempted}")
    check("B4 frames[1] == 0", dr["frames"][1] == 0, f"frames={dr['frames']}")
    check("B5 汇总 == 原始表（同词义）",
          sum(dr["frames"]) == sum(1 for x in recs if x.delivered_idx >= 0),
          f"汇总 Σframes={sum(dr['frames'])}, 原始表 count={sum(1 for x in recs if x.delivered_idx >= 0)}")
    check("B6 预算三分账闭合",
          dr["budget_reserved"] == dr["budget_attempted"] + dr["budget_wasted"],
          f"{dr['budget_reserved']} = {dr['budget_attempted']} + {dr['budget_wasted']}")
    returns_now = sum(1 for x in recs
                      if x.served_idx == 1 and not x.alive[1] and x.attempted_idx == -1)
    check("B7 派给死人的步真的没发帧", returns_now > 0, f"空转步数={returns_now}")

    # B8 阴性对照：过滤 alive 的臂 ⇒ wasted 必须为 0（说明 wasted 不是恒真）
    r2 = run_multiuser([CountingEnv(), CountingEnv(term_step=1)],
                       [ConstCtrl(), ConstCtrl()], RoundRobin(), ConstantSuccess(1.0),
                       n_episodes=1, seed=0, max_steps=100, budget_total=100, cost_fn=cost_obs0)
    dr2 = delivery_rate(r2["records"], 2)
    check("B8 过滤 alive 的臂 wasted == 0", dr2["budget_wasted"] == 0,
          f"wasted={dr2['budget_wasted']}, attempted={dr2['budget_attempted']}, reserved={dr2['budget_reserved']}")
    return {"blind": {k: dr[k] for k in ("budget_reserved", "budget_attempted", "budget_wasted",
                                         "budget_lost", "frames")},
            "alive_filtered": {k: dr2[k] for k in ("budget_reserved", "budget_attempted",
                                                   "budget_wasted", "frames")}}


# ================================================================ C · 定长窗口 + 吸收代价
def _mk_records(ep: int, n_steps: int, cost: float) -> list[StepRecord]:
    return [StepRecord(episode=ep, t=t, served_idx=0, attempted_idx=0, delivered_idx=0,
                       ages=(0,), costs=(cost,), alive=(True,))
            for t in range(1, n_steps + 1)]


def section_c() -> dict:
    print("\n== C · 定长窗口 + 终止吸收代价 ==")
    L = WINDOW
    recs = (_mk_records(0, 50, 0.0)       # 早死（第 50 步后无记录）
            + _mk_records(1, 150, 0.0)    # 晚死
            + _mk_records(2, L, 0.0)      # 全存活·零代价
            + _mk_records(3, L, 1.0)      # 全存活·每秒代价 1
            + _mk_records(4, 10, 0.0))    # 极早死
    J = window_cost(recs, 1, window=L, absorbed=1.0, n_episodes=5)
    J = [x[0] for x in J]
    check("C1 全存活·c=0 ⇒ J=0", abs(J[2]) < 1e-12, f"J={J[2]:.6f}")
    check("C2 全存活·c=1 ⇒ J=1", abs(J[3] - 1.0) < 1e-12, f"J={J[3]:.6f}")
    check("C3 早死(50) ⇒ J=0.75", abs(J[0] - 0.75) < 1e-12, f"J={J[0]:.6f}")
    check("C4 晚死(150) ⇒ J=0.25", abs(J[1] - 0.25) < 1e-12, f"J={J[1]:.6f}")
    check("C5 极早死(10) ⇒ J=0.95", abs(J[4] - 0.95) < 1e-12, f"J={J[4]:.6f}")
    check("C6 ★ 早死严格更贵（旧口径会反过来）", J[0] > J[1],
          f"J(50)={J[0]:.3f} > J(150)={J[1]:.3f}")
    old = [float(np.mean([r.costs[0] for r in _mk_records(e, n, 0.0)])) for e, n in
           ((0, 50), (1, 150))]
    check("C7 旧口径区分不了早死/晚死（缺陷复现）", old[0] == old[1],
          f"旧尾均值 {old[0]:.3f} vs {old[1]:.3f} ⇒ 差异收益全靠新口径")

    # C8 越界必须抛（不许静默截断）
    try:
        window_cost(_mk_records(0, L + 1, 0.0), 1, window=L, absorbed=1.0)
        ok = False
    except ValueError:
        ok = True
    check("C8 记录超过窗口必须抛", ok, "L+1 步未抛错")

    # C9 缺失 episode 必须补齐（不许静默少算）
    J2 = window_cost(_mk_records(0, 20, 0.0), 1, window=L, absorbed=1.0, n_episodes=2)
    check("C9 缺失 episode 补齐", len(J2) == 2 and abs(J2[1][0] - 1.0) < 1e-12,
          f"len={len(J2)}, J[1]={J2[1][0]:.3f}（全窗口 absorbed）")

    # ★★ C10 两用户、一死一活：补齐边界必须**逐用户**（第一版用逐集 max(t)，此检查会打死它）
    recs2 = []
    for t in range(1, L + 1):
        recs2.append(StepRecord(episode=0, t=t, served_idx=0, attempted_idx=0,
                                delivered_idx=0,
                                ages=(0, 0),
                                costs=(1.0, (0.0 if t <= 12 else float("nan"))),
                                alive=(True, t <= 12)))
    J3 = window_cost(recs2, 2, window=L, absorbed=[0.0, 1.0], n_episodes=1)[0]
    check("C10 两用户：一死一活时逐用户补齐",
          abs(J3[1] - (L - 12) / L) < 1e-12,
          f"J=({J3[0]:.4f},{J3[1]:.4f})，期望 (1.0,{(L - 12) / L:.4f})")
    check("C10- 存活用户不被死者的补齐污染", abs(J3[0] - 1.0) < 1e-12, f"J0={J3[0]:.6f}")
    return {"J": J, "old_tail_mean": old, "J_two_user": J3}


def main() -> int:
    print(f"numpy {np.__version__}｜窗口 L={WINDOW}")
    out = {"A": section_a(), "B": section_b(), "C": section_c()}
    out["fails"] = FAILS
    out["pass"] = len(FAILS) == 0
    dest = ROOT / "results"
    dest.mkdir(exist_ok=True)
    (dest / "43_fix_abc_selfcheck.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{'='*60}\n{'ALL PASS' if out['pass'] else 'FAILED: ' + '; '.join(FAILS)}")
    print(f"JSON → {dest / '43_fix_abc_selfcheck.json'}")
    return 0 if out["pass"] else 2


if __name__ == "__main__":
    sys.exit(main())
