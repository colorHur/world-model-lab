#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""T9 第②件 · **N=2 冒烟 + I6 空转反例**（多用户分支的第一次运行）。

一句话
------
**N=1 已经被 E1 钉死了（`scripts/39`），但 N>1 的记账一行都没跑过；而 N>1 记账错了
没有任何锚点会抓。** 本脚本就是给多用户分支造第一个锚点。

为什么先做这个而不是先写 SINR
-----------------------------
SINR 公式写错，E1 会抓（N=1 时有三个已知答案）；但"两用户份额怎么分、死人的份额去哪、
预算是不是每步恰好 1 帧"这类记账错，**在 N=1 上根本不存在**，所以只能靠 N=2 自建判据。

判据（★ 先写判据再跑，跑完不许改）
----------------------------------
S1 基线（两用户都活着，交替服务，`ConstantSuccess(q=0.7)`，预算 200 帧/集）：

| 编号 | 判据 | 失败意味着 |
|---|---|---|
| C1 | `budget_used == n_steps == 6000`，且每条记录恰好派 1 帧 | I1 预算守恒没接上多用户 |
| C2 | 每用户 `alive_steps == 6000`（UAV 不失锁，对齐 E1） | 环境层不是同一条轨迹 |
| C3a | `frames_0 + frames_1 + lost == 6000`（记账闭环） | 送达/丢失/派帧三个计数不自洽 |
| C3b | `|attempts_0 − attempts_1| ≤ 1`（份额平衡） | 交替分配没按 t 走，或预算被单边吃掉 |
| C3c | `|frames_0 − frames_1| ≤ 1`（**原文口径**） | ★ 预期**不可满足**，见下方"判据缺陷" |
| C4 | `p_eff_i` 落在 `1 − q·share_i` 的 4σ 带内（`share_i` 用实测 attempts） | 送达率记账或抽样次数错 |

★★ **判据缺陷（写在这里，不许事后悄悄改）**：T9 文档 §4 原文写的是
`|两用户 n_frames 之差| ≤ 1`。但 `UserSlot.n_frames` 的定义是**累计成功送达帧数**
（`multiuser.py:84`），它带二项噪声：每用户 3000 次派帧、q=0.7 ⇒ 差值标准差
`sqrt(2·3000·0.21) ≈ 35.5`。所以"差 ≤ 1"对**送达数**恒不可满足。
⇒ 本脚本把 C3b 按 **attempts**（确定性，应为 0）判，C3c 按**原文**判并如实报 FAIL，
两个数字都打出来。**这是判据写错，不是代码跑错** —— 结论写进 T9 文档，不改 §2 任何阈值。

S2 I6 空转反例（1 号用户在第 1 个被测步被强制 terminated，仍是交替分配）：

| 编号 | 判据 | 失败意味着 |
|---|---|---|
| S2a | 0 号的 `delivered` 布尔序列与 S1 **逐位相同** | 死者的死亡扰动了生者的信道（I4/D2 破了） |
| S2b | 0 号 `frames_0` 与 S1 相等 | **份额被转嫁**（I6 违约） |
| S2c | `budget_used == n_steps`（死人的帧照样算已用） | 空转的帧没被记账 ⇒ 预算虚高 |

S3 阴性对照（故意违规的"转嫁"分配器：1 号死后把份额给 0 号）：

| 编号 | 判据 | 失败意味着 |
|---|---|---|
| S3 | 0 号 `frames_0` **严格大于** S1 的 `frames_0`，且 `attempts_0 ≈ 6000` | S2 的检查没有分辨力（恒真 = 假检查） |

用法
----
    D:/devtools/miniconda3/envs/ai-lab/python.exe scripts/40_t9_n2_smoke.py

退出码：0 = 有效判据全过（C3c 的 FAIL 单独记为"判据缺陷"，不影响退出码）；2 = 有真失败。
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
from wmlab.envs.base import StepResult  # noqa: E402
from wmlab.eval.multiuser import (  # noqa: E402
    Allocation, ConstantSuccess, View, delivery_rate, run_multiuser,
)

Q = 0.7                 # ⇒ 等效 p_loss = 0.3（沿用 E1 / 判-01 β 档）
N_USERS = 2
N_EPISODES = 30
WARMUP = 300
MEASURED = 200
TAIL_FRAC = 0.25
SEED = 0                # 冒烟只跑 1 个种子：判据是结构性的（记账/份额），不是统计性的
KILL_AT = WARMUP + 1    # 全局第 301 次 step = 第 1 个被测步


# ---------------------------------------------------------------- 分配器
def alternate(view: View, rng: np.random.Generator) -> Allocation:
    """交替服务：t 奇 → 0 号，t 偶 → 1 号。**不看 alive**（故意：I6 要的是"份额空转"）。

    不消耗 rng（I4 的干净一侧）。
    """
    served = np.zeros(len(view.slots), dtype=bool)
    if view.budget_left > 0:
        served[(view.t - 1) % len(view.slots)] = True
    return Allocation(served=served, reason="alternate-by-t")


def transfer_first_alive(view: View, rng: np.random.Generator) -> Allocation:
    """★ **故意违反 I6** 的分配器：把帧给第一个活着的用户（= 死人份额转嫁给生者）。

    存在理由只有一个：做 S3 阴性对照 —— 若 S2 的"没有多拿帧"是恒真的，那它就不是检查。
    """
    served = np.zeros(len(view.slots), dtype=bool)
    if view.budget_left > 0:
        for s in view.slots:
            if s.alive:
                served[s.idx] = True
                break
    return Allocation(served=served, reason="transfer-to-first-alive")


# ---------------------------------------------------------------- 强制死亡包装
class KillAtStep:
    """在第 `kill_at` 次 `step`（1-based，**每次 reset 后重新计数**）强制 `terminated=True`。

    用于 I6 反例：让 1 号用户在第 1 个被测步就死，且不改动环境本身的动力学。
    ★ 只包装 1 号，0 号的 env 原样传入 ⇒ 两条轨迹在死亡前后完全解耦（S2a 的前提）。
    """

    def __init__(self, env, kill_at: int):
        self.env = env
        self.kill_at = int(kill_at)
        self.n = 0

    def reset(self, seed: int | None = None):
        self.n = 0
        return self.env.reset(seed=seed)

    def step(self, action):
        r = self.env.step(action)
        self.n += 1
        if self.n >= self.kill_at and not r.terminated:
            return StepResult(obs=r.obs, reward=r.reward, terminated=True,
                              truncated=r.truncated, info=r.info)
        return r

    def close(self) -> None:
        self.env.close()


# ---------------------------------------------------------------- 运行
def run_case(allocator, *, kill_user1: bool) -> dict:
    envs = [make_env("uav", seed=SEED, max_steps=WARMUP + MEASURED) for _ in range(N_USERS)]
    if kill_user1:
        envs[1] = KillAtStep(envs[1], KILL_AT)
    ctrls = [PDRelativeController() for _ in range(N_USERS)]
    out = run_multiuser(
        envs, ctrls, allocator, ConstantSuccess(Q),
        n_episodes=N_EPISODES, seed=SEED, max_steps=MEASURED,
        warmup_steps=WARMUP, budget_total=MEASURED, tail_frac=TAIL_FRAC,
    )
    for e in envs:
        e.close()
    return out


def summarize(out: dict) -> dict:
    recs = out["records"]
    dr = delivery_rate(recs, N_USERS)
    attempts = [0] * N_USERS
    lost = 0
    for r in recs:
        if r.served_idx >= 0:
            attempts[r.served_idx] += 1
            if r.delivered_idx < 0:
                lost += 1
    deliv_seq = [[bool(r.delivered_idx == i and r.alive[i]) for r in recs]
                 for i in range(N_USERS)]
    return {
        "dr": dr, "attempts": attempts, "lost": lost,
        "frames": dr["frames"], "alive_steps": dr["alive_steps"],
        "deliv_count": [int(sum(s)) for s in deliv_seq],
        "deliv_seq": deliv_seq,
        "budget_used": int(out["budget_used"]), "n_steps": int(out["n_steps"]),
        "n_records": len(recs),
        "n_served_total": int(sum(attempts)),
    }


def sigma_band(q: float, n: int) -> float:
    """二项 4σ 带：`4·sqrt(q(1−q)/n)`。n = 该用户的派帧数（不是总步数）。"""
    return float(4.0 * np.sqrt(max(q * (1.0 - q), 0.0) / max(n, 1)))


def main() -> int:
    print(f"[40] T9 · N=2 冒烟   seed={SEED} · {N_EPISODES} 集 × {MEASURED} 步/集 · "
          f"q={Q} · 预算 {MEASURED} 帧/集")

    s1 = summarize(run_case(alternate, kill_user1=False))
    s2 = summarize(run_case(alternate, kill_user1=True))
    s3 = summarize(run_case(transfer_first_alive, kill_user1=True))

    # ---------------- S1 判据
    exp_p_eff = [1.0 - Q * (s1["attempts"][i] / s1["alive_steps"][i]) if s1["alive_steps"][i] else float("nan")
                 for i in range(N_USERS)]
    band = [sigma_band(Q, s1["attempts"][i]) for i in range(N_USERS)]
    c1 = bool(s1["budget_used"] == s1["n_steps"] == N_EPISODES * MEASURED
              and s1["n_records"] == s1["n_steps"] == s1["n_served_total"])
    c2 = bool(all(x == N_EPISODES * MEASURED for x in s1["alive_steps"]))
    c3a = bool(sum(s1["frames"]) + s1["lost"] == s1["n_served_total"])
    c3b = bool(abs(s1["attempts"][0] - s1["attempts"][1]) <= 1)
    c3c_as_written = bool(abs(s1["frames"][0] - s1["frames"][1]) <= 1)
    c4 = bool(all(abs(s1["dr"]["p_eff"][i] - exp_p_eff[i]) <= band[i] for i in range(N_USERS)))

    # ---------------- S2（I6 空转）
    s2a = bool(s2["deliv_seq"][0] == s1["deliv_seq"][0])
    s2b = bool(s2["frames"][0] == s1["frames"][0])
    s2c = bool(s2["budget_used"] == s2["n_steps"] == N_EPISODES * MEASURED)

    # ---------------- S3（阴性对照）
    s3_ok = bool(s3["frames"][0] > s1["frames"][0] and s3["attempts"][0] > 1.5 * s1["attempts"][0])

    checks = {
        "C1_budget_conserved": c1,
        "C2_alive_steps_full": c2,
        "C3a_accounting_closes": c3a,
        "C3b_attempts_balanced": c3b,
        "C4_p_eff_matches_lower_bound_plus_q": c4,
        "S2a_dead_does_not_perturb_alive": s2a,
        "S2b_no_share_transfer": s2b,
        "S2c_wasted_frames_still_counted": s2c,
        "S3_negative_control_has_teeth": s3_ok,
    }

    print("\n[40] ---- S1 基线（两用户都活）----")
    print(f"[40] {'user':<6}{'attempts':>10}{'frames':>9}{'alive_steps':>13}"
          f"{'p_eff':>9}{'exp':>9}{'band':>8}")
    for i in range(N_USERS):
        print(f"[40] {i:<6}{s1['attempts'][i]:>10}{s1['frames'][i]:>9}"
              f"{s1['alive_steps'][i]:>13}{s1['dr']['p_eff'][i]:>9.4f}"
              f"{exp_p_eff[i]:>9.4f}{band[i]:>8.4f}")
    print(f"[40] lost={s1['lost']}  frames_sum+lost={sum(s1['frames']) + s1['lost']}  "
          f"n_served={s1['n_served_total']}  budget_used={s1['budget_used']}  n_steps={s1['n_steps']}")
    print(f"[40] ★ |Δframes| 实测 = {abs(s1['frames'][0] - s1['frames'][1])}"
          f"（原文 C3c 判据要求 ≤ 1 ⇒ {'PASS' if c3c_as_written else 'FAIL'}）；"
          f"理论 sd = {float(np.sqrt(2 * s1['attempts'][0] * Q * (1 - Q))):.1f}")
    print(f"[40] ★ |Δattempts| = {abs(s1['attempts'][0] - s1['attempts'][1])}（C3b 口径）")
    print(f"[40] ★ 与乐观下界 1−1/N = {1.0 - 1.0 / N_USERS:.2f} 的差："
          f"{s1['dr']['p_eff'][0] - (1.0 - 1.0 / N_USERS):+.4f}（H2：必须实测，不许反推）")

    print("\n[40] ---- S2 I6 空转（1 号第 1 个被测步强制 terminated）----")
    print(f"[40] attempts={s2['attempts']}  frames={s2['frames']}  alive_steps={s2['alive_steps']}")
    print(f"[40] p_eff={['%.4f' % x for x in s2['dr']['p_eff']]}  budget_used={s2['budget_used']}")
    print(f"[40] 0 号 delivered 序列与 S1 逐位相同 = {s2a}；0 号 frames 相等 = {s2b}")

    print("\n[40] ---- S3 阴性对照（故意转嫁：1 号死后给 0 号）----")
    print(f"[40] attempts={s3['attempts']}  frames={s3['frames']}  "
          f"（S1 的 0 号 frames={s1['frames'][0]}）")

    print("\n[40] ---- 判据 ----")
    for k, v in checks.items():
        print(f"[40] {'✓' if v else '✗'} {k}")
    if not c3c_as_written:
        print("[40] ⚠ 原文 C3c（|Δframes| ≤ 1）不可满足 —— 这是**判据**写错，不是代码跑错；"
              "见脚本头 docstring 的『判据缺陷』一节")

    ok = all(checks.values())
    res = {
        "seed": SEED, "n_users": N_USERS, "q": Q, "n_episodes": N_EPISODES,
        "warmup": WARMUP, "measured": MEASURED, "kill_at": KILL_AT,
        "S1": {k: v for k, v in s1.items() if k != "deliv_seq"},
        "S2": {k: v for k, v in s2.items() if k != "deliv_seq"},
        "S3": {k: v for k, v in s3.items() if k != "deliv_seq"},
        "expected_p_eff": exp_p_eff, "band": band,
        "lower_bound_1_minus_1_over_N": 1.0 - 1.0 / N_USERS,
        "checks": checks,
        "C3c_as_written": {"pass": c3c_as_written,
                           "delta_frames": abs(s1["frames"][0] - s1["frames"][1]),
                           "theoretical_sd": float(np.sqrt(2 * s1["attempts"][0] * Q * (1 - Q))),
                           "verdict": "criterion_ill_posed"},
        "all_ok": bool(ok),
    }
    out_path = ROOT / "outputs" / "40_t9_n2_smoke.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[40] {'PASS' if ok else 'FAIL'} · 写出 {out_path}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
