#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""T13 · `lookahead`（期望 1 步前瞻）臂的自检 —— 8 条，全部可打死它。

预注册：`04-实验与代码/2026-10-19-T13-lookahead臂实现口径-预注册-v0.1.md` §4。

    L1  N=1 时 lookahead 与 rr 逐位相同（只有一个候选 ⇒ 前瞻不起作用）
    L2  合成矩阵：argmin 必须跟着行和走；把矩阵转置后必须换人（验 `[cand][user]` 索引）
    L3  `privileged=None` / 缺 `true_cost_next` ⇒ 必须抛 MissingStateError（不许静默选 0）
    L4  接线纪律：`counterfactual=True, diagnostic=False` 必须 ValueError；
        `diagnostic=True, counterfactual=False` 挂 lookahead 必须 MissingStateError
    L5  反事实不扰动主轨迹：rr 带/不带反事实，`StepRecord` 逐位相同
    L6  `budget_total=0` 时 lookahead 不派帧（I1）
    L7  真实 N=2（两个 UAV）：**服务自己必须降低自己的下一步代价**（`Ĉ[0][0] < Ĉ[1][0]`）
    L8  矩阵必须**对决策有分辨力**：同一时刻不同候选的行和必须不同
        （若接了"一步"版本，矩阵对候选恒等 ⇒ 这条会 FAIL）

T14 追加（2026-10-19 04:3x，多步视界 H）：
    L9  `counterfactual_horizon_costs(horizon=1)` 与 `counterfactual_next_costs` **逐位相同**
    L10 H=3 时与**独立实现的暴力参考**（从原始 env 重新 clone、手写循环）逐位相同
    L11 CartPole 被饿 4 步后：`cart_gain(1) ≡ 0` 但 `cart_gain(20) > 0`，且两矩阵不同
    L12 `Lookahead(horizon=20)` 挂 `cf_horizon=1`（及反向）⇒ 必须 ValueError

用法：
    D:/devtools/miniconda3/envs/ai-lab/python.exe scripts/44_t13_lookahead_selfcheck.py
退出码：0 = 全过；1 = 有 FAIL。
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from wmlab.control import PDRelativeController  # noqa: E402
from wmlab.envs import make_env  # noqa: E402
from wmlab.eval.allocators import Lookahead, RoundRobin  # noqa: E402
from wmlab.eval.results_io import write_selfcheck  # noqa: E402  ← 入库唯一入口（P0-3）
from wmlab.eval.multiuser import (  # noqa: E402
    Allocation, MissingStateError, SINRSuccess, UserSlot, View,
    counterfactual_horizon_costs, counterfactual_next_costs,
    default_uav_cost, delivery_rate, require_privileged, run_multiuser,
)

RESULTS: list[dict] = []
FAILS: list[str] = []


def check(tag: str, ok: bool, detail: str) -> None:
    RESULTS.append({"tag": tag, "ok": bool(ok), "detail": str(detail)})
    print(f"  [{'PASS' if ok else 'FAIL'}] {tag}: {detail}")
    if not ok:
        FAILS.append(f"{tag}: {detail}")


class _R:
    """最小 step 返回值（L11 的解析环境用；只要有 obs/terminated/truncated 三个字段）。"""

    __slots__ = ("obs", "terminated", "truncated")

    def __init__(self, obs, terminated=False, truncated=False):
        self.obs = obs
        self.terminated = terminated
        self.truncated = truncated


def _slot(i: int, *, alive=True, age=0, obs=None, csi=0.8, cost_est=1.0) -> UserSlot:
    return UserSlot(idx=i, alive=alive, age=age,
                    last_obs=np.zeros(6, dtype=np.float32) if obs is None else obs,
                    n_frames=1, n_attempts=1, csi=csi, cost_est=cost_est)


def _view(slots, mat=None, budget_left=10, priv_missing=False) -> View:
    priv = None
    if not priv_missing:
        priv = {"true_cost_next": np.asarray(mat, dtype=float)}
    return View(t=1, slots=tuple(slots), budget_left=budget_left,
                budget_total=200, privileged=priv)


def _rec_eq(a, b) -> bool:
    """nan-safe 逐字段比较（`costs` 里死者是 nan，nan != nan 会让 == 恒 False）。"""
    if (a.episode, a.t, a.served_idx, a.attempted_idx, a.delivered_idx,
            a.ages, a.alive) != (b.episode, b.t, b.served_idx, b.attempted_idx,
                                 b.delivered_idx, b.ages, b.alive):
        return False
    return bool(np.array_equal(np.asarray(a.costs, dtype=float),
                               np.asarray(b.costs, dtype=float), equal_nan=True))


def main() -> int:
    print("[44] lookahead 臂自检（8 条）")

    # ---------------- L1：N=1 时与 rr 逐位相同 ----------------
    envs = [make_env("uav", seed=0, max_steps=400)]
    ctrls = [PDRelativeController()]
    r_rr = run_multiuser(envs, ctrls, RoundRobin(), SINRSuccess(0.7), n_episodes=2,
                         seed=0, max_steps=60, warmup_steps=0, budget_total=40,
                         cost_fn=default_uav_cost, csi=0.9)
    envs = [make_env("uav", seed=0, max_steps=400)]
    ctrls = [PDRelativeController()]
    r_la = run_multiuser(envs, ctrls, Lookahead(), SINRSuccess(0.7), n_episodes=2,
                         seed=0, max_steps=60, warmup_steps=0, budget_total=40,
                         cost_fn=default_uav_cost, csi=0.9,
                         diagnostic=True, counterfactual=True)
    same = (len(r_rr["records"]) == len(r_la["records"])
            and all(_rec_eq(a, b) for a, b in zip(r_rr["records"], r_la["records"])))
    check("L1 N=1 lookahead ≡ rr（逐位）", same,
          f"{len(r_la['records'])} 条记录，预算 {r_la['budget_reserved']}")

    # ---------------- L2：argmin 跟着行和走，且索引必须是 [cand][user] ----------------
    # ★ 第一版测试矩阵写错（转置后行和仍以 0 更小 ⇒ 分辨不出索引方向，检查形同虚设）。
    #   换成非对称矩阵：按**行**求和 ⇒ 选 1；按**列**求和（错误实现）⇒ 选 0。
    arm = Lookahead()
    m0 = np.array([[0.10, 0.10], [5.0, 5.0]])         # 行和 0.2 vs 10.0 ⇒ 选 0
    v = _view([_slot(0), _slot(1)], m0)
    pick_a = int(np.flatnonzero(arm(v, np.random.default_rng(0)).served)[0])
    m1 = np.array([[0.10, 10.0], [0.20, 0.30]])       # 行和 10.1 vs 0.5 ⇒ 选 1
    v = _view([_slot(0), _slot(1)], m1)
    pick_b = int(np.flatnonzero(arm(v, np.random.default_rng(0)).served)[0])
    v = _view([_slot(0), _slot(1)], m1.T)             # 列和转成行和 ⇒ 选 0
    pick_c = int(np.flatnonzero(arm(v, np.random.default_rng(0)).served)[0])
    check("L2 argmin 与 [cand][user] 索引（含列优先反例）", (pick_a, pick_b, pick_c) == (0, 1, 0),
          f"m0→{pick_a}（期望0）；非对称 m1→{pick_b}（期望1，按列会得0）；m1.T→{pick_c}（期望0）")

    # ---------------- L3：特权缺失必须抛，不许静默 ----------------
    ok3 = []
    try:
        arm(_view([_slot(0), _slot(1)], None, priv_missing=True), np.random.default_rng(0))
        ok3.append("privileged=None 未抛")
    except MissingStateError:
        pass
    try:
        vv = View(t=1, slots=(_slot(0), _slot(1)), budget_left=10, budget_total=200,
                  privileged={"true_obs": (np.zeros(6), np.zeros(6))})
        arm(vv, np.random.default_rng(0))
        ok3.append("缺 true_cost_next 未抛")
    except MissingStateError:
        pass
    check("L3 特权缺失 ⇒ MissingStateError", not ok3, "；".join(ok3) if ok3 else "两路都抛")

    # ---------------- L4：接线纪律 ----------------
    l4 = []
    try:   # counterfactual 但非诊断
        run_multiuser([make_env("uav", seed=1, max_steps=200)], [PDRelativeController()],
                      Lookahead(), SINRSuccess(0.7), n_episodes=1, seed=1, max_steps=10,
                      budget_total=5, cost_fn=default_uav_cost,
                      diagnostic=False, counterfactual=True)
        l4.append("counterfactual+非诊断 未抛")
    except ValueError:
        pass
    try:   # 诊断但没配反事实 ⇒ lookahead 拿不到矩阵
        run_multiuser([make_env("uav", seed=1, max_steps=200)], [PDRelativeController()],
                      Lookahead(), SINRSuccess(0.7), n_episodes=1, seed=1, max_steps=10,
                      budget_total=5, cost_fn=default_uav_cost,
                      diagnostic=True, counterfactual=False)
        l4.append("漏配 counterfactual 未抛")
    except MissingStateError:
        pass
    check("L4 接线纪律（两路都必须抛）", not l4, "；".join(l4) if l4 else "两路都抛")

    # ---------------- L5：反事实不扰动主轨迹 ----------------
    def _rr(ep, diag, cf):
        e = [make_env("uav", seed=3, max_steps=400), make_env("CartPole-v1", seed=3)]
        c = [PDRelativeController(), _LQR()]
        return run_multiuser(e, c, RoundRobin(), SINRSuccess(0.7), n_episodes=2, seed=3,
                             max_steps=ep, warmup_steps=[50, 0], budget_total=100,
                             cost_fn=[default_uav_cost, _cart_cost], csi=[0.9, 0.55],
                             diagnostic=diag, counterfactual=cf)
    a = _rr(80, False, False)
    b = _rr(80, True, True)
    same5 = (len(a["records"]) == len(b["records"])
             and all(_rec_eq(x, y) for x, y in zip(a["records"], b["records"])))
    check("L5 反事实不扰动 rr 轨迹（逐位）", same5,
          f"{len(a['records'])} vs {len(b['records'])} 条；"
          f"J 尾均值 {np.round(a['per_user_cost'], 6).tolist()} vs "
          f"{np.round(b['per_user_cost'], 6).tolist()}")

    # ---------------- L6：预算 0 时不派帧 ----------------
    e = [make_env("uav", seed=5, max_steps=300)]
    r6 = run_multiuser(e, [PDRelativeController()], Lookahead(), SINRSuccess(0.7),
                       n_episodes=1, seed=5, max_steps=30, budget_total=0,
                       cost_fn=default_uav_cost, csi=0.9, diagnostic=True, counterfactual=True)
    dr6 = delivery_rate(r6["records"], 1)
    check("L6 budget=0 ⇒ 不派帧（I1）",
          dr6["budget_reserved"] == 0 and dr6["budget_attempted"] == 0,
          f"reserved={dr6['budget_reserved']} attempted={dr6['budget_attempted']}")

    # ---------------- L7/L8：真实 N=2（两个 UAV），估一次矩阵 ----------------
    # 造"估计过期"：est 冻结在 reset 观测上，真值已走 20 步（正是无送达时的情形）
    envs = [make_env("uav", seed=7, max_steps=400), make_env("uav", seed=8, max_steps=400)]
    ctrls = [PDRelativeController(), PDRelativeController()]
    obs0 = [envs[i].reset(seed=7 + i) for i in range(2)]
    for i in range(2):
        ctrls[i].reset(obs0[i])
    est = [o.copy() for o in obs0]                     # ★ 冻结（模拟 20 步没送达）
    for _ in range(20):
        for i in range(2):
            envs[i].step(ctrls[i].act(est[i]))          # 用旧 est 决策，但真值在往前走
    q = SINRSuccess(0.7)
    qv = [q.q(0.9), q.q(0.9)]
    mat = counterfactual_next_costs(envs, ctrls, est, [True, True], qv,
                                    [default_uav_cost, default_uav_cost])
    # 服务自己 ⇒ 自己的下一步距离应该更小（刷新了过期估计）
    ok7 = bool(mat[0, 0] < mat[1, 0]) and bool(mat[1, 1] < mat[0, 1])
    check("L7 服务自己降低自己的下一步代价", ok7,
          f"Ĉ=\n{np.round(mat, 4).tolist()}；"
          f"uav0: {mat[0, 0]:.4f}(self) < {mat[1, 0]:.4f}(other)？"
          f" uav1: {mat[1, 1]:.4f}(self) < {mat[0, 1]:.4f}(other)？")
    rows = mat.sum(axis=1)
    ok8 = bool(abs(rows[0] - rows[1]) > 1e-9)
    check("L8 矩阵对决策有分辨力（两候选行和不同）", ok8,
          f"行和={np.round(rows, 4).tolist()}，差={abs(rows[0] - rows[1]):.6f}")

    # ================= T14：多步视界（H）的四条 =================
    ages7 = [20, 20]

    # ---------------- L9：H=1 必须与既有实现逐位相同 ----------------
    mat_h1 = counterfactual_horizon_costs(envs, ctrls, est, [True, True], qv,
                                          [default_uav_cost, default_uav_cost],
                                          horizon=1, ages=ages7)
    ok9 = bool(np.array_equal(mat_h1, mat, equal_nan=True))
    check("L9 H=1 ≡ counterfactual_next_costs（逐位）", ok9,
          f"最大绝对差={np.nanmax(np.abs(mat_h1 - mat)) if np.all(np.isfinite(mat)) else 'nan'}"
          f"；矩阵={np.round(mat_h1, 6).tolist()}")

    # ---------------- L10：H=3 与独立暴力参考逐位相同 ----------------
    ref = _naive_h(envs, ctrls, est, [True, True], qv,
                   [default_uav_cost, default_uav_cost], 3, ages7)
    mat_h3 = counterfactual_horizon_costs(envs, ctrls, est, [True, True], qv,
                                          [default_uav_cost, default_uav_cost],
                                          horizon=3, ages=ages7)
    ok10 = bool(np.array_equal(ref, mat_h3, equal_nan=True))
    check("L10 H=3 ≡ 独立暴力参考（逐位）", ok10,
          f"ref={np.round(ref, 6).tolist()} vs impl={np.round(mat_h3, 6).tolist()}")

    # ---------------- L11：视界必须真的在"累计"，且方向可解析验证 ----------------
    # ★ 判据改写（2026-10-19 T14，原因写在这里也写进预注册 §7）：
    #   第一版把"CartPole 被饿后 cart_gain(20) > 0"当自检 —— 这是**用经验假设冒充结构检查**：
    #   CartPole 的存活在"估计过期 4~6 步"处是刀锋，而两个候选的 q 不同（0.70 vs 0.53），
    #   送达的 Bernoulli 期望会让"服务低 q 的 CartPole"在 47% 分支里白干；
    #   基准策略在 age 打平时按 idx 小选，正好把 CartPole 的下一次刷新推迟一步 ⇒ 决定生死。
    #   实测该场景 cart_gain(20) = −0.1125（符号为负是机制，不是实现错）。
    #   ⇒ 换成**解析环境**：代价只由"步数计数器 ≥ 5"决定，与 est 无关 ⇒ 解析值与实现逐位可比。
    class _CounterEnv:
        """obs = [步数]；代价 = 1 当步数 ≥ 5。忽略 est ⇒ 送达/丢失两分支同值。"""

        def __init__(self):
            self.c = 0

        def reset(self, seed=None):
            self.c = 0
            return np.array([0.0])

        def step(self, action):
            self.c += 1
            return _R(obs=np.array([float(self.c)]), terminated=False, truncated=False)

    class _NullCtrl:
        def reset(self, est=None):
            pass

        def act(self, est):
            return 0

    def _cnt_cost(obs):
        return 1.0 if float(np.asarray(obs).reshape(-1)[0]) >= 5.0 else 0.0

    def _expect(h):
        # 矩阵的第 k 个累计步 = 反事实第 k+1 步 ⇒ 计数器 = c0 + (k+1)，c0 = 0
        return sum(1.0 for k in range(1, h + 1) if (k + 1) >= 5) / float(h)

    m1c = counterfactual_horizon_costs([_CounterEnv(), _CounterEnv()], [_NullCtrl(), _NullCtrl()],
                                       [np.zeros(1), np.zeros(1)], [True, True], [0.7, 0.5],
                                       [_cnt_cost, _cnt_cost], horizon=1, ages=[0, 0])
    m20c = counterfactual_horizon_costs([_CounterEnv(), _CounterEnv()], [_NullCtrl(), _NullCtrl()],
                                        [np.zeros(1), np.zeros(1)], [True, True], [0.7, 0.5],
                                        [_cnt_cost, _cnt_cost], horizon=20, ages=[0, 0])
    e1, e20 = _expect(1), _expect(20)
    ok11 = bool(np.allclose(m1c, e1, atol=1e-15) and np.allclose(m20c, e20, atol=1e-15)
                and not np.allclose(m1c, m20c))
    check("L11 视界在累计：解析环境 H=20 必须等于解析值（H=1 为 0）", ok11,
          f"期望 {e1:.4f}/{e20:.4f}（H=1/H=20），实现 {np.round(m1c, 6).tolist()}/"
          f"{np.round(m20c, 6).tolist()}")

    # 参考量（**非判据**）：真实 CartPole 饿 4 步后的增益。符号是机制，不进检查。
    e11 = [make_env("uav", seed=11, max_steps=400), make_env("CartPole-v1", seed=11)]
    c11 = [PDRelativeController(), _LQR()]
    o11 = [e11[i].reset(seed=11 + i) for i in range(2)]
    for i in range(2):
        c11[i].reset(o11[i])
    est11 = [np.asarray(x, dtype=np.float32).copy() for x in o11]
    for _ in range(4):
        for i in range(2):
            e11[i].step(c11[i].act(est11[i]))          # est 冻结 ⇒ 真值往前走
    q11 = SINRSuccess(0.7)
    qv11 = [q11.q(0.9), q11.q(0.55)]
    mcp1 = counterfactual_horizon_costs(e11, c11, est11, [True, True], qv11,
                                        [default_uav_cost, _cart_cost], horizon=1, ages=[4, 4])
    mcp20 = counterfactual_horizon_costs(e11, c11, est11, [True, True], qv11,
                                         [default_uav_cost, _cart_cost], horizon=20, ages=[4, 4])
    print("        └ 参考（非判据）：真实 CartPole 饿 4 步后 cart_gain(H=1)="
          f"{float(mcp1[0, 1] - mcp1[1, 1]):+.4f}，cart_gain(H=20)="
          f"{float(mcp20[0, 1] - mcp20[1, 1]):+.4f}"
          f"（Ĉ[serve uav][cart]={mcp20[0, 1]:.4f} vs Ĉ[serve cart][cart]={mcp20[1, 1]:.4f}）")

    # ---------------- L12：臂与 runner 的视界必须一致 ----------------
    l12 = []
    for arm_h, runner_h in ((20, 1), (1, 5)):
        try:
            run_multiuser([make_env("uav", seed=2, max_steps=200)], [PDRelativeController()],
                          Lookahead(horizon=arm_h), SINRSuccess(0.7), n_episodes=1, seed=2,
                          max_steps=6, budget_total=3, cost_fn=default_uav_cost,
                          diagnostic=True, counterfactual=True, cf_horizon=runner_h)
            l12.append(f"arm_h={arm_h}/cf_h={runner_h} 未抛")
        except ValueError:
            pass
    check("L12 视界不一致 ⇒ ValueError（两向都抛）", not l12,
          "；".join(l12) if l12 else "arm20/cf1 与 arm1/cf5 都抛")

    out = {"checks": RESULTS, "failed": FAILS}
    # ★ P0-3（2026-10-19）：自检快照走 results/selfcheck/，判定不变则不重写
    write_selfcheck("44_lookahead_selfcheck", out)
    n_ok = sum(1 for r in RESULTS if r["ok"])
    print(f"\n{'ALL CHECKS PASS' if not FAILS else 'FAILED: ' + '; '.join(FAILS)} "
          f"({n_ok}/{len(RESULTS)})")
    return 0 if not FAILS else 1


# ================================================================ 辅助（与 scripts/42 同源）
def _LQR():
    """复用 `scripts/35` 的 DARE 增益（不复制公式；见 scripts/42 的同名实现）。"""
    import importlib.util
    spec = importlib.util.spec_from_file_location("judge01", ROOT / "scripts"
                                                 / "35_judge_task_sensitivity.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["judge01"] = mod
    spec.loader.exec_module(mod)
    ch, _obs, K, _dt = mod.make_cartpole(seed=0, ep_seed=0)
    ch.close()

    class _CP:
        def reset(self, est=None):
            pass

        def act(self, est):
            u = float(-(K @ np.asarray(est, dtype=float))[0])
            return np.array([1 if u > 0 else 0], dtype=np.int64)
    return _CP()


def _cart_cost(obs: np.ndarray) -> float:
    o = np.asarray(obs, dtype=float).reshape(-1)
    return 1.0 if (abs(o[0]) > 2.4 or abs(o[2]) > 0.2095) else 0.0


def _naive_h(envs, ctrls, est, alive, q, cfs, horizon: int, ages):
    """**独立实现**的 H 步参考矩阵（L10 的对照，故意不复用被测代码的任何结构）。

    与 `counterfactual_horizon_costs` 的差别（都是为了"换个写法就能暴露 bug"）：
      · 每个 (候选, 分支) 都**从原始 env/controller 重新 deepcopy**，不经过公共的第 1 步缓存；
      · 死亡用 `dead` 字典记"从哪一步开始冻结"，不靠改副本字典；
      · age 用列表推导（不是就地循环）。
    若被测实现对"被前一个分支改过的 env 副本"、"ages 忘了推进"、"少算/多算一步"有错，
    这里会给出不同的数 ⇒ L10 FAIL。
    """
    n = len(envs)
    mat = np.full((n, n), np.nan, dtype=float)
    for cand in range(n):
        if not alive[cand]:
            continue
        row = np.zeros(n, dtype=float)
        for delivered in (True, False):
            w = float(q[cand]) if delivered else 1.0 - float(q[cand])
            if w <= 0.0:
                continue
            E = [copy.deepcopy(e) for e in envs]
            C = [copy.deepcopy(c) for c in ctrls]
            est_j = [np.asarray(x, dtype=np.float64).reshape(-1).copy() for x in est]
            # ---- 第 t 步 ----
            nxt_j, dead = {}, {}
            for j in range(n):
                if not alive[j]:
                    continue
                r = E[j].step(C[j].act(est_j[j]))
                o = np.asarray(r.obs, dtype=np.float32)
                nxt_j[j] = o
                if r.terminated or r.truncated:
                    dead[j] = float(cfs[j](o))
            if delivered:
                est_j[cand] = nxt_j[cand].astype(np.float64)
            ages_j = [0 if (delivered and j == cand) else int(ages[j]) + 1 for j in range(n)]
            acc = np.zeros(n, dtype=float)
            for j in range(n):
                if alive[j] and j in dead:
                    acc[j] += w * dead[j] * horizon       # 整段窗口按终止代价
            # ---- 第 2 … H+1 步 ----
            for s in range(2, horizon + 2):
                live = [j for j in range(n) if alive[j] and j not in dead]
                if s >= 3 and live:
                    u = live[0]
                    for j in live[1:]:
                        if ages_j[j] > ages_j[u]:
                            u = j
                    est_j[u] = nxt_j[u].astype(np.float64)
                    ages_j = [(0 if j == u else ages_j[j] + 1) for j in range(n)]
                for j in range(n):
                    if not alive[j]:
                        continue
                    if j in dead:
                        acc[j] += w * dead[j]
                        continue
                    r = E[j].step(C[j].act(est_j[j]))
                    o = np.asarray(r.obs, dtype=np.float32)
                    acc[j] += w * float(cfs[j](o))
                    nxt_j[j] = o
                    if r.terminated or r.truncated:
                        dead[j] = float(cfs[j](o))
            row += acc
        mat[cand] = row / float(horizon)
    return mat


if __name__ == "__main__":
    sys.exit(main())
