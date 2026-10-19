#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""T22 · 三种预测的**时序与信息集确认**（P1-2 / P1-3）—— ★ 本轮不出收益表。

口径来源（★ 全部在看任何数字之前冻结）
--------------------------------------
`04-实验与代码/2026-10-19-T22-三种预测-预注册-v0.1.md` §2–§6。

这一轮只回答三件事
------------------
1. **接得上吗**：`predictor=None` 与 `ZOH` 是否逐位相同；`ZOH` 是否与 T21 的 36 cell 逐位相同。
2. **守得住吗**：预测器有没有偷真值（`diagnostic` 开关不改变轨迹；状态里没有 env 对象）；
   预测有没有偷偷进分配器的信息集（`last_obs` 仍是送达值）。
3. **时序对吗**：送达即重置（误差归 0）、断服期误差增长、`actions_since` 与真正施加的动作一致。

为什么这三条比"收益"更该先做
----------------------------
若 `TrueDyn`（诊断上界）都不产生收益，P2 的学习模型就不必做；但若预测器接错了一步，
"没有收益"这个结论本身就是假的。T13–T15 已经踩过三次"接了但没接对"（#46/#47），
所以本脚本的每条检查都配一个**阴性对照**（能打死它的那个错究竟是什么）。

用法
----
    D:/devtools/miniconda3/envs/ai-lab/python.exe scripts/53_t22_predictor_timing.py
    ... --skip-zoh-regression     # 跳过 S2（36 cell，约占本轮 3/4 时间）

产物
----
    **不写任何文件**（自检入口不得触碰入库证据，R17 / T20-P0-3）。
    唯一的证据是 printed 输出与本轮结果文档。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from wmlab.eval.allocators import MaxAge  # noqa: E402
from wmlab.eval.multiuser import (  # noqa: E402
    Allocation, ConstantSuccess, SINRSuccess, default_uav_cost, run_multiuser, window_cost,
)
from wmlab.eval.predictors import Linear, PredictorBase, TrueDyn, ZOH, make_predictor  # noqa: E402
from wmlab.eval.results_io import latest_result_path  # noqa: E402

FAILS: list[str] = []


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


T12 = _load(ROOT / "scripts" / "42_t12_hetero_four_arms.py", "t12_mod")
T15 = _load(ROOT / "scripts" / "47_t15_normalized_age.py", "t15_mod")
T52 = _load(ROOT / "scripts" / "52_t21_paced_matching_ablation.py", "t52_mod")

WINDOW = T12.WINDOW
WARMUP_UAV = T12.WARMUP_UAV
BUDGETS = T12.BUDGETS
COSTS = [default_uav_cost, T12.cartpole_cost]
ABSORBED = [T12.ABSORBED_UAV, T12.ABSORBED_CART]
ARMS = T52.ARMS
MODES = T52.MODES
K = T12.load_lqr_gain()


def check(tag: str, ok: bool, detail: str) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {tag}: {detail}")
    if not ok:
        FAILS.append(f"{tag}: {detail}")


# ================================================================ 装置
def _run(budget, arm, mode, seed, n_ep, *, predictor=None, probe=None,
         allocator=None, success=None):
    envs = [T12.make_env("uav", seed=seed, max_steps=WARMUP_UAV + WINDOW),
            T12.make_env("CartPole-v1", seed=seed)]
    ctrls = [T12.PDRelativeController(), T12.CartPoleLQR(K)]
    alloc = allocator if allocator is not None else T52.make_alloc(arm, mode, budget)
    succ = SINRSuccess(T12.Q_REF) if success is None else success
    try:
        r = _run_core(envs, ctrls, alloc, succ, budget, seed, n_ep,
                      predictor=predictor, probe=probe)
    finally:
        for e in envs:
            e.close()
    return r


def _run_core(envs, ctrls, alloc, succ, budget, seed, n_ep, *, predictor, probe):
    """把参数顺序钉在一处，避免 S1–S7 各自复制一份 `run_multiuser(...)` 调用。"""
    return run_multiuser(envs, ctrls, alloc, succ, n_episodes=n_ep, seed=seed,
                         max_steps=WINDOW, warmup_steps=[WARMUP_UAV, 0],
                         budget_total=int(budget), csi=T12.make_csi(seed, n_ep),
                         cost_fn=COSTS, diagnostic=False, counterfactual=False,
                         predictor=predictor, pred_probe=probe)


def _J(records, n_ep):
    return np.asarray(window_cost(records, 2, window=WINDOW, absorbed=ABSORBED,
                                 n_episodes=n_ep), dtype=float)


def _seq(records):
    return [(r.episode, r.t, r.served_idx, r.attempted_idx, r.delivered_idx) for r in records]


class SpyAllocator:
    """把收到的 View 抄一份（含 `last_obs`），选择交给内层臂 —— 只用来验"预测没进信息集"。"""

    def __init__(self, inner):
        self.inner = inner
        self.seen: list = []

    def __call__(self, view, rng):
        self.seen.append((view.t, [(s.age, float(s.cost_est), np.asarray(s.last_obs).copy())
                                  for s in view.slots]))
        return self.inner(view, rng)


class SpyController:
    """记录 `act()` 的**返回值**（= 真正传给 `env.step` 的那个 action）。"""

    def __init__(self, inner):
        self.inner = inner
        self.log: list[list] = [[], []]

    def reset(self, est=None):
        self.inner.reset(est)

    def act(self, est, i=None):
        a = self.inner.act(est)
        return a


class TaggedController(SpyController):
    """带用户下标的 Spy（runner 只调 `act(est)`，所以用两个不同实例分别记录）。"""

    def __init__(self, inner, i):
        super().__init__(inner)
        self.i = i

    def act(self, est):
        a = self.inner.act(est)
        self.log[self.i].append(np.asarray(a).copy())
        return a


class ServeAt:
    """只在给定步号服务指定用户（用于构造"送 2 帧 → 断服"的确定性时序）。"""

    def __init__(self, user: int, steps: set[int]):
        self.user = int(user)
        self.steps = set(steps)

    def __call__(self, view, rng):
        served = np.zeros(len(view.slots), dtype=bool)
        if int(view.t) in self.steps and view.slots[self.user].alive:
            served[self.user] = True
        return Allocation(served=served, reason=f"serve@{sorted(self.steps)}")


class ShadowOnly(TrueDyn):
    """**阴性对照**：照常 `deepcopy` + 前推影子，但**把输出丢掉**，返回 `last_obs`。

    用处（S7）：若影子前推碰了主 env 或主 rng，轨迹就会变；否则必须与 `predictor=None`
    逐位相同。没有这条对照，"TrueDyn 没污染主轨迹"只是一句声明。
    """

    name = "shadow-only"

    def _predict(self, i, st, *, age, actions_since):
        super()._predict(i, st, age=age, actions_since=actions_since)   # 真前推，丢弃结果
        return st["last_obs"]


def _err(p, i, which):
    a = np.asarray(p[f"est_{which}"][i], dtype=float)
    b = np.asarray(p[f"true_{which}"][i], dtype=float)
    return float(np.linalg.norm(a - b))


# ================================================================ S1
def s1_zoh_equals_none() -> dict:
    print("\n== S1 predictor=None 与 ZOH 逐位相同（打「接线改了既有行为」）==")
    r0 = _run(200, "age", "unpaced", 0, 2)
    r1 = _run(200, "age", "unpaced", 0, 2, predictor=ZOH())
    J0, J1 = _J(r0["records"], 2), _J(r1["records"], 2)
    same_seq = _seq(r0["records"]) == _seq(r1["records"])
    same_J = bool(np.array_equal(J0, J1))
    check("S1 served/attempted/delivered 序列逐位相同", same_seq, f"n={len(r0['records'])} 步")
    check("S1 逐 episode J 逐位相同", same_J,
          f"max|ΔJ|={float(np.max(np.abs(J0 - J1))):.3e}")
    return {"same_seq": same_seq, "same_J": same_J}


# ================================================================ S2
def s2_zoh_36cell_regression() -> dict:
    print("\n== S2 ZOH 重跑 T21 的 36 cell，与 results/52 逐 episode J 逐位相同（回归红线）==")
    path = latest_result_path("52_t21_paced_matching_ablation")
    print(f"  [ref] {path.relative_to(ROOT).as_posix() if path else '<缺失>'}")
    if path is None or not path.exists():
        check("S2 能找到 results/52 的当前证据", False, "<无参考>")
        return {"found": False}
    ref = json.loads(path.read_text(encoding="utf-8"))
    ref_map = {(c["budget"], c["mode"], c["arm"], c["seed"]): c["J_per_episode"]
               for c in ref.get("cells_J", [])}
    t0 = time.time()
    diffs, compared, n_ep = [], 0, 30
    for budget in (100, 200):
        for mode in MODES:
            for arm in ARMS:
                for seed in (0, 1, 2):
                    r = _run(budget, arm, mode, seed, n_ep, predictor=ZOH())
                    a = _J(r["records"], n_ep)
                    b = np.asarray(ref_map[(budget, mode, arm, seed)], dtype=float)
                    compared += 1
                    if a.shape != b.shape or not np.array_equal(a, b):
                        diffs.append(f"B={budget}/{mode}/{arm}/seed{seed} "
                                     f"max|Δ|={np.nanmax(np.abs(a - b)):.3e}")
    print(f"  [S2] {compared} cells，{time.time() - t0:.1f}s")
    check("S2 36 cell 逐 episode J 与 results/52 逐位相同（ZOH ≡ 现状）",
          compared == 36 and not diffs,
          f"{compared} cell 逐位相同" if not diffs else "；".join(diffs[:4]))
    return {"compared": compared, "diffs": diffs}


# ================================================================ S3
def s3_no_privilege_leak() -> dict:
    print("\n== S3 预测器不读特quan（diagnostic 开关不改变轨迹；状态里无 env 对象）==")
    pred_a = Linear()
    r_a = _run(100, "age", "unpaced", 0, 2, predictor=pred_a)
    # diagnostic=True：runner 会把真值塞进 View.privileged（MaxAge 不读它）
    envs = [T12.make_env("uav", seed=0, max_steps=WARMUP_UAV + WINDOW),
            T12.make_env("CartPole-v1", seed=0)]
    ctrls = [T12.PDRelativeController(), T12.CartPoleLQR(K)]
    pred_b = Linear()
    try:
        r_b = run_multiuser(envs, ctrls, MaxAge(), SINRSuccess(T12.Q_REF), n_episodes=2, seed=0,
                            max_steps=WINDOW, warmup_steps=[WARMUP_UAV, 0], budget_total=100,
                            csi=T12.make_csi(0, 2), cost_fn=COSTS,
                            diagnostic=True, counterfactual=False, predictor=pred_b)
    finally:
        for e in envs:
            e.close()
    Ja, Jb = _J(r_a["records"], 2), _J(r_b["records"], 2)
    same = bool(np.array_equal(Ja, Jb)) and _seq(r_a["records"]) == _seq(r_b["records"])
    check("S3 diagnostic=True/False 下 Linear 轨迹与 J 逐位相同（真值没被读走）", same,
          f"max|ΔJ|={float(np.max(np.abs(Ja - Jb))):.3e}")

    env_like = [v for v in pred_a.state_refs() if hasattr(v, "step") and hasattr(v, "reset")]
    check("S3 Linear 的内部状态里不存在任何 env 对象（结构上拿不到真值）", not env_like,
          f"env 型对象 {len(env_like)} 个")
    return {"same_traj": same, "env_refs": len(env_like)}


# ================================================================ S4
def s4_truedyn_exact() -> dict:
    print("\n== S4 TrueDyn 真的在推演（正对照：误差精确 0；反对照：ZOH 误差 > 0）==")
    log_t: list = []
    _run(200, "age", "unpaced", 0, 2, predictor=TrueDyn(), probe=log_t.append)
    log_z: list = []
    _run(200, "age", "unpaced", 0, 2, predictor=ZOH(), probe=log_z.append)

    def collect(log):
        return [(_err(p, i, "pre")) for p in log for i in range(2)
                if p["alive"][i] and p["age"][i] >= 1]

    et, ez = collect(log_t), collect(log_z)
    ok_t = bool(et) and max(et) == 0.0
    ok_z = bool(ez) and np.median(ez) > 0.0
    check("S4a TrueDyn 在 age≥1 步的预测误差**恒等于 0**（精确 0，无容差）",
          ok_t, f"n={len(et)} max={max(et) if et else float('nan'):.3e}")
    check("S4b 反对照：ZOH 在同样步上误差 > 0（否则这条检查没有分辨力）",
          ok_z, f"n={len(ez)} median={np.median(ez) if ez else float('nan'):.3e}")
    return {"truedyn_n": len(et), "truedyn_max": max(et) if et else None,
            "zoh_median": float(np.median(ez)) if ez else None}


# ================================================================ S5
def s5_reset_and_gap() -> dict:
    print("\n== S5 送达即重置 + 断服期误差增长（送 2 帧 → 断服，确定性送达）==")
    log: list = []
    r = _run(200, "age", "unpaced", 0, 1, predictor=ZOH(), probe=log.append,
             allocator=ServeAt(0, {1, 2}), success=ConstantSuccess(1.0))
    dl = {}
    for rec in r["records"]:
        if rec.delivered_idx >= 0:
            dl.setdefault(rec.delivered_idx, []).append(rec.t)
    check("S5a 用户 0 只在 t=1,2 送达（时序被强制，后续 step 无帧）",
          dl.get(0) == [1, 2], f"delivered 步={dl.get(0)}")
    post1 = [_err(p, 0, "post") for p in log if p["t"] in (1, 2)]
    check("S5b 送达步的 est 误差 = 0（送达即重置，且重置用的是本步 nxt）",
          post1 and max(post1) == 0.0, f"t∈{{1,2}} 误差={['%.3e' % x for x in post1]}")
    pre = {p["t"]: _err(p, 0, "pre") for p in log if p["alive"][0]}
    grow = [pre.get(t, float("nan")) for t in (3, 4, 7)]
    ok_grow = (pre.get(3) == 0.0) and all(np.isfinite(grow)) and grow[1] > 0 and grow[2] > grow[1]
    check("S5c 断服期：t=3 误差 0（仍是最新观测），此后严格增长", ok_grow,
          f"err(3,4,7)={['%.4f' % x for x in grow]}")
    bad_len = [p["t"] for p in log if any(p["n_since_before"][i] != p["age"][i]
                                          for i in range(2) if p["alive"][i])]
    check("S5d 每一步 actions_since 长度 == age（差一步就会「预测错步数」）",
          not bad_len, f"违规步 {bad_len[:5]}" if bad_len else "全部满足")
    return {"delivered_steps": dl.get(0), "err_3_4_7": [float(x) for x in grow]}


# ================================================================ S6
def s6_action_provenance() -> dict:
    print("\n== S6 actions_since 用的是**实际施加**的动作（SpyController 逐项对账）==")
    envs = [T12.make_env("uav", seed=0, max_steps=WARMUP_UAV + WINDOW),
            T12.make_env("CartPole-v1", seed=0)]
    ctrls = [TaggedController(T12.PDRelativeController(), 0),
             TaggedController(T12.CartPoleLQR(K), 1)]
    log: list = []
    try:
        # 恒 0 成功概率 ⇒ 全程丢帧；动作照常施加 ⇒ 缓冲必须一直增长、不得被清空
        r = run_multiuser(envs, ctrls, MaxAge(), ConstantSuccess(0.0), n_episodes=1, seed=0,
                          max_steps=WINDOW, warmup_steps=[WARMUP_UAV, 0], budget_total=WINDOW,
                          csi=T12.make_csi(0, 1), cost_fn=COSTS, predictor=Linear(),
                          pred_probe=log.append)
    finally:
        for e in envs:
            e.close()
    n_deliv = sum(1 for x in r["records"] if x.delivered_idx >= 0)
    expect = {i: [p["actions"][i] for p in log if p["alive"][i]] for i in range(2)}
    # ★ 预热期也调 act()（用户 0 预热 300 步）⇒ 对账必须跳过预热那一段，
    #   否则拿"主循环的动作"去比"含预热的调用序列"，会差 300 项（本轮实测踩到）。
    warm = [WARMUP_UAV, 0]
    got = {i: ctrls[i].log[i][warm[i]:] for i in range(2)}
    ok_act = all(len(expect[i]) == len(got[i]) and
                 all(np.array_equal(a, b) for a, b in zip(expect[i], got[i]))
                 for i in range(2))
    check("S6a 主循环逐步动作与 controller.act() 的返回值逐项相同", ok_act,
          f"步数 uav={len(expect[0])}/{len(got[0])} cart={len(expect[1])}/{len(got[1])}")
    check("S6b 全程丢帧（delivered=0）时动作仍入缓冲：n_since == age 恒成立",
          n_deliv == 0 and all(p["n_since_before"][i] == p["age"][i]
                               for p in log for i in range(2) if p["alive"][i]),
          f"delivered={n_deliv}")
    # ★ 诚实边界：本环境下行理想（est 直接喂控制器），"实际施加" ≡ "命令" ~ 同值，
    #   本检查验的是**接线取自 act() 返回**，不是"下行有损时仍正确"（预注册 §3-4 / 限制 R1）。
    return {"delivered": int(n_deliv), "act_match": bool(ok_act)}


# ================================================================ S7
def s7_shadow_no_side_effect() -> dict:
    print("\n== S7 影子前推不污染主轨迹（阴性对照：影子照跑、输出丢弃）==")
    r0 = _run(100, "rr", "unpaced", 0, 1)
    r1 = _run(100, "rr", "unpaced", 0, 1, predictor=ShadowOnly())
    J0, J1 = _J(r0["records"], 1), _J(r1["records"], 1)
    same = _seq(r0["records"]) == _seq(r1["records"]) and bool(np.array_equal(J0, J1))
    check("S7 影子跑但丢弃输出 ⇒ 与 predictor=None 逐位相同（主 env/rng 未被触碰）",
          same, f"max|ΔJ|={float(np.max(np.abs(J0 - J1))):.3e}")
    return {"same": same}


# ================================================================ S8
def s8_guards() -> dict:
    print("\n== S8 口径外推必须抛错（不许静默降级成「能跑但错」）==")
    leaks: list[str] = []
    obs = np.zeros(6, dtype=np.float32)

    def must_raise(tag, fn):
        try:
            fn()
            leaks.append(tag)
        except (ValueError, TypeError):
            pass

    p = Linear()
    must_raise("未送达就 predict", lambda: p.predict(0, age=0, actions_since=[]))
    p.observe_delivery(0, obs, age_before=0)
    must_raise("age<0", lambda: p.predict(0, age=-1, actions_since=[]))
    must_raise("actions_since 太短", lambda: p.predict(0, age=2, actions_since=[np.zeros(2)]))
    must_raise("TrueDyn 无 env 快照", lambda: TrueDyn().observe_delivery(0, obs, age_before=0))
    must_raise("未知预测器名", lambda: make_predictor("nope"))
    td = TrueDyn()
    td.observe_delivery(0, obs, age_before=0, env=T12.make_env("uav", seed=0, max_steps=10))
    td.predict(0, age=2, actions_since=[np.zeros(2), np.zeros(2)])
    must_raise("age 回退（送达后未重建快照）",
               lambda: td.predict(0, age=1, actions_since=[np.zeros(2)]))
    check("S8 六个反例全部抛错", not leaks, "全部抛错" if not leaks else f"漏网 {leaks}")

    # 反向对照：合法调用不许误炸（否则"全都抛错"毫无信息）
    ok = True
    try:
        q = ZOH()
        q.observe_delivery(0, obs, age_before=0)
        q.predict(0, age=0, actions_since=[])
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"    ★ 合法调用误炸：{exc!r}")
    check("S8b 反对照：合法调用不得抛错（否则上面的「全抛」无分辨力）", ok, "ZOH age=0 正常")
    return {"leaks": leaks}


# ================================================================ S9
def s9_allocator_view_unchanged() -> dict:
    """预测只改 `est`（控制器输入），**不得**改分配器的 `View`（信息集边界）。

    反对照：若把预测值写进 `last_obs`/`cost_est`，分配器就会"免费拿到未来"，
    而所有臂（含 rr）的表现都会变好 —— 那会让 T23 的收益表完全失去意义。
    """
    print("\n== S9 预测不进分配器信息集（View.last_obs 仍是送达值）==")
    spy = SpyAllocator(MaxAge())
    log: list = []
    _run(100, "age", "unpaced", 0, 1, predictor=Linear(), probe=log.append, allocator=spy)
    truth = {(p["episode"], p["t"]): p["true_pre"] for p in log}
    viol, checked = [], 0
    for t, slots in spy.seen:
        tp = truth.get((0, t))
        if tp is None:
            continue
        for i, (_age, _cost, last_obs) in enumerate(slots):
            if _age >= 2:                      # age=0/1 时"保持值"本来就可能接近真值，不作判据
                checked += 1
                if np.allclose(last_obs, tp[i], atol=0.0, rtol=0.0):
                    viol.append((t, i, _age))
    check("S9 age≥2 时分配器看到的 last_obs 与真值**不同**（预测没有泄漏进 View）",
          checked > 0 and not viol,
          f"{checked} 个 (t,i) 检查，违例 {viol[:3]}" if viol else f"{checked} 个 (t,i) 全部为送达值")
    return {"checked": checked, "violations": [list(v) for v in viol[:5]]}


# ================================================================ 接线冒烟（非证据）
def wiring_smoke() -> None:
    """★ **非证据**：2 集 × 1 种子 × `age` 臂 × B=200，只打印逐集 J。

    存在的理由：预注册 §0 规定本轮不出收益表，但"接对了"这件事最好有一个可复算的指纹。
    这一节的数字**不得**引用为结论（样本量 2、环境确定性、单臂），T23 才按预注册出表。
    """
    print("\n== 接线冒烟（★ 非证据：2 集 × 1 种子 × age 臂 × B=200；收益表留 T23）==")
    for name, pred in (("none", None), ("zoh", ZOH()), ("linear", Linear()), ("truedyn", TrueDyn())):
        r = _run(200, "age", "unpaced", 0, 2, predictor=pred)
        J = _J(r["records"], 2)
        print(f"  {name:8s} J_uav={np.round(J[:, 0], 4).tolist()} "
              f"J_cart={np.round(J[:, 1], 4).tolist()}")


# ================================================================ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-zoh-regression", action="store_true")
    ap.add_argument("--wiring-smoke", action="store_true",
                    help="额外打印 2 集 × 1 种子的逐集 J（★ 非证据，只作接线指纹）")
    args = ap.parse_args()
    t0 = time.time()
    print(f"[53] T22 预测器时序/信息集确认｜窗口 {WINDOW}｜B={BUDGETS}｜N=2 异质")
    print("     ★ 本轮不出收益表（预注册 §0/§4）；任何 J 只用作「是否逐位相同」的指纹。")

    out = {"S1": s1_zoh_equals_none()}
    if args.skip_zoh_regression:
        print("\n== S2 已跳过（--skip-zoh-regression）==")
        out["S2"] = {"skipped": True}
    else:
        out["S2"] = s2_zoh_36cell_regression()
    out["S3"] = s3_no_privilege_leak()
    out["S4"] = s4_truedyn_exact()
    out["S5"] = s5_reset_and_gap()
    out["S6"] = s6_action_provenance()
    out["S7"] = s7_shadow_no_side_effect()
    out["S8"] = s8_guards()
    out["S9"] = s9_allocator_view_unchanged()
    if args.wiring_smoke:
        wiring_smoke()

    print(f"\n{'ALL CHECKS PASS' if not FAILS else 'FAILED: ' + '; '.join(FAILS)}")
    print(f"[53] 耗时 {time.time() - t0:.1f}s｜检查项 {sum(1 for _ in out)} 组"
          f"｜失败 {len(FAILS)}")
    return 0 if not FAILS else 2


if __name__ == "__main__":
    sys.exit(main())
