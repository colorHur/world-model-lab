#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""T23-a · `ARX` 预测器 + 两个机制分解诊断臂的**只读自检**—— ★ 本轮不出收益表。

口径来源（★ 在看任何数字之前冻结）
----------------------------------
`04-实验与代码/2026-10-19-T23a-ARX预测器-预注册-v0.1.md` §2–§5（含 §4b 的 A1 阈值修正）。

这一轮只回答一件事
------------------
`ARX` 是不是一个"公平、接得对、会显式退化"的预测器；两个诊断臂的维度拆分是不是与实现一致。
**不回答"预测值多少"** —— 那是 T23-b，须另写预注册。

用法
----
    D:/devtools/miniconda3/envs/ai-lab/python.exe scripts/54_t23a_arx_predictor.py
    ... --skip-regression     # 跳过 A7（真跑 2 集 × 2 次，约 2 s；留着也无妨）

产物
----
    **不写任何文件**（R17 / T20-P0-3：只跑自检的入口不得触碰入库证据）。
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from wmlab.eval.multiuser import default_uav_cost, window_cost  # noqa: E402
from wmlab.eval.predictors import (  # noqa: E402
    ARX, Linear, LinearSelfOnly, LinearTargetOnly, ZOH, make_predictor,
)

FAILS: list[str] = []


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


T12 = _load(ROOT / "scripts" / "42_t12_hetero_four_arms.py", "t12_mod")
T52 = _load(ROOT / "scripts" / "52_t21_paced_matching_ablation.py", "t52_mod")

WINDOW = T12.WINDOW
WARMUP_UAV = T12.WARMUP_UAV
COSTS = [default_uav_cost, T12.cartpole_cost]
ABSORBED = [T12.ABSORBED_UAV, T12.ABSORBED_CART]
K = T12.load_lqr_gain()


def check(tag: str, ok: bool, detail: str) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {tag}: {detail}")
    if not ok:
        FAILS.append(f"{tag}: {detail}")


# ================================================================ 装置
def _run(budget, arm, mode, seed, n_ep, *, predictor=None, probe=None):
    envs = [T12.make_env("uav", seed=seed, max_steps=WARMUP_UAV + WINDOW),
            T12.make_env("CartPole-v1", seed=seed)]
    ctrls = [T12.PDRelativeController(), T12.CartPoleLQR(K)]
    try:
        return T12.run_multiuser(
            envs, ctrls, T52.make_alloc(arm, mode, budget), T12.SINRSuccess(T12.Q_REF),
            n_episodes=n_ep, seed=seed, max_steps=WINDOW,
            warmup_steps=[WARMUP_UAV, 0], budget_total=int(budget),
            csi=T12.make_csi(seed, n_ep), cost_fn=COSTS, diagnostic=False,
            counterfactual=False, predictor=predictor, pred_probe=probe)
    finally:
        for e in envs:
            e.close()


def _J(records, n_ep):
    return np.asarray(window_cost(records, 2, window=WINDOW, absorbed=ABSORBED,
                                 n_episodes=n_ep), dtype=float)


def _seq(records):
    return [(r.episode, r.t, r.served_idx, r.attempted_idx, r.delivered_idx) for r in records]


def _trace(predictor, *, budget=200, arm="age", seed=0, n_ep=1):
    """返回每步每用户的 `est_pre`（决策时点送进控制器的估计）。"""
    log: list = []
    _run(budget, arm, "unpaced", seed, n_ep, predictor=predictor, probe=log.append)
    return log


def _same_est(la, lb, *, users) -> tuple[bool, int]:
    """只比较指定用户（★ A5/A6 只关 CartPole：UAV 上这两个臂**本来就该不同**）。"""
    if len(la) != len(lb):
        return False, -1
    n = 0
    for pa, pb in zip(la, lb):
        if pa["t"] != pb["t"]:
            return False, -2
        for i in users:
            n += 1
            if not np.array_equal(np.asarray(pa["est_pre"][i]), np.asarray(pb["est_pre"][i])):
                return False, n
    return True, n


def _n_diff_est(la, lb, *, users) -> int:
    """有多少个 (t,user) 的估计值**不同**（"必须不同"的阴性对照用）。"""
    n = 0
    for pa, pb in zip(la, lb):
        for i in users:
            if not np.array_equal(np.asarray(pa["est_pre"][i]), np.asarray(pb["est_pre"][i])):
                n += 1
    return n


# ================================================================ 合成系统
_SHUFFLE_SEED = 20261019


def _synth(seed: int = 7, dim: int = 3, act_dim: int = 2, n: int = 40, shuffle: bool = False):
    """线性系统 `x⁺ = A x + B a + c`：生成 `n` 个 `gap==1` 训练对 + 一个查询动作。"""
    rng = np.random.default_rng(seed)
    A = rng.normal(0.0, 0.35, (dim, dim))
    A = A / max(np.max(np.abs(np.linalg.eigvals(A))), 1e-12) * 0.9
    B = rng.normal(0.0, 0.5, (dim, act_dim))
    c = rng.normal(0.0, 0.2, dim)
    x = rng.normal(0.0, 1.0, dim)
    xs, acts, ys = [], [], []
    for _ in range(n):
        a = rng.normal(0.0, 1.0, act_dim)
        y = A @ x + B @ a + c
        xs.append(x.copy()); acts.append(a.copy()); ys.append(y.copy())
        x = y
    a_query = rng.normal(0.0, 2.0, act_dim)
    if shuffle:                      # ★ 只打乱"动作↔样本"的配对，不动 rng 的后续抽样
        perm = np.random.default_rng(_SHUFFLE_SEED).permutation(n)
        acts = [acts[j] for j in perm]
    return A, B, c, xs, acts, ys, a_query


def _fit_arx(xs, acts, ys, clamp=None) -> ARX:
    """按"先一帧、再逐帧送达"的时序喂 `gap==1` 对（与 runner 的调用顺序一致）。"""
    p = ARX() if clamp is None else clamp
    p.observe_delivery(0, xs[0], age_before=0)
    for k in range(len(ys)):
        p.observe_delivery(0, ys[k], age_before=0, action=acts[k])
    return p


# ================================================================ A1 / A2
def a1_a2_learns() -> dict:
    print("\n== A1/A2 ARX 在合成线性系统上**能学到**真 A/B（正对照 + 打乱动作的反对照）==")
    out = {}
    for tag, shuffle in (("A1", False), ("A2", True)):
        A, B, c, xs, acts, ys, a_q = _synth(shuffle=shuffle)
        p = _fit_arx(xs, acts, ys)
        x_q = np.asarray(p._st[0]["last_obs"], dtype=np.float64)
        pred = np.asarray(p.predict(0, age=1, actions_since=[a_q]), dtype=np.float64)
        true = A @ x_q + B @ a_q + c
        rel = float(np.linalg.norm(pred - true) / np.linalg.norm(true))
        out[tag] = rel
        print(f"  [{tag}] n_samples={p.describe()['users']['0']['n_samples']} "
              f"rel_err={rel:.3e}")
    e1, e2 = out["A1"], out["A2"]
    check("A1 正对照：一步预测相对误差 < 2e-5（预注册 §4b，ridge 地板 4.5e-6）",
          e1 < 2e-5, f"rel={e1:.3e}")
    check("A2 反对照：动作打乱后误差 > 1e-2，且 A2/A1 ≥ 1e3（A1 有分辨力）",
          e2 > 1e-2 and e2 / max(e1, 1e-300) >= 1e3,
          f"rel={e2:.3e}，比值={e2 / max(e1, 1e-300):.1f}×")
    return out


# ================================================================ A3 / A4
def a3_a4_degrade_and_gap() -> dict:
    print("\n== A3/A4 样本不足必须**显式退化**为 Linear；`gap>1` 的对不得入拟合 ==")
    A, B, c, xs, acts, ys, a_q = _synth(n=3)          # 3 < 2·dim = 6
    p_arx = _fit_arx(xs, acts, ys)
    p_lin = _fit_arx(xs, acts, ys, clamp=Linear())    # 同一串观测喂给 Linear，逐位对比用
    o1 = np.asarray(p_arx.predict(0, age=1, actions_since=[a_q]))
    o2 = np.asarray(p_lin.predict(0, age=1, actions_since=[a_q]))
    d = p_arx.describe()
    check("A3a 样本 3 < 2·3 时输出与 Linear **逐位相同**", bool(np.array_equal(o1, o2)),
          f"max|Δ|={float(np.max(np.abs(o1 - o2))):.3e}")
    check("A3b 退化状态在 describe() 里**显式可见**（不是静默降级）",
          d["users"]["0"]["degraded"] is True and d["degraded_users"] == [0],
          f"degraded_users={d['degraded_users']} n={d['users']['0']['n_samples']}"
          f"/{d['users']['0']['n_needed']}")

    q = ARX()
    q.observe_delivery(0, xs[0], age_before=0)
    q.observe_delivery(0, ys[0], age_before=1, action=acts[0])     # gap = 2
    dq = q.describe()
    n_s = sum(u.get("n_samples", 0) for u in dq["users"].values())
    check("A4 只喂 gap==2 的对 ⇒ n_samples 不加（只计数 gap_gt1）",
          n_s == 0 and dq["total_gap_gt1"] == 1,
          f"n_samples={n_s} gap_gt1={dq['total_gap_gt1']}")
    return {"degraded": d["degraded_users"]}


# ================================================================ A5 / A6
def a5_a6_cartpole_split() -> dict:
    print("\n== A5/A6 诊断臂的维度拆分：CartPole（4 维）没有目标分量 ⇒ TargetOnly≡ZOH ==")
    CART = (1,)      # ★ 只比较 CartPole 用户：UAV（user 0）上这两个臂**本来就该不同**
    base = _trace(ZOH())
    tgt = _trace(LinearTargetOnly())
    same_t, n_t = _same_est(base, tgt, users=CART)
    check("A5a LinearTargetOnly 对 CartPole 与 ZOH **逐位相同**", same_t,
          f"{n_t} 个 (t,user) 估计值逐位比较")

    lin = _trace(Linear())
    slf = _trace(LinearSelfOnly())
    same_s, n_s = _same_est(lin, slf, users=CART)
    check("A6a LinearSelfOnly 对 CartPole 与 Linear **逐位相同**", same_s,
          f"{n_s} 个 (t,user) 估计值逐位比较")

    # ★ 分辨力对照（#39）：若拆分根本没生效，A5a/A6a 会"因为两条都退化成同一个东西"而 PASS
    d_zt = _n_diff_est(base, tgt, users=(0,))
    d_ls = _n_diff_est(lin, slf, users=(0,))
    check("A5b 对照：在 **UAV**（有目标分量）上 TargetOnly 与 ZOH **必须不同**",
          d_zt > 0, f"{d_zt} 个 (t,user) 不同")
    check("A6b 对照：在 **UAV** 上 SelfOnly 与 Linear **必须不同**", d_ls > 0,
          f"{d_ls} 个 (t,user) 不同")

    d = LinearTargetOnly()
    d.observe_delivery(1, np.zeros(4, dtype=np.float32), age_before=0)
    check("A6c 4 维观测被**显式**标为「无目标分量」（不假装它有）",
          d.describe()["target_component"]["4"] is False,
          f"target_component={d.describe()['target_component']}")
    return {"target_eq_zoh": same_t, "self_eq_linear": same_s,
            "uav_diff_target": d_zt, "uav_diff_self": d_ls}


# ================================================================ A7
def a7_none_equals_zoh() -> dict:
    print("\n== A7 回归红线：`predictor=None` 仍与 ZOH 逐位相同（ARX 接线没改既有路径）==")
    r0 = _run(200, "age", "unpaced", 0, 2)
    r1 = _run(200, "age", "unpaced", 0, 2, predictor=ZOH())
    J0, J1 = _J(r0["records"], 2), _J(r1["records"], 2)
    same = _seq(r0["records"]) == _seq(r1["records"]) and bool(np.array_equal(J0, J1))
    check("A7 事件序列与逐集 J 逐位相同", same,
          f"max|ΔJ|={float(np.max(np.abs(J0 - J1))):.3e}")
    return {"same": bool(same)}


# ================================================================ A8
def a8_info_set() -> dict:
    print("\n== A8 信息集：ARX 与两个诊断臂都**拿不到** env（结构上不可能偷真值）==")
    preds = [ARX(), LinearTargetOnly(), LinearSelfOnly()]
    ok_flag = all(getattr(p, "consumes_env", True) is False for p in preds)
    check("A8a consumes_env 全为 False", ok_flag,
          f"{[getattr(p, 'consumes_env', None) for p in preds]}")
    env_like = [v for p in preds for v in p.state_refs()
                if hasattr(v, "step") and hasattr(v, "reset")]
    print(f"  [A8b] 状态值 {sum(len(p.state_refs()) for p in preds)} 个，"
          f"其中 env 型对象 {len(env_like)} 个（空）")
    check("A8b 状态里不存在 env 对象", not env_like, f"env 型 {len(env_like)} 个")

    names = {"zoh", "linear", "truedyn", "arx", "linear_target_only", "linear_self_only"}
    got = {n for n in names if make_predictor(n) is not None}
    check("A8c make_predictor 认全六个名字", got == names, f"缺 {sorted(names - got)}")
    return {"env_refs": len(env_like)}


# ================================================================ A9
def a9_real_wiring() -> dict:
    """真环境接线 + `gap==1` 样本**可行性统计**（★ 非收益）。

    A9b 是**可行性门**而不是程序正确性门：若两个用户全程都停在退化态，说明本协议下
    `ARX ≡ Linear`，"用 ARX 排除『弱预测器』解释"这件事就做不到 —— 这是必须带进 T23-b 的
    事实（预注册 §7-2），不是可以修到 exit 0 的程序错误。
    """
    print("\n== A9 真环境接线 + gap==1 样本可行性统计（★ 非收益）==")
    pred = ARX()
    t0 = time.time()
    log = _trace(pred)
    dt = time.time() - t0
    d = pred.describe()
    print(f"  [A9] {len(log)} 步，{dt:.2f}s")
    print(f"       describe: {d}")

    finite = all(np.all(np.isfinite(np.asarray(x))) for p in log for x in p["est_pre"])
    check("A9a 全程预测输出有限（无 NaN/inf）", finite, f"{len(log)} 步 × 2 用户")

    n_fit = {k: (v["n_samples"], v["n_needed"]) for k, v in d["users"].items()}
    check("A9b（可行性门）至少一个用户真正进入拟合态；否则 ARX≡Linear，须带入 T23-b 解释",
          any(not v["degraded"] for v in d["users"].values()),
          f"degraded_users={d['degraded_users']} n_samples/need={n_fit} "
          f"gap1={d['total_gap1']} gap_gt1={d['total_gap_gt1']}")

    # ★ 非证据：只报"预测得多准"（L2 误差中位数），不报任何调度收益
    print("  [A9c] ★ 非收益·只作测量：age≥1 时估计值 vs 真值的 L2 误差中位数")
    meds = {}
    for nm, mk in (("zoh", ZOH), ("linear", Linear), ("arx", ARX)):
        lg = _trace(mk())
        per = {}
        for i in range(2):
            es = [float(np.linalg.norm(np.asarray(p["est_pre"][i], dtype=float)
                                       - np.asarray(p["true_pre"][i], dtype=float)))
                  for p in lg if p["alive"][i] and p["age"][i] >= 1]
            per[i] = float(np.median(es)) if es else float("nan")
        meds[nm] = per
        print(f"        {nm:7s} uav={per[0]:.4f} cart={per[1]:.4f}  (n={sum(1 for p in lg if p['age'][0]>=1)})")
    return {"describe": d, "median_err": meds, "seconds": dt}


# ================================================================ A9d
def _gap_profile(budget: int, predictor, seed: int = 0) -> dict:
    """同一配置下把"预热末次观测免费可得"记为 t=0 的一次送达，取相邻送达步差分布。"""
    r = _run(budget, "age", "unpaced", seed, 1, predictor=predictor)
    recs = r["records"]
    per = {}
    for i in range(2):
        steps = [x.t for x in recs if x.delivered_idx == i]
        gaps = np.diff(np.asarray([0] + steps, dtype=int)) if steps else np.zeros(0, int)
        per[i] = {"frames": len(steps), "gap1": int((gaps == 1).sum()),
                  "gap2": int((gaps == 2).sum()), "gap3": int((gaps == 3).sum()),
                  "gap_max": int(gaps.max()) if gaps.size else 0,
                  "gap_median": float(np.median(gaps)) if gaps.size else float("nan")}
    both_alive = sum(1 for x in recs if all(x.alive))
    n_step = max(x.t for x in recs)
    return {"per_user": per, "both_alive_steps": int(both_alive), "n_step": int(n_step),
            "frames_total": int(sum(v["frames"] for v in per.values()))}


def _print_profile(tag: str, prof: dict, budget: int) -> None:
    print(f"  [{tag} B={budget}] 两用户同时存活 {prof['both_alive_steps']}/{prof['n_step']} 步")
    for i, v in prof["per_user"].items():
        print(f"       u{i}: 帧 {v['frames']:3d}｜gap=1 {v['gap1']:3d}  gap=2 {v['gap2']:3d}  "
              f"gap=3 {v['gap3']:3d}｜max {v['gap_max']:2d}｜中位 {v['gap_median']:.1f}")


def a9d_gap_profile() -> dict:
    """送达**间隔分布**——解释 A9b 为什么失败（★ 只关"能否辨识"，与收益无关）。

    ★ 必须与 A9b **同一配置**（`predictor=ARX()`，即本轮要用的那一路）测；
      第一版我用的是 `predictor=None`（ZOH）那一路，CartPole 早死、UAV 连续被服务，
      `gap==1` 占 63% —— 那个数字**不属于** A9b 的语境（#46 的同型：验收点指错对象）。
      两种配置都打印出来，判据只锁在 ARX 那一路上。
    """
    print("\n== A9d 送达间隔分布：`gap==1` 在**本轮要用的那一路**上还剩多少 ==")
    out = {}
    for budget in (100, 200):
        p_arx = _gap_profile(budget, ARX())
        out[f"arx_B{budget}"] = p_arx
        _print_profile("arx", p_arx, budget)
    for budget in (200,):
        p_zoh = _gap_profile(budget, None)
        out[f"zoh_B{budget}"] = p_zoh
        _print_profile("zoh（对照，非判据）", p_zoh, budget)

    tot1 = sum(v["gap1"] for k in ("arx_B100", "arx_B200") for v in out[k]["per_user"].values())
    tot = sum(out[k]["frames_total"] for k in ("arx_B100", "arx_B200"))
    check("A9d 在 `predictor=ARX` 的两档上，gap==1 的送达对占比 < 2%"
          "（⇒ 一步映射的辨识数据几乎不存在）",
          tot > 0 and tot1 / tot < 0.02,
          f"gap1={tot1} / 全部送达 {tot} = {tot1 / max(tot, 1):.2%}")
    return out


# ================================================================ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-regression", action="store_true", help="跳过 A7")
    args = ap.parse_args()
    t0 = time.time()
    print(f"[54] T23-a ARX 预测器自检｜窗口 {WINDOW}｜2 用户（uav+cartpole）｜臂 age")
    print("     ★ 本轮不出收益表（预注册 §0）；任何 J 只用作「是否逐位相同」的指纹。")

    out = {}
    out["A1A2"] = a1_a2_learns()
    out["A3A4"] = a3_a4_degrade_and_gap()
    out["A5A6"] = a5_a6_cartpole_split()
    if args.skip_regression:
        print("\n== A7 已跳过（--skip-regression）==")
    else:
        out["A7"] = a7_none_equals_zoh()
    out["A8"] = a8_info_set()
    out["A9"] = a9_real_wiring()
    out["A9d"] = a9d_gap_profile()

    print(f"\n{'ALL CHECKS PASS' if not FAILS else 'FAILED: ' + '; '.join(FAILS)}")
    print(f"[54] 耗时 {time.time() - t0:.1f}s｜失败 {len(FAILS)}")
    return 0 if not FAILS else 2


if __name__ == "__main__":
    sys.exit(main())
