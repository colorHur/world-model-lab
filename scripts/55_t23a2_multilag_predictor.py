#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""T23-a-2 · 分滞后阶模型组 `MultiLag` 的**只读自检 + 可行性统计**—— ★ 本轮仍不出收益表。

口径来源（★ 在看任何数字之前冻结）
----------------------------------
`04-实验与代码/2026-10-19-T23a2-分滞后阶模型-预注册-v0.1.md` §2–§5。

这一轮只回答一件事
------------------
`MultiLag` 是不是一个"公平、接得对、会显式退化、且在这套协议下**真有 `g≥2` 的辨识数据**"的预测器。
**不回答"预测值多少"** —— 收益表是 T23-b，须另写预注册。

用法
----
    D:/devtools/miniconda3/envs/ai-lab/python.exe scripts/55_t23a2_multilag_predictor.py
    ... --skip-regression     # 跳过 B7（真跑 2 集 × 2 次，约 3 s）

产物
----
    **不写任何文件**（R17 / T20-P0-3：只跑自检的入口不得触碰入库证据）。
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from wmlab.eval.multiuser import default_uav_cost, window_cost  # noqa: E402
from wmlab.eval.predictors import Linear, MultiLag, ZOH, make_predictor  # noqa: E402

FAILS: list[str] = []

#: B1 阈值（★ 预注册 §3.1b：首次运行本脚本**之前**已用同一实现做过量级代验）
#: 实测 rel_pred=1.68e-06、rel_A2=1.10e-05、rel_[AB,B]=1.57e-06（float32 观测是误差地板）
#: ⇒ 取 1e-4（比最差实测宽 ~9×）；原写 1e-5 只差 10%，属未做代验就写阈值（#50 同型）
B1_REL_THR = 1e-4
B2_GAIN_THR = 1e2
#: B8 研究可行性门（预注册 §4-3）：`age≥1` 的加权覆盖率
COVERAGE_GATE = 0.20


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


T12 = _load(ROOT / "scripts" / "42_t12_hetero_four_arms.py", "t12_mod55")
T52 = _load(ROOT / "scripts" / "52_t21_paced_matching_ablation.py", "t52_mod55")

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


# ================================================================ 合成系统（gap=2）
_SHUFFLE_SEED = 20261019


def _synth(seed: int = 11, dim: int = 3, act_dim: int = 2, n_steps: int = 100):
    """线性系统 `x⁺ = A x + B a + c`；`acts[j]` 把 `xs[j]` 推到 `xs[j+1]`。

    `gap==2` 的对 = `(x_k, [a_k, a_{k+1}], x_{k+2})`，真映射
    `x_{k+2} = A²x_k + AB a_k + B a_{k+1} + (Ac + c)`。
    """
    rng = np.random.default_rng(seed)
    A = rng.normal(0.0, 0.35, (dim, dim))
    A = A / max(np.max(np.abs(np.linalg.eigvals(A))), 1e-12) * 0.9
    B = rng.normal(0.0, 0.5, (dim, act_dim))
    c = rng.normal(0.0, 0.2, dim)
    xs = [rng.normal(0.0, 1.0, dim)]
    acts = []
    for _ in range(n_steps):
        a = rng.normal(0.0, 1.0, act_dim)
        acts.append(a)
        xs.append(A @ xs[-1] + B @ a + c)
    return A, B, c, xs, acts


def _pairs_gaps(xs, acts, gaps):
    """按**一条送达时间线**切段：`gaps=[2,3,...]` ⇒ 段首尾不重叠（与真实流一致）。"""
    s, out = 0, []
    for g in gaps:
        s2 = s + g
        out.append((xs[s], [acts[j].copy() for j in range(s, s2)], xs[s2]))
        s = s2
    return out


def _feed(p, start_x, pairs):
    """按 runner 的调用顺序喂：先首帧，再逐段送达（`age_before = len(seq)−1` ⇒ `gap_prev = g`）。"""
    p.observe_delivery(0, start_x, age_before=0)
    for _x, seq, y in pairs:
        p.observe_delivery(0, y, age_before=len(seq) - 1, action=seq[-1], action_seq=seq)
    return p


# ================================================================ B1 / B2
def b1_b2_learns_gap2() -> dict:
    print("\n== B1/B2 合成线性系统上 `M_2` 必须恢复 `A²` 与 `[AB, B]`（正/反对照）==")
    out = {}
    for tag, shuffle in (("B1", False), ("B2", True)):
        A, B, c, xs, acts = _synth()
        pairs = _pairs_gaps(xs, acts, [2] * 40)
        if shuffle:                  # ★ 只打乱"动作块 ↔ 样本"的配对，不动其它任何东西
            perm = np.random.default_rng(_SHUFFLE_SEED).permutation(len(pairs))
            pairs = [(x, pairs[j][1], y) for (x, _s, y), j in zip(pairs, perm)]
        p = _feed(MultiLag(), xs[0], pairs)
        d = p.describe()["users"]["0"]["ages"]["2"]
        rng = np.random.default_rng(999)
        x_q = np.asarray(pairs[-1][2], dtype=np.float64)
        aq = [rng.normal(0.0, 2.0, B.shape[1]) for _ in range(2)]
        pred = np.asarray(p.predict(0, age=2, actions_since=aq), dtype=np.float64)
        true = A @ (A @ x_q + B @ aq[0] + c) + B @ aq[1] + c
        rel_pred = float(np.linalg.norm(pred - true) / np.linalg.norm(true))

        W = p._fit[0][2]["W"]
        dim, ad = B.shape
        A_hat = W[:dim].T
        # ★ 行向量约定：`a1 @ W[dim:dim+ad] = (AB) a1` ⇒ `W[dim:dim+ad].T == AB`（(dim, ad)）
        B_hat = np.vstack([W[dim:dim + ad].T, W[dim + ad:dim + 2 * ad].T])
        B_true = np.vstack([A @ B, B])
        rel_A = float(np.linalg.norm(A_hat - A @ A) / np.linalg.norm(A @ A))
        rel_B = float(np.linalg.norm(B_hat - B_true) / np.linalg.norm(B_true))
        out[tag] = {"rel_pred": rel_pred, "rel_A": rel_A, "rel_B": rel_B,
                    "n_samples": d["n_samples"], "F": d["F"]}
        print(f"  [{tag}] n={d['n_samples']}/F={d['F']}  rel_pred={rel_pred:.3e}  "
              f"rel_A2={rel_A:.3e}  rel_[AB,B]={rel_B:.3e}")

    e1, e2 = out["B1"], out["B2"]
    worst = max(e1["rel_pred"], e1["rel_A"], e1["rel_B"])
    check(f"B1 正对照：三处相对误差均 < {B1_REL_THR:.0e}（阈值依据见预注册 §3.1b）",
          worst < B1_REL_THR, f"max={worst:.3e}")
    gain = e2["rel_pred"] / max(e1["rel_pred"], 1e-300)
    check(f"B2 反对照：动作块打乱后误差 ≥ {B2_GAIN_THR:.0f}× B1（B1 有分辨力）",
          e2["rel_pred"] > 1e-3 and gain >= B2_GAIN_THR,
          f"rel={e2['rel_pred']:.3e}，比值={gain:.1f}×")
    return out


# ================================================================ B3
def b3_no_model_for_age() -> dict:
    print("\n== B3 没有 `M_1` 时，`age=1` 必须与 `Linear` **逐位相同** ==")
    _A, B, _c, xs, acts = _synth()
    pairs = _pairs_gaps(xs, acts, [2] * 40)
    p = _feed(MultiLag(), xs[0], pairs)
    lin = _feed(Linear(), xs[0], pairs)
    aq = [np.random.default_rng(5).normal(0.0, 1.0, B.shape[1])]
    o1 = np.asarray(p.predict(0, age=1, actions_since=aq))
    o2 = np.asarray(lin.predict(0, age=1, actions_since=aq))
    d = p.describe()["users"]["0"]
    check("B3a 输出逐位相同", bool(np.array_equal(o1, o2)),
          f"max|Δ|={float(np.max(np.abs(o1 - o2))):.3e}")
    check("B3b `describe().degraded_ages` 含 1（显式退化，不是静默）",
          1 in d["degraded_ages"],
          f"degraded_ages={d['degraded_ages']} usable_ages={d['usable_ages']}")
    return {"degraded_ages": d["degraded_ages"], "usable_ages": d["usable_ages"]}


# ================================================================ B4 / B5
def b4_b5_mixed_and_counts() -> dict:
    print("\n== B4/B5 混合 age **不拼装**；样本计数逐项对得上 ==")
    _A, _B, _c, xs, acts = _synth()
    gaps = [2] * 12 + [3] * 12 + [8]          # 8 > max_gap ⇒ 只计数
    pairs = _pairs_gaps(xs, acts, gaps)
    p = _feed(MultiLag(), xs[0], pairs)
    fed = dict(collections.Counter(gaps))
    d = p.describe()["users"]["0"]
    print(f"  喂入: gap2×12 gap3×12 gap8×1 ⇒ describe: "
          f"n2={d['ages'].get('2', {}).get('n_samples')} "
          f"n3={d['ages'].get('3', {}).get('n_samples')} "
          f"gap_gt_max={d.get('gap_gt_max')} pairs={d.get('pairs')}")
    check("B5a `g=2/3` 的样本数精确等于喂入对数（12/12）",
          d["ages"]["2"]["n_samples"] == 12 and d["ages"]["3"]["n_samples"] == 12,
          f"{d['ages']['2']['n_samples']}/{d['ages']['3']['n_samples']}")
    check("B5b `gap > max_gap` 只计数、不入拟合",
          d.get("gap_gt_max") == 1 and "8" not in d["ages"],
          f"gap_gt_max={d.get('gap_gt_max')} ages={sorted(d['ages'])}")
    check("B5c 恒等式：pairs == Σn_samples + gap_gt_max + no_action + n_len_mismatch",
          d.get("pairs") == (sum(v["n_samples"] for v in d["ages"].values())
                             + d.get("gap_gt_max", 0) + d.get("no_action", 0)
                             + d.get("n_len_mismatch", 0)),
          f"pairs={d.get('pairs')} Σn={sum(v['n_samples'] for v in d['ages'].values())}")

    lin = _feed(Linear(), xs[0], pairs)
    aq5 = [np.random.default_rng(7).normal(0.0, 1.0, acts[0].size) for _ in range(5)]
    o5 = np.asarray(p.predict(0, age=5, actions_since=aq5))
    l5 = np.asarray(lin.predict(0, age=5, actions_since=aq5))
    d2 = p.describe()["users"]["0"]
    check("B4a `age=5`（无 `M_5`）与 `Linear` **逐位相同**（不拼装 2+3）",
          bool(np.array_equal(o5, l5)), f"max|Δ|={float(np.max(np.abs(o5 - l5))):.3e}")
    check("B4b 计数上可区分：`used_degraded[5] > 0` 且 `used_model[5] == 0`",
          d2["used_degraded"].get("5", 0) > 0 and d2["used_model"].get("5", 0) == 0,
          f"used_model={d2['used_model']} used_degraded={d2['used_degraded']}")
    aq3 = aq5[:3]
    p.predict(0, age=3, actions_since=aq3)
    d3 = p.describe()["users"]["0"]
    check("B4c 对照：`age=3`（有 `M_3`）必须真的走模型（否则 B4a 无分辨力）",
          d3["used_model"].get("3", 0) > 0 and d3["used_degraded"].get("3", 0) == 0,
          f"used_model[3]={d3['used_model'].get('3')} "
          f"used_degraded[3]={d3['used_degraded'].get('3', 0)}")
    return {"fed": fed}


# ================================================================ B6 / B8
def _gap_hist_from_records(records, i: int) -> collections.Counter:
    """从记录里独立数"相邻送达的步差"。

    ★ 时点口径（读实现核对过，`multiuser.py` 里 `t` 先自增再决策 ⇒ **记录的 `t` 从 1 开始**）：
      预热末次观测对应 `t=0`（`observe_delivery(..., age_before=0)` 在循环之前），
      所以首个送达的间隔是 `t_first − 0`，不是 `t_first − (−1)`。
      第一版写成 `prev = −1`，两个用户各差 1 档 ⇒ B6 假 FAIL（#49 同型：区间端点没对齐）。
    """
    steps = [x.t for x in records if x.delivered_idx == i]
    prev, out = 0, collections.Counter()
    for s in steps:
        out[int(s - prev)] += 1
        prev = s
    return out


def b6_b8_real(pred: MultiLag, budget: int, seed: int = 0) -> dict:
    print(f"\n== B6/B8 真环境（B={budget}, seed={seed}, 1 集）：接线对账 + 可行性统计（★ 非收益）==")
    log: list = []
    t0 = time.time()
    r = _run(budget, "age", "unpaced", seed, 1, predictor=pred, probe=log.append)
    dt = time.time() - t0
    recs = r["records"]
    d = pred.describe()
    print(f"  [B6] {len(log)} 步，{dt:.2f}s；seq_mismatch={d['seq_mismatch']} "
          f"seq_last_mismatch={d['seq_last_mismatch']}")

    for i in range(2):
        u = d["users"].get(str(i), {})
        hist = _gap_hist_from_records(recs, i)
        fits = {int(g): v["n_samples"] for g, v in u.get("ages", {}).items()}
        over = int(sum(c for g, c in hist.items() if g > pred.max_gap))
        print(f"       u{i}: 记录侧步差直方图 {dict(sorted(hist.items()))} "
              f"｜拟合侧 n_samples {dict(sorted(fits.items()))} ｜>max_gap {over}")
        ok = all(fits.get(g, 0) == c for g, c in hist.items() if 1 <= g <= pred.max_gap) and \
            over == u.get("gap_gt_max", 0)
        check(f"B6.{i} 记录侧与拟合侧的 gap 直方图**逐档相等**", ok,
              f"记录={dict(sorted((g, c) for g, c in hist.items() if 1 <= g <= pred.max_gap))} "
              f"拟合={fits}")

    check("B6a 全程 `action_seq` 长度 == `gap_prev`（0 次不符）",
          d["seq_mismatch"] == 0, f"seq_mismatch={d['seq_mismatch']}")
    check("B6b `action_seq[-1]` 与 runner 传的单步 `action` 逐位一致",
          d["seq_last_mismatch"] == 0, f"seq_last_mismatch={d['seq_last_mismatch']}")

    tot_model = tot_deg = 0
    n_post = sum(1 for p in log for i in range(2) if p["alive"][i] and p["age"][i] >= 1
                 and p["t"] > budget)
    n_pre = sum(1 for p in log for i in range(2) if p["alive"][i] and p["age"][i] >= 1
                and p["t"] <= budget)
    print(f"  [B8] age≥1 的决策点：预算内 {n_pre}、预算后（t>{budget}，R17）{n_post}")
    for i in range(2):
        u = d["users"].get(str(i), {})
        um, ud = u.get("used_model", {}), u.get("used_degraded", {})
        ages = sorted({int(k) for k in list(um) + list(ud)} - {0})
        nm = nd = 0
        rows = []
        for a in ages:
            mm, dd = um.get(str(a), 0), ud.get(str(a), 0)
            nm += mm
            nd += dd
            if a <= 8:
                rows.append(f"age={a:2d} 模型 {mm:3d}/退化 {dd:3d}")
            else:
                rows.append(None)
        tail_m = sum(um.get(str(a), 0) for a in ages if a > 8)
        tail_d = sum(ud.get(str(a), 0) for a in ages if a > 8)
        print(f"       u{i}｜" + "｜".join(r for r in rows if r) +
              (f"｜age>8 模型 {tail_m}/退化 {tail_d}" if any(r is None for r in rows) else ""))
        tot_model += nm
        tot_deg += nd
        print(f"       u{i} 合计 age≥1：模型 {nm}／{nm + nd} = {nm / max(nm + nd, 1):.1%}")
    cov = tot_model / max(tot_model + tot_deg, 1)
    print(f"  [B8] 加权覆盖率（两用户合并、age≥1）= {cov:.1%}")
    check(f"B8 可行性门：age≥1 的加权覆盖率 ≥ {COVERAGE_GATE:.0%}（预注册 §4-3）",
          cov >= COVERAGE_GATE, f"coverage={cov:.1%} 模型 {tot_model}／{tot_model + tot_deg}")

    finite = all(np.all(np.isfinite(np.asarray(x))) for p in log for x in p["est_pre"])
    check("B8b 全程预测输出有限（无 NaN/inf）", finite, f"{len(log)} 步 × 2 用户")
    return {"describe": d, "coverage": cov, "seconds": dt, "records": recs}


# ================================================================ B8c（只测量）
def b8c_error_measure(budget: int) -> dict:
    """★ 非收益、非证据：只报"预测与真值的 L2 误差中位数"，按 age 分档。

    用途只有一个：给下一轮判断"这个混合预测器是否比 `Linear` 更准"。收益表仍是 T23-b，
    且**不得**把这里的误差当收益（预注册 §3 末段；A9c 同性质）。
    """
    print(f"\n== B8c（★ 非收益·只作测量）B={budget}：age≥1 时估计值与真值的 L2 误差中位数 ==")
    out = {}
    for nm, mk in (("mlag", MultiLag), ("linear", Linear), ("zoh", ZOH)):
        log: list = []
        _run(budget, "age", "unpaced", 0, 1, predictor=mk(), probe=log.append)
        bands = {"1": [], "2-3": [], ">=4": []}
        for p in log:
            for i in range(2):
                if not p["alive"][i] or p["age"][i] < 1:
                    continue
                err = float(np.linalg.norm(np.asarray(p["est_pre"][i], dtype=float)
                                           - np.asarray(p["true_pre"][i], dtype=float)))
                a = int(p["age"][i])
                bands["1" if a == 1 else ("2-3" if a <= 3 else ">=4")].append(err)
        out[nm] = {k: (round(float(np.median(v)), 4) if v else None, len(v))
                   for k, v in bands.items()}
        print(f"       {nm:6s} " + "  ".join(f"{k}: {v[0]} (n={v[1]})"
                                            for k, v in out[nm].items()))
    return out


# ================================================================ B7
def b7_none_equals_zoh() -> dict:
    print("\n== B7 回归红线：`predictor=None` 仍与 ZOH 逐位相同（本轮的接线没改既有路径）==")
    r0 = _run(200, "age", "unpaced", 0, 2)
    r1 = _run(200, "age", "unpaced", 0, 2, predictor=ZOH())
    J0, J1 = _J(r0["records"], 2), _J(r1["records"], 2)
    same = _seq(r0["records"]) == _seq(r1["records"]) and bool(np.array_equal(J0, J1))
    check("B7 事件序列与逐集 J 逐位相同", same, f"max|ΔJ|={float(np.max(np.abs(J0 - J1))):.3e}")
    return {"same": bool(same)}


# ================================================================ B9
def b9_info_set() -> dict:
    print("\n== B9 信息集：`MultiLag` 拿不到 env / View（结构上不可能偷真值）==")
    p = MultiLag()
    p.observe_delivery(0, np.zeros(6, dtype=np.float32), age_before=0,
                       action=np.zeros(2, dtype=np.float32),
                       action_seq=[np.zeros(2, dtype=np.float32)])
    env_like = [v for v in p.state_refs() if hasattr(v, "step") and hasattr(v, "reset")]
    check("B9a consumes_env 为 False 且状态里 0 个 env 型对象",
          p.consumes_env is False and not env_like,
          f"consumes_env={p.consumes_env} env型={len(env_like)} 状态值={len(p.state_refs())}")
    ok = make_predictor("mlag") is not None and make_predictor("mlag").name == "mlag"
    check("B9b `make_predictor('mlag')` 可造", ok, f"{make_predictor('mlag').describe()['name']}")
    return {"env_refs": len(env_like)}


# ================================================================ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-regression", action="store_true", help="跳过 B7")
    args = ap.parse_args()
    t0 = time.time()
    print(f"[55] T23-a-2 分滞后阶模型自检｜窗口 {WINDOW}｜2 用户（uav+cartpole）｜臂 age")
    print("     ★ 本轮不出收益表（预注册 §0）；任何 J 只用作「是否逐位相同」的指纹。")

    b1_b2_learns_gap2()
    b3_no_model_for_age()
    b4_b5_mixed_and_counts()
    if args.skip_regression:
        print("\n== B7 已跳过（--skip-regression）==")
    else:
        b7_none_equals_zoh()
    b9_info_set()
    for budget in (100, 200):
        b6_b8_real(MultiLag(), budget)
        b8c_error_measure(budget)

    print(f"\n{'ALL CHECKS PASS' if not FAILS else 'FAILED: ' + '; '.join(FAILS)}")
    print(f"[55] 耗时 {time.time() - t0:.1f}s｜失败 {len(FAILS)}")
    return 0 if not FAILS else 2


if __name__ == "__main__":
    sys.exit(main())
