#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P3-0 · 确认性评估：`Linear` 预测补偿的收益在**未参与选择**的种子上能否复现。

口径来源（★ 全部在看结果之前冻结）
------------------------------------
`04-实验与代码/2026-10-20-P3-0-确认评估-预注册-v0.1.md` §0–§8。

一句话
------
方法在 T24 已**冻结**为 `Linear`（P1 里唯一无结构偏差、全年龄可用的预测器）。
本轮只问一件事：换成从未参与任何选择的种子/轨迹后，`R_uav < 1` 还在不在、区间多宽。

设计（预注册 §2）
-----------------
    臂 = age；predictor ∈ {zoh, linear, hybrid, truedyn}
    mode ∈ {unpaced, paced}；budget ∈ {100, 200}
    主确认种子 {100,130,160}（初值 100..189，90 个互不相同）
    次确认种子 {100,101,102}（初值 100..131，只有 32 个 ⇒ 只能比"区间宽度"，不是独立证据）
    episodes=30, max_steps=200, absorbed=(25,1), GATE=0.90（全部不动）

★ 主表 = B=200 unpaced（`t>B` 发帧恒 0）。B=100 是**失配边界**，单独成节且必须同报 `t>B` 发帧数。
★ bootstrap 的重采样单位是**不同初值（cluster）**，不是 episode 次（预注册 §1b）。
★ `TrueDyn` 一律标 `diagnostic`。

判据（预注册 §4；**看数之前定死**）
-----------------------------------
    F1 主确认集 B=200 unpaced：`R_uav` 的 95% 区间**上界 < 1.0** 且 `R_cart` 点估计 ≤ 1.10
    否定条件：`R_uav` 区间上界 ≥ 1.0 ⇒ 记"未获确认"，不换设置重试

用法
----
    D:/devtools/miniconda3/envs/ai-lab/python.exe scripts/58_p3_confirmation.py
    ... --skip-experiment      # 只跑自检（不写 results/）
    ... --smoke                # 2 集 × 1 种子 × 2 预测器，不写 results/

产物
----
    outputs/58_p3_confirmation[_smoke].json                  完整表（不入库）
    results/runs/<run_id>/58_p3_confirmation.json            **入库**（results_io 唯一入口）
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

SMOKE_TAG = ""
sys.path.insert(0, str(ROOT))

from wmlab.eval.multiuser import default_uav_cost, delivery_rate, window_cost  # noqa: E402
from wmlab.eval.predictors import Linear, ZOH, make_predictor  # noqa: E402
from wmlab.eval.results_io import latest_result_path, write_result  # noqa: E402

FAILS: list[str] = []

AGE_BUCKETS = (("1", 1, 1), ("2-3", 2, 3), (">=4", 4, 10 ** 6))

#: ★ 预注册 §1b：初值指纹由 `reset(seed + e)` 决定，主集必须 90 个互不相同
SEED_SETS = {"primary": (100, 130, 160), "secondary": (100, 101, 102)}
DEV_SEEDS = (0, 1, 2)                      # 开发集：只用来验"不相交"，不参与任何判据
N_EP = 30


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


T12 = _load(ROOT / "scripts" / "42_t12_hetero_four_arms.py", "t12_mod58")
T52 = _load(ROOT / "scripts" / "52_t21_paced_matching_ablation.py", "t52_mod58")

WINDOW = T12.WINDOW
WARMUP_UAV = T12.WARMUP_UAV
BUDGETS = list(T12.BUDGETS)
GATE = T12.GATE
ABSORBED = [T12.ABSORBED_UAV, T12.ABSORBED_CART]
COSTS = [default_uav_cost, T12.cartpole_cost]
ARM = "age"
PREDICTORS = ("zoh", "linear", "hybrid", "truedyn")
MODES = ("unpaced", "paced")
N_BOOT = 10000
BOOT_SEED = 20261020


def check(tag: str, ok: bool, detail: str) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {tag}: {detail}")
    if not ok:
        FAILS.append(f"{tag}: {detail}")


# ================================================================ 装置
class _Timed:
    """透明计时包装：只为量 `predict()` 的墙钟耗时（预注册 §3-4），不改变任何行为。

    ★ 透明性由 C9 证明（包/不包逐 episode J 逐位相同）。`__getattr__` 转发其余属性，
      否则 runner 的 `consumes_env` 探针会静默变成 False（#46 同族的"对象指错"）。
    """

    def __init__(self, inner) -> None:
        self._inner = inner
        self.dts: list[int] = []

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def reset_episode(self) -> None:
        return self._inner.reset_episode()

    def observe_delivery(self, *a, **kw) -> None:
        return self._inner.observe_delivery(*a, **kw)

    def predict(self, *a, **kw):
        t0 = time.perf_counter_ns()
        out = self._inner.predict(*a, **kw)
        self.dts.append(time.perf_counter_ns() - t0)
        return out

    def timing(self) -> dict:
        if not self.dts:
            return {"n_calls": 0}
        v = np.asarray(self.dts, dtype=float) / 1000.0
        return {"n_calls": int(v.size),
                "median_us": float(np.median(v)),
                "mean_us": float(v.mean()),
                "p90_us": float(np.percentile(v, 90)),
                "max_us": float(v.max()),
                "note": "Python 层实现开销，不是实时预算结论"}


def _run(budget, mode, seed, n_ep, *, predictor=None, probe=None):
    envs = [T12.make_env("uav", seed=seed, max_steps=WARMUP_UAV + WINDOW),
            T12.make_env("CartPole-v1", seed=seed)]
    ctrls = [T12.PDRelativeController(), T12.CartPoleLQR(_run.K)]
    alloc = T52.make_alloc(ARM, mode, budget)
    try:
        return T12.run_multiuser(
            envs, ctrls, alloc, T12.SINRSuccess(T12.Q_REF),
            n_episodes=n_ep, seed=seed, max_steps=WINDOW,
            warmup_steps=[WARMUP_UAV, 0], budget_total=int(budget),
            csi=T12.make_csi(seed, n_ep), cost_fn=COSTS, diagnostic=False,
            counterfactual=False, predictor=predictor, pred_probe=probe)
    finally:
        for e in envs:
            e.close()


_run.K = T12.load_lqr_gain()


def _J(records, n_ep):
    return np.asarray(window_cost(records, 2, window=WINDOW, absorbed=ABSORBED,
                                  n_episodes=n_ep), dtype=float)


class _ErrProbe:
    """逐决策点采"预测误差 + age 分档"，只在内存里聚合（同 T23-b 口径）。"""

    def __init__(self) -> None:
        self.err: dict[int, dict[str, list]] = {i: {b[0]: [] for b in AGE_BUCKETS} for i in range(2)}
        self.n_age: dict[int, dict[int, int]] = {0: {}, 1: {}}

    def __call__(self, p: dict) -> None:
        for i in (0, 1):
            if not p["alive"][i]:
                continue
            a = int(p["age"][i])
            self.n_age[i][a] = int(self.n_age[i].get(a, 0)) + 1
            if a == 0:
                continue
            e = float(np.linalg.norm(np.asarray(p["est_pre"][i], dtype=np.float64)
                                     - np.asarray(p["true_pre"][i], dtype=np.float64)))
            for tag, lo, hi in AGE_BUCKETS:
                if lo <= a <= hi:
                    self.err[i][tag].append(e)
                    break

    def summary(self) -> dict:
        out = {}
        for i in (0, 1):
            buckets = {}
            for tag, _lo, _hi in AGE_BUCKETS:
                v = np.asarray(self.err[i][tag], dtype=float)
                buckets[tag] = ({"n": int(v.size)} if v.size == 0 else
                                {"n": int(v.size), "median": float(np.median(v)),
                                 "mean": float(v.mean()), "p90": float(np.percentile(v, 90)),
                                 "max": float(v.max())})
            out[str(i)] = {"buckets": buckets,
                           "n_age": {str(k): int(v) for k, v in sorted(self.n_age[i].items())}}
        return out


# ================================================================ cluster bootstrap
def cluster_bootstrap(rows, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    """`R_i = Σ J_i(num) / Σ J_i(den)` 的 cluster bootstrap 区间（重采样单位 = 不同初值）。

    Args:
        rows: `[{"cluster": int, "J_num": [j0, j1], "J_den": [j0, j1]}, ...]`
              —— `J_num`/`J_den` 是**同一条 episode** 上两个预测器的代价（配对）。
    """
    cl = sorted({int(r["cluster"]) for r in rows})
    idx_by_cl = {c: [] for c in cl}
    for k, r in enumerate(rows):
        idx_by_cl[int(r["cluster"])].append(k)
    # 扁平化：每个 cluster 的行索引放进一行（不足处填 −1），采样时整行取、再滤掉 −1
    maxs = max(len(v) for v in idx_by_cl.values())
    mat = np.full((len(cl), maxs), -1, dtype=np.int64)
    for k, c in enumerate(cl):
        mat[k, :len(idx_by_cl[c])] = idx_by_cl[c]
    Jn = np.asarray([r["J_num"] for r in rows], dtype=float)
    Jd = np.asarray([r["J_den"] for r in rows], dtype=float)

    def _ratio(ii):
        dn = Jd[ii].sum(axis=0)
        safe = np.where(dn > 0, dn, 1.0)
        return np.where(dn > 0, Jn[ii].sum(axis=0) / safe, np.nan)

    point = _ratio(np.arange(len(rows), dtype=np.int64))
    rng = np.random.default_rng(int(seed))
    draws = np.full((n_boot, 2), np.nan)
    n_bad = 0
    for b in range(n_boot):
        pick = rng.integers(0, len(cl), len(cl))
        ii = mat[pick].ravel()
        v = _ratio(ii[ii >= 0])
        if not np.all(np.isfinite(v)):
            n_bad += 1
        draws[b] = v
    ci_lo = np.nanpercentile(draws, 2.5, axis=0)
    ci_hi = np.nanpercentile(draws, 97.5, axis=0)
    per_ep = np.where(Jd > 0, Jn / np.where(Jd > 0, Jd, 1.0), np.nan)
    return {"R": [float(x) for x in point],
            "ci95": [[float(x) for x in ci_lo], [float(x) for x in ci_hi]],
            "geo_R": float(np.sqrt(point[0] * point[1])),
            "n_clusters": len(cl), "n_rows": len(rows), "n_boot": int(n_boot),
            "n_boot_nonfinite": int(n_bad),
            "per_episode_ratio_median": [float(np.nanmedian(per_ep[:, i])) for i in (0, 1)],
            "boot_seed": int(seed)}


# ================================================================ 自检
def selfchecks(n_ep_small: int = 2) -> dict:
    print("\n== 自检 C1–C9（预注册 §6：每条写明能打死什么错）==")
    out: dict = {}

    # ---------------- C5：初值指纹（★ 最先跑：它守的是 P3 的定义）----------------
    def _fp(seeds, n_ep):
        return {int(s) + e for s in seeds for e in range(n_ep)}
    fp_pri = _fp(SEED_SETS["primary"], N_EP)
    fp_sec = _fp(SEED_SETS["secondary"], N_EP)
    fp_dev = _fp(DEV_SEEDS, N_EP)
    out["C5"] = {"primary_n": len(fp_pri), "secondary_n": len(fp_sec), "dev_n": len(fp_dev),
                 "primary∩dev": sorted(fp_pri & fp_dev),
                 "secondary∩dev": sorted(fp_sec & fp_dev),
                 "primary∩secondary": len(fp_pri & fp_sec)}
    check("C5 主确认集 90 个初值、次确认集 32 个、两者与开发集 0..31 **交集为空**",
          len(fp_pri) == 90 and len(fp_sec) == 32 and not (fp_pri & fp_dev)
          and not (fp_sec & fp_dev),
          json.dumps(out["C5"], ensure_ascii=False))
    print(f"       ↳ 次确认集与开发集交集={len(fp_sec & fp_dev)}；主∩次={len(fp_pri & fp_sec)}"
          f"（共用 seed=100 的 30 集 ⇒ 两口径不是独立证据，只能比区间宽度）")

    # ---------------- C1：None ≡ ZOH ----------------
    seed0 = SEED_SETS["primary"][0]
    j_none = _J(_run(200, "unpaced", seed0, n_ep_small, predictor=None)["records"], n_ep_small)
    j_zoh = _J(_run(200, "unpaced", seed0, n_ep_small, predictor=ZOH())["records"], n_ep_small)
    d1 = float(np.max(np.abs(j_none - j_zoh))) if j_none.size else 0.0
    check("C1 `predictor=None` 与 `ZOH` 逐 episode J **逐位相同**（ZOH 列=真基线）",
          d1 == 0.0, f"max|Δ|={d1:.3e}")

    # ---------------- C2：连跑两次逐位相同 ----------------
    a1 = _J(_run(200, "unpaced", seed0, 5, predictor=Linear())["records"], 5)
    a2 = _J(_run(200, "unpaced", seed0, 5, predictor=Linear())["records"], 5)
    check("C2 同一 cell 连跑两次 J **逐位相同**（无隐藏时间/RNG 依赖）",
          np.array_equal(a1, a2), f"max|Δ|={float(np.max(np.abs(a1 - a2))):.3e}")

    # ---------------- C9：计时包装器透明 ----------------
    b1 = _J(_run(200, "unpaced", seed0, n_ep_small, predictor=Linear())["records"], n_ep_small)
    b2 = _J(_run(200, "unpaced", seed0, n_ep_small, predictor=_Timed(Linear()))["records"],
            n_ep_small)
    check("C9 计时包装后的 Linear 与不包逐 episode J **逐位相同**",
          np.array_equal(b1, b2), f"max|Δ|={float(np.max(np.abs(b1 - b2))):.3e}")

    # ---------------- C3：TrueDyn 误差精确 0 / ZOH >0 ----------------
    c3 = {}
    for name in ("truedyn", "zoh"):
        pr = _ErrProbe()
        _run(200, "unpaced", seed0, n_ep_small, predictor=make_predictor(name), probe=pr)
        s = pr.summary()
        n = sum(v["buckets"][t]["n"] for v in s.values() for t, _lo, _hi in AGE_BUCKETS)
        mx = max([v["buckets"][t].get("max", 0.0) for v in s.values()
                  for t, _lo, _hi in AGE_BUCKETS] or [0.0])
        c3[name] = {"n_pred_points": n, "max_err": mx}
    check("C3 TrueDyn 在 age≥1 上误差**精确 0**（n>0）且 ZOH 误差 >0（打「静默退化成 ZOH」）",
          c3["truedyn"]["max_err"] == 0.0 and c3["truedyn"]["n_pred_points"] > 0
          and c3["zoh"]["max_err"] > 0.0, str(c3))
    out["C3"] = c3

    # ---------------- C4：记账（小规模 1 cell）----------------
    r4 = _run(100, "unpaced", seed0, n_ep_small, predictor=ZOH())
    dr4 = delivery_rate(r4["records"], 2)
    served_total = sum(1 for x in r4["records"] if x.attempted_idx is not None
                       and x.attempted_idx >= 0)
    ok4 = (served_total == n_ep_small * 100
           and r4["budget_reserved"] == r4["budget_attempted"] + r4["budget_wasted"])
    r4p = _run(100, "paced", seed0, n_ep_small, predictor=ZOH())
    late_un = sum(1 for x in r4["records"] if x.t > 100 and x.attempted_idx is not None
                  and x.attempted_idx >= 0)
    late_pa = sum(1 for x in r4p["records"] if x.t > 100 and x.attempted_idx is not None
                  and x.attempted_idx >= 0)
    check("C4 记账：served_total==episodes×budget、reserved==attempted+wasted、"
          "unpaced B<L 的 t>B 发帧=0、paced 的 >0",
          ok4 and late_un == 0 and late_pa > 0,
          f"served={served_total}/{n_ep_small * 100} late_un={late_un} late_paced={late_pa} "
          f"reserved={r4['budget_reserved']} att={r4['budget_attempted']} "
          f"waste={r4['budget_wasted']}")
    out["C4"] = {"served_total": served_total, "late_unpaced": late_un, "late_paced": late_pa}

    # ---------------- C7：bootstrap 自身有分辨力 ----------------
    # (a) 比值恒 0.5 ⇒ 宽度 0、点估计 0.5
    rows_a = [{"cluster": k // 3, "J_num": [0.5 * (1 + k), 0.5 * (2 + k)],
               "J_den": [1.0 + k, 2.0 + k]} for k in range(30)]
    ba = cluster_bootstrap(rows_a, n_boot=500)
    w0 = float(max(ba["ci95"][1][i] - ba["ci95"][0][i] for i in (0, 1)))
    ok_a = w0 < 1e-12 and abs(ba["R"][0] - 0.5) < 1e-12 and abs(ba["R"][1] - 0.5) < 1e-12

    def _coverage(sig_jd: float, sig_ratio: float, n_cl: int, n_rep: int, n_boot: int) -> float:
        """真值 R=0.8 的合成覆盖率。

        ★ 修正（2026-10-20 05:0x，跑主实验前）：`lognormal(0, σ)` 的均值是 `exp(σ²/2) > 1`，
          首版直接乘它 ⇒ **真值不是 0.8 而是 `0.8·exp(σ²/2)`**（σ=0.2 时 0.816，σ=0.35 时 0.850），
          覆盖率自然掉到 0.80–0.84。**那是判据的真值写错，不是 bootstrap 欠覆盖。**
          改为 `lognormal(−σ²/2, σ)` ⇒ `E[w]=1`，真值严格是 0.8。
        """
        rng = np.random.default_rng(7)
        cover = 0
        for _rep in range(n_rep):
            jd = rng.lognormal(0.0, sig_jd, (n_cl, 2))
            jn = 0.8 * jd * rng.lognormal(-0.5 * sig_ratio ** 2, sig_ratio, (n_cl, 2))
            rows_b = [{"cluster": c, "J_num": jn[c].tolist(), "J_den": jd[c].tolist()}
                      for c in range(n_cl)]
            rb = cluster_bootstrap(rows_b, n_boot=n_boot)
            if rb["ci95"][0][0] <= 0.8 <= rb["ci95"][1][0]:
                cover += 1
        return cover / n_rep

    # (b) 轻尾、cluster 数 = 本设计的主确认集（90）⇒ 必须接近名义 95%
    rate = _coverage(0.25, 0.20, 90, 120, 200)
    ok_b = 0.90 <= rate <= 0.99
    # (d) 重尾 + 小样本（σ=0.6、40 个 cluster）：★ 只报告，不判 PASS/FAIL ——
    #     percentile bootstrap 在重尾小样本下已知会欠覆盖，这不是实现错误；
    #     写进结果文档的限制节，用来说明"区间上界 <1"的可靠性边界。
    rate_heavy = _coverage(0.60, 0.35, 40, 120, 200)

    # (c) 点估计 == 直接 np.sum
    rows_c = [{"cluster": k // 4, "J_num": [1.0 + k * 0.1, 2.0 + k * 0.2],
               "J_den": [2.0 + k * 0.1, 4.0 + k * 0.2]} for k in range(40)]
    bc = cluster_bootstrap(rows_c, n_boot=200)
    dir_r = [(sum(r["J_num"][i] for r in rows_c)) / (sum(r["J_den"][i] for r in rows_c))
             for i in (0, 1)]
    ok_c = max(abs(bc["R"][i] - dir_r[i]) for i in (0, 1)) < 1e-12
    check("C7 bootstrap：(a) 比值恒 0.5 ⇒ 宽度 0 且点估计 0.5；(b) 轻尾 n=90 时 95% 覆盖率∈[0.90,0.99]；"
          "(c) 点估计==直接 np.sum",
          ok_a and ok_b and ok_c,
          f"(a) width={w0:.3e} R={ba['R']}｜(b) coverage={rate:.3f}（120 次）｜(c) "
          f"max|Δ|={max(abs(bc['R'][i] - dir_r[i]) for i in (0, 1)):.3e}")
    print(f"       ↳ [report-only] 重尾(σ=0.6)+40 cluster 的实测覆盖率={rate_heavy:.3f}"
          f"（percentile bootstrap 已知欠覆盖 ⇒ 写进限制节，不判 PASS/FAIL）")
    out["C7"] = {"a_width": w0, "b_coverage_light90": rate,
                 "d_coverage_heavy40_report_only": rate_heavy, "c_ok": ok_c}

    # ---------------- C6：B=L 时配速恒等（自检里现跑，不依赖主表）----------------
    c6 = {}
    ok6 = True
    for pname in ("zoh", "linear"):
        for mode in ("unpaced", "paced"):
            c6[f"{pname}/{mode}"] = _J(_run(200, mode, seed0, n_ep_small,
                                            predictor=make_predictor(pname))["records"], n_ep_small)
    for pname in ("zoh", "linear"):
        eq = np.array_equal(c6[f"{pname}/unpaced"], c6[f"{pname}/paced"])
        ok6 = ok6 and eq
    check("C6 B=200（=L）时 paced 与 unpaced 逐 episode J **逐位相同**（配速包装器透明）",
          ok6 and bool(c6), "; ".join(
              f"{p}/{m}:max|Δ|={float(np.max(np.abs(c6[f'{p}/{m}'] - c6[f'{p}/unpaced']))):.1e}"
              for p in ("zoh", "linear") for m in ("paced",)))
    out["C6"] = {f"{p}/paced": float(np.max(np.abs(c6[f"{p}/paced"] - c6[f"{p}/unpaced"])))
                 for p in ("zoh", "linear")}
    return out


# ================================================================ 单 cell
def run_cell(budget, mode, pred_name, seed, n_ep, *, timed=False):
    t0 = time.time()
    pred = make_predictor(pred_name)
    tp = _Timed(pred) if timed else None
    probe = _ErrProbe()
    r = _run(budget, mode, seed, n_ep, predictor=(tp or pred), probe=probe)
    rec = r["records"]
    J = _J(rec, n_ep)
    dr = delivery_rate(rec, 2)
    served = [sum(1 for x in rec if x.attempted_idx == i) for i in range(2)]
    late = [sum(1 for x in rec if x.t > budget and x.attempted_idx == i) for i in range(2)]
    return {
        "budget": int(budget), "mode": mode, "predictor": pred_name, "seed": int(seed),
        "J_per_episode": np.asarray(J, dtype=float).tolist(),
        "J_mean": np.asarray(J, dtype=float).mean(axis=0).tolist(),
        "served": served, "served_late": late, "served_total": int(sum(served)),
        "alive_steps": dr["alive_steps"], "frames": dr["frames"], "attempts": dr["attempts"],
        "link_q": dr["link_q"], "p_eff": dr["p_eff"],
        "budget_reserved": dr["budget_reserved"], "budget_attempted": dr["budget_attempted"],
        "budget_wasted": dr["budget_wasted"],
        "pred_error": probe.summary(),
        "predict_timing": (tp.timing() if tp is not None else None),
        "elapsed_s": round(time.time() - t0, 2),
    }


# ================================================================ 主实验
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=N_EP)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip-experiment", action="store_true")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--allow-overwrite", action="store_true")
    args = ap.parse_args()
    global SMOKE_TAG
    SMOKE_TAG = "_smoke" if args.smoke else ""
    n_ep = 2 if args.smoke else args.episodes
    if n_ep != N_EP and not args.smoke:
        print(f"[WARN] episodes={n_ep} ≠ 预注册的 {N_EP}：结果不得标「确认」，只能标「探索」")
    seed_sets = ({"primary": (SEED_SETS["primary"][0],)} if args.smoke else SEED_SETS)
    predictors = ("zoh", "linear") if args.smoke else PREDICTORS
    budgets = (200,) if args.smoke else BUDGETS
    modes = ("unpaced",) if args.smoke else MODES

    t_all = time.time()
    print(f"[58] P3-0 确认性评估｜窗口 {WINDOW}｜臂={ARM}｜种子集 {seed_sets}｜episodes={n_ep}｜"
          f"predictors={predictors}｜budgets={budgets}｜modes={modes}")
    sc = selfchecks()

    if args.skip_experiment:
        _dump({"selfchecks": sc}, None, None, n_ep, seed_sets)
        print(f"\n{'ALL CHECKS PASS' if not FAILS else 'FAILED: ' + '; '.join(FAILS)}")
        return 0 if not FAILS else 2

    cells: list[dict] = []
    for set_name, seeds in seed_sets.items():
        for budget in budgets:
            for mode in modes:
                for pname in predictors:
                    for seed in seeds:
                        c = run_cell(budget, mode, pname, seed, n_ep,
                                     timed=(pname in ("linear", "zoh")))
                        c["seed_set"] = set_name
                        cells.append(c)
                        print(f"  [{set_name}] B={budget:3d} {mode:7s} {pname:8s} seed={seed} "
                              f"J=({c['J_mean'][0]:.3f},{c['J_mean'][1]:.3f}) "
                              f"served=({c['served'][0]},{c['served'][1]}) "
                              f"late=({c['served_late'][0]},{c['served_late'][1]}) "
                              f"{c['elapsed_s']}s")

    def get(set_name, budget, mode, pname, seed):
        for c in cells:
            if (c["seed_set"], c["budget"], c["mode"], c["predictor"], c["seed"]) == \
                    (set_name, budget, mode, pname, seed):
                return c
        return None

    def rows_for(set_name, budget, mode, num, den):
        """配对行：同一 (seed, episode) 上 num/den 两个预测器的 J（cluster = reset 初值）。"""
        seeds = seed_sets[set_name]
        rows = []
        for s in seeds:
            a = get(set_name, budget, mode, num, s)
            b = get(set_name, budget, mode, den, s)
            if a is None or b is None:
                continue
            for e in range(len(a["J_per_episode"])):
                rows.append({"cluster": int(s) + e,
                             "J_num": a["J_per_episode"][e],
                             "J_den": b["J_per_episode"][e]})
        return rows

    def table(set_name, budget, mode, num, den):
        rows = rows_for(set_name, budget, mode, num, den)
        if not rows:
            return None
        d = cluster_bootstrap(rows)
        # 逐种子 R_i（点估计，不给区间：单种子只有 30 集，区间没分辨力）
        per_seed = {}
        for s in seed_sets[set_name]:
            a = np.asarray(get(set_name, budget, mode, num, s)["J_mean"], dtype=float)
            b = np.asarray(get(set_name, budget, mode, den, s)["J_mean"], dtype=float)
            per_seed[str(s)] = [float(x) for x in (a / b)] if np.all(b > 0) else [float("nan")] * 2
        d["per_seed_R"] = per_seed
        rs = np.asarray([v for v in per_seed.values()], dtype=float)
        d["n_seeds_uav_below_1"] = int(np.sum(rs[:, 0] < 1.0))
        d["n_seeds"] = int(rs.shape[0])
        # 分布无关的稳健性：逐 cluster 的 episode 比值均值 <1 的比例（不依赖正态/尾假设）
        by_cl: dict[int, list] = {}
        for r in rows:
            a = np.asarray(r["J_num"], dtype=float)
            b = np.asarray(r["J_den"], dtype=float)
            by_cl.setdefault(int(r["cluster"]), []).append(
                [float("nan") if x <= 0 else y / x for x, y in zip(b, a)])
        cl_means = np.asarray([np.nanmean(np.asarray(v, dtype=float), axis=0)
                               for v in by_cl.values()], dtype=float)
        d["frac_clusters_below_1"] = [float(np.mean(cl_means[:, i] < 1.0)) for i in (0, 1)]
        d["cluster_ratio_median"] = [float(np.nanmedian(cl_means[:, i])) for i in (0, 1)]
        for side in (num, den):
            d[f"served_{side}"] = [float(np.mean([get(set_name, budget, mode, side, s)["served"][i]
                                                 for s in seed_sets[set_name]])) for i in (0, 1)]
            d[f"alive_{side}"] = [float(np.mean([get(set_name, budget, mode, side, s)["alive_steps"][i]
                                                for s in seed_sets[set_name]])) for i in (0, 1)]
            d[f"late_{side}"] = float(np.mean([sum(get(set_name, budget, mode, side, s)["served_late"])
                                              for s in seed_sets[set_name]]))
            d[f"J_{side}"] = [float(np.mean([get(set_name, budget, mode, side, s)["J_mean"][i]
                                            for s in seed_sets[set_name]])) for i in (0, 1)]
        return d

    summary: dict = {}
    for set_name in seed_sets:
        summary[set_name] = {}
        for budget in budgets:
            summary[set_name][str(budget)] = {}
            for mode in modes:
                per = {}
                for pname in predictors[1:]:
                    per[pname] = table(set_name, budget, mode, pname, "zoh")
                summary[set_name][str(budget)][mode] = per

    print("\n== 汇总：R_i = Σ J_i(pred) / Σ J_i(ZOH)，95% cluster bootstrap ==")
    for set_name in seed_sets:
        for budget in budgets:
            for mode in modes:
                for pname in predictors[1:]:
                    d = summary[set_name][str(budget)][mode][pname]
                    if d is None:
                        continue
                    print(f"  [{set_name}] B={budget:3d} {mode:7s} {pname:8s} "
                          f"R=({d['R'][0]:.3f},{d['R'][1]:.3f}) "
                          f"CI_uav=[{d['ci95'][0][0]:.3f},{d['ci95'][1][0]:.3f}] "
                          f"CI_cart=[{d['ci95'][0][1]:.3f},{d['ci95'][1][1]:.3f}] "
                          f"clusters={d['n_clusters']} seeds_uav<1={d['n_seeds_uav_below_1']}/"
                          f"{d['n_seeds']} late={d['late_' + pname]:.0f}")

    # ---------- F1：主判据 ----------
    print("\n== F1（主判据：主确认集 B=200 unpaced，`Linear` vs `ZOH`）==")
    f1 = {}
    main_cell = summary["primary"]["200"]["unpaced"]["linear"] if "primary" in summary else None
    if main_cell is None:
        f1["verdict"] = "N/A"
        print("  [SKIP] 未跑主确认集")
    else:
        hi_uav = main_cell["ci95"][1][0]
        r_cart = main_cell["R"][1]
        pass_uav = hi_uav < 1.0
        pass_cart = r_cart <= 1.10
        f1 = {"R": main_cell["R"], "ci_uav": main_cell["ci95"][0][0:1] + main_cell["ci95"][1][0:1],
              "ci_cart": main_cell["ci95"][0][1:2] + main_cell["ci95"][1][1:2],
              "pass_uav_upper_lt_1": bool(pass_uav), "pass_cart_le_1p10": bool(pass_cart),
              "per_seed_R": main_cell["per_seed_R"],
              "seeds_uav_below_1": f"{main_cell['n_seeds_uav_below_1']}/{main_cell['n_seeds']}",
              "served": {"zoh": main_cell["served_zoh"], "linear": main_cell["served_linear"]},
              "alive": {"zoh": main_cell["alive_zoh"], "linear": main_cell["alive_linear"]},
              "J": {"zoh": main_cell["J_zoh"], "linear": main_cell["J_linear"]}}
        f1["verdict"] = ("CONFIRMED" if (pass_uav and pass_cart)
                         else ("FAIL-UAV-NOT-SIG" if not pass_uav else "FAIL-CART-BOUND"))
        print(f"  R_uav={main_cell['R'][0]:.3f} CI95=[{main_cell['ci95'][0][0]:.3f},"
              f"{main_cell['ci95'][1][0]:.3f}] ⇒ 上界<1: {pass_uav}")
        print(f"  R_cart={r_cart:.3f} ⇒ ≤1.10: {pass_cart}（不进入判据，只报）")
        print(f"  served zoh={main_cell['served_zoh']} → linear={main_cell['served_linear']}；"
              f"alive zoh={main_cell['alive_zoh']} → linear={main_cell['alive_linear']}")
        print(f"  ⇒ {f1['verdict']}")

    # ---------- F2：稳健性（只报不判）----------
    print("\n== F2（稳健性，只报告不判）==")
    f2 = {}
    for set_name in seed_sets:
        for budget in budgets:
            for mode in modes:
                d = summary[set_name][str(budget)][mode]["linear"]
                if d is None:
                    continue
                f2[f"{set_name}/{budget}/{mode}"] = {
                    "R": d["R"], "ci_uav": [d["ci95"][0][0], d["ci95"][1][0]],
                    "seeds_uav_below_1": f"{d['n_seeds_uav_below_1']}/{d['n_seeds']}",
                    "frac_clusters_below_1": d["frac_clusters_below_1"],
                    "cluster_ratio_median": d["cluster_ratio_median"],
                    "per_seed_R": d["per_seed_R"]}
                print(f"  [{set_name}] B={budget:3d} {mode:7s} R=({d['R'][0]:.3f},{d['R'][1]:.3f}) "
                      f"CI_uav=[{d['ci95'][0][0]:.3f},{d['ci95'][1][0]:.3f}] "
                      f"uav<1: {d['n_seeds_uav_below_1']}/{d['n_seeds']}｜"
                      f"cluster 内比值<1 的比例=({d['frac_clusters_below_1'][0]:.2f},"
                      f"{d['frac_clusters_below_1'][1]:.2f})")

    # ---------- 次确认集 vs 主确认集的区间宽度（初值重复的代价）----------
    print("\n== C5b（次确认集 32 个初值 vs 主确认集 90 个：区间宽度对比，只作方法学观察）==")
    c5b = {}
    if "primary" in summary and "secondary" in summary:
        for budget in budgets:
            for mode in modes:
                a = summary["primary"][str(budget)][mode]["linear"]
                b = summary["secondary"][str(budget)][mode]["linear"]
                if a is None or b is None:
                    continue
                wa = a["ci95"][1][0] - a["ci95"][0][0]
                wb = b["ci95"][1][0] - b["ci95"][0][0]
                c5b[f"{budget}/{mode}"] = {"width_primary90": wa, "width_secondary32": wb,
                                           "ratio_sec_over_pri": (wb / wa if wa > 0 else None)}
                print(f"  B={budget:3d} {mode:7s} 主(90 初值) 宽={wa:.3f}｜次(32 初值) 宽={wb:.3f}｜"
                      f"次/主={wb / wa if wa > 0 else float('nan'):.2f}")

    # ---------- C6b：主表上复验"B=L 时 paced≡unpaced"（C6 已在自检里跑过；这里只是全 cell 复验）----------
    print("\n== C6b（主表全 cell 复验：B=200 的 paced 与 unpaced 逐 episode J 逐位相同）==")
    c6b = {}
    for set_name in seed_sets:
        if "200" not in summary[set_name]:
            continue
        for pname in predictors:
            for s in seed_sets[set_name]:
                u = get(set_name, 200, "unpaced", pname, s)
                p = get(set_name, 200, "paced", pname, s)
                if u is None or p is None:
                    continue
                c6b[f"{set_name}/{pname}/{s}"] = bool(
                    np.array_equal(np.asarray(u["J_per_episode"]), np.asarray(p["J_per_episode"])))
    if not c6b:
        print("  [SKIP] 无 B=200 的 paced/unpaced 配对（smoke 模式）")
    else:
        check("C6b 主表全部 B=200 配对逐位相同", all(c6b.values()),
              f"{sum(c6b.values())}/{len(c6b)} cell 相同")

    # ---------- C4（主表级）----------
    print("\n== C4（记账，全 cell）==")
    bad1 = [c for c in cells if c["served_total"] != n_ep * c["budget"]]
    bad2 = [c for c in cells if c["budget_reserved"]
            != c["budget_attempted"] + c["budget_wasted"]]
    bad3 = [c for c in cells if c["mode"] == "unpaced" and c["budget"] < WINDOW
            and sum(c["served_late"]) != 0]
    bad4 = [c for c in cells if c["mode"] == "paced" and c["budget"] < WINDOW
            and sum(c["served_late"]) <= 0]
    check("C4a 每 cell 总发帧 == episodes × budget", not bad1, f"{len(cells)} cells")
    check("C4b reserved == attempted + wasted", not bad2, "全 cell")
    check("C4c unpaced B<L 的 t>B 发帧恒 0（缺陷仍在，未被悄悄修掉）", not bad3, "全部 0")
    check("C4d paced B<L 的 t>B 发帧 >0（R17：配速确实生效）", not bad4, "全部 >0")

    # ---------- C8：ZOH 在开发集上与 T23-b 逐位回归 ----------
    print("\n== C8（ZOH 在开发集 {0,1,2} B=200 unpaced 与 results/56 的 zoh 列逐位回归）==")
    path = latest_result_path("56_t23b_compensation_benefit")
    print(f"  [ref-56] {path.relative_to(ROOT).as_posix() if path else '<缺失>'}")
    c8 = {"ref": (path.relative_to(ROOT).as_posix() if path else None), "compared": 0, "diffs": []}
    if path is None or not path.exists():
        c8["diffs"] = ["<无参考>"]
    elif n_ep != N_EP:
        c8["skipped"] = f"episodes={n_ep} != {N_EP}"
        print(f"  [SKIP] 集数 {n_ep} ≠ 历史 {N_EP} ⇒ 不可比")
    else:
        ref = json.loads(path.read_text(encoding="utf-8"))
        ref_map = {(c["budget"], c["arm"] if "arm" in c else ARM, c["mode"], c["seed"]):
                   c["J_per_episode"] for c in ref.get("cells_J", [])
                   if c.get("predictor") == "zoh"}
        for s in DEV_SEEDS:
            c = run_cell(200, "unpaced", "zoh", s, n_ep)
            key = (200, ARM, "unpaced", s)
            if key not in ref_map:
                c8["diffs"].append(f"{key} 不在参考里")
                continue
            a = np.asarray(c["J_per_episode"], dtype=float)
            b = np.asarray(ref_map[key], dtype=float)
            if a.shape != b.shape:
                c8["diffs"].append(f"{key} 形状 {a.shape} vs {b.shape}")
                continue
            c8["compared"] += 1
            if not np.array_equal(a, b):
                c8["diffs"].append(f"{key} max|Δ|={np.nanmax(np.abs(a - b)):.3e}")
    if "skipped" in c8:
        print(f"  [SKIP] {c8['skipped']}")
    else:
        check("C8 ZOH 开发集 3 cell 与 results/56 逐 episode J **逐位相同**",
              c8["compared"] > 0 and not c8["diffs"],
              (f"{c8['compared']} cells 逐位相同" if not c8["diffs"] else "；".join(c8["diffs"][:4])))

    # ---------- TIMING ----------
    print("\n== TIMING（`predict()` 每次调用的墙钟耗时，μs；实现开销，不是实时预算）==")
    timing = {}
    for set_name in seed_sets:
        for pname in ("linear", "zoh"):
            rows = [c for c in cells if c["seed_set"] == set_name and c["predictor"] == pname
                    and c["budget"] == 200 and c["mode"] == "unpaced"]
            if not rows:
                continue
            ms = [r["predict_timing"] for r in rows if r["predict_timing"]]
            timing[f"{set_name}/{pname}"] = {
                "median_us": float(np.median([m["median_us"] for m in ms])),
                "n_calls": int(sum(m["n_calls"] for m in ms))}
            print(f"  [{set_name}] {pname:8s} median={timing[f'{set_name}/{pname}']['median_us']:.2f} μs/call "
                  f"（{timing[f'{set_name}/{pname}']['n_calls']} 次调用）")

    # ---------- ERR ----------
    print("\n== ERR（预测误差中位，B=200 unpaced，age 三档；预测质量≠收益）==")
    err = {}
    for pname in predictors:
        rows = [c for c in cells if c["predictor"] == pname and c["budget"] == 200
                and c["mode"] == "unpaced"]
        if not rows:
            continue
        err[pname] = {u: {t: rows[0]["pred_error"][u]["buckets"][t] for t, _lo, _hi in AGE_BUCKETS}
                      for u in ("0", "1")}
        print(f"  {pname:8s} UAV(age2-3)="
              f"{rows[0]['pred_error']['0']['buckets']['2-3'].get('median', float('nan')):.4f} "
              f"Cart(age2-3)="
              f"{rows[0]['pred_error']['1']['buckets']['2-3'].get('median', float('nan')):.4f}")

    _dump({"selfchecks": sc}, cells,
          {"summary": summary, "F1": f1, "F2": f2, "C5b_width": c5b, "C6b": c6b, "C8": c8,
           "TIMING": timing, "ERR": err},
          n_ep, seed_sets, run_id=args.run_id, allow_overwrite=args.allow_overwrite,
          persist=not args.smoke)
    print(f"\n{'ALL CHECKS PASS' if not FAILS else 'FAILED: ' + '; '.join(FAILS)}")
    print(f"总耗时 {round(time.time() - t_all, 1)}s")
    return 0 if not FAILS else 2


def _dump(payload, cells, extra, n_ep, seed_sets, *, run_id=None, allow_overwrite=False,
          persist=True):
    seeds_flat = {k: list(v) for k, v in seed_sets.items()}
    cfg = {
        "N": 2, "window": WINDOW, "warmup": [WARMUP_UAV, 0], "budgets": list(BUDGETS),
        "episodes": n_ep, "seed_sets": seeds_flat, "gate": GATE, "arm": ARM,
        "absorbed": [float(x) for x in ABSORBED], "predictors": list(PREDICTORS),
        "modes": list(MODES), "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
        "primary_cell": "B=200 unpaced（t>B 发帧恒 0）",
        "statistic": "R_i = Σ_e J_i(num)/Σ_e J_i(ZOH)，配对；cluster bootstrap 单位为 reset 初值",
        "criterion": "R_uav 95% 区间上界 <1.0 且 R_cart 点估计 ≤1.10（预注册 §4）",
        "prereg": "04-实验与代码/2026-10-20-P3-0-确认评估-预注册-v0.1.md",
    }
    out = {"config": cfg, "selfchecks": payload.get("selfchecks"),
           "checks_failed": list(FAILS)}
    (ROOT / "outputs").mkdir(exist_ok=True)
    (ROOT / "outputs" / ("58_p3_confirmation" + SMOKE_TAG + ".json")).write_text(
        json.dumps({**out, **(extra or {}), "cells": (cells or [])},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    if cells is None or not persist:
        if cells is not None:
            print("[smoke] 不写 results/（smoke 数字不进证据链）")
        return
    keep = ("seed_set", "budget", "mode", "predictor", "seed", "J_per_episode", "J_mean",
            "served", "served_late", "served_total", "alive_steps", "frames", "attempts",
            "link_q", "budget_reserved", "budget_attempted", "budget_wasted", "pred_error",
            "predict_timing", "elapsed_s")
    slim = [{k: c[k] for k in keep} for c in (cells or [])]
    dest = write_result("58_p3_confirmation", {**out, **(extra or {}), "cells_J": slim},
                        run_id=run_id, allow_overwrite=allow_overwrite)
    print(f"[evidence] {dest.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    sys.exit(main())
