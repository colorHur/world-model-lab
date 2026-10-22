# -*- coding: utf-8 -*-
"""评价器自己的反例闸（`CW-EVAL-v1.1` 验收面）。

每一条都要求**两个方向**都成立：该判改善的数据必须判改善，该判退化的数据必须判退化。
只写"某数据判改善了"的检查没有分辨力 —— 一个恒返回 True 的函数也能过。

★ 本套闸的对象是**评价器**，不是策略。它测的是：
  方向、格数、缺格、NaN/零基线、统计量命名、手搓记录的自洽性。
"""

from __future__ import annotations

import numpy as np

from .evaluate import (EVAL_VERSION, VERDICT_BEATS, VERDICT_NOT_BETTER,
                       VERDICT_UNDEFINED, CellStats, UnreachableRequirement,
                       compare_cell, decide, paired_delta, per_episode_ratio,
                       required_cells_from_design)
from .gates import Gate

DESIGN = [(0.85, 0.5), (0.85, 0.8), (0.98, 0.5), (0.98, 0.8)]


# ---------------------------------------------------------------- 旧判据的复刻
#: ★ 这不是"参考实现"，是**故意复刻 bug**，用来证明修正改变了判词（load-bearing）。
def legacy_ratio_inverted(baseline, candidate) -> float:
    """旧代码：`median(baseline) / median(candidate)`，却用 `< 0.9` 判候选更好。"""
    return float(np.median(baseline) / np.median(candidate))


def legacy_ratio_only_pass(baseline, candidate) -> bool:
    return legacy_ratio_inverted(baseline, candidate) < 0.9


def legacy_cell_count_always_fails(n_cells: int, required: int = 6) -> bool:
    """旧代码：`n_win >= 6`，而实验只有 4 格 ⇒ 恒 False。"""
    return n_cells >= required


def _cells_all(baseline_per_cell, candidate_per_cell):
    out = []
    for key, b, c in zip(DESIGN, baseline_per_cell, candidate_per_cell):
        out.append(compare_cell(b, c, cell=key, n_boot=200))
    return out


# ---------------------------------------------------------------- 闸
def run_all(*, n_boot: int = 400) -> Gate:
    g = Gate()
    req = required_cells_from_design(DESIGN)

    # E1 方向：候选更便宜 ⇒ 改善；候选更贵 ⇒ 退化（两向都要对）
    good = compare_cell([2.0] * 6, [1.0] * 6, cell="good", n_boot=n_boot)
    bad = compare_cell([1.0] * 6, [2.0] * 6, cell="bad", n_boot=n_boot)
    g.check("E1 方向：cand=1/base=2 判改善，cand=2/base=1 判退化",
            good.median_delta > 0 and good.median_pairwise_ratio == 0.5
            and bad.median_delta < 0 and bad.median_pairwise_ratio == 2.0,
            {"good": {"delta": good.median_delta, "ratio": good.median_pairwise_ratio},
             "bad": {"delta": bad.median_delta, "ratio": bad.median_pairwise_ratio}})

    # E2 旧判据在同一批数据上给出**相反**结论 ⇒ 修正是 load-bearing
    f_pass = decide([good], required_cells=1)["verdict"] == VERDICT_BEATS
    f_fail = decide([bad], required_cells=1)["verdict"] == VERDICT_NOT_BETTER
    l_pass = legacy_ratio_only_pass([2.0] * 6, [1.0] * 6)      # 候选便宜一半
    l_fail = legacy_ratio_only_pass([1.0] * 6, [2.0] * 6)      # 候选贵一倍
    g.check("E2 旧比值方向在同一数据上判断相反（候选便宜 ⇒ 旧判据说不改善；"
            "候选贵一倍 ⇒ 旧判据说改善）",
            f_pass and f_fail and (not l_pass) and l_fail,
            {"fixed": [f_pass, f_fail], "legacy_ratio_only": [l_pass, l_fail],
             "legacy_ratio_values": [legacy_ratio_inverted([2.0] * 6, [1.0] * 6),
                                     legacy_ratio_inverted([1.0] * 6, [2.0] * 6)]})

    # E3 四格**全成功** ⇒ 必须能判成功（旧 6 格判据在同一数据上必失败）
    win_cells = _cells_all([[2.0] * 6] * 4, [[1.0] * 6] * 4)
    d_win = decide(win_cells, required_cells=req)
    legacy_all_fail = not legacy_cell_count_always_fails(n_cells=len(DESIGN), required=6)
    g.check("E3 四格全成功 ⇒ CANDIDATE-BEATS-BASELINE；同一数据在旧 6 格判据下必失败",
            d_win["verdict"] == VERDICT_BEATS and d_win["n_pass"] == 4
            and legacy_all_fail,
            {"verdict": d_win["verdict"], "n_pass": d_win["n_pass"],
             "required": d_win["required_cells"], "legacy_6_of_4_unsatisfiable": legacy_all_fail})

    # E4 缺一格 ⇒ 不得判成功，且必须点名是哪一格
    partial = list(win_cells)
    partial[2] = compare_cell([1.0] * 6, [2.0] * 6, cell=DESIGN[2], n_boot=n_boot)
    d_part = decide(partial, required_cells=req)
    g.check("E4 缺一格（3/4 过）⇒ NOT-BETTER 且 failing_cells 恰好是那一格",
            d_part["verdict"] == VERDICT_NOT_BETTER and d_part["n_pass"] == 3
            and d_part["failing_cells"] == [list(DESIGN[2])],
            {"verdict": d_part["verdict"], "n_pass": d_part["n_pass"],
             "failing_cells": d_part["failing_cells"]})

    # E5 不可满足的判据必须**抛异常**，不许静默 FAIL
    raised = False
    try:
        decide(win_cells, required_cells=6)
    except UnreachableRequirement:
        raised = True
    ok_normal = decide(win_cells, required_cells=4)["verdict"] == VERDICT_BEATS
    g.check("E5 要求 6 格但只有 4 格 ⇒ 抛 UnreachableRequirement（且 4 格时正常给判词）",
            raised and ok_normal, {"raised": raised, "4_cells_ok": ok_normal})

    # E6 NaN ⇒ UNDEFINED；把 NaN 换成有限值 ⇒ 不再是 UNDEFINED（对照证明不是恒 UNDEFINED）
    nan_cell = compare_cell([1.0, 1.0, float("nan"), 1.0], [0.5] * 4, cell="nan", n_boot=50)
    d_nan = decide([nan_cell], required_cells=1)
    finite_same = decide([compare_cell([1.0, 1.0, 1.0, 1.0], [0.5] * 4,
                                       cell="fixed", n_boot=50)], required_cells=1)
    g.check("E6 NaN 输入 ⇒ UNDEFINED（不是'败'也不是'胜'）；同一数据去掉 NaN ⇒ 正常判词",
            nan_cell.status == "undefined_nan" and d_nan["verdict"] == VERDICT_UNDEFINED
            and finite_same["verdict"] == VERDICT_BEATS,
            {"nan_status": nan_cell.status, "nan_verdict": d_nan["verdict"],
             "finite_verdict": finite_same["verdict"]})

    # E7 零基线 ⇒ 比值无定义 ⇒ UNDEFINED；但**差值**仍要保留正确符号
    zero_cell = compare_cell([0.0, 0.0, 0.0], [0.5, 0.5, 0.5], cell="zero", n_boot=20)
    g.check("E7 零基线 ⇒ UNDEFINED，但 median_delta 仍给出正确符号（信息不被吞掉）",
            zero_cell.status == "undefined_nonpositive_baseline"
            and zero_cell.median_delta == -0.5,
            {"status": zero_cell.status, "median_delta": zero_cell.median_delta})

    # E8 「中位数之比」与「逐 episode 比值中位数」是两个不同的量 ⇒ 必须分开命名与输出
    #    ★ 这组数据是**故意构造**的：配对口径说"候选便宜 33%"，聚合口径说"候选贵 6.7 倍"。
    b8 = [1.0, 2.0, 3.0, 100.0, 1000.0]
    c8 = [10.0, 20.0, 2.0, 30.0, 40.0]
    st8 = compare_cell(b8, c8, cell="twostats", n_boot=50)
    distinct = abs(st8.median_pairwise_ratio - st8.ratio_of_medians) > 0.2
    opposite = (st8.median_pairwise_ratio < 1.0) != (st8.ratio_of_medians < 1.0)
    g.check("E8 两个中位统计量可区分且可给出**相反方向**（不得互换、不得只报一个）",
            distinct and opposite and np.isfinite(st8.median_pairwise_ratio)
            and np.isfinite(st8.ratio_of_medians),
            {"median_pairwise_ratio": st8.median_pairwise_ratio,
             "ratio_of_medians": st8.ratio_of_medians})

    # E9 手搓的 CellStats 若比值与胜率不自洽 ⇒ 该格作废（第二道闸）
    fake = CellStats(cell="fake", n_pairs=4, baseline_median=1.0, candidate_median=1.0,
                     median_delta=0.1, mean_delta=0.1, delta_ci=(0.0, 0.2),
                     win_fraction=0.0, n_wins=0, n_ties=0,
                     median_pairwise_ratio=0.5,      # ← 比值说"便宜一半"
                     ratio_of_medians=1.0, status="ok", reasons=[])
    d_fake = decide([fake], required_cells=1)
    ok_fake_rejected = d_fake["verdict"] == VERDICT_UNDEFINED \
        and d_fake["cells"][0]["status"] == "inconsistent"
    # 对照：同一格把胜率改对 ⇒ 必须能过（证明这条闸不是恒拒）
    fake_ok = CellStats(**{**fake.__dict__, "win_fraction": 1.0, "n_wins": 4})
    ok_fake_pass = decide([fake_ok], required_cells=1)["verdict"] == VERDICT_BEATS
    g.check("E9 比值与胜率不自洽的格作废；改正胜率后可过（两向）",
            ok_fake_rejected and ok_fake_pass,
            {"fake_verdict": d_fake["verdict"],
             "fake_cell_status": d_fake["cells"][0]["status"],
             "fake_fixed_verdict": decide([fake_ok], required_cells=1)["verdict"]})

    # E10 胜率必须等于 delta>0 的 episode 比例（定义级对账，不是抽样）
    rng = np.random.default_rng(7)
    b10 = rng.uniform(0.5, 2.0, size=37)
    c10 = rng.uniform(0.5, 2.0, size=37)
    st10 = compare_cell(b10, c10, cell="winrate", n_boot=50)
    manual = float(np.mean(paired_delta(b10, c10) > 0))
    g.check("E10 胜率 == 逐 episode delta>0 的比例；逐 episode 比值 == cand/base",
            abs(st10.win_fraction - manual) < 1e-12
            and np.allclose(per_episode_ratio(b10, c10), c10 / b10),
            {"win_fraction": st10.win_fraction, "manual": manual})

    # E11 bootstrap 区间必须夹住点估计（否则区间口径错）
    st11 = compare_cell(b10, c10, cell="ci", n_boot=1000)
    lo, hi = st11.delta_ci
    g.check("E11 配对差中位数的 bootstrap 区间夹住点估计",
            lo <= st11.median_delta <= hi,
            {"median_delta": st11.median_delta, "ci": [lo, hi]})

    # E12 判词词表不含"空间为空"类结论（P0-C 的纪律）
    forbidden = {"SPACE-VOID"}
    from . import evaluate as _E
    names = {v for k, v in vars(_E).items() if k.startswith("VERDICT_")}
    g.check("E12 判词词表不含 SPACE-VOID 之类'整个方向无空间'的措辞",
            not (names & forbidden), {"verdicts": sorted(names), "eval_version": EVAL_VERSION})

    # E13 ★ 反向误杀检查（本条是"判据不能比被测物还严"的落地）：
    #   n 为偶数、恰好一半 episode 胜、两个中间比值跨 1 ⇒ 中位数可以略小于 1。
    #   旧写法 `median<1 ⟺ 胜率>50%` 会把这种**合法**格子判成 inconsistent。
    b13 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    c13 = [0.999, 1.0, 1.0, 1.0, 1.0, 1.001]   # delta: +0.001, 0,0,0,0, -0.001
    st13 = compare_cell(b13, c13, cell="even_half", n_boot=50)
    g.check("E13 even-n 恰好一半胜时不得被误判为'统计量混用'",
            st13.status == "ok",
            {"status": st13.status, "median_pairwise_ratio": st13.median_pairwise_ratio,
             "win_fraction": st13.win_fraction, "reasons": st13.reasons})

    # E14 真矛盾仍要被抓（与 E13 配对：放宽之后没把闸拆掉）
    b14 = [1.0] * 6
    c14 = [0.5] * 6                      # 全部候选更便宜
    fake14 = CellStats(cell="x", n_pairs=6, baseline_median=1.0, candidate_median=0.5,
                       median_delta=0.5, mean_delta=0.5, delta_ci=(0.5, 0.5),
                       win_fraction=0.8333, n_wins=5, n_ties=0,
                       median_pairwise_ratio=1.9,   # ← 比值说"贵 90%"，却赢了 5/6
                       ratio_of_medians=0.5, status="ok", reasons=[])
    d14 = decide([fake14], required_cells=1)
    g.check("E14 比值说'贵 90%'但胜率 5/6 ⇒ 仍判作废（放宽后没把闸拆掉）",
            d14["cells"][0]["status"] == "inconsistent",
            {"status": d14["cells"][0]["status"], "verdict": d14["verdict"]})

    return g


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(description="CW-EVAL-v1.1 评价器反例闸")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    g = run_all()
    s = g.summary()
    if args.json:
        print(json.dumps(s, ensure_ascii=False, indent=2))
    else:
        for r in s["rows"]:
            print(f"  {'PASS' if r['pass'] else 'FAIL'}  {r['name']}")
            if not r["pass"]:
                print(f"        {json.dumps(r['detail'], ensure_ascii=False)[:600]}")
        print(f"[EVAL-GATES] {s['n_checks']} 项，{s['n_fail']} FAIL -> "
              f"{'ALL PASS' if s['all_pass'] else 'FAIL'}")
    return 0 if s["all_pass"] else 2


if __name__ == "__main__":
    import sys
    sys.exit(main())
