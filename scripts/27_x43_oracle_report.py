# -*- coding: utf-8 -*-
"""★ X43 汇报：把**按 PER 分块**跑出来的 oracle 产物合并，算跨 PER 的比值中位。

================================================================= 这个实验在问什么
X38 → X39 → X40 → X41 → X42 一路在换**触发统计量**，边际任务价值始终是 0。
剩下的唯一问题：**是不是「统计量还不够好」？**

⇒ 本号用**真值误差**（仿真器内部的 ground truth）当触发量，量出「**误差类触发量**」
  这一族的**可达上界**。若连它都赢不了年龄阈值，则瓶颈**不在触发量的信息量**；
  若它赢了，则瓶颈在「模型自报的不确定度与真值误差的相关性」——
  两种结果都是**结论**，但必须分清楚。

★★ 跑前纠正过一处前提（写进脚本，防止下次又照字面做）
  看板原定的「用真值**逐年龄误差分布**当触发量」，字面实现 `Ê^or(h)` **只依赖 h**
  ⇒ `Ê^or(h) ≥ tol ⟺ h ≥ h*` ⇒ **恒等于年龄阈值**（比值 ≡ 1.000，**同义反复**）。
  ⇒ 本号改用**已实现误差**的逐年龄 z 分数（非退化版本），规则结构与 X40/X42 逐字同构。

================================================================= 为什么也要分块 + 覆盖守卫
同 X42：完整网格单次跑不完（本机后台任务 **10 min 硬上限**）⇒ 按 PER 分块。
分块**数值等价**（`scripts/24` 的等预算配对是逐 PER 独立插值再比，档与档不共享量；
模型训练只由 `seed` 决定，各块相同）。

★★ 但分块引入一个**新的静默失败模式**：某块没跑成 ⇒ 少一个 PER 档 ⇒
  跨 PER 取中位时分母悄悄变小，结论随之变强/变弱，而**看着完全正常**。
  ⇒ 第一件事是**覆盖守卫**（缺一档 raise），复用 `check_per_coverage`（自检 ㊹ 锁它）。

用法：
    python scripts/27_x43_oracle_report.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from wmlab.eval.conformal import check_per_coverage  # noqa: E402
from wmlab.eval.pairing import aggregate_p4  # noqa: E402
from wmlab.utils import output_dir  # noqa: E402

# ★ 引用基线（全部**同档 amp=16**，只有 X40 那条是 amp=4 旁证）
BASELINES = {
    "x41_b_amp16_self": {"tag": "X41-B 自报 z 触发（amp=16, hybrid）",
                         "dist_median": 1.0122, "nmse_median": 1.3133, "n_per": 5},
    "x42_nmse_amp16": {"tag": "X42 共形校准（amp=16, conf-target=nmse）",
                       "dist_median": 0.9962, "nmse_median": None, "n_per": 5},
    "x42_pos_amp16": {"tag": "X42 共形校准（amp=16, conf-target=pos）",
                      "dist_median": 0.9969, "nmse_median": None, "n_per": 5},
    "x40_amp4_self": {"tag": "X40 自报 z 触发（amp=4, hybrid）—— **不同档**，旁证",
                      "dist_median": 1.0648, "nmse_median": 2.0915, "n_per": 4},
}
#: ★ O_c 自洽门槛：oracle 的信息**严格多于**任何自报量 ⇒ 不该明显更差
SELF_CONSISTENCY_MAX = 1.02


def load_chunks(out: str, arm: str) -> tuple[list[dict], list[dict]]:
    """按臂名读取所有分块产物。返回 (P4 行, P2 行)。"""
    files = sorted(glob.glob(os.path.join(out, f"27_x43_{arm}_c*.json")))
    if not files:
        raise FileNotFoundError(f"★ 没找到 {arm} 臂的任何分块产物（27_x43_{arm}_c*.json）")
    p4, p2 = [], []
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        for r in d.get("P4_matched_rate", []):
            r = dict(r)
            r["_src"] = os.path.basename(f)
            p4.append(r)
        for r in d.get("P2_closed_loop", []):
            r = dict(r)
            r["_src"] = os.path.basename(f)
            p2.append(r)
    return p4, p2


def load_arm_diag(out: str, arm: str) -> dict:
    """读第一个块里的 oracle 标定/对齐/生效诊断（逐块相同，取第一块）。"""
    files = sorted(glob.glob(os.path.join(out, f"27_x43_{arm}_c*.json")))
    d = json.load(open(files[0], encoding="utf-8"))
    return {"oracle": d.get("oracle"), "alignment": d.get("oracle_alignment"),
            "active": d.get("oracle_active"), "caveat": d.get("caveat"),
            "source": os.path.basename(files[0])}


def _mad(vals: list[float]) -> float:
    a = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    if a.size < 2:
        return float("nan")
    return float(np.median(np.abs(a - np.median(a))))


def load_baseline_per_per(out: str) -> dict:
    """★ 逐 PER 的基线值（H2/O_c 要做**配对**符号检验，不能只比中位）。"""
    res: dict = {}
    p = os.path.join(out, "24_x41_hybrid16_wind16.json")
    if os.path.exists(p):
        d = json.load(open(p, encoding="utf-8"))
        res["x41_b_amp16_self"] = {
            float(r["per"]): float(r["utrigger_over_threshold_median"])
            for r in d.get("P4_matched_rate", []) if r["metric"] == "mean_dist_tail"}
    p2 = os.path.join(out, "26_x42_report.json")
    if os.path.exists(p2):
        d = json.load(open(p2, encoding="utf-8"))
        for arm, key in (("nmse", "x42_nmse_amp16"), ("pos", "x42_pos_amp16")):
            rows = d.get("arms", {}).get(arm, {}).get("rows", [])
            res[key] = {float(r["per"]): float(r["ratio"])
                        for r in rows if r["metric"] == "mean_dist_tail"}
    return res


def sign_test(deltas: list[float]) -> dict:
    """单边符号检验：H0 = 差值的符号是掷硬币（p=0.5）。

    ★ 为什么必须做（自证伪 ㉗ 的直接产物）：预写的「中位差 ≥0.01 **且** >2×档间散布」
      在 n=5 下**太松** —— 会把倾向性读成结论。必须补逐 PER 配对检验。
    """
    from math import comb
    d = [x for x in deltas if np.isfinite(x) and abs(x) > 0]
    n, k = len(d), int(sum(1 for x in d if x > 0))
    if n == 0:
        return {"n": 0, "k_pos": 0, "p_one_sided": float("nan"),
                "resolvable_at_0.10": False}
    p = sum(comb(n, i) for i in range(k, n + 1)) / (2 ** n)
    return {"n": n, "k_pos": k, "p_one_sided": float(p),
            "resolvable_at_0.10": bool(n >= 5 and p <= 0.10)}


def _figure(report: dict, out: str, tag: str) -> None:
    """★ 进版本库的图：**ASCII 标签**（本机字体链无 CJK，中文会变方框）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6))
    fig.suptitle("X43 oracle (true-error) trigger vs age threshold "
                 f"({report['n_per']} PER, wind_amp=16)", fontsize=11)
    for ax, met, key in ((axes[0], "mean_dist_tail", "dist_median"),
                         (axes[1], "est_nmse", "nmse_median")):
        for arm, dd in report["arms"].items():
            rs = [r for r in dd["rows"] if r["metric"] == met]
            mv = dd.get(key)
            ax.plot([r["per"] for r in rs], [r["ratio"] for r in rs], "o-",
                    label=f"oracle-{arm} (median "
                          + (f"{mv:.4f})" if mv is not None else "n/a)"))
        for bkey, bv, sty in (("x41_b_amp16_self", 1.0122, "s--"),
                              ("x42_pos_amp16", 0.9969, "^:")):
            per = sorted(report["baseline_per_per"].get(bkey, {}))
            if per and met == "mean_dist_tail":
                ax.plot(per, [report["baseline_per_per"][bkey][p] for p in per], sty,
                        color="grey", alpha=0.75, label=f"{bkey} ({bv})")
        ax.axhline(1.0, color="k", ls=":", lw=1)
        ax.axhline(0.98, color="r", ls="--", lw=1, alpha=0.7)
        ax.axhspan(0.98, 1.02, color="orange", alpha=0.10)
        ax.set_xlabel("PER")
        ax.set_ylabel(f"oracle / age-threshold  ({met})")
        ax.set_title("task distance (m)" if met == "mean_dist_tail" else "est NMSE")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p = os.path.join(out, tag + ".png")
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[27] 图：{p}")


def main() -> int:
    ap = argparse.ArgumentParser(description="X43 oracle 触发分块产物合并汇报")
    ap.add_argument("--out", default=None)
    ap.add_argument("--arms", default="res,pos")
    ap.add_argument("--expect-pers", default="0.0,0.1,0.3,0.5,0.7",
                    help="★ 必须**逐档**都在产物里出现（缺档 raise）")
    ap.add_argument("--known-unevaluable", default="0.9")
    ap.add_argument("--reason", default=(
        "PER=0.9 不可评估，根因 = `periodic` 族可比点塌缩到 1 点（强衰落下 T≥2 的周期策略"
        "NMSE>1 / E[age]>max_age_K ⇒ 被统一 diverged 口径剔除）⇒ **强衰落区做不了"
        "等预算比较**，这是**场景限制不是方法限制**（承 X40/X41/X42）"))
    ap.add_argument("--tag", default="27_x43_report")
    args = ap.parse_args()

    out = args.out or output_dir({"output": {"dir": "outputs"}})
    expect = [float(x) for x in args.expect_pers.split(",")]
    base_pp = load_baseline_per_per(out)
    report = {"expect_pers": expect, "arms": {}, "n_per": len(expect),
              "known_unevaluable": [float(x) for x in args.known_unevaluable.split(",") if x],
              "known_unevaluable_reason": args.reason,
              "baseline_per_per": base_pp,
              "self_consistency_max": SELF_CONSISTENCY_MAX}
    print(f"[27]   已知不可评估档（**显式声明**，不是少跑）：{args.known_unevaluable}")
    print("[27] " + "=" * 78)
    print("[27] ★ X43 oracle（真值误差）触发：分块合并（逐 PER 独立配对 ⇒ 分块数值等价）")
    for arm in [a for a in args.arms.split(",") if a]:
        p4, p2 = load_chunks(out, arm)
        cov = check_per_coverage(p4, expect)
        dist, nmse, n = aggregate_p4(p4)
        rows = []
        for r in sorted(p4, key=lambda r: (float(r["per"]), r["metric"])):
            if r["metric"] not in ("mean_dist_tail", "est_nmse"):
                continue
            rows.append({"per": float(r["per"]), "metric": r["metric"],
                         "ratio": float(r["utrigger_over_threshold_median"]),
                         "n_better": int(r["utrigger_better_n"]),
                         "n_grid": int(r["n_grid"]),
                         "rate_lo": float(r["rate_lo"]), "rate_hi": float(r["rate_hi"]),
                         "span": float(r["rate_hi"]) / max(float(r["rate_lo"]), 1e-12)})
        n_div = int(sum(1 for r in p2 if r.get("diverged")))
        diag = load_arm_diag(out, arm)
        dist_rows = [r["ratio"] for r in rows if r["metric"] == "mean_dist_tail"]
        report["arms"][arm] = {"coverage": cov, "dist_median": dist, "nmse_median": nmse,
                               "n_per": n, "dist_mad": _mad(dist_rows), "rows": rows,
                               "n_diverged_points": n_div, "diag": diag}
        print(f"[27] ---- arm = oracle-{arm}"
              f"（target={diag['oracle']['target'] if diag['oracle'] else '?'}）")
        for r in rows:
            print(f"[27]   PER={r['per']:<4g} {r['metric']:<15s} 比值中位 {r['ratio']:.4f}"
                  f"（{r['n_better']}/{r['n_grid']} 更优）"
                  f"  窗口 {r['rate_lo']:.4f}–{r['rate_hi']:.4f}（跨度 {r['span']:.2f}×）")
        print(f"[27]   ⇒ **跨 PER 中位**：任务距离比 **{dist if dist is None else round(dist, 4)}**、"
              f"NMSE 比 {nmse if nmse is None else round(nmse, 4)}（n={n} 档）｜发散点 {n_div}")
        act = diag.get("active") or {}
        print(f"[27]   ★ 披露（S_o4）：该臂工作点里「oracle 真的比安全网 K 更早触发」的比例 = "
              f"**{act.get('frac_active', float('nan')):.2f}**"
              f"（{act.get('n_work_points', 0) - act.get('n_k_dominated', 0)}"
              f"/{act.get('n_work_points', 0)}；其余点等价于纯年龄阈值，对结论不构成证据）")
        al = diag.get("alignment") or {}
        if al:
            print(f"[27]   ★ 口径核对（闭环真值误差|age ÷ 离线 m(h)）：中位 "
                  f"{al['ratio_median']:.3f}（范围 {al['ratio_min']:.3f}–{al['ratio_max']:.3f}）"
                  " ⇒ " + ("✓ 可迁移" if 0.5 <= al["ratio_median"] <= 2.0
                           else "⚠ 不可迁移（只可作探索性对照）"))
    print("[27] " + "-" * 78)
    for k, b in BASELINES.items():
        print(f"[27]   对照基线 {b['tag']}：距离比 {b['dist_median']}、"
              f"NMSE 比 {b['nmse_median']}（n={b['n_per']}）")
    report["baselines"] = BASELINES
    # ---------- 判据判定（预写口径，先判后解释）----------
    print("[27] " + "-" * 78)
    print(f"[27]   预写判据：O1 两臂距离比中位 **<0.98** ⇒ 误差类触发量**有可达上限**"
          f"（瓶颈=信号不够好）")
    print(f"[27]             O0 两臂 ∈[0.98,1.02] ⇒ **连完美知道当前误差都赢不了**"
          f" ⇒ 瓶颈在**任务侧**")
    print(f"[27]             O_c（自洽）oracle 比 ≤ min(自报 1.0122, X42 0.9969) —— "
          f"信息更多就不该更差；若 >{SELF_CONSISTENCY_MAX} ⇒ **先查实现，不报结论**")
    for arm, d in report["arms"].items():
        dm = d["dist_median"]
        if dm is None:
            print(f"[27]   oracle-{arm}：无可评估档 ⇒ 不下结论")
            continue
        if dm < 0.98:
            v = (f"★ O1 **成立**（距离比 {dm:.4f} < 0.98）⇒ 误差类触发量**存在可达上限**"
                 f" ⇒ 瓶颈是「模型自报的不确定度与真值误差的相关性」，**不是任务侧**")
        elif dm <= 1.02:
            v = (f"★ O0 **成立**（距离比 {dm:.4f} ∈ [0.98,1.02]）⇒ **连完美知道当前估计误差"
                 f"都赢不了年龄阈值** ⇒ 瓶颈**不在触发量的信息量**（任务侧 / 或「误差」"
                 f"本身不是好目标）")
        else:
            v = (f"⚠ 距离比 {dm:.4f} > 1.02 ⇒ oracle **净有害**（不该发生：它的信息严格"
                 f"多于任何自报量）⇒ **先查实现**（R12：量纲/接线），不报此结论")
        # ★ 配对符号检验 vs 同档自报基线（X41-B）
        paired_self, paired_x42 = [], []
        b_self = base_pp.get("x41_b_amp16_self", {})
        b_x42 = base_pp.get("x42_pos_amp16", {})
        for r in d["rows"]:
            if r["metric"] != "mean_dist_tail":
                continue
            k = round(float(r["per"]), 6)
            if k in b_self:
                paired_self.append(b_self[k] - r["ratio"])
            if k in b_x42:
                paired_x42.append(b_x42[k] - r["ratio"])
        st_self, st_x42 = sign_test(paired_self), sign_test(paired_x42)
        # ★ 自洽守卫（承 ㉘ 的教训）：本臂逐档中位必须等于本臂聚合值
        _chk = np.median([r["ratio"] for r in d["rows"] if r["metric"] == "mean_dist_tail"])
        if abs(float(_chk) - float(dm)) > 1e-9:
            raise AssertionError(
                f"★ 自洽守卫未通过：臂 {arm} 的逐档中位 {_chk:.6f} ≠ 聚合值 {dm:.6f}"
                " ⇒ 判定用的是**别的臂**的行（跨臂变量复用）")
        note = (f"  ｜ vs 同档自报 1.0122：中位差 {1.0122 - dm:+.4f}"
                f"（{st_self['k_pos']}/{st_self['n']} 档同向，p={st_self['p_one_sided']:.3f}）"
                f"｜ vs X42-pos 0.9969：中位差 {0.9969 - dm:+.4f}"
                f"（{st_x42['k_pos']}/{st_x42['n']} 档同向，p={st_x42['p_one_sided']:.3f}）")
        print(f"[27]   oracle-{arm}：{v}{note}")
        report["arms"][arm].update({
            "o_delta_vs_self": float(1.0122 - dm),
            "o_delta_vs_x42pos": float(0.9969 - dm),
            "o_sign_test_vs_self": st_self, "o_sign_test_vs_x42pos": st_x42,
            "o_self_consistent": bool(dm <= SELF_CONSISTENCY_MAX),
        })
    # ★ 两臂合一的总判定
    ok = {a: d["dist_median"] for a, d in report["arms"].items()
          if d["dist_median"] is not None}
    if ok:
        worst = max(ok.values())
        best = min(ok.values())
        if worst < 0.98:
            verdict = ("★★ **两臂都 <0.98 ⇒ O1 成立**：oracle（完美知道当前估计误差）确实能赢"
                       "年龄阈值 ⇒ **瓶颈在「信号」而不在「任务」** ⇒ 下一步是提高 U 的质量"
                       "（σ 头 / 更好的不确定度估计），不是换任务。")
        elif best > 1.02:
            verdict = "⚠ 两臂都 >1.02 ⇒ 与理论方向相反 ⇒ **先查实现**（R12/量纲），不出结论。"
        elif all(0.98 <= v <= 1.02 for v in ok.values()):
            verdict = ("★★ **两臂都 ∈[0.98,1.02] ⇒ O0 成立**：连「完美知道当前估计误差」都赢不了"
                       "年龄阈值 ⇒ **瓶颈不在触发量的信息量**（任务侧受限 / 或「当前误差」"
                       "本身不是好目标）。这是本号要的「一锤定音」结果——但只能限定在本场景。")
        else:
            verdict = (f"⚠ **两臂不一致**（best {best:.4f} / worst {worst:.4f}）⇒ "
                       "按臂分别下结论，**不许**合成一个数。")
        report["verdict"] = verdict
        print("[27] " + "-" * 78)
        print(f"[27] ★★ 总判定：{verdict}")
    jpath = os.path.join(out, args.tag + ".json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[27] 产物：{jpath}")
    _figure(report, out, args.tag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
