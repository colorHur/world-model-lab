# -*- coding: utf-8 -*-
"""★ X42 汇报：把**按 PER 分块**跑出来的共形产物合并，算跨 PER 的比值中位。

================================================================= 为什么要有这个脚本
X42 的完整网格（6 个 PER 档 × 41 个工作点）单次跑不完（本机后台任务有 **10 min 硬上限**，
实测两臂都在 10m01s 被终止、连日志都没写出来）。⇒ 改成**按 PER 分块**跑。

★ 分块是**数值等价**的，理由写在代码里：`scripts/24` 的等预算配对是
  `for per in pers:` **逐档独立**插值再比（`overlap_window` / `expo_pair`），
  档与档之间不共享任何量；模型训练只由 `seed` 决定（各块相同）⇒ 合并 = 一次跑全档。

★★ 但分块引入了一个**新的静默失败模式**：某一块没跑成 ⇒ 少一个 PER 档 ⇒
  跨 PER 取中位时**分母悄悄变小**，结论随之变强或变弱，而**看着完全正常**。
  ⇒ 所以本脚本的第一件事是**覆盖守卫**：`--expect-pers` 里的每一档都必须出现，
    缺一档就 raise（R14：报错优于给假数字）。这也是 `test_conformal_merge_*` 锁的东西。

用法：
    python scripts/26_x42_conformal_report.py
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

# ★ 引用基线（**同档 amp=16** 的自报量结果，来自 X41 的 B 步产物 `24_x41_hybrid16_wind16`）
BASELINE_SELF_REPORT = {"tag": "X41-B 自报 z 触发（amp=16，hybrid）",
                        "dist_median": 1.0122, "nmse_median": 1.3133, "n_per": 5}
# ★ 另一条基线：X40（amp=4，自报 z 触发）—— **不同档**，只作旁证
BASELINE_X40_AMP4 = {"tag": "X40 自报 z 触发（amp=4，hybrid）",
                     "dist_median": 1.0648, "nmse_median": 2.0915, "n_per": 4}


def load_chunks(out: str, arm: str) -> tuple[list[dict], list[dict]]:
    """按臂名读取所有分块产物。返回 (P4 行, P2 行)。"""
    files = sorted(glob.glob(os.path.join(out, f"26_x42_{arm}_c*.json")))
    if not files:
        raise FileNotFoundError(f"★ 没找到 {arm} 臂的任何分块产物（26_x42_{arm}_c*.json）")
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


def check_coverage(p4: list[dict], expect_pers: list[float]) -> dict:
    """薄封装：真正的守卫在 `wmlab.eval.conformal.check_per_coverage`（自检 ㊹ 锁它）。"""
    return check_per_coverage(p4, expect_pers)


def _mad(vals: list[float]) -> float:
    """中位绝对偏差（只用档间散布当**噪声尺**，不用"我觉得差不多"）。"""
    a = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    if a.size < 2:
        return float("nan")
    return float(np.median(np.abs(a - np.median(a))))


def load_baseline_rows(out: str) -> list[dict]:
    """★ 同档（amp=16）自报量基线的**逐 PER** 结果，直接读 X41-B 的产物。

    ★ 为什么必须读逐档、不能只抄中位：H2 要问"X42 是否**优于**自报量"，
      而两边的逐档散布都是 ±0.01 量级 ⇒ 只看中位差会把噪声读成增量。
      ⇒ 必须**按 PER 配对**后做符号检验（见 `sign_test`）。
    """
    p = os.path.join(out, "24_x41_hybrid16_wind16.json")
    if not os.path.exists(p):
        return []
    d = json.load(open(p, encoding="utf-8"))
    return [{"per": float(r["per"]), "ratio": float(r["utrigger_over_threshold_median"])}
            for r in d.get("P4_matched_rate", []) if r["metric"] == "mean_dist_tail"]


def sign_test(deltas: list[float]) -> dict:
    """单边符号检验：H0 = 差值的符号是掷硬币（p=0.5）。

    ★★ 为什么必须补这一条（2026-09-29，自证伪 ㉗）：
      我预写的 H2 判据是「中位差 ≥0.01 **且** >2×档间散布」——在 n=5 的档数下
      这个门槛**太松**：本轮实测中位差 0.0153、档间散布 0.0056 ⇒ 判"有增量"，
      但**逐档配对**看是 **4/5 同向**，符号检验单边 **p = 0.1875 > 0.1** ⇒
      **不可分辨**。"2×散布"这种拍系数判据会把倾向性读成结论。
    """
    from math import comb
    d = [x for x in deltas if np.isfinite(x) and abs(x) > 0]
    n, k = len(d), int(sum(1 for x in d if x > 0))
    if n == 0:
        return {"n": 0, "k_pos": 0, "p_one_sided": float("nan")}
    p = sum(comb(n, i) for i in range(k, n + 1)) / (2 ** n)
    return {"n": n, "k_pos": k, "p_one_sided": float(p),
            "resolvable_at_0.10": bool(n >= 5 and p <= 0.10)}


def _figure(report: dict, out: str, tag: str) -> None:
    """★ 进版本库的图：**ASCII 标签**（本机字体链无 CJK，中文会变方框）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6))
    fig.suptitle("X42 conformal-calibrated trigger vs age threshold "
                 "(equal tx budget, wind_amp=16, n=5 PER)", fontsize=11)
    base = report.get("baseline_per_per_dist", [])
    for ax, met, key in ((axes[0], "mean_dist_tail", "dist_median"),
                         (axes[1], "est_nmse", "nmse_median")):
        for arm, d in report["arms"].items():
            rs = [r for r in d["rows"] if r["metric"] == met]
            mv = d.get(key)
            ax.plot([r["per"] for r in rs], [r["ratio"] for r in rs], "o-",
                    label=f"{arm}" + (f" (median {mv:.4f})" if mv is not None else ""))
        if bs := (base if met == "mean_dist_tail" else []):
            ax.plot([b["per"] for b in bs], [b["ratio"] for b in bs], "s--",
                    color="grey", alpha=0.8, label="X41-B self-reported z (median 1.0122)")
        ax.axhline(1.0, color="k", ls=":", lw=1)
        ax.axhline(0.98, color="r", ls="--", lw=1, alpha=0.7)
        ax.axhspan(0.98, 1.02, color="orange", alpha=0.10)
        ax.set_xlabel("PER"); ax.set_ylabel(f"conformal / age-threshold  ({met})")
        ax.set_title("task distance (m)" if met == "mean_dist_tail" else "est NMSE")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p = os.path.join(out, tag + ".png")
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[26] 图：{p}")


def main() -> int:
    ap = argparse.ArgumentParser(description="X42 分块产物合并汇报")
    ap.add_argument("--out", default=None)
    ap.add_argument("--arms", default="pos,nmse")
    ap.add_argument("--expect-pers", default="0.0,0.1,0.3,0.5,0.7",
                   help="★ 必须**逐档**都在产物里出现（缺档 raise，不许在部分档上聚合）")
    ap.add_argument("--known-unevaluable", default="0.9",
                   help="★ 已声明为「场景限制导致不可评估」的 PER 档（不是少跑了）")
    ap.add_argument("--reason", default=(
        "PER=0.9 不可评估，根因 = `periodic` 族可比点塌缩到 1 点"
        "（强衰落下 T≥2 的周期策略 NMSE>1 / E[age]>max_age_K ⇒ 被统一 diverged 口径剔除）"
        "⇒ **强衰落区做不了等预算比较**，这是**场景限制不是方法限制**（承 X40/X41）"),
        help="★ 必须同时给出 reason —— 让「不可评估」是**显式声明**而不是静默挖洞")
    ap.add_argument("--tag", default="26_x42_report")
    args = ap.parse_args()

    out = args.out or output_dir({"output": {"dir": "outputs"}})
    expect = [float(x) for x in args.expect_pers.split(",")]
    base_rows = load_baseline_rows(out)
    base_map = {r["per"]: r["ratio"] for r in base_rows}
    base_mad = _mad([r["ratio"] for r in base_rows])
    report = {"expect_pers": expect, "arms": {},
              "known_unevaluable": [float(x) for x in args.known_unevaluable.split(",") if x],
              "known_unevaluable_reason": args.reason,
              "baseline_per_per_dist": base_rows, "baseline_mad": base_mad}
    print(f"[26]   已知不可评估档（**显式声明**，不是少跑）：{args.known_unevaluable}"
          f" ⇒ {args.reason}")
    print("[26] " + "=" * 78)
    print("[26] ★ X42 共形校准触发：分块合并（按 PER 独立配对 ⇒ 分块数值等价）")
    for arm in [a for a in args.arms.split(",") if a]:
        p4, p2 = load_chunks(out, arm)
        cov = check_coverage(p4, expect)
        dist, nmse, n = aggregate_p4(p4)
        rows = []
        for r in sorted(p4, key=lambda r: (float(r["per"]), r["metric"])):
            if r["metric"] != "mean_dist_tail" and r["metric"] != "est_nmse":
                continue
            rows.append({"per": float(r["per"]), "metric": r["metric"],
                         "ratio": float(r["utrigger_over_threshold_median"]),
                         "n_better": int(r["utrigger_better_n"]),
                         "n_grid": int(r["n_grid"]),
                         "rate_lo": float(r["rate_lo"]), "rate_hi": float(r["rate_hi"]),
                         "span": float(r["rate_hi"]) / max(float(r["rate_lo"]), 1e-12)})
        n_div = int(sum(1 for r in p2 if r.get("diverged")))
        # ★★ S_c4 的披露（必须报）：有多少工作点上**共形信封从未比安全网 K 更早触发**
        #   ⇒ 那些点等价于"纯年龄阈值"，对 H1 不构成证据。不报这个数，
        #   读者会把"族里混着一堆退化点"的比值中位当成"信封的效力"。
        act = [json.load(open(os.path.join(out, os.path.basename(f)), encoding="utf-8"))
               .get("conformal_active") for f in
               sorted(glob.glob(os.path.join(out, f"26_x42_{arm}_c*.json")))]
        acts = [a["frac_active"] for a in act if a]
        frac_active = float(np.mean(acts)) if acts else float("nan")
        dist_rows = [r["ratio"] for r in rows if r["metric"] == "mean_dist_tail"]
        arm_mad = _mad(dist_rows)
        report["arms"][arm] = {"coverage": cov, "dist_median": dist,
                               "nmse_median": nmse, "n_per": n,
                               "dist_mad": arm_mad,
                               "conf_frac_active_mean": frac_active,
                               "rows": rows, "n_diverged_points": n_div}
        print(f"[26] ---- arm = {arm}（K 主导/不可评估等情况见各分块 stdout）")
        for r in rows:
            print(f"[26]   PER={r['per']:<4g} {r['metric']:<15s} 比值中位 "
                  f"{r['ratio']:.4f}（{r['n_better']}/{r['n_grid']} 更优）"
                  f"  窗口 {r['rate_lo']:.4f}–{r['rate_hi']:.4f}（跨度 {r['span']:.2f}×）")
        print(f"[26]   ⇒ **跨 PER 中位**：任务距离比 **{dist if dist is None else round(dist, 4)}**、"
              f"NMSE 比 {nmse if nmse is None else round(nmse, 4)}（n={n} 档）"
              f"｜发散点 {n_div}")
        print(f"[26]   ★ 披露：该臂工作点里「共形信封真的比安全网 K 更早触发」的平均比例 = "
              f"**{frac_active:.2f}**（其余点等价于纯年龄阈值，对 H1 **不构成证据**）")
    print("[26] " + "-" * 78)
    for b in (BASELINE_SELF_REPORT, BASELINE_X40_AMP4):
        print(f"[26]   对照基线 {b['tag']}：距离比 {b['dist_median']}、"
              f"NMSE 比 {b['nmse_median']}（n={b['n_per']}）")
    report["baselines"] = {"x41_b_amp16": BASELINE_SELF_REPORT,
                          "x40_amp4": BASELINE_X40_AMP4}
    # ★ 预写判据的判定（H1 / H0 / H2），**先写死口径再打印结论**
    print("[26] " + "-" * 78)
    for arm, d in report["arms"].items():
        dm = d["dist_median"]
        if dm is None:
            print(f"[26]   {arm}：无可评估档 ⇒ 不下结论")
            continue
        if dm < 0.98:
            v = f"★ H1 **成立**（距离比 {dm:.4f} < 0.98）⇒ 校准量有正边际任务价值"
        elif dm <= 1.02:
            v = (f"★ H0 **成立**（距离比 {dm:.4f} ∈ [0.98, 1.02]）⇒ "
                 "换成有覆盖保证的校准量后，边际任务价值仍为 0")
        else:
            v = f"⚠ 距离比 {dm:.4f} > 1.02 ⇒ 校准量**净有害**（比 X40 更强的一档负面结论）"
        base = BASELINE_SELF_REPORT["dist_median"]
        delta = base - dm
        noise = max([x for x in (arm_mad, base_mad) if np.isfinite(x)] or [float("nan")])
        # ★★ H2（2026-09-29 修订，自证伪 ㉗）：**必须做逐 PER 配对的符号检验**。
        #   原判据「中位差 ≥0.01 且 >2×档间散布」在 n=5 下太松 ——
        #   本轮实测它判"有增量"，而符号检验 p=0.19 ⇒ 不可分辨。
        paired = [(base_map[round(float(r["per"]), 6)] - r["ratio"])
                  for r in d["rows"] if r["metric"] == "mean_dist_tail"
                  and round(float(r["per"]), 6) in base_map]
        # ★★ 自洽守卫（2026-09-29，抓到过一次真 bug）：这里原先误用了**上一轮循环残留**的
        #  模块级变量 `rows` ⇒ 两臂的配对符号检验**都用的是最后一个臂的数据**，
        #   而输出完全正常（数值合理、结论都印出来了）⇒ 只有把"这一臂的 rows"
        #   与"这一臂的中位"对账才能发现。⇒ 断言两者一致，否则 raise。
        _chk = np.median([r["ratio"] for r in d["rows"]
                          if r["metric"] == "mean_dist_tail"])
        if abs(float(_chk) - float(dm)) > 1e-9:
            raise AssertionError(
                f"★ 自洽守卫未通过：臂 {arm} 的逐档中位 {_chk:.6f} ≠ 聚合值 {dm:.6f}"
                " ⇒ 判定用的是**别的臂**的行（典型的跨臂变量复用）")
        st = sign_test(paired)
        ok_inc = bool(abs(delta) >= 0.01 and st["resolvable_at_0.10"])
        if abs(delta) >= 0.01 and not ok_inc:
            h2 = (f" ｜ H2：中位差 {delta:+.4f}（≥0.01）但**逐 PER 配对不可分辨**"
                  f"（{st['k_pos']}/{st['n']} 档同向，单边符号检验 p={st['p_one_sided']:.3f} > 0.10）"
                  " ⇒ **不许声称校准带来了增量**，只能说「不再更差 / 与年龄阈值持平」")
        else:
            h2 = (" ｜ H2：与同档自报基线 1.0122 相比 "
                  + ("**有增量**（方向对、差 ≥0.01、且符号检验 p ≤0.10）" if ok_inc
                     else "无增量"))
        print(f"[26]   {arm}：{v}{h2}")
        report["arms"][arm].update({"h2_delta_vs_self_report": float(delta),
                                    "h2_noise_scale": float(noise),
                                    "h2_sign_test": st,
                                    "h2_paired_deltas": [float(x) for x in paired],
                                    "h2_resolvable": bool(ok_inc)})
    jpath = os.path.join(out, args.tag + ".json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[26] 产物：{jpath}")
    _figure(report, out, args.tag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
