"""X41：U 边际任务价值的**剂量—反应**。

★ 要回答的问题（审 X40 前提的产物）
------------------------------------------------------------------
X40 的结论是「世界模型自报的累积不确定度 U 在调度上**没有边际任务价值**」，
但它**只在 `wind_amp = 4` 一档**成立；而该档的离线判别力 `AUC_cond = 0.601`
**只比随机高 0.10（压线）** ⇒ 存在一个未排除的替代解释：

    「不是 U 无用，而是**信息还不够强**。信息再强一点，价值就会涌现。」

本脚本把 **X38-b 的离线信息量** × **X40/X41 的在线任务价值** 放到
**同一条剂量轴**（`CV(σ_true)` —— 异方差强度的物理度量，而不是旋钮 `wind_amp`）上，
直接检验「信息涌现」是否伴随「价值涌现」。

★ 两条独立路径（铁律 4）
------------------------------------------------------------------
① **离线**（便宜）：`--p3-only` 7 档 —— ρ_cond / AUC_cond / ρ(σ_pred,σ_true)；
② **在线**（贵）：完整闭环 hybrid vs 年龄阈值 的等预算比值。

★ 只读已存产物、不重跑仿真 ⇒ 秒级、可反复复算。

用法：`python scripts/25_x41_dose_response.py`
"""

from __future__ import annotations

import io
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wmlab.eval.pairing import aggregate_p4  # noqa: E402
from wmlab.utils.plot import PALETTE, apply_style, save_fig  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")

# ★ 预写门槛（跑前定死，不许事后挪动）—— 与总纲 X41 行一致
AUC_GATE = 0.65      # A1：离线信息「真的够强」的门槛
AUC_WEAK = 0.60      # X38/X40 用过的弱门槛（对照）
CALIB_GATE = 0.60    # A2：σ 头「学得动」的门槛（低于此 ⇒ 不是环境没信息）
VALUE_GATE = 0.98    # B1：任务价值「涌现」的门槛（距离中位比 < 0.98）

AMPS = [0, 1, 2, 4, 8, 16, 32]


def _offline_path(amp: int) -> str | None:
    """离线产物：amp≤4 用 X38-b 的原产物；amp>4 用 X41 新扫的产物。"""
    for p in (os.path.join(OUT, f"24_x41_dose_wind{amp:g}.json"),
              os.path.join(OUT, f"24_triggered_scheduling_wind{amp:g}.json")):
        if os.path.exists(p):
            return p
    return None


def _closed_loop_path(amp: int) -> str | None:
    """完整闭环产物（X40 的 amp=4 / X41 的 amp=16）。"""
    for p in (os.path.join(OUT, f"24_x41_hybrid{amp:g}_wind{amp:g}.json"),
              os.path.join(OUT, f"24_x40_final_wind{amp:g}.json"),
              os.path.join(OUT, f"24_triggered_scheduling_wind{amp:g}.json")):
        if os.path.exists(p):
            d = json.load(io.open(p, encoding="utf-8"))
            if "P4_matched_rate" in d:      # 只有完整跑才有 P4
                return p
    return None


def _value_of(path: str) -> tuple[float | None, float | None, int]:
    """从完整闭环产物读 P4 表并聚合。

    ★ 聚合口径（跨 PER 取**中位**、分别用任务指标与代理指标）已抽到
    `wmlab.eval.pairing.aggregate_p4`，由**回归测试 ㊶** 锁死 —— 见该函数 docstring。
    """
    d = json.load(io.open(path, encoding="utf-8"))
    return aggregate_p4(d.get("P4_matched_rate") or [])


def main() -> int:
    apply_style()
    off, cl = [], []
    for amp in AMPS:
        p = _offline_path(amp)
        if p is None:
            continue
        d = json.load(io.open(p, encoding="utf-8"))
        pc = d["P3_conditional"]
        sc = pc.get("sigma_calib", {}) or {}
        off.append(dict(amp=amp, path=os.path.basename(p),
                        cv_true=sc.get("cv_true"), cv_pred=sc.get("cv_pred"),
                        rho_calib=sc.get("rho_calib"),
                        rho_cond=pc.get("rho_conditional_median"),
                        auc=pc.get("auc_conditional_median")))
        q = _closed_loop_path(amp)
        if q is not None:
            dist, nmse, n_pair = _value_of(q)
            cl.append(dict(amp=amp, path=os.path.basename(q),
                           dist=dist, nmse=nmse, n=n_pair))

    # ---------------- 打印表（可复制的数字） ----------------
    print("=" * 96)
    print("X41 剂量—反应：离线信息量（7 档 --p3-only）")
    print("-" * 96)
    print(f"{'amp':>4} {'CV(σ_true)':>11} {'CV(σ_pred)':>11} {'ρ(σpred,σtrue)':>15} "
          f"{'ρ_cond':>9} {'AUC_cond':>9}  {'判定':<24}")
    for r in off:
        cv = r["cv_true"] if r["cv_true"] is not None else 0.0
        rc = r["rho_cond"]
        rc_txt = "nan" if rc is None else f"{rc:+.4f}"
        calib = r["rho_calib"]
        calib_txt = "nan" if calib is None else f"{calib:+.4f}"
        auc = r["auc"]
        if auc >= AUC_GATE:
            verdict = "★ 够强 ⇒ 须跑闭环 (B1)"
        elif calib is not None and calib < CALIB_GATE:
            verdict = "⚠ σ 头学不动 ⇒ 不是信息问题 (A2)"
        elif auc >= AUC_WEAK:
            verdict = "弱门槛内"
        else:
            verdict = "弱"
        print(f"{r['amp']:>4} {cv:>11.4f} "
              f"{(r['cv_pred'] if r['cv_pred'] is not None else 0):>11.4f} "
              f"{calib_txt:>15} {rc_txt:>9} {auc:>9.4f}  {verdict:<24}")
    print("-" * 96)
    print("X41 剂量—反应：在线任务价值（完整闭环，hybrid ÷ 年龄阈值，<1 = U 有用）")
    print("-" * 96)
    print(f"{'amp':>4} {'距离中位比':>12} {'NMSE 中位比':>12} {'n(PER)':>7}  产物")
    for r in cl:
        dt = "n/a" if r["dist"] is None else f"{r['dist']:.4f}"
        nm = "n/a" if r["nmse"] is None else f"{r['nmse']:.4f}"
        print(f"{r['amp']:>4} {dt:>12} {nm:>12} {r['n']:>7}  {r['path']}")
    print("=" * 96)

    # ---------------- 图 ----------------
    cv = [r["cv_true"] if r["cv_true"] is not None else 0.0 for r in off]
    amp_ticks = [r["amp"] for r in off]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))
    fig.suptitle("X41 dose-response: does stronger heteroscedasticity turn U's "
                 "information into task value?", fontsize=11)

    ax = axes[0]
    ax.plot(cv, [r["auc"] for r in off], "o-", color=PALETTE["blue"], label="conditioned AUC")
    ax.plot(cv, [r["rho_cond"] for r in off], "s--", color=PALETTE["red"], label=r"conditioned $\rho$")
    ax.axhline(AUC_GATE, color="k", ls=":", lw=1)
    ax.text(cv[0], AUC_GATE + 0.008, f"A1 gate {AUC_GATE}", fontsize=8)
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.text(cv[0], 0.507, "random (AUC=0.5)", fontsize=8, color="gray")
    for r, x in zip(off, cv):
        ax.annotate(f"amp={r['amp']:g}", (x, r["auc"]), textcoords="offset points",
                    xytext=(0, 7), fontsize=7, ha="center")
    ax.set_xlabel(r"$CV(\sigma_{true})$  (heteroscedasticity strength)")
    ax.set_ylabel("information given age")
    ax.set_title("(1) offline: residual information")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    calib = [r["rho_calib"] if (r["rho_calib"] is not None and r["rho_calib"] == r["rho_calib"])
             else np.nan for r in off]
    ax.plot(cv, calib, "o-", color=PALETTE["green"], label=r"$\rho(\sigma_{pred},\sigma_{true})$")
    ax.axhline(CALIB_GATE, color="k", ls=":", lw=1)
    ax.text(cv[0], CALIB_GATE + 0.012, f"A2 gate {CALIB_GATE}", fontsize=8)
    for r, x, c in zip(off, cv, calib):
        if c == c:
            ax.annotate(f"amp={r['amp']:g}", (x, c), textcoords="offset points",
                        xytext=(0, 7), fontsize=7, ha="center")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(r"$CV(\sigma_{true})$")
    ax.set_ylabel(r"$\sigma$-head calibration")
    ax.set_title("(2) offline: can the sigma-head learn it?")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[2]
    if cl:
        cx = [r["amp"] for r in cl]
        cvs = []
        for a in cx:
            cvs.append(next((r["cv_true"] for r in off if r["amp"] == a), np.nan))
        ax.plot(cvs, [r["dist"] for r in cl], "o-", color=PALETTE["orange"],
                label="hybrid / age-threshold (dist median)")
        ax.axhline(1.0, color="k", ls=":", lw=1)
        ax.axhline(VALUE_GATE, color="r", ls="--", lw=1)
        ax.text(np.nanmin(cvs), VALUE_GATE - 0.03, f"B1 gate {VALUE_GATE}", fontsize=8, color="r")
        for r, x in zip(cl, cvs):
            ax.annotate(f"amp={r['amp']:g}", (x, r["dist"]), textcoords="offset points",
                        xytext=(0, 7), fontsize=7, ha="center")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "closed-loop runs pending", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_xlabel(r"$CV(\sigma_{true})$")
    ax.set_ylabel("ratio (<1 = U trigger better)")
    ax.set_title("(3) online: marginal task value")
    ax.grid(alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = os.path.join(OUT, "25_x41_dose_response.png")
    save_fig(fig, path)
    plt.close(fig)
    print(f"[25] 产物图：{path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
