#!/usr/bin/env python
"""CW-R5.8 三张主图：**只读既有证据 JSON**，不 import 任何 DP 引擎。

冻结纪律（预注册 §2.4）：
  F-PROV   图上每点都能指回某证据 JSON 的 `(n_w, tau, b, p)` 格（写进 `fig_data.json`）
  F-NOFAKE `IV(p=0)=0` 的格显式标"无靶子"，不当成 0%
  F-SLACK  图 2 的归一化值必须逐位 ∈ {0.0, 1.0}
  F-SRC    本脚本不得 import `r5_6_grid_l1` / `r5_7_price_panel` / `Quad`（AST 自查）

用法：
  python -m experiments.cw_v1.r5_8_figures [--out DIR]
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "cw_v1"

ROUND = "CW-R5.8"
FIG_ROUND = "figs"
PROTOCOL = "CW-POMDP-v1.5-grid"

# ★ 显式目录名，不用 glob(...)[-1]（STATE §7.3）
DIRS = {
    "main2u": "cw_r5_7_price_main2u_20261027T033108_8a649d26fbc4bb1d",
    "ext": "cw_r5_7_price_ext_nonmult_20261027T033216_8a649d26fbc4bb1d",
    "ptot1_nw33": "cw_r5_6_grid_l1_n2_ptot1_nw33_20261027T011900_4c078bdc00322cac",
    "ptotinf_nw33": "cw_r5_6_grid_l1_n2_ptot_inf_nw33_20261027T011845_4c078bdc00322cac",
    "size_nw33_65": "cw_r5_6_grid_l1_size_20261027T011934_4c078bdc00322cac",
    "size_nw129_inf": "cw_r5_6_grid_l1_size_20261027T054055_8a649d26fbc4bb1d",
    "size_nw129_p2": "cw_r5_6_grid_l1_size_20261027T054125_8a649d26fbc4bb1d",
}

FORBIDDEN = ("r5_6_grid_l1", "r5_7_price_panel", "Quad", "world-model-lab.experiments")

P_TICKS = (0, 1, 2, 4, 8)
SEND_TICKS = 8
# ★ 噪声地板：`A0p − VL1` 的浮点尘最大量到 2.78e-17（相对 2.4e-16）。
#   判"无靶子"用相对阈值，**不**用 `== 0.0`（`τ=5,b=24` 在 `n_w=257` 上不是逐位 0）。
NOISE_REL = 1e-15


# ------------------------------------------------------------------ 工具
def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def _now_utc() -> str:
    return time.strftime("%Y%m%dT%H%M%S", time.gmtime())


def load_evidence() -> dict:
    ev = {}
    for key, dname in DIRS.items():
        d = RES / dname
        if not d.is_dir():
            raise FileNotFoundError(f"[{key}] 缺证据目录：{d}")
        js = sorted(p for p in d.glob("*.json"))
        if not js:
            raise FileNotFoundError(f"[{key}] 目录里没有 json：{d}")
        ev[key] = {"dir": dname, "files": [(p.name, _sha(p)) for p in js],
                   "objs": [json.loads(p.read_text(encoding="utf-8")) for p in js]}
    return ev


def panel_cells(ev: dict, keys: tuple[str, ...]) -> dict:
    """把多个 panel 目录的 cells 合并成 {(n_w,tau,b): (cell, srcdir)}。"""
    out = {}
    for key in keys:
        e = ev[key]
        for obj in e["objs"]:
            for i, c in enumerate(obj.get("cells", [])):
                k = (int(c["n_w"]), int(c["tau"]), int(c["b_ticks"]))
                if k in out and out[k][0] != c:
                    raise RuntimeError(f"同格不同值，拒绝合并：{k}")
                out[k] = (c, e["dir"], i)
    return out


# ------------------------------------------------------------------ 闸
def g_src_ast() -> dict:
    """F-SRC：本脚本不得 import 任何 DP 引擎。"""
    src = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if any(f in a.name for f in FORBIDDEN):
                    hits.append(f"import {a.name}")
        elif isinstance(node, ast.ImportFrom):
            mod = (node.module or "")
            names = [a.name for a in node.names]
            if any(f in mod for f in FORBIDDEN):
                hits.append(f"from {mod} import ...")
            for n in names:
                if n in FORBIDDEN:
                    hits.append(f"from {mod} import {n}")
    return {"pass": not hits, "hits": hits,
            "checked": "源码 AST 里的 import 节点", "forbidden": list(FORBIDDEN)}


# ------------------------------------------------------------------ 图 1
def build_fig1(main: dict, points: list) -> dict:
    """rel(p=0) 对 (b, tau)，误差棒 = 三点 n_w 的 max-min。"""
    idx = {}
    for (n_w, tau, b), (c, srcd, ci) in main.items():
        idx[(n_w, tau, b)] = (c, srcd, ci)
    bs = sorted({b for (_, _, b) in idx})
    taus = sorted({t for (_, t, _) in idx})
    series = {}
    for tau in taus:
        xs, ys, los, his, notes = [], [], [], [], []
        for b in bs:
            nws = [n for (n, t, bb) in idx if t == tau and bb == b]
            if not nws:
                continue
            vals = {}
            for n in sorted(nws):
                c, srcd, ci = idx[(n, tau, b)]
                v = c["IV_probe_L1"]["0"]
                a0p = c["A0p"]
                # F-REL：JSON 里的 rel_IV 必须等于 IV/A0p
                if c["rel_IV"]["0"] != (v / a0p):
                    raise RuntimeError(f"rel_IV 与 IV/A0p 不一致：n_w={n} tau={tau} b={b}")
                vals[n] = v / a0p
                points.append({"figure": "fig1", "n_w": n, "tau": tau, "b_ticks": b,
                               "p_ticks": 0, "quantity": "rel(p=0)", "value": v / a0p,
                               "abs_IV": v, "A0p": a0p,
                               "no_target": (v <= NOISE_REL * a0p),
                               "source_dir": srcd, "cell_index": ci})
            n_max = max(vals)
            if vals[n_max] <= NOISE_REL:
                xs.append(b)
                ys.append(float("nan"))
                los.append(0.0)
                his.append(0.0)
                notes.append(f"b={b}: NO-TARGET (IV(0)<=noise)")
            else:
                xs.append(b)
                ys.append(vals[n_max] * 100.0)
                lo, hi = min(vals.values()) * 100.0, max(vals.values()) * 100.0
                los.append(ys[-1] - lo)
                his.append(hi - ys[-1])
                notes.append(f"b={b}: slack={b % SEND_TICKS}")
        series[tau] = {"x": xs, "y": ys, "yerr": [los, his], "notes": notes}

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    markers = {taus[0]: "o", taus[1] if len(taus) > 1 else taus[0]: "s"}
    cols = {taus[0]: "#1f4e79", taus[1] if len(taus) > 1 else taus[0]: "#b45309"}
    for tau, s in series.items():
        ax.errorbar(s["x"], s["y"], yerr=s["yerr"], marker=markers[tau], capsize=3,
                    lw=1.2, ms=5, color=cols[tau], label=fr"$\tau={tau}$ (action slots $\tau-1={tau - 1}$)")
    b_all = sorted({b for (_, _, b) in idx})
    for b in b_all:
        ax.axvline(b, color="0.85", lw=0.6, zorder=0)
    ax.set_xlabel(r"budget $b$ (ticks; 1 send = 8 ticks)")
    ax.set_ylabel(r"$IV_{\rm probe}(p{=}0)\,/\,A_{0p}$  [%]")
    ax.set_title("Fig.1  Measured value of one channel probe (L1, exact grid Bayes, $n{=}2$)")
    ax.set_xticks(b_all)
    ax.set_xticklabels([f"{b}\n(s={b % SEND_TICKS})" for b in b_all], fontsize=8)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(axis="y", ls=":", lw=0.6, alpha=0.7)
    # 无靶子格标注（画空心 × 在 y=0，箭头只指向该 ×）
    for tau, s in series.items():
        for x, y, n in zip(s["x"], s["y"], s["notes"]):
            if y != y:
                ax.plot([x], [0.0], marker="x", ms=7, mew=1.4, color=cols[tau], alpha=0.75)
                ax.annotate("no target\n($IV(0)\\leq 3{\\times}10^{-17}$)",
                            xy=(x, 0.0), xytext=(x - 2.9, 0.24), fontsize=6.0,
                            ha="center", va="bottom", color="0.35",
                            arrowprops=dict(arrowstyle="-", lw=0.6, color="0.5"))
    ax.set_ylim(-0.03, max(0.72, ax.get_ylim()[1]))
    ax.annotate("all measured values < 1% of $A_{0p}$", xy=(0.02, 0.94),
                xycoords="axes fraction", fontsize=7, color="0.35")
    fig.tight_layout()
    return {"fig": fig, "series": series, "b_all": b_all}


# ------------------------------------------------------------------ 图 2
def build_fig2(main: dict, points: list) -> dict:
    bs = sorted({b for (nw, t, b) in main if nw == 513})
    taus = sorted({t for (nw, t, b) in main if nw == 513})
    fig, axes = plt.subplots(1, len(taus), figsize=(9.4, 3.7), sharey=True)
    if len(taus) == 1:
        axes = [axes]
    info = {}
    for ax, tau in zip(axes, taus):
        for b in bs:
            k = (513, tau, b)
            if k not in main:
                continue
            c, srcd, ci = main[k]
            iv0 = c["IV_probe_L1"]["0"]
            if iv0 <= NOISE_REL * c["A0p"]:
                ax.plot([], [])
                info.setdefault(tau, {})[b] = {"status": "NO-TARGET", "IV0": iv0}
                continue
            xs = list(range(len(P_TICKS)))
            ys = []
            for i, p in enumerate(P_TICKS):
                v = c["IV_probe_L1"][str(p)]
                ratio = v / iv0
                if ratio not in (0.0, 1.0):
                    raise RuntimeError(f"F-SLACK 违反：tau={tau} b={b} p={p} ratio={ratio!r}")
                ys.append(ratio)
                points.append({"figure": "fig2", "n_w": 513, "tau": tau, "b_ticks": b,
                               "p_ticks": p, "quantity": "IV(p)/IV(0)", "value": ratio,
                               "abs_IV": v, "IV0": iv0,
                               "source_dir": srcd, "cell_index": ci})
            ax.plot(xs, ys, marker="o", ms=4.5, lw=0.9, ls=":",
                    label=fr"$b{{=}}{b}$ ($s{{=}}{b % SEND_TICKS}$), $IV_0{{=}}{iv0:.1e}$")
            info.setdefault(tau, {})[b] = {"status": "measured", "IV0": iv0}
        ax.axvline(0.0, color="0.8", lw=0.6)
        ax.set_xticks(range(len(P_TICKS)))
        ax.set_xticklabels([str(p) for p in P_TICKS])
        ax.set_xlabel(r"probe price $p$ (ticks; 8 = one full send)")
        ax.set_title(fr"$\tau={tau}$", fontsize=10)
        ax.grid(axis="y", ls=":", lw=0.6, alpha=0.7)
        ax.set_ylim(-0.12, 1.22)
    axes[-1].legend(fontsize=6.2, frameon=False, loc="center left",
                    bbox_to_anchor=(1.02, 0.5), title="curves", title_fontsize=6.5)
    axes[0].set_ylabel(r"$IV_{\rm probe}(p)\,/\,IV_{\rm probe}(0)$")
    fig.suptitle("Fig.2  Price response is a step at $p = b\\ \\mathrm{mod}\\ 8$, not a curve "
                 "($n_w{=}513$)", fontsize=10)
    fig.text(0.5, 0.915,
             "only 5 prices sampled; dotted segments are guides, not interpolation. "
             "Curves with the same $s=b\\,\\mathrm{mod}\\,8$ coincide exactly. "
             "Legend $IV_0$ values are from the $\\tau{=}6$ panel.",
             fontsize=6.4, color="0.4", ha="center")
    fig.tight_layout(rect=(0, 0, 0.83, 0.90))
    return {"fig": fig, "info": info, "bs": bs, "taus": taus}


# ------------------------------------------------------------------ 图 3
def build_fig3(main: dict, ev: dict, points: list) -> dict:
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(8.4, 3.6),
                                  gridspec_kw={"width_ratios": [1.15, 1.0]})
    # ---- 左：n_w 收敛
    show = [(6, 8), (6, 16), (6, 24), (5, 8)]
    mk = ["o", "s", "^", "D"]
    nws = [129, 257, 513]
    devs: list[float] = []
    for (tau, b), m in zip(show, mk):
        ys, prov = [], []
        for n in nws:
            k = (n, tau, b)
            if k not in main:
                ys.append(float("nan"))
                continue
            c, srcd, ci = main[k]
            ys.append(c["IV_probe_L1"]["0"])
            prov.append({"n_w": n, "source_dir": srcd, "cell_index": ci})
        ref = ys[-1]
        if ref != ref or ref <= NOISE_REL:
            continue
        rel = [y / ref for y in ys]
        devs.extend(abs(r_ - 1.0) for r_ in rel)
        axl.plot(nws, rel, marker=m, ms=4, lw=1.1, label=fr"$\tau={tau}, b={b}$")
        for p_, r_ in zip(prov, rel):
            points.append({"figure": "fig3L", "n_w": p_["n_w"], "tau": tau, "b_ticks": b,
                           "quantity": "IV(0) rel. to n_w=513", "value": r_,
                           "source_dir": p_["source_dir"], "cell_index": p_["cell_index"]})
    axl.axhline(1.0, color="0.7", lw=0.7, ls="--")
    axl.set_xticks(nws)
    axl.ticklabel_format(axis="y", useOffset=False, style="plain")
    axl.set_xlabel(r"grid size $n_w$")
    axl.set_ylabel(r"$IV_{\rm probe}(0)$ rel. to $n_w{=}513$")
    axl.set_title("(a) resolution convergence", fontsize=9)
    axl.legend(fontsize=7, frameon=False)
    axl.grid(ls=":", lw=0.6, alpha=0.7)
    axl.annotate(f"max $|{{\\rm dev}}|={max(devs):.2e}$" if devs else "",
                 xy=(0.30, 0.86), xycoords="axes fraction", fontsize=6.5, color="0.4")

    # ---- 右：probe_total=1 的限制代价
    c1 = panel_cells(ev, ("ptot1_nw33",))
    cinf = panel_cells(ev, ("ptotinf_nw33",))
    rows = []
    for k, (c, srcd, ci) in sorted(c1.items()):
        if k not in cinf:
            continue
        v1 = c["IV_probe_L1"]
        vinf = cinf[k][0]["IV_probe_L1"]
        if vinf <= NOISE_REL * c["A0p"]:
            rows.append({"n_w": k[0], "tau": k[1], "b_ticks": k[2], "status": "NO-TARGET",
                         "cost_rel": None, "IV_total1": v1, "IV_totalinf": vinf,
                         "src_total1": srcd, "src_totalinf": cinf[k][1]})
            continue
        rows.append({"n_w": k[0], "tau": k[1], "b_ticks": k[2], "status": "measured",
                     "IV_total1": v1, "IV_totalinf": vinf,
                     "cost_rel": (vinf - v1) / vinf,
                     "delta": vinf - v1,
                     "src_total1": srcd, "src_totalinf": cinf[k][1]})
    meas = [r for r in rows if r["status"] == "measured"]
    labels = [fr"$\tau{{=}}{r['tau']},b{{=}}{r['b_ticks']}$" for r in meas]
    vals = [r["cost_rel"] * 100.0 for r in meas]
    xs = list(range(len(meas)))
    ymax = max(0.30, max(vals) * 2.2 if vals else 0.30)
    axr.bar(xs, vals, color=["#1f4e79" if v > 0 else "#9ca3af" for v in vals], width=0.55)
    for x, v, r in zip(xs, vals, meas):
        if v <= 0.0:
            axr.annotate("0.00%\n$\\Delta$=0", (x, 0.004), fontsize=5.6, ha="center",
                         va="bottom", color="0.35")
        else:
            axr.annotate(f"{v:.2f}%\n$\\Delta$={r['delta']:.2e}", (x, v), fontsize=6.0,
                         ha="center", va="bottom")
    # n_w=129：UNMEASURED —— 画成整格斜纹区，**不画柱高**（没有数值就一个像素都不给）
    x_lo, x_hi = len(meas) + 0.25, len(meas) + 0.95
    axr.axvspan(x_lo, x_hi, ymin=0.0, ymax=1.0, facecolor="none", hatch="///",
                edgecolor="0.55", lw=0.0, alpha=0.9)
    axr.annotate("UNMEASURED\n$p_{tot}{=}\\infty$ and 2\nboth hit cap\nat $n_w{=}129$",
                 (len(meas) + 0.6, ymax * 0.45), fontsize=5.6, ha="center", va="center",
                 color="0.3")
    axr.set_xticks(xs + [len(meas) + 0.6])
    axr.set_xticklabels(labels + ["$n_w{=}129$"], fontsize=6.3)
    axr.set_xlim(-0.6, len(meas) + 1.0)
    axr.set_ylabel("cost of $p_{total}{=}1$  [% of $IV(0)$]")
    axr.set_title("(b) cost of the $p_{total}{=}1$ restriction\n(measured at $n_w{=}33$ only)",
                  fontsize=8.5)
    axr.set_ylim(0, ymax)
    axr.grid(axis="y", ls=":", lw=0.6, alpha=0.7)
    fig.suptitle("Fig.3  What the numbers are worth, and what is still unmeasured", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return {"fig": fig, "rows": rows}


# ------------------------------------------------------------------ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="CW-R5.8 三张主图（只读证据）")
    ap.add_argument("--out", default="")
    ap.add_argument("--tag", default="all")
    args = ap.parse_args(argv)

    dest = Path(args.out) if args.out else (
        RES / f"cw_r5_8_{FIG_ROUND}_{args.tag}_{_now_utc()}_{_sha(Path(__file__))}")
    dest.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    gates = {"g_src_ast": g_src_ast()}
    if not gates["g_src_ast"]["pass"]:
        (dest / "gates.json").write_text(json.dumps(gates, indent=2, ensure_ascii=False),
                                         encoding="utf-8")
        print("[FATAL] F-SRC 违反：脚本 import 了 DP 引擎", flush=True)
        return 2

    ev = load_evidence()
    main_cells = panel_cells(ev, ("main2u", "ext"))
    points: list[dict] = []

    f1 = build_fig1(main_cells, points)
    f1["fig"].savefig(dest / "fig1_value_vs_budget.png", dpi=300)
    f1["fig"].savefig(dest / "fig1_value_vs_budget.svg")
    plt.close(f1["fig"])

    f2 = build_fig2(main_cells, points)
    f2["fig"].savefig(dest / "fig2_price_step_family.png", dpi=300)
    f2["fig"].savefig(dest / "fig2_price_step_family.svg")
    plt.close(f2["fig"])

    f3 = build_fig3(main_cells, ev, points)
    f3["fig"].savefig(dest / "fig3_limits.png", dpi=300)
    f3["fig"].savefig(dest / "fig3_limits.svg")
    plt.close(f3["fig"])

    # ---- F-PROV：每点必须有出处，且出处文件确实存在
    missing = [p for p in points if not p.get("source_dir")]
    prov_ok = not missing
    for p in points:
        d = RES / p["source_dir"]
        if not d.is_dir():
            prov_ok = False

    # ---- F-NOFAKE：fig1 里 IV(0)=0 的格必须被标 no_target，且不被画成 0%
    fig1_nt = [p for p in points if p["figure"] == "fig1" and p.get("no_target")]
    n_nt_b24 = sum(1 for p in fig1_nt if p["b_ticks"] == 24 and p["tau"] == 5)
    gates["g_prov"] = {"pass": prov_ok, "n_points": len(points), "n_missing": len(missing),
                       "checked": "每个绘点都带 source_dir + cell_index，且目录存在"}
    gates["g_nofake"] = {"pass": (n_nt_b24 == 3 and len(fig1_nt) == 3),
                         "n_no_target_points": len(fig1_nt),
                         "n_no_target_tau5_b24": n_nt_b24,
                         "checked": ("IV(0)=0 的格必须带 no_target 标记、且不在图 1 里画成 0%；"
                                     "预期全表只有 tau=5,b=24 的三个 n_w")}
    gates["g_slack"] = {"pass": True, "checked": "fig2 全部归一化值 ∈{0,1}（违反会直接抛异常）",
                        "n_fig2_points": sum(1 for p in points if p["figure"] == "fig2")}
    gates["g_scale_probe"] = {
        "pass": True,
        "checked": "ptotal=inf 与 ptotal=2 在 n_w=129 的规模探针结果（撞顶 = UNMEASURED）",
        "n_w129_ptotinf": next(o for o in ev["size_nw129_inf"]["objs"][0] if o["n_w"] == 129),
        "n_w129_ptot2": next(o for o in ev["size_nw129_p2"]["objs"][0] if o["n_w"] == 129),
    }
    gate_fail = [k for k, v in gates.items() if v.get("pass") is False]

    payload = {
        "round": ROUND, "protocol": PROTOCOL, "utc": _now_utc(),
        "note": ("本轮**零新 DP 计算**：三张图全部由既有证据 JSON 重绘。"
                 "唯一新算 = 两个规模探针（ptotal=inf / ptotal=2 在 n_w=129），均撞顶 ⇒ UNMEASURED。"),
        "sec": round(time.perf_counter() - t0, 3),
        "src_hash": _sha(Path(__file__)),
        "evidence": {k: {"dir": v["dir"], "files": v["files"]} for k, v in ev.items()},
        "fig1": {"series": {str(k): v for k, v in f1["series"].items()}, "b_all": f1["b_all"]},
        "fig2": {"info": {str(k): {str(b): v for b, v in d.items()} for k, d in f2["info"].items()}},
        "fig3": {"ptotal_rows": f3["rows"]},
        "gates": gates,
        "gate_fail": gate_fail,
        "points": points,
    }
    (dest / "fig_data.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False,
                                                   default=str), encoding="utf-8")
    (dest / "gates.json").write_text(json.dumps(gates, indent=2, ensure_ascii=False, default=str),
                                     encoding="utf-8")
    (dest / "commands.txt").write_text(
        f"python -m experiments.cw_v1.r5_8_figures --tag {args.tag}\n"
        f"# 只读证据，不 import DP 引擎；F-SRC 由 AST 自查\n",
        encoding="utf-8")
    print(f"[figs] {dest}", flush=True)
    print(f"[gates] fail={gate_fail}", flush=True)
    print(f"[sec] {payload['sec']}", flush=True)
    return 0 if not gate_fail else 1


if __name__ == "__main__":
    raise SystemExit(main())
