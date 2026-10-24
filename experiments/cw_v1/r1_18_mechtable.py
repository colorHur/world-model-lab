# -*- coding: utf-8 -*-
"""CW-R1.18 用 `ku/kf` 重写机制表，给 `MIXED` 一个替代判词 —— **零新 episode**。

预注册：`docs/research/CW-R1.18-机制表重写-预注册-v0.1.md`（§9 记跑后的修订）

三问（预注册 §1）
-----------------
`Q1` 把份额式逐 ep 标签（`share≥0.60`）换成**方向+量级**式标签后总判词是什么？
`Q2` 新旧逐 ep 标签差在哪 —— 是**问法不同**还是**判断冲突**？（冲突 ≥3 且无法解释 ⇒ 停）
`Q3` `ku` 的闭式是什么？它是不是**同义反复**？

★ 本轮**不跑 episode**，只读 R1.16 的 `cells.json`（80 格）与 `decomp.json`（旧标签对账用）。
★ 被测物 `plan_ref/plan_dp/world/runner/config/policies` 一字节不动。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import gates as G
from .mech import (KAPPA_MIN, SIGN_MIN, aggregate_mech_verdict, ep_mech_label,
                   kf_bound, mech_quantities, mech_state, two_user_closed_form)
from .r1_14_exactbase import RESULTS, _git, _src_hash
from .r1_15_delivery_gap import classify as R15_classify
from .r1_15_delivery_gap import ep_counts

R116_DIR = RESULTS / "cw_r1_16_n10_20261024T105416_fc9f25e16d2c6307"
R116_CELLS = R116_DIR / "cells.json"
R116_DECOMP = R116_DIR / "decomp.json"

EPS10 = tuple(range(100, 110))

#: 预注册 §6：冲突判据里"旧标签断言该通道主导 ⇒ 新标签必须说它有害"
CONFLICT_OLD = ("H-A-FRAME-SELECTION", "H-B-USER-ALLOCATION", "BOTH-MECHANISMS")


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


# ============================================================ 判据自证（零 episode）
def controls() -> dict:
    """预注册 §7 的 `S1–S7`，答案全部**手算/构造已知**。"""
    out: dict = {}

    # ---- S1/S2 2 用户纯平衡交换：ku = ±(ρ_0−ρ_1) = ±0.8（手算）
    def _swap(sign: int):
        ni_a = np.array([[100], [100]], dtype=float)
        nd_a = np.array([[90], [10]], dtype=float)
        ni_e = ni_a + np.array([[40], [-40]], dtype=float) * sign
        return ni_e, nd_a.copy(), ni_a, nd_a

    for tag, sgn, expect_ku, expect_state in (("S1", +1, 0.8, "POS"), ("S2", -1, -0.8, "NEG")):
        cf = two_user_closed_form(*_swap(sgn))
        lb = ep_mech_label(cf["ku_closed_form"], mech_quantities(*_swap(sgn))["kf"])
        out[tag] = {"ku": cf["ku_closed_form"], "expect": expect_ku,
                    "ku_net": cf["ku_net"], "ku_swap": cf["ku_swap"],
                    "residual": cf["abs_residual"], "state": lb["u_state"],
                    "gap_abs": cf["gap_abs"],
                    "pass": bool(abs(cf["ku_closed_form"] - expect_ku) < 1e-12
                                 and cf["abs_residual"] < 1e-12
                                 and lb["u_state"] == expect_state
                                 and abs(cf["gap_abs"] - 0.8) < 1e-12)}

    # ---- S3 净增无差分 + 对称率 ⇒ ku = 0 而 Mu > 0（机制发生 ≠ 有差分伤害）
    ni_a = np.array([[100], [100]], dtype=float)
    nd_a = np.array([[90], [10]], dtype=float)
    ni_e = ni_a + np.array([[40], [40]], dtype=float)
    q3 = mech_quantities(ni_e, nd_a.copy(), ni_a, nd_a)
    cf3 = two_user_closed_form(ni_e, nd_a.copy(), ni_a, nd_a)
    lb3 = ep_mech_label(q3["ku"], q3["kf"])
    out["S3"] = {"ku": q3["ku"], "Mu": q3["Mu"], "ku_net": cf3["ku_net"],
                 "ku_swap": cf3["ku_swap"], "state": lb3["u_state"],
                 "share_user": str(q3["shares"]["user"]), "dr": q3["dr"],
                 # ★ 预注册 §9 更正：首版还断言 `share` 在此处不可定义 —— 错的。
                 #   `dr=0` 只在**总尝试数不变**时成立；"净增"构造必然改 `Σni` ⇒ `dr≠0`。
                 #   本点 `E_user=0` 而 `dr≠0` ⇒ `share=0`（有限）。改成断言"份额给 0、`ku` 也给 0，
                 #   但两者语义不同"这一点不参与 pass，只记录。
                 "pass": bool(q3["Mu"] > 0 and abs(q3["ku"]) < 1e-12
                              and lb3["u_state"] == "NEU"
                              and abs(cf3["ku_net"]) < 1e-12
                              and abs(cf3["ku_swap"]) < 1e-12
                              and abs(q3["dr"]) > 1e-6)}

    # ---- S4 n_users=3 ⇒ 显式 NOT-APPLICABLE；硬套 2 用户公式必错
    ni3_a = np.array([[100], [100], [100]], dtype=float)
    nd3_a = np.array([[90], [50], [10]], dtype=float)
    ni3_e = ni3_a + np.array([[40], [-20], [-20]], dtype=float)
    cf4 = two_user_closed_form(ni3_e, nd3_a.copy(), ni3_a, nd3_a)
    q4 = mech_quantities(ni3_e, nd3_a.copy(), ni3_a, nd3_a)
    G2 = float(ni3_e[0, 0] - ni3_a[0, 0] + ni3_e[1, 0] - ni3_a[1, 0])
    D2 = float((ni3_e[0, 0] - ni3_a[0, 0]) - (ni3_e[1, 0] - ni3_a[1, 0]))
    ku_forced = (G2 * (0.9 + 0.5 - 1.0) + D2 * (0.9 - 0.5)) / max(abs(G2), abs(D2))
    out["S4"] = {"status": cf4["status"], "ku_true": q4["ku"], "ku_if_forced": ku_forced,
                 "diff": abs(ku_forced - q4["ku"]),
                 "pass": bool(cf4["status"] == "NOT-APPLICABLE"
                              and abs(ku_forced - q4["ku"]) > 1e-6)}

    # ---- S5 Mu=0 ⇒ ku 是 NaN ⇒ u_state=NA（**不是** NEU）
    ni_same = np.array([[100], [100]], dtype=float)
    nd_same = np.array([[80], [20]], dtype=float)
    q5 = mech_quantities(ni_same, nd_same, ni_same, nd_same)
    cf5 = two_user_closed_form(ni_same, nd_same, ni_same, nd_same)
    st5 = mech_state(q5["ku"])
    out["S5"] = {"Mu": q5["Mu"], "ku": str(q5["ku"]), "state": st5,
                 "closed_form_status": cf5["status"],
                 "pass": bool(q5["Mu"] == 0.0 and np.isnan(q5["ku"]) and st5 == "NA")}

    # ---- S6 活体负控：分母换回 |Δr| ⇒ ① 纯交换点变 NaN；② 只减 1 个送达就让 share **翻号且放大 40×**
    #   ★ 预注册 §9 更正：首版拿 `S3`（净增）当第二个点 —— 该点 `dr≠0` ⇒ share 是有限的 0，
    #     恒等式下"0 对 0"根本抓不到病。改用 S1 的 nd_e 扰动：`E_user` 不依赖 `nd_e` ⇒ 分子不动、分母变。
    #   ★ 预注册 §9 更正之二：`_bad` 首版用了 `|dr|`，而 R1.16 的 `share` 是 **带符号** `E/dr`
    #     ⇒ 构造出 +32 而不是 −32，"翻号"这条断言必然 FAIL（判据自伤，不是被测物的病）。
    def _bad(q):
        d = q["dr"]
        return q["E_user"] / d if d != 0 else float("nan")

    base = mech_quantities(*_swap(+1))                       # dr = 0 ⇒ share 不可定义
    ni_e1, nd_e1, ni_a1, nd_a1 = _swap(+1)
    nd_shift = np.asarray(nd_e1, dtype=float).copy()
    nd_shift[1, 0] -= 1                                      # 只减 1 个送达
    q1b = mech_quantities(ni_e1, nd_shift, ni_a1, nd_a1)
    bad1 = _bad(base)
    bad1b = _bad(q1b)
    sign_flip = bool(np.isfinite(bad1b) and np.sign(bad1b) != np.sign(base["ku"]))
    blowup = bool(np.isfinite(bad1b) and abs(bad1b) > 10 * abs(base["ku"]))
    out["S6"] = {"bad_at_dr0": str(bad1), "bad_at_dr_minus1": bad1b,
                 "ku_base": base["ku"], "dr_base": base["dr"], "dr_shift": q1b["dr"],
                 "d_ku": abs(q1b["ku"] - base["ku"]),
                 "sign_flip": sign_flip, "blowup_gt10x": blowup,
                 "state_of_bad": mech_state(bad1),
                 "pass": bool(not np.isfinite(bad1) and sign_flip and blowup
                              and abs(q1b["ku"] - base["ku"]) < 1e-12
                              and mech_state(bad1) == "NA")}

    # ---- S7 KAPPA_MIN 单元级分辨力（喂已知 k）
    lo = ep_mech_label(0.009, 0.009, KAPPA_MIN)
    hi = ep_mech_label(-0.011, -0.011, KAPPA_MIN)
    out["S7"] = {"k=0.009": lo, "k=-0.011": hi,
                 "pass": bool(lo["u_state"] == "NEU" and lo["f_state"] == "NEU"
                              and hi["u_state"] == "NEG" and hi["f_state"] == "NEG"
                              and hi["harm_class"] == "BOTH-HARMFUL")}

    keys = ("S1", "S2", "S3", "S4", "S5", "S6", "S7")
    out["all_pass"] = all(out[k]["pass"] for k in keys)
    out["_keys"] = list(keys)
    return out


# ============================================================ 主流程
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="CW-R1.18 机制表重写（零 episode）")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--ctrl-only", action="store_true", help="只跑判据自证，不写证据")
    args = ap.parse_args(argv)

    t_start = time.perf_counter()

    # ---- CTRL 先行（预注册 §7：自证不过不看面板）
    t0 = time.perf_counter()
    ctrl = controls()
    ctrl_s = round(time.perf_counter() - t0, 2)
    kk = ctrl.pop("_keys")
    print(f"[CTRL] all_pass={ctrl['all_pass']} " +
          " ".join(f"{k}={ctrl[k]['pass']}" for k in kk) + f" ({ctrl_s}s)")
    if not ctrl["all_pass"]:
        print("[ABORT] 判据自证未过 ⇒ 按预注册 §7 不看面板")
        return 3
    if args.ctrl_only:
        print(json.dumps(ctrl, ensure_ascii=False, indent=1, default=float))
        return 0

    # ---- 全量闸
    t1 = time.perf_counter()
    gate = G.run_all()
    gate_s = round(time.perf_counter() - t1, 2)
    print(f"[GATE] n_checks={gate['n_checks']} n_fail={gate['n_fail']} ({gate_s}s)")

    head = _git(["rev-parse", "HEAD"])
    wt_sha = _git(["hash-object", "experiments/cw_v1/r1_18_mechtable.py"])
    blob_sha = _git(["rev-parse", f"{head}:experiments/cw_v1/r1_18_mechtable.py"],
                    allow_fail=True)
    dirty = _git(["status", "--porcelain"])
    srchash = _src_hash()
    outdir = RESULTS / f"cw_r1_18_mechtable_{_now_utc()}_{srchash}"
    outdir.mkdir(parents=True, exist_ok=False)
    print(f"[OUT] {outdir.name}")

    cells = json.loads(R116_CELLS.read_text(encoding="utf-8"))
    if len(cells) != 80:
        print(f"[ABORT] R1.16 cells 应为 80 格，实为 {len(cells)}")
        return 4
    decomp = json.loads(R116_DECOMP.read_text(encoding="utf-8"))["10bins"]["per_ep"]

    # ---- 逐 ep：新量 + 闭式 + 新旧标签
    per_ep: dict = {}
    old_match = {}
    for ep in EPS10:
        ni_e, nd_e = ep_counts(cells, "exact", ep)
        ni_a, nd_a = ep_counts(cells, "approx", ep)
        q = mech_quantities(ni_e, nd_e, ni_a, nd_a)
        cf = two_user_closed_form(ni_e, nd_e, ni_a, nd_a)
        kb = kf_bound(ni_e, nd_e, ni_a, nd_a)
        lb = ep_mech_label(q["ku"], q["kf"])
        old = R15_classify(q)
        stored = decomp[f"ep{ep}"]["cls"]["mechanism"]
        old_match[f"ep{ep}"] = bool(old["mechanism"] == stored)
        per_ep[f"ep{ep}"] = {
            "ep": ep, "dr": q["dr"], "E_user": q["E_user"], "E_frame": q["E_frame"],
            "Mu": q["Mu"], "Mf": q["Mf"], "ku": q["ku"], "kf": q["kf"],
            "label": lb, "old_label": old["mechanism"], "old_shares": old["shares"],
            "closed_form": {k: cf.get(k) for k in
                            ("G", "D", "rho_0", "rho_1", "rho_a", "ku_net", "ku_swap",
                             "ku_closed_form", "abs_residual", "gap_abs", "bound_2gap",
                             "ratio_abs_ku_over_2gap", "bound_2gap_ok", "rho_a_is_midpoint",
                             "status")},
            "kf_bound": kb,
        }
        print(f"[EP] {ep} old={old['mechanism']:<22s} new=({lb['u_state']},{lb['f_state']}) "
              f"{lb['harm_class']:<18s} ku={q['ku']:+.4f} kf={q['kf']:+.4f} "
              f"ρ=({cf['rho_0']:.3f},{cf['rho_1']:.3f}) G={cf['G']:+.0f} D={cf['D']:+.0f} "
              f"net={cf['ku_net']:+.4f} swap={cf['ku_swap']:+.4f}")

    if not all(old_match.values()):
        bad = [k for k, v in old_match.items() if not v]
        print(f"[ABORT] 旧标签与 R1.16 入库值不一致：{bad}")
        return 5

    # ---- Q1 总判词
    kus = [v["ku"] for v in per_ep.values()]
    kfs = [v["kf"] for v in per_ep.values()]
    agg = aggregate_mech_verdict(kus, kfs)
    print(f"[Q1] {agg['verdict']} (U={agg['U']} {agg['u_n_neg']}neg/{agg['u_n_pos']}pos | "
          f"F={agg['F']} {agg['f_n_neg']}neg/{agg['f_n_pos']}pos)")

    # ---- 激活维的**区分力**（预注册 §4 的自我否证条件）
    disc = {
        "min_abs_ku": float(np.min(np.abs(kus))),
        "min_abs_kf": float(np.min(np.abs(kfs))),
        "min_abs_ku_over_kappa": float(np.min(np.abs(kus)) / KAPPA_MIN),
        "min_abs_kf_over_kappa": float(np.min(np.abs(kfs)) / KAPPA_MIN),
        "n_u_neu": int(sum(1 for v in per_ep.values() if v["label"]["u_state"] == "NEU")),
        "n_f_neu": int(sum(1 for v in per_ep.values() if v["label"]["f_state"] == "NEU")),
        "harm_class_counts": {},
    }
    for v in per_ep.values():
        c = v["label"]["harm_class"]
        disc["harm_class_counts"][c] = disc["harm_class_counts"].get(c, 0) + 1
    disc["activity_dim_discriminating"] = bool(
        disc["n_u_neu"] > 0 or disc["n_f_neu"] > 0)
    disc["note"] = ("若 `n_u_neu = n_f_neu = 0`（即 `|k|` 全部 ≫ κ），"
                    "则「激活与否」这一维**无区分力** ⇒ 新判据在逐 ep 上退化为**符号计数**，"
                    "与 R1.17 的 `Q_dir` 等价（预注册 §4 已写死此自我否证条件）。")
    print(f"[DISC] min|ku|={disc['min_abs_ku']:.4f} ({disc['min_abs_ku_over_kappa']:.1f}κ) "
          f"min|kf|={disc['min_abs_kf']:.4f} ({disc['min_abs_kf_over_kappa']:.1f}κ) "
          f"harm={disc['harm_class_counts']}")

    # ---- Q2 新旧对照 + 冲突
    rows = []
    conflicts = []
    redescriptions = []
    for ep, v in per_ep.items():
        lb, old = v["label"], v["old_label"]
        row = {"ep": v["ep"], "old": old,
               "old_share_user": v["old_shares"]["user"],
               "old_share_frame": v["old_shares"]["frame"],
               "new_u": lb["u_state"], "new_f": lb["f_state"],
               "new_harm_class": lb["harm_class"],
               "ku": v["ku"], "kf": v["kf"], "dr": v["dr"]}
        conf = []
        if old == "H-A-FRAME-SELECTION" and lb["f_state"] != "NEG":
            conf.append("CONFLICT_A")
        if old == "H-B-USER-ALLOCATION" and lb["u_state"] != "NEG":
            conf.append("CONFLICT_B")
        if old == "BOTH-MECHANISMS" and (lb["u_state"] != "NEG" or lb["f_state"] != "NEG"):
            conf.append("CONFLICT_C")
        row["conflict"] = conf
        if conf:
            conflicts.append(row)
        if old == "H-A-FRAME-SELECTION" and lb["u_state"] == "NEG":
            row["re_description"] = ("旧说帧份额主导；新说用户通道也有害"
                                     "（两者答不同问题，不是冲突）")
            redescriptions.append(row)
        rows.append(row)
        print(f"[Q2] {ep} old={old:<22s} new=({lb['u_state']},{lb['f_state']}) "
              f"{'CONFLICT:' + ','.join(conf) if conf else 'no-conflict'}")

    q2 = {"n_conflict": len(conflicts),
          "conflict_eps": [r["ep"] for r in conflicts],
          "n_re_description": len(redescriptions),
          "re_description_eps": [r["ep"] for r in redescriptions],
          "stop_rule_hit": bool(len(conflicts) >= 3),
          "explanation": ("冲突集中在 `|Δr|` 小的 ep（份额爆过的那些）时，性质是"
                          "「份额口径不可读」，不是「两个判断互相否定」。")}

    # ---- ★ 预注册 §9 更正之三（跑后发现的**判据缺陷**，不是被测物的病）：
    #   预注册 §6 的冲突判据**没有按 `sign(Δr)` 分层**。`H-A` 的语义是"帧项解释了缺口"，
    #   在 `Δr<0` 的 ep 上等价于"帧项有害"；但在 `Δr>0` 的 ep 上它的语义**翻转**成"帧项解释了改善"。
    #   于是 `ep104`（唯一 `Δr>0`）被判成 CONFLICT_A —— 那是**我的规则**造出来的假冲突。
    #   处置：两个数都报（原规则 + 按符号分层后的真冲突），不覆盖原规则的结果。
    genu = [r for r in conflicts if r["dr"] < 0]
    flip_art = [r for r in conflicts if r["dr"] > 0]
    q2["conflicts_genuine_dr_neg"] = [r["ep"] for r in genu]
    q2["conflicts_sign_flip_artifact_dr_pos"] = [r["ep"] for r in flip_art]
    q2["n_conflict_genuine"] = len(genu)
    q2["rule_defect_note"] = (
        "预注册 §6 的冲突判据未按 `sign(Δr)` 分层 ⇒ 在 `Δr>0` 的 ep 上 `H-A` 的语义是"
        "「帧项解释了改善」而不是「帧项有害」，据此产生的冲突是**规则缺陷**（判据自伤），"
        "不是两个判断互相否定。原始计数（`n_conflict`）保留不改，另报 `n_conflict_genuine`。")

    # ---- `ep104` 的正贡献项定位（STATE §7.1-9② 的开放项；纯算术，是恒等式的分解）
    _cells = json.loads(R116_CELLS.read_text(encoding="utf-8"))
    ni_e104, nd_e104 = ep_counts(_cells, "exact", 104)
    ni_a104, nd_a104 = ep_counts(_cells, "approx", 104)
    _q104 = mech_quantities(ni_e104, nd_e104, ni_a104, nd_a104)
    q2["ep104_offset"] = {
        "dr": _q104["dr"], "E_user": _q104["E_user"], "E_frame": _q104["E_frame"],
        "E_within": _q104["E_within"], "ku": _q104["ku"], "kf": _q104["kf"],
        "note": ("唯一 `Δr>0` 的 ep：用户项仍为负（`ku<0`），把 `Δr` 拉正的是"
                 "帧项与 within 项。三项之和恒等于 `Δr`（恒等式）⇒ 这里只**定位**，不新增机制。"),
    }

    # ---- `ku_net` vs `ku_swap` 的描述（预注册 §5.1：**只作描述、不给判词**）
    dom = [abs(v["closed_form"]["ku_swap"]) > abs(v["closed_form"]["ku_net"])
           for v in per_ep.values()]
    q2["ku_decomposition_desc"] = {
        "n_swap_dominates": int(sum(dom)), "n_ep": len(dom),
        "abs_ku_net_max": float(max(abs(v["closed_form"]["ku_net"]) for v in per_ep.values())),
        "abs_ku_swap_min": float(min(abs(v["closed_form"]["ku_swap"]) for v in per_ep.values())),
        "status": "DESCRIPTIVE-ONLY（预注册写死：不给判词、不设期望方向）",
    }
    print(f"[Q2+] 预注册规则冲突 {len(conflicts)} 个；按 `sign(Δr)` 分层后真冲突 "
          f"{len(genu)} 个，符号翻转假冲突 {len(flip_art)} 个 "
          f"({[r['ep'] for r in flip_art]})")
    print(f"[Q2+] ku_net vs ku_swap：swap 占优 {q2['ku_decomposition_desc']['n_swap_dominates']}/10 "
          f"(max|net|={q2['ku_decomposition_desc']['abs_ku_net_max']:.4f}, "
          f"min|swap|={q2['ku_decomposition_desc']['abs_ku_swap_min']:.4f})")

    # ---- Q3 闭式检验 F1–F4
    f1 = max(v["closed_form"]["abs_residual"] for v in per_ep.values())
    f2 = max(abs(v["closed_form"]["ku_net"] + v["closed_form"]["ku_swap"] - v["ku"])
             for v in per_ep.values())
    f3_ok = all(v["closed_form"]["bound_2gap_ok"] for v in per_ep.values())
    f3_ratio = [v["closed_form"]["ratio_abs_ku_over_2gap"] for v in per_ep.values()]
    f4_ok = all(v["kf_bound"]["bound_ok"] for v in per_ep.values())
    f4_slack = [v["kf_bound"]["slack"] for v in per_ep.values()]
    q3 = {"F1_max_abs_residual": float(f1), "F1_pass": bool(f1 < 1e-12),
          "F2_max_abs_residual": float(f2), "F2_pass": bool(f2 < 1e-12),
          "F3_bound_2gap_all_ok": bool(f3_ok),
          "F3_ratio_abs_ku_over_2gap": {"median": float(np.median(f3_ratio)),
                                        "min": float(np.min(f3_ratio)),
                                        "max": float(np.max(f3_ratio))},
          "F4_bound_2max_all_ok": bool(f4_ok),
          "F4_slack_min": float(np.min(f4_slack)),
          "F4_has_strict_slack": bool(np.min(f4_slack) > 1e-9),
          "tautology_warning": ("F1–F4 全是定义的**代数推论**（预注册 §8-1）⇒ "
                                "不是经验发现，只能当实现核对与单位解释。")}
    print(f"[Q3] F1 resid={f1:.3e} F2 resid={f2:.3e} F3 ok={f3_ok} "
          f"ratio中位={np.median(f3_ratio):.3f} F4 ok={f4_ok} slack_min={np.min(f4_slack):.3e}")

    # ---- κ 敏感性（真实数据，只报不改）
    sens = {}
    for km in (0.005, 0.01, 0.02):
        a = aggregate_mech_verdict(kus, kfs, kappa=km)
        sens[f"kappa={km}"] = {"verdict": a["verdict"], "U": a["U"], "F": a["F"],
                               "u_n_neg": a["u_n_neg"], "f_n_neg": a["f_n_neg"]}
    flip = len({v["verdict"] for v in sens.values()}) > 1
    print("[SENS] " + json.dumps({k: v["verdict"] for k, v in sens.items()},
                                 ensure_ascii=False) + f" flip={flip}")

    verdicts = {
        "Q1_aggregate": agg,
        "Q1_discriminating_power": disc,
        "Q2_old_vs_new": {"rows": rows, **q2},
        "Q3_closed_form": q3,
        "sens_kappa": sens, "sens_flip": bool(flip),
        "kappa_min": KAPPA_MIN, "sign_min": SIGN_MIN,
        "n_run_episode": 0,
        "replacement_note": (
            "`MIXED-MECHANISM-AT-N10` **保留原值不改写**，它是**份额口径**的描述；"
            "本轮给出的是**方向口径**的替代判词，两者并列、不互相替代。"),
    }

    n_fail = gate["n_fail"] + (0 if ctrl["all_pass"] else 1)
    (outdir / "analysis.json").write_text(
        json.dumps(per_ep, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    (outdir / "verdicts.json").write_text(
        json.dumps(verdicts, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    (outdir / "ctrl.json").write_text(
        json.dumps(ctrl, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    (outdir / "gates.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=1), encoding="utf-8")

    try:
        import psutil  # type: ignore
        rss = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
    except Exception:
        rss = None
    meta = {"command": "python -m experiments.cw_v1.r1_18_mechtable --tag " + str(args.tag),
            "timestamp_local": datetime.now().astimezone().isoformat(),
            "python": os.sys.version.split()[0], "numpy": np.__version__,
            "head": head, "src_hash": srchash,
            "worktree_sha_r1_18": wt_sha, "blob_sha_r1_18": blob_sha,
            "blob_sha_note": ("运行时该文件尚未提交 ⇒ blob_sha 记 None（不记参数串）；"
                              "提交后真实 blob 见 _POST_COMMIT.json"
                              if blob_sha is None else "运行前该文件已在 HEAD 上"),
            "git_dirty_before_run": dirty.splitlines(),
            "input_cells": str(R116_CELLS), "input_decomp": str(R116_DECOMP),
            "n_cells_in": len(cells), "n_run_episode": 0,
            "old_label_match_10_10": old_match,
            "ctrl_elapsed_s": ctrl_s, "gate_elapsed_s": gate_s,
            "total_elapsed_s": round(time.perf_counter() - t_start, 2),
            "peak_rss_mb": round(rss, 1) if rss else None,
            "n_checks_gates": gate["n_checks"], "n_fail_gates": gate["n_fail"]}
    (outdir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"[DONE] {outdir.name} n_fail={n_fail} total={meta['total_elapsed_s']}s "
          f"peak_rss={meta['peak_rss_mb']}MB")
    print(json.dumps({"Q1": agg["verdict"], "Q2_conflicts": len(conflicts),
                      "Q3_F1": q3["F1_pass"], "sens_flip": flip},
                     ensure_ascii=False, indent=1))
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    import sys
    sys.exit(main())
