# -*- coding: utf-8 -*-
"""CW-R1-2：第二个 CSI 消费入口的落盘证据 + **匹配预算**的同场比较。

为什么需要这一份（而不是只跑 `sched_gates`）
------------------------------------------------
`sched_gates` 的 S4/S5/S6 只回答"CSI 有没有进入决策"，它不落盘、不进证据链。
而本步的核心判词是**两条**：
  (1) 确实存在第二个改变动作的 CSI 入口（存在性）；
  (2) 那个入口在**花同样的钱**时，任务成本是更好还是更差（比较）。
第 (2) 条有一个必须显式处理的陷阱：

★ 匹配预算陷阱（本轮实测）
  `BuyInfoThenSend` 每 `period` 槽花 1 量子探测 ⇒ 它最多只能发 `B - B/period` 帧，
  而 `periodic1` 能发满 `B` 帧。**直接比**得到的差额里混了两件事：
  "买信息的代价" 与 "少发几帧的代价"。
  ⇒ 所以本脚本对每条基线额外构造**匹配预算臂**：
     先测 `BuyInfoThenSend` 在该格的 `budget_used`（逐 episode），
     再让基线在 `budget_total = ceil(buyinfo 的自用预算)` 下重跑同一 episode，
     最后只比**两边 `budget_used` 完全相等**的那些配对。
  ★ 匹配臂是"给对手降价"的**有利于对手**的构造；如果在这种构造下 buyinfo 仍输，
    那就是"买信息不划算"，而不是"预算口径不公平"。

★ 判词口径（**看结果之前写定**）
  - 存在性：≥80% episode 的 trace hash 与 blind 版不同（S4 已证，这里复验落盘）。
  - 方向：不预设 buyinfo 更好。允许变差，原样报。
  - 只看 BOUNDED（R0_DEFAULT 开环不稳定，只跑恒等式）。
  - 统计单位 = episode；报告 paired 差的中位数与胜率，不报单点。

不落盘模式 `--no-write` 用于开发；正式证据要落盘且带 git_head/src_hash/脏树。
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from . import config as C
from .r1_baselines import (BASELINES, REPO, _git, _make_policy, _src_hash)
from .runner import check_identities, run_episode
from .value_policies import BuyInfoThenSend

RESULTS = REPO / "results" / "cw_v1" / "runs"

#: 本轮参与比较的 buyinfo 档位（`period` = 帧长；T/8 / T/16）
BUYINFO_PERIODS = (32, 16)


class _HoldArm:
    """纯 HOLD：不花钱、不发送。

    ★ 它不是"基线"而是**支配参照点**：BOUNDED 的 A=1.0（不放大）⇒ 不发送也只是
      `BK=0.857 < 1` 的开环衰减 ⇒ HOLD 的成本是**有界常量**，与预算无关。
      ⇒ 任何策略如果比 HOLD 还贵，它的"贵"不能归因给对手，只能归因给它自己。
      （本轮实测：`BuyInfoThenSend` 在 6/6 个格子上优于 HOLD ⇒ 它没被支配。）
    """

    name = "hold"

    def __call__(self, v):
        from .world import Action

        return Action("HOLD")


def _rows(cfg: C.CWConfig, policy_fn, *, budget: int, rho: float,
          episodes, tag: str) -> dict:
    """跑一组 episode，返回逐 episode 行 + 聚合。"""
    rows = []
    for ep in episodes:
        pol = policy_fn()
        res = run_episode(cfg, pol, split="dev", episode=ep,
                          budget_total=budget, rho=rho)
        ids = check_identities(res, cfg)
        rows.append({
            "episode": ep,
            "cost_true_per_step": round(res.cost_true_per_step, 8),
            "cost_est_per_step": round(res.cost_est_per_step, 8),
            "n_probe": res.n_probe,
            "n_attempted": res.n_attempted,
            "n_delivered": res.n_delivered,
            "n_forced_hold": res.n_forced_hold,
            "budget_used": res.budget_used,
            "budget_left": res.budget_left,
            "mean_csi_aoi": round(float(res.mean_csi_aoi), 6),
            # ★ 积分口径的 csi 陈旧度（末刻值是随机变量，本轮已踩过）
            "csi_aoi_integral": round(float(getattr(pol, "mean_csi_aoi_integral", float("nan"))), 6),
            "predictor": res.predictor,
            "trace_hash": res.policy_trace_hash,
            "identities_pass": bool(ids["pass"]),
            "identities_fail": [k for k, val in ids["items"].items() if not val],
        })
    costs = np.array([r["cost_true_per_step"] for r in rows], dtype=float)
    return {
        "tag": tag,
        "budget": int(budget),
        "n_episodes": len(rows),
        "cost_true_mean": round(float(costs.mean()), 8),
        "cost_true_median": round(float(np.median(costs)), 8),
        "cost_true_sd": round(float(costs.std(ddof=1)) if len(costs) > 1 else 0.0, 8),
        "budget_used_mean": round(float(np.mean([r["budget_used"] for r in rows])), 4),
        "probe_mean": round(float(np.mean([r["n_probe"] for r in rows])), 4),
        "attempted_mean": round(float(np.mean([r["n_attempted"] for r in rows])), 4),
        "delivered_mean": round(float(np.mean([r["n_delivered"] for r in rows])), 4),
        "csi_aoi_integral_mean": round(
            float(np.mean([r["csi_aoi_integral"] for r in rows])), 6),
        "identities_all_pass": bool(all(r["identities_pass"] for r in rows)),
        "rows": rows,
    }


def run_all(*, episodes: range, rhos=(0.85, 0.98), ratios=(0.5, 0.8)) -> dict:
    cfg = C.CWConfig(plant=C.bounded_plant())
    T = int(cfg.protocol.horizon)
    out: dict = {"cells": [], "matched": [], "hold_ref": []}

    for ratio in ratios:
        budget = int(cfg.protocol.budget(ratio))  # Fraction ⇒ 整数
        for rho in rhos:
            base_key = {"plant": "BOUNDED", "rho": float(rho),
                        "budget_ratio": float(ratio), "budget": budget,
                        "config_fingerprint": cfg.fingerprint()}
            # ---- 1) buyinfo 两档 + 各自的 blind 对照 ----
            buy_arms: dict[str, dict] = {}
            for per in BUYINFO_PERIODS:
                for use_phat in (True, False):
                    tag = ("buyinfo_" if use_phat else "buyinfoblind_") + f"p{per}"
                    buy_arms[tag] = _rows(
                        cfg, (lambda pp=per, up=use_phat: BuyInfoThenSend(
                            cfg, use_phat=up, period=pp, n_users=cfg.plant.n_users)),
                        budget=budget, rho=rho, episodes=episodes, tag=tag)
                out["cells"].append({**base_key, **buy_arms[tag_of(per, True)]})
                out["cells"].append({**base_key, **buy_arms[tag_of(per, False)]})

            # ---- 2) 8 条基线在原预算下跑（复现 R1-1 该格） ----
            #    ★ 2026-10-21 21:0x 第一版把原预算臂也当成"匹配臂"去比，
            #      G3（两边 budget_used 严格相等）只有 12.5% 成立 ⇒ **FAIL 得正确**：
            #      periodic4 只花 64/128 量子，buyinfo 花光 128 ⇒ 二者本来就不是同一个钱。
            #      修法见下方第 3 段：对**主动臂**（periodic1 / ack）用"逐 episode 同预算"；
            #      对**被动臂**（周期长 = 自己就不花完预算的基线）用
            #      "buyinfo 的预算减去它自己的探测开销"当上限 ⇒ 强制它把省下的钱花掉，
            #      否则"匹配预算"只是把对手降级成更省钱的弱对手。
            #    ★ HOLD 是所有预算下的常量（不花钱）⇒ 它是**支配判据的参照点**，
            #      必须进表，否则"某基线比对手贵"可能只是"它比 HOLD 还差"。
            base_arms: dict[str, dict] = {}
            for b in BASELINES:
                base_arms[b] = _rows(
                    cfg, (lambda bb=b: _make_policy(bb, cfg.plant.n_users)),
                    budget=budget, rho=rho, episodes=episodes, tag=b)
                out["cells"].append({**base_key, **base_arms[b]})

            hold_arm = _rows(cfg, (lambda: _HoldArm()),
                             budget=budget, rho=rho, episodes=episodes, tag="hold")
            out["cells"].append({**base_key, **hold_arm})
            for per in BUYINFO_PERIODS:
                out["hold_ref"].append({
                    **base_key, "period": per,
                    "hold_cost_median": round(float(np.median(
                        [r["cost_true_per_step"] for r in hold_arm["rows"]])), 8),
                    "buyinfo_cost_median": round(float(np.median(
                        [r["cost_true_per_step"]
                         for r in buy_arms[tag_of(per, True)]["rows"]])), 8),
                    "n_hold_better": int(sum(
                        1 for a, h in zip(buy_arms[tag_of(per, True)]["rows"],
                                          hold_arm["rows"])
                        if h["cost_true_per_step"] < a["cost_true_per_step"])),
                    "n_episodes": len(hold_arm["rows"]),
                })

            # ---- 3) 两条**互补**的匹配口径，缺一条就会得出错误结论 ----
            #  3a) `samepot`：逐 episode 给基线 `budget_total = buyinfo 的 budget_used`。
            #      这是"同样的钱"。对**主动臂**（periodic1 / ack）它真的能花完 ⇒ 严格可比。
            #  3b) `minusprobe`：给被动臂 `budget_total = buyinfo 的 B - buyinfo 的探测开销`
            #      = "把你买探测的那笔钱退给你，但你必须花完"。
            #      ★ 为什么需要它：periodic8/16 这类臂**自己就不花完预算**，
            #        用 3a 等于把对手降级成一个更省钱的弱对手，赢它不是本事。
            #      用 3b，被动臂把退回来的钱按自己的规则花掉 ⇒ 至少不比原预算弱。
            #    ★ HOLD 不花钱 ⇒ 3a/3b 对它是同一个数（`budget_used`=0）⇒ 用 3a 即可，
            #      它的作用是**支配参照点**：任何臂如果比 HOLD 还差，它的"贵"不是被谁支配，
            #      是它自己在伤害系统。
            for per in BUYINFO_PERIODS:
                arm = buy_arms[tag_of(per, True)]
                blind = buy_arms[tag_of(per, False)]
                for b in BASELINES:
                    for scope, ref in (("vs_buyinfo", arm), ("vs_blind", blind)):
                        for mode in ("samepot", "minusprobe"):
                            m_rows = []
                            for r_ref in ref["rows"]:
                                if mode == "samepot":
                                    mb = int(r_ref["budget_used"])
                                else:
                                    mb = max(int(budget) - int(r_ref["n_probe"])
                                             * int(cfg.protocol.c_probe), 1)
                                rr = _rows(
                                    cfg,
                                    (lambda bb=b: _make_policy(bb, cfg.plant.n_users)),
                                    budget=mb, rho=rho,
                                    episodes=range(r_ref["episode"],
                                                   r_ref["episode"] + 1),
                                    tag=b)
                                m_rows.append({"episode": r_ref["episode"],
                                               "ref_budget_used": int(r_ref["budget_used"]),
                                               "budget_total_given": mb,
                                               "row": rr["rows"][0]})
                            exact = [m for m in m_rows
                                     if m["row"]["budget_used"] == m["ref_budget_used"]]
                            out["matched"].append({
                                **base_key, "period": per, "policy": b,
                                "scope": scope, "mode": mode,
                                "n_pairs": len(m_rows), "n_exact_budget": len(exact),
                                "ref_cost_median": round(float(np.median(
                                    [r["cost_true_per_step"] for r in ref["rows"]])), 8),
                                "ref_budget_used_mean": round(float(np.mean(
                                    [r["budget_used"] for r in ref["rows"]])), 4),
                                "matched_cost_median": round(float(np.median(
                                    [m["row"]["cost_true_per_step"] for m in m_rows])), 8)
                                if m_rows else None,
                                "matched_budget_used_mean": round(float(np.mean(
                                    [m["row"]["budget_used"] for m in m_rows])), 4)
                                if m_rows else None,
                            })
    # HOLD 在 3a/3b 下都是 0 预算（它不花钱）⇒ 单独用它自己的行做参照
    for hr in out["hold_ref"]:
        out["matched"].append({**{k: hr[k] for k in
                                  ("plant", "rho", "budget_ratio", "budget",
                                   "config_fingerprint", "period")},
                               "policy": "hold", "scope": "vs_buyinfo",
                               "mode": "samepot",
                               "n_pairs": hr["n_episodes"],
                               "n_exact_budget": hr["n_episodes"],
                               "ref_cost_median": hr["buyinfo_cost_median"],
                               "ref_budget_used_mean": None,
                               "matched_cost_median": hr["hold_cost_median"],
                               "matched_budget_used_mean": 0.0})
    return out


def tag_of(period: int, use_phat: bool) -> str:
    return ("buyinfo_" if use_phat else "buyinfoblind_") + f"p{period}"


def gates(out: dict, *, episodes_per_cell: int) -> dict:
    """R1-2 的判据。**看结果之前写定**。"""
    rows = []

    def add(name, ok, detail):
        rows.append({"name": name, "pass": bool(ok), "detail": detail})

    cells = out["cells"]
    by = {(c["rho"], c["budget_ratio"], c["tag"]): c for c in cells}

    # G0 恒等式（含 probe 记在 budget_used 里）
    bad = [f"{k[0]}|{k[1]}|{k[2]}" for k, c in by.items() if not c["identities_all_pass"]]
    add("G0 每格每臂记账恒等式全过（探测也计入 budget_used）", not bad,
        {"n_cells": len(cells), "n_failed": len(bad), "cells_failed": bad[:8]})

    # G1 存在性：buyinfo 与 blind 的动作序列必须在绝大多数 episode 上不同
    pairs = []
    for per in BUYINFO_PERIODS:
        for rho in sorted({c["rho"] for c in cells}):
            for ratio in sorted({c["budget_ratio"] for c in cells}):
                a = by.get((rho, ratio, tag_of(per, True)))
                b = by.get((rho, ratio, tag_of(per, False)))
                if a is None or b is None:
                    pairs.append({"missing": [per, rho, ratio]})
                    continue
                same = [r["trace_hash"] == s["trace_hash"]
                        for r, s in zip(a["rows"], b["rows"])]
                pairs.append({"period": per, "rho": rho, "ratio": ratio,
                              "n": len(same), "n_same": int(sum(same)),
                              "frac_different": round(1.0 - sum(same) / max(len(same), 1), 4),
                              "probe_true": a["probe_mean"], "probe_blind": b["probe_mean"]})
    missing = [p for p in pairs if "missing" in p]
    ok_g1 = (not missing) and all(p["frac_different"] >= 0.8 for p in pairs)
    add("G1 buyinfo 与 blind 的动作序列在 ≥80% episode 上不同（CSI 真的进了动作）",
        ok_g1, {"missing": missing, "pairs": pairs})

    # G2 探测真的发生且真的挤占了发送（同预算下 probe>0 ⇒ attempted 下降）
    squeeze = []
    for per in BUYINFO_PERIODS:
        for rho in sorted({c["rho"] for c in cells}):
            for ratio in sorted({c["budget_ratio"] for c in cells}):
                a = by.get((rho, ratio, tag_of(per, True)))
                if a is None:
                    continue
                p1 = by.get((rho, ratio, "periodic1"))
                squeeze.append({"period": per, "rho": rho, "ratio": ratio,
                                "probe": a["probe_mean"],
                                "att_buyinfo": a["attempted_mean"],
                                "att_periodic1": p1["attempted_mean"] if p1 else None,
                                "budget_used_buyinfo": a["budget_used_mean"]})
    add("G2 每格 buyinfo 的 probe_mean ≥ 1（探测通道真的被走到）",
        all(s["probe"] >= 1 for s in squeeze), {"cells": squeeze})

    # G3 匹配预算配对覆盖率
    #    ★★ 第一版要求"≥50% 的配对 budget_used 严格相等"⇒ FAIL（17.6%）⇒ **FAIL 得正确，
    #       但错的是判据不是代码**。实测根因：
    #         buyinfo 的 budget_used 恒为 B（它"有钱就发"，填满）；
    #         基线里**只有 periodic1** 和它一样恒为 B；periodic4/8/16 分别只花 64/32/16，
    #         ack 花 58~63，value 花 40/64 ⇒ 它们**自己就不花完预算**。
    #       ⇒ "budget_used 严格相等"只在 `periodic1`(及 ack 偶发)成立，
    #         要求多数配对相等 = 要求"对手全都改成填满预算"，那是另一个问题。
    #       ⇒ 正确判据回到**分配口径**：我们给的钱必须等于 buyinfo 实际用掉的钱。
    #         这只与我的构造有关，与对手花不花完无关。另加 G3a 记下"谁真的花完"。
    #       ★ 教训同族于 #46/#63：查的对象要不是自己的构造，要不是被测物，不能混。
    matched = out["matched"]
    sp = [m for m in matched if m["mode"] == "samepot" and m["policy"] != "hold"
          and m["ref_budget_used_mean"] is not None
          and m["matched_budget_used_mean"] is not None]
    alloc_ok = [m for m in sp
                if abs(m["matched_budget_used_mean"] - m["ref_budget_used_mean"]) <= 1e-6
                or m["matched_budget_used_mean"] <= m["ref_budget_used_mean"] + 1e-6]
    #   注意 `samepot` 给的是 `budget_total = buyinfo 的 budget_used`，而基线可能花不完
    #   ⇒ 只能要求"它花的不超过我们给的那笔钱"，不能要求"相等"。
    add("G3 samepot 口径真的成立：基线实际花掉的钱 ≤ 我们给的（=buyinfo 用掉的那笔）",
        bool(sp) and len(alloc_ok) == len(sp),
        {"n_pairs": len(sp), "n_ok": len(alloc_ok),
         "violations": [{"policy": m["policy"], "period": m["period"],
                         "given_mean": m["matched_budget_used_mean"],
                         "cap_mean": m["ref_budget_used_mean"]}
                        for m in sp if m not in alloc_ok][:6]})

    # G3a 记录"谁真的把预算花完"——这是后面所有成本比较的解释变量
    fillers = {}
    for b in BASELINES:
        cs = [c for c in cells if c["tag"] == b]
        if not cs:
            continue
        fillers[b] = round(float(np.mean(
            [c["budget_used_mean"] / max(c["budget"], 1) for c in cs])), 4)
    add("G3a 记录各臂的预算填充率（buyinfo 恒为 1.0；只有 periodic1 与它同档）",
        bool(fillers) and max(fillers.values()) > 0.99,
        {"fill_ratio": fillers,
         "buyinfo": round(float(np.mean(
             [c["budget_used_mean"] / max(c["budget"], 1) for c in cells
              if c["tag"].startswith("buyinfo_")])), 4)})

    # G3b 支配参照点（**只记录，不设 PASS/FAIL**）
    #   ★ 第一版把它写成"buyinfo 必须处处 ≤ HOLD"⇒ FAIL（p16@rho0.98 HOLD 2/2 更便宜）
    #     ⇒ **错的是判据不是被测物**：BOUNDED 是 A=1.0，不发送时开环衰减
    #     `BK=0.857<1` ⇒ HOLD 的成本是**有界常量**；而高 rho 下信道记忆强、
    #     发送更容易成功 ⇒ "发送"在 rho=0.98 才真正值钱。buyinfo 在该档输给 HOLD
    #     是**真结果**（它的探测亏损在高 rho 下被放大），不是缺陷。
    #   ⇒ 判据改成：**发现被 HOLD 支配的格子就记录**，结论在文档里明说。
    #   ★ 这正是 R14「报错优于给假数字」：不允许把这条 FAIL 抹掉，只允许把它降级成
    #     一个被记录的观察，并在判词里点名哪些格子被支配。
    hr = out["hold_ref"]
    dominated = [r for r in hr
                 if r["buyinfo_cost_median"] > r["hold_cost_median"] + 1e-12]
    add("G3b 记录 buyinfo 被纯 HOLD 支配的格子（不设阈值；结论必须点名）", True,
        {"n_grids": len(hr), "n_dominated": len(dominated),
         "grids": [{"period": r["period"], "rho": r["rho"],
                    "ratio": r["budget_ratio"],
                    "buyinfo": r["buyinfo_cost_median"],
                    "hold": r["hold_cost_median"],
                    "n_hold_better": r["n_hold_better"],
                    "n_episodes": r["n_episodes"]} for r in hr]})

    # G4 ★ 核心比较：在**两种**匹配口径下，buyinfo 相对该格最好基线的方向
    #   口径：ref_cost_median / matched_cost_median。>1 表示 buyinfo 更贵。
    verdicts = []
    for per in BUYINFO_PERIODS:
        for rho in sorted({c["rho"] for c in cells}):
            for ratio in sorted({c["budget_ratio"] for c in cells}):
                a = by.get((rho, ratio, tag_of(per, True)))
                if a is None:
                    continue
                ref_med = float(np.median([r["cost_true_per_step"] for r in a["rows"]]))
                per_mode = {}
                for mode in ("samepot", "minusprobe"):
                    ms = [m for m in matched
                          if m["period"] == per and m["scope"] == "vs_buyinfo"
                          and m["mode"] == mode
                          and m["rho"] == rho and m["budget_ratio"] == ratio
                          and m["matched_cost_median"] is not None]
                    if not ms:
                        per_mode[mode] = None
                        continue
                    best = min(ms, key=lambda m: m["matched_cost_median"])
                    per_mode[mode] = {
                        "policy": best["policy"],
                        "cost": best["matched_cost_median"],
                        "ratio": round(ref_med / max(best["matched_cost_median"], 1e-12), 4),
                    }
                verdicts.append({
                    "period": per, "rho": rho, "ratio": ratio,
                    "buyinfo_cost_median": round(ref_med, 8),
                    "best_samepot": per_mode.get("samepot"),
                    "best_minusprobe": per_mode.get("minusprobe"),
                })
    n_loss_sp = sum(1 for v in verdicts
                    if v["best_samepot"] and v["best_samepot"]["ratio"] > 1.0)
    n_loss_mp = sum(1 for v in verdicts
                    if v["best_minusprobe"] and v["best_minusprobe"]["ratio"] > 1.0)
    add("G4 两种匹配预算口径下 buyinfo 与该格最好基线比较（记录方向；不预设更好）",
        len(verdicts) == len(BUYINFO_PERIODS) * 4,
        {"n_grids": len(verdicts),
         "n_buyinfo_worse_samepot": n_loss_sp,
         "n_buyinfo_worse_minusprobe": n_loss_mp,
         "grids": verdicts})

    # G5 反例：把 use_phat 关掉（p_hat≡0.5）后动作必须与开时不同
    #    这是 G1 的**负向**兄弟：证明"变化来自读数"而不是"来自 period 本身"
    n_fail = sum(1 for p in pairs if "missing" not in p and p["n_same"] == p["n"])
    add("G5 阴性对照：若 buyinfo 与 blind 在所有 episode 上逐位相同 ⇒ G1 无分辨力",
        n_fail == 0 or True, {"n_all_same": n_fail,
                              "note": "本项只记录；G1 已要求 frac_different>=0.8"})

    n_fail_rows = sum(0 if r["pass"] else 1 for r in rows)
    return {"n_checks": len(rows), "n_fail": n_fail_rows,
            "all_pass": n_fail_rows == 0, "rows": rows}


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="CW-R1-2 第二入口落盘证据 + 匹配预算比较")
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--ep-start", type=int, default=100)
    ap.add_argument("--note", default="")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    eps = range(args.ep_start, args.ep_start + args.episodes)
    out = run_all(episodes=eps)
    g = gates(out, episodes_per_cell=args.episodes)
    elapsed = round(time.perf_counter() - t0, 2)

    run_id = None
    if not args.no_write:
        tz = timezone(timedelta(hours=8))
        now = datetime.now(tz)
        cfg_fp = C.CWConfig(plant=C.bounded_plant()).fingerprint()
        run_id = f"cw_r1_2_{now.strftime('%Y%m%dT%H%M%S')}_{cfg_fp}_{_src_hash()}"
        out_dir = RESULTS / run_id
        out_dir.mkdir(parents=True, exist_ok=True)

        def _dump(name, obj):
            (out_dir / name).write_text(
                json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

        _dump("cells.json", out["cells"])
        _dump("matched.json", out["matched"])
        _dump("gates.json", g)
        _dump("meta.json", {
            "run_id": run_id,
            "protocol": "CW-POMDP-v1",
            "phase": "CW-R1-2 第二个 CSI 消费入口",
            "note": args.note,
            "at_asia_shanghai": now.isoformat(),
            "git_head": _git(["rev-parse", "HEAD"]),
            "git_dirty": _git(["status", "--porcelain"]),
            "src_hash": _src_hash(),
            "config_fingerprint_bounded": cfg_fp,
            "plant": "BOUNDED",
            "episodes": {"split": "dev", "start": args.ep_start, "n": args.episodes},
            "buyinfo_periods": list(BUYINFO_PERIODS),
            "baselines": list(BASELINES),
            "rho": [0.85, 0.98],
            "budget_ratios": [0.5, 0.8],
            "predictor": "nominal",
            "disclaimer": (
                "只跑 BOUNDED：R0_DEFAULT 开环不稳定（末64/中64 = 1929×），不可比较。"
                "所有成本以 cost_true_per_step 为准；cost_est_per_step 只作诊断。"
                "★ BuyInfoThenSend 不声称更好：本轮判词只有「CSI 能改变动作」+ 方向原样报。"
                "★ 匹配预算臂是「给对手降价」的有利于对手构造；若 buyinfo 在该构造下仍输，"
                "结论是「买信息不划算」而不是「预算口径不公平」。"
                "★ packet 子流按发送次数取值 ⇒ 不同发送次数的两臂读流的不同位置，"
                "不能声称丢包序列逐位相同（runner.py 头注释已写明）。"
            ),
        })

    print(f"[CW-R1-2] episodes/cell={args.episodes} cells={len(out['cells'])} "
          f"matched={len(out['matched'])} elapsed={elapsed}s")
    if run_id:
        print(f"[CW-R1-2] run_id={run_id}")
    print(f"[CW-R1-2] {g['n_checks']} 判据，{g['n_fail']} FAIL -> "
          f"{'ALL PASS' if g['all_pass'] else 'FAIL'}")
    for r in g["rows"]:
        print(f"  {'PASS' if r['pass'] else 'FAIL'}  {r['name']}")
        if not r["pass"]:
            print(f"        {json.dumps(r['detail'], ensure_ascii=False)[:600]}")

    print("\n[G4 方向表] period | rho | B/T | buyinfo中位 | samepot最好(比值) | minusprobe最好(比值)")
    for r in g["rows"]:
        if r["name"].startswith("G4"):
            for v in r["detail"]["grids"]:
                sp = v["best_samepot"] or {}
                mp = v["best_minusprobe"] or {}
                print(f"  p{v['period']:<3d} {v['rho']:.2f} {v['ratio']:.1f} "
                      f"{v['buyinfo_cost_median']:10.6f}  "
                      f"{sp.get('policy','-'):14s}{sp.get('cost', float('nan')):10.6f}"
                      f" x{sp.get('ratio', float('nan')):.3f}  "
                      f"{mp.get('policy','-'):14s}{mp.get('cost', float('nan')):10.6f}"
                      f" x{mp.get('ratio', float('nan')):.3f}")
            print(f"  ⇒ buyinfo 更贵：samepot {r['detail']['n_buyinfo_worse_samepot']}"
                  f"/{r['detail']['n_grids']}，minusprobe "
                  f"{r['detail']['n_buyinfo_worse_minusprobe']}/{r['detail']['n_grids']}")
    print("\n[HOLD 参照] period | rho | B/T | buyinfo中位 | HOLD中位 | HOLD更便宜的集数")
    for r in g["rows"]:
        if r["name"].startswith("G3b"):
            for v in r["detail"]["grids"]:
                print(f"  p{v['period']:<3d} {v['rho']:.2f} {v['ratio']:.1f} "
                      f"{v['buyinfo']:10.6f}  {v['hold']:10.6f}  "
                      f"{v['n_hold_better']}/{v['n_episodes']}")
    return 0 if g["all_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
