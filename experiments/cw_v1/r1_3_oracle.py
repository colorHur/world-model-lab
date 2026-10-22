# -*- coding: utf-8 -*-
"""CW-R1-3：`ORACLE-KNOWN` **真值辅助贪心诊断臂**的落盘入口（R1 第 2 步第 1 项）。

★★ 2026-10-22 更正（v4.1 验收 §2）——三件事被撤回，历史记录保留：
  1. **撤回"上界臂"这个名称**。它是**单步真值贪心**，不是最优策略，也不是任何上界：
     它按真值挑"现在最值得发的用户"，但不做前瞻、不评估"现在测一下信道值不值"、
     不考虑剩余预算的续接价值 ⇒ **弱于**最优，因此"它赢不了基线"**不能**推出
     "信息没有价值"。名称与结论一起撤回。
  2. **撤回 `SPACE-VOID` 作为研究结论**。旧判据的门槛是"≥6/8 格"，而本设计只有
     4 格（2 ρ × 2 B/T）⇒ **判据不可满足**（见 `evaluate.required_cells_from_design`：
     格数要求必须从设计推导，不许外部写死）。旧代码还按"该格最好基线"用
     `min(cost_ratio_median)` 选基线，而**比值方向写反了**
     （存的是 `baseline/oracle`，门槛却按 `oracle/baseline < 0.9` 判）——
     于是"选出的最好基线"其实是**赢它最多的**基线，判词必然落在"不过"。
     这个错误已由 `experiments/cw_v1/evaluate.py`（CW-EVAL-v1.1）+ `eval_gates.py` 14 项反例接管。
  3. **本臂从未探测**（`probe_mean = 0.0`，实测）。它拿到真状态后没有理由花钱测信道 ⇒
     它对"测量的价值"**没有信息量**。要回答"测量值不值"，需要的是
     **同信息 + 已知生成参数的有限视界规划参考**（含 PROBE/SEND/HOLD 与续接价值），
     那是纠错后的下一步，不是这个臂。

本入口现在只负责：**记录**这个诊断臂的数值与记账恒等式，并按**逐基线**口径报告
"它相对每条基线是否更好"，**不再给全局结论词**。

比较方向（CW-EVAL-v1.1 固定，不可反）
--------------------------------------
`ratio = candidate / baseline`，`delta = baseline - candidate`。
候选便宜 ⇒ `ratio < 1` 且 `delta > 0`。

★★ 本入口**只跑 BOUNDED**（`R0_DEFAULT` 开环不稳定，末64/中64=1929×，不可比较）。
★★ `ORACLE-KNOWN` 是**故意越界的诊断臂**（读真信道 + 真状态模），任何引用必须带这句限定。

不做的三件事（写下来防止下一轮"顺手"做了）
--------------------------------------------
1. 不扫 `kappa_scale`（那会变成"看到结果再调参"）；只在 §敏感性 里跑 0.25/0.5/1.0 各一遍，
   并**只用于说明结论对 κ 的一阶不敏感**，不进预注册判据。
2. 不把 `OracleKnownProbeable` 的 `n_probe=0` 当作"探测通道已关闭"的证据（设计使然）。
3. 不用本臂的输赢声称"学习方法行或不行"，也不用它声称"信息有无价值"。
"""


from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from . import config as C
from .oracle_known import OracleKnown, OracleKnownProbeable
from .r1_baselines import BASELINES, _make_policy
from .runner import check_identities, run_episode
from .world import Action

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results" / "cw_v1"
RHOS = (0.85, 0.98)
RATIOS = (0.5, 0.8)
#: 预注册门槛（**改它就是改判据**，必须在文档里留痕）
WIN_MEDIAN_RATIO = 0.9
WIN_FRACTION = 0.7
#: ★ v1.1：删掉 `WIN_MIN_CELLS = 6`。它在 4 格设计上**不可满足**，
#:   是"判据写死、与设计脱钩"的典型（教训：格数要求必须从设计推导）。
#:   现在由 `evaluate.required_cells_from_design(cells)` 给。


# ---------------------------------------------------------------- 工具
def _src_hash() -> str:
    h = hashlib.sha256()
    for p in sorted((REPO / "experiments" / "cw_v1").glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def _git(args: list[str]) -> str:
    import subprocess
    return subprocess.run(["D:\\devtools\\Git\\cmd\\git.exe", *args],
                          cwd=str(REPO), capture_output=True, text=True).stdout.strip()


def _git_dirty_self() -> str:
    """只报与实验相关路径的脏状态（本仓里有大量非实验临时文件）。"""
    out = _git(["status", "--porcelain", "--", "experiments/cw_v1"])
    return out


class _AuditWorld:
    """`run_episode` 的 world 代理：在**每个槽开始前**把真值句柄交给策略。

    为什么需要它：`run_episode` 没有"构造 world 的回调"参数；而 `OracleKnown` 需要真值。
    做法是包装 `CWWorld`，在 `step()` 之前调用 `policy.attach_truth(self)`。
    ★ 代理**不转发** `_x_true/_log_h_true` 给策略以外的人；策略拿到的就是 world 本身，
      与 `allow_truth_keys=True` 的白名单语义一致（本臂是上界臂）。
    """

    def __init__(self, w, policy) -> None:
        self._w = w
        self._policy = policy
        self.attach_calls = 0

    def __getattr__(self, name):
        return getattr(self._w, name)

    def attach(self) -> None:
        if hasattr(self._policy, "attach_truth"):
            self._policy.attach_truth(self._w)
            self.attach_calls += 1


def _run_one(cfg, policy_factory, *, ep, budget, rho, needs_truth=False):
    """跑一集；`needs_truth=True` 时用 `_AuditWorld` 代理注入真值。

    ★ 走 `runner.run_episode` 的**同一条**主循环（不复制），
      通过临时替换 `runner.CWWorld` 注入代理 —— 副作用只在本进程内，
      且用一个显式快照/还原包住（异常也不漏）。
    """
    from . import runner as R

    pol = policy_factory()
    if not needs_truth:
        res = R.run_episode(cfg, pol, split="dev", episode=ep,
                            budget_total=budget, rho=rho)
        return res, pol, None

    orig = R.CWWorld
    holder = {}

    class _W(orig):  # type: ignore[misc,valid-type]
        def __init__(self, *a, **kw):
            kw["allow_truth_keys"] = True
            super().__init__(*a, **kw)
            holder["proxy"] = _AuditWorld(self, pol)
            if hasattr(pol, "attach_truth"):
                pol.attach_truth(self)

        def step(self, act, **kw):
            holder["proxy"].attach()      # 每槽前刷新句柄（同一对象，但保证语义显式）
            return super().step(act, **kw)

    R.CWWorld = _W
    try:
        res = R.run_episode(cfg, pol, split="dev", episode=ep,
                            budget_total=budget, rho=rho, allow_truth_keys=True)
    finally:
        R.CWWorld = orig
    return res, pol, holder.get("proxy")


def _rows(cfg, factory, *, budget, rho, episodes, tag, needs_truth=False) -> dict:
    rows = []
    for ep in episodes:
        res, pol, proxy = _run_one(cfg, factory, ep=ep, budget=budget, rho=rho,
                                   needs_truth=needs_truth)
        ids = check_identities(res, cfg)
        rows.append({
            "episode": int(ep),
            "cost_true_per_step": round(res.cost_true_per_step, 8),
            "n_probe": res.n_probe,
            "n_attempted": res.n_attempted,
            "n_delivered": res.n_delivered,
            "n_held": res.n_idle,
            "n_forced_hold": res.n_forced_hold,
            "budget_used": res.budget_used,
            "budget_left": res.budget_left,
            "policy_trace_hash": res.policy_trace_hash,
            "identities_pass": bool(ids["pass"]),
            "identities_fail": [k for k, v in ids["items"].items() if not v],
            "attach_calls": (proxy.attach_calls if proxy is not None else None),
        })
    costs = np.array([r["cost_true_per_step"] for r in rows], dtype=float)
    return {
        "tag": tag, "budget": int(budget), "rho": float(rho),
        "n_episodes": len(rows),
        "cost_true_mean": round(float(costs.mean()), 8),
        "cost_true_median": round(float(np.median(costs)), 8),
        "cost_true_sd": round(float(costs.std(ddof=1)) if len(costs) > 1 else 0.0, 8),
        "budget_used_mean": round(float(np.mean([r["budget_used"] for r in rows])), 4),
        "probe_mean": round(float(np.mean([r["n_probe"] for r in rows])), 4),
        "attempted_mean": round(float(np.mean([r["n_attempted"] for r in rows])), 4),
        "delivered_mean": round(float(np.mean([r["n_delivered"] for r in rows])), 4),
        "identities_all_pass": bool(all(r["identities_pass"] for r in rows)),
        "rows": rows,
    }


class _HoldArm:
    name = "hold"

    def __call__(self, v):
        return Action("HOLD")


# ---------------------------------------------------------------- 主流程
def run_all(*, episodes: range, rhos=RHOS, ratios=RATIOS,
            include_sensitivity: bool = True) -> dict:
    cfg = C.CWConfig(plant=C.bounded_plant())
    T = int(cfg.protocol.horizon)
    out = {"cells": [], "matched": [], "oracle": [], "sensitivity": []}

    for ratio in ratios:
        budget = int(cfg.protocol.budget(ratio))
        for rho in rhos:
            key = {"plant": "BOUNDED", "rho": float(rho), "budget_ratio": float(ratio),
                   "budget": budget, "config_fingerprint": cfg.fingerprint()}

            # ---- 1) 上界臂（主）+ 两个阴性对照变体 ----
            arms = {}
            for var, (th, tx) in (("main", (True, True)),
                                  ("no_h", (False, True)),
                                  ("no_x", (True, False))):
                arms[var] = _rows(
                    cfg,
                    (lambda a=th, b=tx: OracleKnown(cfg, use_true_h=a, use_true_x=b,
                                                    n_users=cfg.plant.n_users)),
                    budget=budget, rho=rho, episodes=episodes,
                    tag=f"oracle_known[{var}]", needs_truth=True)
                out["oracle"].append({**key, "variant": var, "budget_ratio": float(ratio),
                                      **{k: v for k, v in arms[var].items() if k != "rows"},
                                      "rows": arms[var]["rows"]})
                out["cells"].append({**key, **arms[var]})

            # ---- 1b) 可探测变体（证明"不探测"是选择而不是缺失） ----
            pb = _rows(cfg, (lambda: OracleKnownProbeable(cfg, n_users=cfg.plant.n_users)),
                       budget=budget, rho=rho, episodes=episodes,
                       tag="oracle_probeable", needs_truth=True)
            out["cells"].append({**key, **pb})

            # ---- 2) 8 基线 + HOLD（原预算，用于 samepot 匹配的参照） ----
            base_rows, hold_rows = {}, {}
            for b in BASELINES:
                base_rows[b] = _rows(
                    cfg, (lambda bb=b: _make_policy(bb, cfg.plant.n_users)),
                    budget=budget, rho=rho, episodes=episodes, tag=b)
                out["cells"].append({**key, **base_rows[b]})
            hold_rows = _rows(cfg, (lambda: _HoldArm()), budget=budget, rho=rho,
                              episodes=episodes, tag="hold")
            out["cells"].append({**key, **hold_rows})

            # ---- 3) samepot 匹配：给每个基线"上界臂实际花掉的钱"，逐 episode ----
            main = arms["main"]
            per_base = {}
            for b in BASELINES:
                m_rows = []
                for r_ref in main["rows"]:
                    mb = int(r_ref["budget_used"])
                    if mb < 1:
                        m_rows.append({"episode": r_ref["episode"],
                                       "budget_total_given": 0, "row": None})
                        continue
                    rr = _rows(cfg, (lambda bb=b: _make_policy(bb, cfg.plant.n_users)),
                               budget=mb, rho=rho,
                               episodes=range(r_ref["episode"], r_ref["episode"] + 1),
                               tag=b)
                    m_rows.append({"episode": r_ref["episode"],
                                   "ref_budget_used": mb,
                                   "budget_total_given": mb,
                                   "row": rr["rows"][0]})
                per_base[b] = m_rows
                valid = [m for m in m_rows if m["row"] is not None]
                out["matched"].append({
                    **key, "policy": b, "mode": "samepot",
                    "n_pairs": len(valid),
                    "ref_cost_median": main["cost_true_median"],
                    "matched_cost_median": round(float(np.median(
                        [m["row"]["cost_true_per_step"] for m in valid])), 8) if valid else None,
                    "matched_budget_used_mean": round(float(np.mean(
                        [m["row"]["budget_used"] for m in valid])), 4) if valid else None,
                })

            # ---- 4) 逐 episode 配对比较：上界臂 vs 每基线（同预算） ----
            for b in BASELINES:
                m_rows = per_base[b]
                deltas, wins = [], 0
                for r_ref, m in zip(main["rows"], m_rows):
                    if m["row"] is None:
                        continue
                    d = float(m["row"]["cost_true_per_step"]) - float(r_ref["cost_true_per_step"])
                    deltas.append({"episode": r_ref["episode"], "delta": round(d, 8)})
                    if d > 0:      # 基线更贵 ⇒ 上界臂赢
                        wins += 1
                med_m = (float(np.median([m["row"]["cost_true_per_step"]
                                          for m in m_rows if m["row"] is not None]))
                         if any(m["row"] is not None for m in m_rows) else float("nan"))
                out["matched"].append({
                    **key, "policy": b, "mode": "samepot_pairwise",
                    "n_pairs": len(deltas),
                    "n_oracle_wins": wins,
                    "win_fraction": round(wins / max(1, len(deltas)), 4),
                    # ★ v1.1 修正方向：candidate(oracle) / baseline。
                    #   旧代码是 `med_m / main["cost_true_median"]` = baseline/oracle，
                    #   与门槛 `ratio < 0.9`（"候选便宜 10%"）方向相反 ⇒ 判词必然反。
                    "cost_ratio_median": round(main["cost_true_median"] / med_m, 6),
                    "deltas": deltas,
                })

    return out


def gates(out: dict, *, n_cells_expected: int) -> dict:
    """预注册判据 + 三条接线判据。**看结果前写定**。"""
    from collections import defaultdict

    rows = []

    def _row(name, ok, detail):
        rows.append({"name": name, "pass": bool(ok), "detail": detail})

    from .evaluate import required_cells_from_design

    matched = [m for m in out["matched"] if m["mode"] == "samepot_pairwise"]
    by_cell = defaultdict(list)
    for m in matched:
        by_cell[(m["rho"], m["budget_ratio"])].append(m)
    cells = sorted(by_cell)

    # ★★ v1.1：判据改成**逐基线**，不再"每格挑一个基线再汇总"。
    #   旧写法有两个错：
    #     (a) `min(cost_ratio_median)` 在真比例为 candidate/baseline 时选的是
    #         "候选相对最便宜"的基线 —— 由于旧代码方向反了，实际选的是**赢它最多**的那条；
    #     (b) 只报"最好的那条基线"会掩盖"它输给其它基线"，与"一个启发式输了 ≠ 信息无空间"矛盾。
    required = required_cells_from_design(cells)   # 4 格 ⇒ 要求 4，不再写死 6/8
    by_base = defaultdict(dict)
    for m in matched:
        by_base[m["policy"]][(m["rho"], m["budget_ratio"])] = m

    per_baseline = []
    for pol, cellmap in sorted(by_base.items()):
        n_pass = 0
        detail = []
        for key in cells:
            m = cellmap.get(key)
            if m is None:
                detail.append({"cell": list(key), "status": "missing"})
                continue
            ok = (m["cost_ratio_median"] < WIN_MEDIAN_RATIO
                  and m["win_fraction"] >= WIN_FRACTION)
            n_pass += int(ok)
            detail.append({"cell": list(key),
                           "ratio_oracle_over_baseline": m["cost_ratio_median"],
                           "win_fraction": m["win_fraction"], "pass": bool(ok)})
        per_baseline.append({"baseline": pol, "n_cells": len(cells),
                             "n_pass": n_pass,
                             "beats_this_baseline": n_pass == required,
                             "cells": detail})

    n_baselines_beaten = sum(1 for b in per_baseline if b["beats_this_baseline"])
    _row("K0 逐基线报告：诊断臂相对每条基线，是否在**全部**设计格上 "
         f"ratio<{WIN_MEDIAN_RATIO} 且 胜率≥{WIN_FRACTION}",
         True,   # ★ 本条是**记录**，不是"通过/失败"：它测的是诊断臂，不是协议正确性
         {"design_cells": len(cells), "required_cells": required,
          "n_baselines": len(per_baseline),
          "n_baselines_beaten": n_baselines_beaten,
          "per_baseline": per_baseline,
          "note": "本项恒为记录项：诊断臂输赢不构成研究判据（v1.1 撤回 SPACE-VOID）"})

    # K0b（真正会 FAIL 的一条）：格数要求必须由设计推导，且不得有缺格
    missing_cells = [(b["baseline"], c["cell"]) for b in per_baseline
                     for c in b["cells"] if c.get("status") == "missing"]
    _row("K0b 每条基线在**每个设计格**都有配对结果（缺格 ⇒ 判据不可满足）",
         (len(cells) == required) and (not missing_cells),
         {"n_cells": len(cells), "required": required,
          "n_missing": len(missing_cells), "examples": missing_cells[:5]})

    # ---- 接线判据（阴性对照） ----
    orc = {(o["rho"], o["budget_ratio"], o["variant"]): o for o in out["oracle"]}
    keys = sorted({(k[0], k[1]) for k in orc})
    diff_h = sum(1 for k in keys
                 if orc[(k[0], k[1], "main")]["cost_true_median"]
                 != orc[(k[0], k[1], "no_h")]["cost_true_median"])
    _row("K1 真信道特权真的接线（关掉 ⇒ 成本必变）", diff_h == len(keys),
         {"n_cells": len(keys), "n_changed": diff_h})
    diff_x = sum(1 for k in keys
                 if orc[(k[0], k[1], "main")]["cost_true_median"]
                 != orc[(k[0], k[1], "no_x")]["cost_true_median"])
    _row("K2 真状态特权真的接线（关掉 ⇒ 成本必变）", diff_x == len(keys),
         {"n_cells": len(keys), "n_changed": diff_x})

    # ---- 记录：探测次数（预言的检验，不作判据） ----
    probe_rec = [{"rho": k[0], "budget_ratio": k[1],
                  "main_probe": orc[(k[0], k[1], "main")]["probe_mean"],
                  "probeable_probe": next(
                      (c["probe_mean"] for c in out["cells"]
                       if c["rho"] == k[0] and c["budget_ratio"] == k[1]
                       and c["tag"] == "oracle_probeable"), None),
                  "main_attempted": orc[(k[0], k[1], "main")]["attempted_mean"]}
                 for k in keys]
    _row("K3 记录上界臂的探测次数（预言：=0；不作判据，结论必须点名）", True, probe_rec)

    # ---- 记账恒等式 ----
    bad = [(c["tag"], c["rho"], c["budget_ratio"], r["episode"])
           for c in out["cells"] if c.get("tag", "").startswith("oracle")
           for r in c["rows"] if not r["identities_pass"]]
    _row("K4 上界臂 12 项记账恒等式全过", not bad,
         {"n_failed": len(bad), "examples": bad[:5]})

    n_fail = sum(0 if r["pass"] else 1 for r in rows)
    return {"n_checks": len(rows), "n_fail": n_fail, "all_pass": n_fail == 0, "rows": rows}


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="CW-R1-3 ORACLE-KNOWN 上界臂（落盘）")
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--ep-start", type=int, default=100)
    ap.add_argument("--note", default="")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    eps = range(args.ep_start, args.ep_start + args.episodes)
    out = run_all(episodes=eps)
    g = gates(out, n_cells_expected=len(RHOS) * len(RATIOS))
    elapsed = round(time.perf_counter() - t0, 2)

    run_id = None
    if not args.no_write:
        now = datetime.now(timezone(timedelta(hours=8)))
        cfg_fp = C.CWConfig(plant=C.bounded_plant()).fingerprint()
        run_id = f"cw_r1_3_{now.strftime('%Y%m%dT%H%M%S')}_{cfg_fp}_{_src_hash()}"
        out_dir = RESULTS / run_id
        out_dir.mkdir(parents=True, exist_ok=True)

        def _dump(name, obj):
            (out_dir / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2),
                                        encoding="utf-8")

        _dump("cells.json", out["cells"])
        _dump("matched.json", out["matched"])
        _dump("oracle.json", out["oracle"])
        _dump("gates.json", g)
        _dump("meta.json", {
            "run_id": run_id,
            "protocol": C.CWConfig.PROTOCOL_VERSION,
            "phase": "CW-R1-3 ORACLE-KNOWN 真值辅助贪心诊断臂（v1.1 更正版）",
            "note": args.note,
            "at_asia_shanghai": now.isoformat(),
            "git_head": _git(["rev-parse", "HEAD"]),
            "git_dirty_experiments": _git_dirty_self(),
            "src_hash": _src_hash(),
            "config_fingerprint_bounded": cfg_fp,
            "plant": "BOUNDED",
            "episodes": {"split": "dev", "start": args.ep_start, "n": args.episodes},
            "baselines": list(BASELINES),
            "rho": list(RHOS),
            "budget_ratios": list(RATIOS),
            "prereg": {"ratio": "candidate(oracle)/baseline",
                       "median_ratio_lt": WIN_MEDIAN_RATIO,
                       "win_fraction_ge": WIN_FRACTION,
                       "required_cells": "由设计格数推导（不再写死 min_cells）"},
            "disclaimer": (
                "ORACLE-KNOWN 是**真值辅助的单步贪心诊断臂**，不是上界：它按真值挑当前"
                "最值得发的用户，无前瞻、不评估测量的价值、不考虑预算续接。"
                "它读真信道 log h 与真状态 x 的模，通过 run_episode(allow_truth_keys=True) "
                "显式白名单构造。它**不可部署**，任何引用必须带这两句。"
                "★ 2026-10-22 撤回：①「上界」一名；②由它输赢推出的「SPACE-VOID / 信息无空间」结论。"
                "理由：它不是最优策略，且**从不探测**（probe_mean=0）⇒ 对「测量的价值」无信息量。"
                "cost_true_per_step 为准；samepot 匹配臂是「给对手同样的钱」的构造。"
            ),
        })

    print(f"[CW-R1-3] episodes/cell={args.episodes} cells={len(out['cells'])} "
          f"matched={len(out['matched'])} elapsed={elapsed}s")
    if run_id:
        print(f"[CW-R1-3] run_id={run_id}")
    print(f"[CW-R1-3] {g['n_checks']} 判据，{g['n_fail']} FAIL -> "
          f"{'ALL PASS' if g['all_pass'] else 'FAIL'}")
    for r in g["rows"]:
        print(f"  {'PASS' if r['pass'] else 'FAIL'}  {r['name']}")
        if not r["pass"] or r["name"].startswith(("K0", "K3")):
            print(f"        {json.dumps(r['detail'], ensure_ascii=False)[:900]}")
    return 0 if g["all_pass"] else 2


if __name__ == "__main__":
    import sys
    sys.exit(main())
