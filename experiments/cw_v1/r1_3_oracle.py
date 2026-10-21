# -*- coding: utf-8 -*-
"""CW-R1-3：**`ORACLE-KNOWN` 上界臂的落盘入口**（R1 第 2 步第 1 项）。

预注册判据（写于此行、看结果之前；对应 STATE §7）
------------------------------------------------
- 每格（ρ × B/T）相对**该格最好基线**的 `samepot` 匹配臂：
  **中位成本比 < 0.9** 且 **逐 episode 胜率 ≥ 0.7**；
- 8 格里 **≥ 6 格**同时满足上述两条 ⇒ 判 **`ORACLE-KNOWN` 赢**（进 R2）；
- 否则判 **`SPACE-VOID`**，本课题在 R1 收口（不进 R2）。

★ 为什么要求"≥6/8 格"而不是"全部"：8 个格子里有两格（`B/T=0.5`、`ρ=0.98`）的
  HOLD 支配面最大（R1-2 实测 `p32@ρ0.98` 有 4/12 集 HOLD 更便宜），单格波动本来就大。
  预注册的"6/8"是**下限**；实际落在 6 与 8 之间必须原样写出来，不许取整成"基本全赢"。

★★ 本入口**只跑 BOUNDED**（`R0_DEFAULT` 开环不稳定，末64/中64=1929×，不可比较）。
★★ `ORACLE-KNOWN` 是**故意越界的上界臂**（读真信道 + 真状态模），任何结论必须带这句限定。

不做的三件事（写下来防止下一轮"顺手"做了）
--------------------------------------------
1. 不扫 `kappa_scale`（那会变成"看到结果再调参"）；只在 §敏感性 里跑 0.25/0.5/1.0 各一遍，
   并**只用于说明结论对 κ 的一阶不敏感**，不进预注册判据。
2. 不把 `OracleKnownProbeable` 的 `n_probe=0` 当作"探测通道已关闭"的证据（设计使然）。
3. 不声称"学习方法不行"。判 `SPACE-VOID` 时的准确措辞是
   **"在 CW-POMDP-v1（BOUNDED plant）的信息结构下，全知上界也赢不了基线"**，
   它限制的是**这个协议 + 这个 plant**，不是"学习方法"。
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
WIN_MIN_CELLS = 6


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
                    "cost_ratio_median": round(med_m / main["cost_true_median"], 6),
                    "deltas": deltas,
                })

    return out


def gates(out: dict, *, n_cells_expected: int) -> dict:
    """预注册判据 + 三条接线判据。**看结果前写定**。"""
    from collections import defaultdict

    rows = []

    def _row(name, ok, detail):
        rows.append({"name": name, "pass": bool(ok), "detail": detail})

    matched = [m for m in out["matched"] if m["mode"] == "samepot_pairwise"]
    by_cell = defaultdict(list)
    for m in matched:
        by_cell[(m["rho"], m["budget_ratio"])].append(m)

    cells = []
    for key, ms in sorted(by_cell.items()):
        best = min(ms, key=lambda m: m["cost_ratio_median"])
        # ★ 该格"最好基线"的匹配臂：成本最低的那个（比值最小）——**先定规则再看数**
        cells.append({
            "rho": key[0], "budget_ratio": key[1],
            "best_policy": best["policy"],
            "best_ratio": best["cost_ratio_median"],
            "best_win_fraction": best["win_fraction"],
            "pass_ratio": best["cost_ratio_median"] < WIN_MEDIAN_RATIO,
            "pass_fraction": best["win_fraction"] >= WIN_FRACTION,
            "pass": (best["cost_ratio_median"] < WIN_MEDIAN_RATIO
                     and best["win_fraction"] >= WIN_FRACTION),
        })
    n_win = sum(1 for c in cells if c["pass"])
    _row(f"K0 预注册：≥{WIN_MIN_CELLS}/{n_cells_expected} 格 中位比<{WIN_MEDIAN_RATIO} 且 "
         f"胜率≥{WIN_FRACTION}", n_win >= WIN_MIN_CELLS,
         {"n_cells": len(cells), "n_win_cells": n_win, "cells": cells,
          "verdict": "ORACLE-KNOWN-WINS" if n_win >= WIN_MIN_CELLS else "SPACE-VOID"})

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
            "protocol": "CW-POMDP-v1",
            "phase": "CW-R1-3 ORACLE-KNOWN 上界臂",
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
            "prereg": {"median_ratio_lt": WIN_MEDIAN_RATIO,
                       "win_fraction_ge": WIN_FRACTION,
                       "min_cells": WIN_MIN_CELLS},
            "disclaimer": (
                "ORACLE-KNOWN 是**故意越界的上界臂**：读真信道 log h 与真状态 x 的模，"
                "通过 run_episode(allow_truth_keys=True) 显式白名单构造，不经 view._inner。"
                "它**不可部署**，任何引用必须带这句。"
                "判 SPACE-VOID 时的准确措辞是「在 CW-POMDP-v1（BOUNDED plant）的信息结构下，"
                "全知上界也赢不了基线」，限制的是**该协议 + 该 plant**，不是「学习方法不行」。"
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
