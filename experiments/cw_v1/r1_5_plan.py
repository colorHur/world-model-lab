# -*- coding: utf-8 -*-
"""CW-R1 有限视界规划参考的落盘入口：闸 + **开发诊断面板**。

★ 本 run 的 `kind` 是 `PLANNING-REFERENCE-NOT-UPPER-BOUND`：
  有限视界 + Monte-Carlo 的**参考**，不是最优策略，**不是上界**。

★ 为什么面板叫"开发诊断"而不是"收益表"（必须照抄引用）
  本轮实现闸 16 项里 **G4 的第二子判据 FAIL**：`M` 从 16 加到 32 后，**逐槽动作**
  一致率 0.910（过 0.90），但**逐 episode 成本相对差中位 27.7%**（判据 ≤10%）。
  且 32→64 仍有 20%/54%/105%。这不是"采样不够"，是**近并列决定 × 协议路径依赖**：
  1 步贪心 + "无新信息"尾部对"什么时候花这一帧"几乎无差别（与 P3-1 的
  "一帧的价值几乎与陈旧度无关"同源），于是微小抽样差会翻转发帧时点，
  而发帧时点决定后半段的漂移长度 ⇒ 成本方差远大于待测效应。
  ⇒ 依 v4.1「若不完整，点名未完成项并续接」：**本轮不出科学判词**，
    只出 (a) 已验证的规划器代码、(b) 有分辨力的闸、(c) 这个"参考还不够稳"的实测诊断。
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from . import config as C
from .evaluate import CellStats, compare_cell, decide, required_cells_from_design
from .plan_gates import mutation_gates, run_all as run_plan_gates
from .plan_ref import make
from .policies import PacedAge, PeriodicSend
from .runner import check_identities, run_episode
from .world import Action

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results" / "cw_v1"

#: 预注册面板（卡 §5）：ρ × B/T 四格；本轮 `dev` 只用一小段做诊断
RHOS = (0.85, 0.98)
RATIOS = (0.5, 0.8)
EPISODES = tuple(range(100, 108))      # 8 集/格（诊断规模，不是确认规模）
H_PLAN = 5
M_SAMPLES = 32


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


class _Rec:
    """记录逐槽动作（算发帧时点分布用）。"""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.name = getattr(inner, "name", "rec")
        self.acts: list[tuple[int, str, int | None]] = []

    def __call__(self, v):
        a = self.inner(v)
        self.acts.append((int(v.t), a.kind, a.user))
        return a


def _run(cfg: C.CWConfig, pol, *, budget: int, rho: float, ep: int) -> dict:
    rec = _Rec(pol)
    r = run_episode(cfg, rec, split="dev", episode=ep, budget_total=budget, rho=rho)
    ids = check_identities(r, cfg)
    n_win = 8
    win = int(cfg.protocol.horizon) // n_win
    bends = [0] * n_win
    for (t, k, _u) in rec.acts:
        if k in ("SEND", "PROBE"):
            bends[min(t // win, n_win - 1)] += 1
    return {
        "episode": ep,
        "cost_true": r.cost_true_per_step,
        "cost_est": r.cost_est_per_step,
        "n_probe": r.n_probe, "n_attempted": r.n_attempted,
        "n_delivered": r.n_delivered, "n_idle": r.n_idle,
        "n_forced_hold": r.n_forced_hold,
        "budget_used": r.budget_used, "budget_wasted": r.budget_wasted,
        "mean_task_aoi": r.mean_task_aoi, "mean_csi_aoi": r.mean_csi_aoi,
        "cost_true_by_user": r.cost_true_by_user,
        "send_by_window": bends,
        "trace_hash": r.policy_trace_hash,
        "identities_pass": ids["pass"],
        "identities_fail": [k for k, v in ids.items() if k != "pass" and not v],
        "planner_probe_same_argmax": int(getattr(rec.inner, "probe_same_argmax", 0)),
        "planner_probe_diff_argmax": int(getattr(rec.inner, "probe_diff_argmax", 0)),
    }


def build_panel() -> dict:
    cfg = C.CWConfig(plant=C.bounded_plant())
    out: dict = {"cells": [], "config": cfg.to_dict()}
    for ratio in RATIOS:
        budget = cfg.protocol.budget(ratio)
        for rho in RHOS:
            cell = {"rho": rho, "budget_ratio": ratio, "budget": int(budget), "arms": {}}
            arms = {
                "plan_probe": lambda ep, rho=rho: make(cfg, arm="probe", h_plan=H_PLAN,
                                                      m_samples=M_SAMPLES,
                                                      seed=1000 + ep, rho=rho),
                "plan_noprobe": lambda ep, rho=rho: make(cfg, arm="noprobe", h_plan=H_PLAN,
                                                        m_samples=M_SAMPLES,
                                                        seed=1000 + ep, rho=rho),
                "periodic2": lambda ep, rho=rho: PeriodicSend(2),
                "paced_age8": lambda ep, rho=rho: PacedAge(8),
            }
            for name, mk in arms.items():
                rows = [_run(cfg, mk(ep), budget=budget, rho=rho, ep=ep) for ep in EPISODES]
                cell["arms"][name] = rows
            # 配对比较（方向：ratio = candidate / baseline，<1 候选更便宜）
            cand = [r["cost_true"] for r in cell["arms"]["plan_probe"]]
            cell["compare"] = {}
            for base in ("plan_noprobe", "periodic2", "paced_age8"):
                b = [r["cost_true"] for r in cell["arms"][base]]
                st: CellStats = compare_cell(b, cand, cell=(round(rho, 2), ratio, base))
                cell["compare"][base] = st.as_dict()
            # 发帧时点分布（8 个窗口内的资源动作数，均值）
            cell["send_profile"] = {
                name: np.mean([r["send_by_window"] for r in rows], axis=0).round(3).tolist()
                for name, rows in cell["arms"].items()
            }
            out["cells"].append(cell)
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="CW-R1 有限视界规划参考：闸 + 开发诊断面板")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--skip-panel", action="store_true", help="只跑闸")
    ap.add_argument("--note", default="")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    g = run_plan_gates()
    mut_rows, mut_missed = mutation_gates()
    n_fail = g.summary()["n_fail"] + mut_missed
    print(f"[CW-R1.5] plan_gates: {g.summary()['n_checks']} checks, "
          f"{g.summary()['n_fail']} fail")
    for r in g.rows:
        if not r["pass"]:
            print("  FAIL ", r["name"])
    print(f"[CW-R1.5] 变异闸: {len(mut_rows)} 次破坏, {mut_missed} 次未被抓")

    panel = {} if args.skip_panel else build_panel()
    elapsed = round(time.perf_counter() - t0, 2)
    if panel:
        for cell in panel["cells"]:
            print(f"  rho={cell['rho']} B/T={cell['budget_ratio']}: "
                  + ", ".join(f"{k}={np.median([r['cost_true'] for r in v]):.4f}"
                              for k, v in cell["arms"].items()))

    # ★ 判词资格：G4 未过 ⇒ 不出科学判词（卡 §5 的 N1/N2 暂不判）
    gate4 = next((r for r in g.rows if r["name"].startswith("G4")), None)
    eligible = bool(gate4 and gate4["pass"])
    verdict = {
        "eligible": eligible,
        "reason": ("G4（M 加倍稳定性）通过 ⇒ 具备判词资格" if eligible else
                   "G4 未通过：M 加倍后逐 episode 成本相对差远超判据 ⇒ "
                   "待测效应与规划器自身方差不可分离，本轮**不出** N1/N2 判词"),
        "n_gate_fail": g.summary()["n_fail"],
        "n_mutation_missed": mut_missed,
    }
    print(f"[CW-R1.5] 判词资格: {verdict['eligible']} —— {verdict['reason']}")
    print(f"[CW-R1.5] total {n_fail} fail / "
          f"{g.summary()['n_checks'] + len(mut_rows)} checks  ({elapsed} s)")

    if not args.no_write:
        now = datetime.now(timezone(timedelta(hours=8)))
        cfg = C.CWConfig(plant=C.bounded_plant())
        run_id = (f"cw_r1_5_plan_{now.strftime('%Y%m%dT%H%M%S')}_"
                  f"{cfg.fingerprint()}_{_src_hash()}")
        out_dir = RESULTS / run_id
        out_dir.mkdir(parents=True, exist_ok=True)

        def _dump(name, obj):
            (out_dir / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2),
                                        encoding="utf-8")

        _dump("plan_gates.json", g.summary())
        _dump("mutation_gates.json", {"n_mutations": len(mut_rows),
                                      "n_missed": mut_missed, "rows": mut_rows})
        _dump("panel.json", panel)
        _dump("verdict.json", verdict)
        _dump("meta.json", {
            "run_id": run_id,
            "kind": "PLANNING-REFERENCE-NOT-UPPER-BOUND",
            "protocol": C.CWConfig.PROTOCOL_VERSION,
            "event_pairing": cfg.protocol.event_pairing,
            "h_plan": H_PLAN, "m_samples": M_SAMPLES,
            "episodes": list(EPISODES), "rhos": list(RHOS), "ratios": list(RATIOS),
            "config": cfg.to_dict(),
            "note": args.note,
            "at_asia_shanghai": now.isoformat(),
            "elapsed_s": elapsed,
            "git_head": _git(["rev-parse", "HEAD"]),
            "git_status_porcelain": _git(["status", "--porcelain"]),
            "src_hash": _src_hash(),
            "disclaimer": (
                "有限视界 + Monte-Carlo 规划参考，**不是上界、不是最优策略**；"
                "本面板是 8 集/格的开发诊断，不是确认集。"
                "G4（M 加倍稳定性）未通过 ⇒ 本轮不出「测量值不值」的科学判词。"
                "主指标 = cost_true_per_step；cost_est 只作诊断（HOLD 下它恒为 0，"
                "见 plan_gates.G6）。"
            ),
        })
        print(f"[CW-R1.5] 证据落盘 results/cw_v1/{run_id}")

    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
