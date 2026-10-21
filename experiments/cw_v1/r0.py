# -*- coding: utf-8 -*-
"""CW-R0 入口：跑反例闸 + 一份最小开发面板，落盘到 `results/cw_v1/runs/<run-id>/`。

用法（本机工位机）：
    <python> -m experiments.cw_v1.r0 --note "首轮协议验收"

★ 只写自己的目录：`results/cw_v1/runs/`。不碰旧 `results/`。
★ run-id = 协议 + 配置指纹 + 源码树哈希 + UTC 时间，可追溯。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np

from .config import default_config
from .gates import run_all
from .policies import (AlwaysProbe, AlwaysSend, CsiSensitive, PacedAge, PeriodicSend,
                       ProbeThenSend)
from .runner import run_episode

REPO = Path(__file__).resolve().parents[2]
PKG = Path(__file__).resolve().parent


def _src_hash() -> str:
    h = hashlib.sha256()
    for p in sorted(PKG.glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def _git(args: list[str]) -> str:
    try:
        out = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, timeout=30)
        return out.stdout.strip()
    except Exception as exc:  # pragma: no cover
        return f"<git 调用失败: {exc}>"


def _dev_panel(cfg) -> list[dict]:
    """最小开发面板：2 rho × 2 预算 × 6 策略 × 3 episode。

    ★ 这是**开发样本**，不是确认集。仅用于验证协议在真实运行下不崩、记账自洽。
    """
    policies = [
        AlwaysProbe(),
        AlwaysSend(),
        PeriodicSend(period=8, order="aoi"),
        PacedAge(theta=4),
        CsiSensitive(theta=4, use_csi=True),
        ProbeThenSend(csi_max_age=4),
    ]
    rows = []
    for ri, rho in enumerate(cfg.channel.rho_panel):
        for ratio in cfg.protocol.budget_ratios:
            B = cfg.protocol.budget(ratio)
            for pol in policies:
                for ep in range(3):
                    r = run_episode(cfg, pol, split="dev", episode=100 + 10 * ri + ep,
                                    budget_total=B, rho=rho)
                    rows.append({
                        "rho": rho, "budget_ratio": ratio, "budget": B,
                        "policy": r.policy, "episode": r.episode,
                        "cost_true_per_step": r.cost_true_per_step,
                        "cost_est_per_step": r.cost_est_per_step,
                        "cost_true_by_user": r.cost_true_by_user,
                        "n_probe": r.n_probe, "n_attempted": r.n_attempted,
                        "n_delivered": r.n_delivered, "n_idle": r.n_idle,
                        "n_forced_hold": r.n_forced_hold,
                        "budget_used": r.budget_used, "budget_wasted": r.budget_wasted,
                        "mean_task_aoi": r.mean_task_aoi, "mean_csi_aoi": r.mean_csi_aoi,
                        "policy_trace_hash": r.policy_trace_hash,
                        "elapsed_s": r.elapsed_s,
                    })
    return rows


def _aggregate(rows: list[dict]) -> list[dict]:
    import collections

    groups: dict[tuple, list[dict]] = collections.defaultdict(list)
    for r in rows:
        groups[(r["rho"], r["budget_ratio"], r["policy"])].append(r)
    out = []
    for (rho, ratio, pol), rs in sorted(groups.items()):
        cost = np.array([x["cost_true_per_step"] for x in rs])
        out.append({
            "rho": rho, "budget_ratio": ratio, "budget": rs[0]["budget"], "policy": pol,
            "n_episodes": len(rs),
            "cost_mean": float(cost.mean()),
            "cost_std": float(cost.std(ddof=1)) if len(rs) > 1 else 0.0,
            "cost_min": float(cost.min()), "cost_max": float(cost.max()),
            "cost_by_user_mean": [float(np.mean([x["cost_true_by_user"][i] for x in rs]))
                                  for i in range(len(rs[0]["cost_true_by_user"]))],
            "probe_mean": float(np.mean([x["n_probe"] for x in rs])),
            "attempted_mean": float(np.mean([x["n_attempted"] for x in rs])),
            "delivered_mean": float(np.mean([x["n_delivered"] for x in rs])),
            "idle_mean": float(np.mean([x["n_idle"] for x in rs])),
            "forced_hold_mean": float(np.mean([x["n_forced_hold"] for x in rs])),
            "budget_used_mean": float(np.mean([x["budget_used"] for x in rs])),
            "delivery_rate": float(np.mean([x["n_delivered"] / max(1, x["n_attempted"]) for x in rs])),
            "aoi_mean": float(np.mean([x["mean_task_aoi"] for x in rs])),
        })
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="CW-R0 协议验收 + 最小开发面板")
    ap.add_argument("--note", default="")
    ap.add_argument("--skip-panel", action="store_true", help="只跑反例闸（**不写** results/）")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    cfg = default_config()
    tz = timezone(timedelta(hours=8))
    now = datetime.now(tz)
    head = _git(["rev-parse", "HEAD"])
    dirty = _git(["status", "--porcelain"])
    src = _src_hash()
    run_id = f"cw_r0_{now.strftime('%Y%m%dT%H%M%S')}_{cfg.fingerprint()}_{src}"

    gates = run_all(cfg)
    panel = [] if args.skip_panel else _dev_panel(cfg)
    agg = _aggregate(panel) if panel else []
    elapsed = round(time.perf_counter() - t0, 2)

    # ★ `--skip-panel` 是"只跑自检"入口 ⇒ **不得**触碰入库证据（R45：自检入口不得写 results/）。
    #   2026-10-21 修正：初版无论是否 skip 都 mkdir + 写 4 个文件，等于每次自检都往证据链里塞一份
    #   `n_panel_rows=0` 的空 run。现在 skip 时只打印，不落盘。
    if args.skip_panel:
        print(f"[CW-R0] gates: {gates['n_checks']} checks, {gates['n_fail']} fail -> "
              f"{'ALL PASS' if gates['all_pass'] else 'FAIL'}")
        for r in gates["rows"]:
            if not r["pass"]:
                print(f"  FAIL  {r['name']}: {json.dumps(r['detail'], ensure_ascii=False)[:300]}")
        print(f"[CW-R0] --skip-panel：未写 results/（自检不入证据链）")
        return 0 if gates["all_pass"] else 2

    out_dir = REPO / "results" / "cw_v1" / "runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "run_id": run_id,
        "protocol": "CW-POMDP-v1",
        "stage": "CW-R0",
        "note": args.note,
        "time_utc": now.astimezone(timezone.utc).isoformat(),
        "time_local": now.isoformat(),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "git_head": head,
        "git_dirty_before": dirty.splitlines() if dirty else [],
        "src_hash": src,
        "config": cfg.to_dict(),
        "config_fingerprint": cfg.fingerprint(),
        "elapsed_s": elapsed,
        "n_gate_checks": gates["n_checks"],
        "n_gate_fail": gates["n_fail"],
        "gate_all_pass": gates["all_pass"],
        "n_panel_rows": len(panel),
        "disclaimer": (
            "R0 只验证协议（信息边界/时序/记账/子流），不训练模型、不声称方法有效。"
            "面板数字是开发样本，不是确认集。cost_est_per_step 用先验估计计算，"
            "cost_true_per_step 用真状态计算；主表口径以 cost_true 为准，"
            "cost_est 只作诊断（两者量级差约 10^3，见 MEMORY 中的说明）。"
        ),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "gates.json").write_text(json.dumps(gates, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "panel_rows.json").write_text(json.dumps(panel, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "panel_agg.json").write_text(json.dumps(agg, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[CW-R0] run_id={run_id}")
    print(f"[CW-R0] gates: {gates['n_checks']} checks, {gates['n_fail']} fail -> "
          f"{'ALL PASS' if gates['all_pass'] else 'FAIL'}")
    for r in gates["rows"]:
        if not r["pass"]:
            print(f"  FAIL  {r['name']}: {json.dumps(r['detail'], ensure_ascii=False)[:300]}")
    if agg:
        print(f"[CW-R0] panel rows={len(panel)} elapsed={elapsed}s")
        print(f"{'rho':>5} {'B/T':>5} {'policy':<22} {'cost':>9} {'probe':>6} {'att':>5} "
              f"{'deliv':>6} {'idle':>5} {'fhold':>6} {'aoi':>7}")
        for a in agg:
            print(f"{a['rho']:>5} {a['budget_ratio']:>5} {a['policy']:<22} "
                  f"{a['cost_mean']:>9.4f} {a['probe_mean']:>6.1f} {a['attempted_mean']:>5.1f} "
                  f"{a['delivered_mean']:>6.1f} {a['idle_mean']:>5.1f} "
                  f"{a['forced_hold_mean']:>6.1f} {a['aoi_mean']:>7.2f}")
    print(f"[CW-R0] wrote {out_dir}")
    return 0 if gates["all_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
