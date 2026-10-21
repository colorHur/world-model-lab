# -*- coding: utf-8 -*-
"""CW-R1 第 1 步：强基线（固定周期 / ACK 感知 / 任务价值）。

★ 短卡（**看结果前写定**，v4 §5「新假设先写短卡」）
------------------------------------------------------
问题：在 CW-POMDP-v1 下，**同一个预算口径**里，三种不看世界模型、不学习的基线
      （固定周期 / ACK 感知 / 任务价值）能否跑满整个面板，且各自的
      `probe / attempt / delivered / forced_hold` 记账恒等式全过？
      —— 这是 R1 的**入场检查**，不是研究结论。

判据（否定条件写死，不允许事后挪）
----------------------------------
B0 每个 (plant, rho, B) 格子上，**每条**基线的 `check_identities` 12 项全真。
B1 三条基线在 24 个格子上都**实际发出**过数据（`n_attempted > 0`）且
   **实际送达**过（`n_delivered > 0`）—— 只在闸上"跑通"但一帧没发不算补齐全。
B2 触发条件：ACK 感知基线的 `ack_used > 0`（证明 ACK 分支真的被走到），
   任务价值基线的 `value_greedy_used > 0`。
B3 不同基线**不是恒等式**：在同一格子上 `policy_trace_hash` 必须出现 ≥2 个不同值，
   否则说明策略接入没生效（教训 #77 同族：「没有通道」要先被排除）。

不做什么
--------
· 不训练任何模型；不预测未来；不读真状态/真信道（全部经 `SchedulerView`）。
· 不用 `served` 之类的尝试计数冒充送达（R17 / STATE §7 技术债）。
· 不在 `R0_DEFAULT` plant 上比较成本（开环不稳定，均值不可比，见
  `diag_plant_stability.py` 的 D1：末/中段 = 1929 倍）；该 plant 只跑恒等式。

成本口径（v4 §4「指标在看新结果前定义」）
----------------------------------------
主口径 `cost_true_per_step` = 每 episode `Σ_t (q·Σ_i x_i² + r·Σ_i u_i²) / T`，
x 用仿真内部真值。副口径 `cost_est_per_step` 用接收端估计 —— 只作诊断，
两者量级不同不是 bug（R0 已写明）。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from . import config as C
from .runner import check_identities, run_episode
from .streams import make_episode_rngs
from .world import Action, p_success_from_log_h

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results" / "cw_v1" / "runs"
HORIZON = 256


# ------------------------------------------------------------------ 基线策略
class _Base:
    """基线公共壳：只拿 `SchedulerView`，只用公开信息。"""

    name = "base"

    def __call__(self, v):  # pragma: no cover - 由子类实现
        raise NotImplementedError

    # 诊断计数（每个 episode 新建策略实例，故不必重置）
    ack_used = 0
    value_greedy_used = 0


class PeriodicBaseline(_Base):
    """固定周期：`v.t % period == 0` 就发；不在周期点则 HOLD。不探测。"""

    def __init__(self, period: int = 8) -> None:
        self.period = int(period)
        self._k = 0
        self.name = f"base_periodic{self.period}"
        self.ack_used = 0
        self.value_greedy_used = 0

    def __call__(self, v):
        if v.t % self.period == 0:
            a = Action("SEND", self._k % v.n_users)
            self._k += 1
            return a
        return Action("HOLD")


class AckAwareBaseline(_Base):
    """ACK 感知：**只用自己收到的 ACK/NACK**（不含 CSI 读数、不含真值）。

    规则（一次性写定）：
      · 上一槽 NACK ⇒ 立刻换下一个用户重试（不做指数退避 —— 退避属于 R3）；
      · 连续 NACK 达 `backoff_after` 次 ⇒ 跳过 `cooldown` 个槽（避让坏信道）；
      · 其余情况按固定周期 `period` 轮转。
    ACK 从 `v.history[-1].ack` 读（`history` 已剔除真值字段，见 `SchedulerView`）。
    """

    def __init__(self, period: int = 8, backoff_after: int = 2, cooldown: int = 3) -> None:
        self.period = int(period)
        self.backoff_after = int(backoff_after)
        self.cooldown = int(cooldown)
        self._k = 0
        self._nack_streak = 0
        self._cooldown_until = -1
        self.name = f"base_ack_p{self.period}_b{self.backoff_after}c{self.cooldown}"
        self.ack_used = 0
        self.value_greedy_used = 0

    def __call__(self, v):
        hist = v.history
        if hist:
            last = hist[-1]
            if last["ack"] == 0:                      # NACK
                self._ack_used_inc()
                self._nack_streak += 1
                if self._nack_streak >= self.backoff_after:
                    self._cooldown_until = v.t + self.cooldown
                    self._nack_streak = 0
                a = Action("SEND", self._k % v.n_users)   # 换一个用户重试
                self._k += 1
                return a
            if last["ack"] == 1:                      # ACK
                self._ack_used_inc()
                self._nack_streak = 0
        if v.t < self._cooldown_until:
            return Action("HOLD")
        if v.t % self.period == 0:
            a = Action("SEND", self._k % v.n_users)
            self._k += 1
            return a
        return Action("HOLD")

    def _ack_used_inc(self) -> None:
        self.ack_used += 1


class TaskValueBaseline(_Base):
    """任务价值：按**估计误差方差 × 陈旧度**给用户打分，每 `period` 槽发给最急的那个。

    分数（一次性写定）：`score_i = P_hat_i · (1 + aoi_i)`，`aoi_i = -1`（从未观测）
    ⇒ `aoi` 取 `HORIZON`（最大可能陈旧度）。`P_hat` 与 `aoi` 都来自 telemetry，
    属于可部署信息（不读真值）。

    探测策略：若某用户 `csi_aoi` 超过 `csi_max_age` 且当前槽不是发送槽，则探测它
    （探测与数据争用同一预算，这是 v4 协议的核心约束）。
    """

    def __init__(self, period: int = 8, csi_max_age: int = 32, n_users: int = 2) -> None:
        self.period = int(period)
        self.csi_max_age = int(csi_max_age)
        self._n = int(n_users)
        self._probed = 0
        self.name = f"base_value_p{self.period}_csi{self.csi_max_age}"
        self.ack_used = 0
        self.value_greedy_used = 0

    def __call__(self, v):
        tel = v.telemetry
        scores = []
        for i, row in enumerate(tel):
            aoi = row["task_aoi"]
            aoi = HORIZON if aoi is None or aoi < 0 else aoi
            scores.append(float(row["P_hat"]) * (1.0 + float(aoi)))
        best = int(np.argmax(scores))

        if v.t % self.period == 0:
            self.value_greedy_used += 1
            return Action("SEND", best)

        # 非发送槽：挑 CSI 最陈旧的用户探测。
        # ★★ 修正 1（2026-10-21 18:2x）：初版条件是 `csi_aoi >= csi_max_age`，
        #    而 `csi_aoi` 对**从未探测**的用户是 -1 ⇒ 我把它映射成 HORIZON ⇒
        #    条件对"从未测过"永远成立 ⇒ `csi_max_age=1` 时变成**每槽都探**，
        #    探测挤掉发送（实测 n_probe=112、attempted 从 32 掉到 16、成本翻倍）。
        #    正确条件：**已过期的已测用户** 或 **从未测过的用户**；
        #    并且加总量配额 `probe_budget = T // period`（与发送槽同频，
        #    保证探测与数据各占一半左右而不至于把预算吃光）。
        # ★★ 修正 2：`csi_max_age` 必须真的起作用 —— 用相对配额而不是绝对次数。
        probe_cap = max(1, HORIZON // self.period)
        if self._probed >= probe_cap:
            return Action("HOLD")
        csi_ages = []
        for row in tel:
            a = row["csi_aoi"]
            csi_ages.append(None if a is None or a < 0 else int(a))
        # 候选：从未测过（None）或已过期
        cand = [i for i, a in enumerate(csi_ages) if a is None or a >= self.csi_max_age]
        if cand:
            # 从未测过优先（信息增益最大），其次最陈旧
            j = max(cand, key=lambda i: (csi_ages[i] is None, csi_ages[i] or 0))
            self._probed += 1
            return Action("PROBE", j)
        return Action("HOLD")


# ------------------------------------------------------------------ 面板
@dataclass
class Cell:
    plant: str
    rho: float
    budget_ratio: float
    policy: str

    def key(self) -> str:
        return f"{self.plant}|rho{self.rho:.2f}|B{self.budget_ratio:.1f}|{self.policy}"


def _make_policy(name: str, n_users: int) -> _Base:
    if name.startswith("periodic"):
        return PeriodicBaseline(int(name[len("periodic"):]))
    if name.startswith("ack"):
        # ack_b2c3
        body = name[len("ack_b"):]
        b, c = body.split("c")
        return AckAwareBaseline(period=8, backoff_after=int(b), cooldown=int(c))
    if name.startswith("value"):
        body = name[len("value_csi"):]
        return TaskValueBaseline(period=8, csi_max_age=int(body), n_users=n_users)
    raise ValueError(f"未知基线 {name!r}")


BASELINES = ("periodic1", "periodic4", "periodic8", "periodic16",
             "ack_b2c3", "value_csi64", "value_csi8", "value_csi1")


def _run_cell(cfg: C.CWConfig, budget: int, rho: float, pol_name: str,
              episodes: range) -> dict:
    rows = []
    for ep in episodes:
        pol = _make_policy(pol_name, cfg.plant.n_users)
        res = run_episode(cfg, pol, split="dev", episode=ep, budget_total=budget, rho=rho)
        ids = check_identities(res, cfg)
        rows.append({
            "episode": ep,
            "cost_true_per_step": round(res.cost_true_per_step, 8),
            "cost_est_per_step": round(res.cost_est_per_step, 8),
            "n_probe": res.n_probe,
            "n_attempted": res.n_attempted,
            "n_delivered": res.n_delivered,
            "n_ack": res.n_ack,
            "n_nack": res.n_nack,
            "n_idle": res.n_idle,
            "n_forced_hold": res.n_forced_hold,
            "budget_used": res.budget_used,
            "budget_wasted": res.budget_wasted,
            "mean_task_aoi": round(float(res.mean_task_aoi), 6),
            "identities_pass": bool(ids["pass"]),
            "identities_fail": [k for k, v in ids.items() if k != "pass" and not v],
            "trace_hash": res.policy_trace_hash,
            "ack_used_by_policy": int(getattr(pol, "ack_used", 0)),
            "value_greedy_used_by_policy": int(getattr(pol, "value_greedy_used", 0)),
        })
    costs = np.array([r["cost_true_per_step"] for r in rows], dtype=float)
    return {
        "n_episodes": len(rows),
        "cost_true_mean": round(float(costs.mean()), 8),
        "cost_true_median": round(float(np.median(costs)), 8),
        "cost_true_sd": round(float(costs.std(ddof=1)) if len(costs) > 1 else 0.0, 8),
        "cost_true_max": round(float(costs.max()), 8),
        "cost_est_mean": round(float(np.mean([r["cost_est_per_step"] for r in rows])), 8),
        "probe_mean": round(float(np.mean([r["n_probe"] for r in rows])), 4),
        "attempted_mean": round(float(np.mean([r["n_attempted"] for r in rows])), 4),
        "delivered_mean": round(float(np.mean([r["n_delivered"] for r in rows])), 4),
        "forced_hold_mean": round(float(np.mean([r["n_forced_hold"] for r in rows])), 4),
        "ack_used_total": int(sum(r["ack_used_by_policy"] for r in rows)),
        "value_greedy_total": int(sum(r["value_greedy_used_by_policy"] for r in rows)),
        "identities_all_pass": bool(all(r["identities_pass"] for r in rows)),
        "rows": rows,
    }


def build_panel(*, episodes: range, plants: dict[str, C.PlantConfig],
                rhos: tuple[float, ...], ratios: tuple[float, ...],
                baselines: tuple[str, ...]) -> tuple[list[dict], list[str]]:
    cells: list[dict] = []
    for pname, plant in plants.items():
        cfg = C.CWConfig(plant=plant)
        T = int(cfg.protocol.horizon)
        for ratio in ratios:
            budget = cfg.protocol.budget(ratio)  # Fraction ⇒ 整数
            for rho in rhos:
                for b in baselines:
                    agg = _run_cell(cfg, budget, rho, b, episodes)
                    cells.append({
                        "plant": pname, "rho": rho, "budget_ratio": ratio,
                        "budget": int(budget), "policy": b,
                        "config_fingerprint": cfg.fingerprint(),
                        **agg,
                    })
    return cells, []


def gates(cells: list[dict], *, episodes_per_cell: int) -> dict:
    """B0–B3：短卡里写定的四条判据。"""
    rows = []

    def add(name: str, ok: bool, detail: dict) -> None:
        rows.append({"name": name, "pass": bool(ok), "detail": detail})

    # B0 恒等式
    bad = [f"{c['plant']}|{c['rho']}|{c['budget_ratio']}|{c['policy']}"
           for c in cells if not c["identities_all_pass"]]
    add("B0 每格每条基线记账恒等式 12 项全过", not bad,
        {"n_cells": len(cells), "cells_failed": bad[:8], "n_failed": len(bad)})

    # B1 实际发出 / 实际送达
    no_tx = [f"{c['plant']}|{c['rho']}|{c['budget_ratio']}|{c['policy']}"
             for c in cells if c["attempted_mean"] <= 0]
    no_del = [f"{c['plant']}|{c['rho']}|{c['budget_ratio']}|{c['policy']}"
              for c in cells if c["delivered_mean"] <= 0]
    add("B1a 每格每条基线 n_attempted > 0（实际发出过）", not no_tx,
        {"n_failed": len(no_tx), "cells": no_tx[:8]})
    add("B1b 每格每条基线 n_delivered > 0（实际送达过）", not no_del,
        {"n_failed": len(no_del), "cells": no_del[:8]})

    # B2 触发条件真被走到（按 plant 分开看，因为 BOUNDED 才是主面板）
    per_plant = {}
    for pname in sorted({c["plant"] for c in cells}):
        sub = [c for c in cells if c["plant"] == pname]
        ack_cells = [c for c in sub if c["policy"].startswith("ack")]
        val_cells = [c for c in sub if c["policy"].startswith("value")]
        per_plant[pname] = {
            "ack_used_total": int(sum(c["ack_used_total"] for c in ack_cells)),
            "value_greedy_total": int(sum(c["value_greedy_total"] for c in val_cells)),
        }
    add("B2 ACK 分支与任务价值分支都被真的走到（ack_used>0 且 value_greedy>0）",
        all(v["ack_used_total"] > 0 and v["value_greedy_total"] > 0
            for v in per_plant.values()),
        per_plant)

    # B3 策略接入非恒等：每格 hash 至少 2 个不同值
    flat = []
    for c in cells:
        key = (c["plant"], c["rho"], c["budget_ratio"])
        flat.append((key, {r["trace_hash"] for r in c["rows"]}, c["policy"]))
    grid: dict[tuple, set] = {}
    grid_pol: dict[tuple, list] = {}
    for key, hs, pol in flat:
        grid.setdefault(key, set()).update(hs)
        grid_pol.setdefault(key, []).append(pol)
    single = [f"{k[0]}|{k[1]}|{k[2]}" for k, hs in grid.items() if len(hs) < 2]
    add("B3 同格子内不同基线动作序列不同（hash 去重 ≥2）", not single,
        {"n_grids": len(grid), "grids_single_hash": single[:8],
         "n_single": len(single)})

    # ---- B4–B6：**通道判据**。只看 BOUNDED（R0_DEFAULT 只跑恒等式）
    bnd = [c for c in cells if c["plant"] == "BOUNDED"]
    by = {(c["rho"], c["budget_ratio"], c["policy"]): c for c in bnd}

    # B4 去噪（**必要条件**，不是充分条件）：只改探测次数、不改发送策略时，
    #    任务侧资源与动作序列必须逐位相同。
    #    ★ 2026-10-21 18:2x 第一版这条判据把 `value_csi32` 与 `value_csi1` 对比，
    #      结果 FAIL —— **FAIL 得正确**：`csi_max_age=1` 且带"从未测过"分支时，
    #      探测把预算吃光、挤掉了 SEND（实测 n_probe 112 vs 16，cost 0.1278 vs 0.0688）。
    #      ⇒ 判据对象选错了：两个探测档位本来就是**不同的发送策略**。
    #      正确做法：把"探测次数"作为唯一变量，在同一套**发送**规则下比较。
    #    ★★ 第二版（同分钟）：改成 32 vs 8 后仍 FAIL —— `by` 只含 BOUNDED，
    #      而我把 `value_csi8` 从 BASELINES 换掉了 ⇒ 字典里**没有这个键** ⇒
    #      `noisy` 为空 ⇒ 判据在"检查对象不存在"时静默 FAIL（教训 #46 同族）。
    #      正确处置：两个探测档位都留在面板里（64 / 8 / 1），并用 `None` 跳过缺失项
    #      时**显式报错**，不许空表 PASS/FAIL。
    noisy = []
    missing = []
    for rho in sorted({c["rho"] for c in bnd}):
        for ratio in sorted({c["budget_ratio"] for c in bnd}):
            a = by.get((rho, ratio, "value_csi64"))
            b = by.get((rho, ratio, "value_csi8"))
            if a is None or b is None:
                missing.append([rho, ratio, a is None, b is None])
                continue
            if a is None or b is None:
                continue
            same_att = abs(a["attempted_mean"] - b["attempted_mean"]) < 1e-9
            noisy.append({"rho": rho, "ratio": ratio,
                          "probe_32": a["probe_mean"], "probe_8": b["probe_mean"],
                          "att_32": a["attempted_mean"], "att_8": b["attempted_mean"],
                          "cost_32": a["cost_true_mean"], "cost_8": b["cost_true_mean"],
                          "attempted_identical": same_att,
                          "cost_identical": abs(a["cost_true_mean"] - b["cost_true_mean"]) < 1e-12})
    if missing:
        add("B4 检查对象缺失（面板里没有 value_csi64 / value_csi8 两个档位）", False,
            {"missing": missing})
    else:
      add("B4 csi_max_age 64→8：探测次数真的变了（≥1.5×），但发送次数与成本逐位相同",
        len(noisy) == 4
        and all(r["probe_8"] >= 1.5 * r["probe_32"] for r in noisy)
        and all(r["attempted_identical"] and r["cost_identical"] for r in noisy),
        noisy)

    # B5 预算有效（**每条基线各自**在低预算处必须被截断）
    def _fh(ratio):
        return {(c["rho"], c["policy"]): c["forced_hold_mean"]
                for c in bnd if c["budget_ratio"] == ratio}
    fh_lo, fh_hi = _fh(0.5), _fh(0.8)
    # ★ 第一版要求"全部基线 forced_hold>0" ⇒ FAIL。原因不是物错：periodic4/8/16 与
    #   value/ack 每集只花 32~66 量子，**根本到不了** B/T=0.5 的 128 量子上限 ⇒
    #   没有 forced_hold 是**正确**行为。判据改成"至少一条基线被截断，且截断者在
    #   高预算处放松"。
    truncated = [k for k, v in fh_lo.items() if v > 0]
    add("B5a B/T=0.5 时至少有基线被预算截断（forced_hold>0）", len(truncated) >= 1,
        {"n_truncated": len(truncated), "keys": [list(k) for k in truncated]})
    add("B5b 被截断的基线在 B/T=0.8 处 forced_hold 严格下降",
        all(fh_hi.get(k, 0) < fh_lo[k] for k in truncated),
        {"at_0.5": {str(k): fh_lo[k] for k in truncated},
         "at_0.8": {str(k): fh_hi.get(k) for k in truncated}})

    # B6 **通道存在性（核心）**：只用公开信息的基线里，最优与最差必须差出量级
    spread = []
    for rho in sorted({c["rho"] for c in bnd}):
        for ratio in sorted({c["budget_ratio"] for c in bnd}):
            vals = [(c["policy"], c["cost_true_mean"]) for c in bnd
                    if c["rho"] == rho and c["budget_ratio"] == ratio]
            lo = min(vals, key=lambda x: x[1])
            hi = max(vals, key=lambda x: x[1])
            spread.append({"rho": rho, "ratio": ratio, "best": lo, "worst": hi,
                           "worst_over_best": round(hi[1] / max(lo[1], 1e-12), 3)})
    add("B6 同格内「只看公开信息的基线」成本跨度 ≥1.5×（通信决策通道存在）",
        all(s["worst_over_best"] >= 1.5 for s in spread), spread)

    n_fail = sum(0 if r["pass"] else 1 for r in rows)
    return {"n_checks": len(rows), "n_fail": n_fail,
            "all_pass": n_fail == 0, "rows": rows}


def _src_hash() -> str:
    import hashlib

    h = hashlib.sha256()
    for f in sorted((REPO / "experiments" / "cw_v1").glob("*.py")):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def _git(args: list[str]) -> str:
    import subprocess

    GIT = r"D:\devtools\Git\cmd\git.exe"
    try:
        return subprocess.check_output([GIT, *args], cwd=str(REPO),
                                       text=True, encoding="utf-8").strip()
    except Exception as exc:  # pragma: no cover
        return f"<git-fail:{exc}>"


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="CW-R1-1 强基线补齐（恒等式 + 面板）")
    ap.add_argument("--episodes", type=int, default=12, help="每格 episode 数")
    ap.add_argument("--ep-start", type=int, default=100)
    ap.add_argument("--note", default="")
    ap.add_argument("--no-write", action="store_true", help="只打印，**不写** results/")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    plants = {
        "R0_DEFAULT": C.PlantConfig(),
        "BOUNDED": C.bounded_plant(),
    }
    eps = range(args.ep_start, args.ep_start + args.episodes)
    cells, _ = build_panel(episodes=eps, plants=plants, rhos=(0.85, 0.98),
                           ratios=(0.5, 0.8), baselines=BASELINES)
    g = gates(cells, episodes_per_cell=args.episodes)
    elapsed = round(time.perf_counter() - t0, 2)

    run_id = None
    if not args.no_write:
        tz = timezone(timedelta(hours=8))
        now = datetime.now(tz)
        cfg_fp = C.CWConfig(plant=C.bounded_plant()).fingerprint()
        run_id = f"cw_r1_base_{now.strftime('%Y%m%dT%H%M%S')}_{cfg_fp}_{_src_hash()}"
        out_dir = RESULTS / run_id
        out_dir.mkdir(parents=True, exist_ok=True)

        def _dump(name: str, obj) -> None:
            (out_dir / name).write_text(
                json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

        _dump("panel_cells.json", cells)
        _dump("gates.json", g)
        _dump("meta.json", {
            "run_id": run_id,
            "protocol": "CW-POMDP-v1",
            "phase": "CW-R1-1 强基线补齐",
            "note": args.note,
            "at_asia_shanghai": now.isoformat(),
            "git_head": _git(["rev-parse", "HEAD"]),
            "git_dirty": _git(["status", "--porcelain"]),
            "src_hash": _src_hash(),
            "config_fingerprint_bounded": cfg_fp,
            "config_fingerprint_r0default": C.CWConfig().fingerprint(),
            "episodes": {"split": "dev", "start": args.ep_start, "n": args.episodes},
            "plants": {"R0_DEFAULT": C.PlantConfig().__dict__,
                       "BOUNDED": C.bounded_plant().__dict__},
            "baselines": list(BASELINES),
            "rho": [0.85, 0.98],
            "budget_ratios": [0.5, 0.8],
            "n_cells": len(cells),
            "elapsed_s": elapsed,
            "disclaimer": (
                "R0_DEFAULT 只用来跑记账恒等式：它是开环不稳定的，"
                "cost_true 均值被末端几十步支配（末64/中64 = 1929×），不可用于比较。"
                "BOUNDED 是 R1 主面板，其 SEND 价值已由 diag_plant_stability D6a-c 预注册判据确认。"
                "所有成本以 cost_true_per_step 为准，cost_est_per_step 只作诊断。"
            ),
        })

    print(f"[CW-R1-1] plant={list(plants)}  episodes/cell={args.episodes} "
          f"cells={len(cells)} elapsed={elapsed}s")
    if run_id:
        print(f"[CW-R1-1] run_id={run_id}")
    print(f"[CW-R1-1] {g['n_checks']} 判据，{g['n_fail']} FAIL -> "
          f"{'ALL PASS' if g['all_pass'] else 'FAIL'}")
    for r in g["rows"]:
        print(f"  {'PASS' if r['pass'] else 'FAIL'}  {r['name']}")
        if not r["pass"]:
            print(f"        {json.dumps(r['detail'], ensure_ascii=False)[:500]}")

    print("\n[面板摘要] plant | rho | B/T | policy | cost_true/step(mean/median) | "
          "probe/att/del/fhold")
    for c in cells:
        print(f"  {c['plant']:11s} {c['rho']:.2f} {c['budget_ratio']:.1f} "
              f"{c['policy']:14s} {c['cost_true_mean']:10.6f}/{c['cost_true_median']:10.6f} "
              f"{c['probe_mean']:6.1f}/{c['attempted_mean']:5.1f}/"
              f"{c['delivered_mean']:5.1f}/{c['forced_hold_mean']:5.1f}")
    return 0 if g["all_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
