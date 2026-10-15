# -*- coding: utf-8 -*-
"""★ 交付阶段第 1 项：六族调度**统一最小闭环**；第 2 项：用同一 runner **重现 X44**。

================================================================= 为什么有这一号
华腾 2026-10-15 的交付顺序：
    ① 六族最小闭环（一个任务、少量配置跑通六族，共用预算/种子/提前终止/结果表，
       校准集与评测集分开）；
    ② 用统一 runner 重现一个**已有**结果（优先「估计指标改善与任务收益不一致」）；
    ③ 外部验收（另有 `docs/REPRODUCE.md` §6）。

而此前的真实状态是：`periodic` / `threshold` / `timing` 三族**只在空跑里**评估过
（`scripts/30`），`selfreport` / `conformal` / `oracle` 三族**只注册、从未真正进过闭环**
（X38–X43 是离线重演）。本号把六族**放进同一条闭环**，共用：
    · 同一 env / 模型 / 控制器（同一 seed）
    · 同一评测预算（n_ep / measured_steps / warmup / tail_frac）
    · 同一评测 seed（闭环 online_seed）
    · 同一"提前终止"处理（跟丢即终止，**照报** escape_rate / short_frac，不静默丢）
    · 同一结果表（`scheduling.Row`）+ 同一等预算配对（走 `pairing.overlap_window`）
**校准集与评测集分开**：conformal 信封与 oracle 的 z 表只用 `calib_seed_offset` 的
episode 拟合，评测用 `online_seed` 的 episode —— 两组种子**不相交**（K3）。

================================================================= 接线验收（本轮新增的重点）
`state.reads`（在 `scheduling.py`）只证明**读过字段**；本号另做**可证伪**的核对：
  W1 AoI 递推：调度器第 i+1 次读到的 age == 0（第 i 次送达）或 prev_age+1
     ⇒ 证明"读到的是**当前步、正确时点**的状态"，不是旧值；
  W2 送达 ⇒ 下一步 U == 0；W3（oracle）送达 ⇒ 下一步真值误差 == 0；
  W4 每个 episode 首查询 age == 0。
另加**行为探针**：同一族的旋钮（阈值/τ/tol）变化 ⇒ 触发率必须**单调响应**（否则=没接线）。

================================================================= 自检（不过就抛 —— R14）
S_u1 六族**都在闭环里真的跑出数**（n_steps>0，且信号相关族的 `state.reads` 覆盖 `.needs`）
S_u2 校准集与评测集 seed 不相交（`calib_seed_offset` ≠ `eval/online_seed_offset`）
S_u3 接线核对 W1–W4 全过（`closedloop.check_wiring`）
S_u4 行为探针：`selfreport` / `oracle` / `conformal` 的触发率随旋钮**单调不增**
S_u5 ★ 重现 X44：timing 族在**完全相同的设置**下与 `outputs/28_x44_timing.json` 对比
     —— 相对差 ≤ `1e-9` 记为「逐位复现」（归并到统一 runner 没有改变结论）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from wmlab.control import (PDRelativeController, collect_controlled_episodes,  # noqa: E402
                           run_closed_loop_control)
from wmlab.data import split_episodes, transitions_from_episodes  # noqa: E402
from wmlab.envs import make_env  # noqa: E402
from wmlab.eval import closedloop, pairing, scheduling  # noqa: E402
from wmlab.eval.timing import n_tx_budget, plan_times, subsample, timing_dispersion  # noqa: E402
from wmlab.models.prob_world_model import GaussianWorldModel  # noqa: E402
from wmlab.train import train_world_model  # noqa: E402
from wmlab.utils import count_params, get_device, load_config, output_dir, set_seed  # noqa: E402
from wmlab.utils.plot import PALETTE, apply_style, save_fig  # noqa: E402

#: oracle 族用的**真值误差**键名（`control.run_closed_loop_control` 在 `_oracle` 为真时写入）
TRUTH_KEY = "oracle_err"
REPRO_X44_JSON = "outputs/28_x44_timing.json"


# ================================================================= 参数
def parse_args():
    p = argparse.ArgumentParser(
        description="六族调度统一最小闭环 + 用同一 runner 重现 X44")
    p.add_argument("--config", default="configs/uav_triggered.yaml")
    p.add_argument("--tag", default="33_unified_closed_loop")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--mode", default="both", choices=["table", "repro-x44", "both"])
    p.add_argument("--quick", action="store_true")
    p.add_argument("--retrain", action="store_true", help="忽略模型缓存、重训")
    # --- 表（Part A）的设置 ---
    p.add_argument("--per", type=float, nargs="+", default=[0.0, 0.3],
                   help="信号无关族的丢包率档（表用）")
    p.add_argument("--n-ep", type=int, default=None, help="闭环 episode 数（表用）")
    p.add_argument("--meas-steps", type=int, default=None, help="被测段步数（表用）")
    p.add_argument("--warmup", type=int, default=None, help="预热步数")
    p.add_argument("--calib-ep", type=int, default=None, help="独立标定集 episode 数")
    p.add_argument("--calib-samples", type=int, default=None, help="标定 rollout 窗口数")
    p.add_argument("--no-probe", action="store_true", help="跳过接线探针（冒烟用）")
    return p.parse_args()


# ================================================================= 结果行
def mkrow(family: str, knob_axis: str, knob_value: float, loss_prob: float, seed: int,
          r: dict, metric: str = "mean_dist_tail", reads: dict | None = None,
          diverged: int = 0, note: str = "") -> dict:
    """把 `run_closed_loop_control` 的返回压成**统一 schema** 的一行。"""
    return {
        "family": family, "knob_axis": knob_axis, "knob_value": float(knob_value),
        "loss_prob": float(loss_prob), "seed": int(seed),
        "tx_rate": float(r["tx_rate"]), "mean_age": float(r["mean_age"]),
        "tx_rate_target": None,
        "est_nmse": float(r["est_nmse"]),
        "metric": metric, "metric_value": float(r["mean_dist_tail"]),
        "mean_dist_tail": float(r["mean_dist_tail"]),
        "escape_rate": float(r["escape_rate"]), "n_tx": int(r["n_tx"]),
        "n_steps": int(r["n_steps"]), "n_episodes": int(r["n_episodes"]),
        "short_frac": float(np.mean(np.asarray(r["ep_lens"], dtype=float)
                                    < float(r["mean_ep_len"]))) if r["ep_lens"] else float("nan"),
        "diverged": int(diverged),
        "state_reads": dict(reads or {}),
        "deployable": bool(scheduling.REGISTRY[family].deployable),
        "diagnostic": bool(scheduling.REGISTRY[family].diagnostic),
        "note": note,
    }


def to_row(r: dict) -> scheduling.Row:
    """dict 行 → `scheduling.Row`（喂等预算配对）。"""
    return scheduling.Row(
        family=r["family"], knob_axis=r["knob_axis"], knob_value=r["knob_value"],
        loss_prob=r["loss_prob"], seed=r["seed"], tx_rate=r["tx_rate"],
        mean_age=r["mean_age"], metric=r["metric"], metric_value=r["metric_value"],
        n_eval=r["n_episodes"], diverged=r["diverged"],
        deployable=r["deployable"], diagnostic=r["diagnostic"], note=r["note"])


# ================================================================= Part A：六族表
class Runner:
    """六族共用的一条闭环（同一 env/模型/控制器/预算/种子）。"""

    def __init__(self, env, model, ctrl, *, device, var_g, n_ep, meas_steps,
                 warmup, tail_frac, online_seed):
        self.env = env
        self.model = model
        self.ctrl = ctrl
        self.device = device
        self.var_g = float(var_g)
        self.n_ep = int(n_ep)
        self.meas_steps = int(meas_steps)
        self.warmup = int(warmup)
        self.tail_frac = float(tail_frac)
        self.online_seed = int(online_seed)
        self.probe_log: list[dict] = []

    def run(self, family: str, *, loss_prob: float, knobs: dict | None = None,
            use_state: bool | None = None, do_probe: bool = False) -> tuple[dict, dict]:
        """构建该族调度器 → 进**同一条闭环** → 返回 (result, state_reads)。

        `use_state`：None = 由族自动决定（信号相关族为真；否则不传 u_state，
        保持与 `scripts/28` 的历史基线**逐位一致**的估计路径）。
        """
        knobs = dict(knobs or {})
        sp = scheduling.REGISTRY[family]
        sig = family in scheduling.SIGNAL_DEPENDENT
        if use_state is None:
            use_state = sig
        state = scheduling.ClosedLoopState() if use_state else None
        if sig:
            knobs.setdefault("state", state)
        if family == "oracle":
            state["_oracle"] = True                     # 让 control.py 写真值误差
            knobs.setdefault("truth_key", TRUTH_KEY)
        sched = scheduling.build(family, loss_prob=float(loss_prob), **knobs)
        probe = None
        if do_probe and sig:
            probe = closedloop.WiringProbe(
                sched, state, truth_key=(TRUTH_KEY if family == "oracle" else None))
            sched_run = probe
        else:
            sched_run = sched
        try:
            r = run_closed_loop_control(
                self.env, self.model, self.ctrl, sched_run,
                n_episodes=self.n_ep, seed=self.online_seed,
                max_steps=self.meas_steps, device=self.device, estimator="model",
                var_g=self.var_g, label=f"{family}/{getattr(sched, '__name__', '')}",
                tail_frac=self.tail_frac, warmup_steps=self.warmup,
                u_state=state)
            diverged = 0
        except FloatingPointError as e:                 # 世界模型 rollout 发散
            r = {"tx_rate": float("nan"), "mean_age": float("nan"), "est_nmse": float("nan"),
                 "mean_dist_tail": float("nan"), "escape_rate": float("nan"), "n_tx": 0,
                 "n_steps": 0, "n_episodes": self.n_ep, "ep_lens": [], "mean_ep_len": 0.0,
                 "diverged_reason": str(e)}
            diverged = 1
        reads = scheduling.state_read_summary(state) if state is not None else {}
        r["_diverged"] = diverged
        r["_probe"] = probe
        return r, reads


def part_a(runner: Runner, calib: dict, args, per_list: list[float]) -> dict:
    rows: list[dict] = []
    wiring: dict = {}
    probe_seen: dict[str, list] = {}

    def go(family, knob_axis, knob_value, loss_prob, knobs, note="", do_probe=True):
        r, reads = runner.run(family, loss_prob=loss_prob, knobs=knobs, do_probe=do_probe)
        row = mkrow(family, knob_axis, knob_value, loss_prob, runner.online_seed, r,
                    reads=reads, diverged=r["_diverged"], note=note)
        rows.append(row)
        if r["_probe"] is not None:
            probe_seen.setdefault(family, []).extend(r["_probe"].log)
        flag = "" if r["_diverged"] == 0 else "  ⚠发散(已记 diverged=1)"
        print(f"[33]   {family:<11} {knob_axis}={knob_value:<7g} p={loss_prob:<4g} "
              f"tx={r['tx_rate']:.4f} E[age]={r['mean_age']:6.3f} "
              f"nmse={r['est_nmse']:.5f} dist={r['mean_dist_tail']:.4f} "
              f"esc={r['escape_rate']:.2f} reads={reads}{flag}")
        return row

    # --- 信号无关三族：periodic / threshold（跨 PER 扫）；timing（PER=0，T=8）---
    for p in per_list:
        for T in (1, 2, 4, 8, 16):
            go("periodic", "period", T, p, {"period": T})
        for K in (1, 2, 4, 8, 16):
            go("threshold", "threshold", K, p, {"threshold": K})
    T0 = 8
    N0 = n_tx_budget(T0, runner.meas_steps)
    for mode, params in (("phase", subsample(0, T0 - 1, 4)),
                         ("jitter", subsample(0, (T0 - 1) // 2, 3)),
                         ("random", [0, 1])):
        for prm in params:
            go("timing", "param", prm, 0.0,
               {"mode": mode, "period": T0, "param": int(prm),
                "n_steps": runner.meas_steps, "n_tx": N0},
               note=f"mode={mode}")
    # --- 信号相关三族：阈值/容差从**独立标定集**的分位点取（不碰评测集）---
    # ★★ 为什么不拍 0.05/0.2/0.4 这种"看着合理"的数（2026-10-15 冒烟踩到）：
    #   `conformal` 只在 `age < K` 时才看信封 Ê_α；h<K 处 Ê_α 的量级是 ~0.01 ⇒
    #   三档 tol 全部 `< Ê_α` ⇒ 旋钮**扫过了但触发率完全不动**（实测三档都 0.2875，
    #   恰等于 threshold(K=2)）—— 一个**假接线**。⇒ 一律从标定分布取分位点。
    u_q = np.quantile(calib["u_fit_flat"], [0.15, 0.4, 0.65, 0.85, 0.95])
    # tol 取 `Ê_α` 在 h∈{1..K−1} 段（即真正会被分支用到的年龄）上的分位点
    K_conf = 4
    ehat_lo = np.asarray(calib["e_hat_fit"], dtype=np.float64)[:, :max(K_conf - 1, 1)]
    tol_q = np.quantile(ehat_lo, [0.05, 0.3, 0.55, 0.8, 0.95])
    # τ 必须**显式覆盖两端**（不能只取分位点）：z 的尾部极重（`oracle.py` 的 `s_floor`
    #   会让某些年龄的 Ŝ≈0 ⇒ z 可达 ~1e6 级），所以分位点（哪怕 q99.5）都到不了
    #   "age≥K 分支的下限"。⇒ 用 `[min−1, 分位点…, max×3+1]`：两端分别逼出 tx 的上/下端。
    #   冒烟实测：只取分位点 ⇒ tx 停在 0.31–0.41，与 threshold 的公共窗口 1.30× < 1.5×
    #   ⇒ 判"不可评估"（连试三次）。
    z_all = np.asarray(calib["z_fit_flat"], dtype=np.float64)
    z_q = np.quantile(z_all, [0.5, 0.8, 0.92, 0.995])
    tau_grid = np.concatenate([[float(z_all.min()) - 1.0], z_q,
                               [float(z_all.max()) * 3.0 + 1.0]])
    print(f"[33] 信号相关族旋钮（取自标定集分位点）："
          f"selfreport thr={[round(float(x), 4) for x in u_q]}")
    print(f"[33]     conformal(K={K_conf}) tol={[round(float(x), 5) for x in tol_q]}")
    print(f"[33]     oracle(K={K_conf}) tau min={float(z_all.min()):.3g} "
          f"max={float(z_all.max()):.3g}")
    for p in per_list:
        for q, thr in zip((0.15, 0.4, 0.65, 0.85, 0.95), u_q):
            go("selfreport", "threshold_u", float(thr), p, {"threshold_u": float(thr)},
               note=f"thr=U 的 q{q:g}（标定集分位点）")
        for q, tol in zip((0.05, 0.3, 0.55, 0.8, 0.95), tol_q):
            go("conformal", "tol", float(tol), p,
               {"K": K_conf, "tol": float(tol), "tab": calib["conf_tab"]},
               note=f"K={K_conf}, tol=Ê_α 的 q{q:g}（α={calib['conf_tab']['alpha']:g}）")
        for tau in tau_grid:
            go("oracle", "tau", float(tau), p,
               {"K": K_conf, "tau": float(tau), "tab": calib["or_tab"]},
               note=f"K={K_conf}, τ∈[z_min−1, z_max×3+1]（★不可部署·特权基线）")

    # ---------- S_u3 接线核对 ----------
    if not args.no_probe:
        for fam, log in probe_seen.items():
            tk = TRUTH_KEY if fam == "oracle" else None
            wiring[fam] = closedloop.check_wiring(log, family=fam, truth_key=tk)
            w = wiring[fam]
            print(f"[33] S_u3 接线核对[{fam}]：段={w['n_segments']} 查询={w['n_queries']} "
                  f"W1={w['n_w1']} W2={w['n_w2']} W3={w['n_w3']} W4={w['n_w4']} "
                  f"恒定U段={w['n_segments_constant_U']} 违约={len(w['violations'])}")

    # ---------- S_u1 / S_u4 ----------
    s_u1 = check_all_families_present(rows)
    s_u4 = check_monotone_response(rows)
    return {"rows": rows, "wiring": wiring, "s_u1": s_u1, "s_u4": s_u4}


def check_all_families_present(rows: list[dict]) -> dict:
    """S_u1：六族**都在闭环里真的跑出数**；信号相关族的 `state.reads` 覆盖 `.needs`。"""
    got = {}
    for fam in scheduling.FAMILIES:
        rs = [r for r in rows if r["family"] == fam and r["diverged"] == 0 and r["n_steps"] > 0]
        got[fam] = len(rs)
        if not rs:
            raise AssertionError(f"★ S_u1 失败：族 {fam!r} 在闭环里没有跑出任何有效行")
    for fam in sorted(scheduling.SIGNAL_DEPENDENT):
        need_keys = [k for k in ("U", "age", TRUTH_KEY)]
        rs = [r for r in rows if r["family"] == fam and r["diverged"] == 0]
        keys = set().union(*[set(r["state_reads"].keys()) for r in rs]) if rs else set()
        want = {"U"} if fam == "selfreport" else ({"U", "age"})
        if fam == "oracle":
            want = {"age", TRUTH_KEY}
        missing = want - keys
        if missing:
            raise AssertionError(
                f"★ S_u1 失败：族 {fam!r} 的 state.reads 缺 {sorted(missing)}"
                f"（实读 {sorted(keys)}）⇒ 统计量没接线")
    # 共同预算/种子：所有行的 loss_prob ∈ per_list，seed 相同
    seeds = {r["seed"] for r in rows}
    if len(seeds) != 1:
        raise AssertionError(f"★ S_u1 失败：结果表混了多个 seed {seeds} ⇒ 不是同一批评测")
    print(f"[33] S_u1 六族都在闭环里跑出数：{got}")
    return {"rows_per_family": got, "shared_seed": int(next(iter(seeds)))}


def check_monotone_response(rows: list[dict]) -> dict:
    """S_u4：行为探针 —— 触发率随旋钮**单调不增**（否则 = 统计量没进决策）。"""
    out = {}
    for fam, axis, higher_more_likely in (("selfreport", "threshold_u", False),
                                          ("conformal", "tol", False),
                                          ("oracle", "tau", False)):
        for p in sorted({r["loss_prob"] for r in rows if r["family"] == fam}):
            rs = sorted([r for r in rows if r["family"] == fam
                         and abs(r["loss_prob"] - p) < 1e-12 and r["diverged"] == 0],
                        key=lambda r: r["knob_value"])
            if len(rs) < 2:
                continue
            txs = [r["tx_rate"] for r in rs]
            # 阈值/τ/tol 都是"越大越不触发" ⇒ 触发率必须单调不增
            viol = [round(txs[i + 1] - txs[i], 6) for i in range(len(txs) - 1)
                    if (txs[i + 1] - txs[i] > 1e-9) == (not higher_more_likely)]
            out[f"{fam}@p={p}"] = {"knob": [r["knob_value"] for r in rs],
                                   "tx_rate": [round(t, 6) for t in txs],
                                   "n_violations": len(viol)}
            if len(viol):
                raise AssertionError(
                    f"★ S_u4 失败：{fam}@p={p} 触发率未随 {axis} 单调不增 "
                    f"（knob={[r['knob_value'] for r in rs]}, tx={txs}）⇒ 统计量可能没进决策")
            print(f"[33] S_u4 {fam}@p={p}：旋钮={[round(r['knob_value'], 4) for r in rs]} "
                  f"⇒ 触发率={[round(t, 4) for t in txs]} ✓单调不增")
    return out


# ================================================================= Part B：重现 X44
def part_b_repro(runner: Runner, args, cfg, env, ctrl, model) -> dict:
    """用**同一套 runner 构件**重现 X44（固定预算、只动发送时刻）。

    设置必须与 `scripts/28` **逐项相同**：seed=0、n_ep=30、meas=300、warmup=400、PER=0、
    T∈{8,16}、扰动的 subsample/range 相同、random 的 pattern_seed=param+1。
    """
    ref_path = os.path.join(_ROOT, REPRO_X44_JSON)
    if not os.path.isfile(ref_path):
        print(f"[33] Part B 跳过：找不到参照 {ref_path}")
        return {"ok": None, "reason": "no reference json"}
    ref = json.load(open(ref_path, "r", encoding="utf-8"))
    meas, warmup, n_ep = 300, 400, 30
    online_seed = int(cfg["seed"]) + int(cfg["data"].get("online_seed_offset", 2000))
    ref_rows = {(_k(r["kind"], r["T"], r.get("mode", "periodic"), r.get("param", 0))): r
                for r in ref["rows"]}
    print(f"[33] Part B 重现 X44：参照 {REPRO_X44_JSON}（{len(ref['rows'])} 行）")

    def _run(knobs, name):
        """★ 走**统一 runner 的 timing 族**（`scheduling.build`），而不是手写时刻表 ——
        这正是"重现"要证明的东西：归并进统一 runner **没有改变结论**。"""
        sched = scheduling.build("timing", loss_prob=0.0, **knobs)
        return run_closed_loop_control(
            env, model, ctrl, sched, n_episodes=n_ep, seed=online_seed, max_steps=meas,
            device=runner.device, estimator="model", var_g=runner.var_g, label=name,
            tail_frac=float(cfg["task"]["tail_frac"]), warmup_steps=warmup)

    diffs = []
    got_rows = []
    matched_keys = set()
    for T in (8, 16):
        N = n_tx_budget(T, meas)
        specs = [(None, 0)]
        specs += [("phase", int(d)) for d in subsample(0, T - 1, 5)]
        specs += [("jitter", int(j)) for j in subsample(0, (T - 1) // 2, 5)]
        specs += [("random", int(p)) for p in range(3)]
        for mode, prm in specs:
            if mode is None:
                knobs = {"mode": "phase", "period": T, "param": 0, "n_steps": meas, "n_tx": N}
                name, key = f"periodic(T={T})", ("base", T, "periodic", 0)
            elif mode == "random":
                knobs = {"mode": "random", "period": T, "param": prm, "n_steps": meas,
                         "n_tx": N, "pattern_seed": prm + 1}      # ★ 与 scripts/28 同参数
                name, key = f"rand(T={T},#{prm})", ("perturb", T, "random", prm)
            else:
                knobs = {"mode": mode, "period": T, "param": prm, "n_steps": meas, "n_tx": N}
                name, key = f"{mode}(T={T},{prm})", ("perturb", T, mode, prm)
            ts = plan_times(knobs["mode"], T, prm, meas, N,
                            pattern_seed=knobs.get("pattern_seed", 0))
            r = _run(knobs, name)
            rr = ref_rows.get(key)
            rec = {"T": T, "kind": key[0], "mode": mode or "periodic", "param": int(prm),
                   "est_nmse": float(r["est_nmse"]), "mean_dist_tail": float(r["mean_dist_tail"]),
                   "n_tx": int(r["n_tx"]), "disp": float(timing_dispersion(ts))}
            if rr is not None:
                matched_keys.add(key)
                rec["ref_est_nmse"] = float(rr["est_nmse"])
                rec["ref_mean_dist_tail"] = float(rr["mean_dist_tail"])
                rec["d_est_nmse"] = abs(rec["est_nmse"] - rec["ref_est_nmse"])
                rec["d_dist"] = abs(rec["mean_dist_tail"] - rec["ref_mean_dist_tail"])
                diffs.append((rec["d_est_nmse"], rec["d_dist"], name))
            got_rows.append(rec)
            print(f"[33]     {name:<20s} nmse={rec['est_nmse']:.6f} dist={rec['mean_dist_tail']:.6f} "
                  f"n_tx={rec['n_tx']}"
                  + (f"  Δnmse={rec.get('d_est_nmse', float('nan')):.3e}"
                     f" Δdist={rec.get('d_dist', float('nan')):.3e}" if rr is not None else "  (无参照)"))
    # ★★ 键必须**一一对上**：否则"只对上了少数几行"也会静默通过（R12：「名字/期望」≠ 实际核对）。
    if matched_keys != set(ref_rows.keys()):
        missing = sorted(set(ref_rows.keys()) - matched_keys)
        extra = sorted(matched_keys - set(ref_rows.keys()))
        raise AssertionError(
            f"★ S_u5 失败：与参照的键未一一对上（参照 {len(ref_rows)} 个）——"
            f"未匹配 {missing}；多出 {extra}")
    if not diffs:
        raise AssertionError("★ S_u5 失败：没有任何一行与参照对上（键不匹配）")
    max_dn = max(d[0] for d in diffs)
    max_dd = max(d[1] for d in diffs)
    rel_n = max_dn / max(abs(ref_rows[(_k('base', 8, 'periodic', 0))]["est_nmse"]), 1e-18)
    tol = 1e-9
    print(f"[33] S_u5 重现 X44：最大 |Δest_nmse| = {max_dn:.3e}，最大 |Δmean_dist_tail| = {max_dd:.3e}"
          f"（相对 {rel_n:.2e}，判据 ≤ {tol:g}）")
    if max_dn > tol or max_dd > tol:
        raise AssertionError(
            f"★ S_u5 失败：与 scripts/28 的参照不一致（最大 Δnmse={max_dn:.3e}, "
            f"Δdist={max_dd:.3e}）⇒ 归并到统一 runner **改变了**结果，先分根因")
    print("[33] S_u5 ✓ 逐位复现（归并代码没有改变结论）")
    return {"ok": True, "max_abs_d_est_nmse": max_dn, "max_abs_d_dist": max_dd,
            "tolerance": tol, "rows": got_rows,
            "ref_verdict": ref.get("verdict")}


def _k(kind, T, mode, param):
    return (kind, int(T), mode, int(param))


# ================================================================= 图
def make_figure(rows: list[dict], repro: dict | None, outdir, tag: str):
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.0))
    fams = list(scheduling.FAMILIES)
    cols = {"periodic": PALETTE["blue"], "threshold": PALETTE["red"],
            "timing": PALETTE["orange"], "selfreport": PALETTE["green"],
            "conformal": PALETTE["purple"], "oracle": PALETTE["grey"]}
    marks = {"periodic": "o", "threshold": "s", "timing": "^",
             "selfreport": "D", "conformal": "v", "oracle": "P"}
    p0 = min({r["loss_prob"] for r in rows}) if rows else 0.0

    ax = axes[0]
    for fam in fams:
        rs = [r for r in rows if r["family"] == fam and abs(r["loss_prob"] - p0) < 1e-12
              and r["diverged"] == 0]
        rs.sort(key=lambda r: r["tx_rate"])
        if rs:
            dep = bool(scheduling.REGISTRY[fam].deployable)
            ax.plot([r["tx_rate"] for r in rs], [r["mean_dist_tail"] for r in rs],
                    "-", marker=marks[fam], ms=5, lw=1.4, color=cols[fam],
                    label=f"{fam}{'' if dep else ' (diag)'}")
    ax.set_xscale("log")
    ax.set_xlabel(f"送达率 tx_rate（p={p0:g}）")
    ax.set_ylabel("任务指标 mean_dist_tail（稳态距离）")
    ax.set_title("(a) 六族：任务 vs 预算\n（同一条闭环、同一批 episode）")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1]
    for fam in fams:
        rs = [r for r in rows if r["family"] == fam and abs(r["loss_prob"] - p0) < 1e-12
              and r["diverged"] == 0]
        if rs:
            ax.scatter([r["est_nmse"] for r in rs], [r["mean_dist_tail"] for r in rs],
                       s=38, marker=marks[fam], color=cols[fam], alpha=0.85, label=fam)
    ax.set_xlabel("估计指标 est_nmse")
    ax.set_ylabel("任务指标 mean_dist_tail")
    ax.set_title(f"(b) 估计 vs 任务（p={p0:g}）\n（脱节 => 点云不沿对角线）")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)

    ax = axes[2]
    ax.axis("off")
    ax.text(0.0, 1.0, "(c) 统一闭环：六族 × 接线状态", fontsize=12, va="top")
    y = 0.90
    for fam in fams:
        sp = scheduling.REGISTRY[fam]
        rs = [r for r in rows if r["family"] == fam and r["diverged"] == 0]
        sig = fam in scheduling.SIGNAL_DEPENDENT
        reads = sorted(set().union(*[set(r["state_reads"].keys()) for r in rs])) if rs else []
        ax.text(0.0, y, f"{'●' if rs else '○'} {fam}", fontsize=11, va="top",
                color="#3a7" if rs else "#a33")
        ax.text(0.30, y, f"旋钮 {sp.knob}", fontsize=9, va="top")
        ax.text(0.60, y, ("可部署" if sp.deployable else "不可部署·诊断"),
                fontsize=9, va="top", color="#333" if sp.deployable else "#a33")
        ax.text(0.0, y - 0.045,
                f"    闭环 {len(rs)} 行" + (f"｜state 已读 {reads}" if sig else "｜信号无关"),
                fontsize=8, va="top", alpha=0.75)
        y -= 0.155
    ax.text(0.0, y, "★ 校准集与评测集 seed 不相交；oracle 标 diagnostic。",
            fontsize=8, va="top", alpha=0.8)
    if repro and repro.get("ok"):
        ax.text(0.0, y - 0.10, f"★ 重现 X44：最大 |Δ| = {repro['max_abs_d_est_nmse']:.1e}"
                              f"（≤ {repro['tolerance']:g}）=> 逐位复现",
                fontsize=8, va="top", alpha=0.8, color="#a33")

    fig.tight_layout()
    png = save_fig(fig, os.path.join(str(outdir), f"{tag}.png"))
    plt.close(fig)
    return png


# ================================================================= 主流程
def main():
    args = parse_args()
    t0 = time.time()
    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["train"]["epochs"] = int(args.epochs)
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    if args.quick:
        cfg["train"]["epochs"] = min(int(cfg["train"]["epochs"]), 30)
        cfg["data"]["n_episodes"] = min(int(cfg["data"]["n_episodes"]), 40)

    seed = int(cfg["seed"])
    set_seed(seed)
    apply_style()
    device = get_device(args.device or cfg.get("device"))
    out = output_dir(cfg)
    tmp = out / "_temp"
    tmp.mkdir(parents=True, exist_ok=True)

    # 预算（表用）：默认小，`--quick` 更小
    n_ep = int(args.n_ep or (8 if args.quick else 12))
    meas = int(args.meas_steps or (150 if args.quick else 200))
    warmup = int(args.warmup or int(cfg["task"]["warmup_steps"]))
    tail_frac = float(cfg["task"]["tail_frac"])
    calib_ep = int(args.calib_ep or (12 if args.quick else 24))
    calib_samp = int(args.calib_samples or (384 if args.quick else 768))
    online_off = int(cfg["data"].get("online_seed_offset", 2000))
    eval_off = int(cfg["data"]["eval_seed_offset"])
    calib_off = int(cfg["data"].get("calib_seed_offset", 3000))

    print("[33] " + "=" * 78)
    print(f"[33] 六族统一最小闭环：n_ep={n_ep} meas={meas} warmup={warmup} "
          f"PER={args.per} seed={seed}")
    print(f"[33] 评测 seed=seed+{online_off}；评测集 seed=seed+{eval_off}；"
          f"标定集 seed=seed+{calib_off}（三者不相交）")

    # ---------- 环境 / 模型 ----------
    wind_amp = float(cfg["env"].get("wind_amp", 0.0))
    env = make_env(cfg["env"]["id"], seed=seed,
                   noise_std=float(cfg["env"]["noise_std"]),
                   max_steps=int(cfg["env"]["max_steps"]), wind_amp=wind_amp)
    cc = cfg["controller"]
    ctrl = PDRelativeController(dt=env.dt, omega_n=float(cc["omega_n"]),
                                zeta=float(cc["zeta"]), kappa=float(env.kappa),
                                a_max=float(env.a_max))
    obs_dim = int(env.obs_dim)
    cache = tmp / "x33_model.pt"
    model = GaussianWorldModel(
        obs_dim=obs_dim, act_dim=int(env.act_dim), latent_dim=int(cfg["model"]["latent_dim"]),
        hidden=int(cfg["model"]["hidden"]), discrete_act=bool(env.is_discrete),
        logvar_init=float(cfg["model"].get("logvar_init", 0.0))).to(device)
    if cache.is_file() and not args.retrain:
        model.load_state_dict(torch.load(cache, map_location=device))
        print(f"[33] 载入模型缓存 {cache}")
    else:
        n_tr = int(cfg["data"]["n_episodes"])
        train_all = collect_controlled_episodes(env, ctrl, n_episodes=n_tr, seed=seed,
                                                max_steps=int(cfg["env"]["max_steps"]))
        train_eps, val_eps = split_episodes(train_all, float(cfg["train"]["val_ratio"]), seed)
        tr = tuple(torch.as_tensor(x) for x in transitions_from_episodes(train_eps))
        va = tuple(torch.as_tensor(x) for x in transitions_from_episodes(val_eps))
        hist = train_world_model(model, tr, va, cfg, device, verbose=False)
        torch.save(model.state_dict(), cache)
        print(f"[33] 训练完成：params={count_params(model):,} val={hist['val_total'][-1]:.6f} "
              f"⇒ 已缓存（{cache}）")

    # ---------- 评测集（var_g 的口径与 scripts/28 逐位一致）----------
    ev_seed = seed + eval_off
    ev_env = make_env(cfg["env"]["id"], seed=ev_seed,
                      noise_std=float(cfg["env"]["noise_std"]),
                      max_steps=int(cfg["env"]["max_steps"]), wind_amp=wind_amp)
    ev_eps = collect_controlled_episodes(
        ev_env, ctrl, n_episodes=max(30, int(cfg["data"]["n_episodes"]) // 3),
        seed=ev_seed, max_steps=int(cfg["env"]["max_steps"]))
    ev_env.close()
    allobs = np.concatenate([e["obs"][warmup:] for e in ev_eps], axis=0).astype(np.float64)
    var_g = float(allobs.var())
    print(f"[33] var_g={var_g:.10f}（评测集 {len(ev_eps)} 条，与 scripts/28 同口径）")

    # ---------- 独立标定集 ----------
    ca_seed = seed + calib_off
    ca_env = make_env(cfg["env"]["id"], seed=ca_seed,
                      noise_std=float(cfg["env"]["noise_std"]),
                      max_steps=int(cfg["env"]["max_steps"]), wind_amp=wind_amp)
    ca_ctrl = PDRelativeController(dt=ca_env.dt, omega_n=float(cc["omega_n"]),
                                   zeta=float(cc["zeta"]), kappa=float(ca_env.kappa),
                                   a_max=float(ca_env.a_max))
    ca_eps = collect_controlled_episodes(ca_env, ca_ctrl, n_episodes=calib_ep, seed=ca_seed,
                                        max_steps=int(cfg["env"]["max_steps"]))
    ca_env.close()
    calib = closedloop.calibrate(model, ca_eps, horizon=int(cfg["eval"]["horizons"][-1]),
                                 n_samples=calib_samp, var_g=var_g, device=device,
                                 conf_dims=slice(0, 2), or_dims=slice(0, 2),
                                 alpha=0.1, n_knots=int(cfg["design"]["conf_knots"]
                                                       if "conf_knots" in cfg["design"] else 8),
                                 seed=seed, t0_min=warmup)
    cov = calib["conf_cov"]
    print(f"[33] 标定集：可用 {calib['n_usable']} 条（fit {calib['n_fit']} / 留出 "
          f"{calib['n_test']}）｜Q_α={calib['conf_tab']['q']:.4f}"
          f"｜覆盖率 边缘={cov['marginal']:.4f} 逐年龄中位={cov['per_age_median']:.4f}"
          f"（目标 {cov['target']:.2f}）")

    runner = Runner(env, model, ctrl, device=device, var_g=var_g, n_ep=n_ep,
                    meas_steps=meas, warmup=warmup, tail_frac=tail_frac,
                    online_seed=seed + online_off)

    payload = {
        "experiment": "unified-closed-loop", "schema": "closedloop-v0.1",
        "config": args.config, "seed": seed, "model_params": int(count_params(model)),
        "var_g": var_g, "n_ep": n_ep, "meas_steps": meas, "warmup_steps": warmup,
        "tail_frac": tail_frac, "per_list": list(args.per),
        "seeds": {"online": seed + online_off, "eval": ev_seed, "calib": ca_seed},
        "calibration": {
            "n_usable": calib["n_usable"], "n_fit": calib["n_fit"], "n_test": calib["n_test"],
            "q_alpha": float(calib["conf_tab"]["q"]), "alpha": float(calib["conf_tab"]["alpha"]),
            "coverage_marginal": float(cov["marginal"]),
            "coverage_per_age_median": float(cov["per_age_median"]),
            "coverage_target": float(cov["target"]),
            "u_fit_quantiles_30_60_85": [float(x) for x in
                                         np.quantile(calib["u_fit_flat"], [0.3, 0.6, 0.85])],
            "z_fit_quantiles_50_70_85": [float(x) for x in
                                         np.quantile(calib["z_fit_flat"], [0.5, 0.7, 0.85])],
            "note": calib["note"]},
        "checks": {},
    }

    if args.mode in ("table", "both"):
        pa = part_a(runner, calib, args, list(args.per))
        payload["table"] = pa["rows"]
        payload["wiring"] = pa["wiring"]
        payload["checks"]["S_u1"] = pa["s_u1"]
        payload["checks"]["S_u4"] = pa["s_u4"]
        payload["checks"]["S_u2"] = {
            "seeds_disjoint": len({seed + online_off, ev_seed, ca_seed}) == 3,
            "online": seed + online_off, "eval": ev_seed, "calib": ca_seed}
        if not payload["checks"]["S_u2"]["seeds_disjoint"]:
            raise AssertionError("★ S_u2 失败：校准集与评测集 seed 有重叠")
        # 等预算配对（C2）：以 threshold 为参照族，逐 PER 对 mean_dist_tail 与 est_nmse
        def rows_of(fam: str, p: float, metric: str) -> list[scheduling.Row]:
            out = []
            for r in pa["rows"]:
                if r["family"] != fam or abs(r["loss_prob"] - p) > 1e-12:
                    continue
                rr = to_row(r)
                rr.metric = metric
                rr.metric_value = float(r["mean_dist_tail"] if metric == "mean_dist_tail"
                                       else r["est_nmse"])
                out.append(rr)
            return out

        pairs = {}
        for p in args.per:
            for metric in ("mean_dist_tail", "est_nmse"):
                b = rows_of("threshold", p, metric)
                for fam in scheduling.FAMILIES:
                    if fam == "threshold":
                        continue
                    a = rows_of(fam, p, metric)
                    if len(a) < 3 or len(b) < 3:
                        continue
                    key = f"{metric}::{fam}/threshold@p={p}"
                    try:
                        res = scheduling.equal_budget_pair(a, b, metric=metric)
                    except pairing.WindowNotEvaluable as e:
                        pairs[key] = {"error": str(e)}
                        print(f"[33] 等预算 {key}: **不可评估** ⇒ 不产出比值（{type(e).__name__}）")
                        continue
                    pairs[key] = {"ratio_median": res["ratio_median"],
                                  "n_grid_used": res["n_grid_used"],
                                  "n_grid_raw": res["n_grid_raw"],
                                  "window": [res["lo"], res["hi"]]}
                    print(f"[33] 等预算 {key}: ratio 中位={res['ratio_median']:.4f} "
                          f"（窗口 {res['lo']:.3f}–{res['hi']:.3f}, "
                          f"{res['n_grid_used']}/{res['n_grid_raw']} 点）")
        payload["pairings"] = pairs

    if args.mode in ("repro-x44", "both"):
        payload["repro_x44"] = part_b_repro(runner, args, cfg, env, ctrl, model)

    env.close()
    png = make_figure(payload.get("table", []), payload.get("repro_x44"), out, args.tag)
    jpath = os.path.join(str(out), f"{args.tag}.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[33] 产物：{png} / {jpath}   总耗时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
