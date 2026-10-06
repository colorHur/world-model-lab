# -*- coding: utf-8 -*-
"""X44 提前终止审计 —— **只读诊断，不改仿真、不覆盖任何已有结果**（2026-10-06）

================================================================= 为什么要这个脚本
外部审查（`成果归档与方向就业调研-2026-10-06.md` §3.3①）指出：`outputs/28_x44_timing.json`
里 T=16 的 `random#0 / #2` 有 `short_frac` = 6.67% / 3.33%、`n_tx` = 499 / 500
（基准 510），与实验记录里「正式跑**短集 0%**」「所有扰动**发送次数严格相同**」**矛盾**。

审查要求：**查清原因**、区分「发送次数 / 发送率 / 成功送达率」、**披露提前终止
对尾部距离指标的影响**，并且**保留原始 JSON**（不直接作废整轮）。

================================================================= 为什么可以直接量出来
X44 刻意取 **PER = 0**，而 `wmlab.eval.timing.lookup_schedule` 的 `_f(t, rng)`
**忽略 rng**（纯查表）⇒ 闭环里**没有**任何消耗随机数的信道环节
⇒ 给定 seed，闭环轨迹**完全确定** ⇒ 用**同一份模型缓存**重跑可**逐位复现**原 JSON。

复现成立之后，借 X45 已加的 `return_traces=True` 拿到**逐集逐点距离**
⇒ 提前终止对 `mean_dist_tail` 的贡献是**直接测量**，不是从聚合值反推
（这正是 X44 自己那条「把间接推断变成直接测量」的方法论）。

================================================================= 口径（三件必须分清的事）
- **标称尝试次数 `N`**：`n_tx_budget(T, meas_steps)`，由**构造**给出，同一 T 下**逐位相等**。
- **实测发送次数 `n_tx`**：`Σ_e #{1 ≤ t_k ≤ L_e}`（`L_e` = 第 e 集**实测**长度）。episode
  在 `dist > escape_radius(25 m)` 时 `terminated`（**跟丢**）⇒ `L_e < meas_steps`
  ⇒ 落在该集里的时刻变少 ⇒ **`n_tx` 会少**。PER=0 ⇒ 送达 = 尝试，故此处 `n_tx` 同时是
  「成功的发送次数」；**发送率** = `n_tx / n_steps`（`n_steps = Σ_e L_e`），分母同步缩水
  ⇒ 发送率**近似不变**（这才是「近似等发送率」而非「严格等次数」的准确含义）。
- **成功送达率**：PER=0 ⇒ 恒为 1；本实验**没有**丢包，故不能用它冒充全信道公平性验证。

★ 本脚本**只写** `outputs/_temp/`；**不触碰** `outputs/28_x44_timing.json`（保留原始数据）。
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from wmlab.control import (PDRelativeController, collect_controlled_episodes,  # noqa: E402
                           run_closed_loop_control)
from wmlab.envs import make_env  # noqa: E402
from wmlab.eval.timing import lookup_schedule, n_tx_budget, plan_times, subsample  # noqa: E402
from wmlab.models.prob_world_model import GaussianWorldModel  # noqa: E402
from wmlab.utils import get_device, load_config, output_dir, set_seed  # noqa: E402

CFG_PATH = "configs/uav_triggered.yaml"
CACHE = os.path.join(_ROOT, "outputs", "_temp", "x44_model.pt")
SRC_JSON = os.path.join(_ROOT, "outputs", "28_x44_timing.json")


def build_context():
    """复刻 `scripts/28_x44_timing.py` 的**正式跑**前置（seed=0 / 同一缓存 / 同一 var_g）。"""
    cfg = load_config(CFG_PATH)
    seed = int(cfg["seed"])
    set_seed(seed)
    device = get_device(cfg.get("device"))

    wind_amp = float(cfg["env"].get("wind_amp", 0.0))
    env = make_env(cfg["env"]["id"], seed=seed,
                   noise_std=float(cfg["env"]["noise_std"]),
                   max_steps=int(cfg["env"]["max_steps"]),
                   wind_amp=wind_amp)
    cc = cfg["controller"]
    ctrl = PDRelativeController(dt=env.dt, omega_n=float(cc["omega_n"]),
                                zeta=float(cc["zeta"]), kappa=float(env.kappa),
                                a_max=float(env.a_max))
    model = GaussianWorldModel(
        obs_dim=int(env.obs_dim), act_dim=int(env.act_dim),
        latent_dim=int(cfg["model"]["latent_dim"]),
        hidden=int(cfg["model"]["hidden"]),
        discrete_act=bool(env.is_discrete),
        logvar_init=float(cfg["model"].get("logvar_init", 0.0))).to(device)
    if not os.path.isfile(CACHE):
        raise FileNotFoundError(f"★ 模型缓存缺失 {CACHE} ⇒ 审计无法保证与正式跑逐位同源")
    model.load_state_dict(torch.load(CACHE, map_location=device))
    model.eval()

    # var_g 必须与 scripts/28 逐位同源（同 seed、同条数、同截断）
    WARMUP = int(cfg["task"]["warmup_steps"])
    meas_steps = int(cfg["task"]["measured_steps"])
    n_task_ep = 30
    ev_seed = seed + int(cfg["data"]["eval_seed_offset"])
    ev_env = make_env(cfg["env"]["id"], seed=ev_seed,
                      noise_std=float(cfg["env"]["noise_std"]),
                      max_steps=int(cfg["env"]["max_steps"]), wind_amp=wind_amp)
    eval_eps = collect_controlled_episodes(
        ev_env, ctrl, n_episodes=max(30, int(cfg["data"]["n_episodes"]) // 3),
        seed=ev_seed, max_steps=int(cfg["env"]["max_steps"]))
    ev_env.close()
    allobs = np.concatenate([e["obs"][WARMUP:] for e in eval_eps], axis=0).astype(np.float64)
    var_g = float(allobs.var())
    return dict(cfg=cfg, seed=seed, device=device, env=env, ctrl=ctrl, model=model,
                var_g=var_g, WARMUP=WARMUP, meas_steps=meas_steps, n_task_ep=n_task_ep,
                tail_frac=float(cfg["task"]["tail_frac"]),
                online_seed=seed + int(cfg["data"]["online_seed_offset"]))


def decompose(r, meas_steps, tail_frac):
    """逐集拆出「提前终止」对 `mean_dist_tail` 的贡献（只用 `return_traces` 的逐点距离）。"""
    lens = list(r["ep_lens"])
    epd = r["ep_dists"]
    assert len(lens) == len(epd), "ep_lens 与 ep_dists 条数不一致"
    tails, short_idx = [], []
    for e, (L, d) in enumerate(zip(lens, epd)):
        assert len(d) == L, f"第 {e} 集：trace 长度 {len(d)} ≠ ep_len {L}"
        k = max(1, int(round(len(d) * tail_frac)))
        tails.append(float(np.mean(d[-k:])))
        if L < meas_steps:
            short_idx.append(e)
    long_t = [tails[i] for i in range(len(tails)) if i not in set(short_idx)]
    return {
        "n_short": len(short_idx),
        "short_frac": len(short_idx) / max(len(lens), 1),
        "short_idx": short_idx,
        "ep_lens": lens,
        "tail_all_mean": float(np.mean(tails)),
        "tail_excl_short": float(np.mean(long_t)) if long_t else float("nan"),
        "n_long": len(long_t),
        "per_short": [{"ep": i, "L": lens[i], "tail": tails[i],
                       "final_dist": float(epd[i][-1]),
                       "max_dist": float(max(epd[i]))} for i in short_idx],
    }


def main():
    t0 = time.time()
    out = output_dir(load_config(CFG_PATH))
    tmp = out / "_temp"
    tmp.mkdir(parents=True, exist_ok=True)

    src = json.loads(open(SRC_JSON, encoding="utf-8").read())
    stored = {(r["kind"], int(r["T"]), r["mode"], int(r["param"])): r for r in src["rows"]}

    ctx = build_context()
    env, model, ctrl = ctx["env"], ctx["model"], ctx["ctrl"]
    meas_steps, tail_frac, WARMUP = ctx["meas_steps"], ctx["tail_frac"], ctx["WARMUP"]
    n_task_ep, online_seed = ctx["n_task_ep"], ctx["online_seed"]
    device, var_g = ctx["device"], ctx["var_g"]

    print("=" * 118)
    print("X44 提前终止审计（只读；模型缓存逐位同源；PER=0 ⇒ 时刻表纯查表 ⇒ 轨迹可复现）")
    print(f"  var_g = {var_g!r}   meas_steps={meas_steps}  warmup={WARMUP}  "
          f"n_task_ep={n_task_ep}  online_seed={online_seed}  device={device}")
    print(f"  原始 JSON：{SRC_JSON}（**不覆盖**）")
    print("=" * 118)

    def run(sched):
        return run_closed_loop_control(
            env, model, ctrl, sched, n_episodes=n_task_ep, seed=online_seed,
            max_steps=meas_steps, device=device, estimator="model",
            var_g=var_g, label=getattr(sched, "__name__", "sched"),
            tail_frac=tail_frac, warmup_steps=WARMUP, return_traces=True)

    targets = []      # (kind, T, mode, param, times)
    for T in [8, 16]:
        N = n_tx_budget(T, meas_steps)
        jmax = (T - 1) // 2
        targets.append(("base", T, "periodic", 0, plan_times("phase", T, 0, meas_steps, N)))
        for d in subsample(0, T - 1, 5):
            targets.append(("perturb", T, "phase", int(d), plan_times("phase", T, d, meas_steps, N)))
        for j in subsample(0, jmax, 5):
            targets.append(("perturb", T, "jitter", int(j), plan_times("jitter", T, j, meas_steps, N)))
        for p in range(3):
            targets.append(("perturb", T, "random", p,
                            plan_times("random", T, p, meas_steps, N, pattern_seed=p + 1)))

    entries, mismatches = [], []
    print(f"\n{'T':>3} {'kind':<8} {'mode':<9} {'param':>5} {'n_tx':>5} {'n_tx²':>5} "
          f"{'ep_len̄':>8} {'短集':>5} {'NMSE(新)':>10} {'NMSE(存)':>10} "
          f"{'dist(新)':>9} {'dist(存)':>9} {'Δdist%':>8} {'剔除后':>8}")
    for kind, T, mode, param, times in targets:
        name = f"periodic(T={T})" if kind == "base" else f"{mode}(T={T},{param})"
        r = run(lookup_schedule(times, name))
        # ★ 接线：n_tx 必须逐位 = Σ_e #{1 ≤ t_k ≤ L_e}
        n_tx_check = int(sum(sum(1 for x in times if 1 <= x <= int(L)) for L in r["ep_lens"]))
        d = decompose(r, meas_steps, tail_frac)
        ref = stored.get((kind, T, mode, param))
        nmse_new, dist_new = float(r["est_nmse"]), float(r["mean_dist_tail"])
        if ref is None:
            nmse_ref = dist_ref = float("nan")
        else:
            nmse_ref, dist_ref = float(ref["est_nmse"]), float(ref["mean_dist_tail"])
            if abs(nmse_new - nmse_ref) > 1e-12 or abs(dist_new - dist_ref) > 1e-12:
                mismatches.append((kind, T, mode, param, nmse_new, nmse_ref, dist_new, dist_ref))
        if n_tx_check != int(r["n_tx"]):
            mismatches.append((kind, T, mode, param, "n_tx", int(r["n_tx"]), n_tx_check))
        d_dist = (dist_new / dist_ref - 1.0) if np.isfinite(dist_ref) and dist_ref else float("nan")
        print(f"{T:>3} {kind:<8} {mode:<9} {param:>5} {int(r['n_tx']):>5} {n_tx_check:>5} "
              f"{float(np.mean(r['ep_lens'])):>8.2f} {d['n_short']:>5} "
              f"{nmse_new:>10.6f} {nmse_ref:>10.6f} {dist_new:>9.4f} {dist_ref:>9.4f} "
              f"{d_dist * 100:>7.2f}% {d['tail_excl_short']:>8.4f}")
        entries.append({
            "kind": kind, "T": T, "mode": mode, "param": param, "name": name,
            "n_tx": int(r["n_tx"]), "n_tx_check": n_tx_check,
            "n_tx_target": int(n_tx_budget(T, meas_steps)),
            "tx_rate": float(r["tx_rate"]),
            "ep_len_mean": float(np.mean(r["ep_lens"])),
            "est_nmse": nmse_new, "est_nmse_stored": nmse_ref,
            "mean_dist_tail": dist_new, "mean_dist_tail_stored": dist_ref,
            "mean_dist_tail_excl_short": d["tail_excl_short"],
            "n_short": d["n_short"], "short_frac": d["short_frac"], "n_long": d["n_long"],
            "per_short": d["per_short"],
            "reproduced_bit_exact": bool(ref is not None and abs(nmse_new - nmse_ref) <= 1e-12
                                         and abs(dist_new - dist_ref) <= 1e-12),
        })

    print("-" * 118)
    if mismatches:
        print(f"★ 与原始 JSON 不一致的条目 {len(mismatches)} 条：")
        for m in mismatches:
            print("   ", m)
    else:
        print("✓ 全部条目与 `28_x44_timing.json` **逐位一致**（NMSE 与 mean_dist_tail 均 ≤1e-12）"
              " ⇒ PER=0 复现成立，下面的分解可信")

    # ---- 汇总：提前终止只出现在哪些档 ----
    short_rows = [e for e in entries if e["n_short"] > 0]
    print(f"\n★ 有提前终止的档：{len(short_rows)} / {len(entries)}")
    for e in short_rows:
        base = stored[("base", e["T"], "periodic", 0)]
        bd = float(base["mean_dist_tail"])
        print(f"  · T={e['T']} {e['name']}: 短集 {e['n_short']}/30 ({e['short_frac']:.2%})  "
              f"n_tx={e['n_tx']}（基准 {int(base['n_tx'])}）  "
              f"dist {e['mean_dist_tail']:.4f} → 剔除短集后 {e['mean_dist_tail_excl_short']:.4f} "
              f"（基准 {bd:.4f}，即 {e['mean_dist_tail_excl_short'] / bd - 1:+.1%}）")
        for s in e["per_short"]:
            print(f"      └ 第 {s['ep']} 集：L={s['L']}（短 {meas_steps - s['L']} 步）"
                  f"  末点 dist={s['final_dist']:.2f} m  最大 {s['max_dist']:.2f} m  "
                  f"该集尾部均值 {s['tail']:.2f} m")

    payload = {
        "audit": "X44 提前终止审计（只读诊断）",
        "date": "2026-10-06",
        "source_json": SRC_JSON,
        "overwrote_source": False,
        "per": 0.0,
        "why_reproducible": "PER=0 ⇒ lookup_schedule 不消耗 rng ⇒ 闭环轨迹逐位确定",
        "var_g": var_g, "meas_steps": meas_steps, "warmup_steps": WARMUP,
        "n_task_ep": n_task_ep, "online_seed": online_seed,
        "reproduced_bit_exact": len(mismatches) == 0,
        "n_mismatches": len(mismatches),
        "entries": entries,
        "n_rows_with_early_term": len(short_rows),
    }
    dst = tmp / "x44_termination_audit.json"
    dst.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    env.close()
    print(f"\n[audit] 产物：{dst}    耗时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
