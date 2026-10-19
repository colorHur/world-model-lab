#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P3-1 · 失配边界图：`R_uav` 对**实测刷新陈旧度**扫描，定位"外推开始变毒"的位置。

口径来源（★ 全部在看结果之前冻结）
------------------------------------
`04-实验与代码/2026-10-20-P3-1-失配边界图-预注册-v0.1.md` §0–§9。

一句话
------
P3-0 给出**符号相反**的两格：B=200 unpaced `R_uav`=0.315（好）、B=100 unpaced 1.223（坏）。
本轮把横轴从"预算档"换成**实测的 `age` 均值**（被外推那一路 = UAV、`t≥41`、`alive`），
问：`R_uav` 是否随陈旧度单调上升、在哪个间隔上穿过 1。

★ 这是**刻画**，不是检验：不设通过线；非单调就如实写"机制解释不成立"，不许换横轴定义。
★ 方法冻结为 `Linear`（无可学参数），不加预测器、不改臂、不改记账。

设计（预注册 §5）
-----------------
    臂 = age；predictor ∈ {zoh, linear}；mode ∈ {unpaced, paced}
    开发集 {0,1,2}     × B ∈ {60,80,100,120,150,200}  = 72 cell（调图用）
    确认集 {100,130,160} × B ∈ {150,200}               = 24 cell（只复核，不参与选择）
    episodes=30, max_steps=200, absorbed=(25,1), GATE=0.90（只记录）

★ 横轴取 **`linear` 那条轨迹**的 `age`（外推发生的地方）；`zoh` 的同口径统计也照样记录，
  因为预测会改变轨迹 ⇒ 两条轨迹的陈旧度本来就不相等，必须能看见差多少。

用法
----
    D:/devtools/miniconda3/envs/ai-lab/python.exe scripts/59_p3_boundary.py
    ... --skip-experiment      # 只跑自检（不写 results/）
    ... --smoke                # 2 集 × 1 种子 × 2 预算，不写 results/

产物
----
    outputs/59_p3_boundary[_smoke].json              全量表（不入库）
    results/runs/<run_id>/59_p3_boundary.json        **入库**（results_io 唯一入口）
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

SMOKE_TAG = ""
RESULT_NAME = "59_p3_boundary"
sys.path.insert(0, str(ROOT))

from wmlab.eval.multiuser import default_uav_cost, delivery_rate, window_cost  # noqa: E402
from wmlab.eval.predictors import Linear, ZOH  # noqa: E402
from wmlab.eval.results_io import latest_result_path, write_result  # noqa: E402

FAILS: list[str] = []

#: ★ 预注册 §5：开发集只用来画图与定判定，确认集只复核
DEV_SEEDS = (0, 1, 2)
CONF_SEEDS = (100, 130, 160)
BUDGETS_DEV = (60, 80, 100, 120, 150, 200)
BUDGETS_CONF = (150, 200)
MODES = ("unpaced", "paced")
PREDICTORS = ("zoh", "linear")
#: ★ 元数据必须由**实际跑的那一组**生成（P0-2 的教训）：main() 里覆盖
RUN_MODES = MODES
RUN_BUDGETS_DEV = BUDGETS_DEV
RUN_PREDICTORS = PREDICTORS
RUN_REFINE = False
N_EP = 30

#: ★ 预注册 §3：横轴窗口下界（避开预热后的爬升段）与"被外推那一路"的用户号
AGE_T_MIN = 41
UAV = 0


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


T12 = _load(ROOT / "scripts" / "42_t12_hetero_four_arms.py", "t12_mod59")
T52 = _load(ROOT / "scripts" / "52_t21_paced_matching_ablation.py", "t52_mod59")

WINDOW = T12.WINDOW
WARMUP_UAV = T12.WARMUP_UAV
GATE = T12.GATE
ABSORBED = [T12.ABSORBED_UAV, T12.ABSORBED_CART]
COSTS = [default_uav_cost, T12.cartpole_cost]
ARM = "age"
N_BOOT = 10000
BOOT_SEED = 20261020


def check(tag: str, ok: bool, detail: str) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {tag}: {detail}")
    if not ok:
        FAILS.append(f"{tag}: {detail}")


# ================================================================ 装置（与 58 同源，复制而非改 58）
def _run(budget, mode, seed, n_ep, *, predictor=None, probe=None):
    envs = [T12.make_env("uav", seed=seed, max_steps=WARMUP_UAV + WINDOW),
            T12.make_env("CartPole-v1", seed=seed)]
    ctrls = [T12.PDRelativeController(), T12.CartPoleLQR(_run.K)]
    alloc = T52.make_alloc(ARM, mode, budget)
    try:
        return T12.run_multiuser(
            envs, ctrls, alloc, T12.SINRSuccess(T12.Q_REF),
            n_episodes=n_ep, seed=seed, max_steps=WINDOW,
            warmup_steps=[WARMUP_UAV, 0], budget_total=int(budget),
            csi=T12.make_csi(seed, n_ep), cost_fn=COSTS, diagnostic=False,
            counterfactual=False, predictor=predictor, pred_probe=probe)
    finally:
        for e in envs:
            e.close()


_run.K = T12.load_lqr_gain()


def _J(records, n_ep):
    return np.asarray(window_cost(records, 2, window=WINDOW, absorbed=ABSORBED,
                                  n_episodes=n_ep), dtype=float)


class _AgeProbe:
    """逐决策点采 `age`（**决策前**的值，与 `StepRecord.ages` 同一时点）。

    ★ 不只看值：同时保留完整的 `(episode, t, age, delivered, alive)` 序列，
      供 E2 的递推恒等式检查使用（"采在错误时点"必须能被机器打死）。
    """

    def __init__(self) -> None:
        self.seq: list[tuple[int, int, int, bool, bool, int]] = []   # (ep, t, user, age, deliv, alive)
        self._uf = np.zeros(2, dtype=np.int64)
        self._un = np.zeros(2, dtype=np.int64)

    def __call__(self, p: dict) -> None:
        e = int(p["episode"])
        t = int(p["t"])
        for i in (0, 1):
            alive = bool(p["alive"][i])
            dl = bool(p["delivered"][i])
            a = int(p["age"][i])
            self.seq.append((e, t, i, a, dl, alive))
            self._un[i] += 1
            if dl:
                self._uf[i] += 1

    # ---------- 横轴（预注册 §3）----------
    def axis(self, user: int = UAV, t_min: int = AGE_T_MIN) -> dict:
        rows = [(e, t, a, dl) for (e, t, u, a, dl, al) in self.seq
                if u == user and al and t >= t_min]
        if not rows:
            return {"n": 0}
        a = np.asarray([r[2] for r in rows], dtype=float)
        stale = a[a >= 1]
        # 逐集先取均值，再对集取中位（不依赖"合并 30 集"的对照口径）
        per_ep: dict[int, list] = {}
        for e, _t, x, _dl in rows:
            per_ep.setdefault(e, []).append(x)
        ep_means = np.asarray([float(np.mean(v)) for v in per_ep.values()], dtype=float)
        return {
            "n": int(a.size),
            "mean": float(a.mean()),                       # ★ 主轴
            "p50": float(np.percentile(a, 50)),
            "p90": float(np.percentile(a, 90)),
            "stale_mean": (float(stale.mean()) if stale.size else None),
            "n_stale": int(stale.size),
            "ep_median_of_means": float(np.median(ep_means)),
            "n_episodes_used": int(ep_means.size),
            "frac_age0": float(np.mean(a == 0)),
        }

    def delivered_count(self, user: int) -> int:
        return int(self._uf[user])

    def decision_count(self, user: int) -> int:
        return int(self._un[user])


def cluster_bootstrap(rows, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    """`R_i = Σ J_i(num) / Σ J_i(den)` 的 cluster bootstrap（重采样单位 = 不同初值）。

    ★ 与 `scripts/58` 逐行同源（复制，不改 58 —— 它是已入库证据的生成器）。
    """
    cl = sorted({int(r["cluster"]) for r in rows})
    idx_by_cl = {c: [] for c in cl}
    for k, r in enumerate(rows):
        idx_by_cl[int(r["cluster"])].append(k)
    maxs = max(len(v) for v in idx_by_cl.values())
    mat = np.full((len(cl), maxs), -1, dtype=np.int64)
    for k, c in enumerate(cl):
        mat[k, :len(idx_by_cl[c])] = idx_by_cl[c]
    Jn = np.asarray([r["J_num"] for r in rows], dtype=float)
    Jd = np.asarray([r["J_den"] for r in rows], dtype=float)

    def _ratio(ii):
        dn = Jd[ii].sum(axis=0)
        safe = np.where(dn > 0, dn, 1.0)
        return np.where(dn > 0, Jn[ii].sum(axis=0) / safe, np.nan)

    point = _ratio(np.arange(len(rows), dtype=np.int64))
    rng = np.random.default_rng(int(seed))
    draws = np.full((n_boot, 2), np.nan)
    n_bad = 0
    for b in range(n_boot):
        pick = rng.integers(0, len(cl), len(cl))
        ii = mat[pick].ravel()
        v = _ratio(ii[ii >= 0])
        if not np.all(np.isfinite(v)):
            n_bad += 1
        draws[b] = v
    ci_lo = np.nanpercentile(draws, 2.5, axis=0)
    ci_hi = np.nanpercentile(draws, 97.5, axis=0)
    per_ep = np.where(Jd > 0, Jn / np.where(Jd > 0, Jd, 1.0), np.nan)
    return {"R": [float(x) for x in point],
            "ci95": [[float(x) for x in ci_lo], [float(x) for x in ci_hi]],
            "geo_R": float(np.sqrt(point[0] * point[1])),
            "n_clusters": len(cl), "n_rows": len(rows), "n_boot": int(n_boot),
            "n_boot_nonfinite": int(n_bad),
            "per_episode_ratio_median": [float(np.nanmedian(per_ep[:, i])) for i in (0, 1)],
            "boot_seed": int(seed)}


# ================================================================ 单调性判定器（预注册 §6-F1）
def kendall_tau_b(xs, ys) -> dict:
    """Kendall τ-b + 逆序计数。**只用于描述顺序**，不当显著性检验。"""
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    n = x.size
    nc = nd = tx = ty = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[j] - x[i]
            dy = y[j] - y[i]
            if dx == 0 and dy == 0:
                tx += 1
                ty += 1
            elif dx == 0:
                tx += 1
            elif dy == 0:
                ty += 1
            elif (dx > 0) == (dy > 0):
                nc += 1
            else:
                nd += 1
    n0 = n * (n - 1) / 2.0
    den = np.sqrt(max(n0 - tx, 0.0) * max(n0 - ty, 0.0))
    tau = float((nc - nd) / den) if den > 0 else float("nan")
    # 相邻逆序：按 x 排序（同 x 视为并列，不比较）后看相邻对的 y 是否下降
    order = np.argsort(x, kind="stable")
    xs_s, ys_s = x[order], y[order]
    adj_inv = 0
    for k in range(n - 1):
        if xs_s[k + 1] > xs_s[k] and ys_s[k + 1] < ys_s[k]:
            adj_inv += 1
    return {"tau_b": tau, "n_concordant": nc, "n_discordant": nd,
            "n_tie_x": tx, "n_tie_y": ty, "n_adjacent_inversions": adj_inv, "n": n}


def verdict(tau: float) -> str:
    if not np.isfinite(tau):
        return "UNDEFINED"
    if tau >= 0.6:
        return "MONOTONE"
    if tau <= -0.6:
        return "ANTI-MONOTONE"
    return "WEAK"


def crossings(points) -> list[dict]:
    """`R_uav` 穿过 1 的相邻区间（线性插值）。points 需已按 x 排序、x 不重复。"""
    out = []
    for k in range(len(points) - 1):
        a, b = points[k], points[k + 1]
        ya, yb = a["R_uav"], b["R_uav"]
        if (ya - 1.0) == 0.0:
            out.append({"x": a["x"], "kind": "exact", "between": [a["x"], b["x"]]})
        elif (ya - 1.0) * (yb - 1.0) < 0:
            w = (1.0 - ya) / (yb - ya)
            out.append({"x": float(a["x"] + w * (b["x"] - a["x"])), "kind": "interpolated",
                        "between": [a["x"], b["x"]]})
    return out


# ================================================================ 自检
def selfchecks(n_ep_small: int = 2) -> dict:
    print("\n== 自检 E1–E7（预注册 §7：每条写明能打死什么错）==")
    out: dict = {}

    # ---------------- E1：None ≡ ZOH ----------------
    j_none = _J(_run(200, "unpaced", 0, n_ep_small, predictor=None)["records"], n_ep_small)
    j_zoh = _J(_run(200, "unpaced", 0, n_ep_small, predictor=ZOH())["records"], n_ep_small)
    d1 = float(np.max(np.abs(j_none - j_zoh))) if j_none.size else 0.0
    check("E1 `predictor=None` 与 `ZOH` 逐 episode J **逐位相同**", d1 == 0.0,
          f"max|Δ|={d1:.3e}")

    # ---------------- E2：age 递推恒等式（★ 时点检查）----------------
    pr = _AgeProbe()
    r2 = _run(100, "unpaced", 0, n_ep_small, predictor=Linear(), probe=pr)
    by: dict[tuple[int, int], list] = {}
    for (e, t, u, a, dl, al) in pr.seq:
        by.setdefault((e, u), []).append((t, a, dl, al))
    bad_rec = 0
    checked = 0
    for (e, u), v in by.items():
        v.sort()
        for k in range(len(v) - 1):
            t, a, dl, al = v[k]
            t2, a2, _dl2, al2 = v[k + 1]
            if not (al and al2) or t2 != t + 1:
                continue
            checked += 1
            want = 0 if dl else a + 1
            if a2 != want:
                bad_rec += 1
    # 集合恒等式：`{t : age_pre[t]==0, t>首点} == {d+1 : 在 d 送达且 d+1 仍是存活决策点}`
    # ★ 2026-10-20 07:1x 首版写成 `count(age==0) == count(delivered)`，报了两轮假 FAIL：
    #   ① 每集首个决策点的 `age=0` 来自预热初态，与送达无关（差数恰好 = 集数）；
    #   ② **`age==0` 出现在送达的下一步**，写成同一步比较是序号基准错（#52 同族）。
    #   两处都跑在全量之前，且都只改自检、不改被测代码。
    eq_cnt = {}
    for u in (0, 1):
        alive_t: dict[int, set] = {}
        dl_t: dict[int, set] = {}
        zero_t: dict[int, set] = {}
        for (e, t, uu, a, dl, al) in pr.seq:
            if uu != u:
                continue
            if al:
                alive_t.setdefault(e, set()).add(t)
                if a == 0:
                    zero_t.setdefault(e, set()).add(t)
                if dl:
                    dl_t.setdefault(e, set()).add(t)
        bad_set = 0
        for e, ts in alive_t.items():
            first = min(ts)
            want = {d + 1 for d in dl_t.get(e, set()) if (d + 1) in ts}
            got = {t for t in zero_t.get(e, set()) if t > first}
            if want != got:
                bad_set += 1
        eq_cnt[str(u)] = {"n_episodes": len(alive_t), "n_episodes_set_mismatch": bad_set,
                          "n_age0": sum(len(v) for v in zero_t.values()),
                          "n_delivered": sum(len(v) for v in dl_t.values())}
    ok2 = (bad_rec == 0 and checked > 0
           and all(v["n_episodes_set_mismatch"] == 0 for v in eq_cnt.values()))
    check("E2 `age_pre[t+1] == 0 if delivered[t] else age_pre[t]+1` 逐点成立，且"
          "「`age==0` 的时点集合 == 送达时点 +1（排除每集首点）」逐集相等"
          "（打死「`age` 采在错误时点」与序号基准错）",
          ok2, f"checked={checked} violations={bad_rec}｜{json.dumps(eq_cnt)}")
    out["E2"] = {"checked": checked, "violations": bad_rec, "counts": eq_cnt}

    # ---------------- E6：单调性判定器有分辨力（★ 先测工具，再测数据）----------------
    e6 = {}
    xs = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    e6["up"] = kendall_tau_b(xs, [0.2, 0.3, 0.5, 0.7, 0.9, 1.2])
    e6["down"] = kendall_tau_b(xs, [1.2, 0.9, 0.7, 0.5, 0.3, 0.2])
    e6["shuffle"] = kendall_tau_b(xs, [0.5, 1.2, 0.3, 0.9, 0.2, 0.7])
    ok6 = (verdict(e6["up"]["tau_b"]) == "MONOTONE"
           and verdict(e6["down"]["tau_b"]) == "ANTI-MONOTONE"
           and verdict(e6["shuffle"]["tau_b"]) == "WEAK")
    check("E6 判定器有分辨力：严格上升⇒MONOTONE、严格下降⇒ANTI-MONOTONE、乱序⇒WEAK",
          ok6, "；".join(f"{k}:τ={v['tau_b']:+.3f}→{verdict(v['tau_b'])}" for k, v in e6.items()))
    out["E6"] = {k: {**v, "verdict": verdict(v["tau_b"])} for k, v in e6.items()}
    # 交点器分辨力：已知 1.0 落在 x∈[2,4] 中间
    cr = crossings([{"x": 1.0, "R_uav": 0.5}, {"x": 2.0, "R_uav": 0.8},
                    {"x": 4.0, "R_uav": 1.4}, {"x": 8.0, "R_uav": 1.9}])
    ok6b = len(cr) == 1 and abs(cr[0]["x"] - 2.6666667) < 1e-6
    cr2 = crossings([{"x": 1.0, "R_uav": 0.5}, {"x": 2.0, "R_uav": 0.9}])
    check("E6b 交点器：0.8→1.4 在 [2,4] 上给出 2.667；全程 <1 时不给交点", ok6b and cr2 == [],
          f"{cr}｜无交点={cr2 == []}")
    out["E6b"] = {"crossings": cr, "none_when_never_above": cr2 == []}

    # ---------------- E7：bootstrap 点估计 == 直接 np.sum ----------------
    rows = [{"cluster": k // 4, "J_num": [1.0 + k * 0.1, 2.0 + k * 0.2],
             "J_den": [2.0 + k * 0.1, 4.0 + k * 0.2]} for k in range(40)]
    bc = cluster_bootstrap(rows, n_boot=200)
    dir_r = [(sum(r["J_num"][i] for r in rows)) / (sum(r["J_den"][i] for r in rows))
             for i in (0, 1)]
    rows0 = [{"cluster": k // 3, "J_num": [0.5 * (1 + k), 0.5 * (2 + k)],
              "J_den": [1.0 + k, 2.0 + k]} for k in range(30)]
    b0 = cluster_bootstrap(rows0, n_boot=300)
    w0 = float(max(b0["ci95"][1][i] - b0["ci95"][0][i] for i in (0, 1)))
    ok7 = (max(abs(bc["R"][i] - dir_r[i]) for i in (0, 1)) < 1e-12 and w0 < 1e-12)
    check("E7 bootstrap：点估计==直接 np.sum 比值；比值恒 0.5 ⇒ 区间宽度 0",
          ok7, f"max|Δ|={max(abs(bc['R'][i] - dir_r[i]) for i in (0, 1)):.3e} width={w0:.3e}")

    # ---------------- E4：记账 ----------------
    r4 = _run(100, "unpaced", 0, n_ep_small, predictor=ZOH())
    served_total = sum(1 for x in r4["records"] if x.attempted_idx is not None
                       and x.attempted_idx >= 0)
    late_un = sum(1 for x in r4["records"] if x.t > 100 and x.attempted_idx is not None
                  and x.attempted_idx >= 0)
    r4p = _run(100, "paced", 0, n_ep_small, predictor=ZOH())
    late_pa = sum(1 for x in r4p["records"] if x.t > 100 and x.attempted_idx is not None
                  and x.attempted_idx >= 0)
    ok4 = (served_total == n_ep_small * 100
           and r4["budget_reserved"] == r4["budget_attempted"] + r4["budget_wasted"]
           and late_un == 0 and late_pa > 0)
    check("E4 记账：served_total==集数×预算、reserved==attempted+wasted、"
          "unpaced B<L 的 t>B 发帧=0、paced >0", ok4,
          f"served={served_total}/{n_ep_small * 100} late_un={late_un} late_paced={late_pa}")
    out["E4"] = {"served_total": served_total, "late_unpaced": late_un, "late_paced": late_pa}
    return out


# ================================================================ 单 cell
def run_cell(budget, mode, pred_name, seed, n_ep):
    t0 = time.time()
    pred = Linear() if pred_name == "linear" else ZOH()
    probe = _AgeProbe()
    r = _run(budget, mode, seed, n_ep, predictor=pred, probe=probe)
    rec = r["records"]
    J = _J(rec, n_ep)
    dr = delivery_rate(rec, 2)
    served = [sum(1 for x in rec if x.attempted_idx == i) for i in range(2)]
    late = [sum(1 for x in rec if x.t > budget and x.attempted_idx == i) for i in range(2)]
    return {
        "budget": int(budget), "mode": mode, "predictor": pred_name, "seed": int(seed),
        "J_per_episode": np.asarray(J, dtype=float).tolist(),
        "J_mean": np.asarray(J, dtype=float).mean(axis=0).tolist(),
        "served": served, "served_late": late, "served_total": int(sum(served)),
        "alive_steps": dr["alive_steps"], "frames": dr["frames"], "attempts": dr["attempts"],
        "link_q": dr["link_q"], "p_eff": dr["p_eff"],
        "budget_reserved": dr["budget_reserved"], "budget_attempted": dr["budget_attempted"],
        "budget_wasted": dr["budget_wasted"],
        "age_uav": probe.axis(UAV), "age_cart": probe.axis(1),
        "delivered_count": [probe.delivered_count(0), probe.delivered_count(1)],
        "elapsed_s": round(time.time() - t0, 2),
    }


# ================================================================ 主实验
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=N_EP)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip-experiment", action="store_true")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--allow-overwrite", action="store_true")
    ap.add_argument("--budgets", default=None, help="覆盖开发集预算档，如 160,175,190")
    ap.add_argument("--modes", default=None, help="覆盖模式，如 unpaced")
    ap.add_argument("--refine", action="store_true",
                    help="分辨率补充：不跑确认集、不算 F1/F2/F3 判词（判据已在主轮固定）")
    ap.add_argument("--result-name", default="59_p3_boundary")
    args = ap.parse_args()
    global SMOKE_TAG, RESULT_NAME, RUN_MODES, RUN_BUDGETS_DEV, RUN_PREDICTORS, RUN_REFINE
    SMOKE_TAG = "_smoke" if args.smoke else ""
    RESULT_NAME = args.result_name
    modes = tuple(args.modes.split(",")) if args.modes else MODES
    RUN_MODES, RUN_PREDICTORS = modes, PREDICTORS
    n_ep = 2 if args.smoke else args.episodes
    if n_ep != N_EP and not args.smoke:
        print(f"[WARN] episodes={n_ep} ≠ 预注册的 {N_EP}：结果不得标「确认」，只能标「探索」")
    sets = ({"dev": (DEV_SEEDS[0],)} if args.smoke else
            ({"dev": DEV_SEEDS} if args.refine else
             {"dev": DEV_SEEDS, "conf": CONF_SEEDS}))
    budgets_dev = (100, 200) if args.smoke else (
        tuple(int(x) for x in args.budgets.split(",")) if args.budgets else BUDGETS_DEV)
    RUN_BUDGETS_DEV, RUN_REFINE = budgets_dev, bool(args.refine)

    t_all = time.time()
    print(f"[59] P3-1 失配边界图｜窗口 {WINDOW}｜臂={ARM}｜episodes={n_ep}｜"
          f"横轴=UAV age mean @t≥{AGE_T_MIN}｜mode={modes}｜pred={PREDICTORS}")
    sc = selfchecks()

    if args.skip_experiment:
        _dump({"selfchecks": sc}, None, None, n_ep, sets)
        print(f"\n{'ALL CHECKS PASS' if not FAILS else 'FAILED: ' + '; '.join(FAILS)}")
        return 0 if not FAILS else 2

    cells: list[dict] = []
    for set_name, seeds in sets.items():
        for budget in (budgets_dev if set_name == "dev" else BUDGETS_CONF):
            for mode in modes:
                for pname in PREDICTORS:
                    for seed in seeds:
                        c = run_cell(budget, mode, pname, seed, n_ep)
                        c["seed_set"] = set_name
                        cells.append(c)
                        ax = c["age_uav"]
                        print(f"  [{set_name}] B={budget:3d} {mode:7s} {pname:6s} seed={seed:3d} "
                              f"J=({c['J_mean'][0]:.3f},{c['J_mean'][1]:.3f}) "
                              f"x={ax.get('mean', float('nan')):.2f} "
                              f"p50={ax.get('p50', float('nan')):.1f} "
                              f"late=({c['served_late'][0]},{c['served_late'][1]}) "
                              f"{c['elapsed_s']}s")

    def get(set_name, budget, mode, pname, seed):
        for c in cells:
            if (c["seed_set"], c["budget"], c["mode"], c["predictor"], c["seed"]) == \
                    (set_name, budget, mode, pname, seed):
                return c
        return None

    def table(set_name, seeds, budget, mode, num="linear", den="zoh"):
        rows = []
        for s in seeds:
            a, b = get(set_name, budget, mode, num, s), get(set_name, budget, mode, den, s)
            if a is None or b is None:
                continue
            for e in range(len(a["J_per_episode"])):
                rows.append({"cluster": int(s) + e,
                             "J_num": a["J_per_episode"][e], "J_den": b["J_per_episode"][e]})
        if not rows:
            return None
        d = cluster_bootstrap(rows)
        per_seed = {}
        for s in seeds:
            a = np.asarray(get(set_name, budget, mode, num, s)["J_mean"], dtype=float)
            b = np.asarray(get(set_name, budget, mode, den, s)["J_mean"], dtype=float)
            per_seed[str(s)] = [float(x) for x in (a / b)] if np.all(b > 0) else [float("nan")] * 2
        d["per_seed_R"] = per_seed
        rs = np.asarray(list(per_seed.values()), dtype=float)
        d["n_seeds_uav_below_1"] = int(np.sum(rs[:, 0] < 1.0))
        d["n_seeds"] = int(rs.shape[0])
        # 横轴（取 linear 那条轨迹；zoh 的同口径另存，用于看轨迹分叉）
        ax = [get(set_name, budget, mode, p, s)["age_uav"] for s in seeds for p in (num,)]
        az = [get(set_name, budget, mode, "zoh", s)["age_uav"] for s in seeds]
        # ★ 用 .get(…, nan)：UAV 若在 t≥41 前就死，axis() 返回 {"n":0}，不许因此 KeyError
        d["x_linear"] = {k: float(np.nanmean([a.get(k, float("nan")) for a in ax])) for k in
                         ("mean", "p50", "p90", "stale_mean", "ep_median_of_means")}
        d["x_zoh"] = {k: float(np.nanmean([a.get(k, float("nan")) for a in az])) for k in
                      ("mean", "p50", "p90", "stale_mean", "ep_median_of_means")}
        d["age_uav_n"] = [int(a.get("n", 0)) for a in ax]
        d["served"] = {p: [float(np.mean([get(set_name, budget, mode, p, s)["served"][i]
                                           for s in seeds])) for i in (0, 1)]
                       for p in (num, den)}
        d["alive"] = {p: [float(np.mean([get(set_name, budget, mode, p, s)["alive_steps"][i]
                                          for s in seeds])) for i in (0, 1)]
                      for p in (num, den)}
        d["late"] = {p: float(np.mean([sum(get(set_name, budget, mode, p, s)["served_late"])
                                       for s in seeds])) for p in (num, den)}
        d["J"] = {p: [float(np.mean([get(set_name, budget, mode, p, s)["J_mean"][i]
                                     for s in seeds])) for i in (0, 1)]
                  for p in (num, den)}
        return d

    summary: dict = {}
    for set_name, seeds in sets.items():
        summary[set_name] = {}
        for budget in (budgets_dev if set_name == "dev" else BUDGETS_CONF):
            summary[set_name][str(budget)] = {m: table(set_name, seeds, budget, m) for m in modes}

    # ---------- 主表：开发集 12 点（按 x 排序）----------
    print("\n== 主轴（开发集 {0,1,2}：x=UAV age mean @t≥41 于 linear 轨迹；R_uav=Σ J_lin/Σ J_zoh）==")
    pts = []
    for budget in budgets_dev:
        for mode in modes:
            d = summary["dev"][str(budget)][mode]
            if d is None:
                continue
            pts.append({"budget": budget, "mode": mode, "x": d["x_linear"]["mean"],
                        "x_p50": d["x_linear"]["p50"], "x_p90": d["x_linear"]["p90"],
                        "x_zoh": d["x_zoh"]["mean"],
                        "R_uav": d["R"][0], "R_cart": d["R"][1],
                        "ci_uav": [d["ci95"][0][0], d["ci95"][1][0]],
                        "ci_cart": [d["ci95"][0][1], d["ci95"][1][1]],
                        "seeds_uav_below_1": f"{d['n_seeds_uav_below_1']}/{d['n_seeds']}",
                        "late": d["late"]["linear"], "late_zoh": d["late"]["zoh"],
                        "served": d["served"]["linear"], "served_zoh": d["served"]["zoh"],
                        "alive": d["alive"]["linear"], "alive_zoh": d["alive"]["zoh"],
                        "J": d["J"]["linear"], "J_zoh": d["J"]["zoh"]})
    pts.sort(key=lambda r: (r["x"], r["mode"]))
    for r in pts:
        print(f"  B={r['budget']:3d} {r['mode']:7s} x={r['x']:7.3f} (p50={r['x_p50']:5.1f} "
              f"p90={r['x_p90']:6.1f} zoh_x={r['x_zoh']:6.2f}) "
              f"R_uav={r['R_uav']:.3f} CI[{r['ci_uav'][0]:.3f},{r['ci_uav'][1]:.3f}] "
              f"R_cart={r['R_cart']:.3f} late={r['late']:.0f} alive=({r['alive'][0]:.0f},"
              f"{r['alive'][1]:.0f})")

    # ---------- F1：单调性 ----------
    print("\n== F1（单调性判定：主轴 x = UAV age mean；12 点，含 B=200 的 paced/unpaced 重复）==")
    f1: dict = {}
    if RUN_REFINE:
        print("  [SKIP] --refine：判据已在主轮（12 点）固定；补点只提高 F2 的分辨率，不改 F1"
              "（预注册 §4b-1）")
        f1 = {"verdict_full": "SKIP-REFINE", "skipped": True}
    elif len(pts) >= 4:
        kt = kendall_tau_b([r["x"] for r in pts], [r["R_uav"] for r in pts])
        # 去重（同一 x 只留一个，用于"有效点数"）
        seen, uniq = set(), []
        for r in pts:
            k = round(r["x"], 9)
            if k in seen:
                continue
            seen.add(k)
            uniq.append(r)
        kt_u = kendall_tau_b([r["x"] for r in uniq], [r["R_uav"] for r in uniq]) \
            if len(uniq) >= 4 else None
        f1 = {"tau_full": kt, "verdict_full": verdict(kt["tau_b"]), "n_points": len(pts),
              "tau_dedup": kt_u, "verdict_dedup": (verdict(kt_u["tau_b"]) if kt_u else None),
              "n_points_dedup": len(uniq),
              "x_min": float(min(r["x"] for r in pts)),
              "x_max": float(max(r["x"] for r in pts)),
              "x_span_ratio": float(max(r["x"] for r in pts) /
                                    max(min(r["x"] for r in pts), 1e-9))}
        print(f"  τ_b(12 点)={kt['tau_b']:+.3f} ⇒ **{f1['verdict_full']}**"
              f"（一致 {kt['n_concordant']} / 逆序 {kt['n_discordant']} / 相邻逆序 "
              f"{kt['n_adjacent_inversions']}）")
        if kt_u:
            print(f"  τ_b(去重 {len(uniq)} 点)={kt_u['tau_b']:+.3f} ⇒ {f1['verdict_dedup']}")
        print(f"  横轴跨度：x∈[{f1['x_min']:.3f}, {f1['x_max']:.3f}]，max/min={f1['x_span_ratio']:.2f}")
        # 稳健性（★ 只报不判）：把横轴换成 **ZOH 轨迹** 的陈旧度（外生于预测本身）。
        # 理由：x 是在"被处理过"的轨迹上量的（预测会改存活 ⇒ 改服务分摊 ⇒ 改 x），
        # 这是内生性；若两条轴给出同样的顺序，结论就不是"轴挑出来的"。
        kz = kendall_tau_b([r["x_zoh"] for r in pts], [r["R_uav"] for r in pts])
        f1["tau_zoh_axis_report_only"] = {**kz, "verdict": verdict(kz["tau_b"])}
        print(f"  [稳健性·不判] 横轴换成 zoh 轨迹：τ_b={kz['tau_b']:+.3f}"
              f"⇒{verdict(kz['tau_b'])}（一致 {kz['n_concordant']}/逆序 {kz['n_discordant']}）")
    else:
        print("  [SKIP] 点数不足（smoke）")
        f1 = {"verdict_full": "SKIP"}

    # ---------- F2：交点 ----------
    print("\n== F2（R_uav 穿过 1 的位置，相邻点线性插值）==")
    f2: dict = {}
    for mode in modes:
        sub = [r for r in pts if r["mode"] == mode]
        cr = crossings(sub)
        f2[mode] = {"crossings": cr, "x_range": [sub[0]["x"], sub[-1]["x"]] if sub else None,
                    "R_range": [min(r["R_uav"] for r in sub), max(r["R_uav"] for r in sub)]
                    if sub else None}
        print(f"  {mode:7s} x∈[{f2[mode]['x_range'][0]:.3f},{f2[mode]['x_range'][1]:.3f}] "
              f"R∈[{f2[mode]['R_range'][0]:.3f},{f2[mode]['R_range'][1]:.3f}] ⇒ {cr or '窗口内未穿 1'}")

    # ---------- F3：确认集复核 ----------
    print("\n== F3（确认集 {100,130,160} 只复核 B∈{150,200} 的落点一致性）==")
    f3: dict = {}
    if "conf" in summary:
        for budget in BUDGETS_CONF:
            for mode in modes:
                a = summary["conf"][str(budget)][mode]
                b = summary["dev"][str(budget)][mode] if str(budget) in summary["dev"] else None
                if a is None or b is None:
                    continue
                f3[f"{budget}/{mode}"] = {
                    "R_conf": a["R"], "R_dev": b["R"],
                    "d_uav": a["R"][0] - b["R"][0],
                    "x_conf": a["x_linear"]["mean"], "x_dev": b["x_linear"]["mean"],
                    "seeds_uav_below_1_conf": f"{a['n_seeds_uav_below_1']}/{a['n_seeds']}",
                    "late_conf": a["late"]["linear"]}
                print(f"  B={budget:3d} {mode:7s} R_conf=({a['R'][0]:.3f},{a['R'][1]:.3f}) "
                      f"R_dev=({b['R'][0]:.3f},{b['R'][1]:.3f}) Δuav={a['R'][0] - b['R'][0]:+.3f} "
                      f"x_conf={a['x_linear']['mean']:.2f} x_dev={b['x_linear']['mean']:.2f} "
                      f"late={a['late']['linear']:.0f}")
    else:
        print("  [SKIP] smoke 未跑确认集")

    # ---------- E3：横轴有分辨力（主表级）----------
    print("\n== E3（横轴有分辨力：不是常数）==")
    xs_all = [c["age_uav"].get("mean") for c in cells if c["age_uav"].get("n", 0) > 0]
    xs_all = [float(x) for x in xs_all if x is not None]
    n_uniq = len({round(x, 6) for x in xs_all})
    ratio = (max(xs_all) / max(min(xs_all), 1e-9)) if xs_all else float("nan")
    if args.smoke:
        print(f"  [SKIP] smoke 网格只有 {len(xs_all)} cell（distinct={n_uniq} span={ratio:.2f}）"
              f"⇒ 「≥8 个不同值」这条门在冒烟规模下不可判")
        e3 = {"skipped": f"smoke: {len(xs_all)} cells", "distinct": n_uniq, "span": ratio}
    else:
        need = min(8, max(3, len(xs_all) // 4))
        check(f"E3 全部 cell 的 x 至少 {need} 个不同值，且 max/min ≥ 1.5",
              n_uniq >= need and ratio >= 1.5, f"distinct={n_uniq} span={ratio:.2f}")
        e3 = {"distinct": n_uniq, "span": ratio, "n_cells": len(xs_all)}

    # ---------- E4（主表级）----------
    bad1 = [c for c in cells if c["served_total"] != n_ep * c["budget"]]
    bad2 = [c for c in cells if c["budget_reserved"]
            != c["budget_attempted"] + c["budget_wasted"]]
    bad3 = [c for c in cells if c["mode"] == "unpaced" and c["budget"] < WINDOW
            and sum(c["served_late"]) != 0]
    bad4 = [c for c in cells if c["mode"] == "paced" and c["budget"] < WINDOW
            and sum(c["served_late"]) <= 0]
    check("E4a 每 cell 总发帧 == 集数 × 预算", not bad1, f"{len(cells)} cells")
    check("E4b reserved == attempted + wasted", not bad2, "全 cell")
    check("E4c unpaced B<L 的 t>B 发帧恒 0（缺陷仍在，未被悄悄修掉）", not bad3,
          f"{len(bad3)} cells 违反")
    check("E4d paced B<L 的 t>B 发帧 >0（R17：配速确实生效）", not bad4,
          f"{len(bad4)} cells 违反")

    # ---------- E5：ZOH 开发集 B=200 unpaced 与 results/56 逐位回归 ----------
    print("\n== E5（ZOH 开发集 {0,1,2} B=200 unpaced 与 results/56 的 zoh 列逐位回归）==")
    path = latest_result_path("56_t23b_compensation_benefit")
    print(f"  [ref-56] {path.relative_to(ROOT).as_posix() if path else '<缺失>'}")
    e5 = {"ref": (path.relative_to(ROOT).as_posix() if path else None), "compared": 0,
          "diffs": []}
    if path is None or not path.exists():
        e5["diffs"] = ["<无参考>"]
    elif n_ep != N_EP:
        e5["skipped"] = f"episodes={n_ep} != {N_EP}"
        print(f"  [SKIP] 集数 {n_ep} ≠ 历史 {N_EP} ⇒ 不可比")
    elif get("dev", 200, "unpaced", "zoh", DEV_SEEDS[0]) is None:
        # ★ 2026-10-20 07:18 补点轮：网格里没有 B=200 ⇒ 这是"本轮没跑"，不是"不一致"。
        #   （#46 同族：检查对象不存在时不许静默判 FAIL/PASS）
        e5["skipped"] = "本网格不含 B=200 unpaced"
        print(f"  [SKIP] 本轮网格不含 B=200 unpaced（{RUN_BUDGETS_DEV}）⇒ 无可比 cell")
    else:
        ref = json.loads(path.read_text(encoding="utf-8"))
        ref_map = {(c["budget"], c.get("arm", ARM), c["mode"], c["seed"]): c["J_per_episode"]
                   for c in ref.get("cells_J", []) if c.get("predictor") == "zoh"}
        for s in DEV_SEEDS:
            c = get("dev", 200, "unpaced", "zoh", s)
            if c is None:
                continue
            key = (200, ARM, "unpaced", s)
            if key not in ref_map:
                e5["diffs"].append(f"{key} 不在参考里")
                continue
            a = np.asarray(c["J_per_episode"], dtype=float)
            b = np.asarray(ref_map[key], dtype=float)
            if a.shape != b.shape:
                e5["diffs"].append(f"{key} 形状 {a.shape} vs {b.shape}")
                continue
            e5["compared"] += 1
            if not np.array_equal(a, b):
                e5["diffs"].append(f"{key} max|Δ|={np.nanmax(np.abs(a - b)):.3e}")
    if "skipped" in e5:
        print(f"  [SKIP] {e5['skipped']}")
    else:
        check("E5 ZOH 开发集 3 cell 与 results/56 逐 episode J **逐位相同**",
              e5["compared"] > 0 and not e5["diffs"],
              (f"{e5['compared']} cells 逐位相同" if not e5["diffs"] else "；".join(e5["diffs"][:4])))

    # ---------- 判词（预注册 §6）----------
    print("\n== 判词（预注册 §6：刻画，不设通过线）==")
    neg_hit = (f1.get("verdict_full") == "ANTI-MONOTONE"
               or (f1.get("x_span_ratio") is not None and f1["x_span_ratio"] < 1.5))
    verdicts = {
        "F1": f1.get("verdict_full"),
        "F1_dedup": f1.get("verdict_dedup"),
        "F2": {m: (f2[m]["crossings"] if m in f2 else None) for m in modes},
        "negation_condition_hit": bool(neg_hit),
        "negation_reason": ("横轴未分开（span<1.5）" if (f1.get("x_span_ratio") is not None
                                                      and f1["x_span_ratio"] < 1.5)
                            else ("τ_b ≤ −0.6" if f1.get("verdict_full") == "ANTI-MONOTONE"
                                  else "")),
    }
    print(f"  F1={verdicts['F1']}｜F1_dedup={verdicts['F1_dedup']}｜否定条件命中="
          f"{neg_hit}（{verdicts['negation_reason'] or '否'}）")

    _dump({"selfchecks": sc}, cells,
          {"summary": summary, "points": pts, "F1": f1, "F2": f2, "F3": f3,
           "E3": e3, "E5": e5, "verdicts": verdicts, "gate": GATE, "absorbed": ABSORBED},
          n_ep, sets, run_id=args.run_id, allow_overwrite=args.allow_overwrite,
          persist=not args.smoke)
    print(f"\n{'ALL CHECKS PASS' if not FAILS else 'FAILED: ' + '; '.join(FAILS)}")
    print(f"总耗时 {round(time.time() - t_all, 1)}s")
    return 0 if not FAILS else 2


def _dump(payload, cells, extra, n_ep, sets, *, run_id=None, allow_overwrite=False,
          persist=True):
    cfg = {
        "N": 2, "window": WINDOW, "warmup": [WARMUP_UAV, 0],
        "budgets_dev": list(RUN_BUDGETS_DEV), "budgets_conf": list(BUDGETS_CONF),
        "refine": bool(RUN_REFINE),
        "episodes": n_ep, "seed_sets": {k: list(v) for k, v in sets.items()},
        "gate": GATE, "arm": ARM, "absorbed": [float(x) for x in ABSORBED],
        "predictors": list(RUN_PREDICTORS), "modes": list(RUN_MODES),
        "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
        "x_axis": (f"UAV(user0) age 的算术均值，alive 且 t≥{AGE_T_MIN} 的决策点，"
                   "取 linear 轨迹；p50/p90/stale_mean 同集合"),
        "y_axis": "R_uav = Σ J_uav(linear)/Σ J_uav(zoh)，同种子同 episode 配对",
        "criterion": "Kendall τ-b 于 12 点：≥0.6 MONOTONE / ≤−0.6 ANTI-MONOTONE / 否则 WEAK；"
                     "不设通过线（预注册 §6）",
        "prereg": "04-实验与代码/2026-10-20-P3-1-失配边界图-预注册-v0.1.md",
    }
    out = {"config": cfg, "selfchecks": payload.get("selfchecks"),
           "checks_failed": list(FAILS)}
    (ROOT / "outputs").mkdir(exist_ok=True)
    (ROOT / "outputs" / (RESULT_NAME + SMOKE_TAG + ".json")).write_text(
        json.dumps({**out, **(extra or {}), "cells": (cells or [])},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    if cells is None or not persist:
        if cells is not None:
            print("[smoke] 不写 results/（smoke 数字不进证据链）")
        return
    keep = ("seed_set", "budget", "mode", "predictor", "seed", "J_per_episode", "J_mean",
            "served", "served_late", "served_total", "alive_steps", "frames", "attempts",
            "link_q", "budget_reserved", "budget_attempted", "budget_wasted",
            "age_uav", "age_cart", "delivered_count", "elapsed_s")
    slim = [{k: c[k] for k in keep} for c in (cells or [])]
    dest = write_result(RESULT_NAME, {**out, **(extra or {}), "cells_J": slim},
                        run_id=run_id, allow_overwrite=allow_overwrite)
    print(f"[evidence] {dest.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    sys.exit(main())
