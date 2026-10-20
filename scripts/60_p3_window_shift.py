#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P3-2 · 失效窗口平移检验：把任务的失效窗口改成新值，看"外推吃穿 ZOH"的交点是否跟着平移。

口径来源（★ 全部在看数之前冻结）
------------------------------------
`04-实验与代码/2026-10-20-P3-2-失效窗口平移检验-预注册-v0.1.md` §0–§9。

一句话
------
P3-1 把交点画在 `x∈(5.83,7.57)`（`x` = UAV `age` 均值），而"40–50 步断服"与"UAV 失效窗口 53 步"
只是**量级一致** —— 从没有一轮**单独操纵过窗口**。本轮只改**任务侧一个参数**
（`UavTrackAdapter.escape_radius` = `d_fail`）造出 35 / 53 / 104 步三个**实测**窗口
（pilot 见预注册 §3b），问：交点是否随之**同序平移**。

★ 唯一变动量 = `escape_radius`。预测器（`Linear`）、臂（`age`）、`GATE`、`window_cost` 全不动。
★ `absorbed_uav := escape_radius`（预注册 §0a：`scripts/42:83` 的定义就是 `ABSORBED_UAV = ESCAPE_RADIUS`；
  固定 25 会让"活到 45 m"比"第 10 步就死"更贵 ⇒ 违反验收 §3-C 的单调性。G2 有反向对照打死这条）。

用法
----
    D:/devtools/miniconda3/envs/ai-lab/python.exe scripts/60_p3_window_shift.py
    ... --skip-experiment      # 只跑自检（不写 results/）
    ... --smoke                # 2 集 × 1 种子 × 2 预算，不写 results/

产物
----
    outputs/60_p3_window_shift[_smoke].json     全量表（不入库）
    results/runs/<run_id>/60_p3_window_shift.json  **入库**（results_io 唯一入口）
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
RESULT_NAME = "60_p3_window_shift"
sys.path.insert(0, str(ROOT))

from wmlab.eval.multiuser import (  # noqa: E402
    StepRecord, default_uav_cost, delivery_rate, window_cost,
)
from wmlab.eval.predictors import Linear, ZOH  # noqa: E402
from wmlab.eval.results_io import latest_result_path, write_result  # noqa: E402

FAILS: list[str] = []

#: ★ 预注册 §3b：三个窗口由 pilot **实测**选出（不是按参数反推）
RADII = (("short", 16.0, 35), ("base", 25.0, 53), ("long", 35.0, 104))
BUDGETS = (200, 175, 160, 150, 140, 120, 100, 80)
PREDICTORS = ("zoh", "linear")
DEV_SEEDS = (0, 1, 2)
MODE = "unpaced"
N_EP = 30
RUN_BUDGETS = BUDGETS
RUN_RADII = RADII

AGE_T_MIN = 41
UAV = 0
N_BOOT = 10000
BOOT_SEED = 20261020


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


T12 = _load(ROOT / "scripts" / "42_t12_hetero_four_arms.py", "t12_mod60")
T52 = _load(ROOT / "scripts" / "52_t21_paced_matching_ablation.py", "t52_mod60")

WINDOW = T12.WINDOW
WARMUP_UAV = T12.WARMUP_UAV
GATE = T12.GATE
ABSORBED_CART = T12.ABSORBED_CART
ARM = "age"


def check(tag: str, ok: bool, detail: str) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {tag}: {detail}")
    if not ok:
        FAILS.append(f"{tag}: {detail}")


# ================================================================ 装置（与 59 同源，复制而非改 59）
def _run(budget, seed, n_ep, *, predictor=None, radius=25.0, probe=None):
    envs = [T12.make_env("uav", seed=seed, max_steps=WARMUP_UAV + WINDOW, escape_radius=float(radius)),
            T12.make_env("CartPole-v1", seed=seed)]
    ctrls = [T12.PDRelativeController(), T12.CartPoleLQR(_run.K)]
    alloc = T52.make_alloc(ARM, MODE, budget)
    try:
        return T12.run_multiuser(
            envs, ctrls, alloc, T12.SINRSuccess(T12.Q_REF),
            n_episodes=n_ep, seed=seed, max_steps=WINDOW,
            warmup_steps=[WARMUP_UAV, 0], budget_total=int(budget),
            csi=T12.make_csi(seed, n_ep),
            cost_fn=[default_uav_cost, T12.cartpole_cost], diagnostic=False,
            counterfactual=False, predictor=predictor, pred_probe=probe)
    finally:
        for e in envs:
            e.close()


_run.K = T12.load_lqr_gain()


def _J(records, n_ep, radius):
    return np.asarray(window_cost(records, 2, window=WINDOW,
                                  absorbed=[float(radius), float(ABSORBED_CART)],
                                  n_episodes=n_ep), dtype=float)


class _AgeProbe:
    """逐决策点采 `age`（**决策前**的值，与 `StepRecord.ages` 同一时点）。与 59 同源。"""

    def __init__(self) -> None:
        self.seq: list[tuple[int, int, int, int, bool, bool]] = []

    def __call__(self, p: dict) -> None:
        e, t = int(p["episode"]), int(p["t"])
        for i in (0, 1):
            self.seq.append((e, t, i, int(p["age"][i]), bool(p["delivered"][i]), bool(p["alive"][i])))

    def axis(self, user: int = UAV, t_min: int = AGE_T_MIN) -> dict:
        rows = [(e, t, a) for (e, t, u, a, _dl, al) in self.seq if u == user and al and t >= t_min]
        if not rows:
            return {"n": 0}
        a = np.asarray([r[2] for r in rows], dtype=float)
        per_ep: dict[int, list] = {}
        for e, _t, x in rows:
            per_ep.setdefault(e, []).append(x)
        ep_means = np.asarray([float(np.mean(v)) for v in per_ep.values()], dtype=float)
        stale = a[a >= 1]
        return {"n": int(a.size), "mean": float(a.mean()),
                "p50": float(np.percentile(a, 50)), "p90": float(np.percentile(a, 90)),
                "stale_mean": (float(stale.mean()) if stale.size else None),
                "ep_median_of_means": float(np.median(ep_means)),
                "n_episodes_used": int(ep_means.size),
                "frac_age0": float(np.mean(a == 0))}

    def counts(self, user: int) -> tuple[int, int]:
        n = sum(1 for s in self.seq if s[2] == user)
        d = sum(1 for s in self.seq if s[2] == user and s[4])
        return n, d


def cluster_bootstrap(rows, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    """`R_i = Σ J_i(num)/Σ J_i(den)` 的 cluster bootstrap（重采样单位 = reset 初值）。与 58/59 同源。"""
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
    per_ep = np.where(Jd > 0, Jn / np.where(Jd > 0, Jd, 1.0), np.nan)
    return {"R": [float(x) for x in point],
            "ci95": [[float(x) for x in np.nanpercentile(draws, 2.5, axis=0)],
                     [float(x) for x in np.nanpercentile(draws, 97.5, axis=0)]],
            "geo_R": float(np.sqrt(point[0] * point[1])),
            "n_clusters": len(cl), "n_rows": len(rows), "n_boot": int(n_boot),
            "n_boot_nonfinite": int(n_bad),
            "per_episode_ratio_median": [float(np.nanmedian(per_ep[:, i])) for i in (0, 1)],
            "boot_seed": int(seed)}


# ================================================================ 交点 / 区间 / 判词（预注册 §6）
def crossings(points) -> list[dict]:
    """`R_uav` 穿过 1 的相邻区间（线性插值）。points 需按 x 排序、x 不重复。"""
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


def bracket_of(points) -> dict:
    """把一档窗口的 8 个点压成"交点区间 / 单侧界"。"""
    if not points:
        return {"kind": "EMPTY"}
    cr = crossings(points)
    ys = [p["R_uav"] for p in points]
    if cr:
        first = cr[0]
        return {"kind": "BRACKETED", "bracket": sorted(first["between"]),
                "x_star": first["x"], "n_crossings": len(cr), "crossings": cr}
    if min(ys) >= 1.0:
        return {"kind": "UNBRACKETED_LOW", "bound": "x* ≤ 最小 x",
                "x_max": float(max(p["x"] for p in points)), "R_min": float(min(ys))}
    return {"kind": "UNBRACKETED_HIGH", "bound": "x* ≥ 最大 x",
            "x_max": float(max(p["x"] for p in points)), "R_max": float(max(ys))}


def pair_relation(a: dict, b: dict) -> str:
    """窗口 b（更长）相对窗口 a 的位移关系。两档都必须是 BRACKETED。"""
    a_lo, a_hi = a["bracket"]
    b_lo, b_hi = b["bracket"]
    if b_lo >= a_hi:
        return "ORDERED"
    if b_hi <= a_lo:
        return "ANTI"
    return "OVERLAP"


def verdict_of(brackets: list[dict]) -> dict:
    """预注册 §6：① 只要有一个 UNBRACKETED ⇒ INCONCLUSIVE；② 任一对重叠 ⇒ INCONCLUSIVE；
    ③ 任一对反序 ⇒ WINDOW-ANTI；④ 否则 WINDOW-SHIFTS。"""
    kinds = [b["kind"] for b in brackets]
    if any(k != "BRACKETED" for k in kinds):
        return {"verdict": "INCONCLUSIVE", "reason": f"存在非 BRACKETED 窗口：{kinds}",
                "pairs": []}
    pairs = []
    for i in range(len(brackets)):
        for j in range(i + 1, len(brackets)):
            pairs.append({"pair": [i, j], "rel": pair_relation(brackets[i], brackets[j])})
    rels = [p["rel"] for p in pairs]
    if any(r == "OVERLAP" for r in rels):
        return {"verdict": "INCONCLUSIVE", "reason": "交点区间重叠（分不清位移）", "pairs": pairs}
    if any(r == "ANTI" for r in rels):
        return {"verdict": "WINDOW-ANTI", "reason": "窗口↑ 而交点↓（机制错）", "pairs": pairs}
    return {"verdict": "WINDOW-SHIFTS", "reason": "三档区间互不重叠且同序", "pairs": pairs}


# ================================================================ 失效窗口（与 scripts/46 同口径）
def measure_window(radius: float, n_ep: int = 200) -> dict:
    """完全不服务时还能活多少步：warmup 每步送真值 → 冻结 `est` → 记越界或终止的步数。

    ★ 与 `scripts/46` 逐字相同（含 `max_steps`、`PDRelativeController`），唯一变化量 = `escape_radius`。
    """
    ks, d0s = [], []
    for e in range(n_ep):
        env = T12.make_env("uav", seed=e, max_steps=WARMUP_UAV + WINDOW, escape_radius=float(radius))
        ctrl = T12.PDRelativeController()
        obs = np.asarray(env.reset(seed=e), dtype=np.float32).copy()
        ctrl.reset(obs)
        for _ in range(WARMUP_UAV):
            res = env.step(ctrl.act(obs))
            obs = np.asarray(res.obs, dtype=np.float32).copy()
            if res.terminated or res.truncated:
                break
        est = obs.copy()
        d0 = float(default_uav_cost(est))
        k = 0
        while k < WINDOW:
            res = env.step(ctrl.act(est))
            k += 1
            if default_uav_cost(np.asarray(res.obs)) > radius or res.terminated or res.truncated:
                break
        env.close()
        ks.append(k)
        d0s.append(d0)
    a = np.asarray(ks, dtype=float)
    return {"radius": float(radius), "n": int(a.size), "median": float(np.median(a)),
            "p10": float(np.percentile(a, 10)), "p90": float(np.percentile(a, 90)),
            "frac_le_1": float((a <= 1).mean()),
            "dist_at_freeze_median": float(np.median(d0s)),
            "all_identical": bool(np.all(a == a[0]))}


# ================================================================ 自检
def _synth(dead_at: int, cost: float, window: int = WINDOW) -> list[StepRecord]:
    """单用户合成轨迹：`t ≤ dead_at` 存活且代价 = cost，之后死亡（代价 nan、alive False）。"""
    out = []
    for t in range(1, window + 1):
        alive = t <= dead_at
        out.append(StepRecord(episode=0, t=t, served_idx=-1, attempted_idx=-1, delivered_idx=-1,
                              ages=(t,), costs=(float(cost) if alive else float("nan"),),
                              alive=(bool(alive),)))
    return out


def selfchecks(n_ep_small: int = 2, full_window: bool = False) -> dict:
    print("\n== 自检 G1–G8（预注册 §0a/§6；每条写明能打死什么错）==")
    out: dict = {}

    # ---------------- G1：None ≡ ZOH（★ 在短窗口上验，确保新参数没改既有路径）----------------
    j_none = _J(_run(200, 0, n_ep_small, predictor=None, radius=16.0)["records"], n_ep_small, 16.0)
    j_zoh = _J(_run(200, 0, n_ep_small, predictor=ZOH(), radius=16.0)["records"], n_ep_small, 16.0)
    d1 = float(np.max(np.abs(j_none - j_zoh))) if j_none.size else 0.0
    check("G1 `predictor=None` 与 `ZOH` 逐 episode J **逐位相同**（R_f=16）", d1 == 0.0,
          f"max|Δ|={d1:.3e}")
    out["G1"] = {"max_abs_diff": d1}

    # ---------------- G2：absorbed 与 R_f 的一致性（★ 含两处反向对照）----------------
    g2 = {}
    ok2 = True
    for (_, R, _W) in RADII:
        cost = 0.9 * R
        j_dead = float(_J(_synth(10, cost), 1, R)[0, 0])
        j_alive = float(_J(_synth(WINDOW, cost), 1, R)[0, 0])
        j_dead_bad = float(window_cost(_synth(10, cost), 1, window=WINDOW,
                                       absorbed=[0.5 * R], n_episodes=1)[0][0])
        good = j_dead > j_alive          # 早死必须更贵（验收 §3-C）
        bad_inverted = j_dead_bad < j_alive   # 反向对照：absorbed 减半 ⇒ 早死变便宜
        g2[str(R)] = {"J_dead@10": j_dead, "J_alive_full": j_alive,
                      "J_dead_with_half_absorbed": j_dead_bad,
                      "monotone_ok": good, "negative_control_inverted": bad_inverted}
        ok2 = ok2 and good and bad_inverted
    # ★ 第二处对照：**若按 STATE 草稿固定 absorbed=25 而窗口抬到 35**，单调性会破
    R, cost = 35.0, 0.9 * 35.0
    j_dead_fixed = float(window_cost(_synth(10, cost), 1, window=WINDOW,
                                     absorbed=[25.0], n_episodes=1)[0][0])
    j_alive_fixed = float(window_cost(_synth(WINDOW, cost), 1, window=WINDOW,
                                      absorbed=[25.0], n_episodes=1)[0][0])
    g2["fixed25_at_Rf35"] = {"J_dead@10": j_dead_fixed, "J_alive_full": j_alive_fixed,
                             "early_death_cheaper": j_dead_fixed < j_alive_fixed}
    ok2 = ok2 and (j_dead_fixed < j_alive_fixed)
    check("G2 `absorbed=R_f` 时「早死更贵」成立（3 档）；把它减半 ⇒ 该性质反向（判据有分辨力）；"
          "★ 若固定 25 而 R_f=35 ⇒ 早死反而更便宜（= 预注册 §0a 的量级论证）",
          ok2, json.dumps(g2, ensure_ascii=False))
    out["G2"] = g2

    # ---------------- G3：窗口参数真的接线了（★ 有分辨力：没接线时三档会完全相同）----------------
    n_ep_w = 60 if not full_window else 200
    g3 = {}
    for (name, R, W_expected) in RADII:
        m = measure_window(R, n_ep=n_ep_w)
        g3[name] = {**m, "expected": W_expected}
    ws = [g3[n]["median"] for (n, _R, _W) in RADII]
    exact = all(abs(g3[n]["median"] - W) < 1e-9 for (n, _R, W) in RADII)
    strict = ws[0] < ws[1] < ws[2]
    check("G3 实测窗口随 `escape_radius` 单调且与预注册 §3b 的 35/53/104 **精确一致**"
          "（没接线时三档会全等于 53 ⇒ FAIL）",
          exact and strict, "；".join(f"{n}:R={R:g}→{g3[n]['median']:.0f}"
                                     for (n, R, _W) in RADII) + f"｜exact={exact} strict={strict}")
    out["G3"] = g3

    # ---------------- G4：基线窗口（R_f=25）与既有证据逐位回归 ----------------
    print("\n== G4（R_f=25 的 ZOH/Linear 是否与既有入库证据逐位相同）==")
    g4: dict = {"items": [], "compared": 0, "diffs": []}
    ref59 = latest_result_path("59_p3_boundary")
    ref56 = latest_result_path("56_t23b_compensation_benefit")
    print(f"  [ref-59] {ref59.relative_to(ROOT).as_posix() if ref59 else '<缺失>'}")
    print(f"  [ref-56] {ref56.relative_to(ROOT).as_posix() if ref56 else '<缺失>'}")
    if not full_window or n_ep_small != N_EP:
        g4["skipped"] = f"集数 {n_ep_small} ≠ {N_EP}（只在全量运行里做逐位回归）"
        print(f"  [SKIP] {g4['skipped']}")
    else:
        ref = json.loads(ref59.read_text(encoding="utf-8")) if ref59 and ref59.exists() else None
        rmap = ({(c["budget"], c["mode"], c["predictor"], c["seed"]): c["J_per_episode"]
                 for c in ref.get("cells_J", [])} if ref else {})
        r56 = json.loads(ref56.read_text(encoding="utf-8")) if ref56 and ref56.exists() else None
        r56map = ({(c["budget"], c.get("arm", ARM), c["mode"], c["seed"]): c["J_per_episode"]
                   for c in r56.get("cells_J", []) if c.get("predictor") == "zoh"}
                  if r56 else {})
        for budget in (200, 150, 100):
            for pname in PREDICTORS:
                for s in DEV_SEEDS:
                    c = _cell(budget, pname, s, n_ep_small, 25.0)
                    a = np.asarray(c["J_per_episode"], dtype=float)
                    key = (budget, MODE, pname, s)
                    b = None
                    src = ""
                    if key in rmap:
                        b = np.asarray(rmap[key], dtype=float)
                        src = "59"
                    elif pname == "zoh" and key in r56map:
                        b = np.asarray(r56map[key], dtype=float)
                        src = "56"
                    if b is None or a.shape != b.shape:
                        g4["diffs"].append(f"{key} 无参考/形状不符")
                        continue
                    g4["compared"] += 1
                    g4["items"].append({"key": list(key), "src": src,
                                        "max_abs_diff": float(np.nanmax(np.abs(a - b)))})
                    if not np.array_equal(a, b):
                        g4["diffs"].append(f"{key} vs {src} max|Δ|={np.nanmax(np.abs(a - b)):.3e}")
        check("G4 基线窗口（R_f=25）18 个 cell 与 results/59（及 56 的 zoh 列）逐 episode J **逐位相同**",
              g4["compared"] > 0 and not g4["diffs"],
              f"compared={g4['compared']}" + ("" if not g4["diffs"] else "；" + "；".join(g4["diffs"][:3])))
    out["G4"] = g4

    # ---------------- G5：记账（每个半径各一格）----------------
    g5 = {}
    ok5 = True
    for (name, R, _W) in RADII:
        r = _run(100, 0, n_ep_small, predictor=ZOH(), radius=R)
        late = sum(1 for x in r["records"] if x.t > 100 and x.attempted_idx >= 0)
        served = sum(1 for x in r["records"] if x.attempted_idx >= 0)
        ok = (served == n_ep_small * 100
              and r["budget_reserved"] == r["budget_attempted"] + r["budget_wasted"]
              and late == 0)
        g5[name] = {"served": served, "expect": n_ep_small * 100, "late": late, "ok": ok}
        ok5 = ok5 and ok
    check("G5 每档窗口：总发帧 == 集数×预算、reserved==attempted+wasted、unpaced B<L 的 t>B 发帧=0",
          ok5, json.dumps(g5, ensure_ascii=False))
    out["G5"] = g5

    # ---------------- G6：`age` 递推恒等式（在**长窗口**上验：死亡更少 ⇒ 更多相邻点可比）----------------
    pr = _AgeProbe()
    _run(120, 0, n_ep_small, predictor=Linear(), radius=35.0, probe=pr)
    by: dict[tuple[int, int], list] = {}
    for (e, t, u, a, dl, al) in pr.seq:
        by.setdefault((e, u), []).append((t, a, dl, al))
    bad_rec = checked = 0
    for (_e, _u), v in by.items():
        v.sort()
        for k in range(len(v) - 1):
            t, a, dl, al = v[k]
            t2, a2, _d2, al2 = v[k + 1]
            if not (al and al2) or t2 != t + 1:
                continue
            checked += 1
            if a2 != (0 if dl else a + 1):
                bad_rec += 1
    bad_set = 0
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
        for e, ts in alive_t.items():
            first = min(ts)
            if {d + 1 for d in dl_t.get(e, set()) if d + 1 in ts} != \
                    {t for t in zero_t.get(e, set()) if t > first}:
                bad_set += 1
    check("G6 `age_pre[t+1] == 0 if delivered[t] else age_pre[t]+1` 逐点成立，且 `age==0` 的时点集合"
          " == 送达时点+1（排除每集首点）（R_f=35；打死「`age` 采在错误时点」）",
          bad_rec == 0 and checked > 0 and bad_set == 0,
          f"checked={checked} violations={bad_rec} set_mismatch_episodes={bad_set}")
    out["G6"] = {"checked": checked, "violations": bad_rec, "set_mismatch": bad_set}

    # ---------------- G7：交点器 / 区间器 / 判词器有分辨力（先测工具，再测数据）----------------
    g7 = {}
    cr = crossings([{"x": 1.0, "R_uav": 0.5}, {"x": 2.0, "R_uav": 0.8},
                    {"x": 4.0, "R_uav": 1.4}, {"x": 8.0, "R_uav": 1.9}])
    g7["interp"] = cr
    okc = len(cr) == 1 and abs(cr[0]["x"] - 2.6666667) < 1e-6 and cr[0]["between"] == [2.0, 4.0]
    none_cr = crossings([{"x": 1.0, "R_uav": 0.5}, {"x": 2.0, "R_uav": 0.9}])
    gt = crossings([{"x": 1.0, "R_uav": 1.3}, {"x": 2.0, "R_uav": 1.9}])
    okc = okc and none_cr == [] and gt == []

    def _br(lo, hi):
        return {"kind": "BRACKETED", "bracket": [lo, hi], "x_star": 0.5 * (lo + hi)}

    v_shift = verdict_of([_br(4, 6), _br(6, 8), _br(14, 16)])
    v_anti = verdict_of([_br(14, 16), _br(6, 8), _br(4, 6)])
    v_over = verdict_of([_br(4, 8), _br(6, 9), _br(14, 16)])
    v_unbr = verdict_of([_br(4, 6), {"kind": "UNBRACKETED_LOW"}, _br(14, 16)])
    okv = (v_shift["verdict"] == "WINDOW-SHIFTS" and v_anti["verdict"] == "WINDOW-ANTI"
           and v_over["verdict"] == "INCONCLUSIVE" and v_unbr["verdict"] == "INCONCLUSIVE")
    g7.update({"shift": v_shift["verdict"], "anti": v_anti["verdict"],
               "overlap": v_over["verdict"], "unbracketed": v_unbr["verdict"]})
    check("G7a 交点器：0.8→1.4 在 [2,4] 给出 2.667；全程 <1 或全程 >1 时不给交点",
          okc, f"{cr}｜none={none_cr == []}｜all_above={gt == []}")
    check("G7b 判词器：同序不相交⇒SHIFTS、反序⇒WINDOW-ANTI、重叠⇒INCONCLUSIVE、"
          "有一档 UNBRACKETED⇒INCONCLUSIVE（四种都要能区分）", okv, json.dumps(g7, ensure_ascii=False))
    out["G7"] = g7
    return out


# ================================================================ 单 cell
def _cell(budget, pred_name, seed, n_ep, radius):
    t0 = time.time()
    pred = Linear() if pred_name == "linear" else ZOH()
    probe = _AgeProbe()
    r = _run(budget, seed, n_ep, predictor=pred, radius=radius, probe=probe)
    rec = r["records"]
    J = _J(rec, n_ep, radius)
    dr = delivery_rate(rec, 2)
    served = [sum(1 for x in rec if x.attempted_idx == i) for i in range(2)]
    late = [sum(1 for x in rec if x.t > budget and x.attempted_idx == i) for i in range(2)]
    return {"budget": int(budget), "radius": float(radius), "predictor": pred_name,
            "seed": int(seed), "J_per_episode": np.asarray(J, dtype=float).tolist(),
            "J_mean": np.asarray(J, dtype=float).mean(axis=0).tolist(),
            "served": served, "served_late": late, "served_total": int(sum(served)),
            "alive_steps": dr["alive_steps"], "frames": dr["frames"],
            "attempts": dr["attempts"], "link_q": dr["link_q"], "p_eff": dr["p_eff"],
            "budget_reserved": dr["budget_reserved"], "budget_attempted": dr["budget_attempted"],
            "budget_wasted": dr["budget_wasted"],
            "age_uav": probe.axis(UAV), "age_cart": probe.axis(1),
            "elapsed_s": round(time.time() - t0, 2)}


def run_cell(budget, pred_name, seed, n_ep, radius, name):
    c = _cell(budget, pred_name, seed, n_ep, radius)
    c["window"] = name
    return c


# ================================================================ 主实验
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=N_EP)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip-experiment", action="store_true")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--allow-overwrite", action="store_true")
    ap.add_argument("--budgets", default=None)
    ap.add_argument("--windows", default=None, help="只跑某些窗口，如 base,long")
    args = ap.parse_args()
    global SMOKE_TAG, RESULT_NAME, RUN_BUDGETS, RUN_RADII
    SMOKE_TAG = "_smoke" if args.smoke else ""
    RESULT_NAME = "60_p3_window_shift"
    n_ep = 2 if args.smoke else args.episodes
    if n_ep != N_EP and not args.smoke:
        print(f"[WARN] episodes={n_ep} ≠ 预注册的 {N_EP}：结果不得标「确认」，只能标「探索」")
    budgets = ((100, 200) if args.smoke else
               (tuple(int(x) for x in args.budgets.split(",")) if args.budgets else BUDGETS))
    radii = (tuple(r for r in RADII if r[0] in args.windows.split(","))
             if args.windows else RADII)
    RUN_BUDGETS, RUN_RADII = budgets, radii

    t_all = time.time()
    print(f"[60] P3-2 失效窗口平移检验｜窗口 {WINDOW}｜臂={ARM}｜episodes={n_ep}｜"
          f"mode={MODE}｜R_f={[r[1] for r in radii]}")
    sc = selfchecks(n_ep_small=(2 if args.smoke else N_EP), full_window=not args.smoke)

    if args.skip_experiment:
        _dump({"selfchecks": sc}, None, None, n_ep, persist=False)
        print(f"\n{'ALL CHECKS PASS' if not FAILS else 'FAILED: ' + '; '.join(FAILS)}")
        return 0 if not FAILS else 2

    cells: list[dict] = []
    for (name, R, _W) in radii:
        for budget in budgets:
            for pname in PREDICTORS:
                for seed in DEV_SEEDS:
                    c = run_cell(budget, pname, seed, n_ep, R, name)
                    cells.append(c)
                    ax = c["age_uav"]
                    print(f"  [{name:5s} R={R:4.0f}] B={budget:3d} {pname:6s} seed={seed} "
                          f"J=({c['J_mean'][0]:7.3f},{c['J_mean'][1]:.3f}) "
                          f"x={ax.get('mean', float('nan')):7.2f} "
                          f"alive=({c['alive_steps'][0]:4d},{c['alive_steps'][1]:3d}) "
                          f"late=({c['served_late'][0]},{c['served_late'][1]}) {c['elapsed_s']}s")

    def get(name, budget, pname, seed):
        for c in cells:
            if (c["window"], c["budget"], c["predictor"], c["seed"]) == (name, budget, pname, seed):
                return c
        return None

    def table(name, budget):
        rows = []
        for s in DEV_SEEDS:
            a, b = get(name, budget, "linear", s), get(name, budget, "zoh", s)
            if a is None or b is None:
                continue
            for e in range(len(a["J_per_episode"])):
                rows.append({"cluster": int(s) + e,
                             "J_num": a["J_per_episode"][e], "J_den": b["J_per_episode"][e]})
        if not rows:
            return None
        d = cluster_bootstrap(rows)
        per_seed = {}
        for s in DEV_SEEDS:
            a = np.asarray(get(name, budget, "linear", s)["J_mean"], dtype=float)
            b = np.asarray(get(name, budget, "zoh", s)["J_mean"], dtype=float)
            per_seed[str(s)] = [float(x) for x in (a / b)] if np.all(b > 0) else [float("nan")] * 2
        d["per_seed_R"] = per_seed
        rs = np.asarray(list(per_seed.values()), dtype=float)
        d["n_seeds_uav_below_1"] = int(np.sum(rs[:, 0] < 1.0))
        d["n_seeds"] = int(rs.shape[0])
        ax = [get(name, budget, p, s)["age_uav"] for p in ("linear",) for s in DEV_SEEDS]
        az = [get(name, budget, "zoh", s)["age_uav"] for s in DEV_SEEDS]
        d["x_linear"] = {k: float(np.nanmean([a.get(k, float("nan")) for a in ax])) for k in
                         ("mean", "p50", "p90", "stale_mean", "ep_median_of_means")}
        d["x_zoh"] = {k: float(np.nanmean([a.get(k, float("nan")) for a in az])) for k in
                      ("mean", "p50", "p90", "stale_mean", "ep_median_of_means")}
        d["served"] = {p: [float(np.mean([get(name, budget, p, s)["served"][i]
                                          for s in DEV_SEEDS])) for i in (0, 1)]
                       for p in ("linear", "zoh")}
        d["alive"] = {p: [float(np.mean([get(name, budget, p, s)["alive_steps"][i]
                                         for s in DEV_SEEDS])) for i in (0, 1)]
                      for p in ("linear", "zoh")}
        d["late"] = {p: float(np.mean([sum(get(name, budget, p, s)["served_late"])
                                       for s in DEV_SEEDS])) for p in ("linear", "zoh")}
        d["J"] = {p: [float(np.mean([get(name, budget, p, s)["J_mean"][i] for s in DEV_SEEDS]))
                      for i in (0, 1)] for p in ("linear", "zoh")}
        return d

    summary: dict = {}
    for (name, R, W) in radii:
        summary[name] = {"radius": float(R), "window_expected": int(W), "budgets": {}}
        for budget in budgets:
            d = table(name, budget)
            if d is not None:
                summary[name]["budgets"][str(budget)] = d

    # ---------- 主表 ----------
    brackets: list[dict] = []
    print("\n== 主表（unpaced；x=UAV age mean @t≥41 于 linear 轨迹；R_uav=Σ J_lin/Σ J_zoh）==")
    for (name, R, W) in radii:
        pts = []
        for budget in budgets:
            d = summary[name]["budgets"].get(str(budget))
            if d is None:
                continue
            pts.append({"budget": budget, "x": d["x_linear"]["mean"], "x_p50": d["x_linear"]["p50"],
                        "x_p90": d["x_linear"]["p90"], "x_zoh": d["x_zoh"]["mean"],
                        "R_uav": d["R"][0], "R_cart": d["R"][1],
                        "ci_uav": [d["ci95"][0][0], d["ci95"][1][0]],
                        "ci_cart": [d["ci95"][0][1], d["ci95"][1][1]],
                        "seeds_uav_below_1": f"{d['n_seeds_uav_below_1']}/{d['n_seeds']}",
                        "late": d["late"]["linear"], "late_zoh": d["late"]["zoh"],
                        "served": d["served"]["linear"], "served_zoh": d["served"]["zoh"],
                        "alive": d["alive"]["linear"], "alive_zoh": d["alive"]["zoh"],
                        "J": d["J"]["linear"], "J_zoh": d["J"]["zoh"]})
        pts.sort(key=lambda r: r["x"])
        summary[name]["points"] = pts
        br = bracket_of(pts)
        summary[name]["bracket"] = br
        brackets.append(br)
        print(f"\n  -- 窗口 {name}（R_f={R:g}，实测 W={W} 步）--")
        for r in pts:
            print(f"   B={r['budget']:3d} x={r['x']:7.3f}(p50={r['x_p50']:5.1f} zoh_x={r['x_zoh']:6.2f}) "
                  f"R_uav={r['R_uav']:.3f} CI[{r['ci_uav'][0]:.3f},{r['ci_uav'][1]:.3f}] "
                  f"R_cart={r['R_cart']:.3f} | J=({r['J'][0]:7.3f},{r['J'][1]:.3f}) "
                  f"J_zoh=({r['J_zoh'][0]:7.3f},{r['J_zoh'][1]:.3f}) "
                  f"served=({r['served'][0]:5.0f},{r['served'][1]:5.0f}) "
                  f"late={r['late']:.0f} seeds<1={r['seeds_uav_below_1']}")
        print(f"   ⇒ {br['kind']} {br.get('bracket') or br.get('bound')}"
              f"｜x*={br.get('x_star')}")

    # ---------- F1' 判词 ----------
    print("\n== F1'（预注册 §6：三档窗口的交点区间是否互不重叠且同序）==")
    vd = verdict_of(brackets)
    print(f"  ⇒ **{vd['verdict']}**（{vd['reason']}）")
    for p in vd.get("pairs", []):
        print(f"     pair {p['pair']} → {p['rel']}")
    # 附加（只报不判）：x*/W
    ratio_tbl = {}
    for (name, R, W), br in zip(radii, brackets):
        if br["kind"] == "BRACKETED":
            ratio_tbl[name] = {"W": W, "x_star": br["x_star"],
                               "x_star_over_W": br["x_star"] / W,
                               "bracket_over_W": [br["bracket"][0] / W, br["bracket"][1] / W]}
    print("  [附加·只报不判] x*/W：" + json.dumps(ratio_tbl, ensure_ascii=False))

    # ---------- 稳健性：横轴换 zoh ----------
    print("\n== 稳健性（横轴换 zoh 轨迹的陈旧度；x 内生 ⇒ 只报不判）==")
    zoh_br = []
    for (name, _R, _W) in radii:
        pts = sorted(summary[name]["points"], key=lambda r: r["x_zoh"])
        b = bracket_of([{"x": p["x_zoh"], "R_uav": p["R_uav"]} for p in pts])
        zoh_br.append(b)
        print(f"  {name:5s} ⇒ {b['kind']} {b.get('bracket') or b.get('bound')}")
    vd_zoh = verdict_of(zoh_br)
    print(f"  判词（同口径）：{vd_zoh['verdict']}（{vd_zoh['reason']}）")

    # ---------- G3（实测窗口，全量 n=200）----------
    print("\n== G3b（全量 200 集复测三档窗口，用于报告里引用）==")
    g3b = {}
    if not args.smoke:
        for (name, R, W) in radii:
            m = measure_window(R, n_ep=200)
            g3b[name] = {**m, "expected": W}
            print(f"  {name:5s} R_f={R:g} W={m['median']:.0f} (p10={m['p10']:.0f} "
                  f"p90={m['p90']:.0f}) 逐集相同={m['all_identical']} 预注册={W}")
        check("G3b 全量复测的窗口 == 预注册 §3b 的 35/53/104",
              all(abs(g3b[n]["median"] - W) < 1e-9 for (n, _R, W) in radii),
              "；".join(f"{n}:{g3b[n]['median']:.0f}" for (n, _R, _W) in radii))

    # ---------- 主表级记账 ----------
    bad1 = [c for c in cells if c["served_total"] != n_ep * c["budget"]]
    bad2 = [c for c in cells if c["budget_reserved"]
            != c["budget_attempted"] + c["budget_wasted"]]
    bad3 = [c for c in cells if c["budget"] < WINDOW and sum(c["served_late"]) != 0]
    check("G5b 每 cell 总发帧 == 集数×预算", not bad1, f"{len(cells)} cells")
    check("G5c reserved == attempted + wasted", not bad2, "全 cell")
    check("G5d unpaced B<L 的 t>B 发帧恒 0（断服缺陷仍在，未被悄悄修掉）", not bad3,
          f"{len(bad3)} cells 违反")

    extra = {"summary": summary,
             "brackets": {n: summary[n]["bracket"] for (n, _R, _W) in radii},
             "F1p": vd, "F1p_zoh_axis_report_only": vd_zoh,
             "x_star_over_W": ratio_tbl,
             "G3b_full_window": g3b,
             "windows": {n: {"radius": R, "W_expected": W} for (n, R, W) in radii},
             "gate": GATE,
             "absorbed_note": "absorbed_uav := escape_radius（预注册 §0a）；absorbed_cart=1.0 不变"}
    _dump({"selfchecks": sc}, cells, extra, n_ep, run_id=args.run_id,
          allow_overwrite=args.allow_overwrite, persist=not args.smoke)
    print(f"\n{'ALL CHECKS PASS' if not FAILS else 'FAILED: ' + '; '.join(FAILS)}")
    print(f"总耗时 {round(time.time() - t_all, 1)}s")
    return 0 if not FAILS else 2


def _dump(payload, cells, extra, n_ep, *, run_id=None, allow_overwrite=False, persist=True):
    cfg = {"N": 2, "window": WINDOW, "warmup": [WARMUP_UAV, 0],
           "budgets": list(RUN_BUDGETS), "episodes": n_ep, "seeds": list(DEV_SEEDS),
           "mode": MODE, "arm": ARM, "gate": GATE,
           "radii": {n: R for (n, R, _W) in RUN_RADII},
           "windows_expected": {n: W for (n, _R, W) in RUN_RADII},
           "absorbed": "uav := escape_radius (per window); cart := 1.0",
           "predictors": list(PREDICTORS), "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
           "x_axis": (f"UAV(user0) age 的算术均值，alive 且 t≥{AGE_T_MIN} 的决策点，取 linear 轨迹"),
           "y_axis": "R_uav = Σ J_uav(linear)/Σ J_uav(zoh)，同种子同 episode 配对",
           "criterion": "三档窗口的交点区间：不重叠且同序 ⇒ WINDOW-SHIFTS；重叠/缺档 ⇒ INCONCLUSIVE；"
                        "反序 ⇒ WINDOW-ANTI（预注册 §6）",
           "prereg": "04-实验与代码/2026-10-20-P3-2-失效窗口平移检验-预注册-v0.1.md"}
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
    keep = ("window", "radius", "budget", "predictor", "seed", "J_per_episode", "J_mean",
            "served", "served_late", "served_total", "alive_steps", "frames", "attempts",
            "link_q", "budget_reserved", "budget_attempted", "budget_wasted",
            "age_uav", "age_cart", "elapsed_s")
    slim = [{k: c[k] for k in keep} for c in (cells or [])]
    dest = write_result(RESULT_NAME, {**out, **(extra or {}), "cells_J": slim},
                        run_id=run_id, allow_overwrite=allow_overwrite)
    print(f"[evidence] {dest.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    sys.exit(main())
