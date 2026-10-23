# -*- coding: utf-8 -*-
"""CW-R1.7.4 三臂尾部敏感性（`paced` / `dp@legacy` / `dp@r173`）。

设计要点（与预注册 `docs/research/CW-R1.7.4-三臂敏感性-预注册-v0.1.md` 对应）
--------------------------------------------------------------------------
1. **不重写判据**：三臂直接调用 `plan_gates.g8_pacing` / `g4_m_stability`（以及
   `run_all()` 作基准），只临时改两个**模块级默认值**：
     `plan_ref.TAIL_DEFAULT`（"paced"/"dp"）
     `plan_dp.GRID_DEFAULT`（"legacy"/"r173"）
   ⇒ `paced` 臂必须与 `run_all()` 的同名项**逐位相同**（T-A2，否则判 `HARNESS-MISMATCH`）。
   若重写判据，三臂比较的就不是同一个判据；这是本模块存在的理由。

2. `h_plan` / `m_samples` / `seed` / `rho` / `B/T` / episode **全部取自判据实现自身**，
   一个都不改。★ 预注册 §4 已声明：STATE §7.2 写的 `h_plan=1` 不能用（`H=1` ⇒
   `PROBE≡HOLD` ⇒ `n_probe≡0` 是 `G3b` 钉死的结构事实，三臂会退化成同一策略）。

3. `G8a` 的 `_send_profile` 对同一 seed/同集**不依赖** `tail`（它是历史动作的计数），
   所以"CV 变了"只可能来自动作序列变了 —— 这正是要测的。

判词见预注册 §8；本轮不读 `N1/N2`、不宣称"测量值不值"。
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import re
import time
import tracemalloc
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from . import config as C
from . import plan_dp as D
from . import plan_ref as P
from . import plan_gates as G
from .gates import Gate
from .plan_ref import make
from .runner import run_episode

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results" / "cw_v1"

#: 三臂（`tail`, `dp_grid`）。`paced` 臂的 `grid` 无关（`tail != "dp"` 时不建 DP）。
ARMS = (
    ("paced", "paced", None),
    ("dp_legacy", "dp", "legacy"),
    ("dp_r173", "dp", "r173"),
)
#: 阳性对照臂（活体：预算轴被截到 8 ⇒ 值函数饱和 ⇒ 配速必然垮）。
CTRL_B_CAP = 8

#: 8 集次级面板（只报不判）：与 `G8a` 同配置（h/m/seed/rho 一律取自判据实现）。
PANEL_CFG = {"h_plan": 5, "m_samples": 8, "seed": 1100, "rho": 0.85, "ratio": 0.5}
PANEL_EPS = tuple(range(100, 108))


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


@contextlib.contextmanager
def _tail_ctx(tail: str, grid: str | None, b_cap: int | None = None):
    """临时改模块级默认值（`make()` 不传 tail/dp_grid 时生效），`finally` 恢复原值。"""
    old_tail, old_grid = P.TAIL_DEFAULT, D.GRID_DEFAULT
    old_make = P.make
    try:
        P.TAIL_DEFAULT = str(tail)
        if grid is not None:
            D.GRID_DEFAULT = str(grid)
        if b_cap is not None:
            def _patched_make(cfg, **kw):
                kw.setdefault("dp_b_cap", int(b_cap))
                return old_make(cfg, **kw)
            P.make = _patched_make
            G.make = _patched_make
        yield
    finally:
        P.TAIL_DEFAULT, D.GRID_DEFAULT = old_tail, old_grid
        P.make, G.make = old_make, old_make


_GATE_RE = re.compile(r"^(G8a2|G8a|G8b|G4)\b")


def _pick(g: Gate, name: str) -> dict:
    """按**显式正则**取闸行（不用 `split(" ")[0]`，历史上被全角括号坑过）。"""
    for r in g.rows:
        m = _GATE_RE.match(r["name"])
        if m and m.group(1) == name:
            return r
    raise KeyError(f"闸里没有 {name}；现有 {[r['name'][:12] for r in g.rows]}")


def _run_two_gates() -> dict:
    """跑 `g8_pacing` + `g4_m_stability`，抽出 G8a/G8a2/G8b/G4 四行。"""
    g = Gate()
    G.g8_pacing(g)
    G.g4_m_stability(g)
    out = {}
    for name in ("G8a", "G8a2", "G8b", "G4"):
        r = _pick(g, name)
        out[name] = {"pass": bool(r["pass"]), "detail": r["detail"],
                     "name": r["name"]}
    # 抽出标量，方便并排比较
    out["_scalars"] = {
        "cv_paced": float(out["G8a"]["detail"]["paced"]["cv"]),
        "aoi_paced": float(out["G8a2"]["detail"]["aoi_paced"]),
        "agreement": float(out["G4"]["detail"]["agreement"]),
        "median_rel": float(out["G4"]["detail"]["median_rel_cost_diff"]),
        "cv_nopace": float(out["G8b"]["detail"]["cv_nopace"]),
        "pass_G8a": out["G8a"]["pass"], "pass_G8a2": out["G8a2"]["pass"],
        "pass_G4": out["G4"]["pass"],
    }
    return out


def _panel8() -> dict:
    """8 集稳健性（只报不判）：逐集 CV / aoi / cost_true / n_probe。"""
    cfg = C.CWConfig(plant=C.bounded_plant())
    budget = cfg.protocol.budget(PANEL_CFG["ratio"])
    rows = []
    t0 = time.perf_counter()
    for ep in PANEL_EPS:
        pol = G._Rec(make(cfg, arm="probe", h_plan=PANEL_CFG["h_plan"],
                          m_samples=PANEL_CFG["m_samples"],
                          seed=PANEL_CFG["seed"] - 100 + ep,
                          rho=PANEL_CFG["rho"], pacing=True))
        r = run_episode(cfg, pol, split="dev", episode=ep, budget_total=budget,
                        rho=PANEL_CFG["rho"])
        prof = G._send_profile(pol.acts, cfg.protocol.horizon)
        rows.append({"episode": ep, "cv": G._cv(prof),
                     "aoi": float(r.mean_task_aoi),
                     "cost_true": float(r.cost_true_per_step),
                     "n_probe": int(r.n_probe), "n_attempted": int(r.n_attempted),
                     "profile": prof,
                     "probe_same_argmax": int(getattr(pol.inner, "probe_same_argmax", 0)),
                     "probe_diff_argmax": int(getattr(pol.inner, "probe_diff_argmax", 0))})
    return {"episodes": list(PANEL_EPS), "rows": rows,
            "cv_median": float(np.median([r["cv"] for r in rows])),
            "cv_all_below_030": bool(all(r["cv"] < 0.30 for r in rows)),
            "cost_median": float(np.median([r["cost_true"] for r in rows])),
            "aoi_median": float(np.median([r["aoi"] for r in rows])),
            "probe_per_ep_median": float(np.median([r["n_probe"] for r in rows])),
            "elapsed_s": round(time.perf_counter() - t0, 2)}


def _settle(bench: dict, arms: dict, ctrl: dict) -> dict:
    """按预注册 §2/§8 结算。"""
    checks = {}

    # ---- T-A1 重复性：paced 臂两次跑逐位相同
    a, b = arms["paced"]["run1"]["_scalars"], arms["paced"]["run2"]["_scalars"]
    diffs = {k: abs(a[k] - b[k]) for k in ("cv_paced", "aoi_paced", "agreement", "median_rel")}
    checks["T_A1_repeat_exact"] = {"pass": max(diffs.values()) <= 1e-12, "diffs": diffs}

    # ---- T-A2 口径恒等：paced 臂 vs run_all() 基准，逐位相同
    bs = bench["_scalars"]
    d2 = {k: abs(a[k] - bs[k]) for k in ("cv_paced", "aoi_paced", "agreement", "median_rel")}
    checks["T_A2_bench_identity"] = {
        "pass": max(d2.values()) <= 1e-12, "diffs": d2,
        "bench": {k: bs[k] for k in d2}, "arm_paced": {k: a[k] for k in d2},
        "rule": "paced 臂必须与 plan_gates.run_all() 的同名项逐位相同（rel ≤1e-12）"}

    # ---- T-C 阳性对照：nopace 在 G8a 上必须 FAIL（即 G8b 成立）
    checks["T_C_nopace_fails_G8a"] = {
        "pass": float(bs["cv_nopace"]) > 0.30,
        "cv_nopace": bs["cv_nopace"], "line": 0.30,
        "rule": "pacing=False 的 CV 必须 >0.30；否则判据对已知缺陷不敏感 ⇒ NO-POWER"}

    # ---- T-C2 dp 臂的活体对照：b_cap=8 ⇒ 预算轴截断 ⇒ G8a 必须 FAIL
    checks["T_C2_bcap8_fails_G8a"] = {
        "pass": (not ctrl["G8a"]["pass"]) and float(ctrl["_scalars"]["cv_paced"]) > 0.30,
        "cv": ctrl["_scalars"]["cv_paced"], "line": 0.30, "b_cap": CTRL_B_CAP,
        "rule": "dp@r173 且 dp_b_cap=8 ⇒ 值函数饱和 ⇒ CV 必须 >0.30（证明 dp 支路也有分辨力）"}

    # ---- T-B 三臂方向一致
    per = {}
    for name in ("G8a", "G8a2", "G4"):
        vs = {arm: bool(arms[arm]["run1"][name]["pass"]) for arm, _, _ in ARMS}
        per[name] = {"passes": vs, "agree": len(set(vs.values())) == 1}
    checks["T_B_direction_agree"] = {
        "pass": all(v["agree"] for v in per.values()), "per_gate": per,
        "rule": "G8a/G8a2/G4 每一项在三臂上同过或同不过"}

    # ---- 判词
    if not checks["T_A1_repeat_exact"]["pass"]:
        verdict, why = "HARNESS-MISMATCH", "paced 臂两次跑不一致 ⇒ 有未播种随机源"
    elif not checks["T_A2_bench_identity"]["pass"]:
        verdict, why = "HARNESS-MISMATCH", "paced 臂与 run_all() 基准不等 ⇒ 包装改了口径"
    elif not (checks["T_C_nopace_fails_G8a"]["pass"] and checks["T_C2_bcap8_fails_G8a"]["pass"]):
        verdict, why = "NO-POWER", "阳性对照未 FAIL ⇒ 判据无分辨力，整轮作废"
    elif not checks["T_B_direction_agree"]["pass"]:
        verdict, why = "TAILS-SWITCH", "三臂方向不一致 ⇒ 尾部形状改变行为，须先修尾部"
    else:
        verdict, why = "TAILS-AGREE", "三臂同过或同不过 ⇒ 尾部剩余价格偏差不是判据方向项"

    return {"verdict": verdict, "reason": why, "checks": checks,
            "n_fail": sum(1 for c in checks.values() if not c["pass"]),
            "n_checks": len(checks),
            "no_n1_n2": True, "no_threshold_change": True,
            "rule": "本轮不读 N1/N2、不改任何阈值、不宣称「测量值不值」"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="CW-R1.7.4 三臂尾部敏感性")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--skip-panel", action="store_true")
    ap.add_argument("--note", default="")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    tracemalloc.start()
    peak = {"mib": 0.0}

    def _tick(tag: str) -> None:
        cur, pk = tracemalloc.get_traced_memory()
        peak["mib"] = max(peak["mib"], pk / 2 ** 20)
        print(f"  [{time.perf_counter() - t0:7.1f}s] {tag} "
              f"(tracemalloc peak {peak['mib']:.0f} MiB, dp_builds={D.N_BUILDS})", flush=True)

    # ---------- 0 基准：默认口径（paced + legacy 网格）的 run_all
    print("[R1.7.4] 0/5 基准 run_all()（默认 TAIL_DEFAULT=paced）", flush=True)
    gb = G.run_all()
    bench = {"G8a": {"pass": _pick(gb, "G8a")["pass"], "detail": _pick(gb, "G8a")["detail"]},
             "G8a2": {"pass": _pick(gb, "G8a2")["pass"], "detail": _pick(gb, "G8a2")["detail"]},
             "G8b": {"pass": _pick(gb, "G8b")["pass"], "detail": _pick(gb, "G8b")["detail"]},
             "G4": {"pass": _pick(gb, "G4")["pass"], "detail": _pick(gb, "G4")["detail"]},
             "n_plan_gate_checks": gb.summary()["n_checks"],
             "n_plan_gate_fail": gb.summary()["n_fail"]}
    bench["_scalars"] = {
        "cv_paced": float(bench["G8a"]["detail"]["paced"]["cv"]),
        "aoi_paced": float(bench["G8a2"]["detail"]["aoi_paced"]),
        "agreement": float(bench["G4"]["detail"]["agreement"]),
        "median_rel": float(bench["G4"]["detail"]["median_rel_cost_diff"]),
        "cv_nopace": float(bench["G8b"]["detail"]["cv_nopace"]),
        "pass_G8a": bench["G8a"]["pass"], "pass_G8a2": bench["G8a2"]["pass"],
        "pass_G4": bench["G4"]["pass"]}
    _tick("基准完成")

    # ---------- 1~3 三臂
    arms: dict = {}
    for i, (tag, tail, grid) in enumerate(ARMS, start=1):
        print(f"[R1.7.4] {i}/5 臂 {tag}（tail={tail}, grid={grid}）", flush=True)
        nb0 = D.N_BUILDS
        with _tail_ctx(tail, grid):
            run1 = _run_two_gates()
            _tick(f"{tag} run1")
            run2 = None
            if tag == "paced":       # 只有 paced 臂需要重复性闸（它最便宜）
                run2 = _run_two_gates()
                _tick(f"{tag} run2")
        arms[tag] = {"tail": tail, "dp_grid": grid, "run1": run1, "run2": run2,
                     "dp_builds": D.N_BUILDS - nb0}

    # ---------- 4 阳性对照：dp@r173 + b_cap=8
    print(f"[R1.7.4] 4/5 阳性对照 dp@r173 b_cap={CTRL_B_CAP}", flush=True)
    nb0 = D.N_BUILDS
    with _tail_ctx("dp", "r173", b_cap=CTRL_B_CAP):
        ctrl = _run_two_gates()
    ctrl["dp_builds"] = D.N_BUILDS - nb0
    _tick("阳性对照完成")

    # ---------- 5 8 集次级面板（只报不判）
    panel = {}
    if not args.skip_panel:
        print("[R1.7.4] 5/5 8 集次级面板（只报不判）", flush=True)
        for tag, tail, grid in ARMS:
            nb0 = D.N_BUILDS
            with _tail_ctx(tail, grid):
                panel[tag] = _panel8()
            panel[tag]["dp_builds"] = D.N_BUILDS - nb0
            _tick(f"panel8 {tag}")

    verdict = _settle(bench, arms, ctrl)

    # ★ 附带记录：`pacing` 在 dp 支路是否是死参数（决定 `G8b` 的对照覆盖不到 dp 臂）
    #   读实现不读名字（台账 #31）：`plan_ref._terminal()` 在 `tail=="dp"` 时直接
    #   `task_P = self._dp_value(...)` 并 return，**整个分支不出现 `self.pacing`**。
    dead_pacing = {
        "pacing_read_in_dp_branch": False,
        "evidence": ("experiments/cw_v1/plan_ref.py::_terminal —— "
                     "if self.tail == \"dp\": task_P = self._dp_value(...) else: _paced_sumP(...)；"
                     "dp 支路不引用 self.pacing"),
        "consequence": ("G8b（pacing=False）的阴性对照**不覆盖** dp 臂 ⇒ "
                        "dp 臂的分辨力只能由 T-C2（dp_b_cap=8）提供"),
        "verified_by": "static-read",
    }

    tracemalloc.stop()
    elapsed = round(time.perf_counter() - t0, 2)
    print(f"[R1.7.4] 判词 {verdict['verdict']}（{verdict['reason']}）；"
          f"{verdict['n_fail']} fail / {verdict['n_checks']} checks；{elapsed} s；"
          f"peak {peak['mib']:.0f} MiB；dp_builds {D.N_BUILDS}")
    for k, v in verdict["checks"].items():
        print(f"  {'PASS' if v['pass'] else 'FAIL'}  {k}")
    print("  三臂标量：")
    for tag, _, _ in ARMS:
        s = arms[tag]["run1"]["_scalars"]
        print(f"    {tag:10s} cv={s['cv_paced']:.4f} aoi={s['aoi_paced']:.2f} "
              f"agree={s['agreement']:.3f} rel={s['median_rel']:.4f} "
              f"| G8a={s['pass_G8a']} G8a2={s['pass_G8a2']} G4={s['pass_G4']}")

    if not args.no_write:
        now = datetime.now(timezone(timedelta(hours=8)))
        cfg = C.CWConfig(plant=C.bounded_plant())
        run_id = (f"cw_r1_7_4_tails_{now.strftime('%Y%m%dT%H%M%S')}_"
                  f"{cfg.fingerprint()}_{_src_hash()}")
        out_dir = RESULTS / run_id
        out_dir.mkdir(parents=True, exist_ok=True)

        def _dump(name, obj):
            (out_dir / name).write_text(
                json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

        _dump("bench.json", bench)
        _dump("tail_arms.json", arms)
        _dump("control.json", ctrl)
        _dump("panel_tails.json", panel)
        _dump("pacing_in_dp.json", dead_pacing or {})
        _dump("verdict.json", verdict)
        _dump("meta.json", {
            "run_id": run_id, "kind": "TAIL-SENSITIVITY-NOT-PANEL-CONCLUSION",
            "protocol": C.CWConfig.PROTOCOL_VERSION,
            "prereg": "docs/research/CW-R1.7.4-三臂敏感性-预注册-v0.1.md",
            "arms": [{"tag": t, "tail": tl, "dp_grid": gr} for t, tl, gr in ARMS],
            "ctrl_b_cap": CTRL_B_CAP,
            "gate_params_from_impl": {
                "G8a/G8a2": {"h_plan": 5, "m_samples": 8, "seed": 1100,
                             "rho": 0.85, "ratio": 0.5, "episode": 100},
                "G4": {"h_plan": 5, "m_samples": [16, 32], "seed": 777,
                       "rho": G.RHO_DEFAULT, "ratio": 0.5, "episodes": [100, 101]}},
            "deviation_from_STATE_7_2": (
                "STATE §7.2 写 h_plan=1/m_samples=16；h_plan=1 时 PROBE≡HOLD（G3b 结构事实，"
                "n_probe≡0）⇒ 三臂退化成同一策略。故改用判据实现自身参数（见上）。"),
            "panel_cfg": PANEL_CFG, "panel_eps": list(PANEL_EPS),
            "config": cfg.to_dict(), "note": args.note,
            "at_asia_shanghai": now.isoformat(), "elapsed_s": elapsed,
            "peak_tracemalloc_mib": round(peak["mib"], 1),
            "dp_builds_total": D.N_BUILDS,
            "git_head": _git(["rev-parse", "HEAD"]),
            "git_status_porcelain": _git(["status", "--porcelain"]),
            "src_hash": _src_hash(),
            "disclaimer": (
                "三臂只是问「尾部价格偏差会不会改变判据方向」，不是面板科学结论；"
                "不读 N1/N2；不改任何阈值；不宣称「主动测量有价值/无价值」。"
                "8 集是开发诊断，不是确认集。"),
        })
        print(f"[R1.7.4] 证据落盘 results/cw_v1/{run_id}")

    return 0 if verdict["verdict"] in ("TAILS-AGREE", "TAILS-SWITCH") else 3


if __name__ == "__main__":
    raise SystemExit(main())
