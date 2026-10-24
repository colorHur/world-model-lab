# -*- coding: utf-8 -*-
"""`CW-R2.2b` 判据的**二次核查**（零 episode）：α 敏感性 / 多重比较 / **分层内 `beat` 检验的选择偏倚** / 最小显著计数。

为什么必须做这一轮
------------------
`R2.2` 主轮的 LOW 层判出 `PANEL-SEPARATES`（9/10 符号、9/10 过地板，两个 p 都是 0.0107）。
写完结论前必须自问：**按 `floor` 分层之后，`beat` 的零假设还是 `P=1/2` 吗？**
答案是否 —— 条件在"地板小"的子集上，`|Δ| > floor` 本来就更容易成立。这不是阈值问题，是**选择效应**。

本脚本三件事：
1. `D1` 身份：用落盘计数复算 α=0.05 的判词，必须与主轮逐字相同（防"二次核查口径漂了"）。
2. `D2` α 敏感性 + 多重比较（Bonferroni over 4 个预注册检验）。
3. `D3` **合成证明**：造 `H0`（`Δ` 与 `floor` 同分布、符号掷硬币）⇒ 全样本 `beat` 率 ≈0.5，
   但 **LOW 层 ≫0.5、HIGH 层 ≪0.5** ⇒ 分层内 `beat` 检验**不可用**（要报，不是要改）。
4. `D4` 最小显著计数表 + 按观测比例的比例外推（**标注为外推，不是证据**）。

零 `run_episode`；被测物零改动。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import mech2
from .r1_14_exactbase import _git, _src_hash

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results" / "cw_v1"


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


def latest_run() -> Path:
    cands = sorted([p for p in RESULTS.glob("cw_r2_2_criteria_*") if p.is_dir()
                    and not p.name.endswith("_SUPERSEDED")])
    if not cands:
        raise FileNotFoundError("找不到 cw_r2_2_criteria_* 证据目录")
    return cands[-1]


def verdict_of(counts: dict, alpha: float) -> str:
    """用落盘计数复算判词（与 mech2 同一映射；`D1` 就是查它有没有漂）。"""
    n = int(counts["n_units"])
    h_sign = mech2.binom_p_ge(int(counts["n_pos"]), n) < alpha
    h_neg = mech2.binom_p_ge(int(counts["n_neg"]), n) < alpha
    h_beat = mech2.binom_p_ge(int(counts["n_beat"]), n) < alpha
    if h_sign and h_beat:
        return mech2.V_SEP
    if h_neg and h_beat:
        return mech2.V_SEP_REV
    if h_sign:
        return mech2.V_SIGN_ONLY
    if h_neg:
        return mech2.V_SIGN_ONLY_REV
    if h_beat:
        return mech2.V_MAG_ONLY
    return mech2.V_CANNOT


def min_k(n: int, alpha: float) -> int | None:
    """最小的 `k` 使 `P(X ≥ k | n) < alpha`；不存在 ⇒ `None`（这是"阈值按 n 缩放"的实际表）。"""
    for k in range(1, n + 1):
        if mech2.binom_p_ge(k, n) < alpha:
            return k
    return None


def selection_bias_sim(n_units: int = 20, reps: int = 4000, seed: int = 20261025) -> dict:
    """`D3`：`H0` 下（Δ 与 floor 同分布、符号随机）分层后 `beat` 率的分布。

    构造：`floor_u ~ |N(0,1)|`；`|Δ_u| ~ |N(0,1)|`（**同分布**）；`sign ~ ±1` 等概率。
    ⇒ 全样本 `P(beat)=0.5`（零假设成立）；把同一批数据按 `median(floor)` 分层后，
    `LOW` 层的 `P(beat)` 必须显著 **>0.5**，`HIGH` 层 **<0.5** —— 若如此，说明
    "分层内 `beat` 检验"在 `H0` 下就会假阳性，不能当证据。
    """
    rng = np.random.default_rng(seed)
    rec = {"overall": [], "low": [], "high": []}
    for _ in range(reps):
        f = np.abs(rng.normal(size=n_units))
        d_abs = np.abs(rng.normal(size=n_units))
        s = rng.choice([-1.0, 1.0], size=n_units)
        d = d_abs * s
        beat = (np.abs(d) > f).astype(float)
        cut = float(np.median(f))
        low = f <= cut
        rec["overall"].append(float(beat.mean()))
        rec["low"].append(float(beat[low].mean()))
        rec["high"].append(float(beat[~low].mean()))
    out = {}
    for k, v in rec.items():
        a = np.array(v)
        out[k] = {"mean": float(a.mean()),
                  "p05": float(np.percentile(a, 5)), "p95": float(np.percentile(a, 95))}
    # H0 下"LOW 层 beat 达到 9/10"的概率（就是 R2.2 主轮 LOW 层的那个数）
    hit = np.array([mech2.binom_p_ge(int(round(v * 10)), 10) < 0.05
                    for v in rec["low"]])
    out["h0_low_pass_rate_at_alpha05"] = float(hit.mean())
    out["n_units"] = n_units
    out["reps"] = reps
    out["seed"] = seed
    return out


def power_projection(rate_beat: float, n_list=(20, 30, 40, 60),
                     alpha: float = 0.05, reps: int = 4000,
                     seed: int = 77) -> list[dict]:
    """`D4`：**按观测比例的比例外推**（预注册里明写"不是证据"，只用于估预算）。"""
    rng = np.random.default_rng(seed)
    rows = []
    for n in n_list:
        k_need = min_k(n, alpha)
        if k_need is None:
            rows.append({"n": n, "k_need": None, "pass_rate": None})
            continue
        draws = rng.binomial(n, rate_beat, size=reps)
        rows.append({"n": n, "k_need": int(k_need),
                     "expected_k": float(n * rate_beat),
                     "pass_rate": float((draws >= k_need).mean()),
                     "binom_p_at_need": mech2.binom_p_ge(k_need, n)})
    return rows


def analyse(run_dir: Path) -> dict:
    s = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    blocks = {k: s["blocks_v2"][k] for k in ("ALL", "P1", "P2")}
    blocks["LOW"] = s["strata"]["LOW"]
    blocks["HIGH"] = s["strata"]["HIGH"]["descriptive_v2"]

    # ---- D1 身份
    d1 = {k: {"stored": blocks[k]["verdict"], "recomputed": verdict_of(blocks[k], mech2.ALPHA)}
          for k in blocks}
    d1_ok = all(v["stored"] == v["recomputed"] for v in d1.values())

    # ---- D2 α 敏感性（含 Bonferroni over 4 个预注册检验：ALL/P1/P2/LOW）
    alphas = [0.05, 0.025, 0.05 / 4, 0.01, 0.005]
    d2 = {f"{a:.6g}": {k: verdict_of(blocks[k], a) for k in blocks} for a in alphas}

    # ---- D3 选择偏倚（合成，零 episode）
    d3 = selection_bias_sim()

    # ---- D4 最小显著计数 + 外推
    ks = {n: min_k(n, mech2.ALPHA) for n in (10, 15, 20, 30, 40, 60)}
    rate = blocks["ALL"]["n_beat"] / blocks["ALL"]["n_units"]
    d4 = power_projection(rate)

    return {"run": run_dir.name, "blocks": blocks, "D1_identity": d1, "D1_ok": d1_ok,
            "D2_alpha": d2, "D2_alphas": alphas, "D3_selection_bias": d3,
            "D4_min_k": ks, "D4_obs_beat_rate": rate, "D4_projection": d4,
            "inputs": s["inputs"], "head": s["head"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="CW-R2.2b 判据二次核查（零 episode）")
    ap.add_argument("--run", default=None, help="R2.2 证据目录名；缺省取最新")
    args = ap.parse_args(argv)
    t0 = time.perf_counter()
    run_dir = RESULTS / args.run if args.run else latest_run()
    ana = analyse(run_dir)
    print(f"[INPUT] {ana['run']}")
    print(f"[D1] identity_ok={ana['D1_ok']} " +
          " ".join(f"{k}:{v['stored']}=={v['recomputed']}" for k, v in ana["D1_identity"].items()))
    for a in ana["D2_alphas"]:
        print(f"[D2 a={a:.6g}] " + " ".join(f"{k}={v}" for k, v in ana["D2_alpha"][f"{a:.6g}"].items()))
    d3 = ana["D3_selection_bias"]
    print(f"[D3] H0 下 beat 率: overall={d3['overall']['mean']:.4f} "
          f"LOW={d3['low']['mean']:.4f} HIGH={d3['high']['mean']:.4f} "
          f"| H0 下 LOW 层过 α=.05 的概率={d3['h0_low_pass_rate_at_alpha05']:.4f}")
    print(f"[D4] min_k={ana['D4_min_k']} obs_rate={ana['D4_obs_beat_rate']:.4f}")
    for r in ana["D4_projection"]:
        print(f"     n={r['n']} k_need={r['k_need']} expected={r['expected_k']:.2f} "
              f"pass_rate={r['pass_rate']}")

    head = _git(["rev-parse", "HEAD"])
    wt = _git(["hash-object", "experiments/cw_v1/r2_2b_sens.py"])
    dirty = _git(["status", "--porcelain"])
    outdir = RESULTS / f"cw_r2_2b_sens_{_now_utc()}_{_src_hash()}"
    outdir.mkdir(parents=True, exist_ok=False)
    total = round(time.perf_counter() - t0, 2)
    summary = {"stage": "CW-R2.2b", "kind": "CRITERIA-SECONDARY-AUDIT", "zero_episode": True,
               "source_run": ana["run"], "head": head,
               "D1_identity": ana["D1_identity"], "D1_ok": ana["D1_ok"],
               "D2_alpha": ana["D2_alpha"], "D2_alphas": ana["D2_alphas"],
               "D3_selection_bias": ana["D3_selection_bias"],
               "D4_min_k": {str(k): v for k, v in ana["D4_min_k"].items()},
               "D4_obs_beat_rate": ana["D4_obs_beat_rate"],
               "D4_projection": ana["D4_projection"],
               "inputs": ana["inputs"], "total_elapsed_s": total}
    (outdir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    meta = {"command": "python -m experiments.cw_v1.r2_2b_sens", "head": head,
            "src_hash": _src_hash(), "worktree_sha_r2_2b": wt,
            "timestamp_local": datetime.now().astimezone().isoformat(),
            "python": os.sys.version.split()[0], "numpy": np.__version__,
            "git_dirty_before_run": dirty.splitlines(), "zero_episode": True,
            "protocol": "CW-POMDP-v1.1", "total_elapsed_s": total}
    (outdir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    print(f"[OUT] {outdir.name} total={total}s")
    return 0 if ana["D1_ok"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
