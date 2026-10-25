# -*- coding: utf-8 -*-
"""`CW-R2.8b`：**补充诊断**（跑后，无判词）—— `B` 支的 `h_B` 是不是只由 `A>1` 那一侧撑起来的？

为什么必须做这条
----------------
主面板（`r2_8_space.py`）里 `w_A=0.02, p=0.5` 的 `h_B = 45.79%`。这个量级对一个 **2%** 的模型误差来说很大，
必须排除一个解释：`A_true ~ U(0.98,1.02)` 是**双侧**，而 `A>1` 的轨迹会涨到 `max|x|≈8.5`
⇒ 聚合 MSE 可能被这一侧的少数 episode 主导，那样"45%"就不是"参数不确定性普遍值钱"，
而是"在不稳定侧值钱"。

做法：按 `A_true < 1` / `> 1` 分两桶，各自算 `h_B` 与 `MSE`。
★ 无判词；`h_B` 的正负两侧都要报（负 = 边缘化反而更差）。
★ 自校验：本文件独立复算的**全样本** `h_B` 必须与主面板 `panel_b.json` 逐位一致（闸 `D1`），
  否则下面的分桶数字不可信。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .config import mainline_config
from .r1_14_exactbase import _git, _now_utc, _src_hash
from .r2_8_space import B_T, B_THETA, b_cell

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results" / "cw_v1"
MAIN_RUN = "cw_r2_8_space_20261025T120711_31445505a9e40709"


def split_cell(w_A: float, p_deliver: float, sigma_w: float, *, n_ep: int, T: int,
               seed: int, cfg) -> dict:
    """与 `b_cell` 同一套数与同一随机流；额外返回逐 episode 的 `A_true` 与两臂 MSE。"""
    lo, hi = 1.0 - w_A, 1.0 + w_A
    theta = np.linspace(lo, hi, len(B_THETA))
    rng = np.random.default_rng(seed)
    A_true = rng.uniform(lo, hi, n_ep)
    x0 = rng.standard_normal(n_ep) * float(np.sqrt(cfg.plant.prior_var))
    W = rng.standard_normal((n_ep, T)) * sigma_w
    Z = rng.standard_normal((n_ep, T)) * float(cfg.plant.sigma_meas)
    rng.standard_normal((n_ep, T))          # ★ 必须抽掉 `b_cell` 里的 `Z2`，否则 D 流错位（闸 D1 第一次就抓到了）
    D = rng.random((n_ep, T)) < p_deliver
    var_m = float(cfg.plant.sigma_meas) ** 2
    var_w = float(sigma_w) ** 2
    K = len(theta)

    x = np.empty((n_ep, T))
    xt = x0.copy()
    for t in range(T):
        x[:, t] = xt
        xt = A_true * xt + W[:, t]

    m = np.zeros(n_ep); P = np.full(n_ep, float(cfg.plant.prior_var))
    out_p = np.empty((n_ep, T))
    for t in range(T):
        m = 1.00 * m; P = 1.00 * P + var_w
        y = x[:, t] + Z[:, t]
        g = P / (P + var_m)
        m = np.where(D[:, t], m + g * (y - m), m)
        P = np.where(D[:, t], (1.0 - g) * P, P)
        out_p[:, t] = m

    mk = np.zeros((n_ep, K)); Pk = np.full((n_ep, K), float(cfg.plant.prior_var))
    lw = np.full((n_ep, K), -np.log(K))
    out_k = np.empty((n_ep, T))
    for t in range(T):
        mk = theta[None, :] * mk; Pk = theta[None, :] ** 2 * Pk + var_w
        y = (x[:, t] + Z[:, t])[:, None]
        S = Pk + var_m
        ll = -0.5 * ((y - mk) ** 2 / S + np.log(S))
        lw = np.where(D[:, t][:, None], lw + ll, lw)
        lw = lw - lw.max(axis=1, keepdims=True)
        w = np.exp(lw); w = w / w.sum(axis=1, keepdims=True)
        lw = np.log(w + 1e-300)
        g = Pk / S
        mk = np.where(D[:, t][:, None], mk + g * (y - mk), mk)
        Pk = np.where(D[:, t][:, None], (1.0 - g) * Pk, Pk)
        out_k[:, t] = (w * mk).sum(axis=1)

    e_p = ((out_p - x) ** 2).mean(axis=1)
    e_k = ((out_k - x) ** 2).mean(axis=1)
    res = {"h_B_all": float(1.0 - e_k.mean() / e_p.mean()),
           "mse_plug_all": float(e_p.mean()), "mse_mmkf_all": float(e_k.mean())}
    for tag, mask in (("A<1", A_true < 1.0), ("A>1", A_true > 1.0)):
        n = int(mask.sum())
        res[tag] = {"n": n,
                    "mse_plug": float(e_p[mask].mean()), "mse_mmkf": float(e_k[mask].mean()),
                    "h_B": (float(1.0 - e_k[mask].mean() / e_p[mask].mean()) if n else None)}
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--b-ep", type=int, default=1000)
    ap.add_argument("--main-run", default=MAIN_RUN)
    args = ap.parse_args(argv)
    t0 = time.perf_counter()
    cfg = mainline_config()
    main_dir = RESULTS / args.main_run
    ref = json.loads((main_dir / "panel_b.json").read_text("utf-8"))

    out = []
    for i, wA in enumerate((0.005, 0.02)):
        for j, p in enumerate((0.5, 0.1, 0.02)):
            seed = 20261025 + 1000 * i + 10 * j
            r = split_cell(wA, p, 0.02, n_ep=int(args.b_ep), T=B_T, seed=seed, cfg=cfg)
            r.update({"w_A": wA, "p_deliver": p, "seed": seed})
            out.append(r)

    # ---- 闸 D1：全样本 h_B 必须与主面板逐位一致
    d1 = []
    for r, q in zip(out, ref):
        assert abs(r["w_A"] - q["w_A"]) < 1e-15 and abs(r["p_deliver"] - q["p_deliver"]) < 1e-15
        d1.append(abs(r["h_B_all"] - q["h_B"]))
    gates = {"D1": {"pass": max(d1) <= 1e-12, "max_abs_dh": max(d1),
                    "note": "本文件独立复算的全样本 h_B 必须与 panel_b.json 逐位一致"},
             "D2": {"pass": all(0 < sum(r[t]["n"] for t in ("A<1", "A>1")) <= int(args.b_ep)
                                for r in out),
                    "note": "两桶之和 ≤ n_ep 且非空（A_true 恰为 1.0 的概率为 0）"}}
    n_fail = sum(1 for g in gates.values() if not g["pass"])

    outdir = RESULTS / f"cw_r2_8b_diag_theta_{_now_utc()}_{_src_hash()}"
    outdir.mkdir(parents=True, exist_ok=False)
    (outdir / "diag_by_theta.json").write_text(
        json.dumps({"cells": out, "gates": gates,
                    "main_run": args.main_run,
                    "b_ep": int(args.b_ep),
                    "git_head": _git(["rev-parse", "HEAD"], allow_fail=True),
                    "elapsed_s": time.perf_counter() - t0},
                   ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps({"outdir": str(outdir), "n_fail": n_fail, "max_abs_dh_D1": max(d1)},
                     ensure_ascii=False, indent=2))
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
