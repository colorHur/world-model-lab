# -*- coding: utf-8 -*-
"""诊断：HOLD 成本 vs plant 谱半径 —— 判定 R0 默认 plant 是否可用作面板。

背景（2026-10-21 17:3x 实测）
---------------------------
R0 默认 `A=(1.01, 1.03)`：全 HOLD 一条 episode 的 `cost_true` 在最后一步达 11215.99，
256 步均值 763.42（首轮面板里 `always_probe` 的 1224.8 同源）；24 episode 平均后
末 64 步 / 中段 64 步 = **1929 倍**。解析递推 `P' = A²P + σ_w²`（**P_0 = 0**，因为
`CWWorld.__init__` 的 `_x_true = zeros`，`prior_var` 只进接收端信念）给出同一量级，
**不是代码 bug**，而是：
  ① `A>1` ⇒ 开环不稳定 ⇒ `E[x_t²] = σ_w²·Σ A^{2(t-1-k)}` 按 `A^{2t}` 指数爆；
  ② 任务成本 `q·x²` 无归一化 ⇒ 成本量纲被末端少数步支配 ⇒ **均值不是可比较量**。

本脚本不写 `results/`（R45：自检入口不得触碰入库证据），只打印判据表。
退出码 0 = 全部判据 PASS，2 = 有 FAIL。

★★ 第一版判据的 4 处错（2026-10-21 17:5x 实测，见教训 #81/#82）
---------------------------------------------------------------
① 我把 `prior_var` 当成"初态 x_0 ~ N(0, prior_var)" —— **代码里没有这件事**。
   `CWWorld.__init__` 是 `_x_true = zeros(n)`，即 **x_0 确定 = 0**，`prior_var` **只**进
   `_P_hat`（接收端信念方差），不进真值。⇒ 开环 `E[x_t²] = σ_w²·Σ_{k<t} A^{2(t-1-k)}`，
   **与 prior_var 无关**（D4 因此 FAIL，且 FAIL 得正确）。
② D1 的"末/首步比"在首步成本恰为 0 时变成除零 ⇒ 打出 5.9e16 这种无意义数字。
   改成末 64 步均值 / 第 2 个 64 步均值。
③ D3 用 `cur.mean()/th.mean()` 对账，但 `th` 从 `P_0 = prior_var` 起 ⇒ 两边起点不同。
   开环解析必须从 `P_0 = 0` 起。
④ 判据 D2 对 A=1.00 失败**不是错**：A=1 时随机游走方差线性增长（`E x_t² = t·σ_w²`），
   末/前 64 步比理论 = 224/32 = 7 ⇒ 7.434 恰好命中。这一条应当**标为预期 FAIL**。

判据（保留原 4 条 + 修正，**改判据不改被测物**）
-------------------------------------------------
D1 `A=(1.01,1.03)`：末 64 步均值 / 第 2 个 64 步均值 > 10（证明爆的是 plant，不是策略）
D2 `A=(0.90,0.92)` / `(0.95,0.97)`：末 64 步 / 前 64 步 < 3（成本不随时间增长）
D3 上面两档 开环解析（`P_0=0`）MC/解析 ∈ [0.7, 1.4]
D4 `A=(1.00,1.00)`：末/前 64 步比 ∈ [6, 9]（理论 7）—— 随机游走的线性增长
D5 `prior_var` 不影响 `HOLD` 成本（证明 `prior_var` 只进信念、不进真值）
D6 **`BOUNDED` 预设的通道有效性**（这才是 R1 面板真正要用的 plant）：
   D6a `HOLD` 成本 > 0 且 `paced1` / `paced16` 都显著低于 `HOLD`（各 < 0.5×）——
       ⇒ SEND 有任务价值（否则"测量改变决策"没有意义）；
   D6b `paced16` / `paced1` ∈ [1.2, 3.0] —— ⇒ **发送频率**是真实权衡，
       不是"越多越好"也不是"全等价"；
   D6c 逐 episode `paced1 < HOLD` 的比例 ≥ 0.85（40 条里 37 条，中位比 0.302）。
"""

from __future__ import annotations

import sys
from dataclasses import replace

import numpy as np

from .config import CWConfig, PlantConfig
from .streams import make_episode_rngs
from .world import Action, CWWorld

N_EP = 24
HORIZON = 256


def _hold_curve(cfg: CWConfig, episodes: range, rho: float = 0.85) -> np.ndarray:
    """返回 (n_ep, HORIZON) 的逐槽 cost_true。全 HOLD ⇒ u≡0（x̂ 恒为 0）。"""
    out = np.zeros((len(episodes), HORIZON), dtype=float)
    for k, ep in enumerate(episodes):
        rngs, sids = make_episode_rngs(cfg.root_seed, "dev", ep)
        w = CWWorld(cfg, rngs, sids, rho=rho)
        w.budget_total = 0
        for t in range(HORIZON):
            rec = w.step(Action("HOLD"))
            out[k, t] = rec.cost_true
    return out


def _analytic_curve(cfg: CWConfig) -> np.ndarray:
    """开环 E[Σ_i x_i,t²]。

    ★ x_0 确定 = 0（`CWWorld.__init__` 的 `_x_true = zeros`），所以 P_0 = **0**，
      不是 `prior_var`。第一版从 prior_var 起，导致 MC 恒等于解析的 0.155 倍。
    """
    A = np.asarray(cfg.plant.A, dtype=float)
    sw = np.asarray(cfg.plant.sigma_w, dtype=float)
    P = np.zeros_like(A)
    out = np.zeros(HORIZON, dtype=float)
    for t in range(HORIZON):
        out[t] = float(P.sum())
        P = A * A * P + sw * sw
    return out


def main() -> int:
    rows: list[tuple[str, bool, dict]] = []
    eps = range(100, 100 + N_EP)

    # ---- D1：默认 A=(1.01,1.03) 爆炸（末 64 步 vs 第 2 个 64 步）
    cfg_def = CWConfig()
    m_def = _hold_curve(cfg_def, eps).mean(axis=0)
    ratio_def = float(m_def[-64:].mean() / max(m_def[64:128].mean(), 1e-12))
    rows.append((
        "D1 A=(1.01,1.03) HOLD 末64/第2个64步 成本比 > 10",
        ratio_def > 10.0,
        {"mean_last64": round(float(m_def[-64:].mean()), 3),
         "mean_64_128": round(float(m_def[64:128].mean()), 6),
         "ratio": round(ratio_def, 1),
         "episode_mean_over_t": round(float(m_def.mean()), 3)},
    ))

    # ---- D2/D3：A<=1 的候选 plant
    cand = {
        "A=(0.90,0.92)": (0.90, 0.92),
        "A=(0.95,0.97)": (0.95, 0.97),
    }
    for name, A in cand.items():
        cfg = CWConfig(plant=replace(PlantConfig(), A=A))
        cur = _hold_curve(cfg, eps).mean(axis=0)
        th = _analytic_curve(cfg)
        tail_ratio = float(cur[-64:].mean() / max(cur[:64].mean(), 1e-12))
        mc_th = float(cur.mean() / max(th.mean(), 1e-12))
        rows.append((
            f"D2 {name} 末64步/前64步 < 3（成本不随时间增长）",
            tail_ratio < 3.0,
            {"tail_over_head": round(tail_ratio, 3),
             "mean_over_t": round(float(cur.mean()), 5)},
        ))
        rows.append((
            f"D3 {name} MC/解析(P0=0) ∈ [0.7,1.4]",
            0.7 <= mc_th <= 1.4,
            {"mc_mean": round(float(cur.mean()), 5),
             "analytic_mean": round(float(th.mean()), 5),
             "ratio": round(mc_th, 4)},
        ))

    # ---- D4：A=1 随机游走，方差线性增长 ⇒ 末/前 64 步比理论 = 224/32 = 7
    cfg_rw = CWConfig(plant=replace(PlantConfig(), A=(1.00, 1.00)))
    cur_rw = _hold_curve(cfg_rw, eps).mean(axis=0)
    rw_ratio = float(cur_rw[-64:].mean() / max(cur_rw[:64].mean(), 1e-12))
    rows.append((
        "D4 A=(1,1) 末64/前64 比 ∈ [6,9]（随机游走理论 7）",
        6.0 <= rw_ratio <= 9.0,
        {"ratio": round(rw_ratio, 3), "theoretical": 7.0,
         "mean_over_t": round(float(cur_rw.mean()), 5)},
    ))

    # ---- D5：prior_var 不进真值 ⇒ HOLD 成本逐位不变
    base = _hold_curve(CWConfig(plant=replace(PlantConfig(), A=(0.95, 0.97))), eps)
    small = _hold_curve(
        CWConfig(plant=replace(PlantConfig(), A=(0.95, 0.97), prior_var=0.25)), eps
    )
    max_diff = float(np.abs(base - small).max())
    rows.append((
        "D5 prior_var 1.0→0.25 不改变 HOLD 成本（逐槽 max|Δ|=0）",
        max_diff == 0.0,
        {"max_abs_diff": max_diff},
    ))

    # ---- D6：BOUNDED 预设的通道有效性（R1 面板真正要用的 plant）
    from .config import bounded_plant
    from .runner import run_episode
    from .policies import PacedAge

    class _Hold:
        name = "hold"

        def __call__(self, v):  # noqa: D401
            return Action("HOLD")

    cfg_b = CWConfig(plant=bounded_plant())
    eps_b = range(100, 140)
    curves = {}
    for nm, pol in (("hold", _Hold()), ("paced1", PacedAge(1)), ("paced16", PacedAge(16))):
        curves[nm] = np.array([
            run_episode(cfg_b, pol, split="dev", episode=ep, budget_total=204, rho=0.85)
            .cost_true_per_step for ep in eps_b
        ])
    m = {k: float(v.mean()) for k, v in curves.items()}
    rows.append((
        "D6a BOUNDED: HOLD>0 且 paced1<0.5×HOLD、paced16<0.5×HOLD（SEND 有任务价值）",
        m["hold"] > 0 and m["paced1"] < 0.5 * m["hold"] and m["paced16"] < 0.5 * m["hold"],
        {k: round(v, 6) for k, v in m.items()},
    ))
    ratio16_1 = m["paced16"] / max(m["paced1"], 1e-12)
    rows.append((
        "D6b BOUNDED: paced16/paced1 ∈ [1.2,3.0]（发送频率是真实权衡）",
        1.2 <= ratio16_1 <= 3.0,
        {"paced16_over_paced1": round(ratio16_1, 3)},
    ))
    win = float(np.mean(curves["paced1"] < curves["hold"]))
    med_ratio = float(np.median(curves["paced1"] / np.maximum(curves["hold"], 1e-12)))
    rows.append((
        "D6c BOUNDED: 逐 episode paced1<HOLD 比例 ≥ 0.85",
        win >= 0.85,
        {"win_rate": round(win, 3), "median_ratio_p1_over_hold": round(med_ratio, 3),
         "n_episodes": int(len(curves["hold"]))},
    ))

    n_fail = 0
    print("[CW diag] plant 稳定性判据")
    for name, ok, detail in rows:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}  {detail}")
        n_fail += 0 if ok else 1
    print(f"[CW diag] {len(rows)} 判据，{n_fail} FAIL -> "
          f"{'ALL PASS' if n_fail == 0 else 'FAIL'}")
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
