# -*- coding: utf-8 -*-
"""CW-R1.1-2 第一件：**一步预测与协方差的解析验证**（`CW-PRED-v1.1` 验收面）。

为什么需要这一份
----------------
验收报告（`2026-10-22-全面验收与归档-14a411b.md` §4「exact 预测器和多时隙边界未通过新反例」）
指出：名为 `exact` 的预测器在**零噪声、初值完全匹配**时误差 0.0309，而名义预测器误差为 0。
一个"更精确"的模式不可能在精确可解的场景里更差 ⇒ 它一定做错了。
本模块把这件事**解析地**钉住，并按 v4.1 §2 的顺序：**这一件不过就停**。

本模块测什么（每条都写清"什么错误能让它 FAIL"）
------------------------------------------------
P1  零噪声 + 初值匹配 + 纯 HOLD：`x̂_{t+1} = A x̂_t + B u_t` 必须与真值**逐位相等**。
    ⇒ 能让它 FAIL 的：任何多做/少做一次映射、任何把噪声加一遍以上。
P1c P1 的阴性对照：把 `x̂_0` 人为偏 0.1 ⇒ 偏差必须非零（证明 P1 不是恒真）。
P2  零噪声 + 初值不匹配：偏差必须精确等于 `A^k δ_0`（预测器不能是"复制真值"）。
P3  有噪声 + 任意送达模式：`x̂` 与 `P̂` 必须与**独立的教科书式 Kalman 复算**逐位相同。
    ⇒ 能让它 FAIL 的：校正对象错一拍、先校正后滚动顺序写反、gain 用错方差。
P4  协方差校准（大样本）：`E[e²] / E[P̂]` 必须落在抽样噪声允许的带内。
    ⇒ 能让它 FAIL 的：P̂ 系统性偏大/偏小（如漏掉 σ_w²、用 Riccati 的 P_ss 顶替）。
P5  零噪声夹具**不是**合法协议配置：`v11_validate()` 必须拒绝它。
    ⇒ 防止"为了过闸把 σ_w 调成 0 然后拿去跑面板"。
P6  `exact` 两条入口都必须被显式拒绝（构造 / `run_episode`）。
P7  撤回分支的缺陷**解析可算**：复刻旧实现，零噪声下第 2 步偏差必须 == `B·K·x_0`。
P8  物理不变式：`P̂ ≥ σ_w²`（每一步、每一用户）；观测只会降方差。

★ 本模块不读 `results/`、不写 `results/`；证据由 `r1_4_contract.py` 统一落盘。
"""

from __future__ import annotations

import numpy as np

from . import config as C
from .gates import Gate
from .streams import make_episode_rngs
from .world import Action, CWWorld


# ---------------------------------------------------------------- 夹具
def zero_noise_cfg(**plant_over) -> C.CWConfig:
    """零噪声解析夹具。

    ★ 这是**单元测试夹具**，不是协议配置：`v11_validate()` 会拒绝它（P5 就是验这个）。
    """
    return C.CWConfig(plant=C.bounded_plant(sigma_w=(0.0, 0.0), sigma_meas=0.0, **plant_over))


def _world(cfg: C.CWConfig, *, x0=None, P0=None, episode: int = 100,
           rho: float = 0.85, budget: int = 10_000) -> CWWorld:
    rngs, sids = make_episode_rngs(cfg.root_seed, "dev", episode)
    w = CWWorld(cfg, rngs, sids, rho=rho)
    w.budget_total = int(budget)
    if x0 is not None:
        w._x_true = np.asarray(x0, dtype=float).copy()
        w._x_hat = np.asarray(x0, dtype=float).copy()
    if P0 is not None:
        w._P_hat = np.asarray(P0, dtype=float).copy()
    return w


def replay_kalman(records, user: int, cfg: C.CWConfig, K: np.ndarray,
                  prior_var: float) -> list[tuple[float, float]]:
    """**独立**的标量 Kalman 复算（教科书顺序：predict →（送达则）update at t → roll to t+1）。

    只吃"策略自己能看到的东西"：动作序列、ACK、送达的 `y_t`、自己的 `u_t`。
    与被测实现**不共用任何代码路径**，用来对账 `x̂` 与 `P̂`。
    """
    A = np.asarray(cfg.plant.A, dtype=float)
    B = float(cfg.plant.B)
    sw2 = np.asarray(cfg.plant.sigma_w, dtype=float) ** 2
    sm2 = float(cfg.plant.sigma_meas) ** 2
    n = len(A)
    x = np.zeros(n)
    P = np.full(n, float(prior_var))
    out: list[tuple[float, float]] = []
    for r in records:
        u = -K * x                       # 与 step 内部一致：u_t 用槽开始时刻的估计
        x_prior, P_prior = x.copy(), P.copy()
        # (1) 推进到 t+1（无观测时的继续预测）
        x = A * x_prior + B * u
        P = A * A * P_prior + sw2
        # (2) 若本槽送达：校正的是 **t 时刻**，再用已知 u_t 滚到 t+1
        if r.action == "SEND" and r.user == user and r.data_delivered:
            g = P_prior[user] / (P_prior[user] + sm2)
            xc = x_prior[user] + g * (float(r.y_t) - x_prior[user])
            Pc = (1.0 - g) * P_prior[user]
            x[user] = A[user] * xc + B * u[user]
            P[user] = A[user] ** 2 * Pc + sw2[user]
        out.append((float(x[user]), float(P[user])))
    return out


def legacy_exact_trace(cfg: C.CWConfig, x0: float, K: float, steps: int):
    """复刻**已撤回**的 `exact` 分支（故意保留缺陷，用来证明"撤回不是口味问题"）。

    旧实现每个时隙做两次名义映射：
        x̂_{t+1} = A(A x̂_t + B u_{t-1}) + B u_t ,  P ← A⁴P + A²σ_w² + σ_w²
    它假设"一个时隙物理推进两次"，而本环境一个时隙只推进一次。
    """
    A = float(cfg.plant.A[0])
    B = float(cfg.plant.B)
    sw2 = float(cfg.plant.sigma_w[0]) ** 2
    x, P, u_prev = float(x0), 0.0, 0.0
    xs, Ps = [], []
    for _ in range(int(steps)):
        u = -K * x
        P_prior = P
        x = A * (A * x + B * u_prev) + B * u
        P = A ** 4 * P_prior + A ** 2 * sw2 + sw2
        u_prev = u
        xs.append(float(x))
        Ps.append(float(P))
    return xs, Ps


# ---------------------------------------------------------------- 闸
def run_all(*, n_calib: int = 1000, seed_calib: int = 1000) -> Gate:
    g = Gate()

    # ---------------- P1 零噪声一步映射（逐位）
    cfg0 = zero_noise_cfg()
    x0 = np.array([0.7, -0.4])
    w = _world(cfg0, x0=x0, P0=[0.0, 0.0])
    errs = []
    for _ in range(6):
        w.step(Action("HOLD"))
        errs.append(float(np.max(np.abs(w._x_hat - w._x_true))))
    g.check("P1 零噪声+初值匹配+纯 HOLD：x̂ 与真值逐位相等（max|Δ|=0）",
            errs == [0.0] * 6,
            {"max_abs_diff_per_step": errs,
             "note": "名义预测 x̂'=A x̂+B u 就是本环境的一步精确预测"})

    # P1c 阴性对照：初值偏 0.1 ⇒ 必须非零（否则 P1 恒真）
    wc = _world(cfg0, x0=x0, P0=[0.0, 0.0])
    wc._x_hat = wc._x_hat + 0.1
    wc.step(Action("HOLD"))
    d = float(np.max(np.abs(wc._x_hat - wc._x_true)))
    g.check("P1c 阴性对照：x̂_0 偏 0.1 时该偏差必须非零（证明 P1 有分辨力）",
            d > 1e-3, {"max_abs_diff": d})

    # ---------------- P2 零噪声 + 初值不匹配 ⇒ 偏差精确 = A^k δ
    #   推导（写下来，不猜）：x_{t+1}=A x_t + B u_t，x̂_{t+1}=A x̂_t + B u_t，同一个 u_t
    #   ⇒ e_{t+1} = x̂_{t+1} − x_{t+1} = A (x̂_t − x_t) = A e_t ⇒ **e_k = A^k δ₀**。
    #   （控制量在两边完全一样，所以偏差里**没有** (A−BK) 项 —— 第一版写成 (A−BK)^k 是错的。）
    #   用 A=(1.01,1.03) 的零噪声夹具，让 A^k 真的不同，否则 A=1 下这条退化成常数。
    cfgA = C.CWConfig(plant=C.PlantConfig(A=(1.01, 1.03), B=C.PlantConfig().B,
                                          sigma_w=(0.0, 0.0), sigma_meas=0.0))
    delta0 = np.array([0.25, -0.15])
    w2 = _world(cfgA, x0=np.zeros(2), P0=[0.0, 0.0])
    w2._x_true = np.zeros(2)
    w2._x_hat = delta0.copy()          # 估计偏离真值 δ₀；无观测 ⇒ 偏差按 A^k 演化
    A2 = np.asarray(cfgA.plant.A, dtype=float)
    trace, pred = [], []
    e = delta0.copy()
    for k in range(1, 7):
        w2.step(Action("HOLD"))
        trace.append(float(np.max(np.abs(w2._x_hat - w2._x_true))))
        e = A2 * e
        pred.append(float(np.max(np.abs(e))))
    g.check("P2 零噪声+初值不匹配：偏差逐位等于 A^k δ₀（预测器不是复制真值）",
            all(abs(a - b) <= 1e-14 for a, b in zip(trace, pred)),
            {"actual": trace, "analytic": pred})

    # ---------------- P3 有噪声：与独立 Kalman 复算逐位一致
    cfg = C.CWConfig(plant=C.bounded_plant())
    K = np.asarray([_world(cfg).nominal_gain(i) for i in range(2)])
    w3 = _world(cfg, budget=10_000)
    for k in range(60):
        w3.step(Action("SEND", 0) if k % 3 == 0 else
                (Action("SEND", 1) if k % 7 == 0 else Action("HOLD")))
        if k == 19:                      # 中段抓一份内部快照
            snap = (w3._x_hat.copy(), w3._P_hat.copy(), len(w3.records))
    rep0 = replay_kalman(w3.records, 0, cfg, K, cfg.plant.prior_var)
    rep1 = replay_kalman(w3.records, 1, cfg, K, cfg.plant.prior_var)
    rep_at = replay_kalman(w3.records[:snap[2]], 0, cfg, K, cfg.plant.prior_var)
    dx_mid = abs(rep_at[-1][0] - float(snap[0][0]))
    dP_mid = abs(rep_at[-1][1] - float(snap[1][0]))
    dx_end = max(abs(rep0[-1][0] - float(w3._x_hat[0])), abs(rep1[-1][0] - float(w3._x_hat[1])))
    dP_end = max(abs(rep0[-1][1] - float(w3._P_hat[0])), abs(rep1[-1][1] - float(w3._P_hat[1])))
    g.check("P3 有噪声+混合动作：x̂/P̂ 与独立 Kalman 复算逐位一致（中段+末点）",
            max(dx_mid, dP_mid, dx_end, dP_end) <= 1e-15,
            {"mid_dx": dx_mid, "mid_dP": dP_mid, "end_dx": dx_end, "end_dP": dP_end,
             "n_slots": len(w3.records),
             "note": "复算只吃动作/ACK/y_t/u_t；逐槽全程对账见 P3b"})

    # P3b 逐槽全程对账（重跑一次，逐步取内部值）
    w3b = _world(cfg, budget=10_000)
    step_path = []
    for k in range(40):
        w3b.step(Action("SEND", 0) if k % 3 == 0 else
                 (Action("SEND", 1) if k % 7 == 0 else Action("HOLD")))
        step_path.append((w3b._x_hat.copy(), w3b._P_hat.copy()))
    r0 = replay_kalman(w3b.records, 0, cfg, K, cfg.plant.prior_var)
    r1 = replay_kalman(w3b.records, 1, cfg, K, cfg.plant.prior_var)
    dmax = 0.0
    for i, (xh, Ph) in enumerate(step_path):
        dmax = max(dmax,
                   abs(float(xh[0]) - r0[i][0]), abs(float(Ph[0]) - r0[i][1]),
                   abs(float(xh[1]) - r1[i][0]), abs(float(Ph[1]) - r1[i][1]))

    g.check("P3b 逐槽（40 槽 × 2 用户 × x̂/P̂）全程与独立复算逐位一致",
            dmax <= 1e-15, {"max_abs_diff": dmax, "n_compared": len(step_path) * 4})

    # ---------------- P4 协方差校准（大样本）
    E = {0: [], 1: []}
    PH = {0: [], 1: []}
    for ep in range(int(n_calib)):
        wp = _world(cfg, episode=seed_calib + ep, budget=10_000)
        for k in range(40):
            wp.step(Action("SEND", 0) if k % 2 == 0 else Action("HOLD"))
        for i in (0, 1):
            E[i].append(float(wp._x_hat[i] - wp._x_true[i]))
            PH[i].append(float(wp._P_hat[i]))
    ratios = {}
    for i in (0, 1):
        e = np.asarray(E[i])
        p = np.asarray(PH[i])
        ratios[i] = float(np.mean(e ** 2) / np.mean(p))
    sd = float(np.sqrt(2.0 / n_calib))        # 卡方近似的相对标准差
    lo, hi = 1.0 - 3 * sd, 1.0 + 3 * sd
    g.check("P4 协方差校准：E[e²]/E[P̂] ∈ 1±3σ（σ=√(2/N)）",
            all(lo <= v <= hi for v in ratios.values()),
            {"ratio_user0": ratios[0], "ratio_user1": ratios[1],
             "band": [lo, hi], "n_episodes": n_calib, "rel_sd": sd,
             "note": "user1 从不发送 ⇒ 同时检验'无观测时的方差增长'"})

    # ---------------- P5 夹具不是协议配置
    bad = zero_noise_cfg().v11_validate()
    g.check("P5 零噪声夹具被 v11_validate 拒绝（防止拿它跑面板追 PASS）",
            any("sigma_w" in b for b in bad), {"problems": bad})

    # ---------------- P6 exact 的入口全部被拒
    from .runner import run_episode
    from .policies import PeriodicSend

    try:
        CWWorld(cfg, *make_episode_rngs(cfg.root_seed, "dev", 0), exact_predictor=True)
        rej_world = False
    except ValueError:
        rej_world = True
    try:
        run_episode(cfg, PeriodicSend(4), split="dev", episode=100,
                    budget_total=128, rho=0.85, predictor="exact")
        rej_runner = False
    except ValueError:
        rej_runner = True
    g.check("P6 `exact` 在世界构造与 run_episode 两个入口都被拒绝",
            rej_world and rej_runner, {"world_rejected": rej_world, "runner_rejected": rej_runner})

    # ---------------- P7 撤回分支的缺陷解析可算
    K0 = float(K[0])
    xs, Ps = legacy_exact_trace(cfg0, x0=float(x0[0]), K=K0, steps=3)
    # 零噪声、A=1、B=0.32、u_t=-K x̂_t：第 1 步两者相同，第 2 步偏差恰为 -B·K·x̂_0
    analytic_gap = float(cfg0.plant.B) * K0 * float(x0[0])
    measured_gap = abs(xs[1] - ((1 - float(cfg0.plant.B) * K0)
                                * ((1 - float(cfg0.plant.B) * K0) * float(x0[0]))))
    g.check("P7 复刻旧 `exact`：第 2 步偏差 == B·K·x₀（解析值，不是录下来的数）",
            abs(measured_gap - analytic_gap) <= 1e-12,
            {"analytic_gap": analytic_gap, "measured_gap": measured_gap,
             "legacy_x": xs, "legacy_P": Ps,
             "note": "旧分支多做一次 B·u_{t-1} 映射 ⇒ 误差量级就是一步控制量"})

    # ---------------- P8 物理不变式
    sw2 = np.asarray(cfg.plant.sigma_w, dtype=float) ** 2
    viol = []
    w8 = _world(cfg, budget=10_000)
    for k in range(80):
        w8.step(Action("SEND", k % 2) if k % 3 else Action("HOLD"))
        for i in (0, 1):
            if float(w8._P_hat[i]) < float(sw2[i]) - 1e-15:
                viol.append((k, i, float(w8._P_hat[i]), float(sw2[i])))
    g.check("P8 物理不变式：每一步 P̂ ≥ σ_w²（A=1 时估计方差不会低于过程噪声）",
            not viol, {"violations": viol[:5], "n_checked": 80 * 2,
                       "sigma_w2": [float(v) for v in sw2]})

    return g
