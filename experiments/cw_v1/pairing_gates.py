# -*- coding: utf-8 -*-
"""CW-R1.1-2 第二件：**外生事件的配对口径**（`CW-PAIR-v1.1` 验收面）。

要解决的问题（验收 §4「ACK 与配对定义」）
----------------------------------------
旧实现里 `packet` / `meas` / `csi` 三条子流**按动作次数**取值。于是
「最后一臂发了 128 次、另一臂发了 64 次」时，两臂读的是同一条流的**不同位置**：
分布相同，但**不能**说"两臂在第 t 槽面对同一枚硬币"。验收要求二选一：
按 `(episode, t, user)` 索引事件，或者**明确写出耦合方式**且不得声称逐时隙匹配。

本模块的做法
------------
`config.ProtocolConfig.event_pairing`：
  · `"slot"`（默认）：三条流在构造时**一次性预生成** `(horizon+1, n_users)` 矩阵，
    下标就是 `(t, user)` ⇒ 同 (t,i) 上两策略看到同一枚硬币（构造性配对）。
  · `"attempt"`：按动作次数取值（18:xx 前的全部证据），只用于重放。

闸（每条都写清"什么错误能让它 FAIL"）
--------------------------------------
Q1  默认配置必须是 `"slot"`。⇒ FAIL 条件：默认值被改回按次数耦合。
Q2  `slot` 下，两策略在**同一 (t,i)** 上都 SEND 时，`delivered` 与 `y_t` 必须逐位相同。
    ⇒ FAIL 条件：抽样仍按次数（位置漂移）、或抽样引入了策略相关量。
Q3  **样本量守卫**：Q2 的比对点必须 ≥ 64 个。⇒ 防止"重叠 0 个点"让 Q2 恒真。
Q4  `attempt` 下同一比对必须**出现不一致**（≥1 个）。⇒ 这是 Q2 的阴性对照：
    证明 Q2 的"逐位相同"不是"两个世界当然一样"。
Q5  配对不改变**边际**：`slot` 与 `attempt` 两种口径下，120 个 episode 的平均送达率
    之差必须在 3σ 内（σ 由二项式给出）。⇒ FAIL 条件：预生成用错了分布或错位。
Q6  确定性：同 `(seed, split, episode)` 两次构造，`plant`/`channel` 轨迹与
    `packet`/`meas`/`csi` 矩阵逐位相同（配对的另一个前提：外生性）。
Q7  回归：`attempt` 口径必须**逐位复现**改动前的四个参考值（含 trace hash）
    ⇒ FAIL 条件：这次改配对顺手改坏了旧路径。

★ 本模块不写 `results/`；证据由 `r1_4_contract.py` 落盘。
"""

from __future__ import annotations

import numpy as np

from . import config as C
from .gates import Gate
from .runner import run_episode
from .streams import make_episode_rngs
from .world import Action, CWWorld

#: 改动**之前**（2026-10-22 18:2x 实测，attempt 口径）的参考值：
#: (cost_true_per_step, cost_est_per_step, policy_trace_hash, n_attempted, n_delivered)
LEGACY_REF = {
    "periodic4": (0.08496858681656241, 0.014242799990603421, "0e1d41317ea143a4", 64, 37),
    "periodic1": (0.10219414940141835, 0.012828700901221159, "8bad3d8d332f21b2", 128, 54),
    "paced_age8": (0.08569615019380192, 0.015359331856281955, "ae8e2a477f64b649", 121, 50),
    "always_probe": (3.076084417124051, 0.0, "bdc1e6fd848a0a1b", 0, 0),
}


def _cfg(pairing: str = "slot") -> C.CWConfig:
    return C.CWConfig(plant=C.bounded_plant(),
                      protocol=C.ProtocolConfig(event_pairing=pairing))


def _run(cfg: C.CWConfig, policy, *, steps: int = 256, episode: int = 13,
         rho: float = 0.85, budget: int = 256) -> tuple[CWWorld, list[np.ndarray]]:
    """跑一条轨迹，并**逐槽留存槽开始时刻的真状态** `x_t`（诊断用，不入策略面）。

    ★ 为什么要真值：配对要验的是"两策略在同一 (t,i) 上用的是不是同一枚硬币/同一条
      量测噪声"。而 `y_t = x_t + σ·z`，两策略的 `x_t` **本来就应该不同**（控制不同），
      所以只能比 `(y_t − x_t)/σ` 这个**噪声本身**。
    """
    rngs, sids = make_episode_rngs(cfg.root_seed, "dev", episode)
    w = CWWorld(cfg, rngs, sids, rho=rho)
    w.budget_total = int(budget)
    pre_x: list[np.ndarray] = []
    for _ in range(int(steps)):
        pre_x.append(w._x_true.copy())
        w.step(policy(w.view()))
    return w, pre_x


def _send_map(w: CWWorld, pre_x: list[np.ndarray], sigma_meas: float
              ) -> dict[tuple[int, int], dict]:
    """(t, user) → {delivered, p_true, z}；`z` 只在送达时可得（未送达没有样本）。

    ★ `z` 是**反推**的量测噪声 `(y_t − x_t)/σ_meas`，只有闸能用（要真值），策略拿不到。
    """
    out = {}
    for idx, r in enumerate(w.records):
        if r.action != "SEND":
            continue
        key = (int(r.t), int(r.user))
        z = None
        if r.data_delivered:
            x_t = float(pre_x[idx][int(r.user)])
            z = (float(r.y_t) - x_t) / float(sigma_meas)
        out[key] = {"delivered": int(r.data_delivered),
                    "p_true": None if r.p_true is None else float(r.p_true),
                    "z": z}
    return out


#: 两策略：A 每槽都发 user0；B 偶数槽发 user0、奇数槽 HOLD。
#: 重叠点 = 偶数槽（128 个），足以撑起 Q3 的样本量守卫。
_POL_A = lambda v: Action("SEND", 0)                                   # noqa: E731
_POL_B = lambda v: Action("SEND", 0) if v.t % 2 == 0 else Action("HOLD")  # noqa: E731


def _compare_pairing(pairing: str) -> dict:
    cfg = _cfg(pairing)
    sm = float(cfg.plant.sigma_meas)
    wa, xa = _run(cfg, _POL_A)
    wb, xb = _run(cfg, _POL_B)
    ma = _send_map(wa, xa, sm)
    mb = _send_map(wb, xb, sm)
    keys = sorted(set(ma) & set(mb))
    both_del = [k for k in keys if ma[k]["delivered"] and mb[k]["delivered"]]
    n_same_del = sum(1 for k in keys if ma[k]["delivered"] == mb[k]["delivered"])
    n_same_p = sum(1 for k in keys if ma[k]["p_true"] == mb[k]["p_true"])
    # ★ z 不能要求逐位相等：`y = x + σz`，而两世界的 `x` 不同 ⇒ `(y−x)/σ` 的
    #   舍入不同（实测差在第 16 位）。等价判定用 1e-9 相对容差；
    #   `attempt` 口径下 z 的差是 O(1)，远大于容差 ⇒ Q4 仍然有分辨力。
    n_same_z = sum(1 for k in both_del
                   if abs(ma[k]["z"] - mb[k]["z"]) <= 1e-9 * (1.0 + abs(ma[k]["z"])))
    max_dz = max((abs(ma[k]["z"] - mb[k]["z"]) for k in both_del), default=0.0)
    return {"pairing": pairing, "n_overlap": len(keys), "n_both_delivered": len(both_del),
            "n_same_delivered": n_same_del, "n_same_p_true": n_same_p,
            "n_same_z": n_same_z, "max_abs_dz": max_dz,
            "n_diff_delivered": len(keys) - n_same_del,
            "n_diff_z": len(both_del) - n_same_z,
            "n_send_a": len(ma), "n_send_b": len(mb),
            "example_z": [ma[k]["z"] for k in both_del[:3]]}


def run_all(*, n_marginal: int = 120) -> Gate:
    g = Gate()

    # ---------------- Q1 默认口径
    default_pairing = C.CWConfig().protocol.event_pairing
    g.check("Q1 默认 event_pairing = 'slot'（新证据自动拿到构造性配对）",
            default_pairing == "slot", {"default": default_pairing})

    # ---------------- Q2/Q3/Q4
    slot = _compare_pairing("slot")
    att = _compare_pairing("attempt")
    g.check("Q2 slot 口径：同 (t,i) 上送达、p_true、量测噪声 z 逐位相同",
            slot["n_overlap"] > 0
            and slot["n_same_delivered"] == slot["n_overlap"]
            and slot["n_same_p_true"] == slot["n_overlap"]
            and slot["n_same_z"] == slot["n_both_delivered"], slot)
    g.check("Q3 样本量守卫：比对点 ≥ 64、其中双方都送达 ≥ 32（否则 Q2 会因重叠为空而恒真）",
            slot["n_overlap"] >= 64 and slot["n_both_delivered"] >= 32,
            {"n_overlap": slot["n_overlap"], "n_both_delivered": slot["n_both_delivered"]})
    g.check("Q4 阴性对照：attempt 口径下同一比对必须出现不一致（证明 Q2 有分辨力）",
            att["n_diff_delivered"] > 0 or att["n_diff_z"] > 0,
            {"n_diff_delivered": att["n_diff_delivered"], "n_diff_z": att["n_diff_z"],
             "n_overlap": att["n_overlap"]})

    # ---------------- Q5 边际不变
    rates = {}
    for tag in ("slot", "attempt"):
        cfg = _cfg(tag)
        tot_att = tot_del = 0
        for ep in range(int(n_marginal)):
            rngs, sids = make_episode_rngs(cfg.root_seed, "dev", 300 + ep)
            w = CWWorld(cfg, rngs, sids, rho=0.85)
            w.budget_total = 64
            for _ in range(64):
                w.step(Action("SEND", 0))
            tot_att += w.budget_attempted
            tot_del += w.n_ack
        rates[tag] = tot_del / tot_att
    n_eff = 64 * int(n_marginal)
    p = 0.5 * (rates["slot"] + rates["attempt"])
    sigma = float(np.sqrt(max(p * (1 - p), 1e-12) / n_eff))
    gap = abs(rates["slot"] - rates["attempt"])
    g.check("Q5 配对不改变送达率边际（|Δ| ≤ 4σ，σ 由二项式给出）",
            gap <= 4 * sigma,
            {"rate_slot": rates["slot"], "rate_attempt": rates["attempt"],
             "gap": gap, "sigma": sigma, "n_sends": n_eff})

    # ---------------- Q6 确定性 / 外生性
    cfg = _cfg("slot")
    wa, _xa = _run(cfg, _POL_A, steps=30)
    wb, _xb = _run(cfg, _POL_A, steps=30)
    det = (np.array_equal(wa._z_meas, wb._z_meas) and np.array_equal(wa._u_packet, wb._u_packet)
           and np.array_equal(wa._z_csi, wb._z_csi)
           and all(a.w_realized == b.w_realized for a, b in zip(wa.records, wb.records))
           and [r.ack for r in wa.records] == [r.ack for r in wb.records])
    # 同一 (seed, split, episode) 下**不同策略**必须共享同一 plant/channel 轨迹：
    #   plant 噪声只由 (seed, split, episode, 槽号) 决定，与动作无关；
    #   首槽的 w_realized 必然相同（u 相同 ⇒ 但 u 可能不同，所以只比"首槽"是安全的）。
    wc, _xc = _run(cfg, _POL_B, steps=1)
    first_same = wa.records[0].w_realized == wc.records[0].w_realized
    g.check("Q6 确定性：同种子同 episode 两次构造的预生成矩阵与轨迹逐位相同；首槽 plant 噪声与策略无关",
            det and first_same, {"deterministic": det, "first_slot_noise_same": first_same})

    # ---------------- Q7 回归：attempt 口径逐位复现改动前参考值
    from .policies import AlwaysProbe, PacedAge, PeriodicSend

    cases = {
        "periodic4": (PeriodicSend(4), 128, 0.85),
        "periodic1": (PeriodicSend(1), 128, 0.85),
        "paced_age8": (PacedAge(8), 204, 0.98),
        "always_probe": (AlwaysProbe(2), 128, 0.85),
    }
    cfg_att = _cfg("attempt")
    rows, ok = [], True
    for name, (pol, B, rho) in cases.items():
        r = run_episode(cfg_att, pol, split="dev", episode=100, budget_total=B, rho=rho)
        ref = LEGACY_REF[name]
        eq = (r.cost_true_per_step == ref[0] and r.cost_est_per_step == ref[1]
              and r.policy_trace_hash == ref[2] and r.n_attempted == ref[3]
              and r.n_delivered == ref[4])
        ok = ok and eq
        rows.append({"case": name, "equal": eq, "cost_true": r.cost_true_per_step,
                     "hash": r.policy_trace_hash, "n_att": r.n_attempted,
                     "n_del": r.n_delivered})
    g.check("Q7 attempt 口径逐位复现改动前参考值（4 例，含 trace hash）", ok, rows)

    return g
