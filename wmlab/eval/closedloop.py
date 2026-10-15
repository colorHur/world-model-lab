# -*- coding: utf-8 -*-
"""★ 交付阶段（2026-10-15）：六族调度**统一闭环**的共用构件。

================================================================= 为什么要有这一个模块
`wmlab/eval/scheduling.py` 把「六族调度器怎么构建」收成了一张表，但**"怎么跑闭环"**
这件事一直散在 `scripts/24..29`：每轮都要重写一遍「标定集收集 + 状态写入 + 结果落盘」，
而且 **信号相关三族（selfreport / conformal / oracle）历史上从未真正进过同一闭环** ——
X38–X43 是用**离线重演**（收集好的 U/误差池）评估的。

本模块只放**共用构件**（对应"三件硬约定"里 C1 的**接线**部分）：

  (a) **标定集收集** —— `sample_windows` / `rollout_uncertainty_and_error` / `err_from_sq`
      ★ 单实现纪律：标定集的采样口径直接决定 conformal/oracle 的阈值 ⇒ 不能再分两份各自演化。
  (b) **状态接线验收** —— `WiringProbe` / `check_wiring`
      ★★ 华腾 2026-10-15 的要点：`state.reads`（在 `scheduling.py` 里）只证明**读过字段**，
         **不证明"值是当前步、正确时点的"** —— 字段完全可能一直是旧值。
         本模块把"调度器**实际读到**的 (age, U, 真值误差)"逐步记下来，再做**可证伪**的核对
         （见 `check_wiring` 的 W1–W4，每一条都能被一个"常量 state"打死）。
  (c) **标定拟合** —— `calibrate`
      用**独立标定集**（episode 级 fit/留出切分，K3）一次性得到 conformal 信封与 oracle 的 z 表。

★ 本模块**不**提供调度器构建（那是 `scheduling.py` 的职责），也**不**做评测口径
  （那是 `pairing.py` / `tracking.py` 的职责）—— 只补"把信号相关族真正接进闭环"这一环。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from . import scheduling

# ==================================================================== (a) 标定集收集
def sample_windows(episodes: Sequence[dict], horizon: int, n_samples: int,
                   seed: int, t0_min: int = 0):
    """抽 `(obs_t, actions[t:t+H], obs[t+1:t+H+1])` 窗口。返回 numpy 三元组。

    ★ 由 `scripts/24` 的私有版本上移而来（单实现纪律）。`t0_min` 通常传 `warmup`
      —— 只在这条 episode "熬过预热" 之后抽窗（否则会拿瞬态去标定）。
    """
    rng = np.random.default_rng(int(seed))
    H = int(horizon)
    starts = []
    for ei, ep in enumerate(episodes):
        L = int(ep["length"])
        hi = L - H - 1
        if hi <= int(t0_min):
            continue
        starts.append((ei, hi))
    if not starts:
        raise AssertionError(
            f"★ 没有可用的 rollout 窗口（episode 太短：要求 长度 − H − 1 > t0_min={t0_min}）")
    obs0, acts, tgts = [], [], []
    for _ in range(int(n_samples)):
        ei, hi = starts[int(rng.integers(len(starts)))]
        t0 = int(rng.integers(int(t0_min), hi))
        ep = episodes[ei]
        obs0.append(ep["obs"][t0])
        acts.append(ep["act"][t0:t0 + H])
        tgts.append(ep["obs"][t0 + 1:t0 + H + 1])
    return (np.asarray(obs0, dtype=np.float32),
            np.asarray(acts, dtype=np.float32),
            np.asarray(tgts, dtype=np.float32))


@torch.no_grad()
def rollout_uncertainty_and_error(model, obs0, acts, tgts, device):
    """一次 rollout 同时给出 `U` 与**逐维平方误差**（省一次前向）。

    Returns:
        U   (N, H)    —— 累积不确定度 `sqrt(Σ_{i≤h} mean_d σ_i²)`。
                          ★★ **与 `control.py::_predict_next_u` 逐位同定义**（R12：闭环写入的
                          `state["U"]` 与这里标定的 U 必须是**同一个量**，否则 (h,U) 平面上的
                          共形信封在闭环里查的就不是它拟合时的那张面）。
        sq  (N, H, D) —— 逐维平方误差 `(decode(μ_h) − true_h)²`（**未归一化**）；
                         调用方按 `dims` 取均值再除 `var_g` ⇒ 与 `scripts/24` 的 `e_parts` 同口径。

    ★ 用**均值**传播（`next_latent` 返回 μ），σ 只作伴随量、不参与状态演化
      ⇒ 与 X30/X35/X44 的 rollout 口径严格一致。
    """
    o = torch.as_tensor(np.asarray(obs0), dtype=torch.float32, device=device)
    a = torch.as_tensor(np.asarray(acts), dtype=torch.float32, device=device)
    t = torch.as_tensor(np.asarray(tgts), dtype=torch.float32, device=device)
    z = model.encode(o)
    H = int(a.shape[1])
    U2 = torch.zeros(o.shape[0], dtype=torch.float32, device=device)
    Us, Sqs = [], []
    for h in range(H):
        mu, sd, _ = model.next_latent_dist(z, a[:, h])
        U2 = U2 + sd.pow(2).mean(dim=-1)
        z = mu
        Sqs.append((model.decode(z) - t[:, h]) ** 2)
        Us.append(torch.sqrt(U2))
    U = torch.stack(Us, dim=1).detach().cpu().numpy().astype(np.float64)
    sq = torch.stack(Sqs, dim=1).detach().cpu().numpy().astype(np.float64)
    return U, sq


def err_from_sq(sq: np.ndarray, dims, var_g: float) -> np.ndarray:
    """`(N,H,D)` 逐维平方误差 → `(N,H)` **归一化**误差（NMSE 量纲，同 `scripts/24` 的 `e_parts`）。

    `dims` 传 `slice(0, 2)`（位置维）或 `slice(None)`（全维）—— 两者**不可混用**，
    混用就是量纲错配（X43 记录过这条错误犯了两次）。
    """
    s = np.asarray(sq, dtype=np.float64)
    return s[:, :, dims].mean(axis=2) / max(float(var_g), 1e-18)


# ==================================================================== (b) 状态接线验收
class WiringProbe:
    """包一个 `Schedule`，**逐步记录调度器读到的 state**（不改变它的决策）。

    ★ 为什么需要它（华腾 2026-10-15）：`state.reads` 只证明"读过字段"，
      **不证明"值是当前步、正确时点的"** —— 字段可能一直是旧值。
      本探针把"调度器实际读到的 (age, U, 真值误差) + 本步是否送达"记下来，
      交给 `check_wiring` 做四条**可证伪**的核对。

    ★ 探针**只读不写** state（用 `in` / `[]`，不走 `require`，故不污染 `state.reads`
      的计数；`state.reads` 记的是**调度器自己**的读取）。`reset` 会转发给内层调度器
      （`control.run_closed_loop_control` 按 `hasattr(schedule, 'reset')` 判断），
      并记一个 `boundary` ⇒ 日志可分段到 episode。
    """

    def __init__(self, sched, state, truth_key: str | None = None) -> None:
        self.sched = sched
        self.state = state
        self.truth_key = truth_key
        self.log: list[dict] = []

    def reset(self, rng=None) -> None:                      # noqa: D401
        self.log.append({"boundary": True})
        fn = getattr(self.sched, "reset", None)
        if callable(fn):
            fn(rng)

    def __call__(self, t: int, rng: np.random.Generator) -> bool:
        row: dict = {"t": int(t)}
        row["age"] = float(self.state["age"]) if "age" in self.state else None
        row["U"] = float(self.state["U"]) if "U" in self.state else None
        if self.truth_key is not None:
            row["truth"] = (float(self.state[self.truth_key])
                            if self.truth_key in self.state else None)
        row["delivered"] = bool(self.sched(t, rng))
        self.log.append(row)
        return row["delivered"]

    @property
    def __name__(self) -> str:
        return f"probe[{getattr(self.sched, '__name__', 'sched')}]"


def _segments(log: list[dict]) -> list[list[dict]]:
    """按 `boundary` 把探针日志切成**逐 episode** 的段。"""
    segs: list[list[dict]] = []
    cur: list[dict] = []
    for row in log:
        if row.get("boundary"):
            if cur:
                segs.append(cur)
            cur = []
        else:
            cur.append(row)
    if cur:
        segs.append(cur)
    return segs


def check_wiring(probe_log: list[dict], *, family: str,
                 truth_key: str | None = None) -> dict:
    """对探针日志做四条**可证伪**的接线核对（任一破 ⇒ raise；R14）。

    W1 **AoI 递推**（核心）：同一 episode 内，第 i+1 次查询读到的 `age` 必须等于
        `0 if 第 i 次送达 else (第 i 次读到的 age) + 1`。
        ⇒ 这就证明"调度器读到的是**当前步、正确时点**的年龄"，而不是某个旧值
        （闭环的记账递推与调度器看到的逐位一致）。
    W2 **送达即归零**：第 i 次送达 ⇒ 第 i+1 次读到的 `U == 0`（收到新观测 ⇒ 累积量清零）。
    W3 **oracle 送达即归零**（仅 `truth_key` 非 None）：送达 ⇒ 下一步真值误差 == 0
        （`est ← true` ⇒ `d = 0`）。
    W4 **首查询初值**：每个 episode 的**第一次**查询必须读到 `age == 0`
        （`control.py` 在被测段入口显式写 0；见该处注释 —— 缺这一步 conformal/oracle
         在 t=1 会直接抛 `MissingStateError`）。

    ★★ 反向断言（自检 [69]）：喂一个"state 被写成常量"的假探针日志，W1 **必须失败**
      —— 否则本核对根本没有分辨力（这正是第 33 次自证伪的教训）。

    Returns:
        dict：`n_segments` / `n_queries` / `n_w1` / `n_w2` / `n_w3` / `n_w4`（核对的次数）
        以及 `violations`（应为空）。`violations` 非空 ⇒ 直接 raise。
    """
    segs = _segments(probe_log)
    viol: list[str] = []
    n_w1 = n_w2 = n_w3 = n_w4 = 0
    u_unequal = 0
    for si, seg in enumerate(segs):
        if not seg:
            continue
        # W4：首查询的初值
        if seg[0].get("age") is not None:
            n_w4 += 1
            if float(seg[0]["age"]) != 0.0:
                viol.append(f"[W4] seg{si} 首查询 age={seg[0]['age']} ≠ 0（闭环没写初值？）")
        for i in range(len(seg) - 1):
            a, b = seg[i], seg[i + 1]
            # W1：AoI 递推
            if a.get("age") is not None and b.get("age") is not None:
                n_w1 += 1
                expect = 0.0 if a["delivered"] else float(a["age"]) + 1.0
                if float(b["age"]) != expect:
                    viol.append(
                        f"[W1] seg{si} step{a['t']}→{b['t']}: 读到 age={b['age']}，"
                        f"按递推应为 {expect}（delivered={a['delivered']}, prev_age={a['age']}）")
            # W2：送达 ⇒ 下一步 U = 0
            if a["delivered"] and b.get("U") is not None:
                n_w2 += 1
                if float(b["U"]) != 0.0:
                    viol.append(f"[W2] seg{si} step{a['t']} 送达，但 {b['t']} 读到 U={b['U']} ≠ 0")
            # W3：oracle 送达 ⇒ 下一步真值误差 = 0
            if (truth_key is not None and a["delivered"] and b.get("truth") is not None):
                n_w3 += 1
                if float(b["truth"]) != 0.0:
                    viol.append(
                        f"[W3] seg{si} step{a['t']} 送达，但 {b['t']} 读到 {truth_key}={b['truth']} ≠ 0")
        # U 的"活性"（非恒定）：用该段的极差看
        us = [r["U"] for r in seg if r.get("U") is not None]
        if len(us) >= 3 and max(us) - min(us) == 0.0:
            u_unequal += 1
    out = {"family": family, "n_segments": len(segs),
           "n_queries": sum(len(s) for s in segs),
           "n_w1": n_w1, "n_w2": n_w2, "n_w3": n_w3, "n_w4": n_w4,
           "n_segments_constant_U": u_unequal, "violations": viol}
    if viol:
        raise AssertionError(
            f"★ 状态接线核对未通过（family={family}，{len(viol)} 处）：\n  "
            + "\n  ".join(viol[:8])
            + ("\n  …（其余省略）" if len(viol) > 8 else ""))
    return out


# ==================================================================== (c) 标定拟合
def calibrate(model, calib_episodes: Sequence[dict], *, horizon: int = 32,
              n_samples: int = 768, var_g: float = 1.0, device=None,
              conf_dims=slice(0, 2), or_dims=slice(0, 2), alpha: float = 0.1,
              n_knots: int = 8, seed: int = 0, t0_min: int = 0,
              min_usable: int = 6, s_floor: float = 1e-6) -> dict:
    """用**独立标定集**拟合 conformal 信封与 oracle 的逐年龄 z 表。

    ★★ K3（`conformal.py` 文件头）：标定集与评测集必须 **episode 级不相交**，
      且 fit / 留出**按 episode 切**（窗口在 episode 内强自相关 ⇒ 随机切窗口会高估覆盖率）。
      本函数在**标定集内部**再做一次 episode 级 fit/留出切分，用于复核覆盖性。

    Returns:
        dict：`conf_tab` / `conf_cov` / `or_tab` / `u_fit_flat` / `z_fit_flat` /
        `n_usable` / `n_fit` / `n_test` / `usable_len_median` / `note`
    """
    from .conformal import coverage as conf_coverage, envelope as conf_envelope_eval, fit_envelope
    from .oracle import fit_age_cond, z_pool as oracle_z_pool

    usable = [e for e in calib_episodes if int(e["length"]) - int(horizon) - 1 > int(t0_min)]
    if len(usable) < max(2, int(min_usable)):
        raise AssertionError(
            f"★ 可用标定 episode 只有 {len(usable)} 条（< {min_usable}）⇒ "
            "episode 级 fit/留出切分做不了；**不许**降级成全量标定（那会失去覆盖性检验）")
    n_half = len(usable) // 2
    fit_eps, test_eps = usable[:n_half], usable[n_half:]
    o0f, acf, tgf = sample_windows(fit_eps, horizon, int(n_samples), seed + 77, t0_min)
    o0t, act, tgt = sample_windows(test_eps, horizon, max(int(n_samples) // 2, 128),
                                   seed + 77, t0_min)
    U_f, sq_f = rollout_uncertainty_and_error(model, o0f, acf, tgf, device)
    U_t, sq_t = rollout_uncertainty_and_error(model, o0t, act, tgt, device)
    e_f = err_from_sq(sq_f, conf_dims, var_g)
    e_t = err_from_sq(sq_t, conf_dims, var_g)
    conf_tab = fit_envelope(e_f, U_f, alpha=float(alpha), n_knots=int(n_knots))
    conf_cov = conf_coverage(e_t, U_t, conf_tab)

    or_f = err_from_sq(sq_f, or_dims, var_g)
    or_tab = fit_age_cond(or_f, s_floor=float(s_floor))
    z_fit = oracle_z_pool(or_f, or_tab).reshape(-1)

    # ★★ 信封在**标定集**上的取值 `Ê_α(h,u)`（形状 (N_fit, H)）—— 供调用方挑 tol。
    #   为什么必须交出来（2026-10-15 冒烟踩到）：`conformal` 族只在 `age < K` 时才看信封
    #   （否则无条件发）。若 tol 拍成 0.05/0.2/0.4 这类"看着合理"的数，而 h<K 处
    #   `Ê_α` 的量级是 ~0.01 ⇒ 三档 tol 全部 `< Ê_α` ⇒ **旋钮扫过了但触发率完全不动**
    #   （冒烟实测三档都是 0.2875，恰好 == threshold(K=2)）⇒ 一个**假接线**。
    #   交 `Ê_α` 出来 = 让调用方按**分位点**选 tol，扫的一定是"会咬住"的区间。
    _hs = np.tile(np.arange(1, int(conf_tab["H"]) + 1)[None, :], (U_f.shape[0], 1))
    e_hat_fit = conf_envelope_eval(conf_tab, _hs, U_f)

    return {
        "conf_tab": conf_tab, "conf_cov": conf_cov, "or_tab": or_tab,
        "u_fit_flat": np.asarray(U_f).reshape(-1),
        "z_fit_flat": np.asarray(z_fit),
        # ★ 拟合侧的**误差矩阵 / 信封值**（供调用方挑"会真正起作用的"容差/阈值刻度，
        #   而不是拍一组数 —— 拍了就会出现"旋钮扫过了但触发率完全不动"的假接线）。
        "err_fit_conf": np.asarray(e_f), "err_fit_or": np.asarray(or_f),
        "e_hat_fit": np.asarray(e_hat_fit),
        "n_usable": int(len(usable)), "n_fit": int(len(fit_eps)),
        "n_test": int(len(test_eps)),
        "usable_len_median": float(np.median([int(e["length"]) for e in usable])),
        "note": ("K3：fit/留出按 **episode** 切；conformal 只给**边缘**覆盖保证，"
                 "不制造排序能力。"),
    }
