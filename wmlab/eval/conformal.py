# -*- coding: utf-8 -*-
"""★ X42：**共形（conformal）校准的误差信封** —— 把「自报不确定度」换成一个
**有分布无关覆盖保证**的误差上界。

================================================================= 为什么要有这一个文件
X38–X41 反复测到同一件事：世界模型 σ 头的自报量 U 在闭环调度上**没有边际任务价值**。
但 X41 §12.5 把结论收窄成「**未经校准的**自报不确定度」，并点出下一步 = 换成**校准量**
（C30，arXiv:2607.01537）。

★★ 但必须先把 C30 读准（2026-09-29 核实原文，**纠正了我自己在 X41 里的过度转述**）：
   C30 的正面结果来自 **drift-aware 的 certified clock**（谱项 + 校准信封），
   而它**自己写明**：*"in the short-horizon frozen VN-JEPA regime, empirical
   **conformal horizons match the deployed clock** on validity and budget"* ——
   **朴素共形并没有赢**。⇒ 「换成共形就有价值」是**没有根据**的预期。
   本文件存在的意义是**把这件事测出来**，而不是假定它成立。

================================================================= 量是什么（定义先写死）
给标定集 `{(h_i, u_i, e_i)}`：h = 年龄（步），u = 模型自报的**累积**不确定度
`U = sqrt(Σ_{i≤h} mean_d σ_i²)`（与 `control.py::_predict_next_u` 逐位同定义），
e = **归一化后的实测误差**（`mean_d (pred−true)² / var_g`；d 取全维 or 只取位置维）。

    信封   Ê_α(h,u) = ĝ(h,u) + Ŝ(h) · Q_α
      ĝ(h,·) = 在年龄 h 内、对 u 的**单调（PAVA isotonic）**拟合（分位箱 + 池化）
      Ŝ(h)   = 该年龄残差的稳健尺度（1.4826 × MAD）
      Q_α    = 分数 `(e − ĝ)/Ŝ` 的**有限样本共形分位数**（⌈(n+1)(1−α)⌉ 阶统计量）

⇒ 覆盖保证（**边缘**，不是条件）：`P(e ≤ Ê_α) ≥ 1 − α`，前提是标定样本与待测样本可交换。
   它**不保证**条件覆盖，也**不制造排序能力** —— 这是本文件最重要的一句限定。

================================================================= 三个已知的坑（都写进测试）
K1 ★★ **不许省略 Ŝ(h) 的归一化**。`e` 的散布随 h 强烈增长（本 env 里 h=1 时
   残差 ~0.01、h=32 时 ~0.5）。把全体残差**汇池**取分位数 ⇒ Q 被大 h 主导 ⇒
   小 h 处信封被抬到远高于本地散布 ⇒ 触发规则在**所有年龄都立刻触发**（退化）。
   ⇒ 等价于「必须做成局部自适应（locally adaptive）的共形」。
   测试：异方差数据上，归一化版的**逐年龄**覆盖率明显更接近目标。
K2 ★★ **NaN 必须抛**（R14 / 铁律 9）。`nan <= 阈值` 恒为 False ⇒ 一个含 NaN 的
   信封会让"未超阈"被判成"已超阈"或反之，**静默**给出一个看着合理的错答案。
   ⇒ `fit_envelope` / `conformal_quantile` 一律先 `math.isfinite` 复核，宁可 raise。
K3 ★ **标定集与评测集必须 episode 级不相交**。窗口在 episode 内有强自相关
   （时间序列 ⇒ 可交换性被破坏，见 arXiv:2609.13345 的综述）⇒ 随机切窗口
   会让标定与评测共享同一条轨迹的相邻步 ⇒ 覆盖率被**高估**。

================================================================= 与 C29/C30 的归属（不许抢功）
· C29（arXiv:2609.15801）已经提出「**用独立标定单元上的 bounded-loss 增益 + 同时置信下界**
  来决定要不要执行世界模型的提案」⇒ **命题层不是本文件的**。
· C30（arXiv:2607.01537）已经提出「**calibrated native rollout-drift envelope** 作为
  重采样时钟」⇒ **"校准信封当触发量"这个想法也不是本文件的**。
⇒ 本文件的定位只有一个：**在"等传输预算的闭环通信调度"这个设定下，把 C30/C29 的
  校准思想做成一个可跑的策略族，并报出它与年龄阈值的实测比值。**
"""
from __future__ import annotations

import math

import numpy as np

# ★ Q_α 的有限样本阶：⌈(n+1)(1−α)⌉；越界则退回最大值（并**在调用方记下来**）
MIN_ROWS_PER_AGE = 20


def _pava(y: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """Pool-Adjacent-Violators：非降 isotonic 回归，O(n)，返回**拟合值**（与 y 同长）。

    ★ 为什么自己写而不用 sklearn.isotonic：本仓库**刻意不装 sklearn**（自检直跑），
      而 PAVA 只有 20 行、且行为完全可测（见 tests）。
    """
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 1:
        raise ValueError("_pava 只接受 1-D 输入")
    if y.size == 0:
        return y.copy()
    if not np.all(np.isfinite(y)):
        raise FloatingPointError("★ K2：_pava 收到非有限输入（R14：宁可 raise）")
    w = np.ones_like(y) if w is None else np.asarray(w, dtype=np.float64)
    if w.shape != y.shape or np.any(w <= 0):
        raise ValueError("_pava 权重形状不符或非正")
    val = list(y)                 # 每个块的均值
    wt = list(w)                  # 每个块的权重和
    cnt = [1] * int(y.size)       # ★ 必须是**整数**（浮点 count 会让切片索引直接 TypeError）
    i = 0
    while i < len(val) - 1:
        if val[i] > val[i + 1]:   # 违反单调 ⇒ 合并
            nv = (val[i] * wt[i] + val[i + 1] * wt[i + 1]) / (wt[i] + wt[i + 1])
            nw = wt[i] + wt[i + 1]
            nc = cnt[i] + cnt[i + 1]
            val[i:i + 2] = [nv]
            wt[i:i + 2] = [nw]
            cnt[i:i + 2] = [nc]
            if i > 0:
                i -= 1
        else:
            i += 1
    out = np.empty_like(y)
    k = 0
    for v, c in zip(val, cnt):
        out[k:k + c] = v
        k += c
    return out


def conformal_quantile(scores: np.ndarray, alpha: float) -> tuple[float, bool]:
    """有限样本共形分位数。返回 (q, clipped)。

    阶 = ⌈(n+1)(1−α)⌉（Vovk 等）；`clipped=True` 表示该阶超出样本量 ⇒ 取最大值，
    此时覆盖率**只能保证 1**（退化成"总是覆盖"）⇒ 调用方必须如实记录。
    """
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    if s.size == 0:
        raise ValueError("★ K2：conformal_quantile 收到空数组")
    if not np.all(np.isfinite(s)):
        raise FloatingPointError(
            f"★ K2（R14）：共形分数含非有限值（nan={int(np.isnan(s).sum())}, "
            f"inf={int(np.isinf(s).sum())}）⇒ 静默通过会给出一个看着合理的错答案")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha 必须在 (0,1)，收到 {alpha}")
    s = np.sort(s)
    k = int(math.ceil((s.size + 1) * (1.0 - alpha)))
    clipped = k > s.size
    if clipped:
        k = s.size
    return float(s[k - 1]), bool(clipped)


def _age_curve(knots_u: np.ndarray, knots_v: np.ndarray, u: np.ndarray) -> np.ndarray:
    """在 (knots_u, knots_v) 上做**线性插值 + 端点夹紧**（np.interp 的自然行为）。"""
    if knots_u.size == 1:
        return np.full_like(u, float(knots_v[0]), dtype=np.float64)
    return np.interp(u, knots_u, knots_v, left=float(knots_v[0]),
                     right=float(knots_v[-1]))


def fit_envelope(err: np.ndarray, U: np.ndarray, alpha: float = 0.1,
                 n_knots: int = 8, min_rows_per_age: int = MIN_ROWS_PER_AGE) -> dict:
    """拟合共形误差信封。`err`/`U` 形状均为 (N, H)。

    返回 dict（全部为 numpy 数组或标量）：
        knots_u (H, B) 各年龄的 u 节点（严格递增，B ≤ n_knots）
        knots_v (H, B) 对应的 ĝ 值（对 u 单调不减，由 PAVA 保证）
        s_h     (H,)   各年龄残差的稳健尺度 Ŝ(h)
        q       float  共形分位数 Q_α
        clipped bool    Q_α 是否被样本量截断（true ⇒ 覆盖率退化，只作参考）
        q_unnorm float ★ K1 对照：**未归一化**版本的 Q（只用于诊断，不进信封）
        n_per_age (H,) 各年龄样本数
    """
    e = np.asarray(err, dtype=np.float64)
    u = np.asarray(U, dtype=np.float64)
    if e.shape != u.shape or e.ndim != 2:
        raise ValueError(f"err/U 形状必须一致且为 2-D，收到 {e.shape} / {u.shape}")
    if not (np.all(np.isfinite(e)) and np.all(np.isfinite(u))):
        raise FloatingPointError("★ K2（R14）：err/U 含非有限值 ⇒ 拒绝拟合")
    N, H = e.shape
    if N < min_rows_per_age * 2:
        raise ValueError(f"标定样本太少（N={N}）")

    knots_u = np.zeros((H, n_knots), dtype=np.float64)
    knots_v = np.zeros((H, n_knots), dtype=np.float64)
    n_keep = np.zeros(H, dtype=int)
    g_hat = np.empty((N, H), dtype=np.float64)
    n_per_age = np.full(H, N, dtype=int)

    for h in range(H):
        x, y = u[:, h], e[:, h]
        qs = np.linspace(0.0, 1.0, n_knots + 1)
        edges = np.quantile(x, qs)
        edges = np.unique(edges)                 # 结（tie）会让空箱出现 ⇒ 去重
        if edges.size < 3:                       # U 几乎没变化 ⇒ 退化成"只有 h 依赖"
            edges = np.array([x.min() - 1e-9, x.max() + 1e-9])
        idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, edges.size - 2)
        nb = edges.size - 1
        bu, bv, bw = [], [], []
        for b in range(nb):
            m = idx == b
            if int(m.sum()) < 3:                 # 样本太少 ⇒ 不设节点（并入邻箱）
                continue
            bu.append(float(x[m].mean()))
            bv.append(float(y[m].mean()))
            bw.append(float(m.sum()))
        if len(bu) < 2:
            bu = [float(x.min()), float(x.max())]
            bv = [float(y.mean()), float(y.mean())]
            bw = [float(N), float(N)]
        order = np.argsort(np.asarray(bu))
        bu_a = np.asarray(bu)[order]
        bv_a = _pava(np.asarray(bv)[order], np.asarray(bw)[order])
        # 节点 u 必须严格递增，否则 np.interp 行为未定义
        bu_a, uniq = np.unique(bu_a, return_index=True)
        bv_a = bv_a[uniq]                        # PAVA 之后取并集处的最左值即可（单调）
        if bu_a.size < 2:
            bu_a = np.array([float(x.min()), float(x.max())])
            bv_a = np.array([float(y.mean()), float(y.mean())])
        knots_u[h, :bu_a.size] = bu_a
        knots_v[h, :bu_a.size] = bv_a
        n_keep[h] = bu_a.size
        g_hat[:, h] = _age_curve(bu_a, bv_a, x)

    res = e - g_hat
    s_h = 1.4826 * np.median(np.abs(res - np.median(res, axis=0, keepdims=True)), axis=0)
    s_h = np.maximum(s_h, 1e-12)
    q, clipped = conformal_quantile(res / s_h[None, :], alpha)
    # ★ K1 对照：不归一化的 Q（诊断用；若两者差距巨大，说明异方差很重）
    q_raw, clipped_raw = conformal_quantile(res, alpha)
    return {"knots_u": knots_u, "knots_v": knots_v, "n_keep": n_keep,
            "s_h": s_h, "q": q, "clipped": clipped,
            "q_unnorm": q_raw, "clipped_unnorm": clipped_raw,
            "n_per_age": n_per_age, "alpha": float(alpha),
            "n_fit": int(N), "H": int(H)}


def envelope(tab: dict, h_step: np.ndarray, u: np.ndarray) -> np.ndarray:
    """在线查表：`h_step` 是 **1-based 年龄**（= 闭环里的 `state["age"]`），可广播。

    返回 Ê_α(h, u)。越界年龄夹到 [1, H]。
    """
    h = np.asarray(h_step, dtype=np.int64)
    u = np.asarray(u, dtype=np.float64)
    h = np.clip(h, 1, tab["H"]) - 1
    H = tab["H"]
    if h.size == 0:
        return np.zeros_like(u, dtype=np.float64)
    if h.ndim == 0:
        return float(_age_curve(tab["knots_u"][int(h), :tab["n_keep"][int(h)]],
                                tab["knots_v"][int(h), :tab["n_keep"][int(h)]],
                                u.reshape(1))[0]
                     + tab["s_h"][int(h)] * tab["q"])
    out = np.empty(h.shape, dtype=np.float64)
    for hi in np.unique(h):
        m = h == hi
        nk = int(tab["n_keep"][hi])
        out[m] = (_age_curve(tab["knots_u"][hi, :nk], tab["knots_v"][hi, :nk], u[m])
                  + tab["s_h"][hi] * tab["q"])
    return out


def check_per_coverage(p4_rows: list, expect_pers) -> dict:
    """★★ X42 汇报的**覆盖守卫**：期望的每个 PER 档都必须有 `mean_dist_tail` 的配对行。

    ★ 为什么这是一等的检查而不是小工具：X42 的网格（6 PER × 41 工作点）在本机
      **单次跑不完**（后台任务 10 min 硬上限）⇒ 必须**按 PER 分块**跑 ⇒
      一旦某块没跑成，跨 PER 取中位时**分母会悄悄变小**，
      结论随之变强或变弱，而输出**看着完全正常**。
      ⇒ 缺档一律 raise（R14：报错优于给假数字），**不许**在部分档上聚合。
    """
    want = [round(float(p), 6) for p in expect_pers]
    got = sorted({round(float(r["per"]), 6) for r in p4_rows
                  if r.get("metric") == "mean_dist_tail"})
    missing = [p for p in want if p not in got]
    if missing:
        raise AssertionError(
            f"★ 覆盖守卫未通过：期望的 PER 档 {want} 里缺 {missing}（实得 {got}）"
            " ⇒ 不许在**部分档**上聚合（分母变小会让结论悄悄变强）")
    return {"n_per_evaluated": len(got), "pers": got, "missing": []}


def coverage(err: np.ndarray, U: np.ndarray, tab: dict) -> dict:
    """在**留出集**上复核覆盖性（conformal 的"拿已知答案验量级"，R12）。

    ⚠️ 报的是**逐年龄覆盖率的中位/最小**，因为覆盖保证只对**边缘**成立：
       逐年龄都接近目标 ⇒ 该标定在本设定下可迁移；某些年龄严重低 ⇒ 不可迁移，
    ⇒ 此时**不许**用该信封下任何调度结论。
    """
    e = np.asarray(err, dtype=np.float64)
    u = np.asarray(U, dtype=np.float64)
    if not (np.all(np.isfinite(e)) and np.all(np.isfinite(u))):
        raise FloatingPointError("★ K2（R14）：覆盖性复核收到非有限值")
    H = tab["H"]
    if e.shape[1] != H:
        raise ValueError(f"留出集视界 {e.shape[1]} ≠ 标定视界 {H}")
    hs = np.tile(np.arange(1, H + 1)[None, :], (e.shape[0], 1))
    cov = e <= envelope(tab, hs, u)
    per_age = cov.mean(axis=0)
    return {"marginal": float(cov.mean()),
            "per_age_median": float(np.median(per_age)),
            "per_age_min": float(per_age.min()),
            "per_age_max": float(per_age.max()),
            "per_age": per_age.tolist(),
            "target": 1.0 - float(tab["alpha"]),
            # 二项容差：3σ_binom，量化"有限样本波动"
            "tol_3sigma": float(3.0 * math.sqrt(
                tab["alpha"] * (1.0 - tab["alpha"]) / max(cov.size, 1)))}
