# -*- coding: utf-8 -*-
"""★ X43：oracle（特权）触发量的**逐年龄归一化**构件。

为什么单独一个模块（而不是塞进 `scripts/24`）
------------------------------------------------
「先把 age 条件掉」这条教训，本课题已经踩过**两次**：
  · X39 的 P4' 作废：闭环 `ρ(U,age)=0.9941` ⇒ 全局阈值 `U≥thr` **数学上就是**年龄阈值；
  · X40 的第一版：残差用**全局**分位点 ⇒ 尺度随 age 变 ⇒ 小 h 永不触发、大 h 总触发
    ⇒ 实际退化成"只在老年龄才发"（冒烟直接给 4.77× 的**假**结果）。
X43 把同一件事**第三次**做，对象换成**真值误差** ⇒ 归一化必须是**可被单元测试锁住**
的构件，不能再靠"看着对"（自检 ㊻）。所以它在这里，不在脚本里。

★ 概念边界（决定 X43 能说什么、不能说什么）
------------------------------------------------
- oracle 用的是**真值**（仿真器内部有 ground truth）⇒ **不可部署**，只是一个**可达上界**。
- 它上界的对象是「**误差类触发量**」这一族：任何"从当前估计误差出发"的调度器，
  信息量都不超过"直接知道当前误差"。⇒ **若 oracle 也赢不了年龄阈值，则瓶颈不在
  「触发量的信息量」**（要么在任务侧，要么在"误差"本身不是好目标）。
- ⚠️ 它**不**上界「用别的可观测量做**预测性**触发」（例如知道真值阵风场 ⇒ 提前知道
  哪一段误差会长得快）。那一类需要**另一个** oracle，见 `README.md` 的引用限定。

★ 三个必守（各有一条自检盯着）
  O1 逐年龄归一化不可省（自检 ㊻）—— 否则 z 变成"年龄的单调函数"，**实验静默空跑**；
  O2 非有限输入必 raise（自检 ㊼，R14）—— `nan <= 阈值` 恒为 False ⇒ 静默通过会产出
     **看着合理**的错答案；
  O3 `Ŝ(h)` 必须有地板 —— 小年龄处 MAD≈0 会把 z 抬成 1e8 量级的假数（与 X35 报过的
     `7.2e8σ` 是同一个坑）。

★ 量纲（R12：参数接线了没）
  `err` 的两个来源必须**逐位同量纲**：离线的 `e = mean(平方误差)/var_g`（`scripts/24`
  的 `e_parts`）与在线的 `control.py` 写入的 `oracle_err` / `oracle_pos`。
  任何一边改成 RMSE 或未归一化的欧氏距离，z 就会被量纲错配顶飞，而症状看起来像
  "信号无效"。
"""

from __future__ import annotations

import numpy as np

#: 中位绝对偏差 → 稳健标准差的换算系数（正态下 1/MAD/σ = 1.4826）
MAD_TO_STD: float = 1.4826


def fit_age_cond(err: np.ndarray, s_floor: float = 1e-6) -> dict:
    """按年龄拟合「条件中位数 m(h)」与「条件稳健尺度 Ŝ(h)」。

    Args:
        err: (N, H) —— **与在线口径逐位一致**的误差矩阵（NMSE 量纲，见模块头）。
        s_floor: Ŝ 的地板（见 O3）。

    Returns:
        dict，含 `m` / `s`（形状 (H,)）、`H`、`n`，以及两个**必须如实报**的诊断：
        `n_s_floored`（有多少个年龄的 Ŝ 落在地板上）。
    """
    e = np.asarray(err, dtype=np.float64)
    if e.ndim != 2:
        raise ValueError(f"★ err 必须是 (N, H) 二维，收到 {e.shape}")
    if e.shape[1] < 1 or e.shape[0] < 2:
        raise ValueError(f"★ err 至少要 2 个样本 × 1 个年龄，收到 {e.shape}")
    # ---- O2：NaN/Inf 一律 raise（不许 nanmedian 静默吞掉）----
    if not np.all(np.isfinite(e)):
        raise FloatingPointError(
            f"★ O2（R14）：oracle 标定集的误差含非有限值 "
            f"(nan={int(np.isnan(e).sum())}, inf={int(np.isinf(e).sum())}) —— "
            "不许静默丢弃：`nan <= 阈值` 恒为 False，会产出**看着合理**的错答案")
    m = np.median(e, axis=0)
    mad = np.median(np.abs(e - m[None, :]), axis=0)
    s = MAD_TO_STD * mad
    if not np.all(np.isfinite(m)) or not np.all(np.isfinite(s)):
        raise FloatingPointError("★ 条件统计量非有限（m 或 s 含 NaN/Inf）")
    n_floored = int((s < s_floor).sum())
    s = np.maximum(s, float(s_floor))
    return {"m": m, "s": s, "H": int(e.shape[1]), "n": int(e.shape[0]),
            "s_floor": float(s_floor), "n_s_floored": n_floored,
            "m_first": float(m[0]), "m_last": float(m[-1]),
            "s_median": float(np.median(s)),
            "note": ("m(h)=条件中位、s(h)=1.4826×条件 MAD；s 有地板 "
                     f"{s_floor:g}（O3：小年龄 MAD≈0 会把 z 抬成 1e8 级假数）")}


def z_oracle(err_scalar: float, h: int, tab: dict) -> float:
    """单点 z 分数：`z = (e − m(h)) / Ŝ(h)`。`h` 是**年龄**（≥1，自动裁剪到 H）。"""
    e = float(err_scalar)
    if not np.isfinite(e):
        raise FloatingPointError(f"★ O2（R14）：oracle 触发量非有限（{e}）")
    hh = int(min(max(int(h), 1), int(tab["H"])))
    return (e - float(tab["m"][hh - 1])) / float(tab["s"][hh - 1])


def z_pool(err: np.ndarray, tab: dict) -> np.ndarray:
    """整池 z 分数（N, H）—— 供**选 τ 的分位点**用（与 `scripts/24` 的 `z_pool` 同义）。"""
    e = np.asarray(err, dtype=np.float64)
    if e.ndim != 2 or e.shape[1] != int(tab["H"]):
        raise ValueError(f"★ err 形状 {e.shape} 与 tab['H']={tab['H']} 不符")
    if not np.all(np.isfinite(e)):
        raise FloatingPointError("★ O2（R14）：z_pool 的输入误差含非有限值")
    return (e - tab["m"][None, :]) / tab["s"][None, :]


#: `_otrace` 的列 = (age, 6 维量, 位置维量)（`control.py` 写入）
ORACLE_TRACE_COL = {"res": 1, "pos": 2}


def align_age_medians(trace, m_table, col: int, h_refs=(1, 2, 4, 8, 16),
                      min_n: int = 20):
    """闭环 trace 的「逐年龄中位 ÷ 离线 m(h)」核对（R12：量纲/分布能不能迁移）。

    Args:
        trace: (N, ≥3) —— 列 = (age, 6 维量, 位置维量)。
        m_table: 离线条件中位 `m(h)`（1-based 索引 ⇒ `m(h) = m_table[h-1]`）。
        col: **取哪一列当真值误差** —— 1 = 6 维（`--oracle-target res`）、
             2 = 位置维（`--oracle-target pos`）。★ 必须与 `m_table` **同量纲**。
        h_refs: 要核对的年龄点；`min_n` 之下的年龄点跳过（样本太少的中位不值钱）。

    Returns:
        None（无可核对点）或 `dict(per_h, col, ratio_median, ratio_min, ratio_max)`。

    ★★ 这条函数存在的**唯一**理由（2026-09-29，同一条量纲错误犯了**两次**）：
      ① `_otrace` 第一版只有一列（6 维量）⇒ pos 臂拿 6 维量去比位置维基准
         `m_pos(h)`（`m_pos(h=1)/m_res(h=1) = 0.088` ⇒ 比值被顶到 ≈10×）；
      ② 补了第三列之后，**列号变量 `col` 算了却忘了接进取数行**（取数仍硬编码
         `trace[:, 1]`）⇒ 症状与 ① **逐位相同**（"改了却没变"）。
      ⇒ 抽成纯函数，由自检 **㊽** 锁死"列真的接线了"：
        喂一个第 1 列与第 2 列相差 10× 的假 trace，`col=2` 必须返回 ≈1.0。
      症状长什么样值得记住：它**看起来像**"标定不可迁移 / 信号无效"，
      而**不是**像报错 —— 这正是最该被 raise/被测试盯住的一类。
    """
    tr = np.asarray(trace, dtype=np.float64)
    if tr.ndim != 2 or tr.shape[1] < 3:
        raise AssertionError(
            f"★ `_otrace` 必须是 (N,3) = (age, 6维量, 位置维量)，收到 {tr.shape} ⇒ "
            "列数不对时会拿 6 维量去比位置维基准（量纲错配，症状像「不可迁移」）")
    if int(col) not in (1, 2):
        raise ValueError(
            f"★ col 只能是 1（6 维）或 2（位置维），收到 {col} ⇒ 取错列 = 量纲错配")
    if tr.shape[0] == 0:
        return None
    if not np.all(np.isfinite(tr)):
        raise FloatingPointError("★ O2（R14）：闭环 trace 含非有限值（不许静默跳过）")
    off = np.asarray(m_table, dtype=np.float64)
    if off.ndim != 1:
        raise ValueError(f"★ m_table 必须是一维，收到 {off.shape}")
    rows, ratios = [], []
    for h in h_refs:
        msk = tr[:, 0] == float(h)
        if int(msk.sum()) >= int(min_n) and int(h) <= off.size:
            cm = float(np.median(tr[msk, int(col)]))
            om = float(off[int(h) - 1])
            rows.append({"h": int(h), "n": int(msk.sum()), "e_med_closed": cm,
                         "e_med_offline": om, "ratio": cm / max(om, 1e-12)})
            ratios.append(cm / max(om, 1e-12))
    if not ratios:
        return None
    r = np.asarray(ratios, dtype=np.float64)
    return {"per_h": rows, "col": int(col), "ratio_median": float(np.median(r)),
            "ratio_min": float(r.min()), "ratio_max": float(r.max())}
