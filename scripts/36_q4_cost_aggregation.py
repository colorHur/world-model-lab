#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Q4 探针：Gate 2 的「任务代价比」在 N 个用户之间怎么聚合（不是新实验编号）。

一句话
------
**证明「用哪种聚合口径」能单独反转 Gate 2 的判定**，从而说明这个定义必须在看到任何
oracle/round-robin 数字之前预注册（X45 同型风险：同一份数据换口径就翻结论）。

本脚本**不跑仿真、不产生任何新实验数字**。它只做两件事：
  1. 从判-01 的实测结果里取两个族的 β 侧代价（作为"量级真实"的输入）；
  2. 在若干**假想**的 oracle 改善向量上，比较三种聚合口径给出的比值。

★ 所有场景里的 r_i 都是**人为设定的假设值**，不是实验结果。唯一真实的数字是
  `C_UAV_BETA` / `C_CART_BETA` 两个量级锚点（来源见下）。

输入来源（可核）
----------------
`outputs/35_judge_task_sensitivity.json`（判-01，3 种子 × 30 集，p=0.3）：
  uav_track.beta = 2.908987...   [m]，稳态跟踪距离
  cartpole.beta  = 0.171584...   [1−存活步数/500]，失败即代价 1
★ `outputs/` 不入库，故此处**内嵌常量**；若 JSON 在场则用 `--check-json` 逐位对账。

三种口径（定义见 `04-实验与代码/2026-10-18-Q4-代价聚合口径-预注册-v0.1.md`）
--------------------------------------------------------------------------
  sum   : Σ_i C_i^oracle / Σ_i C_i^rr        —— 用量纲直接相加（**仅诊断，不当判据**）
  arith : mean_i ( C_i^oracle / C_i^rr )     —— 逐用户比值的算术平均
  geo   : exp( mean_i ln( C_i^oracle / C_i^rr ) ) —— 逐用户比值的等权几何平均（**建议主判据**）

自检（秒级，见 `python scripts/36_q4_cost_aggregation.py`）
----------------------------------------------------------
  C1 N=1 退化：三种口径都必须逐位返回 r_1（保证 Gate 2 在 N=1 时与单任务比值连续）
  C2 等比例不变：所有 r_i 相同 ⇒ 三种口径都返回该 r
  C3 Σ 比 = 以 w_i = b_i/Σb 为权的加权算术平均（代数恒等式，必须逐位成立）
     ⇒ 「Σ 比」不是中立口径，它把话语权按**基线量级**分配出去了。
     ⚠️ 首版 C3 写成"单用户换单位 ⇒ Σ 比必须变"，实测**不变**（分子分母同步缩放），
        该断言是错的；真正的问题是权重，不是单位。留此记录，防止再犯。
  C4 零分母：某用户 round-robin 基数恰为 0 ⇒ 必须抛 DegenerateBaseline，不许返回 inf/nan
  C5 比例单调：所有 r_i 同乘 0.9 ⇒ 各口径都恰好下降 0.9 倍
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 判-01 实测（outputs/35_judge_task_sensitivity.json，p_loss=0.3，seeds=[0,1,2]，30 集/种子）
# ★ 2026-10-18 首轮写成四舍五入值 2.908987 / 0.171584，被 --check-json 打成 FAIL（偏差 1.7e-06 / 6.2e-06）。
#   现改为 JSON 原值逐位复制，容差收紧到 1e-12。这条记在这里，防止下次再手抄。
C_UAV_BETA = 2.908985343562232  # uav_track，稳态跟踪距离 [m]
C_CART_BETA = 0.1715777777777778  # cartpole，1 − 存活步数/500


class DegenerateBaseline(ValueError):
    """某用户在 round-robin 下的代价恰为 0 ⇒ 比值无定义（不得静默当成 0 或 inf）。"""


def _ratios(oracle: list[float], rr: list[float], eps: float = 0.0) -> list[float]:
    if len(oracle) != len(rr):
        raise ValueError(f"长度不一致: {len(oracle)} vs {len(rr)}")
    if len(oracle) == 0:
        raise ValueError("空输入")
    out = []
    for i, (o, b) in enumerate(zip(oracle, rr)):
        if b <= eps:
            raise DegenerateBaseline(f"用户 {i} 的基线代价 {b!r} ≤ {eps}，比值无定义")
        out.append(o / b)
    return out


def agg_sum(oracle: list[float], rr: list[float]) -> float:
    """ΣC/ΣC —— 量纲可加假设（UAV 用米、CartPole 用 0–1，**不成立**）。"""
    _ratios(oracle, rr)  # 借用校验：长度/零分母
    return sum(oracle) / sum(rr)


def agg_arith(oracle: list[float], rr: list[float]) -> float:
    """逐用户比值的算术平均（无量纲，但对"谁被牺牲"对称惩罚不足）。"""
    r = _ratios(oracle, rr)
    return sum(r) / len(r)


def agg_geo(oracle: list[float], rr: list[float]) -> float:
    """逐用户比值的等权几何平均 = exp(mean(ln r))（建议主判据）。"""
    r = _ratios(oracle, rr)
    return math.exp(sum(math.log(x) for x in r) / len(r))


AGGS = {"sum": agg_sum, "arith": agg_arith, "geo": agg_geo}


def _r(r_uav: float, r_cart: float) -> tuple[list[float], list[float]]:
    """把假想的逐用户比值还原成 (oracle, rr) 代价向量（rr 用判-01 实测 β 侧量级）。"""
    rr = [C_UAV_BETA, C_CART_BETA]
    return [r_uav * rr[0], r_cart * rr[1]], rr


def self_check() -> list[tuple[str, bool, str]]:
    res: list[tuple[str, bool, str]] = []

    # C1 N=1 退化：Gate 2 在 N=1 时必须回到单任务比值
    ok1 = all(abs(f([1.37], [1.0]) - 1.37) < 1e-12 for f in AGGS.values())
    res.append(("C1 N=1 退化（三口径 == r_1）", ok1,
                " / ".join(f"{k}={f([1.37],[1.0]):.3f}" for k, f in AGGS.items())))

    # C2 等比例不变
    o, b = _r(0.8, 0.8)
    ok2 = all(abs(AGGS[k](o, b) - 0.8) < 1e-12 for k in AGGS)
    res.append(("C2 等比例不变（全 r=0.8 ⇒ 三口径=0.8）", ok2,
                " / ".join(f"{k}={AGGS[k](o,b):.6f}" for k in AGGS)))

    # C3 Σ 比 == Σ_i w_i r_i，w_i = b_i/Σb（恒等式；证明 Σ 比是"按基线量级加权"的口径）
    o3, b3 = _r(0.55, 0.15)
    tot = sum(b3)
    w = [x / tot for x in b3]
    rhs = sum(wi * ri for wi, ri in zip(w, _ratios(o3, b3)))
    lhs = agg_sum(o3, b3)
    ok3 = abs(lhs - rhs) < 1e-12
    res.append(("C3 Σ 比 == Σ w_i·r_i（w_i=b_i/Σb，恒等式）", ok3,
                f"Σ比={lhs:.6f} 加权均值={rhs:.6f} 权重=({w[0]:.4f},{w[1]:.4f})"
                f" ⇒ UAV 占 {100*w[0]:.1f}% 话语权"))

    # C4 零分母必须抛，不许 inf/nan
    ok4 = True
    for k, f in AGGS.items():
        try:
            f([0.0, 0.5], [0.0, 1.0])
            ok4 = False
        except DegenerateBaseline:
            pass
    res.append(("C4 零分母抛 DegenerateBaseline（不许 inf/nan）", ok4, "三口径均抛"))

    # C5 比例单调：所有 r 同乘 0.9 ⇒ 各口径恰好下降 0.9
    o5, b5 = _r(0.6, 0.4)
    o5b = [x * 0.9 for x in o5]
    ok5 = all(abs(AGGS[k](o5b, b5) - 0.9 * AGGS[k](o5, b5)) < 1e-12 for k in AGGS)
    res.append(("C5 比例单调（r 同乘 0.9 ⇒ 各口径 ×0.9）", ok5,
                " / ".join(f"{k}:{AGGS[k](o5,b5):.6f}->{AGGS[k](o5b,b5):.6f}" for k in AGGS)))

    return res


SCENARIOS = [
    ("S0 基线（谁都没改善）", 1.00, 1.00),
    ("S1 两个用户都改善 50%", 0.50, 0.50),
    ("S2 只有 CartPole 改善 98%（UAV 不动）", 1.00, 0.02),
    ("S3 只有 UAV 改善 12%（CartPole 不动）", 0.88, 1.00),
    ("S4 牺牲 CartPole 换 UAV（1.6 / 0.10）", 1.60, 0.10),
]


def main() -> None:
    ap = argparse.ArgumentParser(description="Q4 聚合口径探针（假设场景，不跑仿真）")
    ap.add_argument("--check-json", action="store_true", help="与 outputs/35_*.json 逐位对账内嵌常量")
    args = ap.parse_args()

    if args.check_json:
        p = ROOT / "outputs" / "35_judge_task_sensitivity.json"
        if not p.exists():
            raise SystemExit(f"[C0] 找不到 {p}；本脚本的内嵌常量无法对账（不猜）")
        fam = json.loads(p.read_text(encoding="utf-8"))["families"]
        uav_b = fam["uav_track"]["beta"]
        cart_b = fam["cartpole"]["beta"]
        d_uav = abs(uav_b - C_UAV_BETA)
        d_cart = abs(cart_b - C_CART_BETA)
        print(f"[C0] json uav={uav_b!r} cart={cart_b!r} | 内嵌 {C_UAV_BETA} / {C_CART_BETA}")
        print(f"[C0] 偏差 {d_uav:.3e} / {d_cart:.3e} ⇒ {'PASS' if max(d_uav, d_cart) < 1e-12 else 'FAIL'}")
        if max(d_uav, d_cart) >= 1e-12:
            raise SystemExit("[C0] FAIL：内嵌常量与 JSON 不符，先修常量再谈口径")

    print("\n== 自检 ==")
    fails = 0
    for name, ok, detail in self_check():
        fails += 0 if ok else 1
        print(f"[{'PASS' if ok else 'FAIL'}] {name}\n        {detail}")
    if fails:
        raise SystemExit(f"\n自检 {fails} 条失败，不输出场景表")

    print("\n== 假设场景（★ 下列 r 是人为设定，不是实验结果）==")
    print(f"量级锚点（判-01 实测 β 侧）：UAV={C_UAV_BETA} m，CartPole={C_CART_BETA}")
    hdr = f"{'场景':<44}{'sum':>9}{'arith':>9}{'geo':>9}   Gate2(≤0.90) 判定分歧?"
    print(hdr)
    print("-" * len(hdr))
    verdicts = {}
    for name, ru, rc in SCENARIOS:
        o, b = _r(ru, rc)
        vals = {k: f(o, b) for k, f in AGGS.items()}
        verd = {k: ("过" if v <= 0.90 else "不过") for k, v in vals.items()}
        split = len(set(verd.values())) > 1
        verdicts[name] = (vals, verd, split)
        print(f"{name:<44}{vals['sum']:>9.4f}{vals['arith']:>9.4f}{vals['geo']:>9.4f}"
              f"   sum={verd['sum']}/arith={verd['arith']}/geo={verd['geo']}"
              f"{'  ← 分歧' if split else ''}")

    n_split = sum(1 for _, _, s in verdicts.values() if s)
    print("\n== 结论（机器数出）==")
    print(f"{len(SCENARIOS)} 个假设场景中有 {n_split} 个，三种口径的 Gate 2 判定不一致 ⇒ "
          f"口径不是技术细节，它单独决定 Gate 2 的生死。")


if __name__ == "__main__":
    main()
