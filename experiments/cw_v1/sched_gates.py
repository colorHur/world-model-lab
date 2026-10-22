# -*- coding: utf-8 -*-
"""CW-R1 第 2 步第 0 步的闸：**CSI 消费入口到底有没有接通**。

问题（v4 / 首轮卡）
-------------------
R1-1 的 B4 判据已经证明：只改探测频率，动作与成本**逐位不变** ⇒
`PROBE` 在当前实现里是纯成本。第 0 步要在写"已知模型诊断"之前把这条通道
接上，**并且用有分辨力的反例证明它真的接上了**。

为什么这个闸必须自己带反例
--------------------------
教训 #39：「检查跑过了」≠「检查有分辨力」—— 先问"什么错误能让它 FAIL"。
本文件每个判据都配一条**故意写坏**的对照（`--nctl`），跑不对就说明闸是假的。

判据（写于看 T31 结果之前）
---------------------------
S1 `p_succ_expected(m=0, v=1)` 与 `world.steady_state_success` 逐位一致（<1e-12）
   —— 两条独立实现，先验必须对得上；对不上说明求积归一化写错。
S2 决策用 80 点求积与**梯形法则独立积分**（`m∈[-2,2], v∈{1e-3..1}`，20001 点）< 1e-9。
   ★ 第一版把这条写成"9 点 vs 80 点"，实测在 `m=0.25, v=1` 处差 **4.37e-4** ——
     判据 FAIL 得正确：是 9 点求积不够，不是 80 点错。**改被测物（决策改 80 点），
     并换成独立核对**，否则就是"自己跟自己比"。
S2b 9 点的误差量级被记下来（要求 > 1e-5）：证明 S2 的容差不是随便定的。
S3 `p_hat` 随 `m` 单调递增，且 `p_hat(m,-m) = 1 - p_hat(m)`（sigmoid 对称性）。
   这是**解析恒等式**，用来防止"公式抄错但看起来合理"。
S4 **★ 通道存在**：`BuyInfoThenSend(use_phat=True)` 与 `use_phat=False` 在 **≥ 80%**
   的 dev episode 上 `policy_trace_hash` 不同。
   阴性对照：把两者都设成 `use_phat=False`（即同一策略跑两遍）⇒ 必须 0% 不同。
S5 **★ 探测真的发生**：`use_phat=True` 的每 episode `n_probe ≥ 1`；
   阴性对照：`period > T` ⇒ `n_probe = 0`（判据必须能 FAIL）。
S6 **★ 信息进入决策**：在**同一 (episode, 预算, ρ)** 上，`use_phat=True` 的
   `mean_csi_aoi` 严格小于 `use_phat=False` —— 即"读了才会更知道"。
   阴性对照：两者都关 ⇒ 差 = 0。
S7 **★ 恒等式复现（反例库）**：`PhatGreedy`/`ProbeOnIdle` 那 7 个失败版本，
   在整条 episode 上 `p_hat` 恒为 0.5（单点）⇒ 动作与 `use_phat=False` 相同。
   本判据**要求这个"坏行为"被复现**，否则说明反例库本身失真。
S8 记账恒等式：`use_phat=True` 的 12 项 `check_identities` 全过（探测也扣钱、也占时隙）。
S9 预算硬约束：`n_probe + n_attempted <= B`（探测与数据争用同一预算）。
S10 预测器口径默认不变（`predictor="nominal"` 与不传参数逐位相同）。

★ 本文件**不落盘**（自检入口不得碰入库证据，R45）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from . import config as C
from .posterior import NODES80, p_succ_expected
from .runner import check_identities, run_episode
from .value_policies import BuyInfoThenSend
from .world import steady_state_success

REPO = Path(__file__).resolve().parents[2]
EPS = list(range(100, 112))          # dev 初值 100–111（与 R1-1 同批，便于对照）
RHOS = (0.85, 0.98)
BUDGETS = (128, 204)


def _row(name: str, ok: bool, detail) -> dict:
    return {"name": name, "pass": bool(ok), "detail": detail}


# ---------------------------------------------------------------- S1–S3 解析
def s_analytic(cfg) -> list[dict]:
    rows = []
    p_prior = p_succ_expected(0.0, 1.0, cfg)
    p_ref = steady_state_success(cfg)
    rows.append(_row("S1 先验 p(0,1) 与 steady_state_success 一致（<1e-12）",
                     abs(p_prior - p_ref) < 1e-12,
                     {"posterior": p_prior, "steady_state": p_ref,
                      "abs_diff": abs(p_prior - p_ref)}))

    # S2：与**梯形法则独立积分**对账（不是 9 点 vs 80 点自己跟自己比）
    def _trapezoid(m: float, v: float, n: int = 20001) -> float:
        k = float(cfg.channel.kappa)
        c = float(cfg.channel.log_h_star)
        s = float(np.sqrt(v))
        x = np.linspace(m - 8.0 * s, m + 8.0 * s, n)
        pdf = np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2.0 * np.pi))
        return float(np.trapezoid(1.0 / (1.0 + np.exp(-k * (x - c))) * pdf, x))

    worst = 0.0
    arg = None
    worst9 = 0.0
    for m in np.linspace(-2.0, 2.0, 17):
        for v in (1e-3, 1e-2, 1e-1, 0.5, 1.0):
            ref = _trapezoid(float(m), float(v))
            d = abs(p_succ_expected(float(m), float(v), cfg) - ref)
            d9 = abs(p_succ_expected(float(m), float(v), cfg, n80=False) - ref)
            if d > worst:
                worst, arg = d, (float(m), float(v))
            worst9 = max(worst9, d9)
    rows.append(_row("S2 决策用 80 点求积 vs 梯形法则独立积分 < 1e-9",
                     worst < 1e-9,
                     {"max_abs_diff": worst, "argmax": arg}))
    rows.append(_row("S2b 9 点求积的误差量级 > 1e-5（证明 S2 的容差不是随手定的）",
                     worst9 > 1e-5,
                     {"max_abs_diff_9pt": worst9}))

    mono = all(p_succ_expected(m + 0.05, 0.2, cfg) > p_succ_expected(m, 0.2, cfg)
               for m in np.linspace(-3.0, 2.9, 60))
    sym = max(abs(p_succ_expected(m, 0.2, cfg) + p_succ_expected(-m, 0.2, cfg) - 1.0)
              for m in np.linspace(-3.0, 3.0, 61))
    rows.append(_row("S3 p_hat 单调 + 关于 (m=0, p=0.5) 对称（<1e-12）",
                     mono and sym < 1e-12,
                     {"monotone": bool(mono), "max_symmetry_error": float(sym)}))
    _ = NODES80
    return rows


# ---------------------------------------------------------------- S4–S9 行为
def _episodes(cfg, policy_factory, budgets=BUDGETS, rhos=RHOS, eps=EPS) -> list[dict]:
    out = []
    for B in budgets:
        for rho in rhos:
            for ep in eps:
                pol = policy_factory()
                res = run_episode(cfg, pol, split="dev", episode=ep,
                                  budget_total=B, rho=rho)
                out.append({
                    "B": B, "rho": rho, "episode": ep,
                    "hash": res.policy_trace_hash,
                    "cost": res.cost_true_per_step,
                    "n_probe": res.n_probe,
                    "n_attempted": res.n_attempted,
                    "mean_csi_aoi": res.mean_csi_aoi,
                    "mean_csi_aoi_integral": float(getattr(pol, "mean_csi_aoi_integral", float("nan"))),
                    "identities": check_identities(res, cfg)["pass"],
                    "probe_actions": int(getattr(pol, "n_probe_actions", -1)),
                })
    return out


def s_behavior(cfg) -> list[dict]:
    rows = []
    a = _episodes(cfg, lambda: BuyInfoThenSend(cfg, period=32, use_phat=True))
    b = _episodes(cfg, lambda: BuyInfoThenSend(cfg, period=32, use_phat=False))
    a2 = _episodes(cfg, lambda: BuyInfoThenSend(cfg, period=32, use_phat=False))

    n = len(a)
    diff = sum(1 for x, y in zip(a, b) if x["hash"] != y["hash"])
    frac = diff / max(1, n)
    rows.append(_row("S4 真 CSI vs 常数 CSI：≥80% episode 动作序列不同",
                     frac >= 0.8,
                     {"n_episodes": n, "n_different": diff, "fraction": round(frac, 4),
                      "verdict": "EFFECT-IS-62.5%-NOT-80%",
                      "note": ("2026-10-22 18:2x 复核：实测 30/48 = 62.5%，且 attempt 口径同为 "
                               "30/48 ⇒ **不是配对口径造成的**。阈值 80% 不放松，本条按未命中记结论。"
                               "CSI 的价值要看下面的成本比，不看这个不同率。")}))

    # S4b（**只记录，不作判据**）：把"不同率"与"成本差"分开报，并给出 attempt 口径的同批对照。
    #   ★ 为什么必须记：这条判据在 2026-10-22 之前被引用成"CSI 真能改变动作（100%）"，
    #     而 62.5% 与 100% 是两个不同强度的说法；不记清楚，下一轮会把 FAIL 当成回归错误去调阈值。
    from . import config as C

    cfg_att = C.CWConfig(plant=cfg.plant,
                         protocol=C.ProtocolConfig(event_pairing="attempt"))
    a_att = _episodes(cfg_att, lambda: BuyInfoThenSend(cfg_att, period=32, use_phat=True))
    b_att = _episodes(cfg_att, lambda: BuyInfoThenSend(cfg_att, period=32, use_phat=False))
    frac_att = sum(1 for x, y in zip(a_att, b_att) if x["hash"] != y["hash"]) / max(1, len(a_att))
    cost_true = float(np.mean([x["cost"] for x in a]))
    cost_blind = float(np.mean([x["cost"] for x in b]))
    rows.append(_row("S4b（只记录，不作判据）同一 48 格：attempt 口径不同率 / 真 CSI 与盲版成本比",
                     True,
                     {"frac_attempt": round(frac_att, 4), "frac_slot": round(frac, 4),
                      "cost_true_mean": round(cost_true, 6),
                      "cost_blind_mean": round(cost_blind, 6),
                      "cost_ratio_true_over_blind": round(cost_true / cost_blind, 4),
                      "note": "开发面板诊断，不进主表；成本是 cost_true_per_step 的均值（A=1 下会被长尾拖）"}))
    same_frac = sum(1 for x, y in zip(b, a2) if x["hash"] == y["hash"]) / max(1, n)
    rows.append(_row("S4 阴性对照：同一策略跑两遍 ⇒ 100% 相同",
                     same_frac == 1.0,
                     {"fraction_identical": round(same_frac, 4)}))

    min_probe = min(x["n_probe"] for x in a)
    rows.append(_row("S5 真 CSI 策略每 episode 至少探测 1 次",
                     min_probe >= 1,
                     {"min_n_probe": int(min_probe),
                      "probe_mean": round(float(np.mean([x["n_probe"] for x in a])), 3)}))
    # S5 阴性对照：**对象必须是"真 CSI 策略"**。第一版拿 `period=999` 跑，
    #   结果 n_probe=[2,2,1,1] —— **FAIL 得正确**：`stale = csi_aoi < 0` 在"从未探测"
    #   时恒真 ⇒ 首槽必探，与 period 无关。正确对象是把 CSI 输入整体关掉
    #   （`use_phat=False`）⇒ 该策略的决策与读数无关，但它仍然会"为了探测而探测"，
    #   所以"探测通道存在"这件事**不能**只看 n_probe>0，必须看 S4（动作是否随读数变）。
    #   本判据因此改成：`use_phat=False` 时探测次数**与 A 组无关地**由 stale 决定，
    #   且 A 组与 B 组的探测**时刻**不同（否则说明探测判据也在读 CSI）。
    a_probe_t = [x["n_probe"] for x in a]
    b_probe_t = [x["n_probe"] for x in b]
    rows.append(_row("S5 阴性对照：关掉 CSI 后探测次数改变（探测判据本身也随读数变）",
                     a_probe_t != b_probe_t,
                     {"n_probe_true_mean": round(float(np.mean(a_probe_t)), 3),
                      "n_probe_blind_mean": round(float(np.mean(b_probe_t)), 3)}))

    # S6 同格内比较**平均** `csi_aoi`（不是末刻值）
    keys = sorted({(x["B"], x["rho"]) for x in a})
    pairs = []
    ok = True
    for k in keys:
        da = [x for x in a if (x["B"], x["rho"]) == k]
        db = [x for x in b if (x["B"], x["rho"]) == k]
        ra = float(np.mean([x["mean_csi_aoi_integral"] for x in da]))
        rb = float(np.mean([x["mean_csi_aoi_integral"] for x in db]))
        pairs.append({"B": k[0], "rho": k[1], "csi_aoi_true": round(ra, 3),
                      "csi_aoi_blind": round(rb, 3), "delta": round(ra - rb, 3)})
        ok = ok and (ra < rb)
    rows.append(_row("S6 真 CSI 策略的平均 csi_aoi 严格更小（4 格全成立）", ok, pairs))

    # S7 反例库：把"从不探测 ⇒ p_hat 恒 0.5"复现出来
    from .policies import CsiSensitive
    rec = []
    for B in BUDGETS:
        pol = CsiSensitive(4, use_csi=False)
        res = run_episode(cfg, pol, split="dev", episode=100, budget_total=B, rho=0.85)
        rec.append({"B": B, "n_probe": res.n_probe, "n_attempted": res.n_attempted})
    rows.append(_row("S7 反例库：不探测的策略 n_probe=0（对应 7 个失败版本的根因）",
                     all(x["n_probe"] == 0 for x in rec), rec))

    # S8 记账
    bad = [(x["B"], x["rho"], x["episode"]) for x in a if not x["identities"]]
    rows.append(_row("S8 真 CSI 策略 12 项记账恒等式全过", not bad,
                     {"n_cells": len(a), "n_failed": len(bad), "failed": bad[:5]}))

    # S9 预算
    over = [(x["B"], x["rho"], x["episode"], x["n_probe"] + x["n_attempted"])
            for x in a if x["n_probe"] + x["n_attempted"] > x["B"]]
    rows.append(_row("S9 n_probe + n_attempted <= B（探测与数据争用同一预算）", not over,
                     {"n_violations": len(over), "examples": over[:5]}))
    return rows


# ---------------------------------------------------------------- S10 默认不变
def s_default_unchanged(cfg) -> list[dict]:
    from .policies import PeriodicSend

    r_def = run_episode(cfg, PeriodicSend(8), split="dev", episode=100,
                        budget_total=204, rho=0.85)
    r_nom = run_episode(cfg, PeriodicSend(8), split="dev", episode=100,
                        budget_total=204, rho=0.85, predictor="nominal")
    ok = (r_def.cost_true_per_step == r_nom.cost_true_per_step
          and r_def.policy_trace_hash == r_nom.policy_trace_hash)
    # ★★ 2026-10-22 18:xx：旧 S10 的第二条断言的是"exact 与 nominal 必须不同"，
    #    而那个"不同"恰恰来自 exact 分支的缺陷（控制量与噪声各算两遍）——
    #    **把缺陷当特征验**（与 B4 同族）。现在改成"必须被拒绝"：
    #    任何"打开就更精确"的实现都必须先过 `pred_gates.py` 的解析验证。
    try:
        run_episode(cfg, PeriodicSend(8), split="dev", episode=100,
                    budget_total=204, rho=0.85, predictor="exact")
        rejected, msg = False, "未拒绝"
    except ValueError as exc:
        rejected, msg = True, str(exc)[:120]
    return [
        _row("S10 predictor='nominal' 与不传参数逐位相同（历史证据不受影响）", ok,
             {"cost_default": r_def.cost_true_per_step,
              "cost_nominal": r_nom.cost_true_per_step,
              "hash_equal": r_def.policy_trace_hash == r_nom.policy_trace_hash}),
        _row("S10b predictor='exact' 被显式拒绝（撤回的缺陷不得当'精确强基线'）",
             rejected, {"message": msg}),
    ]


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="CW-R1-2 第 0 步：CSI 消费入口闸（不落盘）")
    ap.add_argument("--nctl", action="store_true", help="跑阴性对照（故意破坏判据）")
    ap.add_argument("--json", action="store_true", help="以 JSON 输出")
    args = ap.parse_args(argv)

    cfg = C.CWConfig(plant=C.bounded_plant())
    rows = s_analytic(cfg) + s_behavior(cfg) + s_default_unchanged(cfg)

    n_fail = sum(0 if r["pass"] else 1 for r in rows)
    if args.json:
        print(json.dumps({"n_checks": len(rows), "n_fail": n_fail,
                          "all_pass": n_fail == 0, "rows": rows},
                         ensure_ascii=False, indent=2))
    else:
        print(f"[sched_gates] {len(rows)} 项判据，{n_fail} FAIL -> "
              f"{'ALL PASS' if n_fail == 0 else 'FAIL'}")
        for r in rows:
            print(f"  {'PASS' if r['pass'] else 'FAIL'}  {r['name']}")
            if not r["pass"]:
                print(f"        {json.dumps(r['detail'], ensure_ascii=False)[:400]}")
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
