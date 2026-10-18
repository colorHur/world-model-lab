#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""T9 第③件 · **`SINRSuccess`（csi → 成功概率）+ E1 护栏不动 + I3 反例**。

一句话
------
T9 的交付物是"SINR/成功概率**由分配动作决定**"。`scripts/39` 已经证明 N=1 逐位复现单用户锚点，
但那是 `ConstantSuccess`（外生常数）；本脚本把 `SINRSuccess` 接上，判据是：
**新映射在 `csi = 1` 处必须回到 `q = 0.7`，于是 E1 的三个锚点一位都不许动。**

口径（★ 2026-10-18 选定 A，来源：`04-实验与代码/2026-10-18-T9-N2冒烟-v0.1.md` §6.1）
------------------------------------------------------------------------------
    κ := −ln(q_ref)；q(csi) := exp(−κ / csi)；q(0) := 0
⇒ `csi = 1 ⇔ q = q_ref = 0.7`（= 判-01 β 档那条信道），`csi↑ ⇒ q↑`。
选 A 而不选 B（`csi = γ̄/γ_th`，1 = 最差）的实际理由：B 下 `csi=1 ⇒ q=0.3679`，要表达
β 档得 `csi=2.8037 > 1`，越出 T8 的 `[0,1]` 契约；A 则让 E1 三个锚点原样可复用。

判据（★ 先写死再跑；`scripts/40` 的教训是"判据本身会错"，所以本脚本把**错判据也照原文判**）
------------------------------------------------------------------------------
| 编号 | 判据 | 失败意味着 |
|---|---|---|
| A1 | `|q(1.0) − 0.7| < 1e-12` | 归一化锚错 ⇒ E1 护栏当场失效 |
| A2 | `q(0.5)` 与 `exp(−2κ)` 相对误差 `< 1e-15`（解析恒等） | 公式写成了别的东西 |
| A3 | `q(0) == 0.0` 且不抛（全中断按极限取） | 除以零或悄悄返回 1 |
| A4 | `q(0.25) < q(0.5) < q(1.0)`（单调） | 方向反了 |
| A5 | `csi ∈ {1.2, −0.1, nan}` 各抛 `ValueError` | 越界不报错 ⇒ 静默外推 |
| B1 | 三个种子 `per_user_cost[0]` 与 `scripts/39` 的锚点**逐位相等**（`==`） | `SINRSuccess` 改了轨迹 |
| B2 | 分配器看到的 `slot.csi` 与传入值**逐个相等** | `csi` 参数没接线（R12） |
| B3 | N=2、`csi=[1.0, 0.5]` 时实测 `p_eff_i` 落在 `1 − q_i·share_i` 的 4σ 带内 | csi 没进到成功概率里 |
| D1 | **阴性对照（原文写法）**：把单调方向反接 `q = exp(−κ·csi)` ⇒ 三锚点**必须失配** | —— ★ 见下 |
| D2 | **阴性对照（原文写法）**：反接映射在 `csi=0.5` 处 ⇒ **两路** `p_eff` 都必须跑出 4σ 带 | —— ★ 见下 |
| D2' | **阴性对照（修正写法）**：只要求 `csi ≠ 1` 的那一路出带 | B3 是恒真检查 |
| D3 | **更强的对照**：传 `csi=[1.0, 0.5]` 但用 `ConstantSuccess(0.7)` ⇒ `csi=0.5` 那一路必须出带 | csi 根本没进成功概率（R12 的典型失效） |
| I3a | `require_privileged(view_privileged_None, "true_obs")` 抛 `MissingStateError` | I3 未实现 |
| I3b | **反例的对照**：`view.privileged or {}` 的写法**不抛**，静默返回 0 | 证明 I3a 不是恒真 |
| I3c | 未知键抛 `ValueError` | 特权白名单失效 |
| I3d | 有 dict 但缺键 / 值为 None ⇒ 抛 `MissingStateError` | 只挡了 None 没挡缺键 |

★★ **D1 / D2 两条都是错的判据，本脚本照原文判并如实报 FAIL**（与 `scripts/40` 的 C3c 同类）：
`csi = 1` 时 `exp(−κ·1) = exp(−κ) = q_ref` —— **反接映射在 csi=1 处与正接完全相同**（这是
口径 A 的定义点）。所以"反接 ⇒ 三锚点失配"（D1）与"反接 ⇒ **两路** p_eff 都出带"（D2）
在这个坐标上恒不可满足。这不是代码错，是**判据错**：
写它们时我没有把自己写下的 `csi=1 ⇔ q=q_ref` 代进去（违反 R12「判据写完先拿已知答案代验量级」）。
有效的对照是 D2'/D3：只对 `csi ≠ 1` 的那一路要求出带 —— 那是两个映射**唯一**不同的坐标。

用法
----
    D:/devtools/miniconda3/envs/ai-lab/python.exe scripts/41_t9_sinr_success_n1_regression.py

退出码：0 = 有效判据全过（D1 记为"判据缺陷"，不影响退出码）；2 = 有真失败。
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from wmlab.control import PDRelativeController  # noqa: E402
from wmlab.envs import make_env  # noqa: E402
from wmlab.eval.multiuser import (  # noqa: E402
    Allocation, ConstantSuccess, MissingStateError, SINRSuccess, View, delivery_rate,
    require_privileged, run_multiuser,
)

# ★ 来源：scripts/39 的 ANCHORS（= scripts/35 run_uav(0.3, seed, 30) 的本机实际输出）
ANCHORS = {
    0: 2.9093751196861266,
    1: 2.9058766984939575,
    2: 2.911704212506612,
}
Q_REF = 0.7             # 判-01 β 档：q = 1 − 0.3
N_EPISODES = 30
WARMUP = 300
MEASURED = 200
TAIL_FRAC = 0.25
SEED2 = 0               # B3 的冒烟种子（与 scripts/40 同）
CSI_PAIR = (1.0, 0.5)


# ---------------------------------------------------------------- 分配器
def always_serve_first(view: View, rng: np.random.Generator) -> Allocation:
    """N=1 退化分配器（与 scripts/39 同一个，不消耗 rng）。"""
    served = np.zeros(len(view.slots), dtype=bool)
    if view.budget_left > 0 and view.slots[0].alive:
        served[0] = True
    return Allocation(served=served, reason="always-serve[0]")


def alternate(view: View, rng: np.random.Generator) -> Allocation:
    """交替服务（与 scripts/40 同一个，不看 alive）。"""
    served = np.zeros(len(view.slots), dtype=bool)
    if view.budget_left > 0:
        served[(view.t - 1) % len(view.slots)] = True
    return Allocation(served=served, reason="alternate-by-t")


@dataclass
class CsiRecorder:
    """包一层分配器，记录它在 `View` 里**实际看到**的 csi（B2 的接线检查）。

    ★ 为什么单独记：`csi` 参数错了不会报错 —— 只会让所有臂都活在一个"信道全同"的世界里
      （R12：参数先问"它接线了吗"，判据是"改它会不会改变结果"）。
    """
    inner: object
    seen: list = field(default_factory=list)

    def __call__(self, view: View, rng: np.random.Generator) -> Allocation:
        self.seen.append(tuple(float(s.csi) for s in view.slots))
        return self.inner(view, rng)  # type: ignore[operator]


# ---------------------------------------------------------------- 故意错误的映射（阴性对照）
@dataclass(frozen=True)
class InvertedSINR:
    """**单调方向故意反接**：`q = exp(−κ·csi)`（正接是 `exp(−κ/csi)`）。

    存在理由只有 D1/D2：反接若照样"通过"，说明我的检查是恒真的。
    """

    q_ref: float = Q_REF

    def q(self, csi: float) -> float:
        kappa = -math.log(self.q_ref)
        return float(np.exp(-kappa * float(csi)))

    def __call__(self, view: View, alloc: Allocation) -> np.ndarray:
        return np.array([self.q(s.csi) for s in view.slots], dtype=float)


# ---------------------------------------------------------------- 运行工具
def run_n1(seed: int, success) -> dict:
    env = make_env("uav", seed=seed, max_steps=WARMUP + MEASURED)
    ctrl = PDRelativeController()
    out = run_multiuser(
        [env], [ctrl], always_serve_first, success,
        n_episodes=N_EPISODES, seed=seed, max_steps=MEASURED,
        warmup_steps=WARMUP, budget_total=MEASURED, tail_frac=TAIL_FRAC,
    )
    env.close()
    return out


def run_n2(csi_pair: tuple[float, float], success) -> dict:
    envs = [make_env("uav", seed=SEED2, max_steps=WARMUP + MEASURED) for _ in range(2)]
    ctrls = [PDRelativeController() for _ in range(2)]
    rec = CsiRecorder(alternate)
    out = run_multiuser(
        envs, ctrls, rec, success,
        n_episodes=N_EPISODES, seed=SEED2, max_steps=MEASURED,
        warmup_steps=WARMUP, budget_total=MEASURED, tail_frac=TAIL_FRAC,
        csi=list(csi_pair),
    )
    for e in envs:
        e.close()
    out["seen_csi"] = rec.seen
    return out


def n2_stats(out: dict, n_users: int = 2) -> dict:
    recs = out["records"]
    dr = delivery_rate(recs, n_users)
    attempts = [0] * n_users
    for r in recs:
        if r.served_idx >= 0:
            attempts[r.served_idx] += 1
    share = [attempts[i] / dr["alive_steps"][i] if dr["alive_steps"][i] else float("nan")
             for i in range(n_users)]
    return {"dr": dr, "attempts": attempts, "share": share}


def band4(q: float, n: int) -> float:
    return float(4.0 * math.sqrt(max(q * (1.0 - q), 0.0) / max(n, 1)))


def raises(fn, exc) -> bool:
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False


# ---------------------------------------------------------------- 主流程
def main() -> int:
    res: dict = {"q_ref": Q_REF, "anchors": ANCHORS, "csi_pair": list(CSI_PAIR),
                 "n_episodes": N_EPISODES, "warmup": WARMUP, "measured": MEASURED}
    ok_rows: list[tuple[str, bool]] = []

    model = SINRSuccess(Q_REF)
    kappa = model.kappa
    res["kappa"] = kappa

    print(f"[41] T9 · SINRSuccess  csi→q 口径 A（q(csi)=exp(−κ/csi)，κ=−ln {Q_REF} = {kappa:.10f}）")

    # ---------------- A 组：映射本身的解析判据
    q1 = model.q(1.0)
    q05 = model.q(0.5)
    q_analytic_05 = math.exp(-2.0 * kappa)
    a1 = bool(abs(q1 - Q_REF) < 1e-12)
    a2 = bool(abs(q05 - q_analytic_05) <= 1e-15 * max(abs(q_analytic_05), 1e-30))
    a3 = bool(model.q(0.0) == 0.0)
    a4 = bool(model.q(0.25) < q05 < q1)
    a5 = bool(all(raises(lambda c=c: model.q(c), ValueError) for c in (1.2, -0.1, float("nan"))))
    ok_rows += [("A1 |q(1)-0.7|<1e-12", a1), ("A2 q(0.5)=exp(-2κ)", a2),
                ("A3 q(0)=0", a3), ("A4 monotone", a4), ("A5 out-of-range raises", a5)]

    print(f"[41] A1 q(1.0) = {q1!r}  |q−0.7| = {abs(q1 - Q_REF):.3e}")
    print(f"[41] A2 q(0.5) = {q05!r}  exp(−2κ) = {q_analytic_05!r}  rel_err = "
          f"{abs(q05 - q_analytic_05) / q_analytic_05:.3e}")
    print(f"[41] A3 q(0.0) = {model.q(0.0)!r}   A4 q(.25)={model.q(0.25):.6f} < q(.5)={q05:.6f} "
          f"< q(1)={q1:.6f} = {a4}")

    # ---------------- B1：E1 三锚点逐位（SINRSuccess 取代 ConstantSuccess）
    print(f"\n[41] B1 · N=1 逐位回归（SINRSuccess(csi=1.0) 必须回到 ConstantSuccess(0.7) 的轨迹）")
    print(f"[41] {'seed':<6}{'got':>22}{'anchor':>22}{'bitwise':>9}{'p_eff':>9}{'lost':>7}")
    b1_all = True
    res["per_seed"] = {}
    for seed in sorted(ANCHORS):
        out = run_n1(seed, model)
        got = float(out["per_user_cost"][0])
        anchor = float(ANCHORS[seed])
        bit = bool(got == anchor)
        dr = delivery_rate(out["records"], 1)
        lost = int(dr["alive_steps"][0] - dr["frames"][0])
        b1_all = b1_all and bit
        res["per_seed"][str(seed)] = {"cost": got, "anchor": anchor,
                                      "abs_diff": float(abs(got - anchor)), "bit_exact": bit,
                                      "p_eff": float(dr["p_eff"][0]), "lost": lost}
        print(f"[41] {seed:<6}{got:>22.17g}{anchor:>22.17g}{str(bit):>9}"
              f"{dr['p_eff'][0]:>9.4f}{lost:>7}")
    ok_rows.append(("B1 anchors bitwise", b1_all))

    # ---------------- B2/B3：N=2、csi 异质
    out2 = run_n2(CSI_PAIR, model)
    st = n2_stats(out2)
    seen = out2["seen_csi"]
    expect_csi = tuple(float(x) for x in CSI_PAIR)
    b2 = bool(seen and all(set(s) == set(expect_csi) for s in seen))
    q_i = [model.q(c) for c in CSI_PAIR]
    exp_p = [1.0 - q_i[i] * st["share"][i] for i in range(2)]
    bd = [band4(q_i[i], st["attempts"][i]) for i in range(2)]
    b3 = bool(all(abs(st["dr"]["p_eff"][i] - exp_p[i]) <= bd[i] for i in range(2)))
    ok_rows += [("B2 csi wired into slots", b2), ("B3 p_eff matches 1-q·share", b3)]

    print(f"\n[41] B2/B3 · N=2 · csi={CSI_PAIR} · 交替服务 · seed {SEED2}")
    print(f"[41] 分配器看到的 csi 首末：{seen[0] if seen else None} … {seen[-1] if seen else None}"
          f"（共 {len(seen)} 次）")
    print(f"[41] {'user':<6}{'csi':>6}{'q':>9}{'attempts':>10}{'share':>8}"
          f"{'p_eff':>9}{'exp':>9}{'band':>8}")
    for i in range(2):
        print(f"[41] {i:<6}{CSI_PAIR[i]:>6.2f}{q_i[i]:>9.4f}{st['attempts'][i]:>10}"
              f"{st['share'][i]:>8.3f}{st['dr']['p_eff'][i]:>9.4f}{exp_p[i]:>9.4f}{bd[i]:>8.4f}")
    res["n2"] = {"csi": list(CSI_PAIR), "q": q_i, "attempts": st["attempts"],
                 "share": st["share"], "p_eff": st["dr"]["p_eff"],
                 "expected_p_eff": exp_p, "band": bd,
                 "budget_used": int(out2["budget_used"]), "n_steps": int(out2["n_steps"])}

    # ---------------- D1（原文判据，预期 FAIL）
    inv = InvertedSINR(Q_REF)
    d1_mismatch = []
    for seed in sorted(ANCHORS):
        got = float(run_n1(seed, inv)["per_user_cost"][0])
        d1_mismatch.append(bool(got != float(ANCHORS[seed])))
    d1_as_written = bool(any(d1_mismatch))
    # ---------------- D2（原文：两路都出带）/ D2'（修正：只要求 csi≠1 那一路）/ D3（常数 q 冒充）
    out2_inv = run_n2(CSI_PAIR, inv)
    st_inv = n2_stats(out2_inv)
    q_inv = [inv.q(c) for c in CSI_PAIR]
    # ★ 只对 csi ≠ 1 的用户有意义：csi = 1 时正接/反接恒等（口径 A 的定义点）
    diff_users = [i for i in range(2) if abs(CSI_PAIR[i] - 1.0) > 0.0]
    out_of_band = [bool(abs(st_inv["dr"]["p_eff"][i] - exp_p[i]) > bd[i]) for i in range(2)]
    d2_as_written = bool(all(out_of_band))
    d2_corrected = bool(all(out_of_band[i] for i in diff_users))
    # D3：传了 csi=[1,0.5] 但成功概率用常数 0.7 ⇒ csi 没进 q（R12 的典型失效）
    out2_const = run_n2(CSI_PAIR, ConstantSuccess(Q_REF))
    st_const = n2_stats(out2_const)
    d3_out = [bool(abs(st_const["dr"]["p_eff"][i] - exp_p[i]) > bd[i]) for i in range(2)]
    d3 = bool(all(d3_out[i] for i in diff_users))
    ok_rows += [("D2' inverted at csi!=1 leaves band", d2_corrected),
                ("D3 constant-q ignores csi (must leave band)", d3)]

    print(f"\n[41] D1（★ 原文判据，预期不可满足）反接 q=exp(−κ·csi) 在 csi=1 处："
          f"逐种子失配={d1_mismatch} ⇒ {d1_as_written}")
    print(f"[41]     ⇒ exp(−κ·1) = {math.exp(-kappa)!r} = q_ref ⇒ 反接与正接在 csi=1 **恒等**，"
          f"判据写错（不是代码错）")
    print(f"[41] D2（★ 原文：两路都出带）实测出带={out_of_band} ⇒ {d2_as_written}"
          f"（csi=1 那路**必然**在带内 ⇒ 原文同样不可满足）")
    print(f"[41] D2'（修正：只看 csi≠1 的 {diff_users} 路）q_inv={[round(x, 4) for x in q_inv]} "
          f"vs q_pos={[round(x, 4) for x in q_i]}；实测 p_eff={[round(x, 4) for x in st_inv['dr']['p_eff']]} "
          f"vs 正接期望 {[round(x, 4) for x in exp_p]} ⇒ {d2_corrected}")
    print(f"[41] D3（传 csi={CSI_PAIR} 但用 ConstantSuccess(0.7)）：实测 p_eff="
          f"{[round(x, 4) for x in st_const['dr']['p_eff']]} 出带={d3_out} ⇒ {d3}")

    # ---------------- I3：缺字段必须抛
    slot = View(t=1, slots=(), budget_left=1, budget_total=1, privileged=None)
    i3a = raises(lambda: require_privileged(slot, "true_obs"), MissingStateError)
    i3c = raises(lambda: require_privileged(slot, "true_love"), ValueError)
    i3d = all(raises(lambda v=v: require_privileged(v, "true_obs"), MissingStateError)
              for v in (View(1, (), 1, 1, {"true_age": np.array([0])}),
                        View(1, (), 1, 1, {"true_obs": None})))

    def permissive(view_: View, key: str):
        """**反例的对照**：把 None 当空 dict 继续读 ⇒ 静默降级成 0。这是 I3 要防的那个 bug。"""
        priv = view_.privileged or {}
        return priv.get(key, 0.0)

    i3b = bool(permissive(slot, "true_obs") == 0.0)   # 不抛 ⇒ 证明 i3a 不是恒真
    ok_rows += [("I3a MissingStateError", i3a), ("I3b permissive variant does not raise", i3b),
                ("I3c unknown key ValueError", i3c), ("I3d missing/None value", i3d)]

    print(f"\n[41] I3：a(抛 MissingStateError)={i3a}  c(未知键抛 ValueError)={i3c}  "
          f"d(缺键/None 值)={i3d}  b(对照：`or {{}}` 写法静默返回 {permissive(slot, 'true_obs')!r}，不抛)={i3b}")

    print("\n[41] ---- 判据 ----")
    for k, v in ok_rows:
        print(f"[41] {'✓' if v else '✗'} {k}")
    if not d1_as_written or not d2_as_written:
        print("[41] ⚠ 原文 D1（反接 ⇒ 锚点失配）/ D2（反接 ⇒ 两路都出带）都不可满足 —— "
              "**判据**写错，不是代码跑错（csi=1 处两映射恒等）；有效对照是 D2'/D3")

    ok = all(v for _, v in ok_rows)
    res.update({
        "A": {"q_at_1": q1, "abs_diff_q_ref": float(abs(q1 - Q_REF)),
              "q_at_half": q05, "q_half_analytic": q_analytic_05,
              "q_at_0": model.q(0.0), "monotone": a4},
        "B1_bitwise_all": b1_all,
        "B2_csi_seen_first": seen[0] if seen else None,
        "D1_as_written": {"pass": d1_as_written, "per_seed_mismatch": d1_mismatch,
                          "verdict": "criterion_ill_posed_at_csi_1"},
        "D2_as_written": {"pass": d2_as_written, "out_of_band": out_of_band,
                          "verdict": "criterion_ill_posed_at_csi_1"},
        "D2_corrected": {"pass": d2_corrected, "q_inverted": q_inv,
                         "p_eff_inverted": st_inv["dr"]["p_eff"]},
        "D3_constant_q": {"pass": d3, "p_eff": st_const["dr"]["p_eff"], "out_of_band": d3_out},
        "I3": {"a": i3a, "b": i3b, "c": i3c, "d": i3d},
        "checks": {k: bool(v) for k, v in ok_rows},
        "all_ok": bool(ok),
    })
    out_path = ROOT / "outputs" / "41_t9_sinr_success_n1_regression.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[41] {'PASS' if ok else 'FAIL'} · 写出 {out_path}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
