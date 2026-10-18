"""Q4 §7 待核项：cartpole 代价口径「死亡后每步记 1、均值取全 L 步」是否等价于 `1 − 存活步数/500`。

背景
----
`04-实验与代码/2026-10-18-Q4-代价聚合口径-预注册-v0.1.md` §7 留了一条待核：

> `cartpole` 的 `C_i` 口径（死亡后每步记 1）是按判-01 的公式反推的，
> **没有**单独跑一次验证「逐 step 记 1 / 全 L 步取均值 == `1 − 存活步数/500`」。
> 若不成立，`R1` 的 cartpole 行要改。

这条必须在 Q4 签注**之前**闭环：R1 是预注册的一部分，拿一条没验过的行去要签字，
等于让签字人替一个未核的口径背书。

本脚本只做数值/逐位核对，不跑仿真（秒级）。三类检查：
  C0 判-01 实测 JSON 里 cartpole 的 `cost` 是否**逐位**等于 `1 − survival_frac`，且 `t_max == 500`
  C1 「逐 step 记 1、除以全 L」与「1 − 存活步数/L」在随机存活步数上是否**同一恒等式**（含浮点误差量级）
  C2 边界：第 1 步就死（alive=1）时两种读法给什么数 —— 用来判断判-01 §1「失败即代价 1」这句话是否过强

用法
----
    D:/devtools/miniconda3/envs/ai-lab/python.exe scripts/37_q4_cartpole_cost_identity.py
退出码 0 = 全部通过；1 = 有检查失败。
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
JSON_PATH = REPO / "outputs" / "35_judge_task_sensitivity.json"
L_CARTPOLE = 500          # 判-01 t_max（脚本内 T_MAX 同一值）
FAMILY = "cartpole"


def c0_json_identity() -> list[str]:
    """判-01 实测 JSON：`cost` 是否逐位等于 `1 − survival_frac`，`t_max` 是否 500。"""
    d = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    fam = d["families"][FAMILY]
    lines = [f"[C0] {JSON_PATH.name} · {FAMILY} · p={d['p_loss']} · seeds={d['seeds']}"]
    ok = True
    for cond in ("alpha_detail", "beta_detail"):
        for i, rec in enumerate(fam[cond]):
            cost = float(rec["cost"])
            frac = float(rec["survival_frac"])
            t_max = int(rec["t_max"])
            surv_mean = float(rec["survival_mean"])
            dev = abs(cost - (1.0 - frac))
            dev_frac = abs(frac - surv_mean / t_max)
            ok &= dev == 0.0 and t_max == L_CARTPOLE and dev_frac == 0.0
            lines.append(
                f"      {cond}[{i}] cost={cost!r} 1-frac={1.0 - frac!r} "
                f"Δ={dev:.3e} | frac={frac!r} surv/t_max={surv_mean / t_max!r} "
                f"Δ={dev_frac:.3e} | t_max={t_max}"
            )
    lines.append(f"[{'PASS' if ok else 'FAIL'}] C0 判-01 的 cost 逐位 == 1 − survival_frac，t_max 全为 500")
    return lines


def c1_identity() -> list[str]:
    """两种读法在随机存活步数上是否同一恒等式（不是"近似"，是同一个代数式）。"""
    rng = np.random.default_rng(0)
    n_ep = 30
    alive = rng.integers(0, L_CARTPOLE + 1, size=n_ep)          # 0 步也包含，边界一起测
    # 读法 A：逐 step 记代价（活着 0、死后 1），除以全 L，再对 episode 取均值
    per_step_sum = np.array([L_CARTPOLE - int(a) for a in alive], dtype=np.float64) / L_CARTPOLE
    cost_a = float(np.mean(per_step_sum))
    # 读法 B：判-01 实现（脚本 35 第 263 行）：1 − mean(surv)/t_max
    cost_b = 1.0 - float(np.mean(alive)) / float(L_CARTPOLE)
    dev = abs(cost_a - cost_b)
    lines = [
        f"[C1] {n_ep} 个 episode，alive∈[0,{L_CARTPOLE}]",
        f"      A（逐 step 记 1，除以全 L，再对 episode 取均值）= {cost_a!r}",
        f"      B（1 − mean(surv)/L，判-01 实现）          = {cost_b!r}",
        f"      Δ = {dev:.3e}（浮点级 ⇒ 同一代数式）",
    ]
    ok = dev <= 1e-15
    lines.append(f"[{'PASS' if ok else 'FAIL'}] C1 两种读法在浮点误差内恒等")
    return lines


def c2_death_at_first_step() -> list[str]:
    """边界：alive=1（第 1 步就 terminated）——「失败即代价 1」是否过强。"""
    cases = [1, 50, L_CARTPOLE]
    lines = ["[C2] 单 episode 边界：判-01 §1 写「失败即代价 1」，实际是「1 − 存活步数/L」"]
    ok = True
    for a in cases:
        cost = 1.0 - a / L_CARTPOLE
        is_one = math.isclose(cost, 1.0, rel_tol=0.0, abs_tol=0.0)
        if a == 1 and is_one:      # alive=1 时 formula 给 0.998，不是 1 ⇒ 该期望必须为假
            ok = False
        lines.append(f"      alive={a:>3} ⇒ cost={cost!r}  {'== 1.0' if is_one else '!= 1.0'}")
    lines.append(
        f"[{'PASS' if ok else 'FAIL'}] C2 alive=1 时 cost=0.998≠1.0 ⇒「失败即 1」只是上界，"
        "R1 必须写公式而不是写「失败即 1」"
    )
    return lines


def c3_alpha_zero_reason() -> list[str]:
    """α 侧 cartpole 代价恰为 0 的成因：是否每个 episode 都活满 L。"""
    d = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    fam = d["families"][FAMILY]
    fracs_alpha = [float(r["survival_frac"]) for r in fam["alpha_detail"]]
    fracs_beta = [float(r["survival_frac"]) for r in fam["beta_detail"]]
    ok = all(f == 1.0 for f in fracs_alpha) and any(f < 1.0 for f in fracs_beta)
    lines = [
        f"[C3] α survival_frac = {fracs_alpha} ⇒ cost 全 0（不是数值巧合，是全部活满）",
        f"      β survival_frac = {[round(f, 6) for f in fracs_beta]} ⇒ 有死亡，cost>0",
    ]
    lines.append(f"[{'PASS' if ok else 'FAIL'}] C3 α 的 0 代价可归因到 survival_frac≡1，β 侧有死亡")
    return lines


def main() -> int:
    print("== Q4 §7 待核项：cartpole 代价口径恒等性 ==")
    blocks = [c0_json_identity(), c1_identity(), c2_death_at_first_step(), c3_alpha_zero_reason()]
    fails = 0
    for b in blocks:
        print("\n".join(b))
        fails += sum(1 for ln in b if ln.startswith("[FAIL]"))
        print()
    print(f"== 结论：{4 - fails}/4 通过 ==")
    if fails:
        print("★ 有检查失败 ⇒ R1 的 cartpole 行必须改，不得据此签注 Q4。")
        return 1
    print("★ R1 的 cartpole 行成立（逐位/浮点级），但措辞必须写成公式 `1 − 存活步数/L`，")
    print("  不能写「失败即代价 1」（alive=1 时是 0.998，见 C2）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
