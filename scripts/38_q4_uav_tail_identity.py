"""Q4 §7 剩下那条待核：`uav_track` 的 `C_i`（`mean_dist_tail`，`tail_frac=0.25`）口径是否接对。

背景
----
`04-实验与代码/2026-10-18-Q4-代价聚合口径-预注册-v0.1.md` §7.1 末尾留了一条：

> `R1` 的 uav 行（`mean_dist_tail`，`tail_frac=0.25`）未做同类恒等核对；
> 它与 X30–X45 同源，已有历史回归，但**本轮没有重验**。标**待核**，不阻断签注（数值路径未变）。

`scripts/37` 把 cartpole 那一行闭环了，这一条还没。这里补上，但**不跑仿真**（秒级）：
能靠代码/算术/已有 JSON 判定的，就不重新产生数字 —— 重跑会换 rng 路径，反而不如读同一份 JSON。

四类检查
--------
  C0 **接线**：`run_uav` 返回的 `cost` 是否就是 `r["mean_dist_tail"]` 的直接转发（AST 读，不靠肉眼）；
     `control.py` 里 `mean_dist_tail` 与尾窗 `k` 的表达式是否仍是已知那两条。
  C1 **尾窗算术**：`k = max(1, int(round(L*tail_frac)))` 在 `L=1..600` 上的真实行为 ——
     哪些 `L` 的 `k/L` 不等于 0.25、`round` 与 `floor` 在哪里分叉、`L=200` 时是否恰好 50。
  C2 **每集长度**：判-01 的 uav 六条明细里 `survival_mean == t_max` 且 `fail_rate == 0`
     ⇒ 30 集**全部跑满** 200 步（依据：单集 `L ≤ t_max`，均值取到 `t_max` ⇒ 无集短于 `t_max`）。
  C3 **两种读法在 C2 成立时恒等**：「先逐集取尾窗均值、再对集取均值」≡「把所有尾窗 step 合并取均值」，
     且反过来用变长 `L` 的合成数据证明这个恒等**不是空话**（变长时两者会分开）。

★ 本脚本**不**判断 Gate 2 的任何比值，也不产生新的实验数字；只核对口径。

用法
----
    D:/devtools/miniconda3/envs/ai-lab/python.exe scripts/38_q4_uav_tail_identity.py
退出码 0 = 全部通过；1 = 有检查失败。
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
JSON_PATH = REPO / "outputs" / "35_judge_task_sensitivity.json"
CONTROL = REPO / "wmlab" / "control.py"

FAMILY = "uav_track"
TAIL_FRAC = 0.25
L_UAV = 200          # 判-01 的 MEASURED_UAV；JSON 里 t_max 应与之一致
N_EP = 30            # 判-01 的 n_episodes


# ================================================================ C0 接线
def c0_wiring() -> tuple[list[str], bool]:
    """AST 读 `run_uav` 的返回字面量：`cost` 的值里必须出现 `mean_dist_tail`。"""
    src = (SCRIPTS / "35_judge_task_sensitivity.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name == "run_uav")
    ret = next(n for n in ast.walk(fn) if isinstance(n, ast.Return))
    assert isinstance(ret.value, ast.Dict), "run_uav 的 return 不是字典字面量"
    pairs = {}
    for k, v in zip(ret.value.keys, ret.value.values):
        if isinstance(k, ast.Constant):
            pairs[str(k.value)] = ast.unparse(v)
    cost_expr = pairs.get("cost", "")
    ok = "mean_dist_tail" in cost_expr and cost_expr.startswith("float(")

    lines = [f"[C0] scripts/35 · run_uav().cost 表达式（AST 读出，非肉眼）：",
             f"      cost = {cost_expr}"]
    lines.append(f"[{'PASS' if ok else 'FAIL'}] C0a uav 的 cost 是 mean_dist_tail 的直接转发（无二次变换）")

    ctrl = CONTROL.read_text(encoding="utf-8")
    s1 = '"mean_dist_tail": float(np.mean(tail_dists))'
    s2 = "k = max(1, int(round(len(ep_dist) * tail_frac)))"
    ok_a, ok_b = s1 in ctrl, s2 in ctrl
    lines.append(f"[{'PASS' if ok_a else 'FAIL'}] C0b control.py 的 mean_dist_tail = mean(tail_dists) 仍在")
    lines.append(f"[{'PASS' if ok_b else 'FAIL'}] C0c control.py 的尾窗 k = max(1, int(round(L*tail_frac))) 仍在")
    lines.append("      注：C0b/C0c 是**源码文本**核对，只能证「表达式没被改」；"
                 "它证的强度低于 C1/C2 的算术与数据核对。")
    return lines, ok and ok_a and ok_b


# ================================================================ C1 尾窗算术
def c1_window() -> tuple[list[str], bool]:
    """`k = max(1, int(round(L*0.25)))` 在 L=1..600 上的真实行为。"""
    Ls = np.arange(1, 601)
    ks = np.array([max(1, int(round(float(L) * TAIL_FRAC))) for L in Ls])
    exact = ks.astype(float) == Ls * TAIL_FRAC

    floor_k = np.array([int(float(L) * TAIL_FRAC) for L in Ls])
    fork = Ls[(ks != floor_k) & (ks != 1)]      # 排除 max(1,·) 顶掉的那几个
    L200 = max(1, int(round(L_UAV * TAIL_FRAC)))

    lines = [f"[C1] tail_frac={TAIL_FRAC}，L=1..600",
             f"      k/L 恰为 0.25 的 L 数 = {int(exact.sum())}/{len(Ls)} "
             f"（其余被取整顶偏）",
             f"      round 与 floor 分叉的 L（前 10 个）= {fork[:10].tolist()}"
             f" 共 {int(fork.size)} 个",
             f"      L=1..3: k = {ks[:3].tolist()} ⇒ k/L = "
             f"{(ks[:3] / Ls[:3]).round(3).tolist()}（短集「尾巴」不是 25%）",
             f"      L={L_UAV}（判-01 的被测段）: k = {L200} ⇒ k/L = {L200 / L_UAV:.4f}"]
    ok = (L200 == L_UAV // 4) and bool((ks >= 1).all()) and bool((ks <= Ls).all())
    lines.append(f"[{'PASS' if ok else 'FAIL'}] C1 尾窗非空且不越界；L={L_UAV} 时 k 恰为 {L_UAV // 4}")
    return lines, ok


# ================================================================ C2 每集长度
def c2_ep_len() -> tuple[list[str], bool]:
    """判-01 uav 明细：`survival_mean == t_max` 且 `fail_rate == 0` ⇒ 每集都跑满。"""
    d = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    fam = d["families"][FAMILY]
    lines = [f"[C2] {JSON_PATH.name} · {FAMILY} · p={d['p_loss']} · "
             f"seeds={d['seeds']} · n_episodes={d['n_episodes']}"]
    ok = True
    for cond in ("alpha_detail", "beta_detail"):
        for i, rec in enumerate(fam[cond]):
            t_max = int(rec["t_max"])
            surv = float(rec["survival_mean"] or 0.0)
            fail = float(rec.get("fail_rate", float("nan")))
            same = (surv == float(t_max)) and (t_max == L_UAV) and (fail == 0.0)
            ok &= same
            lines.append(f"      {cond}[{i}] survival_mean={surv!r} t_max={t_max} "
                         f"fail_rate={fail!r} ⇒ {'每集跑满' if same else '★有短集/逃逸'}")
    lines.append(f"[{'PASS' if ok else 'FAIL'}] C2 六条明细全为 survival_mean == t_max == {L_UAV}、fail_rate == 0")
    lines.append("      推论（单集 L ≤ t_max，均值取到 t_max ⇒ 无一集短于 t_max）⇒ "
                 f"30 集长度恒为 {L_UAV}，尾窗 k 恒为 {L_UAV // 4}")
    return lines, ok


# ================================================================ C3 两种读法
def c3_pooling() -> tuple[list[str], bool]:
    """恒定 L 时两种读法恒等；变长 L 时会分开（用来证明 C3 不是空话）。"""
    rng = np.random.default_rng(0)

    def tail_mean_of_episode(x: np.ndarray) -> float:
        k = max(1, int(round(len(x) * TAIL_FRAC)))
        return float(np.mean(x[-k:]))

    # (a) 恒定 L：与判-01 uav 同形（30 集 × 200 步）
    fixed = rng.random((N_EP, L_UAV))
    per_ep = float(np.mean([tail_mean_of_episode(x) for x in fixed]))
    k = max(1, int(round(L_UAV * TAIL_FRAC)))
    pooled = float(np.mean(fixed[:, -k:]))
    dev_fixed = abs(per_ep - pooled)

    # (b) 变长 L：同一份指标、只改集长（CartPole 掉杆那种情形）
    lens = rng.integers(20, L_UAV + 1, size=N_EP)
    seqs = [rng.random(int(L)) for L in lens]
    per_ep_v = float(np.mean([tail_mean_of_episode(x) for x in seqs]))
    pooled_v = float(np.mean(np.concatenate([x[-max(1, int(round(len(x) * TAIL_FRAC))):]
                                             for x in seqs])))
    dev_var = abs(per_ep_v - pooled_v)

    lines = [f"[C3] 恒定 L（{N_EP}×{L_UAV}）：逐集尾均值再平均={per_ep!r} 合并尾窗={pooled!r} "
             f"Δ={dev_fixed:.3e}",
             f"      变长 L（{lens.min()}..{lens.max()}）：逐集={per_ep_v!r} 合并={pooled_v!r} "
             f"Δ={dev_var:.3e} ⇒ 变长时两读法分开（说明上一条不是空话）"]
    ok = dev_fixed <= 1e-12 and dev_var > 1e-6
    lines.append(f"[{'PASS' if ok else 'FAIL'}] C3 恒定 L ⇒ 两读法逐位等价；变长 L ⇒ 分开")
    lines.append("      ⇒ 判-01 的 uav 上，「逐集尾均值再平均」与「合并尾窗」**同值**，"
                 "R1 怎么写都不改数；但 T10 接 CartPole（集长可变）时这条**不成立**，必须写成 R1 的逐集口径。")
    return lines, ok


# ================================================================ C4 聚合路径
def c4_seed_agg() -> tuple[list[str], bool]:
    """`beta == mean(beta_per_seed)`（同一份 JSON 内），防转录/路径错位。"""
    d = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    fam = d["families"][FAMILY]
    lines, ok = [f"[C4] {FAMILY} 的种子聚合路径"], True
    for key in ("alpha", "beta"):
        agg = float(fam[key])
        per = np.asarray(fam[f"{key}_per_seed"], dtype=float)
        dev = abs(agg - float(per.mean()))
        ok &= dev <= 1e-12
        lines.append(f"      {key}={agg!r}  mean(per_seed)={float(per.mean())!r} "
                     f"Δ={dev:.3e}（n_seed={per.size}）")
    lines.append(f"[{'PASS' if ok else 'FAIL'}] C4 汇总值 == 逐种子均值（1e-12 内）")
    return lines, ok


def main() -> int:
    print("=" * 78)
    print("Q4 · uav 行（mean_dist_tail, tail_frac=0.25）口径核对 —— 只读 JSON / 只算算术，不跑仿真")
    print("=" * 78)
    all_ok = True
    for label, fn in (("C0", c0_wiring), ("C1", c1_window),
                      ("C2", c2_ep_len), ("C3", c3_pooling), ("C4", c4_seed_agg)):
        lines, ok = fn()
        for ln in lines:
            print(ln)
        all_ok &= ok
        print()
    print("=" * 78)
    print(f"[{'PASS' if all_ok else 'FAIL'}] 全部检查")
    print("未核（本脚本刻意不碰）：pooled 与 per-episode 之外的第三种读法（如按步加权/按集加权混合）"
          "没有意义，未列；uav 的 `mean_dist_tail` 是否**物理上**代表稳态跟踪误差，属 X30–X45 的历史结论，"
          "本轮不重开。")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
