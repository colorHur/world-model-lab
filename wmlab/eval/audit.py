"""参照学习器有效性审计的**判定规则**（单一真源）。

★ 为什么单独成一个模块
----------------------
2026-10-12 复盘发现：`scripts/32` 的**代码**与**它的 docstring / JSON note**对同一条
验收规则写的是**两套**（代码 = "跨种子均值过线 且 每个种子改善 >0"；文字 = "每个种子过线"）
⇒ 换句话说，同一份产物**能同时给出两个不自洽的判定**。
根因是**规则没有单一真源**：判定逻辑散在脚本里、说明另写一遍。

⇒ 现在把判定抽成**纯函数** `reference_learner_verdict`，脚本 / 文档 / 测试**都从它取**。
"""

from __future__ import annotations

import numpy as np

__all__ = ["reference_learner_verdict", "REFERENCE_AUDIT_RULE", "same_seed_sequence"]


#: 规则的一句话陈述（脚本直接引用这句进 JSON，避免文档与实现再次分叉）
REFERENCE_AUDIT_RULE = (
    "VALID ⇔ **每一个**训练种子的「末 k 次评测均值」都 ≥ bar（bar = 随机基线 per-step + margin）。"
    "只要有一个种子不达标 ⇒ VOID。"
)


def reference_learner_verdict(per_seed_last_k_mean, bar) -> tuple[bool, dict]:
    """参照学习器有效性的**唯一**判定（v0.3，2026-10-12 统一）。

    Parameters
    ----------
    per_seed_last_k_mean : 每个训练种子一个数 —— 该种子「末 k 次评测」的均值。
    bar : 门槛 = 随机基线 per-step + margin。

    Returns
    -------
    (valid, detail)

    ★ 为什么是「**每个**种子都过线」而不是「跨种子均值过线 且 每个种子改善 >0」
    ----------------------------------------------------------------------
    多种子设计的**全部意义**就是防"某一个种子运气好"。取均值会把这种运气**重新摊回来**：
    设 3 个种子的改善为 `[0.1, 0.1, 4.4]`、`margin = 1.5` ⇒ 均值 1.533 ≥ bar，
    但**两个种子其实几乎没动** ⇒ "均值口径"会判 **VALID**，与"多数学会了才算数"的判断相反。
    ⇒ 采用**保守**方向：任一子不达标即 VOID（宁可真 VOID，不可假 VALID）。

    ⚠️ 边界（照实说）：本判断**不**回答"改善是否显著"—— margin 是**事先约定的
    "值得关注的最小改善"**，不是由标准误反推的显著性阈值。三种子阶段只报原始差值与离散度。
    """
    a = np.asarray(per_seed_last_k_mean, dtype=float).reshape(-1)
    if a.size == 0:
        raise ValueError("per_seed_last_k_mean 不能为空（至少要有一个训练种子）")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"per_seed_last_k_mean 含非有限值：{a.tolist()}（R14：不许静默通过）")

    bar = float(bar)
    clears = a >= bar
    valid = bool(clears.all())
    detail = {
        "rule": REFERENCE_AUDIT_RULE,
        "bar": bar,
        "n_seeds": int(a.size),
        "n_clearing_bar": int(clears.sum()),
        "per_seed_clears_bar": [bool(x) for x in clears],
        "per_seed_last_k_mean": [float(x) for x in a],
        "min_last_k_mean": float(a.min()),
        "mean_last_k_over_seeds": float(a.mean()),
        "std_last_k_over_seeds": (float(a.std(ddof=1)) if a.size > 1 else 0.0),
        "failing_seeds_idx": [int(i) for i in np.nonzero(~clears)[0]],
    }
    return valid, detail


def same_seed_sequence(a, b) -> bool:
    """两串**逐回合初值种子**是否**逐位相同**（值相同 **且** 顺序相同）。

    ★ 为什么"配对评测"的检查必须是**直接比对种子序列**（2026-10-12 收紧）
    ------------------------------------------------------------------
    旧检查用的是 `mean(A−B) == mean(A) − mean(B)`。看上去像"配对自检"，其实是个
    **恒等式**：只要两臂的逐回合数组**长度相同**，它**必然成立** ——
    哪怕两臂的 episode 顺序**完全错配**（例如把两臂的种子列表整体倒序），
    或者干脆是两组无关的等长数组，这个等式照样成立。
    ⇒ 它证明的是"长度相等"，**不是**"面对同一批初始条件"。

    真正能证明配对的，只有：把两臂**各自保存**的逐回合 seed 列表拿出来**按位比**。
    本函数就是这个按位比，供 `scripts/32` 与自检**共用**（单一真源）。

    Returns
    -------
    bool
        `True` ⇔ 等长且 `int(a[i]) == int(b[i])` 对每个 `i` 成立。
        ★ 空列表 == 空列表 ⇒ `True`（真空配对）；但调用方一般应另断言"非空"。
    """
    a = list(a)
    b = list(b)
    if len(a) != len(b):
        return False
    return all(int(x) == int(y) for x, y in zip(a, b))
