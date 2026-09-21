# -*- coding: utf-8 -*-
"""[X22 前置] 与理论界对账的可行性检验 —— arXiv:2605.15960 的 safe horizon。

背景
----
C17（*Imperfect World Models are Exploitable*）的 Theorem 2 给出

    delta = 1/2 * max_{s,a} || T(.|s,a) - T'(.|s,a) ||_1      （两转移模型的 TV 距离）
    H(eps,delta) = [ (1+eps) + sqrt( (1-eps)^2 + 4*eps/delta ) ] / 2
    条件： 1/(1-gamma) <= H(eps,delta)  =>  eps-unexploitable（且该界是紧的）

原计划（X22）是"把实测 NMSE 当 delta 代进去，算出 H，与实测 H* 并排报出 gap"。
**本脚本检验这个动作在数学上能不能做。结论是不能，理由有三条，见下。**

本脚本做的全是**【可断言的前提检验】**，不是新实验 —— 不训练、不用 GPU、不依赖
既有产物以外的数据。可以直接 `python scripts/08_x22_theory_check.py` 复现。

三条不匹配（脚本逐条验证）
--------------------------
① measurand 不同：C17 的 H 约束"策略序反转幅度 <= eps"（涉及 J 与 gamma）；
   我方 H* 是"预测误差首次超阈的步数"（不涉及策略与价值）。→ [E] 段
② delta 在确定性环境上退化为二值：环境确定 + 模型确定 ⇒ T 与 T' 都是点质量
   ⇒ TV ∈ {0,1}，取 max 后 delta ≡ 1 ⇒ H(eps,1) = 1+eps（闭式），
   定理条件变成 gamma <= eps/(1+eps)，对任何有意义的 gamma 都不成立。→ [A][B][C] 段
③ 空间不同：我方转移在潜空间，delta 定义在状态-动作空间。→ 见实验记录，非脚本可验

★ 本脚本同时**复算**看板里记的 H*(theta) 幂律，并断言其符号。
"""
from __future__ import annotations

import csv
import math
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FAIL = []
WARN = []


def check(name: str, ok: bool, detail: str = "") -> None:
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def warn(name: str, detail: str = "") -> None:
    """不作为断言失败、但必须被看见的发现。"""
    print(f"  [WARN] {name}" + (f"   {detail}" if detail else ""))
    WARN.append(name)


# ============================================================ [A] 环境确定性
print("[08] " + "=" * 66)
print("[08] [A] 环境转移是否为点质量（决定 T(.|s,a) 的形状）")
print("[08] " + "=" * 66)

import gymnasium as gym

STATES = {
    "Pendulum-v1": np.array([0.30, 0.40], dtype=np.float64),
    "CartPole-v1": np.array([0.05, 0.10, 0.02, -0.05], dtype=np.float64),
}
ACTIONS = {
    "Pendulum-v1": np.array([1.5], dtype=np.float32),
    "CartPole-v1": 1,
}

for env_id, s0 in STATES.items():
    env = gym.make(env_id)
    env.reset(seed=123)
    a = ACTIONS[env_id]

    def _step_once(state):
        env.unwrapped.state = state.copy()
        out = env.step(a)
        return np.asarray(out[0], dtype=np.float64)

    s1 = _step_once(s0)
    s2 = _step_once(s0)
    s3 = _step_once(s0)
    same = np.array_equal(s1, s2) and np.array_equal(s2, s3)
    check(f"{env_id}: 同一 (s,a) 三次 step 得到同一 s'", same,
          f"max|diff|={np.abs(s1 - s2).max():.3e}")
    env.close()

print("[08]   => 给定 (s,a) 下一状态唯一确定 ⇒ T(.|s,a) 是点质量分布。")

# ============================================================ [B] 模型确定性
print()
print("[08] " + "=" * 66)
print("[08] [B] 世界模型转移是否为点质量（读代码 + 实测）")
print("[08] " + "=" * 66)

from wmlab.models.world_model import MLPWorldModel

torch.manual_seed(0)
m_pd = MLPWorldModel(obs_dim=3, act_dim=1, latent_dim=32, hidden=128, discrete_act=False).eval()
m_cp = MLPWorldModel(obs_dim=4, act_dim=2, latent_dim=32, hidden=128, discrete_act=True).eval()

with torch.no_grad():
    obs = torch.randn(4, 3)
    act = torch.randn(4, 1)
    o1 = m_pd.predict_next(obs, act)
    o2 = m_pd.predict_next(obs, act)
    check("MLPWorldModel(Pendulum): 同一输入两次 predict_next 逐位一致",
          torch.equal(o1, o2), f"max|diff|={(o1 - o2).abs().max().item():.3e}")

    obs_c = torch.randn(4, 4)
    act_c = torch.tensor([0, 1, 0, 1]).reshape(4, 1)
    c1 = m_cp.predict_next(obs_c, act_c)
    c2 = m_cp.predict_next(obs_c, act_c)
    check("MLPWorldModel(CartPole): 同上", torch.equal(c1, c2),
          f"max|diff|={(c1 - c2).abs().max().item():.3e}")

has_var_head = any("logvar" in n or "std" in n for n, _ in m_pd.named_parameters())
check("模型无方差头（transition 直出 latent 向量，非分布参数）", not has_var_head,
      f"参数量={sum(p.numel() for p in m_pd.parameters())}")
print("[08]   => T'(.|s,a) 也是点质量分布（确定性网络输出）。")

# ============================================================ [C] TV 二值性
print()
print("[08] " + "=" * 66)
print("[08] [C] 两个点质量分布的 TV 距离只能取 {0,1}")
print("[08] " + "=" * 66)


def tv_two_point_masses(same_bin: bool, n_bins: int = 1000) -> float:
    """把状态空间离散化后，算两个点质量分布的 1/2 * L1 距离。"""
    p = np.zeros(n_bins)
    q = np.zeros(n_bins)
    p[100] = 1.0
    q[100 if same_bin else 500] = 1.0
    return 0.5 * float(np.abs(p - q).sum())


tv_same = tv_two_point_masses(True)
tv_diff = tv_two_point_masses(False)
check("预测完全吻合 => delta = 0", abs(tv_same - 0.0) < 1e-12, f"delta={tv_same:.3e}")
check("预测不吻合   => delta = 1", abs(tv_diff - 1.0) < 1e-12, f"delta={tv_diff:.3e}")
print("[08]   => delta 取 max_{s,a}，故【只要存在一处预测不精确，delta = 1】。")
print("[08]   => 在 Pendulum/CartPole 上（模型必然不完美）delta ≡ 1。")

# ============================================================ [D] H(eps,1)=1+eps
print()
print("[08] " + "=" * 66)
print("[08] [D] delta = 1 时 H 的闭式与数值（两条独立路径）")
print("[08] " + "=" * 66)


def H(eps: float, delta: float) -> float:
    return ((1.0 + eps) + math.sqrt((1.0 - eps) ** 2 + 4.0 * eps / delta)) / 2.0


print("[08]   手算路径: (1-eps)^2 + 4*eps = 1 + 2eps + eps^2 = (1+eps)^2")
print("[08]              => sqrt(...) = 1+eps  => H = [(1+eps)+(1+eps)]/2 = 1+eps")
worst = 0.0
for eps in (0.001, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.9):
    worst = max(worst, abs(H(eps, 1.0) - (1.0 + eps)))
check("数值路径与闭式一致（8 个 eps）", worst < 1e-12, f"最大绝对差={worst:.3e}")

mono = all(H(0.05, d1) > H(0.05, d2) for d1, d2 in [(0.01, 0.1), (0.1, 1.0)])
check("H 随 delta 单调递减（delta=1 给出最小值）", mono)

print()
print("[08]   定理条件在 delta=1 下退化为:  1/(1-gamma) <= 1+eps  <=>  gamma <= eps/(1+eps)")
for eps in (0.01, 0.05, 0.1, 0.2):
    print(f"[08]     eps={eps:<5} 允许的最大 gamma = {eps / (1 + eps):.6f}")
check("对 gamma=0.9/0.99 条件均不成立（界无分辨力）",
      all(0.9 > eps / (1 + eps) for eps in (0.01, 0.05, 0.1, 0.2)))

# ============================================================ [E] 幂律复算
print()
print("[08] " + "=" * 66)
print("[08] [E] 复算实测 H*(theta) 幂律（校正看板记录）")
print("[08] " + "=" * 66)

CSV = ROOT / "outputs" / "07_seeds_threshold.csv"
if not CSV.exists():
    print(f"[08]   跳过：找不到 {CSV}（outputs/ 不入库，需先跑 07）")
else:
    rows = list(csv.DictReader(open(CSV, encoding="utf-8-sig")))
    seeds = sorted({int(r["seed"]) for r in rows})

    def fit(col: str):
        alphas, coefs, r2s = [], [], []
        for s in seeds:
            sub = sorted([r for r in rows if int(r["seed"]) == s],
                         key=lambda r: float(r["threshold"]))
            x = np.log([float(r["threshold"]) for r in sub])
            y = np.log([float(r[col]) for r in sub])
            (a, b), *_ = np.linalg.lstsq(
                np.vstack([x, np.ones_like(x)]).T, y, rcond=None)
            yh = a * x + b
            r2s.append(1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum())
            alphas.append(float(a))
            coefs.append(float(np.exp(b)))
        return alphas, coefs, r2s

    for col in ("open_sparse", "open_dense", "closed_sparse", "closed_dense"):
        a, c, r2 = fit(col)
        print(f"[08]   {col:<15} alpha = {np.mean(a):+.4f} +/- {np.std(a, ddof=1):.4f}"
              f"   coef = {np.mean(c):6.2f}   min R^2 = {min(r2):.4f}")
    a_sparse, _, _ = fit("open_sparse")
    check("实测 H*(theta) 的幂指数为正（阈值越松、视界越长）",
          all(v > 0 for v in a_sparse),
          "alphas = " + ", ".join(f"{v:+.3f}" for v in a_sparse))

    # 看板记录的式子
    print()
    print("[08]   ★ 看板《冲刺看板.md》记录为: 开环 62.19 * theta^(-0.277)  (R2 0.981)")
    print("[08]     逐点核对（用 seed=0 / open_dense 的实测值）:")
    t0 = sorted(float(r["threshold"]) for r in rows if int(r["seed"]) == 0)
    meas = {float(r["threshold"]): float(r["open_dense"])
            for r in rows if int(r["seed"]) == 0}
    print(f"[08]     {'theta':>8}{'实测':>10}{'看板式子':>12}{'相对偏差':>12}")
    for t in t0:
        b = 62.19 * (t ** -0.277)
        print(f"[08]     {t:>8.2f}{meas[t]:>10.2f}{b:>12.2f}"
              f"{(b - meas[t]) / meas[t] * 100:>11.1f}%")
    consistent = abs(float(np.mean(a_sparse)) - (-0.277)) < 0.05
    if consistent:
        check("看板记录的指数（-0.277）与实测一致", True)
    else:
        warn("看板记录的指数符号与实测相反",
             f"看板记 -0.277（负），实测 {np.mean(a_sparse):+.4f}（正）"
             " => 记录的负号是抄写错误；系数 62.19 与数量级本身是对的")

print()
print("[08] " + "=" * 66)
if FAIL:
    print(f"[08] 结论: {len(FAIL)} 项未通过 -> " + "; ".join(FAIL))
    sys.exit(1)
print(f"[08] 结论: 全部前提检验通过（{len(WARN)} 条 WARN 见上）。")
if WARN:
    for w in WARN:
        print(f"[08]   [WARN] {w}")
print("[08]   => C17 的 safe horizon 在【确定性动力学 + 确定性模型】上 delta ≡ 1，")
print("[08]      H(eps,1) = 1+eps，对任何有意义的 gamma 都不满足定理条件，界退化为平凡。")
print("[08]   => 因此【不能】把实测 NMSE 当 delta 代入去'对账'；")
print("[08]      该动作在 measurand / delta 取值 / 空间 三个层面都不成立。")
print("[08] " + "=" * 66)

# ============================================================ [F] 出图
print()
print("[08] [F] 出图（直观展示'两边不在一个量级'）")
print("[08]   标签一律用 ASCII/数学，避免中文依赖字体 —— 否则陌生人跑出来是方框")
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    # 左：H(eps=0.05, delta) 随 delta 变化
    ds = np.logspace(-3, 0, 400)
    hs = [H(0.05, d) for d in ds]
    ax = axes[0]
    ax.semilogx(ds, hs, lw=2, color="tab:blue",
                label=r"$H(\varepsilon{=}0.05,\ \delta)$")
    ax.axvline(1.0, color="tab:red", ls="--", lw=1.2)
    ax.plot([1.0], [H(0.05, 1.0)], "o", color="tab:red", ms=9, zorder=5)
    ax.annotate(r"$\delta=1$: the only value possible"
                "\n" r"in a deterministic setting"
                "\n" r"$\Rightarrow H=1+\varepsilon=1.05$",
                xy=(1.0, H(0.05, 1.0)), xytext=(0.015, 3.4),
                arrowprops=dict(arrowstyle="->", color="tab:red"),
                color="tab:red", fontsize=9)
    ax.set_xlabel(r"$\delta$  (TV distance between $T$ and $T'$)")
    ax.set_ylabel(r"$H(\varepsilon,\delta)$   [steps]")
    ax.set_title(r"C17 safe horizon: $\delta{=}1$ squeezes $H$ into $(1,1{+}\varepsilon]$",
                 fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    # 右：实测 H*(theta)
    ax = axes[1]
    if CSV.exists():
        for s in seeds:
            sub = sorted([r for r in rows if int(r["seed"]) == s],
                         key=lambda r: float(r["threshold"]))
            ax.plot([float(r["threshold"]) for r in sub],
                    [float(r["open_dense"]) for r in sub],
                    "o-", ms=5, lw=1.0, alpha=0.8, label=f"seed {s}")
        ts = np.logspace(np.log10(0.01), np.log10(0.2), 100)
        ax.plot(ts, 61.34 * ts ** 0.2290, "k--", lw=1.8,
                label=r"fit: $61.3\,\theta^{+0.229}$")
        ax.axhspan(1.0, 1.2, color="tab:red", alpha=0.22)
        ax.text(0.0105, 4.0,
                "EVERY possible C17 value at $\\delta{=}1$:\n"
                r"$H\in(1,\ 1.2]$   (red band)",
                color="tab:red", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel(r"threshold $\theta$")
    ax.set_ylabel(r"measured $H^*$   [steps]")
    ax.set_title(r"measured $H^*(\theta)$: two orders of magnitude away",
                 fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")

    fig.suptitle("X22: can C17's safe horizon be reconciled with measured $H^*$? "
                 "-- different measurands, not comparable", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    png = ROOT / "outputs" / "08_x22_theory_check.png"
    fig.savefig(png, dpi=150)
    print(f"[08]   已写出 {png}")
except Exception as exc:  # 出图失败不影响前提检验的结论
    print(f"[08]   出图跳过（{type(exc).__name__}: {exc}）")
