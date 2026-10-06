# -*- coding: utf-8 -*-
"""调度基线库 v0.1 —— 把散在 `scripts/24..29` 里的**六个调度族**收成一张可枚举的表。

★★ 为什么值得单独成库（不是"再加一族"）
     X38–X45 这条链上，调度器一直是在脚本里**内联写的闭包**，而且每一轮都要把
     「等预算配对 + 发散过滤 + 结果落盘」重写一遍。同一件"等率比较"在本仓库出现过
     至少 4 份实现，其中 **2 次因为守卫写错产出过假头条**：
       · X40：「9/9 更优」，实则 9 个网格点挤在 5% 的速率跨度里（⇒ `pairing.py`）
       · X45：同一份数据 **max 口径 T=8 FAIL / CV 口径 PASS** ⇒ 口径本身就是混淆变量
     ⇒ 本模块的目标是把「换一个触发统计量」从"复制 200 行"降级成"换一个字符串"，
       并把三条硬约定**写成代码而不是注释**。

三件硬约定
    C1 **一个调度器 = 一个 `(t, rng) -> bool`**，与 `eval.tracking.Schedule` 同型。
       状态（`age` / `U` / oracle 真值误差）由闭环 runner 写进 `state` 字典，调度器**只读不写**。
    C2 **等预算比较必须走 `pairing.overlap_window`**（点数 + 跨度两条守卫），
       且**同一批比较只准用同一个发散过滤器**。本模块**不提供**"取最大 / 取首个"。
    C3 **oracle 标 `diagnostic=True, deployable=False`**：它需要 ground truth。
       报它的任何数字都要带「特权信息基线（**不主张**性能上界）」。

★ 本版本能做什么、不能做什么（诚实边界 —— 别把"注册了"读成"跑通了"）
    · 能：`periodic` / `threshold` / `timing` 三族是**信号无关**的
      ⇒ `simulate_schedule()` 可以直接给出 **E[age] 与发送率**，并与闭式对账。
    · 不能：`selfreport` / `conformal` / `oracle` 三族是**信号相关**的
      ⇒ 它们需要每一步的 `U`（模型自报）或真值误差，**只能**在真闭环里跑。
      `simulate_schedule()` 对这三族**直接 raise**（R14：报错优于给假数字）——
      这三族的历史结果在 `scripts/24..27`，本模块只负责**统一构建**与**统一口径**。
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np

from wmlab.eval import pairing
from wmlab.eval.conformal import envelope as conf_envelope
from wmlab.eval.oracle import z_oracle
from wmlab.eval.timing import n_tx_budget, plan_times, lookup_schedule
from wmlab.eval.tracking import (
    Schedule,
    periodic_mean_age,
    periodic_lossy_schedule,
    threshold_lossy_schedule,
    threshold_mean_age,
)

# ============================================================ 族注册表（C1 + C3）
FAMILIES = ("periodic", "threshold", "selfreport", "conformal", "oracle", "timing")

#: ★ 信号相关族 —— 需要闭环里的 `state`（模型自报 U 或真值误差）才能算触发。
#: `simulate_schedule()` 会对它们 **raise**，而不是"用空 U 跑出一个数"。
SIGNAL_DEPENDENT = frozenset({"selfreport", "conformal", "oracle"})


@dataclass(frozen=True)
class Spec:
    """一个调度族的元信息。`knob` 是该族的**主旋钮名**（批量扫描时用它做轴）。"""

    family: str
    knob: str
    deployable: bool
    needs: tuple[str, ...]          # 额外素材：state / tab / truth / n_steps ...
    note: str

    @property
    def diagnostic(self) -> bool:
        """不可部署 ⇒ 只能当诊断基线。"""
        return not self.deployable


REGISTRY: dict[str, Spec] = {
    "periodic": Spec(
        family="periodic", knob="period", deployable=True, needs=(),
        note="每 T 步发一次；E[age] = T·q/s + (T−1)/2（闭式，tracking.periodic_mean_age）",
    ),
    "threshold": Spec(
        family="threshold", knob="threshold", deployable=True, needs=(),
        note="age ≥ K 才尝试发；E[age] 闭式见 tracking.threshold_mean_age",
    ),
    "selfreport": Spec(
        family="selfreport", knob="threshold_u", deployable=True, needs=("state",),
        note="★ X38：U ≥ thr 才尝试发。U = 世界模型 σ 头自报的**未校准**累积不确定度；"
             "X38/X39/X41 实测其边际任务价值不可分辨",
    ),
    "conformal": Spec(
        family="conformal", knob="tol", deployable=True, needs=("state", "tab"),
        note="★ X42：age ≥ K 或 Ê_α(age,U) ≥ tol。有覆盖保证，但 X42 实测只把"
             "「更差」变成「持平」",
    ),
    "oracle": Spec(
        family="oracle", knob="tau", deployable=False, needs=("state", "tab", "truth_key"),
        note="★★ X43：用**真值**误差归一化后触发。**不可部署**，仅作诊断用的特权信息基线；"
             "**未证明、也不主张**它是「误差类触发量」这一族的性能上界",
    ),
    "timing": Spec(
        family="timing", knob="param", deployable=True, needs=("n_steps", "n_tx"),
        note="★ X44：固定预算、只动发送时刻（phase / jitter / random）。"
             "预算相等由 timing.plan_times 的硬断言保证",
    ),
}


def spec_of(family: str) -> Spec:
    if family not in REGISTRY:
        raise ValueError(f"未知族 {family!r}，可选 {FAMILIES}")
    return REGISTRY[family]


# ============================================================ 构建（C1）
def _periodic(*, loss_prob: float, period: int, **_) -> Schedule:
    return periodic_lossy_schedule(int(period), float(loss_prob))


def _threshold(*, loss_prob: float, threshold: int, **_) -> Schedule:
    return threshold_lossy_schedule(int(threshold), float(loss_prob))


def _selfreport(*, loss_prob: float, threshold_u: float, state: dict, **_) -> Schedule:
    """U ≥ thr 才尝试发。`state["U"]` 由闭环 runner 每步写入。"""
    thr = float(threshold_u)
    p = float(min(max(loss_prob, 0.0), 1.0))

    def _f(t: int, rng: np.random.Generator) -> bool:
        if float(state.get("U", 0.0)) < thr:
            return False
        return bool(rng.random() >= p)

    def _reset(rng: np.random.Generator | None = None) -> None:
        state["U"] = 0.0

    _f.reset = _reset                                    # type: ignore[attr-defined]
    _f.__name__ = f"selfreport(thr={thr:.4f},p={p})"
    return _f


def _conformal(*, loss_prob: float, tol: float, K: int, state: dict, tab: dict,
               **_) -> Schedule:
    """age ≥ K 或 Ê_α(age, U) ≥ tol。规则结构与 `_selfreport` 的混合版一致（只换统计量）。"""
    K_ = int(K)
    t_ = float(tol)
    p = float(min(max(loss_prob, 0.0), 1.0))
    n_h = int(tab["H"])

    def _f(t: int, rng: np.random.Generator) -> bool:
        age = int(state.get("age", 0))
        if age < K_:
            h = min(max(age, 1), n_h)
            e_hat = float(conf_envelope(tab, np.array([h]),
                                        np.array([float(state.get("U", 0.0))]))[0])
            if e_hat < t_:
                return False
        return bool(rng.random() >= p)

    def _reset(rng: np.random.Generator | None = None) -> None:
        state["U"] = 0.0
        state["age"] = 0

    _f.reset = _reset                                    # type: ignore[attr-defined]
    _f.__name__ = f"conformal(K={K_},tol={t_:g},p={p})"
    return _f


def _oracle(*, loss_prob: float, tau: float, K: int, state: dict, tab: dict,
            truth_key: str, **_) -> Schedule:
    """★ 特权信息基线（C3）：age ≥ K 或 z_oracle(真值误差) ≥ τ。"""
    K_ = int(K)
    t_ = float(tau)
    p = float(min(max(loss_prob, 0.0), 1.0))
    n_h = int(tab["H"])

    def _f(t: int, rng: np.random.Generator) -> bool:
        age = int(state.get("age", 0))
        if age < K_:
            h = min(max(age, 1), n_h)
            z = z_oracle(float(state.get(truth_key, 0.0)), h, tab)   # 非有限值内部 raise
            if z < t_:
                return False
        return bool(rng.random() >= p)

    def _reset(rng: np.random.Generator | None = None) -> None:
        state["U"] = 0.0
        state["age"] = 0
        if truth_key in state:
            state[truth_key] = 0.0

    _f.reset = _reset                                    # type: ignore[attr-defined]
    _f.__name__ = f"oracle(K={K_},tau={t_:.3f},key={truth_key})"
    return _f


def _timing(*, mode: str, period: int, param: int, n_steps: int,
            n_tx: int | None = None, pattern_seed: int = 0, **_) -> Schedule:
    N = int(n_tx) if n_tx is not None else n_tx_budget(int(period), int(n_steps))
    times = plan_times(mode, int(period), int(param), int(n_steps), N,
                       pattern_seed=int(pattern_seed))
    return lookup_schedule(times, f"timing({mode},T={period},x={param})")


_BUILDERS: dict[str, Callable[..., Schedule]] = {
    "periodic": _periodic,
    "threshold": _threshold,
    "selfreport": _selfreport,
    "conformal": _conformal,
    "oracle": _oracle,
    "timing": _timing,
}


def build(family: str, *, loss_prob: float, **knobs) -> Schedule:
    """按族名构建调度器（C1：返回 `(t, rng) -> bool`，可能带 `.reset(rng)`）。

    Args:
        family: `FAMILIES` 之一
        loss_prob: 逐次尝试的丢包率（`selfreport`/`conformal`/`oracle` 也用它决定"尝试"是否成功）
        **knobs: 该族旋钮（见 `REGISTRY[family].knob`）+ 必需素材（见 `.needs`）；
                 `timing` 族额外支持 `n_steps` / `n_tx` / `pattern_seed`

    Raises:
        ValueError: 未知族，或缺少该族 `.needs` 里声明的素材（**不猜默认值** —— 缺 `state`
            却给一个空 dict，会静默跑出"永不触发"这种**看起来合理**的结果）。
    """
    sp = spec_of(family)
    missing = [k for k in sp.needs if k not in knobs]
    if missing:
        raise ValueError(
            f"族 {family!r} 缺少必需素材 {missing}（见 REGISTRY[{family!r}].needs）；"
            "★ 不允许用默认值代替：缺 state 时静默构建会跑出「永不触发」的假结果")
    sched = _BUILDERS[family](loss_prob=float(loss_prob), **knobs)
    # ★ 打上族标记：`simulate_schedule` 靠它拒绝信号相关族（比按 __name__ 字符串匹配可靠）
    sched._wmlab_family = family                          # type: ignore[attr-defined]
    return sched


# ============================================================ 纯调度空跑（信号无关三族）
def simulate_schedule(schedule: Schedule, n_steps: int, seed: int = 0,
                      reset: bool = True) -> dict:
    """把调度器喂进一个**只有调度器、没有环境/模型**的空跑，返回年龄统计。

    ★★ 送达语义（这一条曾经把我坑成"重复扣损"）：
      `Schedule` 的返回值**已经是"本步是否送达"** ——
      `periodic_lossy_schedule` / `threshold_lossy_schedule` / 三个触发族
      **内部自己**消耗随机数判丢包。⇒ 本函数**不得**再叠一层 `rng.random() >= p`，
      否则丢包被扣两次，`p=0.7` 会被静默算成 `p=0.91`，而结果"看起来完全合理"。
      ⇒ 因此本函数**不接受 `loss_prob` 参数**（参数不存在就没法传错）。

    ★ 只对**信号无关**族有意义（`periodic` / `threshold` / `timing`）：
      信号相关族（`selfreport` / `conformal` / `oracle`）的 `(t, rng)` 要读闭环写进
      `state` 的 `U` / 真值误差；空跑里那些量恒为 0 ⇒ 只会得到**恒定不触发**
      ⇒ 一个"看起来像结论"的假数（X39 的全局阈值退化、X40 的第一版都是这么来的）。
      ⇒ 本函数对它们**直接 raise**（R14）。

    Returns:
        {"tx_rate", "mean_age", "n_tx", "n_steps", "mean_age_se", "std_age"}
        ★ `mean_age_se` 用**批量均值法**（`n_batches` 批）估计，不是学生公式：
          age 是**重尾 + 强自相关**（再生周期长）的量，逐点样本方差的 `s/√n`
          会**严重低估**标准误（p=0.7、T=16 时低估一个数量级以上）。
          批量均值法不需要任何解析式，且能让调用方用「4σ」这类可辩护的判据，
          而不是拍一个百分比容差（铁律 17）。
    """
    fam = str(getattr(schedule, "_wmlab_family", ""))
    if fam in SIGNAL_DEPENDENT:
        raise ValueError(
            f"{fam} 族是**信号相关族**，不能在空跑里评估 —— 它需要闭环写入的 state"
            "（U / 真值误差）；空跑会静默给出「永不触发」的假结果。"
            "历史结果见 scripts/24..27；本函数只服务 periodic / threshold / timing。")
    rng = np.random.default_rng(int(seed))
    if reset and hasattr(schedule, "reset"):
        schedule.reset(rng)                               # type: ignore[attr-defined]
    n = int(n_steps)
    n_b = max(2, min(20, n // 50))
    batch_len = n // n_b
    batch_sum = np.zeros(n_b, dtype=np.float64)
    batch_cnt = np.zeros(n_b, dtype=np.int64)
    age = 0
    tot = 0.0
    sq = 0.0
    n_tx = 0
    for t in range(1, n + 1):
        # ★ 注意 t 从 1 起：run_closed_loop_control 里 `t += 1` 在 `schedule(t, rng)` 之前
        #   ⇒ 调度被查询的时刻是 1…L，**t=0 永不出现**（X44 踩过：写成 0 会静默丢首点）
        if bool(schedule(t, rng)):
            n_tx += 1
            age = 0
        else:
            age += 1
        tot += age
        sq += float(age) * float(age)
        b = (t - 1) // batch_len
        if b < n_b:
            batch_sum[b] += age
            batch_cnt[b] += 1
    mean_age = tot / float(n)
    var_age = max(0.0, sq / float(n) - mean_age * mean_age)
    bm = batch_sum[batch_cnt > 0] / batch_cnt[batch_cnt > 0].astype(np.float64)
    se = float(bm.std(ddof=1) / np.sqrt(bm.size)) if bm.size >= 2 else float("nan")
    return {
        "tx_rate": n_tx / float(n),
        "mean_age": mean_age,
        "n_tx": int(n_tx),
        "n_steps": n,
        "mean_age_se": se,
        "std_age": float(np.sqrt(var_age)),
    }


# ============================================================ 结果 schema
SCHEMA_VERSION = "scheduling-v0.1"


@dataclass
class Row:
    """一条 (族 × 旋钮 × 丢包率 × 种子) 的结果。**统一 schema**，便于批量导出与配对。"""

    family: str
    knob_axis: str          # 沿哪根轴扫的（"period" / "threshold" / "thr" / "tau" / "param"）
    knob_value: float
    loss_prob: float
    seed: int
    tx_rate: float
    mean_age: float
    metric: str = ""        # 任务/代理指标名；空 = 本行只有调度统计
    metric_value: float | None = None
    n_eval: int = 0         # 参与该指标的样本数（集数 / 档数）
    diverged: int = 0       # ★ 发散样本数：**不许静默丢弃**，也不许混进均值
    deployable: bool = True
    diagnostic: bool = False
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def export_json(rows: list[Row], path: str, meta: dict | None = None) -> str:
    """批量导出。`meta` 里记口径（seed 数、口径名、代码版本自述）。"""
    payload = {
        "schema": SCHEMA_VERSION,
        "meta": dict(meta or {}),
        "rows": [r.to_dict() for r in rows],
    }
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


# ============================================================ 等预算配对（C2）
def equal_budget_pair(rows_a: list[Row], rows_b: list[Row], metric: str,
                      n_grid: int = 9, min_range: float = 1.5,
                      min_pts: int = 3) -> dict:
    """两族在**公共发送率窗口**上的等预算比较（C2）。

    ★ 硬要求（对齐 `pairing` 的设计）：
      · 两边都只在 `metric_value is not None` **且** `diverged == 0` 的行上取曲线；
      · 窗口至少 `min_pts` 个可比点、跨度 `hi/lo ≥ min_range`；
      · 不满足 ⇒ 抛 `WindowNotEvaluable`，**调用方必须把它打印出来**，
        不许静默 `continue`（X40 的"9/9 更优"就是这么来的）。

    ★ 第三条守卫（比值专属，比 `pairing` 的两条更靠后）：**指标必须严格为正**。
      `mean_age` 在最大预算端会退化为 0（`K=0`/`T=1` ⇒ 每步送达 ⇒ age 恒 0），
      比值在那一端**无定义** ⇒ 网格会被**收缩**到两族都 >0 的区域，
      并报出 `n_grid_used / n_grid_raw`（**收缩这件事必须被看见**，不许静默丢点）；
      收缩后点数 < `min_pts` ⇒ 判「不可评估」。

    Returns:
        {"lo","hi","grid","lo_raw","hi_raw","a":[...],"b":[...],
         "ratio_median": float, "n_grid_used", "n_grid_raw", "n_a","n_b","metric"}
        `lo/hi/grid` 是**收缩后**的窗口；`lo_raw/hi_raw` 是 `pairing.overlap_window`
        给出的原始窗口。`a`/`b` 是在同一 `grid` 上**线性插值**后的指标值
        （两族各自已升序按 tx_rate 排）。
    """
    def curve(rows: list[Row], name: str) -> tuple[list[float], list[float]]:
        rs = [r for r in rows
              if r.metric == metric and r.metric_value is not None and r.diverged == 0]
        rs.sort(key=lambda r: r.tx_rate)
        xs = [float(r.tx_rate) for r in rs]
        ys = [float(r.metric_value) for r in rs]
        if len(xs) < min_pts:
            raise pairing.WindowNotEvaluable(
                f"族 {name} 在 metric={metric!r} 上只有 {len(xs)} 个可用点（< {min_pts}）")
        return xs, ys

    xs_a, ys_a = curve(rows_a, "a")
    xs_b, ys_b = curve(rows_b, "b")
    lo, hi = pairing.overlap_window({"a": xs_a, "b": xs_b},
                                    min_range=min_range, min_pts=min_pts)
    grid = pairing.expo_grid(lo, hi, n=n_grid)
    va = [float(np.interp(g, xs_a, ys_a)) for g in grid]
    vb = [float(np.interp(g, xs_b, ys_b)) for g in grid]
    # ★★ 比值只对**严格为正**的指标有定义（真踩过）：`mean_age` 在**最大预算端退化为 0** ——
    #    `threshold(K=0)` / `periodic(T=1)` ⇒ 每步都送达 ⇒ age 恒 0。
    #    实测症状：直接把 0 放进比值 ⇒ 末点是 `x/0`，而前面 8 个点"看着正常"
    #    ⇒ 报出来的中位数会被末端那个无定义点污染（或直接抛 ZeroDivisionError）。
    #    ⇒ 处理方式：把网格**收缩**到两族都 >0 的区域，并**报出收缩后的点数**；
    #      点数不足就判"不可评估"—— **不许**静默丢掉 0 点后照报中位（那会隐藏"指标在这一端退化"）。
    keep = [i for i in range(len(grid)) if va[i] > 0.0 and vb[i] > 0.0]
    if len(keep) < min_pts:
        raise pairing.WindowNotEvaluable(
            f"指标 {metric!r} 在公共窗口上存在 ≤0 的取值 ⇒ 比值无定义。"
            f"两族都 >0 的网格点只有 {len(keep)} 个（< {min_pts}）。"
            f"常见成因：该指标在**最大预算端退化到 0**（K=0 / T=1 ⇒ 每步送达 ⇒ age 恒 0）。"
            f" va={[round(v, 6) for v in va]} vb={[round(v, 6) for v in vb]}")
    g2 = [grid[i] for i in keep]
    va2 = [va[i] for i in keep]
    vb2 = [vb[i] for i in keep]
    ratios = [a / b for a, b in zip(va2, vb2)]
    return {
        "lo": g2[0], "hi": g2[-1], "grid": g2,
        "lo_raw": lo, "hi_raw": hi,
        "a": va2, "b": vb2,
        "ratio_median": float(np.median(ratios)),
        "n_grid_used": len(g2), "n_grid_raw": len(grid),
        "n_a": len(xs_a), "n_b": len(xs_b), "metric": metric,
    }


# ============================================================ 自检（R12：拿已知答案代验）
def self_check(verbose: bool = True, n: int = 100000, seed: int = 12345) -> dict:
    """把三族信号无关调度器的空跑结果与**闭式**对账。

    ★ 判据用「**4×批量均值标准误**」而不是固定百分比（铁律 17：容差不能拍）。
      `self_check` 返回 `{key: (实测, 参考, 差, 标准误)}`；
      标准误为 `nan` 表示该条是**构造上逐位相等**的（如预算相等），差必须**严格为 0**。
    """
    out: dict = {}
    n, seed = int(n), int(seed)
    for p in (0.0, 0.3, 0.7):
        for T in (1, 4, 10):
            r = simulate_schedule(build("periodic", loss_prob=p, period=T), n, seed)
            ref = periodic_mean_age(p, T)
            out[f"periodic(p={p},T={T})"] = (r["mean_age"], ref,
                                             abs(r["mean_age"] - ref),
                                             r["mean_age_se"])
        for K in (1, 3, 8):
            r = simulate_schedule(build("threshold", loss_prob=p, threshold=K), n, seed)
            ref = threshold_mean_age(K, p)
            out[f"threshold(p={p},K={K})"] = (r["mean_age"], ref,
                                              abs(r["mean_age"] - ref),
                                              r["mean_age_se"])
    # --- timing：预算逐位相等（构造保证 ⇒ 标准误记 nan，差必须严格 0）+ 无丢包年龄 = (T−1)/2 ---
    T, p0 = 4, 0.0
    N = n_tx_budget(T, n)
    n_tx_all, ages, ses = [], [], []
    for mode, param in (("phase", 0), ("phase", 1), ("phase", 3),
                        ("jitter", 0), ("jitter", 1), ("random", 0)):
        s = build("timing", loss_prob=p0, mode=mode, period=T, param=param,
                  n_steps=n, n_tx=N)
        r = simulate_schedule(s, n, seed)
        n_tx_all.append(r["n_tx"])
        ages.append(r["mean_age"])
        ses.append(r["mean_age_se"])
    out["timing_budget_exact_equal"] = (float(min(n_tx_all)), float(max(n_tx_all)),
                                        float(max(n_tx_all) - min(n_tx_all)),
                                        float("nan"))
    out["timing_tx_rate_vs_N/n"] = (n_tx_all[0] / float(n), N / float(n),
                                    abs(n_tx_all[0] / float(n) - N / float(n)),
                                    float("nan"))
    out["timing_phase_mean_age_vs_(T-1)/2"] = (ages[0], (T - 1) / 2.0,
                                               abs(ages[0] - (T - 1) / 2.0),
                                               ses[0])
    if verbose:
        for k, v in out.items():
            print(f"  {k}: 实测={v[0]:.6f} 参考={v[1]:.6f} 差={v[2]:.3e} se={v[3]:.2e}")
    return out


def check_tolerance(diff: float, se: float, k_sigma: float = 4.0) -> float:
    """把 (差, 标准误) 折算成**判据比值** `diff / tol`；≤1 即通过。

    · `se` 有限 ⇒ `tol = k_sigma · se`（默认 4σ，对应 ~99.99%）
    · `se` 为 nan（构造上逐位相等）⇒ `tol = 0`，要求 `diff` 严格为 0
    """
    if se != se:                                          # nan
        return 0.0 if diff == 0.0 else float("inf")
    return diff / max(k_sigma * se, 1e-12)
