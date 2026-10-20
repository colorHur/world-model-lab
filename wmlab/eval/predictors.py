# -*- coding: utf-8 -*-
"""接收端预测器（T22 / P1-2，2026-10-19）。

为什么单独一个模块
------------------
T12–T21 的全部数字都建立在**一件事**上：接收端的估计观测 `est[i]` 只在**送达**时被替换，
其余时刻保持上一次送达值（ZOH，零阶保持）。于是"谁更该被服务"恒等于"谁的观测更旧"。
把 `est` 换成**预测**之后，同一条调度规则面对的输入就变了 —— 这是"预测补偿与调度协同"
这个主问题的第一个直接实验。

信息集边界（**这是本模块的核心，不是实现细节**）
------------------------------------------------
每个预测器只允许用下面这张表里的东西，越界就是"偷真值"：

    ZOH              最后一次送达的观测
    Linear           最近两次送达的观测（差分估速度）
    ARX              最近两次送达的观测 + 两次之间实际施加的动作（只学 gap==1 的一步映射）
    MultiLag         最近两次送达的观测 + 两次之间**全部**实际施加的动作（按滞后阶 g 分模型）
    SharedLag        最近两次送达的观测 + 两次之间**全部**实际施加的动作（**所有 g 共用一套 (A,B_j,c)**）
    Gated            最近两次送达的观测，**按 `age` 二选一**：`age>=θ` 用 `last_obs`，否则线性外推
    TrueDyn          `deepcopy` 的 env + **自己实际施加过的动作**（诊断特权，只作上界）

`LinearTargetOnly` / `LinearSelfOnly` 是 `Linear` 的**维度切片**变体（机制分解，
不是可部署方法）；它们与上面四个共用同一张信息集表。

三者**都不读 `View` / `View.privileged`**：预测器拿不到 `privileged`，这是结构性保证
（`predict()` 的签名里根本没有 view），不是"我们保证不读"。`scripts/53` 的 S3 另外用
"`diagnostic=True/False` 下轨迹逐位相同"再验一次。

时序约定（与 `multiuser.run_multiuser` 逐位对齐；差一步就会让"预测 d 步"变成"d−1 步"）
------------------------------------------------------------------------------------
决策时点 `t`、`age = d` ⇒ 最后一次送达发生在第 `t-1-d` 步，手里的观测是 `x_{t-1-d}`，
真值是 `x_{t-1}` ⇒ 需要补 `d` 步；而 `actions_since` 恰好有 `d` 项
（第 `t-d, …, t-1` 步实际施加的动作）。第 `t` 步的动作在预测时**还没产生**，不得在缓冲里。

三条实现纪律
------------
1. **送达即重置**：`observe_delivery()` 里清空动作缓冲的语义由调用方负责（runner 负责清）；
   本模块只负责"快照换成新观测"。
2. **不得触碰主 env**：`TrueDyn` 只对 `deepcopy` 出来的影子前推；影子有**自己的** rng，
   主轨迹与主 rng 一位都不变（`scripts/53` 的 S7 用"影子跑但丢输出"做阴性对照）。
3. **不许静默降级**：缺 env / `actions_since` 太短 / `age < 0` 一律 `ValueError`
   （同 `multiuser.MissingStateError` 的理由：静默降级会跑出一个"看起来正常"的数字）。
"""

from __future__ import annotations

import copy
from typing import Sequence

import numpy as np

__all__ = ["PredictorBase", "ZOH", "Linear", "TrueDyn", "ARX", "MultiLag", "HybridLag",
           "SharedLag", "Gated", "LinearTargetOnly", "LinearSelfOnly", "make_predictor"]


def _linear_extrapolate(st: dict, age: int) -> np.ndarray:
    """局部线性外推 `x̂ = x_last + age·v`，`v = (x_last − x_prev)/gap_prev`。

    ★ 抽成模块级函数（T23-a）是为了让 `Linear` 与 `ARX` 的**退化路径**共用同一份实现：
      预注册 §5-A3 要求"样本不足时 ARX 与 Linear 逐位相同"，两份实现各写一遍迟早会分叉，
      而分叉之后"A3 PASS"就变成了"A3 恰好在那几个 bit 上相同"。
    """
    prev, gap = st.get("prev_obs"), int(st.get("gap_prev", 0))
    if prev is None or gap <= 0 or age == 0:
        return st["last_obs"]
    v = (np.asarray(st["last_obs"], dtype=np.float64) - prev) / float(gap)
    return st["last_obs"] + float(age) * v


class PredictorBase:
    """预测器基类：管每用户的历史（`last_obs` / `prev_obs` / `gap_prev`），子类只写 `_predict`。

    ★ 为什么历史放在预测器里而不是 runner 里：runner 已经维护 `last_obs`，但 `prev_obs` 与
      "两次送达之间隔了几步"（`gap_prev`）只有预测器要用。放在这里可以让 runner 只有
      两行接线，也避免"runner 里多存一份、两边不同步"。

    ★ `gap_prev` 的定义：两次送达相隔的**步数**（不是帧数）。送达在第 τ1、τ2 步
      ⇒ `gap_prev = τ2 − τ1 = age_before(τ2) + 1`（`age_before` 是决策时点的年龄，
      送达发生在该步**之后**）。用它把观测差分换算成"每步速度"，否则间隔一变速度就差一个因子。
    """

    name: str = "base"
    #: 是否需要在送达时拿到 env 引用（只有 `TrueDyn` 为 True；其余为 False ⇒ 结构上不可能偷真值）
    consumes_env: bool = False

    def __init__(self) -> None:
        self._st: dict[int, dict] = {}

    # ------------------------------------------------------------ 生命周期
    def reset_episode(self) -> None:
        """每集开头清空全部用户状态（不跨集泄漏，等价于 I6 的"跨集不继承"）。"""
        self._st = {}

    def observe_delivery(self, i: int, obs, *, age_before: int, env=None, action=None,
                         action_seq=None) -> None:
        """一帧送达：把历史前移一格，并把新观测记为 `last_obs`。

        Args:
            i: 用户序号。
            obs: 本步送达的观测（= 该步 `env.step()` 返回的 `nxt`）。
            age_before: **决策时点**的年龄（该步尚未结算），用来推 `gap_prev`。
            env: 仅 `consumes_env=True` 的预测器需要（快照源）。
            action: 本步**实际施加**的动作（T23-a 新增，可选）。只有 `ARX` 用它学一步映射；
                `ZOH`/`Linear`/`TrueDyn` 忽略它。**必须是 `act()` 的返回值**（即真正喂给
                `env.step` 的那个），不是命令值 —— 下行有损时这两者才会不同（当前下行理想）。
            action_seq: **两次送达之间**自己实际施加的**全部**动作（T23-a-2 新增，可选）。
                长度必须 == `gap_prev`；由 runner 在清空 `actions_since` 之前整体传入。
                只有 `MultiLag` 用它（多步模型要整段动作）；给 `None` 时退化为 `[action]`。
        """
        i = int(i)
        st = self._st.setdefault(i, {})
        prev = st.get("last_obs")
        st["prev_obs"] = None if prev is None else np.asarray(prev, dtype=np.float64).copy()
        st["gap_prev"] = (int(age_before) + 1) if prev is not None else 0
        st["last_action"] = (None if action is None
                             else np.asarray(action, dtype=np.float64).reshape(-1).copy())
        if action_seq is None:
            st["last_action_seq"] = (None if st["last_action"] is None
                                     else [st["last_action"].copy()])
        else:
            st["last_action_seq"] = [np.asarray(a, dtype=np.float64).reshape(-1).copy()
                                     for a in action_seq]
        st["last_obs"] = np.asarray(obs, dtype=np.float32).reshape(-1).copy()
        st["n_deliveries"] = int(st.get("n_deliveries", 0)) + 1
        if self.consumes_env and env is None:
            raise ValueError(f"{self.name}：consumes_env=True 但 observe_delivery 收到 env=None"
                             f"（没有快照源就无法前推；不许静默退化成 ZOH）")
        self._on_delivery(i, st, env)

    # ------------------------------------------------------------ 预测
    def predict(self, i: int, *, age: int, actions_since: Sequence) -> np.ndarray:
        """返回该用户在当前决策时点的**估计观测**（形状同 `last_obs`）。"""
        i = int(i)
        age = int(age)
        if age < 0:
            raise ValueError(f"{self.name}：age 必须 ≥ 0，收到 {age}")
        st = self._st.get(i)
        if st is None or "last_obs" not in st:
            raise ValueError(f"{self.name}：用户 {i} 还没有任何送达观测，无法预测"
                             f"（先调 observe_delivery；不许拿 0 向量顶替）")
        acts = list(actions_since)
        if len(acts) != age:
            # ★ 长度不等 = 时序接错（多/少一步），必须炸；否则会静默地"预测错步数"
            raise ValueError(f"{self.name}：actions_since 长度 {len(acts)} ≠ age {age}"
                             f"（时序接错会让预测差一步，且结果看起来完全正常）")
        out = self._predict(i, st, age=age, actions_since=acts)
        out = np.asarray(out, dtype=np.float32).reshape(-1)
        if not np.all(np.isfinite(out)):
            raise ValueError(f"{self.name}：预测输出含 NaN/inf：{out.tolist()}")
        return out

    # ------------------------------------------------------------ 子类钩子
    def _on_delivery(self, i: int, st: dict, env) -> None:  # pragma: no cover - 默认空
        pass

    def _predict(self, i: int, st: dict, *, age: int, actions_since: list) -> np.ndarray:
        raise NotImplementedError

    # ------------------------------------------------------------ 自检辅助
    def state_refs(self) -> list:
        """摊平全部内部状态值（`scripts/53` 用它验证"状态里没有 env 对象"）。"""
        out: list = []
        for st in self._st.values():
            out.extend(st.values())
        return out

    def describe(self) -> dict:
        """机器可读的自述（T23-a：`ARX` 必须能区分"在学"与"在退"）。"""
        return {"name": self.name, "consumes_env": bool(self.consumes_env),
                "n_users": len(self._st)}


class ZOH(PredictorBase):
    """零阶保持 = **现状**（`83340e1` 的行为）。它的存在意义是回归红线。

    ★ 它必须与 `predictor=None` **逐位相同**；任何不同都说明接线改变了既有轨迹。
    """

    name = "zoh"

    def _predict(self, i: int, st: dict, *, age: int, actions_since: list) -> np.ndarray:
        return st["last_obs"]


class Linear(PredictorBase):
    """局部线性外推：`x̂ = x_last + age · v`，`v = (x_last − x_prev) / gap_prev`。

    ★ 只用**观测**，**不用动作**（这是刻意的，也是本轮明确登记的结构性弱点）：
      稳定控制器会持续反向修正，"按惯性外推"必然过冲。因此若 T23 出现 `Linear` 劣于 `ZOH`，
      只能说明"这个弱预测器不行"，**不能**说明"预测补偿没有空间"（预注册 §5-R2）。

    ★ 观测都是**全状态**（含速度分量）⇒ 差分得到的 `v` 近似"每步位移"，对本系统是对的量纲。
      换到部分观测的场景（只有位置）时，本类需要重新定义 —— 见模块头的"信息集"表，
      它只承诺"最近两次送达观测"，不承诺任何任务知识。
    """

    name = "linear"

    def _predict(self, i: int, st: dict, *, age: int, actions_since: list) -> np.ndarray:
        return _linear_extrapolate(st, age)


class ARX(PredictorBase):
    """**动作条件一步映射**（T23-a）：`x_{k+1} ≈ A x_k + B a_k + c`，递推最小二乘 + 相对 ridge。

    为什么需要它（预注册 §1）
    ------------------------
    T22 只有 `Linear`，而 `Linear` **只用观测、不用动作**。稳定控制器会持续反向修正，
    按惯性外推必然过冲 ⇒ 若 T23-b 里 `Linear` 输给 `ZOH`，**无法区分**"这个弱预测器不行"
    与"预测补偿没有空间"。`ARX` 把前一种解释排除掉：它用**自己实际施加的动作**，
    且只需"已送达观测 + 自己施加的动作"，不需要任何任务知识。

    样本口径（预注册 §2，写死）
    --------------------------
    **只学 `gap == 1` 的送达对** `(x_{k}, a_{k+1}, x_{k+1})`。`gap > 1` 时中间状态未观测，
    拿一步模型硬套等于把多步误差混进一步拟合；这类对**只计数、不入拟合**（`describe()` 可查）。

    ★ 结构性限制（写在跑之前）：两用户轮流服务时，同一用户的送达间隔常是 2 步 ⇒ `gap==1`
      样本可能很少，`ARX` 会长期停在退化态。这不是缺陷，是"只用一步对"的口径代价，
      必须在结果里量化（`scripts/54` 的 A9），并在 T23-b 的解释里带上。
    """

    name = "arx"
    consumes_env = False          # ★ 结构性：拿不到 env ⇒ 不可能偷真值

    def __init__(self, ridge_rel: float = 1e-6, min_samples_factor: int = 2) -> None:
        super().__init__()
        self.ridge_rel = float(ridge_rel)
        self.min_samples_factor = int(min_samples_factor)
        self._fit: dict[int, dict] = {}
        self._counts: dict[int, dict] = {}

    def reset_episode(self) -> None:
        super().reset_episode()
        self._fit = {}
        self._counts = {}

    # ------------------------------------------------------------ 学
    def _model(self, i: int, *, dim: int, act_dim: int) -> dict:
        m = self._fit.get(i)
        if m is not None and (m["dim"] != dim or m["act_dim"] != act_dim):
            raise ValueError(f"ARX：用户 {i} 的维度中途变了（{m['dim']}/{m['act_dim']} → "
                             f"{dim}/{act_dim}）—— 只可能是接线错误")
        if m is None:
            F = dim + act_dim + 1
            m = {"dim": dim, "act_dim": act_dim, "F": F, "n": 0,
                 "R": np.zeros((F, F), dtype=np.float64),
                 "b": np.zeros((F, dim), dtype=np.float64), "W": None}
            self._fit[i] = m
        return m

    def _on_delivery(self, i: int, st: dict, env) -> None:
        cnt = self._counts.setdefault(i, {"gap1": 0, "gap_gt1": 0, "no_action": 0})
        prev, act = st.get("prev_obs"), st.get("last_action")
        gap = int(st.get("gap_prev", 0))
        if prev is None:
            return                                  # 首个观测：没有"上一拍"，学不了
        if gap != 1:
            cnt["gap_gt1"] += 1
            return
        if act is None:
            cnt["no_action"] += 1
            return
        y = np.asarray(st["last_obs"], dtype=np.float64)
        x = np.asarray(prev, dtype=np.float64)
        if x.shape != y.shape:
            raise ValueError(f"ARX：上一拍观测形状 {x.shape} ≠ 送达观测 {y.shape}")
        m = self._model(i, dim=int(y.size), act_dim=int(act.size))
        z = np.concatenate([x, act, np.ones(1, dtype=np.float64)])
        m["R"] += np.outer(z, z)
        m["b"] += np.outer(z, y)
        m["n"] += 1
        cnt["gap1"] += 1
        self._solve(m)

    def _solve(self, m: dict) -> None:
        """相对 ridge：`(R + λ·diag(R) + εI) W = b`。

        ★ 用**相对** ridge 而非固定值：UAV 的位置量级 ~10 m、速度 ~m/s、动作 ~3 m/s²，
          固定 ridge 会系统性偏向大尺度那一维（等价于给它更强的正则），换个场景就得重调。
        """
        F = int(m["F"])
        d = np.diag(m["R"]).copy()
        scale = float(d.max()) if F else 0.0
        if not np.isfinite(scale) or scale <= 0.0:
            return
        reg = m["R"] + self.ridge_rel * np.diag(d) + (1e-9 * scale) * np.eye(F)
        m["W"] = np.linalg.solve(reg, m["b"])

    # ------------------------------------------------------------ 预测
    def _need(self, dim: int) -> int:
        return self.min_samples_factor * int(dim)

    def _predict(self, i: int, st: dict, *, age: int, actions_since: list) -> np.ndarray:
        dim = int(np.size(st["last_obs"]))
        if age == 0:
            return st["last_obs"]
        m = self._fit.get(i)
        if m is None or m["W"] is None or m["n"] < self._need(dim):
            return _linear_extrapolate(st, age)      # ★ 显式退化（`describe()` 里可见）
        x = np.asarray(st["last_obs"], dtype=np.float64).copy()
        for k, a in enumerate(actions_since):
            av = np.asarray(a, dtype=np.float64).reshape(-1)
            if av.size != m["act_dim"]:
                raise ValueError(f"ARX：第 {k} 个动作维度 {av.size} ≠ 学到的 {m['act_dim']}")
            x = np.concatenate([x, av, np.ones(1, dtype=np.float64)]) @ m["W"]
            if not np.all(np.isfinite(x)):
                raise ValueError(f"ARX：迭代第 {k+1} 步发散（出现 NaN/inf）")
        return x

    def describe(self) -> dict:
        users = {}
        for i, m in sorted(self._fit.items()):
            need = self._need(m["dim"])
            users[str(i)] = {"dim": m["dim"], "act_dim": m["act_dim"], "n_samples": m["n"],
                             "n_needed": need, "degraded": bool(m["n"] < need),
                             "fitted": m["W"] is not None,
                             **self._counts.get(i, {})}
        for i, c in sorted(self._counts.items()):
            users.setdefault(str(i), {"dim": None, "act_dim": None, "n_samples": 0,
                                      "n_needed": None, "degraded": True,
                                      "fitted": False, **c})
        return {"name": self.name, "consumes_env": False, "ridge_rel": self.ridge_rel,
                "min_samples_factor": self.min_samples_factor,
                "users": users,
                "degraded_users": sorted(int(k) for k, v in users.items() if v["degraded"]),
                "total_gap1": int(sum(v.get("gap1", 0) for v in users.values())),
                "total_gap_gt1": int(sum(v.get("gap_gt1", 0) for v in users.values()))}


#: 观测里"外部目标"占的维度区间（**读实现核对过**：`wmlab/envs/uav_track.py:_get_obs`
#: 返回 `concat([uav_p(2), uav_v(2), tgt_p(2)])`）。表里没有的环境 ⇒ 全部维度视为"自身"。
_TARGET_OBS_DIMS: dict[int, tuple[int, int]] = {6: (4, 6)}


class MultiLag(PredictorBase):
    """**分滞后阶线性模型组**（T23-a-2）：对每个 `g` 一套 `x_{k+g} ≈ A_g x_k + B_g[a_{k+1};…;a_{k+g}] + c_g`。

    为什么不是 `ARX` 的迭代前推（T23-a 的事实，`93aa45e`）
    ------------------------------------------------------
    `ARX` 只学 `gap==1` 的送达对，而 1 帧/步 + `N=2` ⇒ 每用户服务率 ≤ 0.5 ⇒ 间隔 ≥ 2
    ⇒ 两用户都存活时 `gap==1` 占 **1.03%** ⇒ 一步映射**没有数据**、全程退化 ≡ `Linear`。
    这里按 `age` 直接查对应阶的模型，把 `gap=2,3`（实测有样本）用起来。

    口径（预注册 `2026-10-19-T23a2-分滞后阶模型-预注册-v0.1.md` §2，写死）
    ------------------------------------------------------------------
    1. 样本 = `gap_prev == g` 的送达对，`g ∈ {1..max_gap}`；`gap > max_gap` 只计数。
    2. **不拼装**：`age = d` 只用 `M_d`；没有 `M_d`/样本不足 ⇒ 整步退化到 `Linear`（逐位相同）。
       链式拼装（5=2+2+1）误差无界，且会把"模型质量"与"拼装规则"混成一个数。
    3. 启用门槛 `n_samples ≥ F_g`（`F_g = dim + g·act_dim + 1`）：秩亏时解完全由 ridge 先验决定，
       那不是"学到的模型"。`n < 2·F_g` 另标 `marginal`（样本刚过线，只作报告，不改门槛）。
    4. **逐 episode 重置**（同 `ARX`）：跨集累积等于"前几集免费拿数据"，收益无法归因到在线可学。
    5. 动作窗口 `action_seq` 必须由 runner 整体传入（长度 == `gap_prev`）；长度不符**计数并跳过**，
       由 `scripts/55` 的 B6 判死（#49：两边对账先问起点终点）。

    ★ 不是"世界模型"，是**线性辨识基线**（v2 §三-P2 才谈学习模型）。它就是"简单线性/局部动力学预测"。
    """

    name = "mlag"
    consumes_env = False          # ★ 结构性：签名里没有 env/view ⇒ 不可能偷真值

    def __init__(self, max_gap: int = 6, ridge_rel: float = 1e-6,
                 min_samples_factor: int = 1) -> None:
        super().__init__()
        self.max_gap = int(max_gap)
        self.ridge_rel = float(ridge_rel)
        self.min_samples_factor = int(min_samples_factor)
        self._fit: dict[int, dict[int, dict]] = {}
        self._counts: dict[int, dict] = {}
        self._pred_stats: dict[int, dict] = {}
        self.seq_mismatch = 0
        self.seq_last_mismatch = 0

    def reset_episode(self) -> None:
        super().reset_episode()
        self._fit = {}
        self._counts = {}
        self._pred_stats = {}
        # ★ 计数器跨集保留：它们是"接线是否对"的全局证据，逐集清零会掩盖早期错误

    # ------------------------------------------------------------ 学
    def _model(self, i: int, *, g: int, dim: int, act_dim: int) -> dict:
        per_user = self._fit.setdefault(i, {})
        m = per_user.get(g)
        F = int(dim) + int(g) * int(act_dim) + 1
        if m is not None and (m["dim"] != dim or m["act_dim"] != act_dim):
            raise ValueError(f"MultiLag：用户 {i} 在 g={g} 上的维度中途变了"
                             f"（{m['dim']}/{m['act_dim']} → {dim}/{act_dim}）—— 只可能是接线错误")
        if m is None:
            m = {"g": int(g), "dim": int(dim), "act_dim": int(act_dim), "F": F, "n": 0,
                 "R": np.zeros((F, F), dtype=np.float64),
                 "b": np.zeros((F, dim), dtype=np.float64), "W": None}
            per_user[g] = m
        return m

    def _on_delivery(self, i: int, st: dict, env) -> None:
        cnt = self._counts.setdefault(i, {"pairs": 0, "gap_gt_max": 0, "no_action": 0,
                                          "n_len_mismatch": 0})
        prev = st.get("prev_obs")
        seq = st.get("last_action_seq")
        gap = int(st.get("gap_prev", 0))
        if prev is None:
            return                                   # 首个观测：没有"上一拍"，学不了
        cnt["pairs"] += 1
        if gap > self.max_gap:
            cnt["gap_gt_max"] += 1
            return
        if not seq:
            cnt["no_action"] += 1
            return
        act0 = st.get("last_action")
        if act0 is not None and not np.array_equal(seq[-1], act0):
            # 同一动作在两处不一致 ⇒ 传参接线错，样本不可信
            self.seq_last_mismatch += 1
        if len(seq) != gap:
            # ★ 长度 ≠ gap：窗口与间隔脱节（多/少一步），这条样本会把错位烘进拟合
            cnt["n_len_mismatch"] += 1
            self.seq_mismatch += 1
            return
        y = np.asarray(st["last_obs"], dtype=np.float64)
        x = np.asarray(prev, dtype=np.float64)
        if x.shape != y.shape:
            raise ValueError(f"MultiLag：上一拍观测形状 {x.shape} ≠ 送达观测 {y.shape}")
        act_dim = int(np.asarray(seq[0]).size)
        m = self._model(i, g=gap, dim=int(y.size), act_dim=act_dim)
        z = np.concatenate([x] + [np.asarray(a, dtype=np.float64).reshape(-1) for a in seq]
                           + [np.ones(1, dtype=np.float64)])
        if z.size != m["F"]:
            raise ValueError(f"MultiLag：特征长度 {z.size} ≠ F {m['F']}（g={gap}）")
        m["R"] += np.outer(z, z)
        m["b"] += np.outer(z, y)
        m["n"] += 1
        self._solve(m)

    def _solve(self, m: dict) -> None:
        """相对 ridge，与 `ARX._solve` 同一条实现思路（量纲差一个数量级，固定 ridge 会偏大尺度维）。"""
        F = int(m["F"])
        d = np.diag(m["R"]).copy()
        scale = float(d.max()) if F else 0.0
        if not np.isfinite(scale) or scale <= 0.0:
            return
        reg = m["R"] + self.ridge_rel * np.diag(d) + (1e-9 * scale) * np.eye(F)
        m["W"] = np.linalg.solve(reg, m["b"])

    # ------------------------------------------------------------ 预测
    def _need(self, m: dict) -> int:
        return self.min_samples_factor * int(m["F"])

    def _stat(self, i: int, age: int, used_model: bool) -> None:
        s = self._pred_stats.setdefault(i, {"used_model": {}, "used_degraded": {}})
        key = "used_model" if used_model else "used_degraded"
        s[key][int(age)] = int(s[key].get(int(age), 0)) + 1

    def _predict(self, i: int, st: dict, *, age: int, actions_since: list) -> np.ndarray:
        if age == 0:
            self._stat(i, 0, True)                   # age=0 = 直接拿送达值，不算"预测"
            return st["last_obs"]
        m = self._fit.get(i, {}).get(int(age))
        if m is None or m["W"] is None or m["n"] < self._need(m):
            self._stat(i, age, False)
            return _linear_extrapolate(st, age)      # ★ 显式退化（describe() 可见）
        acts = [np.asarray(a, dtype=np.float64).reshape(-1) for a in actions_since]
        for k, a in enumerate(acts):
            if a.size != m["act_dim"]:
                raise ValueError(f"MultiLag：第 {k} 个动作维度 {a.size} ≠ 学到的 {m['act_dim']}")
        z = np.concatenate([np.asarray(st["last_obs"], dtype=np.float64)] + acts
                           + [np.ones(1, dtype=np.float64)])
        x = z @ m["W"]
        if not np.all(np.isfinite(x)):
            raise ValueError(f"MultiLag：g={age} 的预测输出含 NaN/inf")
        self._stat(i, age, True)
        return x

    # ------------------------------------------------------------ 自述
    def describe(self) -> dict:
        """★ `degraded_ages` 的定义（两种都算"该年龄没有模型"）：
        ① 拟合了但样本不足（`n < F_g`）；② 从未拟合、却在预测里真被用到（走了 `Linear`）。
        单列 `fitted_degraded_ages` 以便区分 —— 只有 ① 才是"学了一半"，② 是"根本没有数据"。
        """
        users: dict[str, dict] = {}
        for i in sorted(set(self._fit) | set(self._counts) | set(self._pred_stats)):
            ages = {}
            for g, m in sorted(self._fit.get(i, {}).items()):
                need = self._need(m)
                ages[str(g)] = {"n_samples": int(m["n"]), "F": int(m["F"]), "n_needed": need,
                                "n_over_F": round(float(m["n"]) / max(int(m["F"]), 1), 3),
                                "degraded": bool(m["n"] < need),
                                "marginal": bool(need <= m["n"] < 2 * int(m["F"])),
                                "fitted": m["W"] is not None}
            st = self._pred_stats.get(i, {})
            fitted_deg = sorted(int(g) for g, v in ages.items() if v["degraded"])
            used_deg = sorted(int(k) for k, v in st.get("used_degraded", {}).items()
                              if int(v) > 0 and int(k) > 0)
            never_fitted = sorted(set(used_deg) - {int(g) for g in ages})
            users[str(i)] = {
                "ages": ages,
                "fitted_degraded_ages": fitted_deg,
                "never_fitted_ages": never_fitted,
                "degraded_ages": sorted(set(fitted_deg) | set(never_fitted)),
                "usable_ages": sorted(int(g) for g, v in ages.items() if not v["degraded"]),
                "used_model": {str(k): int(v) for k, v in sorted(st.get("used_model", {}).items())},
                "used_degraded": {str(k): int(v) for k, v in
                                  sorted(st.get("used_degraded", {}).items())},
                **self._counts.get(i, {}),
            }
        return {"name": self.name, "consumes_env": False, "max_gap": self.max_gap,
                "min_samples_factor": self.min_samples_factor, "ridge_rel": self.ridge_rel,
                "users": users,
                "seq_mismatch": int(self.seq_mismatch),
                "seq_last_mismatch": int(self.seq_last_mismatch)}


class HybridLag(MultiLag):
    """**`M_2`/`M_3` + `Linear` 回退**（T23-b 主对象；`MultiLag` 的纪律子集）。

    为什么是子集而不是新实现（T23-b 预注册 §2）
    --------------------------------------------
    `MultiLag` 为每个 `g ∈ 1..max_gap` 各学一套模型，但 T23-a-2 实测**只有 `g=2,3` 过了样本门**
    （`F_g` 随 `g` 线性增长、样本被 6 阶摊薄）；`g≥4` 拟合出来基本是 ridge 先验、`g=1` 结构性无样本。
    把它们留在 `_predict` 里等于让"没学到的模型"参与决策 ⇒ 收益无法归因到"学到的那一阶"。

    ★ 实现上**不覆盖 `_predict`**：只把不该用的阶挡在**拟合**之外（`_on_delivery`），
      于是 `age ∉ {2,3}` 走的仍是 `MultiLag._predict` 里那条 `_linear_extrapolate`
      —— 与 `Linear` **同一条函数**。这不是"保证一致"，是**结构上不可能不同**
      （T23-b 预注册 §4-2 的匹配消融就建立在这条上）。

    ★ 必须与 `Linear` 在 `age∉{2,3}` 上逐位相同（`scripts/56` 的 C1/C2 判死）。
    """

    name = "hybrid"
    #: 唯一允许启用模型的滞后阶（T23-a-2 §3：`g=2`、`g=3` 是唯一过 `n ≥ F_g` 的两阶）
    usable_gaps: tuple[int, ...] = (2, 3)

    def __init__(self, ridge_rel: float = 1e-6, min_samples_factor: int = 1) -> None:
        super().__init__(max_gap=max(self.usable_gaps), ridge_rel=ridge_rel,
                         min_samples_factor=min_samples_factor)
        #: ★ 跨集累计（**`reset_episode` 不清空**）：`_pred_stats` 是逐集重置的，
        #:   用它算"整轮覆盖率"只会得到最后一集（T23-b 首跑当场抓到：2 集时 134 vs 199）。
        #:   覆盖率是**描述统计**，不是决策状态 ⇒ 累计不构成"跨集继承信息"（模型仍逐集重置）。
        self.cum_stats: dict[int, dict] = {}

    def reset_episode(self) -> None:
        super().reset_episode()          # `_fit` / `_pred_stats` 清空；`cum_stats` 保留

    def _stat(self, i: int, age: int, used_model: bool) -> None:
        super()._stat(i, age, used_model)
        c = self.cum_stats.setdefault(int(i), {"used_model": {}, "used_degraded": {}})
        k = "used_model" if used_model else "used_degraded"
        c[k][int(age)] = int(c[k].get(int(age), 0)) + 1

    def _on_delivery(self, i: int, st: dict, env) -> None:
        gap = int(st.get("gap_prev", 0))
        if st.get("prev_obs") is not None and gap not in self.usable_gaps:
            # ★ 只计数、不入拟合（C3 查这里）；不许"顺手学一下"，否则收益归因失效
            cnt = self._counts.setdefault(i, {"pairs": 0, "gap_gt_max": 0, "no_action": 0,
                                              "n_len_mismatch": 0, "skipped_gap": {}})
            sk = cnt.setdefault("skipped_gap", {})
            sk[str(gap)] = int(sk.get(str(gap), 0)) + 1
            return
        super()._on_delivery(i, st, env)

    def describe(self) -> dict:
        d = super().describe()
        d["usable_gaps"] = [int(g) for g in self.usable_gaps]
        d["cum_stats"] = {str(i): {"used_model": {str(k): int(v) for k, v in sorted(c["used_model"].items())},
                                   "used_degraded": {str(k): int(v) for k, v in
                                                     sorted(c["used_degraded"].items())}}
                          for i, c in sorted(self.cum_stats.items())}
        return d


class SharedLag(MultiLag):
    """**共享 `(A, B_j, c)` 的单模型**（T24 主对象）：`x_{k+g} ≈ A x_k + Σ_{j≤g} B_j a_{k+j} + c`。

    为什么是这一轮（T23-b 的机制指向，`62663b3`）
    --------------------------------------------
    逐阶 `M_g` 的 `F_g = dim + g·act_dim + 1` 随 `g` 线性增长、样本被 6 阶摊薄 ⇒ 只有 `g=2,3` 过门，
    `age=1` **结构性 0%**（一步对仅 1.03%）⇒ 实跑合并覆盖率只有 **17.6%**。
    共享参数把**所有 `g` 的送达对合并进一套正规方程**：`B_1` 由**每一个**送达对贡献样本，
    于是 `age=1` 第一次有模型可用（它的特征里只有 `A, B_1, c` 三块）。

    ★★ 结构偏差（预注册 `2026-10-20-T24-共享AB单模型-预注册-v0.1.md` §2b，写在跑之前）
    ----------------------------------------------------------------------------------
    真实线性系统的精确 `g` 步映射是 `A^g x + Σ A^{g-j} B a_{k+j} + Σ A^i c`，
    即 `a_{k+1}` 的系数随 `g` 变（`A^{g-1}B`）。共享口径把它写成同一个 `B_1` ⇒
    **`g=1` 无偏，`g≥2` 有偏且偏差随 `g` 累积**。这是一次显式的**偏差-方差交换**：
    用高 `g` 的灵活性换低 `g` 的覆盖率。**高 `g` 变差必须写"交易"，不许写"共享模型更好"。**
    （无偏的参数化 `A_g=A^g` 需要非线性最小二乘，本轮不做。）

    实现口径（写死）
    ----------------
    1. 每个用户**一套** `(R, b)`，特征向量定长 `F_total = dim + max_gap·act_dim + 1`；
       第 `j` 个动作块在 `j ≤ g` 时填 `a_{k+j}`、否则**填 0**（= 该块对该样本无约束）。
    2. **不拼装**：`age=d` 一次前推 `d` 步。
    3. 退化路径：`age > max_gap` 或 `n < F_total` ⇒ `_linear_extrapolate`（与 `Linear` **同一函数**，逐位）。
    4. 逐 episode 重置（同 `ARX`/`MultiLag`）；`cum_stats` 跨集累计**只作描述统计**，不进决策。

    ★ 不是世界模型，是**线性辨识基线**；`B_j` 在高 `j` 上样本少（只来自 `g ≥ j` 的对），
      可能停在 ridge 先验（≈0）⇒ 表现为"窗口末端动作被忽略"，`describe()` 的 `gaps_seen` 可查。
    """

    name = "shared"
    consumes_env = False          # ★ 结构性：签名里没有 env/view ⇒ 不可能偷真值

    def __init__(self, max_gap: int = 6, ridge_rel: float = 1e-6,
                 min_samples_factor: int = 1) -> None:
        super().__init__(max_gap=max_gap, ridge_rel=ridge_rel,
                         min_samples_factor=min_samples_factor)
        #: 每用户**一套**共享系统（区别于 `MultiLag` 的 `_fit[i][g]`）
        self._joint: dict[int, dict] = {}
        #: 每个用户实际见过的 `gap` 直方图（写进 `describe()`，用来解释"为什么 `B_6` 学不到"）
        self._gaps_seen: dict[int, dict[int, int]] = {}
        #: 跨集累计的调用统计（`reset_episode` 不清空；同 `HybridLag.cum_stats` 的理由）
        self.cum_stats: dict[int, dict] = {}

    def reset_episode(self) -> None:
        super().reset_episode()          # `_fit`/`_counts`/`_pred_stats` 清空
        self._joint = {}
        self._gaps_seen = {}
        # ★ `cum_stats` 保留：它是描述统计，不是决策状态

    def _stat(self, i: int, age: int, used_model: bool) -> None:
        super()._stat(i, age, used_model)
        c = self.cum_stats.setdefault(int(i), {"used_model": {}, "used_degraded": {}})
        k = "used_model" if used_model else "used_degraded"
        c[k][int(age)] = int(c[k].get(int(age), 0)) + 1

    # ------------------------------------------------------------ 学（单一正规方程）
    def _model_joint(self, i: int, *, dim: int, act_dim: int) -> dict:
        m = self._joint.get(i)
        F = int(dim) + self.max_gap * int(act_dim) + 1
        if m is not None and (m["dim"] != dim or m["act_dim"] != act_dim):
            raise ValueError(f"SharedLag：用户 {i} 的维度中途变了"
                             f"（{m['dim']}/{m['act_dim']} → {dim}/{act_dim}）—— 只可能是接线错误")
        if m is None:
            m = {"dim": int(dim), "act_dim": int(act_dim), "F": F, "n": 0,
                 "R": np.zeros((F, F), dtype=np.float64),
                 "b": np.zeros((F, dim), dtype=np.float64), "W": None}
            self._joint[i] = m
        return m

    def _features(self, x, acts, *, act_dim: int) -> np.ndarray:
        """定长特征：`[x, a_1..a_g, 0..0, 1]`（长度 `F_total`）。"""
        g = len(acts)
        blocks = []
        for j in range(self.max_gap):
            if j < g:
                a = np.asarray(acts[j], dtype=np.float64).reshape(-1)
                if a.size != act_dim:
                    raise ValueError(f"SharedLag：第 {j+1} 个动作维度 {a.size} ≠ 学到的 {act_dim}")
                blocks.append(a)
            else:
                blocks.append(np.zeros(act_dim, dtype=np.float64))
        return np.concatenate([np.asarray(x, dtype=np.float64)] + blocks
                              + [np.ones(1, dtype=np.float64)])

    def _on_delivery(self, i: int, st: dict, env) -> None:
        cnt = self._counts.setdefault(i, {"pairs": 0, "gap_gt_max": 0, "no_action": 0,
                                          "n_len_mismatch": 0})
        prev = st.get("prev_obs")
        seq = st.get("last_action_seq")
        gap = int(st.get("gap_prev", 0))
        if prev is None:
            return                                   # 首个观测：没有"上一拍"，学不了
        cnt["pairs"] += 1
        gs = self._gaps_seen.setdefault(i, {})
        gs[gap] = int(gs.get(gap, 0)) + 1
        if gap > self.max_gap:
            cnt["gap_gt_max"] += 1
            return
        if not seq:
            cnt["no_action"] += 1
            return
        if len(seq) != gap:
            # ★ 长度 ≠ gap：窗口与间隔脱节 ⇒ 这条样本会把错位烘进拟合（#51/#52 同族）
            cnt["n_len_mismatch"] += 1
            self.seq_mismatch += 1
            return
        y = np.asarray(st["last_obs"], dtype=np.float64)
        x = np.asarray(prev, dtype=np.float64)
        if x.shape != y.shape:
            raise ValueError(f"SharedLag：上一拍观测形状 {x.shape} ≠ 送达观测 {y.shape}")
        act_dim = int(np.asarray(seq[0]).size)
        m = self._model_joint(i, dim=int(y.size), act_dim=act_dim)
        z = self._features(x, list(seq), act_dim=act_dim)
        if z.size != m["F"]:
            raise ValueError(f"SharedLag：特征长度 {z.size} ≠ F_total {m['F']}（gap={gap}）")
        m["R"] += np.outer(z, z)
        m["b"] += np.outer(z, y)
        m["n"] += 1
        self._solve(m)                                # 复用 `MultiLag._solve`（同一份 ridge 实现）

    # ------------------------------------------------------------ 预测
    def _need(self, m: dict) -> int:                  # type: ignore[override]
        return self.min_samples_factor * int(m["F"])

    def _predict(self, i: int, st: dict, *, age: int, actions_since: list) -> np.ndarray:
        if age == 0:
            self._stat(i, 0, True)                    # age=0 = 送达值，不算"预测"
            return st["last_obs"]
        m = self._joint.get(i)
        if m is None or m["W"] is None or age > self.max_gap or m["n"] < self._need(m):
            self._stat(i, age, False)
            return _linear_extrapolate(st, age)       # ★ 显式退化（与 `Linear` 同一函数）
        z = self._features(st["last_obs"], list(actions_since), act_dim=int(m["act_dim"]))
        x = z @ m["W"]
        if not np.all(np.isfinite(x)):
            raise ValueError(f"SharedLag：age={age} 的预测输出含 NaN/inf")
        self._stat(i, age, True)
        return x

    # ------------------------------------------------------------ 自述
    def describe(self) -> dict:
        users: dict[str, dict] = {}
        for i in sorted(set(self._joint) | set(self._counts) | set(self._pred_stats)):
            m = self._joint.get(i)
            st = self._pred_stats.get(i, {})
            need = (self._need(m) if m else None)
            users[str(i)] = {
                "dim": (m["dim"] if m else None), "act_dim": (m["act_dim"] if m else None),
                "F_total": (m["F"] if m else None),
                "n_samples": (int(m["n"]) if m else 0),
                "n_needed": need,
                "degraded": (True if m is None else bool(m["n"] < need)),
                "marginal": (None if m is None else bool(need <= m["n"] < 2 * int(m["F"]))),
                "fitted": (None if m is None else bool(m["W"] is not None)),
                "gaps_seen": {str(k): int(v) for k, v in sorted(self._gaps_seen.get(i, {}).items())},
                "used_model": {str(k): int(v) for k, v in sorted(st.get("used_model", {}).items())},
                "used_degraded": {str(k): int(v) for k, v in
                                  sorted(st.get("used_degraded", {}).items())},
                **self._counts.get(i, {}),
            }
        return {"name": self.name, "consumes_env": False, "max_gap": self.max_gap,
                "min_samples_factor": self.min_samples_factor, "ridge_rel": self.ridge_rel,
                "users": users,
                "degraded_users": sorted(int(k) for k, v in users.items() if v["degraded"]),
                "seq_mismatch": int(self.seq_mismatch),
                "seq_last_mismatch": int(self.seq_last_mismatch),
                "cum_stats": {str(i): {"used_model": {str(k): int(v) for k, v in
                                                      sorted(c["used_model"].items())},
                                       "used_degraded": {str(k): int(v) for k, v in
                                                         sorted(c["used_degraded"].items())}}
                              for i, c in sorted(self.cum_stats.items())}}


class _LinearPartial(Linear):
    """机制分解用：只外推"目标维"或只外推"自身维"，其余维保持。

    ★★ 不是可部署方法，**只做机制分解**，不进 T23-b 的主判据。
    ★★ 拆分规则按**观测维度**查表（`_TARGET_OBS_DIMS`），不按环境名：
      拿不到环境名（预测器本来就不该知道自己在哪个环境），维度是它唯一能看见的东西。
      CartPole（4 维）不在表里 ⇒ `LinearTargetOnly` 在它上面**就是 `ZOH`**，
      `LinearSelfOnly` **就是 `Linear`**（逐位，`scripts/54` 的 A5/A6 验）。
    """

    name = "linear_partial"
    extrapolate_target = True

    def _mask(self, dim: int) -> np.ndarray:
        rng = _TARGET_OBS_DIMS.get(int(dim))
        m = np.zeros(int(dim), dtype=bool)
        if rng is None:
            if not self.extrapolate_target:
                m[:] = True                       # 无目标分量 ⇒ "只外推自身" = 全部
            return m                              # "只外推目标" ⇒ 空 ⇒ 等价 ZOH
        lo, hi = rng
        m[lo:hi] = True
        return m if self.extrapolate_target else ~m

    def _predict(self, i: int, st: dict, *, age: int, actions_since: list) -> np.ndarray:
        dim = int(np.size(st["last_obs"]))
        mask = self._mask(dim)
        if bool(mask.all()):
            return _linear_extrapolate(st, age)   # 与 `Linear` 共用实现 ⇒ 逐位一致
        if not bool(mask.any()):
            return st["last_obs"]                 # 与 `ZOH` 完全同一条返回路径
        prev, gap = st.get("prev_obs"), int(st.get("gap_prev", 0))
        if prev is None or gap <= 0 or age == 0:
            return st["last_obs"]
        v = (np.asarray(st["last_obs"], dtype=np.float64) - prev) / float(gap)
        out = np.asarray(st["last_obs"], dtype=np.float64).copy()
        out[mask] += float(age) * v[mask]
        return out

    def describe(self) -> dict:
        dims = sorted({int(np.size(s["last_obs"])) for s in self._st.values() if "last_obs" in s})
        return {"name": self.name, "consumes_env": False,
                "extrapolate_target": bool(self.extrapolate_target),
                "obs_dims": dims,
                "target_component": {str(d): (d in _TARGET_OBS_DIMS) for d in dims}}


class LinearTargetOnly(_LinearPartial):
    """只外推**外部目标**维（UAV：`obs[4:6]`），自身维保持最后一次送达值。"""
    name = "linear_target_only"
    extrapolate_target = True


class LinearSelfOnly(_LinearPartial):
    """只外推**自身**维（UAV：`obs[0:4]`），目标维保持最后一次送达值。"""
    name = "linear_self_only"
    extrapolate_target = False


class Gated(PredictorBase):
    """**按年龄门控的回退**（T27）：`age >= θ` ⇒ 用 `last_obs`（保持）；否则线性外推。

    为什么是它（T25→T26 的直接后果，不是随手挑的启发式）
    ----------------------------------------------------
    T25-I3：伤害最大的格 `B=100/unpaced/u0` 里，按**预测误差**门控的触发率 = **0**
      （三桶三种子误差全部更低）⇒ "误差低 ⇒ 可以补偿"这个方向是**反的**。
    T26-A2：UAV `τ(ΔJ, age)` = +0.435(B200) / +0.832(B100)，两档同号；
    T26-A3：`age∈{1,2–3}` 上 2234 个决策点 `share(ΔJ>0) = 0.000`（补偿无害），
            `age≥4` 才出现危害，且两档危害份额不同（11.0% vs 73.0%）。
    ⇒ `age` 是决策时点**实际可得**、且有符号证据的唯一天然信号；θ 只能由这份数据给出，
      再由**未参与选择的第三批初值**确认（见 T27 预注册 §1/§2）。

    三条实现纪律
    ------------
    1. **θ=0 必须逐位等于 `ZOH`，θ=10⁹ 必须逐位等于 `Linear`** —— 两条边界退化由
       `scripts/68` 的 G0a/G0b 在**真实 cell** 上验。这不是"顺手加的一致性"：
       门控写错时最常见的失效就是"恒走一支"，而那两种失效**都能让结果看起来完全正常**。
    2. **只改 `est`，不碰调度**：`predict()` 的签名里没有 `View`，分配器读的是 runner 的
       `last_obs`/`age` ⇒ 本类结构上不可能影响"把帧给谁"。
    3. **计数器不进数值路径**：`n_hold`/`n_extrap` 只用于 G2 的接线核验（"门控触发了，
       但 `est` 有没有真的换"），不参与任何返回值的计算。
    """

    name = "gated"

    def __init__(self, threshold: int = 4) -> None:
        super().__init__()
        self.threshold = int(threshold)
        self.n_hold = 0
        self.n_extrap = 0
        self.n_hold_by_user: dict[int, int] = {}
        self.n_extrap_by_user: dict[int, int] = {}

    def reset_episode(self) -> None:
        """★ **只清估计状态，不清计数器**。

        计数器是"这个对象活到现在一共回退了多少次"，而 runner 在**每集开头**都会调
        `reset_episode()`。第一版在这里把计数器一起清零 ⇒ G2（计数器 vs 探针的 `age>=θ`
        决策点数）在 2 集上看起来差近一半 —— **那是判据对象错了，不是门控没工作**（#46 同族）。
        每个 cell 都新建一个实例，所以"生命周期累计"与"该 cell 的累计"是同一件事。
        """
        super().reset_episode()

    def _predict(self, i: int, st: dict, *, age: int, actions_since: list) -> np.ndarray:
        if age >= self.threshold:
            self.n_hold += 1
            self.n_hold_by_user[int(i)] = int(self.n_hold_by_user.get(int(i), 0)) + 1
            return st["last_obs"]                     # 与 `ZOH._predict` 同一条返回路径
        self.n_extrap += 1
        self.n_extrap_by_user[int(i)] = int(self.n_extrap_by_user.get(int(i), 0)) + 1
        return _linear_extrapolate(st, age)            # 与 `Linear._predict` 共用实现

    def describe(self) -> dict:
        return {"name": self.name, "consumes_env": False, "threshold": int(self.threshold),
                "n_hold": int(self.n_hold), "n_extrap": int(self.n_extrap),
                "n_hold_by_user": {str(k): int(v) for k, v in sorted(self.n_hold_by_user.items())},
                "n_extrap_by_user": {str(k): int(v) for k, v in sorted(self.n_extrap_by_user.items())},
                "n_users": len(self._st)}


class TrueDyn(PredictorBase):
    """**真实动力学预测（诊断特权）**：从送达时刻的 env 快照出发，用自己实际施加的动作前推。

    ★★ 这不是可部署方法：它需要一份与真实环境同构的仿真器，且与真实 env 状态同步。
      它只回答一个问题——"如果预测**完全准**，调度/补偿能拿到多少"。
      ⇒ 结果里必须标 `diagnostic`，**不得**写成"我们的世界模型"。

    ★★ 影子必须是 `deepcopy`：`copy.deepcopy(env)` 连 `env._rng` 一起复制，因此影子前推
      **不消耗主 env 的任何随机数**、也不改主状态。这条由 `scripts/53` 的 S7 做阴性对照
      （"影子照跑但输出丢弃" ⇒ 主轨迹必须与 `predictor=None` 逐位相同）。

    ★ 终止语义：影子一旦终止，冻结在终止观测（与 `window_cost` 的 absorbed 同向：不再变化），
      并且不再步进（gym 环境在 done 之后步进是未定义行为）。
    """

    name = "truedyn"
    consumes_env = True

    def _on_delivery(self, i: int, st: dict, env) -> None:
        st["shadow"] = copy.deepcopy(env)
        st["shadow_k"] = 0
        st["shadow_obs"] = np.asarray(st["last_obs"], dtype=np.float32).copy()
        st["shadow_dead"] = False

    def _predict(self, i: int, st: dict, *, age: int, actions_since: list) -> np.ndarray:
        if age == 0:
            return st["shadow_obs"]
        if st.get("shadow_dead"):
            return st["shadow_obs"]
        k0 = int(st["shadow_k"])
        if age < k0:
            # age 回退只可能来自"送达但没重置"，属于接线错误 ⇒ 不猜，直接炸
            raise ValueError(f"TrueDyn：age={age} 小于已前推步数 {k0}"
                             f"（送达后必须重建快照；此刻状态已对不上）")
        shadow = st["shadow"]
        for j in range(k0, age):
            res = shadow.step(actions_since[j])
            st["shadow_obs"] = np.asarray(res.obs, dtype=np.float32).reshape(-1).copy()
            if res.terminated or res.truncated:
                st["shadow_dead"] = True
                st["shadow_k"] = age
                return st["shadow_obs"]
        st["shadow_k"] = age
        return st["shadow_obs"]


_PREDICTORS = {"zoh": ZOH, "linear": Linear, "truedyn": TrueDyn, "arx": ARX, "mlag": MultiLag,
               "hybrid": HybridLag, "shared": SharedLag, "gated": Gated,
               "linear_target_only": LinearTargetOnly, "linear_self_only": LinearSelfOnly}


def make_predictor(name: str | None) -> PredictorBase | None:
    """按名字造预测器；`None` / `"none"` ⇒ `None`（= 现状 ZOH，但**不经过**预测器路径）。

    ★ 为什么不把 `None` 直接映射成 `ZOH`：`predictor=None` 走的是既有代码路径（逐位回归的
      真实对象），`ZOH` 走的是新路径。"两者逐位相同"是 `scripts/53` 的 S1，而不是把两者
      合并成同一个东西 —— 合并了就永远测不出"新路径改了行为"。
    """
    if name is None or str(name).lower() in ("none", ""):
        return None
    key = str(name).lower()
    if key not in _PREDICTORS:
        raise ValueError(f"未知预测器 {name!r}；可选 {sorted(_PREDICTORS)} 或 None")
    return _PREDICTORS[key]()
