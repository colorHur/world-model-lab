# -*- coding: utf-8 -*-
"""`CW-R2.6`：把 `CW-R2-训练接口-v0.1.md` 的 §2–§6 **落成可断言对象**。

为什么单独成文件
----------------
接口卡是文字。文字拦不住 `5e5` 个参数、也拦不住 `primary="ku"`。
本文件把卡片变成**六个可被反例打的东西**（`T1–T6` 的靶子），外加一个
**零训练的 numpy 骨架**（证明"接口能装下一个模型"，而不是"模型能学到东西"）。

★ 本文件**不训练**：`NumpyMLP.trained = False`（恒真），没有任何 `fit()` / 梯度 /
  `optimizer` / `loss.backward`。参数只在 `__init__` 里随机初始化一次。
★ 本文件**不跑世界**：不 import `runner` / `plan_ref`；只吃 `SchedulerView`。
★ 本文件自身必须过 `model_api.scan_source`（`T2` 的负控之一）：不含任何
  真值/未来字段名，不使用 `_pub`，import 只落在 `model_api.ALLOWED_*` 白名单内。

三条口径冲突（读代码时发现，**只报告不改被测物**）
--------------------------------------------------
1. 接口卡 §5 写"主指标 = `A_t`"，`model_api.OBJECTIVE["primary"]` 仍是
   `cost_true_per_step_median`（R2.0 时期立的）。⇒ `EVAL_PRIMARY="A_t"`，
   `OBJECTIVE["primary"]` 记为 `legacy_primary`。
2. `len(VIEW_TOP_KEYS)+len(PRIOR_KEYS)+len(TELEMETRY_KEYS)=20` 是**声明口径的算术**，
   而 `require_view` 只保证**顶层 7 键**；`history` 的 12 个子字段不在 20 内。
3. `≤5e4 参数` 与 `≤2 隐层 / ≤64 宽` 互相**不咬合**：在本文件的实际输入维度下，
   宽度上限先触发（见 `spec_dict()["max_params_at_structure_cap"]`），
   `5e4` 这条**永远等不到**自己先触发 ⇒ 两条必须分开测（`T5` 两条腿）。

★★ 一条**语义**区分（不做区分就会误杀合法量）
------------------------------------------------
`assert_model_input(name)` 管"**模型能不能看**"（`TRUTH_FIELDS ∪ EXTRA_NEVER` 一律拒）；
`assert_objective(name)` 管"**评价能不能用它**"（只拒 `OBJECTIVE["forbidden"]`）。
`cost_true_by_user` 是**合法的必报评价量**（R17 的 `r_i` 由它算），但它**不许进视图**。
把两者混成一个判据，就会把 R17 拦死。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from .model_api import (
    FIELD_PERMISSION, FORBIDDEN_NAMES, OBJECTIVE, PRIOR_KEYS, TELEMETRY_KEYS,
    VIEW_TOP_KEYS, ModelOutput, require_view,
)
from .world import SchedulerView

SPEC_VERSION = "CW-R2-TRAINSPEC-v1.0"
R2_ROUND = "CW-R2.6"

# ==================================================================== 卡 §2–§6 的常量
#: 卡 §2：可训练参数上限
MAX_PARAMS = 50_000
#: 卡 §2：结构上限
MAX_HIDDEN = 2
MAX_WIDTH = 64
#: 卡 §3：训练 / 评测 episode 集（写死，且必须与 dev 不相交）
TRAIN_EPS: tuple[int, ...] = tuple(range(0, 70))
DEV_EPS: tuple[int, ...] = tuple(range(100, 115))
#: 卡 §3：预算
MAX_GRAD_STEPS = 20_000
MAX_WALLCLOCK_MIN = 30.0
TRAIN_SEEDS: tuple[int, ...] = (7, 17, 27)     # 卡 §3：≥3 个、必须分层

#: 卡 §5：主评价量（R2.4 口径）。**不取** `OBJECTIVE["primary"]`，见文件头冲突 1。
EVAL_PRIMARY = "A_t"
#: 卡 §5：必报量
EVAL_REQUIRED = ("cost_true_per_step", "per_user_r_i")

#: 骨架实际消费的特征（**现算**维度，不手抄 `20`）
PER_USER_FEATURES: tuple[str, ...] = (
    "x_hat", "P_hat", "stamp", "task_aoi", "h_mean", "h_var", "csi_stamp", "csi_aoi",
)
GLOBAL_FEATURES: tuple[str, ...] = (
    "prior.x_mean", "prior.x_var", "prior.h_log_mean", "prior.h_log_var",
    "t", "budget_left", "budget_total",
)
N_FEATURES = len(PER_USER_FEATURES) + len(GLOBAL_FEATURES)


# ==================================================================== 异常
class CapacityExceeded(ValueError):
    """结构或参数越界。消息里必须带**实测**数字。"""


class ReportRefused(ValueError):
    """报告形状不合卡 §6（未按训练种子分层 / 缺种子）。"""


# ==================================================================== spec 的机器化表示
def n_declared_fields() -> int:
    """卡说的"20 个公开字段"是从哪三个元组加出来的 —— **现读**，不写常数。"""
    return len(VIEW_TOP_KEYS) + len(PRIOR_KEYS) + len(TELEMETRY_KEYS)


def layer_shapes(d_in: int, hidden: Sequence[int], d_out: int = 2) -> list[tuple[int, int]]:
    """(in, out) 逐层形状 —— 参数计数的**唯一**来源（`T5` 两条腿共用）。"""
    dims = [int(d_in), *[int(h) for h in hidden], int(d_out)]
    return [(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]


def count_params_from_shapes(shapes: Sequence[tuple[int, int]]) -> int:
    """= Σ (W + b)。与 `count_params(model)` 必须恒等（闸 `T5c`）。"""
    return int(sum(a * b + b for a, b in shapes))


def max_params_at_structure_cap(d_in: int = N_FEATURES, d_out: int = 2) -> int:
    """在 `≤MAX_HIDDEN / ≤MAX_WIDTH` 全用满时能装多少参数 —— 用来钉冲突 3。"""
    return count_params_from_shapes(layer_shapes(d_in, [MAX_WIDTH] * MAX_HIDDEN, d_out))


def spec_dict() -> dict:
    """接口卡的**完整快照**（证据里直接落这个，不落散装常量）。"""
    return {
        "spec_version": SPEC_VERSION,
        "round": R2_ROUND,
        "capacity": {"max_params": MAX_PARAMS, "max_hidden": MAX_HIDDEN,
                     "max_width": MAX_WIDTH,
                     "n_features_used": N_FEATURES,
                     "per_user_features": list(PER_USER_FEATURES),
                     "global_features": list(GLOBAL_FEATURES)},
        "budget": {"train_eps": list(TRAIN_EPS), "dev_eps": list(DEV_EPS),
                   "max_grad_steps": MAX_GRAD_STEPS,
                   "max_wallclock_min": MAX_WALLCLOCK_MIN,
                   "train_seeds": list(TRAIN_SEEDS)},
        "objective": {"eval_primary": EVAL_PRIMARY, "eval_required": list(EVAL_REQUIRED),
                      "legacy_primary": OBJECTIVE["primary"],
                      "forbidden": list(OBJECTIVE["forbidden"]),
                      "secondary": list(OBJECTIVE["secondary"])},
        "derived": {"n_declared_fields": n_declared_fields(),
                    "n_features_used": N_FEATURES,
                    "max_params_at_structure_cap": max_params_at_structure_cap(),
                    "structure_cap_binds_first": bool(
                        max_params_at_structure_cap() < MAX_PARAMS)},
        "conflicts": spec_conflicts(),
        "training": "none",
    }


def spec_conflicts() -> list[dict]:
    """卡与代码的三处不一致 —— 机器化记录（不修，只报）。"""
    return [
        {"id": "C1", "where": "cards §5 vs model_api.OBJECTIVE",
         "card": f"primary = {EVAL_PRIMARY}", "code": f"primary = {OBJECTIVE['primary']}",
         "handling": "本轮 EVAL_PRIMARY 取卡片口径；legacy 保留不改，冲突入证据"},
        {"id": "C2", "where": "cards §2 vs model_api.require_view",
         "card": f"输入 = {n_declared_fields()} 个公开字段",
         "code": "require_view 只保证顶层 %d 键" % len(VIEW_TOP_KEYS),
         "handling": ("20 是三个元组的算术和；history 的 12 个子字段不在其中。"
                      "骨架实际消费 %d 维，未用 = telemetry.i / n_users / history" % N_FEATURES)},
        {"id": "C3", "where": "cards §2 两条上限互不咬合",
         "card": f"params ≤ {MAX_PARAMS} 且 hidden ≤ {MAX_HIDDEN} 宽 ≤ {MAX_WIDTH}",
         "code": (f"结构用满也只有 {max_params_at_structure_cap()} 参数 "
                  f"< {MAX_PARAMS} ⇒ 参数上限永不先触发"),
         "handling": "T5 分两条腿：结构腿 + 参数腿，各自用真数出来的计数"},
    ]


# ==================================================================== 守卫 1：评价量
def assert_objective(name: str) -> dict:
    """**评价侧**：拒 `OBJECTIVE["forbidden"]`，放行主量/必报量，并标注是否只许出现在证据层。"""
    n = str(name)
    forb = list(OBJECTIVE["forbidden"])
    if n in forb:
        raise ValueError(f"评价量 {n!r} 在 OBJECTIVE['forbidden'] 里（{forb}）——"
                         f"它是 R1 的机制量（答'多有害'），不是优化/评价目标")
    allowed = {EVAL_PRIMARY, *EVAL_REQUIRED, *OBJECTIVE["secondary"]} | {OBJECTIVE["primary"]}
    if n not in allowed:
        raise ValueError(f"未知评价量 {n!r}：既不在主量/必报量里，也不在 OBJECTIVE 的声明里")
    vis = FIELD_PERMISSION.get(n, {}).get("visibility")
    return {"name": n, "allowed": True,
            "evidence_side_only": bool(vis == "never"),
            "visibility": vis,
            "why": ("真值类量：只许在证据层计算（算 r_i / 报绝对代价），不许进视图、不许当模型输入"
                    if vis == "never" else "公开量")}


# ==================================================================== 守卫 2：模型输入
def assert_model_input(name: str) -> None:
    """**模型侧**：真值 / 未来随机数 / 内部名一律不许当输入。"""
    n = str(name)
    if n in FORBIDDEN_NAMES:
        raise PermissionError(f"{n!r} 是模型**不可见**字段（TRUTH_FIELDS ∪ EXTRA_NEVER）")
    vis = FIELD_PERMISSION.get(n)
    if vis is None:
        raise KeyError(f"{n!r} 不在信息权限表里 —— 未登记字段不许当输入（白名单，不是黑名单）")
    if vis.get("visibility") != "all":
        raise PermissionError(f"{n!r} 的可见性是 {vis.get('visibility')!r}，不是 'all'")


# ==================================================================== 守卫 3：数据不相交
def assert_disjoint(train: Iterable[int], dev: Iterable[int], *,
                    label: str = "train/dev") -> dict:
    """训练集与评测集必须**零交集**（卡 §3）。交集非空 ⇒ `AssertionError`（不是 warning）。"""
    ti = sorted({int(x) for x in train})
    di = sorted({int(x) for x in dev})
    inter = sorted(set(ti) & set(di))
    if inter:
        raise AssertionError(
            f"{label} 有 {len(inter)} 个 episode 重叠：{inter[:10]}"
            f"{'…' if len(inter) > 10 else ''} ⇒ 训练/评测泄漏，接口不可用")
    return {"n_train": len(ti), "n_dev": len(di), "n_inter": 0,
            "train_range": [ti[0], ti[-1]] if ti else None,
            "dev_range": [di[0], di[-1]] if di else None}


# ==================================================================== 守卫 4/5：容量
def assert_structure(hidden: Sequence[int]) -> dict:
    """卡 §2 的结构上限（层数 / 宽度）。"""
    hs = [int(h) for h in hidden]
    if len(hs) > MAX_HIDDEN:
        raise CapacityExceeded(f"隐层数 {len(hs)} > 上限 {MAX_HIDDEN}")
    wide = [h for h in hs if h > MAX_WIDTH]
    if wide:
        raise CapacityExceeded(f"宽度 {wide} 超上限 {MAX_WIDTH}")
    return {"n_hidden": len(hs), "widths": hs}


def assert_param_budget(n_params: int) -> dict:
    """卡 §2 的参数上限。消息里的数字必须是**调用方真数出来的**。"""
    n = int(n_params)
    if n > MAX_PARAMS:
        raise CapacityExceeded(f"参数 {n} > 上限 {MAX_PARAMS}（超出 {n - MAX_PARAMS}）")
    return {"n_params": n, "max_params": MAX_PARAMS, "headroom": MAX_PARAMS - n}


def count_params(model: "NumpyMLP") -> int:
    return int(sum(int(a.size) for a in model.parameters()))


def assert_capacity(model: "NumpyMLP | None" = None, *, width: int | None = None,
                    depth: int | None = None, d_in: int = N_FEATURES) -> dict:
    """结构 + 参数**两条都查**；`width/depth` 给了就按它们造形状（不必真建模型）。"""
    if width is not None or depth is not None:
        hs = [int(width if width is not None else 32)] * int(depth if depth is not None else 1)
        shapes = layer_shapes(d_in, hs)
        st = assert_structure(hs)
        return {**st, **assert_param_budget(count_params_from_shapes(shapes)),
                "source": "shapes", "shapes": shapes}
    if model is None:
        raise ValueError("必须给 model 或 width/depth")
    measured = count_params(model)
    shapes = layer_shapes(model.d_in, model.hidden)
    from_shapes = count_params_from_shapes(shapes)
    if measured != from_shapes:
        raise AssertionError(f"参数计数不自洽：实测 {measured} ≠ 形状算出 {from_shapes}")
    st = assert_structure(model.hidden)
    return {**st, **assert_param_budget(measured), "source": "model",
            "shapes": shapes, "n_params_measured": measured}


# ==================================================================== 零训练骨架
class NumpyMLP:
    """**未训练**的最小前向网络：只证明"接口装得下模型"，不证明"模型能学"。

    - 输入：`SchedulerView` 的 `N_FEATURES` 维特征（per-user 8 + 全局 7 广播）；
    - 输出：`ModelOutput`（逐用户 `x_hat` 与 `P`）—— 与 `BeliefReadout` 同构；
    - `P` 用 `softplus` 保正（`ModelOutput.validate` 要求 `P > 0`）；
    - **没有** `fit()` / `loss` / `grad`；`trained` 恒 `False`。
    """

    name = "numpy_mlp_skeleton"
    privileged = False
    trained = False

    def __init__(self, *, hidden: Sequence[int] = (32,), seed: int = 0,
                 d_in: int = N_FEATURES, d_out: int = 2) -> None:
        self.hidden = tuple(int(h) for h in hidden)
        self.d_in = int(d_in)
        self.d_out = int(d_out)
        self.seed = int(seed)
        self.shapes = layer_shapes(self.d_in, self.hidden, self.d_out)
        assert_structure(self.hidden)                       # 结构先撞
        assert_param_budget(count_params_from_shapes(self.shapes))   # 参数后撞
        rng = np.random.default_rng(self.seed)
        self._w: list[np.ndarray] = []
        self._b: list[np.ndarray] = []
        for a, b in self.shapes:
            self._w.append(rng.normal(0.0, 1.0 / math.sqrt(a), size=(a, b)))
            self._b.append(np.zeros(b))

    # ---- 参数
    def parameters(self) -> list[np.ndarray]:
        return [*self._w, *self._b]

    @property
    def n_params(self) -> int:
        return count_params(self)

    def describe(self) -> dict:
        return {"name": self.name, "n_params": self.n_params, "hidden": list(self.hidden),
                "d_in": self.d_in, "d_out": self.d_out, "seed": self.seed,
                "trained": bool(self.trained), "learned": False,
                "uses": list(PER_USER_FEATURES) + list(GLOBAL_FEATURES),
                "note": "骨架：只跑了前向，没有任何参数更新"}

    # ---- 前向
    def features(self, view: SchedulerView) -> np.ndarray:
        v = require_view(view)
        g = [float(v.prior["x_mean"]), float(v.prior["x_var"]),
             float(v.prior["h_log_mean"]), float(v.prior["h_log_var"]),
             float(v.t), float(v.budget_left), float(v.budget_total)]
        rows = []
        for tel in v.telemetry:
            u = [float(tel[k]) for k in PER_USER_FEATURES]
            rows.append(u + g)
        return np.asarray(rows, dtype=float).reshape(int(v.n_users), self.d_in)

    def predict(self, view: SchedulerView) -> ModelOutput:
        v = require_view(view)
        h = self.features(v)
        for i, (w, b) in enumerate(zip(self._w, self._b)):
            h = h @ w + b
            if i < len(self._w) - 1:
                h = np.tanh(h)
        x_hat = h[:, 0]
        P = np.logaddexp(0.0, h[:, 1]) + 1e-6          # softplus ⇒ 严格为正
        return ModelOutput(x_hat=x_hat, P=P, label=self.name).validate(int(v.n_users))


# ==================================================================== 守卫 6：报告形状
def build_report(by_seed: dict, *, pooled: dict | None = None,
                 required_seeds: Sequence[int] = TRAIN_SEEDS,
                 delta_key: str = "delta") -> dict:
    """卡 §6：**必须**按训练种子分层报；只给合并值 ⇒ 拒答。

    `by_seed[s]` 可以是数字列表或 `[{delta: x}, …]`。`pooled` 允许存在，
    但一律标 `descriptive_only=True`（合并值不许单独承载判词）。
    """
    req = [int(s) for s in required_seeds]
    got: list[int] = []
    for k in by_seed:
        try:
            got.append(int(k))
        except (TypeError, ValueError):
            raise ReportRefused(
                f"报告键 {k!r} 不是训练种子（种子集 {req}）；"
                f"把 'pooled'/'all' 当键 = 用合并值顶分层，卡 §6 禁止") from None
    missing = [s for s in req if s not in got]
    if missing:
        raise ReportRefused(
            f"缺训练种子 {missing}（已有 {sorted(got)}，要求 {req}）⇒ 只报合并 Δ 或部分种子，"
            f"卡 §6 禁止")
    out: dict[str, Any] = {"by_seed": {}, "required_seeds": req,
                           "n_seeds": len(req)}
    for s in req:
        vals = by_seed[s]
        arr = np.asarray([(float(v[delta_key]) if isinstance(v, dict) else float(v))
                          for v in vals], dtype=float)
        if arr.size == 0:
            raise ReportRefused(f"种子 {s} 的 Δ 列表为空")
        out["by_seed"][str(s)] = {"n": int(arr.size),
                                  "median": float(np.median(arr)),
                                  "mean": float(arr.mean()),
                                  "n_pos": int((arr > 0).sum()),
                                  "n_neg": int((arr < 0).sum())}
    if pooled is not None:
        pv = np.asarray([(float(v[delta_key]) if isinstance(v, dict) else float(v))
                         for v in pooled], dtype=float)
        out["pooled"] = {"n": int(pv.size), "median": float(np.median(pv)),
                         "descriptive_only": True,
                         "why": "合并值只用于描述；判词必须落在 by_seed 上（卡 §6）"}
    out["seeds_differ"] = len({v["median"] for v in out["by_seed"].values()})
    return out
