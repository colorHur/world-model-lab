# -*- coding: utf-8 -*-
"""CW-R2.0：R2 的模型接口、信息权限表与**可比性指纹**。

为什么需要这个文件
------------------
R2 要比"同规划器 + 同预算下，强模型 vs 小学习模型"。R1 已经证明两件事会先吃掉待测效应：

1. **规划器自身的开关差异可以到 1.6×** —— 实测（`pacing` 开/关、同 `ep100`、同策略）：
   `cost_true` `0.11067633547679674 → 0.1767897716156801`、`aoi` `16.0 → 128.5`
   （`results/cw_v1/cw_r1_7_4_tails_20261023T130519_.../bench.json`）。
2. **面板信噪比不足**（R1.7.8：`{est,true}×{相对,绝对}` 四口径 8 个数全 < 2，排序乱跳）。

⇒ 先把"可比性"钉成一个**指纹**，把"信息权限"钉成**一张可机检的表**，再谈训练。

三条不许（本文件是它们的唯一实现）
----------------------------------
1. 模型不许读真值 / 真信道 / **未来随机数**（`TRUTH_FIELDS` ∪ `_z_*` 一律 `never`）。
2. 特权臂不许混进部署表（`deploy()` 对 `privileged=True` 直接抛 `PermissionError`）。
3. 不许用 `ku/kf/share` 当目标函数（它们是 R1 的**机制量**，答"多有害"，不是优化目标）。

★ 本文件**不跑 episode、不训练**；`CWWorld` 只在取视图键集时**构造**一次（`step` 次数 = 0）。
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .plan_ref import PlanRef
from .world import PUBLIC_HISTORY_FIELDS, TRUTH_FIELDS, SchedulerView

PROTOCOL_VERSION = "CW-POMDP-v1.1"
R2_ROUND = "CW-R2.0"

# ==================================================================== 视图字段清单
#: `SchedulerView._pub` 的**顶层**键。改这里等于改协议，必须同时改 `world.view()`。
VIEW_TOP_KEYS: tuple[str, ...] = (
    "t", "n_users", "budget_left", "budget_total", "prior", "telemetry", "history",
)
PRIOR_KEYS: tuple[str, ...] = ("x_mean", "x_var", "h_log_mean", "h_log_var")
TELEMETRY_KEYS: tuple[str, ...] = (
    "i", "x_hat", "P_hat", "stamp", "task_aoi", "h_mean", "h_var", "csi_stamp", "csi_aoi",
)

#: 除了 `world.TRUTH_FIELDS` 之外，**任何**模型都不许持有的量。
#: `_z_*` / `_u_packet` 是**预生成的外生随机数**（含未来时隙）——公开它等于送未来。
EXTRA_NEVER: tuple[str, ...] = (
    "x_true", "log_h_true", "truth_override",
    "_x_true", "_log_h_true", "_w_last", "_z_meas", "_z_csi", "_u_packet",
)

#: AST 闸的禁用**名字**（`Name` / 字符串常量都算）。
FORBIDDEN_NAMES: frozenset[str] = frozenset(set(TRUTH_FIELDS) | set(EXTRA_NEVER))
#: AST 闸的禁用**属性**（`x._pub` 这种直取快照容器的写法不许出现在模型里）。
FORBIDDEN_ATTRS: frozenset[str] = frozenset({"_pub"} | set(EXTRA_NEVER))


def permission_table() -> dict[str, dict]:
    """信息权限表 —— **逐字段分类**，`A4` 拿真实视图键集对它做双向核对。

    双向：① 视图里出现的每个键都必须在表里；② 表里标 `never` 的键都不许出现在视图里。
    """
    t: dict[str, dict] = {}
    top_why = {
        "t": "当前时隙下标（公开）",
        "n_users": "用户数（协议常量）",
        "budget_left": "剩余帧预算（由已花动作推出）",
        "budget_total": "总预算（协议常量）",
        "prior": "公开先验（与真初态采样分布一致，v1.1 起）",
        "telemetry": "逐用户可部署状态（全部由观测+动作+ACK 推出）",
        "history": "已发生的公开记账（白名单字段）",
    }
    for k in VIEW_TOP_KEYS:
        t[f"view.{k}"] = {"visibility": "all", "why": top_why[k]}
    prior_why = {
        "x_mean": "初态先验均值（公开，且与真初态同分布）",
        "x_var": "初态先验方差（公开）",
        "h_log_mean": "初信道先验均值（公开）",
        "h_log_var": "初信道先验方差（公开）",
    }
    for k in PRIOR_KEYS:
        t[f"prior.{k}"] = {"visibility": "all", "why": prior_why[k]}
    tel_why = {
        "i": "用户下标",
        "x_hat": "接收端后验均值（部署量）",
        "P_hat": "接收端后验方差（部署量）",
        "stamp": "最近一次送达时隙（-1 = 从未送达）",
        "task_aoi": "任务侧信息年龄 `t-stamp`（-1 同上）",
        "h_mean": "信道信念均值（部署量）",
        "h_var": "信道信念方差（部署量）",
        "csi_stamp": "最近一次 CSI 读数时隙（-1 = 从未测）",
        "csi_aoi": "CSI 年龄",
    }
    for k in TELEMETRY_KEYS:
        t[f"telemetry.{k}"] = {"visibility": "all", "why": tel_why[k]}
    hist_why = {
        "t": "记录所属时隙",
        "action": "已执行动作（自己的动作）",
        "user": "动作目标用户",
        "resource_spent": "本槽花费（含探测）",
        "probe_done": "本槽是否探测",
        "data_attempted": "本槽是否尝试发数据",
        "data_delivered": "本槽是否送达",
        "ack": "是否收到 ACK（公开反馈）",
        "csi_reading": "**自己花钱测的** CSI 读数（部署量）",
        "y_t": "送达帧的任务观测；**只在送达时非 None**（未送达时它是真值+噪声，公开即绕过信道）",
        "cost_est": "信念侧成本（由 `x̂` 算，公开）",
        "cost_est_by_user": "逐用户信念侧成本（由 `x̂` 算，公开）",
        "budget_left": "该槽末的剩余预算",
    }
    for k in PUBLIC_HISTORY_FIELDS:
        t[f"history.{k}"] = {"visibility": "all", "why": hist_why[k]}
    # ---- never
    never_why = {
        "p_true": "真送达概率（真信道）",
        "w_realized": "实现的**过程噪声**（真值侧）",
        "cost_true": "真实任务成本（真状态算）",
        "cost_true_by_user": "**逐用户**真实成本（R17 的 `r_i` 由它算，只进证据不进视图）",
        "true_norm": "真状态范数（`oracle_known` 用过的那条越界通道）",
        "x_true": "真状态",
        "log_h_true": "真信道",
        "truth_override": "反例用的真值替换句柄",
        "_x_true": "真状态（内部名）",
        "_log_h_true": "真信道（内部名）",
        "_w_last": "上一槽实现的过程噪声",
        "_z_meas": "**预生成**的测量噪声（形状含全时隙 ⇒ 公开即送未来）",
        "_z_csi": "**预生成**的 CSI 噪声（同上）",
        "_u_packet": "**预生成**的丢包均匀数（同上）",
    }
    for k in FORBIDDEN_NAMES:
        t[k] = {"visibility": "never", "why": never_why.get(k, "真值/未来类字段")}
    return t


FIELD_PERMISSION: dict[str, dict] = permission_table()


# ==================================================================== 视图守卫
def require_view(view: Any) -> SchedulerView:
    """**唯一**允许模型拿到输入的入口。

    两条独立条件（缺一不可）：
    ① `type(view) is SchedulerView` —— 用 `is` 不用 `isinstance`：否则子类替身可加属性混过；
    ② `_pub` 键集**恰好**等于声明表 —— 手工拼的假视图（多一个键）也被拒。
    """
    if type(view) is not SchedulerView:  # noqa: E721 —— 故意的，见 docstring ①
        raise TypeError(
            f"模型只接受 `SchedulerView` 本体，收到 {type(view).__name__}。"
            f"真状态/真信道/未来随机数不向模型开放。")
    keys = tuple(sorted(view._pub))
    if keys != tuple(sorted(VIEW_TOP_KEYS)):
        raise TypeError(f"视图键集与声明表不符：多出 {sorted(set(keys) - set(VIEW_TOP_KEYS))}，"
                        f"缺少 {sorted(set(VIEW_TOP_KEYS) - set(keys))}")
    return view


def collect_keys(obj: Any) -> list[str]:
    """递归收集 dict 的键（视图是纯数值快照，只有 dict/list/tuple 三层）。"""
    out: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.append(str(k))
            out.extend(collect_keys(v))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out.extend(collect_keys(v))
    return out


# ==================================================================== 模型输出与特权
@dataclass(frozen=True)
class ModelOutput:
    """`predict()` 的返回。**形状/有限性/正定性当场校验**，不留给下游。"""

    x_hat: np.ndarray
    P: np.ndarray
    label: str = ""
    privileged: bool = False

    def validate(self, n_users: int) -> "ModelOutput":
        for name in ("x_hat", "P"):
            a = np.asarray(getattr(self, name), dtype=float)
            if a.ndim != 1:
                raise ValueError(f"{name} 必须是一维，收到 shape={a.shape}")
            if a.shape != (int(n_users),):
                raise ValueError(f"{name} 形状应为 ({n_users},)，收到 {a.shape}")
            if not np.all(np.isfinite(a)):
                raise ValueError(f"{name} 含 NaN/Inf")
        if np.any(np.asarray(self.P, dtype=float) <= 0.0):
            raise ValueError("P 必须严格为正（方差不能 ≤ 0）")
        return self


@runtime_checkable
class Predictor(Protocol):
    name: str
    privileged: bool

    def predict(self, view: SchedulerView) -> ModelOutput:  # pragma: no cover
        ...


def deploy(predictor: Any) -> Any:
    """把模型放进**部署表**。声明 `privileged=True` 的一律拒绝。"""
    if not hasattr(predictor, "predict"):
        raise TypeError(f"{type(predictor).__name__} 没有 predict()")
    if bool(getattr(predictor, "privileged", False)):
        raise PermissionError(
            f"{getattr(predictor, 'name', type(predictor).__name__)} 声明了 privileged=True"
            f"（读真值/真信道），不得进入部署表；诊断请用 `deploy_diagnostic()`。")
    return predictor


class DiagnosticWrapper:
    """诊断路径的包装：输出 `label` 带 `DIAG-` 前缀，且 `privileged=True` 一路透出。"""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.name = f"DIAG-{getattr(inner, 'name', type(inner).__name__)}"
        self.privileged = True

    def predict(self, view: SchedulerView) -> ModelOutput:
        out = self.inner.predict(view)
        return ModelOutput(out.x_hat, out.P, label=f"DIAG-{out.label}", privileged=True)


def deploy_diagnostic(predictor: Any) -> DiagnosticWrapper:
    return DiagnosticWrapper(predictor)


# ==================================================================== 目标函数（先定义）
#: R2 的目标函数表。**禁用名不许出现在主/辅量里**（闸 `A7`）。
OBJECTIVE: dict[str, Any] = {
    "primary": "cost_true_per_step_median",
    "primary_definition": "逐 episode 的 `q·Σ_i mean_t(x_i²) + r·Σ_i mean_t(u_i²)`，再取 episode 中位数",
    "secondary": [
        "per_user_r_i",                     # R17：逐用户比值，任一 >1 ⇒ 写"交易"
        "mean_task_aoi_per_user_timeavg",   # 必须是**逐用户 + 时间平均**（旧末端单帧口径已废）
        "n_delivered", "n_attempted", "n_probe",
        "cost_true_by_user",
    ],
    "forbidden": ["ku", "kf", "share", "share_user", "share_frame"],
    "statistical_unit": "完整 episode（同初值的 seed 不是独立单元，R1.7.8 D5b）",
    "weights": "由公开 `q_cost/r_cost` 固定，不按最终收益调权",
}

#: 零控（R2.1 才算）：同模型、同 episode、同预算，只改规划器内部采样种子。
NULL_PAIR_SPEC: dict[str, Any] = {
    "formula": "null_pair(k) = |cost_true(model, plan_seed=s_k) − cost_true(model, plan_seed=s'_k)|",
    "aggregate": "floor_R2 = median_k null_pair(k)",
    "why": ("'配速残差 ≥ 效应 ⇒ INCONCLUSIVE' 不能再引用 R1.5 的四格比值 ——"
            "那是**策略差**不是配速残差。改为本轮可自算的零控。"),
    "judgement": "|模型效应| < floor_R2 ⇒ INCONCLUSIVE（**不是**'无效应'）",
}


# ==================================================================== 可比性指纹
#: 进指纹的规划器字段。值**取自签名默认值**，不硬编码（闸 `A6c`）。
PLANNER_PINNED: tuple[str, ...] = (
    "h_plan", "m_samples", "use_probe", "pacing", "cont_m", "tail",
    "send_other_advance", "dp_axis_mode",
)


def planner_signature() -> dict[str, Any]:
    """从 `PlanRef.__init__` 的**实际签名**取默认值，并解析模块级默认。"""
    from . import plan_dp, plan_ref

    sig = inspect.signature(PlanRef.__init__)
    out: dict[str, Any] = {}
    for k in PLANNER_PINNED:
        p = sig.parameters.get(k)
        if p is None:
            raise KeyError(f"`PlanRef.__init__` 没有参数 {k!r}（指纹要求它被钉住）")
        d = p.default
        if d is inspect.Parameter.empty:
            raise ValueError(f"`PlanRef.__init__` 的 {k!r} 没有默认值 ⇒ 无法钉住")
        out[k] = d
    # `tail=None` / `dp_grid=None` 在 `__init__` 里解析成模块级默认 ⇒ 必须一起进指纹
    out["_tail_module_default"] = str(plan_ref.TAIL_DEFAULT)
    out["_dp_grid_module_default"] = str(getattr(plan_dp, "GRID_DEFAULT", "?"))
    return out


@dataclass(frozen=True)
class ComparabilitySpec:
    """「同规划器 + 同预算」的**唯一**表示。改任何一项 ⇒ 指纹变。"""

    plant: dict
    channel: dict
    protocol: dict
    planner: dict
    budget_total: int
    split: str
    episodes: tuple[int, ...]
    seeds: tuple[int, ...]
    models: tuple[str, ...]
    metric: dict
    protocol_version: str = PROTOCOL_VERSION

    def canonical(self) -> str:
        return json.dumps({
            "protocol_version": self.protocol_version,
            "plant": self.plant, "channel": self.channel, "protocol": self.protocol,
            "planner": self.planner, "budget_total": int(self.budget_total),
            "split": self.split,
            "episodes": list(self.episodes), "seeds": list(self.seeds),
            "models": list(self.models), "metric": self.metric,
        }, sort_keys=True, ensure_ascii=False, separators=(",", ":"))

    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical().encode("utf-8")).hexdigest()[:16]

    def as_dict(self) -> dict:
        d = json.loads(self.canonical())
        d["fingerprint"] = self.fingerprint()
        return d


def spec_from_config(cfg: Any, *, budget_total: int, split: str,
                     episodes: tuple[int, ...], seeds: tuple[int, ...],
                     models: tuple[str, ...]) -> ComparabilitySpec:
    """从真实 `CWConfig` 造 spec（plant/channel/protocol 全部现读，不抄常量）。"""
    import dataclasses as _dc

    def _plain(x: Any) -> Any:
        if isinstance(x, tuple):
            return [_plain(v) for v in x]
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        return x

    plant = {k: _plain(v) for k, v in _dc.asdict(cfg.plant).items()}
    plant["n_users"] = int(cfg.plant.n_users)
    channel = {k: _plain(v) for k, v in _dc.asdict(cfg.channel).items()}
    protocol = {k: _plain(v) for k, v in _dc.asdict(cfg.protocol).items()}
    planner = {k: _plain(v) for k, v in planner_signature().items()}
    metric = {"primary": OBJECTIVE["primary"],
              "secondary": list(OBJECTIVE["secondary"]),
              "forbidden": list(OBJECTIVE["forbidden"])}
    return ComparabilitySpec(
        plant=plant, channel=channel, protocol=protocol, planner=planner,
        budget_total=int(budget_total), split=str(split),
        episodes=tuple(int(e) for e in episodes), seeds=tuple(int(s) for s in seeds),
        models=tuple(sorted(str(m) for m in models)), metric=metric)


# ==================================================================== AST 闸（供入口调用）
#: 部署路径模块只许 import 这些（相对模块名）。
ALLOWED_REL_IMPORTS: frozenset[str] = frozenset({"world", "model_api"})
#: `from .world import X` 只许这几个名字。
ALLOWED_FROM_WORLD: frozenset[str] = frozenset({"SchedulerView"})
#: 绝对 import 只许这些根（否则 `import experiments.cw_v1.world` 就是绕过口）。
ALLOWED_ABS_IMPORTS: frozenset[str] = frozenset(
    {"numpy", "typing", "dataclasses", "math", "__future__", "json"})


def scan_source(src: str, *, skip_classes: tuple[str, ...] = ("TruthDiagnostic",)) -> dict:
    """扫一段源码：禁用名 / 禁用属性 / 越界 import。

    返回 `{"names": [...], "attrs": [...], "imports": [...], "skipped": [...]}`。
    `skip_classes` 里的类**整块跳过**（目前只有诊断臂 `TruthDiagnostic`，
    它靠 `deploy()` 的 `PermissionError` 拦住，闸 `A5` 钉住）。
    """
    tree = ast.parse(src)
    skip: list[tuple[int, int]] = []
    skipped_names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name in skip_classes:
            skip.append((node.lineno, int(getattr(node, "end_lineno", node.lineno))))
            skipped_names.append(node.name)

    def _skipped(line: int) -> bool:
        return any(a <= line <= b for a, b in skip)

    names, attrs, imports = [], [], []
    for node in ast.walk(tree):
        if _skipped(getattr(node, "lineno", -1)):
            continue
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            names.append({"name": node.id, "line": node.lineno})
        elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRS:
            attrs.append({"attr": node.attr, "line": node.lineno})
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and node.value in FORBIDDEN_NAMES:
            # `getattr(view, "x_true")` / `view.__dict__["cost_true"]` 这类字符串通道
            names.append({"name": node.value, "line": node.lineno, "via": "str-const"})
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if node.level and mod not in ALLOWED_REL_IMPORTS:
                imports.append({"module": f".{mod}", "line": node.lineno})
            elif node.level and mod == "world":
                bad = [a.name for a in node.names if a.name not in ALLOWED_FROM_WORLD]
                if bad:
                    imports.append({"module": ".world", "names": bad, "line": node.lineno})
            elif not node.level and mod.split(".")[0] not in ALLOWED_ABS_IMPORTS:
                imports.append({"module": mod, "line": node.lineno, "abs": True})
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in ALLOWED_ABS_IMPORTS:
                    imports.append({"module": a.name, "line": node.lineno, "abs": True})
    return {"names": names, "attrs": attrs, "imports": imports,
            "skipped": skipped_names}
