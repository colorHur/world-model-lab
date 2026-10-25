# -*- coding: utf-8 -*-
"""`CW-R2.6` 落盘入口：把接口卡 §7 的 `T1–T6` **真跑**，每条带负控。

预注册：`docs/research/CW-R2.6-训练接口验收-预注册-v0.1.md`（写于 2026-10-25 15:01，`run_episode`=0）

设计要点（为什么每条都要两向）
------------------------------
只测"反例被拦"是**没有分辨力**的检验：一个 `raise ValueError` 写在函数第一行的守卫也能全过。
所以每条 `Tk` 都配一条**负控**（正常路径必须放行、且返回可用值）。
`caught ∧ clean` 才是证据；任一为假 ⇒ 判词 `SPEC-GATES-INCOMPLETE`。

`T4b` 是**活体**腿：从已落盘证据里读 R2.1–R2.5 实际用过的 episode 集，
与 `TRAIN_EPS` 求交 ⇒ 必须为空。它专门拦"卡里写 100–114、代码里偷偷训到 114"这类泄漏。

★ 本轮零训练、零 episode、零参数更新；被测物一字节不动；不改 `OBJECTIVE`。
★ `CWWorld` 只**构造**一次（为拿一个真视图做 `T3` 的负控），`step` = 0。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .config import mainline_config
from .model_api import OBJECTIVE, require_view, scan_source
from .r1_14_exactbase import _git, _src_hash
from .r2_train_spec import (
    DEV_EPS, MAX_PARAMS, N_FEATURES, TRAIN_EPS, TRAIN_SEEDS,
    CapacityExceeded, NumpyMLP, ReportRefused, assert_capacity, assert_disjoint,
    assert_model_input, assert_objective, assert_param_budget, assert_structure,
    build_report, count_params, count_params_from_shapes, layer_shapes,
    max_params_at_structure_cap, n_declared_fields, spec_dict,
)
from .streams import make_episode_rngs
from .world import CWWorld, SchedulerView

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results" / "cw_v1"
HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "fixtures" / "dirty_model_src.py.txt"
SELF_SRC = HERE / "r2_train_spec.py"
MODELS_SRC = HERE / "r2_models.py"

#: `T4b` 的活体证据源：R2.5（最近一轮用过完整 dev 面板的 run）
R25_DIR = RESULTS / "cw_r2_5_seed_20261025T045209_f00b22682d18e97f"

#: 被测物（预注册 §2）：本轮**一字节不动**。用 git blob 身份核对，零计算、不重跑。
PINNED_FILES: tuple[str, ...] = (
    "experiments/cw_v1/world.py",
    "experiments/cw_v1/plan_ref.py",
    "experiments/cw_v1/plan_dp.py",
    "experiments/cw_v1/runner.py",
    "experiments/cw_v1/config.py",
    "experiments/cw_v1/policies.py",
    "experiments/cw_v1/model_api.py",
    "experiments/cw_v1/r2_models.py",
    "experiments/cw_v1/mech2.py",
    "experiments/cw_v1/mech3.py",
    "experiments/cw_v1/r2_1_sep.py",
    "experiments/cw_v1/r2_4_aoi.py",
    "experiments/cw_v1/r2_5_seed.py",
)


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _leg(name: str, caught: bool, clean: bool, detail: dict) -> dict:
    return {"id": name, "caught": bool(caught), "clean": bool(clean),
            "pass": bool(caught and clean), **detail}


# ==================================================================== T1 评价量
def t1_objective() -> dict:
    """`primary="ku"` 必拒；`A_t` 与 `cost_true_by_user` 必放行（后者标证据层）。"""
    caught, clean, det = True, True, {}
    ref = {}
    for bad in ("ku", "kf", "share", "share_user"):
        try:
            assert_objective(bad)
            caught = False
            ref[bad] = "NOT-RAISED"
        except ValueError as e:
            ref[bad] = f"ValueError: {str(e)[:40]}"
    # 模型输入侧的同类拦截（语义不同，见 r2_train_spec 文件头）
    try:
        assert_model_input("x_true")
        ref["model_input:x_true"] = "NOT-RAISED"
        caught = False
    except PermissionError as e:
        ref["model_input:x_true"] = f"PermissionError: {str(e)[:40]}"
    det["refused"] = ref

    ok = {}
    try:
        ok["A_t"] = assert_objective("A_t")
    except Exception as e:  # noqa: BLE001
        clean = False
        ok["A_t"] = f"{type(e).__name__}: {e}"
    try:
        ok["cost_true_by_user"] = assert_objective("cost_true_by_user")
    except Exception as e:  # noqa: BLE001
        clean = False
        ok["cost_true_by_user"] = f"{type(e).__name__}: {e}"
    try:
        # 公开量必须能当模型输入（前缀名，权限表的键就是这么写的）
        assert_model_input("telemetry.x_hat")
        assert_model_input("prior.x_mean")
        ok["model_input:telemetry.x_hat"] = "allowed"
    except Exception as e:  # noqa: BLE001
        clean = False
        ok["model_input:telemetry.x_hat"] = f"{type(e).__name__}: {e}"
    # 负控必须**真的是**那个语义，不是"不抛异常就算过"
    if not (isinstance(ok["A_t"], dict) and ok["A_t"].get("allowed") is True):
        clean = False
    ctbu = ok["cost_true_by_user"]
    if not (isinstance(ctbu, dict) and ctbu.get("allowed") is True
            and ctbu.get("evidence_side_only") is True):
        clean = False   # ★ 若把 R17 的必报量当"禁用字段"拦掉，这条立刻 FAIL
    det["allowed"] = ok
    return _leg("T1", caught, clean, det)


# ==================================================================== T2 AST 扫描
def t2_scan() -> dict:
    """脏源码三栏必抓；两个真实文件三栏必空。"""
    caught, clean, det = True, True, {}
    dirty_src = FIXTURE.read_text(encoding="utf-8")
    d = scan_source(dirty_src, skip_classes=())
    det["dirty"] = {"n_names": len(d["names"]), "n_attrs": len(d["attrs"]),
                    "n_imports": len(d["imports"]),
                    "names": [x["name"] for x in d["names"]],
                    "attrs": [x["attr"] for x in d["attrs"]],
                    "imports": [x["module"] for x in d["imports"]]}
    if not (len(d["names"]) >= 2 and len(d["attrs"]) >= 1 and len(d["imports"]) >= 1):
        caught = False
    # 负控 1：诊断臂整类被跳过（否则 `TruthDiagnostic` 永远报脏）
    d2 = scan_source(MODELS_SRC.read_text(encoding="utf-8"))
    det["r2_models"] = {"n_names": len(d2["names"]), "n_attrs": len(d2["attrs"]),
                        "n_imports": len(d2["imports"]), "skipped": d2["skipped"]}
    # 负控 2：本轮新写的骨架文件自身也必须干净（含 NumpyMLP，不跳过任何类）
    d3 = scan_source(SELF_SRC.read_text(encoding="utf-8"), skip_classes=())
    det["r2_train_spec"] = {"n_names": len(d3["names"]), "n_attrs": len(d3["attrs"]),
                            "n_imports": len(d3["imports"]),
                            "hits": (d3["names"] + d3["attrs"] + d3["imports"])[:6]}
    if any(len(d2[k]) for k in ("names", "attrs", "imports")) or \
            any(len(d3[k]) for k in ("names", "attrs", "imports")):
        clean = False
    return _leg("T2", caught, clean, det)


# ==================================================================== T3 视图守卫
def t3_view(view) -> dict:
    """子类替身（塞了真值）必拒；真视图必过。"""
    caught, clean, det = True, True, {}

    class EvilView(SchedulerView):
        """把真值挂在子类属性上 —— `isinstance` 会放它过，`type(...) is` 不会。"""

    evil = EvilView(view._pub)
    evil.x_true = np.zeros(int(view.n_users))     # noqa: B010 —— 反例就是要塞真值
    evil.cost_true_by_user = np.zeros(int(view.n_users))
    try:
        require_view(evil)
        det["subclass"] = "NOT-RAISED"
        caught = False
    except TypeError as e:
        det["subclass"] = f"TypeError: {str(e)[:60]}"
    # 反例 2：手工拼的 dict（根本不是 SchedulerView）
    try:
        require_view({"t": 0})
        det["dict"] = "NOT-RAISED"
        caught = False
    except TypeError as e:
        det["dict"] = f"TypeError: {str(e)[:40]}"
    # 负控：真视图
    try:
        same = require_view(view)
        det["clean_view"] = {"same_object": bool(same is view),
                            "keys": sorted(view._pub)}
        if not (same is view):
            clean = False
    except Exception as e:  # noqa: BLE001
        clean = False
        det["clean_view"] = f"{type(e).__name__}: {e}"
    det["evil_had_attrs"] = {"x_true": True, "cost_true_by_user": True}
    return _leg("T3", caught, clean, det)


# ==================================================================== T4 数据不相交
def t4_disjoint() -> dict:
    """重叠集必报（且报出重叠个数）；训练/dev 必零交；**活体**核对历史面板实际 ep。"""
    caught, clean, det = True, True, {}
    try:
        assert_disjoint(range(0, 70), range(60, 75))
        det["overlap"] = "NOT-RAISED"
        caught = False
    except AssertionError as e:
        det["overlap"] = f"AssertionError: {str(e)[:60]}"
        if "10 个" not in str(e):
            caught = False       # ★ 必须报出正确的重叠规模，不能只是"非空"就抛
    try:
        det["nominal"] = assert_disjoint(TRAIN_EPS, DEV_EPS, label="train/dev(卡 §3)")
    except Exception as e:  # noqa: BLE001
        clean = False
        det["nominal"] = f"{type(e).__name__}: {e}"

    # ---- T4b 活体：历史面板真的用了哪些 ep
    try:
        r25 = json.loads((R25_DIR / "summary.json").read_text(encoding="utf-8"))
        used = sorted({int(e) for e in r25["design"]["eps"]})
        det["r25_dev_eps"] = used
        inter = sorted(set(used) & set(TRAIN_EPS))
        det["r25_intersection_with_train"] = inter
        det["r25_source"] = str(R25_DIR.name)
        if inter:
            caught = False        # 泄漏 ⇒ 反例腿不成立（闸没能拦住真泄漏）
    except Exception as e:  # noqa: BLE001
        caught = False            # ★ 读不到证据 ≠ 通过（R14：报错优于假数字）
        det["r25_dev_eps"] = f"{type(e).__name__}: {e}"
    return _leg("T4", caught, clean, det)


# ==================================================================== T5 容量
def t5_capacity() -> dict:
    """结构上限与参数上限**分开**测（卡里两条互不咬合），且计数必须真数。"""
    caught, clean, det = True, True, {}
    # 腿 a：结构——宽度 256
    try:
        assert_capacity(width=256)
        det["struct_width256"] = "NOT-RAISED"
        caught = False
    except CapacityExceeded as e:
        det["struct_width256"] = f"CapacityExceeded: {str(e)[:50]}"
    # 腿 b：参数——用同一套形状函数算出真计数，再喂给参数守卫
    shapes = layer_shapes(N_FEATURES, [256, 256])
    n_over = count_params_from_shapes(shapes)
    det["param_measured_over"] = {"n_params": n_over, "shapes": shapes,
                                  "by": "count_params_from_shapes"}
    try:
        assert_param_budget(n_over)
        det["param_budget"] = "NOT-RAISED"
        caught = False
    except CapacityExceeded as e:
        det["param_budget"] = f"CapacityExceeded: {str(e)[:60]}"
        if str(n_over) not in str(e):
            caught = False        # ★ 消息里必须是实测数，不许硬编码
    if n_over <= MAX_PARAMS:
        caught = False            # 反例本身必须真的越界，否则测的是空气

    # 负控 a：宽 32 的骨架（真建，真数）
    m32 = NumpyMLP(hidden=(32,), seed=7)
    det["clean_width32"] = assert_capacity(m32)
    det["clean_n_params"] = m32.n_params
    if count_params(m32) != count_params_from_shapes(layer_shapes(N_FEATURES, (32,))):
        clean = False
    # 负控 b：结构上限用满也只有这么多参数（卡 §4 冲突 3 的实测值）
    det["max_params_at_cap"] = max_params_at_structure_cap()
    det["conflict3_confirmed"] = bool(max_params_at_structure_cap() < MAX_PARAMS)
    if det["max_params_at_cap"] != count_params_from_shapes(
            layer_shapes(N_FEATURES, [64, 64])):
        clean = False
    # 负控 c：层数上限
    try:
        assert_structure([16, 16, 16])
        det["struct_depth3"] = "NOT-RAISED"
        caught = False
    except CapacityExceeded as e:
        det["struct_depth3"] = f"CapacityExceeded: {str(e)[:40]}"
    return _leg("T5", caught, clean, det)


# ==================================================================== T6 报告形状
def t6_report() -> dict:
    """只报合并 / 缺种子必拒；三种子齐必过。"""
    caught, clean, det = True, True, {}
    ref = {}
    try:
        build_report({7: [0.1, 0.2, -0.1]})
        ref["one_seed"] = "NOT-RAISED"
        caught = False
    except ReportRefused as e:
        ref["one_seed"] = f"ReportRefused: {str(e)[:50]}"
    try:
        build_report({"pooled": [0.1, 0.2, -0.1]})
        ref["pooled_only"] = "NOT-RAISED"
        caught = False
    except ReportRefused as e:
        ref["pooled_only"] = f"ReportRefused: {str(e)[:50]}"
    det["refused"] = ref

    rng = np.random.default_rng(11)
    by = {s: list(rng.normal(0.05, 0.1, size=10)) for s in TRAIN_SEEDS}
    try:
        rep = build_report(by, pooled=list(rng.normal(0.05, 0.1, size=30)))
        det["report"] = {"n_seeds": rep["n_seeds"],
                         "by_seed_medians": {k: round(v["median"], 6)
                                             for k, v in rep["by_seed"].items()},
                         "pooled_descriptive_only": rep["pooled"]["descriptive_only"],
                         "seeds_differ_in_output": rep["seeds_differ"]}
        if set(rep["by_seed"]) != {str(s) for s in TRAIN_SEEDS}:
            clean = False
        if rep["pooled"]["descriptive_only"] is not True:
            clean = False
    except Exception as e:  # noqa: BLE001
        clean = False
        det["report"] = f"{type(e).__name__}: {e}"
    return _leg("T6", caught, clean, det)


# ==================================================================== 骨架前向（活体）
def skeleton_forward(view) -> dict:
    """零训练骨架必须**能真跑一次前向**并满足 `ModelOutput.validate`。"""
    out: dict = {}
    m = NumpyMLP(hidden=(32,), seed=7)
    mo = m.predict(view)
    out["n_users"] = int(view.n_users)
    out["x_hat_len"] = int(mo.x_hat.size)
    out["P_min"] = float(np.min(mo.P))
    out["finite"] = bool(np.all(np.isfinite(mo.x_hat)) and np.all(np.isfinite(mo.P)))
    out["describe"] = m.describe()
    out["pass"] = bool(mo.x_hat.size == int(view.n_users) and np.all(mo.P > 0)
                       and out["finite"] and m.trained is False)
    # 前向不改参数（零训练的第二重证据：两次前向参数逐位相同）
    p_before = [a.copy() for a in m.parameters()]
    m.predict(view)
    out["params_unchanged_after_two_forwards"] = bool(
        all(np.array_equal(a, b) for a, b in zip(p_before, m.parameters())))
    out["pass"] = bool(out["pass"] and out["params_unchanged_after_two_forwards"])
    return out


# ==================================================================== 被测物身份（零计算）
def identity_check() -> dict:
    """13 个被测物文件的 `hash-object` 必须等于 `HEAD:path` —— 逐字节，不重跑。"""
    head = _git(["rev-parse", "HEAD"])
    rows, bad = [], []
    for rel in PINNED_FILES:
        now = _git(["hash-object", rel])
        at_head = _git(["rev-parse", f"{head}:{rel}"], allow_fail=True)
        same = bool(now and at_head and now == at_head)
        rows.append({"file": rel, "worktree": now[:12] if now else None,
                     "head": at_head[:12] if at_head else None, "same": same})
        if not same:
            bad.append(rel)
    return {"head": head, "n": len(rows), "n_bad": len(bad), "bad": bad,
            "pass": not bad, "rows": rows,
            "why": "被测物若被改过，本轮'零训练/零 episode'的声明就不覆盖同一份代码"}


# ==================================================================== 变异测试（闸的闸）
def _always_allow(name, *a, **k):
    return {"name": str(name), "allowed": True, "evidence_side_only": False,
            "visibility": "all"}


def _no_input_guard(name, *a, **k):
    return None


def _empty_scan(src, **k):
    return {"names": [], "attrs": [], "imports": [], "skipped": []}


def _pass_through(view):
    return view


def _no_disjoint(train, dev, **k):
    return {"n_train": 0, "n_dev": 0, "n_inter": 0}


def _no_structure(hidden):
    return {"n_hidden": len(list(hidden)), "widths": list(hidden)}


def _no_budget(n):
    return {"n_params": int(n), "max_params": 10 ** 9}


def _bad_report(by_seed, **k):
    return {"by_seed": {}, "required_seeds": [], "n_seeds": 0,
            "pooled": {"descriptive_only": False}}


#: 变异体：`(编号, 目标 T, [(全局名, 替身), …])`。
#: 判据：替换后对应 `Tk.pass` **必须变 False**。若仍 True ⇒ 该 T 是**恒绿闸**（无分辨力）。
MUTATIONS: tuple[tuple[str, str, tuple], ...] = (
    ("M1", "T1", (("assert_objective", _always_allow), ("assert_model_input", _no_input_guard))),
    ("M2", "T2", (("scan_source", _empty_scan),)),
    ("M3", "T3", (("require_view", _pass_through),)),
    ("M4", "T4", (("assert_disjoint", _no_disjoint),)),
    ("M5", "T5", (("assert_param_budget", _no_budget), ("assert_structure", _no_structure))),
    ("M6", "T6", (("build_report", _bad_report),)),
)

RUNNERS = {"T1": "t1_objective", "T2": "t2_scan", "T3": "t3_view",
           "T4": "t4_disjoint", "T5": "t5_capacity", "T6": "t6_report"}


def mutation_probe(view) -> list[dict]:
    """把守卫逐个改弱 / 把常量篡改，看对应 `Tk` 是否**真的会红**。

    ★ 这是本轮最重要的一条：`T1–T6` 全绿只能说明"当前实现拦得住"，
      不能说明"检验有分辨力"。只有变异后转红，才证明 `caught` 腿不是装饰。
    ★ `M4b` 是**常量**变异（不是函数替换）：把 `TRAIN_EPS` 篡改成含 `dev[0]`，
      专门打 `T4b` 那条活体腿 —— 它用的是独立断言，`M4` 的替身覆盖不到它。
    """
    g = globals()
    out: list[dict] = []
    for mid, tgt, subs in MUTATIONS:
        saved = [(n, g[n]) for n, _ in subs]
        try:
            for n, fake in subs:
                g[n] = fake
            fn = g[RUNNERS[tgt]]
            res = fn(view) if tgt == "T3" else fn()
            red = not bool(res["pass"])
            out.append({"id": mid, "target": tgt, "turns_red": red,
                        "pass": red,
                        "detail": {"caught_after": res["caught"], "clean_after": res["clean"]},
                        "why": (f"削弱 {[n for n, _ in subs]} 后 {tgt} 仍为绿 ⇒ 该检验无分辨力"
                                if not red else "变异后转红 ⇒ 检验有分辨力")})
        except Exception as e:  # noqa: BLE001
            # ★ 异常一律记 **FAIL**，不记"红"：异常可能来自替身自身（如返回结构不符合
            #   下游字段约定），那证明不了 `caught` 腿有分辨力。宁可这轮红，不拿异常冒充拦截。
            out.append({"id": mid, "target": tgt, "turns_red": False, "pass": False,
                        "detail": {"raised": repr(e)[:160]},
                        "why": "变异后抛异常 ⇒ 无法归因到 caught 腿（可能是替身的副作用）"})
        finally:
            for n, v in saved:
                g[n] = v

    # ---- M4b：常量变异（训练集被污染成含 dev 中的一个 ep）
    saved_train = g["TRAIN_EPS"]
    try:
        polluted = tuple(sorted(set(TRAIN_EPS) | {int(DEV_EPS[0])}))
        g["TRAIN_EPS"] = polluted
        res = t4_disjoint()
        det = res.get("r25_intersection_with_train")
        red = not bool(res["pass"])
        out.append({"id": "M4b", "target": "T4b(活体)", "turns_red": red, "pass": red,
                    "detail": {"polluted_train": list(polluted), "caught_after": res["caught"],
                               "intersection_after": det},
                    "why": ("把 TRAIN_EPS 篡改成含 dev[0] 后，活体腿必须报出交集；"
                            "若仍绿 ⇒ T4b 是恒绿闸（读不到证据也算过）"
                            if not red else "污染训练集后活体腿转红 ⇒ T4b 有分辨力")})
    except Exception as e:  # noqa: BLE001
        out.append({"id": "M4b", "target": "T4b(活体)", "turns_red": False, "pass": False,
                    "detail": {"raised": repr(e)[:160]},
                    "why": "变异后抛异常 ⇒ 无法归因到 T4b"})
    finally:
        g["TRAIN_EPS"] = saved_train
    return out


# ==================================================================== 主流程
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="CW-R2.6 训练接口验收（T1–T6 双向腿，零训练）")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--no-evidence", action="store_true", help="只打印，不落盘（自检用）")
    args = ap.parse_args(argv)

    t_start = time.perf_counter()

    # ---- 真视图（只构造，不 step）
    cfg = mainline_config()
    rngs, sids = make_episode_rngs(cfg.root_seed, "dev", 900)
    w = CWWorld(cfg, rngs, sids)
    view = w.view()
    steps_after_construct = int(w.t)

    legs = [t1_objective(), t2_scan(), t3_view(view), t4_disjoint(), t5_capacity(), t6_report()]
    for L in legs:
        print(f"[{L['id']}] caught={L['caught']} clean={L['clean']} pass={L['pass']}")
    n_caught = sum(1 for L in legs if L["caught"])
    n_clean = sum(1 for L in legs if L["clean"])
    n_pass = sum(1 for L in legs if L["pass"])
    verdict = "SPEC-GATES-READY" if n_pass == len(legs) == 6 else "SPEC-GATES-INCOMPLETE"
    print(f"[VERDICT] {verdict} pass={n_pass}/6 caught={n_caught}/6 clean={n_clean}/6")

    # ---- 骨架前向（活体，零训练）
    skel: dict = {}
    try:
        skel = skeleton_forward(view)
        print(f"[SKELETON] pass={skel['pass']} n_params={skel['describe']['n_params']} "
              f"P_min={skel['P_min']:.3g} params_unchanged={skel['params_unchanged_after_two_forwards']}")
    except Exception as e:  # noqa: BLE001
        skel = {"pass": False, "error": repr(e), "traceback": traceback.format_exc()[-800:]}
        print(f"[SKELETON] FAIL {e!r}")
    n_pass_total = n_pass + (1 if skel.get("pass") else 0)
    verdict_total = ("SPEC-GATES-READY" if (verdict == "SPEC-GATES-READY" and skel.get("pass"))
                     else "SPEC-GATES-INCOMPLETE")

    # ---- 被测物身份（不重跑任何旧实验，只比 blob 哈希）
    ident = identity_check()
    print(f"[IDENT] n_bad={ident['n_bad']}/{ident['n']} head={ident['head']}")

    # ---- 变异测试：闸的闸（若某 T 变异后仍绿 ⇒ 该 T 是恒绿闸）
    muts = mutation_probe(view)
    for M in muts:
        print(f"[{M['id']}->{M['target']}] turns_red={M['turns_red']} pass={M['pass']}")
    n_mut = sum(1 for M in muts if M["pass"])
    print(f"[MUTATION] {n_mut}/{len(muts)}")

    n_pass_total += 1 if ident["pass"] else 0
    n_pass_total += n_mut
    if not ident["pass"] or n_mut != len(muts):
        verdict_total = "SPEC-GATES-INCOMPLETE"
    print(f"[VERDICT-TOTAL] {verdict_total} ({n_pass_total}/15 = 6 反例 + 骨架前向 + 被测物身份 + 7 变异(M1-M6 + M4b))")

    spec = spec_dict()
    try:
        import psutil  # type: ignore
        rss = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
    except Exception:  # noqa: BLE001
        rss = None
    total_s = round(time.perf_counter() - t_start, 1)
    n_fail = ((6 - n_pass) + (0 if skel.get("pass") else 1)
              + (0 if ident["pass"] else 1) + (len(muts) - n_mut))

    print(f"[SPEC] conflicts={[c['id'] for c in spec['conflicts']]} "
          f"n_declared={spec['derived']['n_declared_fields']} n_used={N_FEATURES} "
          f"max_at_cap={spec['derived']['max_params_at_structure_cap']}")
    print(f"[BUDGET] total={total_s}s peak={None if rss is None else round(rss, 1)}MB "
          f"run_episode=0 world_step={steps_after_construct} training=none")

    if args.no_evidence:
        print(json.dumps({"legs": legs, "skeleton": skel, "mutation": muts,
                          "identity": ident, "spec": spec},
                         ensure_ascii=False, indent=1, default=float))
        return 0 if n_fail == 0 else 2

    # ---- 证据目录（**跑完才建**：本轮无面板，不存在"崩溃丢格"；目录里全是已算出的数）
    head = _git(["rev-parse", "HEAD"])
    wt_sha = _git(["hash-object", "experiments/cw_v1/r2_6_gates.py"])
    blob_sha = _git(["rev-parse", f"{head}:experiments/cw_v1/r2_6_gates.py"], allow_fail=True)
    dirty = _git(["status", "--porcelain"])
    srchash = _src_hash()
    outdir = RESULTS / f"cw_r2_6_spec_{_now_utc()}_{srchash}"
    outdir.mkdir(parents=True, exist_ok=False)
    print(f"[OUT] {outdir.name}")

    t_gates = {"verdict": verdict_total, "n_pass": n_pass_total, "n_expected": 15,
               "legs": legs, "skeleton_forward": skel, "identity": ident,
               "mutation": muts}
    (outdir / "t_gates.json").write_text(
        json.dumps(t_gates, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    (outdir / "spec.json").write_text(
        json.dumps(spec, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    (outdir / "identity.json").write_text(
        json.dumps(ident, ensure_ascii=False, indent=1, default=str), encoding="utf-8")

    summary = {
        "protocol": "CW-POMDP-v1.1", "stage": "CW-R2.6",
        "kind": "TRAIN-INTERFACE-ACCEPTANCE-T1-T6",
        "prereg": "docs/research/CW-R2.6-训练接口验收-预注册-v0.1.md",
        "upstream_card": "docs/research/CW-R2-训练接口-v0.1.md",
        "verdict": verdict_total,
        "legs_pass": {L["id"]: L["pass"] for L in legs},
        "n_caught": n_caught, "n_clean": n_clean, "n_pass": n_pass_total,
        "mutation_turns_red": {M["id"]: M["turns_red"] for M in muts},
        "n_mutation_pass": n_mut,
        "skeleton": {"pass": skel.get("pass"), "n_params": skel.get("describe", {}).get("n_params"),
                     "trained": False, "learned": False},
        "spec": spec,
        "spec_conflicts": spec["conflicts"],
        "objective": OBJECTIVE,
        "identity_pinned_files": {"n": ident["n"], "n_bad": ident["n_bad"],
                                  "pass": ident["pass"], "head": ident["head"]},
        "zero_training_evidence": {
            "n_param_updates": 0,
            "skeleton_trained_flag": False,
            "params_unchanged_after_two_forwards": skel.get(
                "params_unchanged_after_two_forwards"),
            "no_fit_method": not hasattr(NumpyMLP, "fit"),
        },
        "r17": {"triggered": False,
                "why": ("本轮零训练、无面板、无臂比较 ⇒ 没有任何'某臂更好'的结论，"
                        "R17（逐用户 r_i）无对象可报；预注册 §8-7 已写该触发条件")},
        "budget": {"total_s": total_s, "peak_rss_mb": None if rss is None else round(rss, 1),
                   "run_episode": 0, "world_step_after_construct": steps_after_construct,
                   "training": "none"},
        "head": head, "src_hash": srchash,
        "inputs": {str(FIXTURE.relative_to(REPO)): _sha256(FIXTURE),
                   str(MODELS_SRC.relative_to(REPO)): _sha256(MODELS_SRC),
                   str(SELF_SRC.relative_to(REPO)): _sha256(SELF_SRC),
                   str((R25_DIR / "summary.json").relative_to(REPO)): _sha256(
                       R25_DIR / "summary.json")},
    }
    (outdir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    meta = {"command": "python -m experiments.cw_v1.r2_6_gates --tag " + str(args.tag),
            "timestamp_local": datetime.now().astimezone().isoformat(),
            "python": os.sys.version.split()[0], "numpy": np.__version__,
            "head": head, "src_hash": srchash,
            "worktree_sha_r2_6": wt_sha, "blob_sha_r2_6": blob_sha,
            "blob_sha_note": (None if blob_sha else "运行前未入库 ⇒ 提交后补 _POST_COMMIT"),
            "git_dirty_before_run": dirty.splitlines(),
            "protocol": "CW-POMDP-v1.1", "event_pairing": "slot",
            "n_episodes": 0, "world_step": steps_after_construct,
            "total_elapsed_s": total_s, "peak_rss_mb": round(rss, 1) if rss else None,
            "n_fail_total": n_fail, "training": "none"}
    (outdir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    print(f"[DONE] {outdir.name} n_fail={n_fail} total={total_s}s")
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    import sys
    sys.exit(main())
