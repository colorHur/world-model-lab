# -*- coding: utf-8 -*-
"""CW-R2.0 落盘入口：R2 立项卡 + 模型接口 + **可比性指纹**的机器验收。

预注册：`docs/research/CW-R2.0-R2立项与接口卡-预注册-v0.1.md`（§8 记跑后的修订）

三问（预注册 §1）
-----------------
`Q1` 「同规划器 + 同预算」能否被**一个指纹**钉住（改任何一项都自动暴露）？
`Q2` 模型接口与信息权限能否被**机器**钉死，且这些闸**有分辨力**（能 FAIL）？
`Q3` R2 面板在本机 CPU 跑得起吗（用已测多点给区间，不用单点外推）？

★ 本轮**不训练、不跑 episode**：`run_episode` = 0，`CWWorld.step` = 0，
  `CWWorld` 只**构造**一次（为取真实视图的键集，闸 `A4`）。
★ 被测物 `plan_ref/plan_dp/world/runner/config/policies/mech` **一字节不动**。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from . import gates as G
from . import plan_dp, plan_ref
from .config import mainline_config
from .mech import KAPPA_MIN  # 只为确认模块可导入（本轮不用它的值）
from .model_api import (FIELD_PERMISSION, FORBIDDEN_NAMES, NULL_PAIR_SPEC, OBJECTIVE,
                        VIEW_TOP_KEYS, ComparabilitySpec, collect_keys, deploy,
                        deploy_diagnostic, planner_signature, require_view, scan_source,
                        spec_from_config)
from .r1_14_exactbase import RESULTS, _git, _src_hash
from .r2_models import BeliefReadout, PriorOnly, TruthDiagnostic
from .runner import run_episode
from .streams import make_episode_rngs
from .world import CWWorld, SchedulerView

REPO = Path(__file__).resolve().parents[2]
MODELS_SRC = REPO / "experiments" / "cw_v1" / "r2_models.py"
CARD = REPO / "docs" / "research" / "CW-R2.0-R2立项与接口卡-预注册-v0.1.md"

#: 预算三点（预注册 §0.3 已列；这里**现读**，不抄）
BUDGET_POINTS = (
    ("R1.12", RESULTS / "cw_r1_12_ackscale_20261024T020453_92d6a8ccd0bc1a26"),
    ("R1.14", RESULTS / "cw_r1_14_exactbase_20261024T063157_c835f84ff417e9d2"),
    ("R1.16", RESULTS / "cw_r1_16_n10_20261024T105416_fc9f25e16d2c6307"),
)
R174 = RESULTS / "cw_r1_7_4_tails_20261023T130519_f11ffdefc06c0201_232550606b401065"

#: 预注册 §0.3 里**手抄进卡**的数字 —— `A9` 逐字节对账（防"记忆外推"）
CARD_PINNED = {
    "cv_paced": "0.583391467891342",
    "cv_nopace": "0.9286036223814766",
    "cost_paced": "0.11067633547679674",
    "cost_nopace": "0.1767897716156801",
    "aoi_paced": "16.0",
    "aoi_nopace": "128.5",
    "tails_paced_cost_median": "0.07871360448861517",
}

#: 计划中的 R2.1 面板规模（写死，`A8` 拿它算预算区间）
R2_PANEL = {"models": 2, "plant_arms": 2, "episodes": 10}
R2_N_CELLS = R2_PANEL["models"] * R2_PANEL["plant_arms"] * R2_PANEL["episodes"]

HYPOTHESIS = {
    "id": "H-R2-1",
    "statement": ("同规划器、同预算、同信息集下，参数 < 1e4、CPU 训练 < 30 min 的小学习模型，"
                  "其 `cost_true_per_step` 中位数与精确名义模型（`belief_readout`）之差 < 5%，"
                  "且逐用户 `r_i` 全部落在 [0.95, 1.05]。"),
    "falsified_if": ("配对差中位 ≥ `floor_R2`（零控，见 `NULL_PAIR_SPEC`）"
                     "且 二项符号检验 `p < 0.05`（要求 ≥9/10 同号）"),
    "inconclusive_if": "|Δ| < `floor_R2` ⇒ 面板信噪比不足；**不许**写'够用'或'不够用'",
    "forbidden_inference": "学习模型输了 **不得** 推出'世界模型无用'（R1.14：协议+plant 限定范围）",
    "needs": ["training（未授权/未开始）", "R2.1 面板（未跑）"],
}


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


# ==================================================================== 判据自证（零 episode）
def controls(view: SchedulerView, spec: ComparabilitySpec) -> dict:
    """`C1–C5`，答案全部**构造已知**。不过就不看 `A1–A10`。"""
    out: dict = {}

    # ---- C1 合法视图必须被接受
    try:
        require_view(view)
        out["C1"] = {"pass": True, "detail": "真实视图通过"}
    except Exception as e:  # noqa: BLE001
        out["C1"] = {"pass": False, "detail": f"真实视图被拒：{e!r}"}

    # ---- C2 指纹确定 + 单字段可分辨
    fp1, fp2 = spec.fingerprint(), spec.fingerprint()
    mut = spec_from_config(mainline_config(), budget_total=spec.budget_total + 1,
                           split=spec.split, episodes=spec.episodes,
                           seeds=spec.seeds, models=spec.models)
    out["C2"] = {"pass": bool(fp1 == fp2 and mut.fingerprint() != fp1),
                 "detail": {"fp": fp1, "mut_fp": mut.fingerprint()}}

    # ---- C3 特权臂必须被 deploy 拒绝
    td = TruthDiagnostic(x=np.zeros(2), log_h=np.zeros(2))
    try:
        deploy(td)
        c3 = False
        note = "deploy 接受了特权臂（不该）"
    except PermissionError:
        c3 = True
        note = "PermissionError"
    out["C3"] = {"pass": c3, "detail": note}

    # ---- C4 干净源码不报、脏源码必报（扫描器两个方向都要对）
    clean = "def predict(self, view):\n    return view.t\n"
    dirty = "def predict(self, view):\n    return view.__dict__['cost_true_by_user']\n"
    sc, sd = scan_source(clean), scan_source(dirty)
    out["C4"] = {"pass": bool(not (sc["names"] or sc["attrs"] or sc["imports"])
                              and (sd["names"] or sd["attrs"] or sd["imports"])),
                 "detail": {"clean_hits": len(sc["names"]) + len(sc["attrs"]) + len(sc["imports"]),
                            "dirty_hits": sd["names"] + sd["attrs"] + sd["imports"]}}

    # ---- C5 目标函数自检器：干净表 0 违规、注入 `ku` 必被抓
    def _violations(obj: dict) -> list[str]:
        bad = []
        fbd = set(obj["forbidden"])
        for k in ("primary",):
            if obj[k] in fbd:
                bad.append(f"{k}={obj[k]}")
        for k in obj["secondary"]:
            if k in fbd:
                bad.append(f"secondary={k}")
        for need in ("ku", "kf", "share"):
            if need not in fbd:
                bad.append(f"forbidden 缺 {need}")
        if "per_user_r_i" not in obj["secondary"]:
            bad.append("secondary 缺逐用户 r_i（R17）")
        return bad

    injected = json.loads(json.dumps(OBJECTIVE))
    injected["primary"] = "ku"
    out["C5"] = {"pass": bool(not _violations(OBJECTIVE) and _violations(injected)),
                 "detail": {"clean": _violations(OBJECTIVE), "injected": _violations(injected)}}

    keys = ("C1", "C2", "C3", "C4", "C5")
    out["all_pass"] = all(out[k]["pass"] for k in keys)
    out["_keys"] = list(keys)
    return out


# ==================================================================== 闸
class _Impostor(SchedulerView):
    """子类替身：`type(x) is SchedulerView` 为假 ⇒ 必须被拒。"""

    def __init__(self, pub: dict) -> None:
        super().__init__(pub)
        self.extra_handle = object()


def _permissive(o) -> object:
    """**活体负控**：只查 `hasattr(o, 't')` 的旧式守卫。"""
    if not hasattr(o, "t"):
        raise TypeError("no .t")
    return o


def gates(view: SchedulerView, spec: ComparabilitySpec) -> dict:
    chk: list[dict] = []

    def add(gid: str, ok: bool, detail=None) -> None:
        chk.append({"id": gid, "ok": bool(ok), "detail": detail})

    # ---------------- A1 类型守卫
    det = {"real": "accepted"}
    ok = True
    try:
        require_view(view)
    except Exception as e:  # noqa: BLE001
        ok = False
        det["real"] = repr(e)
    for name, obj in (("dict", dict(view._pub)),
                      ("namespace", SimpleNamespace(t=0, telemetry=[])),
                      ("subclass_impostor", _Impostor(view._pub))):
        try:
            require_view(obj)
            det[name] = "ACCEPTED(bad)"
            ok = False
        except TypeError:
            det[name] = "rejected"
    add("A1", ok, det)

    # ---------------- A1b 活体负控：旧式守卫**必须**放替身进来，否则 A1 无分辨力
    #   ★ 预注册 §0/§2 只要求"同一个替身被接受"。`dict` 那一条**算术上不可能**：
    #     `hasattr(dict, "t")` 恒 False（属性守卫看不到 dict 的键），要求它通过等于
    #     断言一个不可达的量 ⇒ 首跑 FAIL 是**判据自伤**（记录见 §8-1）。
    accepted = []
    for name, obj in (("dict", dict(view._pub)), ("subclass_impostor", _Impostor(view._pub))):
        try:
            _permissive(obj)
            accepted.append(name)
        except TypeError:
            pass
    add("A1b", "subclass_impostor" in accepted,
        {"permissive_accepts": accepted,
         "dict_rejected_by_attr_guard": "dict" not in accepted,
         "why": "属性守卫放宽后必须放替身进来，才证明 A1 的收紧有分辨力；"
                "`dict` 无属性 ⇒ 任何属性守卫都拒，不参与判定"})

    # ---------------- A2 输出校验
    from .model_api import ModelOutput
    n = int(view.n_users)
    good = ModelOutput(np.zeros(n), np.ones(n)).validate(n)
    neg = {}
    cases = {
        "len_mismatch": ModelOutput(np.zeros(n + 1), np.ones(n + 1)),
        "nan": ModelOutput(np.array([np.nan] * n), np.ones(n)),
        "P_zero": ModelOutput(np.zeros(n), np.zeros(n)),
        "P_negative": ModelOutput(np.zeros(n), np.full(n, -1.0)),
        "two_dim": ModelOutput(np.zeros((n, 1)), np.ones((n, 1))),
        "non_numeric": ModelOutput(["a"] * n, np.ones(n)),  # type: ignore[list-item]
    }
    for k, mo in cases.items():
        try:
            mo.validate(n)
            neg[k] = "ACCEPTED(bad)"
        except (ValueError, TypeError) as e:
            neg[k] = type(e).__name__
    add("A2", all(v != "ACCEPTED(bad)" for v in neg.values()) and good is not None,
        {"good": "ok", "negatives": neg})

    # ---------------- A3 部署路径 AST 闸
    src = MODELS_SRC.read_text(encoding="utf-8")
    s = scan_source(src)
    add("A3", not (s["names"] or s["attrs"] or s["imports"]),
        {"file": MODELS_SRC.name, "names": s["names"], "attrs": s["attrs"],
         "imports": s["imports"], "skipped_classes": s["skipped"]})

    # ---------------- A3b 扫描器两个方向（合成源码）
    synt = {
        "str_truth": "def predict(self, view):\n    return view.__dict__['cost_true_by_user']\n",
        "attr_future": "def predict(self, view):\n    return float(self._z_meas[0, 0])\n",
        "import_world": "from .world import CWWorld\n\ndef predict(self, view):\n    return None\n",
        "import_abs": "import experiments.cw_v1.world as W\n\ndef predict(self, view):\n    return None\n",
        "clean": "def predict(self, view):\n    return float(view.t)\n",
    }
    res = {k: scan_source(v) for k, v in synt.items()}
    flagged = {k: bool(r["names"] or r["attrs"] or r["imports"]) for k, r in res.items()}
    add("A3b", all(flagged[k] for k in ("str_truth", "attr_future", "import_world", "import_abs"))
        and not flagged["clean"], flagged)

    # ---------------- A4 权限表完备 + 视图不含真值
    keys = set(collect_keys(view._pub))
    declared = set()
    for path in FIELD_PERMISSION:
        declared.add(path.split(".", 1)[1] if "." in path else path)
    extra = sorted(keys - declared)
    missing_top = sorted(set(VIEW_TOP_KEYS) - keys)
    truth_in_view = sorted(k for k in keys if k in FORBIDDEN_NAMES)
    never_entries = sorted(k for k, v in FIELD_PERMISSION.items() if v["visibility"] == "never")
    forbid_cov = sorted(set(FORBIDDEN_NAMES) - {k.split(".", 1)[-1] for k in never_entries})
    add("A4", not extra and not missing_top and not truth_in_view and not forbid_cov,
        {"view_keys": sorted(keys), "undeclared": extra, "missing_top": missing_top,
         "truth_in_view": truth_in_view, "forbidden_uncovered": forbid_cov,
         "n_declared_paths": len(FIELD_PERMISSION)})

    # ---------------- A5 特权臂不得进部署表
    td = TruthDiagnostic(x=np.zeros(n), log_h=np.zeros(n))
    try:
        deploy(td)
        a5_refuse = False
    except PermissionError:
        a5_refuse = True
    ok_deploy = deploy(BeliefReadout()) is not None
    diag = deploy_diagnostic(td)
    dout = diag.predict(view)
    add("A5", bool(a5_refuse and ok_deploy and dout.privileged
                   and dout.label.startswith("DIAG-")),
        {"deploy_refused": a5_refuse, "normal_deployed": ok_deploy,
         "diag_label": dout.label, "diag_privileged": dout.privileged})

    # ---------------- A6 指纹
    fp = spec.fingerprint()
    fp2 = spec.fingerprint()
    a6a = fp == fp2
    import dataclasses as _dc
    muts = {
        "plant.B": _dc.replace(spec, plant={**spec.plant, "B": float(spec.plant["B"]) + 0.01}),
        "channel.kappa": _dc.replace(spec, channel={**spec.channel, "kappa": float(spec.channel["kappa"]) + 0.1}),
        "protocol.horizon": _dc.replace(spec, protocol={**spec.protocol, "horizon": int(spec.protocol["horizon"]) + 1}),
        "planner.h_plan": _dc.replace(spec, planner={**spec.planner, "h_plan": int(spec.planner["h_plan"]) + 1}),
        "budget_total": _dc.replace(spec, budget_total=int(spec.budget_total) + 1),
        "split": _dc.replace(spec, split=spec.split + "x"),
        "episodes": _dc.replace(spec, episodes=tuple(spec.episodes) + (999,)),
        "seeds": _dc.replace(spec, seeds=tuple(spec.seeds) + (999,)),
        "models": _dc.replace(spec, models=tuple(spec.models) + ("zzz",)),
        "metric.primary": _dc.replace(spec, metric={**spec.metric, "primary": "x"}),
        "protocol_version": _dc.replace(spec, protocol_version="CW-POMDP-vX"),
    }
    changed = {k: (v.fingerprint() != fp) for k, v in muts.items()}
    add("A6a", a6a, {"fp": fp, "fp_again": fp2})
    add("A6b", all(changed.values()), {"n_mutations": len(changed),
                                       "unchanged": [k for k, v in changed.items() if not v]})

    import inspect
    sig = inspect.signature(plan_ref.PlanRef.__init__)
    sig_defaults = {k: sig.parameters[k].default for k in planner_signature() if not k.startswith("_")}
    a6c = (spec.planner["h_plan"] == sig_defaults["h_plan"]
           and sig_defaults["h_plan"] == 5
           and spec.planner["_tail_module_default"] == str(plan_ref.TAIL_DEFAULT)
           and spec.planner["_dp_grid_module_default"] == str(getattr(plan_dp, "GRID_DEFAULT", "?")))
    add("A6c", a6c, {"planner_in_spec": {k: spec.planner[k] for k in ("h_plan", "m_samples", "pacing", "tail")},
                     "signature_defaults": sig_defaults,
                     "tail_module_default": spec.planner["_tail_module_default"],
                     "dp_grid_module_default": spec.planner["_dp_grid_module_default"]})

    # ---------------- A7 目标函数表
    def _violations(obj: dict) -> list[str]:
        bad, fbd = [], set(obj["forbidden"])
        if obj["primary"] in fbd:
            bad.append(f"primary={obj['primary']}")
        for k in obj["secondary"]:
            if k in fbd:
                bad.append(f"secondary={k}")
        for need in ("ku", "kf", "share"):
            if need not in fbd:
                bad.append(f"forbidden 缺 {need}")
        if "per_user_r_i" not in obj["secondary"]:
            bad.append("secondary 缺逐用户 r_i（R17）")
        return bad

    inj = json.loads(json.dumps(OBJECTIVE))
    inj["primary"] = "share"
    add("A7", not _violations(OBJECTIVE) and bool(_violations(inj)),
        {"clean": _violations(OBJECTIVE), "injected": _violations(inj),
         "primary": OBJECTIVE["primary"]})

    # ---------------- A8 预算（多点区间）
    pts = []
    for tag, d in BUDGET_POINTS:
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        cells = json.loads((d / "cells.json").read_text(encoding="utf-8"))
        n_cells = len(cells)
        tot = float(meta["total_elapsed_s"])
        pts.append({"tag": tag, "n_cells": n_cells, "total_s": tot,
                    "s_per_cell": round(tot / n_cells, 4), "peak_mb": meta.get("peak_rss_mb")})
    spc = [p["s_per_cell"] for p in pts]
    lo, hi = min(spc), max(spc)
    spread = hi / lo
    budget = {"n_cells": R2_N_CELLS, "s_lo": round(R2_N_CELLS * lo, 1),
              "s_hi": round(R2_N_CELLS * hi, 1), "spread": round(spread, 3)}
    add("A8", len(pts) >= 2 and spread >= 1.2,
        {"points": pts, "spread": round(spread, 3), "r2_panel": R2_PANEL, "budget": budget})

    # ---------------- A9 配速残差锚点（现读 + 与卡逐字节对账）
    bench = json.loads((R174 / "bench.json").read_text(encoding="utf-8"))
    g8a = bench["G8a"]["detail"]
    got = {"cv_paced": repr(float(g8a["paced"]["cv"])),
           "cv_nopace": repr(float(g8a["nopace"]["cv"])),
           "cost_paced": repr(float(g8a["paced"]["cost_true"])),
           "cost_nopace": repr(float(g8a["nopace"]["cost_true"])),
           "aoi_paced": repr(float(g8a["paced"]["aoi"])),
           "aoi_nopace": repr(float(g8a["nopace"]["aoi"])),
           "tails_paced_cost_median": repr(float(
               json.loads((R174 / "panel_tails.json").read_text(encoding="utf-8")
                          )["paced"]["cost_median"]))}
    mismatch = {k: {"card": CARD_PINNED[k], "evidence": got[k]}
                for k in CARD_PINNED if CARD_PINNED[k] != got[k]}
    card_text = CARD.read_text(encoding="utf-8")
    in_card = {k: (v in card_text) for k, v in CARD_PINNED.items()}
    tails = json.loads((R174 / "panel_tails.json").read_text(encoding="utf-8"))
    arm_med = {k: tails[k].get("cost_median") for k in tails}
    add("A9", not mismatch and all(in_card.values()),
        {"evidence": got, "mismatch": mismatch, "found_in_card": in_card,
         "three_arm_cost_median": arm_med,
         "note": "另两臂 `cost_median` 本轮现读并原样记录，卡里未引用（预注册 §0.3）"})

    # ---------------- A10 协议闸回归
    t0 = time.perf_counter()
    reg = G.run_all()
    add("A10", int(reg["n_fail"]) == 0,
        {"n_checks": reg["n_checks"], "n_fail": reg["n_fail"],
         "elapsed_s": round(time.perf_counter() - t0, 2)})

    return {"checks": chk, "n_checks": len(chk),
            "n_fail": sum(1 for c in chk if not c["ok"])}


# ==================================================================== 资产清单
def _defines_name(src: str, name: str) -> bool:
    """AST 级判定：该源码是否**定义/导入**了 `name`。

    ★ 不能用 `name in src`：`streams.py` 的 docstring 里就写着"为什么不用
      `ChannelAdapter`"——字符串命中会把它误判成依赖（首跑真错，§8-2）。
    """
    import ast as _ast
    for node in _ast.walk(_ast.parse(src)):
        if isinstance(node, _ast.ClassDef) and node.name == name:
            return True
        if isinstance(node, _ast.ImportFrom) and any(a.name == name for a in node.names):
            return True
        if isinstance(node, _ast.Import) and any(a.name.split(".")[-1] == name for a in node.names):
            return True
    return False


def asset_inventory() -> dict:
    """哪些现成件可复用、哪些不可 —— 每条都带**机器判据**。"""
    srcs = {p.name: p.read_text(encoding="utf-8", errors="ignore")
            for p in (REPO / "experiments" / "cw_v1").glob("*.py")}
    ca = [k for k, s in srcs.items() if k != Path(__file__).name and _defines_name(s, "ChannelAdapter")]
    return {
        "reusable": [
            {"item": "plan_ref.PlanRef（PROBE/SEND/HOLD + 预算 + 续接）", "ok": hasattr(plan_ref, "PlanRef")},
            {"item": "plan_dp.DPTail（`(P0,P1,b)` 后向 DP 尾部）", "ok": hasattr(plan_dp, "DPTail")},
            {"item": "world.SchedulerView（公开快照 + 白名单历史）", "ok": hasattr(SchedulerView, "history")},
            {"item": "world.PUBLIC_HISTORY_FIELDS（12 字段白名单）", "ok": bool(VIEW_TOP_KEYS)},
            {"item": "mech.KAPPA_MIN（机制量常量，仅历史引用）", "ok": float(KAPPA_MIN) > 0},
            {"item": "runner.run_episode（`predictor='nominal'` 唯一取值）", "ok": callable(run_episode)},
            {"item": "streams.make_episode_rngs（6 条子流，CRN 基础）", "ok": callable(make_episode_rngs)},
        ],
        "not_reusable": [
            {"item": "P3 的 `ChannelAdapter`", "ok": not ca, "defining_files": ca,
             "why": "v4 交接已判不兼容（时序与信息权限不一致）；判据 = AST 里无定义/无 import"},
            {"item": "`predictor='exact'` 分支", "ok": True,
             "why": "2026-10-22 已撤回（把控制量与噪声各算两遍）"},
            {"item": "`oracle_known.ORACLE-KNOWN` 当上界", "ok": True,
             "why": "R1.1 撤回上界名称；只作真值辅助贪心诊断"},
        ],
    }


# ==================================================================== 主流程
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="CW-R2.0 R2 立项卡与接口（零 episode）")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--ctrl-only", action="store_true", help="只跑判据自证，不写证据")
    args = ap.parse_args(argv)

    t_start = time.perf_counter()

    # ---- 视图（只构造，不 step）
    cfg = mainline_config()
    rngs, sids = make_episode_rngs(cfg.root_seed, "dev", 900)
    w = CWWorld(cfg, rngs, sids)
    view = w.view()
    n_steps = int(w.t)          # 构造后应为 0
    spec = spec_from_config(cfg, budget_total=int(cfg.protocol.horizon),
                            split="dev", episodes=tuple(range(900, 910)),
                            seeds=(2001, 2002, 2003, 2004),
                            models=("belief_readout", "prior_only"))

    # ---- CTRL 先行
    t0 = time.perf_counter()
    ctrl = controls(view, spec)
    ctrl_s = round(time.perf_counter() - t0, 2)
    kk = ctrl.pop("_keys")
    print(f"[CTRL] all_pass={ctrl['all_pass']} " +
          " ".join(f"{k}={ctrl[k]['pass']}" for k in kk) + f" ({ctrl_s}s)")
    if not ctrl["all_pass"]:
        print("[ABORT] 判据自证未过 ⇒ 按预注册 §2 不看 A1–A10")
        return 3
    if args.ctrl_only:
        print(json.dumps(ctrl, ensure_ascii=False, indent=1, default=float))
        return 0

    # ---- 闸
    t1 = time.perf_counter()
    gate = gates(view, spec)
    gate_s = round(time.perf_counter() - t1, 2)
    print(f"[GATE] n_checks={gate['n_checks']} n_fail={gate['n_fail']} ({gate_s}s)")
    for c in gate["checks"]:
        print(f"  {'PASS' if c['ok'] else 'FAIL'}  {c['id']}")

    head = _git(["rev-parse", "HEAD"])
    wt_sha = _git(["hash-object", "experiments/cw_v1/r2_0_card.py"])
    blob_sha = _git(["rev-parse", f"{head}:experiments/cw_v1/r2_0_card.py"], allow_fail=True)
    dirty = _git(["status", "--porcelain"])
    srchash = _src_hash()
    outdir = RESULTS / f"cw_r2_0_card_{_now_utc()}_{srchash}"
    outdir.mkdir(parents=True, exist_ok=False)
    print(f"[OUT] {outdir.name}")

    # ---- 落盘
    spec_out = spec.as_dict()
    chk_items = gate["checks"]
    budget_item = next(c for c in chk_items if c["id"] == "A8")["detail"]["budget"]
    spec_out["budget"] = budget_item
    spec_out["null_pair"] = NULL_PAIR_SPEC
    spec_out["hypothesis"] = HYPOTHESIS
    spec_out["asset_inventory"] = asset_inventory()
    spec_out["info_permission"] = FIELD_PERMISSION
    spec_out["objective"] = OBJECTIVE
    spec_out["residual_anchor"] = next(c for c in chk_items if c["id"] == "A9")["detail"]
    spec_out["n_run_episode"] = 0
    spec_out["n_world_steps"] = n_steps
    spec_out["n_world_constructed"] = 1

    (outdir / "spec.json").write_text(
        json.dumps(spec_out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    (outdir / "gates.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    (outdir / "ctrl.json").write_text(
        json.dumps(ctrl, ensure_ascii=False, indent=1, default=float), encoding="utf-8")

    try:
        import psutil  # type: ignore
        rss = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
    except Exception:
        rss = None

    #: 资产清单里若有 `ok=False`，整轮不算通过（它会指向一个真的缺口）
    inv_bad = [r["item"] for r in spec_out["asset_inventory"]["reusable"] if not r["ok"]] + \
              [r["item"] for r in spec_out["asset_inventory"]["not_reusable"] if not r["ok"]]
    n_fail = gate["n_fail"] + (0 if ctrl["all_pass"] else 1) + len(inv_bad)

    meta = {
        "command": "python -m experiments.cw_v1.r2_0_card --tag " + str(args.tag),
        "timestamp_local": datetime.now().astimezone().isoformat(),
        "python": os.sys.version.split()[0], "numpy": np.__version__,
        "kind": "R2-PROJECT-CARD-NOT-METHOD-RESULT",
        "head": head, "src_hash": srchash,
        "worktree_sha_r2_0": wt_sha, "blob_sha_r2_0": blob_sha,
        "blob_sha_note": ("运行前文件未提交 ⇒ None（不记参数串）；提交后真实 blob 见 _POST_COMMIT.json"
                          if blob_sha is None else "运行前已在 HEAD 上"),
        "git_dirty_before_run": dirty.splitlines(),
        "protocol": "CW-POMDP-v1.1", "spec_fingerprint": spec.fingerprint(),
        "n_run_episode": 0, "n_world_steps": n_steps, "n_world_constructed": 1,
        "ctrl_elapsed_s": ctrl_s, "gate_elapsed_s": gate_s,
        "total_elapsed_s": round(time.perf_counter() - t_start, 2),
        "peak_rss_mb": round(rss, 1) if rss else None,
        "n_checks_gates": gate["n_checks"], "n_fail_gates": gate["n_fail"],
        "inventory_bad": inv_bad,
    }
    (outdir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"[DONE] {outdir.name} n_fail={n_fail} total={meta['total_elapsed_s']}s "
          f"peak_rss={meta['peak_rss_mb']}MB fp={spec.fingerprint()}")
    print(json.dumps({"spec_fingerprint": spec.fingerprint(),
                      "budget": budget_item, "inventory_bad": inv_bad},
                     ensure_ascii=False, indent=1))
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    import sys
    sys.exit(main())
