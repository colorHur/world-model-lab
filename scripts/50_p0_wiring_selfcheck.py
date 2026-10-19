#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""T19 / P0-2 自检：元数据同源 / NaN 拦截 / 预热初态同步。

三条都属"接线"缺陷（R12：参数接了吗），共同点是**不报错、只让数字悄悄变味**：
  D1 预热按定义"每步送真值"，但 `last_obs`/`cost_est` 没跟着 `est` 走
     ⇒ 主循环第一步的估计代价还停在 t=0 的 reset 观测上（UAV 预热 25 步，差 25）。
  D2 `success` 返回 NaN 时，区间检查 `(p<0)|(p>1)` 恒为 False（NaN 比较恒 False）
     ⇒ NaN 静默变成"丢包"，实验只变差、不报错。
  D3 `scripts/42` 的 JSON 元数据写死 `[0.0, 1.0]`，而计算用 `[25, 1]` ⇒ 元数据在说谎。

每条都配阴性对照：恒真/恒假的检查不是检查。

用法
----
    python scripts/50_p0_wiring_selfcheck.py [--json-out results/50_...json]

退出码：0=全过 / 2=有 FAIL。JSON 落 `results/50_p0_wiring_selfcheck.json`（新文件，不覆盖旧证据）。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from wmlab.eval.results_io import latest_result_path, write_selfcheck  # noqa: E402

FAILS: list[str] = []


def check(tag: str, ok: bool, detail: str) -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {tag}: {detail}")
    if not ok:
        FAILS.append(f"{tag}: {detail}")


def _load(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


T12 = _load("scripts/42_t12_hetero_four_arms.py", "t12_for_p02")
T43 = _load("scripts/43_t9_fix_abc_selfcheck.py", "t43_for_p02")   # 复用合成环境与探针
run_multiuser = T43.run_multiuser
CountingEnv, ConstCtrl = T43.CountingEnv, T43.ConstCtrl
NeverServe, Probe = T43.NeverServe, T43.Probe
cost_obs0 = T43.cost_obs0
ConstantSuccess = T43.ConstantSuccess


# ================================================================ D1 预热初态
def section_d1() -> dict:
    print("\n== D1 预热末态：last_obs / cost_est 必须跟上 est ==")
    # 预热 25 步、预算 0（一帧不发）⇒ cost_est 应停在"末次预热观测"= 25，而不是 t=0 的 0
    probe = Probe(NeverServe())
    run_multiuser([CountingEnv()], [ConstCtrl()], probe, ConstantSuccess(0.7),
                  n_episodes=1, seed=0, max_steps=5, budget_total=0, cost_fn=cost_obs0,
                  warmup_steps=25)
    got = float(probe.seen_cost[0][0])
    check("D1 预热 25 步后首步 cost_est == 25", abs(got - 25.0) < 1e-9,
          f"cost_est={got:.1f}（期望 25.0；旧版会是 0.0 = t=0 的 reset 观测）")

    # 阴性对照：不预热 ⇒ 必须还是 0；否则 D1 是"恒等于 25"的假检查
    probe0 = Probe(NeverServe())
    run_multiuser([CountingEnv()], [ConstCtrl()], probe0, ConstantSuccess(0.7),
                  n_episodes=1, seed=0, max_steps=5, budget_total=0, cost_fn=cost_obs0,
                  warmup_steps=0)
    got0 = float(probe0.seen_cost[0][0])
    check("D1- 不预热时仍为 0（阴性对照）", abs(got0 - 0.0) < 1e-9,
          f"cost_est={got0:.1f}（期望 0.0）")
    return {"warm25_first_cost_est": got, "warm0_first_cost_est": got0}


# ================================================================ D2 NaN 拦截
class NaNSuccess:
    def __call__(self, view, alloc):
        return np.full(len(view.slots), float("nan"))


class InfSuccess:
    def __call__(self, view, alloc):
        return np.full(len(view.slots), float("inf"))


def section_d2() -> dict:
    print("\n== D2 成功概率 NaN/inf 必须在入口炸掉 ==")
    v = np.array([np.nan, 0.5])
    blind = not bool(np.any((v < 0.0) | (v > 1.0)))
    check("D2-old 旧区间检查对 NaN 是瞎的", blind,
          f"(nan<0)|(nan>1) 逐元素={((v < 0.0) | (v > 1.0)).tolist()} ⇒ 旧检查放行，NaN 变丢包")

    def run_with(success) -> str:
        try:
            run_multiuser([CountingEnv()], [ConstCtrl()], NeverServe(), success,
                          n_episodes=1, seed=0, max_steps=5, budget_total=5, cost_fn=cost_obs0)
            return "NO-RAISE"
        except ValueError as exc:
            return f"ValueError: {exc}"

    msg_nan = run_with(NaNSuccess())
    check("D2 NaN ⇒ ValueError", msg_nan.startswith("ValueError") and "NaN" in msg_nan,
          msg_nan[:120])
    msg_inf = run_with(InfSuccess())
    check("D2b inf ⇒ ValueError", msg_inf.startswith("ValueError") and "NaN/inf" in msg_inf,
          msg_inf[:120])

    # 阴性对照：正常概率不许被误杀
    msg_ok = run_with(ConstantSuccess(0.9))
    check("D2- 正常概率不触发（阴性对照）", msg_ok == "NO-RAISE", msg_ok[:120])
    return {"nan": msg_nan[:120], "inf": msg_inf[:120], "normal": msg_ok}


# ================================================================ D3 元数据同源
def section_d3() -> dict:
    print("\n== D3 scripts/42 的 absorbed 元数据必须由常量生成 ==")
    got = tuple(float(x) for x in T12.ABSORBED)
    check("D3 常量同源", got == (float(T12.ABSORBED_UAV), float(T12.ABSORBED_CART)),
          f"ABSORBED={got}")
    check("D3- 且等于计算值 (25, 1)", got == (25.0, 1.0), f"ABSORBED={got}（期望 (25.0, 1.0)）")
    # 已入库 JSON 的元数据（若重跑过 42，这里就是新值）
    # ★ P0-3：读"当前版本"（runs/ 最新优先，退回 v1 冻结文件），否则本检查会永远核对旧文件
    p = latest_result_path("42_t12_four_arms_summary")
    meta = None
    if p is not None and p.exists():
        meta = json.loads(p.read_text(encoding="utf-8")).get("config", {}).get("absorbed")
    check("D3b 入库 JSON 的 absorbed == [25.0, 1.0]",
          meta is not None and [float(x) for x in meta] == [25.0, 1.0],
          f"{p.relative_to(ROOT).as_posix() if p else '<缺失>'} "
          f"config.absorbed={meta}（旧版为 [0.0, 1.0]，与计算不符）")
    return {"ABSORBED": list(got), "json_absorbed": meta}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-out", default=None,
                    help="显式路径（默认走 results/selfcheck/50_p0_wiring_selfcheck.json）")
    args = ap.parse_args()
    print(f"numpy {np.__version__}")
    out = {"D1": section_d1(), "D2": section_d2(), "D3": section_d3()}
    out["fails"] = FAILS
    out["pass"] = len(FAILS) == 0
    if args.json_out:
        dest = Path(args.json_out)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        # ★ P0-3（2026-10-19）：自检快照走 results/selfcheck/，判定不变则不重写
        dest = write_selfcheck("50_p0_wiring_selfcheck", out)
    print(f"\n{'='*60}\n{'ALL PASS' if out['pass'] else 'FAILED: ' + '; '.join(FAILS)}")
    print(f"JSON → {dest}")
    return 0 if out["pass"] else 2


if __name__ == "__main__":
    sys.exit(main())
