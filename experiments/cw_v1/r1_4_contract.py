# -*- coding: utf-8 -*-
"""CW-R1.1-2 落盘入口：一步预测/协方差（`pred_gates`）+ 外生事件配对（`pairing_gates`）。

★ 本入口产出的是**协议与实现的验收证据**，不是方法收益结果：
  `meta.json` 里 `kind = "CONTRACT-GATE-NOT-METHOD-RESULT"`，引用时必须带上。
★ 只跑 CPU、只跑小面板（预测验证 6+6 槽、校准 1000 episode × 40 槽、配对 2×256 槽），
  不重跑旧 P3、不训练任何模型、不动旧 `results/`。
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import config as C
from .pairing_gates import run_all as run_pairing
from .pred_gates import run_all as run_pred
from .world import Action

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results" / "cw_v1"


def _src_hash() -> str:
    h = hashlib.sha256()
    for p in sorted((REPO / "experiments" / "cw_v1").glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def _git(args: list[str]) -> str:
    import subprocess

    return subprocess.run(["D:\\devtools\\Git\\cmd\\git.exe", *args],
                          cwd=str(REPO), capture_output=True, text=True).stdout.strip()


# ---------------------------------------------------------------- 变异闸
def mutation_gates() -> tuple[list[dict], int]:
    """**故意把实现弄坏**，验证上面那套闸真的会 FAIL。

    ★ 纪律（台账 ⑫#39 / #64）：一条闸"跑过"不等于"有分辨力"。
      这里做三次单变量破坏，每次都要求**指定那一类检查**从 PASS 翻成 FAIL：

      M1 把 `_draw_meas` 换回"按次数取值"（即 attempt 耦合冒充 slot）
         ⇒ `pairing_gates.Q2` 必须失败。
      M2 把 `_nominal_map_var` 里的 `+σ_w²` 删掉（P̂ 少算过程噪声）
         ⇒ 逐槽复算对账的最大偏差必须 > 1e-9。
      M3 把 `_nominal_map` 里多加一次 `B·u`（复刻撤回分支的"控制量算两遍"）
         ⇒ 零噪声 HOLD 的逐位相等必须失效（max|Δ| > 1e-3）。
    """
    import numpy as np

    import experiments.cw_v1.world as W

    from . import config as C
    from .pairing_gates import _compare_pairing
    from .pred_gates import _world, replay_kalman, zero_noise_cfg

    rows: list[dict] = []

    # ---- M1
    orig_meas = W.CWWorld._draw_meas
    try:
        def _leaky(self, i):                     # noqa: ANN001
            return float(self.rngs["meas"].standard_normal())

        W.CWWorld._draw_meas = _leaky
        res = _compare_pairing("slot")
    finally:
        W.CWWorld._draw_meas = orig_meas
    caught = res["n_diff_z"] > 0
    rows.append({"mutation": "M1 _draw_meas 退回按次数取值",
                 "target": "pairing_gates.Q2", "caught": bool(caught),
                 "evidence": {"n_diff_z": res["n_diff_z"], "n_both_delivered": res["n_both_delivered"]}})

    # ---- M2
    orig_var = W.CWWorld._nominal_map_var
    try:
        def _no_noise(self, i, P):               # noqa: ANN001
            A = float(self.cfg.plant.A[i])
            return A * A * P

        W.CWWorld._nominal_map_var = _no_noise
        cfg = C.CWConfig(plant=C.bounded_plant())
        K = np.asarray([_world(cfg).nominal_gain(i) for i in range(2)])
        wb = _world(cfg, budget=10_000)
        path = []
        for k in range(20):
            wb.step(Action("SEND", 0) if k % 3 == 0 else Action("HOLD"))
            path.append(wb._P_hat.copy())
        rep = replay_kalman(wb.records, 0, cfg, K, cfg.plant.prior_var)
        dP = max(abs(float(path[i][0]) - rep[i][1]) for i in range(len(path)))
    finally:
        W.CWWorld._nominal_map_var = orig_var
    rows.append({"mutation": "M2 _nominal_map_var 删掉 +σ_w²",
                 "target": "pred_gates.P3b", "caught": bool(dP > 1e-9),
                 "evidence": {"max_dP": dP}})

    # ---- M3
    orig_map = W.CWWorld._nominal_map
    try:
        def _double_u(self, i, x, u):            # noqa: ANN001
            return float(self.cfg.plant.A[i]) * x + 2.0 * float(self.cfg.plant.B) * u

        W.CWWorld._nominal_map = _double_u
        cfg0 = zero_noise_cfg()
        w = _world(cfg0, x0=[0.7, -0.4], P0=[0.0, 0.0])
        w.step(Action("HOLD"))
        w.step(Action("HOLD"))
        dX = float(np.max(np.abs(w._x_hat - w._x_true)))
    finally:
        W.CWWorld._nominal_map = orig_map
    rows.append({"mutation": "M3 _nominal_map 多加一次 B·u",
                 "target": "pred_gates.P1", "caught": bool(dX > 1e-3),
                 "evidence": {"max_abs_diff": dX}})

    return rows, sum(0 if r["caught"] else 1 for r in rows)


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="CW-R1.1-2 预测/协方差 + 事件配对验收（落盘）")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--note", default="")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    gp = run_pred()
    gq = run_pairing()
    mut_rows, mut_fail = mutation_gates()
    elapsed = round(time.perf_counter() - t0, 2)

    n_fail = gp.summary()["n_fail"] + gq.summary()["n_fail"] + mut_fail
    print(f"[CW-R1.1-2] pred_gates: {gp.summary()['n_checks']} checks, "
          f"{gp.summary()['n_fail']} fail")
    for r in gp.rows:
        if not r["pass"]:
            print("  FAIL ", r["name"])
    print(f"[CW-R1.1-2] pairing_gates: {gq.summary()['n_checks']} checks, "
          f"{gq.summary()['n_fail']} fail")
    for r in gq.rows:
        if not r["pass"]:
            print("  FAIL ", r["name"])
    print(f"[CW-R1.1-2] 变异闸: {len(mut_rows)} 次破坏, {mut_fail} 次未被抓")
    for r in mut_rows:
        print(f"   {'CAUGHT' if r['caught'] else 'MISSED'}  {r['mutation']} -> {r['target']}")
    print(f"[CW-R1.1-2] total {n_fail} fail / "
          f"{gp.summary()['n_checks'] + gq.summary()['n_checks'] + len(mut_rows)} checks"
          f"  ({elapsed} s)")

    run_id = None
    if not args.no_write:
        now = datetime.now(timezone(timedelta(hours=8)))
        cfg = C.CWConfig(plant=C.bounded_plant())
        run_id = (f"cw_r1_4_contract_{now.strftime('%Y%m%dT%H%M%S')}_"
                  f"{cfg.fingerprint()}_{_src_hash()}")
        out_dir = RESULTS / run_id
        out_dir.mkdir(parents=True, exist_ok=True)

        def _dump(name, obj):
            (out_dir / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2),
                                        encoding="utf-8")

        _dump("pred_gates.json", gp.summary())
        _dump("pairing_gates.json", gq.summary())
        _dump("mutation_gates.json", {"n_mutations": len(mut_rows), "n_missed": mut_fail,
                                      "rows": mut_rows})
        _dump("meta.json", {
            "run_id": run_id,
            "kind": "CONTRACT-GATE-NOT-METHOD-RESULT",
            "protocol": C.CWConfig.PROTOCOL_VERSION,
            "event_pairing_default": cfg.protocol.event_pairing,
            "config": cfg.to_dict(),
            "note": args.note,
            "at_asia_shanghai": now.isoformat(),
            "elapsed_s": elapsed,
            "git_head": _git(["rev-parse", "HEAD"]),
            "git_status_porcelain": _git(["status", "--porcelain"]),
            "src_hash": _src_hash(),
            "disclaimer": (
                "本 run 是协议/实现验收，不是方法结果。"
                "★ 2026-10-22 18:20 之前的所有 v1.1 证据（含 R1-3 v1.1 重跑与 corrections/）"
                "都是 event_pairing='attempt' 口径；本次起默认 'slot'。"
                "两种口径的数字**不得混表**，重放旧数字须显式 event_pairing='attempt'。"
            ),
        })
        print(f"[CW-R1.1-2] 证据落盘 results/cw_v1/{run_id}")

    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
