"""66 · T25：强基线差异表之外的可识别性检查（**不训模型、不产生新轨迹**）

目的
----
候选主张："动作条件模型的不确定性可以识别**有害补偿**，并通过回退降低损害。"
在启动任何训练/扫描之前，先用**已入库证据**回答三个可数的问题：

  I1  样本量：要检出**已经观测到的**预测器差异，需要多少配对 episode？
      现有 90 个配对（3 种子 × 30 episode，同种子同下标 = 同一初值）够不够？
  I2  误差能不能给预测器排序？（16 个 (B,mode,user) 层内，误差 vs J 的 Spearman）
      以及"所有桶误差都更低、J 却更高"的反例在哪几格？
  I3  一个**按预测误差**触发回退的门控，在现有分布下有没有开火机会？
  I4  逐决策的 (obs, action, next_obs) 是否已记录？（没有 ⇒ 反事实不可辨识，只能 SKIP 并打印缺口）

★ 判据先写（见研究空间 `04-实验与代码/2026-10-20-P3-强基线差异表与可识别性-预注册-v0.1.md` §2）：
  空表必须 FAIL 而不是 SKIP；缺桶/桶与 `n_pred_calls` 对不上必须 FAIL；
  三条阴性对照（空表 / 交换 J / 篡改桶 n）必须**按指定方式失败**，否则本轮结论作废。

★ 输入路径写死 run-id，**不随 latest 漂移**（v3.1 §2 的 B 项要求）；输出记录 sha256 与源 run-id。

用法
----
    python scripts/66_identifiability.py                  # 全量，写 outputs/ 与 results/
    python scripts/66_identifiability.py --no-persist     # 只打印，不写任何文件
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wmlab.eval.results_io import current_run_id, write_result  # noqa: E402

# ---- 固定输入：run-id 写死，不用 latest -------------------------------------
INPUTS = {
    "T23b": ("results/runs/20261020-004151-daacb33/56_t23b_compensation_benefit.json", 48),
    "T24": ("results/runs/20261020-025036-62663b3/57_t24_shared_ab.json", 48),
    "P3-0": ("results/runs/20261020-050355-613f4be/58_p3_confirmation.json", 96),
}
# 可部署预测器（`truedyn` 是诊断特权，不进排序；`shared` 是 T24 的共享版）
DEPLOYABLE = {"T23b": ["zoh", "linear", "hybrid"], "T24": ["zoh", "linear", "shared"]}
BUCKETS = ["1", "2-3", ">=4"]
EPISODES = 30
# 双侧 α=0.05、power=0.8 的系数：(1.959964 + 0.841621)^2
Z_POWER = (1.959964 + 0.841621) ** 2

FAILS: list[str] = []


def fail(msg: str) -> None:
    FAILS.append(msg)


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# I0 登记面：固定路径、cell 数、必备键；空表 = FAIL（不是 SKIP）
# ---------------------------------------------------------------------------
def _registry_status(cells: list[dict], expect: int) -> tuple[str, str]:
    """返回 (status, reason)。★ 空表 = FAIL（不是 SKIP）：`bad == 0` 的写法
    会让"根本没比较"和"比较过且一致"同值（v3.1 复验 A 项的根因）。"""
    if not cells:
        return "FAIL", "cell 数为 0（空表冒充证据）"
    if len(cells) != expect:
        return "FAIL", f"cell 数 {len(cells)} != 期望 {expect}"
    bad = [i for i, c in enumerate(cells)
           if not isinstance(c.get("J_per_episode"), list)
           or len(c["J_per_episode"]) != EPISODES]
    if bad:
        return "FAIL", f"{len(bad)} 个 cell 的 J_per_episode 不是 {EPISODES} 条"
    return "PASS", ""


def check_registry() -> dict:
    out: dict = {"rows": [], "status": "PASS"}
    for tag, (rel, expect) in INPUTS.items():
        p = ROOT / rel
        row = {"tag": tag, "path": rel, "exists": p.exists()}
        if not p.exists():
            fail(f"I0 {tag} 路径不存在：{rel}")
            row["status"] = "FAIL"
            out["rows"].append(row)
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        cells = d.get("cells_J") or []
        status, reason = _registry_status(cells, expect)
        if status == "FAIL":
            fail(f"I0 {tag} {reason}")
        row.update(
            sha256=_sha256(p),
            cells=len(cells),
            checks_failed=d.get("checks_failed") or [],
            config={k: d.get("config", {}).get(k) for k in ("budgets", "modes", "seeds",
                                                             "episodes", "predictors",
                                                             "seed_sets")},
            status=status,
            reason=reason,
        )
        out["rows"].append(row)
    out["status"] = "PASS" if all(r["status"] == "PASS" for r in out["rows"]) else "FAIL"
    return out


# ---------------------------------------------------------------------------
# I1 配对功效：同 (budget,mode) 下 pred 与 zoh 逐 (seed,episode_idx) 配对
# ---------------------------------------------------------------------------
def _cell_map(cells: list[dict]) -> dict:
    m: dict = {}
    for c in cells:
        m.setdefault((c["budget"], c["mode"], c["predictor"]), {})[c["seed"]] = c
    return m


def check_power(cells: list[dict], predictors: list[str], tag: str) -> dict:
    cmap = _cell_map(cells)
    pairs_needed: list[dict] = []
    rows: list[dict] = []
    keys_bad = 0
    for (budget, mode, pred), by_seed in sorted(cmap.items()):
        if pred not in predictors or pred == "zoh":
            continue
        if (budget, mode, "zoh") not in cmap:
            fail(f"I1 {tag} 缺 zoh 对照：B={budget} {mode}")
            continue
        zoh = cmap[(budget, mode, "zoh")]
        if set(by_seed) != set(zoh):
            keys_bad += 1
            fail(f"I1 {tag} seed 集合不一致：{pred}@B={budget}/{mode}")
            continue
        for u in (0, 1):
            d: list[float] = []
            for s in sorted(zoh):
                a = by_seed[s]["J_per_episode"]
                b = zoh[s]["J_per_episode"]
                if len(a) != len(b):
                    keys_bad += 1
                    break
                d.extend(a[i][u] - b[i][u] for i in range(len(a)))
            n = len(d)
            mu = sum(d) / n if n else float("nan")
            var = sum((x - mu) ** 2 for x in d) / (n - 1) if n > 1 else float("nan")
            sd = math.sqrt(var) if var == var else float("nan")
            base = sum(zoh[s]["J_per_episode"][i][u] for s in zoh for i in range(EPISODES)) / (
                len(zoh) * EPISODES)
            n_star = (Z_POWER * (sd / mu) ** 2) if (mu and mu == mu and sd == sd and mu != 0) else None
            rows.append({
                "budget": budget, "mode": mode, "predictor": pred, "user": u,
                "n_pairs": n, "delta_mean": mu, "sd_paired": sd,
                "rel_effect": (mu / base if base else None),
                # |rel| < 2% 视为"效应≈0"：此时 n* 是噪声的倒数，报数会误导（R13 同源）
                "effect_negligible": bool(base and abs(mu / base) < 0.02),
                "n_needed_80pct": (math.ceil(n_star) if n_star and n_star == n_star else None),
            })
            if n_star and n_star == n_star and not (base and abs(mu / base) < 0.02):
                pairs_needed.append({"where": f"B={budget}/{mode}/{pred}/u{u}",
                                     "n_needed": math.ceil(n_star), "n_have": n})
    if keys_bad:
        fail(f"I1 {tag} 配对键对不上 {keys_bad} 处")
    worst = max(pairs_needed, key=lambda r: r["n_needed"]) if pairs_needed else None
    return {
        "rows": rows,
        "n_pairs_available": 3 * EPISODES,
        "worst_case": worst,
        "status": "FAIL" if keys_bad else "PASS",
    }


# ---------------------------------------------------------------------------
# I1b 桶一致性：Σ桶 n == n_pred_calls == Σn_age
# ---------------------------------------------------------------------------
def check_bucket_consistency(cells: list[dict]) -> dict:
    """两条**不同基数**的恒等式，必须分开断言（否则就是我犯过的 #49/#52 同型错：
    把"age 全集"和"预测调用点"当成同一次计数）。

      (a) Σ buckets n  == n_pred_calls      —— 预测调用点
      (b) Σ n_age     == alive_steps[user]  —— 存活步数（含 age==0 的步）
    """
    bad: list[str] = []
    n_checked = 0
    for c in cells:
        for u, rec in (c.get("pred_error") or {}).items():
            n_pred = rec.get("n_pred_calls")
            ui = int(u)
            if n_pred is None:
                bad.append(f"B={c['budget']}/{c['mode']}/{c['predictor']}/s{c['seed']}/u{u} 缺 n_pred_calls")
                continue
            sb = sum(rec["buckets"][b]["n"] for b in BUCKETS if b in rec.get("buckets", {}))
            sa = sum(rec.get("n_age", {}).values())
            n_checked += 1
            if sb != n_pred:
                bad.append(f"B={c['budget']}/{c['mode']}/{c['predictor']}/s{c['seed']}/u{u} "
                           f"Σ桶={sb} != n_pred_calls={n_pred}")
            alive = c.get("alive_steps") or []
            if ui < len(alive) and sa != alive[ui]:
                bad.append(f"B={c['budget']}/{c['mode']}/{c['predictor']}/s{c['seed']}/u{u} "
                           f"Σn_age={sa} != alive_steps={alive[ui]}")
    if bad:
        fail(f"I1b 桶一致性失败 {len(bad)} 处（例：{bad[0]}）")
    return {"n_checked": n_checked, "bad": bad[:6], "n_bad": len(bad),
            "status": "FAIL" if bad else "PASS"}


# ---------------------------------------------------------------------------
# I2 / I2b / I3：层内排序、反例、门控触发率
# ---------------------------------------------------------------------------
def _wmean_err(rec: dict) -> float | None:
    bks = rec.get("buckets") or {}
    num = sum(bks[b]["n"] * bks[b]["mean"] for b in BUCKETS if b in bks)
    den = sum(bks[b]["n"] for b in BUCKETS if b in bks)
    return num / den if den else None


def _spearman3(x: list[float], y: list[float]) -> float:
    """n=3 的 Spearman（无并列时）：τ = 1 - Σd²/4。"""
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0] * len(v)
        for pos, i in enumerate(order):
            r[i] = pos
        return r
    rx, ry = rank(x), rank(y)
    d2 = sum((rx[i] - ry[i]) ** 2 for i in range(len(x)))
    return 1.0 - 6.0 * d2 / (len(x) * (len(x) ** 2 - 1))


def analyze_strata(cells: list[dict], predictors: list[str]) -> dict:
    cmap = _cell_map(cells)
    strata: list[dict] = []
    combos = sorted({(c["budget"], c["mode"]) for c in cells})
    for budget, mode in combos:
        for u in (0, 1):
            errs, js, trig = {}, {}, {}
            err_s: dict = {}
            for pred in predictors:
                by_seed = cmap.get((budget, mode, pred))
                if not by_seed:
                    continue
                errs[pred] = sum(_wmean_err(by_seed[s]["pred_error"][str(u)]) or 0.0
                                 for s in by_seed) / len(by_seed)
                js[pred] = sum(by_seed[s]["J_mean"][u] for s in by_seed) / len(by_seed)
                err_s[pred] = {s: {b: by_seed[s]["pred_error"][str(u)]["buckets"][b]["mean"]
                                   for b in BUCKETS} for s in by_seed}
                ref = cmap[(budget, mode, "zoh")]
                trig[pred] = {
                    b: sum(1 for s in by_seed
                           if by_seed[s]["pred_error"][str(u)]["buckets"][b]["mean"]
                           > ref[s]["pred_error"][str(u)]["buckets"][b]["mean"])
                    for b in BUCKETS
                }
            order = [p for p in predictors if p in errs]
            if len(order) < 3:
                fail(f"I2 B={budget}/{mode} 可用预测器不足 3 个")
                continue
            tau = _spearman3([errs[p] for p in order], [js[p] for p in order])
            # 反例（**有序对**）：p 的加权平均误差比 q 低 ≥1%，但 J_p 比 J_q 高 ≥2%。
            # 用加权平均是**故意的**：T23-b 已发表的"hybrid 误差更低但 J +9%"就是按聚合算的；
            # 若改成"逐种子逐桶都更低"，s1 的 age=1 桶会以 0.0014 的差否掉这条已知反例
            # （2026-10-20 实测），闸就会**漏掉已知结论**。strict 版本另行计数，只作补充。
            cex = []
            for p in order:
                for q in order:
                    if p == q:
                        continue
                    strict = all(err_s[p][s][b] <= err_s[q][s][b]
                                 for s in err_s[p] for b in BUCKETS)
                    rel_j = js[p] / js[q]
                    if errs[p] <= 0.99 * errs[q] and rel_j >= 1.02:
                        cex.append({"better_error": p, "worse_error": q,
                                    "err": {p: round(errs[p], 4), q: round(errs[q], 4)},
                                    "J": {p: round(js[p], 4), q: round(js[q], 4)},
                                    "J_rel": round(rel_j, 4),
                                    "strict_all_buckets": bool(strict),
                                    # |ΔJ| < 2% 的反例只是"排序在噪声里翻转"，不能当机制证据
                                    "substantive": True})
            strata.append({
                "budget": budget, "mode": mode, "user": u,
                "err": {k: round(v, 4) for k, v in errs.items()},
                "J": {k: round(v, 4) for k, v in js.items()},
                "tau_err_J": round(tau, 4),
                "counterexamples": cex,
                "trigger_frac": {p: sum(1 for b in BUCKETS if trig[p][b] > 0) / len(BUCKETS)
                                 for p in order},
                # ★ 决定性的一格：**伤害 ≥2% 但按误差门控永不触发** ⇒ 该形状的回退没有机会
                "harm_without_trigger": [
                    {"predictor": p, "J_rel_vs_zoh": round(js[p] / js["zoh"], 4)}
                    for p in order
                    if p != "zoh" and js[p] / js["zoh"] >= 1.02
                    and sum(1 for b in BUCKETS if trig[p][b] > 0) == 0
                ],
            })
    taus = [s["tau_err_J"] for s in strata]
    n_neg = sum(1 for t in taus if t < 0)
    n_cex = sum(len(s["counterexamples"]) for s in strata)
    n_cex_sub = sum(1 for s in strata for c in s["counterexamples"] if c["substantive"])
    harm_no_trig = [(s["budget"], s["mode"], s["user"], h) for s in strata
                    for h in s.get("harm_without_trigger", [])]
    trig_rates = [v for s in strata for p, v in s["trigger_frac"].items() if p != "zoh"]
    return {
        "strata": strata,
        "n_strata": len(strata),
        "n_tau_negative": n_neg,
        "n_counterexamples": n_cex,
        "n_counterexamples_substantive": n_cex_sub,
        "harm_without_trigger": harm_no_trig,
        "trigger_rate_mean": (sum(trig_rates) / len(trig_rates)) if trig_rates else None,
        "trigger_rate_zero_frac": (sum(1 for v in trig_rates if v == 0.0) / len(trig_rates))
        if trig_rates else None,
    }


# ---------------------------------------------------------------------------
# P1 正对照：B == window 时配速必须是 no-op（paced 与 unpaced 逐位相同）
# ---------------------------------------------------------------------------
def check_pacing_noop(cells: list[dict]) -> dict:
    win = max(c["budget"] for c in cells)  # B=200 档；window 在 config 里，此处按"最大预算=窗口"取
    pairs, bad = 0, []
    for c in cells:
        if c["budget"] != win:
            continue
        if c["mode"] != "paced":
            continue
        twin = next((d for d in cells
                     if d["budget"] == c["budget"] and d["mode"] == "unpaced"
                     and d["predictor"] == c["predictor"] and d["seed"] == c["seed"]), None)
        if twin is None:
            bad.append(f"缺 unpaced 孪生：{c['predictor']}/s{c['seed']}")
            continue
        pairs += 1
        if [round(x, 12) for x in twin["J_mean"]] != [round(x, 12) for x in c["J_mean"]]:
            bad.append(f"{c['predictor']}/s{c['seed']}：paced {c['J_mean']} != unpaced {twin['J_mean']}")
    if bad:
        fail(f"P1 B={win}（=窗口）时配速不是 no-op：{bad[0]}")
    return {"budget": win, "pairs": pairs, "bad": bad[:4],
            "status": "FAIL" if bad else "PASS"}


# ---------------------------------------------------------------------------
# I4 反事实可辨识性：逐决策记录是否存在
# ---------------------------------------------------------------------------
def check_counterfactual_logging(cells: list[dict]) -> dict:
    want = ("obs", "state", "action", "next_state", "next_obs", "pred_dist")
    have = sorted({k for c in cells for k in c})
    hits = [k for k in have if any(w in k.lower() for w in want)]
    if hits:
        fail(f"I4 已入库 cell 里出现了逐决策字段 {hits} —— 本轮'不可辨识'的结论作废，需重算")
        return {"status": "FAIL", "cell_keys": have, "hits": hits}
    return {
        "status": "SKIP",
        "reason": "已入库 cell 只有 episode 级聚合，没有逐决策 (obs, action, next_obs) 记录",
        "cell_keys": have,
        "missing": ["每决策：送达 obs 向量 / age / 实际动作 / 预测的下一步分布 / 真实下一步"],
        "consequence": "动作效应（同一状态不同动作的差）在本证据下**不可辨识**；"
                       "任何'不确定性→回退'的门控都不能只靠现有表验证，必须新增日志与随机化",
    }


# ---------------------------------------------------------------------------
# 阴性对照
# ---------------------------------------------------------------------------
def negative_controls(cells: list[dict], predictors: list[str]) -> dict:
    out: dict = {}

    # N1 空表：I0 的判据必须给 FAIL（不是 SKIP、不是 PASS）
    st, reason = _registry_status([], INPUTS["T23b"][1])
    out["N1_empty_table"] = {"status": st, "reason": reason,
                             "detected": st == "FAIL"}
    if st != "FAIL":
        fail("N1 空表没有被判 FAIL（闸无分辨力）")

    # N2 交换 J：把反例那一对 (better_error, worse_error) 的 J 对调 ⇒ 该层反例必须消失
    base = analyze_strata(cells, predictors)
    cex_strata = [s for s in base["strata"] if s["counterexamples"]]
    if not cex_strata:
        out["N2_swap_J"] = "SKIP(基线里没有反例，无从对照)"
        fail("N2 无从构造：基线里没有反例（与 T23-b 已发表的反例矛盾）")
        return out
    s0 = cex_strata[0]
    c0 = s0["counterexamples"][0]
    pred_p, pred_q = c0["better_error"], c0["worse_error"]
    mut = copy.deepcopy(cells)
    pmap = {(c["budget"], c["mode"], c["seed"]): c for c in mut if c["predictor"] == pred_p}
    qmap = {(c["budget"], c["mode"], c["seed"]): c for c in mut if c["predictor"] == pred_q}
    for k, cp in pmap.items():
        cq = qmap.get(k)
        if cq is None:
            continue
        cp["J_per_episode"], cq["J_per_episode"] = cq["J_per_episode"], cp["J_per_episode"]
        cp["J_mean"], cq["J_mean"] = cq["J_mean"], cp["J_mean"]
    mut_res = analyze_strata(mut, predictors)
    still = [s for s in mut_res["strata"]
             if s["budget"] == s0["budget"] and s["mode"] == s0["mode"] and s["user"] == s0["user"]
             and s["counterexamples"]]
    out["N2_swap_J"] = {
        "swapped_at": f"B={s0['budget']}/{s0['mode']}/u{s0['user']}: J({pred_p})<->J({pred_q})",
        "counterexample_before": c0,
        "counterexample_after": still[0]["counterexamples"] if still else None,
        "disappeared": not still,
    }
    if still:
        fail("N2 交换 J 之后反例仍然存在 ⇒ I2b 读的不是 J 字段")

    # N3 篡改桶 n：I1b 必须 FAIL
    mut2 = copy.deepcopy(cells)
    u0 = list(mut2[0]["pred_error"].keys())[0]
    mut2[0]["pred_error"][u0]["buckets"]["1"]["n"] += 1
    saved = list(FAILS)
    FAILS.clear()
    r3 = check_bucket_consistency(mut2)
    FAILS.clear()
    FAILS.extend(saved)
    out["N3_bucket_tamper_detected"] = r3["status"] == "FAIL"
    if r3["status"] != "FAIL":
        fail("N3 桶 n 被篡改后 I1b 仍 PASS ⇒ 桶一致性无分辨力")
    return out


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-persist", action="store_true", help="只打印，不写 outputs/ 与 results/")
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()
    t0 = time.time()

    reg = check_registry()
    print(f"[I0] 登记面：{reg['status']}")
    for r in reg["rows"]:
        print(f"     {r['tag']:5s} cells={r.get('cells')} sha256={str(r.get('sha256'))[:16]}… "
              f"{r['status']}")
    if not FAILS:
        d23 = json.loads((ROOT / INPUTS["T23b"][0]).read_text(encoding="utf-8"))
        d24 = json.loads((ROOT / INPUTS["T24"][0]).read_text(encoding="utf-8"))
        cells23, cells24 = d23["cells_J"], d24["cells_J"]

        i1b = check_bucket_consistency(cells23)
        print(f"[I1b] 桶一致性：{i1b['status']}（核对 {i1b['n_checked']} 个 (cell,user)）")

        p1 = check_pacing_noop(cells23)
        print(f"[P1] B={p1['budget']} 配速 no-op 正对照：{p1['status']}（{p1['pairs']} 对）")

        i1 = check_power(cells23, DEPLOYABLE["T23b"], "T23b")
        print(f"[I1] 配对功效：{i1['status']}，可用配对 {i1['n_pairs_available']}")
        for r in i1["rows"]:
            print(f"     B={r['budget']:3d} {r['mode']:7s} {r['predictor']:7s} u{r['user']} "
                  f"Δμ={r['delta_mean']:+.4f} σ_d={r['sd_paired']:.4f} "
                  f"rel={r['rel_effect']:+.3f} n*={r['n_needed_80pct']}")
        print(f"     最坏情况：{i1['worst_case']}")

        i2 = analyze_strata(cells23, DEPLOYABLE["T23b"])
        print(f"[I2] 层内 Spearman：{i2['n_strata']} 层，τ<0 有 {i2['n_tau_negative']} 层")
        print(f"[I2b] 反例 {i2['n_counterexamples']} 个：")
        for s in i2["strata"]:
            for c in s["counterexamples"]:
                print(f"     B={s['budget']}/{s['mode']}/u{s['user']} "
                      f"{c['better_error']} 误差 {c['err'][c['better_error']]} "
                      f"<= {c['worse_error']} {c['err'][c['worse_error']]}，"
                      f"但 J {c['J'][c['better_error']]} > {c['J'][c['worse_error']]} "
                      f"(×{c['J_rel']})")
        print(f"[I3] 按误差触发回退：平均触发桶比例 {i2['trigger_rate_mean']}, "
              f"全不触发的层占比 {i2['trigger_rate_zero_frac']}")

        i2b24 = analyze_strata(cells24, DEPLOYABLE["T24"])
        print(f"[I2/T24] 复算：{i2b24['n_strata']} 层，τ<0 有 {i2b24['n_tau_negative']} 层，"
              f"反例 {i2b24['n_counterexamples']} 个")

        i4 = check_counterfactual_logging(cells23)
        print(f"[I4] 反事实可辨识性：{i4['status']} —— {i4.get('reason')}")

        nc = negative_controls(cells23, DEPLOYABLE["T23b"])
        print(f"[NC] {json.dumps(nc, ensure_ascii=False)[:400]}")
    else:
        i1b = i1 = i2 = i2b24 = i4 = nc = p1 = None

    payload = {
        "config": {
            "inputs": {k: v[0] for k, v in INPUTS.items()},
            "deployable": DEPLOYABLE,
            "episodes_per_cell": EPISODES,
            "alpha_power_coef": round(Z_POWER, 4),
            "note": "探索性测量：判据在看数后写，见预注册 §0；不得当确认性证据",
        },
        "I0": reg, "I1b": i1b, "P1": p1, "I1": i1, "I2": i2, "I2_T24": i2b24, "I4": i4,
        "negative_controls": nc,
        "checks_failed": list(FAILS),
        "elapsed_s": round(time.time() - t0, 2),
    }
    print(f"\n{'ALL CHECKS PASS' if not FAILS else 'FAILED: ' + '; '.join(FAILS[:4])}")
    print(f"总耗时 {payload['elapsed_s']}s")

    if not args.no_persist and not FAILS:
        (ROOT / "outputs").mkdir(exist_ok=True)
        (ROOT / "outputs" / "66_identifiability.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        dest = write_result("66_identifiability", payload, run_id=args.run_id or current_run_id())
        print(f"[evidence] {dest.relative_to(ROOT).as_posix()}")
    return 0 if not FAILS else 2


if __name__ == "__main__":
    sys.exit(main())
