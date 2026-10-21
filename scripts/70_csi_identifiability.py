"""70 · T29：CSI 备选的**可识别性**判定（不训模型、不跑新轨迹、不改库）

要回答的一句话
--------------
"下行 CSI 不确定时，测得的**补偿收益**（`Linear` 相对 `ZOH`）是否翻转？"
在开任何实验之前先问：**CSI 的不确定性有没有通道能影响这些臂的动作？**

本脚本用四层证据回答，全部是结构与纯计算，**不产生任何新轨迹**：

  C1 消费面（AST）：全仓库里谁 **读** `xxx.csi`？按 owner 分类（truth / decision / test / tool）
  C2 签名面（inspect）：预测器与分配器的参数里有没有 `csi` / `View` 的入口
  C3 变异测试（运行时）：只改 `slot.csi`、其它字段逐位相同 ⇒ 动作必须逐位不变；
       阴性对照 `MaxChannel` **必须变**（否则这条测试没有分辨力）
  C4 定量门槛（纯计算）：唯一读 csi 的臂 `MaxChannel`，在估计噪声 σ 下会选错多少、
       选错时 q 损失多少；用 `Φ(−|Δ|/(σ√2))` 做**解析代验**（R12：先算再看）

★ 判词（预注册 §1 先写死）：
  `NO-CHANNEL`  = 决策侧通道不存在 ⇒ 该问题在现架构下**不可识别**（不是"效应为零"）；
  `HAS-CHANNEL` = 至少一个决策侧消费点 ⇒ 转下一轮做敏感性实验（须另立预注册）。

★ `NO-CHANNEL` **不等于**"CSI 误差没影响"：它说的是"无论 σ 多大，action 逐位不变"。
   这是恒等式；因此**也不能**用"样本不够"来解释。

用法
----
    python scripts/70_csi_identifiability.py                # 全量（约 3 s），写 results/
    python scripts/70_csi_identifiability.py --no-persist    # 只打印，不写任何文件
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import inspect
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wmlab.eval.allocators import (  # noqa: E402
    MaxAge, MaxChannel, NormalizedAge, Paced, RoundRobin, View,
)
from wmlab.eval.multiuser import UserSlot  # noqa: E402
from wmlab.eval.predictors import Gated, Linear, ZOH  # noqa: E402
from wmlab.eval.results_io import write_result  # noqa: E402

FAILS: list[str] = []
CHECKS: list[dict] = []


def check(cid: str, name: str, ok: bool, detail: str = "") -> bool:
    CHECKS.append({"id": cid, "name": name, "status": "PASS" if ok else "FAIL",
                   "detail": detail})
    if not ok:
        FAILS.append(f"{cid} {name}: {detail}")
    return ok


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# =============================================================== C1 消费面
def _scan_source(text: str) -> list[tuple[int, str]]:
    """返回 [(lineno, "Class.method")]，只收 `Attribute(attr='csi')` 的读取点。"""
    tree = ast.parse(text)
    out: list[tuple[int, str]] = []

    def walk(node, stack: list[str]) -> None:
        for child in ast.iter_child_nodes(node):
            name = None
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                name = child.name
            new_stack = stack + [name] if name else stack
            if isinstance(child, ast.Attribute) and child.attr == "csi":
                out.append((int(child.lineno), ".".join(new_stack)))
            walk(child, new_stack)

    walk(tree, [])
    return out


def _scan_file(path: Path) -> list[tuple[int, str]]:
    return _scan_source(path.read_text(encoding="utf-8"))


#: 预注册 §2-C1 的白名单（**跑之前写死**，不许看结果再补）
def _classify(rel: str, owner: str) -> str:
    if rel == "wmlab/eval/multiuser.py":
        return "truth"                    # SINRSuccess 读真值 / runner 上下文
    if rel == "wmlab/eval/allocators.py" and owner == "MaxChannel.__call__":
        return "decision"                 # 唯一决策侧消费者（T12 对照臂）
    if rel.startswith("scripts/41_"):
        return "test"                     # E6 接线探针
    if rel == "scripts/70_csi_identifiability.py":
        return "tool"                     # 本脚本自身（只读）
    return "unknown"


def c1_consumers() -> list[dict]:
    rows: list[dict] = []
    files = sorted((ROOT / "wmlab" / "eval").glob("*.py")) + sorted((ROOT / "scripts").glob("*.py"))
    for p in files:
        rel = p.relative_to(ROOT).as_posix()
        for lineno, owner in _scan_file(p):
            rows.append({"file": rel, "line": lineno, "owner": owner,
                         "cls": _classify(rel, owner)})
    return rows


# =============================================================== C2 签名面
def c2_signatures() -> list[dict]:
    pred_cls = [("ZOH", ZOH), ("Linear", Linear), ("Gated", Gated)]
    alloc_cls = [("RoundRobin", RoundRobin), ("MaxAge", MaxAge),
                 ("NormalizedAge", NormalizedAge), ("MaxChannel", MaxChannel)]
    rows: list[dict] = []
    for nm, cls in pred_cls:
        for meth in ("__init__", "observe_delivery", "predict"):
            fn = getattr(cls, meth, None)
            if fn is None:
                continue
            params = inspect.signature(fn).parameters
            bad = [k for k in params if "csi" in k.lower()]
            # predictor 不允许拿到 View（View 是唯一携带 slot.csi 的对象）
            bad_view = [k for k in params if k.lower() == "view"]
            rows.append({"target": f"predictor {nm}.{meth}", "params": list(params),
                         "csi_params": bad, "view_params": bad_view,
                         "ok": not bad and not bad_view})
    for nm, cls in alloc_cls:
        fn = getattr(cls, "__call__", None)
        params = inspect.signature(fn).parameters
        bad = [k for k in params if "csi" in k.lower()]
        # allocator **必须**拿 View（它要靠 age/alive 决策），只禁止显式的 csi 入参
        rows.append({"target": f"allocator {nm}.__call__", "params": list(params),
                     "csi_params": bad, "view_params": [],
                     "ok": not bad})
    return rows


# =============================================================== C3 变异测试
def _mk_view(csi: tuple[float, float], t: int = 2, budget: int = 100) -> View:
    slots = tuple(
        UserSlot(idx=i, alive=True, age=1 + 2 * i,
                 last_obs=np.full(2, 0.1 * (i + 1), dtype=np.float32),
                 n_frames=1, n_attempts=2, csi=float(csi[i]),
                 cost_est=0.5 + 0.2 * i)
        for i in range(2)
    )
    return View(t=int(t), slots=slots, budget_left=int(budget),
                budget_total=int(budget), privileged=None)


ALLOC_FACTORIES = {
    "rr": lambda: RoundRobin(),
    "age": lambda: MaxAge(),
    "na": lambda: NormalizedAge(deadlines=(1.0, 1.0)),
    "paced-rr": lambda: Paced(RoundRobin(), budget=100, window=200),
    "paced-age": lambda: Paced(MaxAge(), budget=100, window=200),
    "paced-na": lambda: Paced(NormalizedAge(deadlines=(1.0, 1.0)), budget=100, window=200),
    # 阴性对照（见预注册 N1）：唯一读 csi 的臂，**必须**被这条测试抓出来
    "__max-channel__": lambda: MaxChannel(),
}


def _alloc_signature(factory) -> tuple:
    a = factory()
    v = _mk_view((0.90, 0.55))
    out = a(v, np.random.default_rng(0))
    return (tuple(bool(x) for x in out.served), str(out.reason))


def c3_allocators() -> list[dict]:
    rows = []
    for nm, fac in ALLOC_FACTORIES.items():
        s0 = _alloc_signature(fac)                              # csi = (0.90, 0.55)
        f1 = fac()
        out1 = f1(_mk_view((0.55, 0.90)), np.random.default_rng(0))   # csi 互换
        s1 = (tuple(bool(x) for x in out1.served), str(out1.reason))
        same = s0 == s1
        rows.append({"arm": nm, "csi_a": [0.90, 0.55], "csi_b": [0.55, 0.90],
                     "served_a": list(s0[0]), "served_b": list(s1[0]),
                     "reason_a": s0[1], "reason_b": s1[1], "invariant": bool(same)})
    return rows


def c3_predictors() -> list[dict]:
    """预测器侧：`predict` 的参数里本来就没有 csi ⇒ 结构性恒真。
    仍然跑一遍，是为了在**未来有人给 predictor 塞 View** 时立刻亮红灯。"""
    rows = []
    for nm, mk in (("ZOH", lambda: ZOH()), ("Linear", lambda: Linear()),
                   ("Gated", lambda: Gated(threshold=8))):
        out = []
        for _ in range(2):
            p = mk()
            p.observe_delivery(0, np.array([0.0, 0.0], dtype=np.float32), age_before=0)
            p.observe_delivery(0, np.array([0.2, 0.1], dtype=np.float32), age_before=2)
            out.append(p.predict(0, age=2, actions_since=[np.array([1]), np.array([1])]))
        same = np.array_equal(out[0], out[1])
        rows.append({"predictor": nm, "est": [float(x) for x in out[0]],
                     "invariant": bool(same),
                     "note": "结构性恒真（predict 参数里无 csi/View）"})
    return rows


# =============================================================== C4 定量门槛
SIGMAS = (0.0, 0.01, 0.02, 0.05, 0.10, 0.20)
Q_REF = 0.7
KAPPA = -math.log(Q_REF)


def _phi(x: np.ndarray) -> np.ndarray:
    """标准正态 CDF，**向量化**实现。

    ★ 为什么不用 `math.erfc` 逐点：540k 个点 × 12 档 ⇒ 6.5M 次 Python 调用（约 4 s，纯属浪费）。
      numpy **没有** `erfc`/`erf`（那是 scipy.special 的），所以这里用 Abramowitz–Stegun
      26.2.17（Zelen & Severo）的多项式近似，**绝对误差 < 7.5e-8** —— 本判据的容差是 0.02，
      差 6 个数量级，不影响结论。
    """
    z = np.asarray(x, dtype=float)
    a = np.abs(z)
    t = 1.0 / (1.0 + 0.2316419 * a)
    poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937
                                                      + t * (-1.821255978 + t * 1.330274429))))
    pdf = np.exp(-0.5 * a * a) / math.sqrt(2.0 * math.pi)
    upper = pdf * poly                      # = P(Z > a)
    return np.where(z >= 0.0, 1.0 - upper, upper)


def _q(c: np.ndarray) -> np.ndarray:
    return np.exp(-KAPPA / np.maximum(c, 1e-6))


def c4_gate(n_ep: int, seeds: list[int], t12, rng_seed: int = 20261021) -> list[dict]:
    """返回每个 σ 一行。`flip_raw` 与解析式对账；`flip_clip` 是物理上合理的估计器输出。

    ★ 性能：csi 轨迹按 seed **只算一次**然后缓存。第一版在 σ 循环里重算 ⇒ 122 个 seed × 6 档
      = 732 次 `make_csi`（每次 30×200 的 Python 循环）⇒ 15.9 s，纯属浪费；缓存后约 1.5 s。
    """
    rng = np.random.default_rng(rng_seed)
    cache = {int(s): np.asarray(t12.make_csi(int(s), int(n_ep)), dtype=float) for s in seeds}
    # 无噪声的"可避免差"：选最优 vs 选次优的 q 差 —— dq 的分母
    gap_all = np.concatenate([(_q(np.maximum(cache[s][..., 0], cache[s][..., 1]))
                               - _q(np.minimum(cache[s][..., 0], cache[s][..., 1]))).ravel()
                              for s in seeds])
    delta_all = np.concatenate([(cache[s][..., 0] - cache[s][..., 1]).ravel() for s in seeds])
    n = delta_all.size
    gap_med = float(np.median(gap_all))
    rows = []
    for sig in SIGMAS:
        if sig == 0.0:
            rows.append({"sigma": 0.0, "n_pairs": int(n), "flip_raw": 0.0, "flip_clip": 0.0,
                         "analytic": 0.0, "abs_err": 0.0, "dq_median": 0.0, "dq_mean": 0.0, "dq_min": 0.0,
                         "dq_over_gap": 0.0, "abs_delta_flip_median": 0.0,
                         "abs_delta_all_median": float(np.median(np.abs(delta_all)))})
            continue
        flip_raw = flip_clip = 0.0
        dq: list[float] = []
        absd_flip: list[float] = []
        for s in seeds:
            c0 = cache[s][..., 0].reshape(-1)
            c1 = cache[s][..., 1].reshape(-1)
            e0 = rng.standard_normal(c0.size)
            e1 = rng.standard_normal(c1.size)
            d = c0 - c1
            dh = d + sig * (e0 - e1)
            flip_raw += float(np.mean(np.sign(dh) != np.sign(d)))
            h0 = np.clip(c0 + sig * e0, 0.0, 1.0)
            h1 = np.clip(c1 + sig * e1, 0.0, 1.0)
            bad = (h0 > h1) != (c0 > c1)
            flip_clip += float(np.mean(bad))
            if np.any(bad):
                best = np.where(c0 >= c1, c0, c1)[bad]
                # ★ 选人按**估计**（h），但代价按**真实** csi（c）结算 —— 这才是"选错了损失多少"。
                #   第一版写成 `np.where(h0 >= h1, h0, h1)` ⇒ 拿估计值当真实值，
                #   dq 被系统性低估，σ=0.01 时甚至出现负值（选"错"的人反而算成更好的）。
                #   负的 dq 在物理上不可能（选错必然损失 ≥ 0）⇒ 是判据实现错，不是发现。
                chosen = np.where(h0 >= h1, c0, c1)[bad]
                dq.extend((_q(best) - _q(chosen)).tolist())
                absd_flip.extend(np.abs(d)[bad].tolist())
        flip_raw /= len(seeds)
        flip_clip /= len(seeds)
        rows.append({
            "sigma": sig, "n_pairs": int(n),
            "flip_raw": flip_raw, "flip_clip": flip_clip,
            "analytic": float(np.mean(_phi(-np.abs(delta_all) / (sig * math.sqrt(2.0))))),
            "abs_err": 0.0,   # 下面填
            "dq_median": float(np.median(dq)) if dq else 0.0,
            "dq_mean": float(np.mean(dq)) if dq else 0.0,
            "dq_min": float(np.min(dq)) if dq else 0.0,
            "dq_over_gap": (float(np.median(dq)) / gap_med) if dq and gap_med > 0 else 0.0,
            "abs_delta_flip_median": float(np.median(absd_flip)) if absd_flip else 0.0,
            "abs_delta_all_median": float(np.median(np.abs(delta_all))),
        })
        rows[-1]["abs_err"] = abs(rows[-1]["flip_raw"] - rows[-1]["analytic"])
    return rows


# =============================================================== C6 注入自检
def c6_injection() -> dict:
    """**内存注入**（不碰磁盘）：在 `allocators.py` 的源码副本里给 `MaxAge.__call__`
    插一行读 `csi`，验证 C1 会把它判成 `unknown` ⇒ C1-a FAIL。

    ★ 为什么必须做：C1 全 PASS 只说明"没扫到"，**没说明"扫得到"**（教训 #39 同族）。
      区分"闸有分辨力"与"闸跑过了"的唯一办法是喂它一个必须被抓的输入。
    ★ 为什么在内存里做：原地改库文件会让工作树变脏；如果中途异常退出，
      下一个人会看到一份"看起来正常、其实被改过"的源码。
    """
    rel = "wmlab/eval/allocators.py"
    src = (ROOT / rel).read_text(encoding="utf-8")
    needle = "        i = _pick_alive(view, lambda j: view.slots[j].age)\n"
    if needle not in src:
        return {"ok": False, "detail": "注入锚点不存在（allocators.py 变了 ⇒ 先修注入器，别跳过）"}
    injected = src.replace(needle, "        _inj = view.slots[0].csi  # INJECT\n" + needle, 1)
    rows = [{"file": rel, "line": ln, "owner": own, "cls": _classify(rel, own)}
            for ln, own in _scan_source(injected)]
    unknown = [r for r in rows if r["cls"] == "unknown"]
    ok = (len(unknown) == 1 and unknown[0]["owner"] == "MaxAge.__call__")
    return {"ok": bool(ok), "n_rows": len(rows), "unknown": unknown,
            "detail": "注入点被识别为 unknown" if ok else f"注入未被抓：unknown={unknown}"}


# =============================================================== main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--no-persist", action="store_true")
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    t0 = time.time()
    t12 = _load(ROOT / "scripts" / "42_t12_hetero_four_arms.py", "t12_mod70")

    # ---------------- C1 ----------------
    cons = c1_consumers()
    decision = [r for r in cons if r["cls"] == "decision"]
    unknown = [r for r in cons if r["cls"] == "unknown"]
    check("C1-a", "消费面：无无法分类的 csi 读取点", not unknown,
          f"unknown={unknown}")
    check("C1-b", "消费面：decision 类严格等于 {MaxChannel.__call__}",
          len(decision) == 1 and decision[0]["owner"] == "MaxChannel.__call__",
          f"decision={decision}")
    check("C1-c", "消费面：扫描器确实扫到了 truth 类（不是空扫）",
          any(r["cls"] == "truth" for r in cons),
          f"truth={[ (r['file'], r['line']) for r in cons if r['cls']=='truth']}")

    # ---------------- C2 ----------------
    sig = c2_signatures()
    check("C2", "签名面：预测器无 csi/View 入口，分配器无 csi 显式入参",
          all(r["ok"] for r in sig),
          f"bad={[r['target'] for r in sig if not r['ok']]}")

    # ---------------- C3 ----------------
    a_rows = c3_allocators()
    main_arms = [r for r in a_rows if not r["arm"].startswith("__")]
    ctrl = [r for r in a_rows if r["arm"].startswith("__")]
    check("C3-a", "变异测试：主线分配器在 csi 互换下动作逐位不变",
          all(r["invariant"] for r in main_arms),
          f"changed={[r['arm'] for r in main_arms if not r['invariant']]}")
    check("C3-b", "阴性对照：MaxChannel 必须被这条测试抓出来（否则无分辨力）",
          bool(ctrl) and all(not r["invariant"] for r in ctrl),
          f"ctrl={[(r['arm'], r['invariant']) for r in ctrl]}")
    p_rows = c3_predictors()
    check("C3-c", "变异测试：预测器估计不变（结构性恒真，作回归哨兵）",
          all(r["invariant"] for r in p_rows),
          f"changed={[r['predictor'] for r in p_rows if not r['invariant']]}")

    # ---------------- C4 ----------------
    dev_rows = c4_gate(args.episodes, list(range(0, 32)), t12)
    conf_rows = c4_gate(args.episodes, list(range(100, 190)), t12)
    bad_err = [f"{s['sigma']}:{s['abs_err']:.4f}" for s in (dev_rows + conf_rows)
               if s["sigma"] > 0 and s["abs_err"] > 0.02]
    check("C4-a", "σ>0 时实测反转率与 Φ(−|Δ|/(σ√2)) 相差 ≤ 0.02（解析代验）",
          not bad_err, f"超差={bad_err}")
    check("C4-b", "阴性对照：σ=0 时反转率恰好 0",
          all(r["flip_raw"] == 0.0 and r["flip_clip"] == 0.0 for r in dev_rows + conf_rows
              if r["sigma"] == 0.0), "σ=0 行")
    bad_neg = [f"σ={r['sigma']}:{r['dq_min']:.3e}" for r in dev_rows + conf_rows
               if r["dq_min"] < -1e-12]
    check("C4-c", "选错时的 q 损失必须 ≥ 0（负值 ⇒ 判据取错了对象）",
          not bad_neg, f"负值={bad_neg}")

    # ---------------- C6 注入自检（C1 的分辨力） ----------------
    inj = c6_injection()
    check("C6", "注入自检：给 MaxAge 插一行读 csi ⇒ C1 必须判它为 unknown", inj["ok"],
          inj["detail"])

    # ---------------- 判词 ----------------
    has_channel = (not unknown) and (len(decision) != 1
                                     or decision[0]["owner"] != "MaxChannel.__call__")
    verdict = "HAS-CHANNEL" if has_channel else "NO-CHANNEL"
    # 独立于判词的一条：补偿对比的两端（Linear/ZOH）是不是都不读 csi
    comp_clean = all(r["invariant"] for r in p_rows if r["predictor"] in ("ZOH", "Linear"))
    check("C5", "补偿对比两端（ZOH/Linear）不读 csi ⇒ 补偿收益对该不确定性恒等",
          comp_clean and not has_channel, f"verdict={verdict}")

    payload = {
        "task": "T29", "verdict": verdict,
        "all_pass": not FAILS,
        "n_fail": len(FAILS), "fails": FAILS,
        "checks": CHECKS,
        "c1_consumers": cons,
        "c2_signatures": sig,
        "c3_allocators": a_rows,
        "c3_predictors": p_rows,
        "c6_injection": inj,
        "c4_dev": dev_rows,
        "c4_confirm": conf_rows,
        "inputs": {
            "physical.py": _sha256(ROOT / "wmlab" / "eval" / "physical.py"),
            "multiuser.py": _sha256(ROOT / "wmlab" / "eval" / "multiuser.py"),
            "allocators.py": _sha256(ROOT / "wmlab" / "eval" / "allocators.py"),
            "predictors.py": _sha256(ROOT / "wmlab" / "eval" / "predictors.py"),
            "scripts/42": _sha256(ROOT / "scripts" / "42_t12_hetero_four_arms.py"),
        },
        "config": {"episodes": args.episodes, "sigmas": list(SIGMAS),
                   "q_ref": Q_REF, "dev_seeds": "0-31", "confirm_seeds": "100-189",
                   "note": "csi 轨迹由 scripts/42.make_csi 纯函数重算，不消耗 runner 的 rng，"
                           "不是新实验、不产生新主指标"},
        "elapsed_s": round(time.time() - t0, 3),
    }

    print(f"[70] T29 CSI 可识别性｜判词 {verdict}｜检查 {len(CHECKS) - len(FAILS)}/{len(CHECKS)} PASS"
          f"｜{payload['elapsed_s']} s")
    for r in cons:
        print(f"  csi 读取点  {r['file']}:{r['line']}  owner={r['owner'] or '-'}  → {r['cls']}")
    for r in a_rows:
        print(f"  C3 alloc  {r['arm']:<16} 不变={r['invariant']}  "
              f"{r['served_a']} vs {r['served_b']}")
    for tag, rows in (("dev", dev_rows), ("conf", conf_rows)):
        for s in rows:
            print(f"  C4 {tag} σ={s['sigma']:<5} flip_raw={s['flip_raw']:.4f} "
                  f"flip_clip={s['flip_clip']:.4f} 解析={s['analytic']:.4f} "
                  f"|err|={s['abs_err']:.4f} dq中位={s['dq_median']:.4f} "
                  f"dq/gap={s['dq_over_gap']:.3f} "
                  f"|Δ|翻转中位={s['abs_delta_flip_median']:.3f}"
                  f"(全体{s['abs_delta_all_median']:.3f})")
    if FAILS:
        print("[FAILS]")
        for f in FAILS:
            print("  -", f)

    if not args.no_persist:
        p = write_result("70_csi_identifiability", payload, run_id=args.run_id)
        print(f"[WROTE] {p.relative_to(ROOT).as_posix()}")
    return 0 if not FAILS else 2


if __name__ == "__main__":
    raise SystemExit(main())
