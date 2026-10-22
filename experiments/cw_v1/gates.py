# -*- coding: utf-8 -*-
"""CW-R0 验收闸：首轮实验卡 §「反例与验收」1–8 项逐条实现。

★ 每条判据都写成「什么错误能让它 FAIL」可回答的形式（台账 ⑫#39）。
  只打印 PASS 而没有反例的检查，等于没检查。

★ 本文件只读运行，**不写** `results/`。结果由调用方（`scripts/cw_r0.py`）落盘。
"""

from __future__ import annotations

import numpy as np

from .config import CWConfig, default_config
from .policies import (AlwaysProbe, AlwaysSend, CsiSensitive, ILLEGAL_TruthGreedy,
                       PacedAge, ProbeThenSend)
from .runner import _truth_series, check_identities, run_episode
from .streams import STREAM_NAMES, cross_split_independence_probe, make_episode_rngs, \
    stream_uniqueness_report
from .world import Action, BudgetExhausted, CWWorld, SchedulerView, action_cost, action_space, \
    p_success_from_log_h, steady_state_success


class Gate:
    """收集 (名字, 通过?, 证据)。"""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def check(self, name: str, ok: bool, detail) -> bool:
        self.rows.append({"name": name, "pass": bool(ok),
                          "detail": detail if isinstance(detail, (str, int, float, bool, dict, list))
                          else str(detail)})
        return bool(ok)

    @property
    def all_pass(self) -> bool:
        return all(r["pass"] for r in self.rows)

    def summary(self) -> dict:
        return {"n_checks": len(self.rows), "n_fail": sum(1 for r in self.rows if not r["pass"]),
                "all_pass": self.all_pass, "rows": self.rows}


def _budget(cfg: CWConfig, ratio: float) -> int:
    return cfg.protocol.budget(ratio)


# ============================================================ 1 信息不泄漏
def counterexample_1(cfg: CWConfig, g: Gate) -> None:
    """给策略输入副本加入未公开 true_h/true_x 或未来 CSI 键，接口拒绝；
    仅修改调度器无法看到的真值，当前动作不应变化；后续真实送达可以变化。
    """
    rngs, sids = make_episode_rngs(cfg.root_seed, "dev", 1)
    w = CWWorld(cfg, rngs, sids)
    w.budget_total = _budget(cfg, 0.5)
    v = w.view()

    # (a) 接口对真值封闭：任何真值/未来键都拒绝
    forbidden = ["_x_true", "_log_h_true", "true_x", "true_h", "_w", "world", "env",
                 "future_csi", "_future", "packet_rng"]
    escaped = []
    for key in forbidden:
        try:
            getattr(v, key)
            escaped.append(key)
        except AttributeError:
            pass
    g.check("1a 接口拒绝真值与未来键", not escaped,
            {"probed": forbidden, "escaped": escaped})

    # (b) SchedulerView 的唯一成员是 _w，且它不出现在 telemetry/history 里
    slots = SchedulerView.__slots__
    tel = v.telemetry
    hist = v.history
    leaks = []
    for row in tel:
        for bad in ("x_true", "true_x", "log_h", "true_h", "p_true"):
            if bad in row:
                leaks.append(f"telemetry.{bad}")
    for row in hist:
        for bad in ("p_true", "cost_true", "csi_reading", "true_norm"):
            if bad in row:
                leaks.append(f"history.{bad}")
    g.check("1b telemetry/history 无真值字段", not leaks,
            {"slots": list(slots), "leaks": leaks})

    # (c) 只改真值 ⇒ 动作逐位不变
    #
    # ★★ 这一条改了两版，第二次才找到正确形式（教训写进仓库 README，不写进这里充数）。
    #
    #    错误形式：「改真信道 ⇒ 整条 256 步动作序列完全相同」。
    #    为什么错：这是个 POMDP。改真信道会改变**交付结果**（同一动作、不同 ACK），
    #      而交付结果是可部署的合法信息；一旦 ACK 不同，后续 AoI 就不同，动作**理应变**。
    #      实测：纯 AoI 策略（完全不看信道）在两种世界下也会分叉 —— 正是这个原因。
    #      要求"整条序列不变"，等于要求仿真里不能有信道，判据自相矛盾。
    #    ⇒ 正确形式：**首次分歧之前，两个世界的动作必须完全一致**，
    #      且首个分歧点必须能由"此前交付结果不同"解释。
    #      此外再加一条更强的检查：在**不给 ACK 任何影响**的窗口里（t 小于首次交付），
    #      两个世界必须逐位一致 —— 见 1d-prime。
    use_ack = cfg.channel.use_ack_for_csi
    pol = CsiSensitive(theta=3, use_csi=True)
    pol_aoi = PacedAge(theta=3)

    def run_pair(policy, perturb):
        rngs, sids = make_episode_rngs(cfg.root_seed, "dev", 1)
        w = CWWorld(cfg, rngs, sids, rho=0.85)
        w.budget_total = _budget(cfg, 0.5)
        acts, acks = [], []
        for k in range(int(cfg.protocol.horizon)):
            w.truth_override = _truth_series(perturb, k, w.n_users) if perturb else None
            a = policy(w.view())
            acts.append((a.kind, a.user))
            rec = w.step(a)
            acks.append(rec.ack)
        return acts, acks

    def first_divergence(perturb, policy):
        a0, k0 = run_pair(policy, None)
        a1, k1 = run_pair(policy, perturb)
        for i, (x, y) in enumerate(zip(a0, a1)):
            if x != y:
                return i, a0[:i], a1[:i], k0[:i], k1[:i]
        return None, a0, a1, k0, k1

    for tag, policy in (("CSI 敏感", pol), ("纯 AoI（不看信道）", pol_aoi)):
        i, a0, a1, k0, k1 = first_divergence("h", policy)
        prefix_same = a0[:i] == a1[:i] if i is not None else True
        # 首个分歧之前，是否存在"交付结果不同"的可部署差异？若没有，则动作不该变
        ack_diff_before = (k0 != k1) if i is not None else False
        ok = prefix_same and (i is None or ack_diff_before)
        g.check(f"1d[{tag}] 改真信道后：首次动作分歧之前动作完全一致，且分歧前确有交付差异可解释",
                ok,
                {"first_divergence_at": i,
                 "prefix_identical": prefix_same,
                 "ack_diff_explains": ack_diff_before,
                 "use_ack_for_csi": use_ack})

    # 1d-prime：更强 —— 在**第一次交付发生之前**，两世界必须逐位一致（此时无任何可部署差异）
    i, a0, a1, k0, k1 = first_divergence("h", pol)
    if i is not None:
        n_before_first_ack = next((j for j, k in enumerate(k0) if k is not None), len(k0))
        g.check("1d′ 首次交付之前，改真信道不改变任何动作（无合法信息可用时动作必须不变）",
                i >= n_before_first_ack,
                {"first_divergence_at": i, "first_delivery_at": n_before_first_ack})
    else:
        g.check("1d′ 首次交付之前，改真信道不改变任何动作", True, {"first_divergence_at": None})

    # (c-x) 只改真**状态** x ⇒ 动作必须**整条**逐位不变。
    #   为什么这条比 1d 严格：x 只通过"送达的观测值"进入可部署信息，
    #   而"是否送达"由信道决定、与 x 无关 ⇒ 改 x 不改变任何 ACK ⇒ 动作链条完全不变。
    #   （1d 里的真信道则通过 ACK 合法地影响信息，只能要求"首次分歧前一致"。）
    def run_x(policy, perturb):
        rngs, sids = make_episode_rngs(cfg.root_seed, "dev", 1)
        w = CWWorld(cfg, rngs, sids, rho=0.85)
        w.budget_total = _budget(cfg, 0.5)
        acts = []
        for k in range(int(cfg.protocol.horizon)):
            w.truth_override = _truth_series(perturb, k, w.n_users) if perturb else None
            a = policy(w.view())
            acts.append((a.kind, a.user))
            w.step(a)
        return acts

    for tag, policy in (("CSI 敏感", pol), ("纯 AoI", pol_aoi)):
        same_x = run_x(policy, None) == run_x(policy, "x")
        g.check(f"1c[{tag}] 只改真状态 x ⇒ 整条动作序列逐位不变", same_x,
                {"identical": same_x})

    # (d) 但真实送达确实会变 —— 否则"扰动无效"，上一条就是恒真（无分辨力）
    def delivered(perturb):
        r = run_episode(cfg, pol, split="dev", episode=1,
                        budget_total=_budget(cfg, 0.5), rho=0.85, perturb_truth=perturb)
        return r.n_delivered

    d_none, d_x, d_h = delivered(None), delivered("x"), delivered("h")
    g.check("1e 扰动确实改变了真实送达数（上一条不是恒真）",
            (d_x != d_none) or (d_h != d_none),
            {"delivered_clean": d_none, "delivered_perturb_x": d_x, "delivered_perturb_h": d_h})

    # (f) 试图读真值的策略必须失败
    illegal = ILLEGAL_TruthGreedy(allow_truth_keys=True)
    try:
        illegal(w.view())
        failed = False
    except AttributeError:
        failed = True
    g.check("1f 越界策略读真值失败", failed, "ILLEGAL_TruthGreedy 期望抛 AttributeError")

    # (g) 未授权时不能构造越界策略
    try:
        ILLEGAL_TruthGreedy(allow_truth_keys=False)
        blocked = False
    except PermissionError:
        blocked = True
    g.check("1g 未授权时拒绝构造越界策略", blocked, "期望 PermissionError")

    # ---------------------------------------------------------------- 1h
    # ★★ v1.1 新增（验收要求的正形式）：
    #    「**公开历史完全相同** ⇒ 当前动作不受隐藏真值影响」。
    #    旧形式（"改真值后整条动作序列不变"）在 POMDP 里不成立：改真信道会合法地
    #    改变 ACK，ACK 是合法信息 ⇒ 要求整条不变等于要求仿真里没有信道。
    #
    #    构造方式（比"分歧点解释"更直接）：两个**同种子**世界，用同一策略跑同样的
    #    前 i 步 ⇒ 公开历史逐字段相同；然后只扰动其中一个世界的**隐藏真值**
    #    （不改任何子流，也不再推进）⇒ 两者公开信息此刻完全一样。
    #    合法策略必须给出**相同动作**；偷看真值的策略必然不同。
    from experiments.cw_v1.world import PUBLIC_HISTORY_FIELDS, VIEW_MEMBERS

    def build_prefix(n_steps, policy, *, perturb_hidden: float = 0.0,
                     which: int = 1):
        rngs, sids = make_episode_rngs(cfg.root_seed, "dev", 11)
        w = CWWorld(cfg, rngs, sids, rho=0.85)
        w.budget_total = int(cfg.protocol.horizon)
        for _ in range(int(n_steps)):
            w.step(policy(w.view()))
        if perturb_hidden:
            w._x_true[which] += float(perturb_hidden)
            w._log_h_true[which] -= float(perturb_hidden)
        return w

    N_PREFIX = 20
    for tag, policy in (("CSI 敏感", pol), ("纯 AoI", pol_aoi)):
        wa = build_prefix(N_PREFIX, policy)
        wb = build_prefix(N_PREFIX, policy, perturb_hidden=5.0)
        ha, hb = wa.view().history, wb.view().history
        same_pub = (ha == hb) and (wa.view().telemetry == wb.view().telemetry)
        act_a, act_b = policy(wa.view()), policy(wb.view())
        g.check(f"1h[{tag}] 公开历史相同 ⇒ 隐藏真值变化不改变当前动作",
                same_pub and (act_a == act_b),
                {"public_identical": same_pub, "act_clean": str(act_a),
                 "act_perturbed": str(act_b), "prefix_steps": N_PREFIX})

    # 1h-ctl：同一构造下，**偷看真状态**的策略必须给出不同动作。
    # ★ 没有这一条，1h 可能是恒真（"两个世界当然一样"）—— 阳性对照才有分辨力。
    class _LeakyOn:
        """故意偷看真状态：按 |x_true| 最大的用户发。不读 view。"""

        name = "LEAKY_PEEK"

        def __init__(self, world_box: list) -> None:
            self.box = world_box

        def __call__(self, view):  # noqa: ARG002 - 故意不读 view
            return Action("SEND", int(np.argmax(np.abs(self.box[0]._x_true))))

    box_a = [None]
    pol_leak = _LeakyOn(box_a)
    wa = build_prefix(N_PREFIX, lambda v: Action("SEND", v.t % 2))
    # ★ 2026-10-22 18:xx 修：原来固定扰 `user=1`，能不能翻转 argmax 取决于那条
    #   随机轨迹（`event_pairing` 一改就翻不动了）⇒ 这条对照**当时是靠运气过的**。
    #   现在改成"扰**当前不是** argmax 的那个用户、且幅度 50"，翻转是构造保证的。
    which_flip = 1 - int(np.argmax(np.abs(wa._x_true)))
    wb = build_prefix(N_PREFIX, lambda v: Action("SEND", v.t % 2),
                      perturb_hidden=50.0, which=which_flip)
    box_a[0] = wa
    act_a = pol_leak(None)
    box_a[0] = wb
    act_b = pol_leak(None)
    g.check("1h-ctl 偷看真状态的策略在公开历史相同时给出不同动作（证明 1h 有分辨力）",
            (wa.view().history == wb.view().history) and (act_a != act_b),
            {"public_identical": wa.view().history == wb.view().history,
             "perturbed_user": which_flip, "act_clean": str(act_a), "act_perturbed": str(act_b)})

    # ---------------------------------------------------------------- 1i
    # ★★ v1.1 新增：**白名单本身**要能验。取真实历史行，检查
    #    (a) 白名单外的字段一个都读不到（含 w_realized / cost_true_by_user）；
    #    (b) 白名单内的字段确实存在（否则上面那条会被"全空表"满足 = 恒真）。
    rngs, sids = make_episode_rngs(cfg.root_seed, "dev", 7)
    w = CWWorld(cfg, rngs, sids, rho=0.85)
    w.budget_total = _budget(cfg, 0.5)
    for _ in range(24):
        w.step(Action("SEND", 0))
    hist = w.view().history
    row0 = hist[0]
    leaked = sorted({k for row in hist for k in row} - set(PUBLIC_HISTORY_FIELDS))
    missing = sorted(set(PUBLIC_HISTORY_FIELDS) - set(row0))
    g.check("1i 公开历史 = 白名单（无多余字段、无缺字段）",
            (not leaked) and (not missing),
            {"leaked_fields": leaked, "missing_fields": missing,
             "n_rows": len(hist)})

    # 1i-ctl：把一条真值字段塞回白名单，判据必须 FAIL（否则"无多余字段"是空话）
    import experiments.cw_v1.world as _W
    saved = _W.PUBLIC_HISTORY_FIELDS
    try:
        _W.PUBLIC_HISTORY_FIELDS = saved + ("cost_true_by_user",)
        rngs, sids = make_episode_rngs(cfg.root_seed, "dev", 7)
        w2 = CWWorld(cfg, rngs, sids, rho=0.85)
        w2.budget_total = _budget(cfg, 0.5)
        for _ in range(8):
            w2.step(Action("SEND", 0))
        h2 = w2.view().history
        leaked2 = sorted({k for row in h2 for k in row} - set(saved))
    finally:
        _W.PUBLIC_HISTORY_FIELDS = saved
    g.check("1i-ctl 白名单被篡改后立刻出现越界字段（证明 1i 有分辨力）",
            "cost_true_by_user" in leaked2, {"leaked_fields": leaked2})

    # ---------------------------------------------------------------- 1j
    # ★★ v1.1 新增：**未送达的 SEND 没有样本** —— `y_t` 必须为 None。
    #    旧版无论成没成都把采样值写进记录；未送达时那个值就是 `x_t + 噪声`，
    #    公开它等于绕过信道。
    rngs, sids = make_episode_rngs(cfg.root_seed, "dev", 8)
    w = CWWorld(cfg, rngs, sids, rho=0.85)
    w.budget_total = 60
    w._log_h_true[:] = -30.0          # 必然失败
    rec_fail = w.step(Action("SEND", 0))
    w._log_h_true[:] = 30.0           # 必然成功
    rec_ok = w.step(Action("SEND", 0))
    g.check("1j 失败的 SEND 不公开 y_t；成功的 SEND 才公开",
            (rec_fail.data_delivered == 0 and rec_fail.y_t is None
             and rec_ok.data_delivered == 1 and rec_ok.y_t is not None),
            {"fail_delivered": rec_fail.data_delivered, "fail_y_t": rec_fail.y_t,
             "ok_delivered": rec_ok.data_delivered,
             "ok_y_t": None if rec_ok.y_t is None else "finite"})

    # 1k：视图上不存在 world 句柄（`_inner` 已删）
    rngs, sids = make_episode_rngs(cfg.root_seed, "dev", 9)
    w = CWWorld(cfg, rngs, sids)
    vv = w.view()
    handles = [h for h in ("_inner", "_SchedulerView__w", "_w", "world", "env",
                           "_CWWorld__x_true", "_x_true")
               if hasattr(vv, h)]
    # 快照里**任何**值都不能是真值：把 _pub 全展开扫一遍字段名
    import json as _json
    blob = _json.dumps(vv._pub, ensure_ascii=False, default=str)
    bad_keys = [k for k in ("p_true", "w_realized", "cost_true", "true_norm",
                            "cost_true_by_user")
                if k in blob]
    g.check("1k 视图无 world 句柄，且快照里不含任何真值字段名",
            (not handles) and (not bad_keys),
            {"found_handles": handles, "truth_keys_in_snapshot": bad_keys,
             "slots": list(SchedulerView.__slots__),
             "declared_members": list(VIEW_MEMBERS)})


# ============================================================ 2 消费者接线
def counterexample_2(cfg: CWConfig, g: Gate) -> None:
    """只变历史 CSI 测量：信息敏感策略动作应改变；忽略 CSI 的策略不应改变。

    ★ 只证明通道存在，不声称闭环收益（首轮卡原文）。
    """
    rngs, sids = make_episode_rngs(cfg.root_seed, "dev", 2)
    w = CWWorld(cfg, rngs, sids)
    w.budget_total = _budget(cfg, 0.5)

    # 直接注入不同的 CSI 信念与 AoI，看同一策略是否给出不同动作
    def act_after(m_h: list[float], use_csi: bool) -> Action:
        w._m_h[:] = m_h
        w._v_h[:] = 0.01
        w._x_hat[:] = 0.5
        w._stamp[:] = -5   # AoI = 5 ≥ theta，触发发送
        w.t = 10
        pol = CsiSensitive(theta=3, use_csi=use_csi)
        return pol(w.view())

    a_hi0 = act_after([1.5, -1.5], use_csi=True)
    a_hi1 = act_after([-1.5, 1.5], use_csi=True)
    b_hi0 = act_after([1.5, -1.5], use_csi=False)
    b_hi1 = act_after([-1.5, 1.5], use_csi=False)

    g.check("2a 信息敏感策略随 CSI 改变动作", a_hi0 != a_hi1,
            {"csi_0_high": str(a_hi0), "csi_1_high": str(a_hi1)})
    # ★ 阴性对照：忽略 CSI 的同一结构策略不应随 CSI 改变动作
    g.check("2b 阴性对照：忽略 CSI 的策略不随 CSI 改变动作", b_hi0 == b_hi1,
            {"csi_0_high": str(b_hi0), "csi_1_high": str(b_hi1)})
    g.check("2c 两者在 CSI 相同时动作一致（结构确实相同）",
            act_after([0.3, 0.3], use_csi=True) == act_after([0.3, 0.3], use_csi=False),
            "同一 CSI 下应为同一动作")


# ============================================================ 3 探测记账
def counterexample_3(cfg: CWConfig, g: Gate) -> None:
    """PROBE 扣费、消耗物理时隙、只更新被测链路信念、不更新任务观测；
    budget=0 时拒绝收费动作。"""
    rngs, sids = make_episode_rngs(cfg.root_seed, "dev", 3)
    w = CWWorld(cfg, rngs, sids)
    w.budget_total = 10
    w._x_hat[:] = 0.0
    w._P_hat[:] = 1.0
    w._stamp[:] = -1

    x_hat_before = w._x_hat.copy()
    stamp_before = w._stamp.copy()
    t_before = w.t
    h_var_before = w._v_h.copy()
    rec = w.step(Action("PROBE", 0))

    g.check("3a PROBE 扣费 1 量子", w.budget_used == 1 and rec.resource_spent == 1,
            {"budget_used": w.budget_used, "recorded": rec.resource_spent})
    g.check("3b PROBE 消耗物理时隙", w.t == t_before + 1, {"t_before": t_before, "t_after": w.t})
    g.check("3c PROBE 不刷新任务状态估计与时间戳",
            np.array_equal(w._x_hat, x_hat_before) and np.array_equal(w._stamp, stamp_before),
            {"stamp": w._stamp.tolist()})
    # 3d：被测用户的方差必须显著收缩。
    # ★ 未测用户的方差会因 AR 演化略增，所以**不能**要求"未测者完全不变" ——
    #   严格的对照写在 3e（与纯 AR 演化逐位比较）。
    g.check("3d PROBE 使被测链路的信道信念收缩",
            w._v_h[0] < 0.5 * h_var_before[0],
            {"v_h_0_before": float(h_var_before[0]), "v_h_0_after": float(w._v_h[0]),
             "v_h_1_before": float(h_var_before[1]), "v_h_1_after": float(w._v_h[1])})

    # 3e：严格对照 —— 未测用户方差的演化应等于"纯 AR 演化"，与是否测了别人无关
    rngs2, sids2 = make_episode_rngs(cfg.root_seed, "dev", 3)
    w2 = CWWorld(cfg, rngs2, sids2)
    w2.budget_total = 10
    w2._v_h[:] = 1.0
    w2.step(Action("HOLD"))
    pure_ar = float(w2._v_h[1])
    g.check("3e 未测用户方差 = 纯 AR 演化（未被本次测量污染）",
            abs(float(w._v_h[1]) - pure_ar) < 1e-12,
            {"measured_elsewhere": float(w._v_h[1]), "pure_ar": pure_ar})

    # 3f：budget=0 拒绝收费动作
    rngs3, sids3 = make_episode_rngs(cfg.root_seed, "dev", 3)
    w3 = CWWorld(cfg, rngs3, sids3)
    w3.budget_total = 0
    rejected = {"PROBE": False, "SEND": False}
    for kind in ("PROBE", "SEND"):
        try:
            w3.step(Action(kind, 0))
        except BudgetExhausted:
            rejected[kind] = True
    g.check("3f budget=0 时拒绝 PROBE 与 SEND", all(rejected.values()), rejected)

    # 3g：预算耗尽后拒绝的是"收费动作"，HOLD 仍然可以走
    rngs4, sids4 = make_episode_rngs(cfg.root_seed, "dev", 3)
    w4 = CWWorld(cfg, rngs4, sids4)
    w4.budget_total = 1
    w4.step(Action("PROBE", 0))
    try:
        w4.step(Action("SEND", 0))
        blocked = False
    except BudgetExhausted:
        blocked = True
    hold_ok = w4.step(Action("HOLD")) is not None
    # t 应为：PROBE 占 1 槽 → t=1；SEND 被拒（不推进）；HOLD 占 1 槽 → t=2
    g.check("3g 预算耗尽后拒绝 SEND、但 HOLD 仍可执行且只推进 1 槽",
            blocked and hold_ok and w4.t == 2,
            {"send_blocked": blocked, "hold_ok": hold_ok, "t": w4.t,
             "note": "被拒的 SEND 不得推进物理时间"})


# ============================================================ 4 失败仍扣 / ACK
def counterexample_4(cfg: CWConfig, g: Gate) -> None:
    """数据失败仍扣尝试资源、不重置任务 AoI；ACK 与 NACK 按事件更新；
    HOLD 不产生 ACK。"""
    rngs, sids = make_episode_rngs(cfg.root_seed, "dev", 4)
    w = CWWorld(cfg, rngs, sids)
    w.budget_total = 200
    # 强制全失败：把真信道压到极差（这是"只改真值"，不改子流）
    w._log_h_true[:] = -30.0

    stamp_before = w._stamp.copy()
    rec = w.step(Action("SEND", 0))
    g.check("4a 全失败环境下 SEND 确实失败", rec.data_delivered == 0 and rec.ack == 0 and rec.p_true < 1e-6,
            {"p_true": rec.p_true, "ack": rec.ack})
    g.check("4b 失败仍扣尝试资源", w.budget_attempted == 1 and w.budget_wasted == 1,
            {"attempted": w.budget_attempted, "wasted": w.budget_wasted})
    g.check("4c 失败不刷新任务 AoI（时间戳不变）", np.array_equal(w._stamp, stamp_before),
            {"stamp": w._stamp.tolist()})
    g.check("4d NACK 计入事件计数", w.n_nack == 1 and w.n_ack == 0,
            {"nack": w.n_nack, "ack": w.n_ack})

    # 成功一次，ACK 计数 +1
    w._log_h_true[:] = 30.0
    rec2 = w.step(Action("SEND", 0))
    g.check("4e 成功时 ACK 计数 +1 且时间戳更新",
            rec2.ack == 1 and w.n_ack == 1 and w._stamp[0] == 1,
            {"ack": rec2.ack, "n_ack": w.n_ack, "stamp0": int(w._stamp[0])})

    # HOLD 不产生 ACK
    ack_before, nack_before, att_before = w.n_ack, w.n_nack, w.budget_attempted
    rec3 = w.step(Action("HOLD"))
    g.check("4f HOLD 不产生 ACK、不扣资源、不增加 attempted",
            rec3.ack is None and w.n_ack == ack_before and w.n_nack == nack_before
            and w.budget_attempted == att_before,
            {"ack_field": rec3.ack, "n_ack": w.n_ack, "n_nack": w.n_nack})

    # 4g：ACK/NACK 是信息 —— v1.1 起**默认接入**信念更新（旧版默认关闭，
    #     理由是"关掉它才能让 1d 通过"，属于关信息迁就判据，已废）。
    #     这里验**方向性**：ACK 抬均值、NACK 不抬且不收缩到与 ACK 同水平。
    from dataclasses import replace as _replace

    cfg_ack = CWConfig(plant=cfg.plant,
                       channel=_replace(cfg.channel, use_ack_for_csi=True),
                       protocol=cfg.protocol, root_seed=cfg.root_seed)

    def belief_after(act: Action, force_success: bool | None) -> tuple[float, float]:
        rngs2, sids2 = make_episode_rngs(cfg_ack.root_seed, "dev", 5)
        w2 = CWWorld(cfg_ack, rngs2, sids2)
        w2.budget_total = 50
        if force_success is not None:
            w2._log_h_true[:] = 30.0 if force_success else -30.0
        w2.step(act)
        return float(w2._m_h[0]), float(w2._v_h[0])

    m_ack, v_ack = belief_after(Action("SEND", 0), True)
    m_nack, v_nack = belief_after(Action("SEND", 0), False)
    m_hold, v_hold = belief_after(Action("HOLD"), None)
    # ★ 判据第一版用 `m_ack > m_hold > m_nack`，FAIL 了，原因不是实现错：
    #   NACK 分支的**后验均值恰好被拉到 0**（与 HOLD 的先验均值 0 同值），
    #   所以均值这一个维度上区分不出来 —— 但它的**方差没有收缩**（1.0 vs ACK 的 0.557）。
    #   ⇒ 判据改为"均值与方差两维都不退化"：ACK 应抬均值，NACK 不应抬均值且不应收缩到与 ACK 同水平。
    #   这是"ACK/NACK 方向相反且都可区分"的更完整表述，不是放宽阈值。
    g.check("4g ACK 与 NACK 对信道信念的作用方向相反且可区分",
            (m_ack > m_hold) and (m_nack <= m_hold)
            and (v_nack > v_ack),
            {"m_after_ack": m_ack, "m_after_hold": m_hold, "m_after_nack": m_nack,
             "v_after_ack": v_ack, "v_after_hold": v_hold, "v_after_nack": v_nack,
             "config": "use_ack_for_csi=True（v1.1 默认）"})

    # 4h：默认协议下 ACK **确实**进入信念 —— 这条防的是"开关写着 True 但没接线"。
    # ★ 与旧版**方向相反**：旧版断言"默认不变"，那是为了让 1d 的旧形式成立。
    def belief_of(c, act: Action, force_success: bool | None) -> float:
        rngs3, sids3 = make_episode_rngs(c.root_seed, "dev", 5)
        w3 = CWWorld(c, rngs3, sids3)
        w3.budget_total = 50
        if force_success is not None:
            w3._log_h_true[:] = 30.0 if force_success else -30.0
        w3.step(act)
        return float(w3._m_h[0])

    d_ack = belief_of(cfg, Action("SEND", 0), True)
    d_nack = belief_of(cfg, Action("SEND", 0), False)
    g.check("4h 默认协议下 ACK 与 NACK 给出不同的信道信念（接线活着）",
            abs(d_ack - d_nack) > 1e-6,
            {"m_after_ack": d_ack, "m_after_nack": d_nack,
             "default_use_ack_for_csi": bool(cfg.channel.use_ack_for_csi)})

    # 4i：显式关闭开关后必须**逐位**相同 —— 否则 4h 通过也不能说明是开关在起作用
    cfg_noack = CWConfig(plant=cfg.plant,
                         channel=_replace(cfg.channel, use_ack_for_csi=False),
                         protocol=cfg.protocol, root_seed=cfg.root_seed)
    n_ack = belief_of(cfg_noack, Action("SEND", 0), True)
    n_nack = belief_of(cfg_noack, Action("SEND", 0), False)
    g.check("4i 关闭 use_ack_for_csi 后 ACK 不影响信念（阴性对照）",
            abs(n_ack - n_nack) < 1e-15,
            {"m_after_ack": n_ack, "m_after_nack": n_nack})


# ============================================================ 5 时序正确性
def counterexample_5(cfg: CWConfig, g: Gate) -> None:
    """槽末 x_t 的观测不能直接当 x_(t+1)；核对一次校正与动作预测。

    ★ 判据设计教训：一开始我拿"零过程噪声 + 零测量噪声"的系统做 5b/5c 的有分辨力检查，
      结果两边成本都是 0（没有噪声就没有代价），检查**恒等**、无分辨力。
      正确做法：精确性检查用零噪声（5a），**代价比较改用带回过程噪声的真实系统**（5c/5d）。
    """
    from dataclasses import replace

    # --- 5a：零噪声下，正确接收端的估计必须与真值逐位一致 ---
    zero_plant = replace(cfg.plant, sigma_w=(0.0, 0.0), sigma_meas=0.0, prior_var=1e-12)
    cfg0 = CWConfig(plant=zero_plant, channel=cfg.channel, protocol=cfg.protocol,
                    root_seed=cfg.root_seed)
    rngs, sids = make_episode_rngs(cfg0.root_seed, "dev", 6)
    w = CWWorld(cfg0, rngs, sids)
    w.budget_total = 10
    w._x_hat[:] = 0.0
    w._P_hat[:] = 1e-12
    w._stamp[:] = -1
    w._x_true[:] = 0.0
    w._log_h_true[:] = 30.0          # 强制送达成功
    w.step(Action("SEND", 0))        # 首次观测，估计 = 真值

    max_abs_dev = 0.0
    for _ in range(5):
        w.step(Action("HOLD"))
        max_abs_dev = max(max_abs_dev, abs(w._x_hat[0] - w._x_true[0]))
    g.check("5a 零噪声下正确接收端的估计与真值逐位一致（不偏移一步）",
            max_abs_dev < 1e-9,
            {"max_abs_dev": max_abs_dev, "x_hat": float(w._x_hat[0]),
             "x_true": float(w._x_true[0])})

    # --- 5b/5c：错误接收端必须"能"被分辨 ---
    #
    # ★★ 判据第一版拿**闭环任务成本**做分辨力检查，FAIL 且找不到稳定符号。
    #    查下来不是实现错，而是**判据在这里没有分辨力**：
    #    R=0.1 ≪ Q=1 ⇒ 名义 LQR 增益 K≈2.8，闭环 |A−BK|≈0.73，
    #    控制器把状态压到 1e-3 量级 ⇒ 两种接收端的成本都被压到噪声地板以下，
    #    差值符号由浮点噪声决定。用"成本"去找一个 1e-3 量级的差异，是拿放大镜看毫米级的东西。
    #  ⇒ 正确的检验对象是**接收端估计误差本身**，不是闭环成本。
    #    这也是 v4 §5 的要求：模型/估计的缺陷应在估计误差层面度量，不能靠闭环成本间接猜。
    def est_rmse(correct: bool, ep: int) -> float:
        rngs2, sids2 = make_episode_rngs(cfg.root_seed, "dev", ep)
        w2 = CWWorld(cfg, rngs2, sids2, rho=0.85,
                     allow_truth_keys=not correct, correct_receiver=correct)
        w2.budget_total = _budget(cfg, 0.5)
        pol = PacedAge(theta=4)
        errs = []
        for _ in range(int(cfg.protocol.horizon)):
            a = pol(w2.view())
            w2.step(a)
            errs.append(np.linalg.norm(w2._x_hat - w2._x_true))
        return float(np.sqrt(np.mean(np.square(errs))))

    eps = list(range(40, 46))
    ok_e = [est_rmse(True, e) for e in eps]
    bad_e = [est_rmse(False, e) for e in eps]
    rel_e = [(b - a) / a for a, b in zip(ok_e, bad_e)]
    g.check("5b 错误接收端在估计误差上可分辨（不是恒等）",
            abs(float(np.mean(rel_e))) > 1e-3,
            {"mean_rel_rmse_diff": float(np.mean(rel_e)),
             "correct_rmse": [round(x, 6) for x in ok_e],
             "leaky_rmse": [round(x, 6) for x in bad_e],
             "n_leaky_worse": sum(1 for r in rel_e if r > 0), "n_episodes": len(eps)})

    # ★★ 5c 第一版要求"错误接收端估计误差**更大**"，FAIL（6 个 episode 只有 3 个更差，
    #    另有两个明显更好，-0.09 / -0.033）。查下来**是判据错了，不是实现对了**：
    #      正确实现：观测 y_t 先校正到 t，再用已知 u_t 推一步到 t+1（A·x + B·u）。
    #      错误实现：把 y_t 直接当 t+1 的值（**没有**推那一步）。
    #      两者之差 = (A−1)·x + B·u，A=1.01、B=0.1、u=−Kx̂(K≈2.8)
    #      ⇒ ≈ −0.27·x̂，是一个**收缩**而非噪声；它碰巧起了阻尼作用，误差可大可小。
    #    ⇒ "错位 ⇒ 更贵/更差"在这个系统上**不成立**，不能拿它当判据。
    #    正确的检验是**结构性的**：两个实现的映射关系必须可区分，且正确实现必须
    #    **显式**用已知的 u_t 推那一步。下面直接验这两条，不依赖成本或误差的符号。
    def mapping_after_first_ack(correct: bool) -> dict:
        rngs3, sids3 = make_episode_rngs(cfg.root_seed, "dev", 60)
        w3 = CWWorld(cfg, rngs3, sids3, rho=0.85,
                     allow_truth_keys=not correct, correct_receiver=correct)
        w3.budget_total = 50
        w3._stamp[:] = -1
        w3._log_h_true[:] = -30.0     # 第一次必 NACK，确保第二次是首次接收
        rec1 = w3.step(Action("SEND", 0))
        w3._log_h_true[:] = 30.0      # 第二次必 ACK
        # ★ y_t 与 u_t 直接读记录，不自己重算 ——
        #   重算会消费 meas 子流的**下一个**数，与实际被采样的值不同（第一版正是因此误判）。
        u = float(w3.control_from_estimate()[0])
        rec = w3.step(Action("SEND", 0))
        assert rec.ack == 1, "构造前提失败：第二次发送未送达"
        A = float(cfg.plant.A[0])
        B = float(cfg.plant.B)
        xh = float(w3._x_hat[0])
        return {"x_hat": xh, "y_t": rec.y_t, "u_t": u, "A": A, "B": B,
                "w_t1": float(rec1.w_realized[0]),   # 上一步的 w（已进入 t 时刻真值）
                "w_t": float(rec.w_realized[0]),     # 本步的 w
                "prior_var_at_t": float(cfg.plant.prior_var)}

    m_ok = mapping_after_first_ack(True)
    m_bad = mapping_after_first_ack(False)
    # ★★ 这条判据改了四版，前三版漏的都是**可独立验证**的项，第四版闭合：
    #   ① 卡尔曼增益 K：首次接收 P=1、R=0.01² ⇒ K=0.9999 而不是 1 → 差 9e-7；
    #   ② 接收端**不能用**已实现的过程噪声 w（那是真值）：预测只能是期望
    #      `E[x_{t+1}] = A·x̄_t + B·u_t`，不确定性进 `P` 不进均值；
    #   ③ 先验均值不为 0 时，增益作用在 `y − x̄` 上，`K·x̄` 那一项**不会**被约掉。
    #      完整表达式：
    #          x̄_t = 上一步的预测值（本身已含 `A·w1`）
    #          P_t = A²P_{t-1} + σ_w²
    #          K   = P_t/(P_t + σ_meas²)
    #          x̂_{t+1} = A·(x̄_t + K(y − x̄_t)) + B·u_t
    #      → `A·K·y` 之外还剩 `A(1−K)·x̄_t`，数值 = 1.85e-8，正是残差。
    P0 = float(cfg.plant.prior_var)
    Rn = float(cfg.plant.sigma_meas) ** 2
    sw2 = float(cfg.plant.sigma_w[0]) ** 2
    A, B, u, y_t = m_ok["A"], m_ok["B"], m_ok["u_t"], m_ok["y_t"]
    w_t1 = m_ok["w_t1"]
    # 第 1 步（NACK）：x̄_1 = A·0 + B·0 + 0 = 0，P_1 = A²·P0 + σ_w²
    xbar_t = 0.0
    P_t = A * A * P0 + sw2
    K_t = P_t / (P_t + Rn)
    pred_correct = A * (xbar_t + K_t * (y_t - xbar_t)) + B * u
    pred_leaky = xbar_t + K_t * (y_t - xbar_t)
    g.check("5c 正确接收端：观测校正到 t 再按期望动力学推到 t+1（含 K·(y−x̄) 与 A、B·u）",
            abs(m_ok["x_hat"] - pred_correct) < 1e-15,
            {**m_ok, "K_t": K_t, "P_t": P_t, "pred_analytic": pred_correct,
             "residual": m_ok["x_hat"] - pred_correct})
    g.check("5d 错误接收端：把校正值直接当 t+1（x̂ = x̄ + K(y−x̄)，既不推进也不加噪声）",
            abs(m_bad["x_hat"] - pred_leaky) < 1e-12,
            {**m_bad, "K_t": K_t, "pred_analytic": pred_leaky})
    g.check("5e 两种映射在数值上可区分（差 A·(...) + B·u − (...)，即 (A−1)(x̄+K(y−x̄)) + B·u）",
            abs(pred_correct - pred_leaky) > 1e-9,
            {"diff": pred_correct - pred_leaky})
    # 5f：不许"用 w 修正估计"—— 接收端若把 w 加进均值，就等于偷看真值。
    #     判据：真实 x_{t+1} 与估计之差，必须主要由 w_t 构成（比值 ≈ 1）。
    x_true_next = A * y_t + B * u + m_ok["w_t"]
    gap = (x_true_next - m_ok["x_hat"]) / m_ok["w_t"] if abs(m_ok["w_t"]) > 1e-12 else float("nan")
    g.check("5f 接收端没有偷看 w（真值与估计之差主要由 w 构成）",
            abs(gap - 1.0) < 0.5,
            {"x_true_next_approx": x_true_next, "x_hat": m_ok["x_hat"],
             "gap_over_w": gap, "note": "gap/w ≈ 1 ⇒ 差值主要就是 w，未被估计吃掉"})

    # --- 5d：错误接收端只允许在 allow_truth_keys 下构造 ---
    try:
        run_episode(cfg, PacedAge(theta=4), split="dev", episode=40, budget_total=10,
                    rho=0.85, allow_truth_keys=False, correct_receiver=False)
        blocked = False
    except ValueError:
        blocked = True
    g.check("5d 未加诊断开关时拒绝构造错误接收端", blocked, "期望 ValueError")


# ============================================================ 6 探测信息价值
def counterexample_6(cfg: CWConfig, g: Gate) -> None:
    """关闭测量信息后 PROBE 只保留开销；固定普通策略忽略 CSI 时应对测量内容不敏感，
    信息敏感策略作为阴性对照。"""
    # (a) PROBE 只花钱：与全 HOLD 比，任务成本不应变好（同一外生轨迹）
    r_probe = run_episode(cfg, AlwaysProbe(), split="dev", episode=7,
                          budget_total=_budget(cfg, 0.5), rho=0.85)
    r_hold = run_episode(cfg, lambda v: Action("HOLD"), split="dev", episode=7,
                         budget_total=_budget(cfg, 0.5), rho=0.85)
    g.check("6a 纯 PROBE 的策略任务成本不优于全 HOLD（探测只保留开销）",
            r_probe.cost_true_per_step >= r_hold.cost_true_per_step - 1e-12,
            {"probe": r_probe.cost_true_per_step, "hold": r_hold.cost_true_per_step,
             "probe_spent": r_probe.budget_used, "hold_spent": r_hold.budget_used})
    g.check("6b 纯 PROBE 确实消耗了预算与交付 0 帧数据",
            r_probe.budget_used > 0 and r_probe.n_delivered == 0,
            {"spent": r_probe.budget_used, "delivered": r_probe.n_delivered})

    # (c) 普通策略（不读 CSI）对测量内容不敏感
    r_csi = run_episode(cfg, PacedAge(theta=4), split="dev", episode=7,
                        budget_total=_budget(cfg, 0.5), rho=0.85)
    g.check("6c 不读 CSI 的普通策略动作与其无关（trace 不含探测）",
            r_csi.n_probe == 0,
            {"n_probe": r_csi.n_probe})


# ============================================================ 7 记账对账
def counterexample_7(cfg: CWConfig, g: Gate) -> None:
    """每 episode 输出独立的探测次数/资源、attempted、delivered、idle、失败、总成本、
    逐任务成本；恒等关系能对账。"""
    rho_vals = list(cfg.channel.rho_panel)
    policies = [AlwaysProbe(), AlwaysSend(), PacedAge(theta=4), PacedAge(theta=8),
                CsiSensitive(theta=4, use_csi=True), ProbeThenSend(csi_max_age=4)]
    bad = []
    rows = []
    for ri, rho in enumerate(rho_vals):
        for ratio in cfg.protocol.budget_ratios:
            for pol in policies:
                r = run_episode(cfg, pol, split="dev", episode=10 + ri,
                                budget_total=_budget(cfg, ratio), rho=rho)
                ident = check_identities(r, cfg)
                rows.append({"rho": rho, "budget_ratio": ratio, "policy": r.policy,
                             "pass": ident["pass"],
                             "failed": [k for k, v in ident["items"].items() if not v]})
                if not ident["pass"]:
                    bad.append((rho, ratio, pol.name, ident))
    g.check("7a 全部 cell 的记账恒等式成立", not bad,
            {"n_cells": len(rows), "n_fail": len(bad),
             "failed_items": sorted({k for _, _, _, i in bad for k, v in i["items"].items() if not v})})

    # 7b：阴性对照 —— 人为破坏一个字段，恒等式必须能抓到
    r = run_episode(cfg, PacedAge(theta=4), split="dev", episode=10,
                    budget_total=_budget(cfg, 0.5), rho=rho_vals[0])
    r.n_probe += 1  # 篡改
    caught = check_identities(r, cfg)
    g.check("7b 阴性对照：篡改 probe 计数导致恒等式 FAIL", not caught["pass"],
            {"failed": [k for k, v in caught["items"].items() if not v]})

    # 7c：逐用户成本存在且非负
    g.check("7c 逐用户成本齐全且非负",
            len(r.cost_true_by_user) == cfg.plant.n_users and all(x >= 0 for x in r.cost_true_by_user),
            {"cost_true_by_user": r.cost_true_by_user})


# ============================================================ 8 子流
def counterexample_8(cfg: CWConfig, g: Gate) -> None:
    """split 及 plant/channel/measurement/packet 子流 ID 不重复；
    不同策略共享外生随机流，策略动作影响观测路径但不能改未来外生序列。"""
    plan = {"dev": 12, "train": 4, "test": 2}
    rep = stream_uniqueness_report(cfg.root_seed, plan)
    g.check("8a 子流 ID 全部唯一", rep["pass"],
            {"n_streams": rep["n_streams"], "n_unique": rep["n_unique"],
             "dup": rep["duplicates"][:5], "determinism_fail": rep["determinism_failures"]})

    # 8b：同一 (split, episode) 下，不同策略的 plant/channel 子流 ID 必须相同
    r1 = run_episode(cfg, AlwaysSend(), split="dev", episode=20,
                     budget_total=_budget(cfg, 0.5), rho=0.85)
    r2 = run_episode(cfg, PacedAge(theta=8), split="dev", episode=20,
                     budget_total=_budget(cfg, 0.5), rho=0.85)
    same_plant = r1.stream_ids["spawn_keys"]["plant"] == r2.stream_ids["spawn_keys"]["plant"]
    same_chan = r1.stream_ids["spawn_keys"]["channel"] == r2.stream_ids["spawn_keys"]["channel"]
    g.check("8b 不同策略共享同一 plant/channel 子流 ID", same_plant and same_chan,
            {"plant": r1.stream_ids["spawn_keys"]["plant"],
             "channel": r1.stream_ids["spawn_keys"]["channel"]})

    # 8c：外生序列本身与策略无关 —— 直接取同一子流的头几个数比较
    from .streams import make_episode_rngs as mk
    ra, _ = mk(cfg.root_seed, "dev", 20)
    rb, _ = mk(cfg.root_seed, "dev", 20)
    same = all(np.array_equal(ra[k].standard_normal(5), rb[k].standard_normal(5))
               for k in ("plant", "channel"))
    g.check("8c 同 (split,episode) 重新构造的 plant/channel 序列逐位相同", same, "5 点抽样")

    # 8d：不同 episode 的外生序列必须不同（否则"独立初值"是假的）
    rc, _ = mk(cfg.root_seed, "dev", 21)
    diff = not np.array_equal(ra["plant"].standard_normal(5), rc["plant"].standard_normal(5))
    g.check("8d 不同 episode 的 plant 序列不同（不被错误复用）", diff, "5 点抽样")

    # 8e：健康检查（只报数值，不做闸）
    g.check("8e 跨 split 子流经验相关（健康检查，仅记录）", True,
            cross_split_independence_probe(cfg.root_seed, n=2000))


# ============================================================ 额外：闭环稳定性
def sanity_stability(cfg: CWConfig, g: Gate) -> None:
    """核闭环标称稳定性（首轮卡要求）。"""
    rngs, sids = make_episode_rngs(cfg.root_seed, "dev", 30)
    w = CWWorld(cfg, rngs, sids)
    rows = []
    for i in range(cfg.plant.n_users):
        K = w.nominal_gain(i)
        A_cl = float(cfg.plant.A[i]) - float(cfg.plant.B) * K
        rows.append({"i": i, "K": K, "A_closed": A_cl, "stable": abs(A_cl) < 1.0})
    g.check("S1 名义 LQR 闭环稳定 |A - B*K| < 1", all(r["stable"] for r in rows), rows)

    # 稳态成功率与解析值对账（10 万次抽样）
    rng = np.random.default_rng(7)
    z = rng.standard_normal(200000) * cfg.channel.h_log_init_std
    mc = float(np.mean(1.0 / (1.0 + np.exp(-cfg.channel.kappa * (z - cfg.channel.log_h_star)))))
    ana = steady_state_success(cfg)
    g.check("S2 稳态成功率：蒙特卡洛与解析值一致（<2e-3）", abs(mc - ana) < 2e-3,
            {"mc": mc, "analytic": ana, "diff": abs(mc - ana)})

    # rho 不影响稳态成功率（这是"两档面板可比"的前提）
    vals = []
    for rho in cfg.channel.rho_panel:
        rngs2, sids2 = make_episode_rngs(cfg.root_seed, "test", 0)
        w2 = CWWorld(cfg, rngs2, sids2, rho=rho)
        hs = []
        for _ in range(20000):
            w2._advance_channel()
            hs.extend(w2._log_h_true.tolist())
        vals.append(float(np.mean([p_success_from_log_h(h, cfg) for h in hs])))
    g.check("S3 不同 rho 档的稳态平均成功率相差 < 0.02（面板可比的前提）",
            abs(vals[0] - vals[1]) < 0.02,
            {"rho_panel": list(cfg.channel.rho_panel), "mean_succ": vals})


# ============================================================ 汇总
def run_all(cfg: CWConfig | None = None) -> dict:
    cfg = cfg or default_config()
    g = Gate()
    counterexample_1(cfg, g)
    counterexample_2(cfg, g)
    counterexample_3(cfg, g)
    counterexample_4(cfg, g)
    counterexample_5(cfg, g)
    counterexample_6(cfg, g)
    counterexample_7(cfg, g)
    counterexample_8(cfg, g)
    sanity_stability(cfg, g)
    return g.summary()
