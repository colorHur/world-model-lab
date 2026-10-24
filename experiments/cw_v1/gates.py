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
from .world import Action, BudgetExhausted, CWWorld, GridBelief, SchedulerView, action_cost, \
    action_space, p_success_from_log_h, steady_state_success


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
    # ★★ R1.11 判据修正（**不是放宽**，原判据靠缺陷才成立）：
    #   旧形式 `(m_nack <= m_hold) and (v_nack > v_ack)` 在 R1.10 缺陷下 PASS，
    #   原因是 NACK 的更新被错投到 u1 ⇒ u0 的 `_m_h[0]` 停在先验 0.0 = `m_hold`，
    #   且 u0 的 `_v_h[0]` 从不收缩 ⇒ 恒有 `v_nack > v_ack`。
    #   ⇒ 旧判据把"缺陷"编码成了正确性前提。修复后 NACK 真的作用于 u0，
    #   同一条伪观测（方差 ACK_VAR）必然把 v 收缩到与 ACK **相同**的量
    #   ⇒ `v_nack > v_ack` 必然 FAIL。正确表述是：
    #   均值符号相反（m_ack > m_hold > m_nack），方差**等量**收缩（一次二值观测的
    #   等效方差与方向无关）。这比旧形式更强：旧实现会在 `m_nack < m_hold` 上 FAIL。
    g.check("4g ACK 抬均值、NACK 压均值，两者方差等量收缩",
            (m_ack > m_hold) and (m_nack < m_hold) and abs(v_nack - v_ack) < 1e-12,
            {"m_after_ack": m_ack, "m_after_hold": m_hold, "m_after_nack": m_nack,
             "v_after_ack": v_ack, "v_after_hold": v_hold, "v_after_nack": v_nack,
             "config": "use_ack_for_csi=True（v1.1 默认）",
             "legacy_behavior": "旧实现 m_nack == m_hold (0.0) ⇒ 本条 FAIL ⇒ 有分辨力"})

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

    # ================= ★★ R1.11：逐用户 ACK/NACK **归属**（4j / 4k / 4l）=================
    # 背景（R1.10 实测）：`world.py` 的 NACK 支写的是 `self._ack_bump(i, -1.5)`，
    #   而 `i` 已被接收端名义滚动那行 `for i in range(self.n_users)` 泄漏成 `n_users-1`
    #   ⇒ 所有 NACK 的信道信念更新都落到**下标最大的用户**头上。
    #
    # ★ 为什么上面 4g/4h/4i 全 PASS 而缺陷仍在（台账 #94：闸的 plant 决定它覆盖哪一支）：
    #   它们**全部只读 `_m_h[0]`**，而 4g 的失败支判据是 `m_nack <= m_hold`。
    #   旧实现里 `SEND(0)` 失败 ⇒ 更新跑到 u1 ⇒ `_m_h[0]` 停在先验 0.0，与 `m_hold=0.0`
    #   **相等** ⇒ `m_nack <= m_hold` 仍然成立。判据从没写"必须**下降**"，
    #   也从来没问过"这个更新落在了**谁**身上"。
    #
    # ★★ 纪律落地：**"两个用户对称"必须在闸里被显式破坏一次** —— 4j 送的正是 u0
    #   （唯一让 `n_users-1 != act.user` 的那个用户）；4k 是镜像，旧实现在它上面**恰好
    #   PASS**，所以 4k 单独**没有分辨力**，只有 4j+4k 同时在场才封住"漏改一侧"。

    def _after(send_user: int, force_success: bool) -> tuple[list[float], list[float], float]:
        rngs4, sids4 = make_episode_rngs(cfg_ack.root_seed, "dev", 6)
        w4 = CWWorld(cfg_ack, rngs4, sids4)
        w4.budget_total = 50
        # 只改真信道（与 4a 同法：不动任何子流）
        w4._log_h_true[:] = 30.0 if force_success else -30.0
        w4.step(Action("SEND", send_user))
        return ([float(x) for x in w4._m_h], [float(x) for x in w4._v_h], float(w4.rho))

    def _bare(user: int, direction: float) -> tuple[float, float]:
        """**不走 step**，直接调一次 `_ack_bump` —— 唯一能精确解析对账的路径。"""
        rngs5, sids5 = make_episode_rngs(cfg_ack.root_seed, "dev", 6)
        w5 = CWWorld(cfg_ack, rngs5, sids5)
        w5._ack_bump(user, direction)
        return float(w5._m_h[user]), float(w5._v_h[user])

    # 写卡时手算（★ 已在 §7 事后修订区留痕：第一版漏了 step 末尾的**信道信念滚动**
    #   `_m_h*=rho` / `_v_h=rho²v+(1−rho²)`，导致拿裸值去比 step 后的值 ⇒ 误判 FAIL）
    _g = 1.0 / (1.0 + float(CWWorld.ACK_VAR))
    expect_m = -_g * 1.5          # 裸效应：Δm = g·direction
    expect_v = 1.0 - _g           # 裸效应：v' = (1−g)·1

    bare_m, bare_v = _bare(0, -1.5)
    g.check("4j-1 裸 `_ack_bump` 与解析式逐位对账（g = 1/(1+ACK_VAR)）",
            abs(bare_m - expect_m) < 1e-12 and abs(bare_v - expect_v) < 1e-12,
            {"bare_m": bare_m, "bare_v": bare_v,
             "analytic_m": expect_m, "analytic_v": expect_v,
             "ACK_VAR": float(CWWorld.ACK_VAR)})

    m0, v0, rho0 = _after(0, False)
    # step 后必须 = 裸效应经一次信念滚动
    exp_m0 = expect_m * rho0
    exp_v0 = rho0 * rho0 * expect_v + (1.0 - rho0 * rho0)
    g.check("4j-2 ★判别支：SEND(0) 失败 ⇒ 信念下降落在**用户 0** 上，用户 1 逐位不变"
            "（且数值 = 裸效应经一步滚动，公式对账）",
            (m0[0] < -1e-6) and (m0[1] == 0.0)
            and abs(m0[0] - exp_m0) < 1e-12 and abs(v0[0] - exp_v0) < 1e-12,
            {"m_after_fail_send0": m0, "v_after_fail_send0": v0,
             "expect_m0_step": exp_m0, "expect_v0_step": exp_v0, "rho": rho0,
             "legacy_would_give": [0.0, exp_m0],
             "why_it_fails_on_legacy": "旧实现 _ack_bump(i,-1.5) 的 i = n_users-1 = 1 ⇒ m=[0.0, −0.78] ⇒ 本条 FAIL"})

    m1, v1, rho1 = _after(1, False)
    g.check("4k 镜像：SEND(1) 失败 ⇒ 下降落在用户 1 上，用户 0 逐位不变，且与 4j 逐位对称"
            "（★ 旧实现在此**恰好** PASS ⇒ 单靠本条无分辨力，必须与 4j 同时在场）",
            (m1[1] < -1e-6) and (m1[0] == 0.0)
            and abs(m1[1] - m0[0]) < 1e-12 and abs(v1[1] - v0[0]) < 1e-12,
            {"m_after_fail_send1": m1, "v_after_fail_send1": v1,
             "delta_vs_4j_m": abs(m1[1] - m0[0]), "rho": rho1,
             "note": "legacy: i=n_users-1=1=act.user ⇒ 恰好正确 ⇒ 本条单独不构成判据"})

    # 4l：真实 episode 上的**逐用户归属恒等** —— 两处**独立代码**对账：
    #   runner 侧 `n_attempted_by_user - n_delivered_by_user`（从 SlotRecord 数）
    #   vs world 侧 `n_nack_by_user`（在 SEND 支按 act.user 累加）。
    # ★ 明写它能证什么、不能证什么：能封"计数归属"，**不能**证明 `_ack_bump` 的实参正确
    #   （计数器和 `_ack_bump` 是两处代码）⇒ 那一维由 4j/4k 的行为观察钉死。
    from .policies import AlwaysSend as _AlwaysSend

    att_ok, detail_4l = [], {}
    for ep4 in (100, 101):
        res4 = run_episode(cfg_ack, _AlwaysSend(), split="dev", episode=ep4,
                          budget_total=_budget(cfg_ack, 0.5), rho=0.98)
        att = [int(x) for x in res4.n_attempted_by_user]
        dlv = [int(x) for x in res4.n_delivered_by_user]
        nk = [int(x) for x in (res4.n_nack_by_user or [])]
        ak = [int(x) for x in (res4.n_ack_by_user or [])]
        n_u = len(att)
        ok_ident = (len(nk) == n_u == len(ak) == len(dlv)
                    and all(att[i] == dlv[i] + nk[i] for i in range(n_u))
                    and sum(nk) == int(res4.n_nack) and sum(ak) == int(res4.n_ack)
                    and sum(dlv) == int(res4.n_ack))
        # 荒谬性检查（R1.10 的直接指纹）：真实失败 > 0 而计数侧 = 0 ⇒ 立刻停
        absurd = [i for i in range(n_u) if (att[i] - dlv[i]) > 0 and nk[i] == 0]
        has_fail = int(res4.n_nack) > 0
        att_ok.append(bool(ok_ident and not absurd and has_fail))
        detail_4l[f"ep{ep4}"] = {"attempted_by_user": att, "delivered_by_user": dlv,
                                 "ack_by_user(world)": ak, "nack_by_user(world)": nk,
                                 "n_ack": int(res4.n_ack), "n_nack": int(res4.n_nack),
                                 "absurd_users": absurd, "had_nack": has_fail}
    g.check("4l 真实 episode 上逐用户归属恒等（attempt−deliver == nack，且逐用户/全局双对账）",
            all(att_ok),
            {"per_episode": detail_4l,
             "criterion": "Σ_i nack_i = n_nack 且逐用户 att_i = delv_i + nack_i 且无'真实失败>0 而计数=0'",
             "★": "本条只封计数归属；实参正确性由 4j/4k 的行为观察覆盖"})


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


# ============================================================ 4b 精确贝叶斯（R1.12）
def counterexample_4b_ack_exact(cfg: CWConfig, g: Gate) -> None:
    """R1.12：`ack_mode="exact"` 的网格后验 —— 4m / 4n / 4o / 4p / 4q。

    ★ 这一组不是"新功能验收"，而是**先把数值实现钉死**：如果网格传播本身掺了信息
      （方差自己长或自己缩），或网格与解析积分对不上，那么后面面板上的任何差
      都不能归因到"精确 vs 近似"。
    ★ 为什么 4m/4n/4q 三条一起要：它们分别封住**初值 / 单次更新 / 无观测传播**三种错误。
      只做其中一条时，另两种错误可以互相抵消（用错的核 + 用错的初值仍可能凑出对的矩）。
    """
    kappa = float(cfg.channel.kappa)
    star = float(cfg.channel.log_h_star)

    # ---------------- 4m 初值：网格 ≡ 公开先验 N(0,1)（截断误差必须远小于一切待测效应）
    gb = GridBelief(2, rho=0.98, kappa=kappa, log_h_star=star,
                    init_mean=0.0, init_std=float(cfg.channel.h_log_init_std))
    m0, v0 = gb.moments()
    leak = gb.mass_outside(-6.0, 6.0)
    g.check("4m 网格初值 ≡ 公开先验：mean=0（<1e-12）、var=1（<1e-9）、质量归一",
            abs(float(m0[0])) < 1e-12 and abs(float(m0[1])) < 1e-12
            and abs(float(v0[0]) - 1.0) < 1e-9 and abs(float(v0[1]) - 1.0) < 1e-9
            and abs(float(gb.p.sum(axis=1).min()) - 1.0) < 1e-12,
            {"mean": [float(x) for x in m0], "var": [float(x) for x in v0],
             "mass_outside_6sd": [float(x) for x in leak],
             "grid": {"n": gb.N_GRID, "half": gb.HALF, "dx": round(gb.dx, 6)}})

    # ---------------- 4n 单次更新：网格 ≡ 高分辨数值积分（**同支撑**的独立实现）
    # ★ 第一次跑 FAIL（worst 3.27e-8 > 1e-8）。读完 FAIL 才判"闸错还是物错"：
    #   参考积分在 n=4001/8001/16001/32001 与 half=12/20 上**逐位相同** ⇒ 参考已收敛；
    #   把参考限制到网格的支撑 [−8,8] 再比 ⇒ 误差降到 5.3e-9 ⇒ **那 3.3e-8 全是网格支撑截断**，
    #   不是网格本身不准。⇒ 判据改为"与**同支撑**参考对账 <1e-8"，并**同时报**支撑截断量
    #   （它是模型选择，不是数值误差；若哪天超过 1e-6 就该扩域，而不是调阈值）。
    def _quad(prior_m: float, prior_v: float, success: bool, n: int = 32001,
              half: float = 12.0, support: tuple[float, float] | None = None
              ) -> tuple[float, float]:
        xs = np.linspace(-half, half, n)
        p = 1.0 / (1.0 + np.exp(-kappa * (xs - star)))
        lik = np.log(np.clip(p if success else 1.0 - p, 1e-300, None))
        w = np.exp(-0.5 * (xs - prior_m) ** 2 / prior_v + lik)
        if support is not None:
            lo, hi = support
            w = np.where((xs >= lo) & (xs <= hi), w, 0.0)
        w = w / w.sum()
        mm = float((w * xs).sum())
        return mm, float((w * (xs - mm) ** 2).sum())

    cases = [(0.0, 1.0, True), (0.0, 1.0, False), (0.0, 0.25, True), (1.5, 1.0, True),
             (1.5, 1.0, False), (-2.0, 1.0, False)]
    rows, worst, worst_trunc = [], 0.0, 0.0
    for pm, pv, ok in cases:
        gx = GridBelief(1, rho=0.98, kappa=kappa, log_h_star=star,
                        init_mean=pm, init_std=np.sqrt(pv))
        gx.observe_binary(0, ok)
        mgx, vgx = gx.moments()
        mq, vq = _quad(pm, pv, ok, support=(-gx.HALF, gx.HALF))
        mf, vf = _quad(pm, pv, ok)
        dm, dv = float(mgx[0]) - mq, float(vgx[0]) - vq
        worst = max(worst, abs(dm), abs(dv))
        worst_trunc = max(worst_trunc, abs(mf - mq), abs(vf - vq))
        # Laplace（一阶高斯）在同一先验处的值 —— **只记录**，用来量化"一阶近似自己错多少"
        p_ = 1.0 / (1.0 + np.exp(-kappa * (pm - star)))
        info = kappa * kappa * p_ * (1.0 - p_)
        lap_m = pm + pv * kappa * ((1.0 - p_) if ok else -p_) / (1.0 + pv * info)
        lap_v = pv / (1.0 + pv * info)
        rows.append({"prior": [pm, pv], "ack": ok,
                     "grid": [float(mgx[0]), float(vgx[0])], "quad_same_support": [mq, vq],
                     "quad_full_support": [mf, vf],
                     "err_vs_same_support": [dm, dv],
                     "truncation_shift": [mf - mq, vf - vq],
                     "laplace": [lap_m, lap_v],
                     "laplace_err_vs_quad": [lap_m - mq, lap_v - vq]})
    g.check("4n 单次 ACK/NACK 后网格后验 ≡ **同支撑** 32001 点数值积分（|Δm|,|Δv| < 1e-8）",
            worst < 1e-8, {"worst_abs_err": worst,
                           "worst_support_truncation_shift": worst_trunc, "cases": rows,
                           "note": "`laplace_err_vs_quad` 是一阶高斯近似**自身**的误差（与网格无关），"
                                   "在 m=0 处 Δm 差 0.106、Δv 差 0.133 ⇒ §2 的解析式只能当**方向**判据；"
                                   "`truncation_shift` 是网格把 [−8,8] 之外当作不可能的代价（本批 ≤3.3e-8）"})

    # ---------------- 4o 归属：exact 臂也必须被"非对称"破坏一次（R1.11 纪律迁移）
    cfg_exact = CWConfig(plant=cfg.plant,
                         channel=type(cfg.channel)(**{**cfg.channel.__dict__, "ack_mode": "exact"}),
                         protocol=cfg.protocol, root_seed=cfg.root_seed)

    def _after_exact(send_user: int, force_success: bool):
        rngs4, sids4 = make_episode_rngs(cfg_exact.root_seed, "dev", 7)
        w4 = CWWorld(cfg_exact, rngs4, sids4)
        w4.budget_total = 50
        w4._log_h_true[:] = 30.0 if force_success else -30.0
        w4.step(Action("SEND", send_user))
        return [float(x) for x in w4._m_h], [float(x) for x in w4._v_h]

    me0, ve0 = _after_exact(0, False)
    me1, ve1 = _after_exact(1, False)
    # ★ 第一次跑 FAIL：判据写成 `me0[1] == 0.0`（精确零）。网格传播后的"未受影响用户"
    #   均值是 `−1.07e-16`（`ρ×(浮点零)` 的结果），**不可能**是精确 0 —— 这是判据错，不是实现错。
    #   改成 1e-15（仍比任何有效效应小 14 个数量级），并在 TEXT 里写明"网格的零是浮点零"。
    g.check("4o exact 臂归属：SEND(0) 失败只压 u0 的均值、u1 保持 0（<1e-15，网格浮动零）；"
            "镜像同理且两者对称",
            me0[0] < -1e-9 and abs(me0[1]) < 1e-15 and me1[1] < -1e-9 and abs(me1[0]) < 1e-15
            and abs(me1[1] - me0[0]) < 1e-12,
            {"exact_fail_send0_m": me0, "exact_fail_send1_m": me1,
             "exact_fail_send0_v": ve0, "exact_fail_send1_v": ve1,
             "note": "u1 的均值 −1.07e-16 与方差 0.9999999999999212 都是"
                     "'只做了 AR 传播'的浮点残差 ⇒ 该用户确实没收到任何 ACK 信息"})

    # ---------------- 4p 模式隔离：`ack_mode="approx"` 必须逐位走老公式
    rngs5, sids5 = make_episode_rngs(cfg.root_seed, "dev", 7)
    w5 = CWWorld(cfg, rngs5, sids5)
    w5.budget_total = 50
    w5._log_h_true[:] = 30.0            # 强制成功：这样两条路径的输入完全相同
    mv, vv, dirs = 0.0, float(w5._v_h[0]), []
    for _ in range(5):
        w5.step(Action("SEND", 0))
        gg = vv / (vv + float(CWWorld.ACK_VAR))
        mv = mv + gg * ((mv + 1.0) - mv)
        vv = (1.0 - gg) * vv
        # step 末尾还有一次信念滚动（见 4j-1 的教训：漏了它必然假 FAIL）
        mv = w5.rho * mv
        vv = w5.rho * w5.rho * vv + (1.0 - w5.rho * w5.rho)
        dirs.append((float(w5._m_h[0]), float(w5._v_h[0])))
    g.check("4p 模式隔离：ack_mode='approx' 时 5 次 ACK 后的 (m,v) ≡ 老公式逐位（`_grid is None`）",
            w5._grid is None
            and abs(dirs[-1][0] - mv) < 1e-15 and abs(dirs[-1][1] - vv) < 1e-15,
            {"trajectory": dirs, "closed_form_last": [mv, vv], "ACK_VAR": float(CWWorld.ACK_VAR)})

    # ---------------- 4q 无观测传播 ≡ 闭式矩递推（200 步，两档 rho）
    rows_q = []
    ok_q = True
    for rho in (0.98, 0.85):
        gq = GridBelief(1, rho=rho, kappa=kappa, log_h_star=star)
        mq0, vq0 = gq.moments()
        m_ref, v_ref = float(mq0[0]), float(vq0[0])
        worst_m = worst_v = 0.0
        for _ in range(200):
            gq.propagate()
            m_ref *= rho
            v_ref = rho * rho * v_ref + (1.0 - rho * rho)
            m_cur, v_cur = gq.moments()
            worst_m = max(worst_m, abs(float(m_cur[0]) - m_ref))
            worst_v = max(worst_v, abs(float(v_cur[0]) - v_ref))
        ok_q = ok_q and worst_m < 1e-12 and worst_v < 1e-12
        rows_q.append({"rho": rho, "worst_dm": worst_m, "worst_dv": worst_v,
                       "v_final": float(gq.moments()[1][0]), "v_closed": v_ref})
    g.check("4q 网格 AR 传播 ≡ 闭式 `m'=ρm`、`v'=ρ²v+(1−ρ²)`（200 步，<1e-12）", ok_q, rows_q)


# ============================================================ R1.13：`ack_var` 接线
def counterexample_4b2_ack_var(cfg: CWConfig, g: Gate) -> None:
    """R1.13：`ack_var` 这个标量配置项**真的接线了吗**？（4r / 4s / 4t）

    ★ 动机（R12）：见参数先问"它接线了吗"。`4j-1`/`4p` 都是用 `CWWorld.ACK_VAR` 做解析对账，
      一旦 `_ack_bump` 改读配置，那两条就**不再覆盖**"配置值是否被用上"这一维：
      配置项写得再漂亮，实现里仍读类常量，两条闸照样全 PASS。
    """
    from .config import ChannelConfig

    # ---------------- 4r 默认值锁定：配置默认必须 == 类常量（否则 4j-1/4p 失去意义）
    d_cfg = float(ChannelConfig().ack_var)
    d_cls = float(CWWorld.ACK_VAR)
    bad_cfg = []
    for bad_v in (0.0, -1.0, float("nan"), float("inf")):
        c_bad = CWConfig(plant=cfg.plant,
                         channel=type(cfg.channel)(**{**cfg.channel.__dict__, "ack_var": bad_v}),
                         protocol=cfg.protocol, root_seed=cfg.root_seed)
        # validate 的返回是"错项列表"，非空即拒绝
        if not any("ack_var" in str(x) for x in c_bad.v11_validate()):
            bad_cfg.append(bad_v)
    g.check("4r `ChannelConfig().ack_var` == `CWWorld.ACK_VAR`（默认同值，否则 4j-1/4p 参照物失效）"
            "且 ack_var ≤0 / NaN / inf 一律被 validate 拒绝",
            d_cfg == d_cls and not bad_cfg,
            {"config_default": d_cfg, "class_constant": d_cls,
             "unvalidated_bad_values": bad_cfg,
             "why_it_matters": "配置默认漂走 ⇒ 4j-1/4p 仍在算 0.63 的解析式，"
                               "而被测物已经用别的值 ⇒ 两条闸静默失去分辨力"})

    # ---------------- 4s ★判别：非默认 ack_var 必须真的进公式
    def _bare_with(ack_var: float, user: int = 0, direction: float = -1.5):
        c = CWConfig(plant=cfg.plant,
                     channel=type(cfg.channel)(**{**cfg.channel.__dict__, "ack_var": float(ack_var)}),
                     protocol=cfg.protocol, root_seed=cfg.root_seed)
        rngs6, sids6 = make_episode_rngs(c.root_seed, "dev", 8)
        w6 = CWWorld(c, rngs6, sids6)
        w6._ack_bump(user, direction)
        return float(w6._m_h[user]), float(w6._v_h[user]), float(w6.ack_var)

    av_new = 1.726
    m_new, v_new, seen = _bare_with(av_new)
    g_new = 1.0 / (1.0 + av_new)
    g_old = 1.0 / (1.0 + float(CWWorld.ACK_VAR))
    exp_m, exp_v = -g_new * 1.5, 1.0 - g_new
    m_old, v_old, _ = _bare_with(float(CWWorld.ACK_VAR))
    gap = max(abs(m_new - m_old), abs(v_new - v_old))
    g.check("4s ★判别：`ack_var=1.726` 的裸 `_ack_bump` ≡ `g=1/(1+1.726)` 解析式，"
            "且与默认 0.63 的结果差 >1e-3（接线是死的时候本条必 FAIL）",
            abs(seen - av_new) < 1e-15
            and abs(m_new - exp_m) < 1e-12 and abs(v_new - exp_v) < 1e-12
            and gap > 1e-3,
            {"ack_var_seen_by_world": seen,
             "bare_m_new": m_new, "bare_v_new": v_new,
             "analytic_m_new": exp_m, "analytic_v_new": exp_v,
             "bare_m_default": m_old, "bare_v_default": v_old,
             "max_abs_gap_vs_default": gap, "g_new": g_new, "g_old": g_old})

    # ---------------- 4t 只准动方差：方向不对称必须仍严格是 1.0 / −1.5
    m_ack, v_ack, _ = _bare_with(av_new, 0, +1.0)
    m_nack, v_nack, _ = _bare_with(av_new, 0, -1.5)
    ratio = abs(m_nack) / abs(m_ack) if m_ack else float("inf")
    g.check("4t 非默认 ack_var 下 ACK/NACK 的方向不对称仍严格为 1.0 / −1.5（比值 1.5，≤1e-12）",
            abs(ratio - 1.5) < 1e-12 and m_ack > 0.0 and m_nack < 0.0
            and abs(v_ack - v_nack) < 1e-15,
            {"m_ack": m_ack, "m_nack": m_nack, "abs_ratio": ratio,
             "v_ack": v_ack, "v_nack": v_nack,
             "note": "同一 (m,v) 下 ACK 与 NACK 的**方差**收缩必须相同（只差方向）"})


def counterexample_4b3_exact_mainline(cfg: CWConfig, g: Gate) -> None:
    """R1.14：`ack_mode="exact"` 升为**研究基线** —— 4u / 4v / 4w / 4x。

    预注册：`docs/research/CW-R1.14-精确基线-预注册-v0.1.md` §4.4/§4.5。

    ★ 这四条全是**不变式**，不许把"exact 更可信"这种科学结论写成闸
      （那种结论属于 `M2` 判据，且带否定条件，由入口脚本判）。
    """
    from . import config as Cmod

    # ---------------- 4u 主线 profile **真的接线了吗**（不是名字对了就行）
    ch_default = type(cfg.channel)()
    main = Cmod.mainline_config()
    other_ch_same = all(
        getattr(main.channel, f) == getattr(cfg.channel, f)
        for f in main.channel.__dict__ if f != "ack_mode")
    plant_ok = (main.plant == Cmod.BOUNDED_PLANT)
    rngs_u, sids_u = make_episode_rngs(cfg.root_seed, "dev", 3)

    def _wired(c: CWConfig) -> tuple[bool, bool]:
        rr, ss = make_episode_rngs(c.root_seed, "dev", 3)   # 每次新建，避免子流被上一次吃掉
        w = CWWorld(c, rr, ss)
        return (str(c.channel.ack_mode) == str(Cmod.MAINLINE_ACK_MODE),
                w._grid is not None)

    label_ok, grid_ok = _wired(main)
    # ★ 活体负控：把常量临时改成 "approx" ⇒ 同一个谓词必须**失去** grid_ok
    orig_mode = Cmod.MAINLINE_ACK_MODE
    neg = {}
    try:
        Cmod.MAINLINE_ACK_MODE = "approx"
        neg_label, neg_grid = _wired(Cmod.mainline_config())
        neg = {"label_ok": neg_label, "grid_ok": neg_grid}
    finally:
        Cmod.MAINLINE_ACK_MODE = orig_mode
    restored_label, restored_grid = _wired(Cmod.mainline_config())
    g.check("4u 主线 profile 接线：`mainline_config().ack_mode==MAINLINE_ACK_MODE=='exact'`、"
            "plant==BOUNDED_PLANT、默认值仍 'approx'（未漂）、其余 channel 字段逐字段相同、"
            "真建出网格后验；★活体负控：常量改成 'approx' 时 `grid_ok` 必须 False",
            label_ok and grid_ok and other_ch_same and plant_ok
            and ch_default.ack_mode == "approx"
            and (neg["label_ok"] is True and neg["grid_ok"] is False)
            and (restored_label and restored_grid),
            {"MAINLINE_ACK_MODE": str(Cmod.MAINLINE_ACK_MODE),
             "mainline_ack_mode": str(main.channel.ack_mode),
             "default_ack_mode": str(ch_default.ack_mode),
             "other_channel_fields_identical": bool(other_ch_same),
             "plant_is_bounded": bool(plant_ok),
             "grid_built": bool(grid_ok),
             "negative_control": neg,
             "restored": {"label_ok": restored_label, "grid_ok": restored_grid},
             "note": "`grid_ok=False` 是**唯一**能抓「常量改了但 world 没接网格」的观察点 —— "
                     "只查名字（`label_ok`）在负控里也 True"})

    # ---------------- 4v exact 模式：`_m_h/_v_h` ≡ 网格矩（逐位），且 `ack_var` 成为死参数
    cfg_exact = Cmod.mainline_config(plant=cfg.plant, protocol=cfg.protocol,
                                     root_seed=cfg.root_seed)

    def _run_ops(c: CWConfig) -> tuple[list[dict], tuple]:
        rngs5, sids5 = make_episode_rngs(c.root_seed, "dev", 5)
        w5 = CWWorld(c, rngs5, sids5)
        w5.budget_total = 50
        snap = []
        # ★ 首次跑 `ValueError: 后验质量非法（0.0）` —— 读完判"闸错还是物错"：
        #   本闸为了**确定性**成败把 `_log_h_true` 强推成 ±30（4o 的写法），
        #   ⇒ 随后 PROBE 的读数 30.0 落在网格 [-8,8] 之外，高斯似然全域下溢。
        #   ⇒ 是**测试构造**的问题（把 h 推到物理工作域外），不是 4v 要测的东西。
        #   ★ 但它暴露了一个**真实行为差异**：exact 模式对支撑外读数是**硬报错**，
        #     approx 是**软退化**。本轮不改（改 = 换基线），只记进交付文档的限制节。
        ops = [("SEND", 0, 30.0), ("SEND", 1, -30.0), ("PROBE", 0, "reset"), ("HOLD", None, None)]
        for kind, usr, force in ops:
            if force == "reset":
                w5._log_h_true[:] = 0.0     # 回到工作域，让 PROBE 读到 ~N(0, 0.05²)
            elif force is not None:
                w5._log_h_true[usr] = force
            w5.step(Action(kind, usr))
            if w5._grid is None:        # approx 模式没有网格 ⇒ 这一支只取轨迹哈希
                snap.append({"op": f"{kind}{'' if usr is None else usr}",
                             "grid": None})
                continue
            gm, gv = w5._grid.moments()
            snap.append({"op": f"{kind}{'' if usr is None else usr}",
                         "m_sync_equal": [float(a) == float(b)
                                          for a, b in zip(w5._m_h, gm)],
                         "v_sync_equal": [float(a) == float(b)
                                          for a, b in zip(w5._v_h, gv)]})
        # ★★ 首次跑 FAIL：`bel_a == bel_a_alt`（approx 下换 ack_var 轨迹没变）——
        #   读完判"闸错还是物错"：**动作序列是我手写的常量表**，恒与参数无关
        #   ⇒ 用动作哈希判"参数有没有被消费"是**恒真判据**（#46 族：对象选错）。
        #   改判**信念终值**：参数被消费 ⇒ (m,v) 变；不被消费 ⇒ (m,v) 逐位不变。
        return snap, (tuple(float(x) for x in w5._m_h), tuple(float(x) for x in w5._v_h))

    snap_e, bel_e = _run_ops(cfg_exact)
    sync_ok = all(all(s["m_sync_equal"]) and all(s["v_sync_equal"]) for s in snap_e)
    # `ack_var` 在 exact 模式下**不该被消费** ⇒ 换值不得改**信念终值**（这才叫"对象不存在"）
    alt = type(cfg_exact.channel)(**{**cfg_exact.channel.__dict__, "ack_var": 1.726})
    cfg_exact_alt = CWConfig(plant=cfg_exact.plant, channel=alt,
                            protocol=cfg_exact.protocol, root_seed=cfg_exact.root_seed)
    _, bel_e_alt = _run_ops(cfg_exact_alt)
    dead_param = (bel_e == bel_e_alt)
    # 反向：approx 模式下同一换值**必须**改信念终值（否则上面的"死参数"是假的）
    cfg_apx = cfg
    alt_a = type(cfg_apx.channel)(**{**cfg_apx.channel.__dict__, "ack_mode": "approx",
                                    "ack_var": 1.726})
    cfg_apx_alt = CWConfig(plant=cfg_apx.plant, channel=alt_a,
                           protocol=cfg_apx.protocol, root_seed=cfg_apx.root_seed)
    _, bel_a = _run_ops(cfg_apx)
    _, bel_a_alt = _run_ops(cfg_apx_alt)
    # approx 下 4j-1/4p 必须仍 PASS（否则"SKIP"就变成了掩盖）
    g_apx = Gate()
    counterexample_4(cfg_apx, g_apx)
    counterexample_4b_ack_exact(cfg_apx, g_apx)
    apx_rows = {r["name"][:4]: bool(r["pass"]) for r in g_apx.rows}
    j41_ok = all(v for k, v in apx_rows.items() if k.startswith("4j-1"))
    p4_ok = bool(apx_rows.get("4p 模", False))
    skipped = ["4j-1 `_ack_bump` ≡ 一阶解析式", "4p approx 模式隔离（5 次 ACK ≡ 老公式）"]
    g.check("4v exact 模式下 `_m_h/_v_h` 与 `GridBelief.moments()` **逐位**相等"
            "（4 种操作后各一次）；`ack_var` 在 exact 下是**死参数**（换 1.726 后信念终值逐位不变），"
            "而 approx 下换值**必须**改信念终值 ⇒ `4j-1`/`4p` 的对象在 exact 下**不存在**，"
            "显式记 SKIP-NOT-APPLICABLE（清单非空 + approx 下两条仍 PASS）",
            sync_ok and dead_param and (bel_a != bel_a_alt) and j41_ok and p4_ok
            and len(skipped) == 2,
            {"sync_snapshots": snap_e,
             "exact_belief_vs_alt_ackvar": [list(bel_e), list(bel_e_alt)],
             "approx_belief_vs_alt_ackvar": [list(bel_a), list(bel_a_alt)],
             "approx_4j1_pass": bool(j41_ok), "approx_4p_pass": bool(p4_ok),
             "SKIP_NOT_APPLICABLE_under_exact": skipped,
             "note": "SKIP 的依据是**行为**（换 ack_var 不改信念终值），不是注释；"
                     "所以它不是静默 PASS，而是『对象不存在』的可证伪记录"})

    # ---------------- 4w 纯记录**对齐**：`log_h_history[t]` == 第 t 槽开始时真的读到的那份真值
    rngs6, sids6 = make_episode_rngs(cfg.root_seed, "dev", 6)
    w6 = CWWorld(cfg, rngs6, sids6)
    w6.budget_total = 50
    w6.truth_override = {"h": np.array([5.0, -5.0])}
    w6.step(Action("HOLD"))
    rec0 = list(w6.log_h_history[0])
    w6.truth_override = None
    before1 = [float(x) for x in w6._log_h_true]
    w6.step(Action("HOLD"))
    rec1 = list(w6.log_h_history[1])
    after1 = [float(x) for x in w6._log_h_true]
    len_ok = (len(w6.log_h_history) == len(w6.records) == 2)
    g.check("4w 纯记录对齐：第 0 槽记的是 override 注入值 (5,−5)；第 1 槽记的是**槽开始**的"
            "`_log_h_true`（而非推进后的值，两点必须不相等）；长度 == `len(records)`",
            rec0 == [5.0, -5.0] and rec1 == before1 and rec1 != after1 and len_ok,
            {"slot0_recorded": rec0, "slot1_recorded": rec1,
             "slot1_truth_at_start": before1, "slot1_truth_after_advance": after1,
             "lengths": [len(w6.log_h_history), len(w6.records)],
             "note": "把 append 挪到槽末 ⇒ `rec1` 等于 `after1` ⇒ FAIL（#52/#59 的 off-by-one 族）"})

    # ---------------- 4x 纯记录**纯净**：关掉记录后 SlotRecord 逐字段不变
    class _Null(list):
        def append(self, _x) -> None:      # noqa: ANN001
            return None

    def _scripted(rec_on: bool) -> list[tuple]:
        rngs7, sids7 = make_episode_rngs(cfg.root_seed, "dev", 7)
        w7 = CWWorld(cfg, rngs7, sids7)
        w7.budget_total = 60
        if not rec_on:
            w7._log_h_hist = _Null()       # type: ignore[assignment]
        acts = [Action("PROBE", 0), Action("SEND", 0), Action("SEND", 1),
                Action("HOLD"), Action("PROBE", 1), Action("SEND", 1)]
        for a in acts:
            w7.step(a)
        return [(r.t, r.action, r.user, r.data_delivered, r.cost_est, r.cost_true,
                 r.budget_left) for r in w7.records]
    on, off = _scripted(True), _scripted(False)
    g.check("4x 纯记录纯净：关掉 `_log_h_hist` 后 6 步的 SlotRecord 七元组逐位相同"
            "（纯记录不得改轨迹/记账）",
            on == off,
            {"n_slots": len(on), "identical": bool(on == off),
             "first_diff": (next((i for i, (a, b) in enumerate(zip(on, off)) if a != b), None)
                            if on != off else None)})




# ============================================================ R1.15 送达缺口归因
def counterexample_4b4_delivery_attr(cfg: CWConfig, g: Gate) -> None:
    """R1.15：归因用**纯记录聚合量**的两条不变式 —— 4y（对齐）/ 4z（纯净）。

    预注册：`docs/research/CW-R1.15-送达缺口归因-预注册-v0.1.md` §2/§5。

    ★ 这两条只钉"记录得对不对、纯不纯"，**不许**把"哪个机制成立"写成闸
      （那是入口脚本的 `M` 判据，带否定条件）。
    """
    from types import SimpleNamespace

    from . import runner as Rmod

    # ---------------- 4y 聚合量**对齐**：边界（左闭右开 + p=1 并入末箱）+ 真实轨迹对账
    edges = list(Rmod.P_TRUE_BIN_EDGES)
    # (a) 直接喂合成记录：答案手算（p = 0 / 0.0999 / 0.1 / 0.4999 / 0.5 / 0.9999 / 1.0）
    ps = [0.0, 0.0999, 0.1, 0.4999, 0.5, 0.9999, 1.0]
    ds = [0, 0, 1, 0, 1, 1, 1]
    fake = [SimpleNamespace(action="SEND", user=0, p_true=p, data_delivered=d)
            for p, d in zip(ps, ds)]
    agg = Rmod.p_true_aggregates(fake, 2)
    cnt0 = agg["n_attempt_in_bin_by_user"][0]
    # 左闭右开：0 → 箱0；0.0999 → 箱0；0.1 → 箱1；0.4999 → 箱4；0.5 → 箱5；
    # 0.9999 → 箱9；1.0（饱和）→ 箱9
    exp_cnt0 = [2, 1, 0, 0, 1, 1, 0, 0, 0, 2]
    exp_mean0 = sum(ps) / len(ps)
    bin_ok = (cnt0 == exp_cnt0
              and agg["n_attempt_in_bin_by_user"][1] == [0] * 10
              and agg["n_delivered_in_bin_by_user"][0][1] == 1
              and agg["n_delivered_in_bin_by_user"][0][9] == 2
              and agg["n_attempt_in_bin_by_user"][0].__len__() == len(edges) - 1)
    mean_ok = abs(agg["attempt_p_true_mean_by_user"][0] - exp_mean0) < 1e-15
    q_ok = (agg["attempt_p_true_q"] == sorted(agg["attempt_p_true_q"])
            and len(agg["attempt_p_true_q"]) == len(Rmod.P_TRUE_QUANTILES))
    # (b) 真实轨迹对账：逐用户计数必须等于独立代码路径数出来的 SEND 次数
    from .world import Action as _A

    rngs, sids = make_episode_rngs(cfg.root_seed, "dev", 9)
    w = CWWorld(cfg, rngs, sids)
    w.budget_total = 40
    script = [_A("SEND", 0), _A("SEND", 1), _A("HOLD"), _A("PROBE", 0),
              _A("SEND", 1), _A("SEND", 0), _A("HOLD")]
    for a in script:
        w._log_h_true[:] = [0.0, 1.0] if a.kind != "PROBE" else [0.0, 1.0]
        w.step(a)
    agg2 = Rmod.p_true_aggregates(w.records, 2)
    n_by_user = [sum(agg2["n_attempt_in_bin_by_user"][i]) for i in (0, 1)]
    d_by_user = [sum(agg2["n_delivered_in_bin_by_user"][i]) for i in (0, 1)]
    ref_n = [sum(1 for r in w.records if r.action == "SEND" and r.user == i)
             for i in (0, 1)]
    ref_d = [sum(1 for r in w.records if r.action == "SEND" and r.user == i
                 and r.data_delivered) for i in (0, 1)]
    ref_mean = [float(np.mean([r.p_true for r in w.records
                               if r.action == "SEND" and r.user == i])) for i in (0, 1)]
    # p_true 只用固定 logh 值 ⇒ 两用户的 p 必须是 0.5 与 sigmoid(2)=0.8807970779778823
    p_expect = [0.5, float(1.0 / (1.0 + np.exp(-2.0)))]
    trail_ok = (n_by_user == ref_n and d_by_user == ref_d
                and all(abs(a - b) < 1e-15 for a, b in
                        zip(agg2["attempt_p_true_mean_by_user"], ref_mean))
                and all(abs(a - b) < 1e-12 for a, b in
                        zip(agg2["attempt_p_true_mean_by_user"], p_expect)))
    g.check("4y `p_true` 聚合量对齐：① 合成记录下 10 箱计数逐箱 = 手算（**左闭右开**、"
            "`p=0`→箱0、`p=0.1`→箱1、`p=1` 饱和→箱9）、均值/分位单调；"
            "② 真实 7 槽轨迹下逐用户尝试/送达计数 = **独立代码路径**（直接扫 `records`）重数结果，"
            "且固定 `logh∈{0,1}` 下逐用户 `p_true` 均值 = 0.5 / sigmoid(2)（1e-12）",
            bool(bin_ok and mean_ok and q_ok and trail_ok),
            {"bins_user0": cnt0, "expected_bins_user0": exp_cnt0,
             "mean_user0": agg["attempt_p_true_mean_by_user"][0],
             "expected_mean_user0": exp_mean0,
             "n_attempt_by_user": n_by_user, "ref_n_by_user": ref_n,
             "n_deliv_by_user": d_by_user, "ref_d_by_user": ref_d,
             "p_mean_by_user": agg2["attempt_p_true_mean_by_user"],
             "p_expected": p_expect,
             "n_bins": len(edges) - 1,
             "note": "箱号写错（如用左开右闭）会让 `p=0.1` 落箱0、`p=0.5` 落箱4 ⇒ 前两格必 FAIL"})

    # ---------------- 4z 记录**不越权** + 轨迹**纯净**
    #   ★ 首版只写了"聚合关/开 → 前 8 项逐位相同"，变异测试（把聚合改成 `recs.clear()`）
    #     抓不到 —— 因为聚合是在构造 `EpisodeResult` 的**实参求值**阶段调用的，那时
    #     其余字段已算完。⇒ 这条单独存在时是**空过**（#100）。
    #     改成两段：①信息边界（新记录不得进 `PUBLIC_HISTORY_FIELDS`，带**活体负控**）；
    #     ②轨迹纯净（防"未来把聚合挪进循环并消耗随机数"）。
    from . import world as Wmod

    whitelist = tuple(Wmod.PUBLIC_HISTORY_FIELDS)
    members = tuple(Wmod.VIEW_MEMBERS)
    leak_keys = {"p_true", "attempt_p_true_mean", "n_attempt_in_bin_by_user",
                 "n_delivered_in_bin_by_user", "cost_true", "w_realized"}
    rngs3, sids3 = make_episode_rngs(cfg.root_seed, "dev", 12)
    w3 = CWWorld(cfg, rngs3, sids3)
    w3.budget_total = 20
    w3.step(Action("SEND", 0))
    keys_now = set(w3.view().history[0].keys())
    boundary_ok = (keys_now == set(whitelist)) and not (keys_now & leak_keys)
    # ★ 活体负控：把 `p_true` 塞进白名单后**重建**一次记录 ⇒ 同一谓词必须失去 boundary_ok
    orig_wl = Wmod.PUBLIC_HISTORY_FIELDS
    try:
        Wmod.PUBLIC_HISTORY_FIELDS = orig_wl + ("p_true",)
        rngs4, sids4 = make_episode_rngs(cfg.root_seed, "dev", 12)
        w4 = CWWorld(cfg, rngs4, sids4)
        w4.budget_total = 20
        w4.step(Action("SEND", 0))
        neg_keys = set(w4.view().history[0].keys())
        neg_boundary = (neg_keys == set(Wmod.PUBLIC_HISTORY_FIELDS)) and not (neg_keys & leak_keys)
    finally:
        Wmod.PUBLIC_HISTORY_FIELDS = orig_wl

    def _once(off: bool) -> tuple:
        orig = Rmod.p_true_aggregates
        try:
            if off:
                Rmod.p_true_aggregates = lambda recs, n_users: {}   # type: ignore[assignment]
            pol = _scripted_policy()
            res = run_episode(cfg, pol, split="dev", episode=11, budget_total=6, rho=0.98)
            return (res.cost_est_per_step, res.cost_true_per_step, res.policy_trace_hash,
                    tuple(res.n_attempted_by_user), tuple(res.n_delivered_by_user),
                    res.n_probe, res.n_ack, res.n_nack,
                    tuple(res.attempt_p_true_mean_by_user))
        finally:
            Rmod.p_true_aggregates = orig

    on, offv = _once(False), _once(True)
    g.check("4z ①**信息边界**：`SlotRecord`/视图历史的键集合恰为 `PUBLIC_HISTORY_FIELDS`，"
            "新增的 `p_true` 聚合量一个都没进白名单（`VIEW_MEMBERS` 未增）；"
            "★活体负控：把 `p_true` 塞进白名单后重跑同一谓词必须**变 FAIL**；"
            "②**轨迹纯净**：把聚合换成空实现后，同种子 episode 的 `cost_est`/`cost_true`/轨迹哈希/"
            "逐用户尝试与送达/`n_probe`/ACK-NACK **逐位相同**",
            bool(boundary_ok and neg_boundary is False and (on[:-1] == offv[:-1])
                 and offv[-1] == () and len(members) == 7),
            {"history_keys": sorted(keys_now), "whitelist": list(whitelist),
             "negative_control_leaked": sorted(neg_keys & leak_keys),
             "negative_control_boundary_ok": bool(neg_boundary),
             "with_agg": list(on[:-1]), "without_agg": list(offv[:-1]),
             "agg_fields_with": list(on[-1]), "agg_fields_without": list(offv[-1]),
             "view_members": list(members),
             "note": "负控证明这条闸有分辨力；11 项『逐位相同』只防『聚合被挪进循环』"})


# ============================================================ 4b5 R1.17 尺度不变机制量
def _synth_mix(sign: int = -1, scale: int = 1):
    """4 用户合成面板：`ρ = .9/.7/.3/.1`、`ρ_a = .5`；`m` 全搬向差用户（`sign=-1`）。

    `scale` 把**所有计数**同乘（分配比例不变）⇒ 用来验"规模无关"。
    `ku` 手算：`E_user = −48/N`、`Mu = 160/(2N)` ⇒ `ku = −0.6`（与 `scale` 无关）。
    """
    ni_a = np.array([[100], [100], [100], [100]], dtype=float) * scale
    nd_a = np.array([[90], [70], [30], [10]], dtype=float) * scale
    m = np.array([-40, -40, 40, 40], dtype=float) * (1 if sign < 0 else -1) * scale
    ni_e = ni_a + m[:, None]
    return ni_e, nd_a.copy(), ni_a, nd_a


def _shift_nd(ni_e, nd_e, delta: int):
    """只改 `nd_e` 的整数扰动（从第一个非空格加减 `delta` 个送达）。"""
    nd = np.asarray(nd_e, dtype=int).copy()
    ni = np.asarray(ni_e, dtype=int)
    step = 1 if delta > 0 else -1
    for _ in range(abs(delta)):
        if step > 0:
            i = int(np.argmax((ni - nd)[:, 0]))
            if ni[i, 0] - nd[i, 0] <= 0:
                break
            nd[i, 0] += 1
        else:
            i = int(np.argmax(nd[:, 0]))
            if nd[i, 0] <= 0:
                break
            nd[i, 0] -= 1
    return nd


def counterexample_4b5_scale_invariance(cfg: CWConfig, g: Gate) -> None:
    """R1.17：机制量 `ku=E_user/Mu` 的**尺度不变性 / 有界性 / NaN 语义**（4aa–4ad）。

    预注册：`docs/research/CW-R1.17-尺度不变机制判据-预注册-v0.1.md` §5。
    ★ 只用合成矩阵，**不读** `results/`（闸必须能在干净树上跑）。
    ★ 4aa 的谓词写作"`ku` 不动而 `share` 变了"，并配 4ad 的活体负控
      （把分母换回 `|dr|` 必须 FAIL）—— 否则这条闸只是"碰巧相等"。
    """
    from .mech import dir_verdict, mech_quantities

    ni_e, nd_e0, ni_a, nd_a = _synth_mix(-1)
    q0 = mech_quantities(ni_e, nd_e0, ni_a, nd_a)          # dr = 0 ⇒ share 不可定义
    # A：把送达加满（`Σnd_e = Σni_e` ⇒ dr = +0.5），share 最小
    #   ★ 修订（§8-6）：首版用 +8 个送达（dr=+0.02），与 C 的 −0.0025 只差 **8×**，
    #     而断言写 `>10×` ⇒ 构造上不可达。加满后比值为 200×，与 S1 实测同阶。
    dA = int((np.asarray(ni_e, dtype=int) - np.asarray(nd_e0, dtype=int)).sum())
    qA = mech_quantities(ni_e, _shift_nd(ni_e, nd_e0, dA), ni_a, nd_a)
    # C：−1 个送达 ⇒ dr = −1/400 = −0.0025（符号与 A 相反）
    qC = mech_quantities(ni_e, _shift_nd(ni_e, nd_e0, -1), ni_a, nd_a)
    d_ku_max = max(abs(q["ku"] - q0["ku"]) for q in (qA, qC))
    sh0 = float(q0["shares"]["user"])
    shA = float(qA["shares"]["user"])
    shC = float(qC["shares"]["user"])
    sh_ok = bool(np.isnan(sh0) and shA != 0 and np.sign(shA) != np.sign(shC)
                 and abs(shC) > 10 * abs(shA))
    g.check("4aa **尺度不变性**：只改送达结果 `nd_e`（dr 从 0 → +0.5 → −0.0025，符号翻转）时，"
            "`ku = E_user/Mu` 三版**逐位相同**（<1e-12），而 `share = E/Δr` 从 NaN 变为有限并**符号翻转**"
            "（`|share_C| > 10|share_A|`）⇒ 换分母不是换名字，是真的换了尺度",
            bool(d_ku_max < 1e-12 and sh_ok),
            {"ku_three": [float(q0["ku"]), float(qA["ku"]), float(qC["ku"])],
             "dr_three": [float(q0["dr"]), float(qA["dr"]), float(qC["dr"])],
             "share_three": [str(sh0), shA, shC],
             "d_ku_max": float(d_ku_max), "expect_ku": -0.6,
             "note": "手算 ku=E_user/Mu=(-0.12)/0.2=-0.6 与实测对账"})

    # ---- 4ab 有界性与规模无关
    q10 = mech_quantities(*_synth_mix(-1, scale=10))
    qp = mech_quantities(*_synth_mix(+1))
    bound_ok = all(abs(q["ku"]) <= q["ku_bound"] + 1e-12
                   for q in (q0, qA, qC, q10, qp))
    scale_ok = abs(q10["ku"] - q0["ku"]) < 1e-12
    sign_ok = qp["ku"] > 0 > q0["ku"]
    g.check("4ab **有界性 + 规模无关**：① 五个构造全部满足 `|ku| ≤ 2·max|ρ_i−ρ_a|`"
            "（★ 首版上界写成 `max|…|` 漏了 TVD 的 1/2，实测 ku=−0.6 > 0.4 当场暴露）；"
            "② 所有计数同乘 10 后 `ku` 逐位不变；③ 搬运方向反转时 `ku` 符号反转（不是恒负）",
            bool(bound_ok and scale_ok and sign_ok),
            {"ku": q0["ku"], "ku_bound": q0["ku_bound"], "ku_scale10": q10["ku"],
             "ku_flipped": qp["ku"], "bound_ok": bool(bound_ok),
             "scale_ok": bool(scale_ok), "sign_ok": bool(sign_ok)})

    # ---- 4ac NaN 语义：Mu=0 ⇒ ku=NaN 且**不进**符号计数
    ni_same = np.array([[100], [100]], dtype=float)
    nd_same = np.array([[80], [20]], dtype=float)
    qz = mech_quantities(ni_same, nd_same, ni_same, nd_same)
    v = dir_verdict([qz["ku"]] * 10)
    g.check("4ac **NaN 语义**：两臂分配完全相同（`Mu=0`）时 `ku` 是 NaN，"
            "`dir_verdict` 把它记进 `n_nan` 而**不进** `n_neg`/`n_neu`，判词为 `DIRECTION-UNRESOLVED`"
            "（不许把'没发生'当成'零效应'，更不许当成'确认'）",
            bool(qz["Mu"] == 0.0 and np.isnan(qz["ku"]) and v["n_nan"] == 10
                 and v["n_neg"] == 0 and v["verdict"] == "DIRECTION-UNRESOLVED"),
            {"Mu": qz["Mu"], "ku": str(qz["ku"]), "n_nan": v["n_nan"],
             "n_neg": v["n_neg"], "verdict": v["verdict"]})

    # ---- 4ad 活体负控：把分母换回 `|dr|`（= R1.16 的 share 形态）⇒ 4aa 的谓词必须 FAIL
    #   ★ q0 的 `dr=0` 会让坏实现直接给 NaN ⇒ 用 A/C 两点（dr 分别为 +0.02 / −0.0025）比较，
    #     并把 q0 的 NaN 单独作为"旧形态在 dr=0 处不可定义"的证据。
    def _bad_share(q):
        d = abs(q["dr"])
        return q["E_user"] / d if d > 0 else float("nan")

    bad0, badA, badC = _bad_share(q0), _bad_share(qA), _bad_share(qC)
    d_bad2 = float(abs(badA - badC))
    g.check("4ad **活体负控**：把 `ku` 的分母换成 `|Δr|`（即退回 R1.16 的 `share` 形态）后，"
            "同一批矩阵的 A/C 两点**不再相同**（|−6.0 − (−48.0)| = 42 ≫ 1e-12），"
            "且 `dr=0` 那一点直接变成 NaN（旧形态在 Δr=0 处**不可定义**）"
            "⇒ 证明 4aa 有分辨力，不是'三个数碰巧相等'",
            bool(d_bad2 > 1e-6 and not np.isfinite(bad0)),
            {"bad_at_dr0": str(bad0), "bad_at_A": float(badA), "bad_at_C": float(badC),
             "d_bad_AC": d_bad2,
             "note": f"同一批计数矩阵：ku 三版全等（{q0['ku']:.6f}），"
                     f"share 版 A/C 相差 {d_bad2:.2f} 且 dr=0 处为 NaN"})


def _scripted_policy():
    """给 4z 用的确定性策略：不看视图，只按 t 轮转（避免把策略差异混进来）。"""
    from .world import Action as _A

    seq = [_A("SEND", 0), _A("SEND", 1), _A("HOLD"), _A("PROBE", 1),
           _A("SEND", 1), _A("SEND", 0)]

    def _pol(v):                                    # noqa: ANN001
        return seq[int(v.t) % len(seq)]

    _pol.name = "scripted_rot"
    return _pol


# ============================================================ 汇总
def run_all(cfg: CWConfig | None = None) -> dict:
    cfg = cfg or default_config()
    g = Gate()
    counterexample_1(cfg, g)
    counterexample_2(cfg, g)
    counterexample_3(cfg, g)
    counterexample_4(cfg, g)
    counterexample_4b_ack_exact(cfg, g)
    counterexample_4b2_ack_var(cfg, g)
    counterexample_4b3_exact_mainline(cfg, g)
    counterexample_4b4_delivery_attr(cfg, g)
    counterexample_4b5_scale_invariance(cfg, g)
    counterexample_5(cfg, g)
    counterexample_6(cfg, g)
    counterexample_7(cfg, g)
    counterexample_8(cfg, g)
    sanity_stability(cfg, g)
    return g.summary()
