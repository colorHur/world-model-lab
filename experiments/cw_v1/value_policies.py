# -*- coding: utf-8 -*-
"""R1 第 2 步的策略：**先买信息、再按信息发**（第二个 CSI 消费入口）。

为什么单独放一个文件（2026-10-21 20:5x 的现场决定）
---------------------------------------------------
`policies.py` 在一个文件里已经堆了 7 个"接线了但不生效"的版本，每次替换都要
动几百行；而本轮要交付的是**一个**能证明"CSI 改变动作"的策略。
分开之后，`policies.py` 保持 R0 原样（历史证据不受影响），本文件只是新增。
"""

from __future__ import annotations

import numpy as np

from .posterior import p_hat_vector
from .world import Action, SchedulerView


class BuyInfoThenSend:
    """先买信息、再按信息发。

    实测事实（本轮最硬的一条）
    --------------------------
    在 `PhatGreedy` / `ProbeOnIdle` 上跑完整条 256 槽，`p_hat` 的取值范围打印出来是
    **单点 [0.5000000, 0.5000000]** —— 不是公式错，是策略**从不探测** ⇒ 后验方差
    从不收缩 ⇒ 后验恒等于先验。而"从不探测"的根因是：发送永远优先且总能填满预算
    （B/T=0.5 时 128 量子在第 128 槽用完；B/T=0.8 时在第 205 槽用完）
    ⇒ **没有任何"闲着才探"的时隙**。
    ⇒ 结论：在"有预算就发"的调度下，探测通道**在设计上就不可能被触发**。
      这是调度顺序造成的恒等式，不是"效应很小"（与 B4 / 教训 #77 同族）。

    规则（口径一次性写定，不看收益）
    --------------------------------
    每 `period` 槽买一次（`period` 是**帧长**，问题定义；默认 `T/8 = 32`）：
      1) 当前发货对象的 `csi_aoi >= period` 或从未探测，且预算够"一探一发"
         ⇒ `PROBE(该对象)`；
      2) 否则有预算 ⇒ `SEND(argmax p_hat)`；
      3) 否则 `HOLD`。
    `p_hat` 在探测之后离开 0.5，于是"发给谁"与"发不发"都随读数变化。

    阴性对照 `use_phat=False`：`p_hat ≡ 0.5` ⇒ 目标恒为 0 号用户、探测判据只看
    `csi_aoi`（与读数无关）⇒ 动作与真 CSI **完全无关**。两者 trace 必须不同（S5）。

    ★ 本类**不声称更好**。判词只有一句：**CSI 能改变动作**；成本变化原样报（允许变差）。
    """

    def __init__(self, cfg, *, use_phat: bool = True, period: int = 32,
                 n_users: int = 2) -> None:
        self.cfg = cfg
        self.use_phat = bool(use_phat)
        self.n = int(n_users)
        self.period = int(period)
        self.name = ("buyinfo_" if use_phat else "buyinfoblind_") + f"p{self.period}"
        self.n_probe_actions = 0
        self.n_send_actions = 0
        self.trace = False
        self.last = None
        #: 诊断：逐用户 `csi_aoi` 的**累计和**（每槽一次）。用来回答"这条链路平均
        #: 多久没被测过" —— 不能拿末刻 `mean_csi_aoi` 当它（末刻是随机变量，
        #: 而且"末刻刚好测过"会让顺序反过来，本轮已踩：末刻值 176 vs 160 与累计值相反）。
        self._csi_aoi_sum = np.zeros(self.n, dtype=float)
        self._ticks = 0

    @property
    def mean_csi_aoi_integral(self) -> float:
        """`E_t[mean_i csi_aoi(i)]`（未测过计 `-1` 以外：用当前时刻 t）。"""
        return float(np.mean(self._csi_aoi_sum) / self._ticks) if self._ticks else float("nan")

    def p_hat(self, tel: list[dict]) -> np.ndarray:
        if not self.use_phat:
            return np.full(len(tel), 0.5)
        return p_hat_vector(tel, self.cfg)

    def __call__(self, v: SchedulerView) -> Action:
        return self._decide(v)

    def _decide(self, v: SchedulerView) -> Action:
        tel = v.telemetry
        c_d = max(1, int(self.cfg.protocol.c_data))
        c_p = max(1, int(self.cfg.protocol.c_probe))
        # 诊断累加：未测过的链路记"当前时刻"（信息最老）
        self._ticks += 1
        for i, r in enumerate(tel):
            a = int(r["csi_aoi"])
            self._csi_aoi_sum[i] += float(v.t if a < 0 else a)
        p = self.p_hat(tel)
        i_send = int(np.argmax(p))

        csi_aoi = int(tel[i_send]["csi_aoi"])
        stale = csi_aoi < 0 or csi_aoi >= self.period
        # 探完还要留钱发一帧，否则探测就是把预算烧在"知道了也没用"上
        can_probe = v.budget_left >= c_p and v.budget_left - c_p >= c_d

        if stale and can_probe:
            self.n_probe_actions += 1
            act = Action("PROBE", i_send)
        elif v.budget_left >= c_d:
            self.n_send_actions += 1
            act = Action("SEND", i_send)
        else:
            act = Action("HOLD")
        if self.trace:
            self.last = {
                "t": int(v.t), "act": str(act),
                "p_hat": [round(float(x), 6) for x in p],
                "csi_aoi": [int(r["csi_aoi"]) for r in tel],
            }
        return act
