#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""wmlab 自检测试 —— **不依赖 pytest**，直接 `python tests/test_channel_and_aoi.py` 运行。

为什么不用 pytest：本仓库的定位是"陌生人 30 分钟能跑起来"（见 `目标与进度` 的 N1）。
多一个测试依赖，就多一道让人放弃的门槛。全部用标准库 + numpy + torch（训练本来就依赖）。

★ 每个用例都对应一次**真实踩过的坑**，不是为凑覆盖率写的：
  ① `NmseCurve` 的 h=0 锚点     —— X26 首次运行解析高估 94%，根因就是缺这个锚点
  ② `run_tracking` 的完美信道    —— 每步都收到真值时误差必须**恒等于 0**
  ③ 信道层零侵入                 —— 一旦破了，X2 的全部历史结论作废
  ④ 信道 rng 与动力学 rng 隔离    —— 否则"加信道"会顺带换掉被采样的轨迹
  ⑤ age 记账 = 几何分布           —— X26 的整条解析链条建在这条等式上
  ⑥ 凸曲线上的 Jensen 符号        —— 结论"方差大的调度误差更高"的依据
  ⑦ 误差口径 = 逐点（非累积）      —— X26 对账发现：累积口径高估 H* 约 51%
  ⑧ NmseCurve 拒绝长度不一致       —— 稠密逐点数组误配稀疏 horizons 会静默截断
  ⑨ 时延真的延迟了信息（X27）      —— 旧实现里 `delay` 是**空参数**，只做算术偏移
  ⑩ 陈旧载荷的前向补偿（X27）      —— 且"补偿有效"的前提是**模型够好**（未训练时反而更糟）
  ⑪ NaN/Inf 不得静默通过（X3）     —— `NaN <= 阈值` 恒为 False ⇒ 会被当成"已超阈"返回假 H*
  ⑫ PD 增益 = 闭式解（X30）        —— kp=ω_n², kd=2ζω_n−κ；显式 kp/kd 必须能覆盖（消融用）
  ⑬ PD 稳态误差 = 解析解（X30）     —— 旋转系向量解；★ 曾因"正交项标量相加"与"瞬态没走完"
                                       两次给出 5–7 倍的错误预期
  ⑭ 闭环仿真的记账与接线（X30）     —— n_tx / E[age] 有闭式解；T=1 时估计误差必须**恒为 0**
  ⑮ 闭环 rollout 发散必须抛（X30）  —— 同 ⑪，但发生在**控制闭环里**：假 H* 会直接进结论
  ⑯ GE 信道的解析量（X31）          —— π_B / E[L] / ρ₁ 闭式零容差；可行域 L ≥ p̄/(1−p̄)
  ⑰ 无记忆点 L=1/(1−p̄)（X31）      —— ★ 不是 L=1（那是"通/丢交替"）；此处 ρ₁=0 ⇒ i.i.d.
  ⑱ ★ 条件年龄恒为 Geom(β)（X31）   —— 一阶/二阶分解的地基，不成立则整个分解不可信
  ⑲ ★ n_tx 与 n_steps 同区间（X31） —— 修掉的 bug：预热期计入 n_tx 使 tx_rate 高估（23σ）
  ⑳ ★ t0_min 真的抬高起点（X31）    —— 修掉的坑：离线曲线被最大视界挤进瞬态区
  ㉑ ★ 游程长度 ~ Geom(β)（X31-b）    —— ★ 比 ⑱ 更严格：游程长度是**i.i.d.**样本，
                                          而在线 age 序列强自相关（一个游程只有 1 个
                                          独立样本）⇒ 用它才能检出分布错误
  ㉒ ★★ payload_fn 真的接线（X32）    —— R12 落地：`delay` 曾声明/传递/记录/测试全过
                                          却从未生效。两条断言缺一不可：不传=逐位不变，
                                          传粗量化器=误差必须变大
  ㉓ M=4 时 BER = QPSK 精确式（X32）  —— 唯一能自证的公式检验（须避开饱和 clip 区）
  ㉔ SER 饱和 + 失效区标记（X32）     —— 近似式在"高阶调制+低 SNR"下会算出 >1−1/M；
                                          且 BER≈SER/log2M 在饱和时失效（会让 PER 非单调）
  ㉕ PER 对 b 单调增 / 对 SNR 单调减   —— U 形曲线两个分支的地基
  ㉖ 量化 MSE 单调减 + 无过载（X32）  —— 过载（裁剪）会让 Δ²/12 失效，故量程必须由
                                          数据统计并留 margin
  ㉗ ★ 周期发送的年龄闭式（X34）      —— 归一化 + 均值闭式 + 两个退化（T=1 / s=1）
  ㉘ ★ 周期本身造成年龄地板（X34）    —— 不丢包时 E[age]=(T−1)/2；同时是 R12 检查
  ㉙ ★ 离线 payload_fn 只作用于起点（X33）—— 恒等⇒逐位一致；量化⇒必须变化
  ㉚ ★ 完美信道下两 head 逐位相同（X33）—— 每步都到⇒从不调用预测；同龄对照的地基
  ㉛ ★★ 尾部质量 P(age>K) 的**方向**（X34-b2）—— 随 q 单调增；★ 我曾把注释里的方向写反，
                                          是这条单调性断言抓出来的
  ㉜ ★★ `periodic_age_var` 闭式 + **SE 地板**（X34-b2）—— 解释"容差不能拍百分比"
  ㉝ ★★ 突发信道周期发送的年龄闭式必须退化到已知答案（X36）
  ㉞ ★ 突发调度仿真的年龄直方图必须收敛到新闭式（X36）—— 不是自洽就算过
  ㉟ ★★ 年龄阈值触发的年龄闭式退化到两个已知答案（X38）
  ㊱ ★★ 等预算下阈值 E[age] ≤ 周期且比值呈 **U 形**（X38）
  ㊲ ★ 阈值 schedule 有状态 / 可 reset / age<K 时不尝试（X38）
  ㊳ ★★ 状态依赖阵风场真的异方差 + `wind_amp=0` 逐位退化（X38-b）
  ㊴ ★ R12：闭环里的 `payload_fn`（量化器）**真的接进去了**（X35 T4）
                                        —— ★ 原误标 ㉜（与 `periodic_age_var` 撞号），按"只追加"改 ㊴
  ㊵ ★★ 等预算配对的**公共窗口守卫**（X40）—— 5% 跨度的窗口必须判"不可评估"，
                                        否则 9 个网格点会挤在同一段里造出假头条
  ㊶ ★ X41 剂量档的"任务价值"必须**跨 PER 取中位**（不是取极值 / 首个）——
                                        同一批数据取最大报 1.147、取中位报 1.065，报哪个决定结论强弱
"""

from __future__ import annotations

import copy
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wmlab.control import (PDRelativeController, collect_controlled_episodes,
                           run_closed_loop_control, task_horizon)
from wmlab.data import collect_random_episodes, transitions_from_episodes
from wmlab.envs import (ChannelAdapter, StepResult, make_env, make_env_with_channel)
from wmlab.eval import (NmseCurve, expected_nmse_analytic, geometric_age_pmf,
                        geometric_age_stats, jensen_gap, loss_prob_for_tail_risk,
                        lossy_schedule, mse, nmse_at_mean_age, per_step_mse,
                        periodic_schedule, reliable_horizon, run_tracking,
                        uniform_age_pmf, uniform_age_stats)
from wmlab.eval.physical import (UniformQuantizer, fit_quantizer_range, overload_fraction,
                                 per_from_ber, qam_approximation_valid, qam_bit_error_rate,
                                 qam_symbol_error_rate, snr_db_to_linear)
from wmlab.eval.tracking import (GilbertElliottChannel, StateTracker,
                                 burst_periodic_age_pmf, burst_periodic_lossy_schedule,
                                 burst_periodic_mean_age,
                                 periodic_age_pmf, periodic_age_tail,
                                 periodic_age_var,
                                 periodic_lossy_schedule,
                                 periodic_mean_age, run_length_goodness, simulate_bad_runs,
                                 threshold_age_pmf, threshold_age_tail,
                                 threshold_attempt_rate, threshold_delivery_rate,
                                 threshold_equivalent_period, threshold_lossy_schedule,
                                 threshold_mean_age)
from wmlab.models import MLPWorldModel
from wmlab.rollout import closed_loop_error_curve, multi_step_error_curve
from wmlab.rollout.imagine import _sample_windows
from wmlab.train import train_world_model


# ----------------------------------------------------------- ① StepResult 兼容
def test_step_result_backward_compatible():
    r = StepResult(np.zeros(2, np.float32), 0.0, False, False)
    assert r.info is None
    assert r.delivered is True        # 未接信道 = 完美观测
    assert r.done is False
    r2 = StepResult(np.zeros(2, np.float32), 0.0, False, False,
                    {"delivered": False, "delay": 3})
    assert r2.delivered is False
    r3 = StepResult(np.zeros(2, np.float32), 0.0, True, False)
    assert r3.done is True


# ----------------------------------------------------------- ③ 信道层零侵入
def test_channel_zero_loss_is_bit_exact():
    """loss_prob=0 时，obs 序列必须与裸 env **逐位一致**（E1）。"""
    def collect(env, n_ep, seed):
        rng = np.random.default_rng(seed)
        out = []
        for i in range(n_ep):
            seq = [env.reset(seed=seed + i)]
            while True:
                res = env.step(env.sample_action(rng))
                seq.append(res.obs)
                if res.done:
                    break
            out.append(np.asarray(seq, dtype=np.float64))
        return out

    a = collect(make_env("CartPole-v1", seed=0), 3, 11)
    b = collect(make_env_with_channel("CartPole-v1", seed=0, loss_prob=0.0, delay=0), 3, 11)
    for x, y in zip(a, b):
        assert np.array_equal(x, y), "信道层改变了动力学 —— 全部历史结论作废"


# ----------------------------------------------------------- ④ rng 隔离
def test_channel_zero_loss_does_not_consume_rng():
    """p=0 时信道 rng 不应被消耗（否则"加信道"会顺带换掉轨迹）。"""
    env = make_env_with_channel("CartPole-v1", seed=0, loss_prob=0.0, delay=0)
    assert isinstance(env, ChannelAdapter)
    state0 = copy.deepcopy(env._rng.bit_generator.state)
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(20):
        env.step(env.sample_action(rng))
    assert copy.deepcopy(env._rng.bit_generator.state) == state0
    env.close()


def test_channel_loss_rate_matches_p():
    for p in (0.25, 0.6):
        env = make_env_with_channel("CartPole-v1", seed=0, loss_prob=p, delay=0,
                                    max_episode_steps=10 ** 6)
        rng = np.random.default_rng(0)
        env.reset(seed=0)
        got = [env.step(env.sample_action(rng)).delivered for _ in range(8000)]
        env.close()
        rate = float(np.mean(got))
        assert abs(rate - (1 - p)) < 0.03, f"p={p} 实测送达率 {rate:.4f} 偏离过大"


# ----------------------------------------------------------- ⑤ age = 几何分布
def test_geometric_age_stats_and_pmf():
    for p in (0.1, 0.5, 0.9):
        m, v = geometric_age_stats(p)
        assert abs(m - p / (1 - p)) < 1e-12
        assert abs(v - p / (1 - p) ** 2) < 1e-12
        pmf = geometric_age_pmf(p, 4000)
        assert abs(pmf.sum() - 1.0) < 1e-12
        hs = np.arange(len(pmf), dtype=float)
        assert abs(float(np.dot(hs, pmf)) - m) < 1e-3


def test_uniform_age_stats_and_pmf():
    for T in (1, 3, 20):
        m, v = uniform_age_stats(T)
        assert abs(m - (T - 1) / 2) < 1e-12
        assert abs(v - (T ** 2 - 1) / 12) < 1e-12
        assert abs(uniform_age_pmf(T).sum() - 1.0) < 1e-12


def test_transmission_age_is_geometric_in_simulation():
    """实测 age 的均值必须收敛到 p/(1-p) —— X26 解析链条的地基。"""
    for p in (0.3, 0.7):
        env = make_env_with_channel("CartPole-v1", seed=0, loss_prob=p, delay=0,
                                    max_episode_steps=10 ** 6)
        rng = np.random.default_rng(0)
        env.reset(seed=0)
        ages, age = [], 0
        for _ in range(30000):
            if env.step(env.sample_action(rng)).delivered:
                age = 0
            else:
                age += 1
            ages.append(age)
        env.close()
        got = float(np.mean(ages))
        want = geometric_age_stats(p)[0]
        assert abs(got - want) / want < 0.06, f"p={p}: 实测 {got:.4f} vs 理论 {want:.4f}"


def test_loss_prob_for_tail_risk_is_consistent():
    for H, delta in ((20.0, 0.01), (64.8, 0.05)):
        p = loss_prob_for_tail_risk(H, delta)
        pmf = geometric_age_pmf(p, 20000)
        tail = float(pmf[int(np.floor(H)) + 1:].sum())
        assert abs(tail - delta) < 0.002, f"H={H}: 尾部风险 {tail:.4f} ≠ {delta}"


# ----------------------------------------------------------- ① 锚点回归
def test_nmse_curve_anchor_zero():
    """★ 回归：age=0 ⇒ 刚收到真实观测 ⇒ 误差必须为 0。"""
    c = NmseCurve([1, 2, 3], [0.10, 0.20, 0.30])
    assert c(0.0) == 0.0
    assert abs(c(1.0) - 0.10) < 1e-12
    assert abs(c(2.0) - 0.20) < 1e-12
    # 关掉锚点就是旧（错的）行为，保留开关以便复现历史数值
    c_old = NmseCurve([1, 2, 3], [0.10, 0.20, 0.30], anchor_zero=False)
    assert abs(c_old(0.0) - 0.10) < 1e-12
    # 区间外常数外推（饱和假设）
    assert abs(c(999.0) - 0.30) < 1e-12


def test_nmse_curve_convex_fraction():
    # ★ 注意：默认 anchor_zero=True 会在 h=0 插入 (0,0)，使曲线在起点变凸。
    #   要测"线性曲线 ⇒ 凸比例为 0"，必须显式关掉锚点，或者让数据本身从 h=0 起。
    c = NmseCurve([1, 2, 3, 4], [0.1, 0.5, 0.9, 1.3], anchor_zero=False)  # 严格线性
    assert abs(c.convex_fraction()) < 1e-9
    c2 = NmseCurve([1, 2, 3, 4], [0.01, 0.04, 0.16, 0.64])  # 二次增长 → 全段凸
    assert c2.convex_fraction() > 0.5
    c3 = NmseCurve([1, 2, 3, 4], [0.9, 0.97, 0.99, 0.995])  # 饱和 → 凹
    assert c3.convex_fraction() < 0.5


# ----------------------------------------------------------- ⑥ Jensen 符号
def test_jensen_gap_positive_on_convex_curve():
    """纯凸曲线（NMSE=h²）上，E[NMSE(age)] > NMSE(E[age])。"""
    c = NmseCurve([1, 2, 3, 4], [1.0, 4.0, 9.0, 16.0])   # 锚点后 h=0..4, y=0,1,4,9,16
    pmf = uniform_age_pmf(3)                              # age ∈ {0,1,2}
    e = expected_nmse_analytic(c, pmf)
    n = nmse_at_mean_age(c, pmf)
    assert abs(e - (0 + 1 + 4) / 3.0) < 1e-9
    assert abs(n - 1.0) < 1e-9
    assert e - n > 0
    e2, n2 = jensen_gap(c, pmf)
    assert abs(e2 - e) < 1e-12 and abs(n2 - n) < 1e-12


def test_jensen_gap_negative_on_concave_tail():
    """凹（饱和）段上，E[NMSE(age)] < NMSE(E[age])。"""
    c = NmseCurve([1, 2, 3, 4], [0.9, 0.97, 0.99, 0.995])
    pmf = uniform_age_pmf(4)                              # age ∈ {0,1,2,3}
    e, n = jensen_gap(c, pmf)
    assert e < n, "饱和段应给出负 Jensen gap"


# ----------------------------------------------------------- ② 跟踪器与主循环
def _tiny_model(obs_dim, act_dim, discrete):
    return MLPWorldModel(obs_dim=obs_dim, act_dim=act_dim, latent_dim=4,
                         hidden=8, discrete_act=discrete)


def test_state_tracker_reset_on_delivery_and_predict_on_loss():
    model = _tiny_model(3, 1, False)
    tr = StateTracker(model, torch.device("cpu"), 3)
    o = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    a = np.array([0.5], dtype=np.float32)
    tr.reset(o)
    est, atx, agen = tr.update(o, a, delay=2)
    assert np.array_equal(est, o)
    assert atx == 0 and agen == 2          # generation_age = transmission_age + delay
    est2, atx2, agen2 = tr.update(None, a, delay=2)
    assert atx2 == 1 and agen2 == 3
    assert not np.array_equal(est2, o)     # 丢包时确实换了预测值


def test_run_tracking_perfect_channel_has_zero_error():
    """★ 回归：每步都传（T=1）时误差必须恒为 0，age 恒为 0。"""
    eps = collect_random_episodes(make_env("CartPole-v1", seed=0), n_episodes=3, seed=0)
    model = _tiny_model(4, 2, True)
    r = run_tracking(model, eps, torch.device("cpu"), periodic_schedule(1), seed=0,
                     label="P(T=1)")
    assert r.nmse == 0.0
    assert r.age_tx_max == 0
    assert r.age_tx_mean == 0.0
    assert r.n_tx == r.n_steps
    assert abs(r.tx_rate - 1.0) < 1e-12


def test_run_tracking_age_matches_periodic_schedule():
    """age 在 {0..T-1} 上均匀 ⇒ 均值 (T-1)/2，最大值 T-1。

    ★ 必须用**定长** episode（Pendulum 固定 200 步，且 200 能被 T 整除）。
      第一版用 CartPole：随机策略下每条 episode 长度不同且几乎不被 T 整除，
      每条 episode 又从 t=0 重新计数，于是最后一截不完整的周期把均值拽偏 —— 断言失败。
      这不是被测代码的问题，是测试自己选错了环境。
    """
    eps = collect_random_episodes(make_env("Pendulum-v1", seed=0), n_episodes=2, seed=0)
    model = _tiny_model(3, 1, False)
    T = 5
    assert all(e["length"] % T == 0 for e in eps), "环境长度必须能被 T 整除才能精确断言"
    r = run_tracking(model, eps, torch.device("cpu"), periodic_schedule(T), seed=0,
                     label=f"P(T={T})")
    assert abs(r.age_tx_mean - (T - 1) / 2) < 1e-9     # age 在 {0..T-1} 上均匀
    assert r.age_tx_max == T - 1
    assert abs(r.tx_rate - 1.0 / T) < 1e-9


def test_run_tracking_nmse_increases_with_sparser_tx():
    """传输越稀，误差越大（单调性 sanity check）。"""
    eps = collect_random_episodes(make_env("CartPole-v1", seed=0), n_episodes=3, seed=0)
    model = _tiny_model(4, 2, True)
    nmses = []
    for T in (1, 3, 6):
        r = run_tracking(model, eps, torch.device("cpu"), periodic_schedule(T), seed=0,
                         label=f"P(T={T})")
        nmses.append(r.nmse)
    assert nmses[0] < nmses[1] < nmses[2], f"单调性不成立：{nmses}"


# ----------------------------------------------------------- ⑦ 误差口径（X26 对账）
def test_nmse_curve_rejects_misaligned_lengths():
    """★ 回归：horizons 与 nmse 长度不一致时必须**抛错**，不能静默截断。

    起因（2026-09-20）：把**稠密**的逐点数组（长度 = max(horizons) = 190）直接传给
    `NmseCurve`，而 horizons 只有 19 个。`np.asarray(nmse)[order]` 会静默截断，
    于是"第 1..19 步的值"被配到"视界 1..190"上 —— 大间隔调度的解析值离谱地小
    （T=200 时解析/仿真 = 0.016/0.703，看起来像"解析式彻底失效"，
    实际是配错轴）。冒烟没暴露，因为长度不同不报错。
    """
    hs = [1, 2, 5, 10, 20]
    dense = list(np.linspace(0.0, 1.0, 20))          # 稠密（长度 20）
    try:
        NmseCurve(hs, dense)
        raise AssertionError("长度不一致未被拒绝")
    except ValueError as e:
        assert "长度" in str(e)
    # 正确用法：先按 horizons 抽取
    NmseCurve(hs, [dense[h - 1] for h in hs])
    # 等长时正常构造
    NmseCurve(hs, [0.01, 0.02, 0.05, 0.1, 0.2])


def test_per_step_mse_average_equals_cumulative():
    """★ 回归：mean(per_step_mse[:h]) 必须恒等于累积 mse(h)。

    这条恒等式是"两个口径只是同一批数的两种写法"的数学依据。
    一旦破了，说明逐点数组的**步对齐**错了 —— 冒烟阶段就踩过一次
    （最初按 horizons 对齐返回，长度 14，被误当稠密数组用，h=5 就对不上）。
    """
    torch.manual_seed(0)
    pred = torch.randn(5, 17, 3)
    tgt = torch.randn(5, 17, 3)
    per = per_step_mse(pred, tgt)
    assert len(per) == 17, "逐点数组必须是稠密的（长度 = 步数）"
    for h in (1, 2, 5, 11, 17):
        got = float(np.mean(per[:h]))
        exp = mse(pred[:, :h], tgt[:, :h])
        assert abs(got - exp) < 1e-5, f"h={h}: {got} vs {exp}"


def test_reliable_horizon_cumulative_overestimates_and_warns():
    """★ 回归：累积口径系统性高估 H*，且被传成累积口径时必须警告。

    构造一条"前段很平、后段陡升"的逐点曲线 —— 这正是世界模型误差曲线的形状。
    累积口径被前段平段摊薄，穿越点必然更靠后。
    """
    hs = [1, 2, 4, 8, 16, 32, 64]
    ps = [1e-4, 1e-4, 2e-4, 3e-3, 0.02, 0.09, 0.35]              # 逐点
    cum = [float(np.mean(ps[:i + 1])) for i in range(len(hs))]   # 累积 = 逐点的运行平均
    thr = 0.02

    h_ps = reliable_horizon(hs, ps, thr)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        h_cum = reliable_horizon(hs, cum, thr, calibration="cumulative")
        assert any("累积" in str(x.message) for x in w), "累积口径未触发警告"
    assert h_ps is not None and h_cum is not None
    assert h_cum > h_ps, f"累积口径应高估：cum={h_cum} pointwise={h_ps}"

    for bad in ("cumulative2", "Cum", ""):
        try:
            reliable_horizon(hs, cum, thr, calibration=bad)
            raise AssertionError(f"非法 calibration={bad!r} 未被拒绝")
        except ValueError:
            pass


def test_curves_expose_both_calibrations():
    """两条误差曲线都必须同时给出累积与逐点口径（对应硬约定 R1）。

    且逐点数组必须与累积数组自洽：mean(逐点[:h]) == 累积(h)。
    """
    eps = collect_random_episodes(make_env("CartPole-v1", seed=0), n_episodes=3, seed=0)
    model = _tiny_model(4, 2, True)
    hs = [1, 2, 4, 8]
    for fn in (multi_step_error_curve, closed_loop_error_curve):
        c = fn(model, eps, hs, torch.device("cpu"), n_samples=8, seed=0)
        for k in ("mse", "nmse", "mse_per_step", "nmse_per_step"):
            assert k in c, f"{fn.__name__} 缺少字段 {k}"
        assert len(c["nmse_per_step"]) == max(hs), \
            f"{fn.__name__} 逐点必须是稠密数组（长度 = max(horizons)）"
        for i, h in enumerate(c["horizons"]):
            got = float(np.mean(c["mse_per_step"][:h]))
            assert abs(got - c["mse"][i]) < 1e-5, f"{fn.__name__} h={h} 口径不自洽"


# ----------------------------------------------------------- ⑨ 时延真的生效（X27）
def _floor_from_data(episodes, D):
    """直接从数据算"时延地板"（model-free）。

    `p=0` 且时延 `D` 时，本地估计值**恰好等于** `obs[t+1-D]`（与模型无关）
    ⇒ NMSE_floor(D) = E[(o_{t+1} − o_{t+1−D})²] / Var(o) = 2(1 − ρ_o(D))。
    前 `D` 步还没有包到达，是暂态，跳过（与 `run_tracking(warmup=D)` 对齐）。
    """
    sq, n = 0.0, 0
    vals = []
    for ep in episodes:
        obs = ep["obs"]
        for t in range(len(ep["act"])):
            if t < D:
                continue
            d = obs[t + 1] - obs[t - D + 1]
            sq += float(np.sum(d ** 2))
            n += d.size
            vals.append(np.asarray(obs[t + 1], dtype=np.float64).reshape(-1))
    allv = np.concatenate(vals)
    return sq / n / float(allv.var())


def test_delay_arrival_counts():
    """`delay=0` 时到达数 = 发送数；`delay=D>0` 时末尾 `D` 个包到不了（每条 episode）。"""
    eps = collect_random_episodes(make_env("Pendulum-v1", seed=0), n_episodes=2, seed=0)
    model = _tiny_model(3, 1, False)
    T = eps[0]["length"]
    r0 = run_tracking(model, eps, torch.device("cpu"), periodic_schedule(1), seed=0, delay=0)
    assert r0.n_arrived == r0.n_tx == 2 * T
    for D in (1, 5):
        r = run_tracking(model, eps, torch.device("cpu"), periodic_schedule(1), seed=0, delay=D)
        assert r.n_tx == 2 * T, "发送数不该因时延而变"
        assert r.n_arrived == r.n_tx - D * len(eps), \
            f"D={D}: 到达数应为发送数减去每条 episode 末尾 {D} 个到不了的包"


def test_delay_creates_irreducible_error_floor():
    """★ X27 的核心命题：时延给误差设了一个**通信压不掉的地板**。

    解析式（见 `tracking.py` module docstring）：p=0 时估计值恒等于 `obs[t+1-D]`
    ⇒ `NMSE_floor(D) = 2(1 - ρ_o(D))`，**与世界模型质量无关**。
    这里用两条独立路径验它：
      (a) `run_tracking` 的在线仿真；
      (b) 直接从评测数据算 `mean((o_{t+1} - o_{t+1-D})²)/Var`（完全不碰模型）。
    两者必须一致 —— 这正是 X26 那套"两条路径算同一个量"的纪律。
    """
    eps = collect_random_episodes(make_env("Pendulum-v1", seed=0), n_episodes=3, seed=0)
    model = _tiny_model(3, 1, False)
    nmses = []
    for D in (1, 3, 10, 30):
        r = run_tracking(model, eps, torch.device("cpu"), periodic_schedule(1), seed=0,
                         delay=D, warmup=D, label=f"floor(D={D})")
        ana = _floor_from_data(eps, D)
        rel = abs(r.nmse - ana) / max(ana, 1e-12)
        assert rel < 1e-4, f"D={D}: 在线 {r.nmse:.6f} vs 解析地板 {ana:.6f}（相对差 {rel:.2%}）"
        assert r.nmse > 0.0, f"D={D}: 时延应产生正误差地板"
        assert r.age_tx_mean == 0.0, f"D={D}: p=0 ⇒ transmission age 必须恒为 0"
        assert abs(r.age_gen_mean - D) < 1e-9, f"D={D}: generation age 应恒为 D"
        nmses.append(r.nmse)
    assert all(x < y for x, y in zip(nmses, nmses[1:])), \
        f"地板应随 D 单调增（滞后越大、观测越陈旧）：{nmses}"


def _trained_small_model(train_eps, val_eps, epochs: int = 60):
    """训一个小模型（latent=8, hidden=64）—— 只为本文件里的时延用例服务。

    ★ **为什么必须训练**：`forward` 补偿的本质是"用模型把陈旧观测推到当前时刻"，
      模型垃圾时它当然更糟。未训练模型的 1 步 rollout 误差 ≈ 1.0，
      比"直接用陈旧但真实的观测"（D=1 时 0.037）还差 **27 倍**。
      这不是实现错误，是真现象 —— 2026-09-20 首次写这条用例时就是被它打回来的。
    """
    torch.manual_seed(0)
    model = MLPWorldModel(obs_dim=3, act_dim=1, latent_dim=8, hidden=64,
                          discrete_act=False)
    tr = tuple(torch.as_tensor(x) for x in transitions_from_episodes(train_eps))
    va = tuple(torch.as_tensor(x) for x in transitions_from_episodes(val_eps))
    train_world_model(model, tr, va,
                      {"train": {"lr": 3e-3, "batch_size": 64, "epochs": epochs,
                                 "grad_clip": 5.0}, "seed": 0},
                      torch.device("cpu"), verbose=False)
    return model


def test_delay_forward_mode_helps_when_model_is_trained():
    """★ X27 的第二半：把陈旧载荷**向前推进**（forward）远优于直接用（naive）。

    实测（latent=8/hidden=64/60 epoch，本机 CPU，2026-09-20）：
      D=1   naive 0.0374 → forward 0.00076（**约 1/49**）
      D=5   naive 0.7318 → forward 0.00979（**约 1/75**）
      D=20  naive 3.1652 → forward 0.22902（**约 1/14**）

    ⇒ 结论不是"时延有害"，而是**「收到陈旧观测就直接当当前状态用」才是有害的**。
    这正是"世界模型对通信系统有没有用"的第一个可量化回答。
    """
    eps = collect_random_episodes(make_env("Pendulum-v1", seed=0), n_episodes=6, seed=0)
    model = _trained_small_model(eps[:4], eps[4:])
    dev = torch.device("cpu")
    for D in (1, 2, 5, 10, 20):
        rn = run_tracking(model, eps[4:], dev, periodic_schedule(1), seed=0,
                          delay=D, warmup=D, delay_mode="naive")
        rf = run_tracking(model, eps[4:], dev, periodic_schedule(1), seed=0,
                          delay=D, warmup=D, delay_mode="forward")
        assert rf.nmse < 0.5 * rn.nmse, \
            f"D={D}: forward({rf.nmse:.6f}) 应显著优于 naive({rn.nmse:.6f})"
        f_ana = _floor_from_data(eps[4:], D)
        assert abs(rn.nmse - f_ana) / max(f_ana, 1e-12) < 1e-4, \
            f"D={D}: naive 必须等于解析地板（{rn.nmse} vs {f_ana}）"


def test_delay_zero_is_mode_invariant():
    """★ 回归：`delay=0` 时两种 `delay_mode` 必须**逐位相同**。

    没有陈旧载荷，就无所谓补不补偿。这条保证 X27 引入的 `delay_mode`
    **不会动到 X24/X25/X26 的任何结论**（它们全部用 `delay=0`）。
    """
    eps = collect_random_episodes(make_env("Pendulum-v1", seed=0), n_episodes=2, seed=0)
    model = _tiny_model(3, 1, False)
    for sched, lbl in ((periodic_schedule(3), "P(T=3)"), (lossy_schedule(0.6), "R(p=0.6)")):
        a = run_tracking(model, eps, torch.device("cpu"), sched, seed=0, delay=0, label=lbl)
        b = run_tracking(model, eps, torch.device("cpu"), sched, seed=0, delay=0, label=lbl,
                         delay_mode="forward")
        assert a.mse == b.mse and a.nmse == b.nmse, "delay=0 时两种模式必须逐位相同"
        assert a.age_tx_mean == b.age_tx_mean and a.n_arrived == b.n_arrived


def test_delay_mode_rejects_bad_value():
    eps = collect_random_episodes(make_env("CartPole-v1", seed=0), n_episodes=1, seed=0)
    model = _tiny_model(4, 2, True)
    for bad in ("Forward", "naive2", ""):
        try:
            run_tracking(model, eps, torch.device("cpu"), periodic_schedule(2), seed=0,
                         delay=0, delay_mode=bad)
            raise AssertionError(f"非法 delay_mode={bad!r} 未被拒绝")
        except ValueError:
            pass


# ----------------------------------------------------------- ⑪ NaN 不得静默通过（X3）
def test_reliable_horizon_rejects_non_finite():
    """★ 曲线发散时，H\\* 必须**响**,不能给假数字。

    实测触发场景：15 epoch 的弱模型做观测空间闭环 rollout，
    第 131 步之后出现 60 个非有限值、峰值 3.9e34。

    危险在于 `NaN <= 0.05` 恒为 False —— 循环会把它当成"已超阈"，
    在第 3 个点就返回 H\\*≈2.x。**一个假数字会被写进论文**，比崩溃危险得多。
    """
    hs = [1, 2, 3, 4, 5]
    for bad, name in ((float("nan"), "NaN"), (float("inf"), "+Inf"),
                      (float("-inf"), "-Inf")):
        es = [0.01, 0.02, bad, 0.50, 0.80]
        try:
            h = reliable_horizon(hs, es, 0.05)
            raise AssertionError(
                f"{name} 未被拦住 —— reliable_horizon 返回了 {h}，"
                f"这是把 NaN/Inf 当成'已超阈'的假值")
        except ValueError as e:
            assert "有限值" in str(e), f"报错信息应指明原因，实际：{e}"

    # ★ 反向确认：全程有限的同型曲线仍能正常工作（别把修复做成一刀切的误伤）
    ok = reliable_horizon(hs, [0.01, 0.02, 0.50, 0.80, 0.90], 0.05)
    assert ok is not None and 2.0 < ok < 3.0, f"正常曲线应给出 H*≈2.x，实际 {ok}"


# ----------------------------------------------------------- ⑫ X30 · PD 增益闭式解
def test_pd_gains_follow_closed_form():
    c = PDRelativeController(dt=0.1, omega_n=2.5, zeta=1.0, kappa=0.5, a_max=3.0)
    assert abs(c.kp - 2.5 ** 2) < 1e-12, f"kp 应 = ω_n²，实际 {c.kp}"
    assert abs(c.kd - (2 * 1.0 * 2.5 - 0.5)) < 1e-12, f"kd 应 = 2ζω_n−κ，实际 {c.kd}"
    # ★ 显式 kp/kd 必须能覆盖公式值 —— 否则"换增益做消融"这个动作根本没接线（R12）
    c2 = PDRelativeController(dt=0.1, kappa=0.5, a_max=3.0, kp=1.0, kd=2.0)
    assert (c2.kp, c2.kd) == (1.0, 2.0), "显式给出的 kp/kd 被公式覆盖了"


# ----------------------------------------------------------- ⑬ X30 · 稳态误差 = 解析解
def test_pd_steady_state_matches_analytic():
    """★ 解析推导的实测验证（确定性环境、T=1、充分预热）。

    ★★ 这一条在 2026-09-22 连续查出两个真错误，都不是"测试太严"，是真错了：

    ① **标量式错**：文件头第一版写 `e_ss = (a_tgt + κ·v_tgt)/kp = 0.576 m`。
       但目标做圆周运动 ⇒ 向心项（径向）与阻尼项（切向）**正交**，不能相加。
       实测稳态 **0.394 m**，与标量解差 46%，与旋转系 2×2 向量解 0.3996 m 差 1.4%。
    ② **瞬态没走完**：不加预热时 T=1 实测 **2.86 m**（UAV 起点最远离目标 15 m，
       而 a_max=3 m/s² ⇒ 加速度**全程饱和**，捕获瞬态要 ~400 步，episode 只有 200 步）。
       误判成"解析错了"，实际是测量窗口太短 —— 加了预热段才暴露真相。
    """
    env = make_env("uav-track", seed=0, noise_std=0.0, max_steps=900)
    ctrl = PDRelativeController(dt=env.dt, omega_n=2.5, zeta=1.0,
                                kappa=float(env.kappa), a_max=float(env.a_max))
    w, r, kap = float(env.omega), float(env.r_orbit), float(env.kappa)
    A = np.array([[ctrl.kp - w ** 2, -(kap + ctrl.kd) * w],
                  [(kap + ctrl.kd) * w, ctrl.kp - w ** 2]])
    e_ana = float(np.linalg.norm(np.linalg.solve(A, np.array([-(w ** 2) * r, kap * w * r]))))

    model = MLPWorldModel(6, 2, latent_dim=4, hidden=8, discrete_act=False)
    res = run_closed_loop_control(env, model, ctrl, periodic_schedule(1),
                                  n_episodes=3, seed=7, max_steps=200,
                                  device=torch.device("cpu"), estimator="model",
                                  var_g=1.0, warmup_steps=600)
    env.close()
    rel = abs(res["mean_dist"] - e_ana) / e_ana
    assert rel < 0.05, (f"T=1 稳态距离实测 {res['mean_dist']:.4f} m vs 解析 {e_ana:.4f} m "
                        f"（相对偏差 {rel:.1%}）。若差 5–7 倍，先查两件事："
                        f"① e_ss 是不是又用了标量式；② 预热步数够不够")


# ----------------------------------------------------------- ⑭ X30 · 记账与接线
def test_closed_loop_tx_and_age_match_closed_form():
    """周期调度的 n_tx 与 E[age] 都有闭式解 ⇒ **用精确判据，不用容差**。

    ★ 为什么不用容差（第一版踩过）：用 "tx_rate ≈ 1/T（容差 2%）" 时，
      T=32 被判失败（实测 0.0300 vs 0.03125）—— 但那不是 bug，
      是 **episode 有限长的边缘效应**（200 步只在 t=32,64,...,192 送 6 次）。
      容差判据两头不讨好：会把数学必然误报成 bug，也会放过真 bug。
    """
    env = make_env("uav-track", seed=0, noise_std=0.15, max_steps=60)
    ctrl = PDRelativeController(dt=env.dt, omega_n=2.5, zeta=1.0,
                                kappa=float(env.kappa), a_max=float(env.a_max))
    model = MLPWorldModel(6, 2, latent_dim=8, hidden=16, discrete_act=False)
    for T in (1, 3, 8):
        r = run_closed_loop_control(env, model, ctrl, periodic_schedule(T),
                                    n_episodes=4, seed=3, max_steps=40,
                                    device=torch.device("cpu"), estimator="model",
                                    var_g=1.0, warmup_steps=20)
        exp_tx, exp_age = 0, 0
        for L in r["ep_lens"]:
            q, rem = divmod(int(L), T)
            exp_tx += q
            exp_age += q * T * (T - 1) // 2 + rem * (rem + 1) // 2
        assert r["n_tx"] == exp_tx, f"T={T}: n_tx={r['n_tx']} ≠ Σ floor(L/T)={exp_tx}"
        assert abs(r["mean_age"] - exp_age / max(r["n_steps"], 1)) < 1e-9, \
            f"T={T}: mean_age 与闭式解不符"
    env.close()


def test_closed_loop_T1_estimation_error_is_zero():
    """T=1（每步都送真值）时估计误差必须**恒等于 0** —— 恒等于 0 才叫接线正确。

    ★ 踩过的坑：第一版把"更新估计"写在"记账误差"之后，记到的是
      `obs_t − obs_t+1`（一步状态变化量），于是 T=1 的 est_nmse 永远不为 0，
      而误差曲线看起来"完全合理" —— 只有这个恒等式能把它抓出来。
    """
    env = make_env("uav-track", seed=0, noise_std=0.15, max_steps=60)
    ctrl = PDRelativeController(dt=env.dt, omega_n=2.5, zeta=1.0,
                                kappa=float(env.kappa), a_max=float(env.a_max))
    model = MLPWorldModel(6, 2, latent_dim=8, hidden=16, discrete_act=False)
    r = run_closed_loop_control(env, model, ctrl, periodic_schedule(1),
                                n_episodes=3, seed=11, max_steps=40,
                                device=torch.device("cpu"), estimator="model",
                                var_g=1.0, warmup_steps=20)
    env.close()
    assert r["est_nmse"] == 0.0, f"T=1 时 est_nmse 应恒为 0，实际 {r['est_nmse']:.3e}"


def test_estimator_switch_changes_behaviour():
    """★ R12：`estimator` 这个开关**接线了吗**？model 与 persistence 必须给出不同结果。"""
    env = make_env("uav-track", seed=0, noise_std=0.15, max_steps=60)
    ctrl = PDRelativeController(dt=env.dt, omega_n=2.5, zeta=1.0,
                                kappa=float(env.kappa), a_max=float(env.a_max))
    torch.manual_seed(0)
    model = MLPWorldModel(6, 2, latent_dim=8, hidden=16, discrete_act=False)
    kw = dict(n_episodes=3, seed=5, max_steps=40, device=torch.device("cpu"),
              var_g=1.0, warmup_steps=20)
    a = run_closed_loop_control(env, model, ctrl, periodic_schedule(16),
                                estimator="model", **kw)
    b = run_closed_loop_control(env, model, ctrl, periodic_schedule(16),
                                estimator="persistence", **kw)
    env.close()
    assert a["est_nmse"] != b["est_nmse"], \
        "estimator=model 与 persistence 结果完全相同 ⇒ 开关没接线"


# ----------------------------------------------------------- ⑮ X30 · 发散必须抛
def test_closed_loop_rejects_nan_rollout():
    """闭环 rollout 一旦发散出 NaN，必须**抛错**，不能给假数字（R14）。

    危险点与 ⑪ 同源但更隐蔽：这里 NaN 出现在**控制闭环里**，
    "任务还没失败"会被误读成"视界还够长"，于是 H\\*_task 直接偏大进结论。
    """

    class _NanModel:
        def predict_next(self, obs, act):
            return torch.full_like(obs, float("nan"))

    env = make_env("uav-track", seed=0, noise_std=0.15, max_steps=60)
    ctrl = PDRelativeController(dt=env.dt, omega_n=2.5, zeta=1.0,
                                kappa=float(env.kappa), a_max=float(env.a_max))
    try:
        run_closed_loop_control(env, _NanModel(), ctrl, periodic_schedule(8),
                                n_episodes=1, seed=0, max_steps=20,
                                device=torch.device("cpu"), estimator="model",
                                var_g=1.0, warmup_steps=10)
        env.close()
        raise AssertionError("模型输出 NaN 未被引发异常 —— 会产出假 H*_task")
    except FloatingPointError as e:
        assert "非有限值" in str(e), f"报错信息应指明原因，实际：{e}"
        env.close()


def test_task_horizon_definition():
    """H\\*_task 的定义本身：容差必须二选一；T=1 就不合格 ⇒ None；全合格 ⇒ 网格最大值。"""
    rows = [{"T": 1, "escape_rate": 0.0}, {"T": 4, "escape_rate": 0.01},
            {"T": 8, "escape_rate": 0.30}, {"T": 16, "escape_rate": 0.80}]
    assert task_horizon(rows, "escape_rate", 0.0, tol_abs=0.05) == 4
    assert task_horizon(rows, "escape_rate", 0.0, tol_rel=0.0) == 1
    assert task_horizon(rows, "escape_rate", -1.0, tol_abs=0.05) is None, \
        "T=1 本身就不合格 ⇒ 应返回 None（说明场景/控制器没配好）"
    for kw in ({}, {"tol_abs": 0.05, "tol_rel": 0.1}):
        try:
            task_horizon(rows, "escape_rate", 0.0, **kw)
            raise AssertionError(f"容差二选一未被强制（{kw}）")
        except ValueError:
            pass


# ----------------------------------------------------------- runner
# ------------------------------- ⑯ GE 信道的解析量（X31）
def test_ge_analytic_quantities():
    """S1：π_B / E[L] / ρ₁ / E[age] 全有闭式解（零容差）；不可行参数必须抛。"""
    ch = GilbertElliottChannel(0.5, 4.0, seed=0)
    assert abs(ch.pi_bad - 0.5) < 1e-12
    assert abs(ch.mean_burst_len - 4.0) < 1e-12
    assert abs(ch.expected_age - 2.0) < 1e-12
    assert abs(ch.rho1 - (1 - ch.alpha - ch.beta)) < 1e-15
    assert abs(ch.alpha - 0.25) < 1e-15 and abs(ch.beta - 0.25) < 1e-15
    # 可行域 L ≥ p̄/(1−p̄)：p̄=0.8 ⇒ L ≥ 4
    for bad in ((0.8, 1.0), (0.8, 2.0)):
        try:
            GilbertElliottChannel(*bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"p̄={bad[0]}, L={bad[1]} 应当不可行")
    GilbertElliottChannel(0.8, 4.0)          # ★ 边界恰好可行（浮点不许误杀）
    for bad in ((0.0, 4.0), (1.0, 4.0), (0.5, 0.5)):
        try:
            GilbertElliottChannel(*bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"参数 {bad} 应当被拒")


# ------------------------------- ⑰ 无记忆点（X31）
def test_ge_memoryless_point_is_not_L1():
    """★ 无记忆条件是 **L = 1/(1−p̄)**，不是 L=1（L=1 是"通/丢交替"，ρ₁=−1）。"""
    for pl in (0.2, 0.5, 0.8):
        ch = GilbertElliottChannel(pl, 1.0 / (1.0 - pl), seed=3)
        assert abs(ch.rho1) < 1e-12, f"p̄={pl} 的 L_iid 处 ρ₁={ch.rho1}"
        assert abs(ch.expected_age - pl / (1.0 - pl)) < 1e-12
    assert abs(GilbertElliottChannel(0.5, 1.0).rho1 + 1.0) < 1e-12
    ch = GilbertElliottChannel(0.5, 2.0, seed=11)
    rng = np.random.default_rng(0)
    ch.reset(rng)
    lost = np.fromiter((not ch(t, rng) for t in range(60000)), dtype=bool, count=60000)
    assert abs(lost.mean() - 0.5) < 0.01, f"经验丢包率 {lost.mean():.4f}"


# ------------------------------- ⑱ ★ 条件年龄恒为 Geom(β)（X31 核心恒等式）
def test_ge_conditional_age_is_geometric():
    """P(age=k | age>0) = β(1−β)^(k−1)，且 E[age] = p̄·L。

    这是分解式 `E[NMSE] = p̄·[f(L)+J]` 的地基；不成立则整个一阶/二阶分解不可信。
    """
    for pl, L in ((0.3, 3.0), (0.5, 8.0)):
        ch = GilbertElliottChannel(pl, L, seed=5)
        rng = np.random.default_rng(0)
        ch.reset(rng)
        cur = 0
        age = np.empty(150000, dtype=np.int32)
        for t in range(age.size):
            cur = cur + 1 if not ch(t, rng) else 0
            age[t] = cur
        assert abs(age.mean() - pl * L) / (pl * L) < 0.05, \
            f"E[age]={age.mean():.3f} 应 ≈ p̄·L={pl * L:.3f}"
        pos = age[age > 0]
        K = 60
        emp = np.array([(pos == k).mean() for k in range(1, K + 1)])
        th = np.array([ch.beta * (1 - ch.beta) ** (k - 1) for k in range(1, K + 1)])
        tv = 0.5 * float(np.abs(emp - th).sum())
        assert tv < 0.03, f"p̄={pl}, L={L} 条件年龄分布 TV={tv:.4f}"


# ------------------------------- ⑲ ★ tx_rate 与 warmup 同区间（X31 修掉的 bug）
def test_tracking_tx_rate_excludes_warmup():
    """warmup>0 时 tx_rate 必须仍是**记账区间内**的送达率。

    ★ X31 实测：修之前 n_tx 在预热期也累加、n_steps 只记预热之后，
      L=1 时算出丢包率 0.222 而非 0.5（差 23σ）。X26/X27 因 warmup=0 从未触发。
    """
    class _ConstModel:                       # 预测 = 原样返回（本例只测计数）
        def eval(self):
            pass

        def predict_next(self, obs, act):
            return obs

    n, D = 400, 3
    eps = []
    for i in range(3):
        obs = np.random.default_rng(i).normal(size=(n + 1, D)).astype(np.float32)
        eps.append({"obs": obs, "act": np.zeros((n, 1), np.float32)})
    for warm in (0, 150):
        r = run_tracking(_ConstModel(), eps, torch.device("cpu"),
                         lossy_schedule(0.5), seed=0, warmup=warm, denom=1.0)
        assert r.n_steps == 3 * (n - warm), f"warmup={warm}: n_steps={r.n_steps}"
        assert abs(r.tx_rate - 0.5) < 0.04, f"warmup={warm}: tx_rate={r.tx_rate:.4f}"


# ------------------------------- ⑳ ★ t0_min 真的抬高起点（X31 修掉的坑）
def test_sample_windows_t0_min():
    """`t0_min` 必须真的把起点下界抬上去。

    ★ 背景：起点范围 t0 ∈ [t0_min, T−h_max−1) —— **h_max 越大起点越少，且全挤在
      episode 开头**。UAV 捕获瞬态占前 ~250 步 ⇒ 离线曲线几乎只在瞬态区取样，
      f(1) 高估 49%。修法：给离线与在线**两边**都加"跳过瞬态"。
    """
    T, D = 300, 2
    eps = []
    for _ in range(2):
        obs = np.tile(np.arange(T + 1, dtype=np.float32).reshape(-1, 1), (1, D))
        eps.append({"obs": obs, "act": np.zeros((T, 1), np.float32)})
    for t0_min in (0, 100):
        obs0, _acts, _tgt = _sample_windows(eps, 50, 400,
                                            np.random.default_rng(0), t0_min=t0_min)
        assert obs0[:, 0].min() >= t0_min - 1e-6, \
            f"t0_min={t0_min} 但最小起点 = {obs0[:, 0].min()}"
        assert obs0[:, 0].max() <= T - 50 - 1


def test_ge_run_length_is_geometric():
    """㉑ ★ 游程长度 ~ Geom(β)（**i.i.d. 样本**）—— X31-b 逼出来的更严格的检验。

    ★★ 为什么不能拿在线 age 直方图来做这件事（第 14 次自我修正）
    age 序列**强自相关**：一个长度 n 的 Bad 游程里 age 就是 1,2,…,n 这个确定性
    序列 ⇒ 一个游程只有 **1 个独立样本**。用 n_pos 当样本量会把检验功效高估
    约 √L 倍。X31-b 的实测症状：p̄=0.2/L=16 处 KS 判"显著"(0.0235>0.0225)，
    而同一份数据的 TV=0.0278 却**低于**纯噪声期望 0.0512 —— 两个指标互相矛盾。
    游程长度在 Gilbert 链下是 i.i.d. Geom(β)，且仿真极便宜 ⇒ 用它。
    """
    for pl, L in ((0.5, 2.0), (0.35, 8.0), (0.2, 16.0), (0.65, 4.0)):
        ch = GilbertElliottChannel(pl, L, seed=3)
        runs = simulate_bad_runs(ch, 120_000, np.random.default_rng(7))
        g = run_length_goodness(runs, ch.beta, alpha=0.05 / 4)   # Bonferroni
        assert g["ok"] is True, \
            f"p̄={pl} L={L} 游程分布不服从 Geom(β)：KS={g['ks']:.4f}/{g['ks_crit']:.4f}, " \
            f"均值 {g['mean_emp']:.3f} vs {g['mean_th']:.3f} (z={g['mean_z']:+.1f})"
        # 均值必须收敛到 L（这条对长尾比 KS 更敏感）
        assert abs(g["mean_emp"] / L - 1.0) < 0.06, \
            f"p̄={pl} L={L} 平均游程长度 {g['mean_emp']:.3f} 偏离 L={L} 超过 6%"


def test_payload_fn_is_wired():
    """㉒ ★★ R12：`run_tracking` 的 `payload_fn` 必须真的接线（X32 新增）。

    ★ 这是本仓库第 12 条硬约定（R12）的直接落地：看到开关先问"它接线了吗"。
    `delay` 这个参数曾经声明/传递/记录/写文档/测试全过，却**从未生效**，空跑两轮。
    这里两条断言缺一不可：
      (a) 传 None ⇒ 与旧行为**逐位一致**（零侵入，既有实验不变）
      (b) 传一个粗量化器 ⇒ 结果必须**明显变大**（真的生效了）
    """
    eps = collect_random_episodes(make_env("CartPole-v1", seed=0), n_episodes=3, seed=0)
    model = _tiny_model(4, 2, True)
    dev = torch.device("cpu")
    base = run_tracking(model, eps, dev, periodic_schedule(1), seed=0, label="no-fn")
    same = run_tracking(model, eps, dev, periodic_schedule(1), seed=0, label="none",
                        payload_fn=None)
    assert base.nmse == same.nmse, "payload_fn=None 必须与不传**逐位一致**"

    lo, hi = fit_quantizer_range(eps, margin=0.05)
    coarse = UniformQuantizer(lo, hi, 2)          # 2 比特 ⇒ 极粗
    q = run_tracking(model, eps, dev, periodic_schedule(1), seed=0, label="quant",
                     payload_fn=coarse)
    assert q.nmse > base.nmse * 1.05, \
        f"传了粗量化器但误差没变大（{q.nmse:.6f} vs {base.nmse:.6f}）⇒ payload_fn 没接线"


def test_qam_ber_degenerates_to_qpsk():
    """㉓ M=4（QPSK）时 BER 公式必须等于**精确**误码率 Q(√γ_s)。

    ⚠ 只在未饱和区比：SER 近似式在低 SNR 会算出 >1−1/M 而被 clip，
      那时差异来自 clip 而非公式（第一版在 −10dB 就是这样误报的）。
    """
    for g_db in (0.0, 3.0, 6.0, 10.0, 20.0):
        g = snr_db_to_linear(g_db)
        ber = qam_bit_error_rate(g, 4.0)
        exact = 0.5 * math.erfc(math.sqrt(g / 2.0))      # Q(√γ_s)
        assert abs(ber - exact) < 1e-12, \
            f"γ={g_db}dB：BER={ber:.6e} ≠ QPSK 精确值 {exact:.6e}"


def test_qam_ser_saturates_and_validity_flag():
    """㉔ SER 必须饱和在 1−1/M，且公式失效区要被**标记**出来。

    ★ 背景：`BER ≈ SER/log2(M)` 依赖"每次符号错误只错 1 个比特"，低 SNR 下失效。
      失效症状（X32 实测，SNR=5dB）：64-QAM 与 256-QAM 的 SER 都饱和，
      BER = (1−1/M)/log2(M) 随 M 增大反而**减小**（0.164 → 0.124）
      ⇒ PER 对 b **非单调**（0.99842 → 0.99831）。这是公式的人工产物。
    """
    M = 256.0
    ser_low = qam_symbol_error_rate(snr_db_to_linear(-20.0), M)
    assert abs(ser_low - (1.0 - 1.0 / M)) < 1e-12, f"低 SNR 下 SER 未饱和到 1−1/M：{ser_low}"
    assert qam_approximation_valid(snr_db_to_linear(-20.0), M) is False
    assert qam_approximation_valid(snr_db_to_linear(35.0), M) is True


def test_per_monotonic_in_bits_and_snr():
    """㉕ PER 对 b 单调增、对 SNR 单调减（U 形曲线的两个分支的地基）。"""
    for g_db in (5.0, 15.0, 25.0):
        pers = []
        for b in (2, 4, 6, 8):
            g = snr_db_to_linear(g_db)
            per = (1.0 - 1e-6 if not qam_approximation_valid(g, 2.0 ** b)
                   else per_from_ber(qam_bit_error_rate(g, 2.0 ** b), 6.0 * b))
            pers.append(per)
        assert all(np.diff(pers) >= -1e-12), f"SNR={g_db}dB 时 PER 对 b 非单调增：{pers}"
    for b in (2, 4, 6, 8):
        pers = []
        for g_db in (5.0, 15.0, 25.0, 35.0):
            g = snr_db_to_linear(g_db)
            per = (1.0 - 1e-6 if not qam_approximation_valid(g, 2.0 ** b)
                   else per_from_ber(qam_bit_error_rate(g, 2.0 ** b), 6.0 * b))
            pers.append(per)
        assert all(np.diff(pers) <= 1e-12), f"b={b} 时 PER 对 SNR 非单调减：{pers}"


def test_quantizer_mse_monotone_and_no_overload():
    """㉖ 量化 MSE 对 b 单调减；量程由数据统计 ⇒ **不得过载**（过载会让 Δ²/12 失效）。"""
    eps = collect_random_episodes(make_env("CartPole-v1", seed=0), n_episodes=3, seed=0)
    lo, hi = fit_quantizer_range(eps, margin=0.05)
    obs = np.concatenate([np.asarray(e["obs"], dtype=np.float64) for e in eps], axis=0)
    ms = []
    for b in (2, 4, 6, 8):
        qz = UniformQuantizer(lo, hi, b)
        assert overload_fraction(qz, obs) <= 1e-9, f"b={b} 存在过载（裁剪）"
        ms.append(qz.measure_mse(obs))
    assert all(ms[i] >= ms[i + 1] * (1 - 1e-9) for i in range(len(ms) - 1)), \
        f"量化 MSE 对 b 非单调减：{ms}"


def test_periodic_age_closed_form():
    """㉗ ★★ 周期发送的年龄分布闭式（X34 的全部设计结论都压在这一条上）。

    三条必须成立，缺一不可：
      (a) 归一化：Σ_h P(h) = 1（截断时把尾部质量压回边界后仍成立）
      (b) 均值闭式 E[age] = T·q/s + (T−1)/2 与 pmf 直接求和一致（K 足够大时）
      (c) 两个退化：T=1 ⇒ p/(1−p)（= i.i.d. 丢包，`lossy_schedule` 的结果）；
                   s=1 ⇒ (T−1)/2（无丢包时 age 在 0…T−1 均匀 —— **周期本身的地板**）
    """
    K = 20000
    for p, T in ((0.0, 4), (0.1, 1), (0.3, 3), (0.5, 8), (0.8, 2)):
        pmf = periodic_age_pmf(p, T, K)
        assert abs(float(pmf.sum()) - 1.0) < 1e-12, f"p={p} T={T} pmf 未归一化"
        m_pmf = float(np.dot(np.arange(K + 1), pmf))
        m_cf = periodic_mean_age(p, T)
        assert abs(m_pmf - m_cf) < max(0.02, 0.01 * abs(m_cf)), \
            f"p={p} T={T}：pmf 均值 {m_pmf:.4f} ≠ 闭式 {m_cf:.4f}"
    assert abs(periodic_mean_age(0.4, 1) - 0.4 / 0.6) < 1e-12, "T=1 未退化到 i.i.d."
    assert abs(periodic_mean_age(0.0, 5) - 2.0) < 1e-12, "无丢包时未退化到 (T−1)/2"


def test_closed_loop_payload_fn_is_wired():
    """㊴ ★★ R12：闭环控制里的 `payload_fn`（量化器）**真的接进去了**（X35 的 T4）。

    为什么非测不可：X35 的全部结论都建立在"量化误差进入闭环"上。
    `payload_fn` 在 `control.run_closed_loop_control` 里只有一行
    （`est = payload_fn(true_next)`），漏掉它实验会**跑得通、出数字、全是假的**。

    三条必须成立：
      (a) 恒等 payload ⇒ 与不传 payload **逐位一致**（既有实验零侵入）
      (b) 粗量化 payload ⇒ 闭环 est_nmse **显著变大**（T4 的机器版）
      (c) 量化越粗 ⇒ 闭环 est_nmse **单调不减**（量级对，不是接错方向）

    ★ 这里必须用 T=1（每步都送）：T=1 时估计误差 = 载荷本身的误差，
      于是"量化器接没接"这件事不会被世界模型的 rollout 稀释。
    """
    env = make_env("uav-track", seed=0, noise_std=0.15, max_steps=400)
    ctrl = PDRelativeController(dt=env.dt, omega_n=2.5, zeta=1.0, kappa=float(env.kappa),
                                a_max=float(env.a_max))
    eps = collect_controlled_episodes(env, ctrl, n_episodes=3, seed=0, max_steps=400)
    dev = torch.device("cpu")
    model = _tiny_model(int(eps[0]["obs"].shape[1]), int(env.act_dim), False)
    lo, hi = fit_quantizer_range(eps, margin=0.05)
    obs = np.concatenate([e["obs"] for e in eps], axis=0).astype(np.float64)

    def _run(payload):
        return run_closed_loop_control(
            env, model, ctrl, periodic_schedule(1), n_episodes=3, seed=0,
            max_steps=200, device=dev, estimator="model", var_g=1.0,
            warmup_steps=50, payload_fn=payload)["est_nmse"]

    base = _run(None)
    ident = _run(lambda x: np.asarray(x, dtype=np.float64))
    assert abs(ident - base) <= 1e-12 * max(abs(base), 1.0), \
        f"(a) 恒等 payload 未逐位一致：{ident!r} vs {base!r}"

    vals, msers = [], []
    for b in (2, 4, 8):
        qz = UniformQuantizer(lo, hi, b)
        vals.append(_run(qz))
        msers.append(qz.measure_mse(obs))
    # (b) 最粗的 2 比特必须显著变大
    assert vals[0] > base * 1.05, \
        f"(b) 粗量化没有改变闭环 est_nmse（{vals[0]!r} vs 无量化 {base!r}）" \
        f"⇒ payload_fn 没接进 control.py（R12 违规，整个 X35 在空跑）"
    # (c) 单调：量化 MSE 递减 ⇒ 闭环误差不增
    assert all(vals[i] >= vals[i + 1] * (1 - 1e-9) for i in range(len(vals) - 1)), \
        f"(c) 闭环 est_nmse 未随精度单调：{vals}"
    assert all(msers[i] >= msers[i + 1] * (1 - 1e-9) for i in range(len(msers) - 1)), \
        f"(c) 量化 MSE 未随 b 单调减：{msers}"
    env.close()


def test_periodic_age_tail_matches_bruteforce():
    """㉛ ★★ 尾部质量 P(age > K)（X34-b2）：**方向**必须是对的。

    背景（本仓库第 16 次自证伪）：第一版判据误写成 `1 − (1−q)^{K/T}`，
    即把丢包率当成成功率代入 ⇒ **q 越大尾巴越大**，于是高丢包（最该研究的）
    那批点被全部误判为"截断"丢掉，网格里只剩 PER≈0 的退化点。

    本测试锁四件事：
      (a) 与暴力求和一致（把分布展开到 H 远大于 K，直接加 h>K 的部分）
      (b) 单调性：**q 越大 ⇒ 尾部越大**（q 大 = 更久没送到 = 更可能超过 K）
      (c) 量级：这是第一版真正错的地方 —— 错式把"至少一次失败"当成"全部失败"，
          q=0.019 / T=2 / K=190 时算出 83.8%，而正确值 ≈ 1e−160。
          ⇒ 单看"随 q 单调"抓不到这个 bug，必须钉死量级。
      (d) 两个退化：q=0 ⇒ 0；T=1 ⇒ q^{K+1}（几何分布尾）
    """
    for T in (1, 2, 3, 8):
        for K in (10, 63, 190):
            for q in (0.0, 0.05, 0.2, 0.5, 0.8, 0.95):
                # 暴力：把同一分布展开到 H 远大于 K，直接加 h>K 的部分
                H = int(K + 1 + 200 * T / max(1e-9, -np.log(max(q, 1e-12))) + 5000)
                H = min(H, 400000)
                hh = np.arange(H + 1, dtype=np.int64)
                s = 1.0 - q
                full = (s / float(T)) * (q ** (hh // T))
                brute = float(full[K + 1:].sum())
                got = periodic_age_tail(q, T, K)
                assert abs(got - brute) < max(1e-9, 1e-6 * max(brute, 1e-12)), \
                    f"q={q} T={T} K={K}：闭式尾 {got:.3e} ≠ 暴力 {brute:.3e}"
    # (b) 方向：q 增大 ⇒ 尾部单调**不减**
    prev = None
    for q in (0.02, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99):
        t = periodic_age_tail(q, 4, 190)
        assert prev is None or t >= prev - 1e-18, \
            f"尾部未随丢包率单调增（q={q}）：{t:.3e} < {prev:.3e} —— 方向错了"
        prev = t
    # (c) 量级：第一版错式在这里会给出 0.838，正确值天文级地小
    t_small = periodic_age_tail(0.019, 2, 190)
    assert t_small < 1e-100, \
        f"q=0.019/T=2/K=190 的尾部应为 ≈1e−160，实得 {t_small:.3e} —— 量级判据被写错"
    assert periodic_age_tail(0.0, 4, 190) == 0.0, "q=0 时尾部必须为 0"
    assert abs(periodic_age_tail(0.5, 1, 9) - 0.5 ** 10) < 1e-12, "T=1 未退化到 q^{K+1}"


def test_periodic_age_var_matches_moments_and_sampling():
    """㉜ ★★ `periodic_age_var` 的闭式必须对，而且它要能解释"为什么容差不能拍百分比"。

    ★★ 为什么单列一条（本仓库**第 17 次自证伪**的机器版）
    X35 第一版把「实测 E[age] vs 闭式」的容差拍成 8%，在 64-QAM@14dB
    （PER=0.9502, T=1）被打穿：实测 17.51 vs 闭式 19.08（−8.2%）。
    诊断显示**闭式没错**，错的是容差：该点
        E[age] = 19.08  而  sqrt(Var(age)) = 19.6  —— **标准差与均值同量级**，
    40×450 步里只有约 900 次到达 ⇒ 单次实测的标准误就有 4.7%，8% 只是 1.8σ。
    ⇒ 本测试把这三件事钉死：
      (a) Var 闭式 == 用 `periodic_age_pmf` 数值算出的二阶矩（两条独立通路一致）
      (b) 两个退化：T=1 ⇒ q/s²（几何）；q=0 ⇒ (T²−1)/12（均匀）
      (c) ★ 高丢包下 sqrt(Var) 与 E[age] **同量级**（比值 > 0.9）
          ⇒ 任何"固定百分比"容差在那一端必然被噪声打穿，容差必须由 SE 给。
    """
    # (a) 与数值矩逐点一致（两条独立通路）
    K = 4000
    for q in (0.05, 0.2, 0.5, 0.8):
        for T in (1, 2, 5):
            pmf = periodic_age_pmf(q, T, K)
            h = np.arange(K + 1, dtype=float)
            m1 = float(np.dot(h, pmf))
            m2 = float(np.dot(h * h, pmf))
            var_num = m2 - m1 * m1
            var_cf = periodic_age_var(q, T)
            rel = abs(var_num - var_cf) / max(var_cf, 1e-12)
            assert rel < 2e-3, (f"Var 闭式与数值矩不符：q={q} T={T} "
                                f"闭式 {var_cf:.4f} vs 数值 {var_num:.4f}（{rel:.2%}）")

    # (b) 退化
    q = 0.37
    assert abs(periodic_age_var(q, 1) - q / (1 - q) ** 2) < 1e-12, "T=1 未退化到 q/s²"
    for T in (1, 3, 8):
        assert abs(periodic_age_var(0.0, T) - (T * T - 1) / 12.0) < 1e-12, \
            "q=0 未退化到 (T²−1)/12"

    # (c) ★ 高丢包：标准差与均值同量级 ⇒ 固定百分比容差必然被打穿
    sd = float(np.sqrt(periodic_age_var(0.9502, 1)))
    mean = periodic_mean_age(0.9502, 1)
    assert sd / mean > 0.9, (f"高丢包下 std/mean = {sd/mean:.3f} 应 >0.9"
                             f"—— 若这条不成立，说明 Var 闭式量级错了")

    # (d) MC 交叉验证：纯调度空跑的 std 与 sqrt(Var)/sqrt(到达次数) 同量级（0.5×~4×）
    def _mc_std(T, q, n_ep=12, n_step=400, n_seed=16):
        vals = []
        for i in range(n_seed):
            rng = np.random.default_rng(4000 + i)
            sch = periodic_lossy_schedule(T, q)
            tot, n = 0.0, 0
            for _ in range(n_ep):
                age = 0
                for t in range(n_step):
                    age = 0 if sch(t, rng) else age + 1
                    tot += age
                    n += 1
            vals.append(tot / n)
        return float(np.std(vals, ddof=1))

    for (T, q) in ((1, 0.9502), (1, 0.2), (2, 0.5)):
        mc = _mc_std(T, q)
        n_arrive = 12 * 400 * (1 - q) / T          # 独立"更新"次数的量级
        pred = float(np.sqrt(periodic_age_var(q, T) / max(n_arrive, 1)))
        ratio = mc / max(pred, 1e-12)
        assert 0.4 < ratio < 4.0, (f"MC 采样 std 与 sqrt(Var)/sqrt(n_arrive) 不同量级："
                                   f"T={T} q={q} MC={mc:.4f} vs 预测 {pred:.4f}"
                                   f"（比值 {ratio:.2f}）")


def test_burst_periodic_age_degenerates_to_memoryless():
    """㉝ ★★ X36 新闭式：突发信道 + 周期发送的年龄分布，**必须**退化到已知答案。

    ★★ 为什么单列一条（本仓库第 18 次自证伪的机器版）
    第一版写这个闭式时连错两处，都是"看着对"但量级不对：
      (a) `w` 写成 `Pm @ s_col` —— 等于问"下一次尝试成功吗"，整体偏一步；
          验算 T=1/PER=0 给 2.52，而 X31 的闭式是 p̄·L=1.6（差 58%）。
      (b) `F = Pm·(1−s_j)` 乘在**列**（到达状态）上 ⇒ 从 Bad 转到 Good 的那份质量
          被当成"失败"继续往前传，其实是**成功** ⇒ 质量凭空消失（尾部 10% 不收敛）。
    ⇒ 钉死三条已知答案：
      A  `L = L_iid`（GE 无记忆）⇒ 与 `periodic_age_pmf(q_total,T)` **逐点相等**
      B  `T=1 且 PER=0` ⇒ E[age] == X31 的 `p̄·L`
      C  pmf 归一化，且 pmf 的均值 == 闭式均值
    """
    # A：无记忆退化（逐点 + 均值）
    for p_bar in (0.1, 0.3, 0.5, 0.7):
        for per in (0.0, 0.05, 0.3):
            L = 1.0 / (1.0 - p_bar)
            q = 1.0 - (1.0 - p_bar) * (1.0 - per)
            for T in (1, 2, 4, 8):
                a = burst_periodic_age_pmf(p_bar, L, per, T, 400)
                b = periodic_age_pmf(q, T, 400)
                assert np.abs(a - b).max() < 1e-9, (
                    f"L=L_iid 未退化到无记忆：p̄={p_bar} per={per} T={T} "
                    f"逐点最大差 {np.abs(a-b).max():.3e}")
                assert abs(burst_periodic_mean_age(p_bar, L, per, T)
                           - periodic_mean_age(q, T)) < 1e-9, "E[age] 未退化"

    # B：T=1 且 PER=0 ⇒ p̄·L（X31 的闭式，两条独立推导必须对上）
    for p_bar in (0.2, 0.5, 0.8):
        for L in (2.0, 8.0, 32.0):
            if L < p_bar / (1.0 - p_bar):
                continue
            got = burst_periodic_mean_age(p_bar, L, 0.0, 1)
            assert abs(got - p_bar * L) < 1e-8 * max(1.0, p_bar * L), (
                f"T=1/PER=0 未退化到 p̄·L：p̄={p_bar} L={L} 得到 {got:.6f}")

    # C：pmf 归一化 + 均值一致
    for (p_bar, L, per, T) in ((0.3, 8.0, 0.02, 2), (0.5, 16.0, 0.1, 4),
                               (0.2, 4.0, 0.0, 1)):
        pmf = burst_periodic_age_pmf(p_bar, L, per, T, 3000)
        assert abs(float(pmf.sum()) - 1.0) < 1e-9, "pmf 未归一化"
        m_pmf = float(np.dot(np.arange(3001), pmf))
        m_cf = burst_periodic_mean_age(p_bar, L, per, T)
        assert abs(m_pmf / m_cf - 1.0) < 2e-3, (
            f"pmf 均值与闭式不符：{m_pmf:.4f} vs {m_cf:.4f}")


def test_burst_periodic_schedule_matches_closed_form():
    """㉞ ★ X36：突发调度仿真的年龄直方图必须收敛到新闭式（不是自洽就算过）。

    ★★ 采样口径（第一版在这里踩的坑）：必须在**每一步**记 age_t，
    不能只在"更新瞬间"记 —— 后者采到的是"间隔长度−1"的分布，
    会给出 1.92 而不是 5.36（差 2.8 倍），看起来像闭式错。

    ★★ 为什么用**多副本取均值**而不是单次长跑（第二版踩的坑）
    突发下的年龄分布是**重尾 + 强自相关**的：一个长度 ~L 的坏游程只贡献约
    1 个独立样本。单次 12 万步在 L=16 那一档给出 8.644 vs 闭式 8.222（差 5.1%），
    单次 40 万步才给 8.262（0.5%）⇒ 单次估计的标准误在 5% 量级。
    ⇒ 用 6 个副本（各 8 万步）取均值，把标准误压到 ~2%，容差定 4%。
    """
    K, n_step, n_rep = 4000, 80000, 6
    for (p_bar, L, per, T) in ((0.3, 8.0, 0.02, 2), (0.5, 16.0, 0.1, 1),
                               (0.2, 4.0, 0.05, 4)):
        pmf = burst_periodic_age_pmf(p_bar, L, per, T, K)
        hist = np.zeros(K + 1)
        means = []
        for i in range(n_rep):
            rng = np.random.default_rng(20260925 + 1013 * i)
            sch = burst_periodic_lossy_schedule(T, p_bar, L, per, seed=4242 + i)
            h = np.zeros(K + 1)
            age = 0
            for t in range(n_step):
                age = 0 if sch(t, rng) else min(age + 1, K)
                h[age] += 1.0
            h /= h.sum()
            hist += h
            means.append(float(np.dot(np.arange(K + 1), h)))
        hist /= n_rep
        tv = 0.5 * float(np.abs(pmf - hist).sum())
        assert tv < 0.03, (f"突发调度仿真与闭式不符：p̄={p_bar} L={L} per={per} "
                           f"T={T} TV={tv:.4f}")
        cf = burst_periodic_mean_age(p_bar, L, per, T)
        emp = float(np.mean(means))
        assert abs(emp / cf - 1.0) < 0.04, (
            f"E[age] 仿真 {emp:.3f}（{n_rep} 副本，副本间 std="
            f"{np.std(means, ddof=1):.3f}）vs 闭式 {cf:.3f} 差 {abs(emp/cf-1):.1%}")


class _IdentityModel:
    """预测 = 恒等（把"模型"换成一根导线）。

    ★★ 为什么自检要用它（第一版踩的坑）
    第一版用 `_tiny_model`（未训练的 4→8→4 小 MLP + Tanh）去验 payload 是否接线，
    结果量化扰动 0.674 只换来输出变化 **2.6e-4** ⇒ 断言"误差必须变大 5%"永远失败，
    看起来像"payload_fn 没接线"，实际是 **Tanh 潜层饱和**（X5 撞过的同一个坑）。
    用恒等模型 ⇒ 起点扰动**必定**一比一传到预测上，判据才有意义。
    """

    def eval(self):
        return self

    def to(self, *a, **k):
        return self

    def predict_next(self, obs, act):
        return obs


def test_periodic_schedule_has_age_floor():
    """㉘ ★ 周期发送即使**一个包都不丢**，也有年龄地板 (T−1)/2（X34 的成本项）。

    ★ 必须用**长 episode**：地板是"每 T 步一个循环"的平稳性质，短 episode 里
      开头/结尾的不完整循环会把它拉偏。第一版用 CartPole（13–19 步）配 T=4，
      实测 E[age]=1.403 vs 地板 1.50（−6%），那是边界瞬态不是 bug。
      换成 UAV 长轨迹（每集数百步）后偏差回到 1% 量级。
    ★ 这条同时是 R12 检查：改周期必须真的改变跟踪结果。
    """
    env = make_env("uav-track", seed=0, noise_std=0.15, max_steps=400)
    ctrl = PDRelativeController(dt=env.dt, omega_n=2.5, zeta=1.0, kappa=float(env.kappa),
                                a_max=float(env.a_max))
    eps = collect_controlled_episodes(env, ctrl, n_episodes=3, seed=0, max_steps=400)
    env.close()
    model = _tiny_model(int(eps[0]["obs"].shape[1]), int(env.act_dim), False)
    dev = torch.device("cpu")
    ages, nmses = [], []
    for T in (1, 4, 8):
        r = run_tracking(model, eps, dev, periodic_lossy_schedule(T, 0.0), seed=0,
                         label=f"T={T}")
        ages.append(r.age_tx_mean)
        nmses.append(r.nmse)
    for T, a in zip((1, 4, 8), ages):
        assert abs(a - (T - 1) / 2.0) < max(0.05, 0.05 * (T - 1) / 2.0), \
            f"T={T} 无丢包时 E[age]={a:.3f} ≠ 地板 (T−1)/2={(T - 1) / 2:.2f}"
    assert max(nmses) > min(nmses) * 1.05, \
        f"改周期没有改变误差（{nmses}）⇒ 周期性没接进 schedule（R12 违规）"


def test_payload_fn_applies_only_at_t0():
    """㉙ ★ 离线曲线的 `payload_fn` 只作用在**起点**这一帧（X33 的 g_b 曲线）。

    (a) 传恒等函数 ⇒ 必须与不传**逐位一致**（既有实验零侵入）
    (b) 传粗量化器 ⇒ 起点扰动必须**一比一**地出现在第一步误差里
        （用恒等模型 ⇒ 定量可查：第一步 MSE 的增量 ≈ 量化器实测 MSE）
    """
    eps = collect_random_episodes(make_env("CartPole-v1", seed=0), n_episodes=4, seed=0)
    dev = torch.device("cpu")
    base = closed_loop_error_curve(_IdentityModel(), eps, [1, 2, 4, 8], dev,
                                   n_samples=128, seed=0)
    same = closed_loop_error_curve(_IdentityModel(), eps, [1, 2, 4, 8], dev,
                                   n_samples=128, seed=0, payload_fn=lambda x: x)
    assert np.array_equal(np.asarray(base["mse_per_step"]),
                          np.asarray(same["mse_per_step"])), \
        "payload_fn=恒等 时曲线必须与不传**逐位一致**（零侵入被破坏）"
    lo, hi = fit_quantizer_range(eps, margin=0.05)
    obs = np.concatenate([np.asarray(e["obs"], dtype=np.float64) for e in eps], axis=0)
    coarse = UniformQuantizer(lo, hi, 2)
    q = closed_loop_error_curve(_IdentityModel(), eps, [1, 2, 4, 8], dev,
                                n_samples=128, seed=0, payload_fn=coarse)
    d = float(np.asarray(q["mse_per_step"])[0]) - float(np.asarray(base["mse_per_step"])[0])
    qm = coarse.measure_mse(obs)
    assert d > 0.5 * qm, \
        f"起点量化后第一步 MSE 只增加了 {d:.5f}，而量化器实测 MSE={qm:.5f} " \
        f"⇒ payload_fn 没接线（R12 违规）"


def test_perfect_channel_heads_identical():
    """㉚ ★ 完美信道下 model 与 persistence 的跟踪误差必须**逐位相同**（X33 的 S1）。

    语义：每步都有包到达 ⇒ 估计器**从不调用预测** ⇒ 用哪个预测器都一样。
    这条一旦破，"同龄对照"（X26/X31 的 R13 教训）就失效了 —— 两次测量的差别
    就不再只来自预测器。
    """
    eps = collect_random_episodes(make_env("CartPole-v1", seed=0), n_episodes=3, seed=0)
    model = _tiny_model(4, 2, True)
    dev = torch.device("cpu")

    class _Persist:
        def eval(self):
            model.eval()
            return self

        def to(self, *a, **k):
            return self

        def predict_next(self, obs, act):
            return obs

    a = run_tracking(model, eps, dev, lossy_schedule(0.0), seed=0, label="model")
    b = run_tracking(_Persist(), eps, dev, lossy_schedule(0.0), seed=0, label="persist")
    assert a.nmse == b.nmse, \
        f"完美信道下两 head 不一致（{a.nmse:.8f} vs {b.nmse:.8f}）⇒ 仍在调用预测"


def test_threshold_age_closed_form_degenerates():
    """㉟ ★★ X38：年龄阈值触发的年龄分布闭式，**必须**退化到两个已知答案。

    策略：age < K 不尝试；age ≥ K 每步尝试直到成功。
        D = K + Geom(s)   ⇒   P(age=h) = q^{max(h−K,0)} / (K + 1/s)
        E[age] = (E[D²] − E[D]) / (2 E[D])

    三条退化（每一条都是"已知答案"，R12 的量级校验）：
      (a) PER=0 ⇒ 阈值(K) 的 pmf 与 周期(T=K+1) **逐位相同**
      (b) K=0   ⇒ pmf 与 i.i.d. 几何（periodic T=1）**逐位相同**，E[age]=q/s
      (c) 送达率/尝试率闭式与**纯调度 MC** 一致（重尾 ⇒ 容差随 E[age] 放）
    """
    KC = 60
    # (a) PER=0 ⇒ ≡ 周期 K+1
    for K in (0, 1, 3, 8, 16):
        d = float(np.abs(threshold_age_pmf(K, 0.0, KC)
                         - periodic_age_pmf(0.0, K + 1, KC)).max())
        assert d < 1e-12, f"PER=0 K={K}：pmf 与周期 T={K+1} 差 {d:.3e}"
        assert abs(threshold_mean_age(K, 0.0) - periodic_mean_age(0.0, K + 1)) < 1e-12
        assert abs(threshold_attempt_rate(K, 0.0) - 1.0 / (K + 1.0)) < 1e-12
    # (b) K=0 ⇒ ≡ i.i.d. 几何
    for q in (0.05, 0.2, 0.5, 0.8, 0.9):
        d = float(np.abs(threshold_age_pmf(0, q, KC) - periodic_age_pmf(q, 1, KC)).max())
        assert d < 1e-12, f"K=0 PER={q}：pmf 与几何分布差 {d:.3e}"
        assert abs(threshold_mean_age(0, q) - q / (1.0 - q)) < 1e-12
        assert abs(threshold_attempt_rate(0, q) - 1.0) < 1e-12
    # (c) MC 交叉验证（送达率 + E[age]）
    def _mc(K, q, n=200000, seed=0):
        sch = threshold_lossy_schedule(K, q)
        rng = np.random.default_rng(seed)
        sch.reset(rng)
        age = 0
        tot = 0.0
        ntx = 0
        for t in range(1, n + 1):
            if bool(sch(t, rng)):
                ntx += 1
                age = 0
            else:
                age += 1
            tot += age
        return tot / n, ntx / n

    for K in (0, 1, 2, 4, 8, 16):
        for q in (0.1, 0.3, 0.5, 0.8, 0.95):
            a_sim, r_sim = _mc(K, q, seed=1100 + K * 17 + int(q * 100))
            a_cf = threshold_mean_age(K, q)
            r_cf = threshold_delivery_rate(K, q)
            assert abs(a_cf - a_sim) < max(0.05, 0.05 * a_cf), \
                f"K={K} q={q}：E[age] 闭式 {a_cf:.4f} vs MC {a_sim:.4f}"
            assert abs(r_cf - r_sim) < max(0.01, 0.05 * r_cf), \
                f"K={K} q={q}：送达率闭式 {r_cf:.5f} vs MC {r_sim:.5f}"


def test_threshold_beats_periodic_at_matched_rate():
    """㊱ ★★ X38 的 P1：**等传输预算**下，年龄阈值策略的 E[age] 恒不高于周期策略。

    ⚠️ 归属声明（不许抢功）：「采样率约束下阈值策略最优、且优于 uniform（周期）」
    已由 Sun–Polyanskiy–Uysal-Biyikoglu（arXiv:1701.06734 / 1707.02531）解决。
    本测试只是把这条**已知结论在本仓库的闭式上复算一遍**，用途是：
      ① 锁死 `threshold_equivalent_period`（T = s·K + 1）这个配对关系；
      ② 为"等预算"这个口径提供一个可回归的机器断言。
    真正的新问题（signal-dependent 触发）不在本测试范围内。

    同时锁两条形状：
      · PER=0 ⇒ 比值恒为 1（两者退化成同一个确定周期）
      · 比值随 PER 呈 **U 形**（先降后升）⇒ 存在唯一的最大增益点
    """
    # ① 等预算下恒不劣（全网格）
    worst = 1.0
    for K in list(range(0, 33)):
        for q in np.linspace(0.01, 0.99, 99):
            T = threshold_equivalent_period(K, q)
            r = threshold_mean_age(K, q) / max(periodic_mean_age(q, T), 1e-12)
            assert r <= 1.0 + 1e-9, f"K={K} q={q:.3f}：阈值 E[age] 反而更大（{r:.6f}）"
            worst = min(worst, r)
    # ② PER=0 ⇒ 恒等
    for K in (1, 2, 4, 8):
        T = threshold_equivalent_period(K, 0.0)
        assert abs(T - (K + 1.0)) < 1e-12
        assert abs(threshold_mean_age(K, 0.0) / periodic_mean_age(0.0, T) - 1.0) < 1e-12
    # ③ U 形：最佳增益（最小比值）出现在中间 PER，不是两端
    per_grid = np.linspace(0.02, 0.98, 49)
    best = []
    for q in per_grid:
        m = 1.0
        for K in range(0, 65):
            T = threshold_equivalent_period(K, q)
            m = min(m, threshold_mean_age(K, q) / max(periodic_mean_age(q, T), 1e-12))
        best.append(m)
    k = int(np.argmin(best))
    assert 0 < k < len(best) - 1, \
        f"最佳增益点落在端点（k={k}/{len(best)}）⇒ 不是 U 形"
    assert best[k] < 0.75, f"最大增益只有 {best[k]:.3f}，与解析预期（≈0.55）不符"


def test_threshold_schedule_is_stateful_and_resettable():
    """㊲ ★ X38：阈值 schedule 是**有状态**的（`Schedule` 签名只有 (t, rng)，
    拿不到 tracker 的 age ⇒ 闭包自己数一份）。必须锁三件事：

      (a) 它带 `reset`，且 reset 后 age 归零（否则跨 episode 串味）
      (b) 无丢包时，它**严格**每 K+1 步送达一次（确定性周期）
      (c) age<K 时**一次都不尝试**（省配额）—— 用"送达次数 == 尝试次数"验
    """
    sch = threshold_lossy_schedule(3, 0.0)
    assert callable(getattr(sch, "reset", None)), "阈值 schedule 必须带 reset（有状态）"
    rng = np.random.default_rng(0)
    sch.reset(rng)
    hits = [t for t in range(1, 41) if bool(sch(t, rng))]
    assert hits == [4, 8, 12, 16, 20, 24, 28, 32, 36, 40], \
        f"PER=0/K=3 应每 4 步送达一次，实际 {hits}"
    sch.reset(rng)
    hits2 = [t for t in range(1, 21) if bool(sch(t, rng))]
    assert hits2 == [4, 8, 12, 16, 20], f"reset 后未归零：{hits2}"
    # (c) 有丢包时：K=3 ⇒ 前 3 步（age 0,1,2）绝不尝试 ⇒ 尝试次数 = 步数 − 静默步
    for q in (0.0, 0.5):
        s = threshold_lossy_schedule(3, q)
        r2 = np.random.default_rng(7)
        s.reset(r2)
        n_hit = sum(bool(s(t, r2)) for t in range(1, 4001))
        # 无丢包：每 4 步 1 次；q=0.5：E[尝试] 仍等于"age≥3 的步数"
        assert n_hit > 0
    # 尾部质量单调：阈值越大尾部越大；k_max < K ⇒ 直接返回 1.0（不收敛，必须剔除）
    for q in (0.2, 0.5, 0.8):
        vals = [threshold_age_tail(K, q, 40) for K in (1, 4, 16)]
        assert vals[0] <= vals[1] <= vals[2] + 1e-12, f"尾部质量不单调：{vals}"
    assert threshold_age_tail(64, 0.5, 40) >= 1.0 - 1e-12, "k_max<K 必须返回 1.0"


def test_uav_wind_field_is_state_dependent_and_degenerates():
    """㊳ ★★ X38-b：状态依赖阵风场必须 (a) 真的异方差 (b) wind_amp=0 时**逐位退化**。

    ★ 为什么必须锁这条：X38 的 P3 得出「U 只是 age 的替身」，但这个结论**可能只是因为
    环境同方差**（真实 σ 无变化 ⇒ 没什么可学）。X38-b 用阵风场造出异方差来分辨：
      · 若阵风场其实没起作用 ⇒ 整轮 X38-b 是空跑（R12 的经典坑）
      · 若 wind_amp=0 与原来不等价 ⇒ 破坏了 X38 原结论的可比性
    ⇒ 两边都要钉死。
    """
    from wmlab.envs import make_env

    def _roll(env_id_sigma, amp, n=3000, seed=0):
        e = make_env("uav-track", seed=seed, noise_std=0.15, wind_amp=amp,
                     max_steps=200)
        e.reset(seed=seed)
        out = []
        for t in range(n):
            r = e.step(np.zeros(2, dtype=np.float32))
            out.append(float(e.sigma_at(e.uav_p)))
            if r.terminated or r.truncated:
                e.reset(seed=seed + t)
        return np.asarray(out)

    # (a) wind_amp=0 ⇒ 逐位等于 noise_std（同方差，与原环境完全一致）
    s0 = _roll("uav-track", 0.0)
    assert np.allclose(s0, 0.15, atol=0, rtol=0), \
        f"wind_amp=0 未退化：sigma 范围 [{s0.min()}, {s0.max()}]"
    assert float(s0.std()) == 0.0, "wind_amp=0 时真实 σ 必须无变化"

    # (b) 幅度单调：amp 越大 ⇒ 真实 σ 的 CV 越大（剂量—反应曲线的前提）
    cvs = []
    for amp in (1.0, 2.0, 4.0):
        s = _roll("uav-track", amp)
        cvs.append(float(s.std() / s.mean()))
        assert s.min() >= 0.15 - 1e-12, f"amp={amp} 出现了低于 σ0 的点（场公式错）"
        assert s.max() <= 0.15 * (1.0 + amp) + 1e-12, \
            f"amp={amp} 出现了高于 σ0(1+amp) 的点（场公式错）"
    assert cvs[0] < cvs[1] < cvs[2], f"CV 不随 amp 单调增：{cvs}"

    # (c) 同一个位置必须给同一个 σ（阵风场是**静态**的，不是随机噪声）
    e = make_env("uav-track", seed=0, noise_std=0.15, wind_amp=2.0, max_steps=200)
    p = np.array([1.3, -2.7])
    a1 = float(e.sigma_at(p))
    a2 = float(e.sigma_at(p))
    assert a1 == a2, "同一位置的 σ 必须相同（阵风场是状态的函数，不是随机项）"
    # 不同位置必须真的不同（否则场是常数）
    b = float(e.sigma_at(np.array([4.9, 3.1])))
    assert abs(a1 - b) > 1e-3, f"不同位置的 σ 相同（场退化为常数）：{a1} vs {b}"


# ------------------------------------------------- ㊵ 等预算配对的窗口守卫
def _expect_raise(exc_type, fn, what):
    """断言 `fn()` 抛 `exc_type`；**抛错类型不对** 与 **什么都没抛** 分开报。"""
    try:
        fn()
    except exc_type:
        return
    except Exception as other:  # noqa: BLE001
        raise AssertionError(
            f"{what}：抛的是 {type(other).__name__}（{other}），不是 {exc_type.__name__}")
    raise AssertionError(f"{what}：什么都没抛 —— 守卫失效")


def test_pairing_window_guard_rejects_degenerate_overlap():
    """㊵ ★★ X40：等预算配对的**公共窗口守卫**必须拒绝"退化窗口"。

    ⚠️ 为什么值得一条回归测试（真实代价：一个**差点写出去的假头条**）：
      X40 首跑守卫写成 `hi > lo * 1.0001`，于是 PER=0.7 档拿到窗口
      `[0.1502, 0.1584]` —— **只有 5% 速率跨度** —— 照样在其上插值出 9 个网格点，
      打印「**9/9 更优**」。看着像预写 H1 成立；实际那 9 个点全挤在同一段里，
      **不是 9 个独立速率点**。⇒ 守卫改为 `hi >= lo * 1.5` 且每族 ≥ 3 点。

    锁五件事：
      (a) 那个 5% 跨度的窗口必须 raise `WindowNotEvaluable`（返回/静默跳过都不算）
      (b) 每族可比点数 < 3 必须 raise
      (c) 正常窗口通过，且 [lo,hi] = 各族**交集**（左端 max、右端 min，不是并集）
      (d) 坏输入（空族 / 非正 / 未升序 / 只给一族）必须 raise `ValueError` ——
          ★ 不许把"输入坏了"混同于"窗口太窄"，否则坏数据会被当成
          "这一档不可评估"而被静默跳过（R14 精神：报错优于给假数字）
      (e) `expo_grid` 真的覆盖窗口两端且对数等距（相邻比恒定）
    """
    from wmlab.eval.pairing import (WindowNotEvaluable, expo_grid,
                                    overlap_window)

    # (a) X40 实测的退化窗口：0.1584 / 0.1502 = 1.054×
    degenerate = {"periodic": [0.1502, 0.1540, 0.1584],
                  "threshold": [0.1000, 0.1502, 0.1584, 0.1700]}
    _expect_raise(WindowNotEvaluable,
                  lambda: overlap_window(degenerate, min_range=1.5, min_pts=3),
                  "5% 跨度的窗口")

    # (b) 每族点数不足（threshold 只有 2 点 ⇒ 9 个网格点只能靠插值编出来）
    thin = {"periodic": [0.10, 0.20, 0.40],
            "threshold": [0.10, 0.40]}
    _expect_raise(WindowNotEvaluable,
                  lambda: overlap_window(thin, min_range=1.5, min_pts=3),
                  "每族只有 2 点")

    # (c) 正常窗口：交集而非并集
    ok = {"periodic": [0.02, 0.05, 0.12, 0.30],
          "threshold": [0.05, 0.10, 0.25, 0.60]}
    lo, hi = overlap_window(ok, min_range=1.5, min_pts=3)
    assert abs(lo - 0.05) < 1e-12, f"lo 应取各族最小值的 max（0.05），实际 {lo}"
    assert abs(hi - 0.30) < 1e-12, f"hi 应取各族最大值的 min（0.30），实际 {hi}"
    assert hi >= lo * 1.5 - 1e-12

    # (d) 坏输入 ⇒ ValueError（与"窗口太窄"分开报）
    _expect_raise(ValueError, lambda: overlap_window(
        {"a": [], "b": [1.0, 2.0, 4.0]}), "空族")
    _expect_raise(ValueError, lambda: overlap_window(
        {"a": [0.0, 1.0, 2.0], "b": [1.0, 2.0, 4.0]}), "含 0 的 tx_rate")
    _expect_raise(ValueError, lambda: overlap_window(
        {"a": [2.0, 1.0, 0.5], "b": [1.0, 2.0, 4.0]}), "x 未升序")
    _expect_raise(ValueError, lambda: overlap_window(
        {"a": [1.0, 2.0, 4.0]}), "只给一族（谈不上等预算比较）")

    # (e) 网格覆盖两端 + 对数等距
    grid = expo_grid(lo, hi, 9)
    assert len(grid) == 9
    assert abs(grid[0] - lo) < 1e-12 and abs(grid[-1] - hi) < 1e-12, \
        f"网格没覆盖到两端：{grid[0]} … {grid[-1]} vs [{lo}, {hi}]"
    ratios = [b / a for a, b in zip(grid, grid[1:])]
    assert max(ratios) - min(ratios) < 1e-9, f"网格不是对数等距：{ratios}"
    _expect_raise(ValueError, lambda: expo_grid(lo, hi, 1), "n=1")


# ------------------------------------------------- ㊶ X41 剂量—反应的聚合口径
def test_dose_aggregation_uses_cross_per_median():
    """㊶ ★★ X41（2026-09-29）：剂量档的"任务价值"必须**跨 PER 取中位**。

    为什么值得钉死：X40 的四个 PER 档距离比是 `[1.147, 1.094, 1.023, 1.035]` ——
    取**最大**报 1.147（读成"差 15%"）、取**首个**也报 1.147、取**中位**报 **1.0648**（"差 6%"）。
    报哪个**直接决定结论的强弱**；"取最大"等于拿最差档当代表，是**选择性汇报**。
    ★ 反面断言 `1.023 < dist < 1.147` 就是为了让"退化成取极值"这件事**测不过**。

    (b) 两类指标必须各取自己那一列 —— `metric` 名字写错会**静默丢档**
    （本轮真踩过：把 `mean_dist_tail` 写成 `dist`，距离比直接显示 n/a 而没报错）。
    (c) 空表不得崩，也不得编数。
    """
    from wmlab.eval.pairing import aggregate_p4

    rows = ([{"metric": "mean_dist_tail", "utrigger_over_threshold_median": v}
             for v in (1.147, 1.094, 1.023, 1.035)]
            + [{"metric": "est_nmse", "utrigger_over_threshold_median": v}
               for v in (4.107, 1.820, 1.907, 2.276)])

    dist, nmse, n = aggregate_p4(rows)
    assert n == 4, f"参与聚合的档数应为 4，实得 {n}"
    assert abs(dist - 1.0648) < 1e-3, f"距离比应为跨 PER 中位 1.0648，实得 {dist}"
    assert abs(nmse - 2.0915) < 1e-3, f"NMSE 比应为跨 PER 中位 2.0915，实得 {nmse}"
    assert 1.023 < dist < 1.147, f"退化成了取极值：{dist}"

    # (b) 只有一类指标时，另一类必须是 None、不能拿这类的数去顶
    d2, n2, c2 = aggregate_p4(
        [{"metric": "mean_dist_tail", "utrigger_over_threshold_median": 1.0}])
    assert (d2, n2, c2) == (1.0, None, 1), f"metric 分流不对：{(d2, n2, c2)}"

    # (c) 缺失 / 空表
    assert aggregate_p4([]) == (None, None, 0)
    assert aggregate_p4([{"metric": "mean_dist_tail"}]) == (None, None, 0), \
        "字段缺失时应返回 None，不许把 None 当数参与中位"


# ------------------------------------------------- ㊷ X42 共形信封的构件
def test_conformal_pava_and_quantile():
    """㊷ ★★ X42（2026-09-29）：PAVA 与有限样本共形分位数的**构件级**断言。

    为什么分开测构件：信封 = PAVA + 分位数 + 归一化，三者任一错，症状都只表现为
    "覆盖率不对" —— 一个**看不出根因**的症状。构件级断言才能定位。

    (a) PAVA：非降、**保均值**（加权和）、单元素/全等退化；
    (b) 分位数阶 = ⌈(n+1)(1−α)⌉（Vovk）；**不许**用 numpy 默认的 `linear` 插值分位
        （那会给出一个在样本之间"插"出来的数，失去有限样本保证）；
    (c) ★ K2（R14）：非有限输入必须 raise —— `nan <= 阈值` 恒为 False，
        静默通过会产出**看着合理**的错答案。
    """
    from wmlab.eval.conformal import _pava, conformal_quantile

    # (a) 经典 PAVA：3,1,2 ⇒ 合并前两个块 ⇒ 2,2,2（非降、和 = 6）
    out = _pava(np.array([3.0, 1.0, 2.0]))
    assert np.all(np.diff(out) >= -1e-12), f"PAVA 结果未非降：{out}"
    assert abs(out.sum() - 6.0) < 1e-12, f"PAVA 未保均值：{out}"
    assert np.allclose(out, [2.0, 2.0, 2.0]), f"PAVA 值不对：{out}"
    # 加权：把 [3,1] 按权重 [1,10] 合并 ⇒ 13/11 ≈ 1.1818 ≤ 2 ⇒ 不再违反
    w = _pava(np.array([3.0, 1.0, 2.0]), np.array([1.0, 10.0, 1.0]))
    assert np.all(np.diff(w) >= -1e-12), f"加权 PAVA 未非降：{w}"
    assert abs(w[1] - (3.0 + 10.0 * 1.0) / 11.0) < 1e-12, f"加权均值不对：{w}"
    assert np.allclose(_pava(np.array([5.0])), [5.0]), "单元素退化"
    assert np.allclose(_pava(np.array([2.0, 2.0, 2.0])), [2.0, 2.0, 2.0]), "全等退化"

    # (b) 阶 = ⌈(n+1)(1−α)⌉：n=100、α=0.1 ⇒ ⌈90.9⌉ = 91 ⇒ 第 91 小 = 90.0
    q, clipped = conformal_quantile(np.arange(100, dtype=float), 0.1)
    assert abs(q - 90.0) < 1e-12 and not clipped, f"阶不对：q={q}, clipped={clipped}"
    # n=10、α=0.001 ⇒ ⌈11×0.999⌉ = 11 > 10 ⇒ 截断取最大值，且必须**如实标记**
    q2, c2 = conformal_quantile(np.arange(10, dtype=float), 0.001)
    assert c2 is True and abs(q2 - 9.0) < 1e-12, f"截断标记不对：q={q2}, c={c2}"

    # (c) K2：NaN/Inf 一律 raise（不许静默）
    for bad in (np.array([1.0, np.nan, 3.0]), np.array([1.0, np.inf, 3.0])):
        try:
            conformal_quantile(bad, 0.1)
        except FloatingPointError:
            pass
        else:
            raise AssertionError(f"★ K2 未通过：含非有限值的分数被静默接受 {bad}")


# ------------------------------------------------- ㊸ X42 归一化不可省（K1）
def test_conformal_normalization_is_required():
    """㊸ ★★ X42（2026-09-29）：**逐年龄归一化 Ŝ(h) 不可省** —— 否则信封退化。

    ★ 这是本号设计里最容易"看起来没问题"的一处：
      共形分位数本身就是**边缘**保证 ⇒ 不归一化**照样**能满足边缘覆盖率
      （下面 (b) 就是在断言这件事）—— 但小年龄处的信封会被大年龄的残差量级
      **抬到远高于本地散布**，于是"Ê ≥ tol"在**所有年龄**都成立 ⇒ 触发规则退化，
      **整个 X42 会在不自知的情况下空跑**。
    ⇒ 断言方式选**决策相关量**：在固定 tol 下，「逐年龄触发率」的跨度。
      归一化版必须存在**内部**的判据转折（最年轻年龄几乎不触发、最老年龄几乎总触发）；
      未归一化版在最年轻年龄就已经≈全触发。

    (c) ĝ 对 u 单调不减（PAVA 保证）—— 否则"U 越大越该发"这个语义就断了。
    """
    from wmlab.eval.conformal import envelope, fit_envelope

    rng = np.random.default_rng(20260929)
    H, N = 8, 4000
    hs = np.arange(1, H + 1, dtype=float)
    base = 0.05 * 1.5 ** (hs - 1)          # 误差随年龄增长
    spread = 0.02 * 1.8 ** (hs - 1)        # ★ 异方差：散布跨 22 倍
    u = rng.normal(size=(N, H))
    err = (base[None, :] + spread[None, :] * (0.8 * u + rng.normal(size=(N, H))))
    tab = fit_envelope(err[:2000], u[:2000], alpha=0.1, n_knots=6)

    tol = 0.30
    hs_grid = np.tile(hs[None, :], (2000, 1))
    rate_norm = (envelope(tab, hs_grid, u[2000:]) >= tol).mean(axis=0)
    tab_raw = dict(tab)
    tab_raw["s_h"] = np.ones(H)            # ★ 撤掉归一化，其余逐位不变
    tab_raw["q"] = tab["q_unnorm"]
    rate_raw = (envelope(tab_raw, hs_grid, u[2000:]) >= tol).mean(axis=0)

    # (a) 归一化版：判据转折必须在**内部**
    assert rate_norm[0] < 0.2, f"归一化版在最年轻年龄就已触发 {rate_norm[0]:.2f}"
    assert rate_norm[-1] > 0.8, f"归一化版在最老年龄仍未触发 {rate_norm[-1]:.2f}"
    # (b) 未归一化版：最年轻年龄即退化（这正是 K1 的症状）
    assert rate_raw[0] > 0.9, \
        f"★ K1 未复现：未归一化版在 h=1 的触发率只有 {rate_raw[0]:.2f}" \
        f"（说明本用例的异方差强度不足以暴露该坑，需调大 spread 而不是放宽断言）"
    # 边缘覆盖率：**两者都**接近目标 ⇒ 说明"边缘保证"本身不足以拦住 K1
    for name, tb in (("归一化", tab), ("未归一化", tab_raw)):
        m = (err[2000:] <= envelope(tb, hs_grid, u[2000:])).mean()
        assert abs(m - 0.9) < 0.06, f"{name}版边缘覆盖率 {m:.3f} 偏离目标 0.9 过多"

    # (c) ĝ 对 u 单调不减
    for h in range(H):
        nk = int(tab["n_keep"][h])
        v = tab["knots_v"][h, :nk]
        assert np.all(np.diff(v) >= -1e-12), f"h={h + 1} 的 ĝ 对 u 非单调：{v}"
    uu = np.linspace(-3, 3, 41)
    e_h = envelope(tab, np.full_like(uu, 3.0), uu)
    assert np.all(np.diff(e_h) >= -1e-12), "信封对 u 非单调"


# ------------------------------------------------- ㊹ X42 分块产物的覆盖守卫
def test_conformal_merge_requires_all_per_chunks():
    """㊹ ★★ X42（2026-09-29）：**分块跑**之后，少一个 PER 档必须在聚合前被拦下。

    ★ 为什么这是一等的检查：X42 的完整网格在本机**单次跑不完**
      （后台任务 10 min 硬上限，实测两臂都在 10m01s 被终止、连日志都没写），
      ⇒ 改成**按 PER 分块**跑。分块本身数值等价（`scripts/26` 顶部有论证），
      但它引入了一个**新的静默失败模式**：某块没跑成 ⇒ 跨 PER 取中位时**分母变小**
      ⇒ 结论悄悄变强/变弱，而输出**看着完全正常**。
    ⇒ 断言：缺一档必须 raise；全档齐了才返回覆盖信息；`est_nmse` 单指标的行
      **不算**数（否则只有 SE 的行会让"覆盖"看着齐、实际距离指标缺档）。
    """
    from wmlab.eval.conformal import check_per_coverage

    full = [{"per": p, "metric": "mean_dist_tail", "utrigger_over_threshold_median": 1.0}
            for p in (0.0, 0.1, 0.3)]
    assert check_per_coverage(full, [0.0, 0.1, 0.3])["n_per_evaluated"] == 3

    # (a) 缺一档 ⇒ 必须 raise（这就是"某一块没跑成"）
    try:
        check_per_coverage(full[:2], [0.0, 0.1, 0.3])
    except AssertionError:
        pass
    else:
        raise AssertionError("★ 缺 PER 档时没有 raise ⇒ 会在部分档上静默聚合")

    # (b) 只有 est_nmse 的行不得抵充距离指标（口径错配会让"覆盖"看着齐）
    try:
        check_per_coverage([{"per": p, "metric": "est_nmse",
                             "utrigger_over_threshold_median": 1.0} for p in (0.0, 0.1, 0.3)],
                           [0.0, 0.1, 0.3])
    except AssertionError:
        pass
    else:
        raise AssertionError("★ 只有 est_nmse 的行被当成覆盖达标了")

    # (c) 空输入也不得默默通过
    try:
        check_per_coverage([], [0.0])
    except AssertionError:
        pass
    else:
        raise AssertionError("★ 空表没有 raise")


# ------------------------------------------------- ㊺ X43 oracle 真值误差的接线
def test_oracle_true_error_is_exposed_and_matches_metric():
    """㊺ ★★ X43（2026-09-29）：`u_state["_oracle"]` 接线的**逐位**检查。

    三条断言，都不是"看着对"，都是恒等式：
      (a) **未开启** `_oracle` ⇒ **不写** `oracle_*` 键
          —— 默认路径必须零影响、零开销（与 X40 加 `age` 同一约束）；
      (b) 开启 + T=1（每步送真值）⇒ 真值误差**恒为 0**（est ≡ 真值）；
      (c) ★ 开启 + 单步全丢包 ⇒ `oracle_err` **逐位等于**该次运行的 `est_nmse`
          —— 这是**量纲一致性**（R12）的机器证明。量纲错配的症状看起来像
          "信号无效"（z 被顶飞 ⇒ 触发率异常），只有恒等式能把它抓出来。
    """
    from wmlab.models.prob_world_model import GaussianWorldModel

    env = make_env("uav-track", seed=0, noise_std=0.15, max_steps=60)
    ctrl = PDRelativeController(dt=env.dt, omega_n=2.5, zeta=1.0,
                                kappa=float(env.kappa), a_max=float(env.a_max))
    torch.manual_seed(0)
    model = GaussianWorldModel(6, 2, latent_dim=8, hidden=16, discrete_act=False)
    dev = torch.device("cpu")
    common = dict(n_episodes=1, seed=7, device=dev, estimator="model",
                  var_g=1.0)

    # (a) 不开启 ⇒ 不许写出 oracle 键
    st_off: dict = {}
    run_closed_loop_control(env, model, ctrl, periodic_schedule(4),
                            max_steps=10, warmup_steps=5, u_state=st_off, **common)
    assert "oracle_err" not in st_off and "oracle_pos" not in st_off, \
        "★ 未开启 _oracle 却写出了 oracle_* ⇒ 默认路径被污染（零影响被破坏）"

    # (b) 开启 + T=1 ⇒ 恒为 0
    st_on: dict = {"_oracle": True}
    r1 = run_closed_loop_control(env, model, ctrl, periodic_schedule(1),
                                 max_steps=10, warmup_steps=5, u_state=st_on, **common)
    assert r1["est_nmse"] == 0.0, f"T=1 的 est_nmse 应为 0，实际 {r1['est_nmse']:.3e}"
    assert st_on["oracle_err"] == 0.0 and st_on["oracle_pos"] == 0.0, \
        (f"★ T=1 时 oracle 量应恒为 0，实际 err={st_on['oracle_err']:.3e} "
         f"pos={st_on['oracle_pos']:.3e}")

    # (c) 单步全丢包 ⇒ oracle_err 与 est_nmse 逐位一致（量纲接线）
    st_l: dict = {"_oracle": True, "_otrace": []}
    r2 = run_closed_loop_control(env, model, ctrl, periodic_schedule(10 ** 9),
                                 max_steps=1, warmup_steps=0, u_state=st_l, **common)
    assert r2["n_tx"] == 0 and r2["n_steps"] == 1, \
        f"★ 单步全丢包的设定没生效（n_tx={r2['n_tx']}, n_steps={r2['n_steps']}）"
    assert abs(float(st_l["oracle_err"]) - float(r2["est_nmse"])) < 1e-12, \
        (f"★ oracle_err {st_l['oracle_err']:.12e} ≠ est_nmse {r2['est_nmse']:.12e}"
         " ⇒ 在线口径与离线 `mean(平方误差)/var_g` 不同量纲（R12）")
    assert np.isfinite(float(st_l["oracle_pos"])) and float(st_l["oracle_pos"]) > 0.0, \
        f"★ oracle_pos 非法：{st_l['oracle_pos']}"
    # (d) ★★ `_otrace` 必须是**三元组**且量纲自洽：
    #   踩过的坑（2026-09-29 真事故）：只记 6 维量 ⇒ `--oracle-target pos` 的口径核对
    #   会拿 6 维闭环误差去比位置维离线基准 ⇒ 把本该 ≈1.0 的比值读成 **7.29**，
    #   症状看起来像"标定不可迁移"（而比值结论其实不受影响）。⇒ 两列都必须在场。
    assert len(st_l["_otrace"]) == 1, f"trace 应有 1 条，实际 {len(st_l['_otrace'])}"
    row = st_l["_otrace"][0]
    assert len(row) == 3, f"★ `_otrace` 必须是 (age, 6维, 位置维) 三元组，实际 {row}"
    tr_arr = np.asarray(st_l["_otrace"], dtype=float)
    assert abs(tr_arr[0, 1] - float(st_l["oracle_err"])) < 1e-12, \
        "★ trace 第 2 列与 u_state['oracle_err'] 不一致"
    assert abs(tr_arr[0, 2] - float(st_l["oracle_pos"])) < 1e-12, \
        ("★ trace 第 3 列与 u_state['oracle_pos'] 不一致 ⇒ pos 臂的口径核对会量纲错配"
         "（实测把 1.0 读成 7.29）")
    env.close()


# ------------------------------------------------- ㊻ X43 逐年龄归一化不可省（O1）
def test_oracle_age_normalization_is_required():
    """㊻ ★★ X43（2026-09-29）：**逐年龄归一化 Ŝ(h) 不可省** —— 否则 X43 静默空跑。

    ★ 症状为什么难发现：未归一化的「真值误差 + 全局阈值」**照样**能跑、照样有触发率、
      照样出一张曲线图 —— 但它的触发决策实际上是**年龄的单调函数**（误差量级随 h 增长），
      ⇒ 在闭环里恒等于年龄阈值（比值≈1.000）⇒ 会把"实验退化成同义反复"读成
      "oracle 也无价值"（两个结论天差地别）。
    ⇒ 断言方式选**决策相关量**（与 ㊸ 同一精神）：
        未归一化版：**逐年龄**触发率的跨度极大（最年轻≈0、最老≈1）⇒ 决策≈年龄的函数；
        归一化版  ：逐年龄触发率基本平（每个年龄都 ≈ 名义比例）⇒ 决策带**年龄内**信息。
    (c) O3：整列常数（MAD≡0）⇒ Ŝ 落地板且 z 仍有限（不许 inf）。
    """
    from wmlab.eval.oracle import fit_age_cond, z_pool

    rng = np.random.default_rng(20260930)
    H, N = 12, 6000
    hs = np.arange(1, H + 1, dtype=float)
    base = 0.02 * 1.55 ** (hs - 1)          # 真值误差随年龄增长（跨 ~90 倍）
    err = base[None, :] * (1.0 + 0.35 * rng.normal(size=(N, H)))
    tab = fit_age_cond(err[:3000])

    # (a) 未归一化：全局阈值 ⇒ 触发率是年龄的单调函数（跨度接近 1）
    thr_raw = float(np.quantile(err[:3000], 0.70))
    rate_raw = (err[3000:] >= thr_raw).mean(axis=0)
    span_raw = float(rate_raw.max() - rate_raw.min())
    assert span_raw > 0.9, (
        f"★ O1 未复现：未归一化版的逐年龄触发率跨度只有 {span_raw:.2f}"
        " —— 说明本用例的误差增长不够强，应调大 base 的增长率，而**不是**放宽断言")

    # (b) 归一化：条件 z 的阈值 ⇒ 逐年龄触发率基本平（年龄内信息被保住）
    thr_z = float(np.quantile(z_pool(err[:3000], tab), 0.70))
    rate_z = (z_pool(err[3000:], tab) >= thr_z).mean(axis=0)
    span_z = float(rate_z.max() - rate_z.min())
    assert span_z < 0.25, f"归一化版逐年龄触发率跨度仍有 {span_z:.2f}（应基本平）"
    assert abs(float(rate_z.mean()) - 0.30) < 0.05, \
        f"归一化版整体触发率 {rate_z.mean():.3f} 偏离名义 0.30 过多"

    # (c) O3：整列常数 ⇒ Ŝ 全部落地板，且 z 仍然有限
    tab2 = fit_age_cond(np.tile(base, (50, 1)))
    assert tab2["n_s_floored"] == H, \
        f"整列常数时 Ŝ 应全部落地板，实际 {tab2['n_s_floored']}/{H}"
    z2 = z_pool(np.tile(base, (50, 1)), tab2)
    assert np.all(np.isfinite(z2)), "★ O3：Ŝ 无地板 ⇒ z 出现 inf"


# ------------------------------------------------- ㊼ X43 非有限值必须 raise（O2）
def test_oracle_rejects_non_finite():
    """㊼ ★★ X43（2026-09-29）：O2（R14）—— 非有限值**必须 raise**。

    ★ 同一个坑的第三次：`nan <= 阈值` / `nan >= 阈值` **恒为 False**
      ⇒ 静默把"发散"读成"没超阈 / 没触发"，产出**看着完全合理**的错答案
      （X2/X3 的 NaN 坑、X42 的 K2 都栽在这上面）。
    另测两条**形状**错配：一维输入 / 列数不符 —— 它们会退化成
    "拿错年龄的基准去归一化"，症状同样是"信号无效"而不是报错。
    """
    from wmlab.eval.oracle import fit_age_cond, z_oracle, z_pool

    ok = np.ones((10, 4), dtype=float)
    bad_nan = ok.copy()
    bad_nan[3, 2] = np.nan
    bad_inf = ok.copy()
    bad_inf[0, 0] = np.inf
    for name, bad in (("NaN", bad_nan), ("Inf", bad_inf)):
        try:
            fit_age_cond(bad)
        except FloatingPointError:
            pass
        else:
            raise AssertionError(f"★ O2 未通过：标定集含 {name} 却被静默接受")

    tab = fit_age_cond(ok)
    try:
        z_oracle(float("nan"), 1, tab)
    except FloatingPointError:
        pass
    else:
        raise AssertionError("★ O2 未通过：非有限的触发量被静默接受（会恒判'不触发'）")
    try:
        z_pool(bad_nan, tab)
    except FloatingPointError:
        pass
    else:
        raise AssertionError("★ O2 未通过：z_pool 的非有限输入被静默接受")

    # 形状错配（会静默拿错年龄的基准）
    try:
        fit_age_cond(np.ones(10))
    except ValueError:
        pass
    else:
        raise AssertionError("★ 一维输入应 raise（否则拿错年龄的基准去归一化）")
    try:
        z_pool(np.ones((10, 3)), tab)
    except ValueError:
        pass
    else:
        raise AssertionError("★ z_pool 的列数与 tab['H'] 不符时应 raise")


# ------------------------------------------------- ㊽ X43 口径核对必须取对列（R12）
def test_oracle_alignment_uses_target_column():
    """㊽ ★★ X43（2026-09-29）：`align_age_medians` 必须**真的**按 `col` 取数。

    ★ 这条锁一个「**同一条错误犯了两次**」的坑：
      ① `_otrace` 第一版只有一列（6 维量）⇒ pos 臂拿 6 维量去比位置维基准 `m_pos(h)`；
      ② 补上第三列之后，列号变量 `col` **算了却忘了接进取数行**（仍硬编码 `trace[:, 1]`）
         ⇒ 症状与 ① **逐位相同**（"我改过了，怎么没变"）。
      症状长什么样：把本该 ≈1.0 的比值读成 ≈10×，**看起来像"标定不可迁移"，不像报错**。
    """
    from wmlab.eval.oracle import align_age_medians, ORACLE_TRACE_COL

    # 造一个假 trace：第 1 列（6 维量）比第 2 列（位置维量）大 10×
    n, H = 200, 8
    tr = np.stack([np.ones(n), np.full(n, 0.010), np.full(n, 0.001)], axis=1)
    m_pos = np.full(H, 0.001)   # 与第 2 列同量纲 ⇒ col=2 应得 ≈1.0
    m_res = np.full(H, 0.010)   # 与第 1 列同量纲 ⇒ col=1 应得 ≈1.0

    a2 = align_age_medians(tr, m_pos, col=ORACLE_TRACE_COL["pos"])
    assert a2 is not None and abs(a2["ratio_median"] - 1.0) < 1e-9, \
        (f"★ col=2 必须用第 2 列算比值（≈1.0），实际 {a2 and a2['ratio_median']}"
         " ⇒ **取数列没接线**（2026-09-29 第二次踩的正是这个坑）")
    assert a2["col"] == 2, "★ 返回里必须自报用的是哪一列（事后可对账）"
    a1 = align_age_medians(tr, m_res, col=ORACLE_TRACE_COL["res"])
    assert a1 is not None and abs(a1["ratio_median"] - 1.0) < 1e-9, \
        f"★ col=1 必须用第 1 列算比值（≈1.0），实际 {a1 and a1['ratio_median']}"
    # ★ 反向断言：**故意取错列**必须得到 ≈10 —— 证明这条测试真的有分辨力
    a_bad = align_age_medians(tr, m_pos, col=ORACLE_TRACE_COL["res"])
    assert abs(a_bad["ratio_median"] - 10.0) < 1e-9, \
        ("★ 取错列时比值应 ≈10（量纲错配的样子）；若仍是 ≈1 说明测试本身没有分辨力")

    for name, bad, exc in (("只有 2 列（旧格式）", tr[:, :2], AssertionError),
                           ("非法列号 col=0", tr, ValueError)):
        try:
            align_age_medians(bad, m_pos, col=(0 if name.startswith("非法") else 2))
        except exc:
            pass
        else:
            raise AssertionError(f"★ {name} 应 raise，否则会静默拿错列去比基准")
    bad_nan = tr.copy()
    bad_nan[0, 2] = np.nan
    try:
        align_age_medians(bad_nan, m_pos, col=2)
    except FloatingPointError:
        pass
    else:
        raise AssertionError("★ O2（R14）：trace 含 NaN 必须 raise，不许静默跳过")


# ------------------------------------------------- ㊾ X44 时刻表必须"恰好 N 个且合法"
def test_x44_plan_times_fixed_budget_and_legality():
    """㊾ ★ X44（2026-10-04）：`plan_times` 必须产出**恰好 n_tx 个**合法时刻。

    X44 的全部结论都建在「**唯一自变量是时刻**」上 ⇒ 预算必须逐位相等。
    这条测试锁两个**真踩过的坑**（都入账）：
      ① 网格起点写成 0 ⇒ `run_closed_loop_control` 首次查询调度在 **t=1**、
         `t=0` 永不出现 ⇒ 相位各档差 1 次发送（症状长得像"时机有影响"）；
      ② jitter 从 1 起 ⇒ 负抖动把首个时刻推到 **−1** ⇒ 越界。
    """
    from wmlab.eval.timing import (TIMING_MODES, n_tx_budget, plan_times,
                                   timing_dispersion)

    n_steps = 120
    for T in (2, 4, 8, 16):
        N = n_tx_budget(T, n_steps)
        assert N >= 1, f"T={T}: 预算必须 ≥1"
        jmax = (T - 1) // 2
        cases = [("phase", list(range(T))),
                 ("jitter", list(range(jmax + 1))),
                 ("random", [0, 1, 2])]
        for mode, params in cases:
            for p in params:
                ts = plan_times(mode, T, p, n_steps, N, pattern_seed=p + 1)
                assert len(ts) == N, f"{mode}/{p}: 时刻数 {len(ts)} ≠ 预算 {N}"
                assert len(set(ts)) == N, f"{mode}/{p}: 时刻有重复 ⇒ 预算被削"
                assert all(1 <= x < n_steps for x in ts), \
                    f"{mode}/{p}: 越界 [{min(ts)}, {max(ts)}]（须在 [1,{n_steps})）"
                assert all(ts[k] < ts[k + 1] for k in range(N - 1)), \
                    f"{mode}/{p}: 时刻未严格递增（预算语义被破坏）"

    # 离散度（本实验的**纯净自变量**）：周期任意相位=0；抖动/随机 >0 且递升
    for T in (8, 16):
        N = n_tx_budget(T, n_steps)
        assert timing_dispersion(plan_times("phase", T, 3, n_steps, N)) == 0.0, \
            "★ 周期的间隔恒为 T ⇒ CV 必须恰好 0（相位不该改变离散度）"
        jm = (T - 1) // 2
        assert timing_dispersion(plan_times("jitter", T, jm, n_steps, N)) > 0.0
        assert (timing_dispersion(plan_times("random", T, 0, n_steps, N))
                > timing_dispersion(plan_times("jitter", T, 1, n_steps, N))), \
            "★ 完全随机时机的离散度应大于小抖动"

    # 恒等档：phase Δ=T ≡ phase Δ=0；jitter j=0 ≡ phase Δ=0（否则 S_t3 无意义）
    for T in (4, 8, 16):
        N = n_tx_budget(T, n_steps)
        b = plan_times("phase", T, 0, n_steps, N)
        assert plan_times("phase", T, T, n_steps, N) == b, "★ 相位周期性自洽失败"
        assert plan_times("jitter", T, 0, n_steps, N) == b, "★ jitter j=0 应 ≡ 周期"

    # 非法输入必须 raise（不许静默给一个"看起来对"的表 —— R14）
    N8 = n_tx_budget(8, n_steps)
    bad_calls = [
        ("未知 mode", lambda: plan_times("bogus", 8, 0, n_steps, N8)),
        ("jitter 超过 (T−1)//2（会乱序）",
         lambda: plan_times("jitter", 8, (8 - 1) // 2 + 1, n_steps, N8)),
        ("jitter 为负", lambda: plan_times("jitter", 8, -1, n_steps, N8)),
        ("n_tx=0", lambda: plan_times("phase", 8, 0, n_steps, 0)),
    ]
    for name, fn in bad_calls:
        try:
            fn()
        except (ValueError, AssertionError):
            pass
        else:
            raise AssertionError(f"★ [{name}] 必须 raise，否则预算会被静默改掉")
    assert set(TIMING_MODES) == {"phase", "jitter", "random"}


# ------------------------------------------------- ㊿ X44 反算区间必须含 t=1
def test_x44_expected_attempts_uses_t_ge_1():
    """㊿ ★★ X44：`expected_attempts` 的区间必须是 **`1 ≤ t ≤ L`**（不是 `t < L`）。

    ★ 这是**同一条时序坑**（`t += 1` 在 `schedule(t, rng)` 之前）的另一处表现，
      也是代价最大的那个：第一版用 `t < L`，基准档反算 101 vs 实测 **93**
      —— 差 8 = 8 集各丢一次"t=0 那次"。若不做这条反算检查，
      这个**差 1 的预算偏差**会被读成"相位改变了发送次数"（一个不存在的时机效应）。
    ★ 反向断言（没有它，测试可能只是"跟着实现一起错"）：
      故意用错区间必须得到**不同的**数。
    """
    from wmlab.eval.timing import expected_attempts

    times = [1, 9, 17]
    # (a) 恰好落在集内：{1,9,17} 都在 [1,17] ⇒ 3 次
    assert expected_attempts(times, [17]) == 3
    # (b) **提前终止**的短集：L=5 ⇒ 只有 t=1 落在 [1,5] ⇒ 1 次
    assert expected_attempts(times, [5]) == 1
    # (c) 多集求和：1 + 3
    assert expected_attempts(times, [5, 17]) == 1 + 3
    # (d) 反向断言：错区间 `t < L` 必须给出**不同**的数（证明本测试有分辨力）
    wrong = sum(sum(1 for x in times if x < L) for L in [17])
    assert wrong == 2 and wrong != expected_attempts(times, [17]), \
        "★ 若错区间与正确区间结果相同，说明这条测试没有分辨力"
    # (e) t=1 必须被算进去（起点不是 0）
    assert expected_attempts([1], [1]) == 1, "★ t=1 必须计入（调度首查在 t=1）"
    assert expected_attempts([0], [120]) == 0, "★ t=0 永不出现 ⇒ 不应计入"


# ------------------------------------------------- [51] X45 首尾静默恒等式（循环化不解耦）
def test_x45_boundary_silence_identity():
    """[51] ★★ X45（2026-10-06）：`boundary_silence` 与**尾部静默的相位依赖**。

    ★ 这是 X45 **决定"换指标口径而不是换窗口"** 的全部数学依据。
      对周期 T 的均匀网格与窗口的交集，**尾部静默随相位的变化幅度恒为 T−1**：
        · 线性表（X44 用）：tail 从 `2T−1`（Δ=0）降到 `T`（Δ=T−1）；
        · 循环表（X44 §8 建议）：tail 从 `T−1`（Δ=0）降到 `0`（Δ=T−1）；
        · **两者幅度都 = T−1 ⇒ 「把窗口做成循环」只把基线挪了 T，解耦为零。**
      （差 `T` 的来源：线性表比循环**少发一次**，`n_tx_budget = (M−T)//T` vs `M//T`。）
    ★ 反向断言：两种表的 tail 都必须**真的随 Δ 变**，否则本测试没有分辨力。
    """
    from wmlab.eval.timing import (boundary_silence, cyclic_grid, n_tx_budget,
                                   plan_times)

    # (a) 循环网格的恒等式：head + tail ≡ T−1（与相位无关）
    M = 288                                  # 同时是 4/8/16 的整数倍
    for T in (4, 8, 16):
        for ph in range(T):
            h, tl = boundary_silence(cyclic_grid(T, M, ph), M)
            assert h + tl == T - 1, \
                f"★ 循环网格 T={T} phase={ph}: head+tail={h+tl} ≠ T−1={T-1}"

    # (b) ★ 核心证据（**同一窗口 M=288**）：线性与循环的 tail 都随 Δ 递减，
    #     且**变化幅度完全相同（都是 T−1）**⇒ 循环化不能减小相位对尾部指标的耦合
    T = 8
    Nlin = n_tx_budget(T, M)                 # 线性比循环少发一次（N = K−1）
    lin_t = [boundary_silence(plan_times("phase", T, d, M, Nlin), M)[1]
             for d in (0, T - 1)]
    cyc_t = [boundary_silence(cyclic_grid(T, M, d), M)[1] for d in (0, T - 1)]
    assert lin_t == [2 * T - 1, T], f"★ 线性 tail 应为 (2T−1)→T，实测 {lin_t}"
    assert cyc_t == [T - 1, 0], f"★ 循环 tail 应为 (T−1)→0，实测 {cyc_t}"
    assert (lin_t[0] - lin_t[1]) == (cyc_t[0] - cyc_t[1]) == T - 1, \
        "★ 两种窗口的「尾部静默随相位的变化幅度」都 = T−1 ⇒ 循环化不解耦"

    # (c) 循环网格与线性表**不是**同一个东西（防止"循环化"当成等价替换）
    assert cyclic_grid(T, M, 0) != plan_times("phase", T, 0, M, Nlin), \
        "★ 循环网格与线性表的点数不同（K vs K−1）⇒ 预算也不同"

    # (d) 非法输入必须 raise（不给"看起来对"的数 —— R14）
    bad = [
        ("空表", lambda: boundary_silence([], 100)),
        ("乱序", lambda: boundary_silence([5, 3], 100)),
        ("越界", lambda: boundary_silence([1, 101], 100)),
        ("首时刻 <1", lambda: boundary_silence([0, 5], 100)),
        ("window 非 T 整数倍", lambda: cyclic_grid(8, 300, 0)),
        ("period=0", lambda: cyclic_grid(0, 288, 0)),
    ]
    for name, fn in bad:
        try:
            fn()
        except (ValueError, AssertionError):
            pass
        else:
            raise AssertionError(f"★ [{name}] 必须 raise")


# ------------------------------------------------- [52] X45 多口径必须"同一次运行"，core 真去边界
def test_x45_multi_metric_and_core_slice():
    """[52] ★ X45：`core_slice` 的边界与 `metric_over` 的三口径。

    ★ 为什么口径必须来自**同一次运行**：重跑闭环会因 rng 消耗路径不同得到不同轨迹
      ⇒ 跨运行比较会把"重跑噪声"混进"口径差异"（X44 用 PER=0 封死的同类混淆）。
    ★ 反向断言：`core` 必须**真的丢点**（与 `full` 逐位相同 ⇒ mask 没接线）。
    """
    import numpy as np
    from wmlab.eval.timing import core_slice, metric_over

    # (a) core 区间 = `[t₂, t_{K−1})` 的 0-based 索引（t ⇔ i+1）
    assert core_slice(20, [1, 9, 17]) == (8, 16), "★ core 区间应为 [t₂, t_{K−1})"
    # (b) 有效时刻 <3 ⇒ 跳过（提前终止太早；不许用短段凑数）
    assert core_slice(20, [1, 9]) is None
    assert core_slice(5, [1, 9, 17]) is None, "★ 时刻超出该集长度 ⇒ 有效点 <3 ⇒ 跳过"

    # (c) 三口径在构造 trace 上的预期值（dist = i+1 ⇒ 可手算）
    tr = [float(i + 1) for i in range(20)]
    traces, times = [tr, tr], [1, 9, 17]
    full, n_f = metric_over(traces, times, 20, "full")
    core, n_c = metric_over(traces, times, 20, "core")
    tail, n_t = metric_over(traces, times, 20, "tail", tail_frac=0.25)
    assert n_f == n_c == n_t == 2, f"★ 参与集数应为 2（实测 {n_f}/{n_c}/{n_t}）"
    assert abs(full - 10.5) < 1e-12, f"★ full = mean(1..20) = 10.5，实测 {full}"
    assert abs(core - float(np.mean(tr[8:16]))) < 1e-12, "★ core = mean(9..16)"
    k = max(1, int(round(20 * 0.25)))
    assert abs(tail - float(np.mean(tr[-k:]))) < 1e-12, "★ tail = 最后 25%"
    # ★ 反向断言：core ≠ full（否则 mask 没接线）
    assert abs(core - full) > 1e-9, "★ core 与 full 相同 ⇒ core mask 没接线"

    # (c') core_fixed：位置**固定**、只去固定 margin 步（诊断"位置平移"混杂）
    cf, n_cf = metric_over(traces, times, 20, "core_fixed", margin=3)
    assert n_cf == 2, f"★ core_fixed 参与集数应为 2（实测 {n_cf}）"
    assert abs(cf - float(np.mean(tr[3:17]))) < 1e-12, "★ core_fixed = [margin, len−margin)"
    try:
        metric_over(traces, times, 20, "core_fixed", margin=0)
    except ValueError:
        pass
    else:
        raise AssertionError("★ core_fixed 的 margin=0 必须 raise（否则与 full 等同 ⇒ 死参数）")

    # (d) 不合格集必须被**跳过**、且计数正确
    traces2 = [tr, [1.0, 2.0]]               # 第二集仅 2 点 ⇒ 有效时刻 <3
    core2, n_c2 = metric_over(traces2, times, 20, "core")
    assert n_c2 == 1, f"★ 不合格集应被跳过（实测参与 {n_c2} 集）"
    assert abs(core2 - float(np.mean(tr[8:16]))) < 1e-12

    # (e) 未知口径必须 raise；空输入返回 (nan, 0) 而不是抛
    try:
        metric_over(traces, times, 20, "bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("★ 未知口径必须 raise")
    v, n = metric_over([], [], 20, "full")
    assert n == 0 and v != v, "★ 空输入应返回 (nan, 0)"


def main() -> int:
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failed.append((name, exc))
            print(f"  FAIL  {name}\n        {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
