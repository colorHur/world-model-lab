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
                                 periodic_age_pmf, periodic_age_tail,
                                 periodic_lossy_schedule,
                                 periodic_mean_age, run_length_goodness, simulate_bad_runs)
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
    """㉜ ★★ R12：闭环控制里的 `payload_fn`（量化器）**真的接进去了**（X35 的 T4）。

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
