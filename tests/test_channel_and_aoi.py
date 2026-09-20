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
"""

from __future__ import annotations

import copy
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wmlab.data import collect_random_episodes, transitions_from_episodes
from wmlab.envs import (ChannelAdapter, StepResult, make_env, make_env_with_channel)
from wmlab.eval import (NmseCurve, expected_nmse_analytic, geometric_age_pmf,
                        geometric_age_stats, jensen_gap, loss_prob_for_tail_risk,
                        lossy_schedule, mse, nmse_at_mean_age, per_step_mse,
                        periodic_schedule, reliable_horizon, run_tracking,
                        uniform_age_pmf, uniform_age_stats)
from wmlab.eval.tracking import StateTracker
from wmlab.models import MLPWorldModel
from wmlab.rollout import closed_loop_error_curve, multi_step_error_curve
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


# ----------------------------------------------------------- runner
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
