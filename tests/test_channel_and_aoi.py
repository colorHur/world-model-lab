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
"""

from __future__ import annotations

import copy
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wmlab.data import collect_random_episodes
from wmlab.envs import (ChannelAdapter, StepResult, make_env, make_env_with_channel)
from wmlab.eval import (NmseCurve, expected_nmse_analytic, geometric_age_pmf,
                        geometric_age_stats, jensen_gap, loss_prob_for_tail_risk,
                        mse, nmse_at_mean_age, per_step_mse, periodic_schedule,
                        reliable_horizon, run_tracking, uniform_age_pmf,
                        uniform_age_stats)
from wmlab.eval.tracking import StateTracker
from wmlab.models import MLPWorldModel
from wmlab.rollout import closed_loop_error_curve, multi_step_error_curve


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
