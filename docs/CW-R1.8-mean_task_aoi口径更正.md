# 更正：`mean_task_aoi` 的口径（旧字段不改，只加指针）

- 时间：2026-10-24 01:3x（`date` 实测，Asia/Shanghai）
- 性质：**解释更正**，不是数据更正。旧字段、旧证据、旧数字**一字未改**。

## 1. 旧字段到底是什么

`experiments/cw_v1/runner.py`（R1.7.8 及以前）第 172 / 189 行：

```python
aoi_vals = [x["task_aoi"] for x in w.view().telemetry]     # ← 末端**单帧**快照
mean_task_aoi = float(np.mean([a for a in aoi_vals if a >= 0])) if any(a >= 0 for a in aoi_vals) else -1.0
```

两个机械事实（`G0` 用合成数据逐条证明，见 `experiments/cw_v1/r1_8_mech.py::g0_synthetic_semantics`）：

- **D1 末端单帧**：只在 `T` 时刻取一次 `telemetry` ⇒ 每格只有 **2 个整数**（两个用户），
  256 个时隙的时间信息全部丢弃。同一臂跨 seed 从 3.5 跳到 18.5 **不是效应**，是采样点太稀。
- **D2 剔除未服务用户**：`if a >= 0` 把 `stamp==-1`（从未送达）的用户**排除在分母外**
  ⇒ 用户越被饿死，它越不进指标。这是 R17「不能用总平均隐藏受损用户」的**机制级**版本。

## 2. 哪些旧数字受影响、怎么读

| 旧引用 | 旧读法（**保留**） | 新读法（**加注**） |
|---|---|---|
| `paced` 11.75 / `dp` 31.25（2.66×，R1.7.8 §7.1-5 "唯一 ≥2×"） | 两个数字本身**逐位可复现**（R1.8 重跑值完全相同） | 它是**末端单帧两用户均值**；按 R1.8 逐用户时间平均，`dp` 对 u0 **更好**（11.28→5.22）、对 u1 **更差**（27.95→49.33）⇒ **方向信息被平均掉**，"dp 更差"是错的整体表述 |
| `G8a2`「同一格 `mean_task_aoi ≤ 20`」 | 该闸的历史过/不过记录有效 | 该闸测的是**末端快照**，与"配速"无因果关系；R1.8 起由 `n_never_served` + `aoi_max_by_user` 承担同一构念 |
| `plan_gates.py:441` 的 `"aoi": r.mean_task_aoi` | 诊断读数，保留 | 同上；不得用它与 `cv` 直接互证（两者测的不是同一构念，见 R1.8 交付 §7） |

## 3. 新增口径（`EpisodeResult` 追加字段，纯记录）

| 字段 | 定义 | 用途 |
|---|---|---|
| `aoi_tavg_by_user` | 逐槽 `task_aoi` 的**时间均值**（只统计 `stamp>=0`） | R1.8 起的主量 |
| `aoi_final_by_user` | 末端快照（旧 `mean_task_aoi` 的两个分量） | 与旧证据对账（`G2`：`abs_diff ≤1e-12`） |
| `aoi_max_by_user` | 期内最大陈旧度 | 饥饿守卫 |
| `n_delivered_by_user` / `n_attempted_by_user` | 由 `SlotRecord` 逐用户计数 | R17 逐用户必报 |
| `n_never_served` | 全期未送达用户数 | 守卫（旧量会把它删掉） |

★ 纯净性证据：R1.8 重跑的 **16/16 格**与 R1.7.6 入库值**逐位相同**
（`cost_est/cost_true/n_delivered/n_attempted/n_probe/policy_trace_hash/mean_task_aoi`），
见 `results/cw_v1/cw_r1_8_mech_20261023T170326_08b6bfcbb8c7dec7/identity_all16.json`。

## 4. 引用规则（自本轮起）

1. 引用 `mean_task_aoi` 必须同时写明"**末端单帧、两用户均值、剔除未服务用户**"。
2. 任何"任务侧改善/变差"的表述**必须**给 `aoi_tavg_by_user` 的逐用户值；
   只给 `mean_task_aoi` 的句子一律视为口径不明。
3. 不得因为 D1/D2 而否定旧证据 —— 旧数字是对的，**解释**是错的。
