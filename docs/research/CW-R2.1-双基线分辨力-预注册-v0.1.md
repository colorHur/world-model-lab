# CW-R2.1 预注册：零训练双基线分辨力（`floor_R2` 先测）

- 写卡时间：**2026-10-25 03:41:54 +08:00**（`Get-Date` 实测，Asia/Shanghai）
- 协议：**CW-POMDP-v1.1**（未改）
- 上游：`CW-R2.0-R2立项与接口卡-预注册-v0.1.md`（`R2-CARD-READY`，HEAD `55254a0`）
- 本轮性质：**零训练**。只测"面板能不能把强/弱基线分开"，不检验 `H-R2-1`（小学习模型 ≤5%）。

---

## §0 已见 / 未见披露（写卡前）

**已见（且只到这一层）**

1. R2.0 的卡与证据：指纹 `d3904923ce2c8d82`、视图 20 键、47 条权限声明、`r2_models.py` 的三个参考件源码。
2. R1 线的面板数字与判词（R1.14–R1.18），以及 `plan_ref` / `plan_dp` / `world` / `runner` / `config` 的源码。
3. R1.15 / R1.16 入口脚本的结构（`run_cell` / `_Rec` / 常量 `RHO=0.98, H_PLAN=5, M_MAIN=16, RATIO=0.5, TAIL_DP={"tail":"dp","dp_grid":"r173"}`）。
4. R2.0 的预算三点：`4.4875 / 4.15 / 3.1225` 秒/格。

**未见（本轮要测的）**

- R2.1 的**任何**格：没有跑过 `belief_readout` 或 `prior_only` 驱动的任何 episode，没有看过两者的 `cost_true_per_step`、动作序列、逐用户成本。
- 不知道两个模型的动作会不会相同，也不知道 `floor_R2` 的量级。

**一条代码阅读发现（本轮顺手记录，不作为结论）**

- `plan_ref.py` 的模块 docstring 第 11 行写"只吃 `prior` / `telemetry` / `history` / `budget_left` / `t`"，
  但全文件对 `history` 的实际读取次数 = **0**（`grep "\.history"` 只命中该 docstring 自身）。
  ⇒ **文档与实现不一致**。本轮不依赖 `history`，不因此改任何代码，只记在这里。

---

## §1 问题与本轮的假设（不是 `H-R2-1`）

**R2.1 的科学问题（新编号 `H-R2-0`）**

> 在 R1.14 同款规划器、同预算、同信息集下，把任务状态信念从"接收端后验"（`belief_readout`）
> 换成"只报公开先验"（`prior_only`），`cost_true_per_step` 的差**是否大于**同模型只换规划器
> 采样种子的零控地板 `floor_R2`？

- **为什么必须问**：`H-R2-1` 要问"小学习模型够不够"，而这个面板连"强基线 vs 弱基线"能不能分开都没测过。
  在 `floor_R2` 没测出之前训练，等于重犯 R1.5 的坑（效应被规划器/采样方差淹没）。
- **否定条件（预先写死）**：`median|Δ| ≤ floor_R2` ⇒ `CANNOT-SEPARATE-BASELINES`
  ⇒ **不进训练**，改扩 episode 或换指标。这是**允许的结论**，不是失败。
- **不许推出的结论**：
  ① 不许写"学习模型够/不够"（本轮无学习模型）；
  ② 不许写"世界模型无用"（`prior_only` 只是下界参照，不是对照方法）；
  ③ 不许把 `prior_only` 的差当作"信息无价值"的证据。

---

## §2 面板设计（写死，不看数改）

### §2.1 两个 plant 档（唯一新增的自变量）

| 档 | 构造 | 物理含义 |
|---|---|---|
| `P1` | `bounded_plant()`（`sigma_meas=0.20`） | R1 全部面板的基线 |
| `P2` | `bounded_plant(sigma_meas=0.05)` | **任务观测更准**（同类传感器的不同精度） |

**为什么只动 `sigma_meas`**：它只进两处——送达观测的噪声（`world.py:519`）与卡尔曼增益用的 `noise_var`
（`world.py:574`）。它**不改**动力学 `A/B/σ_w`、不改 `q/r`、不改信道、不改预算、不改协议。
它是"送达观测值多少钱"的直接物理旋钮 ⇒ 强/弱基线差距应当随它变化（这正是要测的"面板有没有分辨力"）。
**不改** `A`（R1.9 已证策略层无下标依赖，改 `A` 不是本轮的旋钮）。

★ 两档**分档报告，不合并中位**。合并会把"哪一档有分辨力"抹掉（R1.16 教训：混合口径）。

### §2.2 模型（零参数、零训练）

| 臂 | 实现 | 吃什么 | 不吃什么 |
|---|---|---|---|
| `belief_readout`（强） | `r2_models.BeliefReadout` | `telemetry.x_hat/P_hat` | 不吃任何真值/未来 |
| `prior_only`（弱） | `r2_models.PriorOnly` | `prior.x_mean/x_var` | 不吃观测/ACK/自己的动作 |

两个模型都只输出 `(x̂, P)`。**信道信念 `h_mean/h_var` 不由模型负责**（仍由世界的 `GridBelief` 维护）
⇒ 本轮只换**任务状态模型**这一维，信道侧完全不变。这是单变量对照的必要条件。

### §2.3 模型如何进入决策（唯一实现 `model_policy.py`）

`ModelPolicy(model, planner)`：
1. `out = model.predict(view)`（模型只看**原始**视图）；
2. `v2 = rebind_view(view, out.x_hat, out.P)`：用 `SchedulerView` 的**公开属性**重建一个视图，
   只覆盖 `telemetry` 的 `x_hat/P_hat`，其余字段原样；
3. `return planner(v2)`。

★ `rebind_view` 必须仍通过 `model_api.require_view`（类型恒等 + 键集相等）——
不通过就等于"模型层成了绕过守卫的口子"。

### §2.4 格与种子

| 项 | 值 | 说明 |
|---|---|---|
| plant | `P1`, `P2` | |
| 模型 | `belief_readout`, `prior_only` | |
| episode | `100..109`（10 个） | **独立单元 = episode**（R1.7.8 D5b：同初值的 seed 不是独立单元） |
| 面板规划器种子 | `2001`（两模型相同） | 严格 CRN：外生子流只由 `(root_seed, split, ep)` 决定，与规划器种子无关 |
| 零控种子对 | `(2001, 3102)` | 只用于 `floor_R2`，模型固定 `belief_readout` |
| 其他 | `rho=0.98`, `H_PLAN=5`, `M=16`, `RATIO=0.5`（`B=128`）, `tail="dp"`, `dp_grid="r173"` | 与 R1.14/R1.15/R1.16 **同款** |

**格数**：主面板 `2 plant × 2 model × 10 ep = 40` 格；零控 `2 plant × 1 model × 10 ep × 2 seed = 40` 格。
合计 **80 格**。

---

## §3 指标与判据（看数前钉死）

### §3.1 主量

```
Δ(P, ep, s) = cost_true_per_step(prior_only, P, ep, s) − cost_true_per_step(belief_readout, P, ep, s)
```

★ **符号先钉死**：`Δ > 0` ⇒ **弱基线更贵** ⇒ 强基线更好。任何"改善"措辞必须与这个方向一致
（R2.0 修过的"比值方向"类错误不再犯）。

`n_pos` = 20 个 `(P, ep)` 单元中 `Δ > 0` 的个数。

### §3.2 零控地板

```
null_pair(P, ep) = | cost(belief_readout, P, ep, plan_seed=2001) − cost(belief_readout, P, ep, plan_seed=3102) |
floor_R2         = median over 20 个 (P, ep) 的 null_pair
```

**为什么需要它**：R1.5/R1.7.8 已证规划器内部采样本身有可观方差；不先量出这个地板，
任何"两模型不同"的结论都可能只是采样噪声。`floor_R2` 是**本面板自算**的零控，
不借任何历史阈值（R2.0 §1-4 的口径）。

### §3.3 判词（四类，穷尽且互斥）

| 条件 | 判词 |
|---|---|
| `median_{20}|Δ| > floor_R2` 且 `n_pos ≥ 18` | `PANEL-SEPARATES` |
| `median_{20}|Δ| > floor_R2` 且 `15 ≤ n_pos ≤ 17` | `SEPARATES-MAGNITUDE-ONLY` |
| `median_{20}|Δ| > floor_R2` 且 `n_pos ≤ 14` | `CANNOT-SEPARATE-BY-SIGN` |
| `median_{20}|Δ| ≤ floor_R2` | `CANNOT-SEPARATE-BASELINES` |

二项单侧 `p = P(X ≥ n_pos | n=20, p=1/2)`，用**整数组合和**精确算（不引 scipy，不近似）。
`n_pos ≥ 18` ⇔ `p = 2.01e-4`；`≥15` ⇔ `p = 2.07e-2`；`= 17` ⇔ `p = 2.01e-3`。

★ 判词**只看 §3.1/§3.2 这两个量**，不看任何事后想到的量。
★ `median|Δ|` 与 `floor_R2` 都报，并同时报它们各自的 IQR（描述用，不进判据）。

### §3.4 必报（R17 + 反掩盖）

1. **逐用户** `r_i = cost_true_by_user(prior_only) / cost_true_by_user(belief_readout)`，逐 `(P, ep)`；
   任一 `r_i > 1` ⇒ 必须写"**交易**"，不许只报总量中位。
2. 逐用户 `n_delivered / n_attempted`、`n_probe` 总数。
3. 逐用户时间平均 `aoi_tavg_by_user`（**不用**旧 `mean_task_aoi` 末端单帧口径）。
4. `action_diff_frac`：两模型在同一 `(P, ep)` 上动作序列的差异比例（描述量，**不作判据**）。
5. 预算实测（逐格耗时、DP 构建次数、峰值内存）。

### §3.5 不做 / 不报

- 不报 `ku / kf / share*`（`OBJECTIVE.forbidden`）。
- 不训练、不扩 episode、不改 `plan_ref` / `plan_dp` / `world` / `runner` / `config` / `policies`。
- 不重跑旧 P3、不动旧 `results/`。
- 不为"让判词好看"调 `sigma_meas` 或 `M`/`H`。

---

## §4 闸（每条都写清"什么错误能让它 FAIL"）

| id | 检查 | 什么错误能让它 FAIL |
|---|---|---|
| `B1` | `model_policy.py` 的 AST 扫描干净（复用 `model_api.scan_source`） | 模型层里出现真值名/`_pub`/越界 import ⇒ 变成非法通道 |
| `B2` | `rebind_view` 守恒：顶层键集 = 20；`t/n_users/budget_left/budget_total/prior/history` 逐位相同；`telemetry` 除 `x_hat/P_hat` 外逐键相同 | 重建时漏字段/串字段/改了预算 ⇒ planner 看到的世界不是同一个 |
| `B2b` | 原视图**不被污染**：`rebind` 前后 `view.telemetry` 的 `x_hat/P_hat` 逐位相同 | 直接改原 dict ⇒ 世界状态被模型写回（信息边界破） |
| `B2c` | **活体负控**：把 `rebind` 的 `budget_left` 故意 +1 ⇒ `B2` 必须 FAIL（`finally` 还原） | 闸没有分辨力（"跑过了"≠"能 FAIL"） |
| `B3` | **直通恒等**：`ModelPolicy(BeliefReadout)` 与裸 `_Rec` 在 `(P1, ep100, s2001)` 上 `policy_trace_hash` 与 `cost_true_per_step` 逐位相同 | 包装层引入扰动（比如 rebind 把 float 降精度、或漏传 `prior`）⇒ 面板量的不是模型效应 |
| `B3b` | **活体负控**：`B3` 的对照物换成 `ModelPolicy(PriorOnly)`，必须**不**恒等 | `B3` 的"恒等"是恒真（两边都空跑） |
| `B4` | 接线分辨力（构造）：同一真实视图上 `prior_only` 与 `belief_readout` 的 `(x̂,P̂)` 至少 1 个用户不同 | 模型没被真正调用 / 输出没进 rebind |
| `B5` | 特权臂 `TruthDiagnostic` 被 `deploy()` 拒绝（`PermissionError`） | 诊断臂混进部署表 |
| `B6` | `null_pair` 口径自检（合成 cost）：同 seed ⇒ 0；已知两值 `(3.0, 1.0)` ⇒ 2.0 | 公式写反/取错绝对值 |
| `B7` | 逐用户成本恒等：逐格 `|Σ_i cost_true_by_user − cost_true_per_step| ≤ 1e-9` | 逐用户量定义漂了（R1.8 的教训：聚合口径先查分母） |
| `B8` | 全量 `gates.run_all()` `n_fail == 0` | 协议回归被本轮改动破坏 |

**CTRL 先行**：`B1/B2/B2b/B2c/B4/B5/B6` 全部零 episode，先在 `--ctrl-only` 跑；
不过就不看 `B3`/`B7`/面板（按 §3 不看数的纪律）。

---

## §5 预算（先估，跑后填实测）

- 单格：R2.0 三点实测 `3.1225 / 4.15 / 4.4875` 秒/格 ⇒ 80 格 = **249.8–359.0 s**。
- 另加：`B3` 需 2 格、`B3b` 需 1 格（同 `(P1, ep100)`）⇒ 实际 **83 格**。
- DP 构建：R1.9 实测单表 **21.0 s / 2048.8 MiB**；R1.16 实测 80 格只建 **1 次**（缓存命中）。
- 预计总耗时 **260–390 s**，峰值内存 ≈ 2.1 GB。**本机 CPU 跑得起**。
- ★ 纪律：先跑 `--probe`（1–2 格计时）再外推；**单点计时不外推**（#98 已复发两次）。

---

## §6 产物（全部新增，不覆盖旧证据）

1. `experiments/cw_v1/model_policy.py`（模型→视图重绑定的唯一实现）
2. `experiments/cw_v1/r2_1_sep.py`（CTRL + 闸 + 面板 + 判词）
3. `docs/research/CW-R2.1-双基线分辨力-预注册-v0.1.md`（本卡）
4. `docs/CW-R2.1-双基线分辨力-交付.md`（实测数字 + 自伤）
5. `results/cw_v1/cw_r2_1_sep_<UTC>_<src_hash>/`：`cells.json` / `null.json` / `gates.json` / `ctrl.json` / `summary.json` / `meta.json` / `_POST_COMMIT.json`

---

## §7 事后修订区（跑后填，写清"当时已看见什么"）

### §7-0 写卡时已存在、跑后必须裁决的两处（**不是事后修改规则**）

1. **§2.1 与 §3.3 冲突**：§2.1 写"两档分档报告，不合并中位"，§3.3 却用 20 个单元（含合并）定判词。
   裁决：**主判取分档**（更严，且写在面板设计节），`ALL` 降为描述。
2. **阈值未按 n 缩放**：`N_POS_SEP=18` / `N_POS_WEAK_LO=15` 是对 n=20 定的，
   套到 n=10 分档时 `≥15` 算术不可能 ⇒ 分档判词结构性偏向 `CANNOT-SEPARATE-BY-SIGN`。
   裁决：**规则不改**（改了就成事后挑选），但必须同时报按-n 的精确二项 p，
   并在交付里写明"结论碰巧一致、推导是错的"。

### §7-1 跑后修订 / 披露（按时间序）

| # | 时间（实测） | 当时已看见什么 | 改了什么 |
|---|---|---|---|
| 1 | 03:45 首跑 | 只看见 `B1` FAIL 的 detail（`imports: hashlib`），**未见任何面板数** | 改**被测物**：`model_policy.py` 去掉 `hashlib`，`readout_digest()` → `readout_signature()`；**白名单不动**；首跑目录挂 `_SUPERSEDED.md` |
| 2 | 03:44 `--probe` | 看见 `est_61_cells_s=1526.5` | **未据此改设计**；事后记为该外推错 7.7×（实测 197.5 s） |
| 3 | 03:48 面板结束 | 看见判词与 `rows` | 补两项**零成本**事后口径：① 逐对胜出 `\|Δ\|>floor_pair`（14/20）；② 按 `median(floor_pair)` 分层的逐对胜出（低层 9/10 / 高层 5/10）。**两者均标"事后、不当主判"** |
| 4 | 03:49 | 看见 `P1/belief_readout` 与 R1.16 `exact` 同格的 cost 相同 | 追加**跨轮身份对账**入口 `r2_1b_identity.py`（零 episode，只读），结果 10/10 逐位相同；`P2` 10 格 `SKIP-NO-REFERENCE` |
| 5 | 03:50 | 比对 R2.0 的 `spec.json` 与本轮 `planner_signature()` | 发现 R2.0 指纹**不覆盖入口显式传参**（`dp_grid="r173"` vs 模块默认 `legacy`）⇒ 补记 `planner_actual.json`，**指纹不改**（改则 R2.0 证据失效，另立 spec-v2） |

### §7-2 未做的（如实）

- 未统计 DP 构建次数（`plan_dp.N_BUILDS` 未接进入口），只能由逐格耗时**反推**为 2 次。
- 未测 `P2` 档的跨轮身份（无同配置历史证据）。
- 未检验 `H-R2-1`（本轮零训练，按设计）。

