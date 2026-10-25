# CW-R2.3 预注册：把面板从 n=20 扩到 n=30（`ep110–114`）

- **写卡时刻**：2026-10-25 **08:08**（`Asia/Shanghai`；按文件 `mtime` 校正 — 初稿手写 "08:1x"，
  `ls --time-style` 实为 `08:08:02`；**写卡时 `run_episode` = 0**）
- **上游**：`STATE.md` §7.2（R2.2 版"下一轮唯一主件"）、v4 §R2、v4.1 修订队列
- **协议**：`CW-POMDP-v1.1`（**不改**）；研究基线 `config.mainline_config()`（`BOUNDED_PLANT` + `ack_mode="exact"`）
- **被测物**：`plan_ref` / `plan_dp` / `world` / `runner` / `config` / `policies` / `mech2.py` / `r2_1_sep.py`
  **一字节不动**。本轮只新增入口脚本 + 本卡 + 交付文档。

---

## §0 已见清单（写卡前我已经看见什么）

★ 按 R1.18 §7.5-1：**"已见"必须现场读文件**，不许从 STATE 摘要或记忆外推。以下数字均于本轮
2026-10-25 08:0x 从 `results/cw_v1/cw_r2_2_criteria_20261024T215859_2d54069c2b4d767e/summary.json`
与 `cw_r2_1_sep_20261024T194859_3c71623d6afe07e4/` 读出：

| 块 | `n_units` | `n_pos` | `n_neg` | `n_beat` | `p_sign` | `p_beat` | `median|Δ|` | `median floor` |
|---|---|---|---|---|---|---|---|---|
| `ALL`（`ep100–109` × 2 plant） | 20 | 17 | 3 | **14** | 1.2884e-3 | **5.7659e-2** | 0.045631 | 0.022158 |
| `P1`（`sigma_meas=0.20`） | 10 | 8 | 2 | 8 | 5.4688e-2 | 5.4688e-2 | 0.070676 | 0.022158 |
| `P2`（`sigma_meas=0.05`） | 10 | 9 | 1 | 6 | 1.0742e-2 | 3.7695e-1 | 0.040086 | 0.024459 |

R2.2 判词：`ALL = SIGN-ONLY-NO-MAGNITUDE`、`P1 = CANNOT-SEPARATE`、`P2 = SIGN-ONLY-NO-MAGNITUDE`。
⇒ 扩样的动机是**`ALL` 的幅度腿差 1 格**（14/20，`k_need=15`）。

★★ **因此本轮不是"独立确认"**：扩样这件事本身由"14/20 差一格"触发。§2 会量化这个选择的代价。

---

## §1 问题

**Q（一句话）**：把独立单元从 20 加到 30（新增 `ep110–114` × 2 plant = 10 个**全新**单元）之后，
"强基线 `belief_readout` 的逐单位绝对成本改善**压过该单位自己的 CRF 噪声地板**"这条
（R2.2 的 `beat` 腿）能不能在 `alpha=0.05` 上被判成立？

- 若成立 ⇒ 面板**有幅度分辨力** ⇒ R2 才允许谈训练（`H-R2-1`）。
- 若不成立 ⇒ 按 STATE §7.2-5 转主指标 `mean_task_aoi`，**不再加 episode**。

★ 本卡**不**回答"学习模型够不够"（`H-R2-1` 仍未检验；本轮零训练）。

---

## §2 ★ 写卡阶段的**规则修正**（跑数前；理由与算式）

`STATE` §7.2-3 原写：「主判只看 `beat` 腿（`k_need=20/30`）」。**这条规则在统计上不成立**，
本卡在跑任何新格之前改掉它，并把代价写在下面。

### §2.1 为什么合并 `n=30` 的名义 `alpha` 是假的

扩样是**在看到 14/20 之后**决定的（选择性续做 / optional stopping）。把 20 个复用单元与
10 个新单元合并做一次 `Binomial(30, 1/2)` 检验，其**条件化**的 I 类错误率不是 0.05：

```
合并判过  ⇔  14 + X ≥ 20  ⇔  X ≥ 6 ，X ~ Binomial(10, 1/2)（H0 下）
P(X ≥ 6) = (1024 − 638) / 1024 = 386/1024 = 0.3770
```

⇒ **合并规则的实效 I 类错误率 ≈ 0.377（名义 0.05 的 7.5 倍）**。
`mech2` 的精确二项本身没错；错的是"把审查过我一次的数据再拿去算 `p`"。

### §2.2 改成两条腿（写死，跑前定）

| 腿 | 单位 | `k_need` (`alpha=0.05`) | 实测 `p`（代验过） | 用途 |
|---|---|---|---|---|
| **`FRESH`（主判）** | `ep110–114` × 2 plant = **10** | **9** | 11/1024 = **0.010742** | **唯一决策依据**（这 10 格此前从未跑过） |
| `POOLED`（描述） | `ep100–114` = 30 | 20 | 386/1024… 名义 0.049369 | 只报数，**不进决策**；必须同时报 §2.1 的 0.3770 |

代验（写卡前算，`python -c` 实测）：`n=20` ⇒ `k_need=15`（`p=0.020695`，与 R2.2 的 14/20
`p=0.057659` 互补，一致）；`n=30` ⇒ `k_need=20`（`p=0.049369`，`k=19` 则 `0.100244`）；
`n=10` ⇒ `k_need=9`（`p=0.010742`，`k=8` 则 `0.054688`）。

### §2.3 这条修正的**代价**（必须写进交付，不许只报好处）

`FRESH` 腿在"真实 beat 率 = 观测到的 0.70"下，功效只有
`P(X≥9 | n=10, p=0.70) = 0.1493`（≈15%）。
⇒ **`FRESH` 不显著时不能推"效应不存在"**，只能说"这个面板在 10 个新单元上分辨不出幅度"。
这与 R2.2 的 `SIGN-ONLY-NO-MAGNITUDE` 并不矛盾：它本来就是"方向可分、幅度压不过每格地板"。

★ 因此 §5 的判词映射里，`FRESH` 不显著一律写成 `FRESH-MAGNITUDE-UNRESOLVED`，
**禁止**写成 `NO-EFFECT` / `BASELINES-EQUAL` / `WORLD-MODEL-USELESS`。

---

## §3 面板（写死）

| 项 | 值 | 与 R2.1/R2.2 的关系 |
|---|---|---|
| plant | `P1 = bounded_plant()`（`sigma_meas=0.20`）、`P2 = bounded_plant(sigma_meas=0.05)` | 同 |
| models | `belief_readout`（强）、`prior_only`（弱） | 同 |
| episodes | **`ep100–114`（15 个）**，其中 `ep100–109` 为复用、`ep110–114` 为**全新** | 扩 5 个 |
| `split` | `"dev"` | 同 |
| 规划器 | `_RecAct(h_plan=5, m_samples=16, rho=0.98, pacing=True, tail="dp", dp_grid="r173")` | 同 `TAIL_DP` |
| `plan_seed` | 面板 `2001`、零控 `3102`（**严格 CRN**，两模型共用同一面板种子） | 同 |
| `ratio` / 预算 | `0.5` ⇒ `budget = protocol.budget(0.5)` | 同 |
| 格数 | 面板 `2 × 2 × 15 = 60`，零控 `2 × 15 = 30`，合计 **90** | 40+20 → 60+30 |

★ 不新增 plant、不改 `sigma_meas`、不改 `rho`、不改 `h_plan`/`m_samples`、不换网格、不换尾部。

---

## §4 判据（**一个常量都不许改**）

判据实现 = `experiments/cw_v1/mech2.py`（R2.2 的唯一实现），本轮**只调用、不修改**。
`ALPHA = 0.05`；`beat_u = 1[ |Δ_u| > floor_u ]`；`Δ_u = cost_true(weak) − cost_true(strong)`；
`floor_u = | cost_true(strong, seed=2001) − cost_true(strong, seed=3102) |`；单位 = `(plant, ep)` 配对。

| 腿 | 定义 | 阈值 |
|---|---|---|
| `sign` | `Δ_u > 0` 的计数 | 精确二项 `p_sign = P(X≥n_pos|n,1/2) < 0.05` |
| `neg` | `Δ_u < 0` 的计数 | 同上（**反向必须有自己的判词**，不许折进 `CANNOT`） |
| `beat` | `|Δ_u| > floor_u` 的计数 | 同上 |

★ **不分层**：R2.2 `D3` 已实测，按 `floor` 分层后层内 `beat` 的 `H0` 通过率为 0.7416
（假阳性 24.9%）⇒ 分层 `beat` **不合法**。本轮的 `beat` 只在**全样本**上算。
本轮**仍报**分层结果，但只报 `LOW` 的 `sign` 腿（那是 R2.2 已经判过合法的部分）。

---

## §5 判词映射（**写死，跑前定**）

### §5.1 主判（`FRESH`，`n=10`）

```
n_beat_fresh ≥ 9 且 n_pos_fresh ≥ 9   ⇒ FRESH-MAGNITUDE-SEPARATES      （→ 才谈训练）
n_beat_fresh ≥ 9 且 n_neg_fresh ≥ 9   ⇒ FRESH-SEPARATES-REVERSED       （→ 不进训练，另立卡）
n_beat_fresh < 9 且 n_pos_fresh ≥ 9   ⇒ FRESH-SIGN-ONLY-NO-MAGNITUDE   （→ 转 mean_task_aoi）
n_beat_fresh < 9 且 n_neg_fresh ≥ 9   ⇒ FRESH-SIGN-ONLY-REVERSED       （→ 转 mean_task_aoi 且必须报反向）
其它                                  ⇒ FRESH-MAGNITUDE-UNRESOLVED     （→ 转 mean_task_aoi）
```

★ 判词里的 `n_beat_fresh < 9` **不等于**"无效应"（§2.3 已写功效 ≈15%）。

### §5.2 描述（`POOLED`，`n=30`）

照 `mech2.verdicts_v2` 原样输出 `ALL/P1/P2` 三块，**并强制附一行**：

> 名义 `alpha=0.05` 无效：扩样由 14/20 触发，条件化后实效 I 类错误率 ≈ 0.3770（§2.1）。

### §5.3 子集对账（真实性闸，硬）

用**同一份判据**重算 `ep100–109` 的 20 个单位，必须**逐字复现** R2.2 三判词
（`ALL=SIGN-ONLY-NO-MAGNITUDE`、`P1=CANNOT-SEPARATE`、`P2=SIGN-ONLY-NO-MAGNITUDE`）
且 `n_pos/n_beat/median_abs_delta/median_floor_pair` 与上表**容差 0**。
不复现 ⇒ `C9` FAIL ⇒ 不出任何判词。

### §5.4 次级指标（**描述，不进判词**；跑前登记以防"事后挑指标"）

`mean_task_aoi_u = mean_i aoi_tavg_by_user[i]`（逐用户时间平均，R1.8 已证该量**不是**
"末端单帧 + 删分母"口径）。`Δaoi_u = aoi(weak) − aoi(strong)`，
`floor_aoi_u = |aoi(strong,2001) − aoi(strong,3102)|`。
只报 `n_pos/n_neg/n_beat` 与中位量，**不出判词、不作决策依据**，用于下一轮换指标时的量级预估。

---

## §6 自证与闸（顺序写死）

| 序 | 件 | 判据 | 不过怎么办 |
|---|---|---|---|
| 1 | `CTRL C1–C8` | 复用 `r2_1_sep.controls()`（零 episode，8 项） | 停，不看闸与面板 |
| 2 | `C9` 子集对账 | §5.3，`ep100–109` 20 单位逐字复现 | 停（判据/落盘有问题） |
| 3 | `GATE B1–B8` | 复用 `r2_1_sep.gates()`（11 项，含 `B3` 直通恒等 + 活体负控） | 停 |
| 4 | **身份闸 `I1`** | `ep100–109` 的 40 格 × 5 字段（`cost_true_per_step`/`cost_est_per_step`/`policy_trace_hash`/`n_attempted`/`n_delivered`）与 R2.1 `cells.json` **逐位相同**；20 零控同 | 停（面板与世界不一致） |
| 5 | `R1` 回归 | `gates.run_all()` **84 项 / 0 fail** | 停 |
| 6 | 面板 | 90 格 | — |

★ 身份闸是本轮唯一能证明"复用单元与 R2.1 是同一个世界"的东西，`I1` 不过 ⇒ 20 个复用单元作废、
`POOLED` 腿也作废（只剩 10 个 `FRESH` 单元）。

---

## §7 预算（先跑再报，禁止 `--probe` 单点外推）

- 参考：R2.0 的 3 个**实测**点 `3.1225 / 4.15 / 4.4875` s/格 ⇒ 90 格 ≈ **281–404 s**。
- R2.1 实测峰值内存 ≈ **4.15 GB**（本机 31.8 GB，够）。
- **若实测 > 20 min ⇒ 停并查因，不许"跑完再解释"**。
- 纯 CPU；无收费资源。

---

## §8 否定条件（写死）

1. `I1` 身份闸不过 ⇒ `POOLED` 作废，只报 `FRESH`（且 `POOLED` 标 `INCOMPLETE-PANEL`）。
2. `FRESH` 的 `n_beat < 9` ⇒ 判 `FRESH-MAGNITUDE-UNRESOLVED` 或 `FRESH-SIGN-ONLY-*`，
   **转 `mean_task_aoi`，不再加 episode**（STATE §7.2-5 原文）。
3. 若出现 `n_neg_fresh ≥ 9` ⇒ 必须报反向判词，**不得**写成"分不开"。
4. 若 `ALL` 的 `sign` 腿在合并集上翻不显著（违反 R2.2 的 17/20）⇒ 说明新 5 个 episode 与旧批
   分布不同，**必须停下写清**，不许硬拼。

---

## §9 事后修订区

（**跑数后**才填；每条必须写明"当时已经看见什么"）

### §9.1 入口脚本：证据目录提前 + 判词段包 `try`

- **改动**：`outdir` 从"分析之后"移到"面板之前"；`cells.json`/`null.json` 在面板一结束就落盘；
  判词段整体 `try/except` 记 `analysis_error`。
- **当时已见**：首跑（`--tag main`）在 `secondary_aoi` 抛
  `TypeError: float() argument must be ... not 'NoneType'`，**崩在 `outdir.mkdir` 之前** ⇒ 90 格读数
  只剩日志（日志里没有逐用户量）⇒ 必须重跑。
- **为什么这不是"改判据"**：`mech2.py`、`PLANT_KEYS`、`EPS`、seeds、判词映射**一个字符未动**；
  改的只是"证据什么时候落盘"与"异常怎么记"。**判据常量与判词映射保持 §4/§5 原文。**

### §9.2 次级指标 `mean_task_aoi` 的"无定义"口径（**跑数前**确定，登记在此）

- **改动**：`aoi_tavg_by_user[i] is None`（该用户全程未送达，`runner.py:278`）⇒ **整个单位无定义**，
  计数并报伴随量 `n_never_served`，**不**对剩下的用户取均值。
- **当时已见**：合成自检（拿 R2.1 落盘复制成 15 个 ep、**未跑任何新 episode**）里
  `P1/ep101` 出现了 `None` ⇒ 提前知道真实数据里会有这个分支。
- **理由**：只对剩余用户取均值就是 R1.8 判死的"删未送达用户"偏差。

### §9.3 事后确认（不改判据）

- `k_need` 代验在**跑数前**完成：`n=10 → 9`（`p=0.010742`）、`n=20 → 15`、`n=30 → 20`（`p=0.049369`）；
  合并规则在"已见 14/20"下的条件化 I 类错误率 `P(X≥6|n=10,1/2)=0.376953` —— 与 §2.1 写死值一致。
- 面板跑完后 `C9` 逐字复现 R2.2、`I1` 60/60 逐位相同 ⇒ **§4 的判据确实没被改动**（这是"改没改"的机器证据）。

