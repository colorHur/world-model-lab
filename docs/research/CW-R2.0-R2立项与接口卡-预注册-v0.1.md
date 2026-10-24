# CW-R2.0 预注册 · R2 立项与接口卡（v0.1）

> **写卡时刻**：2026-10-25 01:33（`Get-Date` 实测 +08:00 = Asia/Shanghai）
> **HEAD**：`43c19ec0c7a294b5a14f3eefcfc6d669af1842dd`（远端 `refs/heads/main` 同值，工作树 0 行）
> **锁**：`auto:CW-R2.0-立项与接口卡`，ttl 90 min，本轮私有 token 不入库
> **本轮性质**：立项卡 + 接口实现 + 闸。**不训练、不跑 episode、不改被测物**。

---

## §0 本轮范围与**已见清单**（先声明，不许事后补）

### 0.1 本轮要交什么

| # | 件 | 判据 |
|---|---|---|
| 1 | 本卡（先写） | 写完才动代码 |
| 2 | `experiments/cw_v1/model_api.py`：`predict(view) -> (x̂, P)` 接口 + 信息权限表 + 可比性指纹 | `A1–A7` 全过 |
| 3 | `experiments/cw_v1/r2_0_card.py`：入口 + 闸 + 证据落盘 | `n_fail = 0` |
| 4 | `docs/CW-R2.0-R2立项与接口卡-交付.md` | 含自伤清单 |

### 0.2 本轮**明确不做**

- 不训练任何模型（不写训练循环、不装新依赖、不建数据集）
- 不跑 episode（`run_episode` 调用次数必须 = **0**；`CWWorld` 只许**构造**，不许 `step`）
- 不改 `plan_ref` / `plan_dp` / `world` / `runner` / `config` / `policies` / `mech.py`
- 不重跑旧 P3、不动旧 `results/`
- 不改任何阈值去凑 PASS（`KAPPA_MIN` / `SIGN_MIN` / `MU_MIN` / `DR_DESC` / `SHARE_MIN` / `DR_MIN`）

### 0.3 已见清单（**现场读过文件**，不是从记忆外推）

| 已见 | 文件 | 具体数字 |
|---|---|---|
| 预算点 1 | `results/cw_v1/cw_r1_16_n10_20261024T105416_fc9f25e16d2c6307/meta.json` | 80 格、`panel_elapsed_s` 245.1、`total_elapsed_s` 249.8、峰值 2096.4 MB |
| 预算点 2 | `results/cw_v1/cw_r1_14_exactbase_20261024T063157_c835f84ff417e9d2/meta.json` | 16 格、`total_elapsed_s` 66.4、峰值 2096.2 MB |
| 预算点 3 | `results/cw_v1/cw_r1_12_ackscale_…/meta.json`（`--write` 时现读，下表 A8 用） | 待闸内现读 |
| 配速开关残差 | `results/cw_v1/cw_r1_7_4_tails_20261023T130519_…/bench.json` | `G8a`：`cv` paced **0.583391467891342** vs nopace **0.9286036223814766**；`cost_true` **0.11067633547679674** vs **0.1767897716156801**；`aoi` **16.0** vs **128.5**（同 `ep100`、同策略、只切 `pacing`） |
| 尾部实现残差（部分） | 同目录 `panel_tails.json` | `paced` 臂 8 集 `cost_median` = **0.07871360448861517**、`elapsed_s` 204.64。**另两臂 `cost_median` 本轮尚未读**，由闸内现读并原样记录 |
| R1.5 「配速压倒效应」 | STATE §7.2（二手摘要） | 四格比值 1.374/2.024/1.017/1.892。**本轮不引用它做阈值**（见 §3.2） |

★ 未读、本轮也不会去挑的数：R1.6.1 / R1.7.5 / R1.7.6 的面板明细。

---

## §1 三问

**Q1** 「同规划器 + 同预算」到底由哪几个字段构成？能不能被**一个指纹**钉住，使得"换了任何一项"都自动暴露？
**Q2** 学习模型的接口与信息权限能不能被**机器**钉死（而不是靠自觉），并且这些闸**有分辨力**（能 FAIL）？
**Q3** R2 面板在本机 CPU 上跑得起吗？—— 用**已测的多个点**给区间，不用单点外推。

---

## §2 判据（**跑数前**写死；每条都配"什么错误能让它 FAIL"）

| 闸 | 断言 | 什么错误会让它 FAIL |
|---|---|---|
| `A1` | `require_view` 只接受 `type(x) is SchedulerView` 且 `_pub` 键集 == 声明表；鸭子类型替身、dict、子类替身**全部抛 `TypeError`** | 守卫退回 `isinstance`/仅查属性 ⇒ 替身混过 |
| `A1b` | **活体负控**：把守卫换成"只查 `.t` 属性存在"的实现，同一个替身**必须**被接受 | 若负控也拒 ⇒ A1 没有分辨力，A1 判 SKIP |
| `A2` | `ModelOutput.validate` 对 ①长度不符 ②`NaN` ③`P ≤ 0` ④非数值 ⑤二维输入 全部抛错；合法输入不抛 | 校验只查 `len` ⇒ ③④⑤ 漏 |
| `A3` | 部署路径模块的 AST 里**不出现** `TRUTH_FIELDS` ∪ `{_x_true,_log_h_true,truth_override,_w_last,_z_meas,_z_csi,_u_packet}` ∪ `_pub` 直接访问 | 有人在模型里读真值/未来噪声 |
| `A3b` | **负控**：把 `cost_true_by_user` 塞进一段合成源码 ⇒ 扫描器必须报；把 `x_hat` 塞进去 ⇒ 必须不报 | 正则匹配过宽（报所有下划线名）或过窄（漏真值名） |
| `A4` | 权限表**完备**：真实 `view()` 的每个键（含 `prior`/`telemetry`/`history` 子键）都在表里被分类；`TRUTH_FIELDS` 全部标 `never`；深度扫描 `view` 的 JSON 化结果里**不含**任何真值字段名 | 视图加字段 ⇒ 表不覆盖；或真值漏进视图 |
| `A5` | 声明 `privileged=True` 的模型经 `deploy()` **抛 `PermissionError`**；经 `deploy_diagnostic()` 返回且 `label` 前缀 `DIAG-` | 特权臂混进部署表 |
| `A6a` | 同 spec ⇒ 指纹**逐字相同**（跑两次） | 指纹里混了时间/内存地址 |
| `A6b` | 8 个字段**逐一**改动 ⇒ 指纹**逐一**变化（8/8） | 指纹漏字段（"同预算"没进指纹） |
| `A6c` | `planner` 字段的值取自 `inspect.signature(PlanRef.__init__)` 的默认值（不是硬编码）；`h_plan` 默认必须 = 5 | 有人改了 `PlanRef` 默认而指纹不变 |
| `A7` | 目标函数表：主量 = 绝对任务代价、辅量必须含**逐用户 `r_i`**、禁用表必须含 `ku/kf/share`；且主/辅量里**不出现**禁用名 | 把 `ku` 当目标函数（R1.18 明令禁止） |
| `A8` | 预算：≥2 个**实测**点、给出 min/median/max 秒/格、R2 面板预算给**区间**、区间跨度 ≥1.2×（不许单点冒充精度） | 只用一个点外推 |
| `A9` | 配速残差锚点：从 `bench.json` **现读** `cv`/`cost_true`/`aoi` 两组值，逐字节与本卡 §0.3 一致（`< 1e-12`） | 本卡数字手抄错/记忆外推 |
| `A10` | 协议闸回归：`gates.run_all()` 84 项 `n_fail = 0` | 接口实现碰坏了信息边界 |
| `CTRL` | 判据自证（零 episode、已知答案）**先过**，不过不看 `A1–A10` 结果 | 判据自身错 |

### 2.1 判词规则（写死）

- `A1–A10` **全过** ⇒ `R2-CARD-READY`。
- 任一不过 ⇒ 先判"闸错还是物错"（不许为过闸改被测物），修完重跑；修不掉则如实记 `R2-CARD-BLOCKED:<闸号>`。

---

## §3 可比性定义（Q1 的答案草案，**先写后验**）

### 3.1 「同规划器同预算」= 下列 8 组字段的指纹

1. `plant`（`A, B, sigma_w, prior_var, sigma_meas, q_cost, r_cost, n_users`）
2. `channel`（`rho_panel, kappa, log_h_star, sigma_csi, h_log_init_std, ack_mode`）
3. `protocol`（`horizon, c_probe, c_data, init_from_prior, event_pairing, allow_multislot`）
4. `planner`（`h_plan, m_samples, use_probe, pacing, cont_m, tail, dp_grid, send_other_advance`）—— **取自签名默认值**
5. `budget_total`（绝对帧数，不是比例）
6. `split` + `episodes` + `seeds`（CRN：同一 `(split, episode)` 即同一外生实现）
7. `models`（本次比较的对象名，排序后）
8. `metric`（主/辅量名 + 逐用户口径）

★ **为什么 `budget_total` 写绝对帧数**：`budget_ratio` 与 `horizon` 联乘，改 `horizon` 会静默改预算。
★ **为什么 `planner` 取签名默认值**：硬编码 5 会在有人改默认时静默撒谎（`A6c`）。

### 3.2 残差锚点：**不用历史阈值，用本轮可自算的零控**

上一轮 STATE §7.2 写"'配速残差 ≥ 效应 ⇒ INCONCLUSIVE'，其依据是 R1.5 的四格比值"。
**本轮降级这条依据**：R1.5 比的是"1 步贪心 vs 周期发送"，那是**策略差**，不是"配速残差"。
替代定义（写进 R2 卡，R2.1 才计算）：

```
null_pair(k) = |cost_true(model_A, planner_seed=s) − cost_true(model_A, planner_seed=s')|
```
同模型、同 episode、同预算，只改 `PlanRef(seed=…)`（规划器内部采样的种子）。
`floor_R2 = median_k null_pair(k)`。
**判定**：模型效应 `|Δ| < floor_R2` ⇒ `INCONCLUSIVE`（不是"无效应"）。

已测的两条**量级**参考（只作预期，不作阈值）：
`pacing` True/False ⇒ `cost_true` 0.1107 → 0.1768（**1.60×**）；
三套尾部实现 ⇒ `cost_median` 见 `panel_tails.json`（闸内现读）。

---

## §4 指标（**先定义，后跑**）

- **主量**：`cost_true_per_step` 的**中位数**（绝对量，`q·Σx² + r·Σu²`，无量纲化由公开 `q/r` 固定）
- **必报（R17）**：逐用户 `r_i = cost_true_by_user_i(model) / cost_true_by_user_i(baseline)`；任一 `>1` ⇒ 写"交易"不写"改进"
- **必报**：`mean_task_aoi` 且必须**逐用户**、时间平均（旧 `mean_task_aoi` 末端单帧且剔除未送达用户 ⇒ 只用其历史复现值，不当主量）
- **禁用为目标函数**：`ku`、`kf`、`share`（R1.18：它们答"多有害/解释多少"，不是优化目标）
- **统计单位**：完整 episode；同初值的 seed **不是**独立单元（R1.7.8 `D5b`）

---

## §5 首个可否定假设（写进卡，R2.1 才检验）

**H-R2-1**：同规划器、同预算、同信息集下，一个**小学习模型**（参数 < 10⁴、CPU 训练 < 30 min）
用于状态/信道预测时，其 `cost_true_per_step` 中位数与"精确名义模型"（R1.4 已证一步精确）
的差 **< 5%**，且逐用户 `r_i` 全部落在 `[0.95, 1.05]`。

- **否证条件**：配对差的中位 ≥ `floor_R2`（§3.2）**且** 二项符号检验 `p < 0.05`（要求 ≥9/10 同号）
- **INCONCLUSIVE 条件**：`|Δ| < floor_R2` ⇒ 面板信噪比不足，**不许**写"学习模型够用"或"不够用"
- **不许的解释**：若学习模型输了，**不得**推出"世界模型无用"（R1.14 的教训：协议 + plant 限定了结论范围）

---

## §6 预算（Q3）

- 已测 3 点（§0.3 + `A8` 现读）⇒ 秒/格取 **区间**，不外推单点（#98 已复发两次）
- R2.1 计划面板：2 模型 × 2 plant 档 × 10 episode = 40 格；预算给 `40 × [min, max]`
- 内存：已测峰值 ≈ 2.1 GB（DP 尾部单表 2048.8 MiB 是已知上限，R1.9）⇒ R2 若用 `tail="paced"` 则 ≪ 此值
- **本轮实测**：`run_episode = 0`，`world.step = 0`，`CWWorld` 构造 1 次（仅取 `view()` 键集）

---

## §7 不做 / 不许

- 不许为了过 `A4` 去删视图字段或改 `world.py`（**被测物不许动**）
- 不许把 `A3` 写成"扫全部 `*.py`"（那只会在历史脚本上大面积报，且不指向"模型是否读真值"）
- 不许把 `A8` 的区间换成中位数单点
- 不许在本轮引用 `ku/kf` 的任何数值当 R2 的依据（R1 线已收口，只作历史）
- 不许声称"R2 已立项完成"以外的进度：本轮**没有**训练、**没有**新 episode

---

## §8 事后修订区（跑后追加，不删原文）

### 8.1 `A1b` 首版断言**不可达**（判据自伤，改判据不改被测物）

原文写"同一个替身必须被接受"，实现时扩大成"`dict` 与子类替身**都要**被属性守卫接受"。
但属性守卫看的是 `hasattr(o, "t")`，而 **dict 没有属性** ⇒ 这条断言算术上不可能成立，首跑 FAIL。
改成：只有子类替身参与判定，`dict` 只作记录（并写明"任何属性守卫都拒 dict，不参与判定"）。
—— 与 R1.17 #4（`|share|>1e3` 不可达）同族。

### 8.2 资产清单判据对象选错（字符串命中 ≠ 依赖）

首版 `ok = not any("ChannelAdapter" in s for s in srcs.values())`。
`streams.py:4` 的 docstring 里**正写着**这个词（解释"为什么不用 `salt+7919k`"）⇒
把注释误判成依赖。改为 AST 级：无 `ClassDef ChannelAdapter`、无 import。

### 8.3 跑前修掉的代码自伤

`scan_source` 首版用 `zip(skip, ["TruthDiagnostic"])` 拼 `skipped` 字段 —— 两个列表长度/顺序不一致，
多类跳过时错配。改为并行收集 `skipped_names`。

### 8.4 未改动的部分（不允许改的都没改）

`A1–A10` 的阈值、`OBJECTIVE` 的禁用表、`FIELD_PERMISSION` 的 47 条路径、`CARD_PINNED` 的 7 个数字
全部**跑前写死、跑后未动**。被测物 `plan_ref/plan_dp/world/runner/config/policies/mech` 一字节未动。
