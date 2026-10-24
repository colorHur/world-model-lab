# CW-R2.0 交付 · R2 立项与接口卡

> **HEAD（开工）**：`43c19ec0c7a294b5a14f3eefcfc6d669af1842dd`（远端同值、工作树 0 行）
> **时间**：2026-10-25 01:30:47 取锁（`Get-Date` 实测 +08:00）→ 01:39 收工
> **协议**：CW-POMDP-v1.1（**未改**）；研究基线 = `mainline_config()`（`BOUNDED` + `ack_mode="exact"`）
> **判词**：**`R2-CARD-READY`**

---

## §0 开工前：队列前四项纠错的**只读复核**（不重做）

任务书 §2 要求"先核最新 HEAD 是否已修，优先第一轮"。四项逐条现场查文件（不改一行）：

| # | 项 | 现场证据 | 结论 |
|---|---|---|---|
| 1 | candidate/baseline 比值方向、配对差、胜率 | `evaluate.py:20-27` 写死 `ratio=candidate/baseline`（`<1` 更好）、`delta=baseline−candidate`（`>0` 更好），判词只用 `median_pairwise_ratio`+胜率+`median_delta` 符号一致性，`ratio_of_medians` 降为参考 | **已修** |
| 2 | `OracleKnown` 撤回"上界" | `oracle_known.py:8-25` 明写"撤回'上界臂'这个名称与它承载的推论"，并给出反证（同格给更多真信息反而差 4.26×） | **已修** |
| 3 | 封 `view._inner` / 白名单只读 | `world.py:808-868`：`SchedulerView` 无 `_inner`，`__slots__=("_pub",)`，`history` 走 `PUBLIC_HISTORY_FIELDS` 白名单，未知属性 `__getattr__` 抛 `AttributeError`；`_inner` 只在 3 个闸里作为**禁用名**出现 | **已修** |
| 4 | 初态与公开先验一致 / 稳态 sd=1 而非 5.03 | `config.py:97` 明确记录"旧写的 5.03 是把未归一 AR 的公式当作归一后的"；`world.view()` 的 `prior` 就是采样分布参数 | **已修** |

⇒ 四项均已在 v1.1 落地，本轮**不重做**，直接进 `CW-R2.0`（STATE §7.2 唯一主件）。

---

## §1 三问的答案

### Q1 「同规划器 + 同预算」= **一个指纹**

`ComparabilitySpec`（`experiments/cw_v1/model_api.py`）把 11 个字段拼成 `sha256[:16]`：

| 组 | 字段 |
|---|---|
| plant | `A, B, sigma_w, prior_var, sigma_meas, q_cost, r_cost, n_users` |
| channel | `rho_panel, kappa, log_h_star, sigma_csi, h_log_init_std, ack_mode` |
| protocol | `horizon, c_probe, c_data, init_from_prior, event_pairing, allow_multislot` |
| planner | `h_plan, m_samples, use_probe, pacing, cont_m, tail, send_other_advance, dp_axis_mode` + **模块级** `TAIL_DEFAULT` / `GRID_DEFAULT` |
| 采样 | `budget_total`（**绝对帧数**）、`split`、`episodes`、`seeds` |
| 对象/指标 | `models`（排序）、`metric` |
| 版本 | `protocol_version` |

- **确定性**：同进程两次、**跨进程两次**，指纹全部 = `d3904923ce2c8d82`。
- **分辨力**：**11 个字段逐一**改动 ⇒ 11/11 指纹变化（`A6b`）；无一遗漏。
- **不硬编码**：`planner` 的值取自 `inspect.signature(PlanRef.__init__)` 的**运行时默认值**，
  且额外记录 `tail=None` 解析后的模块级默认（`paced`）与 `GRID_DEFAULT`（`legacy`）⇒
  有人改 `PlanRef` 默认值，指纹**必然**变（`A6c`，`h_plan` 实测 = 5）。
- **为什么 `budget_total` 写绝对帧数**：`budget_ratio × horizon` 会因改 `horizon` 静默改预算。

### Q2 接口与信息权限被**机器**钉死，且闸有分辨力

| 边界 | 实现 | 活体负控（证明能 FAIL） |
|---|---|---|
| 模型只吃 `SchedulerView` | `require_view`：`type(x) is SchedulerView` **且** `_pub` 键集恰等于声明表 | 放宽成"只查 `.t` 属性"后，子类替身 `_Impostor` **被接受**（`A1b`）⇒ 收紧不是为了好看 |
| 输出形状/有限性/正定性 | `ModelOutput.validate`（长度、NaN、`P≤0`、二维、非数值） | 5 个反例全抛（`A2`） |
| 不许读真值/真信道/**未来噪声** | `r2_models.py` 的 AST 扫描：禁用名 `TRUTH_FIELDS ∪ {x_true,log_h_true,truth_override,_x_true,_log_h_true,_w_last,_z_meas,_z_csi,_u_packet}`、禁用属性 `_pub`、import 白名单 | 合成脏源码 4 例（字符串通道 / `_z_meas` / `from .world import CWWorld` / `import experiments...`）**全被抓**，干净源码 0 命中（`A3b`） |
| 视图里真的没有真值 | 真实 `view()` 的 20 个键双向核对 47 条权限路径；`never` 类 14 条全覆盖 | `truth_in_view` / `undeclared` / `missing_top` / `forbidden_uncovered` 四个集合必须全空（`A4`） |
| 特权臂进不了部署表 | `deploy()` 对 `privileged=True` 抛 `PermissionError`；诊断走 `deploy_diagnostic()`，label 前缀 `DIAG-` | `TruthDiagnostic` 实测被拒；`BeliefReadout` 通过；诊断输出 `privileged=True`（`A5`） |
| `ku/kf/share` 不得当目标函数 | `OBJECTIVE` 表 + 自检器 | 把 `primary` 换成 `ku` / `share` ⇒ 立刻违规（`A7` + `C5`） |

### Q3 预算够（**多点区间**，不是单点外推）

| 证据 | 格数 | 总耗时 | 秒/格 |
|---|---|---|---|
| R1.12（`…T020453_92d6a8cc…`） | 16 | 71.8 s | **4.4875** |
| R1.14（`…T063157_c835f84f…`） | 16 | 66.4 s | **4.1500** |
| R1.16（`…T105416_fc9f25e1…`） | 80 | 249.8 s | **3.1225** |

跨度 **1.437×**（≥1.2× 门槛）⇒ R2.1 计划面板 **40 格**（2 模型 × 2 plant 档 × 10 episode）
预算 **124.9–179.5 s**、峰值内存 ≈ 2.1 GB（若用 `tail="paced"` 则远低于 DP 单表的 2048.8 MiB）。
**结论：本机 CPU 跑得起**，不需要 GPU。

---

## §2 本轮最有信息量的新事实（**不是**我自己预期的那个）

我原以为"换尾部值函数实现"是又一条大残差（R1.7.4 的 `cv` 在三条尾部上差 2.8×）。现读 `panel_tails.json` 的 `cost_median`：

| 尾部 | `cost_median`（同 8 集） | 相对 `paced` |
|---|---|---|
| `paced`（解析配速） | 0.07871360448861517 | 1.000 |
| `dp_legacy` | 0.07638889056773905 | 0.970 |
| `dp_r173` | 0.0762433859018477 | 0.969 |

⇒ **尾部实现只挪 3.1% 成本（1.032×），而 `pacing` 开关挪 60%（1.60×）**。

两条同时成立、且**不能合并成一个"配速残差"**：
- `cv`（发送间隔变异系数）对尾部极敏感（0.59 → 0.21，2.8×），但那是个**形状量**；
- `cost_median` 对它几乎不敏感（3.1%）。

所以 R2 的零控**不能**用"换尾部"冒充：必须用 `PlanRef(seed=…)` 的 `null_pair`（见 `NULL_PAIR_SPEC`），
它把"规划器内部采样随机性"单独隔离出来，且**不涉及任何策略/参数改动**。

---

## §3 本轮交付

| 件 | 位置 | 状态 |
|---|---|---|
| 预注册（**先写**，`run_episode`=0） | `docs/research/CW-R2.0-R2立项与接口卡-预注册-v0.1.md` | 已写（§8 记 2 条跑后修订） |
| 接口 + 权限表 + 指纹 | `experiments/cw_v1/model_api.py` | 新增 |
| 参考模型（强/弱/诊断） | `experiments/cw_v1/r2_models.py` | 新增 |
| 入口 | `experiments/cw_v1/r2_0_card.py`（`--ctrl-only` 只跑自证） | 新增 |
| 判据自证 `C1–C5` | **5/5**（0.0 s） | 通过 |
| 闸 `A1–A10` | **14 项 / 0 fail**（1.17 s 全轮） | 全绿 |
| 证据 | `results/cw_v1/cw_r2_0_card_20261024T173546_2ade27baa8965d97/`（`spec/gates/ctrl/meta`） | 新增 |
| 作废留档 | `…T173524_…_SUPERSEDED/`（首跑 2 FAIL，**均为判据自伤**） | 留档 |
| 预算/资源 | **1.17 s / 47.3 MB / `run_episode`=0 / `world.step`=0 / `CWWorld` 构造 1 次** | 实测 |

`spec.json` 里同时落盘：可比性 spec+指纹、47 条信息权限、资产清单（7 可复用 + 3 不可复用，每条带机器判据）、
目标函数表、零控定义、`H-R2-1`、预算区间、残差锚点读数。

---

## §4 本轮自己的错（**全部是判据/口径错，不是数字错**）

1. ★ **`A1b` 的断言不可达**：要求"属性守卫也能接受 `dict(view._pub)`"。
   `hasattr(dict, "t")` **恒 False** —— 属性守卫看不到 dict 的键，这条断言**算术上不可能**成立。
   首跑因此 FAIL。改判据：只有子类替身参与判定，`dict` 只作记录。
   （与 R1.17 #4 的"`|share|>1e3` 不可达"同族：**写断言前先算可达范围**。）
2. ★ **资产清单判据对象选错**：用 `"ChannelAdapter" in src` 判"是否依赖 P3 的 `ChannelAdapter`"，
   而 `streams.py:4` 的 docstring 里**正写着这个词**（解释"为什么不用 `salt+7919k`"）⇒
   字符串命中把注释误判成依赖，`inventory_bad` 多出 1 项。改为 AST 级判据（无 `ClassDef`/无 import）。
3. 代码层自伤（**跑之前**发现）：`scan_source` 里 `zip(skip, ["TruthDiagnostic"])` 的两个列表
   长度与顺序不一致，多类跳过时会错配 ⇒ 改成并行收集。
4. 首次跑前 `channel["kappa"]` 用 `float(...)+0.1` 造变异 —— 若原值是 `int` 会得到 float，
   指纹变化仍然成立；这条**不是**错误，记录一下以免以后误读为"类型变了才变"。

★ 两条 1/2 都是"判据描述的对象不存在/不符"，**没有为一过闸去改被测物**（被测物一字节未动）。

---

## §5 没做 / 边界（如实）

- **没有训练**任何模型；`H-R2-1` 只是写进卡，**未检验**，`needs` 里明写"训练（未开始）"。
- **没有跑 episode**（`run_episode`=0、`world.step`=0）。`CWWorld` 构造 1 次仅为取真实视图键集。
- **没有**扩 `ep102+`、没有重跑旧 P3、没有动 `results/` 里任何旧目录。
- `A3` 只覆盖声明为部署路径的 `r2_models.py`；**跨模块绕过**（在别处写个 helper 再 import 进来）
  不在本轮覆盖范围，白名单 import 能挡住"直接 import `world` 的非视图名"，但挡不住"两个模块互相递数据"。
  这条写进 §6 的下一步。
- 外部三项（投递/联系/外部验证）**恒 0**。

---

## §6 下一轮（`CW-R2.1`，须先写卡）

唯一主件：**`H-R2-1` 的第一个可否定实例** —— 用**零训练**的 `belief_readout` 与 `prior_only`
在两个 plant 档上各跑 10 episode，先把
① `floor_R2`（`PlanRef(seed=…)` 零控）测出来，
② 把"强基线 vs 弱基线"的差值放到 `floor_R2` 上看**有没有分辨力**。

- 若`belief_readout` vs `prior_only` 的差 ≤ `floor_R2` ⇒ **连强弱基线都分不开**，
  那"小学习模型 vs 强模型"更不可能测出 ⇒ 必须先扩 episode 或换指标，**不进训练**。
- 只有 ② 站得住，才谈"学习模型"。这条顺序写死进卡，防止"先训练再找面板"。
- 预算：40 格 × [3.12, 4.49] s ≈ **125–180 s**（本轮实测区间，不单点外推）。
