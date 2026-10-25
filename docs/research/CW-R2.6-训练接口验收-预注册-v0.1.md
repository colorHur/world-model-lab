# `CW-R2.6` 训练接口验收（`T1–T6` 活体反例）· 预注册 v0.1

- 写于：**2026-10-25 15:01 +08:00**（`Get-Date` 实测）
- 写作时状态：`run_episode` = **0**、`world.step` = **0**、参数更新 = **0**
- 上游：`docs/research/CW-R2-训练接口-v0.1.md`（§2 容量 / §3 预算 / §4 目标 / §5 评价 / §6 分层 / §7 验收 `T1–T6`）
- 前置判词：R2.5 `AOI-SEED-ROBUST`（`docs/CW-R2.5-种子稳健-交付.md`）
- 本轮结论的作用域：**只**回答"接口卡 §7 的六条反例能不能被机器拦住、且不误杀正常路径"，
  **不**回答"学习模型有没有用"（本轮仍零训练，`H-R2-1` 不检验）

---

## §1 问题

接口卡把容量/预算/目标/评价/分层**钉成了文字**，但：

1. 没有可断言的常量对象 ⇒ 谁都能在代码里写 `5e5` 参数继续跑，卡拦不住；
2. §7 的六条反例**一条都没跑过** ⇒ "接口可用"目前是**声明**，不是**证据**；
3. `OBJECTIVE["forbidden"]`（`ku/kf/share*`）在代码里**只有声明、没有执行点**（`grep` 全文无引用）
   ⇒ 今天真有人 `primary="ku"` 是**静默通过**的。

本轮把 1–3 全部变成可执行、可复算、带负控的检查。

## §2 被测物与不动的部分

| 类别 | 文件 | 本轮动作 |
|---|---|---|
| 被测物（**一字节不动**） | `world.py` / `plan_ref.py` / `plan_dp.py` / `runner.py` / `config.py` / `policies.py` / `model_api.py` / `r2_models.py` / `mech2.py` / `mech3.py` / `r2_1_sep.py` / `r2_4_aoi.py` / `r2_5_seed.py` | 只 import / 只读源码文本 |
| 新增 | `experiments/cw_v1/r2_train_spec.py` | spec 常量 + 守卫 + **未训练** numpy 骨架 + 报告生成器 |
| 新增 | `experiments/cw_v1/r2_6_gates.py` | 入口：跑 `T1–T6` 双向腿并落盘 |
| 新增 | `experiments/cw_v1/fixtures/dirty_model_src.py.txt` | `T2` 的脏源码**文本**（`.py.txt` 后缀 ⇒ 不被 import、不被任何闸误扫） |

★ 不改 `model_api.OBJECTIVE`。若发现卡与代码口径不一致，**只报告、不改被测物**（§7 冲突项）。

## §3 六条反例的**双向**判定（本轮的核心设计）

每条 `Tk` 必须两向都成立，缺一向即判本轮不合格：

| 编号 | 反例（必须被拦） | 负控（必须放行） | 无分辨力的两种失败模式 |
|---|---|---|---|
| `T1` | `assert_objective("ku")` ⇒ `ValueError` | `assert_objective("A_t")` ⇒ 返回归一化评价描述 | 无条件 raise（负控炸）；无条件 pass（反例漏） |
| `T2` | `scan_source(脏源码)` ⇒ `names≥2 且 attrs≥1 且 imports≥1` | `scan_source(r2_models.py)` 与 `scan_source(r2_train_spec.py)` ⇒ **三栏全空** | 正则/字符串匹配（docstring 里就有 `ChannelAdapter` 那类误报史，见 R2.0）；或永远返回空 |
| `T3` | `require_view(EvilView())` ⇒ `TypeError` | `require_view(真视图)` ⇒ 返回本体 | 只查 `hasattr`（子类混过）；或拒一切 |
| `T4` | `assert_disjoint(range(0,70), range(60,75))` ⇒ `AssertionError` | `assert_disjoint(TRAIN_EPS, DEV_EPS)` ⇒ 通过 | 交集恒算成空；或恒抛 |
| `T4b` | **活体**：从已落盘证据读 R2.1–R2.5 实际用过的 ep 集，与 `TRAIN_EPS` 求交 ⇒ 必须为空 | 同左（这条本身就是活体，无独立负控） | 若把训练集写成 `0–114`，这条必须 FAIL |
| `T5` | `assert_capacity(width=256)` ⇒ `CapacityExceeded`，且异常里带**实测**参数计数 | `assert_capacity(width=32)` ⇒ 通过并报实测计数 | 硬编码一个 `50001` 充当"实测"；或限额写成立即失败 |
| `T6` | `build_report({7: [...]})`（只 1 个种子）/ `build_report({"pooled": [...]})` ⇒ `ReportRefused` | `build_report({7,17,27 齐})` ⇒ 通过，输出 `by_seed` + `pooled`（后者标 `descriptive`） | 报告器不看种子键（恒过）；或拒一切 |

判词：

- **`SPEC-GATES-READY`** ⟺ `T1–T6` 六条 `caught` 全真 **且** 六条 `clean` 全真；
- 否则 **`SPEC-GATES-INCOMPLETE`**，本轮**不得**声称"R2 训练接口可用"，也不得开训。

## §4 卡与代码口径的**已知冲突**（跑前登记，本节不许跑后修改）

`R2.6` 读代码时发现三处"卡说 A、代码是 B"，本轮**只报告不改**：

1. **主指标**：接口卡 §5 写"主：`A_t`"（R2.4 口径）；`model_api.OBJECTIVE["primary"]` 仍是
   `"cost_true_per_step_median"`（R2.0 立的）。⇒ 两者不一致。本轮 `EVAL_PRIMARY = "A_t"`，
   并把 `OBJECTIVE["primary"]` 记成 `legacy_primary`，**冲突入证据**，不改 `OBJECTIVE`。
2. **"20 个公开字段"**：`len(VIEW_TOP_KEYS)+len(PRIOR_KEYS)+len(TELEMETRY_KEYS) = 20`，
   但 `require_view` 只保证**顶层 7 键**；`history` 的 12 个子字段不在 20 之内。
   ⇒ "20 字段"是**声明口径的算术**，不是守卫的实际覆盖面。本轮现读算出，不手抄。
3. **两个容量上限不自洽**：`≤5e4 参数` 与 `≤2 隐层 / ≤64 宽` 同时成立时，`d_in=16` 的网络
   最多 `16·64+64 + 64·64+64 + 64·2+2 = 5314` 参数 ⇒ `5e4` 这条**永远不会先触发**。
   ⇒ 真正起作用的约束是宽度/层数；`T5` 必须两条都测（参数上限 + 结构上限），
   且反例要选**能先撞到哪条就撞哪条**的宽度，不许只写一个数糊过去。

## §5 指标与效应（判词只用这三列）

1. `caught`（布尔）+ 拦截方式（异常类型 / 命中计数）；
2. `clean`（布尔）+ 负控返回的关键值；
3. `n_pass / n_checks`（12 腿 = 6 × 2，`T4b` 计入 `T4` 的 `caught` 腿，不另立编号）。

**没有**效应量/显著性这一层：这是**软件闸**，判定是确定性的（同一输入必得同一结果）。
⇒ 本轮**不报** `p` 值、不报 `median|Δ|`、不做 α 敏感性（那些是面板判据的语法，用在这里是装饰）。

## §6 预算与资源

| 项 | 上限 | 说明 |
|---|---|---|
| 墙钟 | **≤ 120 s** | 纯 CPU |
| `run_episode` | **0** | 不跑世界；只构造一次 `CWWorld` 取真视图（`step` = 0） |
| `world.step` | **0** | 同上 |
| 参数更新 | **0** | 骨架随机初始化后**不训练**；`trained=False` 写进证据 |
| 新增 `(plant, ep)` 单元 | **0** | 本轮不碰面板 |

★ 反例用的 `EvilView` 只**构造**不**运行**；`T3` 的负控视图由 `CWWorld.view()` 提供（构造后即弃，不 step）。

## §7 否定条件（写清"什么数据能让我这轮 FAIL"）

1. 任一 `Tk.caught == False` ⇒ 该闸**无分辨力** ⇒ `SPEC-GATES-INCOMPLETE`；
2. 任一 `Tk.clean == False` ⇒ 该闸**过严**（会误杀正常路径）⇒ 同上；
3. `T4b` 交点非空 ⇒ 训练/评测**数据泄漏**，接口不可用；
4. `assert_capacity` 打印的实测计数与 `sum(p.size for p in params)` 不等 ⇒ 计数是假的；
5. 六条全过但 `total_s > 120` ⇒ 只报"过但超预算"，**不**改判词（预算不是正确性）；
6. **不许**为了过闸去改 `model_api.OBJECTIVE` / `TRAIN_EPS` / 宽度上限 / 判据函数体 ——
   若必须改，说明闸写错了，要**先改卡再改码**并把改动写进交付 §自伤。

## §8 产物清单（跑前定，交付时逐项对）

| # | 产物 | 位置 |
|---|---|---|
| 1 | spec + 守卫 + 骨架（未训练） | `experiments/cw_v1/r2_train_spec.py` |
| 2 | 入口（`T1–T6` 双向腿 + 落盘） | `experiments/cw_v1/r2_6_gates.py` |
| 3 | 脏源码 fixture | `experiments/cw_v1/fixtures/dirty_model_src.py.txt` |
| 4 | 证据目录（`t_gates.json` / `summary.json` / `meta.json` / `gates.json`） | `results/cw_v1/cw_r2_6_spec_<UTC>_<srchash>/` |
| 5 | 交付 | `docs/CW-R2.6-训练接口验收-交付.md` |
| 6 | README 一行 | `README.md`（R2.6 线段） |
| 7 | **R17 附加件** | **触发条件**：本轮若出现"某臂更好 / 某量改善"的结论 ⇒ 必须逐用户报 `r_i` 并附 `r2_6b`。**本轮预期不触发**（零训练、无面板、无臂比较）⇒ 交付里必须写明"未触发原因" |
| 8 | `_POST_COMMIT.json`（执行版 == 提交版） | 证据目录内 |

## §9 明确不做

- 不训练、不写学习曲线、不调超参、不评估 `H-R2-1`；
- 不扩 episode、不重跑旧 P3、不改 `sigma_meas`、不改任何判据常量；
- 不因"六条过了"就说"学习模型会成功"—— 那只说明**评测基础设施**到位；
- 不做"无限审计"：只测这六条 + 三条口径冲突的记录，不再扩张。

## §10 跑前修订记录（如有，逐条追加，不删不改）

### 修订 A1 · 2026-10-25 15:1x（**跑数前**：`run_episode` = 0、`results/` 下未建任何目录）

**改什么**：§3 的判词从"6 条 × 双向 = 12 腿"**收紧**为 **14 腿**，新增三项：

1. **骨架前向（活体）**：`NumpyMLP` 必须能真跑一次前向、过 `ModelOutput.validate`、
   `P > 0`、**两次前向参数逐位不变**（零训练的第二重证据），且**没有** `fit` 方法。
2. **被测物身份**：预注册 §2 列的 13 个被测物文件，`git hash-object` 必须等于 `HEAD:path`。
   （零计算、不重跑旧实验 —— 比"再跑一遍 `run_all`"更省且更强。）
3. **变异测试 `M1–M6`（闸的闸）**：把每条守卫逐个替换成"永远放行"的替身，
   对应 `Tk` 必须**转红**。若某 T 变异后仍绿 ⇒ 该 T 是**恒绿闸**（`caught` 腿没有分辨力）⇒ 记 FAIL。

**为什么**：原卡的 12 腿只测了"当前实现拦得住"，**没测"检验本身有分辨力"**。
这正是本项目已固化的第 ⑫ 条（"跑过了" ≠ "有分辨力"）。不收紧的话，`T1–T6` 全绿
也可能全部是装饰 —— 一个写在函数第一行的 `raise` 就能让六条全过。

**不变**：`T1–T6` 的双向定义、判词名（`SPEC-GATES-READY`）、零训练 / 零 episode、
120 s 预算、以及 §7 的六条否定条件（仍全部有效，只是不再充分）。

**可核查事实**（防止"看了结果再改卡"的质疑）：
① 修订写于首跑 `--no-evidence` 之后、**任何证据目录创建之前**（`results/cw_v1/` 下无 `cw_r2_6_spec_*`）；
② 修订方向是**收紧**（6/6 → 14/14），不是放宽；
③ 主判 `T1–T6` 的定义一字未改。

### 修订 A2 · （预留，本轮未触发）

