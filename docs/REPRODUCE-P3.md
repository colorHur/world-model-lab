# REPRODUCE-P3 —— 只用 CPU，**秒级**把 P3 主结果那一格重跑出来并逐位对账

> ★ 本文只覆盖 **P3-0 主确认表的一格**（`B=200 unpaced`，臂 `age`，4 个 cell）。
> **不是**"一键复现全部"：P3-1 / P3-2 的扫描是 96 + 144 cell、合计约 400 s，命令见 §5。
> **也不是**跨机验收：逐位一致的要求只在**同机 / 同依赖版本**下成立
> （理由见 `docs/REPRODUCE.md` §6 的容差纪律：跨平台同种子不保证逐位复现）。
>
> ★ **2026-10-20 复验后改过判据**：旧版在"没有比对"时也会打印 `BITWISE MATCH`（见 §4b）。
> 现在的判词只有四种，且**退出码能区分**，自动任务不会把"没检查"当"通过"。

---

## 0. 一句话

```bash
D:/devtools/miniconda3/envs/ai-lab/python.exe scripts/62_min_repro.py
```

本机实测（2026-10-20，修复后）：`expected=4 compared=4 skipped=0 failed=0 invalid=0`，
打印 `BITWISE MATCH`，退出码 `0`，耗时约 **4.4 s**（含启动与 4 个 cell）。

## 1. 前置

| 项 | 要求 | 本机实测 |
|---|---|---|
| Python | 3.11 | `D:/devtools/miniconda3/envs/ai-lab/python.exe`（3.11.16） |
| 依赖 | `numpy` / `gymnasium`（+ 仓库内 `wmlab`） | numpy 2.4.6、gymnasium 1.3.0 |
| GPU | **不需要**（纯 CPU） | 本机无 NVIDIA 独显 |
| 证据 | `docs/evidence_manifest_p3.json` **钉死的那份**必须存在 | 已入库（`20261020-050355-613f4be`） |

★ `git` / `python` 不在 PATH 上，命令要写**绝对路径**（本机习惯）。

★ **证据来源默认是冻结清单，不是 latest**：清单里写死 `run_id` + `sha256_norm` + 协议。
新跑一个 run 不会静默换掉输入；清单与磁盘不一致时脚本直接 `NOT VERIFIED`。
想临时换证据，用 `--ref <path>`（命令行上看得见）；`--latest` 才退回"取最新那份"。

## 2. 跑什么

| 项 | 值 |
|---|---|
| 档 | `budget=200`，`mode=unpaced`（`B=L` ⇒ 配速 no-op，`t>B` 发帧恒 0） |
| 臂 | `age`（唯一臂） |
| 预测器 | `zoh`（基线）、`linear`（冻结方法） |
| 种子 | `100`、`130`（主确认集 3 个种子里取 2 个；`--seeds 100,130,160` 跑满） |
| 集数 | 30（与冻结证据一致；改集数就**不是**复现） |
| 期望格数 | `expected = 种子数 × 2`（默认 4） |

脚本**直接复用** `scripts/58` 的 `run_cell()` —— 吸收项、预热、臂、误差探针全部同源。
这点是刻意的：自己重写一份 runner 得到"相近的数"不叫复现（R12「参数接线了没」的同一件事）。

## 3. 期望输出（逐字，本机实测）

```text
[ref] .../results/runs/20261020-050355-613f4be/58_p3_confirmation.json（来源：冻结清单）
      run_id=20261020-050355-613f4be sha_norm=4d597ca01d80
[cfg] B=200 mode=unpaced arm=age episodes=30 seeds=[100, 130]
  B=200 zoh    seed=100  J=(2.983265, 0.542333)  0.62s
  B=200 linear seed=100  J=(0.918946, 0.543333)  0.66s
  B=200 zoh    seed=130  J=(2.923935, 0.593333)  1.49s
  B=200 linear seed=130  J=(0.899460, 0.456500)  1.57s
[time] 4 cell 共 4.35 s（完整）

[compare] 与冻结证据逐 episode 比对
  [PASS] seed=100 zoh       n=60  max|Δ|=0.000e+00  不一致 0 个
  [PASS] seed=100 linear    n=60  max|Δ|=0.000e+00  不一致 0 个
  [PASS] seed=130 zoh       n=60  max|Δ|=0.000e+00  不一致 0 个
  [PASS] seed=130 linear    n=60  max|Δ|=0.000e+00  不一致 0 个
[out] outputs/62_min_repro.json（outputs/ 不入库）

[count] expected=4 compared=4 skipped=0 failed=0 invalid=0

BITWISE MATCH
```

**判据（四条同时满足才叫匹配）**：`expected > 0`、`compared == expected`、`skipped == 0`、
`failed == 0`、`invalid == 0`，且逐 episode `max|Δ| = 0`（容差 0，同机同版本）。

**退出码分层**（自动任务按码分支，不要只看有没有报错）：

| 码 | 判词 | 含义 |
|---|---|---|
| `0` | `BITWISE MATCH` | 期望集合全部被比对且逐位相等 |
| `2` | `MISMATCH` | 有格比对上了但结果不等 |
| `4` | `NOT VERIFIED` | **缺证据 / 未知种子 / 缺格 / 非有限值** ⇒ 有东西没被比对 |
| `5` | `SMOKE ONLY` | `--quick` 流程冒烟，**不是**复现验收 |

`outputs/` 不入库；`results/` **不写**（复现入口不得触碰入库证据）。

## 4. 它**验证了什么 / 没验证什么**

| 验证了 | 没验证 |
|---|---|
| 4 个 cell 能由当前代码逐位重算（代码没漂移） | 95% 区间（bootstrap 有随机性，脚本不重跑它） |
| 配置（吸收项、预热、臂、探针）在"复现"与"实验"之间一致 | 跨机 / 跨 numpy 版本的数值一致性 |
| 失败会真的报出来（§4b 的 6 条反例） | P3-1 / P3-2 的扫描（§5 另给命令） |

### 4b 反例矩阵（判据的分辨力）—— `scripts/64_gate_selftest.py`

★ 旧版的教训：`--seeds 999` 两格 SKIP，却打印 `BITWISE MATCH`、退出码 `0`。
根因是判据写成了 `bad == 0` —— **"没比较过"和"比较后相等"都被算成 0**。

现在把"错法"逐条固化成可执行反例（全部在 `tempfile` 隔离副本上注入，不碰 `results/`）：

| # | 反例 | 期望 | 实际 |
|---|---|---|---|
| 1 | 正常四格 | 0 + `BITWISE MATCH` | 0 |
| 2 | `--seeds 999`（证据里没有） | 4 + `NOT VERIFIED`，**不得**出现 `BITWISE MATCH` | 4 |
| 3 | `--quick` | 5 + `SMOKE ONLY` | 5 |
| 4 | `--ref` 指向不存在的文件 | 4 | 4 |
| 5 | 证据里删掉一格（seed=130/linear） | 4（`skipped=1`） | 4 |
| 6 | 证据里注入 NaN | 4（`invalid=1`） | 4 |
| 7 | 论文正文正常（claim 绑定全命中） | 0 + `ALL CHECKS PASS` | 0 |
| 8 | **把真实数字 `0.315` 换成 `1.223`（错条件）** | 2（S11c FAIL） | 2 |
| 9 | 把 `0.315` 改成 `0.311`（改末位） | 2（S11c FAIL） | 2 |

```bash
... scripts/64_gate_selftest.py        # 全量约 25 s；--fast 只验 61 的数字闸
```

★ 第 6 条的细节：NaN 的 `|a-b| > 0` 恒为 `False`，旧代码会把它当"一致"。
现在先做有限性检查，非有限值直接进 `invalid`，**不参与相等判定**。

## 4c 工程资产收口实测（2026-10-21，**新机器视角**）

> 目的不是再出一次数，而是**按本文档逐条走一遍**，确认"一个没有本机缓存的人照抄命令能走到哪一步"。
> 本轮**零库改动**（只新增 `README` 索引与本表），`results/` **零写入**。

| # | 命令 | 实测（2026-10-21 11:48–11:55 +08:00） | 结论 |
|---|---|---|---|
| 1 | `scripts/00_check_env.py` | **15/15 通过**（含中文字体链路、`reliable_horizon` 插值 3.2） | 依赖装对 |
| 2 | `tests/test_channel_and_aoi.py` | **89/89 passed**，exit 0 | 回归自检全过 |
| 3 | `scripts/02_train_world_model.py` | 参数量 **67,652**；H\* 开环 **12.945840516581065** / 闭环 **6.955816701033772**；训练 22.9 s，总 25.5 s | **与 §4 判据表的参考值逐位一致**（同机同版本） |
| 4 | `scripts/30_scheduling_baselines.py --n 50000 --tmax 16` | 99 点最差比值 **0.796**（≤1）；`p=0` 比值中位 **1.0000**；总 6.4 s | 闭式对账与已知答案均过 |
| 5 | `scripts/62_min_repro.py` | `expected=4 compared=4 skipped=0 failed=0 invalid=0`，4 cell 共 **2.52 s**（墙钟 6.3 s），**BITWISE MATCH**，exit 0 | 主结果一格逐位重算通过 |
| 6 | `scripts/64_gate_selftest.py` | **9 条反例全部按预期**（未知种子 4 / `--quick` 5 / 缺格 4 / NaN 4 / 错条件 2 / 改末位 2…） | 判据有分辨力 |
| 7 | `scripts/63_freeze_manifest.py`（主 + `--set neg`） | 两清单各 4 entries，`unchanged`，exit 0 | 冻结清单未被本轮触碰 |
| 8 | `scripts/70_csi_identifiability.py --no-persist` | C4 复算与 T29 逐项相同（σ=0 反转 **恰好 0**；σ=0.1 `dq` 中位 **0.0594**），exit 0 | 诊断入口可只读复跑 |

★ **本轮耗时与 §0 的 4.4 s 不同**（2.52 s / 6.3 s 墙钟）：那是 10-20 修复后的一次实测，
耗时随机器与后台负载变，**判据是 `max|Δ| = 0`，不是秒数**（§7 已知坑 4）。

★ **没走的两步与本轮发现的问题**（如实记，不在本轮修）：

1. **§6 外部验收（换机）本轮没做**：`docs/REPRODUCE.md` §6 的克隆地址
   `github.com/colorHur/world-model-lab` 与本机 `git remote -v`（fetch/push 两条）**一致**，不是问题；
   没做的原因是"换机"本身不在本机可完成的范围，**外部验收计数仍为 0**。
2. §6 的清单**没有**列 `62 / 64 / 65 / 70` 这四个诊断入口 —— 它们不产出结论，但**决定"结论有没有资格"**。
   README 已补索引表；§6 清单保持原样（那是给外部执行者的，不和本轮的实测记录混写）。
3. **§6 的 `02` 参考值 `final loss train = 4.5944e-4`** 本轮未逐字核对（脚本只打印每 15 epoch 的采样），
   **不据本轮声明它变了或没变**；`H*` 三个值已逐位命中，足以判定"训练链路未漂移"。
4. `§6` 的 `33_unified_closed_loop.py --mode both`（约 12–18 min CPU）**本轮未跑**：它是全量闭环，
   不属于"秒级复现"这一层；不在本轮预算内，也不把它写成"已验证"。

★ **自伤记录**：本节的第 1 条初稿写成"克隆地址是旧的、照抄会失败"，**未核实**；
   `git remote -v` 一查即知本地 remote 就是该地址。**"看起来像笔误"不等于"确实是错的"**，
   断言前先跑一条命令（R14 同族，自证伪第 61 条）。

## 5. 想跑全量（约 400 s，仍然只要 CPU）

```bash
# P3-0 全量：96 cell，184.6 s（本机实测）
... scripts/58_p3_confirmation.py

# P3-1 收益/退化范围：96 cell，137.5 s；补点 18 cell，约 26 s
... scripts/59_p3_boundary.py
... scripts/59_p3_boundary.py --refine

# P3-2 失效窗口：144 cell，259.7 s
... scripts/60_p3_window_shift.py
```

★ 这三个脚本会往 `results/runs/<新 run_id>/` **写新证据**（不是覆盖旧证据：同名再写会 `exit 3`）。
只想检查逻辑、不产出证据时加 `--skip-experiment`（只跑自检）。

★ 跑完全量**不会**自动改主表：主表读的是冻结清单。要让新证据进表，必须显式
`python scripts/63_freeze_manifest.py --allow-update`（旧清单另存为 `.v1.json`）。
这是刻意的：换证据必须是**可见的动作**，不是"跑了个实验，表就悄悄变了"。

## 6. 表与图怎么重建（不用跑实验）

```bash
... scripts/61_build_tables.py          # → docs/RESULTS-P3.md + trace + claims_p3.json
... scripts/61c_plot_paper_fig.py       # → assets/paper_p3.png
... scripts/61d_plot_cost.py            # → assets/paper_p3_cost.png
... scripts/61e_build_deck.py           # → docs/DEMO-P3.html
```

`61` 不写 `results/`、不重跑实验；它从冻结证据重算 `R` 并做六件事：
① 与 summary 对账（`1e-9`）；② 每个显示值在该 JSON 里**字面 grep** 得到；
③ 派生量按其声明的操作数/口径重算；④ 每条 claim 的绑定键能解析并复现显示值；
⑤ 正文里带 claim-id 的数字必须逐字等于绑定值（含两条阴性对照）；
⑥ 内容不变则不重写（重跑字节相同）。

## 7. 已知坑

1. **`--quick` 不是复现**：它只跑 2 集验流程，退出码 `5`、打印 `SMOKE ONLY`。拿它当验收是错的。
2. **证据缺失**：若清单指向的文件不在（例如只 clone 了代码、没带 `results/`），
   脚本打印 `NOT VERIFIED`、退出码 `4` —— 这是设计行为，不是 PASS。
3. **别改清单去"修"证据**：清单是唯一入口，手改会让 `resolve()` 报内容指纹不符。
4. **耗时随机器变**：4.4 s 是 i7-14700K 单线程附近的实测；换机只记偏差，不改判据（判据是 `max|Δ|=0`）。
5. **CRLF/LF**：本机 `write_text` 写 CRLF、`core.autocrlf=input` 提交时规范成 LF ⇒
   工作树字节 sha 与 git blob sha 必然不同。清单用 `sha256_norm`（行尾归一）做权威指纹，
   原始字节 sha 只备查，**不要把换行差异当成证据被改**。
