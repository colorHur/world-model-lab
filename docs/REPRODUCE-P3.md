# REPRODUCE-P3 —— 只用 CPU，**3.6 秒**把 P3 的主结果那一格重跑出来并逐位对账

> ★ 本文只覆盖 **P3-0 主确认表的一格**（`B=200 unpaced`，臂 `age`，4 个 cell）。
> **不是**"一键复现全部"：P3-1 / P3-2 的扫描是 96 + 144 cell、合计约 400 s，命令见 §5。
> **也不是**跨机验收：逐位一致的要求只在**同机 / 同依赖版本**下成立
> （理由见 `docs/REPRODUCE.md` §6 的容差纪律：PyTorch 官方明确写跨平台同种子不保证逐位复现）。

---

## 0. 一句话

```bash
D:/devtools/miniconda3/envs/ai-lab/python.exe scripts/62_min_repro.py
```

在**本机**上（2026-10-20 12:03 实测）输出 `BITWISE MATCH`，退出码 `0`，耗时 **3.6 s**。

## 1. 前置

| 项 | 要求 | 本机实测 |
|---|---|---|
| Python | 3.11 | `D:/devtools/miniconda3/envs/ai-lab/python.exe`（3.11.16） |
| 依赖 | `numpy` / `gymnasium`（+ 仓库内 `wmlab`） | numpy 2.4.6、gymnasium 1.3.0 |
| GPU | **不需要**（纯 CPU） | 本机无 NVIDIA 独显 |
| 证据 | `results/runs/<run_id>/58_p3_confirmation.json` **必须存在** | 已入库（`20261020-050355-613f4be`） |

★ `git` / `python` 不在 PATH 上，命令要写**绝对路径**（本机习惯）。

## 2. 跑什么

| 项 | 值 |
|---|---|
| 档 | `budget=200`，`mode=unpaced`（`B=L` ⇒ 配速 no-op，`t>B` 发帧恒 0） |
| 臂 | `age`（唯一臂） |
| 预测器 | `zoh`（基线）、`linear`（冻结方法） |
| 种子 | `100`、`130`（主确认集的 3 个种子里取 2 个；`--seeds 100,130,160` 可跑满） |
| 集数 | 30（与入库证据一致；改集数就**不是**复现，见 §4） |

脚本**直接复用** `scripts/58` 的 `run_cell()` —— 吸收项、预热、臂、误差探针全部同源。
这点是刻意的：自己重写一份 runner 得到"相近的数"不叫复现（R12「参数接线了没」的同一件事）。

## 3. 期望输出（2026-10-20 12:03 实测，逐字）

```text
[ref] results/runs/20261020-050355-613f4be/58_p3_confirmation.json
[cfg] B=200 mode=unpaced arm=age episodes=30 seeds=[100, 130]
  B=200 zoh    seed=100  J=(2.983265, 0.542333)  0.62s
  B=200 linear seed=100  J=(0.918946, 0.543333)  0.63s
  B=200 zoh    seed=130  J=(2.923935, 0.593333)  0.82s
  B=200 linear seed=130  J=(0.899460, 0.456500)  1.57s
[time] 4 cell 共 3.64 s（完整）

[compare] 与入库证据逐 episode 比对
  [PASS] seed=100 zoh    n=60  max|Δ|=0.000e+00  不一致 0 个
  [PASS] seed=100 linear n=60  max|Δ|=0.000e+00  不一致 0 个
  [PASS] seed=130 zoh    n=60  max|Δ|=0.000e+00  不一致 0 个
  [PASS] seed=130 linear n=60  max|Δ|=0.000e+00  不一致 0 个
[out] outputs/62_min_repro.json（outputs/ 不入库）

BITWISE MATCH
```

判据：**`max|Δ| = 0` 且不一致个数 = 0**（容差 0，同机同版本）。任一处非 0 ⇒ 退出码 `2`。
`outputs/` 不入库；`results/` **不写**（复现入口不得触碰入库证据）。

## 4. 它**验证了什么 / 没验证什么**

| 验证了 | 没验证 |
|---|---|
| `58` 的 4 个 cell 能由当前代码逐位重算（代码没漂移） | 95% 区间（bootstrap 有随机性，脚本不重跑它） |
| 配置（吸收项、预热、臂、探针）在"复现"与"实验"之间一致 | 跨机 / 跨 numpy 版本的数值一致性 |
| 失败会真的报出来（见 §4b） | P3-1 / P3-2 的扫描（§5 另给命令） |

### 4b 阴性对照（判据的分辨力）

```bash
... scripts/62_min_repro.py --episodes 29      # ⇒ 形状 (29,2) vs (30,2)，4 个 FAIL，退出码 2
```
★ 这一条是在**本机实测过**的（12:02）：退出码确实为 `2`。
它证明"PASS"不是恒真 —— 改一个集数就会 FAIL。

## 5. 想跑全量（约 400 s，仍然只要 CPU）

```bash
# P3-0 全量：96 cell，184.6 s（本机实测）
... scripts/58_p3_confirmation.py

# P3-1 适用域：96 cell，137.5 s；补点 18 cell，约 26 s
... scripts/59_p3_boundary.py
... scripts/59_p3_boundary.py --refine

# P3-2 失效窗口：144 cell，259.7 s
... scripts/60_p3_window_shift.py
```

★ 这三个脚本会往 `results/runs/<新 run_id>/` **写新证据**（不是覆盖旧证据：同名再写会 `exit 3`）。
只想检查逻辑、不产出证据时加 `--skip-experiment`（只跑自检）。

## 6. 表与图怎么重建（不用跑实验）

```bash
... scripts/61_build_tables.py          # → docs/RESULTS-P3.md + docs/results_p3_trace.json
... scripts/61c_plot_paper_fig.py       # → assets/paper_p3.png
```

`61` 不写 `results/`、不重跑实验；它从入库 JSON 重算 `R` 并做三件事：
① 与 summary 对账（`1e-9`）；② 每个显示值在该 JSON 里**字面 grep** 得到；③ 内容不变则不重写（重跑字节相同）。

## 7. 已知坑

1. **`--quick` 不比对**：它只跑 2 集验流程，会打印"不比对"。拿它当复现是错的。
2. **证据缺失**：若 `results/runs/*/58_p3_confirmation.json` 不在（例如只 clone 了代码、没带 `results/`），
   脚本打印 `[SKIP]` 并只跑不比对 —— 这是设计行为，不是 PASS。
3. **耗时随机器变**：3.6 s 是 i7-14700K 单线程附近的实测；换机只记偏差，不改判据（判据是 `max|Δ|=0`）。
