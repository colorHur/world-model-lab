# world-model-lab

> 世界模型 + rollout + 可靠视界评测的最小可复现框架。
> 场景锚点：**UAV / 无线通信**（本框架先用 CartPole 级任务把训练循环跑通，再换场景，只动 `wmlab/envs/`）。

## 这个仓库解决什么问题

世界模型（world model）在 rollout 到第 N 步之后会失准，但**几乎没人明确回答"第几步开始不准、为什么"**。
本仓库的目标是把这件事做成**可测量、可复现、可对比**的：

- 统一的环境接口（`envs`），换场景不改模型与评测代码
- 最小的世界模型实现（`models`），编码器 + 潜空间转移 + 解码器
- 潜空间 rollout（`rollout`），支持任意视界
- **可靠视界（reliable horizon）**定义与测量（`eval`）——预测误差首次超过阈值的那一步

## 快速开始

```bash
# 1) 环境（本机已建好 ai-lab，见项目文档）
conda activate D:\devtools\miniconda3\envs\ai-lab
# 或在 VSCode 里直接用任务：Terminal -> Run Task -> wmlab: 环境自检

# 2) 环境自检（15 项，含中文字体链路）
python scripts/00_check_env.py

# 3) 随机策略 baseline，产出第一张图
python scripts/01_random_baseline.py

# 4) 训练一个世界模型，产出 loss 曲线 + 多步预测误差曲线
python scripts/02_train_world_model.py

# 5) 分析误差增长的形态（饱和 / 线性累积 / 自我放大，三选一）
python scripts/03_analyze_horizon_curve.py outputs/02_world_model_pendulum.json
```

换环境只换配置，代码不动（长视界那轮就是这么做控制变量的）：

```bash
python scripts/02_train_world_model.py --config configs/pendulum.yaml --tag 02_world_model_pendulum
```

所有结果图写入 `outputs/`（不进版本库）。

## 目录结构

```
wmlab/
  envs/     环境适配层。统一接口，换 UAV/信道场景只加 Adapter
  models/   世界模型。编码器 / 潜空间转移 / 解码器
  rollout/  潜空间多步 rollout，以及误差随视界增长的测量
  eval/     指标：NMSE、可靠视界、预测区间覆盖率
  utils/    设备选择（CPU/云 GPU 无痛切换）、随机种子、统一绘图样式
scripts/    可直接运行的入口脚本（编号即依赖顺序；00 = 环境自检）
configs/    YAML 配置（device: auto 时自动选 cuda 或 cpu）
outputs/    ★ 结果图与日志，已 gitignore
```

## 设计约定（为云 GPU 迁移服务）

1. **设备无关**：任何 `.to(...)` 都走 `wmlab.utils.device.get_device()`，代码里不出现硬编码 `.cuda()`。
   本机无独显（CPU 起步），云端租 GPU 后 `--device cuda` 即可，不改代码。
2. **种子固定**：`set_seed()` 统一设置 python / numpy / torch，保证曲线可复现。
3. **配置外置**：超参走 `configs/*.yaml`，不写死在脚本里，便于扫参。
4. **结果落盘**：所有图与指标写 `outputs/`，README 里引用的图都从它来。
5. **样式随代码走**：中文字体由 `wmlab.utils.plot.apply_style()` 显式设置（探测
   YaHei / PingFang / Noto CJK 等并逐个回退），**不依赖 `MPLCONFIGDIR` 等进程环境**——
   换机器、换终端、上云，图里中文都不会变方块；负号渲染同步修复。
6. **依赖锁定**：`requirements.lock.txt` 记录本机全量依赖（torch 为 +cpu 版，
   上云时替换为对应 CUDA 版本即可）。

## 结果（CPU，全部可复现；同种子重跑逐位一致）

### ① 随机策略 baseline

![random baseline](assets/01_random_baseline.png)

CartPole-v1，600 集随机策略：平均回报 **22.9**（std 11.9），最长 69 步，共 13733 步。
随机策略平均 20 步出头就倒 —— 这也是当下 `horizons` 上限取 15 的原因。

### ② 世界模型与可靠视界（第一轮：CartPole，短视界）

![world model](assets/02_world_model.png)

| 指标 | 值 |
|---|---|
| 环境 / 数据 | CartPole-v1，600 集随机策略，13,733 步 |
| 参数量 | 67,652 |
| train / val loss | 0.1502 → 0.0005 / 0.0013 |
| **可靠视界（开环）** | **测不出**（视界上限 15 步内 NMSE 最高 0.040 < 0.05） |
| **可靠视界（闭环）** | **10.1 步**（NMSE 首次超过 0.05，线性插值） |

**注意：这一轮的结论后来被修正了**（见下一节）。当时由图④ 得到「误差在自我放大」，但有两个问题：

- **15 是数据上限，不是模型上限** —— 随机策略在 CartPole 上平均只活 22.9 步，
  `horizons` 物理上就设不到更大，所以「开环 > 15」什么也没测到；
- **图④ 的口径有缺陷** —— 当时写的是 `inc = np.diff(nmse)`，而 `horizons` 间隔不等
  （…2, 2, 3, 5, 5, 10…），**直接差分会把「采样间隔变大」混进「误差增长加速」**。

### ③ ★ 核心结果：Pendulum 长视界（2026-09-18）

![long horizon](assets/02_world_model_pendulum.png)

换到 `Pendulum-v1`（每集固定 200 步）把视界抬到 **190**。
**模型 / 损失 / 优化器 / 阈值全部不动，只换数据供给** —— 这是控制变量，不是调参：

| 指标 | 值 |
|---|---|
| 环境 / 数据 | Pendulum-v1，300 集随机策略，60,000 步（每集 200.0 步） |
| 参数量 | 67,267（obs_dim=3，连续动作） |
| train / val loss | 0.7298 → 0.0007 / 0.0005 |
| **可靠视界（开环）** | **64.8 步** |
| **可靠视界（闭环）** | **52.0 步** |
| 阈值 | NMSE > 0.05（`rel` 口径，与上一轮同口径，未改） |

**三条结论**（每步增量表可用 `scripts/03_analyze_horizon_curve.py` 复现）：

1. **开环 H\* 第一次是个有限数字：64.8 步。** 上一轮那个「> 15」是区间下界，不是测出来的值。

2. **误差增长是「饱和」，不是「自我放大」。** 每步增量在 h≈88 处到达峰值
   （开环 9.4e-3、闭环 3.5e-3 每步），之后**掉头**：开环回撤 17%、闭环回撤 63%。
   上一轮之所以看到「自我放大」，是因为 CartPole 只测到 15 步，**整段都落在饱和曲线的上升段**
   —— 用短视界数据外推长视界行为，得到的是错觉。正确描述是：
   **每步误差增长率先升后降，最终饱和到目标方差（NMSE → 1，即「完全不可预测」的上界）**。

3. **闭环误差在超长视界反而低于开环**（h=190：闭环 NMSE 0.76 vs 开环 1.03）。
   开环 NMSE > 1 意味着**比「直接预测均值」还差**；闭环则像收敛到了动力学的吸引子附近，误差不再扩大。
   **这是最反直觉的一条，明确标注为待验证**（候选解释：闭环不再被真实动作序列牵引，误差有上界；
   需用多种子 / 多阈值复核 —— 对应课题里的 T4 消融）。

**怎么读这张图**：子图② 是对数坐标的 NMSE 曲线，两条都呈 S 形 ——
h < 50 稳步上升、h ≈ 50–90 之间急剧恶化、h > 100 之后趋平。
子图③ 是一条真实轨迹的 190 步开环预测：前 30 步基本贴合，之后相位逐渐偏移。
子图④ 已按步数归一（`ΔNMSE / Δhorizon`，柱宽 ∝ 区间长度），能直接看出峰值位置。

**诚实的边界**：以上都是**随机策略**数据、**CPU**、**单种子（seed=0）**、MLP 世界模型。
**跨环境数字不可比** —— Pendulum 与 CartPole 的 H\* 不能直接比较（状态维度、动作空间、
每集长度都不同），这里只作「各测一次」。也**尚未回答**「该不该预测」这个科研问题（那是消融实验的事）。

## 当前状态

- [x] 环境接口 + CartPole 适配器
- [x] 最小世界模型（MLP 版）
- [x] 多步 rollout 误差曲线 + 可靠视界指标
- [x] 长视界 H\* 测量（Pendulum，视界到 190）＋ 误差形态判定脚本（`scripts/03_`）
- [ ] 换场景：UAV 轨迹 / 无线信道（下一步）
- [ ] 因子化潜空间（确定性 H_t + 随机 Z_t）与不确定性校准
- [ ] 扩散策略动作头（对齐 diffusion policy）

## 参考

- 课题地图与选题：见父项目 `D:\workbuddy\researchproject`
- 方法来源：Dreamer 系（潜空间想象训练）、JEPA 系（联合嵌入预测）
- 误差增长形态的判定口径：`scripts/03_analyze_horizon_curve.py`
  （`inc = ΔNMSE/Δhorizon`；峰值在末段 = 未饱和，峰值后回撤 ≥ 15% = 饱和）
