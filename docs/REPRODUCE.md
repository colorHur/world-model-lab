# REPRODUCE —— 怎么在本机 / 云上把 `world-model-lab` 跑起来

> 本文只讲**怎么跑**与**跑了会看到什么**。结论与引用限定见 [`EXPERIMENTS.md`](EXPERIMENTS.md)，
> 首页速览见 [`../README.md`](../README.md)。
>
> ★ 一句话：**仓库刻意不依赖 pytest**（多一个测试依赖就多一道让人放弃的门槛），
> 全部自检用标准库 + numpy + torch 直接跑。

---

## 0. 最短路径（本机 CPU，约 2 分钟）

```bash
python scripts/00_check_env.py            # 环境自检（含中文字体链路），15 项
python tests/test_channel_and_aoi.py      # ★ 86 项回归自检（不依赖 pytest）
python scripts/30_scheduling_baselines.py --n 50000 --tmax 16   # 调度基线库 v0.1，无模型、秒级~分钟级
```

★ **前两条跑通 ≠ 复现了离线口径**。它们只证明两件事：**依赖装对了**、**仓库代码在当前
环境能跑通**（15 项 + 全部自检）。而"复现"要求换机 / 换环境后**数字对得上**（§4 的判据表）。
自检用的是**构造的输入**（手算已知答案、闭式对账、逐位一致性），它**不**覆盖"多步误差曲线
在别的机器上给出同一个 H\*"这类**跨机数值一致性**。
★ 截至 2026-10-15：**干净检出 + 全新环境的同机复现已完成**（`docs/REPRODUCE.md` §5，含逐位一致的核心结果）；
**换机（外部）独立验收仍待做**（§6）—— 这是本仓库的已知缺口，不写成已完成。

第三条不需要任何训练/GPU，会产出 `outputs/30_scheduling_baselines.{png,json}`
（图已进版本库：`assets/33_scheduling_baselines.png`）。

---

## 1. 依赖

`requirements.txt`（本机 CPU 起步）：

```
torch>=2.2      # ★ 装 CPU 版：pip install torch --index-url https://download.pytorch.org/whl/cpu
numpy>=1.26
matplotlib>=3.8
pyyaml>=6.0
tqdm>=4.66
gymnasium>=0.29
```

`requirements.lock.txt` 是**最小可复现锁** —— 27 个包（6 个直接依赖 + 21 个传递依赖），
由干净隔离 venv（Python 3.11.16 / torch 2.14.1+cpu）实测 `pip freeze` 导出；
**不是**从本机 conda 大环境 freeze 的全量清单（那种会带进 jupyter / diffusers / stable-baselines3 /
tensorboard 等无关包，体积上百，会误导"别人能运行"）。它含 torch 的 `+cpu` 后缀，
安装时须**追加** PyTorch CPU 索引（用 `--extra-index-url`，不是 `--index-url` ——
后者会把索引替换成纯 torch 索引，干净机器上找不到 numpy 直接失败）：

```bash
venv\Scripts\python.exe -m pip install -r requirements.lock.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

上云换 CUDA 版时只替换 torch 那一行、并把 extra-index-url 换成 cu121 对应的即可，**代码不动**
（设备由 `wmlab.utils.device.get_device()` 决定）。

## 2. 干净环境（隔离 venv）的建法与两个真实坑

```bash
# ★ 用与本机一致的基线解释器建 venv（此处为本机 ai-lab 的 Python 3.11）
D:\devtools\miniconda3\envs\ai-lab\python.exe -m venv D:\repro\wmlab-clean\venv3

# 1) torch 走官方 CPU 索引（PYPI 上的默认 wheel 是 CUDA 版，体积大 8 倍）
venv3\Scripts\python.exe -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
# 2) 其余走就近镜像
venv3\Scripts\python.exe -m pip install --no-cache-dir numpy matplotlib pyyaml tqdm gymnasium -i https://mirrors.aliyun.com/pypi/simple/
```

★ **两个坑（都真实踩过，写在这里省你一次调试）**：

1. **不要往已有 venv 里 `--force-reinstall`**。本机环境的 Python 层注入了一个
   **批量删除守卫**（`SAFE_DELETE_BULK_CONFIRM_REQUIRED`，阈值 50 次/轮）：
   pip 装 wheel 时对**已存在的目标文件**做 `os.unlink` 重写，超过阈值会被直接
   `SystemExit(1)` 杀掉 —— 症状是"下载完成、卡在 `Installing collected packages`、然后失败"，
   与网络无关。`--force-reinstall` 还会连带卸掉 venv 自己的 pip。
   ⇒ **要干净重装就建新 venv**（目标文件全不存在 ⇒ 零 unlink ⇒ 不触发守卫）。
2. **首次装 torch 只慢在解包**（数千个文件），不是卡住。观察方式：
   看 `venv3/Lib/site-packages/torch` 下的**文件数是否在增长**，
   不要靠"日志没动"判断（pip 装包期间**不打印进度**）。

## 3. 需要模型的实验（CPU 可跑，但要么等训练、要么用缓存）

| 脚本 | 内容 | 形态 |
|---|---|---|
| `scripts/02_train_world_model.py` | 训练一个世界模型，产出 loss / 多步误差曲线 | 需训练 |
| `scripts/04_channel_tracking.py` | X24 等价性 + X25 调度帕累托 + X26 解析对账 | 约 3 min，`--quick` 冒烟 |
| `scripts/06_delay_floor.py` | X27 时延地板（读 04 的 checkpoint，**不重训**） | 约 1–2 min |
| `scripts/07_seeds_threshold.py` | X3/X4 种子 × 阈值（含 5 次完整重训） | 约 13 min |
| `scripts/16_task_horizon.py` | X30 任务锚定的 H\*（UAV 闭环） | 约 2 min |
| `scripts/24_triggered_scheduling.py` | X38–X43 调度族（`--pol threshold/utrig/conformal/oracle`） | 需训练（有缓存则复用） |
| `scripts/28_x44_timing.py` | X44 发送时机扰动 | 需训练（有缓存则复用） |
| `scripts/29_x45_boundary_attribution.py` | X45 相位边界归因（读 X44 产物） | 不重训 |
| `scripts/31_x9_imagination.py` | X9 想象训练（Pendulum 连续动作；世界模型 + 三档视界 + 真实 PPO 对照） | 约 6 min |
| `scripts/32_reference_learner_audit.py` | 参照学习器审计（3 种子 × 800 轮 = 1,228,800 步） | 约 10 min CPU |

★ **缓存复用**：多数脚本会把训练好的模型存到 `outputs/_temp/*.pt`，重跑时自动载入
（要重训加 `--retrain`）。`outputs/` 已 gitignore ⇒ 缓存**不进版本库**，需要先自己跑一次产生。

★ **别把"本机自检通过"写成"独立复现完成"**：本机自检只证明**装好的依赖 + 仓库代码**能跑通；
   "独立复现"要求换机 / 换环境后**数字对得上**（本仓库的办法是同种子逐位一致 + 闭式对账）。

## 4. 复现判据（怎么才算"跑对了"，而不是"跑通了"）

| 判据 | 期望 | 出自 |
|---|---|---|
| 自检（当前） | 全部通过（`nn/nn passed`；**当前 86 项**） | `tests/test_channel_and_aoi.py` |
| 环境自检 | 15 项全过，含中文字体链路 | `scripts/00_check_env.py` |
| 调度基线闭式对账 | 最差 `diff/(5·SE) ≤ 1`，99 个点 | `scripts/30_scheduling_baselines.py` |
| 调度基线退化 | `periodic(T=1)` / `threshold(K=0)` 的 `tx_rate−1 = 0` | 同上 |
| 调度基线已知答案 | PER=0 时 `threshold/periodic` 的 E[age] 比值 **≡ 1.0000** | 同上 |
| ★ 世界模型 H\*（同种子逐位可复现） | 开环 **12.945840516581065** / 闭环 **6.955816701033772**；参数量 **67,652**（`default.yaml`，seed 0，CartPole-v1） | `scripts/02_train_world_model.py`（容差 **0**） |
| 同种子重跑 | 逐位一致（`outputs/` 里的 JSON 与本文档数字对得上） | 全仓库约定 |
| 参照学习器审计（可选，**约 15 min CPU**） | 判定规则可复跑（单一真源）：**每个**种子末 5 次均值 ≥ 随机基线 + 1.5 ⇒ VALID；否则 VOID。随机基线与 PPO **同初值种子**（配对），且**逐回合种子序列按位相同**（`audit.same_seed_sequence`） | `scripts/32_reference_learner_audit.py` |
| 训练随机源受控 | 同一训练种子**结果不随种子列表顺序改变**（自检 **[65]** + 实测 `ORDER_INDEPENDENT: True`） | 同上 |

★ **口径纪律**（本仓库最容易被违反的一条）：任何"误差"数字都必须同时写清
**逐点还是累积 / 在哪个评测集 / 除以哪个方差 / 曲线定义（开环/观测闭环/潜闭环）**；
H\* 还要加**种子数与采样量**。同一个模型上合法的 H\* 数值可以差 2 倍以上（见 `README.md` §设计约定）。

---

## 5. 复现记录：干净检出 + 全新环境（2026-10-15）

> ★ **这一步回答**：项目核心结果**到底依赖哪些本机状态**。
> ★ **这是"干净复现"，不是"换机复现"** —— **同一台工位机**，但**重新克隆 + 全新 venv + 无旧缓存**。
> 真正跨机的独立验收见 §6（**待做**）。

**环境**

| 项 | 值 |
|---|---|
| 检出 | `git clone` → `D:\repro\wmlab-clean\repo-r1`，**HEAD = `92b843c`**，工作树**无 `outputs/`**（无旧缓存） |
| 解释器 | **全新** venv（基线 = 本机 `ai-lab` 的 Python **3.11.16**） |
| 依赖 | `pip install -r requirements.lock.txt --extra-index-url https://download.pytorch.org/whl/cpu -i <镜像>` ⇒ **28 包**（27 锁 + pip） |
| 版本 | torch **2.14.1+cpu**｜numpy **2.4.6**｜matplotlib **3.11.2**｜gymnasium **1.4.0** |
| 装耗时 | 约 **3–4 min**（torch 124 MB 命中本机 pip HTTP 缓存；**纯网络下会更久**） |

**四项按时序（全部 exit=0）**

| # | 命令 | 耗时 | 实测 | 预期 / 容差 |
|---|---|---|---|---|
| ① | `python scripts/00_check_env.py` | **2.3 s** | 15 项全过 | 15 项全过（exact） |
| ② | `python tests/test_channel_and_aoi.py` | **21.1 s** | **84/84**（`92b843c` 的项数；含本轮新增后为 86） | 全过（exact） |
| ③ | `python scripts/02_train_world_model.py` | **42.5 s**（训练本身 **39.1 s**） | H\* 开环 **12.945840516581065** / 闭环 **6.955816701033772**；参数量 **67,652**；final loss train **4.5944e-4** / val **1.2622e-3** | ★ **与历史逐位一致**（容差 **0**） |
| ④ | `python scripts/30_scheduling_baselines.py --n 50000 --tmax 16` | **12.7 s** | 闭式对账最差比值 **0.796**（≤1，99 点）；退化 `tx_rate−1 = 0`；`p=0` 比值中位 **1.0000** | ≤1 / `=0` / `=1.0000`（脚本内**自检即判**，不过则 `raise`） |

★ **③ 最有信息量**：世界模型训练在**同种子下逐位可复现**（H\*、参数量、loss 全一致）
⇒ 该核心结果**不依赖本机的缓存或环境残留**（至少在同一 CPU 平台 / 同 torch 版本下）。

★ **④ 的比值随 PER 单调**（p=0 ⇒ **1.0000**；p=0.3 ⇒ **0.766**；p=0.7 ⇒ **0.621**）——
与 X38「阈值随丢包率越优」一致，属**独立复现**而非新结论。

★ **确实依赖本机状态的部分（如实）**：
① `outputs/` 缓存（本复现**刻意清空** ⇒ 完整重训，训练 39 s）；
② 本机已有的 **pip HTTP 缓存**（torch wheel 命中缓存 ⇒ 装得快，纯网络会更慢）；
③ **中文字体**（`00_check_env.py` 专门检查；缺字体只影响图的观感，不影响数值）。

★ **本记录的边界**：同机、CPU、同一 torch 版本。**换机 / 换 torch 大版本**见 §6（待做）。

---

## 6. 外部验收：给同学的最短复现说明（**待执行**）

> 这是 `AGENT-BRIEF` §2.4「下一步」第 4 项。目标是**别人在一台全新机器上能跑出 §5 的数字**。
> ★ 复制下面整段给对方即可；**其独立执行结果与卡点**要记回来。

```text
仓库：github.com/colorHur/world-model-lab
前置：一台装了 Python 3.11 的机器（CPU 即可，不需要 GPU）

1) 克隆 + 建全新 venv
   git clone https://github.com/colorHur/world-model-lab && cd world-model-lab
   python -m venv venv-clean
   # Windows: venv-clean\Scripts\python.exe ；Linux/macOS: venv-clean/bin/python

2) 装依赖（torch 的 +cpu 在官方 CPU 索引，必须额外加索引；其余走就近镜像）
   <venv>/python -m pip install -r requirements.lock.txt \
       --extra-index-url https://download.pytorch.org/whl/cpu

3) 跑四项（期望全部 exit=0）
   <venv>/python scripts/00_check_env.py                          # 15 项全过
   <venv>/python tests/test_channel_and_aoi.py                    # nn/nn passed
   <venv>/python scripts/02_train_world_model.py                  # 见下表"H*"
   <venv>/python scripts/30_scheduling_baselines.py --n 50000 --tmax 16

4) 把下面这张表的"你实测"列填回来（命令 stdout 里就有）
   ┌─────────────────────────┬───────────────────────────┬──────────┐
   │ 项                      │ 预期                       │ 你实测   │
   ├─────────────────────────┼───────────────────────────┼──────────┤
   │ 00_check_env            │ 15 项全过                  │          │
   │ tests/*                 │ nn/nn passed               │          │
   │ 02 H* 开环              │ 12.945840516581065 (±1e-6) │          │
   │ 02 H* 闭环              │ 6.955816701033772  (±1e-6) │          │
   │ 02 参数量               │ 67,652                     │          │
   │ 02 final loss train     │ 4.5944e-4          (±1e-8) │          │
   │ 30 闭式对账最差比值     │ ≤ 1.0                      │          │
   │ 30 p=0 比值中位         │ 1.0000             (±1e-9) │          │
   └─────────────────────────┴───────────────────────────┴──────────┘

记录：OS / Python / torch 版本、每步耗时、任何安装或运行报错。
```

★ **为什么 H\* 用严格容差（±1e-6）而调度比值宽松**：训练路径是**同种子确定性**的
（§5 已证明逐位一致）；而调度**空跑**用的是蒙特卡洛，其散度由脚本内的 **5×批量均值标准误**
判据吸收（`--n` 越大越紧），所以对外只要求"最差比值 ≤ 1"。
