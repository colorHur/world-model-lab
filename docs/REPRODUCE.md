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
python tests/test_channel_and_aoi.py      # ★ 74 项回归自检（不依赖 pytest）
python scripts/30_scheduling_baselines.py --n 50000 --tmax 16   # 调度基线库 v0.1，无模型、秒级~分钟级
```

前两条跑通 ⇒ 你已经能复现本仓库的**全部离线口径**。
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

★ **缓存复用**：多数脚本会把训练好的模型存到 `outputs/_temp/*.pt`，重跑时自动载入
（要重训加 `--retrain`）。`outputs/` 已 gitignore ⇒ 缓存**不进版本库**，需要先自己跑一次产生。

★ **别把"本机自检通过"写成"独立复现完成"**：本机自检只证明**装好的依赖 + 仓库代码**能跑通；
   "独立复现"要求换机 / 换环境后**数字对得上**（本仓库的办法是同种子逐位一致 + 闭式对账）。

## 4. 复现判据（怎么才算"跑对了"，而不是"跑通了"）

| 判据 | 期望 | 出自 |
|---|---|---|
| 74 项自检 | `74/74 passed` | `tests/test_channel_and_aoi.py` |
| 环境自检 | 15 项全过，含中文字体链路 | `scripts/00_check_env.py` |
| 调度基线闭式对账 | 最差 `diff/(5·SE) ≤ 1`，99 个点 | `scripts/30_scheduling_baselines.py` |
| 调度基线退化 | `periodic(T=1)` / `threshold(K=0)` 的 `tx_rate−1 = 0` | 同上 |
| 调度基线已知答案 | PER=0 时 `threshold/periodic` 的 E[age] 比值 **≡ 1.0000** | 同上 |
| 同种子重跑 | 逐位一致（`outputs/` 里的 JSON 与本文档数字对得上） | 全仓库约定 |

★ **口径纪律**（本仓库最容易被违反的一条）：任何"误差"数字都必须同时写清
**逐点还是累积 / 在哪个评测集 / 除以哪个方差 / 曲线定义（开环/观测闭环/潜闭环）**；
H\* 还要加**种子数与采样量**。同一个模型上合法的 H\* 数值可以差 2 倍以上（见 `README.md` §设计约定）。
