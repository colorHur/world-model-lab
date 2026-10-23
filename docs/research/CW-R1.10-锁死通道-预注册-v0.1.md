# CW-R1.10 `dp` 的锁死通道 —— 预注册 v0.1

> **写卡时间**：2026-10-24 05:34（`date` 实测，与系统时钟同为 GMT+8）
> **写在**本轮任何新计算之前（本文件落盘时，本轮 `run_episode` 调用次数 = **0**）。
> 证据目录：`results/cw_v1/cw_r1_10_lockin_<UTCstamp>_<hash>/`（新建，不覆盖旧目录）。
> 被测物：`plan_ref.py` / `plan_dp.py` / `world.py` / `runner.py` / `config.py` **零改动**（本轮只读它们）。
> 上位状态：`06-进度管理/自动推进/STATE.md` §7.2 第 1、2 条（同一轮内做，第 3 条**不在本轮**）。

---

## §0 写卡前**已经看见**的东西（如实披露，防止事后倒推）

| # | 已见事实 | 来源 |
|---|---|---|
| 1 | `ep100/seed2001/ack=on`：`n_attempted=[115,13]`、`n_delivered=[44,7]`、`n_probe=0`、`n_hold=128`、`aoi_tavg=[6.23,60.94]` | R1.9b `t6_trace.json` |
| 2 | 同格 `P_hat` 差 `median=−0.0446`、`max=0.0`、`frac(P0>P1)=0.0` ⇒ **P0 从不高于 P1** | 同上 |
| 3 | 同格 `|V(SEND1)−V(SEND0)|` 中位 **0.0218** ≫ `_TIE_EPS=1e-12` ⇒ **并列键不是原因** | 同上 |
| 4 | 同格 `rollout_pref = {u0:242, u1:13}` ⇒ 在 256 槽里，根节点/贪心**几乎每一步**都更想送 u0 | 同上 |
| 5 | 交换 `sigma_w` 两用户后饿死的仍是 u1（`[115,13]`→`[121,7]`） | R1.8b `LABEL-DRIVEN` |
| 6 | 对称 plant 下值表 `V(P0,P1) ≡ V(P1,P0)`，最大相对差 1.2e-15 | R1.8d `TABLE-SYMMETRIC` |
| 7 | 状态→动作映射**逐 bit 等变**（`max_rel` 0.0 / 1.19e-16；40 seed × 4 格 `Δ_swap==−Δ` 160/160） | R1.9 `T1`/`T8` |
| 8 | `_wire_dp` 把 `p̂` **冻结在 episode 起点**（`m_h=0,v_h=1 ⇒ p̂=0.5`）⇒ DP 值表本身**不吃** ACK | 读 `plan_ref.py:392-407` |
| 9 | ACK 进入比较的**唯一入口**是 `_expect_after` 的 SEND 分支：`p = self._p_hat(st.mh, st.vh)[:, i]` | 读 `plan_ref.py:468` |
| 10 | `use_ack_for_csi` 是**联合开关**：`world._ack_bump`（世界信念）＋ `plan_ref._advance` 的 ACK 更新（规划器 rollout）**同时**受它控制 | 读 `world.py:461-467,542`、`plan_ref.py:554-558` |
| 11 | `_advance` 的 SEND 分支**不推进未送用户**（与 `world.step` 不一致，但对称） | R1.9 T4 / `plan_ref.py:542-559` |
| 12 | `_pick` 并列键 `(HOLD 优先, SEND 优先, user 升序)`，容差 1e-12 | 读 `plan_ref.py:498-516` |

**没看见过的东西（本轮才产生）**：`h_mean/h_var`（即 `p̂`）在一条真实 episode 里的**逐槽轨迹**；
关掉 `use_ack_for_csi` 后送达失衡**是否消失**；同一 episode 换 8 个策略 seed 后方向**是否一致**
（"系统性"还是"初始硬币"）。

---

## §1 问题

R1.9 已把"值表轴错配"和"策略层下标依赖"两条排掉（§0-6、§0-7）。剩下能解释 115:13 的只有两类：

- **H-feed（反馈锁定）**：服务 u0 ⇒ 它的 ACK 抬高 `p̂_0` ⇒ `_expect_after` 认为送 u0 的期望下降更大 ⇒
  更该送 u0 ⇒ 自我强化。**这是个正反馈，一旦偏离就不再回头**。
- **H-coin（初始硬币 + 状态惯性）**：首槽两用户完全对称，值差只来自蒙特卡洛噪声（`_Draws` 按用户列取
  不同随机数）⇒ 谁先被送由噪声决定；此后 `P` 的差异（§0-2）让"送同一个"在数值上继续占优。

两者**都**能产生 115:13，且都**不**违反 R1.9 的等变性结论（等变说的是映射，不是状态分布）。
本轮用两条单变量把它们分开：

> **A（切断信息）**：`use_ack_for_csi=False` 时，ACS 通道断掉（`p̂≡0.5`）。
> **B（换硬币）**：同一 episode 换 8 个策略 seed，看方向是 8/8 还是分裂。

---

## §2 判据（写死，跑完不改）

### 配置（与 R1.8/R1.9 面板同参数，不调参）

```
plant   = bounded_plant(sigma_w=(0.02, 0.03))      # 面板值，不对称且故意保留
budget  = floor(0.5 · 256) = 128                    # ratio=0.5
rho     = 0.98
policy  = PlanRef(use_probe=True, sample_hidden=True, H=5, M=16, tail="dp", dp_grid="r173")
grid_arms = use_ack_for_csi ∈ {True(默认), False}
episodes  = {100, 101}
seeds     = {2001, 2002, …, 2008}
```

共 `2 × 2 × 8 = 32` 个 episode。

### G0 接线闸（**零信任**：不假设开关接好了）

在 `ack=off` 的一条 episode 上逐槽检查：

```
max_t |h_mean_i,t| == 0   且   max_t |h_var_i,t − 1| == 0   对 i∈{0,1}
⇒ p̂_i,t ≡ 0.5 逐位（由 `_p_hat` 的 80 点求积在 (0,1) 上给出）
```

**必须有分辨力**：同一检查在 `ack=on` 上**必须 FAIL**（`h_mean` 离开 0）。
若 on 也不动 ⇒ 开关没接线 ⇒ **本轮全部数字作废**（不是"ACK 没用"）。

### G1 确定性闸

`(ep100, seed2001, ack=on)` 跑两次，要求 `n_attempted/n_delivered/cost_est/cost_true/
policy_trace_hash` **逐位相同**。若不同 ⇒ 存在隐藏全局状态 ⇒ 种子扫描（B）作废。

### G2 与入库证据对账

`(ep100, seed2001, ack=on)` 必须复现 §0-1：`n_attempted=[115,13]`、`n_delivered=[44,7]`、
`n_probe=0`。若不复现 ⇒ 本轮配置不是 R1.9b 那一格，**先查配置再谈结论**（不许换解释）。

### T1 主判量

逐 (episode, seed, arm) 记 `d_i = n_delivered_by_user[i]`（实际送达帧，R1.8 更正后的口径），
以及 `a_i = n_attempted_by_user[i]`。

- 符号量 `Δ = d_0 − d_1`（副口径 `Δ_a = a_0 − a_1`）。
- 逐 episode 统计 `k = #{seed: Δ > 0}`（偏 u0 的 seed 数）。

### T2 判词（预定义，四选一，逐 episode 给）

| 观察 | 判词 | 含义 |
|---|---|---|
| `ack=on` **8/8** 偏 u0 | `SYSTEMATIC-GIVEN-EPISODE` | 给定 episode 与 plant，方向不由策略 seed 决定 ⇒ "硬币"被否 |
| `ack=on` 分裂（`2 ≤ k ≤ 6`） | `COIN-FLIP` | 方向由首槽 MC 噪声决定 |
| `ack=off` **8/8** 偏 u0 | `LOCK-ACK-INDEPENDENT` | H-feed 被否 ⇒ 锁定不靠 ACK |
| `ack=off` 分裂 | `LOCK-ACK-FEEDBACK` | H-feed 成立（**且** ACK 打开时 8/8，否则只写"部分"） |

`0/8`（反向）出现时：先按实现缺陷排查（`_pick` 排序 / 面板列），**不**直接写"偏好 u1"。

### T3 机制读数（只记录，不判 PASS/FAIL）

- 逐槽 `p̂_i,t`（由 `h_mean/h_var` 经同一 `_p_hat` 算，与策略同口径）、`P_i,t`、动作；
- `root vals` 的 `V(SEND1)−V(SEND0)` 轨迹（仅 2 个 trace 格：`ep100/seed2001` 的 on 与 off）；
- 逐用户 `aoi_tavg / aoi_max / n_never_served`（R17：比值必须同报绝对量）。

---

## §3 统计口径（**先钉死，防止挑好看的**）

1. **有效独立单元 = episode 数（2）**，不是 seed 数。R1.7.8 `D5b` 已实测同 episode 的 4 个 seed
   角色判定 4/4 一致 ⇒ 把 seed 当样本量会高估显著性。
2. 因此：8 个 seed 只用来回答"**这一集里方向是不是确定的**"，**不**报"16/16 同向 ⇒ 显著"。
   若非要报二项 `p`，必须同时写"这是同一集的重复抽样，不是 16 个独立单元"。
3. 任何写"方向已定"的句子，前面必须有 `p` 或"有效单元数 = 2，不做显著性声明"。
4. 比值口径：`d_0/d_1` 只在 `d_1 > 0` 时给；`d_1 = 0` 记 `+inf` 并**同时**报绝对值。
`_TIE_EPS`、`0.05/0.20/1.5/0.95`、`_CACHE_MAX`、`grid="r173"` **一律不动**。

---

## §4 预算（先探针，后外推；#98 纪律）

- 已知：`r173` 单表构建 **21.0 s / 2048.8 MiB**；单 episode（`dp`）**24.0 s**（R1.9b 实测）。
- 估：32 格 × 24 s ≈ 12.8 min + 1 次 DP 构建 ≈ 13 min；加闸与序列化 ≤ **20 min**。
- **探针闸**：先跑 1 格量实测；若实测 > 45 s/格，则 seed 集缩到 `{2001…2004}`（16 格）
  并在交付里写明"因预算缩样"。

---

## §5 否定条件 / 作废条件

| 编号 | 条件 | 后果 |
|---|---|---|
| N1 | `ack=off` 时 `h_mean` 仍在动 | 开关未接线 ⇒ 本轮作废 |
| N2 | 同 config 两次跑结果不同 | 非确定 ⇒ 种子扫描作废，只报单格 |
| N3 | `ack=on` 的 `ep100/seed2001` 复现不出 `[115,13]/[44,7]` | 配置漂了 ⇒ 先查配置，不许换解释 |
| N4 | `ack=off` 下出现 `d_1 > d_0` 的 8/8 反向 | 先查实现缺陷 |

---

## §6 产物

1. `experiments/cw_v1/r1_10_lockin.py`（闸 G0–G2 + 32 格面板 + 2 个 trace 格 + 判词）
2. 证据目录（逐格 `cells.json`、`gates.json`、`meta.json` 含源码树哈希与未提交差异）
3. `docs/CW-R1.10-锁死通道交付.md`（含自伤清单）
4. 科研空间：`STATE.md` §7 重写 + `RUNLOG/2026-10-24.md` append

---

## §7 事后修订区（**空 = 本卡在跑数前已冻结**）

（若跑后必须改判据，逐条在此留痕并写明"当时已看见什么"；不得改写 §2 原文。）
