# CW-R3.4 预注册：上界形态可修吗？—— 槽价的可识别性与判据缺口

- 协议：`CW-POMDP-v1.1`（**不改**）；研究基线 `mainline_config()`（`BOUNDED` + `ack_mode="exact"`）
- 写卡时间：2026-10-26 08:4x（`Get-Date` 实测 +08:00，见交付稿）；**写卡时 `run_episode=0`、未跑正式面板**
- 上游：`CW-R3.3`（`docs/CW-R3.3-槽与量子-交付.md`）§7.2 的 A 支（默认支）
- 层级：**零 episode、零训练、零参数更新**；只 `import` 被测物，一字节不改

## 1 问题

R3.3 把判据收成 `(b − c_probe)·Δp̂_UB > p̂`，并判它**只能单侧用**：
`UB < 代价` 可判"买不起"；`UB > 代价` 不能判"买得起"（`τ=6,b=5` 实测余量 `2.37×`/`5.64×` 而实测
`IV_paid` 逐位 0）。R3.3 给这个过估列了三条原因：①信念衰减 `ρ^k` ②2 用户翻面概率 `≪1` ③`snap` 量化。

本轮只问一件事：**上界的形态可修吗？** 即把 `(b−1)·Δp̂_UB` 换成一条更贴近真值的表达式后，
`τ=6,b=5` 的 `2.37×/5.64×` 过估能否压到 `±30%`。

## 2 三条候选形态（跑前写死，只这三条，不再加）

| 代号 | 形态 | 依据 |
|---|---|---|
| `F1` | `Σ_{k<b} ρ^k Δp̂_k`（R3.3 §7.2 A 支原文） | 读数对 `t+k` 时信念的贡献按 `ρ^k` 衰减 |
| `F2` | `Δp̂_UB · P_flip` | 信息只能通过**改变动作**兑现（翻面率） |
| `F3` | 判据右侧加**槽价**：`(b−c_probe)Δp̂ΔV_succ > c_probe·ΔV_q + π_slot`，`π_slot ≡ V(HOLD) − min_i V(SEND_i)` | 探测与发送**同吃一个槽** ⇒ 花槽去探测的机会成本 = 放弃的立即服务 |

## 3 基线与信息权限

- **不存在新基线**：本轮不新增策略臂，只对 R3.2/R3.3 已注册的三个臂（`noprobe` / `free` / `paid`）做**量的分解**。
- 信息权限：只用 `V()` 与 `first()`（动作值）与 `p_succ_expected`。**不读** `h_true` / `cost_true*` /
  `w_realized` / `x_true`（闸 `G1` 用 AST 钉住）。
- `first()` 的使用边界：它给的是"首步各动作的值"，是**公开量**（策略自己就能算），不是特权信息。

## 4 判据 / 效应 / 否定条件（跑前写死）

| 代号 | 判据 | 判词 | **否定条件（什么数据能让它 FAIL）** |
|---|---|---|---|
| `J0` | `A(b) ≡ Σ_{k<b} ρ^k Δp̂_k / ((b−1)Δp̂_0)` | `A-FORM-SUFFICIENT` iff `A(5) ≤ 1/2.37` 且 `A(6) ≤ 1/5.64`；否则 `A-FORM-INSUFFICIENT` | `A(5) > 0.422` 或 `A(6) > 0.177` |
| `J1` | 根状态首步动作值：`V(PROBE_i) − V(SEND_best)` | `PROBE-DOMINATED-BY-SEND` iff 该差 `> 0` 在 **全部**判据格成立；否则 `PROBE-COMPETITIVE-SOMEWHERE` | 存在一格 `PROBE < SEND_best − 1e-12` |
| `J2` | 槽价/量子价 `= π_slot / ΔV_q` | `SLOT-DOMINATES-QUANTUM` iff 中位比值 `> 2.0`；`SLOT-BELOW-QUANTUM` iff `< 0.5`；否则 `SLOT-COMPARABLE` | 比值落在 `[0.5, 2.0]` |
| `J3` | 修好的判据（`F3`）在**已入库**面板格上的方向 | `CRITERION-REPAIRED` iff 在 `IV_free/IV_paid` 实测 `≤ IV_NOISE` 的格上判"买不起"的比例 `≥ 0.90` **且** 在实测 `> IV_NOISE` 的 3 格上全部判"买得起"；否则 `CRITERION-STILL-MISCALIBRATED` | 任一条件不满足 |
| `J4` | 活体对照：把 `σ_w[1]` 从 `0.03` 放到 `0.30`（**仅诊断，不改主线**） | `SYMMETRY-BREAK-RESTORES-PROBE` iff 该 plant 上出现 `PROBE < SEND_best − 1e-12` | 破坏后仍全被支配 ⇒ `J1` 的机制解释不成立 |

## 5 预算（跑前写死，超了就停并如实写）

- 墙钟 `≤ 600 s`；引擎 `n_m ∈ {65, 129}`，`τ ∈ {3,4,5,6}`；`P ∈ {0.005, 0.02}`，`b ≤ 6`
- 峰值内存 `≤ 4 GiB`；**不计云资源**（全程本机 CPU）
- 若 `STATE_CAP=6e6` 被撞 ⇒ 该格记 `NaN` 并计入 `n_cells_dropped`，**不许当 0**

## 6 产物

- 判定器：`experiments/cw_v1/r3_4_slotprice.py`（`--ctrl-only` 只跑自证）
- 证据：`results/cw_v1/cw_r3_4_slotprice_<UTC>_<hash>/r34.json` + `commands.txt` + `_POST_COMMIT.json`
- 交付：`docs/CW-R3.4-槽价与判据缺口-交付.md`（含自伤条数）
- 闸：`G1`（AST 无真值）`G2`（`Δp̂_k` 对 GH 点数收敛）`G3`（槽价两条独立算法一致）`G4`（单调性）
  `G5`（变异：`ρ→0` 必须让 `J0` 翻成 `SUFFICIENT`）`G6`（活体阴性对照：`J4`）`G7`（被测物身份）
  `G8`（鞅恒等：`E_y[p̂(post)] = p̂(prior)`，数值腿）

## 7 跑后修订（预注册允许的写法：只追加，不改上面的判据）

（跑完填）

## 8 不做

- 不重跑 P3；不改 `plan_*`/`world`/`runner`/`config`/`policies`/`r3_*`；不训练；不加噪声/改 `c_probe` 追 PASS
- 不把 `τ≤6` 外推到 256 槽；不引用 `run-id` 时间戳当本地时间
- 不因为本轮结论就写"信道信息无用"：本轮只做**代价侧的会计更正**，信息收益侧仍是上限形态
