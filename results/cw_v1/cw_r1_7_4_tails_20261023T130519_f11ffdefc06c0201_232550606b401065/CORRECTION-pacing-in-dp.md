# CORRECTION · 关于 `pacing_in_dp.json` 的静态结论（错）→ 追加更正，不删旧件

- 时间：2026-10-23 13:1x（`date` 实测 `+0800`，Asia/Shanghai）
- 性质：**本轮的自我证伪**（第 61 条）。`pacing_in_dp.json` **保留原样不删**，
  本文件是追加的更正，指向同一份证据。

## 1 我写了什么（错的）

`pacing_in_dp.json` 记：

```json
{"pacing_read_in_dp_branch": false,
 "evidence": "plan_ref.py::_terminal —— if self.tail == \"dp\": task_P = self._dp_value(...)
              else: _paced_sumP(...)；dp 支路不引用 self.pacing",
 "consequence": "G8b（pacing=False）的阴性对照不覆盖 dp 臂"}
```

## 2 实测当场推翻它

主跑同一份 `tail_arms.json` 里，**dp 臂的 `G8b` 行**：

| 臂 | `cv`（`pacing=True`） | `cv_nopace`（`pacing=False`） |
|---|---|---|
| `paced` | 0.5834 | **0.9286** |
| `dp_legacy` | 0.2142 | **0.9417** |
| `dp_r173` | 0.2577 | **0.8677** |

`pacing` 对 dp 臂的 CV 影响是 **0.21 → 0.94**（4.4×），不可能是"死参数"。
若它真是死参数，dp 臂两列必须**逐位相同**。

## 3 错在哪（根因，读代码）

`pacing` 的入口**不在** `_terminal`，而在它的**调用者** `_expect_after`：

```
plan_ref.py:441   if not self.pacing:
plan_ref.py:442       bud_after = np.zeros_like(bud_after)      # ← 强制"尾部再无预算"
plan_ref.py:462   return self._terminal(P2, ..., bud=bud_after)  # ← 把 0 传下去
plan_ref.py:411       task_P = self._dp_value(P, bud, steps)      # ← dp 支路消费 bud
```

⇒ 我读了 `_terminal`（被调者）就下结论，**没读它的调用者 `_expect_after`**。
`pacing` 通过 `bud` 这个**间接通道**进入了 dp 支路。这是"读实现只读了半条链"。

## 4 后果与处置

1. **预注册 §2 T-C2 的立论前提不成立**：`G8b` 的对照**确实覆盖** dp 臂。
   ⇒ `dp_b_cap=8` 这条对照变成**冗余**（不是必需），但**保留**：
   它独立证明了"预算轴被截断也能被 `G8a` 抓到"，是另一条信息（且实测 PASS）。
2. **判词不受影响**：`T_C2_bcap8_fails_G8a` 实测 PASS（`cv=0.8839 > 0.30`，
   profile `[31,31,32,26,3,4,1,0]`、`aoi=66.0` —— 预算前 4 窗花光的极端形态），
   `T_C_nopace_fails_G8a` 也 PASS（`cv_nopace=0.9286`）。`NO-POWER` 不成立。
3. **反过来，这加强了一条结论**：dp 臂的配速改善**确实**经由"预算感知"（`bud`）实现 ——
   因为把它掐掉（`pacing=False`）CV 立刻从 0.21 崩回 0.94。
   这比"值函数恰好算对了"更有解释力。

## 5 记入自证伪台账（第 61 条）

> **只读被调函数，不读调用链** —— 判"某参数在支路上是否生效"时，
> 必须 grep 该参数的**全部**引用点（`self.pacing` 共 3 处：161 赋值 / 193 名字 / 441 判据），
> 而不是看主消费函数里有没有出现。
> 可迁移形式：**"某字段没在这段代码里出现" ≠ "这段代码不受它影响"**（可能是上游改了入参）。
