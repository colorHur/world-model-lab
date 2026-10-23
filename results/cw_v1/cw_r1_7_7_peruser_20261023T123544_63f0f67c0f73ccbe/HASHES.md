# 哈希对账（CW-R1.7.7）

**为什么有这个文件**：本轮 `meta.json` 里 `inputs.*.sha256_16` 记的是**工作树文件**的哈希。
本仓库 `core.autocrlf=input`，仓库内 blob 是 **LF**、本机工作树是 **CRLF**，
于是**干净克隆里的同一个文件哈希不同** —— 用 `meta.json` 的哈希在别的机器上校验会**对不上**。
本文件把两套哈希都记下来，并给出可核的取法。

判定时点：2026-10-23 20:5x（`date` 实测，+08:00），提交 `9a218ee` 之后。

## 1. 入口脚本（LF，两套哈希相同）

| 文件 | worktree | repo blob |
|---|---|---|
| `experiments/cw_v1/r1_7_7_peruser.py` | `63f0f67c0f73ccbe` | `63f0f67c0f73ccbe` |

⇒ `meta.src_hash = 63f0f67c0f73ccbe` **可直接用 blob 复核**（该脚本在 `9a218ee` 中）：
```
git show 9a218ee:experiments/cw_v1/r1_7_7_peruser.py | sha256sum   # 取前 16 位
```

## 2. 输入（CRLF 工作树 vs LF blob，**两套不同**）

| 输入 | worktree（`meta.json` 记的） | repo blob（干净克隆可复现） |
|---|---|---|
| `…/cw_r1_7_6_g4form_20261023T182640_…/cells.json` | `6afc2f616388c8ed` | `77e0039112d3bf6f` |
| `…/cells` 同目录 `acts.json` | `dd05f708d0903dcc` | `c3e8a094c4e9019f` |
| `…/cells` 同目录 `stats.json` | `6a2d90701342323f` | `69329512dccb8661` |

核实取法（`<rev>` = 任何包含这些文件的提交，本批在 `13e58cb` 即有）：
```
git show <rev>:results/cw_v1/<证据目录>/cells.json | sha256sum
```

## 3. 影响与处置

- **不影响本轮结论**：本轮的 7 门与逐用户数字全部由**本机工作树**读入算出，`meta` 记录的正是实际读入的那份字节。
- **影响跨机复核**：别人克隆后按 `meta.inputs` 校验会失败。⇒ 用 §2 的 blob 列复核。
- **下一轮入口必须双记**（`worktree_sha` + `blob_sha`，或直接 `git cat-file blob` 取 LF 字节后算哈希）。
  本轮**不改脚本**：改了 `src_hash` 就与已入库证据脱钩，得不偿失。

## 4. 状态

**这是本轮发现的第 3 条自伤**（前两条见交付文档 §8），性质是"记录口径"而非"计算错误"。
它不改变判词 `ONE-SIDED-AT-8`，但必须写进 STATE 的硬约定里。
